#!/usr/bin/env python3
"""File Designer.

This module provides the core FileDesigner functionality for building
repository skeletons from RPG structures.

Key components:
- FileDesigner: Main orchestrator for skeleton building
- Two-stage process: RawSkeleton + GroupSkeleton
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel, Field

from rpg.models import RPG, Node, NodeType, NodeMetaData
from .skeleton_models import RepoSkeleton
from .skeleton_prompts import (
    RAW_SKELETON_PROMPT,
    GROUP_SKELETON_PROMPT,
    build_component_summary,
    extract_features_from_subtree,
    extract_leaf_descriptions_from_subtree,
    format_feature_list
)

# Import common LLMClient with trajectory support
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import LLMClient
from common.utils import get_project_background_context

# Skeleton design resolves a language backend from the project target
# language so file extensions, package markers, and prompt directives
# live with the rest of per-language decoder behaviour. Python projects
# receive an empty prompt directive; non-Python projects get a compact
# language preamble before skeleton prompts are rendered.
from decoder_lang import (
    get_backend,
    resolve_decoder_language,
    with_language_directive,
)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_directory_structure(
    dir_assignments: Dict[str, str],
    required_components: List[str],
    backend: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Validate that all required components have directory assignments.

    Args:
        dir_assignments: Mapping of component_name -> directory_path
        required_components: List of component names that must be covered
        backend: Optional :class:`decoder_lang.LanguageBackend`. When
            supplied, each path segment is validated against the
            backend's :meth:`is_valid_module_identifier`. When
            ``None``, path segments must be valid Python identifiers.

    Returns:
        (is_valid, error_message)
    """
    errors = []
    assigned_components = set(dir_assignments.keys())
    required_set = set(required_components)

    # Check for missing components
    missing = required_set - assigned_components
    if missing:
        errors.append(f"Missing directory assignments for components: {sorted(missing)}")

    # Check for extra/unrecognized components
    extra = assigned_components - required_set
    if extra:
        errors.append(f"Unrecognized components in assignments: {sorted(extra)}")

    # Identifier validation falls back to Python rules when no backend
    # is supplied.
    if backend is None:
        def _is_valid_segment(seg: str) -> bool:
            return bool(seg) and seg.isidentifier()
        identifier_kind = "Python identifier"
    else:
        _is_valid_segment = backend.is_valid_module_identifier
        identifier_kind = f"{backend.display_name} identifier"

    for comp, dir_path in dir_assignments.items():
        if not dir_path or not dir_path.strip():
            errors.append(f"Component '{comp}' has empty directory path")
            continue
        # Each path segment used as a package name must be a valid
        # identifier for the target language.
        for segment in dir_path.replace("\\", "/").strip("/").split("/"):
            if segment and not _is_valid_segment(segment):
                errors.append(
                    f"Component '{comp}': directory segment '{segment}' is not a valid "
                    f"{identifier_kind} (avoid hyphens; use underscores instead)"
                )

    if errors:
        return False, "\n".join(errors)
    return True, "All components have valid directory assignments."


def validate_file_path_constraint(
    file_path: str,
    allowed_dirs: List[str]
) -> Tuple[bool, str]:
    """Validate that a file path is under one of the allowed directories.
    
    Args:
        file_path: The file path to validate
        allowed_dirs: List of allowed directory prefixes
        
    Returns:
        (is_valid, error_message)
    """
    if not file_path:
        return False, "Empty file path"
    
    # Normalize paths
    normalized_path = file_path.replace("\\", "/").strip("/")
    
    for allowed_dir in allowed_dirs:
        normalized_dir = allowed_dir.replace("\\", "/").strip("/")
        # Check if file_path starts with allowed_dir
        if normalized_path.startswith(normalized_dir + "/") or normalized_path == normalized_dir:
            return True, ""
    
    return False, f"File path '{file_path}' is not under any allowed directory: {allowed_dirs}"


# ============================================================================
# Data Models for Structured Output
# ============================================================================

class DirectoryAssignment(BaseModel):
    """Assignment of a component to a directory."""
    component_name: str = Field(description="Name of the component")
    directory_path: str = Field(description="Directory path (e.g., 'src/parser')")
    reasoning: str = Field(description="Brief explanation for this assignment")


class DirectoryStructureOutput(BaseModel):
    """Output for directory structure generation."""
    assignments: List[DirectoryAssignment] = Field(
        description="List of component-to-directory assignments"
    )
    overall_reasoning: str = Field(
        description="Overall rationale for the directory structure"
    )


class FileAssignment(BaseModel):
    """Assignment of features to a file."""
    file_path: str = Field(description="Full file path (e.g., 'src/parser/tokenizer.py')")
    features: List[str] = Field(description="List of feature paths assigned to this file")
    purpose: str = Field(description="Brief description of the file's purpose")


class FileAssignmentOutput(BaseModel):
    """Output for file assignment step."""
    assignments: List[FileAssignment] = Field(
        ...,
        description="List of file assignments"
    )

# ============================================================================
# File Designer
# ============================================================================

class FileDesigner:
    """Main orchestrator for skeleton building."""

    def __init__(
        self,
        rpg: RPG,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 10,
        config: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[str] = None,
        target_language: Optional[str] = None,
    ):
        """Initialize FileDesigner.

        Args:
            rpg: The RPG structure to build skeleton from
            llm_client: LLM client for API calls
            max_iterations: Maximum iterations for iterative design
            config: Optional configuration dictionary
            trajectory: Optional trajectory tracker for logging steps
            step_id: Optional step ID for trajectory tracking
            target_language: Optional explicit target language
                (e.g. ``"python"``, ``"go"``). When ``None`` the
                effective language is resolved from RPG root meta with
                fallback to ``"python"``. The resolved backend provides
                file-extension, package-marker, and prompt-directive
                behaviour for skeleton generation.
        """
        self.rpg = rpg
        self.llm_client = llm_client or LLMClient(trajectory=trajectory, step_id=step_id)
        self.max_iterations = max_iterations
        self.config = config or {}
        self.trajectory = trajectory
        self.step_id = step_id

        self.logger = logging.getLogger(__name__)

        # Build a minimal RPG-shaped dict so language resolution does
        # not trigger full graph serialization.
        rpg_meta_lang = None
        repo_node = getattr(self.rpg, "repo_node", None)
        if repo_node is not None and getattr(repo_node, "meta", None) is not None:
            rpg_meta_lang = getattr(repo_node.meta, "language", None)
        rpg_dict_minimal = {"root": {"meta": {"language": rpg_meta_lang}}}
        feature_spec_stub = (
            {
                "meta": {
                    "primary_language": target_language,
                    "target_languages": [target_language],
                }
            }
            if target_language
            else None
        )
        self.target_language = resolve_decoder_language(
            feature_spec=feature_spec_stub,
            rpg_obj=rpg_dict_minimal,
        )
        self.backend = get_backend(self.target_language)

        # Load project background / technology context (empty string if unavailable)
        try:
            self._project_background = get_project_background_context()
        except Exception:
            self._project_background = ""

        # Initialize empty skeleton
        self.skeleton = RepoSkeleton({})

        # Component to directory mapping (for RPG update)
        self.component_to_dir: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "components_processed": 0,
            "features_assigned": 0,
            "files_created": 0,
            "init_files_created": 0,
            "iterations_used": 0,
            "llm_calls_made": 0,
            "validation_retries": 0
        }

    def run(self, result_path: Optional[Path] = None) -> Tuple[RepoSkeleton, RPG, Dict[str, Any]]:
        """Execute complete skeleton building workflow.

        Returns:
            Tuple of (skeleton, updated_rpg, results_dict)
        """
        self.logger.info("=" * 70)
        self.logger.info("FILE DESIGNER - SKELETON BUILDING")
        self.logger.info("=" * 70)

        try:
            # Step 1: Extract component data from RPG
            components_data = self._extract_components_from_rpg()
            self.logger.info(f"Extracted {len(components_data)} components from RPG")

            if not components_data:
                return self.skeleton, self.rpg, {"success": False, "error": "No components found"}

            # Step 2: Generate directory structure (Raw Skeleton)
            self.logger.info("\n[Step 1] Generating directory structure...")
            dir_assignments = self._generate_directory_structure(components_data)

            if not dir_assignments:
                return self.skeleton, self.rpg, {"success": False, "error": "Directory structure generation failed"}

            # Step 3: Assign features to files for each component (Group Skeleton)
            self.logger.info("\n[Step 2] Assigning features to files...")
            file_assignments = self._assign_features_to_files(components_data, dir_assignments)

            if not file_assignments:
                return self.skeleton, self.rpg, {"success": False, "error": "Feature assignment failed"}

            # Step 4: Build final skeleton
            self.logger.info("\n[Step 3] Building final skeleton structure...")
            self._build_final_skeleton(file_assignments)

            # Step 5: Update RPG with directory assignments
            self.logger.info("\n[Step 4] Updating RPG with directory assignments...")
            self._update_rpg_with_directories()

            # Step 6: Save results
            if result_path:
                self.skeleton.save_json(str(result_path))
                self.logger.info(f"Skeleton saved to: {result_path}")

            # Build success response
            results = {
                "success": True,
                "statistics": self.stats,
                "components_processed": self.stats["components_processed"],
                "features_assigned": self.stats["features_assigned"],
                "files_created": self.stats["files_created"],
                "skeleton_nodes": len(self.skeleton.path_to_node),
            }

            self.logger.info("\n" + "=" * 70)
            self.logger.info("SKELETON BUILDING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70)
            self._print_statistics()

            return self.skeleton, self.rpg, results

        except Exception as e:
            self.logger.error(f"Skeleton building failed: {e}")
            return self.skeleton, self.rpg, {"success": False, "error": str(e)}

    def _extract_components_from_rpg(self) -> List[Dict[str, Any]]:
        """Extract component data from RPG for skeleton building."""
        components = []

        # Get all level-1 nodes (functional areas) as components
        for node in self.rpg.nodes.values():
            if node.level == 1 and node.name and node.id != self.rpg.repo_node.id:
                # Extract subtree for this component
                subtree = self._extract_subtree_from_node(node)

                component = {
                    "name": node.name,
                    "description": getattr(node.meta, 'description', '') if node.meta else '',
                    "refactored_subtree": subtree
                }
                components.append(component)

        return components

    def _extract_subtree_from_node(self, node: Node) -> Dict[str, Any]:
        """Extract hierarchical subtree from RPG node."""
        children = node.children()

        if not children:
            # Leaf node
            return node.name

        subtree = {}
        for child in children:
            child_subtree = self._extract_subtree_from_node(child)
            if isinstance(child_subtree, str):
                # Child is a leaf
                if child.name not in subtree:
                    subtree[child.name] = child_subtree
            else:
                # Child has subtree
                subtree[child.name] = child_subtree

        return subtree

    def _generate_directory_structure(
        self,
        components_data: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> Optional[Dict[str, str]]:
        """Generate directory structure mapping components to directories with validation."""
        # Extract required component names
        required_components = [comp["name"] for comp in components_data]

        # Build base prompts
        repo_info = f"Repository: {self.rpg.repo_name}\nPurpose: {self.rpg.repo_info}"
        component_summary = build_component_summary(components_data)

        # Include technology context when available
        tech_section = ""
        if self._project_background and self._project_background.strip():
            tech_section = (
                f"\n{self._project_background}\n"
                "When a specific technology stack is described above, design the directory\n"
                "structure to accommodate the target language and framework conventions.\n"
            )

        hints = self.backend.prompt_hints()
        safe_repo_name = self.backend.sanitize_module_identifier(
            self.rpg.repo_name.replace(" ", "_")
        )

        base_user_prompt = f"""## Repository Information
{repo_info}
{tech_section}
## Components to Organize ({len(components_data)} total)
{component_summary}

## Task
Assign each component to an appropriate directory path.
Use "{safe_repo_name}" as the project name in paths (e.g., src/{safe_repo_name}/...).
IMPORTANT: {hints.module_naming_rule}
Target layout example:
{hints.package_layout_example}
IMPORTANT: You MUST assign ALL {len(required_components)} components: {', '.join(required_components)}
"""

        last_error = ""
        
        for attempt in range(max_retries):
            self.logger.info(f"   Directory structure generation attempt {attempt + 1}/{max_retries}")
            
            # Build prompt with error feedback if needed
            user_prompt = base_user_prompt
            if last_error:
                user_prompt += f"\n\n## Previous Attempt Failed\nError: {last_error}\nPlease fix the issues and try again."

            # Call LLM
            _, result, _ = self.llm_client.call_structured(
                system_prompt=with_language_directive(
                    RAW_SKELETON_PROMPT, self.backend,
                ),
                user_prompt=user_prompt,
                response_model=DirectoryStructureOutput,
                purpose=f"directory_structure_{attempt + 1}"
            )

            self.stats["llm_calls_made"] += 1

            if not result:
                last_error = "Failed to parse LLM response"
                self.stats["validation_retries"] += 1
                continue

            # Process assignments into simple mapping
            component_to_dir = {}
            for assignment in result.assignments:
                component_to_dir[assignment.component_name] = assignment.directory_path

            # Validate completeness (identifier rules come from the
            # resolved backend so Go segments are checked against Go
            # naming rules, not Python's).
            is_valid, error_msg = validate_directory_structure(
                component_to_dir, required_components, backend=self.backend,
            )
            
            if is_valid:
                self.logger.info("\n   Directory Structure (validated):")
                for comp, dir_path in component_to_dir.items():
                    self.logger.info(f"   -  {comp} → {dir_path}/")
                self.logger.info(f"\n   Reasoning: {result.overall_reasoning}")
                
                # Store for later RPG update
                self.component_to_dir = component_to_dir
                return component_to_dir
            else:
                self.logger.warning(f"   Validation failed: {error_msg}")
                last_error = error_msg
                self.stats["validation_retries"] += 1

        self.logger.error(f"Directory structure generation failed after {max_retries} attempts")
        return None

    def _assign_features_to_files(
        self,
        components_data: List[Dict[str, Any]],
        dir_assignments: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Assign features to files for each component."""
        all_assignments = []

        for comp_data in components_data:
            comp_name = comp_data["name"]
            comp_desc = comp_data.get("description", "")
            refactored_subtree = comp_data.get("refactored_subtree", {})

            if comp_name not in dir_assignments:
                self.logger.warning(f"No directory assignment for component: {comp_name}")
                continue

            comp_dir = dir_assignments[comp_name]

            # Extract all features for this component
            features = extract_features_from_subtree(refactored_subtree, comp_name)
            feat_descs = extract_leaf_descriptions_from_subtree(refactored_subtree, comp_name)
            if not features:
                self.logger.warning(f"No features found for component: {comp_name}")
                continue

            self.logger.info(f"   Processing: {comp_name}")
            self.logger.info(f"   Directory: {comp_dir}/")
            self.logger.info(f"   Features: {len(features)}")

            # Build user prompt for feature assignment
            feature_list = format_feature_list(features, feat_descs)
            repo_info = f"Repository: {self.rpg.repo_name}\nPurpose: {self.rpg.repo_info}"

            # Include technology context when available
            tech_section = ""
            if self._project_background and self._project_background.strip():
                tech_section = f"\n{self._project_background}\n"

            hints = self.backend.prompt_hints()

            user_prompt = f"""## Repository Information
{repo_info}
{tech_section}
## Component: {comp_name}
Description: {comp_desc}
Directory: {comp_dir}

## Features to Assign ({len(features)} total)
{feature_list}

## Task
Assign ALL the above features to {hints.display_name} source files under {comp_dir}/.
Source files should use the {hints.file_extension} extension.
Every feature MUST be assigned to exactly one file.
"""

            # Call LLM for feature assignment
            _, result, _ = self.llm_client.call_structured(
                system_prompt=with_language_directive(
                    GROUP_SKELETON_PROMPT, self.backend,
                ),
                user_prompt=user_prompt,
                response_model=FileAssignmentOutput,
                purpose=f"feature_assignment_{comp_name}"
            )

            self.stats["llm_calls_made"] += 1

            if not result:
                self.logger.error(f"Feature assignment failed for component: {comp_name}")
                continue

            # Process and validate assignments
            comp_assignments = []
            assigned_features = set()
            path_errors = []

            for assignment in result.assignments:
                file_path = assignment.file_path
                features_list = assignment.features

                # Validate file path is under the allowed directory
                is_valid_path, path_error = validate_file_path_constraint(
                    file_path, [comp_dir]
                )
                if not is_valid_path:
                    path_errors.append(path_error)
                    self.logger.warning(f"   Path constraint violation: {path_error}")
                    # Try to fix by prepending the correct directory
                    if not file_path.startswith(comp_dir):
                        file_name = file_path.split("/")[-1]
                        file_path = f"{comp_dir}/{file_name}"
                        self.logger.info(f"   Auto-corrected to: {file_path}")

                # Validate features exist
                valid_features = []
                for feature in features_list:
                    if feature in features and feature not in assigned_features:
                        valid_features.append(feature)
                        assigned_features.add(feature)

                if valid_features:
                    comp_assignments.append({
                        "file_path": file_path,
                        "features": valid_features,
                        "purpose": assignment.purpose,
                        "component": comp_name
                    })

            if path_errors:
                self.logger.warning(f"   {len(path_errors)} path constraint violations were auto-corrected")

            # Check for unassigned features
            unassigned = [f for f in features if f not in assigned_features]
            if unassigned:
                # Create fallback file for unassigned features. Extension
                # comes from the resolved language backend so a Go run
                # produces ``misc.go`` instead of ``misc.py``.
                fallback_file = f"{comp_dir}/misc{self.backend.file_extension}"
                comp_assignments.append({
                    "file_path": fallback_file,
                    "features": unassigned,
                    "purpose": "Miscellaneous features",
                    "component": comp_name
                })

            all_assignments.extend(comp_assignments)
            self.stats["features_assigned"] += len(features)
            self.stats["components_processed"] += 1

            self.logger.info(f"      Assigned {len(assigned_features)} features to {len(comp_assignments)} files")

        return all_assignments

    def _build_final_skeleton(self, file_assignments: List[Dict[str, Any]]):
        """Build the final skeleton structure from file assignments."""
        # Pre-merge assignments with the same file_path so that features from
        # multiple components going to the same file (e.g. shared misc.py) are
        # all preserved instead of the last write silently overwriting earlier ones.
        merged: Dict[str, List[str]] = {}
        for assignment in file_assignments:
            file_path = assignment["file_path"]
            features = assignment["features"]
            if file_path in merged:
                merged[file_path].extend(features)
            else:
                merged[file_path] = list(features)

        for file_path, features in merged.items():
            self.skeleton.insert_file(
                file_path=file_path,
                code="",
                feature_paths=features
            )
            self.stats["files_created"] += 1

        # Add package-marker files to all directories (Python:
        # ``__init__.py``; Go / Rust / TS: no-op via backend).
        init_files_added = self.skeleton.add_init_files(backend=self.backend)
        self.stats["init_files_created"] = init_files_added
        self.logger.info(f"Added {init_files_added} package marker files")

        self.logger.info(f"Created skeleton with {len(self.skeleton.path_to_node)} total nodes")

    def _update_rpg_with_directories(self):
        """Update RPG nodes with directory path assignments.
        
        This writes the assigned directory paths back into the RPG nodes'
        metadata, similar to ZeroRepo's behavior.
        """
        updated_count = 0
        
        for component_name, dir_path in self.component_to_dir.items():
            # Find the component node in RPG (level 1 node with matching name)
            component_node = None
            for node in self.rpg.nodes.values():
                if node.level == 1 and node.name == component_name:
                    component_node = node
                    break
            
            if not component_node:
                self.logger.warning(f"Could not find RPG node for component: {component_name}")
                continue
            
            # Update node metadata with directory path
            if component_node.meta is None:
                component_node.meta = NodeMetaData(
                    type_name=NodeType.DIRECTORY,
                    path=dir_path
                )
            else:
                component_node.meta.type_name = NodeType.DIRECTORY
                component_node.meta.path = dir_path
            
            updated_count += 1
            self.logger.debug(f"   Updated RPG node '{component_name}' with path: {dir_path}")
        
        self.logger.info(f"   Updated {updated_count} RPG nodes with directory paths")

    def patch(
        self,
        missing_by_component: Dict[str, List[str]],
        dir_assignments: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Assign only missing features to files, reusing existing directory assignments.

        Skips directory structure generation entirely — uses existing assignments
        from the already-built skeleton.

        Args:
            missing_by_component: {component_name: [full_feature_path, ...]}
            dir_assignments: {component_name: directory_path} from existing RPG/skeleton

        Returns:
            List of file assignment dicts (same format as _assign_features_to_files)
        """
        all_assignments = []

        for comp_name, missing_features in missing_by_component.items():
            if not missing_features:
                continue
            if comp_name not in dir_assignments:
                self.logger.warning(f"No directory assignment for component: {comp_name}")
                continue

            comp_dir = dir_assignments[comp_name]
            missing_features_set = set(missing_features)
            self.logger.info(f"   Patching: {comp_name} ({len(missing_features)} missing features)")
            self.logger.info(f"   Directory: {comp_dir}/")

            feature_list = format_feature_list(missing_features)
            repo_info = f"Repository: {self.rpg.repo_name}\nPurpose: {self.rpg.repo_info}"

            tech_section = ""
            if self._project_background and self._project_background.strip():
                tech_section = f"\n{self._project_background}\n"

            hints = self.backend.prompt_hints()

            user_prompt = f"""## Repository Information
{repo_info}
{tech_section}
## Component: {comp_name}
Directory: {comp_dir}

## Missing Features to Assign ({len(missing_features)} total)
{feature_list}

## Task
Assign ALL the above features to {hints.display_name} source files under {comp_dir}/.
Source files should use the {hints.file_extension} extension.
Every feature MUST be assigned to exactly one file.
You may add features to existing files in this directory or create new files.
"""

            _, result, _ = self.llm_client.call_structured(
                system_prompt=with_language_directive(
                    GROUP_SKELETON_PROMPT, self.backend,
                ),
                user_prompt=user_prompt,
                response_model=FileAssignmentOutput,
                purpose=f"patch_feature_assignment_{comp_name}"
            )

            self.stats["llm_calls_made"] += 1

            if not result:
                self.logger.error(f"Patch assignment failed for component: {comp_name}")
                fallback_file = f"{comp_dir}/misc{self.backend.file_extension}"
                all_assignments.append({
                    "file_path": fallback_file,
                    "features": missing_features,
                    "purpose": "Miscellaneous features (patch fallback)",
                    "component": comp_name
                })
                continue

            comp_assignments = []
            assigned_features = set()

            for assignment in result.assignments:
                file_path = assignment.file_path
                features_list = assignment.features

                is_valid_path, path_error = validate_file_path_constraint(file_path, [comp_dir])
                if not is_valid_path:
                    self.logger.warning(f"   Path constraint violation: {path_error}")
                    if not file_path.startswith(comp_dir):
                        file_name = file_path.split("/")[-1]
                        file_path = f"{comp_dir}/{file_name}"
                        self.logger.info(f"   Auto-corrected to: {file_path}")

                valid_features = []
                for feature in features_list:
                    if feature in missing_features_set and feature not in assigned_features:
                        valid_features.append(feature)
                        assigned_features.add(feature)

                if valid_features:
                    comp_assignments.append({
                        "file_path": file_path,
                        "features": valid_features,
                        "purpose": assignment.purpose,
                        "component": comp_name
                    })

            unassigned = [f for f in missing_features if f not in assigned_features]
            if unassigned:
                fallback_file = f"{comp_dir}/misc{self.backend.file_extension}"
                comp_assignments.append({
                    "file_path": fallback_file,
                    "features": unassigned,
                    "purpose": "Miscellaneous features",
                    "component": comp_name
                })

            all_assignments.extend(comp_assignments)
            self.stats["features_assigned"] += len(missing_features)
            self.stats["components_processed"] += 1
            self.logger.info(
                f"      Assigned {len(assigned_features)} features to {len(comp_assignments)} files"
            )

        return all_assignments

    def _print_statistics(self):
        """Print final statistics."""
        print("Statistics:")
        print(f"  Components processed: {self.stats['components_processed']}")
        print(f"  Features assigned: {self.stats['features_assigned']}")
        print(f"  Files created: {self.stats['files_created']}")
        print(f"  __init__.py files added: {self.stats['init_files_created']}")
        print(f"  LLM calls made: {self.stats['llm_calls_made']}")

        skeleton_stats = self.skeleton.get_statistics()
        print(f"  Total skeleton nodes: {skeleton_stats['total_nodes']}")
        print(f"  File nodes: {skeleton_stats['file_nodes']}")
        print(f"  Directory nodes: {skeleton_stats['directory_nodes']}")
        print(f"  __init__.py files: {skeleton_stats.get('init_files', 0)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("FileDesigner module loaded successfully")
    print("Use this module from build_skeleton.py for full functionality")