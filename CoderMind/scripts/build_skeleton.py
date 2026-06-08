#!/usr/bin/env python3
"""Build Skeleton Script - Implementation Level Step 1.

Function: Design repository file structure from component architecture
- Step 1: Build RPG (Repository Program Graph) from component data
- Step 2: Generate directory structure mapping components to directories
- Step 3: Assign features to specific Python files using professional prompts

Input: .cmind/feature_tree.json (component list from refactor step)
Output: .cmind/skeleton.json (tree-structured file skeleton with feature assignments)
       .cmind/repo_rpg.json (intermediate RPG structure)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import trajectory module
from common.trajectory import Trajectory, load_or_create_trajectory

# Import required modules
from rpg import RPG
from rpg.builder import create_initial_rpg, load_refactor_feature_data, get_rpg_statistics
from skeleton.skeleton_models import RepoSkeleton, DirectoryNode, normalize_path
from skeleton.file_designer import FileDesigner

# Import centralized paths
from common.paths import (
    FEATURE_TREE_FILE,
    SKELETON_FILE,
    REPO_RPG_FILE,
)
from common import print_unicode_table
from common.language_meta import extract_language_metadata, metadata_with_languages
from pathlib import Path as PPath
from rpg import NodeMetaData
from skeleton.skeleton_prompts import extract_features_from_subtree


# ============================================================================
# Utility Functions
# ============================================================================

def convert_skeleton_to_cmind_format(skeleton: RepoSkeleton, rpg: RPG) -> Dict[str, Any]:
    """Convert skeleton format to CoderMind's expected format.

    This ensures compatibility with existing validation and summary scripts.
    """

    def convert_node(node):
        """Convert skeleton node to CoderMind format recursively."""
        result = {
            "type": "directory" if node.is_dir else "file",
            "name": node.name,
            "path": node.path,
        }

        if node.is_dir:
            result["children"] = [convert_node(child) for child in node.children()]
        else:
            result["feature_paths"] = getattr(node, 'feature_paths', [])
            # Find component name from feature paths
            if hasattr(node, 'feature_paths') and node.feature_paths:
                # Extract component from first feature path
                first_feature = node.feature_paths[0]
                if '/' in first_feature:
                    component = first_feature.split('/')[0]
                    result["component"] = component

        return result

    # Build CoderMind compatible output
    output = {
        "repository_name": rpg.repo_name,
        "repository_purpose": rpg.repo_info,
        "meta": metadata_with_languages({
            "meta": {
                "primary_language": getattr(rpg.repo_node.meta, "language", None)
                if rpg.repo_node and rpg.repo_node.meta else None
            }
        }),
        "root": convert_node(skeleton.root),
        "statistics": {
            "total_components": len([n for n in rpg.nodes.values() if n.level == 1]),
            "total_features": sum(len(f.feature_paths) for f in skeleton.get_all_file_nodes()),
            "total_files": len(skeleton.get_all_file_nodes()),
            "total_directories": len([n for n in skeleton.path_to_node.values() if n.is_dir]),
        }
    }

    return output


# ============================================================================
# Skeleton Builder
# ============================================================================

class SkeletonBuilder:
    """Skeleton builder."""

    def __init__(self, max_iterations: int = 10, trajectory: Trajectory = None):
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
        self.trajectory = trajectory

        # Build state
        self.repo_name = ""
        self.target_language = None
        self.repo_data = {}
        self.rpg = None
        self.skeleton = None
        self.file_designer = None

        # Statistics
        self.stats = {
            "total_features": 0,
            "assigned_features": 0,
            "total_files": 0,
            "total_components": 0,
            "llm_calls": 0
        }

        # Trajectory step tracking
        self._current_step_id: Optional[int] = None

    def build(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete skeleton building workflow."""
        self.repo_data = input_data
        self.repo_name = input_data.get("repository_name", "project")
        self.target_language = extract_language_metadata(input_data)[0]
        components = input_data.get("components", [])

        if not components:
            return {"error": "No components found in input"}

        print("\n" + "=" * 70)
        print("SKELETON BUILDING")
        print("=" * 70)
        print(f"Repository: {self.repo_name}")
        print(f"Components: {len(components)}")

        # Count total features
        self.stats["total_components"] = len(components)
        self.stats["total_features"] = sum(
            self._count_features_in_component(comp.get("refactored_subtree", {}))
            for comp in components
        )
        print(f"Total Features: {self.stats['total_features']}")
        print("=" * 70)

        # Initialize trajectory metadata
        if self.trajectory:
            self.trajectory.metadata.update({
                "repository_name": self.repo_name,
                "component_count": len(components),
                "total_features": self.stats["total_features"]
            })
            self.trajectory.save()

        try:
            # Step 1: Build RPG from component data
            print("\n[Step 1] Building RPG (Repository Program Graph)...")
            step1 = None
            if self.trajectory:
                step1 = self.trajectory.add_step("build_rpg", "Build RPG from component architecture")
                self.trajectory.start_step(step1.step_id)
                self._current_step_id = step1.step_id

            if not self._step1_build_rpg():
                if self.trajectory and step1:
                    self.trajectory.fail_step(step1.step_id, "RPG building failed")
                return {"error": "RPG building failed"}

            if self.trajectory and step1:
                rpg_stats = get_rpg_statistics(self.rpg)
                self.trajectory.complete_step(step1.step_id, {
                    "rpg_statistics": rpg_stats
                })

            # Step 2: Generate skeleton using FileDesigner
            print("\n[Step 2] Generating skeleton with FileDesigner...")
            step2 = None
            if self.trajectory:
                step2 = self.trajectory.add_step("file_design", "Generate skeleton using FileDesigner")
                self.trajectory.start_step(step2.step_id)
                self._current_step_id = step2.step_id

            if not self._step2_file_design():
                if self.trajectory and step2:
                    self.trajectory.fail_step(step2.step_id, "File design failed")
                return {"error": "File design failed"}

            if self.trajectory and step2:
                skeleton_stats = self.skeleton.get_statistics()
                self.trajectory.complete_step(step2.step_id, {
                    "skeleton_statistics": skeleton_stats
                })

            # Step 2.5: Update RPG paths from skeleton
            print("\n[Step 2.5] Updating RPG paths from skeleton...")
            paths_updated = self._update_rpg_paths_from_skeleton()
            print(f"   [OK] Updated {paths_updated} nodes with path information")

            # Step 3: Convert and save results
            print("\n[Step 3] Converting to CoderMind format...")
            result = self._build_result()
            
            # Save updated RPG (with directory assignments)
            self.rpg.save_json(str(REPO_RPG_FILE), indent=2)
            print(f"   [OK] Updated RPG saved to: {REPO_RPG_FILE.name}")

            self._print_summary()

            return result

        except Exception as e:
            self.logger.error(f"Skeleton building failed: {e}")
            if self.trajectory:
                self.trajectory.fail_step(self._current_step_id or 0, f"Build failed: {e}")
            return {"error": str(e)}

    def _step1_build_rpg(self) -> bool:
        """Step 1: Build RPG from component data."""
        try:
            self.rpg = create_initial_rpg(self.repo_data)

            # Save RPG to intermediate file
            self.rpg.save_json(str(REPO_RPG_FILE), indent=2)

            # Print RPG statistics
            stats = get_rpg_statistics(self.rpg)
            print("   [OK] RPG built successfully:")
            print(f"     - Total nodes: {stats['total_nodes']}")
            print(f"     - Node types: {dict(stats['node_types'])}")
            print(f"     - Level distribution: {dict(stats['levels'])}")
            print(f"   [OK] RPG saved to: {REPO_RPG_FILE.name}")

            return True

        except Exception as e:
            self.logger.error(f"RPG building failed: {e}")
            return False

    def _step2_file_design(self) -> bool:
        """Step 2: Generate skeleton using FileDesigner."""
        try:
            # Initialize FileDesigner with trajectory support
            self.file_designer = FileDesigner(
                rpg=self.rpg,
                max_iterations=self.max_iterations,
                trajectory=self.trajectory,
                step_id=self._current_step_id,
                target_language=self.target_language,
            )

            # Run file design process
            self.skeleton, updated_rpg, design_results = self.file_designer.run()

            if not design_results.get("success", False):
                self.logger.error("FileDesigner failed")
                return False

            # Update RPG with changes from FileDesigner
            self.rpg = updated_rpg

            # Update statistics
            self.stats.update({
                "assigned_features": design_results.get("features_assigned", 0),
                "total_files": design_results.get("files_created", 0),
                "llm_calls": design_results.get("statistics", {}).get("llm_calls_made", 0),
                "validation_retries": design_results.get("statistics", {}).get("validation_retries", 0)
            })

            print("   [OK] Skeleton generated successfully:")
            print(f"     - Components processed: {design_results.get('components_processed', 0)}")
            print(f"     - Features assigned: {self.stats['assigned_features']}")
            print(f"     - Files created: {self.stats['total_files']}")
            print(f"     - LLM calls made: {self.stats['llm_calls']}")
            if self.stats.get('validation_retries', 0) > 0:
                print(f"     - Validation retries: {self.stats['validation_retries']}")

            return True

        except Exception as e:
            self.logger.error(f"File design failed: {e}")
            return False

    def _build_result(self) -> Dict[str, Any]:
        """Build the final result dictionary in CoderMind format."""
        # Convert to CoderMind compatible format
        result = convert_skeleton_to_cmind_format(self.skeleton, self.rpg)
        result["meta"] = metadata_with_languages({
            "meta": {
                "primary_language": self.target_language,
                "target_languages": [self.target_language] if self.target_language else [],
            }
        })

        # Add statistics
        result["statistics"].update({
            "rpg_nodes": len(self.rpg.nodes),
            "rpg_edges": len(self.rpg.edges),
            "llm_calls_made": self.stats["llm_calls"],
            "validation_retries": self.stats.get("validation_retries", 0),
        })

        # Get component directories from FileDesigner (updated via RPG)
        component_directories = {}
        
        # First try to get from RPG nodes (which have been updated with paths)
        for node in self.rpg.nodes.values():
            if node.level == 1 and node.name != self.rpg.repo_name:
                if node.meta and hasattr(node.meta, 'path') and node.meta.path:
                    # Use the path stored in RPG node metadata
                    component_directories[node.name] = node.meta.path
                else:
                    # Fallback: try to infer directory from skeleton
                    for skeleton_node in self.skeleton.path_to_node.values():
                        if (isinstance(skeleton_node, DirectoryNode) and
                            node.name.lower().replace(' ', '_') in skeleton_node.path.lower()):
                            component_directories[node.name] = skeleton_node.path
                            break

        result["component_directories"] = component_directories

        return result

    def _update_rpg_paths_from_skeleton(self) -> int:
        """Update RPG node paths from skeleton file assignments.
        
        For each feature node in RPG, find the corresponding file in skeleton
        and update the node's meta.path accordingly.
        
        Returns:
            Number of nodes updated with path information
        """
        if not self.skeleton or not self.rpg:
            return 0
        
        updated_count = 0
        
        # Build feature_path -> file_path mapping from skeleton
        feature_to_file = {}
        for file_node in self.skeleton.get_all_file_nodes():
            file_path = file_node.path
            for feature_path in file_node.feature_paths:
                feature_to_file[feature_path] = file_path
        
        # Update component (L1) nodes with directory paths
        component_dirs = {}
        for file_path, feature_path in [(f.path, fp) 
                                         for f in self.skeleton.get_all_file_nodes() 
                                         for fp in f.feature_paths]:
            if '/' in feature_path:
                component_name = feature_path.split('/')[0]
                # Get the directory from file path
                dir_path = str(PPath(file_path).parent)
                if component_name not in component_dirs:
                    component_dirs[component_name] = dir_path
        
        # Update RPG nodes
        for node in self.rpg.nodes.values():
            if node.level == 0:
                # Repo root
                if not (node.meta and node.meta.path):
                    if not node.meta:
                        node.meta = NodeMetaData()
                    node.meta.path = "."
                    updated_count += 1
            elif node.level == 1:
                # Component/functional_area level
                if node.name in component_dirs:
                    if not node.meta:
                        node.meta = NodeMetaData()
                    if not node.meta.path:
                        node.meta.path = component_dirs[node.name]
                        updated_count += 1
            else:
                # Feature nodes - use feature path to find file
                feature_path = node.feature_path()
                if feature_path in feature_to_file:
                    file_path = feature_to_file[feature_path]
                    if not node.meta:
                        node.meta = NodeMetaData()
                    if not node.meta.path:
                        node.meta.path = file_path
                        updated_count += 1
        
        if updated_count > 0:
            self.logger.info(f"Updated {updated_count} RPG nodes with path information")
        
        return updated_count

    def _count_features_in_component(self, subtree: Any) -> int:
        """Count features in component subtree."""
        if isinstance(subtree, dict):
            total = 0
            for key, value in subtree.items():
                if key == "description":
                    continue
                total += self._count_features_in_component(value)
            return total
        elif isinstance(subtree, list):
            return len([item for item in subtree if item])
        else:
            return 1 if subtree else 0

    def _print_summary(self):
        """Print build summary."""
        print("\n" + "=" * 70)
        print("SKELETON BUILDING COMPLETE")
        print("=" * 70)
        print(f"Total Components: {self.stats['total_components']}")
        print(f"Total Features: {self.stats['total_features']}")
        print(f"Assigned Features: {self.stats['assigned_features']}")
        print(f"Total Files: {self.stats['total_files']}")
        print(f"LLM Calls Made: {self.stats['llm_calls']}")

        if self.skeleton:
            skeleton_stats = self.skeleton.get_statistics()
            print(f"Skeleton Nodes: {skeleton_stats['total_nodes']}")
            print(f"__init__.py Files: {skeleton_stats.get('init_files', 0)}")

        # Print file summary
        if self.skeleton:
            files = self.skeleton.get_all_file_nodes()
            if files:
                rows = []
                for f in sorted(files, key=lambda x: x.path)[:20]:  # Show first 20
                    rows.append([f.path, len(f.feature_paths)])

                if rows:
                    print_unicode_table(
                        headers=["File Path", "Features"],
                        rows=rows,
                        title="File Assignments (Top 20)"
                    )
                    if len(files) > 20:
                        print(f"   ... and {len(files) - 20} more files")


# ============================================================================
# Patch Mode: Incremental Feature Assignment
# ============================================================================

def patch_missing(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Incrementally assign missing features to the existing skeleton.

    Loads the existing skeleton.json and rpg.json, detects which features
    from the feature tree are not yet assigned, and runs a targeted LLM
    assignment for only those features — reusing existing directory structure.

    Returns a result dict with type "patch", "skip", or "error".
    """
    if not SKELETON_FILE.exists():
        return {"error": "No existing skeleton found. Run build first."}
    if not REPO_RPG_FILE.exists():
        return {"error": "No RPG file found. Run build first."}

    skeleton = RepoSkeleton.load_json(str(SKELETON_FILE))
    rpg = RPG.from_json(str(REPO_RPG_FILE))

    # Collect all features from input, grouped by component
    all_input_features: set = set()
    features_by_component: Dict[str, list] = {}
    for comp in input_data.get("components", []):
        comp_name = comp["name"]
        subtree = comp.get("refactored_subtree", {})
        comp_features = extract_features_from_subtree(subtree, comp_name)
        features_by_component[comp_name] = comp_features
        all_input_features.update(comp_features)

    # Collect features already assigned in skeleton
    existing_features: set = set()
    for file_node in skeleton.get_all_file_nodes():
        existing_features.update(file_node.feature_paths)

    # Find missing features per component
    missing_by_component: Dict[str, list] = {}
    for comp_name, comp_features in features_by_component.items():
        missing = [f for f in comp_features if f not in existing_features]
        if missing:
            missing_by_component[comp_name] = missing

    total_missing = sum(len(v) for v in missing_by_component.values())
    if total_missing == 0:
        print("[OK] No missing features. Skeleton is already complete.")
        return {"type": "skip", "message": "All features already assigned"}

    print(f"\n[Patch] {total_missing} missing features across {len(missing_by_component)} components:")
    for comp_name, features in missing_by_component.items():
        print(f"  - {comp_name}: {len(features)} missing")

    # Extract dir_assignments from RPG L1 nodes (set during original build)
    dir_assignments: Dict[str, str] = {}
    for node in rpg.nodes.values():
        if node.level == 1 and node.name != rpg.repo_name:
            if node.meta and node.meta.path:
                dir_assignments[node.name] = node.meta.path

    missing_without_dir = [c for c in missing_by_component if c not in dir_assignments]
    if missing_without_dir:
        return {
            "error": (
                f"No directory assignments in RPG for: {missing_without_dir}. "
                "Cannot patch — run full build first."
            )
        }

    # Run patch via FileDesigner (skips directory structure generation)
    file_designer = FileDesigner(rpg=rpg)
    new_assignments = file_designer.patch(missing_by_component, dir_assignments)

    if not new_assignments:
        return {"error": "Patch produced no assignments"}

    # Merge new assignments into existing skeleton.
    # insert_file() OVERWRITES feature_paths on existing files, so handle manually.
    merged_count = 0
    new_file_count = 0
    for assignment in new_assignments:
        file_path = assignment["file_path"]
        features = assignment["features"]
        norm = normalize_path(file_path)

        if norm in skeleton.path_to_node:
            existing_node = skeleton.path_to_node[norm]
            if hasattr(existing_node, "feature_paths"):
                existing_node.feature_paths.extend(features)
                merged_count += len(features)
        else:
            skeleton.insert_file(file_path, "", features)
            new_file_count += 1

    skeleton.add_init_files()

    # Update RPG paths for any newly assigned nodes
    feature_to_file: Dict[str, str] = {}
    for file_node in skeleton.get_all_file_nodes():
        for fp in file_node.feature_paths:
            feature_to_file[fp] = file_node.path

    for node in rpg.nodes.values():
        if node.level > 1:
            fp = node.feature_path()
            if fp in feature_to_file and not (node.meta and node.meta.path):
                if not node.meta:
                    node.meta = NodeMetaData()
                node.meta.path = feature_to_file[fp]

    # Re-convert and save
    result = convert_skeleton_to_cmind_format(skeleton, rpg)
    result["statistics"].update({
        "rpg_nodes": len(rpg.nodes),
        "rpg_edges": len(rpg.edges),
        "llm_calls_made": file_designer.stats["llm_calls_made"],
    })

    component_directories: Dict[str, str] = {}
    for node in rpg.nodes.values():
        if node.level == 1 and node.name != rpg.repo_name:
            if node.meta and node.meta.path:
                component_directories[node.name] = node.meta.path
    result["component_directories"] = component_directories

    with open(str(SKELETON_FILE), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    rpg.save_json(str(REPO_RPG_FILE), indent=2)

    print("\n[OK] Patch complete:")
    print(f"  - Missing features patched: {total_missing}")
    print(f"  - New files created: {new_file_count}")
    print(f"  - Features merged into existing files: {merged_count}")

    return {
        "type": "patch",
        "total_missing_patched": total_missing,
        "new_files_created": new_file_count,
        "features_merged": merged_count,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build repository skeleton from component architecture"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(FEATURE_TREE_FILE),
        help=f"Input file (default: {FEATURE_TREE_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(SKELETON_FILE),
        help=f"Output file (default: {SKELETON_FILE})"
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=10,
        help="Max iterations per component (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording"
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Patch mode: only assign missing features to existing skeleton"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        print(f"ERROR: Input file not found: {input_path}")
        print("Please run /cmind.feature_refactor first.")
        return 1

    try:
        input_data = load_refactor_feature_data(input_path)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        print(f"ERROR: {e}")
        return 1

    # --patch mode: incremental assignment of missing features only
    if args.patch:
        try:
            result = patch_missing(input_data)
            if "error" in result:
                print(f"ERROR: {result['error']}")
                return 1
            return 0
        except Exception as e:
            logger.error(f"Patch failed: {e}")
            raise

    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("build_skeleton")

        # Check if there's an in-progress execution
        if trajectory.is_resumable():
            print(f"\n[WARNING] Found in-progress execution from {trajectory.started_at}")
            print(f"  Resume point: {trajectory.resume_point.step_name}")
            print("  (Use --no-trajectory to start fresh)")
            # For now, we don't implement resume - just start fresh

        trajectory.start(metadata={
            "input_file": str(input_path),
            "output_file": str(args.output),
            "max_iterations": args.max_iterations
        })

    try:
        # Build skeleton
        builder = SkeletonBuilder(
            max_iterations=args.max_iterations,
            trajectory=trajectory
        )

        result = builder.build(input_data)

        # Check for errors
        if "error" in result:
            logger.error(f"Build failed: {result['error']}")
            if trajectory:
                trajectory.fail(result["error"])
            return 1

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] Skeleton saved to: {output_path}")
        print(f"\n[OK] Skeleton saved to: {output_path.name}")

        # Save RPG as well
        if REPO_RPG_FILE.exists():
            print(f"[OK] RPG saved to: {REPO_RPG_FILE.name}")

        # Mark trajectory as complete
        if trajectory:
            trajectory.complete(metadata={
                "total_features": builder.stats["total_features"],
                "assigned_features": builder.stats["assigned_features"],
                "total_files": builder.stats["total_files"]
            })
            print(f"[OK] Trajectory saved to: {trajectory.trajectory_file.name}")

        return 0

    except Exception as e:
        logger.error(f"Build failed: {e}")
        if trajectory:
            trajectory.fail(str(e))
        raise


if __name__ == "__main__":
    exit(main())