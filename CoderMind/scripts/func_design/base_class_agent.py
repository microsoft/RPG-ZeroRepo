#!/usr/bin/env python3
"""Base Class Agent.

This module provides the BaseClassAgent for designing shared base classes
and data structures for the repository.

Key components:
- BaseClassAgent: Orchestrates base class generation with validation
- Syntax validation for generated Python code
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator

from .base_class_prompts import (
    BASE_CLASS_PROMPT,
    BASE_CLASS_REVIEW_PROMPT,
)

# Import common LLMClient with trajectory support
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import LLMClient
from decoder_lang import get_backend
from decoder_lang.backend import LanguageBackend
from decoder_lang.prompt_directive import with_language_directive


# ============================================================================
# Data Models
# ============================================================================

class BaseClassDefinition(BaseModel):
    """Definition of a base class or data structure."""
    file_path: str = Field(..., description="Path where this base class should be placed")
    code: str = Field(..., description="Full target-language code for the definition")
    scope: str = Field(..., description="Scope: 'global' or a specific subtree/component name")
    subclasses: Dict[str, List[str]] = Field(..., description="Mapping from base class name to list of concrete subclass names (each list must have at least 2 items)")
    
    @field_validator('subclasses')
    @classmethod
    def validate_subclasses(cls, v):
        if not v:
            raise ValueError("subclasses must be a non-empty dict")
        for base_name, sub_list in v.items():
            if not isinstance(sub_list, list) or len(sub_list) < 2:
                raise ValueError(f"Base class '{base_name}' must have at least 2 subclasses")
        return v


class DataStructureDefinition(BaseModel):
    """Definition of a shared data structure derived from data flow types.
    
    Unlike BaseClassDefinition, these do not require subclasses.
    They represent the concrete data types that flow between components
    as defined in data_flow.json.
    
    Note: file_path is NOT assigned here. It will be assigned later by
    the interface designer and written back to base_classes.json.
    """
    code: str = Field(..., description="Target-language data structure stub code")
    subtree: str = Field(..., description="The functional area / subtree this data structure belongs to (must be a valid subtree name, NOT 'global')")
    data_flow_types: List[str] = Field(..., min_length=1, description="Which data_flow data_type names this definition covers")
    file_path: str = Field(default="", description="File path assigned later by the interface designer. Leave empty during base class design.")


class BaseClassOutput(BaseModel):
    """Output from LLM for base class design."""
    base_classes: List[BaseClassDefinition] = Field(default_factory=list, description="List of base class definitions (may be empty for simple projects that don't need behavioral abstractions)")
    data_structures: List[DataStructureDefinition] = Field(default_factory=list, description="List of data flow data structure stubs (may be empty if all data types are already covered by base classes)")


# ============================================================================
# Validation Functions
# ============================================================================

def validate_base_classes_model(
    model: "BaseClassOutput",
    valid_subtrees: Optional[List[str]] = None,
    backend: Optional[LanguageBackend] = None,
) -> Tuple[bool, str]:
    """Validate base class definitions from a Pydantic model.
    
    Args:
        model: BaseClassOutput Pydantic model
        valid_subtrees: List of valid subtree/component names (from skeleton)
    
    Returns: (is_valid, error_message)
    """
    backend = backend or get_backend("python")
    # Build set of valid scope values
    valid_scopes = {"global"}
    if valid_subtrees:
        for st in valid_subtrees:
            valid_scopes.add(st)
    
    errors = []
    for i, bc in enumerate(model.base_classes):
        # Validate scope value - must be exact match
        if bc.scope not in valid_scopes:
            valid_list = sorted(valid_scopes)
            errors.append(
                f"Base class {i} ({bc.file_path}): invalid scope '{bc.scope}'. "
                f"Must be exactly one of: {valid_list}"
            )
            continue
        
        is_valid, error_msg = backend.syntax_check(bc.code, bc.file_path)
        if not is_valid:
            errors.append(f"Base class {i} ({bc.file_path}): syntax error - {error_msg}")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "All base classes are valid"


def validate_base_classes(
    base_classes: List[Dict[str, Any]],
    valid_subtrees: Optional[List[str]] = None,
    backend: Optional[LanguageBackend] = None,
) -> Tuple[bool, str]:
    """Validate base class definitions.
    
    Args:
        base_classes: List of base class definitions
        valid_subtrees: List of valid subtree/component names (from skeleton)
    
    Returns: (is_valid, error_message)
    """
    backend = backend or get_backend("python")
    if not base_classes:
        return False, "Empty base classes provided"
    
    # Build set of valid scope values
    valid_scopes = {"global"}
    if valid_subtrees:
        for st in valid_subtrees:
            valid_scopes.add(st)
    
    errors = []
    for i, bc in enumerate(base_classes):
        file_path = bc.get("file_path", "")
        code = bc.get("code", "")
        scope = bc.get("scope", "")
        subclasses = bc.get("subclasses", {})
        
        if not file_path:
            errors.append(f"Base class {i}: missing file_path")
            continue
        
        # Validate subclasses - must be a dict with each base class having at least 2 subclasses
        if not isinstance(subclasses, dict) or not subclasses:
            errors.append(f"Base class {i} ({file_path}): 'subclasses' must be a non-empty dict mapping base class names to subclass lists")
            continue
        
        for base_name, sub_list in subclasses.items():
            if not isinstance(sub_list, list) or len(sub_list) < 2:
                errors.append(f"Base class {i} ({file_path}): base class '{base_name}' must have at least 2 subclasses, got {len(sub_list) if isinstance(sub_list, list) else 0}")
                break
        
        if not code:
            errors.append(f"Base class {i} ({file_path}): missing code")
            continue
        
        # # Validate that subclasses keys cover all class definitions in code
        # defined_classes = set(extract_class_names(code))
        # declared_bases = set(subclasses.keys())
        # missing_bases = defined_classes - declared_bases
        # if missing_bases:
        #     errors.append(f"Base class {i} ({file_path}): 'subclasses' is missing entries for: {sorted(missing_bases)}")
        #     continue
        
        if not scope:
            errors.append(f"Base class {i} ({file_path}): missing scope (should be 'global' or a subtree name)")
            continue
        
        # Validate scope value - must be exact match
        if scope not in valid_scopes:
            valid_list = sorted(valid_scopes)
            errors.append(
                f"Base class {i} ({file_path}): invalid scope '{scope}'. "
                f"Must be exactly one of: {valid_list}"
            )
            continue
        
        is_valid, error_msg = backend.syntax_check(code, file_path)
        if not is_valid:
            errors.append(f"Base class {i} ({file_path}): syntax error - {error_msg}")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "All base classes are valid"


# ============================================================================
# Data Flow Type Extraction
# ============================================================================

def extract_data_flow_types(data_flow: List[Dict[str, Any]]) -> List[str]:
    """Extract unique data_type values from data flow edges.
    
    Args:
        data_flow: List of data flow edge dicts
        
    Returns:
        Sorted list of unique data_type strings
    """
    types = set()
    for edge in data_flow:
        dt = edge.get("data_type", "").strip()
        if dt:
            types.add(dt)
    return sorted(types)


def extract_declaration_names(code: str, backend: LanguageBackend) -> List[str]:
    """Extract top-level type or function names from target-language code."""
    names: List[str] = []
    for unit in backend.list_code_units(code):
        if getattr(unit, "parent", None) is not None:
            continue
        if getattr(unit, "unit_type", "") in {
            "class", "struct", "interface", "function",
        }:
            name = getattr(unit, "name", "")
            if name and name not in names:
                names.append(name)
    return names


def validate_data_structures(
    data_structures: List[Dict[str, Any]],
    data_flow_types: List[str],
    valid_subtrees: Optional[List[str]] = None,
    backend: Optional[LanguageBackend] = None,
) -> Tuple[bool, str]:
    """Validate data structure definitions.
    
    Note: file_path is NOT validated here — it is assigned later by the
    interface designer.
    
    Args:
        data_structures: List of data structure definitions
        data_flow_types: All unique data_type values from data flow
        valid_subtrees: List of valid subtree names
        
    Returns: (is_valid, error_message)
    """
    backend = backend or get_backend("python")
    # Build set of valid subtree values (no 'global' for data structures)
    valid_subtree_set = set()
    if valid_subtrees:
        for st in valid_subtrees:
            valid_subtree_set.add(st)
    
    errors = []
    covered_types = set()
    
    for i, ds in enumerate(data_structures):
        code = ds.get("code", "")
        subtree = ds.get("subtree", "")
        ds_types = ds.get("data_flow_types", [])
        
        if not code:
            errors.append(f"Data structure {i}: missing code")
            continue
        
        if not subtree:
            errors.append(f"Data structure {i}: missing subtree")
            continue
        
        if subtree.lower() == "global":
            errors.append(
                f"Data structure {i}: subtree cannot be 'global'. "
                f"Data structures must belong to a specific subtree."
            )
            continue
        
        if valid_subtree_set and subtree not in valid_subtree_set:
            valid_list = sorted(valid_subtree_set)
            errors.append(
                f"Data structure {i}: invalid subtree '{subtree}'. "
                f"Must be exactly one of: {valid_list}"
            )
            continue
        
        if not ds_types:
            errors.append(f"Data structure {i}: data_flow_types must not be empty")
            continue
        
        is_valid, error_msg = backend.syntax_check(code, f"data_structure{backend.file_extension}")
        if not is_valid:
            errors.append(f"Data structure {i} (subtree={subtree}): syntax error - {error_msg}")
        
        covered_types.update(ds_types)
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "All data structures are valid"


# ============================================================================
# Base Class Agent
# ============================================================================

class BaseClassAgent:
    """Agent for designing shared base classes and data structures."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 5,
        logger: Optional[logging.Logger] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None,
        target_language: Optional[str] = None,
    ):
        # Create LLMClient with trajectory support if not provided
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
            # Update trajectory info on existing client
            if trajectory is not None:
                self.llm.set_trajectory(trajectory, step_id)
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_backend(target_language)
    
    def design_base_classes(
        self,
        repo_name: str,
        repo_info: str,
        data_flow: List[Dict[str, Any]],
        skeleton_tree: str,
        functional_areas: List[str],
        functional_areas_overview: str = "",
        project_background: str = "",
    ) -> Dict[str, Any]:
        """Design base classes for the repository.
        
        Args:
            repo_name: Repository name
            repo_info: Repository description
            data_flow: Data flow edges from previous step
            skeleton_tree: Tree string of skeleton
            functional_areas: List of functional area names
            functional_areas_overview: Hierarchical overview of functional areas with sub-features
            project_background: Project background/technology context from feature_spec
            
        Returns:
            Dict containing:
            - base_classes: List of base class definitions
            - success: Whether the operation succeeded
        """
        self.logger.info(f"[BaseClassAgent] Designing base classes for {repo_name}")

        # Build system prompt (tool description is now integrated)
        system_prompt = with_language_directive(BASE_CLASS_PROMPT, self.backend)

        # Extract unique data_type values from data flow (for post-validation)
        data_flow_type_names = extract_data_flow_types(data_flow)
        
        # Format data flow as raw JSON for full context
        data_flow_json_str = json.dumps(data_flow, indent=2, ensure_ascii=False)
        
        # Use hierarchical overview if available, otherwise fall back to flat list
        if functional_areas_overview:
            areas_section = functional_areas_overview
        else:
            areas_section = "Functional Areas: " + ", ".join(functional_areas)
        
        # Build user prompt
        # Include project background when available — gives the LLM context
        # about technology stack so it can design framework-appropriate base classes.
        technology_section = ""
        if project_background and project_background.strip():
            technology_section = f"""
{project_background}
When the project specifies a concrete technology stack (framework, database, etc.),
design base classes that are idiomatic for those technologies rather than purely
abstract. For example, if the project uses Flask, prefer Flask Blueprint patterns
over generic abstract request handlers.  If no specific technology is mentioned,
            use the target language's idiomatic abstraction mechanism.
"""

        hints = self.backend.prompt_hints()

        user_prompt = f"""Based on the repository structure and data flow, generate base class definitions:
Repository Name: {repo_name}
Repository Info: {repo_info}
{technology_section}
Repository Skeleton:
{skeleton_tree}

Functional Areas Overview:
{areas_section}

(Use the exact top-level component names above as scope/subtree values, NOT directory paths.)

Data Flow (JSON):
{data_flow_json_str}

Please use the generate_base_classes tool to create base class definitions and data structure stubs.

Focus on:
1. Shared behavioral abstractions (base classes with abstract methods)
2. Common data structures that flow between components
3. Keep it minimal - only create abstractions that will be reused by multiple components
4. Use idiomatic {hints.display_name} constructs for data structures and behavioral abstractions

Additionally, for data_structures:
- Data flow types that are generic enough to serve as base classes (with subclasses) should go into base_classes, not data_structures
- The remaining data flow types that are NOT absorbed by base classes should be defined as data_structures
- Use idiomatic {hints.display_name} data containers with explicit fields and documentation
- These are stubs (skeleton code) — they will be fully implemented later
- Each data structure must belong to a specific subtree (not global)
- Do NOT specify file_path — it will be assigned by the interface designer later"""
        
        # Iterate until valid or max iterations
        last_error = ""
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"[BaseClassAgent] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Build prompt with error feedback if needed
            current_user_prompt = user_prompt
            if last_error:
                current_user_prompt += f"\n\n[Validation Failed]\nError: {last_error}\nPlease fix the issues and try again."
            
            # Call LLM with Pydantic validation
            _, result_model, _ = self.llm.call_structured(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                response_model=BaseClassOutput,
                purpose=f"base_class_design_{iteration + 1}",
                max_retries=1  # Handle retries at this level
            )
            
            if not result_model:
                last_error = "Failed to parse LLM response or Pydantic validation failed."
                continue
            
            # Convert to dict list for custom validation
            base_classes = [bc.model_dump() for bc in result_model.base_classes]
            data_structures = [ds.model_dump() for ds in result_model.data_structures]
            
            # Custom validation (scope and syntax) for base classes
            is_valid, error_msg = validate_base_classes_model(
                result_model,
                valid_subtrees=functional_areas,
                backend=self.backend,
            )
            
            if not is_valid:
                self.logger.warning(f"[BaseClassAgent] Base class validation failed: {error_msg}")
                last_error = error_msg
                continue
            
            # Validate data structures
            ds_valid, ds_error = validate_data_structures(
                data_structures,
                data_flow_type_names,
                valid_subtrees=functional_areas,
                backend=self.backend,
            )
            
            if not ds_valid:
                self.logger.warning(f"[BaseClassAgent] Data structure validation failed: {ds_error}")
                last_error = ds_error
                continue
            
            # Extract class names for logging
            all_classes = []
            for bc in base_classes:
                all_classes.extend(extract_declaration_names(bc.get("code", ""), self.backend))
            
            # Extract data structure class names
            ds_class_names = []
            for ds in data_structures:
                ds_class_names.extend(extract_declaration_names(ds.get("code", ""), self.backend))
            
            # Check data_flow_type coverage (base_classes code may also cover some types)
            bc_class_set = set(all_classes)
            ds_covered_types = set()
            for ds in data_structures:
                ds_covered_types.update(ds.get("data_flow_types", []))
            uncovered = set(data_flow_type_names) - ds_covered_types - bc_class_set
            
            self.logger.info(
                f"[BaseClassAgent] Validated: {len(base_classes)} base classes, "
                f"{len(data_structures)} data structures, "
                f"{len(uncovered)} uncovered data flow types"
            )
            if uncovered:
                self.logger.warning(f"[BaseClassAgent] Uncovered data flow types: {sorted(uncovered)}")
            
            return {
                "base_classes": base_classes,
                "data_structures": data_structures,
                "class_names": all_classes,
                "data_structure_names": ds_class_names,
                "uncovered_data_flow_types": sorted(uncovered),
                "success": True,
                "iterations": iteration + 1
            }
        
        # Failed after all iterations
        self.logger.error(f"[BaseClassAgent] Failed after {self.max_iterations} iterations")
        return {
            "base_classes": [],
            "data_structures": [],
            "success": False,
            "error": last_error,
            "iterations": self.max_iterations
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    agent = BaseClassAgent()
    result = agent.design_base_classes(
        repo_name="test-repo",
        repo_info="A test repository",
        data_flow=[
            {"source": "A", "target": "B", "data_type": "Data"}
        ],
        skeleton_tree="src/\n  module/\n    file.py",
        functional_areas=["A", "B"]
    )
    print(json.dumps(result, indent=2))
