#!/usr/bin/env python3
"""Interface Agent.

This module provides the InterfaceAgent for designing function/class interfaces
for each file in the repository skeleton.

Key components:
- InterfaceAgent: Orchestrates interface design for a single file
- InterfaceOrchestrator: Manages the full interface design workflow across subtrees
- Validation functions for interface code
"""

import json
import logging
import ast
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from pydantic import BaseModel, Field

# Import ParsedFile and CodeUnit for code parsing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rpg.code_unit import ParsedFile, CodeUnit

# Import common LLMClient with trajectory support
from common import (
    LLMClient,
    format_data_flow_edges,
    format_base_classes,
    format_base_classes_and_data_structures,
)

from .interface_prompts import (
    INTERFACE_PROMPT,
    PLAN_FILE_PROMPT,
    SUBTREE_INTERFACE_PROMPT,
)
from common.import_normalizer import build_import_convention_snippet


# ============================================================================
# Data Models
# ============================================================================

class InterfaceDependency(BaseModel):
    """Dependency information for an interface."""
    inherits_from: List[str] = Field(default_factory=list, description="Base classes inherited")
    calls: List[str] = Field(default_factory=list, description="Functions/methods expected to call")
    uses_types: List[str] = Field(default_factory=list, description="Types used in parameters/returns")


class InterfaceDefinition(BaseModel):
    """Definition of a single interface."""
    features: List[str] = Field(default_factory=list, description="List of feature paths this interface handles (both existing and new)")
    code: str = Field(..., description="Python code for the interface")
    dependencies: Optional[InterfaceDependency] = Field(default=None, description="Declared dependencies")


class InterfaceOutput(BaseModel):
    """Output from LLM for interface design."""
    interfaces: List[InterfaceDefinition] = Field(..., min_length=1, description="List of interface definitions (must not be empty)")


class FileInterfaceBlock(BaseModel):
    """Block of interface definitions for a single file within a subtree batch."""
    file_path: str = Field(..., description="Path to the file being designed")
    interfaces: List[InterfaceDefinition] = Field(..., min_length=1, description="Interface definitions for this file")


class SubtreeInterfaceOutput(BaseModel):
    """Output from LLM for subtree-level interface design (all files at once)."""
    files: List[FileInterfaceBlock] = Field(..., min_length=1, description="Interface blocks organized by file, in implementation order")


class FileImplementationGraph(BaseModel):
    """Graph of file implementation order."""
    file_implementation_graph: List[Dict[str, str]] = Field(default_factory=list)


# ============================================================================
# Dependency Collector
# ============================================================================

class DependencyCollector:
    """Collect fine-grained dependencies discovered during interface design.
    
    Dependencies are collected from two sources:
    1. Program analysis (AST parsing) - inheritance and type references from code
    2. LLM declarations - expected function calls declared by LLM
    """
    
    def __init__(self, known_base_classes: Set[str], known_types: Set[str]):
        """Initialize the dependency collector.
        
        Args:
            known_base_classes: Set of base class names from base_classes.json
            known_types: Set of known type names (data structures, etc.)
        """
        self.known_base_classes = known_base_classes
        self.known_types = known_types
        self.original_edges: List[Dict[str, Any]] = []
        self.inheritance_edges: List[Dict[str, Any]] = []
        self.invocation_edges: List[Dict[str, Any]] = []
        self.reference_edges: List[Dict[str, Any]] = []
    
    def set_original_edges(self, edges: List[Dict[str, Any]]):
        """Store original coarse-grained data flow edges."""
        self.original_edges = edges.copy() if edges else []
    
    def add_inheritance(
        self,
        child_class: str,
        parent_class: str,
        source_file: str,
        parent_file: Optional[str] = None
    ):
        """Add an inheritance relationship (child extends parent)."""
        self.inheritance_edges.append({
            "child": child_class,
            "parent": parent_class,
            "source_file": source_file,
            "parent_file": parent_file,
            "edge_type": "inherits",
            "generator": "design_interfaces"
        })
    
    def add_invocation(
        self,
        caller: str,
        callee: str,
        caller_file: str,
        callee_file: Optional[str] = None
    ):
        """Add an invocation relationship (caller calls callee).
        
        Self-calls (same bare name + same or unknown file) are silently skipped.
        """
        # --- self-call filter ---
        bare_caller = caller.split(" ", 1)[-1] if " " in caller else caller
        bare_callee = callee.split(" ", 1)[-1] if " " in callee else callee
        if bare_caller == bare_callee and (callee_file is None or callee_file == caller_file):
            return

        self.invocation_edges.append({
            "caller": caller,
            "callee": callee,
            "caller_file": caller_file,
            "callee_file": callee_file,
            "edge_type": "invokes",
            "generator": "design_interfaces"
        })
    
    def add_reference(
        self,
        unit_name: str,
        referenced_type: str,
        source_file: str,
        type_file: Optional[str] = None
    ):
        """Add a type reference relationship."""
        self.reference_edges.append({
            "unit": unit_name,
            "referenced_type": referenced_type,
            "source_file": source_file,
            "type_file": type_file,
            "edge_type": "references",
            "generator": "design_interfaces"
        })
    
    def analyze_code_dependencies(
        self,
        code: str,
        file_path: str,
        base_class_files: Dict[str, str]
    ):
        """Analyze code to extract dependencies via AST parsing.
        
        Extracts:
        - Inheritance relationships (class X(BaseClass))
        - Type references in annotations
        
        Args:
            code: Python source code to analyze
            file_path: Path of the file containing this code
            base_class_files: Mapping of class names to their file paths
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        
        for node in ast.walk(tree):
            # Extract inheritance
            if isinstance(node, ast.ClassDef):
                child_class = node.name
                for base in node.bases:
                    parent_name = _extract_name_from_node(base)
                    if parent_name and parent_name in self.known_base_classes:
                        parent_file = base_class_files.get(parent_name)
                        self.add_inheritance(child_class, parent_name, file_path, parent_file)
            
            # Extract type references from function annotations
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                # Check parameter types
                for arg in node.args.args:
                    if arg.annotation:
                        types = _extract_type_names(arg.annotation)
                        for t in types:
                            if t in self.known_types:
                                type_file = base_class_files.get(t)
                                self.add_reference(f"function {func_name}", t, file_path, type_file)
                # Check return type
                if node.returns:
                    types = _extract_type_names(node.returns)
                    for t in types:
                        if t in self.known_types:
                            type_file = base_class_files.get(t)
                            self.add_reference(f"function {func_name}", t, file_path, type_file)
    
    def process_llm_dependencies(
        self,
        unit_name: str,
        dependencies: Optional[Dict[str, Any]],
        file_path: str,
        base_class_files: Dict[str, str]
    ):
        """Process dependencies declared by LLM.
        
        Args:
            unit_name: Name of the interface (e.g., "class Foo" or "function bar")
            dependencies: Dependencies dict from LLM with inherits_from, calls, uses_types
            file_path: Path of the file containing this interface
            base_class_files: Mapping of class/function names to their file paths
        """
        if not dependencies:
            return
        
        # Process calls (LLM-declared invocations)
        for callee in dependencies.get("calls", []):
            callee_file = base_class_files.get(callee)
            self.add_invocation(unit_name, callee, file_path, callee_file)
        
        # Note: inherits_from and uses_types are also analyzed from code,
        # but LLM declarations can catch additional cases not in annotations
        for parent in dependencies.get("inherits_from", []):
            if parent in self.known_base_classes:
                # Check if not already added by code analysis
                existing = [e for e in self.inheritance_edges 
                           if e["child"] in unit_name and e["parent"] == parent]
                if not existing:
                    parent_file = base_class_files.get(parent)
                    # Extract class name from unit_name like "class Foo"
                    class_name = unit_name.replace("class ", "") if unit_name.startswith("class ") else unit_name
                    self.add_inheritance(class_name, parent, file_path, parent_file)
        
        for type_name in dependencies.get("uses_types", []):
            if type_name in self.known_types:
                # Check if not already added by code analysis
                existing = [e for e in self.reference_edges 
                           if e["unit"] == unit_name and e["referenced_type"] == type_name]
                if not existing:
                    type_file = base_class_files.get(type_name)
                    self.add_reference(unit_name, type_name, file_path, type_file)
    
    def post_process_edges(self, global_registry: "GlobalInterfaceRegistry"):
        """Normalise invocation edges after all subtrees have been designed.

        For each invocation edge:
        1. Resolve bare callee names to their full unit name
           (``"function foo"`` / ``"class Bar"``).
        2. Handle ``Class.method`` patterns → resolve to ``"class Class"``.
        3. Fill in missing ``callee_file`` via *global_registry*.
        4. Drop edges whose callee cannot be resolved at all.
        """
        if not global_registry:
            return

        cleaned: List[Dict[str, Any]] = []
        for edge in self.invocation_edges:
            callee = edge["callee"]
            callee_file = edge.get("callee_file")

            # --- 1. Handle "Class.method" patterns ---
            if "." in callee:
                class_name = callee.split(".")[0]
                resolved_file = callee_file or global_registry.resolve_callee(class_name)
                if resolved_file:
                    edge["callee"] = f"class {class_name}"
                    edge["callee_file"] = resolved_file
                    cleaned.append(edge)
                continue  # skip unresolvable Class.method

            # --- 2. Normalise bare name → "function X" / "class X" ---
            if not callee.startswith("function ") and not callee.startswith("class "):
                # Check registry for the canonical unit name
                unit_info = global_registry.units.get(f"function {callee}") or \
                            global_registry.units.get(f"class {callee}")
                if unit_info:
                    edge["callee"] = f"{unit_info['unit_type']} {callee}"
                    if not callee_file:
                        edge["callee_file"] = unit_info["file_path"]
                elif callee in global_registry.function_to_file:
                    edge["callee"] = f"function {callee}"
                    if not callee_file:
                        edge["callee_file"] = global_registry.function_to_file[callee]
                elif callee in global_registry.class_to_file:
                    edge["callee"] = f"class {callee}"
                    if not callee_file:
                        edge["callee_file"] = global_registry.class_to_file[callee]
                # else: keep bare name as-is (external or unresolvable)

            # --- 3. Fill missing callee_file ---
            if not edge.get("callee_file"):
                bare = edge["callee"]
                if bare.startswith("function ") or bare.startswith("class "):
                    bare = bare.split(" ", 1)[1]
                resolved = global_registry.resolve_callee(bare)
                if resolved:
                    edge["callee_file"] = resolved

            cleaned.append(edge)

        self.invocation_edges = cleaned

    def to_dict(self) -> Dict[str, Any]:
        """Convert collected dependencies to dictionary."""
        return {
            "original_edges": self.original_edges,
            "inheritance_edges": self.inheritance_edges,
            "invocation_edges": self.invocation_edges,
            "reference_edges": self.reference_edges
        }
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary counts of collected dependencies."""
        return {
            "original_edges": len(self.original_edges),
            "inheritance_edges": len(self.inheritance_edges),
            "invocation_edges": len(self.invocation_edges),
            "reference_edges": len(self.reference_edges)
        }


def _extract_name_from_node(node: ast.expr) -> Optional[str]:
    """Extract name string from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    return None


def _extract_type_names(node: ast.expr) -> List[str]:
    """Extract all type names from a type annotation AST node."""
    names = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Attribute):
        names.append(node.attr)
    elif isinstance(node, ast.Subscript):
        # Handle generic types like List[X], Optional[X], etc.
        names.extend(_extract_type_names(node.slice))
        if isinstance(node.value, ast.Name):
            names.append(node.value.id)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            names.extend(_extract_type_names(elt))
    elif isinstance(node, ast.BinOp):
        # Handle Union types with | operator (Python 3.10+)
        names.extend(_extract_type_names(node.left))
        names.extend(_extract_type_names(node.right))
    return names


# ============================================================================
# Global Interface Registry
# ============================================================================

class GlobalInterfaceRegistry:
    """Track all designed interfaces across subtrees for cross-subtree dependency resolution.
    
    As each subtree is designed, its interfaces are registered here.
    Later subtrees can use this registry to resolve callee names to file paths,
    enabling accurate cross-subtree dependency edges.
    """
    
    def __init__(self):
        # unit_name -> {file_path, subtree_name, unit_type, signature_summary, features}
        self.units: Dict[str, Dict[str, Any]] = {}
        # class_name -> file_path (for quick lookup)
        self.class_to_file: Dict[str, str] = {}
        # function_name -> file_path
        self.function_to_file: Dict[str, str] = {}
        # file_path -> list of unit info dicts
        self.file_units: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_from_subtree_result(
        self,
        subtree_name: str,
        subtree_interfaces: Dict[str, Dict[str, Any]]
    ):
        """Register all designed interfaces from a completed subtree.
        
        Args:
            subtree_name: Name of the subtree
            subtree_interfaces: Dict mapping file_path -> file result dict
                (with keys: units, units_to_features, units_to_code, file_code)
        """
        for file_path, file_data in subtree_interfaces.items():
            units = file_data.get("units", [])
            units_to_features = file_data.get("units_to_features", {})
            units_to_code = file_data.get("units_to_code", {})
            
            file_unit_list = []
            
            for unit_name in units:
                features = units_to_features.get(unit_name, [])
                code = units_to_code.get(unit_name, "")
                
                # Determine unit type and bare name
                if unit_name.startswith("class "):
                    unit_type = "class"
                    bare_name = unit_name[len("class "):]
                    self.class_to_file[bare_name] = file_path
                elif unit_name.startswith("function "):
                    unit_type = "function"
                    bare_name = unit_name[len("function "):]
                    self.function_to_file[bare_name] = file_path
                else:
                    unit_type = "unknown"
                    bare_name = unit_name
                
                # Extract a signature summary from the code (first non-import, non-blank line)
                signature_summary = self._extract_signature_summary(code, unit_type, bare_name)
                
                unit_info = {
                    "file_path": file_path,
                    "subtree_name": subtree_name,
                    "unit_type": unit_type,
                    "bare_name": bare_name,
                    "signature_summary": signature_summary,
                    "features": features,
                }
                
                self.units[unit_name] = unit_info
                file_unit_list.append(unit_info)
            
            if file_unit_list:
                if file_path not in self.file_units:
                    self.file_units[file_path] = []
                self.file_units[file_path].extend(file_unit_list)
    
    def resolve_callee(self, callee_name: str) -> Optional[str]:
        """Resolve a callee name to its file_path across all registered subtrees.
        
        Tries:
        1. Exact match in class_to_file
        2. Exact match in function_to_file
        3. Fuzzy match (case-insensitive) in both
        
        Returns:
            file_path if found, None otherwise
        """
        # Exact match
        if callee_name in self.class_to_file:
            return self.class_to_file[callee_name]
        if callee_name in self.function_to_file:
            return self.function_to_file[callee_name]
        
        # Try with "class " or "function " prefix stripped
        stripped = callee_name
        if callee_name.startswith("class "):
            stripped = callee_name[len("class "):]
        elif callee_name.startswith("function "):
            stripped = callee_name[len("function "):]
        
        if stripped != callee_name:
            if stripped in self.class_to_file:
                return self.class_to_file[stripped]
            if stripped in self.function_to_file:
                return self.function_to_file[stripped]
        
        # Case-insensitive fallback
        callee_lower = callee_name.lower()
        for name, path in self.class_to_file.items():
            if name.lower() == callee_lower:
                return path
        for name, path in self.function_to_file.items():
            if name.lower() == callee_lower:
                return path
        
        return None
    
    def get_all_public_symbols(self) -> Dict[str, str]:
        """Return {symbol_name: file_path} for all registered public symbols.
        
        This can be merged into base_class_files to enable cross-subtree
        dependency resolution.
        """
        symbols = {}
        symbols.update(self.class_to_file)
        symbols.update(self.function_to_file)
        return symbols
    
    def get_structured_interface_listing(self, subtree_name: str) -> str:
        """Build a structured interface listing for a specific subtree, suitable for inclusion in upstream context prompts.
        
        Returns a formatted string like:
          From "Physics Engine Core":
            - src/physics/forces.py:
              - function calculate_gravity(mass1: float, ...) -> Vector2D
              - function calculate_drag(...)
            - src/physics/dynamics.py:
              - class DynamicsEngine:
                - method step(particles: List[Particle], dt: float) -> None
        """
        parts = []
        
        # Group file_units by file_path for this subtree
        subtree_files: Dict[str, List[Dict[str, Any]]] = {}
        for file_path, unit_list in self.file_units.items():
            for unit_info in unit_list:
                if unit_info["subtree_name"] == subtree_name:
                    if file_path not in subtree_files:
                        subtree_files[file_path] = []
                    subtree_files[file_path].append(unit_info)
        
        if not subtree_files:
            return ""
        
        parts.append(f'From "{subtree_name}":')
        for file_path in sorted(subtree_files.keys()):
            parts.append(f"  - {file_path}:")
            for unit_info in subtree_files[file_path]:
                sig = unit_info.get("signature_summary", unit_info["bare_name"])
                parts.append(f"    - {unit_info['unit_type']} {sig}")
        
        return "\n".join(parts)
    
    def get_all_structured_listings_for_upstream(
        self,
        upstream_subtree_names: Set[str]
    ) -> str:
        """Build structured interface listings for all upstream subtrees.
        
        Args:
            upstream_subtree_names: Set of subtree names to include
            
        Returns:
            Formatted string with all upstream interface listings
        """
        listings = []
        for subtree_name in sorted(upstream_subtree_names):
            listing = self.get_structured_interface_listing(subtree_name)
            if listing:
                listings.append(listing)
        
        if not listings:
            return "No upstream interfaces available."
        
        return "\n\n".join(listings)
    
    @staticmethod
    def _extract_signature_summary(code: str, unit_type: str, bare_name: str) -> str:
        """Extract a concise signature summary from interface code."""
        if not code:
            return bare_name
        
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if unit_type == "class" and isinstance(node, ast.ClassDef) and node.name == bare_name:
                    # For classes, list public methods with signatures
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not item.name.startswith("_") or item.name == "__init__":
                                sig = GlobalInterfaceRegistry._format_func_signature(item)
                                methods.append(sig)
                    bases_str = ""
                    if node.bases:
                        bases = [_extract_name_from_node(b) for b in node.bases]
                        bases = [b for b in bases if b]
                        if bases:
                            bases_str = f"({', '.join(bases)})"
                    if methods:
                        return f"{bare_name}{bases_str} [{', '.join(methods[:5])}]"
                    return f"{bare_name}{bases_str}"
                    
                elif unit_type == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == bare_name:
                    return GlobalInterfaceRegistry._format_func_signature(node)
        except SyntaxError:
            pass
        
        return bare_name
    
    @staticmethod
    def _format_func_signature(node) -> str:
        """Format a function/method AST node into a concise signature string."""
        name = node.name
        params = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            param_str = arg.arg
            if arg.annotation:
                type_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ""
                if type_str:
                    param_str = f"{arg.arg}: {type_str}"
            params.append(param_str)
        
        ret_str = ""
        if node.returns:
            ret_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else ""
            if ret_type:
                ret_str = f" -> {ret_type}"
        
        # Truncate params if too many
        if len(params) > 4:
            params_str = ", ".join(params[:3]) + ", ..."
        else:
            params_str = ", ".join(params)
        
        return f"{name}({params_str}){ret_str}"


# ============================================================================
# Import Cross-Validation (A2)
# ============================================================================

def cross_validate_imports_vs_calls(
    code: str,
    file_path: str,
    declared_calls: List[str],
    global_registry: GlobalInterfaceRegistry
) -> List[Dict[str, str]]:
    """Parse import statements in interface code and cross-validate against declared calls. Identifies symbols that are imported from modules in the global registry but not declared as call dependencies.
    
    This is an auxiliary validation — results are warnings, not auto-added edges.
    
    Args:
        code: Interface source code (signatures + imports only)
        file_path: Path of the file being validated
        declared_calls: List of callee names from LLM's dependencies.calls
        global_registry: Registry of all designed interfaces
        
    Returns:
        List of warning dicts: {imported_symbol, imported_from, resolved_file, file_path}
    """
    warnings = []
    declared_set = set(declared_calls)
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return warnings
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                symbol = alias.name
                # Check if this symbol is in the global registry
                resolved_file = global_registry.resolve_callee(symbol)
                if resolved_file and resolved_file != file_path:
                    # Symbol is a known interface from another file
                    if symbol not in declared_set:
                        warnings.append({
                            "imported_symbol": symbol,
                            "imported_from": module,
                            "resolved_file": resolved_file,
                            "file_path": file_path,
                            "message": (
                                f"'{symbol}' is imported from '{module}' and is a known "
                                f"interface in '{resolved_file}', but not declared in "
                                f"dependencies.calls"
                            )
                        })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbol = alias.name.split(".")[-1] if "." in alias.name else alias.name
                resolved_file = global_registry.resolve_callee(symbol)
                if resolved_file and resolved_file != file_path:
                    if symbol not in declared_set:
                        warnings.append({
                            "imported_symbol": symbol,
                            "imported_from": alias.name,
                            "resolved_file": resolved_file,
                            "file_path": file_path,
                            "message": (
                                f"'{symbol}' is imported and is a known interface in "
                                f"'{resolved_file}', but not declared in dependencies.calls"
                            )
                        })
    
    return warnings


# ============================================================================
# Validation Functions
# ============================================================================

def extract_top_level_definitions(code: str) -> Tuple[List[str], List[str]]:
    """Extract top-level function and class names from code."""
    functions = []
    classes = []
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except SyntaxError:
        pass
    return functions, classes


def check_has_docstring(code: str) -> Tuple[bool, str]:
    """Check if top-level functions/classes have docstrings."""
    errors = []
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    errors.append(f"{type(node).__name__} '{node.name}' is missing a docstring")
    except SyntaxError:
        pass
    
    if errors:
        return False, "; ".join(errors)
    return True, ""


def validate_interface(
    interface: Dict[str, Any],
    target_features: Set[str],
    covered_features: Set[str]
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a single interface definition using ParsedFile.
    
    Returns: (is_valid, error_message, parsed_info)
    """
    features = interface.get("features", [])
    code = interface.get("code", "")
    errors = []
    
    # Check features
    if not features:
        errors.append("Interface must have at least one feature")
    else:
        feature_set = set(features)
        
        # Check for overlap with already covered features
        overlap = feature_set & covered_features
        if overlap:
            errors.append(f"Features {list(overlap)} are already covered by another interface")
        
        # Check if features are in target features
        if target_features:
            invalid_features = feature_set - target_features
            if invalid_features:
                errors.append(f"Features {list(invalid_features)} are not in target features")
    
    # Auto-fix hyphenated module names in import statements
    # (e.g., "from blog-system.security import ..." -> "from blog_system.security import ...")
    code = re.sub(
        r'^(\s*(?:from|import)\s+)([\w\-]+(?:\.[\w\-]+)*)',
        lambda m: m.group(1) + m.group(2).replace('-', '_'),
        code,
        flags=re.MULTILINE,
    )
    # Persist the fixed code back so downstream consumers get corrected imports
    interface["code"] = code

    # Parse code with ParsedFile
    parsed_file = ParsedFile(code=code, file_path="temp_interface.py")
    
    # Check for syntax errors
    if parsed_file.has_error():
        error = parsed_file.error
        errors.append(f"Syntax error: line {error.lineno}, column {error.offset}: {error.msg}")
        return False, "; ".join(errors), {}
    
    # Extract only class and function units (not methods)
    interface_units = [
        unit for unit in parsed_file.units
        if unit.unit_type in ["function", "class"]
    ]
    
    if not interface_units:
        errors.append("No valid functions/classes found in code")
    
    # Check docstrings
    for unit in interface_units:
        if not unit.docstring and unit.unit_type in ["function", "class"]:
            errors.append(
                f"Missing docstring for {unit.unit_type} '{unit.name}' "
                f"in features {features}"
            )
    
    if errors:
        return False, "; ".join(errors), {}
    
    # Build parsed info with CodeUnit objects
    functions = [u.name for u in interface_units if u.unit_type == "function"]
    classes = [u.name for u in interface_units if u.unit_type == "class"]
    
    return True, "", {
        "functions": functions,
        "classes": classes,
        "features": features,
        "units": interface_units  # Include CodeUnit objects
    }


def validate_file_implementation_graph(
    graph: List[Dict[str, str]],
    file_names: List[str]
) -> Tuple[str, bool]:
    """Validate file implementation graph.
    
    Returns: (feedback_message, is_valid)
    """
    file_set = set(file_names)
    feedbacks = []
    is_valid = True
    
    # Check all files are valid
    for edge in graph:
        from_f = edge.get("from", "")
        to_f = edge.get("to", "")
        
        if from_f not in file_set:
            feedbacks.append(f"Invalid file reference: `{from_f}` is not in the file list.")
            is_valid = False
        if to_f not in file_set:
            feedbacks.append(f"Invalid file reference: `{to_f}` is not in the file list.")
            is_valid = False
    
    if feedbacks:
        feedbacks.append("Please ensure all file references are from the provided file list.")
    
    # Check for cycles
    adj = defaultdict(list)
    indegree = defaultdict(int)
    for edge in graph:
        f, t = edge.get("from", ""), edge.get("to", "")
        adj[f].append(t)
        indegree[t] += 1
    
    queue = deque([f for f in file_set if indegree[f] == 0])
    visited = set()
    
    while queue:
        node = queue.popleft()
        visited.add(node)
        for neighbor in adj.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(visited) != len(file_set):
        feedbacks.append(
            "Cycle detected or not all files are connected. "
            "The graph must form a valid DAG (Directed Acyclic Graph)."
        )
        is_valid = False
    
    # Check coverage
    used_files = {e.get("from", "") for e in graph} | {e.get("to", "") for e in graph}
    missing = file_set - used_files
    if missing:
        feedbacks.append(
            f"Missing files: {sorted(missing)}. Please include all files in the graph."
        )
        is_valid = False
    
    return "\n".join(feedbacks) if feedbacks else "Valid graph", is_valid


def topo_sort_file_graph(graph: List[Dict[str, str]]) -> Optional[List[str]]:
    """Topologically sort file graph. Returns None if cycle detected."""
    adj = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()
    
    for edge in graph:
        from_f = edge.get("from", "")
        to_f = edge.get("to", "")
        adj[from_f].append(to_f)
        indegree[to_f] += 1
        nodes.add(from_f)
        nodes.add(to_f)
    
    # Initialize indegree for source nodes
    for node in nodes:
        if node not in indegree:
            indegree[node] = 0
    
    queue = deque([n for n in nodes if indegree[n] == 0])
    sorted_list = []
    
    while queue:
        node = queue.popleft()
        sorted_list.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(sorted_list) != len(nodes):
        return None
    
    return sorted_list


# ============================================================================
# Interface Agent (Single File)
# ============================================================================

class InterfaceAgent:
    """Agent for designing interfaces for a single file."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 10,
        logger: Optional[logging.Logger] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None
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
    
    def design_file_interface(
        self,
        file_path: str,
        file_features: List[str],
        repo_info: str,
        data_flow_str: str,
        base_classes_str: str,
        upstream_context: str,
        implemented_summary: str,
        dependency_collector: Optional[DependencyCollector] = None,
        base_class_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Design interfaces for a single file.
        
        Args:
            file_path: Path to the file being designed
            file_features: List of feature paths for this file
            repo_info: Repository description
            data_flow_str: Formatted data flow string
            base_classes_str: Formatted base classes string
            upstream_context: Context from upstream modules
            implemented_summary: Summary of already implemented files
            dependency_collector: Optional collector for fine-grained dependencies
            base_class_files: Optional mapping of class/type names to file paths
            
        Returns:
            Dict containing interfaces, code, feature_map, success
        """
        self.logger.info(f"[InterfaceAgent] Designing interfaces for {file_path}")
        
        target_features = set(file_features)
        covered_features = set()
        all_interfaces = []
        all_code_blocks = []
        feature_interface_map = {}

        # Build system prompt (tool description is now integrated)
        system_prompt = INTERFACE_PROMPT

        # Build user prompt
        features_str = "\n".join([f"- {f}" for f in file_features])
        
        user_prompt = f"""[Begin Iteration]
Design interfaces for file: `{file_path}`.

Requirements:
- ONLY cover the following feature paths:
{features_str}
- When calling `design_itfs_for_feature`, ONLY use feature paths listed above.
- Do NOT introduce new/unspecified feature paths.
- Define interfaces only (imports + signature + docstring + `pass`).
- Prefer one function/class per feature or a small group of closely related features.
- Keep each interface focused and with narrow responsibility.
- You MAY import and reuse symbols from upstream context and base classes.

Global context you can use:
=== Repository Info ===
{repo_info}

=== Data Flow Graph ===
{data_flow_str}

=== Upstream Context ===
{upstream_context}

=== Implemented Summary ===
{implemented_summary}

=== Available Base Classes ===
{base_classes_str}
"""
        
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        last_error = ""
        
        for iteration in range(self.max_iterations):
            # Check if all features covered
            if covered_features >= target_features:
                self.logger.info(f"[InterfaceAgent] All features covered for {file_path}")
                break
            
            self.logger.info(f"[InterfaceAgent] Iteration {iteration + 1}/{self.max_iterations} for {file_path}")
            
            current_prompt = combined_prompt
            if last_error:
                current_prompt += f"\n\n[Tool Execution Feedback]: {last_error}"
            
            # Add progress info
            remaining = target_features - covered_features
            if covered_features:
                current_prompt += f"\n\n[Progress]: Covered {len(covered_features)}/{len(target_features)} features. Remaining: {list(remaining)}"
            
            try:
                # Use call_structured for Pydantic validation
                _, result_model, _ = self.llm.call_structured(
                    system_prompt="",  # Already included in current_prompt
                    user_prompt=current_prompt,
                    response_model=InterfaceOutput,
                    purpose=f"interface_design_{file_path}_{iteration + 1}",
                    max_retries=1  # Handle retries at this level
                )
                
                if not result_model:
                    last_error = "Failed to parse result_json or Pydantic validation failed. Please use <result_json></result_json> tags with valid JSON."
                    continue
                
                # Convert Pydantic models to dicts for existing validation logic
                interfaces = [iface.model_dump() for iface in result_model.interfaces]
                
                # Validate each interface
                valid_interfaces = []
                for interface in interfaces:
                    is_valid, error, info = validate_interface(
                        interface, target_features, covered_features
                    )
                    
                    if is_valid:
                        # Add name field from parsed info
                        if info.get("classes"):
                            interface["name"] = f"class {info['classes'][0]}"
                        elif info.get("functions"):
                            interface["name"] = f"function {info['functions'][0]}"
                        
                        # Store CodeUnit objects
                        interface["parsed_units"] = info.get("units", [])
                        
                        valid_interfaces.append(interface)
                        # Update covered features
                        for feat in interface.get("features", []):
                            covered_features.add(feat)
                        
                        # Update feature map
                        for func in info.get("functions", []):
                            feature_interface_map[f"function {func}"] = interface.get("features", [])
                        for cls in info.get("classes", []):
                            feature_interface_map[f"class {cls}"] = interface.get("features", [])
                        
                        # Collect code
                        all_code_blocks.append(interface.get("code", ""))
                        
                        # Collect dependencies if collector is provided
                        if dependency_collector and base_class_files:
                            code = interface.get("code", "")
                            unit_name = interface.get("name", "")
                            
                            # Analyze code for inheritance and type references
                            dependency_collector.analyze_code_dependencies(
                                code=code,
                                file_path=file_path,
                                base_class_files=base_class_files
                            )
                            
                            # Process LLM-declared dependencies
                            llm_deps = interface.get("dependencies")
                            if llm_deps:
                                dependency_collector.process_llm_dependencies(
                                    unit_name=unit_name,
                                    dependencies=llm_deps,
                                    file_path=file_path,
                                    base_class_files=base_class_files
                                )
                    else:
                        self.logger.warning(f"Interface validation failed: {error}")
                        last_error = error
                
                if valid_interfaces:
                    all_interfaces.extend(valid_interfaces)
                    last_error = ""  # Clear error on success
                
            except Exception as e:
                self.logger.error(f"[InterfaceAgent] Error: {e}")
                last_error = str(e)
        
        # Merge all code blocks
        final_code = "\n\n".join(all_code_blocks) if all_code_blocks else ""
        
        success = covered_features >= target_features
        
        # Build units list and mappings in the reference format (ZeroRepo compatible)
        units = []
        units_to_features = {}
        units_to_code = {}
        designed_interfaces = {}  # For storing CodeUnit objects
        
        for interface in all_interfaces:
            interface_name = interface.get("name", "")
            if not interface_name:
                continue
            
            features = interface.get("features", [])
            parsed_units = interface.get("parsed_units", [])
            
            if parsed_units:
                # Each parsed unit gets its own entry keyed by its actual name
                for unit in parsed_units:
                    unit_key = f"{unit.unit_type} {unit.name}"
                    if unit_key not in units:
                        units.append(unit_key)
                    units_to_features[unit_key] = features
                    try:
                        # Use count_lines to get the unit code (ZeroRepo compatible)
                        _, unit_code = unit.count_lines(original=True, return_code=True)
                        units_to_code[unit_key] = unit_code
                    except Exception:
                        # Fallback to full interface code
                        units_to_code[unit_key] = interface.get("code", "")
                    # Store the CodeUnit object
                    designed_interfaces[unit_key] = {
                        "unit": unit,
                        "features": features
                    }
            else:
                # No parsed units — use the interface name as-is
                if interface_name not in units:
                    units.append(interface_name)
                units_to_features[interface_name] = features
                units_to_code[interface_name] = interface.get("code", "")
        
        return {
            "file_path": file_path,
            "file_code": final_code,
            "units": units,
            "units_to_features": units_to_features,
            "units_to_code": units_to_code,
            "designed_interfaces": designed_interfaces,
            "success": success,
            "iterations": iteration + 1
        }


# ============================================================================
# Subtree Interface Agent (All Files in One Subtree)
# ============================================================================

class SubtreeInterfaceAgent:
    """Agent for designing interfaces for ALL files in a subtree in a single LLM session.
    
    Instead of making one LLM call per file, this agent batches all files in a subtree
    into a single prompt, instructing the LLM to design interfaces for each file
    sequentially (following file implementation order). This saves LLM calls and avoids
    redundant context loading.
    
    The agent supports iteration: if some files' features are not fully covered after
    the first call, it retries with feedback, including already-accepted interfaces
    as context.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 10,
        logger: Optional[logging.Logger] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None
    ):
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
            if trajectory is not None:
                self.llm.set_trajectory(trajectory, step_id)
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
    
    def design_subtree_interfaces(
        self,
        file_nodes: List[Dict[str, Any]],
        file_order: List[str],
        repo_info: str,
        data_flow_str: str,
        base_classes_str: str,
        upstream_context: str,
        dependency_collector: Optional[DependencyCollector] = None,
        base_class_files: Optional[Dict[str, str]] = None,
        subtree_name: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Design interfaces for all files in a subtree in batched LLM calls.
        
        Args:
            file_nodes: List of file dicts with 'path' and 'feature_paths'
            file_order: Ordered list of file paths (implementation dependency order)
            repo_info: Repository description
            data_flow_str: Formatted data flow string for this subtree
            base_classes_str: Formatted base classes and data structures
            upstream_context: Context from upstream subtrees
            dependency_collector: Optional collector for fine-grained dependencies
            base_class_files: Optional mapping of class/type names to file paths
            
        Returns:
            Dict mapping file_path -> result dict with keys:
                file_code, units, units_to_features, units_to_code, success
        """
        # Build file info lookup
        file_info_map = {f["path"]: f for f in file_nodes}
        
        # Per-file state tracking
        # file_path -> {target_features, covered_features, all_interfaces, all_code_blocks}
        file_states: Dict[str, Dict[str, Any]] = {}
        for file_path in file_order:
            if file_path not in file_info_map:
                continue
            features = file_info_map[file_path].get("feature_paths", [])
            if not features:
                continue
            file_states[file_path] = {
                "target_features": set(features),
                "covered_features": set(),
                "all_interfaces": [],
                "all_code_blocks": [],
            }
        
        if not file_states:
            self.logger.warning("[SubtreeInterfaceAgent] No files with features to design")
            return {}

        # Build system prompt (tool description is now integrated)
        system_prompt = SUBTREE_INTERFACE_PROMPT

        last_error = ""
        
        for iteration in range(self.max_iterations):
            # Determine which files still need work
            remaining_files = [
                fp for fp in file_order
                if fp in file_states
                and file_states[fp]["covered_features"] < file_states[fp]["target_features"]
            ]
            
            if not remaining_files:
                self.logger.info("[SubtreeInterfaceAgent] All files fully covered")
                break
            
            self.logger.info(
                f"[SubtreeInterfaceAgent] Iteration {iteration + 1}/{self.max_iterations}, "
                f"{len(remaining_files)} files remaining"
            )
            
            # Build user prompt
            user_prompt = self._build_subtree_user_prompt(
                remaining_files=remaining_files,
                file_states=file_states,
                file_info_map=file_info_map,
                repo_info=repo_info,
                data_flow_str=data_flow_str,
                base_classes_str=base_classes_str,
                upstream_context=upstream_context,
                last_error=last_error,
            )
            
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            try:
                _, result_model, _ = self.llm.call_structured(
                    system_prompt="",
                    user_prompt=combined_prompt,
                    response_model=SubtreeInterfaceOutput,
                    purpose=f"subtree_interface_design_{subtree_name}_{iteration + 1}",
                    max_retries=1,
                )
                
                if not result_model:
                    last_error = (
                        "Failed to parse result_json or Pydantic validation failed. "
                        "Please use <result_json></result_json> tags with valid JSON "
                        "matching the design_subtree_interfaces schema."
                    )
                    continue
                
                # Process each file block from LLM response
                last_error = ""
                file_errors = []
                
                for file_block in result_model.files:
                    file_path = file_block.file_path
                    
                    if file_path not in file_states:
                        self.logger.warning(
                            f"[SubtreeInterfaceAgent] Unknown file path from LLM: {file_path}"
                        )
                        continue
                    
                    state = file_states[file_path]
                    target_features = state["target_features"]
                    covered_features = state["covered_features"]
                    
                    # Validate each interface in this file block
                    for interface in file_block.interfaces:
                        iface_dict = interface.model_dump()
                        is_valid, error, info = validate_interface(
                            iface_dict, target_features, covered_features
                        )
                        
                        if is_valid:
                            # Add name from parsed info
                            if info.get("classes"):
                                iface_dict["name"] = f"class {info['classes'][0]}"
                            elif info.get("functions"):
                                iface_dict["name"] = f"function {info['functions'][0]}"
                            
                            iface_dict["parsed_units"] = info.get("units", [])
                            
                            state["all_interfaces"].append(iface_dict)
                            state["all_code_blocks"].append(iface_dict.get("code", ""))
                            
                            for feat in iface_dict.get("features", []):
                                covered_features.add(feat)
                            
                            # Collect dependencies
                            if dependency_collector and base_class_files:
                                code = iface_dict.get("code", "")
                                unit_name = iface_dict.get("name", "")
                                
                                dependency_collector.analyze_code_dependencies(
                                    code=code,
                                    file_path=file_path,
                                    base_class_files=base_class_files
                                )
                                
                                llm_deps = iface_dict.get("dependencies")
                                if llm_deps:
                                    dependency_collector.process_llm_dependencies(
                                        unit_name=unit_name,
                                        dependencies=llm_deps,
                                        file_path=file_path,
                                        base_class_files=base_class_files
                                    )
                            
                            # Update base_class_files so later files can reference
                            if base_class_files is not None:
                                name = iface_dict.get("name", "")
                                parts = name.split(" ", 1)
                                if len(parts) == 2:
                                    base_class_files[parts[1]] = file_path
                        else:
                            self.logger.warning(
                                f"[SubtreeInterfaceAgent] Validation failed for "
                                f"{file_path}: {error}"
                            )
                            file_errors.append(f"{file_path}: {error}")
                
                if file_errors:
                    last_error = "[Validation Errors]\n" + "\n".join(file_errors)
                
            except Exception as e:
                self.logger.error(f"[SubtreeInterfaceAgent] Error: {e}")
                last_error = str(e)
        
        # Build final results for each file
        results: Dict[str, Dict[str, Any]] = {}
        all_new_features: List[Dict[str, str]] = []

        for file_path in file_order:
            if file_path not in file_states:
                continue

            state = file_states[file_path]
            file_result, new_features = self._build_file_result(
                file_path=file_path,
                all_interfaces=state["all_interfaces"],
                all_code_blocks=state["all_code_blocks"],
                target_features=state["target_features"],
                covered_features=state["covered_features"],
            )
            results[file_path] = file_result
            all_new_features.extend(new_features)

        # Attach new features to results for caller to process
        if all_new_features:
            # Store in a special key that will be extracted by the orchestrator
            results["__new_features__"] = all_new_features

        return results
    
    def _build_subtree_user_prompt(
        self,
        remaining_files: List[str],
        file_states: Dict[str, Dict[str, Any]],
        file_info_map: Dict[str, Dict[str, Any]],
        repo_info: str,
        data_flow_str: str,
        base_classes_str: str,
        upstream_context: str,
        last_error: str,
    ) -> str:
        """Build the user prompt for subtree interface design."""
        # Build file list section
        files_section_parts = []
        for i, file_path in enumerate(remaining_files, 1):
            state = file_states.get(file_path, {})
            target_features = state.get("target_features", set())
            covered_features = state.get("covered_features", set())
            remaining_features = target_features - covered_features
            
            features_str = "\n".join([f"    - {f}" for f in sorted(remaining_features)])
            files_section_parts.append(
                f"  {i}. `{file_path}`\n"
                f"     Features to design:\n{features_str}"
            )
        
        files_section = "\n\n".join(files_section_parts)
        
        # Build already-completed context (from files fully or partially done)
        completed_parts = []
        for file_path, state in file_states.items():
            if file_path in remaining_files and not state["all_code_blocks"]:
                continue  # Skip files with nothing completed yet in remaining list
            if not state["all_code_blocks"]:
                continue
            
            code_preview = "\n\n".join(state["all_code_blocks"])
            # Truncate if very long
            code_lines = code_preview.split("\n")
            if len(code_lines) > 40:
                code_preview = "\n".join(code_lines[:40]) + "\n# ... (truncated)"
            
            completed_parts.append(
                f"File: `{file_path}` (already designed)\n"
                f"```python\n{code_preview}\n```"
            )
        
        completed_context = (
            "\n\n".join(completed_parts) if completed_parts
            else "No files designed yet in this subtree."
        )
        
        # Assemble user prompt
        # Detect import convention from file paths
        import_convention = ""
        if remaining_files:
            # Infer prefix from file paths in this subtree
            sample_path = remaining_files[0]
            parts = sample_path.replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[0] == "src":
                prefix = f"src.{parts[1]}"
                import_convention = build_import_convention_snippet(prefix=prefix)

        prompt = f"""[Begin Subtree Interface Design]

Design interfaces for ALL of the following files, in the listed order.
Each file's features must be fully covered. Later files may import from earlier ones.

{import_convention}
=== Files to Design (in implementation order) ===
{files_section}

=== Global Context ===

--- Repository Info ---
{repo_info}

--- Data Flow Graph ---
{data_flow_str}

--- Upstream Context (from other subtrees) ---
{upstream_context}

--- Already Designed in This Subtree ---
{completed_context}

--- Available Base Classes & Data Structures ---
{base_classes_str}
"""
        
        if last_error:
            prompt += f"\n\n[Previous Iteration Feedback]: {last_error}"
        
        # Add overall progress
        total_target = sum(
            len(file_states[fp]["target_features"])
            for fp in remaining_files if fp in file_states
        )
        total_covered = sum(
            len(file_states[fp]["covered_features"])
            for fp in remaining_files if fp in file_states
        )
        if total_covered > 0:
            prompt += (
                f"\n\n[Progress]: {total_covered}/{total_target + total_covered} features "
                f"covered across remaining files. "
                f"Please cover all remaining features."
            )
        
        return prompt
    
    @staticmethod
    def _build_file_result(
        file_path: str,
        all_interfaces: List[Dict[str, Any]],
        all_code_blocks: List[str],
        target_features: Set[str],
        covered_features: Set[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Build the result dict for a single file (compatible with InterfaceAgent output).

        Returns:
            Tuple of (file_result_dict, new_features_list)
            where new_features_list contains dicts with feature_path, unit_name, file_path
        """
        final_code = "\n\n".join(all_code_blocks) if all_code_blocks else ""
        success = covered_features >= target_features

        units = []
        units_to_features = {}
        units_to_code = {}
        new_features_list = []  # Collect new features for top-level reporting

        for interface in all_interfaces:
            interface_name = interface.get("name", "")
            if not interface_name:
                continue

            features = interface.get("features", [])
            parsed_units = interface.get("parsed_units", [])

            # Identify new features (those not in target_features)
            new_features = [f for f in features if f not in target_features]

            if parsed_units:
                for unit in parsed_units:
                    unit_key = f"{unit.unit_type} {unit.name}"
                    if unit_key not in units:
                        units.append(unit_key)
                    units_to_features[unit_key] = features
                    # Track new features
                    for nf in new_features:
                        new_features_list.append({
                            "feature_path": nf,
                            "unit_name": unit_key,
                            "file_path": file_path,
                        })
                    try:
                        _, unit_code = unit.count_lines(original=True, return_code=True)
                        units_to_code[unit_key] = unit_code
                    except Exception:
                        units_to_code[unit_key] = interface.get("code", "")
            else:
                if interface_name not in units:
                    units.append(interface_name)
                units_to_features[interface_name] = features
                # Track new features
                for nf in new_features:
                    new_features_list.append({
                        "feature_path": nf,
                        "unit_name": interface_name,
                        "file_path": file_path,
                    })
                units_to_code[interface_name] = interface.get("code", "")

        result = {
            "file_path": file_path,
            "file_code": final_code,
            "units": units,
            "units_to_features": units_to_features,
            "units_to_code": units_to_code,
            "success": success,
        }

        return result, new_features_list


# ============================================================================
# Interface Orchestrator (Full Workflow)
# ============================================================================

class InterfaceOrchestrator:
    """Orchestrates interface design across all subtrees and files."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_file_iterations: int = 10,
        max_planning_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None,
        output_path: Optional[str] = None
    ):
        # Create LLMClient with trajectory support if not provided
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
            # Update trajectory info on existing client
            if trajectory is not None:
                self.llm.set_trajectory(trajectory, step_id)
        self.max_file_iterations = max_file_iterations
        self.max_planning_retries = max_planning_retries
        self.logger = logger or logging.getLogger(__name__)
        self.trajectory = trajectory
        self.step_id = step_id
        self.output_path = output_path
    
    def design_all_interfaces(
        self,
        skeleton: Dict[str, Any],
        data_flow: Dict[str, Any],
        base_classes: List[Dict[str, Any]],
        repo_info: str,
        dependency_collector: Optional[DependencyCollector] = None,
        data_structures: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Design interfaces for all files in the skeleton.
        
        Args:
            skeleton: The skeleton.json data
            data_flow: The data_flow.json data
            base_classes: List of base class definitions
            repo_info: Repository description
            dependency_collector: Optional collector for fine-grained dependencies
            data_structures: Optional list of data flow data structure definitions
            
        Returns:
            Dict with all interfaces organized by subtree
        """
        # Get subtree order from data flow
        subtree_order = data_flow.get("subtree_order", [])
        data_flow_edges = data_flow.get("data_flow", [])
        
        # If no subtree order, extract from skeleton
        if not subtree_order:
            subtree_order = self._extract_subtree_names(skeleton)
        
        self.logger.info(f"[InterfaceOrchestrator] Processing {len(subtree_order)} subtrees")
        self.logger.info(f"[InterfaceOrchestrator] Subtree order: {subtree_order}")
        
        # Format base classes and data structures together for prompt context
        base_classes_str = format_base_classes_and_data_structures(
            base_classes, data_structures or []
        )
        
        # Build base_class_files mapping for dependency analysis
        # Include both base_classes and data_structures
        base_class_files = self._build_base_class_files_mapping(
            base_classes, data_structures=data_structures
        )
        
        # --- Initialize GlobalInterfaceRegistry ---
        global_registry = GlobalInterfaceRegistry()

        # Track state across subtrees
        all_interfaces = {}
        implemented_subtrees = {}  # subtree -> list of implemented file info
        all_import_warnings = []  # collect import cross-validation warnings
        all_new_features = []  # collect new features created across all subtrees

        # Process each subtree
        for subtree_name in subtree_order:
            self.logger.info(f"[InterfaceOrchestrator] Processing subtree: {subtree_name}")
            
            # Find files for this subtree
            file_nodes = self._find_files_for_subtree(skeleton, subtree_name)
            if not file_nodes:
                self.logger.warning(f"No files found for subtree: {subtree_name}")
                continue
            
            self.logger.info(f"[InterfaceOrchestrator] Found {len(file_nodes)} files for {subtree_name}")
            
            # --- Merge global registry symbols into base_class_files ---
            # This allows DependencyCollector to resolve cross-subtree callees
            global_symbols = global_registry.get_all_public_symbols()
            for symbol_name, symbol_file in global_symbols.items():
                if symbol_name not in base_class_files:
                    base_class_files[symbol_name] = symbol_file
            
            # Plan file order
            file_order = self._plan_file_order(file_nodes, repo_info, subtree_name=subtree_name)
            
            # Build context once for the whole subtree
            filtered_data_flow_str = self._filter_data_flow_for_subtree(
                data_flow_edges, subtree_name
            )
            
            # --- Enhanced upstream context with structured interface listings ---
            upstream_context = self._build_upstream_context_for_subtree(
                data_flow_edges, subtree_name, implemented_subtrees,
                global_registry=global_registry
            )
            
            # Design all files in this subtree in a single LLM session
            agent = SubtreeInterfaceAgent(
                llm_client=self.llm,
                max_iterations=self.max_file_iterations,
                logger=self.logger
            )
            
            file_results = agent.design_subtree_interfaces(
                file_nodes=file_nodes,
                file_order=file_order,
                repo_info=repo_info,
                data_flow_str=filtered_data_flow_str,
                base_classes_str=base_classes_str,
                upstream_context=upstream_context,
                dependency_collector=dependency_collector,
                base_class_files=base_class_files,
                subtree_name=subtree_name,
            )

            # Extract new features from this subtree
            subtree_new_features = file_results.pop("__new_features__", [])
            for nf in subtree_new_features:
                nf["subtree"] = subtree_name
            all_new_features.extend(subtree_new_features)

            # Process results for each file
            subtree_implemented = []
            subtree_interfaces = {}
            
            for file_path in file_order:
                result = file_results.get(file_path)
                if not result:
                    continue
                
                # Store interface data
                subtree_interfaces[file_path] = {
                    "file_code": result.get("file_code", ""),
                    "units": result.get("units", []),
                    "units_to_features": result.get("units_to_features", {}),
                    "units_to_code": result.get("units_to_code", {})
                }
                
                file_node = next((f for f in file_nodes if f["path"] == file_path), None)
                file_features = file_node.get("feature_paths", []) if file_node else []
                
                if result.get("success"):
                    subtree_implemented.append({
                        "path": file_path,
                        "features": file_features,
                        "code": result.get("file_code", ""),
                        "units": result.get("units", []),
                        "units_to_features": result.get("units_to_features", {})
                    })
                    self.logger.info(f"[InterfaceOrchestrator] [OK] Completed {file_path}")
                else:
                    self.logger.warning(f"[InterfaceOrchestrator] [FAIL] Failed {file_path}")
            
            # --- A1: Register completed subtree interfaces to GlobalInterfaceRegistry ---
            global_registry.register_from_subtree_result(subtree_name, subtree_interfaces)
            self.logger.info(
                f"[InterfaceOrchestrator] Registered {len(subtree_interfaces)} files "
                f"from '{subtree_name}' to GlobalInterfaceRegistry "
                f"(total symbols: {len(global_registry.get_all_public_symbols())})"
            )
            
            # --- A2: Import cross-validation for this subtree ---
            for file_path, file_data in subtree_interfaces.items():
                file_code = file_data.get("file_code", "")
                # Collect declared calls from dependency_collector for this file
                declared_calls = set()
                if dependency_collector:
                    for edge in dependency_collector.invocation_edges:
                        if edge.get("caller_file") == file_path:
                            declared_calls.add(edge.get("callee", ""))
                
                warnings = cross_validate_imports_vs_calls(
                    code=file_code,
                    file_path=file_path,
                    declared_calls=list(declared_calls),
                    global_registry=global_registry
                )
                if warnings:
                    all_import_warnings.extend(warnings)
                    for w in warnings:
                        self.logger.info(
                            f"[ImportValidation] {w['message']}"
                        )
            
            # Store subtree results
            all_interfaces[subtree_name] = {
                "files_order": file_order,
                "interfaces": subtree_interfaces
            }
            implemented_subtrees[subtree_name] = subtree_implemented
            
            # Save after each subtree
            self._save_interfaces(
                self._build_result(all_interfaces, subtree_order, implemented_subtrees)
            )
        
        # Compile final result
        final_result = self._build_result(all_interfaces, subtree_order, implemented_subtrees)

        # Store import warnings and global registry in result for downstream use
        final_result["_import_warnings"] = all_import_warnings
        final_result["_global_registry"] = global_registry

        # Store new features for output and RPG update
        if all_new_features:
            final_result["new_features"] = all_new_features
            self.logger.info(
                f"[InterfaceOrchestrator] Created {len(all_new_features)} new features "
                f"for glue/orchestration code"
            )

        self._save_interfaces(final_result)
        return final_result
    
    def _build_result(
        self,
        all_interfaces: Dict[str, Any],
        subtree_order: List[str],
        implemented_subtrees: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build the result dict from current state."""
        return {
            "subtrees": all_interfaces,
            "subtree_order": subtree_order,
            "implemented_subtrees": {
                st: [f["path"] for f in files]
                for st, files in implemented_subtrees.items()
            },
            "success": True
        }
    
    def _save_interfaces(self, result: Dict[str, Any]) -> None:
        """Save current interfaces result to output_path (if configured).
        
        Strips internal keys (prefixed with '_') that contain non-serializable
        objects before writing to JSON.
        """
        if not self.output_path:
            return
        try:
            output = Path(self.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Filter out non-serializable internal keys
            serializable = {
                k: v for k, v in result.items()
                if not k.startswith("_")
            }
            with open(output, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[InterfaceOrchestrator] Saved interfaces to {output}")
        except Exception as e:
            self.logger.warning(f"[InterfaceOrchestrator] Failed to save interfaces: {e}")
    
    def _build_base_class_files_mapping(
        self,
        base_classes: List[Dict[str, Any]],
        data_structures: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Build a mapping from class/type names to their file paths.
        
        Args:
            base_classes: List of base class definitions from base_classes.json
            data_structures: Optional list of data structure definitions
                (only entries with file_path assigned are included)
            
        Returns:
            Dict mapping class/type names to file paths
        """
        mapping = {}
        
        # Process base classes
        for bc in base_classes:
            file_path = bc.get("file_path", "")
            code = bc.get("code", "")
            
            if not file_path or not code:
                continue
            
            # Parse code to extract class and type names
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        mapping[node.name] = file_path
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Top-level functions might be utilities
                        mapping[node.name] = file_path
            except SyntaxError:
                continue
        
        # Process data structures (only those with file_path already assigned)
        if data_structures:
            for ds in data_structures:
                file_path = ds.get("file_path", "")
                code = ds.get("code", "")
                
                if not file_path or not code:
                    continue
                
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            mapping[node.name] = file_path
                except SyntaxError:
                    continue
                
                # Also map data_flow_types names to file paths
                for dt_name in ds.get("data_flow_types", []):
                    if dt_name not in mapping:
                        mapping[dt_name] = file_path
        
        return mapping
    
    def _extract_subtree_names(self, skeleton: Dict[str, Any]) -> List[str]:
        """Extract subtree/component names from skeleton."""
        components = set()
        
        def traverse(node):
            if node.get("type") == "file":
                for fp in node.get("feature_paths", []):
                    if "/" in fp:
                        components.add(fp.split("/")[0])
            elif node.get("type") == "directory":
                for child in node.get("children", []):
                    traverse(child)
        
        root = skeleton.get("root", skeleton)
        traverse(root)
        return sorted(list(components))
    
    def _find_files_for_subtree(
        self,
        skeleton: Dict[str, Any],
        subtree_name: str
    ) -> List[Dict[str, Any]]:
        """Find all files belonging to a subtree."""
        files = []
        
        def traverse(node):
            if node.get("type") == "file":
                feature_paths = node.get("feature_paths", [])
                # Check if any feature path belongs to this subtree
                for fp in feature_paths:
                    if fp.startswith(subtree_name + "/") or fp == subtree_name:
                        files.append({
                            "path": node.get("path", ""),
                            "feature_paths": feature_paths
                        })
                        break
            elif node.get("type") == "directory":
                for child in node.get("children", []):
                    traverse(child)
        
        root = skeleton.get("root", skeleton)
        traverse(root)
        return files
    
    def _plan_file_order(
        self,
        file_nodes: List[Dict[str, Any]],
        repo_info: str,
        subtree_name: str,
    ) -> List[str]:
        """Plan the implementation order for files."""
        file_paths = [f["path"] for f in file_nodes]
        
        if len(file_paths) <= 1:
            return file_paths
        
        # Build files info for planning prompt
        files_info = {}
        for node in file_nodes:
            files_info[node["path"]] = node.get("feature_paths", [])
        
        files_to_planned = ""
        for path, features in files_info.items():
            feature_str = "\n  - ".join(features) if features else "(no features)"
            files_to_planned += f"- {path}:\n  - {feature_str}\n\n"
        
        # Build planning prompt
        prompt = PLAN_FILE_PROMPT.format(
            repo_info=repo_info,
            trees_info="(Feature tree omitted for brevity)",
            files_to_planned=files_to_planned
        )
        
        # Try to get valid graph
        for attempt in range(self.max_planning_retries):
            try:
                response = self.llm.generate(prompt, purpose=f"plan_file_order_{subtree_name}_{attempt + 1}")
                parsed = self.llm.parse_json_block(response)
                
                if not parsed:
                    continue
                
                graph = parsed.get("file_implementation_graph", [])
                feedback, is_valid = validate_file_implementation_graph(graph, file_paths)
                
                if is_valid:
                    order = topo_sort_file_graph(graph)
                    if order:
                        return order
                
            except Exception as e:
                self.logger.warning(f"File planning attempt {attempt + 1} failed: {e}")
        
        # Fallback: return files in original order
        self.logger.warning("Using fallback file order (no planning)")
        return file_paths
    
    def _filter_data_flow_for_file(
        self,
        data_flow_edges: List[Dict[str, Any]],
        current_subtree: str,
        file_path: str
    ) -> str:
        """Filter data flow edges to only include those related to current file.
        
        Includes edges where:
        - source or target is the current subtree
        - edges directly connected to the current subtree's neighbors
        """
        if not data_flow_edges:
            return "No data flow defined."
        
        # Find subtrees directly connected to current subtree
        connected_subtrees = {current_subtree}
        for edge in data_flow_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source == current_subtree:
                connected_subtrees.add(target)
            if target == current_subtree:
                connected_subtrees.add(source)
        
        # Filter edges that involve connected subtrees
        filtered_edges = []
        for edge in data_flow_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source in connected_subtrees or target in connected_subtrees:
                filtered_edges.append(edge)
        
        if not filtered_edges:
            return "No related data flow for this file."
        
        return format_data_flow_edges(filtered_edges)
    
    def _filter_data_flow_for_subtree(
        self,
        data_flow_edges: List[Dict[str, Any]],
        current_subtree: str
    ) -> str:
        """Filter data flow edges to include those related to the current subtree.
        
        Includes edges where:
        - source or target is the current subtree
        - edges between subtrees directly connected to current subtree
        """
        if not data_flow_edges:
            return "No data flow defined."
        
        # Find subtrees directly connected to current subtree
        connected_subtrees = {current_subtree}
        for edge in data_flow_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source == current_subtree:
                connected_subtrees.add(target)
            if target == current_subtree:
                connected_subtrees.add(source)
        
        # Filter edges that involve connected subtrees
        filtered_edges = []
        for edge in data_flow_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source in connected_subtrees or target in connected_subtrees:
                filtered_edges.append(edge)
        
        if not filtered_edges:
            return "No related data flow for this subtree."
        
        return format_data_flow_edges(filtered_edges)
    
    def _build_upstream_context_for_subtree(
        self,
        data_flow_edges: List[Dict[str, Any]],
        current_subtree: str,
        implemented_subtrees: Dict[str, List[Dict[str, Any]]],
        top_n: int = 30,
        global_registry: Optional[GlobalInterfaceRegistry] = None
    ) -> str:
        """Build context from upstream subtrees for an entire subtree.
        
        Unlike _build_upstream_context (per-file), this does NOT include
        within-subtree files since all files are being designed together.
        Only includes files from upstream (connected) subtrees.
        
        When global_registry is provided (A3 enhancement), prepends a
        structured interface listing before the code snippets, giving the
        LLM a clear, concise view of all callable interfaces.
        """
        # Find connected subtrees
        upstream_names = set()
        for edge in data_flow_edges:
            if edge.get("target") == current_subtree:
                upstream_names.add(edge.get("source", ""))
            if edge.get("source") == current_subtree:
                upstream_names.add(edge.get("target", ""))
        
        context_parts = []
        
        # --- A3: Structured interface listing (prepended) ---
        if global_registry and upstream_names:
            structured_listing = global_registry.get_all_structured_listings_for_upstream(
                upstream_names
            )
            if structured_listing and structured_listing != "No upstream interfaces available.":
                context_parts.append(
                    "=== Available Interfaces from Upstream Subtrees ===\n"
                    "(You can import and call these interfaces in your designs)\n\n"
                    f"{structured_listing}\n"
                )
        
        # --- Integration directives from data flow ---
        inbound_edges = [
            e for e in data_flow_edges if e.get("target") == current_subtree
        ]
        outbound_edges = [
            e for e in data_flow_edges if e.get("source") == current_subtree
        ]

        if inbound_edges or outbound_edges:
            directive_parts = [
                f'=== Integration Contracts for "{current_subtree}" ===',
                "Your subtree has the following data flow contracts.",
                "Design your interfaces to fulfill these contracts.\n",
            ]

            if inbound_edges:
                directive_parts.append("INBOUND (your subtree must consume):")
                for edge in inbound_edges:
                    source = edge.get("source", "?")
                    data_type = edge.get("data_type", "?")
                    transformation = edge.get("transformation", "")
                    line = f'  - {data_type} from "{source}"'
                    # Try to find the producing interface in global_registry
                    if global_registry:
                        for fp, unit_list in global_registry.file_units.items():
                            found = False
                            for ui in unit_list:
                                if (ui.get("subtree_name") == source
                                        and data_type
                                        in ui.get("signature_summary", "")):
                                    line += (
                                        f"\n    Produced by: {ui['unit_type']} "
                                        f"{ui['bare_name']} in {fp}"
                                    )
                                    found = True
                                    break
                            if found:
                                break
                    if transformation:
                        line += f"\n    Context: {transformation}"
                    directive_parts.append(line)
                directive_parts.append(
                    "  \u2192 Design at least one interface that "
                    "accepts/imports the above data.\n"
                )

            if outbound_edges:
                directive_parts.append("OUTBOUND (your subtree must produce):")
                for edge in outbound_edges:
                    target = edge.get("target", "?")
                    data_type = edge.get("data_type", "?")
                    transformation = edge.get("transformation", "")
                    line = f'  - {data_type} to "{target}"'
                    if transformation:
                        line += f"\n    Context: {transformation}"
                    directive_parts.append(line)
                directive_parts.append(
                    "  \u2192 Design at least one interface that "
                    "produces/returns the above data.\n"
                )

            context_parts.append("\n".join(directive_parts) + "\n")

        # --- Original code snippet context ---
        included_paths = set()
        code_parts = []
        
        for upstream in sorted(upstream_names):
            impl_files = implemented_subtrees.get(upstream, [])
            if not impl_files:
                continue
            
            for file_info in impl_files:
                if len(code_parts) >= top_n:
                    break
                
                path = file_info.get("path", "")
                if path in included_paths:
                    continue
                
                included_paths.add(path)
                features = ", ".join(file_info.get("features", [])[:5])
                code = file_info.get("code", "")
                code_lines = code.split("\n")[:30]
                code_skeleton = "\n".join(code_lines)
                
                code_parts.append(
                    f"### From module: `{upstream}`\n"
                    f"File: `{path}`\n"
                    f"Features: {features}\n"
                    f"```python\n{code_skeleton}\n```\n"
                )
        
        if code_parts:
            context_parts.extend(code_parts[:top_n])
        
        if not context_parts:
            return "No upstream modules connected to this subtree."
        
        return "\n".join(context_parts)
    
    def _build_upstream_context(
        self,
        data_flow_edges: List[Dict[str, Any]],
        current_subtree: str,
        implemented_subtrees: Dict[str, List[Dict[str, Any]]],
        top_n: int = 20,
        file_path: Optional[str] = None,
        subtree_implemented: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build context from upstream subtrees, filtered for relevance.
        
        Includes:
        1. Files in the same directory as current file
        2. Files with edges connected to current file's units
        """
        # Get current file's directory
        current_dir = str(Path(file_path).parent) if file_path else ""
        
        # Find upstream subtrees
        upstream_names = set()
        for edge in data_flow_edges:
            if edge.get("target") == current_subtree:
                upstream_names.add(edge.get("source", ""))
            if edge.get("source") == current_subtree:
                upstream_names.add(edge.get("target", ""))
        
        context_parts = []
        included_paths = set()
        
        # 1. First add files from same directory in current subtree
        if subtree_implemented and current_dir:
            for file_info in subtree_implemented:
                path = file_info.get("path", "")
                if path in included_paths:
                    continue
                file_dir = str(Path(path).parent)
                if file_dir == current_dir:
                    included_paths.add(path)
                    features = ", ".join(file_info.get("features", [])[:5])
                    code = file_info.get("code", "")
                    code_lines = code.split("\n")[:30]
                    code_skeleton = "\n".join(code_lines)
                    
                    context_parts.append(
                        f"### Same directory: `{current_dir}`\n"
                        f"File: `{path}`\n"
                        f"Features: {features}\n"
                        f"```python\n{code_skeleton}\n```\n"
                    )
                    if len(context_parts) >= top_n:
                        break
        
        # 2. Add files from upstream subtrees
        if upstream_names and len(context_parts) < top_n:
            for upstream in sorted(upstream_names):
                impl_files = implemented_subtrees.get(upstream, [])
                if not impl_files:
                    continue
                
                for file_info in impl_files:
                    if len(context_parts) >= top_n:
                        break
                    
                    path = file_info.get("path", "")
                    if path in included_paths:
                        continue
                    
                    included_paths.add(path)
                    features = ", ".join(file_info.get("features", [])[:5])
                    code = file_info.get("code", "")
                    code_lines = code.split("\n")[:30]
                    code_skeleton = "\n".join(code_lines)
                    
                    context_parts.append(
                        f"### From module: `{upstream}`\n"
                        f"File: `{path}`\n"
                        f"Features: {features}\n"
                        f"```python\n{code_skeleton}\n```\n"
                    )
        
        if not context_parts:
            return "No upstream modules connected to this subtree."
        
        return "\n".join(context_parts[:top_n])
    
    def _build_implemented_summary(
        self,
        implemented_files: List[Dict[str, Any]],
        file_path: Optional[str] = None,
        top_n: int = 20
    ) -> str:
        """Build summary of implemented files in current subtree.
        
        Prioritizes files in the same directory as the current file.
        """
        if not implemented_files:
            return "No files implemented yet in this subtree."
        
        # Get current file's directory
        current_dir = str(Path(file_path).parent) if file_path else ""
        
        # Separate files: same directory first, then others
        same_dir_files = []
        other_files = []
        
        for file_info in implemented_files:
            path = file_info.get("path", "")
            file_dir = str(Path(path).parent)
            if current_dir and file_dir == current_dir:
                same_dir_files.append(file_info)
            else:
                other_files.append(file_info)
        
        # Prioritize same directory files, then add others up to top_n
        prioritized_files = same_dir_files + other_files
        selected_files = prioritized_files[-top_n:]  # Take last top_n (most recent)
        
        parts = []
        for file_info in selected_files:
            path = file_info.get("path", "")
            features = file_info.get("features", [])[:5]
            features_str = ", ".join(features)
            
            code = file_info.get("code", "")
            code_lines = code.split("\n")[:20]
            code_skeleton = "\n".join(code_lines)
            
            # Mark if same directory
            dir_marker = " (same dir)" if current_dir and str(Path(path).parent) == current_dir else ""
            
            parts.append(
                f"#### Implemented File: `{path}`{dir_marker}\n"
                f"Features: {features_str}\n"
                f"```python\n{code_skeleton}\n```\n"
            )
        
        return "\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    agent = InterfaceAgent()
    result = agent.design_file_interface(
        file_path="src/module/file.py",
        file_features=["module/feature1", "module/feature2"],
        repo_info="A test repository",
        data_flow_str="A -> B: Data",
        base_classes_str="No base classes",
        upstream_context="No upstream context",
        implemented_summary="No implemented files"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "code"}, indent=2))
