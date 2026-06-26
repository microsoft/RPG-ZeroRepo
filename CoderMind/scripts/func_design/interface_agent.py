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
from typing import Dict, List, Optional, Tuple, Any, Set, Iterator
from collections import defaultdict, deque
from pydantic import BaseModel, Field, model_validator

# Import ParsedFile and CodeUnit for code parsing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rpg.code_unit import ParsedFile, CodeUnit

# AST inspection routes through the decoder language backend so
# code-structure extraction can vary by target language. The direct
# ``ast`` import supports Python-specific node inspection for docstrings
# and annotation syntax below.
from decoder_lang import get_backend
from decoder_lang.backend import LanguageBackend
from decoder_lang.prompt_directive import with_language_directive

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

    @model_validator(mode="before")
    @classmethod
    def _normalise_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalised = dict(value)
        if "features" not in normalised:
            if "feature_paths" in normalised:
                normalised["features"] = normalised["feature_paths"]
            elif "feature_path" in normalised:
                normalised["features"] = [normalised["feature_path"]]
        if "dependencies" not in normalised and "dependency" in normalised:
            normalised["dependencies"] = normalised["dependency"]
        return normalised


class InterfaceOutput(BaseModel):
    """Output from LLM for interface design."""
    interfaces: List[InterfaceDefinition] = Field(..., min_length=1, description="List of interface definitions (must not be empty)")


class FileInterfaceBlock(BaseModel):
    """Block of interface definitions for a single file within a subtree batch."""
    file_path: str = Field(..., description="Path to the file being designed")
    interfaces: List[InterfaceDefinition] = Field(..., min_length=1, description="Interface definitions for this file")

    @model_validator(mode="before")
    @classmethod
    def _normalise_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalised = dict(value)
        if "file_path" not in normalised and "path" in normalised:
            normalised["file_path"] = normalised["path"]
        if "interfaces" not in normalised:
            for alias in ("interface_definitions", "interface_units", "units"):
                if alias in normalised:
                    normalised["interfaces"] = normalised[alias]
                    break
        if "interfaces" not in normalised and "code" in normalised:
            interface_data = {
                key: normalised[key]
                for key in ("features", "feature_paths", "feature_path", "code", "dependencies")
                if key in normalised
            }
            normalised["interfaces"] = [interface_data]
        return normalised


class SubtreeInterfaceOutput(BaseModel):
    """Output from LLM for subtree-level interface design (all files at once)."""
    files: List[FileInterfaceBlock] = Field(..., min_length=1, description="Interface blocks organized by file, in implementation order")


class FileImplementationGraph(BaseModel):
    """Graph of file implementation order."""
    file_implementation_graph: List[Dict[str, str]] = Field(default_factory=list)


# ============================================================================
# Dependency Collector
# ============================================================================

def _dedup_edges_in_place(edges: List[Dict[str, Any]], key_fields: Tuple[str, ...]) -> int:
    """Remove duplicate edges in place (first-seen wins). Returns count removed.

    Edges are identified by the tuple of values for ``key_fields``. The first
    occurrence is kept (preserving earliest ``generator`` provenance).
    """
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for e in edges:
        key = tuple(e.get(f) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    removed = len(edges) - len(deduped)
    if removed:
        edges[:] = deduped  # mutate in place to keep shared references valid
    return removed


class DependencyCollector:
    """Collect fine-grained dependencies discovered during interface design.
    
    Dependencies are collected from two sources:
    1. Program analysis (AST parsing) - inheritance and type references from code
    2. LLM declarations - expected function calls declared by LLM
    """
    
    def __init__(
        self,
        known_base_classes: Set[str],
        known_types: Set[str],
        target_language: Optional[str] = None,
    ):
        """Initialize the dependency collector.
        
        Args:
            known_base_classes: Set of base class names from base_classes.json
            known_types: Set of known type names (data structures, etc.)
        """
        self.known_base_classes = known_base_classes
        self.known_types = known_types
        self.backend = get_backend(target_language)
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
        """Add an inheritance relationship (child extends parent).

        Duplicate (child, parent, source_file, parent_file) entries are
        skipped so repeated calls from AST + LLM paths don't double-count.
        """
        for existing in self.inheritance_edges:
            if (existing.get("child") == child_class
                and existing.get("parent") == parent_class
                and existing.get("source_file") == source_file
                and existing.get("parent_file") == parent_file):
                return
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
        Duplicate edges (same caller/callee/files) are also skipped so
        callers can safely invoke this from multiple paths (AST inspection,
        LLM declarations, global-review fixes) without inflating counts.
        """
        # --- self-call filter ---
        bare_caller = caller.split(" ", 1)[-1] if " " in caller else caller
        bare_callee = callee.split(" ", 1)[-1] if " " in callee else callee
        if bare_caller == bare_callee and (callee_file is None or callee_file == caller_file):
            return

        # --- duplicate filter: ignore identical (caller, callee, files) ---
        for existing in self.invocation_edges:
            if (existing.get("caller") == caller
                and existing.get("callee") == callee
                and existing.get("caller_file") == caller_file
                and existing.get("callee_file") == callee_file):
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
        """Add a type reference relationship.

        Duplicate (unit, referenced_type, source_file, type_file) entries
        are skipped so AST and LLM-declared references don't double-count.
        """
        for existing in self.reference_edges:
            if (existing.get("unit") == unit_name
                and existing.get("referenced_type") == referenced_type
                and existing.get("source_file") == source_file
                and existing.get("type_file") == type_file):
                return
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
        """Extract code-level dependency edges from interface code.

        Inheritance is resolved uniformly across languages through the
        backend's :meth:`list_inheritance` (Python derives it from class
        bases; tree-sitter backends emit ``inherits`` edges). Type
        references from annotations are a Python-specific enrichment;
        other languages supply equivalent ``uses_types`` information via
        the LLM-declared dependencies (see :meth:`process_llm_dependencies`).

        Args:
            code: Interface source code to analyze.
            file_path: Path of the file containing this code.
            base_class_files: Mapping of class/type names to file paths.
        """
        # Inheritance — language-agnostic via the backend.
        for dep in self.backend.list_inheritance(code, file_path):
            child = dep.src
            parent = dep.symbol or dep.dst
            if child and parent and parent in self.known_base_classes:
                parent_file = base_class_files.get(parent)
                self.add_inheritance(child, parent, file_path, parent_file)

        # Type references from annotations — Python-specific rich
        # extraction. Other languages cover this via LLM ``uses_types``.
        if self.backend.name == "python":
            self._analyze_python_invocations(code, file_path)
            self._analyze_python_type_references(code, file_path, base_class_files)

    def _analyze_python_invocations(self, code: str, file_path: str) -> None:
        """Add same-file Python invocation edges from function bodies."""
        units = self.backend.list_code_units(code, file_path)
        local_callables: Dict[str, List[str]] = defaultdict(list)
        caller_nodes: List[Tuple[str, ast.AST, Optional[str]]] = []

        for unit in units:
            if unit.unit_type not in ("function", "method", "class"):
                continue
            if unit.unit_type == "method":
                prefix = "method"
            elif unit.unit_type == "class":
                prefix = "class"
            else:
                prefix = "function"
            unit_name = f"{prefix} {unit.name}"
            local_callables[unit.name].append(unit_name)
            node = (unit.extra or {}).get("ast_node")
            if node is not None and unit.unit_type in ("function", "method"):
                owner_class = None
                if unit.unit_type == "method" and unit.name == "__init__":
                    parent = getattr(unit, "parent", None)
                    if parent:
                        owner_class = f"class {parent}"
                caller_nodes.append((unit_name, node, owner_class))

        local_calls: Dict[str, Set[str]] = defaultdict(set)
        for caller, node, owner_class in caller_nodes:
            for child in _iter_calls_in_own_scope(node):
                callee_name = _python_call_name(child.func)
                if not callee_name:
                    continue
                candidates = local_callables.get(callee_name, [])
                if len(candidates) != 1:
                    continue
                callee = candidates[0]
                local_calls[caller].add(callee)
                if owner_class and callee.startswith("class "):
                    local_calls[owner_class].add(callee)

        for caller, callees in local_calls.items():
            for callee in callees:
                self.add_invocation(caller, callee, file_path, file_path)
                if _is_private_python_unit(callee):
                    for target in _public_targets_reached_via_private(callee, local_calls):
                        self.add_invocation(caller, target, file_path, file_path)

    def _analyze_python_type_references(
        self,
        code: str,
        file_path: str,
        base_class_files: Dict[str, str]
    ):
        """Add reference edges for Python parameter/return type annotations."""
        for unit in self.backend.list_code_units(code, file_path):
            if unit.unit_type not in ("function", "method"):
                continue
            node = (unit.extra or {}).get("ast_node")
            if node is None:
                continue
            func_name = unit.name
            for arg in getattr(node.args, "args", []):
                if arg.annotation is not None:
                    for t in _extract_type_names(arg.annotation):
                        if t in self.known_types:
                            type_file = base_class_files.get(t)
                            self.add_reference(
                                f"function {func_name}", t, file_path, type_file,
                            )
            if getattr(node, "returns", None) is not None:
                for t in _extract_type_names(node.returns):
                    if t in self.known_types:
                        type_file = base_class_files.get(t)
                        self.add_reference(
                            f"function {func_name}", t, file_path, type_file,
                        )
    
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

        # --- 4. Dedup: drop exact-duplicate invocation edges -------------
        # Two sources can emit the same edge: AST inspection
        # (`analyze_code_dependencies`) and LLM declarations
        # (`process_llm_dependencies`). Without this pass duplicates
        # silently inflate `incoming` counts and can mask real orphans.
        seen: Set[Tuple[str, str, Optional[str], Optional[str]]] = set()
        deduped: List[Dict[str, Any]] = []
        for edge in cleaned:
            key = (
                edge.get("caller", ""),
                edge.get("callee", ""),
                edge.get("caller_file"),
                edge.get("callee_file"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        removed_count = len(cleaned) - len(deduped)
        if removed_count:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "[DependencyCollector] post_process_edges deduped "
                "%d / %d invocation edges",
                removed_count, len(cleaned),
            )
        self.invocation_edges = deduped

    def to_dict(self) -> Dict[str, Any]:
        """Convert collected dependencies to dictionary.

        Performs a final in-place dedup on each edge bucket as a backstop:
        callers (especially ``_apply_fixes`` / ``_apply_add_interface``)
        sometimes append edges directly to the lists returned by an earlier
        ``to_dict()`` call (shared references) before re-feeding via
        ``add_invocation``. Without this pass, the same edge appended directly
        and then re-added via ``add_invocation`` could appear twice if the
        write-time dedup in ``add_invocation`` were ever bypassed.

        In-place mutation preserves shared list references so downstream
        callers holding the prior result still see the deduped contents.
        """
        inh_removed = _dedup_edges_in_place(
            self.inheritance_edges,
            ("child", "parent", "source_file", "parent_file"),
        )
        inv_removed = _dedup_edges_in_place(
            self.invocation_edges,
            ("caller", "callee", "caller_file", "callee_file"),
        )
        ref_removed = _dedup_edges_in_place(
            self.reference_edges,
            ("unit", "referenced_type", "source_file", "type_file"),
        )
        if inh_removed or inv_removed or ref_removed:
            logging.getLogger(__name__).info(
                f"[DependencyCollector.to_dict] Final dedup removed "
                f"{inh_removed} inheritance / {inv_removed} invocation / "
                f"{ref_removed} reference duplicate edges"
            )
        return {
            "original_edges": self.original_edges,
            "inheritance_edges": self.inheritance_edges,
            "invocation_edges": self.invocation_edges,
            "reference_edges": self.reference_edges,
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


def _iter_calls_in_own_scope(scope_node: ast.AST) -> "Iterator[ast.Call]":
    """Yield ``Call`` nodes inside ``scope_node`` without descending into
    nested ``def``/``class`` scopes, whose calls are attributed to their own
    unit. Lambdas are not separate units, so their calls remain attributed to
    the enclosing scope."""
    for child in ast.iter_child_nodes(scope_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_calls_in_own_scope(child)


def _python_call_name(node: ast.expr) -> Optional[str]:
    """Return a local callee name for safe same-file call edges."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and _attribute_root_is_self(node):
        return node.attr
    return None


def _attribute_root_is_self(node: ast.Attribute) -> bool:
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return isinstance(value, ast.Name) and value.id == "self"


def _is_private_python_unit(unit_name: str) -> bool:
    bare_name = unit_name.split(" ", 1)[-1]
    return bare_name.startswith("_")


def _public_targets_reached_via_private(
    start: str,
    local_calls: Dict[str, Set[str]],
) -> Set[str]:
    targets: Set[str] = set()
    seen: Set[str] = set()
    stack = list(local_calls.get(start, set()))
    while stack:
        unit_name = stack.pop()
        if unit_name in seen:
            continue
        seen.add(unit_name)
        if _is_private_python_unit(unit_name):
            stack.extend(local_calls.get(unit_name, set()))
        else:
            targets.add(unit_name)
    return targets


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
    
    def __init__(self, backend: Optional[LanguageBackend] = None):
        # Target-language backend for declaration/signature parsing.
        # Defaults to Python so standalone/legacy callers keep working.
        self.backend = backend or get_backend("python")
        # unit_name -> {file_path, subtree_name, unit_type, signature_summary, features}
        self.units: Dict[str, Dict[str, Any]] = {}
        # class_name -> file_path (for quick lookup)
        self.class_to_file: Dict[str, str] = {}
        # function_name -> file_path
        self.function_to_file: Dict[str, str] = {}
        # file_path -> list of unit info dicts
        self.file_units: Dict[str, List[Dict[str, Any]]] = {}
        self._logger = logging.getLogger(__name__)

    def _register_symbol(
        self,
        index: Dict[str, str],
        symbol_kind: str,
        name: str,
        file_path: str,
    ) -> None:
        """Assign ``index[name] = file_path``, warning if it silently overrides
        a different existing entry.

        Same-named top-level symbols across files are uncommon but legal
        (e.g. helper ``Repository`` / ``Manager`` per module). The current
        single-valued maps lose all but one — this helper at least makes
        the loss visible in logs so debuggers can spot ambiguous resolutions.
        A full multi-value fix is tracked separately.
        """
        existing = index.get(name)
        if existing and existing != file_path:
            self._logger.warning(
                "[GlobalInterfaceRegistry] %s name collision: '%s' already "
                "registered to %s, now overwritten by %s. Cross-subtree "
                "callee resolution may pick the wrong file.",
                symbol_kind, name, existing, file_path,
            )
        index[name] = file_path
    
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
                    self._register_symbol(self.class_to_file, "class", bare_name, file_path)
                elif unit_name.startswith("function "):
                    unit_type = "function"
                    bare_name = unit_name[len("function "):]
                    self._register_symbol(self.function_to_file, "function", bare_name, file_path)
                else:
                    unit_type = "unknown"
                    bare_name = unit_name
                
                # Extract a signature summary from the code (first non-import, non-blank line)
                signature_summary = self._extract_signature_summary(
                    code, unit_type, bare_name, self.backend
                )
                
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

    def register_unit(
        self,
        file_path: str,
        unit_name: str,
        subtree_name: str,
        features: Optional[List[str]] = None,
        signature_summary: str = "",
    ) -> bool:
        """Idempotently register a single new unit.

        Used by ``InterfaceReviewer._apply_add_interface`` to plug a handler-added
        interface into the registry without re-running the bulk
        ``register_from_subtree_result`` pass on an already-registered subtree.

        Returns:
            True if newly registered, False if a unit with the same
            ``(file_path, unit_name)`` already exists.
        """
        existing = self.units.get(unit_name)
        if existing is not None and existing.get("file_path") == file_path:
            return False

        if unit_name.startswith("class "):
            unit_type = "class"
            bare_name = unit_name[len("class "):]
            self._register_symbol(self.class_to_file, "class", bare_name, file_path)
        elif unit_name.startswith("function "):
            unit_type = "function"
            bare_name = unit_name[len("function "):]
            self._register_symbol(self.function_to_file, "function", bare_name, file_path)
        else:
            unit_type = "unknown"
            bare_name = unit_name

        unit_info = {
            "file_path": file_path,
            "subtree_name": subtree_name,
            "unit_type": unit_type,
            "bare_name": bare_name,
            "signature_summary": signature_summary,
            "features": list(features) if features else [],
        }
        self.units[unit_name] = unit_info
        self.file_units.setdefault(file_path, []).append(unit_info)
        return True

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
    def _extract_signature_summary(
        code: str, unit_type: str, bare_name: str, backend: LanguageBackend
    ) -> str:
        """Extract a concise signature summary from interface code.

        Declaration discovery routes through ``backend.list_code_units``
        and ``format_signature``. For Python, class summaries additionally
        read direct base-class names from the preserved ``ClassDef`` in
        ``unit.extra['ast_node']``; other backends omit bases gracefully.
        """
        if not code:
            return bare_name

        units = backend.list_code_units(code, "<signature>")
        if not units:
            return bare_name

        def _format_unit_summary(unit: Any) -> str:
            formatted = backend.format_signature(unit)
            if formatted and formatted != bare_name:
                return formatted
            unit_code = (getattr(unit, "code", "") or "").strip()
            for line in unit_code.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped
            return formatted or bare_name

        # Find the matching top-level declaration. Prefer exact
        # kind+name matches, then fall back to name-only matches for
        # non-Python backends whose parser labels differ from registry
        # prefixes (e.g. registry "class Foo" vs backend "struct Foo").
        target = next(
            (u for u in units
             if u.unit_type == unit_type and u.name == bare_name and u.parent is None),
            None,
        )
        if target is None:
            target = next(
                (u for u in units if u.name == bare_name and u.parent is None),
                None,
            )
            if target is None:
                return bare_name
            return _format_unit_summary(target)

        if unit_type == "function":
            return _format_unit_summary(target)

        # Class case: collect direct-child methods + format bases.
        # ``backend.list_code_units`` walks BFS so methods of this
        # class are those whose ``parent`` matches ``bare_name``;
        # source order is preserved within a single parent.
        method_units = [
            u for u in units
            if u.unit_type == "method" and u.parent == bare_name
        ]
        methods: List[str] = []
        for m in method_units:
            if not m.name.startswith("_") or m.name == "__init__":
                methods.append(backend.format_signature(m))

        bases_str = ""
        class_node = (target.extra or {}).get("ast_node")
        if class_node is not None and getattr(class_node, "bases", None):
            base_names = [_extract_name_from_node(b) for b in class_node.bases]
            base_names = [b for b in base_names if b]
            if base_names:
                bases_str = f"({', '.join(base_names)})"

        if methods:
            return f"{bare_name}{bases_str} [{', '.join(methods[:5])}]"
        if bases_str:
            return f"{bare_name}{bases_str}"
        if backend.name != "python":
            return _format_unit_summary(target)
        return bare_name


# ============================================================================
# Import Cross-Validation (A2)
# ============================================================================

def cross_validate_imports_vs_calls(
    code: str,
    file_path: str,
    declared_calls: List[str],
    global_registry: GlobalInterfaceRegistry,
    backend: LanguageBackend,
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

    # Import discovery routes through the target language backend.
    # ``list_imports`` returns one LPDependency per imported symbol;
    # ``extra["module"]`` holds the source module and ``extra["imported"]``
    # is present for ``from X import Y`` statements. Backends whose imports
    # do not populate these fields simply yield no warnings.
    for dep in backend.list_imports(code, file_path):
        extra = dep.extra or {}
        module = extra.get("module") or ""

        if "imported" in extra:
            # ``from <module> import <imported>`` — symbol is the
            # imported name. Aliases do not affect registry lookup.
            symbol = extra.get("imported") or ""
            imported_from = module
            message_suffix = (
                f"'{symbol}' is imported from '{module}' and is a known "
                f"interface in '{{resolved_file}}', but not declared in "
                f"dependencies.calls"
            )
        else:
            # ``import <module>`` — symbol is the last dotted segment
            # of the module path used for registry lookup.
            full_name = module
            symbol = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
            imported_from = full_name
            message_suffix = (
                f"'{symbol}' is imported and is a known interface in "
                f"'{{resolved_file}}', but not declared in dependencies.calls"
            )

        if not symbol:
            continue
        resolved_file = global_registry.resolve_callee(symbol)
        if not (resolved_file and resolved_file != file_path):
            continue
        if symbol in declared_set:
            continue
        warnings.append({
            "imported_symbol": symbol,
            "imported_from": imported_from,
            "resolved_file": resolved_file,
            "file_path": file_path,
            "message": message_suffix.format(resolved_file=resolved_file),
        })

    return warnings


# ============================================================================
# Validation Functions
# ============================================================================

def _unit_has_docstring(unit: Any) -> bool:
    """Return whether a parsed Python unit has a docstring."""
    docstring = getattr(unit, "docstring", None)
    if docstring:
        return True
    node = (getattr(unit, "extra", {}) or {}).get("ast_node")
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return bool(ast.get_docstring(node))
    return False


def _strip_markdown_code_fence(code: str) -> str:
    """Remove a full Markdown code fence around an interface snippet."""
    match = re.fullmatch(r"\s*```[A-Za-z0-9_+-]*\s*\n(.*?)\n```\s*", code, re.DOTALL)
    return f"{match.group(1)}\n" if match else code


def validate_interface(
    interface: Dict[str, Any],
    target_features: Set[str],
    covered_features: Set[str],
    backend: Optional[LanguageBackend] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a single interface definition using ParsedFile.
    
    Returns: (is_valid, error_message, parsed_info)
    """
    backend = backend or get_backend("python")
    raw_features = interface.get("features", [])
    features = list(raw_features) if isinstance(raw_features, list) else []
    code = _strip_markdown_code_fence(interface.get("code", ""))
    interface["code"] = code
    errors = []

    if target_features:
        invalid_features = sorted(set(features) - target_features)
        duplicate_features = sorted(set(features) & covered_features)
        filtered_features = [
            feature for feature in features
            if feature in target_features and feature not in covered_features
        ]
        if invalid_features or duplicate_features:
            warnings = interface.setdefault("_validation_warnings", [])
            if invalid_features:
                warnings.append(
                    "Ignored feature paths outside this file's target set: "
                    + ", ".join(invalid_features)
                )
            if duplicate_features:
                warnings.append(
                    "Ignored feature paths already covered by earlier interfaces: "
                    + ", ".join(duplicate_features)
                )
        features = filtered_features
        interface["features"] = features
    
    # Check features
    if not features:
        errors.append("Interface must cover at least one uncovered target feature")
    
    if backend.name == "python":
        code = re.sub(
            r'^(\s*(?:from|import)\s+)([\w\-]+(?:\.[\w\-]+)*)',
            lambda m: m.group(1) + m.group(2).replace('-', '_'),
            code,
            flags=re.MULTILINE,
        )
        interface["code"] = code

    ok, syntax_error = backend.syntax_check(code, f"temp_interface{backend.file_extension}")
    if not ok:
        errors.append(f"Syntax error: {syntax_error}")
        return False, "; ".join(errors), {}

    interface_units = [
        unit for unit in backend.list_code_units(code, f"temp_interface{backend.file_extension}")
        if unit.unit_type in [
            "function",
            "class",
            "struct",
            "interface",
            "method",
            "type",
            "enum",
        ]
        and (unit.parent is None or unit.unit_type == "method")
    ]

    if not interface_units:
        errors.append("No valid target-language declarations found in code")

    if backend.name == "python":
        for unit in interface_units:
            if not _unit_has_docstring(unit) and unit.unit_type in ["function", "class"]:
                errors.append(
                    f"Missing docstring for {unit.unit_type} '{unit.name}' "
                    f"in features {features}"
                )
    
    if errors:
        return False, "; ".join(errors), {}
    
    # Build parsed info with CodeUnit objects
    functions = [u.name for u in interface_units if u.unit_type in {"function", "method"}]
    classes = [
        u.name for u in interface_units
        if u.unit_type in {"class", "struct", "interface", "type", "enum"}
    ]
    declarations = [f"{u.unit_type} {u.name}" for u in interface_units]
    
    return True, "", {
        "functions": functions,
        "classes": classes,
        "declarations": declarations,
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
        system_prompt = with_language_directive(INTERFACE_PROMPT, self.backend)

        # Build user prompt
        features_str = "\n".join([f"- {f}" for f in file_features])
        
        user_prompt = f"""[Begin Iteration]
Design interfaces for file: `{file_path}`.

Requirements:
- ONLY cover the following feature paths:
{features_str}
- When calling `design_itfs_for_feature`, ONLY use feature paths listed above.
- Do NOT introduce new/unspecified feature paths.
- Define interfaces only (imports + target-language declaration stubs + target-language documentation).
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
                        interface,
                        target_features,
                        covered_features,
                        backend=self.backend,
                    )
                    
                    if is_valid:
                        # Add name field from parsed info
                        if info.get("declarations"):
                            interface["name"] = info["declarations"][0]
                        elif info.get("classes"):
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
        step_id: Optional[int] = None,
        target_language: Optional[str] = None,
    ):
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
            if trajectory is not None:
                self.llm.set_trajectory(trajectory, step_id)
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_backend(target_language)
    
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

        if self._should_use_c_family_verification_fallback(subtree_name):
            for file_path in file_order:
                state = file_states.get(file_path)
                if state is not None:
                    self._complete_remaining_c_family_features(file_path, state)
            return self._build_subtree_results(file_order, file_states)

        # Build system prompt (tool description is now integrated)
        system_prompt = with_language_directive(SUBTREE_INTERFACE_PROMPT, self.backend)

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
                            iface_dict,
                            target_features,
                            covered_features,
                            backend=self.backend,
                        )
                        
                        if is_valid:
                            # Add name from parsed info
                            if info.get("declarations"):
                                iface_dict["name"] = info["declarations"][0]
                            elif info.get("classes"):
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
        
        return self._build_subtree_results(file_order, file_states)

    def _build_subtree_results(
        self,
        file_order: List[str],
        file_states: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build final subtree results from accumulated file states."""
        results: Dict[str, Dict[str, Any]] = {}
        all_new_features: List[Dict[str, str]] = []

        for file_path in file_order:
            if file_path not in file_states:
                continue

            state = file_states[file_path]
            self._complete_remaining_c_family_features(file_path, state)
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

    def _should_use_c_family_verification_fallback(self, subtree_name: str) -> bool:
        """Return whether C-family verification interfaces should be deterministic."""
        if self.backend.name not in {"c", "cpp"}:
            return False
        normalized = subtree_name.casefold()
        return "verification" in normalized or "test" in normalized

    def _complete_remaining_c_family_features(
        self,
        file_path: str,
        state: Dict[str, Any],
    ) -> None:
        """Add deterministic C/C++ declarations for uncovered features."""
        if self.backend.name not in {"c", "cpp"}:
            return
        target_features = state.get("target_features", set())
        covered_features = state.get("covered_features", set())
        remaining_features = sorted(target_features - covered_features)
        if not remaining_features:
            return

        declaration_code = self._fallback_declaration_code(
            file_path=file_path,
            features=remaining_features,
        )
        interface = {
            "features": remaining_features,
            "code": declaration_code,
            "dependencies": {
                "inherits_from": [],
                "calls": [],
                "uses_types": [],
            },
        }
        is_valid, error, info = validate_interface(
            interface,
            target_features,
            covered_features,
            backend=self.backend,
        )
        if not is_valid:
            self.logger.warning(
                "[SubtreeInterfaceAgent] Deterministic %s completion failed for %s: %s",
                self.backend.display_name,
                file_path,
                error,
            )
            return

        if info.get("declarations"):
            interface["name"] = info["declarations"][0]
        elif info.get("classes"):
            interface["name"] = f"class {info['classes'][0]}"
        elif info.get("functions"):
            interface["name"] = f"function {info['functions'][0]}"
        interface["parsed_units"] = info.get("units", [])

        state["all_interfaces"].append(interface)
        state["all_code_blocks"].append(declaration_code)
        covered_features.update(interface.get("features", []))
        self.logger.info(
            "[SubtreeInterfaceAgent] Added deterministic %s interface for %s (%d feature%s)",
            self.backend.display_name,
            file_path,
            len(interface.get("features", [])),
            "" if len(interface.get("features", [])) == 1 else "s",
        )

    def _fallback_declaration_code(self, file_path: str, features: List[str]) -> str:
        """Return a parseable C-family declaration covering ``features``."""
        function_name = self._fallback_function_name(file_path, features)
        feature_lines = "\n".join(f" *   - {feature}" for feature in features)
        if self.backend.name == "c":
            return (
                "/**\n"
                " * Declares the remaining interface contract for:\n"
                f"{feature_lines}\n"
                " *\n"
                " * Returns:\n"
                " *   int status code supplied by the implementation.\n"
                " */\n"
                f"int {function_name}(void);\n"
            )
        return (
            "namespace tasklite {\n"
            "namespace generated {\n"
            "/// Declares the remaining interface contract for:\n"
            + "\n".join(f"/// - {feature}" for feature in features)
            + "\n"
            f"bool {function_name}();\n"
            "}  // namespace generated\n"
            "}  // namespace tasklite\n"
        )

    def _fallback_function_name(self, file_path: str, features: List[str]) -> str:
        """Build a stable C-family function name from file and feature paths."""
        path_stem = Path(file_path).stem
        feature_tail = "_".join(feature.rsplit("/", 1)[-1] for feature in features)
        raw_name = f"{path_stem}_{feature_tail}"
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_").lower()
        if not cleaned:
            cleaned = "generated_interface"
        if cleaned[:1].isdigit():
            cleaned = f"_{cleaned}"
        return self.backend.sanitize_module_identifier(cleaned)
    
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
                f"```{self.backend.markdown_fence}\n{code_preview}\n```"
            )
        
        completed_context = (
            "\n\n".join(completed_parts) if completed_parts
            else "No files designed yet in this subtree."
        )
        
        # Assemble user prompt
        # Detect import convention from file paths
        import_convention = ""
        if remaining_files and self.backend.name == "python":
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
        output_path: Optional[str] = None,
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
        self.max_file_iterations = max_file_iterations
        self.max_planning_retries = max_planning_retries
        self.logger = logger or logging.getLogger(__name__)
        self.trajectory = trajectory
        self.step_id = step_id
        self.output_path = output_path
        self.backend = get_backend(target_language)
    
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
        print(
            f"[InterfaceOrchestrator] Subtrees to process: {len(subtree_order)} "
            f"({', '.join(subtree_order)})",
            flush=True,
        )
        
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
        global_registry = GlobalInterfaceRegistry(backend=self.backend)

        # Track state across subtrees
        all_interfaces = {}
        implemented_subtrees = {}  # subtree -> list of implemented file info
        all_import_warnings = []  # collect import cross-validation warnings
        all_new_features = []  # collect new features created across all subtrees
        coverage_status = self._new_coverage_status()
        restored_subtrees = self._restore_completed_subtrees(
            skeleton=skeleton,
            subtree_order=subtree_order,
            all_interfaces=all_interfaces,
            implemented_subtrees=implemented_subtrees,
            coverage_status=coverage_status,
            global_registry=global_registry,
        )
        if restored_subtrees:
            restored_in_order = [name for name in subtree_order if name in restored_subtrees]
            print(
                f"[InterfaceOrchestrator] Restored completed subtrees: "
                f"{len(restored_subtrees)}/{len(subtree_order)} "
                f"({', '.join(restored_in_order)})",
                flush=True,
            )

        # Process each subtree
        for subtree_index, subtree_name in enumerate(subtree_order, start=1):
            if subtree_name in restored_subtrees:
                self.logger.info(
                    f"[InterfaceOrchestrator] Reusing completed subtree: {subtree_name}"
                )
                self._print_coverage_progress(
                    coverage_status,
                    len(all_interfaces),
                    len(subtree_order),
                )
                continue
            self.logger.info(f"[InterfaceOrchestrator] Processing subtree: {subtree_name}")
            
            # Find files for this subtree
            file_nodes = self._find_files_for_subtree(skeleton, subtree_name)
            if not file_nodes:
                self.logger.warning(f"No files found for subtree: {subtree_name}")
                self._record_missing_subtree(coverage_status, subtree_name)
                continue
            
            self.logger.info(f"[InterfaceOrchestrator] Found {len(file_nodes)} files for {subtree_name}")
            print(
                f"[InterfaceOrchestrator] Subtree {subtree_index}/{len(subtree_order)}: "
                f"{subtree_name} ({len(file_nodes)} files, "
                f"{self._subtree_feature_count(file_nodes)} features)",
                flush=True,
            )
            
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
                logger=self.logger,
                target_language=self.backend.name,
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

            for file_node in file_nodes:
                self._record_file_coverage(
                    coverage_status=coverage_status,
                    subtree_name=subtree_name,
                    file_node=file_node,
                    result=file_results.get(file_node.get("path", "")),
                )
            
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
                    global_registry=global_registry,
                    backend=self.backend,
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
            
            # Save resume data after each subtree without publishing a partial
            # interfaces.json as the final artifact.
            self._save_interfaces(
                self._build_result(
                    all_interfaces,
                    subtree_order,
                    implemented_subtrees,
                    coverage_status,
                ),
                partial=True,
            )
            self._print_coverage_progress(
                coverage_status,
                len(all_interfaces),
                len(subtree_order),
            )
        
        # Compile final result
        final_result = self._build_result(
            all_interfaces,
            subtree_order,
            implemented_subtrees,
            coverage_status,
        )

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
        implemented_subtrees: Dict[str, List[Dict[str, Any]]],
        coverage_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the result dict from current state."""
        coverage = coverage_status or self._new_coverage_status()
        return {
            "meta": {
                "primary_language": self.backend.name,
                "target_languages": [self.backend.name],
            },
            "subtrees": all_interfaces,
            "subtree_order": subtree_order,
            "implemented_subtrees": {
                st: [f["path"] for f in files]
                for st, files in implemented_subtrees.items()
            },
            "coverage": coverage,
            "success": not coverage.get("issues"),
        }

    @staticmethod
    def _new_coverage_status() -> Dict[str, Any]:
        """Return an empty coverage accumulator for interface generation."""
        return {
            "expected_files": 0,
            "successful_files": 0,
            "expected_features": 0,
            "covered_features": 0,
            "missing_features": 0,
            "failed_files": [],
            "missing_subtrees": [],
            "issues": [],
        }

    @staticmethod
    def _features_from_file_result(result: Dict[str, Any]) -> Set[str]:
        """Extract feature paths mapped by a generated file result."""
        features: Set[str] = set()
        for mapped_features in (result.get("units_to_features") or {}).values():
            if isinstance(mapped_features, list):
                features.update(str(feature) for feature in mapped_features)
            elif isinstance(mapped_features, str):
                features.add(mapped_features)
        return features

    @staticmethod
    def _record_missing_subtree(
        coverage_status: Dict[str, Any],
        subtree_name: str,
    ) -> None:
        """Record a subtree referenced by data flow but absent from skeleton."""
        coverage_status["missing_subtrees"].append(subtree_name)
        coverage_status["issues"].append({
            "subtree": subtree_name,
            "file_path": None,
            "reason": "subtree has no skeleton files",
            "missing_features": [],
        })

    @classmethod
    def _record_file_coverage(
        cls,
        coverage_status: Dict[str, Any],
        subtree_name: str,
        file_node: Dict[str, Any],
        result: Optional[Dict[str, Any]],
    ) -> None:
        """Record generated interface coverage for one skeleton file."""
        file_path = file_node.get("path", "")
        expected_features = set(file_node.get("feature_paths", []))
        if not expected_features:
            return

        coverage_status["expected_files"] += 1
        coverage_status["expected_features"] += len(expected_features)

        produced_features = cls._features_from_file_result(result or {})
        covered_features = expected_features & produced_features
        missing_features = sorted(expected_features - produced_features)
        has_units = bool(result and result.get("units"))

        coverage_status["covered_features"] += len(covered_features)
        coverage_status["missing_features"] += len(missing_features)

        if has_units and not missing_features:
            coverage_status["successful_files"] += 1
            return

        reason = "missing features"
        if not result:
            reason = "no result"
        elif not has_units:
            reason = "no units"

        coverage_status["failed_files"].append(file_path)
        coverage_status["issues"].append({
            "subtree": subtree_name,
            "file_path": file_path,
            "reason": reason,
            "missing_features": missing_features,
        })
    
    def _partial_output_path(self) -> Optional[Path]:
        if not self.output_path:
            return None
        return Path(f"{self.output_path}.partial")

    def _save_interfaces(self, result: Dict[str, Any], partial: bool = False) -> None:
        """Save current interfaces result to output_path (if configured).
        
        Strips internal keys (prefixed with '_') that contain non-serializable
        objects before writing to JSON.
        """
        if not self.output_path:
            return
        try:
            output = self._partial_output_path() if partial else Path(self.output_path)
            if output is None:
                return
            output.parent.mkdir(parents=True, exist_ok=True)
            # Filter out non-serializable internal keys
            serializable = {
                k: v for k, v in result.items()
                if not k.startswith("_")
            }
            with open(output, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[InterfaceOrchestrator] Saved interfaces to {output}")
            if not partial:
                partial_path = self._partial_output_path()
                if partial_path and partial_path.exists():
                    partial_path.unlink()
        except Exception as e:
            self.logger.warning(f"[InterfaceOrchestrator] Failed to save interfaces: {e}")

    @staticmethod
    def _subtree_feature_count(file_nodes: List[Dict[str, Any]]) -> int:
        """Return the number of distinct feature paths assigned to files."""
        features: Set[str] = set()
        for file_node in file_nodes:
            features.update(file_node.get("feature_paths", []))
        return len(features)

    @staticmethod
    def _print_coverage_progress(
        coverage_status: Dict[str, Any],
        processed_subtrees: int,
        total_subtrees: int,
    ) -> None:
        """Print compact progress for long-running interface generation."""
        expected_features = coverage_status.get("expected_features", 0)
        covered_features = coverage_status.get("covered_features", 0)
        issue_count = len(coverage_status.get("issues", []) or [])
        print(
            f"[InterfaceOrchestrator] Progress: {processed_subtrees}/{total_subtrees} "
            f"subtrees, {covered_features}/{expected_features} processed features "
            f"covered, issues={issue_count}",
            flush=True,
        )

    def _load_existing_interfaces(self) -> Optional[Dict[str, Any]]:
        """Load an existing interfaces file for subtree-level resume."""
        if not self.output_path:
            return None
        candidates = []
        partial_path = self._partial_output_path()
        if partial_path is not None:
            candidates.append(partial_path)
        candidates.append(Path(self.output_path))
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.logger.warning(
                f"[InterfaceOrchestrator] Failed to load existing interfaces: {exc}"
            )
            return None
        return data if isinstance(data, dict) else None

    def _restore_completed_subtrees(
        self,
        skeleton: Dict[str, Any],
        subtree_order: List[str],
        all_interfaces: Dict[str, Any],
        implemented_subtrees: Dict[str, List[Dict[str, Any]]],
        coverage_status: Dict[str, Any],
        global_registry: "GlobalInterfaceRegistry",
    ) -> Set[str]:
        """Restore a contiguous prefix of complete subtrees from output_path."""
        existing = self._load_existing_interfaces()
        if not existing:
            return set()

        restored: Set[str] = set()
        existing_subtrees = existing.get("subtrees") or {}
        if not isinstance(existing_subtrees, dict):
            return restored

        for subtree_name in subtree_order:
            subtree_data = existing_subtrees.get(subtree_name)
            if not isinstance(subtree_data, dict):
                break

            file_nodes = self._find_files_for_subtree(skeleton, subtree_name)
            file_container = subtree_data.get(
                "interfaces",
                subtree_data.get("files", {}),
            )
            if not isinstance(file_container, dict):
                break
            if not self._subtree_interfaces_complete(file_nodes, file_container):
                break

            all_interfaces[subtree_name] = {
                "files_order": subtree_data.get("files_order")
                or [node.get("path", "") for node in file_nodes],
                "interfaces": file_container,
            }
            implemented_subtrees[subtree_name] = self._implemented_files_from_existing(
                file_nodes,
                file_container,
            )
            for file_node in file_nodes:
                self._record_file_coverage(
                    coverage_status=coverage_status,
                    subtree_name=subtree_name,
                    file_node=file_node,
                    result=file_container.get(file_node.get("path", "")),
                )
            global_registry.register_from_subtree_result(subtree_name, file_container)
            restored.add(subtree_name)

        if restored:
            self.logger.info(
                f"[InterfaceOrchestrator] Restored {len(restored)} completed subtree(s): "
                f"{sorted(restored)}"
            )
        return restored

    @classmethod
    def _subtree_interfaces_complete(
        cls,
        file_nodes: List[Dict[str, Any]],
        file_container: Dict[str, Any],
    ) -> bool:
        """Return True when existing subtree interfaces cover all features."""
        file_container = {
            path: result for path, result in file_container.items()
            if path != "__new_features__"
        }

        expected_features: Set[str] = set()
        for file_node in file_nodes:
            expected_features.update(file_node.get("feature_paths", []))
        if not expected_features:
            return all(
                file_node.get("path", "") in file_container
                for file_node in file_nodes
            )

        produced_features: Set[str] = set()
        for result in file_container.values():
            if isinstance(result, dict):
                produced_features.update(cls._features_from_file_result(result))
        return expected_features <= produced_features

    @staticmethod
    def _implemented_files_from_existing(
        file_nodes: List[Dict[str, Any]],
        file_container: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build implemented_subtrees entries from restored interface data."""
        implemented: List[Dict[str, Any]] = []
        for file_node in file_nodes:
            file_path = file_node.get("path", "")
            result = file_container.get(file_path)
            if not isinstance(result, dict) or not result.get("units"):
                continue
            implemented.append({
                "path": file_path,
                "features": file_node.get("feature_paths", []),
                "code": result.get("file_code", ""),
                "units": result.get("units", []),
                "units_to_features": result.get("units_to_features", {}),
            })
        return implemented
    
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
            
            # Parse code through the target-language backend so declaration
            # discovery is shared with other interface-analysis paths.
            # Syntax errors yield an empty unit list.
            for unit in self.backend.list_code_units(code, file_path):
                if unit.unit_type == "class":
                    mapping[unit.name] = file_path
                elif unit.unit_type in ("function", "method"):
                    # Map every function-like name so nested callable
                    # declarations can still satisfy dependency lookups.
                    mapping[unit.name] = file_path
        
        # Process data structures (only those with file_path already assigned)
        if data_structures:
            for ds in data_structures:
                file_path = ds.get("file_path", "")
                code = ds.get("code", "")
                
                if not file_path or not code:
                    continue
                
                # Parse through the target-language backend to share class
                # discovery with interface dependency analysis. Syntax
                # errors yield an empty unit list.
                for unit in self.backend.list_code_units(code, file_path):
                    if unit.unit_type == "class":
                        mapping[unit.name] = file_path
                
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
