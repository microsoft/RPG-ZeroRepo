#!/usr/bin/env python3
"""RPG Updater - Update repo_rpg.json after code generation.

This module is responsible for updating the repo_rpg.json file after
code generation is complete. It:

1. Analyzes the generated code to extract actual dependencies
2. Checks node consistency (warns on mismatches)
3. Updates feature node metadata (marks implementation status)
4. Updates edges (INHERITS, INVOKES, REFERENCES) based on actual code

Node lookup strategy:
  RPG nodes are feature-level (e.g., "Parse CSV input"), NOT code-level.
  Code-level names (class/function) are stored in ``node.meta.path``
  (format: ``"src/file.py::class ClassName"``).  All lookups match
  against ``meta.path`` rather than ``node.name``.

Usage:
    Called automatically by ``run_batch.py`` after a batch completes successfully.
"""

import ast
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Import RPG models
from rpg import (
    RPG, Node, EdgeType, NodeMetaData, NodeType,
)
from rpg.code_unit import ParsedFile

# Import task batch
from common.task_batch import PlannedTask

# Constants
GENERATOR_NAME = "code_gen"
BACKUP_SUFFIX = ".backup"


# ============================================================================
# Dependency Extraction (similar to interface_agent.py)
# ============================================================================

def extract_name_from_node(node: ast.AST) -> Optional[str]:
    """Extract a name string from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        # e.g., module.ClassName -> return "ClassName"
        return node.attr
    elif isinstance(node, ast.Subscript):
        # e.g., List[int] -> return "List"
        return extract_name_from_node(node.value)
    return None


def extract_type_names(annotation: ast.AST) -> List[str]:
    """Extract all type names from a type annotation."""
    names = []
    if isinstance(annotation, ast.Name):
        names.append(annotation.id)
    elif isinstance(annotation, ast.Attribute):
        names.append(annotation.attr)
    elif isinstance(annotation, ast.Subscript):
        # Handle generic types like List[int], Dict[str, Any]
        names.extend(extract_type_names(annotation.value))
        if isinstance(annotation.slice, ast.Tuple):
            for elt in annotation.slice.elts:
                names.extend(extract_type_names(elt))
        else:
            names.extend(extract_type_names(annotation.slice))
    elif isinstance(annotation, ast.BinOp):
        # Handle Union types with | operator
        names.extend(extract_type_names(annotation.left))
        names.extend(extract_type_names(annotation.right))
    return names


def extract_function_calls(node: ast.AST) -> List[str]:
    """Extract function/method call names from an AST node."""
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func_name = extract_name_from_node(child.func)
            if func_name:
                calls.append(func_name)
    return calls


class CodeDependencyAnalyzer:
    """Analyzes Python code to extract dependencies.
    
    Extracts:
    - Inheritance relationships (class X(BaseClass))
    - Function/method invocations
    - Type references in annotations
    """
    
    def __init__(self, known_units: Set[str] = None):
        """Args: known_units: Set of known unit names (classes, functions) in the repo.

        Only dependencies to these units will be recorded.
        """
        self.known_units = known_units or set()
        self.inheritance_edges: List[Dict[str, Any]] = []
        self.invocation_edges: List[Dict[str, Any]] = []
        self.reference_edges: List[Dict[str, Any]] = []
    
    def analyze_file(self, file_path: Path, code: str) -> None:
        """Analyze a Python file for dependencies.
        
        Args:
            file_path: Path to the file
            code: Source code content
        """
        # Python-AST based: dependencies for other languages come from the
        # tree-sitter dependency graph. Skip non-Python sources quietly rather
        # than emitting a misleading "SyntaxError parsing" warning on every
        # generated .rs/.js/.go/... file.
        if Path(file_path).suffix.lower() != ".py":
            return
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logging.warning(f"SyntaxError parsing {file_path}: {e}")
            return
        
        file_path_str = str(file_path)
        
        for node in ast.walk(tree):
            # Extract inheritance
            if isinstance(node, ast.ClassDef):
                self._analyze_class(node, file_path_str)
            
            # Extract function-level dependencies
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(node, file_path_str)
    
    def _analyze_class(self, cls_node: ast.ClassDef, file_path: str) -> None:
        """Analyze a class definition for inheritance."""
        class_name = cls_node.name
        
        for base in cls_node.bases:
            parent_name = extract_name_from_node(base)
            if parent_name and self._is_known_unit(parent_name):
                self.inheritance_edges.append({
                    "child": class_name,
                    "parent": parent_name,
                    "source_file": file_path,
                    "edge_type": EdgeType.INHERITS,
                    "generator": GENERATOR_NAME
                })
        
        # Analyze methods within the class
        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(
                    item, file_path, 
                    parent_class=class_name
                )
    
    def _analyze_function(
        self, 
        func_node: ast.FunctionDef, 
        file_path: str,
        parent_class: Optional[str] = None
    ) -> None:
        """Analyze a function/method for dependencies."""
        if parent_class:
            unit_name = f"{parent_class}::{func_node.name}"
        else:
            unit_name = func_node.name
        
        # Extract type references from annotations
        for arg in func_node.args.args:
            if arg.annotation:
                for type_name in extract_type_names(arg.annotation):
                    if self._is_known_unit(type_name):
                        self.reference_edges.append({
                            "unit": unit_name,
                            "referenced_type": type_name,
                            "source_file": file_path,
                            "edge_type": EdgeType.REFERENCES,
                            "generator": GENERATOR_NAME
                        })
        
        # Check return type
        if func_node.returns:
            for type_name in extract_type_names(func_node.returns):
                if self._is_known_unit(type_name):
                    self.reference_edges.append({
                        "unit": unit_name,
                        "referenced_type": type_name,
                        "source_file": file_path,
                        "edge_type": EdgeType.REFERENCES,
                        "generator": GENERATOR_NAME
                    })
        
        # Extract function calls
        for call_name in extract_function_calls(func_node):
            if self._is_known_unit(call_name) and call_name != func_node.name:
                self.invocation_edges.append({
                    "caller": unit_name,
                    "callee": call_name,
                    "source_file": file_path,
                    "edge_type": EdgeType.INVOKES,
                    "generator": GENERATOR_NAME
                })
    
    def _is_known_unit(self, name: str) -> bool:
        """Check if a name is a known unit in the repo."""
        if not self.known_units:
            return True  # If no filter, accept all
        return name in self.known_units
    
    def get_all_edges(self) -> Dict[str, List[Dict]]:
        """Get all extracted edges."""
        return {
            "inheritance": self.inheritance_edges,
            "invocation": self.invocation_edges,
            "reference": self.reference_edges
        }


# ============================================================================
# Node Consistency Checker
# ============================================================================

class NodeConsistencyChecker:
    """Checks consistency between generated code and repo_rpg nodes.
    
    Reports warnings when:
    - Code defines units not matching expected node names
    - Unit types don't match (class vs function)
    - Implementation paths don't match node.meta.path
    """
    
    def __init__(self, rpg: RPG):
        self.rpg = rpg
        self.warnings: List[str] = []
    
    def check_batch_consistency(
        self,
        batch: PlannedTask,
        parsed_file: ParsedFile,
        repo_path: Path
    ) -> List[str]:
        """Check consistency for a completed batch.
        
        Args:
            batch: The completed PlannedTask
            parsed_file: Parsed representation of the generated file
            repo_path: Repository root path
            
        Returns:
            List of warning messages
        """
        self.warnings = []
        
        # Get expected units from batch
        expected_units = set(batch.units_key)
        
        # Get actual units from parsed file
        actual_units = {}
        for unit in parsed_file.units:
            if unit.unit_type in ("class", "function", "method"):
                key = unit.name
                if unit.parent:
                    key = f"{unit.parent}::{unit.name}"
                actual_units[key] = unit
        
        # Check for expected units not found in code
        for expected in expected_units:
            # Normalize expected name (remove "class " or "function " prefix)
            normalized = expected
            if expected.startswith("class "):
                normalized = expected[6:]
            elif expected.startswith("function "):
                normalized = expected[9:]
            
            found = False
            for actual_key in actual_units:
                if normalized in actual_key or actual_key in normalized:
                    found = True
                    break
            
            if not found:
                self.warnings.append(
                    f"[WARNING]  Expected unit '{expected}' not found in generated code for {batch.file_path}"
                )
        
        # Check node consistency in RPG
        for unit_key in batch.units_key:
            features = batch.unit_to_features.get(unit_key, [])
            for feature_path in features:
                node = self._find_feature_node(feature_path)
                if node:
                    self._check_node_path_consistency(node, batch.file_path, unit_key)
        
        return self.warnings
    
    def _find_feature_node(self, feature_path: str) -> Optional[Node]:
        """Find a feature node by name or path."""
        # Try direct lookup by name
        feature_name = feature_path.split("/")[-1] if "/" in feature_path else feature_path
        
        for node in self.rpg.nodes.values():
            if node.name == feature_name:
                return node
            if node.node_type == "feature" and node.feature_path() == feature_path:
                return node
        return None
    
    def _check_node_path_consistency(
        self, 
        node: Node, 
        file_path: str, 
        unit_key: str
    ) -> None:
        """Check if node's meta.path is consistent with implementation."""
        if not node.meta or not node.meta.path:
            return
        
        expected_path = f"{file_path}::{unit_key}"
        actual_path = node.meta.path
        
        # Normalize paths for comparison
        if "::" in actual_path:
            # Compare file parts
            actual_file = actual_path.split("::")[0]
            expected_file = file_path
            
            if actual_file != expected_file:
                self.warnings.append(
                    f"[WARNING]  Node '{node.name}' has path '{actual_path}' but "
                    f"code is in '{file_path}'"
                )


# ============================================================================
# Edge Updater
# ============================================================================

class EdgeUpdater:
    """Updates edges in repo_rpg based on analyzed code dependencies.
    
    Handles:
    - Adding new edges discovered in code
    - Removing stale edges (generated by code_gen but no longer in code)
    - Preserving edges from other generators
    
    Uses RPGService for all edge operations (dedup, generator tagging).
    """
    
    def __init__(self, rpg: RPG):
        self.rpg = rpg
        self._svc = None
        self.added_count: int = 0
        self.removed_count: int = 0

    @property
    def svc(self):
        if self._svc is None:
            from rpg.service import RPGService
            self._svc = RPGService(self.rpg)
        return self._svc
    
    def update_edges(
        self,
        analyzed_deps: Dict[str, List[Dict]],
        batch_file_path: str
    ) -> Tuple[int, int]:
        """Update RPG edges based on analyzed dependencies.
        
        Args:
            analyzed_deps: Dependencies from CodeDependencyAnalyzer
            batch_file_path: File path of the batch being completed
            
        Returns:
            Tuple of (edges_added, edges_removed)
        """
        # Step 1: Remove old edges from this file generated by code_gen
        self.removed_count = self.svc.refresh_file_edges(GENERATOR_NAME, batch_file_path)
        
        # Step 2: Add new edges from analysis
        self._add_new_edges(analyzed_deps)
        
        return self.added_count, self.removed_count
    
    def _add_new_edges(self, analyzed_deps: Dict[str, List[Dict]]) -> None:
        """Add new edges from analyzed dependencies."""
        # Process inheritance edges
        for dep in analyzed_deps.get("inheritance", []):
            self._add_edge_if_nodes_exist(
                child_name=dep["child"],
                parent_name=dep["parent"],
                edge_type=EdgeType.INHERITS,
                description=f"{dep['child']} inherits from {dep['parent']} (in {dep['source_file']})"
            )
        
        # Process invocation edges
        for dep in analyzed_deps.get("invocation", []):
            self._add_edge_if_nodes_exist(
                child_name=dep["caller"],
                parent_name=dep["callee"],
                edge_type=EdgeType.INVOKES,
                description=f"{dep['caller']} invokes {dep['callee']} (in {dep['source_file']})"
            )
        
        # Process reference edges
        for dep in analyzed_deps.get("reference", []):
            self._add_edge_if_nodes_exist(
                child_name=dep["unit"],
                parent_name=dep["referenced_type"],
                edge_type=EdgeType.REFERENCES,
                description=f"{dep['unit']} references {dep['referenced_type']} (in {dep['source_file']})"
            )
    
    def _add_edge_if_nodes_exist(
        self,
        child_name: str,
        parent_name: str,
        edge_type: EdgeType,
        description: str
    ) -> None:
        """Add an edge if both source and destination nodes exist."""
        src_node = self.svc.find_node_by_unit_name(child_name)
        dst_node = self.svc.find_node_by_unit_name(parent_name)
        
        if not src_node or not dst_node:
            return
        
        was_added = self.svc.add_dependency_edge(
            src_node, dst_node, edge_type, GENERATOR_NAME,
            description=description,
        )
        if was_added:
            self.added_count += 1


# ============================================================================
# Feature Node Updater
# ============================================================================

def _update_feature_nodes(
    rpg: RPG,
    batch: PlannedTask,
    parsed_file: ParsedFile,
    file_path_str: str,
    code: str,
) -> int:
    """Update feature-node metadata after successful code generation.

    For each unit in the batch, locate the corresponding RPG feature node
    (via ``unit_to_features``) and update:

    * ``meta.description``  – append "[implemented]" marker if not present
    * ``meta.content``      – replace interface skeleton with actual source

    Args:
        rpg: Loaded RPG graph
        batch: The completed PlannedTask
        parsed_file: The parsed representation of the generated file
        file_path_str: Relative file path (e.g. ``"src/parser.py"``)
        code: Full source code of the generated file

    Returns:
        Number of nodes that were updated.
    """
    updated = 0

    # Build a lookup: bare unit name -> actual source text from parsed file
    actual_source_by_name: Dict[str, str] = {}
    for cu in parsed_file.units:
        key = cu.name
        if cu.parent:
            key = f"{cu.parent}::{cu.name}"
        actual_source_by_name[key] = cu.source if hasattr(cu, "source") else ""

    for unit_key in batch.units_key:
        feature_paths = batch.unit_to_features.get(unit_key, [])
        if not feature_paths:
            continue

        # Build the impl_path as design_interfaces does:
        #   "src/file.py::class Foo" or "src/file.py::function bar"
        impl_path = f"{file_path_str}::{unit_key}"

        for feature_ref in feature_paths:
            # feature_ref can be a feature path string or a dict with "path" key
            if isinstance(feature_ref, dict):
                fp = feature_ref.get("path", feature_ref.get("name", ""))
            else:
                fp = str(feature_ref)

            # Find the feature node – try meta.path first, then name
            target_node = _find_feature_node_by_path(rpg, impl_path, fp)
            if target_node is None:
                continue

            if target_node.meta is None:
                target_node.meta = NodeMetaData()

            # --- Mark as implemented in description ---
            desc = target_node.meta.description or ""
            if "[implemented]" not in desc:
                target_node.meta.description = (
                    f"{desc} [implemented]" if desc else "[implemented]"
                )

            # --- Store actual source snippet ---
            # Try to find the matching source by unit key
            source_text = ""
            # Normalize unit_key for lookup: "class Foo" -> "Foo"
            bare_key = unit_key
            if unit_key.startswith("class "):
                bare_key = unit_key[6:]
            elif unit_key.startswith("function "):
                bare_key = unit_key[9:]

            source_text = actual_source_by_name.get(bare_key, "")
            if not source_text:
                # Try with original key
                source_text = actual_source_by_name.get(unit_key, "")

            if source_text:
                target_node.meta.content = source_text

            updated += 1

    if updated:
        logging.info(f"Updated {updated} feature nodes for {file_path_str}")
    return updated


def _find_feature_node_by_path(
    rpg: RPG,
    impl_path: str,
    feature_ref: str,
) -> Optional[Node]:
    """Locate a feature node by implementation path or feature reference.

    Search order:
    1. ``meta.path == impl_path``  (most reliable)
    2. ``node.name`` or ``feature_path()`` matching *feature_ref*
    """
    # 1. Match by meta.path
    for node in rpg.nodes.values():
        if node.meta and node.meta.path:
            p = node.meta.path if isinstance(node.meta.path, str) else ""
            if p == impl_path:
                return node

    # 2. Match by feature reference (name / feature_path)
    if feature_ref:
        feature_name = feature_ref.split("/")[-1] if "/" in feature_ref else feature_ref
        for node in rpg.nodes.values():
            if node.name == feature_name:
                return node
            if node.name == feature_ref:
                return node
            try:
                if node.feature_path() == feature_ref:
                    return node
            except Exception:
                pass

    return None


# ============================================================================
# Main Entry Point
# ============================================================================

def backup_rpg_file(rpg_path: Path) -> Optional[Path]:
    """Backup the repo_rpg.json file before modification.
    
    Args:
        rpg_path: Path to repo_rpg.json
        
    Returns:
        Path to backup file, or None if backup failed
    """
    if not rpg_path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = rpg_path.parent / f"repo_rpg_{timestamp}{BACKUP_SUFFIX}.json"
    
    try:
        shutil.copy2(rpg_path, backup_path)
        logging.info(f"Backed up repo_rpg.json to {backup_path}")
        return backup_path
    except Exception as e:
        logging.warning(f"Failed to backup repo_rpg.json: {e}")
        return None


def collect_known_units(rpg: RPG) -> Set[str]:
    """Collect all known unit names from RPG for filtering.

    RPG nodes are feature-level; code-level identifiers live in
    ``node.meta.path`` (canonical form: ``"src/file.py::Foo"`` for classes,
    ``"src/file.py::bar"`` for functions, ``"src/file.py::Foo::m"`` for
    methods).  We extract the unit name from those paths so the
    ``CodeDependencyAnalyzer`` can filter against real identifiers.

    The returned set also includes legacy "class Foo" / "function bar"
    forms so downstream callers that still pass type-prefixed identifiers
    continue to match while the rest of the codebase migrates.
    """
    known: Set[str] = set()
    for node in rpg.nodes.values():
        if not node.meta or not node.meta.path:
            continue
        path_str = node.meta.path if isinstance(node.meta.path, str) else ""
        if "::" not in path_str:
            continue
        # Canonical form: ``file::Name`` or ``file::Class::method``.
        # We add every non-empty segment after the file boundary so both
        # class names and method names become matchable.
        _file, _sep, sym_chain = path_str.partition("::")
        segments = [s for s in sym_chain.split("::") if s]
        for seg in segments:
            # Strip legacy ``class ``/``function ``/``method `` prefix
            # if it slipped through from older encoder runs.
            bare = seg
            for legacy_prefix in ("class ", "function ", "method "):
                if bare.startswith(legacy_prefix):
                    bare = bare[len(legacy_prefix):]
                    break
            known.add(bare)

        # Backward-compat: also emit ``"class Foo"`` / ``"function bar"``
        # for downstream code that has not yet migrated to bare names.
        if node.meta.type_name == NodeType.CLASS and segments:
            known.add(f"class {segments[-1]}")
        elif node.meta.type_name == NodeType.FUNCTION and segments:
            known.add(f"function {segments[-1]}")
        elif node.meta.type_name == NodeType.METHOD and segments:
            known.add(f"method {segments[-1]}")

    # Also include base-class nodes added by design_base_classes
    # (their node.name IS the class name).
    for node in rpg.nodes.values():
        if node.meta and node.meta.generator == "design_base_classes":
            known.add(node.name)
    return known


def run_rpg_update(
    batch: PlannedTask,
    repo_path: Path,
    rpg_path: Path,
    backup: bool = True
) -> Dict[str, Any]:
    """Main entry point for updating repo_rpg after code generation.
    
    Args:
        batch: The completed PlannedTask
        repo_path: Repository root path
        rpg_path: Path to repo_rpg.json
        backup: Whether to backup before modification
        
    Returns:
        Dict with update results:
        - success: bool
        - warnings: List[str]
        - edges_added: int
        - edges_removed: int
        - backup_path: Optional[str]
    """
    result = {
        "success": False,
        "warnings": [],
        "edges_added": 0,
        "edges_removed": 0,
        "nodes_updated": 0,
        "backup_path": None
    }
    
    # Check if RPG file exists
    if not rpg_path.exists():
        result["warnings"].append(f"RPG file not found: {rpg_path}")
        return result
    
    # Load RPG
    try:
        rpg = RPG.load_json(str(rpg_path))
    except Exception as e:
        result["warnings"].append(f"Failed to load RPG: {e}")
        return result
    
    # Get the generated file path
    file_path = repo_path / batch.file_path
    
    if not file_path.exists():
        result["warnings"].append(f"Generated file not found: {file_path}")
        return result
    
    # Read and parse the generated code
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        parsed_file = ParsedFile(code, str(file_path))
    except Exception as e:
        result["warnings"].append(f"Failed to parse generated file: {e}")
        return result
    
    # Step 1: Check node consistency
    consistency_checker = NodeConsistencyChecker(rpg)
    warnings = consistency_checker.check_batch_consistency(batch, parsed_file, repo_path)
    result["warnings"].extend(warnings)

    # Step 2: Update feature node metadata (mark as implemented)
    nodes_updated = _update_feature_nodes(
        rpg=rpg,
        batch=batch,
        parsed_file=parsed_file,
        file_path_str=batch.file_path,
        code=code,
    )
    result["nodes_updated"] = nodes_updated
    
    # Step 3: Analyze code dependencies
    known_units = collect_known_units(rpg)
    analyzer = CodeDependencyAnalyzer(known_units)
    # Pass the repo-relative path (batch.file_path) instead of the absolute
    # filesystem path. The path is only used to populate ``source_file`` in
    # edge metadata, which feeds into edge ``description`` text injected into
    # LLM prompts. Absolute paths leak host-specific prefixes
    # (e.g. /home/.../CoderMind-backup/...) and mislead agents.
    analyzer.analyze_file(Path(batch.file_path), code)
    analyzed_deps = analyzer.get_all_edges()
    
    # Step 4: Backup before modification
    if backup:
        backup_path = backup_rpg_file(rpg_path)
        if backup_path:
            result["backup_path"] = str(backup_path)
    
    # Step 5: Update edges
    edge_updater = EdgeUpdater(rpg)
    edges_added, edges_removed = edge_updater.update_edges(
        analyzed_deps, 
        batch.file_path
    )
    result["edges_added"] = edges_added
    result["edges_removed"] = edges_removed
    
    # Step 6: Save updated RPG
    try:
        rpg.save_json(str(rpg_path))
        result["success"] = True
        logging.info(
            f"RPG updated: {nodes_updated} nodes, "
            f"+{edges_added} edges, -{edges_removed} edges, "
            f"{len(result['warnings'])} warnings"
        )
    except Exception as e:
        result["warnings"].append(f"Failed to save RPG: {e}")
    
    return result
