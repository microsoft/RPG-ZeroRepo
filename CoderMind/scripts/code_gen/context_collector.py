#!/usr/bin/env python3
"""Context Collector for Code Generation.

Collects dependency and design context from earlier pipeline stages
(repo_rpg, data_flow, base_classes, interfaces) and provides it to
code generation prompts so that the agent understands how the current
batch relates to the rest of the project.

Also handles writing interface skeletons to actual source files so that
they are visible to the agent during implementation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from common.import_normalizer import (
    detect_project_import_prefix,
    normalize_code,
    ensure_future_annotations,
    fix_missing_stdlib_imports,
)
from common.language_meta import extract_language_metadata
from common.utils import get_project_background_context
from decoder_lang import get_backend

if TYPE_CHECKING:
    from common.task_batch import PlannedTask
    from common.execution_state import CodeGenState

logger = logging.getLogger(__name__)


# ============================================================================
# Interface Skeleton Writer
# ============================================================================

def write_interface_skeletons(
    interfaces_path: Path,
    repo_path: Path
) -> Dict[str, Any]:
    """Write interface skeletons from interfaces.json to actual source files.

    For each file in interfaces.json that has a ``file_code`` entry, this
    function writes it to disk **only if** the file does not already exist
    or the existing content is shorter than the skeleton (meaning the
    skeleton is more informative).

    Import prefixes are automatically normalized based on the project's
    source layout (e.g. ``from vibeanim.`` → ``from src.vibeanim.``).

    Args:
        interfaces_path: Path to interfaces.json
        repo_path: Root path of the target repository

    Returns:
        {"written": [file_paths…], "skipped": [file_paths…]}
    """
    result: Dict[str, List[str]] = {"written": [], "skipped": []}

    if not interfaces_path.exists():
        logger.warning("interfaces.json not found at %s", interfaces_path)
        return result

    try:
        with open(interfaces_path, "r", encoding="utf-8") as f:
            interfaces = json.load(f)
    except Exception as e:
        logger.error("Failed to read interfaces.json: %s", e)
        return result

    subtrees = interfaces.get("subtrees", {})
    primary_language, _ = extract_language_metadata(interfaces)
    backend = get_backend(primary_language)

    import_prefix = ""
    if backend.name == "python":
        import_prefix = detect_project_import_prefix(
            interfaces_subtrees=subtrees,
        )

    for _subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", {})
        for file_path, file_info in file_interfaces.items():
            file_code = file_info.get("file_code", "")
            if not file_code or not file_code.strip():
                continue

            if backend.name == "python" and import_prefix:
                file_code = normalize_code(file_code, import_prefix)

            if backend.name == "python":
                file_code = ensure_future_annotations(file_code)
                file_code = fix_missing_stdlib_imports(file_code)

            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    existing = full_path.read_text(encoding="utf-8")
                except Exception:
                    existing = ""
                # Skip if the file already has more content than the skeleton
                if len(existing.strip()) > len(file_code.strip()):
                    result["skipped"].append(file_path)
                    continue

            # Write skeleton
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(file_code, encoding="utf-8")
                result["written"].append(file_path)
            except Exception as e:
                logger.error("Failed to write skeleton for %s: %s", file_path, e)

    logger.info(
        "Interface skeletons: wrote %d files, skipped %d files",
        len(result["written"]),
        len(result["skipped"]),
    )
    return result


# ============================================================================
# Base Classes & Data Structures
# ============================================================================

def collect_base_classes_context(
    base_classes_path: Path,
    current_subtree: str
) -> Dict[str, Any]:
    """Collect base-class code (all) and data-structure code (current subtree only).

    Args:
        base_classes_path: Path to base_classes.json
        current_subtree: Name of the current subtree/functional area

    Returns:
        {
            "base_classes": [{"file_path": …, "code": …, "subclasses": …}, …],
            "data_structures": [{"code": …, "subtree": …, "data_flow_types": …}, …]
        }
    """
    result: Dict[str, list] = {"base_classes": [], "data_structures": []}

    if not base_classes_path.exists():
        return result

    try:
        with open(base_classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return result

    # All base classes — no filtering
    for bc in data.get("base_classes", []):
        result["base_classes"].append({
            "file_path": bc.get("file_path", ""),
            "code": bc.get("code", ""),
            "subclasses": bc.get("subclasses", {}),
        })

    # Data structures — only those matching current_subtree
    for ds in data.get("data_structures", []):
        if ds.get("subtree", "") == current_subtree:
            result["data_structures"].append({
                "code": ds.get("code", ""),
                "subtree": ds.get("subtree", ""),
                "data_flow_types": ds.get("data_flow_types", []),
                "file_path": ds.get("file_path", ""),
            })

    return result


# ============================================================================
# Data Flow Edges
# ============================================================================

def collect_data_flow_edges(
    data_flow_path: Path,
    current_subtree: str
) -> List[Dict[str, str]]:
    """Return data-flow edges involving *current_subtree* (as source or target).

    Args:
        data_flow_path: Path to data_flow.json
        current_subtree: Name of the current subtree

    Returns:
        List of edge dicts (original JSON shape, unmodified).
    """
    if not data_flow_path.exists():
        return []

    try:
        with open(data_flow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    edges = data.get("data_flow", [])
    return [
        e for e in edges
        if e.get("source") == current_subtree
        or e.get("target") == current_subtree
    ]


def collect_all_data_flow_edges(
    data_flow_path: Path,
) -> List[Dict[str, str]]:
    """Return ALL data-flow edges (no subtree filter).

    Used by wiring tasks that need a global view of cross-module connections.

    Args:
        data_flow_path: Path to data_flow.json

    Returns:
        List of all edge dicts.
    """
    if not data_flow_path.exists():
        return []

    try:
        with open(data_flow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    return data.get("data_flow", [])


# ============================================================================
# Dependency Files (from enhanced_data_flow in interfaces.json)
# ============================================================================

def collect_dependency_files(
    interfaces_path: Path,
    file_path: str
) -> Dict[str, Any]:
    """Identify files that the current file depends on, using the ``enhanced_data_flow`` section of interfaces.json.

    Args:
        interfaces_path: Path to interfaces.json
        file_path: The target file path for the current batch

    Returns:
        {
            "inherits_from": [{"parent": …, "parent_file": …}, …],
            "invokes":       [{"callee": …, "callee_file": …}, …],
            "references":    [{"type": …, "type_file": …}, …],
            "dependent_files": [sorted unique file paths]
        }
    """
    result: Dict[str, Any] = {
        "inherits_from": [],
        "invokes": [],
        "references": [],
        "dependent_files": [],
    }

    if not interfaces_path.exists():
        return result

    try:
        with open(interfaces_path, "r", encoding="utf-8") as f:
            interfaces = json.load(f)
    except Exception:
        return result

    edf = interfaces.get("enhanced_data_flow", {})
    dep_files: set = set()

    # Inheritance edges: source_file == file_path → depends on parent_file
    for edge in edf.get("inheritance_edges", []):
        if edge.get("source_file") == file_path and edge.get("parent_file"):
            result["inherits_from"].append({
                "parent": edge.get("parent", ""),
                "parent_file": edge["parent_file"],
            })
            dep_files.add(edge["parent_file"])

    # Invocation edges: caller_file == file_path → depends on callee_file
    for edge in edf.get("invocation_edges", []):
        if edge.get("caller_file") == file_path and edge.get("callee_file"):
            result["invokes"].append({
                "callee": edge.get("callee", ""),
                "callee_file": edge["callee_file"],
            })
            dep_files.add(edge["callee_file"])

    # Reference edges: source_file == file_path → depends on type_file
    for edge in edf.get("reference_edges", []):
        if edge.get("source_file") == file_path and edge.get("type_file"):
            result["references"].append({
                "type": edge.get("referenced_type", ""),
                "type_file": edge["type_file"],
            })
            dep_files.add(edge["type_file"])

    # Remove self-references
    dep_files.discard(file_path)
    result["dependent_files"] = sorted(dep_files)

    return result


# ============================================================================
# Completed Modules
# ============================================================================

def collect_completed_context(
    completed_task_ids: List[str],
    tasks_path: Path
) -> Dict[str, List[str]]:
    """Build a mapping of already-completed files → unit lists.

    Args:
        completed_task_ids: List of completed task IDs from CodeGenState
        tasks_path: Path to tasks.json

    Returns:
        {"src/core/parser.py": ["class Parser", "function tokenize"], …}
    """
    if not tasks_path.exists() or not completed_task_ids:
        return {}

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    completed_set = set(completed_task_ids)
    file_units: Dict[str, List[str]] = {}

    for _subtree, files_dict in data.get("planned_tasks_dict", {}).items():
        for _file_path, batches_list in files_dict.items():
            for batch_data in batches_list:
                if batch_data.get("task_id") in completed_set:
                    fp = batch_data.get("file_path", _file_path)
                    units = batch_data.get("units_key", [])
                    if fp not in file_units:
                        file_units[fp] = []
                    file_units[fp].extend(units)

    return file_units


# ============================================================================
# ORM Model Registry (cross-subtree relationship awareness)
# ============================================================================

def scan_orm_model_registry(
    interfaces_path: Path,
    repo_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan interfaces.json for ORM model classes and their relationship() targets to build a model registry with cross-file dependencies.

    This solves the SQLAlchemy mapper configuration problem: when any test
    instantiates a model, SQLAlchemy eagerly configures ALL mappers in the
    registry.  If model A has ``relationship('B')``, class B must be
    imported (even if unused in the test) before mapper configuration runs.

    Returns:
        {
            "models": {"User": "src/.../models.py", ...},
            "relationships": [
                {"source_file": ..., "source_class": ...,
                 "target_class": ..., "target_file": ..., "field": ...},
            ],
            "model_files": ["src/.../models.py", ...]  # sorted, deduped
        }
    Returns empty dict if no ORM models are detected.
    """
    import ast as _ast

    models: Dict[str, str] = {}       # class_name -> file_path
    relationships: List[Dict] = []
    model_files_set: set = set()
    seen_rels: set = set()  # dedup key: (source_class, field, target_class)

    # --- Strategy 1: scan interfaces.json file_code blocks ---
    if interfaces_path and interfaces_path.exists():
        try:
            with open(interfaces_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        for _subtree, subtree_data in data.get("subtrees", {}).items():
            for file_path, file_info in subtree_data.get("interfaces", {}).items():
                code = file_info.get("file_code", "")
                if not code:
                    continue
                _scan_code_for_models(
                    _ast, code, file_path, models, relationships,
                    model_files_set, seen_rels,
                )

    # --- Strategy 2: if repo_path given, scan actual **/models*.py files ---
    # Catches models added during codegen (not in skeleton) and handles
    # projects where model files aren't named models.py.
    if repo_path and repo_path.is_dir():
        src_dir = repo_path / "src"
        search_dir = src_dir if src_dir.is_dir() else repo_path
        for py_file in search_dir.rglob("model*.py"):
            rel_path = str(py_file.relative_to(repo_path))
            try:
                code = py_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            _scan_code_for_models(
                _ast, code, rel_path, models, relationships,
                model_files_set, seen_rels,
            )

    if not models:
        return {}

    # Resolve target_file for relationships
    for rel in relationships:
        if not rel.get("target_file"):
            rel["target_file"] = models.get(rel["target_class"])

    return {
        "models": models,
        "relationships": relationships,
        "model_files": sorted(model_files_set),
    }


def _scan_code_for_models(
    _ast, code: str, file_path: str,
    models: Dict[str, str],
    relationships: List[Dict],
    model_files_set: set,
    seen_rels: Optional[set] = None,
) -> None:
    """Parse a single file's code for ORM model classes and relationships."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return

    for node in _ast.iter_child_nodes(tree):
        if not isinstance(node, _ast.ClassDef):
            continue

        # --- Detect ORM model classes ---
        # Heuristic 1: inherits from *Model / BaseModel / db.Model
        base_names = []
        for b in node.bases:
            if isinstance(b, _ast.Name):
                base_names.append(b.id)
            elif isinstance(b, _ast.Attribute):
                base_names.append(b.attr)
        is_model = any(
            n == "BaseModel" or n == "Model"
            for n in base_names
        )

        # Heuristic 2: has __tablename__ attribute (strongest ORM signal)
        has_tablename = False
        for item in node.body:
            if isinstance(item, _ast.Assign):
                for target in item.targets:
                    if isinstance(target, _ast.Name) and target.id == "__tablename__":
                        has_tablename = True
                        break

        # Heuristic 3: inherits from a known ORM model already in the registry
        inherits_known_model = any(n in models for n in base_names)

        if not (is_model or has_tablename or inherits_known_model):
            continue

        class_name = node.name
        models[class_name] = file_path
        model_files_set.add(file_path)

        # Scan class body for db.relationship() calls
        for item in _ast.walk(node):
            if not isinstance(item, _ast.Call):
                continue
            func = item.func
            # Match: db.relationship('TargetClass', ...) or relationship('...')
            is_rel = False
            if isinstance(func, _ast.Attribute) and func.attr == "relationship":
                is_rel = True
            elif isinstance(func, _ast.Name) and func.id == "relationship":
                is_rel = True
            if not is_rel:
                continue
            # Extract first string argument = target class name
            if item.args and isinstance(item.args[0], _ast.Constant) and isinstance(item.args[0].value, str):
                target_class = item.args[0].value
                # Find the field name (the assignment target)
                field_name = _find_assignment_target(_ast, node, item)
                # Dedup: skip if already seen from another strategy
                rel_key = (class_name, field_name or "?", target_class)
                if seen_rels is not None:
                    if rel_key in seen_rels:
                        continue
                    seen_rels.add(rel_key)
                relationships.append({
                    "source_file": file_path,
                    "source_class": class_name,
                    "target_class": target_class,
                    "target_file": None,  # resolved later
                    "field": field_name or "?",
                })


def _find_assignment_target(_ast, class_node, call_node) -> Optional[str]:
    """Find the attribute name that a call is assigned to within a class body."""
    for item in class_node.body:
        if isinstance(item, _ast.Assign):
            if item.value is call_node:
                for t in item.targets:
                    if isinstance(t, _ast.Name):
                        return t.id
    return None


def collect_reverse_dependencies(
    interfaces_path: Path,
    file_path: str,
) -> List[Dict[str, str]]:
    """Collect reverse dependencies: who depends on the current file.

    Returns list of edges where current file is the *target* (callee/parent/type).
    This answers: "which other files will break if I change this file?"

    Returns:
        [{"dependent_file": ..., "dependent_unit": ..., "edge_type": ..., "via": ...}, ...]
    """
    result: List[Dict[str, str]] = []
    if not interfaces_path or not interfaces_path.exists():
        return result

    try:
        with open(interfaces_path, "r", encoding="utf-8") as f:
            interfaces = json.load(f)
    except Exception:
        return result

    edf = interfaces.get("enhanced_data_flow", {})

    # Files that inherit from something in this file
    for edge in edf.get("inheritance_edges", []):
        if edge.get("parent_file") == file_path:
            result.append({
                "dependent_file": edge.get("source_file", ""),
                "dependent_unit": edge.get("child", ""),
                "edge_type": "inherits_from",
                "via": edge.get("parent", ""),
            })

    # Files that call something in this file
    for edge in edf.get("invocation_edges", []):
        if edge.get("callee_file") == file_path:
            result.append({
                "dependent_file": edge.get("caller_file", ""),
                "dependent_unit": edge.get("caller", ""),
                "edge_type": "calls",
                "via": edge.get("callee", ""),
            })

    # Files that reference types from this file
    for edge in edf.get("reference_edges", []):
        if edge.get("type_file") == file_path:
            result.append({
                "dependent_file": edge.get("source_file", ""),
                "dependent_unit": edge.get("unit", ""),
                "edge_type": "references",
                "via": edge.get("referenced_type", ""),
            })

    return result


# ============================================================================
# Main Entry Point
# ============================================================================

def build_dependency_context(
    batch: "PlannedTask",
    interfaces_path: Path,
    base_classes_path: Path,
    data_flow_path: Path,
    tasks_path: Path,
    completed_task_ids: List[str],
    feature_spec_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect all dependency context for a task.

    This is the single entry point used by ``run_batch.py``'s
    batch-prep flow (``_prepare_batch_context``).

    Args:
        batch: The current PlannedTask
        interfaces_path: Path to interfaces.json
        base_classes_path: Path to base_classes.json
        data_flow_path: Path to data_flow.json
        tasks_path: Path to tasks.json
        completed_task_ids: List of completed task IDs
        feature_spec_path: Path to feature_spec.json (for project background context)

    Returns:
        A dict containing all context sections, ready for prompt injection.
    """
    # Load project background/technology context from feature_spec.json
    project_background = ""
    if feature_spec_path and feature_spec_path.exists():
        try:
            project_background = get_project_background_context(feature_spec_path)
        except Exception as _exc:
            logger.warning("Failed to load project background context: %s", _exc)

    return {
        "project_background": project_background,
        "base_classes": collect_base_classes_context(
            base_classes_path, batch.subtree
        ),
        "data_flow_edges": collect_data_flow_edges(
            data_flow_path, batch.subtree
        ) if batch.task_type != "wiring" else collect_all_data_flow_edges(
            data_flow_path
        ),
        "dependencies": collect_dependency_files(
            interfaces_path, batch.file_path
        ),
        "completed": collect_completed_context(
            completed_task_ids, tasks_path
        ),
        "current_subtree": batch.subtree,
        "current_file": batch.file_path,
        "model_registry": scan_orm_model_registry(interfaces_path),
        "reverse_deps": collect_reverse_dependencies(
            interfaces_path, batch.file_path
        ),
    }
