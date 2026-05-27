#!/usr/bin/env python3
"""Import Normalizer — Detect and fix inconsistent import prefixes.

This module solves the problem where LLM-generated skeleton code uses bare
package names (``from vibeanim.foo import Bar``) while the project layout
requires a ``src.`` prefix (``from src.vibeanim.foo import Bar``).

Usage::

    from common.import_normalizer import (
        detect_project_import_prefix,
        normalize_code,
        normalize_files,
    )

    # Detect the correct prefix from file paths or repo layout
    prefix = detect_project_import_prefix(repo_path)
    # e.g. "src.vibeanim"

    # Normalize a code string before writing to disk
    fixed_code = normalize_code(code_string, prefix)

    # Normalize all .py files in the repo (safety net before testing)
    changed_files = normalize_files(repo_path, prefix)

Design:
    - ``detect_project_import_prefix`` inspects the repo directory structure
      to determine the correct import prefix (e.g. ``src.vibeanim``).
    - ``normalize_code`` rewrites import lines in a code string.
    - ``normalize_files`` scans all ``.py`` files under ``src/`` and ``tests/``
      and fixes imports on disk.

All three can also accept an ``interfaces.json``-style dict to derive the
prefix from file paths inside the JSON, for use before the repo exists.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import ast as _ast

logger = logging.getLogger(__name__)


# ============================================================================
# Prefix Detection
# ============================================================================

def detect_project_import_prefix(
    repo_path: Optional[Path] = None,
    interfaces_subtrees: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Detect the correct import prefix for this project.

    Checks two sources (in priority order):

    1. **Repo directory structure** — if ``repo_path/src/<package>/`` exists,
       the prefix is ``src.<package>``.
    2. **interfaces.json subtrees** — if file paths start with ``src/<pkg>/``,
       the prefix is ``src.<pkg>``.

    Args:
        repo_path: Path to the project repository root.
        interfaces_subtrees: The ``subtrees`` dict from interfaces.json.

    Returns:
        The import prefix string (e.g. ``"src.vibeanim"``) or *None*.
    """
    # Strategy 1: on-disk directory structure
    if repo_path is not None:
        prefix = _detect_from_directory(repo_path)
        if prefix:
            return prefix

    # Strategy 2: interfaces.json file paths
    if interfaces_subtrees is not None:
        prefix = _detect_from_interfaces(interfaces_subtrees)
        if prefix:
            return prefix

    return None


def _detect_from_directory(repo_path: Path) -> Optional[str]:
    """Detect prefix from ``repo_path/src/<package>/``."""
    src_dir = repo_path / "src"
    if not src_dir.is_dir():
        return None

    candidates: Dict[str, int] = {}
    for child in src_dir.iterdir():
        if child.is_dir() and not child.name.startswith((".", "_")):
            # Count .py files to confirm it's a real package
            py_count = sum(1 for _ in child.rglob("*.py"))
            if py_count > 0:
                candidates[child.name] = py_count

    if not candidates:
        return None

    best_pkg = max(candidates, key=candidates.get)
    return f"src.{best_pkg}"


def _detect_from_interfaces(subtrees: Dict[str, Any]) -> Optional[str]:
    """Detect prefix from file paths in interfaces.json subtrees."""
    counts: Dict[str, int] = {}
    for subtree_data in subtrees.values():
        for file_path in subtree_data.get("interfaces", {}):
            parts = file_path.replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[0] == "src":
                key = f"src.{parts[1]}"
                counts[key] = counts.get(key, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


# ============================================================================
# Code Normalization
# ============================================================================

def normalize_code(code: str, expected_prefix: str) -> str:
    """Rewrite import lines in *code* to use the correct prefix.

    If *expected_prefix* is ``"src.vibeanim"`` and the code contains
    ``from vibeanim.foo import Bar``, it becomes
    ``from src.vibeanim.foo import Bar``.

    Already-correct lines (``from src.vibeanim.…``) are left untouched.
    String literals and comments are not modified.

    Args:
        code: Python source code string.
        expected_prefix: The full correct prefix (e.g. ``"src.vibeanim"``).

    Returns:
        Normalized source code string.
    """
    if not code or not expected_prefix:
        return code

    parts = expected_prefix.split(".", 1)
    if len(parts) != 2 or parts[0] != "src":
        return code

    bare_package = parts[1]  # e.g. "vibeanim"

    pattern = re.compile(
        r"^(\s*(?:from|import)\s+)" + re.escape(bare_package) + r"\b",
        re.MULTILINE,
    )

    def _replace(m: re.Match) -> str:
        return m.group(1) + expected_prefix

    return pattern.sub(_replace, code)


# ============================================================================
# File Normalization (on-disk)
# ============================================================================

def normalize_files(
    repo_path: Path,
    prefix: Optional[str] = None,
) -> List[str]:
    """Scan ``.py`` files under ``src/`` and ``tests/`` and fix import prefixes.

    This is a safety-net step that should run before pytest to catch any
    imports that slipped through earlier normalization (e.g. from code
    generated by sub-agents during the TDD loop).

    Args:
        repo_path: Root of the project repository.
        prefix: Expected import prefix. Auto-detected if not given.

    Returns:
        List of file paths (relative to *repo_path*) that were modified.
    """
    if prefix is None:
        prefix = detect_project_import_prefix(repo_path=repo_path)

    if not prefix:
        return []

    parts = prefix.split(".", 1)
    if len(parts) != 2 or parts[0] != "src":
        return []

    bare_package = parts[1]

    # Only search if src/<package> exists
    if not (repo_path / "src" / bare_package).is_dir():
        return []

    pattern = re.compile(
        r"^(\s*(?:from|import)\s+)" + re.escape(bare_package) + r"\b",
    )

    modified_files: List[str] = []
    search_dirs = [repo_path / "src", repo_path / "tests"]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for py_file in search_dir.rglob("*.py"):
            try:
                original = py_file.read_text(encoding="utf-8")
            except OSError:
                continue

            lines = original.split("\n")
            changed = False
            new_lines: List[str] = []

            for line in lines:
                stripped = line.lstrip()
                # Skip string literals and comments
                if stripped.startswith(("'", '"', "#")):
                    new_lines.append(line)
                    continue

                m = pattern.match(stripped)
                if m:
                    new_line = line.replace(
                        m.group(0),
                        m.group(1) + prefix,
                        1,
                    )
                    new_lines.append(new_line)
                    changed = True
                else:
                    new_lines.append(line)

            if changed:
                py_file.write_text("\n".join(new_lines), encoding="utf-8")
                rel = str(py_file.relative_to(repo_path))
                modified_files.append(rel)
                logger.info("Fixed import prefixes in %s", rel)

    return modified_files


# ============================================================================
# Future Annotations & Missing Imports
# ============================================================================

_FUTURE_ANNOTATIONS = "from __future__ import annotations"


def ensure_future_annotations(code: str) -> str:
    """Ensure ``from __future__ import annotations`` is at the top of the file.

    If already present but in the wrong position (after other imports),
    it is moved to the correct position.  Python requires ``__future__``
    imports to appear before any other code or imports.

    Args:
        code: Python source code string.

    Returns:
        Code with the future import guaranteed at the very top
        (after shebang / encoding declarations only).
    """
    if not code or not code.strip():
        return code

    lines = code.split("\n")

    # Remove any existing __future__ annotations line (may be misplaced)
    had_future = False
    filtered_lines = []
    for line in lines:
        if line.strip() == _FUTURE_ANNOTATIONS:
            had_future = True
            continue
        filtered_lines.append(line)

    # If code doesn't use any type annotations and didn't have __future__,
    # add it anyway for safety (skeleton files almost always need it)

    # Find insertion point: after shebang (#!) and encoding (# -*- coding)
    insert_idx = 0
    for i, line in enumerate(filtered_lines):
        stripped = line.strip()
        if i == 0 and stripped.startswith("#!"):
            insert_idx = 1
            continue
        if stripped.startswith("# -*-") or stripped.startswith("# coding"):
            insert_idx = i + 1
            continue
        break

    filtered_lines.insert(insert_idx, _FUTURE_ANNOTATIONS)
    return "\n".join(filtered_lines)


# Common standard-library symbols that often appear in type annotations
# but are forgotten in skeleton imports.
_STDLIB_ANNOTATION_IMPORTS = {
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "Enum": "from enum import Enum",
    "ABC": "from abc import ABC, abstractmethod",
    "abstractmethod": "from abc import ABC, abstractmethod",
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Sequence": "from typing import Sequence",
    "Mapping": "from typing import Mapping",
    "Callable": "from typing import Callable",
    "Union": "from typing import Union",
    "Any": "from typing import Any",
    "Iterator": "from typing import Iterator",
    "Iterable": "from typing import Iterable",
    "TYPE_CHECKING": "from typing import TYPE_CHECKING",
    "Literal": "from typing import Literal",
}


def fix_missing_stdlib_imports(code: str) -> str:
    """Add missing standard-library imports for symbols used in the code.

    Scans for common symbols (``@dataclass``, ``Callable``, ``Optional``, etc.)
    that appear in the code but are not imported, and adds the necessary
    import statements.

    Args:
        code: Python source code string.

    Returns:
        Code with missing stdlib imports added.
    """
    if not code or not code.strip():
        return code

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code

    # Collect all names already imported
    imported_names: set = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    # Collect all names used in the code
    used_names: set = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            used_names.add(node.id)
        elif isinstance(node, _ast.Attribute):
            used_names.add(node.attr)

    # Also check for @dataclass decorator usage
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            for deco in node.decorator_list:
                if isinstance(deco, _ast.Name):
                    used_names.add(deco.id)
                elif isinstance(deco, _ast.Call) and isinstance(deco.func, _ast.Name):
                    used_names.add(deco.func.id)

    # Determine which imports to add
    needed_imports: dict = {}
    for symbol, import_line in _STDLIB_ANNOTATION_IMPORTS.items():
        if symbol in used_names and symbol not in imported_names:
            # Group by import line to avoid duplicates
            needed_imports[import_line] = True

    if not needed_imports:
        return code

    # Insert imports in the file header (before the first class/function def).
    # We only look at top-level import lines to avoid being confused by
    # 'from'/'import' lines that accidentally ended up inside docstrings.
    lines = code.split("\n")

    # Find the first class/function definition to bound the header region
    first_def_line = len(lines)
    for node in tree.body:
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
            first_def_line = node.lineno - 1  # 0-indexed
            break

    last_import_line = -1
    future_line = -1
    for i in range(first_def_line):
        stripped = lines[i].strip()
        if stripped.startswith("from __future__"):
            future_line = i
        elif stripped.startswith(("import ", "from ")):
            last_import_line = i

    # Insert after the last header import, or after __future__
    if last_import_line >= 0:
        insert_at = last_import_line + 1
    elif future_line >= 0:
        insert_at = future_line + 1
    else:
        insert_at = 0
        for i in range(first_def_line):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith("#"):
                insert_at = i
                break

    for imp_line in sorted(needed_imports.keys()):
        lines.insert(insert_at, imp_line)
        insert_at += 1

    return "\n".join(lines)


def fix_skeleton_files(repo_path: Path) -> List[str]:
    """Fix common skeleton file issues across the entire repo.

    Applies all automated fixes to ``.py`` files under ``src/``:

    1. Add ``from __future__ import annotations`` (forward ref fix)
    2. Fix missing stdlib imports (``dataclass``, ``Callable``, etc.)
    3. Fix import prefixes (``from pkg.*`` → ``from src.pkg.*``)
    4. Fix missing base-class imports (``MathEntity``, ``Animation``, etc.)

    This should run once after ``write_interface_skeletons`` and also
    as a safety net before each test run.

    Args:
        repo_path: Root of the project repository.

    Returns:
        List of file paths (relative to *repo_path*) that were modified.
    """
    modified: List[str] = []
    src_dir = repo_path / "src"
    if not src_dir.is_dir():
        return modified

    prefix = detect_project_import_prefix(repo_path=repo_path)

    for py_file in src_dir.rglob("*.py"):
        try:
            original = py_file.read_text(encoding="utf-8")
        except OSError:
            continue

        code = original

        # 1. Ensure from __future__ import annotations
        code = ensure_future_annotations(code)

        # 2. Fix missing stdlib imports
        code = fix_missing_stdlib_imports(code)

        # 3. Fix import prefixes (inline, not calling normalize_files to avoid double I/O)
        if prefix:
            code = normalize_code(code, prefix)

        if code != original:
            py_file.write_text(code, encoding="utf-8")
            rel = str(py_file.relative_to(repo_path))
            modified.append(rel)
            logger.info("Fixed skeleton issues in %s", rel)

    # 4. Fix missing base-class / project-internal imports
    base_fixed = _fix_missing_base_class_imports(repo_path, prefix)
    modified.extend(base_fixed)

    return modified


def _fix_missing_base_class_imports(repo_path: Path, prefix: Optional[str] = None) -> List[str]:
    """Find classes/names used but not imported and add the import.

    Covers:
    - ``class Foo(Bar):`` where ``Bar`` is not imported
    - Default parameter values like ``easing: X = EasingFunction.LINEAR``
      where ``EasingFunction`` is not imported

    Only resolves names that are defined as classes in other project files.
    """
    src_dir = repo_path / "src"
    if not src_dir.is_dir():
        return []

    # Build a map: class_name -> file_path (relative to repo_path)
    class_to_file: Dict[str, str] = {}
    for py_file in src_dir.rglob("*.py"):
        try:
            tree = _ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        rel = str(py_file.relative_to(repo_path))
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef):
                if node.name not in class_to_file:
                    class_to_file[node.name] = rel

    modified: List[str] = []
    builtins = {"object", "Exception", "ValueError", "TypeError", "RuntimeError",
                "KeyError", "IndexError", "AttributeError", "NotImplementedError",
                "str", "int", "float", "dict", "list", "tuple", "set", "bool", "bytes",
                "type", "property", "staticmethod", "classmethod", "super", "None",
                "True", "False", "print", "len", "range", "enumerate", "zip", "map",
                "filter", "sorted", "reversed", "isinstance", "issubclass", "hasattr",
                "getattr", "setattr", "delattr", "id", "hash", "repr", "abs", "round",
                "min", "max", "sum", "all", "any", "iter", "next", "open"}

    for py_file in src_dir.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8")
            tree = _ast.parse(code)
        except (OSError, SyntaxError):
            continue

        # Collect imported names
        imported = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for a in node.names:
                    imported.add(a.asname or a.name.split(".")[-1])
            elif isinstance(node, _ast.ImportFrom):
                for a in node.names:
                    imported.add(a.asname or a.name)

        # Collect all top-level Name references that resolve to project classes
        # This covers: base classes, default values, type refs in non-annotation positions
        needed_names: set = set()
        for node in _ast.walk(tree):
            # Base classes
            if isinstance(node, _ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, _ast.Name):
                        needed_names.add(base.id)
            # Default argument values (e.g. EasingFunction.LINEAR)
            if isinstance(node, _ast.Attribute):
                if isinstance(node.value, _ast.Name):
                    needed_names.add(node.value.id)

        # Filter to only unimported names that exist as project classes
        missing_imports: List[str] = []
        for name in sorted(needed_names):
            if name in imported or name in builtins:
                continue
            if name in _STDLIB_ANNOTATION_IMPORTS:
                continue
            if name not in class_to_file:
                continue
            src_file = class_to_file[name]
            rel_current = str(py_file.relative_to(repo_path))
            if src_file == rel_current:
                continue

            module = src_file.replace("/", ".").replace("\\", ".")
            if module.endswith(".py"):
                module = module[:-3]
            imp_line = f"from {module} import {name}"
            if prefix:
                imp_line = normalize_code(imp_line, prefix).strip()
            missing_imports.append(imp_line)
            imported.add(name)

        if not missing_imports:
            continue

        lines = code.split("\n")

        # Find header region (before first class/function def)
        first_def_line = len(lines)
        for node in tree.body:
            if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                first_def_line = node.lineno - 1
                break

        last_import = -1
        for i in range(first_def_line):
            stripped = lines[i].strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith("from __future__"):
                last_import = i

        insert_at = last_import + 1 if last_import >= 0 else 1
        for imp in sorted(set(missing_imports)):
            lines.insert(insert_at, imp)
            insert_at += 1

        py_file.write_text("\n".join(lines), encoding="utf-8")
        rel = str(py_file.relative_to(repo_path))
        modified.append(rel)
        logger.info("Added missing project imports in %s", rel)

    return modified


# ============================================================================
# Import Convention Snippet (for LLM prompts)
# ============================================================================

def build_import_convention_snippet(
    repo_path: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> str:
    """Build a prompt snippet describing the project's import convention.

    This can be injected into LLM prompts (interface design, code gen,
    test gen) so the LLM knows which import style to use.

    Args:
        repo_path: Project repo root (used for auto-detection).
        prefix: Explicit prefix (skips detection).

    Returns:
        Markdown-formatted instruction string, or empty string if
        the convention cannot be determined.
    """
    if prefix is None and repo_path is not None:
        prefix = detect_project_import_prefix(repo_path=repo_path)

    if not prefix:
        return ""

    parts = prefix.split(".", 1)
    if len(parts) != 2 or parts[0] != "src":
        return ""

    bare_package = parts[1]

    return f"""\
## Import Convention (CRITICAL)
- This project's source code lives under `src/{bare_package}/`.
- ALL internal imports MUST use the full path with `src.` prefix:
  - [OK] `from {prefix}.module import ClassName`
  - [FAIL] `from {bare_package}.module import ClassName`
- The `src.` prefix is required because the Python path is set to the repo root,
  not to `src/`.
"""
