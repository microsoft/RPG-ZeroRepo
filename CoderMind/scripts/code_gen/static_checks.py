#!/usr/bin/env python3
"""Static Completeness Checks for CoderMind Code Generation.

Project-type-agnostic static checks run after a subtree completes.
These detect unimplemented stubs and placeholder returns without LLM cost.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, List

from common.paths import FEATURE_SPEC_FILE, REPO_RPG_FILE
from decoder_lang import LanguageBackend, get_backend, resolve_decoder_language

logger = logging.getLogger(__name__)


def _load_json_if_exists(path: Path) -> Any:
    """Load JSON from ``path`` or return None when unavailable."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_static_backend(files: List[str]) -> LanguageBackend:
    """Resolve the backend used for static codegen completeness checks."""
    feature_spec = _load_json_if_exists(FEATURE_SPEC_FILE)
    rpg_obj = _load_json_if_exists(REPO_RPG_FILE)
    language = resolve_decoder_language(
        feature_spec=feature_spec,
        rpg_obj=rpg_obj,
        valid_files=files,
    )
    return get_backend(language)


def static_completeness_check(files: List[str], repo_path: Path) -> List[str]:
    """Project-type-agnostic static completeness check.

    Run after ALL tasks in a subtree are completed. Checks for:
    1. Functions/methods whose only real body is ``pass`` (stub)
    2. Return statements returning TODO/PLACEHOLDER strings
    3. Functions that raise NotImplementedError
    4. Functions whose only real body is ``...`` (Ellipsis)

    Args:
        files: List of file paths (relative to *repo_path*) to check.
        repo_path: Absolute path to the project repository root.

    Returns:
        List of human-readable issue strings (empty = all clean).
    """
    issues: List[str] = []
    backend = _resolve_static_backend(files)

    for filepath in files:
        full_path = repo_path / filepath
        if not full_path.exists():
            issues.append(f"MISSING: {filepath} does not exist")
            continue

        if not backend.is_source_file(filepath):
            continue

        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            issues.append(f"PARSE_ERROR: {filepath} — {exc}")
            continue

        if backend.name != "python":
            ok, error = backend.syntax_check(content, filepath)
            if not ok:
                issues.append(f"PARSE_ERROR: {filepath} — {error}")
            if backend.has_placeholder(content, filepath):
                issues.append(f"PLACEHOLDER: {filepath} contains placeholder code")
            continue

        try:
            tree = ast.parse(content, filename=filepath)
        except SyntaxError as exc:
            issues.append(f"PARSE_ERROR: {filepath} — {exc}")
            continue

        # Build set of abstract class names in this file (classes that
        # inherit from ABC, ABCMeta, or have names ending in Base/Abstract)
        _abc_class_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_abc = False
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name in ("ABC", "ABCMeta", "Protocol"):
                        is_abc = True
                for kw in node.keywords:
                    if kw.arg == "metaclass" and isinstance(kw.value, ast.Name):
                        if kw.value.id == "ABCMeta":
                            is_abc = True
                if is_abc:
                    _abc_class_names.add(node.name)

        # Walk with parent context to detect abstract methods
        _parent_map: dict = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                _parent_map[child] = node

        def _is_abstract_method(func_node) -> bool:
            """Check if a function node is a legitimate abstract/protocol method."""
            # Check @abstractmethod decorator
            for dec in func_node.decorator_list:
                dec_name = ""
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Attribute):
                    dec_name = dec.attr
                if dec_name == "abstractmethod":
                    return True
            # Check if parent class is ABC or Protocol
            parent = _parent_map.get(func_node)
            if isinstance(parent, ast.ClassDef):
                if parent.name in _abc_class_names:
                    return True
                for base in parent.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "Protocol":
                        return True
            return False

        for node in ast.walk(tree):
            # Check 1: function/method body is only ``pass``
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                # Filter out docstrings (Expr(Constant(str)))
                real_body = [
                    n
                    for n in body
                    if not (
                        isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str)
                    )
                ]
                if len(real_body) == 1 and isinstance(real_body[0], ast.Pass):
                    if not _is_abstract_method(node):
                        issues.append(
                            f"STUB: {filepath}:{node.lineno} "
                            f"{node.name}() has only `pass` — not implemented"
                        )
                # Check 4: function body is only ``...`` (Ellipsis)
                # Skip if it's an abstract method or Protocol method
                elif (
                    len(real_body) == 1
                    and isinstance(real_body[0], ast.Expr)
                    and isinstance(real_body[0].value, ast.Constant)
                    and real_body[0].value.value is ...
                ):
                    if not _is_abstract_method(node):
                        issues.append(
                            f"STUB: {filepath}:{node.lineno} "
                            f"{node.name}() has only `...` — not implemented"
                        )
                # Check 3: function body is only ``raise NotImplementedError``
                # Skip if it's an abstract method or Protocol method
                elif len(real_body) == 1 and isinstance(real_body[0], ast.Raise):
                    exc_node = real_body[0].exc
                    if exc_node is not None and (
                        (isinstance(exc_node, ast.Name) and exc_node.id == "NotImplementedError")
                        or (
                            isinstance(exc_node, ast.Call)
                            and isinstance(exc_node.func, ast.Name)
                            and exc_node.func.id == "NotImplementedError"
                        )
                    ):
                        if not _is_abstract_method(node):
                            issues.append(
                                f"STUB: {filepath}:{node.lineno} "
                                f"{node.name}() raises NotImplementedError — not implemented"
                            )

            # Check 2: return TODO/PLACEHOLDER string
            if isinstance(node, ast.Return) and isinstance(
                node.value, ast.Constant
            ):
                val = node.value.value
                if isinstance(val, str) and any(
                    marker in val.upper()
                    for marker in ("TODO", "PLACEHOLDER", "NOT IMPLEMENTED")
                ):
                    issues.append(
                        f"PLACEHOLDER: {filepath}:{node.lineno} "
                        f"returns placeholder string"
                    )

    return issues
