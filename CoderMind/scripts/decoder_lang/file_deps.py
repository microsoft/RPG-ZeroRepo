"""Language-specific file dependency helpers for task planning.

The planner needs file-level implementation order, but import/include/module
syntax is language-specific.  This module keeps that knowledge out of
``plan_tasks.py`` and gives each backend a small helper that emits resolved,
repo-relative file dependency edges.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable


@dataclass(frozen=True)
class FileDependencyEdge:
    """A file-level dependency edge.

    ``dependency`` must be implemented before ``dependent``.  ``confidence`` is
    informational for diagnostics; all returned edges are conservative enough
    for ordering, but callers may surface weak/unresolved notes.
    """

    dependency: str
    dependent: str
    reason: str
    confidence: str = "resolved"
    detail: dict[str, Any] = field(default_factory=dict)


_TS_JS_IMPORT_RE = re.compile(
    r"""
    (?:import\s+(?:type\s+)?(?:[^'\"]+?\s+from\s+)?|export\s+[^'\"]*?\s+from\s+|import\s*\()\s*
    ["']([^"']+)["']
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)
_C_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_RUST_PATH_MOD_RE = re.compile(
    r'#\s*\[\s*path\s*=\s*"([^"]+)"\s*\]\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;',
    re.MULTILINE | re.DOTALL,
)
_RUST_MOD_RE = re.compile(r'(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;')
_RUST_USE_RE = re.compile(r'(?m)^\s*(?:pub\s+)?use\s+crate::([A-Za-z_][A-Za-z0-9_]*)\b')
_GO_IMPORT_BLOCK_RE = re.compile(r'import\s*\((.*?)\)', re.DOTALL)
_GO_IMPORT_SINGLE_RE = re.compile(r'import\s+(?:[._A-Za-z0-9]+\s+)?"([^"]+)"')
_GO_PACKAGE_RE = re.compile(r'(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)\b')
_GO_MODULE_RE = re.compile(r'(?m)^\s*module\s+(\S+)')

_SOURCE_EXTENSIONS = {
    "python": (".py",),
    "go": (".go",),
    "rust": (".rs",),
    "typescript": (".ts", ".tsx"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"),
}
_TS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")
_C_HEADER_EXTENSIONS = (".h",)
_CPP_HEADER_EXTENSIONS = (".hpp", ".hh", ".hxx", ".h")
_CPP_SOURCE_EXTENSIONS = (".cpp", ".cc", ".cxx")
_C_SOURCE_EXTENSIONS = (".c",)


def normalize_path(path: str) -> str:
    """Return a repo-relative POSIX path without leading ``./`` segments."""
    norm = str(path).replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def infer_language_from_path(path: str) -> str | None:
    """Infer a backend language name from a source path extension."""
    norm = normalize_path(path)
    if norm.endswith(".d.ts"):
        return "typescript"
    for language in ("typescript", "javascript", "cpp", "c", "rust", "go", "python"):
        if norm.endswith(_SOURCE_EXTENSIONS[language]):
            return language
    return None


def source_code_for(file_path: str, file_sources: dict[str, str]) -> str:
    """Fetch source by normalized path while tolerating legacy key shapes."""
    norm = normalize_path(file_path)
    if norm in file_sources:
        return file_sources[norm]
    return file_sources.get(file_path, "")


def resolved_python_edges(files: list[str], file_sources: dict[str, str]) -> list[FileDependencyEdge]:
    """Resolve Python import edges using dotted module names and relative imports."""
    normalized_files = [normalize_path(path) for path in files]
    module_to_file: dict[str, str] = {}
    package_init_to_file: dict[str, str] = {}
    for file_path in normalized_files:
        if not file_path.endswith(".py"):
            continue
        module = _python_file_to_module(file_path)
        if module:
            module_to_file.setdefault(module, file_path)
            if module.endswith(".__init__"):
                package_init_to_file.setdefault(module[: -len(".__init__")], file_path)
    for package, init_file in package_init_to_file.items():
        module_to_file.setdefault(package, init_file)

    edges: list[FileDependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for file_path in normalized_files:
        imports = _python_imported_modules(source_code_for(file_path, file_sources), file_path)
        for module in imports:
            dependency = _match_python_module(module, module_to_file)
            if not dependency or dependency == file_path:
                continue
            _append_edge(
                edges,
                seen,
                dependency=dependency,
                dependent=file_path,
                reason="python_import",
                detail={"module": module},
            )
    return edges


def resolved_go_edges(files: list[str], file_sources: dict[str, str]) -> list[FileDependencyEdge]:
    """Resolve Go file ordering using import package directories and test adjacency."""
    normalized_files = [normalize_path(path) for path in files]
    file_set = set(normalized_files)
    source_by_dir: dict[str, list[str]] = {}
    module_name = _go_module_name(file_sources)
    for file_path in normalized_files:
        if not file_path.endswith(".go"):
            continue
        if file_path.endswith("_test.go"):
            continue
        source_by_dir.setdefault(_dirname(file_path), []).append(file_path)

    edges: list[FileDependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for file_path in normalized_files:
        if not file_path.endswith(".go"):
            continue
        code = source_code_for(file_path, file_sources)
        current_dir = _dirname(file_path)
        if file_path.endswith("_test.go"):
            for source_file in sorted(source_by_dir.get(current_dir, [])):
                _append_edge(
                    edges,
                    seen,
                    dependency=source_file,
                    dependent=file_path,
                    reason="go_test_depends_on_package_source",
                    detail={"package_dir": current_dir},
                )
        for import_path in _go_import_paths(code):
            package_dir = _resolve_go_import_dir(import_path, module_name, source_by_dir)
            if not package_dir:
                continue
            for source_file in sorted(source_by_dir.get(package_dir, [])):
                if source_file == file_path:
                    continue
                _append_edge(
                    edges,
                    seen,
                    dependency=source_file,
                    dependent=file_path,
                    reason="go_import_package_dir",
                    detail={"import_path": import_path, "package_dir": package_dir},
                )
    return edges


def resolved_rust_edges(files: list[str], file_sources: dict[str, str]) -> list[FileDependencyEdge]:
    """Resolve Rust module file edges from ``mod`` declarations.

    ``use crate::foo::Type`` is a weak fallback for skeletons that omitted the
    corresponding ``mod foo;`` in the parsed file set.  The fallback uses the
    module segment (``foo``), never the terminal type name (``Type``).
    """
    normalized_files = [normalize_path(path) for path in files]
    file_set = set(normalized_files)
    module_name_to_files: dict[str, list[str]] = {}
    for file_path in normalized_files:
        if not file_path.endswith(".rs"):
            continue
        module_name_to_files.setdefault(PurePosixPath(file_path).stem, []).append(file_path)

    edges: list[FileDependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for file_path in normalized_files:
        if not file_path.endswith(".rs"):
            continue
        code = source_code_for(file_path, file_sources)
        consumed_spans: list[tuple[int, int]] = []
        for match in _RUST_PATH_MOD_RE.finditer(code):
            rel_path, module_name = match.groups()
            consumed_spans.append(match.span())
            dependency = normalize_path(str(PurePosixPath(_dirname(file_path)) / rel_path))
            if dependency in file_set and dependency != file_path:
                _append_edge(
                    edges,
                    seen,
                    dependency=dependency,
                    dependent=file_path,
                    reason="rust_path_mod",
                    detail={"module": module_name, "path": rel_path},
                )
        for match in _RUST_MOD_RE.finditer(code):
            if any(start <= match.start() < end for start, end in consumed_spans):
                continue
            module_name = match.group(1)
            for dependency in _rust_default_module_candidates(file_path, module_name):
                if dependency in file_set and dependency != file_path:
                    _append_edge(
                        edges,
                        seen,
                        dependency=dependency,
                        dependent=file_path,
                        reason="rust_mod_declaration",
                        detail={"module": module_name},
                    )
                    break
        for match in _RUST_USE_RE.finditer(code):
            module_name = match.group(1)
            candidates = module_name_to_files.get(module_name, [])
            for dependency in sorted(candidates):
                if dependency == file_path:
                    continue
                _append_edge(
                    edges,
                    seen,
                    dependency=dependency,
                    dependent=file_path,
                    reason="rust_use_crate_module",
                    confidence="weak",
                    detail={"module": module_name},
                )
                break
    return edges


def resolved_ts_js_edges(
    files: list[str],
    file_sources: dict[str, str],
    *,
    language: str,
) -> list[FileDependencyEdge]:
    """Resolve TypeScript/JavaScript relative import/export edges."""
    normalized_files = [normalize_path(path) for path in files]
    file_set = set(normalized_files)
    edges: list[FileDependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for file_path in normalized_files:
        code = source_code_for(file_path, file_sources)
        for specifier in _ts_js_import_specifiers(code):
            if not specifier.startswith(("./", "../")):
                continue
            dependency = _resolve_ts_js_specifier(file_path, specifier, file_set)
            if not dependency or dependency == file_path:
                continue
            _append_edge(
                edges,
                seen,
                dependency=dependency,
                dependent=file_path,
                reason=f"{language}_relative_import",
                detail={"specifier": specifier},
            )
    return edges


def resolved_c_family_edges(
    files: list[str],
    file_sources: dict[str, str],
    *,
    language: str,
) -> list[FileDependencyEdge]:
    """Resolve C/C++ quoted include edges and header-before-source convention."""
    normalized_files = [normalize_path(path) for path in files]
    file_set = set(normalized_files)
    header_exts = _CPP_HEADER_EXTENSIONS if language == "cpp" else _C_HEADER_EXTENSIONS
    source_exts = _CPP_SOURCE_EXTENSIONS if language == "cpp" else _C_SOURCE_EXTENSIONS
    headers_by_stem: dict[tuple[str, str], str] = {}
    for file_path in normalized_files:
        posix = PurePosixPath(file_path)
        if file_path.endswith(header_exts):
            headers_by_stem.setdefault((_dirname(file_path), posix.stem), file_path)

    edges: list[FileDependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for file_path in normalized_files:
        code = source_code_for(file_path, file_sources)
        for include in _C_INCLUDE_RE.findall(code):
            dependency = _resolve_include(file_path, include, file_set)
            if dependency and dependency != file_path:
                _append_edge(
                    edges,
                    seen,
                    dependency=dependency,
                    dependent=file_path,
                    reason=f"{language}_quoted_include",
                    detail={"include": include},
                )
        if file_path.endswith(source_exts):
            header = headers_by_stem.get((_dirname(file_path), PurePosixPath(file_path).stem))
            if header and header != file_path:
                _append_edge(
                    edges,
                    seen,
                    dependency=header,
                    dependent=file_path,
                    reason=f"{language}_matching_header",
                    detail={"stem": PurePosixPath(file_path).stem},
                )
    return edges


def _append_edge(
    edges: list[FileDependencyEdge],
    seen: set[tuple[str, str, str]],
    *,
    dependency: str,
    dependent: str,
    reason: str,
    confidence: str = "resolved",
    detail: dict[str, Any] | None = None,
) -> None:
    key = (dependency, dependent, reason)
    if key in seen:
        return
    seen.add(key)
    edges.append(FileDependencyEdge(
        dependency=dependency,
        dependent=dependent,
        reason=reason,
        confidence=confidence,
        detail=detail or {},
    ))


def _dirname(file_path: str) -> str:
    parent = str(PurePosixPath(normalize_path(file_path)).parent)
    return "" if parent == "." else parent


def _python_file_to_module(file_path: str) -> str:
    norm = normalize_path(file_path)
    if norm.endswith(".py"):
        norm = norm[:-3]
    if norm.startswith("src/"):
        norm = norm[4:]
    return norm.replace("/", ".")


def _normalize_python_module(module_name: str | None) -> str:
    if not module_name:
        return ""
    normalized = module_name.strip()
    while normalized.startswith("."):
        normalized = normalized[1:]
    if normalized.startswith("src."):
        normalized = normalized[4:]
    return normalized


def _resolve_python_relative_import(module_name: str | None, level: int, current_file: str) -> str | None:
    current_module = _python_file_to_module(current_file)
    package_parts = current_module.split(".")[:-1]
    if level <= 0:
        return _normalize_python_module(module_name)
    if level > len(package_parts):
        return None
    anchor_parts = package_parts[: len(package_parts) - level + 1]
    if module_name:
        anchor_parts.extend(module_name.split("."))
    return _normalize_python_module(".".join(anchor_parts))


def _is_type_checking_test(test_node: ast.AST) -> bool:
    if isinstance(test_node, ast.Name):
        return test_node.id == "TYPE_CHECKING"
    if isinstance(test_node, ast.Attribute):
        return test_node.attr == "TYPE_CHECKING"
    return False


def _iter_python_import_nodes(tree: ast.AST, inside_type_checking: bool = False):
    for node in ast.iter_child_nodes(tree):
        child_inside_type_checking = inside_type_checking
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            child_inside_type_checking = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, inside_type_checking
        yield from _iter_python_import_nodes(node, child_inside_type_checking)


def _python_imported_modules(file_code: str, current_file: str) -> set[str]:
    if not file_code.strip():
        return set()
    try:
        tree = ast.parse(file_code)
    except SyntaxError:
        return set()
    imported_modules: set[str] = set()
    for node, inside_type_checking in _iter_python_import_nodes(tree):
        if inside_type_checking:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imported_modules.add(_normalize_python_module(alias.name))
        elif isinstance(node, ast.ImportFrom):
            resolved_module = _resolve_python_relative_import(node.module, node.level, current_file)
            if resolved_module:
                imported_modules.add(resolved_module)
            base_module = _resolve_python_relative_import(
                node.module if node.module else "",
                node.level,
                current_file,
            )
            if base_module:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported_modules.add(_normalize_python_module(f"{base_module}.{alias.name}"))
    return imported_modules


def _match_python_module(module: str, module_to_file: dict[str, str]) -> str | None:
    candidate = _normalize_python_module(module)
    while candidate:
        if candidate in module_to_file:
            return module_to_file[candidate]
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _go_import_paths(code: str) -> list[str]:
    imports: list[str] = []
    for block in _GO_IMPORT_BLOCK_RE.findall(code):
        imports.extend(re.findall(r'"([^"]+)"', block))
    imports.extend(_GO_IMPORT_SINGLE_RE.findall(_GO_IMPORT_BLOCK_RE.sub("", code)))
    return imports


def _go_module_name(file_sources: dict[str, str]) -> str | None:
    for path, source in file_sources.items():
        if normalize_path(path) == "go.mod":
            match = _GO_MODULE_RE.search(source or "")
            if match:
                return match.group(1)
    return None


def _resolve_go_import_dir(
    import_path: str,
    module_name: str | None,
    source_by_dir: dict[str, list[str]],
) -> str | None:
    candidates: list[str] = []
    if module_name and import_path == module_name:
        candidates.append("")
    if module_name and import_path.startswith(f"{module_name}/"):
        candidates.append(import_path[len(module_name) + 1:])
    candidates.append(import_path)
    candidates.append(import_path.rsplit("/", 1)[-1])
    for candidate in candidates:
        norm = normalize_path(candidate)
        if norm in source_by_dir:
            return norm
    return None


def _rust_default_module_candidates(file_path: str, module_name: str) -> list[str]:
    current = normalize_path(file_path)
    current_dir = _dirname(current)
    stem = PurePosixPath(current).stem
    if stem in {"main", "lib", "mod"}:
        base_dir = current_dir
    else:
        base_dir = normalize_path(str(PurePosixPath(current_dir) / stem)) if current_dir else stem
    return [
        normalize_path(str(PurePosixPath(base_dir) / f"{module_name}.rs")) if base_dir else f"{module_name}.rs",
        normalize_path(str(PurePosixPath(base_dir) / module_name / "mod.rs")) if base_dir else f"{module_name}/mod.rs",
    ]


def _ts_js_import_specifiers(code: str) -> Iterable[str]:
    for match in _TS_JS_IMPORT_RE.finditer(code):
        yield match.group(1)


def _resolve_ts_js_specifier(current_file: str, specifier: str, file_set: set[str]) -> str | None:
    base = normalize_path(str(PurePosixPath(_dirname(current_file)) / specifier))
    candidates = [base]
    suffixes = _TS_EXTENSIONS if not PurePosixPath(base).suffix else ("",)
    for suffix in suffixes:
        candidates.append(f"{base}{suffix}")
    for suffix in suffixes:
        candidates.append(normalize_path(str(PurePosixPath(base) / f"index{suffix}")))
    for candidate in candidates:
        if candidate in file_set:
            return candidate
    return None


def _resolve_include(current_file: str, include: str, file_set: set[str]) -> str | None:
    include_norm = normalize_path(include)
    candidates = [
        normalize_path(str(PurePosixPath(_dirname(current_file)) / include_norm)),
        include_norm,
    ]
    for candidate in candidates:
        if candidate in file_set:
            return candidate
    basename = PurePosixPath(include_norm).name
    matches = sorted(path for path in file_set if PurePosixPath(path).name == basename)
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "FileDependencyEdge",
    "infer_language_from_path",
    "normalize_path",
    "resolved_c_family_edges",
    "resolved_go_edges",
    "resolved_python_edges",
    "resolved_rust_edges",
    "resolved_ts_js_edges",
    "source_code_for",
]
