#!/usr/bin/env python3
"""DependencyGraph — AST-based code dependency graph.

Ported from RPG-ZeroRepo/zerorepo/rpg_gen/base/rpg/dep_graph.py.

This module provides the DependencyGraph class that:
1. Scans a repository's physical structure (directories, files)
2. Parses Python code to extract classes, functions, and methods
3. Extracts dependency relationships (import, invokes, inherits)
4. Provides query interfaces and subgraph views for upstream modules

Graph views:
    G_tree     — CONTAINS edges only (directory/file hierarchy)
    G_imports  — IMPORTS edges only
    G_invokes  — INVOKES edges only
    G_inherits — INHERITS edges only
    G_code     — nodes that have AST info (classes, functions, methods, parsed files)
"""

import ast
import hashlib
import logging
import os
import posixpath
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

import lang_parser
from .models import EdgeType, NodeType
from common.utils import (
    normalize_path,
    is_test_file,
    get_node_range_robust,
    extract_source_by_lines,
    is_skip_dir,
    path_has_skip_dir,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DependencyGraph
# Filter functions
# ============================================================================

def _hash_content(content: str) -> str:
    """Return a stable hex digest of file contents for incremental change detection.

    Uses SHA-256.  The digest is stored as ``content_hash`` on every file node
    and read by :meth:`DependencyGraph.update_files` to skip files whose
    on-disk content is byte-identical to the last parse (e.g. whitespace
    canonicalisation, commit-message-only changes that touch mtime but not
    code, or files reported "modified" by git that were actually unchanged).
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _exclude_irrelevant_for_build(file_id: str) -> bool:
    """Default filter for ``DependencyGraph.build()``.

    Returns ``True`` if the path should be **included** in the graph.
    """
    EXT_BLACKLIST = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".mp3", ".mp4", ".zip", ".tar", ".gz",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".exe", ".dll", ".so", ".o", ".a",
        ".rlib", ".rmeta",
        ".log",
    }

    FILE_BLACKLIST = {
        "Makefile", "CMakeLists.txt",
        "Dockerfile", "LICENSE", "LICENSE.txt",
        "COPYING", "requirements.txt", "environment.yml",
        "pyproject.toml",
    }

    path_obj = PurePosixPath(file_id)

    if path_obj.suffix.lower() in EXT_BLACKLIST:
        return False

    if is_skip_dir(path_obj.name) or path_has_skip_dir(file_id):
        return False

    if path_obj.name in FILE_BLACKLIST:
        return False

    if path_obj.name.startswith("."):
        return False

    if lang_parser.is_supported_source(file_id):
        if lang_parser.is_test_file(file_id):
            return False
    elif is_test_file(file_id):
        return False

    return True


def _exclude_irrelevant_for_parse(file_id: str) -> bool:
    """Default filter for ``DependencyGraph.parse()``.

    Returns ``True`` if the file should be parsed for source analysis.
    Accepts any source language registered with ``lang_parser`` (Python,
    Go, TypeScript / JavaScript, C / C++, Rust, ...).
    """
    if not lang_parser.is_supported_source(file_id):
        return False

    if lang_parser.is_test_file(file_id):
        return False

    path_lower = file_id.lower()
    EXCLUDE_FILES = {
        "setup.py", "__main__.py",
        "conftest.py", "requirements.py",
    }

    if any(path_lower.endswith(f"/{f}") for f in EXCLUDE_FILES):
        return False

    return True


# ============================================================================
# Module path conversion
# ============================================================================

def path_to_module(node_id: str) -> str:
    """Convert a file-system node ID to a Python dotted module path."""
    s = str(node_id).strip()
    if ":" in s:
        s = s.split(":", 1)[0]

    s = s.removeprefix("./")

    if s == ".":
        return ""

    path = PurePosixPath(s)
    if path.suffix == ".py":
        if path.stem == "__init__":
            parent = path.parent.as_posix()
            return parent.replace("/", ".") if parent != "" else ""
        else:
            mod = path.with_suffix("").as_posix()
            return mod.replace("/", ".")
    else:
        return path.as_posix().replace("/", ".")


# ============================================================================
# DependencyGraph
# ============================================================================

class DependencyGraph:
    """AST-based code dependency graph for a Python repository.

    After construction, call :meth:`build` to scan the file system and
    :meth:`parse` to perform AST analysis.

    Attributes:
        repo_dir: Absolute path to the repository root.
        G: The underlying ``networkx.MultiDiGraph``.
        G_tree: Subgraph view showing only CONTAINS edges.
        G_imports: Subgraph view showing only IMPORTS edges.
        G_invokes: Subgraph view showing only INVOKES edges.
        G_inherits: Subgraph view showing only INHERITS edges.
        G_code: Subgraph view showing only nodes with AST information.
    """

    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        # Lazy cache of the Go module path declared in go.mod; resolved on
        # first cross-package import lookup so we don't pay the I/O on
        # Python-only repositories.
        self._go_module_path_cache: str | None = None
        self._go_module_path_loaded = False

        # Filtered subgraph views
        self.G_tree = nx.subgraph_view(
            self.G,
            filter_edge=lambda u, v, k: self.G.edges[u, v, k].get("type") == EdgeType.CONTAINS,
        )
        self.G_imports = nx.subgraph_view(
            self.G,
            filter_edge=lambda u, v, k: self.G.edges[u, v, k].get("type") == EdgeType.IMPORTS,
        )
        self.G_invokes = nx.subgraph_view(
            self.G,
            filter_edge=lambda u, v, k: self.G.edges[u, v, k].get("type") == EdgeType.INVOKES,
        )
        self.G_inherits = nx.subgraph_view(
            self.G,
            filter_edge=lambda u, v, k: self.G.edges[u, v, k].get("type") == EdgeType.INHERITS,
        )
        self.G_code = nx.subgraph_view(
            self.G,
            filter_node=lambda n: self.G.nodes[n].get("ast") is not None,
        )

    # ------------------------------------------------------------------
    # Internal node/edge helpers
    # ------------------------------------------------------------------

    # Fields computed by to_dict() that should not persist in G.nodes
    _DERIVED_FIELDS = frozenset({
        "imports_from", "calls", "called_by",
        "inherits", "inherited_by",
    })

    # ------------------------------------------------------------------
    # lang_parser integration constants
    # ------------------------------------------------------------------
    # Unit types from lang_parser that should map to a CLASS-like RPG node.
    _LP_CLASS_LIKE_UNIT_TYPES = frozenset({"class", "struct", "interface", "enum", "trait"})
    # lang_parser unit_type -> RPG NodeType.
    _LP_NODE_TYPES = {
        "class": NodeType.CLASS,
        "struct": NodeType.CLASS,
        "enum": NodeType.CLASS,
        "interface": NodeType.INTERFACE,
        "trait": NodeType.INTERFACE,
        "function": NodeType.FUNCTION,
        "method": NodeType.METHOD,
        "import": NodeType.IMPORT,
        "package": NodeType.PACKAGE,
    }
    # Extensions to try when resolving an ECMAScript import that omitted the suffix.
    _TS_JS_IMPORT_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx")
    # File extensions belonging to the C-family for include-graph heuristics.
    _C_FAMILY_EXTENSIONS = frozenset({".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"})
    # C-family header extensions. A function call resolves to its definition in
    # an implementation file, not a prototype/declaration in a header.
    _C_HEADER_EXTENSIONS = frozenset({".h", ".hpp", ".hh", ".hxx"})
    # File extensions belonging to Rust crates.
    _RUST_EXTENSIONS = frozenset({".rs"})

    @staticmethod
    def _extract_signature(node: ast.AST, is_method: bool = False) -> str:
        """Extract a human-readable signature string from a FunctionDef AST node.

        Returns a string like ``"(page: int = 1) -> str"``.
        For methods, ``self``/``cls`` are omitted.
        """
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""

        args_node = node.args
        parts: List[str] = []

        # Collect all positional args (posonlyargs + args)
        all_pos = list(getattr(args_node, "posonlyargs", [])) + list(args_node.args)

        # Defaults are right-aligned to args
        num_defaults = len(args_node.defaults)
        default_offset = len(all_pos) - num_defaults

        skip_first = is_method and all_pos and all_pos[0].arg in ("self", "cls")

        for i, arg in enumerate(all_pos):
            if skip_first and i == 0:
                continue
            s = arg.arg
            if arg.annotation:
                try:
                    s += ": " + ast.unparse(arg.annotation)
                except Exception:
                    pass
            # Check for default value
            default_idx = i - default_offset
            if default_idx >= 0 and default_idx < num_defaults:
                try:
                    s += " = " + ast.unparse(args_node.defaults[default_idx])
                except Exception:
                    pass
            parts.append(s)

        # *args
        if args_node.vararg:
            s = "*" + args_node.vararg.arg
            if args_node.vararg.annotation:
                try:
                    s += ": " + ast.unparse(args_node.vararg.annotation)
                except Exception:
                    pass
            parts.append(s)
        elif args_node.kwonlyargs:
            # bare * separator when there are keyword-only args but no *args
            parts.append("*")

        # keyword-only args
        for i, arg in enumerate(args_node.kwonlyargs):
            s = arg.arg
            if arg.annotation:
                try:
                    s += ": " + ast.unparse(arg.annotation)
                except Exception:
                    pass
            if i < len(args_node.kw_defaults) and args_node.kw_defaults[i] is not None:
                try:
                    s += " = " + ast.unparse(args_node.kw_defaults[i])
                except Exception:
                    pass
            parts.append(s)

        # **kwargs
        if args_node.kwarg:
            s = "**" + args_node.kwarg.arg
            if args_node.kwarg.annotation:
                try:
                    s += ": " + ast.unparse(args_node.kwarg.annotation)
                except Exception:
                    pass
            parts.append(s)

        sig = "(" + ", ".join(parts) + ")"

        # Return type
        if node.returns:
            try:
                sig += " -> " + ast.unparse(node.returns)
            except Exception:
                pass

        return sig

    def _add_node(
        self,
        node_id: str,
        type: str,
        name: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a node to the graph, auto-creating parent nodes if needed."""
        nid = normalize_path(node_id)
        if type not in NodeType:
            logger.warning("Unknown node type: %s at %s", type, nid)
        if not name:
            name = self.get_name(nid, ntype=type, with_badge=False)

        if nid in self.G:
            old_type = self.G.nodes[nid].get("type")
            if old_type != type:
                logger.warning("Node type conflict at node_id: %s. Existing: %s, New: %s", nid, old_type, type)

        self.G.add_node(nid, type=type, module=path_to_module(nid), name=name, **kwargs)

        if parent_id is None:
            parent_exists, parent_id = self.get_parent(nid)
        else:
            parent_exists = parent_id in self.G
        if not parent_exists and parent_id is not None:
            if type in [NodeType.DIRECTORY, NodeType.FILE]:
                parent_type = NodeType.DIRECTORY
            elif type in [NodeType.CLASS, NodeType.FUNCTION]:
                parent_type = NodeType.FILE
            else:
                parent_type = NodeType.CLASS
            self._add_node(parent_id, parent_type)
        if parent_id is not None:
            self._add_edge(parent_id, nid, type=EdgeType.CONTAINS)

    def _ensure_node(self, node_id: str, type: str) -> None:
        """Ensure the given node exists in the graph; add it if not."""
        nid = normalize_path(node_id)
        if nid not in self.G:
            self._add_node(nid, type=type)

    def _add_edge(self, src: str, dst: str, type: str, **kwargs: Any) -> bool:
        """Add a typed edge; skip if an edge with the same type already exists."""
        u = normalize_path(src)
        v = normalize_path(dst)

        if u not in self.G or v not in self.G:
            logger.error("Missing node(s): '%s' or '%s' not in graph", u, v)
            return False

        edge_data = self.G.get_edge_data(u, v, default={})
        for _key, data in edge_data.items():
            if data.get("type") == type:
                return False

        self.G.add_edge(u, v, type=type, **kwargs)
        return True

    # ------------------------------------------------------------------
    # Build — scan file system
    # ------------------------------------------------------------------

    def build(self, filter_func: Callable[[str], bool] = _exclude_irrelevant_for_build) -> None:
        """Construct the file/directory structure graph by walking the repo."""
        logger.info("Building DependencyGraph for repo: %s", self.repo_dir)

        repo_root = Path(self.repo_dir)
        if not repo_root.exists():
            raise FileNotFoundError(f"Repo root not found: {self.repo_dir}")
        self._add_node(".", type=NodeType.DIRECTORY, code_path=self.repo_dir)

        for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True, followlinks=False):
            dir_path = Path(dirpath)
            dir_rel = normalize_path(dir_path.relative_to(repo_root))
            if not filter_func(dir_rel):
                dirnames[:] = []
                continue

            self._ensure_node(dir_rel, type=NodeType.DIRECTORY)

            for dname in dirnames:
                subdir_path = dir_path / dname
                subdir_rel = normalize_path(subdir_path.relative_to(repo_root))
                if not filter_func(subdir_rel):
                    continue
                self._add_node(subdir_rel, type=NodeType.DIRECTORY, parent_id=dir_rel, code_path=str(subdir_path))

            for fname in filenames:
                file_path = dir_path / fname
                file_rel = normalize_path(str(file_path.relative_to(repo_root)))
                if not filter_func(file_rel):
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    logger.debug("[skip] Cannot read %s: %s", file_path, e)
                    continue
                self._add_node(
                    file_rel,
                    type=NodeType.FILE,
                    code=content,
                    code_path=str(file_path),
                    parent_id=dir_rel,
                    # Hash for incremental update_files(): lets us skip
                    # files whose content is byte-identical to the last
                    # parse (e.g. only commit message / metadata changed).
                    content_hash=_hash_content(content),
                )

        logger.info(
            "Finished building DependencyGraph, now has %d nodes and %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Parse — AST analysis
    # ------------------------------------------------------------------

    def parse(self, filter_func: Callable[[str], bool] = _exclude_irrelevant_for_parse) -> None:
        """Parse supported source files to extract structure and dependency edges.

        Python files keep the original ``ast``-based path (unchanged). Other
        supported languages (Go, TypeScript / JavaScript, C / C++, Rust, ...)
        are routed through ``lang_parser`` and adapted into the same RPG
        node / edge schema via the ``_parse_lp_*`` helpers.
        """
        # 1) Parse files for code units. Python keeps the exact AST helper path.
        logger.info("Parsing DependencyGraph to extract code structure")
        lp_results: list[tuple[str, lang_parser.LPFileResult]] = []
        for file_id, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE or not filter_func(file_id):
                continue

            content = self._read_code(file_id)
            language = lang_parser.detect_language(file_id)
            if language == "python":
                try:
                    tree = ast.parse(content)
                except SyntaxError as e:
                    logger.debug("[parse:skip] %s: %s", file_id, e)
                    continue

                self.G.nodes[file_id]["ast"] = tree
                self.G.nodes[file_id]["language"] = "python"
                self._parse_file(file_id, tree, content)
                continue

            if language is None:
                continue

            try:
                result = lang_parser.parse_file(file_id, content)
            except lang_parser.NotSupported as e:
                logger.debug("[parse:skip] %s: %s", file_id, e)
                continue
            self._parse_lp_file_result(file_id, result)
            lp_results.append((file_id, result))

        # Second pass: invoke edges for lang_parser results need the full unit
        # registry from the first pass so cross-file calls can be resolved.
        for file_id, result in lp_results:
            self._parse_lp_invoke_dependencies(file_id, result)
        logger.info(
            "Finished parsing code structure, now has %d nodes and %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

        # 2) Parse import relationships
        logger.info("Parsing DependencyGraph to extract imports")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._init_alias_map(node_id)

        alias_links: nx.DiGraph = nx.DiGraph()
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._parse_imports(node_id, attrs["ast"], alias_links)
        logger.info("Finished parsing imports, added %d edges", self.G_imports.number_of_edges())

        # 3) Parse inheritance relationships
        logger.info("Parsing DependencyGraph to extract inherits")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            if attrs.get("type") == NodeType.CLASS:
                self._parse_inherits(node_id, attrs["ast"])
        logger.info("Finished parsing inherits, added %d edges", self.G_inherits.number_of_edges())

        # 4) Parse invocation relationships
        logger.info("Parsing DependencyGraph to extract invokes")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._parse_invokes(node_id, attrs["ast"])
        logger.info("Finished parsing invokes, added %d edges", self.G_invokes.number_of_edges())

        logger.info(
            "Finished parsing DependencyGraph, now has %d nodes and %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Incremental update API
    # ------------------------------------------------------------------
    #
    # These methods allow callers (the pre-commit hook, codegen, and
    # :class:`RPGService`) to refresh only the files that actually changed
    # rather than rebuilding the entire graph from disk.  The expensive
    # operations skipped are:
    #
    #   * Walking the filesystem (``os.walk`` of the whole repo).
    #   * AST parsing of every Python file (re-parses only changed files;
    #     unchanged files keep their cached ``ast`` attr).
    #
    # The 3 semantic passes (imports / inherits / invokes) are still re-run
    # globally, but they re-use the cached ASTs and so are fast.
    #
    # Semantic correctness contract: after :meth:`update_files` returns,
    # the graph must be byte-equivalent to what a full ``build()`` +
    # ``parse()`` cycle would have produced.  The test
    # ``test_update_files_matches_full_rebuild`` is the canary for this.

    def _file_descendants(self, file_id: str) -> set:
        """Return the file node + all of its descendant code units.

        Walks CONTAINS edges outward from ``file_id``.  Used by
        :meth:`remove_file` to drop a whole file's worth of nodes in
        one ``remove_nodes_from`` call (which also cleans up every edge
        attached to those nodes).
        """
        nid = normalize_path(file_id)
        if nid not in self.G:
            return set()
        result: set = {nid}
        stack: list = [nid]
        while stack:
            cur = stack.pop()
            for _src, child, attrs in self.G.out_edges(cur, data=True):
                if attrs.get("type") == EdgeType.CONTAINS and child not in result:
                    result.add(child)
                    stack.append(child)
        return result

    def remove_file(self, file_rel: str) -> int:
        """Drop a file node, all its descendant code units, and all attached edges.

        Idempotent: returns ``0`` if the file isn't in the graph.  The
        file's parent directory node is preserved (it may still own
        other files).  Returns the total number of graph nodes removed.

        **Safety**: refuses to operate on non-FILE nodes (directories,
        code units) — returns ``0`` instead.  This prevents a careless
        caller from passing a directory path (e.g. ``"."``) and
        recursively wiping a large subtree.  Use ``remove_nodes_from``
        directly if you need that.
        """
        nid = normalize_path(file_rel)
        attrs = self.G.nodes.get(nid)
        if not attrs or attrs.get("type") != NodeType.FILE:
            return 0
        victims = self._file_descendants(nid)
        if not victims:
            return 0
        # remove_nodes_from also drops every in/out edge of each victim,
        # so callers don't need to worry about dangling INVOKES / INHERITS
        # / IMPORTS edges from other files pointing at the deleted units.
        self.G.remove_nodes_from(victims)
        return len(victims)

    def add_file(
        self,
        file_rel: str,
        *,
        filter_func: Callable[[str], bool] = _exclude_irrelevant_for_build,
    ) -> bool:
        """Read ``<repo_dir>/<file_rel>`` from disk and add it as a fresh file node.

        Recursively creates any missing parent directory nodes.  Runs
        :meth:`_parse_file` to create code-unit children (classes /
        functions / methods).  Stores both ``code`` and ``content_hash``
        on the file node so the next :meth:`update_files` call can
        cheaply detect when this file hasn't actually changed.

        Does **not** run the semantic passes (imports / inherits /
        invokes) — those are global and must be re-run after all
        per-file edits via :meth:`_rerun_semantic_passes`.

        Returns:
            ``True`` if the file was added (or was already present and
            got refreshed code units), ``False`` if the file is missing
            on disk, unreadable, or filtered out by ``filter_func``.
        """
        nid = normalize_path(file_rel)
        if not filter_func(nid):
            return False

        repo_root = Path(self.repo_dir)
        abs_path = repo_root / nid
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        if not abs_path.is_file():
            return False

        # Walk parent directories from root downward so every intermediate
        # ``DIRECTORY`` node exists before we link the file in.  Mirrors
        # what build() does naturally via os.walk.
        rel_parts = PurePosixPath(nid).parts
        if len(rel_parts) > 1:
            cur = "."
            self._ensure_node(cur, type=NodeType.DIRECTORY)
            for part in rel_parts[:-1]:
                child = part if cur == "." else f"{cur}/{part}"
                self._add_node(
                    child,
                    type=NodeType.DIRECTORY,
                    parent_id=cur,
                    code_path=str(repo_root / child),
                )
                cur = child
            parent_dir = cur
        else:
            self._ensure_node(".", type=NodeType.DIRECTORY)
            parent_dir = "."

        self._add_node(
            nid,
            type=NodeType.FILE,
            code=content,
            code_path=str(abs_path),
            parent_id=parent_dir,
            content_hash=_hash_content(content),
        )

        # Dispatch on language exactly like :meth:`parse` does: Python
        # files use ast + ``_parse_file``; other supported languages
        # route through lang_parser + ``_parse_lp_file_result``. Without
        # this, an incremental ``update_files`` call (which goes through
        # ``add_file``) silently dropped all units / imports / invokes
        # for non-Python files because ``ast.parse`` raised SyntaxError
        # and we returned early.
        try:
            language = lang_parser.detect_language(nid)
        except Exception:  # pragma: no cover - defensive
            language = None

        if language == "python" or language is None:
            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                logger.debug("[add_file:syntax] %s: %s", nid, exc)
                return True
            self.G.nodes[nid]["ast"] = tree
            self.G.nodes[nid]["language"] = "python"
            self._parse_file(nid, tree, content)
            return True

        try:
            result = lang_parser.parse_file(nid, content)
        except lang_parser.NotSupported as exc:
            logger.debug("[add_file:lp_unsupported] %s: %s", nid, exc)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[add_file:lp_error] %s: %s", nid, exc)
            return True
        self._parse_lp_file_result(nid, result, include_dependencies=False)
        # NOTE: dependency edges are deliberately NOT added here. Cross-file
        # resolution needs the global unit registry to be in its final state,
        # which is only guaranteed after ``_rerun_semantic_passes`` runs at
        # the end of ``update_files``. That helper will re-discover this
        # file's language attr and replay both the file-result + dependency
        # passes.
        return True

    def _wipe_semantic_edges(self) -> int:
        """Remove every IMPORTS / INHERITS / INVOKES edge.

        Preserves CONTAINS edges (the tree structure).  Returns the
        number of edges removed.  Called by :meth:`update_files` before
        re-running the 3 semantic passes, so we don't accumulate stale
        cross-file edges (e.g. an INVOKES edge pointing at a method
        that the new file revision no longer exports).
        """
        semantic = {EdgeType.IMPORTS, EdgeType.INHERITS, EdgeType.INVOKES}
        # MultiDiGraph: must enumerate by (u, v, key) tuples.
        to_remove: List[Tuple[str, str, Any]] = [
            (u, v, k)
            for u, v, k, attrs in self.G.edges(keys=True, data=True)
            if attrs.get("type") in semantic
        ]
        for u, v, k in to_remove:
            self.G.remove_edge(u, v, k)
        # Also clear any stale alias_to_entity maps so _parse_imports
        # rebuilds them cleanly from the fresh ASTs.
        for nid in list(self.G.nodes):
            if "alias_to_entity" in self.G.nodes[nid]:
                self.G.nodes[nid].pop("alias_to_entity", None)
        return len(to_remove)

    def _rerun_semantic_passes(self) -> None:
        """Re-run import / inherit / invoke parsing across the whole graph.

        Cheaper than it sounds because every code-bearing node carries a
        cached ``ast`` attr from the last successful parse — unchanged
        files don't get re-AST-parsed.

        Mirrors passes 2-4 of :meth:`parse` so its behaviour stays in
        lockstep with full rebuild.  Callers must ensure ``ast`` attrs
        are fresh on any file they've edited (the public entry points
        :meth:`add_file` and :meth:`update_files` do this).

        Non-Python files (Go / TS / C / Rust / ...) don't carry an
        ``ast`` attr (lang_parser produces an :class:`LPFileResult`,
        not a :class:`ast.Module`), and their import + invoke edges are
        produced by ``_parse_lp_file_result`` / ``_parse_lp_invoke_
        dependencies``. Re-run those passes too — using the cached
        ``code`` attr — otherwise an incremental ``update_files`` call
        wipes the semantic edges and never restores them for non-Python
        sources.
        """
        # Pass 2 prereq: alias maps must exist on every code node before
        # any _parse_imports call (it propagates aliases bidirectionally).
        for nid in list(self.G_code.nodes):
            self._init_alias_map(nid)

        # ----- Python ast-based passes (unchanged behaviour) -----
        # Pass 2: imports
        alias_links: nx.DiGraph = nx.DiGraph()
        for nid, attrs in list(self.G_code.nodes(data=True)):
            tree = attrs.get("ast")
            if tree is not None:
                self._parse_imports(nid, tree, alias_links)

        # Pass 3: inherits (classes only)
        for nid, attrs in list(self.G_code.nodes(data=True)):
            if attrs.get("type") != NodeType.CLASS:
                continue
            tree = attrs.get("ast")
            if tree is not None:
                self._parse_inherits(nid, tree)

        # Pass 4: invokes
        for nid, attrs in list(self.G_code.nodes(data=True)):
            tree = attrs.get("ast")
            if tree is not None:
                self._parse_invokes(nid, tree)

        # ----- lang_parser passes for non-Python file nodes -----
        # Walk file nodes that carry a non-Python language and re-parse
        # them via lang_parser. ``_parse_lp_file_result`` rebuilds
        # IMPORTS edges (and keeps CONTAINS structure idempotently),
        # and ``_parse_lp_invoke_dependencies`` rebuilds INVOKES edges
        # using the unit registry the first pass populates. Two-pass
        # ordering mirrors :meth:`parse`.
        lp_results: List[Tuple[str, "lang_parser.LPFileResult"]] = []
        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE:
                continue
            language = attrs.get("language")
            if not language or language == "python":
                continue
            content = attrs.get("code")
            if not content:
                continue
            try:
                result = lang_parser.parse_file(nid, content)
            except lang_parser.NotSupported:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[rerun_semantic_passes:lp_skip] %s: %s", nid, exc)
                continue
            self._parse_lp_file_result(nid, result)
            lp_results.append((nid, result))
        for nid, result in lp_results:
            self._parse_lp_invoke_dependencies(nid, result)

    def update_files(
        self,
        file_rels: List[str],
        *,
        renames: Optional[Dict[str, str]] = None,
        rebuild_semantic_edges: bool = True,
        filter_func: Callable[[str], bool] = _exclude_irrelevant_for_build,
    ) -> Dict[str, Any]:
        """Incrementally re-parse the given files in place.

        Pipeline for each path:

        1. **Rename**: if ``renames[old] == new``, drop the old file
           node and add ``new`` from disk.  Git's ``--name-status -M``
           output should populate ``renames`` so editor refactors don't
           destroy and recreate edges spuriously.
        2. **Delete**: if the file is missing on disk, drop its node.
        3. **Hash check**: if the file's recorded ``content_hash``
           matches the disk content, skip — no graph edits.
        4. **Modify**: otherwise drop the file node + units and re-add
           from disk via :meth:`add_file`.

        After the per-file work is done, all IMPORTS / INHERITS /
        INVOKES edges are wiped and the 3 semantic passes are re-run
        globally (cheap because cached ASTs are reused for unchanged
        files).  Set ``rebuild_semantic_edges=False`` to skip that step
        when batching many ``update_files`` calls.

        Args:
            file_rels: Repo-relative paths to refresh.  Files not
                currently in the graph are treated as additions.
            renames: Optional ``{old_path: new_path}`` map produced by
                ``git diff --name-status -M`` — preserves IDs across
                ``git mv`` so cross-file edges (callers, imports) don't
                briefly point at deleted nodes.  ``old_path`` is also
                automatically appended to ``file_rels`` for deletion.
            rebuild_semantic_edges: Wipe + re-run imports/inherits/
                invokes passes after the per-file pass.  Defaults to
                ``True``; only the very rare batch caller needs to
                defer.
            filter_func: Inclusion test for newly-added paths (default
                matches what :meth:`build` would have included).

        Returns:
            ``{
                "renamed":           int,   # entries handled via ``renames``
                "deleted":           int,   # files missing on disk
                "unchanged_hash":    int,   # files skipped because hash matched
                "modified":          int,   # files re-parsed
                "added":             int,   # files new to the graph
                "nodes_removed":     int,
                "nodes_added":       int,
                "edges_resemanticised": int, # 0 if rebuild_semantic_edges=False
            }``
        """
        stats = {
            "renamed": 0, "deleted": 0,
            "unchanged_hash": 0, "modified": 0, "added": 0,
            "nodes_removed": 0, "nodes_added": 0,
            "edges_resemanticised": 0,
        }
        repo_root = Path(self.repo_dir)
        renames = renames or {}

        # Normalise input.  Renames imply the old path needs to be dropped
        # even if the caller didn't explicitly list it.
        targets = [normalize_path(f) for f in file_rels]
        for old in renames:
            targets.append(normalize_path(old))
        # Preserve order but dedupe (cheap with small input).
        seen: set = set()
        ordered: List[str] = []
        for f in targets:
            if f not in seen:
                seen.add(f)
                ordered.append(f)

        nodes_before = self.G.number_of_nodes()
        rename_dsts = {normalize_path(v) for v in renames.values()}

        for f in ordered:
            old_rename_target = renames.get(f)
            if old_rename_target is not None:
                # ``f`` is the OLD path in a rename: drop, then add NEW.
                new_path = normalize_path(old_rename_target)
                stats["nodes_removed"] += self.remove_file(f)
                if self.add_file(new_path, filter_func=filter_func):
                    stats["renamed"] += 1
                continue
            if f in rename_dsts:
                # Already handled when we processed the corresponding old path.
                continue

            abs_path = repo_root / f
            if not abs_path.is_file():
                removed = self.remove_file(f)
                if removed:
                    stats["nodes_removed"] += removed
                    stats["deleted"] += 1
                continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                # Treat unreadable file as missing.
                removed = self.remove_file(f)
                if removed:
                    stats["nodes_removed"] += removed
                    stats["deleted"] += 1
                continue

            new_hash = _hash_content(content)
            old_hash = self.G.nodes.get(f, {}).get("content_hash")
            if old_hash and old_hash == new_hash:
                stats["unchanged_hash"] += 1
                continue

            existed_before = f in self.G
            stats["nodes_removed"] += self.remove_file(f)
            if self.add_file(f, filter_func=filter_func):
                if existed_before:
                    stats["modified"] += 1
                else:
                    stats["added"] += 1

        # Net node delta: counts additions even when nothing was removed
        # (and avoids double-counting churn within a single batch).
        delta = self.G.number_of_nodes() - nodes_before
        if delta > 0:
            stats["nodes_added"] = delta

        if rebuild_semantic_edges and (
            stats["renamed"] + stats["deleted"] + stats["modified"] + stats["added"] > 0
        ):
            stats["edges_resemanticised"] = self._wipe_semantic_edges()
            self._rerun_semantic_passes()

        return stats

    # ------------------------------------------------------------------
    # File-level AST parsing
    # ------------------------------------------------------------------

    # Control flow types whose bodies may contain nested function/class definitions
    _CONTROL_FLOW_TYPES = (
        ast.If, ast.Try, ast.With, ast.For, ast.While,
        ast.AsyncWith, ast.AsyncFor,
    )

    @staticmethod
    def _get_control_flow_bodies(node: ast.AST) -> list:
        """Extract all branch bodies from a control flow node."""
        bodies: list = []
        if isinstance(node, ast.If):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        elif isinstance(node, ast.Try):
            bodies.append(node.body)
            for handler in node.handlers:
                bodies.append(handler.body)
            if node.orelse:
                bodies.append(node.orelse)
            if node.finalbody:
                bodies.append(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            bodies.append(node.body)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        return bodies

    def _extract_from_control_flow(
        self,
        stmts: list,
        file_id: str,
        source_code: str,
        get_range: Callable,
        parent_id: str,
    ) -> None:
        """Recursively scan control flow blocks for FunctionDef / ClassDef."""
        file_code_path = self.G.nodes[file_id].get("code_path", "") if file_id in self.G else ""
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_nid = f"{file_id}:{func_name}"
                if func_nid not in self.G:
                    start, end = get_range(node)
                    self._add_node(
                        func_nid,
                        type=NodeType.FUNCTION,
                        code_path=file_code_path,
                        parent_id=parent_id,
                        ast=node,
                        start_line=start,
                        end_line=end,
                        code=extract_source_by_lines(source_code, start, end),
                        signature=self._extract_signature(node, is_method=False),
                    )
            elif isinstance(node, ast.ClassDef):
                cls_name = node.name
                cls_nid = f"{file_id}:{cls_name}"
                if cls_nid not in self.G:
                    start, end = get_range(node)
                    self._add_node(
                        cls_nid,
                        type=NodeType.CLASS,
                        code_path=file_code_path,
                        parent_id=parent_id,
                        ast=node,
                        start_line=start,
                        end_line=end,
                        code=extract_source_by_lines(source_code, start, end),
                    )
                    # Methods inside the class
                    for body_node in node.body:
                        if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            meth_nid = f"{file_id}:{cls_name}.{body_node.name}"
                            if meth_nid not in self.G:
                                s2, e2 = get_range(body_node)
                                self._add_node(
                                    meth_nid,
                                    type=NodeType.METHOD,
                                    code_path=file_code_path,
                                    parent_id=cls_nid,
                                    ast=body_node,
                                    start_line=s2,
                                    end_line=e2,
                                    code=extract_source_by_lines(source_code, s2, e2),
                                    signature=self._extract_signature(body_node, is_method=True),
                                )
            # Recurse into nested control flow
            if isinstance(node, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node):
                    self._extract_from_control_flow(block, file_id, source_code, get_range, parent_id)

    def _parse_file(self, file_id: str, tree: ast.AST, source_code: str) -> None:
        """Parse definitions at the top level and within control-flow blocks.

        Extracts:
        - Top-level classes:    ``path/to/file.py:ClassName``
        - Top-level functions:  ``path/to/file.py:func``
        - Class methods:        ``path/to/file.py:ClassName.method``
        - Conditionally defined functions/classes (inside if/try/with blocks)
        """

        def get_range(node: ast.AST) -> Tuple[int, int]:
            start_inc, _, body_end, end_exc = get_node_range_robust(node, source_code)
            return start_inc, body_end

        file_code_path = self.G.nodes[file_id].get("code_path", "") if file_id in self.G else ""

        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_nid = f"{file_id}:{func_name}"
                start, end = get_range(node)

                self._add_node(
                    func_nid,
                    type=NodeType.FUNCTION,
                    code_path=file_code_path,
                    parent_id=file_id,
                    ast=node,
                    start_line=start,
                    end_line=end,
                    code=extract_source_by_lines(source_code, start, end),
                    signature=self._extract_signature(node, is_method=False),
                )

            elif isinstance(node, ast.ClassDef):
                cls_name = node.name
                cls_nid = f"{file_id}:{cls_name}"
                start, end = get_range(node)

                self._add_node(
                    cls_nid,
                    type=NodeType.CLASS,
                    code_path=file_code_path,
                    parent_id=file_id,
                    ast=node,
                    start_line=start,
                    end_line=end,
                    code=extract_source_by_lines(source_code, start, end),
                )

                for body in node.body:
                    if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        meth_name = body.name
                        meth_nid = f"{file_id}:{cls_name}.{meth_name}"
                        start2, end2 = get_range(body)

                        self._add_node(
                            meth_nid,
                            type=NodeType.METHOD,
                            code_path=file_code_path,
                            parent_id=cls_nid,
                            ast=body,
                            start_line=start2,
                            end_line=end2,
                            code=extract_source_by_lines(source_code, start2, end2),
                            signature=self._extract_signature(body, is_method=True),
                        )

        # Extract function/class definitions from control flow blocks
        for node in getattr(tree, "body", []):
            if isinstance(node, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node):
                    self._extract_from_control_flow(block, file_id, source_code, get_range, file_id)

    # ------------------------------------------------------------------
    # Alias map and import parsing
    # ------------------------------------------------------------------

    def _init_alias_map(self, node_id: str) -> dict:
        """Initialize the alias_to_entity mapping for a node from its children."""
        self.G.nodes[node_id]["alias_to_entity"] = {}
        alias_map = self.G.nodes[node_id]["alias_to_entity"]
        for _, child_id, edata in self.G_tree.out_edges(node_id, data=True):
            if edata.get("type") == EdgeType.CONTAINS:
                child_name = self.G.nodes[child_id].get("name")
                alias_map[child_name] = child_id

        if node_id.endswith("__init__.py"):
            _, parent_id = self.get_parent(node_id)
            for _, child_id, edata in self.G_tree.out_edges(parent_id, data=True):
                if edata.get("type") == EdgeType.CONTAINS and child_id != node_id:
                    child_name = self.G.nodes[child_id].get("name")
                    child_type = self.G.nodes[child_id].get("type")
                    if child_type == NodeType.DIRECTORY:
                        child_init_id = normalize_path(f"{child_id}/__init__.py")
                        if child_init_id in self.G:
                            alias_map[child_name] = child_init_id
                    else:
                        alias_map[child_name] = child_id
        return alias_map

    def _parse_imports(self, node_id: str, tree: ast.AST, alias_links: nx.DiGraph) -> None:
        """Parse import statements and create IMPORTS edges."""
        if node_id not in self.G or self.G.nodes[node_id].get("type") not in [
            NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD,
        ]:
            logger.warning("[imports:skip] invalid src code for parse: %s", node_id)
            return

        current_module = self.G.nodes[node_id].get("module", "")

        for node in ast.iter_child_nodes(tree):
            alias_map: dict = self.G.nodes[node_id].get("alias_to_entity")
            if alias_map is None:
                logger.warning("missing alias map for: %s", node_id)
                alias_map = self._init_alias_map(node_id)

            if isinstance(node, ast.Import):
                for al in node.names:
                    module_id = self._find_module_file(al.name)
                    alias = al.asname or al.name
                    if module_id:
                        self._add_edge(node_id, module_id, type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = module_id
                        self._propagate_aliases(node_id, alias, alias_links)

            elif isinstance(node, ast.ImportFrom):
                level = node.level
                abs_module = self._resolve_relative_module(current_module, node.module, level)
                if not abs_module:
                    continue
                module_id = self._find_module_file(abs_module)
                if module_id is None:
                    continue
                module_alias_map: dict = self.G.nodes[module_id].get("alias_to_entity")
                if module_alias_map is None:
                    logger.warning("missing alias map for: %s", module_id)
                    module_alias_map = self._init_alias_map(module_id)

                if any(al.name == "*" for al in node.names):
                    for alias, entity in module_alias_map.items():
                        self._add_edge(node_id, entity, type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = entity
                        self._propagate_aliases(node_id, alias, alias_links)
                    alias_links.add_edge(f"{module_id}:*", f"{node_id}:*")
                    continue

                for al in node.names:
                    target = al.name
                    alias = al.asname or al.name
                    if target in module_alias_map:
                        self._add_edge(node_id, module_alias_map[target], type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = module_alias_map[target]
                        self._propagate_aliases(node_id, alias, alias_links)
                    else:
                        alias_links.add_edge(f"{module_id}:{target}", f"{node_id}:{alias}")

    def _propagate_aliases(self, node_id: str, alias: str, alias_links: nx.DiGraph) -> None:
        """Propagate resolved aliases through deferred alias links."""
        alias_link = f"{node_id}:{alias}"
        entity = self.G.nodes[node_id].get("alias_to_entity", {}).get(alias)
        if not entity:
            logger.warning("missing entity for alias link: %s", alias_link)
            return

        for _, dst in alias_links.out_edges(alias_link):
            dst_node, dst_alias = dst.split(":", 1)
            alias_map: dict = self.G.nodes[dst_node].get("alias_to_entity")
            if alias_map is None:
                logger.warning("missing alias map for: %s", dst_node)
                alias_map = self._init_alias_map(dst_node)
            if alias_map.get(dst_alias) != entity:
                alias_map[dst_alias] = entity
                self._add_edge(dst_node, entity, type=EdgeType.IMPORTS, alias=dst_alias)
                self._propagate_aliases(dst_node, dst_alias, alias_links)

        star_link = f"{node_id}:*"
        for _, dst in alias_links.out_edges(star_link):
            dst_node, dst_alias = dst.split(":", 1)
            assert dst_alias == "*", f"Expected '*', got {dst_alias}"
            dst_alias = alias
            alias_map = self.G.nodes[dst_node].get("alias_to_entity")
            if alias_map is None:
                logger.warning("missing alias map for: %s", dst_node)
                alias_map = self._init_alias_map(dst_node)
            if alias_map.get(dst_alias) != entity:
                alias_map[dst_alias] = entity
                self._add_edge(dst_node, entity, type=EdgeType.IMPORTS, alias=dst_alias)
                self._propagate_aliases(dst_node, dst_alias, alias_links)

    # ------------------------------------------------------------------
    # Inheritance parsing
    # ------------------------------------------------------------------

    def _parse_inherits(self, node_id: str, tree: ast.AST) -> None:
        """Parse class bases and create INHERITS edges."""
        if (
            node_id not in self.G
            or self.G.nodes[node_id].get("type") != NodeType.CLASS
            or not isinstance(tree, ast.ClassDef)
        ):
            logger.warning("[inherits:skip] invalid src class for parse: %s", node_id)
            return

        for base in tree.bases:
            if isinstance(base, (ast.Name, ast.Attribute)):
                base_name = ast.unparse(base)
                entity = self._find_entity(node_id, base_name)
                if entity:
                    self._add_edge(node_id, entity, type=EdgeType.INHERITS)
                else:
                    logger.debug("[inherits:miss] base class not found: %s in %s", base_name, node_id)

    # ------------------------------------------------------------------
    # Invocation parsing
    # ------------------------------------------------------------------

    def _parse_invokes(self, node_id: str, tree: ast.AST) -> None:
        """Parse function/method calls and create INVOKES edges."""
        if node_id not in self.G or self.G.nodes[node_id].get("type") not in [
            NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD,
        ]:
            logger.warning("[invokes:skip] invalid src code for parse: %s", node_id)
            return

        ntype = self.G.nodes[node_id].get("type")
        nodes_to_walk: list = []
        if ntype == NodeType.FILE:
            for child in ast.iter_child_nodes(tree):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nodes_to_walk.append(child)
        elif ntype == NodeType.CLASS:
            for child in ast.iter_child_nodes(tree):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes_to_walk.append(child)
        else:
            nodes_to_walk.append(tree)

        for subtree in nodes_to_walk:
            for child in ast.walk(subtree):
                if isinstance(child, (ast.Call, ast.Await)):
                    if isinstance(child, ast.Await):
                        child = child.value
                    if not isinstance(child, ast.Call):
                        continue
                    func_name = ast.unparse(child.func)
                    entity = self._find_entity(node_id, func_name)
                    if entity:
                        if entity != node_id:
                            self._add_edge(node_id, entity, type=EdgeType.INVOKES)
                    else:
                        entity = self._find_entity_fuzzy(node_id, func_name)
                        if entity:
                            if entity != node_id:
                                self._add_edge(node_id, entity, type=EdgeType.INVOKES)
                        else:
                            logger.debug("[invokes:miss] invoked entity not found: %s in %s", func_name, node_id)

    # ------------------------------------------------------------------
    # Module / entity resolution helpers
    # ------------------------------------------------------------------

    def _resolve_relative_module(self, current_module: str, module: Optional[str], level: int) -> str:
        """Resolve a relative import to an absolute module path."""
        if level == 0:
            return module
        base_parts = current_module.split(".") if current_module else []
        base_parts = base_parts[: len(base_parts) - level] if level <= len(base_parts) else []
        if module:
            return ".".join(base_parts + [module]) if base_parts else module
        else:
            return ".".join(base_parts)

    def _find_module_file(self, module_name: str) -> Optional[str]:
        """Look up a module name and return the corresponding file node ID."""
        module_path = normalize_path("./" + module_name.replace(".", "/"))
        file_path = normalize_path(module_path + ".py")
        init_path = normalize_path(f"{module_path}/__init__.py")

        if file_path in self.G:
            return file_path
        elif init_path in self.G:
            return init_path
        else:
            return None

    def _find_entity(self, module_id: str, qual_name: str) -> Optional[str]:
        """Resolve a qualified name to a graph node ID via alias maps."""
        if module_id not in self.G:
            return None

        parts = qual_name.split(".")
        cur_entity = module_id
        for part in parts:
            alias_map: dict = self.G.nodes[cur_entity].get("alias_to_entity", {})
            if alias_map.get(part):
                cur_entity = alias_map[part]
            elif self.G.nodes[module_id].get("type") in [NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                parent_exists, parent_id = self.get_parent(module_id)
                if parent_exists:
                    return self._find_entity(parent_id, qual_name)
            else:
                return None
        return cur_entity

    # ------------------------------------------------------------------
    # Fuzzy entity resolution
    # ------------------------------------------------------------------

    def _find_entity_fuzzy(self, node_id: str, qual_name: str) -> Optional[str]:
        """Fuzzily resolve entities for calls that cannot be precisely parsed.

        Strategy:
        1. ``self.method()`` -- look up method in the current class and its
           inheritance chain
        2. ``self.attr.method()`` -- infer the type of *attr*, or fall back
           to a global match by method name
        3. ``super().method()`` -- look up method in parent classes
        4. ``var.method()`` -- try the current class first, then global search
        5. ``var.attr.method()`` -- take the final method name and search
           the current class or globally
        """
        parts = qual_name.split(".")
        if not parts:
            return None

        containing_class = self._get_containing_class(node_id)

        if parts[0] == "self" and len(parts) >= 2:
            method_name = parts[-1]

            if len(parts) == 2 and containing_class:
                entity = self._find_method_in_class_hierarchy(containing_class, method_name)
                if entity:
                    return entity

            if len(parts) >= 3 and containing_class:
                attr_name = parts[1]
                attr_type = self._infer_attribute_type(containing_class, attr_name)
                if attr_type:
                    entity = self._find_method_in_class_hierarchy(attr_type, method_name)
                    if entity:
                        return entity

            return self._find_method_by_name_global(method_name)

        if parts[0] == "super()" and len(parts) == 2 and containing_class:
            method_name = parts[1]
            return self._find_method_in_parent_classes(containing_class, method_name)

        if len(parts) >= 2 and containing_class:
            method_name = parts[-1]

            if len(parts) == 2:
                entity = self._find_method_in_class_hierarchy(containing_class, method_name)
                if entity:
                    return entity

            var_name = parts[0]
            var_type = self._infer_local_var_type(node_id, var_name)
            if var_type:
                if len(parts) > 2:
                    type_infer_failed = False
                    for i in range(1, len(parts) - 1):
                        attr_name = parts[i]
                        attr_type = self._infer_attribute_type(var_type, attr_name)
                        if attr_type:
                            var_type = attr_type
                        else:
                            type_infer_failed = True
                            break

                    if type_infer_failed:
                        return self._find_method_by_name_global(method_name)

                entity = self._find_method_in_class_hierarchy(var_type, method_name)
                if entity:
                    return entity

            return self._find_method_by_name_global(method_name)

        return None

    def _infer_local_var_type(self, node_id: str, var_name: str) -> Optional[str]:
        """Infer the type of a local variable from assignment patterns.

        Strategy:
        1. ``var = self.method()`` -- return the current class
        2. ``var = SomeClass(...)`` -- resolve SomeClass
        3. ``var = some_instance.method()`` -- heuristic
        """
        if node_id not in self.G:
            return None

        node_ast = self.G.nodes[node_id].get("ast")
        if not node_ast:
            return None

        containing_class = self._get_containing_class(node_id)
        _, file_id = self.get_parent(node_id)

        for stmt in ast.walk(node_ast):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if (
                            isinstance(stmt.value, ast.Call)
                            and isinstance(stmt.value.func, ast.Attribute)
                            and isinstance(stmt.value.func.value, ast.Name)
                            and stmt.value.func.value.id == "self"
                        ):
                            if containing_class:
                                return containing_class

                        # var = SomeClass(...)
                        type_name = self._extract_type_from_value(stmt.value)
                        if type_name:
                            entity = self._find_entity(file_id, type_name) if file_id else None
                            if entity:
                                return entity
                            if containing_class:
                                entity = self._find_entity(containing_class, type_name)
                                if entity:
                                    return entity

        return None

    def _get_containing_class(self, node_id: str) -> Optional[str]:
        """Return the class node ID that contains the given node, if any."""
        ntype = self.G.nodes.get(node_id, {}).get("type")
        if ntype == NodeType.CLASS:
            return node_id
        elif ntype == NodeType.METHOD:
            _, parent_id = self.get_parent(node_id)
            if parent_id and self.G.nodes.get(parent_id, {}).get("type") == NodeType.CLASS:
                return parent_id
        return None

    def _find_method_in_class_hierarchy(self, class_id: str, method_name: str) -> Optional[str]:
        """Search for a method in the class and its inheritance chain."""
        if class_id not in self.G:
            return None

        alias_map = self.G.nodes[class_id].get("alias_to_entity", {})
        if method_name in alias_map:
            return alias_map[method_name]

        for _, dst, edata in self.G.out_edges(class_id, data=True):
            if edata.get("type") == EdgeType.INHERITS:
                result = self._find_method_in_class_hierarchy(dst, method_name)
                if result:
                    return result

        return None

    def _find_method_in_parent_classes(self, class_id: str, method_name: str) -> Optional[str]:
        """Search for a method in parent classes only (for super() calls)."""
        if class_id not in self.G:
            return None

        for _, dst, edata in self.G.out_edges(class_id, data=True):
            if edata.get("type") == EdgeType.INHERITS:
                alias_map = self.G.nodes[dst].get("alias_to_entity", {})
                if method_name in alias_map:
                    return alias_map[method_name]

                result = self._find_method_in_class_hierarchy(dst, method_name)
                if result:
                    return result

        return None

    def _infer_attribute_type(self, class_id: str, attr_name: str) -> Optional[str]:
        """Try to infer the type of a class attribute.

        Strategy:
        1. Look for assignments in ``__init__`` such as ``self.attr = SomeClass(...)``
        2. Look for type annotations like ``attr: SomeClass``
        """
        if class_id not in self.G:
            return None

        class_ast = self.G.nodes[class_id].get("ast")
        if not isinstance(class_ast, ast.ClassDef):
            return None

        attr_variants = [attr_name, f"_{attr_name}"]

        _, file_id = self.get_parent(class_id)

        for node in class_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and target.attr in attr_variants
                            ):
                                type_name = self._extract_type_from_value(stmt.value)
                                if type_name:
                                    entity = self._find_entity(file_id, type_name) if file_id else None
                                    if entity:
                                        return entity
                                    return self._find_entity(class_id, type_name)

        for node in class_ast.body:
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in attr_variants:
                    if node.annotation:
                        type_name = ast.unparse(node.annotation)
                        entity = self._find_entity(file_id, type_name) if file_id else None
                        if entity:
                            return entity
                        return self._find_entity(class_id, type_name)

        return None

    def _extract_type_from_value(self, value_node: ast.AST) -> Optional[str]:
        """Extract a type name from an assignment value node."""
        # self.attr = SomeClass(...)
        if isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                return value_node.func.id
            elif isinstance(value_node.func, ast.Attribute):
                return ast.unparse(value_node.func)

        if isinstance(value_node, ast.BoolOp) and isinstance(value_node.op, ast.Or):
            for val in value_node.values:
                result = self._extract_type_from_value(val)
                if result:
                    return result

        if isinstance(value_node, ast.IfExp):
            result = self._extract_type_from_value(value_node.orelse)
            if result:
                return result
            result = self._extract_type_from_value(value_node.body)
            if result:
                return result

        return None

    def _find_method_by_name_global(self, method_name: str) -> Optional[str]:
        """Search for methods with the same name across the entire repo.

        Returns only a unique match; if multiple methods share the same name,
        returns ``None`` to avoid false positives.
        """
        COMMON_METHODS = {
            "__init__", "__str__", "__repr__", "__eq__", "__hash__",
            "get", "set", "add", "remove", "update", "delete", "save",
            "clone", "copy", "items", "keys", "values", "as_sql",
        }
        if method_name in COMMON_METHODS or method_name.startswith("_"):
            return None

        matches = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == NodeType.METHOD:
                if attrs.get("name") == method_name:
                    matches.append(nid)

        if len(matches) == 1:
            return matches[0]

        return None

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_parent(self, node_id: str) -> Tuple[bool, Optional[str]]:
        """Return ``(parent_exists_in_graph, parent_id)`` for the given node.

        For the root node (``"."``), returns ``(True, None)``.
        """
        nid = normalize_path(node_id)
        if nid not in self.G:
            return False, None

        if nid == ".":
            return True, None
        elif ":" in nid:
            path_part, qual = nid.split(":", 1)
            parts = qual.split(".")
            if len(parts) <= 1:
                parent_id = path_part
            else:
                parent_qual = ".".join(parts[:-1])
                parent_id = f"{path_part}:{parent_qual}"
            parent_id = normalize_path(parent_id)
        else:
            parent_id = normalize_path(Path(nid).parent)
        return parent_id in self.G, parent_id

    def get_name(
        self,
        nid: str,
        ntype: Optional[str] = None,
        for_print: bool = False,
        with_badge: bool = False,
    ) -> str:
        """Return a human-readable name for the node.

        Args:
            nid: Node identifier.
            ntype: Node type override (if ``None``, looked up from graph).
            for_print: If ``True``, use a short display-friendly form.
            with_badge: If ``True``, append a type badge (e.g. ``@file``).
        """
        badge_map = {
            NodeType.DIRECTORY: "@dir",
            NodeType.FILE: "@file",
            NodeType.CLASS: "@class",
            NodeType.FUNCTION: "@func",
            NodeType.METHOD: "@method",
        }

        if ntype is None:
            ntype = self.G.nodes.get(nid, {}).get("type")

        name = ""
        if ntype == NodeType.DIRECTORY or ntype == NodeType.FILE:
            if nid == ".":
                name = "."
            else:
                name = nid.split("/")[-1]
        else:
            if ":" in nid:
                qual = nid.split(":", 1)[1]
                parts = [p for p in qual.split(".") if p]
                last = parts[-1] if parts else qual
                name = f".{last}" if for_print and len(parts) > 1 else last
            else:
                name = nid.split("/")[-1]

        if with_badge and ntype in badge_map:
            name += f" {badge_map[ntype]}"

        if ntype == NodeType.FILE and not for_print:
            name = name.rstrip(".py")

        return name

    def find_node(self, path: str, suffix_match: bool = True) -> Optional[str]:
        """Find a node by exact or suffix match on its ID."""
        norm_input = normalize_path(path)
        if norm_input in self.G:
            return norm_input

        if suffix_match:
            for nid in self.G.nodes:
                if nid.endswith(norm_input):
                    return nid
        return None

    def find_file(self, path: str, suffix_match: bool = True) -> Optional[str]:
        """Find a FILE node by exact or suffix match."""
        norm_input = normalize_path(path)
        if norm_input in self.G and self.G.nodes[norm_input].get("type") == NodeType.FILE:
            return norm_input

        if suffix_match:
            for nid, attrs in self.G.nodes(data=True):
                if attrs.get("type") == NodeType.FILE and nid.endswith(norm_input):
                    return nid
        return None

    def all_paths(self, include_types: List[str]) -> List[str]:
        """Return sorted list of node IDs whose type is in *include_types*."""
        include_set = set(include_types)
        return sorted(
            nid for nid, attrs in self.G.nodes(data=True)
            if attrs.get("type") in include_set
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, dep_to_rpg_map: Optional[Dict[str, List[str]]] = None) -> dict:
        """Serialize graph to dict (AST objects are excluded).

        Enriches nodes with pre-aggregated dependency summaries:
        - File nodes: ``imports_from`` (external files imported)
        - Function/Method nodes: ``calls``, ``called_by``
        - Class nodes: ``calls``, ``called_by``, ``inherits``, ``inherited_by``

        Args:
            dep_to_rpg_map: Optional mapping from dep node IDs to RPG node IDs.
                If provided, adds ``rpg_nodes`` field to each node and
                ``src_rpg_nodes``/``dst_rpg_nodes`` to each edge.
        """
        # Pre-compute invokes / inherits indexes from edges
        calls_out: Dict[str, set] = defaultdict(set)
        calls_in: Dict[str, set] = defaultdict(set)
        inherits_out: Dict[str, set] = defaultdict(set)
        inherits_in: Dict[str, set] = defaultdict(set)

        for src, dst, edata in self.G.edges(data=True):
            etype = edata.get("type", "")
            if etype == EdgeType.INVOKES:
                calls_out[src].add(dst)
                calls_in[dst].add(src)
            elif etype == EdgeType.INHERITS:
                inherits_out[src].add(dst)
                inherits_in[dst].add(src)

        data: Dict[str, Any] = {
            "repo_dir": self.repo_dir,
            "nodes": {},
            "edges": [],
        }

        for nid, attrs in self.G.nodes(data=True):
            # Exclude AST, code, and any stale derived fields
            node_data = {
                k: v for k, v in attrs.items()
                if k not in ("ast", "code") and k not in self._DERIVED_FIELDS
            }

            ntype = node_data.get("type", "")

            # File nodes: imports_from
            if ntype == NodeType.FILE:
                alias_map = node_data.get("alias_to_entity", {})
                file_prefix = nid.rsplit(":", 1)[0] if ":" in nid else nid
                # Extract unique external file paths from alias targets
                ext_files: set = set()
                for target in alias_map.values():
                    target_file = target.split(":")[0] if ":" in target else target
                    if target_file != file_prefix and target_file != nid:
                        ext_files.add(target_file)
                if ext_files:
                    node_data["imports_from"] = sorted(ext_files)

            # Function/Method/Class nodes: calls, called_by
            if ntype in (NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS):
                out = calls_out.get(nid)
                if out:
                    node_data["calls"] = sorted(out)
                inp = calls_in.get(nid)
                if inp:
                    node_data["called_by"] = sorted(inp)

            # Class nodes: inherits, inherited_by
            if ntype == NodeType.CLASS:
                inh_out = inherits_out.get(nid)
                if inh_out:
                    node_data["inherits"] = sorted(inh_out)
                inh_in = inherits_in.get(nid)
                if inh_in:
                    node_data["inherited_by"] = sorted(inh_in)

            if dep_to_rpg_map is not None:
                node_data["rpg_nodes"] = dep_to_rpg_map.get(nid, [])
            data["nodes"][nid] = node_data

        for u, v, attrs in self.G.edges(data=True):
            edge_data: Dict[str, Any] = {
                "src": u,
                "dst": v,
                "attrs": dict(attrs),
            }
            if dep_to_rpg_map is not None:
                edge_data["src_rpg_nodes"] = dep_to_rpg_map.get(u, [])
                edge_data["dst_rpg_nodes"] = dep_to_rpg_map.get(v, [])
            data["edges"].append(edge_data)

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Reconstruct graph from dict.

        After loading, call :meth:`reparse_ast` to restore AST objects.
        """
        obj = cls(repo_dir=data.get("repo_dir", ""))

        for nid, attrs in data["nodes"].items():
            obj.G.add_node(nid, **attrs)

        for e in data["edges"]:
            u = e["src"]
            v = e["dst"]
            attrs = e["attrs"]
            obj.G.add_edge(u, v, **attrs)

        return obj

    def _read_code(self, nid: str) -> str:
        """Read source code for a node from disk via code_path.

        Falls back to the in-memory ``code`` attribute if present.
        """
        attrs = self.G.nodes.get(nid, {})
        content = attrs.get("code")
        if content:
            return content
        code_path = attrs.get("code_path", "")
        if code_path:
            try:
                with open(code_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def reparse_ast(self, filter_func: Callable[[str], bool] = _exclude_irrelevant_for_parse) -> None:
        """Reparse source code to restore AST and code structure.

        Must be called after :meth:`from_dict` to reconstruct AST objects and
        semantic edges that are not serialized. Mirrors the language
        dispatch in :meth:`parse`: Python files keep the original ast path,
        non-Python files re-run through ``lang_parser``.
        """
        lp_results: list[tuple[str, lang_parser.LPFileResult]] = []
        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE or not filter_func(nid):
                continue

            content = self._read_code(nid)
            language = lang_parser.detect_language(nid)
            if language == "python":
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue

                self.G.nodes[nid]["ast"] = tree
                self.G.nodes[nid]["language"] = "python"

                # Rebuild functions / classes nodes
                self._parse_file(nid, tree, content)
                continue

            if language is None:
                continue

            try:
                result = lang_parser.parse_file(nid, content)
            except lang_parser.NotSupported:
                continue
            self._parse_lp_file_result(nid, result)
            lp_results.append((nid, result))

        # Second pass for lang_parser invokes (needs full unit registry).
        for nid, result in lp_results:
            self._parse_lp_invoke_dependencies(nid, result)

        # Re-run import / invoke / inherit pass
        alias_links: nx.DiGraph = nx.DiGraph()
        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") in [NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                self._init_alias_map(nid)
                node_ast = attrs.get("ast")
                if node_ast is not None:
                    self._parse_imports(nid, node_ast, alias_links)

        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") == NodeType.CLASS:
                node_ast = attrs.get("ast")
                if node_ast is not None:
                    self._parse_inherits(nid, node_ast)

        for nid, attrs in list(self.G.nodes(data=True)):
            node_ast = attrs.get("ast")
            if node_ast is not None:
                self._parse_invokes(nid, node_ast)

        logger.info("AST re-parsed & semantic edges reconstructed")

    # ------------------------------------------------------------------
    # LPFileResult parsing for non-Python languages
    # ------------------------------------------------------------------

    def _parse_lp_file_result(
        self,
        file_id: str,
        result: lang_parser.LPFileResult,
        *,
        include_dependencies: bool = True,
    ) -> None:
        """Add graph nodes and dependency edges from a language-parser result."""
        file_attrs = self.G.nodes[file_id]
        file_attrs["language"] = result.language
        file_attrs["unit_type"] = "file"
        if result.syntax_error is not None:
            file_attrs["syntax_error"] = result.syntax_error
        else:
            file_attrs.pop("syntax_error", None)

        file_code_path = file_attrs.get("code_path", "")
        unit_node_ids: dict[int, str] = {}
        class_like_by_name: dict[str, str] = {}
        import_nodes_by_line: dict[int, str] = {}
        import_nodes_by_module: dict[str, str] = {}

        for index, unit in enumerate(result.units):
            unit_id = self._lp_unit_node_id(file_id, unit, index)
            unit_node_ids[index] = unit_id
            if unit.unit_type in self._LP_CLASS_LIKE_UNIT_TYPES and unit.name:
                class_like_by_name[unit.name] = unit_id

        for index, unit in enumerate(result.units):
            unit_id = unit_node_ids[index]
            parent_id = file_id
            if unit.parent:
                parent_id = class_like_by_name.get(unit.parent, file_id)

            self._add_node(
                unit_id,
                type=self._lp_node_type(unit.unit_type),
                name=unit.name or self._lp_fallback_unit_name(unit, index),
                parent_id=parent_id,
                **self._lp_unit_node_attrs(unit, file_code_path),
            )

            if unit.unit_type == "import":
                if unit.line_start is not None:
                    import_nodes_by_line.setdefault(unit.line_start, unit_id)
                for module in (unit.name, unit.extra.get("module"), unit.extra.get("import_path")):
                    if module:
                        import_nodes_by_module[module] = unit_id

        if include_dependencies:
            self._parse_lp_dependencies(file_id, result, import_nodes_by_line, import_nodes_by_module)

    def _lp_node_type(self, unit_type: str) -> NodeType:
        """Map a language-parser unit type onto an existing graph node type."""
        return self._LP_NODE_TYPES.get(unit_type, NodeType.MODULE)

    def _lp_unit_node_id(self, file_id: str, unit: lang_parser.LPCodeUnit, index: int) -> str:
        """Build a deterministic graph node ID for a language-parser code unit."""
        line = unit.line_start if unit.line_start is not None else index + 1
        if unit.unit_type == "import":
            import_index = (unit.extra or {}).get("import_index")
            if import_index is not None:
                return normalize_path(f"{file_id}:import:{line}:{import_index}")
            return normalize_path(f"{file_id}:import:{line}")
        if unit.unit_type == "package":
            return normalize_path(f"{file_id}:package:{line}")
        if unit.parent and unit.name:
            return normalize_path(f"{file_id}:{unit.parent}.{unit.name}")
        if unit.name:
            return normalize_path(f"{file_id}:{unit.name}")
        return normalize_path(f"{file_id}:{unit.unit_type}:{line}")

    def _lp_fallback_unit_name(self, unit: lang_parser.LPCodeUnit, index: int) -> str:
        line = unit.line_start if unit.line_start is not None else index + 1
        return f"{unit.unit_type}:{line}"

    def _lp_unit_node_attrs(self, unit: lang_parser.LPCodeUnit, file_code_path: str) -> dict[str, Any]:
        extra = dict(unit.extra or {})
        attrs: dict[str, Any] = {
            "language": unit.language,
            "unit_type": unit.unit_type,
            "start_line": unit.line_start,
            "end_line": unit.line_end,
            "code": unit.code,
            "code_path": file_code_path,
            "extra": extra,
        }
        for key, value in extra.items():
            attr_key = "import_module" if key == "module" else key
            if attr_key in {"module", "name", "type"}:
                attr_key = f"unit_{attr_key}"
            if attr_key not in attrs:
                attrs[attr_key] = value
        return attrs

    def _parse_lp_dependencies(
        self,
        file_id: str,
        result: lang_parser.LPFileResult,
        import_nodes_by_line: dict[int, str],
        import_nodes_by_module: dict[str, str],
    ) -> None:
        for dependency_index, dep in enumerate(result.dependencies):
            if dep.relation == "imports":
                self._add_lp_import_edge(
                    file_id,
                    dep,
                    dependency_index,
                    import_nodes_by_line,
                    import_nodes_by_module,
                )
            elif dep.relation == "contains":
                self._add_lp_contains_edge(file_id, dep)
            elif dep.relation == "invokes":
                self._add_lp_invoke_edge(file_id, dep)
            elif dep.relation == "inherits":
                self._add_lp_inherit_edge(file_id, dep)

    def _parse_lp_invoke_dependencies(self, file_id: str, result: lang_parser.LPFileResult) -> None:
        for dep in result.dependencies:
            if dep.relation == "invokes":
                self._add_lp_invoke_edge(file_id, dep)

    def _add_lp_import_edge(
        self,
        file_id: str,
        dep: lang_parser.LPDependency,
        dependency_index: int,
        import_nodes_by_line: dict[int, str],
        import_nodes_by_module: dict[str, str],
    ) -> None:
        src_id = self._resolve_lp_reference(dep.src, file_id) or file_id
        target_id = self._resolve_lp_import_destination(file_id, dep.dst, dep)
        resolved = target_id is not None
        if target_id is None:
            target_id = self._ensure_lp_import_placeholder(
                file_id,
                dep,
                dependency_index,
                import_nodes_by_line,
                import_nodes_by_module,
            )
        else:
            self._mark_lp_import_unit(dep, import_nodes_by_line, import_nodes_by_module, True, target_id)

        self._add_edge(
            src_id,
            target_id,
            type=EdgeType.IMPORTS,
            **self._lp_dependency_edge_attrs(dep, resolved),
        )

    def _add_lp_contains_edge(self, file_id: str, dep: lang_parser.LPDependency) -> None:
        src_id = self._resolve_lp_reference(dep.src, file_id)
        dst_id = self._resolve_lp_reference(dep.dst, file_id)
        if src_id and dst_id:
            self._add_edge(src_id, dst_id, type=EdgeType.CONTAINS, **self._lp_dependency_edge_attrs(dep, True))

    def _add_lp_inherit_edge(self, file_id: str, dep: lang_parser.LPDependency) -> None:
        src_id = self._resolve_lp_reference(dep.src, file_id)
        dst_id = self._resolve_lp_reference(dep.dst, file_id)
        if src_id and dst_id and src_id != dst_id:
            self._add_edge(src_id, dst_id, type=EdgeType.INHERITS, **self._lp_dependency_edge_attrs(dep, True))

    def _add_lp_invoke_edge(self, file_id: str, dep: lang_parser.LPDependency) -> None:
        src_id = self._resolve_lp_reference(dep.src, file_id) or file_id
        dst_id = self._resolve_lp_invoke_destination(file_id, dep)
        if dst_id is None or dst_id == src_id:
            return
        self._add_edge(
            src_id,
            dst_id,
            type=EdgeType.INVOKES,
            **self._lp_dependency_edge_attrs(dep, True),
        )

    def _lp_dependency_edge_attrs(self, dep: lang_parser.LPDependency, resolved: bool) -> dict[str, Any]:
        confidence = "resolved" if resolved else (dep.confidence or "unresolved")
        attrs: dict[str, Any] = {
            "relation": dep.relation,
            "symbol": dep.symbol,
            "line": dep.line,
            "confidence": confidence,
            "resolved": resolved,
            "extra": dict(dep.extra or {}),
        }
        if dep.dst is not None:
            attrs["import_module"] = dep.dst
        if not resolved:
            attrs["heuristic"] = True
        for key, value in (dep.extra or {}).items():
            attr_key = "unit_type" if key == "type" else key
            if attr_key not in attrs and attr_key != "type":
                attrs[attr_key] = value
        return attrs

    def _resolve_lp_import_destination(
        self,
        file_id: str,
        module: str | None,
        dep: lang_parser.LPDependency | None = None,
    ) -> Optional[str]:
        if not module:
            return None

        if self._is_c_family_import(file_id, dep):
            if (dep.extra or {}).get("include_style") != "quote":
                return None
            return self._resolve_c_local_include(file_id, module)

        if self._is_rust_import(dep):
            return self._resolve_rust_import(file_id, module, dep)

        direct_id = normalize_path(module)
        if direct_id in self.G and self.G.nodes[direct_id].get("type") == NodeType.FILE:
            return direct_id

        if module.startswith("."):
            return self._resolve_relative_ts_js_import(file_id, module)

        go_target = self._resolve_go_module_import(module)
        if go_target is not None:
            return go_target

        return None

    def _resolve_relative_ts_js_import(self, file_id: str, module: str) -> Optional[str]:
        base_dir = PurePosixPath(file_id).parent.as_posix()
        if base_dir == ".":
            base_dir = ""
        raw_path = posixpath.normpath(posixpath.join(base_dir, module)).removeprefix("./")
        extension = posixpath.splitext(raw_path)[1]

        candidates: list[str] = []
        if extension:
            candidates.append(raw_path)
        else:
            candidates.extend(f"{raw_path}{ext}" for ext in self._TS_JS_IMPORT_EXTENSIONS)
            candidates.extend(posixpath.join(raw_path, f"index{ext}") for ext in self._TS_JS_IMPORT_EXTENSIONS)

        for candidate in candidates:
            candidate_id = normalize_path(candidate)
            if candidate_id in self.G and self.G.nodes[candidate_id].get("type") == NodeType.FILE:
                return candidate_id
        return None

    def _is_c_family_import(self, file_id: str, dep: lang_parser.LPDependency | None) -> bool:
        if dep is None:
            return False
        extra = dep.extra or {}
        language = extra.get("language")
        import_kind = extra.get("import_kind")
        if language in {"c", "cpp"} or import_kind in {"c_include", "cpp_include"}:
            return True
        return PurePosixPath(file_id).suffix.lower() in self._C_FAMILY_EXTENSIONS and extra.get("include_style")

    def _resolve_c_local_include(self, file_id: str, include_path: str) -> Optional[str]:
        base_dir = PurePosixPath(file_id).parent.as_posix()
        if base_dir == ".":
            base_dir = ""

        candidates = []
        joined = posixpath.join(base_dir, include_path) if base_dir else include_path
        candidates.append(posixpath.normpath(joined).removeprefix("./"))
        candidates.append(posixpath.normpath(include_path).removeprefix("./"))

        for candidate in candidates:
            candidate_id = normalize_path(candidate)
            if candidate_id in self.G and self.G.nodes[candidate_id].get("type") == NodeType.FILE:
                return candidate_id

        suffix = normalize_path(posixpath.normpath(include_path).removeprefix("./"))
        matches = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") != NodeType.FILE:
                continue
            if PurePosixPath(node_id).suffix.lower() not in self._C_FAMILY_EXTENSIONS:
                continue
            if node_id == suffix or node_id.endswith(f"/{suffix}"):
                matches.append(node_id)
        matches.sort()
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_lp_invoke_destination(self, file_id: str, dep: lang_parser.LPDependency) -> Optional[str]:
        language = (dep.extra or {}).get("language")
        if language in {"typescript", "javascript"}:
            return self._resolve_ecmascript_invoke(file_id, dep)
        if language == "go":
            return self._resolve_go_invoke(file_id, dep)
        if language in {"c", "cpp"}:
            return self._resolve_c_invoke(file_id, dep)
        if language == "rust":
            return self._resolve_rust_invoke(file_id, dep)
        return self._resolve_lp_reference(dep.dst or dep.symbol, file_id)

    def _resolve_ecmascript_invoke(self, file_id: str, dep: lang_parser.LPDependency) -> Optional[str]:
        extra = dep.extra or {}
        module = extra.get("module")
        symbol = extra.get("imported") or dep.symbol or dep.dst
        if module:
            target_file = self._resolve_lp_import_destination(file_id, module)
            if target_file is None:
                return None
            named_target = self._find_named_unit_in_file(target_file, symbol)
            if named_target is not None:
                return named_target
            if extra.get("kind") == "default":
                return self._find_default_ecmascript_export_in_file(target_file)
            return None
        return self._find_named_unit_in_file(file_id, symbol)

    def _resolve_go_invoke(self, file_id: str, dep: lang_parser.LPDependency) -> Optional[str]:
        extra = dep.extra or {}
        symbol = dep.symbol or dep.dst
        if not symbol:
            return None
        module = extra.get("module")
        if module:
            package_file = self._resolve_lp_import_destination(file_id, module)
            if package_file is None:
                return None
            return self._find_go_package_symbol(package_file, symbol)
        return self._find_go_package_symbol(file_id, symbol)

    def _resolve_c_invoke(self, file_id: str, dep: lang_parser.LPDependency) -> Optional[str]:
        symbol = dep.symbol or dep.dst
        if not symbol:
            return None

        extra = dep.extra or {}
        call_kind = extra.get("call_kind")
        qualifier = extra.get("qualifier")
        if call_kind == "static" and qualifier:
            return self._find_c_static_method(file_id, qualifier, symbol)
        if call_kind == "constructor":
            same_file_class = self._find_c_symbol_in_file(file_id, symbol, {NodeType.CLASS})
            if same_file_class is not None:
                return same_file_class
            return self._find_c_directory_symbol(file_id, symbol, {NodeType.CLASS})

        same_file = self._find_c_symbol_in_file(file_id, symbol, {NodeType.FUNCTION, NodeType.METHOD})
        if same_file is not None:
            return same_file
        return self._find_c_directory_symbol(file_id, symbol, {NodeType.FUNCTION, NodeType.METHOD})

    def _is_rust_import(self, dep: lang_parser.LPDependency | None) -> bool:
        if dep is None:
            return False
        extra = dep.extra or {}
        import_kind = extra.get("import_kind") or ""
        return extra.get("language") == "rust" or import_kind.startswith("rust_")

    def _resolve_rust_import(
        self,
        file_id: str,
        module: str | None,
        dep: lang_parser.LPDependency | None = None,
    ) -> Optional[str]:
        if not module:
            return None
        extra = dep.extra or {} if dep is not None else {}
        import_kind = extra.get("import_kind")
        if import_kind == "rust_mod_decl":
            return self._resolve_rust_mod_decl(file_id, module)
        return self._resolve_rust_path_to_file(file_id, module)

    def _resolve_rust_mod_decl(self, file_id: str, mod_name: str) -> Optional[str]:
        file_dir = self._node_parent_dir(file_id)
        candidates = [
            posixpath.join(file_dir, f"{mod_name}.rs") if file_dir else f"{mod_name}.rs",
            posixpath.join(file_dir, mod_name, "mod.rs") if file_dir else posixpath.join(mod_name, "mod.rs"),
        ]
        return self._first_existing_rust_file(candidates)

    def _resolve_rust_path_to_file(self, file_id: str, module_path: str | None) -> Optional[str]:
        if not module_path:
            return None
        if module_path.startswith(("std::", "core::", "alloc::")):
            return None
        if module_path.startswith("crate::"):
            return self._resolve_rust_crate_path(file_id, module_path[len("crate::"):])
        if module_path.startswith("self::"):
            return self._resolve_rust_self_path(file_id, module_path[len("self::"):])
        if module_path.startswith("super"):
            return self._resolve_rust_super_path(file_id, module_path)

        crate_target = self._resolve_rust_crate_path(file_id, module_path)
        if crate_target is not None:
            return crate_target
        return self._resolve_rust_self_path(file_id, module_path)

    def _resolve_rust_crate_path(self, file_id: str, path_after_crate: str) -> Optional[str]:
        src_root = self._find_rust_crate_src_root(file_id)
        if src_root is None:
            return None
        return self._resolve_rust_module_path_from_bases(path_after_crate, [src_root])

    def _resolve_rust_self_path(self, file_id: str, module_path: str) -> Optional[str]:
        return self._resolve_rust_module_path_from_bases(module_path, self._rust_self_base_dirs(file_id))

    def _resolve_rust_super_path(self, file_id: str, module_path: str) -> Optional[str]:
        levels = 0
        rest = module_path
        while rest == "super" or rest.startswith("super::"):
            levels += 1
            rest = "" if rest == "super" else rest[len("super::"):]
        base_dir = self._rust_super_base_dir(file_id, levels)
        if base_dir is None:
            return None
        if not rest:
            mod_rs = posixpath.join(base_dir, "mod.rs") if base_dir else "mod.rs"
            return self._first_existing_rust_file([mod_rs])
        return self._resolve_rust_module_path_from_bases(rest, [base_dir])

    def _resolve_rust_module_path_from_bases(self, module_path: str, base_dirs: list[str]) -> Optional[str]:
        segments = self._rust_path_segments(module_path)
        if not segments:
            return None
        candidates: list[str] = []
        for base_dir in base_dirs:
            normalized_base = normalize_path(base_dir)
            if normalized_base == ".":
                normalized_base = ""
            for end in range(len(segments), 0, -1):
                partial = posixpath.join(*segments[:end])
                if normalized_base:
                    candidates.append(posixpath.join(normalized_base, f"{partial}.rs"))
                    candidates.append(posixpath.join(normalized_base, partial, "mod.rs"))
                else:
                    candidates.append(f"{partial}.rs")
                    candidates.append(posixpath.join(partial, "mod.rs"))
        return self._first_existing_rust_file(candidates)

    def _rust_path_segments(self, module_path: str) -> list[str]:
        return [
            segment
            for segment in module_path.split("::")
            if segment and segment not in {"crate", "self", "super", "*"}
        ]

    def _first_existing_rust_file(self, candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            candidate_id = normalize_path(posixpath.normpath(candidate).removeprefix("./"))
            if candidate_id in self.G and self.G.nodes[candidate_id].get("type") == NodeType.FILE:
                return candidate_id
        return None

    def _find_rust_crate_src_root(self, file_id: str) -> Optional[str]:
        parts = PurePosixPath(file_id).parts[:-1]
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "src":
                return "/".join(parts[:index + 1])
        return None

    def _rust_self_base_dirs(self, file_id: str) -> list[str]:
        path = PurePosixPath(file_id)
        parent = path.parent.as_posix()
        if parent == ".":
            parent = ""
        if path.name == "mod.rs":
            return [parent]
        stem_dir = posixpath.join(parent, path.stem) if parent else path.stem
        return [stem_dir, parent]

    def _rust_super_base_dir(self, file_id: str, levels: int) -> Optional[str]:
        if levels <= 0:
            return self._node_parent_dir(file_id)
        path = PurePosixPath(file_id)
        base = path.parent
        remaining = levels
        if path.name == "mod.rs":
            base = base.parent
            remaining -= 1
        else:
            remaining -= 1
        while remaining > 0:
            base = base.parent
            remaining -= 1
        base_posix = base.as_posix()
        return "" if base_posix == "." else base_posix

    def _resolve_rust_invoke(self, file_id: str, dep: lang_parser.LPDependency) -> Optional[str]:
        extra = dep.extra or {}
        symbol = dep.symbol or dep.dst
        if not symbol:
            return None

        if extra.get("call_kind") == "path":
            qualifier = extra.get("qualifier") or dep.dst
            if not qualifier:
                return None
            target_file = self._resolve_rust_path_to_file(file_id, qualifier)
            if target_file is not None:
                target = self._find_rust_symbol_in_file(target_file, symbol, {NodeType.FUNCTION, NodeType.METHOD})
                if target is not None:
                    return target
            if "::" not in qualifier:
                return self._find_rust_qualified_in_file(file_id, qualifier, symbol)
            return None

        same_file = self._find_rust_symbol_in_file(file_id, symbol, {NodeType.FUNCTION, NodeType.METHOD})
        if same_file is not None:
            return same_file
        return self._find_rust_crate_function(file_id, symbol)

    def _find_rust_symbol_in_file(
        self,
        file_id: str,
        symbol: str | None,
        node_types: set[NodeType],
    ) -> Optional[str]:
        if not symbol:
            return None
        direct_id = normalize_path(f"{file_id}:{symbol}")
        if direct_id in self.G and self.G.nodes[direct_id].get("type") in node_types:
            return direct_id

        matches = []
        prefix = f"{file_id}:"
        for node_id, attrs in self.G.nodes(data=True):
            if not node_id.startswith(prefix):
                continue
            if attrs.get("name") != symbol:
                continue
            if attrs.get("type") in node_types:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_rust_qualified_in_file(self, file_id: str, qualifier: str, symbol: str | None) -> Optional[str]:
        if not symbol:
            return None
        qualified_id = normalize_path(f"{file_id}:{qualifier}.{symbol}")
        if qualified_id in self.G and self.G.nodes[qualified_id].get("type") == NodeType.METHOD:
            return qualified_id
        return None

    def _find_rust_crate_function(self, file_id: str, symbol: str | None) -> Optional[str]:
        if not symbol:
            return None
        src_root = self._find_rust_crate_src_root(file_id)
        if src_root is None:
            return None
        matches = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") != NodeType.FUNCTION or attrs.get("name") != symbol:
                continue
            node_file = node_id.split(":", 1)[0]
            if PurePosixPath(node_file).suffix.lower() not in self._RUST_EXTENSIONS:
                continue
            if node_file == f"{src_root}/lib.rs" or node_file == f"{src_root}/main.rs" or node_file.startswith(f"{src_root}/"):
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_c_symbol_in_file(self, file_id: str, symbol: str | None, node_types: set[NodeType]) -> Optional[str]:
        if not symbol:
            return None
        direct_id = normalize_path(f"{file_id}:{symbol}")
        if direct_id in self.G and self.G.nodes[direct_id].get("type") in node_types:
            return direct_id

        matches = []
        prefix = f"{file_id}:"
        for node_id, attrs in self.G.nodes(data=True):
            if not node_id.startswith(prefix):
                continue
            if attrs.get("name") != symbol:
                continue
            if attrs.get("type") in node_types:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_c_static_method(self, file_id: str, qualifier: str, symbol: str | None) -> Optional[str]:
        if not symbol:
            return None
        direct_id = normalize_path(f"{file_id}:{qualifier}.{symbol}")
        if direct_id in self.G and self.G.nodes[direct_id].get("type") == NodeType.METHOD:
            return direct_id

        file_dir = self._node_parent_dir(file_id)
        matches = []
        suffix = f":{qualifier}.{symbol}"
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") != NodeType.METHOD:
                continue
            if attrs.get("name") != symbol or not node_id.endswith(suffix):
                continue
            node_file = node_id.split(":", 1)[0]
            if PurePosixPath(node_file).suffix.lower() not in self._C_FAMILY_EXTENSIONS:
                continue
            if self._node_parent_dir(node_file) == file_dir:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_c_directory_symbol(
        self,
        file_id: str,
        symbol: str | None,
        node_types: set[NodeType],
    ) -> Optional[str]:
        if not symbol:
            return None
        file_dir = self._node_parent_dir(file_id)
        matches = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") not in node_types:
                continue
            if attrs.get("name") != symbol:
                continue
            node_file = node_id.split(":", 1)[0]
            if PurePosixPath(node_file).suffix.lower() not in self._C_FAMILY_EXTENSIONS:
                continue
            if self._node_parent_dir(node_file) == file_dir:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        # A function declared in a header (prototype) and defined in an
        # implementation file yields two same-named nodes. The call target is
        # the definition, so prefer implementation-file matches over headers.
        impl_matches = [
            node_id for node_id in matches
            if PurePosixPath(node_id.split(":", 1)[0]).suffix.lower()
            not in self._C_HEADER_EXTENSIONS
        ]
        if len(impl_matches) == 1:
            return impl_matches[0]
        return None

    def _node_parent_dir(self, node_id: str) -> str:
        parent = PurePosixPath(node_id).parent.as_posix()
        return "" if parent == "." else parent

    def _find_named_unit_in_file(self, file_id: str, symbol: str | None) -> Optional[str]:
        if not symbol:
            return None
        direct_id = normalize_path(f"{file_id}:{symbol}")
        if direct_id in self.G and self.G.nodes[direct_id].get("type") in {
            NodeType.CLASS,
            NodeType.FUNCTION,
            NodeType.METHOD,
        }:
            return direct_id

        matches = []
        prefix = f"{file_id}:"
        for node_id, attrs in self.G.nodes(data=True):
            if not node_id.startswith(prefix):
                continue
            if attrs.get("name") != symbol:
                continue
            if attrs.get("type") in {NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD}:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_default_ecmascript_export_in_file(self, file_id: str) -> Optional[str]:
        candidates: list[str] = []
        default_exports: list[str] = []
        prefix = f"{file_id}:"
        for node_id, attrs in self.G.nodes(data=True):
            if not node_id.startswith(prefix):
                continue
            if attrs.get("type") not in {NodeType.CLASS, NodeType.FUNCTION}:
                continue
            candidates.append(node_id)
            if attrs.get("export_default") is True or attrs.get("extra", {}).get("export_default") is True:
                default_exports.append(node_id)

        if len(default_exports) == 1:
            return default_exports[0]
        if default_exports:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _find_go_package_symbol(self, package_file_id: str, symbol: str | None) -> Optional[str]:
        if not symbol:
            return None
        package_dir = PurePosixPath(package_file_id).parent.as_posix()
        if package_dir == ".":
            package_dir = ""

        matches = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") not in {NodeType.FUNCTION, NodeType.METHOD}:
                continue
            if attrs.get("name") != symbol:
                continue
            node_file = node_id.split(":", 1)[0]
            node_dir = PurePosixPath(node_file).parent.as_posix()
            if node_dir == ".":
                node_dir = ""
            if node_dir == package_dir:
                matches.append(node_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_go_module_import(self, module: str | None) -> Optional[str]:
        if not module:
            return None
        module_path = self._read_go_module_path()
        if not module_path:
            return None
        if module == module_path:
            package_dir = "."
        elif module.startswith(f"{module_path}/"):
            package_dir = module[len(module_path) + 1:]
        else:
            return None
        return self._go_package_representative_file(package_dir)

    def _read_go_module_path(self) -> Optional[str]:
        if self._go_module_path_loaded:
            return self._go_module_path_cache
        self._go_module_path_loaded = True
        go_mod = Path(self.repo_dir) / "go.mod"
        try:
            for line in go_mod.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue
                if stripped.startswith("module "):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        self._go_module_path_cache = parts[1]
                    break
        except OSError:
            self._go_module_path_cache = None
        return self._go_module_path_cache

    def _go_package_representative_file(self, package_dir: str) -> Optional[str]:
        normalized_dir = normalize_path(package_dir)
        if normalized_dir == ".":
            normalized_dir = ""
        files = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") != NodeType.FILE or not node_id.endswith(".go"):
                continue
            parent = PurePosixPath(node_id).parent.as_posix()
            if parent == ".":
                parent = ""
            if parent == normalized_dir:
                files.append(node_id)
        files.sort()
        if not files:
            return None
        if len(files) == 1:
            return files[0]
        doc_go = posixpath.join(normalized_dir, "doc.go") if normalized_dir else "doc.go"
        if doc_go in files:
            return doc_go
        basename = PurePosixPath(normalized_dir).name if normalized_dir else ""
        if basename:
            basename_go = posixpath.join(normalized_dir, f"{basename}.go")
            if basename_go in files:
                return basename_go
        return files[0]

    def _ensure_lp_import_placeholder(
        self,
        file_id: str,
        dep: lang_parser.LPDependency,
        dependency_index: int,
        import_nodes_by_line: dict[int, str],
        import_nodes_by_module: dict[str, str],
    ) -> str:
        node_id = self._find_lp_import_unit(dep, import_nodes_by_line, import_nodes_by_module)
        if node_id is None:
            suffix = self._lp_dependency_suffix(dep, dependency_index)
            node_id = normalize_path(f"{file_id}:import:{suffix}")
            self._add_node(
                node_id,
                type=NodeType.IMPORT,
                name=dep.dst or dep.symbol or "unresolved import",
                parent_id=file_id,
                language=(dep.extra or {}).get("language"),
                unit_type="import",
                start_line=dep.line,
                end_line=dep.line,
                code="",
                import_module=dep.dst,
                extra=dict(dep.extra or {}),
            )

        self._mark_lp_import_unit(dep, import_nodes_by_line, import_nodes_by_module, False, None)
        self.G.nodes[node_id].update(
            resolved=False,
            confidence=dep.confidence or "unresolved",
            heuristic=True,
            import_module=dep.dst,
        )
        return node_id

    def _mark_lp_import_unit(
        self,
        dep: lang_parser.LPDependency,
        import_nodes_by_line: dict[int, str],
        import_nodes_by_module: dict[str, str],
        resolved: bool,
        target_id: Optional[str],
    ) -> None:
        node_id = self._find_lp_import_unit(dep, import_nodes_by_line, import_nodes_by_module)
        if node_id is None or node_id not in self.G:
            return
        attrs: dict[str, Any] = {
            "resolved": resolved,
            "confidence": "resolved" if resolved else (dep.confidence or "unresolved"),
            "import_module": dep.dst,
        }
        if resolved:
            attrs["resolved_to"] = target_id
        else:
            attrs["heuristic"] = True
        self.G.nodes[node_id].update(attrs)

    def _find_lp_import_unit(
        self,
        dep: lang_parser.LPDependency,
        import_nodes_by_line: dict[int, str],
        import_nodes_by_module: dict[str, str],
    ) -> Optional[str]:
        for module in (dep.dst, dep.symbol, (dep.extra or {}).get("module"), (dep.extra or {}).get("import_path")):
            if module and module in import_nodes_by_module:
                return import_nodes_by_module[module]
        if dep.line is not None and dep.line in import_nodes_by_line:
            return import_nodes_by_line[dep.line]
        return None

    def _lp_dependency_suffix(self, dep: lang_parser.LPDependency, dependency_index: int) -> str:
        if dep.line is not None:
            return str(dep.line)
        label = dep.dst or dep.symbol or dep.relation or "unknown"
        safe = "".join(char if char.isalnum() else "_" for char in label).strip("_") or "unknown"
        return f"dep:{dependency_index + 1}:{safe}"

    def _resolve_lp_reference(self, reference: str | None, file_id: str) -> Optional[str]:
        if reference is None:
            return None
        node_id = normalize_path(reference)
        if node_id in self.G:
            return node_id
        local_id = normalize_path(f"{file_id}:{reference}")
        if local_id in self.G:
            return local_id
        return None

