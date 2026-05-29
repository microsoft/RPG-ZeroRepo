#!/usr/bin/env python3
"""
Build dependency graph + _dep_to_rpg_map, enrich RPG edges, and combine with existing RPG feature graph.

Self-contained script: includes DependencyGraph class with AST-based
dependency analysis (imports, invokes, inherits).

Usage:
    python3 utils/build_dep_graph.py \
        --repo-dir ./repo \
        --rpg-in .cmind/data/repo_rpg.json \
        --rpg-out .cmind/data/rpg.json
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    sys.exit("networkx is required: pip install networkx")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Utility functions (inlined from common/utils) ──────────────────────

def normalize_path(path) -> str:
    s = str(path).strip()
    if ":" in s:
        left, right = s.split(":", 1)
    else:
        left, right = s, None

    norm = PurePosixPath(str(left).strip()).as_posix()
    norm = norm.removeprefix("./").removeprefix("/")
    base = "." if (norm == "" or norm == ".") else norm

    if right is not None:
        segs = [seg.strip() for seg in right.strip().strip(".").split(".") if seg.strip()]
        if segs:
            return f"{base}:{'.'.join(segs)}"
    return base


def is_test_file(nid: str) -> bool:
    file_path = nid.split(":")[0]
    word_list = re.split(r" |_|/", file_path.lower())
    return any(word.startswith("test") for word in word_list)


def get_node_range_robust(node: ast.AST, source: str) -> Tuple[int, int, int, int]:
    lines = source.splitlines()

    # start with decorators
    start_inclusive = node.lineno
    if hasattr(node, "decorator_list") and node.decorator_list:
        first_dec = node.decorator_list[0]
        start_inclusive = getattr(first_dec, "lineno", node.lineno)

    header_end_inclusive = getattr(node, "lineno", start_inclusive)
    body_end_inclusive = getattr(node, "end_lineno", None)
    if not isinstance(body_end_inclusive, int):
        body_end_inclusive = header_end_inclusive
    end_exclusive = body_end_inclusive + 1
    return start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive


def extract_source_by_lines(source: str, start_inclusive: int, end_inclusive: int) -> str:
    if start_inclusive is None or end_inclusive is None:
        return ""
    lines = source.splitlines(keepends=True)
    n = len(lines)
    s = max(1, start_inclusive)
    e = min(n, end_inclusive)
    if s > e:
        return ""
    return "".join(lines[s - 1 : e]).strip()


# ── Node/Edge types ────────────────────────────────────────────────────

class NodeType(str, Enum):
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"

    def __str__(self):
        return self.value


class EdgeType(str, Enum):
    CONTAINS = "contains"
    INHERITS = "inherits"
    INVOKES = "invokes"
    IMPORTS = "imports"

    def __str__(self):
        return self.value


# ── Module path conversion ─────────────────────────────────────────────

def path_to_module(node_id: str) -> str:
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
            return path.with_suffix("").as_posix().replace("/", ".")
    else:
        return path.as_posix().replace("/", ".")


# ── Filter functions ───────────────────────────────────────────────────

def _filter_build(file_id: str) -> bool:
    EXT_BL = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".mp3", ".mp4", ".zip", ".tar", ".gz",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".exe", ".dll", ".so", ".o", ".a", ".log",
    }
    PATH_BL = {
        ".git", "__pycache__", "node_modules",
        ".venv", "venv", ".idea", ".vscode",
        ".pytest_cache", ".mypy_cache", "build", "dist",
        ".cmind", ".venv_dev",
        ".rpgkit",  # legacy runtime dir (pre-rename), skip if still present
    }
    FILE_BL = {
        "Makefile", "CMakeLists.txt", "Dockerfile",
        "LICENSE", "LICENSE.txt", "COPYING",
        "requirements.txt", "environment.yml", "pyproject.toml",
    }
    p = PurePosixPath(file_id)
    if p.suffix.lower() in EXT_BL:
        return False
    if any(part in PATH_BL for part in p.parts):
        return False
    if p.name in FILE_BL:
        return False
    if p.name.startswith("."):
        return False
    if is_test_file(file_id):
        return False
    return True


def _filter_parse(file_id: str) -> bool:
    if not file_id.endswith(".py"):
        return False
    if is_test_file(file_id):
        return False
    EXCLUDE = {"setup.py", "__main__.py", "conftest.py", "requirements.py"}
    if any(file_id.endswith(f"/{f}") for f in EXCLUDE):
        return False
    base = os.path.basename(file_id)
    if base.startswith("test_") or base.endswith("_test.py"):
        return False
    return True


# ── DependencyGraph ────────────────────────────────────────────────────

class DependencyGraph:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
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

    # ── Node/edge helpers ──────────────────────────────────────────

    def _add_node(self, node_id, type, name=None, parent_id=None, **kwargs):
        nid = normalize_path(node_id)
        if not name:
            name = self._get_name(nid, type)
        self.G.add_node(nid, type=type, module=path_to_module(nid), name=name, **kwargs)
        if parent_id is None:
            _, parent_id = self._get_parent(nid)
        if parent_id is not None and parent_id not in self.G:
            if type in [NodeType.DIRECTORY, NodeType.FILE]:
                self._add_node(parent_id, NodeType.DIRECTORY)
            elif type in [NodeType.CLASS, NodeType.FUNCTION]:
                self._add_node(parent_id, NodeType.FILE)
            else:
                self._add_node(parent_id, NodeType.CLASS)
        if parent_id is not None:
            self._add_edge(parent_id, nid, type=EdgeType.CONTAINS)

    def _ensure_node(self, node_id, type):
        nid = normalize_path(node_id)
        if nid not in self.G:
            self._add_node(nid, type=type)

    def _add_edge(self, src, dst, type, **kwargs):
        u, v = normalize_path(src), normalize_path(dst)
        if u not in self.G or v not in self.G:
            return False
        for _key, data in self.G.get_edge_data(u, v, default={}).items():
            if data.get("type") == type:
                return False
        self.G.add_edge(u, v, type=type, **kwargs)
        return True

    def _get_parent(self, nid):
        if nid == ".":
            return True, None
        if ":" in nid:
            path_part, qual = nid.split(":", 1)
            parts = qual.split(".")
            parent_id = path_part if len(parts) <= 1 else f"{path_part}:{'.'.join(parts[:-1])}"
            parent_id = normalize_path(parent_id)
        else:
            parent_id = normalize_path(Path(nid).parent)
        return parent_id in self.G, parent_id

    def _get_name(self, nid, ntype=None):
        if ntype in (NodeType.DIRECTORY, NodeType.FILE):
            return "." if nid == "." else nid.split("/")[-1]
        if ":" in nid:
            qual = nid.split(":", 1)[1]
            parts = [p for p in qual.split(".") if p]
            return parts[-1] if parts else qual
        return nid.split("/")[-1]

    # ── Build (scan filesystem) ────────────────────────────────────

    def build(self, filter_func=_filter_build):
        logger.info("Building dependency graph for: %s", self.repo_dir)
        repo_root = Path(self.repo_dir)
        if not repo_root.exists():
            raise FileNotFoundError(f"Repo not found: {self.repo_dir}")
        self._add_node(".", type=NodeType.DIRECTORY, code=None)

        for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True, followlinks=False):
            dir_path = Path(dirpath)
            dir_rel = normalize_path(dir_path.relative_to(repo_root))
            if not filter_func(dir_rel):
                dirnames[:] = []
                continue
            self._ensure_node(dir_rel, NodeType.DIRECTORY)
            for d in dirnames:
                sub_rel = normalize_path((dir_path / d).relative_to(repo_root))
                if filter_func(sub_rel):
                    self._add_node(sub_rel, type=NodeType.DIRECTORY, parent_id=dir_rel, code="")
            for f in filenames:
                file_rel = normalize_path(str((dir_path / f).relative_to(repo_root)))
                if not filter_func(file_rel):
                    continue
                try:
                    content = (dir_path / f).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                self._add_node(file_rel, type=NodeType.FILE, code=content, parent_id=dir_rel)

        logger.info("Build done: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())

    # ── Parse (AST analysis) ───────────────────────────────────────

    def parse(self, filter_func=_filter_parse):
        logger.info("Parsing AST...")
        for file_id, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE or not filter_func(file_id):
                continue
            content = attrs.get("code") or ""
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue
            self.G.nodes[file_id]["ast"] = tree
            self._parse_file(file_id, tree, content)

        # imports
        for nid in list(self.G_code.nodes()):
            self._init_alias_map(nid)
        alias_links = nx.DiGraph()
        for nid, attrs in list(self.G_code.nodes(data=True)):
            self._parse_imports(nid, attrs["ast"], alias_links)
        logger.info("Imports: %d edges", self.G_imports.number_of_edges())

        # inherits
        for nid, attrs in list(self.G_code.nodes(data=True)):
            if attrs.get("type") == NodeType.CLASS:
                self._parse_inherits(nid, attrs["ast"])
        logger.info("Inherits: %d edges", self.G_inherits.number_of_edges())

        # invokes
        for nid, attrs in list(self.G_code.nodes(data=True)):
            self._parse_invokes(nid, attrs["ast"])
        logger.info("Invokes: %d edges", self.G_invokes.number_of_edges())
        logger.info("Parse done: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())

    # ── File parsing ───────────────────────────────────────────────

    _CF_TYPES = (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.AsyncWith, ast.AsyncFor)

    @staticmethod
    def _cf_bodies(node):
        bodies = []
        if isinstance(node, ast.If):
            bodies.append(node.body)
            if node.orelse: bodies.append(node.orelse)
        elif isinstance(node, ast.Try):
            bodies.append(node.body)
            for h in node.handlers: bodies.append(h.body)
            if node.orelse: bodies.append(node.orelse)
            if node.finalbody: bodies.append(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            bodies.append(node.body)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            bodies.append(node.body)
            if node.orelse: bodies.append(node.orelse)
        return bodies

    def _extract_cf(self, stmts, file_id, source, get_range, parent_id):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"{file_id}:{node.name}"
                if fid not in self.G:
                    s, e = get_range(node)
                    self._add_node(fid, type=NodeType.FUNCTION, code=extract_source_by_lines(source, s, e),
                                   parent_id=parent_id, ast=node, start_line=s, end_line=e)
            elif isinstance(node, ast.ClassDef):
                cid = f"{file_id}:{node.name}"
                if cid not in self.G:
                    s, e = get_range(node)
                    self._add_node(cid, type=NodeType.CLASS, code=extract_source_by_lines(source, s, e),
                                   parent_id=parent_id, ast=node, start_line=s, end_line=e)
                    for bn in node.body:
                        if isinstance(bn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            mid = f"{file_id}:{node.name}.{bn.name}"
                            if mid not in self.G:
                                s2, e2 = get_range(bn)
                                self._add_node(mid, type=NodeType.METHOD, code=extract_source_by_lines(source, s2, e2),
                                               parent_id=cid, ast=bn, start_line=s2, end_line=e2)
            if isinstance(node, self._CF_TYPES):
                for block in self._cf_bodies(node):
                    self._extract_cf(block, file_id, source, get_range, parent_id)

    def _parse_file(self, file_id, tree, source):
        def get_range(node):
            s, _, be, _ = get_node_range_robust(node, source)
            return s, be

        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"{file_id}:{node.name}"
                s, e = get_range(node)
                self._add_node(fid, type=NodeType.FUNCTION, code=extract_source_by_lines(source, s, e),
                               parent_id=file_id, ast=node, start_line=s, end_line=e)
            elif isinstance(node, ast.ClassDef):
                cid = f"{file_id}:{node.name}"
                s, e = get_range(node)
                self._add_node(cid, type=NodeType.CLASS, code=extract_source_by_lines(source, s, e),
                               parent_id=file_id, ast=node, start_line=s, end_line=e)
                for bn in node.body:
                    if isinstance(bn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mid = f"{file_id}:{node.name}.{bn.name}"
                        s2, e2 = get_range(bn)
                        self._add_node(mid, type=NodeType.METHOD, code=extract_source_by_lines(source, s2, e2),
                                       parent_id=cid, ast=bn, start_line=s2, end_line=e2)

        for node in getattr(tree, "body", []):
            if isinstance(node, self._CF_TYPES):
                for block in self._cf_bodies(node):
                    self._extract_cf(block, file_id, source, get_range, file_id)

    # ── Alias / import parsing ─────────────────────────────────────

    def _init_alias_map(self, nid):
        self.G.nodes[nid]["alias_to_entity"] = {}
        am = self.G.nodes[nid]["alias_to_entity"]
        for _, child_id, ed in self.G_tree.out_edges(nid, data=True):
            if ed.get("type") == EdgeType.CONTAINS:
                am[self.G.nodes[child_id].get("name")] = child_id
        if nid.endswith("__init__.py"):
            _, parent_id = self._get_parent(nid)
            if parent_id:
                for _, child_id, ed in self.G_tree.out_edges(parent_id, data=True):
                    if ed.get("type") == EdgeType.CONTAINS and child_id != nid:
                        cn = self.G.nodes[child_id].get("name")
                        ct = self.G.nodes[child_id].get("type")
                        if ct == NodeType.DIRECTORY:
                            init_id = normalize_path(f"{child_id}/__init__.py")
                            if init_id in self.G:
                                am[cn] = init_id
                        else:
                            am[cn] = child_id
        return am

    def _parse_imports(self, nid, tree, alias_links):
        if nid not in self.G or self.G.nodes[nid].get("type") not in [
            NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
            return
        cur_mod = self.G.nodes[nid].get("module", "")
        for node in ast.iter_child_nodes(tree):
            am = self.G.nodes[nid].get("alias_to_entity")
            if am is None:
                am = self._init_alias_map(nid)
            if isinstance(node, ast.Import):
                for al in node.names:
                    mid = self._find_module_file(al.name)
                    alias = al.asname or al.name
                    if mid:
                        self._add_edge(nid, mid, type=EdgeType.IMPORTS, alias=alias)
                        am[alias] = mid
                        self._propagate_aliases(nid, alias, alias_links)
            elif isinstance(node, ast.ImportFrom):
                abs_mod = self._resolve_rel_module(cur_mod, node.module, node.level)
                if not abs_mod:
                    continue
                mid = self._find_module_file(abs_mod)
                if mid is None:
                    continue
                mam = self.G.nodes[mid].get("alias_to_entity")
                if mam is None:
                    mam = self._init_alias_map(mid)
                if any(al.name == "*" for al in node.names):
                    for alias, entity in mam.items():
                        self._add_edge(nid, entity, type=EdgeType.IMPORTS, alias=alias)
                        am[alias] = entity
                        self._propagate_aliases(nid, alias, alias_links)
                    alias_links.add_edge(f"{mid}:*", f"{nid}:*")
                    continue
                for al in node.names:
                    target = al.name
                    alias = al.asname or al.name
                    if target in mam:
                        self._add_edge(nid, mam[target], type=EdgeType.IMPORTS, alias=alias)
                        am[alias] = mam[target]
                        self._propagate_aliases(nid, alias, alias_links)
                    else:
                        alias_links.add_edge(f"{mid}:{target}", f"{nid}:{alias}")

    def _propagate_aliases(self, nid, alias, alias_links):
        entity = self.G.nodes[nid].get("alias_to_entity", {}).get(alias)
        if not entity:
            return
        for _, dst in alias_links.out_edges(f"{nid}:{alias}"):
            dn, da = dst.split(":", 1)
            am = self.G.nodes[dn].get("alias_to_entity")
            if am is None:
                am = self._init_alias_map(dn)
            if am.get(da) != entity:
                am[da] = entity
                self._add_edge(dn, entity, type=EdgeType.IMPORTS, alias=da)
                self._propagate_aliases(dn, da, alias_links)
        for _, dst in alias_links.out_edges(f"{nid}:*"):
            dn, _ = dst.split(":", 1)
            da = alias
            am = self.G.nodes[dn].get("alias_to_entity")
            if am is None:
                am = self._init_alias_map(dn)
            if am.get(da) != entity:
                am[da] = entity
                self._add_edge(dn, entity, type=EdgeType.IMPORTS, alias=da)
                self._propagate_aliases(dn, da, alias_links)

    # ── Inheritance ────────────────────────────────────────────────

    def _parse_inherits(self, nid, tree):
        if self.G.nodes[nid].get("type") != NodeType.CLASS or not isinstance(tree, ast.ClassDef):
            return
        for base in tree.bases:
            if isinstance(base, (ast.Name, ast.Attribute)):
                bn = ast.unparse(base)
                entity = self._find_entity(nid, bn)
                if entity:
                    self._add_edge(nid, entity, type=EdgeType.INHERITS)

    # ── Invocation ─────────────────────────────────────────────────

    def _parse_invokes(self, nid, tree):
        if self.G.nodes[nid].get("type") not in [
            NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
            return
        ntype = self.G.nodes[nid].get("type")
        to_walk = []
        if ntype == NodeType.FILE:
            for c in ast.iter_child_nodes(tree):
                if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    to_walk.append(c)
        elif ntype == NodeType.CLASS:
            for c in ast.iter_child_nodes(tree):
                if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    to_walk.append(c)
        else:
            to_walk.append(tree)
        for subtree in to_walk:
            for child in ast.walk(subtree):
                if isinstance(child, (ast.Call, ast.Await)):
                    if isinstance(child, ast.Await):
                        child = child.value
                    if not isinstance(child, ast.Call):
                        continue
                    fn = ast.unparse(child.func)
                    entity = self._find_entity(nid, fn)
                    if entity and entity != nid:
                        self._add_edge(nid, entity, type=EdgeType.INVOKES)
                    elif not entity:
                        entity = self._find_entity_fuzzy(nid, fn)
                        if entity and entity != nid:
                            self._add_edge(nid, entity, type=EdgeType.INVOKES)

    # ── Resolution helpers ─────────────────────────────────────────

    def _resolve_rel_module(self, cur_mod, module, level):
        if level == 0:
            return module
        parts = cur_mod.split(".") if cur_mod else []
        parts = parts[:len(parts) - level] if level <= len(parts) else []
        if module:
            return ".".join(parts + [module]) if parts else module
        return ".".join(parts)

    def _find_module_file(self, module_name):
        mp = normalize_path("./" + module_name.replace(".", "/"))
        fp = normalize_path(mp + ".py")
        ip = normalize_path(f"{mp}/__init__.py")
        if fp in self.G:
            return fp
        if ip in self.G:
            return ip
        return None

    def _find_entity(self, module_id, qual_name):
        if module_id not in self.G:
            return None
        parts = qual_name.split(".")
        cur = module_id
        for part in parts:
            am = self.G.nodes[cur].get("alias_to_entity", {})
            if am.get(part):
                cur = am[part]
            elif self.G.nodes[module_id].get("type") in [NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                _, pid = self._get_parent(module_id)
                if pid:
                    return self._find_entity(pid, qual_name)
                return None
            else:
                return None
        return cur

    def _find_entity_fuzzy(self, nid, qual_name):
        parts = qual_name.split(".")
        if not parts:
            return None
        cc = self._get_containing_class(nid)
        if parts[0] == "self" and len(parts) >= 2 and cc:
            method = parts[-1]
            entity = self._find_method_in_hierarchy(cc, method)
            if entity:
                return entity
            return self._find_method_global(method)
        if parts[0] == "super()" and len(parts) == 2 and cc:
            return self._find_method_in_parents(cc, parts[1])
        if len(parts) >= 2 and cc:
            method = parts[-1]
            entity = self._find_method_in_hierarchy(cc, method)
            if entity:
                return entity
            return self._find_method_global(method)
        return None

    def _get_containing_class(self, nid):
        nt = self.G.nodes.get(nid, {}).get("type")
        if nt == NodeType.CLASS:
            return nid
        if nt == NodeType.METHOD:
            _, pid = self._get_parent(nid)
            if pid and self.G.nodes.get(pid, {}).get("type") == NodeType.CLASS:
                return pid
        return None

    def _find_method_in_hierarchy(self, class_id, method_name):
        if class_id not in self.G:
            return None
        am = self.G.nodes[class_id].get("alias_to_entity", {})
        if method_name in am:
            return am[method_name]
        for _, dst, ed in self.G.out_edges(class_id, data=True):
            if ed.get("type") == EdgeType.INHERITS:
                r = self._find_method_in_hierarchy(dst, method_name)
                if r:
                    return r
        return None

    def _find_method_in_parents(self, class_id, method_name):
        if class_id not in self.G:
            return None
        for _, dst, ed in self.G.out_edges(class_id, data=True):
            if ed.get("type") == EdgeType.INHERITS:
                am = self.G.nodes[dst].get("alias_to_entity", {})
                if method_name in am:
                    return am[method_name]
                r = self._find_method_in_hierarchy(dst, method_name)
                if r:
                    return r
        return None

    COMMON_METHODS = {
        "__init__", "__str__", "__repr__", "__eq__", "__hash__",
        "get", "set", "add", "remove", "update", "delete", "save",
        "load", "run", "start", "stop", "close", "open", "read", "write",
        "setup", "teardown", "reset", "clear", "copy", "keys", "values",
        "items", "append", "extend", "pop", "push", "insert",
    }

    def _find_method_global(self, method_name):
        if method_name in self.COMMON_METHODS:
            return None
        matches = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == NodeType.METHOD and attrs.get("name") == method_name:
                matches.append(nid)
        return matches[0] if len(matches) == 1 else None

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {"repo_dir": self.repo_dir, "nodes": {}, "edges": []}
        for nid, attrs in self.G.nodes(data=True):
            data["nodes"][nid] = {k: v for k, v in attrs.items() if k not in ("ast", "alias_to_entity", "code")}
        for u, v, attrs in self.G.edges(data=True):
            data["edges"].append({"src": u, "dst": v, "attrs": dict(attrs)})
        return data


# ── RPG mapping and enrichment ─────────────────────────────────────────


def collect_rpg_nodes(node: dict, out: list):
    """Recursively collect all RPG nodes from the tree."""
    out.append(node)
    for c in node.get("children", []):
        collect_rpg_nodes(c, out)


def rpg_path_to_dep_id(rpg_path: str) -> Optional[str]:
    """Convert RPG meta.path to dep graph node ID.

    Examples:
        'src/foo/bar.py::class MyClass'  -> 'src/foo/bar.py:MyClass'
        'src/foo/bar.py::function my_fn' -> 'src/foo/bar.py:my_fn'
        'src/foo/bar'                    -> 'src/foo/bar'  (directory)
        '.'                              -> '.'
    """
    if "::" not in rpg_path:
        # directory or root
        return rpg_path

    file_part, qual_part = rpg_path.split("::", 1)
    # qual_part is like "class Vec2" or "function compute_position_step"
    parts = qual_part.strip().split(None, 1)
    if len(parts) == 2:
        # kind, name = parts  -> dep node id is file:name
        return f"{file_part}:{parts[1]}"
    elif len(parts) == 1:
        return f"{file_part}:{parts[0]}"
    return file_part


def build_dep_to_rpg_map(
    rpg_nodes: list,
    dep_node_ids: set,
) -> Dict[str, List[str]]:
    """Build mapping from dep graph node IDs to RPG feature node IDs.

    Uses the meta.path field in RPG nodes to find the corresponding
    dep graph node. Multiple RPG features can map to the same dep node.
    """
    dep2rpg: Dict[str, List[str]] = defaultdict(list)

    for rn in rpg_nodes:
        meta = rn.get("meta")
        if not meta:
            continue
        rpg_path = meta.get("path")
        if not rpg_path:
            continue
        rpg_id = rn.get("id")
        if not rpg_id:
            continue

        dep_id = rpg_path_to_dep_id(rpg_path)
        if dep_id is None:
            continue

        if dep_id in dep_node_ids:
            dep2rpg[dep_id].append(rpg_id)
        else:
            logger.debug("No dep node for RPG path %s -> %s", rpg_path, dep_id)

    return dict(dep2rpg)


def enrich_rpg_edges(
    rpg_edges: List[dict],
    dep_dict: dict,
    dep_to_rpg: Dict[str, List[str]],
) -> List[dict]:
    """Project dep graph invokes/inherits edges onto RPG feature edges.

    For each dep edge of type invokes or inherits, find the corresponding
    RPG feature nodes via _dep_to_rpg_map and add a new RPG edge if one
    doesn't already exist.
    """
    # Index existing RPG edges to avoid duplicates
    existing = set()
    for e in rpg_edges:
        existing.add((e.get("src"), e.get("dst"), e.get("relation")))

    new_edges = []
    for dep_edge in dep_dict.get("edges", []):
        etype = dep_edge.get("attrs", {}).get("type", "")
        if etype not in ("invokes", "inherits"):
            continue

        src_rpg_ids = dep_to_rpg.get(dep_edge["src"], [])
        dst_rpg_ids = dep_to_rpg.get(dep_edge["dst"], [])

        for src_rpg in src_rpg_ids:
            for dst_rpg in dst_rpg_ids:
                if src_rpg == dst_rpg:
                    continue
                key = (src_rpg, dst_rpg, etype)
                if key in existing:
                    continue
                existing.add(key)
                new_edges.append({
                    "src": src_rpg,
                    "dst": dst_rpg,
                    "relation": etype,
                    "inferred_from": "dep_graph",
                })

    return new_edges


def main():
    parser = argparse.ArgumentParser(
        description="Build dep graph with _dep_to_rpg_map and combine with RPG feature graph"
    )
    parser.add_argument("--repo-dir", required=True, help="Path to source repo")
    parser.add_argument("--rpg-in", required=True, help="Path to existing repo_rpg.json")
    parser.add_argument("--rpg-out", required=True, help="Output path for rpg.json")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    rpg_in = os.path.abspath(args.rpg_in)
    rpg_out = os.path.abspath(args.rpg_out)

    # 1. Load existing feature graph
    logger.info("Loading feature graph from %s", rpg_in)
    with open(rpg_in, "r", encoding="utf-8") as f:
        rpg_data = json.load(f)

    # 2. Build dependency graph
    dg = DependencyGraph(repo_dir)
    dg.build()
    dg.parse()

    # 3. Collect RPG nodes and build mapping
    rpg_nodes = []
    collect_rpg_nodes(rpg_data["root"], rpg_nodes)
    dep_node_ids = set(dg.G.nodes())

    dep_to_rpg = build_dep_to_rpg_map(rpg_nodes, dep_node_ids)

    # Stats
    mapped_dep_nodes = len(dep_to_rpg)
    mapped_rpg_nodes = len(set(rid for rids in dep_to_rpg.values() for rid in rids))
    logger.info(
        "Mapping: %d dep nodes -> %d RPG feature nodes (out of %d dep / %d RPG total)",
        mapped_dep_nodes, mapped_rpg_nodes,
        len(dep_node_ids), len(rpg_nodes),
    )

    # 4. Serialize dep_graph with rpg_nodes annotations
    dep_dict = dg.to_dict()
    # Add rpg_nodes to each dep node
    for nid in dep_dict["nodes"]:
        dep_dict["nodes"][nid]["rpg_nodes"] = dep_to_rpg.get(nid, [])
    # Add src_rpg_nodes / dst_rpg_nodes to each dep edge
    for edge in dep_dict["edges"]:
        edge["src_rpg_nodes"] = dep_to_rpg.get(edge["src"], [])
        edge["dst_rpg_nodes"] = dep_to_rpg.get(edge["dst"], [])

    # 5. Enrich RPG edges from dep graph invokes/inherits
    rpg_edges = rpg_data.get("edges", [])
    new_edges = enrich_rpg_edges(rpg_edges, dep_dict, dep_to_rpg)
    logger.info("Enriched RPG edges: +%d (invokes: %d, inherits: %d)",
                len(new_edges),
                sum(1 for e in new_edges if e["relation"] == "invokes"),
                sum(1 for e in new_edges if e["relation"] == "inherits"))
    rpg_edges = rpg_edges + new_edges

    # 6. Combine into output
    output = {
        "repo_name": rpg_data.get("repo_name", ""),
        "repo_info": rpg_data.get("repo_info", ""),
        "excluded_files": rpg_data.get("excluded_files", []),
        "repo_node_id": rpg_data.get("repo_node_id"),
        "root": rpg_data.get("root"),
        "edges": rpg_edges,
        "_dep_to_rpg_map": dep_to_rpg,
        "dep_graph": dep_dict,
    }

    # 7. Write output
    os.makedirs(os.path.dirname(rpg_out) or ".", exist_ok=True)
    with open(rpg_out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done! Wrote %s (dep: %d nodes, %d edges; features: %d edges; map: %d entries)",
        rpg_out,
        len(dep_dict["nodes"]),
        len(dep_dict["edges"]),
        len(output["edges"]),
        len(dep_to_rpg),
    )


if __name__ == "__main__":
    main()
