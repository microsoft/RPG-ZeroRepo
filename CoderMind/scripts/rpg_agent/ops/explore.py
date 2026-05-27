#!/usr/bin/env python3
"""Explore RPG graph structure — traverse code and feature views.

Provides tree and JSON-based graph exploration starting from code entity IDs
or feature paths, following code dependencies and functional composition edges.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/ops/explore_rpg.py
"""

import json
import logging
from collections import defaultdict, OrderedDict
from typing import (
    Any, Dict, Iterable, List,
    Literal, Optional, Tuple,
)

from rapidfuzz import fuzz, process

from common.utils import is_test_file, normalize_text
from rpg_agent.env.searcher import RepoEntitySearcher
from rpg_agent.ops.search_by_feature import fuzzy_match_search_feature
from rpg_agent.ops.search_by_meta import fuzzy_retrieve
from rpg import EdgeType, Node, NodeType, RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Input validation
# ============================================================================

def _fuzzy_match_feature_path(
    rpg: RPG,
    query_path: str,
    top_k: int = 5,
    score_cutoff: float = 50.0,
) -> List[Tuple[str, Node, float]]:
    """Match *query_path* against all feature paths using fuzz.ratio.

    Source: RPG-ZeroRepo explore_rpg.py ``_fuzzy_match_feature_path``
    """
    paths_map: Dict[str, Node] = {}
    for node_type in NodeType:
        try:
            for node in rpg.get_nodes_by_type(type_name=node_type):
                fp = node.feature_path()
                if fp and fp not in paths_map:
                    paths_map[fp] = node
        except Exception:
            continue

    if not paths_map:
        return []

    matches = process.extract(
        query_path,
        list(paths_map.keys()),
        scorer=fuzz.ratio,
        limit=top_k,
        score_cutoff=score_cutoff,
    )
    return [(path, paths_map[path], score) for path, score, _ in matches]


def _validate_graph_explorer_inputs(
    start_code_entities: List[str],
    start_feature_entities: Optional[List[str]] = None,
    direction: str = "downstream",
    traversal_depth: int = 1,
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[RepoEntitySearcher] = None,
) -> Tuple[List[str], str, List[str], str]:
    """Validate and resolve start entities; return valid code/feature entities and hint text.

    Returns:
        (valid_code_entities, code_hints, valid_feature_entities, feature_hints)

    Source: RPG-ZeroRepo explore_rpg.py ``_validate_graph_explorer_inputs``
    """
    if start_feature_entities is None:
        start_feature_entities = []

    assert direction in ("downstream", "upstream", "both"), (
        f"Invalid direction '{direction}'. Expected 'downstream', 'upstream', or 'both'."
    )
    assert traversal_depth == -1 or traversal_depth >= 0, (
        f"Invalid traversal_depth {traversal_depth}. Must be -1 or >= 0."
    )

    valid_edge_types = [e.value for e in EdgeType]
    valid_node_types = [n.value for n in NodeType]

    if isinstance(node_type_filter, list):
        invalid = [nt for nt in node_type_filter if nt not in valid_node_types]
        assert not invalid, f"Invalid node types {invalid}. Expected: {valid_node_types}"
    if isinstance(edge_type_filter, list):
        invalid = [et for et in edge_type_filter if et not in valid_edge_types]
        assert not invalid, f"Invalid edge types {invalid}. Expected: {valid_edge_types}"

    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}

    # --- Validate code entities ---
    code_hints = ""
    valid_code_entities: List[str] = []

    for root in start_code_entities:
        if root != "/":
            root = root.strip("/")
        if root.endswith(".__init__"):
            root = root[: -len(".__init__")]

        if dep_graph is None or root not in dep_graph:
            # Path-level fuzzy matching
            dep_path_matches = []
            if dep_graph is not None:
                dep_path_matches = process.extract(
                    root, list(dep_graph.nodes()),
                    scorer=fuzz.ratio, limit=5, score_cutoff=50,
                )

            if dep_path_matches:
                best_match, best_score, _ = dep_path_matches[0]
                if best_score >= 80:
                    code_hints += f"The entity path `{root}` does not exist. Did you mean `{best_match}`?\n"
                else:
                    code_hints += f"The entity path `{root}` does not exist. Similar entities:\n"
                for match_id, score, _ in dep_path_matches:
                    rpg_ids = dep2rpg.get(match_id, [])
                    rpg_ids = rpg_ids if isinstance(rpg_ids, list) else [rpg_ids]
                    feature_paths = []
                    for rpg_id in rpg_ids:
                        rpg_node = rpg.get_node_by_id(rpg_id) if rpg else None
                        if rpg_node:
                            feature_paths.append(rpg_node.feature_path())
                    fps = ", ".join(feature_paths) if feature_paths else ""
                    hint = f"  - `{match_id}`"
                    if fps:
                        hint += f" (feature paths: {fps})"
                    hint += f" [similarity: {score:.0f}%]\n"
                    code_hints += hint
                code_hints += "Source: Retrieved using path-level fuzzy match.\n\n"
            else:
                # Token-level fuzzy matching
                module_nids = fuzzy_retrieve(keyword=root, rpg=rpg) if rpg else []
                module_datas = entity_searcher.get_node_data(module_nids, return_code_content=False) if entity_searcher else []
                if module_datas:
                    code_hints += f"The entity name `{root}` is invalid. Candidate entities:\n"
                    for module in module_datas[:5]:
                        ntype = module["type"]
                        nid = module["node_id"]
                        rpg_ids = dep2rpg.get(nid, [])
                        rpg_ids = rpg_ids if isinstance(rpg_ids, list) else [rpg_ids]
                        fps = []
                        for rid in rpg_ids:
                            rn = rpg.get_node_by_id(rid) if rpg else None
                            if rn:
                                fps.append(rn.feature_path())
                        code_hints += f"{ntype}: feature paths: {', '.join(fps)}\nentity path: {nid}\n"
                    code_hints += "Source: Retrieved entity using keyword search (fuzzy match).\n\n"
                else:
                    code_hints += f"The entity name `{root}` is invalid. No candidate entities found.\n"
        else:
            valid_code_entities.append(root)

    # --- Validate feature entities ---
    feature_hints = ""
    valid_feature_entities: List[str] = []

    for f_path in start_feature_entities:
        if f_path.startswith("/"):
            f_path = f_path[1:]
        f_path = f_path.strip()

        f_node = rpg.get_node_by_feature_path(feature_path=f_path) if rpg else None

        if f_node is None:
            path_matches = _fuzzy_match_feature_path(rpg, f_path) if rpg else []
            if path_matches:
                best_path, best_node, best_score = path_matches[0]
                if best_score >= 80:
                    feature_hints += f"The feature path `{f_path}` does not exist. Did you mean `{best_path}`?\n"
                else:
                    feature_hints += f"The feature path `{f_path}` does not exist. Similar feature paths:\n"
                for m_path, m_node, m_score in path_matches[:5]:
                    m_type = m_node.meta.type_name if m_node.meta else "unknown"
                    m_entity = m_node.meta.path if m_node.meta else "unknown"
                    feature_hints += f"  - {m_type}: feature path: {m_path}\n    entity path: {m_entity} [similarity: {m_score:.0f}%]\n"
                feature_hints += "Source: Retrieved using path-level fuzzy match.\n\n"
            else:
                fuzzy_results = fuzzy_match_search_feature(rpg=rpg, keyword=f_path, valid_nodes=[]) if rpg else []
                matched_nodes = [r[0] for r in fuzzy_results]
                if matched_nodes:
                    feature_hints += f"The feature path `{f_path}` does not exist. Candidates:\n"
                    for mn in matched_nodes[:5]:
                        feature_hints += f"  - {mn.meta.type_name if mn.meta else 'unknown'}: feature path: {mn.feature_path()}\n    entity path: {mn.meta.path if mn.meta else 'unknown'}\n"
                    feature_hints += "Source: Retrieved entity using keyword search (fuzzy match).\n\n"
                else:
                    feature_hints += f"The feature path `{f_path}` does not exist. No similar features found.\n"
        else:
            valid_feature_entities.append(f_path)

    return valid_code_entities, code_hints, valid_feature_entities, feature_hints


# ============================================================================
# Feature labeler (for code-view legend)
# ============================================================================

class FeatureLabeler:
    """Assign compact labels (F1, F2, ...) to sets of feature paths.

    Source: RPG-ZeroRepo explore_rpg.py ``FeatureLabeler``
    """

    def __init__(self, rpg: RPG, strip_root: Optional[str] = None):
        self.rpg = rpg
        self.dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}
        self.strip_root = strip_root
        self._setkey_to_label: Dict[Tuple[str, ...], str] = {}
        self._label_to_paths: Dict[str, Tuple[str, ...]] = {}
        self._used_labels: List[str] = []

    @staticmethod
    def _split(p: str) -> List[str]:
        return p.split("/") if p else []

    @staticmethod
    def _join(parts: Iterable[str]) -> str:
        return "/".join(parts)

    def _normalize_path(self, p: str) -> str:
        toks = self._split(p)
        if self.strip_root and toks and toks[0] == self.strip_root:
            toks = toks[1:]
        return self._join(toks)

    def _paths_for_dep(self, dep_id: str) -> Tuple[str, ...]:
        fids = self.dep2rpg.get(dep_id, []) or []
        paths: List[str] = []
        for fid in fids:
            fnode = self.rpg.get_node_by_id(fid)
            if fnode:
                paths.append(self._normalize_path(fnode.feature_path()))
        return tuple(sorted(set(paths)))

    def label_for_dep(self, dep_id: str) -> Optional[str]:
        paths = self._paths_for_dep(dep_id)
        if not paths:
            return None
        key = paths
        if key not in self._setkey_to_label:
            label = f"F{len(self._setkey_to_label) + 1}"
            self._setkey_to_label[key] = label
            self._label_to_paths[label] = key
        else:
            label = self._setkey_to_label[key]
        if label not in self._used_labels:
            self._used_labels.append(label)
        return label

    def used_label_to_paths(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for lb in self._used_labels:
            out[lb] = list(self._label_to_paths.get(lb, ()))
        return out


# ============================================================================
# Trie for feature-path legend rendering
# ============================================================================

class _TrieNode:
    def __init__(self, name: str):
        self.name = name
        self.children: Dict[str, "_TrieNode"] = {}
        self.labels: List[str] = []


def _insert_path(root: _TrieNode, path: str, label: str) -> None:
    cur = root
    for seg in path.split("/"):
        if seg not in cur.children:
            cur.children[seg] = _TrieNode(seg)
        cur = cur.children[seg]
    if label not in cur.labels:
        cur.labels.append(label)


def _render_trie(root: _TrieNode) -> List[str]:
    lines: List[str] = []

    def dfs(node: _TrieNode, prefix: str, _is_last: bool):
        children = sorted(node.children.values(), key=lambda n: n.name)
        for i, ch in enumerate(children):
            last = (i == len(children) - 1)
            conn = "\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 "
            tag = f" [{', '.join(sorted(ch.labels))}]" if ch.labels else ""
            lines.append(f"{prefix}{conn}{ch.name}{tag}")
            new_prefix = prefix + ("    " if last else "\u2502   ")
            dfs(ch, new_prefix, last)

    dfs(root, "", True)
    return lines


def render_feature_paths_tree(label_to_paths: Dict[str, List[str]]) -> List[str]:
    trie_root = _TrieNode("")
    for label, paths in sorted(label_to_paths.items(), key=lambda kv: kv[0]):
        for p in sorted(paths):
            _insert_path(trie_root, p, label)
    return _render_trie(trie_root)


# ============================================================================
# Tree-structure traversal (code view + feature view)
# ============================================================================

def traverse_tree_structure(
    rpg: RPG,
    root: str,
    direction: str = "downstream",
    hops: int = 2,
    visual_type: Literal["code", "feature"] = "code",
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
) -> str:
    """Traverse the RPG graph as an indented tree string.

    Source: RPG-ZeroRepo explore_rpg.py ``traverse_tree_structure``
    """
    if hops == -1:
        hops = 20

    G = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}
    rtn_str: List[str] = []

    # --- Helpers ---
    def _as_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    def _feature_by_id_or_path(arg):
        n = rpg.get_node_by_id(arg)
        if n:
            return n
        return rpg.get_node_by_feature_path(arg) if isinstance(arg, str) else None

    def _get_features_from_dep(dep_id: str) -> List[str]:
        return _as_list(dep2rpg.get(dep_id, []))

    def _get_deps_from_feature_id(fid: str) -> List[str]:
        result = []
        for dep_id, rpg_ids in dep2rpg.items():
            if fid in _as_list(rpg_ids):
                result.append(dep_id)
        return result

    def _ntype_invalid(nid: str) -> bool:
        if node_type_filter is None or G is None:
            return False
        return G.nodes[nid].get("type") not in node_type_filter

    def _etype_invalid(etype: str) -> bool:
        if edge_type_filter is None:
            return False
        return etype not in edge_type_filter

    def _is_test_file_safe(nid: str) -> bool:
        try:
            return bool(is_test_file(nid))
        except Exception:
            return False

    def _infer_strip_root(dep_id: str) -> Optional[str]:
        s = dep_id
        if s.startswith("src/"):
            s = s[4:]
        parts = s.split("/")
        return parts[0] if parts else None

    strip_root = _infer_strip_root(root)
    labeler = FeatureLabeler(rpg=rpg, strip_root=strip_root)

    # ========== Code view ==========
    def traverse_code(node_id, prefix, is_last, level, edge_type, edirection):
        if level > hops:
            return
        label = labeler.label_for_dep(node_id)
        ftag = f"[{label}]" if label else ""
        if node_id == root and level == 0:
            rtn_str.append(f"{node_id}  {ftag}".rstrip())
            new_prefix = ""
            edirection = direction
        else:
            conn = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            full_conn = f"{conn}{edge_type} \u2500\u2500 "
            rtn_str.append(f"{prefix}{full_conn}{node_id}  {ftag}".rstrip())
            new_prefix = prefix + (" " if is_last else "\u2502") + " " * (len(full_conn) - 1)

        neigh_ids, etypes, edirs = [], [], []
        seen_local: set = set()

        if edirection in ("downstream", None) or (node_id == root and direction == "both"):
            for neigh in G.successors(node_id):
                if _ntype_invalid(neigh):
                    continue
                edges = G[node_id][neigh]
                for key in edges:
                    etype = edges[key].get("type")
                    if _etype_invalid(etype) or _is_test_file_safe(neigh):
                        continue
                    k = (etype, node_id, neigh)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(neigh)
                    etypes.append(etype)
                    edirs.append("downstream")

        if edirection in ("upstream", None) or (node_id == root and direction == "both"):
            for neigh in G.predecessors(node_id):
                if _ntype_invalid(neigh):
                    continue
                edges = G[neigh][node_id]
                for key in edges:
                    etype = edges[key].get("type")
                    if _etype_invalid(etype) or _is_test_file_safe(neigh):
                        continue
                    k = (etype + "-by", neigh, node_id)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(neigh)
                    etypes.append(etype + "-by")
                    edirs.append("upstream")

        # Functional containment edges (composes)
        mapped_fids = _get_features_from_dep(node_id)
        if edirection in ("downstream", None) or (node_id == root and direction == "both"):
            for fid in mapped_fids:
                fnode = rpg.get_node_by_id(fid)
                if not fnode:
                    continue
                for ch in fnode.children():
                    for dep_child in _get_deps_from_feature_id(ch.id):
                        if dep_child not in G or _ntype_invalid(dep_child):
                            continue
                        k = ("composes", node_id, dep_child)
                        if k in seen_local:
                            continue
                        seen_local.add(k)
                        neigh_ids.append(dep_child)
                        etypes.append("composes")
                        edirs.append("downstream")

        if edirection in ("upstream", None) or (node_id == root and direction == "both"):
            for fid in mapped_fids:
                fnode = rpg.get_node_by_id(fid)
                if not fnode:
                    continue
                p = fnode.parent()
                if not p:
                    continue
                for dep_parent in _get_deps_from_feature_id(p.id):
                    if dep_parent not in G or _ntype_invalid(dep_parent):
                        continue
                    k = ("composed-by", dep_parent, node_id)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(dep_parent)
                    etypes.append("composed-by")
                    edirs.append("upstream")

        for i, (nid, et, edir) in enumerate(zip(neigh_ids, etypes, edirs)):
            traverse_code(nid, new_prefix, i == len(neigh_ids) - 1, level + 1, et, edir)

    # ========== Feature view ==========
    def _tail_name(path: str, fallback: str = "") -> str:
        if not path:
            return fallback
        parts = path.split("/")
        return parts[-1] if parts else (fallback or path)

    def _compact_dep_files(dep_ids: List[str]) -> str:
        return ", ".join(dep_ids) if dep_ids else "\u2014"

    def _sorted_types(types: Iterable[str]) -> List[str]:
        return sorted(set(types), key=lambda s: (s not in ("composes", "composed-by"), s))

    def _gather_feature_relations(feature_node):
        rel_map: Dict[str, set] = defaultdict(set)
        child_map: Dict[str, Any] = {}
        parent_map: Dict[str, Any] = {}

        dep_ids = _get_deps_from_feature_id(feature_node.id)
        for dep_id in dep_ids:
            if dep_id not in G:
                continue
            if direction in ("downstream", "both"):
                for neigh in G.successors(dep_id):
                    edges = G[dep_id][neigh]
                    for key in edges:
                        etype = edges[key].get("type")
                        if _etype_invalid(etype):
                            continue
                        for tfid in _get_features_from_dep(neigh):
                            if rpg.get_node_by_id(tfid):
                                rel_map[tfid].add(etype)
            if direction in ("upstream", "both"):
                for neigh in G.predecessors(dep_id):
                    edges = G[neigh][dep_id]
                    for key in edges:
                        etype = edges[key].get("type")
                        if _etype_invalid(etype):
                            continue
                        for tfid in _get_features_from_dep(neigh):
                            if rpg.get_node_by_id(tfid):
                                rel_map[tfid].add(etype + "-by")

        if direction in ("downstream", "both"):
            for ch in feature_node.children():
                rel_map[ch.id].add("composes")
                child_map[ch.id] = ch
        if direction in ("upstream", "both"):
            p = feature_node.parent()
            if p:
                rel_map[p.id].add("composed-by")
                parent_map[p.id] = p

        return rel_map, child_map, parent_map

    visited_features: set = set()

    def traverse_feature(feature_id, prefix, is_last, level, print_header=True):
        if level > hops:
            return
        fnode = rpg.get_node_by_id(feature_id)
        if not fnode:
            conn = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            if print_header:
                rtn_str.append(f"{prefix}{conn}[missing feature {feature_id}]")
            return

        if print_header:
            label = fnode.feature_path() if level == 0 else _tail_name(fnode.feature_path(), getattr(fnode, "name", ""))
            cur_deps = _get_deps_from_feature_id(fnode.id)
            conn = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            rtn_str.append(f"{prefix}{conn}{label} [{_compact_dep_files(cur_deps)}]")
            new_prefix = prefix + (" " if is_last else "\u2502") + " " * (len(conn) - 1)
        else:
            new_prefix = prefix

        rel_map, child_map, parent_map = _gather_feature_relations(fnode)
        child_ids = set(child_map.keys())
        parent_ids = set(parent_map.keys())
        dep_only = [fid for fid in rel_map if fid not in child_ids and fid not in parent_ids]

        items: List[Tuple[str, str]] = []
        for fid in sorted(dep_only, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(("dep", fid))
        for fid in sorted(parent_ids, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(("parent", fid))
        for fid in sorted(child_ids, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(("child", fid))

        for idx, (kind, tfid) in enumerate(items):
            sub_last = idx == len(items) - 1
            sub_conn = "\u2514\u2500\u2500 " if sub_last else "\u251c\u2500\u2500 "
            tnode = rpg.get_node_by_id(tfid)
            if not tnode:
                rtn_str.append(f"{new_prefix}{sub_conn}[missing feature {tfid}]")
                continue
            tlabel = _tail_name(tnode.feature_path(), getattr(tnode, "name", ""))
            tdeps = _get_deps_from_feature_id(tnode.id)
            types_str = ", ".join(_sorted_types(rel_map[tfid])) or "rel"
            rtn_str.append(f"{new_prefix}{sub_conn}{types_str} \u2500\u2500 {tlabel} [{_compact_dep_files(tdeps)}]")

            if kind == "child":
                branch_prefix = new_prefix + ("    " if sub_last else "\u2502   ")
                if tfid not in visited_features:
                    visited_features.add(tfid)
                    traverse_feature(tfid, branch_prefix, True, level + 1, print_header=False)

    # --- Dispatch ---
    if visual_type == "code":
        if root not in G:
            return f"Root dep node not found in G: {root}"
        traverse_code(root, "", False, 0, None, None)
        used_map = labeler.used_label_to_paths()
        if any(used_map.values()):
            rtn_str.append("")
            rtn_str.append("Feature Paths (Legend):")
            rtn_str.extend(render_feature_paths_tree(used_map))
        return "\n".join(rtn_str)

    elif visual_type == "feature":
        root_node = _feature_by_id_or_path(root)
        if not root_node:
            return f"Root feature not found (id or path): {root}"
        visited_features.clear()
        visited_features.add(root_node.id)
        traverse_feature(root_node.id, "", False, 0, print_header=True)
        return "\n".join(rtn_str)

    return f"Invalid visual_type: {visual_type}. Expected 'code' or 'feature'."


# ============================================================================
# JSON-structure traversal
# ============================================================================

def traverse_json_structure(
    rpg: RPG,
    root: str,
    direction: str = "downstream",
    hops: int = 2,
    visual_type: Literal["code", "feature"] = "code",
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a normalized JSON graph representation.

    Source: RPG-ZeroRepo explore_rpg.py ``traverse_json_structure``
    """
    if hops == -1:
        hops = 20

    G = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}

    def _as_list(x):
        return [] if x is None else (x if isinstance(x, list) else [x])

    def _feature_by_id_or_path(arg):
        n = rpg.get_node_by_id(arg)
        if n:
            return n
        return rpg.get_node_by_feature_path(arg) if isinstance(arg, str) else None

    def _get_features_from_dep(dep_id: str) -> List[str]:
        return _as_list(dep2rpg.get(dep_id, []))

    def _get_deps_from_feature_id(fid: str) -> List[str]:
        res = []
        for dep_id, rpg_ids in dep2rpg.items():
            if fid in _as_list(rpg_ids):
                res.append(dep_id)
        return res

    def _ntype_invalid(nid: str) -> bool:
        if node_type_filter is None:
            return False
        return G.nodes[nid].get("type") not in node_type_filter

    def _etype_invalid(etype: str) -> bool:
        if edge_type_filter is None:
            return False
        return etype not in edge_type_filter

    def _is_test_safe(nid: str) -> bool:
        try:
            return bool(is_test_file(nid))
        except Exception:
            return False

    # --- Code view ---
    def build_code_normalized() -> Dict[str, Any]:
        if root not in G:
            return {"error": f"Root dep node not found in G: {root}"}

        nodes: Dict[str, Dict[str, Any]] = {}
        edge_types_map: Dict[Tuple[str, str], set] = defaultdict(set)
        spanning: Dict[str, List[Dict[str, Any]]] = OrderedDict()
        visited_depth: Dict[str, int] = {root: 0}

        def _ensure_node(dep_id: str):
            if dep_id in nodes:
                return
            feats = []
            for fid in _get_features_from_dep(dep_id):
                fn = rpg.get_node_by_id(fid)
                if fn:
                    feats.append(fn.feature_path())
            nodes[dep_id] = {"dep": dep_id, "features": feats}
            spanning.setdefault(dep_id, [])

        _ensure_node(root)

        stack = [(root, None, None, 0, None)]

        def _dir_ok(d, ed, is_root):
            if is_root and direction == "both":
                return True
            if ed is None:
                return direction in ("downstream", "upstream", "both")
            return d == ed

        while stack:
            dep, from_dep, etype_from_parent, level, edirection = stack.pop()
            _ensure_node(dep)

            if from_dep is not None and etype_from_parent is not None:
                edge_types_map[(from_dep, dep)].add(etype_from_parent)
                lst = spanning.setdefault(from_dep, [])
                if lst and lst[-1]["to"] == dep:
                    lst[-1]["types"].add(etype_from_parent)
                else:
                    lst.append({"to": dep, "types": {etype_from_parent}})

            if level >= hops:
                continue

            nexts: List[Tuple[str, str, str]] = []

            if _dir_ok("downstream", edirection, dep == root):
                for neigh in G.successors(dep):
                    if _ntype_invalid(neigh) or _is_test_safe(neigh):
                        continue
                    for _, e in G[dep][neigh].items():
                        etype = e.get("type")
                        if _etype_invalid(etype):
                            continue
                        nexts.append((etype, neigh, "downstream"))

            if _dir_ok("upstream", edirection, dep == root):
                for neigh in G.predecessors(dep):
                    if _ntype_invalid(neigh) or _is_test_safe(neigh):
                        continue
                    for _, e in G[neigh][dep].items():
                        etype = e.get("type")
                        if _etype_invalid(etype):
                            continue
                        nexts.append((etype + "-by", neigh, "upstream"))

            # Functional containment
            mapped_fids = _get_features_from_dep(dep)
            if _dir_ok("downstream", edirection, dep == root):
                for fid in mapped_fids:
                    fn = rpg.get_node_by_id(fid)
                    if not fn:
                        continue
                    for ch in fn.children():
                        for dc in _get_deps_from_feature_id(ch.id):
                            if dc not in G or _ntype_invalid(dc):
                                continue
                            nexts.append(("composes", dc, "downstream"))

            if _dir_ok("upstream", edirection, dep == root):
                for fid in mapped_fids:
                    fn = rpg.get_node_by_id(fid)
                    if not fn:
                        continue
                    p = fn.parent()
                    if not p:
                        continue
                    for dp in _get_deps_from_feature_id(p.id):
                        if dp not in G or _ntype_invalid(dp):
                            continue
                        nexts.append(("composed-by", dp, "upstream"))

            seen_here: set = set()
            for et, to_dep, ndir in nexts:
                key = (et, to_dep)
                if key in seen_here:
                    continue
                seen_here.add(key)
                _ensure_node(to_dep)
                prev = visited_depth.get(to_dep)
                if prev is None or level + 1 < prev:
                    visited_depth[to_dep] = level + 1
                    stack.append((to_dep, dep, et, level + 1, ndir))
                edge_types_map[(dep, to_dep)].add(et)

        edges_out = [
            {"from": src, "to": tgt, "types": sorted(ts)}
            for (src, tgt), ts in edge_types_map.items()
        ]
        for _src, lst in spanning.items():
            for item in lst:
                item["types"] = sorted(item["types"])

        return {
            "type": "code",
            "meta": {"direction": direction, "hops": hops},
            "root_dep": root,
            "nodes": nodes,
            "edges": edges_out,
            "spanning": spanning,
        }

    # --- Feature view ---
    def build_feature_normalized() -> Dict[str, Any]:
        root_node = _feature_by_id_or_path(root)
        if not root_node:
            return {"error": f"Root feature not found (id or path): {root}"}

        features: Dict[str, Dict[str, Any]] = {}
        rel_types: Dict[Tuple[str, str], set] = defaultdict(set)
        spanning_children: Dict[str, List[str]] = OrderedDict()
        seen_pairs: set = set()

        def _tail(path: str) -> str:
            return path.split("/")[-1] if path else path

        def _ensure_feature(fid: str):
            n = rpg.get_node_by_id(fid)
            if not n:
                return None
            fpath = n.feature_path()
            if fpath not in features:
                features[fpath] = {
                    "path": fpath,
                    "label": _tail(fpath),
                    "deps": _get_deps_from_feature_id(fid),
                }
                spanning_children.setdefault(fpath, [])
            return n

        def _gather_rels(fid: str):
            n = rpg.get_node_by_id(fid)
            if not n:
                return [], None
            from_path = n.feature_path()
            _ensure_feature(fid)

            dep_ids = _get_deps_from_feature_id(fid)
            if direction in ("downstream", "both"):
                for did in dep_ids:
                    if did not in G:
                        continue
                    for neigh in G.successors(did):
                        for _, e in G[did][neigh].items():
                            etype = e.get("type")
                            if _etype_invalid(etype):
                                continue
                            for tfid in _get_features_from_dep(neigh):
                                tn = rpg.get_node_by_id(tfid)
                                if not tn:
                                    continue
                                _ensure_feature(tfid)
                                rel_types[(from_path, tn.feature_path())].add(etype)

            if direction in ("upstream", "both"):
                for did in dep_ids:
                    if did not in G:
                        continue
                    for neigh in G.predecessors(did):
                        for _, e in G[neigh][did].items():
                            etype = e.get("type")
                            if _etype_invalid(etype):
                                continue
                            for tfid in _get_features_from_dep(neigh):
                                tn = rpg.get_node_by_id(tfid)
                                if not tn:
                                    continue
                                _ensure_feature(tfid)
                                rel_types[(from_path, tn.feature_path())].add(etype + "-by")

            child_ids = []
            if direction in ("downstream", "both"):
                for ch in n.children():
                    _ensure_feature(ch.id)
                    rel_types[(from_path, ch.feature_path())].add("composes")
                    child_ids.append(ch.id)

            parent_id = None
            if direction in ("upstream", "both"):
                p = n.parent()
                if p:
                    _ensure_feature(p.id)
                    rel_types[(from_path, p.feature_path())].add("composed-by")
                    parent_id = p.id

            return child_ids, parent_id

        root_id = root_node.id
        _ensure_feature(root_id)

        bfs: List[Tuple[str, int]] = [(root_id, 0)]
        visited: set = {root_id}

        while bfs:
            fid, level = bfs.pop()
            n = rpg.get_node_by_id(fid)
            if not n:
                continue
            cur_path = n.feature_path()

            child_ids, _ = _gather_rels(fid)
            if level >= hops:
                continue

            for ch_id in child_ids:
                ch_node = rpg.get_node_by_id(ch_id)
                if not ch_node:
                    continue
                ch_path = ch_node.feature_path()
                if ch_id not in visited:
                    visited.add(ch_id)
                    bfs.append((ch_id, level + 1))
                if ch_path not in spanning_children.get(cur_path, []):
                    spanning_children.setdefault(cur_path, []).append(ch_path)

        relations_out = []
        for (src, tgt), ts in rel_types.items():
            key = (src, tgt, tuple(sorted(ts)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            relations_out.append({"from": src, "to": tgt, "types": sorted(ts)})

        return {
            "type": "feature",
            "meta": {"direction": direction, "hops": hops},
            "root_feature": root_node.feature_path(),
            "features": features,
            "relations": relations_out,
            "spanning_children": spanning_children,
        }

    if visual_type == "code":
        return build_code_normalized()
    elif visual_type == "feature":
        return build_feature_normalized()
    return {"error": "Invalid visual_type: expected 'code' or 'feature'."}


# ============================================================================
# Main entry point
# ============================================================================

def explore_tree_structure(
    start_code_entities: List[str],
    start_feature_entities: Optional[List[str]] = None,
    direction: str = "downstream",
    traversal_depth: int = 2,
    entity_type_filter: Optional[List[str]] = None,
    dependency_type_filter: Optional[List[str]] = None,
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[RepoEntitySearcher] = None,
    return_json: bool = False,
) -> Tuple[str, bool]:
    """Explore RPG graph structure from given starting entities.

    Args:
        start_code_entities: Code entity IDs to start from.
        start_feature_entities: Feature paths to start from.
        direction: ``'downstream'``, ``'upstream'``, or ``'both'``.
        traversal_depth: Depth of traversal (-1 = unlimited).
        entity_type_filter: Filter by node types.
        dependency_type_filter: Filter by edge types.
        rpg: RPG instance.
        entity_searcher: RepoEntitySearcher instance.
        return_json: Return JSON instead of tree text.

    Returns:
        (formatted_result, success) tuple.

    Source: RPG-ZeroRepo explore_rpg.py ``explore_tree_structure``
    """
    if start_feature_entities is None:
        start_feature_entities = []

    code_entities, code_hints, feature_entities, feature_hints = \
        _validate_graph_explorer_inputs(
            start_code_entities=start_code_entities,
            start_feature_entities=start_feature_entities,
            direction=direction,
            traversal_depth=traversal_depth,
            node_type_filter=entity_type_filter,
            edge_type_filter=dependency_type_filter,
            rpg=rpg,
            entity_searcher=entity_searcher,
        )

    suc = bool(code_entities or feature_entities)

    if return_json:
        code_rtns = {
            node: traverse_json_structure(
                rpg, node, direction, traversal_depth, "code",
                node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter,
            )
            for node in code_entities
        }
        feature_rtns = {
            node: traverse_json_structure(
                rpg, node, direction, traversal_depth, "feature",
                node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter,
            )
            for node in feature_entities
        }
        code_str = json.dumps(code_rtns)
        feature_str = json.dumps(feature_rtns)
    else:
        code_rtns_list = [
            traverse_tree_structure(
                rpg, node, direction, traversal_depth, "code",
                node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter,
            )
            for node in code_entities
        ]
        feature_rtns_list = [
            traverse_tree_structure(
                rpg, node, direction, traversal_depth, "feature",
                node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter,
            )
            for node in feature_entities
        ]
        code_str = "\n\n".join(code_rtns_list)
        feature_str = "\n\n".join(feature_rtns_list)

    sections = []
    if code_str.strip():
        sections.append("==== Code Results ====\n" + code_str)
    if feature_str.strip():
        sections.append("==== Feature Results ====\n" + feature_str)

    rtns_str = "\n\n".join(sections)
    hints = code_hints + "\n\n" + feature_hints
    if hints.strip():
        rtns_str += "\n\n==== Hints ====\n" + hints.strip()

    return rtns_str.strip(), suc
