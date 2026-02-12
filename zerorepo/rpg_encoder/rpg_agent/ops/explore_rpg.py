import json
from typing import (
    List, Optional, Dict,
    Literal, Any, Tuple,
    Iterable
)
from collections import defaultdict, OrderedDict
from networkx import MultiDiGraph
from zerorepo.rpg_gen.base.rpg import NodeType, EdgeType, RPG, Node
from zerorepo.utils.repo import is_test_file
from ..env import RepoEntitySearcher
from .search_node_by_meta import fuzzy_retrieve
from .search_node_by_feature import fuzzy_match_search_feature
from rapidfuzz import process, fuzz

def _fuzzy_match_feature_path(
    rpg: RPG,
    query_path: str,
    top_k: int = 5,
    score_cutoff: float = 50.0
) -> List[Tuple[str, Node, float]]:
    """Match query_path against all feature paths in the RPG using
    character-level similarity (fuzz.ratio). Catches typos in paths.
    Returns [(matched_path, node, score)] sorted by score desc.
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
        score_cutoff=score_cutoff
    )
    return [(path, paths_map[path], score) for path, score, _ in matches]


def _validate_graph_explorer_inputs(
    start_code_entities: List[str],
    start_feature_entities: List[str]=[],
    direction: str = 'downstream',
    traversal_depth: int = 1,
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[RepoEntitySearcher] = None
):
    """evaluate input arguments
    """
    assert direction in ['downstream', 'upstream', 'both'], (
        "Invalid value for `direction`: Expected one of 'downstream', 'upstream', and 'both'. "
        f"Received: '{direction}'."
    )
    assert traversal_depth == -1 or traversal_depth >= 0, (
        "Invalid value for `traversal_depth`: It must be either -1 or a non-negative integer (>= 0). "
        f"Received: {traversal_depth}."
    )

    valid_edge_types = [n.value for n in EdgeType]
    valid_node_types = [e.value for e in NodeType]
    
    if isinstance(node_type_filter, list):
        invalid_ntypes = []
        for ntype in invalid_ntypes:
            if ntype not in valid_node_types:
                invalid_ntypes.append(ntype)
        assert len(invalid_ntypes) == 0, \
            f"Invalid node types {invalid_ntypes} in node_type_filter. Expected node type in {valid_node_types}"
    if isinstance(edge_type_filter, list):
        invalid_etypes = []
        for etype in edge_type_filter:
            if etype not in valid_edge_types:
                invalid_etypes.append(etype)
        assert len(invalid_etypes) == 0, \
            f"Invalid edge types {invalid_etypes} in edge_type_filter. Expected edge type in {valid_edge_types}"

    # Get dep_graph and dep2rpg from RPG
    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}

    code_hints = ''
    valid_code_entities = []
    for i, root in enumerate(start_code_entities):
        # process node name
        if root != '/':
            root = root.strip('/')
        if root.endswith('.__init__'):
            root = root[:-(len('.__init__'))]

        # validate node name
        if dep_graph is None or root not in dep_graph:
            # 1. Try path-level fuzzy matching first (catches typos)
            dep_path_matches = []
            if dep_graph is not None:
                dep_path_matches = process.extract(
                    root, list(dep_graph.nodes()),
                    scorer=fuzz.ratio,
                    limit=5,
                    score_cutoff=50
                )

            if dep_path_matches:
                best_match, best_score, _ = dep_path_matches[0]
                if best_score >= 80:
                    code_hints += f'The entity path `{root}` does not exist. Did you mean `{best_match}`?\n'
                else:
                    code_hints += f'The entity path `{root}` does not exist. Similar entities:\n'
                for match_id, score, _ in dep_path_matches:
                    rpg_ids = dep2rpg.get(match_id, [])
                    rpg_ids = rpg_ids if isinstance(rpg_ids, list) else [rpg_ids]
                    feature_paths = []
                    for rpg_id in rpg_ids:
                        rpg_node = rpg.get_node_by_id(rpg_id)
                        if rpg_node:
                            feature_paths.append(rpg_node.feature_path())
                    fps = ", ".join(feature_paths) if feature_paths else ""
                    hint_line = f'  - `{match_id}`'
                    if fps:
                        hint_line += f' (feature paths: {fps})'
                    hint_line += f' [similarity: {score:.0f}%]\n'
                    code_hints += hint_line
                code_hints += "Source: Retrieved using path-level fuzzy match.\n\n"
            else:
                # 2. Fall back to token-level fuzzy matching (BM25 + token_set_ratio)
                module_nids = fuzzy_retrieve(
                    keyword=root,
                    rpg=rpg
                ) if rpg else []

                module_datas = entity_searcher.get_node_data(module_nids, return_code_content=False) if entity_searcher else []
                if len(module_datas) > 0:
                    code_hints += f'The entity name `{root}` is invalid. Based on your input, here are some candidate entities you might be referring to:\n'
                    for module in module_datas[:5]:
                        ntype = module['type']
                        nid = module['node_id']
                        rpg_ids = dep2rpg.get(nid, [])
                        rpg_ids = rpg_ids if isinstance(rpg_ids, list) \
                            else [rpg_ids]

                        feature_paths = []
                        for rpg_id in rpg_ids:
                            rpg_node = rpg.get_node_by_id(rpg_id)
                            if not rpg_node:
                                continue
                            rpg_f_path = rpg_node.feature_path()
                            feature_paths.append(rpg_f_path)
                        feature_paths_str = ", ".join(feature_paths)
                        code_hints += f'{ntype}: feature paths: {feature_paths_str}\nentity path: {nid}\n'
                    code_hints += "Source: Retrieved entity using keyword search (fuzzy match).\n\n"
                else:
                    code_hints += f'The entity name `{root}` is invalid. There are no possible candidate entities in record.\n'
        else:
            valid_code_entities.append(root)

    feature_hints = ''
    valid_feature_entities = []

    for i, f_path in enumerate(start_feature_entities):
   
        if f_path.startswith("/"):
            f_path = f_path[1:]
        f_path = f_path.strip()
        
        f_node: Node = rpg.get_node_by_feature_path(
            feature_path=f_path
        )
        
        if f_node is None:
            # 1. Try path-level fuzzy matching first (catches typos in paths)
            path_matches = _fuzzy_match_feature_path(rpg, f_path) if rpg else []

            if path_matches:
                best_path, best_node, best_score = path_matches[0]
                if best_score >= 80:
                    feature_hints += f'The feature path `{f_path}` does not exist. Did you mean `{best_path}`?\n'
                else:
                    feature_hints += f'The feature path `{f_path}` does not exist. Similar feature paths:\n'
                for m_path_str, m_node, m_score in path_matches[:5]:
                    m_type = m_node.meta.type_name
                    m_entity_path = m_node.meta.path
                    feature_hints += f"  - {m_type}: feature path: {m_path_str}\n    entity path: {m_entity_path} [similarity: {m_score:.0f}%]\n"
                feature_hints += "Source: Retrieved using path-level fuzzy match.\n\n"
            else:
                # 2. Fall back to token-level fuzzy matching
                fuzzy_results: List[Tuple[Node, float]] = fuzzy_match_search_feature(
                    rpg=rpg,
                    keyword=f_path,
                    valid_nodes=[]
                )
                matched_nodes: List[Node] = [result[0] for result in fuzzy_results]

                if len(matched_nodes) > 0:
                    feature_hints += f'The feature path `{f_path}` does not exist. Based on your input, here are some candidate entities you might be referring to:\n'
                    for m_node in matched_nodes[:5]:
                        m_feature_path = m_node.feature_path()
                        m_type = m_node.meta.type_name
                        m_path = m_node.meta.path
                        feature_hints += f"  - {m_type}: feature path: {m_feature_path}\n    entity path: {m_path}\n"
                    feature_hints += "Source: Retrieved entity using keyword search (fuzzy match).\n\n"
                else:
                    feature_hints += f'The feature path `{f_path}` does not exist. No similar features found.\n'

        else:
            valid_feature_entities.append(f_path)

    return valid_code_entities, code_hints, valid_feature_entities, feature_hints


class FeatureLabeler:
    def __init__(self, rpg: "RPG", strip_root: Optional[str] = None):
        self.rpg = rpg
        self.dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}
        self.strip_root = strip_root  # e.g., 'requests'
        self._setkey_to_label: Dict[Tuple[str, ...], str] = {}
        self._label_to_paths: Dict[str, Tuple[str, ...]] = {}
        self._used_labels: List[str] = []

    @staticmethod
    def _split(p: str) -> List[str]:
        return p.split('/') if p else []

    @staticmethod
    def _join(parts: Iterable[str]) -> str:
        return '/'.join(parts)

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


class _TrieNode:
    def __init__(self, name: str):
        self.name = name
        self.children: Dict[str, "_TrieNode"] = {}
        self.labels: List[str] = []

def _insert_path(root: _TrieNode, path: str, label: str):
    cur = root
    for seg in path.split('/'):
        if seg not in cur.children:
            cur.children[seg] = _TrieNode(seg)
        cur = cur.children[seg]
    if label not in cur.labels:
        cur.labels.append(label)

def _render_trie(root: _TrieNode) -> List[str]:
    lines: List[str] = []
    def dfs(node: _TrieNode, prefix: str, is_last: bool):
        children = sorted(node.children.values(), key=lambda n: n.name)
        for i, ch in enumerate(children):
            last = (i == len(children) - 1)
            conn = '└── ' if last else '├── '
            tag = f" [{', '.join(sorted(ch.labels))}]" if ch.labels else ""
            lines.append(f"{prefix}{conn}{ch.name}{tag}")
            new_prefix = prefix + ('    ' if last else '│   ')
            dfs(ch, new_prefix, last)
    dfs(root, "", True)
    return lines

def render_feature_paths_tree(label_to_paths: Dict[str, List[str]]) -> List[str]:
    trie_root = _TrieNode("")
    for label, paths in sorted(label_to_paths.items(), key=lambda kv: kv[0]):
        for p in sorted(paths):
            _insert_path(trie_root, p, label)
    return _render_trie(trie_root)


def traverse_tree_structure(
    rpg: "RPG",
    root,
    direction: str = 'downstream',
    hops: int = 2,
    visual_type: Literal["code", "feature"] = "code",
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
) -> str:
    """
    - visual_type="code": Code dependency tree; nodes show only [F*] suffixes; output a Feature Paths (Legend) tree at the end.
    - visual_type="feature": Each line is `feature_label [dep ids as-is]`;
        * the root node label is the full path; other nodes use only the last segment;
        * if the same target has multiple relationships (imports/invokes/contains/... + composes/composed-by), merge them into one line;
        * for non-structural dependencies: print only one edge line (no recursion);
        * for structural edges (composes/composed-by): print one edge line, then recursively expand the target, but do not print the target header line again.    """
    if hops == -1:
        hops = 20

    # Get dep_graph and dep2rpg from RPG
    G = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}

    rtn_str: List[str] = []

    # ----------------------------
    # Helpers
    # ----------------------------
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
        ntype = G.nodes[nid].get("type")
        return ntype not in node_type_filter

    def _etype_invalid(etype: str) -> bool:
        if edge_type_filter is None:
            return False
        return etype not in edge_type_filter

    def _is_test_file_safe(nid: str) -> bool:
        try:
            return bool(is_test_file(nid))
        except Exception:
            return False

    def _infer_strip_root_from_dep_id(dep_id: str) -> Optional[str]:
        if not isinstance(dep_id, str):
            return None
        s = dep_id
        if s.startswith("src/"):
            s = s[4:]
        parts = s.split("/")
        return parts[0] if parts else None

    strip_root = _infer_strip_root_from_dep_id(root)
    labeler = FeatureLabeler(rpg=rpg, strip_root=strip_root)

    # ========== Code view (F* + Legend) ==========
    def traverse_code(node_id, prefix, is_last, level, edge_type, edirection):
        if level > hops:
            return
        label = labeler.label_for_dep(node_id)
        feature_tag = f"[{label}]" if label else ""
        if node_id == root and level == 0:
            rtn_str.append(f"{node_id}  {feature_tag}".rstrip())
            new_prefix = ''
            edirection = direction
        else:
            connector = '└── ' if is_last else '├── '
            conn = f"{connector}{edge_type} ── "
            rtn_str.append(f"{prefix}{conn}{node_id}  {feature_tag}".rstrip())
            new_prefix = prefix + (' ' if is_last else '│') + ' ' * (len(conn) - 1)

        neigh_ids, etypes, edirs = [], [], []
        seen_local = set()

        if edirection in ('downstream', None) or (node_id == root and direction == 'both'):
            for neigh in G.successors(node_id):
                if _ntype_invalid(neigh):
                    continue
                edges = G[node_id][neigh]
                for key in edges:
                    etype = edges[key].get('type')
                    if _etype_invalid(etype) or _is_test_file_safe(neigh):
                        continue
                    k = (etype, node_id, neigh)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(neigh)
                    etypes.append(etype)
                    edirs.append('downstream')

        if edirection in ('upstream', None) or (node_id == root and direction == 'both'):
            for neigh in G.predecessors(node_id):
                if _ntype_invalid(neigh):
                    continue
                edges = G[neigh][node_id]
                for key in edges:
                    etype = edges[key].get('type')
                    if _etype_invalid(etype) or _is_test_file_safe(neigh):
                        continue
                    k = (etype + '-by', neigh, node_id)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(neigh)
                    etypes.append(etype + '-by')
                    edirs.append('upstream')

        mapped_fids = _get_features_from_dep(node_id)
        if edirection in ('downstream', None) or (node_id == root and direction == 'both'):
            for fid in mapped_fids:
                fnode = rpg.get_node_by_id(fid)
                if not fnode:
                    continue
                for ch in fnode.children():
                    for dep_child in _get_deps_from_feature_id(ch.id):
                        if dep_child not in G or _ntype_invalid(dep_child):
                            continue
                        k = ('composes', node_id, dep_child)
                        if k in seen_local:
                            continue
                        seen_local.add(k)
                        neigh_ids.append(dep_child)
                        etypes.append('composes')
                        edirs.append('downstream')

        if edirection in ('upstream', None) or (node_id == root and direction == 'both'):
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
                    k = ('composed-by', dep_parent, node_id)
                    if k in seen_local:
                        continue
                    seen_local.add(k)
                    neigh_ids.append(dep_parent)
                    etypes.append('composed-by')
                    edirs.append('upstream')

        for i, (nid, et, edir) in enumerate(zip(neigh_ids, etypes, edirs)):
            is_last_child = (i == len(neigh_ids) - 1)
            traverse_code(nid, new_prefix, is_last_child, level + 1, et, edir)

    # ========== Feature view ==========
    def _tail_name(path: str, fallback: str = "") -> str:
        if not path:
            return fallback
        parts = path.split("/")
        return parts[-1] if parts else (fallback or path)

    def _compact_dep_files(dep_ids: List[str]) -> str:
        """Output dep IDs exactly as-is (no truncation, no deduplication, no omission)."""
        return ", ".join(dep_ids) if dep_ids else "—"

    def _sorted_types(types: Iterable[str]) -> List[str]:
        order_key = lambda s: (s not in ('composes', 'composed-by'), s)
        return sorted(set(types), key=order_key)

    def _gather_relations(feature_node) -> Tuple[Dict[str, set], Dict[str, Any], Dict[str, Any]]:
        """
        Return:
            - rel_map: { target_fid -> set(edge_types) } (includes code dependencies + structural edges)
            - child_map: { child_fid -> child_node } (structural downstream)
            - parent_map: { parent_fid -> parent_node } (structural upstream, at most one)
        """
        rel_map: Dict[str, set] = defaultdict(set)
        child_map: Dict[str, Any] = {}
        parent_map: Dict[str, Any] = {}

        dep_ids = _get_deps_from_feature_id(feature_node.id)
        for dep_id in dep_ids:
            if dep_id not in G:
                continue
            # downstream
            if direction in ('downstream', 'both'):
                for neigh in G.successors(dep_id):
                    edges = G[dep_id][neigh]
                    for key in edges:
                        etype = edges[key].get('type')
                        if _etype_invalid(etype):
                            continue
                        for tfid in _get_features_from_dep(neigh):
                            if rpg.get_node_by_id(tfid):
                                rel_map[tfid].add(etype)
            # upstream
            if direction in ('upstream', 'both'):
                for neigh in G.predecessors(dep_id):
                    edges = G[neigh][dep_id]
                    for key in edges:
                        etype = edges[key].get('type')
                        if _etype_invalid(etype):
                            continue
                        etype2 = etype + "-by"
                        for tfid in _get_features_from_dep(neigh):
                            if rpg.get_node_by_id(tfid):
                                rel_map[tfid].add(etype2)

        if direction in ('downstream', 'both'):
            for ch in feature_node.children():
                rel_map[ch.id].add('composes')
                child_map[ch.id] = ch
        if direction in ('upstream', 'both'):
            p = feature_node.parent()
            if p:
                rel_map[p.id].add('composed-by')
                parent_map[p.id] = p

        return rel_map, child_map, parent_map

    visited_features: set = set()

    def traverse_feature(feature_id: str, prefix: str, is_last: bool, level: int, print_header: bool = True):
        if level > hops:
            return

        fnode = rpg.get_node_by_id(feature_id)
        if not fnode:
            connector = '└── ' if is_last else '├── '
            if print_header:
                rtn_str.append(f"{prefix}{connector}[missing feature {feature_id}]")
            return

        if print_header:
            label = fnode.feature_path() if level == 0 else _tail_name(fnode.feature_path(), getattr(fnode, "name", ""))
            cur_dep_ids = _get_deps_from_feature_id(fnode.id)
            code_paths = _compact_dep_files(cur_dep_ids)
            connector = '└── ' if is_last else '├── '
            rtn_str.append(f"{prefix}{connector}{label} [{code_paths}]")
            new_prefix = prefix + (' ' if is_last else '│') + ' ' * (len(connector) - 1)
        else:
            new_prefix = prefix

        rel_map, child_map, parent_map = _gather_relations(fnode)
        child_ids = set(child_map.keys())
        parent_ids = set(parent_map.keys())
        dep_only_targets = [fid for fid in rel_map.keys() if fid not in child_ids and fid not in parent_ids]
        
        items: List[Tuple[str, str]] = []
        for fid in sorted(dep_only_targets, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(('dep', fid))
        for fid in sorted(parent_ids, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(('parent', fid))
        for fid in sorted(child_ids, key=lambda x: rpg.get_node_by_id(x).feature_path() if rpg.get_node_by_id(x) else x):
            items.append(('child', fid))

        for idx, (kind, tfid) in enumerate(items):
            sub_is_last = (idx == len(items) - 1)
            sub_conn = '└── ' if sub_is_last else '├── '
            tnode = rpg.get_node_by_id(tfid)
            if not tnode:
                rtn_str.append(f"{new_prefix}{sub_conn}[missing feature {tfid}]")
                continue
            tlabel = _tail_name(tnode.feature_path(), getattr(tnode, "name", ""))
            tdeps = _get_deps_from_feature_id(tnode.id)
            types_str = ", ".join(_sorted_types(rel_map[tfid])) or "rel"
            rtn_str.append(f"{new_prefix}{sub_conn}{types_str} ── {tlabel} [{_compact_dep_files(tdeps)}]")

            if kind == 'child':
                branch_prefix = new_prefix + ('    ' if sub_is_last else '│   ')
                if tfid not in visited_features:
                    visited_features.add(tfid)
                    traverse_feature(tfid, branch_prefix, True, level + 1, print_header=False)

    # ----------------------------
    # Dispatch + Legend
    # ----------------------------
    if visual_type == "code":
        if root not in G:
            return f"Root dep node not found in G: {root}"
        traverse_code(root, '', False, 0, None, None)

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
        traverse_feature(root_node.id, '', False, 0, print_header=True)
        return "\n".join(rtn_str)

    else:
        return f"Invalid visual_type: {visual_type}. Expected 'code' or 'feature'."



def traverse_json_structure(
    rpg: "RPG",
    root,
    direction: str = "downstream",
    hops: int = 2,
    visual_type: Literal["code", "feature"] = "code",
    node_type_filter: Optional[List[str]] = None,
    edge_type_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Normalize JSON (deduplicate / remove redundancy):

    - Code view:
    {
        "type": "code",
        "meta": {"direction": ..., "hops": ...},
        "root_dep": "<dep_id>",
        "nodes": { "<dep_id>": { "dep": "...", "features": [<feature_path>...] }, ... },
        "edges": [ {"from":"<dep_id>","to":"<dep_id>","types":["contains","invokes",...]} , ... ],
        "spanning": { "<dep_id>": [ {"to":"<dep_id>","types":["contains"]}, ... ], ... }  # adjacency order for the spanning tree only
    }

    - Feature view (only recurse on children via composes; other dependencies are merged and listed but not recursed):
    {
        "type": "feature",
        "meta": {"direction": ..., "hops": ...},
        "root_feature": "<feature_path>",
        "features": { "<feature_path>": { "path": "...", "label": "last segment", "deps": [<dep_id>...] }, ... },
        "relations": [ {"from":"<feature_path>","to":"<feature_path>","types":[...]} , ... ],  # imports/invokes/.../composed-by, etc.
        "spanning_children": { "<feature_path>": ["<child_path>", ...], ... }                 # child order for the spanning tree only
    }

    Rules:
    - Do not trim/omit anything: dep/feature paths must remain exactly as the original values.
    - Merge multi-edges: for the same (from, to), merge all relations into the `types` list (deduplicated).
    - Code dependency edges are subject to `node_type_filter` / `edge_type_filter`; functional structural edges (composes/composed-by) are not subject to `edge_type_filter`.
    - Code view: the spanning tree includes code dependencies + derived functional containment; Feature view: only recurse on children, while listing all other relations in `relations`.
    """
    if hops == -1:
        hops = 20

    # Get dep_graph and dep2rpg from RPG
    G = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}

    # ----------------------------
    # Helpers
    # ----------------------------
    def _as_list(x):
        if x is None: return []
        return x if isinstance(x, list) else [x]

    def _feature_by_id_or_path(arg):
        n = rpg.get_node_by_id(arg)
        if n: return n
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
        if node_type_filter is None: return False
        ntype = G.nodes[nid].get("type")
        return ntype not in node_type_filter

    def _etype_invalid(etype: str) -> bool:
        if edge_type_filter is None: return False
        return etype not in edge_type_filter

    def _is_test_file_safe(nid: str) -> bool:
        try:
            return bool(is_test_file(nid))
        except Exception:
            return False

    # ------------------------------------------------------------
    # =============== CODE VIEW =====================
    # ------------------------------------------------------------
    def build_code_normalized() -> Dict[str, Any]:
        if root not in G:
            return {"error": f"Root dep node not found in G: {root}"}

        nodes: Dict[str, Dict[str, Any]] = {}
        edge_types: Dict[Tuple[str, str], set] = defaultdict(set)  # (src, tgt) -> {types}
        spanning: Dict[str, List[Dict[str, Any]]] = OrderedDict()  # 展示顺序（生成树）
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
            if dep_id not in spanning:
                spanning[dep_id] = []

        _ensure_node(root)

        stack = [(root, None, None, 0, None)]  # (dep, from_dep, edge_type, level, edirection)

        def _dir_ok(d: str, ed: Optional[str], is_root: bool) -> bool:
            if is_root and direction == "both":
                return True
            if ed is None:  
                return direction in ("downstream", "upstream", "both")
            return d == ed

        while stack:
            dep, from_dep, etype_from_parent, level, edirection = stack.pop()
            _ensure_node(dep)

            if from_dep is not None and etype_from_parent is not None:
                edge_types[(from_dep, dep)].add(etype_from_parent)
    
                lst = spanning.setdefault(from_dep, [])
                if lst and lst[-1]["to"] == dep:
                    lst[-1]["types"].add(etype_from_parent)
                else:
                    lst.append({"to": dep, "types": {etype_from_parent}})

            if level >= hops:
                continue

            nexts: List[Tuple[str, str, str]] = []  # (etype, to_dep, ndir)

            # downstream
            if _dir_ok("downstream", edirection, dep == root):
                for neigh in G.successors(dep):
                    if _ntype_invalid(neigh) or _is_test_file_safe(neigh):
                        continue
                    for _, e in G[dep][neigh].items():
                        etype = e.get("type")
                        if _etype_invalid(etype):
                            continue
                        nexts.append((etype, neigh, "downstream"))
            # upstream
            if _dir_ok("upstream", edirection, dep == root):
                for neigh in G.predecessors(dep):
                    if _ntype_invalid(neigh) or _is_test_file_safe(neigh):
                        continue
                    for _, e in G[neigh][dep].items():
                        etype = e.get("type")
                        if _etype_invalid(etype):
                            continue
                        nexts.append((etype + "-by", neigh, "upstream"))

            # 2) Functional containment (not subject to edge_type_filter)
            mapped_fids = _get_features_from_dep(dep)
            # composes: child feature -> its mapped dep
            if _dir_ok("downstream", edirection, dep == root):
                for fid in mapped_fids:
                    fnode = rpg.get_node_by_id(fid)
                    if not fnode: continue
                    for ch in fnode.children():
                        for dep_child in _get_deps_from_feature_id(ch.id):
                            if dep_child not in G or _ntype_invalid(dep_child):
                                continue
                            nexts.append(("composes", dep_child, "downstream"))
            # composed-by: parent feature -> its mapped dep
            if _dir_ok("upstream", edirection, dep == root):
                for fid in mapped_fids:
                    fnode = rpg.get_node_by_id(fid)
                    if not fnode: continue
                    p = fnode.parent()
                    if not p: continue
                    for dep_parent in _get_deps_from_feature_id(p.id):
                        if dep_parent not in G or _ntype_invalid(dep_parent):
                            continue
                        nexts.append(("composed-by", dep_parent, "upstream"))

            seen_here = set()
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
                edge_types[(dep, to_dep)].add(et)
     
        edges_out = []
        for (src, tgt), ts in edge_types.items():
            edges_out.append({"from": src, "to": tgt, "types": sorted(ts)})

        for src, lst in spanning.items():
            for item in lst:
                item["types"] = sorted(item["types"])

        return {
            "type": "code",
            "meta": {"direction": direction, "hops": hops},
            "root_dep": root,
            "nodes": nodes,
            "edges": edges_out,
            "spanning": spanning
        }

    # ------------------------------------------------------------
    # ============= FEATURE VIEW ================
    # ------------------------------------------------------------
    def build_feature_normalized() -> Dict[str, Any]:
        root_node = _feature_by_id_or_path(root)
        if not root_node:
            return {"error": f"Root feature not found (id or path): {root}"}

        features: Dict[str, Dict[str, Any]] = {}
        rel_types: Dict[Tuple[str, str], set] = defaultdict(set)  # (from_path, to_path) -> {types}
        spanning_children: Dict[str, List[str]] = OrderedDict()
        seen_pairs_global: set = set() 

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
                    "deps": _get_deps_from_feature_id(fid)
                }
                spanning_children.setdefault(fpath, [])
            return n

        def _gather_relations(fid: str):
            """Add all dependency edges related to the fid into rel_types; return the children/parent lists."""
            n = rpg.get_node_by_id(fid)
            if not n: return [], None
            from_path = n.feature_path()
            _ensure_feature(fid)

            dep_ids = _get_deps_from_feature_id(fid)
            if direction in ("downstream", "both"):
                for dep_id in dep_ids:
                    if dep_id not in G: continue
                    for neigh in G.successors(dep_id):
                        for _, e in G[dep_id][neigh].items():
                            etype = e.get("type")
                            if _etype_invalid(etype): continue
                            for tfid in _get_features_from_dep(neigh):
                                tnode = rpg.get_node_by_id(tfid)
                                if not tnode: continue
                                to_path = tnode.feature_path()
                                _ensure_feature(tfid)
                                rel_types[(from_path, to_path)].add(etype)
            if direction in ("upstream", "both"):
                for dep_id in dep_ids:
                    if dep_id not in G: continue
                    for neigh in G.predecessors(dep_id):
                        for _, e in G[neigh][dep_id].items():
                            etype = e.get("type")
                            if _etype_invalid(etype): continue
                            etype2 = etype + "-by"
                            for tfid in _get_features_from_dep(neigh):
                                tnode = rpg.get_node_by_id(tfid)
                                if not tnode: continue
                                to_path = tnode.feature_path()
                                _ensure_feature(tfid)
                                rel_types[(from_path, to_path)].add(etype2)

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

        stack: List[Tuple[str, int]] = [(root_id, 0)]
        visited: set = set([root_id])

        while stack:
            fid, level = stack.pop()
            n = rpg.get_node_by_id(fid)
            if not n: continue
            cur_path = n.feature_path()

            child_ids, parent_id = _gather_relations(fid)

            if level >= hops:
                continue

            for ch in child_ids:
                ch_node = rpg.get_node_by_id(ch)
                if not ch_node: continue
                ch_path = ch_node.feature_path()
                if ch not in visited:
                    visited.add(ch)
                    stack.append((ch, level + 1))
                if ch_path not in spanning_children[cur_path]:
                    spanning_children[cur_path].append(ch_path)

        relations_out = []
        for (src, tgt), ts in rel_types.items():
            key = (src, tgt, tuple(sorted(ts)))
            if key in seen_pairs_global:
                continue
            seen_pairs_global.add(key)
            relations_out.append({
                "from": src,
                "to": tgt,
                "types": sorted(ts)
            })

        return {
            "type": "feature",
            "meta": {"direction": direction, "hops": hops},
            "root_feature": root_node.feature_path(),
            "features": features,
            "relations": relations_out,
            "spanning_children": spanning_children
        }

    # ----------------------------
    # Dispatch
    # ----------------------------
    if visual_type == "code":
        return build_code_normalized()
    elif visual_type == "feature":
        return build_feature_normalized()
    else:
        return {"error": "Invalid visual_type: expected 'code' or 'feature'."}
    
    
    

def explore_tree_structure(
    start_code_entites: List[str],
    start_feature_entites: List[str],
    direction: str = 'downstream',
    traversal_depth: int = 2,
    entity_type_filter: Optional[List[str]] = None,
    dependency_type_filter: Optional[List[str]] = None,
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[RepoEntitySearcher] = None,
    return_json: bool = False
):
    """
    Explore tree structure from RPG.

    Args:
        start_code_entites: List of code entity IDs to start from
        start_feature_entites: List of feature paths to start from
        direction: 'downstream', 'upstream', or 'both'
        traversal_depth: Depth of traversal
        entity_type_filter: Filter by node types
        dependency_type_filter: Filter by edge types
        rpg: RPG instance (contains dep_graph and _dep_to_rpg_map)
        entity_searcher: RepoEntitySearcher instance
        return_json: Whether to return JSON format
    """
    code_entities, code_hints, feature_entities, feature_hints = \
        _validate_graph_explorer_inputs(
            start_code_entities=start_code_entites,
            start_feature_entities=start_feature_entites,
            direction=direction,
            traversal_depth=traversal_depth,
            node_type_filter=entity_type_filter,
            edge_type_filter=dependency_type_filter,
            rpg=rpg,
            entity_searcher=entity_searcher
        )

    suc = bool(code_entities or feature_entities)

    if return_json:
        code_rtns = {node: traverse_json_structure(rpg, node, direction,
                traversal_depth, "code", node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter)
                for node in code_entities
            }

        feature_rtns = {node: traverse_json_structure(rpg, node, direction,
                traversal_depth, "feature", node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter)
                for node in feature_entities
            }

        code_rtns_str = json.dumps(code_rtns)
        feature_rtns_str = json.dumps(feature_rtns)
    else:
        code_rtns = [traverse_tree_structure(rpg, node, direction,
                traversal_depth, "code", node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter)
                for node in code_entities
            ]

        feature_rtns = [traverse_tree_structure(rpg, node, direction,
                traversal_depth, "feature", node_type_filter=entity_type_filter,
                edge_type_filter=dependency_type_filter)
                for node in feature_entities
            ]

        code_rtns_str = "\n\n".join(code_rtns)
        feature_rtns_str = "\n\n".join(feature_rtns)
    
    
    sections = []

    if code_rtns_str.strip():
        sections.append("==== Code Results ====\n" + code_rtns_str)

    if feature_rtns_str.strip():
        sections.append("==== Feature Results ====\n" + feature_rtns_str)

    rtns_str = "\n\n".join(sections)
    
    hints = code_hints + "\n\n" + feature_hints
    if hints.strip():
        rtns_str += "\n\n" + "==== Hints ====\n" + hints.strip()
    
    return rtns_str.strip(), suc
    