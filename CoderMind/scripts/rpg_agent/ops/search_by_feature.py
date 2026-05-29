#!/usr/bin/env python3
"""Search RPG nodes by functional features (feature names, descriptions).

Provides exact, substring, and fuzzy matching against RPG node names and
feature paths, using the RPG's semantic layer rather than raw code.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/ops/search_node_by_feature.py
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from common.utils import normalize_text
from rpg_agent.env.query import QueryInfo, QueryResult
from rpg_agent.env.searcher import RepoEntitySearcher
from rpg_agent.ops.search_by_meta import (
    merge_query_results,
    rank_and_aggr_query_results,
)
from rpg import Node, NodeType, RPG


# ============================================================================
# Helpers
# ============================================================================

def _get_default_valid_nodes(rpg: RPG) -> List[Node]:
    """Return all FILE / CLASS / METHOD / FUNCTION nodes from the RPG."""
    detailed_node_types = [
        NodeType.FILE, NodeType.CLASS,
        NodeType.METHOD, NodeType.FUNCTION,
    ]
    valid_nodes: List[Node] = []
    for ntype in detailed_node_types:
        valid_nodes.extend(rpg.get_nodes_by_type(type_name=ntype))
    return valid_nodes


# ============================================================================
# Match algorithms
# ============================================================================

def exact_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: Optional[List[Node]] = None,
) -> List[Node]:
    """Find nodes whose normalized name exactly equals *keyword*.

    Source: RPG-ZeroRepo search_node_by_feature.py ``exact_match_search_feature``
    """
    if not keyword or not isinstance(keyword, str):
        return []

    norm_kw = normalize_text(keyword)
    if valid_nodes is None or not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    feature2node: Dict[str, List[Node]] = defaultdict(list)
    for node in valid_nodes:
        feature2node[normalize_text(node.name)].append(node)

    return list(feature2node.get(norm_kw, []))


def substring_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: Optional[List[Node]] = None,
    max_results: int = 10,
) -> List[Tuple[Node, float]]:
    """Rule-based substring search: checks node names and feature paths.

    Returns ``(Node, score)`` pairs sorted by relevance.

    Source: RPG-ZeroRepo search_node_by_feature.py ``substring_match_search_feature``
    """
    if not keyword or not isinstance(keyword, str):
        return []

    norm_kw = normalize_text(keyword)
    kw_parts = re.split(r"[\s._]+", norm_kw)
    kw_parts = [p for p in kw_parts if p and len(p) > 1]

    if valid_nodes is None or not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    results: List[Tuple[Node, float]] = []
    for node in valid_nodes:
        norm_name = normalize_text(node.name)
        norm_path = normalize_text(node.feature_path())

        score = 0.0

        # Full keyword match
        if norm_kw in norm_name:
            score = 90.0 if norm_kw == norm_name else 80.0
        elif norm_kw in norm_path:
            score = 70.0

        # Token-level match
        if score == 0 and kw_parts:
            matched_parts = sum(
                1 for p in kw_parts if p in norm_name or p in norm_path
            )
            if matched_parts == len(kw_parts):
                score = 60.0 + (matched_parts / len(kw_parts)) * 10
            elif matched_parts > 0:
                score = 40.0 + (matched_parts / len(kw_parts)) * 20

        # Reverse token match
        if score == 0:
            name_tokens = re.split(r"[\s._/]+", norm_path)
            name_tokens = [t for t in name_tokens if t and len(t) >= 3]
            if name_tokens:
                reverse_matched = sum(1 for t in name_tokens if t in norm_kw)
                if reverse_matched > 0:
                    coverage = sum(
                        len(t) for t in name_tokens if t in norm_kw
                    ) / max(len(norm_kw), 1)
                    score = 30.0 + min(coverage, 1.0) * 25.0

        if score > 0:
            results.append((node, score))

    results.sort(key=lambda x: -x[1])
    return results[:max_results]


def fuzzy_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: Optional[List[Node]] = None,
    top_k: int = 5,
) -> List[Tuple[Node, float]]:
    """Fuzzy matching via rapidfuzz ``token_set_ratio`` + ``WRatio``.

    Matches against node name, feature path, and description.

    Source: RPG-ZeroRepo search_node_by_feature.py ``fuzzy_match_search_feature``
    """
    if not keyword or not isinstance(keyword, str):
        return []

    keyword = normalize_text(keyword)
    if valid_nodes is None or not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    name2node: Dict[str, List[Node]] = defaultdict(list)
    path2node: Dict[str, List[Node]] = defaultdict(list)
    desc2node: Dict[str, List[Node]] = defaultdict(list)

    for node in valid_nodes:
        name2node[normalize_text(node.name)].append(node)
        path2node[normalize_text(node.feature_path())].append(node)
        if node.meta and node.meta.description:
            desc2node[normalize_text(node.meta.description)].append(node)

    all_names = list(name2node.keys())
    all_paths = list(path2node.keys())
    all_descs = list(desc2node.keys())

    if not all_names:
        return []

    node_best_score: Dict[str, Tuple[float, Node]] = {}

    def _update_best(matched_key: str, score: float, key2node_map: dict):
        for n in key2node_map[matched_key]:
            nid = n.id
            if nid not in node_best_score or score > node_best_score[nid][0]:
                node_best_score[nid] = (score, n)

    # Scorer 1: token_set_ratio on name and path
    for matched_name, score, _ in process.extract(
        keyword, all_names, scorer=fuzz.token_set_ratio, limit=top_k
    ):
        _update_best(matched_name, score, name2node)
    for matched_path, score, _ in process.extract(
        keyword, all_paths, scorer=fuzz.token_set_ratio, limit=top_k
    ):
        _update_best(matched_path, score, path2node)

    # Scorer 2: WRatio on name and path
    for matched_name, score, _ in process.extract(
        keyword, all_names, scorer=fuzz.WRatio, limit=top_k
    ):
        _update_best(matched_name, score, name2node)
    for matched_path, score, _ in process.extract(
        keyword, all_paths, scorer=fuzz.WRatio, limit=top_k
    ):
        _update_best(matched_path, score, path2node)

    # Scorer 3: description matching (weighted 0.8x)
    if all_descs:
        for matched_desc, score, _ in process.extract(
            keyword, all_descs, scorer=fuzz.token_set_ratio, limit=top_k
        ):
            weighted = score * 0.8
            for n in desc2node[matched_desc]:
                nid = n.id
                if nid not in node_best_score or weighted > node_best_score[nid][0]:
                    node_best_score[nid] = (weighted, n)

    results = [(node, sc) for sc, node in node_best_score.values()]
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


# ============================================================================
# Main entry point
# ============================================================================

def search_features_by_keywords(
    rpg: RPG,
    entity_searcher: RepoEntitySearcher,
    keywords: List[str],
    search_scopes: Optional[List[str]] = None,
    top_k: int = 5,
) -> Tuple[str, bool]:
    """Search RPG nodes by functional feature keywords.

    Uses a tiered strategy: exact match -> substring -> fuzzy (rapidfuzz).
    Results are ranked, merged, and formatted as a markdown string.

    Args:
        rpg: RPG instance (must have ``dep_graph``).
        entity_searcher: RepoEntitySearcher instance.
        keywords: List of search keywords.
        search_scopes: Feature paths to limit search scope.
        top_k: Number of top results per keyword.

    Returns:
        ``(formatted_result, success)`` tuple.

    Source: RPG-ZeroRepo search_node_by_feature.py ``search_features_by_keywords``
    """
    if not keywords:
        return "", False

    if search_scopes is None:
        search_scopes = []

    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    keywords = [normalize_text(k) for k in keywords]
    id2dep_node = (
        {nid: node for nid, node in dep_graph.nodes().items()}
        if dep_graph else {}
    )

    all_query_results: List[QueryResult] = []
    query_info_list: List[QueryInfo] = []

    # --- Resolve scope nodes ---
    valid_scopes_dict: Dict[str, Node] = {}
    if search_scopes:
        def add_subtree_nodes(node: Node):
            if node is None:
                return
            if node.id in valid_scopes_dict and node.meta.type_name == NodeType.DIRECTORY:
                return
            valid_scopes_dict[node.id] = node
            children = rpg.get_children(node.id, recursive=True)
            for child_id in children:
                child_node = rpg.get_node_by_id(child_id)
                if child_node:
                    add_subtree_nodes(child_node)

        for scope in search_scopes:
            node = rpg.get_node_by_feature_path(scope)
            if node:
                add_subtree_nodes(node)

    valid_scopes: List[Node] = list(valid_scopes_dict.values()) if valid_scopes_dict else []
    if not valid_scopes:
        valid_scopes = _get_default_valid_nodes(rpg)

    # Type-based scoring weight
    _TYPE_WEIGHT = {
        NodeType.METHOD: 1.15,
        NodeType.FUNCTION: 1.15,
        NodeType.CLASS: 1.05,
        NodeType.FILE: 0.85,
        NodeType.DIRECTORY: 0.70,
    }

    for keyword in keywords:
        selected: List[Tuple[Node, float, str]] = []

        # 1) Exact match
        exact_nodes = exact_match_search_feature(rpg, keyword, valid_scopes)
        if exact_nodes:
            selected = [(n, 100.0, "exact") for n in exact_nodes]

        # 2) Substring match
        if not selected:
            substr_nodes = substring_match_search_feature(rpg, keyword, valid_scopes, top_k)
            substr_nodes = [(n, s) for n, s in substr_nodes if s >= 55]
            selected = [(n, s, "substring") for n, s in substr_nodes]

        # 3) Fuzzy match
        if not selected:
            fuzzy_nodes = fuzzy_match_search_feature(rpg, keyword, valid_scopes, top_k)
            fuzzy_nodes = [(n, s) for n, s in fuzzy_nodes if s >= 60]
            selected = [(n, s, "fuzzy") for n, s in fuzzy_nodes]

        if not selected:
            continue

        # Apply type-based weight
        selected = [
            (
                n,
                s * _TYPE_WEIGHT.get(n.meta.type_name, 1.0) if n.meta else s,
                mt,
            )
            for n, s, mt in selected
        ]

        # Convert to QueryResult
        for node, score, match_type in selected:
            nid = node.meta.path
            dep_node = id2dep_node.get(nid)
            if not dep_node:
                continue

            node_data_list = entity_searcher.get_node_data([nid], return_code_content=True)
            if not node_data_list:
                continue
            node_data = node_data_list[0]

            if match_type == "exact":
                retrieve_src = f"`{node.name}` EXACTLY matches `{keyword}`."
                format_mode = "complete"
            elif match_type == "substring":
                retrieve_src = (
                    f"`{node.name}` contains `{keyword}` "
                    f"(match score {score:.1f}%)."
                )
                format_mode = "preview"
            else:
                retrieve_src = (
                    f"`{node.name}` loosely matches `{keyword}` "
                    f"(similarity {score:.1f}%)."
                )
                format_mode = "preview"

            if "start_line" not in node_data or "end_line" not in node_data:
                continue

            qinfo = QueryInfo(term=keyword)
            qr = QueryResult(
                query_info=qinfo,
                format_mode=format_mode,
                nid=nid,
                ntype=node.meta.type_name,
                start_line=node_data["start_line"],
                end_line=node_data["end_line"],
                retrieve_src=retrieve_src,
            )
            all_query_results.append(qr)
            query_info_list.append(qinfo)

    if not all_query_results:
        return "No matching features found.", False

    merged_results = merge_query_results(all_query_results)
    ranked = rank_and_aggr_query_results(merged_results, query_info_list)

    # === Format output ===
    result = ""
    all_suc: List[bool] = []
    for query_infos, format_to_results in ranked.items():
        term_desc = ", ".join([f'"{qi.term}"' for qi in query_infos])
        result += f"##Searching for term {term_desc}...\n"
        result += "### Search Result:\n"
        cur_result = ""

        for format_mode, query_results in format_to_results.items():
            if format_mode == "fold":
                cur_retrieve_src = ""
                for qr in query_results:
                    if not cur_retrieve_src:
                        cur_retrieve_src = qr.retrieve_src
                    if cur_retrieve_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_retrieve_src + "\n\n"
                        cur_retrieve_src = qr.retrieve_src
                    cur_result += qr.format_output(entity_searcher)
                cur_result += "Source: " + cur_retrieve_src + "\n"
                if len(query_results) > 1:
                    cur_result += "Hint: Use more detailed query to get the full content of some if needed.\n"
                else:
                    cur_result += f"Hint: Search `{query_results[0].nid}` for the full content if needed.\n"
                cur_result += "\n"

            elif format_mode == "complete":
                for qr in query_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += "\n"

            elif format_mode == "preview":
                filtered_results: List[QueryResult] = []
                grouped_by_file: Dict[str, List[QueryResult]] = defaultdict(list)
                for qr in query_results:
                    if qr.start_line is None or qr.end_line is None:
                        filtered_results.append(qr)
                        continue
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)

                for _fp, results in grouped_by_file.items():
                    sorted_results = sorted(
                        results, key=lambda q: (q.start_line or 0, -(q.end_line or 0))
                    )
                    max_end_line = -1
                    for qr in sorted_results:
                        if qr.end_line and qr.end_line > max_end_line:
                            filtered_results.append(qr)
                            max_end_line = max(max_end_line, qr.end_line)

                for qr in filtered_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += "\n"

            elif format_mode == "code_snippet":
                for qr in query_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += "\n"

        cur_result += "\n\n"
        if cur_result.strip():
            result += cur_result
            all_suc.append(True)
        else:
            result += "No locations found.\n\n"
            all_suc.append(False)

    suc = all(all_suc)
    return result.strip(), suc
