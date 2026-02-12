import re
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import fnmatch
from rapidfuzz import process, fuzz
from copy import deepcopy
from networkx import MultiDiGraph
from zerorepo.rpg_encoder.rpg_agent.env.query import QueryInfo, QueryResult
from zerorepo.rpg_gen.base.rpg import RPG, Node, NodeType, EdgeType
from zerorepo.utils.repo import is_test_file, merge_intervals, normalize_text
from zerorepo.rpg_gen.base.node import RepoSkeleton
from .search_node_by_meta import merge_query_results, rank_and_aggr_query_results
from ..env import RepoEntitySearcher, RepoDependencySearcher

def _get_default_valid_nodes(rpg: RPG) -> List[Node]:
    detailed_node_type = [
        NodeType.FILE, NodeType.CLASS,
        NodeType.METHOD, NodeType.FUNCTION
    ]
    valid_nodes = []
    for node_type in detailed_node_type:
        valid_nodes.extend(rpg.get_nodes_by_type(type_name=node_type))
    return valid_nodes

def substring_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: List[Node] = [],
    max_results: int = 10,
) -> List[Tuple[Node, float]]:
    """
    基于规则的子串匹配搜索：
    1. 检查节点名称是否包含 keyword 作为子串（正向）
    2. 检查 feature_path 是否包含 keyword 作为子串（正向）
    3. 检查 keyword 是否包含节点名称/路径中的 token 作为子串（反向）
    返回 (Node, score) 列表，score 基于匹配程度
    """
    if not keyword or not isinstance(keyword, str):
        return []

    norm_kw = normalize_text(keyword)
    kw_parts = re.split(r'[\s._]+', norm_kw)
    kw_parts = [p for p in kw_parts if p and len(p) > 1]

    if not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    results = []
    for node in valid_nodes:
        norm_name = normalize_text(node.name)
        norm_path = normalize_text(node.feature_path())

        score = 0.0

        # 完整关键词匹配（正向）
        if norm_kw in norm_name:
            score = 90.0 if norm_kw == norm_name else 80.0
        elif norm_kw in norm_path:
            score = 70.0

        if score == 0 and kw_parts:
            matched_parts = sum(1 for p in kw_parts if p in norm_name or p in norm_path)
            if matched_parts == len(kw_parts):
                score = 60.0 + (matched_parts / len(kw_parts)) * 10
            elif matched_parts > 0:
                score = 40.0 + (matched_parts / len(kw_parts)) * 20

        if score == 0:
            name_tokens = re.split(r'[\s._/]+', norm_path)
            name_tokens = [t for t in name_tokens if t and len(t) >= 3]
            if name_tokens:
                reverse_matched = sum(1 for t in name_tokens if t in norm_kw)
                if reverse_matched > 0:
                    coverage = sum(len(t) for t in name_tokens if t in norm_kw) / max(len(norm_kw), 1)
                    score = 30.0 + min(coverage, 1.0) * 25.0

        if score > 0:
            results.append((node, score))

    results.sort(key=lambda x: -x[1])
    return results[:max_results]


def exact_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: List[Node] = []  # defaults to empty; if empty, search all nodes
) -> List[Node]:
    """
    Find nodes in the RPG that exactly match (after normalization) the keyword,
    searching only within `valid_nodes`.
    If `valid_nodes` is empty, search across all nodes in the RPG.
    """
    if not keyword or not isinstance(keyword, str):
        return []

    norm_kw = normalize_text(keyword)

    if not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    feature2node: Dict[str, List[Node]] = defaultdict(list)
    for node in valid_nodes:
        feature2node[normalize_text(node.name)].append(node)

    return [
        node for node in feature2node.get(norm_kw, [])
    ]


def fuzzy_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: List[Node] = [],
    top_k: int = 5,
) -> List[Tuple[Node, float]]:
    """
    Perform fuzzy matching for a single keyword within `valid_nodes`.
    If `valid_nodes` is empty, search across all nodes in the RPG and return a list of (Node, similarity_score).
    Match against node.name, node.feature_path(), and node.meta.description, taking the highest score.
    Use both token_set_ratio and WRatio as scorers.
    """
    if not keyword or not isinstance(keyword, str):
        return []

    keyword = normalize_text(keyword)

    if not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    name2node = defaultdict(list)
    path2node = defaultdict(list)
    desc2node = defaultdict(list)
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

    def _update_best(matched_key, score, key2node_map):
        for node in key2node_map[matched_key]:
            nid = node.id
            if nid not in node_best_score or score > node_best_score[nid][0]:
                node_best_score[nid] = (score, node)

    # Scorer 1: token_set_ratio on name and path
    name_matches = process.extract(
        keyword, all_names, scorer=fuzz.token_set_ratio, limit=top_k
    )
    path_matches = process.extract(
        keyword, all_paths, scorer=fuzz.token_set_ratio, limit=top_k
    )

    for matched_name, score, _ in name_matches:
        _update_best(matched_name, score, name2node)
    for matched_path, score, _ in path_matches:
        _update_best(matched_path, score, path2node)

    # Scorer 2: WRatio (weighted ratio — considers ordering) on name and path
    wratio_name_matches = process.extract(
        keyword, all_names, scorer=fuzz.WRatio, limit=top_k
    )
    wratio_path_matches = process.extract(
        keyword, all_paths, scorer=fuzz.WRatio, limit=top_k
    )

    for matched_name, score, _ in wratio_name_matches:
        _update_best(matched_name, score, name2node)
    for matched_path, score, _ in wratio_path_matches:
        _update_best(matched_path, score, path2node)

    # Scorer 3: description matching (weighted at 0.8x)
    if all_descs:
        desc_matches = process.extract(
            keyword, all_descs, scorer=fuzz.token_set_ratio, limit=top_k
        )
        for matched_desc, score, _ in desc_matches:
            weighted = score * 0.8
            for node in desc2node[matched_desc]:
                nid = node.id
                if nid not in node_best_score or weighted > node_best_score[nid][0]:
                    node_best_score[nid] = (weighted, node)

    results = [(node, score) for score, node in node_best_score.values()]
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def search_features_by_keywords(
    rpg: RPG,
    entity_searcher: RepoEntitySearcher,
    keywords: List[str],
    search_scopes: List[str] = [],
    top_k: int = 5
):
    """
    Search features by keywords.

    Args:
        rpg: RPG instance (contains dep_graph)
        entity_searcher: RepoEntitySearcher instance
        keywords: List of search keywords
        search_scopes: Feature paths to limit search scope
        top_k: Number of top results
    """
    if not keywords:
        return "", False

    # Get dep_graph from RPG
    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None

    keywords = [normalize_text(k) for k in keywords]
    id2dep_node = {nid: node for nid, node in dep_graph.nodes().items()} if dep_graph else {}

    all_query_results = []
    query_info_list = []

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
        detailed_node_type = [
            NodeType.FILE, NodeType.CLASS,
            NodeType.METHOD, NodeType.FUNCTION
        ]
        valid_scopes = []
        for node_type in detailed_node_type:
            valid_scopes.extend(rpg.get_nodes_by_type(type_name=node_type))

    # Type-based weight: prefer leaf nodes (METHOD/FUNCTION), penalize broad nodes (FILE/DIRECTORY)
    _TYPE_WEIGHT = {
        NodeType.METHOD: 1.15,
        NodeType.FUNCTION: 1.15,
        NodeType.CLASS: 1.05,
        NodeType.FILE: 0.85,
        NodeType.DIRECTORY: 0.70,
    }

    for keyword in keywords:
        selected = []

        # 1) exact match first
        exact_nodes = exact_match_search_feature(rpg, keyword, valid_scopes)
        if exact_nodes:
            selected = [(node, 100.0, "exact") for node in exact_nodes]

        # 2) substring match — threshold raised from 40 to 55
        if not selected:
            substr_nodes = substring_match_search_feature(rpg, keyword, valid_scopes, top_k)
            substr_nodes = [(node, score) for node, score in substr_nodes if score >= 55]
            selected = [(node, score, "substring") for node, score in substr_nodes]

        # 3) rapidfuzz fallback — threshold raised from 50 to 60
        if not selected:
            fuzzy_nodes = fuzzy_match_search_feature(rpg, keyword, valid_scopes, top_k)
            fuzzy_nodes = [(node, score) for node, score in fuzzy_nodes if score >= 60]
            selected = [(node, score, "fuzzy") for node, score in fuzzy_nodes]

        if not selected:
            continue

        # Apply type-based weighting to scores
        selected = [
            (node, score * _TYPE_WEIGHT.get(node.meta.type_name, 1.0) if node.meta else score, match_type)
            for node, score, match_type in selected
        ]

        # 4) convert to QueryResult
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
                retrieve_src=retrieve_src
            )

            all_query_results.append(qr)
            query_info_list.append(qinfo)

    if not all_query_results:
        return "No matching features found.", False

    merged_results = merge_query_results(all_query_results)
    ranked = rank_and_aggr_query_results(merged_results, query_info_list)

    # === Format output ===
    result = ""
    all_suc = []
    for query_infos, format_to_results in ranked.items():
        term_desc = ', '.join([f'"{query.term}"' for query in query_infos])
        result += f'##Searching for term {term_desc}...\n'
        result += f'### Search Result:\n'
        cur_result = ''
        
        for format_mode, query_results in format_to_results.items():
            if format_mode == 'fold':
                cur_retrieve_src = ''
                for qr in query_results:
                    if not cur_retrieve_src:
                        cur_retrieve_src = qr.retrieve_src
                        
                    if cur_retrieve_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_retrieve_src + '\n\n'
                        cur_retrieve_src = qr.retrieve_src
                        
                    cur_result += qr.format_output(entity_searcher)
                    
                cur_result += "Source: " + cur_retrieve_src + '\n'
                if len(query_results) > 1:
                    cur_result += 'Hint: Use more detailed query to get the full content of some if needed.\n'
                else:
                    cur_result += f'Hint: Search `{query_results[0].nid}` for the full content if needed.\n'
                cur_result += '\n'
                
            elif format_mode == 'complete':
                for qr in query_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += '\n'

            elif format_mode == 'preview':
                # Remove the small modules, leaving only the large ones
                filtered_results = []
                grouped_by_file = defaultdict(list)
                for qr in query_results:
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)
                
                for file_path, results in grouped_by_file.items():
                    # Sort by start_line and then by end_line in descending order
                    sorted_results = sorted(results, key=lambda qr: (qr.start_line, -qr.end_line))

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
                        if qr.end_line > max_end_line:
                            filtered_results.append(qr)
                            max_end_line = max(max_end_line, qr.end_line)
                
                # filtered_results = query_results
                for qr in filtered_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += '\n'
            
            elif format_mode == 'code_snippet':
                for qr in query_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += '\n'
            
        cur_result += '\n\n'
        
        if cur_result.strip():
            result += cur_result
            suc = True
        else:
            result += 'No locations found.\n\n'
            suc = False
        all_suc.append(suc)
    
    suc = all(all_suc)
    
    return result.strip(), suc
