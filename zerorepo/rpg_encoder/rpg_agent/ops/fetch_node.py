import re
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import fnmatch
from copy import deepcopy

import numpy as np
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from networkx import MultiDiGraph

from zerorepo.rpg_encoder.rpg_agent.env.query import QueryInfo, QueryResult
from zerorepo.rpg_gen.base.rpg import RPG, Node, NodeMetaData, NodeType, EdgeType
from zerorepo.utils.repo import is_test_file, merge_intervals, normalize_text
from zerorepo.rpg_gen.base.node import RepoSkeleton
from ..env import RepoEntitySearcher, RepoDependencySearcher
from .search_node_by_meta import (
    merge_query_results,
    rank_and_aggr_query_results,
    fuzzy_retrieve,
)

# =========================
# Upper section: feature name search
# =========================

def _get_default_valid_nodes(rpg: RPG) -> List[Node]:
    """Get the default list of valid node types (FILE, CLASS, METHOD, FUNCTION)."""
    detailed_node_type = [
        NodeType.FILE,
        NodeType.CLASS,
        NodeType.METHOD,
        NodeType.FUNCTION,
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
    Rule-based substring search:
    1. Check whether the node name contains the keyword as a substring
    2. Check whether the feature_path contains the keyword as a substring
    Return a list of (Node, score), where the score is based on match strength.
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

        if score > 0:
            results.append((node, score))

    results.sort(key=lambda x: -x[1])
    return results[:max_results]


def exact_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: List[Node] = [],  # defaults to empty; if empty, search all nodes
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

    return [node for node in feature2node.get(norm_kw, [])]


def fuzzy_match_search_feature(
    rpg: RPG,
    keyword: str,
    valid_nodes: List[Node] = [],
    top_k: int = 5,
) -> List[Tuple[Node, float]]:
    """
    Perform fuzzy matching for a single keyword within `valid_nodes`.
    If `valid_nodes` is empty, search across all nodes in the RPG.
    Returns a list of (Node, similarity_score).
    (Keeps the rapidfuzz-based implementation as previously used.)
    """
    if not keyword or not isinstance(keyword, str):
        return []
    
    keyword = normalize_text(keyword)

    if not valid_nodes:
        valid_nodes = _get_default_valid_nodes(rpg)

    feature2node = defaultdict(list)
    for node in valid_nodes:
        feature2node[normalize_text(node.name)].append(node)

    all_feature_names = list(feature2node.keys())

    if not all_feature_names:
        return []

    matches = process.extract(
        keyword,
        all_feature_names,
        scorer=fuzz.token_set_ratio,
        limit=top_k,
    )

    results = []
    for matched_name, score, _ in matches:
        for node in feature2node[matched_name]:
            results.append((node, float(score)))

    return results


def search_features_by_keywords(
    rpg: RPG,
    entity_searcher: RepoEntitySearcher,
    keywords: List[str],
    search_scopes: List[str] = [],
    top_k: int = 5,
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
            """Recursively add a node and all of its descendants."""
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

    valid_scope_ids = {n.id for n in valid_scopes}

    for keyword in keywords:
        selected = []

        # 1) exact match first
        exact_nodes = exact_match_search_feature(rpg, keyword, valid_scopes)
        if exact_nodes:
            selected = [(node, 100.0, "exact") for node in exact_nodes]

        # 2) substring match
        if not selected:
            substr_nodes = substring_match_search_feature(rpg, keyword, valid_scopes, top_k)
            # Filter out low score matches (< 40%)
            substr_nodes = [(node, score) for node, score in substr_nodes if score >= 40]
            selected = [(node, score, "substring") for node, score in substr_nodes]

        # 3) rapidfuzz fallback
        if not selected:
            fuzzy_nodes = fuzzy_match_search_feature(rpg, keyword, valid_scopes, top_k)
            # Filter out low score matches (< 50%)
            fuzzy_nodes = [(node, score) for node, score in fuzzy_nodes if score >= 50]
            selected = [(node, score, "fuzzy") for node, score in fuzzy_nodes]

        # 4) TF-IDF fallback if still no good results
        if not selected:
            tfidf_results = _fuzzy_feature_paths(rpg, keyword, top_k)
            for fpath, score in tfidf_results:
                fnode = rpg.get_node_by_feature_path(fpath)
                if fnode:
                    # Check if node is in valid_scopes
                    if valid_scope_ids and fnode.id not in valid_scope_ids:
                        continue
                    selected.append((fnode, score, "tfidf"))

        # 5) Final fallback: TF-IDF without scope restriction if still no results
        if not selected:
            tfidf_results = _fuzzy_feature_paths(rpg, keyword, top_k)
            for fpath, score in tfidf_results:
                fnode = rpg.get_node_by_feature_path(fpath)
                if fnode:
                    selected.append((fnode, score, "tfidf_global"))

        if not selected:
            continue

        # 3) convert to QueryResult
        for node, score, match_type in selected:
            nid = node.meta.path
            dep_node = id2dep_node.get(nid)
            if not dep_node:
                continue

            if not entity_searcher.has_node(nid):
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
            elif match_type == "tfidf":
                retrieve_src = (
                    f"`{node.name}` matches `{keyword}` by TF-IDF "
                    f"(similarity {score:.1f}%)."
                )
                format_mode = "preview"
            else:
                retrieve_src = (
                    f"`{node.name}` loosely matches `{keyword}` "
                    f"(similarity {score:.1f}%)."
                )
                format_mode = "preview"

            qinfo = QueryInfo(term=keyword)
            qr = QueryResult(
                query_info=qinfo,
                format_mode=format_mode,
                nid=nid,
                ntype=node.meta.type_name,
                start_line=node_data.get("start_line"),
                end_line=node_data.get("end_line"),
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
    all_suc = []
    for query_infos, format_to_results in ranked.items():
        term_desc = ", ".join([f'"{query.term}"' for query in query_infos])
        result += f"##Searching for term {term_desc}...\n"
        result += f"### Search Result:\n"
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
                    cur_result += (
                        "Hint: Use more detailed query to get the full content of some if needed.\n"
                    )
                else:
                    cur_result += (
                        f"Hint: Search `{query_results[0].nid}` for the full content if needed.\n"
                    )
                cur_result += "\n"

            elif format_mode == "complete":
                for qr in query_results:
                    cur_result += qr.format_output(entity_searcher)
                    cur_result += "\n"

            elif format_mode == "preview":
                # Remove the small modules, leaving only the large ones
                filtered_results = []
                grouped_by_file = defaultdict(list)
                for qr in query_results:
                    # Skip if start_line or end_line is None
                    if qr.start_line is None or qr.end_line is None:
                        filtered_results.append(qr)
                        continue
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)

                for file_path, results in grouped_by_file.items():
                    # Sort by start_line and then by end_line in descending order
                    sorted_results = sorted(
                        results, key=lambda qr: (qr.start_line or 0, -(qr.end_line or 0))
                    )

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
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
            suc = True
        else:
            result += "No locations found.\n\n"
            suc = False
        all_suc.append(suc)

    suc = all(all_suc)

    return result.strip(), suc


# =========================
# retrieve_entity + TF-IDF feature fuzzy
# =========================

def _safe(val, default="-"):
    return val if val is not None else default


def _collect_all_feature_paths(rpg: RPG) -> List[str]:
    paths: List[str] = []
    for node in rpg.nodes.values():
        fp = None
        try:
            fp = node.feature_path()
        except TypeError:
            if hasattr(node, "feature_path") and isinstance(node.feature_path, str):
                fp = node.feature_path
        if fp:
            paths.append(fp)
    return sorted(set(paths))


def _get_deps_from_feature_id(feature_id: str, rpg: RPG) -> List[str]:
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}
    dep_ids: List[str] = []
    for dep_id, rpg_ids in dep2rpg.items():
        ids = rpg_ids if isinstance(rpg_ids, list) else [rpg_ids]
        if feature_id in ids:
            dep_ids.append(dep_id)
    return dep_ids


# ==== TF-IDF ====

_FEATURE_TFIDF_CACHE: Dict[int, Dict[str, object]] = {}


def _build_feature_tfidf_index(rpg: RPG) -> Dict[str, object]:
    key = id(rpg)
    if key in _FEATURE_TFIDF_CACHE:
        return _FEATURE_TFIDF_CACHE[key]

    all_paths = _collect_all_feature_paths(rpg)
    if not all_paths:
        index = {
            "vectorizer": None,
            "tfidf": None,
            "paths": [],
        }
        _FEATURE_TFIDF_CACHE[key] = index
        return index

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    tfidf_matrix = vectorizer.fit_transform(all_paths)

    index = {
        "vectorizer": vectorizer,
        "tfidf": tfidf_matrix,
        "paths": all_paths,
    }
    _FEATURE_TFIDF_CACHE[key] = index
    return index


def _fuzzy_feature_paths(rpg: RPG, keyword: str, top_k: int = 5) -> List[Tuple[str, float]]:

    keyword = normalize_text(keyword or "")
    if not keyword:
        return []

    index = _build_feature_tfidf_index(rpg)
    vectorizer: Optional[TfidfVectorizer] = index["vectorizer"]  # type: ignore
    tfidf_matrix = index["tfidf"]
    paths: List[str] = index["paths"]  # type: ignore

    if not paths or vectorizer is None or tfidf_matrix is None:
        return []

    query_vec = vectorizer.transform([keyword])
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()  # (N,)

    if top_k is None or top_k <= 0 or top_k > len(paths):
        top_k = len(paths)

    top_indices = np.argsort(sims)[::-1][:top_k]

    results: List[Tuple[str, float]] = []
    for idx in top_indices:
        s = sims[idx]
        if s <= 0:
            continue

        score = float(s * 100.0)
        results.append((paths[idx], score))

    return results


def fetch_node(
    rpg: RPG,
    entity_searcher: RepoEntitySearcher,
    code_entities: List[str] = [],
    feature_entities: List[str] = [],
    similarity_top_k: int = 5,
) -> Tuple[str, bool]:
    """
    Return search result string (markdown) and success flag.

    Args:
        rpg: RPG instance (contains dep_graph and _dep_to_rpg_map)
        entity_searcher: RepoEntitySearcher instance
        code_entities: List of code entity IDs to retrieve
        feature_entities: List of feature paths to retrieve
        similarity_top_k: Number of top fuzzy matches


    """
    # Get dep_graph from RPG
    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None

    results: List[QueryResult] = []

    # ===================== Code Entities =====================
    if code_entities:
        for nid in code_entities:
            nid = (nid or "").strip()

            if nid.endswith(".__init__"):
                nid = nid[: -(len(".__init__"))]

            if not nid:
                continue

            # Exact match
            if entity_searcher.has_node(nid):
                ndata_list = entity_searcher.get_node_data([nid], return_code_content=True)
                if not ndata_list:
                    continue
                ndata = ndata_list[0]

                qr = QueryResult(
                    query_info=QueryInfo(term=nid),
                    format_mode="complete",
                    nid=nid,
                    ntype=ndata.get("type"),
                    start_line=ndata.get("start_line"),
                    end_line=ndata.get("end_line"),
                    retrieve_src="Exact entity match in repo graph",
                )
                results.append(qr)
                continue

            # Fuzzy match candidates
            try:
                matches = fuzzy_retrieve(
                    keyword=nid,
                    rpg=rpg,
                    search_scope="all",
                    include_files=None,
                    similarity_top_k=similarity_top_k,
                    return_score=True,
                )
            except Exception:
                all_nids = list(dep_graph.nodes()) if dep_graph else []
                matches = process.extract(
                    nid, all_nids, scorer=fuzz.token_set_ratio, limit=similarity_top_k
                )

            # Normalize matches to (candidate, score) and append previews
            if matches:
                normalized: List[Tuple[str, float]] = []
                for m in matches:
                    # Support 2-tuple, 3-tuple, or ExtractResult-like objects
                    try:
                        cand, score, *_ = m  # works for rapidfuzz (choice, score, index)
                    except Exception:
                        cand = m[0]
                        score = m[1] if len(m) > 1 else 0
                    normalized.append((str(cand), float(score)))

                for cand, score in normalized:
                    if not entity_searcher.has_node(cand):
                        continue

                    ndata_list = entity_searcher.get_node_data([cand], return_code_content=True)
                    if not ndata_list:
                        continue
                    ndata = ndata_list[0]

                    qr = QueryResult(
                        query_info=QueryInfo(term=nid),
                        format_mode="preview",
                        nid=cand,
                        ntype=ndata.get("type"),
                        start_line=ndata.get("start_line"),
                        end_line=ndata.get("end_line"),
                        retrieve_src=f"Fuzzy match (score {score:.1f}) — refine query to get full code",
                    )
                    results.append(qr)

    # ===================== Feature Entities =====================
    if feature_entities:
        for fpath in feature_entities:
            raw = (fpath or "").strip()
            if not raw:
                continue
            if raw.startswith("/"):
                raw = raw[1:]

            # exact feature path → node → dep_id(s)
            fnode: Optional[Node] = rpg.get_node_by_feature_path(raw)

            if fnode:
                mapped_dep_ids = _get_deps_from_feature_id(fnode.id, rpg)

                for did in mapped_dep_ids:
                    if not entity_searcher.has_node(did):
                        continue

                    ndata_list = entity_searcher.get_node_data([did], return_code_content=True)
                    if not ndata_list:
                        continue
                    ndata = ndata_list[0]

                    qr = QueryResult(
                        query_info=QueryInfo(term=raw),
                        format_mode="complete",
                        nid=did,
                        ntype=ndata.get("type"),
                        start_line=ndata.get("start_line"),
                        end_line=ndata.get("end_line"),
                        retrieve_src="Feature → code mapping",
                    )
                    results.append(qr)
            else:
                # ❓ fuzzy feature match (TF-IDF)
                cands = _fuzzy_feature_paths(rpg, raw, top_k=similarity_top_k)

                for cand_path, score in cands:
    
                    cand_node = rpg.get_node_by_feature_path(cand_path)
                    if not cand_node:
                        continue

                    dep_ids = _get_deps_from_feature_id(cand_node.id, rpg)
                    if not dep_ids:
                        continue

                    for did in dep_ids:
                        if not entity_searcher.has_node(did):
                            continue

                        ndata_list = entity_searcher.get_node_data(
                            [did], return_code_content=True
                        )
                        if not ndata_list:
                            continue
                        ndata = ndata_list[0]

                        qr = QueryResult(
                            query_info=QueryInfo(term=raw),
                            format_mode="preview",
                            nid=did,
                            ntype=ndata.get("type"),
                            start_line=ndata.get("start_line"),
                            end_line=ndata.get("end_line"),
                            retrieve_src=(
                                f"Fuzzy feature match `{cand_path}` by TF-IDF "
                                f"(similarity {score:.1f}%)"
                            ),
                        )
                        results.append(qr)

    # ===================== Format Output =====================

    if not results:
        return "No entities provided or no match found.", False

    searcher = entity_searcher
    out = "## Search Results\n\n"
    last_src = None

    for qr in results:
        text = qr.format_output(searcher)

        # group same source messages together
        if qr.retrieve_src != last_src:
            out += f"### Source: {qr.retrieve_src}\n"
            last_src = qr.retrieve_src

        out += text + "\n\n"

    return out.strip(), True