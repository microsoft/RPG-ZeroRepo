#!/usr/bin/env python3
"""Search RPG nodes by metadata (entity names, code paths, BM25, fuzzy).

This module provides the 'code entity' search path — given entity names like
``src/auth/login.py:LoginManager`` or short names like ``LoginManager``, it
tries exact match, global-name-dict lookup, BM25, and fuzzy retrieve in
cascading order.

It also provides shared helpers ``merge_query_results`` and
``rank_and_aggr_query_results`` used by both meta and feature search.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/ops/search_node_by_meta.py
"""

import logging
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from common.utils import is_test_file, merge_intervals, normalize_text
from rpg_agent.env.query import QueryInfo, QueryResult
from rpg_agent.env.searcher import RepoEntitySearcher, RepoDependencySearcher
from rpg_agent.ops.bm25_model import build_module_retriever
from rpg import EdgeType, NodeType, RPG

logger = logging.getLogger(__name__)


# ============================================================================
# BM25 module-level retrieval
# ============================================================================

def bm25_module_retrieve(
    query: str,
    entity_searcher: RepoEntitySearcher,
    include_files: Optional[List[str]] = None,
    search_scope: str = "all",
    similarity_top_k: int = 10,
) -> List[str]:
    """Use BM25 to find entity node IDs most relevant to *query*.

    Source: RPG-ZeroRepo search_node_by_meta.py ``bm25_module_retrieve``
    """
    retriever = build_module_retriever(
        entity_searcher=entity_searcher,
        search_scope=search_scope,
        similarity_top_k=similarity_top_k,
    )
    try:
        results = retriever.retrieve(query)
    except (IndexError, Exception) as e:
        logger.warning("BM25 retrieve error for '%s': %s", query, e)
        return []

    filter_nodes: List[str] = []
    all_nodes: List[str] = []
    for nid, score in results:
        if score <= 0:
            continue
        if not include_files or nid.split(":")[0] in include_files:
            filter_nodes.append(nid)
        all_nodes.append(nid)

    return filter_nodes if filter_nodes else all_nodes


# ============================================================================
# Fuzzy retrieval
# ============================================================================

def fuzzy_retrieve(
    keyword: str,
    rpg: Optional[RPG] = None,
    search_scope: str = "all",
    include_files: Optional[str] = None,
    similarity_top_k: int = 5,
    return_score: bool = False,
):
    """Fuzzy retrieve entities by keyword using rapidfuzz.

    Source: RPG-ZeroRepo search_node_by_meta.py ``fuzzy_retrieve``
    """
    graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    if graph is None:
        return []

    selected_nids: List[str] = []
    filter_nids: List[str] = []

    for nid in graph:
        ndata = graph.nodes[nid]
        if search_scope == "all" and ndata["type"] in [
            NodeType.FILE, NodeType.CLASS, NodeType.METHOD, NodeType.FUNCTION,
        ]:
            nfile = nid.split(":")[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)
        elif ndata["type"] == search_scope:
            nfile = nid.split(":")[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)

    if not filter_nids:
        filter_nids = selected_nids

    if not filter_nids:
        return []

    def custom_tokenizer(s: str) -> str:
        return " ".join(re.findall(r"\b\w+\b", s.replace("_", " ").replace("-", " ")))

    matches = process.extract(
        keyword,
        filter_nids,
        scorer=fuzz.token_set_ratio,
        processor=custom_tokenizer,
        limit=similarity_top_k,
    )

    if not return_score:
        return [match[0] for match in matches]
    return matches


# ============================================================================
# Global name dict search
# ============================================================================

def parse_node_id(nid: str) -> Tuple[str, str]:
    nfile = nid.split(":")[0]
    nname = nid.split(":")[-1]
    return nfile, nname


def search_entity_in_global_dict(
    searcher: RepoEntitySearcher,
    query: str,
    include_files: Optional[List[str]] = None,
    prefix_term=None,
) -> Optional[Dict[str, List[str]]]:
    """Look up *query* in the searcher's global name dict.

    Source: RPG-ZeroRepo search_node_by_meta.py ``search_entity_in_global_dict``
    """
    # Strip type prefixes
    for prefix in ("class ", "Class ", "function ", "Function ", "method ", "Method ", "def "):
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
            break

    # Try exact in global_name_dict
    if query in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[query]
    elif query.lower() in searcher.global_name_dict_lowercase:
        query = query.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[query]
    elif query.endswith(".py"):
        if query in searcher.global_name_dict:
            global_name_dict = searcher.global_name_dict
            nids = global_name_dict[query]
        elif query.lower() in searcher.global_name_dict_lowercase:
            query = query.lower()
            global_name_dict = searcher.global_name_dict_lowercase
            nids = global_name_dict[query]
        else:
            return None
    else:
        return None

    node_datas = searcher.get_node_data(nids, return_code_content=False)
    found_entities_filter_dict: Dict[str, List[str]] = defaultdict(list)

    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata["node_id"])
        if not include_files or nfile in include_files:
            candidite_prefixes = re.split(
                r"[./:]", ndata["node_id"].lower().replace(".py", "")
            )[:-1]
            prefix_terms = (
                prefix_term.lower().split(".") if prefix_term else []
            )
            if not prefix_term or all(p in candidite_prefixes for p in prefix_terms):
                found_entities_filter_dict[ndata["type"]].append(ndata["node_id"])

    return found_entities_filter_dict if found_entities_filter_dict else None


# ============================================================================
# Entity search (cascading: exact -> global_name_dict -> BM25 -> fuzzy)
# ============================================================================

def search_entity(
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    include_files: Optional[List[str]] = None,
) -> Tuple[List[QueryResult], bool]:
    """Cascading search for a code entity.

    Source: RPG-ZeroRepo search_node_by_meta.py ``search_entity``
    """
    query = query_info.term
    continue_search = True
    cur_query_results: List[QueryResult] = []

    # 1) Exact node ID match
    if entity_searcher.has_node(query):
        continue_search = False
        qr = QueryResult(
            query_info=query_info,
            format_mode="complete",
            nid=query,
            retrieve_src=f"Exact match found for entity name `{query}`.",
        )
        cur_query_results.append(qr)

    elif query.endswith(".__init__"):
        nid = query[: -len(".__init__")]
        if entity_searcher.has_node(nid):
            continue_search = False
            node_data = entity_searcher.get_node_data([nid], return_code_content=True)[0]
            qr = QueryResult(
                query_info=query_info,
                format_mode="preview",
                nid=nid,
                ntype=node_data["type"],
                start_line=node_data.get("start_line"),
                end_line=node_data.get("end_line"),
                retrieve_src=f"Exact match found for entity name `{nid}`.",
            )
            cur_query_results.append(qr)

    # 2) Global name dict
    if continue_search:
        found_dict = search_entity_in_global_dict(entity_searcher, query, include_files)
        if not found_dict:
            found_dict = search_entity_in_global_dict(entity_searcher, query)

        use_sub_term = False
        used_term = query

        if not found_dict and "." in query:
            try:
                prefix_term = ".".join(query.split(".")[:-1]).split()[-1]
            except IndexError:
                prefix_term = None
            split_term = query.split(".")[-1].strip()
            used_term = split_term
            found_dict = search_entity_in_global_dict(
                entity_searcher, split_term, include_files, prefix_term
            )
            if not found_dict:
                found_dict = search_entity_in_global_dict(
                    entity_searcher, split_term, prefix_term
                )
            if not found_dict:
                use_sub_term = True
                found_dict = search_entity_in_global_dict(
                    entity_searcher, split_term
                )

        if found_dict:
            max_fold_results = 5
            for ntype, nids in found_dict.items():
                if not nids:
                    continue
                if ntype in [NodeType.FUNCTION, NodeType.CLASS, NodeType.FILE, NodeType.METHOD]:
                    if len(nids) <= 3:
                        node_datas = entity_searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            qr = QueryResult(
                                query_info=query_info,
                                format_mode="preview",
                                nid=ndata["node_id"],
                                ntype=ndata["type"],
                                start_line=ndata.get("start_line"),
                                end_line=ndata.get("end_line"),
                                retrieve_src=f"Match found for entity name `{used_term}`.",
                            )
                            cur_query_results.append(qr)
                    else:
                        limited_nids = nids[:max_fold_results]
                        node_datas = entity_searcher.get_node_data(limited_nids, return_code_content=False)
                        for ndata in node_datas:
                            qr = QueryResult(
                                query_info=query_info,
                                format_mode="fold",
                                nid=ndata["node_id"],
                                ntype=ndata["type"],
                                retrieve_src=f"Match found for entity name `{used_term}`.",
                            )
                            cur_query_results.append(qr)

                    if not use_sub_term:
                        continue_search = False
                    else:
                        continue_search = True

    # 3) BM25 + fuzzy
    if continue_search:
        module_nids = bm25_module_retrieve(
            query=query, entity_searcher=entity_searcher, include_files=include_files
        )
        if not module_nids:
            module_nids = bm25_module_retrieve(
                query=query, entity_searcher=entity_searcher
            )
        if not module_nids:
            module_nids = fuzzy_retrieve(query, rpg=entity_searcher.rpg, similarity_top_k=3)

        module_datas = entity_searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        showed_file_num = 0
        max_file_results = 3
        max_module_results = 3

        for module in module_datas[:5]:
            if module["type"] in [NodeType.FILE, NodeType.DIRECTORY]:
                if showed_file_num < max_file_results:
                    showed_file_num += 1
                    qr = QueryResult(
                        query_info=query_info,
                        format_mode="fold",
                        nid=module["node_id"],
                        ntype=module["type"],
                        retrieve_src="Retrieved entity using keyword search (bm25).",
                    )
                    cur_query_results.append(qr)
            elif showed_module_num < max_module_results:
                showed_module_num += 1
                qr = QueryResult(
                    query_info=query_info,
                    format_mode="preview",
                    nid=module["node_id"],
                    ntype=module["type"],
                    start_line=module.get("start_line"),
                    end_line=module.get("end_line"),
                    retrieve_src="Retrieved entity using keyword search (bm25).",
                )
                cur_query_results.append(qr)

    return (cur_query_results, continue_search)


# ============================================================================
# Helpers for code-block retrieval by line numbers
# ============================================================================

def get_module_name_by_line_num(
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    file_path: str,
    line_num: int,
) -> Optional[Dict]:
    """Find the module (function/class) containing *line_num* in *file_path*.

    Source: RPG-ZeroRepo search_node_by_meta.py ``get_module_name_by_line_num``
    """
    cur_module = None
    if entity_searcher.has_node(file_path):
        module_nids, _ = dep_searcher.get_neighbors(
            file_path, etype_filter=[EdgeType.CONTAINS]
        )
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            sl = module.get("start_line", 0)
            el = module.get("end_line", 0)
            if sl <= line_num <= el:
                cur_module = module
                break
        if cur_module and cur_module["type"] == NodeType.CLASS:
            func_nids, _ = dep_searcher.get_neighbors(
                cur_module["node_id"], etype_filter=[EdgeType.CONTAINS]
            )
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=True)
            for func in func_ndatas:
                if func.get("start_line", 0) <= line_num <= func.get("end_line", 0):
                    cur_module = func
                    break
    return cur_module


def get_code_block_by_line_nums(
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    context_window: int = 20,
) -> List[QueryResult]:
    """Return QueryResult objects for the code blocks around *line_nums*.

    Source: RPG-ZeroRepo search_node_by_meta.py ``get_code_block_by_line_nums``
    """
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums or []
    cur_query_results: List[QueryResult] = []

    file_data_list = entity_searcher.get_node_data([file_path], return_code_content=False)
    if not file_data_list:
        return cur_query_results
    file_data = file_data_list[0]

    line_intervals: List[Tuple[int, int]] = []
    res_modules: List[str] = []

    for line in line_nums:
        module_data = get_module_name_by_line_num(
            entity_searcher, dep_searcher, file_path, line
        )
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data.get("end_line", line + context_window), line + context_window)
            line_intervals.append((min_line_num, max_line_num))
        elif module_data["node_id"] not in res_modules:
            qr = QueryResult(
                query_info=query_info,
                format_mode="preview",
                nid=module_data["node_id"],
                ntype=module_data["type"],
                start_line=module_data["start_line"],
                end_line=module_data["end_line"],
                retrieve_src=f"Retrieved code context including {query_info.term}.",
            )
            cur_query_results.append(qr)
            res_modules.append(module_data["node_id"])

    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for start_line, end_line in line_intervals:
            qr = QueryResult(
                query_info=query_info,
                format_mode="code_snippet",
                nid=file_path,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                retrieve_src=f"Retrieved code context including {query_info.term}.",
            )
            cur_query_results.append(qr)

    return cur_query_results


# ============================================================================
# Entity content retrieval
# ============================================================================

def get_entity_contents(
    entity_searcher: RepoEntitySearcher,
    entity_names: List[str],
) -> str:
    """Retrieve full content for named entities.

    Source: RPG-ZeroRepo search_node_by_meta.py ``get_entity_contents``
    """
    result = ""
    for name in entity_names:
        name = name.strip().strip(".")
        if not name:
            continue

        result += f"##Searching for entity `{name}`...\n"
        result += "### Search Result:\n"
        query_info = QueryInfo(term=name)

        if entity_searcher.has_node(name):
            qr = QueryResult(
                query_info=query_info,
                format_mode="complete",
                nid=name,
                retrieve_src=f"Exact match found for entity name `{name}`.",
            )
            result += qr.format_output(entity_searcher)
            result += "\n\n"
        else:
            result += (
                "Invalid name.\n"
                'Hint: Valid entity name should be formatted as '
                '"file_path:QualifiedName" or just "file_path".'
            )
            result += "\n\n"
    return result.strip()


# ============================================================================
# File pattern matching
# ============================================================================

def find_matching_files_from_list(
    file_list: List[str],
    file_pattern: str,
) -> List[str]:
    """Find files matching *file_pattern* (glob or keyword) in *file_list*.

    Source: RPG-ZeroRepo search_node_by_meta.py ``find_matching_files_from_list``
    """
    if "*" in file_pattern or "?" in file_pattern or "[" in file_pattern:
        return fnmatch.filter(file_list, file_pattern)
    return [f for f in file_list if file_pattern in f]


# ============================================================================
# Merge / rank query results
# ============================================================================

def merge_query_results(query_results: List[QueryResult]) -> List[QueryResult]:
    """De-duplicate and merge query results by node ID.

    Source: RPG-ZeroRepo search_node_by_meta.py ``merge_query_results``
    """
    priority = ["complete", "code_snippet", "preview", "fold"]
    merged: Dict[str, QueryResult] = {}
    all_results: List[QueryResult] = []

    for qr in query_results:
        if qr.format_mode == "code_snippet":
            all_results.append(qr)
        elif qr.nid and qr.nid in merged:
            # Merge query_info_list
            if qr.query_info_list[0] not in merged[qr.nid].query_info_list:
                merged[qr.nid].query_info_list.extend(qr.query_info_list)
            # Prefer higher-priority format_mode
            existing_mode = merged[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_mode):
                merged[qr.nid].format_mode = qr.format_mode
                merged[qr.nid].start_line = qr.start_line
                merged[qr.nid].end_line = qr.end_line
                merged[qr.nid].retrieve_src = qr.retrieve_src
        elif qr.nid:
            merged[qr.nid] = qr

    all_results += list(merged.values())
    return all_results


def rank_and_aggr_query_results(
    query_results: List[QueryResult],
    fixed_query_info_list: List[QueryInfo],
) -> Dict:
    """Group and rank query results by query_info_list and format_mode.

    Returns a dict: ``{ (QueryInfo, ...) : { format_mode : [QueryResult, ...] } }``

    Source: RPG-ZeroRepo search_node_by_meta.py ``rank_and_aggr_query_results``
    """
    query_info_list_dict: Dict[tuple, List[QueryResult]] = {}
    for qr in query_results:
        key = tuple(qr.query_info_list)
        query_info_list_dict.setdefault(key, []).append(qr)

    # Sort keys by their first appearance in fixed_query_info_list
    def sorting_key(key):
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)

    priority = {"complete": 1, "code_snippet": 2, "preview": 3, "fold": 4}
    organized_dict: Dict = {}
    for key in sorted_keys:
        values = query_info_list_dict[key]
        nested = {pk: [] for pk in priority}
        for qr in values:
            if qr.format_mode in nested:
                nested[qr.format_mode].append(qr)
        organized_dict[key] = {k: v for k, v in nested.items() if v}

    return organized_dict


# ============================================================================
# Grep content search (fallback for short queries)
# ============================================================================

def grep_content_search(
    file2code: Dict[str, str],
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    include_files: Optional[List[str]] = None,
    max_results: int = 5,
    context_lines: int = 3,
) -> List[QueryResult]:
    """Grep-style fallback search for short queries.

    Source: RPG-ZeroRepo search_node_by_meta.py ``grep_content_search``
    """
    query = query_info.term
    results: List[QueryResult] = []
    matches_found = 0

    search_files = include_files if include_files else list(file2code.keys())

    for file_path in search_files:
        if matches_found >= max_results:
            break
        code = file2code.get(file_path, "")
        if not code:
            continue

        lines = code.split("\n")
        matched_lines: List[int] = []
        for line_num, line in enumerate(lines, start=1):
            if query in line:
                matched_lines.append(line_num)

        if matched_lines:
            for ln in matched_lines[:3]:
                if matches_found >= max_results:
                    break

                module_data = get_module_name_by_line_num(
                    entity_searcher, dep_searcher, file_path, ln
                )
                if module_data:
                    qr = QueryResult(
                        query_info=query_info,
                        format_mode="preview",
                        nid=module_data["node_id"],
                        ntype=module_data["type"],
                        start_line=module_data["start_line"],
                        end_line=module_data["end_line"],
                        retrieve_src=f"Found `{query}` at line {ln} using grep search.",
                    )
                else:
                    start = max(1, ln - context_lines)
                    end = min(len(lines), ln + context_lines)
                    qr = QueryResult(
                        query_info=query_info,
                        format_mode="code_snippet",
                        nid=file_path,
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        retrieve_src=f"Found `{query}` at line {ln} using grep search.",
                    )
                results.append(qr)
                matches_found += 1

    return results


# ============================================================================
# Code snippets search (unified search_terms + line_nums)
# ============================================================================

def search_code_snippets(
    file2code: Dict[str, str],
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    search_terms: Optional[List[str]] = None,
    line_nums: Optional[List[int]] = None,
    file_path_or_pattern: Optional[str] = "**/*.py",
) -> Tuple[str, bool]:
    """Unified code search by terms and/or line numbers.

    Source: RPG-ZeroRepo search_node_by_meta.py ``search_code_snippets``
    """
    all_file_paths = list(file2code.keys())

    # If only file pattern provided and exactly one file matches
    if not search_terms and not line_nums and file_path_or_pattern:
        matched_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if len(matched_files) == 1:
            fp = matched_files[0]
            qi = QueryInfo(term=fp)
            qrs, _ = search_entity(entity_searcher=entity_searcher, query_info=qi)
            if qrs:
                res = f"##Searching for file `{fp}`...\n### Search Result:\n"
                for qr in qrs:
                    res += qr.format_output(entity_searcher) + "\n"
                return res.strip(), True
            return f"File `{fp}` not found in repository.", False
        elif len(matched_files) > 1:
            res = f"Multiple files matched pattern `{file_path_or_pattern}`:\n"
            for f in matched_files[:10]:
                res += f"  - {f}\n"
            if len(matched_files) > 10:
                res += f"  ... and {len(matched_files) - 10} more files.\n"
            res += "\nPlease provide a more specific file path or use 'search_terms' to search within these files."
            return res, False

    if not search_terms and not line_nums:
        return (
            "Error: Please provide at least one of 'search_terms' or 'line_nums'. "
            "Use 'search_terms' to search for code by keywords, or use 'line_nums' "
            "with a specific file path to retrieve code at specific lines.",
            False,
        )

    result = ""
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths

    query_info_list: List[QueryInfo] = []
    all_query_results: List[QueryResult] = []
    filter_terms: List[str] = []

    if search_terms:
        filter_terms = []
        for term in search_terms:
            if is_test_file(term):
                result += f"No results for test files: `{term}`. Please do not search for any test files.\n\n"
            else:
                filter_terms.append(term)

        joint_terms = deepcopy(filter_terms)
        if len(filter_terms) > 1:
            filter_terms.append(" ".join(filter_terms))

        for i, term in enumerate(filter_terms):
            term = term.strip().strip(".")
            if not term:
                continue

            qi = QueryInfo(term=term)
            query_info_list.append(qi)
            cur_results: List[QueryResult] = []

            qrs, continue_search = search_entity(
                entity_searcher=entity_searcher,
                query_info=qi,
                include_files=include_files,
            )
            cur_results.extend(qrs)

            if continue_search:
                grep_results = grep_content_search(
                    file2code=file2code,
                    query_info=qi,
                    entity_searcher=entity_searcher,
                    dep_searcher=dep_searcher,
                    include_files=include_files,
                )
                cur_results.extend(grep_results)

            if i != (len(filter_terms) - 1):
                joint_terms[i] = ""
                filter_terms[-1] = " ".join(t for t in joint_terms if t.strip())
                if filter_terms[-1] in filter_terms[:-1]:
                    filter_terms[-1] = ""

            all_query_results.extend(cur_results)

    if line_nums:
        if isinstance(line_nums, int):
            line_nums = [line_nums]

        file_path = None
        if file_path_or_pattern in all_file_paths:
            file_path = file_path_or_pattern
        else:
            matched = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
            if len(matched) == 1:
                file_path = matched[0]
                result += f"Found file `{file_path}` matching pattern `{file_path_or_pattern}`.\n"
            elif len(matched) > 1:
                result += f"Multiple files matched pattern `{file_path_or_pattern}`. Please specify the exact file path.\n"
                for f in matched[:5]:
                    result += f"  - {f}\n"

        if file_path:
            term = file_path + ":line " + ", ".join(str(ln) for ln in line_nums)
            qi = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)
            qrs = get_code_block_by_line_nums(
                qi, entity_searcher=entity_searcher, dep_searcher=dep_searcher
            )
            all_query_results.extend(qrs)

    merged = merge_query_results(all_query_results)
    ranked = rank_and_aggr_query_results(merged, query_info_list)

    all_suc: List[bool] = []
    for query_infos, format_to_results in ranked.items():
        term_desc = ", ".join(f'"{qi.term}"' for qi in query_infos)
        result += f"##Searching for term {term_desc}...\n"
        result += "### Search Result:\n"
        cur_result = ""

        for fmt, qrs in format_to_results.items():
            if fmt == "fold":
                cur_src = ""
                for qr in qrs:
                    if not cur_src:
                        cur_src = qr.retrieve_src
                    if cur_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_src + "\n\n"
                        cur_src = qr.retrieve_src
                    cur_result += qr.format_output(entity_searcher)
                cur_result += "Source: " + cur_src + "\n"
                if len(qrs) > 1:
                    cur_result += "Hint: Use more detailed query to get the full content of some if needed.\n"
                else:
                    cur_result += f"Hint: Search `{qrs[0].nid}` for the full content if needed.\n"
                cur_result += "\n"
            elif fmt == "complete":
                for qr in qrs:
                    cur_result += qr.format_output(entity_searcher) + "\n"
            elif fmt == "preview":
                filtered: List[QueryResult] = []
                grouped: Dict[str, List[QueryResult]] = defaultdict(list)
                for qr in qrs:
                    if qr.start_line is None or qr.end_line is None:
                        filtered.append(qr)
                        continue
                    if (qr.end_line - qr.start_line) < 100:
                        grouped[qr.file_path].append(qr)
                    else:
                        filtered.append(qr)
                for _fp, group in grouped.items():
                    sorted_group = sorted(group, key=lambda q: (q.start_line or 0, -(q.end_line or 0)))
                    max_el = -1
                    for qr in sorted_group:
                        if qr.end_line and qr.end_line > max_el:
                            filtered.append(qr)
                            max_el = max(max_el, qr.end_line)
                for qr in filtered:
                    cur_result += qr.format_output(entity_searcher) + "\n"
            elif fmt == "code_snippet":
                for qr in qrs:
                    cur_result += qr.format_output(entity_searcher) + "\n"

        cur_result += "\n\n"
        if cur_result.strip():
            result += cur_result
            all_suc.append(True)
        else:
            result += "No locations found.\n\n"
            all_suc.append(False)

    has_line_input = line_nums and file_path_or_pattern in all_file_paths
    has_any_input = len(filter_terms) > 0 or has_line_input
    suc = has_any_input and (len(all_suc) == 0 or all(all_suc))

    return result.strip(), suc
