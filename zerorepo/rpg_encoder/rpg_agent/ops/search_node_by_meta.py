import re
import logging
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import fnmatch
from rapidfuzz import process, fuzz
from copy import deepcopy
from networkx import MultiDiGraph
from llama_index.retrievers.bm25 import BM25Retriever
from zerorepo.rpg_encoder.rpg_agent.env.query import QueryInfo, QueryResult
from zerorepo.utils.repo import is_test_file, merge_intervals
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.rpg import NodeType, EdgeType, RPG
from ..env import RepoEntitySearcher, RepoDependencySearcher
from .bm25_model import build_module_retriever

def bm25_module_retrieve(
    query: str,
    entity_searcher: RepoEntitySearcher,
    include_files: Optional[List[str]] = None,
    # file_pattern: Optional[str] = None,
    search_scope: str = 'all',
    similarity_top_k: int = 10,
    # sort_by_type = False
):
    retriever = build_module_retriever(
        entity_searcher=entity_searcher,
        search_scope=search_scope,
        similarity_top_k=similarity_top_k
    )
    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        logging.warning(f'{e}. Probably because the query `{query}` is too short.')
        return []

    filter_nodes = []
    all_nodes = []
    for node in retrieved_nodes:
        if node.score <= 0:
            continue
        if not include_files or node.text.split(':')[0] in include_files:
            filter_nodes.append(node.text)
        all_nodes.append(node.text)

    if filter_nodes:
        return filter_nodes
    else:
        return all_nodes
    
    
    

def grep_content_search(
    file2code: Dict[str, str],
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    include_files: Optional[List[str]] = None,
    max_results: int = 5,
    context_lines: int = 3
) -> List[QueryResult]:
    """
    Fallback grep-style search for short queries that BM25 cannot handle.
    Searches for exact string matches in code files.
    """
    query = query_info.term
    cur_query_results = []
    matches_found = 0

    search_files = include_files if include_files else list(file2code.keys())

    for file_path in search_files:
        if matches_found >= max_results:
            break

        code = file2code.get(file_path, '')
        if not code:
            continue

        lines = code.split('\n')
        matched_lines = []

        for line_num, line in enumerate(lines, start=1):
            if query in line:
                matched_lines.append(line_num)

        if matched_lines:
            # Try to get module context for matched lines
            for line_num in matched_lines[:3]:  # Limit per file
                if matches_found >= max_results:
                    break

                module_data = get_module_name_by_line_num(
                    entity_searcher, dep_searcher, file_path, line_num
                )

                if module_data:
                    query_result = QueryResult(
                        query_info=query_info,
                        format_mode='preview',
                        nid=module_data['node_id'],
                        ntype=module_data['type'],
                        start_line=module_data['start_line'],
                        end_line=module_data['end_line'],
                        retrieve_src=f"Found `{query}` at line {line_num} using grep search."
                    )
                else:
                    # Return code snippet with context
                    start_line = max(1, line_num - context_lines)
                    end_line = min(len(lines), line_num + context_lines)
                    query_result = QueryResult(
                        query_info=query_info,
                        format_mode='code_snippet',
                        nid=file_path,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        retrieve_src=f"Found `{query}` at line {line_num} using grep search."
                    )

                cur_query_results.append(query_result)
                matches_found += 1

    return cur_query_results


def bm25_content_retrieve(
    retriever: BM25Retriever,
    entity_searcher: RepoEntitySearcher,
    query_info: QueryInfo,
    include_files: Optional[List[str]] = None
) -> List[QueryResult]:

    cur_query_results = []
    query = query_info.term

    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        logging.warning(f'{e}. Probably because the query `{query}` is too short. Will use grep fallback.')
        return None  # Return None to signal grep fallback needed
    
    for node in retrieved_nodes:
        file = node.metadata['file_path']
        if not include_files or file in include_files:
            if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                query_result = QueryResult(
                                    query_info=query_info, 
                                    format_mode='code_snippet',
                                    nid=node.metadata['file_path'],
                                    file_path=node.metadata['file_path'],
                                    start_line=node.metadata['start_line'],
                                    end_line=node.metadata['end_line'],
                                    retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                )
                cur_query_results.append(query_result)
            elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                nids = []
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    searcher = entity_searcher
                    if searcher.has_node(nid):
                        nids.append(nid)
                    # TODO: warning if not find
                    
                node_datas = searcher.get_node_data(nids, return_code_content=True)
                sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]
                
                message = ''
                if sorted_nids:
                    if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
                        nid = sorted_ndatas[0]['node_id']
                        ntype = sorted_ndatas[0]['type']
                        # The code for {ntype} {nid} is incomplete; search {nid} for the full content if needed.
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
                        nid = sorted_ndatas[-1]['node_id']
                        ntype = sorted_ndatas[-1]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if message.strip():
                        message = "Hint: \n"+ message
                
                nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                desc = f"Found {nids_str}."
                query_result = QueryResult(query_info=query_info, 
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           desc=desc,
                                           message=message,
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                
                cur_query_results.append(query_result)
            else:
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    searcher = entity_searcher
                    print(nid)
                    if searcher.has_node(nid):
                        ndata_list = searcher.get_node_data([nid], return_code_content=True)
                        if not ndata_list:
                            continue
                        ndata = ndata_list[0]
                        query_result = QueryResult(query_info=query_info, format_mode='preview',
                                                   nid=ndata['node_id'],
                                                   ntype=ndata['type'],
                                                   start_line=ndata.get('start_line'),
                                                   end_line=ndata.get('end_line'),
                                                   retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                                   )
                        cur_query_results.append(query_result)
                    else:
                        continue
        
    cur_query_results = cur_query_results[:5]
    return cur_query_results



def find_matching_files_from_list(file_list, file_pattern):
    """
    Find and return a list of file paths from the given list that match the given keyword or pattern.
    
    :param file_list: A list of file paths to search through.
    :param file_pattern: A keyword or pattern for file matching. Can be a simple keyword or a glob-style pattern.
    :return: A list of matching file paths
    """
    # If the pattern contains any of these glob-like characters, treat it as a glob pattern.
    if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
        matching_files = fnmatch.filter(file_list, file_pattern)
    else:
        # Otherwise, treat it as a keyword search
        matching_files = [file for file in file_list if file_pattern in file]
    
    return matching_files

def merge_query_results(query_results: List[QueryResult]):
    
    priority = ['complete', 'code_snippet', 'preview', 'fold']
    merged_results: Dict[str, QueryResult] = {}
    all_query_results: List[QueryResult] = []

    for qr in query_results:
        if qr.format_mode == 'code_snippet':
            all_query_results.append(qr)
        
        elif qr.nid and qr.nid in merged_results:
            # Merge query_info_list
            if qr.query_info_list[0] not in merged_results[qr.nid].query_info_list:
                merged_results[qr.nid].query_info_list.extend(qr.query_info_list)

            # Select the format_mode with the highest priority
            existing_format_mode = merged_results[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_format_mode):
                merged_results[qr.nid].format_mode = qr.format_mode
                merged_results[qr.nid].start_line = qr.start_line
                merged_results[qr.nid].end_line = qr.end_line
                merged_results[qr.nid].retrieve_src = qr.retrieve_src
                
        elif qr.nid:
            merged_results[qr.nid] = qr
    
    all_query_results += list(merged_results.values())
    return all_query_results


def rank_and_aggr_query_results(query_results, fixed_query_info_list):
    query_info_list_dict = {}

    for qr in query_results:
        # Convert the query_info_list to a tuple so it can be used as a dictionary key
        key = tuple(qr.query_info_list)

        if key in query_info_list_dict:
            query_info_list_dict[key].append(qr)
        else:
            query_info_list_dict[key] = [qr]
            
    # for the key: sort by query
    def sorting_key(key):
        # Find the first matching element index from fixed_query_info_list in the key (tuple of query_info_list)
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        # If no match is found, assign a large index to push it to the end
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)
    sorted_query_info_list_dict = {key: query_info_list_dict[key] for key in sorted_keys}
    
    # for the value: sort by format priority
    priority = {'complete': 1, 'code_snippet': 2, 'preview': 3,  'fold': 4}  # Lower value indicates higher priority
    # TODO: merge the same node in 'code_snippet' and 'preview'
    
    organized_dict = {}
    for key, values in sorted_query_info_list_dict.items():
        nested_dict = {priority_key: [] for priority_key in priority.keys()}
        for qr in values:
            # Place the qr in the nested dictionary based on its format_mode
            if qr.format_mode in nested_dict:
                nested_dict[qr.format_mode].append(qr)

        # Only add keys with non-empty lists to keep the result clean
        organized_dict[key] = {k: v for k, v in nested_dict.items() if v}
    
    return organized_dict


def parse_node_id(nid: str):
    nfile = nid.split(':')[0]
    nname = nid.split(':')[-1]
    return nfile, nname


def fuzzy_retrieve(
    keyword: str,
    rpg: Optional[RPG] = None,
    search_scope: str = 'all',  # enum = {'function', 'class', 'file', 'all'}
    include_files: Optional[str] = None,
    similarity_top_k: int = 5,
    return_score: bool = False,
):
    """
    Fuzzy retrieve entities by keyword.

    Args:
        keyword: Search keyword
        rpg: RPG instance (contains dep_graph)
        search_scope: 'all', 'function', 'class', 'file'
        include_files: Filter by file paths
        similarity_top_k: Number of top results
        return_score: Whether to return scores
    """
    # Get graph from RPG
    graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    if graph is None:
        return [] if not return_score else []

    selected_nids = list()
    filter_nids = list()
    for nid in graph:

        ndata = graph.nodes[nid]
        if search_scope == 'all' and \
            ndata['type'] in [NodeType.FILE, NodeType.CLASS, 
            NodeType.METHOD, NodeType.FUNCTION]:    
            nfile = nid.split(':')[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)
            
        elif ndata['type'] == search_scope:
            nfile = nid.split(':')[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)
    
    if not filter_nids:
        filter_nids = selected_nids
        
    # Custom function to split tokens on underscores and hyphens
    def custom_tokenizer(s):
        return re.findall(r'\b\w+\b', s.replace('_', ' ').replace('-', ' '))

    # Use token_set_ratio with custom tokenizer
    matches = process.extract(
        keyword,
        filter_nids,
        scorer=fuzz.token_set_ratio,
        processor=lambda s: ' '.join(custom_tokenizer(s)),
        limit=similarity_top_k
    )
    if not return_score:
        return_nids = [match[0] for match in matches]
        return return_nids
    
    # matches: List[Tuple(nid, score)]
    return matches


def search_entity_in_global_dict(
    searcher: RepoEntitySearcher,
    query: str,
    include_files: Optional[List[str]] = None,
    prefix_term=None
):
    # 如果 query 是类名、函数名或方法名，去掉前缀
    if query.startswith(('class ', 'Class')):
        query = query[len('class '):].strip()
    elif query.startswith(('function ', 'Function ')):
        query = query[len('function '):].strip()
    elif query.startswith(('method ', 'Method ')):
        query = query[len('method '):].strip()
    elif query.startswith('def '):
        query = query[len('def '):].strip()

    # 尝试在 global_name_dict 中查找
    if query in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[query]
    # 如果没有找到，尝试在 global_name_dict_lowercase 中查找（大小写不敏感）
    elif query.lower() in searcher.global_name_dict_lowercase:
        query = query.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[query]
    # 处理文件名：假设 query 是文件名，首先尝试直接查找
    elif query.endswith('.py'):
        # 处理查询为文件名的情况（考虑文件名大小写）
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
    found_entities_filter_dict = defaultdict(list)

    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata['node_id'])
        
        if not include_files or nfile in include_files:
            prefix_terms = []
            candidite_prefixes = re.split(r'[./:]', ndata['node_id'].lower().replace('.py', ''))[:-1]
            
            if prefix_term:
                prefix_terms = prefix_term.lower().split('.')
                
            if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
                found_entities_filter_dict[ndata['type']].append(ndata['node_id'])

    return found_entities_filter_dict



def search_entity(
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    include_files: List[str]=None
) :

    query = query_info.term

    continue_search = True
    cur_query_results = []

    if entity_searcher.has_node(query):
        continue_search = False
        
        query_result = QueryResult(
            query_info=query_info, format_mode='complete', nid=query,
            retrieve_src=f"Exact match found for entity name `{query}`."
        )
        cur_query_results.append(query_result)

    elif query.endswith('.__init__'):
        nid = query[:-(len('.__init__'))]
        
        if entity_searcher.has_node(nid):
            continue_search = False
            node_data = entity_searcher.get_node_data(
                [nid], return_code_content=True
            )[0]

            query_result = QueryResult(
                query_info=query_info, format_mode='preview',
                nid=nid,
                ntype=node_data['type'],
                start_line=node_data.get('start_line'),
                end_line=node_data.get('end_line'),
                retrieve_src=f"Exact match found for entity name `{nid}`."
            )
            cur_query_results.append(query_result)

    # second: search in global name dict
    if continue_search: 
        found_entities_dict = search_entity_in_global_dict(entity_searcher, query, include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(entity_searcher, query)
            
        use_sub_term = False
        used_term = query
        
        if not found_entities_dict and "." in query:
            # for cases: class_name.method_name
            try:
                prefix_term = '.'.join(query.split('.')[:-1]).split()[-1] # incase of 'class '/ 'function '
            except IndexError:
                prefix_term = None
            split_term = query.split('.')[-1].strip()
            used_term = split_term
            found_entities_dict = search_entity_in_global_dict(entity_searcher, split_term, include_files, prefix_term)
            if not found_entities_dict:
                found_entities_dict = search_entity_in_global_dict(entity_searcher, split_term, prefix_term)
            if not found_entities_dict:
                use_sub_term = True
                found_entities_dict = search_entity_in_global_dict(entity_searcher, split_term)
        
        if found_entities_dict:
            max_fold_results = 5  # Limit fold results when there are many matches
            for ntype, nids in found_entities_dict.items():
                if not nids: continue

                # procee class and function in the same way
                if ntype in [NodeType.FUNCTION, NodeType.CLASS, NodeType.FILE, NodeType.METHOD]:
                    if len(nids) <= 3:
                        node_datas = entity_searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='preview',
                                                       nid=ndata['node_id'],
                                                       ntype=ndata['type'],
                                                       start_line=ndata.get('start_line'),
                                                       end_line=ndata.get('end_line'),
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                    else:
                        # Limit the number of fold results to avoid overwhelming output
                        limited_nids = nids[:max_fold_results]
                        node_datas = entity_searcher.get_node_data(limited_nids, return_code_content=False)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='fold',
                                                       nid=ndata['node_id'],
                                                       ntype=ndata['type'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                    if not use_sub_term:
                        continue_search = False
                    else:
                        continue_search = True

    # third: fuzzy search (entity + content)
    if continue_search:
        module_nids = []

        # search entity by keyword
        module_nids = bm25_module_retrieve(query=query, entity_searcher=entity_searcher, include_files=include_files)
        if not module_nids:
            module_nids = bm25_module_retrieve(query=query, entity_searcher=entity_searcher)

        if not module_nids:
            # result += f"No entity found using BM25 search. Try to use fuzzy search...\n"
            module_nids = fuzzy_retrieve(query, rpg=entity_searcher.rpg, similarity_top_k=3)

        module_datas = entity_searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        showed_file_num = 0
        max_file_results = 3  # Limit file/directory results
        max_module_results = 3  # Limit function/class/method results

        for module in module_datas[:5]:
            if module['type'] in [NodeType.FILE, NodeType.DIRECTORY]:
                if showed_file_num < max_file_results:
                    showed_file_num += 1
                    query_result = QueryResult(query_info=query_info, format_mode='fold',
                                            nid=module['node_id'],
                                            ntype=module['type'],
                                            retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                            )
                    cur_query_results.append(query_result)
            elif showed_module_num < max_module_results:
                showed_module_num += 1
                query_result = QueryResult(query_info=query_info, format_mode='preview',
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        start_line=module.get('start_line'),
                                        end_line=module.get('end_line'),
                                        retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)


    return (cur_query_results, continue_search)


def get_module_name_by_line_num(
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,    
    file_path: str, 
    line_num: int
):
    # TODO: 
    # if the given line isn't in a function of a class and the class is large, 
    # find the nearest two member functions and return

    cur_module = None
    if entity_searcher.has_node(file_path):
        module_nids, _ = dep_searcher.get_neighbors(file_path,
            etype_filter=[EdgeType.CONTAINS])
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            if module['start_line'] <= line_num <= module['end_line']:
                cur_module = module  # ['node_id']
                break
        if cur_module and cur_module['type'] == NodeType.CLASS:
            func_nids, _ = dep_searcher.get_neighbors(cur_module['node_id'], etype_filter=[EdgeType.CONTAINS])
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=True)
            for func in func_ndatas:
                if func['start_line'] <= line_num <= func['end_line']:
                    cur_module = func  # ['node_id']
                    break

    if cur_module: 
        return cur_module
    
    return None


def get_code_block_by_line_nums(
    query_info: QueryInfo,
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoDependencySearcher,
    context_window=20
):
    # file_path: str, line_nums: List[int]
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums
    cur_query_results = []

    file_data_list = entity_searcher.get_node_data([file_path], return_code_content=False)
    if not file_data_list:
        return cur_query_results
    file_data = file_data_list[0]
    line_intervals = []
    res_modules = []
    # res_code_blocks = None
    for line in line_nums:
        module_data = get_module_name_by_line_num(
            entity_searcher,
            dep_searcher,
            file_path, 
            line
        )
        
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data['end_line'], line + context_window)
            line_intervals.append((min_line_num, max_line_num))
            
        elif module_data['node_id'] not in res_modules:
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                       nid=module_data['node_id'],
                                       ntype=module_data['type'],
                                       start_line=module_data['start_line'],
                                       end_line=module_data['end_line'],
                                       retrieve_src=f"Retrieved code context including {query_info.term}."
                                       )
            cur_query_results.append(query_result)
            res_modules.append(module_data['node_id'])
            
    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for interval in line_intervals:
            start_line, end_line = interval
            query_result = QueryResult(query_info=query_info, 
                                        format_mode='code_snippet',
                                        nid=file_path,
                                        file_path=file_path,
                                        start_line=start_line,
                                        end_line=end_line,
                                        retrieve_src=f"Retrieved code context including {query_info.term}."
                                        )
            cur_query_results.append(query_result)
        # res_code_blocks = line_wrap_content('\n'.join(file_content), line_intervals)

    # return res_code_blocks, res_modules
    return cur_query_results


def get_entity_contents(
    entity_searcher: RepoEntitySearcher,
    entity_names: List[str]
):
    result = ''
    for name in entity_names:
        name = name.strip().strip('.')
        if not name: continue
        
        result += f'##Searching for entity `{name}`...\n'
        result += f'### Search Result:\n'
        query_info = QueryInfo(term=name)
        
        if entity_searcher.has_node(name):
            query_result = QueryResult(query_info=query_info, format_mode='complete', nid=name,
                                    retrieve_src=f"Exact match found for entity name `{name}`."
                                    )
            result += query_result.format_output(entity_searcher)
            result += '\n\n'
        else:
            result += 'Invalid name. \nHint: Valid entity name should be formatted as "file_path:QualifiedName" or just "file_path".'
            result += '\n\n'
    return result.strip()


def search_code_snippets(
    repo_skeleton: RepoSkeleton,
    bm_25_retriever: Optional[BM25Retriever],
    entity_searcher: RepoEntitySearcher,
    dep_searcher: RepoEntitySearcher,
    search_terms: Optional[List[str]] = None,
    line_nums: Optional[List] = None,
    file_path_or_pattern: Optional[str]= "**/*.py",
):
    file2code: Dict[str, str] = repo_skeleton.get_file_code_map()
    all_file_paths = list(file2code.keys())

    # If only file_path_or_pattern is provided and it matches exactly one file, return its content
    if not search_terms and not line_nums and file_path_or_pattern:
        matched_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if len(matched_files) == 1:
            # Exact file match - return file content using search_entity
            file_path = matched_files[0]
            query_info = QueryInfo(term=file_path)
            query_results, _ = search_entity(
                entity_searcher=entity_searcher,
                query_info=query_info,
                include_files=None
            )
            if query_results:
                result = f'##Searching for file `{file_path}`...\n'
                result += f'### Search Result:\n'
                for qr in query_results:
                    result += qr.format_output(entity_searcher)
                    result += '\n'
                return result.strip(), True
            else:
                return f"File `{file_path}` not found in repository.", False
        elif len(matched_files) > 1:
            # Multiple files matched - list them and ask user to be more specific
            result = f"Multiple files matched pattern `{file_path_or_pattern}`:\n"
            for f in matched_files[:10]:
                result += f"  - {f}\n"
            if len(matched_files) > 10:
                result += f"  ... and {len(matched_files) - 10} more files.\n"
            result += "\nPlease provide a more specific file path or use 'search_terms' to search within these files."
            return result, False

    if not search_terms and not line_nums:
        return "Error: Please provide at least one of 'search_terms' or 'line_nums'. " \
               "Use 'search_terms' to search for code by keywords, or use 'line_nums' with a specific file path to retrieve code at specific lines.", False

    result = ""
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths
        
    
    query_info_list = []
    all_query_results = []
    filter_terms = []

    if search_terms:
        filter_terms = []
        for term in search_terms:
            if is_test_file(term):
                result += f'No results for test files: `{term}`. Please do not search for any test files.\n\n'
            else:
                filter_terms.append(term)
        
        joint_terms = deepcopy(filter_terms)
        if len(filter_terms) > 1:
            filter_terms.append(' '.join(filter_terms))
        
        for i, term in enumerate(filter_terms):
            term = term.strip().strip('.')
            if not term: continue
                
            query_info = QueryInfo(term=term)
            query_info_list.append(query_info)
            
            cur_query_results = []

            # search entity
            query_results, continue_search = search_entity(
                entity_searcher=entity_searcher,
                query_info=query_info, 
                include_files=include_files
            )
            cur_query_results.extend(query_results)
            
            # search content
            if continue_search:
                query_results = None
                if bm_25_retriever is not None:
                    query_results = bm25_content_retrieve(retriever=bm_25_retriever,
                        entity_searcher=entity_searcher,
                        query_info=query_info, include_files=include_files
                    )

                # If BM25 is unavailable or failed (returned None), use grep fallback
                if query_results is None:
                    query_results = grep_content_search(
                        file2code=file2code,
                        query_info=query_info,
                        entity_searcher=entity_searcher,
                        dep_searcher=dep_searcher,
                        include_files=include_files
                    )

                cur_query_results.extend(query_results)
            
                            
            if i != (len(filter_terms)-1):
                joint_terms[i] = ''
                filter_terms[-1] = ' '.join([t for t in joint_terms if t.strip()])
                if filter_terms[-1] in filter_terms[:-1]:
                    filter_terms[-1] = ''
                
            all_query_results.extend(cur_query_results)
    
    if line_nums:
        if isinstance(line_nums, int):
            line_nums = [line_nums]

        # Try to find the file - first exact match, then pattern match
        file_path = None
        if file_path_or_pattern in all_file_paths:
            file_path = file_path_or_pattern
        else:
            # Try to find matching files using pattern
            matched_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
            if len(matched_files) == 1:
                file_path = matched_files[0]
                result += f"Found file `{file_path}` matching pattern `{file_path_or_pattern}`.\n"
            elif len(matched_files) > 1:
                result += f"Multiple files matched pattern `{file_path_or_pattern}`. Please specify the exact file path.\n"
                for f in matched_files[:5]:
                    result += f"  - {f}\n"
                if len(matched_files) > 5:
                    result += f"  ... and {len(matched_files) - 5} more files.\n"
            else:
                result += f"File `{file_path_or_pattern}` not found in repository.\n"

        if file_path:
            term = file_path + ':line ' + ', '.join([str(line) for line in line_nums])
            query_info = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)

            # Search for codes based on file name and line number
            query_results = get_code_block_by_line_nums(query_info, entity_searcher=entity_searcher,
                dep_searcher=dep_searcher
            )
            all_query_results.extend(query_results)
    
    merged_results = merge_query_results(all_query_results)
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list)
    
    # format output
    # format_mode: 'complete', 'preview', 'code_snippet', 'fold': 4
    all_suc = []
    for query_infos, format_to_results in ranked_query_to_results.items():
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
                    # Note: all results here are guaranteed to have non-None start_line and end_line
                    sorted_results = sorted(results, key=lambda qr: (qr.start_line or 0, -(qr.end_line or 0)))

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
                        if qr.end_line and qr.end_line > max_end_line:
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
    
    # Success if we have any successful results (from search_terms or line_nums)
    has_line_nums_input = line_nums and file_path_or_pattern in all_file_paths
    has_any_input = len(filter_terms) > 0 or has_line_nums_input
    suc = has_any_input and (len(all_suc) == 0 or all(all_suc))

    return result.strip(), suc
