from typing import List, Optional, Union, Tuple, Dict
from zerorepo.rpg_gen.base.rpg import NodeType, RPG
from zerorepo.utils.compress import get_skeleton
from .searcher import RepoEntitySearcher

class QueryInfo:
    
    query_type: str = 'keyword'
    term: Optional[str] = None
    line_nums: Optional[List] = None
    file_path_or_pattern: Optional[str] = None
                 
    def __init__(self, 
        query_type: str = 'keyword',
        term: Optional[str] = None,
        line_nums: Optional[List] = None,
        file_path_or_pattern: Optional[str] = None,
    ):
        self.query_type = query_type
        if term is not None:
            self.term = term
        if line_nums is not None:
            self.line_nums = line_nums
        if file_path_or_pattern is not None:
            self.file_path_or_pattern = file_path_or_pattern

    def __str__(self):
        parts = []
        if self.term is not None:
            parts.append(f"term: {self.term}")
        if self.line_nums is not None:
            parts.append(f"line_nums: {self.line_nums}")
        if self.file_path_or_pattern is not None:
            parts.append(f"file_path_or_pattern: {self.file_path_or_pattern}")
        return ", ".join(parts)

    def __repr__(self):
        return self.__str__()


   
class QueryResult:
    file_path: Optional[str] = None
    format_mode: Optional[str] = 'complete'
    nid: Optional[str] = None
    ntype: Optional[str] = None
    # code_snippet: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    query_info_list: Optional[List[QueryInfo]] = None
    desc: Optional[str] = None
    message: Optional[str] = None
    warning: Optional[str] = None
    retrieve_src: Optional[str] = None
    
    def __init__(self, query_info: QueryInfo, format_mode: str, 
            nid: Optional[str] = None, 
            ntype: Optional[str] = None, 
            file_path: Optional[str] = None, start_line: Optional[int] = None, end_line: Optional[int] = None,
            desc: Optional[str] = None, 
            message: Optional[str] = None, 
            warning: Optional[str] = None, 
            retrieve_src: Optional[str] = None, 
        ):
        
        self.format_mode = format_mode
        self.query_info_list = []
        self.insert_query_info(query_info)
        
        if nid is not None:
            self.nid = nid
            
        if ntype is not None:
            self.ntype = ntype
            if ntype in [NodeType.FILE, NodeType.CLASS, NodeType.METHOD, NodeType.FUNCTION]:
                self.file_path = nid.split(':')[0]
        
        if file_path is not None:
            self.file_path = file_path
        if start_line is not None and end_line is not None:
            self.start_line = start_line
            self.end_line = end_line
        
        if retrieve_src is not None:
            self.retrieve_src = retrieve_src
            
        if desc is not None:
            self.desc = desc
        if message is not None:
            self.message = message
        if warning is not None:
            self.warning = warning
    
    def insert_query_info(self, query_info: QueryInfo):
        self.query_info_list.append(query_info)
    
    
    def format_output(self, searcher: RepoEntitySearcher):
        cur_result = ''

        if self.format_mode == 'complete':
            node_data_list = searcher.get_node_data([self.nid], return_code_content=True)
            if not node_data_list:
                cur_result += f'Entity `{self.nid}` not found in repository.\n'
                return cur_result
            node_data = node_data_list[0]
            ntype = node_data.get('type')
            feature_paths = node_data.get("feature_paths", [])[:2]
            feature_paths_str = "\n".join(feature_paths)
            cur_result += f'Found {ntype} `{self.nid}`.\n'
            cur_result += "Source: " + self.retrieve_src + '\n'
            if feature_paths_str.strip():
                cur_result += f"It Functionality Features: {feature_paths_str}" + "\n"
            if 'code_content' in node_data:
                code = node_data['code_content'] 
                code_lines = len(code.split("\n"))
                
                if code_lines > 400:
                    new_node_data_list = searcher.get_node_data([self.nid],
                        return_code_content=True,
                        wrap_with_ln=False
                    )
                    code = new_node_data_list[0].get('code_content', '') if new_node_data_list else ''
                    code_skeleton = get_skeleton(
                        code,
                        keep_constant=True,
                        keep_indent=True,
                        keep_imports=True,
                        compress_assign=False,
                        keep_docstring=False,
                        total_lines=400,
                        prefix_lines=200,
                        suffix_lines=200,
                        line_number_mode="original",
                    )
                    hints = (
                        f"Note: The code for `{self.nid}` is very long ({code_lines} lines) and exceeds the display limit. "
                        "Since this entity spans many lines, only a structural skeleton is shown to help you understand "
                        "the overall layout, key definitions, and function signatures. "
                        "If you need to examine a specific section in detail, please refine your query using a more precise "
                        "entity name or line range.\n"
                    )
                    cur_result += hints
                    cur_result += code_skeleton + "\n"
                else:
                    cur_result += node_data.get('code_content', '') + '\n'
            
        elif self.format_mode == 'preview':
            node_data_list = searcher.get_node_data([self.nid], return_code_content=True)
            if not node_data_list:
                cur_result += f'Entity `{self.nid}` not found in repository.\n'
                return cur_result
            node_data = node_data_list[0]
            ntype = node_data['type']
            feature_paths = node_data.get("feature_paths", [])[:2]
            feature_paths_str = "\n".join(feature_paths)
            cur_result += f'Found {ntype} `{self.nid}`.\n'
            cur_result += "Source: " + self.retrieve_src + '\n'
            if feature_paths_str:
                cur_result += f"It Functionality Features: {feature_paths_str}" + "\n"
            if ntype == NodeType.FUNCTION or ntype == NodeType.METHOD:
                cur_result += node_data.get('code_content', '') + '\n'

            elif ntype in [NodeType.CLASS, NodeType.FILE]:
                start_line = node_data.get('start_line', 0)
                end_line = node_data.get('end_line', 0)
                content_size = (end_line - start_line) if (end_line and start_line) else 0
                if content_size <= 100:
                    cur_result += node_data.get('code_content', '') + '\n'
                else:
                    cur_result += f"Just show the structure of this {ntype} due to response length limitations:\n"
                    code_content = searcher.G.nodes[self.nid].get('code', "")
                    structure = get_skeleton(
                        code_content,
                        keep_constant= True,
                        keep_indent = True,
                        keep_imports = True,
                        compress_assign=False,
                        keep_docstring=False,
                        total_lines=500,
                        prefix_lines=200,
                        suffix_lines=200,
                        line_number_mode="original"
                    )
                    cur_result += '```\n' + structure + '\n```\n'
                    cur_result += f'Hint: Search `{self.nid}` to get the full content if needed.\n'
                    
            elif ntype == NodeType.DIRECTORY:
                pass
            
        elif self.format_mode == 'code_snippet':
            if self.desc:
                cur_result += self.desc + '\n'
            else:
                cur_result += f"Found code snippet in file `{self.file_path}`.\n"
            cur_result += "Source: " + self.retrieve_src + '\n'
            # content = get_file_content_(qr.file_path, return_str=True)
            # result_content = line_wrap_content(content, [(, )])
            node_data_list = searcher.get_node_data([self.file_path], return_code_content=True)
            if not node_data_list:
                cur_result += f'File `{self.file_path}` not found in repository.\n'
                return cur_result
            node_data = node_data_list[0]
            feature_paths = node_data.get("feature_paths", [])[:2]
            feature_paths_str = "\n".join(feature_paths)
            if feature_paths_str.strip():
                cur_result += f"It Functionality Features: {feature_paths_str}" + "\n"
            code_content = node_data.get('code_content', '')
            content = code_content.split('\n')[1:-1] if code_content else []
            
            code_snippet = content[(self.start_line-1): self.end_line] # TODO: try-catch
            
            code_snippet = '```\n' + '\n'.join(code_snippet) + '\n```'
            cur_result += code_snippet + '\n'
            if self.message and self.message.strip():
                cur_result += self.message
            
        elif self.format_mode == 'fold':
            node_data_list = searcher.get_node_data([self.nid], return_code_content=False)
            if not node_data_list:
                cur_result += f'Entity `{self.nid}` not found in repository.\n'
                return cur_result
            node_data = node_data_list[0]
            feature_paths = node_data.get("feature_paths", [])[:2]
            feature_paths_str = "\n".join(feature_paths)
            self.ntype = node_data.get('type')
            cur_result += f'Found {self.ntype} `{self.nid}`.\n'
            if feature_paths_str:
                cur_result += f"It Functionality Features: {feature_paths_str}" + "\n"

        return cur_result


    def __str__(self):
        return (
            f"QueryResult(\n"
            f"  query_info_list: {str(self.query_info_list)},\n"
            f"  format_mode: {self.format_mode},\n"
            f"  nid: {self.nid},\n"
            f"  file_path: {self.file_path},\n"
            f"  start_line: {self.start_line},\n"
            f"  end_line: {self.end_line}\n"
            f")"
        )
        