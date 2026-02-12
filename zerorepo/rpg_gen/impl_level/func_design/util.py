import json5, logging, ast, re
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from copy import deepcopy
from zerorepo.utils.api import parse_thinking_output
from zerorepo.rpg_gen.base.unit import CodeUnit, ParsedFile

def omit_leaf_subtree(self, subtree):
    """
    将三层嵌套的结构中所有叶子替换为 "...", 保持 JSON 样式结构
    """
    def replace_leaves(tree):
        if isinstance(tree, dict):
            return {
                key: replace_leaves(value)
                if isinstance(value, dict) else "..."
                for key, value in tree.items()
            }
        elif isinstance(tree, list):
            return "..."
        else:
            return "..."

    return replace_leaves(subtree)
    
    
    
def get_upstream_files(file_path: str, reverse_graph, implemented_paths, max_count=5):
    visited = set()
    queue = deque([file_path])
    result = []

    while queue and len(result) < max_count:
        current = queue.popleft()
        for parent in reverse_graph.get(current, []):
            if parent not in visited:
                visited.add(parent)
                if parent in implemented_paths:
                    result.append(parent)
                queue.append(parent)
                if len(result) >= max_count:
                    break
    return result

def parse_response_to_json(response: str) -> Dict:
    try:
        parsed_output = parse_thinking_output(response)
        cleaned_output = parsed_output.replace("```json", "").replace("```", "").replace("\n", "").replace("\t", "")
        output = json5.loads(cleaned_output)
        return True, output
    except Exception as e:
        return False, f"Failed to parse json from your solution, because of {e}."        

def topo_sort_file_graph(graph: list[dict]) -> list[str] | None:
    """
    Given a list of {"from": str, "to": str} edges, return a topologically sorted list of files.
    If a cycle is detected, return None.
    """
    # Build adjacency list and indegree map
    adj = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()

    for edge in graph:
        from_f = edge["from"]
        to_f = edge["to"]
        adj[from_f].append(to_f)
        indegree[to_f] += 1
        nodes.add(from_f)
        nodes.add(to_f)

    # Initialize queue with nodes that have zero indegree
    zero_indegree = deque([n for n in nodes if indegree[n] == 0])
    sorted_list = []

    while zero_indegree:
        node = zero_indegree.popleft()
        sorted_list.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree.append(neighbor)

    if len(sorted_list) != len(nodes):
        # Cycle detected
        return None

    return sorted_list


def validate_file_implementation_graph(graph, file_names):
    from collections import defaultdict, deque

    file_set = set(file_names)
    feedbacks = []
    is_valid = True

    # 1. 检查文件是否存在
    for edge in graph:
        from_f = edge.get("from")
        to_f = edge.get("to")

        if from_f not in file_set:
            feedbacks.append(
                f"Invalid file reference: `{from_f}` is not found in the provided file list. "
            )
            is_valid = False
        if to_f not in file_set:
            feedbacks.append(
                f"Invalid file reference: `{to_f}` is not found in the provided file list. "
            )
            is_valid = False

    if feedbacks:
        feedbacks.append(f"Please ensure all `to` fields reference actual files from the file paths provieded.")

    # 2. 检查是否是 DAG（无环）
    adj = defaultdict(list)
    indegree = defaultdict(int)
    for edge in graph:
        f, t = edge["from"], edge["to"]
        adj[f].append(t)
        indegree[t] += 1

    queue = deque([f for f in file_set if indegree[f] == 0])
    visited = set()

    while queue:
        node = queue.popleft()
        visited.add(node)
        for neighbor in adj.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(visited) != len(file_set):
        feedbacks.append(
            "Cycle detected or not all files are connected. "
            "The `file_implementation_graph` must form a valid DAG (Directed Acyclic Graph). "
            "Please remove any circular dependencies or ensure all files are reachable."
        )
        is_valid = False

    # 3. 检查是否覆盖所有文件
    used_files = {e["from"] for e in graph} | {e["to"] for e in graph}
    missing = file_set - used_files
    if missing:
        feedbacks.append(
            f"Missing file(s): The following file(s) are not referenced in the graph: {sorted(missing)}. "
            f"Please include edges involving every file so that the full implementation flow is covered."
        )
        is_valid = False

    # 4. 汇总
    if is_valid:
        return "`file_implementation_graph` is valid. All structural checks passed.", True
    else:
        return "Validation failed:\n\n" + "\n\n".join(f"- {msg}" for msg in feedbacks), False


def validate_file_implementation_list(file_list: List[str], file_names: List[str]) -> Tuple[str, bool]:
    """
    Validate a file implementation order list.

    Args:
        file_list: The ordered list of file paths from LLM output
        file_names: The expected list of file paths

    Returns:
        Tuple of (feedback_message, is_valid)
    """
    file_set = set(file_names)
    list_set = set(file_list)
    feedbacks = []
    is_valid = True

    # 1. Check for invalid file references
    invalid_files = list_set - file_set
    if invalid_files:
        feedbacks.append(
            f"Invalid file reference(s): {sorted(invalid_files)} are not in the provided file list. "
            f"Only use file paths from the provided list."
        )
        is_valid = False

    # 2. Check for missing files
    missing_files = file_set - list_set
    if missing_files:
        feedbacks.append(
            f"Missing file(s): {sorted(missing_files)} are not included in the implementation order. "
            f"All files must appear exactly once in the list."
        )
        is_valid = False

    # 3. Check for duplicates
    if len(file_list) != len(list_set):
        from collections import Counter
        counter = Counter(file_list)
        duplicates = [f for f, count in counter.items() if count > 1]
        feedbacks.append(
            f"Duplicate file(s) detected: {sorted(duplicates)}. "
            f"Each file must appear exactly once in the implementation order."
        )
        is_valid = False

    # 4. Summary
    if is_valid:
        return "`implementation_order` is valid. All structural checks passed.", True
    else:
        return "Validation failed:\n\n" + "\n\n".join(f"- {msg}" for msg in feedbacks), False
    

def format_dep_graph_to_string(dep_graph_json: list) -> str:
    if not isinstance(dep_graph_json, list):
        return "Invalid input: Expected a list of dependency items."

    # 建图 + 入度统计
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    node_set = set()
    name_map = {}

    for item in dep_graph_json:
        name = item.get("name")
        depends_on = item.get("depends_on", [])
        if not name or not isinstance(depends_on, list):
            continue
        name_map[name] = depends_on
        for dep in depends_on:
            graph[dep].append(name)  # reverse edge for topological sort
            in_degree[name] += 1
        node_set.add(name)
        node_set.update(depends_on)

    for node in node_set:
        in_degree.setdefault(node, 0)

    # 拓扑排序
    queue = deque([n for n in node_set if in_degree[n] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != len(node_set):
        return "Cycle detected in dependency graph. Cannot perform topological sort."

    # 生成输出
    lines = []
    for name in topo_order:
        deps = name_map.get(name, [])
        if deps:
            lines.append(f"- `{name}` depends on: {', '.join(f'`{d}`' for d in deps)}")
        else:
            lines.append(f"- `{name}` has no dependencies")

    return "\n".join(lines)


def format_data_flow_to_string(data_flow_json: List[Dict]) -> str:
    if not isinstance(data_flow_json, list):
        return "Invalid input: expected a list of data flow entries."

    # 构建图
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    edges = []

    # 节点收集 + 统计入度
    nodes = set()
    for item in data_flow_json:
        src = item.get("from")
        dst = item.get("to")
        if src and dst:
            graph[src].append(dst)
            in_degree[dst] += 1
            nodes.update([src, dst])
            edges.append(item)

    for node in nodes:
        in_degree.setdefault(node, 0)

    # 拓扑排序（Kahn's algorithm）
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != len(nodes):
        return "Cycle detected in data flow. Cannot perform topological sort."

    # 映射节点顺序用于排序 edges
    node_rank = {node: i for i, node in enumerate(topo_order)}
    sorted_edges = sorted(edges, key=lambda e: node_rank.get(e.get("from", ""), 0))

    # 生成描述文本
    lines = []
    for i, item in enumerate(sorted_edges):
        from_node = item.get("from", "<unknown>")
        to_node = item.get("to", "<unknown>")
        data_id = item.get("data_id", "<data>")
        data_type = item.get("data_type", "<type>")
        transformation = item.get("transformation", "transferred")

        line = f"{i+1}. `{from_node}` -> `{to_node}` with `{data_id}` ({data_type}) — transformation: *{transformation}*"
        lines.append(line)

    return "\n".join(lines)



def parse_code_blocks_by_subtree_heading(markdown: str):
    from collections import defaultdict

    grouped = defaultdict(dict)
    current_subtree = None
    raw_code_block_count = 0

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("## "):
            current_subtree = line[3:].strip()
            i += 1
            continue

        if current_subtree and line.startswith("### "):
            file_path = line[4:].strip()
            i += 1

            if i < len(lines) and lines[i].strip() == "```python":
                i += 1
                code_lines = []
                while i < len(lines) and lines[i].strip() != "```":
                    code_lines.append(lines[i])
                    i += 1
                i += 1  # Skip closing ```
                raw_code_block_count += 1

                code = "\n".join(code_lines).strip()
                if file_path in grouped[current_subtree]:
                    grouped[current_subtree][file_path] += "\n\n" + code
                else:
                    grouped[current_subtree][file_path] = code
                continue

        i += 1

    # Convert grouped dict-of-dicts to dict-of-lists
    result = {
        subtree: [
            {"file_path": fp, "code": code}
            for fp, code in files.items()
        ]
        for subtree, files in grouped.items()
    }

    return result, raw_code_block_count


 
def detect_dep_graph_is_cycle(output, required_keys, required_values=None):
    from collections import defaultdict

    # 初始化
    graph = defaultdict(list)
    all_names = set()
    all_dependencies = set()

    # 构建图
    for item in output:
        name = item.get("name")
        if name is None:
            return "Each item must have a 'name' key."
        all_names.add(name)
        depends_on = item.get("depends_on", [])
        for dep in depends_on:
            graph[name].append(dep)
            all_dependencies.add(dep)

    # 检查：所有 name 必须在 required_keys 中
    invalid_names = all_names - set(required_keys)
    if invalid_names:
        return f"Names in graph not allowed by required_keys: {invalid_names}"

    # 检查：所有 required_keys 必须出现在 name 中
    missing_keys = set(required_keys) - all_names
    if missing_keys:
        return f"Missing required keys in graph: {missing_keys}"

    # 检查依赖值是否合法（如果提供 required_values）
    if required_values is not None:
        invalid_deps = all_dependencies - set(required_values)
        if invalid_deps:
            return f"Invalid dependency values not in required_values: {invalid_deps}"

    # 检查是否有环（DFS）
    visited = set()
    path_stack = []

    def dfs(node, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        path_stack.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                result = dfs(neighbor, rec_stack)
                if result:
                    return result
            elif neighbor in rec_stack:
                cycle_start = path_stack.index(neighbor)
                cycle = path_stack[cycle_start:] + [neighbor]
                return f"Cycle detected: {' -> '.join(cycle)}"

        rec_stack.remove(node)
        path_stack.pop()
        return None

    for node in all_names:
        if node not in visited:
            result = dfs(node, set())
            if result:
                return result

    return None  # 所有检查通过


def parse_feature_blocks(text):
    """
    Parses blocks of the form:
    design_itfs_for_feature(features=[...]):
    ```python
    ...
    ```
    Also supports:
    design_itfs_for_feature(features="Feature/Path"):
    """
    pattern = re.compile(
        r'design_itfs_for_feature\s*\(\s*features\s*=\s*(\[.*?\]|".*?"|\'.*?\')\s*\)?\s*:?\s*[\r\n\s]*```python\s*\n(.*?)```',
        re.DOTALL
    )
    results = []
    for match in pattern.finditer(text):
        raw = match.group(1)
        try:
            val = ast.literal_eval(raw)
            feature_paths = [val] if isinstance(val, str) else val
        except Exception as e:
            logging.warning(f"Failed to parse feature paths: {raw}, error: {e}")
            continue

        code = match.group(2).strip()
        results.append((feature_paths, code))

    return results


def detect_unparsed_blocks(text):
    """
    Compare declared 'design_itfs_for_feature' blocks with parsed results,
    to detect any missed matches.
    """
    # Count all design_itfs_for_feature declarations
    all_calls = re.findall(
        r'design_itfs_for_feature\s*\(',
        text,
        flags=re.DOTALL
    )

    # Use the parser
    parsed_blocks = parse_feature_blocks(text)

    # Print discrepancy
    declared_count = len(all_calls)
    parsed_count = len(parsed_blocks)

    if declared_count != parsed_count:
        logging.warning(f"[Warning] Declared blocks: {declared_count}, Parsed blocks: {parsed_count}")
        logging.warning(f"Likely {declared_count - parsed_count} block(s) not properly parsed.")

    return declared_count, parsed_count



def filter_import_and_assignment_units(units: List[CodeUnit]) -> List[CodeUnit]:
    return [unit for unit in units if unit.unit_type in {"import", "assignment"}]


def code_blocks_to_units(blocks: List[Tuple[List[str], str]], file_path: str) -> List[Tuple[List[str], CodeUnit]]:
    code_units_with_features = []
    for features, code_str in blocks:
        try:
            parsed = ast.parse(code_str)
        except SyntaxError as e:
            logging.warning(f"Failed to parse block: {features} due to SyntaxError: {e}")
            continue

        # 使用 ParsedFile 的逻辑提取单元
        pf = ParsedFile(code=code_str, file_path=file_path)
        for unit in pf.units:
            if unit.unit_type in {"function", "method", "class"}:
                cloned_unit = deepcopy(unit)
                cloned_unit.extra["features"] = features
                code_units_with_features.append((features, cloned_unit))
            else:
                logging.debug(f"Skipping non-interface unit of type: {unit.unit_type}")
                
    return code_units_with_features


def generate_full_code_with_feature_comments(units: List[CodeUnit]) -> str:
    code_blocks = []
    for unit in units:
        needs_comment = unit.unit_type in {"class", "method", "function"}
        feature_info = unit.extra.get("features")
        comment = f"# Features: {', '.join(feature_info)}" if (needs_comment and feature_info) else ""
        block = f"{comment}\n{unit.unparse()}" if comment else unit.unparse()
        code_blocks.append(block)
    return "\n\n".join(code_blocks)
