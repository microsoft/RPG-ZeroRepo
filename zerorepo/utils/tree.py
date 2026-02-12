import json, re, copy, os, random
from typing import List, Dict, Union, Iterable, Tuple, Any

Tree = Dict[str, Union["Tree", List[str]]]

def prune_tree(feature_tree, sampled_feature_tree, in_place=False):
    def prune_recursive(feature_subtree, sampled_subtree):
        if isinstance(sampled_subtree, list):
            if isinstance(feature_subtree, list):
                for leaf in sampled_subtree:
                    if leaf in feature_subtree:
                        feature_subtree.remove(leaf)
                return feature_subtree if feature_subtree else None
            else:
                return feature_subtree
        elif isinstance(sampled_subtree, dict):
            if isinstance(feature_subtree, dict):
                keys_to_delete = []
                for key in sampled_subtree:
                    if key in feature_subtree:
                        result = prune_recursive(feature_subtree[key], sampled_subtree[key])
                        if result is None:
                            keys_to_delete.append(key)
                for key in keys_to_delete:
                    del feature_subtree[key]
                return feature_subtree if feature_subtree else None
            else:
                return feature_subtree
        return feature_subtree

    # 决定是否复制
    tree_to_prune = feature_tree if in_place else copy.deepcopy(feature_tree)
    prune_recursive(tree_to_prune, sampled_feature_tree)
    return tree_to_prune

def omit_leaf_subtree(subtree):
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
    
    
def filter_tree_by_leaf_nodes(tree, target_leaf_names):
    """
    从原始树中过滤出包含指定叶子节点的子树结构
    :param tree: 原始嵌套字典树
    :param target_leaf_names: 需要保留的叶子节点名称（列表）
    :return: 一个新树，仅包含与目标叶子节点有关的路径
    """
    if isinstance(tree, dict):
        new_subtree = {}
        for key, value in tree.items():
            if isinstance(value, dict):
                filtered = filter_tree_by_leaf_nodes(value, target_leaf_names)
                if filtered:
                    new_subtree[key] = filtered
            elif isinstance(value, list):
                filtered_items = [item for item in value if item in target_leaf_names]
                if filtered_items:
                    new_subtree[key] = filtered_items
        return new_subtree if new_subtree else None
    return None

def find_leaf_paths_by_node(tree, target_leaf_names, prefix=''):
    """
    在树中查找指定叶子节点名的完整路径
    :param tree: 原始树（嵌套字典）
    :param target_leaf_names: 目标叶子名称列表（字符串）
    :param prefix: 当前路径前缀（用于递归）
    :return: 包含路径字符串的列表
    """
    matches = []

    if isinstance(tree, dict):
        for key, value in tree.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if not value and key in target_leaf_names:
                    matches.append(new_prefix)
                else:
                    matches.extend(find_leaf_paths_by_node(value, target_leaf_names, new_prefix))
            elif isinstance(value, list):
                for item in value:
                    if item in target_leaf_names:
                        matches.append(f"{new_prefix}/{item}")
            else:
                if value in target_leaf_names:
                    matches.append(new_prefix)

    return matches


def pre_order_traversal_to_list(feature_dict):
    result = []
    if not isinstance(feature_dict, dict):
        return result

    for key, value in feature_dict.items():
        result.append(key)
                
        if isinstance(value, dict):
            result.extend(pre_order_traversal_to_list(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):  
                    result.extend(pre_order_traversal_to_list(item)) 
                elif isinstance(item, (str, int)):  
                    result.append(item) 
    
    return result


def filter_tree(tree, selected_features):
    if isinstance(tree, dict):
        filtered = {}
        for key, value in tree.items():
            sub_tree = filter_tree(value, selected_features)
            if sub_tree:
                filtered[key] = sub_tree
        return filtered if filtered else None

    elif isinstance(tree, list):
        return [item for item in tree if item in selected_features] or None

    return None


def flatten_feature_tree_paths(feature_dict, path=""):
    """
    Traverse the feature tree in pre-order and return a flat list of all feature paths.
    
    Each path is represented as a string in the format 'parent/child/.../leaf'.
    """
    result = []
    if not isinstance(feature_dict, dict):
        return result

    for key, value in feature_dict.items():
        current_path = f"{path}/{key}" if path else key

        if isinstance(value, dict):
            result.append(current_path)
            result.extend(flatten_feature_tree_paths(value, current_path))

        elif isinstance(value, list):
            result.append(current_path)
            result.extend(f"{current_path}/{item}" for item in value)

    return result


def remove_paths(tree, paths, inplace=False):
    if not inplace:
        tree = copy.deepcopy(tree)  # 返回新对象

    def delete_path(node, path_parts):
        if not path_parts:
            return False

        key = path_parts[0]

        if isinstance(node, dict):
            if key not in node:
                return False

            # 删除列表中的元素
            if len(path_parts) == 2 and isinstance(node[key], list):
                value_to_remove = path_parts[1]
                if value_to_remove in node[key]:
                    node[key].remove(value_to_remove)
                    if not node[key]:
                        del node[key]
                    return True

            # 删除整个键
            if len(path_parts) == 1:
                del node[key]
                return True

            # 递归子结构
            child_deleted = delete_path(node[key], path_parts[1:])
            if isinstance(node[key], dict) and not node[key]:
                del node[key]
                return True
            elif isinstance(node[key], list) and not node[key]:
                del node[key]
                return True
            return child_deleted

        elif isinstance(node, list):
            deleted = False
            for item in node[:]:
                if isinstance(item, dict) and delete_path(item, path_parts):
                    deleted = True
                    if not item:
                        node.remove(item)
            return deleted

        return False

    for path in paths:
        if not path or not isinstance(path, str):
            continue
        path_parts = [p for p in path.split('/') if p]
        delete_path(tree, path_parts)

    return tree

def get_all_leaf_paths(tree, prefix=''):
    """
    获取所有叶子节点的完整路径（包括：
    - 空 dict 的 key
    - 空 list 的 key
    - list 中的元素
    ）
    """
    paths = []

    if isinstance(tree, dict):
        if not tree:
            # 空字典作为叶子节点
            paths.append(prefix)
        else:
            for key, value in tree.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                paths.extend(get_all_leaf_paths(value, new_prefix))
    elif isinstance(tree, list):
        if not tree:
            # 空列表作为叶子节点
            paths.append(prefix)
        else:
            for item in tree:
                path = f"{prefix}/{item}" if prefix else item
                paths.append(path)
    else:
        # 非 dict/list，直接作为叶子路径
        paths.append(prefix)

    return paths



def convert_leaves_to_list(tree):
    if isinstance(tree, dict):
        return {k: convert_leaves_to_list(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        if not tree:
            return {}   # 空列表替换为空字典
        else:
            # 如果有内容，可根据需求处理，这里保持原样
            return tree
    else:
        return tree

def _collapse_leaf_dicts(node: Union[Tree, List[str]]) -> Union[Tree, List[str]]:
    """
    递归遍历，把「所有 value 都是空 list」的 dict 折叠成 list(keys)。
    """
    if isinstance(node, dict):
        if not node:
            return {}  # 空 dict 保持空 dict
        collapsed = {k: _collapse_leaf_dicts(v) for k, v in node.items()}
        if all(isinstance(v, list) and len(v) == 0 for v in collapsed.values()):
            return list(collapsed.keys())
        return collapsed
    elif isinstance(node, list):
        # 如果 list 中可能有 dict 也递归（可选逻辑）
        return [ _collapse_leaf_dicts(v) for v in node ]
    else:
        return node

def split_path(path: str, delimiters: Union[str, Iterable[str]]) -> List[str]:
    """Split a path string by one or more delimiters."""
    if isinstance(delimiters, str):
        delimiters = [delimiters]
    # 构造正则 pattern
    pattern = '|'.join(map(re.escape, delimiters))
    parts = [p.strip() for p in re.split(pattern, path) if p.strip()]
    return parts


def insert_path(tree: Tree, path: str, delimiters: Union[str, Iterable[str]] = "/") -> None:
    """Insert a path into the tree structure, supporting multiple delimiters."""
    if isinstance(delimiters, str):
        parts = [p.strip() for p in path.split(delimiters) if p.strip()]
    else:
        parts = split_path(path, delimiters)
    parent, key_in_parent = None, None
    node: Union[Tree, List[str]] = tree
    i = 0

    while i < len(parts):
        part, last = parts[i], i == len(parts) - 1

        # ---------- dict ----------
        if isinstance(node, dict):
            # 大小写不敏感匹配已存在 key
            mk = next((k for k in node if k.lower() == part.lower()), None)

            if last:
                if mk is None:
                    node[part] = []
                elif isinstance(node[mk], dict):
                    # 已存在 dict，什么都不做
                    pass
                elif isinstance(node[mk], list):
                    # 已是 list，什么都不做
                    pass
                else:
                    # 非 dict/list，自动升级为 list
                    node[mk] = []
                break
            else:
                if mk is None:
                    node[part] = {}
                    mk = part
                elif isinstance(node[mk], list):
                    # list → dict 升级，保留元素为 {item: []}
                    node[mk] = {x: [] for x in node[mk]}
                elif not isinstance(node[mk], dict):
                    # 自动升级为 dict
                    node[mk] = {}
                # 进入下一层
                parent, key_in_parent = node, mk
                node = node[mk]
                i += 1
                continue

        # ---------- list ----------
        elif isinstance(node, list):
            if last:
                if part.lower() not in (x.lower() for x in node):
                    node.append(part)
                break
            else:
                # list → dict 升级
                upgraded = {x: [] for x in node}
                parent[key_in_parent] = upgraded
                node = upgraded
                # 不递增 i，本轮继续处理 part
                continue

        else:
            # 非 dict/list 节点，自动升级为 dict 并继续
            upgraded = {}
            parent[key_in_parent] = upgraded
            node = upgraded
            # 不递增 i，本轮继续处理 part
            continue
        
def apply_changes(tree: Tree,
                  changes: Union[str, Iterable[str]],
                  *,
                  delimiters: Union[str, Iterable[str]] = "/",
                  inplace: bool = True,
                  auto_collapse: bool = True) -> Tree:
    """
    批量插入路径并可选地做叶子规整。

    Args:
        tree: 原树；`inplace=False` 时自动 deep copy。
        changes: 单条或多条路径。
        delimiters: 路径分隔符，支持单个或多个。
        inplace: False → 返回新对象；True → 就地修改并返回同一对象。
        auto_collapse: True → 把纯叶子 dict 折叠成 list。
    """
    target = tree if inplace else copy.deepcopy(tree)
    # 逐条插入
    if isinstance(changes, str):
        changes = [changes]
    for p in changes:
        insert_path(target, p, delimiters)
    # 叶子规整（可选）
    if auto_collapse:
        collapsed = _collapse_leaf_dicts(target)
        # 若 inplace=True，要把折叠结果写回原 dict
        if inplace:
            # 清空原 dict，再就地写入 collapsed（保持同一引用）
            tree.clear()
            tree.update(collapsed)
            return tree
        else:
            return collapsed
    return target


def extract_feature_list(skeleton: dict) -> List[str]:
    """
    遍历嵌套结构，从所有叶子节点为 list 的位置提取其中的字符串元素，汇总成一个 feature 列表。
    """
    features = []

    def traverse(node):
        if isinstance(node, dict):
            for value in node.values():
                traverse(value)
        elif isinstance(node, list):
            features.extend([item for item in node if isinstance(item, str)])

    traverse(skeleton)
    return features

def replace_leaf_lists_with_empty(tree):
    """
    递归地将所有值为 list 的叶子节点替换为空字典 {}
    """
    if isinstance(tree, dict):
        new_tree = {}
        for key, value in tree.items():
            if isinstance(value, list):
                new_tree[key] = {}
            elif value is None:
                new_tree[key] = []
            elif isinstance(value, dict):
                new_tree[key] = replace_leaf_lists_with_empty(value)
            else:
                # 非法结构可视需求处理：忽略 or 抛错
                new_tree[key] = value  # 或者直接跳过
        return new_tree
    else:
        return tree
    
def omit_features_as_string(skeleton: dict) -> str:
    """
    将 skeleton 中所有叶子节点（列表或字符串）隐藏为 "...",
    保持结构不变，并以 JSON 字符串形式返回。
    """
    def mask_leaf_nodes(tree):
        if isinstance(tree, dict):
            return {
                key: mask_leaf_nodes(value)
                if isinstance(value, dict) else "..."
                for key, value in tree.items()
            }
        elif isinstance(tree, list):
            return "..."
        else:
            return "..."

    masked = mask_leaf_nodes(skeleton)
    return json.dumps(masked, indent=4)

def extract_all_paths(tree: Dict[str, Union[dict, list, None]], prefix: str = "") -> List[str]:
    """
    递归提取嵌套 skeleton 结构中所有文件路径（以 / 分隔）。
    支持 value 为 None（空文件节点）或 list（特征挂载）。
    """
    paths = []

    for name, value in tree.items():
        current_path = f"{prefix}/{name}" if prefix else name

        if isinstance(value, dict) and value:
            # 继续深入子目录
            paths.extend(extract_all_paths(value, current_path))
        elif not value or isinstance(value, list):
            # 当前是文件节点（文件名 -> None 或文件名 -> [feature1, ...]）
            paths.append(current_path)

    return paths


def find_leaf_paths_by_node(tree, target_leaf_names, prefix=''):
    """
    Find the full path(s) of specified leaf node names in a tree.

    :param tree: The original tree (nested dictionaries)
    :param target_leaf_names: List of target leaf names (strings)
    :param prefix: Current path prefix (for recursion)
    :return: A list of path strings
    """
    matches = []

    if isinstance(tree, dict):
        for key, value in tree.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if not value and key in target_leaf_names:
                    matches.append(new_prefix)
                else:
                    matches.extend(find_leaf_paths_by_node(value, target_leaf_names, new_prefix))
            elif isinstance(value, list):
                for item in value:
                    if item in target_leaf_names:
                        matches.append(f"{new_prefix}/{item}")
            else:
                if value in target_leaf_names:
                    matches.append(new_prefix)

    return matches

def extract_leaf_nodes(tree):
    """
    Extract the names of all leaf nodes in a tree (without paths).
    - A key whose value is {} is treated as a leaf.
    - A key whose value is [] is treated as a leaf.
    - Elements inside a non-empty list are treated as leaves.
    """
    leaf_names = set()

    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                if not value:  # 空字典是叶子
                    leaf_names.add(key)
                else:
                    leaf_names.update(extract_leaf_nodes(value))
            elif isinstance(value, list):
                if not value:  # 空列表，key 本身是叶子
                    leaf_names.add(key)
                else:
                    leaf_names.update(value)
            else:
                leaf_names.add(value)

    return list(leaf_names)


# ============================================================
# Functions migrated from rpg_encoder/rpg_parsing/util.py
# ============================================================
def extract_subtree_diff(before: List[Dict], after: List[Dict]) -> Dict:
    """
    Extract the diff between two refactored_tree lists (new/modified subtrees).
    Both before and after follow the structure: [{ "name": ..., "refactored_subtree": {...} }].
    Returns: { area_name: diff_subtree }
    """
    diff = {}

    def dict_diff(before_dict: Dict, after_dict: Dict) -> Dict:
        """Recursively compute dictionary diffs, keeping only nodes that are new or modified in `after`."""
        changes = {}
        for k, v in after_dict.items():
            if k not in before_dict:
                changes[k] = v  # new node
            else:
                # recursively compare sub-structures
                if isinstance(v, dict) and isinstance(before_dict[k], dict):
                    sub_diff = dict_diff(before_dict[k], v)
                    if sub_diff:
                        changes[k] = sub_diff
                elif v != before_dict[k]:
                    changes[k] = v  # content changed
        return changes

    for after_node in after:
        name = after_node.get("name")
        after_subtree = after_node.get("refactored_subtree", {})
        before_node = next((n for n in before if n.get("name") == name), None)
        before_subtree = before_node.get("refactored_subtree", {}) if before_node else {}

        area_diff = dict_diff(before_subtree, after_subtree)
        if area_diff:
            diff[name] = area_diff

    return diff


def transfer_parsed_tree(
    input_tree: Dict
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Transform a parsed feature tree into:
    1) format_tree:      { file_summary: [features...] }
    2) feature_to_files: { feature: [file paths...] }

    - Merge all nested function/class-level descriptions into the file-level node.
    - Automatically deduplicate feature text.
    """

    def collect_texts(value: Union[str, List, Dict]) -> List[str]:
        """Recursively collect all text leaves from any nested structure."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            result = []
            for v in value:
                result.extend(collect_texts(v))
            return result
        elif isinstance(value, dict):
            result = []
            for v in value.values():
                result.extend(collect_texts(v))
            return result
        else:
            return [str(value)]

    format_tree: Dict[str, List[str]] = {}
    feature_to_files: Dict[str, List[str]] = {}

    for file_path, file_tree in input_tree.items():
        # Top-level file summary (or default to file name)
        file_summary = file_tree.get("_file_summary_", os.path.basename(file_path).replace(".py", ""))

        # Collect all child-node texts
        all_texts = []
        for key, value in file_tree.items():
            if key == "_file_summary_":
                continue
            all_texts.extend(collect_texts(value))

        # Deduplicate + sort
        deduped_texts = sorted(set(all_texts))
        format_tree[file_summary] = deduped_texts

        # Reverse mapping
        for feature in deduped_texts:
            feature_to_files.setdefault(feature, []).append(file_path)

    return format_tree, feature_to_files


def format_parsed_tree(input_tree: Dict, omit_full_leaf_nodes: bool = False, max_features: int = 2) -> str:
    """
    Format the parsed feature tree into a condensed, human-readable JSON structure.

    This version reuses `transfer_parsed_tree` to build the base mapping,
    and then applies optional truncation for readability.
    """
    # Reuse existing logic
    format_tree, _ = transfer_parsed_tree(input_tree)

    # Optionally truncate long feature lists
    for key, features in format_tree.items():
        if omit_full_leaf_nodes and len(features) > 2:
            sampled = random.sample(features, min(max_features, len(features)))
            format_tree[key] = sampled + ["..."]

    return json.dumps(format_tree, ensure_ascii=False, separators=(',', ':'))


def get_rpg_info(
    rpg_tree: List[Dict],
    omit_leaf_nodes: bool = True,
    sample_size: int = 2,
    indent: int | None = None,   # pass None to save tokens
) -> str:
    """
    Get a summarized string representation of an RPG tree structure.
    """
    def _prune(node: Any) -> Any:
        # leaf: features list
        if isinstance(node, list):
            if not omit_leaf_nodes:
                return node
            if sample_size <= 0:
                return {}  # hide leaf features but keep a placeholder indicating the leaf exists
            if len(node) > sample_size:
                return random.sample(node, sample_size) + ["..."]
            return node

        # internal: dict
        if isinstance(node, dict):
            if not node:
                return {}  # empty dict is treated as an empty leaf

            out: Dict[str, Any] = {}
            leaf_keys: List[str] = []

            for k, v in node.items():
                pv = _prune(v)

                # pv is an empty-leaf placeholder
                if isinstance(pv, dict) and not pv:
                    leaf_keys.append(k)
                else:
                    out[k] = pv

            # ✅ All are empty leaves: compress to list(keys)
            if not out and leaf_keys:
                return leaf_keys

            # (Optional) Mixed case: both non-leaves and empty leaves
            # To save tokens, merge empty-leaf keys into a single "_" field
            if leaf_keys:
                out["_"] = leaf_keys

            return out

        # other primitive
        return node

    rpg_info = {}
    for sub_tree in rpg_tree:
        name = sub_tree.get("name")
        tree = sub_tree.get("refactored_subtree", {})
        rpg_info[name] = _prune(tree)

    if indent is None:
        return json.dumps(rpg_info, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(rpg_info, ensure_ascii=False, indent=indent)


def iterative_by_folder(parsed_tree: Dict) -> Dict[str, List[str]]:
    """
    Group file paths by their parent folder.
    Returns: { folder_path: [file_paths...] }
    """
    file_paths = list(parsed_tree.keys())

    grouped: Dict[str, List[str]] = {}

    for p in file_paths:
        p_norm = p.rstrip("/")
        parent_dir = os.path.dirname(p_norm)
        folder = parent_dir if parent_dir else "(root)"
        grouped.setdefault(folder, []).append(p)

    return grouped