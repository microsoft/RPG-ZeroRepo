
import os
import re
from typing import Dict, List, Union, Iterable, Tuple, Any
from copy import deepcopy
import copy
import random
import json
Tree = Dict[str, Union["Tree", List[str]]]


def extract_subtree_diff(before: List[Dict], after: List[Dict]) -> Dict:
    diff = {}

    def dict_diff(before_dict: Dict, after_dict: Dict) -> Dict:
       
        changes = {}
        for k, v in after_dict.items():
            if k not in before_dict:
                changes[k] = v 
            else:
                if isinstance(v, dict) and isinstance(before_dict[k], dict):
                    sub_diff = dict_diff(before_dict[k], v)
                    if sub_diff:
                        changes[k] = sub_diff
                elif v != before_dict[k]:
                    changes[k] = v 
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
    Transform parsed feature tree into:
    1. format_tree:  { file_summary: [features...] }
    2. feature_to_files: { feature: [file paths...] }

    - Merges all nested function/class-level descriptions into their file-level node.
    - Automatically deduplicates feature text.
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
        file_summary = file_tree.get("_file_summary_", os.path.basename(file_path).replace(".py", ""))

        # 收集所有子节点文本
        all_texts = []
        for key, value in file_tree.items():
            if key == "_file_summary_":
                continue
            all_texts.extend(collect_texts(value))
            
        deduped_texts = sorted(set(all_texts))
        format_tree[file_summary] = deduped_texts

        for feature in deduped_texts:
            feature_to_files.setdefault(feature, []).append(file_path)

    return format_tree, feature_to_files


def format_parsed_tree(input_tree: Dict, omit_full_leaf_nodes: bool = False, max_features: int=2) -> str:
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
    indent: int | None = None, 
) -> str:
    def _prune(node: Any) -> Any:
        # leaf: features list
        if isinstance(node, list):
            if not omit_leaf_nodes:
                return node
            if sample_size <= 0:
                return {} 
            if len(node) > sample_size:
                return random.sample(node, sample_size) + ["..."]
            return node

        # internal: dict
        if isinstance(node, dict):
            if not node:
                return {} 

            out: Dict[str, Any] = {}
            leaf_keys: List[str] = []

            for k, v in node.items():
                pv = _prune(v)

                if isinstance(pv, dict) and not pv:
                    leaf_keys.append(k)
                else:
                    out[k] = pv

            if not out and leaf_keys:
                return leaf_keys

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


def split_path(path: str, delimiters: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(delimiters, str):
        delimiters = [delimiters]
  
    pattern = '|'.join(map(re.escape, delimiters))
    parts = [p.strip() for p in re.split(pattern, path) if p.strip()]
    return parts

def convert_leaves_to_list(tree):
    if isinstance(tree, dict):
        return {k: convert_leaves_to_list(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        if not tree:
            return {}   
        else:
            return tree
    else:
        return tree

def _collapse_leaf_dicts(node: Union[Tree, List[str]]) -> Union[Tree, List[str]]:
    if isinstance(node, dict):
        if not node:
            return {} 
        collapsed = {k: _collapse_leaf_dicts(v) for k, v in node.items()}
        if all(isinstance(v, list) and len(v) == 0 for v in collapsed.values()):
            return list(collapsed.keys())
        return collapsed
    elif isinstance(node, list):
        return [ _collapse_leaf_dicts(v) for v in node ]
    else:
        return node

def insert_path(tree: Tree, path: str, delimiters: Union[str, Iterable[str]] = ["/", " -> "]) -> None:
    parts = split_path(path, delimiters)
    parent, key_in_parent = None, None
    node = tree
    i = 0

    while i < len(parts):
        part, last = parts[i], i == len(parts) - 1

        if isinstance(node, dict):
            mk = next((k for k in node if k.lower() == part.lower()), None)

            if last:
                if mk is None:
                    node[part] = []
                elif isinstance(node[mk], dict) or isinstance(node[mk], list):
                    pass
                else:
                    node[mk] = []
                break
            else:
                if mk is None:
                    node[part] = {}
                    mk = part
                elif isinstance(node[mk], list):
                    node[mk] = {x: [] for x in node[mk]}
                elif not isinstance(node[mk], dict):
                    node[mk] = {}
                parent, key_in_parent = node, mk
                node = node[mk]
                i += 1
                continue

        elif isinstance(node, list):
            if last:
                if part.lower() not in (x.lower() for x in node):
                    node.append(part)
                break
            else:
                upgraded = {x: [] for x in node}
                parent[key_in_parent] = upgraded
                node = upgraded
                continue

        else:
            upgraded = {}
            parent[key_in_parent] = upgraded
            node = upgraded
            continue

        
def apply_changes(tree: Tree,
                  changes: Union[str, Iterable[str]],
                  *,
                  delimiters: Union[str, Iterable[str]] = ["/", " -> "],
                  inplace: bool = True,
                  auto_collapse: bool = True) -> Tree:
    target = tree if inplace else copy.deepcopy(tree)

    if isinstance(changes, str):
        changes = [changes]
    for p in changes:
        insert_path(target, p, delimiters)

    if auto_collapse:
        collapsed = _collapse_leaf_dicts(target)
        if inplace:
            tree.clear()
            tree.update(collapsed)
            return tree
        else:
            return collapsed
    return target


def iterative_by_folder(
    parsed_tree: Dict
):
    file_paths = list(parsed_tree.keys())
    
    grouped: Dict[str, List[str]] = {}

    for p in file_paths:
        p_norm = p.rstrip("/")
        parent_dir = os.path.dirname(p_norm)
        folder = parent_dir if parent_dir else "(root)"
        grouped.setdefault(folder, []).append(p)

    return grouped