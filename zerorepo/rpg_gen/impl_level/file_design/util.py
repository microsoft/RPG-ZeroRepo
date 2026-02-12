import os
import logging, re
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple, Union
from zerorepo.rpg_gen.base.node import DirectoryNode, FileNode
from zerorepo.utils.tree import (
    extract_leaf_nodes,
    find_leaf_paths_by_node
)


def extract_all_strings(obj):
    results = []
    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(extract_all_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            results.append(item) if isinstance(item, str) else results.extend(extract_all_strings(item))
    return results
            
def remove_comments_from_json(json_like_str: str) -> str:
    """
    Remove comments from a JSON string, including // single-line comments and /* multi-line comments */.
    """
    # Remove // comments
    no_single_line = re.sub(r"//.*?$", "", json_like_str, flags=re.MULTILINE)
    # Remove /* */ comments
    no_multi_line = re.sub(r"/\*.*?\*/", "", no_single_line, flags=re.DOTALL)
    return no_multi_line

def ensure_directory_path(base_node: DirectoryNode, dir_path: str) -> DirectoryNode:
    parts = dir_path.strip("/").split("/")
    current_node = base_node
    current_accum_path = base_node.path if base_node.path != "." else ""

    for part_name in parts:
        if not part_name:
            continue
        if current_accum_path in ["", "."]:
            child_dir_full_path = part_name
        else:
            child_dir_full_path = f"{current_accum_path}/{part_name}"

        found_child_dir = None
        for child in current_node.children():
            if child.name == part_name and child.is_dir and child.path == child_dir_full_path:
                found_child_dir = child
                break

        if found_child_dir:
            current_node = found_child_dir
        else:
            new_dir_node = DirectoryNode(name=part_name, path=child_dir_full_path)
            current_node.add_child(new_dir_node)
            current_node = new_dir_node
        current_accum_path = current_node.path

    return current_node


def add_file_node_with_features(
    base_node: DirectoryNode,
    file_path: str,
    feature_paths: List[str],
    file_map: Dict[str, str]
) -> None:
    file_path_cleaned = file_path.strip("/")
    parts = file_path_cleaned.split("/")
    if not parts or not parts[-1]:
        logging.error(f"Invalid file path provided for adding node: '{file_path}'")
        return

    file_name = parts[-1]
    dir_path_str = "/".join(parts[:-1])
    parent_dir_node = ensure_directory_path(base_node, dir_path_str)

    existing_file_node: Optional[FileNode] = None
    child_idx_to_remove = -1
    for idx, child in enumerate(parent_dir_node.children()):
        if child.path == file_path_cleaned:
            if child.is_file and isinstance(child, FileNode):
                existing_file_node = child
                break
            else:
                logging.error(f"Path conflict: '{file_path_cleaned}' is not a FileNode. Replacing.")
                child_idx_to_remove = idx
                break

    if child_idx_to_remove != -1:
        parent_dir_node._children.pop(child_idx_to_remove)  # type: ignore

    if existing_file_node:
        current_features = set(existing_file_node.feature_paths)
        current_features.update(feature_paths)
        existing_file_node.feature_paths = sorted(list(current_features))
    else:
        code_content = file_map.get(file_path_cleaned, f"")
        new_file_node = FileNode(name=file_name, path=file_path_cleaned, code=code_content, feature_paths=sorted(set(feature_paths)))
        parent_dir_node.add_child(new_file_node)


def convert_dict_to_repo_node_tree(
    skeleton_dict: Dict,
    file_content_map_for_code: Dict[str, str],
    root_name: str = "dict_converted_root"
) -> DirectoryNode:
    new_root_node = DirectoryNode(name=root_name, path=".")

    paths_to_process: List[Tuple[str, Union[List[str], Dict]]] = []

    def _collect_paths(current_dict: Dict, current_prefix: str = ""):
        for key, value in current_dict.items():
            processed_key = key.strip("/")
            full_path = f"{current_prefix}/{processed_key}" if current_prefix else processed_key
            paths_to_process.append((full_path, value))
            if isinstance(value, dict):
                _collect_paths(value, full_path)

    _collect_paths(skeleton_dict)
    paths_to_process.sort(key=lambda x: (len(x[0].split("/")), x[0]))

    for path_str, value in paths_to_process:
        if isinstance(value, dict):
            ensure_directory_path(new_root_node, path_str)
        elif isinstance(value, list):
            if len(value) == 0:
                ensure_directory_path(new_root_node, path_str)
            else:
                add_file_node_with_features(new_root_node, path_str, value, file_content_map_for_code)
        else:
            add_file_node_with_features(new_root_node, path_str, [], file_content_map_for_code)

    return new_root_node


def process_grouping_assignments(
    assignments: Dict[str, Union[str, List[str]]],
    top_paths: List[str],
    subtree_dict: Dict
) -> Tuple[List[Tuple[str, List[str]]], str, bool]:
    known_leaf_names = extract_leaf_nodes(subtree_dict)
    processed = []
    skipped_files = []
    skipped_features = {}

    BANNED_PATTERNS = {"other", "misc", "temp", "unknown", "part", "etc"}
    def contains_banned_path_component(path: str, banned_keywords: set) -> bool:
        parts = path.lower().split(os.sep)
        return any(k == part for part in parts for k in banned_keywords)
    bad_paths = [
        file_path for file_path in assignments.keys()
        if contains_banned_path_component(file_path, BANNED_PATTERNS)
    ]

    if bad_paths:
        msg_lines = [
            "Invalid file paths detected due to the presence of generic or ambiguous keywords.",
            f"Keywords considered invalid: {sorted(BANNED_PATTERNS)}",
            "Affected file paths:"
        ]
        for p in sorted(bad_paths):
            msg_lines.append(f"  - {p}")
        msg_lines.append("Please rename or reorgnize these paths to be meaningful and specific before proceeding.")
        return [], "\n".join(msg_lines), False
        
    bad_groups = detect_incremental_patterns(list(assignments.keys()))
    if bad_groups:
        msg_lines = [
            "Detected problematic incremental naming patterns in file assignments. Please refactor the file names to use clear, meaningful names.",
            "Details of detected groups:"
        ]
        for prefix, files in bad_groups.items():
            msg_lines.append(f"  Prefix: {prefix}")
            for f in files:
                msg_lines.append(f"    {f}")
        msg_lines.append("Assignment processing was aborted due to these naming issues.")
        return [], "\n".join(msg_lines), False

    for file_path, assigned in assignments.items():
        if not any(file_path.startswith(p) for p in top_paths):
            skipped_files.append(file_path)
            continue
        
        leaf_names = assigned if isinstance(assigned, list) else [assigned]
        valid_names = [name for name in leaf_names if name in known_leaf_names]
        invalid_names = [name for name in leaf_names if name not in known_leaf_names]

        if invalid_names:
            # Build fuzzy match suggestions
            fuzzy_suggestions = {}
            for name in invalid_names:
                matches = get_close_matches(name, known_leaf_names, n=1, cutoff=0.2)
                fuzzy_suggestions[name] = matches[0] if matches else None
            skipped_features[file_path] = (invalid_names, fuzzy_suggestions)

        if not valid_names:
            skipped_files.append(file_path)
            continue

        feature_paths = find_leaf_paths_by_node(subtree_dict, valid_names)
        if feature_paths:
            # add_file_node_with_features(repo_root, file_path, feature_paths, file_map)
            processed.append((file_path, feature_paths))

    if not processed:
        msg_lines = [
            "No valid file-feature pairs were found or processed. Details:",
        ]
        if skipped_files:
            msg_lines.append(f"- Skipped files (invalid path or no valid features): {sorted(skipped_files)}")
        if skipped_features:
            msg_lines.append("- Skipped features by file (not found in known subtree, fuzzy suggestions provided):")
            for fp, (feats, suggestions) in skipped_features.items():
                msg_lines.append(f"  * {fp}:")
                for feat in feats:
                    suggestion = suggestions.get(feat)
                    if suggestion:
                        msg_lines.append(f"      {feat} -> Did you mean: {suggestion}?")
                    else:
                        msg_lines.append(f"      {feat} -> No close match found")
        msg_lines.append("Please check your input format and ensure it matches the expected structure.")
        msg = "\n".join(msg_lines)
        return [], msg, False

    msg_lines = []
    if skipped_files:
        msg_lines.append(f"[Info] Skipped files with no valid assignments: {sorted(skipped_files)}")
    if skipped_features:
        msg_lines.append("[Info] Skipped features not found in known subtree (fuzzy suggestions provided):")
        for fp, (feats, suggestions) in skipped_features.items():
            msg_lines.append(f"  * {fp}:")
            for feat in feats:
                suggestion = suggestions.get(feat)
                if suggestion:
                    msg_lines.append(f"      {feat} -> Did you mean: {suggestion}?")
                else:
                    msg_lines.append(f"      {feat} -> No close match found")

    msg = "\n".join(msg_lines) if msg_lines else ""
    return processed, msg, True

def convert_raw_skeleton_to_repo_node_tree(
    tree_dict: Dict[str, Optional[Union[Dict, list, str]]],
    file_map: Dict[str, str],
    root_name: str = "project_root"
) -> DirectoryNode:
    root = DirectoryNode(name=root_name, path=".")

    def walk(subtree: Dict[str, Optional[Union[Dict, list, str]]], parent: DirectoryNode, parent_path: str):
        for name, value in subtree.items():
            current_path = f"{parent_path}/{name}" if parent_path not in ["", "."] else name

            if value is None:
                # ✅ This is a file
                code_content = file_map.get(current_path, "")
                file_node = FileNode(name=name, path=current_path, code=code_content)
                parent.add_child(file_node)
            else:
                # ✅ Treat everything else as a directory
                dir_node = DirectoryNode(name=name, path=current_path)
                parent.add_child(dir_node)
                if isinstance(value, dict):
                    walk(value, dir_node, current_path)

    walk(tree_dict, root, ".")
    return root


import re
from collections import defaultdict

def detect_incremental_patterns(file_list):
    """
    Detect file groups that share a prefix and differ only by a trailing _<digit|letter>.py or <digit|letter>.py
    """
    pattern = re.compile(r"^(.*?)(?:_)?([0-9a-zA-Z])\.py$")
    groups = defaultdict(list)

    for file_name in file_list:
        match = pattern.match(file_name)
        if match:
            prefix = match.group(1)
            groups[prefix].append(file_name)

    bad_groups = {prefix: names for prefix, names in groups.items() if len(names) > 1}
    return bad_groups