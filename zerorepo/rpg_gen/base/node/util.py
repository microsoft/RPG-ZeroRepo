from .node import DirectoryNode, FileNode, RepoNode
from typing import Dict, List, Optional, Union, Callable
from pathlib import PurePosixPath

def is_same_logical_path(path1: str, path2: str) -> bool:
    norm1 = str(PurePosixPath(path1))
    norm2 = str(PurePosixPath(path2))
    return norm1 == norm2

def build_tree_from_file_map(file_map: Dict[str, str], repo_root_name: str = "root") -> DirectoryNode:
    """
    Builds an initial RepoNode tree from a file_map (path: code).
    FileNodes are created with empty feature_paths initially.
    """
    root = DirectoryNode(name=repo_root_name, path=".") # Assuming root path is '.'

    sorted_paths = sorted(file_map.keys())

    for file_path in sorted_paths:
        path_parts = file_path.strip("/").split("/")
        file_name = path_parts[-1]
        dir_path_str = "/".join(path_parts[:-1])

        current_parent_node = root
        if dir_path_str: # If the file is not in the root directory
            current_accum_path = ""
            for dir_name in dir_path_str.split("/"):
                if not dir_name: continue
                
                # Determine the full path of the child directory
                if current_accum_path == "" or current_accum_path == ".":
                    child_dir_full_path = dir_name
                else:
                    child_dir_full_path = f"{current_accum_path}/{dir_name}"

                found_child = None
                for child in current_parent_node.children():
                    if child.name == dir_name and child.is_dir and child.path == child_dir_full_path:
                        found_child = child
                        break
                
                if found_child:
                    current_parent_node = found_child
                else:
                    new_dir_node = DirectoryNode(name=dir_name, path=child_dir_full_path)
                    current_parent_node.add_child(new_dir_node)
                    current_parent_node = new_dir_node
                current_accum_path = current_parent_node.path
        
        code = file_map[file_path]
        file_node = FileNode(name=file_name, path=file_path, code=code, feature_paths=[])
        current_parent_node.add_child(file_node)
            
    return root


def extract_all_file_paths_from_tree(node: RepoNode) -> List[str]: # Copied for completeness
    paths = []
    if node.is_file:
        paths.append(node.path) # type: ignore
    elif node.is_dir:
        for child in node.children():
            paths.extend(extract_all_file_paths_from_tree(child))
    return paths

def extract_feature_list_from_tree(node: RepoNode) -> List[str]: # Copied for completeness
    features = []
    if node.is_file and isinstance(node, FileNode): # Check if FileNode

        features.extend(node.feature_paths)
    elif node.is_dir:
        for child in node.children():
            features.extend(extract_feature_list_from_tree(child))
    return list(set(features))


def merge_structure_patch(refined_dict: Dict, original_root: DirectoryNode, file_map: Dict[str, str]) -> None:
    """
    Merge the structure from `refined_dict` into `original_root`.
    - Preserves existing FileNodes and their content.
    - Creates missing directories or placeholder FileNodes if needed.
    """
    def _merge(d: Dict, parent: DirectoryNode, prefix: str):
        for name, value in d.items():
            full_path = f"{prefix}/{name}".strip("/")
            existing = next((c for c in parent.children() if c.name == name), None)

            if isinstance(value, dict):
                if not existing or not existing.is_dir:
                    new_dir = DirectoryNode(name=name, path=full_path)
                    parent.add_child(new_dir)
                    existing = new_dir
                _merge(value, existing, full_path)  # type: ignore
            else:
                if not existing:
                    code = file_map.get(full_path, f"")
                    new_file = FileNode(name=name, path=full_path, code=code, feature_paths=[])
                    parent.add_child(new_file)

    _merge(refined_dict, original_root, "")



def collect_non_test_py_files(root: RepoNode) -> List[FileNode]:
    """
    Traverse the repository tree and collect all .py FileNode objects
    that are not located under any 'tests/' directory.
    """
    result = []

    def dfs(node: RepoNode):
        # Skip anything in or under 'tests/' directories
        normalized_path = node.path.lower()
        if "tests/" in normalized_path or normalized_path == "tests":
            return

        if isinstance(node, FileNode) and node.name.endswith(".py"):
            result.append(node)
        elif isinstance(node, DirectoryNode):
            for child in node.children():
                dfs(child)

    dfs(root)
    return result


def default_filter_include_all(node: RepoNode) -> bool:
    """Default filter: always includes the node."""
    return True


def filter_non_test_py_files(current_node: Union[RepoNode, str]) -> bool:
    """
    Filter function: includes only .py files that are NOT located in 'tests/' directories.
    Directories are included only if they contain relevant files or subdirectories.

    Args:
        current_node: Either a RepoNode object or a string path
    """
    # Handle string path input
    if isinstance(current_node, str):
        normalized_path = current_node.lower()
        # Exclude test directories
        if "tests/" in normalized_path or "/test/" in normalized_path or normalized_path.startswith("test/"):
            return False
        if normalized_path == "tests" or normalized_path == "test":
            return False
        # Only include .py files
        return normalized_path.endswith(".py")

    # Handle RepoNode input
    # 1. Immediately exclude anything under 'tests/'
    normalized_path = current_node.path.lower()

    # Check for both "tests/" within the path and if the path itself is "tests"
    if "tests/" in normalized_path or normalized_path == "tests":
        return False

    # 2. For FileNodes not in 'tests/', only include if it's a .py file
    if isinstance(current_node, FileNode):
        return current_node.name.endswith(".py")

    # 3. For DirectoryNodes not in 'tests/', check their children recursively
    elif isinstance(current_node, DirectoryNode):
        for child in current_node.children():
            # If any child (recursively) is a non-test .py file, then include this directory
            if filter_non_test_py_files(child):
                return True
        return False # If no relevant children, exclude this directory

    return False # Fallback for unexpected node types


def show_project_structure_from_tree(
    node: "RepoNode",
    indent: str = "",
    *,
    skip_root: bool = False,
    root_path: str = ".",
    show_features: bool = True,
    show_leaves_only: bool = True,
    shown: Optional[set] = None,
    filter_func: Callable[["RepoNode"], bool] = default_filter_include_all,
    prune_dirs_by_filter: bool = True,
) -> str:
    """
    Unified tree pretty-printer.

    Features:
    - Optional skip_root behavior (for virtual root like '.')
    - Optional feature display for FileNode.feature_paths
    - Optional leaf-only feature display
    - Filter function support:
        - prune_dirs_by_filter=True  -> strong pruning (if a dir fails filter, skip whole subtree)
        - prune_dirs_by_filter=False -> weak pruning for dirs (dir can be shown even if it fails,
                                       but its children still filtered)
    - 'shown' set to prevent loops / duplicates (useful if underlying structure isn't a strict tree)

    Args:
        node: current RepoNode
        indent: indentation prefix
        skip_root: if True and node.path == root_path, don't print root, only its children
        root_path: virtual root path marker (default '.')
        show_features: if True, print FileNode.feature_paths
        show_leaves_only: if True, only show last component of each feature path
        shown: set of paths already printed
        filter_func: node inclusion predicate
        prune_dirs_by_filter: controls whether dirs are also pruned by filter_func

    Returns:
        str: formatted tree
    """
    if shown is None:
        shown = set()

    def _should_include(n: "RepoNode") -> bool:
        # strong pruning for dirs unless explicitly disabled
        if n.is_dir and not prune_dirs_by_filter:
            return True
        return filter_func(n)

    # Skip already shown paths
    if node.path in shown:
        return ""

    # Special root skipping: don't print root, only eligible children
    if skip_root and node.path == root_path and node.is_dir:
        shown.add(node.path)
        lines: list[str] = []
        children = sorted(node.children(), key=lambda n: (not n.is_dir, n.name))
        for child in children:
            if not _should_include(child):
                continue
            child_str = show_project_structure_from_tree(
                child,
                indent="",
                skip_root=False,
                root_path=root_path,
                show_features=show_features,
                show_leaves_only=show_leaves_only,
                shown=shown,
                filter_func=filter_func,
                prune_dirs_by_filter=prune_dirs_by_filter,
            )
            if child_str:
                lines.append(child_str)
        return "\n".join(lines)

    # Normal inclusion check for this node
    if not _should_include(node):
        return ""

    shown.add(node.path)

    # Build line content
    line = node.name + ("/" if node.is_dir else "")
    if show_features and node.is_file and isinstance(node, FileNode):
        feature_paths = getattr(node, "feature_paths", None)
        if feature_paths:
            if show_leaves_only:
                leaves = [p.split("/")[-1] for p in feature_paths]
                line += f": [{', '.join(leaves)}]"
            else:
                line += f": [{', '.join(feature_paths)}]"

    output_lines = [f"{indent}{line}"]

    # Recurse into children
    if node.is_dir:
        children = sorted(node.children(), key=lambda n: (not n.is_dir, n.name))
        for child in children:
            if not _should_include(child):
                continue
            subtree = show_project_structure_from_tree(
                child,
                indent=indent + "  ",
                skip_root=False,
                root_path=root_path,
                show_features=show_features,
                show_leaves_only=show_leaves_only,
                shown=shown,
                filter_func=filter_func,
                prune_dirs_by_filter=prune_dirs_by_filter,
            )
            if subtree:
                output_lines.append(subtree)

    return "\n".join(output_lines)


def format_feature_tree_with_paths(
    tree: dict,
    node_file_map: Dict[str, List[Dict]],
    *,
    omit_units: bool = False,
    indent: int = 4,
    prefix_path: str = ""
) -> str:
    """
    将 Feature Tree 渲染为缩进形式的字符串树表示，每个节点后附带实现路径标签。
    支持 infra 文件显示，并去掉 .py 后缀。
    """
    from pathlib import Path
    lines: List[str] = []

    def _should_skip(path: str) -> bool:
        if not omit_units:
            return False
        items = node_file_map.get(path, [])
        return items and all(item.get("type") in ("function", "class") for item in items)

    def _tag(path: str) -> str:
        items = node_file_map.get(path, [])
        if not items:
            return ""
        parts = []
        for it in items:
            if it.get("tag") == "infra":
                continue
            t = it.get("type", "file")
            if t in ("function", "class"):
                parts.append(f'{t}: {it["name"]} @ {it["path"]}')
            else:
                parts.append(f'{t}: {it["path"]}')
        return f" [-> {', '.join(parts)}]" if parts else ""

    def _recurse(subtree, depth=0, path_prefix=""):
        pad = " " * (indent * depth)
        if isinstance(subtree, dict):
            for key, child in subtree.items():
                child_path = f"{path_prefix}/{key}" if path_prefix else key
                if _should_skip(child_path):
                    continue
                lines.append(f"{pad}{key}{_tag(child_path)}")
                _recurse(child, depth + 1, child_path)

                for item in node_file_map.get(child_path, []):
                    if item.get("tag") == "infra":
                        fname = Path(item["path"]).stem
                        lines.append(f"{pad}{' ' * indent}{fname} [-> file: {item['path']}]")
        elif isinstance(subtree, list):
            for leaf in subtree:
                leaf_path = f"{path_prefix}/{leaf}" if path_prefix else leaf
                if _should_skip(leaf_path):
                    continue
                lines.append(f"{pad}{leaf}{_tag(leaf_path)}")
                for item in node_file_map.get(leaf_path, []):
                    if item.get("tag") == "infra":
                        fname = Path(item["path"]).stem
                        lines.append(f"{pad}{' ' * indent}{fname} [-> file: {item['path']}]")

    for root_key, root_sub in tree.items():
        if _should_skip(root_key):
            continue
        lines.append(f"{root_key}{_tag(root_key)}")
        _recurse(root_sub, 1, root_key)
        for item in node_file_map.get(root_key, []):
            if item.get("tag") == "infra":
                fname = Path(item["path"]).stem
                lines.append(f"{' ' * indent}{fname} [-> file: {item['path']}]")

    return "\n".join(lines)

def get_min_common_dir(paths: List[str]) -> Optional[str]:
    """
    Return the minimal common directory (not a file) that contains all given paths.
    """
    if not paths:
        return None
    split_paths = [p.split("/") for p in paths]
    min_len = min(len(p) for p in split_paths)
    common = []
    for i in range(min_len):
        tokens = {p[i] for p in split_paths}
        if len(tokens) == 1:
            common.append(tokens.pop())
        else:
            break
    if not common:
        return None
    return "/".join(common)


def path_under_allowed(path: str, allowed_root: Union[str, List[str], None]) -> bool:
    if allowed_root is None:
        return True
    if isinstance(allowed_root, str):
        return path.startswith(allowed_root)
    if isinstance(allowed_root, list):
        return any(path.startswith(root) for root in allowed_root)
    return False
