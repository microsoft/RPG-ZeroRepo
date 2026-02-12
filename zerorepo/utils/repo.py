from typing import List, Dict, Callable, Union
import os
from pathlib import Path
from collections import Counter
import re
from pathlib import PurePosixPath
from zerorepo.rpg_gen.base.unit import CodeUnit
from zerorepo.rpg_gen.base.node import RepoSkeleton, FileNode, DirectoryNode

def backtrack_to_lcp_node(
    paths: List[str], 
    min_support: float = 0.6
) -> List[str]:
    """
    Given a list of paths, this function finds the longest common prefix (LCP) directories
    and returns a list of potential common parent directories. The result contains only those
    directories that have support greater than or equal to min_support.

    Args:
        paths: A list of file paths to be processed.
        min_support: The minimum support threshold (as a fraction of the number of paths) required
                      for a parent directory to be considered as the common parent.
    
    Returns:
        A list of the longest common parent directory paths.
    """
    if not paths:
        return []

    # Step 1: Extract parent directories of the paths
    parent_dirs = [str(Path(p).parent) for p in paths]
    parent_counter = Counter(parent_dirs)

    # Find the most common parent directory
    most_common_parent, count = parent_counter.most_common(1)[0]

    result = []

    # Step 2: If all paths have the same parent, return it as the common parent
    if count == len(paths):
        result.append(most_common_parent)

    # Step 3: If most paths have a common parent directory above a certain support threshold, return it
    elif count / len(paths) >= min_support:
        result.append(most_common_parent)

    # Step 4: Backtrack and find common prefixes among paths
    split_paths = [Path(p).parts for p in paths]
    max_len = min(len(p) for p in split_paths)  # Get the minimum path length

    prefix_candidates = []
    for i in range(max_len):
        tokens = [p[i] for p in split_paths]
        most_common_token, count = Counter(tokens).most_common(1)[0]

        # If the token is sufficiently common, add it to the prefix
        if count / len(paths) >= min_support:
            prefix_candidates.append(most_common_token)
        else:
            break  # Stop if the token is not sufficiently common across paths

    # If a prefix was found, create the path for the common prefix
    if prefix_candidates:
        lcp_path = str(Path(*prefix_candidates))
        result.append(lcp_path)

    return result

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\.[a-zA-Z0-9]+$", "", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[/_.\-:]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def is_test_file(nid):
    # input node id (e.g., 'tests/_core.py:test') and output whether it belongs to a test file
    file_path = nid.split(':')[0]
    word_list = re.split(r" |_|\/", file_path.lower())  # split by ' ', '_', and '/'
    return any([word.startswith('test') for word in word_list])


def merge_intervals(intervals):
    # intervals inclusive
    if not intervals:
        return []

    # Sort the intervals based on the starting value of each tuple
    intervals.sort(key=lambda interval: interval[0])

    merged_intervals = [intervals[0]]

    for current in intervals[1:]:
        last = merged_intervals[-1]

        # Check if there is overlap
        if current[0] <= last[1]:
            # If there is overlap, merge the intervals
            merged_intervals[-1] = (last[0], max(last[1], current[1]))
        else:
            # If there is no overlap, just add the current interval to the result list
            merged_intervals.append(current)

    return merged_intervals


def wrap_code_snippet(code_snippet, start_line, end_line):
    lines = code_snippet.split("\n")
    max_line_number = start_line + len(lines) - 1

    if not start_line == end_line == 1:  # which is a file
        assert max_line_number == end_line
    number_width = len(str(max_line_number))
    return (f"```\n"
            + "\n".join(f"{str(i + start_line).rjust(number_width)} | {line}" for i, line in enumerate(lines))
            + "\n```")


def filter_units(
    d: Union[List[CodeUnit], Dict[str, List[CodeUnit]]]
) -> Dict[str, List[CodeUnit]]:
    
    filter_d = None
    if isinstance(d, dict):
        filter_d = {
            f: [u for u in us if u.unit_type in {"class", "function", "method"}]
            for f, us in d.items()
            if any(u.unit_type in {"class", "function", "method"} for u in us)
        }
    elif isinstance(d, list):
        filter_d = [u for u in d if u.unit_type in {"class", "function", "method"}]
    
    return filter_d


def normalize_path(path: str) -> str:
    """
    Normalize a path to POSIX format as a relative path, removing redundant slashes,
    and compatible with both Windows and Linux.
    Examples:
        "a/b/../c/" -> "a/c"
        ".\\src\\utils\\" -> "src/utils"
        "/absolute/path" -> "absolute/path" (note: leading slash is not preserved)
    """
    # Convert to POSIX format and remove leading "./" or "/"
    return str(PurePosixPath(path.strip().lstrip("./\\")))


def filter_excluded_files(valid_files: List[str], excluded_files: List[str]) -> List[str]:
    """
    Exclude paths from valid_files that match any path in excluded_files.
    excluded_files may contain files or directories:
        - If it is a file: remove on exact match;
        - If it is a directory: remove all files under that directory.

    Args:
        valid_files: All valid file paths (typically .py files in the repo)
        excluded_files: List of file or directory paths to exclude

    Returns:
        The filtered list of valid_files
    """
    # Preprocess
    excluded_files = [normalize_path(p) for p in excluded_files if p.strip()]
    filtered_files = []

    for vf in valid_files:
        norm_vf = normalize_path(vf)
        excluded = False
        for excl in excluded_files:
            # Exclusion logic: exact match, or under an excluded directory
            if norm_vf == excl or norm_vf.startswith(excl + "/"):
                excluded = True
                break
        if not excluded:
            filtered_files.append(vf)

    return filtered_files


def exclude_files(files: List[str]) -> List[str]:
    """
    Filter out irrelevant files from the given list of file paths, including:
    - Directories such as tests, examples, docs, scripts, etc.
    - Specific build/meta files such as __init__.py, __version__.py, setup.py, etc.
    - Hidden files (e.g., .gitignore, .env)
    """

    # 🧩 Common irrelevant directories or prefix keywords
    exclude_keywords = {
        "docs", "doc",
        "scripts", "script",
        "example", "examples",
        "benchmark", "benchmarks",
        "tests", "testing", "test_",
        "resource", "resources",
        "sandbox", "demo", "demos"
    }

    # 🧩 Explicitly excluded file names (exact match)
    exclude_exact_files = {
        "__init__.py",
        "__version__.py",
        "__main__.py",
        "setup.py",
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        "MANIFEST.in",
        "LICENSE",
        "COPYING",
        "CONTRIBUTING.md",
        "README.md",
        "readme.md",
    }

    def should_exclude(path: str) -> bool:
        # Normalize path separators
        parts = path.replace("\\", "/").split("/")
        for part in parts:
            part_lower = part.lower().strip()

            # 🚫 Exclude hidden files or directories
            if part_lower.startswith("."):
                return True

            # 🚫 Exclude specific file names
            if part_lower in exclude_exact_files:
                return True

            # 🚫 Exclude directories/files containing certain keywords
            if any(
                part_lower == keyword or part_lower.startswith(keyword)
                for keyword in exclude_keywords
            ):
                return True

        return False

    # Return the filtered file list
    return [f for f in files if should_exclude(f)]
    


def load_skeleton_from_repo(
    repo_dir: str,
    filter_func: Callable = lambda x: True
):
    """
    Build a RepoSkeleton and filter files using the given filter_func,
    returning only "valid" files (e.g., .py files that are not test files).

    Returns:
        repo_skeleton: RepoSkeleton object (full tree)
        skeleton_info: Stringified directory tree (according to filter_func)
        valid_files:   List[str] - file paths (relative paths) that satisfy the filter conditions
    """
    import os

    file_map: Dict[str, str] = {}

    # Step 1: Walk the directory and read all readable files
    all_files: List[str] = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, repo_dir)
            rel_path = normalize_path(rel_path)

            # Use filter_func to decide whether the file is valid
            if not filter_func(rel_path):
                continue  # skip test files

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_map[rel_path] = content
                all_files.append(rel_path)
            except (UnicodeDecodeError, OSError) as e:
                continue
                # print(f"[skip] Cannot read {abs_path}: {e}")

    # Step 2: Build RepoSkeleton (full structure)
    repo_skeleton = RepoSkeleton(file_map=file_map)

    # Step 3: Output filtered directory structure using filter_func
    skeleton_info = repo_skeleton.to_tree_string(filter_func=filter_func)

    # Step 4: Collect valid files from the tree based on filter_func
    valid_files: List[str] = []

    def _collect_valid_files(node):
        if isinstance(node, FileNode) and filter_func(node):
            valid_files.append(node.path)
        elif isinstance(node, DirectoryNode) and filter_func(node):
            for child in node.children():
                _collect_valid_files(child)

    root = repo_skeleton.root
    _collect_valid_files(root)

    # Step 5: Return results
    return repo_skeleton, skeleton_info, sorted(valid_files)

def calculate_diff(
    ori_file_units: List[CodeUnit],
    new_file_units: List[CodeUnit]
) -> Dict[str, List[CodeUnit]]:
    """
    Compare two versions of CodeUnits and return diffs only at the class/function level.
    """
    # Keep only class/function
    ori_units: List[CodeUnit] = filter_units(ori_file_units)
    new_units: List[CodeUnit] = filter_units(new_file_units)

    changed, added, deleted, unchanged = [], [], [], []

    ori_map = {u.key(): u for u in ori_units}
    new_map = {u.key(): u for u in new_units}


    # Traverse the old units to find changed/deleted
    for key, old_unit in ori_map.items():
        if key in new_map:
            new_unit = new_map[key]
            if old_unit.semantic_equals(new_unit):
                unchanged.append(new_unit)
            else:
                changed.append(new_unit)
        else:
            deleted.append(old_unit)

    # Traverse the new units to find added
    for key, new_unit in new_map.items():
        if key not in ori_map:
            added.append(new_unit)

    result =  {
        "changed": changed,
        "added": added,
        "deleted": deleted
    }
    
    return result

    
    
