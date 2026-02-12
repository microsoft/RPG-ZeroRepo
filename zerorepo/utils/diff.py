from typing import List
import json
from zerorepo.rpg_gen.base.unit import ParsedFile, CodeUnit
from zerorepo.rpg_gen.base.node import filter_non_test_py_files
from zerorepo.utils.repo import (
    load_skeleton_from_repo,
    exclude_files,
    filter_excluded_files,
    calculate_diff,
    filter_units
)


def generate_detailed_diff(
    last_repo_dir: str,
    cur_repo_dir: str,
    last_excluded_files: List=[],
):
    
    # Only need skeletons for Python files and non-test files
    last_repo_skeleton, _, _ = load_skeleton_from_repo(repo_dir=last_repo_dir, filter_func=filter_non_test_py_files)
    cur_repo_skeleton, _, _ = load_skeleton_from_repo(repo_dir=cur_repo_dir, filter_func=filter_non_test_py_files)
    
    last_file_code_map = last_repo_skeleton.get_file_code_map()
    cur_file_code_map = cur_repo_skeleton.get_file_code_map()
    
    cur_repo_paths = list(cur_repo_skeleton.path_to_node.keys())
    print(f"Current Repo Paths: {json.dumps(cur_repo_paths, indent=4)}")
    
    # Get current files that need to be filtered
    cur_exclude_files = exclude_files(files=list(cur_file_code_map.keys()))
    need_to_exclude = last_excluded_files + cur_exclude_files
    
    cur_filtered_files = filter_excluded_files(valid_files=list(cur_file_code_map.keys()), excluded_files=need_to_exclude)
    
    # Only update RPG for Python files with valid changes
    cur_file_code_map = {file: code for file, code in cur_file_code_map.items() if file in cur_filtered_files}
    
    # Distinguish new files / existing files
    already_repo_paths = list()
    remain_repo_paths = [path for path in cur_repo_paths if path not in already_repo_paths and path.endswith(".py")]

    print(f"  Already existing files: {len(already_repo_paths)}")
    print(f"  New files to parse: {len(remain_repo_paths)}")

    last_file_code_map = last_repo_skeleton.get_file_code_map()
    cur_file_code_map = cur_repo_skeleton.get_file_code_map()

    last_files = set(last_file_code_map.keys())
    cur_files = set(cur_file_code_map.keys())

    added_files = cur_files - last_files
    deleted_files = last_files - cur_files
    common_files = cur_files & last_files

    # 3) Parse into CodeUnits (functions, classes, methods)
    def parse_units(file_map):
        result = {}
        for path, code in file_map.items():
            parsed = ParsedFile(code=code, file_path=path)
            result[path] = [
                u for u in parsed.units if u.unit_type in {"class", "function", "method"}
            ]
        return result

    last_units_map = parse_units(last_file_code_map)
    cur_units_map = parse_units(cur_file_code_map)

    # 4) Added files -> all CodeUnits are treated as added
    added = {
        f: cur_units_map[f]
        for f in added_files
        if f in cur_units_map
    }

    # 5) Deleted files -> all CodeUnits are treated as deleted
    deleted = {
        f: last_units_map[f]
        for f in deleted_files
        if f in last_units_map
    }

    # 6) Modified files -> unit-level diff
    modified = {}
    for f in common_files:
        last_f_units = last_units_map.get(f, [])
        cur_f_units = cur_units_map.get(f, [])
        
        modified[f] = calculate_diff(
            ori_file_units=last_f_units,
            new_file_units=cur_f_units
        )
            
    added = filter_units(added)
    # modified = filter_units(modified)
    deleted = filter_units(deleted)

    # ---------- 7) Summary ----------
    diff_result = {
        "added": added,
        "modified": modified,
        "deleted": deleted
    }

    return diff_result