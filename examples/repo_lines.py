import json
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import re
import pandas as pd
import multiprocessing as mp

sys.path.append(str(Path(__file__).absolute().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# 多进程：每个进程初始化一次 encoder
# -----------------------------
_ENCODER = None
_MODEL = None

def _init_worker(model: str):
    global _ENCODER, _MODEL
    _MODEL = model
    import tiktoken
    try:
        _ENCODER = tiktoken.encoding_for_model(model)
    except KeyError:
        _ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens_tiktoken(code_str: str) -> int:
    # 依赖每个进程的全局 encoder
    return len(_ENCODER.encode(code_str)) if _ENCODER is not None else 0


def get_all_files(directory: str) -> List[str]:
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    logging.info(f"Found {len(all_files)} files in {directory}")
    return all_files


import os
import re
from typing import List

def exclude_files(files: List[str], repo_dir: str) -> List[str]:
    exclude_tokens = {
        "docs", "doc", "scripts",
        "example", "examples",
        "benchmark", "benchmarks",
        "tests", "test", "testing",
        "build_tools",
        ".py_deps",
        "resource", "resources",
        "demo", "demos",
        "get-pip", "getpip",
        # "__init__",
        ".venv", ".pytest_cache",
    }

    # 文件名级前缀（test_*.py）
    exclude_prefixes = {"test_"}

    # 路径段级前缀（目录名/文件名原样判断，用于 .py_dep_* 这种）
    exclude_part_prefixes = {".py_dep_", "py_dep_", ".py_deps", ".py_deps_", "py_deps_"} #, "build_tools", "maint_tools",}

    splitter = re.compile(r"[\/\\._\-]+")

    def should_exclude(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
        except (UnicodeDecodeError, OSError, Exception):
            return True

        if not data:
            return True

        rel = os.path.relpath(str(path), repo_dir).replace("\\", "/")
        parts = rel.split("/")

        for part in parts:
            p = part.lower()

            # 1) 路径段原样前缀匹配：.py_dep_* / py_dep_*
            if any(p.startswith(pref) for pref in exclude_part_prefixes):
                return True

            # 2) 文件名前缀匹配：test_*
            if any(p.startswith(pref) for pref in exclude_prefixes):
                return True

            # 3) token 匹配：unit_tests / docs-v2 / my-tests
            tokens = [t for t in splitter.split(p) if t]
            if any(t in exclude_tokens for t in tokens):
                return True

        return False

    return [f for f in files if not should_exclude(f)]

# -----------------------------
# Worker：处理单个文件（返回纯 dict 列表）
# -----------------------------
def process_one_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_code = f.read()
    except Exception:
        return []

    # 延迟导入：避免主进程导入的对象被 pickle / 触发 fork 问题
    from zerorepo.rpg_gen.base.unit import ParsedFile

    results: List[Dict[str, Any]] = []
    try:
        units = ParsedFile(code=file_code, file_path=file_path).units
    except Exception:
        return results

    for tgt_unit in units:
        if tgt_unit.unit_type not in {"class", "function", "import", "assignment"}:
            continue

        try:
            lines_count, unit_code = tgt_unit.count_lines(original=False, return_code=True)
            token_count = count_tokens_tiktoken(unit_code)
        except Exception:
            continue

        results.append({
            "file_path": file_path,
            "unit_name": tgt_unit.name,
            "unit_type": tgt_unit.unit_type,
            "lines_count": int(lines_count),
            "token_count": int(token_count),
        })

    return results


def calculate_real_repo_mp(
    dir_path: str,
    model: str = "gpt-4o",
    num_workers: Optional[int] = None,
    chunksize: int = 8,
) -> Dict[str, Any]:
    all_files = get_all_files(dir_path)
    original_file_count = len(all_files)
    
    py_files = [f for f in all_files if f.endswith(".py")]
    py_files = exclude_files(py_files, dir_path)

    filtered_file_count = len(py_files)

    if num_workers is None:
        # IO + 解析型任务：用 cpu_count() 比较合适（也可自己调小一点）
        num_workers = 16

    logging.info(f"Processing {filtered_file_count} .py files with {num_workers} workers...")

    # 使用 spawn 更稳（尤其在一些复杂依赖/日志/环境下）
    ctx = mp.get_context("spawn")

    all_results: List[Dict[str, Any]] = []
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(model,),
    ) as pool:
        # imap_unordered：更快返回，整体吞吐更好
        for file_results in pool.imap_unordered(process_one_file, py_files, chunksize=chunksize):
            if file_results:
                all_results.extend(file_results)

    classes_results = [r for r in all_results if r["unit_type"] == "class"]
    function_results = [r for r in all_results if r["unit_type"] == "function"]
    import_results = [r for r in all_results if r["unit_type"] == "import"]
    assignment_results = [r for r in all_results if r["unit_type"] == "assignment"]

    return {
        "class": classes_results,
        "function": function_results,
        "import": import_results,
        "assignment": assignment_results,
        "all": all_results,
        "original_file_count": original_file_count,
        "filtered_file_count": filtered_file_count,
    }


def get_task_progress(checkpoints_path: str) -> Dict[str, Any]:
    """
    从 tasks.json 和 task_manager_state.json 获取任务进度统计

    统计逻辑：
    - tasks.json 的 planned_batches_dict: 统计所有计划的 tasks 数量
    - task_manager_state.json: 获取 completed_tasks 和 failed_tasks 数量

    Returns:
        包含 planned_tasks, completed_tasks, failed_tasks, progress_rate 等信息的字典
    """
    # 读取 tasks.json 获取计划任务数
    tasks_json_path = os.path.join(checkpoints_path, "tasks.json")
    if not os.path.exists(tasks_json_path):
        return None

    try:
        with open(tasks_json_path, 'r') as f:
            tasks_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Error reading {tasks_json_path}: {e}")
        return None

    # 统计计划任务数
    planned_batches_dict = tasks_data.get('planned_batches_dict', {})
    total_planned = 0
    for subtree, files in planned_batches_dict.items():
        for file_path, tasks in files.items():
            total_planned += len(tasks)

    # 读取 task_manager_state.json 获取完成/失败任务数
    state_json_path = os.path.join(checkpoints_path, "task_manager_state.json")
    completed_tasks = 0
    failed_tasks = 0
    current_task = None

    if os.path.exists(state_json_path):
        try:
            with open(state_json_path, 'r') as f:
                state_data = json.load(f)
            completed_tasks = len(state_data.get('completed_tasks', []))
            failed_tasks = len(state_data.get('failed_tasks', []))
            current_task = state_data.get('current_task')
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Error reading {state_json_path}: {e}")

    # 计算进度（completed + failed 都算处理过的）
    processed_tasks = completed_tasks + failed_tasks
    progress_rate = (processed_tasks / total_planned * 100) if total_planned > 0 else 0

    # 判断是否完成
    is_done = (current_task is None) and (processed_tasks >= total_planned)

    return {
        "planned_tasks": total_planned,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "progress_rate": round(progress_rate, 1),
        "is_done": is_done,
    }


def analyze_single_repo(repo_dir: str, repo_name: str, model_name: str) -> Optional[Dict[str, Any]]:
    """分析单个 repo 的代码统计"""
    if not os.path.exists(repo_dir):
        logging.warning(f"Repo dir not found: {repo_dir}")
        return None

    try:
        repo_result = calculate_real_repo_mp(
            repo_dir,
            model="gpt-4o",
            num_workers=None,
            chunksize=8,
        )
    except Exception as e:
        logging.error(f"Error processing {repo_dir}: {e}")
        return None

    return {
        "model": model_name,
        "repo": repo_name,
        "original_files": repo_result["original_file_count"],
        "filtered_files": repo_result["filtered_file_count"],
        "class_count": len(repo_result["class"]),
        "function_count": len(repo_result["function"]),
        "class_lines": sum(unit["lines_count"] for unit in repo_result["class"]),
        "function_lines": sum(unit["lines_count"] for unit in repo_result["function"]),
        "total_lines": sum(unit["lines_count"] for unit in repo_result["all"]),
        "total_tokens": sum(unit["token_count"] for unit in repo_result["all"]),
    }


def analyze_all_models_ref(
    base_path: str = "/home/v-jianwenluo/temp/repo_encoder_exp",
    models: List[str] = None,
    repos: List[str] = None,
    mode: str = "ref",
) -> Dict[str, Any]:
    """
    分析多个模型下指定模式的所有 repo 的代码合成情况

    Args:
        base_path: 实验根目录
        models: 模型列表，默认 ["gpt-4.1", "gpt-5-mini", "o4-mini"]
        repos: repo 列表，默认自动扫描
        mode: 模式，如 "ref" 或 "docs"
    """
    if models is None:
        models = ["gpt-4.1", "gpt-5-mini", "o4-mini"]

    all_results = []
    model_summaries = {}

    for model in models:
        model_path = os.path.join(base_path, model, mode)
        if not os.path.exists(model_path):
            logging.warning(f"Model path not found: {model_path}")
            continue

        # 自动扫描 repos
        if repos is None:
            repo_list = sorted([
                d for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d))
            ])
        else:
            repo_list = repos

        model_results = []
        print(f"\n{'='*70}")
        print(f" Model: {model} | Mode: {mode}")
        print(f"{'='*70}")

        for repo_name in repo_list:
            # workspace 可能是 workspace 或 Workspace
            workspace_path = os.path.join(model_path, repo_name, "workspace")
            if not os.path.exists(workspace_path):
                workspace_path = os.path.join(model_path, repo_name, "Workspace")

            if not os.path.exists(workspace_path):
                print(f"  {repo_name}: workspace not found, skipping")
                continue

            print(f"  Processing {repo_name}...")
            result = analyze_single_repo(workspace_path, repo_name, model)
            if result:
                # 添加任务进度统计
                checkpoints_path = os.path.join(model_path, repo_name, "checkpoints")
                task_progress = get_task_progress(checkpoints_path)
                if task_progress:
                    result["is_done"] = task_progress["is_done"]
                    result["planned_tasks"] = task_progress["planned_tasks"]
                    result["completed_tasks"] = task_progress["completed_tasks"]
                    result["failed_tasks"] = task_progress["failed_tasks"]
                    result["task_progress"] = task_progress["progress_rate"]
                else:
                    result["is_done"] = False
                    result["planned_tasks"] = 0
                    result["completed_tasks"] = 0
                    result["failed_tasks"] = 0
                    result["task_progress"] = 0.0

                model_results.append(result)
                all_results.append(result)

        # 模型汇总
        if model_results:
            df = pd.DataFrame(model_results)
            total_planned = int(df["planned_tasks"].sum())
            total_completed = int(df["completed_tasks"].sum())
            total_failed = int(df["failed_tasks"].sum())
            done_count = int(df["is_done"].sum())
            total_processed = total_completed + total_failed
            overall_progress = (total_processed / total_planned * 100) if total_planned > 0 else 0
            model_summaries[model] = {
                "repo_count": len(model_results),
                "done_count": done_count,
                "total_files": int(df["filtered_files"].sum()),
                "total_classes": int(df["class_count"].sum()),
                "total_functions": int(df["function_count"].sum()),
                "total_lines": int(df["total_lines"].sum()),
                "total_tokens": int(df["total_tokens"].sum()),
                "avg_lines_per_repo": round(df["total_lines"].mean(), 1),
                "avg_tokens_per_repo": round(df["total_tokens"].mean(), 1),
                "total_planned_tasks": total_planned,
                "total_completed_tasks": total_completed,
                "total_failed_tasks": total_failed,
                "overall_task_progress": round(overall_progress, 1),
            }

    return {
        "details": all_results,
        "model_summaries": model_summaries,
    }


def print_results_table(results: Dict[str, Any]):
    """打印结果表格"""
    details = results["details"]
    if not details:
        print("No results found.")
        return

    # 按模型分组打印详细表格
    df = pd.DataFrame(details)

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{'='*120}")
        print(f" {model}")
        print(f"{'='*120}")
        print(f"{'Repo':<20} {'Files':>6} {'Classes':>7} {'Funcs':>6} {'Lines':>8} {'Tokens':>10} {'Plan':>6} {'Succ':>6} {'Fail':>6} {'Prog':>7} {'Status':>8}")
        print(f"{'-'*120}")

        for _, row in model_df.iterrows():
            progress_str = f"{row['task_progress']:.1f}%"
            status = "Done" if row['is_done'] else "Running"
            print(f"{row['repo']:<20} {row['filtered_files']:>6} {row['class_count']:>7} "
                  f"{row['function_count']:>6} {row['total_lines']:>8} {row['total_tokens']:>10} "
                  f"{row['planned_tasks']:>6} {row['completed_tasks']:>6} {row['failed_tasks']:>6} {progress_str:>7} {status:>8}")

        print(f"{'-'*120}")
        total_planned = model_df['planned_tasks'].sum()
        total_completed = model_df['completed_tasks'].sum()
        total_failed = model_df['failed_tasks'].sum()
        done_count = model_df['is_done'].sum()
        total_processed = total_completed + total_failed
        total_progress = (total_processed / total_planned * 100) if total_planned > 0 else 0
        print(f"{'TOTAL':<20} {model_df['filtered_files'].sum():>6} {model_df['class_count'].sum():>7} "
              f"{model_df['function_count'].sum():>6} {model_df['total_lines'].sum():>8} {model_df['total_tokens'].sum():>10} "
              f"{total_planned:>6} {total_completed:>6} {total_failed:>6} {total_progress:>6.1f}% {done_count:>5}/{len(model_df)}")

    # 跨模型对比
    print(f"\n{'='*120}")
    print(" CROSS-MODEL COMPARISON")
    print(f"{'='*120}")
    summaries = results["model_summaries"]
    print(f"{'Model':<12} {'Repos':>6} {'Done':>5} {'Files':>6} {'Classes':>7} {'Funcs':>6} {'Lines':>8} {'Tokens':>10} {'Plan':>6} {'Succ':>6} {'Fail':>6} {'Prog':>7}")
    print(f"{'-'*120}")
    for model, summary in summaries.items():
        progress_str = f"{summary['overall_task_progress']:.1f}%"
        print(f"{model:<12} {summary['repo_count']:>6} {summary['done_count']:>5} {summary['total_files']:>6} "
              f"{summary['total_classes']:>7} {summary['total_functions']:>6} "
              f"{summary['total_lines']:>8} {summary['total_tokens']:>10} "
              f"{summary['total_planned_tasks']:>6} {summary['total_completed_tasks']:>6} {summary['total_failed_tasks']:>6} {progress_str:>7}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="统计模型合成 repo 的代码情况")
    parser.add_argument("--base-path", type=str,
                        help="实验根目录")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["gpt-4.1", "gpt-5-mini"],
                        help="模型列表")
    parser.add_argument("--repos", type=str, nargs="+",
                        default=None,
                        help="Repo 列表（默认自动扫描）")
    parser.add_argument("--mode", type=str, default="ref",
                        help="模式：ref 或 docs")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 文件路径")

    args = parser.parse_args()

    results = analyze_all_models_ref(
        base_path=args.base_path,
        models=args.models,
        repos=args.repos,
        mode=args.mode,
    )

    print_results_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")