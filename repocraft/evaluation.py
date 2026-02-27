import os
import json
import pandas as pd
import subprocess
from multiprocessing import Pool, cpu_count
import re
import numpy as np
import sys
import importlib
import argparse
from typing import Union, List, Dict, Any, Optional
from collections import defaultdict


def is_test_successful(return_code: int, test_output: Union[str, bytes, None]) -> bool:
    """
    Determine if a test run was successful based on return code and output.
    
    Args:
        return_code: The exit code from the test execution
        test_output: The stdout/stderr output from the test execution (can be None)
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    # Handle None or empty output
    if test_output is None:
        # Only rely on return code if no output
        return return_code == 0
    
    # Convert bytes to string if necessary
    if isinstance(test_output, bytes):
        try:
            test_output = test_output.decode('utf-8', errors='ignore')
        except Exception:
            # If decoding fails, treat as empty
            test_output = ""
    
    # Ensure test_output is string
    test_output = str(test_output)
    
    # Check return code first - non-zero typically means failure
    # But some test runners may return 0 even on failures, so we check output too
    if return_code != 0:
        # Double-check for false positives - some tools return non-zero on warnings
        warning_only_patterns = [
            r"warning[s]?\s*:?\s*\d+",
            r"\d+\s*warning[s]?",
            r"DeprecationWarning",
            r"PendingDeprecationWarning",
            r"FutureWarning",
        ]
        
        # If only warnings and no failures, might still be successful
        has_only_warnings = any(re.search(p, test_output, re.IGNORECASE) for p in warning_only_patterns)
        has_failures = any(re.search(p, test_output, re.IGNORECASE) for p in [
            r"FAILED", r"ERROR", r"FAIL:", r"\d+\s*failed", r"AssertionError", r"Exception:"
        ])
        
        if has_only_warnings and not has_failures:
            # Continue checking, don't immediately return False
            pass
        else:
            return False
    
    # Additional failure patterns for various test frameworks
    failure_patterns = [
        # pytest patterns - specific test output indicators
        r"FAILED\s+\[",  # pytest-style "FAILED [100%]"
        r"::.*FAILED",   # pytest-style "test::path FAILED"
        r"={3,}\s*FAILURES\s*={3,}",  # pytest-style "======= FAILURES ======="
        r"={3,}\s*ERRORS\s*={3,}",    # pytest-style "======= ERRORS ======="
        r"[1-9]\d*\s*failed",  # Only match non-zero failures
        r"[1-9]\d*\s*error",   # Only match non-zero errors
        r"short test summary info",
        
        # unittest patterns
        r"FAIL:",
        r"ERROR:",
        r"failures=[1-9]\d*",  # Only match non-zero failures
        r"errors=[1-9]\d*",    # Only match non-zero errors
        r"unittest.*FAIL",
        
        # Generic patterns - be more specific
        r"AssertionError",
        r"Test.*failed:",   # "Test xyz failed:" style messages
        r"test.*failed:",   # "test xyz failed:" style messages
        r"Test.*error:",    # "Test xyz error:" style messages (be more specific)
        r"test.*error:",    # "test xyz error:" style messages (be more specific)
        r"Traceback\s*\(most recent call last\)",
        r"Exception:",
        r"Failed:",
        r"Failure:",
        r"FAILED:",
        
        # nose patterns
        r"FAIL\s*:",
        r"ERROR\s*:",
        
        # JavaScript test frameworks  
        r"[1-9]\d*\s*failing",  # Only match non-zero failing
        r"✖\s*[1-9]\d*\s*test[s]?\s*failed",  # Only match non-zero failed
        r"×\s*[1-9]\d*\s*test[s]?\s*failed",   # Only match non-zero failed
        
        # Build/compilation errors
        r"compilation failed",
        r"build failed",
        r"error:\s*",
        r"fatal:\s*",
        
        # Timeout/crash indicators
        #r"timeout",
        #r"timed out",
        #r"segmentation fault",
        #r"core dumped",
        #r"aborted",
        # r"killed",
    ]
    
    # Success patterns for various test frameworks
    success_patterns = [
        # pytest patterns
        r"passed",
        r"\d+\s*passed",
        r"={3,}\s*\d+\s*passed",
        r"all\s+tests?\s+passed",
        
        # unittest patterns
        r"OK\s*$",
        r"OK\s*\(",
        r"Ran\s+\d+\s+test[s]?\s+in\s+[\d.]+s\s*OK",
        
        # Generic patterns
        r"Success",
        r"successful",
        r"0\s*failed",
        r"0\s*error[s]?",
        r"test[s]?\s+passed",
        r"Test[s]?\s+passed",
        r"passed\s+\d+/\d+",
        r"✓",  # checkmark
        r"✔",  # checkmark variant
        r"√",  # checkmark variant
        
        # JavaScript test frameworks
        r"\d+\s*passing",
        r"✓\s*\d+\s*test[s]?",
        
        # JUnit/Maven patterns
        r"BUILD\s+SUCCESS",
        r"Tests\s+run:\s*\d+,\s*Failures:\s*0,\s*Errors:\s*0",
        r"Tests\s+run:\s*\d+.*Failures:\s*0.*Errors:\s*0",
        
        # RSpec patterns
        r"\d+\s*examples?,\s*0\s*failures?",
        r"examples?,\s*0\s*failures?",
    ]
    
    # Special handling for empty or very short output
    if len(test_output.strip()) < 10:
        # Very short output - rely more on return code
        return return_code == 0
    
    # Check for failure patterns first (more definitive)
    for pattern in failure_patterns:
        if re.search(pattern, test_output, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Check for explicit success patterns
    found_success = False
    for pattern in success_patterns:
        if re.search(pattern, test_output, re.IGNORECASE | re.MULTILINE):
            found_success = True
            break
    
    # Special case: Check for test count summary patterns
    # e.g., "5 passed, 0 failed" or "Tests: 5 passed, 0 failed" or "8 tests passed, 0 tests failed"
    test_summary_pattern = r"(\d+)\s*(?:test[s]?\s+)?passed[,\s]+(\d+)\s*(?:test[s]?\s+)?failed"
    match = re.search(test_summary_pattern, test_output, re.IGNORECASE)
    if match:
        passed_count = int(match.group(1))
        failed_count = int(match.group(2))
        return failed_count == 0 and passed_count > 0
    
    # If we found explicit success indicators, trust them
    if found_success:
        return True
    
    # No explicit success/failure found - use heuristics
    # Check if output looks like a test run at all
    test_run_indicators = [
        r"test",
        r"Test",
        r"spec",
        r"Spec",
        r"passed",
        r"failed",
        r"ran\s+\d+",
        r"running",
        r"executed",
    ]
    
    looks_like_test_output = any(re.search(p, test_output, re.IGNORECASE) for p in test_run_indicators)
    
    if not looks_like_test_output:
        # Doesn't look like test output - rely on return code
        return return_code == 0
    
    # If return code is 0 and no failure patterns found, consider it successful
    # This handles cases where test runners don't output explicit success messages
    return return_code == 0



repo_name_to_craft = {
    "sklearn": "MLKit-Py",
    "sympy": "SymbolicMath",
    "statsmodels": "StatModeler",
    "requests": "HttpEasy",
    "django": "PyWebEngine",
    "pandas": "TableKit"
}

ABLATION = False

def eval_one_task(args):
    """评估单个 result_dir 下的 test_file.py"""
    
    result_dir, repo_root = args
    print(f"\n▶️ 开始评估: {result_dir}")

    final_result = {
        "result_dir": result_dir,
        "repo_name": None,   # 新增字段
        "voting": False,
        "passed": False,
        "returncode": None,
        "output": "",
        "include_in_metrics": False,   # 是否计入最终指标
        "skip_reason": ""              # 若不计入，给出原因
    }

    # 找到 repo 名字
    # 对于 ablation 路径，使用 ablation 子目录名作为 repo_name
    # 例如: .../ablation/sklearn_feature_file/sklearn-01/ -> sklearn_feature_file
    if "/ablation/" in result_dir:
        # 提取 ablation 后面的子目录名
        ablation_match = re.search(r'/ablation/([^/]+)/', result_dir)
        if ablation_match:
            real_repo_name = ablation_match.group(1)
        else:
            real_repo_name = None
    else:
        real_repo_name = next((repo for repo in repo_name_to_craft.keys() if repo in result_dir), None)

    if not real_repo_name or not os.path.exists(result_dir):
        final_result["skip_reason"] = "no_repo_match_or_result_dir_missing"
        print(f"  ⚠️ 未匹配到 repo_name 或 result_dir 不存在, 跳过: {final_result['skip_reason']}")
        return final_result

    final_result["repo_name"] = real_repo_name

    result_json_path = os.path.join(result_dir, "results", "result.json")
    if not os.path.exists(result_json_path):
        final_result["skip_reason"] = "result_json_missing"
        print(f"  ⚠️ 没找到 result.json, 跳过")
        return final_result

    # 读取 json
    with open(result_json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    final_result["voting"] = data.get("voting", False)
    print(f"  🗳️ voting = {final_result['voting']}")
   
    final_result["output"] = data.get("test_output", "")
    final_result["returncode"] = 0
    # 到这里，repo_dir 存在且 result.json 存在，认为可计入指标
    final_result["include_in_metrics"] = True
        
    # 解析测试结果
    test_passed = is_test_successful(0, final_result["output"])
    final_result["passed"] = (test_passed and final_result["returncode"] == 0 and final_result["voting"])
    print(f"  🎯 passed = {final_result['passed']}, test_passed={test_passed}")

    return final_result


def find_result_dirs(base_dir: str, models: List[str] = None, experiment_types: List[str] = None) -> List[tuple]:
    """
    Find all result directories under the base directory.

    Supports multiple directory structures:
    1. {base_dir}/{model}/{exp_type}/{project}/{task_id}/results/result.json  (new structure)
    2. {base_dir}/{model}/{exp_type}/output/{task_id}/results/result.json     (legacy)
    3. {base_dir}/{model}/{exp_type}/{project}/results/result.json            (legacy)

    Args:
        base_dir: Base directory to search (e.g., /home/v-jianwenluo/temp/RebuildTest/exp_results)
        models: List of model names to include (e.g., ['gpt-4.1', 'o4-mini'])
        experiment_types: List of experiment types (e.g., ['docs', 'ref', 'wo_ref'])

    Returns:
        List of (result_dir, repo_root) tuples
    """
    if models is None:
        models = ['gpt-4.1', 'gpt-5-mini', 'human'] # , 'o4-mini']
    if experiment_types is None:
        experiment_types = ['docs', 'ref', 'wo_ref', 'ablation']

    result_dirs = []

    # Known project names (repo names)
    known_projects = set(repo_name_to_craft.keys())
    # Also skip these directories
    skip_dirs = {'logs', 'workspaces', 'output', '__pycache__'}

    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            continue

        for exp_type in experiment_types:
            exp_path = os.path.join(model_path, exp_type)
            if not os.path.exists(exp_path):
                continue

            # Check for output directory structure (legacy): {exp_type}/output/{task_id}/results/result.json
            output_path = os.path.join(exp_path, "output")
            if os.path.exists(output_path) and os.path.isdir(output_path):
                for task_id in os.listdir(output_path):
                    task_path = os.path.join(output_path, task_id)
                    if os.path.isdir(task_path):
                        result_json = os.path.join(task_path, "results", "result.json")
                        if os.path.exists(result_json):
                            result_dirs.append((task_path, task_path))

            # Check for new structure: {exp_type}/{project}/{task_id}/results/result.json
            for project in os.listdir(exp_path):
                if project in skip_dirs:
                    continue

                project_path = os.path.join(exp_path, project)
                if not os.path.isdir(project_path):
                    continue

                # Check if this is a project directory (contains task subdirectories)
                if project in known_projects:
                    # New structure: iterate over task_id subdirectories
                    for task_id in os.listdir(project_path):
                        task_path = os.path.join(project_path, task_id)
                        if os.path.isdir(task_path):
                            result_json = os.path.join(task_path, "results", "result.json")
                            if os.path.exists(result_json):
                                result_dirs.append((task_path, task_path))
                else:
                    # Legacy structure: check if project itself has results
                    results_path = os.path.join(project_path, "results")
                    result_json = os.path.join(results_path, "result.json")

                    if os.path.exists(result_json):
                        result_dirs.append((project_path, project_path))
                    else:
                        # Check for workspace/results structure
                        workspace_results = os.path.join(project_path, "workspace", "results")
                        if os.path.exists(os.path.join(workspace_results, "result.json")):
                            result_dirs.append((project_path, project_path))
                        else:
                            # Ablation structure: {exp_type}/{ablation_type}/{task_id}/results/result.json
                            # e.g., ablation/sklearn_feature_file/sklearn-01/results/result.json
                            for subdir in os.listdir(project_path):
                                subdir_path = os.path.join(project_path, subdir)
                                if os.path.isdir(subdir_path):
                                    sub_result_json = os.path.join(subdir_path, "results", "result.json")
                                    if os.path.exists(sub_result_json):
                                        result_dirs.append((subdir_path, subdir_path))

    return result_dirs


def print_summary_table(results: List[Dict], group_by: str = "repo_name"):
    """Print a summary table of results."""

    # Filter to only include results that should be in metrics
    valid_results = [r for r in results if r.get("include_in_metrics", False)]

    if not valid_results:
        print("\n⚠️  No valid results to summarize.")
        return

    # Group by the specified field
    groups = defaultdict(list)
    for r in valid_results:
        key = r.get(group_by, "unknown")
        groups[key].append(r)

    print("\n" + "=" * 80)
    print(f"{'EVALUATION RESULTS BY ' + group_by.upper():^80}")
    print("=" * 80)

    # Print header
    print(f"{'Repo':<20} {'Total':>8} {'Passed':>8} {'Voting':>8} {'Pass Rate':>12} {'Vote Rate':>12} {'Pass/Vote':>12}")
    print("-" * 100)

    total_all = 0
    passed_all = 0
    voting_all = 0

    for key in sorted(groups.keys()):
        group = groups[key]
        total = len(group)
        passed = sum(1 for r in group if r.get("passed", False))
        voting = sum(1 for r in group if r.get("voting", False))
        pass_rate = (passed / total * 100) if total > 0 else 0
        vote_rate = (voting / total * 100) if total > 0 else 0
        pass_vote_ratio = (passed / voting * 100) if voting > 0 else 0

        print(f"{key:<20} {total:>8} {passed:>8} {voting:>8} {pass_rate:>11.1f}% {vote_rate:>11.1f}% {pass_vote_ratio:>11.1f}%")

        total_all += total
        passed_all += passed
        voting_all += voting

    print("-" * 100)
    overall_rate = (passed_all / total_all * 100) if total_all > 0 else 0
    overall_vote_rate = (voting_all / total_all * 100) if total_all > 0 else 0
    overall_pass_vote_ratio = (passed_all / voting_all * 100) if voting_all > 0 else 0
    print(f"{'TOTAL':<20} {total_all:>8} {passed_all:>8} {voting_all:>8} {overall_rate:>11.1f}% {overall_vote_rate:>11.1f}% {overall_pass_vote_ratio:>11.1f}%")
    print("=" * 100)


def print_detailed_report(results: List[Dict], by_model: bool = True):
    """Print detailed report with results grouped by model and experiment type."""

    if by_model:
        # Group by (model, exp_type), then by repo
        model_exp_groups = defaultdict(lambda: defaultdict(list))

        for r in results:
            result_dir = r.get("result_dir", "")
            # Extract model name from path
            model = "unknown"
            for m in ['gpt-4.1', 'gpt-5-mini', 'o4-mini', 'human']:
                if m in result_dir:
                    model = m
                    break

            # Extract experiment type
            exp_type = "unknown"
            for e in ['docs', 'ref', 'wo_ref', 'ablation']:
                if f"/{e}/" in result_dir:
                    exp_type = e
                    break

            repo = r.get("repo_name", "unknown")
            model_exp_groups[(model, exp_type)][repo].append(r)

        print("\n" + "=" * 100)
        print(f"{'DETAILED RESULTS BY MODEL + EXPERIMENT TYPE':^100}")
        print("=" * 100)

        for (model, exp_type) in sorted(model_exp_groups.keys()):
            print(f"\n{'─' * 60}")
            print(f"📦 MODEL: {model}  |  🧪 EXP TYPE: {exp_type}")
            print(f"{'─' * 60}")

            print(f"  {'Repo':<20} {'Total':>8} {'Passed':>8} {'Voting':>8} {'PassRate':>12} {'VoteRate':>12} {'Pass/Vote':>12}")
            print(f"  {'-' * 90}")

            section_total = 0
            section_passed = 0
            section_voting = 0

            for repo in sorted(model_exp_groups[(model, exp_type)].keys()):
                group = model_exp_groups[(model, exp_type)][repo]
                valid = [r for r in group if r.get("include_in_metrics", False)]

                if not valid:
                    continue

                total = len(valid)
                passed = sum(1 for r in valid if r.get("passed", False))
                voting = sum(1 for r in valid if r.get("voting", False))
                pass_rate = (passed / total * 100) if total > 0 else 0
                vote_rate = (voting / total * 100) if total > 0 else 0
                pass_vote_ratio = (passed / voting * 100) if voting > 0 else 0

                status = "✅" if passed == total else "❌" if passed == 0 else "⚠️"
                print(f"  {repo:<20} {total:>8} {passed:>8} {voting:>8} {pass_rate:>11.1f}% {vote_rate:>11.1f}% {pass_vote_ratio:>11.1f}% {status}")

                section_total += total
                section_passed += passed
                section_voting += voting

            if section_total > 0:
                section_pass_rate = (section_passed / section_total * 100)
                section_vote_rate = (section_voting / section_total * 100)
                section_pass_vote_ratio = (section_passed / section_voting * 100) if section_voting > 0 else 0
                print(f"  {'-' * 90}")
                print(f"  {'SUBTOTAL':<20} {section_total:>8} {section_passed:>8} {section_voting:>8} {section_pass_rate:>11.1f}% {section_vote_rate:>11.1f}% {section_pass_vote_ratio:>11.1f}%")


def print_failed_details(results: List[Dict]):
    """Print details of failed tests."""

    failed = [r for r in results if r.get("include_in_metrics") and not r.get("passed")]

    if not failed:
        print("\n✅ All tests passed!")
        return

    print("\n" + "=" * 80)
    print(f"{'FAILED TESTS DETAILS':^80}")
    print("=" * 80)

    for r in failed:
        result_dir = r.get("result_dir", "unknown")
        repo = r.get("repo_name", "unknown")
        voting = r.get("voting", False)
        skip_reason = r.get("skip_reason", "")

        print(f"\n❌ {repo}")
        print(f"   Path: {result_dir}")
        print(f"   Voting: {voting}")
        if skip_reason:
            print(f"   Skip Reason: {skip_reason}")

        # Print first 500 chars of output if available
        output = r.get("output", "")
        if output:
            preview = output[:500].replace('\n', '\n   ')
            print(f"   Output Preview:\n   {preview}...")


def save_results_json(results: List[Dict], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test results across repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all results in default directory
    python evaluation.py --base-dir /home/v-jianwenluo/temp/repo_encoder_exp

    # Evaluate specific models
    python evaluation.py --base-dir /path/to/results --models gpt-4.1 o4-mini

    # Show failed test details
    python evaluation.py --base-dir /path/to/results --show-failed

    # Save results to JSON
    python evaluation.py --base-dir /path/to/results --output results.json
        """
    )

    parser.add_argument(
        "--base-dir", "-d",
        type=str,
        default="/home/v-jianwenluo/temp/RebuildTest/exp_results",
        help="Base directory containing model results"
    )

    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (default: all)"
    )

    parser.add_argument(
        "--exp-types", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Experiment types to evaluate (default: docs, ref, wo_ref)"
    )

    parser.add_argument(
        "--show-failed", "-f",
        action="store_true",
        help="Show details of failed tests"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path for saving results"
    )

    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"{'REPOSITORY EVALUATION':^80}")
    print("=" * 80)
    print(f"Base Directory: {args.base_dir}")
    print(f"Models: {args.models or 'all'}")
    print(f"Experiment Types: {args.exp_types or 'all'}")
    print("=" * 80)

    # Find all result directories
    result_dirs = find_result_dirs(
        args.base_dir,
        models=args.models,
        experiment_types=args.exp_types
    )

    print(f"\nFound {len(result_dirs)} result directories to evaluate")

    if not result_dirs:
        print("⚠️  No result directories found. Check your base directory path.")
        return 1

    # Run evaluation
    if args.parallel > 1:
        with Pool(min(args.parallel, cpu_count())) as pool:
            results = pool.map(eval_one_task, result_dirs)
    else:
        results = [eval_one_task(rd) for rd in result_dirs]

    # Print summaries
    print_summary_table(results, group_by="repo_name")
    print_detailed_report(results, by_model=True)

    if args.show_failed:
        print_failed_details(results)

    # Save results if output path specified
    if args.output:
        save_results_json(results, args.output)

    # Return exit code
    valid_results = [r for r in results if r.get("include_in_metrics", False)]
    passed = sum(1 for r in valid_results if r.get("passed", False))
    voting = sum(1 for r in valid_results if r.get("voting", False))
    total = len(valid_results)

    # 按 (model, experiment type) 分组统计，ablation 按 repo_name 细分
    model_exp_stats = defaultdict(lambda: {"total": 0, "passed": 0, "voting": 0})
    for r in valid_results:
        result_dir = r.get("result_dir", "")
        # Extract model
        model = "unknown"
        for m in ['gpt-4.1', 'gpt-5-mini', 'o4-mini', 'human']:
            if m in result_dir:
                model = m
                break
        # Extract exp_type
        exp_type = "unknown"
        for e in ['docs', 'ref', 'wo_ref', "ablation"]:
            if f"/{e}/" in result_dir:
                exp_type = e
                break
        # 对于 ablation，使用 repo_name 作为 exp_type（细分显示）
        if exp_type == "ablation":
            repo_name = r.get("repo_name", "unknown")
            exp_type = repo_name  # 使用 sklearn_feature_file / sklearn_feature_only 等
        model_exp_stats[(model, exp_type)]["total"] += 1
        if r.get("passed", False):
            model_exp_stats[(model, exp_type)]["passed"] += 1
        if r.get("voting", False):
            model_exp_stats[(model, exp_type)]["voting"] += 1

    print(f"\n{'=' * 100}")
    print(f"{'FINAL RESULTS BY MODEL + EXPERIMENT TYPE':^100}")
    print(f"{'=' * 100}")
    print(f"{'Model':<15} {'Exp Type':<25} {'Total':>8} {'Passed':>8} {'Voting':>8} {'Pass Rate':>12} {'Vote Rate':>12} {'Pass/Vote':>12}")
    print(f"{'-' * 115}")

    for (model, exp_type) in sorted(model_exp_stats.keys()):
        stats = model_exp_stats[(model, exp_type)]
        t, p, v = stats["total"], stats["passed"], stats["voting"]
        pr = (p / t * 100) if t > 0 else 0
        vr = (v / t * 100) if t > 0 else 0
        pvr = (p / v * 100) if v > 0 else 0
        print(f"{model:<15} {exp_type:<25} {t:>8} {p:>8} {v:>8} {pr:>11.1f}% {vr:>11.1f}% {pvr:>11.1f}%")

    print(f"{'-' * 115}")
    if total > 0:
        pass_rate = passed / total * 100
        vote_rate = voting / total * 100
        pass_vote_ratio = (passed / voting * 100) if voting > 0 else 0
        print(f"{'TOTAL':<15} {'':<25} {total:>8} {passed:>8} {voting:>8} {pass_rate:>11.1f}% {vote_rate:>11.1f}% {pass_vote_ratio:>11.1f}%")
    else:
        print("No valid results")
    print("=" * 115)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
    
    
    
'''
================================================================================
                        EVALUATION RESULTS BY REPO_NAME                         
================================================================================
Repo                    Total   Passed   Voting    Pass Rate    Vote Rate    Pass/Vote
----------------------------------------------------------------------------------------------------
django                    603      438      553        72.6%        91.7%        79.2%
pandas                    739      557      710        75.4%        96.1%        78.5%
requests                  180       92      118        51.1%        65.6%        78.0%
sklearn                   584      409      469        70.0%        80.3%        87.2%
sklearn_feature_file      198      139      197        70.2%        99.5%        70.6%
sklearn_feature_only      123       80      109        65.0%        88.6%        73.4%
statsmodels               656      489      610        74.5%        93.0%        80.2%
sympy                     471      336      388        71.3%        82.4%        86.6%
----------------------------------------------------------------------------------------------------
TOTAL                    3554     2540     3154        71.5%        88.7%        80.5%
====================================================================================================

====================================================================================================
                            DETAILED RESULTS BY MODEL + EXPERIMENT TYPE                             
====================================================================================================

────────────────────────────────────────────────────────────
📦 MODEL: gpt-4.1  |  🧪 EXP TYPE: docs
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  django                    109       50       75        45.9%        68.8%        66.7% ⚠️
  pandas                    160       91      147        56.9%        91.9%        61.9% ⚠️
  requests                   53        7       17        13.2%        32.1%        41.2% ⚠️
  sklearn                   111       41       56        36.9%        50.5%        73.2% ⚠️
  statsmodels                88       41       57        46.6%        64.8%        71.9% ⚠️
  sympy                      66        7       17        10.6%        25.8%        41.2% ⚠️
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  587      237      369        40.4%        62.9%        64.2%

────────────────────────────────────────────────────────────
📦 MODEL: gpt-4.1  |  🧪 EXP TYPE: ref
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  django                    102      102      102       100.0%       100.0%       100.0% ✅
  pandas                    112      112      112       100.0%       100.0%       100.0% ✅
  requests                   17       17       17       100.0%       100.0%       100.0% ✅
  sklearn                    74       74       74       100.0%       100.0%       100.0% ✅
  statsmodels                92       92       92       100.0%       100.0%       100.0% ✅
  sympy                       9        7        9        77.8%       100.0%        77.8% ⚠️
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  406      404      406        99.5%       100.0%        99.5%

────────────────────────────────────────────────────────────
📦 MODEL: gpt-5-mini  |  🧪 EXP TYPE: ablation
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  sklearn_feature_file      198      139      197        70.2%        99.5%        70.6% ⚠️
  sklearn_feature_only      123       80      109        65.0%        88.6%        73.4% ⚠️
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  321      219      306        68.2%        95.3%        71.6%

────────────────────────────────────────────────────────────
📦 MODEL: gpt-5-mini  |  🧪 EXP TYPE: docs
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  django                    131       81      115        61.8%        87.8%        70.4% ⚠️
  pandas                    183       99      168        54.1%        91.8%        58.9% ⚠️
  requests                   51       14       28        27.5%        54.9%        50.0% ⚠️
  sklearn                   118       42       59        35.6%        50.0%        71.2% ⚠️
  statsmodels               187      108      172        57.8%        92.0%        62.8% ⚠️
  sympy                      86       31       52        36.0%        60.5%        59.6% ⚠️
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  756      375      594        49.6%        78.6%        63.1%

────────────────────────────────────────────────────────────
📦 MODEL: gpt-5-mini  |  🧪 EXP TYPE: ref
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  django                    101      101      101       100.0%       100.0%       100.0% ✅
  pandas                     90       90       90       100.0%       100.0%       100.0% ✅
  requests                   16       16       16       100.0%       100.0%       100.0% ✅
  sklearn                    83       83       83       100.0%       100.0%       100.0% ✅
  statsmodels               102      102      102       100.0%       100.0%       100.0% ✅
  sympy                      98       98       98       100.0%       100.0%       100.0% ✅
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  490      490      490       100.0%       100.0%       100.0%

────────────────────────────────────────────────────────────
📦 MODEL: human  |  🧪 EXP TYPE: ref
────────────────────────────────────────────────────────────
  Repo                    Total   Passed   Voting     PassRate     VoteRate    Pass/Vote
  ------------------------------------------------------------------------------------------
  django                    160      104      160        65.0%       100.0%        65.0% ⚠️
  pandas                    194      165      193        85.1%        99.5%        85.5% ⚠️
  requests                   43       38       40        88.4%        93.0%        95.0% ⚠️
  sklearn                   198      169      197        85.4%        99.5%        85.8% ⚠️
  statsmodels               187      146      187        78.1%       100.0%        78.1% ⚠️
  sympy                     212      193      212        91.0%       100.0%        91.0% ⚠️
  ------------------------------------------------------------------------------------------
  SUBTOTAL                  994      815      989        82.0%        99.5%        82.4%

====================================================================================================
                              FINAL RESULTS BY MODEL + EXPERIMENT TYPE                              
====================================================================================================
Model           Exp Type                     Total   Passed   Voting    Pass Rate    Vote Rate    Pass/Vote
-------------------------------------------------------------------------------------------------------------------
gpt-4.1         docs                           587      237      369        40.4%        62.9%        64.2%
gpt-4.1         ref                            406      404      406        99.5%       100.0%        99.5%
gpt-5-mini      docs                           756      375      594        49.6%        78.6%        63.1%
gpt-5-mini      ref                            490      490      490       100.0%       100.0%       100.0%
gpt-5-mini      sklearn_feature_file           198      139      197        70.2%        99.5%        70.6%
gpt-5-mini      sklearn_feature_only           123       80      109        65.0%        88.6%        73.4%
human           ref                            994      815      989        82.0%        99.5%        82.4%
-------------------------------------------------------------------------------------------------------------------
TOTAL                                         3554     2540     3154        71.5%        88.7%        80.5%
===================================================================================================================
'''