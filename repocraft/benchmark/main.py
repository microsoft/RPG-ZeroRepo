"""
Unified CLI for the RepoCraft benchmark construction pipeline.

Pipeline: parse_test -> refactor -> sample -> generate_query

Usage:
    # Run individual stages
    python -m repocraft.benchmark.main parse --repo_dir /path/to/repo --result_path ./results/parsed.json
    python -m repocraft.benchmark.main refactor --parsed_test ./results/parsed.json --result_path ./results/refactored.json
    python -m repocraft.benchmark.main sample --refactored_test ./results/refactored.json --result_path ./results/sampled.json
    python -m repocraft.benchmark.main generate --sampled_test ./results/sampled.json --parsed_test ./results/parsed.json --result_path ./results/tasks.json

    # Run the full pipeline end-to-end
    python -m repocraft.benchmark.main pipeline --repo_dir /path/to/repo --output_dir ./results --repo_name sklearn
"""

import logging
import argparse
import os
import json
from typing import Optional, Union, Dict

from zerorepo.rpg_gen.base.llm_client import LLMConfig

from .parse_test import ParseTestFeatures
from .refactor_test_tree import TestClassifier
from .sample import sample_tests, count_sampled_algorithms
from .generate_query import batch_generate_queries


# ----------------------------
# Logging Configuration
# ----------------------------

def setup_logging(log_file_path: str = "parse.log"):
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        ]
    )


# ----------------------------
# Stage 1: Parse Test Tree
# ----------------------------

def run_parse(
    repo_dir: str,
    result_path: str,
    llm_cfg=None,
    max_iterations: int = 10,
    max_workers: int = 4,
    context_window: int = 10,
):
    """Parse a repository's test suite into a feature-grouped test tree."""
    logging.info(f"[Stage 1] Parsing test tree from repo: {repo_dir}")

    parse_features = ParseTestFeatures(
        repo_dir=repo_dir,
        llm_cfg=llm_cfg,
        context_window=context_window
    )

    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    result = parse_features.parse_repo_test(
        max_iterations=max_iterations,
        max_workers=max_workers,
        output_path=result_path
    )

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    logging.info(f"[Stage 1] Parsing completed. Total files: {len(result)}")
    logging.info(f"[Stage 1] Result saved to: {result_path}")
    return result


# ----------------------------
# Stage 2: Refactor Test Tree
# ----------------------------

def run_refactor(
    parsed_test_path: str,
    result_path: str,
):
    """Refactor flat test tree into categorized tree by directory structure."""
    logging.info(f"[Stage 2] Refactoring test tree from: {parsed_test_path}")

    with open(parsed_test_path, 'r') as f:
        parsed_tree = json.load(f)

    refactored_tree = TestClassifier.build_classification_tree(parsed_tree)

    data = {
        "files": parsed_tree,
        "refactor": refactored_tree
    }

    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    with open(result_path, 'w') as f:
        json.dump(data, f, indent=4)

    num_categories = len(refactored_tree)
    total_modules = sum(len(modules) for modules in refactored_tree.values())
    logging.info(f"[Stage 2] Refactoring completed. Categories: {num_categories}, Modules: {total_modules}")
    logging.info(f"[Stage 2] Result saved to: {result_path}")
    return data


# ----------------------------
# Stage 3: Sample Tests
# ----------------------------

def run_sample(
    refactored_test_path: str,
    result_path: str,
    num_files: int = 12,
    num_classes_per_file: int = 20,
    num_modules_per_class: int = 10,
):
    """Apply hierarchical 3-level sampling to the refactored test tree."""
    logging.info(f"[Stage 3] Sampling tests from: {refactored_test_path}")

    with open(refactored_test_path, 'r') as f:
        data = json.load(f)

    refactored_tree = data["refactor"]

    sampled_data = {}
    for category, test_data in refactored_tree.items():
        sampled_data[category] = sample_tests(
            category=category,
            test_data=test_data,
            num_files=num_files,
            num_classes_or_functions_per_file=num_classes_per_file,
            num_modules_per_class=num_modules_per_class
        )

    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    with open(result_path, 'w') as f:
        json.dump(sampled_data, f, indent=4)

    logging.info(f"[Stage 3] Sampling completed.")
    count_sampled_algorithms(sampled_data)
    logging.info(f"[Stage 3] Result saved to: {result_path}")
    return sampled_data


# ----------------------------
# Stage 4: Generate Task Queries
# ----------------------------

def run_generate(
    sampled_test_path: str,
    parsed_test_path: str,
    result_path: str,
    llm_cfg=None,
    max_workers: int = 6,
):
    """Generate natural language task queries for each sampled test group."""
    logging.info(f"[Stage 4] Generating task queries from: {sampled_test_path}")

    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    batch_generate_queries(
        sampled_test_json=sampled_test_path,
        result_test_json=parsed_test_path,
        output_json=result_path,
        llm_cfg=llm_cfg,
        max_workers=max_workers
    )

    logging.info(f"[Stage 4] Query generation completed.")
    logging.info(f"[Stage 4] Result saved to: {result_path}")


# ----------------------------
# Full Pipeline
# ----------------------------

def run_pipeline(
    repo_dir: str,
    output_dir: str,
    repo_name: str,
    llm_cfg=None,
    max_iterations: int = 10,
    max_parse_workers: int = 4,
    num_files: int = 12,
    num_classes_per_file: int = 20,
    num_modules_per_class: int = 10,
    max_query_workers: int = 6,
    context_window: int = 10,
):
    """Run the full benchmark construction pipeline: parse -> refactor -> sample -> generate."""
    os.makedirs(output_dir, exist_ok=True)

    parsed_path = os.path.join(output_dir, f"result_tests/{repo_name}.json")
    refactored_path = os.path.join(output_dir, f"refactored_test/{repo_name}.json")
    sampled_path = os.path.join(output_dir, f"sampled_test/sample_{repo_name}.json")
    tasks_path = os.path.join(output_dir, f"task_results/{repo_name}.json")

    logging.info(f"=== RepoCraft Benchmark Pipeline for '{repo_name}' ===")
    logging.info(f"Repository: {repo_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Stage 1: Parse
    run_parse(
        repo_dir=repo_dir,
        result_path=parsed_path,
        llm_cfg=llm_cfg,
        max_iterations=max_iterations,
        max_workers=max_parse_workers,
        context_window=context_window,
    )

    # Stage 2: Refactor
    run_refactor(
        parsed_test_path=parsed_path,
        result_path=refactored_path,
    )

    # Stage 3: Sample
    run_sample(
        refactored_test_path=refactored_path,
        result_path=sampled_path,
        num_files=num_files,
        num_classes_per_file=num_classes_per_file,
        num_modules_per_class=num_modules_per_class,
    )

    # Stage 4: Generate Queries
    run_generate(
        sampled_test_path=sampled_path,
        parsed_test_path=parsed_path,
        result_path=tasks_path,
        llm_cfg=llm_cfg,
        max_workers=max_query_workers,
    )

    logging.info(f"=== Pipeline completed. Tasks saved to: {tasks_path} ===")


# ----------------------------
# CLI Interface
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RepoCraft Benchmark Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run individual stages
  python -m repocraft.benchmark.main parse --repo_dir /path/to/sklearn --result_path ./results/parsed.json
  python -m repocraft.benchmark.main refactor --parsed_test ./results/parsed.json --result_path ./results/refactored.json
  python -m repocraft.benchmark.main sample --refactored_test ./results/refactored.json --result_path ./results/sampled.json
  python -m repocraft.benchmark.main generate --sampled_test ./results/sampled.json --parsed_test ./results/parsed.json --result_path ./results/tasks.json

  # Run the full pipeline
  python -m repocraft.benchmark.main pipeline --repo_dir /path/to/sklearn --output_dir ./results --repo_name sklearn
        """
    )

    # Common args
    parser.add_argument("--log_file", default="./logs/benchmark.log", help="Log file path")
    parser.add_argument("--llm_config", type=str, default=None, help="Path to LLM config file (JSON/YAML)")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stages")

    # --- parse ---
    parse_p = subparsers.add_parser("parse", help="Stage 1: Parse test tree from repository")
    parse_p.add_argument("--repo_dir", required=True, help="Path to the repository")
    parse_p.add_argument("--result_path", required=True, help="Output JSON path for parsed test tree")
    parse_p.add_argument("--max_iterations", type=int, default=10, help="Max LLM iterations per unit")
    parse_p.add_argument("--max_workers", type=int, default=4, help="Parallel workers for parsing")
    parse_p.add_argument("--context_window", type=int, default=10, help="LLM memory context window")

    # --- refactor ---
    refactor_p = subparsers.add_parser("refactor", help="Stage 2: Refactor flat test tree into categorized tree")
    refactor_p.add_argument("--parsed_test", required=True, help="Path to parsed test tree JSON (from Stage 1)")
    refactor_p.add_argument("--result_path", required=True, help="Output JSON path for refactored tree")

    # --- sample ---
    sample_p = subparsers.add_parser("sample", help="Stage 3: Hierarchical test sampling")
    sample_p.add_argument("--refactored_test", required=True, help="Path to refactored test JSON (from Stage 2)")
    sample_p.add_argument("--result_path", required=True, help="Output JSON path for sampled tests")
    sample_p.add_argument("--num_files", type=int, default=12, help="Level 1: files to sample per category")
    sample_p.add_argument("--num_classes_per_file", type=int, default=20, help="Level 2: classes/functions per file")
    sample_p.add_argument("--num_modules_per_class", type=int, default=10, help="Level 3: features per class")

    # --- generate ---
    generate_p = subparsers.add_parser("generate", help="Stage 4: Generate task queries with LLM")
    generate_p.add_argument("--sampled_test", required=True, help="Path to sampled test JSON (from Stage 3)")
    generate_p.add_argument("--parsed_test", required=True, help="Path to parsed test tree JSON (from Stage 1)")
    generate_p.add_argument("--result_path", required=True, help="Output JSON path for task queries")
    generate_p.add_argument("--max_workers", type=int, default=6, help="Parallel workers for query generation")

    # --- pipeline (full) ---
    pipeline_p = subparsers.add_parser("pipeline", help="Run the full pipeline: parse -> refactor -> sample -> generate")
    pipeline_p.add_argument("--repo_dir", required=True, help="Path to the repository")
    pipeline_p.add_argument("--output_dir", required=True, help="Base output directory for all results")
    pipeline_p.add_argument("--repo_name", required=True, help="Short name for the repo (e.g. sklearn, pandas)")
    pipeline_p.add_argument("--max_iterations", type=int, default=10, help="Max LLM iterations per unit (parse)")
    pipeline_p.add_argument("--max_parse_workers", type=int, default=4, help="Parallel workers for parsing")
    pipeline_p.add_argument("--num_files", type=int, default=12, help="Level 1: files to sample per category")
    pipeline_p.add_argument("--num_classes_per_file", type=int, default=20, help="Level 2: classes/functions per file")
    pipeline_p.add_argument("--num_modules_per_class", type=int, default=10, help="Level 3: features per class")
    pipeline_p.add_argument("--max_query_workers", type=int, default=6, help="Parallel workers for query generation")
    pipeline_p.add_argument("--context_window", type=int, default=10, help="LLM memory context window")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.log_file)

    # Load LLM config if provided
    llm_cfg = None
    if args.llm_config:
        llm_cfg = LLMConfig.from_source(args.llm_config)

    if args.command == "parse":
        run_parse(
            repo_dir=args.repo_dir,
            result_path=args.result_path,
            llm_cfg=llm_cfg,
            max_iterations=args.max_iterations,
            max_workers=args.max_workers,
            context_window=args.context_window,
        )

    elif args.command == "refactor":
        run_refactor(
            parsed_test_path=args.parsed_test,
            result_path=args.result_path,
        )

    elif args.command == "sample":
        run_sample(
            refactored_test_path=args.refactored_test,
            result_path=args.result_path,
            num_files=args.num_files,
            num_classes_per_file=args.num_classes_per_file,
            num_modules_per_class=args.num_modules_per_class,
        )

    elif args.command == "generate":
        run_generate(
            sampled_test_path=args.sampled_test,
            parsed_test_path=args.parsed_test,
            result_path=args.result_path,
            llm_cfg=llm_cfg,
            max_workers=args.max_workers,
        )

    elif args.command == "pipeline":
        run_pipeline(
            repo_dir=args.repo_dir,
            output_dir=args.output_dir,
            repo_name=args.repo_name,
            llm_cfg=llm_cfg,
            max_iterations=args.max_iterations,
            max_parse_workers=args.max_parse_workers,
            num_files=args.num_files,
            num_classes_per_file=args.num_classes_per_file,
            num_modules_per_class=args.num_modules_per_class,
            max_query_workers=args.max_query_workers,
            context_window=args.context_window,
        )


if __name__ == "__main__":
    main()
