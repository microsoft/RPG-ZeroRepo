import logging
import argparse
import os
import json
from typing import Optional, Union, Dict
from .parse_test import ParseTestFeatures
from zerorepo.rpg_gen.base.llm_client import LLMConfig


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
# Command: parse
# ----------------------------

def run_parse_command(args):
    setup_logging(args.log_file)
    logging.info(f"Starting feature parser for repo: {args.repo_dir}")

    # 加载 LLM 配置
    llm_cfg = None
    if args.llm_config:
        llm_cfg = LLMConfig.from_source(args.llm_config)

    parse_features = ParseTestFeatures(
        repo_dir=args.repo_dir,
        llm_cfg=llm_cfg,
        context_window=args.context_window
    )

    result_dir = os.path.dirname(args.result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    result = parse_features.parse_repo_test(
        max_iterations=args.max_iterations,
        max_workers=args.max_workers,
        output_path=args.result_path
    )

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=4)

    logging.info("Parsing completed.")
    logging.info(f"Total Python files parsed: {len(result)}")


# ----------------------------
# CLI Interface
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Parser (using zerorepo base classes)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: parse
    parse_parser = subparsers.add_parser("parse", help="Parse features from a Python repository")
    parse_parser.add_argument("--repo_dir", required=True, help="Path to the Python code repository")
    parse_parser.add_argument("--log_file", default="./logs/parse.log", help="Log file output path")
    parse_parser.add_argument("--max_iterations", type=int, default=10, help="Max iterations per parsing unit")
    parse_parser.add_argument("--max_workers", type=int, default=4, help="Max number of threads")
    parse_parser.add_argument("--result_path", type=str, default="result_tests/result.json", help="Result save path")
    parse_parser.add_argument("--llm_config", type=str, default=None, help="Path to LLM config file (JSON/YAML)")
    parse_parser.add_argument("--context_window", type=int, default=10, help="Context window for LLM memory")

    args = parser.parse_args()

    if args.command == "parse":
        run_parse_command(args)
    else:
        parser.print_help()


# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    main()
