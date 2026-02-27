#!/usr/bin/env python3
"""
Run script for the Evaluation Framework.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

from .framework.eval_framework import EvaluationFramework
from zerorepo.rpg_gen.base.llm_client import LLMClient, LLMConfig
from zerorepo.rpg_gen.base.rpg import DependencyGraph


# Local library experiment directory configuration
LOCAL_LIBS_BASE = "/home/v-jianwenluo/temp/repo_encoder_exp/human"

# Extra dependencies per library
_DEPS_REQUESTS = ["urllib3", "charset-normalizer", "idna", "certifi", "socks", "PySocks"]
_DEPS_SKLEARN = ["pillow", "matplotlib", "seaborn"]
_DEPS_DJANGO = [
    "jinja2", "asgiref", "sqlparse", "pytz", "tzdata",
    "argon2-cffi", "bcrypt",
]
_DEPS_STATSMODELS = ["matplotlib", "seaborn"]
_DEPS_SYMPY = ["antlr4-python3-runtime", "matplotlib"]
_DEPS_PANDAS = [
    "pytz", "tzdata", "openpyxl", "xlrd", "xlsxwriter",
    "lxml", "html5lib", "beautifulsoup4", "jinja2",
    "matplotlib", "tabulate", "pyarrow",
]

LOCAL_LIBS_REPOS = {
    "HttpEasy": ("requests", "", False, _DEPS_REQUESTS),
    "MLKit-Py": ("sklearn", "", True, _DEPS_SKLEARN),
    "PyWebEngine": ("django", "", False, _DEPS_DJANGO),
    "StatModeler": ("statsmodels", "", True, _DEPS_STATSMODELS),
    "SymbolicMath": ("sympy", "", False, _DEPS_SYMPY),
    "TableKit": ("pandas", "", True, _DEPS_PANDAS),
    "ref/HttpEasy": ("requests", "", False, _DEPS_REQUESTS),
    "ref/MLKit-Py": ("sklearn", "", True, _DEPS_SKLEARN),
    "ref/PyWebEngine": ("django", "", False, _DEPS_DJANGO),
    "ref/StatModeler": ("statsmodels", "", True, _DEPS_STATSMODELS),
    "ref/SymbolicMath": ("sympy", "", False, _DEPS_SYMPY),
    "ref/TableKit": ("pandas", "", True, _DEPS_PANDAS),
}


def get_local_lib_info(method_path: str, container_repo_path: str = "/repo") -> dict:
    """
    Detect if method_path is a local library experiment directory and return config info.
    """
    abs_path = os.path.abspath(method_path)
    if not abs_path.startswith(LOCAL_LIBS_BASE):
        return {}

    rel_path = os.path.relpath(abs_path, LOCAL_LIBS_BASE)

    repo_key = None
    for key in LOCAL_LIBS_REPOS:
        if rel_path == key or rel_path.startswith(key + os.sep):
            repo_key = key
            break

    if repo_key is None:
        return {}

    lib_name, src_subdir, needs_build, extra_deps = LOCAL_LIBS_REPOS[repo_key]

    if src_subdir:
        lib_path = f"{container_repo_path}/{src_subdir}"
    else:
        lib_path = container_repo_path

    return {
        "repo_name": repo_key,
        "lib_name": lib_name,
        "lib_path": lib_path,
        "needs_build": needs_build,
        "extra_deps": extra_deps,
        "workspace_path": os.path.join(abs_path, "workspace"),
    }



def setup_logging(log_dir: str = None, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Framework for Repository Code Localization and Testing"
    )

    # Required arguments
    parser.add_argument("--tasks_file", type=str, required=True,
                        help="Path to tasks JSON file")

    # Optional data arguments
    parser.add_argument("--method_path", type=str, default=None,
                        help="Path to Method's Output")
    # Directory arguments
    parser.add_argument("--cache_dir", type=str, default="./eval_cache",
                        help="Cache directory for results (default: ./eval_cache)")
    parser.add_argument("--mnt_dir", type=str, default="/tmp/workspace",
                        help="Mount directory for Docker (default: /tmp/workspace)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for log files (default: None, console only)")

    # LLM arguments
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model to use for all tasks (overrides model_loc_vote and model_test)")
    parser.add_argument("--model_loc_vote", type=str, default="o3-mini-20250131",
                        help="LLM model for localization and voting (default: o3-mini-20250131)")
    parser.add_argument("--model_test", type=str, default="o3-mini-20250131",
                        help="LLM model for test generation (default: o3-mini-20250131)")

    # Iteration arguments
    parser.add_argument("--max_loc_iters", type=int, default=40,
                        help="Max localization iterations (default: 40)")
    parser.add_argument("--max_coding_iters", type=int, default=15,
                        help="Max coding iterations (default: 15)")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts (default: 5)")

    # Docker arguments
    parser.add_argument("--image_name", type=str, default="zerorepo",
                        help="Docker image name (default: zerorepo)")
    parser.add_argument("--container_name", type=str, default=None,
                        help="Docker container name (default: auto-generated)")

    # Resume/skip arguments
    parser.add_argument("--skip_existing", "--resume", action="store_true",
                        help="Skip tasks that already have results in cache_dir")

    # Other arguments
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir, verbose=args.verbose)
    logger = logging.getLogger("run")


    if not os.path.isfile(args.tasks_file):
        logger.error(f"Tasks file not found: {args.tasks_file}")
        sys.exit(1)

    # Load tasks
    logger.info(f"Loading tasks from: {args.tasks_file}")
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)

    if isinstance(tasks, dict):
        tasks = [tasks]

    logger.info(f"Loaded {len(tasks)} task(s)")

    # Initialize LLM configs
    if args.model:
        model_loc_vote = args.model
        model_test = args.model
    else:
        model_loc_vote = args.model_loc_vote
        model_test = args.model_test

    llm_cfg_loc_vote = LLMConfig(model=model_loc_vote)
    llm_cfg_test = LLMConfig(model=model_test)
    logger.info(f"Using LLM model for localization: {model_loc_vote}")
    logger.info(f"Using LLM model for voting and test generation: {model_test}")

    # Prepare kwargs for Docker
    docker_kwargs = {
        "image_name": args.image_name,
    }
    if args.container_name:
        docker_kwargs["container_name"] = args.container_name

    repo_dir = os.path.join(args.method_path, "workspace")
    repo_data_path = os.path.join(args.method_path, "checkpoints", "repo_data.json")
    repo_rpg_path = os.path.join(args.method_path, "checkpoints", "cur_repo_rpg.json")
    dep_graph_path = os.path.join(args.method_path, "checkpoints", "dep_graph.json")

    dep_graph = DependencyGraph(repo_dir=repo_dir)
    dep_graph.build()
    dep_graph.parse()
    with open(dep_graph_path, 'w') as f:
        json.dump(dep_graph.to_dict(), f, indent=4)

    # Detect if this is a local library experiment
    lib_info = get_local_lib_info(args.method_path)
    if lib_info:
        logger.info(f"Detected local library experiment: {lib_info['repo_name']} ({lib_info['lib_name']})")
        logger.info(f"Container will add to sys.path: {lib_info['lib_path']}")
        docker_kwargs["local_lib_path"] = lib_info["lib_path"]

    # Initialize framework
    logger.info("Initializing EvaluationFramework...")
    framework = EvaluationFramework(
        mnt_dir=args.mnt_dir,
        workspace="/workspace",
        repo_dir=os.path.abspath(repo_dir),
        llm_cfg_loc_vote=llm_cfg_loc_vote,
        llm_cfg_test=llm_cfg_test,
        logger=logger,
        **docker_kwargs
    )

    # Preprocess local library (install dependencies, build, etc.)
    if lib_info:
        if not framework.preprocess_local_lib(lib_info):
            logger.error("Local library preprocessing failed, exiting")
            sys.exit(1)

    # Run evaluation
    if args.skip_existing:
        logger.info("Starting evaluation (skip_existing=True, will resume from cache)...")
    else:
        logger.info("Starting evaluation...")


    results = framework.run(
        cache_dir=args.cache_dir,
        tasks=tasks,
        repo_data_path=repo_data_path,
        rpg_data_path=repo_rpg_path,
        dep_graph_path=dep_graph_path,
        max_loc_iters=args.max_loc_iters,
        max_coding_iters=args.max_coding_iters,
        max_retries=args.max_retries,
        skip_existing=args.skip_existing
    )

    # Summary
    passed = sum(1 for r in results if r.get("voting", False))
    total = len(results)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Tasks passed voting: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Results saved to: {args.cache_dir}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
