"""
RPG Parser CLI

Supports two modes:
1. parse: Parse a repository from scratch into RPG structure
2. update: Incrementally update an existing RPG based on repository changes

Usage:
    # Initial parse
    python parse_rpg.py parse --repo-dir /path/to/repo --repo-name myrepo --save-dir ./output

    # Incremental update
    python parse_rpg.py update --repo-dir /path/to/updated/repo --last-repo-dir /path/to/old/repo \
        --load-path ./output/rpg_encoder.json --save-dir ./output
"""

import os
import json
import logging
import argparse
from typing import Optional
from zerorepo.rpg_encoder import RPGEncoder
from zerorepo.rpg_gen.base.llm_client import LLMConfig
from zerorepo.utils.tree import get_all_leaf_paths, apply_changes, convert_leaves_to_list

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse or update RPG structure from a Python repository."
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # ============ Common arguments ============
    def add_common_args(subparser):
        subparser.add_argument(
            "--repo-dir",
            type=str,
            required=True,
            help="Path to the target repository root directory.",
        )
        subparser.add_argument(
            "--repo-name",
            type=str,
            help="Repository name. If not provided, will be inferred from repo-dir or loaded file.",
        )
        subparser.add_argument(
            "--save-dir",
            type=str,
            required=True,
            help="Directory to save outputs / logs.",
        )
        subparser.add_argument(
            "--logfile",
            type=str,
            default="parse.log",
            help="Path to log file (default: parse.log under save-dir).",
        )
        # LLM Configuration
        subparser.add_argument(
            "--llm-config", type=str, default=None,
            help="Path to LLM configuration file (YAML/JSON). "
                 "Supports all LLMConfig fields: model, provider, api_key, base_url, etc."
        )
        # Dependency graph
        subparser.add_argument("--no-dep-graph", action="store_true", help="Skip dependency graph update")

    # ============ Parse subcommand ============
    parse_parser = subparsers.add_parser("parse", help="Parse repository from scratch")
    add_common_args(parse_parser)

    # Parse-specific arguments
    parse_parser.add_argument("--repo-info", type=str, default=None, help="Optional repository description")
    parse_parser.add_argument("--max-repo-info-iters", type=int, default=3)
    parse_parser.add_argument("--max-exclude-votes", type=int, default=3)
    parse_parser.add_argument("--max-parse-iters", type=int, default=20)
    # Batch token parameters
    parse_parser.add_argument("--min-batch-tokens", type=int, default=10_000)
    parse_parser.add_argument("--max-batch-tokens", type=int, default=50_000)
    parse_parser.add_argument("--summary-min-batch-tokens", type=int, default=10_000)
    parse_parser.add_argument("--summary-max-batch-tokens", type=int, default=50_000)
    # Context windows
    parse_parser.add_argument("--class-context-window", type=int, default=10)
    parse_parser.add_argument("--func-context-window", type=int, default=10)
    parse_parser.add_argument("--max-parse-workers", type=int, default=4)
    parse_parser.add_argument("--refactor-context-window", type=int, default=20)
    parse_parser.add_argument("--refactor-max-iters", type=int, default=20)

    # ============ Update subcommand ============
    update_parser = subparsers.add_parser("update", help="Incrementally update existing RPG")
    add_common_args(update_parser)

    # Update-specific arguments
    update_parser.add_argument(
        "--last-repo-dir",
        type=str,
        required=True,
        help="Path to the previous version of the repository.",
    )
    update_parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to load existing RPG encoder state (rpg_encoder.json).",
    )

    return parser.parse_args()


def _setup_logging(save_dir: str, logfile: Optional[str]) -> Optional[str]:
    """Setup logging to console + optional file. Returns resolved logfile path (or None)."""
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)

    resolved_logfile: Optional[str] = None
    if logfile:
        if not os.path.isabs(logfile) and os.path.dirname(logfile) == "":
            resolved_logfile = os.path.join(save_dir, logfile)
        else:
            resolved_logfile = logfile

        os.makedirs(os.path.dirname(resolved_logfile) or ".", exist_ok=True)

        file_handler = logging.FileHandler(resolved_logfile, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    return resolved_logfile


def _save_outputs(encoder: RPGEncoder, save_dir: str, logger: logging.Logger) -> None:
    """Save RPG encoder outputs to files."""

    # Save encoder state (can be loaded for updates)
    encoder_path = os.path.join(save_dir, "rpg_encoder.json")
    encoder.save(encoder_path)

    # Generate and save feature tree in legacy format
    components = encoder.rpg.get_functionality_graph() if encoder.rpg else []

    all_paths = []
    for cmpt in components:
        name = cmpt["name"]
        refactored_tree = cmpt.get("refactored_subtree", {})
        subtree_paths = get_all_leaf_paths(refactored_tree)
        subtree_paths = [name + "/" + path for path in subtree_paths]
        all_paths.extend(subtree_paths)

    feature_tree = apply_changes({}, all_paths)
    feature_tree = convert_leaves_to_list(feature_tree)

    repo_data = {
        "repository_name": encoder.repo_name,
        "repository_purpose": encoder.repo_info,
        "Feature_tree": feature_tree,
        "Component": components,
    }

    # Save individual files
    repo_data_path = os.path.join(save_dir, "repo_data.json")
    rpg_path = os.path.join(save_dir, "global_repo_rpg.json")
    skeleton_path = os.path.join(save_dir, "skeleton.json")

    if encoder.skeleton:
        skeleton_dict = encoder.skeleton.to_dict()
        with open(skeleton_path, "w", encoding="utf-8") as f:
            json.dump(skeleton_dict, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved skeleton to {skeleton_path}")

    with open(repo_data_path, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved repo_data to {repo_data_path}")

    if encoder.rpg:
        rpg_dict = encoder.rpg.to_dict(include_dep_graph=True)
        with open(rpg_path, "w", encoding="utf-8") as f:
            json.dump(rpg_dict, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved RPG to {rpg_path}")

    # Print stats
    stats = encoder.get_stats()
    logger.info(f"RPG Stats: {json.dumps(stats, indent=2)}")


def run_parse(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run initial RPG parsing."""
    repo_dir = args.repo_dir
    repo_name = args.repo_name or os.path.basename(os.path.abspath(repo_dir))
    save_dir = args.save_dir

    logger.info(f"=== Starting RPG Parse ===")
    logger.info(f"repo_dir={repo_dir}, repo_name={repo_name}, save_dir={save_dir}")

    # Create LLM config
    llm_config = LLMConfig.from_source(args.llm_config) if args.llm_config else LLMConfig()
    logger.info(f"LLM Config: model={llm_config.model}, provider={llm_config.resolve_provider()}")

    # Create encoder
    encoder = RPGEncoder(
        repo_dir=repo_dir,
        repo_name=repo_name,
        repo_info=args.repo_info,
        llm_config=llm_config,
    )

    # Run encoding
    rpg, feature_tree, skeleton = encoder.encode(
        max_repo_info_iters=args.max_repo_info_iters,
        max_exclude_votes=args.max_exclude_votes,
        max_parse_iters=args.max_parse_iters,
        min_batch_tokens=args.min_batch_tokens,
        max_batch_tokens=args.max_batch_tokens,
        summary_min_batch_tokens=args.summary_min_batch_tokens,
        summary_max_batch_tokens=args.summary_max_batch_tokens,
        class_context_window=args.class_context_window,
        func_context_window=args.func_context_window,
        max_parse_workers=args.max_parse_workers,
        refactor_context_window=args.refactor_context_window,
        refactor_max_iters=args.refactor_max_iters,
        update_dep_graph=not args.no_dep_graph,
    )

    # Save outputs
    _save_outputs(encoder, save_dir, logger)
    logger.info("=== RPG Parse Complete ===")


def run_update(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run incremental RPG update."""
    repo_dir = args.repo_dir
    last_repo_dir = args.last_repo_dir
    load_path = args.load_path
    save_dir = args.save_dir

    logger.info(f"=== Starting RPG Update ===")
    logger.info(f"cur_repo_dir={repo_dir}, last_repo_dir={last_repo_dir}")
    logger.info(f"load_path={load_path}, save_dir={save_dir}")

    # Create LLM config
    llm_config = LLMConfig.from_source(args.llm_config) if args.llm_config else LLMConfig()
    logger.info(f"LLM Config: model={llm_config.model}, provider={llm_config.resolve_provider()}")

    # Load existing encoder
    encoder = RPGEncoder.from_saved(
        save_path=load_path,
        cur_repo_dir=repo_dir,
        llm_config=llm_config,
    )

    # Override repo_name if provided
    if args.repo_name:
        encoder.repo_name = args.repo_name

    logger.info(f"Loaded RPG with {len(encoder.rpg.nodes) if encoder.rpg else 0} nodes")

    # Run update
    rpg = encoder.update(
        last_repo_dir=last_repo_dir,
        update_dep_graph=not args.no_dep_graph,
    )

    # Save outputs
    _save_outputs(encoder, save_dir, logger)
    logger.info("=== RPG Update Complete ===")


def main() -> None:
    args = parse_args()

    if not args.mode:
        print("Error: Please specify a mode (parse or update)")
        print("Usage: python parse_rpg.py parse --help")
        print("       python parse_rpg.py update --help")
        return

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    resolved_logfile = _setup_logging(save_dir=save_dir, logfile=args.logfile)
    logger = logging.getLogger(__name__)

    if resolved_logfile:
        logger.info(f"Logging to file: {resolved_logfile}")

    if args.mode == "parse":
        run_parse(args, logger)
    elif args.mode == "update":
        run_update(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
