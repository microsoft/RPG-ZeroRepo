#!/usr/bin/env python3
"""
Standalone script for rebuilding repositories using parsed RPG data.
This script uses the Rebuild class to reconstruct repositories with different preservation modes.
"""

import os
import json
import logging
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

from zerorepo.rpg_gen.base.llm_client import LLMConfig
from zerorepo.config.checkpoint_config import CheckpointFiles
from zerorepo.rpg_encoder.rebuild import Rebuild, RebuildConfig, RebuildMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild repository using different preservation modes based on parsed RPG."
    )

    # Required arguments
    parser.add_argument(
        "--repo-dir",
        type=str,
        required=True,
        help="Path to the source repository directory to rebuild from.",
    )
    parser.add_argument(
        "--repo-name", 
        type=str,
        required=True,
        help="Repository name.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory for checkpoints and intermediate files.",
    )
    
    # Rebuild mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["feature_only", "feature_file", "full_preserve"],
        default="feature_only",
        help="Rebuild mode: 'feature_only' (redesign files and functions), "
             "'feature_file' (preserve files, redesign functions), "
             "'full_preserve' (preserve all, only plan data flow)",
    )
    
    # RPG extraction control
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip RPG parsing and load from existing checkpoint data. "
             "Requires repo_data.json, skeleton.json, and global_repo_rpg.json in checkpoint-dir.",
    )
    
    
    # LLM configuration
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        help="Path to LLM configuration file (YAML/JSON). "
             "See LLMConfig for supported fields: model, provider, api_key, base_url, etc.",
    )
    
    # Configuration files
    parser.add_argument(
        "--skeleton-cfg",
        type=str,
        default="",
        help="Path to skeleton configuration file",
    )
    parser.add_argument(
        "--graph-cfg",
        type=str,
        default="",
        help="Path to graph configuration file",
    )
    
    # Data flow analysis
    parser.add_argument(
        "--run-data-flow",
        action="store_true",
        help="Run data flow analysis (recommended for full_preserve mode)",
    )
    parser.add_argument(
        "--data-flow-max-results",
        type=int,
        default=3,
        help="Max results for data flow analysis",
    )
    
    # RPG parsing parameters (used when extracting from source)
    parser.add_argument(
        "--max-repo-info-iters",
        type=int,
        default=3,
        help="Max iterations for collecting repo info.",
    )
    parser.add_argument(
        "--max-exclude-votes", 
        type=int,
        default=3,
        help="Max exclude votes in parsing.",
    )
    parser.add_argument(
        "--max-parse-iters",
        type=int,
        default=10,
        help="Max parse iterations.",
    )
    parser.add_argument(
        "--class-chunk-size",
        type=int,
        default=10,
        help="Class chunk size.",
    )
    parser.add_argument(
        "--class-context-window",
        type=int,
        default=10,
        help="Class context window.",
    )
    parser.add_argument(
        "--func-context-window",
        type=int,
        default=10,
        help="Function context window.",
    )
    parser.add_argument(
        "--max-parse-workers",
        type=int,
        default=4,
        help="Max number of parse workers.",
    )
    parser.add_argument(
        "--refactor-context-window",
        type=int,
        default=10,
        help="Refactor context window.",
    )
    parser.add_argument(
        "--refactor-max-iters",
        type=int,
        default=10,
        help="Max refactor iterations.",
    )
    
    # NEW: logfile
    parser.add_argument(
        "--log-file",
        type=str,
        default="zerorepo.log",
        help=(
            "Path to log file. If a filename is provided (no directory), "
            "it will be created under --checkpoint-dir. "
            "If omitted, defaults to <checkpoint-dir>/rebuild.log."
        ),
    )


    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(checkpoint_dir: str, log_level: str, logfile: str | None = None) -> logging.Logger:
    """Setup logging with both console and file output."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = logging.getLogger("RebuildRPG")
    # Set logger to DEBUG so file can capture all messages, handlers will filter
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid double logging via root

    # IMPORTANT: clear old handlers (e.g., in notebooks / repeated runs)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    level_value = getattr(logging, log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_value)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Resolve logfile path
    if logfile is None:
        resolved_logfile = os.path.join(checkpoint_dir, "rebuild.log")
    else:
        # If user passed a bare filename, place it under checkpoint_dir
        if (not os.path.isabs(logfile)) and (os.path.dirname(logfile) == ""):
            resolved_logfile = os.path.join(checkpoint_dir, logfile)
        else:
            resolved_logfile = logfile

    os.makedirs(os.path.dirname(resolved_logfile) or ".", exist_ok=True)

    # File handler (keep DEBUG in file for troubleshooting)
    file_handler = logging.FileHandler(resolved_logfile, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized. logfile=%s, console_level=%s, file_level=DEBUG", resolved_logfile, log_level)
    return logger


def validate_checkpoint_data(checkpoint_dir: str, logger: logging.Logger) -> bool:
    """Validate that required checkpoint files exist for skip-parse mode"""
    required_files = ["repo_data.json", "skeleton.json", "global_repo_rpg.json"]
    
    for file_name in required_files:
        file_path = os.path.join(checkpoint_dir, file_name)
        if not os.path.exists(file_path):
            logger.error(f"Required checkpoint file not found: {file_path}")
            return False
        
        # Check if file is valid JSON
        try:
            with open(file_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in checkpoint file {file_path}: {e}")
            return False
    
    logger.info("✅ All required checkpoint files found and valid")
    return True




def save_rebuild_results(
    checkpoint_dir: str,
    task_batches: list,
    rebuild_mode: str,
    logger: logging.Logger
) -> None:
    """Save rebuild results and summary"""
    
    # Create summary
    summary = {
        "rebuild_mode": rebuild_mode,
        "total_task_batches": len(task_batches),
        "checkpoint_dir": str(checkpoint_dir),
        "task_batches": []
    }
    
    # Process each task batch
    for i, batch in enumerate(task_batches):
        batch_info = {
            "index": i,
            "id": getattr(batch, 'id', f"batch_{i}"),
            "total_tasks": len(getattr(batch, 'tasks', [])),
            "description": getattr(batch, 'description', 'N/A'),
        }
        
        # Extract task details if available
        if hasattr(batch, 'tasks'):
            batch_info["tasks"] = [
                {
                    "id": getattr(task, 'id', f"task_{j}"),
                    "type": getattr(task, 'type', 'unknown'),
                    "target": getattr(task, 'target', 'N/A'),
                }
                for j, task in enumerate(batch.tasks)
            ]
        
        summary["task_batches"].append(batch_info)
    
    # Save summary to checkpoint directory
    summary_path = os.path.join(checkpoint_dir, "rebuild_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Rebuild summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(checkpoint_dir), args.log_level, args.log_file)
    
    logger.info("=== Starting Repository Rebuild Process ===")
    logger.info(f"Source repository: {args.repo_dir}")
    logger.info(f"Repository name: {args.repo_name}")
    logger.info(f"Rebuild mode: {args.mode}")
    logger.info(f"Skip parse: {args.skip_parse}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Validate checkpoint data if skip-parse is enabled
    if args.skip_parse:
        logger.info("Skip-parse mode enabled, validating existing checkpoint data...")
        if not validate_checkpoint_data(str(checkpoint_dir), logger):
            logger.error("❌ Checkpoint validation failed. Cannot use skip-parse mode.")
            logger.error("Either provide valid checkpoint data or remove --skip-parse flag.")
            return
        logger.info("✅ Checkpoint data validation passed")
    else:
        logger.info("Parse mode enabled, will extract RPG from repository")
    
    # Create LLM config
    if args.llm_config:
        llm_config = LLMConfig.from_source(args.llm_config)
    else:
        llm_config = LLMConfig()
    logger.info(f"LLM config: model={llm_config.model}, provider={llm_config.resolve_provider()}")
    
    # Create rebuild config
    rebuild_mode_map = {
        "feature_only": RebuildMode.FEATURE_ONLY,
        "feature_file": RebuildMode.FEATURE_FILE,
        "full_preserve": RebuildMode.FULL_PRESERVE
    }
    
    rebuild_config = RebuildConfig(
        mode=rebuild_mode_map[args.mode],
        llm_config=llm_config,
        skeleton_cfg_path=args.skeleton_cfg,
        graph_cfg_path=args.graph_cfg,
        # RPG parsing parameters
        max_repo_info_iters=args.max_repo_info_iters,
        max_exclude_votes=args.max_exclude_votes,
        max_parse_iters=args.max_parse_iters,
        class_chunk_size=args.class_chunk_size,
        class_context_window=args.class_context_window,
        func_context_window=args.func_context_window,
        max_parse_workers=args.max_parse_workers,
        refactor_context_window=args.refactor_context_window,
        refactor_max_iters=args.refactor_max_iters,
        # Data flow analysis
        run_data_flow_analysis=args.run_data_flow,
        data_flow_max_results=args.data_flow_max_results,
        # Parse control
        skip_parse=args.skip_parse
    )
    
    # Create checkpoint config
    checkpoint_config = CheckpointFiles(
        repo_data="repo_data.json",
        skeleton="skeleton.json",
        global_repo_rpg="global_repo_rpg.json",
        current_repo_rpg="current_rpg.json",
        graph="graph.json",
        tasks="tasks.json",
        task_manager_state="task_manager_state.json"
    )
    
    # Initialize Rebuild
    rebuild = Rebuild(
        repo_dir=args.repo_dir,
        repo_name=args.repo_name,
        checkpoint_dir=str(checkpoint_dir),
        config=rebuild_config,
        checkpoint_config=checkpoint_config,
        logger=logger
    )
    
    try:
        # Run the rebuild process
        logger.info("Starting rebuild process...")
        task_batches = rebuild.rebuild()
        
        logger.info("=== Rebuild Process Completed Successfully ===")
        logger.info(f"Generated {len(task_batches)} task batches")
        
        # Save results
        save_rebuild_results(
            checkpoint_dir=str(checkpoint_dir),
            task_batches=task_batches,
            rebuild_mode=args.mode,
            logger=logger
        )
        
        logger.info(f"Results saved to: {checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    logger.info("=== Rebuild Process Finished ===")


if __name__ == "__main__":
    main()