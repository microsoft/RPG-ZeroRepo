import os
import json
import argparse
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
from zerorepo.rpg_gen.base.rpg import RPG
from zerorepo.rpg_gen.base.llm_client import LLMConfig
from zerorepo.rpg_encoder import RPGEncoder


def create_logger(
    repo_name: str,
    commit_instance_id: str = None, 
    log_dir: str = "logs"
) -> logging.Logger:

    logger_name = f"{repo_name}_{commit_instance_id if commit_instance_id else 'base'}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if getattr(logger, "_is_configured", False):
        return logger

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"{logger_name}_processing.log")
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger._is_configured = True
    return logger


def build_repo_mapping(
    repo_grouped: str,
    repo_base: str
) -> Dict[str, List[str]]:
    """
    Build mapping: base_repo -> list(commit repo paths)
    """
    base_to_grouped: Dict[str, List[str]] = {}

    for repo_name in os.listdir(repo_base):
        base_repo_path = os.path.join(repo_base, repo_name)
        grouped_path = os.path.join(repo_grouped, repo_name)

        if not (os.path.isdir(base_repo_path) and os.path.isdir(grouped_path)):
            continue

        commit_repo_list = [
            os.path.join(grouped_path, inst, repo_name)
            for inst in os.listdir(grouped_path)
            if os.path.isdir(os.path.join(grouped_path, inst, repo_name))
        ]

        if commit_repo_list:
            base_to_grouped[base_repo_path] = commit_repo_list

    return base_to_grouped


def process_repo_and_commits(
    base_repo_path: str,
    commit_repos: List[str],
    save_dir: str,
    log_dir: str = "logs",
    llm_config: Optional[LLMConfig] = None,
):
    """
    Process a repository and its commits using RPGEncoder.
    - First commit is used as baseline, full RPG extraction
    - Subsequent commits do incremental updates
    """
    repo_name = os.path.basename(base_repo_path)
    base_logger = create_logger(repo_name, log_dir=log_dir)

    # Sort commit repos
    commit_repos = sorted(commit_repos)
    if not commit_repos:
        base_logger.warning("No commit repos found for %s, skip.", repo_name)
        return

    base_logger.info(
        "=== Processing repo: %s (commits: %d) ===",
        repo_name, len(commit_repos)
    )

    # ========== Step 1: First commit as baseline ==========
    first_commit_repo_path = commit_repos[0]
    first_instance_id = os.path.basename(os.path.dirname(first_commit_repo_path))
    first_save_path = os.path.join(save_dir, f"{first_instance_id}.json")

    base_logger.info(
        "[Init] Use first commit as base: repo_dir=%s, instance_id=%s",
        first_commit_repo_path, first_instance_id
    )

    if os.path.exists(first_save_path):
        # Load existing RPG
        base_logger.info("[Base] Found cached artifacts: %s", first_save_path)
        encoder = RPGEncoder.from_saved(
            save_path=first_save_path,
            cur_repo_dir=first_commit_repo_path,
            llm_config=llm_config,
            logger=base_logger
        )
    else:
        # Build new RPG using RPGEncoder
        base_logger.info("[Base] Building RPG for first commit: %s", first_commit_repo_path)

        encoder = RPGEncoder(
            repo_dir=first_commit_repo_path,
            repo_name=repo_name,
            llm_config=llm_config,
            logger=base_logger
        )

        encoder.encode(
            max_repo_info_iters=3,
            max_exclude_votes=3,
            max_parse_iters=20,
            min_batch_tokens=10_000,
            max_batch_tokens=50_000,
            class_context_window=10,
            func_context_window=10,
            max_parse_workers=8,
            refactor_context_window=20,
            refactor_max_iters=20,
            update_dep_graph=True,
        )

        encoder.save(first_save_path)
        base_logger.info("[Base Saved] → %s", first_save_path)

    prev_repo_path = first_commit_repo_path

    # ========== Step 2: Process subsequent commits with diff ==========
    for commit_repo_path in tqdm(
        commit_repos[1:],
        desc=f"Commits for {repo_name}",
        ncols=100
    ):
        commit_instance_id = os.path.basename(os.path.dirname(commit_repo_path))
        save_path = os.path.join(save_dir, f"{commit_instance_id}.json")
        commit_logger = create_logger(repo_name, commit_instance_id, log_dir=log_dir)

        if os.path.exists(save_path):
            commit_logger.info("[Skip] %s already processed → %s", commit_instance_id, save_path)
            # Load the skipped version for next iteration
            encoder = RPGEncoder.from_saved(
                save_path=save_path,
                cur_repo_dir=commit_repo_path,
                llm_config=llm_config,
                logger=commit_logger
            )
            prev_repo_path = commit_repo_path
            continue

        commit_logger.info("[Diff] Start processing commit instance: %s", commit_instance_id)

        try:
            # Update encoder's repo_dir to current commit
            encoder.repo_dir = commit_repo_path
            encoder.logger = commit_logger

            # Perform incremental update
            encoder.update(
                last_repo_dir=prev_repo_path,
                update_dep_graph=True
            )

            # Save updated RPG
            encoder.save(save_path)

            commit_logger.info(
                "[Cur Feature Tree]: %s",
                json.dumps(encoder.feature_tree, indent=4)
            )
            commit_logger.info("[OK] Saved → %s", save_path)

            prev_repo_path = commit_repo_path

        except Exception as e:
            commit_logger.exception("[Error] %s: %s", commit_instance_id, e)
            continue


def main():
    parser = argparse.ArgumentParser(description="Parse SWE-bench repositories into RPG format using RPGEncoder")
    parser.add_argument(
        "--repo-grouped",
        type=str,
        default="/mnt/jianwen/RepoEncoder/exp/exp_repos/swe_bench_live/repos_grouped",
        help="Root directory for grouped repositories (by instance)"
    )
    parser.add_argument(
        "--repo-base",
        type=str,
        default="/mnt/jianwen/RepoEncoder/exp/exp_repos/swe_bench_live/repos_base",
        help="Root directory for base repositories"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/mnt/jianwen/RepoEncoder/output/swe_bench_live_repo_graph",
        help="Directory to save parsed RPG outputs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Specific repository name to process (optional, process all if not specified)"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Specific instance ID to process (optional)"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        help="Path to LLM configuration file (YAML/JSON). "
             "Supports all LLMConfig fields: model, provider, api_key, base_url, etc."
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load LLM config
    llm_config = LLMConfig.from_source(args.llm_config) if args.llm_config else LLMConfig()

    # Reduce third-party library noise
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.WARNING)

    # Build repo mapping
    base_to_grouped = build_repo_mapping(args.repo_grouped, args.repo_base)
    print(f"Discovered {len(base_to_grouped)} base repos to process.")

    # Filter by repo name if specified
    if args.repo_name:
        filtered = {
            k: v for k, v in base_to_grouped.items()
            if os.path.basename(k) == args.repo_name
        }
        if not filtered:
            print(f"[ERROR] Repository '{args.repo_name}' not found")
            return
        base_to_grouped = filtered

    # Filter by instance ID if specified
    if args.instance_id:
        filtered = {}
        for base_repo, commits in base_to_grouped.items():
            matching = [
                c for c in commits
                if os.path.basename(os.path.dirname(c)) == args.instance_id
            ]
            if matching:
                filtered[base_repo] = matching
        if not filtered:
            print(f"[ERROR] Instance ID '{args.instance_id}' not found")
            return
        base_to_grouped = filtered

    # Process repositories
    for base_repo_path, commit_repos in base_to_grouped.items():
        process_repo_and_commits(
            base_repo_path=base_repo_path,
            commit_repos=commit_repos,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            llm_config=llm_config,
        )


if __name__ == "__main__":
    main()
