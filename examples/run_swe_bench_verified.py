import os

# Must be set before importing zerorepo (envs.py reads them at import time)
os.environ["ANSWER_START_TAG"] = "<action>"
os.environ["ANSWER_END_TAG"] = "</action>"

import json
import json5
import jsonlines
import logging
import argparse
import multiprocessing
from typing import Dict
from datasets import load_dataset
from zerorepo.rpg_gen.base.rpg import RPG
from zerorepo.rpg_encoder.rpg_agent import RPGAgent
from zerorepo.rpg_gen.base.llm_client import LLMConfig
from zerorepo.rpg_encoder.rpg_agent.tools import (
    SearchCodeSnippets,
    SearchCodeByFeatures,
    SearchNode,
    FetchNode,
    ExploreRPG,
    Terminate
)

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Run RPG Agent on SWE-bench")
parser.add_argument("--model", type=str, default=None, help="Model name (overrides --llm-config model)")
parser.add_argument("--llm-config", type=str, default=None,
                    help="Path to LLM configuration file (YAML/JSON). "
                         "Supports all LLMConfig fields: model, provider, api_key, base_url, etc.")
parser.add_argument("--num_processes", type=int, default=2, help="Number of parallel processes to use")
parser.add_argument("--run_id", type=str, default="exp_1", help="Run ID for experiment tracking")
parser.add_argument("--benchmark", type=str, default="princeton-nlp/SWE-bench_Verified",
                    help="HuggingFace dataset name (e.g., princeton-nlp/SWE-bench_Verified)")
parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
parser.add_argument("--max_steps", type=int, default=40, help="Maximum agent steps per instance")
parser.add_argument("--max_error_times", type=int, default=8, help="Maximum consecutive errors before stopping")
parser.add_argument("--instance_id", type=str, default=None, help="Run a single specific instance (optional)")
parser.add_argument("--repo_grouped", type=str, required=True, help="Root directory for grouped repositories")
parser.add_argument("--rpg_dir", type=str, required=True, help="Directory containing RPG JSON files")
parser.add_argument("--save_dir", type=str, required=True, help="Root directory for saving results")
parser.add_argument("--log_dir", type=str, required=True, help="Root directory for agent logs")
parser.add_argument("--persist_dir", type=str, default=None, help="Directory to persist/load BM25 retrievers (speeds up repeated runs)")
parser.add_argument("--skip_instances", type=str, nargs="*", default=[], help="Instance IDs to skip")

args = parser.parse_args()

NUM_PROCESSES = args.num_processes
RUN_ID = args.run_id
BENCHMARK = args.benchmark
DATASET_SPLIT = args.dataset_split
MAX_STEPS = args.max_steps
MAX_ERROR_TIMES = args.max_error_times

# Build LLM config: --llm-config file takes priority, --model overrides model name
if args.llm_config:
    LLM_CONFIG = LLMConfig.from_source(args.llm_config)
else:
    LLM_CONFIG = LLMConfig(model="o3-mini", max_tokens=16134, top_p=1.0)

if args.model:
    LLM_CONFIG.model = args.model

MODEL_NAME = LLM_CONFIG.model

REPO_GROUPED = args.repo_grouped
RPG_DIR = args.rpg_dir
SKIP_INSTANCES = args.skip_instances

# Derive a short benchmark tag for file naming (e.g., "SWE-bench_Verified" -> "swe-bench_verified")
BENCHMARK_TAG = BENCHMARK.split("/")[-1].lower()

# Derived paths (include run_id and model name)
SAVE_PATH = os.path.join(args.save_dir, RUN_ID, f"{BENCHMARK_TAG}_{MODEL_NAME}.jsonl")
LOG_DIR = os.path.join(args.log_dir, RUN_ID, f"agent_logs_{MODEL_NAME}")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'w') as f:
        pass

REGISTER_TOOLS = [
    SearchCodeSnippets,
    SearchCodeByFeatures,
    # SearchNode,
    FetchNode,
    ExploreRPG,
    Terminate
]


# Create logger per instance
def create_instance_logger(instance_id: str):
    log_file = os.path.join(LOG_DIR, f"{instance_id}.log")
    logger_name = f"agent_{instance_id}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# Process a single instance
def process_bug(bug: Dict):
    instance_id = bug['instance_id']

    # Skip certain instances if configured
    if instance_id in SKIP_INSTANCES:
        return

    repo_name = bug["repo"].split("/")[-1]
    problem = bug["problem_statement"]

    logger = create_instance_logger(instance_id)
    logger.info(f"Processing instance: {instance_id} ({repo_name})")

    # Find repo directory
    repo_dir = ""
    for repo_dir_name in os.listdir(os.path.join(REPO_GROUPED, repo_name)):
        if instance_id in repo_dir_name:
            repo_dir = os.path.join(REPO_GROUPED, repo_name, repo_dir_name, repo_name)
            break

    if not repo_dir:
        logger.error(f"No repo dir found for {instance_id}")
        return

    # Load RPG (new format: single JSON containing rpg + dep_graph)
    rpg_files = os.listdir(RPG_DIR)
    rpg_file_path = next(
        (os.path.join(RPG_DIR, f) for f in rpg_files if instance_id in f),
        None
    )
    if not rpg_file_path:
        logger.error(f"No RPG file found for {instance_id}")
        return

    try:
        rpg = RPG.load_json(rpg_file_path)
    except Exception as e:
        logger.error(f"Failed to load RPG for {instance_id}: {e}")
        return

    # Get dep_graph from RPG
    dep_graph = rpg.dep_graph.G if rpg.dep_graph else None

    agent = RPGAgent(
        llm_cfg=LLM_CONFIG,
        instance_id=instance_id,
        repo_name=repo_name,
        repo_dir=repo_dir,
        dep_graph=dep_graph,
        repo_rpg=rpg,
        max_steps=MAX_STEPS,
        context_window=MAX_STEPS * 2,
        register_tools=REGISTER_TOOLS,
        persist_dir=args.persist_dir,
        task=problem,
        logger=logger
    )

    try:
        final_results = agent.run(max_error_times=MAX_ERROR_TIMES)
    except Exception as e:
        logger.exception(f"Agent run failed for {instance_id}: {e}")
        return

    with jsonlines.open(SAVE_PATH, mode='a') as writer:
        writer.write({
            "instance_id": instance_id,
            "repo": repo_name,
            "result": final_results
        })

    logger.info(f"Finished {instance_id}")


# --- Main Script --- #
def main():
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    logging.basicConfig(level=logging.CRITICAL)
    print(f"[RPGAgent] Starting {BENCHMARK} ({DATASET_SPLIT}) with model {MODEL_NAME}, run_id={RUN_ID}...")

    # Load dataset from HuggingFace
    dataset = load_dataset(BENCHMARK)
    data_split = dataset[DATASET_SPLIT]

    rpg_files = os.listdir(RPG_DIR)

    # Filter instances that have RPG files ready
    ready = [d for d in data_split if any(d["instance_id"] in f for f in rpg_files)]

    # If a specific instance_id is requested, filter to just that one
    if args.instance_id:
        ready = [d for d in ready if d["instance_id"] == args.instance_id]
        if not ready:
            print(f"[RPGAgent] Instance {args.instance_id} not found or RPG not ready.")
            return

    # Load already processed instances
    already_processed = []
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                already_processed.append(json5.loads(line))

    processed_ids = [data["instance_id"] for data in already_processed]
    filtered_ready = [data for data in ready if data["instance_id"] not in processed_ids]

    print(f"[RPGAgent] Total ready: {len(ready)}, Already processed: {len(processed_ids)}, To process: {len(filtered_ready)}")

    if not filtered_ready:
        print("[RPGAgent] Nothing to process. Done.")
        return

    # Set up multiprocessing pool
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        pool.map(process_bug, filtered_ready)


if __name__ == '__main__':
    main()
