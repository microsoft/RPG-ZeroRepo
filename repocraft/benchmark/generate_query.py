import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
from copy import deepcopy
from typing import Dict, List, Optional, Union

from zerorepo.rpg_gen.base.llm_client import (
    LLMClient, LLMConfig, Memory,
    SystemMessage, UserMessage, AssistantMessage
)
from zerorepo.rpg_gen.base.unit import CodeSnippetBuilder, CodeUnit, ParsedFile
from zerorepo.utils.api import parse_thinking_output


SYS_PROMPT = """
You are a retrieval query generation assistant.
Given a structured test suite, your task is to generate a **concise natural language query** and an **abstract description** of the algorithm being tested. These queries and descriptions will later be used to retrieve relevant code from different libraries that implement similar functionalities.

For each test group, you will be provided:
- **category**: The category under which the test belongs (e.g., "metrics", "clustering").
- **file**: The file name containing the test (e.g., "test_distance").
- **structure type**: The structure being tested, either a **class** or a **group of functions**.
- **module**: The name of the module representing the algorithm or functionality being tested (e.g., "pairwise_distance_chunked").
- **functions**: A list of test function names for the module.

### Your goal:
For each test module, generate **two outputs**:
1. **Algorithm Description**:
   - **Provide an abstract, high-level description** of the algorithm being tested. Focus on what the algorithm is designed to do (e.g., "calculating pairwise distances between data points").
   - **Avoid specific implementation details** or names of particular functions or libraries.
   - The description should capture the **core functionality** and **purpose** of the algorithm.
2. **Task Query**:
   - **Summarize the test's purpose** in a **natural language query** that reflects the high-level goal of the test.
   - Use the structure: **"You are testing..."**, to simulate what this algorithm is meant to do in a real-world scenario.
   - Be **clear, concise, and focused** on the core functionality being tested. For example: "Tests for the efficient computation of pairwise distances using a chunked approach."
   - Avoid focusing on individual test cases, edge cases, or specific scenarios unless they are necessary to convey the test's purpose.

### Example Input:
{{
  "category": "metrics",
  "file": "test_pairwise",
  "structure": "functions",
  "module": "pairwise_distances_chunked",
  "functions": [
    "test_pairwise_distances_chunked_reduce",
    "test_pairwise_distances_chunked_reduce_none",
    "test_pairwise_distances_chunked_reduce_invalid",
    "test_pairwise_distances_chunked_diagonal"
  ]
}}

### Example Output:
<think>
The algorithm seems to focus on computing pairwise distances efficiently, particularly with chunking strategies for handling large datasets.
</think>
<solution>
{{
  "alg_description": "An algorithm for calculating pairwise distances between data points using efficient chunking strategies to handle large datasets.",
  "task_query": "You are testing an algorithm designed to compute pairwise distances between data points efficiently. The algorithm should handle large datasets using a chunking strategy, ensuring that the computation is done in manageable parts without compromising accuracy. Your goal is to validate that this chunked processing strategy works correctly across different input configurations, ensuring efficient and accurate distance computation."
}}
</solution>

## Notes:
1. Keep the algorithm description abstract and high-level, focusing on the functionality rather than specific implementation.
2. The task query should clearly reflect the test's purpose, i.e., validating an algorithm's behavior, not focusing on test function names or specific scenarios.
3. The query should be brief but complete enough to describe the general intent of the test.
4. Avoid referencing specific libraries, as the goal is to generalize the tests to work with any similar algorithm across different libraries.
"""


# Global LLM client for multiprocessing
_llm_client: Optional[LLMClient] = None


def init_llm_client(llm_cfg: Optional[Union[str, Dict, LLMConfig]] = None):
    """初始化全局 LLM 客户端"""
    global _llm_client
    _llm_client = LLMClient(config=llm_cfg)


def generate_query(task_config: dict) -> dict:
    """使用 LLMClient 生成查询"""
    global _llm_client

    if _llm_client is None:
        _llm_client = LLMClient()

    query_code = deepcopy(task_config).pop("query_code", "")

    env_prompt = (
        f"Below is the configuration for the task you're working on, followed by the associated test code.\n"
        f"Note that both the algorithm description and task query should focus on the core functionality of the algorithm, not edge cases.\n"
        f"{json.dumps(task_config, indent=4)}\n"
        f"### Their Gold Test Code:\n"
        f"{query_code}\n"
    )

    # 使用 Memory 管理对话
    memory = Memory(context_window=5)
    memory.add_message(SystemMessage(content=SYS_PROMPT))
    memory.add_message(UserMessage(content=env_prompt))

    final_result = {}
    for i in range(3):
        try:
            response = _llm_client.generate(memory=memory)
            if not response:
                continue

            print(response)
            parsed_rp = parse_thinking_output(output=response)
            parsed_rp = parsed_rp.replace("```json", "").replace("```", "").replace("\n", "").replace("\t", "")
            rp_json = json.loads(parsed_rp)
            final_result = rp_json

            if rp_json.get("alg_description", "") and rp_json.get("task_query", ""):
                break
        except Exception as e:
            import time
            print(f"Error at iteration {i + 1}: {e}")
            time.sleep(10)
            continue

    final_result2 = {
        **task_config,
        **final_result
    }

    return final_result2


def load_json(file_path: str) -> dict:
    """读取 json 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)


def process_test_data(sampled_test: dict, result_test: dict) -> List[dict]:
    """
    遍历采样的测试数据并生成查询。
    """
    queries = []

    for category, test_files in sampled_test.items():
        for test_file, modules in test_files.items():

            abs_file_path = next(
                (file_path for file_path in result_test.keys() if category in file_path and test_file in file_path),
                None
            )
            if not abs_file_path:
                continue
            
            with open(abs_file_path, 'r') as f:
                file_code = f.read()

            file_units = ParsedFile(code=file_code, file_path=abs_file_path).units

            code_snippt = CodeSnippetBuilder(
                file_code_map={abs_file_path: file_code},
                parsed_files={abs_file_path: ParsedFile(code=file_code, file_path=abs_file_path)}
            )

            for module, caps in modules.items():
                for cap, funcs in caps.items():
                    valid_units = []

                    if module == "functions":
                        valid_units = [unit for unit in file_units if unit.unit_type == "function" and unit.name in funcs]
                    elif "class" in module:
                        valid_units = [unit for unit in file_units if unit.unit_type == "method" and unit.name in funcs and unit.parent in module]

                    if not valid_units:
                        continue

                    valid_code = code_snippt.build(merged=valid_units)

                    query = {
                        "category": category,
                        "file": abs_file_path,
                        "module": module,
                        "cap": cap,
                        "functions": funcs,
                        "query_code": valid_code
                    }

                    queries.append(query)

    return queries


def batch_generate_queries(
    sampled_test_json: str,
    result_test_json: str,
    output_json: str,
    llm_cfg: Optional[Union[str, Dict, LLMConfig]] = None,
    max_workers: int = 6
):

    init_llm_client(llm_cfg)

    result_test = load_json(result_test_json)
    sampled_test = load_json(sampled_test_json)

    queries = process_test_data(sampled_test, result_test)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(generate_query, queries), total=len(queries), desc="Processing Queries"))

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Batch generate queries for a repo.")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository")
    parser.add_argument("--llm_config", type=str, default=None, help="Path to LLM config file (JSON/YAML)")
    parser.add_argument("--max_workers", type=int, default=6, help="Number of parallel workers")

    args = parser.parse_args()
    repo_name = args.repo_name

    llm_cfg = None
    if args.llm_config:
        llm_cfg = LLMConfig.from_source(args.llm_config)

    batch_generate_queries(
        f"./all_results/sampled_test/sample_{repo_name}.json",
        f"./all_results/result_tests/{repo_name}.json",
        output_json=f"./all_results/task_results2/{repo_name}.json",
        llm_cfg=llm_cfg,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
