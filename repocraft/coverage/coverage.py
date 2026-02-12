import os
import re
import torch
import json, json5
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import Optional, List, Dict
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from evaluation.coverage.sys_prompt import REORGANIZE_PROMPT, REORGANIZE_PROMPT_NO_THINKING
from zerorepo.utils.tree import extract_leaf_nodes
from zerorepo.utils.api import parse_thinking_output
from zerorepo.rpg_gen.base.llm_client import LLMClient, LLMConfig, Memory
from zerorepo.rpg_gen.base.llm_client.message import SystemMessage, UserMessage, AssistantMessage

# 默认模型配置，可以通过环境变量覆盖
USE_MODEL = os.environ.get("USE_MODEL", "gpt-4.1-20250414")

# 别名映射：将输入文件中的 repository_name 映射到 ground truth 文件名
REPO_NAME_ALIAS = {
    "requests": "HttpEasy",
    "scikit-learn": "MLKit-Py",
    "django": "PyWebEngine",
    "statsmodels": "StatModeler",
    "sympy": "SymbolicMath",
    "pandas": "TableKit",
}

_current_repo_handler: Optional[logging.FileHandler] = None

def normalize_string(s):
    """Normalize string by lowercasing and removing non-alphanumerics."""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def attach_repo_log_handler(repo_name: str,
                            log_root: str = "repo-logs",
                            level: int = logging.INFO) -> None:
    """
    将 logs/<repo_name>.log 附着到根 logger，同时也支持输出到终端。
    若之前已有别的 repo handler，会先移除之。
    """
    global _current_repo_handler

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 移除旧的文件 handler
    if _current_repo_handler:
        root_logger.removeHandler(_current_repo_handler)
        try:
            _current_repo_handler.close()
        finally:
            _current_repo_handler = None

    # 创建新的文件 handler
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, f"{repo_name}.log")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    file_handler.setLevel(level)

    root_logger.addHandler(file_handler)
    _current_repo_handler = file_handler

    # 添加终端输出 handler（如果尚未添加）
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    root_logger.propagate = False
    root_logger.info(f"Switched to repo '{repo_name}', log file: {log_path}")
    
    
def detach_repo_log_handler() -> None:
    """
    在处理完一个 repo 后调用，安全移除并关闭当前的 FileHandler。
    防止文件句柄泄漏或之后的日志误写。
    """
    global _current_repo_handler
    if _current_repo_handler:
        root_logger = logging.getLogger()
        root_logger.removeHandler(_current_repo_handler)
        try:
            _current_repo_handler.close()
        finally:
            root_logger.info("Detached repo log handler.")
            _current_repo_handler = None
    

class SubtreeCoverageEvaluator:


    def __init__(
        self,
        model_id: str="Qwen/Qwen3-Embedding-0.6B",
        outlier_tag: str="outlier_features",
        context_window: int=5,
        llm_model: str=None
    ):
        self.outlier_tag = outlier_tag
        self.context_window = context_window
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

        # 初始化 LLM Client
        llm_model = llm_model or USE_MODEL
        self.llm_config = LLMConfig(model=llm_model, max_tokens=16134)
        self.llm_client = LLMClient(config=self.llm_config)
    
    def embedding(self, texts: List=[]):
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # 使用 [CLS] 位置向量
            embeddings = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
        
        return embeddings
    
        
    def smart_cluster_with_outlier_filter(self, X, centers=None, outlier_method="isoforest", re_cluster=True):
        """
        聚类 + 自适应离群点识别 + 过滤

        Args:
            X (np.ndarray): 原始嵌入数据
            centers (np.ndarray or None): 初始聚类中心；如果为 None，则自动使用 KMeans++
            outlier_method (str): 'lof' | 'isoforest' | 'ocsvm'
            re_cluster (bool): 是否在过滤后重新聚类

        Returns:
            dict: 包含干净数据、标签、离群掩码等
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # Step 1: 初始聚类
        if centers is not None:
            centers = np.array(centers)
            k = len(centers)
            if centers.shape[1] != n_features:
                raise ValueError("Center dimension does not match input feature dimension.")

            if n_samples < k:
                print(f"[Info] n_samples={n_samples} < n_clusters={k}, randomly selecting {n_samples} centers for clustering.")
                selected_indices = np.random.choice(k, size=n_samples, replace=False)
                centers = centers[selected_indices]
                k = n_samples

            kmeans = KMeans(n_clusters=k, init=centers, n_init=1, random_state=42)
            labels = kmeans.fit_predict(X)
        else:
            k = min(5, n_samples)
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            labels = kmeans.fit_predict(X)

        # Step 2: 离群点检测
        try:
            if outlier_method == "lof":
                if n_samples <= 20:
                    outlier_mask = np.array([False] * n_samples)
                else:
                    model = LocalOutlierFactor(n_neighbors=20, contamination="auto")
                    outlier_mask = model.fit_predict(X) == -1

            elif outlier_method == "isoforest":
                model = IsolationForest(contamination="auto", random_state=42)
                outlier_mask = model.fit_predict(X) == -1

            elif outlier_method == "ocsvm":
                model = OneClassSVM(gamma="auto")
                outlier_mask = model.fit_predict(X) == -1
            else:
                raise ValueError(f"Unsupported outlier method: {outlier_method}")
        except Exception as e:
            print(f"[Warning] Outlier detection failed: {e}")
            outlier_mask = np.array([False] * n_samples)

        # Step 3: 剔除离群点
        X_clean = X[~outlier_mask]

        # Step 4: 重新聚类（可选）
        if re_cluster and len(X_clean) > 0:
            if centers is not None:
                if len(X_clean) < k:
                    print(f"[Info] Cleaned samples={len(X_clean)} < centers={k}, selecting {len(X_clean)} centers.")
                    selected_indices = np.random.choice(k, size=len(X_clean), replace=False)
                    centers = centers[selected_indices]
                    k = len(X_clean)

                kmeans_clean = KMeans(n_clusters=k, init=centers, n_init=1, random_state=42)
            else:
                k = min(5, len(X_clean))
                kmeans_clean = KMeans(n_clusters=k, init='k-means++', random_state=42)

            clean_labels = kmeans_clean.fit_predict(X_clean)
        else:
            clean_labels = labels[~outlier_mask] if len(X_clean) == len(labels) else [-1] * len(X_clean)

        return {
            "X_clean": X_clean,
            "labels_clean": clean_labels,
            "outlier_mask": outlier_mask,
            "kmeans_model": kmeans
        }
    
    def evaluate(self, data, pre_trees, gt_tree):
        
        repo_paths = []
        
        for pre_tree in pre_trees:
            pre_name = pre_tree.get("name", "")    
            pre_tree_dict = pre_tree.get("refactored_subtree", "")
            
            # pre_paths = get_all_leaf_paths(pre_tree_dict)
            pre_paths = extract_leaf_nodes(pre_tree_dict)
            
            #if pre_name:
            #    pre_paths = list(map(lambda x: pre_name + "/" + x, pre_paths))
            repo_paths.extend(pre_paths)
        
        # gt_paths = get_all_leaf_paths(gt_tree)
        gt_paths = extract_leaf_nodes(gt_tree)
        if not repo_paths or not gt_paths:
            return {
                "coverage_ratio": 0.0,
                "new_feature_ratio": 0.0,
                "num_gt_paths": len(gt_paths),
                "num_repo_paths": len(repo_paths),
                "retained_repo_paths": [],
                "gt_path_to_cluster": {},
                "note": "No paths to compare."
            }
        
        gt_embeddings = self.embedding(texts=gt_paths)
        repo_embeddings = self.embedding(texts=repo_paths)
        
        result = self.smart_cluster_with_outlier_filter(
            X=repo_embeddings,
            centers=gt_embeddings, 
            outlier_method="lof",
            re_cluster=True
        )
        
        outlier_mask = result["outlier_mask"]
        
        # 非离群点
        retained_repo_paths = [p for p, keep in zip(repo_paths, ~outlier_mask) if keep]
        # 离群点
        outlier_repo_paths = [p for p, flag in zip(repo_paths, outlier_mask) if flag]
        retained_repo_paths = [p for p, keep in zip(repo_paths, ~outlier_mask) if keep]
        
        clusters = defaultdict(list)
        labels = result["kmeans_model"].labels_

        for repo_path, label, is_outlier in zip(repo_paths, labels, outlier_mask):
            if not is_outlier and 0 <= label < len(gt_paths):
                clusters[gt_paths[label]].append(repo_path)
        
        
        clusters = dict(clusters)
        
        for gt_node in gt_paths:
            if not gt_node in clusters.keys():
                clusters[gt_node] = []
        
        clusters[self.outlier_tag] = outlier_repo_paths
        
        
        batch_size = 5 if USE_MODEL in ["gpt-5", "o4-mini-20250416"] else 10
        
        final_result = self.reorgnize_tree(
            repo_data=data,
            pre_trees=pre_trees,
            gt_tree=gt_tree,
            batch_size=batch_size,
            cluster_result=clusters
        )
        
        coverage_paths = [c_key for c_key in final_result.keys() if c_key != self.outlier_tag and len(final_result[c_key]) > 0]
        
        return {
            "coverage_ratio":  len(coverage_paths) /len(gt_paths),
            "new_feature_ratio": len(outlier_repo_paths) / len(repo_paths),
            "num_gt_paths": len(gt_paths),
            "num_repo_paths": len(repo_paths),
            "retained_repo_paths": retained_repo_paths,
            "gt_path_to_cluster": final_result
        }


    def moving_keys_in_batch(
        self,
        cur_keys,
        cluster_result: Dict,
        repo_data,
        pre_trees,
        gt_tree,
        max_iterations: int = 5
    ):
        import json, json5
        from copy import deepcopy

        def build_repo_info(data):
            return "".join([f"{k.replace('_', ' ').capitalize()}: {v}\n" for k, v in data.items()])

        def build_feature_tree(trees):
            tree_str = ""
            for t in trees:
                tree_str += f"Subtree Name: {t.get('name', '')}\n"
                tree_str += f"Subtree: {json.dumps(t.get('refactored_subtree', {}))}\n\n"
            return tree_str

        def build_user_prompt(current_groups, other_keys):
            return (
                "[Iteration Begin]\n"
                "Here is the current grouping of items. Please help reorganize them if needed.\n"
                "You can either:\n"
                "1. Adjust which group an item belongs to within the current structure, or\n"
                "2. Move an item out of these groups and assign it to one of the other available group names listed below.\n\n"
                f"Current Groups:\n{json.dumps(current_groups, indent=1)}\n\n"
                f"Other Available Group Names:\n{json.dumps(other_keys, indent=1)}"
            )

        # 1. Build system prompt
        if USE_MODEL not in ["gpt-5", "o4-mini-20250416"]:
            sys_prompt = REORGANIZE_PROMPT.format(
                outlier_tag=self.outlier_tag,
                real_tree=json.dumps(gt_tree, indent=1),
                generated_tree=build_feature_tree(pre_trees),
                repo_info=build_repo_info(repo_data)
            )
        else:
            sys_prompt = REORGANIZE_PROMPT_NO_THINKING.format(
                outlier_tag=self.outlier_tag,
                real_tree=json.dumps(gt_tree, indent=1),
                generated_tree=build_feature_tree(pre_trees),
                repo_info=build_repo_info(repo_data)
            )
        logging.info(f"[System Prompt] =====\n{sys_prompt}")

        if USE_MODEL in ["gpt-5", "o4-mini-20250416", "gpt-5.1-chat-20251113"]:
            sys_prompt += "\nPlease avoid excessive thinking; just perform reasonable reasoning and provide the action directly."

        # 2. 使用 Memory 管理对话历史
        memory = Memory(context_window=self.context_window)
        memory.add_message(SystemMessage(content=sys_prompt))

        filter_cluster = deepcopy({k: cluster_result[k] for k in cur_keys})
        other_keys = [k for k in cluster_result if k not in cur_keys]
        remaining_results = deepcopy({k: cluster_result[k] for k in other_keys})

        # 添加初始用户消息
        memory.add_message(UserMessage(content=build_user_prompt(filter_cluster, other_keys)))
        final_results = deepcopy(cluster_result)

        for i in range(max_iterations):
            try:
                # 使用 LLMClient 生成响应
                response = self.llm_client.generate(memory)
                if not response:
                    logging.warning(f"[Iteration {i + 1}] Empty response from LLM")
                    continue

                logging.info(f"[Iteration {i + 1} Response]: {response}")

                # 添加助手回复到 Memory
                memory.add_message(AssistantMessage(content=response))

                # Parse and clean response
                parsed = parse_thinking_output(response)
                cleaned = parsed.replace("```json", "").replace("```", "").replace("\n", "").replace("\t", "")
                actions = json5.loads(cleaned)
                actions = actions if isinstance(actions, list) else [actions]

                is_terminate = False

                # Apply actions
                for action in actions:
                    name = action.get("action_name", "").lower()

                    if name == "terminate" and i >= 1:
                        logging.info("Received terminate signal at iteration %d", i + 1)
                        is_terminate = True

                    if name == "move":
                        src = action.get("source")
                        dst = action.get("destination")
                        items = action.get("target", [])
                        items = items if isinstance(items, list) else [items]

                        if src not in filter_cluster:
                            logging.warning("Invalid source group: %s", src)
                            continue

                        # Create dest if moving to remaining keys
                        if dst not in filter_cluster:
                            if dst in other_keys:
                                filter_cluster[dst] = []
                            else:
                                logging.warning("Invalid destination group: %s", dst)
                                continue

                        for item in items:
                            if item in filter_cluster[src]:
                                filter_cluster[src].remove(item)
                                filter_cluster[dst].append(item)
                                logging.info("Moved '%s' from '%s' to '%s'", item, src, dst)

                # Append next user message with updated state
                other_keys = [k for k in cluster_result if k not in cur_keys]

                input_cluster = {key: filter_cluster[key] for key in filter_cluster.keys() if key in cur_keys}
                next_user_content = (
                    f"[Iteration {i + 1}] Review the following partial clustering structure:\n\n"
                    f"- Current Mapping (group names and their items):\n{json.dumps(input_cluster, indent=1)}\n\n"
                    f"- Other Available Group Names (you may move items into these as well):\n{json.dumps(other_keys, indent=1)}\n\n"
                    f"- **Important**:\n"
                    f"- When using the `move` action, the `source` must be a key from the **Current Mapping**.\n"
                    f"- The `destination` must be chosen from either the **Current Mapping** keys or the **Other Available Group Names**.\n"
                    f"- Do **not invent** or modify group names.\n"
                )
                memory.add_message(UserMessage(content=next_user_content))

                logging.info(f"[Iteration {i + 2} Input]: {json.dumps(input_cluster, indent=1)}")

                if is_terminate:
                    break
            except Exception as e:
                logging.exception(f"Error during iteration {i + 1}: {e}")
                continue
        
        # 3. Final merge: update cluster_result based on filter_cluster
        final_results = deepcopy(cluster_result)
        for key in filter_cluster:
            if key in cur_keys:
                final_results[key] = filter_cluster[key]
            else:
                final_results[key].extend(filter_cluster[key])

        return final_results


    def reorgnize_tree(
        self,
        repo_data,
        pre_trees,
        gt_tree,
        cluster_result,
        batch_size: int=5,
        max_iterations: int=10
    ):
        from copy import deepcopy
        from tqdm import tqdm
        
        cluster_keys = list(cluster_result.keys())
        cluster_keys_wo_outlier = [key for key in cluster_keys if key != self.outlier_tag]
        
    
        final_result = deepcopy(cluster_result)
        
        total_batches = (len(cluster_keys_wo_outlier) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches + 1, desc="Reorganizing Tree", unit="batch")

        for i in range(0, len(cluster_keys_wo_outlier), batch_size):
            cur_keys = cluster_keys_wo_outlier[i:i + batch_size]
            logging.info(f"[Batch {i // batch_size + 1}] Processing keys: {cur_keys}")

            if all(len(final_result.get(key, [])) == 0 for key in cur_keys):
                logging.info(f"[Batch {i // batch_size + 1}] All keys are empty, skipping.")
                pbar.update(1)
                continue
                        
            updated_result = self.moving_keys_in_batch(
                cur_keys=cur_keys,
                cluster_result=final_result,
                repo_data=repo_data,
                pre_trees=pre_trees,
                gt_tree=gt_tree,
                max_iterations=max_iterations
            )

            final_result = updated_result
            pbar.update(1)

        updated_result = self.moving_keys_in_batch(
            cur_keys=[self.outlier_tag],
            cluster_result=final_result,
            repo_data=repo_data,
            pre_trees=pre_trees,
            gt_tree=gt_tree,
            max_iterations=max_iterations
        )
        
        final_result = updated_result
        
        final_result = self.post_process(final_result)
           
        return final_result   
        
    def post_process(self, cluster_results):
        # Step 1: Build normalized group name → real group name map
        norm_group_map = {normalize_string(group): group for group in cluster_results}

        # Step 2: Identify entries that match a group name (normalized), but are not inside that group
        moves = []
        for group_name, entries in cluster_results.items():
            for entry in entries:
                norm_entry = normalize_string(entry)
                # Only move if entry matches a group name but is in the wrong group
                for norm_group in norm_group_map:
                    if norm_group in norm_entry or norm_entry in norm_group:
                        correct_group = norm_group_map.get(norm_entry)
                        if correct_group and correct_group != group_name:
                            moves.append((entry, group_name, correct_group))

        # Step 3: Apply the moves
        for entry, source_group, destination_group in moves:
            if entry in cluster_results[source_group]:
                cluster_results[source_group].remove(entry)
            if destination_group not in cluster_results:
                cluster_results[destination_group] = []
            cluster_results[destination_group].append(entry)

        return cluster_results
        
    def run(self, input_file: str, gt_dir: str, output_file: str):
        """
        Run the subtree coverage evaluation on a single input JSON file.

        Args:
            input_file (str): Path to the input JSON file containing feature data.
            gt_dir (str): Directory containing ground truth JSON files.
            output_file (str): Path to save the evaluation result JSON file.
        """

        assert os.path.exists(input_file), f"Input file {input_file} does not exist."
        assert os.path.exists(gt_dir), f"Ground truth directory {gt_dir} does not exist."

        # 确保输出文件的目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Begin to Process {input_file}")

        with open(input_file, 'r') as f:
            data = json.load(f)

        repo_name = data.get("repository_name", os.path.basename(input_file).replace(".json", ""))

        # 使用别名映射查找 ground truth 文件
        gt_name = REPO_NAME_ALIAS.get(repo_name, repo_name)
        if gt_name != repo_name:
            logging.info(f"Using alias mapping: {repo_name} -> {gt_name}")

        repo_keys = ["repository_name", "repository_purpose", "category", "scope"]
        repo_data = {k: data.get(k, "N/A") for k in repo_keys}

        gt_file_path = os.path.join(gt_dir, f"{gt_name}.json")
        if not os.path.exists(gt_file_path):
            logging.warning(f"Ground truth file for {gt_name} (repo: {repo_name}) not found in {gt_dir}. Skipping evaluation.")
            return None

        with open(gt_file_path, 'r') as f:
            gt_tree = json.load(f)

        attach_repo_log_handler(repo_name)

        try:
            pre_trees = data.get("Component", [])

            eval_result = self.evaluate(repo_data, pre_trees, gt_tree)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, indent=1, ensure_ascii=False)

            logging.info(f"Evaluation completed for {repo_name}")
            logging.info(f"Results written to: {output_file}")
            logging.info(f"Coverage: {eval_result.get('coverage_ratio', 0):.2%}")
            logging.info(f"New Feature Ratio: {eval_result.get('new_feature_ratio', 0):.2%}")

            return eval_result
        except Exception as e:
            logging.error(f"Evaluation failed for {repo_name}: {e}")
            return None
        finally:
            detach_repo_log_handler()
    
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subtree Coverage Evaluation")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input JSON file containing feature data")
    parser.add_argument("--gt-dir", type=str, default="./gt_repo_tree",
                        help="Directory containing ground truth JSON files")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the evaluation result JSON file")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model for reorganization (default: USE_MODEL env or gpt-4o-20241120)")

    args = parser.parse_args()

    evaluator = SubtreeCoverageEvaluator(llm_model=args.llm_model)
    evaluator.run(args.input_file, args.gt_dir, args.output_file)

