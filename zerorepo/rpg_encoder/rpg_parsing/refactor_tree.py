'''
This module defines the RefactorTree class, which uses a language model to iteratively refactor a codebase based on its parsed structure and features.
'''

import json
import os
import logging
import uuid
from copy import deepcopy
from typing import List, Optional, Union, Dict, Tuple
from tqdm import tqdm
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.unit import ParsedFile
from zerorepo.rpg_gen.base.rpg import RPG, Node, NodeMetaData, NodeType

from zerorepo.utils.tree import (
    apply_changes,
    convert_leaves_to_list,
    iterative_by_folder,
    format_parsed_tree,
    transfer_parsed_tree,
    get_rpg_info
)
from zerorepo.utils.api import (
    parse_solution_output,
    calculate_tokens
)
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient,
    LLMConfig,
    Memory,
    UserMessage,
    AssistantMessage,
    SystemMessage
)
from zerorepo.rpg_gen.base.unit import CodeSnippetBuilder
from zerorepo.utils.logs import setup_logger
from .prompts import REFACTOR_TREE, REFACTOR_MODIFIED, FUNCTIONAL_AREA

class RefactorTree:
    def __init__(
        self,
        repo_dir: str,
        repo_info: str,
        repo_skeleton: RepoSkeleton,
        repo_name: str,
        logger: Optional[logging.Logger] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs
    ):
        self.repo_name = repo_name
        self.repo_dir = repo_dir
        self.repo_info = repo_info

        self.rpg = RPG(repo_name=self.repo_name)

        self.repo_skeleton = repo_skeleton

        if not logger:
            self.logger = setup_logger(logging.getLogger(f"RefactorTree[{repo_name}]"))
        else:
            self.logger = logger

        self.llm_client = LLMClient(llm_config)
    
    def _uuid8(self) -> str:
        """短uuid（8位十六进制），用来拼接节点 id，减少碰撞."""
        return uuid.uuid4().hex[:8]

    def step(self, memory: Memory):
        action = {}

        try:
            response = self.llm_client.generate(memory)
            parsed_output = parse_solution_output(response)
            parsed_output = (
                parsed_output.replace("```json", "")
                .replace("```", "")
                .replace("\n", "")
                .replace("\t", "")
            )
            action = json.loads(parsed_output)
            self.logger.info(f"GPT Response: {response}")
        except Exception as e:
            self.logger.error(f"Error calling GPT: {e}")
            return {}, ""

        return action, response

    def process_action(
        self,
        action: Dict,
        processed_features: List,
        functional_areas: List,
        trans_tree: Dict,
        cur_refactored_tree: List,
    ):
        """
        Execute a single action and insert the resulting feature into `refactored_tree`.
        Returns:
            env_prompt: environment feedback prompt
            updated_subtree: the subtree info added/modified in this round (incremental)
            new_paths: full paths newly inserted into the structure in this round (for syncing back to the RPG)
        """

        fa_map = {fa.strip().lower(): fa.strip() for fa in functional_areas}

        def normalize_fa(name: str) -> Optional[str]:
            return fa_map.get(name.strip().lower())

        refactored_paths = list(action.keys())
        trans_features = list(trans_tree.keys())

        valid_paths = []
        invalid_paths = []
        normalized_valid_paths = {}

        for path in refactored_paths:
            parts = path.split("/")
            if len(parts) != 3:
                invalid_paths.append(path)
                continue

            raw_fa = parts[0]
            real_fa = normalize_fa(raw_fa)

            if real_fa is None:
                invalid_paths.append(path)
            else:
                valid_paths.append(path)
                normalized_valid_paths[path] = real_fa

        import difflib

        invalid_details = []
        for path in invalid_paths:
            parts = path.split("/")
            if len(parts) != 3:
                invalid_details.append(
                    f"- {path}\n"
                    f"  Reason: Path must contain exactly 3 segments: FunctionalArea/SubCategory/Feature\n"
                )
                continue

            fa = parts[0].strip()
            fa_lower = fa.lower()
            candidates = difflib.get_close_matches(fa_lower, fa_map.keys(), n=1, cutoff=0.6)

            if fa_lower not in fa_map:
                if candidates:
                    suggestion = fa_map[candidates[0]]
                    invalid_details.append(
                        f"- {path}\n"
                        f"  Reason: Functional area \"{fa}\" not recognized\n"
                        f"  Suggestion: Use \"{suggestion}\""
                    )
                else:
                    valid_list = ", ".join(functional_areas)
                    invalid_details.append(
                        f"- {path}\n"
                        f"  Reason: Functional area \"{fa}\" not recognized\n"
                        f"  Valid functional areas: {valid_list}"
                    )
            else:
                invalid_details.append(f"- {path}\n" f"  Reason: Unknown formatting issue")

        invalid_block = "\n".join(invalid_details) if invalid_details else "None"

        updated_subtree = {}
        new_paths = []

        for valid_path in valid_paths:
            real_fa = normalized_valid_paths[valid_path]
            parts = valid_path.split("/")
            parts[0] = real_fa
            valid_path_std = "/".join(parts)

            re_features = action[valid_path]
            re_features = [re_features] if not isinstance(re_features, list) else re_features

            valid_features = [f for f in re_features if f in trans_features]
            insert_feature_paths = []

            func_area = real_fa
            valid_path_tail = "/".join(valid_path_std.split("/")[1:])

            for vf in valid_features:
                if vf in processed_features:
                    continue
                processed_features.append(vf)
                sub_features = trans_tree[vf]

                if sub_features:
                    full_paths = [f"{valid_path_tail}/{vf}/{sub}" for sub in sub_features]
                else:
                    full_paths = [f"{valid_path_tail}/{vf}"]

                insert_feature_paths.extend(full_paths)
                new_paths.extend([f"{func_area}/{p}" for p in full_paths])

            for sub_tree in cur_refactored_tree:
                if sub_tree.get("name", "").strip().lower() == func_area.lower():
                    re_subtree = sub_tree.get("refactored_subtree", {})
                    re_subtree = apply_changes(re_subtree, insert_feature_paths)
                    re_subtree = convert_leaves_to_list(tree=re_subtree)
                    sub_tree["refactored_subtree"] = re_subtree
                    updated_subtree[func_area] = re_subtree

        missing_features = list(set(trans_features) - set(processed_features))

        env_prompt = (
            "Environment feedback after this iteration:\n\n"
            f"Valid paths:\n{json.dumps(valid_paths)}\n\n"
            f"Invalid paths and reasons:\n{invalid_block}\n\n"
            f"Updated functional areas: {[normalized_valid_paths[p] for p in valid_paths]}\n"
            f"Remaining features to organize:\n{json.dumps(missing_features)}\n\n"
            "Please fix the invalid paths using the correct functional area names and format:\n"
            "FunctionalArea/SubCategory/Feature\n"
            "Do NOT modify or repeat any previously assigned features. Ensure each remaining feature is mapped exactly once to a valid path."
        )

        return env_prompt, updated_subtree, new_paths

    # =========================
    # NEW: prompt construction + token estimation (for batch splitting)
    # =========================
    def _build_process_folder_user_prompt(
        self,
        *,
        repo_name: str,
        folder_path: str,
        functional_areas: List,
        format_tree_str: str,
        cur_rpg_info: str,
    ) -> str:
        return (
            f"### Repository Name:\n"
            f"<repo_name>\n{repo_name}\n</repo_name>\n\n"
            f"### Current Folder Path:\n"
            f"<current_folder_path>\n{folder_path}\n</current_folder_path>\n\n"
            f"### Functional Areas:\n"
            f"<functional_areas>\n{json.dumps(functional_areas)}\n</functional_areas>\n\n"
            f"### Parsed Feature Tree for This Folder:\n"
            f"<parsed_folder_tree>\n{format_tree_str}\n</parsed_folder_tree>\n\n"
            f"### Current Refactored Repository Tree (for reference):\n"
            f"<current_refactored_tree>\n{cur_rpg_info}\n</current_refactored_tree>\n\n"
            "Please analyze the above information and propose how to reorganize the top-level feature groups "
            "from <parsed_folder_tree> into appropriate target paths under the given <functional_areas>. "
            "Remember to only operate at the top-level group granularity, and return the result strictly within "
            "<solution> ... </solution> as specified."
        )

    def _estimate_batch_tokens_for_process_folder(
        self,
        *,
        functional_areas: List,
        folder_path: str,
        cur_feature_tree: List[Dict],
        folder_sub_tree: Dict,
    ) -> int:
        format_tree_str = format_parsed_tree(
            input_tree=folder_sub_tree,
            omit_full_leaf_nodes=True,
            max_features=2,
        )

        cur_rpg_info = get_rpg_info(
            rpg_tree=cur_feature_tree,
            omit_leaf_nodes=True,
            sample_size=0,
        )

        user_prompt = self._build_process_folder_user_prompt(
            repo_name=self.repo_name,
            folder_path=folder_path,
            functional_areas=functional_areas,
            format_tree_str=format_tree_str,
            cur_rpg_info=cur_rpg_info,
        )

        return calculate_tokens(user_prompt)

    def process_folder(
        self,
        functional_areas: List,
        folder_path: str,
        cur_feature_tree: Dict,
        dir_file2node: Dict[str, Node],
        area_update: Dict[str, Dict[str, Node]],
        parsed_tree: Dict,
        context_window: int = 20,
        max_iters: int = 20
    ):
        cur_tree = deepcopy(cur_feature_tree)

        feature2node = {f_node.name: f_node for file, f_node in dir_file2node.items()}

        transfered_tree, _ = transfer_parsed_tree(input_tree=parsed_tree)

    
        format_tree_str = format_parsed_tree(
            input_tree=parsed_tree, omit_full_leaf_nodes=True, max_features=2
        )
        cur_rpg_info = get_rpg_info(rpg_tree=cur_tree, omit_leaf_nodes=True, sample_size=0)

        user_prompt = self._build_process_folder_user_prompt(
            repo_name=self.repo_name,
            folder_path=folder_path,
            functional_areas=functional_areas,
            format_tree_str=format_tree_str,
            cur_rpg_info=cur_rpg_info,
        )

        self.logger.info(f"Iteration Begin")

        # Initialize Memory with context window
        agent_memory = Memory(context_window=context_window)
        agent_memory.add_message(SystemMessage(REFACTOR_TREE))
        agent_memory.add_message(UserMessage(user_prompt))

        all_new_paths = []
        processed_features = []
        self.logger.info(f"Initialized REFACTOR agent with {len(transfered_tree)} top-level feature groups.")

        # --- Iterative refinement loop ---
        for it in range(max_iters):
            self.logger.info(f"🌀 Iteration {it + 1}/{max_iters} started")

            try:
                action, response = self.step(agent_memory)
                self.logger.info(f"Model action: {action}")
                self.logger.info(f"Raw response snippet: {str(response)[:300]}...")

                agent_memory.add_message(AssistantMessage(response))

                env_prompt, _, new_paths = self.process_action(
                    action=action,
                    processed_features=processed_features,
                    functional_areas=functional_areas,
                    trans_tree=transfered_tree,
                    cur_refactored_tree=cur_tree,
                )


                for path in new_paths:
                    parts = path.split("/")
                    parts = parts[:-1] if len(parts) == 5 else parts
                    feature_part = parts[-1]
                    new_path = "/".join(parts)
                    area_name = parts[0]

                    file_node = feature2node.get(feature_part, None)
                    if not file_node or area_name not in area_update.keys():
                        continue
                    area_update[area_name][new_path] = file_node

                self.logger.info(f"Env Feedback: {env_prompt}")

                all_new_paths.extend(new_paths)
                agent_memory.add_message(UserMessage(env_prompt))

                self.logger.info(
                    f"Processed features so far: {len(set(processed_features))}/{len(set(transfered_tree.keys()))}"
                )
                if len(set(processed_features)) == len(set(transfered_tree.keys())):
                    self.logger.info("All features processed. Exiting loop.")
                    break

            except Exception as e:
                self.logger.exception(f"Error during iteration {it + 1}: {e}")
                continue

        self.logger.info(f"📦 Folder processing complete: {folder_path}")
        return cur_tree, agent_memory

    def plan_functional_areas(self, parsed_tree: Dict, max_iters: int = 3):
        """
        Plan functional areas based on the parsed repo feature tree.
        Runs multiple refinement rounds and synthesizes the final best result.
        """

        # Step 1: Format summarized tree info
        format_tree_info = format_parsed_tree(input_tree=parsed_tree, omit_full_leaf_nodes=True)

        user_prompt = (
            f"### Repository Name:\n<repo_name>\n{self.repo_name}\n</repo_name>\n\n"
            f"### Repository Overview:\n<repo_info>\n{self.repo_info}\n</repo_info>\n\n"
            f"### Parsed Feature Summary:\n<repo_features>\n{format_tree_info}\n</repo_features>\n\n"
            "Based on the information above, please analyze the repository and determine its main functional areas."
        )

        # Initialize LLM client and memory
        memory = Memory(context_window=10)
        memory.add_message(SystemMessage(FUNCTIONAL_AREA))
        memory.add_message(UserMessage(user_prompt))

        candidate_results = []

        # Step 2: Iterative reasoning rounds
        for i in range(max_iters):
            try:
                response = self.llm_client.generate(memory)
                if i == 0:
                    memory.add_message(AssistantMessage(response))

                parsed_response = parse_solution_output(response)
                parsed_response = (
                    parsed_response.replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                    .replace("\t", "")
                )

                parsed_result = json.loads(parsed_response)
                candidate_results.append(parsed_result)

                # Add GPT’s answer into memory for next refinement
                memory.add_message(AssistantMessage(response))

                followup_prompt = (
                    f"Here is your plan iteration {i+1}. "
                    "If you believe there are missing or redundant functional areas, refine the grouping. "
                    "Return only a corrected JSON."
                )
                memory.add_message(UserMessage(followup_prompt))

            except Exception as e:
                self.logger.warning(f"[WARN] Iteration {i+1} failed: {e}")
                continue

        final_result = {}
        # Step 3: Combine / synthesize all results into a single best-fit plan
        if candidate_results:
            combined_sections = []
            for idx, paths in enumerate(candidate_results, start=1):
                if not paths:
                    continue
                section = "### Round {idx} Functional Areas\n" + "\n".join(f"- {p}" for p in paths)
                combined_sections.append(section)

            combined_text = "\n\n".join(combined_sections)

            # Step 4: Ask GPT to produce a clean, summarized version (optional)
            synthesis_prompt = (
                "You are given multiple candidate groupings of repository functional areas.\n"
                "Here is the merged result:\n"
                f"{combined_text}\n\n"
                "Please refine and produce the most coherent, non-overlapping version of this plan.\n"
            )

            try:
                memory.add_message(UserMessage(synthesis_prompt))
                synthesis_response = self.llm_client.generate(memory)
                synthesis_parsed = parse_solution_output(synthesis_response)
                synthesis_parsed = (
                    synthesis_parsed.replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                    .replace("\t", "")
                )
                final_result = json.loads(synthesis_parsed)
            except Exception as e:
                self.logger.warning(f"[WARN] Synthesis step failed, using merged fallback: {e}")

        else:
            self.logger.warning("No valid results from GPT — returning empty dict.")
            final_result = {}

        return {"candidates": candidate_results, "final_plan": final_result}


    def run(
        self,
        parsed_tree: Dict | str,
        context_window: int = 1,
        max_batch_size: int = 50,          
        max_prompt_tokens: int = 90000,    
        reserve_output_tokens: int = 20000, 
        max_iters: int = 20,
    ) -> Tuple[Dict, List, RPG]:
        """
        parsed_tree: Dict Tree, Like
        {
            "path/to/file": {
                "function name": [
                    "feature_1",
                    "feature_2"
                ]
            }
        }
        """

        # Step 1. Load parsed tree if file path provided
        if isinstance(parsed_tree, str) and os.path.exists(parsed_tree):
            with open(parsed_tree, "r", encoding="utf-8") as f:
                parsed_tree = json.load(f)
            self.logger.info(f"Loaded parsed_tree from file: {parsed_tree}")

        # ===== token budget =====
        token_budget = max_prompt_tokens - reserve_output_tokens
        assert token_budget > 0, "token_budget must be > 0"

        file2code = self.repo_skeleton.get_file_code_map()
        file2unit = {
            file: ParsedFile(code=code, file_path=file).units for file, code in file2code.items()
        }

        file2node = {}

        for file_path, f_features in parsed_tree.items():
            file_feature = f_features.get(
                "_file_summary_", os.path.basename(file_path).replace(".py", "")
            )

            uid = str(uuid.uuid4())
            file_node = Node(
                id=file_feature + "_" + uid[:4],
                name=file_feature,
                meta=NodeMetaData(type_name=NodeType.FILE, path=file_path),
            )

            file2node[file_path] = file_node

            file_units = file2unit.get(file_path, [])

            key2unit = {}
            for file_unit in file_units:
                if file_unit.unit_type != "method":
                    key2unit[f"{file_unit.unit_type} {file_unit.name}"] = file_unit
                else:
                    key2unit[f"method {file_unit.parent}.{file_unit.name}"] = file_unit

            unit_nodes = []
            for unit_name, unit_features in f_features.items():
                if unit_name == "_file_summary_":
                    continue

                # function
                if unit_name.startswith("function"):
                    unit_name_clean = unit_name.replace("function ", "").strip()

                    func_unit = key2unit.get(f"function {unit_name_clean}", None)
                    if not func_unit:
                        continue

                    for feature in unit_features:
                        uid = str(uuid.uuid4())
                        unit_node = Node(
                            id=feature + "_" + uid[4],
                            name=feature,
                            meta=NodeMetaData(
                                type_name=NodeType.FUNCTION, path=f"{file_path}:{unit_name_clean}"
                            ),
                            unit=func_unit.key(),
                        )
                        unit_nodes.append(unit_node)

                # class
                elif unit_name.startswith("class"):
                    unit_name_clean = unit_name.replace("class ", "").strip()

                    # class-level feature list
                    if isinstance(unit_features, List):
                        cls_unit = key2unit.get(f"class {unit_name_clean}", None)
                        if not cls_unit:
                            continue

                        for m_feature in unit_features:
                            uid = str(uuid.uuid4())
                            unit_node = Node(
                                id=m_feature + "_" + uid[4],
                                name=m_feature,
                                meta=NodeMetaData(
                                    type_name=NodeType.CLASS, path=f"{file_path}:{unit_name_clean}"
                                ),
                                unit=cls_unit.key(),
                            )
                            unit_nodes.append(unit_node)

                    # method -> feature list
                    elif isinstance(unit_features, dict):
                        for method_name, m_features in unit_features.items():
                            mtd_unit = key2unit.get(
                                f"method {unit_name_clean}.{method_name}", None
                            )
                            if not mtd_unit:
                                continue

                            for m_feature in m_features:
                                uid = str(uuid.uuid4())
                                unit_node = Node(
                                    id=m_feature + "_" + uid[4],
                                    name=m_feature,
                                    meta=NodeMetaData(
                                        type_name=NodeType.METHOD,
                                        path=f"{file_path}:{unit_name_clean}.{method_name}",
                                    ),
                                    unit=mtd_unit.key(),
                                )
                                unit_nodes.append(unit_node)

            # add node
            self.rpg.add_node(file_node)
            for unit_node in unit_nodes:
                self.rpg.add_node(unit_node)
                self.rpg.add_edge(src=file_node, dst=unit_node)

        # Step 2. Plan functional areas
        self.logger.info("Planning functional areas...")
        functional_result = self.plan_functional_areas(parsed_tree=parsed_tree, max_iters=3)

        functional_areas = functional_result.get("final_plan", [])
        assert functional_areas, "functional_areas must not be empty"

        self.logger.info(f"Planned functional areas: {functional_areas}")

        # Step 3. Prepare folder iteration
        folders_dict = iterative_by_folder(parsed_tree)
        folder_items = list(folders_dict.items())
        total_folders = len(folder_items)
        self.logger.info(f"Total folders to process: {total_folders}")

        cur_feature_tree = [{"name": area, "refactored_subtree": {}} for area in functional_areas]
        trajectory = {}

        area_update = {area: {} for area in functional_areas}  # area -> path -> file node

        processed_files = set()
        batch_id = 0

        def _process_batch(batch_folders: List[str], batch_files: List[str]):
            nonlocal cur_feature_tree, batch_id, trajectory

            if not batch_files:
                return

            batch_id += 1
            batch_folder_path = ", ".join(batch_folders)

            self.logger.info(
                f"Processing batch #{batch_id}: folders={batch_folders}, files={len(batch_files)}"
            )

            folder_sub_tree = {k: v for k, v in parsed_tree.items() if k in batch_files}
            dir_file2node = {file: node for file, node in file2node.items() if file in batch_files}

            try:
                cur_tree, history = self.process_folder(
                    functional_areas=functional_areas,
                    folder_path=batch_folder_path,
                    cur_feature_tree=cur_feature_tree,
                    dir_file2node=dir_file2node,
                    area_update=area_update,
                    parsed_tree=folder_sub_tree,
                    context_window=context_window,
                    max_iters=max_iters,
                )

                cur_feature_tree = cur_tree

                for f in batch_folders:
                    trajectory[f] = {"id": batch_id, "trajectory": history}

                self.logger.info(f"Finished batch #{batch_id} for folders: {batch_folders}")

            except Exception as e:
                self.logger.error(
                    f"Failed to process batch #{batch_id} for folders {batch_folders}: {e}",
                    exc_info=True,
                )

        pending_folders: List[str] = []
        pending_files: List[str] = []

        def _flush_pending():
            nonlocal pending_folders, pending_files
            if not pending_files:
                return
            _process_batch(pending_folders, pending_files)
            processed_files.update(pending_files)
            pending_folders, pending_files = [], []

        # Step 4. Iterate folders with progress bar（max_batch_size + token_budget 切分）
        with tqdm(total=total_folders, desc="Refactoring folders", ncols=100) as pbar:
            for idx, (folder, file_paths) in enumerate(folder_items):
                self.logger.info(f"Scanning folder [{idx+1}/{total_folders}]: {folder}")

                file_paths = list(file_paths)
                unprocessed_files = [f for f in file_paths if f not in processed_files]

                if not unprocessed_files:
                    pbar.update(1)
                    continue

                for f in unprocessed_files:
                 
                    if pending_files and len(pending_files) >= max_batch_size:
                        _flush_pending()

                    if pending_files:
                        test_files = pending_files + [f]
                        test_folders = (
                            pending_folders[:] if folder in pending_folders else pending_folders + [folder]
                        )
                        folder_path = ", ".join(test_folders)
                        folder_sub_tree = {k: parsed_tree[k] for k in test_files}
                        est = self._estimate_batch_tokens_for_process_folder(
                            functional_areas=functional_areas,
                            folder_path=folder_path,
                            cur_feature_tree=cur_feature_tree,
                            folder_sub_tree=folder_sub_tree,
                        )
                        if est > token_budget:
                            _flush_pending()

                    pending_files.append(f)
                    if folder not in pending_folders:
                        pending_folders.append(folder)

                    if len(pending_files) == 1:
                        folder_path = ", ".join(pending_folders)
                        folder_sub_tree = {k: parsed_tree[k] for k in pending_files}
                        est1 = self._estimate_batch_tokens_for_process_folder(
                            functional_areas=functional_areas,
                            folder_path=folder_path,
                            cur_feature_tree=cur_feature_tree,
                            folder_sub_tree=folder_sub_tree,
                        )
                        if est1 > token_budget:
                            _flush_pending()

                pbar.update(1)

        _flush_pending()

        self.rpg.update_result_to_rpg(area_update)
        
        # Remove empty subtrees after refactoring
        removal_stats = self.rpg.remove_empty_subtrees()
        if removal_stats["removed_nodes"] > 0:
            self.logger.info(f"Cleaned up {removal_stats['removed_nodes']} empty subtrees after refactoring")

      
        return cur_feature_tree, trajectory, self.rpg

    @classmethod
    def refactor_new_files(
        cls,
        parsed_tree: Dict,
        existing_feature_tree: List[Dict],
        existing_rpg: RPG,
        repo_dir: str,
        repo_name: str,
        repo_info: str,
        repo_skeleton: RepoSkeleton,
        skeleton_info: str,
        functional_areas: Optional[List[str]] = None,
        context_window: int = 5,
        max_batch_size: int = 50,          # NEW
        max_prompt_tokens: int = 90000,    # NEW
        reserve_output_tokens: int = 6000, # NEW
        max_iters: int = 10,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[List[Dict], Dict, RPG]:
        """
        Class method version — incrementally refactor new/modified files using existing results.
        Includes robust deduplication logic to prevent node name collisions with existing RPG nodes.
        """

        token_budget = max_prompt_tokens - reserve_output_tokens
        assert token_budget > 0, "token_budget must be > 0"

        if logger is None:
            logger = setup_logger(logging.getLogger(f"RPGRefactorIncremental[{repo_name}]"))

        logger.info("🔁 [ClassMethod] Starting incremental refactor on new files...")

        # === Step 1. Create temporary instance for GPT + repo context ===
        instance = cls(
            repo_dir=repo_dir,
            repo_info=repo_info,
            repo_skeleton=repo_skeleton,
            skeleton_info=skeleton_info,
            repo_name=repo_name,
            logger=logger,
        )

        # === Step 2. Restore RPG ===
        instance.rpg = existing_rpg

        # === Step 3. Determine functional areas ===
        if not functional_areas:
            instance.logger.info("Re-planning functional areas from parsed_tree...")
            plan = instance.plan_functional_areas(parsed_tree, max_iters=2)
            functional_areas = plan.get("final_plan", [])
            assert functional_areas, "Functional areas could not be inferred."
        else:
            instance.logger.info(f"Using existing functional areas: {functional_areas}")

        # === Step 4. Prepare code structure mappings ===
        file2code = repo_skeleton.get_file_code_map()
        file2unit = {
            file: ParsedFile(code=code, file_path=file).units for file, code in file2code.items()
        }

        new_file2node: Dict[str, Node] = {}

        # === Global used names set ===
        used_file_names = set()
        for node in instance.rpg.nodes.values():
            if node.meta.type_name == NodeType.FILE:
                used_file_names.add(node.name)

        sorted_files = sorted(parsed_tree.keys())

        # === Step 5. Build nodes for new/modified files ===
        for file_path in sorted_files:
            f_features = parsed_tree[file_path]

            original_summary = f_features.get(
                "_file_summary_", os.path.basename(file_path).replace(".py", "")
            )

            current_summary = original_summary
            if current_summary in used_file_names:
                counter = 1
                while True:
                    candidate = f"{original_summary}_{counter}"
                    if candidate not in used_file_names:
                        current_summary = candidate
                        break
                    counter += 1

            used_file_names.add(current_summary)
            f_features["_file_summary_"] = current_summary

            uid = str(uuid.uuid4())[:6]
            file_node = Node(
                id=f"{current_summary}_{uid}",
                name=current_summary,
                meta=NodeMetaData(type_name=NodeType.FILE, path=file_path),
            )
            instance.rpg.add_node(file_node)
            new_file2node[file_path] = file_node

            file_units = file2unit.get(file_path, [])
            key2unit: Dict[str, ParsedFile] = {}
            for u in file_units:
                if u.unit_type == "method":
                    key = f"method {u.parent}.{u.name}"
                else:
                    key = f"{u.unit_type} {u.name}"
                key2unit[key] = u

            for unit_name, unit_features in f_features.items():
                if unit_name == "_file_summary_":
                    continue

                if unit_name.startswith("function"):
                    func_name = unit_name.replace("function ", "").strip()
                    func_unit = key2unit.get(f"function {func_name}", None)
                    if not func_unit:
                        continue

                    for feature in unit_features:
                        uid = str(uuid.uuid4())[:6]
                        node = Node(
                            id=f"{feature}_{uid}",
                            name=feature,
                            meta=NodeMetaData(
                                type_name=NodeType.FUNCTION, path=f"{file_path}:{func_name}"
                            ),
                            unit=func_unit.key(),
                        )
                        instance.rpg.add_node(node)
                        instance.rpg.add_edge(src=file_node, dst=node)

                elif unit_name.startswith("class"):
                    class_name = unit_name.replace("class ", "").strip()
                    cls_unit = key2unit.get(f"class {class_name}", None)

                    if isinstance(unit_features, list):
                        if not cls_unit:
                            continue
                        for feat in unit_features:
                            uid = str(uuid.uuid4())[:6]
                            node = Node(
                                id=f"{feat}_{uid}",
                                name=feat,
                                meta=NodeMetaData(
                                    type_name=NodeType.CLASS, path=f"{file_path}:{class_name}"
                                ),
                                unit=cls_unit.key(),
                            )
                            instance.rpg.add_node(node)
                            instance.rpg.add_edge(src=file_node, dst=node)

                    elif isinstance(unit_features, dict):
                        for m_name, m_feats in unit_features.items():
                            mtd_unit = key2unit.get(f"method {class_name}.{m_name}", None)
                            if not mtd_unit:
                                continue
                            for feat in m_feats:
                                uid = str(uuid.uuid4())[:6]
                                node = Node(
                                    id=f"{feat}_{uid}",
                                    name=feat,
                                    meta=NodeMetaData(
                                        type_name=NodeType.METHOD,
                                        path=f"{file_path}:{class_name}.{m_name}",
                                    ),
                                    unit=mtd_unit.key(),
                                )
                                instance.rpg.add_node(node)
                                instance.rpg.add_edge(src=file_node, dst=node)

        # === Step 6. Folder-level iteration (max_batch_size + token_budget) ===
        folders_dict = iterative_by_folder(parsed_tree)
        folder_items = list(folders_dict.items())
        total_folders = len(folder_items)
        instance.logger.info(f"📂 Incremental folders to process: {total_folders}")

        cur_feature_tree = deepcopy(existing_feature_tree)
        trajectory: Dict[str, Dict] = {}

        area_update = {}
        for area in existing_feature_tree:
            if isinstance(area, dict) and "name" in area:
                area_update[area["name"]] = {}

        processed_files: set = set()
        batch_id = 0

        def _process_batch(batch_folders: List[str], batch_files: List[str]):
            nonlocal cur_feature_tree, batch_id, trajectory

            if not batch_files:
                return

            batch_id += 1
            batch_folder_path = ", ".join(batch_folders)

            instance.logger.info(
                f"[Incremental] Processing batch #{batch_id}: folders={batch_folders}, files={len(batch_files)}"
            )

            folder_sub_tree = {k: v for k, v in parsed_tree.items() if k in batch_files}
            dir_file2node = {f: n for f, n in new_file2node.items() if f in batch_files}

            try:
                cur_tree, history = instance.process_folder(
                    functional_areas=functional_areas,
                    folder_path=batch_folder_path,
                    cur_feature_tree=cur_feature_tree,
                    dir_file2node=dir_file2node,
                    area_update=area_update,
                    parsed_tree=folder_sub_tree,
                    context_window=context_window,
                    max_iters=max_iters,
                )
                cur_feature_tree = cur_tree
                for f in batch_folders:
                    trajectory[f] = {"id": batch_id, "trajectory": history}

            except Exception as e:
                instance.logger.error(
                    f"[Incremental] Failed to process batch #{batch_id} for folders {batch_folders}: {e}",
                    exc_info=True,
                )

        pending_folders: List[str] = []
        pending_files: List[str] = []

        def _flush_pending():
            nonlocal pending_folders, pending_files
            if not pending_files:
                return
            _process_batch(pending_folders, pending_files)
            processed_files.update(pending_files)
            pending_folders, pending_files = [], []

        with tqdm(total=total_folders, desc="Incremental Refactoring", ncols=100) as pbar:
            for idx, (folder, file_paths) in enumerate(folder_items):
                instance.logger.info(f"[Incremental] Scanning folder [{idx+1}/{total_folders}]: {folder}")

                file_paths = [f for f in list(file_paths) if f not in processed_files]
                if not file_paths:
                    pbar.update(1)
                    continue

                for f in file_paths:
                    if pending_files and len(pending_files) >= max_batch_size:
                        _flush_pending()

                    if pending_files:
                        test_files = pending_files + [f]
                        test_folders = (
                            pending_folders[:] if folder in pending_folders else pending_folders + [folder]
                        )
                        folder_path = ", ".join(test_folders)
                        folder_sub_tree = {k: parsed_tree[k] for k in test_files}

                        est = instance._estimate_batch_tokens_for_process_folder(
                            functional_areas=functional_areas,
                            folder_path=folder_path,
                            cur_feature_tree=cur_feature_tree,
                            folder_sub_tree=folder_sub_tree,
                        )
                        if est > token_budget:
                            _flush_pending()

                    pending_files.append(f)
                    if folder not in pending_folders:
                        pending_folders.append(folder)

                    if len(pending_files) == 1:
                        folder_path = ", ".join(pending_folders)
                        folder_sub_tree = {k: parsed_tree[k] for k in pending_files}
                        est1 = instance._estimate_batch_tokens_for_process_folder(
                            functional_areas=functional_areas,
                            folder_path=folder_path,
                            cur_feature_tree=cur_feature_tree,
                            folder_sub_tree=folder_sub_tree,
                        )
                        if est1 > token_budget:
                            _flush_pending()

                pbar.update(1)

        _flush_pending()

        # === Step 7. Update RPG structure ===
        instance.rpg.update_result_to_rpg(area_update)
        
        # Remove empty subtrees after refactoring
        removal_stats = instance.rpg.remove_empty_subtrees()
        if removal_stats["removed_nodes"] > 0:
            instance.logger.info(f"Cleaned up {removal_stats['removed_nodes']} empty subtrees after incremental refactoring")
        
        instance.logger.info("Incremental refactor complete (classmethod).")

        return cur_feature_tree, trajectory, instance.rpg

    # =========================================================================
    # Modified files refactoring
    # =========================================================================

    def _get_file_feature_paths(self, file_paths: List[str]) -> Dict[str, str]:
        """Get current feature paths for file nodes in the RPG.

        Returns:
            Dict mapping file summary name -> feature path string
        """
        paths: Dict[str, str] = {}
        for _nid, node in self.rpg.nodes.items():
            if node.meta and node.meta.type_name == NodeType.FILE:
                fp = node.meta.path
                if isinstance(fp, list):
                    fp = fp[0] if fp else None
                if fp and fp in file_paths:
                    paths[node.name] = node.feature_path()
        return paths

    def _detach_file_from_feature_tree(self, file_node_id: str):
        """Detach a file node from its current parent in the feature tree.

        Removes the edge between the file node and its parent DIRECTORY node,
        but keeps the file node and all its children (unit nodes) intact.
        """
        parent_id = self.rpg._parents.get(file_node_id)
        if not parent_id:
            return

        # Remove from parent's adjacency list
        adj = self.rpg._adjacency.get(parent_id, [])
        if file_node_id in adj:
            adj.remove(file_node_id)

        # Remove edge
        self.rpg.edges = [
            e for e in self.rpg.edges
            if not (e.src == parent_id and e.dst == file_node_id)
        ]

        # Remove parent reference
        if file_node_id in self.rpg._parents:
            del self.rpg._parents[file_node_id]

    def _build_process_modified_user_prompt(
        self,
        *,
        functional_areas: List,
        cur_rpg_info: str,
        modified_files_info: str,
    ) -> str:
        return (
            f"### Functional Areas (L1 — do NOT change):\n"
            f"<functional_areas>\n{json.dumps(functional_areas)}\n</functional_areas>\n\n"
            f"### Current Refactored Repository Tree:\n"
            f"<current_refactored_tree>\n{cur_rpg_info}\n</current_refactored_tree>\n\n"
            f"### Modified Files:\n"
            f"<modified_files>\n{modified_files_info}\n</modified_files>\n\n"
            "For each file above, decide whether its current L2-L3 placement still "
            "makes sense given the updated features. Return the path mapping as specified."
        )

    def _validate_modified_action(
        self,
        action: Dict,
        modified_files_input: Dict[str, Dict],
        functional_areas: List[str],
    ) -> Tuple[Dict[str, str], str]:
        """Validate LLM output for modified files.

        Returns:
            (valid_mapping, feedback_str)
            - valid_mapping: {old_path: new_path} for correctly mapped files
            - feedback_str: non-empty if there are issues the LLM should fix
        """
        input_keys = set(modified_files_input.keys())
        valid_fa_set = {fa.strip().lower(): fa for fa in functional_areas}
        errors = []
        valid_mapping: Dict[str, str] = {}

        for old_path, new_path in action.items():
            # Check key exists in input
            if old_path not in input_keys:
                errors.append(
                    f"- Key \"{old_path}\" is not a valid original path. "
                    f"You must use the exact keys from <modified_files>."
                )
                continue

            if not isinstance(new_path, str):
                errors.append(
                    f"- Value for \"{old_path}\" must be a string path, "
                    f"got {type(new_path).__name__}."
                )
                continue

            new_parts = new_path.split("/")
            if len(new_parts) != 4:
                errors.append(
                    f"- \"{new_path}\" has {len(new_parts)} levels, expected exactly 4."
                )
                continue

            # Check L1 matches original
            old_parts = old_path.split("/")
            if new_parts[0] != old_parts[0]:
                real_l1 = old_parts[0]
                errors.append(
                    f"- \"{old_path}\": L1 must stay \"{real_l1}\", "
                    f"but you wrote \"{new_parts[0]}\"."
                )
                continue

            # Check L1 is a valid functional area
            if new_parts[0].strip().lower() not in valid_fa_set:
                errors.append(
                    f"- \"{new_path}\": L1 \"{new_parts[0]}\" is not a recognized "
                    f"functional area. Valid: {functional_areas}"
                )
                continue

            # Check L4 matches the required new_name
            expected_l4 = modified_files_input[old_path].get("new_name")
            if expected_l4 and new_parts[3] != expected_l4:
                errors.append(
                    f"- \"{old_path}\": L4 must be \"{expected_l4}\" "
                    f"(the new_name), but you wrote \"{new_parts[3]}\"."
                )
                continue

            valid_mapping[old_path] = new_path

        # Check for missing files
        missing = input_keys - set(action.keys())
        if missing:
            errors.append(
                f"- Missing files (every input key must appear in output): "
                f"{json.dumps(sorted(missing))}"
            )

        feedback = ""
        if errors:
            feedback = (
                "Your output has the following issues:\n\n"
                + "\n".join(errors)
                + "\n\nPlease fix these issues and return the complete JSON again. "
                "Remember:\n"
                "- Keys must be the EXACT original paths from <modified_files>.\n"
                "- L1 must NOT change from the original.\n"
                "- L4 must be the `new_name` provided for each file.\n"
                "- Every input file must appear in the output.\n"
            )

        return valid_mapping, feedback

    def process_modified_batch(
        self,
        functional_areas: List,
        cur_feature_tree: List[Dict],
        modified_files_input: Dict[str, Dict],
        context_window: int = 5,
        max_iters: int = 3,
    ) -> Dict[str, str]:
        """LLM call with feedback loop to decide path adjustments for modified files.

        Args:
            functional_areas: L1 functional area names.
            cur_feature_tree: current refactored tree (for context).
            modified_files_input: {old_L1/L2/L3/L4: {new_name: str, features: [str]}}
            max_iters: max feedback rounds before accepting partial results.

        Returns:
            {old_path: new_path} mapping.
        """
        cur_rpg_info = get_rpg_info(
            rpg_tree=cur_feature_tree, omit_leaf_nodes=True, sample_size=0
        )
        modified_files_info = json.dumps(
            modified_files_input, ensure_ascii=False, indent=2
        )

        user_prompt = self._build_process_modified_user_prompt(
            functional_areas=functional_areas,
            cur_rpg_info=cur_rpg_info,
            modified_files_info=modified_files_info,
        )

        self.logger.info(
            f"Calling LLM for {len(modified_files_input)} modified files..."
        )

        agent_memory = Memory(context_window=context_window)
        agent_memory.add_message(SystemMessage(REFACTOR_MODIFIED))
        agent_memory.add_message(UserMessage(user_prompt))

        for it in range(max_iters):
            action, response = self.step(agent_memory)
            agent_memory.add_message(AssistantMessage(response))

            valid_mapping, feedback = self._validate_modified_action(
                action, modified_files_input, functional_areas
            )

            if not feedback:
                self.logger.info(
                    f"All {len(valid_mapping)} files validated on iteration {it + 1}"
                )
                return valid_mapping

            self.logger.info(
                f"Iteration {it + 1}: {len(valid_mapping)} valid, "
                f"sending feedback to LLM"
            )
            agent_memory.add_message(UserMessage(feedback))

        # After max_iters, fill in any remaining files with fallback
        self.logger.warning(
            f"Max iterations reached. "
            f"Valid: {len(valid_mapping)}/{len(modified_files_input)}"
        )
        for old_path, info in modified_files_input.items():
            if old_path not in valid_mapping:
                old_parts = old_path.split("/")
                if len(old_parts) == 4:
                    new_l4 = info.get("new_name", old_parts[3])
                    fallback = "/".join(old_parts[:3] + [new_l4])
                    valid_mapping[old_path] = fallback
                    self.logger.info(
                        f"Fallback for {old_path}: {fallback}"
                    )

        return valid_mapping

    @classmethod
    def refactor_modified_files(
        cls,
        parsed_tree: Dict,
        existing_feature_tree: List[Dict],
        existing_rpg: RPG,
        repo_dir: str,
        repo_name: str,
        repo_info: str,
        repo_skeleton: RepoSkeleton,
        skeleton_info: str,
        functional_areas: Optional[List[str]] = None,
        context_window: int = 5,
        max_iters: int = 10,
        logger: Optional[logging.Logger] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> Tuple[List[Dict], Dict, RPG]:
        """
        Refactor modified files in the RPG.

        Shows the LLM each file's original 4-level feature path (L1/L2/L3/L4)
        plus its updated features, then lets the LLM decide whether to keep
        the current L2-L3 placement or adjust it.

        LLM output format: {original_L1/L2/L3/L4: new_L1/L2/L3/new_L4}

        Unlike ``refactor_new_files``, this method does NOT create new file or
        unit nodes — those should already exist in ``existing_rpg`` (e.g. via
        ``RPG.update_from_parsed_tree``).  It only re-evaluates *where* the
        file nodes sit in the feature-tree hierarchy.
        """

        if logger is None:
            logger = setup_logger(logging.getLogger(f"RPGRefactorModified[{repo_name}]"))

        logger.info("Starting refactor for modified files...")

        # === Step 1. Create temporary instance ===
        instance = cls(
            repo_dir=repo_dir,
            repo_info=repo_info,
            repo_skeleton=repo_skeleton,
            skeleton_info=skeleton_info,
            repo_name=repo_name,
            logger=logger,
            llm_config=llm_config,
        )
        instance.rpg = existing_rpg

        # === Step 2. Determine functional areas ===
        if not functional_areas:
            functional_areas = instance.rpg.get_functional_areas()
            assert functional_areas, "Functional areas could not be inferred."
        else:
            instance.logger.info(f"Using existing functional areas: {functional_areas}")

        # === Step 3. Collect file nodes, update L4 names, build LLM input ===
        modified_files = set(parsed_tree.keys())
        existing_file_nodes: Dict[str, Node] = {}   # file_path -> Node
        old_path_to_file: Dict[str, str] = {}       # old_feature_path -> file_path

        # Get flat features per file via transfer_parsed_tree
        transfered_tree, _ = transfer_parsed_tree(input_tree=parsed_tree)

        # Build modified_files_input: {old_L1/L2/L3/L4: {new_name, features}}
        modified_files_input: Dict[str, Dict] = {}

        for _nid, node in instance.rpg.nodes.items():
            if node.meta and node.meta.type_name == NodeType.FILE:
                fp = node.meta.path
                if isinstance(fp, list):
                    fp = fp[0] if fp else None
                if fp and fp in modified_files:
                    existing_file_nodes[fp] = node
                    old_feature_path = node.feature_path()

                    new_summary = parsed_tree[fp].get(
                        "_file_summary_",
                        os.path.basename(fp).replace(".py", ""),
                    )
                    old_path_to_file[old_feature_path] = fp

                    # Get flat features for this file
                    flat_features = transfered_tree.get(new_summary, [])
                    if not flat_features:
                        flat_features = transfered_tree.get(node.name, [])

                    modified_files_input[old_feature_path] = {
                        "new_name": new_summary,
                        "features": flat_features,
                    }

                    # Update L4 FILE node name to the new summary
                    if node.name != new_summary:
                        instance.logger.info(
                            f"Updating FILE node name: '{node.name}' -> '{new_summary}'"
                        )
                        node.name = new_summary

        instance.logger.info(
            f"Found {len(existing_file_nodes)} file nodes for "
            f"{len(modified_files)} modified files"
        )

        if not modified_files_input:
            instance.logger.info("No modified files to refactor.")
            return existing_feature_tree, {}, instance.rpg

        # === Step 4. Call LLM to get path mapping ===
        cur_feature_tree = deepcopy(existing_feature_tree)
        path_mapping = instance.process_modified_batch(
            functional_areas=functional_areas,
            cur_feature_tree=cur_feature_tree,
            modified_files_input=modified_files_input,
            context_window=context_window,
        )

        instance.logger.info(
            f"LLM path mapping: {json.dumps(path_mapping, indent=2)}"
        )

        # === Step 5. Detach files from old locations & re-attach at new paths ===
        area_update: Dict[str, Dict[str, Node]] = {}
        for area in existing_feature_tree:
            if isinstance(area, dict) and "name" in area:
                area_update[area["name"]] = {}

        # Fallback: if existing_feature_tree was empty, derive from RPG
        if not area_update:
            for fa in functional_areas:
                area_update[fa] = {}

        trajectory: Dict[str, str] = {}

        for old_path, new_path in path_mapping.items():
            file_path = old_path_to_file.get(old_path)
            if not file_path:
                instance.logger.warning(
                    f"Old path not found in mapping: {old_path}"
                )
                continue

            file_node = existing_file_nodes.get(file_path)
            if not file_node:
                instance.logger.warning(f"No file node for: {file_path}")
                continue

            parts = new_path.split("/")
            if len(parts) != 4:
                instance.logger.warning(
                    f"Invalid new path (expected 4 levels): {new_path}"
                )
                continue

            area_name = parts[0]
            if area_name not in area_update:
                instance.logger.warning(
                    f"Unknown functional area: {area_name}"
                )
                continue

            # Detach from old parent
            instance._detach_file_from_feature_tree(file_node.id)

            # Register for re-attachment
            area_update[area_name][new_path] = file_node
            trajectory[old_path] = new_path
            instance.logger.info(f"  {old_path} -> {new_path}")

        # Re-attach at new locations
        instance.rpg.update_result_to_rpg(area_update)

        # Clean up empty subtrees
        removal_stats = instance.rpg.remove_empty_subtrees()
        if removal_stats["removed_nodes"] > 0:
            instance.logger.info(
                f"Cleaned up {removal_stats['removed_nodes']} empty subtrees "
                f"after modified-file refactoring"
            )

        instance.logger.info("Modified-file refactoring complete.")

        return cur_feature_tree, trajectory, instance.rpg

