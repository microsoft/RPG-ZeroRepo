"""
Given A repo, we need to parse it into the RPG
"""
import logging
import json
import time
import os
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import defaultdict
from networkx import MultiDiGraph

from .semantic_parsing import ParseFeatures
from .refactor_tree import RefactorTree
from .prompts import GENERATE_REPO_INFO, EXCLUDE_FILES
from zerorepo.utils.api import (
    parse_code_blocks,
    truncate_by_token,
    parse_solution_output
)
from zerorepo.rpg_gen.base.rpg import RPG, NodeMetaData
from zerorepo.rpg_gen.base.node import (
    FileNode, RepoSkeleton, 
    filter_non_test_py_files
)
from zerorepo.utils.repo import (
    load_skeleton_from_repo, normalize_path,
    exclude_files, is_test_file
)
from zerorepo.utils.compress import get_skeleton
from zerorepo.rpg_gen.base.rpg import NodeType, EdgeType, DependencyGraph
from zerorepo.rpg_encoder.rpg_agent.env import RepoEntitySearcher
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient,
    LLMConfig,
    Memory,
    UserMessage,
    AssistantMessage,
    SystemMessage
)
from zerorepo.utils.logs import setup_logger


class RPGParser:
    
    def __init__(
        self,
        repo_dir,
        repo_name, 
        logger: Optional[logging.Logger]=None, 
        llm_config: Optional[LLMConfig]=LLMConfig({"model": "gpt-4o-20241120"})
    ):
        self.repo_dir = repo_dir
        self.repo_name = repo_name
        
        if not logger:
            self.logger = setup_logger(logging.getLogger(f"RPGParser[{repo_name}]"))
        else:
            self.logger= logger
            
        # Initialize LLM client
        self.llm_config = llm_config
        self.llm_client = LLMClient(llm_config)
       
        self.logger.info("Initializing RPGParser: dir=%s, name=%s", repo_dir, repo_name)
        self.repo_skeleton, self.skeleton_info, self.valid_files \
            = load_skeleton_from_repo(repo_dir=self.repo_dir) #  filter_func=filter_non_test_py_files)
        self.skeleton_info = truncate_by_token(self.skeleton_info, max_tokens=50000).strip()
        self.logger.info("Skeleton loaded: files=%d", len(self.valid_files))
        self.logger.info("Filtered skeleton (non-test .py only):\n%s", self.skeleton_info)
        
    def generate_repo_info(self, max_iters=3):
        """Generate Description of the Repo"""    
        self.logger.info("Generating repo info (max_iters=%d)...", max_iters)

        readme_files = ["README.md", "readme.md", "readme.txt", "README", "readme"]
        readme = ""
        for r_file in readme_files:
            file_node: FileNode = self.repo_skeleton.find_file(path=r_file)
            if file_node is None:
                continue
            readme = file_node.code
            self.logger.info("README found: %s (length=%d)", r_file, len(readme))
            break
        if not readme:
            self.logger.warning("README not found; proceeding with empty README content.")

        readme = truncate_by_token(
            text=readme,
            max_tokens=50000
        ).strip()
        user_prompt = (
            f"Repository Name:\n<repo_name>\n{self.repo_name}\n</repo_name>\n"
            f"Repository Structure:\n<skeleton>\n{self.skeleton_info}\n</skeleton>\n"
            f"Repository README Content:\n<readme>\n{readme}\n</readme>\n"
            f"Based on the information above, please summarize and generate a comprehensive Repository Overview."
        )
        
     
        self.logger.info("GENERATE_REPO_INFO user prompt:\n%s",
                              user_prompt)
        
        # Initialize memory
        memory = Memory(context_window=10)
        memory.add_message(SystemMessage(GENERATE_REPO_INFO))
        memory.add_message(UserMessage(user_prompt))
        
        repo_info = ""
        for i in range(max_iters):
            try:
                self.logger.info("LLM call for repo info, iter=%d...", i+1)
                response = self.llm_client.generate(memory)
                self.logger.info(f"Iter {i + 1} Response:{response}")
                parsed_response = parse_solution_output(response)
                code_blocks = parse_code_blocks(output=parsed_response, type="general")
                if not code_blocks:
                    self.logger.warning("No code blocks parsed in iter=%d; skipping.", i+1)
                    continue
                repo_info = "\n".join(code_blocks)
                self.logger.info("Repo info received (len=%d) in iter=%d.", len(repo_info), i+1)
                if repo_info:
                    break
            except Exception as e:
                self.logger.exception("Error generating repo info at iter=%d: %s", i+1, e)
                continue
        
        if not repo_info:
            self.logger.warning("Repo info is empty after %d iterations.", max_iters)
        return repo_info

    
    def exclude_irrelvant_files(self, repo_info, max_votes: int = 3):
        """
        Identify and summarize irrelevant files or folders in the repository 
        that should be excluded from functional extraction.
        """
        self.logger.info("Excluding irrelevant files (max_votes=%d)...", max_votes)

        user_prompt = (
            f"Repository Name:\n<readme>\n{self.repo_name}\n</readme>\n"
            f"Repository Structure:\n<skeleton>\n{self.skeleton_info}\n</skeleton>\n"
            f"Repository Overview:\n<repo_info>\n{repo_info}\n</repo_info>\n"
            f"Based on the information above, please analyze this repository and identify "
            f"the paths that are likely unrelated to the core algorithms or main implementation, "
            f"such as folders or files that appear to be forked from other repositories, "
            f"third-party code, demo data, documentation, or build/test artifacts. "
            f"List these unrelated paths clearly so they can be excluded from further functional extraction."
        )
        
        self.logger.info("EXCLUDE_FILES user prompt:\n%s",
                              user_prompt)

        # Initialize memory
        memory = Memory(context_window=10)
        memory.add_message(SystemMessage(EXCLUDE_FILES))
        memory.add_message(UserMessage(user_prompt))

        excluded_files_all_rounds: List[List[str]] = []

        for i in range(max_votes):
            try:
                self.logger.info("LLM vote #%d for exclude list...", i+1)
                response = self.llm_client.generate(memory)
                self.logger.info(f"Iter {i + 1} Response:{response}")
                parsed_response = parse_solution_output(response)
                code_blocks = parse_code_blocks(output=parsed_response, type="general")
                code_blocks = code_blocks if code_blocks else parsed_response.split("\n")
                if i == 0:
                    memory.add_message(AssistantMessage(response))
                if not code_blocks:
                    self.logger.warning("Vote #%d produced no code block; skipping.", i+1)
                    continue

                file_text = "\n".join(code_blocks)
                file_paths = [
                    normalize_path(file.strip())
                    for file in file_text.split("\n")
                    if file.strip() and self.repo_skeleton.path_exists(file)
                ]
                excluded_files_all_rounds.append(file_paths)

                self.logger.info("Vote #%d candidates: %d (after existence check).",
                                 i+1, len(file_paths))
                if file_paths:
                    head = file_paths[:5]
                    tail = file_paths[-5:] if len(file_paths) > 5 else []
                    self.logger.debug("Vote #%d sample head: %s", i+1, head)
                    if tail:
                        self.logger.debug("Vote #%d sample tail: %s", i+1, tail)

            except Exception as e:
                self.logger.exception("Error during exclude vote #%d: %s", i+1, e)
                continue

        flat_list = [p for sublist in excluded_files_all_rounds for p in sublist]
        self.logger.info("Total raw candidates across votes: %d", len(flat_list))
        combined_sections = []
        for idx, paths in enumerate(excluded_files_all_rounds, start=1):
            if not paths:
                continue
            section = (
                f"### Round {idx} Excluded Paths\n"
                + "\n".join(f"- {p}" for p in paths)
            )
            combined_sections.append(section)

        combined_text = "\n\n".join(combined_sections)
        self.logger.info("Combined exclude candidates text:\n%s", combined_text[:1000])
        
        summarize_prompt = (
            "Below are multiple rounds of proposed irrelevant paths extracted from the repository:\n"
            f"<round_outputs>\n{combined_text}\n</round_outputs>\n"
            "Please review and consolidate these results into a single, clean list of paths "
            "that are most likely irrelevant or should be excluded from further analysis. "
            "Remove duplicates and keep only the final recommended paths.\n"
            "Output one path per line, no explanations."
        )
        memory.add_message(UserMessage(summarize_prompt))
        
        self.logger.info("Summarize prompt:\n%s",
                              summarize_prompt)

        final_result: List[str] = []
        try:
            self.logger.info("Consolidating exclude list with a final LLM pass...")
            response = self.llm_client.generate(memory)
            parsed_response = parse_solution_output(response)
            code_blocks = parse_code_blocks(output=parsed_response, type="general")
            if code_blocks:
                final_text = "\n".join(code_blocks)
                final_result = [
                    normalize_path(line.strip())
                    for line in final_text.split("\n")
                    if line.strip() and self.repo_skeleton.path_exists(line)
                ]
        
            final_result = sorted(set(final_result))
            self.logger.info("Final exclude list size: %d", len(final_result))
            if final_result:
                self.logger.debug("Exclude list head: %s", final_result[:5])
                if len(final_result) > 10:
                    self.logger.debug("Exclude list tail: %s", final_result[-5:])
        except Exception as e:
            self.logger.exception("Final consolidation failed: %s", e)

        all_paths = list(self.repo_skeleton.path_to_node.keys())
        filter_paths = exclude_files(files=all_paths)
        final_result = final_result + filter_paths
    
        return final_result
            

    def parse_rpg_from_repo(
        self,
        repo_info: Optional[str] = None,
        max_repo_info_iters: int=3,
        max_exclude_votes: int=3,
        max_parse_iters: int=10,
        # Parse features parameters
        min_batch_tokens: int=10_000,
        max_batch_tokens: int=50_000,
        summary_min_batch_tokens: int=10_000,
        summary_max_batch_tokens: int=50_000,
        class_context_window: int=10,
        func_context_window: int=10,
        max_parse_workers: int=8,
        # Refactor parameters
        refactor_context_window: int=10,
        refactor_max_iters: int=10,
        save_path: str="",
        # Data flow analysis parameters
        run_data_flow_analysis: bool=True,
        data_flow_max_results: int=3
    ) -> Tuple[RPG, List[Dict], "RepoSkeleton"]:

        final_result = {}

        self.logger.info("=== RPG parsing pipeline started ===")
        self.logger.info("Params: repo_info_iters=%d, exclude_votes=%d, parse_iters=%d, "
                         "min_batch=%d, max_batch=%d, class_ctx=%d, func_ctx=%d, workers=%d, "
                         "refactor_ctx=%d, refactor_iters=%d",
                         max_repo_info_iters, max_exclude_votes, max_parse_iters,
                         min_batch_tokens, max_batch_tokens, class_context_window, func_context_window,
                         max_parse_workers, refactor_context_window, refactor_max_iters
                    )

        # 1) Repo overview - use provided or generate
        if not repo_info:
            repo_info = self.generate_repo_info(max_iters=max_repo_info_iters)

        final_result["repo_name"] = self.repo_name
        final_result["repo_info"] = repo_info
        # 2) Exclude irrelevant
        excluded_files = self.exclude_irrelvant_files(repo_info=repo_info, max_votes=max_exclude_votes)
        final_result["excluded_files"] = excluded_files

        self.logger.info("Excluded paths decided: %d", len(excluded_files))
        if excluded_files:
            self.logger.debug("Excluded sample: %s", excluded_files)

        # 3) Parse features
        feature_parser = ParseFeatures(
            repo_dir=self.repo_dir,
            repo_info=repo_info,
            repo_skeleton=self.repo_skeleton,
            llm_config=self.llm_config,
            valid_files=self.valid_files,
            repo_name=self.repo_name,        
            logger=self.logger
        )
        self.logger.info("Parsing features...")


        file2feature, parse_traj = feature_parser.parse_repo(
            excluded_files=excluded_files,
            max_iterations=max_parse_iters,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_parse_workers
        )

        # final_result["file2feature"] = {"structure": file2feature} #"traj": parse_traj}
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(final_result, f, indent=4)

        self.logger.info("Features parsed: files=%d", len(file2feature))
        if file2feature:
            some_keys = list(file2feature.keys())[:10]
            self.logger.info("Feature map keys sample: %s", some_keys)

        # 4) Refactor to RPG
        refactor_agent = RefactorTree(
            repo_dir=self.repo_dir,
            repo_info=repo_info,
            repo_skeleton=self.repo_skeleton,
            llm_config=self.llm_config,
            repo_name=self.repo_name,
            logger=self.logger
        )
        self.logger.info("Refactoring to RPG...")
        final_rpg, refactor_traj, repo_rpg = refactor_agent.run(
            parsed_tree=file2feature,
            context_window=refactor_context_window,
            max_iters=refactor_max_iters
        )

        repo_rpg.repo_info = repo_info
        repo_rpg.excluded_files = excluded_files

        repo_rpg.update_all_metadata_bottom_up()
        
        # Remove empty subtrees after refactoring
        removal_stats = repo_rpg.remove_empty_subtrees()
        if removal_stats["removed_nodes"] > 0:
            self.logger.info(f"Cleaned up {removal_stats['removed_nodes']} empty subtrees after refactoring")

        final_result["rpg"] = {"structure": repo_rpg.to_dict(), "feature_tree": final_rpg, "traj": refactor_traj}

        # 5) Optional: Analyze data flow
        data_flow_result = {}
        if run_data_flow_analysis:
            self.logger.info("=== Running data flow analysis ===")
            try:
                # Build dependency graph
                dep_graph_builder = DependencyGraph(repo_dir=self.repo_dir)
                dep_graph_builder.build()
                dep_graph_builder.parse()

                # Build dep2rpg mapping and set on RPG
                dep2rpg = self._build_dep2rpg_mapping(dep_graph_builder.G, repo_rpg)
                repo_rpg.dep_graph = dep_graph_builder
                repo_rpg._dep_to_rpg_map = dep2rpg

                # Create entity searcher using RPG (which now contains dep_graph and mapping)
                entity_searcher = RepoEntitySearcher(rpg=repo_rpg)

                # Run data flow analysis
                data_flow_result = self.analyze_data_flow(
                    rpg=repo_rpg,
                    entity_searcher=entity_searcher,
                    include_code=True,
                    max_generate_results=data_flow_max_results
                )

                final_result["data_flow"] = data_flow_result
                self.logger.info("Data flow analysis completed")
            except Exception as e:
                self.logger.exception("Data flow analysis failed: %s", e)
                final_result["data_flow"] = {"error": str(e)}

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(final_result, f, indent=4)

        # Assign feature paths to skeleton file nodes
        self.assign_feature_paths_to_skeleton(repo_rpg)

        self.logger.info("RPG refactoring done.")
        self.logger.info("=== RPG parsing pipeline finished ===")


        return repo_rpg, final_rpg, self.repo_skeleton

    def _build_dep2rpg_mapping(
        self,
        dep_graph: MultiDiGraph,
        rpg: RPG
    ) -> Dict[str, List[str]]:
        """Build mapping from dependency graph node IDs to RPG node IDs"""
        dep2rpg_node = defaultdict(list)

        for nid in dep_graph.nodes():
            dep_node = dep_graph.nodes[nid]
            dep_node_type = dep_node.get("type")

            for node_id, rpg_node in rpg.nodes.items():
                rpg_node_meta = rpg_node.meta
                if not rpg_node_meta:
                    continue

                rpg_node_type = rpg_node_meta.type_name
                if rpg_node_type != dep_node_type:
                    continue

                rpg_node_paths = rpg_node_meta.path
                rpg_node_paths = rpg_node_paths if isinstance(rpg_node_paths, list) \
                    else [rpg_node_paths] if rpg_node_paths else []

                for rpg_node_path in rpg_node_paths:
                    if rpg_node_path == nid:
                        dep2rpg_node[nid].append(node_id)

        return dep2rpg_node

    def assign_feature_paths_to_skeleton(self, rpg: RPG) -> RepoSkeleton:
        """
        Assign feature paths to file nodes in the skeleton based on RPG structure.
        
        For each file node in the RPG, this method collects all child nodes' feature paths
        and assigns them to the corresponding file node in the skeleton.
        
        Returns:
            The updated skeleton with feature paths assigned to file nodes
        """
        self.logger.info("Assigning feature paths to skeleton file nodes...")
        
        # Build mapping from file paths to their feature paths in RPG
        file_to_feature_paths = {}
        
        # Traverse RPG nodes to find file nodes and collect all child feature paths
        for node_id, node in rpg.nodes.items():
            if node.meta and node.meta.type_name and node.meta.type_name == NodeType.FILE:
                # Get the file path from node metadata
                if node.meta.path:
                    file_path = node.meta.path
                    if isinstance(file_path, list):
                        file_path = file_path[0] if file_path else None
                    
                    if file_path:
                        if file_path not in file_to_feature_paths:
                            file_to_feature_paths[file_path] = []
                        
                        # Get all child nodes (recursive) under this file node
                        child_nodes = node.children(recursive=True)
                        
                        if child_nodes:
                            # If there are child nodes, collect their feature paths
                            for child_node in child_nodes:
                                # Get feature path for each child node
                                child_feature_path = child_node.feature_path()
                                file_to_feature_paths[file_path].append(child_feature_path)
                        else:
                            # If no child nodes, use the file node's own feature path
                            file_feature_path = node.feature_path()
                            file_to_feature_paths[file_path].append(file_feature_path)
        
        # Apply feature paths to skeleton file nodes
        updated_count = 0
        for file_path, feature_paths in file_to_feature_paths.items():
            file_node = self.repo_skeleton.find_file(path=file_path)
            if file_node and isinstance(file_node, FileNode):
                # Remove duplicates and assign feature paths
                unique_feature_paths = list(set(feature_paths))
                file_node.feature_paths = unique_feature_paths
                updated_count += 1
                self.logger.debug(f"Assigned feature paths to {file_path}: {unique_feature_paths}")
        
        self.logger.info(f"Successfully assigned feature paths to {updated_count} skeleton file nodes")
        
        return self.repo_skeleton


    def analyze_data_flow(
        self,
        rpg: RPG,
        entity_searcher: "RepoEntitySearcher",
        max_examples_per_pair: int = 12,
        include_code: bool = True,
        code_types: Tuple[NodeType, ...] = (NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD),
        exclude_tests: bool = True,
        code_from_examples_only: bool = True,
        keep_docstring: bool = False,
        max_generate_results: int = 3
    ) -> Dict[str, Union[str, Dict]]:
        """
        Analyze data flow between functional areas using DataFlowAgent logic.

        Args:
            rpg: The RPG graph (contains dep_graph and _dep_to_rpg_map)
            entity_searcher: Entity searcher for retrieving code content
            max_examples_per_pair: Max examples per cross-area call pair
            include_code: Include code snippets in output
            code_types: Types of code nodes to include
            exclude_tests: Exclude test files
            code_from_examples_only: Only include code from examples
            keep_docstring: Keep docstrings in code
            max_generate_results: Max LLM generation attempts

        Returns:
            Dict with final_result and all_traj
        """
        from zerorepo.rpg_gen.impl_level.func_design.agents.data_flow_agent import DataFlowAgent

        self.logger.info("Analyzing data flow using DataFlowAgent...")

        try:
            # Build cross-area code information for extra context
            extra_message = self._build_cross_area_code_context(
                rpg=rpg,
                entity_searcher=entity_searcher,
                include_code=include_code,
                code_types=code_types,
                exclude_tests=exclude_tests,
                code_from_examples_only=code_from_examples_only,
                keep_docstring=keep_docstring,
                max_examples_per_pair=max_examples_per_pair
            )

            data_flow_agent = DataFlowAgent(
                llm_cfg=self.llm_config,
                repo_rpg=rpg,
                max_review_times=0,
                repo_skeleton=self.repo_skeleton,
                logger=self.logger
            )

            # Generate data flow using the agent with extra code context
            result = data_flow_agent.generate_data_flow(
                max_retry=max_generate_results,
                max_steps=10,
                extra_message=extra_message
            )

            if result.get("success", False):
                self.logger.info("Data flow analysis completed successfully")
                return {
                    "final_result": result.get("data_flow", {}),
                    "all_traj": result.get("agent_results", {})
                }
            else:
                self.logger.warning("Data flow analysis did not complete successfully")
                return {
                    "final_result": result.get("data_flow", {}),
                    "all_traj": result.get("agent_results", {})
                }

        except Exception as e:
            self.logger.exception("Data flow analysis failed: %s", e)
            return {
                "final_result": {},
                "all_traj": {"error": str(e)}
            }

    def _build_cross_area_code_context(
        self,
        rpg: RPG,
        entity_searcher: "RepoEntitySearcher",
        include_code: bool = True,
        code_types: Tuple[NodeType, ...] = (NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD),
        exclude_tests: bool = True,
        code_from_examples_only: bool = True,
        keep_docstring: bool = False,
        max_examples_per_pair: int = 12
    ) -> str:
        """
        Build cross-area code context information for data flow analysis.
        This includes cross-area function calls and relevant code snippets.

        Args:
            rpg: The RPG graph (contains dep_graph and _dep_to_rpg_map)
            entity_searcher: Entity searcher for retrieving code content
        """
        # Get dep_graph and dep2rpg from RPG
        if rpg.dep_graph is None:
            return ""
        dep_graph = rpg.dep_graph.G
        dep2rpg = rpg._dep_to_rpg_map or {}

        # Helper functions
        def _l1_ancestor(rpg_node_id: str) -> Optional[str]:
            node = rpg.nodes.get(rpg_node_id)
            while node is not None and node.level is not None and node.level > 1:
                node = node.parent()
            if node and node.level == 1 and node.meta and node.meta.type_name == NodeType.DIRECTORY:
                return node.id
            return None

        def _dep_to_l1_areas(dep_id: str) -> List[str]:
            out: set = set()
            for rpg_id in dep2rpg.get(dep_id, []):
                l1 = _l1_ancestor(rpg_id)
                if l1 is not None:
                    out.add(l1)
            return list(out)

        def _pretty_dep_entity(nid: str) -> str:
            return nid

        def _is_invokes(t) -> bool:
            if t == EdgeType.INVOKES:
                return True
            if isinstance(t, str) and t.lower() in {"invokes", "call", "calls"}:
                return True
            return False

        def _is_test_nid(nid: str) -> bool:
            file_part = nid.split(":", 1)[0]
            return is_test_file(file_part)

        # Collect area metadata
        area_meta: Dict[str, Dict] = {}
        pair_stats: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {"count": 0, "examples": []})
        example_participants_by_pair: Dict[Tuple[str, str], set] = defaultdict(set)

        for n in rpg.nodes.values():
            if n.meta and n.meta.type_name == NodeType.DIRECTORY and n.level == 1:
                p = n.meta.path
                paths = p if isinstance(p, list) else ([p] if isinstance(p, str) and p else [])
                if not paths:
                    paths = ["."]
                area_meta[n.id] = {"id": n.id, "name": n.name, "paths": paths}

        # Collect cross-area invocations
        for u, v, edata in dep_graph.edges(data=True):
            et = edata.get("type")
            if not _is_invokes(et):
                continue
            if exclude_tests and (_is_test_nid(u) or _is_test_nid(v)):
                continue

            src_areas = _dep_to_l1_areas(u)
            dst_areas = _dep_to_l1_areas(v)
            if not src_areas or not dst_areas:
                continue

            for sa in src_areas:
                for da in dst_areas:
                    if sa == da:
                        continue
                    key = (sa, da)
                    entry = pair_stats[key]
                    entry["count"] += 1
                    if len(entry["examples"]) < max_examples_per_pair:
                        entry["examples"].append({"caller": _pretty_dep_entity(u), "callee": _pretty_dep_entity(v)})
                        example_participants_by_pair[key].add(u)
                        example_participants_by_pair[key].add(v)

        # Build cross-area calls summary
        edges_out = []
        for (sa, da), info in sorted(pair_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
            src_name = area_meta.get(sa, {}).get("name", sa)
            dst_name = area_meta.get(da, {}).get("name", da)
            edges_out.append({
                "src_area_id": sa,
                "dst_area_id": da,
                "src_area": src_name,
                "dst_area": dst_name,
                "count": info["count"],
                "examples": info["examples"],
            })

        # Build textual summary of cross-area calls
        lines = ["# Cross-area function calls (L1 → L1)"]
        if not edges_out:
            lines.append("(no cross-area invokes found)")
        else:
            for e in edges_out:
                lines.append(f"- {e['src_area']} → {e['dst_area']} (calls: {e['count']})")
                for ex in e["examples"]:
                    lines.append(f"    · {ex['caller']} → {ex['callee']}")
        summary_text = "\n".join(lines)

        context_parts = [summary_text]

        # Add code context if requested
        if include_code and entity_searcher is not None:
            code_text = self._build_code_context(
                edges_out=edges_out,
                example_participants_by_pair=example_participants_by_pair,
                pair_stats=pair_stats,
                dep_graph=dep_graph,
                entity_searcher=entity_searcher,
                code_types=code_types,
                exclude_tests=exclude_tests,
                code_from_examples_only=code_from_examples_only,
                keep_docstring=keep_docstring
            )
            if code_text:
                context_parts.append("\n# Relevant Code Snippets\n" + code_text)

        return "\n".join(context_parts)

    def _build_code_context(
        self,
        edges_out: List[Dict],
        example_participants_by_pair: Dict[Tuple[str, str], set],
        pair_stats: Dict[Tuple[str, str], Dict],
        dep_graph: MultiDiGraph,
        entity_searcher,
        code_types: Tuple[NodeType, ...],
        exclude_tests: bool,
        code_from_examples_only: bool,
        keep_docstring: bool
    ) -> str:
        """Build code context from cross-area participants"""
        
        def _is_test_nid(nid: str) -> bool:
            file_part = nid.split(":", 1)[0]
            return is_test_file(file_part)

        # Collect participants
        if code_from_examples_only:
            participants: set = set()
            pair_order = [(e["src_area_id"], e["dst_area_id"]) for e in edges_out]
            for key in pair_order:
                participants |= example_participants_by_pair.get(key, set())
        else:
            participants = set()
            for (sa, da), info in pair_stats.items():
                for ex in info["examples"]:
                    participants.add(ex["caller"])
                    participants.add(ex["callee"])

        # Filter and sort participants
        kept: List[str] = []
        for nid in sorted(participants, key=lambda x: (x.split(":", 1)[0], x)):
            if exclude_tests and _is_test_nid(nid):
                continue
            ntype = dep_graph.nodes.get(nid, {}).get("type")
            if ntype not in set(code_types):
                continue
            kept.append(nid)

        if not kept:
            return ""

        try:
            # Get code blocks
            code_blocks = entity_searcher.get_node_data(
                kept,
                return_code_content=True,
                wrap_with_ln=False,
            )

            bundle_lines: List[str] = []
            for item in code_blocks:
                nid = item.get("node_id")
                ntype = item.get("type")
                code = item.get("code_content", "")
                feature_paths = item.get("feature_paths", []) or []
                if feature_paths:
                    feature_paths = [feature_paths[0]]

                header = f"===== {nid} ({ntype.value if isinstance(ntype, NodeType) else str(ntype)}) ====="
                bundle_lines.append(header)
                if feature_paths:
                    bundle_lines.append(f"# feature_paths: {', '.join(feature_paths)}")
                if code:
                    skeleton_code = get_skeleton(
                        raw_code=code,
                        keep_indent=True,
                        keep_constant=False,
                        keep_docstring=keep_docstring,
                        compress_assign=True
                    )
                    bundle_lines.append(skeleton_code)
                bundle_lines.append("")

            return "\n".join(bundle_lines)

        except Exception as e:
            self.logger.warning(f"Failed to build code context: {e}")
            return ""