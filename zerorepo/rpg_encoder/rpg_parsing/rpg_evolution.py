"""
RPG Evolution Module

Handles repository-level changes (added, deleted, modified files)
and updates both the feature tree and the RPG graph incrementally.
Supports branch switching for major modifications and maintains
dependency graph index consistency.
"""
import time
import json
import logging
from typing import Dict, List, Optional

from zerorepo.rpg_gen.base import (
    RPG, NodeType,
    RepoSkeleton, FileNode,
    LLMConfig,
    ParsedFile, CodeSnippetBuilder
)
from zerorepo.rpg_gen.base.node import filter_non_test_py_files
from zerorepo.utils.repo import (
    load_skeleton_from_repo
)
from zerorepo.utils.diff import generate_detailed_diff
from .semantic_parsing import ParseFeatures
from .refactor_tree import RefactorTree
from .rpg_encoding import RPGParser


class RPGEvolution:
    """
    Handles repository-level changes (added, deleted, modified files)
    and updates both the feature tree and the RPG graph incrementally.

    Key features:
    - Supports branch switching for major modifications
    - Updates dep_graph and RPG node index after each diff
    - Provides detailed logging and statistics
    """

    @staticmethod
    def _log_stage_summary(stage: str, stats: Dict, start_time: float, logger: logging.Logger):
        elapsed = time.time() - start_time
        logger.info(
            f"\n{'=' * 60}\n"
            f"📊 {stage.upper()} SUMMARY\n"
            f"{'-' * 60}\n"
            f"⏱️  Time taken: {elapsed:.2f} sec\n"
            + "\n".join([f"🔹 {k}: {v}" for k, v in stats.items()])
            + f"\n{'=' * 60}\n"
        )

    @staticmethod
    def _assign_feature_paths_to_skeleton(rpg: RPG, skeleton: RepoSkeleton, logger: logging.Logger) -> RepoSkeleton:
        """
        Assign feature paths to file nodes in the skeleton based on RPG structure.

        For each file node in the RPG, this method collects all child nodes' feature paths
        and assigns them to the corresponding file node in the skeleton.
        """
        logger.info("Assigning feature paths to skeleton file nodes...")

        file_to_feature_paths = {}

        for node_id, node in rpg.nodes.items():
            if node.meta and node.meta.type_name and node.meta.type_name == NodeType.FILE:
                if node.meta.path:
                    file_path = node.meta.path
                    if isinstance(file_path, list):
                        file_path = file_path[0] if file_path else None

                    if file_path:
                        if file_path not in file_to_feature_paths:
                            file_to_feature_paths[file_path] = []

                        child_nodes = node.children(recursive=True)

                        if child_nodes:
                            for child_node in child_nodes:
                                child_feature_path = child_node.feature_path()
                                file_to_feature_paths[file_path].append(child_feature_path)
                        else:
                            file_feature_path = node.feature_path()
                            file_to_feature_paths[file_path].append(file_feature_path)

        updated_count = 0
        for file_path, feature_paths in file_to_feature_paths.items():
            file_node = skeleton.find_file(path=file_path)
            if file_node and isinstance(file_node, FileNode):
                unique_feature_paths = list(set(feature_paths))
                file_node.feature_paths = unique_feature_paths
                updated_count += 1
                logger.debug(f"Assigned feature paths to {file_path}: {unique_feature_paths}")

        logger.info(f"Successfully assigned feature paths to {updated_count} skeleton file nodes")

        return skeleton

    @staticmethod
    def _update_dep_graph_index(
        rpg: RPG,
        repo_dir: str,
        logger: logging.Logger
    ) -> None:
        """
        Update the dependency graph and rebuild RPG node index.

        This should be called after any modification to ensure
        dep_graph and RPG node mappings are in sync.
        """
        logger.info("Updating dependency graph and RPG node index...")

        try:
            # Parse or rebuild dependency graph
            rpg.parse_dep_graph(repo_dir)

            dep_count = len(rpg.dep_graph.G.nodes()) if rpg.dep_graph else 0
            map_count = len(rpg._dep_to_rpg_map) if rpg._dep_to_rpg_map else 0

            logger.info(
                f"Dependency graph updated: {dep_count} dep nodes, "
                f"{map_count} dep-to-rpg mappings"
            )
        except Exception as e:
            logger.warning(f"Failed to update dependency graph: {e}")

    @staticmethod
    def _process_add_files(
        ctx: Dict,
        new_files: List[str],
        logger: logging.Logger,
        llm_config: Optional[LLMConfig] = None
    ) -> Dict:
        """Process newly added files by parsing and refactoring them into the RPG."""
        start_time = time.time()
        logger.info(f"Processing {len(new_files)} new files...")

        cur_repo_skeleton, skeleton_info, _ = load_skeleton_from_repo(
            ctx["cur_repo_dir"], filter_func=filter_non_test_py_files
        )

        feature_parser = ParseFeatures(
            repo_dir=ctx["cur_repo_dir"],
            repo_info=ctx["repo_info"],
            repo_skeleton=cur_repo_skeleton,
            skeleton_info=skeleton_info,
            valid_files=[],
            repo_name=ctx["repo_name"],
            logger=logger,
            llm_config=llm_config
        )

        file_code_map = {}
        for file in new_files:
            if not file.endswith(".py"):
                continue
            file_node = cur_repo_skeleton.find_file(path=file)
            if not file_node:
                logger.warning(f"File not found in skeleton: {file}")
                continue
            file_code_map[file] = file_node.code

        file2feature, _ = feature_parser.parse_partial_repo(
            file_code_map=file_code_map,
            max_iterations=5,
            min_batch_tokens=10_000,
            max_batch_tokens=50_000,
            summary_min_batch_tokens=10_000,
            summary_max_batch_tokens=50_000,
            class_context_window=10,
            func_context_window=3,
            max_workers=8,
        )

        functional_areas = ctx["last_rpg"].get_functional_areas()
        cur_feature_tree, _, cur_rpg = RefactorTree.refactor_new_files(
            parsed_tree=file2feature,
            existing_feature_tree=ctx["last_feature_tree"],
            existing_rpg=ctx["last_rpg"],
            repo_dir=ctx["cur_repo_dir"],
            repo_name=ctx["repo_name"],
            repo_info=ctx["repo_info"],
            repo_skeleton=cur_repo_skeleton,
            skeleton_info=skeleton_info,
            functional_areas=functional_areas,
            context_window=5,
            max_iters=10,
            logger=logger,
            llm_config=llm_config
        )

        stats = {"added_files": len(new_files), "added_nodes": len(file2feature)}
        RPGEvolution._log_stage_summary("ADD FILES", stats, start_time, logger)
        return {"feature_tree": cur_feature_tree, "rpg": cur_rpg, "summary": stats}

    @staticmethod
    def _process_delete_files(
        ctx: Dict,
        deleted_files: List[str],
        logger: logging.Logger
    ) -> Dict:
        """Process deleted files by removing them from the RPG."""
        start_time = time.time()
        logger.info(f"🗑️ Processing deleted files: {deleted_files}")

        result = ctx["last_rpg"].delete_file_nodes(deleted_files)
        RPGEvolution._log_stage_summary("DELETE FILES", result, start_time, logger)
        return {"rpg": ctx["last_rpg"], "summary": result}

    @staticmethod
    def _process_modified_files(
        ctx: Dict,
        modified_result: Dict[str, Dict],
        logger: logging.Logger,
        llm_config: Optional[LLMConfig] = None
    ) -> Dict:
        """
        Process modified files using incremental update.

        Args:
            ctx: Context dictionary with repo info and last_rpg
            modified_result: Dict of file -> changes (added/deleted/changed units)
            logger: Logger instance
            llm_config: LLM configuration
        """
        start_time = time.time()
        logger.info(f"✏️ Processing modified files: {list(modified_result.keys())}")

        cur_repo_skeleton, skeleton_info, _ = load_skeleton_from_repo(
            ctx["cur_repo_dir"], filter_func=filter_non_test_py_files
        )

        # Build file code map for modified files
        cur_file_code_map = cur_repo_skeleton.get_file_code_map()
        cur_file_parsed = {f: ParsedFile(code=code, file_path=f) for f, code in cur_file_code_map.items()}
        code_builder = CodeSnippetBuilder(file_code_map=cur_file_code_map, parsed_files=cur_file_parsed)

        # Collect units to parse and deleted units info
        file2unit, deleted_unit_info = {}, {}
        for file, changes in modified_result.items():
            add_unit = changes.get("added", [])
            modified_unit = changes.get("changed", [])
            file2unit[file] = add_unit + modified_unit

            deleted_info = []
            for d_unit in changes.get("deleted", []):
                if d_unit.unit_type in ["class", "function"]:
                    deleted_info.append(d_unit.name)
                elif d_unit.unit_type == "method":
                    deleted_info.append(f"{d_unit.parent}.{d_unit.name}")
            deleted_unit_info[file] = deleted_info

        file2code = code_builder.build_file_map(grouped_units=file2unit)

        # Parse features for modified files
        feature_parser = ParseFeatures(
            repo_dir=ctx["cur_repo_dir"],
            repo_info=ctx["repo_info"],
            repo_skeleton=cur_repo_skeleton,
            skeleton_info=skeleton_info,
            valid_files=[],
            repo_name=ctx["repo_name"],
            logger=logger,
            llm_config=llm_config
        )

        file2feature, _ = feature_parser.parse_partial_repo(
            file_code_map=file2code,
            max_iterations=5,
            min_batch_tokens=10_000,
            max_batch_tokens=50_000,
            summary_min_batch_tokens=10_000,
            summary_max_batch_tokens=50_000,
            class_context_window=10,
            func_context_window=3,
            max_workers=8,
        )

        # Use incremental update
        repo_info_str = (
            f"Repository Name: {ctx['repo_name']}\n"
            f"Repository Info: {ctx['repo_info']}\n"
            f"Repository Skeleton: {skeleton_info}"
        )

        update_result = ctx["last_rpg"].update_from_parsed_tree(
            parsed_tree=file2feature,
            deleted_units=deleted_unit_info,
            repo_info=repo_info_str,
            file2unit={},
        )

        RPGEvolution._log_stage_summary("MODIFIED FILES", update_result, start_time, logger)
        return {"rpg": ctx["last_rpg"], "summary": update_result}

    @classmethod
    def process_diff(
        cls,
        repo_name: str,
        repo_info: str,
        save_path: str,
        last_repo_dir: str,
        cur_repo_dir: str,
        last_rpg: RPG,
        last_feature_tree: str,
        logger: logging.Logger = None,
        llm_config: Optional[LLMConfig] = None,
        update_dep_graph: bool = True,
    ) -> RPG:
        """
        Unified handler for repo diffs with logger support.

        Args:
            repo_name: Name of the repository
            repo_info: Repository description/info
            save_path: Path to save results
            last_repo_dir: Previous version repo directory
            cur_repo_dir: Current version repo directory
            last_rpg: Previous RPG instance
            last_feature_tree: Previous feature tree
            logger: Logger instance
            llm_config: LLM configuration
            update_dep_graph: Whether to update dep_graph after diff

        Returns:
            Updated RPG instance
        """
        # Ensure logger exists
        if logger is None:
            logger = logging.getLogger(f"{repo_name}_diff")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        global_start = time.time()
        logger.info("🧭 Starting process_diff...")

        last_excluded_files = last_rpg.excluded_files if last_rpg else []

        rpg_agent = RPGParser(
            repo_dir=cur_repo_dir,
            repo_name=repo_name,
            logger=logger,
            llm_config=llm_config
        )

        cur_exclude_files = rpg_agent.exclude_irrelvant_files(
            repo_info=repo_info,
            max_votes=3
        )

        all_exclude_files = list(sorted(set(last_excluded_files + cur_exclude_files)))
        logger.info("Excluded files for current repo: %d", len(all_exclude_files))

        all_diff = generate_detailed_diff(
            last_repo_dir=last_repo_dir,
            cur_repo_dir=cur_repo_dir,
            last_excluded_files=all_exclude_files,
        )

        ctx = {
            "repo_name": repo_name,
            "repo_info": repo_info,
            "cur_repo_dir": cur_repo_dir,
            "last_feature_tree": last_feature_tree,
            "last_rpg": last_rpg,
        }

        # Check if major regeneration is needed
        rpg_parser = RPGParser(
            repo_dir=cur_repo_dir,
            repo_name=repo_name,
            llm_config=llm_config
        )
        if rpg_parser.judge_regenereate_rpg():
            logger.info("♻️ Major changes detected, regenerating RPG...")
            result_rpg, _, _ = rpg_parser.parse_rpg_from_repo(
                max_repo_info_iters=3,
                max_exclude_votes=3,
                max_parse_iters=10,
                class_chunk_size=10,
                class_context_window=10,
                func_context_window=10,
                max_parse_workers=8,
                refactor_context_window=10,
                refactor_max_iters=10,
                save_path=save_path,
            )

            # Update dep_graph for regenerated RPG
            if update_dep_graph:
                cls._update_dep_graph_index(result_rpg, cur_repo_dir, logger)

            return result_rpg

        add_files = list(all_diff.get("added", {}).keys())
        add_files = [file for file in add_files if file.endswith(".py")]
        deleted_files = list(all_diff.get("deleted", {}).keys())
        deleted_files = [file for file in deleted_files if file.endswith(".py")]
        modified_result = {
            f: d for f, d in all_diff.get("modified", {}).items()
            if isinstance(d, dict) and any(d.get(k) for k in ("changed", "added", "deleted")) and f.endswith(".py")
        }

        if not add_files and not deleted_files and not modified_result:
            # No changes detected
            cur_skeleton, _, _ = load_skeleton_from_repo(
                repo_dir=cur_repo_dir,
                filter_func=filter_non_test_py_files
            )
            cls._assign_feature_paths_to_skeleton(last_rpg, cur_skeleton, logger)

            # Still update dep_graph to ensure consistency
            if update_dep_graph:
                cls._update_dep_graph_index(last_rpg, cur_repo_dir, logger)

            total_time = time.time() - global_start
            logger.info(
                f"\n✅ No changes detected for [{repo_name}]. "
                f"RPG remains unchanged.\n"
                f"⏱️  Elapsed time: {total_time:.2f}s\n"
                f"📦 Nodes: {len(last_rpg.nodes)} | Edges: {len(last_rpg.edges)}\n"
                f"{'=' * 60}\n"
            )
            return last_rpg

        # Process additions
        if add_files:
            add_result = cls._process_add_files(ctx, add_files, logger, llm_config)
            ctx["last_rpg"] = add_result["rpg"]
            ctx["last_feature_tree"] = add_result["feature_tree"]

        # Process deletions
        if deleted_files:
            del_result = cls._process_delete_files(ctx, deleted_files, logger)
            ctx["last_rpg"] = del_result["rpg"]

        # Process modifications
        if modified_result:
            mod_result = cls._process_modified_files(ctx, modified_result, logger, llm_config)
            ctx["last_rpg"] = mod_result["rpg"]

        # Assign feature paths to skeleton file nodes after all processing
        cur_skeleton, _, _ = load_skeleton_from_repo(
            repo_dir=cur_repo_dir,
            filter_func=filter_non_test_py_files
        )
        cls._assign_feature_paths_to_skeleton(ctx["last_rpg"], cur_skeleton, logger)

        # Update dependency graph and RPG node index
        if update_dep_graph:
            cls._update_dep_graph_index(ctx["last_rpg"], cur_repo_dir, logger)

        # Save results
        result = {
            "repo_name": repo_name,
            "repo_info": repo_info,
            "excluded_files": all_exclude_files,
            "rpg": {
                "structure": ctx["last_rpg"].to_dict(),
                "feature_tree": ctx["last_rpg"].get_functionality_graph(),
            },
            "diff_summary": {
                "added": len(add_files),
                "deleted": len(deleted_files),
                "modified": len(modified_result)
            }
        }

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

        total_time = time.time() - global_start
        logger.info(
            f"\n🏁 DIFF PROCESS COMPLETED for [{repo_name}] in {total_time:.2f}s\n"
            f"📂 Added: {len(add_files)} | 🗑️ Deleted: {len(deleted_files)} | ✏️ Modified: {len(modified_result)}\n"
            f"📦 RPG nodes: {len(ctx['last_rpg'].nodes)} | edges: {len(ctx['last_rpg'].edges)}\n"
            f"🔗 Dep-to-RPG mappings: {len(ctx['last_rpg']._dep_to_rpg_map) if ctx['last_rpg']._dep_to_rpg_map else 0}\n"
            f"{'=' * 60}\n"
        )
        return ctx["last_rpg"]
