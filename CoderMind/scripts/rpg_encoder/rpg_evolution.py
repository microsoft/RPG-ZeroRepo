"""RPG Evolution Module.

Handles repository-level changes (added, deleted, modified files)
and updates both the feature tree and the RPG graph incrementally.

Key workflow:
1. Compute a detailed diff between two repo versions
2. Process added files   -- parse features, refactor into RPG
3. Process deleted files -- remove corresponding RPG nodes
4. Process modified files -- incremental update + optional re-refactor
5. Update dependency graph index
6. Sync skeleton feature paths

Ported from RPG-ZeroRepo ``zerorepo/rpg_encoder/rpg_parsing/rpg_evolution.py``
with the following adaptations for CoderMind:
- No ``RepoSkeleton`` / ``FileNode`` dependency -- uses simplified skeleton
  loading via ``os.walk`` (same pattern as ``RPGParser``).
- Uses ``LLMClient`` from ``scripts.common.llm_client``
- Uses ``ParsedFile`` / ``CodeSnippetBuilder`` from ``scripts.skeleton.code_unit``
- Uses ``is_test_file`` from ``scripts.common.utils``
- Relies on new RPG methods: ``delete_file_nodes``, ``update_from_parsed_tree``,
  ``get_functionality_graph``, ``parse_dep_graph``
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from common.rpg_io import atomic_write_rpg
from common.utils import (
    is_skip_dir,
    exclude_files,
    filter_excluded_files,
    normalize_path,
)
from lang_parser import is_supported_source, is_test_file as is_supported_test_file
from rpg.code_unit import CodeSnippetBuilder, CodeUnit, ParsedFile
from rpg import NodeType, RPG

from .refactor_tree import RefactorTree
from .semantic_parsing import ParseFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


def _filter_non_test_py_files(path: str) -> bool:
    """Return True if *path* is a parseable, non-test source file.

    Used as a filter predicate when walking the repository directory.
    The function keeps its public name for API stability, but the
    predicate accepts any language supported by ``lang_parser``.
    """
    if not is_supported_source(path):
        return False
    return not is_supported_test_file(path)


def _load_skeleton_from_repo(
    repo_dir: str,
    filter_func=None,
) -> Tuple[str, Dict[str, str], List[str]]:
    """Walk *repo_dir* and build a text skeleton, file-code map, valid file list.

    This is a simplified replacement for ZeroRepo's
    ``load_skeleton_from_repo`` that does **not** depend on ``RepoSkeleton``
    / ``FileNode``.  It replicates the same essential outputs:

    Returns:
        (skeleton_info, file_code_map, valid_files)

        - ``skeleton_info``: newline-joined relative paths
        - ``file_code_map``: ``{rel_path: source_code}`` for filtered files
        - ``valid_files``: list of relative paths that pass the filter
    """
    if filter_func is None:
        filter_func = _filter_non_test_py_files

    tree_lines: List[str] = []
    file_code_map: Dict[str, str] = {}
    valid_files: List[str] = []

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not is_skip_dir(d)]
        dirs.sort()

        rel_root = os.path.relpath(root, repo_dir)
        if rel_root == ".":
            rel_root = ""

        for fname in sorted(files):
            if fname.startswith("."):
                continue
            rel_path = (
                os.path.join(rel_root, fname) if rel_root else fname
            )
            rel_path = rel_path.replace("\\", "/")

            if not filter_func(rel_path):
                continue

            tree_lines.append(rel_path)
            valid_files.append(rel_path)

            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
                    file_code_map[rel_path] = fh.read()
            except (OSError, UnicodeDecodeError):
                continue

    skeleton_info = "\n".join(tree_lines)
    return skeleton_info, file_code_map, sorted(valid_files)


def _filter_units(
    d: Union[List, Dict[str, List]],
) -> Dict[str, List]:
    """Keep only ``class``/``function``/``method`` CodeUnits.

    Ported from ZeroRepo ``zerorepo/utils/repo.py:filter_units``.
    """
    valid_types = {"class", "function", "method"}
    if isinstance(d, dict):
        return {
            f: [u for u in us if u.unit_type in valid_types]
            for f, us in d.items()
            if any(u.unit_type in valid_types for u in us)
        }
    elif isinstance(d, list):
        return [u for u in d if u.unit_type in valid_types]
    return d


def _calculate_diff(
    ori_file_units: List,
    new_file_units: List,
) -> Dict[str, List]:
    """Unit-level diff between two versions of a file.

    Ported from ZeroRepo ``zerorepo/utils/repo.py:calculate_diff``.
    """
    ori_units = _filter_units(ori_file_units) if ori_file_units else []
    new_units = _filter_units(new_file_units) if new_file_units else []

    ori_map = {u.key(): u for u in ori_units}
    new_map = {u.key(): u for u in new_units}

    changed, added, deleted = [], [], []

    for key, old_unit in ori_map.items():
        if key in new_map:
            new_unit = new_map[key]
            if old_unit.semantic_equals(new_unit):
                pass  # unchanged
            else:
                changed.append(new_unit)
        else:
            deleted.append(old_unit)

    for key, new_unit in new_map.items():
        if key not in ori_map:
            added.append(new_unit)

    return {"changed": changed, "added": added, "deleted": deleted}


def generate_detailed_diff(
    last_repo_dir: str,
    cur_repo_dir: str,
    last_excluded_files: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Compute a detailed, unit-level diff between two repo snapshots.

    For each file, determines whether it was added, deleted, or modified.
    For modified files, computes unit-level changes (added / deleted /
    changed CodeUnits).

    Ported from RPG-ZeroRepo ``zerorepo/utils/diff.py:generate_detailed_diff``
    with the following adaptations:
    - Replaces ``load_skeleton_from_repo`` with ``_load_skeleton_from_repo``
    - Uses ``_filter_non_test_py_files`` instead of ``filter_non_test_py_files``

    Args:
        last_repo_dir: Path to the previous version of the repository.
        cur_repo_dir: Path to the current version of the repository.
        last_excluded_files: Files already excluded from the previous RPG.

    Returns:
        ``{"added": {...}, "deleted": {...}, "modified": {...}}``
    """
    if last_excluded_files is None:
        last_excluded_files = []

    # Load both repo snapshots
    _, last_file_code_map, _ = _load_skeleton_from_repo(last_repo_dir)
    _, cur_file_code_map, _ = _load_skeleton_from_repo(cur_repo_dir)

    # Apply exclusions to current repo
    cur_exclude_files = exclude_files(files=list(cur_file_code_map.keys()))
    need_to_exclude = last_excluded_files + cur_exclude_files

    cur_filtered_files = filter_excluded_files(
        valid_files=list(cur_file_code_map.keys()),
        excluded_files=need_to_exclude,
    )
    cur_file_code_map = {
        f: code for f, code in cur_file_code_map.items()
        if f in cur_filtered_files
    }

    last_files = set(last_file_code_map.keys())
    cur_files = set(cur_file_code_map.keys())

    added_files = cur_files - last_files
    deleted_files = last_files - cur_files
    common_files = cur_files & last_files

    # Parse into CodeUnits
    def parse_units(file_map: Dict[str, str]) -> Dict[str, List]:
        result = {}
        for path, code in file_map.items():
            try:
                parsed = ParsedFile(code=code, file_path=path)
                result[path] = [
                    u for u in parsed.units
                    if u.unit_type in {"class", "function", "method"}
                ]
            except Exception:
                result[path] = []
        return result

    last_units_map = parse_units(last_file_code_map)
    cur_units_map = parse_units(cur_file_code_map)

    # Added files
    added = {
        f: cur_units_map[f]
        for f in added_files
        if f in cur_units_map
    }

    # Deleted files
    deleted = {
        f: last_units_map[f]
        for f in deleted_files
        if f in last_units_map
    }

    # Modified files — unit-level diff
    modified: Dict[str, Dict] = {}
    for f in common_files:
        last_f_units = last_units_map.get(f, [])
        cur_f_units = cur_units_map.get(f, [])
        modified[f] = _calculate_diff(
            ori_file_units=last_f_units,
            new_file_units=cur_f_units,
        )

    # Filter out non-code units
    added = _filter_units(added)
    deleted = _filter_units(deleted)

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
    }


# ---------------------------------------------------------------------------
# RPGEvolution
# ---------------------------------------------------------------------------


class RPGEvolution:
    """Handles repository-level changes and incrementally updates the RPG.

    Key features:
    - Supports incremental add/delete/modify of files
    - Updates dependency graph and RPG node index after each diff
    - Provides detailed logging and statistics

    Ported from RPG-ZeroRepo ``rpg_evolution.py`` :class:`RPGEvolution`,
    adapted for CoderMind infrastructure.
    """

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_stage_summary(
        stage: str,
        stats: Dict,
        start_time: float,
        logger: logging.Logger,
    ) -> None:
        elapsed = time.time() - start_time
        logger.info(
            "\n%s\n%s SUMMARY\n%s\nTime taken: %.2f sec\n%s\n%s\n",
            "=" * 60,
            stage.upper(),
            "-" * 60,
            elapsed,
            "\n".join(f"  {k}: {v}" for k, v in stats.items()),
            "=" * 60,
        )

    # ------------------------------------------------------------------
    # Dependency-graph update
    # ------------------------------------------------------------------

    @staticmethod
    def _update_dep_graph_index(
        rpg: RPG,
        repo_dir: str,
        logger: logging.Logger,
        save_path: Optional[str] = None,
    ) -> None:
        """Update the dependency graph and rebuild RPG node index.

        Routes through :class:`rpg.service.RPGService` so the in-memory
        dep_graph, dep-to-RPG mappings, and cross maps are refreshed
        together. When ``save_path`` is provided, a standalone dep_graph
        snapshot is also written for compatibility tooling.

        Args:
            rpg: The RPG to attach the rebuilt dep_graph to.
            repo_dir: Workspace root (which is also the project repo root
                after the workspace=repo unification).
            logger: Logger for status output.
            save_path: Optional legacy path where a standalone
                ``dep_graph.json`` should be written. If ``None``, the
                caller persists the refreshed graph by saving ``rpg.json``.
        """
        logger.info("Updating dependency graph and RPG node index...")
        try:
            from pathlib import Path as _Path
            from rpg.service import RPGService

            # Workspace root == project repo root; no subdirectory probing.
            code_dir = repo_dir

            svc = RPGService(rpg)
            # Anchor ``_rpg_dir`` to the dep_graph's parent dir so the
            # service can store ``_dep_graph_file`` as a clean relative
            # path.  Without this, ``RPGService(rpg)`` defaults
            # ``_rpg_dir`` to the cwd, which is almost never the dir
            # the caller wants the persisted reference relative to.
            if save_path:
                svc._rpg_dir = _Path(save_path).resolve().parent
            svc.refresh_dep_graph(
                code_dir=code_dir,
                workspace_root=repo_dir,
                save_path=save_path,
            )

            dep_count = (
                len(rpg.dep_graph.G.nodes()) if rpg.dep_graph else 0
            )
            map_count = len(rpg._dep_to_rpg_map)
            if save_path:
                logger.info(
                    "Dependency graph updated: %d dep nodes, %d dep-to-rpg "
                    "mappings, saved to %s",
                    dep_count, map_count, save_path,
                )
            else:
                # The new default: dep_graph rides inside rpg.json (single
                # source of truth).  No standalone dep_graph.json is
                # written from this call; the caller's ``svc.save(rpg)``
                # embeds the in-memory graph via ``RPG.to_dict``.
                logger.info(
                    "Dependency graph updated in-memory (%d nodes, "
                    "%d mappings); caller embeds into rpg.json on save.",
                    dep_count, map_count,
                )
        except Exception as e:
            logger.warning("Failed to update dependency graph: %s", e)

    # ------------------------------------------------------------------
    # Process: add files
    # ------------------------------------------------------------------

    @staticmethod
    def _process_add_files(
        ctx: Dict,
        new_files: List[str],
        logger: logging.Logger,
    ) -> Dict:
        """Parse new files and refactor them into the RPG.

        Source: ZeroRepo ``RPGEvolution._process_add_files``.
        """
        start_time = time.time()
        logger.info("Processing %d new files...", len(new_files))

        skeleton_info, file_code_map_all, _ = _load_skeleton_from_repo(
            ctx["cur_repo_dir"]
        )

        feature_parser = ParseFeatures(
            repo_dir=ctx["cur_repo_dir"],
            repo_info=ctx["repo_info"],
            repo_skeleton=skeleton_info,
            valid_files=[],
            repo_name=ctx["repo_name"],
            logger=logger,
        )

        # Build code map for new files only
        file_code_map: Dict[str, str] = {}
        for fpath in new_files:
            if not is_supported_source(fpath) or is_supported_test_file(fpath):
                continue
            if fpath in file_code_map_all:
                file_code_map[fpath] = file_code_map_all[fpath]
            else:
                full = os.path.join(ctx["cur_repo_dir"], fpath)
                if os.path.isfile(full):
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            file_code_map[fpath] = f.read()
                    except Exception:
                        logger.warning("Cannot read file: %s", fpath)

        file2feature, _ = feature_parser.parse_partial_repo(
            file_code_map=file_code_map,
            max_iterations=5,
            min_batch_tokens=10_000,
            max_batch_tokens=50_000,
            summary_min_batch_tokens=10_000,
            summary_max_batch_tokens=50_000,
            class_context_window=10,
            func_context_window=3,
            max_workers=1,
        )

        functional_areas = ctx["last_rpg"].get_functional_areas()
        cur_feature_tree, _, cur_rpg = RefactorTree.refactor_new_files(
            parsed_tree=file2feature,
            existing_feature_tree=ctx["last_feature_tree"],
            existing_rpg=ctx["last_rpg"],
            repo_dir=ctx["cur_repo_dir"],
            repo_name=ctx["repo_name"],
            repo_info=ctx["repo_info"],
            repo_skeleton=skeleton_info,
            skeleton_info=skeleton_info,
            functional_areas=functional_areas,
            context_window=5,
            max_iters=10,
            logger=logger,
        )

        stats = {"added_files": len(new_files), "added_nodes": len(file2feature)}
        RPGEvolution._log_stage_summary("ADD FILES", stats, start_time, logger)
        return {"feature_tree": cur_feature_tree, "rpg": cur_rpg, "summary": stats}

    # ------------------------------------------------------------------
    # Process: delete files
    # ------------------------------------------------------------------

    @staticmethod
    def _process_delete_files(
        ctx: Dict,
        deleted_files: List[str],
        logger: logging.Logger,
    ) -> Dict:
        """Remove deleted files from the RPG.

        Source: ZeroRepo ``RPGEvolution._process_delete_files``.
        """
        start_time = time.time()
        logger.info("Processing deleted files: %s", deleted_files)

        result = ctx["last_rpg"].delete_file_nodes(deleted_files)
        RPGEvolution._log_stage_summary("DELETE FILES", result, start_time, logger)
        return {"rpg": ctx["last_rpg"], "summary": result}

    # ------------------------------------------------------------------
    # Process: modified files
    # ------------------------------------------------------------------

    @staticmethod
    def _process_modified_files(
        ctx: Dict,
        modified_result: Dict[str, Dict],
        logger: logging.Logger,
    ) -> Dict:
        """Process modified files using incremental update.

        Source: ZeroRepo ``RPGEvolution._process_modified_files``.
        """
        start_time = time.time()
        logger.info("Processing modified files: %s", list(modified_result.keys()))

        skeleton_info, file_code_map_all, _ = _load_skeleton_from_repo(
            ctx["cur_repo_dir"]
        )

        # Build parsed-file map and snippet builder
        cur_file_parsed = {
            f: ParsedFile(code=code, file_path=f)
            for f, code in file_code_map_all.items()
        }
        code_builder = CodeSnippetBuilder(
            file_code_map=file_code_map_all,
            parsed_files=cur_file_parsed,
        )

        # Collect units to parse and deleted units info
        file2unit: Dict[str, List] = {}
        deleted_unit_info: Dict[str, List[str]] = {}
        for fpath, changes in modified_result.items():
            add_unit = changes.get("added", [])
            modified_unit = changes.get("changed", [])
            file2unit[fpath] = add_unit + modified_unit

            deleted_info: List[str] = []
            for d_unit in changes.get("deleted", []):
                if d_unit.unit_type in ("class", "function"):
                    deleted_info.append(d_unit.name)
                elif d_unit.unit_type == "method":
                    deleted_info.append(f"{d_unit.parent}.{d_unit.name}")
            deleted_unit_info[fpath] = deleted_info

        file2code = code_builder.build_file_map(grouped_units=file2unit)

        # Parse features for modified files
        feature_parser = ParseFeatures(
            repo_dir=ctx["cur_repo_dir"],
            repo_info=ctx["repo_info"],
            repo_skeleton=skeleton_info,
            valid_files=[],
            repo_name=ctx["repo_name"],
            logger=logger,
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
            max_workers=1,
        )

        # Incremental update
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

        # Only refactor files whose L4 name (_file_summary_) has changed
        files_to_refactor: Dict[str, Dict] = {}
        for file_path, features in file2feature.items():
            new_summary = features.get(
                "_file_summary_",
                os.path.basename(file_path).replace(".py", ""),
            )
            for _nid, node in ctx["last_rpg"].nodes.items():
                if node.meta and node.meta.type_name == NodeType.FILE:
                    fp = node.meta.path
                    if isinstance(fp, list):
                        fp = fp[0] if fp else None
                    if fp == file_path:
                        if node.name != new_summary:
                            logger.info(
                                "File summary changed: %s: '%s' -> '%s'",
                                file_path, node.name, new_summary,
                            )
                            files_to_refactor[file_path] = features
                        else:
                            logger.info(
                                "File summary unchanged, skip refactor: %s",
                                file_path,
                            )
                        break

        if files_to_refactor:
            functional_areas = ctx["last_rpg"].get_functional_areas()
            cur_feature_tree, _, cur_rpg = RefactorTree.refactor_modified_files(
                parsed_tree=files_to_refactor,
                existing_feature_tree=ctx.get("last_feature_tree", []),
                existing_rpg=ctx["last_rpg"],
                repo_dir=ctx["cur_repo_dir"],
                repo_name=ctx["repo_name"],
                repo_info=ctx["repo_info"],
                repo_skeleton=skeleton_info,
                skeleton_info=skeleton_info,
                functional_areas=functional_areas,
                context_window=5,
                max_iters=10,
                logger=logger,
            )
            RPGEvolution._log_stage_summary(
                "MODIFIED FILES", update_result, start_time, logger
            )
            return {
                "rpg": cur_rpg,
                "feature_tree": cur_feature_tree,
                "summary": update_result,
            }
        else:
            logger.info("No file summaries changed, skipping refactoring step")
            RPGEvolution._log_stage_summary(
                "MODIFIED FILES", update_result, start_time, logger
            )
            return {"rpg": ctx["last_rpg"], "summary": update_result}

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

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
        logger: Optional[logging.Logger] = None,
        update_dep_graph: bool = True,
        dep_graph_save_path: Optional[str] = None,
        max_exclude_votes: int = 1,
    ) -> RPG:
        """Unified handler for repo diffs with incremental RPG updates.

        This is the main entry-point for M8 -- RPG Evolution.  It mirrors
        the ``RPGEvolution.process_diff`` classmethod in RPG-ZeroRepo.

        Args:
            dep_graph_save_path: When ``update_dep_graph=True``, the
                refreshed dep_graph is written here.  Required so the
                standalone ``dep_graph.json`` doesn't drift from the
                embedded copy inside ``rpg.json``.  If ``None``, the
                dep_graph is only updated in memory and ``rpg.json`` is
                relied upon to carry it forward — that's the legacy
                behaviour and will log a warning.

        Pipeline:
        1. Carry forward deterministic exclusions (no per-commit LLM vote)
        2. Compute detailed diff (``generate_detailed_diff``)
        3. Process additions / deletions / modifications
        4. Update dependency graph index
        5. Save results (optional)

        Args:
            repo_name: Name of the repository.
            repo_info: Repository description / overview.
            save_path: Path to save JSON results (empty = no save).
            last_repo_dir: Previous version repo directory.
            cur_repo_dir: Current version repo directory.
            last_rpg: Previous RPG instance.
            last_feature_tree: Previous feature tree (string or list).
            logger: Logger instance (auto-created if None).
            update_dep_graph: Whether to update dep_graph after diff.

        Returns:
            Updated ``RPG`` instance.

        Source: ZeroRepo ``RPGEvolution.process_diff`` (:345).
        """
        # Ensure logger exists
        if logger is None:
            logger = logging.getLogger(f"{repo_name}_diff")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        global_start = time.time()
        logger.info("Starting process_diff...")

        last_excluded_files = last_rpg.excluded_files if last_rpg else []

        # Incremental updates run on every commit (post-commit hook). Exclusion
        # is fully deterministic here — no per-commit LLM vote (it cost ~30
        # round-trips per generation and only re-derived paths the rules below
        # already cover). ``generate_detailed_diff`` loads both snapshots via
        # ``_load_skeleton_from_repo`` (which prunes skip-dirs and keeps only
        # supported, non-test source) and re-applies the ``exclude_files``
        # prefix rules itself, so the only thing to carry forward is the
        # encode-time exclusion list — which may include LLM-identified
        # vendored / third-party paths the prefix rules can't infer.
        all_exclude_files = sorted(set(last_excluded_files))
        logger.info(
            "Carrying forward %d excluded path(s) from the encode baseline.",
            len(all_exclude_files),
        )

        # Compute detailed diff (re-applies deterministic exclusion internally)
        all_diff = generate_detailed_diff(
            last_repo_dir=last_repo_dir,
            cur_repo_dir=cur_repo_dir,
            last_excluded_files=all_exclude_files,
        )

        ctx: Dict[str, Any] = {
            "repo_name": repo_name,
            "repo_info": repo_info,
            "cur_repo_dir": cur_repo_dir,
            "last_feature_tree": last_feature_tree,
            "last_rpg": last_rpg,
        }

        # Filter to supported source files (any language registered with lang_parser),
        # excluding tests so the encoder doesn't index test code as features.
        add_files = [
            f for f in all_diff.get("added", {}).keys()
            if is_supported_source(f) and not is_supported_test_file(f)
        ]
        deleted_files = [
            f for f in all_diff.get("deleted", {}).keys()
            if is_supported_source(f) and not is_supported_test_file(f)
        ]
        modified_result = {
            f: d
            for f, d in all_diff.get("modified", {}).items()
            if (
                isinstance(d, dict)
                and any(d.get(k) for k in ("changed", "added", "deleted"))
                and is_supported_source(f)
                and not is_supported_test_file(f)
            )
        }

        if not add_files and not deleted_files and not modified_result:
            # No changes detected
            if update_dep_graph:
                cls._update_dep_graph_index(
                    last_rpg, cur_repo_dir, logger,
                    save_path=dep_graph_save_path,
                )

            total_time = time.time() - global_start
            logger.info(
                "\nNo changes detected for [%s]. RPG remains unchanged.\n"
                "Elapsed time: %.2fs\nNodes: %d | Edges: %d\n%s",
                repo_name, total_time,
                len(last_rpg.nodes), len(last_rpg.edges),
                "=" * 60,
            )
            return last_rpg

        # Process additions
        if add_files:
            add_result = cls._process_add_files(ctx, add_files, logger)
            ctx["last_rpg"] = add_result["rpg"]
            ctx["last_feature_tree"] = add_result["feature_tree"]

        # Process deletions
        if deleted_files:
            del_result = cls._process_delete_files(ctx, deleted_files, logger)
            ctx["last_rpg"] = del_result["rpg"]

        # Process modifications
        if modified_result:
            mod_result = cls._process_modified_files(
                ctx, modified_result, logger
            )
            ctx["last_rpg"] = mod_result["rpg"]
            if "feature_tree" in mod_result:
                ctx["last_feature_tree"] = mod_result["feature_tree"]

        # Update dependency graph
        if update_dep_graph:
            cls._update_dep_graph_index(
                ctx["last_rpg"], cur_repo_dir, logger,
                save_path=dep_graph_save_path,
            )

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
                "modified": len(modified_result),
            },
        }

        if save_path:
            # Atomic write: ``result`` embeds ``rpg.to_dict()``; a killed
            # diff job used to leave a half-truncated artefact that
            # downstream consumers (``cmind diff``, debug tools) would
            # fail to parse on the next read.
            atomic_write_rpg(save_path, result, indent=4)

        total_time = time.time() - global_start
        logger.info(
            "\nDIFF PROCESS COMPLETED for [%s] in %.2fs\n"
            "Added: %d | Deleted: %d | Modified: %d\n"
            "RPG nodes: %d | edges: %d\n"
            "Dep-to-RPG mappings: %d\n%s",
            repo_name, total_time,
            len(add_files), len(deleted_files), len(modified_result),
            len(ctx["last_rpg"].nodes), len(ctx["last_rpg"].edges),
            len(ctx["last_rpg"]._dep_to_rpg_map),
            "=" * 60,
        )
        return ctx["last_rpg"]
