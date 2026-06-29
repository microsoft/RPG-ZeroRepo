#!/usr/bin/env python3
r"""Run Update RPG Script.

Incremental RPG update: calls RPGEvolution.process_diff() to apply
repository changes to an existing RPG graph.

Prints a single JSON result to stdout with status and diff statistics.

Usage:
    cmind script rpg_encoder/run_update_rpg.py --json \\
        --rpg-file .cmind/data/rpg.json --last-repo-dir ./old-version
"""

import json
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure scripts/ is importable
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from common.paths import (  # noqa: E402
    RPG_FILE,
    DEP_GRAPH_FILE,
    WORKSPACE_ROOT,
)
from common.rpg_io import atomic_write_rpg  # noqa: E402


def _count_serialized_items(value: Any) -> int:
    return len(value) if isinstance(value, (list, dict)) else 0


def _serialized_feature_payload(data: dict[str, Any]) -> dict[str, Any]:
    rpg_data = data.get("rpg")
    if isinstance(rpg_data, dict) and isinstance(rpg_data.get("structure"), dict):
        return rpg_data["structure"]
    return data


def _serialized_feature_edges(data: dict[str, Any]) -> int:
    edges = _serialized_feature_payload(data).get("edges", [])
    if isinstance(edges, list):
        # Flat-format RPGs may include hierarchy edges; exclude them so counts
        # match the edges actually persisted by RPG.to_dict().
        return sum(
            1
            for e in edges
            if not (
                isinstance(e, dict)
                and str(e.get("relation", "")).lower()
                in ("contains", "composes", "contains_base_class")
            )
        )
    return _count_serialized_items(edges)


def _serialized_dep_graph(data: dict[str, Any]) -> dict[str, Any]:
    dep_graph = data.get("dep_graph")
    if isinstance(dep_graph, dict):
        return dep_graph
    payload = _serialized_feature_payload(data)
    dep_graph = payload.get("dep_graph")
    return dep_graph if isinstance(dep_graph, dict) else {}


def _serialized_dep_stats(data: dict[str, Any]) -> dict[str, int]:
    dep_graph = _serialized_dep_graph(data)
    return {
        "nodes": _count_serialized_items(dep_graph.get("nodes", [])),
        "edges": _count_serialized_items(dep_graph.get("edges", [])),
    }


def _serialized_dep_to_rpg_map_size(data: dict[str, Any]) -> int:
    dep_to_rpg_map = data.get("_dep_to_rpg_map")
    if isinstance(dep_to_rpg_map, dict):
        return len(dep_to_rpg_map)

    nodes = _serialized_dep_graph(data).get("nodes", {})
    if isinstance(nodes, dict):
        return sum(
            1
            for attrs in nodes.values()
            if isinstance(attrs, dict) and attrs.get("rpg_nodes")
        )
    if isinstance(nodes, list):
        return sum(
            1
            for attrs in nodes
            if isinstance(attrs, dict) and attrs.get("rpg_nodes")
        )
    return 0


def run_update_rpg(
    rpg_file: str,
    last_repo_dir: str,
    cur_repo_dir: str | None = None,
    output: str | None = None,
    dep_graph_path: str | None = None,
    max_exclude_votes: int = 1,
) -> dict:
    """Run incremental RPG update and return result dict.

    Pipeline:
      1. Load existing RPG from ``rpg_file``.
      2. Run :class:`RPGEvolution.process_diff` (LLM-driven feature
         tree refactor + structural dep_graph refresh).
      3. **Align meta.path** via ``RPGService.enrich_from_code`` so the
         LLM-generated paths actually match dep_graph node IDs.
      4. **Advance ``meta.git``** to the current workspace HEAD so the
         next pre-commit hook can take an incremental shortcut from
         this baseline.
    """
    # ``cur_repo_dir`` defaults to ``WORKSPACE_ROOT`` — the directory
    # the user ran ``cmind init --here`` in (their existing source
    # repo).  Pass an explicit path to override.
    if cur_repo_dir is None:
        cur_repo_dir = str(WORKSPACE_ROOT)
    cur_repo_dir = os.path.abspath(cur_repo_dir)
    last_repo_dir = os.path.abspath(last_repo_dir)
    rpg_file = os.path.abspath(rpg_file)
    # ``dep_graph_path`` is a legacy standalone location retained for
    # callers that still pass ``--dep-graph``. Normal updates embed the
    # refreshed dependency graph in ``rpg.json``.
    if dep_graph_path is None:
        dep_graph_path = str(DEP_GRAPH_FILE)
    else:
        dep_graph_path = os.path.abspath(dep_graph_path)

    if not os.path.isfile(rpg_file):
        return {"status": "error", "error": f"RPG file not found: {rpg_file}"}

    if not os.path.isdir(cur_repo_dir):
        return {"status": "error", "error": f"Current repo directory not found: {cur_repo_dir}"}

    if not os.path.isdir(last_repo_dir):
        return {"status": "error", "error": f"Previous repo directory not found: {last_repo_dir}"}

    if output is None:
        output = rpg_file

    try:
        from rpg import RPG
        from rpg.service import RPGService
        from rpg_encoder.rpg_evolution import RPGEvolution
        from common.git_utils import read_head

        # Load existing RPG
        with open(rpg_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        repo_name = data.get("repo_name", "unknown")
        repo_info = data.get("repo_info", "")

        # Parse RPG from saved data -- handle tree, flat, and nested formats
        rpg_data = data.get("rpg", {})
        if isinstance(rpg_data, dict) and "structure" in rpg_data:
            rpg = RPG.from_dict(rpg_data["structure"])
            feature_tree = rpg_data.get("feature_tree", [])
        elif "root" in data or "nodes" in data:
            rpg = RPG.from_dict(data)
            feature_tree = rpg.get_functionality_graph() if rpg else []
        else:
            return {"status": "error", "error": "Invalid RPG file format."}

        rpg.repo_info = repo_info
        rpg.excluded_files = data.get("excluded_files", [])

        # Record pre-update stats + previous git meta so we can report
        # how the sync baseline advanced.
        pre_nodes = len(rpg.nodes)
        pre_edges = _serialized_feature_edges(data)
        pre_dep_stats = _serialized_dep_stats(data)
        pre_commit = (rpg.git_meta or {}).get("head_commit")

        # === Step 1: LLM-driven feature graph refactor ===
        # ``dep_graph_save_path=None``: the dep_graph rides inside
        # ``rpg.json`` as the single source of truth (embedded by
        # ``RPG.to_dict`` and persisted by the ``atomic_write_rpg`` below).
        # The legacy standalone ``dep_graph.json`` is no longer produced;
        # readers tolerate its absence and use the embedded copy.
        updated_rpg = RPGEvolution.process_diff(
            repo_name=repo_name,
            repo_info=repo_info,
            save_path="",  # Don't save inside process_diff; we save below in unified format
            last_repo_dir=last_repo_dir,
            cur_repo_dir=cur_repo_dir,
            last_rpg=rpg,
            last_feature_tree=feature_tree,
            update_dep_graph=True,
            max_exclude_votes=max_exclude_votes,
        )

        # === Step 2: Align meta.path on freshly-added feature nodes ===
        # process_diff generates feature nodes via LLM; the LLM emits
        # paths that may not exactly match dep_graph node IDs (prefix
        # differences, ``::`` separators).  enrich_from_code(align_only=True)
        # walks the feature tree and snaps each node's meta.path to a
        # real dep_graph node where possible.  Skipping this step is
        # the difference between "feature nodes are present" and
        # "feature nodes are queryable via the rpg-tools MCP server".
        enrich_stats: dict = {}
        try:
            svc = RPGService(updated_rpg)
            # _rpg_dir is needed by enrich_from_code for relative path math.
            svc._rpg_dir = Path(rpg_file).parent.resolve()
            enrich_stats = svc.enrich_from_code(
                code_dir=cur_repo_dir,
                align_only=True,
                skip_dep_rebuild=True,  # dep_graph already fresh from process_diff
            )
        except Exception as exc:
            logger.warning("enrich_from_code(align_only=True) failed: %s", exc)

        # === Step 3: Advance meta.git to the current workspace HEAD ===
        # The pre-commit hook reads this on the next commit and takes
        # an incremental shortcut starting from this commit.  Skipping
        # this step would force every subsequent commit-hook run back
        # to a full rebuild (rebase / diverged path).
        meta_git_advanced = False
        try:
            ws_root = WORKSPACE_ROOT
            current = read_head(ws_root)
            if current:
                updated_rpg.set_git_meta(
                    head_commit=current["head_commit"],
                    head_short=current["head_short"],
                    head_branch=current["head_branch"],
                    head_timestamp=current["head_timestamp"],
                )
                meta_git_advanced = True
        except Exception as exc:
            logger.warning("set_git_meta after update_rpg failed: %s", exc)

        # Save updated RPG in the same format as run_encode (rpg.to_dict()).
        # Atomic write: a kill mid-update used to leave a half-truncated
        # rpg.json that bricked every subsequent ``cmind`` invocation;
        # ``atomic_write_rpg`` swaps a fully-written ``<output>.tmp`` into
        # place so readers always see either the previous good rpg.json
        # or the new one.
        result_data = updated_rpg.to_dict()
        atomic_write_rpg(output, result_data, indent=2, ensure_ascii=False)

        # Collect stats
        post_nodes = len(updated_rpg.nodes)
        post_edges = _serialized_feature_edges(result_data)
        post_dep_stats = _serialized_dep_stats(result_data)

        stats = {
            "repo_name": repo_name,
            "output_path": output,
            "dep_graph_path": dep_graph_path,
            "node_count": post_nodes,
            "edge_count": post_edges,
            "nodes_delta": post_nodes - pre_nodes,
            "edges_delta": post_edges - pre_edges,
            "dep_nodes": post_dep_stats["nodes"],
            "dep_edges": post_dep_stats["edges"],
            "dep_nodes_delta": post_dep_stats["nodes"] - pre_dep_stats["nodes"],
            "dep_edges_delta": post_dep_stats["edges"] - pre_dep_stats["edges"],
            "dep_to_rpg_map_size": _serialized_dep_to_rpg_map_size(result_data),
            "aligned": enrich_stats.get("aligned", 0),
            "groups_pathed": enrich_stats.get("groups_pathed", 0),
            "l1_pathed": enrich_stats.get("l1_pathed", 0),
            "meta_git_advanced": meta_git_advanced,
            "previous_commit": pre_commit,
            "new_commit": (updated_rpg.git_meta or {}).get("head_commit"),
        }
        try:
            stats["functional_areas"] = len(updated_rpg.get_functional_areas())
        except Exception:
            stats["functional_areas"] = 0

        return {"status": "success", **stats}

    except Exception as exc:
        logger.exception("Update failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Incremental RPG update")
    parser.add_argument("--json", action="store_true", help="Output as JSON (always JSON)")
    parser.add_argument("--rpg-file", "-i", default=str(RPG_FILE), help="Path to existing RPG JSON")
    parser.add_argument(
        "--repo-dir",
        "-r",
        default=None,
        help=(
            "Current repository directory. Defaults to the workspace "
            "root (the directory containing ``.cmind/``)."
        ),
    )
    parser.add_argument("--last-repo-dir", "-l", required=True, help="Previous version repo directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument(
        "--dep-graph",
        default=None,
        help=(
            "Legacy standalone dep_graph path. Normal updates embed the "
            "dependency graph in rpg.json."
        ),
    )
    parser.add_argument(
        "--max-exclude-votes",
        type=int,
        default=1,
        help=(
            "Number of LLM votes used to identify irrelevant files "
            "(default: 1 — single shot, no consolidation)."
        ),
    )
    args = parser.parse_args()

    result = run_update_rpg(
        rpg_file=args.rpg_file,
        last_repo_dir=args.last_repo_dir,
        cur_repo_dir=args.repo_dir,
        output=args.output,
        dep_graph_path=args.dep_graph,
        max_exclude_votes=args.max_exclude_votes,
    )
    print(json.dumps(result, indent=2))

    if result["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
