#!/usr/bin/env python3
"""Run Encode Script.

Full repository encode: calls RPGParser.parse_rpg_from_repo() to build
an RPG from scratch and saves it to .rpgkit/data/rpg.json.

Prints a single JSON result to stdout with status and statistics.

Usage:
    rpgkit script rpg_encoder/run_encode.py --json
    rpgkit script rpg_encoder/run_encode.py --repo-dir ./my-project
"""

import json
import logging
import os
import sys
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure scripts/ is importable
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from common.paths import RPG_FILE, DEP_GRAPH_FILE, RPG_HTML_FILE, WORKSPACE_ROOT, ensure_rpgkit_dir  # noqa: E402
from common.trajectory import Trajectory  # noqa: E402


def run_encode(
    repo_dir: str | None = None,
    repo_name: str | None = None,
    output: str | None = None,
    max_exclude_votes: int = 1,
) -> dict:
    """Run full RPG encode and return result dict.

    Args:
        repo_dir: Code directory to scan.  Defaults to
            :data:`common.paths.WORKSPACE_ROOT` — the directory the
            user ran ``rpgkit init --here`` in (their existing source
            repo).  Pass an explicit path to override.
        repo_name: Override the inferred repo name.
        output: Override the RPG output path.
        max_exclude_votes: Number of LLM votes used to identify irrelevant
            files. Defaults to ``1`` (single shot, no consolidation).
            Set higher for noisier repos that benefit from voting.
    """
    if repo_dir is None:
        repo_dir = str(WORKSPACE_ROOT)
    repo_dir = os.path.abspath(repo_dir)

    if not os.path.isdir(repo_dir):
        return {"status": "error", "error": f"Repository directory not found: {repo_dir}"}

    if repo_name is None:
        repo_name = os.path.basename(repo_dir) or "unknown"

    if output is None:
        output = str(RPG_FILE)

    # Ensure output directory exists
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize trajectory
    traj = Trajectory("encode", base_dir=Path(repo_dir))
    traj.start({"repo_dir": repo_dir, "repo_name": repo_name, "output_path": output})

    try:
        from rpg_encoder.rpg_encoding import RPGParser

        # Step 1: Parse RPG from repository
        step_parse = traj.add_step("parse_rpg", "Parse RPG from repository (repo info, exclude files, parse features, refactor)")
        traj.start_step(step_parse.step_id)

        parser = RPGParser(
            repo_dir=repo_dir,
            repo_name=repo_name,
        )

        # Pass trajectory to parser's LLM client for recording LLM calls
        parser.llm_client.set_trajectory(traj, step_parse.step_id)

        rpg, feature_tree, skeleton_info = parser.parse_rpg_from_repo(
            save_path=output,
            max_exclude_votes=max_exclude_votes,
        )

        traj.complete_step(step_parse.step_id, {
            "node_count": len(rpg.nodes),
            "edge_count": len(rpg.edges),
        })

        # Step 2: Build dependency graph
        step_dep = traj.add_step("dep_graph", "Build AST-level dependency graph")
        traj.start_step(step_dep.step_id)

        dep_graph_stats = {}
        dep_graph_output = None
        try:
            rpg.parse_dep_graph(repo_dir)
            if rpg.dep_graph:
                # Save dep_graph as a standalone file so that:
                #   1. rpg.json stays small (feature tree + maps only)
                #   2. git hooks can update dep_graph.json independently
                #   3. file layout is consistent from first encode onward
                dep_graph_output = str(DEP_GRAPH_FILE)
                os.makedirs(os.path.dirname(dep_graph_output), exist_ok=True)
                dg_dict = rpg.dep_graph.to_dict(
                    dep_to_rpg_map=rpg._dep_to_rpg_map,
                )
                with open(dep_graph_output, "w", encoding="utf-8") as dgf:
                    json.dump(dg_dict, dgf, indent=2, ensure_ascii=False)

                # Store a relative reference from rpg.json's directory to
                # dep_graph.json so the layout is portable.  Fall back to
                # the absolute path when they live in different trees
                # (e.g. user passed --output to a custom location).
                rpg_dir = Path(output).resolve().parent
                dep_graph_resolved = Path(dep_graph_output).resolve()
                try:
                    rpg._dep_graph_file = str(
                        dep_graph_resolved.relative_to(rpg_dir)
                    )
                except ValueError:
                    rpg._dep_graph_file = str(dep_graph_resolved)

                dep_graph_stats = {
                    "dep_nodes": rpg.dep_graph.G.number_of_nodes(),
                    "dep_edges": rpg.dep_graph.G.number_of_edges(),
                    "dep_to_rpg_map_size": len(rpg._dep_to_rpg_map),
                }
            traj.complete_step(step_dep.step_id, dep_graph_stats)
        except Exception as exc:
            logger.warning("Failed to update dependency graph: %s", exc)
            traj.fail_step(step_dep.step_id, str(exc))

        # Step 3: Save RPG to disk
        step_save = traj.add_step("save_rpg", "Save RPG to disk")
        traj.start_step(step_save.step_id)

        result_data = rpg.to_dict()

        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result_data, fh, indent=2, ensure_ascii=False)

        output_size = os.path.getsize(output)
        traj.complete_step(step_save.step_id, {
            "output_path": output,
            "output_size_bytes": output_size,
        })

        # Step 4: Generate visualization HTML
        step_viz = traj.add_step("visualize", "Generate interactive visualization HTML")
        traj.start_step(step_viz.step_id)

        viz_output = None
        try:
            from rpg_visualize import load_rpg, generate_html

            viz_data = load_rpg(output)
            html_content = generate_html(viz_data)
            # rpg.html is a user-facing artefact: keep it in the
            # workspace's .rpgkit/reports/ rather than next to the
            # machine-side rpg.json under ~/.rpgkit/workspaces/<workspace-id>/.
            RPG_HTML_FILE.parent.mkdir(parents=True, exist_ok=True)
            viz_output = str(RPG_HTML_FILE)
            RPG_HTML_FILE.write_text(html_content, encoding="utf-8")
            traj.complete_step(step_viz.step_id, {"viz_path": viz_output})
        except Exception as viz_exc:
            logger.warning("Failed to generate visualization: %s", viz_exc)
            traj.fail_step(step_viz.step_id, str(viz_exc))

        # Collect stats — use result_data (serialized) edge count since
        # to_dict() merges dep-graph semantic edges that aren't in self.edges.
        stats = {
            "repo_name": repo_name,
            "output_path": output,
            "node_count": len(rpg.nodes),
            "edge_count": len(result_data.get("edges", [])),
        }
        if viz_output:
            stats["viz_path"] = viz_output
        try:
            stats["functional_areas"] = len(rpg.get_functional_areas())
        except Exception:
            stats["functional_areas"] = 0
        stats.update(dep_graph_stats)

        traj.complete(stats)
        stats["trajectory"] = str(traj.trajectory_file)

        return {"status": "success", **stats}

    except Exception as exc:
        logger.exception("Encoding failed: %s", exc)
        traj.fail(str(exc))
        return {"status": "error", "error": str(exc), "trajectory": str(traj.trajectory_file)}


def main():
    parser = argparse.ArgumentParser(description="Full RPG encode")
    parser.add_argument("--json", action="store_true", help="Output as JSON (always JSON)")
    parser.add_argument(
        "--repo-dir",
        "-r",
        default=None,
        help=(
            "Repository directory to scan. Defaults to the workspace "
            "root (the directory containing ``.rpgkit/``, i.e. where "
            "``rpgkit init --here`` was run)."
        ),
    )
    parser.add_argument("--repo-name", default=None, help="Repository name")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument(
        "--max-exclude-votes",
        type=int,
        default=1,
        help=(
            "Number of LLM votes used to identify irrelevant files "
            "(default: 1 — single shot, no consolidation). "
            "Increase for noisier repos that benefit from voting."
        ),
    )
    args = parser.parse_args()

    ensure_rpgkit_dir()
    result = run_encode(
        repo_dir=args.repo_dir,
        repo_name=args.repo_name,
        output=args.output,
        max_exclude_votes=args.max_exclude_votes,
    )
    print(json.dumps(result, indent=2))

    if result["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
