#!/usr/bin/env python3
"""Pre-check for rpg_edit inputs and the embedded dependency graph."""

import argparse
import json
import sys
from pathlib import Path

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import REPO_RPG_FILE, DEP_GRAPH_FILE  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpg", type=Path,
                        default=REPO_RPG_FILE)
    parser.add_argument("--dep-graph", type=Path,
                        default=DEP_GRAPH_FILE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    if not args.rpg.exists():
        result = {"type": "error", "error_code": "rpg_not_found",
                  "message": f"RPG file not found: {args.rpg}"}
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    try:
        from rpg.service import RPGService
        svc = RPGService.load(str(args.rpg))
    except Exception as e:
        result = {"type": "error", "error_code": "rpg_load_failed",
                  "message": f"Failed to load RPG: {e}"}
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    has_dep_graph = svc.rpg.dep_graph is not None
    if not has_dep_graph and not args.dep_graph.exists():
        result = {"type": "error", "error_code": "dep_graph_not_found",
                  "message": (
                      f"rpg.json has no embedded dep_graph and no legacy "
                      f"standalone dep_graph.json at {args.dep_graph}. "
                      "Run /cmind.encode to (re)build it; the embedded "
                      "dep_graph rides inside rpg.json."
                  )}
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    result = {
        "type": "ready",
        "rpg_path": str(args.rpg),
        "nodes": len(svc.rpg._node_index),
        "edges": len(svc.rpg.edges),
        "has_dep_graph": has_dep_graph,
        "dep_to_rpg": len(svc.rpg._dep_to_rpg_map),
        "feature_to_dep": len(svc.rpg._feature_to_dep_map),
    }
    print(json.dumps(result, indent=2) if args.json else
          f"Ready: {result['nodes']} nodes, dep_graph={'yes' if has_dep_graph else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
