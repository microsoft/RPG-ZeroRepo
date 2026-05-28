#!/usr/bin/env python3
"""Check Encode Script.

Inspect .cmind/data/rpg.json existence and validity to determine
the appropriate encode action.

Decision rules:
- If rpg.json does not exist → type "init" (first-time encode needed)
- If rpg.json exists and is valid → type "update" (incremental update possible)
- If rpg.json exists but is invalid → type "error"

The script prints EXACTLY ONE JSON object to stdout.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure scripts/ is importable (for common.paths etc.)
_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from common.paths import RPG_FILE  # noqa: E402


def load_json(path: Path) -> Dict[str, Any] | None:
    """Load JSON file safely."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                return data
    except Exception:
        pass
    return None


def get_rpg_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic statistics from RPG JSON data."""
    stats: Dict[str, Any] = {}
    stats["repo_name"] = data.get("repo_name", "unknown")

    # Count nodes
    nodes = data.get("nodes", [])
    if isinstance(nodes, list):
        stats["node_count"] = len(nodes)
    elif isinstance(nodes, dict):
        stats["node_count"] = len(nodes)
    else:
        stats["node_count"] = 0

    # Count edges
    edges = data.get("edges", [])
    if isinstance(edges, list):
        stats["edge_count"] = len(edges)
    elif isinstance(edges, dict):
        stats["edge_count"] = len(edges)
    else:
        stats["edge_count"] = 0

    # Check for nested rpg.structure format
    rpg_data = data.get("rpg", {})
    if isinstance(rpg_data, dict) and "structure" in rpg_data:
        structure = rpg_data["structure"]
        if isinstance(structure, dict):
            nodes_s = structure.get("nodes", [])
            edges_s = structure.get("edges", [])
            stats["node_count"] = len(nodes_s) if isinstance(nodes_s, (list, dict)) else 0
            stats["edge_count"] = len(edges_s) if isinstance(edges_s, (list, dict)) else 0

    # Check for tree format with 'root' key (nested children structure)
    root = data.get("root")
    if isinstance(root, dict) and stats["node_count"] == 0:
        def _count_tree_nodes(node: Dict[str, Any]) -> int:
            count = 1
            for child in node.get("children", []):
                if isinstance(child, dict):
                    count += _count_tree_nodes(child)
            return count
        stats["node_count"] = _count_tree_nodes(root)

    return stats


def check_encode() -> Dict[str, Any]:
    """Check encode state and return a result dict."""
    rpg_path = Path(RPG_FILE)

    # Case 1: RPG file does not exist → init
    if not rpg_path.exists():
        return {
            "type": "init",
            "message": "No RPG file found. Full encode is required.",
            "rpg_file": str(rpg_path),
        }

    # Case 2: RPG file exists — try to load and validate
    data = load_json(rpg_path)
    if data is None:
        return {
            "type": "error",
            "message": f"RPG file exists but is empty or invalid JSON: {rpg_path}",
            "rpg_file": str(rpg_path),
        }

    # Validate that the file has expected RPG structure
    has_nodes = "nodes" in data
    has_nested = isinstance(data.get("rpg", {}), dict) and "structure" in data.get("rpg", {})
    has_root = "root" in data and isinstance(data.get("root"), dict)

    if not has_nodes and not has_nested and not has_root:
        return {
            "type": "error",
            "message": (
                f"RPG file exists but has invalid format (missing 'root', "
                f"'nodes', or 'rpg.structure'): {rpg_path}"
            ),
            "rpg_file": str(rpg_path),
        }

    # Case 3: Valid RPG file → update
    stats = get_rpg_stats(data)
    return {
        "type": "update",
        "message": "Valid RPG file found. Incremental update or full re-encode available.",
        "rpg_file": str(rpg_path),
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Check RPG encode state")
    parser.add_argument("--json", action="store_true", help="Output as JSON (always JSON)")
    parser.parse_args()

    result = check_encode()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
