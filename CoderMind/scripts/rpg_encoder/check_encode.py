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

from common.paths import RPG_FILE, WORKSPACE_ROOT  # noqa: E402


def _cwd_workspace_rpg_path() -> Path:
    """Return the workspace-local RPG path nearest to the current cwd."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        cmind_dir = candidate / ".cmind"
        if cmind_dir.is_dir():
            return cmind_dir / "data" / "rpg.json"
    return cwd / ".cmind" / "data" / "rpg.json"


def _rpg_path_candidates() -> list[Path]:
    """Return RPG paths to probe, ordered by preferred storage layout."""
    cwd = Path.cwd().resolve()
    workspace_root = Path(WORKSPACE_ROOT).resolve()
    workspace_local = _cwd_workspace_rpg_path()
    candidates: list[Path] = []
    try:
        in_import_workspace = cwd.is_relative_to(workspace_root)
    except ValueError:
        in_import_workspace = False
    if in_import_workspace:
        candidates.append(Path(RPG_FILE))
    candidates.append(workspace_local)
    return candidates


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


def _count_graph_items(value: Any) -> int:
    return len(value) if isinstance(value, (list, dict)) else 0


def _embedded_dep_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    dep_graph = data.get("dep_graph")
    if isinstance(dep_graph, dict):
        return dep_graph

    rpg_data = data.get("rpg")
    if isinstance(rpg_data, dict) and isinstance(rpg_data.get("structure"), dict):
        dep_graph = rpg_data["structure"].get("dep_graph")
        if isinstance(dep_graph, dict):
            return dep_graph

    return {}


def get_rpg_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic statistics from RPG JSON data."""
    stats: Dict[str, Any] = {}
    stats["repo_name"] = data.get("repo_name", "unknown")

    # Count nodes
    nodes = data.get("nodes", [])
    stats["node_count"] = _count_graph_items(nodes)

    # Count edges
    edges = data.get("edges", [])
    stats["edge_count"] = _count_graph_items(edges)

    # Check for nested rpg.structure format
    rpg_data = data.get("rpg", {})
    if isinstance(rpg_data, dict) and "structure" in rpg_data:
        structure = rpg_data["structure"]
        if isinstance(structure, dict):
            nodes_s = structure.get("nodes", [])
            edges_s = structure.get("edges", [])
            stats["node_count"] = _count_graph_items(nodes_s)
            stats["edge_count"] = _count_graph_items(edges_s)

    dep_graph = _embedded_dep_graph(data)
    stats["dep_nodes"] = _count_graph_items(dep_graph.get("nodes", []))
    stats["dep_edges"] = _count_graph_items(dep_graph.get("edges", []))

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
    candidates = _rpg_path_candidates()
    rpg_path = next((path for path in candidates if path.exists()), candidates[0])

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
