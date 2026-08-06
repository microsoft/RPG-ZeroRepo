#!/usr/bin/env python3
"""RPG Visualizer — Generate an interactive graph visualization of RPG data.

Renders three views:
1. **Feat Graph** — collapsible tree layout (D3.js) from rpg.json
2. **Dep Graph** — collapsible force-directed layout from the dep_graph
  embedded in rpg.json. Nodes are grouped by file hierarchy, collapsible at any level.
   Edges merge when groups are collapsed.
3. **Mapping** — RPG feature tree (L→R) linked to dep tree (R→L) via _dep_to_rpg_map

Default: only the first level (functional areas) is expanded.

Usage:
    python3 scripts/rpg_visualize.py [rpg.json] [--dep-graph legacy_dep_graph.json] [-o output.html]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_relative_to_rpg(rpg_path: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else rpg_path.parent / candidate


def resolve_dep_graph_path(rpg_path: Path, data: dict, dep_graph_path: str | Path | None = None) -> Path | None:
    if dep_graph_path:
        path = resolve_relative_to_rpg(rpg_path, dep_graph_path)
        return path if path.is_file() else None

    candidates: List[Path] = []
    if data.get("dep_graph_file"):
        candidates.append(Path(data["dep_graph_file"]).expanduser())
    candidates.append(Path("dep_graph.json"))

    for candidate in candidates:
        path = resolve_relative_to_rpg(rpg_path, candidate)
        if path.is_file():
            return path
    return None


def load_rpg(path: str | Path, dep_graph_path: str | Path | None = None) -> dict:
    rpg_path = Path(path).expanduser()
    data = load_json(rpg_path)

    embedded_dep = data.get("dep_graph", {})
    has_embedded_dep = isinstance(embedded_dep, dict) and bool(embedded_dep.get("nodes"))
    if dep_graph_path or not has_embedded_dep:
        resolved_dep_path = resolve_dep_graph_path(rpg_path, data, dep_graph_path)
        if resolved_dep_path:
            data["dep_graph"] = load_json(resolved_dep_path)
        elif dep_graph_path:
            raise FileNotFoundError(f"dep_graph override not found: {dep_graph_path}")

    return data


def normalize_to_tree(data: dict) -> dict:
    """Normalize both tree and flat format into a unified tree dict for D3."""
    if "root" in data and isinstance(data["root"], dict):
        return data["root"]

    # Flat format: reconstruct tree from nodes + contains edges
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    children_map: Dict[str, List[str]] = {nid: [] for nid in nodes}
    child_set = set()
    for e in data.get("edges", []):
        rel = e.get("relation", "")
        if rel in ("contains", "CONTAINS", "composes", "COMPOSES"):
            src, dst = e.get("src", ""), e.get("dst", "")
            if src in children_map:
                children_map[src].append(dst)
                child_set.add(dst)

    roots = [nid for nid in nodes if nid not in child_set]

    def to_tree(nid: str) -> dict:
        node = dict(nodes.get(nid, {"id": nid}))
        node["children"] = [to_tree(cid) for cid in children_map.get(nid, [])]
        return node

    if len(roots) == 1:
        return to_tree(roots[0])
    return {
        "id": "__root__",
        "name": data.get("repo_name", "root"),
        "node_type": "repository",
        "level": 0,
        "meta": {"type_name": "root", "path": "."},
        "children": [to_tree(r) for r in roots],
    }


def get_semantic_edges(data: dict) -> List[dict]:
    """Extract non-containment edges."""
    edges = data.get("edges", [])
    return [
        e for e in edges
        if e.get("relation", "") not in ("contains", "CONTAINS", "composes", "COMPOSES")
    ]


def count_nodes(node: dict) -> int:
    c = 1
    for ch in node.get("children", []):
        c += count_nodes(ch)
    return c


def extract_dep_graph(data: dict) -> Dict[str, Any]:
    """Extract dep_graph nodes, hierarchy, and semantic edges for D3."""
    dg = data.get("dep_graph", {})
    if not dg:
        return {"nodes": [], "edges": [], "parent_map": {}, "stats": {}}

    raw_nodes = dg.get("nodes", {})
    raw_edges = dg.get("edges", [])

    # Build parent map from CONTAINS edges
    parent_map: Dict[str, str] = {}
    for e in raw_edges:
        etype = e.get("attrs", {}).get("type", "")
        if etype in ("contains", "CONTAINS"):
            parent_map[e["dst"]] = e["src"]

    # Semantic edges only
    edges = []
    edge_types: Dict[str, int] = {}
    for e in raw_edges:
        etype = e.get("attrs", {}).get("type", "")
        if etype in ("contains", "CONTAINS"):
            continue
        edges.append({
            "source": e["src"],
            "target": e["dst"],
            "type": etype,
        })
        edge_types[etype] = edge_types.get(etype, 0) + 1

    # Find nodes that participate in semantic edges
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["source"])
        connected_ids.add(e["target"])

    # Add all ancestors of connected nodes (so hierarchy is complete)
    relevant = set(connected_ids)
    for nid in connected_ids:
        cur = nid
        while cur in parent_map:
            cur = parent_map[cur]
            relevant.add(cur)

    # Compute depth for each node
    def get_depth(nid: str) -> int:
        d = 0
        cur = nid
        while cur in parent_map:
            cur = parent_map[cur]
            d += 1
        return d

    nodes = []
    for nid, attrs in raw_nodes.items():
        if nid not in relevant:
            continue
        nodes.append({
            "id": nid,
            "name": attrs.get("name", nid.split("/")[-1].split(":")[-1]),
            "type": attrs.get("type", "unknown"),
            "module": attrs.get("module", ""),
            "rpg_nodes": attrs.get("rpg_nodes", []),
            "depth": get_depth(nid),
        })

    # Filter parent_map to only relevant nodes
    filtered_parent = {k: v for k, v in parent_map.items() if k in relevant and v in relevant}

    return {
        "nodes": nodes,
        "edges": edges,
        "parent_map": filtered_parent,
        "stats": edge_types,
    }


def build_dep_tree(data: dict) -> dict:
    """Build a tree structure from dep_graph for the mapping tab (full tree, not just connected)."""
    dg = data.get("dep_graph", {})
    if not dg:
        return {"id": ".", "name": ".", "type": "directory", "children": []}
    raw_nodes = dg.get("nodes", {})
    raw_edges = dg.get("edges", [])

    parent_map: Dict[str, str] = {}
    children_map: Dict[str, List[str]] = {}
    for e in raw_edges:
        if e.get("attrs", {}).get("type", "") in ("contains", "CONTAINS"):
            parent_map[e["dst"]] = e["src"]
            children_map.setdefault(e["src"], []).append(e["dst"])

    roots = [nid for nid in raw_nodes if nid not in parent_map]

    def to_tree(nid):
        attrs = raw_nodes.get(nid, {})
        node = {
            "id": nid,
            "name": attrs.get("name", nid.split("/")[-1].split(":")[-1]),
            "type": attrs.get("type", "unknown"),
            "rpg_nodes": attrs.get("rpg_nodes", []),
            "children": [to_tree(c) for c in sorted(children_map.get(nid, []))],
        }
        return node

    if len(roots) == 1:
        return to_tree(roots[0])
    return {"id": "__dep_root__", "name": "repo", "type": "directory",
            "children": [to_tree(r) for r in sorted(roots)]}


def generate_html(data: dict, change_data: dict | None = None) -> str:
    tree = normalize_to_tree(data)
    semantic_edges = get_semantic_edges(data)
    dep = extract_dep_graph(data)
    dep_tree = build_dep_tree(data)
    dep_to_rpg = data.get("_dep_to_rpg_map", {})
    repo_name = data.get("repo_name", "Unknown")
    feat_node_count = count_nodes(tree)
    feat_edge_count = len(semantic_edges)

    # Feat edge summary
    edge_types = {}
    for e in semantic_edges:
        r = e.get("relation", "unknown")
        edge_types[r] = edge_types.get(r, 0) + 1
    feat_edge_summary = ", ".join(f"{k}: {v}" for k, v in sorted(edge_types.items()))

    # Dep stats
    dep_node_count = len(dep["nodes"])
    dep_edge_count = len(dep["edges"])
    dep_edge_summary = ", ".join(f"{k}: {v}" for k, v in sorted(dep["stats"].items()))
    has_dep = dep_node_count > 0

    raw_dep_graph = data.get("dep_graph") or {}
    raw_dep_nodes = raw_dep_graph.get("nodes") or {}
    raw_dep_node_count = len(raw_dep_nodes) if isinstance(raw_dep_nodes, (dict, list)) else 0
    raw_dep_edge_count = len(raw_dep_graph.get("edges") or [])

    map_count = sum(len(v) for v in dep_to_rpg.values())
    has_map = len(dep_to_rpg) > 0

    tree_json = json.dumps(tree)
    edges_json = json.dumps(semantic_edges)
    dep_nodes_json = json.dumps(dep["nodes"])
    dep_edges_json = json.dumps(dep["edges"])
    dep_parent_json = json.dumps(dep["parent_map"])
    dep_tree_json = json.dumps(dep_tree)
    dep_to_rpg_json = json.dumps(dep_to_rpg)

    change_json = json.dumps(change_data or {})
    has_change = 'true' if (change_data and change_data.get("available")) else 'false'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RPG: {repo_name}</title>
<script>
(function() {{
  var theme = 'dark';
  try {{ theme = localStorage.getItem('cmind-report-theme') || theme; }} catch (e) {{}}
  document.documentElement.setAttribute('data-theme', theme === 'light' ? 'light' : 'dark');
}})();
</script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>document.documentElement.classList.toggle('d3-unavailable', !window.d3);</script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --change-added: #4ade80; --change-added-strong: #22c55e; --change-added-soft: rgba(34,197,94,.16);
  --change-removed: #ff6b6b; --change-removed-strong: #ef4444; --change-removed-soft: rgba(239,68,68,.16);
  --change-modified: #f6c453; --change-modified-strong: #eab308; --change-modified-soft: rgba(234,179,8,.16);
  --change-focus: #60a5fa; --change-focus-soft: rgba(96,165,250,.22);
}}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont,
       'Segoe UI', monospace; overflow: hidden; }}

#header {{ position: fixed; top: 0; left: 0; right: 0; z-index: 100;
           background: #161b22; border-bottom: 1px solid #30363d; padding: 8px 16px;
           display: flex; align-items: center; gap: 16px; font-size: 13px; }}
#header h1 {{ font-size: 15px; color: #58a6ff; white-space: nowrap; }}
.stat {{ color: #8b949e; }}
.stat b {{ color: #c9d1d9; }}

#tabs {{ display: flex; gap: 2px; }}
#tabs button {{ background: #21262d; color: #8b949e; border: 1px solid #30363d;
                padding: 3px 12px; border-radius: 4px 4px 0 0; cursor: pointer; font-size: 12px;
                border-bottom: 2px solid transparent; }}
#tabs button:hover {{ color: #c9d1d9; }}
#tabs button.active {{ background: #0d1117; color: #58a6ff; border-bottom-color: #58a6ff; }}

#controls {{ display: flex; gap: 4px; align-items: center; }}
#controls button {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                    padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
#controls button:hover {{ background: #30363d; }}
#controls button.active {{ background: #1f6feb; border-color: #1f6feb; }}

#search {{ background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
           padding: 4px 8px; border-radius: 4px; width: 200px; font-size: 12px; }}

#legend {{ display: flex; gap: 12px; align-items: center; margin-left: auto; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #8b949e; }}
.legend-line {{ width: 20px; height: 2px; }}

#canvas-overlay {{
  position: fixed; top: 52px; left: 12px; z-index: 50;
  background: rgba(22, 27, 34, 0.85); border: 1px solid #30363d; border-radius: 6px;
  padding: 10px 14px; font-size: 11px; color: #8b949e; pointer-events: none;
  max-width: 220px; line-height: 1.6;
}}
#canvas-overlay .ov-title {{ color: #c9d1d9; font-weight: 600; margin-bottom: 4px; font-size: 12px; }}
#canvas-overlay .ov-section {{ margin-top: 6px; }}
#canvas-overlay .ov-key {{ color: #58a6ff; }}

#stats-feat, #stats-dep, #stats-map {{ display: flex; gap: 16px; align-items: center; }}

svg {{ width: 100vw; height: 100vh; }}
#canvas {{ cursor: grab; }}
#canvas:active {{ cursor: grabbing; }}

.link {{ fill: none; stroke: #21262d; stroke-width: 1; transition: opacity .14s, stroke-width .14s, stroke .14s; }}
.link.feat-related {{ stroke: #58a6ff; stroke-width: 2.4; opacity: .95; }}
.link.feat-dim {{ opacity: .08; }}
.semantic-edge {{ transition: stroke-opacity .14s, stroke-width .14s; }}
.semantic-edge.feat-related {{ stroke-opacity: .98 !important; stroke-width: 2.8 !important; }}
.semantic-edge.feat-dim {{ stroke-opacity: .06 !important; }}
.node {{ transition: opacity .14s; }}
.node circle {{ stroke: #30363d; stroke-width: 1.5; cursor: pointer; }}
.node text {{ font-size: 13px; fill: #c9d1d9; pointer-events: none; }}
.node .node-label {{ pointer-events: visiblePainted; cursor: pointer; }}
.node.feat-selected > .node-dot {{ stroke: #f0883e !important; stroke-width: 3.5 !important; filter: drop-shadow(0 0 5px rgba(240,136,62,.75)); }}
.node.feat-selected > .node-label {{ fill: #fff; font-weight: 800; }}
.node.feat-related {{ opacity: 1; }}
.node.feat-related > .node-dot {{ stroke: #58a6ff; stroke-width: 2.5; }}
.node.feat-related > .node-label {{ fill: #e6edf3; font-weight: 650; }}
.node.feat-dim {{ opacity: .18; }}
.node-collapsed circle {{ fill: #1f6feb !important; }}

.edge-imports {{ stroke: #f0883e; stroke-opacity: 0.5; }}
.edge-inherits {{ stroke: #a371f7; stroke-opacity: 0.6; }}
.edge-invokes {{ stroke: #3fb950; stroke-opacity: 0.5; }}
.edge-references {{ stroke: #79c0ff; stroke-opacity: 0.4; }}
.edge-default {{ stroke: #8b949e; stroke-opacity: 0.3; }}

/* Dep graph specific */
.dep-link {{ fill: none; stroke-width: 1.2; stroke-opacity: 0.5; transition: opacity .14s, stroke-width .14s; }}
.dep-link-imports {{ stroke: #f0883e; }}
.dep-link-invokes {{ stroke: #3fb950; }}
.dep-link-inherits {{ stroke: #a371f7; }}
.dep-link-default {{ stroke: #8b949e; }}
.dep-node {{ transition: opacity .14s; }}
.dep-node circle {{ cursor: pointer; stroke: #30363d; stroke-width: 1.5; }}
.dep-node text {{ font-size: 10px; fill: #c9d1d9; pointer-events: none; }}
.dep-node-collapsed circle {{ stroke: #58a6ff; stroke-width: 2; }}

.map-link {{ fill: none; stroke: #f0883e; stroke-opacity: 0.35; stroke-width: 1.2; }}
.map-link-hover {{ stroke-opacity: 0.9; stroke-width: 2.5; }}
.map-node-highlight circle {{ stroke: #f0883e !important; stroke-width: 3 !important; }}
.change-added > circle:not(.change-focus-ring):not(.change-status-dot) {{ stroke: var(--change-added) !important; stroke-width: 3 !important; filter: drop-shadow(0 0 3px var(--change-added-soft)); }}
.change-added text {{ fill: var(--change-added) !important; font-weight: 700; }}
.change-modified > circle:not(.change-focus-ring):not(.change-status-dot) {{ stroke: var(--change-modified) !important; stroke-width: 3 !important; filter: drop-shadow(0 0 3px var(--change-modified-soft)); }}
.change-modified text {{ fill: var(--change-modified) !important; font-weight: 700; }}
.change-dim {{ opacity: 0.28; }}
.change-dim > circle:not(.change-focus-ring):not(.change-status-dot) {{ fill: #6e7681 !important; stroke: #484f58 !important; filter: none !important; }}
.change-dim text {{ fill: #8b949e !important; }}
.change-status-dot,.change-status-glyph {{ display: none; pointer-events: none; }}
g.change-added > .change-status-dot {{ display: inline; fill: var(--change-added-strong); stroke: var(--change-added); stroke-width: 2; }}
g.change-modified > .change-status-dot {{ display: inline; fill: var(--change-modified-strong); stroke: var(--change-modified); stroke-width: 2; }}
g.change-added > .change-status-glyph,g.change-modified > .change-status-glyph {{ display: block; font-size: 9px; font-weight: 900; text-anchor: middle; fill: #fff !important; }}
g.change-modified > .change-status-glyph {{ fill: #1f2937 !important; }}
.change-focused {{ opacity: 1 !important; }}
.change-focus-ring {{ fill: none !important; stroke: var(--change-focus) !important; stroke-width: 2.5 !important; filter: drop-shadow(0 0 5px var(--change-focus)); pointer-events: none; vector-effect: non-scaling-stroke; }}
.node.change-focused text:not(.change-status-glyph) {{ fill: #fff !important; font-weight: 800; paint-order: stroke; stroke: #0d1117; stroke-width: 3px; }}
.dep-node.change-focused text:not(.change-status-glyph) {{ fill: #fff !important; font-weight: 700; }}
.dep-hull.change-added {{ stroke: var(--change-added) !important; stroke-width: 3 !important; fill: var(--change-added-soft) !important; }}
.dep-hull.change-modified {{ stroke: var(--change-modified) !important; stroke-width: 3 !important; fill: var(--change-modified-soft) !important; }}
.dep-hull.change-focused {{ filter: drop-shadow(0 0 5px var(--change-focus)); }}
.dep-hull-label.change-added {{ fill: var(--change-added) !important; font-weight: 700; }}
.dep-hull-label.change-modified {{ fill: var(--change-modified) !important; font-weight: 700; }}
.dep-hull-label.change-focused {{ fill: var(--change-focus) !important; font-weight: 800; }}
.dep-change-group rect {{ fill: var(--change-modified-soft); stroke: var(--change-modified); stroke-width: 2; rx: 5; cursor: pointer; }}
.dep-change-group.added rect {{ fill: var(--change-added-soft); stroke: var(--change-added); }}
.dep-change-group text {{ fill: var(--change-modified); font-size: 10px; font-weight: 700; pointer-events: none; }}
.dep-change-group.added text {{ fill: var(--change-added); }}
.dep-change-group.focused rect {{ filter: drop-shadow(0 0 6px var(--change-focus)); }}
.dep-change-group.focused text {{ fill: #fff; }}
.feat-ghost-link,.dep-ghost-link {{ fill: none; stroke: var(--change-removed); stroke-opacity: .62; stroke-width: 1.4; stroke-dasharray: 5 4; }}
.feat-ghost .change-status-dot,.dep-ghost .change-status-dot {{ display: inline; fill: var(--change-removed-strong); stroke: var(--change-removed); stroke-width: 2; cursor: pointer; }}
.feat-ghost .change-node-label,.dep-ghost .change-node-label {{ fill: var(--change-removed); font-size: 11px; font-weight: 650; cursor: pointer; paint-order: stroke; stroke: #0d1117; stroke-width: 3px; stroke-linejoin: round; }}
.feat-ghost.branch .change-node-label,.dep-ghost.branch .change-node-label {{ font-weight: 750; text-anchor: middle; }}
.feat-ghost.leaf .change-node-label,.dep-ghost.leaf .change-node-label {{ text-anchor: start; }}
.feat-ghost .change-status-glyph,.dep-ghost .change-status-glyph {{ display: block; fill: #fff !important; font-size: 9px; font-weight: 900; text-anchor: middle; pointer-events: none; }}
.feat-ghost.focused .change-node-label,.dep-ghost.focused .change-node-label {{ fill: var(--change-focus); font-weight: 800; paint-order: stroke; stroke: #0d1117; stroke-width: 3px; }}
.change-graph-link {{ stroke: #6e7681; stroke-width: 1.1; stroke-opacity: 0.35; transition: stroke-opacity .14s, stroke-width .14s; }}
.change-graph-link.rel-imports {{ stroke: #f0883e; stroke-opacity: .62; }}
.change-graph-link.rel-invokes {{ stroke: #3fb950; stroke-opacity: .62; }}
.change-graph-link.rel-inherits {{ stroke: #a371f7; stroke-opacity: .68; }}
.change-graph-link.rel-references {{ stroke: #79c0ff; stroke-opacity: .58; }}
.change-graph-link.rel-contains {{ stroke: #6e7681; stroke-dasharray: 3 4; stroke-opacity: .48; }}
.change-graph-link.rel-maps-to {{ stroke: #f0883e; stroke-dasharray: 7 3; stroke-opacity: .68; }}
.change-graph-link.added {{ stroke: #3fb950; stroke-width: 2; stroke-opacity: 0.8; }}
.change-graph-link.removed {{ stroke: #f85149; stroke-width: 1.8; stroke-dasharray: 5 4; stroke-opacity: 0.75; }}
.change-graph-link.focus-related {{ stroke-width: 2.6; stroke-opacity: .95; }}
.change-graph-link.focus-dim {{ stroke-opacity: .06; }}
.change-graph-node {{ pointer-events: all; cursor: pointer; transition: opacity .14s; }}
.change-graph-node circle {{ fill: var(--node-fill,#484f58); stroke: #30363d; stroke-width: 2; pointer-events: all; cursor: pointer; }}
.change-graph-node .node-label {{ fill: #e6edf3; font-size: 11.5px; font-weight: 600; pointer-events: none;
  paint-order: stroke; stroke: #0d1117; stroke-width: 3px; stroke-linejoin: round; }}
.change-graph-node .status-glyph {{ fill: #0d1117; font-size: 10px; font-weight: 800; text-anchor: middle; }}
.change-graph-node.added circle {{ stroke: #3fb950; stroke-width: 4; }}
.change-graph-node.removed circle {{ stroke: #f85149; stroke-width: 3; stroke-dasharray: 5 3; }}
.change-graph-node.modified circle {{ stroke: #d29922; stroke-width: 4; }}
.change-graph-node.context circle {{ stroke: #6e7681; }}
.change-graph-node.normal circle {{ stroke: #30363d; }}
.change-graph-node.context {{ opacity: .42; }}
.change-graph-node.focus-neighbor {{ opacity: 1; }}
.change-graph-node.focus-dim {{ opacity: .16; }}
.change-graph-node.focused circle {{ stroke: #58a6ff !important; stroke-width: 5 !important; filter: drop-shadow(0 0 7px #58a6ff); }}
.change-graph-node.focused text {{ fill: #fff; font-weight: 700; }}
.removed-rail-bg {{ fill: rgba(22,27,34,.92); stroke: #30363d; rx: 7; }}
.removed-rail-title {{ fill: #f85149; font-size: 12px; font-weight: 700; }}
.removed-rail-node circle {{ fill: rgba(248,81,73,.14); stroke: #f85149; stroke-width: 2; stroke-dasharray: 4 3; }}
.removed-rail-node text {{ fill: #c9d1d9; font-size: 10px; }}
.removed-rail-more {{ fill: #8b949e; font-size: 10px; }}

#change-summary {{
  position: fixed; top: 58px; right: 12px; z-index: 55; display: flex; gap: 8px;
  align-items: center; background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d;
  border-radius: 6px; padding: 7px 10px; color: #8b949e; font-size: 11px;
  pointer-events: none; backdrop-filter: blur(6px);
}}
#change-summary[hidden] {{ display: none; }}
#change-summary b {{ color: #c9d1d9; }}
#change-summary .added {{ color: #3fb950; }}
#change-summary .removed {{ color: #f85149; }}
#change-summary .modified {{ color: #d29922; }}

.no-data {{ display: flex; align-items: center; justify-content: center;
            height: 80vh; color: #484f58; font-size: 16px; }}

.tooltip {{ position: fixed; background: #1c2128; border: 1px solid #30363d; padding: 8px 12px;
            border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 200;
            max-width: 350px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.tooltip .tt-name {{ color: #58a6ff; font-weight: bold; }}
.tooltip .tt-type {{ color: #8b949e; font-size: 11px; }}
.tooltip .tt-path {{ color: #7ee787; font-size: 11px; }}
.tooltip .tt-edges {{ color: #f0883e; font-size: 11px; margin-top: 4px; }}

/* ── Self-contained change workbench (two-row header + right panel) ── */
#header {{ height: 78px; padding: 7px 12px; flex-direction: column; align-items: stretch; gap: 4px; overflow: hidden; }}
#header-row1 {{ display: grid; grid-template-columns: auto auto minmax(95px,1fr); align-items: center; gap: 9px; min-width: 0; min-height: 31px; }}
#header-row1 h1 {{ font-size: 14px; }}
#header-row2 {{ display: flex; align-items: center; gap: 14px; min-height: 29px; padding-top: 4px; padding-right: 450px; border-top: 1px solid #21262d; }}
#tabs {{ flex-wrap: nowrap; }}
#tabs button {{ padding: 3px 9px; white-space: nowrap; }}
#stats-feat, #stats-dep, #stats-map {{ min-width: 0; gap: 8px; overflow: hidden; white-space: nowrap; }}
#stats-feat .stat, #stats-dep .stat, #stats-map .stat {{ overflow: hidden; text-overflow: ellipsis; font-size: 10.5px; }}
#controls {{ position: absolute; top: 43px; right: 12px; height: 28px; flex-wrap: nowrap; min-width: 0; gap: 3px; }}
#controls > span {{ display: inline-flex; flex-wrap: nowrap; gap: 3px; }}
#controls button {{ padding: 3px 7px; white-space: nowrap; }}
#search {{ width: 120px; min-width: 80px; }}
@media (max-width: 600px) {{
  #header {{ height: 136px; padding: 6px; gap: 4px; }}
  #header-row1 {{ grid-template-columns: minmax(0,1fr); gap: 4px; min-height: 55px; }}
  #header-row1 h1 {{ overflow: hidden; text-overflow: ellipsis; }}
  #tabs {{ width: 100%; }}
  #tabs button {{ flex: 1; padding: 3px 5px; }}
  #stats-feat, #stats-dep, #stats-map {{ display: none !important; }}
  #header-row2 {{ min-height: 30px; padding: 3px 0 0; gap: 6px; padding-right: 0; }}
  #header-row2 .hdr-group {{ min-width: 0; width: 100%; }}
  #mode-seg {{ flex: 1; }}
  #mode-seg button {{ flex: 1; padding: 4px 6px; }}
  #controls {{ top: 101px; left: 6px; right: 6px; width: auto; max-width: none; height: 29px;
    gap: 2px; overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }}
  #controls::-webkit-scrollbar {{ display: none; }}
  #controls > span {{ gap: 2px; }}
  #controls button {{ padding: 3px; font-size: 11px; }}
  #search {{ width: 94px; min-width: 94px; padding: 4px 5px; }}
  #changes-panel {{ top: 136px; width: min(300px, 82vw); }}
  #canvas-overlay {{ top: 146px; }}
  #d3-offline {{ inset: 136px 0 0; }}
}}
.hdr-group {{ display: flex; align-items: center; gap: 8px; }}
.hdr-label {{ color: #6e7681; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }}
.seg2 {{ display: inline-flex; gap: 2px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 2px; }}
.seg2 button {{ background: none; border: 0; color: #8b949e; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
.seg2 button:hover {{ color: #c9d1d9; }}
.seg2 button.active {{ background: #21262d; color: #58a6ff; }}
#status-seg button b {{ color: #6e7681; font-weight: 700; margin-left: 3px; }}
#status-seg button.active b {{ color: #c9d1d9; }}
#status-seg button[data-status="added"].active {{ color: #3fb950; }}
#status-seg button[data-status="removed"].active {{ color: #f85149; }}
#status-seg button[data-status="modified"].active {{ color: #d29922; }}
.seg2 button:disabled {{ opacity: .4; cursor: default; }}
#canvas-overlay {{ top: 88px; width: 186px; max-width: 186px; padding: 8px 10px; line-height: 1.45; }}
#canvas-overlay .legend-item {{ font-size: 10px; }}
.ov-help {{ margin-top: 7px; border-top: 1px solid #30363d; padding-top: 6px; pointer-events: auto; }}
.ov-help summary {{ color: #8b949e; cursor: pointer; font-size: 10.5px; font-weight: 600; }}
.ov-help div {{ margin-top: 5px; }}
#change-summary {{ display: none !important; }}
body.cp-open #canvas-overlay {{ left: 12px; }}
#changes-panel {{ position: fixed; top: 78px; right: 0; bottom: 0; width: clamp(300px, 25vw, 340px); z-index: 90;
  background: #161b22; border-left: 1px solid #30363d; display: flex; flex-direction: column;
  padding: 10px; gap: 7px; box-shadow: -12px 0 28px rgba(0,0,0,.18); }}
.cp-filters {{ flex: none; }}
.cp-filter-seg {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.cp-filter-seg button {{ display: flex; align-items: center; justify-content: space-between; gap: 7px; min-height: 40px;
  background: #0d1117; border: 1px solid #30363d; color: #8b949e; border-radius: 7px; padding: 6px 8px; cursor: pointer; font-size: 11px; transition: border-color .14s, background .14s, box-shadow .14s; }}
.cp-filter-seg button:hover {{ color: #c9d1d9; border-color: #58a6ff; background: #111a26; }}
.cp-filter-seg button.active {{ color: #c9d1d9; border-color: #58a6ff; background: rgba(88,166,255,.13); box-shadow: inset 0 0 0 1px rgba(88,166,255,.28); }}
.cp-filter-main {{ display: flex; align-items: center; gap: 6px; min-width: 0; }}
.cp-filter-icon {{ width: 20px; height: 20px; display: grid; place-items: center; flex: none; border-radius: 50%; background: #2563eb; color: #fff; font-weight: 900; font-size: 12px; }}
.cp-filter-seg button[data-status="added"] .cp-filter-icon {{ color: #fff; background: var(--change-added-strong); }}
.cp-filter-seg button[data-status="removed"] .cp-filter-icon {{ color: #fff; background: var(--change-removed-strong); }}
.cp-filter-seg button[data-status="modified"] .cp-filter-icon {{ color: #1f2937; background: var(--change-modified-strong); }}
.cp-filter-seg button b {{ color: #6e7681; font-weight: 700; }}
.cp-filter-seg button.active b {{ color: #c9d1d9; }}
.cp-filter-seg button[data-status="added"].active {{ color: var(--change-added); border-color: var(--change-added); background: var(--change-added-soft); box-shadow: inset 0 0 0 1px var(--change-added-soft); }}
.cp-filter-seg button[data-status="removed"].active {{ color: var(--change-removed); border-color: var(--change-removed); background: var(--change-removed-soft); box-shadow: inset 0 0 0 1px var(--change-removed-soft); }}
.cp-filter-seg button[data-status="modified"].active {{ color: var(--change-modified); border-color: var(--change-modified); background: var(--change-modified-soft); box-shadow: inset 0 0 0 1px var(--change-modified-soft); }}
#cp-head {{ display: flex; align-items: center; justify-content: space-between; }}
#cp-head strong {{ font-size: 13px; color: #c9d1d9; }}
.cp-count {{ color: #6e7681; font-size: 11px; }}
#cp-version {{ display: block; max-width: 250px; margin-top: 2px; color: #8b949e; font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
#cp-collapse {{ background: #21262d; border: 1px solid #30363d; color: #8b949e; border-radius: 5px; cursor: pointer; padding: 2px 8px; }}
#cp-search-wrap {{ display: flex; align-items: center; gap: 6px; min-height: 30px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; color: #6e7681; }}
#cp-search {{ flex: 1; background: none; border: 0; outline: 0; color: #c9d1d9; font-size: 12px; }}
#cp-list {{ display: flex; flex-direction: column; gap: 4px; overflow: auto; min-height: 0; flex: 1; }}
.cp-row {{ position: relative; display: grid; grid-template-columns:auto minmax(0,1fr); gap: 8px; align-items: center; text-align: left;
  min-height: 42px; overflow: hidden; background: #0d1117; border: 1px solid #21262d; border-radius: 7px; color: #c9d1d9; padding: 5px 7px 5px 10px; cursor: pointer; transition: border-color .14s, background .14s, box-shadow .14s; }}
.cp-row::before {{ content: ''; position: absolute; inset: 0 auto 0 0; width: 3px; background: #58a6ff; opacity: .55; }}
.cp-row.added::before {{ background: var(--change-added); }}
.cp-row.removed::before {{ background: var(--change-removed); }}
.cp-row.modified::before {{ background: var(--change-modified); }}
.cp-row:hover {{ border-color: #58a6ff; background: #101925; }}
.cp-row:focus-visible {{ outline: 2px solid #58a6ff; outline-offset: -2px; }}
.cp-row.selected {{ border-color: #58a6ff; background: linear-gradient(90deg, rgba(88,166,255,.20), #132030 72%); box-shadow: inset 0 0 0 1px rgba(88,166,255,.35), 0 0 0 1px rgba(88,166,255,.15); }}
.cp-row.selected::before {{ width: 4px; opacity: 1; }}
.cp-row.selected .cp-name {{ color: #fff; font-weight: 800; }}
.cp-row.selected .cp-path {{ color: #b8c6d8; }}
.cp-mark {{ min-width: 19px; min-height: 19px; display: inline-flex; align-items: center; justify-content: center; gap: 4px;
  border-radius: 999px; padding: 2px 7px; color: #fff; font-weight: 800; font-size: 10px; white-space: nowrap; }}
.cp-mark b {{ font-size: 12px; line-height: 1; }}
.cp-mark.added {{ background: var(--change-added-strong); }}
.cp-mark.removed {{ background: var(--change-removed-strong); }}
.cp-mark.modified {{ color: #1f2937; background: var(--change-modified-strong); }}
.change-legend-mark {{ width: 15px; height: 15px; display: grid; place-items: center; flex: none; border-radius: 50%; color: #fff; font-size: 9px; font-weight: 900; }}
.change-legend-mark.added {{ background: var(--change-added-strong); }}
.change-legend-mark.removed {{ background: var(--change-removed-strong); }}
.change-legend-mark.modified {{ color: #1f2937; background: var(--change-modified-strong); }}
.change-legend-mark.context {{ color: #fff; background: #64748b; }}
.cp-row-main {{ min-width: 0; }}
.cp-row-main .cp-name {{ display: block; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.cp-row-main .cp-path {{ display: block; font-size: 9px; color: #8b949e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.cp-detail {{ display: none; flex: none; max-height: 150px; overflow: auto; border: 1px solid #30363d; border-left: 4px solid #58a6ff; border-radius: 7px; padding: 8px 9px; background: #111923; font-size: 10.5px; color: #8b949e; }}
.cp-detail.selected {{ display: block; }}
.cp-detail.selected.added {{ border-left-color: #3fb950; }}
.cp-detail.selected.removed {{ border-left-color: #f85149; }}
.cp-detail.selected.modified {{ border-left-color: #d29922; }}
.cp-detail.empty {{ display: none; }}
.cp-d-title {{ display: flex; align-items: center; gap: 7px; color: #c9d1d9; font-size: 12px; font-weight: 600; margin-bottom: 7px; }}
.cp-detail dl {{ display: grid; grid-template-columns: 78px 1fr; gap: 4px 8px; margin: 0; }}
.cp-detail dt {{ color: #6e7681; }}
.cp-detail dd {{ color: #c9d1d9; word-break: break-word; margin: 0; }}

#btn-theme {{ width: 29px; padding-left: 0 !important; padding-right: 0 !important; }}
:root[data-theme="light"] {{ color-scheme: light; }}
:root[data-theme="light"] {{
  --change-added: #15803d; --change-added-strong: #16a34a; --change-added-soft: rgba(22,163,74,.12);
  --change-removed: #dc2626; --change-removed-strong: #dc2626; --change-removed-soft: rgba(220,38,38,.11);
  --change-modified: #a16207; --change-modified-strong: #eab308; --change-modified-soft: rgba(234,179,8,.15);
  --change-focus: #2563eb; --change-focus-soft: rgba(37,99,235,.18);
}}
:root[data-theme="light"] body {{ background: #f6f8fb; color: #1f2937; }}
:root[data-theme="light"] #header,
:root[data-theme="light"] #changes-panel {{ background: #ffffff; border-color: #d7dee8; }}
:root[data-theme="light"] #header-row2 {{ border-color: #e3e8ef; }}
:root[data-theme="light"] #header h1,
:root[data-theme="light"] #canvas-overlay .ov-key {{ color: #2563eb; }}
:root[data-theme="light"] .stat,
:root[data-theme="light"] .legend-item,
:root[data-theme="light"] .hdr-label,
:root[data-theme="light"] .cp-count,
:root[data-theme="light"] .cp-detail,
:root[data-theme="light"] .cp-detail dt {{ color: #475569; }}
:root[data-theme="light"] .stat b,
:root[data-theme="light"] #canvas-overlay .ov-title,
:root[data-theme="light"] #cp-head strong,
:root[data-theme="light"] .cp-detail dd,
:root[data-theme="light"] .cp-d-title {{ color: #1f2937; }}
:root[data-theme="light"] #tabs button,
:root[data-theme="light"] #controls button,
:root[data-theme="light"] #cp-collapse {{ background: #eef2f7; color: #64748b; border-color: #d7dee8; }}
:root[data-theme="light"] #tabs button:hover,
:root[data-theme="light"] #controls button:hover {{ background: #e2e8f0; color: #1f2937; }}
:root[data-theme="light"] #tabs button.active {{ background: #f6f8fb; color: #2563eb; border-color: #d7dee8; border-bottom-color: #2563eb; }}
:root[data-theme="light"] #controls button.active {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
:root[data-theme="light"] #search,
:root[data-theme="light"] #cp-search-wrap,
:root[data-theme="light"] .cp-filter-seg button,
:root[data-theme="light"] .cp-row,
:root[data-theme="light"] .seg2 {{ background: #f8fafc; border-color: #d7dee8; color: #334155; }}
:root[data-theme="light"] #search,
:root[data-theme="light"] #cp-search {{ color: #1f2937; }}
:root[data-theme="light"] .seg2 button {{ color: #64748b; }}
:root[data-theme="light"] .seg2 button:hover {{ color: #1f2937; }}
:root[data-theme="light"] .seg2 button.active {{ background: #e2e8f0; color: #2563eb; }}
:root[data-theme="light"] .cp-filter-seg button:hover,
:root[data-theme="light"] .cp-filter-seg button.active,
:root[data-theme="light"] .cp-row:hover,
:root[data-theme="light"] .cp-row.selected {{ border-color: #2563eb; background: #eaf1ff; }}
:root[data-theme="light"] .cp-filter-seg button.active {{ box-shadow: inset 0 0 0 1px rgba(37,99,235,.20); }}
:root[data-theme="light"] .cp-filter-seg button[data-status="added"].active {{ background: #e9f8ee; border-color: #1a7f37; color: #1a7f37; }}
:root[data-theme="light"] .cp-filter-seg button[data-status="removed"].active {{ background: #fff0f0; border-color: #cf222e; color: #b42318; }}
:root[data-theme="light"] .cp-filter-seg button[data-status="modified"].active {{ background: #fff7df; border-color: #9a6700; color: #7a5200; }}
:root[data-theme="light"] .cp-row {{ color: #1f2937; border-color: #e3e8ef; }}
:root[data-theme="light"] .cp-row.selected {{ background: linear-gradient(90deg, #dbeafe, #eef5ff 72%); box-shadow: inset 0 0 0 1px rgba(37,99,235,.22); }}
:root[data-theme="light"] .cp-row.selected .cp-name {{ color: #1e3a8a; }}
:root[data-theme="light"] .cp-row.selected .cp-path {{ color: #475569; }}
:root[data-theme="light"] .cp-row-main .cp-path {{ color: #64748b; }}
:root[data-theme="light"] .cp-detail,
:root[data-theme="light"] .ov-help {{ border-color: #e3e8ef; }}
:root[data-theme="light"] #canvas-overlay {{ background: rgba(255,255,255,.94); border-color: #d7dee8; color: #64748b; box-shadow: 0 8px 24px rgba(30,41,59,.10); }}
:root[data-theme="light"] .ov-help summary {{ color: #475569; }}
:root[data-theme="light"] .tooltip {{ background: #ffffff; border-color: #d7dee8; color: #1f2937; box-shadow: 0 6px 20px rgba(30,41,59,.16); }}
:root[data-theme="light"] .tooltip .tt-name {{ color: #2563eb; }}
:root[data-theme="light"] .tooltip .tt-type {{ color: #64748b; }}
:root[data-theme="light"] .link {{ stroke: #94a3b8; }}
:root[data-theme="light"] .node circle,
:root[data-theme="light"] .dep-node circle {{ stroke: #94a3b8; }}
:root[data-theme="light"] g.change-added > .change-status-dot {{ fill: var(--change-added-strong); stroke: var(--change-added); }}
:root[data-theme="light"] g.change-modified > .change-status-dot {{ fill: var(--change-modified-strong); stroke: var(--change-modified); }}
:root[data-theme="light"] .node text,
:root[data-theme="light"] .dep-node text,
:root[data-theme="light"] .map-feat-node text,
:root[data-theme="light"] .map-dep-node text {{ fill: #334155 !important; }}
:root[data-theme="light"] .node.feat-selected > .node-label {{ fill: #1d4ed8 !important; paint-order: stroke; stroke: #fff; stroke-width: 3px; }}
:root[data-theme="light"] .node.feat-related > .node-label {{ fill: #0f172a !important; font-weight: 700; }}
:root[data-theme="light"] .change-graph-node .node-label {{ fill: #263548 !important; stroke: #f6f8fb !important; stroke-width: 4px; font-weight: 650; }}
:root[data-theme="light"] .change-graph-node circle {{ stroke: #64748b; }}
:root[data-theme="light"] .edge-default,
:root[data-theme="light"] .dep-link-default {{ stroke: #94a3b8; }}
:root[data-theme="light"] .change-dim {{ opacity: .70; }}
:root[data-theme="light"] .change-dim > circle:not(.change-focus-ring):not(.change-status-dot) {{ fill: #b8c3d1 !important; stroke: #64748b !important; }}
:root[data-theme="light"] .change-dim text {{ fill: #334155 !important; }}
:root[data-theme="light"] .node.change-focused text:not(.change-status-glyph),
:root[data-theme="light"] .dep-node.change-focused text:not(.change-status-glyph),
:root[data-theme="light"] .feat-ghost.focused .change-node-label,
:root[data-theme="light"] .dep-ghost.focused .change-node-label {{ fill: #1d4ed8 !important; stroke: #fff !important; stroke-width: 4px; }}
:root[data-theme="light"] .feat-ghost .change-node-label,
:root[data-theme="light"] .dep-ghost .change-node-label {{ stroke: #f6f8fb; stroke-width: 4px; }}
:root[data-theme="light"] .cp-detail {{ background: #f8fafc; border-color: #cbd5e1; color: #475569; }}
:root[data-theme="light"] .cp-detail.selected.added {{ border-left-color: #1a7f37; }}
:root[data-theme="light"] .cp-detail.selected.removed {{ border-left-color: #cf222e; }}
:root[data-theme="light"] .cp-detail.selected.modified {{ border-left-color: #9a6700; }}
:root[data-theme="light"] .feat-ghost text,
:root[data-theme="light"] .dep-ghost text {{ fill: #b42318; }}
:root[data-theme="light"] .feat-ghost-link,
:root[data-theme="light"] .dep-ghost-link {{ stroke: #c2413a; stroke-opacity: .68; }}
:root[data-theme="light"] .feat-ghost .change-status-dot,
:root[data-theme="light"] .dep-ghost .change-status-dot {{ fill: var(--change-removed-strong); stroke: var(--change-removed); }}
:root[data-theme="light"] #search::placeholder,
:root[data-theme="light"] #cp-search::placeholder {{ color: #64748b; opacity: 1; }}
:root[data-theme="light"] #status-seg button:not(.active),
:root[data-theme="light"] .cp-filter-seg button:not(.active) {{ color: #475569; }}
:root[data-theme="light"] .cp-filter-seg button b {{ color: #64748b; }}
:root[data-theme="light"] .cp-filter-seg button.active b {{ color: #1f2937; }}
:root[data-theme="light"] .no-data {{ color: #94a3b8; }}
#d3-offline {{ display: none; position: fixed; inset: 78px 0 0; z-index: 120; place-items: center; padding: 32px; background: #0d1117; color: #c9d1d9; text-align: center; }}
.d3-unavailable #d3-offline {{ display: grid; }}
#d3-offline strong {{ display: block; margin-bottom: 8px; font-size: 16px; color: #f0f6fc; }}
#d3-offline p {{ max-width: 620px; color: #8b949e; line-height: 1.6; }}
#d3-offline code {{ color: #79c0ff; }}
:root[data-theme="light"] #d3-offline {{ background: #f6f8fb; color: #1f2937; }}
:root[data-theme="light"] #d3-offline strong {{ color: #1f2937; }}
:root[data-theme="light"] #d3-offline p {{ color: #475569; }}
:root[data-theme="light"] #d3-offline code {{ color: #1d4ed8; }}
@media (max-width: 600px) {{
  #changes-panel {{ top: 136px; width: min(300px, 82vw); }}
  #canvas-overlay {{ top: 146px; }}
  #d3-offline {{ inset: 136px 0 0; }}
}}
</style>
</head>
<body>
<div id="header">
  <div id="header-row1">
  <h1>RPG: {repo_name}</h1>
  <div id="tabs">
    <button id="tab-feat" class="active" onclick="switchTab('feat')">Feat Graph</button>
    <button id="tab-dep" onclick="switchTab('dep')">Dep Graph</button>
    <button id="tab-map" onclick="openMapping()">Mapping</button>
  </div>
  <div id="stats-feat">
    <span class="stat" title="Non-hierarchy edge types: {feat_edge_summary}"><b>{feat_node_count}</b> tree nodes · <b>{feat_edge_count}</b> semantic edges</span>
  </div>
  <div id="stats-dep" style="display:none">
    <span class="stat" id="dep-visible-stat"></span>
    <span class="stat" title="Connected semantic subgraph. Raw graph including contains edges: {raw_dep_node_count} nodes / {raw_dep_edge_count} edges. Relation types: {dep_edge_summary}">Full: <b>{dep_node_count}</b> nodes · <b>{dep_edge_count}</b> relations</span>
  </div>
  <div id="stats-map" style="display:none">
    <span class="stat"><b>{len(dep_to_rpg)}</b> mapped dep nodes · <b>{map_count}</b> mapping relations</span>
  </div>
  <div id="controls">
    <input id="search" type="text" placeholder="Search nodes...">
    <span id="feat-controls">
      <button onclick="expandToDepth(1)">L1</button>
      <button onclick="expandToDepth(2)">L2</button>
      <button onclick="expandToDepth(3)">L3</button>
      <button onclick="expandAll()">All</button>
      <button onclick="expandMore()">+</button>
      <button onclick="expandLess()">−</button>
    </span>
    <span id="dep-controls" style="display:none">
      <button onclick="depExpandToDepth(1)">L1</button>
      <button onclick="depExpandToDepth(2)">L2</button>
      <button onclick="depExpandToDepth(3)">L3</button>
      <button onclick="depExpandAll()">All</button>
      <button onclick="depExpandMore()">+</button>
      <button onclick="depExpandLess()">−</button>
    </span>
    <span id="map-controls" style="display:none">
      <button onclick="mapExpandToDepth(1)">L1</button>
      <button onclick="mapExpandToDepth(2)">L2</button>
      <button onclick="mapExpandToDepth(3)">L3</button>
      <button onclick="mapExpandAll()">All</button>
      <button onclick="mapExpandMore()">+</button>
      <button onclick="mapExpandLess()">−</button>
    </span>
    <button id="btn-edges" class="active" onclick="toggleEdges()">Edges</button>
    <button id="btn-fit" onclick="fitCurrent()">Fit</button>
    <button id="btn-reset" onclick="resetCurrent()">Reset</button>
    <button id="btn-theme" onclick="toggleRpgTheme()" title="Toggle light / dark theme" aria-label="Toggle theme">☾</button>
    <button id="btn-changes-panel" onclick="toggleChangesPanel()" style="display:none">Changes ⟩</button>
  </div>
  <div id="legend">
  </div>
  </div>
  <div id="header-row2" style="display:none">
    <div class="hdr-group"><span class="hdr-label">View</span>
      <div id="mode-seg" class="seg2">
        <button id="mode-changes" class="active" onclick="setMode('changes')">Current Changes</button>
        <button id="mode-full" onclick="setMode('full')">Full Graph</button>
      </div>
    </div>
  </div>
</div>
<aside id="changes-panel" style="display:none">
  <div id="cp-head">
    <div><div><strong id="cp-title">Changes</strong> <span id="cp-count" class="cp-count">0</span></div><small id="cp-version"></small></div>
    <button id="cp-collapse" onclick="toggleChangesPanel()" title="Hide changes" aria-label="Hide changes">Hide</button>
  </div>
  <div id="status-group" class="cp-filters">
    <div id="status-seg" class="cp-filter-seg">
      <button data-status="all" class="active" onclick="setStatusFilter('all')"><span class="cp-filter-main"><span class="cp-filter-icon">Δ</span><span>All changes</span></span><b id="cnt-all">0</b></button>
      <button data-status="added" onclick="setStatusFilter('added')"><span class="cp-filter-main"><span class="cp-filter-icon">+</span><span>Added</span></span><b id="cnt-added">0</b></button>
      <button data-status="removed" onclick="setStatusFilter('removed')"><span class="cp-filter-main"><span class="cp-filter-icon">−</span><span>Removed</span></span><b id="cnt-removed">0</b></button>
      <button data-status="modified" onclick="setStatusFilter('modified')"><span class="cp-filter-main"><span class="cp-filter-icon">~</span><span>Modified</span></span><b id="cnt-modified">0</b></button>
    </div>
  </div>
  <label id="cp-search-wrap"><span>⌕</span><input id="cp-search" type="text" placeholder="Search changed nodes"></label>
  <div id="cp-detail" class="cp-detail empty">Select a changed node to see what changed.</div>
  <div id="cp-list"></div>
</aside>
<div id="tooltip" class="tooltip" style="display:none"></div>
<div id="change-summary" hidden></div>
<div id="d3-offline"><div><strong>Graph library unavailable offline</strong><p>This report opened successfully, but the interactive RPG graph requires a local <code>assets/d3.v7.min.js</code>. No development server is required once that fixed asset is included.</p></div></div>
<div id="canvas-overlay">
  <div class="ov-title">Legend</div>
  <div class="legend-item"><div class="legend-line" style="background:#f0883e"></div>imports</div>
  <div class="legend-item"><div class="legend-line" style="background:#a371f7"></div>inherits</div>
  <div class="legend-item"><div class="legend-line" style="background:#3fb950"></div>invokes</div>
  <div class="legend-item"><div class="legend-line" style="background:#79c0ff"></div>references</div>
  <details class="ov-help"><summary>Controls</summary><div>
    <span class="ov-key">Click</span> — select / filter<br>
    <span class="ov-key">Double-click</span> — expand / collapse<br>
    <span class="ov-key">Drag</span> — move · <span class="ov-key">Scroll</span> — zoom
  </div></details>
</div>
<svg id="canvas"></svg>

<script>
if (window.d3) {{
// ── Data ──
const treeData = {tree_json};
const semanticEdges = {edges_json};
const depNodesRaw = {dep_nodes_json};
const depEdgesRaw = {dep_edges_json};
const depParentMap = {dep_parent_json};
const depTreeData = {dep_tree_json};
const depToRpgMap = {dep_to_rpg_json};
const hasDep = {'true' if has_dep else 'false'};
const hasMap = {'true' if has_map else 'false'};
const changeData = {change_json};
const hasChangeData = {has_change};

function applyRpgTheme(theme, persist) {{
  const value = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', value);
  const button = document.getElementById('btn-theme');
  if (button) button.textContent = value === 'light' ? '☀' : '☾';
  if (persist) {{
    try {{ localStorage.setItem('cmind-report-theme', value); }} catch (e) {{}}
    if (window.parent !== window) window.parent.postMessage({{type: 'cmind:theme-change', theme: value}}, '*');
  }}
}}
function toggleRpgTheme() {{
  applyRpgTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light', true);
}}
applyRpgTheme(document.documentElement.getAttribute('data-theme'), false);
window.addEventListener('storage', event => {{
  if (event.key === 'cmind-report-theme' && event.newValue) applyRpgTheme(event.newValue, false);
}});

const nodeTypeColors = {{
  root: '#8b949e', repository: '#8b949e', repo: '#8b949e',
  directory: '#1f6feb', feature_area: '#1f6feb',
  file: '#3fb950', module: '#3fb950',
  class: '#a371f7', function: '#d2a8ff',
  method: '#d2a8ff', variable: '#79c0ff',
  default: '#484f58',
}};

const edgeClassMap = {{
  imports: 'edge-imports', inherits: 'edge-inherits',
  invokes: 'edge-invokes', references: 'edge-references',
}};

const depEdgeColors = {{
  imports: '#f0883e', invokes: '#3fb950', inherits: '#a371f7',
}};
const depEdgeClassMap = {{
  imports: 'dep-link-imports', invokes: 'dep-link-invokes',
  inherits: 'dep-link-inherits',
}};

const margin = {{ top: 50, right: 200, bottom: 20, left: 80 }};
const svg = d3.select('#canvas');
const width = window.innerWidth;
const height = window.innerHeight;

// ── Arrow markers for feat graph edges ──
const defs = svg.append('defs');
const arrowColors = {{
  'edge-imports': '#f0883e',
  'edge-inherits': '#a371f7',
  'edge-invokes': '#3fb950',
  'edge-references': '#79c0ff',
  'edge-default': '#8b949e',
}};
Object.entries(arrowColors).forEach(([cls, color]) => {{
  defs.append('marker')
    .attr('id', 'arrow-' + cls)
    .attr('viewBox', '0 0 10 6')
    .attr('refX', 10).attr('refY', 3)
    .attr('markerWidth', 8).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,0 L10,3 L0,6 Z')
    .attr('fill', color);
}});
const directedEdgeColors = {{
  imports: '#f0883e', invokes: '#3fb950', inherits: '#a371f7', references: '#79c0ff',
  contains: '#8b949e', maps_to: '#f0883e', default: '#8b949e',
}};
Object.entries(directedEdgeColors).forEach(([relation, color]) => {{
  defs.append('marker')
    .attr('id', 'arrow-rel-' + relation)
    .attr('viewBox', '0 0 10 6')
    .attr('refX', 10).attr('refY', 3)
    .attr('markerWidth', 7).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,0 L10,3 L0,6 Z')
    .attr('fill', color);
}});

// ── Tab state ──
let activeTab = 'feat';
let showEdges = true;
let externalChanges = {{
  active: false, mode: 'full', filter: 'all', contextMode: 'context', emphasize: false,
  feature: {{ added: new Set(), removed: new Set(), modified: new Set(), rows: {{}} }},
  dependency: {{ added: new Set(), removed: new Set(), modified: new Set(), rows: {{}} }},
}};
const originalGraphUi = {{
  featStats: document.getElementById('stats-feat').innerHTML,
  depStats: document.getElementById('stats-dep').innerHTML,
  mapStats: document.getElementById('stats-map').innerHTML,
  overlay: document.getElementById('canvas-overlay').innerHTML,
}};

// ══════════════════════════════════════════════
// FEAT GRAPH (existing tree layout)
// ══════════════════════════════════════════════

const gFeat = svg.append('g').attr('class', 'feat-group').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
const edgeLayer = gFeat.append('g').attr('class', 'edge-layer');
const linkLayer = gFeat.append('g').attr('class', 'link-layer');
const nodeLayer = gFeat.append('g').attr('class', 'node-layer');
const featGhostLayer = gFeat.append('g').attr('class', 'feat-ghost-layer');

const zoomFeat = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => gFeat.attr('transform', e.transform));

const gChange = svg.append('g').attr('class', 'change-group').style('display', 'none');
const changeLinkLayer = gChange.append('g').attr('class', 'change-links');
const changeNodeLayer = gChange.append('g').attr('class', 'change-nodes');
const zoomChange = d3.zoom().scaleExtent([0.08, 5]).on('zoom', e => gChange.attr('transform', e.transform));
let changeSimulation = null;
const gRemovedRail = svg.append('g').attr('class', 'removed-rail').style('display', 'none');

let featSelectedNodeId = null;
const root = d3.hierarchy(treeData, d => d.children);
root.descendants().forEach(d => {{
  if (d.depth >= 1 && d.children) {{
    d._children = d.children;
    d.children = null;
  }}
}});

const treemap = d3.tree().nodeSize([18, 220]);
let nodeById = {{}};

function getNodeColor(d) {{
  const tn = d.data.meta?.type_name || d.data.node_type || 'default';
  return nodeTypeColors[tn] || nodeTypeColors.default;
}}

function featSelectionState() {{
  const selectedId = featSelectedNodeId == null ? null : String(featSelectedNodeId);
  const relatedIds = new Set();
  if (!selectedId) return {{selectedId, relatedIds}};
  const selected = nodeById[selectedId];
  if (selected) {{
    if (selected.parent) relatedIds.add(String(selected.parent.data.id));
    [...(selected.children || []), ...(selected._children || [])].forEach(child => relatedIds.add(String(child.data.id)));
  }}
  semanticEdges.forEach(edge => {{
    const sourceId = String(edge.src), targetId = String(edge.dst);
    if (sourceId === selectedId) relatedIds.add(targetId);
    if (targetId === selectedId) relatedIds.add(sourceId);
  }});
  relatedIds.delete(selectedId);
  return {{selectedId, relatedIds}};
}}

function featApplySelectionHighlight() {{
  const {{selectedId, relatedIds}} = featSelectionState();
  nodeLayer.selectAll('g.node')
    .classed('feat-selected', node => !!selectedId && String(node.data.id) === selectedId)
    .classed('feat-related', node => !!selectedId && relatedIds.has(String(node.data.id)))
    .classed('feat-dim', node => !!selectedId && String(node.data.id) !== selectedId && !relatedIds.has(String(node.data.id)));
  linkLayer.selectAll('path.link')
    .classed('feat-related', edge => !!selectedId && (String(edge.source.data.id) === selectedId || String(edge.target.data.id) === selectedId))
    .classed('feat-dim', edge => !!selectedId && String(edge.source.data.id) !== selectedId && String(edge.target.data.id) !== selectedId);
  edgeLayer.selectAll('path.semantic-edge')
    .classed('feat-related', edge => !!selectedId && (String(edge.src) === selectedId || String(edge.dst) === selectedId))
    .classed('feat-dim', edge => !!selectedId && String(edge.src) !== selectedId && String(edge.dst) !== selectedId);
}}

function update(source) {{
  const treeDataLayout = treemap(root);
  const nodes = treeDataLayout.descendants();
  const links = treeDataLayout.links();
  nodes.forEach(d => {{ d.y = d.depth * 220; }});

  nodeById = {{}};
  nodes.forEach(d => {{ nodeById[d.data.id] = d; }});
  if (featSelectedNodeId && !nodeById[String(featSelectedNodeId)]) featSelectedNodeId = null;

  const node = nodeLayer.selectAll('g.node').data(nodes, d => d.data.id);
  const nodeEnter = node.enter().append('g')
    .attr('class', d => 'node' + (d._children ? ' node-collapsed' : ''))
    .attr('transform', `translate(${{source.y0 || 0}},${{source.x0 || 0}})`)
    .on('click', (event, d) => {{
      event.stopPropagation();
      const changeKind = externalKind(String(d.data.id), externalChanges.feature);
      if (changeKind) {{ focusChangeNode(String(d.data.id), 'feature'); return; }}
      // Single click: select/deselect node
      const nodeId = String(d.data.id);
      featSelectedNodeId = featSelectedNodeId === nodeId ? null : nodeId;
      featApplySelectionHighlight();
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      update(d);
    }})
    .on('mouseover', showTooltipFeat)
    .on('mouseout', hideTooltip);

  nodeEnter.append('circle').attr('class', 'node-dot')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', getNodeColor);
  nodeEnter.append('circle').attr('class', 'change-status-dot').attr('r', 6);
  nodeEnter.append('text').attr('class', 'change-status-glyph').attr('dy', 3).text(d => externalChanges.feature.added.has(String(d.data.id)) ? '+' : '~');
  nodeEnter.append('circle').attr('class', 'change-focus-ring').attr('r', 9).style('display', 'none');
  nodeEnter.append('text').attr('class', 'node-label')
    .attr('dy', 3.5)
    .attr('x', -10)
    .attr('text-anchor', 'end')
    .text(d => {{
      const name = d.data.name || d.data.id;
      return name.length > 35 ? name.slice(0, 33) + '...' : name;
    }});

  const nodeUpdate = nodeEnter.merge(node);
  const featureFocusId = externalChanges.focus?.scope === 'feature' ? String(externalChanges.focus.node_id) : null;
  nodeUpdate.attr('class', d => 'node' + (d._children ? ' node-collapsed' : '')
    + (featureFocusId === String(d.data.id) ? ' change-focused' : ''));
  nodeUpdate.transition().duration(300)
    .attr('transform', d => `translate(${{d.y}},${{d.x}})`);
  nodeUpdate.select('.node-dot')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', getNodeColor);
  nodeUpdate.select('.change-focus-ring').attr('r', 9)
    .style('display', d => featureFocusId === String(d.data.id) ? null : 'none');
  nodeUpdate.select('.node-label')
    .attr('x', -10)
    .attr('text-anchor', 'end');

  node.exit().transition().duration(200)
    .attr('transform', `translate(${{source.y}},${{source.x}})`)
    .remove();

  const link = linkLayer.selectAll('path.link').data(links, d => d.target.data.id);
  const linkEnter = link.enter().insert('path', 'g')
    .attr('class', 'link')
    .attr('d', () => {{
      const o = {{ x: source.x0 || 0, y: source.y0 || 0 }};
      return diagonal(o, o);
    }});
  linkEnter.merge(link).transition().duration(300)
    .attr('d', d => diagonal(d.source, d.target));
  link.exit().transition().duration(200)
    .attr('d', () => {{
      const o = {{ x: source.x, y: source.y }};
      return diagonal(o, o);
    }}).remove();

  drawSemanticEdges();
  featApplySelectionHighlight();
  applyExternalChangeHighlights();
  if (externalChanges.active) setTimeout(applyExternalChangeHighlights, 350);
  nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
}}

function diagonal(s, d) {{
  return `M${{s.y}},${{s.x}} C${{(s.y + d.y) / 2}},${{s.x}} ${{(s.y + d.y) / 2}},${{d.x}} ${{d.y}},${{d.x}}`;
}}

// Click background to deselect feat node
svg.on('click.feat-deselect', event => {{
  if (activeTab !== 'feat' || event.target !== svg.node()) return;
  featSelectedNodeId = null;
  featApplySelectionHighlight();
}});

function drawSemanticEdges() {{
  edgeLayer.selectAll('path.semantic-edge').remove();
  if (!showEdges) return;
  semanticEdges.forEach(e => {{
    const src = nodeById[e.src];
    const dst = nodeById[e.dst];
    if (!src || !dst) return;
    const rel = (e.relation || 'default').toLowerCase();
    const cls = edgeClassMap[rel] || 'edge-default';
    const sx = dst.y + 6, sy = dst.x;
    const dx = src.y - 6, dy = src.x;
    const midY = (sy + dy) / 2;
    // Curvature scales with vertical distance between endpoints,
    // so far-apart leaf edges bow out more and don't overlap nearby ones.
    const vertDist = Math.abs(sy - dy);
    const bulge = 60 + vertDist * 0.35;
    edgeLayer.append('path').datum(e)
      .attr('class', 'semantic-edge ' + cls)
      .attr('d', `M${{sx}},${{sy}} Q${{Math.max(sx, dx) + bulge}},${{midY}} ${{dx}},${{dy}}`)
      .attr('fill', 'none')
      .attr('stroke-width', 1.2)
      .attr('marker-end', `url(#arrow-${{cls}})`);
  }});
  featApplySelectionHighlight();
}}

function showTooltipFeat(event, d) {{
  const tip = document.getElementById('tooltip');
  const meta = d.data.meta || {{}};
  const tn = meta.type_name || d.data.node_type || '';
  const path = meta.path || '';
  const desc = meta.description || '';
  const connected = semanticEdges.filter(e => e.src === d.data.id || e.dst === d.data.id);
  const edgeInfo = connected.length > 0
    ? `<div class="tt-edges">${{connected.length}} edge(s): ${{
        [...new Set(connected.map(e => e.relation))].join(', ')
      }}</div>` : '';
  tip.innerHTML = `
    <div class="tt-name">${{d.data.name || d.data.id}}</div>
    ${{tn ? `<div class="tt-type">${{tn}}</div>` : ''}}
    ${{path && path !== '.' ? `<div class="tt-path">${{path}}</div>` : ''}}
    ${{desc ? `<div style="color:#8b949e;font-size:11px;margin-top:2px">${{desc.slice(0, 200)}}</div>` : ''}}
    ${{edgeInfo}}
    ${{d._children ? `<div style="color:#1f6feb;font-size:11px">${{d._children.length}} children (collapsed)</div>` : ''}}
  `;
  tip.style.display = 'block';
  tip.style.left = (event.clientX + 12) + 'px';
  tip.style.top = (event.clientY - 10) + 'px';
}}

function hideTooltip() {{
  document.getElementById('tooltip').style.display = 'none';
}}

// ── Feat controls ──
function expandToDepth(maxDepth) {{
  root.descendants().forEach(d => {{
    if (d._children && d.depth < maxDepth) {{ d.children = d._children; d._children = null; }}
    else if (d.children && d.depth >= maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  update(root);
}}
function expandAll() {{
  function visitAll(d) {{
    if (d._children) {{ d.children = d._children; d._children = null; }}
    if (d.children) d.children.forEach(visitAll);
  }}
  visitAll(root);
  update(root);
}}
function expandMore() {{
  // Expand one more level: find the shallowest collapsed depth, expand that level
  let minDepth = Infinity;
  root.descendants().forEach(d => {{ if (d._children && d.depth < minDepth) minDepth = d.depth; }});
  if (minDepth === Infinity) return;
  root.descendants().forEach(d => {{
    if (d._children && d.depth === minDepth) {{ d.children = d._children; d._children = null; }}
  }});
  update(root);
}}
function expandLess() {{
  // Collapse one level: find the deepest expanded depth that has children, collapse that level
  let maxDepth = -1;
  root.descendants().forEach(d => {{ if (d.children && d.children.length > 0 && d.depth > maxDepth) maxDepth = d.depth; }});
  if (maxDepth <= 0) return;
  root.descendants().forEach(d => {{
    if (d.children && d.children.length > 0 && d.depth === maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  update(root);
}}
function toggleEdges() {{
  showEdges = !showEdges;
  document.getElementById('btn-edges').classList.toggle('active', showEdges);
  if (activeTab === 'feat') drawSemanticEdges();
  else if (activeTab === 'dep') changeLinkLayer.style('display', showEdges ? null : 'none');
  else if (activeTab === 'map') mapDrawLinks();
}}

// ══════════════════════════════════════════════
// DEP GRAPH — collapsible force layout with group hulls
// ══════════════════════════════════════════════

const gDep = svg.append('g').attr('class', 'dep-group').style('display', 'none');
const depHullG = gDep.append('g').attr('class', 'dep-hulls');
const depLinkG = gDep.append('g').attr('class', 'dep-links');
const depNodeG = gDep.append('g').attr('class', 'dep-nodes');
const depLabelG = gDep.append('g').attr('class', 'dep-labels');
const depChangeGroupG = gDep.append('g').attr('class', 'dep-change-groups');
const depGhostG = gDep.append('g').attr('class', 'dep-ghosts');
const zoomDep = d3.zoom().scaleExtent([0.05, 4]).on('zoom', e => gDep.attr('transform', e.transform));

// Arrow markers
svg.append('defs').selectAll('marker')
  .data(['imports', 'invokes', 'inherits', 'default'])
  .join('marker')
    .attr('id', d => 'arrow-' + d)
    .attr('viewBox', '0 -3 6 6')
    .attr('refX', 14)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
  .append('path')
    .attr('d', 'M0,-3L6,0L0,3')
    .attr('fill', d => depEdgeColors[d] || '#8b949e');

// Hull depth colors (semi-transparent, layered)
const hullColors = [
  'rgba(30,80,160,0.08)',  // depth 0
  'rgba(40,100,180,0.07)', // depth 1
  'rgba(50,120,200,0.06)', // depth 2
  'rgba(60,140,220,0.05)', // depth 3+
];
const hullBorderColors = [
  'rgba(48,54,61,0.6)',
  'rgba(48,54,61,0.4)',
  'rgba(48,54,61,0.3)',
  'rgba(48,54,61,0.2)',
];

// ── Dep graph data structures ──
let depInitialized = false;
let depSimulation = null;
let pendingDepFocusId = null;

const depNodeMap = {{}};
const depChildrenOf = {{}};
const depCollapsed = new Set();
let depAllDescendants = {{}};

// Runtime: maps for visible nodes
let depVisNodeDataMap = {{}};
let depSelectedNodes = new Set();  // currently selected node ids for edge filtering

function depInit() {{
  if (depInitialized || !hasDep) return;
  depInitialized = true;

  depNodesRaw.forEach(n => {{ depNodeMap[n.id] = n; }});

  for (const [child, parent] of Object.entries(depParentMap)) {{
    if (!depChildrenOf[parent]) depChildrenOf[parent] = [];
    depChildrenOf[parent].push(child);
  }}

  function getAllDescendants(id) {{
    if (depAllDescendants[id]) return depAllDescendants[id];
    const result = new Set();
    const children = depChildrenOf[id] || [];
    for (const cid of children) {{
      result.add(cid);
      for (const did of getAllDescendants(cid)) result.add(did);
    }}
    depAllDescendants[id] = result;
    return result;
  }}
  Object.keys(depNodeMap).forEach(id => getAllDescendants(id));

  // Default: collapse to depth 1
  depNodesRaw.forEach(n => {{
    if ((depChildrenOf[n.id] || []).length > 0 && n.depth >= 1) {{
      depCollapsed.add(n.id);
    }}
  }});

  depRedraw();
}}

// ── Visibility logic ──

function depIsVisible(id) {{
  let cur = depParentMap[id];
  while (cur) {{
    if (depCollapsed.has(cur)) return false;
    cur = depParentMap[cur];
  }}
  return true;
}}

function depGetVisibleRep(id) {{
  let cur = id;
  while (depParentMap[cur] && depCollapsed.has(depParentMap[cur])) {{
    cur = depParentMap[cur];
  }}
  return cur;
}}

function depGetVisibleNodes() {{
  const visible = [];
  depNodesRaw.forEach(n => {{
    if (depIsVisible(n.id)) visible.push(n);
  }});
  return visible;
}}

function depGetMergedEdges(visibleIds) {{
  const visSet = new Set(visibleIds);
  const seen = new Set();
  const merged = [];

  depEdgesRaw.forEach(e => {{
    let src = e.source;
    let dst = e.target;
    if (typeof src === 'object') src = src.id;
    if (typeof dst === 'object') dst = dst.id;

    const visSrc = depGetVisibleRep(src);
    const visDst = depGetVisibleRep(dst);

    if (!visSet.has(visSrc) || !visSet.has(visDst)) return;
    if (visSrc === visDst) return;

    const key = `${{visSrc}}|${{visDst}}|${{e.type}}`;
    if (seen.has(key)) return;
    seen.add(key);

    merged.push({{ source: visSrc, target: visDst, type: e.type }});
  }});

  return merged;
}}

// ── Convex hull helper ──
// Build a rounded hull: expand each point into a circle of sample points,
// then compute convex hull. Result is always smooth/rounded.
function paddedHull(points, pad) {{
  if (points.length < 1) return null;
  // Expand each point into 8 samples on a circle of radius=pad
  const expanded = [];
  const steps = 8;
  for (const p of points) {{
    for (let i = 0; i < steps; i++) {{
      const angle = (2 * Math.PI * i) / steps;
      expanded.push([p[0] + Math.cos(angle) * pad, p[1] + Math.sin(angle) * pad]);
    }}
  }}
  if (expanded.length < 3) return expanded;
  return d3.polygonHull(expanded);
}}

// ── Find which expanded parents need hulls ──
function depGetExpandedGroups() {{
  // An expanded group is a node that has children AND is not collapsed
  // AND is visible (not hidden by a collapsed ancestor)
  const groups = [];
  depNodesRaw.forEach(n => {{
    const children = depChildrenOf[n.id] || [];
    if (children.length > 0 && !depCollapsed.has(n.id) && depIsVisible(n.id)) {{
      groups.push(n);
    }}
  }});
  // Sort by depth descending so inner hulls draw first (outer hulls on top visually
  // but since we want outer hulls behind, we draw deepest last — actually we want
  // outer behind, so sort ascending: draw outermost first (background), innermost on top
  groups.sort((a, b) => a.depth - b.depth);
  return groups;
}}

// Collect all visible descendant positions for a group (recursively)
function depGroupChildPositions(groupId) {{
  const points = [];
  const stack = [groupId];
  while (stack.length) {{
    const gid = stack.pop();
    const children = depChildrenOf[gid] || [];
    for (const cid of children) {{
      const nd = depVisNodeDataMap[cid];
      if (nd && nd.x !== undefined) {{
        points.push([nd.x, nd.y]);
      }}
      // If child is an expanded parent (not collapsed, has children), recurse
      // even if it's not in depVisNodeDataMap (expanded parents are hull-only)
      if (!depCollapsed.has(cid) && depChildrenOf[cid] && depChildrenOf[cid].length > 0) {{
        stack.push(cid);
      }}
    }}
  }}
  return points;
}}

// ── Redraw ──

function depRedraw() {{
  const visNodes = depGetVisibleNodes();
  const visIds = visNodes.map(n => n.id);
  const mergedEdges = depGetMergedEdges(visIds);

  // Identify expanded parent nodes (they become hull labels, not force nodes)
  const expandedParentIds = new Set();
  depNodesRaw.forEach(n => {{
    const children = depChildrenOf[n.id] || [];
    if (children.length > 0 && !depCollapsed.has(n.id) && depIsVisible(n.id)) {{
      expandedParentIds.add(n.id);
    }}
  }});

  // Force-layout nodes: visible nodes EXCEPT expanded parents
  // (expanded parents will be drawn as hull labels instead)
  const forceNodes = [];
  visNodes.forEach(n => {{
    if (expandedParentIds.has(n.id)) return;  // skip — will be a hull label
    const existing = depNodeMap[n.id];
    // If no prior position, spawn near parent's last position (not random center)
    let initX, initY;
    if (existing._x) {{
      initX = existing._x;
      initY = existing._y;
    }} else {{
      const pid = depParentMap[n.id];
      const parentPos = pid && depNodeMap[pid];
      if (parentPos && parentPos._x) {{
        initX = parentPos._x + (Math.random() - 0.5) * 60;
        initY = parentPos._y + (Math.random() - 0.5) * 60;
      }} else {{
        initX = width / 2 + (Math.random() - 0.5) * 400;
        initY = height / 2 + (Math.random() - 0.5) * 400;
      }}
    }}
    forceNodes.push({{
      id: n.id,
      name: n.name,
      type: n.type,
      module: n.module,
      rpg_nodes: n.rpg_nodes,
      depth: n.depth,
      hasChildren: (depChildrenOf[n.id] || []).length > 0,
      isCollapsed: depCollapsed.has(n.id),
      x: initX,
      y: initY,
    }});
  }});

  depVisNodeDataMap = {{}};
  forceNodes.forEach(n => {{ depVisNodeDataMap[n.id] = n; }});
  drawDepChangedGroups();
  drawDepRemovedGhosts();

  // Edges: filter to force-node endpoints only
  // For edges involving expanded parents, remap to the parent's visible rep
  const validEdges = mergedEdges.filter(e => depVisNodeDataMap[e.source] && depVisNodeDataMap[e.target]);
  const visibleStat = document.getElementById('dep-visible-stat');
  if (visibleStat) visibleStat.innerHTML = `<b>${{visNodes.length}}</b> visible nodes · <b>${{validEdges.length}}</b> drawn relations`;

  if (depSimulation) depSimulation.stop();

  // ── Links ──
  const linkSel = depLinkG.selectAll('line.dep-link')
    .data(validEdges, d => d.source + '|' + d.target + '|' + d.type);
  linkSel.exit().remove();
  const linkEnter = linkSel.enter().append('line')
    .attr('class', d => 'dep-link ' + (depEdgeClassMap[d.type] || 'dep-link-default'))
    .attr('marker-end', d => `url(#arrow-rel-${{directedEdgeColors[d.type] ? d.type : 'default'}})`)
    .style('display', showEdges ? null : 'none');
  const linkAll = linkEnter.merge(linkSel);

  // ── Nodes ──
  const nodeSel = depNodeG.selectAll('g.dep-node').data(forceNodes, d => d.id);
  nodeSel.exit().remove();

  const nodeEnter = nodeSel.enter().append('g')
    .attr('class', d => 'dep-node' + (d.isCollapsed ? ' dep-node-collapsed' : ''))
    .on('click', (event, d) => {{
      event.stopPropagation();
      // Toggle selection: click same node deselects
      if (depSelectedNodes.size === 1 && depSelectedNodes.has(d.id)) {{
        depSelectedNodes.clear();
      }} else {{
        depSelectedNodes.clear();
        depSelectedNodes.add(d.id);
      }}
      depUpdateEdgeVisibility();
      depUpdateNodeHighlight();
      const changed = externalKind(String(d.id), externalChanges.dependency);
      const removed = externalChanges.dependency.removed.has(String(d.id));
      if (changed || removed) focusChangeNode(String(d.id), 'dependency');
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      if (!d.hasChildren) return;
      if (depCollapsed.has(d.id)) {{
        depCollapsed.delete(d.id);
      }} else {{
        depCollapsed.add(d.id);
      }}
      depSelectedNodes.clear();
      depRedraw();
    }})
    .on('mouseover', showTooltipDep)
    .on('mouseout', hideTooltip)
    .call(d3.drag()
      .on('start', (event, d) => {{
        d._dragged = false;
        d.fx = d.x; d.fy = d.y;
      }})
      .on('drag', (event, d) => {{
        if (!d._dragged) {{
          d._dragged = true;
          if (!event.active) depSimulation.alphaTarget(0.3).restart();
        }}
        d.fx = event.x; d.fy = event.y;
      }})
      .on('end', (event, d) => {{
        if (d._dragged && !event.active) depSimulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      }}));

  nodeEnter.append('circle').attr('class', 'dep-node-dot');
  nodeEnter.append('circle').attr('class', 'change-status-dot').attr('r', 6);
  nodeEnter.append('text').attr('class', 'change-status-glyph').attr('dy', 3).text(d => externalChanges.dependency.added.has(String(d.id)) ? '+' : '~');
  nodeEnter.append('circle').attr('class', 'change-focus-ring').attr('r', 10).style('display', 'none');
  nodeEnter.append('text');

  const nodeAll = nodeEnter.merge(nodeSel);
  nodeAll.attr('class', d => 'dep-node' + (d.isCollapsed ? ' dep-node-collapsed' : ''));

  nodeAll.select('.dep-node-dot')
    .attr('r', d => {{
      if (d.isCollapsed) {{
        const desc = depAllDescendants[d.id];
        return Math.min(3 + Math.sqrt(desc ? desc.size : 1) * 1.5, 20);
      }}
      const t = d.type;
      return t === 'module' || t === 'file' ? 5 : t === 'class' ? 4.5 : 3.5;
    }})
    .attr('fill', d => nodeTypeColors[d.type] || nodeTypeColors.default);

  nodeAll.select('text')
    .attr('dx', d => {{
      if (d.isCollapsed) {{
        const desc = depAllDescendants[d.id];
        return Math.min(3 + Math.sqrt(desc ? desc.size : 1) * 1.5, 20) + 3;
      }}
      return 9;
    }})
    .attr('dy', 3.5)
    .text(d => {{
      const name = d.name || d.id;
      const suffix = d.isCollapsed ? ` (${{(depAllDescendants[d.id] || new Set()).size}})` : '';
      const label = name + suffix;
      return label.length > 30 ? label.slice(0, 28) + '...' : label;
    }});
  applyExternalChangeHighlights();
  if (depSelectedNodes.size > 0) depUpdateNodeHighlight();

  // ── Hierarchical cluster force: attraction decays with ancestor distance ──
  // Pre-compute ancestor chains for each force node (up to visible ancestors)
  const ancestorChains = {{}};
  forceNodes.forEach(n => {{
    const chain = [];
    let cur = n.id;
    while (cur) {{
      cur = depParentMap[cur];
      if (cur) chain.push(cur);
    }}
    ancestorChains[n.id] = chain;  // [parent, grandparent, great-grandparent, ...]
  }});

  // Find LCA depth between two nodes (higher depth = closer relationship)
  function lcaDepth(aId, bId) {{
    const aChain = ancestorChains[aId];
    const bChain = ancestorChains[bId];
    if (!aChain || !bChain) return 0;
    // Direct siblings: share parent at chain[0]
    const bAncSet = new Set(bChain);
    for (let i = 0; i < aChain.length; i++) {{
      if (bAncSet.has(aChain[i])) {{
        // LCA is aChain[i], which is i+1 steps up from a
        // Depth of LCA in the tree
        const lcaNode = depNodesRaw.find(n => n.id === aChain[i]);
        return lcaNode ? lcaNode.depth + 1 : 1;
      }}
    }}
    return 0;  // no common ancestor (root level)
  }}

  // Cache LCA depths (computed once per redraw, O(N²) but N = visible force nodes only)
  const lcaCache = {{}};
  function getCachedLcaDepth(aId, bId) {{
    const key = aId < bId ? aId + '|' + bId : bId + '|' + aId;
    if (lcaCache[key] === undefined) lcaCache[key] = lcaDepth(aId, bId);
    return lcaCache[key];
  }}

  // Find max depth among all force nodes for normalization
  const maxDepth = d3.max(forceNodes, n => {{
    const chain = ancestorChains[n.id];
    return chain ? chain.length : 0;
  }}) || 1;

  function clusterForce(alpha) {{
    // For each pair of force nodes, apply attraction proportional to LCA depth
    // To keep O manageable, use centroid-based approach per ancestor level
    // Level approach: for each ancestor, compute centroid of its visible descendants,
    // pull each descendant toward that centroid with strength proportional to depth

    // Collect centroids at each ancestor level
    const ancestorGroups = {{}};
    forceNodes.forEach(n => {{
      const chain = ancestorChains[n.id] || [];
      for (let i = 0; i < chain.length; i++) {{
        const ancId = chain[i];
        if (!ancestorGroups[ancId]) ancestorGroups[ancId] = {{ sx: 0, sy: 0, count: 0, depth: 0 }};
        ancestorGroups[ancId].sx += n.x;
        ancestorGroups[ancId].sy += n.y;
        ancestorGroups[ancId].count++;
      }}
    }});
    // Set depth for each ancestor
    depNodesRaw.forEach(n => {{
      if (ancestorGroups[n.id]) ancestorGroups[n.id].depth = n.depth;
    }});

    // Pull each node toward each ancestor's centroid, strength decays with distance in tree
    forceNodes.forEach(n => {{
      const chain = ancestorChains[n.id] || [];
      for (let i = 0; i < chain.length; i++) {{
        const ancId = chain[i];
        const g = ancestorGroups[ancId];
        if (!g || g.count < 2) continue;
        const cx = (g.sx - n.x) / (g.count - 1);
        const cy = (g.sy - n.y) / (g.count - 1);
        // Strength decays with tree distance: parent=strongest, grandparent=weaker, etc.
        // i=0 is parent, i=1 is grandparent, etc.
        const strength = 0.08 / (i + 1);
        n.vx += (cx - n.x) * alpha * strength;
        n.vy += (cy - n.y) * alpha * strength;
      }}
    }});
  }}

  // ── Simulation ──
  let depTickCount = 0;
  depSimulation = d3.forceSimulation(forceNodes)
    .force('link', d3.forceLink(validEdges).id(d => d.id).distance(d => {{
      const sp = depParentMap[typeof d.source === 'object' ? d.source.id : d.source];
      const tp = depParentMap[typeof d.target === 'object' ? d.target.id : d.target];
      return sp === tp ? 25 : 500;  // cross-group: large distance = no real constraint
    }}).strength(d => {{
      const sp = depParentMap[typeof d.source === 'object' ? d.source.id : d.source];
      const tp = depParentMap[typeof d.target === 'object' ? d.target.id : d.target];
      return sp === tp ? 0.8 : 0.05;  // cross-group links are very weak
    }}))
    .force('charge', d3.forceManyBody()
      .strength(d => d.isCollapsed ? -200 : -80)
      .distanceMax(300))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(d => {{
      if (d.isCollapsed) {{
        const desc = depAllDescendants[d.id];
        return Math.min(3 + Math.sqrt(desc ? desc.size : 1) * 1.5, 20) + 5;
      }}
      return 10;
    }}).strength(0.5))
    .force('cluster', clusterForce)
    .velocityDecay(0.35)
    .alphaDecay(0.03)
    .alpha(0.8)
    .on('tick', () => {{
      // Update links
      linkAll
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);

      // Update nodes
      nodeAll.attr('transform', d => {{
        depNodeMap[d.id]._x = d.x;
        depNodeMap[d.id]._y = d.y;
        return `translate(${{d.x}},${{d.y}})`;
      }});

      // Update hulls
      depDrawHulls();
      depTickCount++;
      if (activeTab === 'dep' && depTickCount % 8 === 0) {{
        applyExternalChangeHighlights();
        drawDepChangedGroups();
        drawDepRemovedGhosts();
      }}
    }})
    .on('end', () => {{
      if (activeTab !== 'dep') return;
      applyExternalChangeHighlights();
      if (depSelectedNodes.size > 0) depUpdateNodeHighlight();
      drawDepChangedGroups();
      drawDepRemovedGhosts();
      if (pendingDepFocusId) {{
        const id = pendingDepFocusId;
        pendingDepFocusId = null;
        focusDepRenderedNode(id);
      }} else depFitVisible();
    }});
}}

function depFitVisible() {{
  const bounds = gDep.node().getBBox();
  if (!bounds.width || !bounds.height) return;
  const padding = 100;
  const rightInset = (typeof cpOpen !== 'undefined' && cpOpen) ? 330 : 20;
  const usableWidth = Math.max(width - rightInset, 180);
  const scale = Math.min(
    usableWidth / (bounds.width + padding),
    height / (bounds.height + padding),
    1.15,
  ) * 0.88;
  svg.call(
    zoomDep.transform,
    d3.zoomIdentity
      .translate(
        usableWidth / 2 - (bounds.x + bounds.width / 2) * scale,
        height / 2 - (bounds.y + bounds.height / 2) * scale,
      )
      .scale(scale),
  );
}}

// ── Draw hulls for expanded groups ──

// ── Edge visibility based on selection ──
function depUpdateEdgeVisibility() {{
  const relatedNodes = new Set(depSelectedNodes);
  if (depSelectedNodes.size > 0) {{
    depLinkG.selectAll('line.dep-link').each(function(d) {{
      const sid = typeof d.source === 'object' ? d.source.id : d.source;
      const tid = typeof d.target === 'object' ? d.target.id : d.target;
      if (depSelectedNodes.has(sid) || depSelectedNodes.has(tid)) {{ relatedNodes.add(sid); relatedNodes.add(tid); }}
    }});
  }}
  depLinkG.selectAll('line.dep-link').each(function(d) {{
    if (!showEdges) {{
      d3.select(this).style('display', 'none');
      return;
    }}
    if (depSelectedNodes.size === 0) {{
      d3.select(this).style('display', null).style('opacity', null).style('stroke-width', null);
      return;
    }}
    const sid = typeof d.source === 'object' ? d.source.id : d.source;
    const tid = typeof d.target === 'object' ? d.target.id : d.target;
    if (depSelectedNodes.has(sid) || depSelectedNodes.has(tid)) {{
      d3.select(this).style('display', null).style('opacity', 0.95).style('stroke-width', 2.6);
    }} else {{
      d3.select(this).style('display', null).style('opacity', 0.06).style('stroke-width', null);
    }}
  }});
  return relatedNodes;
}}

// ── Node highlight based on selection ──
function depUpdateNodeHighlight() {{
  const relatedNodes = depUpdateEdgeVisibility();
  depNodeG.selectAll('g.dep-node').each(function(d) {{
    const el = d3.select(this);
    if (depSelectedNodes.size === 0) {{
      el.select('circle').style('stroke', null).style('stroke-width', null);
      el.style('opacity', 1);
      return;
    }}
    if (depSelectedNodes.has(d.id)) {{
      el.select('circle').style('stroke', '#58a6ff').style('stroke-width', 3);
      el.style('opacity', 1);
    }} else if (relatedNodes.has(d.id)) {{
      el.select('circle').style('stroke', null).style('stroke-width', null);
      el.style('opacity', 1);
    }} else {{
      el.select('circle').style('stroke', null).style('stroke-width', null);
      el.style('opacity', 0.16);
    }}
  }});
}}

// ── Click on background to deselect ──
svg.on('click.dep-deselect', () => {{
  if (depSelectedNodes.size > 0) {{
    depSelectedNodes.clear();
    depUpdateEdgeVisibility();
    depUpdateNodeHighlight();
  }}
}});

function depDrawHulls() {{
  const groups = depGetExpandedGroups();

  // Build hull data
  const hullData = [];
  groups.forEach(g => {{
    const points = depGroupChildPositions(g.id);
    if (points.length === 0) return;
    const hull = paddedHull(points, 25 + g.depth * 3);
    if (!hull) return;
    const di = Math.min(g.depth, hullColors.length - 1);
    hullData.push({{
      id: g.id,
      name: g.name,
      type: g.type,
      depth: g.depth,
      hull: hull,
      fill: hullColors[di],
      stroke: hullBorderColors[di],
      labelX: d3.min(hull, p => p[0]) + 8,
      labelY: d3.min(hull, p => p[1]) + 14,
    }});
  }});

  // Hull paths
  const hullSel = depHullG.selectAll('path.dep-hull').data(hullData, d => d.id);
  hullSel.exit().remove();
  const hullEnter = hullSel.enter().append('path')
    .attr('class', 'dep-hull')
    .style('cursor', 'pointer')
    .on('click', (event, d) => {{
      event.stopPropagation();
      const desc = depAllDescendants[d.id] || new Set();
      // Toggle: if already selected this group, deselect
      if (depSelectedNodes.size > 0 && depSelectedNodes.has(d.id)) {{
        depSelectedNodes.clear();
      }} else {{
        depSelectedNodes.clear();
        desc.forEach(id => depSelectedNodes.add(id));
        depSelectedNodes.add(d.id);
      }}
      depUpdateEdgeVisibility();
      depUpdateNodeHighlight();
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      depCollapsed.add(d.id);
      depSelectedNodes.clear();
      depRedraw();
    }})
    .on('mouseover', (event, d) => {{
      const tip = document.getElementById('tooltip');
      const descCount = (depAllDescendants[d.id] || new Set()).size;
      tip.innerHTML = `
        <div class="tt-name">${{d.name || d.id}}</div>
        <div class="tt-type">${{d.type}}</div>
        <div style="color:#8b949e;font-size:11px">Click to select · Double-click to collapse (${{descCount}} nodes)</div>
      `;
      tip.style.display = 'block';
      tip.style.left = (event.clientX + 12) + 'px';
      tip.style.top = (event.clientY - 10) + 'px';
    }})
    .on('mouseout', hideTooltip);

  hullEnter.merge(hullSel)
    .attr('d', d => 'M' + d.hull.map(p => p.join(',')).join('L') + 'Z')
    .attr('fill', d => d.fill)
    .attr('stroke', d => d.stroke)
    .attr('stroke-width', 1);

  // Hull labels (parent name on the boundary)
  const lblSel = depLabelG.selectAll('text.dep-hull-label').data(hullData, d => d.id);
  lblSel.exit().remove();
  const lblEnter = lblSel.enter().append('text')
    .attr('class', 'dep-hull-label')
    .attr('font-size', '11px')
    .attr('fill', '#58a6ff')
    .attr('text-anchor', 'start')
    .attr('pointer-events', 'none')
    .attr('opacity', 0.8);

  lblEnter.merge(lblSel)
    .attr('x', d => d.labelX)
    .attr('y', d => d.labelY)
    .text(d => {{
      const name = d.name || d.id;
      return name.length > 40 ? name.slice(0, 38) + '...' : name;
    }});
}}

function showTooltipDep(event, d) {{
  const tip = document.getElementById('tooltip');
  const rpgInfo = d.rpg_nodes && d.rpg_nodes.length > 0
    ? `<div style="color:#79c0ff;font-size:11px;margin-top:2px">RPG nodes: ${{d.rpg_nodes.join(', ')}}</div>` : '';

  const descCount = d.isCollapsed ? (depAllDescendants[d.id] || new Set()).size : 0;
  const collapseInfo = d.isCollapsed
    ? `<div style="color:#1f6feb;font-size:11px">${{descCount}} nodes collapsed (double-click to expand)</div>` : '';
  const expandInfo = d.hasChildren && !d.isCollapsed
    ? `<div style="color:#8b949e;font-size:11px">Double-click to collapse</div>` : '';

  tip.innerHTML = `
    <div class="tt-name">${{d.name || d.id}}</div>
    <div class="tt-type">${{d.type}}</div>
    <div class="tt-path">${{d.id}}</div>
    ${{d.module ? `<div style="color:#8b949e;font-size:11px">${{d.module}}</div>` : ''}}
    ${{collapseInfo}}
    ${{expandInfo}}
    ${{rpgInfo}}
  `;
  tip.style.display = 'block';
  tip.style.left = (event.clientX + 12) + 'px';
  tip.style.top = (event.clientY - 10) + 'px';
}}

// ── Dep controls ──

function depExpandToDepth(maxDepth) {{
  depCollapsed.clear();
  depNodesRaw.forEach(n => {{
    if ((depChildrenOf[n.id] || []).length > 0 && n.depth >= maxDepth) {{
      depCollapsed.add(n.id);
    }}
  }});
  depRedraw();
}}

function depExpandAll() {{
  depCollapsed.clear();
  depRedraw();
}}
function depExpandMore() {{
  // Expand one more level: find shallowest collapsed depth, uncollapse that level
  let minDepth = Infinity;
  depNodesRaw.forEach(n => {{
    if (depCollapsed.has(n.id) && n.depth < minDepth) minDepth = n.depth;
  }});
  if (minDepth === Infinity) return;
  depNodesRaw.forEach(n => {{
    if (depCollapsed.has(n.id) && n.depth === minDepth) depCollapsed.delete(n.id);
  }});
  depRedraw();
}}
function depExpandLess() {{
  // Collapse one level: find deepest expanded depth that has children, collapse that level
  let maxDepth = -1;
  depNodesRaw.forEach(n => {{
    if (!depCollapsed.has(n.id) && (depChildrenOf[n.id] || []).length > 0 && n.depth > maxDepth) maxDepth = n.depth;
  }});
  if (maxDepth <= 0) return;
  depNodesRaw.forEach(n => {{
    if (!depCollapsed.has(n.id) && (depChildrenOf[n.id] || []).length > 0 && n.depth === maxDepth) depCollapsed.add(n.id);
  }});
  depRedraw();
}}


// ══════════════════════════════════════════════
// MAPPING TAB — RPG tree (L→R) ← links → Dep tree (R→L)
// ══════════════════════════════════════════════

const gMap = svg.append('g').attr('class', 'map-group').style('display', 'none');
const mapLinkLayer = gMap.append('g').attr('class', 'map-link-layer');
const mapFeatLinkLayer = gMap.append('g').attr('class', 'map-feat-link-layer');
const mapFeatNodeLayer = gMap.append('g').attr('class', 'map-feat-node-layer');
const mapDepLinkLayer = gMap.append('g').attr('class', 'map-dep-link-layer');
const mapDepNodeLayer = gMap.append('g').attr('class', 'map-dep-node-layer');

const zoomMap = d3.zoom().scaleExtent([0.05, 4]).on('zoom', e => gMap.attr('transform', e.transform));

let mapInitialized = false;
let mapFeatRoot = null;
let mapDepRoot = null;
let mapFeatNodeById = {{}};
let mapDepNodeById = {{}};

const MAP_GAP = 300;
const MAP_NODE_H = 18;
const MAP_FEAT_SPAN = 180;
const MAP_DEP_SPAN = 180;

// Build mapping edge list
const mapEdges = [];
for (const [depId, rpgIds] of Object.entries(depToRpgMap)) {{
  for (const rpgId of rpgIds) {{
    mapEdges.push({{ feat_id: rpgId, dep_id: depId }});
  }}
}}

const mapFeatIds = new Set(mapEdges.map(e => e.feat_id));
const mapDepIds = new Set(mapEdges.map(e => e.dep_id));

function mapInit() {{
  if (mapInitialized) return;
  mapInitialized = true;

  mapFeatRoot = d3.hierarchy(treeData, d => d.children);
  mapFeatRoot.descendants().forEach(d => {{
    if (d.depth >= 2 && d.children) {{ d._children = d.children; d.children = null; }}
  }});

  mapDepRoot = d3.hierarchy(depTreeData, d => d.children);
  mapDepRoot.descendants().forEach(d => {{
    if (d.depth >= 2 && d.children) {{ d._children = d.children; d.children = null; }}
  }});

  mapUpdate();
}}

const mapFeatTree = d3.tree().nodeSize([MAP_NODE_H, MAP_FEAT_SPAN]);
const mapDepTree = d3.tree().nodeSize([MAP_NODE_H, MAP_DEP_SPAN]);

function mapSortDepTree(featNodeById) {{
  const gravityCache = {{}};

  function getGravity(node) {{
    if (gravityCache[node.data.id] !== undefined) return gravityCache[node.data.id];
    const positions = [];
    const rpgIds = depToRpgMap[node.data.id] || [];
    for (const rid of rpgIds) {{
      const fn = featNodeById[rid];
      if (fn) positions.push(fn.x);
    }}
    const kids = node.children || node._children || [];
    for (const child of kids) {{
      const cg = getGravity(child);
      if (cg !== Infinity) positions.push(cg);
    }}
    const result = positions.length > 0
      ? positions.reduce((a, b) => a + b, 0) / positions.length
      : Infinity;
    gravityCache[node.data.id] = result;
    return result;
  }}

  getGravity(mapDepRoot);

  function sortChildren(node) {{
    const kids = node.children || node._children;
    if (!kids || kids.length === 0) return;
    kids.sort((a, b) => getGravity(a) - getGravity(b));
    if (node.children) node.children = kids;
    else node._children = kids;
    for (const child of kids) sortChildren(child);
  }}
  sortChildren(mapDepRoot);
}}

function mapUpdate() {{
  mapFeatTree(mapFeatRoot);
  const featNodes = mapFeatRoot.descendants();
  const featLinks = mapFeatRoot.links();
  featNodes.forEach(d => {{ d.y = d.depth * MAP_FEAT_SPAN; }});

  const tmpFeatById = {{}};
  featNodes.forEach(d => {{ tmpFeatById[d.data.id] = d; }});

  mapSortDepTree(tmpFeatById);

  const featMaxY = d3.max(featNodes, d => d.y) || 0;

  mapDepTree(mapDepRoot);
  const depNodes = mapDepRoot.descendants();
  const depLinks = mapDepRoot.links();
  const depBaseX = featMaxY + MAP_GAP + (d3.max(depNodes, d => d.y) || 0);
  depNodes.forEach(d => {{ d.y = depBaseX - d.depth * MAP_DEP_SPAN; }});

  mapFeatNodeById = {{}};
  featNodes.forEach(d => {{ mapFeatNodeById[d.data.id] = d; }});
  mapDepNodeById = {{}};
  depNodes.forEach(d => {{ mapDepNodeById[d.data.id] = d; }});

  // Draw feat tree
  const fn = mapFeatNodeLayer.selectAll('g.map-feat-node').data(featNodes, d => d.data.id);
  fn.exit().remove();
  const fnEnter = fn.enter().append('g').attr('class', 'map-feat-node')
    .on('click', (event, d) => {{
      event.stopPropagation();
      // Single click: select node (highlight mapping)
      mapHighlight(d.data.id, 'feat', event);
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      mapUpdate();
    }})
    .on('mouseover', (event, d) => mapHighlight(d.data.id, 'feat', event))
    .on('mouseout', () => mapClearHighlight());
  fnEnter.append('circle'); fnEnter.append('text');

  const fnAll = fnEnter.merge(fn);
  fnAll.transition().duration(300).attr('transform', d => `translate(${{d.y}},${{d.x}})`);
  fnAll.select('circle')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', d => {{ const tn = d.data.meta?.type_name || d.data.node_type || 'default'; return nodeTypeColors[tn] || nodeTypeColors.default; }})
    .attr('stroke', d => mapFeatIds.has(d.data.id) ? '#f0883e' : '#30363d')
    .attr('stroke-width', d => mapFeatIds.has(d.data.id) ? 2 : 1.5)
    .style('cursor', 'pointer');
  fnAll.select('text')
    .attr('dy', 3.5)
    .attr('x', -10)
    .attr('text-anchor', 'end')
    .attr('fill', '#c9d1d9').attr('font-size', '13px')
    .text(d => {{ const n = d.data.name || d.data.id; return n.length > 30 ? n.slice(0, 28) + '...' : n; }});

  // feat links
  const fl = mapFeatLinkLayer.selectAll('path.map-feat-link').data(featLinks, d => d.target.data.id);
  fl.exit().remove();
  fl.enter().append('path').attr('class', 'map-feat-link link')
    .merge(fl).transition().duration(300)
    .attr('d', d => diagonal(d.source, d.target));

  // Draw dep tree
  const dn = mapDepNodeLayer.selectAll('g.map-dep-node').data(depNodes, d => d.data.id);
  dn.exit().remove();
  const dnEnter = dn.enter().append('g').attr('class', 'map-dep-node')
    .on('click', (event, d) => {{
      event.stopPropagation();
      // Single click: select node (highlight mapping)
      mapHighlight(d.data.id, 'dep', event);
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      mapUpdate();
    }})
    .on('mouseover', (event, d) => mapHighlight(d.data.id, 'dep', event))
    .on('mouseout', () => mapClearHighlight());
  dnEnter.append('circle'); dnEnter.append('text');

  const dnAll = dnEnter.merge(dn);
  dnAll.transition().duration(300).attr('transform', d => `translate(${{d.y}},${{d.x}})`);
  dnAll.select('circle')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', d => nodeTypeColors[d.data.type] || nodeTypeColors.default)
    .attr('stroke', d => mapDepIds.has(d.data.id) ? '#f0883e' : '#30363d')
    .attr('stroke-width', d => mapDepIds.has(d.data.id) ? 2 : 1.5)
    .style('cursor', 'pointer');
  dnAll.select('text')
    .attr('dy', 3.5)
    .attr('x', 10)
    .attr('text-anchor', 'start')
    .attr('fill', '#c9d1d9').attr('font-size', '13px')
    .text(d => {{ const n = d.data.name || d.data.id; return n.length > 30 ? n.slice(0, 28) + '...' : n; }});

  // dep links
  const dl = mapDepLinkLayer.selectAll('path.map-dep-link').data(depLinks, d => d.target.data.id);
  dl.exit().remove();
  dl.enter().append('path').attr('class', 'map-dep-link link')
    .merge(dl).transition().duration(300)
    .attr('d', d => diagonal(d.source, d.target));

  // Draw mapping links
  mapDrawLinks();
  applyExternalChangeHighlights();
}}

function mapDrawLinks() {{
  mapLinkLayer.selectAll('path.map-link').remove();
  if (!showEdges) return;

  mapEdges.forEach(e => {{
    const fn = mapFeatNodeById[e.feat_id];
    const dn = mapDepNodeById[e.dep_id];
    if (!fn || !dn) return;

    const sx = fn.y + 6, sy = fn.x;
    const dx = dn.y - 6, dy = dn.x;
    const midX = (sx + dx) / 2;

    mapLinkLayer.append('path')
      .attr('class', 'map-link')
      .attr('data-feat', e.feat_id)
      .attr('data-dep', e.dep_id)
      .attr('d', `M${{sx}},${{sy}} C${{midX}},${{sy}} ${{midX}},${{dy}} ${{dx}},${{dy}}`)
      .attr('fill', 'none');
  }});
}}

function mapHighlight(nodeId, side, event) {{
  const connected = mapEdges.filter(e => side === 'feat' ? e.feat_id === nodeId : e.dep_id === nodeId);
  if (connected.length === 0) {{
    const tip = document.getElementById('tooltip');
    tip.innerHTML = `<div class="tt-name">${{nodeId}}</div><div class="tt-type" style="color:#484f58">No mapping</div>`;
    tip.style.display = 'block';
    tip.style.left = (event.clientX + 12) + 'px';
    tip.style.top = (event.clientY - 10) + 'px';
    return;
  }}

  const connectedFeatIds = new Set(connected.map(e => e.feat_id));
  const connectedDepIds = new Set(connected.map(e => e.dep_id));

  mapLinkLayer.selectAll('path.map-link').each(function(d) {{
    const el = d3.select(this);
    const fid = el.attr('data-feat'), did = el.attr('data-dep');
    if (connectedFeatIds.has(fid) && connectedDepIds.has(did)) {{
      el.classed('map-link-hover', true).raise();
    }} else {{
      el.style('stroke-opacity', 0.08);
    }}
  }});

  mapFeatNodeLayer.selectAll('g.map-feat-node').each(function(d) {{
    if (connectedFeatIds.has(d.data.id)) d3.select(this).classed('map-node-highlight', true);
    else d3.select(this).style('opacity', 0.2);
  }});
  mapDepNodeLayer.selectAll('g.map-dep-node').each(function(d) {{
    if (connectedDepIds.has(d.data.id)) d3.select(this).classed('map-node-highlight', true);
    else d3.select(this).style('opacity', 0.2);
  }});

  const tip = document.getElementById('tooltip');
  const names = side === 'feat'
    ? connected.map(e => e.dep_id).join('<br>')
    : connected.map(e => e.feat_id).join('<br>');
  tip.innerHTML = `<div class="tt-name">${{nodeId}}</div>
    <div class="tt-type">${{connected.length}} mapping(s)</div>
    <div class="tt-path">${{names}}</div>`;
  tip.style.display = 'block';
  tip.style.left = (event.clientX + 12) + 'px';
  tip.style.top = (event.clientY - 10) + 'px';
}}

function mapClearHighlight() {{
  hideTooltip();
  mapLinkLayer.selectAll('path.map-link').classed('map-link-hover', false).style('stroke-opacity', null);
  mapFeatNodeLayer.selectAll('g.map-feat-node').classed('map-node-highlight', false).style('opacity', null);
  mapDepNodeLayer.selectAll('g.map-dep-node').classed('map-node-highlight', false).style('opacity', null);
}}

function mapExpandToDepth(maxDepth) {{
  if (!mapFeatRoot || !mapDepRoot) return;
  mapFeatRoot.descendants().forEach(d => {{
    if (d._children && d.depth < maxDepth) {{ d.children = d._children; d._children = null; }}
    else if (d.children && d.depth >= maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  mapDepRoot.descendants().forEach(d => {{
    if (d._children && d.depth < maxDepth) {{ d.children = d._children; d._children = null; }}
    else if (d.children && d.depth >= maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  mapUpdate();
}}
function mapExpandAll() {{
  if (!mapFeatRoot || !mapDepRoot) return;
  function visitAll(d) {{
    if (d._children) {{ d.children = d._children; d._children = null; }}
    if (d.children) d.children.forEach(visitAll);
  }}
  visitAll(mapFeatRoot);
  visitAll(mapDepRoot);
  mapUpdate();
}}
function mapExpandMore() {{
  if (!mapFeatRoot || !mapDepRoot) return;
  let minDepth = Infinity;
  mapFeatRoot.descendants().forEach(d => {{ if (d._children && d.depth < minDepth) minDepth = d.depth; }});
  mapDepRoot.descendants().forEach(d => {{ if (d._children && d.depth < minDepth) minDepth = d.depth; }});
  if (minDepth === Infinity) return;
  mapFeatRoot.descendants().forEach(d => {{
    if (d._children && d.depth === minDepth) {{ d.children = d._children; d._children = null; }}
  }});
  mapDepRoot.descendants().forEach(d => {{
    if (d._children && d.depth === minDepth) {{ d.children = d._children; d._children = null; }}
  }});
  mapUpdate();
}}
function mapExpandLess() {{
  if (!mapFeatRoot || !mapDepRoot) return;
  let maxDepth = -1;
  mapFeatRoot.descendants().forEach(d => {{ if (d.children && d.children.length > 0 && d.depth > maxDepth) maxDepth = d.depth; }});
  mapDepRoot.descendants().forEach(d => {{ if (d.children && d.children.length > 0 && d.depth > maxDepth) maxDepth = d.depth; }});
  if (maxDepth <= 0) return;
  mapFeatRoot.descendants().forEach(d => {{
    if (d.children && d.children.length > 0 && d.depth === maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  mapDepRoot.descendants().forEach(d => {{
    if (d.children && d.children.length > 0 && d.depth === maxDepth) {{ d._children = d.children; d.children = null; }}
  }});
  mapUpdate();
}}

// ══════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════

function switchTab(tab) {{
  if (tab !== activeTab && typeof externalChanges === 'object' && externalChanges) externalChanges.focus = null;
  activeTab = tab;
  document.getElementById('tab-feat').classList.toggle('active', tab === 'feat');
  document.getElementById('tab-dep').classList.toggle('active', tab === 'dep');
  document.getElementById('tab-map').classList.toggle('active', tab === 'map');
  document.getElementById('stats-feat').style.display = tab === 'feat' ? 'flex' : 'none';
  document.getElementById('stats-dep').style.display = tab === 'dep' ? 'flex' : 'none';
  document.getElementById('stats-map').style.display = tab === 'map' ? 'flex' : 'none';
  document.getElementById('feat-controls').style.display = tab === 'feat' ? 'inline-flex' : 'none';
  document.getElementById('dep-controls').style.display = tab === 'dep' ? 'inline-flex' : 'none';
  document.getElementById('map-controls').style.display = tab === 'map' ? 'inline-flex' : 'none';

  gChange.style('display', 'none');
  document.getElementById('stats-feat').innerHTML = originalGraphUi.featStats;
  document.getElementById('stats-dep').innerHTML = originalGraphUi.depStats;
  document.getElementById('stats-map').innerHTML = originalGraphUi.mapStats;
  document.getElementById('canvas-overlay').innerHTML = originalGraphUi.overlay;
  document.getElementById('search').style.display = null;
  document.getElementById('btn-edges').style.display = null;

  gFeat.style('display', tab === 'feat' ? null : 'none');
  gDep.style('display', tab === 'dep' ? null : 'none');
  gMap.style('display', tab === 'map' ? null : 'none');

  if (tab === 'feat') {{
    if (depSimulation) depSimulation.stop();
    svg.on('.zoom', null).call(zoomFeat);
  }} else if (tab === 'dep') {{
    if (depSimulation) depSimulation.stop();
    svg.on('.zoom', null).call(zoomChange);
  }} else if (tab === 'map') {{
    if (depSimulation) depSimulation.stop();
    if (!hasMap) {{
      gMap.selectAll('*').remove();
      gMap.append('foreignObject').attr('width', width).attr('height', height)
        .append('xhtml:div').attr('class', 'no-data').text('No _dep_to_rpg_map data');
    }} else {{
      mapInit();
    }}
    svg.on('.zoom', null).call(zoomMap);
    // Auto-fit
    setTimeout(() => {{
      const bounds = gMap.node().getBBox();
      if (bounds.width === 0) return;
      const fw = bounds.width + 100, fh = bounds.height + 100;
      const scale = Math.min(width / fw, height / fh, 1) * 0.85;
      svg.call(zoomMap.transform,
        d3.zoomIdentity
          .translate(width / 2 - bounds.x * scale - bounds.width * scale / 2,
                     height / 2 - bounds.y * scale - bounds.height * scale / 2)
          .scale(scale));
    }}, 350);
  }}
  if (activeTab === 'dep' && typeof drawChangeGraph === 'function') {{
    document.getElementById('search').style.display = 'none';
    document.getElementById('feat-controls').style.display = 'none';
    document.getElementById('dep-controls').style.display = 'none';
    document.getElementById('map-controls').style.display = 'none';
    drawChangeGraph();
  }} else if (typeof applyChangeEmphasis === 'function') {{
    applyChangeEmphasis();
  }}
  if (typeof syncChangesPanelVisibility === 'function') syncChangesPanelVisibility();
  if (typeof renderChangesPanel === 'function') renderChangesPanel();
}}

// ── External change highlighting (dashboard parent → graph iframe) ──
function externalNodeSets(value) {{
  const rows = value || {{}};
  const ids = key => new Set((rows[key] || []).map(item => String(item.node_id || item.id || item)));
  return {{ added: ids('added'), removed: ids('removed'), modified: ids('modified'), rows }};
}}

function externalKind(id, sets) {{
  let kind = null;
  if (sets.added.has(id)) kind = 'added';
  else if (sets.modified.has(id)) kind = 'modified';
  if (externalChanges.filter !== 'all' && externalChanges.filter !== kind) return null;
  return kind;
}}

function applyChangeClasses(selection, idOf, sets) {{
  const hasVisible = externalChanges.filter === 'all'
    ? sets.added.size + sets.removed.size + sets.modified.size > 0
    : (sets[externalChanges.filter]?.size || 0) > 0;
  const dimUnchanged = externalChanges.mode === 'changes' && hasVisible;
  selection.each(function(d) {{
    const id = String(idOf(d));
    const kind = externalKind(id, sets);
    const element = d3.select(this);
    element
      .classed('change-added', kind === 'added')
      .classed('change-modified', kind === 'modified')
      .classed('change-dim', dimUnchanged && !kind);
    element.select('.change-status-glyph').text(kind === 'added' ? '+' : kind === 'modified' ? '~' : '');
  }});
}}

function clearGraphFocusVisuals() {{
  nodeLayer.selectAll('g.node').classed('change-focused', false).select('.change-focus-ring').style('display', 'none');
  depNodeG.selectAll('g.dep-node').classed('change-focused', false).select('.change-focus-ring').style('display', 'none');
  depHullG.selectAll('.dep-hull').classed('change-focused', false);
  depLabelG.selectAll('.dep-hull-label').classed('change-focused', false);
  depChangeGroupG.selectAll('.dep-change-group').classed('focused', false);
  featGhostLayer.selectAll('.feat-ghost').classed('focused', false).select('.change-focus-ring').style('display', 'none');
  depGhostG.selectAll('.dep-ghost').classed('focused', false).select('.change-focus-ring').style('display', 'none');
}}

function applyExternalChangeHighlights() {{
  if (!hasChangeData || (externalChanges.mode !== 'changes' && !externalChanges.emphasize)) return;
  applyChangeClasses(nodeLayer.selectAll('g.node'), d => d.data.id, externalChanges.feature);
  applyChangeClasses(depNodeG.selectAll('g.dep-node'), d => d.id, externalChanges.dependency);
  applyChangeClasses(depHullG.selectAll('path.dep-hull'), d => d.id, externalChanges.dependency);
  applyChangeClasses(depLabelG.selectAll('text.dep-hull-label'), d => d.id, externalChanges.dependency);
  applyChangeClasses(mapFeatNodeLayer.selectAll('g.map-feat-node'), d => d.data.id, externalChanges.feature);
  applyChangeClasses(mapDepNodeLayer.selectAll('g.map-dep-node'), d => d.data.id, externalChanges.dependency);
}}

function clearExternalChangeHighlights() {{
  [nodeLayer.selectAll('g.node'), depNodeG.selectAll('g.dep-node'), depHullG.selectAll('path.dep-hull'),
    depLabelG.selectAll('text.dep-hull-label'), mapFeatNodeLayer.selectAll('g.map-feat-node'), mapDepNodeLayer.selectAll('g.map-dep-node')]
    .forEach(selection => selection.classed('change-added', false).classed('change-modified', false).classed('change-dim', false));
  featSelectedNodeId = null;
  featApplySelectionHighlight();
  depSelectedNodes.clear();
  depUpdateEdgeVisibility();
  depUpdateNodeHighlight();
}}

function expandFeatureChangePaths(ids) {{
  function visit(node) {{
    const children = node.children || node._children || [];
    let contains = ids.has(String(node.data.id));
    children.forEach(child => {{ if (visit(child)) contains = true; }});
    if (contains && node._children) {{ node.children = node._children; node._children = null; }}
    return contains;
  }}
  visit(root);
}}

function expandDependencyChangePaths(ids) {{
  ids.forEach(id => {{
    let current = id;
    while (current) {{
      depCollapsed.delete(current);
      current = depParentMap[current];
    }}
  }});
}}

function applyChangeEmphasis() {{
  featGhostLayer.selectAll('*').remove();
  document.getElementById('canvas-overlay').innerHTML = originalGraphUi.overlay;
  if (!hasChangeData || (externalChanges.mode !== 'changes' && !externalChanges.emphasize)) {{
    clearExternalChangeHighlights();
    clearGraphFocusVisuals();
    return;
  }}
  if (!externalChanges.focus) clearGraphFocusVisuals();
  const focusMode = externalChanges.mode === 'changes';
  if (activeTab === 'feat') {{
    if (focusMode) {{
      const vis = new Set([...externalChanges.feature.added, ...externalChanges.feature.modified]);
      (externalChanges.feature.rows?.removed || []).forEach(n => {{ if (n.parent_id) vis.add(String(n.parent_id)); }});
      if (vis.size) {{ expandFeatureChangePaths(vis); update(root); }}
    }}
    applyExternalChangeHighlights();
    drawFeatRemovedGhosts();
    if (focusMode) requestAnimationFrame(fitFeatEmphasis);
  }} else if (activeTab === 'dep') {{
    if (focusMode) {{
      const vis = new Set([...externalChanges.dependency.added, ...externalChanges.dependency.modified]);
      (externalChanges.dependency.rows?.removed || []).forEach(n => {{ if (n.parent_id) vis.add(String(n.parent_id)); }});
      if (vis.size) {{ expandDependencyChangePaths(vis); depRedraw(); }}
    }}
    applyExternalChangeHighlights();
  }}
  renderChangeStatusOverlay();
}}

function renderChangeStatusOverlay() {{
  if (!hasChangeData) return;
  const el = document.getElementById('canvas-overlay');
  const dimNote = externalChanges.mode === 'changes' ? `<div class="legend-item"><span class="change-legend-mark context">○</span> Unchanged (dim)</div>` : '';
  el.innerHTML = originalGraphUi.overlay
    + `<div class="ov-section"><div class="ov-title">Change status</div>`
    + `<div class="legend-item"><span class="change-legend-mark added">+</span> Added</div>`
    + `<div class="legend-item"><span class="change-legend-mark removed">−</span> Removed</div>`
    + `<div class="legend-item"><span class="change-legend-mark modified">~</span> Modified</div>`
    + dimNote + `</div>`;
}}

function drawFeatRemovedGhosts() {{
  featGhostLayer.selectAll('*').remove();
  if (!hasChangeData || activeTab !== 'feat') return;
  const filter = externalChanges.filter;
  if (filter !== 'all' && filter !== 'removed') return;
  const removed = (externalChanges.feature.rows?.removed || []).map(node => ({{...node, children: []}}));
  const removedById = new Map(removed.map(node => [String(node.node_id), node]));
  const byParent = {{}};
  removed.forEach(n => {{
    const removedParent = removedById.get(String(n.parent_id));
    if (removedParent) {{ removedParent.children.push(n); return; }}
    const currentParent = nodeById[String(n.parent_id)];
    if (!currentParent) return;
    (byParent[String(n.parent_id)] = byParent[String(n.parent_id)] || {{parent: currentParent, roots: []}}).roots.push(n);
  }});
  const focusId = externalChanges.focus?.node_id;
  Object.values(byParent).forEach(grp => {{
    const ghostRoot = d3.hierarchy({{children: grp.roots}}, node => node.children);
    const ghostLayout = d3.tree().nodeSize([18, 210])(ghostRoot);
    const positions = new Map();
    ghostLayout.descendants().slice(1).forEach(d => {{
      positions.set(String(d.data.node_id), {{x: grp.parent.y + d.y, y: grp.parent.x + d.x}});
    }});
    ghostLayout.descendants().slice(1).forEach(d => {{
      const n = d.data;
      const pos = positions.get(String(n.node_id));
      const parentPos = d.parent.depth === 0
        ? {{x: grp.parent.y, y: grp.parent.x}}
        : positions.get(String(d.parent.data.node_id));
      featGhostLayer.append('path').attr('class', 'feat-ghost-link')
        .attr('d', `M${{parentPos.x}},${{parentPos.y}} C${{(parentPos.x + pos.x) / 2}},${{parentPos.y}} ${{(parentPos.x + pos.x) / 2}},${{pos.y}} ${{pos.x}},${{pos.y}}`);
      const g = featGhostLayer.append('g')
        .attr('class', 'feat-ghost ' + (d.children ? 'branch' : 'leaf') + (focusId === String(n.node_id) ? ' focused' : ''))
        .attr('data-ghost-id', String(n.node_id))
        .attr('transform', `translate(${{pos.x}},${{pos.y}})`)
        .on('click', event => {{ event.stopPropagation(); focusChangeNode(String(n.node_id), 'feature'); }});
      g.append('circle').attr('class', 'change-status-dot').attr('r', 6);
      g.append('text').attr('class', 'change-status-glyph').attr('dy', 3).text('−');
      g.append('circle').attr('class', 'change-focus-ring').attr('r', 11).style('display', focusId === String(n.node_id) ? null : 'none');
      const nm = n.name || n.node_id;
      g.append('text').attr('class', 'change-node-label')
        .attr('x', d.children ? 0 : 10).attr('y', d.children ? 18 : 0).attr('dy', d.children ? 0 : 3.5)
        .text(nm.length > 30 ? nm.slice(0, 28) + '…' : nm);
    }});
  }});
}}

function drawDepRemovedGhosts() {{
  depGhostG.selectAll('*').remove();
  if (!hasChangeData || activeTab !== 'dep' || externalChanges.mode !== 'full' || !externalChanges.emphasize) return;
  const filter = externalChanges.filter;
  if (filter !== 'all' && filter !== 'removed') return;
  const removed = (externalChanges.dependency.rows?.removed || []).map(node => ({{...node, children: []}}));
  const removedById = new Map(removed.map(node => [String(node.node_id), node]));
  const roots = [];
  removed.forEach(node => {{
    const removedParent = removedById.get(String(node.parent_id));
    if (removedParent) {{ removedParent.children.push(node); return; }}
    const parent = depNodeMap[String(node.parent_id)];
    if (parent && Number.isFinite(parent._x) && Number.isFinite(parent._y)) roots.push({{parent, node}});
  }});
  const focusId = externalChanges.focus?.node_id;
  roots.forEach(rootItem => {{
    const hierarchy = d3.tree().nodeSize([18, 150])(d3.hierarchy(rootItem.node, node => node.children));
    const positions = new Map();
    hierarchy.descendants().forEach(d => positions.set(String(d.data.node_id), {{x: rootItem.parent._x + d.y + 90, y: rootItem.parent._y + d.x}}));
    hierarchy.descendants().forEach(d => {{
      const id = String(d.data.node_id);
      const pos = positions.get(id);
      const parentPos = d.parent ? positions.get(String(d.parent.data.node_id)) : {{x: rootItem.parent._x, y: rootItem.parent._y}};
      depGhostG.append('line').attr('class', 'dep-ghost-link')
        .attr('x1', parentPos.x).attr('y1', parentPos.y).attr('x2', pos.x).attr('y2', pos.y);
      const g = depGhostG.append('g').attr('class', 'dep-ghost ' + (d.children ? 'branch' : 'leaf') + (focusId === id ? ' focused' : ''))
        .attr('data-ghost-id', id).attr('transform', `translate(${{pos.x}},${{pos.y}})`)
        .on('click', event => {{ event.stopPropagation(); focusChangeNode(id, 'dependency'); }});
      g.append('circle').attr('class', 'change-status-dot').attr('r', 6);
      g.append('text').attr('class', 'change-status-glyph').attr('dy', 3).text('−');
      g.append('circle').attr('class', 'change-focus-ring').attr('r', 11).style('display', focusId === id ? null : 'none');
      const name = d.data.name || id;
      g.append('text').attr('class', 'change-node-label')
        .attr('x', d.children ? 0 : 10).attr('y', d.children ? 18 : 0).attr('dy', d.children ? 0 : 3.5)
        .text(name.length > 28 ? name.slice(0, 26) + '…' : name);
    }});
  }});
}}

function drawDepChangedGroups() {{
  depChangeGroupG.selectAll('*').remove();
  if (!hasChangeData || activeTab !== 'dep' || externalChanges.mode !== 'full' || !externalChanges.emphasize) return;
  const rows = [
    ...(externalChanges.dependency.rows?.added || []).map(node => ({{...node, status: 'added'}})),
    ...(externalChanges.dependency.rows?.modified || []).map(node => ({{...node, status: 'modified'}})),
  ].filter(node => externalChanges.filter === 'all' || externalChanges.filter === node.status);
  const focusId = externalChanges.focus?.node_id;
  rows.forEach(node => {{
    const id = String(node.node_id);
    if (depVisNodeDataMap[id]) return;
    const descendants = depAllDescendants[id] || new Set();
    const points = [...descendants].map(childId => depVisNodeDataMap[childId])
      .filter(child => child && Number.isFinite(child.x) && Number.isFinite(child.y));
    if (!points.length) return;
    const x = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const y = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    const label = node.name || id;
    const text = label.length > 22 ? label.slice(0, 20) + '…' : label;
    const width = Math.max(54, text.length * 6 + 14);
    const group = depChangeGroupG.append('g')
      .attr('class', `dep-change-group ${{node.status}}${{focusId === id ? ' focused' : ''}}`)
      .attr('data-change-group-id', id).attr('transform', `translate(${{x}},${{y - 18}})`)
      .on('click', event => {{ event.stopPropagation(); focusChangeNode(id, 'dependency'); }});
    group.append('rect').attr('x', -width / 2).attr('y', -9).attr('width', width).attr('height', 18);
    group.append('text').attr('text-anchor', 'middle').attr('dy', 3.5).text(text);
  }});
}}

function centerDepOn(x, y) {{
  const scale = 1.15;
  const rightInset = (typeof cpOpen !== 'undefined' && cpOpen) ? 330 : 0;
  const centerX = (width - rightInset) / 2;
  svg.call(zoomDep.transform, d3.zoomIdentity.translate(centerX - x * scale, height / 2 - y * scale).scale(scale));
}}

function focusDepRenderedNode(id) {{
  depNodeG.selectAll('g.dep-node').classed('change-focused', d => String(d.id) === String(id));
  depNodeG.selectAll('g.dep-node .change-focus-ring').style('display', function() {{ return String(this.parentNode.__data__.id) === String(id) ? null : 'none'; }});
  depHullG.selectAll('path.dep-hull').classed('change-focused', d => String(d.id) === String(id));
  depLabelG.selectAll('text.dep-hull-label').classed('change-focused', d => String(d.id) === String(id));
  const node = depVisNodeDataMap[String(id)];
  if (node) {{ centerDepOn(node.x, node.y); return; }}
  const source = depNodeMap[String(id)];
  if (source && Number.isFinite(source._x) && Number.isFinite(source._y)) {{ centerDepOn(source._x, source._y); return; }}
  const changedGroup = depChangeGroupG.selectAll('.dep-change-group').filter(function() {{ return this.getAttribute('data-change-group-id') === String(id); }})
    .classed('focused', true).node();
  if (changedGroup) {{
    const match = (changedGroup.getAttribute('transform') || '').match(/translate\\(([-\\d.]+),([-\\d.]+)\\)/);
    if (match) {{ centerDepOn(Number(match[1]), Number(match[2])); return; }}
  }}
  const ghost = depGhostG.selectAll('.dep-ghost').filter(function() {{ return this.getAttribute('data-ghost-id') === String(id); }}).node();
  if (ghost) {{
    const match = (ghost.getAttribute('transform') || '').match(/translate\\(([-\\d.]+),([-\\d.]+)\\)/);
    if (match) centerDepOn(Number(match[1]), Number(match[2]));
  }}
}}

function centerFeatOn(x, y) {{
  const scale = 1.1;
  svg.call(zoomFeat.transform, d3.zoomIdentity.translate(width / 2 - x * scale, height / 2 - y * scale).scale(scale));
}}

function fitFeatEmphasis() {{
  const b = gFeat.node().getBBox();
  if (!b.width || !b.height) return;
  const rightInset = (typeof cpOpen !== 'undefined' && cpOpen) ? 342 : 30;
  const usableW = Math.max(width - 40 - rightInset, 200);
  const usableH = Math.max(height - 120, 200);
  const scale = Math.min(usableW / (b.width + 80), usableH / (b.height + 80), 1) * 0.9;
  svg.call(zoomFeat.transform, d3.zoomIdentity.translate(40 + usableW / 2 - (b.x + b.width / 2) * scale, 110 + usableH / 2 - (b.y + b.height / 2) * scale).scale(scale));
}}

function focusChangeNode(id, scope) {{
  scope = scope || (activeTab === 'dep' ? 'dependency' : 'feature');
  const sets = externalChanges[scope];
  const removedSet = new Set((sets.rows?.removed || []).map(n => String(n.node_id)));
  const kind = sets.added.has(String(id)) ? 'added' : sets.modified.has(String(id)) ? 'modified' : removedSet.has(String(id)) ? 'removed' : 'context';
  const sameDependencyFocus = scope === 'dependency' && activeTab === 'dep'
    && externalChanges.focus?.scope === 'dependency' && String(externalChanges.focus.node_id) === String(id);
  const sameFeatureFocus = scope === 'feature' && activeTab === 'feat'
    && externalChanges.focus?.scope === 'feature' && String(externalChanges.focus.node_id) === String(id);
  externalChanges.focus = (sameDependencyFocus || sameFeatureFocus) ? null : {{scope, node_id: String(id), kind}};
  if (scope === 'dependency' && activeTab === 'dep') {{
    drawChangeGraph();
    if (typeof renderChangesPanel === 'function') renderChangesPanel();
    return;
  }}
  clearGraphFocusVisuals();
  if (scope === 'feature' && activeTab === 'feat') {{
    featSelectedNodeId = externalChanges.focus ? String(id) : null;
    if (!externalChanges.focus) {{
      update(root);
      if (typeof renderChangesPanel === 'function') renderChangesPanel();
      return;
    }}
    if (kind === 'removed') {{
      drawFeatRemovedGhosts();
      const el = featGhostLayer.selectAll('.feat-ghost').filter(function() {{ return this.getAttribute('data-ghost-id') === String(id); }}).node();
      if (el) {{
        const m = (el.getAttribute('transform') || '').match(/translate\\(([-\\d.]+),([-\\d.]+)\\)/);
        if (m) centerFeatOn(Number(m[1]), Number(m[2]));
      }}
    }} else {{
      expandFeatureChangePaths(new Set([String(id)]));
      update(root);
      applyExternalChangeHighlights();
      drawFeatRemovedGhosts();
      nodeLayer.selectAll('g.node').classed('change-focused', d => String(d.data.id) === String(id));
      nodeLayer.selectAll('g.node .change-focus-ring').style('display', function() {{ return String(this.parentNode.__data__.data.id) === String(id) ? null : 'none'; }});
      const t = nodeById[String(id)];
      if (t) centerFeatOn(t.y, t.x);
    }}
  }}
  if (scope === 'dependency' && activeTab === 'dep') {{
    depInit();
    expandDependencyChangePaths(new Set([String(id)]));
    pendingDepFocusId = String(id);
    depRedraw();
    setTimeout(() => {{
      if (pendingDepFocusId === String(id)) {{
        drawDepChangedGroups();
        drawDepRemovedGhosts();
        focusDepRenderedNode(String(id));
      }}
    }}, 350);
  }}
  if (typeof renderChangesPanel === 'function') renderChangesPanel();
}}

function changeRows(scope) {{
  const source = externalChanges[scope] || {{}};
  const rows = [];
  ['added', 'removed', 'modified'].forEach(kind => {{
    (source.rows?.[kind] || []).forEach(node => rows.push({{
      ...node,
      id: String(node.node_id || node.id),
      status: kind,
      scope,
    }}));
  }});
  return rows;
}}

function allFeatureNodes() {{
  const values = [];
  const seen = new Set();
  function visit(node) {{
    const id = String(node.data.id);
    if (seen.has(id)) return;
    seen.add(id); values.push(node);
    [...(node.children || []), ...(node._children || [])].forEach(visit);
  }}
  visit(root);
  return values;
}}

function currentFeatureNode(id) {{
  const node = allFeatureNodes().find(item => String(item.data.id) === String(id));
  if (!node) return null;
  return {{
    id: String(node.data.id), name: node.data.name || node.data.id,
    node_type: node.data.node_type || node.data.meta?.type_name,
    path: node.data.meta?.path, status: 'context', scope: 'feature',
    parent_id: node.parent ? String(node.parent.data.id) : null,
  }};
}}

function currentDependencyNode(id) {{
  const node = depNodeMap[String(id)] || depNodesRaw.find(item => String(item.id) === String(id));
  if (!node) return null;
  return {{
    id: String(node.id), name: String(node.id) === '.' ? 'repository root' : (node.name || node.id), node_type: node.type,
    status: 'context', scope: 'dependency', parent_id: depParentMap[node.id] || null,
  }};
}}

function edgeId(value) {{ return String(typeof value === 'object' ? value.id : value); }}
function graphEdgeKey(edge) {{ return `${{edgeId(edge.source)}}|${{edgeId(edge.target)}}|${{edge.relation || edge.type || ''}}`; }}

function currentFeatureRows() {{
  return allFeatureNodes().map(node => currentFeatureNode(node.data.id)).filter(Boolean);
}}

function currentFeatureEdges() {{
  const hierarchy = allFeatureNodes().filter(node => node.parent).map(node => ({{
    source: String(node.parent.data.id), target: String(node.data.id), relation: 'contains', status: 'context',
  }}));
  return [
    ...hierarchy,
    ...semanticEdges.map(edge => ({{source: String(edge.src), target: String(edge.dst), relation: edge.relation, status: 'context'}})),
  ];
}}

function currentDependencyRows() {{
  return depNodesRaw.map(node => currentDependencyNode(node.id)).filter(Boolean);
}}

function currentDependencyEdges() {{
  const semantic = depEdgesRaw.map(edge => ({{
    source: edgeId(edge.source), target: edgeId(edge.target), relation: edge.type, status: 'context',
  }}));
  const hierarchy = Object.entries(depParentMap).map(([child, parent]) => ({{
    source: String(parent), target: String(child), relation: 'contains', status: 'context',
  }}));
  return [...hierarchy, ...semantic];
}}

function buildFullDependencyGraph() {{
  const nodes = currentDependencyRows().map(node => ({{...node, status: 'normal'}}));
  const nodeIds = new Set(nodes.map(node => String(node.id)));
  const edges = currentDependencyEdges()
    .filter(edge => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)))
    .map(edge => ({{...edge, status: 'normal'}}));
  return {{nodes, edges}};
}}

function currentMappingEdges() {{
  return Object.entries(depToRpgMap).flatMap(([depId, featureIds]) =>
    (featureIds || []).map(featureId => ({{source: String(depId), target: String(featureId), relation: 'maps_to', status: 'context'}})));
}}

function buildChangeGraph(scope) {{
  const allRows = scope === 'map' ? [...changeRows('feature'), ...changeRows('dependency')] : changeRows(scope);
  const filtered = externalChanges.filter === 'all'
    ? allRows
    : allRows.filter(node => node.status === externalChanges.filter);
  const nodes = new Map(filtered.map(node => [node.id, {{...node}}]));
  const edges = new Map();
  const diffEdges = scope === 'feature'
    ? [
        ...(externalChanges.semanticEdges?.added || []).map(edge => ({{...edge, status: 'added'}})),
        ...(externalChanges.semanticEdges?.removed || []).map(edge => ({{...edge, status: 'removed'}})),
        ...(externalChanges.hierarchyEdges?.added || []).map(edge => ({{...edge, status: 'added'}})),
        ...(externalChanges.hierarchyEdges?.removed || []).map(edge => ({{...edge, status: 'removed'}})),
      ]
    : scope === 'dependency' ? [
        ...(externalChanges.dependencyEdges?.added || []).map(edge => ({{...edge, status: 'added'}})),
        ...(externalChanges.dependencyEdges?.removed || []).map(edge => ({{...edge, status: 'removed'}})),
      ] : [
        ...(externalChanges.mappingEdges?.added || []).map(edge => ({{...edge, status: 'added'}})),
        ...(externalChanges.mappingEdges?.removed || []).map(edge => ({{...edge, status: 'removed'}})),
      ];
  const currentRows = scope === 'feature' ? currentFeatureRows() : scope === 'dependency' ? currentDependencyRows() : [...currentFeatureRows(), ...currentDependencyRows()];
  const currentEdges = scope === 'feature' ? currentFeatureEdges() : scope === 'dependency' ? currentDependencyEdges() : currentMappingEdges();
  if (externalChanges.contextMode === 'full') currentRows.forEach(node => {{ if (!nodes.has(node.id)) nodes.set(node.id, node); }});
  const sourceEdges = [...currentEdges, ...diffEdges];

  function addContext(id) {{
    if (!id || nodes.has(String(id))) return;
    const row = scope === 'feature' ? currentFeatureNode(id) : scope === 'dependency' ? currentDependencyNode(id) : (currentFeatureNode(id) || currentDependencyNode(id));
    if (row) nodes.set(row.id, row);
  }}
  filtered.forEach(node => {{
    addContext(node.parent_id || node.previous_parent_id);
  }});
  sourceEdges.forEach(raw => {{
    const edge = {{
      source: String(raw.source), target: String(raw.target),
      relation: raw.relation || 'related', status: raw.status || 'context',
    }};
    if (nodes.has(edge.source) && nodes.has(edge.target)) edges.set(graphEdgeKey(edge), edge);
  }});
  filtered.forEach(node => {{
    const parent = node.parent_id || node.previous_parent_id;
    if (!parent || !nodes.has(String(parent))) return;
    const edge = {{source: String(parent), target: node.id, relation: 'contains', status: node.status === 'removed' ? 'removed' : 'context'}};
    edges.set(graphEdgeKey(edge), edge);
  }});
  return {{nodes: [...nodes.values()], edges: [...edges.values()]}};
}}

function changeNodeGlyph(status) {{ return status === 'added' ? '+' : status === 'removed' ? '−' : status === 'modified' ? '~' : ''; }}
function dependencyDisplayLabel(node) {{
  const fullName = String(node.name || node.id);
  const conciseName = fullName.startsWith('example_') ? fullName.slice('example_'.length) : fullName;
  return conciseName.length > 20 ? conciseName.slice(0, 18) + '…' : conciseName;
}}

function placeChangeGraphLabels(selection, forceNodeById) {{
  const placed = [];
  const nodePoints = [...forceNodeById.values()];
  const entries = selection.nodes().map(element => ({{
    element,
    node: element.__data__,
    label: d3.select(element).select('.node-label'),
  }})).filter(entry => entry.label.node() && forceNodeById.has(String(entry.node.id)));
  entries.sort((a, b) => b.label.node().getComputedTextLength() - a.label.node().getComputedTextLength());

  function overlapArea(a, b) {{
    return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
      * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  }}

  entries.forEach(entry => {{
    const point = forceNodeById.get(String(entry.node.id));
    const textWidth = Math.max(entry.label.node().getComputedTextLength(), 20);
    const textHeight = 12;
    const candidates = [
      {{x: 16, y: 4, anchor: 'start', left: point.x + 16, top: point.y - 6}},
      {{x: -16, y: 4, anchor: 'end', left: point.x - 16 - textWidth, top: point.y - 6}},
      {{x: 0, y: -16, anchor: 'middle', left: point.x - textWidth / 2, top: point.y - 27}},
      {{x: 0, y: 22, anchor: 'middle', left: point.x - textWidth / 2, top: point.y + 10}},
      {{x: 12, y: -14, anchor: 'start', left: point.x + 12, top: point.y - 25}},
      {{x: -12, y: -14, anchor: 'end', left: point.x - 12 - textWidth, top: point.y - 25}},
      {{x: 12, y: 20, anchor: 'start', left: point.x + 12, top: point.y + 8}},
      {{x: -12, y: 20, anchor: 'end', left: point.x - 12 - textWidth, top: point.y + 8}},
    ].map(candidate => ({{
      ...candidate,
      right: candidate.left + textWidth,
      bottom: candidate.top + textHeight,
    }}));
    candidates.forEach(candidate => {{
      candidate.score = placed.reduce((score, box) => score + overlapArea(candidate, box) * 4, 0);
      candidate.score += nodePoints.reduce((score, node) => {{
        if (String(node.id) === String(entry.node.id)) return score;
        return score + (node.x >= candidate.left - 5 && node.x <= candidate.right + 5
          && node.y >= candidate.top - 5 && node.y <= candidate.bottom + 5 ? 120 : 0);
      }}, 0);
    }});
    const choice = candidates.reduce((best, candidate) => candidate.score < best.score ? candidate : best);
    entry.label.attr('x', choice.x).attr('y', choice.y).attr('dy', 0).attr('text-anchor', choice.anchor);
    placed.push(choice);
  }});
}}

function updateChangeChrome(scope, graph) {{
  const fullMode = externalChanges.mode === 'full';
  const changed = graph.nodes.filter(node => node.status !== 'context').length;
  const context = graph.nodes.length - changed;
  const target = document.getElementById(scope === 'feature' ? 'stats-feat' : scope === 'dependency' ? 'stats-dep' : 'stats-map');
  if (fullMode && scope === 'dependency') {{
    const semanticCount = graph.edges.filter(edge => edge.relation !== 'contains').length;
    const hierarchyCount = graph.edges.length - semanticCount;
    target.innerHTML = `<span class="stat">Nodes: <b>${{graph.nodes.length}}</b></span>`
      + `<span class="stat">Semantic: <b>${{semanticCount}}</b></span>`
      + `<span class="stat">Hierarchy: <b>${{hierarchyCount}}</b></span>`;
  }} else {{
    target.innerHTML = `<span class="stat">Changed: <b>${{changed}}</b></span>`
      + `<span class="stat">Context: <b>${{context}}</b></span>`
      + `<span class="stat">Visible relations: <b>${{graph.edges.length}}</b></span>`;
  }}
  const relationLegend = scope === 'dependency'
    ? `<div class="ov-section"><div class="ov-title">Directed relations</div>
       <div class="legend-item"><div class="legend-line" style="background:#f0883e"></div>imports</div>
       <div class="legend-item"><div class="legend-line" style="background:#3fb950"></div>invokes</div>
       <div class="legend-item"><div class="legend-line" style="background:#a371f7"></div>inherits</div>
       <div class="legend-item"><div class="legend-line" style="border-top:1px dashed #6e7681"></div>contains</div></div>`
    : '';
  const nodeTypeLegend = scope === 'dependency'
    ? `<div class="ov-section"><div class="ov-title">Node fill · code type</div>
       <div class="legend-item"><span style="width:10px;height:10px;border-radius:50%;background:#1f6feb"></span>directory</div>
       <div class="legend-item"><span style="width:10px;height:10px;border-radius:50%;background:#3fb950"></span>file / module</div>
       <div class="legend-item"><span style="width:10px;height:10px;border-radius:50%;background:#a371f7"></span>class</div>
       <div class="legend-item"><span style="width:10px;height:10px;border-radius:50%;background:#d2a8ff"></span>function / method</div></div>`
    : '';
  const changeLegend = fullMode ? '' : `<div class="ov-title">Change status</div>
    <div class="legend-item"><span style="color:#3fb950;font-weight:800">+</span> Added</div>
    <div class="legend-item"><span style="color:#f85149;font-weight:800">−</span> Removed</div>
    <div class="legend-item"><span style="color:#d29922;font-weight:800">~</span> Modified</div>
    <div class="legend-item"><span style="color:#8b949e;font-weight:800">○</span> Context</div>`;
  document.getElementById('canvas-overlay').innerHTML = `${{changeLegend}}${{nodeTypeLegend}}${{relationLegend}}
    <div class="ov-section"><span class="ov-key">Arrow</span> source → target</div>
    <div class="ov-section"><span class="ov-key">Click</span> node — focus<br><span class="ov-key">Scroll</span> — zoom</div>`;
}}

function drawChangeGraph() {{
  if (activeTab !== 'dep') {{
    if (changeSimulation) changeSimulation.stop();
    gChange.style('display', 'none');
    return;
  }}
  const scope = 'dependency';
  const graph = externalChanges.mode === 'full' ? buildFullDependencyGraph() : buildChangeGraph(scope);
  updateChangeChrome(scope, graph);
  gFeat.style('display', 'none'); gDep.style('display', 'none'); gMap.style('display', 'none');
  gChange.style('display', null);
  svg.on('.zoom', null).call(zoomChange);
  if (changeSimulation) changeSimulation.stop();
  gChange.selectAll('text.change-empty').remove();
  if (!graph.nodes.length) {{
    changeLinkLayer.selectAll('*').remove();
    changeNodeLayer.selectAll('*').remove();
    gChange.append('text').attr('class', 'change-empty').attr('x', width / 2).attr('y', height / 2)
      .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 15)
      .text(`No ${{externalChanges.filter}} ${{scope}} nodes in this change`);
    return;
  }}

  const links = changeLinkLayer.selectAll('line.change-graph-link').data(graph.edges, graphEdgeKey);
  links.exit().remove();
  const linkAll = links.enter().append('line').merge(links)
    .attr('class', edge => `change-graph-link rel-${{String(edge.relation || 'related').toLowerCase().replace(/[^a-z0-9_-]+/g, '-')}} ${{edge.status || 'context'}}`)
    .attr('marker-end', edge => `url(#arrow-rel-${{directedEdgeColors[edge.relation] ? edge.relation : 'default'}})`);

  const selection = changeNodeLayer.selectAll('g.change-graph-node').data(graph.nodes, node => node.id);
  selection.exit().remove();
  const entered = selection.enter().append('g').on('click', (event, node) => {{
    event.stopPropagation();
    focusChangeNode(node.id, node.scope);
  }});
  entered.append('circle').attr('r', 11);
  entered.append('title');
  entered.append('text').attr('class', 'status-glyph').attr('dy', 3.5);
  entered.append('text').attr('class', 'node-label').attr('x', 16).attr('dy', 4);
  const all = entered.merge(selection)
    .attr('class', node => `change-graph-node ${{node.status}}${{externalChanges.focus?.node_id === node.id ? ' focused' : ''}}`)
    .style('--node-fill', node => nodeTypeColors[node.node_type] || nodeTypeColors.default);
  all.select('.status-glyph').text(node => changeNodeGlyph(node.status));
  all.select('title').text(node => `${{node.name || node.id}}\n${{node.id}}\n${{node.status}}`);
  all.select('.node-label').text(node => {{
    return dependencyDisplayLabel(node);
  }});

  const forceNodes = graph.nodes.map(node => ({{...node}}));
  const forceNodeById = new Map(forceNodes.map(node => [String(node.id), node]));
  changeSimulation = d3.forceSimulation(forceNodes)
    .force('link', d3.forceLink(graph.edges).id(node => node.id)
      .distance(edge => edge.relation === 'contains' ? 62 : 108)
      .strength(edge => edge.relation === 'contains' ? .78 : .58))
    .force('charge', d3.forceManyBody().strength(graph.nodes.length > 80 ? -55 : -135).distanceMax(280))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(node => 42 + Math.min(String(node.name || node.id).length, 20) * 2.8).strength(.96))
    .alpha(.9).alphaDecay(.045)
    .stop();
  for (let iteration = 0; iteration < 180; iteration++) changeSimulation.tick();
  linkAll.attr('x1', edge => edge.source.x).attr('y1', edge => edge.source.y)
    .attr('x2', edge => edge.target.x).attr('y2', edge => edge.target.y);
  all.attr('transform', node => {{
    const forceNode = forceNodeById.get(String(node.id));
    return forceNode ? `translate(${{forceNode.x}},${{forceNode.y}})` : '';
  }});
  placeChangeGraphLabels(all, forceNodeById);
  const focusedId = externalChanges.focus?.scope === 'dependency' ? String(externalChanges.focus.node_id) : null;
  const neighborIds = new Set(focusedId ? [focusedId] : []);
  if (focusedId) graph.edges.forEach(edge => {{
    const sourceId = edgeId(edge.source), targetId = edgeId(edge.target);
    if (sourceId === focusedId || targetId === focusedId) {{ neighborIds.add(sourceId); neighborIds.add(targetId); }}
  }});
  linkAll
    .classed('focus-related', edge => {{
      if (!focusedId) return false;
      const sourceId = edgeId(edge.source), targetId = edgeId(edge.target);
      return sourceId === focusedId || targetId === focusedId;
    }})
    .classed('focus-dim', edge => {{
      if (!focusedId) return false;
      const sourceId = edgeId(edge.source), targetId = edgeId(edge.target);
      return sourceId !== focusedId && targetId !== focusedId;
    }});
  all
    .classed('focus-neighbor', node => !!focusedId && neighborIds.has(String(node.id)))
    .classed('focus-dim', node => !!focusedId && !neighborIds.has(String(node.id)));
  svg.on('click.change-deselect', event => {{
    if (event.target !== svg.node() || activeTab !== 'dep' || !externalChanges.focus) return;
    externalChanges.focus = null;
    drawChangeGraph();
    if (typeof renderChangesPanel === 'function') renderChangesPanel();
  }});
  requestAnimationFrame(fitChangeGraph);
}}

function fitChangeGraph() {{
  if (activeTab !== 'dep') return;
  const rightInset = (typeof cpOpen !== 'undefined' && cpOpen) ? 342 : 30;
  const leftInset = 30, topInset = 96, bottomInset = 30;
  const usableWidth = Math.max(width - leftInset - rightInset, 120);
  const usableHeight = Math.max(height - topInset - bottomInset, 120);
  const centerX = leftInset + usableWidth / 2;
  const centerY = topInset + usableHeight / 2;
  const focused = externalChanges.focus?.node_id;
  if (focused) {{
    const target = changeNodeLayer.selectAll('g.change-graph-node').filter(node => node.id === focused).node();
    if (target) {{
      const transform = target.getAttribute('transform') || '';
      const match = transform.match(/translate\\(([-\\d.]+),([-\\d.]+)\\)/);
      if (match) {{
        const scale = 1.25;
        svg.call(zoomChange.transform, d3.zoomIdentity.translate(centerX - Number(match[1]) * scale, centerY - Number(match[2]) * scale).scale(scale));
        return;
      }}
    }}
  }}
  const bounds = gChange.node().getBBox();
  if (!bounds.width || !bounds.height) return;
  const fitScale = Math.min(usableWidth / (bounds.width + 80), usableHeight / (bounds.height + 80), 1.4) * .9;
  const scale = Math.max(fitScale, .5);
  if (!Number.isFinite(scale) || scale <= 0) return;
  svg.call(zoomChange.transform, d3.zoomIdentity
    .translate(centerX - (bounds.x + bounds.width / 2) * scale, centerY - (bounds.y + bounds.height / 2) * scale)
    .scale(scale));
}}

function drawRemovedRail() {{
  const removed = [
    ...(externalChanges.feature.rows?.removed || []),
    ...(externalChanges.dependency.rows?.removed || []),
  ];
  if (externalChanges.mode !== 'full' || !externalChanges.emphasize || !removed.length) {{
    gRemovedRail.style('display', 'none').selectAll('*').remove();
    return;
  }}
  const shown = removed.slice(0, 12);
  const railWidth = 225, rowHeight = 24;
  const railHeight = 44 + shown.length * rowHeight + (removed.length > shown.length ? 22 : 0);
  gRemovedRail.style('display', null).attr('transform', `translate(${{width - railWidth - 14}},72)`);
  gRemovedRail.selectAll('*').remove();
  gRemovedRail.append('rect').attr('class', 'removed-rail-bg').attr('width', railWidth).attr('height', railHeight);
  gRemovedRail.append('text').attr('class', 'removed-rail-title').attr('x', 12).attr('y', 22).text(`− Removed nodes (${{removed.length}})`);
  const row = gRemovedRail.selectAll('g.removed-rail-node').data(shown).enter().append('g')
    .attr('class', 'removed-rail-node').attr('transform', (node, index) => `translate(15,${{40 + index * rowHeight}})`);
  row.append('circle').attr('r', 6);
  row.append('text').attr('x', 12).attr('dy', 3.5).text(node => {{
    const name = node.name || node.node_id;
    return name.length > 28 ? name.slice(0, 26) + '…' : name;
  }});
  if (removed.length > shown.length) gRemovedRail.append('text').attr('class', 'removed-rail-more')
    .attr('x', 15).attr('y', 44 + shown.length * rowHeight).text(`+${{removed.length - shown.length}} more in Current Changes`);
}}

function updateChangeSummary() {{
  const target = document.getElementById('change-summary');
  if (!externalChanges.active) {{ target.hidden = true; return; }}
  const f = externalChanges.feature, d = externalChanges.dependency;
  const count = (sets, kind) => externalChanges.filter === 'all' || externalChanges.filter === kind ? sets[kind].size : 0;
  target.innerHTML = '<b>Current changes</b>'
    + `<span class="added">+${{count(f,'added') + count(d,'added')}}</span>`
    + `<span class="removed">−${{count(f,'removed') + count(d,'removed')}}</span>`
    + `<span class="modified">~${{count(f,'modified') + count(d,'modified')}}</span>`;
  target.hidden = false;
}}

window.addEventListener('message', event => {{
  if (event.source !== window.parent) return;
  if (event.data?.type === 'cmind:theme') {{ applyRpgTheme(event.data.theme, false); return; }}
  if (event.data?.type !== 'cmind:rpg-highlight') return;
  externalChanges = {{
    active: !!event.data.emphasize,
    mode: event.data.mode || 'changes',
    filter: event.data.filter || 'all',
    contextMode: event.data.contextMode || 'context',
    emphasize: !!event.data.emphasize,
    focus: event.data.focus || null,
    feature: externalNodeSets(event.data.feature),
    dependency: externalNodeSets(event.data.dependency),
    semanticEdges: event.data.semanticEdges || {{}},
    dependencyEdges: event.data.dependencyEdges || {{}},
    hierarchyEdges: event.data.hierarchyEdges || {{}},
    mappingEdges: event.data.mappingEdges || {{}},
  }};
  if (externalChanges.mode === 'changes') {{
    gRemovedRail.style('display', 'none');
    const requestedTab = externalChanges.focus?.scope === 'dependency' ? 'dep' : externalChanges.focus?.scope === 'feature' ? 'feat' : activeTab;
    switchTab(requestedTab || 'feat');
    updateChangeSummary();
    return;
  }}
  if (changeSimulation) changeSimulation.stop();
  gChange.style('display', 'none');
  clearExternalChangeHighlights();
  const featureVisible = new Set([...externalChanges.feature.added, ...externalChanges.feature.modified]);
  const dependencyVisible = new Set([...externalChanges.dependency.added, ...externalChanges.dependency.modified]);
  if (externalChanges.emphasize) {{
    if (featureVisible.size) {{
      expandFeatureChangePaths(featureVisible);
      update(root);
    }}
    if (dependencyVisible.size) {{
      depInit();
      expandDependencyChangePaths(dependencyVisible);
      depRedraw();
    }}
  }}
  switchTab(activeTab || 'feat');
  updateChangeSummary();
  if (externalChanges.emphasize) applyExternalChangeHighlights();
  else document.getElementById('change-summary').hidden = true;
  drawRemovedRail();
}});

// ── Search ──
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.toLowerCase().trim();

  if (activeTab === 'feat') {{
    if (q.length < 2) {{
      nodeLayer.selectAll('.node-dot').attr('stroke', '#30363d').attr('stroke-width', 1.5);
      return;
    }}
    root.descendants().forEach(d => {{
      const name = (d.data.name || d.data.id || '').toLowerCase();
      const path = (d.data.meta?.path || '').toLowerCase();
      if (name.includes(q) || path.includes(q)) {{
        let p = d.parent;
        while (p) {{
          if (p._children) {{ p.children = p._children; p._children = null; }}
          p = p.parent;
        }}
      }}
    }});
    update(root);
    nodeLayer.selectAll('g.node').each(function(d) {{
      const name = (d.data.name || d.data.id || '').toLowerCase();
      const path = (d.data.meta?.path || '').toLowerCase();
      const match = name.includes(q) || path.includes(q);
      d3.select(this).select('circle')
        .attr('stroke', match ? '#f0883e' : '#30363d')
        .attr('stroke-width', match ? 3 : 1.5);
    }});
  }} else if (activeTab === 'dep') {{
    const depNodeEls = depNodeG.selectAll('g.dep-node');
    const depLinkEls = depLinkG.selectAll('line.dep-link');
    if (q.length < 2) {{
      depNodeEls.select('circle').attr('stroke', '#30363d').attr('stroke-width', 1.5);
      depNodeEls.style('opacity', 1);
      depLinkEls.style('opacity', showEdges ? 0.5 : 0);
      return;
    }}
    depNodeEls.each(function(d) {{
      const name = (d.name || d.id || '').toLowerCase();
      const mod = (d.module || '').toLowerCase();
      const match = name.includes(q) || mod.includes(q) || d.id.toLowerCase().includes(q);
      d3.select(this).select('circle')
        .attr('stroke', match ? '#f0883e' : '#30363d')
        .attr('stroke-width', match ? 3 : 1.5);
      d3.select(this).style('opacity', match ? 1 : 0.15);
    }});
    depLinkEls.style('opacity', d => {{
      if (!showEdges) return 0;
      const sn = (d.source.name || d.source.id || '').toLowerCase();
      const tn = (d.target.name || d.target.id || '').toLowerCase();
      const si = (d.source.id || '').toLowerCase();
      const ti = (d.target.id || '').toLowerCase();
      return (sn.includes(q) || tn.includes(q) || si.includes(q) || ti.includes(q)) ? 0.7 : 0.03;
    }});
  }} else if (activeTab === 'map') {{
    if (q.length < 2) {{
      mapFeatNodeLayer.selectAll('g').style('opacity', null);
      mapDepNodeLayer.selectAll('g').style('opacity', null);
      return;
    }}
    mapFeatNodeLayer.selectAll('g.map-feat-node').each(function(d) {{
      const match = (d.data.name||d.data.id||'').toLowerCase().includes(q);
      d3.select(this).style('opacity', match ? 1 : 0.15);
    }});
    mapDepNodeLayer.selectAll('g.map-dep-node').each(function(d) {{
      const match = (d.data.name||d.data.id||'').toLowerCase().includes(q);
      d3.select(this).style('opacity', match ? 1 : 0.15);
    }});
  }}
}});

// ══ Self-contained change workbench (in-page controls; no parent needed) ══
let cpOpen = true;
function cpEsc(s) {{ return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function cpPath(p) {{ return Array.isArray(p) ? p.join(' / ') : (p == null ? '' : String(p)); }}
function cpScope() {{ return activeTab === 'dep' ? 'dependency' : 'feature'; }}

function renderStatusCounts() {{
  const c = (cpScope() === 'dependency' ? changeData.dependency_nodes : changeData.feature_nodes) || {{}};
  const k = c.counts || {{added: 0, removed: 0, modified: 0}};
  document.getElementById('cnt-added').textContent = k.added || 0;
  document.getElementById('cnt-removed').textContent = k.removed || 0;
  document.getElementById('cnt-modified').textContent = k.modified || 0;
  document.getElementById('cnt-all').textContent = (k.added || 0) + (k.removed || 0) + (k.modified || 0);
}}

function cpFilteredRows() {{
  const q = (document.getElementById('cp-search').value || '').toLowerCase().trim();
  let rows = changeRows(cpScope());
  if (externalChanges.filter !== 'all') rows = rows.filter(r => r.status === externalChanges.filter);
  if (q) rows = rows.filter(r => ((r.name || '') + ' ' + (r.id || '') + ' ' + cpPath(r.path)).toLowerCase().includes(q));
  return rows;
}}

function renderChangeDetail(node) {{
  const el = document.getElementById('cp-detail');
  if (!node) {{ el.className = 'cp-detail empty'; el.textContent = 'Select a changed node to see what changed.'; return; }}
  el.className = `cp-detail selected ${{node.status}}`;
  const dl = [];
  dl.push(`<dt>Status</dt><dd>${{node.status}}</dd>`);
  if (node.node_type) dl.push(`<dt>Type</dt><dd>${{cpEsc(node.node_type)}}</dd>`);
  dl.push(`<dt>Path</dt><dd>${{cpEsc(cpPath(node.path) || node.id)}}</dd>`);
  if (node.changed_fields && node.changed_fields.length) dl.push(`<dt>Changed</dt><dd>${{cpEsc(node.changed_fields.join(', '))}}</dd>`);
  if (node.previous_parent_id) dl.push(`<dt>Prev parent</dt><dd>${{cpEsc(node.previous_parent_id)}}</dd>`);
  if (node.parent_id) dl.push(`<dt>${{node.status === 'removed' ? 'Old parent' : 'Parent'}}</dt><dd>${{cpEsc(node.parent_id)}}</dd>`);
  const statusLabel = node.status === 'added' ? 'Added' : node.status === 'removed' ? 'Removed' : node.status === 'modified' ? 'Modified' : 'Context';
  el.innerHTML = `<div class="cp-d-title"><span class="cp-mark ${{node.status}}"><b>${{changeNodeGlyph(node.status)}}</b>${{statusLabel}}</span>${{cpEsc(node.name || node.id)}}</div><dl>${{dl.join('')}}</dl>`;
}}

function renderChangesPanel() {{
  if (!hasChangeData) return;
  renderStatusCounts();
  document.getElementById('cp-title').textContent = (cpScope() === 'dependency' ? 'Dependency' : 'Feature') + ' changes';
  const version = document.getElementById('cp-version');
  const range = (changeData.parent_short || 'empty') + ' → ' + (changeData.short_commit || 'current');
  const operation = String(changeData.operation || 'RPG update').replace(/--json/g, '').trim();
  version.textContent = 'RPG versions ' + range + ' · ' + operation;
  version.title = 'Meta-git RPG versions ' + range + (changeData.committed_at ? ' · ' + changeData.committed_at : '') + ' · ' + operation;
  const rows = cpFilteredRows();
  document.getElementById('cp-count').textContent = rows.length + ' node' + (rows.length === 1 ? '' : 's');
  const focusId = externalChanges.focus?.node_id;
  const listEl = document.getElementById('cp-list');
  listEl.innerHTML = rows.map(r => {{
    const statusLabel = r.status === 'added' ? 'Added' : r.status === 'removed' ? 'Removed' : r.status === 'modified' ? 'Modified' : 'Context';
    return `<button class="cp-row ${{r.status}}${{focusId === r.id ? ' selected' : ''}}" data-cp-id="${{cpEsc(r.id)}}" aria-pressed="${{focusId === r.id}}">`
    + `<span class="cp-mark ${{r.status}}"><b>${{changeNodeGlyph(r.status)}}</b>${{statusLabel}}</span>`
    + `<span class="cp-row-main"><span class="cp-name">${{cpEsc(r.name || r.id)}}</span><span class="cp-path">${{cpEsc(cpPath(r.path) || r.id)}}</span></span></button>`;
  }}).join('')
    || '<div style="color:#6e7681;font-size:11px;padding:12px;text-align:center">No matching nodes.</div>';
  listEl.querySelectorAll('[data-cp-id]').forEach(el => {{ el.onclick = () => selectChangeNode(el.getAttribute('data-cp-id')); }});
  renderChangeDetail(rows.find(r => r.id === focusId) || null);
  const selectedRow = listEl.querySelector('.cp-row.selected');
  if (selectedRow) requestAnimationFrame(() => selectedRow.scrollIntoView({{block: 'nearest'}}));
}}

function selectChangeNode(id) {{
  focusChangeNode(String(id), cpScope());
}}

function syncChangesPanelVisibility() {{
  const show = cpOpen && hasChangeData && activeTab !== 'map';
  document.getElementById('changes-panel').style.display = show ? 'flex' : 'none';
  document.body.classList.toggle('cp-open', show);
  const btn = document.getElementById('btn-changes-panel');
  if (btn) {{
    btn.style.display = activeTab === 'map' ? 'none' : '';
    btn.textContent = cpOpen ? 'Hide changes' : 'Show changes';
  }}
}}
function openChangesPanel(open) {{
  cpOpen = open;
  syncChangesPanelVisibility();
}}
function toggleChangesPanel() {{ openChangesPanel(!cpOpen); requestAnimationFrame(fitCurrent); }}

function setStatusFilter(filter) {{
  externalChanges.filter = filter;
  externalChanges.focus = null;
  document.querySelectorAll('#status-seg button').forEach(b => b.classList.toggle('active', b.dataset.status === filter));
  if (activeTab === 'dep') drawChangeGraph();
  else applyChangeEmphasis();
  renderChangesPanel();
}}

function setMode(mode) {{
  externalChanges.mode = mode;
  externalChanges.active = (mode === 'changes');
  externalChanges.focus = null;
  document.getElementById('mode-changes').classList.toggle('active', mode === 'changes');
  document.getElementById('mode-full').classList.toggle('active', mode === 'full');
  if (mode === 'changes' && activeTab === 'map') activeTab = 'feat';
  openChangesPanel(cpOpen);
  switchTab(activeTab);
  renderChangesPanel();
}}

function openMapping() {{
  if (externalChanges.mode === 'changes') setMode('full');
  switchTab('map');
}}

function fitCurrent() {{
  if (activeTab === 'dep') {{ fitChangeGraph(); return; }}
  if (activeTab === 'feat') {{ fitFeatEmphasis(); return; }}
  if (activeTab === 'dep') {{ depFitVisible(); return; }}
  if (activeTab === 'map') {{ switchTab('map'); return; }}
}}
function resetCurrent() {{
  externalChanges.focus = null;
  clearGraphFocusVisuals();
  if (activeTab === 'dep') {{ drawChangeGraph(); return; }}
  if (activeTab === 'feat') fitFeatEmphasis();
  else if (activeTab === 'dep') depFitVisible();
  else svg.call(zoomMap.transform, d3.zoomIdentity);
  if (typeof renderChangesPanel === 'function') renderChangesPanel();
}}

function initChangeWorkbench() {{
  if (!hasChangeData) {{ document.getElementById('header-row2').style.display = 'none'; return; }}
  externalChanges = {{
    active: true, mode: 'changes', filter: 'all', contextMode: 'context', emphasize: false, focus: null,
    feature: externalNodeSets(changeData.feature_nodes),
    dependency: externalNodeSets(changeData.dependency_nodes),
    semanticEdges: changeData.semantic_edges || {{}},
    dependencyEdges: changeData.dependency_edges || {{}},
    hierarchyEdges: changeData.feature_hierarchy_edges || {{}},
    mappingEdges: changeData.mapping_edges || {{}},
  }};
  document.getElementById('header-row2').style.display = 'flex';
  document.getElementById('btn-changes-panel').style.display = '';
  document.getElementById('cp-search').addEventListener('input', renderChangesPanel);
  setMode('changes');
}}

// ── Init feat graph ──
svg.call(zoomFeat);
svg.call(zoomFeat.transform, d3.zoomIdentity.translate(margin.left, margin.top));
root.x0 = height / 2;
root.y0 = 0;
update(root);
initChangeWorkbench();

setTimeout(() => {{
  const bounds = gFeat.node().getBBox();
  if (!bounds.width || !bounds.height) return;
  const fullWidth = bounds.width + margin.left + margin.right;
  const fullHeight = bounds.height + margin.top + margin.bottom;
  const scale = Math.min(width / fullWidth, height / fullHeight, 1) * 0.9;
  svg.call(zoomFeat.transform,
    d3.zoomIdentity
      .translate(width / 2 - bounds.x * scale - bounds.width * scale / 2,
                 height / 2 - bounds.y * scale - bounds.height * scale / 2)
      .scale(scale));
}}, 400);
}}
</script>
</body>
</html>"""


def main():
    from common.paths import RPG_FILE

    parser = argparse.ArgumentParser(description="Visualize RPG as interactive graph")
    parser.add_argument("rpg_file", nargs="?", default=str(RPG_FILE),
                        help="Path to rpg.json (default: home-side workspace store at ~/.cmind/workspaces/<id>/data/rpg.json)")
    parser.add_argument("--dep-graph", default=None,
              help=(
                "Optional legacy external dep_graph override. "
                "By default the embedded dep_graph in rpg.json is used."
              ))
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file (default: <rpg_file>.html)")
    parser.add_argument("--change-data", default=None,
                        help="Optional rpg_latest_change JSON to embed for the in-page Current Changes view")
    args = parser.parse_args()

    rpg_path = Path(args.rpg_file).expanduser()
    if not rpg_path.exists():
        print(f"Error: {rpg_path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_rpg(rpg_path, args.dep_graph)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    change_data = None
    if args.change_data:
        change_path = Path(args.change_data).expanduser()
        if change_path.is_file():
            change_data = json.loads(change_path.read_text(encoding="utf-8"))
    html_content = generate_html(data, change_data)

    output = args.output or str(rpg_path.with_suffix(".html"))
    Path(output).write_text(html_content, encoding="utf-8")

    tree = normalize_to_tree(data)
    dep = extract_dep_graph(data)
    dep_map = data.get("_dep_to_rpg_map", {})
    n = count_nodes(tree)
    e = len(get_semantic_edges(data))
    print(f"Generated: {output}")
    print(f"  Feat Graph — Nodes: {n}, Semantic Edges: {e}")
    print(f"  Dep Graph  — Nodes: {len(dep['nodes'])}, Edges: {len(dep['edges'])}")
    print(f"  Mapping    — {len(dep_map)} dep nodes -> {sum(len(v) for v in dep_map.values())} RPG features")
    print(f"  Open in browser: file://{Path(output).resolve()}")


if __name__ == "__main__":
    main()
