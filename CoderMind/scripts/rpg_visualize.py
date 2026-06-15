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


def generate_html(data: dict) -> str:
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

    map_count = sum(len(v) for v in dep_to_rpg.values())
    has_map = len(dep_to_rpg) > 0

    tree_json = json.dumps(tree)
    edges_json = json.dumps(semantic_edges)
    dep_nodes_json = json.dumps(dep["nodes"])
    dep_edges_json = json.dumps(dep["edges"])
    dep_parent_json = json.dumps(dep["parent_map"])
    dep_tree_json = json.dumps(dep_tree)
    dep_to_rpg_json = json.dumps(dep_to_rpg)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RPG: {repo_name}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
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

.link {{ fill: none; stroke: #21262d; stroke-width: 1; }}
.node circle {{ stroke: #30363d; stroke-width: 1.5; cursor: pointer; }}
.node text {{ font-size: 13px; fill: #c9d1d9; pointer-events: none; }}
.node-collapsed circle {{ fill: #1f6feb !important; }}

.edge-imports {{ stroke: #f0883e; stroke-opacity: 0.5; }}
.edge-inherits {{ stroke: #a371f7; stroke-opacity: 0.6; }}
.edge-invokes {{ stroke: #3fb950; stroke-opacity: 0.5; }}
.edge-references {{ stroke: #79c0ff; stroke-opacity: 0.4; }}
.edge-default {{ stroke: #8b949e; stroke-opacity: 0.3; }}

/* Dep graph specific */
.dep-link {{ fill: none; stroke-width: 1.2; stroke-opacity: 0.5; }}
.dep-link-imports {{ stroke: #f0883e; }}
.dep-link-invokes {{ stroke: #3fb950; }}
.dep-link-inherits {{ stroke: #a371f7; }}
.dep-link-default {{ stroke: #8b949e; }}
.dep-node circle {{ cursor: pointer; stroke: #30363d; stroke-width: 1.5; }}
.dep-node text {{ font-size: 10px; fill: #c9d1d9; pointer-events: none; }}
.dep-node-collapsed circle {{ stroke: #58a6ff; stroke-width: 2; }}

.map-link {{ fill: none; stroke: #f0883e; stroke-opacity: 0.35; stroke-width: 1.2; }}
.map-link-hover {{ stroke-opacity: 0.9; stroke-width: 2.5; }}
.map-node-highlight circle {{ stroke: #f0883e !important; stroke-width: 3 !important; }}

.no-data {{ display: flex; align-items: center; justify-content: center;
            height: 80vh; color: #484f58; font-size: 16px; }}

.tooltip {{ position: fixed; background: #1c2128; border: 1px solid #30363d; padding: 8px 12px;
            border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 200;
            max-width: 350px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.tooltip .tt-name {{ color: #58a6ff; font-weight: bold; }}
.tooltip .tt-type {{ color: #8b949e; font-size: 11px; }}
.tooltip .tt-path {{ color: #7ee787; font-size: 11px; }}
.tooltip .tt-edges {{ color: #f0883e; font-size: 11px; margin-top: 4px; }}
</style>
</head>
<body>
<div id="header">
  <h1>RPG: {repo_name}</h1>
  <div id="tabs">
    <button id="tab-feat" class="active" onclick="switchTab('feat')">Feat Graph</button>
    <button id="tab-dep" onclick="switchTab('dep')">Dep Graph</button>
    <button id="tab-map" onclick="switchTab('map')">Mapping</button>
  </div>
  <div id="stats-feat">
    <span class="stat">Nodes: <b>{feat_node_count}</b></span>
    <span class="stat">Edges: <b>{feat_edge_count}</b> ({feat_edge_summary})</span>
  </div>
  <div id="stats-dep" style="display:none">
    <span class="stat">Nodes: <b>{dep_node_count}</b></span>
    <span class="stat">Edges: <b>{dep_edge_count}</b> ({dep_edge_summary})</span>
    <span class="stat" id="dep-visible-stat"></span>
  </div>
  <div id="stats-map" style="display:none">
    <span class="stat">Mappings: <b>{len(dep_to_rpg)}</b> dep nodes → <b>{map_count}</b> RPG features</span>
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
  </div>
  <div id="legend">
  </div>
</div>
<div id="tooltip" class="tooltip" style="display:none"></div>
<div id="canvas-overlay">
  <div class="ov-title">Legend</div>
  <div class="legend-item"><div class="legend-line" style="background:#f0883e"></div>imports</div>
  <div class="legend-item"><div class="legend-line" style="background:#a371f7"></div>inherits</div>
  <div class="legend-item"><div class="legend-line" style="background:#3fb950"></div>invokes</div>
  <div class="legend-item"><div class="legend-line" style="background:#79c0ff"></div>references</div>
  <div class="ov-section">
    <div class="ov-title">Controls</div>
    <span class="ov-key">Click</span> node/group — select &amp; filter edges<br>
    <span class="ov-key">Double-click</span> node — expand / collapse<br>
    <span class="ov-key">Double-click</span> group border — collapse<br>
    <span class="ov-key">Click</span> background — deselect<br>
    <span class="ov-key">Drag</span> node — move<br>
    <span class="ov-key">Scroll</span> — zoom
  </div>
</div>
<svg id="canvas"></svg>

<script>
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

// ── Tab state ──
let activeTab = 'feat';
let showEdges = true;

// ══════════════════════════════════════════════
// FEAT GRAPH (existing tree layout)
// ══════════════════════════════════════════════

const gFeat = svg.append('g').attr('class', 'feat-group').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
const edgeLayer = gFeat.append('g').attr('class', 'edge-layer');
const linkLayer = gFeat.append('g').attr('class', 'link-layer');
const nodeLayer = gFeat.append('g').attr('class', 'node-layer');

const zoomFeat = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => gFeat.attr('transform', e.transform));

let featSelectedNode = null;
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

function update(source) {{
  const treeDataLayout = treemap(root);
  const nodes = treeDataLayout.descendants();
  const links = treeDataLayout.links();
  nodes.forEach(d => {{ d.y = d.depth * 220; }});

  nodeById = {{}};
  nodes.forEach(d => {{ nodeById[d.data.id] = d; }});

  const node = nodeLayer.selectAll('g.node').data(nodes, d => d.data.id);
  const nodeEnter = node.enter().append('g')
    .attr('class', d => 'node' + (d._children ? ' node-collapsed' : ''))
    .attr('transform', `translate(${{source.y0 || 0}},${{source.x0 || 0}})`)
    .on('click', (event, d) => {{
      event.stopPropagation();
      // Single click: select/deselect node
      const circle = d3.select(event.currentTarget).select('circle');
      if (featSelectedNode === d) {{
        featSelectedNode = null;
        nodeLayer.selectAll('circle').attr('stroke-width', 1.5).attr('stroke', '#30363d');
      }} else {{
        featSelectedNode = d;
        nodeLayer.selectAll('circle').attr('stroke-width', 1.5).attr('stroke', '#30363d');
        circle.attr('stroke', '#f0883e').attr('stroke-width', 2.5);
      }}
    }})
    .on('dblclick', (event, d) => {{
      event.stopPropagation();
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      update(d);
    }})
    .on('mouseover', showTooltipFeat)
    .on('mouseout', hideTooltip);

  nodeEnter.append('circle')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', getNodeColor);
  nodeEnter.append('text')
    .attr('dy', 3.5)
    .attr('x', -10)
    .attr('text-anchor', 'end')
    .text(d => {{
      const name = d.data.name || d.data.id;
      return name.length > 35 ? name.slice(0, 33) + '...' : name;
    }});

  const nodeUpdate = nodeEnter.merge(node);
  nodeUpdate.transition().duration(300)
    .attr('transform', d => `translate(${{d.y}},${{d.x}})`)
    .attr('class', d => 'node' + (d._children ? ' node-collapsed' : ''));
  nodeUpdate.select('circle')
    .attr('r', d => d._children ? 5 : (d.children ? 4 : 3))
    .attr('fill', getNodeColor);
  nodeUpdate.select('text')
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
  nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
}}

function diagonal(s, d) {{
  return `M${{s.y}},${{s.x}} C${{(s.y + d.y) / 2}},${{s.x}} ${{(s.y + d.y) / 2}},${{d.x}} ${{d.y}},${{d.x}}`;
}}

// Click background to deselect feat node
svg.on('click.feat-deselect', () => {{
  featSelectedNode = null;
  nodeLayer.selectAll('circle').attr('stroke-width', 1.5).attr('stroke', '#30363d');
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
    edgeLayer.append('path')
      .attr('class', 'semantic-edge ' + cls)
      .attr('d', `M${{sx}},${{sy}} Q${{Math.max(sx, dx) + bulge}},${{midY}} ${{dx}},${{dy}}`)
      .attr('fill', 'none')
      .attr('stroke-width', 1.2)
      .attr('marker-end', `url(#arrow-${{cls}})`);
  }});
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
  else if (activeTab === 'dep') depRedraw();
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

  const el = document.getElementById('dep-visible-stat');
  if (el) el.innerHTML = `Visible: <b>${{visNodes.length}}</b> nodes, <b>${{mergedEdges.length}}</b> edges`;

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

  // Edges: filter to force-node endpoints only
  // For edges involving expanded parents, remap to the parent's visible rep
  const validEdges = mergedEdges.filter(e => depVisNodeDataMap[e.source] && depVisNodeDataMap[e.target]);

  if (depSimulation) depSimulation.stop();

  // ── Links ──
  const linkSel = depLinkG.selectAll('line.dep-link')
    .data(validEdges, d => d.source + '|' + d.target + '|' + d.type);
  linkSel.exit().remove();
  const linkEnter = linkSel.enter().append('line')
    .attr('class', d => 'dep-link ' + (depEdgeClassMap[d.type] || 'dep-link-default'))
    .attr('marker-end', d => `url(#arrow-${{depEdgeClassMap[d.type] ? d.type : 'default'}})`)
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

  nodeEnter.append('circle');
  nodeEnter.append('text');

  const nodeAll = nodeEnter.merge(nodeSel);
  nodeAll.attr('class', d => 'dep-node' + (d.isCollapsed ? ' dep-node-collapsed' : ''));

  nodeAll.select('circle')
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
    }});
}}

// ── Draw hulls for expanded groups ──

// ── Edge visibility based on selection ──
function depUpdateEdgeVisibility() {{
  depLinkG.selectAll('line.dep-link').each(function(d) {{
    if (!showEdges) {{
      d3.select(this).style('display', 'none');
      return;
    }}
    if (depSelectedNodes.size === 0) {{
      d3.select(this).style('display', null).style('opacity', 0.5);
      return;
    }}
    const sid = typeof d.source === 'object' ? d.source.id : d.source;
    const tid = typeof d.target === 'object' ? d.target.id : d.target;
    if (depSelectedNodes.has(sid) || depSelectedNodes.has(tid)) {{
      d3.select(this).style('display', null).style('opacity', 0.9);
    }} else {{
      d3.select(this).style('display', 'none');
    }}
  }});
}}

// ── Node highlight based on selection ──
function depUpdateNodeHighlight() {{
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
    }} else {{
      el.select('circle').style('stroke', null).style('stroke-width', null);
      el.style('opacity', 0.4);
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
  activeTab = tab;
  document.getElementById('tab-feat').classList.toggle('active', tab === 'feat');
  document.getElementById('tab-dep').classList.toggle('active', tab === 'dep');
  document.getElementById('tab-map').classList.toggle('active', tab === 'map');
  document.getElementById('stats-feat').style.display = tab === 'feat' ? 'flex' : 'none';
  document.getElementById('stats-dep').style.display = tab === 'dep' ? 'flex' : 'none';
  document.getElementById('stats-map').style.display = tab === 'map' ? 'flex' : 'none';
  document.getElementById('feat-controls').style.display = tab === 'feat' ? 'inline' : 'none';
  document.getElementById('dep-controls').style.display = tab === 'dep' ? 'inline' : 'none';
  document.getElementById('map-controls').style.display = tab === 'map' ? 'inline' : 'none';

  gFeat.style('display', tab === 'feat' ? null : 'none');
  gDep.style('display', tab === 'dep' ? null : 'none');
  gMap.style('display', tab === 'map' ? null : 'none');

  if (tab === 'feat') {{
    if (depSimulation) depSimulation.stop();
    svg.on('.zoom', null).call(zoomFeat);
  }} else if (tab === 'dep') {{
    if (!hasDep) {{
      gDep.selectAll('*').remove();
      gDep.append('foreignObject').attr('width', width).attr('height', height)
        .append('xhtml:div').attr('class', 'no-data').text('No dep graph data');
    }} else {{
      depInit();
      if (depSimulation) depSimulation.alpha(0.1).restart();
    }}
    svg.on('.zoom', null).call(zoomDep);
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
}}

// ── Search ──
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.toLowerCase().trim();

  if (activeTab === 'feat') {{
    if (q.length < 2) {{
      nodeLayer.selectAll('circle').attr('stroke', '#30363d').attr('stroke-width', 1.5);
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

// ── Init feat graph ──
svg.call(zoomFeat);
svg.call(zoomFeat.transform, d3.zoomIdentity.translate(margin.left, margin.top));
root.x0 = height / 2;
root.y0 = 0;
update(root);

setTimeout(() => {{
  const bounds = gFeat.node().getBBox();
  const fullWidth = bounds.width + margin.left + margin.right;
  const fullHeight = bounds.height + margin.top + margin.bottom;
  const scale = Math.min(width / fullWidth, height / fullHeight, 1) * 0.9;
  svg.call(zoomFeat.transform,
    d3.zoomIdentity
      .translate(width / 2 - bounds.x * scale - bounds.width * scale / 2,
                 height / 2 - bounds.y * scale - bounds.height * scale / 2)
      .scale(scale));
}}, 400);
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
    html_content = generate_html(data)

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
