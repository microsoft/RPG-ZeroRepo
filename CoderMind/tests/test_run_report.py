from __future__ import annotations

import html as html_lib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.run_events import (
    ArtifactEvent,
    CodeDeltaEvent,
    CommandRun,
    DepGraphDeltaEvent,
    RPGDeltaEvent,
    RetrievalEvent,
    StepEvent,
    UserDecisionEvent,
    VerificationEvent,
)
import common.run_report as run_report
from common.run_report import write_command_report


def test_common_run_report_run_report_exposes_current_impact_renderers_only() -> None:
    assert hasattr(run_report, "_render_focused_graph")
    assert hasattr(run_report, "_inline_d3")
    assert not hasattr(run_report, "_render_semantic_code_impact_chain")
    assert not hasattr(run_report, "_render_focused_nodes_map")
    assert not hasattr(run_report, "_render_why_these_nodes")
    assert not hasattr(run_report, "_render_node_rows")
    assert not hasattr(run_report, "_render_focused_impact")
    assert not hasattr(run_report, "_render_focused_impact_group")


def test_common_run_report_write_command_report_escapes_content_and_writes_sections(tmp_path: Path) -> None:
    report = write_command_report(
        {
            "command": "rpg/edit <script>alert(1)</script>",
            "title": "Title <unsafe>",
            "status": "ok <bad>",
            "summary": [
                {"label": "node", "value": "<script>alert(1)</script>"},
                {"label": "count", "value": 3},
            ],
            "steps": [StepEvent(name="locate <x>", status="done", reason="score > 1")],
            "rpg_deltas": [RPGDeltaEvent(node_id="feature<script>", name="Explain", path="a.py")],
            "dep_graph_deltas": [DepGraphDeltaEvent(dep_node_id="a.py:f", path="a.py")],
            "code_deltas": [CodeDeltaEvent(file="a.py", change_type="modify", diff="+safe")],
            "focused_view": {
                "nodes_view": {
                    "summary": {"semantic_nodes": 1, "code_nodes": 0, "mappings": 0, "edges": 0, "warnings": 0},
                    "semantic_nodes": [{"node_id": "feature<script>", "link_id": "rpg-feature-script", "name": "Explain"}],
                    "code_nodes": [],
                    "mappings": [],
                    "edges": [],
                    "hidden_counts": {},
                    "warnings": [],
                }
            },
            "artifacts": [ArtifactEvent(label="plan", path=tmp_path / "plan.json")],
            "verification": [VerificationEvent(name="pytest", status="passed")],
            "user_decisions": [
                UserDecisionEvent(
                    decision="apply <unsafe>",
                    branch="rpg-edit/<branch>",
                    before_state={"head_branch": "<main>", "head_commit": "abc<script>"},
                    rollback_path="backup/<script>",
                    confirmed=True,
                    apply_status="success <ok>",
                    test_status="passed <ok>",
                )
            ],
            "evidence": {"raw": "<script>evil()</script>"},
            "timestamp": "2026-06-30T12:34:56Z",
        },
        report_dir=tmp_path,
    )

    assert report.parent == tmp_path
    assert report.name.startswith("cmind_run_rpg_edit_script_alert_1_script_")
    html = report.read_text(encoding="utf-8")
    section_order = [
        "Summary",
        "Focused graph",
        "What changed?",
        "Stage timeline",
        "Safety boundary",
        "Artifact links",
        "Evidence JSON",
    ]
    positions = [html.index(title) for title in section_order if title in html]
    assert positions == sorted(positions)
    assert "Summary" in html
    assert "Stage timeline" in html
    assert "Safety boundary" in html
    assert "Verification status" not in html
    assert "pytest" in html
    assert "Artifact links" in html
    assert "Evidence JSON" in html
    assert "Before state" in html
    assert "overflow-x:auto" in html
    assert "min-width:680px" in html
    assert "Confirmation" in html
    assert "Branch" in html
    assert "Apply status" in html
    assert "Test status" in html
    assert "Rollback path" in html
    assert "apply &lt;unsafe&gt;" in html
    assert "rpg-edit/&lt;branch&gt;" in html
    assert "abc&lt;script&gt;" in html
    assert "backup/&lt;script&gt;" in html
    assert "success &lt;ok&gt;" in html
    assert "passed &lt;ok&gt;" in html
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "backup/<script>" not in html


def test_common_run_report_write_command_report_renders_retrievals_code_deltas_and_focused_view(tmp_path: Path) -> None:
    long_diff = "diff --git a/a.py b/a.py\n" + "\n".join(
        f"+line {i} <script>alert({i})</script>" for i in range(40)
    )

    report = write_command_report(
        {
            "command": "rpg_edit",
            "retrievals": [
                RetrievalEvent(
                    query="a.py",
                    tool="RPG_EDIT_LOCATE_FILE",
                    hits=[{"node_id": "n1<script>", "reason": "score < 1"}],
                    reason="selected <hit>",
                ).to_dict()
            ],
            "code_deltas": [
                CodeDeltaEvent(file="a.py", change_type="modify", diff=long_diff).to_dict()
            ],
            "focused_view": {
                "summary": {
                    "primary_rpg_nodes": 2,
                    "primary_code_nodes": 1,
                    "mapped_code_relations": 1,
                    "missing_mappings": 1,
                    "edges": 2,
                    "warnings": 4,
                },
                "nodes_view": {
                    "summary": {
                        "semantic_nodes": 2,
                        "code_nodes": 1,
                        "mappings": 2,
                        "edges": 3,
                        "warnings": 3,
                    },
                    "semantic_nodes": [
                        {
                            "node_id": "n1<script>",
                            "link_id": "rpg-n1-script",
                            "name": "Node <unsafe>",
                            "symbol": "NodeSymbol <unsafe>",
                            "node_type": "feature<script>",
                            "breadcrumb": ["Root", "Feature <unsafe>"],
                            "breadcrumb_path": "Root / Feature <unsafe>",
                            "state": "mapped",
                            "mapping_status": "mapped",
                            "reason": "selected <reason>",
                            "mapped_code": [
                                {"node_id": "a.py:f<script>", "link_id": "code-a.py-f-script", "path": "a.py<script>", "symbol": "func <unsafe>", "type": "function<script>", "line_range": {"start": 10, "end": 12}},
                                {"node_id": "duplicate-node-id", "link_id": "duplicate-link-id", "path": "a.py<script>", "symbol": "func <unsafe>", "source": "duplicate-source"},
                            ],
                            "mapped_code_node_ids": ["a.py:f<script>"],
                            "changed_files": [{"path": "a.py", "diff_anchor": "diff-a.py"}],
                            "hidden_counts": {"callers": 3},
                        },
                        {
                            "node_id": "n2",
                            "link_id": "rpg-n2",
                            "name": "Missing <mapping>",
                            "breadcrumb_path": "Root / Missing <path>",
                            "state": "missing_mapping",
                            "mapping_status": "missing",
                            "warning_types": ["missing_mapping"],
                        },
                    ],
                    "code_nodes": [
                        {
                            "node_id": "a.py:f<script>",
                            "dep_node_id": "a.py:f<script>",
                            "link_id": "code-a.py-f-script",
                            "symbol": "func <unsafe>",
                            "path": "a.py",
                            "feature_path": "SHOULD NOT RENDER FROM CODE",
                            "breadcrumb_path": "SHOULD NOT RENDER FROM CODE BREADCRUMB",
                            "type": "function<script>",
                            "line_range": {"start": 10, "end": 12},
                            "state": "mapped",
                            "source": "locate<unsafe>",
                            "mapped_rpg_node_ids": ["n1<script>"],
                            "changed_files": [{"path": "a.py", "diff_anchor": "diff-a.py"}],
                        }
                    ],
                    "mappings": [
                        {
                            "rpg_node_id": "n1<script>",
                            "code_node_id": "a.py:f<script>",
                            "source_link_id": "rpg-n1-script",
                            "target_link_id": "code-a.py-f-script",
                            "status": "mapped",
                            "state": "mapped",
                            "path": "a.py<script>",
                            "source": "locate+impact",
                            "reason": "maps because <reason>",
                            "changed_files": [{"path": "a.py", "diff_anchor": "diff-a.py"}],
                        },
                        {"rpg_node_id": "n2", "source_link_id": "rpg-n2", "status": "missing", "state": "missing_mapping", "reason": "no dep <reason>"},
                    ],
                    "edges": [
                        {
                            "source_node_id": "caller<script>",
                            "target_node_id": "a.py:f<script>",
                            "source_link_id": "context-caller-script",
                            "target_link_id": "code-a.py-f-script",
                            "relation": "caller",
                            "direction": "upstream",
                            "path": "caller.py<script>",
                            "source": "impact",
                            "reason": "calls <reason>",
                        },
                        {
                            "source_node_id": "a.py:f<script>",
                            "target_node_id": "callee<script>",
                            "source_link_id": "code-a.py-f-script",
                            "target_link_id": "context-callee-script",
                            "relation": "callee",
                            "direction": "downstream",
                            "path": "callee.py<script>",
                            "source": "impact",
                            "reason": "reaches <reason>",
                        },
                        {
                            "source_node_id": "a.py:f<script>",
                            "target_node_id": "imported<script>",
                            "source_link_id": "code-a.py-f-script",
                            "target_link_id": "context-imported-script",
                            "relation": "imports",
                            "direction": "downstream",
                            "path": "imported.py<script>",
                            "source": "dep_graph",
                            "source_graph": "dep_graph",
                            "reason": "imports <reason>",
                        },
                    ],
                    "hidden_counts": {"callers": 3},
                    "warnings": [
                        {"type": "missing_mapping", "message": "Missing <mapping>", "node_id": "n2", "node_link_id": "rpg-n2"},
                        {"type": "missing_reason", "message": "Missing <reason>", "node_id": "n3"},
                        {"type": "stale_graph", "message": "Stale <graph>", "dep_node_id": "old.py:f"},
                    ],
                    "changed_files": [{"path": "a.py", "diff_anchor": "diff-a.py"}],
                    "hierarchy": {
                        "id": "focused-graph-root",
                        "name": "Focused graph",
                        "kind": "root",
                        "children": [
                            {"id": "rpg-n1-script", "name": "Node <unsafe>", "kind": "semantic"},
                            {"id": "rpg-background", "name": "Background feature", "kind": "feature", "feature_path": "Root / Background feature"},
                            {"id": "code-a.py-f-script", "name": "func <unsafe>", "kind": "code"},
                        ],
                    },
                    "default_focus": {
                        "node_link_ids": ["rpg-n1-script", "code-a.py-f-script"],
                        "focused_node_ids": ["rpg-n1-script", "code-a.py-f-script"],
                        "focused_tree_node_ids": ["rpg-n1-script", "code-a.py-f-script"],
                        "focused_code_link_ids": ["code-a.py-f-script"],
                        "expanded_node_ids": ["focused-graph-root"],
                        "default_expanded_node_ids": ["focused-graph-root"],
                        "focused_path_node_ids": ["focused-graph-root", "rpg-n1-script", "code-a.py-f-script"],
                        "relation_endpoint_link_ids": ["context-caller-script", "code-a.py-f-script", "context-callee-script", "context-imported-script"],
                        "context_node_ids": ["context-caller-script", "context-callee-script", "context-imported-script"],
                        "edge_depth": 1,
                        "show_edges": True,
                    },
                    "focused_graph": {
                        "schema": "cmind.focused_graph.v1",
                        "hierarchy": {"id": "focused-graph-root"},
                        "default_focus": {
                            "node_link_ids": ["rpg-n1-script", "code-a.py-f-script"],
                            "focused_node_ids": ["rpg-n1-script", "code-a.py-f-script"],
                            "focused_tree_node_ids": ["rpg-n1-script", "code-a.py-f-script"],
                            "focused_code_link_ids": ["code-a.py-f-script"],
                            "expanded_node_ids": ["focused-graph-root"],
                            "default_expanded_node_ids": ["focused-graph-root"],
                            "focused_path_node_ids": ["focused-graph-root", "rpg-n1-script", "code-a.py-f-script"],
                            "relation_endpoint_link_ids": ["context-caller-script", "code-a.py-f-script", "context-callee-script", "context-imported-script"],
                            "context_node_ids": ["context-caller-script", "context-callee-script", "context-imported-script"],
                            "edge_depth": 1,
                            "show_edges": True,
                        },
                    },
                },
                "primary_rpg_nodes": [
                    {
                        "node_id": "n1<script>",
                        "name": "Node <unsafe>",
                        "path": "feature/<unsafe>",
                        "status": "mapped",
                        "reason": "selected <reason>",
                        "hidden_counts": {"callers": 3},
                    },
                    {"node_id": "n2", "name": "Missing <mapping>", "status": "missing", "path": "missing/<path>"},
                ],
                "primary_code_nodes": [
                    {
                        "node_id": "a.py:f<script>",
                        "dep_node_id": "a.py:f<script>",
                        "name": "func <unsafe>",
                        "path": "a.py<script>",
                        "type": "function",
                        "status": "mapped",
                        "source": "locate<unsafe>",
                    }
                ],
                "mappings": [
                    {
                        "rpg_node_id": "n1<script>",
                        "code_node_id": "a.py:f<script>",
                        "status": "mapped",
                        "path": "a.py<script>",
                        "source": "locate+impact",
                        "reason": "maps because <reason>",
                        "changed_files": ["a.py"],
                    },
                    {"rpg_node_id": "n2", "status": "missing", "reason": "no dep <reason>"},
                ],
                "edges": [
                    {
                        "source_node_id": "caller<script>",
                        "target_node_id": "n1<script>",
                        "relation": "caller",
                        "direction": "upstream",
                        "path": "caller.py<script>",
                        "source": "impact",
                        "reason": "calls <reason>",
                    },
                    {
                        "source_node_id": "n1<script>",
                        "target_node_id": "callee<script>",
                        "relation": "callee",
                        "direction": "downstream",
                        "path": "callee.py<script>",
                        "source": "impact",
                        "reason": "reaches <reason>",
                    },
                    {
                        "source_node_id": "n1<script>",
                        "target_node_id": "imported<script>",
                        "relation": "imports",
                        "direction": "downstream",
                        "path": "imported.py<script>",
                        "source": "dep_graph",
                        "source_graph": "dep_graph",
                        "reason": "imports <reason>",
                    },
                ],
                "hidden_counts": {"callers": 3},
                "warnings": [
                    {"type": "missing_mapping", "message": "Missing <mapping>", "node_id": "n2"},
                    {"type": "missing_reason", "message": "Missing <reason>", "node_id": "n3"},
                    {"type": "stale_graph", "message": "Stale <graph>", "dep_node_id": "old.py:f"},
                ],
            },
            "timestamp": "fixed",
        },
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "Focused graph" in html
    assert "main { max-width:1440px;" in html
    assert "height=\"680\"" not in html
    assert "width=\"960\"" not in html
    assert '<div class="focused-graph-stage"><div class="focused-graph-toolbar">' in html
    assert '<div class="focused-graph-legend" aria-label="Focused graph legend">' in html
    assert "Focused nodes map" not in html
    assert "semantic-code impact chain" not in html
    assert html.count("<h2>Focused graph</h2>") == 1
    assert "Why these nodes?" not in html
    assert "Focused impact view" not in html
    assert "What changed?" in html
    assert "Focused graph evidence" not in html
    assert '<div class="focus-map">' not in html
    assert "One-hop context" not in html
    assert "data-focused-graph-json" in html
    assert "focused-graph-svg" in html
    assert "https://d3js.org v7.9.0" in html
    assert "<script src=" not in html
    assert "data-action=\"reset\"" in html
    assert "data-action=\"expand-all\"" not in html
    assert "data-action=\"depth-plus\"" not in html
    assert "data-action=\"depth-minus\"" not in html
    assert "data-action=\"edges\"" in html
    assert "data-action=\"search\"" in html
    assert "data-focused-graph-status" in html
    assert "data-focused-graph-detail" in html
    assert '<aside class="focused-graph-detail" data-focused-graph-detail aria-live="polite">' in html
    assert "Node details" in html
    assert "Select a node to inspect metadata." in html
    assert "Visible relation edges: 0/3" in html
    assert "Reset" in html
    assert "Expand all" not in html
    assert ">+1</button>" not in html
    assert ">-1</button>" not in html
    assert "Edges" in html
    assert "Search nodes" in html
    assert "d3.zoom()" in html
    assert "dblclick" in html
    assert "focused-graph-layer" in html
    assert "focused-graph-node.selected" in html
    assert ".focused-graph-node.non-focused circle { fill:#cbd5e1; }" in html
    assert ".focused-graph-node.non-focused text { fill:#94a3b8; }" in html
    assert ".focused-graph-node.selected circle, .focused-graph-node.active circle, .focused-graph-node.focused circle" in html
    assert "focused-graph-node.hidden" in html
    assert "focused-graph-link.active" in html
    assert "focused-graph-link.dimmed" in html
    assert ".focused-graph-stage { border:1px solid #334155; border-radius:12px; background:#0f172a; height:clamp(520px,72vh,820px);" in html
    assert ".focused-graph-svg { display:block; width:100%; height:100%;" in html
    assert ".focused-graph-toolbar { position:absolute; top:14px; left:14px; z-index:3; display:flex; flex-wrap:wrap; gap:8px; align-items:center; max-width:calc(100% - 28px); margin:0;" in html
    assert ".focused-graph-toolbar button, .focused-graph-toolbar input { border:1px solid #475569; border-radius:8px; background:#1e293b; color:#e5e7eb;" in html
    assert ".focused-graph-legend { position:absolute; left:14px; bottom:14px; z-index:3; display:flex; flex-wrap:wrap; gap:8px; max-width:calc(100% - 28px); margin:0;" in html
    assert ".focused-graph-detail { position:absolute; top:14px; right:14px;" in html
    assert "rootHierarchyId" in html
    assert "defaultExpandedIds" in html
    assert "visibleEndpoint" in html
    assert "updateStatus" in html
    assert "const svgSelection = d3.select(svg);" in html
    assert "function refreshGraphViewport()" in html
    assert "svg.getBoundingClientRect()" in html
    assert "stage.getBoundingClientRect()" in html
    assert "svgSelection.attr('viewBox', `0 0 ${width} ${height}`);" in html
    assert "window.ResizeObserver" in html
    assert "new ResizeObserver(scheduleResize)" in html
    assert "window.requestAnimationFrame" in html
    assert "Number(svg.getAttribute('width'))" not in html
    assert "Number(svg.getAttribute('height'))" not in html
    assert "const visible = showEdges ? currentRelationEdges.length : 0;" in html
    assert "const total = relationEdges.length;" in html
    assert "statusEl.textContent = `Visible relation edges: ${visible}/${total}`;" in html
    assert "const detailEl = section.querySelector('[data-focused-graph-detail]');" in html
    assert "function nodeDetailData(d)" in html
    assert "function renderFocusedGraphDetail(d)" in html
    assert "function searchText(value)" in html
    assert "return `${nodeLabel(d)} ${searchText(nodeDetailData(d))}`.toLowerCase();" in html
    assert "renderFocusedGraphDetail(null);" in html
    assert "renderFocusedGraphDetail(selectedId ? d : null);" in html
    assert "return text(detail.feature_name || detail.name || detail.node_id || detail.dep_node_id || detail.id || d.id || 'node');" in html
    assert "function mappedCodeLabel" not in html
    assert "return mapped && !base.includes(mapped)" not in html
    assert "function canonicalMappedCodeRefs" in html
    assert "if (!isCodeContextDetail(detail)) addValueRow(rows, 'Feature path', detail.breadcrumb_path || detail.feature_path);" in html
    assert "item.source" not in html
    assert "SHOULD NOT RENDER FROM CODE" in html
    assert "expandAll" not in html
    assert "expandDepth" not in html
    assert "collapseDepth" not in html
    assert "Expand hierarchy depth" not in html
    assert "Collapse hierarchy depth" not in html
    assert "d3.zoomIdentity" in html
    assert "Static focused graph fallback is available when D3 cannot run." in html
    for label in ("Tree link", "RPG semantic edge", "dep_graph dependency edge", "invokes", "imports", "inherits", "references"):
        assert label in html
    for klass in ("legend-tree-link", "legend-invokes-edge", "legend-imports-edge", "legend-inherits-edge", "legend-references-edge"):
        assert klass in html
    assert "legend-caller-edge" not in html
    assert "legend-callee-edge" not in html
    assert "semantic_nodes" in html
    assert "code_nodes" in html
    assert "mappings" in html
    assert "hierarchy" in html
    assert "default_focus" in html
    assert "focused_graph" in html
    assert "caller.py&lt;script&gt;" in html
    assert "Root / Feature &lt;unsafe&gt;" in html
    assert "NodeSymbol &lt;unsafe&gt;" in html
    assert "function&lt;script&gt;" in html
    assert '"kind": "context"' in html
    assert '"relation": "caller"' in html
    assert '"href": "#diff-a.py"' in html
    assert "Lines: 10-12" not in html
    assert "Mapped code:" not in html
    assert "id=\"diff-a.py\"" in html
    assert html.count("<summary>Inspector JSON</summary>") == 1
    assert "mapped" in html
    assert "missing" in html
    assert "n1&lt;script&gt;" in html
    assert "Node &lt;unsafe&gt;" in html
    assert "a.py:f&lt;script&gt;" in html
    assert "a.py&lt;script&gt;" in html
    assert "maps because &lt;reason&gt;" in html
    assert "<details><summary>Warnings</summary>" in html
    assert "<details open><summary>Warnings</summary>" not in html
    assert "Missing &lt;mapping&gt;" in html
    assert "missing_mapping" in html
    assert "missing_reason" in html
    assert "too_many_neighbors" not in html
    assert "stale_graph" in html
    assert "Hidden 3 additional caller neighbors." in html
    assert '"edges": 1' not in html
    assert html.count("+line 0 &lt;script&gt;alert(0)&lt;/script&gt;") == 1
    assert "+line 0 <script>alert(0)</script>" not in html
    assert "Node <unsafe>" not in html
    assert "a.py<script>" not in html
    assert "maps because <reason>" not in html
    assert "<details><summary>View diff</summary>" in html
    assert "<details open" not in html
    graph_json = html.split("data-focused-graph-json>", 1)[1].split("</script>", 1)[0]
    graph_payload = json.loads(html_lib.unescape(graph_json))
    assert len(graph_payload["edges"]) == 3
    assert len(graph_payload["relation_edges"]) == 3
    assert graph_payload["summary"]["edges"] == 3
    assert graph_payload["summary"]["relation_edges"] == 3
    assert graph_payload["summary"]["context_edges"] == 3
    assert any(node.get("id") == "rpg-background" for node in graph_payload["hierarchy"]["children"])
    relation_links = [link for link in graph_payload["links"] if link.get("relation") != "contains"]
    assert len(relation_links) == 3
    nodes_by_id = {node["id"]: node for node in graph_payload["nodes"]}
    assert nodes_by_id["rpg-n1-script"]["symbol"] == "NodeSymbol <unsafe>"
    assert nodes_by_id["rpg-n1-script"]["reason"] == "selected <reason>"
    assert len(nodes_by_id["rpg-n1-script"]["mapped_code"]) == 2
    canonical_refs = {
        (ref.get("path"), ref.get("symbol"))
        for ref in nodes_by_id["rpg-n1-script"]["mapped_code"]
    }
    assert canonical_refs == {("a.py<script>", "func <unsafe>")}
    assert nodes_by_id["rpg-n1-script"]["changed_files"] == [{"path": "a.py", "diff_anchor": "diff-a.py"}]
    assert nodes_by_id["rpg-n1-script"]["diff"] == {"path": "a.py", "href": "#diff-a.py"}
    assert nodes_by_id["code-a.py-f-script"]["symbol"] == "func <unsafe>"
    assert nodes_by_id["code-a.py-f-script"]["line_range"] == {"start": 10, "end": 12}
    assert nodes_by_id["code-a.py-f-script"]["source"] == "locate<unsafe>"
    assert nodes_by_id["context-caller-script"]["relation"] == "caller"
    assert nodes_by_id["context-caller-script"]["direction"] == "upstream"
    assert nodes_by_id["context-caller-script"]["source"] == "impact"
    assert nodes_by_id["context-caller-script"]["reason"] == "calls <reason>"
    assert nodes_by_id["context-imported-script"]["relation"] == "imports"
    assert nodes_by_id["context-imported-script"]["source_graph"] == "dep_graph"
    assert any(node["id"] == "context-callee-script" for node in graph_payload["nodes"])
    assert any(node["id"] == "context-imported-script" for node in graph_payload["nodes"])
    assert any("context-callee-script" in edge["target_candidates"] for edge in graph_payload["relation_edges"])
    assert any(edge["relation"] == "imports" and "context-imported-script" in edge["target_candidates"] for edge in graph_payload["relation_edges"])
    assert any("rpg-n1-script" in edge["source_candidates"] for edge in graph_payload["relation_edges"])
    assert any("code-a.py-f-script" in edge["source_candidates"] for edge in graph_payload["relation_edges"])
    default_focus = graph_payload["default_focus"]
    assert "focused-graph-root" in default_focus["default_expanded_node_ids"]
    assert "rpg-n1-script" in default_focus["focused_path_node_ids"]
    assert "context-callee-script" not in default_focus["default_expanded_node_ids"]
    assert "const isDefaultFocused = d =>" in html
    assert "${isDefaultFocused(d) ? ' focused' : ' non-focused'}" in html
    assert "focusedNodeIds.has(id) || focusedCodeLinkIds.has(id)" in html
    assert "focusedNodeIds.has(d.id) || focusedCodeLinkIds.has(d.id)" not in html
    assert "rpg-background" not in default_focus["focused_tree_node_ids"]
    assert "context-callee-script" not in default_focus["focused_tree_node_ids"]
    assert graph_payload["hidden_counts"] == {"callers": 3}
    inspector_json = html.split("<summary>Inspector JSON</summary><pre>", 1)[1].split("</pre>", 1)[0]
    assert "focused_graph" in inspector_json
    assert "nodes_view" in inspector_json
    assert "semantic_nodes" in inspector_json
    assert "hierarchy" in inspector_json
    assert "default_focus" in inspector_json
    assert "primary_rpg_nodes" not in inspector_json
    assert "primary_code_nodes" not in inspector_json
    assert "caps" not in inspector_json
    assert long_diff not in inspector_json
    evidence_json = html.split("<summary>Evidence JSON</summary><pre>", 1)[1].split("</pre>", 1)[0]
    assert "code_deltas" not in evidence_json
    assert "focused_view" not in evidence_json
    assert "nodes_view" not in evidence_json
    assert "focused_impact" not in evidence_json
    assert "focused_graph" not in evidence_json
    assert long_diff not in evidence_json
    assert "Focused impact summary" not in html
    assert "Graph artifact" not in html
    assert "Inspector metadata" not in html


def test_common_run_report_write_command_report_renders_static_fallback_without_d3(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_report, "_D3_ASSET", tmp_path / "missing-d3.v7.min.js")

    report = write_command_report(
        {
            "command": "rpg_edit",
            "focused_view": {
                "nodes_view": {
                    "summary": {"semantic_nodes": 1, "code_nodes": 0, "mappings": 0, "edges": 0, "warnings": 0},
                    "semantic_nodes": [{"node_id": "n1", "link_id": "rpg-n1", "name": "Node"}],
                    "code_nodes": [],
                    "mappings": [],
                    "edges": [],
                    "hidden_counts": {},
                    "warnings": [],
                }
            },
            "timestamp": "fixed",
        },
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "Focused graph" in html
    assert "Local D3 asset missing; showing the static fallback." in html
    assert "https://d3js.org" not in html
    assert "Static focused graph fallback is available when D3 cannot run." in html


def test_common_run_report_write_command_report_limits_summary_cards(tmp_path: Path) -> None:
    report = write_command_report(
        CommandRun(
            command="encode",
            summary=[{"label": f"card-{i}", "value": i} for i in range(9)],
            timestamp="fixed",
        ),
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert html.count('class="card-label"') == 7
    visible_cards = html.split("<section><details>", 1)[0]
    assert "card-0" in visible_cards
    assert "card-6" in visible_cards
    assert "card-7" not in visible_cards
    assert "card-8" not in visible_cards


def test_common_run_report_write_command_report_preserves_same_timestamp_runs(tmp_path: Path) -> None:
    first = write_command_report(CommandRun("update_rpg", timestamp="fixed"), report_dir=tmp_path)
    second = write_command_report(CommandRun("update_rpg", timestamp="fixed"), report_dir=tmp_path)

    assert first != second
    assert first.name == "cmind_run_update_rpg_fixed.html"
    assert second.name == "cmind_run_update_rpg_fixed_2.html"
    assert first.exists()
    assert second.exists()


def test_common_run_report_write_command_report_does_not_invent_node_rows_from_counts(tmp_path: Path) -> None:
    report = write_command_report(
        CommandRun(
            command="encode",
            evidence={"dep_nodes": 4, "rpg_nodes": 6},
            timestamp="fixed",
        ),
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "Why these nodes?" not in html
    assert "Focused impact view" not in html
    assert "semantic-code impact chain" not in html
    assert '"dep_nodes": 4' in html
    assert '<td><code>4</code></td>' not in html


def test_common_run_report_write_command_report_infers_artifact_status_and_preserves_verification_detail(tmp_path: Path) -> None:
    available = tmp_path / "available.json"
    available.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.json"

    report = write_command_report(
        CommandRun(
            command="encode",
            artifacts=[
                ArtifactEvent(label="rpg_json", path=available),
                ArtifactEvent(label="missing_json", path=missing),
            ],
            verification=[
                VerificationEvent(name="message", status="ok", detail="from message"),
                VerificationEvent(name="reason", status="warn", detail="from reason"),
            ],
            timestamp="fixed",
        ),
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "<td>rpg_json</td>" in html
    assert "<td>missing_json</td>" in html
    assert html.count("<td>available</td>") == 1
    assert html.count("<td>missing</td>") == 1
    assert "Verification status" not in html
    assert "Stage timeline" in html
    assert "<strong>message</strong>" in html
    assert "<strong>reason</strong>" in html
    assert "from message" in html
    assert "from reason" in html
    assert "<td>from message</td>" not in html
    assert "<td>from reason</td>" not in html


def test_common_run_report_all_event_types_serialize_with_optional_fields(tmp_path: Path) -> None:
    available = tmp_path / "artifact.txt"
    available.write_text("ok", encoding="utf-8")

    events = [
        StepEvent(),
        RetrievalEvent(query="grep", tool="grep", hits=[{"path": "a.py"}], reason="matched"),
        RPGDeltaEvent(node_id="feature", name="Feature", type="function", path="a.py", change="modified", score=1.0),
        DepGraphDeltaEvent(dep_node_id="a.py:f", path="a.py", source_feature="feature", change="modified"),
        CodeDeltaEvent(file="a.py", change_type="modify", before="old", after="new", diff="@@"),
        VerificationEvent(),
        UserDecisionEvent(
            decision="apply",
            branch="rpg-edit/x",
            before_state={"clean": True},
            rollback_path="backup",
            confirmed=True,
            apply_status="success",
            test_status="passed",
        ),
        ArtifactEvent(label="artifact", path=available),
    ]

    for event in events:
        assert isinstance(event.to_dict(), dict)

    run = CommandRun(
        command="events",
        retrievals=[events[1]],
        rpg_deltas=[events[2]],
        dep_graph_deltas=[events[3]],
        code_deltas=[events[4]],
        user_decisions=[events[6]],
        artifacts=[events[7]],
    ).to_dict()

    assert run["retrievals"][0]["tool"] == "grep"
    assert run["code_deltas"][0]["file"] == "a.py"
    assert run["user_decisions"][0]["confirmed"] is True
    assert run["user_decisions"][0]["apply_status"] == "success"
    assert run["user_decisions"][0]["test_status"] == "passed"
    assert run["artifacts"][0]["status"] == "available"


def test_common_run_report_write_command_report_accepts_command_run_mapping(tmp_path: Path) -> None:
    run = CommandRun(
        command="mapping",
        summary=[{"label": "safe", "value": "<ok>"}],
        timestamp="fixed",
    ).to_dict()

    report = write_command_report(run, report_dir=tmp_path)

    html = report.read_text(encoding="utf-8")
    assert "&lt;ok&gt;" in html
