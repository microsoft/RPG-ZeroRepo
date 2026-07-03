from __future__ import annotations

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
from common.run_report import write_command_report


def test_write_command_report_escapes_content_and_writes_sections(tmp_path: Path) -> None:
    report = write_command_report(
        CommandRun(
            command="rpg/edit <script>alert(1)</script>",
            title="Title <unsafe>",
            status="ok <bad>",
            summary=[
                {"label": "node", "value": "<script>alert(1)</script>"},
                {"label": "count", "value": 3},
            ],
            steps=[StepEvent(name="locate <x>", status="done", reason="score > 1")],
            rpg_deltas=[RPGDeltaEvent(node_id="feature<script>", name="Explain", path="a.py")],
            dep_graph_deltas=[DepGraphDeltaEvent(dep_node_id="a.py:f", path="a.py")],
            artifacts=[ArtifactEvent(label="plan", path=tmp_path / "plan.json")],
            verification=[VerificationEvent(name="pytest", status="passed")],
            user_decisions=[
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
            evidence={"raw": "<script>evil()</script>"},
            timestamp="2026-06-30T12:34:56Z",
        ),
        report_dir=tmp_path,
    )

    assert report.parent == tmp_path
    assert report.name.startswith("cmind_run_rpg_edit_script_alert_1_script_")
    html = report.read_text(encoding="utf-8")
    section_order = [
        "Summary",
        "Stage timeline",
        "Safety boundary",
        "Focused nodes map",
        "semantic-code impact chain",
        "What changed?",
        "Verification status",
        "Artifact links",
        "Evidence JSON",
    ]
    positions = [html.index(title) for title in section_order if title in html]
    assert positions == sorted(positions)
    assert "Summary" in html
    assert "Stage timeline" in html
    assert "Safety boundary" in html
    assert "Verification status" in html
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


def test_write_command_report_renders_retrievals_code_deltas_and_focused_view(tmp_path: Path) -> None:
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
                        "edges": 2,
                        "warnings": 4,
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
                    ],
                    "hidden_counts": {"callers": 3, "edges": 1, "relations": {"caller": 1}},
                    "warnings": [
                        {"type": "missing_mapping", "message": "Missing <mapping>", "node_id": "n2", "node_link_id": "rpg-n2"},
                        {"type": "missing_reason", "message": "Missing <reason>", "node_id": "n3"},
                        {"type": "too_many_neighbors", "message": "Too many <neighbors>", "hidden_counts": {"edges": 1}},
                        {"type": "stale_graph", "message": "Stale <graph>", "dep_node_id": "old.py:f"},
                    ],
                    "changed_files": [{"path": "a.py", "diff_anchor": "diff-a.py"}],
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
                ],
                "hidden_counts": {"callers": 3, "edges": 1, "relations": {"caller": 1}},
                "warnings": [
                    {"type": "missing_mapping", "message": "Missing <mapping>", "node_id": "n2"},
                    {"type": "missing_reason", "message": "Missing <reason>", "node_id": "n3"},
                    {"type": "too_many_neighbors", "message": "Too many <neighbors>", "hidden_counts": {"edges": 1}},
                    {"type": "stale_graph", "message": "Stale <graph>", "dep_node_id": "old.py:f"},
                ],
            },
            "timestamp": "fixed",
        },
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "Focused nodes map" in html
    assert "semantic-code impact chain" in html
    assert html.count("<h2>Focused nodes map</h2>") == 1
    assert html.count("<h2>semantic-code impact chain</h2>") == 1
    assert "Why these nodes?" not in html
    assert "Focused impact view" not in html
    assert "What changed?" in html
    assert "Feature group" in html
    assert "Semantic → code evidence" in html
    assert "focus-map" in html
    assert "One-hop context" in html
    assert "caller.py&lt;script&gt;" in html
    assert "Root / Feature &lt;unsafe&gt;" in html
    assert "NodeSymbol &lt;unsafe&gt;" in html
    assert "function&lt;script&gt;" in html
    assert "Lines: 10-12" in html
    assert "Mapped code:" in html
    assert "Mapped code: <span class=\"empty\">missing mapping</span>" in html
    assert "href=\"#diff-a.py\"" in html
    assert "id=\"diff-a.py\"" in html
    assert html.count("<summary>Inspector JSON</summary>") == 1
    assert "mapped" in html
    assert "missing" in html
    assert "n1&lt;script&gt;" in html
    assert "Node &lt;unsafe&gt;" in html
    assert "feature/&lt;unsafe&gt;" in html
    assert "a.py:f&lt;script&gt;" in html
    assert "a.py&lt;script&gt;" in html
    assert "maps because &lt;reason&gt;" in html
    assert "Missing &lt;mapping&gt;" in html
    assert "missing_mapping" in html
    assert "missing_reason" in html
    assert "too_many_neighbors" in html
    assert "stale_graph" in html
    assert "Hidden 4 additional caller neighbors." in html
    assert '"caller": 1' in html
    assert html.count("+line 0 &lt;script&gt;alert(0)&lt;/script&gt;") == 1
    assert "+line 0 <script>alert(0)</script>" not in html
    assert "Node <unsafe>" not in html
    assert "a.py<script>" not in html
    assert "maps because <reason>" not in html
    assert "<details><summary>View diff</summary>" in html
    assert "<details open" not in html
    inspector_json = html.split("<summary>Inspector JSON</summary><pre>", 1)[1].split("</pre>", 1)[0]
    assert "nodes_view" in inspector_json
    assert "semantic_nodes" in inspector_json
    assert "primary_rpg_nodes" not in inspector_json
    assert "primary_code_nodes" not in inspector_json
    assert long_diff not in inspector_json
    evidence_json = html.split("<summary>Evidence JSON</summary><pre>", 1)[1].split("</pre>", 1)[0]
    assert "code_deltas" not in evidence_json
    assert "focused_view" not in evidence_json
    assert "focused_impact" not in evidence_json
    assert "focused_graph" not in evidence_json
    assert long_diff not in evidence_json
    assert len(html) < 22000
    assert "Focused impact summary" not in html
    assert "Focused graph evidence" not in html
    assert "Graph artifact" not in html
    assert "Inspector metadata" not in html
    assert "focused_graph" not in html


def test_write_command_report_limits_summary_cards(tmp_path: Path) -> None:
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


def test_write_command_report_preserves_same_timestamp_runs(tmp_path: Path) -> None:
    first = write_command_report(CommandRun("update_rpg", timestamp="fixed"), report_dir=tmp_path)
    second = write_command_report(CommandRun("update_rpg", timestamp="fixed"), report_dir=tmp_path)

    assert first != second
    assert first.name == "cmind_run_update_rpg_fixed.html"
    assert second.name == "cmind_run_update_rpg_fixed_2.html"
    assert first.exists()
    assert second.exists()


def test_write_command_report_does_not_invent_node_rows_from_counts(tmp_path: Path) -> None:
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


def test_write_command_report_infers_artifact_status_and_preserves_verification_detail(tmp_path: Path) -> None:
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
    assert "<td>from message</td>" in html
    assert "<td>from reason</td>" in html


def test_all_event_types_serialize_with_optional_fields(tmp_path: Path) -> None:
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


def test_write_command_report_accepts_command_run_mapping(tmp_path: Path) -> None:
    run = CommandRun(
        command="mapping",
        summary=[{"label": "safe", "value": "<ok>"}],
        timestamp="fixed",
    ).to_dict()

    report = write_command_report(run, report_dir=tmp_path)

    html = report.read_text(encoding="utf-8")
    assert "&lt;ok&gt;" in html
