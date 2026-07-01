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
            evidence={"raw": "<script>evil()</script>"},
            timestamp="2026-06-30T12:34:56Z",
        ),
        report_dir=tmp_path,
    )

    assert report.parent == tmp_path
    assert report.name.startswith("cmind_run_rpg_edit_script_alert_1_script_")
    html = report.read_text(encoding="utf-8")
    assert "Summary" in html
    assert "Stage timeline" in html
    assert "Artifact links" in html
    assert "Evidence JSON" in html
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_write_command_report_renders_retrievals_code_deltas_and_focused_graph(tmp_path: Path) -> None:
    graph = tmp_path / "focused.html"
    graph.write_text("graph", encoding="utf-8")
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
            "focused_graph": {
                "path": graph,
                "status": "available",
                "selected_rpg_nodes": ["n1"],
                "selected_dep_nodes": ["a.py:f"],
                "rpg_node_count": 2,
                "dep_node_count": 1,
            },
            "timestamp": "fixed",
        },
        report_dir=tmp_path,
    )

    html = report.read_text(encoding="utf-8")
    assert "Retrieval evidence" in html
    assert "Code deltas" in html
    assert "Focused graph evidence" in html
    assert "RPG_EDIT_LOCATE_FILE" in html
    assert "n1&lt;script&gt;" in html
    assert "+line 0 &lt;script&gt;alert(0)&lt;/script&gt;" in html
    assert "+line 0 <script>alert(0)</script>" not in html
    assert "<details><summary>View diff</summary>" in html
    assert "<details open" not in html
    assert html.count("<h2>Focused graph evidence</h2>") == 1
    assert "Graph artifact" in html
    assert "Inspector metadata" in html


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
    assert html.count("No node evidence recorded.") == 2
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
        UserDecisionEvent(decision="apply", branch="rpg-edit/x", before_state={"clean": True}, rollback_path="backup", confirmed=True),
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
