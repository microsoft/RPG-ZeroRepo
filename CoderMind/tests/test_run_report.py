from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.run_report import write_command_report


def test_write_command_report_escapes_content_and_writes_sections(tmp_path: Path) -> None:
    report = write_command_report(
        "rpg/edit <script>alert(1)</script>",
        title="Title <unsafe>",
        status="ok <bad>",
        summary_cards=[
            {"label": "node", "value": "<script>alert(1)</script>"},
            {"label": "count", "value": 3},
        ],
        stages=[{"name": "locate <x>", "status": "done", "reason": "score > 1"}],
        rpg_nodes=[{"node_id": "feature<script>", "name": "Explain", "meta_path": "a.py"}],
        dep_nodes=[{"node_id": "a.py:f", "path": "a.py"}],
        artifacts={"plan": tmp_path / "plan.json"},
        verification={"pytest": "passed"},
        evidence={"raw": "<script>evil()</script>"},
        report_dir=tmp_path,
        timestamp="2026-06-30T12:34:56Z",
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


def test_write_command_report_limits_summary_cards(tmp_path: Path) -> None:
    report = write_command_report(
        "encode",
        summary_cards=[{"label": f"card-{i}", "value": i} for i in range(9)],
        report_dir=tmp_path,
        timestamp="fixed",
    )

    html = report.read_text(encoding="utf-8")
    assert html.count('class="card-label"') == 7
    visible_cards = html.split("<section><details>", 1)[0]
    assert "card-0" in visible_cards
    assert "card-6" in visible_cards
    assert "card-7" not in visible_cards
    assert "card-8" not in visible_cards


def test_write_command_report_preserves_same_timestamp_runs(tmp_path: Path) -> None:
    first = write_command_report("update_rpg", report_dir=tmp_path, timestamp="fixed")
    second = write_command_report("update_rpg", report_dir=tmp_path, timestamp="fixed")

    assert first != second
    assert first.name == "cmind_run_update_rpg_fixed.html"
    assert second.name == "cmind_run_update_rpg_fixed_2.html"
    assert first.exists()
    assert second.exists()


def test_write_command_report_does_not_invent_node_rows_from_counts(tmp_path: Path) -> None:
    report = write_command_report(
        "encode",
        evidence={"dep_nodes": 4, "rpg_nodes": 6},
        report_dir=tmp_path,
        timestamp="fixed",
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
        "encode",
        artifacts=[("rpg_json", available), {"missing_json": missing}],
        verification=[
            {"name": "message", "status": "ok", "message": "from message"},
            {"name": "reason", "status": "warn", "reason": "from reason"},
        ],
        report_dir=tmp_path,
        timestamp="fixed",
    )

    html = report.read_text(encoding="utf-8")
    assert "<td>rpg_json</td>" in html
    assert "<td>missing_json</td>" in html
    assert html.count("<td>available</td>") == 1
    assert html.count("<td>missing</td>") == 1
    assert "<td>from message</td>" in html
    assert "<td>from reason</td>" in html
