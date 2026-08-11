from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.activity_events import (  # noqa: E402
    ActivityWriter,
    record_activity,
    record_completed_activity,
    script_activity_mode,
)
from common.activity_history import collect_run_history  # noqa: E402


def test_smoke_advisory_activity_mode_is_explicit() -> None:
    assert script_activity_mode("smoke_test.py", ["--json", "--advisory"]) == "advisory"
    assert script_activity_mode("smoke_test.py", ["--json"]) is None


def test_history_prefers_v2_and_builds_parent_child_tree(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    with record_activity("workflow", "plan", logical_key="decoder-plan", writer=writer) as run:
        run.note(run_id="run-1")
        with record_activity(
            "workflow.stage",
            "skeleton",
            logical_key="decoder-plan-skeleton",
            writer=writer,
        ):
            pass

    legacy = [{
        "run_id": "run-1",
        "command": "plan",
        "status": "success",
        "display_status": "success",
        "started_at": "2026-08-05T00:00:00Z",
        "finished_at": "2026-08-05T00:00:01Z",
        "duration_s": 1,
        "stages": [],
    }]
    history = collect_run_history(tmp_path / "logs", legacy, [], [])

    assert history["summary"]["root_count"] == 1
    assert history["summary"]["legacy_count"] == 0
    assert history["roots"][0]["logical_key"] == "decoder-plan"
    assert history["roots"][0]["children"][0]["logical_key"] == "decoder-plan-skeleton"


def test_history_deduplicates_trajectory_with_independent_run_id(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    writer.append(
        "span_started",
        trace_id="trc_v2",
        span_id="spn_v2",
        parent_span_id=None,
        kind="workflow",
        name="feature_spec",
        logical_key="decoder-feature-spec",
        status="running",
        timestamp="2026-08-05T00:00:00.000Z",
    )
    writer.append(
        "span_finished",
        trace_id="trc_v2",
        span_id="spn_v2",
        parent_span_id=None,
        kind="workflow",
        name="feature_spec",
        logical_key="decoder-feature-spec",
        status="success",
        timestamp="2026-08-05T00:00:01.000Z",
    )
    legacy = [{
        "run_id": "trajectory:independent-id",
        "source": "trajectory",
        "command": "feature_spec",
        "status": "success",
        "display_status": "success",
        "started_at": "2026-08-05T00:00:00Z",
        "finished_at": "2026-08-05T00:00:01Z",
        "duration_s": 1,
        "stages": [],
    }]

    history = collect_run_history(tmp_path / "logs", legacy, [], [])

    assert history["summary"]["root_count"] == 1
    assert history["summary"]["legacy_count"] == 0
    assert history["roots"][0]["source"] == "activity_v2"


def test_history_adapts_legacy_runs_hooks_and_mcp(tmp_path: Path) -> None:
    runs = [{
        "run_id": "run-legacy",
        "command": "update_rpg",
        "trigger": "script",
        "status": "success",
        "display_status": "success",
        "started_at": "2026-08-05T00:00:00Z",
        "finished_at": "2026-08-05T00:00:01Z",
        "duration_s": 1,
        "stages": [{
            "stage_id": "stage-1",
            "name": "load_rpg",
            "status": "success",
            "started_at": "2026-08-05T00:00:00Z",
            "finished_at": "2026-08-05T00:00:00Z",
            "duration_s": 0.2,
            "attempt": 1,
            "sequence": 1,
        }],
    }]
    hooks = [{"ts": "2026-08-05T00:00:02Z", "hook": "sync", "duration_ms": 10, "run_id": "run-legacy"}]
    mcp = [{"ts": "2026-08-05T00:00:03Z", "tool": "search_rpg", "duration_ms": 5}]

    history = collect_run_history(tmp_path / "logs", runs, hooks, mcp)

    assert history["summary"]["root_count"] == 2
    assert history["summary"]["legacy_count"] == 3
    workflow = next(root for root in history["roots"] if root["kind"] == "workflow")
    assert workflow["logical_key"] == "encoder-update-rpg"
    assert {child["logical_key"] for child in workflow["children"]} == {
        "encoder-update-rpg-load-rpg", "encoder-hooks-sync",
    }
    assert {root["kind"] for root in history["roots"]} == {"workflow", "mcp.session"}
    mcp_root = next(root for root in history["roots"] if root["kind"] == "mcp.session")
    assert mcp_root["logical_key"] == "encoder-mcp"
    assert mcp_root["children"][0]["kind"] == "tool.mcp"


def test_history_deduplicates_call_ids_dual_written_to_v2_and_legacy(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    record_completed_activity(
        "tool.mcp",
        "search_rpg",
        fields={"call_id": "mcp_unique", "tool": "search_rpg"},
        writer=writer,
    )
    mcp = [{"call_id": "mcp_unique", "ts": "2026-08-05T00:00:03Z", "tool": "search_rpg"}]

    history = collect_run_history(tmp_path / "logs", [], [], mcp)

    assert history["summary"]["root_count"] == 1
    assert history["summary"]["legacy_count"] == 0
    assert history["roots"][0]["source"] == "activity_grouping"
    assert history["roots"][0]["children"][0]["source"] == "activity_v2"
    assert history["roots"][0]["children"][0]["details"]["call_id"] == "mcp_unique"


def test_history_groups_mcp_calls_by_server_session(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    for trace_id, tool in (("trc_one", "list_rpg_tree"), ("trc_two", "search_rpg")):
        record_completed_activity(
            "tool.mcp", tool, logical_key=f"mcp-{tool}", trace_id=trace_id,
            fields={"server_session_id": "mcp_session_shared", "tool": tool},
            writer=writer,
        )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["summary"]["root_count"] == 1
    root = history["roots"][0]
    assert root["name"] == "MCP session"
    assert root["metrics"]["calls"] == 2
    assert root["details"]["server_session_id"] == "mcp_session_shared"


def test_history_rolls_async_hook_child_into_parent(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    with record_activity(
        "hook.workflow", "post-commit", logical_key="encoder-hooks", writer=writer,
    ) as hook:
        hook.note(hook_type="post-commit", git_sha="abc1234")
    child = "spn_async_child"
    common = {
        "trace_id": hook.trace_id, "span_id": child, "parent_span_id": hook.span_id,
        "kind": "hook.operation", "name": "update-rpg",
        "logical_key": "encoder-hooks-update-rpg",
    }
    writer.append(
        "span_started", **common, status="running",
        timestamp="2099-01-01T00:00:00.000Z",
        fields={"started_at": "2099-01-01T00:00:00.000Z"},
    )
    writer.append(
        "span_finished", **common, status="success",
        timestamp="2099-01-01T00:00:00.100Z",
        fields={
            "started_at": "2099-01-01T00:00:00.000Z",
            "finished_at": "2099-01-01T00:00:00.100Z",
            "duration_ms": 100,
        },
    )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    root = history["roots"][0]
    assert root["kind"] == "hook.workflow"
    assert root["details"]["dispatch_finished_at"]
    assert root["metrics"]["operations"] == 1
    assert root["quality"] == "derived"


def test_history_groups_rpg_edit_phases_and_collapses_script_wrapper(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    for phase in ("validate", "locate", "impact"):
        with record_activity(
            "command.script",
            f"rpg_edit/{phase}.py",
            logical_key=f"decoder-rpg-edit-{phase}",
            trace_id="trc_rpg_edit",
            writer=writer,
        ):
            pass
    with record_activity("command.script", "plan.py", logical_key="script-plan", writer=writer):
        with record_activity("workflow", "plan", logical_key="decoder-plan", writer=writer):
            pass
        record_completed_activity(
            "artifact.write", "tasks.json", logical_key="artifact-data-tasks-json",
            writer=writer,
        )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert {root["logical_key"] for root in history["roots"]} == {"decoder-rpg-edit", "decoder-plan"}
    rpg_edit = next(root for root in history["roots"] if root["logical_key"] == "decoder-rpg-edit")
    assert [child["logical_key"] for child in rpg_edit["children"]] == [
        "decoder-rpg-edit-validate", "decoder-rpg-edit-locate", "decoder-rpg-edit-impact",
    ]
    plan = next(root for root in history["roots"] if root["logical_key"] == "decoder-plan")
    assert plan["children"][0]["kind"] == "artifact.write"


def test_history_attaches_summary_script_as_workflow_evidence(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    with record_activity(
        "workflow", "build_skeleton", logical_key="decoder-build-skeleton", writer=writer,
    ):
        pass
    with record_activity(
        "command.script", "summary_skeleton.py", logical_key="script-summary-skeleton", writer=writer,
    ):
        pass

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["summary"]["root_count"] == 1
    root = history["roots"][0]
    assert root["logical_key"] == "decoder-build-skeleton"
    helper = next(child for child in root["children"] if child["logical_key"] == "script-summary-skeleton")
    assert helper["details"]["grouped_as"] == "workflow_evidence"


def test_history_attaches_orphan_smoke_as_direct_rpg_edit_child(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    phase_common = {
        "trace_id": "trc_rpg_edit",
        "parent_span_id": None,
        "kind": "command.script",
    }
    phases = (
        ("spn_validate", "validate", "2026-08-05T00:00:00.000Z", "2026-08-05T00:00:01.000Z"),
        ("spn_review", "review", "2026-08-05T00:00:10.000Z", "2026-08-05T00:00:20.000Z"),
    )
    for span_id, phase, started_at, finished_at in phases:
        writer.append(
            "span_started", span_id=span_id, name=f"rpg_edit/{phase}.py",
            logical_key=f"decoder-rpg-edit-{phase}", status="running",
            timestamp=started_at, **phase_common,
        )
        writer.append(
            "span_finished", span_id=span_id, name=f"rpg_edit/{phase}.py",
            logical_key=f"decoder-rpg-edit-{phase}", status="success",
            timestamp=finished_at,
            fields={"started_at": started_at, "finished_at": finished_at, "duration_ms": 1000},
            **phase_common,
        )
    writer.append(
        "span_started", trace_id="trc_lost", span_id="spn_smoke",
        parent_span_id=None, kind="command.script", name="smoke_test.py",
        logical_key="script-smoke-test", status="running",
        timestamp="2026-08-05T00:00:05.000Z",
    )
    writer.append(
        "span_finished", trace_id="trc_lost", span_id="spn_smoke",
        parent_span_id=None, kind="command.script", name="smoke_test.py",
        logical_key="script-smoke-test", status="advisory",
        timestamp="2026-08-05T00:00:06.000Z",
        fields={
            "started_at": "2026-08-05T00:00:05.000Z",
            "finished_at": "2026-08-05T00:00:06.000Z",
            "duration_ms": 1000,
            "mode": "advisory",
        },
    )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["summary"]["root_count"] == 1
    rpg_edit = history["roots"][0]
    smoke = next(child for child in rpg_edit["children"] if child["logical_key"] == "script-smoke-test")
    assert smoke["parent_span_id"] == rpg_edit["span_id"]
    assert smoke["details"]["grouped_as"] == "rpg_edit_check"
    assert smoke["details"]["blocking"] is False
    assert smoke["status"] == "advisory"


@pytest.mark.parametrize(
    ("workflow_name", "workflow_key", "script_name", "script_key"),
    [
        ("plan", "decoder-plan", "plan.py", "script-plan"),
        (
            "feature_construct", "decoder-feature-construct",
            "feature_construct.py", "script-feature-construct",
        ),
    ],
)
def test_history_attaches_merged_stage_check_as_workflow_evidence(
    tmp_path: Path,
    workflow_name: str,
    workflow_key: str,
    script_name: str,
    script_key: str,
) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    with record_activity(
        "workflow", workflow_name, logical_key=workflow_key, writer=writer,
    ):
        pass
    with record_activity(
        "command.script", script_name, logical_key=script_key, writer=writer,
    ):
        pass

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["summary"]["root_count"] == 1
    root = history["roots"][0]
    assert root["logical_key"] == workflow_key
    helper = next(child for child in root["children"] if child["logical_key"] == script_key)
    assert helper["details"]["grouped_as"] == "workflow_evidence"


def test_history_marks_stale_unfinished_activity_interrupted(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    writer.append(
        "span_started",
        trace_id="trc_stale",
        span_id="spn_stale",
        parent_span_id=None,
        kind="workflow",
        name="stale",
        logical_key="decoder-stale",
        status="running",
        timestamp="2020-01-01T00:00:00.000Z",
    )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["roots"][0]["status"] == "interrupted"
    assert history["roots"][0]["quality"] == "derived"


def test_history_marks_failed_attempts_recovered_by_later_success(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    attempts = (
        ("spn_old", "trc_old", "2020-01-01T00:00:00.000Z", None),
        ("spn_new", "trc_new", "2020-01-01T01:00:00.000Z", "2020-01-01T01:01:00.000Z"),
    )
    for span_id, trace_id, started_at, finished_at in attempts:
        writer.append(
            "span_started", trace_id=trace_id, span_id=span_id,
            parent_span_id=None, kind="workflow", name="plan",
            logical_key="decoder-plan", status="running", timestamp=started_at,
        )
        child_id = f"{span_id}_interfaces"
        writer.append(
            "span_started", trace_id=trace_id, span_id=child_id,
            parent_span_id=span_id, kind="workflow", name="design_interfaces",
            logical_key="decoder-design-interfaces", status="running",
            timestamp=started_at,
        )
        if finished_at:
            writer.append(
                "span_finished", trace_id=trace_id, span_id=child_id,
                parent_span_id=span_id, kind="workflow", name="design_interfaces",
                logical_key="decoder-design-interfaces", status="success",
                timestamp=finished_at,
            )
            writer.append(
                "span_finished", trace_id=trace_id, span_id=span_id,
                parent_span_id=None, kind="workflow", name="plan",
                logical_key="decoder-plan", status="success", timestamp=finished_at,
            )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    newer, older = history["roots"]
    assert older["status"] == "interrupted"
    assert older["recovery"]["status"] == "recovered"
    assert older["recovery"]["by_span_id"] == newer["span_id"]
    assert older["children"][0]["recovery"]["by_span_id"] == newer["children"][0]["span_id"]
    assert newer["recovered_attempts"] == [{
        "span_id": older["span_id"],
        "status": "interrupted",
        "started_at": "2020-01-01T00:00:00.000Z",
    }]


def test_history_does_not_recover_unrelated_failure(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    writer.append(
        "span_started", trace_id="trc_failed", span_id="spn_failed",
        parent_span_id=None, kind="workflow", name="plan",
        logical_key="decoder-plan", status="running",
        timestamp="2020-01-01T00:00:00.000Z",
    )
    writer.append(
        "span_finished", trace_id="trc_failed", span_id="spn_failed",
        parent_span_id=None, kind="workflow", name="plan",
        logical_key="decoder-plan", status="failed",
        timestamp="2020-01-01T00:01:00.000Z",
    )
    writer.append(
        "span_started", trace_id="trc_other", span_id="spn_other",
        parent_span_id=None, kind="workflow", name="feature_construct",
        logical_key="decoder-feature-construct", status="running",
        timestamp="2020-01-01T01:00:00.000Z",
    )
    writer.append(
        "span_finished", trace_id="trc_other", span_id="spn_other",
        parent_span_id=None, kind="workflow", name="feature_construct",
        logical_key="decoder-feature-construct", status="success",
        timestamp="2020-01-01T01:01:00.000Z",
    )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    failed = next(root for root in history["roots"] if root["span_id"] == "spn_failed")
    assert "recovery" not in failed


def test_history_hides_snapshot_maintenance_spans(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "logs/activity", workspace_id="ws_test")
    record_completed_activity(
        "report.snapshot", "dashboard snapshot",
        logical_key="dashboard-snapshot-frozen", writer=writer,
    )

    history = collect_run_history(tmp_path / "logs", [], [], [])

    assert history["roots"] == []
    assert history["summary"]["exact_count"] == 0