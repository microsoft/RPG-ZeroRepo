from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import common.trajectory as trajectory_module  # noqa: E402
from common.activity_events import ActivityWriter, load_activity_events, record_activity  # noqa: E402


def _trajectory(tmp_path: Path, monkeypatch) -> trajectory_module.Trajectory:
    monkeypatch.setattr(trajectory_module, "TRAJECTORY_DIR", Path("trajectory"))
    monkeypatch.setattr(
        trajectory_module,
        "ACTIVITY_WRITER",
        ActivityWriter(tmp_path / "activity", workspace_id="ws_test"),
    )
    return trajectory_module.Trajectory("plan", tmp_path)


def test_trajectory_records_command_step_script_and_llm_tree(tmp_path: Path, monkeypatch) -> None:
    trajectory = _trajectory(tmp_path, monkeypatch)
    trajectory.start()
    step = trajectory.add_step("skeleton")
    trajectory.start_step(step.step_id)
    script = trajectory.record_script_start(step.step_id, "cmind script build_skeleton.py")
    trajectory.record_script_end(step.step_id, script, 0)
    interaction = trajectory.start_llm_interaction(step.step_id, "design", "secret prompt")
    trajectory.complete_llm_interaction(step.step_id, interaction, "secret response", success=True, duration_seconds=1.2)
    trajectory.complete_step(step.step_id)
    trajectory.complete()

    events = load_activity_events(tmp_path / "activity")
    assert len(events) == 8
    assert len({event["event_id"] for event in events}) == 8
    command = next(event for event in events if event["kind"] == "workflow" and event["event_type"] == "span_started")
    command_finished = next(event for event in events if event["kind"] == "workflow" and event["event_type"] == "span_finished")
    stage = next(event for event in events if event["kind"] == "workflow.stage" and event["event_type"] == "span_started")
    stage_finished = next(event for event in events if event["kind"] == "workflow.stage" and event["event_type"] == "span_finished")
    assert stage["parent_span_id"] == command["span_id"]
    assert command["started_at"].endswith("Z")
    assert command_finished["finished_at"].endswith("Z")
    assert command_finished["duration_ms"] >= 0
    assert stage_finished["duration_ms"] >= 0
    assert {event["kind"] for event in events} == {"workflow", "workflow.stage", "tool.script", "tool.llm"}
    assert "secret prompt" not in str(events)
    assert "secret response" not in str(events)


def test_trajectory_does_not_duplicate_activity_inside_measured_workflow(tmp_path: Path, monkeypatch) -> None:
    trajectory = _trajectory(tmp_path, monkeypatch)
    outer_writer = ActivityWriter(tmp_path / "outer", workspace_id="ws_test")
    with record_activity("workflow", "encode", writer=outer_writer):
        trajectory.start()
        step = trajectory.add_step("encode")
        trajectory.start_step(step.step_id)
        trajectory.complete_step(step.step_id)
        trajectory.complete()

    assert load_activity_events(tmp_path / "activity") == []


def test_trajectory_filenames_are_unique_within_same_second(tmp_path: Path, monkeypatch) -> None:
    first = _trajectory(tmp_path, monkeypatch)
    second = _trajectory(tmp_path, monkeypatch)
    assert first.trajectory_file != second.trajectory_file
    assert len(first.trajectory_file.stem.rsplit("_", 1)[-1]) == 32