from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_batch as run_batch_module  # noqa: E402
from common.activity_events import ActivityWriter, load_activity_events  # noqa: E402


def test_code_gen_batch_keeps_task_id_stable_but_execution_span_unique(tmp_path: Path, monkeypatch) -> None:
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")
    monkeypatch.setattr(run_batch_module, "ACTIVITY_WRITER", writer)
    monkeypatch.setattr(run_batch_module, "_run_batch_impl", lambda **kwargs: {
        "success": True,
        "type": "batch_complete",
        "batch_id": kwargs["retry"],
        "task_type": "implementation",
        "attempts_used": 1,
        "file_path": "src/demo.py",
    })

    first = run_batch_module.run_batch(retry="task-stable")
    second = run_batch_module.run_batch(retry="task-stable")

    assert first["batch_id"] == second["batch_id"] == "task-stable"
    events = load_activity_events(tmp_path / "activity")
    started = [event for event in events if event["event_type"] == "span_started"]
    finished = [event for event in events if event["event_type"] == "span_finished"]
    assert len(started) == len(finished) == 2
    assert len({event["span_id"] for event in started}) == 2
    assert {event["batch_id"] for event in finished} == {"task-stable"}
    assert {event["trigger"] for event in started} == {"retry"}


def test_code_gen_batch_records_false_result_as_failed(tmp_path: Path, monkeypatch) -> None:
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")
    monkeypatch.setattr(run_batch_module, "ACTIVITY_WRITER", writer)
    monkeypatch.setattr(run_batch_module, "_run_batch_impl", lambda **kwargs: {
        "success": False,
        "type": "batch_failed",
        "batch_id": kwargs["batch_id"],
        "failure_reason": "tests failed",
    })

    run_batch_module.run_batch(batch_id="task-failed")

    finished = load_activity_events(tmp_path / "activity")[-1]
    assert finished["status"] == "failed"
    assert finished["error"] == {"type": "CodegenBatchError", "message": "tests failed"}