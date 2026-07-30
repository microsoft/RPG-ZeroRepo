from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import common.run_events as run_events


@pytest.fixture
def events_file(tmp_path, monkeypatch):
    path = tmp_path / "run_events.jsonl"
    monkeypatch.setattr(run_events, "EVENTS_FILE", path)
    return path


def test_record_stage_writes_started_and_finished(events_file):
    run_id = run_events.new_run_id("encode")
    with run_events.record_stage(run_id, "encode", "parse_rpg", phase="encoder") as stage:
        stage.note(nodes=90)

    started, finished = run_events.load_events(events_file)
    assert [started["event_type"], finished["event_type"]] == ["stage_started", "stage_finished"]
    assert started["schema_version"] == run_events.SCHEMA_VERSION
    assert "v" not in started
    assert started["stage_id"] == finished["stage_id"]
    assert started["status"] == "running"
    assert finished["status"] == "success"
    assert started["started_at"].endswith("Z")
    assert finished["finished_at"].endswith("Z")
    assert finished["duration_s"] >= 0
    assert finished["metrics"] == {"nodes": 90}
    assert finished["error"] is None
    assert "tokens" not in finished


def test_record_stage_captures_failure_and_reraises(events_file):
    run_id = run_events.new_run_id("plan")
    with pytest.raises(ValueError):
        with run_events.record_stage(run_id, "plan", "build_skeleton", phase="P2"):
            raise ValueError("boom")

    started, finished = run_events.load_events(events_file)
    assert started["status"] == "running"
    assert finished["status"] == "failed"
    assert finished["error"] == {"type": "ValueError", "message": "boom"}


def test_run_lifecycle_wraps_stage_events(events_file):
    with run_events.record_run("encode", trigger="cli") as run:
        with run_events.record_stage(run.run_id, "encode", "save_rpg", phase="encoder"):
            pass

    events = run_events.load_events(events_file)
    assert [event["event_type"] for event in events] == [
        "run_started",
        "stage_started",
        "stage_finished",
        "run_finished",
    ]
    assert {event["run_id"] for event in events} == {run.run_id}
    assert events[0]["trigger"] == "cli"
    assert events[-1]["status"] == "success"


def test_progress_and_enrichment_are_appended(events_file):
    run_id = run_events.new_run_id("code_gen")
    with run_events.record_stage(run_id, "code_gen", "dispatch_agent", phase="P3") as stage:
        stage.progress(completed=1, total=3)

    run_events.enrich_stage(
        run_id,
        "code_gen",
        stage.stage_id,
        tokens={"input": 100, "output": 20, "total": 120},
        model="test-model",
    )
    events = run_events.load_events(events_file)
    assert [event["event_type"] for event in events] == [
        "stage_started",
        "stage_progress",
        "stage_finished",
        "stage_enriched",
    ]
    assert events[1]["metrics"] == {"completed": 1, "total": 3}
    assert events[-1]["stage_id"] == stage.stage_id
    assert events[-1]["tokens"]["total"] == 120


def test_stage_sequence_is_scoped_to_run(events_file):
    first_run = run_events.new_run_id("encode")
    second_run = run_events.new_run_id("encode")
    for stage_name in ("parse_rpg", "dep_graph", "save_rpg"):
        with run_events.record_stage(first_run, "encode", stage_name):
            pass
    with run_events.record_stage(second_run, "encode", "parse_rpg"):
        pass

    started = [event for event in run_events.load_events(events_file) if event["event_type"] == "stage_started"]
    assert [event["sequence"] for event in started[:3]] == [1, 2, 3]
    assert started[3]["sequence"] == 1


def test_concurrent_appends_produce_valid_json_lines(events_file):
    run_id = run_events.new_run_id("code_gen")

    def record(index: int) -> None:
        with run_events.record_stage(run_id, "code_gen", f"batch_{index}"):
            pass

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(20)))

    events = run_events.load_events(events_file)
    assert len(events) == 40
    assert len({event["event_id"] for event in events}) == 40
    started = [event for event in events if event["event_type"] == "stage_started"]
    assert sorted(event["sequence"] for event in started) == list(range(1, 21))


def test_loader_skips_incomplete_line(events_file):
    events_file.write_text(
        '{"event_type":"run_started"}\n{"event_type":',
        encoding="utf-8",
    )
    assert run_events.load_events(events_file) == [{"event_type": "run_started"}]


def test_event_context_and_llm_call_are_linked(events_file):
    assert run_events.current_event_context() is None
    with run_events.record_run("encode") as run:
        assert run_events.event_context_environment()["CMIND_RUN_ID"] == run.run_id
        with run_events.record_stage(run.run_id, "encode", "parse_rpg") as stage:
            environment = run_events.event_context_environment()
            assert environment["CMIND_STAGE_ID"] == stage.stage_id
            event_id = run_events.record_llm_call(
                provider="copilot",
                model="gpt-test",
                purpose="parse_features",
                success=True,
                duration_s=1.25,
                token_status="available_in_log",
                log_file="/private/path/process-1.log",
            )
            assert event_id and event_id.startswith("evt-")
        assert "CMIND_STAGE_ID" not in run_events.event_context_environment()
    assert run_events.current_event_context() is None

    llm_event = next(event for event in run_events.load_events(events_file) if event["event_type"] == "llm_call")
    assert llm_event["run_id"] == run.run_id
    assert llm_event["stage_id"] == stage.stage_id
    assert llm_event["log_file"] == "process-1.log"
    assert "prompt" not in llm_event and "response" not in llm_event