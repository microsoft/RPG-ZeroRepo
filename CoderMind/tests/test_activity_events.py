from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.activity_events import (  # noqa: E402
    ActivityWriter,
    activity_environment,
    artifact_inventory,
    load_activity_events,
    load_activity_events_with_health,
    maybe_prune_activity_logs,
    prune_activity_logs,
    record_artifact_changes,
    record_activity,
    record_completed_activity,
    workspace_instance_id,
)


def test_concurrent_writers_use_distinct_shards_and_event_ids(tmp_path: Path) -> None:
    root = tmp_path / "activity"

    def write(writer_number: int) -> None:
        writer = ActivityWriter(root, workspace_id="ws_test")
        for item in range(40):
            writer.append(
                "span_progress",
                trace_id=f"trc_{writer_number}",
                span_id=f"spn_{writer_number}",
                parent_span_id=None,
                kind="test",
                name="concurrent writer",
                logical_key="test.concurrent",
                fields={"item": item},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    events = load_activity_events(root)
    assert len(events) == 320
    assert len({event["event_id"] for event in events}) == 320
    assert len(list(root.glob("*/*.jsonl"))) == 8
    for writer_id in {event["writer_id"] for event in events}:
        sequences = [event["writer_sequence"] for event in events if event["writer_id"] == writer_id]
        assert sorted(sequences) == list(range(1, 41))


def test_activity_span_propagates_context_and_records_failure(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")

    with pytest.raises(ValueError):
        with record_activity(
            "workflow",
            "Plan",
            logical_key="decoder-plan",
            trigger="agent",
            writer=writer,
        ) as root:
            environment = activity_environment()
            assert environment["CMIND_TRACE_ID"] == root.trace_id
            assert environment["CMIND_PARENT_SPAN_ID"] == root.span_id
            with record_activity(
                "workflow.stage",
                "Skeleton",
                logical_key="decoder-plan-skeleton",
                writer=writer,
            ):
                raise ValueError("boom")

    events = load_activity_events(tmp_path / "activity")
    assert [event["event_type"] for event in events] == [
        "span_started",
        "span_started",
        "span_finished",
        "span_finished",
    ]
    root_started, child_started, child_finished, root_finished = events
    assert child_started["trace_id"] == root_started["trace_id"]
    assert child_started["parent_span_id"] == root_started["span_id"]
    assert child_finished["span_id"] == child_started["span_id"]
    assert child_finished["status"] == "failed"
    assert root_finished["status"] == "failed"
    assert child_finished["error"] == {"type": "ValueError", "message": "boom"}


def test_loader_ignores_incomplete_final_line(tmp_path: Path) -> None:
    path = tmp_path / "activity" / "2026-08-05" / "wrt_test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"event_id":"evt_valid"}\n{"event_id":', encoding="utf-8")

    assert load_activity_events(tmp_path / "activity") == [{"event_id": "evt_valid"}]


def test_completed_activity_sanitizes_sensitive_and_reserved_fields(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")
    span_id = record_completed_activity(
        "tool.mcp",
        "search_rpg",
        duration_ms=12,
        fields={
            "params": {"query": "secret"},
            "prompt": "secret prompt",
            "result_count": 2,
            "event_id": "must-not-override",
        },
        writer=writer,
    )

    started, finished = load_activity_events(tmp_path / "activity")
    assert started["span_id"] == finished["span_id"] == span_id
    assert finished["duration_ms"] == 12
    assert finished["result_count"] == 2
    assert "params" not in finished and "prompt" not in finished
    assert finished["event_id"] != "must-not-override"


def test_retention_removes_old_closed_shards_but_keeps_today(tmp_path: Path) -> None:
    root = tmp_path / "activity"
    for day, size in (("2026-01-01", 12), ("2026-07-31", 20), ("2026-08-05", 30)):
        path = root / day / "wrt_test.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * size)

    result = prune_activity_logs(
        root,
        retention_days=90,
        max_bytes=35,
        today=date(2026, 8, 5),
    )

    assert result["status"] == "completed"
    assert result["removed_days"] == ["2026-01-01", "2026-07-31"]
    assert not (root / "2026-01-01").exists()
    assert not (root / "2026-07-31").exists()
    assert (root / "2026-08-05" / "wrt_test.jsonl").is_file()
    assert result["bytes_after"] == 30


def test_retention_skips_when_another_maintenance_process_holds_lock(tmp_path: Path) -> None:
    root = tmp_path / "activity"
    root.mkdir()
    (root / ".retention.lock").write_text("locked", encoding="utf-8")

    result = prune_activity_logs(root)

    assert result["status"] == "skipped_locked"


def test_writer_rotates_identity_when_process_id_changes(tmp_path: Path, monkeypatch) -> None:
    writer = ActivityWriter(tmp_path / "activity", writer_id="wrt_parent", workspace_id="ws_test")
    parent_pid = writer._pid
    writer.append(
        "span_progress", trace_id="trc", span_id="spn", parent_span_id=None,
        kind="test", name="parent", logical_key="test",
    )
    monkeypatch.setattr("common.activity_events.os.getpid", lambda: parent_pid + 1)
    child = writer.append(
        "span_progress", trace_id="trc", span_id="spn", parent_span_id=None,
        kind="test", name="child", logical_key="test",
    )

    assert child["writer_id"] != "wrt_parent"
    assert child["writer_sequence"] == 1
    assert len(list((tmp_path / "activity").glob("*/*.jsonl"))) == 2


def test_automatic_retention_runs_only_once_per_day(tmp_path: Path) -> None:
    root = tmp_path / "activity"
    first = maybe_prune_activity_logs(root, today=date(2026, 8, 5))
    second = maybe_prune_activity_logs(root, today=date(2026, 8, 5))

    assert first["status"] == "completed"
    assert second["status"] == "skipped_today"
    assert (root / ".retention-last-run").read_text(encoding="utf-8").strip() == "2026-08-05"


def test_explicit_writer_triggers_daily_retention_marker(tmp_path: Path) -> None:
    root = tmp_path / "activity"
    ActivityWriter(root, workspace_id="ws_test")
    assert (root / ".retention-last-run").is_file()


def test_artifact_changes_record_hashes_without_content(tmp_path: Path) -> None:
    data = tmp_path / "data"
    reports = tmp_path / "reports"
    data.mkdir()
    reports.mkdir()
    existing = data / "existing.json"
    existing.write_text('{"before": true}', encoding="utf-8")
    before = artifact_inventory({"data": data, "reports": reports})

    existing.write_text('{"after": true}', encoding="utf-8")
    (reports / "report.html").write_text("<html>safe</html>", encoding="utf-8")
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")
    with record_activity("command.script", "demo", writer=writer):
        changed = record_artifact_changes(
            before,
            {"data": data, "reports": reports},
            origin="demo.py",
            writer=writer,
        )

    assert changed == ["data/existing.json", "reports/report.html"]
    artifacts = [
        event for event in load_activity_events(tmp_path / "activity")
        if event.get("kind") == "artifact.write" and event["event_type"] == "span_finished"
    ]
    assert len(artifacts) == 2
    assert all(len(event["content_sha256"]) == 64 for event in artifacts)
    assert all(event["artifact_origin"] == "demo.py" for event in artifacts)
    assert "<html>safe</html>" not in str(artifacts)


def test_activity_span_writes_heartbeats_until_finished(tmp_path: Path) -> None:
    writer = ActivityWriter(tmp_path / "activity", workspace_id="ws_test")
    with record_activity("workflow", "slow", writer=writer, heartbeat_interval_s=0.01):
        time.sleep(0.035)

    events = load_activity_events(tmp_path / "activity")
    assert events[0]["event_type"] == "span_started"
    assert events[-1]["event_type"] == "span_finished"
    assert sum(event["event_type"] == "span_heartbeat" for event in events) >= 2
    assert len({event["span_id"] for event in events}) == 1


def test_workspace_instance_id_is_persistent_full_uuid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CMIND_WORKSPACE_INSTANCE_ID", raising=False)
    identity_file = tmp_path / "home-workspace/.workspace-instance-id"
    monkeypatch.setenv("CMIND_WORKSPACE_INSTANCE_FILE", str(identity_file))
    first = workspace_instance_id(tmp_path)
    second = workspace_instance_id(tmp_path)

    assert first == second
    assert first.startswith("ws_")
    assert len(first) == 35
    assert identity_file.read_text(encoding="utf-8").strip() == first


def test_loader_rejects_duplicate_event_ids_and_reports_health(tmp_path: Path) -> None:
    path = tmp_path / "activity/2026-08-05/wrt_test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"event_id":"evt_unique","timestamp":"2026-08-05T00:00:00Z"}\n'
        '{"event_id":"evt_unique","timestamp":"2026-08-05T00:00:01Z"}\n'
        '{broken\n',
        encoding="utf-8",
    )

    events, health = load_activity_events_with_health(tmp_path / "activity")

    assert len(events) == 1
    assert health["status"] == "invalid"
    assert health["duplicate_event_ids"] == ["evt_unique"]
    assert health["invalid_records"] == 1