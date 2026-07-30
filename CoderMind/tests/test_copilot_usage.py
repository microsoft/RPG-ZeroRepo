from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.copilot_usage import associate_copilot_usage, collect_copilot_usage


def _log_text() -> str:
    return """2026-07-29T10:00:01.000Z [DEBUG] Using model: gpt-test
2026-07-29T10:00:03.000Z [DEBUG] response (Request-ID request-1):
2026-07-29T10:00:03.000Z [DEBUG] data:
2026-07-29T10:00:03.000Z [DEBUG] {
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 40},
    "completion_tokens_details": {"reasoning_tokens": 5}
  },
  "latency_checkpoint": {"total_duration_ms": 1500},
  "choices": [{"message": {"content": "sensitive response"}}]
}
2026-07-29T10:00:04.000Z [INFO] finished
"""


def test_collects_redirected_usage_without_content(tmp_path):
    log_dir = tmp_path / "copilot"
    log_dir.mkdir()
    (log_dir / "process-1.log").write_text(_log_text(), encoding="utf-8")

    usage, health = collect_copilot_usage(log_dir)
    assert health == {
        "source": "copilot_logs",
        "path": str(log_dir),
        "status": "available",
        "records": 1,
        "invalid_records": 0,
        "files": 1,
    }
    assert usage["calls"] == 1
    assert usage["tokens"] == {
        "input": 100,
        "output": 20,
        "total": 120,
        "cache_read": 40,
        "reasoning": 5,
    }
    assert usage["models"][0]["name"] == "gpt-test"
    assert usage["calls_detail"][0]["duration_ms"] == 1500
    assert "sensitive response" not in str(usage)


def test_associates_usage_to_run_and_stage_window(tmp_path):
    log_dir = tmp_path / "copilot"
    log_dir.mkdir()
    (log_dir / "process-1.log").write_text(_log_text(), encoding="utf-8")
    usage, _ = collect_copilot_usage(log_dir)
    runs = [{
        "run_id": "run-1",
        "started_at": "2026-07-29T10:00:00Z",
        "finished_at": "2026-07-29T10:00:05Z",
        "telemetry": {},
        "stages": [{
            "stage_id": "stage-1",
            "started_at": "2026-07-29T10:00:01Z",
            "finished_at": "2026-07-29T10:00:04Z",
        }],
    }]

    association = associate_copilot_usage(runs, usage)
    assert association["associated_calls"] == 1
    assert association["exact_log_file"] == 0
    assert association["inferred_time_window"] == 1
    assert association["quality"] == "inferred_time_window"
    assert association["unassociated_calls"] == 0
    assert runs[0]["telemetry"]["llm"]["tokens"]["total"] == 120
    assert runs[0]["stages"][0]["telemetry"]["llm"]["calls"] == 1
    assert runs[0]["stages"][0]["telemetry"]["llm_calls"][0]["association_quality"] == "inferred_time_window"


def test_associates_legacy_naive_trajectory_timestamps(tmp_path):
    log_dir = tmp_path / "copilot"
    log_dir.mkdir()
    (log_dir / "process-1.log").write_text(_log_text(), encoding="utf-8")
    usage, _ = collect_copilot_usage(log_dir)
    runs = [{
        "run_id": "trajectory:legacy",
        "started_at": "2026-07-29T10:00:00",
        "finished_at": "2026-07-29T10:00:05",
        "telemetry": {},
        "stages": [],
    }]

    association = associate_copilot_usage(runs, usage)
    assert association["associated_calls"] == 1
    assert association["inferred_time_window"] == 1


def test_prefers_exact_log_file_association(tmp_path):
    log_dir = tmp_path / "copilot"
    log_dir.mkdir()
    (log_dir / "process-1.log").write_text(_log_text(), encoding="utf-8")
    usage, _ = collect_copilot_usage(log_dir)
    runs = [{
        "run_id": "run-1",
        "started_at": "2026-07-29T11:00:00Z",
        "finished_at": "2026-07-29T11:00:05Z",
        "telemetry": {"llm_invocations": [{
            "stage_id": "stage-1",
            "provider": "copilot",
            "log_file": "process-1.log",
            "token_status": "available_in_workspace_log",
        }]},
        "stages": [{
            "stage_id": "stage-1",
            "started_at": "2026-07-29T11:00:01Z",
            "finished_at": "2026-07-29T11:00:04Z",
            "telemetry": {"llm_invocations": [{
                "stage_id": "stage-1",
                "provider": "copilot",
                "log_file": "process-1.log",
                "token_status": "available_in_workspace_log",
            }]},
        }],
    }]

    association = associate_copilot_usage(runs, usage)
    assert association["exact_log_file"] == 1
    assert association["inferred_time_window"] == 0
    assert association["quality"] == "exact_log_file"
    assert runs[0]["telemetry"]["llm_calls"][0]["association_quality"] == "exact_log_file"
    assert runs[0]["telemetry"]["llm"]["calls"] == 1
    assert runs[0]["telemetry"]["llm"]["tokens"]["total"] == 120
    assert runs[0]["stages"][0]["telemetry"]["llm"]["calls"] == 1