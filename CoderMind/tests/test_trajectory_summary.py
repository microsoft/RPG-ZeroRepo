from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.trajectory_summary import collect_trajectory_runs, merge_trajectory_runs


def _trajectory(run_id: str | None = None) -> dict:
    metadata = {
        "repo_name": "demo",
        "node_count": 90,
        "output_path": "/private/rpg.json",
        "unknown_sensitive": "do not copy",
    }
    if run_id:
        metadata["run_id"] = run_id
    return {
        "command": "encode",
        "status": "completed",
        "started_at": "2026-07-29T10:00:00",
        "finished_at": "2026-07-29T10:00:05",
        "error": None,
        "metadata": metadata,
        "steps": [{
            "step_id": 1,
            "name": "parse_rpg",
            "description": "Parse repository",
            "status": "completed",
            "started_at": "2026-07-29T10:00:00",
            "finished_at": "2026-07-29T10:00:04",
            "metadata": {"node_count": 90, "private": "no"},
            "script_calls": [{
                "command": "secret command",
                "started_at": "2026-07-29T10:00:00",
                "finished_at": "2026-07-29T10:00:01",
                "exit_code": 0,
                "stdout": "secret stdout",
                "stderr": "secret stderr",
            }],
            "llm_interactions": [{
                "purpose": "parse_features",
                "prompt": "secret prompt",
                "response": "secret response",
                "success": True,
                "duration_seconds": 2.5,
            }],
        }],
    }


def test_collects_safe_trajectory_summary(tmp_path):
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    path = trajectory_dir / "encode_trajectory_20260729_100000.json"
    path.write_text(json.dumps(_trajectory()), encoding="utf-8")

    runs, health = collect_trajectory_runs(trajectory_dir)
    assert health["status"] == "available"
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == "trajectory:encode_trajectory_20260729_100000"
    assert run["status"] == "success"
    assert run["duration_s"] == 5.0
    assert run["metrics"] == {"node_count": 90}
    assert run["stages"][0]["metrics"] == {"node_count": 90}
    assert run["stages"][0]["telemetry"]["llm"]["calls"] == 1
    assert run["stages"][0]["telemetry"]["scripts"]["calls"] == 1
    serialized = json.dumps(run)
    for secret in ("secret prompt", "secret response", "secret command", "secret stdout", "secret stderr", "do not copy"):
        assert secret not in serialized


def test_merges_exact_run_id_and_keeps_legacy_runs(tmp_path):
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    (trajectory_dir / "matched.json").write_text(json.dumps(_trajectory("run-1")), encoding="utf-8")
    (trajectory_dir / "legacy.json").write_text(json.dumps(_trajectory()), encoding="utf-8")
    trajectory_runs, _ = collect_trajectory_runs(trajectory_dir)
    event_runs = [{
        "run_id": "run-1",
        "started_at": "2026-07-29T10:00:00Z",
        "stages": [{"name": "parse_rpg", "telemetry": {}}],
    }]

    merged = merge_trajectory_runs(event_runs, trajectory_runs)
    assert len(merged) == 2
    matched = next(run for run in merged if run["run_id"] == "run-1")
    assert matched["trajectory_file"] == "matched.json"
    assert matched["stages"][0]["description"] == "Parse repository"
    assert matched["stages"][0]["telemetry"]["trajectory"]["llm"]["calls"] == 1