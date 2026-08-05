"""Safe, renderer-independent summaries of legacy command trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_METRIC_KEYS = {
    "node_count",
    "edge_count",
    "functional_areas",
    "dep_nodes",
    "dep_edges",
    "dep_to_rpg_map_size",
    "output_size_bytes",
    "total_tasks",
    "completed_tasks",
    "failed_tasks",
    "remaining_tasks",
    "task_count",
    "file_count",
    "interface_count",
    "base_class_count",
    "success",
}

_STATUS_MAP = {
    "completed": "success",
    "complete": "success",
    "in_progress": "running",
    "not_started": "not_started",
    "pending": "not_started",
    "failed": "failed",
    "skipped": "skipped",
}


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _duration(started_at: Any, finished_at: Any) -> float | None:
    start = _timestamp(started_at)
    finish = _timestamp(finished_at)
    return round((finish - start).total_seconds(), 3) if start and finish else None


def _metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in _METRIC_KEYS and isinstance(item, (str, int, float, bool))
    }


def _limited_error(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    text = str(value)
    return {"type": "TrajectoryError", "message": text[:1000], "truncated": len(text) > 1000}


def _interaction_summary(interactions: Any) -> dict[str, Any]:
    rows = [row for row in interactions if isinstance(row, dict)] if isinstance(interactions, list) else []
    purposes: dict[str, int] = defaultdict(int)
    total_duration = 0.0
    succeeded = 0
    for row in rows:
        purposes[str(row.get("purpose") or "unknown")] += 1
        total_duration += float(row.get("duration_seconds") or 0)
        succeeded += int(row.get("success") is True)
    return {
        "calls": len(rows),
        "succeeded": succeeded,
        "failed": len(rows) - succeeded,
        "duration_s": round(total_duration, 3),
        "purposes": [
            {"name": name, "calls": count}
            for name, count in sorted(purposes.items(), key=lambda item: (-item[1], item[0]))
        ],
        "token_status": "unavailable_trajectory",
    }


def _script_summary(script_calls: Any) -> dict[str, Any]:
    rows = [row for row in script_calls if isinstance(row, dict)] if isinstance(script_calls, list) else []
    exit_codes: dict[str, int] = defaultdict(int)
    total_duration = 0.0
    for row in rows:
        exit_codes[str(row.get("exit_code"))] += 1
        duration = _duration(row.get("started_at"), row.get("finished_at"))
        total_duration += duration or 0
    return {
        "calls": len(rows),
        "duration_s": round(total_duration, 3),
        "exit_codes": [
            {"code": code, "calls": count}
            for code, count in sorted(exit_codes.items())
        ],
    }


def summarize_trajectory(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    command = str(data.get("command") or "unknown")
    run_id = str(metadata.get("run_id") or f"trajectory:{path.stem}")
    status = _STATUS_MAP.get(str(data.get("status") or ""), str(data.get("status") or "unknown"))
    stages: list[dict[str, Any]] = []
    for index, raw_step in enumerate(data.get("steps") or [], start=1):
        if not isinstance(raw_step, dict):
            continue
        step_id = raw_step.get("step_id", index)
        interactions = _interaction_summary(raw_step.get("llm_interactions"))
        scripts = _script_summary(raw_step.get("script_calls"))
        telemetry: dict[str, Any] = {}
        if interactions["calls"]:
            telemetry["llm"] = interactions
        if scripts["calls"]:
            telemetry["scripts"] = scripts
        stages.append({
            "stage_id": f"trajectory:{path.stem}:{step_id}",
            "sequence": step_id,
            "name": raw_step.get("name"),
            "description": raw_step.get("description"),
            "phase": None,
            "attempt": 1,
            "status": _STATUS_MAP.get(str(raw_step.get("status") or ""), str(raw_step.get("status") or "unknown")),
            "started_at": raw_step.get("started_at"),
            "finished_at": raw_step.get("finished_at"),
            "duration_s": _duration(raw_step.get("started_at"), raw_step.get("finished_at")),
            "metrics": _metrics(raw_step.get("metadata")),
            "error": _limited_error(raw_step.get("error")),
            "progress": [],
            "telemetry": telemetry,
            "event_count": 0,
            "source": "trajectory",
            "quality": "reported_trajectory",
        })

    # Older plan_tasks trajectories passed the Step object where the API
    # expected its integer ID, leaving the sole step pending even though the
    # command completed and wrote its artifact. Reconcile only this strict
    # single-step shape; unrelated pending work remains visible.
    if (
        status == "success"
        and len(stages) == 1
        and stages[0].get("name") == command
        and stages[0].get("status") == "not_started"
    ):
        stages[0].update({
            "status": "success",
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "duration_s": _duration(data.get("started_at"), data.get("finished_at")),
            "quality": "derived_trajectory",
        })
    failed_stages = [stage for stage in stages if stage["status"] in {"failed", "interrupted"}]
    display_status = "completed_with_warnings" if status == "success" and failed_stages else status
    llm_calls = sum(int(stage.get("telemetry", {}).get("llm", {}).get("calls") or 0) for stage in stages)
    llm_duration = sum(float(stage.get("telemetry", {}).get("llm", {}).get("duration_s") or 0) for stage in stages)
    return {
        "run_id": run_id,
        "parent_run_id": metadata.get("parent_run_id"),
        "command": command,
        "trigger": "trajectory",
        "metadata": {"repo_name": metadata.get("repo_name")} if metadata.get("repo_name") else {},
        "status": status,
        "display_status": display_status,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "duration_s": _duration(data.get("started_at"), data.get("finished_at")),
        "metrics": _metrics(metadata),
        "error": _limited_error(data.get("error")),
        "warning_count": len(failed_stages),
        "stages": stages,
        "changes": {},
        "verification": [],
        "artifacts": [],
        "retrievals": [],
        "decisions": [],
        "telemetry": {
            "llm": {
                "calls": llm_calls,
                "duration_s": round(llm_duration, 3),
                "token_status": "unavailable_trajectory",
            }
        } if llm_calls else {},
        "evidence": {},
        "event_count": 0,
        "trajectory_file": path.name,
        "source": "trajectory",
        "quality": "reported_trajectory",
    }


def collect_trajectory_runs(trajectory_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not trajectory_dir.is_dir():
        return [], {
            "source": "trajectories",
            "path": str(trajectory_dir),
            "status": "missing",
            "records": 0,
            "invalid_records": 0,
        }
    runs: list[dict[str, Any]] = []
    invalid = 0
    files = sorted(trajectory_dir.glob("*.json"))
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        if not isinstance(value, dict):
            invalid += 1
            continue
        runs.append(summarize_trajectory(path, value))
    return runs, {
        "source": "trajectories",
        "path": str(trajectory_dir),
        "status": "partial" if invalid else "available",
        "records": len(runs),
        "invalid_records": invalid,
        "files": len(files),
    }


def _started_sort_key(run: dict[str, Any]) -> datetime:
    return _timestamp(run.get("started_at")) or datetime.min.replace(tzinfo=timezone.utc)


def merge_trajectory_runs(
    event_runs: list[dict[str, Any]],
    trajectory_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich exact run-id matches and retain legacy trajectory-only runs."""
    by_id = {str(run.get("run_id")): run for run in event_runs if run.get("run_id")}
    for trajectory_run in trajectory_runs:
        run_id = str(trajectory_run.get("run_id"))
        existing = by_id.get(run_id)
        if existing is None:
            event_runs.append(trajectory_run)
            by_id[run_id] = trajectory_run
            continue
        existing["trajectory_file"] = trajectory_run.get("trajectory_file")
        existing["trajectory_quality"] = "reported_trajectory"
        trajectory_by_name = {
            str(stage.get("name")): stage for stage in trajectory_run.get("stages", []) if stage.get("name")
        }
        for stage in existing.get("stages", []):
            trajectory_stage = trajectory_by_name.get(str(stage.get("name")))
            if not trajectory_stage:
                continue
            stage.setdefault("description", trajectory_stage.get("description"))
            if trajectory_stage.get("telemetry"):
                stage.setdefault("telemetry", {}).setdefault(
                    "trajectory",
                    trajectory_stage["telemetry"],
                )
    return sorted(event_runs, key=_started_sort_key, reverse=True)