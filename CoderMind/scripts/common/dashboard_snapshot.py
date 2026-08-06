"""Collect CoderMind runtime facts into a renderer-independent snapshot."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from common.activity_history import collect_run_history
from common.copilot_usage import associate_copilot_usage, collect_copilot_usage
from common.rpg_diff import previous_rpg_version, read_rpg_version, semantic_rpg_diff
from common.dashboard_schema import assert_valid_snapshot, sanitize_snapshot
from common.trajectory_summary import collect_trajectory_runs, merge_trajectory_runs
from common.paths import (
    DASHBOARD_SNAPSHOT_FILE,
    DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    RPG_FILE,
    RUN_EVENTS_FILE,
    WORKSPACE_ROOT,
)


SNAPSHOT_SCHEMA_VERSION = 1

_DECODER_ARTIFACTS = (
    "feature_spec.json",
    "feature_build.json",
    "feature_tree.json",
    "skeleton.json",
    "data_flow.json",
    "base_classes.json",
    "interfaces.json",
    "tasks.json",
    "code_gen_state.jsonl",
)

_ARTIFACT_SPECS = (
    ("feature_spec", "data", "feature_spec.json"),
    ("feature_build", "data", "feature_build.json"),
    ("feature_tree", "data", "feature_tree.json"),
    ("skeleton", "data", "skeleton.json"),
    ("skeleton_summary", "data", "skeleton_summary.txt"),
    ("data_flow", "data", "data_flow.json"),
    ("data_flow_viz", "data", "data_flow_viz.html"),
    ("base_classes", "data", "base_classes.json"),
    ("interfaces", "data", "interfaces.json"),
    ("rpg_json", "data", "rpg.json"),
    ("tasks", "data", "tasks.json"),
    ("code_gen_state", "data", "code_gen_state.jsonl"),
    ("codegen_final_test", "logs", "codegen_final_test.json"),
    ("rpg_edit_plan", "data", "rpg_edit_plan.json"),
    ("rpg_edit_impact", "data", "rpg_edit_impact.json"),
    ("rpg_edit_validate", "data", "rpg_edit_validate.json"),
    ("rpg_edit_locate", "data", "rpg_edit_locate.json"),
    ("rpg_edit_code_result", "data", "rpg_edit_code_result.json"),
    ("rpg_edit_apply_result", "data", "rpg_edit_apply_result.json"),
    ("rpg_edit_review_result", "data", "rpg_edit_review_result.json"),
    ("rpg_html", "reports", "rpg.html"),
    ("run_events", "logs", "run_events.jsonl"),
    ("mcp_calls", "logs", "mcp_calls.jsonl"),
    ("hook_calls", "logs", "hook_calls.jsonl"),
)

_DECODER_PIPELINE = (
    ("feature_spec", "Feature spec", "feature_spec"),
    ("feature_build", "Feature build", "feature_build"),
    ("feature_refactor", "Feature tree", "feature_tree"),
    ("build_skeleton", "Skeleton", "skeleton"),
    ("build_data_flow", "Data flow", "data_flow"),
    ("design_base_classes", "Base classes", "base_classes"),
    ("design_interfaces", "Interfaces", "interfaces"),
    ("plan_tasks", "Tasks", "tasks"),
    ("code_gen", "Code generation", "code_gen_state"),
)

_ENCODER_PIPELINE = (
    ("parse_rpg", "Parse RPG", "rpg_json"),
    ("dep_graph", "Dependency graph", "rpg_json"),
    ("save_rpg", "Save RPG", "rpg_json"),
    ("visualize", "Visualization", "rpg_html"),
)

_NEXT_COMMAND = {
    "feature_spec": "/cmind.feature_spec",
    "feature_build": "/cmind.feature_build",
    "feature_refactor": "/cmind.feature_refactor",
    "build_skeleton": "/cmind.build_skeleton",
    "build_data_flow": "/cmind.build_data_flow",
    "design_base_classes": "/cmind.design_base_classes",
    "design_interfaces": "/cmind.design_interfaces",
    "plan_tasks": "/cmind.plan_tasks",
    "code_gen": "/cmind.code_gen",
    "parse_rpg": "/cmind.encode",
    "dep_graph": "/cmind.encode",
    "save_rpg": "/cmind.encode",
    "visualize": "/cmind.encode",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class DashboardSources:
    workspace_root: Path
    data_dir: Path
    logs_dir: Path
    reports_dir: Path
    run_events_file: Path
    rpg_file: Path
    snapshot_file: Path

    @classmethod
    def defaults(cls) -> "DashboardSources":
        return cls(
            workspace_root=WORKSPACE_ROOT,
            data_dir=DATA_DIR,
            logs_dir=LOGS_DIR,
            reports_dir=REPORTS_DIR,
            run_events_file=RUN_EVENTS_FILE,
            rpg_file=RPG_FILE,
            snapshot_file=DASHBOARD_SNAPSHOT_FILE,
        )


def _source_health(
    name: str,
    path: Path,
    status: str,
    *,
    records: int = 0,
    invalid_records: int = 0,
    detail: str | None = None,
) -> dict[str, Any]:
    health: dict[str, Any] = {
        "source": name,
        "path": str(path),
        "status": status,
        "records": records,
        "invalid_records": invalid_records,
    }
    if detail:
        health["detail"] = detail
    return health


def classify_source_expectations(
    health: list[dict[str, Any]],
    *,
    mode: str,
    has_runs: bool,
    has_rpg_edit: bool,
) -> list[dict[str, Any]]:
    """Mark whether each source is required, optional, or not expected here."""
    required = {"git"}
    optional = {"activity", "rpg_history", "rpg_latest_change", "copilot_logs"}
    if has_runs:
        optional.update({"run_events", "trajectories"})
    if mode == "encoder":
        required.add("rpg")
        optional.update({"run_events", "hook_calls", "mcp_calls", "trajectories"})
    elif mode == "decoder":
        optional.update({"rpg", "run_events", "trajectories", "code_gen_state"})
    if has_rpg_edit:
        optional.update({
            "rpg_edit_validate", "rpg_edit_locate", "rpg_edit_plan",
            "rpg_edit_impact", "rpg_edit_code", "rpg_edit_apply",
            "rpg_edit_review",
        })

    classified: list[dict[str, Any]] = []
    for item in health:
        source = str(item.get("source") or "")
        expectation = "required" if source in required else "optional" if source in optional else "not_expected"
        classified.append({**item, "expectation": expectation})
    return classified


def load_jsonl_records(path: Path, *, source_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read valid JSON objects while reporting malformed or unsupported rows."""
    if not path.exists():
        return [], _source_health(source_name, path, "missing")

    records: list[dict[str, Any]] = []
    invalid_records = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid_records += 1
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    invalid_records += 1
    except OSError as exc:
        return [], _source_health(source_name, path, "invalid", detail=str(exc))

    status = "available" if not invalid_records else "partial"
    return records, _source_health(
        source_name,
        path,
        status,
        records=len(records),
        invalid_records=invalid_records,
    )


def load_json_object(path: Path, *, source_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one JSON object and report missing, malformed, or wrong-shaped data."""
    if not path.exists():
        return {}, _source_health(source_name, path, "missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, _source_health(source_name, path, "invalid", detail=str(exc))
    if not isinstance(value, dict):
        return {}, _source_health(source_name, path, "invalid", detail="expected a JSON object")
    return value, _source_health(source_name, path, "available", records=1)


def _first(events: Iterable[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("event_type") == event_type), None)


def _last(events: Iterable[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in reversed(list(events)) if event.get("event_type") == event_type), None)


def _aggregate_stage(stage_id: str, events: list[dict[str, Any]], *, run_finished: bool) -> dict[str, Any]:
    started = _first(events, "stage_started")
    finished = _last(events, "stage_finished")
    anchor = started or finished or events[0]
    progress_events = [event for event in events if event.get("event_type") == "stage_progress"]
    enrichments = [event for event in events if event.get("event_type") == "stage_enriched"]
    llm_invocations = [
        {
            key: event.get(key)
            for key in (
                "event_id", "timestamp", "stage_id", "provider", "model", "purpose",
                "success", "duration_s", "tokens", "token_status", "log_file",
            )
            if event.get(key) is not None
        }
        for event in events
        if event.get("event_type") == "llm_call"
    ]

    if finished:
        status = finished.get("status") or "unknown"
    elif run_finished:
        status = "interrupted"
    else:
        status = "running"

    metrics: dict[str, Any] = {}
    if isinstance(finished, dict) and isinstance(finished.get("metrics"), dict):
        metrics.update(finished["metrics"])

    enrichment: dict[str, Any] = {}
    for event in enrichments:
        for key, value in event.items():
            if key not in {
                "schema_version",
                "event_id",
                "event_type",
                "timestamp",
                "run_id",
                "command",
                "stage_id",
            }:
                enrichment[key] = value

    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "sequence": anchor.get("sequence"),
        "name": anchor.get("stage"),
        "phase": anchor.get("phase"),
        "attempt": anchor.get("attempt", 1),
        "status": status,
        "started_at": (started or anchor).get("started_at") or (started or anchor).get("timestamp"),
        "finished_at": finished.get("finished_at") if finished else None,
        "duration_s": finished.get("duration_s") if finished else None,
        "metrics": metrics,
        "error": finished.get("error") if finished else None,
        "progress": [event.get("metrics", {}) for event in progress_events],
        "telemetry": {"llm_invocations": llm_invocations} if llm_invocations else {},
        "event_count": len(events),
    }
    stage.update(enrichment)
    return stage


def aggregate_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge append-only lifecycle events into renderer-ready run records."""
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_order: list[str] = []
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue
        run_id = str(run_id)
        if run_id not in by_run:
            run_order.append(run_id)
        by_run[run_id].append(event)

    runs: list[dict[str, Any]] = []
    for run_id in run_order:
        run_events = by_run[run_id]
        started = _first(run_events, "run_started")
        finished = _last(run_events, "run_finished")
        anchor = started or finished or run_events[0]

        stage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        stage_order: list[str] = []
        for event in run_events:
            stage_id = event.get("stage_id")
            if not stage_id:
                continue
            stage_id = str(stage_id)
            if stage_id not in stage_groups:
                stage_order.append(stage_id)
            stage_groups[stage_id].append(event)
        stages = [
            _aggregate_stage(stage_id, stage_groups[stage_id], run_finished=finished is not None)
            for stage_id in stage_order
        ]
        stages.sort(key=lambda stage: (stage.get("sequence") is None, stage.get("sequence") or 0))
        llm_invocations = [
            {
                key: event.get(key)
                for key in (
                    "event_id", "timestamp", "stage_id", "provider", "model", "purpose",
                    "success", "duration_s", "tokens", "token_status", "log_file",
                )
                if event.get(key) is not None
            }
            for event in run_events
            if event.get("event_type") == "llm_call"
        ]

        raw_status = (finished or anchor).get("status") or ("running" if not finished else "unknown")
        failed_stages = [
            stage for stage in stages
            if stage["status"] in {"failed", "cancelled", "timed_out", "interrupted"}
        ]
        display_status = raw_status
        if raw_status == "success" and failed_stages:
            display_status = "completed_with_warnings"

        runs.append({
            "run_id": run_id,
            "parent_run_id": anchor.get("parent_run_id"),
            "command": anchor.get("command"),
            "trigger": (started or {}).get("trigger"),
            "metadata": (started or {}).get("metadata", {}),
            "status": raw_status,
            "display_status": display_status,
            "started_at": (started or anchor).get("started_at") or (started or anchor).get("timestamp"),
            "finished_at": finished.get("finished_at") if finished else None,
            "duration_s": finished.get("duration_s") if finished else None,
            "metrics": finished.get("metrics", {}) if finished else {},
            "error": finished.get("error") if finished else None,
            "warning_count": len(failed_stages) if display_status == "completed_with_warnings" else 0,
            "stages": stages,
            "changes": {},
            "verification": [],
            "artifacts": [],
            "retrievals": [],
            "decisions": [],
            "telemetry": {"llm_invocations": llm_invocations} if llm_invocations else {},
            "evidence": {},
            "event_count": len(run_events),
        })

    return list(reversed(runs))


def _run_git(workspace_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _git_change_type(status: str) -> str:
    code = status[:1].upper()
    return {
        "A": "added",
        "D": "deleted",
        "R": "renamed",
        "C": "copied",
        "M": "modified",
        "T": "type_changed",
    }.get(code, "changed")


def _git_change_rows(workspace_root: Path, base_commit: str, target_commit: str) -> list[dict[str, Any]]:
    name_status = _run_git(
        workspace_root,
        "diff",
        "--name-status",
        "--find-renames",
        f"{base_commit}..{target_commit}",
        "--",
        ".",
    )
    if name_status is None:
        return []

    numstat = _run_git(
        workspace_root,
        "diff",
        "--numstat",
        "--find-renames",
        f"{base_commit}..{target_commit}",
        "--",
        ".",
    ) or ""
    line_counts: dict[str, tuple[int | None, int | None]] = {}
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = int(parts[0]) if parts[0].isdigit() else None
        deleted = int(parts[1]) if parts[1].isdigit() else None
        line_counts[parts[-1]] = (added, deleted)

    rows: list[dict[str, Any]] = []
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[-1]
        before = parts[1] if status.upper().startswith(("R", "C")) and len(parts) >= 3 else None
        added, deleted = line_counts.get(path, (None, None))
        row: dict[str, Any] = {
            "path": path,
            "change_type": _git_change_type(status),
            "git_status": status,
            "lines_added": added,
            "lines_deleted": deleted,
        }
        if before and before != path:
            row["before_path"] = before
        rows.append(row)
    return rows


def _rpg_node_index(root: Any) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id") or node.get("node_id")): node
        for node in _walk_rpg_tree(root)
        if node.get("id") or node.get("node_id")
    }


def _dep_node_items(nodes: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(nodes, dict):
        return [
            (str(node_id), attrs if isinstance(attrs, dict) else {})
            for node_id, attrs in nodes.items()
        ]
    if isinstance(nodes, list):
        return [
            (str(attrs.get("id") or attrs.get("node_id") or index), attrs)
            for index, attrs in enumerate(nodes)
            if isinstance(attrs, dict)
        ]
    return []


def _normalize_path(value: Any, workspace_root: Path) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        try:
            path = path.relative_to(workspace_root)
        except ValueError:
            return path.as_posix()
    text = path.as_posix()
    return text[2:] if text.startswith("./") else text


def _dep_path(dep_id: str, attrs: dict[str, Any], workspace_root: Path) -> str:
    for key in ("path", "file", "code_path", "module"):
        if attrs.get(key) not in (None, ""):
            return _normalize_path(attrs[key], workspace_root)
    return _normalize_path(dep_id.split(":", 1)[0], workspace_root)


def _path_matches(dep_path: str, changed_path: str) -> bool:
    if not dep_path or not changed_path:
        return False
    return (
        dep_path == changed_path
        or dep_path.endswith("/" + changed_path)
        or changed_path.endswith("/" + dep_path)
    )


def collect_focused_impact(
    changed_files: list[dict[str, Any]],
    rpg_data: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    """Map changed files to current dep nodes and their RPG feature nodes."""
    dep_graph = rpg_data.get("dep_graph") if isinstance(rpg_data.get("dep_graph"), dict) else {}
    dep_items = _dep_node_items(dep_graph.get("nodes"))
    dep_to_rpg = rpg_data.get("_dep_to_rpg_map") if isinstance(rpg_data.get("_dep_to_rpg_map"), dict) else {}
    rpg_nodes = _rpg_node_index(rpg_data.get("root"))

    dep_rows: list[dict[str, Any]] = []
    mapped_rpg_ids: set[str] = set()
    mapped_files: set[str] = set()
    changed_paths = [str(row.get("path") or "") for row in changed_files]
    for dep_id, attrs in dep_items:
        path = _dep_path(dep_id, attrs, workspace_root)
        matches = [changed_path for changed_path in changed_paths if _path_matches(path, changed_path)]
        if not matches:
            continue
        mapped_ids = {
            str(node_id)
            for node_id in [*(attrs.get("rpg_nodes") or []), *(dep_to_rpg.get(dep_id) or [])]
            if node_id not in (None, "")
        }
        mapped_rpg_ids.update(mapped_ids)
        mapped_files.update(matches)
        dep_rows.append({
            "dep_node_id": dep_id,
            "path": path,
            "type": attrs.get("type") or attrs.get("type_name"),
            "changed_files": matches,
            "mapped_rpg_node_ids": sorted(mapped_ids),
        })

    rpg_rows = []
    for node_id in sorted(mapped_rpg_ids):
        node = rpg_nodes.get(node_id, {})
        meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
        rpg_rows.append({
            "node_id": node_id,
            "name": node.get("name"),
            "node_type": node.get("node_type"),
            "path": meta.get("path"),
        })

    return {
        "available": bool(dep_items),
        "quality": "derived" if dep_items else "missing",
        "changed_files": changed_paths,
        "mapped_files": sorted(mapped_files),
        "unmapped_files": sorted(set(changed_paths) - mapped_files),
        "dependency_nodes": dep_rows,
        "rpg_nodes": rpg_rows,
        "summary": {
            "changed_files": len(changed_paths),
            "mapped_files": len(mapped_files),
            "dependency_nodes": len(dep_rows),
            "rpg_nodes": len(rpg_rows),
        },
    }


def collect_run_changes(
    run: dict[str, Any],
    sources: DashboardSources,
    rpg_data: dict[str, Any],
) -> dict[str, Any]:
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    base_commit = metrics.get("previous_commit") or metrics.get("prev_ref")
    target_commit = metrics.get("new_commit")
    graph_deltas = {
        key: metrics[key]
        for key in (
            "nodes_delta",
            "edges_delta",
            "dep_nodes_delta",
            "dep_edges_delta",
            "dep_to_rpg_map_size",
        )
        if metrics.get(key) is not None
    }
    if not base_commit or not target_commit:
        return {
            "available": False,
            "quality": "missing",
            "reason": "run does not record previous_commit and new_commit",
            "graph_deltas": graph_deltas,
            "files": [],
            "focused_impact": {},
        }

    files = _git_change_rows(sources.workspace_root, str(base_commit), str(target_commit))
    summary = defaultdict(int)
    for row in files:
        summary[str(row["change_type"])] += 1
    return {
        "available": True,
        "quality": "derived",
        "source": "git diff previous_commit..new_commit",
        "base_commit": base_commit,
        "target_commit": target_commit,
        "summary": dict(summary),
        "files": files,
        "graph_deltas": graph_deltas,
        "focused_impact": collect_focused_impact(files, rpg_data, sources.workspace_root),
    }


def _normalize_verification(value: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, list):
        values = [row for row in value if isinstance(row, dict)]
    else:
        return []
    checks: list[dict[str, Any]] = []
    for index, row in enumerate(values, start=1):
        checks.append({
            "name": row.get("name") or row.get("check") or f"verification_{index}",
            "status": row.get("status") if row.get("status") is not None else row.get("passed"),
            "detail": row.get("detail") or row.get("message") or row.get("reason"),
            "source": source,
            "quality": "reported",
        })
    return checks


def collect_run_verification(run: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build checks only from events belonging to this run."""
    checks: list[dict[str, Any]] = [{
        "name": "run lifecycle",
        "status": run.get("display_status"),
        "detail": (run.get("error") or {}).get("message") if isinstance(run.get("error"), dict) else run.get("error"),
        "source": "run_events",
        "quality": "measured",
    }]
    next_actions: list[dict[str, Any]] = []

    result_metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    checks.extend(_normalize_verification(result_metrics.get("verification"), source="run_finished.metrics"))
    if result_metrics.get("next_action"):
        next_actions.append({
            "label": "Reported next action",
            "command": None,
            "detail": str(result_metrics["next_action"]),
            "source": "run_finished.metrics",
            "quality": "reported",
        })

    for stage in run.get("stages", []):
        stage_name = str(stage.get("name") or "stage")
        detail = None
        error = stage.get("error")
        if isinstance(error, dict):
            detail = error.get("message")
        elif error:
            detail = str(error)
        checks.append({
            "name": stage_name,
            "status": stage.get("status"),
            "detail": detail,
            "source": f"run_events:{stage.get('stage_id')}",
            "quality": "measured",
        })
        metrics = stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {}
        checks.extend(_normalize_verification(metrics.get("verification"), source=f"stage:{stage_name}"))
        if metrics.get("next_action"):
            next_actions.append({
                "label": f"Next action for {stage_name}",
                "command": None,
                "detail": str(metrics["next_action"]),
                "source": f"stage:{stage_name}",
                "quality": "reported",
            })
        if stage.get("status") in {"failed", "interrupted", "cancelled", "timed_out"}:
            next_actions.append({
                "label": f"Retry {stage_name}",
                "command": _NEXT_COMMAND.get(stage_name),
                "detail": detail or f"Stage status is {stage.get('status')}",
                "source": f"run_events:{stage.get('stage_id')}",
                "quality": "derived",
            })
    return checks, next_actions


def collect_workspace_verification(
    pipeline: list[dict[str, Any]],
    rpg: dict[str, Any],
    tasks: dict[str, Any],
    codegen_final_test: dict[str, Any],
) -> dict[str, Any]:
    """Summarize current workspace readiness separately from historical runs."""
    checks: list[dict[str, Any]] = []
    for step in pipeline:
        checks.append({
            "name": step["id"],
            "status": step["status"],
            "detail": f"artifact={step['artifact']}",
            "source": "pipeline",
            "quality": step["quality"],
        })

    mapping = rpg.get("mapping") if isinstance(rpg.get("mapping"), dict) else {}
    if mapping:
        checks.append({
            "name": "RPG mapping coverage",
            "status": "available" if mapping.get("total_dep_nodes") else "not_applicable",
            "detail": {
                "mapped": mapping.get("mapped_dep_nodes"),
                "total": mapping.get("total_dep_nodes"),
                "coverage_percent": mapping.get("coverage_percent"),
            },
            "source": "rpg.json",
            "quality": "derived",
        })
    if tasks:
        checks.append({
            "name": "code generation tasks",
            "status": "failed" if tasks.get("failed") else "in_progress" if tasks.get("pending") else "completed",
            "detail": {
                "total": tasks.get("total"),
                "completed": tasks.get("completed"),
                "failed": tasks.get("failed"),
                "pending": tasks.get("pending"),
            },
            "source": "code_gen_state.jsonl",
            "quality": "reported",
        })
    if codegen_final_test:
        checks.append({
            "name": "code_gen final test",
            "status": "completed" if codegen_final_test.get("success") is True else "failed",
            "detail": {
                key: codegen_final_test.get(key)
                for key in (
                    "passed", "failed", "errors", "no_tests_executed",
                    "toolchain_unavailable",
                )
                if codegen_final_test.get(key) is not None
            },
            "source": "codegen_final_test.json",
            "quality": "reported",
        })

    next_step = next((step for step in pipeline if step["status"] != "completed"), None)
    next_actions = []
    if next_step:
        next_actions.append({
            "label": f"Continue with {next_step['label']}",
            "command": _NEXT_COMMAND.get(next_step["id"]),
            "detail": f"Current status: {next_step['status']}",
            "source": "pipeline",
            "quality": "derived",
        })
    return {"checks": checks, "next_actions": next_actions}


def collect_automation_activity(history: dict[str, Any]) -> dict[str, Any]:
    """Summarize on-demand MCP sessions and Git-triggered hook workflows."""
    roots = [root for root in history.get("roots") or [] if isinstance(root, dict)]

    def walk(node: dict[str, Any]) -> list[dict[str, Any]]:
        children = [child for child in node.get("children") or [] if isinstance(child, dict)]
        return [node, *(item for child in children for item in walk(child))]

    mcp_roots = [root for root in roots if root.get("kind") == "mcp.session"]
    hook_roots = [root for root in roots if root.get("kind") == "hook.workflow"]
    mcp_calls = [node for root in mcp_roots for node in walk(root) if node.get("kind") == "tool.mcp"]
    hook_nodes = [node for root in hook_roots for node in walk(root)]
    hook_operations = [node for node in hook_nodes if node.get("kind") == "hook.operation"]
    hook_updates = [
        node for node in hook_nodes
        if node.get("kind") == "workflow" and node.get("logical_key") == "encoder-update-rpg"
    ]
    hook_attribution_mismatches = 0
    for root in hook_roots:
        details = root.get("details") if isinstance(root.get("details"), dict) else {}
        trigger_sha = str(details.get("git_sha") or "")
        updates = [
            node for node in walk(root)
            if node.get("kind") == "workflow" and node.get("logical_key") == "encoder-update-rpg"
        ]
        for update in updates:
            metrics = update.get("metrics") if isinstance(update.get("metrics"), dict) else {}
            target_sha = str(metrics.get("new_commit") or "")
            if trigger_sha and target_sha and not target_sha.startswith(trigger_sha):
                hook_attribution_mismatches += 1

    special_roots = [*mcp_roots, *hook_roots]
    latest = max(
        special_roots,
        key=lambda root: str(root.get("finished_at") or root.get("started_at") or ""),
        default=None,
    )
    latest_summary: dict[str, Any] = {}
    if latest is not None:
        details = latest.get("details") if isinstance(latest.get("details"), dict) else {}
        latest_summary = {
            "type": "mcp" if latest.get("kind") == "mcp.session" else "hook",
            "label": (
                f"{latest.get('name') or 'MCP session'} / {details.get('client_context')}"
                if latest.get("kind") == "mcp.session" and details.get("client_context")
                else latest.get("name") or latest.get("logical_key")
            ),
            "status": latest.get("status"),
            "started_at": latest.get("started_at"),
            "finished_at": latest.get("finished_at"),
            "duration_ms": latest.get("duration_ms"),
            "trace_id": latest.get("trace_id"),
            "hook_type": details.get("hook_type"),
            "git_sha": details.get("git_sha"),
            "client_context": details.get("client_context"),
        }

    failure_statuses = {"failed", "error", "timed_out", "cancelled", "interrupted"}
    return {
        "latest": latest_summary,
        "mcp": {
            "sessions": len(mcp_roots),
            "calls": len(mcp_calls),
            "succeeded": sum(str(call.get("status")) == "success" for call in mcp_calls),
            "degraded": sum(str(call.get("status")) == "degraded" for call in mcp_calls),
            "failed": sum(str(call.get("status")) in failure_statuses for call in mcp_calls),
        },
        "hooks": {
            "invocations": len(hook_roots),
            "post_commit": sum(root.get("name") == "post-commit" for root in hook_roots),
            "post_merge": sum(root.get("name") == "post-merge" for root in hook_roots),
            "operations": len(hook_operations),
            "updates": len(hook_updates),
            "failed": sum(str(root.get("status")) in failure_statuses for root in hook_roots),
            "attribution_mismatches": hook_attribution_mismatches,
        },
    }


def collect_workspace(sources: DashboardSources) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect reproducibility context without exposing configuration contents."""
    commit = _run_git(sources.workspace_root, "rev-parse", "HEAD")
    branch = _run_git(sources.workspace_root, "branch", "--show-current")
    status = _run_git(sources.workspace_root, "status", "--porcelain")
    git_available = commit is not None
    try:
        tool_version = version("cmind-cli")
    except PackageNotFoundError:
        tool_version = None

    workspace = {
        "name": os.environ.get("CMIND_REPO_NAME", "").strip() or sources.workspace_root.name,
        "path": str(sources.workspace_root),
        "tool_version": tool_version,
        "git": {
            "available": git_available,
            "branch": branch or None,
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
        },
    }
    git_health = _source_health(
        "git",
        sources.workspace_root / ".git",
        "available" if git_available else "missing",
        records=1 if git_available else 0,
    )
    return workspace, git_health


def _walk_rpg_tree(root: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        nodes.append(node)
        for child in node.get("children") or []:
            visit(child)

    visit(root)
    return nodes


def _graph_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list):
        return value
    return []


def _distribution(values: Iterable[Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value in (None, ""):
            continue
        counts[str(value)] += 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _semantic_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in _graph_items(data.get("edges"))
        if isinstance(edge, dict)
        and str(edge.get("relation") or "").lower() not in {"contains", "composes"}
    ]


def collect_rpg(data: dict[str, Any]) -> dict[str, Any]:
    """Compute transparent RPG, dependency, and mapping metrics."""
    tree_nodes = _walk_rpg_tree(data.get("root"))
    if not tree_nodes:
        tree_nodes = [node for node in _graph_items(data.get("nodes")) if isinstance(node, dict)]
    semantic_edges = _semantic_edges(data)
    dep_graph = data.get("dep_graph") if isinstance(data.get("dep_graph"), dict) else {}
    dep_nodes_value = dep_graph.get("nodes", {})
    dep_nodes = [node for node in _graph_items(dep_nodes_value) if isinstance(node, dict)]
    dep_edges = [edge for edge in _graph_items(dep_graph.get("edges")) if isinstance(edge, dict)]
    dep_to_rpg = data.get("_dep_to_rpg_map") if isinstance(data.get("_dep_to_rpg_map"), dict) else {}

    mapped_dep_ids = {
        str(dep_id)
        for dep_id, mapped in dep_to_rpg.items()
        if isinstance(mapped, (list, tuple, set)) and bool(mapped)
    }
    if isinstance(dep_nodes_value, dict):
        mapped_dep_ids.update(
            str(dep_id)
            for dep_id, node in dep_nodes_value.items()
            if isinstance(node, dict) and bool(node.get("rpg_nodes"))
        )
        dep_ids = {str(dep_id) for dep_id in dep_nodes_value}
    else:
        dep_ids = {
            str(node.get("id") or node.get("node_id"))
            for node in dep_nodes
            if node.get("id") or node.get("node_id")
        }
    mapped_dep_count = len(mapped_dep_ids & dep_ids) if dep_ids else len(mapped_dep_ids)
    dep_node_count = len(dep_nodes)
    coverage_percent = round(mapped_dep_count / dep_node_count * 100, 1) if dep_node_count else None

    metadata = [node.get("meta", {}) for node in tree_nodes if isinstance(node.get("meta"), dict)]
    return {
        "repo_name": data.get("repo_name") or "unknown",
        "feature_graph": {
            "nodes": len(tree_nodes),
            "semantic_edges": len(semantic_edges),
            "functional_areas": sum(1 for node in tree_nodes if node.get("node_type") == "functional_area"),
            "node_types": _distribution(node.get("node_type") for node in tree_nodes),
        },
        "dependency_graph": {
            "nodes": dep_node_count,
            "edges": len(dep_edges),
            "node_types": _distribution(node.get("type") or node.get("type_name") for node in dep_nodes),
            "edge_types": _distribution(
                (edge.get("attrs") or {}).get("type") if isinstance(edge.get("attrs"), dict) else edge.get("type")
                for edge in dep_edges
            ),
        },
        "mapping": {
            "mapped_dep_nodes": mapped_dep_count,
            "total_dep_nodes": dep_node_count,
            "unmapped_dep_nodes": max(dep_node_count - mapped_dep_count, 0),
            "coverage_percent": coverage_percent,
            "definition": "dep nodes with at least one RPG mapping / all dep graph nodes",
            "mapping_relations": sum(
                len(mapped) for mapped in dep_to_rpg.values() if isinstance(mapped, (list, tuple, set))
            ),
        },
        "code": {
            "type_distribution": _distribution(meta.get("type_name") for meta in metadata),
            "language_distribution": _distribution(meta.get("language") for meta in metadata),
        },
    }


def collect_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Preserve graph payloads required by the renderer without rereading RPG JSON."""
    if not data:
        return {}
    dep_graph = data.get("dep_graph") if isinstance(data.get("dep_graph"), dict) else {}
    dep_to_rpg = data.get("_dep_to_rpg_map") if isinstance(data.get("_dep_to_rpg_map"), dict) else {}
    return {
        "feature_root": data.get("root") if isinstance(data.get("root"), dict) else None,
        "feature_nodes": data.get("nodes") if isinstance(data.get("nodes"), (list, dict)) else None,
        "semantic_edges": _semantic_edges(data),
        "dependency_graph": dep_graph,
        "dep_to_rpg_map": dep_to_rpg,
    }


def _artifact_path(sources: DashboardSources, location: str, filename: str) -> Path:
    if location == "data":
        return sources.data_dir / filename
    if location == "reports":
        return sources.reports_dir / filename
    return sources.logs_dir / filename


def collect_artifacts(sources: DashboardSources) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for label, location, filename in _ARTIFACT_SPECS:
        path = _artifact_path(sources, location, filename)
        exists = path.is_file()
        artifact: dict[str, Any] = {
            "label": label,
            "location": location,
            "path": str(path),
            "status": "available" if exists else "missing",
            "size_bytes": None,
            "modified_at": None,
        }
        if exists:
            try:
                stat = path.stat()
                artifact["size_bytes"] = stat.st_size
                artifact["modified_at"] = datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except OSError:
                artifact["status"] = "unreadable"
        artifacts.append(artifact)
    return artifacts


def _latest_stage_by_name(runs: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], str]]:
    latest: dict[str, tuple[dict[str, Any], str]] = {}
    for run in runs:
        stages = run.get("stages", [])
        for stage in stages:
            name = stage.get("name")
            if name and name not in latest:
                latest[str(name)] = (stage, str(run.get("run_id") or ""))
        command = run.get("command")
        has_unfinished_stage = any(
            stage.get("status") in {"not_started", "pending", "running", "in_progress", "unknown"}
            for stage in stages
        )
        if command and command not in latest and not has_unfinished_stage:
            latest[str(command)] = ({
                "name": command,
                "status": run.get("display_status") or run.get("status") or "unknown",
                "duration_s": run.get("duration_s"),
                "attempt": run.get("attempt"),
                "error": run.get("error"),
            }, str(run.get("run_id") or ""))
    return latest


def _codegen_final_test_passed(sources: DashboardSources) -> bool:
    state_path = sources.data_dir / "code_gen_state.jsonl"
    result_path = sources.logs_dir / "codegen_final_test.json"
    try:
        state_lines = [
            line for line in state_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        state = json.loads(state_lines[-1])
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, IndexError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict) or not isinstance(result, dict):
        return False
    total = int(state.get("total_tasks") or 0)
    completed = int(state.get("completed_tasks") or 0)
    failed = int(state.get("failed_tasks") or 0)
    return total > 0 and completed >= total and failed == 0 and result.get("success") is True


def _load_codegen_final_test(sources: DashboardSources) -> dict[str, Any]:
    try:
        result = json.loads(
            (sources.logs_dir / "codegen_final_test.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def _codegen_duration_s(sources: DashboardSources) -> float | None:
    state_path = sources.data_dir / "code_gen_state.jsonl"
    try:
        states = [
            json.loads(line)
            for line in state_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        state = states[-1]
        started = datetime.fromisoformat(str(state["started_at"]).replace("Z", "+00:00"))
        updated = datetime.fromisoformat(str(state["last_updated"]).replace("Z", "+00:00"))
    except (OSError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    duration = (updated - started).total_seconds()
    return duration if duration >= 0 else None


def collect_pipeline(
    sources: DashboardSources,
    artifacts: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    rpg_data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    artifact_by_label = {artifact["label"]: artifact for artifact in artifacts}
    decoder_present = any((sources.data_dir / name).is_file() for name in _DECODER_ARTIFACTS)
    mode = "decoder" if decoder_present else "encoder" if rpg_data or runs else "unknown"
    specs = _DECODER_PIPELINE if mode == "decoder" else _ENCODER_PIPELINE
    latest_stages = _latest_stage_by_name(runs)
    codegen_final_test_passed = _codegen_final_test_passed(sources)
    codegen_duration_s = _codegen_duration_s(sources)
    pipeline: list[dict[str, Any]] = []

    for sequence, (step_id, label, artifact_label) in enumerate(specs, start=1):
        artifact = artifact_by_label.get(artifact_label, {})
        stage_info = latest_stages.get(step_id)
        if stage_info:
            stage, run_id = stage_info
            raw_status = stage.get("status") or "unknown"
            if (
                raw_status in {"not_started", "pending", "unknown"}
                and artifact.get("status") == "available"
            ):
                status = "completed"
                quality = "inferred"
            else:
                status = "completed" if raw_status == "success" else raw_status
                quality = "measured"
            duration_s = stage.get("duration_s")
            if step_id == "code_gen" and duration_s is None:
                duration_s = codegen_duration_s
            attempt = stage.get("attempt")
            error = stage.get("error")
            if (
                step_id == "code_gen"
                and codegen_final_test_passed
                and raw_status in {
                    "not_started", "pending", "running", "in_progress", "unknown",
                    "interrupted",
                }
            ):
                status = "completed"
                quality = "measured"
                error = None
        else:
            run_id = None
            duration_s = None
            if step_id == "code_gen":
                duration_s = codegen_duration_s
            attempt = None
            error = None
            if step_id == "dep_graph" and mode == "encoder":
                exists = isinstance(rpg_data.get("dep_graph"), dict)
            else:
                exists = artifact.get("status") == "available"
            status = "completed" if exists else "not_started"
            quality = "inferred"
            if step_id == "code_gen" and codegen_final_test_passed:
                status = "completed"
                quality = "measured"
        pipeline.append({
            "id": step_id,
            "label": label,
            "sequence": sequence,
            "status": status,
            "quality": quality,
            "run_id": run_id,
            "artifact": artifact_label,
            "duration_s": duration_s,
            "attempt": attempt,
            "error": error,
        })
    return mode, pipeline


def collect_tasks(sources: DashboardSources) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = sources.data_dir / "code_gen_state.jsonl"
    states, health = load_jsonl_records(state_path, source_name="code_gen_state")
    if not states:
        return {}, health
    state = states[-1]
    total = int(state.get("total_tasks") or 0)
    completed = int(state.get("completed_tasks") or 0)
    failed = int(state.get("failed_tasks") or 0)
    skipped = len(state.get("skipped_task_ids") or [])
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "pending": max(total - completed - failed - skipped, 0),
        "completion_percent": round(completed / total * 100, 1) if total else None,
        "current_batch_id": state.get("current_batch_id"),
        "initialized": state.get("initialized"),
        "started_at": state.get("started_at"),
        "last_updated": state.get("last_updated"),
    }, health


def _compact_rows(values: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        row = {key: value[key] for key in keys if value.get(key) not in (None, "", [], {})}
        if row:
            rows.append(row)
    return rows


def _impact_context(impact: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = impact.get("results") if isinstance(impact.get("results"), dict) else {}
    hits: list[dict[str, Any]] = []
    affected_files: set[str] = set()
    mapped_relations = 0
    errors = 0
    for node_id, value in results.items():
        row = value if isinstance(value, dict) else {}
        dep_nodes = [str(item) for item in row.get("dep_nodes") or []]
        files = [str(item) for item in row.get("affected_files") or []]
        affected_files.update(files)
        mapped_relations += len(dep_nodes)
        if row.get("error"):
            errors += 1
        summary = row.get("impact_summary") if isinstance(row.get("impact_summary"), dict) else {}
        hits.append({
            "node_id": str(node_id),
            "name": row.get("name"),
            "mapping_status": "mapped" if dep_nodes else "missing_mapping",
            "dep_nodes": dep_nodes,
            "affected_files": files,
            "callers": _compact_rows(row.get("callers"), ("node_id", "name", "type")),
            "callees": _compact_rows(row.get("callees"), ("node_id", "name", "type")),
            "inheritance": _compact_rows(row.get("inheritance"), ("node_id", "name", "type", "direction")),
            "imports": _compact_rows(row.get("imports"), ("node_id", "name", "type", "module")),
            "summary": {
                "callers": summary.get("total_callers", len(row.get("callers") or [])),
                "callees": summary.get("total_callees", len(row.get("callees") or [])),
                "inheritance": summary.get("total_inheritance", len(row.get("inheritance") or [])),
                "affected_files": summary.get("affected_file_count", len(files)),
            },
            "error": row.get("error"),
            "message": row.get("message"),
        })
    return hits, {
        "nodes": len(results),
        "mapped_code_relations": mapped_relations,
        "affected_files": len(affected_files),
        "errors": errors,
    }


def _decision_context(apply_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not apply_result:
        return []
    test_result = apply_result.get("test_result") if isinstance(apply_result.get("test_result"), dict) else {}
    backups = apply_result.get("backups") if isinstance(apply_result.get("backups"), dict) else {}
    rollback_path = (
        apply_result.get("rollback_path")
        or backups.get("rpg")
        or backups.get("dep_graph")
        or apply_result.get("rollback_command")
    )
    return [{
        "decision": "apply",
        "status": apply_result.get("type") or apply_result.get("status"),
        "confirmed": apply_result.get("confirmed"),
        "before_state": apply_result.get("before_state"),
        "applied_features": len(apply_result.get("applied_features") or []),
        "code_changes_planned": apply_result.get("code_changes_planned"),
        "dep_graph_refreshed": apply_result.get("dep_graph_refreshed"),
        "test_status": "passed" if test_result.get("passed") is True else "failed" if test_result.get("passed") is False else None,
        "rolled_back": apply_result.get("rolled_back"),
        "rollback_path": rollback_path,
        "backup_timestamp": apply_result.get("backup_timestamp"),
        "source": "rpg_edit_apply_result.json",
        "quality": "reported",
    }]


def collect_rpg_edit_context(
    sources: DashboardSources,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect current RPG-edit evidence without assigning stale artifacts to a run."""
    filenames = {
        "validate": "rpg_edit_validate.json",
        "locate": "rpg_edit_locate.json",
        "plan": "rpg_edit_plan.json",
        "impact": "rpg_edit_impact.json",
        "code": "rpg_edit_code_result.json",
        "apply": "rpg_edit_apply_result.json",
        "review": "rpg_edit_review_result.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    health: list[dict[str, Any]] = []
    for name, filename in filenames.items():
        value, source_health = load_json_object(
            sources.data_dir / filename,
            source_name=f"rpg_edit_{name}",
        )
        artifacts[name] = value
        health.append(source_health)

    plan = artifacts["plan"]
    locate = artifacts["locate"]
    impact = artifacts["impact"]
    code = artifacts["code"]
    apply_result = artifacts["apply"]
    review = artifacts["review"]
    validate = artifacts["validate"]

    locate_hits = _compact_rows(
        locate.get("results"),
        ("node_id", "name", "path", "meta_path", "feature_path", "score", "reason", "status"),
    )
    impact_hits, impact_summary = _impact_context(impact)
    retrievals: list[dict[str, Any]] = []
    if locate or locate_hits:
        retrievals.append({
            "kind": "locate",
            "query": locate.get("query"),
            "tool": "rpg_edit/locate",
            "hits": locate_hits,
            "summary": {"candidates": len(locate.get("results") or []), "retained": len(locate_hits)},
            "quality": "reported",
        })
    if impact or impact_hits:
        retrievals.append({
            "kind": "impact",
            "query": [hit["node_id"] for hit in impact_hits],
            "tool": "rpg_edit/impact",
            "hits": impact_hits,
            "summary": impact_summary,
            "quality": "reported",
        })

    verification: list[dict[str, Any]] = []
    if validate:
        verification.append({
            "name": "validate",
            "status": validate.get("type") or validate.get("status"),
            "detail": validate.get("message"),
            "quality": "reported",
        })
    if code:
        verification.append({
            "name": "code",
            "status": code.get("last_status") or code.get("success"),
            "detail": code.get("last_error"),
            "quality": "reported",
        })
    if apply_result:
        test_result = apply_result.get("test_result") if isinstance(apply_result.get("test_result"), dict) else {}
        verification.extend([
            {
                "name": "apply",
                "status": apply_result.get("type") or apply_result.get("status"),
                "detail": f"{len(apply_result.get('applied_features') or [])} applied features",
                "quality": "reported",
            },
            {
                "name": "test",
                "status": test_result.get("passed"),
                "detail": "rpg_edit apply test_result",
                "quality": "reported",
            },
        ])
    if review:
        verification.append({
            "name": "review",
            "status": review.get("success") if review.get("success") is not None else review.get("type"),
            "detail": review.get("reason"),
            "quality": "reported",
        })

    return {
        "available": any(bool(value) for value in artifacts.values()),
        "scope": "current_workspace",
        "plan": {
            "affected_nodes": [str(node_id) for node_id in plan.get("affected_nodes") or []],
            "feature_changes": _compact_rows(
                plan.get("feature_changes"),
                ("node_id", "action", "change", "name", "path"),
            ),
            "code_changes": _compact_rows(
                plan.get("code_changes"),
                ("file_path", "change_type", "action", "description", "reason"),
            ),
        },
        "retrievals": retrievals,
        "impact_summary": impact_summary,
        "code": {
            "success": code.get("success"),
            "last_status": code.get("last_status"),
            "last_error": code.get("last_error"),
            "commit_sha": code.get("commit_sha"),
            "files_modified": [str(path) for path in code.get("files_modified") or []],
            "iterations": len(code.get("iterations") or []),
        },
        "decisions": _decision_context(apply_result),
        "review": {
            "status": review.get("type") or review.get("status"),
            "success": review.get("success"),
            "iterations": len(review.get("iterations") or []),
            "suggestions": review.get("suggestions") or [],
            "reason": review.get("reason"),
        },
        "verification": verification,
    }, health


def _telemetry_breakdown(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value not in (None, ""):
            grouped[str(value)].append(record)
    return [
        {
            "name": name,
            "calls": len(rows),
            "total_duration_ms": sum(int(row.get("duration_ms") or 0) for row in rows),
            "average_duration_ms": round(
                sum(int(row.get("duration_ms") or 0) for row in rows) / len(rows),
                1,
            ),
        }
        for name, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def collect_telemetry(
    sources: DashboardSources,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate existing MCP and hook logs without copying sensitive params."""
    mcp_path = sources.logs_dir / "mcp_calls.jsonl"
    hook_path = sources.logs_dir / "hook_calls.jsonl"
    mcp_records, mcp_health = load_jsonl_records(mcp_path, source_name="mcp_calls")
    hook_records, hook_health = load_jsonl_records(hook_path, source_name="hook_calls")
    copilot_usage, copilot_health = collect_copilot_usage(sources.logs_dir / "copilot")

    mcp_duration = sum(int(record.get("duration_ms") or 0) for record in mcp_records)
    hook_duration = sum(int(record.get("duration_ms") or 0) for record in hook_records)
    telemetry = {
        "mcp": {
            "calls": len(mcp_records),
            "total_duration_ms": mcp_duration,
            "average_duration_ms": round(mcp_duration / len(mcp_records), 1) if mcp_records else None,
            "latest_at": mcp_records[-1].get("ts") if mcp_records else None,
            "tools": _telemetry_breakdown(mcp_records, "tool"),
        },
        "hooks": {
            "calls": len(hook_records),
            "total_duration_ms": hook_duration,
            "average_duration_ms": round(hook_duration / len(hook_records), 1) if hook_records else None,
            "latest_at": hook_records[-1].get("ts") if hook_records else None,
            "types": _telemetry_breakdown(hook_records, "hook"),
            "modes": _distribution(record.get("mode") for record in hook_records),
            "change_totals": {
                key: sum(int(record.get(key) or 0) for record in hook_records)
                for key in ("added", "modified", "deleted")
            },
            "latest_graph": {
                key: hook_records[-1].get(key) if hook_records else None
                for key in ("rpg_nodes", "dep_nodes", "dep_edges")
            },
        },
        "llm": {
            "copilot_cli": copilot_usage,
        },
    }
    return telemetry, [mcp_health, hook_health, copilot_health]


def collect_trends(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Group comparable runs by command for later trend rendering."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in reversed(runs):
        command = str(run.get("command") or "unknown")
        token_total = 0
        has_tokens = False
        for stage in run.get("stages", []):
            tokens = stage.get("tokens")
            if isinstance(tokens, dict) and isinstance(tokens.get("total"), (int, float)):
                token_total += tokens["total"]
                has_tokens = True
        grouped[command].append({
            "run_id": run.get("run_id"),
            "started_at": run.get("started_at"),
            "status": run.get("display_status"),
            "duration_s": run.get("duration_s"),
            "tokens_total": token_total if has_tokens else None,
            "metrics": run.get("metrics", {}),
        })
    return {"by_command": dict(grouped)}


_META_SUBJECT_RE = re.compile(r"^\[hook:([^\]@]+?)\s*@\s*([0-9a-fA-F]+)\]\s*(.*)$")


def _operation_command(value: Any) -> str:
    token = str(value or "").strip().split(maxsplit=1)[0]
    return token.lstrip("/").replace("-", "_").lower()


def collect_rpg_history(
    sources: DashboardSources,
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cheap RPG version index from the meta-git (one ``git log``, no file reads).

    Every operation commits ``rpg.json`` into the home-side meta-git, so the
    complete, git-compressed version history already exists.  This lists only
    lightweight metadata per version for the UI; the full ``rpg.json`` for any
    listed commit is read on demand (see ``rpg_version.py``), never embedded
    here, which keeps the snapshot small while retaining full history.
    """
    meta_root = sources.data_dir.parent
    try:
        rel = sources.rpg_file.relative_to(meta_root).as_posix()
    except ValueError:
        rel = "data/rpg.json"

    log = _run_git(
        meta_root,
        "log",
        "--max-count=200",
        "--format=%H%x1f%cI%x1f%s",
        "--",
        rel,
    )
    if not log:
        return [], _source_health("rpg_history", meta_root, "not_applicable")

    # Join with already-collected run metrics by source commit — no file reads.
    run_by_commit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        commit = metrics.get("new_commit")
        if commit:
            candidates = run_by_commit[str(commit)]
            if run not in candidates:
                candidates.append(run)

    versions: list[dict[str, Any]] = []
    for line in log.splitlines():
        parts = line.split("\x1f")
        if len(parts) < 3:
            continue
        sha, committed_at, subject = parts[0], parts[1], parts[2]
        match = _META_SUBJECT_RE.match(subject)
        if match:
            hook, source_commit, operation = match.group(1), match.group(2), match.group(3).strip()
        else:
            hook, source_commit, operation = None, None, subject.strip()
        candidates = [
            candidate
            for full_commit, commit_runs in run_by_commit.items()
            if source_commit
            and (full_commit.startswith(source_commit) or source_commit.startswith(full_commit))
            for candidate in commit_runs
        ]
        operation_command = _operation_command(operation)
        run = next(
            (
                candidate
                for candidate in candidates
                if _operation_command(candidate.get("command")) == operation_command
            ),
            None,
        )
        metrics = run.get("metrics") if run and isinstance(run.get("metrics"), dict) else {}
        versions.append({
            "commit": sha,
            "short_commit": sha[:8],
            "committed_at": committed_at,
            "operation": operation or hook or "commit",
            "hook": hook,
            "source_commit": source_commit,
            "source_short": source_commit[:8] if source_commit else None,
            "message": subject,
            "node_count": metrics.get("node_count"),
            "edge_count": metrics.get("edge_count"),
            "nodes_delta": metrics.get("nodes_delta"),
            "edges_delta": metrics.get("edges_delta"),
            "run_id": run.get("run_id") if run else None,
        })
    for index, version_info in enumerate(versions):
        previous = versions[index + 1]["commit"] if index + 1 < len(versions) else None
        version_info["previous_version_commit"] = previous
        version_info["previous_version_short"] = previous[:8] if previous else None
    return versions, _source_health("rpg_history", meta_root, "available", records=len(versions))


def collect_latest_rpg_change(
    sources: DashboardSources,
    versions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the newest two RPG blobs and compute their semantic change."""
    meta_root = sources.data_dir.parent
    if not versions:
        return {}, _source_health("rpg_latest_change", meta_root, "not_applicable")
    try:
        relative_path = sources.rpg_file.relative_to(meta_root).as_posix()
    except ValueError:
        relative_path = "data/rpg.json"

    latest = versions[0]
    commit = str(latest["commit"])
    parent = latest.get("previous_version_commit") or previous_rpg_version(
        meta_root,
        relative_path,
        commit,
    )
    current = read_rpg_version(meta_root, relative_path, commit)
    if current is None:
        return {}, _source_health(
            "rpg_latest_change",
            meta_root,
            "invalid",
            detail=f"cannot read {commit}:{relative_path}",
        )
    before = read_rpg_version(meta_root, relative_path, str(parent)) if parent else {}
    if parent and before is None:
        return {}, _source_health(
            "rpg_latest_change",
            meta_root,
            "partial",
            detail=f"cannot read {parent}:{relative_path}",
        )
    change = semantic_rpg_diff(
        before or {},
        current,
        commit=commit,
        parent_commit=str(parent) if parent else None,
    )
    change["committed_at"] = latest.get("committed_at")
    change["operation"] = latest.get("operation")
    change["source_commit"] = latest.get("source_commit")
    change["run_id"] = latest.get("run_id")
    return change, _source_health("rpg_latest_change", meta_root, "available", records=1)


def build_dashboard_snapshot(
    sources: DashboardSources | None = None,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the current dashboard snapshot without rendering HTML."""
    sources = sources or DashboardSources.defaults()
    events, event_health = load_jsonl_records(sources.run_events_file, source_name="run_events")
    runs = aggregate_runs(events)
    trajectory_runs, trajectory_health = collect_trajectory_runs(sources.data_dir / "trajectory")
    runs = merge_trajectory_runs(runs, trajectory_runs)
    current_run = runs[0] if runs else None
    workspace, git_health = collect_workspace(sources)
    rpg_data, rpg_health = load_json_object(sources.rpg_file, source_name="rpg")
    rpg = collect_rpg(rpg_data) if rpg_data else {}
    graph = collect_graph(rpg_data)
    for run in runs:
        run["changes"] = collect_run_changes(run, sources, rpg_data)
    artifacts = collect_artifacts(sources)
    mode, pipeline = collect_pipeline(sources, artifacts, runs, rpg_data)
    tasks, task_health = collect_tasks(sources)
    telemetry, telemetry_health = collect_telemetry(sources)
    telemetry["llm"]["association"] = associate_copilot_usage(
        runs,
        telemetry["llm"]["copilot_cli"],
    )
    rpg_edit, rpg_edit_health = collect_rpg_edit_context(sources)
    rpg_history, rpg_history_health = collect_rpg_history(sources, runs)
    rpg_latest_change, rpg_latest_change_health = collect_latest_rpg_change(sources, rpg_history)
    trends = collect_trends(runs)
    history_hook_records, _ = load_jsonl_records(
        sources.logs_dir / "hook_calls.jsonl",
        source_name="hook_calls",
    )
    history_mcp_records, _ = load_jsonl_records(
        sources.logs_dir / "mcp_calls.jsonl",
        source_name="mcp_calls",
    )
    history = collect_run_history(
        sources.logs_dir,
        runs,
        history_hook_records,
        history_mcp_records,
    )
    automation = collect_automation_activity(history)
    for run in runs:
        run["verification"], run["next_actions"] = collect_run_verification(run)
    verification = collect_workspace_verification(
        pipeline, rpg, tasks, _load_codegen_final_test(sources),
    )
    if current_run and current_run.get("display_status") in {
        "failed", "interrupted", "cancelled", "timed_out", "completed_with_warnings",
    }:
        run_actions = current_run.get("next_actions") or []
        retry_commands = {
            action.get("command")
            for action in run_actions
            if isinstance(action, dict) and action.get("command")
        }
        verification["next_actions"] = [
            *run_actions,
            *[
                action
                for action in verification["next_actions"]
                if action.get("command") not in retry_commands
            ],
        ]
    workspace["mode"] = mode
    if mode != "decoder" and task_health["status"] == "missing":
        task_health["status"] = "not_applicable"
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": generated_at or _iso_now(),
        "workspace": workspace,
        "current_state": {
            "run_id": current_run.get("run_id") if current_run else None,
            "command": current_run.get("command") if current_run else None,
            "status": current_run.get("display_status") if current_run else "not_started",
            "current_stage": next(
                (stage.get("name") for stage in current_run.get("stages", []) if stage.get("status") == "running"),
                None,
            ) if current_run else None,
            "pipeline_completed": sum(1 for step in pipeline if step["status"] == "completed"),
            "pipeline_total": len(pipeline),
            "pipeline_percent": round(
                sum(1 for step in pipeline if step["status"] == "completed") / len(pipeline) * 100,
                1,
            ) if pipeline else None,
        },
        "pipeline": pipeline,
        "rpg": rpg,
        "graph": graph,
        "tasks": tasks,
        "rpg_edit": rpg_edit,
        "rpg_history": rpg_history,
        "rpg_latest_change": rpg_latest_change,
        "artifacts": artifacts,
        "telemetry": telemetry,
        "verification": verification,
        "runs": runs,
        "history": history,
        "automation": automation,
        "trends": trends,
        "source_health": classify_source_expectations([
            event_health,
            rpg_health,
            task_health,
            git_health,
            *telemetry_health,
            *rpg_edit_health,
            rpg_history_health,
            rpg_latest_change_health,
            trajectory_health,
            history["source_health"],
        ], mode=mode, has_runs=bool(runs), has_rpg_edit=bool(rpg_edit.get("available"))),
    }
    sanitized = sanitize_snapshot(snapshot)
    assert_valid_snapshot(sanitized)
    return sanitized


def write_dashboard_snapshot(
    snapshot: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Atomically write a replaceable dashboard cache."""
    assert_valid_snapshot(snapshot)
    target = path or DASHBOARD_SNAPSHOT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target