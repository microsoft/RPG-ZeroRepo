"""Build renderer-independent Run History trees from activity and legacy logs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.activity_events import (
    DEFAULT_RETENTION_BYTES,
    DEFAULT_RETENTION_DAYS,
    load_activity_events_with_health,
)


_FAILURE_STATUSES = frozenset({
    "failed", "error", "timed_out", "cancelled", "interrupted",
})
_SUCCESS_STATUSES = frozenset({"success", "completed", "passed", "ok"})


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _activity_spans(
    events: list[dict[str, Any]],
    *,
    stale_after_seconds: int = 120,
) -> list[dict[str, Any]]:
    by_span: dict[str, dict[str, Any]] = {}
    for event in events:
        span_id = event.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        current = by_span.setdefault(span_id, {})
        current.update(event)
        if event.get("event_type") == "span_started":
            current.setdefault("started_at", event.get("started_at") or event.get("timestamp"))
        elif event.get("event_type") == "span_finished":
            current["finished_at"] = event.get("finished_at") or event.get("timestamp")
            current["_finished"] = True

    spans: list[dict[str, Any]] = []
    detail_keys = (
        "artifact_key", "artifact_origin", "change_type", "size_bytes",
        "content_sha256", "provider", "model", "purpose", "tool", "mode",
        "call_id", "server_session_id", "client_context", "batch_id", "task_id", "task_type",
        "attempts_used", "file_path", "result_type", "script", "exit_code",
        "prev_ref", "previous_commit", "new_commit",
        "argument_count", "rpg_edit_session_id", "git_sha", "hook_type",
    )
    for span_id, raw in by_span.items():
        status = raw.get("status") or "running"
        quality = "measured"
        if not raw.get("_finished"):
            last_seen = _parse_timestamp(raw.get("timestamp"))
            age = (datetime.now(timezone.utc) - last_seen).total_seconds()
            if last_seen != datetime.min.replace(tzinfo=timezone.utc) and age > stale_after_seconds:
                status = "interrupted"
                quality = "derived"
        spans.append({
            "span_id": span_id,
            "trace_id": raw.get("trace_id"),
            "parent_span_id": raw.get("parent_span_id"),
            "kind": raw.get("kind") or "activity",
            "logical_key": raw.get("logical_key") or raw.get("name") or "activity",
            "name": raw.get("name") or raw.get("logical_key") or "Activity",
            "status": status,
            "started_at": raw.get("started_at") or raw.get("timestamp"),
            "finished_at": raw.get("finished_at"),
            "duration_ms": raw.get("duration_ms"),
            "trigger": raw.get("trigger"),
            "attempt": raw.get("attempt"),
            "sequence": raw.get("sequence"),
            "quality": quality,
            "source": "activity_v2",
            "run_id": raw.get("run_id"),
            "stage_id": raw.get("stage_id"),
            "call_id": raw.get("call_id"),
            "provider": raw.get("provider"),
            "model": raw.get("model"),
            "tool": raw.get("tool"),
            "mode": raw.get("mode"),
            "error": raw.get("error"),
            "metrics": raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {},
            "details": {key: raw[key] for key in detail_keys if raw.get(key) is not None},
            "children": [],
        })
    return spans


def _legacy_run_span(run: dict[str, Any]) -> dict[str, Any]:
    quality = "reported" if run.get("source") == "trajectory" else "measured"
    run_id = str(run.get("run_id") or "legacy-run")
    command = str(run.get("command") or "unknown")
    pipeline = "encoder" if command in {"encode", "update_rpg"} else "decoder"
    children = []
    for stage in run.get("stages") or []:
        duration_s = stage.get("duration_s")
        children.append({
            "span_id": f"legacy-stage:{stage.get('stage_id')}",
            "trace_id": f"legacy-trace:{run_id}",
            "parent_span_id": f"legacy-run:{run_id}",
            "kind": "workflow.stage",
            "logical_key": f"{pipeline}-{command.replace('_', '-')}-{str(stage.get('name') or 'stage').replace('_', '-')}",
            "name": stage.get("name") or "Stage",
            "status": stage.get("status") or "unknown",
            "started_at": stage.get("started_at"),
            "finished_at": stage.get("finished_at"),
            "duration_ms": round(float(duration_s) * 1000, 3) if duration_s is not None else None,
            "trigger": run.get("trigger"),
            "attempt": stage.get("attempt"),
            "sequence": stage.get("sequence"),
            "quality": quality,
            "source": stage.get("source") or run.get("source") or "run_events",
            "run_id": run_id,
            "stage_id": stage.get("stage_id"),
            "error": stage.get("error"),
            "metrics": stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {},
            "children": [],
        })
    duration_s = run.get("duration_s")
    return {
        "span_id": f"legacy-run:{run_id}",
        "trace_id": f"legacy-trace:{run_id}",
        "parent_span_id": None,
        "kind": "workflow",
        "logical_key": f"{pipeline}-{command.replace('_', '-')}",
        "name": command,
        "status": run.get("display_status") or run.get("status") or "unknown",
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_ms": round(float(duration_s) * 1000, 3) if duration_s is not None else None,
        "trigger": run.get("trigger"),
        "attempt": None,
        "quality": quality,
        "source": run.get("source") or "run_events",
        "run_id": run_id,
        "error": run.get("error"),
        "metrics": run.get("metrics") if isinstance(run.get("metrics"), dict) else {},
        "details": {},
        "children": children,
    }


def _workflow_fingerprint(span: dict[str, Any]) -> tuple[str, datetime] | None:
    started_at = _parse_timestamp(span.get("started_at"))
    if started_at == datetime.min.replace(tzinfo=timezone.utc):
        return None
    return str(span.get("logical_key") or ""), started_at


def _legacy_call_span(
    record: dict[str, Any],
    kind: str,
    *,
    parent_span_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    name = str(record.get("tool") or record.get("hook") or kind)
    duration = record.get("duration_ms")
    timestamp = record.get("ts") or record.get("timestamp")
    logical_prefix = "mcp" if kind == "tool.mcp" else "encoder-hooks"
    call_id = str(record.get("call_id") or f"legacy:{kind}:{timestamp}:{name}")
    return {
        "span_id": f"legacy-call:{call_id}",
        "trace_id": trace_id or f"legacy-trace:{record.get('run_id') or call_id}",
        "parent_span_id": parent_span_id,
        "kind": kind,
        "logical_key": f"{logical_prefix}-{name.replace('_', '-')}",
        "name": name,
        "status": "failed" if record.get("error") not in (None, False, "") else "success",
        "started_at": timestamp,
        "finished_at": timestamp,
        "duration_ms": duration,
        "trigger": "mcp" if kind == "tool.mcp" else "hook",
        "attempt": None,
        "quality": "reported",
        "source": "mcp_calls" if kind == "tool.mcp" else "hook_calls",
        "run_id": record.get("run_id"),
        "stage_id": record.get("stage_id"),
        "call_id": record.get("call_id"),
        "tool": record.get("tool"),
        "mode": record.get("mode"),
        "error": record.get("error"),
        "metrics": {
            key: record[key]
            for key in ("results", "total_nodes", "dep_nodes", "dep_edges", "rpg_nodes")
            if record.get(key) is not None
        },
        "details": {
            key: record[key]
            for key in ("tool", "mode", "call_id")
            if record.get(key) is not None
        },
        "children": [],
    }


def _tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(span["span_id"]): span for span in spans}
    roots: list[dict[str, Any]] = []
    for span in spans:
        parent = by_id.get(str(span.get("parent_span_id")))
        if parent is None:
            roots.append(span)
        else:
            parent["children"].append(span)

    def sort_children(span: dict[str, Any]) -> None:
        span["children"].sort(key=lambda item: (
            _parse_timestamp(item.get("started_at")),
            item.get("sequence") is None,
            int(item.get("sequence") or 0),
            str(item.get("span_id")),
        ))
        for child in span["children"]:
            sort_children(child)

    for root in roots:
        sort_children(root)
    roots.sort(key=lambda item: (_parse_timestamp(item.get("started_at")), str(item.get("span_id"))), reverse=True)
    return roots


def _annotate_recovered_attempts(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link failed history attempts to a later success for the same operation."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def visit(node: dict[str, Any]) -> None:
        kind = str(node.get("kind") or "")
        logical_key = str(node.get("logical_key") or "")
        if kind and logical_key:
            grouped[(kind, logical_key)].append(node)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                visit(child)

    for root in roots:
        visit(root)

    for attempts in grouped.values():
        attempts.sort(key=lambda item: (
            _parse_timestamp(item.get("started_at")),
            str(item.get("span_id") or ""),
        ))
        next_success: dict[str, Any] | None = None
        for attempt in reversed(attempts):
            status = str(attempt.get("status") or "").lower()
            if status in _SUCCESS_STATUSES:
                next_success = attempt
                continue
            if status not in _FAILURE_STATUSES or next_success is None:
                continue
            attempt_started = _parse_timestamp(attempt.get("started_at"))
            success_started = _parse_timestamp(next_success.get("started_at"))
            if success_started <= attempt_started:
                continue
            attempt["recovery"] = {
                "status": "recovered",
                "by_span_id": next_success.get("span_id"),
                "by_started_at": next_success.get("started_at"),
                "by_finished_at": next_success.get("finished_at"),
            }
            next_success.setdefault("recovered_attempts", []).append({
                "span_id": attempt.get("span_id"),
                "status": attempt.get("status"),
                "started_at": attempt.get("started_at"),
            })

    for attempts in grouped.values():
        for attempt in attempts:
            recovered = attempt.get("recovered_attempts")
            if isinstance(recovered, list):
                recovered.sort(key=lambda item: _parse_timestamp(item.get("started_at")))
    return roots


def _virtual_call_roots(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for span in spans:
        kind = span.get("kind")
        if span.get("parent_span_id") is not None or kind not in {"hook.operation", "tool.mcp"}:
            retained.append(span)
            continue
        if span.get("source") == "activity_v2":
            details = span.get("details") if isinstance(span.get("details"), dict) else {}
            group_identity = str(
                details.get("server_session_id")
                or span.get("trace_id")
                or span.get("span_id")
            )
        else:
            group_identity = str(span.get("started_at") or "legacy")[:10]
        grouped[(str(kind), group_identity)].append(span)

    for (kind, identity), children in grouped.items():
        is_mcp = kind == "tool.mcp"
        logical_key = "encoder-mcp" if is_mcp else "encoder-hooks"
        label = "MCP session" if is_mcp else "Hooks"
        digest = hashlib.sha256(f"{kind}:{identity}".encode("utf-8")).hexdigest()
        span_id = f"virtual_{digest}"
        starts = [child.get("started_at") for child in children if child.get("started_at")]
        finishes = [child.get("finished_at") for child in children if child.get("finished_at")]
        statuses = [str(child.get("status") or "unknown") for child in children]
        status = "failed" if any(value in {"failed", "error", "timed_out"} for value in statuses) else "success"
        for child in children:
            child["parent_span_id"] = span_id
        retained.extend(children)
        retained.append({
            "span_id": span_id,
            "trace_id": str(children[0].get("trace_id") or f"virtual_trace_{digest}"),
            "parent_span_id": None,
            "kind": "mcp.session" if is_mcp else "hook.workflow",
            "logical_key": logical_key,
            "name": label,
            "status": status,
            "started_at": min(starts, key=_parse_timestamp) if starts else None,
            "finished_at": max(finishes, key=_parse_timestamp) if finishes else None,
            "duration_ms": sum(float(child.get("duration_ms") or 0) for child in children),
            "trigger": "mcp" if is_mcp else "hook",
            "attempt": None,
            "quality": "derived",
            "source": "activity_grouping",
            "run_id": None,
            "error": None,
            "metrics": {"calls": len(children)},
            "details": {
                "server_session_id": identity,
                "client_context": next((
                    child.get("details", {}).get("client_context")
                    for child in children
                    if isinstance(child.get("details"), dict)
                    and child.get("details", {}).get("client_context")
                ), None),
            } if is_mcp else {},
            "children": [],
        })
    return retained


def _rollup_hook_workflows(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reflect detached hook work in the parent workflow's final outcome."""
    failure_statuses = {"failed", "error", "timed_out", "cancelled", "interrupted"}

    def descendants(node: dict[str, Any]) -> list[dict[str, Any]]:
        children = [child for child in node.get("children") or [] if isinstance(child, dict)]
        return [*children, *(item for child in children for item in descendants(child))]

    for root in roots:
        if root.get("kind") != "hook.workflow":
            continue
        nested = descendants(root)
        finishes = [
            item.get("finished_at")
            for item in nested
            if item.get("finished_at")
        ]
        original_finish = root.get("finished_at")
        if original_finish:
            finishes.append(original_finish)
        if finishes:
            effective_finish = max(finishes, key=_parse_timestamp)
            if _parse_timestamp(effective_finish) > _parse_timestamp(original_finish):
                root.setdefault("details", {})["dispatch_finished_at"] = original_finish
                root["finished_at"] = effective_finish
                started = _parse_timestamp(root.get("started_at"))
                finished = _parse_timestamp(effective_finish)
                if started != datetime.min.replace(tzinfo=timezone.utc):
                    root["duration_ms"] = round((finished - started).total_seconds() * 1000, 3)
                root["quality"] = "derived"
        failed = next(
            (item for item in nested if str(item.get("status") or "") in failure_statuses),
            None,
        )
        if failed is not None:
            root["status"] = "failed"
            root["error"] = failed.get("error") or {
                "type": "HookChildError",
                "message": f"{failed.get('name') or failed.get('kind')} failed",
            }
            root["quality"] = "derived"
        root.setdefault("metrics", {})["operations"] = sum(
            item.get("kind") == "hook.operation" for item in nested
        )
    return roots


def _virtual_rpg_edit_roots(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for span in spans:
        if (
            span.get("kind") == "command.script"
            and str(span.get("logical_key") or "").startswith("decoder-rpg-edit-")
            and span.get("parent_span_id") is None
        ):
            grouped[str(span.get("trace_id") or span.get("span_id"))].append(span)
        else:
            retained.append(span)
    phase_order = {name: index for index, name in enumerate((
        "validate", "locate", "save-plan", "impact", "code", "apply", "review",
    ))}
    for trace_id, children in grouped.items():
        children.sort(key=lambda child: (
            phase_order.get(str(child.get("logical_key") or "").removeprefix("decoder-rpg-edit-"), 99),
            _parse_timestamp(child.get("started_at")),
        ))
        digest = hashlib.sha256(f"rpg-edit:{trace_id}".encode("utf-8")).hexdigest()
        span_id = f"virtual_{digest}"
        for sequence, child in enumerate(children, start=1):
            child["parent_span_id"] = span_id
            child["sequence"] = sequence
        retained.extend(children)
        retained.append({
            "span_id": span_id,
            "trace_id": trace_id,
            "parent_span_id": None,
            "kind": "workflow",
            "logical_key": "decoder-rpg-edit",
            "name": "rpg_edit",
            "status": "failed" if any(str(child.get("status")) == "failed" for child in children) else "success",
            "started_at": children[0].get("started_at"),
            "finished_at": children[-1].get("finished_at"),
            "duration_ms": sum(float(child.get("duration_ms") or 0) for child in children),
            "trigger": "cmind-cli",
            "attempt": None,
            "quality": "measured",
            "source": "activity_grouping",
            "run_id": None,
            "error": None,
            "metrics": {"phases": len(children)},
            "details": {},
            "children": [],
        })
    return retained


def _collapse_script_wrappers(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for root in roots:
        root["children"] = _collapse_script_wrappers(root.get("children") or [])
        if root.get("kind") == "command.script":
            workflows = [child for child in root["children"] if child.get("kind") == "workflow"]
            evidence = [child for child in root["children"] if child.get("kind") != "workflow"]
            if len(workflows) == 1:
                child = workflows[0]
                child["parent_span_id"] = root.get("parent_span_id")
                child.setdefault("wrapper_span_id", root.get("span_id"))
                for item in evidence:
                    item["parent_span_id"] = child.get("span_id")
                child["children"].extend(evidence)
                child["children"].sort(key=lambda item: (
                    _parse_timestamp(item.get("started_at")),
                    item.get("sequence") is None,
                    int(item.get("sequence") or 0),
                    str(item.get("span_id")),
                ))
                collapsed.append(child)
                continue
        collapsed.append(root)
    return collapsed


def _attach_workflow_helpers(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helper_owners = {
        "script-summary-skeleton": "decoder-build-skeleton",
        "script-generate-viz": "decoder-build-data-flow",
        "script-plan": "decoder-plan",
        "script-feature-construct": "decoder-feature-construct",
    }
    retained: list[dict[str, Any]] = []
    helpers: list[dict[str, Any]] = []
    for root in roots:
        if root.get("logical_key") in helper_owners and root.get("kind") == "command.script":
            helpers.append(root)
        else:
            retained.append(root)

    for helper in helpers:
        owner_key = helper_owners[str(helper["logical_key"])]
        helper_started = _parse_timestamp(helper.get("started_at"))
        candidates = [
            root
            for root in retained
            if root.get("logical_key") == owner_key
            and _parse_timestamp(root.get("started_at")) <= helper_started
        ]
        if not candidates:
            retained.append(helper)
            continue
        owner = max(candidates, key=lambda root: _parse_timestamp(root.get("started_at")))
        helper["parent_span_id"] = owner.get("span_id")
        helper.setdefault("details", {})["grouped_as"] = "workflow_evidence"
        owner.setdefault("children", []).append(helper)
        owner["children"].sort(key=lambda item: (
            _parse_timestamp(item.get("started_at")),
            item.get("sequence") is None,
            int(item.get("sequence") or 0),
            str(item.get("span_id")),
        ))

    retained.sort(
        key=lambda item: (_parse_timestamp(item.get("started_at")), str(item.get("span_id"))),
        reverse=True,
    )
    return retained


def _count_nodes(spans: list[dict[str, Any]]) -> int:
    return sum(1 + _count_nodes(span.get("children") or []) for span in spans)


def collect_run_history(
    logs_dir: Path,
    runs: list[dict[str, Any]],
    hook_records: list[dict[str, Any]],
    mcp_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect exact v2 spans and compatible legacy history without duplicates."""
    activity_events, activity_health = load_activity_events_with_health(logs_dir / "activity")
    v2 = [span for span in _activity_spans(activity_events) if span.get("kind") != "report.snapshot"]
    represented_runs = {str(span.get("run_id")) for span in v2 if span.get("run_id")}
    represented_calls = {str(span.get("call_id")) for span in v2 if span.get("call_id")}
    represented_workflows = {
        fingerprint
        for span in v2
        if span.get("kind") == "workflow"
        if (fingerprint := _workflow_fingerprint(span)) is not None
    }
    legacy_runs = []
    for run in runs:
        legacy_span = _legacy_run_span(run)
        if str(run.get("run_id")) in represented_runs:
            continue
        if _workflow_fingerprint(legacy_span) in represented_workflows:
            continue
        legacy_runs.append(legacy_span)
    run_parents = {
        str(span.get("run_id")): (str(span.get("span_id")), str(span.get("trace_id")))
        for span in [*v2, *legacy_runs]
        if span.get("run_id") and span.get("kind") == "workflow"
    }
    legacy_hooks = [
        _legacy_call_span(
            record,
            "hook.operation",
            parent_span_id=run_parents.get(str(record.get("run_id")), (None, None))[0],
            trace_id=run_parents.get(str(record.get("run_id")), (None, None))[1],
        )
        for record in hook_records
        if not record.get("call_id") or str(record.get("call_id")) not in represented_calls
    ]
    legacy_mcp = [
        _legacy_call_span(
            record,
            "tool.mcp",
            parent_span_id=run_parents.get(str(record.get("run_id")), (None, None))[0],
            trace_id=run_parents.get(str(record.get("run_id")), (None, None))[1],
        )
        for record in mcp_records
        if not record.get("call_id") or str(record.get("call_id")) not in represented_calls
    ]
    spans = _virtual_rpg_edit_roots([*v2, *legacy_runs, *legacy_hooks, *legacy_mcp])
    roots = _annotate_recovered_attempts(
        _rollup_hook_workflows(
            _attach_workflow_helpers(
                _collapse_script_wrappers(_tree(_virtual_call_roots(spans)))
            )
        )
    )
    activity_root = logs_dir / "activity"
    bytes_used = sum(
        path.stat().st_size
        for path in activity_root.glob("*/*.jsonl")
        if path.is_file()
    ) if activity_root.is_dir() else 0
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "retention": {
            "days": DEFAULT_RETENTION_DAYS,
            "max_bytes": DEFAULT_RETENTION_BYTES,
            "bytes_used": bytes_used,
            "automatic": True,
        },
        "summary": {
            "root_count": len(roots),
            "activity_count": _count_nodes(roots),
            "exact_count": len(v2),
            "legacy_count": len(legacy_runs) + len(legacy_hooks) + len(legacy_mcp),
        },
        "source_health": activity_health,
        "roots": roots,
    }