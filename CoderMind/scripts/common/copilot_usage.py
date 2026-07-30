"""Extract non-sensitive usage facts from workspace-local Copilot CLI logs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


_TIMESTAMP = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+\[[^]]+\]\s*")
_MODEL = re.compile(r"Using model:\s*(?P<model>\S+)")
_RESPONSE_MARKER = "response (Request-ID"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.astimezone()


def _line_timestamp(line: str) -> str | None:
    match = _TIMESTAMP.match(line)
    return match.group("timestamp") if match else None


def _strip_prefix(line: str) -> str:
    return _TIMESTAMP.sub("", line)


def _response_block(lines: list[str], marker_index: int) -> dict[str, Any] | None:
    start_index = marker_index + 1
    while start_index < len(lines) and "{" not in _strip_prefix(lines[start_index]):
        start_index += 1
    if start_index >= len(lines):
        return None

    first = _strip_prefix(lines[start_index])
    payload = first[first.find("{"):] + "\n"
    payload += "\n".join(_strip_prefix(line) for line in lines[start_index + 1:])
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_copilot_log(path: Path) -> dict[str, Any]:
    """Parse one Copilot log without retaining prompt or response content."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"file": path.name, "status": "invalid", "error": str(exc), "calls": []}

    timestamps = [timestamp for line in lines if (timestamp := _line_timestamp(line))]
    current_model: str | None = None
    calls: list[dict[str, Any]] = []
    invalid_responses = 0
    for index, line in enumerate(lines):
        model_match = _MODEL.search(line)
        if model_match:
            current_model = model_match.group("model")
        if _RESPONSE_MARKER not in line:
            continue
        block = _response_block(lines, index)
        if not block:
            invalid_responses += 1
            continue
        usage = block.get("usage") if isinstance(block.get("usage"), dict) else None
        if not usage:
            continue
        prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        latency = block.get("latency_checkpoint") if isinstance(block.get("latency_checkpoint"), dict) else {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        calls.append({
            "timestamp": _line_timestamp(line),
            "provider": "copilot_cli",
            "model": block.get("model") or current_model or "unknown",
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": int(usage.get("total_tokens") or input_tokens + output_tokens),
                "cache_read": int(prompt_details.get("cached_tokens") or 0),
                "reasoning": int(completion_details.get("reasoning_tokens") or 0),
            },
            "duration_ms": int(latency.get("total_duration_ms") or 0) or None,
            "request_id": (
                line.split(_RESPONSE_MARKER, 1)[1].split(")", 1)[0].strip()
                if ")" in line.split(_RESPONSE_MARKER, 1)[1]
                else None
            ),
            "log_file": path.name,
            "quality": "measured",
        })

    return {
        "file": path.name,
        "status": "available" if not invalid_responses else "partial",
        "started_at": timestamps[0] if timestamps else None,
        "finished_at": timestamps[-1] if timestamps else None,
        "duration_s": (
            (_parse_timestamp(timestamps[-1]) - _parse_timestamp(timestamps[0])).total_seconds()
            if len(timestamps) >= 2 and _parse_timestamp(timestamps[0]) and _parse_timestamp(timestamps[-1])
            else None
        ),
        "models": sorted({call["model"] for call in calls}),
        "calls": calls,
        "invalid_responses": invalid_responses,
    }


def _normalized_tokens(call: dict[str, Any]) -> dict[str, int]:
    tokens = call.get("tokens") if isinstance(call.get("tokens"), dict) else {}
    input_tokens = int(tokens.get("input") or tokens.get("input_tokens") or 0)
    output_tokens = int(tokens.get("output") or tokens.get("output_tokens") or 0)
    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": int(tokens.get("total") or tokens.get("total_tokens") or input_tokens + output_tokens),
        "cache_read": int(tokens.get("cache_read") or tokens.get("cache_read_input_tokens") or 0),
        "reasoning": int(tokens.get("reasoning") or tokens.get("reasoning_tokens") or 0),
    }


def summarize_usage_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    token_keys = ("input", "output", "total", "cache_read", "reasoning")
    models: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "tokens": {key: 0 for key in token_keys}, "duration_ms": 0}
    )
    totals = {key: 0 for key in token_keys}
    duration_ms = 0
    for call in calls:
        model = str(call.get("model") or "unknown")
        models[model]["calls"] += 1
        call_duration = int(call.get("duration_ms") or 0)
        models[model]["duration_ms"] += call_duration
        duration_ms += call_duration
        tokens = _normalized_tokens(call)
        for key in token_keys:
            value = int(tokens.get(key) or 0)
            totals[key] += value
            models[model]["tokens"][key] += value
    return {
        "calls": len(calls),
        "tokens": totals,
        "duration_ms": duration_ms,
        "models": [
            {"name": name, **values}
            for name, values in sorted(models.items(), key=lambda item: (-item[1]["calls"], item[0]))
        ],
    }


def collect_copilot_usage(log_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect measured usage from logs redirected by CopilotSessionManager."""
    if not log_dir.is_dir():
        return {
            "provider": "copilot_cli",
            "quality": "missing",
            "sessions": [],
            "calls_detail": [],
            **summarize_usage_calls([]),
        }, {
            "source": "copilot_logs",
            "path": str(log_dir),
            "status": "missing",
            "records": 0,
            "invalid_records": 0,
        }

    sessions = [parse_copilot_log(path) for path in sorted(log_dir.glob("*.log"))]
    calls = [call for session in sessions for call in session["calls"]]
    invalid = sum(int(session.get("invalid_responses") or 0) for session in sessions)
    status = "partial" if invalid else "available"
    return {
        "provider": "copilot_cli",
        "quality": "measured",
        "sessions": [
            {
                key: session.get(key)
                for key in ("file", "status", "started_at", "finished_at", "duration_s", "models")
            }
            for session in sessions
        ],
        "calls_detail": calls,
        **summarize_usage_calls(calls),
    }, {
        "source": "copilot_logs",
        "path": str(log_dir),
        "status": status,
        "records": len(calls),
        "invalid_records": invalid,
        "files": len(sessions),
    }


def _window_contains(timestamp: datetime, started_at: Any, finished_at: Any) -> bool:
    start = _parse_timestamp(started_at)
    end = _parse_timestamp(finished_at)
    return bool(start and end and start <= timestamp <= end)


def _merge_invocations_and_usage(
    invocations: list[dict[str, Any]],
    usage_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge a CLI invocation event with usage parsed from its exact log file."""
    usage_by_log = {
        str(call.get("log_file")): call
        for call in usage_calls
        if call.get("log_file")
    }
    consumed_logs: set[str] = set()
    merged: list[dict[str, Any]] = []
    for invocation in invocations:
        log_file = str(invocation.get("log_file") or "")
        usage_call = usage_by_log.get(log_file) if log_file else None
        if usage_call is None:
            merged.append(invocation)
            continue
        consumed_logs.add(log_file)
        merged.append({
            **invocation,
            **usage_call,
            "invocation_event_id": invocation.get("event_id"),
            "association_quality": usage_call.get("association_quality") or "exact_log_file",
        })
    merged.extend(
        call
        for call in usage_calls
        if not call.get("log_file") or str(call.get("log_file")) not in consumed_logs
    )
    return merged


def associate_copilot_usage(runs: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    """Attach measured calls by time window; retain all unassociated calls."""
    associated = 0
    exact_associations = 0
    time_window_associations = 0
    unassociated: list[dict[str, Any]] = []

    exact_targets: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for run in runs:
        stages_by_id = {
            str(stage.get("stage_id")): stage
            for stage in run.get("stages", [])
            if stage.get("stage_id")
        }
        for invocation in run.get("telemetry", {}).get("llm_invocations", []):
            log_file = invocation.get("log_file")
            if not log_file:
                continue
            stage = stages_by_id.get(str(invocation.get("stage_id")))
            exact_targets[str(log_file)] = (run, stage)

    for call in usage.get("calls_detail", []):
        timestamp = _parse_timestamp(call.get("timestamp"))
        matched_run: dict[str, Any] | None = None
        matched_stage: dict[str, Any] | None = None
        association_quality: str | None = None
        exact_target = exact_targets.get(str(call.get("log_file")))
        if exact_target:
            matched_run, matched_stage = exact_target
            association_quality = "exact_log_file"
            exact_associations += 1
        elif timestamp:
            for run in runs:
                if not _window_contains(timestamp, run.get("started_at"), run.get("finished_at")):
                    continue
                matched_run = run
                candidates = [
                    stage for stage in run.get("stages", [])
                    if _window_contains(timestamp, stage.get("started_at"), stage.get("finished_at"))
                ]
                if candidates:
                    matched_stage = candidates[-1]
                association_quality = "inferred_time_window"
                time_window_associations += 1
                break
        if not matched_run:
            unassociated.append(call)
            continue

        run_calls = matched_run.setdefault("telemetry", {}).setdefault("llm_calls", [])
        linked = {**call, "association_quality": association_quality}
        run_calls.append(linked)
        if matched_stage is not None:
            matched_stage.setdefault("telemetry", {}).setdefault("llm_calls", []).append(linked)
        associated += 1

    for run in runs:
        llm_calls = run.get("telemetry", {}).get("llm_calls", [])
        invocations = run.get("telemetry", {}).get("llm_invocations", [])
        combined = _merge_invocations_and_usage(invocations, llm_calls)
        if combined:
            run["telemetry"]["llm"] = summarize_usage_calls(combined)
        for stage in run.get("stages", []):
            stage_calls = stage.get("telemetry", {}).get("llm_calls", [])
            stage_invocations = stage.get("telemetry", {}).get("llm_invocations", [])
            stage_combined = _merge_invocations_and_usage(stage_invocations, stage_calls)
            if stage_combined:
                stage["telemetry"]["llm"] = summarize_usage_calls(stage_combined)
    return {
        "associated_calls": associated,
        "exact_log_file": exact_associations,
        "inferred_time_window": time_window_associations,
        "unassociated_calls": len(unassociated),
        "unassociated_detail": unassociated,
        "quality": (
            "exact_log_file"
            if exact_associations and not time_window_associations
            else "inferred_time_window"
            if time_window_associations and not exact_associations
            else "mixed"
            if associated
            else "unassociated"
        ),
    }