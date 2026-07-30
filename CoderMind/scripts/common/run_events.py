"""Append-only run and stage lifecycle events for the dashboard."""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from common.paths import RUN_EVENTS_FILE

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
EVENTS_FILE = RUN_EVENTS_FILE

_append_lock = threading.Lock()
_sequence_lock = threading.Lock()
_sequence_counters: dict[str, int] = {}


@dataclass(frozen=True)
class EventContext:
    run_id: str
    command: str
    stage_id: str | None = None
    stage: str | None = None


_event_context: ContextVar[EventContext | None] = ContextVar("cmind_event_context", default=None)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


def new_run_id(command: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{command}-{stamp}-{secrets.token_hex(2)}"


def current_event_context() -> EventContext | None:
    return _event_context.get()


def event_context_environment() -> dict[str, str]:
    context = current_event_context()
    if context is None:
        return {}
    environment = {
        "CMIND_RUN_ID": context.run_id,
        "CMIND_COMMAND": context.command,
    }
    if context.stage_id:
        environment["CMIND_STAGE_ID"] = context.stage_id
    if context.stage:
        environment["CMIND_STAGE"] = context.stage
    return environment


def _next_sequence(run_id: str) -> int:
    with _sequence_lock:
        sequence = _sequence_counters.get(run_id, 0) + 1
        _sequence_counters[run_id] = sequence
        return sequence


def _base_event(event_type: str, run_id: str, command: str, timestamp: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": _new_id("evt"),
        "event_type": event_type,
        "timestamp": timestamp,
        "run_id": run_id,
        "command": command,
    }


def _append_event(event: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(event), ensure_ascii=False) + "\n").encode("utf-8")
    try:
        EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _append_lock:
            fd = os.open(EVENTS_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    remaining = remaining[written:]
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
    except OSError:
        pass  # telemetry must never break a command


def load_events(path: Path | None = None) -> list[dict[str, Any]]:
    """Load valid events, ignoring an incomplete final line after a crash."""
    source = path or EVENTS_FILE
    if not source.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    except OSError:
        return []
    return events


class RunEvent:
    """Mutable result handle yielded by :func:`record_run`."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "success"
        self.metrics: dict[str, Any] = {}
        self.error: dict[str, Any] | None = None

    def note(self, **metrics: Any) -> None:
        self.metrics.update(metrics)


class StageEvent:
    """Mutable result handle yielded by :func:`record_stage`."""

    def __init__(
        self,
        *,
        run_id: str,
        command: str,
        stage_id: str,
        sequence: int,
        stage: str,
        phase: str | None,
        attempt: int,
        started_at: str,
    ) -> None:
        self.run_id = run_id
        self.command = command
        self.stage_id = stage_id
        self.sequence = sequence
        self.stage = stage
        self.phase = phase
        self.attempt = attempt
        self.started_at = started_at
        self.status = "success"
        self.metrics: dict[str, Any] = {}
        self.error: dict[str, Any] | None = None

    def note(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def progress(self, **metrics: Any) -> None:
        event = _stage_event("stage_progress", self, _iso_now(), status="running")
        if metrics:
            event["metrics"] = metrics
        _append_event(event)


def _stage_event(
    event_type: str,
    handle: StageEvent,
    timestamp: str,
    *,
    status: str,
) -> dict[str, Any]:
    event = _base_event(event_type, handle.run_id, handle.command, timestamp)
    event.update(
        {
            "stage_id": handle.stage_id,
            "sequence": handle.sequence,
            "stage": handle.stage,
            "phase": handle.phase,
            "attempt": handle.attempt,
            "status": status,
            "started_at": handle.started_at,
        }
    )
    return event


@contextmanager
def record_run(
    command: str,
    *,
    run_id: str | None = None,
    trigger: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[RunEvent]:
    """Record run start and finish events around a command execution."""
    resolved_run_id = run_id or new_run_id(command)
    started_at = _iso_now()
    start_perf = time.perf_counter()
    handle = RunEvent(resolved_run_id)

    started = _base_event("run_started", resolved_run_id, command, started_at)
    started.update({"status": "running", "started_at": started_at})
    if trigger:
        started["trigger"] = trigger
    if metadata:
        started["metadata"] = dict(metadata)
    _append_event(started)
    context_token = _event_context.set(EventContext(resolved_run_id, command))

    try:
        yield handle
    finally:
        exc = sys.exc_info()[1]
        if exc is not None and handle.status == "success":
            handle.status = "failed"
            handle.error = {"type": type(exc).__name__, "message": str(exc)}
        finished_at = _iso_now()
        finished = _base_event("run_finished", resolved_run_id, command, finished_at)
        finished.update(
            {
                "status": handle.status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_s": round(time.perf_counter() - start_perf, 3),
                "error": handle.error,
            }
        )
        if handle.metrics:
            finished["metrics"] = handle.metrics
        _append_event(finished)
        with _sequence_lock:
            _sequence_counters.pop(resolved_run_id, None)
        _event_context.reset(context_token)


@contextmanager
def record_stage(
    run_id: str,
    command: str,
    stage: str,
    phase: str | None = None,
    *,
    sequence: int | None = None,
    attempt: int = 1,
) -> Iterator[StageEvent]:
    """Record stage start immediately and append its final result on exit."""
    resolved_sequence = sequence or _next_sequence(run_id)
    started_at = _iso_now()
    handle = StageEvent(
        run_id=run_id,
        command=command,
        stage_id=_new_id("stage"),
        sequence=resolved_sequence,
        stage=stage,
        phase=phase,
        attempt=attempt,
        started_at=started_at,
    )
    _append_event(_stage_event("stage_started", handle, started_at, status="running"))
    start_perf = time.perf_counter()
    context_token = _event_context.set(EventContext(run_id, command, handle.stage_id, stage))

    try:
        yield handle
    finally:
        exc = sys.exc_info()[1]
        if exc is not None and handle.status == "success":
            handle.status = "failed"
            handle.error = {"type": type(exc).__name__, "message": str(exc)}
        finished_at = _iso_now()
        finished = _stage_event("stage_finished", handle, finished_at, status=handle.status)
        finished.update(
            {
                "finished_at": finished_at,
                "duration_s": round(time.perf_counter() - start_perf, 3),
                "error": handle.error,
            }
        )
        if handle.metrics:
            finished["metrics"] = handle.metrics
        _append_event(finished)
        _event_context.reset(context_token)


def enrich_stage(
    run_id: str,
    command: str,
    stage_id: str,
    **enrichment: Any,
) -> None:
    """Append token, model, call-count, or cost data discovered later."""
    event = _base_event("stage_enriched", run_id, command, _iso_now())
    event["stage_id"] = stage_id
    event.update({key: value for key, value in enrichment.items() if value is not None})
    _append_event(event)


def record_llm_call(
    *,
    provider: str,
    model: str | None = None,
    purpose: str | None = None,
    success: bool,
    duration_s: float | None = None,
    tokens: Mapping[str, int] | None = None,
    token_status: str | None = None,
    log_file: str | None = None,
) -> str | None:
    """Append one LLM invocation linked to the active run/stage context."""
    context = current_event_context()
    if context is None:
        return None
    timestamp = _iso_now()
    event = _base_event("llm_call", context.run_id, context.command, timestamp)
    event.update({
        "provider": provider,
        "model": model,
        "purpose": purpose,
        "success": success,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "tokens": dict(tokens) if tokens is not None else None,
        "token_status": token_status,
        "log_file": Path(log_file).name if log_file else None,
    })
    if context.stage_id:
        event["stage_id"] = context.stage_id
    if context.stage:
        event["stage"] = context.stage
    event = {key: value for key, value in event.items() if value is not None}
    _append_event(event)
    return str(event["event_id"])