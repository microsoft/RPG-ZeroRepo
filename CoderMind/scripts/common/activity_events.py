"""Conflict-free activity v2 events for workspace run history."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from common.paths import ACTIVITY_LOG_DIR, WORKSPACE_ROOT


SCHEMA_VERSION = 2
DEFAULT_RETENTION_DAYS = 90
DEFAULT_RETENTION_BYTES = 100 * 1024 * 1024

_RESERVED_FIELDS = {
    "schema_version", "event_id", "event_type", "timestamp", "monotonic_ns",
    "workspace_instance_id", "writer_id", "writer_sequence", "trace_id",
    "span_id", "parent_span_id", "kind", "name", "logical_key", "status",
}
_SENSITIVE_FIELDS = {
    "prompt", "response", "raw_response", "parsed_result", "stdout", "stderr",
    "content", "api_key", "apikey", "authorization", "access_token",
    "refresh_token", "client_secret", "password", "params",
}

_TRACE_ID_ENV = "CMIND_TRACE_ID"
_PARENT_SPAN_ID_ENV = "CMIND_PARENT_SPAN_ID"
_WORKSPACE_ID_ENV = "CMIND_WORKSPACE_INSTANCE_ID"
_WORKSPACE_ID_FILE_ENV = "CMIND_WORKSPACE_INSTANCE_FILE"

_RUN_BATCH_MODES = (
    ("--next", "next"),
    ("--loop", "loop"),
    ("--resume", "resume"),
    ("--retry", "retry"),
    ("--batch-id", "batch-id"),
    ("--final-test", "final-test"),
    ("--smoke-test", "smoke-test"),
    ("--global-review", "global-review"),
    ("--prune-failed", "prune-failed"),
)


def script_activity_mode(script: str, arguments: list[str]) -> str | None:
    """Return a non-sensitive execution mode for supported script commands."""
    script_name = Path(script).name
    if script_name == "smoke_test.py" and "--advisory" in arguments:
        return "advisory"
    if script_name != "run_batch.py":
        return None
    argument_set = set(arguments)
    return next((mode for flag, mode in _RUN_BATCH_MODES if flag in argument_set), None)


def new_id(prefix: str) -> str:
    """Return a non-truncated 128-bit execution identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


def workspace_instance_id(workspace_root: Path | None = None) -> str:
    """Return a persistent random 128-bit identity for one workspace instance."""
    configured = os.environ.get(_WORKSPACE_ID_ENV)
    if configured:
        return configured
    configured_file = os.environ.get(_WORKSPACE_ID_FILE_ENV)
    if configured_file:
        identity_file = Path(configured_file).expanduser()
    elif workspace_root is not None:
        try:
            from cmind_cli import _storage
            identity_file = _storage.home_workspace_dir(Path(workspace_root)) / ".workspace-instance-id"
        except (ImportError, OSError):
            identity_file = Path(workspace_root).resolve() / ".cmind" / ".workspace-instance-id"
    else:
        identity_file = ACTIVITY_LOG_DIR.parent.parent / ".workspace-instance-id"
    try:
        value = identity_file.read_text(encoding="utf-8").strip()
        if value.startswith("ws_") and len(value) == 35:
            int(value[3:], 16)
            return value
    except (OSError, ValueError):
        pass
    candidate = new_id("ws")
    try:
        identity_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(identity_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (candidate + "\n").encode("ascii"))
        finally:
            os.close(descriptor)
        return candidate
    except FileExistsError:
        try:
            existing = identity_file.read_text(encoding="utf-8").strip()
            if existing.startswith("ws_") and len(existing) == 35:
                int(existing[3:], 16)
                return existing
        except (OSError, ValueError):
            pass
    except OSError:
        pass
    canonical = str((workspace_root or WORKSPACE_ROOT).resolve())
    return f"ws_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_FIELDS
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return value[:4000] + ("...[truncated]" if len(value) > 4000 else "")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:4000]


@dataclass(frozen=True)
class ActivityContext:
    trace_id: str
    span_id: str


_activity_context: ContextVar[ActivityContext | None] = ContextVar(
    "cmind_activity_context",
    default=None,
)


def current_activity_context() -> ActivityContext | None:
    return _activity_context.get()


def activity_environment() -> dict[str, str]:
    """Return trace context suitable for child-process environment overlays."""
    context = current_activity_context()
    environment = {_WORKSPACE_ID_ENV: workspace_instance_id()}
    if context is not None:
        environment[_TRACE_ID_ENV] = context.trace_id
        environment[_PARENT_SPAN_ID_ENV] = context.span_id
    return environment


class ActivityWriter:
    """Append activity events to a file owned by this writer instance."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        writer_id: str | None = None,
        workspace_id: str | None = None,
        auto_prune: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else ACTIVITY_LOG_DIR
        self.writer_id = writer_id or new_id("wrt")
        self.workspace_id = workspace_id or workspace_instance_id()
        self._sequence = 0
        self._lock = threading.Lock()
        self._pid = os.getpid()
        if auto_prune:
            maybe_prune_activity_logs(self.root)

    def append(
        self,
        event_type: str,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        kind: str,
        name: str,
        logical_key: str,
        status: str | None = None,
        timestamp: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one complete JSON line and return the recorded event."""
        recorded_at = timestamp or _utc_now()
        day = recorded_at[:10]
        with self._lock:
            current_pid = os.getpid()
            if current_pid != self._pid:
                self.writer_id = new_id("wrt")
                self._sequence = 0
                self._pid = current_pid
            self._sequence += 1
            sequence = self._sequence
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": f"evt_{self.writer_id.removeprefix('wrt_')}_{sequence:016x}",
                "event_type": event_type,
                "timestamp": recorded_at,
                "monotonic_ns": time.monotonic_ns(),
                "workspace_instance_id": self.workspace_id,
                "writer_id": self.writer_id,
                "writer_sequence": sequence,
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "kind": kind,
                "name": name,
                "logical_key": logical_key,
            }
            if status is not None:
                event["status"] = status
            if fields:
                event.update({
                    str(key): _safe_value(value)
                    for key, value in fields.items()
                    if value is not None
                    and str(key) not in _RESERVED_FIELDS
                    and str(key).lower() not in _SENSITIVE_FIELDS
                })

            path = self.root / day / f"{self.writer_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                remaining = memoryview(payload)
                while remaining:
                    remaining = remaining[os.write(descriptor, remaining):]
            finally:
                os.close(descriptor)
        return event


_default_writer: ActivityWriter | None = None
_default_writer_lock = threading.Lock()


def default_writer() -> ActivityWriter:
    global _default_writer
    with _default_writer_lock:
        if _default_writer is None:
            _default_writer = ActivityWriter()
        return _default_writer


class ActivitySpan:
    def __init__(self, trace_id: str, span_id: str) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.status = "success"
        self.fields: dict[str, Any] = {}
        self.error: dict[str, str] | None = None

    def note(self, **fields: Any) -> None:
        self.fields.update(fields)


@contextmanager
def record_activity(
    kind: str,
    name: str,
    *,
    logical_key: str | None = None,
    trigger: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    writer: ActivityWriter | None = None,
    heartbeat_interval_s: float | None = None,
) -> Iterator[ActivitySpan]:
    """Record a start/finish span while propagating trace context to children."""
    active = current_activity_context()
    resolved_trace_id = (
        trace_id
        or (active.trace_id if active else None)
        or os.environ.get(_TRACE_ID_ENV)
        or new_id("trc")
    )
    resolved_parent_id = (
        parent_span_id
        or (active.span_id if active else None)
        or os.environ.get(_PARENT_SPAN_ID_ENV)
    )
    resolved_span_id = new_id("spn")
    resolved_key = logical_key or name
    event_writer = writer or default_writer()
    started_at = _utc_now()
    started_perf = time.perf_counter()
    span = ActivitySpan(resolved_trace_id, resolved_span_id)
    event_writer.append(
        "span_started",
        trace_id=resolved_trace_id,
        span_id=resolved_span_id,
        parent_span_id=resolved_parent_id,
        kind=kind,
        name=name,
        logical_key=resolved_key,
        status="running",
        timestamp=started_at,
        fields={"started_at": started_at, "trigger": trigger},
    )
    token = _activity_context.set(ActivityContext(resolved_trace_id, resolved_span_id))
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if heartbeat_interval_s is not None and heartbeat_interval_s > 0:
        def heartbeat() -> None:
            while not heartbeat_stop.wait(heartbeat_interval_s):
                event_writer.append(
                    "span_heartbeat",
                    trace_id=resolved_trace_id,
                    span_id=resolved_span_id,
                    parent_span_id=resolved_parent_id,
                    kind=kind,
                    name=name,
                    logical_key=resolved_key,
                    status="running",
                    fields={"started_at": started_at},
                )
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
    try:
        yield span
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=max(heartbeat_interval_s or 0, 0.1) + 0.1)
        exception = sys.exc_info()[1]
        if exception is not None and span.status == "success":
            span.status = "failed"
            span.error = {"type": type(exception).__name__, "message": str(exception)}
        finished_at = _utc_now()
        event_writer.append(
            "span_finished",
            trace_id=resolved_trace_id,
            span_id=resolved_span_id,
            parent_span_id=resolved_parent_id,
            kind=kind,
            name=name,
            logical_key=resolved_key,
            status=span.status,
            timestamp=finished_at,
            fields={
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": round((time.perf_counter() - started_perf) * 1000, 3),
                "error": span.error,
                **span.fields,
            },
        )
        _activity_context.reset(token)


def record_completed_activity(
    kind: str,
    name: str,
    *,
    logical_key: str | None = None,
    status: str = "success",
    duration_ms: float | None = None,
    trigger: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    fields: Mapping[str, Any] | None = None,
    writer: ActivityWriter | None = None,
) -> str:
    """Record a completed external call when only its final result is available."""
    active = current_activity_context()
    resolved_trace_id = (
        trace_id
        or (active.trace_id if active else None)
        or os.environ.get(_TRACE_ID_ENV)
        or new_id("trc")
    )
    resolved_parent_id = (
        parent_span_id
        or (active.span_id if active else None)
        or os.environ.get(_PARENT_SPAN_ID_ENV)
    )
    span_id = new_id("spn")
    finished = datetime.now(timezone.utc)
    started = finished - timedelta(milliseconds=max(duration_ms or 0, 0))
    started_at = started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    finished_at = finished.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    event_writer = writer or default_writer()
    common = {
        "trace_id": resolved_trace_id,
        "span_id": span_id,
        "parent_span_id": resolved_parent_id,
        "kind": kind,
        "name": name,
        "logical_key": logical_key or name,
    }
    event_writer.append(
        "span_started",
        **common,
        status="running",
        timestamp=started_at,
        fields={"started_at": started_at, "trigger": trigger},
    )
    event_writer.append(
        "span_finished",
        **common,
        status=status,
        timestamp=finished_at,
        fields={
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
            "trigger": trigger,
            **(dict(fields) if fields else {}),
        },
    )
    return span_id


def load_activity_events_with_health(
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load events and report malformed lines or duplicate global event IDs."""
    source = Path(root) if root is not None else ACTIVITY_LOG_DIR
    events: list[dict[str, Any]] = []
    invalid_records = 0
    duplicate_event_ids: list[str] = []
    seen_event_ids: set[str] = set()
    files = 0
    if not source.is_dir():
        return events, {
            "source": "activity",
            "path": str(source),
            "status": "missing",
            "records": 0,
            "invalid_records": 0,
            "duplicate_event_ids": [],
            "files": 0,
        }
    for path in sorted(source.glob("*/*.jsonl")):
        files += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            invalid_records += 1
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue
            if isinstance(value, dict):
                event_id = value.get("event_id")
                if not isinstance(event_id, str) or not event_id:
                    invalid_records += 1
                    continue
                if event_id in seen_event_ids:
                    duplicate_event_ids.append(event_id)
                    continue
                seen_event_ids.add(event_id)
                events.append(value)
            else:
                invalid_records += 1
    status = "invalid" if duplicate_event_ids else "partial" if invalid_records else "available"
    return sorted(events, key=lambda item: (str(item.get("timestamp") or ""), str(item.get("event_id") or ""))), {
        "source": "activity",
        "path": str(source),
        "status": status,
        "records": len(events),
        "invalid_records": invalid_records,
        "duplicate_event_ids": sorted(set(duplicate_event_ids)),
        "files": files,
    }


def load_activity_events(root: Path | None = None) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only valid, globally unique events."""
    return load_activity_events_with_health(root)[0]


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def prune_activity_logs(
    root: Path | None = None,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    max_bytes: int = DEFAULT_RETENTION_BYTES,
    today: date | None = None,
) -> dict[str, Any]:
    """Prune closed daily shards by age and size without touching today's writers."""
    source = Path(root) if root is not None else ACTIVITY_LOG_DIR
    current_day = today or datetime.now(timezone.utc).date()
    source.mkdir(parents=True, exist_ok=True)
    lock_path = source / ".retention.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return {
            "status": "skipped_locked",
            "retention_days": retention_days,
            "max_bytes": max_bytes,
        }

    os.close(descriptor)
    try:
        dated: list[tuple[date, Path]] = []
        for path in source.iterdir():
            if not path.is_dir():
                continue
            try:
                parsed = date.fromisoformat(path.name)
            except ValueError:
                continue
            dated.append((parsed, path))
        dated.sort(key=lambda item: item[0])

        bytes_before = sum(_directory_size(path) for _, path in dated)
        removed_days: list[str] = []
        cutoff = current_day - timedelta(days=max(retention_days, 0))
        for day, path in list(dated):
            if day >= cutoff or day == current_day:
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed_days.append(day.isoformat())

        remaining = [(day, path) for day, path in dated if path.is_dir()]
        bytes_after_age = sum(_directory_size(path) for _, path in remaining)
        bytes_after = bytes_after_age
        for day, path in remaining:
            if bytes_after <= max(max_bytes, 0) or day == current_day:
                continue
            size = _directory_size(path)
            shutil.rmtree(path, ignore_errors=True)
            bytes_after -= size
            removed_days.append(day.isoformat())

        remaining_days = sorted(
            path.name
            for path in source.iterdir()
            if path.is_dir()
            and path.name != current_day.isoformat()
            and _is_iso_date(path.name)
        )
        today_path = source / current_day.isoformat()
        return {
            "status": "completed",
            "retention_days": retention_days,
            "max_bytes": max_bytes,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "removed_days": sorted(set(removed_days)),
            "oldest_day": remaining_days[0] if remaining_days else (current_day.isoformat() if today_path.is_dir() else None),
            "over_limit": bytes_after > max_bytes,
        }
    finally:
        lock_path.unlink(missing_ok=True)


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def maybe_prune_activity_logs(
    root: Path | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Run retention at most once per UTC day for this workspace."""
    source = Path(root) if root is not None else ACTIVITY_LOG_DIR
    current_day = today or datetime.now(timezone.utc).date()
    marker = source / ".retention-last-run"
    try:
        if marker.read_text(encoding="utf-8").strip() == current_day.isoformat():
            return {"status": "skipped_today"}
    except OSError:
        pass
    result = prune_activity_logs(source, today=current_day)
    if result.get("status") == "completed":
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(current_day.isoformat() + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    return result


def artifact_inventory(roots: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Return lightweight file fingerprints for managed artifact roots."""
    inventory: dict[str, dict[str, Any]] = {}
    for location, root in roots.items():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name.startswith(".") or path.suffix == ".tmp":
                continue
            try:
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
            except OSError:
                continue
            inventory[f"{location}/{relative}"] = {
                "path": path,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
    return inventory


def record_artifact_changes(
    before: Mapping[str, Mapping[str, Any]],
    roots: Mapping[str, Path],
    *,
    origin: str,
    writer: ActivityWriter | None = None,
) -> list[str]:
    """Record created, modified, and deleted managed artifacts as child spans."""
    after = artifact_inventory(roots)
    changed: list[str] = []
    for artifact_key in sorted(set(before) | set(after)):
        previous = before.get(artifact_key)
        current = after.get(artifact_key)
        if previous and current and (
            previous.get("size_bytes"), previous.get("modified_ns")
        ) == (
            current.get("size_bytes"), current.get("modified_ns")
        ):
            continue
        change_type = "created" if previous is None else "deleted" if current is None else "modified"
        sha256 = None
        if current is not None:
            try:
                digest = hashlib.sha256()
                with Path(current["path"]).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                sha256 = digest.hexdigest()
            except OSError:
                sha256 = None
        record_completed_activity(
            "artifact.write",
            Path(artifact_key).name,
            logical_key=f"artifact-{artifact_key.replace('/', '-')}",
            status="success",
            fields={
                "artifact_key": artifact_key,
                "artifact_origin": origin,
                "change_type": change_type,
                "size_bytes": current.get("size_bytes") if current else None,
                "content_sha256": sha256,
            },
            writer=writer,
        )
        changed.append(artifact_key)
    return changed