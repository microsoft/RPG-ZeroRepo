"""Structured progress updates for supervised long-running workflows."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.json_io import atomic_write_json
from common.process_lock import LockHeldError, ProcessLock


PROGRESS_FILE_ENV = "CMIND_PROGRESS_FILE"


def configured_progress_path() -> Path | None:
    value = os.environ.get(PROGRESS_FILE_ENV, "").strip()
    return Path(value) if value else None


def read_progress(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def update_progress(path: Path | None = None, **values: Any) -> None:
    target = path or configured_progress_path()
    if target is None:
        return
    lock_path = target.with_name(f".{target.name}.lock")
    try:
        with ProcessLock(lock_path, blocking=True):
            current = read_progress(target) or {}
            current.update(values)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(target, current)
    except (LockHeldError, OSError):
        return