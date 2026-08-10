"""Cross-platform advisory locks for workspace-scoped pipeline processes."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LockHeldError(RuntimeError):
    def __init__(self, metadata: dict[str, Any]):
        super().__init__("another process owns the workspace lock")
        self.metadata = metadata


class ProcessLock:
    """Advisory file lock with owner metadata and automatic release."""

    def __init__(self, path: Path, *, blocking: bool = False):
        self.path = path
        self.blocking = blocking
        self._handle: Any = None
        self._metadata: dict[str, Any] = {}

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._handle = self.path.open("r+", encoding="utf-8")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write("{}")
            self._handle.flush()
        self._handle.seek(0)
        try:
            self._acquire()
        except OSError as exc:
            metadata = self._read_metadata()
            self._handle.close()
            self._handle = None
            raise LockHeldError(metadata) from exc
        self.update(
            pid=os.getpid(),
            status="running",
            stage="starting",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        return self

    def _acquire(self) -> None:
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
            msvcrt.locking(self._handle.fileno(), mode, 1)
            return
        import fcntl

        flags = fcntl.LOCK_EX
        if not self.blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(self._handle.fileno(), flags)

    def _release(self) -> None:
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def _read_metadata(self) -> dict[str, Any]:
        try:
            self._handle.seek(0)
            value = json.loads(self._handle.read())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def active_metadata(cls, path: Path) -> dict[str, Any] | None:
        """Return lock-owner metadata without modifying the lock file."""
        if not path.is_file():
            return None
        lock = cls(path)
        lock._handle = path.open("r+", encoding="utf-8")
        try:
            try:
                lock._acquire()
            except OSError:
                return lock._read_metadata()
            lock._release()
            return None
        finally:
            lock._handle.close()
            lock._handle = None

    def update(self, **values: Any) -> None:
        if self._handle is None:
            return
        self._metadata.update(values)
        self._metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._handle.seek(0)
        json.dump(self._metadata, self._handle, ensure_ascii=False)
        self._handle.truncate()
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        try:
            self.update(status="released")
            self._release()
        finally:
            self._handle.close()
            self._handle = None