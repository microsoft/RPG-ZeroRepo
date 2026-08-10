"""Small atomic JSON persistence helpers for replaceable pipeline artifacts."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON completely before atomically replacing ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            try:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(path.parent, flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
    finally:
        temporary.unlink(missing_ok=True)