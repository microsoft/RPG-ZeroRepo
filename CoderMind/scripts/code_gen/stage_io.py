#!/usr/bin/env python3
"""Per-stage result persistence for the codegen pipeline.

Each pipeline stage (``final_test``, ``smoke_test``, ``global_review``)
writes its outcome to a JSON sidecar under
``.cmind/logs/codegen_<name>.json`` so:

* ``global_review`` can load earlier stages' findings without re-running
  them.
* Users / debugging can ``cat`` the file to inspect a stage in isolation.

These helpers were lifted from ``scripts.run_batch`` Module 6b's
"Stage results persistence" block.  Internal to the codegen package;
no external API contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from common.paths import LOGS_DIR

logger = logging.getLogger(__name__)


def stage_path(name: str):
    """Return the absolute path of a stage's JSON sidecar."""
    return LOGS_DIR / f"codegen_{name}.json"


def save_stage_result(name: str, data: Dict[str, Any]) -> None:
    """Save a stage result to ``.cmind/logs/codegen_<name>.json``.

    Each pipeline stage (final_test, smoke_test, global_review) saves
    its output independently. Global review loads all of them as context.

    Uses :func:`common.rpg_io.atomic_write_rpg` so a killed codegen run
    can't leave a half-truncated sidecar that ``global_review`` would
    then try (and fail) to load.  ``default=str`` is forwarded through
    ``**dump_kwargs`` to preserve the original fall-back serialiser for
    non-JSON-native objects (e.g. ``Path``, datetimes).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    dest = stage_path(name)
    try:
        from common.rpg_io import atomic_write_rpg
        atomic_write_rpg(dest, data, indent=2, default=str)
        logger.info("Saved stage result: %s", dest)
    except Exception as exc:
        logger.debug("Failed to save stage result %s: %s", name, exc)


def load_stage_result(name: str) -> Optional[Dict[str, Any]]:
    """Load a stage result, or ``None`` if not found / unreadable."""
    src = stage_path(name)
    if not src.is_file():
        return None
    try:
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
