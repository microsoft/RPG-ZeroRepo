"""Centralized logging configuration for CoderMind scripts.

All scripts that produce non-trivial work should call
:func:`setup_file_logging` once in their ``main()`` so that logs are
captured to ``<workspace>/.cmind/logs/<name>.log`` for later inspection.

Design goals
------------
* **Idempotent** — calling the function multiple times in one process
  (e.g. when scripts import each other and each runs ``main``) does not
  produce duplicated log lines or duplicate file handlers.
* **Console-friendly** — does not change console handlers; scripts that
  output ``--json`` to stdout still work.  Console verbosity is each
  script's own decision; this helper only attaches a *file* handler.
* **Symlink-safe** — the log directory comes from ``common.paths``,
  which already resolves the workspace root correctly when
  ``.cmind/scripts`` is a symlink in dev workflows.
* **Non-blocking on read-only filesystems** — if ``LOGS_DIR`` cannot be
  created or written to (e.g. CI, container, sandbox), the helper logs
  one warning to stderr and returns ``None`` instead of raising; the
  caller's business logic is never blocked by a log-setup failure.

Typical usage
-------------
::

    from common.logging_setup import setup_file_logging

    def main():
        setup_file_logging("rpg_edit")          # → .cmind/logs/rpg_edit.log
        # … rest of script …

The single ``log_name`` argument is the file *stem*; the ``.log``
extension is added automatically.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Set

from .paths import LOGS_DIR

_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Track which absolute log-file paths already have a file handler attached
# in *this* process so repeated calls in the same Python interpreter (e.g.
# scripts importing each other) don't duplicate handlers.
_ATTACHED: Set[Path] = set()


def setup_file_logging(
    log_name: str,
    *,
    level: int = logging.DEBUG,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    logs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Attach a file handler that writes script logs under ``.cmind/logs/``.

    The root logger is reconfigured (only its level, never its existing
    handlers) so the new file handler actually receives records at
    ``level``.  Existing handlers (e.g. the console handler set by
    ``logging.basicConfig``) are left untouched.

    Args:
        log_name: Stem for the log file (``"rpg_edit"`` →
            ``.cmind/logs/rpg_edit.log``).
        level: Minimum level the file handler captures.  Defaults to
            ``DEBUG`` so verbose runs are inspectable after the fact.
        fmt: ``logging.Formatter`` format string.
        datefmt: ``logging.Formatter`` datefmt string.
        logs_dir: Override the destination directory.  Defaults to
            :data:`common.paths.LOGS_DIR`.  Useful only for tests.

    Returns:
        Absolute path to the log file that will receive records, or
        ``None`` if the helper could not attach a file handler (e.g.
        the destination is read-only).  In the failure case a single
        warning has already been printed to ``stderr`` — the caller
        does not need to handle the return value.
    """
    target_dir = logs_dir if logs_dir is not None else LOGS_DIR

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"[cmind logging_setup] could not create {target_dir}: {exc}; "
            "file logging disabled (console logs unaffected).",
            file=sys.stderr,
        )
        return None

    log_path = target_dir / f"{log_name}.log"
    # Use absolute() instead of resolve() so that workspace symlinks are
    # preserved (mirrors common.paths.WORKSPACE_ROOT logic).
    resolved = log_path.absolute()

    # Idempotent guard — never attach two file handlers for the same path.
    if resolved in _ATTACHED:
        return log_path

    try:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError as exc:
        print(
            f"[cmind logging_setup] could not open {log_path}: {exc}; "
            "file logging disabled (console logs unaffected).",
            file=sys.stderr,
        )
        return None

    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root_logger = logging.getLogger()
    # Allow records at the requested level through to handlers; if the
    # root level is stricter than what we want to capture, lower it.
    # Handlers retain their own level filtering, so existing console
    # handlers are unaffected.
    if root_logger.level == logging.WARNING or root_logger.level > level:
        root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    _ATTACHED.add(resolved)
    return log_path
