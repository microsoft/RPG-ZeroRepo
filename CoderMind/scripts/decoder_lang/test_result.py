"""Parser-agnostic test-execution result types.

The decoder used to talk to pytest directly through
:func:`code_gen.test_runner.run_pytest`. Phase 4 generalises this to
``run_tests(backend, env, ...)`` returning :class:`TestRunResult`, and
each backend supplies its own parser that maps native test-tool
output (pytest / ``go test`` / ``cargo test`` / ...) into this shape.

Defined here (not in :mod:`code_gen`) so backends can return the type
without an import cycle through the decoder package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Status that downstream callers (post_verify / global_review /
# final_validation) branch on. ``skipped`` is reserved for the
# toolchain-unavailable case (non-Python backends only) so the
# caller can mark the verification step as a non-fatal WARN.
TestRunStatus = Literal["passed", "failed", "errored", "skipped"]


@dataclass(frozen=True)
class TestFailure:
    """One failing test case extracted from the native test tool output."""

    test_id: str                       # e.g. "tests/test_foo.py::test_bar"
    short_message: str                 # one-line summary for LLM context
    long_message: str = ""             # full traceback / failure detail
    file_path: str | None = None       # file the failure points at, if known
    line: int | None = None


@dataclass(frozen=True)
class TestRunResult:
    """Canonical outcome of a backend-driven test invocation.

    Backends populate ``raw_output`` even on success so callers can
    fall back to LLM-driven parsing when structured extraction fails.
    ``failures`` is empty when ``status != "failed"``.
    """

    status: TestRunStatus
    exit_code: int
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    duration_sec: float = 0.0
    failures: list[TestFailure] = field(default_factory=list)
    raw_output: str = ""
    # Free-form per-backend diagnostics, e.g. the toolchain name when
    # ``status == "skipped"``. Never relied on by generic callers.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvHandle:
    """Opaque environment handle returned by ``detect_env`` / ``ensure_env``.

    Only the backend that produced it interprets ``extra``. Generic
    decoder code reads at most ``runtime_executable`` and
    ``project_root``; everything else is backend-private metadata
    (e.g. Go module cache path, Cargo target directory).
    """

    project_root: Path
    runtime_executable: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
