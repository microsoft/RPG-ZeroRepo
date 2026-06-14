"""Parser-agnostic test-execution result types.

Backend-driven test execution returns :class:`TestRunResult` so
decoder stages can reason about pytest, ``go test``, ``cargo test``,
and other native test tools through one result shape.

Defined here (not in :mod:`code_gen`) so backends can return the type
without an import cycle through the decoder package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence


# Status that downstream callers (post_verify / global_review /
# final_validation) branch on. ``skipped`` is reserved for the
# toolchain-unavailable case (non-Python backends only) so the
# caller can mark the verification step as a non-fatal WARN.
TestRunStatus = Literal["passed", "failed", "errored", "skipped"]


def ran_no_tests(
    exit_code: int,
    raw_output: str,
    *,
    observed_tests: int | None = None,
    no_tests_markers: Sequence[str] = (),
    empty_output_is_no_op: bool = True,
) -> bool:
    """Return True when a test command exited 0 but executed no tests.

    A zero-test run is a no-op, not a pass: it is the native-toolchain
    equivalent of pytest collecting zero items. Treating ``exit_code == 0``
    alone as success is how a verification gate silently approves a repo
    whose tests never ran — e.g. ``go test ./...`` matching no packages, or
    a runner invoked before the sources are in the working tree.

    The check is deliberately fail-safe toward "tests ran" so a real
    passing run is never mis-flagged: evidence of *no* tests, in order,

    * empty / whitespace-only output (universal — a real run always emits
      progress lines), unless ``empty_output_is_no_op`` is False;
    * a tool-specific "no tests" marker phrase in the output;
    * a reliably parsed ``observed_tests == 0``.

    ``empty_output_is_no_op`` must be False for backends whose test command
    falls back to a compile-only check (C / C++ ``-fsyntax-only``): a clean
    compile legitimately produces no output and is the strongest signal
    that language has, so empty output there means "passed", not "no-op".

    Backends that cannot parse a trustworthy count pass
    ``observed_tests=None`` and rely on the output signals only, so an
    unrecognized-but-non-empty output is treated as a pass rather than a
    false failure. Only meaningful when ``exit_code == 0``; a non-zero exit
    is already a failure reported by the caller.
    """
    if exit_code != 0:
        return False
    text = (raw_output or "").strip()
    if not text:
        return empty_output_is_no_op
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in no_tests_markers):
        return True
    if observed_tests is not None and observed_tests == 0:
        return True
    return False


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
