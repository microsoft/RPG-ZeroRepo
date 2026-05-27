#!/usr/bin/env python3
"""Test Output Parser — Unified pytest output analysis.

This module provides a single-pass parser for pytest output that extracts
all information needed by the TDD workflow:

- Statistics (passed/failed/errors/skipped/duration)
- Failure classification (ENV_ERROR / TEST_ERROR / CODE_ERROR / UNKNOWN_ERROR)
- ENV_ERROR sub-classification (missing_import / wrong_import_path / missing_package)
- Structured error extraction (all NameErrors, all ModuleNotFoundErrors)
- Failing test file paths (from actual output, not naming heuristics)
- Compact failure line summary (for prompt injection)

Usage::

    from code_gen.test_output_parser import analyze_test_output

    analysis = analyze_test_output(pytest_raw_output)
    # analysis.failure_type   → "ENV_ERROR"
    # analysis.missing_names  → ["Enum", "dataclass", "Callable"]
    # analysis.has_tests_run  → True/False
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from common.import_normalizer import detect_project_import_prefix
from common.paths import REPO_DIR as _REPO_DIR


# ============================================================================
# Data class
# ============================================================================

@dataclass
class TestOutputAnalysis:
    """Complete analysis of a pytest run output.

    Produced once by ``analyze_test_output()``, then consumed by
    ``run_batch.py`` (post-verify) and the orchestrator's analyse-failure
    paths without re-parsing.
    """

    # --- Statistics ----------------------------------------------------------
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0

    # --- Top-level classification --------------------------------------------
    failure_type: str = ""  # ENV_ERROR | TEST_ERROR | CODE_ERROR | UNKNOWN_ERROR

    # --- ENV_ERROR sub-classification ----------------------------------------
    env_sub_type: str = ""       # missing_import | wrong_import_path | missing_package
    env_fix_target: str = ""     # code | test
    env_instruction: str = ""    # human-readable fix guidance for sub-agent
    env_details: str = ""        # short detail string

    # --- Structured error info -----------------------------------------------
    missing_names: List[str] = field(default_factory=list)
    missing_modules: List[str] = field(default_factory=list)
    failing_test_files: List[str] = field(default_factory=list)
    failure_lines: str = ""  # compact excerpt of failure-relevant lines

    # --- Meta ----------------------------------------------------------------
    has_tests_run: bool = False  # True if at least one test was executed
    raw_output: str = ""         # original pytest output (for LLM fallback)

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON persistence in BatchExecutionState.

        ``raw_output`` is excluded to keep the serialized size small;
        ``last_test_output`` in BatchExecutionState already stores it.
        """
        d = asdict(self)
        d.pop("raw_output", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_output: str = "") -> "TestOutputAnalysis":
        valid = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        obj = cls(**filtered)
        obj.raw_output = raw_output
        return obj


# ============================================================================
# Unified parse entry point
# ============================================================================

def analyze_test_output(raw_output: str) -> TestOutputAnalysis:
    """Parse pytest output in a single pass.

    This is the **only** place in the codebase that parses raw pytest text.
    The returned ``TestOutputAnalysis`` is then shared by ``run_batch.py``'s
    post-verify path (for pass/fail decision) and the orchestrator's
    analyse-failure handler (for failure routing).
    """
    result = TestOutputAnalysis(raw_output=raw_output)

    # 1. Statistics
    _parse_stats(raw_output, result)
    result.has_tests_run = (result.passed + result.failed + result.errors) > 0

    # 2. Extract all structured errors (one-pass findall)
    result.missing_names = _extract_all_name_errors(raw_output)
    result.missing_modules = _extract_all_module_errors(raw_output)

    # 3. Failing test file paths
    result.failing_test_files = _extract_failing_files(raw_output)

    # 4. Classify
    _classify(raw_output, result)

    # 5. Compact failure lines
    result.failure_lines = _extract_relevant_lines(raw_output)

    return result


# ============================================================================
# Internal helpers
# ============================================================================

_SUMMARY_RE = re.compile(
    r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error|(\d+)\s+skipped|"
    r"in\s+([\d.]+)s"
)


def _parse_stats(output: str, result: TestOutputAnalysis) -> None:
    for m in _SUMMARY_RE.finditer(output):
        if m.group(1):
            result.passed = int(m.group(1))
        if m.group(2):
            result.failed = int(m.group(2))
        if m.group(3):
            result.errors = int(m.group(3))
        if m.group(4):
            result.skipped = int(m.group(4))
        if m.group(5):
            result.duration = float(m.group(5))


# -- Error extraction --------------------------------------------------------

def _extract_all_name_errors(output: str) -> List[str]:
    """Extract ALL NameError names (deduplicated, order-preserved)."""
    names = re.findall(r"NameError: name '(\w+)' is not defined", output)
    return list(dict.fromkeys(names))


def _extract_all_module_errors(output: str) -> List[str]:
    """Extract ALL ModuleNotFoundError / ImportError module names."""
    mods = re.findall(
        r"(?:ModuleNotFoundError|ImportError):.*?No module named '([^']+)'",
        output,
    )
    return list(dict.fromkeys(mods))


def _extract_failing_files(output: str) -> List[str]:
    """Extract test file paths from FAILED/ERROR lines, sorted by frequency."""
    raw = re.findall(r"(?:FAILED|ERROR)\s+(tests/\S+\.py)", output)
    # Strip ::TestClass::test_method, keep only the file path
    files = [f.split("::")[0] for f in raw]
    if not files:
        return []
    # Most-frequently-failing file first
    return [f for f, _ in Counter(files).most_common()]


# -- Classification ----------------------------------------------------------

# Keywords checked in priority order (first match wins).
_ENV_KEYWORDS = (
    "modulenotfounderror",
    "importerror",
    "no module named",
    "nameerror",
    "package not found",
    "pip install",
    "missing dependency",
    "command not found",
)

_TEST_ERROR_KEYWORDS = (
    "fixture",
    "conftest",
    "test setup failed",
    "test collection failed",
    "@pytest",
    "parametrize",
    "test file",
)

_CODE_ERROR_KEYWORDS = (
    "assertionerror",
    "assert",
    "expected",
    "actual",
    "!=",
    "not equal",
    "typeerror",
    "valueerror",
    "attributeerror",
    "keyerror",
)


def _classify(output: str, result: TestOutputAnalysis) -> None:
    """Set ``failure_type`` and, for ENV_ERROR, the sub-classification."""
    lower = output.lower()

    # Priority: ENV → TEST → CODE → UNKNOWN
    if any(kw in lower for kw in _ENV_KEYWORDS):
        result.failure_type = "ENV_ERROR"
        _classify_env(output, result)
        return

    if any(kw in lower for kw in _TEST_ERROR_KEYWORDS):
        result.failure_type = "TEST_ERROR"
        return

    if any(kw in lower for kw in _CODE_ERROR_KEYWORDS):
        result.failure_type = "CODE_ERROR"
        return

    result.failure_type = "UNKNOWN_ERROR"


def _classify_env(output: str, result: TestOutputAnalysis) -> None:
    """Sub-classify an ENV_ERROR and populate ``env_*`` fields.

    This consolidates the logic from the old ``_classify_env_error()`` in
    Earlier failure routing in ``run_batch.py``, enhanced to extract ALL
    missing names at once.
    """
    # --- 1. NameError: missing imports in source file ---
    if result.missing_names:
        names = result.missing_names
        names_str = ", ".join(f"`{n}`" for n in names)
        result.env_sub_type = "missing_import"
        result.env_fix_target = "code"
        result.env_instruction = (
            f"The source file uses {names_str} but they are not imported. "
            f"Add the correct import statements for ALL of these names "
            f"at the top of the file (after `from __future__` imports). "
            f"Common mappings: Enum→enum, dataclass→dataclasses, "
            f"Callable/Optional/List→typing. "
            f"Do NOT remove any existing code. Do NOT modify test files."
        )
        result.env_details = f"Undefined names: {', '.join(names)}"
        return

    # --- 2. ModuleNotFoundError / ImportError ---
    if result.missing_modules:
        missing_mod = result.missing_modules[0]
        top_level = missing_mod.split(".")[0]

        # Project-internal wrong path?
        # Dynamically detect project package names from repo layout.
        _detected_prefix = detect_project_import_prefix(repo_path=_REPO_DIR)
        project_indicators: set = set()
        if _detected_prefix:
            _parts = _detected_prefix.split('.', 1)
            if len(_parts) == 2:
                project_indicators.add(_parts[1])
        if top_level in project_indicators:
            prefix_str = _detected_prefix or f"src.{top_level}"
            is_test_import = "importing test module" in output.lower()
            if is_test_import:
                result.env_sub_type = "wrong_import_path"
                result.env_fix_target = "test"
                result.env_instruction = (
                    f"The test file uses the wrong import path `{missing_mod}`. "
                    f"This project uses `{prefix_str}.*` (with `src.` prefix). "
                    f"Change ALL occurrences of `from {missing_mod}` to "
                    f"`from src.{missing_mod}` in the test file. "
                    f"Do NOT modify production/source code."
                )
                result.env_details = f"Wrong path: {missing_mod} -> src.{missing_mod}"
            else:
                result.env_sub_type = "wrong_import_path"
                result.env_fix_target = "code"
                result.env_instruction = (
                    f"The source file uses the wrong import path `{missing_mod}`. "
                    f"This project uses `{prefix_str}.*` (with `src.` prefix). "
                    f"Fix the import path. Do NOT modify test files."
                )
                result.env_details = f"Wrong path: {missing_mod}"
            return

        # Third-party package
        result.env_sub_type = "missing_package"
        result.env_fix_target = "code"
        result.env_instruction = (
            f"Third-party package `{missing_mod}` is not installed. "
            f"The build system will attempt auto-installation. "
            f"If the package is genuinely needed, keep the import. "
            f"Only remove the import if it is truly NOT used in the code. "
            f"Do NOT modify test files."
        )
        result.env_details = f"Missing package: {missing_mod}"
        return

    # --- 3. Fallback ---
    result.env_sub_type = "missing_package"
    result.env_fix_target = "code"
    result.env_instruction = (
        "Environment or import issue detected. "
        "Check the error output and fix the import in the appropriate file."
    )
    result.env_details = ""


# -- Failure line extraction --------------------------------------------------

_FAILURE_LINE_KEYWORDS = (
    "FAILED",
    "ERROR",
    "AssertionError",
    "TypeError",
    "ValueError",
    "NameError",
    "AttributeError",
    "KeyError",
    "ModuleNotFoundError",
    "ImportError",
    "E   ",  # pytest indented assertion detail lines
)


def _extract_relevant_lines(output: str, max_chars: int = 1500) -> str:
    """Extract only failure-relevant lines from pytest output.

    Returns a compact excerpt suitable for prompt injection (~1.5 KB max).
    """
    lines = [
        line
        for line in output.split("\n")
        if any(kw in line for kw in _FAILURE_LINE_KEYWORDS)
    ]
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n... (truncated)"
    return excerpt


# ============================================================================
# Keyword filter helpers (used by ``run_batch.py`` to derive ``-k`` patterns)
# ============================================================================

def build_keyword_filter(units_key: List[str]) -> Optional[str]:
    """Build a pytest ``-k`` filter expression from unit class names.

    Returns ``None`` when *units_key* is empty.

    Strips common prefixes like "class " and "def " from unit names.

    Example::

        build_keyword_filter(["class DirtyRegion", "class DirtyRegionTracker"])
        # → "DirtyRegion or DirtyRegionTracker"
    """
    if not units_key:
        return None
    unique = list(dict.fromkeys(units_key))
    # Strip common prefixes like "class ", "def ", "function " from unit names
    cleaned = []
    for unit in unique:
        name = unit
        for prefix in ["class ", "def ", "async def ", "function "]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned.append(name)
    return " or ".join(cleaned)


def validate_test_ran(analysis: TestOutputAnalysis) -> bool:
    """Return True if at least one test was actually executed.

    Use after ``is_test_successful()`` to guard against the -k filter
    matching zero tests (pytest returns exit 0 with ``-v`` in that case).
    """
    return analysis.has_tests_run
