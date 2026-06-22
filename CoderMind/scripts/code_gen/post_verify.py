#!/usr/bin/env python3
"""Post-verification: independent pytest re-run after a sub-agent batch.

This module hosts :func:`post_verify`, extracted from
``scripts/run_batch.py`` Module 4 ("Post-Verification").

The sub-agent self-reports ``BATCH_RESULT: PASS`` or ``FAIL`` after its
TDD cycle, but we do **not** trust that signal — :func:`post_verify`
re-runs pytest from the orchestrator process to get an authoritative
answer.  This catches two failure modes:

* Sub-agent claims PASS but actually skipped failing tests.
* Sub-agent's environment differed from the orchestrator's (different
  PYTHONPATH, stale ``__pycache__``, etc.).

This is an internal helper used only by ``scripts.run_batch``; no
external API contract.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Tuple

from common.generated_artifacts import (
    ensure_generated_artifact_excludes,
    find_persisted_generated_artifact_changes,
    format_generated_artifact_violation,
)
from common.git_utils import GitRunner
from common.task_batch import PlannedTask
from code_gen.test_runner import (
    ensure_deps_installed,
    find_related_test_files,
    resolve_test_backend,
    run_project_tests,
)

logger = logging.getLogger(__name__)


from code_gen._constants import (  # noqa: E402
    DEFAULT_PYTEST_OVERALL_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
)


def post_verify(
    repo_path: Path,
    task: PlannedTask,
    timeout: int = 0,  # 0 = auto-select based on task type
) -> Tuple[bool, str]:
    """Run an independent pytest to verify the batch result.

    This is the authoritative check — we do NOT trust the sub-agent's
    self-reported BATCH_RESULT.

    Args:
        repo_path: Project repo path.
        task: The PlannedTask for this batch.
        timeout: Overall pytest timeout.

    Returns:
        ``(passed, test_output_summary)``
    """
    ensure_generated_artifact_excludes(repo_path)

    generated_artifact_changes = find_persisted_generated_artifact_changes(
        repo_path,
        base_ref=GitRunner.MAIN_BRANCH,
    )
    if generated_artifact_changes:
        summary = format_generated_artifact_violation(generated_artifact_changes)
        logger.warning("Post-verification rejected generated artifact changes:\n%s", summary)
        return False, summary

    # Use the global safety-net timeout for all task types.
    # Per-test hang prevention is handled by pytest-timeout (--timeout=DEFAULT_TEST_TIMEOUT).
    if timeout == 0:
        timeout = DEFAULT_PYTEST_OVERALL_TIMEOUT

    def _git_diff_test_files(prefix: str = "tests/") -> list:
        """Return test files added/modified by this batch branch vs the main branch."""
        try:
            main_branch = GitRunner(str(repo_path)).main_branch
            diff = subprocess.run(
                ["git", "diff", f"{main_branch}..HEAD", "--name-only"],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            return [
                str(repo_path / f) for f in diff.stdout.splitlines()
                if f.startswith(prefix) and (repo_path / f).exists()
            ]
        except Exception:
            return []

    # Find test files to scope post-verification.
    # Special file_path values like "<INTEGRATION_TEST>" or "<WIRING>" indicate
    # synthetic tasks; use git diff to find only what this batch added/modified.
    test_files = []
    if not (task.file_path.startswith("<") and task.file_path.endswith(">")):
        # Regular file batch: find tests related to the target source file.
        test_files = find_related_test_files(task.file_path, repo_path)
    elif task.task_type == "integration_test":
        # Find integration test files added/modified in this batch via git diff.
        # Falls back to deriving the filename from the unit name.
        test_files = _git_diff_test_files("tests/test_integration_")
        if not test_files:
            # Derived fallback: "Application Core_integration_tests" → test_integration_app_core.py
            for unit in task.units_key:
                subtree_name = unit.replace("_integration_tests", "").strip()
                fname = "test_integration_" + subtree_name.lower().replace(" ", "_") + ".py"
                candidate = repo_path / "tests" / fname
                if candidate.exists():
                    test_files.append(str(candidate))
    elif task.task_type == "wiring":
        # Wiring verifies cross-module connections; run every test file the batch
        # added or modified.  If git diff finds nothing (e.g., on a bare retry),
        # fall back to all tests so no regression goes undetected.
        test_files = _git_diff_test_files("tests/test_")

    regular_file = not (task.file_path.startswith("<") and task.file_path.endswith(">"))
    backend_hint_files = test_files or ([task.file_path] if regular_file else None)
    backend = resolve_test_backend(valid_files=backend_hint_files, repo_path=repo_path)
    run_test_files = test_files if backend.name == "python" else None

    logger.info(
        "Post-verification: related test files=%s; running %s project tests on %s",
        test_files if test_files else "none",
        backend.display_name,
        run_test_files if run_test_files else "all tests",
    )

    if backend.name == "python":
        try:
            ensure_deps_installed(repo_path)
        except Exception as exc:
            logger.warning("ensure_deps_installed failed: %s", exc)

    result = run_project_tests(
        repo_path,
        test_files=run_test_files,
        timeout=timeout,
        extra_args=[f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread"],
        backend=backend,
    )

    generated_artifact_changes = find_persisted_generated_artifact_changes(
        repo_path,
        base_ref=GitRunner.MAIN_BRANCH,
    )
    if generated_artifact_changes:
        summary = format_generated_artifact_violation(generated_artifact_changes)
        logger.warning("Post-verification rejected generated artifact changes:\n%s", summary)
        return False, summary

    # Build summary
    summary_lines = [
        f"passed={result.passed} failed={result.failed} "
        f"errors={result.errors} skipped={result.skipped}",
    ]
    if not result.success:
        # Include truncated output for the resume prompt
        output = result.output
        if len(output) > 4000:
            output = output[:4000] + "\n...(truncated)"
        summary_lines.append(output)

    summary = "\n".join(summary_lines)
    logger.info("Post-verification result: success=%s %s", result.success, summary_lines[0])
    if not result.success:
        logger.debug("Post-verification pytest output:\n%s", result.output)
    return result.success, summary
