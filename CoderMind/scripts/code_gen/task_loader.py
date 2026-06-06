#!/usr/bin/env python3
"""Task selection helpers for the codegen batch orchestrator.

This module hosts the two task-picker helpers used by
``scripts.run_batch``:

* :func:`get_next_pending_task_id` — pick the next single task to run,
  with git-based auto-recovery and integration-test deferral.
* :func:`get_next_merged_tasks` — pick a same-file group of pending
  implementation tasks for "file-merge" mode batches.

They share three private helpers — ``_git_grep_pattern``,
``_git_has_gen_code_commit``, ``_has_failed_impl_dependencies`` — kept
local to this module since they have no callers elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from common.execution_state import (
    CodeGenState,
    STATE_FILE,
    save_code_gen_state,
)
from common.git_utils import GitRunner
from common.paths import REPO_DIR
from common.task_batch import PlannedTask, load_tasks_from_tasks_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git-based recovery helpers (private)
# ---------------------------------------------------------------------------

def _git_grep_pattern(repo_path: Path, pattern: str) -> bool:
    """``True`` iff ``git log --grep=<pattern>`` finds at least one commit."""
    try:
        git = GitRunner(str(repo_path))
        result = git.run_git(
            ["log", "--all", "--oneline", "--grep", pattern, "--max-count=1"]
        )
        return result.success and bool(result.stdout.strip())
    except Exception:
        return False


def _git_has_gen_code_commit(
    repo_path: Path,
    batch_id: str,
    completed_ids: Optional[set] = None,
) -> bool:
    """Check if a batch was **successfully completed** based on git history.

    Uses a two-tier strategy:
    1. Check for the new ``batch_completed: <id>`` marker (reliable).
    2. Fall back to the old ``gen_code iter 1 — batch <id>`` pattern,
       but ONLY if *batch_id* is already in *completed_ids* — this avoids
       false positives for tasks that had gen_code commits but failed tests.
    """
    # 1. New completion marker
    if _git_grep_pattern(repo_path, f"batch_completed: {batch_id}"):
        return True
    # 2. Legacy fallback — only trust for known-completed tasks
    if completed_ids and batch_id in completed_ids:
        return _git_grep_pattern(repo_path, f"gen_code iter 1 — batch {batch_id}")
    return False


def _has_failed_impl_dependencies(
    integration_task: PlannedTask,
    failed_ids: set,
    all_tasks: list,
) -> bool:
    """Check if an integration test has failed implementation dependencies.

    Heuristic: if any implementation task in the same subtree is failed,
    the integration test likely depends on it and will fail too.
    """
    subtree = integration_task.subtree
    for t in all_tasks:
        if (t.task_type == "implementation"
                and t.subtree == subtree
                and t.task_id in failed_ids):
            return True
    return False


# ---------------------------------------------------------------------------
# Public task pickers
# ---------------------------------------------------------------------------

def get_next_pending_task_id(
    tasks_path: Path,
    state: CodeGenState,
    repo_path: Optional[Path] = None,
    auto_recover: bool = True,
    state_path: Path = STATE_FILE,
) -> Optional[str]:
    """Get the task_id of the next pending task to process.

    If ``auto_recover`` is True and a pending task already has gen_code
    commits in git history, it is auto-completed (added to
    ``completed_task_ids`` and persisted) to avoid redundant TDD cycles
    after state resets.

    Integration tests are deferred until all implementation tasks are
    complete, since they often depend on modules from multiple subtrees.
    """
    completed = set(state.completed_task_ids)
    failed = set(state.failed_task_ids)
    repo_path = repo_path or REPO_DIR
    state_modified = False

    all_tasks = load_tasks_from_tasks_json(tasks_path)

    # Track pending implementation task IDs.  Updated during auto-recovery
    # so the integration-test deferral check stays accurate.
    _pending_impl_ids = {
        t.task_id for t in all_tasks
        if t.task_id not in completed and t.task_id not in failed
        and t.task_type == "implementation"
    }

    for t in all_tasks:
        if t.task_id in completed or t.task_id in failed:
            continue
        # Git-based auto-recovery: skip tasks whose code was already generated
        if auto_recover and t.task_id not in failed and _git_has_gen_code_commit(repo_path, t.task_id, completed):
            logger.info(
                "Git-based recovery: auto-completing %s "
                "(gen_code commits found in git history)",
                t.task_id,
            )
            state.completed_task_ids.append(t.task_id)
            completed.add(t.task_id)
            _pending_impl_ids.discard(t.task_id)
            state_modified = True
            continue
        # Defer integration tests until all implementation tasks are done.
        if t.task_type == "integration_test" and _pending_impl_ids:
            continue
        # Skip integration tests whose implementation dependencies have failed
        if t.task_type == "integration_test" and not _pending_impl_ids:
            if _has_failed_impl_dependencies(t, failed, all_tasks):
                logger.info(
                    "Skipping integration test %s: "
                    "dependent implementation tasks are in failed state",
                    t.task_id,
                )
                state.failed_task_ids.append(t.task_id)
                failed.add(t.task_id)
                state_modified = True
                continue
        # Found a genuinely pending task
        if state_modified:
            state.completed_tasks = len(state.completed_task_ids)
            save_code_gen_state(state, state_path)
        return t.task_id

    # All tasks processed — persist any auto-recoveries
    if state_modified:
        state.completed_tasks = len(state.completed_task_ids)
        save_code_gen_state(state, state_path)
    return None


def get_next_merged_tasks(
    tasks_path: Path,
    state: CodeGenState,
    max_units: int = 0,
    repo_path: Optional[Path] = None,
    state_path: Path = STATE_FILE,
) -> Optional[List[PlannedTask]]:
    """Get the next group of pending tasks for one merged batch (file-merge mode).

    Rules:
    - Only merge ``task_type == "implementation"`` tasks from the same ``file_path``.
    - Special types (integration_test, final_test_docs, main_entry, project_*)
      are never merged; they are returned as a single-element list.
    - If ``max_units > 0``, cap the merged group so total units ``<= max_units``.
    - If ``max_units == 0`` (default), merge all tasks for the same file.
    - Tasks with gen_code commits in git history are auto-completed (skipped).
    - Integration tests are deferred until all implementation tasks are complete.

    Returns:
        List of PlannedTask objects to implement together, or None if nothing pending.
    """
    completed = set(state.completed_task_ids)
    failed = set(state.failed_task_ids)
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    repo_path = repo_path or REPO_DIR
    state_modified = False

    # Track pending implementation task IDs.  Updated during auto-recovery
    # so the integration-test deferral check stays accurate.
    _pending_impl_ids = {
        t.task_id for t in all_tasks
        if t.task_id not in completed and t.task_id not in failed
        and t.task_type == "implementation"
    }

    # 1. Find the first pending task (with auto-recovery)
    first_pending: Optional[PlannedTask] = None
    for t in all_tasks:
        if t.task_id in completed or t.task_id in failed:
            continue
        # Auto-recover tasks with existing gen_code commits
        if _git_has_gen_code_commit(repo_path, t.task_id, completed):
            logger.info("Git-based recovery (merge mode): auto-completing %s", t.task_id)
            state.completed_task_ids.append(t.task_id)
            completed.add(t.task_id)
            _pending_impl_ids.discard(t.task_id)
            state_modified = True
            continue
        # Defer integration tests until all implementation tasks are done
        if t.task_type == "integration_test" and _pending_impl_ids:
            continue
        # Skip integration tests whose impl dependencies failed
        if t.task_type == "integration_test" and not _pending_impl_ids:
            if _has_failed_impl_dependencies(t, failed, all_tasks):
                logger.info(
                    "Skipping integration test %s (merge mode): "
                    "dependent implementation tasks are in failed state",
                    t.task_id,
                )
                state.failed_task_ids.append(t.task_id)
                failed.add(t.task_id)
                state_modified = True
                continue
        first_pending = t
        break

    # Persist any auto-recoveries
    if state_modified:
        state.completed_tasks = len(state.completed_task_ids)
        save_code_gen_state(state, state_path)

    if not first_pending:
        return None

    # 2. Non-implementation types are never merged
    if first_pending.task_type != "implementation":
        return [first_pending]

    # 3. Collect all pending implementation tasks for the same file_path
    target_file = first_pending.file_path
    file_tasks = [
        t for t in all_tasks
        if t.file_path == target_file
        and t.task_type == "implementation"
        and t.task_id not in completed
        and t.task_id not in failed
    ]

    # 4. If max_units is set, greedily collect tasks up to the limit
    if max_units > 0:
        selected: List[PlannedTask] = []
        unit_count = 0
        for t in file_tasks:
            if unit_count + len(t.units_key) <= max_units:
                selected.append(t)
                unit_count += len(t.units_key)
            else:
                break
        # Always return at least one task (even if it alone exceeds max_units)
        return selected if selected else [file_tasks[0]]

    return file_tasks if file_tasks else [first_pending]
