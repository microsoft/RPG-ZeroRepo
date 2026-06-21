#!/usr/bin/env python3
"""Git branch operations for the codegen batch lifecycle.

This module hosts the four helpers extracted from
``scripts/run_batch.py`` Module 2 ("Git Operations"):

* :func:`ensure_on_main` — guarantee we're on ``main``, autosaving WIP changes.
* :func:`setup_batch_branch` — create / reuse a ``batch/<id>`` branch from main.
* :func:`merge_batch_branch` — merge a batch branch into main (``--no-ff``) and delete it.
* :func:`abandon_batch_branch` — leave a failed batch branch in place for inspection.

All four are internal helpers used only by ``scripts.run_batch``;
they have **no** stable public API.  External callers should not import
from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from common.generated_artifacts import (
    find_persisted_generated_artifact_changes,
    format_generated_artifact_violation,
)
from common.git_utils import GitRunner, sanitize_branch_component

logger = logging.getLogger(__name__)


def ensure_on_main(git: GitRunner) -> None:
    """Ensure we are on the main branch, switching if necessary.

    If there are uncommitted changes on a non-main branch, they are
    committed with a WIP message before switching.

    Raises:
        RuntimeError: If switching to main fails.
    """
    current = git.get_current_branch()
    if current == git.main_branch:
        return

    logger.info("Currently on branch '%s', switching to '%s'", current, git.main_branch)
    if git.has_uncommitted_changes():
        logger.warning("Committing uncommitted changes on '%s' before switching", current)
        git.stage_and_commit(f"WIP: auto-save before switching to {git.main_branch}")

    if not git.switch_branch(git.main_branch):
        raise RuntimeError(
            f"Failed to switch to {git.main_branch}. "
            f"Current branch: {current}. Manual intervention needed."
        )


def setup_batch_branch(
    git: GitRunner,
    batch_id: str,
    repo_path: Path,
    reuse_existing: bool = False,
) -> Tuple[bool, str, str]:
    """Create (or reuse) a batch branch from latest main HEAD.

    Args:
        git: GitRunner instance.
        batch_id: Batch identifier.
        repo_path: Repo root path.
        reuse_existing: If True and branch exists, switch to it instead of
                        deleting and recreating.

    Returns:
        (success, branch_name, initial_commit)
    """
    ensure_on_main(git)

    safe_id = sanitize_branch_component(batch_id, max_len=50, fallback="batch")
    branch_name = f"batch/{safe_id}"

    if git.branch_exists(branch_name):
        if reuse_existing:
            logger.info("Reusing existing branch '%s'", branch_name)
            if not git.switch_branch(branch_name):
                return False, branch_name, ""
            initial_commit = git.get_head_commit()
            return True, branch_name, initial_commit
        else:
            logger.info("Deleting stale branch '%s' (will recreate from main)", branch_name)
            git.delete_branch(branch_name, force=True)

    initial_commit = git.get_head_commit()
    success = git.create_branch(branch_name)
    if not success:
        logger.error("Failed to create branch '%s'", branch_name)
    return success, branch_name, initial_commit


def merge_batch_branch(
    git: GitRunner,
    branch_name: str,
    batch_id: str,
    file_path: str = "",
    units: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Merge a batch branch into main and delete it.

    1. Commit any remaining changes on the branch.
    2. Build a merge message with batch_completed marker in body.
    3. Merge into main (--no-ff) with custom message.
    4. Delete the branch.

    The merge message body contains ``batch_completed: <id>`` so that
    git-based state recovery (``git log --grep``) can detect completed batches.

    Args:
        git: GitRunner instance.
        branch_name: Branch to merge.
        batch_id: Batch ID for recovery marker.
        file_path: Target file path for readable message.
        units: List of unit names for readable message.

    Returns:
        ``(success, error_description)``

    error_description values:
        - None when the merge succeeded.
        - ``"branch_missing"`` when ``branch_name`` does not exist.  Callers
          should treat this as a skip (sub-agent setup issue), NOT as a
          retryable failure.
        - Any other string is propagated from ``GitRunner.merge_branch``
          (e.g. ``"merge_conflict"``, ``"merge_failed"``).
    """
    # Branch went missing → caller must skip, not consume a retry slot.
    # Happens when the sub-agent committed straight to main or deleted
    # the branch.  Stage any local changes first so they aren't lost.
    if not git.branch_exists(branch_name):
        logger.warning(
            "Cannot merge: branch '%s' does not exist (sub-agent did not "
            "use the batch branch). Treating as skip.",
            branch_name,
        )
        if git.has_uncommitted_changes():
            git.stage_and_commit(
                f"WIP: salvage uncommitted changes after missing branch '{branch_name}'"
            )
        return False, "branch_missing"

    generated_artifact_changes = find_persisted_generated_artifact_changes(
        git.repo_path,
        base_ref=git.main_branch,
    )
    if generated_artifact_changes:
        summary = format_generated_artifact_violation(generated_artifact_changes)
        logger.error("Cannot merge generated artifact changes:\n%s", summary)
        return False, summary

    # Commit any leftover changes
    if git.has_uncommitted_changes():
        git.stage_and_commit(f"batch: final changes for {batch_id}")

    # Build merge message: readable subject + marker in body
    units_str = ", ".join(units) if units else ""
    is_marker = file_path.startswith("<") and file_path.endswith(">")
    if is_marker:
        scope = file_path.strip("<>").lower().replace("_", "-")
    elif file_path:
        scope = file_path.split("/")[-1].replace(".py", "")
    else:
        scope = ""

    if scope and units_str:
        subject = f"merge({scope}): {units_str}"
    elif scope:
        subject = f"merge: {scope}"
    else:
        subject = f"merge: {branch_name}"

    body_lines = [f"batch_completed: {batch_id}"]
    if file_path:
        body_lines.append(f"Target: {file_path}")
    if units_str:
        body_lines.append(f"Units: {units_str}")
    merge_msg = subject + "\n\n" + "\n".join(body_lines)

    merge_ok, error = git.merge_branch(branch_name, message=merge_msg)
    if merge_ok:
        git.delete_branch(branch_name)
        logger.info("Merged branch '%s' into main and deleted it", branch_name)
        return True, None
    else:
        logger.error("Failed to merge branch '%s': %s", branch_name, error)
        return False, error


def abandon_batch_branch(git: GitRunner, branch_name: str) -> None:
    """Switch back to main, leaving the batch branch intact for inspection."""
    if git.has_uncommitted_changes():
        git.stage_and_commit("WIP: batch failed, preserving state")

    logger.info("Abandoning branch '%s', switching to main", branch_name)
    git.switch_branch(git.main_branch)
