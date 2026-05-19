#!/usr/bin/env python3
"""Result-dict builders for codegen batch outcomes.

These helpers shape the JSON payloads that ``scripts.run_batch`` returns
to the slash-command driver (via ``--json`` output) so the AI agent can
read ``next_action`` and decide which command to run next.

Extracted from ``scripts/run_batch.py`` Module 7 ("Result Builders").
All four functions are internal helpers used only by Module 5's batch
orchestrator; no external API contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.execution_state import BatchExecutionState, CodeGenState, load_code_gen_state
from common.task_batch import PlannedTask, load_tasks_from_tasks_json
from common.paths import cmd_for


def _error(message: str, scripts: str) -> Dict[str, Any]:
    """Build an error result dict."""
    return {
        "success": False,
        "error": message,
        "next_action": f"Fix the issue, then run: {cmd_for('run_batch.py')} --next --json",
    }


def _all_done(global_state: CodeGenState, tasks_path: Path, scripts: str) -> Dict[str, Any]:
    """Build a result dict when all tasks are processed."""
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    total = len(all_tasks)
    completed = len(global_state.completed_task_ids)
    failed = len(global_state.failed_task_ids)

    if failed > 0:
        msg = f"All batches processed: {completed} completed, {failed} failed out of {total}."
        next_act = (
            f"Some batches failed. You can retry them with: "
            f"{cmd_for('run_batch.py')} --retry <batch_id> --json, "
            f"or run final validation: {cmd_for('run_batch.py')} --final-test --json"
        )
    else:
        msg = f"All {completed} batches completed successfully!"
        next_act = f"Run final validation: {cmd_for('run_batch.py')} --final-test --json"

    return {
        "success": True,
        "type": "complete",
        "message": msg,
        "stats": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
        },
        "next_action": next_act,
    }


def _success_result(
    batch_id: str,
    task: PlannedTask,
    batch_state: BatchExecutionState,
    attempts: List[Dict],
    total_duration: float,
    branch_merged: bool,
    scripts: str,
    tasks_path: Path,
    state_path: Path,
) -> Dict[str, Any]:
    """Build result dict for a successful batch."""
    global_state = load_code_gen_state(state_path)
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    completed = len(global_state.completed_task_ids)
    failed = len(global_state.failed_task_ids)
    total = len(all_tasks)
    remaining = total - completed - failed

    merged_ids = batch_state.merged_task_ids or []
    return {
        "success": True,
        "type": "batch_complete",
        "batch_id": batch_id,
        "file_path": task.file_path,
        "task_type": task.task_type,
        "attempts_used": len(attempts),
        "total_duration": round(total_duration, 1),
        "branch_merged": branch_merged,
        "merged_mode": len(merged_ids) > 1,
        "merged_task_count": len(merged_ids) if len(merged_ids) > 1 else 1,
        "stats": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
        },
        "next_action": (
            f"Batch completed. {remaining} tasks remaining. "
            f"Run: {cmd_for('run_batch.py')} --next --json"
            if remaining > 0 else
            f"All batches done! Run: {cmd_for('run_batch.py')} --final-test --json\n"
            f"Then run: {cmd_for('run_batch.py')} --global-review --json"
        ),
    }


def _failure_result(
    batch_id: str,
    task: PlannedTask,
    batch_state: BatchExecutionState,
    attempts: List[Dict],
    total_duration: float,
    scripts: str,
    tasks_path: Path,
    state_path: Path,
) -> Dict[str, Any]:
    """Build result dict for a failed batch."""
    global_state = load_code_gen_state(state_path)
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    completed = len(global_state.completed_task_ids)
    failed = len(global_state.failed_task_ids)
    total = len(all_tasks)
    remaining = total - completed - failed

    last_attempt = attempts[-1] if attempts else {}
    return {
        "success": False,
        "type": "batch_failed",
        "batch_id": batch_id,
        "file_path": task.file_path,
        "task_type": task.task_type,
        "attempts_used": len(attempts),
        "total_duration": round(total_duration, 1),
        "failure_reason": last_attempt.get("failure_reason", "Unknown"),
        "branch_preserved": batch_state.branch_name,
        "stats": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
        },
        "next_action": (
            f"Batch failed after {len(attempts)} attempts. "
            f"Branch '{batch_state.branch_name}' preserved for inspection. "
            f"Retry: {cmd_for('run_batch.py')} --retry {batch_id} --json, "
            f"or continue: {cmd_for('run_batch.py')} --next --json"
        ),
    }
