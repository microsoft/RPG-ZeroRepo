#!/usr/bin/env python3
"""Check Code Gen Script - Validation for Code Generation State.

Validates the current state of code generation and determines execution state:
- "error": Required input files missing or invalid
- "init": No code_gen_state.jsonl exists, ready to start
- "in_progress": A batch is currently being processed
- "continue": Ready to continue with next batch
- "complete": All batches completed

Returns JSON with validation status and statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import centralized paths and state loader
from common.paths import (
    TASKS_FILE,
    CODE_GEN_STATE_FILE as STATE_FILE,
    get_scripts_dir,
    cmd_for,
    REPO_DIR,
)
from common.execution_state import load_code_gen_state
from common.execution_state import load_code_gen_state as _load_state, save_code_gen_state as _save_state
from common.execution_state import complete_batch as _complete_batch


def validate_tasks_file(tasks_path: Path) -> Tuple[bool, List[str], int]:
    """Validate that tasks.json exists and count tasks.
    
    Returns: (valid, errors, total_tasks)
    """
    errors = []
    total_tasks = 0
    
    if not tasks_path.exists():
        errors.append(f"Tasks file not found: {tasks_path}")
        return False, errors, 0
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in tasks file: {e}")
        return False, errors, 0
    
    # Count tasks
    planned_tasks_dict = data.get("planned_tasks_dict", {})
    
    for subtree, files_dict in planned_tasks_dict.items():
        for file_path, tasks_list in files_dict.items():
            total_tasks += len(tasks_list)
    
    if total_tasks == 0:
        errors.append("No tasks found in tasks.json")
        return False, errors, 0
    
    return True, errors, total_tasks


def load_state(state_path: Path) -> Dict[str, Any]:
    """Load code gen state from file via centralized loader.
    
    Returns a raw dict for backward compatibility with the rest of this script.
    Returns empty dict if file doesn't exist or is a fresh (empty) state.
    """
    state_obj = load_code_gen_state(state_path)
    state_dict = state_obj.to_dict()
    # A fresh CodeGenState (no file) has no completed/failed tasks and no current batch
    # Treat it as "no state" to trigger the "init" path
    if (not state_dict.get("completed_task_ids")
            and not state_dict.get("failed_task_ids")
            and not state_dict.get("current_batch_id")
            and not state_dict.get("initialized")):
        return {}
    return state_dict


def get_all_task_ids(tasks_path: Path) -> List[str]:
    """Get all task IDs from tasks.json."""
    task_ids = []
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        planned_tasks_dict = data.get("planned_tasks_dict", {})
        subtree_order = data.get("subtree_order", list(planned_tasks_dict.keys()))
        
        for subtree in subtree_order:
            if subtree not in planned_tasks_dict:
                continue
            
            files_dict = planned_tasks_dict[subtree]
            for file_path, tasks_list in files_dict.items():
                for task_data in tasks_list:
                    if "task_id" in task_data:
                        task_ids.append(task_data["task_id"])
    except Exception:
        pass
    
    return task_ids


def determine_state(
    tasks_path: Path,
    state_path: Path
) -> Dict[str, Any]:
    """Determine the current execution state.
    
    Returns dict with:
    - type: "error" | "init" | "in_progress" | "continue" | "complete"
    - message: Human-readable message
    - current_batch: Current batch info (if in_progress)
    - next_batch: Next batch to process (if continue)
    - stats: Statistics about progress
    """
    result = {
        "type": "error",
        "message": "",
        "current_batch": None,
        "next_batch": None,
        "stats": {}
    }
    
    # Validate tasks file
    valid, errors, total_tasks = validate_tasks_file(tasks_path)
    
    if not valid:
        result["type"] = "error"
        result["message"] = "; ".join(errors)
        result["next_action"] = "Fix the reported issues. If tasks.json is missing, run /cmind.plan_tasks first."
        return result
    
    # Get all task IDs
    all_task_ids = get_all_task_ids(tasks_path)
    
    # Load state
    state = load_state(state_path)
    
    scripts = get_scripts_dir()

    if not state:
        # No state file - ready to start
        result["type"] = "init"
        result["message"] = f"Ready to start code generation ({total_tasks} tasks)"
        result["next_batch"] = all_task_ids[0] if all_task_ids else None
        result["stats"] = {
            "total_tasks": total_tasks,
            "completed": 0,
            "failed": 0,
            "remaining": total_tasks
        }
        result["next_action"] = (
            f"Run: {cmd_for('init_codebase.py')} --json to initialize the repository, "
            f"then run: {cmd_for('run_batch.py')} --next --json to start the first batch."
        )
        result["workflow_hint"] = (
            "run_batch.py --next dispatches a sub-agent that autonomously "
            "writes tests → code → runs pytest → fixes → repeats (up to 5x)"
        )
        return result
    
    # Parse state
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))
    current_batch_id = state.get("current_batch_id")
    current_batch_state = state.get("current_batch_state")
    
    # Calculate progress
    completed = len(completed_ids)
    failed = len(failed_ids)
    processed = completed + failed
    remaining = total_tasks - processed
    
    result["stats"] = {
        "total_tasks": total_tasks,
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
        "success_rate": (completed / total_tasks * 100) if total_tasks > 0 else 0
    }
    
    # Determine state
    if current_batch_id and current_batch_state:
        # Consistency check: if the current batch's task is already in completed
        # or failed lists, the batch was actually finished but current_batch_id
        # was not properly cleared (stale state). Auto-recover by clearing it.
        if current_batch_id in completed_ids or current_batch_id in failed_ids:
            try:
                _gs = _load_state(state_path)
                _gs.current_batch_id = None
                _gs.current_batch_state = None
                _save_state(_gs, state_path)
                # Reload and recalculate all derived variables
                state = load_state(state_path)
                completed_ids = set(state.get("completed_task_ids", []))
                failed_ids = set(state.get("failed_task_ids", []))
                current_batch_id = state.get("current_batch_id")
                current_batch_state = state.get("current_batch_state")
                completed = len(completed_ids)
                failed = len(failed_ids)
                processed = completed + failed
                remaining = total_tasks - processed
                result["stats"] = {
                    "total_tasks": total_tasks,
                    "completed": completed,
                    "failed": failed,
                    "remaining": remaining,
                    "success_rate": (completed / total_tasks * 100) if total_tasks > 0 else 0
                }
            except Exception:
                pass  # If cleanup fails, proceed with stale state

    if current_batch_id and current_batch_state:
        # A batch is in progress
        phase = current_batch_state.get("phase", "unknown")
        
        # Auto-recover: if tests passed (phase=complete) but complete_batch was
        # never called, finalize the batch now to prevent it being counted as failed.
        # Note: this only updates state tracking; git commit/merge and RPG updates
        # are skipped since code changes were already committed during the TDD loop.
        if phase == "complete":
            try:
                _complete_batch(current_batch_id, True, state_path)
                # Reload state after auto-recovery
                state = load_state(state_path)
                completed_ids = set(state.get("completed_task_ids", []))
                failed_ids = set(state.get("failed_task_ids", []))
                completed = len(completed_ids)
                failed = len(failed_ids)
                processed = completed + failed
                remaining = total_tasks - processed
                result["stats"] = {
                    "total_tasks": total_tasks,
                    "completed": completed,
                    "failed": failed,
                    "remaining": remaining,
                    "success_rate": (completed / total_tasks * 100) if total_tasks > 0 else 0
                }
                result["auto_recovered"] = True
                result["auto_recovered_batch"] = current_batch_id
                # Fall through to the next-batch / complete logic below
            except Exception as e:
                # If auto-recovery fails, report as in_progress so the agent
                # can fall back to ``run_batch.py --resume`` which re-runs the
                # batch and lets the orchestrator's own completion path finalise it.
                result["type"] = "in_progress"
                result["message"] = f"Batch in progress: {current_batch_id}"
                result["current_batch"] = {
                    "batch_id": current_batch_id,
                    "iteration": current_batch_state.get("iteration", 0),
                    "phase": phase,
                    "file_path": current_batch_state.get("file_path", ""),
                    "max_iterations": current_batch_state.get("max_iterations", 5),
                    "merged_mode": len(current_batch_state.get("merged_task_ids", [])) > 1,
                    "merged_task_count": len(current_batch_state.get("merged_task_ids", [])),
                }
                result["auto_recovery_error"] = str(e)
                result["next_action"] = (
                    f"Tests passed but auto-recovery failed ({e}). "
                    f"Run: {cmd_for('run_batch.py')} --resume --json to retry."
                )
                return result
        else:
            result["type"] = "in_progress"
            result["message"] = f"Batch in progress: {current_batch_id}"
            result["current_batch"] = {
                "batch_id": current_batch_id,
                "iteration": current_batch_state.get("iteration", 0),
                "phase": phase,
                "file_path": current_batch_state.get("file_path", ""),
                "max_iterations": current_batch_state.get("max_iterations", 5),
                "merged_mode": len(current_batch_state.get("merged_task_ids", [])) > 1,
                "merged_task_count": len(current_batch_state.get("merged_task_ids", [])),
            }
            if phase == "failed":
                result["next_action"] = (
                    f"Batch {current_batch_id} has failed. "
                    f"Run: {cmd_for('run_batch.py')} --retry {current_batch_id} --json "
                    f"to retry, or {cmd_for('run_batch.py')} --next --json to skip "
                    f"it and move on."
                )
            else:
                result["next_action"] = (
                    f"Resume the current batch (phase: {phase}). "
                    f"Run: {cmd_for('run_batch.py')} --resume --json"
                )
            result["workflow_hint"] = (
                "run_batch.py --resume dispatches a sub-agent that autonomously "
                "writes tests → code → runs pytest → fixes → repeats (up to 5x)"
            )
            return result
    
    # Find next batch
    next_batch = None
    for batch_id in all_task_ids:
        if batch_id not in completed_ids and batch_id not in failed_ids:
            next_batch = batch_id
            break
    
    if next_batch:
        result["type"] = "continue"
        result["message"] = f"Ready to continue ({remaining} tasks remaining)"
        result["next_batch"] = next_batch
        result["next_action"] = (
            f"Run: {cmd_for('run_batch.py')} --next --json "
            f"to start the next batch."
        )
        result["workflow_hint"] = (
            "run_batch.py --next dispatches a sub-agent that autonomously "
            "writes tests → code → runs pytest → fixes → repeats (up to 5x)"
        )
    else:
        result["type"] = "complete"
        if failed > 0:
            result["message"] = f"All tasks processed: {completed} completed, {failed} failed"
        else:
            result["message"] = f"All {completed} tasks completed successfully!"

        # Check stage files to determine which post-completion step is next
        logs_dir = Path(scripts).parent / "logs"
        ft_file = logs_dir / "codegen_final_test.json"
        gr_file = logs_dir / "codegen_global_review.json"

        ft_passed = False
        if ft_file.exists():
            try:
                ft_data = json.loads(ft_file.read_text(encoding="utf-8"))
                ft_passed = ft_data.get("success", False)
            except Exception:
                pass

        gr_passed = False
        if gr_file.exists():
            try:
                gr_data = json.loads(gr_file.read_text(encoding="utf-8"))
                gr_passed = gr_data.get("success", False)
            except Exception:
                pass

        if not ft_passed:
            result["next_action"] = (
                f"Run: {cmd_for('run_batch.py')} --final-test --json"
            )
        elif not gr_passed:
            result["next_action"] = (
                f"Final test passed. Run: {cmd_for('run_batch.py')} --global-review --json"
            )
        else:
            result["next_action"] = (
                "All steps complete (batches + final test + global review). "
                "Display the final summary to the user."
            )
        
        # Artifact verification: check that special task outputs actually exist.
        # This prevents false "complete" when tasks were marked done without
        # actually generating the expected files.
        missing_artifacts = []
        repo_root = REPO_DIR

        # Resolve the target language so entry-point / dependency artifact
        # checks are not hard-coded to Python's ``main.py`` /
        # ``requirements.txt``. Routes through the canonical repo resolver so
        # the language is inferred from the real on-disk sources when the rpg
        # metadata is missing, rather than silently assuming Python. Falls
        # back to Python on any failure so the check degrades to its previous
        # behaviour rather than crashing.
        backend = None
        try:
            from common.paths import REPO_RPG_FILE
            from decoder_lang import resolve_repo_backend
            rpg_obj = None
            if Path(REPO_RPG_FILE).is_file():
                rpg_obj = json.loads(Path(REPO_RPG_FILE).read_text(encoding="utf-8"))
            backend = resolve_repo_backend(repo_root, rpg_obj=rpg_obj)
        except Exception:  # noqa: BLE001 — degraded mode: assume Python
            backend = None

        # Check for main_entry task artifact (language-aware entry path).
        main_entry_ids = [tid for tid in completed_ids if tid.startswith("<MAIN_ENTRY>")]
        if main_entry_ids:
            if backend is not None:
                # Accept any of the backend's entry-point shapes. A single
                # canonical path is too strict when the skeleton placed the
                # entry off-canonical (e.g. C++ ``src/cli/main.cpp``) or the
                # language uses a glob convention (Go ``cmd/*/main.go``).
                candidates = backend.entry_point_candidates()
                entry_exists = any(
                    (any(repo_root.glob(c)) if "*" in c else (repo_root / c).exists())
                    for c in candidates
                )
                if not entry_exists:
                    missing_artifacts.append(
                        f"{candidates[0]} (from <MAIN_ENTRY> task)"
                    )
            elif not (repo_root / "main.py").exists():
                missing_artifacts.append("main.py (from <MAIN_ENTRY> task)")

        # Check for requirements task artifact. The dependency-manifest
        # filename is language-specific; only Python's is asserted here
        # (other languages manage deps via go.mod / Cargo.toml / package.json
        # which the dependency task and build steps validate separately).
        req_ids = [tid for tid in completed_ids if tid.startswith("<REQUIREMENTS>")]
        if req_ids and (backend is None or backend.name == "python"):
            if not (repo_root / "requirements.txt").exists():
                missing_artifacts.append("requirements.txt (from <REQUIREMENTS> task)")
        
        if missing_artifacts:
            result["type"] = "incomplete"
            result["missing_artifacts"] = missing_artifacts
            result["message"] = (
                f"All tasks marked complete but {len(missing_artifacts)} expected "
                f"artifact(s) missing: {', '.join(missing_artifacts)}"
            )
            result["next_action"] = (
                f"WARNING: The following files were expected but not found: "
                f"{', '.join(missing_artifacts)}. "
                f"These tasks may have been marked complete without actual generation. "
                f"Re-run the affected tasks or generate the files manually."
            )
    
    return result


def print_status(result: Dict[str, Any], json_output: bool = False) -> None:
    """Print the status in human-readable or JSON format."""
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    state_type = result["type"]
    
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              CODE GENERATION STATUS                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Status badge
    badges = {
        "error": "ERROR",
        "init": "READY TO START",
        "in_progress": "IN PROGRESS",
        "continue": "READY TO CONTINUE",
        "complete": "COMPLETE"
    }
    
    print(f"\n   Status: {badges.get(state_type, state_type)}")
    print(f"   {result['message']}")
    
    # Stats
    stats = result.get("stats", {})
    if stats:
        print("\n   Progress:")
        print(f"      -  Total tasks: {stats.get('total_tasks', 0)}")
        print(f"      -  Completed: {stats.get('completed', 0)}")
        print(f"      -  Failed: {stats.get('failed', 0)}")
        print(f"      -  Remaining: {stats.get('remaining', 0)}")
        if stats.get('success_rate'):
            print(f"      -  Success rate: {stats['success_rate']:.1f}%")
    
    # Current batch info
    if result.get("current_batch"):
        batch = result["current_batch"]
        print("\n   Current Batch:")
        print(f"      -  ID: {batch.get('batch_id', 'unknown')}")
        print(f"      -  File: {batch.get('file_path', 'unknown')}")
        print(f"      -  Phase: {batch.get('phase', 'unknown')}")
        print(f"      -  Iteration: {batch.get('iteration', 0)}/{batch.get('max_iterations', 5)}")
    
    # Next batch info
    if result.get("next_batch"):
        print(f"\n   Next Batch: {result['next_batch']}")
    
    # Guidance
    print("\n   " + "─" * 60)
    
    if state_type == "error":
        print("   Fix the errors above before proceeding.")
        print("   Run /cmind.plan_tasks to generate tasks.json")
    elif state_type == "init":
        print("   Run /cmind.code_gen to start code generation")
    elif state_type == "in_progress":
        print("   Run /cmind.code_gen to continue current batch")
    elif state_type == "continue":
        print("   Run /cmind.code_gen to process next batch")
    elif state_type == "complete":
        print("   All done! Review the generated code.")


def main():
    parser = argparse.ArgumentParser(
        description="Check code generation state"
    )
    parser.add_argument(
        "--tasks", "-t",
        type=Path,
        default=TASKS_FILE,
        help=f"Input tasks file (default: {TASKS_FILE})"
    )
    parser.add_argument(
        "--state", "-s",
        type=Path,
        default=STATE_FILE,
        help=f"Input state file (default: {STATE_FILE})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    result = determine_state(args.tasks, args.state)
    print_status(result, json_output=args.json)
    
    # Return exit code based on state
    if result["type"] == "error":
        return 1
    elif result["type"] == "complete":
        # Check if there were failures
        if result.get("stats", {}).get("failed", 0) > 0:
            return 2
    return 0


if __name__ == "__main__":
    exit(main())
