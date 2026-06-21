#!/usr/bin/env python3
"""Run Batch — Unified TDD batch executor.

Prepares a batch, dispatches a single sub-agent session that autonomously
completes the full write → test → fix cycle, then verifies and merges.

Architecture:
    run_batch.py (this file)
        ├── batch_prepare   — reuse code_gen.task_loader pickers
        ├── batch_prompt    — build TDD prompt for the sub-agent
        ├── batch_dispatch  — call LLMClient to dispatch sub-agent
        ├── batch_verify    — post-verification pytest run
        └── batch_complete  — merge branch, update state

Each batch gets at most 2 attempts (initial + one auto-retry).
Each attempt gives the sub-agent up to 5 internal TDD iterations.

Usage:
    python3 run_batch.py --next --json           # Next pending batch
    python3 run_batch.py --next --merge-file --json  # File-merge mode
    python3 run_batch.py --resume --json         # Resume interrupted batch
    python3 run_batch.py --retry <id> --json     # Retry a failed batch
    python3 run_batch.py --final-test --json     # Full repo validation (pytest + smoke)
    python3 run_batch.py --global-review --json  # Full feature review + visual QA (run after --final-test)
"""

import json
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ---------------------------------------------------------------------------
# Path setup — ensure scripts/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from common.execution_state import (
    BatchExecutionState,
    CodeGenState,
    load_code_gen_state,
    save_code_gen_state,
    get_or_create_code_gen_trajectory,
    complete_batch as state_complete_batch,
    skip_current_batch as state_skip_batch,
)
from common.generated_artifacts import ensure_generated_artifact_excludes
from common.git_utils import GitRunner
from common.task_batch import PlannedTask, get_task_by_id
from common.paths import (
    TASKS_FILE,
    INTERFACES_FILE,
    CODE_GEN_STATE_FILE as STATE_FILE,
    BASE_CLASSES_FILE,
    DATA_FLOW_FILE,
    DEP_GRAPH_FILE,
    REPO_RPG_FILE,
    FEATURE_SPEC_FILE,
    LOGS_DIR as _LOGS_DIR,
    WORKSPACE_ROOT,
    get_scripts_dir,
    cmd_for,
    REPO_DIR,
)
from code_gen.context_collector import build_dependency_context
from code_gen.prompts import (
    build_test_prompt_from_batch,
    build_code_prompt_from_batch,
    build_project_file_prompt_from_batch,
    build_merged_test_prompt,
    build_merged_code_prompt,
    is_project_file_batch,
    is_project_docs_batch,
    _format_dependency_context,
)
from code_gen.test_runner import (
    ensure_dev_venv,
    ensure_deps_installed,
    resolve_test_backend,
)
from code_gen.rpg_updater import run_rpg_update

# Git branch helpers extracted to code_gen.git_ops.  These are
# internal helpers used only by Module 5 ("Batch Orchestrator")
# below; no external surface contract.
from code_gen.git_ops import (
    ensure_on_main,
    setup_batch_branch,
    merge_batch_branch,
    abandon_batch_branch,
)

# Post-verification helper extracted to code_gen.post_verify.
from code_gen.post_verify import post_verify

# Result-dict builders extracted to code_gen.result_builders.  Internal
# helpers used only by Module 5's orchestrator.
from code_gen.result_builders import (
    _error,
    _all_done,
    _success_result,
    _failure_result,
)

# Final-test stage extracted to code_gen.final_validation.
from code_gen.final_validation import final_test

# Global-review stage extracted to code_gen.global_review.
from code_gen.global_review import global_review

# Per-batch TDD prompt builders extracted to code_gen.batch_prompts.
from code_gen.batch_prompts import (
    build_tdd_prompt,
    build_resume_prompt,
)

# Sub-agent dispatch (re-exported from code_gen.sub_agent).  External
# callers — ``code_gen.subtree_review``, ``rpg_edit.review``,
# ``rpg_edit.code`` — still do ``from run_batch import dispatch_sub_agent``;
# keep these names live at the module level for backwards compatibility.
# ``test_run_batch_preserves_external_surface`` guards this contract.
from code_gen.sub_agent import (  # noqa: F401
    dispatch_sub_agent,
    parse_batch_result,
    parse_pytest_summary,
    truncate_test_output,
)

# Task-picker helpers extracted to code_gen.task_loader.
from code_gen.task_loader import (
    get_next_pending_task_id,
    get_next_merged_tasks,
)
from smoke_test import run_smoke_test

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

from code_gen._constants import DEFAULT_AGENT_TIMEOUT  # noqa: E402

MAX_BATCH_ATTEMPTS = 2               # initial + 1 auto-retry

# Timeout values used inside the extracted modules
# (``code_gen.batch_prompts`` / ``.post_verify`` / ``.final_validation`` /
# ``.global_review``) live in ``code_gen._constants``; the orchestrator
# only needs the sub-agent timeout directly for its argparse default.


def _setup_codegen_environment(repo_path: Path) -> None:
    """Prepare the language-specific codegen environment."""
    backend = resolve_test_backend(repo_path=repo_path)
    if backend.name != "python":
        logger.info("Skipping Python venv setup for %s codegen", backend.display_name)
        return

    try:
        created_new, venv_path = ensure_dev_venv(repo_path)
        if created_new:
            logger.info("Created dev venv at %s", venv_path)
        ensure_deps_installed(repo_path)
    except Exception as exc:
        logger.warning("Venv setup issue (non-fatal): %s", exc)



# ============================================================================
# Module 1: Prompt Builder
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.batch_prompts``.  Imported above.
# ============================================================================


# ============================================================================
# Module 2: Git Operations
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.git_ops``.  Imported above; nothing to define here.
# ============================================================================


# ============================================================================
# Module 3: Sub-agent Dispatch
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.sub_agent``.  The names are re-exported below
# (see "Sub-agent dispatch (re-exported from code_gen.sub_agent)") so
# legacy callers ``from run_batch import dispatch_sub_agent`` still work.
# ============================================================================


# ============================================================================
# Module 4: Post-Verification
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.post_verify``.  Imported above.
# ============================================================================


# ============================================================================
# Module 5: Batch Orchestrator
# ============================================================================

def _prepare_batch_context(
    global_state: CodeGenState,
    task: PlannedTask,
    merged_tasks: Optional[List[PlannedTask]],
) -> Tuple[BatchExecutionState, Optional[Dict[str, Any]]]:
    """Build BatchExecutionState and dependency context for a task.

    Returns structured state rather than printing JSON, so the batch
    runner can reuse the prepared dependency context directly.

    Returns:
        (batch_state, dependency_context)
    """
    batch_state = BatchExecutionState(
        batch_id=task.task_id,
        file_path=task.file_path,
        subtree=task.subtree,
    )

    # Build dependency context
    dep_context = None
    try:
        dep_context = build_dependency_context(
            batch=task,
            completed_task_ids=global_state.completed_task_ids,
            tasks_path=TASKS_FILE,
            interfaces_path=INTERFACES_FILE,
            base_classes_path=BASE_CLASSES_FILE,
            data_flow_path=DATA_FLOW_FILE,
            feature_spec_path=FEATURE_SPEC_FILE,
        )
    except Exception as exc:
        logger.warning("Failed to build dependency context: %s", exc)

    is_merged = merged_tasks and len(merged_tasks) > 1

    # Generate prompts based on task type
    if is_project_file_batch(task):
        batch_state.test_prompt = ""
        batch_state.code_prompt = build_project_file_prompt_from_batch(task, dependency_context=dep_context)
        batch_state.skip_tests = is_project_docs_batch(task)
    elif is_merged:
        batch_state.merged_task_ids = [t.task_id for t in merged_tasks]
        batch_state.test_prompt = build_merged_test_prompt(merged_tasks, dependency_context=dep_context)
        batch_state.code_prompt = build_merged_code_prompt(merged_tasks, dependency_context=dep_context)
    elif task.task_type in ("integration_test", "final_test_docs"):
        batch_state.test_prompt = build_test_prompt_from_batch(task, dependency_context=dep_context)
        batch_state.code_prompt = build_code_prompt_from_batch(task, dependency_context=dep_context)
        # skip_code_gen stays False — agent can fix genuine integration bugs
    else:
        batch_state.test_prompt = build_test_prompt_from_batch(task, dependency_context=dep_context)
        batch_state.code_prompt = build_code_prompt_from_batch(task, dependency_context=dep_context)

    return batch_state, dep_context


def run_single_attempt(
    prompt: str,
    repo_path: Path,
    task: PlannedTask,
    attempt: int,
    agent_timeout: int = DEFAULT_AGENT_TIMEOUT,
    trajectory=None,
) -> Dict[str, Any]:
    """Execute a single sub-agent attempt and post-verify.

    Args:
        prompt: Full prompt for the sub-agent.
        repo_path: Project repo path.
        task: PlannedTask object.
        attempt: Attempt number (1 or 2).
        agent_timeout: Timeout for sub-agent.
        trajectory: Trajectory for recording.

    Returns:
        Dict with keys: passed, agent_passed, verify_passed,
        agent_error, failure_reason, test_output, duration.
    """
    step_id = None
    if trajectory:
        try:
            # Derive stage prefix from task_type for clear trajectory naming
            _stage_map = {
                "integration_test": "gen_test",
                "final_test_docs": "gen_test",
                "main_entry": "gen_code",
                "project_requirements": "gen_code",
                "project_docs": "gen_code",
                "implementation": "gen_code",
            }
            stage = _stage_map.get(task.task_type, "gen_code")
            step = trajectory.add_step(
                f"{stage}_{task.task_id}_attempt{attempt}",
                f"Sub-agent attempt {attempt}",
            )
            trajectory.start_step(step.step_id)
            step_id = step.step_id
        except Exception:
            pass

    start = time.time()
    result = {
        "attempt": attempt,
        "agent_passed": False,
        "verify_passed": False,
        "passed": False,
        "agent_error": None,
        "failure_reason": "",
        "test_output": "",
        "agent_pytest_summary": None,
        "duration": 0.0,
    }

    # --- Dispatch sub-agent ---
    response, error = dispatch_sub_agent(
        prompt, repo_path,
        timeout=agent_timeout,
        trajectory=trajectory,
        step_id=step_id,
        purpose="run_batch",
        max_retries=3,
    )

    if error:
        result["agent_error"] = error
        result["failure_reason"] = f"Sub-agent error: {error}"
        result["duration"] = time.time() - start
        return result

    # --- Parse sub-agent's self-report ---
    agent_passed, agent_reason = parse_batch_result(response)
    agent_summary = parse_pytest_summary(response)
    result["agent_passed"] = agent_passed
    result["agent_pytest_summary"] = agent_summary
    if not agent_passed:
        result["failure_reason"] = agent_reason
        logger.info("Sub-agent self-reported FAIL: %s", agent_reason)
    elif agent_summary is None and not is_project_docs_batch(task):
        # PASS without the required PYTEST_SUMMARY line is suspicious for a
        # test-bearing task; log it so post_verify_failure analysis is easier.
        # Docs/entry batches (README, requirements) run no tests and are
        # post-verified by skip, so a missing summary is expected there.
        logger.warning(
            "Sub-agent reported PASS but did not provide PYTEST_SUMMARY line"
        )

    # --- Post-verification (authoritative) ---
    verify_passed, test_output = post_verify(repo_path, task)
    result["verify_passed"] = verify_passed
    result["test_output"] = test_output
    result["passed"] = verify_passed  # Post-verify is the authority

    if verify_passed and not agent_passed:
        logger.info("Sub-agent reported FAIL but post-verification PASSED — treating as success")
        result["failure_reason"] = ""
    elif not verify_passed and agent_passed:
        logger.warning(
            "Sub-agent reported PASS (PYTEST_SUMMARY=%r) but post-verification FAILED",
            agent_summary,
        )
        first_line = test_output.splitlines()[0] if test_output.strip() else "no output"
        result["failure_reason"] = (
            f"Post-verification rejected sub-agent's PASS claim "
            f"(its PYTEST_SUMMARY={agent_summary!r}); pytest re-run says: {first_line}"
        )
    elif not verify_passed:
        result["failure_reason"] = agent_reason

    result["duration"] = time.time() - start

    # Complete trajectory step
    if trajectory and step_id:
        try:
            trajectory.complete_step(step_id, {
                "attempt": attempt,
                "passed": result["passed"],
                "duration": result["duration"],
            })
        except Exception:
            pass

    return result


def run_rpg_update_safe(
    task: PlannedTask,
    repo_path: Path,
    global_state: CodeGenState,
) -> Optional[str]:
    """Run RPG update, logging but not raising on failure.

    Returns:
        rpg_backup_path if a new backup was created, else None.
    """
    if task.task_type in (
        "integration_test", "final_test_docs", "main_entry",
        "project_requirements", "project_docs",
    ) or (task.file_path.startswith("<") and task.file_path.endswith(">")):
        logger.info("Skipping RPG update for %s task", task.task_type)
        return None
    try:
        should_backup = global_state.rpg_backup_path is None
        rpg_result = run_rpg_update(
            batch=task,
            repo_path=repo_path,
            rpg_path=REPO_RPG_FILE,
            backup=should_backup,
        )
        logger.info("RPG update: edges_added=%s", rpg_result.get("edges_added", 0))
        if should_backup and rpg_result.get("backup_path"):
            return rpg_result["backup_path"]
    except Exception as exc:
        logger.warning("RPG update failed (non-fatal): %s", exc)
    return None


def _refresh_dep_graph_safe(
    repo_path: Path,
    changed_files: Optional[List[str]] = None,
) -> None:
    """Refresh dep_graph after code changes (non-fatal on error).

    Strategy:
      * If ``changed_files`` is provided (typical codegen path: a single
        file just got generated/edited), use the incremental
        ``RPGService.sync_from_file_list`` path so we only re-AST the
        touched file.  This is the ~10× speed-up codegen benefits from.
      * If ``changed_files`` is empty / ``None`` (e.g. integration-test
        batches that don't have a single owning file), fall back to a
        full ``refresh_dep_graph`` so the graph still stays correct.

    The codegen pipeline does its own commit hygiene (each batch lands
    on its own git branch then merges), so this entry point intentionally
    does NOT advance ``meta.git`` — that's owned by the pre-commit /
    post-merge hooks and ``/cmind.update_rpg``.
    """
    try:
        import sys
        scripts_dir = Path(get_scripts_dir())
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from rpg.service import RPGService

        rpg_path = REPO_RPG_FILE
        if not rpg_path.exists():
            return

        svc = RPGService.load(str(rpg_path))

        # ── Incremental path: codegen knows exactly which file changed ──
        if changed_files:
            # Keep only files lang_parser can build dep edges for. This spans
            # every supported language (py/go/rs/ts/js/c/cpp), so non-Python
            # projects keep an up-to-date dep_graph across batches too.
            from lang_parser import is_supported_source

            source_files = [f for f in changed_files if is_supported_source(f)]
            if not source_files:
                # No analysable source touched (e.g. only docs/config edits).
                logger.info("dep_graph: no supported source files in batch, skipping refresh")
                svc.save(str(rpg_path))
                return

            # ``save_path=None``: dep_graph rides inside rpg.json. The
            # subsequent ``svc.save(rpg_path)`` embeds it.
            result = svc.sync_from_file_list(
                file_paths=source_files,
                code_dir=str(repo_path),
                workspace_root=str(WORKSPACE_ROOT),
            )
            svc.rpg._dep_graph_file = None
            svc.save(str(rpg_path))
            logger.info(
                "dep_graph refreshed (mode=%s reason=%s): %d nodes, %d dep→rpg mappings",
                result.get("mode"), result.get("reason"),
                len(svc.rpg.dep_graph.G.nodes()),
                len(svc.rpg._dep_to_rpg_map),
            )
            return

        # ── Fallback: full rebuild ──
        svc.refresh_dep_graph(
            str(repo_path),
            workspace_root=str(WORKSPACE_ROOT),
        )
        svc.rpg._dep_graph_file = None
        svc.save(str(rpg_path))
        logger.info("dep_graph refreshed (full): %d nodes, %d dep→rpg mappings",
                    len(svc.rpg.dep_graph.G.nodes()),
                    len(svc.rpg._dep_to_rpg_map))
    except Exception as exc:
        logger.warning("dep_graph refresh failed (non-fatal): %s", exc)


def _task_files_for_dep_graph(task: PlannedTask) -> Optional[List[str]]:
    """Return the list of files to pass to ``_refresh_dep_graph_safe``.

    Returns ``None`` for batches where the file set is ambiguous or
    irrelevant (integration tests, docs, project files), so the caller
    falls back to a full refresh.  This mirrors the same skip criteria
    used by ``run_rpg_update_safe``.
    """
    if task.task_type in (
        "integration_test", "final_test_docs", "main_entry",
        "project_requirements", "project_docs",
    ):
        return None
    # Marker paths like ``<INTEGRATION_TEST>`` aren't real files.
    if task.file_path.startswith("<") and task.file_path.endswith(">"):
        return None
    return [task.file_path]


def run_batch(
    batch_id: Optional[str] = None,
    next_batch: bool = False,
    resume: bool = False,
    retry: Optional[str] = None,
    merge_file: bool = False,
    max_units: int = 0,
    agent_timeout: int = DEFAULT_AGENT_TIMEOUT,
    tasks_path: Path = TASKS_FILE,
    state_path: Path = STATE_FILE,
    repo_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main batch orchestrator.

    Coordinates the full lifecycle of a single batch:
      1. Determine which batch to run
      2. Create git branch from main
      3. Setup venv and install deps
      4. Build prompt and dispatch sub-agent
      5. Post-verify with pytest
      6. On failure: auto-retry once with resume context
      7. Merge branch on success / abandon on failure
      8. Update state and report

    Args:
        batch_id: Specific batch ID to run.
        next_batch: Run the next pending batch.
        resume: Resume an interrupted batch.
        retry: Retry a specific failed batch ID.
        merge_file: Enable file-merge mode.
        max_units: Max units per merged batch (0 = no limit).
        agent_timeout: Sub-agent timeout in seconds.
        tasks_path: Path to tasks.json.
        state_path: Path to code_gen_state.jsonl.
        repo_path: Repo root (default: from paths.py).

    Returns:
        JSON-serializable result dict.
    """
    repo_path = repo_path or REPO_DIR
    scripts = get_scripts_dir()
    global_state = load_code_gen_state(state_path)
    git = GitRunner(str(repo_path))

    # ── Step 1: Determine batch ─────────────────────────────────────

    merged_tasks = None

    if retry:
        batch_id = retry
        # Remove from failed list so it can be retried
        if batch_id in global_state.failed_task_ids:
            global_state.failed_task_ids.remove(batch_id)
            global_state.failed_tasks = len(global_state.failed_task_ids)
            save_code_gen_state(global_state, state_path)
        logger.info("Retrying batch: %s", batch_id)

    elif resume:
        if not global_state.current_batch_id:
            return _error("No batch in progress to resume", scripts)
        batch_id = global_state.current_batch_id
        logger.info("Resuming batch: %s", batch_id)

    elif next_batch:
        if merge_file:
            merged_tasks = get_next_merged_tasks(
                tasks_path, global_state, max_units,
                repo_path=repo_path, state_path=state_path,
            )
            if not merged_tasks:
                return _all_done(global_state, tasks_path, scripts)
            batch_id = merged_tasks[0].task_id
        else:
            batch_id = get_next_pending_task_id(
                tasks_path, global_state,
                repo_path=repo_path, state_path=state_path,
            )
            if not batch_id:
                return _all_done(global_state, tasks_path, scripts)
        logger.info("Next batch: %s (merge_file=%s)", batch_id, merge_file)

    elif batch_id:
        logger.info("Running specific batch: %s", batch_id)

    else:
        return _error("No batch specified. Use --next, --resume, --retry, or --batch-id.", scripts)

    # ── Step 2: Load task ────────────────────────────────────────────

    task = get_task_by_id(tasks_path, batch_id)
    if not task:
        return _error(f"Task '{batch_id}' not found in tasks.json", scripts)

    if batch_id in global_state.completed_task_ids:
        return _error(f"Task '{batch_id}' is already completed", scripts)

    logger.info(
        "Batch: id=%s file=%s type=%s units=%s",
        batch_id, task.file_path, task.task_type, task.units_key,
    )

    # ── Step 3: Setup git branch ─────────────────────────────────────

    reuse_branch = bool(retry) or resume
    try:
        branch_ok, branch_name, initial_commit = setup_batch_branch(
            git, batch_id, repo_path, reuse_existing=reuse_branch,
        )
    except RuntimeError as exc:
        return _error(f"Git setup failed: {exc}", scripts)

    if not branch_ok:
        return _error(f"Failed to create branch for batch '{batch_id}'", scripts)

    logger.info("Branch: %s (initial_commit=%s)", branch_name, initial_commit[:8] if initial_commit else "none")
    ensure_generated_artifact_excludes(repo_path)

    # ── Step 4: Setup language environment ──────────────────────────

    _setup_codegen_environment(repo_path)

    # ── Step 5: Build prompts ────────────────────────────────────────

    # Track whether we entered resume-but-not-yet-passing mode
    _resume_check_output = ""

    # For resume mode, try to recover existing state first
    if resume and global_state.current_batch_state:
        batch_state = BatchExecutionState.from_dict(global_state.current_batch_state)
        dep_context = None
        try:
            dep_context = build_dependency_context(
                batch=task,
                completed_task_ids=global_state.completed_task_ids,
                tasks_path=TASKS_FILE,
                interfaces_path=INTERFACES_FILE,
                base_classes_path=BASE_CLASSES_FILE,
                data_flow_path=DATA_FLOW_FILE,
                feature_spec_path=FEATURE_SPEC_FILE,
            )
        except Exception as exc:
            logger.warning("Failed to build dependency context: %s", exc)

        # Check if batch already passes (sub-agent may have finished before interrupt)
        logger.info("Resume: checking if batch already passes...")
        already_passed, check_output = post_verify(repo_path, task)
        if already_passed:
            logger.info("Resume: batch already passes! Completing directly.")
            rpg_backup = run_rpg_update_safe(task, repo_path, global_state)
            _refresh_dep_graph_safe(
                repo_path,
                changed_files=_task_files_for_dep_graph(task),
            )
            merge_ok, merge_error = merge_batch_branch(
                git, branch_name, batch_id,
                file_path=task.file_path, units=task.units_key,
            )
            if not merge_ok:
                # Ensure we're on main; clear batch state so --retry works
                try:
                    ensure_on_main(git)
                except RuntimeError:
                    pass
                if merge_error == "branch_missing":
                    # Sub-agent didn't use the batch branch — skip without
                    # consuming a retry slot. The helper
                    # promotes to failed after _MAX_BATCH_PREPARES skips.
                    skipped = state_skip_batch(batch_id, state_path)
                    if skipped:
                        return _error(
                            f"Batch '{batch_id}' skipped: branch '{branch_name}' "
                            f"was not created. Re-run --next to retry.",
                            scripts,
                        )
                    return _error(
                        f"Batch '{batch_id}' kept skipping (sub-agent never "
                        f"used the batch branch); promoted to failed. "
                        f"Investigate why, then `--retry {batch_id}` to try again.",
                        scripts,
                    )
                state_complete_batch(batch_id, False, state_path)
                return _error(
                    f"Tests pass but branch merge failed: {merge_error}. "
                    f"Branch '{branch_name}' preserved. "
                    f"Retry: {cmd_for('run_batch.py')} --retry {batch_id} --json",
                    scripts,
                )
            state_complete_batch(batch_id, True, state_path, rpg_backup_path=rpg_backup)
            return _success_result(
                batch_id, task, batch_state, [{"attempt": 0, "passed": True, "duration": 0}],
                0.0, branch_merged=True, scripts=scripts,
                tasks_path=tasks_path, state_path=state_path,
            )
        # Tests didn't pass — will proceed to attempt loop with resume prompt
        _resume_check_output = check_output
    else:
        batch_state, dep_context = _prepare_batch_context(
            global_state, task, merged_tasks,
        )
        batch_state.branch_name = branch_name
        batch_state.initial_commit = initial_commit
        batch_state.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        batch_state.start_iteration()

    # Save state (for resume capability)
    global_state.set_current_batch(batch_state)
    save_code_gen_state(global_state, state_path)

    tdd_prompt = build_tdd_prompt(
        batch_state, task, repo_path,
        merged_tasks=merged_tasks,
        dependency_context=dep_context,
    )

    # Trajectory
    trajectory = None
    try:
        trajectory = get_or_create_code_gen_trajectory(
            global_state, base_dir=repo_path, state_path=state_path,
        )
    except Exception:
        pass

    # ── Step 6: Attempt loop ─────────────────────────────────────────

    attempts = []
    final_passed = False

    # For resume mode where tests didn't pass in the early check,
    # start with a resume prompt instead of a fresh one
    if _resume_check_output:
        current_prompt = build_resume_prompt(
            original_prompt=tdd_prompt,
            attempt_number=1,
            failure_reason="Resumed after interruption",
            last_test_output=_resume_check_output,
        )
    else:
        current_prompt = tdd_prompt

    for attempt_num in range(1, MAX_BATCH_ATTEMPTS + 1):
        logger.info("━━━ Attempt %d/%d for batch %s ━━━", attempt_num, MAX_BATCH_ATTEMPTS, batch_id)

        attempt_result = run_single_attempt(
            current_prompt, repo_path, task, attempt_num,
            agent_timeout=agent_timeout,
            trajectory=trajectory,
        )
        attempts.append(attempt_result)

        if attempt_result["passed"]:
            final_passed = True
            logger.info("[OK] Batch PASSED on attempt %d", attempt_num)
            break

        # Prepare resume prompt for next attempt
        if attempt_num < MAX_BATCH_ATTEMPTS:
            logger.info("[FAIL] Attempt %d failed, preparing retry...", attempt_num)
            # If the agent self-reported PASS but post-verify caught the
            # failure, surface that explicitly in the next prompt so the
            # agent doesn't repeat the false-positive pattern (A1 + F2).
            sub_agent_claimed_pass = (
                attempt_result["agent_passed"]
                and not attempt_result["verify_passed"]
            )
            current_prompt = build_resume_prompt(
                original_prompt=tdd_prompt,
                attempt_number=attempt_num + 1,
                failure_reason=attempt_result["failure_reason"],
                last_test_output=attempt_result["test_output"],
                sub_agent_claimed_pass=sub_agent_claimed_pass,
                agent_pytest_summary=attempt_result.get("agent_pytest_summary"),
            )
        else:
            logger.info("[FAIL] All %d attempts exhausted for batch %s", MAX_BATCH_ATTEMPTS, batch_id)

    # ── Step 7: Finalize ─────────────────────────────────────────────

    total_duration = sum(a["duration"] for a in attempts)

    if final_passed:
        # RPG update
        rpg_backup = run_rpg_update_safe(task, repo_path, global_state)
        _refresh_dep_graph_safe(
            repo_path,
            changed_files=_task_files_for_dep_graph(task),
        )

        # Merge branch
        merge_ok, merge_error = merge_batch_branch(
            git, branch_name, batch_id,
            file_path=task.file_path, units=task.units_key,
        )
        if not merge_ok:
            logger.error("Branch merge failed: %s", merge_error)
            # Ensure we're on main; clear batch state so --retry works
            try:
                ensure_on_main(git)
            except RuntimeError:
                pass
            if merge_error == "branch_missing":
                # Sub-agent didn't use the batch branch — skip without
                # consuming a retry slot. The helper
                # promotes to failed after _MAX_BATCH_PREPARES skips.
                skipped = state_skip_batch(batch_id, state_path)
                if skipped:
                    return _error(
                        f"Batch '{batch_id}' skipped: branch '{branch_name}' "
                        f"was not created. Re-run --next to retry.",
                        scripts,
                    )
                return _error(
                    f"Batch '{batch_id}' kept skipping (sub-agent never "
                    f"used the batch branch); promoted to failed. "
                    f"Investigate why, then `--retry {batch_id}` to try again.",
                    scripts,
                )
            state_complete_batch(batch_id, False, state_path)
            return _error(
                f"Tests passed but branch merge failed: {merge_error}. "
                f"Branch '{branch_name}' preserved. "
                f"Retry: {cmd_for('run_batch.py')} --retry {batch_id} --json",
                scripts,
            )

        # Update state
        state_complete_batch(batch_id, True, state_path, rpg_backup_path=rpg_backup)

        # ── Subtree review: check if the subtree just completed ──
        try:
            from code_gen.subtree_review import is_subtree_just_completed, run_subtree_review

            # Reload state to get the freshly-updated completed_task_ids
            fresh_state = load_code_gen_state(state_path)
            completed_subtree = is_subtree_just_completed(
                batch_id, fresh_state.completed_task_ids, tasks_path,
            )
            if completed_subtree:
                logger.info(
                    "━━━ Subtree '%s' complete — running review ━━━",
                    completed_subtree,
                )
                review_result = run_subtree_review(
                    subtree_name=completed_subtree,
                    completed_task_ids=fresh_state.completed_task_ids,
                    repo_path=repo_path,
                    tasks_path=tasks_path,
                    agent_timeout=agent_timeout,
                )
                logger.info(
                    "Review result for '%s': %s (%.1fs)",
                    completed_subtree,
                    review_result.status,
                    review_result.duration,
                )
                # Persist review result
                fresh_state.subtree_reviews[completed_subtree] = review_result.to_dict()
                save_code_gen_state(fresh_state, state_path)
        except Exception as exc:
            logger.warning("Subtree review failed (non-blocking): %s", exc)
            # Ensure we're back on main after any review failure
            try:
                ensure_on_main(git)
            except RuntimeError:
                pass

        return _success_result(
            batch_id, task, batch_state, attempts, total_duration,
            branch_merged=True, scripts=scripts, tasks_path=tasks_path,
            state_path=state_path,
        )
    else:
        # Mark failed, preserve branch
        abandon_batch_branch(git, branch_name)
        state_complete_batch(batch_id, False, state_path)

        return _failure_result(
            batch_id, task, batch_state, attempts, total_duration,
            scripts=scripts, tasks_path=tasks_path, state_path=state_path,
        )


# ============================================================================
# Module 6: Final Test
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.final_validation``.  Imported above.
# ============================================================================


# ============================================================================
# Module 6b: Global Review
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.global_review``.  Imported below; nothing here.
# ============================================================================


# ============================================================================
# Module 7: Result Builders
# ----------------------------------------------------------------------------
# Extracted to ``code_gen.result_builders``.  Imported above.
# ============================================================================


# ============================================================================
# CLI
# ============================================================================

def print_result(result: Dict[str, Any], json_output: bool = False) -> None:
    """Print result to stdout and log it."""
    # Always log the result as JSON for the file log
    logger.info("Batch result: %s", json.dumps(result, indent=2))

    if json_output:
        print(json.dumps(result, indent=2))
        return

    success = result.get("success", False)
    rtype = result.get("type", "")

    if rtype == "final_test":
        icon = "[OK]" if success else "[FAIL]"
        print(f"\n  {icon} Final Test: passed={result.get('passed',0)} "
              f"failed={result.get('failed',0)} errors={result.get('errors',0)}")
    elif rtype == "complete":
        print(f"\n  [END] {result.get('message', '')}")
    elif rtype == "batch_complete":
        print(f"\n  [OK] Batch {result.get('batch_id','')} completed "
              f"({result.get('attempts_used',0)} attempt(s), "
              f"{result.get('total_duration',0):.1f}s)")
    elif rtype == "batch_failed":
        print(f"\n  [FAIL] Batch {result.get('batch_id','')} failed "
              f"({result.get('attempts_used',0)} attempt(s))")
        print(f"     Reason: {result.get('failure_reason','')}")
    else:
        icon = "[OK]" if success else "[FAIL]"
        msg = result.get("message", result.get("error", ""))
        print(f"\n  {icon} {msg}")

    if "stats" in result:
        s = result["stats"]
        print(f"  Progress: {s.get('completed',0)}/{s.get('total',0)} completed, "
              f"{s.get('failed',0)} failed")

    if "next_action" in result:
        print(f"\n   ->  {result['next_action']}")


def main() -> int:
    # Convert SIGTERM → SystemExit so "except BaseException" in Popen calls
    # triggers killpg cleanup instead of the process being silently killed.
    # Install before argparse so the handler is active as early as possible.
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(1))

    parser = argparse.ArgumentParser(
        description="Run Batch — unified TDD batch executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--next", action="store_true", help="Run the next pending batch")
    mode.add_argument("--loop", action="store_true",
                      help="Run ALL pending batches sequentially (recommended)")
    mode.add_argument("--resume", action="store_true", help="Resume an interrupted batch")
    mode.add_argument("--retry", metavar="BATCH_ID", help="Retry a specific failed batch")
    mode.add_argument("--batch-id", metavar="ID", help="Run a specific batch by ID")
    mode.add_argument("--final-test", action="store_true",
                      help="Run full repo test suite (pytest + smoke, no global review)")
    mode.add_argument("--smoke-test", action="store_true", help="Run post-codegen smoke tests")
    mode.add_argument("--global-review", action="store_true",
                      help="Run global feature review + repair (standalone)")
    mode.add_argument("--prune-failed", action="store_true",
                      help="Delete all preserved failed batch/* branches (cleanup)")

    parser.add_argument("--merge-file", action="store_true",
                        help="File-merge mode: group same-file tasks into one batch")
    parser.add_argument("--max-units", type=int, default=0,
                        help="Max units per merged batch (0 = no limit)")
    parser.add_argument("--agent-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT,
                        help=f"Sub-agent timeout in seconds (default: {DEFAULT_AGENT_TIMEOUT})")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Stop --loop after this many batches (0 = no limit)")
    parser.add_argument("--review-iterations", type=int, default=10,
                        help="Max iterations for global review (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if not args.json else logging.WARNING
    logging.basicConfig(
        level=logging.DEBUG,  # root logger accepts all; handlers filter
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler: respect --json (WARNING) vs interactive (DEBUG)
    root_logger = logging.getLogger()
    # basicConfig already added a StreamHandler; adjust its level
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)

    # File handler: capture DEBUG records to .cmind/logs/code_gen.log via
    # the shared helper (idempotent; degrades gracefully on read-only FS).
    from common.logging_setup import setup_file_logging
    setup_file_logging("code_gen")

    if args.final_test:
        result = final_test()
        print_result(result, json_output=args.json)
        return 0 if result.get("success") else 1

    if args.smoke_test:
        smoke_result = run_smoke_test()
        result = smoke_result.to_dict()
        print_result(result, json_output=args.json)
        return 0 if result.get("success") else 1

    if args.global_review:
        result = global_review(
            max_iterations=args.review_iterations,
            timeout_per_iteration=args.agent_timeout,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            icon = "[OK]" if result.get("success") else "[FAIL]"
            iters = len(result.get("iterations", []))
            print(f"\n  {icon} Global Review: {iters} iteration(s), "
                  f"{result.get('total_duration', 0):.1f}s")
            for it in result.get("iterations", []):
                detail = it.get("detail", it.get("error", it.get("reason", "")))
                it_icon = "[OK]" if it.get("review_passed") else "[FAIL]"
                print(f"    {it_icon} Iteration {it.get('iteration', '?')}: {detail}")
        return 0 if result.get("success") else 1

    if args.prune_failed:
        repo_path = REPO_DIR
        git = GitRunner(str(repo_path))
        import subprocess as _sp
        raw = _sp.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=repo_path, capture_output=True, text=True,
        )
        all_branches = [b.strip() for b in raw.stdout.splitlines() if b.strip()]
        current = git.get_current_branch()
        deleted = []
        errors = []
        for branch in all_branches:
            if not branch.startswith("batch/"):
                continue
            if branch == current:
                continue
            try:
                git.delete_branch(branch, force=True)
                deleted.append(branch)
                logger.info("Pruned failed branch: %s", branch)
            except Exception as exc:
                errors.append(f"{branch}: {exc}")
        result = {
            "success": not errors,
            "deleted": deleted,
            "errors": errors,
            "type": "prune_failed",
        }
        print_result(result, json_output=args.json)
        return 0 if not errors else 1

    if args.loop:
        return _run_loop(args)

    result = run_batch(
        batch_id=args.batch_id,
        next_batch=args.next,
        resume=args.resume,
        retry=args.retry,
        merge_file=args.merge_file,
        max_units=args.max_units,
        agent_timeout=args.agent_timeout,
    )

    print_result(result, json_output=args.json)
    return 0 if result.get("success") else 1


def _run_loop(args) -> int:
    """Run all pending batches sequentially until done or interrupted.

    Prints a summary line after each batch. Ctrl+C stops gracefully
    between batches (current batch completes first).
    """
    batch_num = 0
    total_passed = 0
    total_failed = 0
    start_time = time.time()
    max_batches = max(0, int(args.max_batches or 0))

    print("\n  [START] Starting batch loop (Ctrl+C to stop after current batch)\n")

    try:
        while True:
            if max_batches and batch_num >= max_batches:
                elapsed = time.time() - start_time
                print(f"\n  [STOP] Reached max-batches={max_batches} "
                      f"({total_passed} passed, {total_failed} failed, "
                      f"{elapsed/60:.1f} min)")
                logger.info("Loop stopped after max-batches=%d", max_batches)
                return 0 if total_failed == 0 else 1

            batch_num += 1

            result = run_batch(
                next_batch=True,
                merge_file=args.merge_file,
                max_units=args.max_units,
                agent_timeout=args.agent_timeout,
            )

            rtype = result.get("type", "")

            # All done
            if rtype == "complete":
                print_result(result, json_output=args.json)
                elapsed = time.time() - start_time
                print(f"\n  [TIME]  Total time: {elapsed/60:.1f} min "
                      f"({total_passed} passed, {total_failed} failed)")
                return 0

            # Batch completed or failed — log and continue
            # Always log full result to file
            logger.info("Batch result: %s", json.dumps(result, indent=2))

            if rtype == "batch_complete":
                total_passed += 1
                stats = result.get("stats", {})
                print(f"  [OK] [{batch_num}] {result.get('batch_id','')} — "
                      f"PASS ({result.get('attempts_used',0)} attempt(s), "
                      f"{result.get('total_duration',0):.0f}s) — "
                      f"{stats.get('completed',0)}/{stats.get('total',0)} done")
            elif rtype == "batch_failed":
                total_failed += 1
                stats = result.get("stats", {})
                print(f"  [FAIL] [{batch_num}] {result.get('batch_id','')} — "
                      f"FAIL: {result.get('failure_reason','')[:80]} — "
                      f"{stats.get('completed',0)}/{stats.get('total',0)} done")
            else:
                # Error or unexpected — print and stop
                print_result(result, json_output=args.json)
                return 1 if not result.get("success") else 0
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n  [WARNING]  Interrupted after {batch_num} batches "
              f"({total_passed} passed, {total_failed} failed, "
              f"{elapsed/60:.1f} min)")
        logger.info("Loop interrupted by user after %d batches", batch_num)
        return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  [WARNING]  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        # Try to log to file even if main() setup failed
        try:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LOGS_DIR / "code_gen.log", "a", encoding="utf-8") as _f:
                _f.write(f"\nUNHANDLED EXCEPTION:\n{tb}\n")
        except Exception:
            pass
        print(json.dumps({
            "success": False,
            "error": str(exc),
            "traceback": tb,
        }, indent=2))
        sys.exit(1)
