#!/usr/bin/env python3
"""Execution State Management for CoderMind Code Generation.

Manages the state of code generation execution, including:
- Current batch being processed
- Iteration tracking within a batch
- Failure history and analysis
- Git commit tracking

This module handles state persistence between command invocations,
which is essential since CoderMind uses multiple CLI sessions.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from .paths import CODE_GEN_STATE_FILE as STATE_FILE
from .trajectory import Trajectory
from .paths import WORKSPACE_ROOT


# ============================================================================
# Enums
# ============================================================================

class WorkflowPhase(str, Enum):
    """Current phase in the TDD workflow.

    The code generation process follows a fixed phase progression:
    INIT → TEST_GEN → CODE_GEN → TESTING → ANALYZING → COMPLETE/FAILED.
    """
    INIT = "init"                   # Initial state, not started
    TEST_GEN = "test_gen"           # Generating tests
    CODE_GEN = "code_gen"           # Generating implementation
    TESTING = "testing"             # Running tests
    ANALYZING = "analyzing"         # Analyzing failure
    COMPLETE = "complete"           # Batch completed successfully
    FAILED = "failed"               # Batch failed after max iterations


class FailureType(str, Enum):
    """Classification of failures during code generation.

    Used by the failure analysis step to decide whether to
    regenerate tests, fix implementation code, or adjust the environment.
    """
    TEST_ERROR = "test_error"       # Test itself is wrong
    CODE_ERROR = "code_error"       # Code implementation is wrong
    ENV_ERROR = "env_error"         # Environment/setup issues
    UNKNOWN_ERROR = "unknown_error" # Unknown/unclassified error


class WorkflowType(str, Enum):
    """Workflow classification for commit messages and progress tracking.

    Each TDD iteration is labeled with a workflow type so that
    git history and trajectory logs are easy to filter.
    """
    TEST_DEVELOPMENT = "test_development"
    TEST_FIX = "test_fix"
    CODE_INCREMENTAL = "code_incremental"
    CODE_BUG_FIX = "code_bug_fix"
    ENV_SETUP = "env_setup"
    # Legacy support (aliases)
    TEST_GENERATION = "test_development"
    CODE_GENERATION = "code_incremental"


# ============================================================================
# Iteration Record
# ============================================================================

@dataclass
class IterationRecord:
    """Record of a single iteration attempt."""
    iteration: int
    timestamp: str
    phase: str
    test_generated: bool = False
    code_generated: bool = False
    test_passed: bool = False
    failure_type: Optional[str] = None
    failure_analysis: Optional[str] = None
    test_output: Optional[str] = None
    commits: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        return cls(**data)


# ============================================================================
# Batch Execution State
# ============================================================================

@dataclass
class BatchExecutionState:
    """State of execution for a single batch.
    
    This is persisted between command invocations to maintain
    continuity across the TDD iteration loop.
    """
    # Batch identification
    batch_id: str
    file_path: str
    subtree: str
    
    # Iteration control
    iteration: int = 0
    max_iterations: int = 5
    phase: str = WorkflowPhase.INIT.value
    
    # Current iteration state
    test_prompt: str = ""           # Current test generation prompt
    code_prompt: str = ""           # Current code generation prompt
    test_generated: bool = False    # Test code generated this iteration
    code_generated: bool = False    # Implementation code generated this iteration
    
    # Failure tracking (for next iteration)
    last_test_output: str = ""
    last_failure_type: Optional[str] = None
    last_failure_analysis: str = ""
    failure_history: List[str] = field(default_factory=list)
    
    # Git tracking
    branch_name: str = ""
    initial_commit: str = ""        # Commit when batch started
    current_commit: str = ""        # Latest commit
    commits_this_batch: List[str] = field(default_factory=list)
    
    # Iteration history
    iterations: List[Dict] = field(default_factory=list)
    
    # Timestamps
    started_at: str = ""
    completed_at: str = ""
    
    # Agent call tracking
    last_agent_result: Optional[Dict[str, Any]] = None
    agent_call_count: int = 0

    # Merged task tracking (file-level merge mode)
    # Contains all original task IDs when multiple tasks are merged into one batch.
    # Empty list means single-task mode.
    merged_task_ids: List[str] = field(default_factory=list)
    
    # Phase skip flags
    skip_tests: bool = False        # Skip test running (e.g., documentation batches)
    skip_code_gen: bool = False     # Skip code generation (e.g., integration test batches)

    # Pending test fix (for code_then_test flow)
    # When the failure-analysis pass determines both code and test need fixing,
    # it sets phase=code_gen first and stores the test fix plan here.
    # After code-gen + post-verify, if pending_test_fix is True, the main
    # loop should go directly to test-fix (skip analyse-failure).
    pending_test_fix: bool = False
    pending_test_fix_plan: str = ""

    # Structured test output analysis (from code_gen.test_output_parser).
    # Stored as dict so the failure-analysis handler can consume it without re-parsing.
    last_test_analysis: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchExecutionState":
        # Handle enum conversion, filter out unknown fields for backward compatibility
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def start_iteration(self) -> None:
        """Start a new iteration."""
        self.iteration += 1
        self.test_generated = False
        self.code_generated = False
        
        # Record iteration start
        record = IterationRecord(
            iteration=self.iteration,
            timestamp=datetime.now().isoformat(),
            phase=self.phase
        )
        self.iterations.append(record.to_dict())
    
    def update_iteration(self, **kwargs) -> None:
        """Update current iteration record."""
        if self.iterations:
            self.iterations[-1].update(kwargs)
    
    def record_failure(self, failure_type: str, analysis: str, test_output: str) -> None:
        """Record a failure for the current iteration."""
        self.last_failure_type = failure_type
        self.last_failure_analysis = analysis
        self.last_test_output = test_output
        self.failure_history.append(failure_type)
        
        self.update_iteration(
            failure_type=failure_type,
            failure_analysis=analysis,
            test_output=test_output[:2000]  # Truncate for storage
        )
    
    def record_commit(self, commit_hash: str) -> None:
        """Record a commit made during this batch."""
        self.current_commit = commit_hash
        self.commits_this_batch.append(commit_hash)
        
        if self.iterations:
            commits = self.iterations[-1].get("commits", [])
            commits.append(commit_hash)
            self.iterations[-1]["commits"] = commits
    
    def mark_complete(self, success: bool) -> None:
        """Mark the batch as complete."""
        self.phase = WorkflowPhase.COMPLETE.value if success else WorkflowPhase.FAILED.value
        self.completed_at = datetime.now().isoformat()
        
        if self.iterations:
            self.iterations[-1]["test_passed"] = success
    
    def can_continue(self) -> bool:
        """Check if more iterations are allowed."""
        return self.iteration < self.max_iterations and self.phase not in [
            WorkflowPhase.COMPLETE.value,
            WorkflowPhase.FAILED.value
        ]
    
    def get_workflow_type(
        self, 
        for_test: bool = True,
        task_description: str = ""
    ) -> str:
        """Determine workflow type based on current state.
        
        Args:
            for_test: Whether this is for test generation (True) or code generation (False)
            task_description: Task description to check for fix-related keywords
        """
        # Check for fix-related keywords in task description (like ZeroRepo)
        fix_keywords = ['fix', 'repair', 'correct', 'debug', 'resolve', 'bug', 'issue', 'error', 'problem']
        is_fix_task = any(keyword in task_description.lower() for keyword in fix_keywords) if task_description else False
        
        if for_test:
            # Test workflow
            if is_fix_task:
                return WorkflowType.TEST_FIX.value
            if self.iteration == 1 and not self.failure_history:
                return WorkflowType.TEST_DEVELOPMENT.value
            elif FailureType.TEST_ERROR.value in (self.failure_history[-3:] if self.failure_history else []):
                return WorkflowType.TEST_FIX.value
            return WorkflowType.TEST_DEVELOPMENT.value
        else:
            # Code workflow
            if is_fix_task:
                return WorkflowType.CODE_BUG_FIX.value
            if self.iteration == 1 and not self.failure_history:
                return WorkflowType.CODE_INCREMENTAL.value
            elif any(f in (self.failure_history[-3:] if self.failure_history else [])
                     for f in [FailureType.CODE_ERROR.value, FailureType.UNKNOWN_ERROR.value]):
                return WorkflowType.CODE_BUG_FIX.value
            return WorkflowType.CODE_INCREMENTAL.value


# ============================================================================
# Global Execution State
# ============================================================================

@dataclass
class CodeGenState:
    """Global state for the entire code generation process.
    
    Tracks overall progress across all tasks.
    """
    # Overall progress
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Task tracking (individual PlannedTask IDs)
    completed_task_ids: List[str] = field(default_factory=list)
    failed_task_ids: List[str] = field(default_factory=list)
    skipped_task_ids: List[str] = field(default_factory=list)
    
    # Current batch (if any)
    current_batch_id: Optional[str] = None
    current_batch_state: Optional[Dict] = None
    
    # Initialization tracking
    initialized: bool = False
    initialized_at: Optional[str] = None
    initial_commit: Optional[str] = None
    
    # Timestamps
    started_at: str = ""
    last_updated: str = ""
    
    # RPG backup tracking (to avoid multiple backups per code_gen run)
    rpg_backup_path: Optional[str] = None
    
    # Trajectory file path (relative to repo root)
    trajectory_file: Optional[str] = None

    # Whether interface skeletons have been written to source files
    interfaces_written: bool = False

    # Track how many times each batch_id has been prepared, to prevent infinite loops.
    # Maps batch_id -> prepare count. A batch prepared more than _MAX_BATCH_PREPARES
    # times is automatically rejected.
    batch_prepare_counts: Dict[str, int] = field(default_factory=dict)

    # Subtree review results (subtree_name -> review result dict).
    # Populated by subtree_review.run_subtree_review() after each subtree completes.
    subtree_reviews: Dict[str, Dict] = field(default_factory=dict)

    # Class-level constant — ClassVar is excluded from asdict() serialization.
    _MAX_BATCH_PREPARES: ClassVar[int] = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeGenState":
        # Filter out unknown fields to handle backward compatibility
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def set_current_batch(self, batch_state: BatchExecutionState) -> None:
        """Set the current batch being processed."""
        self.current_batch_id = batch_state.batch_id
        self.current_batch_state = batch_state.to_dict()
        self.last_updated = datetime.now().isoformat()
    
    def complete_current_batch(self, success: bool) -> None:
        """Mark current batch as complete.
        
        If the current batch has merged_task_ids (file-level merge mode),
        all constituent task IDs are marked as completed/failed.
        """
        if self.current_batch_id:
            # Collect all task IDs to mark (merged mode or single)
            batch_state = self.current_batch_state or {}
            merged_ids = batch_state.get("merged_task_ids", [])
            
            # Build deduplicated list: current_batch_id + any merged IDs
            all_ids = [self.current_batch_id]
            for mid in merged_ids:
                if mid != self.current_batch_id and mid not in all_ids:
                    all_ids.append(mid)
            
            for tid in all_ids:
                if success:
                    if tid not in self.completed_task_ids:
                        self.completed_task_ids.append(tid)
                else:
                    if tid not in self.failed_task_ids:
                        self.failed_task_ids.append(tid)
            
            if success:
                self.completed_tasks = len(self.completed_task_ids)
            else:
                self.failed_tasks = len(self.failed_task_ids)
            
            self.current_batch_id = None
            self.current_batch_state = None
            self.last_updated = datetime.now().isoformat()
    
    def get_current_batch_state(self) -> Optional[BatchExecutionState]:
        """Get the current batch state as an object."""
        if self.current_batch_state:
            return BatchExecutionState.from_dict(self.current_batch_state)
        return None


# ============================================================================
# State Persistence
# ============================================================================


def _count_total_tasks_from_tasks_json(state_path: Path = STATE_FILE) -> int:
    """Count total planned tasks by reading tasks.json.

    Returns 0 if tasks.json doesn't exist or cannot be parsed. Used to backfill
    ``CodeGenState.total_tasks`` since nothing else writes that field after
    ``plan_tasks`` runs.

    The tasks.json path is derived from ``state_path`` (assumed to live in
    the same ``.cmind/data/`` directory) so callers passing a custom
    state_path see the matching tasks.json instead of the workspace
    default.
    """
    try:
        from .task_batch import load_tasks_from_tasks_json
        tasks_path = state_path.parent / "tasks.json"
        if not tasks_path.exists():
            return 0
        return len(load_tasks_from_tasks_json(tasks_path))
    except Exception as exc:
        logging.debug("total_tasks backfill skipped: %s", exc)
        return 0


def _maybe_backfill_total_tasks(
    state: CodeGenState,
    state_path: Path = STATE_FILE,
) -> CodeGenState:
    """Ensure ``state.total_tasks`` reflects the current tasks.json size.

    The field defaults to 0 because ``CodeGenState`` is constructed before
    ``plan_tasks`` produces tasks.json. Backfilling on each load keeps the
    persisted state in sync with the actual task count without requiring
    every call site to remember to update it.
    """
    if state.total_tasks > 0:
        return state
    counted = _count_total_tasks_from_tasks_json(state_path)
    if counted > 0:
        state.total_tasks = counted
    return state


def load_code_gen_state(state_path: Path = STATE_FILE) -> CodeGenState:
    """Load the global code generation state from JSONL file (last valid line).
    
    The state file uses JSONL (JSON Lines) format where each line is a complete
    JSON snapshot of the state. Reading the last valid line gives the latest state.
    If the last line is corrupted (e.g. due to a write failure), the second-to-last
    line is used as a fallback.
    """
    if not state_path.exists():
        return _maybe_backfill_total_tasks(
            CodeGenState(started_at=datetime.now().isoformat()),
            state_path,
        )

    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Try lines from last to first, skipping empty/whitespace lines
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return _maybe_backfill_total_tasks(
                    CodeGenState.from_dict(data), state_path,
                )
            except json.JSONDecodeError:
                logging.warning("Skipping corrupted state line, trying previous line")
                continue
        
        # All lines corrupted or file empty
        logging.warning("All lines in state file are corrupted or empty, starting fresh")
        return _maybe_backfill_total_tasks(
            CodeGenState(started_at=datetime.now().isoformat()),
            state_path,
        )
    except Exception as e:
        logging.warning(f"Failed to load code gen state: {e}")
        return _maybe_backfill_total_tasks(
            CodeGenState(started_at=datetime.now().isoformat()),
            state_path,
        )


def save_code_gen_state(state: CodeGenState, state_path: Path = STATE_FILE) -> None:
    """Append the global code generation state as a new line to the JSONL file.
    
    Each save appends a single JSON line. This is crash-safe: if the append
    fails mid-write, the previous lines remain intact and can be recovered.
    
    A diagnostic check logs if the new state has fewer completed/failed IDs
    than the persisted state, but does NOT auto-restore them (to avoid
    deadlocks caused by phantom restored IDs).
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.now().isoformat()
    
    # Defensive check: warn if the state being saved has fewer completed/failed
    # task IDs than the last persisted state.  We intentionally do NOT auto-restore
    # the "lost" IDs because:
    #   1. A --skip operation legitimately does not add to completed_task_ids.
    #   2. Auto-restoring IDs from a corrupted/externally-modified state file
    #      can create phantom "all tasks completed" states and deadlocks.
    # The warning is logged so operators can investigate if needed.
    if state_path.exists():
        try:
            existing = load_code_gen_state(state_path)
            existing_completed = set(existing.completed_task_ids)
            existing_failed = set(existing.failed_task_ids)
            new_completed = set(state.completed_task_ids)
            new_failed = set(state.failed_task_ids)
            
            lost_completed = existing_completed - new_completed
            lost_failed = existing_failed - new_failed
            
            if lost_completed:
                logging.info(
                    f"State save: {len(lost_completed)} completed task IDs in persisted "
                    f"state are not in new state (this is expected after --skip or state reset)."
                )
            
            if lost_failed:
                logging.info(
                    f"State save: {len(lost_failed)} failed task IDs in persisted "
                    f"state are not in new state (this is expected after retry or state reset)."
                )
        except Exception:
            pass  # If check fails, proceed with normal save
    
    # Serialize — truncate redundant large text fields in the persisted copy to
    # keep the state file small.  The in-memory state object is NOT modified.
    # NOTE: test_prompt and code_prompt are NOT truncated because the
    # orchestrator's sub-agent prompts load them back from the persisted state.
    state_dict = state.to_dict()
    batch_dict = state_dict.get('current_batch_state')
    if batch_dict and isinstance(batch_dict, dict):
        _MAX_OUTPUT_PERSIST = 8000   # chars of test output kept
        # Truncate last_test_output — keep enough for the failure-analysis
        # handler to work with (it reads from persisted state).  Pytest error
        # summaries appear at the end, so keep the tail when truncating.
        if isinstance(batch_dict.get('last_test_output'), str):
            output = batch_dict['last_test_output']
            if len(output) > _MAX_OUTPUT_PERSIST:
                head_size = 1000
                tail_size = _MAX_OUTPUT_PERSIST - head_size
                batch_dict['last_test_output'] = (
                    output[:head_size]
                    + '\n...(middle truncated)...\n'
                    + output[-tail_size:]
                )
        # Truncate iteration history — keep only the last 2 entries
        iterations = batch_dict.get('iterations', [])
        if isinstance(iterations, list) and len(iterations) > 2:
            batch_dict['iterations'] = iterations[-2:]

    line = json.dumps(state_dict, ensure_ascii=False)
    with open(state_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

    # Auto-compact: when file exceeds threshold, archive old lines and keep
    # only the most recent N snapshots. The previous threshold was 50 MB
    # which never triggered in practice, leaving the state file at ~880 KB
    # after a single 100-batch run because every save dumps the full state.
    # Lowered to 200 KB and we keep the last 20 snapshots so debugging can
    # still walk back a few steps without bloating the file.
    _COMPACT_THRESHOLD = 200 * 1024        # 200 KB
    _KEEP_LAST_N = 20                      # snapshots retained after compact
    try:
        file_size = state_path.stat().st_size
        if file_size > _COMPACT_THRESHOLD:
            with open(state_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            if len(all_lines) > _KEEP_LAST_N:
                archive_path = state_path.with_suffix('.jsonl.archive')
                with open(archive_path, 'a', encoding='utf-8') as af:
                    af.writelines(all_lines[:-_KEEP_LAST_N])
                # Write the compacted file to a temp location first, then
                # atomically rename to avoid data loss on crash.
                tmp_path = state_path.with_suffix('.jsonl.tmp')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.writelines(all_lines[-_KEEP_LAST_N:])
                tmp_path.replace(state_path)  # atomic on POSIX
                kept_size = sum(len(l) for l in all_lines[-_KEEP_LAST_N:])
                logging.info(
                    f"State file compacted: archived {len(all_lines)-_KEEP_LAST_N} lines, "
                    f"kept last {_KEEP_LAST_N} "
                    f"({file_size/1024:.0f}KB -> {kept_size/1024:.0f}KB)"
                )
    except Exception as e:
        logging.warning(f"State file compaction failed (non-fatal): {e}")


def get_or_create_batch_state(
    batch_id: str,
    file_path: str = "",
    subtree: str = "",
    state_path: Path = STATE_FILE
) -> BatchExecutionState:
    """Get existing batch state or create a new one.
    
    If a batch is already in progress, returns its state.
    Otherwise creates a new BatchExecutionState.
    """
    global_state = load_code_gen_state(state_path)
    
    # Check if this batch is already in progress
    if global_state.current_batch_id == batch_id and global_state.current_batch_state:
        return BatchExecutionState.from_dict(global_state.current_batch_state)
    
    # Check if already completed
    if batch_id in global_state.completed_task_ids:
        raise ValueError(f"Batch {batch_id} is already completed")
    
    # Create new batch state
    batch_state = BatchExecutionState(
        batch_id=batch_id,
        file_path=file_path,
        subtree=subtree,
        started_at=datetime.now().isoformat()
    )
    
    return batch_state


def update_batch_state(
    batch_state: BatchExecutionState,
    state_path: Path = STATE_FILE
) -> None:
    """Update the batch state in the global state file."""
    global_state = load_code_gen_state(state_path)
    global_state.set_current_batch(batch_state)
    save_code_gen_state(global_state, state_path)


def complete_batch(
    batch_id: str,
    success: bool,
    state_path: Path = STATE_FILE,
    rpg_backup_path: Optional[str] = None
) -> None:
    """Mark a batch as complete (success or failure).
    
    Args:
        batch_id: The batch to complete.
        success: Whether the batch succeeded.
        state_path: Path to state file.
        rpg_backup_path: If provided, update rpg_backup_path atomically
            with the completion (avoids stale intermediate saves).
    """
    global_state = load_code_gen_state(state_path)
    
    if global_state.current_batch_id == batch_id:
        if rpg_backup_path:
            global_state.rpg_backup_path = rpg_backup_path
        global_state.complete_current_batch(success)
        save_code_gen_state(global_state, state_path)


def skip_current_batch(batch_id: str, state_path: Path = STATE_FILE) -> bool:
    """Clear the current batch without marking it completed or failed.

    Used when an out-of-band condition (e.g. the batch branch went missing
    because the sub-agent committed to main directly) prevents the batch
    from being merged, but is not a code-quality failure. The batch_id is
    recorded in ``skipped_task_ids`` for observability, yet remains absent
    from ``completed_task_ids`` and ``failed_task_ids`` so the next
    ``--next`` invocation re-attempts it without consuming a retry slot.

    Loop guard: ``batch_prepare_counts[batch_id]`` is incremented on each
    skip; once it reaches ``_MAX_BATCH_PREPARES`` the batch is recorded
    in ``failed_task_ids`` instead, so a sub-agent that keeps making the
    same mistake (e.g. always committing to main) cannot trap the
    workflow in an infinite skip loop.

    Returns ``True`` when the skip succeeded, ``False`` when the loop
    guard converted the skip into a hard failure.
    """
    global_state = load_code_gen_state(state_path)
    if global_state.current_batch_id != batch_id:
        return False

    # Increment skip counter and check the loop guard. We reuse
    # batch_prepare_counts since it's already wired into the orchestrator's
    # batch-prep flow and asdict() persists it through to_dict().
    skip_count = global_state.batch_prepare_counts.get(batch_id, 0) + 1
    global_state.batch_prepare_counts[batch_id] = skip_count

    max_skips = CodeGenState._MAX_BATCH_PREPARES
    if skip_count >= max_skips:
        # Promote to a hard failure so the orchestrator stops picking
        # this task up. The operator can investigate and either --retry
        # (which clears failed_task_ids) or --skip explicitly.
        if batch_id not in global_state.failed_task_ids:
            global_state.failed_task_ids.append(batch_id)
        global_state.failed_tasks = len(global_state.failed_task_ids)
        global_state.current_batch_id = None
        global_state.current_batch_state = None
        global_state.last_updated = datetime.now().isoformat()
        save_code_gen_state(global_state, state_path)
        logging.warning(
            "Batch %s skipped %d times (limit: %d) — promoted to failed_task_ids",
            batch_id, skip_count, max_skips,
        )
        return False

    if batch_id not in global_state.skipped_task_ids:
        global_state.skipped_task_ids.append(batch_id)
    global_state.current_batch_id = None
    global_state.current_batch_state = None
    global_state.last_updated = datetime.now().isoformat()
    save_code_gen_state(global_state, state_path)
    return True


def reset_rpg_backup_tracking(state_path: Path = STATE_FILE) -> None:
    """Reset the RPG backup tracking for a fresh code_gen session.
    
    Call this when you want to force a new backup on the next batch completion.
    """
    global_state = load_code_gen_state(state_path)
    global_state.rpg_backup_path = None
    save_code_gen_state(global_state, state_path)


def get_or_create_code_gen_trajectory(
    global_state: 'CodeGenState',
    base_dir: Path = None,
    state_path: Path = STATE_FILE
) -> 'Any':
    """Get existing or create new Trajectory for the code_gen workflow.
    
    All scripts in the code_gen pipeline share the same trajectory file
    within a single code_gen session. The trajectory file path is stored
    in CodeGenState.trajectory_file.
    
    Returns:
        Trajectory instance (loaded or newly created)
    """
    # Trajectory files live under .cmind/data/trajectory/ (workspace level),
    # not inside repo/, so base_dir should be the workspace root.
    base_dir = base_dir or WORKSPACE_ROOT
    
    # Try to load existing trajectory
    if global_state.trajectory_file:
        traj_path = base_dir / global_state.trajectory_file
        if traj_path.exists():
            traj = Trajectory("code_gen", base_dir)
            traj.trajectory_file = traj_path
            if traj.load():
                return traj
    
    # Create new trajectory
    traj = Trajectory("code_gen", base_dir)
    traj.start({"workflow": "code_gen"})
    
    # Save trajectory file path in state (relative to base_dir)
    try:
        rel_path = traj.trajectory_file.relative_to(base_dir)
        global_state.trajectory_file = str(rel_path)
    except ValueError:
        global_state.trajectory_file = str(traj.trajectory_file)
    save_code_gen_state(global_state, state_path)
    
    return traj
