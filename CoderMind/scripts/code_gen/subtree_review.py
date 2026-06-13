#!/usr/bin/env python3
"""Subtree Review — Feature-level completeness verification.

After all tasks in a subtree are completed, this module runs:
1. Static checks (no LLM cost) for stubs and placeholders
2. An LLM review agent that traces user journeys through the code
3. Post-verification pytest to ensure fixes don't break anything

The review is **non-blocking**: failures are recorded but do not
prevent subsequent subtrees from proceeding.
"""

import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.task_batch import load_tasks_from_tasks_json
from common.paths import TASKS_FILE
from code_gen.static_checks import static_completeness_check

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class _SubtreeReviewResult:
    """Outcome of a single subtree review."""

    subtree: str
    status: str = "NOT_RUN"  # ALL_COMPLETE | FIXED | BLOCKED | NOT_RUN
    timestamp: str = ""
    issues_found: int = 0
    issues_fixed: int = 0
    static_issues: List[str] = field(default_factory=list)
    duration: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Subtree info gathering
# ============================================================================

def _get_subtree_files(tasks_path: Path, subtree: str) -> List[str]:
    """Return deduplicated list of file paths for a subtree."""
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    seen: Set[str] = set()
    files: List[str] = []
    for t in all_tasks:
        if t.subtree == subtree and t.file_path not in seen:
            # Skip marker paths like <INTEGRATION_TEST>
            if t.file_path.startswith("<") and t.file_path.endswith(">"):
                continue
            seen.add(t.file_path)
            files.append(t.file_path)
    return files


def _get_subtree_description(tasks_path: Path, subtree: str) -> str:
    """Build a description string for a subtree from its task descriptions."""
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    descs = [t.task for t in all_tasks if t.subtree == subtree]
    return "\n".join(f"- {d}" for d in descs[:10])  # cap at 10


def _needs_llm_review(subtree_files: List[str], repo_path: Path) -> bool:
    """Determine if a subtree needs full LLM review.

    Skips LLM review (saves ~95s) when the subtree has:
    - No static issues (no stubs/placeholders)
    - No string-based cross-module references (href, form action, event emit, etc.)

    For pure-logic subtrees (services, models, utilities), this avoids
    unnecessary LLM calls that would return ALL_COMPLETE anyway.

    Args:
        subtree_files: List of file paths in this subtree.
        repo_path: Project repo root.

    Returns:
        True if LLM review is recommended, False if safe to skip.
    """
    from lang_parser import is_supported_source

    for filepath in subtree_files:
        full_path = repo_path / filepath
        if not full_path.exists() or not is_supported_source(filepath):
            continue
        try:
            content = full_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        # Check for string-based cross-module references
        # These patterns match HTML link generation, event emission, etc.
        # Match both regular quotes and escaped quotes (\" in Python strings)
        if re.search(r'href\s*=\s*(?:["\']|\\["\'])\s*/', content):
            return True
        if re.search(r'action\s*=\s*(?:["\']|\\["\'])\s*/', content):
            return True
        if re.search(r'\.emit\s*\(\s*["\']', content):
            return True
        if re.search(r'\.publish\s*\(\s*["\']', content):
            return True

    return False


def is_subtree_just_completed(
    batch_id: str,
    completed_task_ids: List[str],
    tasks_path: Path,
) -> Optional[str]:
    """Check if completing *batch_id* causes its subtree to be fully done.

    Args:
        batch_id: The batch that just completed.
        completed_task_ids: All completed task IDs (including batch_id).
        tasks_path: Path to tasks.json.

    Returns:
        Subtree name if it just completed, else ``None``.
    """
    all_tasks = load_tasks_from_tasks_json(tasks_path)

    # Find the subtree for this batch
    task = None
    for t in all_tasks:
        if t.task_id == batch_id:
            task = t
            break
    if task is None:
        return None

    subtree = task.subtree

    # Special subtrees are never reviewed
    if subtree in ("FINAL_TASKS", "PROJECT_FILES", ""):
        return None

    # Check if all tasks in this subtree are now completed
    subtree_ids = {t.task_id for t in all_tasks if t.subtree == subtree}
    completed_set = set(completed_task_ids)
    if subtree_ids.issubset(completed_set):
        return subtree
    return None


# ============================================================================
# Review prompt
# ============================================================================

REVIEW_PROMPT = """\
# Feature Completeness Review

## Your Role
You are a code reviewer verifying that a feature is complete and functional.
Think from the END USER's perspective, not the developer's.
"User" means end-user for applications, or developer-user for libraries/SDKs.

## Feature: "{subtree_name}"
{subtree_description}

## Files Implementing This Feature
{file_list}

## Project Background
{project_background}

## Static Check Results
{static_check_results}

## Instructions

1. **Read ALL files listed above** — understand what has been built.

2. **Think about the user journey**:
   - How would a user access this feature? (URL? command? button? API call?)
   - What steps would they take to complete the workflow?
   - What would they see at each step?
   - What happens on success? On failure? On invalid input?

3. **Verify completeness** — check that:
   - Every step in the user journey has working code behind it
   - Entry points exist (routes registered, commands wired, menus populated —
     whatever the project uses)
   - User-facing output is generated (pages, CLI output, widgets — whatever applies)
   - Error handling provides feedback to the user
   - The feature connects to the rest of the application

4. **You MUST trace at least one complete user journey through the code.**
   For EACH step, cite the specific file:function that handles it.
   If you cannot trace a complete journey, the feature is NOT complete.

5. **Report findings** with a structured checklist:

   ## Self-Generated Checklist
   - [ ] <item you decided to check>: PASS / MISSING / BROKEN
   - [ ] ...

   ## User Journey Trace
   Step 1: User does X → file.py:func() → Result: ✓/✗
   Step 2: ...

6. **Fix issues**: If you found MISSING or BROKEN items, write the fix code.
   Then run the test command to verify no regressions.

## Constraints
- Do NOT refactor working code
- Do NOT change function signatures that already have passing tests
- Only ADD missing pieces or FIX broken connections
- Your changes must not break existing tests

## Already Completed (other subtrees — do not modify)
{completed_modules_from_other_subtrees}

## Skeleton Only (not yet implemented — will be done in later subtrees)
{skeleton_only_files}

## Test Command
{test_cmd}

## Output
Last line MUST be one of:
  REVIEW_RESULT: ALL_COMPLETE
  REVIEW_RESULT: FIXED <N> issues
  REVIEW_RESULT: BLOCKED | <reason>
"""


def _build_review_prompt(
    subtree_name: str,
    subtree_description: str,
    subtree_files: List[str],
    static_check_results: str,
    completed_task_ids: List[str],
    tasks_path: Path,
    repo_path: Path,
    project_background: str = "",
    test_cmd: str = "",
) -> str:
    """Construct the review prompt for an LLM sub-agent."""
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    completed_set = set(completed_task_ids)

    # Classify files
    all_completed_files: Set[str] = set()
    for t in all_tasks:
        if t.task_id in completed_set:
            fp = t.file_path
            if not (fp.startswith("<") and fp.endswith(">")):
                all_completed_files.add(fp)

    current_files = set(subtree_files)
    other_completed = sorted(all_completed_files - current_files)

    # Find skeleton-only files (exist on disk but not completed)
    all_source_files: Set[str] = set()
    for t in all_tasks:
        fp = t.file_path
        if not (fp.startswith("<") and fp.endswith(">")):
            all_source_files.add(fp)
    skeleton_files = sorted(all_source_files - all_completed_files)

    file_list = "\n".join(f"- `{f}`" for f in subtree_files)
    other_list = "\n".join(f"- `{f}`" for f in other_completed) if other_completed else "(none)"
    skel_list = "\n".join(f"- `{f}`" for f in skeleton_files) if skeleton_files else "(none)"

    base_prompt = REVIEW_PROMPT.format(
        subtree_name=subtree_name,
        subtree_description=subtree_description,
        file_list=file_list,
        project_background=project_background or "(see existing source files)",
        static_check_results=static_check_results or "All static checks passed.",
        completed_modules_from_other_subtrees=other_list,
        skeleton_only_files=skel_list,
        test_cmd=test_cmd or "the project's native test command",
    )

    # Append cross-subtree connection check if there are completed dependencies
    cross_subtree_section = ""
    if other_completed:
        # Build a list of cross-subtree dependencies from interfaces.json
        try:
            from common.paths import INTERFACES_FILE
            from code_gen.context_collector import collect_dependency_files
            cross_deps = []
            for f in subtree_files:
                try:
                    deps = collect_dependency_files(INTERFACES_FILE, f)
                    for dep_type in ("inherits_from", "invokes", "references"):
                        for dep in deps.get(dep_type, []):
                            dep_file = dep.get("parent_file") or dep.get("callee_file") or dep.get("type_file", "")
                            if dep_file and dep_file not in current_files and dep_file in all_completed_files:
                                cross_deps.append(f"- `{f}` → `{dep_file}` ({dep_type})")
                except Exception:
                    pass

            if cross_deps:
                cross_subtree_section = (
                    "\n\n## Cross-Subtree Connection Verification\n\n"
                    "The following files in YOUR subtree depend on files from OTHER completed subtrees.\n"
                    "For each connection, verify that:\n"
                    "1. The function/class your code imports from the other subtree actually exists\n"
                    "2. The parameters your code passes match the other module's signature\n"
                    "3. Any string identifiers (URLs, event names, etc.) match exactly\n\n"
                    "Dependencies on other subtrees:\n"
                    + "\n".join(sorted(set(cross_deps)))
                    + "\n"
                )
        except Exception as exc:
            logger.debug("Could not build cross-subtree deps: %s", exc)

    return base_prompt + cross_subtree_section


# ============================================================================
# Review execution
# ============================================================================

def _parse_review_result(response: Optional[str]) -> _SubtreeReviewResult:
    """Parse the review agent's output for the result marker."""
    result = _SubtreeReviewResult(subtree="")
    if not response:
        result.status = "BLOCKED"
        result.reason = "No response from review agent"
        return result

    lines = response.strip().splitlines()
    search_lines = lines[-20:] if len(lines) > 20 else lines

    for line in reversed(search_lines):
        line = line.strip()
        if line.startswith("REVIEW_RESULT: ALL_COMPLETE"):
            result.status = "ALL_COMPLETE"
            return result
        if line.startswith("REVIEW_RESULT: FIXED"):
            result.status = "FIXED"
            # Try to extract count
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    result.issues_fixed = int(p)
                    break
            return result
        if line.startswith("REVIEW_RESULT: BLOCKED"):
            result.status = "BLOCKED"
            result.reason = line.split("|", 1)[1].strip() if "|" in line else "Unknown"
            return result

    result.status = "BLOCKED"
    result.reason = "Review agent did not output REVIEW_RESULT marker"
    return result


def run_subtree_review(
    subtree_name: str,
    completed_task_ids: List[str],
    repo_path: Path,
    tasks_path: Path = TASKS_FILE,
    project_background: str = "",
    agent_timeout: int = 900,
) -> _SubtreeReviewResult:
    """Execute a full subtree review: static checks → LLM review → post-verify.

    This function is designed to be called from ``run_batch.py`` after a
    batch merge succeeds and ``is_subtree_just_completed()`` returns the
    subtree name.

    The review is **non-blocking**: exceptions are caught and recorded.

    Args:
        subtree_name: Name of the subtree that just completed.
        completed_task_ids: All completed task IDs so far.
        repo_path: Absolute path to the project repository.
        tasks_path: Path to tasks.json.
        project_background: Project background text for the prompt.
        agent_timeout: Timeout for the LLM review agent (seconds).

    Returns:
        _SubtreeReviewResult with status and metrics.
    """
    start_time = time.time()
    result = _SubtreeReviewResult(
        subtree=subtree_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # 1. Collect subtree info
    subtree_files = _get_subtree_files(tasks_path, subtree_name)
    subtree_description = _get_subtree_description(tasks_path, subtree_name)

    if not subtree_files:
        result.status = "ALL_COMPLETE"
        result.reason = "No source files in subtree"
        result.duration = time.time() - start_time
        return result

    # 2. Static checks
    static_issues = static_completeness_check(subtree_files, repo_path)
    result.static_issues = static_issues
    result.issues_found = len(static_issues)
    static_result_str = "\n".join(static_issues) if static_issues else "All static checks passed."

    if static_issues:
        logger.warning(
            "Subtree '%s' has %d static issues", subtree_name, len(static_issues)
        )

    # 2b. Smart skip: if no static issues AND no cross-module string references,
    # skip the LLM review (saves ~95s per subtree)
    if not static_issues and not _needs_llm_review(subtree_files, repo_path):
        result.status = "ALL_COMPLETE"
        result.reason = "No static issues and no cross-module string references"
        result.duration = time.time() - start_time
        logger.info(
            "Subtree '%s' skipped LLM review (no cross-ref patterns)",
            subtree_name,
        )
        return result

    # 3. Build the target language's native test command for the review prompt
    from code_gen.test_runner import get_dev_python, resolve_test_backend
    from code_gen.batch_prompts import _build_backend_test_cmd
    venv_python = get_dev_python(repo_path) or "python3"
    backend = resolve_test_backend(valid_files=subtree_files)
    test_cmd = _build_backend_test_cmd(backend, repo_path, [], venv_python)

    # 4. Build review prompt
    prompt = _build_review_prompt(
        subtree_name=subtree_name,
        subtree_description=subtree_description,
        subtree_files=subtree_files,
        static_check_results=static_result_str,
        completed_task_ids=completed_task_ids,
        tasks_path=tasks_path,
        repo_path=repo_path,
        project_background=project_background,
        test_cmd=test_cmd,
    )

    # 5. Setup review branch
    from common.git_utils import GitRunner, sanitize_branch_component
    git = GitRunner(str(repo_path))

    safe_name = sanitize_branch_component(subtree_name.lower(), max_len=40, fallback="review")
    branch_name = f"review/{safe_name}"

    # Ensure on main first
    current = git.get_current_branch()
    if current != git.main_branch:
        if git.has_uncommitted_changes():
            git.stage_and_commit("WIP: auto-save before review")
        git.switch_branch(git.main_branch)

    if git.branch_exists(branch_name):
        git.delete_branch(branch_name, force=True)
    git.create_branch(branch_name)

    # 6. Dispatch review agent (reuse existing sub-agent mechanism)
    try:
        # Import here to avoid circular imports at module level
        from run_batch import dispatch_sub_agent

        response, error = dispatch_sub_agent(
            prompt, repo_path, timeout=agent_timeout, purpose="subtree_review"
        )

        if error:
            result.status = "BLOCKED"
            result.reason = f"Review agent error: {error}"
            # Switch back to main
            if git.has_uncommitted_changes():
                git.stage_and_commit("WIP: review agent failed")
            git.switch_branch(git.main_branch)
            result.duration = time.time() - start_time
            return result

        # 7. Parse review result
        parsed = _parse_review_result(response)
        result.status = parsed.status
        result.issues_fixed = parsed.issues_fixed
        result.reason = parsed.reason

        # 8. Post-verify if review made changes
        if result.status in ("FIXED", "ALL_COMPLETE"):
            # Run pytest to verify no regressions
            from code_gen.test_runner import run_project_tests, ensure_deps_installed
            if backend.name == "python":
                try:
                    ensure_deps_installed(repo_path)
                except Exception:
                    pass
            verify_result = run_project_tests(
                repo_path,
                timeout=180,
                extra_args=["--timeout=30", "--timeout-method=signal"],
                backend=backend,
            )
            verify_passed = verify_result.success
            if verify_passed:
                # Merge review branch
                if git.has_uncommitted_changes():
                    git.stage_and_commit(
                        f"review({safe_name}): fix {result.issues_fixed} issues"
                    )
                merge_ok, merge_err = git.merge_branch(
                    branch_name,
                    message=f"merge: review {subtree_name}\n\nsubtree_review: {subtree_name}",
                )
                if merge_ok:
                    git.delete_branch(branch_name)
                    logger.info("Review branch '%s' merged", branch_name)
                else:
                    result.status = "BLOCKED"
                    result.reason = f"Merge failed: {merge_err}"
                    logger.warning("Review merge failed: %s", merge_err)
                    # Ensure we're on main after failed merge
                    try:
                        current = git.get_current_branch()
                        if current != git.main_branch:
                            git.switch_branch(git.main_branch)
                    except Exception:
                        pass
            else:
                result.status = "BLOCKED"
                result.reason = "Fix introduced test regression"
                if git.has_uncommitted_changes():
                    git.stage_and_commit("WIP: review fix caused regression")
                git.switch_branch(git.main_branch)
        else:
            # BLOCKED — switch back to main
            if git.has_uncommitted_changes():
                git.stage_and_commit("WIP: review blocked")
            git.switch_branch(git.main_branch)

    except Exception as exc:
        logger.warning("Subtree review for '%s' failed: %s", subtree_name, exc)
        result.status = "BLOCKED"
        result.reason = str(exc)
        try:
            if git.has_uncommitted_changes():
                git.stage_and_commit("WIP: review exception")
            git.switch_branch(git.main_branch)
        except Exception:
            pass

    result.duration = time.time() - start_time
    return result
