#!/usr/bin/env python3
"""Git Utilities for CoderMind Code Generation.

Provides Git operations for branch management and version control
during the code generation phase:
- Branch creation, switching, and deletion
- Commit and merge operations with conflict detection
- Stash management for safe branch switching
- Task branch lifecycle (create / merge / abandon)
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


_INVALID_REF_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_branch_component(
    component: str,
    max_len: int = 50,
    fallback: str = "x",
) -> str:
    """Normalize a dynamic string into a git-safe branch path component.

    The result is safe to embed as ``<prefix>/<component>`` in a branch name.
    It replaces characters git rejects in refs (spaces, ``~^:?*[`` and
    backslash, control chars) with ``_``, collapses ``..`` and repeated
    separators, strips leading/trailing separators, caps length, and avoids a
    trailing ``.`` or ``.lock`` suffix. Always returns a non-empty token so
    callers can build a valid ref for any language's task identifiers.

    Args:
        component: Raw dynamic text (task id, subtree name, ...).
        max_len: Maximum length of the returned component.
        fallback: Token returned when sanitization yields an empty string.

    Returns:
        A git-ref-safe, non-empty component string.
    """
    raw = (component or "").strip()
    if not raw:
        return fallback

    safe = _INVALID_REF_CHARS.sub("_", raw.replace("\\", "/").replace("/", "_"))
    safe = safe.replace("..", "_")
    safe = re.sub(r"[._-]{2,}", "_", safe)
    safe = safe.strip("._-")
    if not safe:
        return fallback

    safe = safe[:max_len].rstrip("._-")
    if safe.endswith(".lock"):
        safe = safe[: -len(".lock")].rstrip("._-")

    return safe or fallback


@dataclass
class GitResult:
    """Result of a Git command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class GitRunner:
    """Git command runner for code generation workflow.
    
    Handles:
    - Branch creation and switching
    - Commits and merges
    - Stash operations
    - Safe directory handling
    """
    
    # The canonical main branch name used by all CoderMind repos.
    MAIN_BRANCH = "main"

    def __init__(
        self,
        repo_path: str,
        main_branch: str = "main",
        logger: Optional[logging.Logger] = None
    ):
        self.repo_path = Path(repo_path)
        self.logger = logger or logging.getLogger(__name__)
        self.main_branch = self.MAIN_BRANCH

        # Ensure repo exists and is a git repo
        self._ensure_git_repository()

    def run_git(
        self,
        args: List[str],
        check: bool = False,
        capture_output: bool = True
    ) -> GitResult:
        """Run a git subcommand (automatically prepends 'git').

        Args:
            args: Git subcommand arguments (e.g., ["add", "-A"])
            check: Raise exception on failure
            capture_output: Capture stdout/stderr

        Returns:
            GitResult with success status and output
        """
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                timeout=60
            )

            git_result = GitResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip() if result.stdout else "",
                stderr=result.stderr.strip() if result.stderr else "",
                returncode=result.returncode
            )

            if check and not git_result.success:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return git_result

        except subprocess.TimeoutExpired:
            self.logger.error(f"Git command timed out: {' '.join(cmd)}")
            return GitResult(success=False, stderr="Command timed out", returncode=-1)
        except Exception as e:
            self.logger.error(f"Git command failed: {e}")
            return GitResult(success=False, stderr=str(e), returncode=-1)

    def _ensure_git_repository(self) -> None:
        """Ensure the repository is a valid git repo with 'main' as the default branch."""
        git_dir = self.repo_path / ".git"

        if not git_dir.exists():
            self.logger.info("Initializing git repository...")
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self.run_git(["init", "-b", self.MAIN_BRANCH])

        # Configure safe directory
        self.run_git([
            "config", "--global", "--add",
            "safe.directory", str(self.repo_path.resolve())
        ])
    
    def get_current_branch(self) -> str:
        """Get the name of the current branch."""
        result = self.run_git(["branch", "--show-current"])
        return result.stdout if result.success else ""
    
    def get_head_commit(self) -> str:
        """Get the current HEAD commit hash."""
        result = self.run_git(["rev-parse", "HEAD"])
        return result.stdout if result.success else ""
    
    def get_main_branch_commit(self) -> Optional[str]:
        """Get the commit hash of the main branch."""
        result = self.run_git(["rev-parse", self.main_branch])
        return result.stdout if result.success else None
    
    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        result = self.run_git(["status", "--porcelain"])
        return bool(result.stdout) if result.success else False
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        """Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            from_branch: Branch to create from (default: current)
            
        Returns:
            True if successful
        """
        if from_branch:
            result = self.run_git(["checkout", "-b", branch_name, from_branch])
        else:
            result = self.run_git(["checkout", "-b", branch_name])
        
        if result.success:
            self.logger.info(f"Created branch: {branch_name}")
        else:
            self.logger.error(f"Failed to create branch: {result.stderr}")
        
        return result.success
    
    def switch_branch(self, branch_name: str, force: bool = False) -> bool:
        """Switch to an existing branch.
        
        Args:
            branch_name: Branch to switch to
            force: Force switch even with uncommitted changes
            
        Returns:
            True if successful
        """
        args = ["checkout"]
        if force:
            args.append("-f")
        args.append(branch_name)
        
        result = self.run_git(args)
        
        if result.success:
            self.logger.info(f"Switched to branch: {branch_name}")
        else:
            self.logger.error(f"Failed to switch branch: {result.stderr}")
        
        return result.success
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        result = self.run_git(["rev-parse", "--verify", branch_name])
        return result.success
    
    def stage_all(self) -> bool:
        """Stage all changes."""
        result = self.run_git(["add", "-A"])
        return result.success
    
    def commit(self, message: str) -> Tuple[bool, str]:
        """Commit staged changes.
        
        Args:
            message: Commit message
            
        Returns:
            Tuple of (success, commit_hash)
        """
        # Check if there are changes to commit
        result = self.run_git(["diff", "--staged", "--quiet"])
        if result.success:
            self.logger.info("No changes to commit")
            return True, self.get_head_commit()
        
        # Commit
        result = self.run_git(["commit", "-m", message])
        if result.success:
            commit_hash = self.get_head_commit()
            self.logger.info(f"Committed: {commit_hash[:8]}")
            return True, commit_hash
        else:
            self.logger.error(f"Commit failed: {result.stderr}")
            return False, ""
    
    def stage_and_commit(self, message: str) -> Tuple[bool, str]:
        """Stage all changes and commit.
        
        Returns:
            Tuple of (success, commit_hash)
        """
        self.stage_all()
        return self.commit(message)
    
    def merge_branch(
        self,
        source_branch: str,
        target_branch: Optional[str] = None,
        no_ff: bool = True,
        message: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Merge a branch into target (default: main).
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (default: main)
            no_ff: Use --no-ff flag
            message: Custom merge commit message (default: auto-generated)
            
        Returns:
            Tuple of (success, error_type)
            - success: True if merge succeeded
            - error_type: None on success, or one of:
              'uncommitted_changes', 'switch_failed', 'merge_conflict', 'merge_failed'
        """
        target = target_branch or self.main_branch
        
        # Check for uncommitted changes before switching
        if self.has_uncommitted_changes():
            self.logger.error("Cannot merge: uncommitted changes exist")
            return False, "uncommitted_changes"
        
        # Switch to target branch
        if not self.switch_branch(target):
            return False, "switch_failed"
        
        # Merge
        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        merge_msg = message or f"Merge branch '{source_branch}'"
        args.extend(["-m", merge_msg, source_branch])
        
        result = self.run_git(args)
        
        if result.success:
            self.logger.info(f"Merged {source_branch} into {target}")
            return True, None
        
        # Check if it's a merge conflict
        if "CONFLICT" in result.stdout or "CONFLICT" in result.stderr:
            self.logger.error("Merge conflict detected, aborting merge")
            self.run_git(["merge", "--abort"])
            return False, "merge_conflict"
        
        self.logger.error(f"Merge failed: {result.stderr}")
        return False, "merge_failed"
    
    def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        result = self.run_git(["branch", flag, branch_name])
        return result.success
    
    def reset_hard(self, commit: Optional[str] = None) -> bool:
        """Hard reset to a commit.
        
        Args:
            commit: Commit to reset to (default: HEAD)
            
        Returns:
            True if successful
        """
        args = ["reset", "--hard"]
        if commit:
            args.append(commit)
        
        result = self.run_git(args)
        
        if result.success:
            self.logger.info(f"Reset to: {commit or 'HEAD'}")
        else:
            self.logger.error(f"Reset failed: {result.stderr}")
        
        return result.success
    
    def stash(self, message: Optional[str] = None) -> bool:
        """Stash current changes."""
        args = ["stash", "push"]
        if message:
            args.extend(["-m", message])
        
        result = self.run_git(args)
        return result.success
    
    def stash_if_dirty(self, message: Optional[str] = None) -> Tuple[bool, bool]:
        """Stash changes only if there are uncommitted changes.
        
        Args:
            message: Optional stash message
            
        Returns:
            Tuple of (success, was_dirty)
            - success: True if operation succeeded (including when no stash needed)
            - was_dirty: True if there were changes that got stashed
        """
        if not self.has_uncommitted_changes():
            return True, False
        
        stash_msg = message or "auto-stash before git operation"
        success = self.stash(stash_msg)
        if success:
            self.logger.info(f"Stashed uncommitted changes: {stash_msg}")
        else:
            self.logger.error("Failed to stash uncommitted changes")
        return success, success
    
    def stash_pop(self) -> bool:
        """Pop the most recent stash."""
        result = self.run_git(["stash", "pop"])
        return result.success
    
    def get_diff(
        self,
        from_commit: Optional[str] = None,
        to_commit: str = "HEAD"
    ) -> str:
        """Get diff between commits.
        
        Args:
            from_commit: Start commit (default: parent of to_commit)
            to_commit: End commit (default: HEAD)
            
        Returns:
            Diff content as string
        """
        if from_commit:
            result = self.run_git(["diff", from_commit, to_commit])
        else:
            result = self.run_git(["diff", f"{to_commit}^", to_commit])
        
        return result.stdout if result.success else ""
    
    def get_changed_files(
        self,
        from_commit: Optional[str] = None,
        to_commit: str = "HEAD"
    ) -> List[str]:
        """Get list of files changed between commits.
        
        Returns:
            List of file paths
        """
        if from_commit:
            result = self.run_git([
                "diff", "--name-only", from_commit, to_commit
            ])
        else:
            result = self.run_git([
                "diff", "--name-only", f"{to_commit}^", to_commit
            ])
        
        if result.success and result.stdout:
            return result.stdout.split('\n')
        return []

    def ensure_main_branch(self) -> Tuple[bool, str]:
        """Ensure we're on the main branch.

        Returns:
            Tuple of (success, message)
        """
        try:
            current_branch = self.get_current_branch()
            if not current_branch:
                return False, "Failed to get current branch"

            if current_branch == self.main_branch:
                return True, f"Already on {self.main_branch} branch"

            if self.switch_branch(self.main_branch):
                return True, f"Switched to {self.main_branch} branch"

            return False, f"Could not switch to {self.main_branch} branch (currently on {current_branch})"

        except Exception as e:
            return False, f"Git error: {str(e)}"

    def ensure_clean_workspace(self, message: str = "pre-init-codebase") -> bool:
        """Stash any uncommitted changes.

        Args:
            message: Stash message

        Returns:
            True if workspace is clean (or was successfully stashed)
        """
        try:
            success, _ = self.stash_if_dirty(message)
            return success
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Module-level read-only helpers for hooks / status commands
# ---------------------------------------------------------------------------
#
# These functions intentionally avoid the GitRunner class because hooks and
# status commands need:
#   1. No exceptions on missing / shallow / non-git repos (silent failure
#      with ``None`` return so the caller falls back gracefully).
#   2. Sub-second timeouts (a slow git call must not stall ``cmind init``,
#      a pre-commit hook, or VS Code's folderOpen task).
#   3. No mutation of the working tree, index, or any git state.
#
# Used by:
#   - rpg.models.RPG.set_git_meta(...) callers
#   - scripts/update_graphs.py status output
#   - (future Step 3) RPGService.sync_from_commit_diff

def _run_git_readonly(
    args: List[str],
    cwd: Path,
    timeout: float = 5.0,
) -> Optional[str]:
    """Run a read-only git command, return stdout stripped or None on any failure.

    Never raises.  Used by helpers below to keep them silent-fail.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def read_head(repo_dir: str | Path) -> Optional[dict]:
    """Read the current git HEAD for ``repo_dir``.

    Returns ``None`` if:
      * ``repo_dir`` does not exist
      * ``git`` is not on PATH
      * ``repo_dir`` is not a git working tree
      * The repository has no commits yet (unborn HEAD)

    Otherwise returns a dict with these keys (any individual value may be
    ``None`` on best-effort failures, e.g. detached HEAD has no branch):

        {
          "head_commit":    "8a3f9c1d4e2b...",   # 40-char SHA
          "head_short":     "8a3f9c1",            # short SHA
          "head_branch":    "main" | None,        # None on detached HEAD
          "head_timestamp": "2026-05-12T08:30:00+00:00",  # ISO 8601 UTC
        }

    Designed for SessionStart / pre-commit hook use — must never raise and
    must complete in well under a second on a healthy repo.
    """
    if not repo_dir:
        # Empty string would otherwise reach subprocess as cwd="" which
        # silently falls back to the caller's working directory — never
        # what callers of this helper intend.
        return None
    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        return None

    head_commit = _run_git_readonly(["rev-parse", "HEAD"], repo_path)
    if not head_commit:
        return None

    head_short = _run_git_readonly(["rev-parse", "--short", "HEAD"], repo_path)

    # symbolic-ref fails on detached HEAD with exit 128 — that's expected,
    # _run_git_readonly returns None and we keep head_branch as None.
    head_branch = _run_git_readonly(
        ["symbolic-ref", "--short", "HEAD"], repo_path
    )

    # ISO 8601 UTC timestamp of the HEAD commit.
    head_timestamp = _run_git_readonly(
        ["show", "-s", "--format=%cI", "HEAD"], repo_path
    )

    return {
        "head_commit": head_commit,
        "head_short": head_short,
        "head_branch": head_branch,
        "head_timestamp": head_timestamp,
    }


# ---------------------------------------------------------------------------
# Diff helpers — produce ``(modified, renames)`` from various git scopes.
# ---------------------------------------------------------------------------
#
# Every helper:
#   * returns ``(modified: list[str], renames: dict[old, new])``;
#   * returns ``([], {})`` on any failure (caller can't distinguish
#     "no changes" from "git not available" without consulting
#     ``read_head`` first — that's intentional, falling back to full
#     sync in either case is safe);
#   * filters to ``.py`` files at the source — CoderMind doesn't currently
#     parse anything else.  When that changes, lift the filter into the
#     caller.

# Single-letter status codes that ``git diff --name-status`` emits.
# Anything else (e.g. T = type change, U = unmerged) we ignore; full
# sync will eventually pick those up.
_GIT_STATUS_ADDED = "A"
_GIT_STATUS_DELETED = "D"
_GIT_STATUS_MODIFIED = "M"
_GIT_STATUS_RENAME_PREFIX = "R"  # may be followed by similarity score: "R98"
_GIT_STATUS_COPY_PREFIX = "C"


def _parse_name_status(
    raw: Optional[str],
    *,
    py_only: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    r"""Parse output of ``git diff --name-status -M``.

    Format per line is tab-separated:

      ``A\\tpath``         — added
      ``D\\tpath``         — deleted
      ``M\\tpath``         — modified
      ``R<score>\\told\\tnew`` — rename (score is similarity 0-100)
      ``C<score>\\told\\tnew`` — copy   (treated as rename for our purposes)

    Returns:
        ``(modified, renames)`` where ``modified`` lists every path that
        the dep_graph must re-examine (additions, deletions, plain
        modifications, **and** rename targets), and ``renames`` maps old
        paths to new paths so callers can pass it straight into
        :meth:`DependencyGraph.update_files(renames=...)`.
    """
    modified: List[str] = []
    renames: Dict[str, str] = {}
    if not raw:
        return modified, renames

    def _keep(p: str) -> bool:
        return (not py_only) or p.endswith(".py")

    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith(_GIT_STATUS_RENAME_PREFIX) or status.startswith(
            _GIT_STATUS_COPY_PREFIX
        ):
            if len(parts) < 3:
                continue
            old_path, new_path = parts[1], parts[2]
            if _keep(new_path) or _keep(old_path):
                renames[old_path] = new_path
                # update_files() treats the OLD path as a deletion (via
                # ``renames``) and the NEW path as something it must
                # reparse — so we surface the new path through the
                # modified list as well.
                if _keep(new_path):
                    modified.append(new_path)
            continue
        path = parts[1]
        if not _keep(path):
            continue
        if status in (_GIT_STATUS_ADDED, _GIT_STATUS_DELETED, _GIT_STATUS_MODIFIED):
            modified.append(path)
        # Type / unmerged / other status letters → ignore (caller will
        # fall back to full sync via the safety threshold if there are
        # many of them).
    return modified, renames


def staged_changes(
    repo_dir: str | Path,
) -> Tuple[List[str], Dict[str, str]]:
    """Return the paths in the **index** (i.e. ``git add``'d) vs HEAD.

    Used by the pre-commit hook: at hook time the new commit hasn't
    been recorded yet, so the right scope is "what's about to be
    committed" = index vs HEAD.  Anything in the working tree that
    hasn't been ``git add``'d is intentionally out of scope.

    Silent-fail: returns ``([], {})`` if not a git repo / git missing /
    timeout.  The caller's safety net is to fall back to full sync.
    """
    if not repo_dir:
        return [], {}
    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        return [], {}
    raw = _run_git_readonly(
        ["diff", "--cached", "--name-status", "-M", "HEAD"],
        repo_path,
    )
    if raw is None:
        # ``HEAD`` may not exist yet (unborn branch); try without it so
        # the very first commit's staged files still get picked up.
        raw = _run_git_readonly(
            ["diff", "--cached", "--name-status", "-M"],
            repo_path,
        )
    return _parse_name_status(raw)


def working_tree_changes(
    repo_dir: str | Path,
    *,
    include_untracked: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """Return tracked-and-modified + (optionally) untracked paths vs HEAD.

    Used by the **manual** ``update_graphs.py sync`` invocation (i.e.
    when a user runs it from the CLI without ``--staged-only``).  This
    covers everything dirty on disk, regardless of whether it's been
    ``git add``'d.

    Untracked files are reported as additions (no rename pairing
    possible since they have no git history).

    Silent-fail like its siblings.
    """
    if not repo_dir:
        return [], {}
    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        return [], {}

    raw = _run_git_readonly(
        ["diff", "--name-status", "-M", "HEAD"],
        repo_path,
    )
    modified, renames = _parse_name_status(raw)

    if include_untracked:
        untracked_raw = _run_git_readonly(
            ["ls-files", "--others", "--exclude-standard"],
            repo_path,
        )
        if untracked_raw:
            for line in untracked_raw.splitlines():
                line = line.strip()
                if line.endswith(".py") and line not in modified:
                    modified.append(line)
    return modified, renames


def changed_files_between(
    repo_dir: str | Path,
    old_ref: str,
    new_ref: str = "HEAD",
) -> Tuple[List[str], Dict[str, str]]:
    """Return ``.py`` changes between two arbitrary commits / refs.

    This is the workhorse for incremental sync: ``old_ref`` is the
    commit RPG was last synced against (from ``meta.git.head_commit``)
    and ``new_ref`` is typically the current HEAD.  Git stitches
    together every intermediate commit's diff for us, so this handles
    "user committed 5 times since last sync" naturally.

    Silent-fail returns ``([], {})``.  An empty list is **ambiguous**:
    it could mean "no .py files changed" or "old_ref doesn't exist any
    more in the current history".  The caller is responsible for
    pre-checking the relationship via :func:`merge_base` before
    interpreting this output as "incremental is safe".
    """
    if not repo_dir or not old_ref:
        return [], {}
    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        return [], {}
    raw = _run_git_readonly(
        ["diff", "--name-status", "-M", f"{old_ref}..{new_ref}"],
        repo_path,
    )
    return _parse_name_status(raw)


def merge_base(
    repo_dir: str | Path,
    ref_a: str,
    ref_b: str,
) -> Optional[str]:
    """Return the longest common ancestor commit of ``ref_a`` and ``ref_b``.

    Used by ``RPGService.sync_from_commit_diff`` to decide whether
    ``meta.git.head_commit`` is still on the current history:

      * ``merge_base(last, HEAD) == last`` → linear advance, safe to
        diff ``last..HEAD`` for incremental update.
      * ``merge_base(last, HEAD) != last`` → history was rewritten
        (rebase, amend, reset, branch fork); must fall back to full
        sync because ``last..HEAD`` would mix unrelated changes.

    Returns ``None`` on any failure — caller treats this the same as
    "diverged" and falls back to full sync.
    """
    if not repo_dir or not ref_a or not ref_b:
        return None
    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        return None
    return _run_git_readonly(
        ["merge-base", ref_a, ref_b],
        repo_path,
    )


def create_task_branch(
    repo_path: str,
    batch_id: str,
    stash_if_dirty: bool = True
) -> Tuple[bool, str, str]:
    """Create a new branch for a task batch, always from latest main HEAD.
    
    Key invariants (serial workflow, no concurrent batches):
    1. Always switch to main first — branches are NEVER created from other
       task branches.
    2. If a branch with the same name already exists (e.g., from a previous
       failed run), delete it and recreate from current main HEAD.  Reusing
       a stale branch causes merge conflicts because the old fork point is
       behind main.
    3. initial_commit is recorded AFTER switching to main, so it always
       reflects the latest main HEAD.
    
    Args:
        repo_path: Path to the repository
        batch_id: ID of the batch (used in branch name)
        stash_if_dirty: If True, stash uncommitted changes before switching
        
    Returns:
        Tuple of (success, branch_name, initial_commit)
    """
    git = GitRunner(repo_path)
    
    # Create sanitized branch name
    safe_id = sanitize_branch_component(batch_id, max_len=50, fallback="task")
    branch_name = f"task/{safe_id}"
    
    # Handle uncommitted changes
    was_stashed = False
    if stash_if_dirty:
        success, was_stashed = git.stash_if_dirty(f"pre-task-{safe_id}")
        if not success:
            return False, "", ""
    
    # ALWAYS switch to main first — this is the core invariant.
    # Branches must fork from latest main HEAD, never from another task branch.
    current_branch = git.get_current_branch()
    if current_branch != git.main_branch:
        if git.branch_exists(git.main_branch):
            if not git.switch_branch(git.main_branch):
                if was_stashed:
                    git.stash_pop()
                return False, "", ""
        else:
            git.logger.warning("Main branch does not exist, creating branch from current HEAD")
    
    initial_commit = git.get_head_commit()
    
    # If a branch with this name already exists (from a previous failed run),
    # delete it first.  The old branch has a stale fork point that would cause
    # merge conflicts.  We recreate from the current (latest) main HEAD.
    if git.branch_exists(branch_name):
        git.logger.info(
            f"Deleting stale branch {branch_name} (will recreate from main HEAD)"
        )
        git.delete_branch(branch_name, force=True)
    
    # Create new branch from current main HEAD
    success = git.create_branch(branch_name)
    
    # Restore stashed changes regardless of branch creation outcome
    if was_stashed:
        git.stash_pop()
    
    return success, branch_name, initial_commit


def complete_task_branch(
    repo_path: str,
    branch_name: str,
    success: bool,
) -> Tuple[bool, Optional[str]]:
    """Complete a task branch by merging (success) or abandoning (failure).
    
    Args:
        repo_path: Path to the repository
        branch_name: Task branch name
        success: Whether the task succeeded
        
    Returns:
        Tuple of (success, error_type)
        - success: True if operation succeeded
        - error_type: None on success, or error description
    """
    git = GitRunner(repo_path)
    
    # Check for uncommitted changes
    if git.has_uncommitted_changes():
        git.logger.warning("Uncommitted changes detected, committing before branch completion")
        commit_success, _ = git.stage_and_commit(f"WIP: auto-commit before completing {branch_name}")
        if not commit_success:
            return False, "commit_failed"
    
    if success:
        # Merge the branch
        merge_success, error_type = git.merge_branch(branch_name)
        if merge_success:
            # Delete the task branch
            git.delete_branch(branch_name)
            return True, None
        return False, error_type
    else:
        # Abandon the branch - switch to main
        if git.switch_branch(git.main_branch):
            return True, None
        return False, "switch_failed"
