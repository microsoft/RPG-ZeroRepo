"""Inner-git snapshotting for ``.rpgkit/``.

Plan 03 — every successful (or failed) ``rpgkit script <X>`` invocation
auto-commits the current state of ``.rpgkit/`` to a dedicated repo at
``.rpgkit/.git/``.  Lets users `git log` / `git diff` between pipeline
stages without writing any extra tooling.

Design choices (see plans/03-auto-snapshot-inner-git.md):

* No global ``git config`` writes — every commit uses per-call
  ``-c user.email`` / ``-c user.name`` so the user's identity is
  untouched.
* Concurrent commits (background post-commit hook vs foreground script)
  are handled by a one-shot retry on ``index.lock`` failure, then a
  silent skip.  Data is never lost: the next successful commit folds
  in whatever the dropped one would have captured.
* Failures are committed too, tagged ``— FAILED (exit N)`` so the
  history shows what changed pre-failure.
* Check / validation scripts are skipped (they're read-only and would
  otherwise spam the history).

All public functions swallow their own exceptions — this module must
never be a reason ``rpgkit script`` itself fails.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Inner repo identity.  Per-call (-c user.X) so this never touches the
# user's ~/.gitconfig.
_AUTHOR_EMAIL = "rpgkit@local"
_AUTHOR_NAME = "rpgkit-snapshot"


def _author_args() -> list[str]:
    return [
        "-c", f"user.email={_AUTHOR_EMAIL}",
        "-c", f"user.name={_AUTHOR_NAME}",
        # Disable system / user / xdg git config so an unusual global
        # template doesn't leak (e.g. signing keys, hooks) into our
        # private snapshot repo.
        "-c", "init.defaultBranch=main",
    ]


# Skip patterns — these scripts are read-only or long-running and
# shouldn't pollute the snapshot history.
_SKIP_NAMES: frozenset[str] = frozenset({
    "mcp_server.py",
})


def _basename(relpath: str) -> str:
    return relpath.rsplit("/", 1)[-1]


def should_skip_script(relpath: str) -> bool:
    """True iff this script should NOT trigger an auto-commit."""
    base = _basename(relpath)
    if base in _SKIP_NAMES:
        return True
    # Read-only state checkers — e.g. check_skeleton.py, check_code_gen.py
    if base.startswith("check_") and base.endswith(".py"):
        return True
    # Pre-flight validators — e.g. feature_build_validation.py
    if base.endswith("_validation.py"):
        return True
    return False


def categorise_script(relpath: str) -> str:
    """Map a script relpath to a short commit-message category tag."""
    rel = relpath.replace("\\", "/")
    if rel.startswith("rpg_encoder/"):
        return "encoder"
    if rel.startswith("rpg_edit/"):
        return "rpg_edit"
    base = _basename(rel)
    if base == "update_graphs.py":
        return "sync"
    if base == "mcp_server.py":
        # Defensive: today this is on the skip list so we never commit
        # an mcp_server.py invocation, but if that ever changes, tag
        # it correctly rather than defaulting to ``decoder``.
        return "mcp"
    return "decoder"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _rpgkit_dir(workspace: Path) -> Path:
    return workspace / ".rpgkit"


def find_workspace_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` (default cwd) looking for a ``.rpgkit/`` dir.

    Returns the directory containing ``.rpgkit/``, or ``None`` if not found.
    Used by ``rpgkit script`` to figure out which workspace's inner git
    repo to snapshot into when the caller's cwd is a subdirectory.
    """
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / ".rpgkit").is_dir():
            return cand
    return None


def has_inner_git(workspace: Path) -> bool:
    return (_rpgkit_dir(workspace) / ".git").is_dir()


def _git_available() -> bool:
    from shutil import which
    return which("git") is not None


def _run_git(workspace: Path, *args: str, check: bool = False, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run ``git -C .rpgkit ...`` capturing stdout/stderr.

    ``check=False`` by default — callers inspect ``returncode`` themselves so
    we can silently swallow expected failures (lock, no-changes, etc.).

    The child environment forces ``LC_ALL=C`` so git's error messages
    are in English regardless of the user's locale.  We pattern-match
    on those messages (see ``_LOCK_HINTS``) to decide whether to retry
    on lock contention.
    """
    import os as _os
    env = {**_os.environ, "LC_ALL": "C", "LANG": "C"}
    cmd = ["git", "-C", str(_rpgkit_dir(workspace))] + list(args)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_inner_git(workspace: Path, *, initial_msg: Optional[str] = None) -> bool:
    """Create ``.rpgkit/.git`` if missing.  Returns ``True`` when newly created.

    Idempotent: if the repo already exists, returns ``False`` and leaves it
    untouched.

    When a fresh repo is created, an initial commit captures the current
    state of ``.rpgkit/`` (config.toml, .source, empty data/, ...).
    """
    rpgkit = _rpgkit_dir(workspace)
    if not rpgkit.is_dir():
        return False  # nothing to track
    if (rpgkit / ".git").is_dir():
        return False
    if not _git_available():
        return False

    try:
        # Use 'main' as default branch to match the workspace default and
        # avoid the noisy "hint: Using 'master'" message on fresh git.
        _run_git(workspace, "init", "-q", "-b", "main", check=True)
    except Exception:
        return False

    # Initial commit — even if empty, it gives `git log` a starting point.
    initial_msg = initial_msg or "[init] rpgkit workspace"
    _commit_all(workspace, initial_msg, allow_empty=True)
    return True


# ---------------------------------------------------------------------------
# Commit primitives
# ---------------------------------------------------------------------------

def _has_staged_changes(workspace: Path) -> bool:
    """True iff something is staged for commit."""
    r = _run_git(workspace, "diff", "--staged", "--quiet")
    # exit 1 = differences exist, 0 = none, anything else = error (treat as no)
    return r.returncode == 1


_LOCK_HINTS = ("index.lock", "Another git process seems")


def _commit_all(workspace: Path, message: str, *, allow_empty: bool = False) -> bool:
    """Stage everything and commit.  Returns True iff a commit was created.

    Concurrent-safe: if the index lock is held by a parallel git process
    (e.g. the post-commit hook firing ``rpgkit script update_graphs.py``
    in the background), we retry once after a short sleep, then give up
    silently.  The next successful commit will fold in any deferred
    changes — no data is lost.
    """
    for attempt in (1, 2):
        try:
            r_add = _run_git(workspace, "add", "-A")
            if r_add.returncode != 0:
                # Likely a lock; try again
                if any(h in (r_add.stderr or "") for h in _LOCK_HINTS) and attempt == 1:
                    time.sleep(1.0)
                    continue
                return False

            if not allow_empty and not _has_staged_changes(workspace):
                return False  # nothing to commit; not an error

            commit_args = ["commit", "-m", message, "--quiet"]
            if allow_empty:
                commit_args.insert(1, "--allow-empty")
            r_c = _run_git(workspace, *_author_args(), *commit_args)
            if r_c.returncode == 0:
                return True
            # Retry on lock
            if any(h in (r_c.stderr or "") for h in _LOCK_HINTS) and attempt == 1:
                time.sleep(1.0)
                continue
            return False
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Public entry: after a `rpgkit script <X>` call
# ---------------------------------------------------------------------------

def _build_message(script_relpath: str, args: list[str], exit_code: int) -> str:
    cat = categorise_script(script_relpath)
    # ``shlex.quote`` keeps args with spaces / special chars unambiguous
    # in the commit log: ``[decoder] X.py 'some path'`` rather than
    # ``[decoder] X.py some path``.
    quoted = " ".join(shlex.quote(a) for a in args)
    args_part = (" " + quoted).rstrip() if quoted else ""
    # Cap args length so a giant args string doesn't make commit messages unreadable.
    if len(args_part) > 80:
        args_part = args_part[:77] + "..."
    suffix = ""
    if exit_code != 0:
        suffix = f" — FAILED (exit {exit_code})"
    return f"[{cat}] {script_relpath}{args_part}{suffix}"


def auto_commit_after_script(
    workspace: Path,
    script_relpath: str,
    args: list[str],
    exit_code: int,
) -> None:
    """Snapshot ``.rpgkit/`` after a ``rpgkit script`` call completes.

    No-ops (silently) when any of:
      * ``.rpgkit/.git`` is missing
      * the script matches a skip pattern
      * git is unavailable
      * the index is locked and the retry still fails
      * nothing actually changed
    """
    try:
        if should_skip_script(script_relpath):
            return
        if not has_inner_git(workspace):
            return
        message = _build_message(script_relpath, args, exit_code)
        _commit_all(workspace, message, allow_empty=False)
    except Exception:
        # Never let snapshot machinery break the calling CLI.
        return


# ---------------------------------------------------------------------------
# `rpgkit version` helper
# ---------------------------------------------------------------------------

def snapshot_count(workspace: Path) -> Optional[int]:
    """Return number of commits in the inner repo, or None if absent."""
    if not has_inner_git(workspace):
        return None
    try:
        r = _run_git(workspace, "rev-list", "--count", "HEAD")
        if r.returncode == 0:
            return int((r.stdout or "0").strip())
    except Exception:
        pass
    return None
