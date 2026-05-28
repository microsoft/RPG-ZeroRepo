"""Inner-git snapshotting for the user-home workspace directory.

Every successful (or failed) ``cmind script <X>`` invocation
auto-commits the current state of the per-workspace home directory at
``~/.cmind/workspaces/<workspace-id>/`` into a dedicated git repo at
``~/.cmind/workspaces/<workspace-id>/.git/``. This lets ``git log`` and
``git diff`` show how pipeline stages change between runs.

What gets tracked:

* ``data/`` — all encoder / pipeline output (rpg.json, dep_graph.json,
  feature_*.json, …)
* ``logs/`` (except ``logs/copilot/``) — per-stage text/JSONL logs;
  tracking them lets users ``git log -p logs/<stage>.log`` to debug
  pipeline regressions across snapshots.
* ``.meta.toml`` — captures channel + CLI version at each snapshot;
  changes only on ``cmind init/update``.

What is NOT tracked (see :data:`_INNER_GIT_IGNORE` below):

* ``logs/copilot/`` — full LLM session traces, MB-scale per run, too
  noisy and too large to be useful in snapshot history.
* The inner ``.git/`` itself — git's own auto-exclusion.

Design choices:

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
never be a reason ``cmind script`` itself fails.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import _storage


# Environment variables set by ``cmind hook <name>`` before invoking
# any ``cmind script`` calls.  They flow through every subprocess so
# the snapshot commit message can record *which* git hook fired *which*
# user-facing commit instead of just naming the underlying script.
#
# Set only by :func:`cmind_cli.hook` -- never by manual invocations -
# so the presence of ``CMIND_HOOK`` is a reliable trigger-source flag.
_ENV_HOOK_NAME = "CMIND_HOOK"        # e.g. "post-commit" / "pre-commit"
_ENV_HOOK_SHA = "CMIND_HOOK_SHA"     # short SHA of the user-facing commit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Inner repo identity.  Per-call (-c user.X) so this never touches the
# user's ~/.gitconfig.
_AUTHOR_EMAIL = "cmind@local"
_AUTHOR_NAME = "cmind-snapshot"


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


# Contents of the ``.gitignore`` written into the inner repo on init.
#
# ``logs/`` is tracked so users can run ``git log -p logs/<stage>.log``
# to inspect how a pipeline stage's output changed between snapshots.
#
# ``logs/copilot/`` is excluded: it contains full LLM session traces
# (typically MB per session) and would dominate the snapshot history.
_INNER_GIT_IGNORE = """\
# Managed by cmind-cli: do not edit.
# Logs are tracked to support `git log -p logs/<stage>.log` debugging.
# Exception: logs/copilot/ holds LLM session traces (large, not useful
# in history); inspect those files directly.
logs/copilot/
"""


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
        # mcp_server.py is on the skip list, so this branch is unreachable
        # today.  Kept so adjusting the skip list still produces a correct
        # tag instead of falling through to "decoder".
        return "mcp"
    return "decoder"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _inner_git_dir(workspace: Path) -> Path:
    """Return the home directory used as ``git -C <dir>`` for the snapshots.

    The directory is ``~/.cmind/workspaces/<workspace-id>/``; the inner repo's
    ``.git`` sits directly inside it.
    """
    return _storage.home_workspace_dir(workspace)


# Backwards-compatible alias for the (now-misleading) historical name
# used in earlier docstrings.  No external caller should rely on this;
# it stays only to keep grep-friendly when reading older commit
# messages and plan documents.
_cmind_dir = _inner_git_dir


def find_workspace_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` (default cwd) looking for a workspace marker.

    Returns the directory containing ``.cmind/config.toml`` (the
    workspace marker), or ``None`` if not found.  Used by
    ``cmind script`` to figure out which workspace's inner git repo to
    snapshot into when the caller's cwd is a subdirectory.
    """
    return _storage.find_workspace_root_from(start)


def has_inner_git(workspace: Path) -> bool:
    """True iff a ``.git`` directory exists under the workspace's home dir."""
    return (_inner_git_dir(workspace) / ".git").is_dir()


def _git_available() -> bool:
    from shutil import which
    return which("git") is not None


def _run_git(workspace: Path, *args: str, check: bool = False, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <home_dir> ...`` capturing stdout/stderr.

    ``check=False`` by default — callers inspect ``returncode`` themselves so
    we can silently swallow expected failures (lock, no-changes, etc.).

    The child environment forces ``LC_ALL=C`` so git's error messages
    are in English regardless of the user's locale.  We pattern-match
    on those messages (see ``_LOCK_HINTS``) to decide whether to retry
    on lock contention.
    """
    import os as _os
    env = {**_os.environ, "LC_ALL": "C", "LANG": "C"}
    # Strip inherited git env vars: a foreground hook caller may have set
    # GIT_INDEX_FILE / GIT_DIR / GIT_WORK_TREE pointing at the outer repo.
    # If we leak those into the inner-git call the outer repo's index gets
    # corrupted (entries from $HOME/.cmind get written into the outer index.lock).
    for _v in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY"):
        env.pop(_v, None)
    cmd = ["git", "-C", str(_inner_git_dir(workspace))] + list(args)
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
    """Create ``~/.cmind/workspaces/<workspace-id>/.git`` if missing.

    Returns ``True`` when a fresh repo was created, ``False`` when it
    already existed or when setup was skipped (git missing, home dir
    unavailable, …).

    The home dir must already exist — it's the responsibility of
    ``ensure_workspace_storage`` (called from ``cmind init/update``
    earlier in the bootstrap) to create it.  We don't create it here
    because that requires picking a ``channel`` (bundle vs legacy),
    which is information only the caller has.

    When a fresh repo is created we also drop a ``.gitignore`` that
    excludes ``logs/copilot/`` (LLM session traces — large, not useful
    in history; see :data:`_INNER_GIT_IGNORE`), then commit the current
    state of ``data/`` + ``.meta.toml`` so ``git log`` has a starting
    point.
    """
    home_dir = _inner_git_dir(workspace)
    if not home_dir.is_dir():
        return False  # ensure_workspace_storage hasn't run yet
    if (home_dir / ".git").is_dir():
        return False
    if not _git_available():
        return False

    try:
        # Use 'main' as default branch to match the workspace default and
        # avoid the noisy "hint: Using 'master'" message on fresh git.
        _run_git(workspace, "init", "-q", "-b", "main", check=True)
    except Exception:
        return False

    # Drop the ignore file so logs don't appear in `git status` from the
    # very first commit.  Best-effort; failure here doesn't block.
    try:
        (home_dir / ".gitignore").write_text(_INNER_GIT_IGNORE, encoding="utf-8")
    except OSError:
        pass

    # Initial commit — even if empty, it gives `git log` a starting point.
    initial_msg = initial_msg or "[init] cmind workspace"
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


def _ensure_gitignore_current(workspace: Path) -> None:
    """Rewrite the inner repo's ``.gitignore`` if it drifted from the
    current :data:`_INNER_GIT_IGNORE`.

    Called before every commit so existing inner repos that were
    initialised under an older ignore policy (e.g. the original
    "ignore all of ``logs/``" rule) silently upgrade on next snapshot.
    No-op when the file is already up to date.
    """
    home_dir = _inner_git_dir(workspace)
    gi = home_dir / ".gitignore"
    try:
        current = gi.read_text(encoding="utf-8") if gi.is_file() else ""
        if current != _INNER_GIT_IGNORE:
            gi.write_text(_INNER_GIT_IGNORE, encoding="utf-8")
    except OSError:
        # Best-effort: ignore policy is not critical enough to fail a commit.
        pass


def _commit_all(workspace: Path, message: str, *, allow_empty: bool = False) -> bool:
    """Stage everything and commit.  Returns True iff a commit was created.

    Concurrent-safe: if the index lock is held by a parallel git process
    (e.g. the post-commit hook firing ``cmind script update_graphs.py``
    in the background), we retry once after a short sleep, then give up
    silently.  The next successful commit will fold in any deferred
    changes — no data is lost.
    """
    _ensure_gitignore_current(workspace)
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
# Public entry: after a `cmind script <X>` call
# ---------------------------------------------------------------------------

def _build_message(script_relpath: str, args: list[str], exit_code: int) -> str:
    """Compose the inner-git commit message for a ``cmind script`` call.

    Two output shapes:

    * **Hook-triggered** (``CMIND_HOOK`` is set by ``cmind hook``)::

          [hook:post-commit @ a1b2c3d] update-rpg
          [hook:pre-commit  @ a1b2c3d] sync --staged-only

      Both the triggering hook name and the user-facing commit short
      SHA are surfaced so ``git log`` in the inner repo reads as a
      timeline of *user activity*, not a timeline of internal scripts.

    * **Manual** (no ``CMIND_HOOK``)::

          [decoder] feature_build.py
          [encoder] rpg_encoder/run_encode.py --json
          [sync]    update_graphs.py update-rpg — FAILED (exit 2)

      Tagged by category (see :func:`categorise_script`) plus the full
      script relpath - kept verbose so power-users running scripts by
      hand can see exactly which file produced each snapshot.
    """
    suffix = f" — FAILED (exit {exit_code})" if exit_code != 0 else ""

    hook = os.environ.get(_ENV_HOOK_NAME, "").strip()
    if hook:
        # Action = first positional arg (the script's subcommand, e.g.
        # ``update-rpg``/``sync``) when present, otherwise the script
        # stem.  Subsequent args are appended but capped so the message
        # stays one-line friendly in ``git log --oneline``.
        if args:
            action = args[0]
            extra = " ".join(shlex.quote(a) for a in args[1:])
            extra_part = (" " + extra) if extra else ""
        else:
            action = _basename(script_relpath).removesuffix(".py")
            extra_part = ""
        if len(extra_part) > 40:
            extra_part = extra_part[:37] + "..."
        sha = os.environ.get(_ENV_HOOK_SHA, "").strip()
        sha_part = f" @ {sha}" if sha and sha != "?" else ""
        return f"[hook:{hook}{sha_part}] {action}{extra_part}{suffix}"

    # Manual / interactive path -- preserve historical format.
    cat = categorise_script(script_relpath)
    quoted = " ".join(shlex.quote(a) for a in args)
    args_part = (" " + quoted).rstrip() if quoted else ""
    if len(args_part) > 80:
        args_part = args_part[:77] + "..."
    return f"[{cat}] {script_relpath}{args_part}{suffix}"


def auto_commit_after_script(
    workspace: Path,
    script_relpath: str,
    args: list[str],
    exit_code: int,
) -> None:
    """Snapshot ``.cmind/`` after a ``cmind script`` call completes.

    No-ops (silently) when any of:
      * ``.cmind/.git`` is missing
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
# `cmind version` helper
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
