"""Atomic write and corruption-recovery helpers for ``rpg.json``.

``rpg.json`` is the central pipeline artefact; corruption blocks every
downstream stage.  Two failure modes have hurt users in the past:

1. **Interrupted writes** — encoder dumps the full JSON in one
   ``json.dump`` call.  If the process is killed (Ctrl-C, OOM, power
   loss) mid-write, the file is left half-truncated and every
   subsequent read raises ``JSONDecodeError``.  The workspace is
   effectively bricked until the user re-runs the encoder.

2. **Silent corruption with no recovery path** — once truncated, the
   only "fix" was to re-encode from scratch.  But the inner-git
   snapshot repo already holds the previous good state at
   ``~/.cmind/workspaces/<workspace-id>/.git/``; we just weren't using it.

This module fixes both with two complementary primitives:

* :func:`atomic_write_rpg` — serialise to ``<path>.tmp`` first, then
  ``os.replace()`` into place.  POSIX (and Windows since 2018)
  guarantee the rename is atomic, so any reader either sees the
  complete previous version or the complete new one — never a partial
  write.
* :func:`safe_load_rpg` — on ``JSONDecodeError`` (corruption), walk
  the inner-git history of the workspace looking for the most recent
  commit where the file parsed cleanly, restore it on disk (so
  subsequent callers don't pay the recovery cost), emit a single
  warning to ``logging``, and return the recovered data.  If no good
  snapshot exists, the original ``JSONDecodeError`` is re-raised so
  callers can decide how to degrade.

Design constraints
------------------

* No new dependencies; uses ``os`` + ``subprocess`` + ``json``.
* Recovery is best-effort: a failure to invoke git, a missing inner
  repo, or a missing good snapshot all fall through cleanly to
  re-raising the original parse error.
* The recovered file is written back atomically (same
  :func:`atomic_write_rpg` path) so the workspace doesn't relapse on
  the next read.
* Logging uses ``logging.getLogger(__name__)`` so calls from scripts
  that have configured logging will surface the warning, while
  callers in quiet contexts (e.g. MCP server with stderr-redirect)
  won't be perturbed.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: atomic write
# ---------------------------------------------------------------------------

def atomic_write_rpg(
    path: Path | str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    **dump_kwargs: Any,
) -> None:
    """Serialise ``data`` to ``path`` atomically as JSON.

    Writes to ``<path>.tmp`` first then renames into place.  If the
    write fails mid-way (e.g. disk full), the original file (if any)
    remains intact and we clean up the partial ``.tmp``.

    The signature matches ``json.dump`` for indent / ensure_ascii so
    callers swapping ``open(path, "w") + json.dump`` for this helper
    don't have to rethink their JSON formatting choices. Additional
    ``**dump_kwargs`` are forwarded to ``json.dump`` (e.g. ``default=``
    for non-serialisable encoder rounds), letting every legacy caller
    migrate without losing custom serialiser hooks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                data, f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                **dump_kwargs,
            )
            f.write("\n")
            # fsync gives us strong durability guarantees: an os.replace
            # immediately after a crash could otherwise expose the
            # rename without the bytes if the kernel hadn't flushed.
            # On filesystems that don't support fsync (rare), the call
            # is a harmless no-op.
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        # Clean up a stray .tmp so the next attempt isn't confused by
        # a leftover.  Swallowing this secondary error preserves the
        # original traceback for the caller.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public: safe load with inner-git recovery
# ---------------------------------------------------------------------------

def safe_load_rpg(path: Path | str) -> Any:
    """Parse the JSON at ``path``, with automatic recovery on corruption.

    Behaviour:

    * Success path — file parses cleanly: return the deserialised data,
      no side effects.
    * Corruption path — ``json.JSONDecodeError`` from the read attempt
      triggers :func:`_try_restore_from_inner_git`, which scans the
      inner-git repo looking for the most recent commit where
      the file was valid JSON.  If one is found, the file is rewritten
      on disk (atomically) with that content, a warning is logged, and
      the recovered data is returned.
    * Unrecoverable path — no inner git, no valid history, or git
      unavailable: the original ``JSONDecodeError`` is re-raised.
    * Missing file: ``FileNotFoundError`` is propagated unchanged
      (recovery is for *corruption*, not for never-encoded
      workspaces).
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        recovered = _try_restore_from_inner_git(path, exc)
        if recovered is None:
            # Recovery failed — surface the original parse error so the
            # caller can decide how to react (MCP server returns
            # ``rpg_unavailable``; scripts may want to abort).
            raise
        return recovered


# ---------------------------------------------------------------------------
# Internal: inner-git recovery
# ---------------------------------------------------------------------------

# Filenames inside the inner-git repo that we know how to recover.
# Mirrors the layout produced by :mod:`cmind_cli._inner_git`:
# ``data/rpg.json``, ``data/feature_spec.json``, etc.
def _git_relpath_for(path: Path) -> Optional[str]:
    """Return the path relative to the home-workspace dir for git lookup.

    ``rpg.json`` lives at ``~/.cmind/workspaces/<workspace-id>/data/rpg.json``;
    the inner git repo is rooted at ``~/.cmind/workspaces/<workspace-id>/``,
    so the path we ``git checkout`` is ``data/rpg.json``.  Falls back
    to ``None`` when ``path`` doesn't look like it lives under such a
    home dir (e.g. test fixtures passing absolute paths into ``/tmp``).
    """
    parts = path.resolve().parts
    # Look for ".cmind/workspaces/<workspace-id>/..." in the path's components.
    try:
        idx = parts.index(".cmind")
        if (
            idx + 2 < len(parts)
            and parts[idx + 1] == "workspaces"
            # parts[idx+2] is the hash
        ):
            return "/".join(parts[idx + 3 :])
    except ValueError:
        pass
    return None


def _inner_git_dir_for(path: Path) -> Optional[Path]:
    """Find the home-workspace dir (containing ``.git/``) for ``path``."""
    cur = path.resolve().parent
    while True:
        if (cur / ".git").is_dir() and cur.parent.name == "workspaces":
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def _try_restore_from_inner_git(
    path: Path, original_exc: json.JSONDecodeError
) -> Optional[Any]:
    """Recover ``path`` from inner-git; return data or None on failure.

    Walks the linear history of the inner repo from HEAD backwards,
    fetching the file content at each commit via ``git show``.  The
    first commit where the content parses as valid JSON wins.  When
    a winner is found we also re-write the file on disk (atomically)
    so subsequent reads don't pay the recovery cost.
    """
    git_dir = _inner_git_dir_for(path)
    if git_dir is None:
        return None
    relpath = _git_relpath_for(path)
    if relpath is None:
        return None
    from shutil import which
    if which("git") is None:
        return None

    # Force English git messages (consistent with _inner_git.py).
    # Strip any inherited ``GIT_*`` vars (e.g. ``GIT_DIR``,
    # ``GIT_INDEX_FILE``) that would point ``git`` at the **outer**
    # repository when this recovery runs inside a hook context.  This
    # mirrors the env-sanitisation done in ``cmind_cli._inner_git._run_git``.
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_INDEX_FILE", "GIT_DIR",
                        "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY")}
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    # Walk linear history (most recent first).  ``--follow`` keeps
    # working when a script ever renames data files in the future.
    try:
        log = subprocess.run(
            ["git", "-C", str(git_dir), "log", "--follow",
             "--format=%H", "--", relpath],
            capture_output=True, text=True, env=env, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if log.returncode != 0:
        return None

    commits = [c.strip() for c in log.stdout.splitlines() if c.strip()]
    for commit in commits:
        try:
            show = subprocess.run(
                ["git", "-C", str(git_dir), "show", f"{commit}:{relpath}"],
                capture_output=True, text=True, env=env, timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if show.returncode != 0:
            continue
        try:
            data = json.loads(show.stdout)
        except json.JSONDecodeError:
            # Older snapshot also broken — skip and keep walking.
            continue

        # Found a good snapshot — restore it on disk + return.
        try:
            atomic_write_rpg(path, data)
        except OSError:
            # If we can't write back (read-only fs?), still return the
            # recovered data so the caller can proceed; the next
            # successful write will heal the file on disk.
            pass

        logger.warning(
            "rpg-io: %s was corrupted (%s at line %d col %d); auto-restored "
            "from inner-git snapshot %s. Run `cmind version` to see the "
            "exact inner-git path.",
            path,
            original_exc.msg,
            original_exc.lineno,
            original_exc.colno,
            commit[:8],
        )
        return data

    return None
