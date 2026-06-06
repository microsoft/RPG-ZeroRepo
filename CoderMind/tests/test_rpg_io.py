"""Tests for ``scripts.common.rpg_io`` (atomic write + recovery)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make sure the bundled scripts/ tree is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from common import rpg_io  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_git() -> bool:
    from shutil import which
    return which("git") is not None


requires_git = pytest.mark.skipif(not _has_git(), reason="git not on PATH")


def _make_home_layout(tmp_path: Path, hash_id: str = "abc123def456") -> Path:
    """Create the ``~/.cmind/workspaces/<workspace-id>/`` layout for tests.

    Returns the home_dir (the dir that gets ``git init``).  Caller is
    responsible for git-initialising and snapshotting it.
    """
    home_root = tmp_path / ".cmind" / "workspaces" / hash_id
    (home_root / "data").mkdir(parents=True)
    return home_root


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Convenience wrapper for tests."""
    env = {
        **os.environ,
        "LC_ALL": "C", "LANG": "C",
        "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@x",
        "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@x",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, env=env, timeout=10, check=True,
    )


# ---------------------------------------------------------------------------
# atomic_write_rpg
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "rpg.json"
        rpg_io.atomic_write_rpg(target, {"hello": "world"})
        assert target.is_file()
        assert json.loads(target.read_text()) == {"hello": "world"}

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "rpg.json"
        target.write_text('{"old": true}')
        rpg_io.atomic_write_rpg(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "rpg.json"
        rpg_io.atomic_write_rpg(target, {"x": 1})
        assert target.is_file()

    def test_no_tmp_leftover_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "rpg.json"
        rpg_io.atomic_write_rpg(target, {"x": 1})
        tmp = target.with_suffix(".json.tmp")
        assert not tmp.exists()

    def test_no_tmp_leftover_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``os.replace`` fails, the partial ``.tmp`` is cleaned up."""
        target = tmp_path / "rpg.json"
        # Pre-existing valid content we shouldn't lose.
        target.write_text('{"existing": "data"}')

        def boom(*_a, **_kw):
            raise OSError("simulated replace failure")
        monkeypatch.setattr(rpg_io.os, "replace", boom)

        with pytest.raises(OSError):
            rpg_io.atomic_write_rpg(target, {"new": "would-be"})

        # Original file untouched
        assert json.loads(target.read_text()) == {"existing": "data"}
        # No stray .tmp
        tmp = target.with_suffix(".json.tmp")
        assert not tmp.exists()

    def test_preserves_unicode(self, tmp_path: Path) -> None:
        target = tmp_path / "rpg.json"
        rpg_io.atomic_write_rpg(target, {"name": "测试 \u2014 ✓"})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded["name"] == "测试 \u2014 ✓"

    def test_forwards_dump_kwargs(self, tmp_path: Path) -> None:
        """``**dump_kwargs`` is forwarded to ``json.dump`` so callers
        can pass custom serialiser hooks such as ``default=``."""
        target = tmp_path / "rpg.json"

        class _NotSerialisable:
            def to_dict(self):
                return {"recovered": True}

        # Without ``default=`` this would raise TypeError; passing the
        # legacy lambda the encoder used proves the kwarg reaches json.dump.
        rpg_io.atomic_write_rpg(
            target,
            {"obj": _NotSerialisable()},
            default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o),
        )
        assert json.loads(target.read_text()) == {"obj": {"recovered": True}}

    def test_no_partial_file_on_serialise_failure(self, tmp_path: Path) -> None:
        """A TypeError mid-``json.dump`` (no ``default=`` for an
        unserialisable object) must leave the original file intact and
        clean up the ``.tmp`` — the bug we kept hitting when the bench
        killed cobra encode mid-write."""
        target = tmp_path / "rpg.json"
        target.write_text('{"existing": "intact"}')

        class _Bad:
            pass

        with pytest.raises(TypeError):
            rpg_io.atomic_write_rpg(target, {"obj": _Bad()})

        # Original survives because os.replace never ran.
        assert json.loads(target.read_text()) == {"existing": "intact"}
        # The .tmp file must be cleaned up so a re-run doesn't see stale
        # crud from the failed attempt.
        tmp = target.with_suffix(".json.tmp")
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# safe_load_rpg — success path + propagation of FileNotFoundError
# ---------------------------------------------------------------------------

class TestSafeLoadBasic:
    def test_returns_data_on_valid_file(self, tmp_path: Path) -> None:
        target = tmp_path / "rpg.json"
        target.write_text(json.dumps({"ok": True}))
        assert rpg_io.safe_load_rpg(target) == {"ok": True}

    def test_raises_filenotfound_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            rpg_io.safe_load_rpg(tmp_path / "absent.json")

    def test_raises_jsondecodeerror_when_no_inner_git(
        self, tmp_path: Path
    ) -> None:
        """Without an inner-git nearby, corruption propagates as-is."""
        target = tmp_path / "rpg.json"
        target.write_text("not { valid json")
        with pytest.raises(json.JSONDecodeError):
            rpg_io.safe_load_rpg(target)


# ---------------------------------------------------------------------------
# safe_load_rpg — recovery via inner git
# ---------------------------------------------------------------------------

@requires_git
class TestSafeLoadRecovery:
    def _setup_with_history(self, tmp_path: Path) -> tuple[Path, Path, dict]:
        """Build a home-layout with one good snapshot of data/rpg.json.

        Returns (home_dir, target_path, good_payload).
        """
        home = _make_home_layout(tmp_path)
        target = home / "data" / "rpg.json"

        # Good v1 → commit
        good = {"version": 1, "nodes": [{"id": "x"}]}
        rpg_io.atomic_write_rpg(target, good)
        _git(home, "init", "-q", "-b", "main")
        _git(home, "add", "-A")
        _git(home, "commit", "-q", "-m", "v1")
        return home, target, good

    def test_recovers_from_last_good_snapshot(self, tmp_path: Path) -> None:
        home, target, good = self._setup_with_history(tmp_path)

        # Corrupt the file (simulate interrupted write).
        target.write_text('{"version": 2, "nod')  # truncated

        recovered = rpg_io.safe_load_rpg(target)
        assert recovered == good

        # File on disk has been healed too.
        assert json.loads(target.read_text()) == good
        # No stray .tmp from the heal write.
        assert not (home / "data" / "rpg.json.tmp").exists()

    def test_skips_bad_snapshots(self, tmp_path: Path) -> None:
        """If recent commits are also broken, walks further back."""
        home, target, good = self._setup_with_history(tmp_path)

        # Commit an invalid JSON snapshot to bury the good one.
        target.write_text('{"broken')
        _git(home, "add", "-A")
        _git(home, "commit", "-q", "-m", "broken commit")

        recovered = rpg_io.safe_load_rpg(target)
        assert recovered == good

    def test_returns_none_when_history_has_no_valid_snapshot(
        self, tmp_path: Path
    ) -> None:
        """No valid history → original parse error re-raised."""
        home = _make_home_layout(tmp_path)
        target = home / "data" / "rpg.json"

        # First commit: already broken (pathological).
        target.write_text('{not json')
        _git(home, "init", "-q", "-b", "main")
        _git(home, "add", "-A")
        _git(home, "commit", "-q", "-m", "broken from the start")

        # Read it: corruption can't be recovered.
        with pytest.raises(json.JSONDecodeError):
            rpg_io.safe_load_rpg(target)

    def test_works_when_target_outside_known_layout(
        self, tmp_path: Path
    ) -> None:
        """For paths that don't look like ``~/.cmind/workspaces/...``,
        recovery silently no-ops and the original error re-raises."""
        target = tmp_path / "rpg.json"  # not in a home-layout
        target.write_text("not valid")
        with pytest.raises(json.JSONDecodeError):
            rpg_io.safe_load_rpg(target)
