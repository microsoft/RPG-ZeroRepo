#!/usr/bin/env python3
"""Tests for the 4 Step-3 polish items.

A. ``_resolve_git_hooks_dir`` recognises git **worktree** ``.git`` files
   (worktree-as-file format ``gitdir: <path>``) in addition to the
   ordinary ``.git`` directory.

B. ``update_graphs.py status`` text output surfaces **branch** info
   alongside commit shorts so the user sees branch-switch staleness
   immediately.

C. ``sync_from_commit_diff`` refreshes ``meta.git.head_branch`` /
   ``head_timestamp`` even in **noop** mode (covers ``git checkout
   other_branch_at_same_sha`` and ``git branch -m`` cases).

D. Git hook setup installs post-commit/post-merge dispatcher hooks and
    removes CoderMind-owned pre-commit blocks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "scripts"))

import cmind_cli  # noqa: E402
from rpg.models import RPG  # noqa: E402
from rpg.service import RPGService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sh(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


# ===========================================================================
# A. Worktree support
# ===========================================================================

def test_resolve_git_hooks_dir_for_plain_repo(tmp_path):
    """Ordinary ``.git`` directory: hooks live at ``.git/hooks``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")
    hooks = cmind_cli._resolve_git_hooks_dir(repo)
    assert hooks is not None
    assert hooks == repo / ".git" / "hooks"
    assert hooks.is_dir()


def test_resolve_git_hooks_dir_for_worktree(tmp_path):
    """``git worktree add`` workspaces have ``.git`` as a *file* pointing at ``<main>/.git/worktrees/<name>``.  Hook installation must succeed and shared hooks must live in the main repo's ``hooks/`` directory."""
    main = tmp_path / "main"
    main.mkdir()
    _sh(main, "init", "-q", "-b", "main")
    _sh(main, "config", "user.email", "t@t.com")
    _sh(main, "config", "user.name", "t")
    (main / "f.py").write_text("x = 1\n")
    _sh(main, "add", ".")
    _sh(main, "commit", "-q", "-m", "init")

    wt = tmp_path / "wt"
    _sh(main, "worktree", "add", "--detach", str(wt))

    # Sanity: ``.git`` inside the worktree is indeed a file, not a dir
    assert (wt / ".git").is_file()

    hooks = cmind_cli._resolve_git_hooks_dir(wt)
    assert hooks is not None, "worktree must resolve to a hooks dir"
    # Worktrees share the main repo's hooks
    assert hooks == main / ".git" / "hooks"


def test_resolve_git_hooks_dir_for_non_git_returns_none(tmp_path):
    assert cmind_cli._resolve_git_hooks_dir(tmp_path) is None


def test_resolve_git_hooks_dir_honors_core_hooks_path_override(tmp_path):
    """``core.hooksPath`` redirects git to look for hooks elsewhere.

    Teams using ``husky`` / ``pre-commit`` / ``lefthook`` set this to
    a checked-in directory (typically ``.husky/``).  Without this
    detection, the installer would write into ``.git/hooks/`` where
    git no longer reads from, leaving a silent no-op install.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q", "-b", "main")
    custom_hooks = repo / ".husky"
    custom_hooks.mkdir()
    _sh(repo, "config", "core.hooksPath", str(custom_hooks))

    resolved = cmind_cli._resolve_git_hooks_dir(repo)
    assert resolved is not None
    assert resolved == custom_hooks


def test_resolve_git_hooks_dir_with_relative_core_hooks_path(tmp_path):
    """``core.hooksPath`` relative values are resolved against the repo root, matching git's own behavior."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q", "-b", "main")
    (repo / ".husky").mkdir()
    _sh(repo, "config", "core.hooksPath", ".husky")

    resolved = cmind_cli._resolve_git_hooks_dir(repo)
    assert resolved is not None
    assert resolved.resolve() == (repo / ".husky").resolve()


def test_resolve_git_hooks_dir_empty_core_hooks_path_falls_back(tmp_path):
    """``git config --get core.hooksPath`` returning an empty string (or unset) must not divert resolution \u2014 standard ``.git/hooks`` wins."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q", "-b", "main")
    # Explicitly set then unset to exercise the empty-value path.
    _sh(repo, "config", "core.hooksPath", "")

    resolved = cmind_cli._resolve_git_hooks_dir(repo)
    assert resolved is not None
    assert resolved == repo / ".git" / "hooks"


def test_uninstall_pre_commit_hook_via_core_hooks_path(tmp_path):
    """``core.hooksPath`` directs pre-commit cleanup to the active hooks dir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q", "-b", "main")
    custom_hooks = repo / ".husky"
    custom_hooks.mkdir()
    _sh(repo, "config", "core.hooksPath", str(custom_hooks))
    (custom_hooks / "pre-commit").write_text(
        "#!/bin/sh\n"
        "# CoderMind: full RPG sync on commit\n"
        "cmind script update_graphs.py sync --staged-only\n"
    )

    assert cmind_cli._uninstall_git_pre_commit_hook(repo) is True

    assert not (custom_hooks / "pre-commit").exists()
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()


def test_uninstall_pre_commit_hook_in_worktree(tmp_path):
    """Pre-commit cleanup works through a worktree-style ``.git`` file."""
    main = tmp_path / "main"
    main.mkdir()
    _sh(main, "init", "-q", "-b", "main")
    _sh(main, "config", "user.email", "t@t.com")
    _sh(main, "config", "user.name", "t")
    (main / "f.py").write_text("x = 1\n")
    _sh(main, "add", ".")
    _sh(main, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _sh(main, "worktree", "add", "--detach", str(wt))
    pre_commit = main / ".git" / "hooks" / "pre-commit"
    pre_commit.write_text(
        "#!/bin/sh\n"
        "# CoderMind: incremental RPG sync on commit\n"
        "cmind script update_graphs.py sync --staged-only\n"
    )

    assert cmind_cli._uninstall_git_pre_commit_hook(wt) is True
    assert not pre_commit.exists()


# ===========================================================================
# B. Status text shows branch info
# ===========================================================================

@pytest.fixture
def synced_repo_with_branch(tmp_path):
    """A git repo with a single commit, RPG synced to it, on ``main``."""
    repo = tmp_path / "ws"
    code = repo / "src"
    code.mkdir(parents=True)
    (code / "f.py").write_text("def f(): pass\n")
    _sh(repo, "init", "-q", "-b", "main")
    _sh(repo, "config", "user.email", "t@t.com")
    _sh(repo, "config", "user.name", "t")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c1")
    head = _sh(repo, "rev-parse", "HEAD")

    data_dir = repo / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"
    svc = RPGService(RPG(repo_name="ws"))
    svc._rpg_dir = data_dir.resolve()
    svc.refresh_dep_graph(
        code_dir=str(code), workspace_root=str(repo), save_path=str(dep_graph_path),
    )
    svc.rpg.set_git_meta(
        head_commit=head, head_short=head[:7],
        head_branch="main", head_timestamp="2026-01-01T00:00:00+00:00",
    )
    svc.save(str(rpg_path))
    return repo, rpg_path, dep_graph_path, code, head


def _run_status_text(repo: Path, rpg_path: Path, dep_graph_path: Path) -> str:
    """Run ``update_graphs.py status`` in text mode and return stdout."""
    script = _project_root / "scripts" / "update_graphs.py"
    return subprocess.run(
        [sys.executable, str(script), "status",
         "--rpg", str(rpg_path), "--dep-graph", str(dep_graph_path)],
        cwd=repo, capture_output=True, text=True,
    ).stdout


def test_status_text_shows_branch_when_in_sync(synced_repo_with_branch):
    repo, rpg_path, dep_graph_path, _, _ = synced_repo_with_branch
    out = _run_status_text(repo, rpg_path, dep_graph_path)
    assert "on branch 'main'" in out
    assert "in sync with current HEAD" in out


def test_status_text_flags_branch_switch(synced_repo_with_branch):
    """Switch branch (without changing commit) → status text must advertise both branches so the user immediately sees the cause."""
    repo, rpg_path, dep_graph_path, _, _ = synced_repo_with_branch
    _sh(repo, "checkout", "-q", "-b", "feature/x")
    # Make sure HEAD moves (otherwise this is the noop case which is C).
    (repo / "src" / "g.py").write_text("def g(): pass\n")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c2")

    out = _run_status_text(repo, rpg_path, dep_graph_path)
    assert "branch changed: 'main' → 'feature/x'" in out


def test_status_text_omits_branch_when_detached(tmp_path):
    """Detached HEAD: branch is None, the text helper drops the suffix rather than printing ``on branch 'None'``."""
    repo = tmp_path / "ws"
    code = repo / "src"
    code.mkdir(parents=True)
    (code / "f.py").write_text("def f(): pass\n")
    _sh(repo, "init", "-q", "-b", "main")
    _sh(repo, "config", "user.email", "t@t.com")
    _sh(repo, "config", "user.name", "t")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c1")
    head = _sh(repo, "rev-parse", "HEAD")
    _sh(repo, "-c", "advice.detachedHead=false", "checkout", "-q", head)

    data_dir = repo / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"
    rpg = RPG(repo_name="ws")
    rpg.set_git_meta(
        head_commit=head, head_short=head[:7],
        head_branch=None, head_timestamp="2026-01-01T00:00:00+00:00",
    )
    rpg.save_json(str(rpg_path))

    out = _run_status_text(repo, rpg_path, dep_graph_path)
    assert "on branch 'None'" not in out
    # The "Last synced..." line should still be present
    assert "Last synced at commit" in out


# ===========================================================================
# C. noop also refreshes branch field
# ===========================================================================

def test_noop_refreshes_branch_on_rename(synced_repo_with_branch):
    """``git branch -m main develop`` keeps HEAD on the same commit but renames the branch.  After sync the ``meta.git.head_branch`` should reflect the new name even though no graph edits happened."""
    repo, rpg_path, dep_graph_path, code, head = synced_repo_with_branch
    _sh(repo, "branch", "-m", "develop")

    svc = RPGService.load(str(rpg_path))
    result = svc.sync_from_commit_diff(
        code_dir=str(code), workspace_root=str(repo),
        save_path=str(dep_graph_path), staged_only=True,
    )
    assert result["mode"] == "noop"
    assert result.get("meta_git_refreshed") is True
    assert svc.rpg.git_meta["head_branch"] == "develop"
    # head_commit is unchanged (that's the noop guarantee)
    assert svc.rpg.git_meta["head_commit"] == head


def test_noop_skips_refresh_when_nothing_changed(synced_repo_with_branch):
    """If branch + timestamp are already current, ``noop`` reports nothing was refreshed (idempotent — no spurious writes)."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo_with_branch

    svc = RPGService.load(str(rpg_path))
    # Force the timestamp on meta.git to match what read_head will report
    # so the only field that *could* drift is head_branch, which is
    # also already correct.
    from common.git_utils import read_head
    cur = read_head(repo)
    svc.rpg.set_git_meta(
        head_commit=cur["head_commit"],
        head_short=cur["head_short"],
        head_branch=cur["head_branch"],
        head_timestamp=cur["head_timestamp"],
    )
    svc.save(str(rpg_path))

    svc = RPGService.load(str(rpg_path))
    result = svc.sync_from_commit_diff(
        code_dir=str(code), workspace_root=str(repo),
        save_path=str(dep_graph_path), staged_only=True,
    )
    assert result["mode"] == "noop"
    assert result.get("meta_git_refreshed") is None


def test_noop_respects_no_git_meta_env(synced_repo_with_branch, monkeypatch):
    """``CMIND_NO_GIT_META=1`` must veto the branch refresh too."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo_with_branch
    _sh(repo, "branch", "-m", "develop")

    monkeypatch.setenv("CMIND_NO_GIT_META", "1")
    svc = RPGService.load(str(rpg_path))
    result = svc.sync_from_commit_diff(
        code_dir=str(code), workspace_root=str(repo),
        save_path=str(dep_graph_path), staged_only=True,
    )
    assert result["mode"] == "noop"
    assert result.get("meta_git_refreshed") is None
    # Branch is still the stale "main"
    assert svc.rpg.git_meta["head_branch"] == "main"


# ===========================================================================
# D. post-merge hook
# ===========================================================================

def test_install_post_merge_hook_writes_script(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")

    assert cmind_cli._install_git_post_merge_hook(repo) is True
    post_merge = repo / ".git" / "hooks" / "post-merge"
    assert post_merge.is_file()
    content = post_merge.read_text()
    assert "CoderMind: post-merge dispatcher" in content
    assert "cmind hook post-merge" in content
    assert "--staged-only" not in content
    import stat
    assert post_merge.stat().st_mode & stat.S_IXUSR


def test_install_post_merge_hook_is_idempotent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")
    cmind_cli._install_git_post_merge_hook(repo)
    cmind_cli._install_git_post_merge_hook(repo)
    cmind_cli._install_git_post_merge_hook(repo)
    post_merge = (repo / ".git" / "hooks" / "post-merge").read_text()
    assert post_merge.count("CoderMind: post-merge dispatcher") == 1


def test_install_post_merge_hook_preserves_existing_user_hook(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    user_hook = hooks_dir / "post-merge"
    user_hook.write_text("#!/bin/sh\necho 'user custom hook'\n")
    user_hook.chmod(0o755)

    cmind_cli._install_git_post_merge_hook(repo)
    content = user_hook.read_text()
    assert "echo 'user custom hook'" in content
    assert "CoderMind: post-merge dispatcher" in content
    assert "cmind hook post-merge" in content


def test_install_hooks_installs_post_hooks_and_removes_pre_commit(tmp_path):
    """End-to-end: ``_install_hooks`` writes post hooks and no pre-commit."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".cmind" / "scripts").mkdir(parents=True)
    # Stub script so installer reports OK (only the path string matters)
    (project / ".cmind" / "scripts" / "update_graphs.py").write_text("")
    _sh(project, "init", "-q")

    cmind_cli._install_hooks(project, "copilot", tracker=None)

    pre_commit = project / ".git" / "hooks" / "pre-commit"
    post_commit = project / ".git" / "hooks" / "post-commit"
    post_merge = project / ".git" / "hooks" / "post-merge"
    assert not pre_commit.exists()
    assert post_commit.is_file()
    assert post_merge.is_file()
    assert "--staged-only" not in post_commit.read_text()
    assert "--staged-only" not in post_merge.read_text()


def test_install_post_commit_hook_writes_script(tmp_path):
    """``post-commit`` delegates to the Python hook dispatcher."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")

    assert cmind_cli._install_git_post_commit_hook(repo) is True
    post_commit = repo / ".git" / "hooks" / "post-commit"
    assert post_commit.is_file()
    content = post_commit.read_text()
    assert "CoderMind: post-commit dispatcher" in content
    assert "cmind hook post-commit" in content
    assert "update_graphs.py" not in content
    assert "update-rpg" not in content
    assert "GIT_INDEX_FILE" not in content
    assert "nohup" not in content
    assert "setsid" not in content
    assert "mkdir " not in content
    assert "rmdir " not in content
    assert "-mmin +60" not in content
    assert "--staged-only" not in content
    import stat
    assert post_commit.stat().st_mode & stat.S_IXUSR


def test_install_post_commit_hook_is_idempotent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _sh(repo, "init", "-q")
    cmind_cli._install_git_post_commit_hook(repo)
    cmind_cli._install_git_post_commit_hook(repo)
    cmind_cli._install_git_post_commit_hook(repo)
    text = (repo / ".git" / "hooks" / "post-commit").read_text()
    assert text.count("CoderMind: post-commit dispatcher") == 1


def test_workspace_root_resolution_prefers_cwd_over_env(tmp_path, monkeypatch):
    """Git hooks must prefer the repository cwd over inherited workspace env."""
    # Set up two distinct workspaces
    real_ws = tmp_path / "real-ws"
    (real_ws / ".cmind").mkdir(parents=True)
    decoy_ws = tmp_path / "decoy-ws"
    (decoy_ws / ".cmind").mkdir(parents=True)

    monkeypatch.setenv("CMIND_WORKSPACE", str(decoy_ws))
    monkeypatch.chdir(real_ws)

    # Importing common.paths now should resolve to real_ws (cwd wins)
    # We re-import to bypass any module-level caching.
    import importlib
    import common.paths as paths_mod
    importlib.reload(paths_mod)
    assert paths_mod.WORKSPACE_ROOT == real_ws, (
        f"cwd-based detection should win, got {paths_mod.WORKSPACE_ROOT}"
    )
