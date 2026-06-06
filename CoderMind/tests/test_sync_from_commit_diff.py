#!/usr/bin/env python3
"""Tests for Step 3: commit-aware incremental dep_graph sync.

Covers ``RPGService.sync_from_commit_diff`` and ``sync_from_file_list``,
plus the ``--staged-only`` / ``--force-full`` / ``--file-limit`` flags
on ``update_graphs.py sync``.

The headline correctness invariant carried forward from Step 2:

    Incremental sync of an arbitrary commit sequence MUST yield a
    dep_graph that is structurally identical to a full rebuild on the
    same final on-disk state.

We use real git fixtures (not mocks) because subtle interactions
between ``git diff --cached``, ``git merge-base``, and rename detection
are exactly what we want to verify end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))

from rpg.dep_graph import DependencyGraph  # noqa: E402
from rpg.models import RPG  # noqa: E402
from rpg.service import RPGService  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a tiny git repo with an existing RPG synced to its first commit
# ---------------------------------------------------------------------------

def _sh(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    )


def _head_sha(cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def synced_repo(tmp_path):
    """A repo with 2 .py files + an RPG already synced to HEAD.

    Returns ``(workspace_root, rpg_path, dep_graph_path, code_dir, head_sha)``
    where ``rpg_path`` already exists with ``meta.git.head_commit ==
    head_sha`` so subsequent ``sync_from_commit_diff`` runs exercise the
    "incremental" branches.
    """
    repo = tmp_path / "ws"
    code = repo / "src"
    code.mkdir(parents=True)

    # Two cross-referencing files so the graph has real semantic edges.
    (code / "base.py").write_text(
        "class Base:\n"
        "    def greet(self):\n"
        "        return 'hi'\n"
        "def helper():\n"
        "    return 42\n"
    )
    (code / "consumer.py").write_text(
        "from base import Base, helper\n"
        "class Child(Base):\n"
        "    def go(self):\n"
        "        return helper()\n"
    )

    _sh(repo, "init", "-q", "-b", "main")
    _sh(repo, "config", "user.email", "test@example.com")
    _sh(repo, "config", "user.name", "Test")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "initial")
    head = _head_sha(repo)

    # Build the RPG fresh and seed meta.git so we're at "in sync".
    data_dir = repo / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"

    rpg = RPG(repo_name="ws")
    svc = RPGService(rpg)
    svc._rpg_dir = data_dir.resolve()
    svc.refresh_dep_graph(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    rpg.set_git_meta(head_commit=head, head_short=head[:7], head_branch="main")
    rpg.save_json(str(rpg_path))

    return repo, rpg_path, dep_graph_path, code, head


def _load(rpg_path: Path) -> RPGService:
    svc = RPGService.load(str(rpg_path))
    # Service loader sets _rpg_dir from the path's parent automatically.
    return svc


def _node_edge_snapshot(g: DependencyGraph) -> Tuple[Dict[str, dict], Set[Tuple[str, str, str]]]:
    nodes = {
        nid: {
            "type": a.get("type"),
            "name": a.get("name"),
            "signature": a.get("signature"),
        }
        for nid, a in g.G.nodes(data=True)
    }
    edges = {
        (u, v, a.get("type", ""))
        for u, v, a in g.G.edges(data=True)
    }
    return nodes, edges


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def test_first_sync_runs_full(tmp_path):
    """An RPG without ``meta.git`` must trigger ``mode=full``."""
    repo = tmp_path / "ws"
    code = repo / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text("def a(): pass\n")
    _sh(repo, "init", "-q", "-b", "main")
    _sh(repo, "config", "user.email", "t@t.com")
    _sh(repo, "config", "user.name", "t")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c1")

    data_dir = repo / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"

    rpg = RPG(repo_name="ws")  # NO set_git_meta
    rpg.save_json(str(rpg_path))

    svc = RPGService.load(str(rpg_path))
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "full"
    assert result["reason"] == "baseline"
    # And meta.git should have advanced to HEAD afterwards
    assert svc.rpg.git_meta is not None
    assert svc.rpg.git_meta["head_commit"] == _head_sha(repo)


def test_noop_when_head_unchanged_and_clean(synced_repo):
    repo, rpg_path, dep_graph_path, code, head = synced_repo
    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
        staged_only=True,
    )
    assert result["mode"] == "noop"
    assert result["reason"] == "head_unchanged_clean"
    # meta.git refresh is still safe (idempotent) but commit unchanged.
    assert svc.rpg.git_meta["head_commit"] == head


def test_incremental_when_head_unchanged_but_staged(synced_repo):
    """HEAD didn't move, but a file is in the staging area → incremental."""
    repo, rpg_path, dep_graph_path, code, head = synced_repo
    (code / "base.py").write_text(
        "class Base:\n"
        "    def greet(self):\n"
        "        return 'hello (new!)'\n"
        "def helper():\n"
        "    return 100\n"  # body changed
    )
    _sh(repo, "add", "src/base.py")

    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
        staged_only=True,
    )
    assert result["mode"] == "incremental"
    assert result["reason"] == "head_unchanged_dirty"
    # Hash check must have caught base.py as actually changed.
    assert result.get("modified", 0) + result.get("added", 0) >= 1


def test_linear_advance_runs_incremental(synced_repo):
    """Add a commit on top of the synced commit → mode=incremental, linear."""
    repo, rpg_path, dep_graph_path, code, _last_head = synced_repo
    (code / "extra.py").write_text(
        "from base import helper\n"
        "def use():\n"
        "    return helper() + 1\n"
    )
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "add extra")
    new_head = _head_sha(repo)

    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "incremental"
    assert result["reason"] == "linear"
    assert result["meta_git_advanced_to"] == new_head
    assert svc.rpg.git_meta["head_commit"] == new_head
    # The dep_graph must now contain the new file
    assert "extra.py" in svc.rpg.dep_graph.G


def test_diverged_history_falls_back_to_full(synced_repo):
    """Rebase / amend / reset makes the old commit unreachable → full sync."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo

    # ``git commit --amend`` rewrites the synced commit so its SHA changes.
    # meta.git still points at the OLD SHA → merge-base will not equal it.
    (code / "base.py").write_text(
        "class Base:\n"
        "    pass\n"
        "def helper():\n"
        "    return 0\n"
    )
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "--amend", "--no-edit")

    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "full"
    assert result["reason"] == "diverged"


def test_force_full_overrides_decision_tree(synced_repo):
    repo, rpg_path, dep_graph_path, code, _ = synced_repo
    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
        force_full=True,
    )
    assert result["mode"] == "full"
    assert result["reason"] == "force_full"


def test_over_limit_falls_back_to_full(synced_repo, monkeypatch):
    """When ``file_limit`` is tripped, fall back to full rebuild."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo
    # Create 5 new files so a limit of 2 trips.
    for i in range(5):
        (code / f"f{i}.py").write_text(f"def f{i}(): return {i}\n")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "bulk")

    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
        file_limit=2,
    )
    assert result["mode"] == "full"
    assert result["reason"].startswith("over_limit_")


def test_no_git_meta_env_var_skips_meta_write(synced_repo, monkeypatch):
    """``CMIND_NO_GIT_META=1`` must not advance ``meta.git``."""
    repo, rpg_path, dep_graph_path, code, original_head = synced_repo

    # Make a real commit so HEAD changes
    (code / "y.py").write_text("def y(): pass\n")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "y")
    new_head = _head_sha(repo)

    monkeypatch.setenv("CMIND_NO_GIT_META", "1")
    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "incremental"
    assert "meta_git_advanced_to" not in result
    # meta.git unchanged (still the original head)
    assert svc.rpg.git_meta["head_commit"] == original_head
    assert new_head != original_head  # sanity


# ---------------------------------------------------------------------------
# THE GOLDEN INVARIANT: incremental == full rebuild
# ---------------------------------------------------------------------------

def test_incremental_matches_full_rebuild_across_commit_chain(synced_repo):
    """3 commits' worth of incremental updates must converge on the same graph a single from-scratch ``build()+parse()`` would have produced."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo

    # ── Commit 2: modify base.py
    (code / "base.py").write_text(
        "class Base:\n"
        "    def greet(self):\n"
        "        return 'hello v2'\n"
        "def helper():\n"
        "    return 99\n"
    )
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c2")

    svc = _load(rpg_path)
    svc.sync_from_commit_diff(str(code), str(repo), str(dep_graph_path))
    svc.save(str(rpg_path))

    # ── Commit 3: add a new file
    (code / "extra.py").write_text(
        "from consumer import Child\n"
        "def make():\n"
        "    return Child()\n"
    )
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c3")
    svc = _load(rpg_path)
    svc.sync_from_commit_diff(str(code), str(repo), str(dep_graph_path))
    svc.save(str(rpg_path))

    # ── Commit 4: delete consumer.py
    (code / "consumer.py").unlink()
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c4")
    svc = _load(rpg_path)
    svc.sync_from_commit_diff(str(code), str(repo), str(dep_graph_path))
    svc.save(str(rpg_path))

    inc_nodes, inc_edges = _node_edge_snapshot(svc.rpg.dep_graph)

    # Ground truth: full rebuild on the final on-disk state.
    ref = DependencyGraph(str(code))
    ref.build()
    ref.parse()
    full_nodes, full_edges = _node_edge_snapshot(ref)

    assert inc_nodes == full_nodes, (
        f"Node sets diverged.\nincremental_only={set(inc_nodes) - set(full_nodes)}\n"
        f"full_only={set(full_nodes) - set(inc_nodes)}"
    )
    assert inc_edges == full_edges, (
        f"Edge sets diverged.\nincremental_only={inc_edges - full_edges}\n"
        f"full_only={full_edges - inc_edges}"
    )


def test_rename_via_git_mv_preserves_edge_set(synced_repo):
    """``git mv`` must be detected as a rename by ``-M`` and end up equivalent to a full rebuild — no orphan edges into the old path."""
    repo, rpg_path, dep_graph_path, code, _ = synced_repo

    _sh(repo, "mv", "src/base.py", "src/core.py")
    # consumer.py still imports from `base`; update it too so the rebuild
    # represents real refactor semantics.
    (code / "consumer.py").write_text(
        "from core import Base, helper\n"
        "class Child(Base):\n"
        "    def go(self):\n"
        "        return helper()\n"
    )
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "rename")

    svc = _load(rpg_path)
    result = svc.sync_from_commit_diff(str(code), str(repo), str(dep_graph_path))
    svc.save(str(rpg_path))
    assert result["mode"] == "incremental"
    assert result.get("renamed", 0) == 1
    assert "base.py" not in svc.rpg.dep_graph.G
    assert "core.py" in svc.rpg.dep_graph.G

    ref = DependencyGraph(str(code))
    ref.build()
    ref.parse()
    inc_nodes, inc_edges = _node_edge_snapshot(svc.rpg.dep_graph)
    full_nodes, full_edges = _node_edge_snapshot(ref)
    assert inc_nodes == full_nodes
    assert inc_edges == full_edges


# ---------------------------------------------------------------------------
# sync_from_file_list (codegen helper)
# ---------------------------------------------------------------------------

def test_sync_from_file_list_explicit(synced_repo):
    """Codegen path: hand the service an explicit file list, get incremental update without touching git or meta.git."""
    repo, rpg_path, dep_graph_path, code, head = synced_repo
    (code / "base.py").write_text(
        "class Base:\n"
        "    def greet(self):\n"
        "        return 'new'\n"
        "def helper():\n"
        "    return 7\n"
    )

    svc = _load(rpg_path)
    result = svc.sync_from_file_list(
        file_paths=["base.py"],
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "incremental"
    assert result["reason"] == "explicit_file_list"
    # meta.git is the caller's responsibility for this entry point.
    assert svc.rpg.git_meta["head_commit"] == head


def test_sync_from_file_list_bootstraps_full_when_no_dep_graph(tmp_path):
    """If the RPG has no dep_graph yet (very first codegen batch), ``sync_from_file_list`` must fall back to a full ``refresh_dep_graph``."""
    repo = tmp_path / "ws"
    code = repo / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text("def a(): pass\n")
    _sh(repo, "init", "-q", "-b", "main")
    _sh(repo, "config", "user.email", "t@t.com")
    _sh(repo, "config", "user.name", "t")
    _sh(repo, "add", ".")
    _sh(repo, "commit", "-q", "-m", "c1")

    data_dir = repo / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"
    rpg = RPG(repo_name="ws")
    rpg.save_json(str(rpg_path))

    svc = RPGService.load(str(rpg_path))
    assert svc.rpg.dep_graph is None  # precondition
    result = svc.sync_from_file_list(
        file_paths=["a.py"],
        code_dir=str(code),
        workspace_root=str(repo),
        save_path=str(dep_graph_path),
    )
    assert result["mode"] == "full"
    assert result["reason"] == "no_existing_dep_graph"
    assert svc.rpg.dep_graph is not None


# ---------------------------------------------------------------------------
# CLI integration: update_graphs.py sync
# ---------------------------------------------------------------------------

def _run_cli_sync(*args, cwd: Path) -> dict:
    script = _project_root / "scripts" / "update_graphs.py"
    cmd = [sys.executable, str(script), "sync", "--json", *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"sync CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return json.loads(result.stdout)


def test_cli_sync_staged_only_flag(synced_repo):
    repo, rpg_path, dep_graph_path, code, _ = synced_repo

    # Working-tree change that is NOT staged → should not trigger sync
    (code / "base.py").write_text(
        "class Base:\n    def greet(self):\n        return 'unstaged'\n"
        "def helper():\n    return 5\n"
    )
    data = _run_cli_sync(
        "--rpg", str(rpg_path),
        "--dep-graph", str(dep_graph_path),
        "--code-dir", str(code),
        "--staged-only",
        cwd=repo,
    )
    assert data["mode"] == "noop", (
        f"staged_only should ignore unstaged WT changes, got {data}"
    )

    # Without --staged-only, the WT change DOES count
    data2 = _run_cli_sync(
        "--rpg", str(rpg_path),
        "--dep-graph", str(dep_graph_path),
        "--code-dir", str(code),
        cwd=repo,
    )
    assert data2["mode"] == "incremental"


def test_cli_sync_force_full(synced_repo):
    repo, rpg_path, dep_graph_path, code, _ = synced_repo
    data = _run_cli_sync(
        "--rpg", str(rpg_path),
        "--dep-graph", str(dep_graph_path),
        "--code-dir", str(code),
        "--force-full",
        cwd=repo,
    )
    assert data["mode"] == "full"
    assert data["reason"] == "force_full"


def test_cli_sync_missing_rpg_returns_actionable_error(tmp_path):
    """``sync`` must early-return with a /cmind.encode hint when rpg.json is absent.

    Regression guard: missing RPG files should produce a structured
    error visible in the hook log so the user can tell why the
    background updater did nothing.
    """
    script = _project_root / "scripts" / "update_graphs.py"
    missing = tmp_path / "does_not_exist.json"
    for sub in ("sync", "update-rpg"):
        result = subprocess.run(
            [sys.executable, str(script), sub, "--rpg", str(missing), "--json"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{sub} should exit cleanly on missing rpg, got rc={result.returncode}\n"
            f"stderr={result.stderr}"
        )
        payload = json.loads(result.stdout)
        assert payload["mode"] == sub
        assert "error" in payload, payload
        assert "/cmind.encode" in payload["error"], payload["error"]
        assert str(missing) in payload["error"]
