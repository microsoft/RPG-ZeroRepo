#!/usr/bin/env python3
"""Tests for the encoder default code-directory contract.

The encoder entry points (``run_encode.py`` /
``run_update_rpg.py``) and the ``update_graphs.py sync`` hook all
default to scanning :data:`common.paths.WORKSPACE_ROOT` — the
directory the user ran ``cmind init --here`` in (their existing
source repository).  There is no ``repo/`` sub-convention to honour
on the encoder side; the decoder pipeline writes code to ``REPO_DIR``
through entirely separate entry points.

These tests pin the contract down so a future refactor can't
re-introduce a ``repo/`` fallback (which used to silently break
encoder workspaces that didn't happen to have such a subdir).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures — reload ``common.paths`` against an arbitrary workspace
# ---------------------------------------------------------------------------

def _reload_paths_against(workspace: Path):
    """Import / reload ``common.paths`` with ``workspace`` as cwd.

    ``WORKSPACE_ROOT`` is computed at module import time from the
    current working directory, so to test against different workspace
    layouts we must reload after chdir'ing.
    """
    os.chdir(workspace)
    os.environ.pop("CMIND_WORKSPACE", None)
    import common.paths as paths_mod
    importlib.reload(paths_mod)
    return paths_mod


@pytest.fixture
def encoder_workspace(tmp_path, monkeypatch):
    """A workspace with ``.cmind/`` but NO ``repo/`` subdirectory — the canonical encoder layout (``cmind init --here`` inside an existing code repository)."""
    ws = tmp_path / "enc_ws"
    ws.mkdir()
    (ws / ".cmind").mkdir()
    (ws / "auth.py").write_text("def login(): pass\n")
    (ws / "db.py").write_text("def connect(): pass\n")
    monkeypatch.chdir(ws)
    monkeypatch.delenv("CMIND_WORKSPACE", raising=False)
    return ws


@pytest.fixture
def workspace_with_repo_subdir(tmp_path, monkeypatch):
    """A workspace that happens to contain a ``repo/`` subdirectory.

    The encoder default should STILL pick the workspace root (not
    the ``repo/`` subdir) — that's the deliberate simplification
    we landed on.  Decoder code goes through different entry points
    that target ``REPO_DIR`` explicitly.
    """
    ws = tmp_path / "ws_with_repo"
    ws.mkdir()
    (ws / ".cmind").mkdir()
    (ws / "repo").mkdir()
    (ws / "repo" / "main.py").write_text("def main(): pass\n")
    monkeypatch.chdir(ws)
    monkeypatch.delenv("CMIND_WORKSPACE", raising=False)
    return ws


# ---------------------------------------------------------------------------
# Encoder entry point defaults
# ---------------------------------------------------------------------------

def test_run_encode_default_assignment_uses_workspace_root(encoder_workspace):
    """``run_encode`` defaults ``repo_dir`` to ``WORKSPACE_ROOT``.

    Verified by source inspection (single deterministic assignment
    line) rather than by stubbing the heavy LLM path, so this stays
    fast and resistant to internal refactors of the encoder body.
    """
    paths = _reload_paths_against(encoder_workspace)
    import rpg_encoder.run_encode as enc
    importlib.reload(enc)
    src = Path(enc.__file__).read_text()
    assert "str(WORKSPACE_ROOT)" in src, (
        "run_encode must default ``repo_dir`` to WORKSPACE_ROOT"
    )
    # The encoder workspace's WORKSPACE_ROOT is the workspace itself
    assert paths.WORKSPACE_ROOT == encoder_workspace


def test_run_update_rpg_default_assignment_uses_workspace_root(encoder_workspace):
    _reload_paths_against(encoder_workspace)
    import rpg_encoder.run_update_rpg as upd
    importlib.reload(upd)
    src = Path(upd.__file__).read_text()
    assert "str(WORKSPACE_ROOT)" in src


def test_run_update_rpg_error_path_in_encoder_layout(encoder_workspace):
    """End-to-end: ``run_update_rpg(cur_repo_dir=None)`` must NOT report "Current repo directory not found" in an encoder workspace — that would mean the default still pointed at a non-existent ``<workspace>/repo``."""
    _reload_paths_against(encoder_workspace)
    import rpg_encoder.run_update_rpg as upd
    importlib.reload(upd)

    rpg_path = encoder_workspace / ".cmind" / "data" / "rpg.json"
    rpg_path.parent.mkdir(parents=True, exist_ok=True)
    rpg_path.write_text('{"repo_name": "test", "root": {}}')

    result = upd.run_update_rpg(
        rpg_file=str(rpg_path),
        last_repo_dir="/tmp/definitely-does-not-exist-xyz-encoder",
        cur_repo_dir=None,
    )
    assert result["status"] == "error"
    # Error should be about ``last_repo_dir``, NOT ``cur_repo_dir``
    assert "Previous repo directory not found" in result["error"]
    assert "Current repo directory not found" not in result["error"]


def test_encoder_default_ignores_repo_subdir_when_present(
    workspace_with_repo_subdir,
):
    """A ``repo/`` subdir is NOT consulted — even when it exists.

    Guards against re-introducing the "fall back to
    ``<workspace>/repo`` if it exists" heuristic.
    """
    _reload_paths_against(workspace_with_repo_subdir)
    import rpg_encoder.run_update_rpg as upd
    importlib.reload(upd)
    import rpg_encoder.run_encode as enc
    importlib.reload(enc)

    for module in (upd, enc):
        src = Path(module.__file__).read_text()
        # Default lands on WORKSPACE_ROOT (the assignment line)
        assert "str(WORKSPACE_ROOT)" in src, (
            f"{module.__name__} must default to WORKSPACE_ROOT"
        )
        # And critically, NO ``"repo"`` suffix appended anywhere
        # near the default assignment.  We strip ``dep_graph`` to
        # avoid the obvious "dep_graph" substring false positive.
        cleaned = src.replace("dep_graph", "")
        assert 'WORKSPACE_ROOT / "repo"' not in cleaned
        assert "REPO_DIR" not in cleaned


def test_run_update_rpg_explicit_override_wins(
    workspace_with_repo_subdir, tmp_path,
):
    """An explicit ``cur_repo_dir`` always wins over the default."""
    _reload_paths_against(workspace_with_repo_subdir)
    import rpg_encoder.run_update_rpg as upd
    importlib.reload(upd)

    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    (explicit / "x.py").write_text("x = 1\n")
    rpg_path = workspace_with_repo_subdir / ".cmind" / "data" / "rpg.json"
    rpg_path.parent.mkdir(parents=True, exist_ok=True)
    rpg_path.write_text('{"repo_name": "test", "root": {}}')

    result = upd.run_update_rpg(
        rpg_file=str(rpg_path),
        last_repo_dir="/tmp/definitely-does-not-exist-explicit",
        cur_repo_dir=str(explicit),
    )
    # cur_repo_dir is accepted (exists), so the only error is about
    # the missing last_repo_dir.
    assert result["status"] == "error"
    assert "Previous repo directory not found" in result["error"]


# ---------------------------------------------------------------------------
# update_graphs.py sync auto-detect contract
# ---------------------------------------------------------------------------

def test_update_graphs_auto_detect_returns_workspace_root(encoder_workspace):
    """``_auto_detect_code_dir`` returns the workspace root by default — no longer probes for ``["repo", "src", "."]`` (which could surprise encoder users who happen to have an unrelated ``src/`` directory)."""
    _reload_paths_against(encoder_workspace)
    import update_graphs
    importlib.reload(update_graphs)
    result = update_graphs._auto_detect_code_dir(str(encoder_workspace))
    assert result == str(encoder_workspace)


def test_update_graphs_auto_detect_explicit_arg_wins(
    encoder_workspace, tmp_path,
):
    """Explicit ``--code-dir`` always overrides the default."""
    _reload_paths_against(encoder_workspace)
    import update_graphs
    importlib.reload(update_graphs)
    other = tmp_path / "other"
    other.mkdir()
    result = update_graphs._auto_detect_code_dir(
        str(encoder_workspace), str(other),
    )
    assert result == str(other)


def test_update_graphs_auto_detect_ignores_present_repo_subdir(
    workspace_with_repo_subdir,
):
    """A bare ``repo/`` subdir is NOT auto-selected anymore.

    Used to be: ``_auto_detect_code_dir`` walked ``["repo", "src", "."]``
    and picked the first that existed.  That heuristic surprised
    encoder users with unrelated subdirs.  Now the default is always
    the workspace root.
    """
    _reload_paths_against(workspace_with_repo_subdir)
    import update_graphs
    importlib.reload(update_graphs)
    result = update_graphs._auto_detect_code_dir(
        str(workspace_with_repo_subdir),
    )
    assert result == str(workspace_with_repo_subdir)
    # Critically NOT the repo/ subdir
    assert result != str(workspace_with_repo_subdir / "repo")
