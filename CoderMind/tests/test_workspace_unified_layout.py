#!/usr/bin/env python3
"""Tests for the unified workspace==repo layout contract.

Locks in the following invariants:

* ``REPO_DIR == WORKSPACE_ROOT`` (no ``repo/`` subdirectory).
* ``REPO_DIR_NAME`` constant no longer exists.
* ``RPG.parse_dep_graph`` never auto-probes a ``repo/`` subdir.
* All ``_dep_graph_code_dir`` writers normalise ``"."`` to ``""``
  (otherwise downstream prefix logic produces ``"./"`` and silently
  corrupts paths).
* ``GraphQueryEngine`` handles an empty ``_code_dir_prefix`` cleanly.

These tests are decoupled from the heavier
``test_encoder_workspace_layout.py`` so they can be run on their own
during the refactor without dragging the full encoder stack along.
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
# Fixtures
# ---------------------------------------------------------------------------

def _reload_paths_against(workspace: Path):
    """Import / reload ``common.paths`` with ``workspace`` as cwd.

    Same approach as ``test_encoder_workspace_layout.py``: WORKSPACE_ROOT
    is computed at module import time from cwd, so we ``chdir`` and
    reload.
    """
    os.chdir(workspace)
    os.environ.pop("CMIND_WORKSPACE", None)
    import common.paths as paths_mod
    importlib.reload(paths_mod)
    return paths_mod


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".cmind").mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.delenv("CMIND_WORKSPACE", raising=False)
    return ws


@pytest.fixture
def workspace_with_unrelated_repo_subdir(tmp_path, monkeypatch):
    """A workspace that happens to contain a ``repo/`` sibling directory.

    The new layout MUST NOT treat this as the code root automatically —
    that would re-introduce the old auto-probing footgun.
    """
    ws = tmp_path / "ws_with_repo"
    ws.mkdir()
    (ws / ".cmind").mkdir()
    (ws / "repo").mkdir()
    (ws / "repo" / "stale.py").write_text("# legacy / unrelated\n")
    (ws / "auth.py").write_text("def login(): pass\n")
    monkeypatch.chdir(ws)
    monkeypatch.delenv("CMIND_WORKSPACE", raising=False)
    return ws


# ---------------------------------------------------------------------------
# Core path invariants
# ---------------------------------------------------------------------------

def test_repo_dir_equals_workspace_root(workspace):
    paths = _reload_paths_against(workspace)
    assert paths.REPO_DIR == paths.WORKSPACE_ROOT
    assert paths.REPO_DIR == workspace


def test_repo_dir_name_removed(workspace):
    """The legacy ``REPO_DIR_NAME = "repo"`` constant is gone."""
    paths = _reload_paths_against(workspace)
    assert not hasattr(paths, "REPO_DIR_NAME"), (
        "REPO_DIR_NAME is the canary for the old layout; "
        "if you reintroduce it the unified-layout contract regresses."
    )


def test_get_repo_path_removed(workspace):
    """The ``get_repo_path()`` getter is gone — callers use REPO_DIR directly."""
    paths = _reload_paths_against(workspace)
    assert not hasattr(paths, "get_repo_path"), (
        "get_repo_path() was a pure alias for REPO_DIR; both should be gone."
    )


# ---------------------------------------------------------------------------
# parse_dep_graph no longer auto-probes <repo_dir>/repo
# ---------------------------------------------------------------------------

def test_parse_dep_graph_ignores_repo_subdir(workspace_with_unrelated_repo_subdir):
    """When caller passes ``repo_dir`` and a ``repo/`` sibling exists, ``parse_dep_graph`` MUST NOT silently descend into it."""
    _reload_paths_against(workspace_with_unrelated_repo_subdir)
    import rpg.models
    importlib.reload(rpg.models)
    from rpg.models import RPG  # re-import after reload

    rpg = RPG(repo_name="ws_with_repo", repo_info="test")
    dg = rpg.parse_dep_graph(str(workspace_with_unrelated_repo_subdir))

    # ``_dep_graph_code_dir`` should be empty ("." normalised to ""),
    # not "repo", because we should NOT have auto-probed into the subdir.
    assert rpg._dep_graph_code_dir == "", (
        f"Expected empty code_dir (workspace==repo), got "
        f"{rpg._dep_graph_code_dir!r}.  The auto-probe heuristic is back."
    )
    # And dep_graph should see ``auth.py`` (workspace-root file), not
    # the legacy ``stale.py`` under ``repo/``.
    node_ids = set(dg.G.nodes())
    assert "auth.py" in node_ids
    # ``repo/stale.py`` is fine to appear (it's a child file of the
    # workspace), but it must NOT be ``stale.py`` (which would indicate
    # the auto-probe stripped the prefix).
    assert "stale.py" not in node_ids


# ---------------------------------------------------------------------------
# §4.4: "." → "" normalisation at all 4 writer sites
# ---------------------------------------------------------------------------

def test_dep_graph_code_dir_empty_when_workspace_eq_repo(tmp_path):
    """All writers of ``_dep_graph_code_dir`` must normalise relpath==``.`` to the empty string.  Without this, downstream ``if prefix:`` checks silently treat ``"."`` as truthy and produce ``"./"`` prefixes."""
    from rpg.models import RPG
    from rpg.service import RPGService

    repo = tmp_path / "myproj"
    repo.mkdir()
    (repo / "a.py").write_text("def f(): pass\n")

    rpg = RPG(repo_name="myproj", repo_info="test")

    # Path 1: RPG.parse_dep_graph
    rpg.parse_dep_graph(str(repo))
    assert rpg._dep_graph_code_dir == "", (
        f"RPG.parse_dep_graph wrote {rpg._dep_graph_code_dir!r}; "
        f"expected empty string."
    )

    # Path 2: RPGService.refresh_dep_graph
    rpg2 = RPG(repo_name="myproj", repo_info="test")
    svc = RPGService(rpg2)
    svc._rpg_dir = repo
    svc.refresh_dep_graph(
        code_dir=str(repo),
        workspace_root=str(repo),
        save_path=str(repo / "dep_graph.json"),
    )
    assert rpg2._dep_graph_code_dir == "", (
        f"refresh_dep_graph wrote {rpg2._dep_graph_code_dir!r}"
    )

    # Path 4: update_graphs.update_dep_only — verify via the on-disk JSON
    import json
    from update_graphs import update_dep_only

    dep_graph_path = repo / "dep_graph_updated.json"
    update_dep_only(str(repo), str(repo), dep_graph_path)
    data = json.loads(dep_graph_path.read_text())
    assert data["code_dir"] == "", (
        f"update_dep_only wrote code_dir={data['code_dir']!r}; "
        f"expected empty string."
    )


def test_graph_query_handles_empty_code_dir(tmp_path):
    """GraphQueryEngine must not corrupt paths when ``_dep_graph_code_dir`` is empty."""
    from rpg.graph_query import GraphQueryEngine

    rpg_data = {
        "_dep_graph_code_dir": "",
        "root": {"id": "r", "name": "root", "children": []},
        "_dep_to_rpg_map": {},
        "_feature_to_dep_map": {},
    }
    dep_data = {"nodes": {}, "edges": []}
    engine = GraphQueryEngine(rpg_data, dep_data)

    # The prefix used internally must be empty (NOT ``"./"``).
    assert engine._code_dir_prefix == ""
    # And ``_normalize_path`` must be an identity in this case.
    assert engine._normalize_path("models/user.py") == "models/user.py"
    assert engine._normalize_path("src/app.py") == "src/app.py"


def test_graph_query_legacy_repo_prefix_still_works(tmp_path):
    """Legacy data with ``_dep_graph_code_dir == "repo"`` should still have its prefix stripped — the unified-layout switch is forward- compatible with old persisted RPGs."""
    from rpg.graph_query import GraphQueryEngine

    rpg_data = {
        "_dep_graph_code_dir": "repo",
        "root": {"id": "r", "name": "root", "children": []},
        "_dep_to_rpg_map": {},
        "_feature_to_dep_map": {},
    }
    engine = GraphQueryEngine(rpg_data, {"nodes": {}, "edges": []})
    assert engine._code_dir_prefix == "repo/"
    assert engine._normalize_path("repo/models/user.py") == "models/user.py"
    # Non-matching paths pass through unchanged.
    assert engine._normalize_path("src/app.py") == "src/app.py"


# ---------------------------------------------------------------------------
# rpg_edit apply / review: no more ["repo", "src", "."] auto-probe
# ---------------------------------------------------------------------------

def test_rpg_edit_apply_no_subdir_fallback_in_source():
    """``rpg_edit/apply.py`` must not contain the old fallback list.

    Source-level check (rather than running the script) so the test
    stays fast and doesn't need the full LLM/git scaffolding.
    """
    src = (_project_root / "scripts" / "rpg_edit" / "apply.py").read_text()
    assert '["repo", "src", "."]' not in src, (
        "The auto-probe fallback ['repo', 'src', '.'] is back in "
        "rpg_edit/apply.py — that was supposed to be deleted."
    )
    # And the new default points at REPO_DIR explicitly.
    assert "args.repo or REPO_DIR" in src, (
        "rpg_edit/apply.py should default to REPO_DIR when --repo is omitted."
    )


def test_rpg_edit_review_no_subdir_fallback_in_source():
    src = (_project_root / "scripts" / "rpg_edit" / "review.py").read_text()
    assert '["repo", "src", "."]' not in src
    assert "args.repo or REPO_DIR" in src


# ---------------------------------------------------------------------------
# run_batch.py — guard against silent breakage of the monkeypatch /
# external-import surface
# ---------------------------------------------------------------------------

def test_run_batch_preserves_external_surface(monkeypatch):
    """``tests/test_step4_integration.py`` and the rpg_edit pipeline patch ``run_batch.<name>`` directly.  If those module-level names disappear from ``run_batch`` (e.g. someone decides to import them indirectly via a helper module rather than at the top of run_batch.py), the monkeypatch target is silently lost — pytest still passes because ``monkeypatch.setattr`` happily creates a new attribute — but production calls fall through to the un-patched original and the failure surfaces only in the user's real workspace.

    This test pins the contract down so any future refactor that removes
    or hides one of these names fails CI immediately.

    Required surface:
      * Module-level path constants patched by tests:
        ``REPO_RPG_FILE`` / ``DEP_GRAPH_FILE`` / ``WORKSPACE_ROOT``
      * Helper patched by tests:
        ``get_scripts_dir``
      * Internal helpers called directly by tests:
        ``_refresh_dep_graph_safe`` / ``_task_files_for_dep_graph``
      * Public symbol consumed by rpg_edit/* and subtree_review:
        ``dispatch_sub_agent``
    """
    monkeypatch.chdir(_project_root)  # avoid stale cwd from previous fixtures
    monkeypatch.delenv("CMIND_WORKSPACE", raising=False)
    import run_batch
    required = (
        "REPO_RPG_FILE",
        "DEP_GRAPH_FILE",
        "WORKSPACE_ROOT",
        "get_scripts_dir",
        "_refresh_dep_graph_safe",
        "_task_files_for_dep_graph",
        "dispatch_sub_agent",
    )
    missing = [name for name in required if not hasattr(run_batch, name)]
    assert not missing, (
        f"run_batch.* lost the following names: {missing!r}. "
        f"These are monkeypatch / external-import targets — removing them "
        f"silently breaks test_step4_integration and rpg_edit/* callers."
    )

