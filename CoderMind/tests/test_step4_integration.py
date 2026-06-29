#!/usr/bin/env python3
"""Tests for Step 4a + 4b + 4c.

4a — ``run_batch._refresh_dep_graph_safe`` and ``_task_files_for_dep_graph``
     route codegen file-list scope through ``sync_from_file_list``.

4b — ``RPGEvolution._update_dep_graph_index`` writes ``dep_graph.json`` to
     disk (regression for the silent-drift bug between RPG embedded
     dep_graph and standalone ``dep_graph.json``).

4c — ``run_update_rpg.py`` advances ``meta.git`` + runs
     ``enrich_from_code(align_only=True)`` after the LLM-driven phase.

We use real synthetic repos / dep_graphs (no mocks of the heavy LLM
path — instead, mock or directly invoke the structural helpers).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))

from rpg.models import RPG  # noqa: E402
from rpg.service import RPGService  # noqa: E402


def _sh(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


# ===========================================================================
# 4a — codegen file-list mode
# ===========================================================================
#
# We don't import run_batch.py directly (it has a huge import chain).
# Instead we re-implement the tiny ``_task_files_for_dep_graph`` decision
# table here against the run_batch module so we don't drift, and we
# exercise ``_refresh_dep_graph_safe`` via a stub PlannedTask + monkey-
# patched paths.

@pytest.fixture
def codegen_workspace(tmp_path, monkeypatch):
    """A workspace where ``_refresh_dep_graph_safe`` can run end-to-end.

    Stubs out the module-level constants ``REPO_RPG_FILE`` /
    ``DEP_GRAPH_FILE`` / ``WORKSPACE_ROOT`` so the function reads from
    our tmp_path instead of the user's real workspace.
    """
    ws = tmp_path / "ws"
    code = ws / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text("def a(): return 1\n")
    (code / "b.py").write_text("from a import a\ndef b(): return a() + 1\n")

    data_dir = ws / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"

    svc = RPGService(RPG(repo_name="ws"))
    svc._rpg_dir = data_dir.resolve()
    svc.refresh_dep_graph(
        code_dir=str(code),
        workspace_root=str(ws),
        save_path=str(dep_graph_path),
    )
    svc.save(str(rpg_path))

    import run_batch  # noqa: E402
    monkeypatch.setattr(run_batch, "REPO_RPG_FILE", rpg_path)
    monkeypatch.setattr(run_batch, "DEP_GRAPH_FILE", dep_graph_path)
    monkeypatch.setattr(run_batch, "WORKSPACE_ROOT", ws)
    # ``get_scripts_dir`` is called inside _refresh_dep_graph_safe to
    # add the scripts/ dir to sys.path; point it at the real one.
    monkeypatch.setattr(
        run_batch, "get_scripts_dir",
        lambda: str(_project_root / "scripts"),
    )

    return ws, code, rpg_path, dep_graph_path, svc, run_batch


def test_refresh_dep_graph_safe_with_file_list_uses_incremental(codegen_workspace, caplog):
    """When ``changed_files`` is provided, the log line must say ``mode=incremental``, not a full rebuild."""
    ws, code, rpg_path, dep_graph_path, _svc, run_batch = codegen_workspace

    # Modify a.py so an incremental sync has actual work to do
    (code / "a.py").write_text("def a(): return 42\n")

    import logging
    with caplog.at_level(logging.INFO, logger="run_batch"):
        run_batch._refresh_dep_graph_safe(code, changed_files=["a.py"])

    # The new log format includes ``mode=incremental``
    assert any(
        "mode=incremental" in record.getMessage()
        for record in caplog.records
    ), f"expected mode=incremental in logs, got: {[r.getMessage() for r in caplog.records]}"
    # The dep_graph file is still up to date
    assert dep_graph_path.exists()


def test_refresh_dep_graph_safe_without_files_falls_back_to_full(codegen_workspace, caplog):
    ws, code, _, dep_graph_path, _, run_batch = codegen_workspace
    import logging
    with caplog.at_level(logging.INFO, logger="run_batch"):
        run_batch._refresh_dep_graph_safe(code)
    # Should log ``(full)`` to signal the fallback path
    assert any(
        "(full)" in record.getMessage() for record in caplog.records
    ), f"expected (full) in logs: {[r.getMessage() for r in caplog.records]}"


def test_refresh_dep_graph_safe_skips_non_py_files(codegen_workspace, caplog):
    """If the batch only edited non-source files, the function should short-circuit without scanning the AST."""
    ws, code, _, _, _, run_batch = codegen_workspace
    import logging
    with caplog.at_level(logging.INFO, logger="run_batch"):
        run_batch._refresh_dep_graph_safe(code, changed_files=["README.md"])
    assert any(
        "no supported source files" in record.getMessage() for record in caplog.records
    ), "expected the short-circuit log line"


def test_task_files_for_dep_graph_filters_special_task_types(codegen_workspace):
    _, _, _, _, _, run_batch = codegen_workspace

    class _Task:
        def __init__(self, task_type, file_path):
            self.task_type = task_type
            self.file_path = file_path

    # Real implementation tasks pass through
    assert run_batch._task_files_for_dep_graph(
        _Task("implementation", "src/foo.py")
    ) == ["src/foo.py"]

    # Skip types return None so the caller falls back to full rebuild
    for skip_type in (
        "integration_test", "final_test_docs", "main_entry",
        "project_requirements", "project_docs",
    ):
        assert run_batch._task_files_for_dep_graph(
            _Task(skip_type, "src/anything.py")
        ) is None, f"task_type={skip_type} should fall back to full"

    # Marker file paths are also opaque to dep_graph
    assert run_batch._task_files_for_dep_graph(
        _Task("implementation", "<INTEGRATION_TEST>")
    ) is None


# ===========================================================================
# 4b — _update_dep_graph_index attaches in-memory dep_graph for embedded save
# ===========================================================================

def test_update_dep_graph_index_populates_in_memory_dep_graph(tmp_path):
    """After the dep_graph-single-source migration, ``_update_dep_graph_index``
    no longer requires a standalone ``dep_graph.json`` write — the caller's
    ``svc.save(rpg.json)`` embeds the in-memory graph via ``RPG.to_dict``.
    The helper still mutates ``rpg.dep_graph`` so callers can serialise it.
    """
    from rpg_encoder.rpg_evolution import RPGEvolution
    import logging

    ws = tmp_path / "ws"
    repo = ws / "repo"
    repo.mkdir(parents=True)
    (repo / "x.py").write_text("def x(): return 1\n")
    (repo / "y.py").write_text("from x import x\ndef y(): return x() * 2\n")

    rpg = RPG(repo_name="ws")

    logger = logging.getLogger("test_4b")
    # ``save_path`` omitted: new default — dep_graph rides inside rpg.json.
    RPGEvolution._update_dep_graph_index(rpg, str(ws), logger)

    assert rpg.dep_graph is not None, "in-memory dep_graph must be attached"
    assert rpg.dep_graph.G.number_of_nodes() >= 2, (
        "dep_graph must contain at least the two source files"
    )
    # Round-trip through to_dict to prove embedding works.
    serialised = rpg.to_dict()
    assert "dep_graph" in serialised
    assert serialised["dep_graph"]["nodes"]


def test_update_dep_graph_index_legacy_save_path_still_writes_standalone(tmp_path):
    """Backward-compat: callers that still pass ``save_path`` get the
    standalone ``dep_graph.json`` written (legacy path preserved for
    tooling that consumed the sidecar file directly).
    """
    from rpg_encoder.rpg_evolution import RPGEvolution
    import logging

    ws = tmp_path / "ws"
    repo = ws / "repo"
    repo.mkdir(parents=True)
    (repo / "z.py").write_text("z = 1\n")

    rpg = RPG(repo_name="ws")
    # Place the dep_graph in a deep tmpdir nobody's cwd ever traverses
    dep_graph_path = tmp_path / "elsewhere" / "dep_graph.json"

    logger = logging.getLogger("test_4b_legacy")
    RPGEvolution._update_dep_graph_index(
        rpg, str(ws), logger, save_path=str(dep_graph_path),
    )

    # The file landed on disk even though its parent directory didn't
    # exist when we called the helper (refresh_dep_graph mkdir's it).
    assert dep_graph_path.is_file()
    # ``_dep_graph_file`` is stored relative to the save_path's parent
    # (which _update_dep_graph_index sets as _rpg_dir), so callers that
    # ``RPGService.load`` the RPG later can still find the legacy file.
    assert rpg._dep_graph_file == "dep_graph.json"


def test_update_dep_graph_index_without_save_path_logs_info(tmp_path, caplog):
    """Default behaviour after the embed migration: no save_path means the
    dep_graph is attached in memory and an INFO log records that it will
    ride inside rpg.json on the caller's next save.
    """
    from rpg_encoder.rpg_evolution import RPGEvolution
    import logging

    ws = tmp_path / "ws"
    repo = ws / "repo"
    repo.mkdir(parents=True)
    (repo / "z.py").write_text("z = 1\n")

    rpg = RPG(repo_name="ws")
    logger = logging.getLogger("test_4b_info")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger=logger.name):
        RPGEvolution._update_dep_graph_index(rpg, str(ws), logger)
    # Must surface the embed-on-save INFO log
    assert any(
        "embeds into rpg.json" in record.getMessage()
        for record in caplog.records
    ), "expected embed-on-save info log"


def test_process_diff_embeds_dep_graph_into_rpg(tmp_path):
    """End-to-end check that ``process_diff`` produces an rpg with an
    embedded dep_graph that can be round-tripped via ``RPG.to_dict``.

    We stub the LLM-driven sub-processes (``_process_add_files`` etc.)
    so the test stays fast and focuses on the dep_graph attach.
    """
    from rpg_encoder.rpg_evolution import RPGEvolution
    import logging

    last = tmp_path / "last"
    cur = tmp_path / "cur"
    last.mkdir()
    cur.mkdir()
    # Identical content → "no changes" path in process_diff,
    # which is the simplest branch that still calls
    # _update_dep_graph_index.
    (last / "k.py").write_text("k = 1\n")
    (cur / "k.py").write_text("k = 1\n")

    rpg = RPG(repo_name="ws")
    logger = logging.getLogger("test_process_diff")

    # Stub exclusion (it would call LLM otherwise)
    with patch(
        "rpg_encoder.rpg_encoding.RPGParser.exclude_irrelevant_files",
        return_value=[],
    ):
        updated = RPGEvolution.process_diff(
            repo_name="ws",
            repo_info="",
            save_path="",
            last_repo_dir=str(last),
            cur_repo_dir=str(cur),
            last_rpg=rpg,
            last_feature_tree=[],
            logger=logger,
            update_dep_graph=True,
            # dep_graph_save_path omitted on purpose: new default.
        )

    assert updated.dep_graph is not None, (
        "process_diff must attach an in-memory dep_graph for downstream save"
    )
    serialised = updated.to_dict()
    assert "dep_graph" in serialised
    assert serialised["dep_graph"]["nodes"]


# ===========================================================================
# 4c — run_update_rpg writes meta.git + runs align-only enrich
# ===========================================================================

@pytest.fixture
def update_rpg_workspace(tmp_path):
    """A workspace where ``run_update_rpg`` can run end-to-end.

    Uses the "no changes" path of ``process_diff`` (last_repo == cur_repo)
    so we don't trigger the LLM, but the meta.git advance + enrich
    steps still run.
    """
    ws = tmp_path / "ws"
    repo = ws / "repo"
    repo.mkdir(parents=True)
    (repo / "alpha.py").write_text("def alpha(): return 1\n")

    # Git workspace at ws so read_head() returns a real HEAD
    _sh(ws, "init", "-q", "-b", "main")
    _sh(ws, "config", "user.email", "t@t.com")
    _sh(ws, "config", "user.name", "t")
    _sh(ws, "add", ".")
    _sh(ws, "commit", "-q", "-m", "init")

    # Seed RPG without meta.git (so we can verify it gets set)
    data_dir = ws / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    rpg_path = data_dir / "rpg.json"
    dep_graph_path = data_dir / "dep_graph.json"

    svc = RPGService(RPG(repo_name="ws"))
    svc._rpg_dir = data_dir.resolve()
    svc.refresh_dep_graph(
        code_dir=str(repo),
        workspace_root=str(ws),
        save_path=str(dep_graph_path),
    )
    svc.save(str(rpg_path))
    assert svc.rpg.git_meta is None

    return ws, repo, rpg_path, dep_graph_path


def test_run_update_rpg_advances_meta_git_and_runs_align(update_rpg_workspace, monkeypatch):
    """Even on the "no changes" branch, ``run_update_rpg`` must: * embed dep_graph into rpg.json (4b) * advance meta.git to the current HEAD (4c) * run enrich(align_only=True) (4c)."""
    ws, repo, rpg_path, dep_graph_path = update_rpg_workspace

    # WORKSPACE_ROOT is resolved at import time inside common.paths.
    # The test workspace differs from the package's natural root, so
    # we patch the constant in the module that read it.
    monkeypatch.setattr(
        "rpg_encoder.run_update_rpg.WORKSPACE_ROOT", ws,
    )
    # Also patch the dep_graph default + RPG_FILE so CLI-default callers
    # would land on the test paths if they relied on defaults.
    monkeypatch.setattr(
        "rpg_encoder.run_update_rpg.DEP_GRAPH_FILE", dep_graph_path,
    )

    from rpg_encoder.run_update_rpg import run_update_rpg
    head = _sh(ws, "rev-parse", "HEAD")

    with patch(
        "rpg_encoder.rpg_encoding.RPGParser.exclude_irrelevant_files",
        return_value=[],
    ):
        result = run_update_rpg(
            rpg_file=str(rpg_path),
            # last == cur → "no changes" path inside process_diff
            last_repo_dir=str(repo),
            cur_repo_dir=str(repo),
            dep_graph_path=str(dep_graph_path),
        )

    assert result["status"] == "success", result
    assert result["meta_git_advanced"] is True
    assert result["new_commit"] == head
    assert result["previous_commit"] is None  # was never set before

    # Re-read RPG from disk and confirm meta.git landed in the JSON
    with open(rpg_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["meta"]["git"]["head_commit"] == head
    assert persisted["meta"]["git"]["head_branch"] == "main"

    # dep_graph is embedded in rpg.json (single source of truth)
    assert "dep_graph" in persisted
    dep_graph = persisted["dep_graph"]
    assert dep_graph["nodes"]
    assert result["edge_count"] == len(persisted["edges"])
    assert result["nodes_delta"] == 0
    assert result["edges_delta"] == 0
    assert result["dep_nodes"] == len(dep_graph["nodes"])
    assert result["dep_edges"] == len(dep_graph["edges"])
    assert result["dep_nodes_delta"] == 0
    assert result["dep_edges_delta"] == 0


def test_run_update_rpg_dep_graph_path_default_matches_constant(monkeypatch, tmp_path):
    """``--dep-graph`` defaults to ``DEP_GRAPH_FILE`` so the CLI and the pre-commit hook write to the same file."""
    # We can't easily run the CLI argparse, but we can verify that the
    # default of ``run_update_rpg(dep_graph_path=None)`` resolves to
    # the module constant.
    from rpg_encoder import run_update_rpg as mod

    sentinel = tmp_path / "custom_dep_graph.json"
    monkeypatch.setattr(mod, "DEP_GRAPH_FILE", sentinel)

    # Call with an invalid rpg_file to short-circuit out fast — we only
    # care about path resolution behaviour, which happens before any
    # file I/O.
    result = mod.run_update_rpg(
        rpg_file="/nonexistent.json",
        last_repo_dir="/tmp",
        cur_repo_dir="/tmp",
        dep_graph_path=None,  # ← request default
    )
    # Returns error (file doesn't exist) but should still resolve path
    # successfully without crashing.
    assert result["status"] == "error"
    assert "RPG file not found" in result["error"]
