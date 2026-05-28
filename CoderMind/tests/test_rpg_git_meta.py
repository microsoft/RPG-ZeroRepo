#!/usr/bin/env python3
"""Tests for Step 1 of the commit-based incremental-sync plan.

Covers:
  - ``common.git_utils.read_head`` silent-fail behaviour on missing /
    non-git / unborn-HEAD / empty-string paths.
  - ``RPG.set_git_meta`` + ``to_dict`` / ``from_dict`` round-trip,
    including the legacy-without-``meta`` and ``set_git_meta(None)``
    clearing cases.
  - **Decoder-chain safety**: a simulated ``load -> mutate ->
    save -> load -> mutate -> save`` sequence (matching what
    ``build_skeleton`` -> ``build_data_flow`` -> ``design_interfaces``
    -> ``code_gen/rpg_updater`` do in the forward pipeline) must
    preserve ``meta.git`` so subsequent sync hooks know which
    commit the RPG was built against.
  - ``update_graphs.py status`` surfaces ``meta.git`` in JSON output,
    detects in-sync vs. stale, and silently degrades when git is not
    available in the workspace.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "scripts"))

from common.git_utils import read_head  # noqa: E402
from rpg.models import RPG  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def git_repo(tmp_path):
    """Initialise a small git repo with one commit and return its path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# read_head() silent-fail contract
# ---------------------------------------------------------------------------

def test_read_head_returns_dict_on_real_repo(git_repo):
    info = read_head(git_repo)
    assert isinstance(info, dict)
    assert info["head_commit"] and len(info["head_commit"]) == 40
    assert info["head_short"] and info["head_short"].startswith(info["head_commit"][:7])
    assert info["head_branch"] == "main"
    assert info["head_timestamp"] and "T" in info["head_timestamp"]


def test_read_head_returns_none_on_missing_path():
    assert read_head("/tmp/this/path/does/not/exist/zzzz") is None


def test_read_head_returns_none_on_non_git_dir(tmp_path):
    assert read_head(tmp_path) is None


def test_read_head_returns_none_on_empty_string():
    # Empty string would resolve to cwd if not guarded; we want None.
    assert read_head("") is None


def test_read_head_returns_none_on_unborn_head(tmp_path):
    """A freshly-init'd repo with zero commits has no HEAD to resolve."""
    repo = tmp_path / "empty"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    assert read_head(repo) is None


def test_read_head_returns_none_branch_on_detached_head(git_repo):
    """Detached HEAD: head_commit is set but head_branch is None."""
    # Detach by checking out the SHA directly
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-c", "advice.detachedHead=false", "checkout", "-q", sha],
        cwd=git_repo, check=True, capture_output=True,
    )
    info = read_head(git_repo)
    assert info is not None
    assert info["head_commit"] == sha
    assert info["head_branch"] is None  # detached → no branch


# ---------------------------------------------------------------------------
# RPG.set_git_meta + to_dict/from_dict round-trip
# ---------------------------------------------------------------------------

GIT_META_SAMPLE = {
    "head_commit": "8a3f9c1d4e2b1234567890abcdef0123456789ab",
    "head_short": "8a3f9c1",
    "head_branch": "main",
    "head_timestamp": "2026-05-12T08:30:00+00:00",
}


def test_fresh_rpg_has_no_git_meta():
    rpg = RPG(repo_name="demo")
    assert rpg.git_meta is None
    d = rpg.to_dict()
    # ``meta`` is always emitted as an object but ``git`` is omitted when unset
    assert d.get("meta") == {}


def test_set_and_round_trip_git_meta():
    rpg = RPG(repo_name="demo")
    rpg.set_git_meta(**GIT_META_SAMPLE)
    d = rpg.to_dict()
    assert d["meta"]["git"] == GIT_META_SAMPLE

    rpg2 = RPG.from_dict(d)
    assert rpg2.git_meta == GIT_META_SAMPLE


def test_set_git_meta_with_none_clears_state():
    rpg = RPG(repo_name="demo")
    rpg.set_git_meta(**GIT_META_SAMPLE)
    assert rpg.git_meta is not None
    rpg.set_git_meta(None)
    assert rpg.git_meta is None
    assert rpg.to_dict()["meta"] == {}


def test_legacy_rpg_without_meta_loads_cleanly():
    """Old rpg.json predating Step 1 must load with ``git_meta=None``."""
    legacy = {
        "repo_name": "legacy",
        "repo_info": "",
        "excluded_files": [],
        "root": None,
        "edges": [],
    }
    rpg = RPG.from_dict(legacy)
    assert rpg.git_meta is None


def test_meta_git_with_unknown_keys_is_filtered():
    """Defence against future schema evolution: unknown ``meta.git`` keys must not pollute the in-memory representation."""
    payload = RPG(repo_name="x").to_dict()
    payload["meta"] = {
        "git": {
            **GIT_META_SAMPLE,
            "rogue_field": "should be dropped",
            "another": 42,
        },
    }
    rpg = RPG.from_dict(payload)
    assert rpg.git_meta is not None
    assert set(rpg.git_meta) == set(GIT_META_SAMPLE)


def test_meta_git_without_head_commit_is_ignored():
    """``meta.git`` without ``head_commit`` is useless — must produce ``git_meta=None`` rather than a half-populated dict."""
    payload = RPG(repo_name="x").to_dict()
    payload["meta"] = {"git": {"head_branch": "main"}}
    rpg = RPG.from_dict(payload)
    assert rpg.git_meta is None


def test_partial_git_meta_keeps_none_for_optional_fields():
    """Only ``head_commit`` is required; the rest may legitimately be ``None`` (e.g. detached HEAD has no branch)."""
    rpg = RPG(repo_name="x")
    rpg.set_git_meta(head_commit="abc123" * 7)
    d = rpg.to_dict()
    assert d["meta"]["git"]["head_commit"].startswith("abc")
    assert d["meta"]["git"]["head_branch"] is None

    rpg2 = RPG.from_dict(d)
    assert rpg2.git_meta is not None
    assert rpg2.git_meta["head_branch"] is None


# ---------------------------------------------------------------------------
# Decoder-chain safety: meta.git must survive load/save loops
# ---------------------------------------------------------------------------

def test_meta_git_survives_decoder_chain_save_load_loops(tmp_path):
    """Simulate the forward pipeline: build_skeleton (set meta.git) → load+modify → save → load+modify → save.

    The chain mutates other fields each iteration (mimicking what
    ``build_data_flow.py`` / ``design_interfaces.py`` / ``code_gen``
    do).  ``meta.git`` must survive every round-trip untouched.
    """
    rpg = RPG(repo_name="demo", repo_info="initial")
    rpg.set_git_meta(**GIT_META_SAMPLE)

    rpg_path = tmp_path / "rpg.json"
    rpg.save_json(str(rpg_path))

    for iteration in range(3):
        # load
        with open(rpg_path, "r", encoding="utf-8") as f:
            loaded = RPG.from_dict(json.load(f))
        assert loaded.git_meta == GIT_META_SAMPLE, (
            f"meta.git lost on iteration {iteration}"
        )

        # mutate something orthogonal (excluded_files counts as a "phase"
        # touching RPG state without intentionally clearing meta).
        loaded.excluded_files = list(loaded.excluded_files) + [
            f"junk_{iteration}.py"
        ]
        loaded.repo_info = f"phase-{iteration}"

        # save
        loaded.save_json(str(rpg_path))

    # Final load
    with open(rpg_path, "r", encoding="utf-8") as f:
        final = RPG.from_dict(json.load(f))
    assert final.git_meta == GIT_META_SAMPLE
    assert final.repo_info == "phase-2"
    assert "junk_0.py" in final.excluded_files
    assert "junk_2.py" in final.excluded_files


# ---------------------------------------------------------------------------
# update_graphs.py status integration
# ---------------------------------------------------------------------------

def _run_status_json(rpg_path: Path, cwd: Path) -> dict:
    """Run ``update_graphs.py status --json`` and return parsed dict.

    ``cwd`` controls which git repo (if any) the helper inspects for the
    current HEAD, so callers can simulate "RPG in non-git workspace" vs.
    "RPG inside an active git repo".
    """
    script = _project_root / "scripts" / "update_graphs.py"
    result = subprocess.run(
        [sys.executable, str(script), "status", "--rpg", str(rpg_path), "--json"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_status_shows_in_sync_when_meta_matches_head(tmp_path, git_repo):
    rpg = RPG(repo_name="x")
    info = read_head(git_repo)
    rpg.set_git_meta(
        head_commit=info["head_commit"],
        head_short=info["head_short"],
        head_branch=info["head_branch"],
        head_timestamp=info["head_timestamp"],
    )
    rpg_path = tmp_path / "rpg.json"
    rpg.save_json(str(rpg_path))

    data = _run_status_json(rpg_path, cwd=git_repo)
    assert data["last_synced_commit"] == info["head_commit"]
    assert data["current_commit"] == info["head_commit"]
    assert data["rpg_in_sync_with_head"] is True


def test_status_shows_stale_when_meta_differs_from_head(tmp_path, git_repo):
    rpg = RPG(repo_name="x")
    rpg.set_git_meta(**GIT_META_SAMPLE)  # synthetic, definitely not HEAD
    rpg_path = tmp_path / "rpg.json"
    rpg.save_json(str(rpg_path))

    data = _run_status_json(rpg_path, cwd=git_repo)
    assert data["last_synced_commit"] == GIT_META_SAMPLE["head_commit"]
    assert data["current_commit"] != GIT_META_SAMPLE["head_commit"]
    assert data["rpg_in_sync_with_head"] is False


def test_status_omits_current_commit_outside_git(tmp_path):
    """Outside a git workspace, ``current_commit`` is absent and ``rpg_in_sync_with_head`` is left unset — never crashes."""
    rpg = RPG(repo_name="x")
    rpg.set_git_meta(**GIT_META_SAMPLE)
    rpg_path = tmp_path / "rpg.json"
    rpg.save_json(str(rpg_path))

    # tmp_path is not a git repo
    data = _run_status_json(rpg_path, cwd=tmp_path)
    assert data["last_synced_commit"] == GIT_META_SAMPLE["head_commit"]
    assert "current_commit" not in data
    assert "rpg_in_sync_with_head" not in data


def test_status_legacy_rpg_without_meta_still_works(tmp_path, git_repo):
    """An rpg.json that pre-dates Step 1 must still produce valid status output (no ``last_synced_*`` keys), and the agent text rendering must not include any "Last synced" line."""
    rpg = RPG(repo_name="legacy")
    rpg_path = tmp_path / "rpg.json"
    rpg.save_json(str(rpg_path))

    data = _run_status_json(rpg_path, cwd=git_repo)
    assert "last_synced_commit" not in data
    # current_commit is still surfaced — useful when /cmind.encode runs next
    assert "current_commit" in data

    # Text-mode output should not advertise sync state
    script = _project_root / "scripts" / "update_graphs.py"
    text = subprocess.run(
        [sys.executable, str(script), "status", "--rpg", str(rpg_path)],
        cwd=str(git_repo), capture_output=True, text=True,
    ).stdout
    assert "Last synced" not in text
