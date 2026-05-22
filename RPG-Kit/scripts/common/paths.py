#!/usr/bin/env python3
"""Centralized Path Definitions.

This module contains all file path constants used across RPG-Kit scripts.

Directory layout (``~/.rpgkit/`` home storage):

    <workspace_root>/             ← user's source repo
    ├── .rpgkit/                  ← minimal marker tree (in workspace)
    │   ├── config.toml           ← team-shared AI config (committable)
    │   └── reports/              ← user-facing artefacts (rpg.html, …)
    ├── .claude/  or  .vscode/    ← agent instructions
    ├── src/ tests/ …             ← project code (user-owned)
    └── .git/                     ← single git repo at the workspace root

    ~/.rpgkit/                                  ← user-global storage
    └── workspaces/<workspace-id>/
        ├── .meta.toml                          ← channel, timestamps, version
        ├── .git/                               ← Plan-03 inner snapshot repo
        ├── data/                               ← rpg.json, dep_graph.json, …
        │   └── trajectory/
        └── logs/                               ← *.log, mcp_calls.jsonl, …

Machine-local data (``data/``, ``logs/``, the inner snapshot ``.git/``)
lives under ``~/.rpgkit/workspaces/<workspace-id>/`` so it survives independently
of the workspace, never gets accidentally committed, and stays scoped
to one user.  The workspace dir keeps only the lightweight, team-shared
files that benefit from being version-controlled alongside the code.

All constants below resolve at module-import time.  ``WORKSPACE_ROOT``
is discovered once; if you need it to track a different workspace
later in the same process (rare), spawn a subprocess instead of
monkey-patching the module.

``REPO_DIR`` is an alias for ``WORKSPACE_ROOT`` kept for backwards
compatibility with call sites that use "project repo root" phrasing;
both refer to the same directory.
"""

import os
from pathlib import Path

# Import the home-storage helpers.  rpgkit_cli is always installed in
# the same Python environment as the scripts (the wheel ships the
# scripts under ``rpgkit_cli/core_pack/scripts/``), so the import is
# robust to where the script gets invoked from.  We keep a fallback
# that mirrors the legacy in-workspace layout in case someone imports
# this module from a standalone python install that doesn't have
# rpgkit_cli on sys.path — e.g. a third-party tool dropping in for
# inspection.
try:
    from rpgkit_cli import _storage as _rpgkit_storage  # type: ignore[import-not-found]
    _HOME_STORAGE_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _rpgkit_storage = None  # type: ignore[assignment]
    _HOME_STORAGE_AVAILABLE = False


# ============================================================================
# Workspace Root (absolute)
# ============================================================================
#
# WORKSPACE_ROOT is the directory that contains ``.rpgkit/``.  Resolving it
# from ``__file__`` is unreliable in dev workflows where ``.rpgkit/scripts``
# is a symlink to a shared code repo: Python 3.11+ realpath-normalizes the
# script's ``__file__`` at launch, which silently strips the symlink and
# makes ``WORKSPACE_ROOT`` point at the *code* repo instead of the user's
# workspace — every ``DATA_DIR`` / ``REPO_DIR`` derivation then breaks.
#
# Strategy (in order):
#   1. Walk up from ``cwd`` looking for a ``.rpgkit/`` marker — works for
#      all normal invocations (rpgkit slash-commands and git hooks launch
#      with cwd at the workspace root).  Authoritative when found, even
#      if a stale ``RPGKIT_WORKSPACE`` env var inherited from a parent
#      process points elsewhere.
#   2. ``RPGKIT_WORKSPACE`` env var    — explicit override / fallback
#      when cwd doesn't contain ``.rpgkit/`` (e.g. running CLI scripts
#      from outside the workspace).
#   3. ``__file__`` fallback           — preserves the standard deployment
#      layout when neither of the above applies.

def _find_workspace_root() -> Path:
    # Prefer cwd-based detection so subprocesses always see the workspace
    # they were launched against, not a stale value inherited from the
    # parent process's environment.  This matters for git hooks, which
    # are spawned by ``git`` (cwd = repo root) from arbitrary parent
    # contexts that may have set RPGKIT_WORKSPACE long ago.
    #
    # Use the ``.rpgkit/config.toml`` marker (the canonical workspace
    # signal of "this is an rpgkit workspace").  Falling back to just
    # ``.rpgkit/`` would still work for newly-init'd workspaces, but
    # using ``config.toml`` matches :func:`rpgkit_cli._storage
    # .find_workspace_root_from` exactly so the MCP server and pipeline
    # scripts agree on the boundary.
    cwd = Path.cwd().absolute()
    for cand in [cwd, *cwd.parents]:
        if (cand / ".rpgkit" / "config.toml").is_file():
            return cand
        # Belt-and-braces fallback: also accept a bare ``.rpgkit/``
        # directory.  This lets a freshly-cloned workspace whose
        # ``config.toml`` was somehow missing still be discovered
        # rather than silently degrading to the env-var path below.
        if (cand / ".rpgkit").is_dir():
            return cand

    env = os.environ.get("RPGKIT_WORKSPACE")
    if env:
        p = Path(env).absolute()
        if p.is_dir():
            return p

    # Last resort: standard deployment layout
    # <workspace_root>/.rpgkit/scripts/common/paths.py
    return Path(__file__).absolute().parent.parent.parent.parent


WORKSPACE_ROOT = _find_workspace_root()


# ============================================================================
# Project Repo Directory
# ============================================================================
#
# Historically the user's code lived at ``<workspace_root>/repo/``, with
# a separate inner git repo.  That layout has been retired: the
# workspace root **is** the project repo root, so ``REPO_DIR`` and
# ``WORKSPACE_ROOT`` are now aliases for the same directory.  Callers
# may prefer one name over the other based on which concept reads
# more naturally at the call site.

REPO_DIR = WORKSPACE_ROOT


# ============================================================================
# Scripts Directory (absolute path on the filesystem)
# ============================================================================
#
# Anchor SCRIPTS_DIR to ``__file__``'s parent so the constant resolves
# correctly regardless of how the scripts were deployed.  Scripts live
# inside the installed wheel at
# ``<site-packages>/rpgkit_cli/core_pack/scripts/`` and are invoked via
# ``rpgkit script <name>``.
#
# The surrounding ``common/`` package is at
# ``SCRIPTS_DIR/common/``, so ``Path(__file__).parent.parent`` is the
# scripts root.  Callers that need to spawn or sys.path-insert sibling
# code (e.g. ``rpg_edit/impact.py``) get a working path automatically.
#
# For *user-facing hints* embedded in ``next_action`` messages, prefer
# :func:`cmd_for` instead of stringifying ``SCRIPTS_DIR`` — the former
# emits the supported ``rpgkit script <name>`` invocation rather than a
# raw filesystem path the user can't easily re-run.

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = SCRIPTS_DIR / "tools"


def get_scripts_dir() -> str:
    """Return the scripts directory as a string (filesystem path).

    Kept for backward compatibility with code that uses this as a
    base path for sibling-script Path/sys.path operations.  Do NOT
    use this to build invocation strings shown to the user — use
    :func:`cmd_for` instead.
    """
    return str(SCRIPTS_DIR)


def cmd_for(script_relpath: str) -> str:
    """Return the canonical ``rpgkit script`` invocation for a script.

    Args:
        script_relpath: Path relative to the scripts root, e.g.
            ``"run_batch.py"`` or ``"rpg_edit/validate.py"``.  Leading
            slashes are stripped; ``.py`` suffix is preserved.

    Returns:
        A shell-ready string such as ``"rpgkit script run_batch.py"``.

    Use this for any ``next_action`` hint or error message that
    suggests the user run a script.  The workspace no
    longer hosts a ``.rpgkit/scripts/`` copy, so the historic
    ``python3 .rpgkit/scripts/X.py`` form would fail; ``rpgkit script
    X.py`` works regardless of workspace layout.
    """
    return f"rpgkit script {script_relpath.lstrip('/')}"


# ============================================================================
# .rpgkit Directory Structure (runtime state in user home)
# ==========================================================================
#
# Layout:
#
#   RPGKIT_DIR    = <workspace>/.rpgkit/      (minimal marker tree: config.toml + .source)
#   DATA_DIR      = ~/.rpgkit/workspaces/<workspace-id>/data/
#   LOGS_DIR      = ~/.rpgkit/workspaces/<workspace-id>/logs/
#   REPORTS_DIR   = <workspace>/.rpgkit/reports/   (kept in workspace by
#                   design: small, user-facing, may be git-tracked)
#
# Falling back to the legacy in-workspace paths when ``_storage`` is
# unavailable keeps this module importable from third-party tools that
# don't ship rpgkit_cli in the same env.

RPGKIT_DIR = WORKSPACE_ROOT / ".rpgkit"

if _HOME_STORAGE_AVAILABLE and _rpgkit_storage is not None:
    DATA_DIR = _rpgkit_storage.workspace_data_dir(WORKSPACE_ROOT)
    LOGS_DIR = _rpgkit_storage.workspace_logs_dir(WORKSPACE_ROOT)
    REPORTS_DIR = _rpgkit_storage.workspace_reports_dir(WORKSPACE_ROOT)
else:
    DATA_DIR = RPGKIT_DIR / "data"
    LOGS_DIR = RPGKIT_DIR / "logs"
    REPORTS_DIR = RPGKIT_DIR / "reports"

COPILOT_LOGS_DIR = LOGS_DIR / "copilot"
CLAUDE_LOGS_DIR = LOGS_DIR / "claude"


# ============================================================================
# Dev Virtual Environment
# ============================================================================
#
# The codegen pipeline creates an isolated venv under the project repo so
# tests run against an exact dependency set without polluting the user's
# global Python.  ``DEV_VENV_NAME`` is the directory name (used in
# ``.gitignore`` patterns); ``DEV_VENV_DIR`` is the absolute path.

DEV_VENV_NAME = ".venv_dev"
DEV_VENV_DIR = REPO_DIR / DEV_VENV_NAME


# ============================================================================
# Feature Specification & Build (data/ subfolder)
# ============================================================================

FEATURE_SPEC_FILE = DATA_DIR / "feature_spec.json"
FEATURE_BUILD_FILE = DATA_DIR / "feature_build.json"
FEATURE_TREE_FILE = DATA_DIR / "feature_tree.json"


# ============================================================================
# Skeleton Files
# ============================================================================

SKELETON_FILE = DATA_DIR / "skeleton.json"
SKELETON_SUMMARY_FILE = DATA_DIR / "skeleton_summary.txt"


# ============================================================================
# Data Flow & Interfaces
# ============================================================================

DATA_FLOW_FILE = DATA_DIR / "data_flow.json"
DATA_FLOW_VIZ_FILE = DATA_DIR / "data_flow_viz.html"
INTERFACES_FILE = DATA_DIR / "interfaces.json"
BASE_CLASSES_FILE = DATA_DIR / "base_classes.json"


# ============================================================================
# RPG (Repository Program Graph)
# ============================================================================

RPG_FILE = DATA_DIR / "rpg.json"
REPO_RPG_FILE = RPG_FILE  # Unified: both encoder and decoder use rpg.json
DEP_GRAPH_FILE = DATA_DIR / "dep_graph.json"
REPO_INFO_FILE = DATA_DIR / "repo_info.json"

# rpg.html lives in REPORTS_DIR (workspace-side) rather than next to
# rpg.json (home-side) because the HTML is a *user-facing* artefact -
# something the developer opens in a browser and may want to share /
# commit alongside the source.  Keeping it in ``.rpgkit/reports/`` also
# means double-clicking it from a file explorer "just works" without
# having to dig into ``~/.rpgkit/workspaces/<workspace-id>/``.
RPG_HTML_FILE = REPORTS_DIR / "rpg.html"


# ============================================================================
# Task Planning & Execution
# ============================================================================

TASKS_FILE = DATA_DIR / "tasks.json"
CODE_GEN_STATE_FILE = DATA_DIR / "code_gen_state.jsonl"


# ============================================================================
# RPG Edit (surgical edit pipeline) — well-known artefact locations under
# ``DATA_DIR``. Scripts default their ``--plan`` / ``--impact`` arguments
# to these paths so slash-command templates don't need to know the
# physical (home-dir) location of the workspace.
# ============================================================================

RPG_EDIT_PLAN_FILE = DATA_DIR / "rpg_edit_plan.json"
RPG_EDIT_IMPACT_FILE = DATA_DIR / "rpg_edit_impact.json"
RPG_EDIT_CODE_RESULT_FILE = DATA_DIR / "rpg_edit_code_result.json"
RPG_EDIT_REVIEW_RESULT_FILE = DATA_DIR / "rpg_edit_review_result.json"


# ============================================================================
# Trajectory & Logging
# ============================================================================

TRAJECTORY_DIR = DATA_DIR / "trajectory"


# ============================================================================
# Telemetry (JSONL append-only logs for usage statistics)
# ============================================================================

MCP_CALLS_LOG = LOGS_DIR / "mcp_calls.jsonl"
HOOK_CALLS_LOG = LOGS_DIR / "hook_calls.jsonl"
# REPORTS_DIR is defined above (workspace-local in the new layout).


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_rpgkit_dir() -> Path:
    """Ensure ``DATA_DIR`` exists and return its path.

    In the home-storage layout, ``DATA_DIR`` lives under
    ``~/.rpgkit/workspaces/<workspace-id>/data/``.  We only create the leaf
    directory here; full home-layout bootstrap (including
    ``.meta.toml``) is the responsibility of ``rpgkit init`` /
    ``rpgkit update``.  Calling this from a script that lands in a
    workspace without a meta file is supported — the data dir still
    gets created and the script can write its output — but the
    workspace won't be properly registered until the user runs
    ``rpgkit update`` (or ``init``).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_trajectory_file(script_name: str) -> Path:
    """Get trajectory file path for a specific script."""
    return TRAJECTORY_DIR / f"{script_name}_trajectory.json"
