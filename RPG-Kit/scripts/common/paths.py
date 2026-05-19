#!/usr/bin/env python3
"""Centralized Path Definitions.

This module contains all file path constants used across RPG-Kit scripts.

Directory layout (workspace == repo):
    <workspace_root>/             ← user's source repo + RPG-Kit data
    ├── .rpgkit/                  ← scripts, data, state (machine-local)
    ├── .claude/  or  .vscode/    ← agent instructions
    ├── src/ tests/ …             ← project code (user-owned)
    └── .git/                     ← single git repo at the workspace root

All paths under ``.rpgkit/`` and ``.claude/`` are relative to
``WORKSPACE_ROOT``.  ``REPO_DIR`` is an alias for ``WORKSPACE_ROOT`` kept for
backwards-compatibility with call sites that use "project repo root"
phrasing; both refer to the same directory.
"""

import os
from pathlib import Path


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
    cwd = Path.cwd().absolute()
    for cand in [cwd, *cwd.parents]:
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
# correctly regardless of how the scripts were deployed:
#
#   * Pre-0.1.3 layout: scripts copied into ``<workspace>/.rpgkit/scripts/``.
#   * Post-0.1.3 layout: scripts live inside the installed wheel at
#     ``<site-packages>/rpgkit_cli/core_pack/scripts/`` and are invoked
#     via ``rpgkit script <name>`` (see plan 02).
#
# In both cases the surrounding ``common/`` package is at
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
    suggests the user run a script.  After plan 02, the workspace no
    longer hosts a ``.rpgkit/scripts/`` copy, so the historic
    ``python3 .rpgkit/scripts/X.py`` form would fail; ``rpgkit script
    X.py`` works regardless of workspace layout.
    """
    return f"rpgkit script {script_relpath.lstrip('/')}"


# ============================================================================
# .rpgkit Directory Structure (absolute, derived from WORKSPACE_ROOT)
# ============================================================================

RPGKIT_DIR = WORKSPACE_ROOT / ".rpgkit"
DATA_DIR = RPGKIT_DIR / "data"
LOGS_DIR = RPGKIT_DIR / "logs"
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


# ============================================================================
# Task Planning & Execution
# ============================================================================

TASKS_FILE = DATA_DIR / "tasks.json"
CODE_GEN_STATE_FILE = DATA_DIR / "code_gen_state.jsonl"


# ============================================================================
# Trajectory & Logging
# ============================================================================

TRAJECTORY_DIR = DATA_DIR / "trajectory"


# ============================================================================
# Telemetry (JSONL append-only logs for usage statistics)
# ============================================================================

MCP_CALLS_LOG = LOGS_DIR / "mcp_calls.jsonl"
HOOK_CALLS_LOG = LOGS_DIR / "hook_calls.jsonl"
REPORTS_DIR = RPGKIT_DIR / "reports"


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_rpgkit_dir() -> Path:
    """Ensure .rpgkit/data directory exists and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_trajectory_file(script_name: str) -> Path:
    """Get trajectory file path for a specific script."""
    return TRAJECTORY_DIR / f"{script_name}_trajectory.json"
