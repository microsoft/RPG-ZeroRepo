#!/usr/bin/env python3
"""Centralized Path Definitions.

This module contains all file path constants used across CoderMind scripts.

Directory layout (``~/.cmind/`` home storage):

    <workspace_root>/             ← user's source repo
    ├── .cmind/                  ← minimal marker tree (in workspace)
    │   ├── config.toml           ← team-shared AI config (committable)
    │   └── reports/              ← user-facing artefacts (rpg.html, …)
    ├── .claude/  or  .vscode/    ← agent instructions
    ├── src/ tests/ …             ← project code (user-owned)
    └── .git/                     ← single git repo at the workspace root

    ~/.cmind/                                  ← user-global storage
    └── workspaces/<workspace-id>/
        ├── .meta.toml                          ← channel, timestamps, version
        ├── .git/                               ← Plan-03 inner snapshot repo
        ├── data/                               ← rpg.json, feature specs, …
        │   └── trajectory/
        └── logs/                               ← *.log, mcp_calls.jsonl, …

Machine-local data (``data/``, ``logs/``, the inner snapshot ``.git/``)
lives under ``~/.cmind/workspaces/<workspace-id>/`` so it survives independently
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
import sys
from pathlib import Path

# Every bundled script imports this module near the top, which makes it
# the natural single choke point to fix a Windows-only crash: when a
# script's stdout/stderr is not a real console (piped by `cmind script`,
# captured by a Claude Code hook, redirected in a test, ...), CPython
# falls back to `locale.getpreferredencoding()` for stdio instead of
# UTF-8. That's a legacy code page (cp1252, cp936, ...) on most Windows
# installs, so a bare `print()` of any non-ASCII character (e.g. the
# "->" arrow in update_graphs.py's status-guidance text) raises
# UnicodeEncodeError and kills the whole script instead of completing.
# Reconfiguring here, at import time, protects every script uniformly
# regardless of how it ends up being invoked.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # AttributeError: stream has no reconfigure (e.g. replaced by a
        # test harness / io.StringIO). ValueError: stream is detached.
        # Either way, this is best-effort — never block script startup.
        pass
del _stream

# Import the home-storage helpers.  cmind_cli is always installed in
# the same Python environment as the scripts (the wheel ships the
# scripts under ``cmind_cli/core_pack/scripts/``), so the import is
# robust to where the script gets invoked from.  We keep a fallback
# that mirrors the legacy in-workspace layout in case someone imports
# this module from a standalone python install that doesn't have
# cmind_cli on sys.path — e.g. a third-party tool dropping in for
# inspection.
try:
    from cmind_cli import _storage as _cmind_storage  # type: ignore[import-not-found]
    _HOME_STORAGE_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _cmind_storage = None  # type: ignore[assignment]
    _HOME_STORAGE_AVAILABLE = False


# ============================================================================
# Workspace Root (absolute)
# ============================================================================
#
# WORKSPACE_ROOT is the directory that contains ``.cmind/``.  Resolving it
# from ``__file__`` is unreliable in dev workflows where ``.cmind/scripts``
# is a symlink to a shared code repo: Python 3.11+ realpath-normalizes the
# script's ``__file__`` at launch, which silently strips the symlink and
# makes ``WORKSPACE_ROOT`` point at the *code* repo instead of the user's
# workspace — every ``DATA_DIR`` / ``REPO_DIR`` derivation then breaks.
#
# Strategy (in order):
#   1. Walk up from ``cwd`` looking for a ``.cmind/`` marker — works for
#      all normal invocations (cmind slash-commands and git hooks launch
#      with cwd at the workspace root).  Authoritative when found, even
#      if a stale ``CMIND_WORKSPACE`` env var inherited from a parent
#      process points elsewhere.
#   2. ``CMIND_WORKSPACE`` env var    — explicit override / fallback
#      when cwd doesn't contain ``.cmind/`` (e.g. running CLI scripts
#      from outside the workspace).
#   3. ``__file__`` fallback           — preserves the standard deployment
#      layout when neither of the above applies.

def _find_workspace_root() -> Path:
    # Prefer cwd-based detection so subprocesses always see the workspace
    # they were launched against, not a stale value inherited from the
    # parent process's environment.  This matters for git hooks, which
    # are spawned by ``git`` (cwd = repo root) from arbitrary parent
    # contexts that may have set CMIND_WORKSPACE long ago.
    #
    # Use the ``.cmind/config.toml`` marker (the canonical workspace
    # signal of "this is an cmind workspace").  Falling back to just
    # ``.cmind/`` would still work for newly-init'd workspaces, but
    # using ``config.toml`` matches :func:`cmind_cli._storage
    # .find_workspace_root_from` exactly so the MCP server and pipeline
    # scripts agree on the boundary.
    cwd = Path.cwd().absolute()
    for cand in [cwd, *cwd.parents]:
        if (cand / ".cmind" / "config.toml").is_file():
            return cand
        # Belt-and-braces fallback: also accept a bare ``.cmind/``
        # directory.  This lets a freshly-cloned workspace whose
        # ``config.toml`` was somehow missing still be discovered
        # rather than silently degrading to the env-var path below.
        if (cand / ".cmind").is_dir():
            return cand

    env = os.environ.get("CMIND_WORKSPACE")
    if env:
        p = Path(env).absolute()
        if p.is_dir():
            return p

    # Last resort: standard deployment layout
    # <workspace_root>/.cmind/scripts/common/paths.py
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
# ``<site-packages>/cmind_cli/core_pack/scripts/`` and are invoked via
# ``cmind script <name>``.
#
# The surrounding ``common/`` package is at
# ``SCRIPTS_DIR/common/``, so ``Path(__file__).parent.parent`` is the
# scripts root.  Callers that need to spawn or sys.path-insert sibling
# code (e.g. ``rpg_edit/impact.py``) get a working path automatically.
#
# For *user-facing hints* embedded in ``next_action`` messages, prefer
# :func:`cmd_for` instead of stringifying ``SCRIPTS_DIR`` — the former
# emits the supported ``cmind script <name>`` invocation rather than a
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
    """Return the canonical ``cmind script`` invocation for a script.

    Args:
        script_relpath: Path relative to the scripts root, e.g.
            ``"run_batch.py"`` or ``"rpg_edit/validate.py"``.  Leading
            slashes are stripped; ``.py`` suffix is preserved.

    Returns:
        A shell-ready string such as ``"cmind script run_batch.py"``.

    Use this for any ``next_action`` hint or error message that
    suggests the user run a script.  The workspace no
    longer hosts a ``.cmind/scripts/`` copy, so the historic
    ``python3 .cmind/scripts/X.py`` form would fail; ``cmind script
    X.py`` works regardless of workspace layout.
    """
    return f"cmind script {script_relpath.lstrip('/')}"


# ============================================================================
# .cmind Directory Structure (runtime state in user home)
# ==========================================================================
#
# Layout:
#
#   CMIND_DIR    = <workspace>/.cmind/      (minimal marker tree: config.toml + .source)
#   DATA_DIR      = ~/.cmind/workspaces/<workspace-id>/data/
#   LOGS_DIR      = ~/.cmind/workspaces/<workspace-id>/logs/
#   REPORTS_DIR   = <workspace>/.cmind/reports/   (kept in workspace by
#                   design: small, user-facing, may be git-tracked)
#
# Falling back to the legacy in-workspace paths when ``_storage`` is
# unavailable keeps this module importable from third-party tools that
# don't ship cmind_cli in the same env.

CMIND_DIR = WORKSPACE_ROOT / ".cmind"

if _HOME_STORAGE_AVAILABLE and _cmind_storage is not None:
    DATA_DIR = _cmind_storage.workspace_data_dir(WORKSPACE_ROOT)
    LOGS_DIR = _cmind_storage.workspace_logs_dir(WORKSPACE_ROOT)
    REPORTS_DIR = _cmind_storage.workspace_reports_dir(WORKSPACE_ROOT)
else:
    DATA_DIR = CMIND_DIR / "data"
    LOGS_DIR = CMIND_DIR / "logs"
    REPORTS_DIR = CMIND_DIR / "reports"

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
# ``DEP_GRAPH_FILE``: legacy standalone dep_graph location.
# As of the embed migration the dep_graph rides inside ``rpg.json``
# (``RPG.to_dict(include_dep_graph=True)``).  New code no longer writes
# this file; the constant stays so legacy workspaces with an existing
# ``dep_graph.json`` continue to load via ``RPGService.load``'s compat
# path, and so a few CLI flags (``--dep-graph`` in update_graphs.py /
# rpg_visualize.py / rpg_edit/apply.py) still resolve a sensible
# default.  Safe to remove once those CLIs are pruned in a future
# breaking-change release.
DEP_GRAPH_FILE = DATA_DIR / "dep_graph.json"
REPO_INFO_FILE = DATA_DIR / "repo_info.json"

# rpg.html lives in REPORTS_DIR (workspace-side) rather than next to
# rpg.json (home-side) because the HTML is a *user-facing* artefact -
# something the developer opens in a browser and may want to share /
# commit alongside the source.  Keeping it in ``.cmind/reports/`` also
# means double-clicking it from a file explorer "just works" without
# having to dig into ``~/.cmind/workspaces/<workspace-id>/``.
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

def ensure_cmind_dir() -> Path:
    """Ensure ``DATA_DIR`` exists and return its path.

    In the home-storage layout, ``DATA_DIR`` lives under
    ``~/.cmind/workspaces/<workspace-id>/data/``.  We only create the leaf
    directory here; full home-layout bootstrap (including
    ``.meta.toml``) is the responsibility of ``cmind init`` /
    ``cmind update``.  Calling this from a script that lands in a
    workspace without a meta file is supported — the data dir still
    gets created and the script can write its output — but the
    workspace won't be properly registered until the user runs
    ``cmind update`` (or ``init``).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_trajectory_file(script_name: str) -> Path:
    """Get trajectory file path for a specific script."""
    return TRAJECTORY_DIR / f"{script_name}_trajectory.json"
