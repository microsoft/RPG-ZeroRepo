"""CoderMind CLI - Provision and manage Repository Planning Graph (RPG) workspaces for AI coding agents.

Usage:
    uvx cmind-cli init <project-name>
    uvx cmind-cli init .
    uvx cmind-cli init --here

Or install globally:
    uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
    cmind init <project-name>
    cmind init .
    cmind init --here
"""

import os
import re
import subprocess
import sys
import threading
import time
import zipfile
import tempfile
import shutil
import shlex
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.table import Table
from rich.tree import Tree
from typer.core import TyperGroup

# For cross-platform keyboard input
import readchar
import ssl
import truststore
from datetime import datetime, timezone
import platform
import importlib.metadata
import tomllib

from . import _storage

ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
client = httpx.Client(verify=ssl_context)

# Default fallback values — only used when git remote and pyproject.toml are unavailable
_FALLBACK_REPO_OWNER = "microsoft"
_FALLBACK_REPO_NAME = "RPG-ZeroRepo"
_CMIND_RELEASE_TAG_PREFIX = "cmind-v"


# ---------------------------------------------------------------------------
# DEPRECATED: GitHub release-zip provisioning helpers.
#
# As of v0.1.4 ``cmind init`` / ``cmind update`` are bundle-only and no
# longer fetch templates from GitHub releases at runtime — users upgrade
# the CLI itself to pick up newer prompts.  The helpers below
# (``_parse_github_owner_repo``, ``_github_token``,
# ``_github_auth_headers``, ``_parse_rate_limit_headers``,
# ``_format_rate_limit_error``, ``_is_private_repo``,
# ``_get_asset_download_url``, ``_fetch_latest_cmind_release``,
# ``download_template_from_github``, ``_download_and_extract_release_zip``)
# are kept temporarily so the change is reversible and so any third-party
# callers don't break on upgrade.  They are slated for removal in v0.2.0
# along with the ``httpx`` dependency they bring in.
# ---------------------------------------------------------------------------


def _parse_github_owner_repo(url: str) -> Tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub remote URL.

    Supports:
      - git@github.com:Owner/Repo.git
      - https://github.com/Owner/Repo.git
      - https://github.com/Owner/Repo
    """
    import re

    # SSH format: git@github.com:Owner/Repo.git
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)

    # HTTPS format: https://github.com/Owner/Repo[.git]
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)

    return None


def _get_repo_info() -> Tuple[str, str]:
    """Resolve the GitHub owner/repo for CoderMind template downloads.

    Priority:
      1. git remote 'upstream' (fork scenario — points to original repo)
      2. git remote 'origin' (most common default)
      3. pyproject.toml [project.urls].Repository
      4. Hardcoded fallback
    """
    # Try git remotes: upstream first, then origin
    for remote_name in ("upstream", "origin"):
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", remote_name],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                parsed = _parse_github_owner_repo(url)
                if parsed:
                    return parsed
        except Exception:
            pass

    # Try pyproject.toml
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            repo_url = data.get("project", {}).get("urls", {}).get("Repository", "")
            if repo_url:
                parsed = _parse_github_owner_repo(repo_url)
                if parsed:
                    return parsed
    except Exception:
        pass

    return _FALLBACK_REPO_OWNER, _FALLBACK_REPO_NAME


def _github_token(cli_token: str | None = None) -> str | None:
    """Return sanitized GitHub token (cli arg takes precedence) or None."""
    return (
        (cli_token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    ) or None


def _github_auth_headers(
    cli_token: str | None = None, accept_asset: bool = False
) -> dict:
    """Return Authorization header dict only when a non-empty token exists.

    Args:
        cli_token: Optional GitHub token
        accept_asset: If True, add Accept header for binary asset download (required for private repos)

    Returns:
        Dictionary with appropriate headers
    """
    token = _github_token(cli_token)
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if accept_asset:
        # Required for downloading release assets from private repos via API
        headers["Accept"] = "application/octet-stream"

    return headers


def _parse_rate_limit_headers(headers: httpx.Headers) -> dict:
    """Extract and parse GitHub rate-limit headers."""
    info = {}

    # Standard GitHub rate-limit headers
    if "X-RateLimit-Limit" in headers:
        info["limit"] = headers.get("X-RateLimit-Limit")
    if "X-RateLimit-Remaining" in headers:
        info["remaining"] = headers.get("X-RateLimit-Remaining")
    if "X-RateLimit-Reset" in headers:
        reset_epoch = int(headers.get("X-RateLimit-Reset", "0"))
        if reset_epoch:
            reset_time = datetime.fromtimestamp(reset_epoch, tz=timezone.utc)
            info["reset_epoch"] = reset_epoch
            info["reset_time"] = reset_time
            info["reset_local"] = reset_time.astimezone()

    # Retry-After header (seconds or HTTP-date)
    if "Retry-After" in headers:
        retry_after = headers.get("Retry-After")
        try:
            info["retry_after_seconds"] = int(retry_after)
        except ValueError:
            # HTTP-date format - not implemented, just store as string
            info["retry_after"] = retry_after

    return info


def _format_rate_limit_error(status_code: int, headers: httpx.Headers, url: str) -> str:
    """Format a user-friendly error message with rate-limit information."""
    rate_info = _parse_rate_limit_headers(headers)

    lines = [f"GitHub API returned status {status_code} for {url}"]
    lines.append("")

    if rate_info:
        lines.append("[bold]Rate Limit Information:[/bold]")
        if "limit" in rate_info:
            lines.append(f"  • Rate Limit: {rate_info['limit']} requests/hour")
        if "remaining" in rate_info:
            lines.append(f"  • Remaining: {rate_info['remaining']}")
        if "reset_local" in rate_info:
            reset_str = rate_info["reset_local"].strftime("%Y-%m-%d %H:%M:%S %Z")
            lines.append(f"  • Resets at: {reset_str}")
        if "retry_after_seconds" in rate_info:
            lines.append(f"  • Retry after: {rate_info['retry_after_seconds']} seconds")
        lines.append("")

    # Add troubleshooting guidance
    lines.append("[bold]Troubleshooting Tips:[/bold] !")
    lines.append(
        "  • If you're on a shared CI or corporate environment, you may be rate-limited."
    )
    lines.append(
        "  • Consider using a GitHub token via --github-token or the GH_TOKEN/GITHUB_TOKEN"
    )
    lines.append("    environment variable to increase rate limits.")
    lines.append(
        "  • Authenticated requests have a limit of 5,000/hour vs 60/hour for unauthenticated."
    )

    return "\n".join(lines)


# Agent configuration with name, folder, install URL, and CLI tool requirement
AGENT_CONFIG = {
    "copilot": {
        "name": "GitHub Copilot",
        "folder": ".github/",
        "install_url": "https://docs.github.com/en/copilot/how-tos/copilot-cli/install-copilot-cli",  # IDE-based, no CLI check needed
        "requires_cli": True,
    },
    "claude": {
        "name": "Claude Code",
        "folder": ".claude/",
        "install_url": "https://docs.anthropic.com/en/docs/claude-code/setup",
        "requires_cli": True,
    },
    # --- Unverified agents (commented out until tested) ---
    # "gemini": {
    #     "name": "Gemini CLI",
    #     "folder": ".gemini/",
    #     "install_url": "https://github.com/google-gemini/gemini-cli",
    #     "requires_cli": True,
    # },
    # "cursor-agent": {
    #     "name": "Cursor",
    #     "folder": ".cursor/",
    #     "install_url": "https://cursor.com/cn/docs/get-started/quickstart",
    #     "requires_cli": True,
    # },
    # "qwen": {
    #     "name": "Qwen Code",
    #     "folder": ".qwen/",
    #     "install_url": "https://github.com/QwenLM/qwen-code",
    #     "requires_cli": True,
    # },
    # "opencode": {
    #     "name": "OpenCode",
    #     "folder": ".opencode/",
    #     "install_url": "https://opencode.ai",
    #     "requires_cli": True,
    # },
    # "codex": {
    #     "name": "Codex CLI",
    #     "folder": ".codex/",
    #     "install_url": "https://github.com/openai/codex",
    #     "requires_cli": True,
    # },
    # "codebuddy": {
    #     "name": "CodeBuddy",
    #     "folder": ".codebuddy/",
    #     "install_url": "https://www.codebuddy.ai/cli",
    #     "requires_cli": True,
    # },
    # "qoder": {
    #     "name": "Qoder",
    #     "folder": ".qoder/",
    #     "install_url": "https://qoder.com/cli",
    #     "requires_cli": True,
    # },
    # "amp": {
    #     "name": "Amp",
    #     "folder": ".agents/",
    #     "install_url": "https://ampcode.com/manual#install",
    #     "requires_cli": True,
    # },
}

SCRIPT_TYPE_CHOICES = {"sh": "POSIX Shell (bash/zsh)", "ps": "PowerShell"}

CLAUDE_LOCAL_PATH = Path.home() / ".claude" / "local" / "claude"


# ---------------------------------------------------------------------------
# Bundle mode (packaged assets) — added in 0.1.3
# ---------------------------------------------------------------------------
#
# cmind-cli ships ``scripts/`` and ``templates/commands/`` as packaged
# assets under ``cmind_cli/core_pack/`` so that ``cmind init`` works
# offline.  This block exposes:
#
#   _AI_TO_CLI_CMD        — single source of truth for "selected AI" →
#                           "AI CLI command to invoke from scripts".
#                           Must stay in sync with the corresponding case
#                           statement in
#                           ``.github/workflows/scripts/cmind/create-release-packages.sh``
#                           (the release-zip pipeline) and with
#                           ``scripts/common/llm_client.py:_CLI_TO_AGENT``
#                           (the reverse mapping consumed by detect_agent_type()).
#
#   _SOURCE_BUNDLE / _SOURCE_LEGACY  — provisioning channel; persisted as
#                                       ``channel`` in ``~/.cmind/workspaces/
#                                       <workspace-id>/.meta.toml`` so subsequent
#                                       ``cmind update`` calls honour the
#                                       user's original choice.  Mirrors the
#                                       constants in :mod:`cmind_cli._storage`.

_AI_TO_CLI_CMD = {
    # NOTE: values below are copied verbatim from
    # .github/workflows/scripts/cmind/create-release-packages.sh lines ~142-169
    # to guarantee bundle mode and legacy-download mode behave identically.
    "copilot":      "copilot",
    "claude":       "claude",
    "gemini":       "gemini -p",
    "qwen":         "qwen -p",
    "cursor-agent": "agent -p",
    "auggie":       "augment -p",
    "codex":        "codex exec",
    "codebuddy":    "codebuddy -p",
    "qoder":        "qodercli -p",
    "opencode":     "opencode run",
    "amp":          "amp --execute",
}

# Re-exported (under the older names) to minimise churn at call sites;
# the canonical strings now live in :mod:`cmind_cli._storage`.
_SOURCE_BUNDLE = _storage.CHANNEL_BUNDLE
_SOURCE_LEGACY = _storage.CHANNEL_LEGACY
_CONFIG_RELPATH = _storage.WORKSPACE_MARKER_RELPATH


def _current_cli_version() -> str:
    """Return the installed ``cmind-cli`` version, or ``"dev"`` on failure.

    Used to stamp ``.meta.toml`` with the version that last touched a
    given workspace.  Failures (editable install, missing METADATA,
    namespace package weirdness) are silently swallowed -- the version
    field is purely informational.
    """
    try:
        return importlib.metadata.version("cmind-cli")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _read_source_marker(project_path: Path) -> str | None:
    """Return the recorded provisioning channel for ``project_path``.

    Reads ``channel`` from ``~/.cmind/workspaces/<workspace-id>/.meta.toml``.
    Returns ``None`` when no meta file exists (fresh workspace) or the
    channel field is missing.
    """
    meta = _storage.read_meta(project_path)
    if meta is None:
        return None
    channel = meta.get("channel")
    if isinstance(channel, str) and channel:
        return channel
    return None


def _write_source_marker(project_path: Path, source: str) -> None:
    """Persist the provisioning channel in the home-side ``.meta.toml``.

    Replaces the legacy ``workspace/.cmind/.source`` text file with a
    structured TOML record under ``~/.cmind/workspaces/<workspace-id>/`` that
    also carries timestamps and the version of cmind-cli that last
    touched the workspace.  See :mod:`cmind_cli._storage` for the
    layout rationale.
    """
    _storage.write_meta(
        project_path,
        channel=source,
        cmind_cli_version=_current_cli_version(),
    )


def _write_workspace_config(project_path: Path, selected_ai: str) -> None:
    """Materialise ``.cmind/config.toml`` with the selected AI's CLI command.

    Idempotent: if the file already exists and already contains
    ``ai_cli_cmd``, leave it alone (the user may have customised it).
    Only writes a fresh file when one is missing.
    """
    cfg_path = project_path / _CONFIG_RELPATH
    cli_cmd = _AI_TO_CLI_CMD.get(selected_ai, selected_ai)

    if cfg_path.exists():
        # Don't clobber user edits.  We could merge here, but plain
        # workspaces don't need the complexity and a stale value is a
        # supported configuration (env var override remains available).
        return

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "# CoderMind workspace configuration\n"
        "# Managed by `cmind init` / `cmind update`.  Safe to commit.\n"
        "# See: https://github.com/microsoft/RPG-ZeroRepo (CoderMind/docs/configuration.md)\n"
        "\n"
        "[cmind]\n"
        f'ai_cli_cmd = "{cli_cmd}"\n',
        encoding="utf-8",
    )


def _detect_install_method() -> str:
    """Best-effort detection of how ``cmind-cli`` was installed.

    Returns one of ``"uv"``, ``"pipx"``, ``"pip-user"``, ``"pip-system"``,
    ``"editable"``, ``"unknown"``.  Used by ``cmind update`` to pick the
    right self-upgrade command.
    """
    try:
        # Do not call ``.resolve()`` here.  The python
        # interpreter inside a uv tool venv is typically a symlink to
        # the system python (``/usr/bin/python3.12`` on Linux); resolving
        # it discards the ``~/.local/share/uv/tools/cmind-cli/`` prefix
        # we depend on for installer detection.  We want the path *as*
        # the kernel saw it for ``sys.executable``, not the underlying
        # interpreter binary it points to.
        exe = Path(sys.executable)
        exe_str = str(exe)
    except Exception:
        return "unknown"

    exe_posix = exe_str.replace("\\", "/")

    # IMPORTANT: editable detection must run FIRST.  An editable install
    # placed inside a uv-managed venv would otherwise be reported as
    # "uv" and ``cmind update`` would try to upgrade from the
    # registry instead of leaving the local checkout alone.
    try:
        import importlib.metadata as _im

        dist = _im.distribution("cmind-cli")
        durl = dist.read_text("direct_url.json")
        if durl and '"editable": true' in durl:
            return "editable"
    except Exception:
        pass

    # uv tool install creates venvs under ~/.local/share/uv/tools/<name>/
    # (or %LOCALAPPDATA%\uv\tools\<name>\ on Windows).
    if "/uv/tools/" in exe_posix:
        return "uv"
    try:
        # uv writes a receipt file at the venv root one level above bin/.
        # Newer uv versions use ``uv-receipt.toml``; older releases used
        # ``uv-receipt.json``.  Check both so the heuristic stays robust
        # across the version most users have installed at any given time.
        receipt_parent = exe.parent.parent
        if (
            (receipt_parent / "uv-receipt.toml").exists()
            or (receipt_parent / "uv-receipt.json").exists()
        ):
            return "uv"
    except Exception:
        pass

    # pipx puts each tool's venv under ~/.local/share/pipx/venvs/<name>/
    if "/pipx/venvs/" in exe_posix:
        return "pipx"

    # Plain pip: distinguish user-site vs system-site by path prefix.
    try:
        import site

        if site.ENABLE_USER_SITE and exe_str.startswith(site.getuserbase()):
            return "pip-user"
    except Exception:
        pass

    return "pip-system"


def _upgrade_command(method: str) -> list[str] | None:
    """Return the shell command argv that upgrades the installed CLI.

    Returns ``None`` when no automatic command is appropriate (editable
    install, or unknown installer).
    """
    if method == "uv":
        return ["uv", "tool", "upgrade", "cmind-cli"]
    if method == "pipx":
        return ["pipx", "upgrade", "cmind-cli"]
    if method == "pip-user":
        return [sys.executable, "-m", "pip", "install", "-U", "--user", "cmind-cli"]
    if method == "pip-system":
        return [sys.executable, "-m", "pip", "install", "-U", "cmind-cli"]
    return None


def _install_source() -> str:
    """Identify *where* the installed ``cmind-cli`` came from.

    Used by the default-on auto-upgrade flow to skip dev-mode installs
    (local checkout, editable) that the user is actively iterating on —
    blindly running ``uv tool upgrade`` on those would either no-op
    (uv complains it's not a registry release) or, worse, replace the
    user's local working copy with the registry build.

    Returns:
        * ``"git"``       — installed from a ``git+https://...`` URL.
          Safe to auto-upgrade.
        * ``"pypi"``      — installed from a PyPI release (no
          ``direct_url.json`` recorded).  Safe to auto-upgrade.
        * ``"file"``      — installed from a local path
          (``uv tool install .``).  Skip auto-upgrade — the user is
          developing.
        * ``"editable"``  — installed with ``--editable``.  Skip.
        * ``"unknown"``   — couldn't determine source.  Skip (conservative).

    The detection reads PEP 610's ``direct_url.json`` from the
    installed distribution's metadata.  We never shell out to ``uv``
    or ``pip`` for this — the local metadata is the single source of
    truth and works in offline environments.
    """
    try:
        import importlib.metadata as _im
        dist = _im.distribution("cmind-cli")
        raw = dist.read_text("direct_url.json")
    except Exception:
        return "unknown"

    if raw is None:
        # No direct_url.json file recorded -> installed from a PyPI
        # release (PEP 610 mandates this file only for non-registry
        # installs).
        return "pypi"

    try:
        info = json.loads(raw)
    except Exception:
        return "unknown"

    # Editable installs always set ``dir_info.editable: true``.
    dir_info = info.get("dir_info") or {}
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        return "editable"

    url = info.get("url")
    if isinstance(url, str):
        if url.startswith("git+") or info.get("vcs_info"):
            return "git"
        if url.startswith("file://"):
            return "file"

    return "unknown"


#: Sources where auto-upgrade is safe to run by default in
#: ``cmind update``.  Matches the values returned by
#: :func:`_install_source`.
_AUTO_UPGRADE_SOURCES: frozenset[str] = frozenset({"git", "pypi"})


# ── Default .gitignore template ──────────────────────────────────────────
# Split into three parts so init can compose the right output depending on
# project state:
#   * PYTHON template  → written *only* when both .git/ and .gitignore are
#                         absent (greenfield), so we don't impose Python
#                         conventions on an existing repo that already has
#                         its own .gitignore preferences.
#   * CMIND_COMMON    → always injected; these files must be ignored
#                         (runtime data, machine-specific config).
#   * CMIND_AI[ai]    → always injected for the selected AI assistant.
#
# The Python template is a verbatim copy of GitHub's official
# ``github/gitignore/Python.gitignore`` (220-line community baseline).
# Keeping it byte-for-byte identical means:
#   * No "two sources of truth" against the canonical upstream — when
#     PEP 582 or a new packaging tool emerges, we just re-sync this block
#     instead of guessing which patterns matter.
#   * Users opening their .gitignore see a familiar, well-commented file
#     covering PyInstaller, Django, Flask, Jupyter, Celery, mypy/Pyre,
#     poetry/uv/pdm/pixi, Ruff, Marimo, Streamlit, etc.
#   * Source:  https://github.com/github/gitignore/blob/main/Python.gitignore
_GITIGNORE_PYTHON_TEMPLATE = """\
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#   Usually these files are written by a python script from a template
#   before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
*.lcov
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
# Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
# uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
# poetry.lock
# poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#   pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
#   https://pdm-project.org/en/latest/usage/project/#working-with-version-control
# pdm.lock
# pdm.toml
.pdm-python
.pdm-build/

# pixi
#   Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
# pixi.lock
#   Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
#   in the .venv directory. It is recommended not to include this directory in version control.
.pixi/*
!.pixi/config.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule*
celerybeat.pid

# Redis
*.rdb
*.aof
*.pid

# RabbitMQ
mnesia/
rabbitmq/
rabbitmq-data/

# ActiveMQ
activemq-data/

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#   JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#   be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#   and can be added to the global gitignore or merged into this file.  For a more nuclear
#   option (not recommended) you can uncomment the following to ignore the entire idea folder.
# .idea/

# Abstra
#   Abstra is an AI-powered process automation framework.
#   Ignore directories containing user credentials, local state, and settings.
#   Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#   Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore
#   that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#   and can be added to the global gitignore or merged into this file. However, if you prefer,
#   you could uncomment the following to ignore the entire vscode folder
# .vscode/
# Temporary file for partial code execution
tempCodeRunnerFile.py

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/

# Streamlit
.streamlit/secrets.toml
"""

_GITIGNORE_CMIND_HEADER = "# CoderMind ignores (managed by `cmind init/update`)"

_GITIGNORE_CMIND_COMMON = """\
# Runtime workspace (logs, generated data, trajectory)
# NOTE: ``.cmind/*`` (glob), not ``.cmind/`` (whole-dir).  Git does not
# descend into a directory ignored as a whole, so the ``!`` negation
# below would have no effect with the directory form.
.cmind/*
# but DO track the workspace AI config so collaborators see the same
# default — see docs/configuration.md
!.cmind/config.toml

# Legacy runtime dir from pre-cmind (rpgkit) workspaces — kept so users
# upgrading don't accidentally commit stale data while the old directory
# still exists alongside .cmind/.
.rpgkit/

# Codegen dev environments
.venv_dev/
.cmind_dev_env/

# Machine-specific config (absolute interpreter paths)
.vscode/mcp.json
.vscode/tasks.json
.mcp.json
"""

# AI-specific slash-command directories that CoderMind regenerates each time
# `cmind init/update` runs. Each entry covers only a sub-directory of
# the agent folder so unrelated assets in ``.github/`` (workflows,
# CODEOWNERS, …) or ``.claude/`` (settings.json with team-shared
# permissions) remain trackable.
_GITIGNORE_CMIND_AI = {
    "copilot": """\
# Copilot slash command definitions (regenerated by cmind)
.github/agents/
.github/prompts/
""",
    "claude": """\
# Claude Code slash command definitions (regenerated by cmind)
.claude/commands/
""",
}

BANNER = """
 ██████╗ ██████╗ ██████╗ ███████╗██████╗ ███╗   ███╗██╗███╗   ██╗██████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗
██║     ██║   ██║██║  ██║█████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║██║  ██║
██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██║  ██║
╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
"""

TAGLINE = (
    "CoderMind — Repository Planning Graphs for AI coding agents"
)


class StepTracker:
    """Track and render hierarchical steps without emojis, similar to Claude Code tree output.

    Supports live auto-refresh via an attached refresh callback.
    """

    def __init__(self, title: str):
        self.title = title
        self.steps = []  # list of dicts: {key, label, status, detail}
        self.status_order = {
            "pending": 0,
            "running": 1,
            "done": 2,
            "error": 3,
            "skipped": 4,
        }
        self._refresh_cb = None  # callable to trigger UI refresh

    def attach_refresh(self, cb):
        self._refresh_cb = cb

    def add(self, key: str, label: str):
        if key not in [s["key"] for s in self.steps]:
            self.steps.append(
                {"key": key, "label": label, "status": "pending", "detail": ""}
            )
            self._maybe_refresh()

    def start(self, key: str, detail: str = ""):
        self._update(key, status="running", detail=detail)

    def complete(self, key: str, detail: str = ""):
        self._update(key, status="done", detail=detail)

    def error(self, key: str, detail: str = ""):
        self._update(key, status="error", detail=detail)

    def skip(self, key: str, detail: str = ""):
        self._update(key, status="skipped", detail=detail)

    def _update(self, key: str, status: str, detail: str):
        for s in self.steps:
            if s["key"] == key:
                s["status"] = status
                if detail:
                    s["detail"] = detail
                self._maybe_refresh()
                return

        self.steps.append(
            {"key": key, "label": key, "status": status, "detail": detail}
        )
        self._maybe_refresh()

    def _maybe_refresh(self):
        if self._refresh_cb:
            try:
                self._refresh_cb()
            except Exception:
                pass

    def render(self):
        tree = Tree(f"[cyan]{self.title}[/cyan]", guide_style="grey50")
        for step in self.steps:
            label = step["label"]
            detail_text = step["detail"].strip() if step["detail"] else ""

            status = step["status"]
            if status == "done":
                symbol = "[green]●[/green]"
            elif status == "pending":
                symbol = "[green dim]○[/green dim]"
            elif status == "running":
                symbol = "[cyan]○[/cyan]"
            elif status == "error":
                symbol = "[red]●[/red]"
            elif status == "skipped":
                symbol = "[yellow]○[/yellow]"
            else:
                symbol = " "

            if status == "pending":
                # Entire line light gray (pending)
                if detail_text:
                    line = (
                        f"{symbol} [bright_black]{label} ({detail_text})[/bright_black]"
                    )
                else:
                    line = f"{symbol} [bright_black]{label}[/bright_black]"
            else:
                # Label white, detail (if any) light gray in parentheses
                if detail_text:
                    line = f"{symbol} [white]{label}[/white] [bright_black]({detail_text})[/bright_black]"
                else:
                    line = f"{symbol} [white]{label}[/white]"

            tree.add(line)
        return tree


def get_key():
    """Get a single keypress in a cross-platform way using readchar."""
    key = readchar.readkey()

    if key == readchar.key.UP or key == readchar.key.CTRL_P:
        return "up"
    if key == readchar.key.DOWN or key == readchar.key.CTRL_N:
        return "down"

    if key == readchar.key.ENTER:
        return "enter"

    if key == readchar.key.ESC:
        return "escape"

    if key == readchar.key.CTRL_C:
        raise KeyboardInterrupt

    return key


def select_with_arrows(
    options: dict, prompt_text: str = "Select an option", default_key: str = None
) -> str:
    """Interactive selection using arrow keys with Rich Live display.

    Args:
        options: Dict with keys as option keys and values as descriptions
        prompt_text: Text to show above the options
        default_key: Default option key to start with

    Returns:
        Selected option key
    """
    option_keys = list(options.keys())
    if default_key and default_key in option_keys:
        selected_index = option_keys.index(default_key)
    else:
        selected_index = 0

    selected_key = None

    def create_selection_panel():
        """Create the selection panel with current selection highlighted."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="left", width=3)
        table.add_column(style="white", justify="left")

        for i, key in enumerate(option_keys):
            if i == selected_index:
                table.add_row("▶", f"[cyan]{key}[/cyan] [dim]({options[key]})[/dim]")
            else:
                table.add_row(" ", f"[cyan]{key}[/cyan] [dim]({options[key]})[/dim]")

        table.add_row("", "")
        table.add_row(
            "", "[dim]Use ↑/↓ to navigate, Enter to select, Esc to cancel[/dim]"
        )

        return Panel(
            table,
            title=f"[bold]{prompt_text}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

    console.print()

    def run_selection_loop():
        nonlocal selected_key, selected_index
        with Live(
            create_selection_panel(),
            console=console,
            transient=True,
            auto_refresh=False,
        ) as live:
            while True:
                try:
                    key = get_key()
                    if key == "up":
                        selected_index = (selected_index - 1) % len(option_keys)
                    elif key == "down":
                        selected_index = (selected_index + 1) % len(option_keys)
                    elif key == "enter":
                        selected_key = option_keys[selected_index]
                        break
                    elif key == "escape":
                        console.print("\n[yellow]Selection cancelled[/yellow]")
                        raise typer.Exit(1)

                    live.update(create_selection_panel(), refresh=True)

                except KeyboardInterrupt:
                    console.print("\n[yellow]Selection cancelled[/yellow]")
                    raise typer.Exit(1)

    run_selection_loop()

    if selected_key is None:
        console.print("\n[red]Selection failed.[/red]")
        raise typer.Exit(1)

    return selected_key


console = Console()


class BannerGroup(TyperGroup):
    """Custom group that shows banner before help."""

    def format_help(self, ctx, formatter):
        # Show banner before help
        show_banner()
        super().format_help(ctx, formatter)


app = typer.Typer(
    name="cmind",
    help="Provision and manage Repository Planning Graph (RPG) workspaces for AI coding agents.",
    add_completion=False,
    invoke_without_command=True,
    cls=BannerGroup,
)


def show_banner():
    """Display the ASCII art banner."""
    banner_lines = BANNER.strip().split("\n")
    colors = ["bright_blue", "blue", "cyan", "bright_cyan", "white", "bright_white"]

    styled_banner = Text()
    for i, line in enumerate(banner_lines):
        color = colors[i % len(colors)]
        styled_banner.append(line + "\n", style=color)

    console.print(Align.center(styled_banner))
    console.print(Align.center(Text(TAGLINE, style="italic bright_yellow")))
    console.print()


@app.callback()
def callback(ctx: typer.Context):
    """Show banner when no subcommand is provided."""
    if (
        ctx.invoked_subcommand is None
        and "--help" not in sys.argv
        and "-h" not in sys.argv
    ):
        show_banner()
        console.print(
            Align.center("[dim]Run 'cmind --help' for usage information[/dim]")
        )
        console.print()


def run_command(
    cmd: list[str],
    check_return: bool = True,
    capture: bool = False,
    shell: bool = False,
) -> Optional[str]:
    """Run a shell command and optionally capture output."""
    try:
        if capture:
            result = subprocess.run(
                cmd, check=check_return, capture_output=True, text=True, shell=shell
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=check_return, shell=shell)
            return None
    except subprocess.CalledProcessError as e:
        if check_return:
            console.print(f"[red]Error running command:[/red] {' '.join(cmd)}")
            console.print(f"[red]Exit code:[/red] {e.returncode}")
            if hasattr(e, "stderr") and e.stderr:
                console.print(f"[red]Error output:[/red] {e.stderr}")
            raise
        return None


def check_tool(tool: str, tracker: StepTracker = None) -> bool:
    """Check if a tool is installed. Optionally update tracker.

    Args:
        tool: Name of the tool to check
        tracker: Optional StepTracker to update with results

    Returns:
        True if tool is found, False otherwise
    """
    # Special handling for Claude CLI after `claude migrate-installer`
    # See: https://github.com/github/spec-kit/issues/123
    # The migrate-installer command REMOVES the original executable from PATH
    # and creates an alias at ~/.claude/local/claude instead
    # This path should be prioritized over other claude executables in PATH
    if tool == "claude":
        if CLAUDE_LOCAL_PATH.exists() and CLAUDE_LOCAL_PATH.is_file():
            if tracker:
                tracker.complete(tool, "available")
            return True

    found = shutil.which(tool) is not None

    if tracker:
        if found:
            tracker.complete(tool, "available")
        else:
            tracker.error(tool, "not found")

    return found


def is_git_repo(path: Path = None) -> bool:
    """Check if the specified path is inside a git repository."""
    if path is None:
        path = Path.cwd()

    if not path.is_dir():
        return False

    try:
        # Use git command to check if inside a work tree
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            cwd=path,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _setup_gitignore(project_path: Path, selected_ai: str) -> None:
    """Materialize ``.gitignore`` with CoderMind's required rules.

    This is the single injection point for all CoderMind gitignore
    management.  Other init steps (``_generate_mcp_config``,
    ``_install_copilot_hooks``) must not modify ``.gitignore``
    themselves; all rules they used to inject have been folded into
    ``_GITIGNORE_CMIND_COMMON`` / ``_GITIGNORE_CMIND_AI``.

    Behavior:

    * **Greenfield** — both ``.git/`` and ``.gitignore`` are absent:
      write Python standard template + CoderMind common + AI-specific
      rules.  Gives new projects a complete, sensible default.

    * **Existing repo or existing ``.gitignore``** — do not overwrite
      the user's Python conventions.  Only append CoderMind rules
      (deduplicated by exact line match) under a single
      ``# CoderMind ignores`` header.

    Args:
        project_path: Project root that may or may not be a git repo.
        selected_ai:  ``"copilot"`` or ``"claude"`` — selects which AI
                      slash-command directories to ignore.
    """
    gitignore = project_path / ".gitignore"
    git_dir = project_path / ".git"

    cmind_block = _GITIGNORE_CMIND_COMMON
    ai_rules = _GITIGNORE_CMIND_AI.get(selected_ai)
    if ai_rules:
        cmind_block += "\n" + ai_rules

    # Greenfield: brand-new project, no git, no existing .gitignore.
    # Lay down the full template (Python conventions + CoderMind rules).
    if not git_dir.exists() and not gitignore.exists():
        gitignore.write_text(
            _GITIGNORE_PYTHON_TEMPLATE
            + "\n"
            + _GITIGNORE_CMIND_HEADER
            + "\n"
            + cmind_block,
            encoding="utf-8",
        )
        return

    # Brownfield: respect the user's existing setup, only ensure CoderMind
    # rules are present.  Parse existing entries (strip whitespace, drop
    # comments and leading ``/``) so we can compare line-by-line.
    if gitignore.exists():
        existing_text = gitignore.read_text(encoding="utf-8")
        existing_lines = existing_text.splitlines()
    else:
        existing_text = ""
        existing_lines = []

    def _norm(line: str) -> str:
        return line.strip().lstrip("/")

    existing_norm = {
        _norm(line)
        for line in existing_lines
        if line.strip() and not line.strip().startswith("#")
    }

    # Collect CoderMind pattern lines (skip comments and blanks in the
    # block — comments are kept for the appended section but not used
    # for dedup checks).
    missing_lines: list[str] = []
    for line in cmind_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _norm(stripped) not in existing_norm:
            missing_lines.append(stripped)

    if not missing_lines:
        return

    # Append under a single, idempotent CoderMind header so repeated runs
    # don't create duplicate section markers.
    parts: list[str] = []
    if existing_text and not existing_text.endswith("\n"):
        parts.append("\n")
    if existing_text:
        parts.append("\n")
    if _GITIGNORE_CMIND_HEADER not in existing_text:
        parts.append(_GITIGNORE_CMIND_HEADER + "\n")
    parts.extend(line + "\n" for line in missing_lines)

    with open(gitignore, "a", encoding="utf-8") as f:
        f.write("".join(parts))


def init_git_repo(
    project_path: Path, quiet: bool = False
) -> Tuple[bool, Optional[str]]:
    """Initialize a fresh git repository and create the first commit.

    Caller MUST guarantee ``project_path/.git`` does not already exist —
    this function unconditionally runs ``git init`` and a first
    ``git commit``, which would otherwise pollute an existing user
    history.  The check lives in ``init()`` via :func:`is_git_repo`.

    ``.gitignore`` is assumed to have been written by
    :func:`_setup_gitignore` earlier in the init flow.

    Args:
        project_path: Path to initialize git repository in
        quiet: if True suppress console output (tracker handles status)

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        original_cwd = Path.cwd()
        os.chdir(project_path)
        if not quiet:
            console.print("[cyan]Initializing git repository...[/cyan]")
        subprocess.run(["git", "init"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "add", "."], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit from CoderMind template"],
            check=True,
            capture_output=True,
            text=True,
        )
        if not quiet:
            console.print("[green]✓[/green] Git repository initialized")
        return True, None

    except subprocess.CalledProcessError as e:
        error_msg = f"Command: {' '.join(e.cmd)}\nExit code: {e.returncode}"
        if e.stderr:
            error_msg += f"\nError: {e.stderr.strip()}"
        elif e.stdout:
            error_msg += f"\nOutput: {e.stdout.strip()}"

        if not quiet:
            console.print(f"[red]Error initializing git repository:[/red] {e}")
        return False, error_msg
    finally:
        os.chdir(original_cwd)


def handle_vscode_settings(
    sub_item, dest_file, rel_path, verbose=False, tracker=None
) -> None:
    """Handle merging or copying of .vscode/settings.json files."""

    def log(message, color="green"):
        if verbose and not tracker:
            console.print(f"[{color}]{message}[/] {rel_path}")

    try:
        with open(sub_item, "r", encoding="utf-8") as f:
            new_settings = json.load(f)

        if dest_file.exists():
            merged = merge_json_files(
                dest_file, new_settings, verbose=verbose and not tracker
            )
            with open(dest_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=4)
                f.write("\n")
            log("Merged:", "green")
        else:
            shutil.copy2(sub_item, dest_file)
            log("Copied (no existing settings.json):", "blue")

    except Exception as e:
        log(f"Warning: Could not merge, copying instead: {e}", "yellow")
        shutil.copy2(sub_item, dest_file)


def merge_json_files(
    existing_path: Path, new_content: dict, verbose: bool = False
) -> dict:
    """Merge new JSON content into existing JSON file.

    Performs a deep merge where:
    - New keys are added
    - Existing keys are preserved unless overwritten by new content
    - Nested dictionaries are merged recursively
    - Lists and other values are replaced (not merged)

    Args:
        existing_path: Path to existing JSON file
        new_content: New JSON content to merge in
        verbose: Whether to print merge details

    Returns:
        Merged JSON content as dict
    """
    try:
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_content = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, just use new content
        return new_content

    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge update dict into base dict."""
        result = base.copy()
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = deep_merge(result[key], value)
            else:
                # Add new key or replace existing value
                result[key] = value
        return result

    merged = deep_merge(existing_content, new_content)

    if verbose:
        console.print(f"[cyan]Merged JSON file:[/cyan] {existing_path.name}")

    return merged


def _load_json_dict(path: Path) -> dict:
    """Read a JSON object from ``path``; return ``{}`` on any error or non-object content.

    Used when merging into existing AI-assistant config files: we never want
    a malformed or unexpected JSON shape to crash the init/update flow.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _cleanup_legacy_vscode_mcp(project_path: Path) -> None:
    """Remove a stale ``mcp.servers.rpg-tools`` entry from ``.vscode/settings.json``.

    Earlier versions of ``cmind init`` registered the MCP server inside
    ``settings.json``.  We've since moved to ``.vscode/mcp.json``; this
    helper deletes only the stale entry so users upgrading via
    ``cmind update`` don't end up with two registrations.

    Other settings — and any non-rpg-tools MCP servers the user may have
    added — are preserved untouched.
    """
    settings_file = project_path / ".vscode" / "settings.json"
    settings = _load_json_dict(settings_file)
    if not settings:
        return

    mcp = settings.get("mcp")
    if not isinstance(mcp, dict):
        return
    servers = mcp.get("servers")
    if not isinstance(servers, dict) or "rpg-tools" not in servers:
        return

    del servers["rpg-tools"]
    if not servers:
        del mcp["servers"]
    if not mcp:
        del settings["mcp"]

    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
            f.write("\n")
    except OSError:
        pass


def _generate_mcp_config(
    project_path: Path,
    selected_ai: str,
    tracker=None,
) -> None:
    """Generate MCP server configuration for the selected AI assistant.

    Both Claude and VS Code Copilot launch the MCP server via the
    ``cmind-mcp`` console script installed alongside ``cmind-cli``.
    This keeps the config portable across machines (no absolute paths
    to a workspace-local copy) and ensures the server always runs
    against the bundled scripts that match the installed CLI version.

    - Claude:  ``.mcp.json``         (key ``mcpServers.rpg-tools``)
    - Copilot: ``.vscode/mcp.json``  (key ``servers.rpg-tools``,
      VS Code 1.102+ standard layout)

    The ``cmind-mcp`` command must be on ``PATH``.  ``cmind init``
    emits a warning at the end of the run when it isn't, so MCP
    clients fail with a clear cause rather than the opaque
    ``Connection closed`` error.
    """
    project_path = project_path.resolve()

    mcp_server_config = {
        "command": "cmind-mcp",
        "args": [],
    }

    try:
        if selected_ai == "claude":
            # Claude Code uses .mcp.json at project root
            mcp_file = project_path / ".mcp.json"
            mcp_data = _load_json_dict(mcp_file)
            mcp_data.setdefault("mcpServers", {})
            mcp_data["mcpServers"]["rpg-tools"] = mcp_server_config
            with open(mcp_file, "w", encoding="utf-8") as f:
                json.dump(mcp_data, f, indent=2)
                f.write("\n")

        elif selected_ai == "copilot":
            # VS Code Copilot (1.102+): .vscode/mcp.json with top-level "servers".
            # No ``sandbox`` block: VS Code's MCP sandbox requires bwrap +
            # socat which are absent on most Linux desktops, WSL, minimal
            # Docker images, and fresh macOS installs, causing the server
            # to crash with "Connection closed".  Tool auto-approval is
            # handled by VS Code's "Always allow this server" setting.
            vscode_dir = project_path / ".vscode"
            vscode_dir.mkdir(parents=True, exist_ok=True)
            mcp_file = vscode_dir / "mcp.json"
            mcp_data = _load_json_dict(mcp_file)
            mcp_data.setdefault("servers", {})
            mcp_data["servers"]["rpg-tools"] = mcp_server_config
            with open(mcp_file, "w", encoding="utf-8") as f:
                json.dump(mcp_data, f, indent=2)
                f.write("\n")
            # Migration: drop a stale rpg-tools entry from .vscode/settings.json
            # (older versions registered MCP there).
            _cleanup_legacy_vscode_mcp(project_path)

        else:
            # For other/future agents, fall back to .mcp.json (Claude format)
            mcp_file = project_path / ".mcp.json"
            mcp_data = _load_json_dict(mcp_file)
            mcp_data.setdefault("mcpServers", {})
            mcp_data["mcpServers"]["rpg-tools"] = mcp_server_config
            with open(mcp_file, "w", encoding="utf-8") as f:
                json.dump(mcp_data, f, indent=2)
                f.write("\n")

        if tracker:
            tracker.complete("mcp", f"configured for {selected_ai}")
    except Exception as e:
        if tracker:
            tracker.error("mcp", f"failed: {e}")
        else:
            console.print(f"[yellow]Warning: Could not generate MCP config: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Copilot CLI: global MCP registration
# ---------------------------------------------------------------------------

_COPILOT_CLI_MCP_CONFIG = Path.home() / ".copilot" / "mcp-config.json"


def _register_copilot_cli_global_mcp(tracker=None) -> None:
    """Register ``rpg-tools`` in ``~/.copilot/mcp-config.json`` (global).

    The GitHub Copilot CLI (``copilot``) — unlike the VS Code Copilot
    extension — does NOT read workspace-local ``.vscode/mcp.json``.  It
    only reads the global ``~/.copilot/mcp-config.json`` (or accepts
    inline JSON via ``--additional-mcp-config``).

    To make ``copilot`` find ``rpg-tools`` automatically in any
    cmind-initialised workspace, we register the server globally on
    first ``cmind init --ai copilot`` (or ``cmind update``).

    This is safe because ``cmind-mcp`` is cwd-aware (it walks up to
    find ``rpg.json``) and stateless across workspaces — one global
    registration serves every workspace the user ``cd``-s into.  In
    workspaces without ``rpg.json`` the server starts in degraded mode
    and tool calls return a ``rpg_unavailable`` hint instructing the
    user to run ``/cmind.encode``.

    Safety rules (see audit decisions D-globalmcp-1..4):
      - **No-op when in-sync.**  If the file already contains exactly
        the entry we'd write, we don't touch it at all (no mtime bump,
        no .bak).  This makes ``cmind update`` cheap to run repeatedly.
      - **Refuse to wipe a malformed config.**  If the file exists but
        isn't valid JSON we abort with a clear error instead of
        overwriting; the user is expected to fix it (or run with
        ``--no-copilot-cli-mcp``).  Without this guard a stray comma
        in the user's config would have us silently drop every
        non-rpg-tools server.
      - **Atomic write.**  We serialise to ``mcp-config.json.tmp``
        first and then ``os.replace()`` into place, so a Ctrl-C or
        crash mid-write can't leave the file half-written.
      - **Respect user-customised entries.**  If an existing
        ``rpg-tools`` entry uses a different ``command`` (the user has
        intentionally pointed it elsewhere, e.g. to a dev checkout) we
        leave it alone and ask them to use ``--no-copilot-cli-mcp``.
      - **One-shot .bak.**  Only created on the first modification we
        actually perform — never on no-op runs, never overwritten.
    """
    config_path = _COPILOT_CLI_MCP_CONFIG
    bak_path = config_path.with_suffix(".json.bak")
    tmp_path = config_path.with_suffix(".json.tmp")
    desired = {
        "type": "stdio",
        "command": "cmind-mcp",
        "args": [],
    }

    def _report_skip(detail: str) -> None:
        if tracker:
            tracker.skip("copilot-cli-mcp", detail)

    def _report_error(detail: str) -> None:
        if tracker:
            tracker.error("copilot-cli-mcp", detail)
        else:
            console.print(
                f"[yellow]Warning: could not register rpg-tools in "
                f"{config_path}: {detail}[/yellow]"
            )

    def _report_done(detail: str) -> None:
        if tracker:
            tracker.complete("copilot-cli-mcp", detail)

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # ----- Parse existing file (strictly, so we can refuse to
        # clobber a malformed user config).  An empty/missing file is
        # fine — we treat that as "start fresh".
        if config_path.exists():
            raw = config_path.read_text(encoding="utf-8")
            if raw.strip() == "":
                existing: Dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    _report_error(
                        f"{config_path} is not valid JSON ({exc.msg} "
                        f"at line {exc.lineno} col {exc.colno}); refusing to "
                        f"overwrite. Fix the file or re-run with "
                        f"--no-copilot-cli-mcp."
                    )
                    return
                if not isinstance(parsed, dict):
                    _report_error(
                        f"{config_path} top-level is not a JSON object; "
                        f"refusing to overwrite. Re-run with "
                        f"--no-copilot-cli-mcp."
                    )
                    return
                existing = parsed
        else:
            existing = {}

        servers = existing.get("mcpServers")
        if servers is None:
            existing["mcpServers"] = {}
            servers = existing["mcpServers"]
        elif not isinstance(servers, dict):
            _report_error(
                f"{config_path}: `mcpServers` is not a JSON object; "
                f"refusing to overwrite. Re-run with --no-copilot-cli-mcp."
            )
            return

        current = servers.get("rpg-tools")
        # No-op fast path: file already contains exactly what we'd write.
        if current == desired:
            _report_skip(f"already up-to-date at {config_path}")
            return

        # Respect a user-customised entry — only touch entries that
        # either don't exist or already point at our `cmind-mcp`
        # console script (the latter happens on a version bump where
        # we'd want to e.g. add new default args).
        if (
            isinstance(current, dict)
            and current.get("command")
            and current.get("command") != "cmind-mcp"
        ):
            _report_skip(
                f"existing entry uses custom command "
                f"{current.get('command')!r}; leaving alone "
                f"(use --no-copilot-cli-mcp to silence)"
            )
            return

        # We're going to write — back up the original (one-shot).
        if config_path.exists() and not bak_path.exists():
            try:
                shutil.copy2(config_path, bak_path)
            except OSError:
                pass  # backup is best-effort

        servers["rpg-tools"] = desired

        # Atomic write: serialise to .tmp then rename.  ``os.replace``
        # is atomic on POSIX and Windows.
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, config_path)
        except Exception:
            # Clean up a stray .tmp on failure so the next run isn't
            # confused by a leftover.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

        action = "updated" if current is not None else "registered"
        _report_done(f"{action} at {config_path}")
    except Exception as exc:
        _report_error(f"failed: {exc}")


# ---------------------------------------------------------------------------
# Optional initial encode
# ---------------------------------------------------------------------------

def _workspace_has_python_code(project_path: Path) -> bool:
    """Return True if the workspace contains any ``*.py`` file outside ``.cmind/``.

    Used to decide whether ``cmind init`` should offer to build the RPG
    immediately.  Greenfield workspaces (or repos that don't ship Python
    code) skip the prompt because the encoder would produce an empty
    graph and waste LLM tokens.

    The walk prunes the ``.cmind`` directory in-place so workspace
    runtime state (``data/``, ``logs/``) doesn't influence the detection.
    Common boilerplate dirs (``.git``, ``.venv``, ``node_modules``,
    ``__pycache__``) are pruned too — a ``*.py`` under any of them
    would not indicate user code.
    """
    PRUNE = {".cmind", ".git", ".venv", "venv", "node_modules",
             "__pycache__", ".tox", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(project_path):
        # In-place mutation so os.walk doesn't descend into pruned dirs.
        dirnames[:] = [d for d in dirnames if d not in PRUNE]
        for name in filenames:
            if name.endswith(".py"):
                return True
    return False


# Regexes used to extract progress markers from the encoder's stderr.
# The encoder logs through Python ``logging`` with a fixed format;
# we don't depend on internals (a missing match just leaves the spinner
# in its previous state), so these are best-effort and fail-soft.
_ENCODE_RE_REPO_ITER = re.compile(r"LLM call for repo info, iter=(\d+)")
_ENCODE_RE_EXCLUDE_VOTE = re.compile(r"LLM vote #(\d+)")
_ENCODE_RE_TOTAL_FILES = re.compile(r"Total valid source files to parse:\s*(\d+)")
_ENCODE_RE_CLASS_BATCHES = re.compile(r"\[GLOBAL\] kind=class,\s*groups=\d+,\s*batches=(\d+)")
_ENCODE_RE_FUNC_BATCHES = re.compile(r"\[GLOBAL\] kind=function,\s*groups=\d+,\s*batches=(\d+)")
_ENCODE_RE_CLASS_PROCESS = re.compile(r"\[GLOBAL\] process_class_batch:")
_ENCODE_RE_FUNC_PROCESS = re.compile(r"\[GLOBAL\] process_func_batch:")
_ENCODE_RE_CLASS_FINISHED = re.compile(r"\[GLOBAL\] finished class batch with \d+ units")
_ENCODE_RE_FUNC_FINISHED = re.compile(r"\[GLOBAL\] finished function batch with \d+ units")
_ENCODE_RE_FILE_REMAP = re.compile(r"\[GLOBAL\] file=")
_ENCODE_RE_SUMMARY_BATCHES = re.compile(r"\[SUMMARY\] total files=(\d+),\s*batches=(\d+)")
_ENCODE_RE_SUMMARY_PROCESS = re.compile(r"\[SUMMARY\] processing batch with (\d+) files")
_ENCODE_RE_SUMMARY_FINISHED = re.compile(r"\[SUMMARY\] finished batch with (\d+) files")


def _parse_encoder_line(line: str, state: Dict[str, Any]) -> None:
    """Mutate ``state`` based on a single line of encoder stderr.

    Recognised phase markers (in roughly chronological order):
      * ``Skeleton loaded`` → setup done
      * ``Generating repo info`` / ``LLM call for repo info, iter=N``
      * ``Computing exclude list`` / ``LLM vote #N``
      * ``Excluded paths decided`` / ``Parsing features`` /
        ``Total valid source files to parse: N``
      * ``[GLOBAL] kind=class, ..., batches=N``  → class batch total
      * ``[GLOBAL] finished class batch``        → +1 class batch done
      * ``[GLOBAL] kind=function, ..., batches=N`` → function batch total
      * ``[GLOBAL] finished function batch``     → +1 function batch done
      * ``[GLOBAL] file=...``                    → feature-to-file mapping
      * ``[SUMMARY] total files=N, batches=M``   → file summary batch total
      * ``[SUMMARY] processing batch with N files`` / ``finished batch``
      * ``Refactoring to RPG`` / ``RPG refactoring done``
    """
    if "Skeleton loaded" in line:
        state["phase"] = "Skeleton loaded"
        return
    m = _ENCODE_RE_REPO_ITER.search(line)
    if m:
        state["phase"] = f"Repository overview — LLM iter {m.group(1)}"
        return
    if "Generating repo info" in line:
        state["phase"] = "Generating repository overview"
        return
    m = _ENCODE_RE_EXCLUDE_VOTE.search(line)
    if m:
        state["phase"] = f"Selecting files to exclude — vote #{m.group(1)}"
        return
    if "Excluding irrelevant files" in line:
        state["phase"] = "Selecting files to exclude"
        return
    if "Excluded paths decided" in line:
        state["phase"] = "Exclude list finalised"
        return
    m = _ENCODE_RE_TOTAL_FILES.search(line)
    if m:
        state["total_files"] = int(m.group(1))
        state["phase"] = f"Parsing features ({m.group(1)} files)"
        return
    if "Parsing features" in line:
        state["phase"] = "Parsing features"
        return
    m = _ENCODE_RE_CLASS_BATCHES.search(line)
    if m:
        state["class_total"] = int(m.group(1))
        state["kind"] = "class"
        state["phase"] = "Parsing class batches"
        return
    m = _ENCODE_RE_FUNC_BATCHES.search(line)
    if m:
        state["func_total"] = int(m.group(1))
        state["kind"] = "function"
        state["phase"] = "Parsing function batches"
        return
    if _ENCODE_RE_CLASS_PROCESS.search(line):
        state["class_done"] += 1
        if state.get("class_total"):
            state["class_done"] = min(state["class_done"], state["class_total"])
        state["_class_counted_on_process"] = True
        state["kind"] = "class"
        state["phase"] = "Parsing class batches"
        return
    if _ENCODE_RE_CLASS_FINISHED.search(line):
        if not state.get("_class_counted_on_process"):
            state["class_done"] += 1
        if state.get("class_total"):
            state["class_done"] = min(state["class_done"], state["class_total"])
        state["kind"] = "class"
        state["phase"] = "Parsing class batches"
        return
    if _ENCODE_RE_FUNC_PROCESS.search(line):
        state["func_done"] += 1
        if state.get("func_total"):
            state["func_done"] = min(state["func_done"], state["func_total"])
        state["_func_counted_on_process"] = True
        state["kind"] = "function"
        state["phase"] = "Parsing function batches"
        return
    if _ENCODE_RE_FUNC_FINISHED.search(line):
        if not state.get("_func_counted_on_process"):
            state["func_done"] += 1
        if state.get("func_total"):
            state["func_done"] = min(state["func_done"], state["func_total"])
        state["kind"] = "function"
        state["phase"] = "Parsing function batches"
        return
    if _ENCODE_RE_FILE_REMAP.search(line):
        if state.get("class_total"):
            state["class_done"] = max(state["class_done"], state["class_total"])
        if state.get("func_total"):
            state["func_done"] = max(state["func_done"], state["func_total"])
        state["kind"] = None
        state["phase"] = "Mapping features to files"
        return
    m = _ENCODE_RE_SUMMARY_BATCHES.search(line)
    if m:
        state["summary_total_files"] = int(m.group(1))
        state["summary_total"] = int(m.group(2))
        state["summary_done"] = 0
        state["kind"] = "summary"
        state["phase"] = "Summarizing file batches"
        return
    m = _ENCODE_RE_SUMMARY_PROCESS.search(line)
    if m:
        state["summary_current_files"] = int(m.group(1))
        state["kind"] = "summary"
        state["phase"] = f"Processing summary batch with {m.group(1)} files"
        return
    m = _ENCODE_RE_SUMMARY_FINISHED.search(line)
    if m:
        state["summary_done"] += 1
        state["summary_current_files"] = int(m.group(1))
        state["kind"] = "summary"
        state["phase"] = f"Finished summary batch with {m.group(1)} files"
        return
    if "Refactoring to RPG" in line:
        if state.get("summary_total"):
            state["summary_done"] = max(state["summary_done"], state["summary_total"])
        state["phase"] = "Refactoring to RPG"
        state["kind"] = None
        return
    if "RPG refactoring done" in line:
        state["phase"] = "Finalising"
        state["kind"] = None
        return


def _run_initial_encode(project_path: Path) -> bool:
    """Run the encoder in a subprocess, showing a Rich progress UI.

    The encoder logs verbose progress through Python ``logging`` to
    stderr and writes its final JSON summary to stdout.  Streaming those
    logs directly to the terminal looks alarming for end users (hundreds
    of lines of ``RPGParser - INFO - ...``), so instead we:

      * Capture stderr in a reader thread and write it verbatim to
        ``~/.cmind/workspaces/<workspace-id>/logs/encode.log`` — power users
        can ``tail -f`` it for the full firehose.
      * Parse a handful of phase markers off each line to drive a
        :class:`rich.progress.Progress` bar with a spinner + current
        phase + (when known) an M/N batch counter.
      * Capture stdout and surface the encoder's JSON summary on
        failure so the user has something concrete to debug.

    Returns True on success (exit code 0), False otherwise.  Never
    raises: ``cmind init`` itself has already succeeded by the time we
    get here and we don't want a flaky LLM call to make the whole
    command look like it failed.
    """
    encoder = project_path / ".cmind" / "scripts" / "rpg_encoder" / "run_encode.py"
    if not encoder.is_file():
        # Scripts live inside the installed wheel under
        # ``cmind_cli/core_pack/scripts/``.  Resolve the encoder
        # from there so the optional initial-encode kickoff works after
        # ``cmind init`` — which no longer copies scripts into the
        # workspace.
        from . import _assets
        candidate = _assets.scripts_dir() / "rpg_encoder" / "run_encode.py"
        if candidate.is_file():
            encoder = candidate
        else:
            console.print(
                f"[yellow]Encoder script not found at {candidate}; "
                f"run [cyan]/cmind.encode[/] in your AI agent later.[/yellow]"
            )
            return False

    # Keep all generated artefacts (logs/data/inner-git) in the
    # per-workspace home dir under ~/.cmind/workspaces/<workspace-id>/.  The
    # workspace tree should stay clean — no .cmind/logs/ written here.
    from . import _storage
    log_dir = _storage.workspace_logs_dir(project_path)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(f"[yellow]Could not create log directory {log_dir}: {exc}[/yellow]")
        return False
    log_path = log_dir / "encode.log"

    console.print()
    console.print(
        Panel(
            "[cyan]Running the encoder now…[/]\n\n"
            "Building [cyan]rpg.json[/] from your code via the LLM.  "
            "Verbose logs stream to [cyan]" + str(log_path) + "[/] — "
            "`tail -f` it in another terminal for the gory details.  "
            "Press Ctrl-C to abort; re-run later with [cyan]/cmind.encode[/].",
            title="[bold]Initial encode[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    state: Dict[str, Any] = {
        "phase": "Starting encoder…",
        "kind": None,
        "class_total": 0,
        "class_done": 0,
        "func_total": 0,
        "func_done": 0,
        "summary_total": 0,
        "summary_done": 0,
        "summary_total_files": 0,
        "summary_current_files": 0,
        "total_files": 0,
    }

    try:
        log_fp = open(log_path, "w", encoding="utf-8")
    except OSError as exc:
        console.print(f"[yellow]Could not open log file {log_path}: {exc}[/yellow]")
        return False

    # Force UTF-8 stdio in the encoder subprocess — see the matching
    # comment in the `script()` command for why this is required on
    # Windows (non-tty stdout/stderr fall back to a legacy code page).
    encoder_env = os.environ.copy()
    encoder_env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    encoder_env.setdefault("PYTHONUTF8", "1")

    try:
        proc = subprocess.Popen(
            [sys.executable, str(encoder), "--json"],
            cwd=str(project_path),
            env=encoder_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered so the reader thread sees lines promptly
        )
    except Exception as exc:  # noqa: BLE001
        log_fp.close()
        console.print(f"[yellow]Encoder failed to start: {exc}[/yellow]")
        return False

    def _stderr_reader() -> None:
        try:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, ""):
                if not raw:
                    break
                try:
                    log_fp.write(raw)
                    log_fp.flush()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    _parse_encoder_line(raw, state)
                except Exception:  # noqa: BLE001
                    # Progress parsing is best-effort; never let it kill the encoder.
                    pass
        except Exception:  # noqa: BLE001
            pass

    reader = threading.Thread(target=_stderr_reader, daemon=True)
    reader.start()

    stdout_chunks: List[str] = []
    interrupted = False

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task(state["phase"], total=None)

    try:
        with progress:
            while True:
                kind = state["kind"]
                if kind == "class" and state["class_total"]:
                    progress.update(
                        task_id,
                        description=state["phase"],
                        total=state["class_total"],
                        completed=state["class_done"],
                    )
                elif kind == "function" and state["func_total"]:
                    progress.update(
                        task_id,
                        description=state["phase"],
                        total=state["func_total"],
                        completed=state["func_done"],
                    )
                elif kind == "summary" and state["summary_total"]:
                    progress.update(
                        task_id,
                        description=state["phase"],
                        total=state["summary_total"],
                        completed=state["summary_done"],
                    )
                else:
                    # Indeterminate phase (e.g. "Refactoring to RPG",
                    # "Finalising").  Update the description, but also
                    # unfreeze the task whenever the previous determinate
                    # phase ended with ``completed == total`` — Rich
                    # sets ``task.finished_time`` at that point, and
                    # ``TimeElapsedColumn`` then renders the frozen
                    # ``finished_time`` instead of the live ``elapsed``,
                    # so the timer appears stuck.  We have to mutate the
                    # Task directly because ``Progress.update`` provides
                    # no public way to clear ``finished_time`` and
                    # ``update(total=None)`` is a no-op (None means
                    # "leave unchanged").
                    progress.update(task_id, description=state["phase"])
                    if progress.tasks:
                        t = progress.tasks[0]
                        if t.finished_time is not None:
                            t.total = None
                            t.completed = 0
                            t.finished_time = None
                            t.finished_speed = None

                if proc.poll() is not None:
                    break
                time.sleep(0.2)

            # Process exited — drain remaining stdout (JSON summary)
            # and wait for the reader to consume any trailing stderr
            # lines still buffered in the pipe, so the final progress
            # frame reflects the *complete* phase state.
            try:
                if proc.stdout is not None:
                    stdout_chunks.append(proc.stdout.read())
            except Exception:  # noqa: BLE001
                pass
            reader.join(timeout=2)
            # Final frame: show the *latest* batch state we know about,
            # not whatever the previous polling iteration captured.  If
            # the encoder zipped through function batches between two
            # 0.2-second polls and is now in "Finalising", we still want
            # the bar to read "3/3" rather than "1/3".
            if state["summary_total"]:
                progress.update(
                    task_id,
                    description=state["phase"],
                    total=state["summary_total"],
                    completed=state["summary_done"],
                )
            elif state["func_total"]:
                progress.update(
                    task_id,
                    description=state["phase"],
                    total=state["func_total"],
                    completed=state["func_done"],
                )
            elif state["class_total"]:
                progress.update(
                    task_id,
                    description=state["phase"],
                    total=state["class_total"],
                    completed=state["class_done"],
                )
            else:
                progress.update(task_id, description=state["phase"])
    except KeyboardInterrupt:
        interrupted = True
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    finally:
        reader.join(timeout=2)
        try:
            log_fp.close()
        except Exception:  # noqa: BLE001
            pass

    if interrupted:
        console.print(
            "\n[yellow]Encoder interrupted. Re-run later with "
            "[cyan]/cmind.encode[/].[/yellow]"
        )
        return False

    if proc.returncode == 0:
        console.print()
        console.print(
            Panel(
                "[green]Encoder finished successfully.[/]\n\n"
                "The RPG graph is now available under your home-dir "
                "workspace store ([cyan]rpg.json[/]).  The post-commit hook will "
                "keep it in sync on every commit; the MCP tools "
                "([cyan]search_rpg[/], [cyan]explore_rpg[/], …) are now usable.",
                title="[bold green]Encode complete[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return True

    # Surface the encoder's JSON summary (if any) so the failure isn't opaque.
    summary_blurb = ""
    stdout_text = "".join(stdout_chunks).strip()
    if stdout_text:
        # Keep it short — full details are in encode.log.
        snippet = stdout_text if len(stdout_text) <= 600 else stdout_text[:600] + "…"
        summary_blurb = f"\n\n[dim]Encoder output:[/]\n{snippet}"

    console.print()
    console.print(
        Panel(
            f"[red]Encoder exited with code {proc.returncode}.[/]\n\n"
            f"Check [cyan]{log_path}[/] for the full log.  You can retry "
            "with [cyan]/cmind.encode[/] after fixing the issue."
            f"{summary_blurb}",
            title="[bold red]Encode failed[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )
    return False


def _maybe_offer_initial_encode(
    project_path: Path,
    *,
    encode_choice: Optional[bool],
) -> None:
    """Optionally prompt the user, then run the encoder.

    Decision tree:

    * ``encode_choice == True``   → run unconditionally (``--encode``).
    * ``encode_choice == False``  → skip silently (``--no-encode``).
    * ``encode_choice is None``   → interactive: only ask when stdin is
      a TTY, the workspace contains user Python code, and rpg.json
      hasn't been built before.  Defaults to "No" so an accidental
      Enter doesn't kick off a long LLM job.

    Failures never propagate — ``cmind init`` is already done and we
    don't want a flaky encoder to taint the exit code.
    """
    # Already encoded: nothing to do.  rpg.json lives in the home-side
    # workspace store (``~/.cmind/workspaces/<id>/data/rpg.json``), not
    # in the workspace-local ``.cmind/data/`` — use the storage helper
    # so this check matches where the encoder actually writes.
    try:
        rpg_file = _storage.workspace_data_dir(project_path) / "rpg.json"
    except Exception:
        # Fallback for environments where storage resolution fails;
        # err on the side of running the encoder rather than skipping it.
        rpg_file = project_path / ".cmind" / "data" / "rpg.json"
    legacy_rpg_file = project_path / ".cmind" / "data" / "rpg.json"
    if rpg_file.exists() or legacy_rpg_file.exists():
        return

    if encode_choice is False:
        return

    if encode_choice is None:
        # Don't ask in non-interactive contexts (CI, piped stdin).
        if not sys.stdin.isatty():
            return
        # Don't ask when there's nothing to encode.
        if not _workspace_has_python_code(project_path):
            return
        console.print()
        console.print(
            Panel(
                "CoderMind can build the initial graph for this repo now by "
                "running the encoder against your existing code.  This is "
                "what the [cyan]/cmind.encode[/] slash command does — kicking "
                "it off here saves you a step.\n\n"
                "[yellow]Heads up:[/] the encoder calls an LLM and can take "
                "a few minutes on a real-sized repo.  You can always say "
                "No and run [cyan]/cmind.encode[/] in your AI agent later.",
                title="[bold]Build the RPG now?[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        try:
            answer = typer.confirm("Run the encoder now?", default=False)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not answer:
            return

    _run_initial_encode(project_path)


def _install_claude_hooks(project_path: Path) -> None:
    """Merge RPG SessionStart hook + rpg-tools MCP pre-approval into ``.claude/settings.json``.

    Merges with existing hooks/permissions without overwriting
    user-defined entries.  A backup of the original file is created
    before any modification.  Idempotent across Python interpreter
    upgrades and repeated ``cmind init/update`` runs:

    * Any prior CoderMind SessionStart entry (identified by the
      ``update_graphs.py`` marker in its command) is replaced rather
      than duplicated.
    * The ``mcp__rpg-tools`` allow rule is added only if absent.

    Why pre-authorize ``mcp__rpg-tools``?
        Claude Code prompts the user before each MCP tool invocation
        unless the rule is present in ``permissions.allow``.  Since
        the CoderMind server only exposes four read-only graph-query
        tools (``search_rpg``, ``explore_rpg``, ``get_node_detail``,
        ``list_rpg_tree``) that touch no external state, requiring
        confirmation for every call is pure friction.  The
        ``mcp__rpg-tools`` server-level rule auto-approves all four.
    """
    settings_dir = project_path / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"

    existing = _load_json_dict(settings_path)
    if settings_path.exists():
        shutil.copy2(settings_path, settings_dir / "settings.json.bak")

    # The command is executed by Claude Code via ``sh -c``, so we inline
    # the same PATH-fallback used by git hooks (see _HOOK_PATH_FALLBACK).
    # Use ``;`` rather than ``&&`` so the cmind call always runs after
    # the (possibly no-op) PATH adjustment.
    marker = "update_graphs.py"  # used for idempotent dedupe across upgrades

    rpg_session_entry = {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": (
                    f"{_HOOK_PATH_FALLBACK}; "
                    "cmind script update_graphs.py status 2>/dev/null"
                    " || echo '[CoderMind] RPG status unavailable'"
                ),
                "timeout": 10,
            }
        ],
    }

    existing_hooks = existing.get("hooks", {})
    if not isinstance(existing_hooks, dict):
        existing_hooks = {}
    merged = dict(existing_hooks)

    session_start = merged.get("SessionStart")
    if not isinstance(session_start, list):
        session_start = []

    def _is_cmind_entry(entry: object) -> bool:
        """Detect an existing CoderMind SessionStart entry.

        Matches the supported command shapes plus any custom CoderMind
        entry the user may have added that still calls ``update_graphs.py``.
        """
        if not isinstance(entry, dict):
            return False
        for h in entry.get("hooks", []) or []:
            cmd = h.get("command", "") if isinstance(h, dict) else ""
            if marker in cmd:
                return True
        return False

    # Drop any stale CoderMind entry before appending the fresh one.
    session_start = [e for e in session_start if not _is_cmind_entry(e)]
    session_start.append(rpg_session_entry)
    merged["SessionStart"] = session_start

    existing["hooks"] = merged

    # Pre-authorize the rpg-tools MCP server.  Claude Code's permission
    # syntax uses the prefix ``mcp__<server-name>`` to grant access to
    # every tool exposed by that server.  We dedupe on the exact rule
    # string so user-added related rules (e.g. per-tool allows) are
    # preserved untouched.
    permissions = existing.get("permissions")
    if not isinstance(permissions, dict):
        permissions = {}
    allow = permissions.get("allow")
    if not isinstance(allow, list):
        allow = []
    cmind_rule = "mcp__rpg-tools"
    if cmind_rule not in allow:
        allow.append(cmind_rule)
    permissions["allow"] = allow
    existing["permissions"] = permissions

    settings_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _read_core_hooks_path(project_path: Path) -> Optional[Path]:
    """Return the directory configured via ``git config core.hooksPath``.

    Returns ``None`` when:
      * git is not installed / not on PATH;
      * ``project_path`` is not inside a git checkout;
      * the config is unset, empty, or the call times out.

    Relative paths are resolved against ``project_path`` (matching git's
    own resolution rules for this config).  No filesystem check is done
    here — the path is returned as-is so callers can decide whether to
    create the directory or warn.

    Why this matters: teams using ``pre-commit``, ``husky``, ``lefthook``
    and similar hook frameworks routinely override ``core.hooksPath`` to
    point at a checked-in directory (e.g. ``.husky/``).  Without this
    lookup, ``cmind init`` would write into ``.git/hooks/`` where git
    never reads from, leaving the user with a silent no-op install.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value:
        return None
    # Expand ``~`` so users who set core.hooksPath = ~/dotfiles/hooks see
    # consistent behavior with git's own expansion.
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        expanded = (project_path / expanded).resolve()
    return expanded


def _resolve_git_hooks_dir(project_path: Path) -> Optional[Path]:
    """Locate the ``hooks/`` directory for ``project_path``'s git checkout.

    Resolution order:

    1. **``core.hooksPath`` override** — if the user (or a tool like
       ``husky`` / ``pre-commit`` / ``lefthook``) has set this config,
       git will only read hooks from that directory.  We honor it so
       our install actually fires.
    2. **Plain repo**: ``<project>/.git/`` is a real directory →
       hooks live at ``<project>/.git/hooks``.
    3. **Worktree**: ``<project>/.git`` is a *file* whose contents
       look like ``gitdir: /path/to/real/gitdir``.  Hooks for the
       whole repo live under that gitdir's ``hooks/`` directory
       (worktrees share hooks with the main repo by default).
    4. **Submodule** (same as worktree but with a different gitdir
       shape) — handled by the same gitdir-file logic.

    Returns ``None`` if no git checkout was found, so callers can
    skip hook installation cleanly in non-git workspaces.
    """
    custom = _read_core_hooks_path(project_path)
    if custom is not None:
        return custom

    git_marker = project_path / ".git"
    if git_marker.is_dir():
        return git_marker / "hooks"
    if git_marker.is_file():
        # The file's first line is ``gitdir: <path>``.  The path may be
        # absolute or relative to project_path; resolve through Path().
        try:
            content = git_marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not content.startswith("gitdir:"):
            return None
        gitdir_value = content.split("gitdir:", 1)[1].strip()
        gitdir_path = Path(gitdir_value)
        if not gitdir_path.is_absolute():
            gitdir_path = (project_path / gitdir_path).resolve()
        # For worktrees ``gitdir`` typically points at
        # ``<main>/.git/worktrees/<name>``.  Hooks for the whole repo
        # are shared and live in ``<main>/.git/hooks`` (two levels up
        # from gitdir_path).  Standalone repos using
        # ``--separate-git-dir`` instead point directly at the gitdir,
        # in which case ``hooks`` is a sibling of the gitdir contents.
        # ``gitdir_path.parent.name == "worktrees"`` is the
        # discriminator for the linked-worktree case.
        if gitdir_path.parent.name == "worktrees":
            return gitdir_path.parent.parent / "hooks"
        return gitdir_path / "hooks"
    return None


# Each entry describes a CoderMind-owned hook snippet shape that can be
# recognized without sentinels. The first element is a substring of the
# snippet's marker comment; the second is the total number of consecutive
# lines occupied by that snippet. These are removed before the sentinel
# block is written so users do not end up with duplicate CoderMind logic.
LegacyBlock = Tuple[str, int]


def _strip_hook_block(
    text: str,
    block_name: str,
    legacy_blocks: Tuple[LegacyBlock, ...] = (),
) -> str:
    """Return ``text`` with any CoderMind-owned hook content removed.

    Two cleanup passes:

    1. Strip the sentinel block::

           # CMIND-BEGIN <block_name>
           ...
           # CMIND-END <block_name>

       Range-based, so multi-line bodies of any shape are atomically
       removed in one shot.

        2. Strip each compatibility snippet described by
             ``(marker_substring, line_count)``. The marker line plus
             ``line_count - 1`` lines following it are dropped. Multiple
             shapes are removed in a single pass so the order of entries in
             ``legacy_blocks`` doesn't matter.

    Lines outside both passes are preserved verbatim so user-authored
    hook content (and shebangs) survive untouched.
    """
    begin_sent = f"# CMIND-BEGIN {block_name}"
    end_sent = f"# CMIND-END {block_name}"
    lines = text.splitlines()

    # Pass 1: strip sentinel block (matching pair).
    after_sentinels: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if not inside and stripped == begin_sent:
            inside = True
            continue
        if inside and stripped == end_sent:
            inside = False
            continue
        if inside:
            continue
        after_sentinels.append(line)

    # Pass 2: strip compatibility snippets by (marker, line_count).
    if not legacy_blocks:
        return "\n".join(after_sentinels)

    out: list[str] = []
    skip = 0
    for line in after_sentinels:
        if skip > 0:
            skip -= 1
            continue
        matched = False
        for marker, count in legacy_blocks:
            if marker in line:
                skip = max(count - 1, 0)
                matched = True
                break
        if not matched:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PATH fallback for hook bodies
# ---------------------------------------------------------------------------
#
# Hooks invoke ``cmind`` (the globally-installed CLI) rather than a
# workspace-local script copy.  When the hook is triggered from a GUI
# editor's source-control panel (VS Code, IntelliJ, GitHub Desktop, ...)
# the process environment may not include the user's shell PATH, so
# ``cmind`` is unresolvable and the hook silently fails.
#
# This snippet is prepended to every hook body.  When ``cmind`` is
# already on PATH (terminal invocations) the test short-circuits and
# the ``export`` is skipped — zero overhead.  When it isn't, we
# prepend ``$HOME/.local/bin`` which is ``uv tool install``'s default
# bin directory.
_HOOK_PATH_FALLBACK = (
    'command -v cmind >/dev/null 2>&1 || '
    'export PATH="$HOME/.local/bin:$PATH"'
)


def _install_hook_snippet(
    hooks_dir: Path,
    hook_name: str,
    block_name: str,
    body: str,
    *,
    legacy_blocks: Tuple[LegacyBlock, ...] = (),
) -> bool:
    """Install or replace an CoderMind-owned block in ``<hooks_dir>/<hook_name>``.

    File layout written::

        #!/bin/sh
        <any pre-existing user content>

        # CMIND-BEGIN <block_name>
        <body>
        # CMIND-END <block_name>

    The block is **atomically replaceable**: subsequent ``cmind init`` /
    ``cmind update`` runs find the existing sentinels and replace the
    whole block, so behavior upgrades land cleanly without duplicate
    snippets. ``legacy_blocks`` recognizes CoderMind-owned hook bodies
    that do not have sentinels yet.

    Creates the hook file with a ``#!/bin/sh`` shebang if absent;
    preserves any user-authored shebang otherwise.  Always returns
    ``True`` (the hook is active on disk after the call); the bool
    return is kept for symmetry with the caller-level
    ``_install_git_*_hook`` functions, which return ``False`` only when
    the workspace has no git checkout at all.
    """
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / hook_name
    existing = hook_path.read_text(encoding="utf-8") if hook_path.exists() else ""

    cleaned = _strip_hook_block(existing, block_name, legacy_blocks).rstrip("\n")

    if not cleaned.strip():
        prefix = "#!/bin/sh\n"
    elif cleaned.lstrip().startswith("#!"):
        prefix = cleaned + "\n"
    else:
        prefix = "#!/bin/sh\n" + cleaned + "\n"

    begin = f"# CMIND-BEGIN {block_name}"
    end = f"# CMIND-END {block_name}"
    block = f"\n{begin}\n{body.rstrip()}\n{end}\n"

    hook_path.write_text(prefix + block, encoding="utf-8")
    hook_path.chmod(0o755)
    return True


def _uninstall_git_pre_commit_hook(project_path: Path) -> bool:
    """Remove any CoderMind-owned ``pre-commit`` block.

    The active git hook contract uses ``post-commit`` and ``post-merge``.
    CoderMind-owned pre-commit blocks are stripped here; user-authored hook
    content (and other tools' blocks such as husky / pre-commit /
    lefthook) is preserved untouched.

    Returns ``True`` when the workspace had a hooks dir to clean,
    ``False`` only when no git checkout was found at all.
    """
    hooks_dir = _resolve_git_hooks_dir(project_path)
    if hooks_dir is None:
        return False

    hook_path = hooks_dir / "pre-commit"
    if not hook_path.is_file():
        return True

    existing = hook_path.read_text(encoding="utf-8")
    legacy = (
        ("# CoderMind: pre-commit dispatcher", 3),
        ("# CoderMind: full RPG sync on commit", 2),
        ("# CoderMind: incremental RPG sync on commit", 3),
    )
    cleaned = _strip_hook_block(existing, "pre-commit", legacy).rstrip("\n")

    # If nothing user-authored remains, delete the hook file so git
    # falls back to its default no-hook behaviour.
    if not cleaned.strip() or cleaned.strip() == "#!/bin/sh":
        try:
            hook_path.unlink()
        except OSError:
            pass
    else:
        hook_path.write_text(cleaned + "\n", encoding="utf-8")
        hook_path.chmod(0o755)
    return True


def _install_git_post_merge_hook(project_path: Path) -> bool:
    """Install the RPG sync command into ``post-merge``.

    Fires after ``git pull`` / ``git merge`` so the dep_graph stays
    aligned with code the user just received from a teammate.  Cannot
    use ``--staged-only`` here because there is no staging area after
    a merge — we want every working-tree change (which is all the new
    code) to be considered.

    The hook is best-effort: any failure is swallowed (``|| true``) so
    a slow / broken sync never blocks the pull.
    """
    hooks_dir = _resolve_git_hooks_dir(project_path)
    if hooks_dir is None:
        return False

    # Dispatcher stub delegates to ``cmind hook post-merge``.
    marker = "# CoderMind: post-merge dispatcher"
    body = (
        f"{marker}\n"
        f"{_HOOK_PATH_FALLBACK}\n"
        f"cmind hook post-merge 2>/dev/null || true"
    )
    return _install_hook_snippet(
        hooks_dir,
        "post-merge",
        "post-merge",
        body,
        legacy_blocks=(
            ("# CoderMind: incremental RPG sync after merge / pull", 3),
        ),
    )


def _install_git_post_commit_hook(project_path: Path) -> bool:
    """Install the ``post-commit`` dispatcher stub.

    The on-disk hook is a short shell snippet that delegates to
    ``cmind hook post-commit``. All orchestration lives in the
    :func:`hook` Python command:

        * **Foreground sync**: ``update_graphs.py sync`` advances
            ``meta.git`` to the new HEAD. Output is teed into
            ``~/.cmind/workspaces/<workspace-id>/logs/hooks.log``.

        * **Background update**: ``update_graphs.py update-rpg`` is
            detached via ``subprocess.Popen(start_new_session=True)``. A
            mkdir-based directory lock at
            ``~/.cmind/workspaces/<workspace-id>/logs/.update_rpg.lock`` serialises
            overlapping commits; locks older than 60 minutes are treated as
            orphaned and removed. The worker's stdout/stderr land in
            ``~/.cmind/workspaces/<workspace-id>/logs/update_rpg.log``.

    Both steps are best-effort: every failure path is swallowed inside
    :func:`hook` so a hook misbehaviour never blocks ``git commit``.

    CoderMind-owned multi-line shell bodies are stripped on upgrade by
    the ``legacy_blocks`` compatibility patterns below.
    """
    hooks_dir = _resolve_git_hooks_dir(project_path)
    if hooks_dir is None:
        return False

    marker = "# CoderMind: post-commit dispatcher"
    body = (
        f"{marker}\n"
        f"{_HOOK_PATH_FALLBACK}\n"
        f"cmind hook post-commit 2>/dev/null || true"
    )
    return _install_hook_snippet(
        hooks_dir,
        "post-commit",
        "post-commit",
        body,
        legacy_blocks=(
            # Two-line sync-only snippet.
            ("# CoderMind: advance meta.git after commit", 2),
            # Five-line sync + setsid background-update snippet.
            ("# CoderMind: advance meta.git + background feature graph update", 5),
        ),
    )


def _install_copilot_hooks(project_path: Path) -> None:
    """Merge an RPG status task into ``.vscode/tasks.json``.

    GitHub Copilot in VS Code does not expose a ``SessionStart`` hook the
    way Claude Code does.  The closest analogue is a VS Code task with
    ``runOptions.runOn = "folderOpen"`` — it fires once when the user
    opens the workspace, before they start chatting with Copilot.  The
    task prints RPG status + ``rpg-tools`` MCP guidance to a terminal,
    which Copilot can read via its terminal-context tools and which also
    nudges the user toward graph-aware queries.

    Merges into any existing ``tasks.json`` without clobbering user
    tasks, preserves the file's ``version`` field, and writes a backup
    when the file already exists.
    """
    vscode_dir = project_path / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = vscode_dir / "tasks.json"

    existing = _load_json_dict(tasks_path)
    if tasks_path.exists():
        try:
            shutil.copy2(tasks_path, vscode_dir / "tasks.json.bak")
        except OSError:
            # Backup is best-effort; never block installation on it.
            pass

    rpg_status_task = {
        "label": "CoderMind: load status",
        "type": "shell",
        # Invoke the globally-installed CLI rather than a workspace
        # script copy (which no longer exists).  Same
        # rationale as the git-hook bodies: portable command name,
        # auto-tracks the installed wheel's scripts.
        "command": "cmind",
        "args": ["script", "update_graphs.py", "status"],
        "presentation": {
            "echo": False,
            "reveal": "silent",
            "focus": False,
            "panel": "dedicated",
            "showReuseMessage": False,
            "clear": False,
            "close": False,
        },
        "runOptions": {"runOn": "folderOpen"},
        "problemMatcher": [],
        "detail": (
            "Prints CoderMind status and rpg-tools MCP usage guidance "
            "so GitHub Copilot can locate and generate code against "
            "the Repository Program Graph."
        ),
    }

    existing.setdefault("version", "2.0.0")
    tasks_list = existing.get("tasks")
    if not isinstance(tasks_list, list):
        tasks_list = []

    # Replace any prior CoderMind task with the same label rather than
    # appending duplicates on repeated ``cmind update`` runs.
    label = rpg_status_task["label"]
    tasks_list = [t for t in tasks_list if not (isinstance(t, dict) and t.get("label") == label)]
    tasks_list.append(rpg_status_task)
    existing["tasks"] = tasks_list

    tasks_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    # Tasks file is workspace-specific (absolute Python path); it is
    # ignored via :func:`_setup_gitignore`, which runs earlier in the
    # init flow.


def _install_hooks(
    project_path: Path,
    selected_ai: str,
    tracker=None,
) -> None:
    """Install RPG auto-update hooks for the selected AI assistant.

    - Claude:  merges a ``SessionStart`` hook into ``.claude/settings.json``
      that runs ``update_graphs.py status`` so stdout (RPG stats + MCP
      usage guidance) is injected into Claude's session context.
    - Copilot: merges a ``runOn: folderOpen`` task into
      ``.vscode/tasks.json`` that runs the same status command on
      workspace open — VS Code's closest analogue to a SessionStart
      hook for GitHub Copilot.
    - All:     installs an RPG sync trigger on ``.git/hooks/post-commit``
      (fired after every successful commit) AND ``.git/hooks/post-merge``
      (fired after ``git pull`` / ``git merge`` so teammate-incoming
      changes get picked up immediately). Any legacy ``pre-commit``
      block from earlier releases is stripped on upgrade — the design
      now relies on post-commit only, so commit latency stays low and
      the inner-git history is cleaner.
      Complements the MCP server already registered in
      ``.mcp.json`` / ``.vscode/mcp.json``.
    """
    try:
        installed = []
        if selected_ai == "claude":
            _install_claude_hooks(project_path)
            installed.append("claude")
        elif selected_ai == "copilot":
            _install_copilot_hooks(project_path)
            installed.append("copilot")

        # Strip any leftover pre-commit block from older installs.
        _uninstall_git_pre_commit_hook(project_path)
        if _install_git_post_commit_hook(project_path):
            installed.append("git:post-commit")
        if _install_git_post_merge_hook(project_path):
            installed.append("git:post-merge")

        if tracker:
            if installed:
                tracker.complete("hooks", ", ".join(installed))
            else:
                tracker.skip("hooks", "no git repo")
    except Exception as e:
        if tracker:
            tracker.error("hooks", str(e))
        else:
            console.print(f"[yellow]Warning: Could not install hooks: {e}[/yellow]")


def _is_private_repo(
    repo_owner: str, repo_name: str, client: httpx.Client, github_token: str = None
) -> bool:
    """Check if a repository is private.

    Args:
        repo_owner: Repository owner username
        repo_name: Repository name
        client: HTTP client
        github_token: Optional GitHub token

    Returns:
        True if repository is private, False if public
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    try:
        response = client.get(
            api_url,
            timeout=10,
            headers=_github_auth_headers(github_token),
        )
        if response.status_code == 200:
            repo_data = response.json()
            return repo_data.get("private", False)
        # If we get 404 without auth, might be private
        return github_token is not None
    except Exception:
        # If error, assume public to fall back to original behavior
        return False


def _get_asset_download_url(
    asset: dict, repo_owner: str, repo_name: str, is_private: bool
) -> str:
    """Get the appropriate download URL for an asset.

    For public repositories, uses browser_download_url.
    For private repositories, uses API endpoint which requires authentication.

    Args:
        asset: Asset dictionary from GitHub API
        repo_owner: Repository owner username
        repo_name: Repository name
        is_private: Whether the repository is private

    Returns:
        Download URL for the asset
    """
    if is_private:
        # Private repo: use API endpoint with asset ID
        asset_id = asset["id"]
        return f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/assets/{asset_id}"
    else:
        # Public repo: use browser download URL
        return asset["browser_download_url"]


def _release_sort_key(release: dict) -> str:
    return release.get("published_at") or release.get("created_at") or ""


def _select_latest_cmind_release(releases: List[dict], *, pre: bool) -> dict | None:
    candidates = [
        release
        for release in releases
        if not release.get("draft")
        and release.get("prerelease", False) is pre
        and release.get("tag_name", "").startswith(_CMIND_RELEASE_TAG_PREFIX)
    ]
    candidates.sort(key=_release_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _format_cmind_version(tag_name: str) -> str:
    if tag_name.startswith(_CMIND_RELEASE_TAG_PREFIX):
        return tag_name[len(_CMIND_RELEASE_TAG_PREFIX) :]
    if tag_name.startswith("v"):
        return tag_name[1:]
    return tag_name


def _fetch_latest_cmind_release(
    repo_owner: str,
    repo_name: str,
    client: httpx.Client,
    *,
    github_token: str = None,
    pre: bool = False,
    timeout: int = 30,
    debug: bool = False,
) -> dict:
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases?per_page=100"
    response = client.get(
        api_url,
        timeout=timeout,
        follow_redirects=True,
        headers=_github_auth_headers(github_token, accept_asset=False),
    )
    status = response.status_code
    if status != 200:
        error_msg = _format_rate_limit_error(status, response.headers, api_url)
        if debug:
            error_msg += f"\n\n[dim]Response body (truncated 500):[/dim]\n{response.text[:500]}"
        raise RuntimeError(error_msg)

    try:
        releases = response.json()
    except ValueError as je:
        raise RuntimeError(
            f"Failed to parse release JSON: {je}\nRaw (truncated 400): {response.text[:400]}"
        )

    if not isinstance(releases, list):
        raise RuntimeError("Unexpected response format when fetching releases list.")

    release_data = _select_latest_cmind_release(releases, pre=pre)
    if release_data is None:
        release_type = "pre-release" if pre else "release"
        raise RuntimeError(
            f"No CoderMind {release_type} found in {repo_owner}/{repo_name}. "
            f"Expected tags to start with {_CMIND_RELEASE_TAG_PREFIX}."
        )
    return release_data


def download_template_from_github(
    ai_assistant: str,
    download_dir: Path,
    *,
    script_type: str = "sh",
    verbose: bool = True,
    show_progress: bool = True,
    client: httpx.Client = None,
    debug: bool = False,
    github_token: str = None,
    pre: bool = False,
) -> Tuple[Path, dict]:
    repo_owner, repo_name = _get_repo_info()
    if client is None:
        client = httpx.Client(verify=ssl_context)

    # Check if repository is private
    is_private = _is_private_repo(repo_owner, repo_name, client, github_token)
    if verbose and debug:
        console.print(
            f"[dim]Repository type: {'private' if is_private else 'public'}[/dim]"
        )

    if verbose:
        if pre:
            console.print("[cyan]Fetching latest pre-release information...[/cyan]")
        else:
            console.print("[cyan]Fetching latest release information...[/cyan]")

    try:
        release_data = _fetch_latest_cmind_release(
            repo_owner,
            repo_name,
            client,
            github_token=github_token,
            pre=pre,
            timeout=30,
            debug=debug,
        )
    except Exception as e:
        console.print("[red]Error fetching release information[/red]")
        console.print(Panel(str(e), title="Fetch Error", border_style="red"))
        raise typer.Exit(1)

    assets = release_data.get("assets", [])
    pattern = f"cmind-template-{ai_assistant}-{script_type}"
    matching_assets = [
        asset
        for asset in assets
        if pattern in asset["name"] and asset["name"].endswith(".zip")
    ]

    asset = matching_assets[0] if matching_assets else None

    if asset is None:
        console.print(
            f"[red]No matching release asset found[/red] for [bold]{ai_assistant}[/bold] (expected pattern: [bold]{pattern}[/bold])"
        )
        asset_names = [a.get("name", "?") for a in assets]
        console.print(
            Panel(
                "\n".join(asset_names) or "(no assets)",
                title="Available Assets",
                border_style="yellow",
            )
        )
        raise typer.Exit(1)

    # Get appropriate download URL based on repository type
    download_url = _get_asset_download_url(asset, repo_owner, repo_name, is_private)
    filename = asset["name"]
    file_size = asset["size"]

    if verbose:
        console.print(f"[cyan]Found template:[/cyan] {filename}")
        console.print(f"[cyan]Size:[/cyan] {file_size:,} bytes")
        console.print(f"[cyan]Release:[/cyan] {release_data['tag_name']}")
        if debug:
            console.print(f"[dim]Download URL: {download_url}[/dim]")

    zip_path = download_dir / filename
    if verbose:
        console.print("[cyan]Downloading template...[/cyan]")

    try:
        with client.stream(
            "GET",
            download_url,
            timeout=60,
            follow_redirects=True,
            headers=_github_auth_headers(github_token, accept_asset=is_private),
        ) as response:
            if response.status_code != 200:
                # Handle rate-limiting on download as well
                error_msg = _format_rate_limit_error(
                    response.status_code, response.headers, download_url
                )
                if debug:
                    error_msg += f"\n\n[dim]Response body (truncated 400):[/dim]\n{response.text[:400]}"
                raise RuntimeError(error_msg)
            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f:
                if total_size == 0:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                else:
                    if show_progress:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            console=console,
                        ) as progress:
                            task = progress.add_task("Downloading...", total=total_size)
                            downloaded = 0
                            for chunk in response.iter_bytes(chunk_size=8192):
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress.update(task, completed=downloaded)
                    else:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
    except Exception as e:
        console.print(
            f"[red]Error downloading template[/red] download_url: {download_url}"
        )
        detail = str(e)
        if zip_path.exists():
            zip_path.unlink()
        console.print(Panel(detail, title="Download Error", border_style="red"))
        raise typer.Exit(1)
    if verbose:
        console.print(f"Downloaded: {filename}")
    metadata = {
        "filename": filename,
        "size": file_size,
        "release": release_data["tag_name"],
        "asset_url": download_url,
    }
    return zip_path, metadata


def download_and_extract_template(
    project_path: Path,
    ai_assistant: str,
    script_type: str,
    is_current_dir: bool = False,
    *,
    verbose: bool = True,
    tracker: StepTracker | None = None,
    client: httpx.Client = None,
    debug: bool = False,
    # DEPRECATED params (kept for source-compat; CLI no longer passes
    # them as of v0.1.4 and they are slated for removal in v0.2.0).
    github_token: str = None,
    pre: bool = False,
    legacy_download: bool = False,
) -> Path:
    """Provision the workspace with scripts + command templates.

    Bundle-only as of v0.1.4: templates are always sourced from the
    packaged assets shipped inside ``cmind_cli/core_pack/``.  To pick
    up newer prompts the user upgrades the CLI itself (``uv tool
    upgrade cmind-cli`` etc.), which ``cmind update`` does
    automatically by default.

    The ``github_token`` / ``pre`` / ``legacy_download`` parameters and
    the underlying ``_download_and_extract_release_zip`` path are kept
    for now as dead code so the change is reversible, but they are no
    longer reachable from the CLI surface.

    Returns ``project_path``.  Uses the supplied :class:`StepTracker`
    to report progress when provided.
    """
    return _install_from_bundle(
        project_path,
        ai_assistant,
        script_type,
        is_current_dir,
        verbose=verbose,
        tracker=tracker,
    )


def _install_from_bundle(
    project_path: Path,
    ai_assistant: str,
    script_type: str,
    is_current_dir: bool,
    *,
    verbose: bool = True,
    tracker: StepTracker | None = None,
) -> Path:
    """Materialise per-AI command templates into the workspace.

    The pipeline scripts themselves live inside the installed wheel at
    ``cmind_cli/core_pack/scripts/`` and are invoked via ``cmind
    script <name>`` (and ``cmind-mcp`` for the MCP server) — they are
    NOT copied to ``<workspace>/.cmind/scripts/`` anymore.  This gives
    one source of truth per CLI install, no
    risk of workspace/wheel drift, and no per-workspace scripts dir
    to keep in sync.

    Only slash-command templates land in the workspace, plus the
    provisioning marker that records which channel was used.
    """
    from . import _assets

    if tracker:
        # init()/update() already registered fetch/download/extract step keys,
        # so just transition them through completed states instead of
        # re-adding (which would overwrite the existing label).
        tracker.start("fetch", "packaged assets (offline)")
        tracker.complete("fetch", "bundle ready")
        tracker.skip("download", "bundle mode (no network)")

    if not is_current_dir:
        project_path.mkdir(parents=True)

    if tracker:
        tracker.start("extract")

    try:
        cmind_root = project_path / ".cmind"
        cmind_root.mkdir(parents=True, exist_ok=True)

        # 1. Materialise slash-command templates into the AI-specific
        #    directory.  _materialise_commands_for_agent owns the
        #    per-agent file-name / folder rules.
        _materialise_commands_for_agent(
            ai_assistant, _assets.commands_dir(), project_path
        )

        # 2. Record the provisioning source so subsequent ``cmind update``
        #    invocations default to the same channel.
        _write_source_marker(project_path, _SOURCE_BUNDLE)

        if tracker:
            tracker.skip("zip-list", "bundle (no archive)")
            tracker.skip("extracted-summary", "templates only")
            tracker.complete("extract")
            tracker.skip("cleanup", "bundle mode")
    except Exception as e:
        if tracker:
            tracker.error("extract", str(e))
        else:
            console.print(f"[red]Error installing from bundle:[/red] {e}")
        raise

    return project_path


def _materialise_commands_for_agent(
    ai_assistant: str,
    src_commands_dir: Path,
    project_path: Path,
) -> None:
    """Place command templates into the agent-specific workspace location.

    This intentionally mirrors what the legacy release-zip path produces
    (see ``.github/workflows/scripts/cmind/create-release-packages.sh``
    ``generate_commands`` / ``generate_copilot_prompts``), so that
    downstream consumers see the same layout regardless of provisioning
    source.

    Layout produced:
      claude  → ``.claude/commands/cmind.<name>.md``
      copilot → ``.github/agents/cmind.<name>.agent.md``
                ``.github/prompts/cmind.<name>.prompt.md`` (frontmatter
                points at the corresponding agent)
      others  → fallback: ``.cmind/commands/cmind.<name>.md`` (same
                ``cmind.<name>.md`` prefix for consistency with the
                supported agents above)

    NOTE: ``claude`` and ``copilot`` are the only verified agents in
    AGENT_CONFIG today.  Add new agents here when AGENT_CONFIG grows.
    """
    def _read_body(src: Path) -> str:
        # Normalise CRLF → LF, matching what the CI's ``tr -d '\r'`` does.
        return src.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")

    if ai_assistant == "claude":
        dest = project_path / ".claude" / "commands"
        dest.mkdir(parents=True, exist_ok=True)
        for src in src_commands_dir.glob("*.md"):
            target = dest / f"cmind.{src.stem}.md"
            target.write_text(_read_body(src), encoding="utf-8")
    elif ai_assistant == "copilot":
        agents = project_path / ".github" / "agents"
        prompts = project_path / ".github" / "prompts"
        agents.mkdir(parents=True, exist_ok=True)
        prompts.mkdir(parents=True, exist_ok=True)
        for src in src_commands_dir.glob("*.md"):
            stem = f"cmind.{src.stem}"
            body = _read_body(src)
            (agents / f"{stem}.agent.md").write_text(body, encoding="utf-8")
            # Copilot prompt files reference the agent by name in
            # frontmatter; the body is empty so the agent prompt
            # (already written above) is the source of truth.
            (prompts / f"{stem}.prompt.md").write_text(
                f"---\nagent: {stem}\n---\n", encoding="utf-8"
            )
    else:
        # Unknown agent (init() validates against AGENT_CONFIG so this
        # branch is unreachable from the public CLI, but provides a
        # well-defined behaviour if a future caller bypasses validation).
        dest = project_path / ".cmind" / "commands"
        dest.mkdir(parents=True, exist_ok=True)
        for src in src_commands_dir.glob("*.md"):
            (dest / f"cmind.{src.stem}.md").write_text(_read_body(src), encoding="utf-8")


# DEPRECATED: legacy release-zip provisioning path — no longer reachable
# from the CLI as of v0.1.4 (see top-of-file DEPRECATED block).  Slated for
# removal in v0.2.0.
def _download_and_extract_release_zip(
    project_path: Path,
    ai_assistant: str,
    script_type: str,
    is_current_dir: bool = False,
    *,
    verbose: bool = True,
    tracker: StepTracker | None = None,
    client: httpx.Client = None,
    debug: bool = False,
    github_token: str = None,
    pre: bool = False,
) -> Path:
    """Release-zip download + extract path.

    Kept available for users that need the very latest prompts before the
    next CLI release, or to bypass packaging glitches.  Activated via
    ``cmind init --legacy-download``.
    """
    current_dir = Path.cwd()

    if tracker:
        tracker.start("fetch", "contacting GitHub API")
    try:
        zip_path, meta = download_template_from_github(
            ai_assistant,
            current_dir,
            script_type=script_type,
            verbose=verbose and tracker is None,
            show_progress=(tracker is None),
            client=client,
            debug=debug,
            github_token=github_token,
            pre=pre,
        )
        if tracker:
            tracker.complete(
                "fetch", f"release {meta['release']} ({meta['size']:,} bytes)"
            )
            tracker.add("download", "Download template")
            tracker.complete("download", meta["filename"])
    except Exception as e:
        if tracker:
            tracker.error("fetch", str(e))
        else:
            if verbose:
                console.print(f"[red]Error downloading template:[/red] {e}")
        raise

    if tracker:
        tracker.add("extract", "Extract template")
        tracker.start("extract")
    elif verbose:
        console.print("Extracting template...")

    try:
        if not is_current_dir:
            project_path.mkdir(parents=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_contents = zip_ref.namelist()
            if tracker:
                tracker.start("zip-list")
                tracker.complete("zip-list", f"{len(zip_contents)} entries")
            elif verbose:
                console.print(f"[cyan]ZIP contains {len(zip_contents)} items[/cyan]")

            if is_current_dir:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    zip_ref.extractall(temp_path)

                    extracted_items = list(temp_path.iterdir())
                    if tracker:
                        tracker.start("extracted-summary")
                        tracker.complete(
                            "extracted-summary", f"temp {len(extracted_items)} items"
                        )
                    elif verbose:
                        console.print(
                            f"[cyan]Extracted {len(extracted_items)} items to temp location[/cyan]"
                        )

                    source_dir = temp_path
                    if len(extracted_items) == 1 and extracted_items[0].is_dir():
                        source_dir = extracted_items[0]
                        if tracker:
                            tracker.add("flatten", "Flatten nested directory")
                            tracker.complete("flatten")
                        elif verbose:
                            console.print(
                                "[cyan]Found nested directory structure[/cyan]"
                            )

                    for item in source_dir.iterdir():
                        dest_path = project_path / item.name
                        if item.is_dir():
                            if dest_path.exists():
                                if verbose and not tracker:
                                    console.print(
                                        f"[yellow]Merging directory:[/yellow] {item.name}"
                                    )
                                for sub_item in item.rglob("*"):
                                    if sub_item.is_file():
                                        rel_path = sub_item.relative_to(item)
                                        dest_file = dest_path / rel_path
                                        dest_file.parent.mkdir(
                                            parents=True, exist_ok=True
                                        )
                                        # Special handling for .vscode/settings.json - merge instead of overwrite
                                        if (
                                            dest_file.name == "settings.json"
                                            and dest_file.parent.name == ".vscode"
                                        ):
                                            handle_vscode_settings(
                                                sub_item,
                                                dest_file,
                                                rel_path,
                                                verbose,
                                                tracker,
                                            )
                                        else:
                                            shutil.copy2(sub_item, dest_file)
                            else:
                                shutil.copytree(item, dest_path)
                        else:
                            if dest_path.exists() and verbose and not tracker:
                                console.print(
                                    f"[yellow]Overwriting file:[/yellow] {item.name}"
                                )
                            shutil.copy2(item, dest_path)
                    if verbose and not tracker:
                        console.print(
                            "[cyan]Template files merged into current directory[/cyan]"
                        )
            else:
                zip_ref.extractall(project_path)

                extracted_items = list(project_path.iterdir())
                if tracker:
                    tracker.start("extracted-summary")
                    tracker.complete(
                        "extracted-summary", f"{len(extracted_items)} top-level items"
                    )
                elif verbose:
                    console.print(
                        f"[cyan]Extracted {len(extracted_items)} items to {project_path}:[/cyan]"
                    )
                    for item in extracted_items:
                        console.print(
                            f"  - {item.name} ({'dir' if item.is_dir() else 'file'})"
                        )

                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    nested_dir = extracted_items[0]
                    temp_move_dir = project_path.parent / f"{project_path.name}_temp"

                    shutil.move(str(nested_dir), str(temp_move_dir))

                    project_path.rmdir()

                    shutil.move(str(temp_move_dir), str(project_path))
                    if tracker:
                        tracker.add("flatten", "Flatten nested directory")
                        tracker.complete("flatten")
                    elif verbose:
                        console.print(
                            "[cyan]Flattened nested directory structure[/cyan]"
                        )

    except Exception as e:
        if tracker:
            tracker.error("extract", str(e))
        else:
            if verbose:
                console.print(f"[red]Error extracting template:[/red] {e}")
                if debug:
                    console.print(
                        Panel(str(e), title="Extraction Error", border_style="red")
                    )

        if not is_current_dir and project_path.exists():
            shutil.rmtree(project_path)
        raise typer.Exit(1)
    else:
        if tracker:
            tracker.complete("extract")
    finally:
        if tracker:
            tracker.add("cleanup", "Remove temporary archive")

        if zip_path.exists():
            zip_path.unlink()
            if tracker:
                tracker.complete("cleanup")
            elif verbose:
                console.print(f"Cleaned up: {zip_path.name}")

    # Record provisioning source so a later ``cmind update`` defaults
    # to the same channel.  Counterpart to ``_install_from_bundle`` which
    # writes ``bundle``.
    _write_source_marker(project_path, _SOURCE_LEGACY)

    # Discard the scripts copy extracted from the zip — they're not
    # used at runtime anymore (the workspace invokes ``cmind script
    # <name>`` which resolves to the packaged scripts dir).  Keeping
    # them would just be dead weight that drifts vs the installed CLI.
    # Legacy zip contributes commands only.
    legacy_scripts_dir = project_path / ".cmind" / "scripts"
    if legacy_scripts_dir.is_dir():
        shutil.rmtree(legacy_scripts_dir, ignore_errors=True)

    return project_path


def ensure_cmind_runtime_dirs(
    project_path: Path, tracker: StepTracker | None = None
) -> None:
    """Pre-create CoderMind runtime directories under ``~/.cmind/``.

    The per-workspace data, logs, and inner-git snapshot repo live
    under the user's home directory at ``~/.cmind/workspaces/<workspace-id>/``
    rather than inside the workspace.  Reports stay in the workspace
    (``<workspace>/.cmind/reports/``) because they're user-facing
    artefacts.

    This function is the central bootstrap for the home layout: it's
    idempotent and safe to call from both ``cmind init`` (when the
    channel was just chosen) and ``cmind update`` (when the channel
    is read from the existing meta file).  Some early-pipeline prompts
    redirect stdout/stderr to ``<logs>/<stage>.log`` via shell ``>``
    before any Python code runs, so we must create the directories
    upfront rather than lazily.

    Created (idempotent):
        - ``~/.cmind/workspaces/<workspace-id>/data/``
        - ``~/.cmind/workspaces/<workspace-id>/data/trajectory/``
        - ``~/.cmind/workspaces/<workspace-id>/logs/``
        - ``<workspace>/.cmind/reports/``
        - ``~/.cmind/workspaces/<workspace-id>/.meta.toml`` (refreshed)

    The inner ``.git/`` directory is NOT created here; that's
    the responsibility of :mod:`cmind_cli._inner_git`, which seeds an
    initial commit with a meaningful message.
    """
    # Resolve channel: prefer what's already recorded, fall back to
    # bundle.  The caller (init) will explicitly call
    # ``_write_source_marker`` afterwards to lock in the final value,
    # so this lookup is just a sensible default for the first run.
    existing_channel = _read_source_marker(project_path)
    channel = existing_channel or _storage.CHANNEL_BUNDLE

    try:
        home_dir = _storage.ensure_workspace_storage(
            project_path,
            channel=channel,
            cmind_cli_version=_current_cli_version(),
        )
    except _storage.WorkspaceMetaMismatch as exc:
        # Hash collision or manual rename.  Surface clearly: silently
        # writing into the wrong workspace would corrupt the other
        # one's data.
        if tracker:
            tracker.add("runtime-dirs", "Ensure ~/.cmind/{logs,data} directories")
            tracker.error("runtime-dirs", str(exc))
        else:
            console.print(f"[red]error:[/red] {exc}")
        raise
    except OSError as exc:
        # Filesystem read-only / permission issue — non-blocking.
        if tracker:
            tracker.add("runtime-dirs", "Ensure ~/.cmind/{logs,data} directories")
            tracker.error("runtime-dirs", f"could not create: {exc}")
        return

    # data/trajectory is a script-specific subdir; create explicitly
    # so the encoder's early stages can write into it without their
    # own ``mkdir -p`` dance.
    try:
        (home_dir / "data" / "trajectory").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if tracker:
        tracker.add("runtime-dirs", "Ensure ~/.cmind/{logs,data} directories")
        tracker.complete(
            "runtime-dirs",
            f"home dir at {home_dir}",
        )


def _detect_ai_agent(project_path: Path) -> str | None:
    """Detect AI agent from existing project directory.

    Scans for known agent folders (from AGENT_CONFIG) and checks if they
    contain cmind.* command files. Returns the agent key or None.
    """
    found = []
    for key, config in AGENT_CONFIG.items():
        agent_dir = project_path / config["folder"]
        if agent_dir.is_dir():
            # Check common command subdirectories for cmind.* files
            for sub in ("commands", "agents", "prompts"):
                candidate = agent_dir / sub
                if candidate.is_dir() and any(candidate.glob("cmind.*")):
                    found.append(key)
                    break
            else:
                # Folder exists even without cmind commands subdirectory
                found.append(key)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        # Multiple agents detected — caller should let user choose
        return None
    return None


@app.command()
def init(
    project_name: str = typer.Argument(
        None,
        help="Name for your new project directory (optional if using --here, or use '.' for current directory)",
    ),
    ai_assistant: str = typer.Option(
        None,
        "--ai",
        help="AI assistant to use: copilot or claude",
    ),
    script_type: str = typer.Option(
        None, "--script", help="Script type to use: sh or ps"
    ),
    ignore_agent_tools: bool = typer.Option(
        False,
        "--ignore-agent-tools",
        help="Skip checks for AI agent tools like Claude Code",
    ),
    no_git: bool = typer.Option(
        False, "--no-git", help="Skip git repository initialization"
    ),
    here: bool = typer.Option(
        False,
        "--here",
        help="Initialize project in the current directory instead of creating a new one",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force merge/overwrite when using --here (skip confirmation)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show verbose diagnostic output",
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP server registration (rpg-tools won't be exposed to the AI agent)",
    ),
    no_copilot_cli_mcp: bool = typer.Option(
        False,
        "--no-copilot-cli-mcp",
        help=(
            "When --ai copilot is selected, skip also registering "
            "rpg-tools globally in ~/.copilot/mcp-config.json.  The "
            "Copilot CLI does not read workspace .vscode/mcp.json, so "
            "this global registration is what makes `copilot` find "
            "rpg-tools.  Pass this flag if you manage your Copilot CLI "
            "MCP config by hand."
        ),
    ),
    encode: Optional[bool] = typer.Option(
        None,
        "--encode/--no-encode",
        help=(
            "Run the encoder at the end of init to build the initial RPG "
            "graph from existing code.  By default we ask interactively when "
            "there's Python code in the workspace; pass --encode to skip the "
            "prompt and run, or --no-encode to skip the prompt and not run."
        ),
    ),
    no_cmind_git: bool = typer.Option(
        False,
        "--no-cmind-git",
        help=(
            "Skip initialising a private git repository inside .cmind/. "
            "Default is ON: cmind init seeds .cmind/.git "
            "so every subsequent `cmind script` invocation auto-snapshots "
            "the workspace state, letting you `git log` / `git diff` "
            "between pipeline stages without extra tooling.  This flag "
            "disables the feature for the current init only."
        ),
    ),
):
    """Initialize a new CoderMind project from the latest template.

    This command will:
    1. Check that required tools are installed (git is optional)
    2. Let you choose your AI assistant
    3. Install command templates from the packaged bundle
    4. Place them into a new project directory or current directory
    5. Initialize a fresh git repository (if not --no-git and no existing repo)
    6. Optionally set up AI assistant commands

    Examples:
        cmind init my-project
        cmind init my-project --ai claude
        cmind init my-project --ai copilot --no-git
        cmind init --ignore-agent-tools my-project
        cmind init . --ai claude         # Initialize in current directory
        cmind init .                     # Initialize in current directory (interactive AI selection)
        cmind init --here --ai claude    # Alternative syntax for current directory
        cmind init --here --ai codex
        cmind init --here --ai codebuddy
        cmind init --here
        cmind init --here --force  # Skip confirmation when current directory not empty
    """
    show_banner()

    if project_name == ".":
        here = True
        project_name = None  # Clear project_name to use existing validation logic

    if here and project_name:
        console.print(
            "[red]Error:[/red] Cannot specify both project name and --here flag"
        )
        raise typer.Exit(1)

    if not here and not project_name:
        console.print(
            "[red]Error:[/red] Must specify either a project name, use '.' for current directory, or use --here flag"
        )
        raise typer.Exit(1)

    if here:
        project_name = Path.cwd().name
        project_path = Path.cwd()

        existing_items = list(project_path.iterdir())
        if existing_items:
            console.print(
                f"[yellow]Warning:[/yellow] Current directory is not empty ({len(existing_items)} items)"
            )
            console.print(
                "[yellow]Template files will be merged with existing content and may overwrite existing files[/yellow]"
            )
            if force:
                console.print(
                    "[cyan]--force supplied: skipping confirmation and proceeding with merge[/cyan]"
                )
            else:
                response = typer.confirm("Do you want to continue?")
                if not response:
                    console.print("[yellow]Operation cancelled[/yellow]")
                    raise typer.Exit(0)
    else:
        project_path = Path(project_name).resolve()
        if project_path.exists():
            error_panel = Panel(
                f"Directory '[cyan]{project_name}[/cyan]' already exists\n"
                "Please choose a different project name or remove the existing directory.",
                title="[red]Directory Conflict[/red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print()
            console.print(error_panel)
            raise typer.Exit(1)

    current_dir = Path.cwd()

    setup_lines = [
        "[cyan]CoderMind Project Setup[/cyan]",
        "",
        f"{'Project':<15} [green]{project_path.name}[/green]",
        f"{'Working Path':<15} [dim]{current_dir}[/dim]",
    ]

    if not here:
        setup_lines.append(f"{'Target Path':<15} [dim]{project_path}[/dim]")

    console.print(Panel("\n".join(setup_lines), border_style="cyan", padding=(1, 2)))

    should_init_git = False
    if not no_git:
        should_init_git = check_tool("git")
        if not should_init_git:
            console.print(
                "[yellow]Git not found - will skip repository initialization[/yellow]"
            )

    if ai_assistant:
        if ai_assistant not in AGENT_CONFIG:
            console.print(
                f"[red]Error:[/red] Invalid AI assistant '{ai_assistant}'. Choose from: {', '.join(AGENT_CONFIG.keys())}"
            )
            raise typer.Exit(1)
        selected_ai = ai_assistant
    else:
        # Create options dict for selection (agent_key: display_name)
        ai_choices = {key: config["name"] for key, config in AGENT_CONFIG.items()}
        selected_ai = select_with_arrows(
            ai_choices, "Choose your AI assistant:", "copilot"
        )

    if not ignore_agent_tools:
        agent_config = AGENT_CONFIG.get(selected_ai)
        if agent_config and agent_config["requires_cli"]:
            install_url = agent_config["install_url"]
            if not check_tool(selected_ai):
                error_panel = Panel(
                    f"[cyan]{selected_ai}[/cyan] not found\n"
                    f"Install from: [cyan]{install_url}[/cyan]\n"
                    f"{agent_config['name']} is required to continue with this project type.\n\n"
                    "Tip: Use [cyan]--ignore-agent-tools[/cyan] to skip this check",
                    title="[red]Agent Detection Error[/red]",
                    border_style="red",
                    padding=(1, 2),
                )
                console.print()
                console.print(error_panel)
                raise typer.Exit(1)

    if script_type:
        if script_type not in SCRIPT_TYPE_CHOICES:
            console.print(
                f"[red]Error:[/red] Invalid script type '{script_type}'. Choose from: {', '.join(SCRIPT_TYPE_CHOICES.keys())}"
            )
            raise typer.Exit(1)
        # PowerShell support is planned but not yet wired into the
        # bundled templates / pipeline scripts.  Reject explicit
        # --script ps with a friendly message so users aren't surprised
        # by missing files later.
        if script_type == "ps":
            console.print(
                "[yellow]PowerShell (--script ps) is not yet supported and will "
                "be added in a future release. Please use --script sh for now.[/yellow]"
            )
            raise typer.Exit(1)
        selected_script = script_type
    else:
        # Default to sh on every platform until PowerShell templates land.
        default_script = "sh"

        if sys.stdin.isatty():
            selected_script = select_with_arrows(
                SCRIPT_TYPE_CHOICES,
                "Choose script type (or press Enter)",
                default_script,
            )
        else:
            selected_script = default_script

    console.print(f"[cyan]Selected AI assistant:[/cyan] {selected_ai}")
    console.print(f"[cyan]Selected script type:[/cyan] {selected_script}")

    tracker = StepTracker("Initialize CoderMind Project")

    sys._cmind_tracker_active = True

    tracker.add("precheck", "Check required tools")
    tracker.complete("precheck", "ok")
    tracker.add("ai-select", "Select AI assistant")
    tracker.complete("ai-select", f"{selected_ai}")
    tracker.add("script-select", "Select script type")
    tracker.complete("script-select", selected_script)
    for key, label in [
        ("fetch", "Install bundled templates"),
        ("download", "Download template"),
        ("extract", "Extract template"),
        ("zip-list", "Archive contents"),
        ("extracted-summary", "Extraction summary"),
        ("chmod", "Ensure scripts executable"),
        ("gitignore", "Configure .gitignore"),
        ("mcp", "Configure MCP server"),
        ("copilot-cli-mcp", "Register rpg-tools in ~/.copilot/mcp-config.json"),
        ("cleanup", "Cleanup"),
        ("git", "Initialize git repository"),
        ("hooks", "Install auto-update hooks"),
        ("final", "Finalize"),
    ]:
        tracker.add(key, label)

    # Track git error message outside Live context so it persists
    git_error_message = None

    with Live(
        tracker.render(), console=console, refresh_per_second=8, transient=True
    ) as live:
        tracker.attach_refresh(lambda: live.update(tracker.render()))
        try:
            download_and_extract_template(
                project_path,
                selected_ai,
                selected_script,
                here,
                verbose=False,
                tracker=tracker,
                debug=debug,
            )

            # .cmind/.source is written by whichever provisioning path
            # actually ran (_install_from_bundle / _download_and_extract_release_zip).

            # Materialise .cmind/config.toml with the resolved AI CLI
            # command.  llm_client.py reads this at runtime to invoke
            # the right sub-agent.
            _write_workspace_config(project_path, selected_ai)

            # Materialize .gitignore *before* MCP/hook generation so the
            # files those steps create (.vscode/mcp.json, .vscode/tasks.json,
            # .mcp.json) are ignored from the moment they hit disk.  This is
            # the single point of truth for gitignore management; downstream
            # steps must NOT modify .gitignore themselves.
            tracker.start("gitignore")
            try:
                _setup_gitignore(project_path, selected_ai)
                tracker.complete("gitignore", "configured")
            except Exception as exc:
                tracker.error("gitignore", str(exc))

            # Generate MCP server configuration (unless explicitly skipped)
            if no_mcp:
                tracker.skip("mcp", "--no-mcp flag")
            else:
                _generate_mcp_config(project_path, selected_ai, tracker=tracker)

            # Global registration for Copilot CLI (which doesn't read
            # workspace .vscode/mcp.json).  Skipped for non-copilot AIs,
            # when --no-mcp is set, or when the user opts out explicitly.
            if no_mcp:
                pass
            elif selected_ai != "copilot":
                tracker.skip("copilot-cli-mcp", f"ai={selected_ai}")
            elif no_copilot_cli_mcp:
                tracker.skip("copilot-cli-mcp", "--no-copilot-cli-mcp flag")
            else:
                tracker.start("copilot-cli-mcp")
                _register_copilot_cli_global_mcp(tracker=tracker)

            if not no_git:
                tracker.start("git")
                if is_git_repo(project_path):
                    tracker.complete("git", "existing repo detected")
                elif should_init_git:
                    success, error_msg = init_git_repo(project_path, quiet=True)
                    if success:
                        tracker.complete("git", "initialized")
                    else:
                        tracker.error("git", "init failed")
                        git_error_message = error_msg
                else:
                    tracker.skip("git", "git not available")
            else:
                tracker.skip("git", "--no-git flag")

            _install_hooks(project_path, selected_ai, tracker=tracker)

            tracker.complete("final", "project ready")
        except Exception as e:
            tracker.error("final", str(e))
            console.print(
                Panel(
                    f"Initialization failed: {e}", title="Failure", border_style="red"
                )
            )
            if debug:
                _env_pairs = [
                    ("Python", sys.version.split()[0]),
                    ("Platform", sys.platform),
                    ("CWD", str(Path.cwd())),
                ]
                _label_width = max(len(k) for k, _ in _env_pairs)
                env_lines = [
                    f"{k.ljust(_label_width)} → [bright_black]{v}[/bright_black]"
                    for k, v in _env_pairs
                ]
                console.print(
                    Panel(
                        "\n".join(env_lines),
                        title="Debug Environment",
                        border_style="magenta",
                    )
                )
            if not here and project_path.exists():
                shutil.rmtree(project_path)
            raise typer.Exit(1)
        finally:
            pass

    console.print(tracker.render())
    console.print("\n[bold green]Project ready.[/bold green]")

    # PATH self-check: hooks and MCP rely on ``cmind`` / ``cmind-mcp``
    # being resolvable.  If they aren't on PATH, the user will hit
    # opaque failures from git hooks and MCP clients later — surface
    # the actionable hint now.
    import shutil as _shutil
    if _shutil.which("cmind-mcp") is None or _shutil.which("cmind") is None:
        reinstall_cmd: Optional[list[str]] = _upgrade_command(_detect_install_method())
        # ``--force`` reinstalls in place which fixes most PATH issues
        # caused by partial installs / corrupted shim links.
        if reinstall_cmd and reinstall_cmd[:3] == ["uv", "tool", "upgrade"]:
            reinstall_hint = "uv tool install cmind-cli --force"
        elif reinstall_cmd and reinstall_cmd[:2] == ["pipx", "upgrade"]:
            reinstall_hint = "pipx install cmind-cli --force"
        elif reinstall_cmd:
            reinstall_hint = " ".join(reinstall_cmd)
        else:
            reinstall_hint = "uv tool install cmind-cli --force  # or your installer's equivalent"
        console.print()
        path_panel = Panel(
            "[yellow]Warning:[/yellow] [cyan]cmind[/cyan] / [cyan]cmind-mcp[/cyan] "
            "not found on PATH.\n\n"
            "Git hooks and the MCP server invoke these commands; they will "
            "fail until PATH is fixed.\n\n"
            "[bold]Fix:[/bold]\n"
            "  - Linux/macOS: add [cyan]~/.local/bin[/cyan] to PATH in your shell rc\n"
            "  - Windows:     add [cyan]%USERPROFILE%\\.local\\bin[/cyan] to PATH\n"
            f"  - Or reinstall: [cyan]{reinstall_hint}[/cyan]",
            title="[red]PATH check[/red]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print(path_panel)

    # Show git error details if initialization failed
    if git_error_message:
        console.print()
        git_error_panel = Panel(
            f"[yellow]Warning:[/yellow] Git repository initialization failed\n\n"
            f"{git_error_message}\n\n"
            f"[dim]You can initialize git manually later with:[/dim]\n"
            f"[cyan]cd {project_path if not here else '.'}[/cyan]\n"
            f"[cyan]git init[/cyan]\n"
            f"[cyan]git add .[/cyan]\n"
            f'[cyan]git commit -m "Initial commit"[/cyan]',
            title="[red]Git Initialization Failed[/red]",
            border_style="red",
            padding=(1, 2),
        )
        console.print(git_error_panel)

    # Agent folder security notice
    agent_config = AGENT_CONFIG.get(selected_ai)
    if agent_config:
        if selected_ai == "copilot":
            ignored_path_desc = ".github/agents/ and .github/prompts/"
        else:
            ignored_path_desc = agent_config["folder"]
        security_notice = Panel(
            f"CoderMind's slash command definitions under [cyan]{ignored_path_desc}[/cyan] are regenerated by [cyan]cmind init/update[/cyan] and are excluded from git by default.\n"
            f"Collaborators should run [cyan]cmind init[/cyan] in their clone to materialize the prompt files locally.",
            title="[yellow]Agent Folder Notice[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print()
        console.print(security_notice)

    # Pre-create runtime directories so early pipeline prompts that redirect
    # to ~/.cmind/workspaces/<workspace-id>/logs/<stage>.log don't fail with "No such file or directory".
    ensure_cmind_runtime_dirs(project_path)

    steps_lines = []
    if not here:
        steps_lines.append(
            f"1. Go to the project folder: [cyan]cd {project_name}[/cyan]"
        )
        step_num = 2
    else:
        steps_lines.append("1. You're already in the project directory!")
        step_num = 2

    # Add Codex-specific setup step if needed
    if selected_ai == "codex":
        codex_path = project_path / ".codex"
        quoted_path = shlex.quote(str(codex_path))
        if os.name == "nt":  # Windows
            cmd = f"setx CODEX_HOME {quoted_path}"
        else:  # Unix-like systems
            cmd = f"export CODEX_HOME={quoted_path}"

        steps_lines.append(
            f"{step_num}. Set [cyan]CODEX_HOME[/cyan] environment variable before running Codex: [cyan]{cmd}[/cyan]"
        )
        step_num += 1

    steps_lines.append(f"{step_num}. Start using high-level slash commands with your AI agent:")

    steps_lines.extend([
        "   For new projects / requirements-to-code:",
        f"   {step_num}.1  [cyan]/cmind.feature_construct <feature description>[/] - Build the feature tree from requirements",
        f"   {step_num}.2  [dim][Optional][/dim] [cyan]/cmind.feature_edit <edit instructions>[/] - Edit Feature Tree Nodes",
        f"   {step_num}.3  [cyan]/cmind.plan[/] - Run RPG construction and planning",
        f"   {step_num}.4  [cyan]/cmind.code_gen[/] - Code Generation",
        f"   {step_num}.5  [dim][Optional][/dim] [cyan]/cmind.rpg_edit <edit instructions>[/] - Surgical RPG/code edit",
        "",
        "   For existing repositories / code-to-RPG:",
        f"   {step_num}.6  [cyan]/cmind.encode[/] - Encode an existing repo into RPG",
        f"   {step_num}.7  [cyan]/cmind.update_rpg[/] - Manual incremental RPG update fallback",
        f"   {step_num}.8  [dim][Optional][/dim] [cyan]/cmind.rpg_edit <edit instructions>[/] - Surgical RPG/code edit",
        "",
        "   For finer-grained commands and stage-by-stage reruns, see:",
        "   [link=https://github.com/microsoft/RPG-ZeroRepo/blob/main/CoderMind/docs/commands.md]https://github.com/microsoft/RPG-ZeroRepo/blob/main/CoderMind/docs/commands.md[/link]",
    ])

    step_num += 1
    steps_lines.append(
        f"{step_num}. You can inspect each step's output under [cyan]~/.cmind/workspaces/<workspace-id>/data/[/cyan], "
        f"and review detailed execution trajectories under [cyan]~/.cmind/workspaces/<workspace-id>/data/trajectory/[/cyan]. "
        f"Run [cyan]cmind version[/cyan] from inside the workspace to see the resolved Data / Logs / Inner-git paths."
    )

    step_num += 1
    steps_lines.append(
        f"{step_num}. The CoderMind MCP server provides [cyan]search_rpg[/], [cyan]explore_rpg[/], "
        f"[cyan]get_node_detail[/], and [cyan]list_rpg_tree[/] "
        f"tools for AI agents to query RPG graphs via the Model Context Protocol."
    )
    # First-run note: the MCP tools are wired up at init time, but they
    # only return useful data once the encoder has built rpg.json.  Make
    # the requirement loud-and-clear here so users don't hit the silent
    # "rpg_unavailable" payload on their first /cmind.* call.
    steps_lines.append(
        "   [yellow]Note:[/] the MCP tools query [cyan]rpg.json[/] in the workspace's home-dir "
        "store, which is created by the encoder. For existing codebases, run [cyan]/cmind.encode[/] "
        "once now to populate it; the post-commit hook keeps it in sync afterwards."
    )

    steps_panel = Panel(
        "\n".join(steps_lines), title="Next Steps", border_style="cyan", padding=(1, 2)
    )
    console.print()
    console.print(steps_panel)

    # Permissions hint for .claude/ settings
    if selected_ai == "claude":
        claude_settings = project_path / ".claude" / "settings.json"
        permissions_hint = Panel(
            f"The template pre-configures [cyan].claude/settings.json[/cyan] with broad permissions "
            f"(e.g. [cyan]Bash[/cyan], [cyan]Write[/cyan], [cyan]Edit[/cyan]) so that Claude Code can run scripts and "
            f"modify files without repeated approval prompts.\n"
            f"These permissions may be more permissive than you need. "
            f"You can review and adjust them at any time by editing [cyan]{claude_settings.relative_to(project_path)}[/cyan].",
            title="[yellow]Pre-granted Permissions[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print()
        console.print(permissions_hint)

    # Initialise the private snapshot repo inside .cmind/.  Done BEFORE
    # the optional initial encode so the encoder's output, if it runs,
    # becomes a fresh commit on top of the [init] baseline — a useful
    # diff target.
    if not no_cmind_git:
        from . import _inner_git
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            ver = _pkg_version("cmind-cli")
        except PackageNotFoundError:
            ver = "dev"
        channel = "bundle"
        script_label = script_type if script_type else "sh"
        ai_label = selected_ai if selected_ai else "?"
        if _inner_git.ensure_inner_git(
            project_path,
            initial_msg=f"[init] v{ver} \u2014 {ai_label}/{script_label}, {channel} channel",
        ):
            console.print(
                "[dim]Inner snapshot repo initialised at "
                "[cyan]~/.cmind/workspaces/<workspace-id>/.git[/cyan] \u2014 "
                "run [cyan]cmind version[/cyan] for the exact path "
                "and a ready-to-paste `git -C` invocation.[/dim]"
            )

    # Final step: optionally build the initial RPG by running the
    # encoder.  Skipped silently for empty workspaces / non-tty / when
    # the user passes --no-encode.
    _maybe_offer_initial_encode(project_path, encode_choice=encode)


@app.command()
def update(
    ai_assistant: str = typer.Option(
        None,
        "--ai",
        help="AI assistant to use (auto-detected from existing project if not specified)",
    ),
    script_type: str = typer.Option(
        None, "--script", help="Script type to use: sh or ps"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show verbose diagnostic output",
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP server registration (rpg-tools won't be exposed to the AI agent)",
    ),
    no_copilot_cli_mcp: bool = typer.Option(
        False,
        "--no-copilot-cli-mcp",
        help=(
            "When --ai copilot is selected, skip also registering "
            "rpg-tools globally in ~/.copilot/mcp-config.json.  The "
            "Copilot CLI does not read workspace .vscode/mcp.json, so "
            "this global registration is what makes `copilot` find "
            "rpg-tools.  Pass this flag if you manage your Copilot CLI "
            "MCP config by hand."
        ),
    ),
    no_upgrade: bool = typer.Option(
        False,
        "--no-upgrade",
        help=(
            "Skip the default-on CLI self-upgrade step.  Use when offline, "
            "on a version-pinned CI runner, or when you've just "
            "installed the CLI manually."
        ),
    ),
    no_cmind_git: bool = typer.Option(
        False,
        "--no-cmind-git",
        help=(
            "Skip backfilling the private snapshot repo at .cmind/.git "
            "for older workspaces that don't have one yet.  Default is ON: "
            "if the inner repo is missing, `cmind update` creates it and "
            "commits a catch-up snapshot.  Pre-existing inner repos are "
            "never touched."
        ),
    ),
):
    """Update CoderMind template files in an existing project to the latest version.

    This command updates scripts, templates, command definitions, MCP
    config, gitignore rules, and git hooks in the current directory.
    It auto-detects the AI assistant from existing project configuration.

    Equivalent to re-running 'cmind init --here --force' but with proper
    semantics and automatic detection of existing settings.

    Examples:
        cmind update
        cmind update --ai claude
        cmind update --no-upgrade
    """
    show_banner()

    project_path = Path.cwd()

    # Verify this is an existing CoderMind project
    cmind_dir = project_path / ".cmind"
    if not cmind_dir.is_dir():
        console.print(
            Panel(
                "No [cyan].cmind/[/cyan] directory found in the current directory.\n"
                "This command updates an existing CoderMind project.\n\n"
                "To create a new project, use: [cyan]cmind init[/cyan]",
                title="[red]Not a CoderMind Project[/red]",
                border_style="red",
                padding=(1, 2),
            )
        )
        raise typer.Exit(1)

    # Determine AI assistant
    if ai_assistant:
        if ai_assistant not in AGENT_CONFIG:
            console.print(
                f"[red]Error:[/red] Invalid AI assistant '{ai_assistant}'. "
                f"Choose from: {', '.join(AGENT_CONFIG.keys())}"
            )
            raise typer.Exit(1)
        selected_ai = ai_assistant
    else:
        detected = _detect_ai_agent(project_path)
        if detected:
            console.print(
                f"[cyan]Auto-detected AI assistant:[/cyan] {detected} "
                f"({AGENT_CONFIG[detected]['name']})"
            )
            selected_ai = detected
        else:
            ai_choices = {key: config["name"] for key, config in AGENT_CONFIG.items()}
            selected_ai = select_with_arrows(
                ai_choices, "Choose your AI assistant:", "copilot"
            )

    # Determine script type
    if script_type:
        if script_type not in SCRIPT_TYPE_CHOICES:
            console.print(
                f"[red]Error:[/red] Invalid script type '{script_type}'. "
                f"Choose from: {', '.join(SCRIPT_TYPE_CHOICES.keys())}"
            )
            raise typer.Exit(1)
        # PowerShell support is planned but not yet wired into the
        # bundled templates / pipeline scripts.  Reject explicit
        # --script ps with a friendly message so users aren't surprised
        # by missing files later.
        if script_type == "ps":
            console.print(
                "[yellow]PowerShell (--script ps) is not yet supported and will "
                "be added in a future release. Please use --script sh for now.[/yellow]"
            )
            raise typer.Exit(1)
        selected_script = script_type
    else:
        # Default to sh on every platform until PowerShell templates land.
        default_script = "sh"
        if sys.stdin.isatty():
            selected_script = select_with_arrows(
                SCRIPT_TYPE_CHOICES,
                "Choose script type (or press Enter)",
                default_script,
            )
        else:
            selected_script = default_script

    console.print(f"[cyan]Selected AI assistant:[/cyan] {selected_ai}")
    console.print(f"[cyan]Selected script type:[/cyan] {selected_script}")

    # Pre-update CLI upgrade -------------------------------------------------
    #
    # By default, ``cmind update`` first runs the appropriate upgrade
    # command (``uv tool upgrade cmind-cli`` for uv installs etc.) so
    # the workspace's prompts/scripts/templates always match the
    # *latest* released version of the CLI.  Without this, users who
    # never re-install the CLI would silently drift behind upstream.
    #
    # We auto-upgrade only when:
    #   * the install method has a known upgrade command (uv, pipx, pip…)
    #     AND
    #   * the install source is remote (git URL or PyPI), meaning the
    #     user isn't actively developing the CLI from a local checkout.
    #
    # ``--no-upgrade`` skips this step (offline / pinned CI / freshly
    # re-installed manually).
    #
    # After a successful upgrade we ``os.execvp`` the (now-upgraded)
    # cmind binary so the rest of update runs against the freshly
    # installed code + assets.  Mixing old in-memory logic with new
    # on-disk core_pack/ used to cause logic vs assets drift bugs.
    #
    # Loop guard: ``CMIND_UPGRADE_DONE`` is set on the re-exec'd
    # process's environment.  When present, this block skips the
    # upgrade attempt unconditionally so an idempotent ``uv tool
    # upgrade`` (which returns 0 even when there's nothing to upgrade)
    # doesn't loop forever.
    _UPGRADE_DONE_ENV = "CMIND_UPGRADE_DONE"
    already_upgraded = bool(os.environ.get(_UPGRADE_DONE_ENV))

    method = _detect_install_method()
    source = _install_source()
    cmd = _upgrade_command(method)

    if already_upgraded:
        do_upgrade = False
        skip_reason = ""  # silent — internal marker, not user-visible
    elif no_upgrade:
        do_upgrade = False
        skip_reason = "--no-upgrade"
    elif cmd is None:
        do_upgrade = False
        skip_reason = (
            f"install method '{method}' has no auto-upgrade path "
            f"(upgrade manually)"
        )
    elif source not in _AUTO_UPGRADE_SOURCES:
        do_upgrade = False
        skip_reason = (
            f"local/dev install (source={source!r}); skipping auto-upgrade."
        )
    else:
        do_upgrade = True
        skip_reason = ""

    if do_upgrade:
        console.print(
            f"[cyan]Upgrading cmind-cli via {method} (source={source})...[/cyan]"
        )
        try:
            rc = subprocess.call(cmd)  # type: ignore[arg-type]
        except FileNotFoundError:
            # Upgrade tool (uv, pipx, pip) not on PATH — surface, then
            # carry on with the current build.  Stripping the upgrade
            # is a worse user experience than failing fast here would
            # be, but ``cmind update`` is "make my workspace match the
            # installed CLI", and the installed CLI is still functional.
            console.print(
                f"[yellow]Upgrade tool {cmd[0]!r} not found on PATH; "
                f"continuing with currently installed version.[/yellow]"
            )
            rc = -1
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]CLI upgrade raised an unexpected error "
                f"({type(exc).__name__}: {exc}); continuing with "
                f"currently installed version.[/yellow]"
            )
            rc = -1

        if rc == 0:
            # Re-exec the upgraded binary so the rest of update runs
            # against the freshly-installed code + assets.  Set the
            # loop-guard env var so the re-exec'd process doesn't
            # immediately try to upgrade again.
            new_argv = list(sys.argv)
            cmind_bin = shutil.which("cmind") or new_argv[0]
            console.print(
                "[cyan]CLI upgrade complete; re-exec'ing to apply "
                "new templates...[/cyan]"
            )
            try:
                os.environ[_UPGRADE_DONE_ENV] = "1"
                os.execvp(cmind_bin, [cmind_bin, *new_argv[1:]])
            except OSError as exc:
                # execvp failed — fall back to running the update
                # in-process with the (now-on-disk) new code.  This
                # mixes old in-memory logic with new assets, but
                # that's strictly better than crashing here: the user
                # already paid for the upgrade and wants the result.
                console.print(
                    f"[yellow]re-exec failed ({exc}); proceeding with "
                    f"in-process update.[/yellow]"
                )
                os.environ.pop(_UPGRADE_DONE_ENV, None)
        elif rc != -1:
            console.print(
                f"[yellow]CLI upgrade exited with code {rc}; "
                f"continuing with currently installed version.[/yellow]"
            )
    elif skip_reason:
        # Surface the reason only when the user explicitly opted out;
        # the default-on skip paths (editable, no upgrade cmd) stay
        # quiet for the 99% case where nothing to do.
        if skip_reason == "--no-upgrade":
            console.print(f"[dim]update: skipping CLI upgrade ({skip_reason}).[/dim]")

    # Build step tracker
    tracker = StepTracker("Update CoderMind Project")

    sys._cmind_tracker_active = True

    tracker.add("ai-select", "Select AI assistant")
    tracker.complete("ai-select", f"{selected_ai}")
    tracker.add("script-select", "Select script type")
    tracker.complete("script-select", selected_script)
    for key, label in [
        ("fetch", "Install bundled templates"),
        ("download", "Download template"),
        ("extract", "Extract template"),
        ("zip-list", "Archive contents"),
        ("extracted-summary", "Extraction summary"),
        ("chmod", "Ensure scripts executable"),
        ("gitignore", "Configure .gitignore"),
        ("mcp", "Configure MCP server"),
        ("copilot-cli-mcp", "Register rpg-tools in ~/.copilot/mcp-config.json"),
        ("hooks", "Install auto-update hooks"),
        ("cleanup", "Cleanup"),
        ("final", "Finalize"),
    ]:
        tracker.add(key, label)

    with Live(
        tracker.render(), console=console, refresh_per_second=8, transient=True
    ) as live:
        tracker.attach_refresh(lambda: live.update(tracker.render()))
        try:
            download_and_extract_template(
                project_path,
                selected_ai,
                selected_script,
                True,  # is_current_dir — always merge/overwrite for update
                verbose=False,
                tracker=tracker,
                debug=debug,
            )

            # .cmind/.source is written by whichever provisioning path
            # actually ran (_install_from_bundle / _download_and_extract_release_zip).

            # Refresh .cmind/config.toml only when missing (preserves
            # user customisations on re-update).
            _write_workspace_config(project_path, selected_ai)

            # Pre-create runtime directories so stage prompts that redirect
            # to ~/.cmind/workspaces/<workspace-id>/logs/<stage>.log don't fail when the folder is
            # missing (e.g. user removed it, or workspace was created by an
            # older cmind init that didn't pre-create logs/).
            ensure_cmind_runtime_dirs(project_path, tracker=tracker)

            # Ensure CoderMind gitignore rules are in place — re-runs are
            # idempotent (existing rules are detected and skipped) and this
            # also fixes workspaces created by older cmind versions that
            # didn't manage gitignore at all.
            tracker.start("gitignore")
            try:
                _setup_gitignore(project_path, selected_ai)
                tracker.complete("gitignore", "configured")
            except Exception as exc:
                tracker.error("gitignore", str(exc))

            # Generate/update MCP server configuration (unless explicitly skipped)
            if no_mcp:
                tracker.skip("mcp", "--no-mcp flag")
            else:
                _generate_mcp_config(project_path, selected_ai, tracker=tracker)

            # Global registration for Copilot CLI (see init for rationale).
            if no_mcp:
                pass
            elif selected_ai != "copilot":
                tracker.skip("copilot-cli-mcp", f"ai={selected_ai}")
            elif no_copilot_cli_mcp:
                tracker.skip("copilot-cli-mcp", "--no-copilot-cli-mcp flag")
            else:
                tracker.start("copilot-cli-mcp")
                _register_copilot_cli_global_mcp(tracker=tracker)

            # Reconcile hook files so existing workspaces receive the
            # current post-commit/post-merge dispatcher contract.
            _install_hooks(project_path, selected_ai, tracker=tracker)

            tracker.complete("final", "update complete")
        except Exception as e:
            tracker.error("final", str(e))
            console.print(
                Panel(
                    f"Update failed: {e}", title="Failure", border_style="red"
                )
            )
            if debug:
                _env_pairs = [
                    ("Python", sys.version.split()[0]),
                    ("Platform", sys.platform),
                    ("CWD", str(Path.cwd())),
                ]
                _label_width = max(len(k) for k, _ in _env_pairs)
                env_lines = [
                    f"{k.ljust(_label_width)} → [bright_black]{v}[/bright_black]"
                    for k, v in _env_pairs
                ]
                console.print(
                    Panel(
                        "\n".join(env_lines),
                        title="Debug Environment",
                        border_style="magenta",
                    )
                )
            raise typer.Exit(1)
        finally:
            pass

    console.print(tracker.render())
    console.print(
        "\n[bold green]CoderMind templates updated successfully.[/bold green]"
    )
    console.print(
        f"[dim]Updated: scripts, templates, and {AGENT_CONFIG[selected_ai]['name']} "
        f"command definitions in [cyan]{project_path}[/cyan][/dim]"
    )

    # Backfill inner snapshot repo for workspaces created before
    # this feature shipped.  Idempotent — does nothing if .cmind/.git
    # already exists, and silently noops if --no-cmind-git was passed.
    if not no_cmind_git:
        from . import _inner_git
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            ver = _pkg_version("cmind-cli")
        except PackageNotFoundError:
            ver = "dev"
        if _inner_git.ensure_inner_git(
            project_path,
            initial_msg=f"[update] v{ver} \u2014 catch-up snapshot",
        ):
            console.print(
                "[dim]Initialised inner snapshot repo at "
                "[cyan]~/.cmind/workspaces/<workspace-id>/.git[/cyan] for this workspace.[/dim]"
            )


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        # Disable click's auto-help so ``--help`` is forwarded to the
        # target script.  Use ``cmind script`` (no args) or
        # ``cmind --help script`` to see this command's own help.
        "help_option_names": [],
    },
)
def script(
    ctx: typer.Context,
    relpath: Optional[str] = typer.Argument(
        None,
        help="Script path relative to the packaged scripts directory "
        "(e.g. 'smoke_test.py' or 'rpg_edit/validate.py'). "
        "The '.py' suffix is optional.",
    ),
    list_all: bool = typer.Option(
        False,
        "--list",
        help="List all available scripts and exit.",
    ),
    where: Optional[str] = typer.Option(
        None,
        "--where",
        metavar="NAME",
        help="Print the absolute filesystem path of NAME and exit.",
    ),
) -> None:
    """Execute a bundled CoderMind pipeline script.

    All arguments after ``<relpath>`` are forwarded verbatim to the
    target script.  Standard input/output/error are inherited so the
    child's behaviour matches direct invocation.

    Examples::

        cmind script smoke_test.py --json
        cmind script rpg_edit/validate.py
        cmind script --list
        cmind script --where mcp_server.py
    """
    from . import _assets

    if list_all:
        for name in _assets.list_scripts():
            console.print(name)
        raise typer.Exit(0)

    if where is not None:
        path = _resolve_script_path(where)
        if path is None:
            console.print(f"[red]script not found: {where}[/red]")
            raise typer.Exit(1)
        # Print plain path (no markup) so it pipes cleanly into $(...)
        print(str(path))
        raise typer.Exit(0)

    if not relpath:
        console.print(
            "[red]error:[/red] missing script path. "
            "Use [cyan]cmind script --list[/cyan] to see available scripts."
        )
        raise typer.Exit(2)

    path = _resolve_script_path(relpath)
    if path is None:
        console.print(f"[red]script not found: {relpath}[/red]")
        raise typer.Exit(1)

    # Build child env: inherit, plus disable .pyc writes so the read-mostly
    # tool-venv install dir doesn't accumulate __pycache__ noise.
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Force UTF-8 stdio in the child regardless of the host's locale.  On
    # Windows, a non-tty stdout (always true here: we either pipe it or it's
    # inherited into another pipe/redirect) makes CPython fall back to
    # locale.getpreferredencoding(), which is a legacy code page (cp1252,
    # cp936, ...) on most Windows installs. Any bundled script that prints
    # a non-ASCII character (e.g. update_graphs.py's "branch changed: 'a' →
    # 'b'") then raises UnicodeEncodeError and crashes outright instead of
    # completing. errors="replace" additionally protects the reverse
    # direction (decoding non-UTF-8 bytes on stdin) from crashing.
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("PYTHONUTF8", "1")

    # Tee stdout to a per-stage log file so the workspace has a persistent
    # record of every script invocation.  The log path is resolved from
    # _storage at run time; if the home-side dir doesn't exist yet (e.g.
    # cmind init hasn't run), skip silently — no log is better than
    # crashing.
    log_path: Optional[Path] = None
    from . import _inner_git as _ig
    ws_root = _ig.find_workspace_root()
    if ws_root is not None:
        from . import _storage
        logs_dir = _storage.workspace_logs_dir(ws_root)
        if logs_dir.is_dir():
            script_stem = path.stem  # e.g. "feature_build"
            log_path = logs_dir / f"{script_stem}.log"

    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8")
        cmd = [sys.executable, str(path), *ctx.args]
        proc = subprocess.run(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        # Write captured output to both terminal and log file.
        output = proc.stdout or b""
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        try:
            log_fh.write(output.decode("utf-8", errors="replace"))
            log_fh.flush()
        except OSError:
            pass
        finally:
            log_fh.close()
    else:
        cmd = [sys.executable, str(path), *ctx.args]
        proc = subprocess.run(cmd, env=env)

    # Snapshot the current state of .cmind/ into the inner git
    # repo so users can `git log` / `git diff` between pipeline stages.
    # No-op (silently) when the script is read-only (check_*, *_validation),
    # the inner repo is absent (--no-cmind-git on init), or git is busy.
    #
    # Use the *resolved* path (always carries .py) for the commit message
    # so `cmind script smoke_test` and `cmind script smoke_test.py`
    # produce identical history entries.
    from . import _inner_git, _assets
    ws_root = _inner_git.find_workspace_root()
    if ws_root is not None:
        try:
            commit_relpath = str(path.relative_to(_assets.scripts_dir())).replace("\\", "/")
        except ValueError:
            commit_relpath = relpath.replace("\\", "/")
        _inner_git.auto_commit_after_script(
            ws_root,
            commit_relpath,
            list(ctx.args),
            proc.returncode,
        )

    raise typer.Exit(proc.returncode)


def _resolve_script_path(relpath: str) -> Optional[Path]:
    """Resolve ``relpath`` against the packaged scripts dir.

    Rejects path-traversal and absolute paths; appends ``.py`` when no
    suffix is given.  Returns ``None`` if the resolved path is not a
    regular file inside :func:`_assets.scripts_dir`.
    """
    from . import _assets

    # Normalise separators for cross-platform invocation
    rel = relpath.replace("\\", "/")
    # Security: reject parent-traversal and absolute paths
    if rel.startswith("/") or ".." in rel.split("/"):
        return None
    p = Path(rel)
    if p.is_absolute():
        return None
    if p.suffix == "":
        p = p.with_suffix(".py")
    root = _assets.scripts_dir()
    candidate = (root / p).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        # Resolved outside the scripts root (e.g. via symlink) — refuse
        return None
    if not candidate.is_file():
        return None
    return candidate


# ---------------------------------------------------------------------------
# Git-hook dispatch: ``cmind hook <name>``
# ---------------------------------------------------------------------------
#
# Python entry-point for git hooks.  The on-disk hook files in
# ``.git/hooks/`` are short shell stubs that ``exec`` this command;
# path resolution, logging, locking, and detach logic live here so they
# can be updated by upgrading the CLI rather than reinstalling hooks.

_HOOK_ENV_NAME = "CMIND_HOOK"
_HOOK_ENV_SHA = "CMIND_HOOK_SHA"
_HOOK_LOG_FILENAME = "hooks.log"
_HOOK_BACKGROUND_LOG = "update_rpg.log"
_HOOK_LOCK_DIRNAME = ".update_rpg.lock"
_HOOK_LOCK_STALE_SECONDS = 60 * 60  # 60 minutes -- matches the old shell impl


def _hook_log_line(log_path: Path, msg: str) -> None:
    """Append a timestamped line to the hook log.  Best-effort."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            fh.write(f"[{ts}] {msg}\n")
    except OSError:
        # Logging is observability; we never fail a hook because we
        # couldn't write a line.
        pass


def _short_head_sha(workspace: Path) -> str:
    """Return ``git rev-parse --short HEAD`` for ``workspace`` or ``"?"``."""
    try:
        r = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or "?"
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"


def _hook_run_foreground(
    workspace: Path,
    log_path: Path,
    env: Dict[str, str],
    script_args: List[str],
    label: str,
) -> int:
    """Run ``cmind script <script_args>`` and tee output into ``log_path``."""
    _hook_log_line(log_path, f"{label}: start ({' '.join(script_args)})")
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            proc = subprocess.run(
                ["cmind", "script", *script_args],
                cwd=str(workspace),
                env=env,
                stdout=fh, stderr=subprocess.STDOUT,
                timeout=300,
            )
        _hook_log_line(log_path, f"{label}: done (exit {proc.returncode})")
        return proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        _hook_log_line(log_path, f"{label}: ERROR {exc!r}")
        return -1


def _hook_spawn_background(
    workspace: Path,
    home_dir: Path,
    hook_log: Path,
    env: Dict[str, str],
) -> None:
    """Acquire a directory lock and detach ``update_graphs.py update-rpg``.

    The lock is a *directory* (``mkdir`` is the only POSIX-atomic
    exclusive-create primitive); a directory older than
    :data:`_HOOK_LOCK_STALE_SECONDS` is treated as orphaned (worker
    killed by OOM / reboot / SIGKILL) and removed before re-trying.
    """
    lock_dir = home_dir / "logs" / _HOOK_LOCK_DIRNAME
    bg_log = home_dir / "logs" / _HOOK_BACKGROUND_LOG

    # Stale-lock recovery -- match the 60-minute window the shell hook used.
    try:
        if lock_dir.is_dir():
            age = time.time() - lock_dir.stat().st_mtime
            if age > _HOOK_LOCK_STALE_SECONDS:
                shutil.rmtree(lock_dir, ignore_errors=True)
                _hook_log_line(hook_log, f"phase2: removed stale lock (age={age:.0f}s)")
    except OSError:
        pass

    # Try to acquire.
    try:
        lock_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        _hook_log_line(hook_log, "phase2: skipped (another worker holds the lock)")
        return
    except OSError as exc:
        _hook_log_line(hook_log, f"phase2: lock acquire failed: {exc!r}")
        return

    # Background worker: run update-rpg, then release the lock.  We
    # cannot use ``Popen`` alone because nothing would ``rmdir`` the
    # lock after the worker completes; a tiny ``sh -c`` wrapper does
    # the cleanup deterministically.
    #
    # ``start_new_session=True`` is the cross-platform equivalent of
    # ``nohup``/``setsid`` -- the child survives the hook's exit.
    bg_log.parent.mkdir(parents=True, exist_ok=True)
    lock_q = shlex.quote(str(lock_dir))
    log_q = shlex.quote(str(bg_log))
    workspace_q = shlex.quote(str(workspace))
    shell_cmd = (
        f"cd {workspace_q}; sleep 2; "
        f"cmind script update_graphs.py update-rpg --json >> {log_q} 2>&1; "
        f"rmdir {lock_q}"
    )
    # Strip GIT_INDEX_FILE / GIT_DIR which git sets during hooks -
    # if they leak into the worker, ``git worktree add`` fails with
    # cryptic index errors.
    worker_env = {k: v for k, v in env.items() if k not in ("GIT_INDEX_FILE", "GIT_DIR")}
    try:
        subprocess.Popen(
            ["sh", "-c", shell_cmd],
            cwd=str(workspace),
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _hook_log_line(hook_log, f"phase2: dispatched -> {bg_log}")
    except OSError as exc:
        _hook_log_line(hook_log, f"phase2: spawn failed: {exc!r}")
        # Release the lock so the next commit can retry.
        try:
            lock_dir.rmdir()
        except OSError:
            pass


@app.command(
    "hook",
    hidden=True,
    help="Internal git-hook dispatcher (called from .git/hooks/*).",
)
def hook(name: str = typer.Argument(..., help="Hook name: post-commit | post-merge")) -> None:
    """Dispatch from ``.git/hooks/<name>`` to the matching Python handler.

    Resolves the current workspace via the standard cwd-walk, attaches
    a hook log under ``~/.cmind/workspaces/<workspace-id>/logs/hooks.log``,
    and runs the per-hook orchestration.  Every failure path is
    swallowed (logged, never raised) so a misbehaving hook never blocks
    the user's git operation.

    Supported hooks: ``post-commit`` and ``post-merge``. The dispatcher
    also accepts ``pre-commit`` as a deliberate no-op for backward
    compatibility — old workspaces whose hook file still calls
    ``cmind hook pre-commit`` should be cleaned up on the next
    ``cmind init`` / ``cmind update`` run, which strips the block.

    All ``cmind script`` subprocess invocations inherit two env vars:

      * ``CMIND_HOOK`` -- the hook name (``post-commit`` etc.)
      * ``CMIND_HOOK_SHA`` -- short SHA of the user-facing commit

    The inner-git snapshot's commit message picks these up
    (:func:`cmind_cli._inner_git._build_message`) so ``git log`` in the
    home-side repo reads as a timeline of *user activity*, e.g.::

        [hook:post-commit @ a1b2c3d] sync
        [hook:post-merge  @ 9f8e7d6] sync
    """
    from . import _storage

    try:
        ws = _storage.find_workspace_root_from(Path.cwd())
        if ws is None:
            # Not in an cmind workspace -- silently exit success;
            # the hook may be running in a repo that was provisioned
            # then un-init'd, and we never want to block git.
            raise typer.Exit(0)

        home_dir = _storage.home_workspace_dir(ws)
        log_path = _storage.workspace_logs_dir(ws) / _HOOK_LOG_FILENAME
        sha = _short_head_sha(ws)

        env = os.environ.copy()
        env[_HOOK_ENV_NAME] = name
        env[_HOOK_ENV_SHA] = sha
        # Ensure ``cmind`` itself is on PATH when the hook is fired
        # from a GUI editor that lacks the user's interactive shell PATH.
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")

        _hook_log_line(log_path, f"== {name} fired @ {sha} (ws={ws})")

        if name == "pre-commit":
            # Retired: pre-commit is now a deliberate no-op for backward
            # compatibility with workspaces whose stub hasn't been
            # stripped yet. Just log and exit success so git proceeds.
            _hook_log_line(log_path, "pre-commit hook is a no-op (retired)")
        elif name == "post-merge":
            _hook_run_foreground(
                ws, log_path, env,
                ["update_graphs.py", "sync"],
                "sync",
            )
        elif name == "post-commit":
            # Fast foreground sync keeps meta.git aligned with HEAD.
            _hook_run_foreground(
                ws, log_path, env,
                ["update_graphs.py", "sync"],
                "foreground-sync",
            )
            # The LLM-driven RPG update runs detached from git commit.
            _hook_spawn_background(ws, home_dir, log_path, env)
        else:
            _hook_log_line(log_path, f"unknown hook name: {name!r}")
            raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as exc:
        # Last-ditch swallow: anything reaching here means our hook
        # dispatcher itself is broken, but a broken hook must not
        # break ``git commit`` -- log and exit cleanly.
        try:
            ws = _storage.find_workspace_root_from(Path.cwd())
            if ws is not None:
                _hook_log_line(
                    _storage.workspace_logs_dir(ws) / _HOOK_LOG_FILENAME,
                    f"FATAL in hook dispatcher: {exc!r}",
                )
        except Exception:
            pass
        raise typer.Exit(0)

    raise typer.Exit(0)


@app.command()
def check():
    """Check that all required tools are installed."""
    show_banner()
    console.print("[bold]Checking for installed tools...[/bold]\n")

    tracker = StepTracker("Check Available Tools")

    tracker.add("git", "Git version control")
    git_ok = check_tool("git", tracker=tracker)

    agent_results = {}
    for agent_key, agent_config in AGENT_CONFIG.items():
        agent_name = agent_config["name"]
        requires_cli = agent_config["requires_cli"]

        tracker.add(agent_key, agent_name)

        if requires_cli:
            agent_results[agent_key] = check_tool(agent_key, tracker=tracker)
        else:
            # IDE-based agent - skip CLI check and mark as optional
            tracker.skip(agent_key, "IDE-based, no CLI check")
            agent_results[agent_key] = False  # Don't count IDE agents as "found"

    # Check VS Code variants (not in agent config)
    tracker.add("code", "Visual Studio Code")

    tracker.add("code-insiders", "Visual Studio Code Insiders")

    console.print(tracker.render())

    console.print("\n[bold green]CoderMind CLI is ready to use![/bold green]")

    if not git_ok:
        console.print("[dim]Tip: Install git for repository management[/dim]")

    if not any(agent_results.values()):
        console.print("[dim]Tip: Install an AI assistant for the best experience[/dim]")


@app.command()
def version():
    """Display version and system information.

    Also fetches the latest release tag from GitHub and reports whether
    the locally installed CLI is up to date, behind, or ahead (dev
    build).  Network failures are swallowed and surface as "offline".
    """
    show_banner()

    # Get CLI version from package metadata
    cli_version = "unknown"
    try:
        cli_version = importlib.metadata.version("cmind-cli")
    except Exception:
        # Fallback: try reading from pyproject.toml if running from source
        try:

            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    cli_version = data.get("project", {}).get("version", "unknown")
        except Exception:
            pass

    # Fetch latest template release version
    repo_owner, repo_name = _get_repo_info()

    latest_version = "unknown"
    release_date = "unknown"
    fetch_error: str | None = None

    try:
        release_data = _fetch_latest_cmind_release(
            repo_owner,
            repo_name,
            client,
            timeout=10,
        )
        latest_version = _format_cmind_version(release_data.get("tag_name", "unknown"))
        release_date = release_data.get("published_at", "unknown")
        if release_date != "unknown":
            # Format the date nicely
            try:
                dt = datetime.fromisoformat(release_date.replace("Z", "+00:00"))
                release_date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    except Exception as exc:
        fetch_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__

    # ------------------------------------------------------------------
    # Compute the status hint: up-to-date / outdated / ahead / offline.
    # Uses ``packaging.version`` (stdlib-ish — ships with setuptools and
    # is a transitive dep of pip itself) so PEP 440 pre-release / dev
    # suffixes are compared correctly.  Falls back to a plain string
    # comparison when ``packaging`` is unavailable.
    # ------------------------------------------------------------------
    status_label = "[dim]unknown[/dim]"
    status_hint: str | None = None

    if fetch_error is not None:
        status_label = "[yellow]offline[/yellow]"
        status_hint = (
            f"Could not query GitHub for the latest release: {fetch_error}. "
            "Local install is still usable; rerun `cmind version` when "
            "you have network access to compare."
        )
    elif cli_version != "unknown" and latest_version != "unknown":
        try:
            from packaging.version import Version as _Ver

            local_v = _Ver(cli_version)
            remote_v = _Ver(latest_version)
        except Exception:
            local_v = cli_version
            remote_v = latest_version

        if local_v == remote_v:
            status_label = "[green]up to date[/green]"
        elif local_v < remote_v:
            status_label = f"[yellow]outdated → {latest_version}[/yellow]"
            status_hint = (
                f"A newer release ([cyan]{latest_version}[/cyan]) is "
                f"available.  Upgrade with one of:\n"
                f"  [cyan]uv tool upgrade cmind-cli[/cyan]\n"
                f"  [cyan]pipx upgrade cmind-cli[/cyan]\n"
                f"  [cyan]pip install -U cmind-cli[/cyan]\n"
                f"After upgrading, run [cyan]cmind update[/cyan] in each "
                f"existing workspace to apply the new prompts."
            )
        else:
            status_label = f"[cyan]ahead of release ({latest_version})[/cyan]"
            status_hint = (
                f"Local CLI ({cli_version}) is newer than the latest "
                f"published release ({latest_version}) — typically a dev "
                f"build from git.  No action needed."
            )

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="cyan", justify="right")
    info_table.add_column("Value", style="white")

    info_table.add_row("CLI Version", cli_version)
    info_table.add_row("Latest Release", latest_version)
    info_table.add_row("Released", release_date)
    info_table.add_row("Status", status_label)
    info_table.add_row("", "")
    info_table.add_row("Python", platform.python_version())
    info_table.add_row("Platform", platform.system())
    info_table.add_row("Architecture", platform.machine())
    info_table.add_row("OS Version", platform.version())

    # Surface the per-workspace home-side storage when
    # invoked from inside an cmind workspace.  Without this the user
    # has no obvious way to find their generated artefacts / logs after
    # we moved them out of the repo tree into ``~/.cmind/workspaces/
    # <workspace-id>/`` — they'd have to derive the workspace id themselves.
    try:
        from . import _inner_git
        ws = _inner_git.find_workspace_root()
        if ws is not None:
            home_dir = _storage.home_workspace_dir(ws)
            data_dir = _storage.workspace_data_dir(ws)
            logs_dir = _storage.workspace_logs_dir(ws)
            # Annotate each row when the dir doesn't exist yet so the
            # user doesn't mistake a computed path for a real artefact.
            # Important after partial cleanup or before the first
            # ``cmind init`` populates the home-side store — we used
            # to print non-existent paths as if they were live.
            def _tag(p: Path) -> str:
                return str(p) if p.exists() else f"{p}  [dim](not created yet)[/dim]"

            info_table.add_row("", "")
            info_table.add_row("Workspace", str(ws))
            info_table.add_row("Data", _tag(data_dir))
            info_table.add_row("Logs", _tag(logs_dir))
            # Inner-git: distinguish absent (no .git dir) from empty
            # (.git exists but zero commits).  snapshot_count returns
            # None for both, so probe has_inner_git directly.
            if not home_dir.exists():
                inner_git_value = f"{home_dir}  [dim](home-side dir not created — run `cmind init` here)[/dim]"
            elif not _inner_git.has_inner_git(ws):
                inner_git_value = f"{home_dir}  [dim](no inner-git repo)[/dim]"
            else:
                count = _inner_git.snapshot_count(ws)
                if count is None or count == 0:
                    inner_git_value = f"{home_dir}  [dim](no snapshots yet)[/dim]"
                else:
                    inner_git_value = f"{home_dir}  [dim]({count} snapshots — git -C {home_dir} log)[/dim]"
            info_table.add_row("Inner git", inner_git_value)
    except Exception:
        pass

    panel = Panel(
        info_table,
        title="[bold cyan]CoderMind CLI Information[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    if status_hint:
        console.print()
        console.print(
            Panel(
                status_hint,
                title="[bold]Upgrade tip[/bold]",
                border_style="yellow"
                if "outdated" in status_label or "offline" in status_label
                else "cyan",
                padding=(1, 2),
            )
        )
    console.print()


def main():
    app()


if __name__ == "__main__":
    main()
