"""RPG-Kit CLI - Setup tool for RPG-Kit projects.

Usage:
    uvx rpgkit-cli init <project-name>
    uvx rpgkit-cli init .
    uvx rpgkit-cli init --here

Or install globally:
    uv tool install rpgkit-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=RPG-Kit"
    rpgkit init <project-name>
    rpgkit init .
    rpgkit init --here
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

ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
client = httpx.Client(verify=ssl_context)

# Default fallback values — only used when git remote and pyproject.toml are unavailable
_FALLBACK_REPO_OWNER = "microsoft"
_FALLBACK_REPO_NAME = "RPG-ZeroRepo"
_RPGKIT_RELEASE_TAG_PREFIX = "rpgkit-v"


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
    """Resolve the GitHub owner/repo for RPG-Kit template downloads.

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

# ── Default .gitignore template ──────────────────────────────────────────
# Split into three parts so init can compose the right output depending on
# project state:
#   * PYTHON template  → written *only* when both .git/ and .gitignore are
#                         absent (greenfield), so we don't impose Python
#                         conventions on an existing repo that already has
#                         its own .gitignore preferences.
#   * RPGKIT_COMMON    → always injected — these files MUST be ignored
#                         (runtime data, machine-specific config).
#   * RPGKIT_AI[ai]    → always injected for the selected AI assistant —
#                         RPG-Kit regenerates slash command files on every
#                         `rpgkit init/update`, so they are build artifacts,
#                         not source.
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

_GITIGNORE_RPGKIT_HEADER = "# RPG-Kit ignores (managed by `rpgkit init/update`)"

_GITIGNORE_RPGKIT_COMMON = """\
# Runtime workspace (logs, generated data, trajectory)
.rpgkit/

# RPG-Kit Python environment
.venv_rpgkit/

# Codegen dev environments
.venv_dev/
.rpgkit_dev_env/

# Machine-specific config (absolute interpreter paths)
.vscode/mcp.json
.vscode/tasks.json
.mcp.json
"""

# AI-specific slash-command directories that RPG-Kit regenerates each time
# `rpgkit init/update` runs. We deliberately scope each entry to a sub-
# directory rather than the whole agent folder so unrelated assets in
# ``.github/`` (workflows, CODEOWNERS, …) or ``.claude/`` (settings.json
# with team-shared permissions) remain trackable.
_GITIGNORE_RPGKIT_AI = {
    "copilot": """\
# Copilot slash command definitions (regenerated by rpgkit)
.github/agents/
.github/prompts/
""",
    "claude": """\
# Claude Code slash command definitions (regenerated by rpgkit)
.claude/commands/
""",
}

BANNER = """
██████╗ ██████╗  ██████╗       ██╗  ██╗██╗████████╗
██╔══██╗██╔══██╗██╔════╝       ██║ ██╔╝██║╚══██╔══╝
██████╔╝██████╔╝██║  ███╗█████╗█████╔╝ ██║   ██║   
██╔══██╗██╔═══╝ ██║   ██║╚════╝██╔═██╗ ██║   ██║   
██║  ██║██║     ╚██████╔╝      ██║  ██╗██║   ██║   
╚═╝  ╚═╝╚═╝      ╚═════╝       ╚═╝  ╚═╝╚═╝   ╚═╝   
"""

TAGLINE = "RPG-Kit Plugin - LLM-based Automated Code Generation System Toolkit"


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
    name="rpgkit",
    help="Setup tool for RPG-Kit feature tree generation projects",
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
            Align.center("[dim]Run 'rpgkit --help' for usage information[/dim]")
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
    """Materialize ``.gitignore`` with RPG-Kit's required rules.

    This is the **single injection point** for all RPG-Kit gitignore
    management.  Other init steps (``_generate_mcp_config``,
    ``_install_copilot_hooks``) MUST NOT modify ``.gitignore``
    themselves — all rules they used to inject have been folded into
    ``_GITIGNORE_RPGKIT_COMMON`` / ``_GITIGNORE_RPGKIT_AI``.

    Behavior (decided by the user via interactive design review):

    * **Greenfield** — both ``.git/`` and ``.gitignore`` are absent:
      write Python standard template + RPG-Kit common + AI-specific
      rules.  Gives new projects a complete, sensible default.

    * **Existing repo or existing ``.gitignore``** — *do not* overwrite
      the user's Python conventions.  Only append RPG-Kit rules
      (deduplicated by exact line match) under a single
      ``# RPG-Kit ignores`` header.

    Args:
        project_path: Project root that may or may not be a git repo.
        selected_ai:  ``"copilot"`` or ``"claude"`` — selects which AI
                      slash-command directories to ignore.
    """
    gitignore = project_path / ".gitignore"
    git_dir = project_path / ".git"

    rpgkit_block = _GITIGNORE_RPGKIT_COMMON
    ai_rules = _GITIGNORE_RPGKIT_AI.get(selected_ai)
    if ai_rules:
        rpgkit_block += "\n" + ai_rules

    # Greenfield: brand-new project, no git, no existing .gitignore.
    # Lay down the full template (Python conventions + RPG-Kit rules).
    if not git_dir.exists() and not gitignore.exists():
        gitignore.write_text(
            _GITIGNORE_PYTHON_TEMPLATE
            + "\n"
            + _GITIGNORE_RPGKIT_HEADER
            + "\n"
            + rpgkit_block,
            encoding="utf-8",
        )
        return

    # Brownfield: respect the user's existing setup, only ensure RPG-Kit
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

    # Collect RPG-Kit pattern lines (skip comments and blanks in the
    # block — comments are kept for the appended section but not used
    # for dedup checks).
    missing_lines: list[str] = []
    for line in rpgkit_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _norm(stripped) not in existing_norm:
            missing_lines.append(stripped)

    if not missing_lines:
        return

    # Append under a single, idempotent RPG-Kit header so repeated runs
    # don't create duplicate section markers.
    parts: list[str] = []
    if existing_text and not existing_text.endswith("\n"):
        parts.append("\n")
    if existing_text:
        parts.append("\n")
    if _GITIGNORE_RPGKIT_HEADER not in existing_text:
        parts.append(_GITIGNORE_RPGKIT_HEADER + "\n")
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
            ["git", "commit", "-m", "Initial commit from RPG-Kit template"],
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

    Earlier versions of ``rpgkit init`` registered the MCP server inside
    ``settings.json``.  We've since moved to ``.vscode/mcp.json``; this
    helper deletes only the stale entry so users upgrading via
    ``rpgkit update`` don't end up with two registrations.

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


def _cleanup_legacy_codegen_persistent(project_path: Path) -> list[str]:
    """Delete obsolete ``rpgkit-codegen.*`` persistent-instruction files.

    Earlier versions of ``rpgkit init`` (pre-C4 cleanup) wrote a
    codegen-specific instructions file that AI agents would auto-load on
    every session, polluting unrelated commands (rpg_edit, encode, plain
    Q&A) with codegen workflow noise.  See ``plans/20260508-1-rpgkit-
    optimization*.md`` § C4.

    This helper:

    * Removes ``<project>/.claude/rules/rpgkit-codegen.md``
    * Removes ``<project>/.github/instructions/rpgkit-codegen.instructions.md``
    * Also cleans the legacy ``<project>/repo/.claude/...`` and
      ``<project>/repo/.github/...`` paths, so workspaces created
      under the old ``<workspace>/repo`` layout are upgraded on the
      next ``rpgkit init`` / ``rpgkit update`` run.
    * Tidies up empty parent directories the file leaves behind.
    * Returns the list of paths actually removed (for tracker reporting).

    The function is safe to call repeatedly and on workspaces that never
    had the legacy file.
    """
    legacy_repo_dir = project_path / "repo"
    candidates = [
        # New layout (workspace == repo)
        project_path / ".claude" / "rules" / "rpgkit-codegen.md",
        project_path / ".github" / "instructions" / "rpgkit-codegen.instructions.md",
        # Legacy layout (<workspace>/repo) — keep scanning so users who
        # upgrade from old workspaces still get the file removed.
        legacy_repo_dir / ".claude" / "rules" / "rpgkit-codegen.md",
        legacy_repo_dir / ".github" / "instructions" / "rpgkit-codegen.instructions.md",
    ]

    removed: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed.append(str(path.relative_to(project_path)))
        except OSError:
            continue

        # Tidy up empty parent dirs (only if the parent contains nothing
        # else; we never delete user-owned content).
        parent = path.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    return removed


def _generate_mcp_config(
    project_path: Path,
    selected_ai: str,
    tracker=None,
) -> None:
    """Generate MCP server configuration for the selected AI assistant.

    Both Claude and VS Code Copilot launch the MCP server via the current
    Python interpreter (``sys.executable``) running
    ``<project>/.rpgkit/scripts/mcp_server.py`` — this guarantees the
    interpreter that has ``rpgkit-cli``'s dependencies (mcp, rapidfuzz, …)
    installed is used to host the server.

    - Claude:  ``.mcp.json``         (key ``mcpServers.rpg-tools``)
    - Copilot: ``.vscode/mcp.json``  (key ``servers.rpg-tools``,
      VS Code 1.102+ standard layout)

    Generated paths are absolute and machine-specific; the corresponding
    files are ignored via :func:`_setup_gitignore` (called earlier in the
    init flow), not by this function.
    """
    # Resolve absolute paths up-front so we never write a stale/relative path.
    project_path = project_path.resolve()
    server_script = (project_path / ".rpgkit" / "scripts" / "mcp_server.py").resolve()

    if not server_script.is_file():
        # Should not happen — extraction step runs before us — but bail out
        # cleanly instead of writing a config that would fail at runtime.
        msg = f"mcp_server.py not found at {server_script}"
        if tracker:
            tracker.error("mcp", msg)
        else:
            console.print(f"[yellow]Warning: {msg}[/yellow]")
        return

    mcp_server_config = {
        "command": sys.executable,
        "args": [str(server_script)],
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
            #
            # We deliberately do NOT write a ``sandbox`` block here.  VS
            # Code's MCP sandbox requires ``bubblewrap`` (bwrap) and
            # ``socat`` on PATH; most Linux desktops, WSL, minimal Docker
            # images and fresh macOS installs lack these, causing the
            # server to crash on startup with the opaque ``Connection
            # closed`` error.  The only thing sandbox gained us was
            # auto-approving tool confirmations — a one-click setting in
            # VS Code's MCP UI ("Always allow this server") covers the
            # same UX without the dependency landmine.  RPG-Kit's MCP
            # server is also read-only and offline, so sandbox added no
            # security value.
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
# Optional initial encode
# ---------------------------------------------------------------------------

def _workspace_has_python_code(project_path: Path) -> bool:
    """Return True if the workspace contains any ``*.py`` file outside ``.rpgkit/``.

    Used to decide whether ``rpgkit init`` should offer to build the RPG
    immediately.  Greenfield workspaces (or repos that don't ship Python
    code) skip the prompt because the encoder would produce an empty
    graph and waste LLM tokens.

    The walk prunes the ``.rpgkit`` directory in-place so we don't
    accidentally count the runtime scripts we just extracted (every
    workspace has ``.rpgkit/scripts/*.py`` after init).  Common
    boilerplate dirs (``.git``, ``.venv``, ``node_modules``,
    ``__pycache__``) are pruned too — a ``*.py`` under any of them
    would not indicate user code.
    """
    PRUNE = {".rpgkit", ".git", ".venv", ".venv_rpgkit", "venv", "node_modules",
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
_ENCODE_RE_TOTAL_FILES = re.compile(r"Total valid Python files to parse:\s*(\d+)")
_ENCODE_RE_CLASS_BATCHES = re.compile(r"\[GLOBAL\] kind=class,\s*groups=\d+,\s*batches=(\d+)")
_ENCODE_RE_FUNC_BATCHES = re.compile(r"\[GLOBAL\] kind=function,\s*groups=\d+,\s*batches=(\d+)")
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
        ``Total valid Python files to parse: N``
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
    if "process_class_batch:" in line:
        state["kind"] = "class"
        state["phase"] = "Parsing class batches"
        return
    if _ENCODE_RE_CLASS_FINISHED.search(line):
        state["class_done"] += 1
        if state.get("class_total"):
            state["class_done"] = min(state["class_done"], state["class_total"])
        state["kind"] = "class"
        state["phase"] = "Parsing class batches"
        return
    if "process_func_batch:" in line:
        state["kind"] = "function"
        state["phase"] = "Parsing function batches"
        return
    if _ENCODE_RE_FUNC_FINISHED.search(line):
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
        ``.rpgkit/logs/encode.log`` — power users can ``tail -f`` it
        for the full firehose.
      * Parse a handful of phase markers off each line to drive a
        :class:`rich.progress.Progress` bar with a spinner + current
        phase + (when known) an M/N batch counter.
      * Capture stdout and surface the encoder's JSON summary on
        failure so the user has something concrete to debug.

    Returns True on success (exit code 0), False otherwise.  Never
    raises: ``rpgkit init`` itself has already succeeded by the time we
    get here and we don't want a flaky LLM call to make the whole
    command look like it failed.
    """
    encoder = project_path / ".rpgkit" / "scripts" / "rpg_encoder" / "run_encode.py"
    if not encoder.is_file():
        console.print(
            f"[yellow]Encoder script not found at {encoder}; "
            f"run [cyan]/rpgkit.encode[/] in your AI agent later.[/yellow]"
        )
        return False

    log_dir = project_path / ".rpgkit" / "logs"
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
            "Building [cyan].rpgkit/data/rpg.json[/] from your code via the "
            "LLM.  Verbose logs stream to [cyan].rpgkit/logs/encode.log[/] — "
            "`tail -f` it in another terminal for the gory details.  "
            "Press Ctrl-C to abort; re-run later with [cyan]/rpgkit.encode[/].",
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

    try:
        proc = subprocess.Popen(
            [sys.executable, str(encoder), "--json"],
            cwd=str(project_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
            "[cyan]/rpgkit.encode[/].[/yellow]"
        )
        return False

    if proc.returncode == 0:
        console.print()
        console.print(
            Panel(
                "[green]Encoder finished successfully.[/]\n\n"
                "The RPG graph is now available at "
                "[cyan].rpgkit/data/rpg.json[/].  The post-commit hook will "
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
            "with [cyan]/rpgkit.encode[/] after fixing the issue."
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

    Failures never propagate — ``rpgkit init`` is already done and we
    don't want a flaky encoder to taint the exit code.
    """
    # Already encoded: nothing to do.
    rpg_file = project_path / ".rpgkit" / "data" / "rpg.json"
    if rpg_file.exists():
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
                "RPG-Kit can build the initial graph for this repo now by "
                "running the encoder against your existing code.  This is "
                "what the [cyan]/rpgkit.encode[/] slash command does — kicking "
                "it off here saves you a step.\n\n"
                "[yellow]Heads up:[/] the encoder calls an LLM and can take "
                "a few minutes on a real-sized repo.  You can always say "
                "No and run [cyan]/rpgkit.encode[/] in your AI agent later.",
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
    upgrades and repeated ``rpgkit init/update`` runs:

    * Any prior RPG-Kit SessionStart entry (identified by the
      ``update_graphs.py`` marker in its command) is replaced rather
      than duplicated.
    * The ``mcp__rpg-tools`` allow rule is added only if absent.

    Why pre-authorize ``mcp__rpg-tools``?
        Claude Code prompts the user before each MCP tool invocation
        unless the rule is present in ``permissions.allow``.  Since
        the RPG-Kit server only exposes four read-only graph-query
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

    # Shell form: ``command`` is passed to ``sh -c``. Use shlex.quote so
    # paths containing spaces or special characters survive shell
    # tokenisation (json.dumps is JSON-safe but not shell-safe).
    update_script = shlex.quote(
        str((project_path / ".rpgkit" / "scripts" / "update_graphs.py").resolve())
    )
    python = shlex.quote(sys.executable)
    marker = "update_graphs.py"  # used for idempotent dedupe across upgrades

    rpg_session_entry = {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": (
                    f"{python} {update_script} status 2>/dev/null"
                    " || echo '[RPG-Kit] RPG status unavailable'"
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

    def _is_rpgkit_entry(entry: object) -> bool:
        """Detect a previously-installed RPG-Kit SessionStart entry.

        Matches both the current (shlex-quoted) and earlier
        (json.dumps-quoted) command shapes, plus any custom RPG-Kit
        entry the user may have added that still calls update_graphs.py.
        """
        if not isinstance(entry, dict):
            return False
        for h in entry.get("hooks", []) or []:
            cmd = h.get("command", "") if isinstance(h, dict) else ""
            if marker in cmd:
                return True
        return False

    # Drop any stale RPG-Kit entry before appending the fresh one.
    session_start = [e for e in session_start if not _is_rpgkit_entry(e)]
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
    rpgkit_rule = "mcp__rpg-tools"
    if rpgkit_rule not in allow:
        allow.append(rpgkit_rule)
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
    lookup, ``rpgkit init`` would write into ``.git/hooks/`` where git
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


# Each entry describes one shape of legacy (pre-sentinel) RPG-Kit snippet
# that may exist in a user's hook file from an older release.  The first
# element is a substring of the snippet's first line (a marker comment);
# the second is the *total* number of consecutive lines that snippet
# occupies starting at the marker line.  These are removed before the
# new sentinel block is written so users upgrading don't end up with the
# old snippet running alongside the new one.
LegacyBlock = Tuple[str, int]


def _strip_hook_block(
    text: str,
    block_name: str,
    legacy_blocks: Tuple[LegacyBlock, ...] = (),
) -> str:
    """Return ``text`` with any RPG-Kit-owned hook content removed.

    Two cleanup passes:

    1. Strip the new-style sentinel block::

           # RPGKIT-BEGIN <block_name>
           ...
           # RPGKIT-END <block_name>

       Range-based, so multi-line bodies of any shape are atomically
       removed in one shot.

    2. Strip each ``(marker_substring, line_count)`` legacy snippet
       (the pre-sentinel format used through release v0.0.99-dev.72).
       The marker line plus ``line_count - 1`` lines following it are
       dropped.  Multiple legacy shapes are removed in a single pass
       so the order of entries in ``legacy_blocks`` doesn't matter.

    Lines outside both passes are preserved verbatim so user-authored
    hook content (and shebangs) survive untouched.
    """
    begin_sent = f"# RPGKIT-BEGIN {block_name}"
    end_sent = f"# RPGKIT-END {block_name}"
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

    # Pass 2: strip legacy snippets by (marker, line_count).
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


def _install_hook_snippet(
    hooks_dir: Path,
    hook_name: str,
    block_name: str,
    body: str,
    *,
    legacy_blocks: Tuple[LegacyBlock, ...] = (),
) -> bool:
    """Install or replace an RPG-Kit-owned block in ``<hooks_dir>/<hook_name>``.

    File layout written::

        #!/bin/sh
        <any pre-existing user content>

        # RPGKIT-BEGIN <block_name>
        <body>
        # RPGKIT-END <block_name>

    The block is **atomically replaceable**: subsequent ``rpgkit init`` /
    ``rpgkit update`` runs find the existing sentinels and replace the
    whole block, so behavior upgrades land cleanly without piling new
    snippets on top of old ones.  ``legacy_blocks`` is used **once** to
    migrate pre-sentinel installs (released through v0.0.99-dev.72) onto
    this scheme; once a user has been migrated their hook contains the
    sentinels and the legacy patterns are no-ops.

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

    begin = f"# RPGKIT-BEGIN {block_name}"
    end = f"# RPGKIT-END {block_name}"
    block = f"\n{begin}\n{body.rstrip()}\n{end}\n"

    hook_path.write_text(prefix + block, encoding="utf-8")
    hook_path.chmod(0o755)
    return True


def _install_git_pre_commit_hook(project_path: Path) -> bool:
    """Install the RPG incremental-sync command into ``pre-commit``.

    Returns ``True`` when the hook is active on disk, ``False`` only
    when no git checkout was found at all.

    The hook passes ``--staged-only`` so only files the user
    ``git add``'d contribute to the diff — working-tree-but-not-staged
    changes are out of scope for the imminent commit.
    """
    hooks_dir = _resolve_git_hooks_dir(project_path)
    if hooks_dir is None:
        return False

    python = shlex.quote(sys.executable)
    update_script = shlex.quote(
        str((project_path / ".rpgkit" / "scripts" / "update_graphs.py").resolve())
    )
    marker = "# RPG-Kit: incremental RPG sync on commit"
    body = (
        f"{marker}\n"
        f"{python} {update_script} sync --staged-only 2>/dev/null || true"
    )
    # Legacy: pre-Step-3 pre-commit shipped a 2-line snippet under the
    # marker below.  Removed on upgrade so users don't end up running
    # both the old full-sync and the new staged-only path.
    return _install_hook_snippet(
        hooks_dir,
        "pre-commit",
        "pre-commit",
        body,
        legacy_blocks=(("# RPG-Kit: full RPG sync on commit", 2),),
    )


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

    python = shlex.quote(sys.executable)
    update_script = shlex.quote(
        str((project_path / ".rpgkit" / "scripts" / "update_graphs.py").resolve())
    )
    marker = "# RPG-Kit: incremental RPG sync after merge / pull"
    body = (
        f"{marker}\n"
        f"{python} {update_script} sync 2>/dev/null || true"
    )
    # post-merge was introduced with the sentinel-block design already
    # in mind, so no legacy migration is needed here.
    return _install_hook_snippet(hooks_dir, "post-merge", "post-merge", body)


def _install_git_post_commit_hook(project_path: Path) -> bool:
    """Install sync + background RPG update into ``post-commit``.

    Two phases run after every commit:

    1. **Synchronous** (foreground): ``update_graphs.py sync`` advances
       ``meta.git`` to the new HEAD (~50ms).  The pre-commit hook already
       updated dep_graph for the staged files, so this is a cheap
       hash-verify pass.

    2. **Asynchronous** (background): ``update_graphs.py update-rpg``
       creates a git worktree for ``HEAD~1``, runs the LLM-driven
       ``RPGEvolution.process_diff`` to update the feature graph, and
       cleans up the worktree.  Detached via ``nohup ... &`` (POSIX,
       portable to macOS where ``setsid`` is absent).  Output goes to
       ``.rpgkit/logs/update_rpg.log``.

       Concurrency is serialised by a *directory* lock at
       ``.rpgkit/logs/.update_rpg.lock`` \u2014 ``mkdir`` is the only
       POSIX-atomic exclusive-create primitive available from shell,
       so two commits firing in the same second reliably get one and
       only one worker.  Stale locks left by a SIGKILL'd previous run
       are auto-recovered after 60 minutes.

    Both phases are best-effort: failures are swallowed so they never
    block a commit.
    """
    hooks_dir = _resolve_git_hooks_dir(project_path)
    if hooks_dir is None:
        return False

    python = shlex.quote(sys.executable)
    update_script = shlex.quote(
        str((project_path / ".rpgkit" / "scripts" / "update_graphs.py").resolve())
    )
    log_file = shlex.quote(
        str((project_path / ".rpgkit" / "logs" / "update_rpg.log").resolve())
    )
    lock_file = shlex.quote(
        str((project_path / ".rpgkit" / "logs" / ".update_rpg.lock").resolve())
    )
    marker = "# RPG-Kit: advance meta.git + background feature graph update"
    workspace_dir = shlex.quote(str(project_path.resolve()))
    body = (
        f"{marker}\n"
        # Phase 1: synchronous meta.git advance
        f"{python} {update_script} sync 2>/dev/null || true\n"
        # Phase 2: background full RPG update.
        #
        # Lock semantics (v4):
        #   The lock is a *directory* created with ``mkdir`` — the only
        #   POSIX-atomic exclusive-create primitive available from shell.
        #   Two commits firing within the same second (interactive rebase,
        #   squash merge) reliably get serialised: exactly one wins the
        #   ``mkdir`` and spawns the background worker; the other no-ops.
        #
        # Lock recovery:
        #   (a) Pre-v4 installs used a *file* at this path.  ``rm -f``
        #       removes that file but silently no-ops on a directory
        #       ("Is a directory" error swallowed), so an active v4 lock
        #       is preserved.
        #   (b) Any v4 lock directory older than 60 minutes is assumed
        #       orphaned (worker SIGKILL'd, OOM, machine rebooted) and
        #       wiped.  Without this, a single crashed run would silently
        #       disable all future background updates.
        #
        # Detach strategy:
        #   ``nohup ... &`` is POSIX-portable.  We previously used
        #   ``setsid`` which is util-linux-only and absent from default
        #   macOS installs, leaving every macOS commit's phase-2 silently
        #   dead.
        #
        # env -u GIT_INDEX_FILE -u GIT_DIR:
        #   git sets these during hooks; if they leak into the background
        #   worker, ``git worktree add`` fails with cryptic index errors.
        f"rm -f {lock_file} 2>/dev/null\n"
        f"find {lock_file} -maxdepth 0 -mmin +60 -exec rm -rf {{}} + 2>/dev/null || true\n"
        f"if mkdir {lock_file} 2>/dev/null; then\n"
        f"  nohup env -u GIT_INDEX_FILE -u GIT_DIR "
        f'sh -c "cd {workspace_dir}; sleep 2; '
        f'{python} {update_script} update-rpg --json >> {log_file} 2>&1; '
        f'rmdir {lock_file}" </dev/null >/dev/null 2>&1 &\n'
        f"fi"
    )
    # Legacy shapes that may exist in users' .git/hooks/post-commit from
    # earlier releases.  Both are stripped before the new sentinel block
    # is written so the upgrade is a true replace, not an append.
    #   v1 (pre-Step-3 polish): 2-line sync-only snippet.
    #   v3 (release 0576393):   5-line snippet with the same first-line
    #                           marker we use today plus phase-1 sync,
    #                           phase-2 setsid background, and the
    #                           wrapping ``if/fi`` lock check.
    return _install_hook_snippet(
        hooks_dir,
        "post-commit",
        "post-commit",
        body,
        legacy_blocks=(
            ("# RPG-Kit: advance meta.git after commit", 2),
            ("# RPG-Kit: advance meta.git + background feature graph update", 5),
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

    update_script = str(
        (project_path / ".rpgkit" / "scripts" / "update_graphs.py").resolve()
    )

    rpg_status_task = {
        "label": "RPG-Kit: load status",
        "type": "shell",
        "command": sys.executable,
        "args": [update_script, "status"],
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
            "Prints RPG-Kit status and rpg-tools MCP usage guidance "
            "so GitHub Copilot can locate and generate code against "
            "the Repository Program Graph."
        ),
    }

    existing.setdefault("version", "2.0.0")
    tasks_list = existing.get("tasks")
    if not isinstance(tasks_list, list):
        tasks_list = []

    # Replace any prior RPG-Kit task with the same label rather than
    # appending duplicates on repeated ``rpgkit update`` runs.
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
    - All:     appends an RPG incremental sync (``update_graphs.py sync``)
      to ``.git/hooks/pre-commit`` AND ``.git/hooks/post-merge``.
      The pre-commit hook uses ``--staged-only`` so it sees only what's
      about to be committed; the post-merge hook (fired after
      ``git pull`` / ``git merge``) considers the whole working tree
      so teammate-incoming changes get picked up immediately.
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

        if _install_git_pre_commit_hook(project_path):
            installed.append("git:pre-commit")
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


def _select_latest_rpgkit_release(releases: List[dict], *, pre: bool) -> dict | None:
    candidates = [
        release
        for release in releases
        if not release.get("draft")
        and release.get("prerelease", False) is pre
        and release.get("tag_name", "").startswith(_RPGKIT_RELEASE_TAG_PREFIX)
    ]
    candidates.sort(key=_release_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _format_rpgkit_version(tag_name: str) -> str:
    if tag_name.startswith(_RPGKIT_RELEASE_TAG_PREFIX):
        return tag_name[len(_RPGKIT_RELEASE_TAG_PREFIX) :]
    if tag_name.startswith("v"):
        return tag_name[1:]
    return tag_name


def _fetch_latest_rpgkit_release(
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

    release_data = _select_latest_rpgkit_release(releases, pre=pre)
    if release_data is None:
        release_type = "pre-release" if pre else "release"
        raise RuntimeError(
            f"No RPG-Kit {release_type} found in {repo_owner}/{repo_name}. "
            f"Expected tags to start with {_RPGKIT_RELEASE_TAG_PREFIX}."
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
        release_data = _fetch_latest_rpgkit_release(
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
    pattern = f"rpgkit-template-{ai_assistant}-{script_type}"
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


def _resolve_rpgkit_source_root(source: Path) -> Path:
    source = source.expanduser().resolve()
    candidates = [source]
    if (source / "RPG-Kit").is_dir():
        candidates.insert(0, source / "RPG-Kit")

    for candidate in candidates:
        if (
            (candidate / "templates" / "commands").is_dir()
            and (candidate / "scripts").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    raise RuntimeError(
        f"Invalid RPG-Kit source path: {source}. Expected the RPG-Kit directory "
        "or the repository root containing RPG-Kit/."
    )


def _build_local_template_package(
    source: Path,
    ai_assistant: str,
    script_type: str,
) -> Tuple[Path, dict]:
    source_root = _resolve_rpgkit_source_root(source)
    repo_root = source_root.parent
    project_dir = source_root.relative_to(repo_root).as_posix()
    scripts_root = repo_root / ".github" / "workflows" / "scripts" / "rpgkit"
    version = "v0.0.0-local"
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_WORKSPACE": str(repo_root),
            "PROJECT_DIR": project_dir,
            "AGENTS": ai_assistant,
            "SCRIPTS": script_type,
            "PYTHON": sys.executable,
        }
    )

    if os.name == "nt":
        release_script = scripts_root / "create-release-packages.ps1"
        runner = shutil.which("pwsh")
        command = (
            [
                runner,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(release_script),
                version,
                "-Agents",
                ai_assistant,
                "-Scripts",
                script_type,
            ]
            if runner
            else None
        )
    else:
        release_script = scripts_root / "create-release-packages.sh"
        runner = shutil.which("bash")
        command = [runner, str(release_script), version] if runner else None

    if not release_script.is_file():
        raise RuntimeError(
            f"Release packaging script not found: {release_script}. "
            "Pass the RPG-ZeroRepo root or its RPG-Kit/ directory to --source."
        )
    if command is None:
        requirement = "PowerShell 7 (pwsh)" if os.name == "nt" else "bash"
        raise RuntimeError(
            f"Local --source packaging requires {requirement}, but it was not found on PATH."
        )

    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (
            result.stderr or result.stdout or "local package build failed"
        ).strip()
        raise RuntimeError(
            f"Failed to build local RPG-Kit template package from {source_root}: {detail}"
        )

    archive = (
        source_root
        / ".genreleases"
        / f"rpgkit-template-{ai_assistant}-{script_type}-{version}.zip"
    )
    if not archive.is_file():
        raise RuntimeError(
            f"Local RPG-Kit template package was not created: {archive}"
        )

    return archive, {
        "filename": archive.name,
        "size": archive.stat().st_size,
        "release": version,
        "source": str(source_root),
    }


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
    github_token: str = None,
    pre: bool = False,
    source: Path | None = None,
) -> Path:
    """Download or build a template archive and extract it to create a project.

    Returns project_path. Uses tracker if provided (with keys: fetch, download, extract, cleanup).
    """
    current_dir = Path.cwd()
    cleanup_zip = source is None

    if tracker:
        fetch_detail = (
            "building local template package" if source else "contacting GitHub API"
        )
        tracker.start("fetch", fetch_detail)
    try:
        if source:
            zip_path, meta = _build_local_template_package(
                source,
                ai_assistant,
                script_type,
            )
        else:
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
                "fetch", f"template {meta['release']} ({meta['size']:,} bytes)"
            )
            tracker.add("download", "Use template archive" if source else "Download template")
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

        if cleanup_zip and zip_path.exists():
            zip_path.unlink()
            if tracker:
                tracker.complete("cleanup")
            elif verbose:
                console.print(f"Cleaned up: {zip_path.name}")
        elif tracker:
            tracker.skip("cleanup", "local package retained")

    return project_path


def ensure_executable_scripts(
    project_path: Path, tracker: StepTracker | None = None
) -> None:
    """Ensure POSIX .sh scripts under .rpgkit/scripts (recursively) have execute bits (no-op on Windows)."""
    if os.name == "nt":
        return  # Windows: skip silently
    scripts_root = project_path / ".rpgkit" / "scripts"
    if not scripts_root.is_dir():
        return
    failures: list[str] = []
    updated = 0
    for script in scripts_root.rglob("*.sh"):
        try:
            if script.is_symlink() or not script.is_file():
                continue
            try:
                with script.open("rb") as f:
                    if f.read(2) != b"#!":
                        continue
            except Exception:
                continue
            st = script.stat()
            mode = st.st_mode
            if mode & 0o111:
                continue
            new_mode = mode
            if mode & 0o400:
                new_mode |= 0o100
            if mode & 0o040:
                new_mode |= 0o010
            if mode & 0o004:
                new_mode |= 0o001
            if not (new_mode & 0o100):
                new_mode |= 0o100
            os.chmod(script, new_mode)
            updated += 1
        except Exception as e:
            failures.append(f"{script.relative_to(scripts_root)}: {e}")
    if tracker:
        detail = f"{updated} updated" + (
            f", {len(failures)} failed" if failures else ""
        )
        tracker.add("chmod", "Set script permissions recursively")
        (tracker.error if failures else tracker.complete)("chmod", detail)
    else:
        if updated:
            console.print(
                f"[cyan]Updated execute permissions on {updated} script(s) recursively[/cyan]"
            )
        if failures:
            console.print("[yellow]Some scripts could not be updated:[/yellow]")
            for f in failures:
                console.print(f"  - {f}")


def setup_venv_rpgkit(
    project_path: Path, tracker: StepTracker | None = None
) -> None:
    """Create or update .venv_rpgkit with RPG-Kit Python dependencies."""
    venv_dir = project_path / ".venv_rpgkit"
    rpgkit_dir = project_path / ".rpgkit"
    pyproject = rpgkit_dir / "pyproject.toml"

    if tracker:
        tracker.start("venv")

    if not pyproject.is_file():
        msg = ".rpgkit/pyproject.toml not found — cannot install Python dependencies"
        if tracker:
            tracker.skip("venv", msg)
        else:
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        return

    try:
        is_new = not venv_dir.exists()

        if is_new:
            subprocess.run(
                ["uv", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

        if os.name == "nt":
            pip_python = venv_dir / "Scripts" / "python.exe"
        else:
            pip_python = venv_dir / "bin" / "python3"

        subprocess.run(
            ["uv", "pip", "install", str(rpgkit_dir), "--python", str(pip_python)],
            check=True,
            capture_output=True,
            text=True,
        )

        if tracker:
            tracker.complete(
                "venv",
                "created .venv_rpgkit" if is_new else "updated .venv_rpgkit",
            )
    except FileNotFoundError:
        msg = "uv not found — install uv (https://docs.astral.sh/uv/) to enable auto-setup"
        if tracker:
            tracker.skip("venv", msg)
        console.print(f"[yellow]Warning:[/yellow] {msg}")
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() if e.stderr else str(e)
        msg = f"Failed to set up .venv_rpgkit:\n{detail}"
        if tracker:
            tracker.error("venv", detail[:120])
        console.print(f"[red]Error:[/red] {msg}")
    except Exception as e:
        msg = f"Failed to set up .venv_rpgkit: {e}"
        if tracker:
            tracker.error("venv", str(e)[:120])
        console.print(f"[red]Error:[/red] {msg}")


def ensure_rpgkit_runtime_dirs(
    project_path: Path, tracker: StepTracker | None = None
) -> None:
    """Pre-create RPG-Kit runtime directories under ``.rpgkit/``.

    Some early-pipeline prompts redirect stdout/stderr to
    ``.rpgkit/logs/<stage>.log`` via shell ``>``, which fails with
    "No such file or directory" if the parent directory does not yet exist.
    The first script that calls ``setup_file_logging`` would normally
    auto-create ``.rpgkit/logs/``, but that only helps stages that use
    the Python logging helper — shell-redirected stages fail BEFORE the
    Python process even starts.

    Creating the runtime directories upfront (during ``rpgkit init`` /
    ``rpgkit update``) makes all stage prompts robust without each one
    having to ``mkdir -p`` defensively.

    Created (idempotent):
        - ``.rpgkit/logs/``        — per-stage log files
        - ``.rpgkit/data/``        — encoder / pipeline JSON artifacts
        - ``.rpgkit/data/trajectory/`` — execution trajectories
    """
    subdirs = ("logs", "data", "data/trajectory")
    created: list[str] = []
    for sub in subdirs:
        path = project_path / ".rpgkit" / sub
        existed = path.exists()
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not existed:
                created.append(sub)
        except OSError:
            # Filesystem read-only / permission issue — non-blocking.
            continue
    if tracker:
        tracker.add("runtime-dirs", "Ensure .rpgkit/{logs,data} directories")
        detail = (
            f"created {', '.join(created)}" if created else "all already present"
        )
        tracker.complete("runtime-dirs", detail)


def _detect_ai_agent(project_path: Path) -> str | None:
    """Detect AI agent from existing project directory.

    Scans for known agent folders (from AGENT_CONFIG) and checks if they
    contain rpgkit.* command files. Returns the agent key or None.
    """
    found = []
    for key, config in AGENT_CONFIG.items():
        agent_dir = project_path / config["folder"]
        if agent_dir.is_dir():
            # Check common command subdirectories for rpgkit.* files
            for sub in ("commands", "agents", "prompts"):
                candidate = agent_dir / sub
                if candidate.is_dir() and any(candidate.glob("rpgkit.*")):
                    found.append(key)
                    break
            else:
                # Folder exists even without rpgkit commands subdirectory
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
    skip_tls: bool = typer.Option(
        False, "--skip-tls", help="Skip SSL/TLS verification (not recommended)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show verbose diagnostic output for network and extraction failures",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        help="GitHub token to use for API requests (or set GH_TOKEN or GITHUB_TOKEN environment variable)",
    ),
    pre: bool = typer.Option(
        False,
        "--pre",
        help="Download the latest pre-release (dev build) instead of the latest stable release",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help=(
            "Use a local RPG-Kit source checkout to build and install the "
            "template package instead of downloading a release asset."
        ),
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP server registration (rpg-tools won't be exposed to the AI agent)",
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
):
    """Initialize a new RPG-Kit project from the latest template.

    This command will:
    1. Check that required tools are installed (git is optional)
    2. Let you choose your AI assistant
    3. Download the appropriate template from GitHub
    4. Extract the template to a new project directory or current directory
    5. Initialize a fresh git repository (if not --no-git and no existing repo)
    6. Optionally set up AI assistant commands

    Examples:
        rpgkit init my-project
        rpgkit init my-project --ai claude
        rpgkit init my-project --ai copilot --no-git
        rpgkit init --ignore-agent-tools my-project
        rpgkit init . --ai claude         # Initialize in current directory
        rpgkit init .                     # Initialize in current directory (interactive AI selection)
        rpgkit init --here --ai claude    # Alternative syntax for current directory
        rpgkit init --here --ai codex
        rpgkit init --here --ai codebuddy
        rpgkit init --here
        rpgkit init --here --force  # Skip confirmation when current directory not empty
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
        "[cyan]RPG-Kit Project Setup[/cyan]",
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
        selected_script = script_type
    else:
        default_script = "ps" if os.name == "nt" else "sh"

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
    if source:
        console.print(f"[cyan]Template source:[/cyan] {source}")
        if pre:
            console.print(
                "[yellow]Warning:[/yellow] --pre is ignored when --source is provided"
            )

    tracker = StepTracker("Initialize RPG-Kit Project")

    sys._rpgkit_tracker_active = True

    tracker.add("precheck", "Check required tools")
    tracker.complete("precheck", "ok")
    tracker.add("ai-select", "Select AI assistant")
    tracker.complete("ai-select", f"{selected_ai}")
    tracker.add("script-select", "Select script type")
    tracker.complete("script-select", selected_script)
    for key, label in [
        (
            "fetch",
            "Build local template package"
            if source
            else "Fetch latest pre-release"
            if pre
            else "Fetch latest release",
        ),
        ("download", "Use local template package" if source else "Download template"),
        ("extract", "Extract template"),
        ("zip-list", "Archive contents"),
        ("extracted-summary", "Extraction summary"),
        ("chmod", "Ensure scripts executable"),
        ("gitignore", "Configure .gitignore"),
        ("mcp", "Configure MCP server"),
        ("legacy-cleanup", "Remove obsolete persistent rules"),
        ("venv", "Set up Python environment"),
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
            verify = not skip_tls
            local_ssl_context = ssl_context if verify else False
            local_client = httpx.Client(verify=local_ssl_context)

            download_and_extract_template(
                project_path,
                selected_ai,
                selected_script,
                here,
                verbose=False,
                tracker=tracker,
                client=local_client,
                debug=debug,
                github_token=github_token,
                pre=pre,
                source=source,
            )

            ensure_executable_scripts(project_path, tracker=tracker)

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

            # Migrate workspaces created before C4: drop the auto-loaded
            # rpgkit-codegen.* persistent-instruction files.
            tracker.start("legacy-cleanup")
            try:
                removed = _cleanup_legacy_codegen_persistent(project_path)
                if removed:
                    tracker.complete(
                        "legacy-cleanup",
                        f"removed {len(removed)} file(s)",
                    )
                else:
                    tracker.skip("legacy-cleanup", "none")
            except Exception as exc:
                tracker.error("legacy-cleanup", str(exc))

            setup_venv_rpgkit(project_path, tracker=tracker)

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
            f"RPG-Kit's slash command definitions under [cyan]{ignored_path_desc}[/cyan] are regenerated by [cyan]rpgkit init/update[/cyan] and are excluded from git by default.\n"
            f"Collaborators should run [cyan]rpgkit init[/cyan] in their clone to materialize the prompt files locally.",
            title="[yellow]Agent Folder Notice[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print()
        console.print(security_notice)

    # Pre-create runtime directories so early pipeline prompts that redirect
    # to .rpgkit/logs/<stage>.log don't fail with "No such file or directory".
    ensure_rpgkit_runtime_dirs(project_path)

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

    venv_path = project_path / ".venv_rpgkit"
    if os.name == "nt":
        activate_cmd = r".venv_rpgkit\Scripts\activate"
    else:
        activate_cmd = "source .venv_rpgkit/bin/activate"
    steps_lines.append(
        f"{step_num}. Activate the RPG-Kit Python environment: "
        f"[cyan]{activate_cmd}[/cyan]"
    )

    step_num += 1

    steps_lines.append(f"{step_num}. Start using slash commands with your AI agent:")

    steps_lines.extend([
        f"   {step_num}.1  [cyan]/rpgkit.feature_spec[/] - Create feature spec from docs",
        f"   {step_num}.2  [cyan]/rpgkit.feature_build[/] - Generate and Expand Feature Tree",
        f"   {step_num}.3  [cyan]/rpgkit.feature_refactor[/] - Refactor Feature Tree",
        f"   {step_num}.4  [cyan]/rpgkit.feature_edit[/] - Edit Feature Tree Nodes",
        f"   {step_num}.5  [cyan]/rpgkit.build_skeleton[/] - Repository Skeleton Structure",
        f"   {step_num}.6  [cyan]/rpgkit.build_data_flow[/] - Data Flow Design",
        f"   {step_num}.7  [cyan]/rpgkit.design_base_classes[/] - Base Classes Design",
        f"   {step_num}.8  [cyan]/rpgkit.design_interfaces[/] - Interface Design",
        f"   {step_num}.9  [cyan]/rpgkit.plan_tasks[/] - Task Planning",
        f"   {step_num}.10 [cyan]/rpgkit.code_gen[/] - Code Generation",
        f"   {step_num}.11 [cyan]/rpgkit.rpg_edit[/] - Surgical RPG/code edit",
        f"   {step_num}.12 [cyan]/rpgkit.encode[/] - Encode repo into RPG",
        f"   {step_num}.13 [cyan]/rpgkit.update_rpg[/] - Incremental RPG update",
    ])

    step_num += 1
    steps_lines.append(
        f"{step_num}. You can inspect each step's output under [cyan].rpgkit/data/[/cyan], "
        f"and review detailed execution trajectories in [cyan].rpgkit/data/trajectory/[/cyan]."
    )

    step_num += 1
    steps_lines.append(
        f"{step_num}. The RPG-Kit MCP server provides [cyan]search_rpg[/], [cyan]explore_rpg[/], "
        f"[cyan]get_node_detail[/], and [cyan]list_rpg_tree[/] "
        f"tools for AI agents to query RPG graphs via the Model Context Protocol."
    )
    # First-run note: the MCP tools are wired up at init time, but they
    # only return useful data once the encoder has built rpg.json.  Make
    # the requirement loud-and-clear here so users don't hit the silent
    # "rpg_unavailable" payload on their first /rpgkit.* call.
    steps_lines.append(
        f"   [yellow]Note:[/] the MCP tools query [cyan].rpgkit/data/rpg.json[/], which is "
        f"created by the encoder. For existing codebases, run [cyan]/rpgkit.encode[/] "
        f"once now to populate it; the post-commit hook keeps it in sync afterwards."
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
    skip_tls: bool = typer.Option(
        False, "--skip-tls", help="Skip SSL/TLS verification (not recommended)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show verbose diagnostic output for network and extraction failures",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        help="GitHub token to use for API requests (or set GH_TOKEN or GITHUB_TOKEN environment variable)",
    ),
    pre: bool = typer.Option(
        False,
        "--pre",
        help="Download the latest pre-release (dev build) instead of the latest stable release",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help=(
            "Use a local RPG-Kit source checkout to build and install the "
            "template package instead of downloading a release asset."
        ),
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP server registration (rpg-tools won't be exposed to the AI agent)",
    ),
):
    """Update RPG-Kit template files in an existing project to the latest version.

    This command updates scripts, templates, command definitions, MCP
    config, gitignore rules, and git hooks in the current directory.
    It auto-detects the AI assistant from existing project configuration.

    Equivalent to re-running 'rpgkit init --here --force' but with proper
    semantics and automatic detection of existing settings.

    Examples:
        rpgkit update
        rpgkit update --ai claude
        rpgkit update --pre
        rpgkit update --github-token $GITHUB_TOKEN
    """
    show_banner()

    project_path = Path.cwd()

    # Verify this is an existing RPG-Kit project
    rpgkit_dir = project_path / ".rpgkit"
    if not rpgkit_dir.is_dir():
        console.print(
            Panel(
                "No [cyan].rpgkit/[/cyan] directory found in the current directory.\n"
                "This command updates an existing RPG-Kit project.\n\n"
                "To create a new project, use: [cyan]rpgkit init[/cyan]",
                title="[red]Not an RPG-Kit Project[/red]",
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
        selected_script = script_type
    else:
        default_script = "ps" if os.name == "nt" else "sh"
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
    if source:
        console.print(f"[cyan]Template source:[/cyan] {source}")
        if pre:
            console.print(
                "[yellow]Warning:[/yellow] --pre is ignored when --source is provided"
            )

    # Build step tracker
    tracker = StepTracker("Update RPG-Kit Project")

    sys._rpgkit_tracker_active = True

    tracker.add("ai-select", "Select AI assistant")
    tracker.complete("ai-select", f"{selected_ai}")
    tracker.add("script-select", "Select script type")
    tracker.complete("script-select", selected_script)
    for key, label in [
        (
            "fetch",
            "Build local template package"
            if source
            else "Fetch latest pre-release"
            if pre
            else "Fetch latest release",
        ),
        ("download", "Use local template package" if source else "Download template"),
        ("extract", "Extract template"),
        ("zip-list", "Archive contents"),
        ("extracted-summary", "Extraction summary"),
        ("chmod", "Ensure scripts executable"),
        ("gitignore", "Configure .gitignore"),
        ("mcp", "Configure MCP server"),
        ("legacy-cleanup", "Remove obsolete persistent rules"),
        ("venv", "Set up Python environment"),
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
            verify = not skip_tls
            local_ssl_context = ssl_context if verify else False
            local_client = httpx.Client(verify=local_ssl_context)

            download_and_extract_template(
                project_path,
                selected_ai,
                selected_script,
                True,  # is_current_dir — always merge/overwrite for update
                verbose=False,
                tracker=tracker,
                client=local_client,
                debug=debug,
                github_token=github_token,
                pre=pre,
                source=source,
            )

            ensure_executable_scripts(project_path, tracker=tracker)

            # Pre-create runtime directories so stage prompts that redirect
            # to .rpgkit/logs/<stage>.log don't fail when the folder is
            # missing (e.g. user removed it, or workspace was created by an
            # older rpgkit init that didn't pre-create logs/).
            ensure_rpgkit_runtime_dirs(project_path, tracker=tracker)

            # Ensure RPG-Kit gitignore rules are in place — re-runs are
            # idempotent (existing rules are detected and skipped) and this
            # also fixes workspaces created by older rpgkit versions that
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

            # Migrate workspaces created before C4: drop the auto-loaded
            # rpgkit-codegen.* persistent-instruction files.
            tracker.start("legacy-cleanup")
            try:
                removed = _cleanup_legacy_codegen_persistent(project_path)
                if removed:
                    tracker.complete(
                        "legacy-cleanup",
                        f"removed {len(removed)} file(s)",
                    )
                else:
                    tracker.skip("legacy-cleanup", "none")
            except Exception as exc:
                tracker.error("legacy-cleanup", str(exc))

            setup_venv_rpgkit(project_path, tracker=tracker)

            # Re-install hooks so behavior fixes propagate to existing
            # workspaces.  Without this, the .git/hooks/* files stay
            # frozen at whatever version was active during the original
            # `rpgkit init`, and the sentinel-block migration in
            # _install_hook_snippet (the upgrade mechanism for hooks)
            # never gets a chance to run.
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
        "\n[bold green]RPG-Kit templates updated successfully.[/bold green]"
    )
    console.print(
        f"[dim]Updated: scripts, templates, and {AGENT_CONFIG[selected_ai]['name']} "
        f"command definitions in [cyan]{project_path}[/cyan][/dim]"
    )
    console.print()
    venv_path = Path(project_path) / ".venv_rpgkit"
    if venv_path.exists():
        activate_cmd = (
            r".venv_rpgkit\Scripts\activate"
            if os.name == "nt"
            else "source .venv_rpgkit/bin/activate"
        )
        console.print(
            Panel(
                "Activate the RPG-Kit Python environment before using slash commands:\n\n"
                f"[cyan]{activate_cmd}[/cyan]",
                title="[yellow]Environment Setup[/yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

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

    console.print("\n[bold green]RPG-Kit CLI is ready to use![/bold green]")

    if not git_ok:
        console.print("[dim]Tip: Install git for repository management[/dim]")

    if not any(agent_results.values()):
        console.print("[dim]Tip: Install an AI assistant for the best experience[/dim]")


@app.command()
def version():
    """Display version and system information."""
    show_banner()

    # Get CLI version from package metadata
    cli_version = "unknown"
    try:
        cli_version = importlib.metadata.version("rpgkit-cli")
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

    template_version = "unknown"
    release_date = "unknown"

    try:
        release_data = _fetch_latest_rpgkit_release(
            repo_owner,
            repo_name,
            client,
            timeout=10,
        )
        template_version = _format_rpgkit_version(release_data.get("tag_name", "unknown"))
        release_date = release_data.get("published_at", "unknown")
        if release_date != "unknown":
            # Format the date nicely
            try:
                dt = datetime.fromisoformat(release_date.replace("Z", "+00:00"))
                release_date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    except Exception:
        pass

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="cyan", justify="right")
    info_table.add_column("Value", style="white")

    info_table.add_row("CLI Version", cli_version)
    info_table.add_row("Template Version", template_version)
    info_table.add_row("Released", release_date)
    info_table.add_row("", "")
    info_table.add_row("Python", platform.python_version())
    info_table.add_row("Platform", platform.system())
    info_table.add_row("Architecture", platform.machine())
    info_table.add_row("OS Version", platform.version())

    panel = Panel(
        info_table,
        title="[bold cyan]RPG-Kit CLI Information[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()


def main():
    app()


if __name__ == "__main__":
    main()
