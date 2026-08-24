"""Read and update the team-shared CoderMind workspace configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError


AGENT_CLI_COMMANDS: dict[str, str] = {
    "copilot": "copilot",
    "claude": "claude",
    "gemini": "gemini -p",
    "qwen": "qwen -p",
    "cursor-agent": "agent -p",
    "auggie": "augment -p",
    "codex": "codex exec",
    "codebuddy": "codebuddy -p",
    "qoder": "qodercli -p",
    "opencode": "opencode run",
    "amp": "amp --execute",
}

SUPPORTED_BACKENDS = ("copilot", "claude", "codex")
_CONFIG_RELPATH = Path(".cmind/config.toml")
_CONFIG_HEADER = """# CoderMind workspace configuration
# Managed by `cmind init` / `cmind update`.  Safe to commit.
# See: https://github.com/microsoft/RPG-ZeroRepo (CoderMind/docs/configuration.md)

"""


class WorkspaceConfigError(ValueError):
    """Raised when a workspace configuration cannot be read or updated."""


@dataclass(frozen=True)
class ActiveBackend:
    """The configured backend and its resolved CLI command."""

    agent: str | None
    command: str


def config_path(workspace: Path) -> Path:
    return workspace / _CONFIG_RELPATH


def initialize(workspace: Path, agent: str) -> Path:
    """Create the workspace config when missing, preserving existing files."""
    path = config_path(workspace)
    if path.exists():
        return path

    command = _registered_command_for(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = tomlkit.document()
    document.add("cmind", {"ai_cli_cmd": command})
    _atomic_write(path, _CONFIG_HEADER + tomlkit.dumps(document))
    return path


def read_active_backend(workspace: Path) -> ActiveBackend:
    """Read the active backend from a CoderMind workspace."""
    path = config_path(workspace)
    document = _load(path)
    cmind = document.get("cmind")
    command = cmind.get("ai_cli_cmd") if isinstance(cmind, dict) else None
    if not isinstance(command, str) or not command.strip():
        raise WorkspaceConfigError(f"{path} does not define [cmind].ai_cli_cmd")

    normalized = command.strip()
    agent = next(
        (
            name
            for name in SUPPORTED_BACKENDS
            if AGENT_CLI_COMMANDS[name] == normalized
        ),
        None,
    )
    return ActiveBackend(agent=agent, command=normalized)


def set_active_backend(workspace: Path, agent: str) -> ActiveBackend:
    """Set the active backend while preserving comments and unrelated keys."""
    path = config_path(workspace)
    document = _load(path)
    if agent not in SUPPORTED_BACKENDS:
        choices = ", ".join(SUPPORTED_BACKENDS)
        raise WorkspaceConfigError(
            f"unsupported backend {agent!r}; choose one of: {choices}"
        )
    command = _registered_command_for(agent)

    cmind = document.get("cmind")
    if cmind is None:
        cmind = tomlkit.table()
        document.add("cmind", cmind)
    if not isinstance(cmind, dict):
        raise WorkspaceConfigError(f"{path} has a non-table [cmind] value")

    cmind["ai_cli_cmd"] = command
    _atomic_write(path, tomlkit.dumps(document))
    return ActiveBackend(agent=agent, command=command)


def _registered_command_for(agent: str) -> str:
    if agent not in AGENT_CLI_COMMANDS:
        choices = ", ".join(AGENT_CLI_COMMANDS)
        raise WorkspaceConfigError(
            f"unknown agent {agent!r}; choose one of: {choices}"
        )
    return AGENT_CLI_COMMANDS[agent]


def _load(path: Path):
    if not path.is_file():
        raise WorkspaceConfigError(
            f"no CoderMind workspace config found at {path}; run `cmind init .` first"
        )
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, ParseError) as exc:
        raise WorkspaceConfigError(f"could not read {path}: {exc}") from exc


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        raise WorkspaceConfigError(f"could not write {path}: {exc}") from exc
