"""Manage project-scoped Codex configuration owned by CoderMind."""

from __future__ import annotations

import os
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError


class CodexConfigError(ValueError):
    """Raised when project Codex configuration cannot be merged safely."""


def configure_rpg_tools(workspace: Path) -> Path:
    """Register ``rpg-tools`` without overwriting a user-owned command."""
    path = workspace / ".codex" / "config.toml"
    document = _load_or_create(path)

    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        document.add("mcp_servers", servers)
    if not isinstance(servers, dict):
        raise CodexConfigError(f"{path} has a non-table mcp_servers value")

    current = servers.get("rpg-tools")
    if current is not None:
        if not isinstance(current, dict):
            raise CodexConfigError(f"{path} has an invalid rpg-tools MCP entry")
        command = current.get("command")
        if command and command != "cmind-mcp":
            raise CodexConfigError(
                f"{path} already defines rpg-tools with command {command!r}; "
                "refusing to overwrite it"
            )
        args = current.get("args")
        if args not in (None, []):
            raise CodexConfigError(
                f"{path} already defines custom rpg-tools args; "
                "refusing to overwrite them"
            )
    else:
        current = tomlkit.table()
        servers["rpg-tools"] = current

    current["command"] = "cmind-mcp"
    current["args"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, tomlkit.dumps(document))
    return path


def _load_or_create(path: Path):
    if not path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, ParseError) as exc:
        raise CodexConfigError(f"could not read {path}: {exc}") from exc


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        raise CodexConfigError(f"could not write {path}: {exc}") from exc
