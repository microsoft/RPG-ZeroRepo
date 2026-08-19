"""Integration contracts for simultaneous Claude, Copilot, and Codex setup."""

import json
from pathlib import Path

import tomllib

import cmind_cli
from cmind_cli import _assets
from cmind_cli import _workspace_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMANDS_DIR = PROJECT_ROOT / "templates" / "commands"


def test_bundle_materializes_all_agent_integrations(tmp_path, monkeypatch):
    monkeypatch.setattr(_assets, "commands_dir", lambda: COMMANDS_DIR)

    cmind_cli._install_from_bundle(
        tmp_path,
        "codex",
        "sh",
        True,
    )

    assert len(list((tmp_path / ".claude" / "commands").glob("cmind.*.md"))) == 15
    assert len(list((tmp_path / ".github" / "agents").glob("cmind.*.agent.md"))) == 15
    assert len(list((tmp_path / ".github" / "prompts").glob("cmind.*.prompt.md"))) == 15
    assert len(list((tmp_path / ".agents" / "skills").glob("cmind-*/SKILL.md"))) == 15


def test_backend_selection_does_not_change_generated_integrations(tmp_path, monkeypatch):
    monkeypatch.setattr(_assets, "commands_dir", lambda: COMMANDS_DIR)
    first = tmp_path / "copilot-backend"
    second = tmp_path / "codex-backend"
    first.mkdir()
    second.mkdir()

    cmind_cli._install_from_bundle(first, "copilot", "sh", True)
    cmind_cli._install_from_bundle(second, "codex", "sh", True)

    relative = lambda root: sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and ".cmind" not in path.parts
    )
    assert relative(first) == relative(second)


def test_mcp_generation_registers_all_project_clients(tmp_path):
    cmind_cli._generate_mcp_config(tmp_path, "codex")

    claude = json.loads((tmp_path / ".mcp.json").read_text())
    copilot = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
    with open(tmp_path / ".codex" / "config.toml", "rb") as file:
        codex = tomllib.load(file)

    assert claude["mcpServers"]["rpg-tools"]["command"] == "cmind-mcp"
    assert copilot["servers"]["rpg-tools"]["command"] == "cmind-mcp"
    assert codex["mcp_servers"]["rpg-tools"]["command"] == "cmind-mcp"


def test_backend_switch_does_not_remove_integrations(tmp_path, monkeypatch):
    monkeypatch.setattr(_assets, "commands_dir", lambda: COMMANDS_DIR)
    cmind_cli._install_from_bundle(tmp_path, "copilot", "sh", True)
    _workspace_config.initialize(tmp_path, "copilot")

    _workspace_config.set_active_backend(tmp_path, "codex")

    assert _workspace_config.read_active_backend(tmp_path).agent == "codex"
    assert (tmp_path / ".claude" / "commands" / "cmind.encode.md").is_file()
    assert (tmp_path / ".github" / "agents" / "cmind.encode.agent.md").is_file()
    assert (tmp_path / ".agents" / "skills" / "cmind-encode" / "SKILL.md").is_file()
