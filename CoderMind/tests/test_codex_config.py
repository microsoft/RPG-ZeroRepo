"""Tests for safe project-scoped Codex MCP configuration."""

import pytest

from cmind_cli import _codex_config


def test_configure_rpg_tools_preserves_existing_config(tmp_path):
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('# keep\nmodel = "custom"\n\n[other]\nvalue = 42\n')

    _codex_config.configure_rpg_tools(tmp_path)

    content = path.read_text()
    assert "# keep" in content
    assert 'model = "custom"' in content
    assert "[other]" in content
    assert "value = 42" in content
    assert '[mcp_servers.rpg-tools]' in content
    assert 'command = "cmind-mcp"' in content
    assert 'CMIND_MCP_CLIENT_CONTEXT = "codex-agent"' in content


def test_configure_rpg_tools_is_idempotent(tmp_path):
    path = _codex_config.configure_rpg_tools(tmp_path)
    first = path.read_text()

    _codex_config.configure_rpg_tools(tmp_path)

    assert path.read_text() == first


def test_configure_rpg_tools_refuses_custom_command(tmp_path):
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        '[mcp_servers.rpg-tools]\ncommand = "user-mcp"\nargs = []\n'
    )

    with pytest.raises(_codex_config.CodexConfigError, match="refusing"):
        _codex_config.configure_rpg_tools(tmp_path)

    assert 'command = "user-mcp"' in path.read_text()


def test_configure_rpg_tools_refuses_custom_args(tmp_path):
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        '[mcp_servers.rpg-tools]\ncommand = "cmind-mcp"\nargs = ["--custom"]\n'
    )

    with pytest.raises(_codex_config.CodexConfigError, match="custom.*args"):
        _codex_config.configure_rpg_tools(tmp_path)

    assert 'args = ["--custom"]' in path.read_text()


def test_configure_rpg_tools_preserves_custom_env(tmp_path):
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        '[mcp_servers.rpg-tools]\n'
        'command = "cmind-mcp"\n'
        'args = []\n'
        'env = { USER_SETTING = "keep" }\n'
    )

    _codex_config.configure_rpg_tools(tmp_path)

    content = path.read_text()
    assert 'USER_SETTING = "keep"' in content
    assert 'CMIND_MCP_CLIENT_CONTEXT = "codex-agent"' in content