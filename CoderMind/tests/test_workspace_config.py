"""Tests for workspace LLM backend configuration and CLI commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cmind_cli import _workspace_config
from cmind_cli import app


runner = CliRunner()


def _write_config(workspace: Path, content: str) -> Path:
    path = workspace / ".cmind" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_set_backend_preserves_comments_and_unrelated_values(tmp_path):
    path = _write_config(
        tmp_path,
        """# user comment
[cmind]
ai_cli_cmd = "copilot"
custom = "keep"

[other]
value = 42
""",
    )

    backend = _workspace_config.set_active_backend(tmp_path, "codex")

    assert backend.agent == "codex"
    assert backend.command == "codex exec"
    content = path.read_text(encoding="utf-8")
    assert "# user comment" in content
    assert 'custom = "keep"' in content
    assert "[other]" in content
    assert "value = 42" in content
    assert 'ai_cli_cmd = "codex exec"' in content


def test_initialize_is_idempotent(tmp_path):
    path = _workspace_config.initialize(tmp_path, "claude")
    original = path.read_text(encoding="utf-8")

    _workspace_config.initialize(tmp_path, "codex")

    assert path.read_text(encoding="utf-8") == original
    assert _workspace_config.read_active_backend(tmp_path).agent == "claude"


def test_initialize_preserves_registered_unverified_agent_mapping(tmp_path):
    _workspace_config.initialize(tmp_path, "gemini")

    assert _workspace_config.read_active_backend(tmp_path).command == "gemini -p"


def test_set_backend_rejects_unknown_agent(tmp_path):
    _workspace_config.initialize(tmp_path, "copilot")

    with pytest.raises(_workspace_config.WorkspaceConfigError, match="unsupported"):
        _workspace_config.set_active_backend(tmp_path, "unknown")


def test_config_show_reports_active_backend(tmp_path, monkeypatch):
    _workspace_config.initialize(tmp_path, "copilot")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "Active backend: copilot" in result.output
    assert "CLI command: copilot" in result.output


def test_config_set_agent_updates_backend(tmp_path, monkeypatch):
    _workspace_config.initialize(tmp_path, "copilot")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set-agent", "codex"])

    assert result.exit_code == 0
    assert "Active backend set to codex" in result.output
    assert _workspace_config.read_active_backend(tmp_path).agent == "codex"


def test_config_command_requires_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 1
    assert "No CoderMind workspace found" in result.output
