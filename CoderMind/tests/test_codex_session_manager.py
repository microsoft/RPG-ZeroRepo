"""Tests for the Codex CLI runtime adapter."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common.session_manager import CodexSessionManager, create_session_manager


def test_codex_manager_uses_stdin_and_bounded_permissions(tmp_path):
    manager = create_session_manager("codex", tmp_path)

    with manager.trace("Reply with exactly: codex-ok") as context:
        assert isinstance(manager, CodexSessionManager)
        assert context.extra_args == [
            "--approve-for-me",
            "--ephemeral",
            "--skip-git-repo-check",
            "-",
        ]
        assert context.stdin.read() == "Reply with exactly: codex-ok"

    assert context.stdin.closed


def test_codex_manager_rewinds_stdin_for_retry(tmp_path):
    manager = create_session_manager("codex", tmp_path)

    with manager.trace("retry me") as context:
        assert context.stdin.read() == "retry me"
        context.refresh_for_retry()
        assert context.stdin.read() == "retry me"


def test_codex_manager_allows_explicit_sandbox_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("CMIND_CODEX_BYPASS_SANDBOX", "1")
    manager = create_session_manager("codex", tmp_path)

    with manager.trace("automate me") as context:
        assert context.extra_args == [
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "--skip-git-repo-check",
            "-",
        ]