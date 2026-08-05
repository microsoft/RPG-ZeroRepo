from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_mcp_log_includes_inherited_correlation(tmp_path, monkeypatch):
    import mcp_server
    from common.activity_events import ActivityWriter, load_activity_events

    path = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(mcp_server, "MCP_CALLS_LOG", path)
    monkeypatch.setattr(mcp_server, "ACTIVITY_WRITER", ActivityWriter(tmp_path / "activity", workspace_id="ws_test"))
    monkeypatch.setenv("CMIND_RUN_ID", "run-1")
    monkeypatch.setenv("CMIND_STAGE_ID", "stage-1")
    monkeypatch.setenv("CMIND_TRACE_ID", "trc_test")
    monkeypatch.setenv("CMIND_PARENT_SPAN_ID", "spn_parent")
    monkeypatch.setenv("CMIND_MCP_CLIENT_CONTEXT", "copilot-agent")
    mcp_server._log_tool_call("search_rpg", {"query": "x"}, {"hits": 2}, 12)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["call_id"].startswith("mcp_")
    assert row["run_id"] == "run-1"
    assert row["stage_id"] == "stage-1"
    assert row["server_session_id"] == mcp_server.MCP_SERVER_SESSION_ID
    assert row["trace_id"] == "trc_test"
    assert row["client_context"] == "copilot-agent"
    started, finished = load_activity_events(tmp_path / "activity")
    assert started["trace_id"] == finished["trace_id"] == "trc_test"
    assert started["parent_span_id"] == finished["parent_span_id"] == "spn_parent"
    assert finished["kind"] == "tool.mcp"
    assert finished["duration_ms"] == 12
    assert finished["client_context"] == "copilot-agent"
    assert "params" not in finished


def test_hook_log_prefers_result_correlation(tmp_path, monkeypatch):
    import update_graphs
    from common.activity_events import ActivityWriter, load_activity_events

    path = tmp_path / "hook_calls.jsonl"
    monkeypatch.setattr(update_graphs, "HOOK_CALLS_LOG", path)
    monkeypatch.setattr(update_graphs, "ACTIVITY_WRITER", ActivityWriter(tmp_path / "activity", workspace_id="ws_test"))
    monkeypatch.setenv("CMIND_RUN_ID", "run-env")
    monkeypatch.setenv("CMIND_STAGE_ID", "stage-env")
    monkeypatch.setenv("CMIND_HOOK", "post-commit")
    monkeypatch.setenv("CMIND_HOOK_SHA", "abc1234")
    update_graphs._log_hook_call("post-commit", {
        "run_id": "run-result",
        "stage_id": "stage-result",
        "mode": "incremental",
        "duration": 0.1,
    })

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["call_id"].startswith("hook_")
    assert row["run_id"] == "run-result"
    assert row["stage_id"] == "stage-result"
    assert row["hook_type"] == "post-commit"
    assert row["git_sha"] == "abc1234"
    started, finished = load_activity_events(tmp_path / "activity")
    assert started["kind"] == finished["kind"] == "hook.operation"
    assert finished["trigger"] == "hook"
    assert finished["mode"] == "incremental"


def test_mcp_path_without_workspace_does_not_use_package_workspace(tmp_path, monkeypatch):
    import mcp_server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp_server.sys, "argv", ["cmind-mcp"])

    assert mcp_server._resolve_rpg_path() == str(tmp_path / ".cmind/data/rpg.json")
    assert not os.path.isfile(mcp_server._resolve_rpg_path())


def test_update_rpg_run_records_hook_trigger(monkeypatch):
    import importlib

    module = importlib.import_module("rpg_encoder.run_update_rpg")
    captured = {}

    class FakeRun:
        run_id = "run-hook"
        status = "success"
        error = None

        def note(self, **metrics):
            captured["metrics"] = metrics

    @contextmanager
    def fake_record_run(command, *, trigger=None, metadata=None):
        captured.update(command=command, trigger=trigger, metadata=metadata)
        yield FakeRun()

    monkeypatch.setattr(module, "record_run", fake_record_run)
    monkeypatch.setattr(module, "_run_update_rpg", lambda **kwargs: {"status": "success"})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "parent123\n")
    monkeypatch.setenv("CMIND_HOOK", "post-commit")
    monkeypatch.setenv("CMIND_HOOK_SHA", "abc1234")
    monkeypatch.setenv("CMIND_HOOK_INVOCATION_ID", "spn_hook")

    result = module.run_update_rpg("rpg.json", "old", "new")

    assert result["status"] == "success"
    assert captured["trigger"] == "hook"
    assert captured["metadata"] == {
        "hook_type": "post-commit",
        "hook_sha": "abc1234",
        "hook_invocation_id": "spn_hook",
    }
    assert captured["metrics"]["prev_ref"] == "parent123"