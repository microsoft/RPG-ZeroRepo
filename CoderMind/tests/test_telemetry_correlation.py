from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_mcp_log_includes_inherited_correlation(tmp_path, monkeypatch):
    import mcp_server

    path = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(mcp_server, "MCP_CALLS_LOG", path)
    monkeypatch.setenv("CMIND_RUN_ID", "run-1")
    monkeypatch.setenv("CMIND_STAGE_ID", "stage-1")
    mcp_server._log_tool_call("search_rpg", {"query": "x"}, {"hits": 2}, 12)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["run_id"] == "run-1"
    assert row["stage_id"] == "stage-1"


def test_hook_log_prefers_result_correlation(tmp_path, monkeypatch):
    import update_graphs

    path = tmp_path / "hook_calls.jsonl"
    monkeypatch.setattr(update_graphs, "HOOK_CALLS_LOG", path)
    monkeypatch.setenv("CMIND_RUN_ID", "run-env")
    monkeypatch.setenv("CMIND_STAGE_ID", "stage-env")
    update_graphs._log_hook_call("post-commit", {
        "run_id": "run-result",
        "stage_id": "stage-result",
        "mode": "incremental",
        "duration": 0.1,
    })

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["run_id"] == "run-result"
    assert row["stage_id"] == "stage-result"