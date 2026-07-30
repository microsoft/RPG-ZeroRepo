from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.dashboard_schema import sanitize_snapshot, validate_snapshot
from common.dashboard_snapshot import DashboardSources, build_dashboard_snapshot, write_dashboard_snapshot


def _sources(tmp_path: Path) -> DashboardSources:
    return DashboardSources(
        workspace_root=tmp_path / "demo",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        run_events_file=tmp_path / "logs" / "run_events.jsonl",
        rpg_file=tmp_path / "data" / "rpg.json",
        snapshot_file=tmp_path / "data" / "dashboard_snapshot.json",
    )


def test_sanitize_removes_sensitive_fields_and_values():
    sanitized = sanitize_snapshot({
        "prompt": "secret prompt",
        "nested": {
            "response": "secret response",
            "api_key": "secret-key",
            "detail": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "token_count": 123,
        },
        "graph": {"content": "source body", "description": "safe"},
    })
    assert sanitized == {
        "nested": {
            "detail": "Authorization: [REDACTED]",
            "token_count": 123,
        },
        "graph": {"description": "safe"},
    }


def test_build_snapshot_sanitizes_rpg_content(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.rpg_file.parent.mkdir(parents=True)
    sources.rpg_file.write_text(
        '{"repo_name":"demo","root":{"id":"repo","name":"demo","node_type":"repo",'
        '"meta":{"content":"private source","description":"safe description"},"children":[]}}',
        encoding="utf-8",
    )
    snapshot = build_dashboard_snapshot(sources)
    assert validate_snapshot(snapshot) == []
    assert "content" not in snapshot["graph"]["feature_root"]["meta"]
    assert snapshot["graph"]["feature_root"]["meta"]["description"] == "safe description"


def test_write_rejects_invalid_snapshot(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        write_dashboard_snapshot({"schema_version": 2}, tmp_path / "snapshot.json")