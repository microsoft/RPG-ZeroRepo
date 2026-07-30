from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _assert_atomic_result(path: Path, expected: dict) -> None:
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_validate_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import validate

    path = tmp_path / "rpg_edit_validate.json"
    monkeypatch.setattr(validate, "RPG_EDIT_VALIDATE_FILE", path)
    result = {"type": "error", "error_code": "rpg_not_found", "message": "missing"}
    validate._write_validate_result(result)
    _assert_atomic_result(path, result)


def test_locate_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import locate

    path = tmp_path / "rpg_edit_locate.json"
    monkeypatch.setattr(locate, "RPG_EDIT_LOCATE_FILE", path)
    result = {"type": "candidates", "query": "service", "results": []}
    locate._write_locate_result(result)
    _assert_atomic_result(path, result)


def test_code_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import code

    path = tmp_path / "rpg_edit_code_result.json"
    monkeypatch.setattr(code, "RPG_EDIT_CODE_RESULT_FILE", path)
    result = {"type": "code_applied", "success": True, "files_modified": ["src/a.py"]}
    code._write_code_result(result)
    _assert_atomic_result(path, result)


def test_apply_result_merges_split_phases_atomically(tmp_path, monkeypatch):
    from rpg_edit import apply

    path = tmp_path / "rpg_edit_apply_result.json"
    monkeypatch.setattr(apply, "RPG_EDIT_APPLY_RESULT_FILE", path)
    first = apply._record_apply_result(
        {"type": "rpg_updated"},
        backup_timestamp="123",
        backups={"rpg": "/tmp/rpg.backup"},
        applied_features=[{"node_id": "feature-1", "action": "modified"}],
        before_state={"head_branch": "rpg-edit/demo", "head_commit": "abc"},
    )
    second = apply._record_apply_result(
        {"type": "dep_refreshed", "dep_graph_refreshed": True},
        backup_timestamp="123",
    )

    assert first["rollback_path"] == "/tmp/rpg.backup"
    assert second["applied_features"] == first["applied_features"]
    assert second["backups"] == first["backups"]
    assert second["before_state"] == first["before_state"]
    assert "--rollback 123" in second["rollback_command"]
    assert "--rollback-branch rpg-edit/demo" in second["rollback_command"]
    _assert_atomic_result(path, second)


def test_review_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import review

    path = tmp_path / "rpg_edit_review_result.json"
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", path)
    result = {"type": "skipped", "reason": "impact too small"}
    review._write_review_result(result)
    _assert_atomic_result(path, result)