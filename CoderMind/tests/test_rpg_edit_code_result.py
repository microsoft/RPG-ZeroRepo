"""Persistence tests for the RPG Edit code-stage result."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rpg_edit.code import persist_code_result  # noqa: E402
from rpg_edit.review import persist_review_result  # noqa: E402


def test_persist_code_result_atomically_replaces_existing_result(tmp_path) -> None:
    path = tmp_path / "data" / "rpg_edit_code_result.json"
    path.parent.mkdir()
    path.write_text('{"success": false}\n', encoding="utf-8")
    result = {
        "type": "code_applied",
        "success": True,
        "files_modified": ["src/example.py"],
        "commit_sha": "abc123",
    }

    persist_code_result(result, path)

    assert json.loads(path.read_text(encoding="utf-8")) == result
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_persist_review_result_creates_standard_artifact(tmp_path) -> None:
    path = tmp_path / "data" / "rpg_edit_review_result.json"
    result = {"type": "impact_review", "success": True, "iterations": []}

    persist_review_result(result, path)

    assert json.loads(path.read_text(encoding="utf-8")) == result