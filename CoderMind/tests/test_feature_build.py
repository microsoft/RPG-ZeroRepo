from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from feature_build import _load_feature_data, apply_changes  # noqa: E402


def test_apply_changes_promotes_dict_list_leaf_to_branch() -> None:
    tree = {
        "ui": {
            "homepage": [
                {
                    "name": "render recent todos",
                    "description": "Render existing todo items.",
                    "children": [],
                }
            ]
        }
    }

    result = apply_changes(
        tree,
        ["ui/homepage/render recent todos/escape todo title"],
    )

    assert result["ui"]["homepage"] == {
        "render recent todos": ["escape todo title"]
    }
    assert tree["ui"]["homepage"][0]["name"] == "render recent todos"


def test_apply_changes_preserves_single_key_dict_leaf_when_promoted() -> None:
    tree = {"storage": {"file": [{"load dataset": []}]}}

    result = apply_changes(tree, ["storage/file/load dataset/handle corrupt json"])

    assert result["storage"]["file"] == {
        "load dataset": ["handle corrupt json"]
    }


def test_load_feature_data_preserves_target_languages(tmp_path) -> None:
    feature_spec = tmp_path / "feature_spec.json"
    feature_build = tmp_path / "feature_build.json"
    feature_spec.write_text(
        json.dumps({
            "repository_name": "tasklite",
            "repository_purpose": "Go CLI task tracker.",
            "meta": {
                "primary_language": "go",
                "target_languages": ["go"],
            },
            "functional_requirements": [],
        }),
        encoding="utf-8",
    )

    data = _load_feature_data(feature_build, feature_spec)

    assert data["meta"]["primary_language"] == "go"
    assert data["meta"]["target_languages"] == ["go"]