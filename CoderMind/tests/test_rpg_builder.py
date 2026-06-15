from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rpg.builder import create_initial_rpg  # noqa: E402


def test_create_initial_rpg_preserves_target_language() -> None:
    rpg = create_initial_rpg({
        "repository_name": "tasklite",
        "repository_purpose": "Go CLI task tracker.",
        "meta": {"primary_language": "Go", "target_languages": ["Go"]},
        "components": [],
    })

    assert rpg.repo_node is not None
    assert rpg.repo_node.meta.language == "go"
