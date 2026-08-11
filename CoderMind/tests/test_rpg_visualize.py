from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rpg_visualize import generate_html


def test_generated_search_normalizes_array_paths() -> None:
    data = {
        "repo_name": "demo",
        "root": {
            "id": "root",
            "name": "demo",
            "meta": {"path": ["src", "demo"]},
            "children": [],
        },
    }

    html = generate_html(data)

    assert "function searchText(value)" in html
    assert "searchText(d.data.meta?.path)" in html
    assert "(d.data.meta?.path || '').toLowerCase()" not in html