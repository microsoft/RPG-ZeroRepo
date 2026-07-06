from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rpg_visualize import generate_html


RAW_SCRIPT = "<script>alert(1)</script>"
SCRIPT_BREAKOUT = "</script><script>alert(1)</script>"


def _malicious_data() -> dict:
    feature_id = f"feature::{SCRIPT_BREAKOUT}"
    dep_id = f"pkg/mod.py:func::{SCRIPT_BREAKOUT}"
    other_dep_id = f"pkg/other.py:other::{RAW_SCRIPT}"
    return {
        "repo_name": f"Repo {RAW_SCRIPT}",
        "root": {
            "id": "root",
            "name": "Root",
            "node_type": "repository",
            "meta": {"path": "."},
            "children": [
                {
                    "id": feature_id,
                    "name": f"Feature {SCRIPT_BREAKOUT}",
                    "node_type": "feature",
                    "meta": {
                        "type_name": f"type {RAW_SCRIPT}",
                        "path": f"features/{RAW_SCRIPT}",
                        "description": f"Description {SCRIPT_BREAKOUT}",
                    },
                    "children": [],
                }
            ],
        },
        "edges": [
            {
                "src": feature_id,
                "dst": "root",
                "relation": f"references {RAW_SCRIPT}",
            }
        ],
        "dep_graph": {
            "nodes": {
                ".": {"name": ".", "type": "directory"},
                "pkg": {"name": f"pkg {RAW_SCRIPT}", "type": "directory"},
                "pkg/mod.py": {
                    "name": f"mod.py {RAW_SCRIPT}",
                    "type": "file",
                    "module": f"pkg.mod {RAW_SCRIPT}",
                },
                dep_id: {
                    "name": f"func {SCRIPT_BREAKOUT}",
                    "type": "function",
                    "module": f"pkg.mod {SCRIPT_BREAKOUT}",
                    "rpg_nodes": [f"rpg-node {RAW_SCRIPT}"],
                },
                other_dep_id: {
                    "name": f"other {RAW_SCRIPT}",
                    "type": "function",
                    "module": f"pkg.other {RAW_SCRIPT}",
                },
            },
            "edges": [
                {"src": ".", "dst": "pkg", "attrs": {"type": "contains"}},
                {"src": "pkg", "dst": "pkg/mod.py", "attrs": {"type": "contains"}},
                {"src": "pkg/mod.py", "dst": dep_id, "attrs": {"type": "contains"}},
                {"src": "pkg", "dst": other_dep_id, "attrs": {"type": "contains"}},
                {"src": dep_id, "dst": other_dep_id, "attrs": {"type": f"invokes {RAW_SCRIPT}"}},
            ],
        },
        "_dep_to_rpg_map": {dep_id: [feature_id]},
    }


def test_generate_html_escapes_html_and_script_contexts() -> None:
    html = generate_html(_malicious_data())

    assert "Repo &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert RAW_SCRIPT not in html
    assert SCRIPT_BREAKOUT not in html
    assert "https://d3js.org v7.9.0" in html
    assert '<script src="https://d3js.org' not in html
    assert 'src="https://d3js.org/d3.v7.min.js"' not in html


def test_generate_html_tooltip_builders_escape_graph_controlled_values() -> None:
    html = generate_html(_malicious_data())

    assert "escapeHtml(d.data.name || d.data.id)" in html
    assert "escapeHtml(tn)" in html
    assert "escapeHtml(path)" in html
    assert "escapeHtml(desc.slice(0, 200))" in html
    assert "connected.map(e => e.relation || 'unknown')" in html
    assert "map(escapeHtml).join(', ')" in html
    assert "escapeHtml(d.name || d.id)" in html
    assert "escapeHtml(d.module)" in html
    assert "escapeHtml(nodeId)" in html
    assert "escapeHtml(e.dep_id)" in html
    assert "escapeHtml(e.feat_id)" in html
