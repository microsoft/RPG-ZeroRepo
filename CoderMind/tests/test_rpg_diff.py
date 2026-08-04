from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.rpg_diff import semantic_rpg_diff


def test_semantic_rpg_diff_reports_compact_nodes_and_edges():
    before = {
        "root": {
            "id": "repo",
            "name": "demo",
            "children": [{
                "id": "old-group",
                "name": "Old group",
                "children": [
                    {"id": "feature-a", "name": "Old name", "children": []},
                    {"id": "feature-removed", "name": "Removed", "children": []},
                ],
            }],
        },
        "edges": [{"src": "repo", "dst": "feature-a", "relation": "invokes", "meta": {"content": "large"}}],
        "dep_graph": {
            "nodes": {
                "a.py": {"type": "file"},
                "a.py:f": {"type": "function", "rpg_nodes": ["feature-a", "feature-removed"]},
            },
            "edges": [{"src": "a.py", "dst": "a.py:f", "attrs": {"type": "contains"}}],
        },
        "_dep_to_rpg_map": {"a.py:f": ["feature-a", "feature-removed"]},
    }
    after = {
        "root": {
            "id": "repo",
            "name": "demo",
            "children": [
                {"id": "new-group", "name": "New group", "children": [
                    {"id": "feature-a", "name": "New name", "children": []},
                    {"id": "feature-b", "name": "Added", "children": []},
                ]},
            ],
        },
        "edges": [{"src": "repo", "dst": "feature-b", "relation": "references", "meta": {"content": "large"}}],
        "dep_graph": {
            "nodes": {
                "a.py": {"type": "file"},
                "b.py": {"type": "file"},
                "a.py:f": {"type": "function", "rpg_nodes": ["feature-a", "feature-b"]},
            },
            "edges": [{"src": "b.py", "dst": "a.py:f", "attrs": {"type": "contains"}}],
        },
        "_dep_to_rpg_map": {"a.py:f": ["feature-a", "feature-b"]},
    }

    change = semantic_rpg_diff(before, after, commit="new", parent_commit="old")

    assert change["feature_nodes"]["counts"] == {
        "added": 2,
        "removed": 2,
        "modified": 1,
        "total": 4,
    }
    assert {node["node_id"] for node in change["feature_nodes"]["added"]} == {"feature-b", "new-group"}
    feature_a = next(node for node in change["feature_nodes"]["modified"] if node["node_id"] == "feature-a")
    assert feature_a["parent_id"] == "new-group"
    assert feature_a["previous_parent_id"] == "old-group"
    assert feature_a["changed_fields"] == ["name", "parent_id"]
    removed = next(node for node in change["feature_nodes"]["removed"] if node["node_id"] == "feature-removed")
    assert removed["parent_id"] == "old-group"
    dependency = next(node for node in change["dependency_nodes"]["modified"] if node["node_id"] == "a.py:f")
    assert dependency["parent_id"] == "b.py"
    assert dependency["previous_parent_id"] == "a.py"
    assert dependency["changed_fields"] == ["parent_id", "rpg_nodes"]
    assert change["semantic_edges"]["counts"] == {"added": 1, "removed": 1, "total": 1}
    assert change["semantic_edges"]["added"] == [{
        "source": "repo",
        "target": "feature-b",
        "relation": "references",
    }]
    assert "meta" not in change["semantic_edges"]["added"][0]
    assert {edge["target"] for edge in change["feature_hierarchy_edges"]["removed"]} == {
        "feature-a",
        "feature-removed",
        "old-group",
    }
    assert {edge["target"] for edge in change["feature_hierarchy_edges"]["added"]} == {
        "feature-a",
        "feature-b",
        "new-group",
    }
    assert change["mapping_edges"]["removed"] == [{
        "source": "a.py:f",
        "target": "feature-removed",
        "relation": "maps_to",
    }]
    assert change["mapping_edges"]["added"] == [{
        "source": "a.py:f",
        "target": "feature-b",
        "relation": "maps_to",
    }]