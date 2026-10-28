"""Search still works when a feature node's ``meta.path`` is a list.

``enrich_from_code`` stores a list when a group or area spans more than one
file or directory. Search and path normalisation must accept that shape.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from rpg.graph_query import GraphQueryEngine

_LIST_PATH = ["services/inference/router.py", "api/inference/routes.py"]


def _engine(path, code_dir: str = "") -> GraphQueryEngine:
    rpg = {
        "_dep_graph_code_dir": code_dir,
        "root": {
            "id": "repo",
            "name": "brain",
            "node_type": "repo",
            "meta": {"path": "."},
            "children": [
                {
                    "id": "area",
                    "name": "Inference Serving",
                    "node_type": "functional_area",
                    "meta": {"path": path, "description": "routes a request"},
                    "children": [],
                }
            ],
        },
    }
    dep = {
        "nodes": {
            "services/inference/router.py:route": {
                "name": "route",
                "type": "function",
            }
        },
        "edges": [],
    }
    return GraphQueryEngine(rpg, dep)


def _area(results: list[dict]) -> dict:
    matches = [item for item in results if item.get("id") == "area"]
    assert matches, results
    return matches[0]


def test_feature_search_matches_a_later_list_entry() -> None:
    """A query in the second path scores the path match, not a crash."""
    engine = _engine(_LIST_PATH)
    for scope in ("feature", "all"):
        hit = _area(engine.search("routes", scope=scope))
        assert hit["score"] == 60
        assert hit["path"] == _LIST_PATH


def test_string_path_search_is_unchanged() -> None:
    """A string path still matches and comes back as a string."""
    path = "services/inference/router.py"
    hit = _area(_engine(path).search("router", scope="feature"))
    assert hit["score"] == 60
    assert hit["path"] == path


def test_normalize_path_strips_prefix_from_each_list_entry() -> None:
    """A legacy code-dir prefix is removed from every path in a list."""
    engine = _engine(
        ["repo/services/inference", "repo/api/inference"],
        code_dir="repo",
    )
    assert engine._normalize_path(["repo/services/inference", "other/api"]) == [
        "services/inference",
        "other/api",
    ]
    assert engine._normalize_path("repo/services/inference/router.py") == (
        "services/inference/router.py"
    )


def test_tree_and_detail_keep_list_paths() -> None:
    """Browsing and detail return the same list search does."""
    engine = _engine(
        ["repo/services/inference", "repo/api/inference"],
        code_dir="repo",
    )
    expected = ["services/inference", "api/inference"]
    tree = engine.list_tree(max_depth=2)
    area = tree["children"][0]
    assert area["path"] == expected
    assert engine.get_node_detail("area")["code_path"] == expected
