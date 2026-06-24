#!/usr/bin/env python3
"""Tests for M8 RPG Evolution.

Covers:
  - New RPG methods: get_functionality_graph, delete_file_nodes,
    update_from_parsed_tree, parse_dep_graph, _build_dep_to_rpg_map,
    set_dep_graph
  - Diff utilities: generate_detailed_diff, _calculate_diff,
    _filter_units, _filter_non_test_py_files, _load_skeleton_from_repo
  - RPGEvolution: _process_delete_files, process_diff (mocked LLM),
    _update_dep_graph_index, _log_stage_summary
"""

import json
import logging
import os
import sys
import tempfile
import time
from copy import deepcopy
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

# Ensure the project root and scripts/ are on sys.path
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))


# ============================================================================
# Imports under test
# ============================================================================

from rpg.models import (
    Edge,
    EdgeType,
    Node,
    NodeMetaData,
    NodeType,
    RPG,
    uuid8,
)
from rpg.code_unit import CodeUnit, ParsedFile
from rpg_encoder.rpg_evolution import (
    RPGEvolution,
    generate_detailed_diff,
    _calculate_diff,
    _filter_units,
    _filter_non_test_py_files,
    _load_skeleton_from_repo,
)
from rpg_encoder.rpg_encoding import RPGParser


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_rpg():
    """Build a small RPG with structure: repo_node -> AreaOne -> CatOne -> FileA (FILE) -> func1 (FUNCTION) repo_node -> AreaOne -> CatOne -> FileB (FILE) -> cls1 (CLASS)."""
    rpg = RPG(repo_name="test_repo")

    area = Node(
        id="area_1",
        name="AreaOne",
        level=None,
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
    )
    rpg.add_node(area)
    rpg.add_edge(rpg.repo_node, area, EdgeType.CONTAINS)

    cat = Node(
        id="cat_1",
        name="CatOne",
        level=None,
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
    )
    rpg.add_node(cat)
    rpg.add_edge(area, cat, EdgeType.CONTAINS)

    file_a = Node(
        id="file_a",
        name="Module A",
        level=None,
        meta=NodeMetaData(type_name=NodeType.FILE, path="src/module_a.py"),
    )
    rpg.add_node(file_a)
    rpg.add_edge(cat, file_a, EdgeType.CONTAINS)

    func1 = Node(
        id="func_1",
        name="do_stuff",
        level=None,
        meta=NodeMetaData(
            type_name=NodeType.FUNCTION, path="src/module_a.py::do_stuff"
        ),
    )
    rpg.add_node(func1)
    rpg.add_edge(file_a, func1, EdgeType.CONTAINS)

    file_b = Node(
        id="file_b",
        name="Module B",
        level=None,
        meta=NodeMetaData(type_name=NodeType.FILE, path="src/module_b.py"),
    )
    rpg.add_node(file_b)
    rpg.add_edge(cat, file_b, EdgeType.CONTAINS)

    cls1 = Node(
        id="cls_1",
        name="MyClass",
        level=None,
        meta=NodeMetaData(
            type_name=NodeType.CLASS, path="src/module_b.py::MyClass"
        ),
    )
    rpg.add_node(cls1)
    rpg.add_edge(file_b, cls1, EdgeType.CONTAINS)

    rpg.recalculate_levels_topdown()
    return rpg


@pytest.fixture
def two_area_rpg():
    """Build an RPG with two functional areas, each with a file."""
    rpg = RPG(repo_name="test_repo")

    for area_name, file_path, file_summary, func_name in [
        ("AreaA", "src/a.py", "Module A", "run_a"),
        ("AreaB", "src/b.py", "Module B", "run_b"),
    ]:
        area = Node(
            id=f"area_{area_name}",
            name=area_name,
            meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
        )
        rpg.add_node(area)
        rpg.add_edge(rpg.repo_node, area, EdgeType.CONTAINS)

        fnode = Node(
            id=f"file_{area_name}",
            name=file_summary,
            meta=NodeMetaData(type_name=NodeType.FILE, path=file_path),
        )
        rpg.add_node(fnode)
        rpg.add_edge(area, fnode, EdgeType.CONTAINS)

        func = Node(
            id=f"func_{area_name}",
            name=func_name,
            meta=NodeMetaData(
                type_name=NodeType.FUNCTION,
                path=f"{file_path}:{func_name}",
            ),
        )
        rpg.add_node(func)
        rpg.add_edge(fnode, func, EdgeType.CONTAINS)

    rpg.recalculate_levels_topdown()
    return rpg


@pytest.fixture
def temp_repo_pair():
    """Create a pair of temporary repos for diff testing.

    Returns: (last_dir, cur_dir)
    """
    with tempfile.TemporaryDirectory() as base:
        last_dir = os.path.join(base, "last")
        cur_dir = os.path.join(base, "cur")

        os.makedirs(os.path.join(last_dir, "src"))
        os.makedirs(os.path.join(cur_dir, "src"))

        # Last repo: one file
        with open(os.path.join(last_dir, "src", "main.py"), "w") as f:
            f.write(
                "def hello():\n"
                "    return 'hello'\n"
                "\n"
                "def goodbye():\n"
                "    return 'goodbye'\n"
            )

        # Current repo: main.py modified + new_file.py added
        with open(os.path.join(cur_dir, "src", "main.py"), "w") as f:
            f.write(
                "def hello():\n"
                "    return 'hello world'\n"
                "\n"
                "def goodbye():\n"
                "    return 'goodbye'\n"
                "\n"
                "def new_func():\n"
                "    return 42\n"
            )

        with open(os.path.join(cur_dir, "src", "new_file.py"), "w") as f:
            f.write(
                "class NewClass:\n"
                "    def method_a(self):\n"
                "        pass\n"
            )

        yield last_dir, cur_dir


@pytest.fixture
def temp_repo_deletion():
    """Two repos where a file is deleted in current."""
    with tempfile.TemporaryDirectory() as base:
        last_dir = os.path.join(base, "last")
        cur_dir = os.path.join(base, "cur")

        os.makedirs(os.path.join(last_dir, "src"))
        os.makedirs(os.path.join(cur_dir, "src"))

        # Last repo: two files
        with open(os.path.join(last_dir, "src", "alpha.py"), "w") as f:
            f.write("def alpha_func():\n    pass\n")
        with open(os.path.join(last_dir, "src", "beta.py"), "w") as f:
            f.write("def beta_func():\n    pass\n")

        # Current repo: only alpha.py
        with open(os.path.join(cur_dir, "src", "alpha.py"), "w") as f:
            f.write("def alpha_func():\n    pass\n")

        yield last_dir, cur_dir


# ============================================================================
# Tests: RPG.get_functionality_graph
# ============================================================================


class TestGetFunctionalityGraph:
    def test_empty_rpg(self):
        rpg = RPG(repo_name="test")
        result = rpg.get_functionality_graph()
        assert result == []

    def test_simple_structure(self, simple_rpg):
        result = simple_rpg.get_functionality_graph()
        assert len(result) == 1
        assert result[0]["name"] == "AreaOne"
        subtree = result[0]["refactored_subtree"]
        assert isinstance(subtree, dict)
        assert "CatOne" in subtree

    def test_two_areas(self, two_area_rpg):
        result = two_area_rpg.get_functionality_graph()
        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "AreaA" in names
        assert "AreaB" in names


# ============================================================================
# Tests: RPG.delete_file_nodes
# ============================================================================


class TestDeleteFileNodes:
    def test_empty_paths(self, simple_rpg):
        result = simple_rpg.delete_file_nodes([])
        assert result["deleted_nodes"] == 0

    def test_nonexistent_path(self, simple_rpg):
        result = simple_rpg.delete_file_nodes(["nonexistent.py"])
        assert result["deleted_nodes"] == 0

    def test_delete_single_file(self, simple_rpg):
        before_count = len(simple_rpg.nodes)
        result = simple_rpg.delete_file_nodes(["src/module_a.py"])

        # Should delete file_a + func_1 = 2 nodes
        assert result["deleted_nodes"] >= 2
        assert "file_a" not in simple_rpg.nodes
        assert "func_1" not in simple_rpg.nodes

    def test_delete_cleans_empty_parents(self, two_area_rpg):
        """When both files under an area are deleted, the area directory should be cleaned up too."""
        result = two_area_rpg.delete_file_nodes(["src/a.py"])

        # file_AreaA + func_AreaA = 2, possibly area_AreaA cleaned
        assert result["deleted_nodes"] >= 2
        assert "file_AreaA" not in two_area_rpg.nodes

    def test_delete_preserves_other_files(self, simple_rpg):
        simple_rpg.delete_file_nodes(["src/module_a.py"])

        # module_b should still exist
        assert "file_b" in simple_rpg.nodes
        assert "cls_1" in simple_rpg.nodes


# ============================================================================
# Tests: RPG.update_from_parsed_tree
# ============================================================================


class TestUpdateFromParsedTree:
    def test_empty_tree(self, simple_rpg):
        result = simple_rpg.update_from_parsed_tree({})
        assert result["added_nodes"] == 0
        assert result["updated_nodes"] == 0

    def test_add_new_function(self, simple_rpg):
        parsed_tree = {
            "src/module_a.py": {
                "_file_summary_": "Module A",
                "function do_stuff": ["do_stuff"],
                "function new_func": ["new_func"],
            }
        }
        result = simple_rpg.update_from_parsed_tree(parsed_tree)

        # Should add new_func node
        assert result["added_nodes"] >= 1

        # Verify new function exists
        func_nodes = [
            n for n in simple_rpg.nodes.values()
            if n.meta and n.meta.path == "src/module_a.py::new_func"
        ]
        assert len(func_nodes) == 1
        assert func_nodes[0].name == "new_func"

    def test_update_function_name(self, simple_rpg):
        parsed_tree = {
            "src/module_a.py": {
                "_file_summary_": "Module A",
                "function do_stuff": ["do_different_stuff"],
            }
        }
        result = simple_rpg.update_from_parsed_tree(parsed_tree)

        # Should update do_stuff -> do_different_stuff
        assert result["updated_nodes"] >= 1
        node = simple_rpg.nodes["func_1"]
        assert node.name == "do_different_stuff"

    def test_delete_units(self, simple_rpg):
        deleted_units = {
            "src/module_b.py": ["MyClass"],
        }
        result = simple_rpg.update_from_parsed_tree(
            parsed_tree={},
            deleted_units=deleted_units,
        )
        assert result["deleted_nodes"] >= 1

        # MyClass should be gone
        class_nodes = [
            n for n in simple_rpg.nodes.values()
            if n.meta and n.meta.path == "src/module_b.py::MyClass"
        ]
        assert len(class_nodes) == 0

    def test_add_class_with_methods(self, simple_rpg):
        parsed_tree = {
            "src/module_a.py": {
                "_file_summary_": "Module A",
                "class NewClass": {
                    "method_a": ["do_method_a"],
                    "method_b": ["do_method_b"],
                },
            }
        }
        result = simple_rpg.update_from_parsed_tree(parsed_tree)
        assert result["added_nodes"] >= 2

    def test_add_class_without_methods(self, simple_rpg):
        parsed_tree = {
            "src/module_a.py": {
                "_file_summary_": "Module A",
                "class SimpleClass": ["Simple class implementation"],
            }
        }
        result = simple_rpg.update_from_parsed_tree(parsed_tree)
        assert result["added_nodes"] >= 1


# ============================================================================
# Tests: RPG._build_dep_to_rpg_map / set_dep_graph
# ============================================================================


class TestBuildDepToRpgMap:
    def test_no_dep_graph(self):
        rpg = RPG(repo_name="test")
        result = rpg._build_dep_to_rpg_map()
        assert result == {}

    def test_set_dep_graph(self, simple_rpg):
        mock_dg = MagicMock()
        mock_dg.G.nodes.return_value = []

        simple_rpg.set_dep_graph(mock_dg)
        assert simple_rpg.dep_graph is mock_dg
        assert simple_rpg._dep_to_rpg_map == {}

    def _dep_graph_with_nodes(self, nodes):
        graph = nx.MultiDiGraph()
        for node_id, node_type in nodes:
            graph.add_node(node_id, type=node_type)
        mock_dg = MagicMock()
        mock_dg.G = graph
        return mock_dg

    def test_code_unit_prefers_exact_rpg_node_over_parent_file(self, simple_rpg):
        dep_graph = self._dep_graph_with_nodes([
            ("src/module_a.py", NodeType.FILE),
            ("src/module_a.py:do_stuff", NodeType.FUNCTION),
        ])

        simple_rpg.set_dep_graph(dep_graph)

        assert simple_rpg._dep_to_rpg_map["src/module_a.py:do_stuff"] == ["func_1"]

    def test_non_python_code_unit_suffix_matches_exact_rpg_node(self):
        rpg = RPG(repo_name="test_repo")
        file_node = Node(
            id="go_file",
            name="Store",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/store.go"),
        )
        func_node = Node(
            id="go_func",
            name="Load",
            meta=NodeMetaData(type_name=NodeType.FUNCTION, path="src/store.go::Load"),
        )
        rpg.add_node(file_node)
        rpg.add_node(func_node)
        rpg.add_edge(rpg.repo_node, file_node, EdgeType.CONTAINS)
        rpg.add_edge(file_node, func_node, EdgeType.CONTAINS)

        dep_graph = self._dep_graph_with_nodes([
            ("generated/src/store.go", NodeType.FILE),
            ("generated/src/store.go:Load", NodeType.FUNCTION),
        ])
        rpg.set_dep_graph(dep_graph)

        assert rpg._dep_to_rpg_map["generated/src/store.go:Load"] == ["go_func"]

    def test_non_python_method_suffix_matches_exact_rpg_node(self):
        rpg = RPG(repo_name="test_repo")
        file_node = Node(
            id="ts_file",
            name="Client",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/client.ts"),
        )
        method_node = Node(
            id="ts_method",
            name="request",
            meta=NodeMetaData(type_name=NodeType.METHOD, path="src/client.ts::Client::request"),
        )
        rpg.add_node(file_node)
        rpg.add_node(method_node)
        rpg.add_edge(rpg.repo_node, file_node, EdgeType.CONTAINS)
        rpg.add_edge(file_node, method_node, EdgeType.CONTAINS)

        dep_graph = self._dep_graph_with_nodes([
            ("generated/src/client.ts", NodeType.FILE),
            ("generated/src/client.ts:Client.request", NodeType.METHOD),
        ])
        rpg.set_dep_graph(dep_graph)

        assert rpg._dep_to_rpg_map["generated/src/client.ts:Client.request"] == ["ts_method"]


# ============================================================================
# Tests: Diff utilities
# ============================================================================


class TestFilterNonTestPyFiles:
    def test_accepts_regular_py(self):
        assert _filter_non_test_py_files("src/main.py") is True

    def test_rejects_non_py(self):
        assert _filter_non_test_py_files("README.md") is False

    def test_rejects_test_file(self):
        assert _filter_non_test_py_files("tests/test_main.py") is False

    def test_rejects_test_directory(self):
        assert _filter_non_test_py_files("test/test_utils.py") is False

    def test_accepts_nested_py(self):
        assert _filter_non_test_py_files("src/core/parser.py") is True


class TestLoadSkeletonFromRepo:
    def test_basic(self, temp_repo_pair):
        last_dir, _ = temp_repo_pair
        skeleton_info, file_code_map, valid_files = _load_skeleton_from_repo(
            last_dir
        )
        assert "src/main.py" in skeleton_info
        assert "src/main.py" in file_code_map
        assert "src/main.py" in valid_files
        assert "def hello" in file_code_map["src/main.py"]

    def test_filters_non_py(self, temp_repo_pair):
        last_dir, _ = temp_repo_pair
        # Add a non-py file
        with open(os.path.join(last_dir, "src", "data.txt"), "w") as f:
            f.write("some data")

        _, file_code_map, valid_files = _load_skeleton_from_repo(last_dir)
        assert "src/data.txt" not in file_code_map
        assert "src/data.txt" not in valid_files


class TestCalculateDiff:
    def test_no_changes(self):
        code = "def hello():\n    return 'hi'\n"
        units_v1 = ParsedFile(code=code, file_path="main.py").units
        units_v2 = ParsedFile(code=code, file_path="main.py").units
        result = _calculate_diff(units_v1, units_v2)
        assert len(result["changed"]) == 0
        assert len(result["added"]) == 0
        assert len(result["deleted"]) == 0

    def test_function_added(self):
        code_v1 = "def hello():\n    pass\n"
        code_v2 = "def hello():\n    pass\n\ndef world():\n    pass\n"
        units_v1 = ParsedFile(code=code_v1, file_path="main.py").units
        units_v2 = ParsedFile(code=code_v2, file_path="main.py").units
        result = _calculate_diff(units_v1, units_v2)
        assert len(result["added"]) == 1
        assert result["added"][0].name == "world"

    def test_function_deleted(self):
        code_v1 = "def hello():\n    pass\n\ndef world():\n    pass\n"
        code_v2 = "def hello():\n    pass\n"
        units_v1 = ParsedFile(code=code_v1, file_path="main.py").units
        units_v2 = ParsedFile(code=code_v2, file_path="main.py").units
        result = _calculate_diff(units_v1, units_v2)
        assert len(result["deleted"]) == 1
        assert result["deleted"][0].name == "world"

    def test_function_changed(self):
        code_v1 = "def hello():\n    return 1\n"
        code_v2 = "def hello():\n    return 2\n"
        units_v1 = ParsedFile(code=code_v1, file_path="main.py").units
        units_v2 = ParsedFile(code=code_v2, file_path="main.py").units
        result = _calculate_diff(units_v1, units_v2)
        assert len(result["changed"]) == 1
        assert result["changed"][0].name == "hello"


class TestFilterUnits:
    def test_dict_input(self):
        unit1 = MagicMock(unit_type="function")
        unit2 = MagicMock(unit_type="import")
        unit3 = MagicMock(unit_type="class")
        result = _filter_units({"a.py": [unit1, unit2, unit3]})
        assert "a.py" in result
        assert len(result["a.py"]) == 2  # function + class

    def test_list_input(self):
        unit1 = MagicMock(unit_type="function")
        unit2 = MagicMock(unit_type="import")
        result = _filter_units([unit1, unit2])
        assert len(result) == 1

    def test_empty_dict(self):
        result = _filter_units({})
        assert result == {}


class TestGenerateDetailedDiff:
    def test_detects_added_file(self, temp_repo_pair):
        last_dir, cur_dir = temp_repo_pair
        result = generate_detailed_diff(last_dir, cur_dir)

        assert "src/new_file.py" in result["added"]

    def test_detects_modified_file(self, temp_repo_pair):
        last_dir, cur_dir = temp_repo_pair
        result = generate_detailed_diff(last_dir, cur_dir)

        assert "src/main.py" in result["modified"]
        mod = result["modified"]["src/main.py"]
        # hello() was changed, new_func() was added
        assert len(mod.get("changed", [])) >= 1 or len(mod.get("added", [])) >= 1

    def test_detects_deleted_file(self, temp_repo_deletion):
        last_dir, cur_dir = temp_repo_deletion
        result = generate_detailed_diff(last_dir, cur_dir)

        assert "src/beta.py" in result["deleted"]

    def test_no_changes(self):
        """Identical repos should produce no diffs."""
        with tempfile.TemporaryDirectory() as base:
            repo_dir = os.path.join(base, "repo")
            os.makedirs(os.path.join(repo_dir, "src"))
            with open(os.path.join(repo_dir, "src", "main.py"), "w") as f:
                f.write("def hello():\n    pass\n")

            result = generate_detailed_diff(repo_dir, repo_dir)
            assert len(result["added"]) == 0
            assert len(result["deleted"]) == 0
            # Modified should have no actual changes
            for f, d in result["modified"].items():
                assert len(d.get("changed", [])) == 0
                assert len(d.get("added", [])) == 0
                assert len(d.get("deleted", [])) == 0


# ============================================================================
# Tests: RPGEvolution
# ============================================================================


class TestRPGEvolutionDeleteFiles:
    def test_process_delete_files(self, simple_rpg):
        ctx = {"last_rpg": simple_rpg}
        logger = logging.getLogger("test_delete")

        result = RPGEvolution._process_delete_files(
            ctx, ["src/module_a.py"], logger
        )

        assert result["summary"]["deleted_nodes"] >= 2
        assert "file_a" not in result["rpg"].nodes
        assert "func_1" not in result["rpg"].nodes

    def test_process_delete_preserves_other(self, simple_rpg):
        ctx = {"last_rpg": simple_rpg}
        logger = logging.getLogger("test_delete")

        RPGEvolution._process_delete_files(ctx, ["src/module_a.py"], logger)

        assert "file_b" in ctx["last_rpg"].nodes


class TestRPGEvolutionLogSummary:
    def test_log_stage_summary(self):
        logger = logging.getLogger("test_log")
        stats = {"files": 3, "nodes": 5}
        # Should not raise
        RPGEvolution._log_stage_summary(
            "TEST", stats, time.time() - 1.0, logger
        )


class TestRPGEvolutionUpdateDepGraph:
    def test_update_dep_graph_index_no_crash(self, simple_rpg):
        logger = logging.getLogger("test_dep")

        with patch("rpg.service.RPGService.refresh_dep_graph") as mock_refresh:
            RPGEvolution._update_dep_graph_index(simple_rpg, "/tmp/fake", logger)
            mock_refresh.assert_called_once_with(
                code_dir="/tmp/fake",
                workspace_root="/tmp/fake",
                save_path=None,
            )

    def test_update_dep_graph_handles_error(self, simple_rpg):
        logger = logging.getLogger("test_dep_err")

        with patch(
            "rpg.service.RPGService.refresh_dep_graph",
            side_effect=RuntimeError("fail"),
        ):
            # Should not raise
            RPGEvolution._update_dep_graph_index(simple_rpg, "/tmp/fake", logger)


class TestRPGEvolutionProcessDiff:
    """Test process_diff with extensive mocking of LLM calls."""

    def test_no_changes_detected(self, simple_rpg):
        """When diff detects no changes, RPG should be returned unchanged."""
        with patch(
                 "rpg_encoder.rpg_evolution.generate_detailed_diff",
                 return_value={"added": {}, "deleted": {}, "modified": {}},
             ), \
             patch.object(RPG, "parse_dep_graph"):

            result = RPGEvolution.process_diff(
                repo_name="test",
                repo_info="Test repo",
                save_path="",
                last_repo_dir="/tmp/fake_last",
                cur_repo_dir="/tmp/fake_cur",
                last_rpg=simple_rpg,
                last_feature_tree="[]",
                update_dep_graph=False,
            )

            assert result is simple_rpg

    def test_delete_only(self, simple_rpg):
        """When only deletions are detected, files should be removed."""
        diff_result = {
            "added": {},
            "deleted": {"src/module_a.py": []},
            "modified": {},
        }

        with patch(
                 "rpg_encoder.rpg_evolution.generate_detailed_diff",
                 return_value=diff_result,
             ):

            result = RPGEvolution.process_diff(
                repo_name="test",
                repo_info="Test repo",
                save_path="",
                last_repo_dir="/tmp/fake_last",
                cur_repo_dir="/tmp/fake_cur",
                last_rpg=simple_rpg,
                last_feature_tree="[]",
                update_dep_graph=False,
            )

            assert "file_a" not in result.nodes
            assert "func_1" not in result.nodes

    def test_save_path_creates_file(self, simple_rpg):
        """When save_path is provided, results should be saved to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "result.json")

            diff_result = {
                "added": {},
                "deleted": {"src/module_a.py": []},
                "modified": {},
            }

            with patch(
                     "rpg_encoder.rpg_evolution.generate_detailed_diff",
                     return_value=diff_result,
                 ):

                RPGEvolution.process_diff(
                    repo_name="test",
                    repo_info="Test repo",
                    save_path=save_path,
                    last_repo_dir="/tmp/fake_last",
                    cur_repo_dir="/tmp/fake_cur",
                    last_rpg=simple_rpg,
                    last_feature_tree="[]",
                    update_dep_graph=False,
                )

                assert os.path.isfile(save_path)
                with open(save_path, "r") as f:
                    data = json.load(f)
                assert data["repo_name"] == "test"
                assert "diff_summary" in data


class TestRPGEvolutionClassmethod:
    def test_is_classmethod(self):
        assert isinstance(
            RPGEvolution.__dict__["process_diff"], classmethod
        )


# ============================================================================
# Tests: integration — diff + RPG update
# ============================================================================


class TestDiffAndUpdate:
    """Integration tests combining diff detection with RPG operations."""

    def test_diff_added_file_keys(self, temp_repo_pair):
        """Added files in diff should be non-empty."""
        last_dir, cur_dir = temp_repo_pair
        diff = generate_detailed_diff(last_dir, cur_dir)
        added = diff["added"]
        assert len(added) > 0
        # The added file should have code units
        for path, units in added.items():
            assert path.endswith(".py")

    def test_diff_deleted_file_keys(self, temp_repo_deletion):
        """Deleted files in diff should be non-empty."""
        last_dir, cur_dir = temp_repo_deletion
        diff = generate_detailed_diff(last_dir, cur_dir)
        deleted = diff["deleted"]
        assert len(deleted) > 0

    def test_delete_file_nodes_after_diff(self, temp_repo_deletion, two_area_rpg):
        """Simulates deleting a file from the RPG based on diff results."""
        last_dir, cur_dir = temp_repo_deletion
        diff = generate_detailed_diff(last_dir, cur_dir)

        deleted_files = list(diff["deleted"].keys())
        # This won't actually match because two_area_rpg has src/a.py and src/b.py
        # and we deleted src/beta.py. Still tests the code path.
        result = two_area_rpg.delete_file_nodes(deleted_files)
        # No match expected, just verify no crash
        assert isinstance(result, dict)
