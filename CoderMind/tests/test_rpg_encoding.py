#!/usr/bin/env python3
"""Tests for M7 RPG Encoding.

Covers:
  - New RPG methods: update_result_to_rpg, remove_empty_subtrees,
    update_all_metadata_bottom_up, _iter_bottom_up_ids,
    _delete_root_level_file_subtrees
  - New utility functions: apply_changes, convert_leaves_to_list,
    get_rpg_info, exclude_files
  - Encoding prompt templates: all constants are non-empty and contain
    expected markers
  - RefactorTree: process_action, plan_functional_areas (mocked LLM)
  - RPGParser: generate_repo_info, exclude_irrelevant_files (mocked LLM)
"""

import json
import os
import sys
import tempfile
from copy import deepcopy
from unittest.mock import MagicMock, patch

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
)
from common.utils import (
    apply_changes,
    convert_leaves_to_list,
    exclude_files,
    get_rpg_info,
)
from rpg_encoder.prompts import (
    ANALYZE_DATA_FLOW,
    EXCLUDE_FILES as EXCLUDE_FILES_PROMPT,
    FUNCTIONAL_AREA,
    GENERATE_REPO_INFO,
    REFACTOR_MODIFIED,
    REFACTOR_TREE,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_rpg():
    """Build a small RPG with repo_node -> area -> cat -> subcat -> file -> func."""
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
        name="category",
        level=None,
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
    )
    rpg.add_node(cat)
    rpg.add_edge(area, cat, EdgeType.CONTAINS)

    subcat = Node(
        id="subcat_1",
        name="subcategory",
        level=None,
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
    )
    rpg.add_node(subcat)
    rpg.add_edge(cat, subcat, EdgeType.CONTAINS)

    file_node = Node(
        id="file_1",
        name="my_module",
        level=None,
        meta=NodeMetaData(type_name=NodeType.FILE, path="src/my_module.py"),
    )
    rpg.add_node(file_node)
    rpg.add_edge(subcat, file_node, EdgeType.CONTAINS)

    func_node = Node(
        id="func_1",
        name="do_something",
        level=None,
        meta=NodeMetaData(type_name=NodeType.FUNCTION, path="src/my_module.py:do_something"),
    )
    rpg.add_node(func_node)
    rpg.add_edge(file_node, func_node, EdgeType.CONTAINS)

    return rpg


@pytest.fixture
def empty_rpg():
    """RPG with only the repo_node."""
    return RPG(repo_name="empty")


# ============================================================================
# RPG Method Tests
# ============================================================================


class TestRPGUpdateResultToRPG:
    """Tests for RPG.update_result_to_rpg."""

    def test_creates_functional_area_and_subtree(self, empty_rpg):
        rpg = empty_rpg

        file_node = Node(
            id="file_a",
            name="file_summary_a",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/a.py"),
        )

        area_update = {
            "DataProcessing": {
                "DataProcessing/transform/normalize/file_summary_a": file_node,
            }
        }
        rpg.update_result_to_rpg(area_update)

        # The functional area should be L1
        fa_names = rpg.get_functional_areas()
        assert "DataProcessing" in fa_names

        # File node should be reachable in the tree
        assert file_node.id in rpg.nodes
        assert file_node._parent is not None

    def test_multiple_areas(self, empty_rpg):
        rpg = empty_rpg

        file_a = Node(
            id="f_a", name="summary_a",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/a.py"),
        )
        file_b = Node(
            id="f_b", name="summary_b",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/b.py"),
        )

        area_update = {
            "AreaA": {"AreaA/cat/sub/summary_a": file_a},
            "AreaB": {"AreaB/cat/sub/summary_b": file_b},
        }
        rpg.update_result_to_rpg(area_update)

        fa_names = rpg.get_functional_areas()
        assert "AreaA" in fa_names
        assert "AreaB" in fa_names

    def test_reuses_existing_area(self, simple_rpg):
        rpg = simple_rpg

        file_new = Node(
            id="f_new", name="new_file",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/new.py"),
        )
        area_update = {
            "AreaOne": {"AreaOne/cat2/sub2/new_file": file_new},
        }
        rpg.update_result_to_rpg(area_update)

        # Should still have only one AreaOne
        count = sum(
            1 for ch in rpg.repo_node._children if ch.name == "AreaOne"
        )
        assert count == 1


class TestRPGRemoveEmptySubtrees:
    """Tests for RPG.remove_empty_subtrees."""

    def test_removes_empty_l1(self, empty_rpg):
        rpg = empty_rpg
        empty_area = Node(
            id="empty_area", name="EmptyArea",
            meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."),
        )
        rpg.add_node(empty_area)
        rpg.add_edge(rpg.repo_node, empty_area, EdgeType.CONTAINS)

        result = rpg.remove_empty_subtrees()
        assert result["removed_nodes"] == 1

    def test_does_not_remove_non_empty_l1(self, simple_rpg):
        result = simple_rpg.remove_empty_subtrees()
        assert result["removed_nodes"] == 0


class TestRPGIterBottomUpIds:
    """Tests for RPG._iter_bottom_up_ids."""

    def test_order_is_postorder(self, simple_rpg):
        order = simple_rpg._iter_bottom_up_ids()
        # The leaf (func_1) should come before its parent (file_1)
        assert order.index("func_1") < order.index("file_1")
        # file before subcat
        assert order.index("file_1") < order.index("subcat_1")
        # repo_node should be last
        assert order[-1] == simple_rpg.repo_node.id


class TestRPGUpdateAllMetadataBottomUp:
    """Tests for RPG.update_all_metadata_bottom_up."""

    def test_updates_directory_paths(self, simple_rpg):
        rpg = simple_rpg
        updated = rpg.update_all_metadata_bottom_up()
        # At least the subcategory should be updated to reflect file's dir
        subcat = rpg.nodes["subcat_1"]
        # Its path should be derived from src/my_module.py -> "src"
        assert subcat.meta.path is not None
        assert updated > 0


class TestRPGDeleteRootLevelFileSubtrees:
    """Tests for RPG._delete_root_level_file_subtrees."""

    def test_removes_file_under_root(self, empty_rpg):
        rpg = empty_rpg
        stray_file = Node(
            id="stray_f", name="stray",
            meta=NodeMetaData(type_name=NodeType.FILE, path="stray.py"),
        )
        rpg.add_node(stray_file)
        rpg.add_edge(rpg.repo_node, stray_file, EdgeType.CONTAINS)

        result = rpg._delete_root_level_file_subtrees()
        assert result["deleted_nodes"] > 0
        assert "stray_f" not in rpg.nodes

    def test_no_op_when_no_stray_files(self, simple_rpg):
        result = simple_rpg._delete_root_level_file_subtrees()
        assert result["deleted_nodes"] == 0


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestApplyChanges:
    """Tests for apply_changes."""

    def test_basic_insert(self):
        tree = {}
        result = apply_changes(tree, "A/B/C")
        assert "A" in result
        assert "B" in result["A"]

    def test_multiple_paths(self):
        tree = {}
        result = apply_changes(tree, ["A/B/C", "A/B/D", "X/Y/Z"])
        assert "A" in result
        assert "C" in result["A"]["B"]  # list containing C and D
        assert "D" in result["A"]["B"]
        assert "X" in result

    def test_inplace_modification(self):
        tree = {"existing": ["val"]}
        result = apply_changes(tree, "new_key/sub", inplace=True)
        assert result is tree
        assert "new_key" in tree


class TestConvertLeavesToList:
    """Tests for convert_leaves_to_list."""

    def test_empty_list_becomes_empty_dict(self):
        result = convert_leaves_to_list({"a": []})
        assert result == {"a": {}}

    def test_non_empty_list_stays(self):
        result = convert_leaves_to_list({"a": ["x", "y"]})
        assert result == {"a": ["x", "y"]}

    def test_nested(self):
        result = convert_leaves_to_list({"a": {"b": []}})
        assert result == {"a": {"b": {}}}


class TestGetRPGInfo:
    """Tests for get_rpg_info."""

    def test_basic(self):
        rpg_tree = [
            {
                "name": "Area1",
                "refactored_subtree": {
                    "cat": {"sub": ["feat1", "feat2", "feat3"]}
                },
            }
        ]
        result = get_rpg_info(rpg_tree, omit_leaf_nodes=True, sample_size=0)
        parsed = json.loads(result)
        assert "Area1" in parsed

    def test_empty_tree(self):
        rpg_tree = [{"name": "Empty", "refactored_subtree": {}}]
        result = get_rpg_info(rpg_tree, omit_leaf_nodes=True, sample_size=0)
        parsed = json.loads(result)
        assert "Empty" in parsed


class TestExcludeFiles:
    """Tests for exclude_files utility."""

    def test_excludes_test_dirs(self):
        files = ["tests/test_foo.py", "src/main.py"]
        result = exclude_files(files)
        assert "tests/test_foo.py" in result
        assert "src/main.py" not in result

    def test_excludes_docs(self):
        files = ["docs/conf.py", "src/main.py"]
        result = exclude_files(files)
        assert "docs/conf.py" in result


# ============================================================================
# Encoding Prompt Template Tests
# ============================================================================


class TestEncodingPrompts:
    """Verify that encoding prompts are non-empty and have key markers."""

    def test_generate_repo_info_non_empty(self):
        assert len(GENERATE_REPO_INFO) > 100

    def test_generate_repo_info_has_solution_tag(self):
        assert "<solution>" in GENERATE_REPO_INFO

    def test_exclude_files_non_empty(self):
        assert len(EXCLUDE_FILES_PROMPT) > 100

    def test_exclude_files_has_solution_tag(self):
        assert "<solution>" in EXCLUDE_FILES_PROMPT

    def test_analyze_data_flow_non_empty(self):
        assert len(ANALYZE_DATA_FLOW) > 100

    def test_analyze_data_flow_has_placeholders(self):
        assert "{trees_names}" in ANALYZE_DATA_FLOW
        assert "{repo_name}" in ANALYZE_DATA_FLOW

    def test_refactor_tree_non_empty(self):
        assert len(REFACTOR_TREE) > 100

    def test_refactor_tree_has_solution_tag(self):
        assert "<solution>" in REFACTOR_TREE

    def test_refactor_modified_non_empty(self):
        assert len(REFACTOR_MODIFIED) > 100

    def test_refactor_modified_has_solution_tag(self):
        assert "<solution>" in REFACTOR_MODIFIED

    def test_functional_area_non_empty(self):
        assert len(FUNCTIONAL_AREA) > 100

    def test_functional_area_has_solution_tag(self):
        assert "<solution>" in FUNCTIONAL_AREA


# ============================================================================
# RefactorTree Tests (mocked LLM)
# ============================================================================


class TestRefactorTreeProcessAction:
    """Test RefactorTree.process_action without actual LLM calls."""

    def _make_refactor_tree(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.repo_name = "test_repo"
        rt.repo_dir = "/tmp/test"
        rt.repo_info = "test"
        rt.repo_skeleton = "test"
        rt.skeleton_info = ""
        rt.rpg = RPG(repo_name="test_repo")
        rt.logger = MagicMock()
        rt.llm_client = MagicMock()
        return rt

    def test_valid_action_processes_features(self):
        rt = self._make_refactor_tree()

        functional_areas = ["DataProcessing", "ModelTraining"]
        trans_tree = {
            "file_summary_a": ["feat1", "feat2"],
            "file_summary_b": ["feat3"],
        }
        cur_refactored_tree = [
            {"name": "DataProcessing", "refactored_subtree": {}},
            {"name": "ModelTraining", "refactored_subtree": {}},
        ]

        action = {
            "DataProcessing/transform/normalize": ["file_summary_a"],
            "ModelTraining/train/optimize": ["file_summary_b"],
        }

        processed_features = []
        env_prompt, updated_subtree, new_paths = rt.process_action(
            action=action,
            processed_features=processed_features,
            functional_areas=functional_areas,
            trans_tree=trans_tree,
            cur_refactored_tree=cur_refactored_tree,
        )

        assert "file_summary_a" in processed_features
        assert "file_summary_b" in processed_features
        assert len(new_paths) > 0

    def test_invalid_functional_area_reported(self):
        rt = self._make_refactor_tree()

        functional_areas = ["DataProcessing"]
        trans_tree = {"file_summary_a": ["feat1"]}
        cur_refactored_tree = [
            {"name": "DataProcessing", "refactored_subtree": {}},
        ]

        action = {
            "NonExistentArea/cat/sub": ["file_summary_a"],
        }

        processed_features = []
        env_prompt, _, _ = rt.process_action(
            action=action,
            processed_features=processed_features,
            functional_areas=functional_areas,
            trans_tree=trans_tree,
            cur_refactored_tree=cur_refactored_tree,
        )

        # Feature should NOT be processed
        assert "file_summary_a" not in processed_features
        # Env prompt should mention the issue
        assert "not recognized" in env_prompt.lower() or "invalid" in env_prompt.lower()

    def test_wrong_path_depth_rejected(self):
        rt = self._make_refactor_tree()

        functional_areas = ["DataProcessing"]
        trans_tree = {"file_summary_a": ["feat1"]}
        cur_refactored_tree = [
            {"name": "DataProcessing", "refactored_subtree": {}},
        ]

        # Path has 4 levels instead of 3
        action = {
            "DataProcessing/cat/sub/extra": ["file_summary_a"],
        }

        processed_features = []
        env_prompt, _, _ = rt.process_action(
            action=action,
            processed_features=processed_features,
            functional_areas=functional_areas,
            trans_tree=trans_tree,
            cur_refactored_tree=cur_refactored_tree,
        )

        assert "file_summary_a" not in processed_features


class TestRefactorTreePlanFunctionalAreas:
    """Test RefactorTree.plan_functional_areas with mocked LLM."""

    def test_plan_returns_candidates_and_final(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.repo_name = "test_repo"
        rt.repo_dir = "/tmp/test"
        rt.repo_info = "test"
        rt.repo_skeleton = "test"
        rt.skeleton_info = ""
        rt.rpg = RPG(repo_name="test_repo")
        rt.logger = MagicMock()

        mock_client = MagicMock()
        # First 3 calls: candidates; 4th call: synthesis
        mock_client.generate_with_memory.side_effect = [
            '<think>notes</think>\n<solution>\n["AreaA", "AreaB"]\n</solution>',
            '<think>notes</think>\n<solution>\n["AreaA", "AreaB"]\n</solution>',
            '<think>notes</think>\n<solution>\n["AreaA", "AreaB"]\n</solution>',
            '<think>final</think>\n<solution>\n["AreaA", "AreaB", "AreaC"]\n</solution>',
        ]
        rt.llm_client = mock_client

        parsed_tree = {
            "src/a.py": {"_file_summary_": "module a", "function foo": ["feat1"]},
            "src/b.py": {"_file_summary_": "module b", "function bar": ["feat2"]},
        }

        result = rt.plan_functional_areas(parsed_tree, max_iters=3)

        assert "candidates" in result
        assert "final_plan" in result
        assert len(result["candidates"]) > 0
        assert isinstance(result["final_plan"], list)


class TestRefactorTreeValidateModifiedAction:
    """Test RefactorTree._validate_modified_action."""

    def test_valid_mapping_no_feedback(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.logger = MagicMock()

        modified_input = {
            "Area/old_cat/old_sub/old_file": {
                "new_name": "new_file",
                "features": ["f1"],
            }
        }
        action = {
            "Area/old_cat/old_sub/old_file": "Area/old_cat/old_sub/new_file"
        }
        functional_areas = ["Area"]

        mapping, feedback = rt._validate_modified_action(
            action, modified_input, functional_areas
        )

        assert not feedback
        assert len(mapping) == 1

    def test_wrong_l1_gives_feedback(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.logger = MagicMock()

        modified_input = {
            "Area/cat/sub/old": {"new_name": "new", "features": ["f1"]}
        }
        action = {
            "Area/cat/sub/old": "WrongArea/cat/sub/new"
        }
        functional_areas = ["Area"]

        mapping, feedback = rt._validate_modified_action(
            action, modified_input, functional_areas
        )

        assert feedback
        assert "L1 must stay" in feedback

    def test_missing_files_gives_feedback(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.logger = MagicMock()

        modified_input = {
            "Area/cat/sub/a": {"new_name": "a_new", "features": ["f1"]},
            "Area/cat/sub/b": {"new_name": "b_new", "features": ["f2"]},
        }
        action = {
            "Area/cat/sub/a": "Area/cat/sub/a_new"
            # missing "Area/cat/sub/b"
        }
        functional_areas = ["Area"]

        mapping, feedback = rt._validate_modified_action(
            action, modified_input, functional_areas
        )

        assert feedback
        assert "Missing" in feedback


# ============================================================================
# RPGParser Tests (mocked LLM + filesystem)
# ============================================================================


class TestRPGParserGenerateRepoInfo:
    """Test RPGParser.generate_repo_info with mocked LLM."""

    def test_returns_repo_info(self):
        from rpg_encoder.rpg_encoding import RPGParser

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal repo
            os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
            with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
                f.write("print('hello')")
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# Test Project\nA test repo.")

            with patch.object(RPGParser, "__init__", lambda self, **kw: None):
                parser = RPGParser.__new__(RPGParser)
                parser.repo_dir = tmpdir
                parser.repo_name = "test_project"
                parser.logger = MagicMock()
                parser.skeleton_info = "src/main.py"
                parser.valid_files = ["src/main.py"]

                mock_client = MagicMock()
                mock_client.generate_with_memory.return_value = (
                    "<solution>\n```\nThis is a test project that prints hello.\n```\n</solution>"
                )
                parser.llm_client = mock_client

                result = parser.generate_repo_info(max_iters=1)

                assert "test project" in result.lower() or len(result) > 0
                mock_client.generate_with_memory.assert_called_once()


class TestRPGParserExcludeIrrelevantFiles:
    """Test RPGParser.exclude_irrelevant_files with mocked LLM."""

    def test_returns_exclude_list(self):
        from rpg_encoder.rpg_encoding import RPGParser

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
            with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
                f.write("print('hello')")
            with open(os.path.join(tmpdir, "tests", "test_main.py"), "w") as f:
                f.write("def test(): pass")

            with patch.object(RPGParser, "__init__", lambda self, **kw: None):
                parser = RPGParser.__new__(RPGParser)
                parser.repo_dir = tmpdir
                parser.repo_name = "test_project"
                parser.logger = MagicMock()
                parser.skeleton_info = "src/main.py\ntests/test_main.py"
                parser.valid_files = ["src/main.py", "tests/test_main.py"]

                mock_client = MagicMock()
                mock_client.generate_with_memory.side_effect = [
                    "<solution>\n```\ntests/\n```\n</solution>",
                    "<solution>\n```\ntests/\n```\n</solution>",
                ]
                parser.llm_client = mock_client

                result = parser.exclude_irrelevant_files(
                    repo_info="A test project", max_votes=1
                )

                # tests/ should be excluded (from LLM output or standard filter)
                assert any("test" in p for p in result)


class TestRPGParserLoadSkeleton:
    """Test RPGParser._load_skeleton_from_repo."""

    def test_finds_py_files(self):
        from rpg_encoder.rpg_encoding import RPGParser

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "pkg"), exist_ok=True)
            with open(os.path.join(tmpdir, "pkg", "mod.py"), "w") as f:
                f.write("x = 1")
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# Readme")

            with patch.object(RPGParser, "__init__", lambda self, **kw: None):
                parser = RPGParser.__new__(RPGParser)
                parser.repo_dir = tmpdir
                parser.repo_name = "test"
                parser.logger = MagicMock()

                skel_info, valid_files = parser._load_skeleton_from_repo()

                assert "pkg/mod.py" in valid_files
                assert "pkg/mod.py" in skel_info
                assert "README.md" in skel_info

    def test_skips_hidden_dirs(self):
        from rpg_encoder.rpg_encoding import RPGParser

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".hidden"), exist_ok=True)
            with open(os.path.join(tmpdir, ".hidden", "secret.py"), "w") as f:
                f.write("x = 1")
            with open(os.path.join(tmpdir, "visible.py"), "w") as f:
                f.write("y = 2")

            with patch.object(RPGParser, "__init__", lambda self, **kw: None):
                parser = RPGParser.__new__(RPGParser)
                parser.repo_dir = tmpdir
                parser.repo_name = "test"
                parser.logger = MagicMock()

                _, valid_files = parser._load_skeleton_from_repo()

                assert "visible.py" in valid_files
                assert not any(".hidden" in f for f in valid_files)


# ============================================================================
# Integration-level test: update_result_to_rpg end-to-end
# ============================================================================


class TestUpdateResultToRPGEndToEnd:
    """Simulate a mini refactoring pipeline to verify tree correctness."""

    def test_full_flow(self):
        rpg = RPG(repo_name="demo")

        # Create file nodes
        file_a = Node(
            id="fa", name="auth manager",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/auth.py"),
        )
        func_a = Node(
            id="fn_a", name="login_feature",
            meta=NodeMetaData(type_name=NodeType.FUNCTION, path="src/auth.py:login"),
        )
        file_b = Node(
            id="fb", name="data loader",
            meta=NodeMetaData(type_name=NodeType.FILE, path="src/data.py"),
        )
        func_b = Node(
            id="fn_b", name="load_feature",
            meta=NodeMetaData(type_name=NodeType.FUNCTION, path="src/data.py:load"),
        )

        rpg.add_node(file_a)
        rpg.add_node(func_a)
        rpg.add_edge(file_a, func_a, EdgeType.CONTAINS)

        rpg.add_node(file_b)
        rpg.add_node(func_b)
        rpg.add_edge(file_b, func_b, EdgeType.CONTAINS)

        # Simulate refactoring result
        area_update = {
            "Authentication": {
                "Authentication/user_mgmt/session/auth manager": file_a,
            },
            "DataPipeline": {
                "DataPipeline/ingest/read/data loader": file_b,
            },
        }

        rpg.update_result_to_rpg(area_update)

        # Verify structure
        fa_names = rpg.get_functional_areas()
        assert "Authentication" in fa_names
        assert "DataPipeline" in fa_names

        # File nodes should have parents
        assert file_a._parent is not None
        assert file_b._parent is not None

        # Function nodes should still be under file nodes
        assert func_a._parent.id == file_a.id
        assert func_b._parent.id == file_b.id

        # Metadata should be updated bottom-up
        updated = rpg.update_all_metadata_bottom_up()
        assert updated >= 0

        # Visualize should work without error
        vis = rpg.visualize_dir_map(max_depth=5)
        assert len(vis) > 0


# ============================================================================
# RefactorTree — additional coverage tests
# ============================================================================


class TestRefactorTreeInit:
    """Test RefactorTree.__init__ initializes all attributes correctly."""

    def test_refactor_tree_init(self):
        from rpg_encoder.refactor_tree import RefactorTree

        with patch(
            "rpg_encoder.refactor_tree.LLMClient"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            rt = RefactorTree(
                repo_dir="/tmp/test_repo",
                repo_info="A test repository",
                repo_skeleton="src/main.py\nsrc/utils.py",
                repo_name="test_project",
                skeleton_info="skeleton info",
            )

            assert rt.repo_name == "test_project"
            assert rt.repo_dir == "/tmp/test_repo"
            assert rt.repo_info == "A test repository"
            assert rt.repo_skeleton == "src/main.py\nsrc/utils.py"
            assert rt.skeleton_info == "skeleton info"
            assert isinstance(rt.rpg, RPG)
            assert rt.rpg.repo_name == "test_project"
            assert rt.logger is not None
            assert rt.llm_client is mock_instance


class TestRefactorTreeStep:
    """Test RefactorTree.step parses LLM output into action dict."""

    def test_refactor_tree_step(self):
        from rpg_encoder.refactor_tree import RefactorTree
        from common.llm_types import Memory, SystemMessage, UserMessage

        rt = RefactorTree.__new__(RefactorTree)
        rt.repo_name = "test_repo"
        rt.logger = MagicMock()

        action_json = json.dumps({
            "DataProcessing/transform/normalize": ["file_summary_a"],
        })
        llm_response = f"<think>analysis</think>\n<solution>\n{action_json}\n</solution>"

        mock_client = MagicMock()
        mock_client.generate_with_memory.return_value = llm_response
        rt.llm_client = mock_client

        memory = Memory()
        memory._history.append(SystemMessage(content="System prompt"))
        memory._history.append(UserMessage(content="User request"))

        action, response = rt.step(memory)

        assert isinstance(action, dict)
        assert "DataProcessing/transform/normalize" in action
        assert action["DataProcessing/transform/normalize"] == ["file_summary_a"]
        assert response == llm_response
        mock_client.generate_with_memory.assert_called_once_with(memory)


class TestRefactorTreeEstimateBatchTokens:
    """Test RefactorTree._estimate_batch_tokens_for_process_folder."""

    def test_estimate_batch_tokens(self):
        from rpg_encoder.refactor_tree import RefactorTree

        rt = RefactorTree.__new__(RefactorTree)
        rt.repo_name = "test_repo"
        rt.logger = MagicMock()

        functional_areas = ["DataProcessing", "ModelTraining"]
        cur_feature_tree = [
            {"name": "DataProcessing", "refactored_subtree": {}},
            {"name": "ModelTraining", "refactored_subtree": {}},
        ]
        folder_sub_tree = {
            "src/module_a.py": {
                "_file_summary_": "Module A handles data loading",
                "function load_data": ["loads CSV files", "validates schema"],
            },
            "src/module_b.py": {
                "_file_summary_": "Module B trains models",
                "function train": ["trains the model"],
            },
        }

        token_count = rt._estimate_batch_tokens_for_process_folder(
            functional_areas=functional_areas,
            folder_path="src/",
            cur_feature_tree=cur_feature_tree,
            folder_sub_tree=folder_sub_tree,
        )

        assert isinstance(token_count, int)
        assert token_count > 0
