#!/usr/bin/env python3
"""Tests for M13 Workflow Integration.

Covers:
  - CMindConfig: load from YAML, from_dict, defaults, validation, save
  - RPGVersionControl: save_version, rollback, diff, list_versions, pruning
  - WorkflowIntegration: prepare_for_codegen, merge_generated_code,
    save_rpg, load_rpg, detect_workflow_mode
  - Internal helpers: _resolve_node, _infer_rpg_source, _gather_existing_interfaces
"""

import json
import os
import sys
import tempfile
from pathlib import Path
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
    uuid8,
)
from rpg_encoder.config import (
    CMindConfig,
    WorkflowConfig,
    EncodeConfig,
    CodegenConfig,
    VersioningConfig,
    CONFIG_FILE_NAME,
    CMIND_DIR_NAME,
    _parse_workflow,
)
from rpg_encoder.version_control import (
    RPGVersionControl,
    _make_version_filename,
    _parse_version_from_filename,
    _collect_node_ids,
    _collect_edge_tuples,
    _compare_shared_node_metadata,
)
from rpg_encoder.workflow import (
    WorkflowIntegration,
    _resolve_node,
    _build_node_context,
    _gather_existing_interfaces,
    _gather_dependency_edges,
    _infer_rpg_source,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_rpg():
    """Build a small RPG with structure: repo_node -> AreaOne -> FileA (FILE) -> ClassFoo (CLASS) repo_node -> AreaOne -> FileA (FILE) -> func_bar (FUNCTION)."""
    rpg = RPG(repo_name="test_repo")

    area = Node(
        id="area_1",
        name="AreaOne",
        level=None,
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src"),
    )
    rpg.add_node(area)
    rpg.add_edge(rpg.repo_node, area, EdgeType.CONTAINS)

    file_a = Node(
        id="file_a",
        name="foo.py",
        node_type="feature",
        level=None,
        meta=NodeMetaData(
            type_name=NodeType.FILE,
            path="src/foo.py",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(file_a)
    rpg.add_edge(area, file_a, EdgeType.CONTAINS)

    class_foo = Node(
        id="class_foo",
        name="Foo",
        node_type="feature",
        level=None,
        meta=NodeMetaData(
            type_name=NodeType.CLASS,
            path="src/foo.py::Foo",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(class_foo)
    rpg.add_edge(file_a, class_foo, EdgeType.CONTAINS)

    func_bar = Node(
        id="func_bar",
        name="bar",
        node_type="feature",
        level=None,
        meta=NodeMetaData(
            type_name=NodeType.FUNCTION,
            path="src/foo.py::bar",
            generator="rpg_encoder",
        ),
    )
    rpg.add_node(func_bar)
    rpg.add_edge(file_a, func_bar, EdgeType.CONTAINS)

    # Add a non-containment edge
    rpg.add_edge("class_foo", "func_bar", EdgeType.INVOKES)

    return rpg


@pytest.fixture
def tmp_cmind_dir():
    """Create a temporary .cmind directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmind_dir = os.path.join(tmpdir, CMIND_DIR_NAME)
        os.makedirs(cmind_dir, exist_ok=True)
        yield tmpdir, cmind_dir


# ============================================================================
# Tests: CMindConfig
# ============================================================================


class TestCMindConfig:
    """Tests for the configuration module."""

    def test_default_config(self):
        """Default config has expected values."""
        config = CMindConfig()
        assert config.workflow.default_mode == "mixed"
        assert config.workflow.encode.auto_exclude == ["tests/", "docs/"]
        assert config.workflow.encode.run_data_flow is False
        assert config.workflow.codegen.style == "pythonic"
        assert config.workflow.codegen.include_tests is True
        assert config.workflow.versioning.enabled is True
        assert config.workflow.versioning.max_history == 10

    def test_from_dict_full(self):
        """Config can be created from a complete dictionary."""
        data = {
            "workflow": {
                "default_mode": "forward",
                "encode": {
                    "auto_exclude": ["vendor/"],
                    "run_data_flow": True,
                },
                "codegen": {
                    "style": "google",
                    "include_tests": False,
                },
                "versioning": {
                    "enabled": False,
                    "max_history": 5,
                },
            }
        }
        config = CMindConfig.from_dict(data)
        assert config.workflow.default_mode == "forward"
        assert config.workflow.encode.auto_exclude == ["vendor/"]
        assert config.workflow.encode.run_data_flow is True
        assert config.workflow.codegen.style == "google"
        assert config.workflow.codegen.include_tests is False
        assert config.workflow.versioning.enabled is False
        assert config.workflow.versioning.max_history == 5

    def test_from_dict_partial(self):
        """Missing keys use defaults."""
        data = {"workflow": {"default_mode": "reverse"}}
        config = CMindConfig.from_dict(data)
        assert config.workflow.default_mode == "reverse"
        # Defaults for sub-configs
        assert config.workflow.encode.auto_exclude == ["tests/", "docs/"]
        assert config.workflow.versioning.max_history == 10

    def test_from_dict_empty(self):
        """Empty dict gives full defaults."""
        config = CMindConfig.from_dict({})
        assert config.workflow.default_mode == "mixed"

    def test_invalid_default_mode_falls_back(self):
        """Invalid default_mode falls back to 'mixed'."""
        data = {"workflow": {"default_mode": "invalid_mode"}}
        config = CMindConfig.from_dict(data)
        assert config.workflow.default_mode == "mixed"

    def test_to_dict_roundtrip(self):
        """to_dict produces data that from_dict can consume back."""
        original = CMindConfig.from_dict({
            "workflow": {
                "default_mode": "forward",
                "encode": {"auto_exclude": ["build/"]},
                "versioning": {"max_history": 20},
            }
        })
        exported = original.to_dict()
        restored = CMindConfig.from_dict(exported)
        assert restored.workflow.default_mode == "forward"
        assert restored.workflow.encode.auto_exclude == ["build/"]
        assert restored.workflow.versioning.max_history == 20

    def test_load_with_yaml_file(self, tmp_cmind_dir):
        """Config.load reads from .cmind/config.yaml."""
        repo_dir, cmind_dir = tmp_cmind_dir
        config_content = {
            "workflow": {
                "default_mode": "reverse",
                "versioning": {"max_history": 3},
            }
        }
        config_path = os.path.join(cmind_dir, CONFIG_FILE_NAME)
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config_content, f)
        except ImportError:
            pytest.skip("PyYAML not installed")

        config = CMindConfig.load(repo_dir)
        assert config.workflow.default_mode == "reverse"
        assert config.workflow.versioning.max_history == 3
        assert config.config_path == config_path

    def test_load_no_file_gives_defaults(self, tmp_cmind_dir):
        """Config.load returns defaults when no config file exists."""
        repo_dir, _ = tmp_cmind_dir
        config = CMindConfig.load(repo_dir)
        assert config.workflow.default_mode == "mixed"
        assert config.config_path is None

    def test_save_and_load(self, tmp_cmind_dir):
        """save() creates a YAML file that load() can read."""
        repo_dir, cmind_dir = tmp_cmind_dir
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        config = CMindConfig.from_dict(
            {"workflow": {"default_mode": "forward"}},
            cmind_dir=cmind_dir,
        )
        saved_path = config.save()
        assert os.path.isfile(saved_path)

        loaded = CMindConfig.load(repo_dir)
        assert loaded.workflow.default_mode == "forward"

    def test_parse_workflow_none(self):
        """_parse_workflow handles None input."""
        result = _parse_workflow(None)
        assert result.default_mode == "mixed"


# ============================================================================
# Tests: RPGVersionControl
# ============================================================================


class TestRPGVersionControl:
    """Tests for the version control module."""

    def test_make_version_filename(self):
        assert _make_version_filename(1) == "rpg.v1.json"
        assert _make_version_filename(42) == "rpg.v42.json"

    def test_parse_version_from_filename(self):
        assert _parse_version_from_filename("rpg.v1.json") == 1
        assert _parse_version_from_filename("rpg.v42.json") == 42
        assert _parse_version_from_filename("rpg.json") is None
        assert _parse_version_from_filename("other.txt") is None
        assert _parse_version_from_filename("rpg.vabc.json") is None

    def test_save_and_list(self, simple_rpg, tmp_cmind_dir):
        """save_version creates files; list_versions enumerates them."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir, max_history=10)

        v1 = vc.save_version(simple_rpg, message="First version")
        assert v1 == 1

        v2 = vc.save_version(simple_rpg, message="Second version")
        assert v2 == 2

        versions = vc.list_versions()
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[0]["message"] == "First version"
        assert versions[1]["version"] == 2

    def test_save_with_source(self, simple_rpg, tmp_cmind_dir):
        """save_version stores source metadata."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        vc.save_version(simple_rpg, message="Encoded", source="encoded")

        versions = vc.list_versions()
        assert versions[0]["source"] == "encoded"

    def test_rollback(self, simple_rpg, tmp_cmind_dir):
        """Rollback loads RPG from a saved version."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        vc.save_version(simple_rpg, message="Original")

        restored = vc.rollback(version=1)
        assert isinstance(restored, RPG)
        assert restored.repo_name == "test_repo"
        # Check that main rpg.json was also written
        main_rpg = os.path.join(cmind_dir, "data", "rpg.json")
        assert os.path.isfile(main_rpg)

    def test_rollback_nonexistent_raises(self, tmp_cmind_dir):
        """Rollback raises FileNotFoundError for missing versions."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        with pytest.raises(FileNotFoundError):
            vc.rollback(version=999)

    def test_diff_nodes(self, tmp_cmind_dir):
        """Diff detects added/removed nodes between versions."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)

        # Version 1: small RPG
        rpg1 = RPG(repo_name="test")
        vc.save_version(rpg1, message="V1")

        # Version 2: add a node
        rpg2 = RPG(repo_name="test")
        extra_node = Node(
            id="extra_node",
            name="Extra",
            level=None,
            meta=NodeMetaData(type_name=NodeType.FUNCTION, path="extra.py"),
        )
        rpg2.add_node(extra_node)
        rpg2.add_edge(rpg2.repo_node, extra_node, EdgeType.CONTAINS)
        vc.save_version(rpg2, message="V2")

        diff = vc.diff(version1=1, version2=2)
        assert "extra_node" in diff["nodes_added"]
        assert diff["summary"]["nodes_added"] >= 1

    def test_diff_edges(self, tmp_cmind_dir):
        """Diff detects added/removed non-containment edges."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)

        rpg1 = RPG(repo_name="test")
        n1 = Node(id="n1", name="A", level=None)
        n2 = Node(id="n2", name="B", level=None)
        rpg1.add_node(n1)
        rpg1.add_node(n2)
        rpg1.add_edge(rpg1.repo_node, n1, EdgeType.CONTAINS)
        rpg1.add_edge(rpg1.repo_node, n2, EdgeType.CONTAINS)
        vc.save_version(rpg1, message="V1")

        # V2: add an invokes edge
        rpg2 = RPG(repo_name="test")
        n1b = Node(id="n1", name="A", level=None)
        n2b = Node(id="n2", name="B", level=None)
        rpg2.add_node(n1b)
        rpg2.add_node(n2b)
        rpg2.add_edge(rpg2.repo_node, n1b, EdgeType.CONTAINS)
        rpg2.add_edge(rpg2.repo_node, n2b, EdgeType.CONTAINS)
        rpg2.add_edge("n1", "n2", EdgeType.INVOKES)
        vc.save_version(rpg2, message="V2")

        diff = vc.diff(1, 2)
        assert diff["summary"]["edges_added"] >= 1

    def test_diff_nonexistent_raises(self, tmp_cmind_dir):
        """Diff raises FileNotFoundError for missing versions."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        with pytest.raises(FileNotFoundError):
            vc.diff(1, 2)

    def test_max_history_prunes(self, simple_rpg, tmp_cmind_dir):
        """save_version prunes old versions when max_history is exceeded."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir, max_history=3)

        for i in range(5):
            vc.save_version(simple_rpg, message=f"Version {i+1}")

        versions = vc.list_versions()
        assert len(versions) == 3
        # Oldest versions should have been pruned
        assert versions[0]["version"] == 3

    def test_get_latest_version(self, simple_rpg, tmp_cmind_dir):
        """get_latest_version returns the highest version number."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        assert vc.get_latest_version() is None

        vc.save_version(simple_rpg, message="V1")
        assert vc.get_latest_version() == 1

        vc.save_version(simple_rpg, message="V2")
        assert vc.get_latest_version() == 2

    def test_list_versions_empty_dir(self, tmp_cmind_dir):
        """list_versions returns empty list for fresh directory."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        assert vc.list_versions() == []


# ============================================================================
# Tests: Version control diff helpers
# ============================================================================


class TestDiffHelpers:
    """Tests for version_control diff utility functions."""

    def test_collect_node_ids_nested(self):
        """Collect IDs from nested tree format."""
        rpg_dict = {
            "root": {
                "id": "root_1",
                "children": [
                    {"id": "child_1", "children": []},
                    {"id": "child_2", "children": [
                        {"id": "grandchild_1", "children": []}
                    ]},
                ],
            },
            "edges": [],
        }
        ids = _collect_node_ids(rpg_dict)
        assert ids == {"root_1", "child_1", "child_2", "grandchild_1"}

    def test_collect_node_ids_flat(self):
        """Collect IDs from flat format."""
        rpg_dict = {
            "nodes": [
                {"id": "n1"},
                {"id": "n2"},
            ],
            "edges": [],
        }
        ids = _collect_node_ids(rpg_dict)
        assert ids == {"n1", "n2"}

    def test_collect_edge_tuples(self):
        """Collect edge tuples from an RPG dict."""
        rpg_dict = {
            "edges": [
                {"src": "a", "dst": "b", "relation": "invokes"},
                {"src": "b", "dst": "c", "relation": "inherits"},
            ]
        }
        tuples = _collect_edge_tuples(rpg_dict)
        assert ("a", "b", "invokes") in tuples
        assert ("b", "c", "inherits") in tuples

    def test_compare_shared_node_metadata(self):
        """Detect metadata changes between versions."""
        rpg1 = {
            "root": {
                "id": "n1",
                "meta": {"description": "old"},
                "children": [
                    {"id": "n2", "meta": {"description": "same"}, "children": []},
                ],
            },
            "edges": [],
        }
        rpg2 = {
            "root": {
                "id": "n1",
                "meta": {"description": "new"},
                "children": [
                    {"id": "n2", "meta": {"description": "same"}, "children": []},
                ],
            },
            "edges": [],
        }
        changed = _compare_shared_node_metadata(rpg1, rpg2)
        assert "n1" in changed
        assert "n2" not in changed


# ============================================================================
# Tests: WorkflowIntegration
# ============================================================================


class TestWorkflowIntegration:
    """Tests for WorkflowIntegration."""

    def test_prepare_for_codegen_basic(self, simple_rpg):
        """prepare_for_codegen returns expected keys."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
        )
        assert "rpg_dict" in context
        assert "repo_name" in context
        assert context["repo_name"] == "test_repo"
        assert "functional_areas" in context
        assert "existing_interfaces" in context
        assert "dependency_edges" in context

    def test_prepare_for_codegen_with_targets(self, simple_rpg):
        """prepare_for_codegen resolves target nodes."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["class_foo"],
        )
        assert len(context["target_context"]) == 1
        assert context["target_context"][0]["id"] == "class_foo"
        assert context["target_context"][0]["name"] == "Foo"

    def test_prepare_for_codegen_target_by_name(self, simple_rpg):
        """Targets can be resolved by node name."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["Foo"],
        )
        assert len(context["target_context"]) == 1
        assert context["target_context"][0]["name"] == "Foo"

    def test_prepare_for_codegen_target_by_path(self, simple_rpg):
        """Targets can be resolved by meta.path."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["src/foo.py"],
        )
        assert len(context["target_context"]) == 1
        assert context["target_context"][0]["name"] == "foo.py"

    def test_prepare_for_codegen_missing_target(self, simple_rpg):
        """Missing target nodes are skipped with a warning."""
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["nonexistent_node"],
        )
        assert len(context["target_context"]) == 0

    def test_prepare_dependency_edges_filtered(self, simple_rpg):
        """Dependency edges are filtered to target nodes."""
        # With target: only edges involving target node or its descendants
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["class_foo"],
        )
        # The INVOKES edge between class_foo and func_bar should be included
        edge_srcs = [e.get("src") for e in context["dependency_edges"]]
        assert "class_foo" in edge_srcs

    def test_merge_generated_code_new_file(self, simple_rpg):
        """merge_generated_code adds new file and code units to RPG."""
        code = (
            "class NewClass:\n"
            "    def method_one(self):\n"
            "        pass\n"
            "\n"
            "def standalone_func():\n"
            "    return 42\n"
        )
        updated = WorkflowIntegration.merge_generated_code(
            rpg=simple_rpg,
            generated_files={"src/new_module.py": code},
        )
        # The RPG should now have the file node
        file_node = updated.find_node_by_path("src/new_module.py")
        assert file_node is not None, "New file node should exist"

        # Check that code units were added
        class_node = updated.find_node_by_path("src/new_module.py::NewClass")
        assert class_node is not None, "Class node should exist"

        func_node = updated.find_node_by_path("src/new_module.py::standalone_func")
        assert func_node is not None, "Function node should exist"

    def test_merge_generated_code_empty(self, simple_rpg):
        """merge_generated_code with empty files does nothing."""
        node_count_before = len(simple_rpg.nodes)
        updated = WorkflowIntegration.merge_generated_code(
            rpg=simple_rpg,
            generated_files={},
        )
        assert len(updated.nodes) == node_count_before

    def test_merge_generated_code_invalid_syntax(self, simple_rpg):
        """merge_generated_code handles unparseable files gracefully."""
        node_count_before = len(simple_rpg.nodes)
        updated = WorkflowIntegration.merge_generated_code(
            rpg=simple_rpg,
            generated_files={"src/bad.py": "def broken(:\n"},
        )
        # Should not crash; may or may not add the file depending on parser
        assert isinstance(updated, RPG)

    def test_save_rpg(self, simple_rpg, tmp_cmind_dir):
        """save_rpg writes to disk and creates a version."""
        _, cmind_dir = tmp_cmind_dir
        result = WorkflowIntegration.save_rpg(
            rpg=simple_rpg,
            cmind_dir=cmind_dir,
            message="Test save",
            source="encoded",
        )
        assert os.path.isfile(result["rpg_path"])
        assert "version" in result

    def test_save_rpg_without_versioning(self, simple_rpg, tmp_cmind_dir):
        """save_rpg with version_control=False skips versioning."""
        _, cmind_dir = tmp_cmind_dir
        result = WorkflowIntegration.save_rpg(
            rpg=simple_rpg,
            cmind_dir=cmind_dir,
            message="No version",
            version_control=False,
        )
        assert os.path.isfile(result["rpg_path"])
        assert "version" not in result

    def test_load_rpg(self, simple_rpg, tmp_cmind_dir):
        """load_rpg reads RPG from saved file."""
        _, cmind_dir = tmp_cmind_dir
        WorkflowIntegration.save_rpg(
            rpg=simple_rpg,
            cmind_dir=cmind_dir,
            version_control=False,
        )
        loaded = WorkflowIntegration.load_rpg(cmind_dir)
        assert loaded is not None
        assert loaded.repo_name == "test_repo"

    def test_load_rpg_nonexistent(self, tmp_cmind_dir):
        """load_rpg returns None when file doesn't exist."""
        _, cmind_dir = tmp_cmind_dir
        loaded = WorkflowIntegration.load_rpg(cmind_dir)
        assert loaded is None

    def test_detect_workflow_mode_no_rpg(self):
        """No RPG -> forward mode."""
        assert WorkflowIntegration.detect_workflow_mode(rpg=None) == "forward"

    def test_detect_workflow_mode_reverse(self, simple_rpg):
        """RPG exists, no feature_spec -> reverse mode."""
        assert WorkflowIntegration.detect_workflow_mode(
            rpg=simple_rpg, has_feature_spec=False
        ) == "reverse"

    def test_detect_workflow_mode_mixed(self, simple_rpg):
        """RPG exists, feature_spec exists -> mixed mode."""
        assert WorkflowIntegration.detect_workflow_mode(
            rpg=simple_rpg, has_feature_spec=True
        ) == "mixed"


# ============================================================================
# Tests: Internal helpers
# ============================================================================


class TestInternalHelpers:
    """Tests for workflow module internal helpers."""

    def test_resolve_node_by_id(self, simple_rpg):
        node = _resolve_node(simple_rpg, "class_foo")
        assert node is not None
        assert node.name == "Foo"

    def test_resolve_node_by_path(self, simple_rpg):
        node = _resolve_node(simple_rpg, "src/foo.py")
        assert node is not None
        assert node.name == "foo.py"

    def test_resolve_node_by_name(self, simple_rpg):
        node = _resolve_node(simple_rpg, "Foo")
        assert node is not None
        assert node.id == "class_foo"

    def test_resolve_node_case_insensitive(self, simple_rpg):
        node = _resolve_node(simple_rpg, "areaone")
        # Should match "AreaOne" via case-insensitive name lookup
        assert node is not None
        assert node.name == "AreaOne"

    def test_resolve_node_not_found(self, simple_rpg):
        node = _resolve_node(simple_rpg, "definitely_does_not_exist_xyz")
        assert node is None

    def test_build_node_context(self, simple_rpg):
        node = simple_rpg.get_node_by_id("file_a")
        ctx = _build_node_context(simple_rpg, node)
        assert ctx["id"] == "file_a"
        assert ctx["name"] == "foo.py"
        assert len(ctx["children"]) == 2  # class_foo and func_bar
        assert ctx["path"] == "src/foo.py"

    def test_gather_existing_interfaces(self, simple_rpg):
        interfaces = _gather_existing_interfaces(simple_rpg)
        assert "src/foo.py" in interfaces
        names = [e["name"] for e in interfaces["src/foo.py"]]
        assert "Foo" in names
        assert "bar" in names

    def test_gather_dependency_edges_all(self, simple_rpg):
        """No target IDs -> return all edges."""
        edges = _gather_dependency_edges(simple_rpg, set())
        assert len(edges) >= 1  # At least the INVOKES edge

    def test_gather_dependency_edges_filtered(self, simple_rpg):
        """Filter edges to specific target IDs."""
        edges = _gather_dependency_edges(simple_rpg, {"class_foo"})
        # Should include the INVOKES edge from class_foo -> func_bar
        assert len(edges) >= 1
        srcs = [e.get("src") for e in edges]
        assert "class_foo" in srcs

    def test_infer_rpg_source_encoded(self, simple_rpg):
        """RPG with rpg_encoder generator -> 'encoded'."""
        source = _infer_rpg_source(simple_rpg)
        assert source == "encoded"

    def test_infer_rpg_source_generated(self):
        """RPG with code_gen generator -> 'generated'."""
        rpg = RPG(repo_name="test")
        node = Node(
            id="n1",
            name="A",
            level=None,
            meta=NodeMetaData(generator="code_gen"),
        )
        rpg.add_node(node)
        rpg.add_edge(rpg.repo_node, node, EdgeType.CONTAINS)
        assert _infer_rpg_source(rpg) == "generated"

    def test_infer_rpg_source_mixed(self):
        """RPG with both generators -> 'mixed'."""
        rpg = RPG(repo_name="test")
        n1 = Node(id="n1", name="A", level=None, meta=NodeMetaData(generator="code_gen"))
        n2 = Node(id="n2", name="B", level=None, meta=NodeMetaData(generator="rpg_encoder"))
        rpg.add_node(n1)
        rpg.add_node(n2)
        rpg.add_edge(rpg.repo_node, n1, EdgeType.CONTAINS)
        rpg.add_edge(rpg.repo_node, n2, EdgeType.CONTAINS)
        assert _infer_rpg_source(rpg) == "mixed"

    def test_infer_rpg_source_no_generators(self):
        """RPG with no generator metadata -> 'generated' (default)."""
        rpg = RPG(repo_name="test")
        assert _infer_rpg_source(rpg) == "generated"


# ============================================================================
# Tests: End-to-end workflow scenarios
# ============================================================================


class TestWorkflowScenarios:
    """Integration tests for the four workflow scenarios."""

    def test_pure_reverse_scenario(self, simple_rpg, tmp_cmind_dir):
        """Pure reverse: encode -> save -> load -> explore."""
        _, cmind_dir = tmp_cmind_dir

        # Save the encoded RPG
        result = WorkflowIntegration.save_rpg(
            rpg=simple_rpg,
            cmind_dir=cmind_dir,
            message="Encoded from repo",
            source="encoded",
        )
        assert result["version"] == 1

        # Load it back
        loaded = WorkflowIntegration.load_rpg(cmind_dir)
        assert loaded is not None
        assert loaded.repo_name == "test_repo"

        # Prepare context (simulate explore)
        context = WorkflowIntegration.prepare_for_codegen(rpg=loaded)
        assert context["source"] == "encoded"

    def test_mixed_enhance_scenario(self, simple_rpg, tmp_cmind_dir):
        """Mixed: encode -> save -> merge new code -> save."""
        _, cmind_dir = tmp_cmind_dir

        # Step 1: Save encoded RPG
        WorkflowIntegration.save_rpg(
            rpg=simple_rpg,
            cmind_dir=cmind_dir,
            message="Initial encode",
            source="encoded",
        )

        # Step 2: Prepare context for code generation
        context = WorkflowIntegration.prepare_for_codegen(
            rpg=simple_rpg,
            target_nodes=["AreaOne"],
        )
        assert len(context["target_context"]) == 1

        # Step 3: Merge generated code
        new_code = "def helper():\n    return 'hello'\n"
        updated = WorkflowIntegration.merge_generated_code(
            rpg=simple_rpg,
            generated_files={"src/helper.py": new_code},
        )

        # Step 4: Save updated RPG
        result = WorkflowIntegration.save_rpg(
            rpg=updated,
            cmind_dir=cmind_dir,
            message="Added helper module",
            source="mixed",
        )
        assert result["version"] == 2

        # Verify version history
        vc = RPGVersionControl(cmind_dir=cmind_dir)
        versions = vc.list_versions()
        assert len(versions) == 2

    def test_iterative_scenario(self, simple_rpg, tmp_cmind_dir):
        """Iterative: merge -> save -> merge -> save -> diff."""
        _, cmind_dir = tmp_cmind_dir
        vc = RPGVersionControl(cmind_dir=cmind_dir)

        # Iteration 1
        vc.save_version(simple_rpg, message="Before iteration 1")
        code1 = "class Alpha:\n    pass\n"
        updated = WorkflowIntegration.merge_generated_code(
            rpg=simple_rpg,
            generated_files={"src/alpha.py": code1},
        )
        vc.save_version(updated, message="After iteration 1")

        # Iteration 2
        code2 = "class Beta:\n    pass\n"
        updated = WorkflowIntegration.merge_generated_code(
            rpg=updated,
            generated_files={"src/beta.py": code2},
        )
        vc.save_version(updated, message="After iteration 2")

        # Diff between versions
        diff = vc.diff(1, 3)
        assert diff["summary"]["nodes_added"] > 0
