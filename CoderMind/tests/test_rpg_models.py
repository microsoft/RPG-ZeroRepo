"""Unit tests for M1 — Unified Data Model (rpg_models.py).

Tests cover:
- EdgeType extension (COMPOSES, IMPORTS, is_hierarchy)
- RPG new attributes (dep_graph, _dep_to_rpg_map)
- RPG new query methods (get_node_by_id, get_nodes_by_type, etc.)
- CoderMind nested format serialization round-trip
- ZeroRepo flat format loading
- Backward compatibility (existing to_dict/from_dict unchanged)
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rpg.models import (
    RPG,
    Node,
    Edge,
    EdgeType,
    NodeMetaData,
    NodeType,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rpg():
    """Build a small RPG with two functional areas and one feature node."""
    rpg = RPG("myapp", repo_info="Test application")

    auth = Node(
        id="auth_fa_001",
        name="Authentication",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src/auth"),
    )
    rpg.add_node(auth)
    rpg.add_edge(rpg.repo_node.id, auth.id, EdgeType.CONTAINS)

    api = Node(
        id="api_fa_002",
        name="API",
        meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="src/api"),
    )
    rpg.add_node(api)
    rpg.add_edge(rpg.repo_node.id, api.id, EdgeType.CONTAINS)

    login = Node(
        id="login_feat_003",
        name="Login",
        meta=NodeMetaData(type_name=NodeType.FILE, path="src/auth/login.py"),
    )
    rpg.add_node(login)
    rpg.add_edge(auth.id, login.id, EdgeType.CONTAINS)

    validate = Node(
        id="validate_feat_004",
        name="validate_credentials",
        meta=NodeMetaData(
            type_name=NodeType.FUNCTION,
            path="src/auth/login.py::validate_credentials",
        ),
    )
    rpg.add_node(validate)
    rpg.add_edge(login.id, validate.id, EdgeType.CONTAINS)

    # Non-containment edge
    rpg.add_edge(login.id, api.id, EdgeType.INVOKES)

    return rpg


@pytest.fixture
def zerorepo_flat_data():
    """Sample ZeroRepo flat-format JSON dict."""
    return {
        "repo_name": "flat_repo",
        "repo_info": "A flat-format RPG",
        "excluded_files": ["test_*.py"],
        "repo_node_id": "flat_repo_L0",
        "nodes": [
            {
                "id": "flat_repo_L0",
                "name": "flat_repo",
                "node_type": "repo",
                "level": 0,
                "meta": {
                    "type_name": "directory",
                    "path": ".",
                    "description": "",
                    "content": "",
                },
            },
            {
                "id": "comp_A_001",
                "name": "ComponentA",
                "node_type": "functional_area",
                "level": 1,
                "meta": {
                    "type_name": "directory",
                    "path": "src/comp_a",
                    "description": "Component A",
                    "content": "",
                },
            },
            {
                "id": "feat_B_002",
                "name": "FeatureB",
                "node_type": "feature",
                "level": 5,
                "meta": {
                    "type_name": "function",
                    "path": "src/comp_a/feature.py:do_stuff",
                    "description": "Does stuff",
                    "content": "",
                },
            },
        ],
        "edges": [
            {
                "src": "flat_repo_L0",
                "dst": "comp_A_001",
                "relation": "composes",
                "meta": None,
            },
            {
                "src": "comp_A_001",
                "dst": "feat_B_002",
                "relation": "composes",
                "meta": None,
            },
        ],
        "_dep_to_rpg_map": {"dep_x": ["comp_A_001"]},
    }


# ──────────────────────────────────────────────────────────────
# EdgeType tests
# ──────────────────────────────────────────────────────────────

class TestEdgeType:
    """Tests for EdgeType enum extension."""

    def test_composes_exists(self):
        assert EdgeType.COMPOSES.value == "composes"

    def test_imports_exists(self):
        assert EdgeType.IMPORTS.value == "imports"

    def test_existing_types_preserved(self):
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.INHERITS.value == "inherits"
        assert EdgeType.INVOKES.value == "invokes"
        assert EdgeType.REFERENCES.value == "references"
        assert EdgeType.SAME_UNIT.value == "same_unit"
        assert EdgeType.CONTAINS_BASE_CLASS.value == "contains_base_class"

    def test_is_hierarchy_true(self):
        assert EdgeType.is_hierarchy(EdgeType.COMPOSES) is True
        assert EdgeType.is_hierarchy(EdgeType.CONTAINS) is True
        assert EdgeType.is_hierarchy(EdgeType.CONTAINS_BASE_CLASS) is True
        assert EdgeType.is_hierarchy("composes") is True
        assert EdgeType.is_hierarchy("contains") is True
        assert EdgeType.is_hierarchy("CONTAINS") is True

    def test_is_hierarchy_false(self):
        assert EdgeType.is_hierarchy(EdgeType.INVOKES) is False
        assert EdgeType.is_hierarchy(EdgeType.IMPORTS) is False
        assert EdgeType.is_hierarchy(EdgeType.INHERITS) is False
        assert EdgeType.is_hierarchy(EdgeType.REFERENCES) is False
        assert EdgeType.is_hierarchy(EdgeType.SAME_UNIT) is False
        assert EdgeType.is_hierarchy("invokes") is False

    def test_composes_used_as_hierarchy_edge(self):
        """COMPOSES edges should be treated like CONTAINS (build tree)."""
        rpg = RPG("test")
        child = Node(id="child_001", name="child")
        rpg.add_node(child)
        rpg.add_edge(rpg.repo_node.id, child.id, EdgeType.COMPOSES)

        assert child in rpg.repo_node._children
        assert child._parent is rpg.repo_node


# ──────────────────────────────────────────────────────────────
# RPG new attributes
# ──────────────────────────────────────────────────────────────

class TestRPGAttributes:
    """Tests for RPG.dep_graph and RPG._dep_to_rpg_map."""

    def test_dep_graph_default_none(self):
        rpg = RPG("test")
        assert rpg.dep_graph is None

    def test_dep_to_rpg_map_default_empty(self):
        rpg = RPG("test")
        assert rpg._dep_to_rpg_map == {}


# ──────────────────────────────────────────────────────────────
# RPG query methods
# ──────────────────────────────────────────────────────────────

class TestGetNodeById:
    def test_found(self, sample_rpg):
        node = sample_rpg.get_node_by_id("auth_fa_001")
        assert node is not None
        assert node.name == "Authentication"

    def test_not_found(self, sample_rpg):
        assert sample_rpg.get_node_by_id("nonexistent") is None

    def test_repo_node(self, sample_rpg):
        node = sample_rpg.get_node_by_id(sample_rpg.repo_node.id)
        assert node is sample_rpg.repo_node


class TestGetNodesByType:
    def test_directory(self, sample_rpg):
        dirs = sample_rpg.get_nodes_by_type(NodeType.DIRECTORY)
        names = {n.name for n in dirs}
        assert "Authentication" in names
        assert "API" in names

    def test_file(self, sample_rpg):
        files = sample_rpg.get_nodes_by_type(NodeType.FILE)
        assert len(files) == 1
        assert files[0].name == "Login"

    def test_string_arg(self, sample_rpg):
        funcs = sample_rpg.get_nodes_by_type("function")
        assert len(funcs) == 1
        assert funcs[0].name == "validate_credentials"

    def test_no_match(self, sample_rpg):
        assert sample_rpg.get_nodes_by_type(NodeType.CLASS) == []


class TestGetNodeByFeaturePath:
    def test_found(self, sample_rpg):
        node = sample_rpg.get_node_by_feature_path("Authentication/Login")
        assert node is not None
        assert node.id == "login_feat_003"

    def test_deep_path(self, sample_rpg):
        node = sample_rpg.get_node_by_feature_path(
            "Authentication/Login/validate_credentials"
        )
        assert node is not None
        assert node.id == "validate_feat_004"

    def test_not_found(self, sample_rpg):
        assert sample_rpg.get_node_by_feature_path("Nonexistent/Path") is None

    def test_empty_path(self, sample_rpg):
        assert sample_rpg.get_node_by_feature_path("") is None

    def test_strip_separators(self, sample_rpg):
        node = sample_rpg.get_node_by_feature_path("/Authentication/Login/")
        assert node is not None
        assert node.id == "login_feat_003"


class TestGetFunctionalAreas:
    def test_basic(self, sample_rpg):
        areas = sample_rpg.get_functional_areas()
        assert areas == ["API", "Authentication"]

    def test_empty_rpg(self):
        rpg = RPG("empty")
        assert rpg.get_functional_areas() == []


class TestVisualizeDirMap:
    def test_text_format(self, sample_rpg):
        text = sample_rpg.visualize_dir_map(max_depth=4)
        assert "Authentication" in text
        assert "Login" in text
        assert "API" in text

    def test_json_format(self, sample_rpg):
        out = sample_rpg.visualize_dir_map(json_format=True)
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2  # Two L1 nodes
        names = {item["name"] for item in parsed}
        assert "Authentication" in names
        assert "API" in names

    def test_max_depth_limits(self, sample_rpg):
        # depth=1 should show L1 only, not children
        text = sample_rpg.visualize_dir_map(max_depth=1)
        assert "Authentication" in text
        assert "Login" not in text

    def test_empty_rpg(self):
        rpg = RPG("empty")
        assert rpg.visualize_dir_map() == ""

    def test_tree_markers(self, sample_rpg):
        text = sample_rpg.visualize_dir_map(
            max_depth=4, use_tree_markers=True
        )
        # Should contain tree markers for children
        assert "├─" in text or "└─" in text

    def test_indent_only(self, sample_rpg):
        text = sample_rpg.visualize_dir_map(
            max_depth=4, use_tree_markers=False
        )
        assert "├─" not in text
        assert "└─" not in text

    def test_start_from_specific_node(self, sample_rpg):
        text = sample_rpg.visualize_dir_map(start="auth_fa_001", max_depth=4)
        assert "Login" in text
        assert "API" not in text

    def test_feature_only_false(self, sample_rpg):
        text = sample_rpg.visualize_dir_map(
            max_depth=4, feature_only=False
        )
        # Should include path info in brackets
        assert "[" in text


# ──────────────────────────────────────────────────────────────
# Serialization: CoderMind nested format round-trip
# ──────────────────────────────────────────────────────────────

class TestNestedFormatRoundTrip:
    def test_structure_preserved(self, sample_rpg):
        d = sample_rpg.to_dict()
        rpg2 = RPG.from_dict(d)

        assert rpg2.repo_name == "myapp"
        assert rpg2.repo_info == "Test application"
        assert rpg2.get_node_by_id("auth_fa_001") is not None
        assert rpg2.get_node_by_id("login_feat_003") is not None
        assert rpg2.get_node_by_id("validate_feat_004") is not None

    def test_tree_structure(self, sample_rpg):
        d = sample_rpg.to_dict()
        rpg2 = RPG.from_dict(d)

        # Check parent-child relationships
        auth = rpg2.get_node_by_id("auth_fa_001")
        assert auth._parent is rpg2.repo_node
        assert len(auth._children) == 1
        assert auth._children[0].id == "login_feat_003"

    def test_edges_preserved(self, sample_rpg):
        d = sample_rpg.to_dict()
        rpg2 = RPG.from_dict(d)

        # Only non-containment edges stored
        assert len(rpg2.edges) == 1
        assert rpg2.edges[0].relation == EdgeType.INVOKES

    def test_file_round_trip(self, sample_rpg):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            tmppath = f.name
        try:
            sample_rpg.save_json(tmppath)
            rpg2 = RPG.load_json(tmppath)
            assert rpg2.repo_name == "myapp"
            assert rpg2.get_node_by_id("auth_fa_001") is not None
        finally:
            os.unlink(tmppath)

    def test_has_root_field(self, sample_rpg):
        d = sample_rpg.to_dict()
        assert "root" in d
        assert "nodes" not in d  # CoderMind format has root, not nodes
        assert "_dep_to_rpg_map" in d


# ──────────────────────────────────────────────────────────────
# dep_graph serialization
# ──────────────────────────────────────────────────────────────

class TestDepGraphSerialization:
    def test_dep_graph_serialized(self, sample_rpg):
        """dep_graph should appear in to_dict output when set."""
        from rpg.dep_graph import DependencyGraph
        sample_rpg.dep_graph = DependencyGraph("/tmp/fake")
        d = sample_rpg.to_dict()
        assert "dep_graph" in d

    def test_dep_graph_not_serialized_when_none(self, sample_rpg):
        """dep_graph should not appear when None."""
        d = sample_rpg.to_dict()
        assert "dep_graph" not in d

    def test_dep_graph_roundtrip(self, sample_rpg):
        """dep_graph should survive to_dict -> from_dict round-trip."""
        from rpg.dep_graph import DependencyGraph
        sample_rpg.dep_graph = DependencyGraph("/tmp/fake")
        d = sample_rpg.to_dict()
        rpg2 = RPG.from_dict(d)
        assert rpg2.dep_graph is not None

    def test_dep_to_rpg_map_roundtrip(self, sample_rpg):
        """_dep_to_rpg_map should survive round-trip."""
        sample_rpg._dep_to_rpg_map = {"some_dep": ["auth_fa_001"]}
        d = sample_rpg.to_dict()
        rpg2 = RPG.from_dict(d)
        assert rpg2._dep_to_rpg_map == {"some_dep": ["auth_fa_001"]}


# ──────────────────────────────────────────────────────────────
# type_name protection
# ──────────────────────────────────────────────────────────────

class TestTypeNameProtection:
    def test_type_name_not_overwritten(self):
        """add_edge should not overwrite an explicitly set type_name."""
        rpg = RPG("test")
        cls_node = Node(
            id="cls_001",
            name="MyClass",
            meta=NodeMetaData(type_name=NodeType.CLASS, path="src/mod.py:MyClass"),
        )
        rpg.add_node(cls_node)
        rpg.add_edge(rpg.repo_node.id, cls_node.id, EdgeType.CONTAINS)
        assert cls_node.meta.type_name == NodeType.CLASS

    def test_type_name_inferred_when_none(self):
        """add_edge should infer type_name when not set."""
        rpg = RPG("test")
        node = Node(
            id="fn_001",
            name="do_stuff",
            meta=NodeMetaData(path="src/mod.py:do_stuff"),
        )
        rpg.add_node(node)
        rpg.add_edge(rpg.repo_node.id, node.id, EdgeType.CONTAINS)
        assert node.meta.type_name == NodeType.FUNCTION

    def test_single_colon_separator_class(self):
        """Single : separator with uppercase name should infer class."""
        from rpg.models import infer_type_name_from_path
        assert infer_type_name_from_path("src/mod.py:MyClass") == "class"

    def test_single_colon_separator_method(self):
        """Single : with 3 parts should infer method."""
        from rpg.models import infer_type_name_from_path
        assert infer_type_name_from_path("src/mod.py:MyClass:method") == "method"


# ──────────────────────────────────────────────────────────────
# Serialization: ZeroRepo flat format loading
# ──────────────────────────────────────────────────────────────

class TestFlatFormatLoading:
    def test_basic_loading(self, zerorepo_flat_data):
        rpg = RPG.from_dict(zerorepo_flat_data)
        assert rpg.repo_name == "flat_repo"
        assert rpg.repo_info == "A flat-format RPG"
        assert rpg.excluded_files == ["test_*.py"]

    def test_repo_node(self, zerorepo_flat_data):
        rpg = RPG.from_dict(zerorepo_flat_data)
        assert rpg.repo_node.id == "flat_repo_L0"
        assert rpg.repo_node.level == 0
        assert rpg.repo_node.node_type == "repo"

    def test_nodes_loaded(self, zerorepo_flat_data):
        rpg = RPG.from_dict(zerorepo_flat_data)
        assert rpg.get_node_by_id("comp_A_001") is not None
        assert rpg.get_node_by_id("feat_B_002") is not None

    def test_tree_from_edges(self, zerorepo_flat_data):
        rpg = RPG.from_dict(zerorepo_flat_data)

        comp_a = rpg.get_node_by_id("comp_A_001")
        assert comp_a._parent is rpg.repo_node
        assert len(comp_a._children) == 1
        assert comp_a._children[0].id == "feat_B_002"

    def test_dep_to_rpg_map_loaded(self, zerorepo_flat_data):
        rpg = RPG.from_dict(zerorepo_flat_data)
        assert rpg._dep_to_rpg_map == {"dep_x": ["comp_A_001"]}

    def test_format_detection_flat(self, zerorepo_flat_data):
        """Presence of 'nodes' key triggers flat format path."""
        rpg = RPG.from_dict(zerorepo_flat_data)
        assert rpg.repo_name == "flat_repo"

    def test_format_detection_nested(self, sample_rpg):
        """Absence of 'nodes' key triggers nested format path."""
        d = sample_rpg.to_dict()
        assert "nodes" not in d
        rpg2 = RPG.from_dict(d)
        assert rpg2.repo_name == "myapp"

    def test_flat_with_non_hierarchy_edges(self):
        """Non-hierarchy edges in flat format should be stored, not tree-ified."""
        data = {
            "repo_name": "test",
            "repo_node_id": "test_L0",
            "nodes": [
                {"id": "test_L0", "name": "test", "node_type": "repo", "level": 0,
                 "meta": {"type_name": "directory", "path": "."}},
                {"id": "a_001", "name": "A", "level": 1,
                 "meta": {"type_name": "directory", "path": "a"}},
                {"id": "b_002", "name": "B", "level": 1,
                 "meta": {"type_name": "directory", "path": "b"}},
            ],
            "edges": [
                {"src": "test_L0", "dst": "a_001", "relation": "composes"},
                {"src": "test_L0", "dst": "b_002", "relation": "composes"},
                {"src": "a_001", "dst": "b_002", "relation": "invokes"},
            ],
        }
        rpg = RPG.from_dict(data)
        assert len(rpg.repo_node._children) == 2
        # INVOKES should be in edges list, not tree
        assert len(rpg.edges) == 1
        assert rpg.edges[0].relation == EdgeType.INVOKES


# ──────────────────────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_existing_methods_still_work(self, sample_rpg):
        """Existing RPG methods should not be broken."""
        # get_children
        children = sample_rpg.get_children(sample_rpg.repo_node.id)
        assert len(children) == 2

        # get_path_to_root
        path = sample_rpg.get_path_to_root("validate_feat_004")
        assert path[0] == sample_rpg.repo_node.id
        assert path[-1] == "validate_feat_004"

        # find_node_by_path
        node = sample_rpg.find_node_by_path("src/auth")
        assert node is not None
        assert node.id == "auth_fa_001"

        # find_child_by_name
        child = sample_rpg.find_child_by_name(sample_rpg.repo_node.id, "Authentication")
        assert child is not None
        assert child.id == "auth_fa_001"

    def test_nodes_property(self, sample_rpg):
        """rpg.nodes should still return dict of all nodes."""
        nodes = sample_rpg.nodes
        assert isinstance(nodes, dict)
        assert "auth_fa_001" in nodes
        assert sample_rpg.repo_node.id in nodes
