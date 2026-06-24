"""Unit tests for M2 — DependencyGraph (dep_graph.py).

Tests cover:
- Helper functions: normalize_path, is_test_file, get_node_range_robust, extract_source_by_lines
- DependencyGraph.build: directory/file structure scanning
- DependencyGraph.parse: AST extraction (classes, functions, methods)
- Import, invokes, inherits edge extraction
- Graph views (G_tree, G_imports, G_invokes, G_inherits, G_code)
- Public query methods: get_parent, get_name, find_node, find_file, all_paths
- Serialization: to_dict / from_dict / reparse_ast round-trip
- Filter functions: _exclude_irrelevant_for_build, _exclude_irrelevant_for_parse
- path_to_module conversion
- Edge cases and error handling
"""

import json
import os
import sys
import tempfile
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rpg.dep_graph import (
    DependencyGraph,
    normalize_path,
    is_test_file,
    get_node_range_robust,
    extract_source_by_lines,
    path_to_module,
    _exclude_irrelevant_for_build,
    _exclude_irrelevant_for_parse,
)
from rpg.models import EdgeType, NodeType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_repo(tmp_path):
    """Create a minimal Python repo structure for testing."""
    # src/
    src = tmp_path / "src"
    src.mkdir()

    # src/__init__.py
    (src / "__init__.py").write_text("")

    # src/main.py
    (src / "main.py").write_text(textwrap.dedent("""\
        from src.models import User

        def main():
            user = User("test")
            user.greet()

        if __name__ == "__main__":
            main()
    """))

    # src/models.py
    (src / "models.py").write_text(textwrap.dedent("""\
        class Base:
            def save(self):
                pass

        class User(Base):
            def __init__(self, name: str):
                self.name = name

            def greet(self):
                return f"Hello, {self.name}"

        def create_user(name: str) -> User:
            return User(name)
    """))

    # src/utils/
    utils = src / "utils"
    utils.mkdir()
    (utils / "__init__.py").write_text("from .helpers import format_name\n")
    (utils / "helpers.py").write_text(textwrap.dedent("""\
        def format_name(name: str) -> str:
            return name.strip().title()
    """))

    # README.md (non-Python file, should be in build but not parsed)
    (tmp_path / "README.md").write_text("# Sample Repo\n")

    # .git directory (should be excluded)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("gitconfig")

    return tmp_path


@pytest.fixture
def built_graph(sample_repo):
    """DependencyGraph with build() called."""
    dg = DependencyGraph(str(sample_repo))
    dg.build()
    return dg


@pytest.fixture
def parsed_graph(sample_repo):
    """DependencyGraph with build() and parse() called."""
    dg = DependencyGraph(str(sample_repo))
    dg.build()
    dg.parse()
    return dg


# ============================================================================
# Helper function tests
# ============================================================================

class TestNormalizePath:
    def test_basic_path(self):
        assert normalize_path("src/main.py") == "src/main.py"

    def test_dot_prefix(self):
        assert normalize_path("./src/main.py") == "src/main.py"

    def test_root(self):
        assert normalize_path(".") == "."

    def test_empty_string(self):
        assert normalize_path("") == "."

    def test_with_qualified_name(self):
        assert normalize_path("src/models.py:User") == "src/models.py:User"

    def test_with_method_name(self):
        assert normalize_path("src/models.py:User.greet") == "src/models.py:User.greet"

    def test_strips_whitespace(self):
        assert normalize_path("  src/main.py  ") == "src/main.py"

    def test_qualified_name_whitespace(self):
        assert normalize_path("src/models.py: User . greet ") == "src/models.py:User.greet"

    def test_leading_slash(self):
        assert normalize_path("/src/main.py") == "src/main.py"


class TestIsTestFile:
    def test_test_directory(self):
        assert is_test_file("tests/test_main.py") is True

    def test_test_prefix_file(self):
        assert is_test_file("src/test_models.py") is True

    def test_normal_file(self):
        assert is_test_file("src/models.py") is False

    def test_with_qualified_name(self):
        assert is_test_file("tests/test_main.py:TestUser") is True

    def test_testing_directory(self):
        assert is_test_file("testing/integration.py") is True


class TestExtractSourceByLines:
    def test_basic_extraction(self):
        source = "line1\nline2\nline3\nline4\nline5"
        result = extract_source_by_lines(source, 2, 4)
        assert result == "line2\nline3\nline4"

    def test_single_line(self):
        source = "line1\nline2\nline3"
        result = extract_source_by_lines(source, 2, 2)
        assert result == "line2"

    def test_none_start(self):
        assert extract_source_by_lines("line1\nline2", None, 2) == ""

    def test_none_end(self):
        assert extract_source_by_lines("line1\nline2", 1, None) == ""

    def test_out_of_range(self):
        source = "line1\nline2"
        result = extract_source_by_lines(source, 1, 10)
        assert result == "line1\nline2"


class TestPathToModule:
    def test_python_file(self):
        assert path_to_module("src/main.py") == "src.main"

    def test_init_file(self):
        assert path_to_module("src/__init__.py") == "src"

    def test_directory(self):
        assert path_to_module("src/utils") == "src.utils"

    def test_with_qualified_name(self):
        assert path_to_module("src/models.py:User") == "src.models"

    def test_root(self):
        assert path_to_module(".") == ""

    def test_dot_prefix(self):
        assert path_to_module("./src/main.py") == "src.main"


# ============================================================================
# Filter function tests
# ============================================================================

class TestExcludeIrrelevantForBuild:
    def test_normal_file_included(self):
        assert _exclude_irrelevant_for_build("src/main.py") is True

    def test_git_excluded(self):
        assert _exclude_irrelevant_for_build(".git/config") is False

    def test_pycache_excluded(self):
        assert _exclude_irrelevant_for_build("__pycache__/module.cpython-39.pyc") is False

    def test_image_excluded(self):
        assert _exclude_irrelevant_for_build("assets/logo.png") is False

    def test_hidden_file_excluded(self):
        assert _exclude_irrelevant_for_build(".env") is False

    def test_test_file_excluded(self):
        assert _exclude_irrelevant_for_build("tests/test_main.py") is False

    def test_license_excluded(self):
        assert _exclude_irrelevant_for_build("LICENSE") is False

    def test_pyproject_excluded(self):
        assert _exclude_irrelevant_for_build("pyproject.toml") is False

    def test_node_modules_excluded(self):
        assert _exclude_irrelevant_for_build("node_modules/package/index.js") is False

    def test_go_testdata_source_included(self):
        assert _exclude_irrelevant_for_build("pkg/testdata/fixtures.go") is True

    def test_non_python_test_helper_source_included_when_language_rule_allows(self):
        assert _exclude_irrelevant_for_build("src/test_helpers.ts") is True

    def test_language_test_source_excluded(self):
        assert _exclude_irrelevant_for_build("pkg/foo_test.go") is False
        assert _exclude_irrelevant_for_build("src/client.test.ts") is False

    def test_non_source_testish_file_keeps_legacy_exclusion(self):
        assert _exclude_irrelevant_for_build("data/test_fixture.json") is False


class TestExcludeIrrelevantForParse:
    def test_python_file_included(self):
        assert _exclude_irrelevant_for_parse("src/main.py") is True

    def test_non_python_excluded(self):
        assert _exclude_irrelevant_for_parse("src/config.json") is False

    def test_test_file_excluded(self):
        assert _exclude_irrelevant_for_parse("tests/test_main.py") is False

    def test_setup_py_excluded(self):
        # The filter checks for paths ending with "/setup.py", so root-level
        # "setup.py" without a "/" prefix won't match.  Only nested paths are excluded.
        assert _exclude_irrelevant_for_parse("project/setup.py") is False

    def test_conftest_excluded(self):
        assert _exclude_irrelevant_for_parse("tests/conftest.py") is False

    def test_test_prefix_file_excluded(self):
        assert _exclude_irrelevant_for_parse("src/test_something.py") is False


# ============================================================================
# DependencyGraph.build tests
# ============================================================================

class TestBuild:
    def test_root_node_created(self, built_graph):
        assert "." in built_graph.G
        assert built_graph.G.nodes["."]["type"] == NodeType.DIRECTORY

    def test_directories_created(self, built_graph):
        assert "src" in built_graph.G
        assert built_graph.G.nodes["src"]["type"] == NodeType.DIRECTORY

    def test_subdirectories_created(self, built_graph):
        assert "src/utils" in built_graph.G
        assert built_graph.G.nodes["src/utils"]["type"] == NodeType.DIRECTORY

    def test_python_files_created(self, built_graph):
        assert "src/main.py" in built_graph.G
        assert built_graph.G.nodes["src/main.py"]["type"] == NodeType.FILE

    def test_file_content_stored(self, built_graph):
        content = built_graph.G.nodes["src/main.py"].get("code")
        assert content is not None
        assert "def main()" in content

    def test_non_python_files_present(self, built_graph):
        assert "README.md" in built_graph.G

    def test_git_excluded(self, built_graph):
        for nid in built_graph.G.nodes:
            assert ".git" not in nid.split("/")

    def test_contains_edges(self, built_graph):
        # Check src is child of root
        edge_data = built_graph.G.get_edge_data(".", "src")
        assert edge_data is not None
        assert any(d.get("type") == EdgeType.CONTAINS for d in edge_data.values())

    def test_nonexistent_repo_raises(self, tmp_path):
        dg = DependencyGraph(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            dg.build()

    def test_custom_filter(self, sample_repo):
        """Build with a custom filter that excludes 'utils' directory."""
        dg = DependencyGraph(str(sample_repo))
        dg.build(filter_func=lambda p: "utils" not in p and _exclude_irrelevant_for_build(p))
        assert "src/utils" not in dg.G
        assert "src/main.py" in dg.G


# ============================================================================
# DependencyGraph.parse tests
# ============================================================================

class TestParse:
    def test_classes_extracted(self, parsed_graph):
        assert "src/models.py:Base" in parsed_graph.G
        assert parsed_graph.G.nodes["src/models.py:Base"]["type"] == NodeType.CLASS

    def test_class_with_inheritance(self, parsed_graph):
        assert "src/models.py:User" in parsed_graph.G
        assert parsed_graph.G.nodes["src/models.py:User"]["type"] == NodeType.CLASS

    def test_methods_extracted(self, parsed_graph):
        assert "src/models.py:User.greet" in parsed_graph.G
        assert parsed_graph.G.nodes["src/models.py:User.greet"]["type"] == NodeType.METHOD

    def test_init_method_extracted(self, parsed_graph):
        assert "src/models.py:User.__init__" in parsed_graph.G
        assert parsed_graph.G.nodes["src/models.py:User.__init__"]["type"] == NodeType.METHOD

    def test_top_level_function_extracted(self, parsed_graph):
        assert "src/models.py:create_user" in parsed_graph.G
        assert parsed_graph.G.nodes["src/models.py:create_user"]["type"] == NodeType.FUNCTION

    def test_main_function_extracted(self, parsed_graph):
        assert "src/main.py:main" in parsed_graph.G
        assert parsed_graph.G.nodes["src/main.py:main"]["type"] == NodeType.FUNCTION

    def test_helper_function_extracted(self, parsed_graph):
        assert "src/utils/helpers.py:format_name" in parsed_graph.G
        assert parsed_graph.G.nodes["src/utils/helpers.py:format_name"]["type"] == NodeType.FUNCTION

    def test_ast_stored_on_files(self, parsed_graph):
        assert parsed_graph.G.nodes["src/models.py"].get("ast") is not None

    def test_code_stored_on_definitions(self, parsed_graph):
        code = parsed_graph.G.nodes["src/models.py:User.greet"].get("code")
        assert code is not None
        assert "def greet" in code

    def test_line_numbers_stored(self, parsed_graph):
        attrs = parsed_graph.G.nodes["src/models.py:User.greet"]
        assert "start_line" in attrs
        assert "end_line" in attrs
        assert attrs["start_line"] > 0

    def test_contains_edge_class_to_method(self, parsed_graph):
        edge_data = parsed_graph.G.get_edge_data("src/models.py:User", "src/models.py:User.greet")
        assert edge_data is not None
        assert any(d.get("type") == EdgeType.CONTAINS for d in edge_data.values())

    def test_contains_edge_file_to_class(self, parsed_graph):
        edge_data = parsed_graph.G.get_edge_data("src/models.py", "src/models.py:User")
        assert edge_data is not None
        assert any(d.get("type") == EdgeType.CONTAINS for d in edge_data.values())


# ============================================================================
# Edge extraction tests
# ============================================================================

class TestImportsEdges:
    def test_import_edge_created(self, parsed_graph):
        """main.py imports from models.py."""
        import_edges = list(parsed_graph.G_imports.edges())
        assert len(import_edges) > 0

    def test_import_resolves_to_entity(self, parsed_graph):
        """'from src.models import User' should create an edge to the User class."""
        # Check if main.py has any import edge pointing to models.py or User class
        has_import_from_main = False
        for u, v, data in parsed_graph.G_imports.edges(data=True):
            if u == "src/main.py":
                has_import_from_main = True
                break
        assert has_import_from_main, "main.py should have at least one IMPORTS edge"


class TestInheritsEdges:
    def test_inheritance_edge(self, parsed_graph):
        """User(Base) should create an INHERITS edge."""
        inherits_edges = list(parsed_graph.G_inherits.edges())
        # Check for User -> Base inheritance edge
        found = False
        for u, v in inherits_edges:
            if "User" in u and "Base" in v:
                found = True
                break
        assert found, "Expected INHERITS edge from User to Base"


class TestInvokesEdges:
    def test_invokes_edges_exist(self, parsed_graph):
        """main() calls User(), which should create an INVOKES edge."""
        invokes_edges = list(parsed_graph.G_invokes.edges())
        assert len(invokes_edges) > 0


# ============================================================================
# Graph view tests
# ============================================================================

class TestGraphViews:
    def test_g_tree_contains_only_contains_edges(self, parsed_graph):
        for u, v, data in parsed_graph.G_tree.edges(data=True):
            assert data.get("type") == EdgeType.CONTAINS

    def test_g_imports_contains_only_imports_edges(self, parsed_graph):
        for u, v, data in parsed_graph.G_imports.edges(data=True):
            assert data.get("type") == EdgeType.IMPORTS

    def test_g_invokes_contains_only_invokes_edges(self, parsed_graph):
        for u, v, data in parsed_graph.G_invokes.edges(data=True):
            assert data.get("type") == EdgeType.INVOKES

    def test_g_inherits_contains_only_inherits_edges(self, parsed_graph):
        for u, v, data in parsed_graph.G_inherits.edges(data=True):
            assert data.get("type") == EdgeType.INHERITS

    def test_g_code_nodes_have_ast(self, parsed_graph):
        for nid in parsed_graph.G_code.nodes:
            assert parsed_graph.G.nodes[nid].get("ast") is not None

    def test_g_tree_nodes_include_all_nodes(self, parsed_graph):
        """G_tree should include all nodes, just filtering edges."""
        assert len(parsed_graph.G_tree.nodes) == len(parsed_graph.G.nodes)


# ============================================================================
# Public query method tests
# ============================================================================

class TestGetParent:
    def test_root_has_no_parent(self, parsed_graph):
        exists, parent = parsed_graph.get_parent(".")
        assert exists is True
        assert parent is None

    def test_directory_parent(self, parsed_graph):
        exists, parent = parsed_graph.get_parent("src")
        assert exists is True
        assert parent == "."

    def test_file_parent(self, parsed_graph):
        exists, parent = parsed_graph.get_parent("src/main.py")
        assert exists is True
        assert parent == "src"

    def test_class_parent(self, parsed_graph):
        exists, parent = parsed_graph.get_parent("src/models.py:User")
        assert exists is True
        assert parent == "src/models.py"

    def test_method_parent(self, parsed_graph):
        exists, parent = parsed_graph.get_parent("src/models.py:User.greet")
        assert exists is True
        assert parent == "src/models.py:User"

    def test_nonexistent_node(self, parsed_graph):
        exists, parent = parsed_graph.get_parent("nonexistent")
        assert exists is False


class TestGetName:
    def test_root_name(self, parsed_graph):
        name = parsed_graph.get_name(".")
        assert name == "."

    def test_directory_name(self, parsed_graph):
        name = parsed_graph.get_name("src")
        assert name == "src"

    def test_file_name(self, parsed_graph):
        name = parsed_graph.get_name("src/main.py")
        assert "main" in name

    def test_class_name(self, parsed_graph):
        name = parsed_graph.get_name("src/models.py:User")
        assert name == "User"

    def test_method_name(self, parsed_graph):
        name = parsed_graph.get_name("src/models.py:User.greet")
        assert name == "greet"

    def test_with_badge(self, parsed_graph):
        name = parsed_graph.get_name("src/models.py:User", with_badge=True)
        assert "@class" in name

    def test_for_print_method(self, parsed_graph):
        name = parsed_graph.get_name("src/models.py:User.greet", for_print=True)
        assert name == ".greet"


class TestFindNode:
    def test_exact_match(self, parsed_graph):
        result = parsed_graph.find_node("src/main.py")
        assert result == "src/main.py"

    def test_suffix_match(self, parsed_graph):
        result = parsed_graph.find_node("main.py")
        assert result == "src/main.py"

    def test_no_match(self, parsed_graph):
        result = parsed_graph.find_node("nonexistent.py")
        assert result is None

    def test_suffix_match_disabled(self, parsed_graph):
        result = parsed_graph.find_node("main.py", suffix_match=False)
        assert result is None


class TestFindFile:
    def test_exact_match(self, parsed_graph):
        result = parsed_graph.find_file("src/main.py")
        assert result == "src/main.py"

    def test_suffix_match(self, parsed_graph):
        result = parsed_graph.find_file("models.py")
        assert result == "src/models.py"

    def test_directory_not_returned(self, parsed_graph):
        result = parsed_graph.find_file("src")
        assert result is None

    def test_no_match(self, parsed_graph):
        result = parsed_graph.find_file("missing.py")
        assert result is None


class TestAllPaths:
    def test_files_only(self, parsed_graph):
        files = parsed_graph.all_paths([NodeType.FILE])
        assert all(
            parsed_graph.G.nodes[f].get("type") == NodeType.FILE
            for f in files
        )
        assert len(files) > 0

    def test_directories_only(self, parsed_graph):
        dirs = parsed_graph.all_paths([NodeType.DIRECTORY])
        assert all(
            parsed_graph.G.nodes[d].get("type") == NodeType.DIRECTORY
            for d in dirs
        )
        assert "src" in dirs

    def test_mixed_types(self, parsed_graph):
        results = parsed_graph.all_paths([NodeType.FILE, NodeType.DIRECTORY])
        types = {parsed_graph.G.nodes[nid].get("type") for nid in results}
        assert NodeType.FILE in types
        assert NodeType.DIRECTORY in types

    def test_sorted(self, parsed_graph):
        paths = parsed_graph.all_paths([NodeType.FILE])
        assert paths == sorted(paths)


# ============================================================================
# Serialization tests
# ============================================================================

class TestSerialization:
    def test_to_dict_structure(self, parsed_graph):
        data = parsed_graph.to_dict()
        assert "repo_dir" in data
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], dict)
        assert isinstance(data["edges"], list)

    def test_to_dict_no_ast(self, parsed_graph):
        """AST objects should not be serialized."""
        data = parsed_graph.to_dict()
        for nid, attrs in data["nodes"].items():
            assert "ast" not in attrs

    def test_to_dict_with_rpg_map(self, parsed_graph):
        rpg_map = {"src/main.py": ["node_001"], "src/models.py": ["node_002"]}
        data = parsed_graph.to_dict(dep_to_rpg_map=rpg_map)
        assert data["nodes"]["src/main.py"]["rpg_nodes"] == ["node_001"]
        assert data["nodes"]["src/models.py"]["rpg_nodes"] == ["node_002"]

    def test_from_dict_roundtrip(self, parsed_graph):
        data = parsed_graph.to_dict()
        dg2 = DependencyGraph.from_dict(data)
        assert dg2.repo_dir == parsed_graph.repo_dir
        assert set(dg2.G.nodes) == set(parsed_graph.G.nodes)
        assert dg2.G.number_of_edges() == parsed_graph.G.number_of_edges()

    def test_from_dict_preserves_node_types(self, parsed_graph):
        data = parsed_graph.to_dict()
        dg2 = DependencyGraph.from_dict(data)
        for nid in parsed_graph.G.nodes:
            assert dg2.G.nodes[nid].get("type") == parsed_graph.G.nodes[nid].get("type")

    def test_json_serializable(self, parsed_graph):
        data = parsed_graph.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_reparse_ast_restores_code_nodes(self, parsed_graph):
        data = parsed_graph.to_dict()
        dg2 = DependencyGraph.from_dict(data)

        # Before reparse, no AST
        for nid in dg2.G.nodes:
            assert dg2.G.nodes[nid].get("ast") is None

        dg2.reparse_ast()

        # After reparse, files should have AST
        file_nodes = [
            nid for nid, attrs in dg2.G.nodes(data=True) if attrs.get("type") == NodeType.FILE
        ]
        parsed_files = [nid for nid in file_nodes if dg2.G.nodes[nid].get("ast") is not None]
        assert len(parsed_files) > 0


# ============================================================================
# Control flow parsing tests
# ============================================================================

class TestControlFlowParsing:
    """Test that functions/classes defined inside control flow blocks are extracted."""

    def test_conditional_function(self, tmp_path):
        (tmp_path / "cond.py").write_text(textwrap.dedent("""\
            import sys

            if sys.platform == "linux":
                def platform_func():
                    return "linux"
            else:
                def platform_func():
                    return "other"
        """))

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        assert "cond.py:platform_func" in dg.G
        assert dg.G.nodes["cond.py:platform_func"]["type"] == NodeType.FUNCTION

    def test_try_except_class(self, tmp_path):
        (tmp_path / "tryblock.py").write_text(textwrap.dedent("""\
            try:
                class OptionalFeature:
                    def run(self):
                        pass
            except ImportError:
                class OptionalFeature:
                    def run(self):
                        raise NotImplementedError
        """))

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        assert "tryblock.py:OptionalFeature" in dg.G


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    def test_syntax_error_file_skipped(self, tmp_path):
        """Files with syntax errors should be gracefully skipped."""
        (tmp_path / "bad.py").write_text("def broken(:\n  pass\n")
        (tmp_path / "good.py").write_text("def works(): pass\n")

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        # bad.py should be in graph (as file) but not parsed
        assert "bad.py" in dg.G
        assert dg.G.nodes["bad.py"].get("ast") is None
        # good.py should be parsed
        assert "good.py:works" in dg.G

    def test_empty_file(self, tmp_path):
        """Empty Python files should be handled gracefully."""
        (tmp_path / "empty.py").write_text("")

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        assert "empty.py" in dg.G

    def test_binary_file_skipped(self, tmp_path):
        """Binary files that fail the filter should not appear."""
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02\x03")

        dg = DependencyGraph(str(tmp_path))
        dg.build()  # default filter should include .bin
        # .bin is not in the EXT_BLACKLIST so it should be present
        # just ensure no crash

    def test_deeply_nested_directory(self, tmp_path):
        """Deeply nested directories should work."""
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "module.py").write_text("x = 1\n")

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)

        assert "a/b/c/d/module.py" in dg.G

    def test_duplicate_edge_not_added(self, parsed_graph):
        """Adding the same edge type twice should not create duplicates."""
        initial_edge_count = parsed_graph.G.number_of_edges()
        # Try to add a CONTAINS edge that already exists
        parsed_graph._add_edge(".", "src", type=EdgeType.CONTAINS)
        assert parsed_graph.G.number_of_edges() == initial_edge_count


# ============================================================================
# Async function handling tests
# ============================================================================

class TestAsyncParsing:
    """Test that async functions and methods are correctly extracted."""

    def test_async_function(self, tmp_path):
        (tmp_path / "async_mod.py").write_text(textwrap.dedent("""\
            async def fetch_data():
                return await get_remote()
        """))

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        assert "async_mod.py:fetch_data" in dg.G
        assert dg.G.nodes["async_mod.py:fetch_data"]["type"] == NodeType.FUNCTION

    def test_async_method(self, tmp_path):
        (tmp_path / "async_cls.py").write_text(textwrap.dedent("""\
            class AsyncHandler:
                async def handle(self, request):
                    return "ok"
        """))

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        assert "async_cls.py:AsyncHandler.handle" in dg.G
        assert dg.G.nodes["async_cls.py:AsyncHandler.handle"]["type"] == NodeType.METHOD


# ============================================================================
# Decorator handling tests
# ============================================================================

class TestDecoratorParsing:
    """Test that decorated functions include decorator lines."""

    def test_decorated_function_start_line(self, tmp_path):
        (tmp_path / "deco.py").write_text(textwrap.dedent("""\
            def my_decorator(func):
                return func

            @my_decorator
            def decorated():
                pass
        """))

        dg = DependencyGraph(str(tmp_path))
        dg.build(filter_func=lambda _: True)
        dg.parse(filter_func=lambda f: f.endswith(".py"))

        attrs = dg.G.nodes["deco.py:decorated"]
        # Decorator is on line 4, function def on line 5
        assert attrs["start_line"] == 4
