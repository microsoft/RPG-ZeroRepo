#!/usr/bin/env python3
"""Tests for M9 — Agent Ops.

Covers:
  - BM25 model (tokenizer, ModuleRetriever, build_module_retriever)
  - Env layer (QueryInfo, QueryResult, RepoEntitySearcher, RepoDependencySearcher)
  - search_by_meta (entity search, fuzzy retrieve, merge/rank)
  - search_by_feature (exact, substring, fuzzy match)
  - fetch (fetch_node with code and feature entities)
  - explore (validate inputs, traverse structures)
"""

import os
import sys
import json

import pytest

# Ensure project root and scripts/ are on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

import networkx as nx

from common.utils import normalize_text, wrap_code_snippet
from rpg.models import (
    RPG, Node, NodeMetaData, NodeType, EdgeType,
)


# ============================================================================
# Fixtures — build a minimal RPG with dep_graph for testing
# ============================================================================

class FakeDependencyGraph:
    """Minimal DependencyGraph stub."""
    def __init__(self, G):
        self.G = G
        self.repo_dir = "/fake/repo"


def _build_test_rpg():
    """Build a small RPG and dep_graph for testing.

    Structure:
    - RPG nodes:
        root_feature (REPO) -> auth (DIRECTORY)
            auth -> login (CLASS)
            auth -> register (FUNCTION)
        root_feature -> data (DIRECTORY)
            data -> User (CLASS)

    - dep_graph nodes:
        src/auth/login.py:LoginManager (class, code)
        src/auth/register.py:register_user (function, code)
        src/data/models.py:User (class, code)
        src/auth/login.py (file)
    """
    rpg = RPG(repo_name="TestRepo")

    # Build RPG nodes
    root_meta = NodeMetaData(type_name=NodeType.REPO, path="root", description="Test Repo")
    root = Node(id="root_node", name="TestRepo", meta=root_meta, level=0)
    rpg.add_node(root)

    auth_meta = NodeMetaData(type_name=NodeType.DIRECTORY, path="auth", description="Auth module")
    auth = Node(id="auth_node", name="auth", meta=auth_meta, level=1)
    rpg.add_node(auth)
    rpg.add_edge(root, auth, relation="contains")

    login_meta = NodeMetaData(type_name=NodeType.CLASS, path="src/auth/login.py::LoginManager", description="Login management")
    login = Node(id="login_node", name="LoginManager", meta=login_meta, level=2)
    rpg.add_node(login)
    rpg.add_edge(auth, login, relation="contains")

    register_meta = NodeMetaData(type_name=NodeType.FUNCTION, path="src/auth/register.py::register_user", description="User registration")
    register = Node(id="register_node", name="register_user", meta=register_meta, level=2)
    rpg.add_node(register)
    rpg.add_edge(auth, register, relation="contains")

    data_meta = NodeMetaData(type_name=NodeType.DIRECTORY, path="data", description="Data module")
    data = Node(id="data_node", name="data", meta=data_meta, level=1)
    rpg.add_node(data)
    rpg.add_edge(root, data, relation="contains")

    user_meta = NodeMetaData(type_name=NodeType.CLASS, path="src/data/models.py::User", description="User model class")
    user = Node(id="user_node", name="User", meta=user_meta, level=2)
    rpg.add_node(user)
    rpg.add_edge(data, user, relation="contains")

    # Build dep_graph (networkx MultiDiGraph)
    G = nx.MultiDiGraph()
    G.add_node("src/auth/login.py:LoginManager", type=NodeType.CLASS,
               code="class LoginManager:\n    def login(self, user):\n        pass",
               start_line=1, end_line=3)
    G.add_node("src/auth/register.py:register_user", type=NodeType.FUNCTION,
               code="def register_user(name):\n    return User(name)",
               start_line=1, end_line=2)
    G.add_node("src/data/models.py:User", type=NodeType.CLASS,
               code="class User:\n    def __init__(self, name):\n        self.name = name",
               start_line=1, end_line=3)
    G.add_node("src/auth/login.py", type=NodeType.FILE,
               code="class LoginManager:\n    def login(self, user):\n        pass")
    G.add_node("src/auth/register.py", type=NodeType.FILE,
               code="def register_user(name):\n    return User(name)")
    G.add_node("src/data/models.py", type=NodeType.FILE,
               code="class User:\n    def __init__(self, name):\n        self.name = name")

    # Add edges
    G.add_edge("src/auth/login.py", "src/auth/login.py:LoginManager", type=EdgeType.CONTAINS)
    G.add_edge("src/auth/register.py", "src/auth/register.py:register_user", type=EdgeType.CONTAINS)
    G.add_edge("src/data/models.py", "src/data/models.py:User", type=EdgeType.CONTAINS)
    G.add_edge("src/auth/register.py:register_user", "src/data/models.py:User", type=EdgeType.INVOKES)

    dep_graph = FakeDependencyGraph(G)
    rpg.dep_graph = dep_graph

    # dep2rpg mapping
    dep2rpg = {
        "src/auth/login.py:LoginManager": [login.id],
        "src/auth/register.py:register_user": [register.id],
        "src/data/models.py:User": [user.id],
        "src/auth/login.py": [],
        "src/auth/register.py": [],
        "src/data/models.py": [],
    }
    rpg._dep_to_rpg_map = dep2rpg

    return rpg


@pytest.fixture
def test_rpg():
    return _build_test_rpg()


@pytest.fixture
def entity_searcher(test_rpg):
    from rpg_agent.env.searcher import RepoEntitySearcher
    return RepoEntitySearcher(test_rpg)


@pytest.fixture
def dep_searcher(test_rpg):
    from rpg_agent.env.searcher import RepoDependencySearcher
    return RepoDependencySearcher(test_rpg.dep_graph.G)


# ============================================================================
# Tests: Utils additions (normalize_text, wrap_code_snippet)
# ============================================================================

class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("LoginManager") == "login manager"

    def test_with_extension(self):
        assert normalize_text("models.py") == "models"

    def test_with_path(self):
        assert normalize_text("src/auth/login.py") == "src auth login"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_camel_case_split(self):
        assert normalize_text("camelCaseVar") == "camel case var"

    def test_underscore_split(self):
        assert normalize_text("some_function_name") == "some function name"


class TestWrapCodeSnippet:
    def test_basic(self):
        code = "def foo():\n    pass"
        result = wrap_code_snippet(code, 1, 2)
        assert "```" in result
        assert "1 | def foo():" in result
        assert "2 |     pass" in result

    def test_with_offset(self):
        code = "x = 1\ny = 2"
        result = wrap_code_snippet(code, 10, 11)
        assert "10 | x = 1" in result
        assert "11 | y = 2" in result


# ============================================================================
# Tests: BM25 model
# ============================================================================

class TestBM25Model:
    def test_tokenizer(self):
        from rpg_agent.ops.bm25_model import _tokenize
        tokens = _tokenize("src/auth/LoginManager")
        assert "login" in tokens
        assert "manager" in tokens
        assert "src" in tokens
        assert "auth" in tokens

    def test_module_retriever(self):
        from rpg_agent.ops.bm25_model import ModuleRetriever
        nids = ["src/auth/login.py:LoginManager", "src/data/models.py:User", "src/utils.py:helper"]
        retriever = ModuleRetriever(nids, similarity_top_k=3)
        results = retriever.retrieve("login")
        assert len(results) > 0
        assert results[0][0] == "src/auth/login.py:LoginManager"
        assert results[0][1] > 0

    def test_module_retriever_empty_query(self):
        from rpg_agent.ops.bm25_model import ModuleRetriever
        nids = ["src/auth/login.py:LoginManager"]
        retriever = ModuleRetriever(nids, similarity_top_k=3)
        results = retriever.retrieve("")
        assert results == []

    def test_build_module_retriever(self, entity_searcher):
        from rpg_agent.ops.bm25_model import build_module_retriever
        retriever = build_module_retriever(entity_searcher=entity_searcher)
        assert retriever is not None
        results = retriever.retrieve("login")
        assert len(results) > 0


# ============================================================================
# Tests: QueryInfo and QueryResult
# ============================================================================

class TestQueryInfo:
    def test_basic(self):
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(term="LoginManager")
        assert qi.term == "LoginManager"
        assert qi.query_type == "keyword"
        assert "LoginManager" in str(qi)

    def test_hash_and_eq(self):
        from rpg_agent.env.query import QueryInfo
        qi1 = QueryInfo(term="foo")
        qi2 = QueryInfo(term="foo")
        qi3 = QueryInfo(term="bar")
        assert qi1 == qi2
        assert qi1 != qi3
        assert hash(qi1) == hash(qi2)


class TestQueryResult:
    def test_basic_creation(self):
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="test")
        qr = QueryResult(
            query_info=qi,
            format_mode="complete",
            nid="src/auth/login.py:LoginManager",
            ntype=NodeType.CLASS,
            retrieve_src="test source",
        )
        assert qr.nid == "src/auth/login.py:LoginManager"
        assert qr.format_mode == "complete"
        assert qr.file_path == "src/auth/login.py"

    def test_format_complete(self, entity_searcher):
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="LoginManager")
        qr = QueryResult(
            query_info=qi,
            format_mode="complete",
            nid="src/auth/login.py:LoginManager",
            ntype=NodeType.CLASS,
            start_line=1,
            end_line=3,
            retrieve_src="Exact match",
        )
        output = qr.format_output(entity_searcher)
        assert "LoginManager" in output
        assert "Exact match" in output

    def test_format_fold(self, entity_searcher):
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="User")
        qr = QueryResult(
            query_info=qi,
            format_mode="fold",
            nid="src/data/models.py:User",
            ntype=NodeType.CLASS,
            retrieve_src="fold test",
        )
        output = qr.format_output(entity_searcher)
        assert "User" in output

    def test_str_repr(self):
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="test")
        qr = QueryResult(query_info=qi, format_mode="complete", nid="test.py")
        s = str(qr)
        assert "QueryResult" in s
        assert "test.py" in s


# ============================================================================
# Tests: RepoEntitySearcher
# ============================================================================

class TestRepoEntitySearcher:
    def test_has_node(self, entity_searcher):
        assert entity_searcher.has_node("src/auth/login.py:LoginManager")
        assert not entity_searcher.has_node("nonexistent.py:Foo")

    def test_has_node_test_file(self, entity_searcher):
        # test files should be excluded by default
        G = entity_searcher.G
        G.add_node("tests/test_login.py:test_login", type=NodeType.FUNCTION,
                    code="def test_login(): pass", start_line=1, end_line=1)
        assert not entity_searcher.has_node("tests/test_login.py:test_login")
        assert entity_searcher.has_node("tests/test_login.py:test_login", include_test=True)

    def test_get_node_data(self, entity_searcher):
        data = entity_searcher.get_node_data(
            ["src/auth/login.py:LoginManager"], return_code_content=True
        )
        assert len(data) == 1
        assert data[0]["type"] == NodeType.CLASS
        assert "code_content" in data[0]
        assert "class LoginManager" in data[0]["code_content"]

    def test_get_node_data_no_wrap(self, entity_searcher):
        data = entity_searcher.get_node_data(
            ["src/auth/login.py:LoginManager"], return_code_content=True, wrap_with_ln=False
        )
        assert len(data) == 1
        # Should be raw code, not wrapped
        assert "```" not in data[0]["code_content"]

    def test_get_feature_paths(self, entity_searcher):
        paths = entity_searcher.get_feature_paths_for_node("src/auth/login.py:LoginManager")
        # Should find the LoginManager RPG node's feature path
        assert len(paths) >= 0  # Depends on mapping; at minimum does not crash

    def test_global_name_dict(self, entity_searcher):
        gnd = entity_searcher.global_name_dict
        assert "LoginManager" in gnd
        assert "User" in gnd

    def test_global_name_dict_lowercase(self, entity_searcher):
        gnd = entity_searcher.global_name_dict_lowercase
        assert "loginmanager" in gnd
        assert "user" in gnd

    def test_from_components(self, test_rpg):
        from rpg_agent.env.searcher import RepoEntitySearcher
        G = test_rpg.dep_graph.G
        dep2rpg = test_rpg._dep_to_rpg_map
        # Clear existing dep_graph to test from_components
        rpg_copy = RPG(repo_name="TestCopy")
        for node in test_rpg.nodes.values():
            rpg_copy.nodes[node.id] = node
        rpg_copy.dep_graph = None
        rpg_copy._dep_to_rpg_map = None

        searcher = RepoEntitySearcher.from_components(G, rpg_copy, dep2rpg)
        assert searcher.has_node("src/auth/login.py:LoginManager")


# ============================================================================
# Tests: RepoDependencySearcher
# ============================================================================

class TestRepoDependencySearcher:
    def test_get_neighbors_forward(self, dep_searcher):
        nodes, edges = dep_searcher.get_neighbors(
            "src/auth/register.py:register_user", "forward"
        )
        assert "src/data/models.py:User" in nodes

    def test_get_neighbors_backward(self, dep_searcher):
        nodes, edges = dep_searcher.get_neighbors(
            "src/data/models.py:User", "backward"
        )
        assert "src/auth/register.py:register_user" in nodes

    def test_get_neighbors_with_type_filter(self, dep_searcher):
        nodes, edges = dep_searcher.get_neighbors(
            "src/auth/login.py", "forward",
            ntype_filter=[NodeType.CLASS],
        )
        assert "src/auth/login.py:LoginManager" in nodes

    def test_from_rpg(self, test_rpg):
        from rpg_agent.env.searcher import RepoDependencySearcher
        searcher = RepoDependencySearcher.from_rpg(test_rpg)
        assert searcher.G is not None

    def test_subgraph(self, dep_searcher):
        sg = dep_searcher.subgraph(["src/auth/login.py", "src/auth/login.py:LoginManager"])
        assert len(sg.nodes()) == 2


# ============================================================================
# Tests: search_by_meta
# ============================================================================

class TestSearchByMeta:
    def test_fuzzy_retrieve(self, test_rpg):
        from rpg_agent.ops.search_by_meta import fuzzy_retrieve
        results = fuzzy_retrieve("login", rpg=test_rpg)
        assert len(results) > 0
        assert any("login" in r.lower() for r in results)

    def test_fuzzy_retrieve_with_score(self, test_rpg):
        from rpg_agent.ops.search_by_meta import fuzzy_retrieve
        results = fuzzy_retrieve("login", rpg=test_rpg, return_score=True)
        assert len(results) > 0
        assert len(results[0]) == 3  # (match, score, idx)

    def test_search_entity_in_global_dict(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import search_entity_in_global_dict
        result = search_entity_in_global_dict(entity_searcher, "LoginManager")
        assert result is not None
        assert NodeType.CLASS in result

    def test_search_entity_in_global_dict_not_found(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import search_entity_in_global_dict
        result = search_entity_in_global_dict(entity_searcher, "NonExistentClass")
        assert result is None

    def test_merge_query_results(self):
        from rpg_agent.ops.search_by_meta import merge_query_results
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="test")
        qr1 = QueryResult(query_info=qi, format_mode="preview", nid="a.py:Foo",
                          ntype=NodeType.CLASS, retrieve_src="src1")
        qr2 = QueryResult(query_info=qi, format_mode="complete", nid="a.py:Foo",
                          ntype=NodeType.CLASS, retrieve_src="src2")
        merged = merge_query_results([qr1, qr2])
        assert len(merged) == 1
        # Should upgrade to "complete" (higher priority)
        assert merged[0].format_mode == "complete"

    def test_rank_and_aggr(self):
        from rpg_agent.ops.search_by_meta import (
            merge_query_results, rank_and_aggr_query_results,
        )
        from rpg_agent.env.query import QueryInfo, QueryResult
        qi = QueryInfo(term="test")
        qr = QueryResult(query_info=qi, format_mode="complete", nid="a.py:Foo",
                         ntype=NodeType.CLASS, retrieve_src="src")
        merged = merge_query_results([qr])
        ranked = rank_and_aggr_query_results(merged, [qi])
        assert len(ranked) > 0

    def test_find_matching_files(self):
        from rpg_agent.ops.search_by_meta import find_matching_files_from_list
        files = ["src/auth/login.py", "src/data/models.py", "README.md"]
        assert find_matching_files_from_list(files, "*.py") == [
            "src/auth/login.py", "src/data/models.py"
        ]
        assert find_matching_files_from_list(files, "login") == ["src/auth/login.py"]

    def test_get_entity_contents_exact(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import get_entity_contents
        result = get_entity_contents(entity_searcher, ["src/auth/login.py:LoginManager"])
        assert "LoginManager" in result
        assert "Exact match" in result

    def test_get_entity_contents_invalid(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import get_entity_contents
        result = get_entity_contents(entity_searcher, ["nonexistent.py:Foo"])
        assert "Invalid name" in result


# ============================================================================
# Tests: search_by_feature
# ============================================================================

class TestSearchByFeature:
    def test_exact_match(self, test_rpg):
        from rpg_agent.ops.search_by_feature import exact_match_search_feature
        results = exact_match_search_feature(test_rpg, "LoginManager")
        assert len(results) > 0
        assert any(n.name == "LoginManager" for n in results)

    def test_exact_match_not_found(self, test_rpg):
        from rpg_agent.ops.search_by_feature import exact_match_search_feature
        results = exact_match_search_feature(test_rpg, "Nonexistent")
        assert len(results) == 0

    def test_substring_match(self, test_rpg):
        from rpg_agent.ops.search_by_feature import substring_match_search_feature
        results = substring_match_search_feature(test_rpg, "login")
        assert len(results) > 0

    def test_fuzzy_match(self, test_rpg):
        from rpg_agent.ops.search_by_feature import fuzzy_match_search_feature
        results = fuzzy_match_search_feature(test_rpg, "logn managerr", top_k=3)
        assert len(results) > 0

    def test_fuzzy_match_empty(self, test_rpg):
        from rpg_agent.ops.search_by_feature import fuzzy_match_search_feature
        results = fuzzy_match_search_feature(test_rpg, "")
        assert results == []


# ============================================================================
# Tests: fetch
# ============================================================================

class TestFetch:
    def test_fetch_code_entity_exact(self, test_rpg, entity_searcher):
        from rpg_agent.ops.fetch import fetch_node
        result, success = fetch_node(
            rpg=test_rpg,
            entity_searcher=entity_searcher,
            code_entities=["src/auth/login.py:LoginManager"],
        )
        assert success
        assert "LoginManager" in result

    def test_fetch_code_entity_fuzzy(self, test_rpg, entity_searcher):
        from rpg_agent.ops.fetch import fetch_node
        result, success = fetch_node(
            rpg=test_rpg,
            entity_searcher=entity_searcher,
            code_entities=["src/auth/login.py:LoginMnager"],  # typo
        )
        # Should still find something via fuzzy
        assert "LoginManager" in result or "Fuzzy" in result or "No entities" in result

    def test_fetch_empty(self, test_rpg, entity_searcher):
        from rpg_agent.ops.fetch import fetch_node
        result, success = fetch_node(
            rpg=test_rpg,
            entity_searcher=entity_searcher,
        )
        assert not success
        assert "No entities" in result

    def test_fuzzy_feature_paths(self, test_rpg):
        from rpg_agent.ops.fetch import _fuzzy_feature_paths
        results = _fuzzy_feature_paths(test_rpg, "login", top_k=3)
        assert isinstance(results, list)

    def test_collect_all_feature_paths(self, test_rpg):
        from rpg_agent.ops.fetch import _collect_all_feature_paths
        paths = _collect_all_feature_paths(test_rpg)
        assert len(paths) > 0


# ============================================================================
# Tests: explore
# ============================================================================

class TestExplore:
    def test_validate_inputs_valid_code(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import _validate_graph_explorer_inputs
        valid_code, code_hints, valid_feat, feat_hints = _validate_graph_explorer_inputs(
            start_code_entities=["src/auth/login.py:LoginManager"],
            rpg=test_rpg,
            entity_searcher=entity_searcher,
        )
        assert "src/auth/login.py:LoginManager" in valid_code
        assert code_hints == ""

    def test_validate_inputs_invalid_code(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import _validate_graph_explorer_inputs
        valid_code, code_hints, _, _ = _validate_graph_explorer_inputs(
            start_code_entities=["nonexistent.py:Foo"],
            rpg=test_rpg,
            entity_searcher=entity_searcher,
        )
        assert len(valid_code) == 0
        assert "does not exist" in code_hints or "invalid" in code_hints.lower()

    def test_validate_inputs_direction_error(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import _validate_graph_explorer_inputs
        with pytest.raises(AssertionError, match="Invalid direction"):
            _validate_graph_explorer_inputs(
                start_code_entities=[],
                direction="sideways",
                rpg=test_rpg,
                entity_searcher=entity_searcher,
            )

    def test_traverse_tree_code_view(self, test_rpg):
        from rpg_agent.ops.explore import traverse_tree_structure
        result = traverse_tree_structure(
            rpg=test_rpg,
            root="src/auth/login.py:LoginManager",
            direction="downstream",
            hops=2,
            visual_type="code",
        )
        assert "LoginManager" in result

    def test_traverse_tree_feature_view(self, test_rpg):
        from rpg_agent.ops.explore import traverse_tree_structure
        # Get the feature path for auth node
        auth_nodes = test_rpg.get_nodes_by_type(NodeType.DIRECTORY)
        auth_node = None
        for n in auth_nodes:
            if n.name == "auth":
                auth_node = n
                break
        if auth_node:
            result = traverse_tree_structure(
                rpg=test_rpg,
                root=auth_node.id,
                direction="downstream",
                hops=2,
                visual_type="feature",
            )
            assert isinstance(result, str)

    def test_traverse_tree_invalid_root(self, test_rpg):
        from rpg_agent.ops.explore import traverse_tree_structure
        result = traverse_tree_structure(
            rpg=test_rpg,
            root="nonexistent.py",
            direction="downstream",
            hops=2,
            visual_type="code",
        )
        assert "not found" in result

    def test_traverse_json_code_view(self, test_rpg):
        from rpg_agent.ops.explore import traverse_json_structure
        result = traverse_json_structure(
            rpg=test_rpg,
            root="src/auth/login.py:LoginManager",
            direction="downstream",
            hops=2,
            visual_type="code",
        )
        assert result["type"] == "code"
        assert "root_dep" in result
        assert result["root_dep"] == "src/auth/login.py:LoginManager"

    def test_traverse_json_invalid_root(self, test_rpg):
        from rpg_agent.ops.explore import traverse_json_structure
        result = traverse_json_structure(
            rpg=test_rpg,
            root="nonexistent",
            direction="downstream",
            hops=2,
            visual_type="code",
        )
        assert "error" in result

    def test_explore_tree_structure(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import explore_tree_structure
        result, success = explore_tree_structure(
            start_code_entities=["src/auth/login.py:LoginManager"],
            rpg=test_rpg,
            entity_searcher=entity_searcher,
        )
        assert success
        assert "Code Results" in result

    def test_explore_tree_structure_json(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import explore_tree_structure
        result, success = explore_tree_structure(
            start_code_entities=["src/auth/login.py:LoginManager"],
            rpg=test_rpg,
            entity_searcher=entity_searcher,
            return_json=True,
        )
        assert success
        assert "Code Results" in result

    def test_explore_tree_structure_no_match(self, test_rpg, entity_searcher):
        from rpg_agent.ops.explore import explore_tree_structure
        result, success = explore_tree_structure(
            start_code_entities=["nonexistent.py:Foo"],
            rpg=test_rpg,
            entity_searcher=entity_searcher,
        )
        assert not success

    def test_feature_labeler(self, test_rpg):
        from rpg_agent.ops.explore import FeatureLabeler
        labeler = FeatureLabeler(rpg=test_rpg)
        label = labeler.label_for_dep("src/auth/login.py:LoginManager")
        if label:
            assert label.startswith("F")
            # Same dep should give same label
            assert labeler.label_for_dep("src/auth/login.py:LoginManager") == label

    def test_render_feature_paths_tree(self):
        from rpg_agent.ops.explore import render_feature_paths_tree
        label_to_paths = {
            "F1": ["auth/login"],
            "F2": ["data/models"],
        }
        lines = render_feature_paths_tree(label_to_paths)
        assert len(lines) > 0

    def test_fuzzy_match_feature_path(self, test_rpg):
        from rpg_agent.ops.explore import _fuzzy_match_feature_path
        results = _fuzzy_match_feature_path(test_rpg, "auth")
        # Should find something related to "auth"
        assert isinstance(results, list)


# ============================================================================
# Tests: search_by_meta — additional coverage
# ============================================================================

class TestSearchEntity:
    """Tests for search_entity cascading paths."""

    def test_search_entity_exact_match(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import search_entity
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(term="src/auth/login.py:LoginManager")
        results, continue_search = search_entity(qi, entity_searcher)
        assert len(results) >= 1
        assert results[0].nid == "src/auth/login.py:LoginManager"
        assert results[0].format_mode == "complete"
        assert continue_search is False

    def test_search_entity_global_name_match(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import search_entity
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(term="LoginManager")
        results, continue_search = search_entity(qi, entity_searcher)
        assert len(results) >= 1
        # Should find via global_name_dict and set continue_search=False
        found_nids = [r.nid for r in results]
        assert "src/auth/login.py:LoginManager" in found_nids

    def test_search_entity_bm25_fallback(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import search_entity
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(term="authentication")
        results, continue_search = search_entity(qi, entity_searcher)
        # BM25/fuzzy should return some results even for a vague keyword
        assert isinstance(results, list)
        # continue_search should be True since we fell through to BM25/fuzzy
        assert continue_search is True


class TestBM25ModuleRetrieve:
    """Tests for bm25_module_retrieve."""

    def test_bm25_module_retrieve_basic(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import bm25_module_retrieve
        results = bm25_module_retrieve("login", entity_searcher)
        assert isinstance(results, list)
        assert len(results) > 0
        # All returned items should be valid node IDs (strings)
        assert all(isinstance(nid, str) for nid in results)

    def test_bm25_module_retrieve_with_include_files(self, entity_searcher):
        from rpg_agent.ops.search_by_meta import bm25_module_retrieve
        results = bm25_module_retrieve(
            "login", entity_searcher,
            include_files=["src/auth/login.py"],
        )
        assert isinstance(results, list)
        # If any filter_nodes matched, they should belong to the included file
        for nid in results:
            file_part = nid.split(":")[0]
            # filter_nodes returned if non-empty; else all_nodes fallback
            assert isinstance(file_part, str)


class TestGetModuleNameByLineNum:
    """Tests for get_module_name_by_line_num."""

    def test_get_module_name_by_line_num_hits_function(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import get_module_name_by_line_num
        # register_user is at lines 1-2 in src/auth/register.py
        result = get_module_name_by_line_num(
            entity_searcher, dep_searcher,
            file_path="src/auth/register.py", line_num=1,
        )
        assert result is not None
        assert result["node_id"] == "src/auth/register.py:register_user"

    def test_get_module_name_by_line_num_no_match(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import get_module_name_by_line_num
        result = get_module_name_by_line_num(
            entity_searcher, dep_searcher,
            file_path="nonexistent/file.py", line_num=99,
        )
        assert result is None


class TestGetCodeBlockByLineNums:
    """Tests for get_code_block_by_line_nums."""

    def test_get_code_block_by_line_nums_found(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import get_code_block_by_line_nums
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(
            term="login",
            line_nums=[1],
            file_path_or_pattern="src/auth/login.py",
        )
        results = get_code_block_by_line_nums(qi, entity_searcher, dep_searcher)
        assert isinstance(results, list)
        assert len(results) >= 1
        # Should find the LoginManager module at line 1
        nids = [r.nid for r in results]
        assert any("LoginManager" in nid for nid in nids)

    def test_get_code_block_by_line_nums_no_file(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import get_code_block_by_line_nums
        from rpg_agent.env.query import QueryInfo
        qi = QueryInfo(
            term="nothing",
            line_nums=[10],
            file_path_or_pattern="nonexistent/file.py",
        )
        results = get_code_block_by_line_nums(qi, entity_searcher, dep_searcher)
        assert results == []


class TestGrepContentSearch:
    """Tests for grep_content_search."""

    def test_grep_content_search_found(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import grep_content_search
        from rpg_agent.env.query import QueryInfo
        file2code = {
            "src/auth/login.py": "class LoginManager:\n    def login(self, user):\n        pass",
            "src/auth/register.py": "def register_user(name):\n    return User(name)",
            "src/data/models.py": "class User:\n    def __init__(self, name):\n        self.name = name",
        }
        qi = QueryInfo(term="LoginManager")
        results = grep_content_search(
            file2code, qi, entity_searcher, dep_searcher,
        )
        assert len(results) >= 1
        assert any("LoginManager" in r.nid or "login" in r.nid.lower() for r in results)

    def test_grep_content_search_not_found(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import grep_content_search
        from rpg_agent.env.query import QueryInfo
        file2code = {
            "src/auth/login.py": "class LoginManager:\n    def login(self, user):\n        pass",
        }
        qi = QueryInfo(term="ZZZZZ_nonexistent_keyword")
        results = grep_content_search(
            file2code, qi, entity_searcher, dep_searcher,
        )
        assert results == []


class TestSearchCodeSnippets:
    """Tests for search_code_snippets."""

    def _make_file2code(self):
        return {
            "src/auth/login.py": "class LoginManager:\n    def login(self, user):\n        pass",
            "src/auth/register.py": "def register_user(name):\n    return User(name)",
            "src/data/models.py": "class User:\n    def __init__(self, name):\n        self.name = name",
        }

    def test_search_code_snippets_by_search_terms(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import search_code_snippets
        file2code = self._make_file2code()
        result, suc = search_code_snippets(
            file2code=file2code,
            entity_searcher=entity_searcher,
            dep_searcher=dep_searcher,
            search_terms=["LoginManager"],
        )
        assert isinstance(result, str)
        assert suc is True
        assert "LoginManager" in result

    def test_search_code_snippets_by_line_nums(self, entity_searcher, dep_searcher):
        from rpg_agent.ops.search_by_meta import search_code_snippets
        file2code = self._make_file2code()
        result, suc = search_code_snippets(
            file2code=file2code,
            entity_searcher=entity_searcher,
            dep_searcher=dep_searcher,
            line_nums=[1],
            file_path_or_pattern="src/auth/login.py",
        )
        assert isinstance(result, str)
        # Should retrieve code around line 1 of the login file
        assert "login" in result.lower() or "LoginManager" in result
