#!/usr/bin/env python3
"""Tests for M10 — Agent Tools & Environment.

Covers:
  - env/env.py: Env construction, step(), reset(), accessors
  - tools/: custom_parse and execute for all 6 tool classes
  - env __init__ exports
"""

import asyncio
import json
import os
import sys

import pytest

# Ensure project root and scripts/ are on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

import networkx as nx

from common.tools import (
    Tool,
    ToolCall,
    ToolCallArguments,
    ToolExecResult,
    ToolExecutor,
    ToolHandler,
    ToolResult,
)
from rpg.models import (
    RPG,
    Edge,
    EdgeType,
    Node,
    NodeMetaData,
    NodeType,
)


# ============================================================================
# Helpers — minimal RPG fixture (matches test_agent_ops.py pattern)
# ============================================================================

class FakeDependencyGraph:
    """Minimal DependencyGraph stub."""
    def __init__(self, G):
        self.G = G
        self.repo_dir = "/fake/repo"


def _build_test_rpg():
    """Build a small RPG and dep_graph for testing.

    RPG nodes: root -> auth -> login, register
                    -> data -> user
    dep_graph nodes: src/auth.py:Auth (class), src/utils.py:helper (function)
    """
    rpg = RPG(repo_name="TestRepo")

    root_meta = NodeMetaData(type_name=NodeType.REPO, path="root", description="Test Repo")
    root = Node(id="root_node", name="TestRepo", meta=root_meta, level=0)
    rpg.add_node(root)

    auth_meta = NodeMetaData(type_name=NodeType.DIRECTORY, path="auth", description="Auth module")
    auth = Node(id="auth_node", name="auth", meta=auth_meta, level=1)
    rpg.add_node(auth)
    rpg.add_edge(root, auth, relation="contains")

    login_meta = NodeMetaData(type_name=NodeType.CLASS, path="src/auth.py::Auth", description="Auth class")
    login = Node(id="login_node", name="Auth", meta=login_meta, level=2)
    rpg.add_node(login)
    rpg.add_edge(auth, login, relation="contains")

    login_method_meta = NodeMetaData(
        type_name=NodeType.FUNCTION, path="src/auth.py::Auth.login",
        description="Login method",
    )
    login_method = Node(id="login_method_node", name="login", meta=login_method_meta, level=3)
    rpg.add_node(login_method)
    rpg.add_edge(login, login_method, relation="contains")

    utils_meta = NodeMetaData(type_name=NodeType.FUNCTION, path="src/utils.py::helper", description="Helper function")
    utils = Node(id="utils_node", name="helper", meta=utils_meta, level=2)
    rpg.add_node(utils)
    rpg.add_edge(auth, utils, relation="contains")

    # Build dep_graph
    G = nx.MultiDiGraph()
    G.add_node("src/auth.py", type=NodeType.FILE,
               code="class Auth:\n    def login(self): pass\n    def logout(self): pass\n")
    G.add_node("src/utils.py", type=NodeType.FILE,
               code="def helper(): pass\n")
    G.add_node("src/auth.py:Auth", type=NodeType.CLASS,
               code="class Auth:\n    def login(self): pass\n    def logout(self): pass\n",
               start_line=1, end_line=3)
    G.add_node("src/auth.py:Auth.login", type=NodeType.METHOD,
               code="    def login(self): pass",
               start_line=2, end_line=2)
    G.add_node("src/auth.py:Auth.logout", type=NodeType.METHOD,
               code="    def logout(self): pass",
               start_line=3, end_line=3)
    G.add_node("src/utils.py:helper", type=NodeType.FUNCTION,
               code="def helper(): pass",
               start_line=1, end_line=1)

    G.add_edge("src/auth.py", "src/auth.py:Auth", type=EdgeType.CONTAINS)
    G.add_edge("src/auth.py:Auth", "src/auth.py:Auth.login", type=EdgeType.CONTAINS)
    G.add_edge("src/auth.py:Auth", "src/auth.py:Auth.logout", type=EdgeType.CONTAINS)
    G.add_edge("src/utils.py", "src/utils.py:helper", type=EdgeType.CONTAINS)
    G.add_edge("src/auth.py:Auth.login", "src/utils.py:helper", type=EdgeType.INVOKES)

    dep_graph = FakeDependencyGraph(G)
    rpg.dep_graph = dep_graph

    dep2rpg = {
        "src/auth.py:Auth": ["login_node"],
        "src/auth.py:Auth.login": ["login_method_node"],
        "src/utils.py:helper": ["utils_node"],
        "src/auth.py": [],
        "src/utils.py": [],
    }
    rpg._dep_to_rpg_map = dep2rpg

    return rpg


# ============================================================================
# 1. parse_thinking_output
# ============================================================================

class TestParseThinkingOutput:
    def test_no_thinking(self):
        from rpg_agent.env.env import parse_thinking_output
        assert parse_thinking_output("hello world") == "hello world"

    def test_with_thinking_tags(self):
        from rpg_agent.env.env import parse_thinking_output
        text = "reasoning...<answer>the result</answer>...extra"
        assert parse_thinking_output(text, thinking=True) == "the result"

    def test_without_thinking_flag(self):
        from rpg_agent.env.env import parse_thinking_output
        text = "reasoning...<answer>the result</answer>...extra"
        result = parse_thinking_output(text, thinking=False)
        assert "<answer>" in result


# ============================================================================
# 2. Env construction and basic operations
# ============================================================================

class TestEnvConstruction:
    """Test Env class creation and basic APIs."""

    def test_env_creates_with_minimal_rpg(self, tmp_path):
        """Env can be constructed with a valid RPG and empty repo_dir."""
        from rpg_agent.env.env import Env
        from rpg_agent.tools import ALL_TOOLS

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test-instance",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=ALL_TOOLS,
            load_bm25=True,
        )
        assert env.instance_id == "test-instance"
        assert env.rpg is rpg
        assert env.entity_searcher is not None
        assert env.dep_searcher is not None
        assert env.step_count == 0
        assert env.final_results == []

    def test_env_file2code_loads_py_files(self, tmp_path):
        """Env._load_file2code reads .py files from repo_dir."""
        from rpg_agent.env.env import Env

        # Create sample files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")
        (src_dir / "readme.txt").write_text("not python")
        (tmp_path / "setup.py").write_text("setup()")

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=[],
            load_bm25=False,
        )
        assert "src/main.py" in env.file2code
        assert "setup.py" in env.file2code
        assert "src/readme.txt" not in env.file2code

    def test_env_file2code_skips_test_files(self, tmp_path):
        """Env._load_file2code excludes test files."""
        from rpg_agent.env.env import Env

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_x(): pass")
        (tmp_path / "src.py").write_text("x = 1")

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=[],
            load_bm25=False,
        )
        assert "src.py" in env.file2code
        assert "tests/test_main.py" not in env.file2code

    def test_env_reset(self, tmp_path):
        """Env.reset() clears all state."""
        from rpg_agent.env.env import Env

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=[],
            load_bm25=False,
        )
        env.step_count = 5
        env.action_history.append(None)
        env.feedback_history.append("test")
        env.final_results.append({"x": 1})

        env.reset()
        assert env.step_count == 0
        assert env.action_history == []
        assert env.feedback_history == []
        assert env.final_results == []
        assert env.last_action is None
        assert env.last_feedback is None

    def test_env_get_history_empty(self, tmp_path):
        """get_history returns empty list when no steps."""
        from rpg_agent.env.env import Env

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=[],
            load_bm25=False,
        )
        assert env.get_history() == []

    def test_env_get_last_action_info_initial(self, tmp_path):
        """get_last_action_info returns None fields initially."""
        from rpg_agent.env.env import Env

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=[],
            load_bm25=False,
        )
        info = env.get_last_action_info()
        assert info["step"] == 0
        assert info["action"] is None
        assert info["feedback"] is None


# ============================================================================
# 3. Tool custom_parse tests
# ============================================================================

class TestToolCustomParse:
    """Test custom_parse for all 6 tools."""

    def test_search_node_parse_feature(self):
        from rpg_agent.tools.search_node import SearchNode
        raw = json.dumps({
            "tool_name": "search_node",
            "parameters": {
                "feature_search": {
                    "feature_terms": ["authentication"],
                    "search_scopes": [],
                }
            }
        })
        result = SearchNode.custom_parse(raw)
        assert result is not None
        assert result["feature_search"]["feature_terms"] == ["authentication"]

    def test_search_node_parse_code(self):
        from rpg_agent.tools.search_node import SearchNode
        raw = json.dumps({
            "tool_name": "search_node",
            "parameters": {
                "code_search": {
                    "search_terms": ["src/auth.py:Auth"],
                }
            }
        })
        result = SearchNode.custom_parse(raw)
        assert result is not None
        assert result["code_search"]["search_terms"] == ["src/auth.py:Auth"]

    def test_search_node_parse_wrong_tool_name(self):
        from rpg_agent.tools.search_node import SearchNode
        raw = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {}
        })
        result = SearchNode.custom_parse(raw)
        assert result is None

    def test_search_code_by_features_parse(self):
        from rpg_agent.tools.search_code_by_features import SearchCodeByFeatures
        raw = json.dumps({
            "tool_name": "search_code_by_features",
            "parameters": {
                "feature_terms": ["error handling"],
            }
        })
        result = SearchCodeByFeatures.custom_parse(raw)
        assert result is not None
        assert result["feature_terms"] == ["error handling"]

    def test_search_code_by_features_coerce_string(self):
        from rpg_agent.tools.search_code_by_features import SearchCodeByFeatures
        raw = json.dumps({
            "tool_name": "search_code_by_features",
            "parameters": {
                "feature_terms": "single term",
            }
        })
        result = SearchCodeByFeatures.custom_parse(raw)
        assert result is not None
        assert result["feature_terms"] == ["single term"]

    def test_search_code_snippets_parse(self):
        from rpg_agent.tools.search_code_snippets import SearchCodeSnippets
        raw = json.dumps({
            "tool_name": "search_code_snippets",
            "parameters": {
                "search_terms": ["src/auth.py"],
                "line_nums": [1, 10],
            }
        })
        result = SearchCodeSnippets.custom_parse(raw)
        assert result is not None
        assert result["search_terms"] == ["src/auth.py"]
        assert result["line_nums"] == [1, 10]

    def test_fetch_node_parse(self):
        from rpg_agent.tools.fetch_node import FetchNode
        raw = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {
                "code_entities": ["src/auth.py"],
            }
        })
        result = FetchNode.custom_parse(raw)
        assert result is not None
        assert result["code_entities"] == ["src/auth.py"]

    def test_explore_rpg_parse(self):
        from rpg_agent.tools.explore_rpg import ExploreRPG
        raw = json.dumps({
            "tool_name": "explore_rpg_structure",
            "parameters": {
                "start_code_entities": ["src/auth.py:Auth"],
                "direction": "upstream",
                "traversal_depth": 3,
            }
        })
        result = ExploreRPG.custom_parse(raw)
        assert result is not None
        assert result["direction"] == "upstream"
        assert result["traversal_depth"] == 3

    def test_terminate_parse(self):
        from rpg_agent.tools.terminate import Terminate
        raw = json.dumps({
            "tool_name": "terminate",
            "parameters": {
                "results": [{
                    "file_path": "src/auth.py",
                    "func_name": "Auth.login",
                    "line_nums": [2, 2],
                }]
            }
        })
        result = Terminate.custom_parse(raw)
        assert result is not None
        assert len(result["results"]) == 1
        assert result["results"][0]["file_path"] == "src/auth.py"

    def test_terminate_parse_invalid_json(self):
        from rpg_agent.tools.terminate import Terminate
        result = Terminate.custom_parse("not json at all")
        assert result is None

    def test_parse_with_markdown_fences(self):
        """Markdown fences without language tag are handled."""
        from rpg_agent.tools.fetch_node import FetchNode
        raw = "```\n" + json.dumps({
            "tool_name": "fetch_node",
            "parameters": {"code_entities": ["a.py"]},
        }) + "\n```"
        result = FetchNode.custom_parse(raw)
        assert result is not None
        assert result["code_entities"] == ["a.py"]


# ============================================================================
# 4. Tool execute tests (async)
# ============================================================================

def _run_async(coro):
    """Helper to run async functions in sync test context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestToolExecute:
    """Test execute for each tool with mock environment."""

    def _make_env_dict(self):
        """Create a minimal env dict for tool execution."""
        rpg = _build_test_rpg()
        from rpg_agent.env.searcher import RepoEntitySearcher, RepoDependencySearcher
        entity_searcher = RepoEntitySearcher(rpg=rpg)
        dep_searcher = RepoDependencySearcher(rpg.dep_graph.G)
        return {
            "rpg": rpg,
            "file2code": {
                "src/auth.py": "class Auth:\n    def login(self): pass\n    def logout(self): pass\n",
                "src/utils.py": "def helper(): pass\n",
            },
            "entity_searcher": entity_searcher,
            "dep_searcher": dep_searcher,
            "bm25_retriever": None,
        }

    def test_search_node_execute_no_rpg(self):
        from rpg_agent.tools.search_node import SearchNode
        result = _run_async(SearchNode.execute(
            {"feature_search": {"feature_terms": ["auth"]}},
            env={},
        ))
        assert result.error_code == 1
        assert "RPG not available" in result.error

    def test_search_node_execute_no_params(self):
        from rpg_agent.tools.search_node import SearchNode
        env_dict = self._make_env_dict()
        result = _run_async(SearchNode.execute({}, env=env_dict))
        assert result.error_code == 1
        assert "No search parameters" in result.error

    def test_search_code_by_features_execute_no_terms(self):
        from rpg_agent.tools.search_code_by_features import SearchCodeByFeatures
        env_dict = self._make_env_dict()
        result = _run_async(SearchCodeByFeatures.execute(
            {"feature_terms": []}, env=env_dict,
        ))
        assert result.error_code == 1
        assert "No feature_terms" in result.error

    def test_search_code_snippets_execute_no_terms(self):
        from rpg_agent.tools.search_code_snippets import SearchCodeSnippets
        env_dict = self._make_env_dict()
        result = _run_async(SearchCodeSnippets.execute(
            {"search_terms": [], "line_nums": []}, env=env_dict,
        ))
        assert result.error_code == 1
        assert "No search_terms" in result.error

    def test_fetch_node_execute_no_rpg(self):
        from rpg_agent.tools.fetch_node import FetchNode
        result = _run_async(FetchNode.execute(
            {"code_entities": ["a.py"]}, env={},
        ))
        assert result.error_code == 1

    def test_explore_rpg_execute_no_rpg(self):
        from rpg_agent.tools.explore_rpg import ExploreRPG
        result = _run_async(ExploreRPG.execute(
            {"start_code_entities": ["a.py"]}, env={},
        ))
        assert result.error_code == 1

    def test_terminate_execute_empty_results(self):
        from rpg_agent.tools.terminate import Terminate
        env_dict = self._make_env_dict()
        result = _run_async(Terminate.execute(
            {"results": []}, env=env_dict,
        ))
        assert result.error_code == 1
        assert "results" in result.error.lower()

    def test_fetch_node_execute_with_valid_entity(self):
        from rpg_agent.tools.fetch_node import FetchNode
        env_dict = self._make_env_dict()
        result = _run_async(FetchNode.execute(
            {"code_entities": ["src/auth.py"], "feature_entities": []},
            env=env_dict,
        ))
        assert result.error_code == 0
        assert result.output is not None

    def test_explore_rpg_execute_with_valid_entity(self):
        from rpg_agent.tools.explore_rpg import ExploreRPG
        env_dict = self._make_env_dict()
        result = _run_async(ExploreRPG.execute(
            {
                "start_code_entities": ["src/auth.py:Auth"],
                "direction": "downstream",
                "traversal_depth": 1,
            },
            env=env_dict,
        ))
        assert result.error_code == 0
        assert result.output is not None

    def test_terminate_execute_valid_result(self):
        """Terminate with a valid entity that exists in the graph."""
        from rpg_agent.tools.terminate import Terminate

        rpg = _build_test_rpg()
        from rpg_agent.env.searcher import RepoEntitySearcher
        entity_searcher = RepoEntitySearcher(rpg=rpg)

        class MockEnv:
            final_results = []
            step_count = 5
        mock_env = MockEnv()

        env_dict = {
            "rpg": rpg,
            "entity_searcher": entity_searcher,
            "environment": mock_env,
        }
        result = _run_async(Terminate.execute(
            {"results": [{
                "file_path": "src/auth.py",
                "func_name": "Auth.login",
                "line_nums": [2, 2],
            }]},
            env=env_dict,
        ))
        assert result.error_code == 0
        assert mock_env.final_results


# ============================================================================
# 5. Tool registration and ToolHandler/ToolExecutor integration
# ============================================================================

class TestToolRegistration:
    """Test that all tools register correctly with ToolHandler/ToolExecutor."""

    def test_all_tools_register(self):
        from rpg_agent.tools import ALL_TOOLS
        handler = ToolHandler(tools=ALL_TOOLS)
        executor = ToolExecutor(tools=ALL_TOOLS)

        names = handler.list_registered()
        assert len(names) == 6
        assert "search_node" in names
        assert "search_code_by_features" in names
        assert "search_code_snippets" in names
        assert "fetch_node" in names
        assert "explore_rpg_structure" in names
        assert "terminate" in names

        executor_names = executor.list_tools()
        assert len(executor_names) == 6

    def test_tool_handler_parses_search_node(self):
        from rpg_agent.tools import ALL_TOOLS
        handler = ToolHandler(tools=ALL_TOOLS)
        llm_output = json.dumps({
            "tool_name": "search_node",
            "parameters": {
                "feature_search": {"feature_terms": ["auth"]},
            }
        })
        calls = handler.parse_and_match_tool(llm_output)
        assert len(calls) >= 1
        assert calls[0].name == "search_node"

    def test_tool_handler_parses_fetch_node(self):
        from rpg_agent.tools import ALL_TOOLS
        handler = ToolHandler(tools=ALL_TOOLS)
        llm_output = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {
                "code_entities": ["src/auth.py"],
            }
        })
        calls = handler.parse_and_match_tool(llm_output)
        assert len(calls) >= 1
        assert calls[0].name == "fetch_node"

    def test_tool_handler_no_match(self):
        from rpg_agent.tools import ALL_TOOLS
        handler = ToolHandler(tools=ALL_TOOLS)
        calls = handler.parse_and_match_tool("random text with no JSON")
        assert len(calls) == 0

    def test_describe_registered_tools(self):
        from rpg_agent.tools import ALL_TOOLS
        handler = ToolHandler(tools=ALL_TOOLS)
        desc = handler.describe_registered_tools()
        assert "search_node" in desc
        assert "fetch_node" in desc
        assert "terminate" in desc


# ============================================================================
# 6. Env.step integration test
# ============================================================================

class TestEnvStep:
    """Test Env.step with a registered tool."""

    def test_step_no_valid_tool(self, tmp_path):
        """step() with unrecognizable input."""
        from rpg_agent.env.env import Env
        from rpg_agent.tools import ALL_TOOLS

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=ALL_TOOLS,
            load_bm25=False,
        )
        feedback, success, is_terminate = env.step("random gibberish")
        assert not success
        assert not is_terminate
        assert "No valid tool action" in feedback
        assert env.step_count == 1

    def test_step_valid_fetch_node(self, tmp_path):
        """step() with a valid fetch_node call."""
        from rpg_agent.env.env import Env
        from rpg_agent.tools import ALL_TOOLS

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=ALL_TOOLS,
            load_bm25=False,
        )
        llm_output = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {
                "code_entities": ["src/auth.py"],
            }
        })
        feedback, success, is_terminate = env.step(llm_output)
        assert success
        assert not is_terminate
        assert env.step_count == 1
        assert "fetch_node" in feedback

    def test_step_duplicate_action(self, tmp_path):
        """step() rejects identical consecutive actions."""
        from rpg_agent.env.env import Env
        from rpg_agent.tools import ALL_TOOLS

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=ALL_TOOLS,
            load_bm25=False,
        )
        llm_output = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {
                "code_entities": ["src/auth.py"],
            }
        })
        # First call
        env.step(llm_output)
        # Second identical call
        feedback, success, is_terminate = env.step(llm_output)
        assert not success
        assert "identical" in feedback.lower()

    def test_step_history_tracking(self, tmp_path):
        """step() records history correctly."""
        from rpg_agent.env.env import Env
        from rpg_agent.tools import ALL_TOOLS

        rpg = _build_test_rpg()
        env = Env(
            instance_id="test",
            repo_dir=str(tmp_path),
            rpg=rpg,
            register_tools=ALL_TOOLS,
            load_bm25=False,
        )
        llm_output = json.dumps({
            "tool_name": "fetch_node",
            "parameters": {"code_entities": ["src/auth.py"]},
        })
        env.step(llm_output)

        history = env.get_history()
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["action"] == "fetch_node"

        info = env.get_last_action_info()
        assert info["action"] == "fetch_node"
        assert info["step"] == 1


# ============================================================================
# 7. Package exports
# ============================================================================

class TestExports:
    """Verify env and tools package exports."""

    def test_env_package_exports(self):
        from rpg_agent.env import (
            Env,
            QueryInfo,
            QueryResult,
            RepoDependencySearcher,
            RepoEntitySearcher,
        )
        assert Env is not None
        assert QueryInfo is not None
        assert QueryResult is not None

    def test_tools_package_exports(self):
        from rpg_agent.tools import (
            ALL_TOOLS,
            ExploreRPG,
            FetchNode,
            SearchCodeByFeatures,
            SearchCodeSnippets,
            SearchNode,
            Terminate,
        )
        assert len(ALL_TOOLS) == 6
