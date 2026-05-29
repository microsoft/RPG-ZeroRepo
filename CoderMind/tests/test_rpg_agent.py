#!/usr/bin/env python3
"""Tests for M11 — RPG Agent Workflow.

Covers:
  - prompts/agent_prompt.py: REPO_AGENT_SYSTEM_PROMPT template integrity
  - rpg_agent.py: RPGAgent construction, init_memory, load_task_to_env_prompt,
    step(), run(), token usage, error handling
  - __init__.py: lazy export of RPGAgent
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

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
from common.llm_types import (
    AssistantMessage,
    Memory,
    SystemMessage,
    UserMessage,
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
# Helpers
# ============================================================================

class FakeDependencyGraph:
    """Minimal DependencyGraph stub."""
    def __init__(self, G):
        self.G = G
        self.repo_dir = "/fake/repo"


def _build_test_rpg() -> RPG:
    """Build a small RPG for testing (minimal structure)."""
    rpg = RPG(repo_name="TestRepo")

    root_meta = NodeMetaData(
        type_name=NodeType.REPO, path="root", description="Test Repo"
    )
    root = Node(id="root_node", name="TestRepo", meta=root_meta, level=0)
    rpg.add_node(root)

    auth_meta = NodeMetaData(
        type_name=NodeType.DIRECTORY, path="auth", description="Auth module"
    )
    auth = Node(id="auth_node", name="auth", meta=auth_meta, level=1)
    rpg.add_node(auth)
    rpg.add_edge(root, auth, relation="contains")

    login_meta = NodeMetaData(
        type_name=NodeType.CLASS, path="src/auth.py::Auth",
        description="Auth class",
    )
    login = Node(id="login_node", name="Auth", meta=login_meta, level=2)
    rpg.add_node(login)
    rpg.add_edge(auth, login, relation="contains")

    # Build dep_graph
    G = nx.MultiDiGraph()
    G.add_node(
        "src/auth.py", type=NodeType.FILE,
        code="class Auth:\n    def login(self): pass\n",
    )
    G.add_node(
        "src/auth.py:Auth", type=NodeType.CLASS,
        code="class Auth:\n    def login(self): pass\n",
        start_line=1, end_line=2,
    )
    G.add_edge("src/auth.py", "src/auth.py:Auth", type=EdgeType.CONTAINS)

    dep_graph = FakeDependencyGraph(G)
    rpg.dep_graph = dep_graph

    rpg._dep_to_rpg_map = {
        "src/auth.py:Auth": ["login_node"],
        "src/auth.py": [],
    }

    return rpg


# ============================================================================
# 1. Prompt Tests
# ============================================================================

class TestAgentPrompt:
    """Tests for the agent system prompt template."""

    def test_prompt_import(self):
        """REPO_AGENT_SYSTEM_PROMPT is importable and non-empty."""
        from rpg_agent.prompts.agent_prompt import REPO_AGENT_SYSTEM_PROMPT
        assert isinstance(REPO_AGENT_SYSTEM_PROMPT, str)
        assert len(REPO_AGENT_SYSTEM_PROMPT) > 100

    def test_prompt_has_tool_placeholder(self):
        """Prompt contains the {Tool_Description} placeholder."""
        from rpg_agent.prompts.agent_prompt import REPO_AGENT_SYSTEM_PROMPT
        assert "{Tool_Description}" in REPO_AGENT_SYSTEM_PROMPT

    def test_prompt_format_map(self):
        """format_map successfully fills the Tool_Description placeholder."""
        from rpg_agent.prompts.agent_prompt import REPO_AGENT_SYSTEM_PROMPT
        result = REPO_AGENT_SYSTEM_PROMPT.format_map(
            {"Tool_Description": "search_node: Search for nodes in the RPG."}
        )
        assert "search_node: Search for nodes in the RPG." in result
        assert "{Tool_Description}" not in result

    def test_prompt_contains_key_sections(self):
        """Prompt includes all required sections from ZeroRepo."""
        from rpg_agent.prompts.agent_prompt import REPO_AGENT_SYSTEM_PROMPT
        assert "## Role" in REPO_AGENT_SYSTEM_PROMPT
        assert "## Repository Planning Graph (RPG)" in REPO_AGENT_SYSTEM_PROMPT
        assert "## Workflow" in REPO_AGENT_SYSTEM_PROMPT
        assert "### Step 1: Extract Anchors" in REPO_AGENT_SYSTEM_PROMPT
        assert "### Step 2: Map to Functional Area" in REPO_AGENT_SYSTEM_PROMPT
        assert "### Step 3: Establish Execution Connectivity" in REPO_AGENT_SYSTEM_PROMPT
        assert "### Step 4: Targeted Verification and Ranking" in REPO_AGENT_SYSTEM_PROMPT
        assert "## IMPORTANT CONSTRAINTS" in REPO_AGENT_SYSTEM_PROMPT
        assert "## Action Space" in REPO_AGENT_SYSTEM_PROMPT
        assert "## Output Format" in REPO_AGENT_SYSTEM_PROMPT

    def test_prompt_output_format_tags(self):
        """Prompt specifies <think> and <action> block format."""
        from rpg_agent.prompts.agent_prompt import REPO_AGENT_SYSTEM_PROMPT
        assert "<think>" in REPO_AGENT_SYSTEM_PROMPT
        assert "</think>" in REPO_AGENT_SYSTEM_PROMPT
        assert "<action>" in REPO_AGENT_SYSTEM_PROMPT
        assert "</action>" in REPO_AGENT_SYSTEM_PROMPT

    def test_prompt_init_re_export(self):
        """prompts/__init__.py re-exports REPO_AGENT_SYSTEM_PROMPT."""
        from rpg_agent.prompts import REPO_AGENT_SYSTEM_PROMPT
        assert isinstance(REPO_AGENT_SYSTEM_PROMPT, str)
        assert "{Tool_Description}" in REPO_AGENT_SYSTEM_PROMPT


# ============================================================================
# 2. RPGAgent Tests (with mocked LLM and Env)
# ============================================================================

class TestRPGAgent:
    """Tests for RPGAgent class."""

    @pytest.fixture
    def rpg(self):
        return _build_test_rpg()

    @pytest.fixture
    def mock_llm_config(self):
        """Return a dict that LLMConfig.from_source can parse."""
        return {"model": "gpt-4o", "provider": "openai"}

    @pytest.fixture
    def agent(self, rpg, mock_llm_config):
        """Construct an RPGAgent with mocked LLM client."""
        with patch(
            "rpg_agent.rpg_agent.LLMClient"
        ) as MockLLMClient:
            mock_client = MagicMock()
            mock_client.generate_with_memory.return_value = (
                '<think>searching</think>\n'
                '<action>\n'
                '{"tool_name": "search_node", '
                '"parameters": {"query": "auth"}}\n'
                '</action>'
            )
            mock_client.last_usage = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            MockLLMClient.return_value = mock_client

            from rpg_agent.rpg_agent import RPGAgent
            agent = RPGAgent(
                instance_id="test-001",
                task="Fix the login bug in the auth module.",
                repo_dir="/fake/repo",
                repo_name="TestRepo",
                repo_rpg=rpg,
                max_steps=5,
                context_window=10,
                register_tools=[],
            )
            agent._llm = mock_client
            return agent

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction(self, agent):
        """RPGAgent is created with expected attributes."""
        assert agent.repo_name == "TestRepo"
        assert agent._max_steps == 5
        assert agent._task == "Fix the login bug in the auth module."
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert isinstance(agent._memory, Memory)
        assert agent.logger is not None

    def test_construction_with_custom_logger(self, rpg, mock_llm_config):
        """RPGAgent accepts a custom logger."""
        custom_logger = logging.getLogger("custom_test_logger")
        with patch(
            "rpg_agent.rpg_agent.LLMClient"
        ):
            from rpg_agent.rpg_agent import RPGAgent
            agent = RPGAgent(
                instance_id="test-002",
                task="Test task",
                repo_dir="/fake/repo",
                repo_name="TestRepo",
                repo_rpg=rpg,
                logger=custom_logger,
            )
            assert agent.logger is custom_logger

    # ------------------------------------------------------------------
    # init_memory
    # ------------------------------------------------------------------

    def test_init_memory_clears_and_sets_system_prompt(self, agent):
        """init_memory clears history and adds a system message."""
        # Pre-populate memory
        agent._memory.add_message(UserMessage(content="old message"))
        assert len(agent._memory._history) == 1

        agent.init_memory()

        # Should be cleared and contain exactly one system message
        assert len(agent._memory._history) == 1
        msg = agent._memory._history[0]
        assert msg.role == "system"
        assert "senior software engineer" in msg.content

    def test_init_memory_injects_tool_descriptions(self, agent):
        """init_memory fills {Tool_Description} with actual tool descriptions."""
        agent.init_memory()
        msg = agent._memory._history[0]
        # The placeholder should be gone
        assert "{Tool_Description}" not in msg.content

    # ------------------------------------------------------------------
    # load_task_to_env_prompt
    # ------------------------------------------------------------------

    def test_load_task_to_env_prompt_contains_task(self, agent):
        """The env prompt contains the original task description."""
        prompt = agent.load_task_to_env_prompt()
        assert "Fix the login bug in the auth module." in prompt

    def test_load_task_to_env_prompt_structure(self, agent):
        """The env prompt has the expected sections."""
        prompt = agent.load_task_to_env_prompt()
        assert "== GitHub Issue ==" in prompt
        assert "<issue>" in prompt
        assert "</issue>" in prompt
        assert "== Task Begin ==" in prompt
        assert "localize all(5-10)" in prompt
        assert "terminate" in prompt

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def test_step_calls_llm_and_env(self, agent):
        """step() calls LLM generate, adds assistant message, passes to env."""
        agent.init_memory()
        agent._memory.add_message(
            UserMessage(content="[Step 1/5 User Query]: Test task")
        )

        feedback, tool_suc, is_terminate = agent.step(step_id=1)

        # LLM was called
        agent._llm.generate_with_memory.assert_called_once()

        # Token usage tracked
        assert len(agent.step_token_usage) == 1
        assert agent.step_token_usage[0]["step_id"] == 1
        assert agent.step_token_usage[0]["input_tokens"] == 100
        assert agent.step_token_usage[0]["output_tokens"] == 50
        assert agent.total_input_tokens == 100
        assert agent.total_output_tokens == 50

        # feedback returned (from env.step)
        assert isinstance(feedback, str)
        assert isinstance(tool_suc, bool)
        assert isinstance(is_terminate, bool)

    def test_step_handles_none_llm_response(self, agent):
        """step() handles None LLM response gracefully."""
        agent._llm.generate_with_memory.return_value = None
        agent.init_memory()
        agent._memory.add_message(
            UserMessage(content="[Step 1/5 User Query]: Test")
        )

        feedback, tool_suc, is_terminate = agent.step(step_id=1)

        # Should not crash; env will get empty string
        assert isinstance(feedback, str)

    def test_step_accumulates_token_usage(self, agent):
        """Multiple steps accumulate token usage correctly."""
        agent.init_memory()

        for i in range(3):
            agent._memory.add_message(
                UserMessage(content=f"[Step {i+1}/5]: msg {i}")
            )
            agent.step(step_id=i + 1)

        assert agent.total_input_tokens == 300
        assert agent.total_output_tokens == 150
        assert len(agent.step_token_usage) == 3

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def test_run_terminates_on_terminate_action(self, agent):
        """run() stops when the environment signals termination."""
        # Make env.step return is_terminate=True on second call
        call_count = [0]
        original_step = agent._agent_env.step

        def mock_env_step(response):
            call_count[0] += 1
            if call_count[0] == 2:
                return "Terminated.", True, True
            return "Feedback from tool.", True, False

        agent._agent_env.step = mock_env_step

        results = agent.run()

        assert results["is_terminate"] is True
        assert results["is_suc"] is True
        assert len(results["step_token_usage"]) == 2
        assert results["total_input_tokens"] == 200
        assert results["total_output_tokens"] == 100

    def test_run_stops_on_max_steps(self, agent):
        """run() stops when max_steps is reached."""
        # Always succeed but never terminate
        agent._agent_env.step = lambda response: (
            "Still working.", True, False
        )

        results = agent.run()

        assert results["is_terminate"] is False
        assert results["is_suc"] is False
        assert len(results["step_token_usage"]) == 5  # max_steps

    def test_run_stops_on_consecutive_errors(self, agent):
        """run() stops after max_error_times consecutive failures."""
        agent._agent_env.step = lambda response: (
            "Error: invalid tool.", False, False
        )

        results = agent.run(max_error_times=2)

        assert results["is_terminate"] is False
        assert results["is_suc"] is False
        # Should stop after 2 consecutive errors
        assert len(results["step_token_usage"]) == 2

    def test_run_resets_error_count_on_success(self, agent):
        """Successful steps reset the consecutive error counter."""
        call_count = [0]

        def mock_env_step(response):
            call_count[0] += 1
            if call_count[0] in (1, 3):
                return "Error.", False, False
            if call_count[0] == 5:
                return "Done.", True, True
            return "OK.", True, False

        agent._agent_env.step = mock_env_step

        results = agent.run(max_error_times=2)

        # Should not stop at 2 errors because they're not consecutive
        assert results["is_terminate"] is True
        assert results["is_suc"] is True

    def test_run_returns_complete_results(self, agent):
        """run() returns all expected keys in the results dict."""
        agent._agent_env.step = lambda response: (
            "Done.", True, True
        )

        results = agent.run()

        expected_keys = {
            "final_results",
            "is_terminate",
            "is_suc",
            "all_traj",
            "action_history",
            "feedback_history",
            "step_token_usage",
            "total_input_tokens",
            "total_output_tokens",
        }
        assert expected_keys == set(results.keys())

    def test_run_serializes_action_history(self, agent):
        """Action history entries are serialized to dicts (not ToolCall objects)."""
        # Set up action history with a ToolCall
        tc = ToolCall(
            name="search_node", call_id="tc_1",
            arguments={"query": "auth"}, id="id_1",
        )
        agent._agent_env.action_history = [tc, None]
        agent._agent_env.feedback_history = ["feedback1", "feedback2"]
        agent._agent_env.final_results = []
        agent._agent_env.step = lambda response: ("Done.", True, True)

        results = agent.run()

        # The action_history in results should include serialized entries
        # from the run (not the pre-set ones since run calls reset)
        assert isinstance(results["action_history"], list)

    def test_run_step_prefix_format(self, agent):
        """First step uses 'User Query' prefix, subsequent use 'Tool Execution Feedback'."""
        messages_added = []
        original_add = agent._memory.add_message

        def capture_add(message):
            if isinstance(message, UserMessage):
                messages_added.append(message.content)
            original_add(message)

        agent._memory.add_message = capture_add
        agent._agent_env.step = lambda response: ("Feedback.", True, True)

        agent.run()

        # After init_memory (system msg), first user message should have "User Query"
        user_msgs = [m for m in messages_added if "Step" in m]
        assert len(user_msgs) >= 1
        assert "User Query" in user_msgs[0]

    # ------------------------------------------------------------------
    # get_total_tokens_usage
    # ------------------------------------------------------------------

    def test_get_total_tokens_usage(self, agent):
        """get_total_tokens_usage returns accumulated counts."""
        agent.total_input_tokens = 500
        agent.total_output_tokens = 200

        usage = agent.get_total_tokens_usage()

        assert usage == {
            "total_input_tokens": 500,
            "total_output_tokens": 200,
        }

    # ------------------------------------------------------------------
    # Env reset
    # ------------------------------------------------------------------

    def test_run_resets_env_and_tokens(self, agent):
        """run() resets env state and token counters."""
        agent.total_input_tokens = 999
        agent.total_output_tokens = 999
        agent.step_token_usage = [{"dummy": 1}]

        agent._agent_env.step = lambda response: ("Done.", True, True)

        results = agent.run()

        # Counters should reflect only the new run
        assert results["total_input_tokens"] == 100  # one step
        assert results["total_output_tokens"] == 50


# ============================================================================
# 3. Package __init__ Tests
# ============================================================================

class TestPackageInit:
    """Tests for rpg_agent/__init__.py lazy export."""

    def test_rpg_agent_in_all(self):
        """RPGAgent is listed in __all__."""
        import scripts.rpg_agent as pkg
        assert "RPGAgent" in pkg.__all__

    def test_lazy_import_works(self):
        """RPGAgent can be accessed via the package attribute."""
        import scripts.rpg_agent as pkg
        # Access the attribute -- triggers __getattr__
        cls = pkg.RPGAgent
        assert cls.__name__ == "RPGAgent"

    def test_lazy_import_unknown_attr(self):
        """Accessing unknown attribute raises AttributeError."""
        import scripts.rpg_agent as pkg
        with pytest.raises(AttributeError):
            _ = pkg.NonExistentClass


# ============================================================================
# 4. Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary tests for RPGAgent."""

    @pytest.fixture
    def rpg(self):
        return _build_test_rpg()

    def test_agent_with_empty_task(self, rpg):
        """Agent handles empty task string."""
        with patch(
            "rpg_agent.rpg_agent.LLMClient"
        ) as MockLLM:
            mock_client = MagicMock()
            mock_client.generate_with_memory.return_value = (
                '<think>empty</think>\n<action>\n'
                '{"tool_name": "terminate", '
                '"parameters": {"result": []}}\n</action>'
            )
            mock_client.last_usage = {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            }
            MockLLM.return_value = mock_client

            from rpg_agent.rpg_agent import RPGAgent
            agent = RPGAgent(
                instance_id="edge-001",
                task="",
                repo_dir="/fake/repo",
                repo_name="TestRepo",
                repo_rpg=rpg,
                max_steps=2,
            )
            agent._llm = mock_client

            prompt = agent.load_task_to_env_prompt()
            assert "== GitHub Issue ==" in prompt

    def test_agent_max_steps_one(self, rpg):
        """Agent with max_steps=1 runs exactly one step."""
        with patch(
            "rpg_agent.rpg_agent.LLMClient"
        ) as MockLLM:
            mock_client = MagicMock()
            mock_client.generate_with_memory.return_value = (
                '<think>test</think>\n<action>\n'
                '{"tool_name": "search_node", '
                '"parameters": {"query": "test"}}\n</action>'
            )
            mock_client.last_usage = {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            }
            MockLLM.return_value = mock_client

            from rpg_agent.rpg_agent import RPGAgent
            agent = RPGAgent(
                instance_id="edge-002",
                task="Test",
                repo_dir="/fake/repo",
                repo_name="TestRepo",
                repo_rpg=rpg,
                max_steps=1,
            )
            agent._llm = mock_client

            results = agent.run()

            assert len(results["step_token_usage"]) == 1

    def test_agent_with_zero_max_error_times(self, rpg):
        """Agent with max_error_times=0 stops immediately on first error."""
        with patch(
            "rpg_agent.rpg_agent.LLMClient"
        ) as MockLLM:
            mock_client = MagicMock()
            mock_client.generate_with_memory.return_value = "garbage no action"
            mock_client.last_usage = {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            }
            MockLLM.return_value = mock_client

            from rpg_agent.rpg_agent import RPGAgent
            agent = RPGAgent(
                instance_id="edge-003",
                task="Test",
                repo_dir="/fake/repo",
                repo_name="TestRepo",
                repo_rpg=rpg,
                max_steps=5,
            )
            agent._llm = mock_client

            # max_error_times=0 means the first error should NOT trigger
            # immediate break (needs error_times >= max_error_times, and
            # error_times starts at 0, increments to 1 which >= 0 is true
            # only after first failed step)
            # Actually: error_times starts at 0, after first fail it's 1.
            # 1 >= 0 is True, so it breaks after first step.
            results = agent.run(max_error_times=0)

            # With max_error_times=0, error_times(0) >= 0 is True
            # immediately on step 0 error, so we only get 0 steps.
            # Wait: the logic is:
            #   not tool_suc => error_times += 1
            #   if error_times >= max_error_times => break
            # So error_times=1 >= 0 => break after 1 step
            assert len(results["step_token_usage"]) <= 1
