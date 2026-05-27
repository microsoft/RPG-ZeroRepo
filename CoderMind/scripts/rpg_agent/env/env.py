#!/usr/bin/env python3
"""RPG Agent Environment — orchestrates tool execution and state tracking.

The ``Env`` class holds references to the RPG, searchers, and tool
handler/executor.  Its ``step`` method parses an LLM response, matches
the output against registered tools, executes the matched tool, and
returns structured feedback.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/env/env.py
Adaptations:
  - Replaced ``load_skeleton_from_repo`` / ``filter_non_test_py_files``
    with repo-dir walking that builds a ``file2code`` dict (avoids
    importing the full RepoSkeleton build infra here).
  - Replaced llama_index BM25 with RPG-Kit's lightweight
    ``ModuleRetriever`` from ``bm25_model.py``.
  - Uses RPG-Kit imports (``scripts.common.tools``, ``scripts.rpg_agent``).
  - ``parse_thinking_output`` inlined (tag-stripping helper).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from common.tools import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolHandler,
    ToolResult,
)
from common.utils import is_test_file
from rpg_agent.env.searcher import RepoDependencySearcher, RepoEntitySearcher
from rpg_agent.ops.bm25_model import ModuleRetriever, build_module_retriever
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: extract answer from thinking-model output
# ============================================================================

# Tag constants matching RPG-ZeroRepo/zerorepo/utils/envs.py
_ANSWER_START_TAG = "<answer>"
_ANSWER_END_TAG = "</answer>"


def parse_thinking_output(output: str, thinking: bool = False) -> str:
    """Strip thinking-model wrapper tags if *thinking* mode is on.

    When *thinking* is ``True``, extracts the content between
    ``<answer>...</answer>`` tags.  Otherwise the output is returned
    as-is (stripped).

    Source: RPG-ZeroRepo/zerorepo/utils/api.py ``parse_thinking_output``
    """
    if thinking:
        output = output.split(_ANSWER_START_TAG, 1)[-1]
        output = output.split(_ANSWER_END_TAG, 1)[0]
    return output.strip()


# ============================================================================
# Env
# ============================================================================

class Env:
    """Agent execution environment — manages searchers, tools, and state.

    Args:
        instance_id: Unique identifier for this environment instance.
        repo_dir: Path to the target repository root.
        rpg: Pre-loaded RPG instance.
        register_tools: List of ``Tool`` classes to register.
        load_bm25: Whether to build a BM25 retriever on construction.

    Source: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/env/env.py ``Env``
    """

    def __init__(
        self,
        instance_id: str,
        repo_dir: str,
        rpg: RPG,
        register_tools: List[type[Tool]],
        load_bm25: bool = True,
    ):
        self.instance_id = instance_id
        self.repo_dir = repo_dir
        self.rpg = rpg

        # Build file2code map by walking repo_dir
        self.file2code: Dict[str, str] = self._load_file2code()

        # Tool infrastructure (from M3)
        self.tool_handler = ToolHandler(tools=register_tools)
        self.tool_executor = ToolExecutor(tools=register_tools)

        # Local environment memory
        self.last_action: Optional[ToolCall] = None
        self.last_action_result: Optional[ToolResult] = None
        self.last_feedback: Optional[str] = None
        self.step_count: int = 0

        # Full interaction history
        self.action_history: List[Optional[ToolCall]] = []
        self.feedback_history: List[str] = []

        # Initialize searchers from RPG
        self.entity_searcher: Optional[RepoEntitySearcher] = (
            RepoEntitySearcher(rpg=rpg) if rpg else None
        )
        self.dep_searcher: Optional[RepoDependencySearcher] = (
            RepoDependencySearcher.from_rpg(rpg) if rpg and rpg.dep_graph else None
        )

        # BM25 module retriever (lightweight rank_bm25 implementation)
        self.bm25_retriever: Optional[ModuleRetriever] = None
        if load_bm25:
            self._load_bm25_retriever()

        # Terminate results storage
        self.final_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_file2code(self) -> Dict[str, str]:
        """Walk ``repo_dir`` and build a mapping of relative path -> code.

        Only includes non-test ``.py`` files.
        """
        file2code: Dict[str, str] = {}
        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            return file2code

        for root, _dirs, files in os.walk(self.repo_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, self.repo_dir).replace("\\", "/")
                if is_test_file(rel_path):
                    continue
                try:
                    with open(abs_path, "r", encoding="utf-8") as fh:
                        file2code[rel_path] = fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
        return file2code

    def _load_bm25_retriever(self) -> None:
        """Build a BM25 retriever from the RPG dependency graph."""
        try:
            if self.rpg and self.rpg.dep_graph:
                logger.info("Building BM25 module retriever from RPG dep_graph...")
                self.bm25_retriever = build_module_retriever(
                    rpg=self.rpg,
                    similarity_top_k=10,
                )
                logger.info("BM25 module retriever built successfully.")
            else:
                logger.warning("No dep_graph available; BM25 retriever not built.")
        except Exception as exc:
            logger.warning("Failed to build BM25 retriever: %s", exc)
            self.bm25_retriever = None

    # ==================================================================
    # Core step logic
    # ==================================================================

    def step(self, response: str) -> tuple:
        """Parse an LLM response, execute the matched tool, return feedback.

        Keeps track of action history.  If the current action is identical
        to the previous one, asks the model to revise.

        Args:
            response: Raw LLM text output.

        Returns:
            Tuple of ``(feedback, tool_success, is_terminate)`` where
            *feedback* is a human-readable string, *tool_success* is a bool,
            and *is_terminate* indicates whether the agent chose to stop.
        """
        parsed_response = parse_thinking_output(output=response)
        try:
            parsed_action_list = self.tool_handler.parse_and_match_tool(
                llm_output=parsed_response,
            )
            parsed_action = parsed_action_list[0] if parsed_action_list else None

            self.step_count += 1

            # --- duplicate action guard ---
            if (
                parsed_action
                and self.last_action
                and parsed_action.name == self.last_action.name
                and parsed_action.arguments == self.last_action.arguments
            ):
                feedback = (
                    "The current tool call is identical to your previous one. "
                    "Please revise your reasoning and propose a different tool action "
                    "or modify the arguments to move the task forward."
                )
                self.last_feedback = feedback
                self.action_history.append(parsed_action)
                self.feedback_history.append(feedback)
                return feedback, False, False

            self.last_action = parsed_action

            # --- no valid action ---
            if not parsed_action:
                feedback = (
                    "No valid tool action was detected from your previous output. "
                    "Please specify the tool you want to use and provide arguments "
                    "in a clear format.\n\n"
                    f"Available tools: {self.tool_handler.list_registered()}\n"
                    "Make sure to output only the tool call, without extra explanation."
                )
                self.last_feedback = feedback
                self.action_history.append(None)
                self.feedback_history.append(feedback)
                return feedback, False, False

            # --- execute the tool ---
            env_param: Dict[str, Any] = {
                "environment": self,
                "rpg": self.rpg,
                "file2code": self.file2code,
                "entity_searcher": self.entity_searcher,
                "dep_searcher": self.dep_searcher,
                "bm25_retriever": self.bm25_retriever,
            }
            result: ToolResult = asyncio.run(
                self.tool_executor.execute_tool_call(parsed_action, env=env_param),
            )
            self.last_action_result = result
            tool_suc = result.success

            # --- build feedback ---
            if result.success:
                feedback = (
                    f"Tool '{result.name}' executed successfully.\n"
                    f"Output: {result.result}"
                )
            else:
                feedback = (
                    f"Tool '{result.name}' execution failed.\n"
                    f"Error: {result.error}"
                )

            self.last_feedback = feedback
            self.action_history.append(parsed_action)
            self.feedback_history.append(feedback)

            is_terminate = (
                parsed_action.name.lower() == "terminate" and tool_suc
            )
            return feedback, tool_suc, is_terminate

        except Exception as exc:
            feedback = (
                f"An error occurred while parsing or executing the tool: {exc}\n"
                "Please reformat your response and specify the correct tool name "
                "and arguments."
            )
            self.last_feedback = feedback
            self.action_history.append(None)
            self.feedback_history.append(feedback)
            return feedback, False, False

    # ==================================================================
    # State management
    # ==================================================================

    def reset(self) -> None:
        """Reset the environment memory (between agent runs)."""
        self.last_action = None
        self.last_action_result = None
        self.last_feedback = None
        self.step_count = 0
        self.action_history.clear()
        self.feedback_history.clear()
        self.final_results.clear()

    # ==================================================================
    # Accessors
    # ==================================================================

    def get_last_action_info(self) -> Dict[str, Any]:
        """Return summary info about the last tool execution."""
        return {
            "step": self.step_count,
            "action": self.last_action.name if self.last_action else None,
            "arguments": self.last_action.arguments if self.last_action else None,
            "success": (
                self.last_action_result.success
                if self.last_action_result
                else None
            ),
            "output": (
                self.last_action_result.result
                if self.last_action_result
                else None
            ),
            "error": (
                self.last_action_result.error
                if self.last_action_result
                else None
            ),
            "feedback": self.last_feedback,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full history of all steps (action + feedback)."""
        history: List[Dict[str, Any]] = []
        for idx, (action, feedback) in enumerate(
            zip(self.action_history, self.feedback_history), start=1
        ):
            history.append({
                "step": idx,
                "action": action.name if action else None,
                "arguments": action.arguments if action else None,
                "feedback": feedback,
            })
        return history
