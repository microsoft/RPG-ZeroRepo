#!/usr/bin/env python3
"""RPG Agent — main agent class for RPG-based code localization.

The ``RPGAgent`` class orchestrates the LLM reasoning loop:
it sends conversational context to the LLM, parses the response for
tool calls, delegates execution to the ``Env`` environment, and
feeds the result back into the next turn.

Ported from:
    RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/rpg_agent.py

Adaptations for CoderMind:
  - Uses ``LLMClient`` (``scripts.common.llm_client``) for CLI-based generation.
  - Uses ``Memory``, ``SystemMessage``, ``UserMessage``, ``AssistantMessage``
    from ``scripts.common.llm_types``.
  - Uses ``Env`` from ``scripts.rpg_agent.env.env`` (no ``persist_dir`` param).
  - Token usage keys aligned with CoderMind's ``LLMUsage.to_dict()`` output
    (``input_tokens`` / ``output_tokens`` / ``total_tokens``).
  - Removed unused ``repo_skeleton`` / ``data_flow`` computations from
    ``load_task_to_env_prompt`` (they were computed but never used in
    the returned prompt string in ZeroRepo).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from common.llm_client import LLMClient
from common.llm_types import (
    AssistantMessage,
    Memory,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from common.tools import Tool
from rpg_agent.env.env import Env
from rpg_agent.prompts import REPO_AGENT_SYSTEM_PROMPT
from rpg import RPG

logger = logging.getLogger(__name__)


class RPGAgent:
    """Agent that uses LLM + RPG-powered tools to localize code issues.

    The agent follows a *reasoning -> action -> feedback* loop:

    1. The LLM receives conversational history and produces an action.
    2. The ``Env`` parses the action, executes the matched tool, and
       returns structured feedback.
    3. The feedback is appended to memory and fed into the next turn.

    Args:
        instance_id: Unique identifier for this agent run.
        task: The task description (e.g. a GitHub issue body).
        repo_dir: Path to the target repository root.
        repo_name: Human-readable repository name (for logging).
        repo_rpg: Pre-loaded ``RPG`` instance for the target repository.
        max_steps: Maximum number of reasoning steps before stopping.
        context_window: Number of message pairs kept in active LLM context.
        register_tools: List of ``Tool`` **classes** to register with the
            environment.  If ``None``, defaults to an empty list.
        logger: Optional pre-configured logger.  If ``None``, a default
            logger with a ``StreamHandler`` is created.

    Source: RPG-ZeroRepo ``RPGAgent``
    """

    def __init__(
        self,
        instance_id: str,
        task: str,
        repo_dir: str,
        repo_name: str,
        repo_rpg: RPG,
        max_steps: int = 30,
        context_window: int = 30,
        register_tools: Optional[List[type[Tool]]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self.repo_name = repo_name

        self._llm = LLMClient()

        self._memory = Memory(context_window=context_window)
        self._task = task
        self._max_steps = max_steps

        self._sys_prompt = REPO_AGENT_SYSTEM_PROMPT
        self._agent_env = Env(
            instance_id=instance_id,
            repo_dir=repo_dir,
            rpg=repo_rpg,
            register_tools=register_tools or [],
        )

        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.step_token_usage: List[Dict[str, int]] = []

        # Logger setup
        if logger is None:
            logger = logging.getLogger(f"RPGAgent-{instance_id}")
            logger.setLevel(logging.INFO)

            # Only add handler if none exists (avoid duplicate log lines)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        self.logger = logger

    # ------------------------------------------------------------------
    # Memory initialization
    # ------------------------------------------------------------------

    def init_memory(self) -> None:
        """Clear memory and inject the system prompt with tool descriptions."""
        self._memory.clear_memory()

        tool_descriptions = self._agent_env.tool_handler.describe_registered_tools()

        system_prompt = self._sys_prompt.format_map(
            {"Tool_Description": tool_descriptions}
        )

        self.logger.info("Agent System Prompt: %s", system_prompt)

        self._memory.add_message(SystemMessage(content=system_prompt))

    # ------------------------------------------------------------------
    # Task prompt construction
    # ------------------------------------------------------------------

    def load_task_to_env_prompt(self) -> str:
        """Build the initial user prompt that presents the task to the LLM.

        Returns:
            A prompt string containing the issue text and localization
            instructions.

        Note:
            The original ZeroRepo implementation computed ``skeleton_info``,
            ``rpg_area``, and ``data_flow_str`` here but never included
            them in the returned prompt.  Those unused computations have
            been omitted.
        """
        env_prompt = (
            "== GitHub Issue ==\n"
            "<issue>\n"
            f"{self._task.strip()}\n"
            "</issue>\n\n"
            "== Task Begin ==\n"
            "Given the following GitHub problem description, please begin to "
            "localize all(5-10) the specific files, classes or functions, and "
            "lines of code that need modification or contain key information "
            "to resolve the issue.\n"
            "In every step, base your reasoning on evidence you have already "
            "observed (issue text, repository browsing results, and code you "
            "have opened); do not invent or guess file paths, filenames, "
            "symbols, or line numbers that you cannot verify.\n"
            'Do **not** call `terminate` at the first message.'
        )

        return env_prompt

    # ------------------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------------------

    def step(self, step_id: Optional[int] = None) -> tuple[str, bool, bool]:
        """Run one reasoning -> action -> feedback loop.

        Args:
            step_id: Optional step number for token usage tracking.

        Returns:
            Tuple of ``(feedback, tool_success, is_terminate)``.
        """
        # 1. Ask the LLM for the next action
        llm_response = self._llm.generate_with_memory(memory=self._memory)

        # Track token usage
        usage = self._llm.last_usage
        step_usage: Dict[str, int] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if step_id is not None:
            step_usage["step_id"] = step_id
        self.total_input_tokens += step_usage["input_tokens"]
        self.total_output_tokens += step_usage["output_tokens"]
        self.step_token_usage.append(step_usage)

        # 2. Append assistant message to memory
        self._memory.add_message(
            message=AssistantMessage(content=llm_response or "")
        )
        self.logger.info("[LLM Response]: %s", llm_response)

        # 3. Pass the LLM response to the environment
        feedback, tool_suc, is_terminate = self._agent_env.step(
            response=llm_response or ""
        )

        self.logger.info("[Tool Feedback]: %s", feedback)

        return feedback, tool_suc, is_terminate

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------

    def run(self, max_error_times: int = 3) -> Dict[str, Any]:
        """Run the agent until the task is completed or ``max_steps`` reached.

        Args:
            max_error_times: Maximum consecutive tool-call failures before
                the agent gives up.

        Returns:
            A results dict containing trajectory, action/feedback histories,
            token usage, and termination status.
        """
        self._agent_env.reset()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.step_token_usage = []
        self.init_memory()

        self.logger.info("---------- Task Begin ! --------------")
        env_prompt = self.load_task_to_env_prompt()

        self.logger.info("[INIT PROMPT]: %s", env_prompt)

        is_terminate = False
        error_times = 0

        for step_id in range(self._max_steps):
            self.logger.info("---------- Step %d ------------", step_id + 1)

            if step_id == 0:
                env_prompt = (
                    f"[Step {step_id + 1}/{self._max_steps} User Query]: "
                    + env_prompt
                )
            else:
                env_prompt = (
                    f"[Step {step_id + 1}/{self._max_steps} "
                    f"Tool Execution Feedback]: " + env_prompt
                )

            self._memory.add_message(
                message=UserMessage(content=env_prompt)
            )
            feedback, tool_suc, is_terminate = self.step(step_id=step_id + 1)
            env_prompt = feedback

            if not tool_suc:
                error_times += 1
                if error_times >= max_error_times:
                    self.logger.error(
                        "LLM failed %d consecutive attempts to generate "
                        "a valid tool call",
                        max_error_times,
                    )
                    break
            else:
                error_times = 0

            if is_terminate:
                break
        else:
            self.logger.info(
                "[RPGAgent] Reached maximum steps without completion."
            )

        all_traj = self._memory.to_dict()

        # Convert ToolCall objects to serializable dicts
        action_history_serialized: List[Optional[Dict[str, Any]]] = []
        for a in self._agent_env.action_history:
            if a is None:
                action_history_serialized.append(None)
            else:
                action_history_serialized.append({
                    "name": a.name,
                    "call_id": a.call_id,
                    "arguments": a.arguments,
                    "id": a.id,
                })

        final_results: Dict[str, Any] = {
            "final_results": self._agent_env.final_results,
            "is_terminate": is_terminate,
            "is_suc": is_terminate and error_times < max_error_times,
            "all_traj": all_traj,
            "action_history": action_history_serialized,
            "feedback_history": list(self._agent_env.feedback_history),
            "step_token_usage": list(self.step_token_usage),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        self.logger.info(
            "[Token Usage] input=%d output=%d total=%d",
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_input_tokens + self.total_output_tokens,
        )

        return final_results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_total_tokens_usage(self) -> Dict[str, int]:
        """Return cumulative token usage for the run."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
