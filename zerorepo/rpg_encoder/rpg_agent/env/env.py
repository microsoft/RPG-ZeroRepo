import os
import logging
from typing import List, Dict, Optional
import asyncio
from zerorepo.rpg_gen.base.node import (
    RepoSkeleton,
    filter_non_test_py_files
)
from zerorepo.rpg_gen.base.rpg import RPG, DependencyGraph
from zerorepo.rpg_gen.base.tools import (
    Tool, ToolCall,
    ToolResult, ToolExecutor,
    ToolHandler
)
from zerorepo.utils.api import parse_thinking_output
from zerorepo.utils.repo import load_skeleton_from_repo
from .searcher import RepoEntitySearcher, RepoDependencySearcher
from ..ops.bm25_model import build_code_retriever_from_repo


class Env:

    def __init__(
        self,
        instance_id: str,
        repo_dir: str,
        rpg: RPG,
        register_tools: List[Tool],
        persist_dir: str = None,
        load_bm25: bool = True
    ):

        self.instance_id = instance_id
        self.repo_dir = repo_dir
        self.repo_skeleton, _, _ = load_skeleton_from_repo(
            repo_dir=self.repo_dir,
            filter_func=filter_non_test_py_files
        )
        self.rpg = rpg

        self.tool_handler = ToolHandler(tools=register_tools)
        self.tool_executor = ToolExecutor(tools=register_tools)

        # Local environment memory
        self.last_action: Optional[ToolCall] = None
        self.last_action_result: Optional[ToolResult] = None
        self.last_feedback: Optional[str] = None
        self.step_count: int = 0

        # New: record full interaction history
        self.action_history: List[Optional[ToolCall]] = []
        self.feedback_history: List[str] = []

        # Initialize searchers from RPG
        self.entity_searcher = RepoEntitySearcher(rpg=rpg) if rpg else None
        self.dep_searcher = RepoDependencySearcher.from_rpg(rpg) if rpg else None

        # BM25 Retriever
        self.bm25_retriever = None
        self.persist_dir = persist_dir

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.bm25_persist_path = os.path.join(
                persist_dir,
                f"{instance_id.replace('/', '_')}_bm25"
            )
        else:
            self.bm25_persist_path = None

        # Load BM25 retriever
        if load_bm25:
            self._load_bm25_retriever()

        self.final_results = []

    def _load_bm25_retriever(self):
        """Load or build BM25 retriever for code search."""
        try:
            # Try to load from persist path first
            if self.bm25_persist_path and os.path.exists(self.bm25_persist_path):
                from ..ops.bm25_model import build_retriever_from_persist_dir
                logging.info(f"Loading BM25 retriever from {self.bm25_persist_path}")
                self.bm25_retriever = build_retriever_from_persist_dir(self.bm25_persist_path)
            else:
                # Build new retriever from repo skeleton
                logging.info("Building BM25 retriever from repo skeleton...")
                self.bm25_retriever = build_code_retriever_from_repo(
                    skeleton=self.repo_skeleton,
                    similarity_top_k=10,
                    persist_path=self.bm25_persist_path,
                    show_progress=False
                )
                logging.info("BM25 retriever built successfully.")
        except Exception as e:
            logging.warning(f"Failed to load/build BM25 retriever: {e}")
            self.bm25_retriever = None

    # ============================================================
    # Core step logic
    # ============================================================
    def step(self, response: str) -> tuple[str, bool]:
        """
        Parse an LLM output, execute the matched tool, and return feedback.
        Keeps track of last action and result.
        If the current action is identical to the previous one,
        return a prompt asking the model to propose a different action.
        """
        parsed_response = parse_thinking_output(output=response)
        try:
            parsed_action_list = self.tool_handler.parse_and_match_tool(
                llm_output=parsed_response
            )
            # parse_and_match_tool returns List[ToolCall]; extract first element
            parsed_action = parsed_action_list[0] if parsed_action_list else None

            self.step_count += 1

            # --- check for identical action to previous one ---
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

            # --- handle missing action ---
            if not parsed_action:
                feedback = (
                    "No valid tool action was detected from your previous output. "
                    "Please specify the tool you want to use and provide arguments in a clear format.\n\n"
                    f"Available tools: {self.tool_handler.list_registered()}\n"
                    "Make sure to output only the tool call, without extra explanation."
                )
                self.last_feedback = feedback
                self.action_history.append(None)
                self.feedback_history.append(feedback)
                return feedback, False, False

            # --- execute the tool ---
            env_param = {
                "environment": self,
                "rpg": self.rpg,
                "repo_skeleton": self.repo_skeleton,
                "entity_searcher": self.entity_searcher,
                "dep_searcher": self.dep_searcher,
                "bm25_retriever": self.bm25_retriever
            }
            result: ToolResult = asyncio.run(
                self.tool_executor.execute_tool_call(parsed_action, env=env_param)
            )
            self.last_action_result = result
            tool_suc = result.success

            # --- generate feedback ---
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

            # --- record feedback and action ---
            self.last_feedback = feedback
            self.action_history.append(parsed_action)
            self.feedback_history.append(feedback)

            is_terminate = True \
                if parsed_action.name.lower() == "terminate" and tool_suc \
                    else False
            
            return feedback, tool_suc, is_terminate

        except Exception as e:
            feedback = (
                f"An error occurred while parsing or executing the tool: {e}\n"
                "Please reformat your response and specify the correct tool name and arguments."
            )
            self.last_feedback = feedback
            self.action_history.append(None)
            self.feedback_history.append(feedback)
            return feedback, False, False

    # ============================================================
    # State management
    # ============================================================
    def reset(self):
        """Reset the environment memory."""
        self.last_action = None
        self.last_action_result = None
        self.last_feedback = None
        self.step_count = 0
        self.action_history.clear()
        self.feedback_history.clear()

    # ============================================================
    # Accessors
    # ============================================================
    def get_last_action_info(self) -> Dict:
        """Return summary info about the last tool execution."""
        return {
            "step": self.step_count,
            "action": self.last_action.name if self.last_action else None,
            "arguments": self.last_action.arguments if self.last_action else None,
            "success": self.last_action_result.success if self.last_action_result else None,
            "output": self.last_action_result.result if self.last_action_result else None,
            "error": self.last_action_result.error if self.last_action_result else None,
            "feedback": self.last_feedback,
        }

    def get_history(self) -> List[Dict]:
        """Return full history of all steps (action + feedback)."""
        history = []
        for idx, (action, feedback) in enumerate(zip(self.action_history, self.feedback_history), start=1):
            history.append({
                "step": idx,
                "action": action.name if action else None,
                "arguments": action.arguments if action else None,
                "feedback": feedback,
            })
        return history