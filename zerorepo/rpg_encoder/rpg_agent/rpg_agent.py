import json
import logging
from typing import (
    Dict, Optional, 
    Union, List
)
from networkx import MultiDiGraph
from zerorepo.rpg_gen.base.tools import ( 
    Tool, 
    ToolHandler, 
    ToolExecutor                          
)
from zerorepo.rpg_gen.base.node import filter_non_test_py_files
from zerorepo.rpg_gen.base.rpg import RPG
from .prompts import REPO_AGENT_SYSTEM_PROMPT
from .env import Env
from zerorepo.rpg_gen.base.llm_client import (
    Memory,
    LLMClient, LLMConfig,
    UserMessage, 
    SystemMessage,
    AssistantMessage
)
from zerorepo.utils.data_flow import format_data_flow_for_llm

class RPGAgent:
    def __init__(
        self, 
        llm_cfg: Union[str, Dict, LLMConfig],
        instance_id: str,
        task: str,
        repo_dir: str,
        repo_name: str,
        dep_graph: MultiDiGraph,
        repo_rpg: RPG,
        max_steps: int = 30,
        context_window: int = 30,
        register_tools: Optional[List[Tool]] = None,
        persist_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.repo_name = repo_name

        self._llm = LLMClient(config=llm_cfg)

        self._memory = Memory(context_window=context_window)
        self._task = task
        self._max_steps = max_steps

        self._sys_prompt = REPO_AGENT_SYSTEM_PROMPT
        self._agent_env = Env(
            instance_id=instance_id,
            repo_dir=repo_dir,
            rpg=repo_rpg,
            register_tools=register_tools or [],
            persist_dir=persist_dir
        )
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.step_token_usage: List[Dict[str, int]] = []
        
        
        if logger is None:
            logger = logging.getLogger(f"RepoAgent-{instance_id}")
            logger.setLevel(logging.INFO)

            # Only add handler if no handler exists (避免重复 log)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                    "%Y-%m-%d %H:%M:%S"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        self.logger = logger


    def init_memory(self):
        self._memory.clear_memory()        
        
        tool_decriptions = self._agent_env.tool_handler.describe_registered_tools()
        
        system_prompt = self._sys_prompt.format_map(
            {"Tool_Description": tool_decriptions}
        )
        
        self.logger.info(f"Agent System Prompt: {system_prompt}")
        
        self._memory.add_message(SystemMessage(
            content=system_prompt
        ))
    
    
    def load_task_to_env_prompt(self) -> str:
        skeleton_info = self._agent_env.repo_skeleton.to_tree_string(
            filter_func=filter_non_test_py_files,
        )
        rpg_area = self._agent_env.rpg.visualize_dir_map(max_depth=1)
        data_flow_str = format_data_flow_for_llm(data_flow=self._agent_env.rpg.data_flow)
        env_prompt = (
            "== GitHub Issue ==\n"
            "<issue>\n"
            f"{self._task.strip()}\n"
            "</issue>\n\n"
            "== Task Begin ==\n"
            "Given the following GitHub problem description, please begin to localize all(5-10) the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.\n"
            "In every step, base your reasoning on evidence you have already observed (issue text, repository browsing results, and code you have opened); do not invent or guess file paths, filenames, symbols, or line numbers that you cannot verify.\n"
            "Do **not** call `terminate` at the first message."
        )
       
        return env_prompt
        
    # ------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------
    def step(self, step_id: Optional[int] = None) -> tuple[str, bool, bool]:
        """
        Run one reasoning→action→feedback loop.
        Returns the feedback text produced by the environment.
        """
        # 1. Ask the LLM for the next action
        llm_response = self._llm.generate(memory=self._memory)
        usage = self._llm.last_usage
        step_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if step_id is not None:
            step_usage["step_id"] = step_id
        self.total_prompt_tokens += step_usage["prompt_tokens"]
        self.total_completion_tokens += step_usage["completion_tokens"]
        self.step_token_usage.append(step_usage)

        self._memory.add_message(
            message=AssistantMessage(content=llm_response)
        )
        self.logger.info(f"[LLM Response]: {llm_response}")
        
        # 2. Pass the LLM response to the environment
        feedback, tool_suc, is_terminate = self._agent_env.step(response=llm_response)
        
        self.logger.info(f"[Tool Feedback]: {feedback}")
        
        return feedback, tool_suc, is_terminate
    
    # ------------------------------------------------------
    # Full loop
    # ------------------------------------------------------
    def run(self, max_error_times:int=3) -> List[Dict]:
        """
        Run the agent until the task is completed or max_steps reached.
        """
        self._agent_env.reset()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.step_token_usage = []
        self.init_memory()

        self.logger.info("----------Task Begin !--------------")
        env_prompt = self.load_task_to_env_prompt()
        
        self.logger.info(f"[INIT PROMPT]: {env_prompt}")
        
        is_terminate = False
        error_times = 0
        
        for step_id in range(self._max_steps):
            self.logger.info(f"---------- Step {step_id + 1} ------------")

            if step_id == 0:
                env_prompt = f"[Step {step_id + 1}/{self._max_steps} User Query]: " + env_prompt
            else:
                env_prompt = f"[Step {step_id + 1}/{self._max_steps} Tool Execution Feedback]: " + env_prompt
                
            self._memory.add_message(message=UserMessage(content=env_prompt))
            feedback, tool_suc, is_terminate = self.step(step_id=step_id + 1)
            env_prompt = feedback

            if not tool_suc:
                error_times += 1
                if error_times >= max_error_times:
                    self.logger.error(f"LLM Attempt {max_error_times} to generate a valid tool call")
                    break
            else:
                error_times = 0
                            
            if is_terminate:
                break
        else:
            self.logger.info("[RepoAgent] Reached maximum steps without completion.")

        all_traj = self._memory.to_dict()
        
        # Convert ToolCall objects to dicts
        action_history_serialized = []
        for a in self._agent_env.action_history:
            if a is None:
                action_history_serialized.append(None)
            else:
                # convert ToolCall to dict, including arguments
                action_history_serialized.append({
                    "name": a.name,
                    "call_id": a.call_id,
                    "arguments": a.arguments,   # if ToolCallArguments is not JSON-safe, convert here
                    "id": a.id,
                })

        final_results = {
            "final_results": self._agent_env.final_results,
            "is_terminate": is_terminate,
            "is_suc": is_terminate and error_times < max_error_times,
            "all_traj": all_traj,
            # NEW: dedicated histories
            "action_history": action_history_serialized,
            "feedback_history": list(self._agent_env.feedback_history),
            "step_token_usage": list(self.step_token_usage),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        self.logger.info(
            "[Token Usage] prompt=%d completion=%d total=%d",
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_prompt_tokens + self.total_completion_tokens,
        )
        
        return final_results

    def get_total_tokens_usage(self) -> Dict[str, int]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
