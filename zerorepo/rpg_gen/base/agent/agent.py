import logging
from typing import Union, Dict, Optional, List, Any
from ..llm_client import (
    Memory,
    LLMClient, LLMConfig,
    UserMessage, 
    SystemMessage,
    AssistantMessage
)
from ..tools import Tool
from pydantic import BaseModel
from zerorepo.utils.api import parse_thinking_output
from .env import BaseEnv, ReviewEnv


class BaseAgent:
    def __init__(
        self, 
        llm_cfg: Union[str, Dict, LLMConfig],
        system_prompt: str="",
        context_window: int = 30,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self._setup_logger(logger)
        
        self._llm = LLMClient(
            config=llm_cfg
        )
        self._sys_prompt = system_prompt
        self._memory = Memory(context_window=context_window)
        self._env = BaseEnv(
            register_tools=register_tools,
            logger=self.logger,
            **kwargs
        )
        
    
    def _setup_logger(self, logger: Optional[logging.Logger]=None):
        if logger is None:
            logger = logging.getLogger(f"Agent")
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
        tool_decriptions = self._env.tool_handler.describe_registered_tools()
        system_prompt = self._sys_prompt.format_map(
            {"Tool_Description": tool_decriptions}
        )
        self.logger.info(f"Agent System Prompt: {system_prompt}")    
        self._memory.add_message(SystemMessage(
            content=system_prompt
        ))
    
    
    def step(self) -> str:
        """
        Run one reasoning→action→feedback loop.
        Returns the feedback text produced by the environment.
        """
        # 1. Ask the LLM for the next action
        llm_response = self._llm.generate(memory=self._memory)

        self._memory.add_message(
            message=AssistantMessage(content=llm_response)
        )
        self.logger.info(f"[LLM Response]: {llm_response}")
        # 2. Pass the LLM response to the environment
        tool_output = parse_thinking_output(llm_response)
        
        feedback, tool_suc, is_terminate = self._env.step(response=tool_output)
        self.logger.info(f"[Tool Feedback]: {feedback}")
        
        return feedback, tool_suc, is_terminate
        
    # ------------------------------------------------------
    # Full loop
    # ------------------------------------------------------
    def run(self, task: str, max_error_times:int=3, max_steps: int=20) -> List[Dict]:
        """
        Run the agent until the task is completed or max_steps reached.
        """
        self._env.reset()
        self.init_memory()

        self.logger.info("----------Task Begin !--------------")
        env_prompt = task
        
        self.logger.info(f"[INIT PROMPT]: {env_prompt}")
        
        is_terminate = False
        error_times = 0
        
        for step_id in range(max_steps):
            self.logger.info(f"---------- Step {step_id + 1} ------------")

            if step_id == 0: 
                env_prompt = f"[Step {step_id + 1} User Query]: " + env_prompt
            else:
                env_prompt = f"[Step {step_id + 1} Tool Execution Feedback]: " + env_prompt
                
            self._memory.add_message(message=UserMessage(content=env_prompt))
            feedback, tool_suc, is_terminate = self.step()
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
        for action_list in self._env.action_history:
            if action_list is None or not action_list:
                action_history_serialized.append(None)
            elif isinstance(action_list, list):
                # Convert list of tool calls
                serialized_list = []
                for a in action_list:
                    if hasattr(a, 'name'):
                        serialized_list.append({
                            "name": a.name,
                            "call_id": getattr(a, 'call_id', None),
                            "arguments": getattr(a, 'arguments', {}),
                            "id": getattr(a, 'id', None),
                        })
                action_history_serialized.append(serialized_list)
            else:
                # Single tool call (backward compatibility)
                if hasattr(action_list, 'name'):
                    action_history_serialized.append({
                        "name": action_list.name,
                        "call_id": getattr(action_list, 'call_id', None),
                        "arguments": getattr(action_list, 'arguments', {}),
                        "id": getattr(action_list, 'id', None),
                    })
                else:
                    action_history_serialized.append(None)

        final_results = {
            "final_results": self._env.final_results,
            "is_terminate": is_terminate,
            "is_suc": is_terminate and error_times < max_error_times,
            "all_traj": all_traj,
            "action_history": action_history_serialized,
            "feedback_history": list(self._env.feedback_history),
        }

        return final_results
    
    
class AgentwithReview(BaseAgent):
    
    def __init__(       
        self, 
        llm_cfg: Union[str, Dict, LLMConfig],
        design_system_prompt: str="",
        design_context_window: int = 30,
        review_system_prompt: str="",
        review_format: BaseModel=None,
        review_context_window: int=10,
        max_review_times: int=3,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        super().__init__(
            llm_cfg=llm_cfg,
            system_prompt=design_system_prompt,
            context_window=design_context_window,
            register_tools=register_tools,
            logger=logger,
            **kwargs
        )

        self.review_llm = LLMClient(llm_cfg)
        self.review_format = review_format
        self.review_system_prompt = ""
        self.review_memory = Memory(
            context_window=review_context_window
        )
        self.review_memory.add_message(
            SystemMessage(
                content=review_system_prompt
            )
        )

        self._env = ReviewEnv(
            review_llm=self.review_llm,
            review_format=review_format,
            review_memory=self.review_memory,
            review_times=max_review_times,
            register_tools=register_tools,
            logger=self.logger
        )
    
    
    def step(self) -> str:
        """
        Run one reasoning→action→feedback loop.
        Returns the feedback text produced by the environment.
        """
        # 1. Ask the LLM for the next action
        llm_response = self._llm.generate(memory=self._memory)

        self._memory.add_message(
            message=AssistantMessage(content=llm_response)
        )
        self.logger.info(f"[LLM Response]: {llm_response}")
        # 2. Pass the LLM response to the environment
        tool_output = parse_thinking_output(llm_response)
        feedback, tool_suc, is_terminate = self._env.step(response=tool_output)
        self.logger.info(f"[Tool Feedback]: {feedback}")
        
        return feedback, tool_suc, is_terminate
        
    # ------------------------------------------------------
    # Full loop
    # ------------------------------------------------------
    def run(self, task: str, max_error_times:int=3, max_steps:int=20) -> List[Dict]:
        """
        Run the agent until the task is completed or max_steps reached.
        """
        self._env.reset()
        self.init_memory()

        self.logger.info("----------Task Begin !--------------")
        env_prompt = task
        
        self.logger.info(f"[INIT PROMPT]: {env_prompt}")
        
        is_terminate = False
        error_times = 0
        
        for step_id in range(max_steps):
            self.logger.info(f"---------- Step {step_id + 1} ------------")

            if step_id == 0: 
                env_prompt = f"[Step {step_id + 1} User Query]: " + env_prompt
            else:
                env_prompt = f"[Step {step_id + 1} Tool Execution Feedback]: " + env_prompt
                
            self._memory.add_message(message=UserMessage(content=env_prompt))
            feedback, tool_suc, is_terminate = self.step()
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
        for action_list in self._env.action_history:
            if action_list is None or not action_list:
                action_history_serialized.append(None)
            elif isinstance(action_list, list):
                # Convert list of tool calls
                serialized_list = []
                for a in action_list:
                    if hasattr(a, 'name'):
                        serialized_list.append({
                            "name": a.name,
                            "call_id": getattr(a, 'call_id', None),
                            "arguments": getattr(a, 'arguments', {}),
                            "id": getattr(a, 'id', None),
                        })
                action_history_serialized.append(serialized_list)
            else:
                # Single tool call (backward compatibility)
                if hasattr(action_list, 'name'):
                    action_history_serialized.append({
                        "name": action_list.name,
                        "call_id": getattr(action_list, 'call_id', None),
                        "arguments": getattr(action_list, 'arguments', {}),
                        "id": getattr(action_list, 'id', None),
                    })
                else:
                    action_history_serialized.append(None)

        all_review_traj = self.review_memory.to_dict()
        
        final_results = {
            "final_results": self._env.final_results,
            "is_terminate": is_terminate,
            "is_suc": is_terminate and error_times < max_error_times,
            "all_traj": all_traj,
            "all_review_traj": all_review_traj,
            "action_history": action_history_serialized,
            "feedback_history": list(self._env.feedback_history),
        }

        return final_results
        
        
        
        
        