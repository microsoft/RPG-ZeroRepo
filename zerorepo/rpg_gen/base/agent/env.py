import asyncio
import json
import logging
from typing import List, Optional, Dict, Union
from pydantic import BaseModel
from ..tools import (
    Tool,
    ToolCall,
    ToolExecResult,
    ToolResult,
    ToolHandler,
    ToolExecutor
)
from ..llm_client import (
    LLMClient,
    Memory,
    SystemMessage,
    UserMessage, 
    AssistantMessage
)


class BaseEnv:
    
    def __init__(
        self,
        register_tools: List[Tool],
        logger: logging.Logger
    ):
        self.tool_handler = ToolHandler(
            tools=register_tools
        )
        self.tool_executor = ToolExecutor(
            tools=register_tools
        )
        # Local environment memory
        self.last_action: List[ToolCall] = []
        self.last_action_suc: bool = True
        self.last_action_result: Optional[ToolResult] = None
        self.last_feedback: Optional[str] = None
        self.step_count: int = 0
        # New: record full interaction history
        self.action_history: List[Optional[ToolCall]] = []
        self.feedback_history: List[str] = []
        
        self.logger = logger
        
        self.final_results = []



    def reset(self):
        """Reset the environment memory."""
        self.last_action = []
        self.last_action_result = None
        self.last_feedback = None
        self.step_count = 0
        self.action_history.clear()
        self.feedback_history.clear()

    def get_last_action_info(self) -> Dict:
        """Return summary info about the last tool execution."""
        last_action = self.last_action[0] if self.last_action else None
        return {
            "step": self.step_count,
            "action": last_action.name if last_action else None,
            "arguments": last_action.arguments if last_action else None,
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
    
    
    def should_terminate(
        self,
        tool_calls: List[ToolCall],
        tool_results: List[ToolResult]
    ): 
        return any([tool_result.name == "terminate" and tool_result.success \
            for tool_result in tool_results]) and len(tool_results) > 0 
    

    def post_feedback(self):
        return ""
    
    def update_env(
        self,
        tool_call: ToolCall,
        result: ToolResult
    ):
        return None
    
    def step(self, response: str, is_update_env: bool=True) -> tuple[str, bool]:
        """
        Parse an LLM output, execute the matched tool, and return feedback.
        Keeps track of last action and result.
        If the current action is identical to the previous one,
        return a prompt asking the model to propose a different action.
        """
        try:
            parsed_actions: List[Tool] = self.tool_handler.parse_and_match_tool(
                llm_output=response
            )

            self.step_count += 1

            # --- check for identical action to previous one ---
            last_action_names = [last_act.name for last_act in self.last_action]
            last_action_args = [last_act.arguments for last_act in self.last_action]
            if (
                parsed_actions
                and self.last_action
                and all([parsed_action.name in last_action_names for parsed_action in parsed_actions])
                and all([parsed_action.arguments in last_action_args for parsed_action in parsed_actions])
            ):
                feedback = (
                    "The current tool call is identical to your previous one. This is not acceptable."
                    "You must revise your reasoning and provide a different tool action, or adjust the arguments significantly."
                    "Do not repeat the same tool call again."
                )
                self.last_feedback = feedback
                self.action_history.append(parsed_actions)
                self.feedback_history.append(feedback)
                return feedback, False, False

            self.last_action = parsed_actions

            # --- handle missing action ---
            if not parsed_actions:
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
            if len(parsed_actions) == 1:
                parsed_action = parsed_actions[0]
                result: ToolResult = asyncio.run(
                    self.tool_executor.execute_tool_call(parsed_action, env=self)
                )
                all_results = [result]
            else:
                all_results: List[ToolResult] = asyncio.run(
                    self.tool_excutor.sequential_tool_call(
                        tool_calls=parsed_actions,
                        env_params=[self * len(parsed_actions)]
                    )
                )
            
            self.last_action_result = all_results
            
            any_suc = any([result.success for result in all_results]) and len(all_results) >= 0
            is_terminate = False
            all_feedbacks = []
            for result in all_results:

                tool_call = next((tool_call for tool_call in parsed_actions if tool_call.call_id == result.call_id), None)
                assert tool_call, ""
                # --- generate feedback ---
                if result.success:
                    feedback = (
                        f"Idx: {result.call_id}: Tool '{result.name}' executed successfully.\n"
                        f"Output: {result.result}"
                    )
                    if is_update_env:
                        self.update_env(tool_call=tool_call, result=result)
                else:
                    feedback = (
                        f"Idx: {result.call_id}: Tool '{result.name}' execution failed.\n"
                        f"Error: {result.error}"
                    )
                
                all_feedbacks.append(feedback)

            is_terminate = self.should_terminate(tool_calls=parsed_actions, tool_results=all_results)
            # --- record feedback and action ---
            self.last_action_suc = any_suc
            self.last_feedback = all_feedbacks
            self.action_history.append(parsed_actions)
            self.feedback_history.append(all_feedbacks)
            
            final_feedback = "\n".join(all_feedbacks)
            final_feedback += "\n"
            final_feedback += self.post_feedback()
            
            return final_feedback, any_suc, is_terminate
        except Exception as e:
            feedback = (
                f"An error occurred while parsing or executing the tool: {e}\n"
                "Please reformat your response and specify the correct tool name and arguments."
            )
            self.last_feedback = [feedback]
            self.action_history.append([])
            self.feedback_history.append([feedback])
            return feedback, False, False


class ReviewEnv(BaseEnv):
    def __init__(
        self,    
        review_llm: LLMClient,
        review_format: BaseModel,
        review_memory: Memory,
        review_times: int=1,
        
        register_tools: List=[],
        logger: logging.Logger=None
    ):
        super().__init__(
            register_tools=register_tools,
            logger=logger or logging.getLogger(__name__)
        )
        
        self.review_llm = review_llm
        self.review_format = review_format
        self.review_memory = review_memory
        self.max_review_times = review_times
        
        self.cur_review_times = 0
    
    def post_feedback(self):
        return ""
    
    def load_review_message(self, tool_calls: List[ToolCall]):
        suc_tools_str = json.dumps([tool_call.to_dict() for tool_call in tool_calls], indent=2)
        try:
            schema = self.review_format.model_json_schema()  # pydantic v2
        except AttributeError:
            schema = self.review_format.schema()  # pydantic v1
        schema_text = json.dumps(schema, ensure_ascii=False)

        return (
            "Please review the following tool calls based on the information below:\n\n"
            f"{suc_tools_str}\n\n"
            "Your reply must be a valid JSON object that strictly follows the schema below:\n"
            f"{schema_text}"
        )
    
    def review_tool_calls(self, suc_tools: List[ToolCall]):

        review_input = self.load_review_message(tool_calls=suc_tools)
        self.review_memory.add_message(
            UserMessage(
                content=review_input
            )
        )

        review_result, response = self.review_llm.call_with_structure_output(
            memory=self.review_memory,
            response_model=self.review_format
        )
        
        self.review_memory.add_message(
            AssistantMessage(
                content=response
            )
        )
        
        return review_result
    
    
    def step(self, response: str, is_update_env: bool=True) -> tuple[str, bool]:
        """
        Parse an LLM output, execute the matched tool, and return feedback.
        Keeps track of last action and result.
        If the current action is identical to the previous one,
        return a prompt asking the model to propose a different action.
        """
        try:
            parsed_actions: List[ToolCall] = self.tool_handler.parse_and_match_tool(
                llm_output=response
            )

            self.step_count += 1

            # --- check for identical action to previous one ---
            last_action_names = [last_act.name for last_act in self.last_action]
            last_action_args = [last_act.arguments for last_act in self.last_action]
            if (
                parsed_actions
                and self.last_action
                and all([parsed_action.name in last_action_names for parsed_action in parsed_actions])
                and all([parsed_action.arguments in last_action_args for parsed_action in parsed_actions])
            ):
                feedback = (
                    "The current tool call is identical to your previous one. "
                    "Please revise your reasoning and propose a different tool action "
                    "or modify the arguments to move the task forward."
                )
                self.last_feedback = feedback
                self.action_history.append(parsed_actions)
                self.feedback_history.append(feedback)
                return feedback, False, False

            self.last_action = parsed_actions

            # --- handle missing action ---
            if not parsed_actions:
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
            if len(parsed_actions) == 1:
                parsed_action = parsed_actions[0]
                result: ToolResult = asyncio.run(
                    self.tool_executor.execute_tool_call(parsed_action, env=self)
                )
                all_results = [result]
            else:
                all_results: List[ToolResult] = asyncio.run(
                    self.tool_excutor.sequential_tool_call(
                        tool_calls=parsed_actions,
                        env_params=[self * len(parsed_actions)]
                    )
                )
            
            self.last_action_result = all_results
            
            any_suc = any([result.success for result in all_results]) and len(all_results) >= 0
            review_info = ""
            review_passed = True
            review_result = {}
            is_terminate = False
            all_feedbacks = []
            suc_tools = []
            
            for result in all_results:
                tool_call = next((tool_call for tool_call in parsed_actions if tool_call.call_id == result.call_id), None)
                assert tool_call, ""
                # --- generate feedback ---
                if result.success:
                    feedback = (
                        f"Idx: {result.call_id}: Tool '{result.name}' executed successfully.\n"
                        f"Output: {result.result}"
                    )
                    suc_tools.append(tool_call)
                    # if is_update_env:
                    #    self.update_env(tool_call=tool_call)
                else:
                    feedback = (
                        f"Idx: {result.call_id}: Tool '{result.name}' execution failed.\n"
                        f"Error: {result.error}"
                    )
                
                all_feedbacks.append(feedback)

            if self.cur_review_times < self.max_review_times:
                review_result = self.review_tool_calls(
                    suc_tools=suc_tools
                )
                review_passed = review_result.get("final_pass", False)
                
                review_info = json.dumps(review_result, indent=2)
                review_info = (
                    "Here is the review result for your tool call decisions. "
                    "Please make adjustments to any part that did not pass:\n"
                    f"{review_info}"
                )
                if not review_passed:
                    self.cur_review_times += 1 
                else:
                    self.cur_review_times = 0
            else:
                self.cur_review_times = 0 if any_suc else self.cur_review_times
            

            if review_passed:
                for result in all_results:
                    tool_call = next((tool_call for tool_call in parsed_actions if tool_call.call_id == result.call_id), None)
                    assert tool_call, ""
                    # --- generate feedback ---
                    if result.success and is_update_env:
                        self.update_env(tool_call=tool_call, result=result)
                
            is_terminate = self.should_terminate(tool_calls=parsed_actions, tool_results=all_results)
            all_feedbacks = all_feedbacks if not review_info else all_feedbacks + [review_info]

            # --- record feedback and action ---
            self.last_feedback = all_feedbacks 
            self.last_action_suc = any_suc
            self.action_history.append(parsed_actions)
            self.feedback_history.append(all_feedbacks)
            
            final_feedback = "\n".join(all_feedbacks)
            final_feedback += self.post_feedback()
     
            return final_feedback, any_suc, is_terminate
        except Exception as e:
            feedback = (
                f"An error occurred while parsing or executing the tool: {e}\n"
                "Please reformat your response and specify the correct tool name and arguments."
            )
            self.last_feedback = [feedback]
            self.action_history.append([])
            self.feedback_history.append([feedback])
            return feedback, False, False
        