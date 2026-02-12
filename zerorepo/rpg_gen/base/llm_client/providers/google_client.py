"""Google Gemini API client."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage, ToolCall, ToolResult
from .retry_utils import retry_with

if TYPE_CHECKING:
    from ..client import LLMConfig


class GoogleClient(BaseLLMClient):
    """Google Gemini client with tool support."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = (
            config.api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = genai.Client(**kwargs)
        self.message_history: list[types.Content] = []
        self.system_instruction: str | None = None

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        self.message_history, self.system_instruction = self.parse_messages(messages)

    def _create_google_response(
        self,
        current_chat_contents: list[types.Content],
        generation_config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Raw API call (decorated with retry by caller)."""
        return self.client.models.generate_content(
            model=self.model,
            contents=current_chat_contents,
            config=generation_config,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        newly_parsed, system_from_msg = self.parse_messages(messages)
        current_system = system_from_msg or self.system_instruction

        if reuse_history:
            current_contents = self.message_history + newly_parsed
        else:
            current_contents = newly_parsed

        # Generation config
        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
            system_instruction=current_system,
        )
        if self.config.stop:
            generation_config.stop_sequences = self.config.stop

        # Add tools if provided
        if tools:
            tool_schemas = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.get_name(),
                            description=tool.get_description(),
                            parameters=tool.get_input_schema(),
                        )
                    ]
                )
                for tool in tools
            ]
            generation_config.tools = tool_schemas

        # Call with retry
        retry_fn = retry_with(
            func=self._create_google_response,
            provider_name="Google Gemini",
            max_retries=self.config.max_retries,
        )
        response = retry_fn(current_contents, generation_config)

        # Parse response
        content = ""
        tool_calls: list[ToolCall] = []
        assistant_response_content = None

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                assistant_response_content = candidate.content
                for part in candidate.content.parts:
                    if part.text:
                        content += part.text
                    elif part.function_call:
                        tool_calls.append(
                            ToolCall(
                                call_id=str(uuid.uuid4()),
                                name=part.function_call.name or "tool",
                                arguments=dict(part.function_call.args)
                                if part.function_call.args
                                else {},
                            )
                        )

        # Update history
        if reuse_history:
            new_history = self.message_history + newly_parsed
        else:
            new_history = newly_parsed

        if assistant_response_content:
            new_history.append(assistant_response_content)

        self.message_history = new_history
        if current_system:
            self.system_instruction = current_system

        usage = None
        if response.usage_metadata:
            usage = LLMUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                cache_read_input_tokens=getattr(
                    response.usage_metadata, "cached_content_token_count", 0
                )
                or 0,
            )

        return LLMResponse(
            content=content,
            usage=usage,
            model=self.model,
            finish_reason=str(
                response.candidates[0].finish_reason.name
                if response.candidates and response.candidates[0].finish_reason
                else "unknown"
            ),
            tool_calls=tool_calls if tool_calls else None,
        )

    def parse_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[list[types.Content], str | None]:
        """Parse messages to Gemini format, separating system instructions."""
        gemini_messages: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.tool_result:
                gemini_messages.append(
                    types.Content(
                        role="tool",
                        parts=[self._parse_tool_result(msg.tool_result)],
                    )
                )
            elif msg.tool_call:
                gemini_messages.append(
                    types.Content(
                        role="model",
                        parts=[self._parse_tool_call(msg.tool_call)],
                    )
                )
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg.content or "")],
                    )
                )

        return gemini_messages, system_instruction

    @staticmethod
    def _parse_tool_call(tool_call: ToolCall) -> types.Part:
        return types.Part.from_function_call(
            name=tool_call.name, args=tool_call.arguments
        )

    @staticmethod
    def _parse_tool_result(tool_result: ToolResult) -> types.Part:
        result_content: dict[str, str] = {}
        if tool_result.result is not None:
            try:
                json.dumps(tool_result.result)
                result_content["result"] = tool_result.result
            except (TypeError, OverflowError) as e:
                tb = traceback.format_exc()
                err_msg = f"JSON serialization failed: {e}\n{tb}"
                if tool_result.error:
                    result_content["error"] = f"{tool_result.error}\n\n{err_msg}"
                else:
                    result_content["error"] = err_msg
                result_content["result"] = str(tool_result.result)

        if tool_result.error and "error" not in result_content:
            result_content["error"] = tool_result.error

        if not result_content:
            result_content["status"] = (
                "Tool executed successfully but returned no output."
            )

        if not hasattr(tool_result, "name") or not tool_result.name:
            raise AttributeError(
                "ToolResult must have a 'name' attribute matching the function that was called."
            )
        return types.Part.from_function_response(
            name=tool_result.name, response=result_content
        )
