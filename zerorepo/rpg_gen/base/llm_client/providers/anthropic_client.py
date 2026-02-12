"""Anthropic Claude API client."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import anthropic

from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage, ToolCall, ToolResult
from .retry_utils import retry_with

if TYPE_CHECKING:
    from ..client import LLMConfig


class AnthropicClient(BaseLLMClient):
    """Anthropic client with tool support."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client: anthropic.Anthropic = anthropic.Anthropic(**kwargs)
        self.message_history: list[anthropic.types.MessageParam] = []
        self.system_message: str | anthropic.NotGiven = anthropic.NOT_GIVEN

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        self.message_history = self.parse_messages(messages)

    def _create_anthropic_response(
        self,
        tool_schemas: list[anthropic.types.ToolUnionParam] | anthropic.NotGiven,
    ) -> anthropic.types.Message:
        """Raw API call (decorated with retry by caller)."""
        return self.client.messages.create(
            model=self.model,
            messages=self.message_history,
            max_tokens=self.config.max_tokens,
            system=self.system_message,
            tools=tool_schemas,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        anthropic_messages = self.parse_messages(messages)
        self.message_history = (
            self.message_history + anthropic_messages
            if reuse_history
            else anthropic_messages
        )

        # Build tool schemas
        tool_schemas: list[anthropic.types.ToolUnionParam] | anthropic.NotGiven = (
            anthropic.NOT_GIVEN
        )
        if tools:
            tool_schemas = []
            for tool in tools:
                tool_schemas.append(
                    anthropic.types.ToolParam(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.get_input_schema(),
                    )
                )

        # Call with retry
        retry_fn = retry_with(
            func=self._create_anthropic_response,
            provider_name="Anthropic",
            max_retries=self.config.max_retries,
        )
        response = retry_fn(tool_schemas)

        # Parse response
        content = ""
        tool_calls: list[ToolCall] = []

        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
                self.message_history.append(
                    anthropic.types.MessageParam(
                        role="assistant", content=content_block.text
                    )
                )
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input,
                    )
                )
                self.message_history.append(
                    anthropic.types.MessageParam(
                        role="assistant", content=[content_block]
                    )
                )

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                cache_read_input_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                )
                or 0,
            )

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls if tool_calls else None,
        )

    def parse_messages(
        self, messages: list[LLMMessage]
    ) -> list[anthropic.types.MessageParam]:
        """Parse LLMMessage list to Anthropic format."""
        anthropic_messages: list[anthropic.types.MessageParam] = []
        for msg in messages:
            if msg.role == "system":
                self.system_message = msg.content if msg.content else anthropic.NOT_GIVEN
            elif msg.tool_result:
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="user",
                        content=[self._parse_tool_result(msg.tool_result)],
                    )
                )
            elif msg.tool_call:
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="assistant",
                        content=[self._parse_tool_call(msg.tool_call)],
                    )
                )
            else:
                if msg.role not in ("user", "assistant"):
                    raise ValueError(f"Invalid message role: {msg.role}")
                if not msg.content:
                    raise ValueError("Message content is required")
                anthropic_messages.append(
                    anthropic.types.MessageParam(role=msg.role, content=msg.content)
                )
        return anthropic_messages

    @staticmethod
    def _parse_tool_call(tool_call: ToolCall) -> anthropic.types.ToolUseBlockParam:
        return anthropic.types.ToolUseBlockParam(
            type="tool_use",
            id=tool_call.call_id,
            name=tool_call.name,
            input=json.dumps(tool_call.arguments),
        )

    @staticmethod
    def _parse_tool_result(
        tool_result: ToolResult,
    ) -> anthropic.types.ToolResultBlockParam:
        result: str = ""
        if tool_result.result:
            result += tool_result.result + "\n"
        if tool_result.error:
            result += "Tool call failed with error:\n" + tool_result.error
        result = result.strip()
        if not tool_result.success and not result:
            result = "Tool execution failed without providing error details."

        return anthropic.types.ToolResultBlockParam(
            tool_use_id=tool_result.call_id,
            type="tool_result",
            content=result,
            is_error=not tool_result.success,
        )
