"""Ollama API client (local inference)."""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING

import openai
from ollama import chat as ollama_chat

from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, ToolCall, ToolResult
from .retry_utils import retry_with

if TYPE_CHECKING:
    from ..client import LLMConfig


class OllamaClient(BaseLLMClient):
    """Ollama client for local model inference."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        base_url = config.base_url or "http://localhost:11434/v1"
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=api_key, base_url=base_url
        )
        self.message_history: list[dict] = []

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        self.message_history = self.parse_messages(messages)

    def _create_ollama_response(self, tool_schemas: list[dict] | None):
        """Raw API call (decorated with retry by caller)."""
        tools_param = None
        if tool_schemas:
            tools_param = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tool_schemas
            ]
        return ollama_chat(
            messages=self.message_history,
            model=self.model,
            tools=tools_param,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        msgs = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_input_schema(),
                    "type": "function",
                }
                for tool in tools
            ]

        if reuse_history:
            self.message_history = self.message_history + msgs
        else:
            self.message_history = msgs

        retry_fn = retry_with(
            func=self._create_ollama_response,
            provider_name="Ollama",
            max_retries=self.config.max_retries,
        )
        response = retry_fn(tool_schemas)

        content = ""
        tool_calls: list[ToolCall] = []

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=str(uuid.uuid4()),
                        name=tool.function.name,
                        arguments=dict(tool.function.arguments),
                        id=str(uuid.uuid4()),
                    )
                )
        else:
            content = str(response.message.content)

        return LLMResponse(
            content=content,
            usage=None,
            model=self.model,
            finish_reason=None,
            tool_calls=tool_calls if tool_calls else None,
        )

    def parse_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Parse LLMMessage list to simple dict format for Ollama."""
        parsed: list[dict] = []
        for msg in messages:
            if msg.tool_result:
                result: str = ""
                if msg.tool_result.result:
                    result += msg.tool_result.result + "\n"
                if msg.tool_result.error:
                    result += msg.tool_result.error
                parsed.append({
                    "role": "tool",
                    "content": result.strip(),
                    "tool_call_id": msg.tool_result.call_id,
                })
            elif msg.tool_call:
                parsed.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "name": msg.tool_call.name,
                        "arguments": msg.tool_call.arguments,
                    }),
                })
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                parsed.append({"role": msg.role, "content": msg.content})
        return parsed
