"""Base class for OpenAI-compatible clients with shared logic."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage, ToolCall
from .retry_utils import retry_with

if TYPE_CHECKING:
    from ..client import LLMConfig


class ProviderConfig(ABC):
    """Abstract base class for provider-specific configurations."""

    @abstractmethod
    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        """Create the OpenAI-compatible client instance."""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for identification."""
        pass

    def get_extra_headers(self) -> dict[str, str]:
        """Get any extra headers needed for the API call."""
        return {}

    def is_reasoning_model(self, model: str) -> bool:
        """Check if this model is a reasoning model (needs special param handling)."""
        return False


class OpenAICompatibleClient(BaseLLMClient):
    """Base class for OpenAI-compatible clients with shared logic."""

    def __init__(self, config: LLMConfig, provider_config: ProviderConfig):
        super().__init__(config)
        self.provider_config = provider_config
        self.client = provider_config.create_client(config)
        self.message_history: list[ChatCompletionMessageParam] = []

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_response(
        self,
        tool_schemas: list[ChatCompletionToolParam] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Create a response using the provider's API."""
        # Re-create client to refresh credentials (e.g. Azure token)
        self.client = self.provider_config.create_client(self.config)

        kwargs: dict = {
            "model": self.model,
            "messages": self.message_history,
            "tools": tool_schemas if tool_schemas else openai.NOT_GIVEN,
            "top_p": self.config.top_p,
            "n": 1,
        }

        # Reasoning models don't support temperature; use max_completion_tokens
        if self.provider_config.is_reasoning_model(self.model):
            kwargs["temperature"] = openai.NOT_GIVEN
            kwargs["reasoning_effort"] = self.config.extra.get("reasoning_effort", "high")
            kwargs["max_completion_tokens"] = self.config.max_tokens
        else:
            kwargs["temperature"] = self.config.temperature
            kwargs["max_tokens"] = self.config.max_tokens

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        if self.config.stop:
            kwargs["stop"] = self.config.stop

        return self.client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages with optional tool support."""
        parsed_messages = self.parse_messages(messages)
        if reuse_history:
            self.message_history = self.message_history + parsed_messages
        else:
            self.message_history = parsed_messages

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                )
                for tool in tools
            ]

        extra_headers = self.provider_config.get_extra_headers()

        # Apply retry decorator to the API call
        retry_fn = retry_with(
            func=self._create_response,
            provider_name=self.provider_config.get_service_name(),
            max_retries=self.config.max_retries,
        )
        response = retry_fn(tool_schemas, extra_headers)

        choice = response.choices[0]

        tool_calls: list[ToolCall] | None = None
        if choice.message.tool_calls:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=(
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        ),
                    )
                )

        llm_response = LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=(
                LLMUsage(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )
                if response.usage
                else None
            ),
        )

        # Update message history
        if llm_response.tool_calls:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=llm_response.content,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tc.call_id,
                            function=Function(
                                name=tc.name,
                                arguments=json.dumps(tc.arguments),
                            ),
                            type="function",
                        )
                        for tc in llm_response.tool_calls
                    ],
                )
            )
        elif llm_response.content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    content=llm_response.content, role="assistant"
                )
            )

        return llm_response

    def parse_messages(
        self, messages: list[LLMMessage]
    ) -> list[ChatCompletionMessageParam]:
        """Parse LLM messages to OpenAI format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_call is not None:
                _msg_tool_call_handler(openai_messages, msg)
            elif msg.tool_result is not None:
                _msg_tool_result_handler(openai_messages, msg)
            else:
                _msg_role_handler(openai_messages, msg)
        return openai_messages


def _msg_tool_call_handler(
    messages: list[ChatCompletionMessageParam], msg: LLMMessage
) -> None:
    if msg.tool_call:
        messages.append(
            ChatCompletionFunctionMessageParam(
                content=json.dumps(
                    {
                        "name": msg.tool_call.name,
                        "arguments": msg.tool_call.arguments,
                    }
                ),
                role="function",
                name=msg.tool_call.name,
            )
        )


def _msg_tool_result_handler(
    messages: list[ChatCompletionMessageParam], msg: LLMMessage
) -> None:
    if msg.tool_result:
        result: str = ""
        if msg.tool_result.result:
            result = result + msg.tool_result.result + "\n"
        if msg.tool_result.error:
            result += "Tool call failed with error:\n"
            result += msg.tool_result.error
        result = result.strip()
        messages.append(
            ChatCompletionToolMessageParam(
                content=result,
                role="tool",
                tool_call_id=msg.tool_result.call_id,
            )
        )


def _msg_role_handler(
    messages: list[ChatCompletionMessageParam], msg: LLMMessage
) -> None:
    if msg.role:
        match msg.role:
            case "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                messages.append(
                    ChatCompletionSystemMessageParam(
                        content=msg.content, role="system"
                    )
                )
            case "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                messages.append(
                    ChatCompletionUserMessageParam(
                        content=msg.content, role="user"
                    )
                )
            case "assistant":
                if not msg.content:
                    raise ValueError("Assistant message content is required")
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        content=msg.content, role="assistant"
                    )
                )
            case _:
                raise ValueError(f"Invalid message role: {msg.role}")
