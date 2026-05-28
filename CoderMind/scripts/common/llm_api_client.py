#!/usr/bin/env python3
"""API-Based LLM Client for CoderMind.

This module provides direct API access to LLM providers as an optional
complement to the existing CLI-based LLM client in ``llm_client.py``.

Ported from RPG-ZeroRepo (zerorepo/rpg_gen/base/llm_client/) with adaptations
for CoderMind's project structure and coding conventions.

Key components:
- LLMConfig: Model configuration for unified LLM access across providers
- BaseLLMClient: Abstract base class for provider implementations
- OpenAICompatibleClient: Shared base for OpenAI-API-compatible providers
- OpenAIClient: OpenAI provider implementation
- AnthropicClient: Anthropic Claude provider implementation
- APILLMClient: High-level unified LLM client (factory/router pattern)

Usage:
    from common.llm_api_client import APILLMClient, LLMConfig

    config = LLMConfig(model="gpt-4o", provider="openai")
    client = APILLMClient(config)
    response = client.generate(memory)
    print(response)
    print(client.last_usage)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from common.llm_types import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    Memory,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Provider Registry
# ============================================================================

class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    DOUBAO = "doubao"


# String constants for backward compatibility
PROVIDER_AZURE = LLMProvider.AZURE.value
PROVIDER_OPENAI = LLMProvider.OPENAI.value
PROVIDER_ANTHROPIC = LLMProvider.ANTHROPIC.value
PROVIDER_DEEPSEEK = LLMProvider.DEEPSEEK.value
PROVIDER_GOOGLE = LLMProvider.GOOGLE.value
PROVIDER_VLLM = LLMProvider.VLLM.value
PROVIDER_OPENROUTER = LLMProvider.OPENROUTER.value
PROVIDER_OLLAMA = LLMProvider.OLLAMA.value
PROVIDER_DOUBAO = LLMProvider.DOUBAO.value

ALL_PROVIDERS = [p.value for p in LLMProvider]

# Model prefix -> provider auto-detection
_MODEL_PREFIX_TO_PROVIDER: Dict[str, str] = {
    "claude": PROVIDER_ANTHROPIC,
    "deepseek": PROVIDER_DEEPSEEK,
    "gemini": PROVIDER_GOOGLE,
}


def infer_provider(model: str, base_url: str | None = None) -> str:
    """Infer provider from model name or base_url.

    Args:
        model: The model name (e.g. "gpt-4o", "claude-3-opus").
        base_url: Optional base URL hint.

    Returns:
        Provider name string (e.g. "openai", "anthropic").
    """
    m = model.lower().strip()
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER.items():
        if m.startswith(prefix):
            return provider
    if base_url:
        u = base_url.lower()
        if "openai.azure" in u or "azure-api" in u:
            return PROVIDER_AZURE
        if "api.openai.com" in u:
            return PROVIDER_OPENAI
        if "deepseek.com" in u:
            return PROVIDER_DEEPSEEK
        if "generativelanguage.googleapis.com" in u:
            return PROVIDER_GOOGLE
        if "openrouter.ai" in u:
            return PROVIDER_OPENROUTER
        if "localhost" in u or "127.0.0.1" in u:
            return PROVIDER_VLLM
    return PROVIDER_OPENAI  # default


# ============================================================================
# Retry Utility
# ============================================================================

def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """Decorator that adds retry logic with randomized backoff.

    Args:
        func: The function to decorate.
        provider_name: The name of the model provider (for logging).
        max_retries: Maximum number of retry attempts.

    Returns:
        Decorated function with retry logic.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries:
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                logger.warning(
                    "%s API call failed: %s. Will sleep for %d seconds "
                    "and retry.\n%s",
                    provider_name,
                    this_error_message,
                    sleep_time,
                    traceback.format_exc(),
                )
                time.sleep(sleep_time)

        raise last_exception or Exception(
            "Retry failed for unknown reason"
        )

    return wrapper


# ============================================================================
# LLMConfig
# ============================================================================

@dataclass
class LLMConfig:
    """Model configuration for unified LLM access across providers.

    Attributes:
        model: Model name (e.g. "gpt-4o", "claude-3-opus").
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling probability.
        stream: Whether to stream the response.
        stop: Stop sequences.
        provider: Provider name (auto-detected from model if not set).
        api_key: API key (falls back to environment variable).
        base_url: Base URL override for the provider.
        endpoint_url: Azure-specific endpoint URL.
        deployment_name: Azure-specific deployment name.
        api_version: Azure API version.
        tenant_id: Azure tenant ID.
        token_scope: Azure token scope.
        max_retries: Maximum retry attempts for API calls.
        log: Whether to log provider initialization and responses.
        extra: Provider-specific params without explicit fields.
    """

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2000
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

    # Provider & connection
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Azure-specific
    endpoint_url: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: str = "2025-01-01-preview"
    tenant_id: Optional[str] = None
    token_scope: Optional[str] = None

    # Retry
    max_retries: int = 3

    log: bool = True

    # Provider-specific params that don't have explicit fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_provider(self) -> str:
        """Return effective provider, auto-detecting from model name if not explicitly set."""
        if self.provider:
            return self.provider
        return infer_provider(
            self.model, self.base_url or self.endpoint_url
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        d: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop": self.stop,
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "endpoint_url": self.endpoint_url,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "tenant_id": self.tenant_id,
            "token_scope": self.token_scope,
            "max_retries": self.max_retries,
            "log": self.log,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMConfig:
        """Create an LLMConfig from a dictionary.

        Unknown keys are placed into the ``extra`` dict.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in data.items():
            if k in valid_fields:
                filtered[k] = v
            else:
                extra[k] = v
        cfg = cls(**filtered)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_source(
        cls, source: Union[str, Dict[str, Any], LLMConfig]
    ) -> LLMConfig:
        """Create an LLMConfig from various source types.

        Supports:
        - LLMConfig instance -> return as-is
        - dict -> from_dict
        - JSON/YAML string -> parse
        - JSON/YAML file path -> read & parse
        """
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)

        if isinstance(source, str):
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = source

            try:
                return cls.from_dict(json.loads(text))
            except json.JSONDecodeError:
                pass

            # Try YAML (optional dependency)
            try:
                import yaml

                parsed = yaml.safe_load(text)
                if isinstance(parsed, dict):
                    return cls.from_dict(parsed)
            except Exception:
                pass

            raise ValueError(
                "Cannot parse config: not valid JSON / YAML / dict / LLMConfig"
            )

        raise TypeError(f"Unsupported config type: {type(source)}")

    def save(self, path: str) -> None:
        """Save configuration to a file (JSON or YAML based on extension)."""
        data = self.to_dict()
        if path.endswith((".yml", ".yaml")):
            import yaml

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# BaseLLMClient (Abstract Provider Base)
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider clients.

    Concrete implementations must provide ``set_chat_history`` and ``chat``.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model: str = config.model.strip()
        self.api_key: str | None = config.api_key
        self.base_url: str | None = config.base_url
        self.api_version: str | None = config.api_version

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM and return a structured response."""


# ============================================================================
# OpenAI Compatible Base
# ============================================================================

class ProviderConfig(ABC):
    """Abstract base class for provider-specific configurations."""

    @abstractmethod
    def create_client(self, config: LLMConfig) -> Any:
        """Create the OpenAI-compatible client instance."""

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name for retry logging."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for identification."""

    def get_extra_headers(self) -> dict[str, str]:
        """Get any extra headers needed for the API call."""
        return {}

    def is_reasoning_model(self, model: str) -> bool:
        """Check if this model is a reasoning model (needs special param handling)."""
        return False


class OpenAICompatibleClient(BaseLLMClient):
    """Base class for OpenAI-compatible clients with shared logic.

    Handles message parsing, tool schemas, retry, and response construction
    for any provider that uses the OpenAI chat completions API format.
    """

    def __init__(
        self, config: LLMConfig, provider_config: ProviderConfig
    ):
        super().__init__(config)
        self.provider_config = provider_config
        self.client = provider_config.create_client(config)
        self.message_history: list = []

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_response(
        self,
        tool_schemas: list | None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        """Create a response using the provider's API."""
        import openai

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
            kwargs["reasoning_effort"] = self.config.extra.get(
                "reasoning_effort", "high"
            )
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
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionMessageToolCallParam,
            ChatCompletionToolParam,
        )
        from openai.types.chat.chat_completion_message_tool_call_param import (
            Function,
        )
        from openai.types.shared_params.function_definition import (
            FunctionDefinition,
        )

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

    def parse_messages(self, messages: list[LLMMessage]) -> list:
        """Parse LLM messages to OpenAI format."""
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionFunctionMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
        )
        from openai.types.chat.chat_completion_tool_message_param import (
            ChatCompletionToolMessageParam,
        )

        openai_messages: list = []
        for msg in messages:
            if msg.tool_call is not None:
                # Tool call message
                if msg.tool_call:
                    openai_messages.append(
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
            elif msg.tool_result is not None:
                # Tool result message
                if msg.tool_result:
                    result_text: str = ""
                    if msg.tool_result.result:
                        result_text = result_text + msg.tool_result.result + "\n"
                    if msg.tool_result.error:
                        result_text += "Tool call failed with error:\n"
                        result_text += msg.tool_result.error
                    result_text = result_text.strip()
                    openai_messages.append(
                        ChatCompletionToolMessageParam(
                            content=result_text,
                            role="tool",
                            tool_call_id=msg.tool_result.call_id,
                        )
                    )
            else:
                # Standard role-based message
                if msg.role == "system":
                    if not msg.content:
                        raise ValueError(
                            "System message content is required"
                        )
                    openai_messages.append(
                        ChatCompletionSystemMessageParam(
                            content=msg.content, role="system"
                        )
                    )
                elif msg.role == "user":
                    if not msg.content:
                        raise ValueError(
                            "User message content is required"
                        )
                    openai_messages.append(
                        ChatCompletionUserMessageParam(
                            content=msg.content, role="user"
                        )
                    )
                elif msg.role == "assistant":
                    if not msg.content:
                        raise ValueError(
                            "Assistant message content is required"
                        )
                    openai_messages.append(
                        ChatCompletionAssistantMessageParam(
                            content=msg.content, role="assistant"
                        )
                    )
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(ProviderConfig):
    """OpenAI provider configuration."""

    def create_client(self, config: LLMConfig) -> Any:
        import openai

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return openai.OpenAI(**kwargs)

    def get_service_name(self) -> str:
        return "OpenAI"

    def get_provider_name(self) -> str:
        return "openai"

    def is_reasoning_model(self, model: str) -> bool:
        return "o3" in model or "o4-mini" in model or "gpt-5" in model


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI client using chat.completions API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, OpenAIProvider())


# ============================================================================
# Anthropic Provider
# ============================================================================

class AnthropicClient(BaseLLMClient):
    """Anthropic client with tool support.

    Uses the Anthropic SDK directly (not OpenAI-compatible).
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import anthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client: anthropic.Anthropic = anthropic.Anthropic(**kwargs)
        self.message_history: list = []
        self.system_message: str | Any = anthropic.NOT_GIVEN

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_anthropic_response(self, tool_schemas: Any) -> Any:
        """Raw API call (decorated with retry by caller)."""
        import anthropic

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
        """Send chat messages with optional tool support."""
        import anthropic

        anthropic_messages = self.parse_messages(messages)
        self.message_history = (
            self.message_history + anthropic_messages
            if reuse_history
            else anthropic_messages
        )

        # Build tool schemas
        tool_schemas: Any = anthropic.NOT_GIVEN
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
    ) -> list:
        """Parse LLMMessage list to Anthropic format."""
        import anthropic

        anthropic_messages: list = []
        for msg in messages:
            if msg.role == "system":
                self.system_message = (
                    msg.content if msg.content else anthropic.NOT_GIVEN
                )
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
                    anthropic.types.MessageParam(
                        role=msg.role, content=msg.content
                    )
                )
        return anthropic_messages

    @staticmethod
    def _parse_tool_call(tool_call: ToolCall) -> Any:
        """Convert ToolCall to Anthropic ToolUseBlockParam."""
        import anthropic

        return anthropic.types.ToolUseBlockParam(
            type="tool_use",
            id=tool_call.call_id,
            name=tool_call.name,
            input=json.dumps(tool_call.arguments),
        )

    @staticmethod
    def _parse_tool_result(tool_result: ToolResult) -> Any:
        """Convert ToolResult to Anthropic ToolResultBlockParam."""
        import anthropic

        result_text: str = ""
        if tool_result.result:
            result_text += tool_result.result + "\n"
        if tool_result.error:
            result_text += "Tool call failed with error:\n" + tool_result.error
        result_text = result_text.strip()
        if not tool_result.success and not result_text:
            result_text = (
                "Tool execution failed without providing error details."
            )

        return anthropic.types.ToolResultBlockParam(
            tool_use_id=tool_result.call_id,
            type="tool_result",
            content=result_text,
            is_error=not tool_result.success,
        )


# ============================================================================
# APILLMClient — Factory / Router
# ============================================================================

class APILLMClient:
    """Unified API-based LLM client supporting multiple providers.

    Factory pattern: lazy-imports the correct provider implementation
    based on the resolved provider name from LLMConfig.

    This is the API-based counterpart to the existing CLI-based
    ``LLMClient`` in ``llm_client.py``.

    Public API:
        - generate(memory) -> Optional[str]
        - call_with_structure_output(memory, response_model) -> (Optional[Dict], str)
    """

    def __init__(
        self,
        config: Optional[Union[LLMConfig, Dict[str, Any], str]] = None,
    ):
        self.config = LLMConfig.from_source(config or {})
        self.model = self.config.model.strip()
        self.provider_name = self.config.resolve_provider()
        self.provider = LLMProvider(self.provider_name)

        # Lazy import -- only the selected provider's SDK is loaded
        match self.provider:
            case LLMProvider.OPENAI:
                self.client: BaseLLMClient = OpenAIClient(self.config)
            case LLMProvider.ANTHROPIC:
                self.client = AnthropicClient(self.config)
            case _:
                # For unsupported providers, attempt OpenAI-compatible
                logger.warning(
                    "Provider '%s' not natively supported, "
                    "falling back to OpenAI-compatible client.",
                    self.provider_name,
                )
                self.client = OpenAIClient(self.config)

        if self.config.log:
            logger.info(
                "Initialized API LLM client: provider='%s' model='%s'",
                self.provider_name,
                self.model,
            )

    # ----------------------------------------------------------------
    # Token usage
    # ----------------------------------------------------------------

    @property
    def last_usage(self) -> Dict[str, int]:
        """Backward-compatible usage dict."""
        resp = getattr(self, "_last_response", None)
        if resp and resp.usage:
            return resp.usage.to_dict()
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ----------------------------------------------------------------
    # Core call
    # ----------------------------------------------------------------

    def _call(self, messages: List[LLMMessage]) -> Optional[str]:
        """Call provider with LLMMessage list, return text."""
        response: LLMResponse = self.client.chat(
            messages, reuse_history=False
        )
        self._last_response = response
        return response.content.strip() if response.content else None

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def generate(
        self,
        memory: Memory,
        max_retries: int = 8,
        retry_delay: float = 20.0,
    ) -> Optional[str]:
        """Generate response from memory context with retry logic.

        Args:
            memory: Conversational memory containing message history.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.

        Returns:
            Response text, or None if all retries fail.
        """
        messages = memory.to_llm_messages()
        retries = 0
        start = time.time()

        while retries < max_retries:
            try:
                result = self._call(messages)

                if self.config.log:
                    duration = round(time.time() - start, 2)
                    logger.info(
                        "Model '%s' response in %ss", self.model, duration
                    )

                if not result:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            "Maximum retries reached (empty result). Aborting."
                        )
                        return None
                    delay = retry_delay + random.uniform(0, 10)
                    logger.warning(
                        "Empty result, retry %d/%d in %.2f seconds...",
                        retries,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                return result

            except Exception as e:
                error_str = str(e).lower()

                if (
                    "context_length_exceeded" in error_str
                    or "context_length" in error_str
                    or "list index out of range" in error_str
                ):
                    messages = self._truncate_context(messages)
                    if messages is None:
                        logger.error(
                            "Context too long and no more messages to "
                            "remove. Aborting."
                        )
                        return None
                    logger.warning(
                        "Context truncated, remaining %d messages. "
                        "Retrying...",
                        len(messages),
                    )
                    continue

                retries += 1
                logger.error(
                    "Error calling model '%s': %s", self.model, e
                )
                if retries >= max_retries:
                    logger.error("Maximum retries reached. Aborting.")
                    return None

                delay = retry_delay + random.uniform(0, 10)
                logger.warning("Retrying in %.2f seconds...", delay)
                time.sleep(delay)

        return None

    def _truncate_context(
        self, messages: List[LLMMessage]
    ) -> Optional[List[LLMMessage]]:
        """Remove oldest user-assistant pair to reduce context length."""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        if len(other_msgs) <= 2:
            return None

        removed_count = 0
        while removed_count < 2 and other_msgs:
            removed_msg = other_msgs.pop(0)
            removed_count += 1
            content_len = (
                len(removed_msg.content) if removed_msg.content else 0
            )
            logger.info(
                "Removed %s message (length: %d chars)",
                removed_msg.role,
                content_len,
            )
            if (
                removed_msg.role == "user"
                and other_msgs
                and other_msgs[0].role == "assistant"
            ):
                removed_msg = other_msgs.pop(0)
                removed_count += 1
                content_len = (
                    len(removed_msg.content) if removed_msg.content else 0
                )
                logger.info(
                    "Removed %s message (length: %d chars)",
                    removed_msg.role,
                    content_len,
                )
                break

        if not other_msgs:
            return None

        return system_msgs + other_msgs

    def call_with_structure_output(
        self,
        memory: Memory,
        response_model: Type[BaseModel],
        max_retries: int = 3,
        retry_delay: float = 40.0,
    ) -> tuple:
        """Generate structured output matching a Pydantic model.

        Args:
            memory: Conversational memory containing message history.
            response_model: Pydantic model class for response validation.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.

        Returns:
            Tuple of (validated_dict, raw_response_string),
            or (None, "") on failure.
        """
        messages = memory.to_llm_messages()

        retries = 0
        start = time.time()

        while retries < max_retries:
            try:
                raw_response = self._call(messages)

                if not raw_response:
                    raise ValueError("Empty response from model")
                text = raw_response.strip()

                # Strip ```json wrapper
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines and lines[0].lstrip().startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].rstrip().startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()

                # Parse JSON
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    start_brace = text.find("{")
                    end_brace = text.rfind("}")
                    if (
                        start_brace != -1
                        and end_brace != -1
                        and end_brace > start_brace
                    ):
                        json_str = text[start_brace : end_brace + 1]
                        data = json.loads(json_str)
                    else:
                        raise

                # Validate with Pydantic (v2 compatible)
                try:
                    model_instance = response_model.model_validate(data)
                except AttributeError:
                    model_instance = response_model.parse_obj(data)

                try:
                    result = model_instance.model_dump()
                except AttributeError:
                    result = model_instance.dict()

                if self.config.log:
                    duration = round(time.time() - start, 2)
                    logger.info(
                        "Structured output from '%s' in %ss",
                        self.model,
                        duration,
                    )

                return result, raw_response

            except Exception as e:
                logger.error(
                    "Error in call_with_structure_output: %s", e
                )
                if retries >= max_retries:
                    logger.error(
                        "Maximum retries reached for structured output."
                    )
                    return None, ""

                delay = retry_delay + random.uniform(0, 10)
                logger.warning(
                    "Retrying structured call in %.2f seconds...", delay
                )
                time.sleep(delay)
            finally:
                retries += 1

        return None, ""

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize client configuration."""
        return {"config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> APILLMClient:
        """Create client from serialized configuration."""
        config = LLMConfig.from_dict(data.get("config", {}))
        return cls(config=config)

    def __repr__(self) -> str:
        return (
            f"<APILLMClient provider='{self.provider_name}' "
            f"model='{self.model}'>"
        )
