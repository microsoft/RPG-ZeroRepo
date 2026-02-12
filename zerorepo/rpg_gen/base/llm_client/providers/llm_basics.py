"""Core data structures for LLM interactions."""

from dataclasses import dataclass, field
from typing import Any


# Type alias
ToolCallArguments = dict[str, Any]


@dataclass
class ToolCall:
    """A parsed tool call from the model."""

    name: str
    call_id: str
    arguments: ToolCallArguments = field(default_factory=dict)
    id: str | None = None  # OpenAI-specific

    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, call_id={self.call_id}, arguments={self.arguments})"


@dataclass
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    name: str
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None  # OpenAI-specific


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions."""

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class LLMUsage:
    """Token usage from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def __str__(self) -> str:
        return (
            f"LLMUsage(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"cache_creation_input_tokens={self.cache_creation_input_tokens}, "
            f"cache_read_input_tokens={self.cache_read_input_tokens}, "
            f"reasoning_tokens={self.reasoning_tokens})"
        )


@dataclass
class LLMResponse:
    """Standard LLM response format."""

    content: str
    usage: LLMUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
