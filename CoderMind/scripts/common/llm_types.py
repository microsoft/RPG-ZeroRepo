#!/usr/bin/env python3
"""LLM Type Definitions for CoderMind.

This module provides unified data structures for LLM interactions, including
messages, responses, token usage tracking, and conversational memory.

Ported from RPG-ZeroRepo (zerorepo/rpg_gen/base/llm_client/) with adaptations
for CoderMind's project structure and coding conventions.

Key components:
- LLMMessage: Standard message format for LLM interactions
- LLMResponse: Standard LLM response format
- LLMUsage: Token usage tracking
- ToolCall / ToolResult: Tool calling data structures
- Message / UserMessage / SystemMessage / AssistantMessage / ToolMessage:
  Higher-level message wrappers with metadata and timestamps
- Memory: Conversational memory with context-window management
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Call / Result (used by LLMMessage)
# ============================================================================

ToolCallArguments = dict[str, Any]
"""Type alias for tool call argument dictionaries."""


@dataclass
class ToolCall:
    """A parsed tool call from the model.

    Attributes:
        name: The name of the tool to invoke.
        call_id: A unique identifier for this call.
        arguments: The argument dictionary.
        id: Optional cross-provider identifier (e.g. OpenAI-specific).
    """

    name: str
    call_id: str
    arguments: ToolCallArguments = field(default_factory=dict)
    id: str | None = None  # OpenAI-specific

    def __str__(self) -> str:
        return (
            f"ToolCall(name={self.name}, call_id={self.call_id}, "
            f"arguments={self.arguments})"
        )


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        call_id: Identifier for the tool call this result corresponds to.
        name: The tool name that was invoked.
        success: Whether the execution succeeded.
        result: Textual result on success.
        error: Error message on failure.
        id: Optional cross-provider identifier (e.g. OpenAI-specific).
    """

    call_id: str
    name: str
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None  # OpenAI-specific


# ============================================================================
# Core LLM Types
# ============================================================================

@dataclass
class LLMMessage:
    """Standard message format for LLM interactions.

    Attributes:
        role: The message role (system, user, assistant, tool).
        content: The text content of the message.
        tool_call: Optional parsed tool call from the model.
        tool_result: Optional tool execution result.
    """

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class LLMUsage:
    """Token usage from an LLM call.

    Attributes:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        cache_creation_input_tokens: Tokens used to create cache.
        cache_read_input_tokens: Tokens read from cache.
        reasoning_tokens: Tokens used for reasoning (e.g. o3 models).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens
                + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dictionary."""
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
    """Standard LLM response format.

    Attributes:
        content: The text content of the response.
        usage: Token usage information.
        model: The model that generated the response.
        finish_reason: Why the model stopped generating (e.g. "stop", "tool_use").
        tool_calls: List of tool calls requested by the model.
    """

    content: str
    usage: LLMUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


# ============================================================================
# High-Level Message Wrappers
# ============================================================================

@dataclass
class Message:
    """General message structure for LLM conversations.

    Extends LLMMessage with metadata and timestamps for richer
    conversation tracking.

    Roles: system / user / assistant / tool

    Attributes:
        role: The message role.
        content: The text content.
        name: Optional sender name.
        tool_call: Optional tool call from the model.
        tool_result: Optional tool execution result.
        metadata: Arbitrary metadata dictionary.
        timestamp: ISO-format timestamp (auto-generated).
    """

    role: str
    content: str
    name: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (role + content only)."""
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data

    def to_llm_message(self) -> LLMMessage:
        """Convert to LLMMessage for provider chat() calls."""
        return LLMMessage(
            role=self.role,
            content=self.content,
            tool_call=self.tool_call,
            tool_result=self.tool_result,
        )


@dataclass
class UserMessage(Message):
    """Convenience wrapper for user messages."""

    def __init__(self, content: str, name: Optional[str] = None, **meta: Any):
        super().__init__(role="user", content=content, name=name, metadata=meta)


@dataclass
class SystemMessage(Message):
    """Convenience wrapper for system messages."""

    def __init__(self, content: str, **meta: Any):
        super().__init__(role="system", content=content, metadata=meta)


@dataclass
class AssistantMessage(Message):
    """Convenience wrapper for assistant messages."""

    def __init__(self, content: str, **meta: Any):
        super().__init__(role="assistant", content=content, metadata=meta)


@dataclass
class ToolMessage(Message):
    """Convenience wrapper for tool result messages."""

    def __init__(
        self,
        content: str,
        tool_result: Optional[ToolResult] = None,
        **meta: Any,
    ):
        super().__init__(
            role="tool",
            content=content,
            tool_result=tool_result,
            metadata=meta,
        )


# ============================================================================
# Conversational Memory
# ============================================================================

class Memory:
    """General-purpose conversational memory for LLM agents.

    Keeps a full ``_history`` of ``Message`` objects and exposes a
    context-limited ``.history`` property.  ``to_llm_messages()`` returns
    ``list[LLMMessage]`` ready for provider ``chat()`` calls.

    Args:
        context_window: Number of message pairs (user + assistant) to include
            in active context.  If <= 0, no limit is applied.
    """

    def __init__(self, context_window: int = 5):
        self._history: List[Message] = []
        self.context_window = context_window

    # ----------------------------------------------------------------
    # Message Management
    # ----------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Add a ``Message`` instance to memory."""
        self._history.append(message)

    def add(self, role: str, content: str) -> None:
        """Quickly add a plain message without creating the object manually."""
        self.add_message(Message(role=role, content=content))

    def last(self, role: Optional[str] = None) -> Optional[Message]:
        """Return the most recent message, optionally filtered by role."""
        if not self._history:
            return None
        if role:
            for m in reversed(self._history):
                if m.role == role:
                    return m
        return self._history[-1]

    # ----------------------------------------------------------------
    # Context Handling
    # ----------------------------------------------------------------

    def keep_message_window(
        self, messages: List[Message]
    ) -> List[Message]:
        """Return a context-trimmed view of messages.

        Keeps:
        - the first system message (if any)
        - the most recent N * 2 dialogue messages (user/assistant)
        - the last user message (if exists)
        """
        if not messages:
            return []

        has_system = messages[0].role == "system"
        context_limit = (
            2 * self.context_window if self.context_window > 0 else 0
        )

        last_message = (
            messages[-1] if messages[-1].role == "user" else None
        )

        start_index = 1 if has_system else 0
        context_messages = (
            messages[start_index:-1] if last_message
            else messages[start_index:]
        )
        context_messages = (
            context_messages[-context_limit:] if context_limit else []
        )

        result: List[Message] = []
        if has_system:
            result.append(messages[0])
        result.extend(context_messages)
        if last_message:
            result.append(last_message)
        return result

    @property
    def history(self) -> List[Message]:
        """Expose trimmed history."""
        return self.keep_message_window(self._history)

    def to_llm_messages(self) -> List[LLMMessage]:
        """Return history as ``list[LLMMessage]`` for provider ``chat()`` calls."""
        return [
            m.to_llm_message()
            for m in self.keep_message_window(self._history)
            if m.role in ("system", "user", "assistant", "tool")
        ]

    def to_messages(self) -> List[Dict[str, str]]:
        """Return history as a list of message dicts (backward compatible)."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.keep_message_window(self._history)
            if m.role in ("system", "user", "assistant")
        ]

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of memory."""
        return {"history": [m.__dict__ for m in self._history]}

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        """Restore memory from snapshot data."""
        self._history = [Message(**h) for h in data.get("history", [])]

    def save_to_file(self, path: str) -> None:
        """Save full memory to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)

    def load_from_file(self, path: str) -> None:
        """Load full memory from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_snapshot(data)

    # ----------------------------------------------------------------
    # Maintenance
    # ----------------------------------------------------------------

    def clear_memory(self) -> None:
        """Completely clear the stored conversation history."""
        self._history.clear()

    # ----------------------------------------------------------------
    # Display / Debug
    # ----------------------------------------------------------------

    def show(self, n: int = 10) -> None:
        """Print the latest messages (untrimmed)."""
        logger.info("Memory Snapshot:")
        for m in self._history[-n:]:
            logger.info("[%s] %s: %s", m.timestamp, m.role, m.content)

    def to_dict(self) -> Dict[str, Any]:
        """Return the entire memory as a serializable Python dict."""
        return {
            "context_window": self.context_window,
            "history": [m.__dict__ for m in self._history],
        }
