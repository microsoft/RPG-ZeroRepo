from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from .providers.llm_basics import LLMMessage, ToolCall, ToolResult


@dataclass
class Message:
    """
    General message structure for LLM conversations.
    Roles: system / user / assistant / tool
    """

    role: str
    content: str
    name: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data

    def to_llm_message(self) -> LLMMessage:
        """Convert to LLMMessage for provider calls."""
        return LLMMessage(
            role=self.role,
            content=self.content,
            tool_call=self.tool_call,
            tool_result=self.tool_result,
        )


@dataclass
class UserMessage(Message):
    def __init__(self, content: str, name: Optional[str] = None, **meta):
        super().__init__(role="user", content=content, name=name, metadata=meta)


@dataclass
class SystemMessage(Message):
    def __init__(self, content: str, **meta):
        super().__init__(role="system", content=content, metadata=meta)


@dataclass
class AssistantMessage(Message):
    def __init__(self, content: str, **meta):
        super().__init__(role="assistant", content=content, metadata=meta)


@dataclass
class ToolMessage(Message):
    """Tool result message."""

    def __init__(self, content: str, tool_result: Optional[ToolResult] = None, **meta):
        super().__init__(
            role="tool", content=content, tool_result=tool_result, metadata=meta
        )
