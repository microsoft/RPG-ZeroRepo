import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .message import Message
from .providers.llm_basics import LLMMessage


class Memory:
    """
    General-purpose conversational memory for LLM agents.
    - Keeps a full `_history` of `Message` objects.
    - Exposes a context-limited `.history` property.
    - `to_llm_messages()` returns `list[LLMMessage]` ready for provider calls.
    """

    def __init__(self, context_window: int = 5):
        """
        Args:
            context_window: number of message pairs (user+assistant) to include
                            in active context. If <= 0, no limit is applied.
        """
        self._history: List[Message] = []
        self.context_window = context_window

    # ============================================================
    # Message Management
    # ============================================================
    def add_message(self, message: Message):
        """Add a `Message` instance to memory."""
        self._history.append(message)

    def add(self, role: str, content: str):
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

    # ============================================================
    # Context Handling
    # ============================================================
    def keep_message_window(self, messages: List[Message]) -> List[Message]:
        """
        Return a context-trimmed view of messages.
        Keeps:
        - the first system message (if any)
        - the most recent N * 2 dialogue messages (user/assistant)
        - the last user message (if exists)
        """
        if not messages:
            return []

        has_system = messages[0].role == "system"
        context_limit = 2 * self.context_window if self.context_window > 0 else 0

        last_message = messages[-1] if messages[-1].role == "user" else None

        start_index = 1 if has_system else 0
        context_messages = messages[start_index:-1] if last_message else messages[start_index:]
        context_messages = context_messages[-context_limit:] if context_limit else []

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
        """Return history as a list of LLMMessage for provider chat() calls."""
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

    # ============================================================
    # Persistence
    # ============================================================
    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of memory."""
        return {"history": [m.__dict__ for m in self._history]}

    def load_snapshot(self, data: Dict[str, Any]):
        """Restore memory from snapshot data."""
        self._history = [Message(**h) for h in data.get("history", [])]

    def save_to_file(self, path: str):
        """Save full memory to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)

    def load_from_file(self, path: str):
        """Load full memory from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_snapshot(data)

    # ============================================================
    # Maintenance
    # ============================================================
    def clear_memory(self):
        """Completely clear the stored conversation history."""
        self._history.clear()

    # ============================================================
    # Display / Debug
    # ============================================================
    def show(self, n: int = 10):
        """Print the latest messages (untrimmed)."""
        print("Memory Snapshot:")
        for m in self._history[-n:]:
            print(f"[{m.timestamp}] {m.role}: {m.content}")

    def to_dict(self) -> Dict[str, Any]:
        """Return the entire memory as a serializable Python dict."""
        return {
            "context_window": self.context_window,
            "history": [m.__dict__ for m in self._history]
        }
