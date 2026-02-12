"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .llm_basics import LLMMessage, LLMResponse

if TYPE_CHECKING:
    from ..client import LLMConfig


class BaseLLMClient(ABC):
    """Base class for all LLM provider clients."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model: str = config.model.strip()
        self.api_key: str | None = config.api_key
        self.base_url: str | None = config.base_url
        self.api_version: str | None = config.api_version

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM and return a structured response."""
        pass
