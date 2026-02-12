"""OpenAI API client."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class OpenAIProvider(ProviderConfig):
    """OpenAI provider configuration."""

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
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
        return (
            "o3" in model
            or "o4-mini" in model
            or "gpt-5" in model
        )


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI client using chat.completions API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, OpenAIProvider())
