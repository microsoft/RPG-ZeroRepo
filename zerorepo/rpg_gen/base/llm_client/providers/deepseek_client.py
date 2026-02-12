"""Deepseek API client (OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class DeepseekProvider(ProviderConfig):
    """Deepseek provider configuration."""

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = config.base_url or "https://api.deepseek.com"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        return "Deepseek"

    def get_provider_name(self) -> str:
        return "deepseek"


class DeepseekClient(OpenAICompatibleClient):
    """Deepseek client using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, DeepseekProvider())
