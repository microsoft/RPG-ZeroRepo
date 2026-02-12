"""OpenRouter API client (OpenAI-compatible proxy)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class OpenRouterProvider(ProviderConfig):
    """OpenRouter provider configuration."""

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or "https://openrouter.ai/api/v1"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        return "OpenRouter"

    def get_provider_name(self) -> str:
        return "openrouter"

    def get_extra_headers(self) -> dict[str, str]:
        extra_headers: dict[str, str] = {}
        openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
        if openrouter_site_url:
            extra_headers["HTTP-Referer"] = openrouter_site_url
        openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME")
        if openrouter_site_name:
            extra_headers["X-Title"] = openrouter_site_name
        return extra_headers

    def is_reasoning_model(self, model: str) -> bool:
        return (
            "o3" in model
            or "o4-mini" in model
            or "gpt-5" in model
        )


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter client using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        # Default base_url if not set
        if not config.base_url:
            config.base_url = "https://openrouter.ai/api/v1"
        super().__init__(config, OpenRouterProvider())
