"""Doubao (ByteDance) API client (OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class DoubaoProvider(ProviderConfig):
    """Doubao provider configuration."""

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        api_key = config.api_key or os.getenv("DOUBAO_API_KEY")
        base_url = config.base_url
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def get_service_name(self) -> str:
        return "Doubao"

    def get_provider_name(self) -> str:
        return "doubao"


class DoubaoClient(OpenAICompatibleClient):
    """Doubao client using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, DoubaoProvider())
