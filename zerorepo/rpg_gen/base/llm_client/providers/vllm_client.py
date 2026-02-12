"""VLLM / OpenAI-compatible local endpoint client."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class VLLMProvider(ProviderConfig):
    """VLLM provider configuration."""

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        api_key = config.api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        base_url = config.base_url or "http://localhost:8000/v1"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        return "VLLM"

    def get_provider_name(self) -> str:
        return "vllm"


class VLLMClient(OpenAICompatibleClient):
    """VLLM client using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, VLLMProvider())
