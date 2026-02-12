"""Azure OpenAI client."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import openai
from openai import AzureOpenAI

from .openai_compatible_base import OpenAICompatibleClient, ProviderConfig

if TYPE_CHECKING:
    from ..client import LLMConfig


class AzureProvider(ProviderConfig):
    """Azure OpenAI provider configuration.

    Authentication (in priority order):
        1. Key-based:  api_key in config, or AZURE_OPENAI_API_KEY env var
        2. Azure AD:   tenant_id + token_scope in config, or AZURE_TENANT_ID +
                        AZURE_TOKEN_SCOPE env vars  (requires `az login`)

    Connection (in priority order):
        1. endpoint_url in config, or AZURE_OPENAI_ENDPOINT env var  → azure_endpoint
        2. base_url in config                                         → base_url
    """

    def create_client(self, config: LLMConfig) -> openai.OpenAI:
        # --- connection ---
        azure_endpoint = config.endpoint_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        base_url = config.base_url
        api_version = config.api_version

        conn_kwargs: dict = {"api_version": api_version}
        if azure_endpoint:
            conn_kwargs["azure_endpoint"] = azure_endpoint
        elif base_url:
            conn_kwargs["base_url"] = base_url

        # --- auth: key-based ---
        api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            return AzureOpenAI(api_key=api_key, **conn_kwargs)

        # --- auth: Azure AD token ---
        from azure.identity import AzureCliCredential, get_bearer_token_provider

        tenant_id = config.tenant_id or os.getenv("AZURE_TENANT_ID")
        scope = config.token_scope or os.getenv("AZURE_TOKEN_SCOPE")

        credential = (
            AzureCliCredential(tenant_id=tenant_id)
            if tenant_id
            else AzureCliCredential()
        )
        token_provider = get_bearer_token_provider(credential, scope)
        conn_kwargs["azure_ad_token_provider"] = token_provider

        return AzureOpenAI(**conn_kwargs)

    def get_service_name(self) -> str:
        return "Azure OpenAI"

    def get_provider_name(self) -> str:
        return "azure"

    def is_reasoning_model(self, model: str) -> bool:
        non_reasoning = ("gpt-4o", "gpt-4.1", "DeepSeek-V")
        return not any(tag in model for tag in non_reasoning)


class AzureClient(OpenAICompatibleClient):
    """Azure OpenAI client."""

    def __init__(self, config: LLMConfig):
        super().__init__(config, AzureProvider())
        # Use deployment_name as the model name for API calls if provided
        deployment = config.deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if deployment:
            self.model = deployment
