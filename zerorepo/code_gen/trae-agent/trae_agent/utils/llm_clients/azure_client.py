# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Azure client wrapper with tool integrations"""

import os

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)
from openai import AzureOpenAI


class AzureProvider(ProviderConfig):
    """Azure OpenAI provider configuration.

    All connection parameters come from ModelProvider config or environment variables:
        - api_key          -> ModelProvider.api_key or AZURE_OPENAI_API_KEY
        - base_url         -> ModelProvider.base_url or AZURE_OPENAI_ENDPOINT
        - api_version      -> ModelProvider.api_version or AZURE_API_VERSION
        - tenant_id        -> ModelProvider.tenant_id or AZURE_TENANT_ID
        - token_scope      -> ModelProvider.token_scope or AZURE_TOKEN_SCOPE
    """

    def __init__(self, tenant_id: str | None = None, token_scope: str | None = None):
        self.tenant_id = tenant_id
        self.token_scope = token_scope

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create Azure OpenAI client.

        If api_key is provided, uses key-based authentication.
        Otherwise, falls back to Azure AD token authentication
        using tenant_id/token_scope from config or env vars.
        """
        endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = api_version or os.getenv("AZURE_API_VERSION", "2025-04-01-preview")

        # Shared connection kwargs
        conn_kwargs: dict = {"api_version": api_version}
        if endpoint:
            conn_kwargs["azure_endpoint"] = endpoint

        # Key-based auth
        if api_key:
            return AzureOpenAI(api_key=api_key, **conn_kwargs)

        # Azure AD token auth
        from azure.identity import AzureCliCredential, get_bearer_token_provider

        tenant_id = self.tenant_id or os.getenv("AZURE_TENANT_ID")
        scope = self.token_scope or os.getenv("AZURE_TOKEN_SCOPE")

        credential = (
            AzureCliCredential(tenant_id=tenant_id)
            if tenant_id
            else AzureCliCredential()
        )
        token_provider = get_bearer_token_provider(credential, scope)
        conn_kwargs["azure_ad_token_provider"] = token_provider

        return AzureOpenAI(**conn_kwargs)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Azure OpenAI"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "azure"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Azure-specific headers (none needed)."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        return True


class AzureClient(OpenAICompatibleClient):
    """Azure client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        provider = AzureProvider(
            tenant_id=model_config.model_provider.tenant_id,
            token_scope=model_config.model_provider.token_scope,
        )
        super().__init__(model_config, provider)
