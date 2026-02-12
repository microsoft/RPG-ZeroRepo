"""LLM provider implementations.

Architecture follows trae-agent's llm_clients pattern:
- BaseLLMClient: abstract base
- OpenAICompatibleClient: shared base for OpenAI-API-compatible providers
- Individual *_client.py per provider
- retry_utils: retry decorator
"""

from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    DOUBAO = "doubao"


# String constants for backward compatibility
PROVIDER_AZURE = LLMProvider.AZURE.value
PROVIDER_OPENAI = LLMProvider.OPENAI.value
PROVIDER_ANTHROPIC = LLMProvider.ANTHROPIC.value
PROVIDER_DEEPSEEK = LLMProvider.DEEPSEEK.value
PROVIDER_GOOGLE = LLMProvider.GOOGLE.value
PROVIDER_VLLM = LLMProvider.VLLM.value
PROVIDER_OPENROUTER = LLMProvider.OPENROUTER.value
PROVIDER_OLLAMA = LLMProvider.OLLAMA.value
PROVIDER_DOUBAO = LLMProvider.DOUBAO.value

ALL_PROVIDERS = [p.value for p in LLMProvider]

# Model prefix -> provider auto-detection
_MODEL_PREFIX_TO_PROVIDER = {
    "claude": PROVIDER_ANTHROPIC,
    "deepseek": PROVIDER_DEEPSEEK,
    "gemini": PROVIDER_GOOGLE,
}


def infer_provider(model: str, base_url: str | None = None) -> str:
    """Infer provider from model name or base_url."""
    m = model.lower().strip()
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER.items():
        if m.startswith(prefix):
            return provider
    if base_url:
        u = base_url.lower()
        if "openai.azure" in u or "azure-api" in u:
            return PROVIDER_AZURE
        if "api.openai.com" in u:
            return PROVIDER_OPENAI
        if "deepseek.com" in u:
            return PROVIDER_DEEPSEEK
        if "generativelanguage.googleapis.com" in u:
            return PROVIDER_GOOGLE
        if "openrouter.ai" in u:
            return PROVIDER_OPENROUTER
        if "localhost" in u or "127.0.0.1" in u:
            return PROVIDER_VLLM
    return PROVIDER_AZURE  # default for gpt-*, o3-*, etc.
