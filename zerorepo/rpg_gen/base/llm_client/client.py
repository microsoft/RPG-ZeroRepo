"""LLM Client — thin factory/router for multiple providers.

Architecture follows trae-agent's pattern:
  - LLMConfig holds all configuration
  - LLMClient dispatches to the correct provider via lazy imports
  - Each provider is an independent class in providers/
"""

import os
import time
import json
import yaml
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel

from .memory import Memory
from .providers import (
    LLMProvider,
    PROVIDER_AZURE,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE,
    PROVIDER_VLLM,
    PROVIDER_OPENROUTER,
    PROVIDER_OLLAMA,
    PROVIDER_DOUBAO,
    ALL_PROVIDERS,
    infer_provider,
)
from .providers.base_client import BaseLLMClient
from .providers.llm_basics import LLMMessage, LLMResponse, LLMUsage
from zerorepo.utils.api import parse_thinking_output


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Model configuration for unified LLM access across providers."""

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2000
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

    # Provider & connection
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Azure-specific
    endpoint_url: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: str = "2025-01-01-preview"
    tenant_id: Optional[str] = None
    token_scope: Optional[str] = None

    # Retry
    max_retries: int = 3

    log: bool = True

    # Provider-specific params that don't have explicit fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_provider(self) -> str:
        """Return effective provider, auto-detecting from model name if not explicitly set."""
        if self.provider:
            return self.provider
        return infer_provider(self.model, self.base_url or self.endpoint_url)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop": self.stop,
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "endpoint_url": self.endpoint_url,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "tenant_id": self.tenant_id,
            "token_scope": self.token_scope,
            "max_retries": self.max_retries,
            "log": self.log,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        extra: Dict[str, Any] = {}
        for k, v in data.items():
            if k in valid_fields:
                filtered[k] = v
            else:
                extra[k] = v
        cfg = cls(**filtered)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_source(cls, source: Union[str, Dict[str, Any], "LLMConfig"]) -> "LLMConfig":
        """
        Supports:
        - LLMConfig instance -> return as-is
        - dict -> from_dict
        - JSON/YAML string -> parse
        - JSON/YAML file path -> read & parse
        """
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)

        if isinstance(source, str):
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = source

            try:
                return cls.from_dict(json.loads(text))
            except json.JSONDecodeError:
                pass

            try:
                parsed = yaml.safe_load(text)
                if isinstance(parsed, dict):
                    return cls.from_dict(parsed)
            except Exception:
                pass

            raise ValueError("Cannot parse config: not valid JSON / YAML / dict / LLMConfig")

        raise TypeError(f"Unsupported config type: {type(source)}")

    def save(self, path: str):
        data = self.to_dict()
        if path.endswith((".yml", ".yaml")):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLMClient — factory / router
# ---------------------------------------------------------------------------
class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Factory pattern (trae-agent style): lazy-imports the correct provider
    implementation based on resolved provider name.

    Public API:
        - generate(memory) -> Optional[str]
        - call_with_structure_output(memory, response_model) -> (Optional[Dict], str)
    """

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any], str]] = None):
        self.config = LLMConfig.from_source(config or {})
        self.model = self.config.model.strip()
        self.provider_name = self.config.resolve_provider()
        self.provider = LLMProvider(self.provider_name)

        # Lazy import — only the selected provider's SDK is loaded
        match self.provider:
            case LLMProvider.OPENAI:
                from .providers.openai_client import OpenAIClient
                self.client: BaseLLMClient = OpenAIClient(self.config)
            case LLMProvider.ANTHROPIC:
                from .providers.anthropic_client import AnthropicClient
                self.client = AnthropicClient(self.config)
            case LLMProvider.AZURE:
                from .providers.azure_client import AzureClient
                self.client = AzureClient(self.config)
            case LLMProvider.DEEPSEEK:
                from .providers.deepseek_client import DeepseekClient
                self.client = DeepseekClient(self.config)
            case LLMProvider.GOOGLE:
                from .providers.google_client import GoogleClient
                self.client = GoogleClient(self.config)
            case LLMProvider.VLLM:
                from .providers.vllm_client import VLLMClient
                self.client = VLLMClient(self.config)
            case LLMProvider.OPENROUTER:
                from .providers.openrouter_client import OpenRouterClient
                self.client = OpenRouterClient(self.config)
            case LLMProvider.OLLAMA:
                from .providers.ollama_client import OllamaClient
                self.client = OllamaClient(self.config)
            case LLMProvider.DOUBAO:
                from .providers.doubao_client import DoubaoClient
                self.client = DoubaoClient(self.config)

        if self.config.log:
            print(f"[LLMClient] Initialized provider='{self.provider_name}' model='{self.model}'")

    # ===================================================================
    # Token usage (delegated)
    # ===================================================================
    @property
    def last_usage(self) -> Dict[str, int]:
        """Backward-compatible usage dict."""
        resp = getattr(self, "_last_response", None)
        if resp and resp.usage:
            return resp.usage.to_dict()
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ===================================================================
    # Core call
    # ===================================================================
    def _call(self, messages: List[LLMMessage]) -> Optional[str]:
        """Call provider with LLMMessage list, return text."""
        response: LLMResponse = self.client.chat(messages, reuse_history=False)
        self._last_response = response
        return response.content.strip() if response.content else None

    # ===================================================================
    # Public API (unchanged interface)
    # ===================================================================
    def generate(
        self, memory: Memory, max_retries: int = 8, retry_delay: float = 20.0
    ) -> Optional[str]:
        """Generate response from memory context with retry logic."""
        messages = memory.to_llm_messages()
        retries = 0
        start = time.time()
        
        while retries < max_retries:
            try:
                result = self._call(messages)

                if self.config.log:
                    duration = round(time.time() - start, 2)
                    print(f"[LLMClient] Model '{self.model}' response in {duration}s")

                if not result:
                    retries += 1
                    if retries >= max_retries:
                        print("[LLMClient] Maximum retries reached (empty result). Aborting.")
                        return None
                    delay = retry_delay + random.uniform(0, 10)
                    print(
                        f"[LLMClient] Empty result, retry {retries}/{max_retries} "
                        f"in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                    continue

                return result

            except Exception as e:
                error_str = str(e).lower()

                if (
                    "context_length_exceeded" in error_str
                    or "context_length" in error_str
                    or "list index out of range" in error_str
                ):
                    messages = self._truncate_context(messages)
                    if messages is None:
                        print("[LLMClient] Context too long and no more messages to remove. Aborting.")
                        return None
                    print(
                        f"[LLMClient] Context truncated, remaining {len(messages)} messages. Retrying..."
                    )
                    continue

                retries += 1
                print(f"[LLMClient] Error calling model '{self.model}': {e}")
                if retries >= max_retries:
                    print("[LLMClient] Maximum retries reached. Aborting.")
                    return None

                delay = retry_delay + random.uniform(0, 10)
                print(f"[LLMClient] Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

        return None

    def _truncate_context(
        self, messages: List[LLMMessage]
    ) -> Optional[List[LLMMessage]]:
        """Remove oldest user-assistant pair to reduce context length."""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        if len(other_msgs) <= 2:
            return None

        removed_count = 0
        while removed_count < 2 and other_msgs:
            removed_msg = other_msgs.pop(0)
            removed_count += 1
            content_len = len(removed_msg.content) if removed_msg.content else 0
            print(
                f"[LLMClient] Removed {removed_msg.role} message "
                f"(length: {content_len} chars)"
            )
            if (
                removed_msg.role == "user"
                and other_msgs
                and other_msgs[0].role == "assistant"
            ):
                removed_msg = other_msgs.pop(0)
                removed_count += 1
                content_len = len(removed_msg.content) if removed_msg.content else 0
                print(
                    f"[LLMClient] Removed {removed_msg.role} message "
                    f"(length: {content_len} chars)"
                )
                break

        if not other_msgs:
            return None

        return system_msgs + other_msgs

    def call_with_structure_output(
        self,
        memory: Memory,
        response_model: Type[BaseModel],
        max_retries: int = 3,
        retry_delay: float = 40.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate structured output matching a Pydantic model.

        Returns:
            Tuple of (validated_dict, raw_response_string), or (None, "") on failure.
        """
        messages = memory.to_llm_messages()

        retries = 0
        start = time.time()

        while retries < max_retries:
            try:
                raw_response = self._call(messages)

                if not raw_response:
                    raise ValueError("Empty response from model")
                raw = parse_thinking_output(raw_response)
                text = raw.strip()

                # Strip ```json wrapper
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines and lines[0].lstrip().startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].rstrip().startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()

                # Parse JSON
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    start_brace = text.find("{")
                    end_brace = text.rfind("}")
                    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                        json_str = text[start_brace : end_brace + 1]
                        data = json.loads(json_str)
                    else:
                        raise

                # Validate with Pydantic (v1 / v2 compatible)
                try:
                    model_instance = response_model.model_validate(data)
                except AttributeError:
                    model_instance = response_model.parse_obj(data)

                try:
                    result = model_instance.model_dump()
                except AttributeError:
                    result = model_instance.dict()

                if self.config.log:
                    duration = round(time.time() - start, 2)
                    print(f"[LLMClient] Structured output from '{self.model}' in {duration}s")

                return result, raw_response

            except Exception as e:
                print(f"[LLMClient] Error in call_with_structure_output: {e}")
                if retries >= max_retries:
                    print("[LLMClient] Maximum retries reached for structured output.")
                    return None, ""

                delay = retry_delay + random.uniform(0, 10)
                print(f"[LLMClient] Retrying structured call in {delay:.2f} seconds...")
                time.sleep(delay)
            finally:
                retries += 1

        return None, ""

    # ===================================================================
    # Serialization
    # ===================================================================
    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMClient":
        config = LLMConfig.from_dict(data.get("config", {}))
        return cls(config=config)

    def __repr__(self):
        return f"<LLMClient provider='{self.provider_name}' model='{self.model}'>"
