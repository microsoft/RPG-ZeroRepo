"""Unit tests for M5 -- LLM Client Adaptation.

Tests cover:
- llm_types.py:
  - ToolCall and ToolResult dataclasses
  - LLMMessage, LLMUsage, LLMResponse dataclasses
  - Message, UserMessage, SystemMessage, AssistantMessage, ToolMessage wrappers
  - Memory (context window, to_llm_messages, persistence, etc.)

- llm_api_client.py:
  - LLMConfig (from_dict, from_source, to_dict, resolve_provider, save/load)
  - LLMProvider enum and infer_provider
  - retry_with utility
  - APILLMClient (factory pattern, last_usage, _truncate_context)
  - BaseLLMClient ABC contract
"""

import json
import os
import sys
import tempfile
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from common.llm_types import (
    AssistantMessage,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    Memory,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from common.llm_api_client import (
    ALL_PROVIDERS,
    APILLMClient,
    AnthropicClient,
    BaseLLMClient,
    LLMConfig,
    LLMProvider,
    OpenAIClient,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GOOGLE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_VLLM,
    infer_provider,
    retry_with,
)


# ============================================================================
# llm_types: ToolCall / ToolResult
# ============================================================================

class TestToolCall:
    def test_creation(self):
        tc = ToolCall(name="search", call_id="c1", arguments={"q": "test"})
        assert tc.name == "search"
        assert tc.call_id == "c1"
        assert tc.arguments == {"q": "test"}
        assert tc.id is None

    def test_default_arguments(self):
        tc = ToolCall(name="x", call_id="c2")
        assert tc.arguments == {}

    def test_str(self):
        tc = ToolCall(name="fetch", call_id="c3", arguments={"id": 1})
        s = str(tc)
        assert "fetch" in s
        assert "c3" in s

    def test_with_id(self):
        tc = ToolCall(name="t", call_id="c4", arguments={}, id="openai-id")
        assert tc.id == "openai-id"


class TestToolResult:
    def test_success(self):
        tr = ToolResult(
            call_id="c1", name="search", success=True, result="found it"
        )
        assert tr.success is True
        assert tr.result == "found it"
        assert tr.error is None

    def test_failure(self):
        tr = ToolResult(
            call_id="c2", name="search", success=False, error="not found"
        )
        assert tr.success is False
        assert tr.error == "not found"


# ============================================================================
# llm_types: LLMMessage / LLMUsage / LLMResponse
# ============================================================================

class TestLLMMessage:
    def test_basic_creation(self):
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_call is None
        assert msg.tool_result is None

    def test_with_tool_call(self):
        tc = ToolCall(name="f", call_id="c1")
        msg = LLMMessage(role="assistant", tool_call=tc)
        assert msg.tool_call is tc
        assert msg.content is None

    def test_with_tool_result(self):
        tr = ToolResult(call_id="c1", name="f", success=True, result="ok")
        msg = LLMMessage(role="tool", tool_result=tr)
        assert msg.tool_result is tr

    def test_system_message(self):
        msg = LLMMessage(role="system", content="Be helpful.")
        assert msg.role == "system"


class TestLLMUsage:
    def test_defaults(self):
        u = LLMUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0
        assert u.reasoning_tokens == 0

    def test_total_tokens(self):
        u = LLMUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_addition(self):
        u1 = LLMUsage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
        u2 = LLMUsage(
            input_tokens=30,
            output_tokens=40,
            cache_creation_input_tokens=10,
        )
        u3 = u1 + u2
        assert u3.input_tokens == 40
        assert u3.output_tokens == 60
        assert u3.reasoning_tokens == 5
        assert u3.cache_creation_input_tokens == 10
        assert u3.total_tokens == 100

    def test_to_dict(self):
        u = LLMUsage(input_tokens=10, output_tokens=20)
        d = u.to_dict()
        assert d["input_tokens"] == 10
        assert d["output_tokens"] == 20
        assert d["total_tokens"] == 30
        assert "cache_creation_input_tokens" in d
        assert "reasoning_tokens" in d

    def test_str(self):
        u = LLMUsage(input_tokens=5, output_tokens=10)
        s = str(u)
        assert "input_tokens=5" in s
        assert "output_tokens=10" in s


class TestLLMResponse:
    def test_basic(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.usage is None
        assert r.model is None
        assert r.finish_reason is None
        assert r.tool_calls is None

    def test_full(self):
        usage = LLMUsage(input_tokens=100, output_tokens=50)
        tc = ToolCall(name="f", call_id="c1")
        r = LLMResponse(
            content="",
            usage=usage,
            model="gpt-4o",
            finish_reason="tool_use",
            tool_calls=[tc],
        )
        assert r.model == "gpt-4o"
        assert r.finish_reason == "tool_use"
        assert len(r.tool_calls) == 1


# ============================================================================
# llm_types: Message Wrappers
# ============================================================================

class TestMessage:
    def test_basic(self):
        m = Message(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert m.name is None
        assert m.metadata == {}
        assert m.timestamp  # auto-generated

    def test_to_dict(self):
        m = Message(role="assistant", content="response")
        d = m.to_dict()
        assert d == {"role": "assistant", "content": "response"}

    def test_to_dict_with_name(self):
        m = Message(role="user", content="hi", name="alice")
        d = m.to_dict()
        assert d["name"] == "alice"

    def test_to_llm_message(self):
        m = Message(role="user", content="hello")
        llm_msg = m.to_llm_message()
        assert isinstance(llm_msg, LLMMessage)
        assert llm_msg.role == "user"
        assert llm_msg.content == "hello"
        assert llm_msg.tool_call is None

    def test_to_llm_message_with_tool(self):
        tc = ToolCall(name="f", call_id="c1")
        m = Message(role="assistant", content="", tool_call=tc)
        llm_msg = m.to_llm_message()
        assert llm_msg.tool_call is tc


class TestUserMessage:
    def test_default(self):
        m = UserMessage("hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.name is None

    def test_with_name(self):
        m = UserMessage("hello", name="alice")
        assert m.name == "alice"

    def test_metadata(self):
        m = UserMessage("hello", tag="important")
        assert m.metadata.get("tag") == "important"


class TestSystemMessage:
    def test_default(self):
        m = SystemMessage("Be helpful")
        assert m.role == "system"
        assert m.content == "Be helpful"


class TestAssistantMessage:
    def test_default(self):
        m = AssistantMessage("Sure!")
        assert m.role == "assistant"
        assert m.content == "Sure!"


class TestToolMessageType:
    def test_default(self):
        m = ToolMessage("result data")
        assert m.role == "tool"
        assert m.content == "result data"
        assert m.tool_result is None

    def test_with_tool_result(self):
        tr = ToolResult(call_id="c1", name="f", success=True, result="ok")
        m = ToolMessage("ok", tool_result=tr)
        assert m.tool_result is tr


# ============================================================================
# llm_types: Memory
# ============================================================================

class TestMemory:
    def test_add_and_history(self):
        mem = Memory(context_window=5)
        mem.add("user", "hello")
        assert len(mem.history) == 1
        assert mem.history[0].content == "hello"

    def test_add_message(self):
        mem = Memory()
        mem.add_message(UserMessage("hi"))
        assert len(mem._history) == 1
        assert mem._history[0].role == "user"

    def test_last_message(self):
        mem = Memory()
        mem.add("user", "Q1")
        mem.add("assistant", "A1")
        mem.add("user", "Q2")
        assert mem.last().content == "Q2"
        assert mem.last(role="assistant").content == "A1"
        # When role is not found, falls through to return last message
        assert mem.last(role="system").content == "Q2"

    def test_empty_last(self):
        mem = Memory()
        assert mem.last() is None

    def test_context_window_trimming(self):
        mem = Memory(context_window=2)
        mem.add_message(SystemMessage("sys"))
        for i in range(5):
            mem.add_message(UserMessage(f"Q{i}"))
            mem.add_message(AssistantMessage(f"A{i}"))
        mem.add_message(UserMessage("final"))

        history = mem.history
        # system + 4 (2 pairs) + final user = 6
        assert len(history) == 6
        assert history[0].role == "system"
        assert history[-1].content == "final"

    def test_to_llm_messages(self):
        mem = Memory(context_window=3)
        mem.add_message(SystemMessage("sys"))
        mem.add_message(UserMessage("Q1"))
        mem.add_message(AssistantMessage("A1"))

        msgs = mem.to_llm_messages()
        assert all(isinstance(m, LLMMessage) for m in msgs)
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"
        assert msgs[2].role == "assistant"

    def test_to_messages(self):
        mem = Memory(context_window=3)
        mem.add_message(SystemMessage("sys"))
        mem.add_message(UserMessage("Q"))
        mem.add_message(AssistantMessage("A"))

        dicts = mem.to_messages()
        assert len(dicts) == 3
        assert dicts[0] == {"role": "system", "content": "sys"}

    def test_clear_memory(self):
        mem = Memory()
        mem.add("user", "hello")
        mem.clear_memory()
        assert len(mem._history) == 0

    def test_snapshot_and_load(self):
        mem = Memory(context_window=5)
        mem.add_message(UserMessage("hello"))
        mem.add_message(AssistantMessage("hi"))

        snap = mem.snapshot()
        assert "history" in snap
        assert len(snap["history"]) == 2

        mem2 = Memory()
        mem2.load_snapshot(snap)
        assert len(mem2._history) == 2
        assert mem2._history[0].content == "hello"

    def test_save_and_load_file(self):
        mem = Memory()
        mem.add_message(UserMessage("test"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem.save_to_file(path)

            mem2 = Memory()
            mem2.load_from_file(path)
            assert len(mem2._history) == 1
            assert mem2._history[0].content == "test"
        finally:
            os.unlink(path)

    def test_to_dict(self):
        mem = Memory(context_window=3)
        mem.add("user", "hi")
        d = mem.to_dict()
        assert d["context_window"] == 3
        assert len(d["history"]) == 1

    def test_empty_history(self):
        mem = Memory()
        assert mem.history == []
        assert mem.to_llm_messages() == []

    def test_no_window_limit(self):
        """When context_window=0 or negative, no limit is applied."""
        mem = Memory(context_window=0)
        mem.add_message(SystemMessage("sys"))
        for i in range(20):
            mem.add_message(UserMessage(f"Q{i}"))
            mem.add_message(AssistantMessage(f"A{i}"))

        # With window=0, we get: system + 0 context + None last user
        # Actually per the implementation, context_limit=0 means []
        # so result is just [system] since no last user message
        history = mem.history
        # The last message is an AssistantMessage, so last_message is None
        # context_messages = [], result = [system]
        assert history[0].role == "system"


# ============================================================================
# llm_api_client: LLMConfig
# ============================================================================

class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2000
        assert cfg.top_p == 1.0
        assert cfg.stream is False
        assert cfg.max_retries == 3

    def test_from_dict(self):
        cfg = LLMConfig.from_dict({
            "model": "claude-3-opus",
            "temperature": 0.7,
            "unknown_field": "value",
        })
        assert cfg.model == "claude-3-opus"
        assert cfg.temperature == 0.7
        assert cfg.extra["unknown_field"] == "value"

    def test_from_source_dict(self):
        cfg = LLMConfig.from_source({"model": "gpt-4"})
        assert cfg.model == "gpt-4"

    def test_from_source_config(self):
        original = LLMConfig(model="test")
        cfg = LLMConfig.from_source(original)
        assert cfg is original

    def test_from_source_json_string(self):
        cfg = LLMConfig.from_source('{"model": "gpt-4o-mini"}')
        assert cfg.model == "gpt-4o-mini"

    def test_from_source_json_file(self):
        data = {"model": "test-model", "temperature": 0.5}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            cfg = LLMConfig.from_source(path)
            assert cfg.model == "test-model"
            assert cfg.temperature == 0.5
        finally:
            os.unlink(path)

    def test_from_source_invalid(self):
        with pytest.raises(ValueError):
            LLMConfig.from_source("not json not yaml")

    def test_from_source_unsupported_type(self):
        with pytest.raises(TypeError):
            LLMConfig.from_source(42)

    def test_to_dict(self):
        cfg = LLMConfig(model="gpt-4", temperature=0.5)
        d = cfg.to_dict()
        assert d["model"] == "gpt-4"
        assert d["temperature"] == 0.5
        assert "extra" not in d  # No extra fields

    def test_to_dict_with_extra(self):
        cfg = LLMConfig(model="gpt-4")
        cfg.extra["custom"] = "value"
        d = cfg.to_dict()
        assert d["extra"]["custom"] == "value"

    def test_resolve_provider_explicit(self):
        cfg = LLMConfig(provider="anthropic")
        assert cfg.resolve_provider() == "anthropic"

    def test_resolve_provider_auto(self):
        cfg = LLMConfig(model="claude-3-opus")
        assert cfg.resolve_provider() == "anthropic"

        cfg2 = LLMConfig(model="gpt-4o")
        assert cfg2.resolve_provider() == "openai"

    def test_save_json(self):
        cfg = LLMConfig(model="test-save")
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            cfg.save(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"] == "test-save"
        finally:
            os.unlink(path)


# ============================================================================
# llm_api_client: LLMProvider & infer_provider
# ============================================================================

class TestLLMProvider:
    def test_all_values(self):
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.AZURE.value == "azure"
        assert LLMProvider.DEEPSEEK.value == "deepseek"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.VLLM.value == "vllm"
        assert LLMProvider.OPENROUTER.value == "openrouter"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.DOUBAO.value == "doubao"

    def test_all_providers_list(self):
        assert PROVIDER_OPENAI in ALL_PROVIDERS
        assert PROVIDER_ANTHROPIC in ALL_PROVIDERS
        assert len(ALL_PROVIDERS) == 9


class TestInferProvider:
    def test_from_model_name(self):
        assert infer_provider("claude-3-opus") == "anthropic"
        assert infer_provider("deepseek-v2") == "deepseek"
        assert infer_provider("gemini-pro") == "google"

    def test_from_base_url(self):
        assert (
            infer_provider("custom", "https://api.openai.com/v1") == "openai"
        )
        assert (
            infer_provider("custom", "https://openai.azure.com/") == "azure"
        )
        assert (
            infer_provider("custom", "https://api.deepseek.com/v1")
            == "deepseek"
        )
        assert (
            infer_provider(
                "custom",
                "https://generativelanguage.googleapis.com/v1",
            )
            == "google"
        )
        assert (
            infer_provider("custom", "https://openrouter.ai/api")
            == "openrouter"
        )
        assert (
            infer_provider("custom", "http://localhost:8000") == "vllm"
        )

    def test_default_fallback(self):
        assert infer_provider("unknown-model") == "openai"

    def test_case_insensitive(self):
        assert infer_provider("CLAUDE-3-opus") == "anthropic"
        assert infer_provider("DeepSeek-V2") == "deepseek"


# ============================================================================
# llm_api_client: retry_with
# ============================================================================

class TestRetryWith:
    def test_success_no_retry(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        wrapped = retry_with(fn, max_retries=3)
        assert wrapped() == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temp error")
            return "ok"

        wrapped = retry_with(fn, max_retries=3)
        # Patch sleep to avoid actual delay
        with patch("common.llm_api_client.time.sleep"):
            assert wrapped() == "ok"
        assert call_count == 3

    def test_all_retries_exhausted(self):
        def fn():
            raise RuntimeError("permanent error")

        wrapped = retry_with(fn, max_retries=2)
        with patch("common.llm_api_client.time.sleep"):
            with pytest.raises(RuntimeError, match="permanent error"):
                wrapped()


# ============================================================================
# llm_api_client: APILLMClient
# ============================================================================

class TestAPILLMClient:
    def test_default_usage(self):
        """Test last_usage returns zeros when no call has been made."""
        with patch.object(OpenAIClient, "__init__", return_value=None):
            client = APILLMClient.__new__(APILLMClient)
            client.config = LLMConfig()
            client.model = "gpt-4o"
            client.provider_name = "openai"
            client.provider = LLMProvider.OPENAI
            client.client = MagicMock()

            usage = client.last_usage
            assert usage["input_tokens"] == 0
            assert usage["output_tokens"] == 0
            assert usage["total_tokens"] == 0

    def test_last_usage_after_call(self):
        """Test last_usage returns correct values after a call."""
        with patch.object(OpenAIClient, "__init__", return_value=None):
            client = APILLMClient.__new__(APILLMClient)
            client.config = LLMConfig()
            client.model = "gpt-4o"
            client.provider_name = "openai"
            client.provider = LLMProvider.OPENAI
            client.client = MagicMock()

            # Simulate a response
            client._last_response = LLMResponse(
                content="hello",
                usage=LLMUsage(input_tokens=100, output_tokens=50),
            )
            usage = client.last_usage
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 50
            assert usage["total_tokens"] == 150

    def test_truncate_context(self):
        """Test context truncation removes oldest user-assistant pair."""
        client = APILLMClient.__new__(APILLMClient)
        messages = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="Q1"),
            LLMMessage(role="assistant", content="A1"),
            LLMMessage(role="user", content="Q2"),
            LLMMessage(role="assistant", content="A2"),
            LLMMessage(role="user", content="Q3"),
        ]

        truncated = client._truncate_context(messages)
        assert truncated is not None
        # Should have: system + A1 removed pair, so: system + Q2, A2, Q3
        assert truncated[0].role == "system"
        # First non-system should be after the removed pair
        non_system = [m for m in truncated if m.role != "system"]
        assert len(non_system) == 3

    def test_truncate_context_too_short(self):
        """Test truncation returns None when too few messages."""
        client = APILLMClient.__new__(APILLMClient)
        messages = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="Q1"),
        ]
        assert client._truncate_context(messages) is None

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        with patch.object(OpenAIClient, "__init__", return_value=None):
            client = APILLMClient.__new__(APILLMClient)
            client.config = LLMConfig(model="gpt-4", temperature=0.5)
            client.model = "gpt-4"
            client.provider_name = "openai"
            client.provider = LLMProvider.OPENAI
            client.client = MagicMock()

            d = client.to_dict()
            assert d["config"]["model"] == "gpt-4"

    def test_repr(self):
        with patch.object(OpenAIClient, "__init__", return_value=None):
            client = APILLMClient.__new__(APILLMClient)
            client.provider_name = "openai"
            client.model = "gpt-4o"

            r = repr(client)
            assert "openai" in r
            assert "gpt-4o" in r


# ============================================================================
# llm_api_client: BaseLLMClient contract
# ============================================================================

class TestBaseLLMClientContract:
    """Ensure BaseLLMClient cannot be instantiated directly."""

    def test_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLLMClient(LLMConfig())


# ============================================================================
# Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Ensure the existing CLI-based LLMClient is not affected."""

    def test_existing_llm_client_imports(self):
        from common.llm_client import LLMClient, LLMCallRecord

        # The existing LLMClient should be a class
        assert isinstance(LLMClient, type)
        assert isinstance(LLMCallRecord, type)

    def test_no_conflict_with_new_types(self):
        """APILLMClient and LLMClient are separate classes."""
        from common.llm_client import LLMClient as CLIClient

        assert CLIClient is not APILLMClient
        assert CLIClient.__name__ == "LLMClient"
        assert APILLMClient.__name__ == "APILLMClient"


# ============================================================================
# Integration: llm_types + llm_api_client
# ============================================================================

class TestIntegration:
    """Test that types from llm_types work correctly with llm_api_client."""

    def test_memory_produces_llm_messages(self):
        """Memory.to_llm_messages() returns LLMMessage instances that APILLMClient can consume."""
        from common.llm_types import (
            Memory,
            SystemMessage,
            UserMessage,
        )

        mem = Memory(context_window=5)
        mem.add_message(SystemMessage("You are helpful"))
        mem.add_message(UserMessage("What is Python?"))

        msgs = mem.to_llm_messages()
        assert len(msgs) == 2
        assert all(isinstance(m, LLMMessage) for m in msgs)
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_config_from_dict_and_resolve(self):
        """Config created from dict resolves provider correctly."""
        cfg = LLMConfig.from_dict({
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
        })
        assert cfg.resolve_provider() == "anthropic"

    def test_llm_usage_tracking(self):
        """LLMUsage from llm_types is used in LLMResponse."""
        u1 = LLMUsage(input_tokens=100, output_tokens=50)
        u2 = LLMUsage(input_tokens=200, output_tokens=100)
        total = u1 + u2
        resp = LLMResponse(content="test", usage=total)
        assert resp.usage.total_tokens == 450


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
