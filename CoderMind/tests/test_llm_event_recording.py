from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import common.run_events as run_events
from common.llm_api_client import APILLMClient, LLMConfig, LLMProvider
from common.llm_client import LLMClient
from common.llm_types import LLMResponse, LLMUsage, Memory
from common.session_manager import TraceContext


class _FakeProcess:
    returncode = 0
    pid = 123

    def communicate(self, timeout=None):
        return "answer", ""


class _FakeSessionManager:
    @contextmanager
    def trace(self, prompt: str, purpose: str):
        context = TraceContext()
        yield context
        context.captured_path = Path("/workspace/logs/copilot/process-1.log")


def test_cli_call_records_context_and_log_file(tmp_path, monkeypatch):
    events_file = tmp_path / "run_events.jsonl"
    monkeypatch.setattr(run_events, "EVENTS_FILE", events_file)
    monkeypatch.setattr("common.llm_client.create_session_manager", lambda **kwargs: _FakeSessionManager())
    popen = MagicMock(return_value=_FakeProcess())
    monkeypatch.setattr("common.llm_client.subprocess.Popen", popen)

    client = LLMClient(tool="copilot")
    with run_events.record_run("encode") as run:
        with run_events.record_stage(run.run_id, "encode", "parse_rpg") as stage:
            assert client.generate("secret prompt", purpose="parse_features") == "answer"

    llm_event = next(event for event in run_events.load_events(events_file) if event["event_type"] == "llm_call")
    assert llm_event["run_id"] == run.run_id
    assert llm_event["stage_id"] == stage.stage_id
    assert llm_event["provider"] == "copilot"
    assert llm_event["log_file"] == "process-1.log"
    assert llm_event["token_status"] == "available_in_workspace_log"
    assert "prompt" not in llm_event and "response" not in llm_event
    child_env = popen.call_args.kwargs["env"]
    assert child_env["CMIND_RUN_ID"] == run.run_id
    assert child_env["CMIND_STAGE_ID"] == stage.stage_id


def test_api_call_records_exact_usage(tmp_path, monkeypatch):
    events_file = tmp_path / "run_events.jsonl"
    monkeypatch.setattr(run_events, "EVENTS_FILE", events_file)
    client = APILLMClient.__new__(APILLMClient)
    client.config = LLMConfig()
    client.model = "gpt-test"
    client.provider_name = "openai"
    client.provider = LLMProvider.OPENAI
    client.client = MagicMock()
    client.client.chat.return_value = LLMResponse(
        content="answer",
        model="gpt-test-versioned",
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=40,
            reasoning_tokens=5,
        ),
    )

    with run_events.record_run("plan") as run:
        with run_events.record_stage(run.run_id, "plan", "build_skeleton") as stage:
            assert client.generate(Memory(), max_retries=1) == "answer"

    llm_event = next(event for event in run_events.load_events(events_file) if event["event_type"] == "llm_call")
    assert llm_event["stage_id"] == stage.stage_id
    assert llm_event["provider"] == "openai"
    assert llm_event["model"] == "gpt-test-versioned"
    assert llm_event["tokens"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 40,
        "reasoning_tokens": 5,
    }
    assert llm_event["token_status"] == "measured"