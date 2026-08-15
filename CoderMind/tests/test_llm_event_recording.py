from __future__ import annotations

import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock
from pydantic import BaseModel

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import common.run_events as run_events
import code_gen.sub_agent as sub_agent
from common.llm_api_client import APILLMClient, LLMConfig, LLMProvider
from common.llm_client import (
    LLMClient,
    _load_execution_int,
    _resolve_execution_limit,
    _terminate_cli_process,
)
from common.llm_types import LLMResponse, LLMUsage, Memory
from common.session_manager import CopilotSessionManager, TraceContext


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


def test_cli_process_cleanup_escalates_to_sigkill(monkeypatch):
    class StuckProcess:
        pid = 321
        returncode = None
        wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("copilot", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

        def send_signal(self, signum):
            raise AssertionError(f"unexpected fallback signal {signum}")

        def kill(self):
            raise AssertionError("unexpected direct kill fallback")

    signals = []
    monkeypatch.setattr("common.llm_client._os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        "common.llm_client._os.killpg",
        lambda pgid, signum: signals.append(signum),
    )

    _terminate_cli_process(StuckProcess(), grace_sec=0.1)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_cli_process_signal_falls_back_without_killpg(monkeypatch):
    class FakeProcess:
        pid = 321

        def __init__(self):
            self.actions = []

        def terminate(self):
            self.actions.append("terminate")

        def kill(self):
            self.actions.append("kill")

        def send_signal(self, signum):
            self.actions.append(("signal", signum))

    from common import llm_client

    process = FakeProcess()
    monkeypatch.setattr(llm_client._os, "name", "nt")

    llm_client._signal_process_group(process, signal.SIGTERM)
    llm_client._signal_process_group(process, signal.SIGTERM, force=True)

    assert process.actions == ["terminate", "kill"]


def test_execution_limits_resolve_explicit_env_and_workspace_config(
    tmp_path, monkeypatch,
):
    config_dir = tmp_path / ".cmind"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        "[execution]\nllm_timeout_sec = 120\nllm_max_attempts = 4\n"
    )
    monkeypatch.setattr(
        "common.llm_client._paths._find_workspace_root", lambda: tmp_path,
    )
    monkeypatch.delenv("CMIND_LLM_TIMEOUT_SEC", raising=False)

    assert _load_execution_int("llm_timeout_sec", "CMIND_LLM_TIMEOUT_SEC", 1800) == 120

    monkeypatch.setenv("CMIND_LLM_TIMEOUT_SEC", "45")
    assert _load_execution_int("llm_timeout_sec", "CMIND_LLM_TIMEOUT_SEC", 1800) == 45
    assert _resolve_execution_limit(
        15, "llm_timeout_sec", "CMIND_LLM_TIMEOUT_SEC", 1800,
    ) == 15


def test_cli_call_uses_resolved_environment_timeout(monkeypatch):
    observed = {}

    class CapturingProcess(_FakeProcess):
        def communicate(self, timeout=None):
            observed["timeout"] = timeout
            return super().communicate(timeout=timeout)

    monkeypatch.setenv("CMIND_LLM_TIMEOUT_SEC", "37")
    monkeypatch.setenv("CMIND_LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(
        "common.llm_client.create_session_manager",
        lambda **kwargs: _FakeSessionManager(),
    )
    monkeypatch.setattr(
        "common.llm_client.subprocess.Popen",
        MagicMock(return_value=CapturingProcess()),
    )

    assert LLMClient(tool="copilot").generate("prompt") == "answer"
    assert observed["timeout"] == 37


def test_copilot_model_calls_disable_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("common.session_manager.COPILOT_LOGS_DIR", tmp_path / "logs")
    manager = CopilotSessionManager(project_dir=tmp_path)
    context = TraceContext()

    manager.before(context, "return json")

    assert "--available-tools=" in context.extra_args
    assert "--silent" in context.extra_args
    assert "--disable-builtin-mcps" in context.extra_args
    assert "--no-custom-instructions" in context.extra_args
    assert "--allow-all" not in context.extra_args


def test_copilot_agentic_calls_allow_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("common.session_manager.COPILOT_LOGS_DIR", tmp_path / "logs")
    manager = CopilotSessionManager(project_dir=tmp_path, agentic=True)
    context = TraceContext()

    manager.before(context, "edit code")

    assert "--allow-all" in context.extra_args
    assert "--available-tools=" not in context.extra_args


def test_structured_call_prefers_final_schema_valid_result(monkeypatch):
    class Result(BaseModel):
        value: int

    response = (
        '<result_json>{"status":"error","message":"permission denied"}</result_json>'
        '<result_json>{"value":42}</result_json>'
    )
    client = LLMClient(tool="copilot")
    monkeypatch.setattr(client, "generate", lambda **kwargs: response)
    monkeypatch.setattr(client, "update_last_parsed_result", lambda parsed: None)

    _, result, raw = client.call_structured("system", "user", Result, max_retries=1)

    assert raw == response
    assert result is not None
    assert result.value == 42


def test_parse_result_json_repairs_missing_commas():
    client = LLMClient(tool="copilot")

    result = client.parse_result_json(
        '<result_json>{"items":[{"name":"one"} {"name":"two"}]}</result_json>'
    )

    assert result == {"items": [{"name": "one"}, {"name": "two"}]}


def test_structured_call_normalizes_model_specific_payload(monkeypatch):
    class Result(BaseModel):
        value: int

        @classmethod
        def normalize_llm_payload(cls, payload):
            return {"value": payload["count"]}

    client = LLMClient(tool="copilot")
    monkeypatch.setattr(client, "generate", lambda **kwargs: '<result_json>{"count":7}</result_json>')
    monkeypatch.setattr(client, "update_last_parsed_result", lambda parsed: None)

    _, result, _ = client.call_structured("system", "user", Result, max_retries=1)

    assert result is not None
    assert result.value == 7


def test_code_gen_sub_agent_explicitly_enables_tools(tmp_path, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, prompt, **kwargs):
            return "BATCH_RESULT: PASS"

    monkeypatch.setattr(sub_agent, "LLMClient", FakeClient)

    response, error = sub_agent.dispatch_sub_agent("implement", tmp_path)

    assert error is None
    assert response == "BATCH_RESULT: PASS"
    assert captured["agentic"] is True


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