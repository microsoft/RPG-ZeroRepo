"""Unit tests for M3 — Tool Abstraction Layer (scripts/common/tools.py).

Tests cover:
- Error hierarchy (ToolError, ToolNotFoundError, ToolValidationError, ToolExecutionError)
- ToolExecResult and ToolResult dataclasses
- ToolCall dataclass (creation, __str__, to_dict)
- ToolParameter base model
- Tool ABC (check with/without ParamModel, lifecycle hooks)
- ToolExecutor (register, execute_tool_call, parallel/sequential, concurrency limit)
- ToolHandler (parse_and_match_tool, register/unregister, describe)
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common.tools import (
    Tool,
    ToolCall,
    ToolCallArguments,
    ToolError,
    ToolExecResult,
    ToolExecutionError,
    ToolExecutor,
    ToolHandler,
    ToolNotFoundError,
    ToolParameter,
    ToolResult,
    ToolValidationError,
)


# ──────────────────────────────────────────────────────────────
# Helper: run async functions in sync tests
# ──────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ──────────────────────────────────────────────────────────────
# Helper Tool Implementations for Testing
# ──────────────────────────────────────────────────────────────

class EchoParams(BaseModel):
    """Parameter model for the echo tool."""
    message: str
    repeat: int = 1


class EchoTool(Tool):
    """Simple echo tool that returns the message repeated N times."""
    ParamModel = EchoParams
    name = "echo"
    description = "Echo a message back."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        msg = arguments.get("message", "")
        repeat = arguments.get("repeat", 1)
        return ToolExecResult(output=msg * repeat)

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        if "echo:" in raw.lower():
            text = raw.split("echo:", 1)[1].strip()
            return {"message": text, "repeat": 1}
        return None


class FailingTool(Tool):
    """Tool that always returns an error."""
    name = "fail"
    description = "A tool that always fails."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        return ToolExecResult(error="intentional failure", error_code=1)

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        return None


class ExceptionTool(Tool):
    """Tool that raises an exception during execute."""
    name = "explode"
    description = "A tool that raises an exception."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        raise RuntimeError("boom")

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        return None


class ToolErrorTool(Tool):
    """Tool that raises a ToolExecutionError during execute."""
    name = "tool_error"
    description = "A tool that raises ToolExecutionError."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        raise ToolExecutionError("controlled failure")

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        return None


class NoParamTool(Tool):
    """Tool without a ParamModel -- raw dict passthrough."""
    name = "no_param"
    description = "A tool with no parameter model."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        return ToolExecResult(output="ok")

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        if "noparam" in raw.lower():
            return {"trigger": "yes"}  # non-empty so handler does not skip
        return None


class StatefulTool(Tool):
    """Tool that returns state in the result."""
    name = "stateful"
    description = "A tool that carries state."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        return ToolExecResult(
            output="done",
            state={"step": arguments.get("step", 0)},
        )

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        return None


class MultiParseTool(Tool):
    """Tool whose custom_parse returns multiple argument dicts."""
    ParamModel = EchoParams
    name = "multi_parse"
    description = "A tool that can parse multiple calls."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        return ToolExecResult(output=arguments.get("message", ""))

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[List[ToolCallArguments]]:
        if "multi:" not in raw.lower():
            return None
        parts = raw.split("multi:", 1)[1].strip().split(",")
        return [{"message": p.strip(), "repeat": 1} for p in parts if p.strip()]


class BadParseTool(Tool):
    """Tool whose custom_parse raises an exception."""
    name = "bad_parse"
    description = "A tool with broken parsing."

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        return ToolExecResult(output="ok")

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        raise ValueError("parse exploded")


# ──────────────────────────────────────────────────────────────
# Tests — Error Hierarchy
# ──────────────────────────────────────────────────────────────

class TestErrorHierarchy:
    """Verify the error class hierarchy."""

    def test_tool_error_is_exception(self):
        assert issubclass(ToolError, Exception)

    def test_not_found_is_tool_error(self):
        assert issubclass(ToolNotFoundError, ToolError)

    def test_validation_is_tool_error(self):
        assert issubclass(ToolValidationError, ToolError)

    def test_execution_is_tool_error(self):
        assert issubclass(ToolExecutionError, ToolError)

    def test_catch_as_tool_error(self):
        """All specific errors can be caught as ToolError."""
        for cls in (ToolNotFoundError, ToolValidationError, ToolExecutionError):
            with pytest.raises(ToolError):
                raise cls("test")


# ──────────────────────────────────────────────────────────────
# Tests — Data Types
# ──────────────────────────────────────────────────────────────

class TestToolExecResult:
    """Tests for ToolExecResult."""

    def test_defaults(self):
        r = ToolExecResult()
        assert r.output is None
        assert r.error is None
        assert r.error_code == 0
        assert r.state is None

    def test_success_result(self):
        r = ToolExecResult(output="hello", error_code=0)
        assert r.output == "hello"
        assert r.error_code == 0

    def test_error_result(self):
        r = ToolExecResult(error="fail", error_code=1)
        assert r.error == "fail"
        assert r.error_code == 1

    def test_state_carried(self):
        r = ToolExecResult(output="ok", state={"key": "val"})
        assert r.state == {"key": "val"}


class TestToolResult:
    """Tests for ToolResult."""

    def test_success(self):
        r = ToolResult(name="t", success=True, result="ok")
        assert r.name == "t"
        assert r.success is True
        assert r.result == "ok"
        assert r.error is None

    def test_failure(self):
        r = ToolResult(name="t", success=False, error="bad")
        assert r.success is False
        assert r.error == "bad"

    def test_optional_fields(self):
        r = ToolResult(name="t", success=True, call_id="c1", id="i1", state={"x": 1})
        assert r.call_id == "c1"
        assert r.id == "i1"
        assert r.state == {"x": 1}


class TestToolCall:
    """Tests for ToolCall."""

    def test_creation(self):
        tc = ToolCall(name="echo", call_id="c1", arguments={"msg": "hi"})
        assert tc.name == "echo"
        assert tc.call_id == "c1"
        assert tc.arguments == {"msg": "hi"}
        assert tc.id is None

    def test_str_representation(self):
        tc = ToolCall(name="echo", call_id="c1", arguments={"a": 1})
        s = str(tc)
        assert "echo" in s
        assert "c1" in s

    def test_to_dict(self):
        tc = ToolCall(name="echo", call_id="c1", arguments={"a": 1})
        d = tc.to_dict()
        assert d == {
            "call_id": "c1",
            "name": "echo",
            "arguments": {"a": 1},
        }

    def test_optional_id(self):
        tc = ToolCall(name="x", call_id="c", arguments={}, id="openai_123")
        assert tc.id == "openai_123"


class TestToolParameter:
    """Tests for ToolParameter base model."""

    def test_is_base_model(self):
        assert issubclass(ToolParameter, BaseModel)

    def test_instantiation(self):
        tp = ToolParameter()
        assert tp is not None

    def test_subclassing(self):
        class MyParams(ToolParameter):
            x: int = 0
        p = MyParams(x=42)
        assert p.x == 42


# ──────────────────────────────────────────────────────────────
# Tests — Tool ABC
# ──────────────────────────────────────────────────────────────

class TestToolABC:
    """Tests for the Tool abstract base class."""

    def test_get_name(self):
        assert EchoTool.get_name() == "echo"

    def test_get_description(self):
        assert EchoTool.get_description() == "Echo a message back."

    def test_check_with_param_model(self):
        """check() validates and normalizes arguments via ParamModel."""
        result = run_async(EchoTool.check({"message": "hi", "repeat": 2}))
        assert result == {"message": "hi", "repeat": 2}

    def test_check_with_defaults(self):
        """check() fills in defaults from ParamModel."""
        result = run_async(EchoTool.check({"message": "hi"}))
        assert result == {"message": "hi", "repeat": 1}

    def test_check_validation_error(self):
        """check() raises ToolValidationError for invalid arguments."""
        with pytest.raises(ToolValidationError):
            run_async(EchoTool.check({"repeat": 2}))  # missing required 'message'

    def test_check_no_param_model(self):
        """check() passes through raw dict when ParamModel is None."""
        args = {"anything": "goes"}
        result = run_async(NoParamTool.check(args))
        assert result == args

    def test_execute_echo(self):
        """execute() returns output correctly."""
        result = run_async(EchoTool.execute({"message": "hi", "repeat": 3}))
        assert result.output == "hihihi"
        assert result.error_code == 0

    def test_lifecycle_hooks_default(self):
        """Default lifecycle hooks return None without error."""
        run_async(EchoTool.before_execute({}))
        run_async(EchoTool.after_execute({}, ToolExecResult()))

    def test_close_default(self):
        """Default close() returns None without error."""
        run_async(EchoTool.close())


# ──────────────────────────────────────────────────────────────
# Tests — ToolExecutor
# ──────────────────────────────────────────────────────────────

class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_register_and_list(self):
        executor = ToolExecutor(tools=[EchoTool, FailingTool])
        names = executor.list_tools()
        assert "echo" in names
        assert "fail" in names

    def test_register_duplicate_raises(self):
        executor = ToolExecutor(tools=[EchoTool])
        with pytest.raises(ValueError, match="already registered"):
            executor.register(EchoTool)

    def test_tools_property(self):
        executor = ToolExecutor(tools=[EchoTool])
        assert EchoTool in executor.tools

    def test_execute_success(self):
        executor = ToolExecutor(tools=[EchoTool])
        call = ToolCall(name="echo", call_id="c1", arguments={"message": "hi", "repeat": 2})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is True
        assert result.result == "hihi"
        assert result.name == "echo"
        assert result.call_id == "c1"

    def test_execute_failure(self):
        executor = ToolExecutor(tools=[FailingTool])
        call = ToolCall(name="fail", call_id="c2", arguments={})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is False
        assert result.error == "intentional failure"

    def test_execute_not_found(self):
        executor = ToolExecutor(tools=[EchoTool])
        call = ToolCall(name="nonexistent", call_id="c3", arguments={})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_unhandled_exception(self):
        executor = ToolExecutor(tools=[ExceptionTool])
        call = ToolCall(name="explode", call_id="c4", arguments={})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is False
        assert "boom" in result.error

    def test_execute_tool_error(self):
        executor = ToolExecutor(tools=[ToolErrorTool])
        call = ToolCall(name="tool_error", call_id="c5", arguments={})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is False
        assert "controlled failure" in result.error

    def test_execute_validation_error(self):
        """Validation error during check() is caught gracefully."""
        executor = ToolExecutor(tools=[EchoTool])
        # Missing required 'message' field
        call = ToolCall(name="echo", call_id="c6", arguments={"repeat": 2})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is False

    def test_execute_with_state(self):
        executor = ToolExecutor(tools=[StatefulTool])
        call = ToolCall(name="stateful", call_id="c7", arguments={"step": 5})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is True
        assert result.state == {"step": 5}

    def test_execute_preserves_call_id_and_id(self):
        executor = ToolExecutor(tools=[NoParamTool])
        call = ToolCall(name="no_param", call_id="c8", arguments={}, id="openai_42")
        result = run_async(executor.execute_tool_call(call))
        assert result.call_id == "c8"
        assert result.id == "openai_42"

    def test_execute_case_insensitive_name(self):
        """Tool name lookup is case- and underscore-insensitive."""
        executor = ToolExecutor(tools=[EchoTool])
        call = ToolCall(name="Echo", call_id="c9", arguments={"message": "hi"})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is True

    def test_execute_underscore_insensitive(self):
        """Tool name lookup ignores underscores."""
        executor = ToolExecutor(tools=[NoParamTool])
        # no_param normalized to "noparam"; try "NoParam" -> "noparam"
        call = ToolCall(name="NoParam", call_id="c10", arguments={})
        result = run_async(executor.execute_tool_call(call))
        assert result.success is True

    def test_parallel_tool_call(self):
        executor = ToolExecutor(tools=[EchoTool])
        calls = [
            ToolCall(name="echo", call_id="p1", arguments={"message": "a"}),
            ToolCall(name="echo", call_id="p2", arguments={"message": "b"}),
        ]
        results = run_async(executor.parallel_tool_call(calls))
        assert len(results) == 2
        assert all(r.success for r in results)
        outputs = {r.result for r in results}
        assert outputs == {"a", "b"}

    def test_sequential_tool_call(self):
        executor = ToolExecutor(tools=[EchoTool])
        calls = [
            ToolCall(name="echo", call_id="s1", arguments={"message": "x"}),
            ToolCall(name="echo", call_id="s2", arguments={"message": "y"}),
        ]
        results = run_async(executor.sequential_tool_call(calls))
        assert len(results) == 2
        assert results[0].result == "x"
        assert results[1].result == "y"

    def test_parallel_with_concurrency_limit(self):
        executor = ToolExecutor(tools=[EchoTool], max_concurrency=1)
        calls = [
            ToolCall(name="echo", call_id=f"lim{i}", arguments={"message": str(i)})
            for i in range(3)
        ]
        results = run_async(executor.parallel_tool_call(calls))
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_close(self):
        executor = ToolExecutor(tools=[EchoTool])
        # Should not raise
        run_async(executor.close())

    def test_parallel_with_env_params(self):
        executor = ToolExecutor(tools=[NoParamTool])
        calls = [
            ToolCall(name="no_param", call_id="e1", arguments={}),
            ToolCall(name="no_param", call_id="e2", arguments={}),
        ]
        results = run_async(executor.parallel_tool_call(
            calls,
            env_params=["env1", "env2"],
        ))
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_sequential_with_env_params(self):
        executor = ToolExecutor(tools=[NoParamTool])
        calls = [
            ToolCall(name="no_param", call_id="se1", arguments={}),
        ]
        results = run_async(executor.sequential_tool_call(
            calls,
            env_params=["env1"],
        ))
        assert len(results) == 1
        assert results[0].success is True

    def test_parallel_default_env_and_kwargs(self):
        """parallel_tool_call works with default None env_params and extra_kwargs."""
        executor = ToolExecutor(tools=[EchoTool])
        calls = [ToolCall(name="echo", call_id="d1", arguments={"message": "x"})]
        results = run_async(executor.parallel_tool_call(calls))
        assert len(results) == 1
        assert results[0].success is True

    def test_sequential_default_env_and_kwargs(self):
        """sequential_tool_call works with default None env_params and extra_kwargs."""
        executor = ToolExecutor(tools=[EchoTool])
        calls = [ToolCall(name="echo", call_id="d2", arguments={"message": "y"})]
        results = run_async(executor.sequential_tool_call(calls))
        assert len(results) == 1
        assert results[0].success is True


# ──────────────────────────────────────────────────────────────
# Tests — ToolHandler
# ──────────────────────────────────────────────────────────────

class TestToolHandler:
    """Tests for ToolHandler."""

    def test_parse_single_tool(self):
        handler = ToolHandler(tools=[EchoTool])
        calls = handler.parse_and_match_tool("echo: hello world")
        assert len(calls) == 1
        assert calls[0].name == "echo"
        assert calls[0].arguments["message"] == "hello world"

    def test_parse_no_match(self):
        handler = ToolHandler(tools=[EchoTool])
        calls = handler.parse_and_match_tool("some random text")
        assert len(calls) == 0

    def test_parse_multiple_tools(self):
        """Multiple tools can be parsed from the same LLM output."""
        handler = ToolHandler(tools=[EchoTool, NoParamTool])
        calls = handler.parse_and_match_tool("echo: hi noparam here")
        names = [c.name for c in calls]
        assert "echo" in names
        assert "no_param" in names

    def test_parse_multi_return(self):
        """Tool that returns multiple argument dicts from custom_parse."""
        handler = ToolHandler(tools=[MultiParseTool])
        calls = handler.parse_and_match_tool("multi: alpha, beta, gamma")
        assert len(calls) == 3
        messages = [c.arguments["message"] for c in calls]
        assert messages == ["alpha", "beta", "gamma"]

    def test_parse_bad_parser_logged_not_raised(self):
        """Tool with a broken custom_parse does not crash the handler."""
        handler = ToolHandler(tools=[BadParseTool, EchoTool])
        calls = handler.parse_and_match_tool("echo: safe text")
        # EchoTool should still succeed despite BadParseTool erroring
        assert len(calls) == 1
        assert calls[0].name == "echo"

    def test_register_tool(self):
        handler = ToolHandler(tools=[])
        handler.register_tool(EchoTool)
        assert "echo" in handler.list_registered()

    def test_unregister_tool(self):
        handler = ToolHandler(tools=[EchoTool])
        handler.unregister_tool("echo")
        assert "echo" not in handler.list_registered()

    def test_unregister_nonexistent(self):
        """Unregistering a tool that doesn't exist does not raise."""
        handler = ToolHandler(tools=[])
        handler.unregister_tool("ghost")  # should not raise

    def test_list_registered(self):
        handler = ToolHandler(tools=[EchoTool, FailingTool])
        names = handler.list_registered()
        assert "echo" in names
        assert "fail" in names

    def test_describe_empty(self):
        handler = ToolHandler(tools=[])
        assert handler.describe_registered_tools() == "No tools registered."

    def test_describe_with_tools(self):
        handler = ToolHandler(tools=[EchoTool, FailingTool])
        desc = handler.describe_registered_tools()
        assert "Echo a message back." in desc
        assert "A tool that always fails." in desc

    def test_validation_rejects_invalid_args(self):
        """custom_parse returning invalid args for ParamModel should be skipped."""

        class StrictParams(BaseModel):
            required_field: str

        class StrictTool(Tool):
            ParamModel = StrictParams
            name = "strict"
            description = "Strict."

            @classmethod
            async def execute(cls, arguments, env=None, **kwargs):
                return ToolExecResult(output="ok")

            @classmethod
            def custom_parse(cls, raw: str):
                # Return args missing required_field
                if "strict" in raw:
                    return {"wrong_field": "value"}
                return None

        handler = ToolHandler(tools=[StrictTool])
        calls = handler.parse_and_match_tool("strict call")
        # Should be empty because validation fails
        assert len(calls) == 0

    def test_call_id_format(self):
        """Verify call_id format matches expected pattern."""
        handler = ToolHandler(tools=[EchoTool])
        calls = handler.parse_and_match_tool("echo: test")
        assert len(calls) == 1
        assert calls[0].call_id.startswith("call_echo_idx_")

    def test_parse_skips_none_in_list(self):
        """If custom_parse returns a list with None elements, they are skipped."""

        class NoneListTool(Tool):
            name = "none_list"
            description = "Returns list with None."

            @classmethod
            async def execute(cls, arguments, env=None, **kwargs):
                return ToolExecResult(output="ok")

            @classmethod
            def custom_parse(cls, raw: str):
                if "nonelist" in raw:
                    return [None, {"key": "val"}, None]
                return None

        handler = ToolHandler(tools=[NoneListTool])
        calls = handler.parse_and_match_tool("nonelist input")
        assert len(calls) == 1
        assert calls[0].arguments == {"key": "val"}

    def test_parse_returns_empty_when_all_none(self):
        """If all tools return None from custom_parse, result is empty."""
        handler = ToolHandler(tools=[FailingTool])  # custom_parse always returns None
        calls = handler.parse_and_match_tool("anything")
        assert calls == []

    def test_handler_empty_dict_args_skipped(self):
        """Empty dict arguments from custom_parse are treated as falsy and skipped."""

        class EmptyDictTool(Tool):
            name = "empty_dict"
            description = "Returns empty dict."

            @classmethod
            async def execute(cls, arguments, env=None, **kwargs):
                return ToolExecResult(output="ok")

            @classmethod
            def custom_parse(cls, raw: str):
                if "empty" in raw:
                    return {}
                return None

        handler = ToolHandler(tools=[EmptyDictTool])
        calls = handler.parse_and_match_tool("empty input")
        # Empty dict is falsy, so handler skips it
        assert len(calls) == 0
