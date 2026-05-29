#!/usr/bin/env python3
"""Tool Abstraction Layer for CoderMind.

This module provides a unified tool abstraction for the RPG Agent, enabling
standardized tool definition, parameter validation, execution, and result handling.

Ported from RPG-ZeroRepo (zerorepo/rpg_gen/base/tools/) with adaptations for
CoderMind's project structure and coding conventions.

Key components:
- Tool (ABC): Abstract base class for all agent tools
- ToolExecutor: Registry and executor for tool instances
- ToolHandler: Parses LLM text output to extract tool calls
- ToolCall / ToolResult / ToolExecResult: Data types for tool invocation flow
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)


# ============================================================================
# Error Hierarchy
# ============================================================================

class ToolError(Exception):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation."""


class ToolExecutionError(ToolError):
    """Raised when a tool execution encounters an error."""


# ============================================================================
# Core Data Types
# ============================================================================

ToolCallArguments = Dict[str, Any]
"""Type alias for tool call argument dictionaries."""


@dataclass
class ToolExecResult:
    """Intermediate result of a tool execution.

    Attributes:
        output: The textual output on success.
        error: Error message on failure.
        error_code: 0 indicates success; non-zero indicates failure.
        state: Optional state dict to carry between tool invocations.
    """
    output: Optional[str] = None
    error: Optional[str] = None
    error_code: int = 0
    state: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Final result of a tool call, surfaced by the executor.

    Attributes:
        name: The tool name that was invoked.
        success: Whether the execution succeeded.
        call_id: Identifier for this particular call.
        result: Textual result on success.
        error: Error message on failure.
        id: Optional cross-provider identifier (e.g., OpenAI tool_call id).
        state: Optional state dict carried from execution.
    """
    name: str
    success: bool
    call_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    id: Optional[str] = None
    state: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Represents a parsed tool call input.

    Attributes:
        name: The tool name to invoke.
        call_id: A unique identifier for this call.
        arguments: The argument dictionary.
        id: Optional cross-provider identifier.
    """
    name: str
    call_id: str
    arguments: ToolCallArguments
    id: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"ToolCall(name={self.name}, call_id={self.call_id}, "
            f"arguments={self.arguments}, id={self.id})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


# ============================================================================
# Parameter Model Base
# ============================================================================

class ToolParameter(BaseModel):
    """Abstract base for tool argument models.

    This class is intentionally empty; concrete tools should define their own
    Pydantic models by subclassing ``BaseModel`` or this class.  Keeping this
    around makes it easy to add shared mixins / validators later.
    """
    pass


# ============================================================================
# Tool Abstract Base Class
# ============================================================================

class Tool(ABC):
    """Abstract base for tools / actions with runtime Pydantic validation.

    Each concrete Tool may set ``ParamModel`` to a Pydantic model type.
    If ``ParamModel`` is None the raw ``dict`` is passed through to ``execute``.

    Class attributes:
        ParamModel: Optional Pydantic model for argument validation.
        name: Canonical tool name (must be unique within an executor).
        description: Human-readable description of the tool's purpose.
    """

    ParamModel: Optional[Type[BaseModel]] = None
    name: str = ""
    description: str = ""

    # --- Required metadata ---------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        """Return the canonical tool name."""
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return a concise human-readable description of the tool."""
        return cls.description

    # --- Core execution ------------------------------------------------------

    @classmethod
    @abstractmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, BaseModel],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        """Run the tool with validated arguments.

        Implementations can type-narrow ``arguments`` to their custom ``ParamModel``.
        """

    # --- Validation hook -----------------------------------------------------

    @classmethod
    async def check(cls, arguments: ToolCallArguments) -> ToolCallArguments:
        """Validate / normalize input arguments using ``ParamModel`` if provided.

        Returns the validated / normalized payload (a dict from the Pydantic model
        dump, or the original dict if no ``ParamModel`` is set).

        Raises:
            ToolValidationError: If the arguments fail validation.
        """
        if cls.ParamModel is None:
            return arguments
        try:
            return cls.ParamModel(**arguments).model_dump()
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

    # --- Optional lifecycle hooks --------------------------------------------

    @classmethod
    async def before_execute(
        cls,
        payload: Union[BaseModel, ToolCallArguments],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Hook called right before ``execute``. Override if needed."""
        return None

    @classmethod
    async def after_execute(
        cls,
        payload: ToolCallArguments,
        result: ToolExecResult,
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Hook called right after ``execute``. Override if needed."""
        return None

    # --- Resource cleanup ----------------------------------------------------

    @classmethod
    async def close(cls) -> None:
        """Override to release resources if necessary."""
        return None

    # --- Custom text parsing -------------------------------------------------

    @classmethod
    @abstractmethod
    def custom_parse(cls, raw: str) -> Optional[Union[ToolCallArguments, List[ToolCallArguments]]]:
        """Parse tool arguments from raw LLM text output.

        Returns:
            A single argument dict, a list of argument dicts, or None
            if this tool cannot be parsed from the given text.
        """


# ============================================================================
# ToolExecutor — Registration and Invocation
# ============================================================================

class ToolExecutor:
    """Async executor that manages tool registration, invocation, and shared state.

    Args:
        tools: Optional list of Tool classes to register on construction.
        max_concurrency: Optional semaphore limit for parallel execution.
    """

    def __init__(
        self,
        tools: Optional[List[type[Tool]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ):
        self._tool_map: Dict[str, type[Tool]] = {}
        if tools:
            for tool in tools:
                self.register(tool)
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    # --- Registration --------------------------------------------------------

    def register(self, tool: type[Tool]) -> None:
        """Register a tool class.

        Raises:
            ValueError: If a tool with the same normalized name is already registered.
        """
        key = self._normalize_name(tool.name)
        if key in self._tool_map:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tool_map[key] = tool

    def _normalize_name(self, name: str) -> str:
        """Normalize a tool name for case- and underscore-insensitive lookup."""
        return name.lower().replace("_", "")

    @property
    def tools(self) -> List[type[Tool]]:
        """Return a list of all registered tool classes."""
        return list(self._tool_map.values())

    def list_tools(self) -> List[str]:
        """Return a list of registered tool names."""
        return [t.name for t in self._tool_map.values()]

    # --- Close all tools -----------------------------------------------------

    async def close(self) -> None:
        """Close all registered tools (release resources)."""
        await asyncio.gather(*(t.close() for t in self._tool_map.values()))

    # --- Single call ---------------------------------------------------------

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a single tool call.

        Looks up the tool by normalized name, validates arguments, runs lifecycle
        hooks (before_execute / execute / after_execute), and wraps the result.
        """
        key = self._normalize_name(tool_call.name)
        tool = self._tool_map.get(key)
        if not tool:
            available = [t.name for t in self._tool_map.values()]
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available: {available}",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

        async def _run() -> ToolResult:
            try:
                payload: ToolCallArguments = await tool.check(tool_call.arguments)
                await tool.before_execute(payload, env, **kwargs)

                exec_result = await tool.execute(payload, env, **kwargs)
                await tool.after_execute(payload, exec_result, env, **kwargs)

                return ToolResult(
                    name=tool_call.name,
                    success=(exec_result.error_code == 0),
                    result=exec_result.output,
                    state=exec_result.state,
                    error=exec_result.error,
                    call_id=tool_call.call_id,
                    id=tool_call.id,
                )
            except ToolError as exc:
                return ToolResult(
                    name=tool_call.name,
                    success=False,
                    error=str(exc),
                    call_id=tool_call.call_id,
                    id=tool_call.id,
                )
            except Exception as exc:
                logger.exception("Unhandled error in tool '%s'", tool_call.name)
                return ToolResult(
                    name=tool_call.name,
                    success=False,
                    error=f"Unhandled error in tool '{tool_call.name}': {exc}",
                    call_id=tool_call.call_id,
                    id=tool_call.id,
                )

        if self._sem is None:
            return await _run()
        async with self._sem:
            return await _run()

    # --- Multiple calls ------------------------------------------------------

    async def parallel_tool_call(
        self,
        tool_calls: List[ToolCall],
        env_params: Optional[List[Any]] = None,
        extra_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel (concurrency-limited).

        Args:
            tool_calls: List of tool calls to execute.
            env_params: Per-call environment objects (defaults to None for each).
            extra_kwargs: Per-call extra keyword arguments (defaults to empty dict).
        """
        count = len(tool_calls)
        if env_params is None:
            env_params = [None] * count
        if extra_kwargs is None:
            extra_kwargs = [{}] * count

        tasks = [
            self.execute_tool_call(call, env_param, **kw)
            for call, env_param, kw in zip(tool_calls, env_params, extra_kwargs)
        ]
        return list(await asyncio.gather(*tasks))

    async def sequential_tool_call(
        self,
        tool_calls: List[ToolCall],
        env_params: Optional[List[Any]] = None,
        extra_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ToolResult]:
        """Execute multiple tool calls sequentially.

        Args:
            tool_calls: List of tool calls to execute in order.
            env_params: Per-call environment objects (defaults to None for each).
            extra_kwargs: Per-call extra keyword arguments (defaults to empty dict).
        """
        count = len(tool_calls)
        if env_params is None:
            env_params = [None] * count
        if extra_kwargs is None:
            extra_kwargs = [{}] * count

        results: List[ToolResult] = []
        for call, env_param, kw in zip(tool_calls, env_params, extra_kwargs):
            result = await self.execute_tool_call(call, env_param, **kw)
            results.append(result)
        return results


# ============================================================================
# ToolHandler — Parse LLM Text Output into ToolCalls
# ============================================================================

class ToolHandler:
    """Parses LLM text output and matches it against registered tools.

    Each tool defines its own ``custom_parse`` method to extract arguments
    from free-form text.  The handler iterates over all registered tools,
    collects successful parses, validates their arguments, and returns a
    list of ``ToolCall`` objects.

    Args:
        tools: List of Tool classes to register for parsing.
    """

    def __init__(self, tools: List[type[Tool]]):
        self._tool_map: Dict[str, type[Tool]] = {
            t.name.lower(): t for t in tools
        }

    # --- Parsing -------------------------------------------------------------

    def parse_and_match_tool(self, llm_output: str) -> List[ToolCall]:
        """Try to parse tool calls from LLM output.

        Each registered tool's ``custom_parse`` is invoked.  Successful parses
        whose arguments pass validation are collected and returned.

        Args:
            llm_output: Raw LLM text output.

        Returns:
            List of parsed and validated ToolCall objects (may be empty).
        """
        all_parsed_tools: List[ToolCall] = []

        for tool_name, tool in self._tool_map.items():
            try:
                parsed_args = tool.custom_parse(llm_output)
                if parsed_args is None:
                    continue

                # Normalize to a list
                if not isinstance(parsed_args, list):
                    parsed_args = [parsed_args]

                for idx, parsed_arg in enumerate(parsed_args):
                    if not parsed_arg:
                        continue
                    if self._validate_arguments(tool, parsed_arg):
                        all_parsed_tools.append(
                            ToolCall(
                                name=tool.name,
                                call_id=f"call_{tool_name}_idx_{idx + 1}",
                                arguments=parsed_arg,
                            )
                        )
            except Exception as exc:
                logger.warning(
                    "%s.custom_parse() error: %s", tool_name, exc
                )

        if not all_parsed_tools:
            logger.debug("No tool could parse this output.")

        return all_parsed_tools

    # --- Validation ----------------------------------------------------------

    def _validate_arguments(
        self, tool: type[Tool], arguments: Dict[str, Any]
    ) -> bool:
        """Validate whether the arguments conform to the tool's ParamModel."""
        try:
            if tool.ParamModel:
                tool.ParamModel(**arguments)
            return True
        except Exception as exc:
            logger.warning(
                "Argument validation error (%s): %s", tool.name, exc
            )
            return False

    # --- Dynamic registration ------------------------------------------------

    def register_tool(self, tool: type[Tool]) -> None:
        """Dynamically register a new tool."""
        self._tool_map[tool.name.lower()] = tool

    def unregister_tool(self, name: str) -> None:
        """Remove a tool by name."""
        self._tool_map.pop(name.lower(), None)

    def list_registered(self) -> List[str]:
        """List names of registered tools."""
        return list(self._tool_map.keys())

    def describe_registered_tools(self) -> str:
        """Return descriptions of currently registered tools.

        Useful for displaying available tools to the LLM or for logging.
        """
        if not self._tool_map:
            return "No tools registered."

        lines = []
        for _name, tool in self._tool_map.items():
            lines.append(tool.description)
        return "\n".join(lines)
