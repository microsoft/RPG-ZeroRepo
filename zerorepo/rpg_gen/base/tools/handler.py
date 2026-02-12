
import asyncio
from .error import ToolError
from .tool import Tool, ToolResult, ToolCall, ToolCallArguments
from typing import Any, Dict, Optional, List

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class ToolExecutor:
    """Async executor that manages tool registration, invocation, and shared state."""

    def __init__(self, tools: Optional[list[Tool]] = None, *, max_concurrency: Optional[int] = None):
        self._tool_map: Dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None


    # --- Registration ---
    def register(self, tool: Tool) -> None:
        key = self._normalize_name(tool.name)
        if key in self._tool_map:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tool_map[key] = tool

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace("_", "")

    # --- Close all tools ---
    async def close(self) -> None:
        await asyncio.gather(*[t.close() for t in self._tool_map.values()])

    # --- Single call ---
    async def execute_tool_call(self, tool_call: ToolCall, env: Optional[Any]=None, **kwargs) -> ToolResult:
        key = self._normalize_name(tool_call.name)
        tool = self._tool_map.get(key)
        if not tool:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available: {[t.name for t in self._tool_map.values()]}",
                call_id=tool_call.call_id,
                id=tool_call.id
            )

        async def _run() -> ToolResult:
            try:
                args: ToolCallArguments = tool_call.arguments

                payload: ToolCallArguments = await tool.check(args)
                await tool.before_execute(payload, env, **kwargs)

                exec_res = await tool.execute(payload, env, **kwargs)
                await tool.after_execute(payload, exec_res, env, **kwargs)

                return ToolResult(
                    name=tool_call.name,
                    success=(exec_res.error_code == 0),
                    result=exec_res.output,
                    state=exec_res.state,
                    error=exec_res.error,
                    call_id=tool_call.call_id,
                    id=tool_call.id
                )
            except ToolError as e:
                return ToolResult(
                    name=tool_call.name,
                    success=False,
                    error=str(e),
                    call_id=tool_call.call_id,
                    id=tool_call.id
                )
            except Exception as e:
                return ToolResult(
                    name=tool_call.name,
                    success=False,
                    error=f"Unhandled error in tool '{tool_call.name}': {e}",
                    call_id=tool_call.call_id,
                    id=tool_call.id
                )

        if self._sem is None:
            return await _run()
        async with self._sem:
            return await _run()

    # --- Multiple calls ---
    async def parallel_tool_call(self, tool_calls: List[ToolCall], env_params: List[Any]=[], extra_kwargs: List[Dict]=[]) -> list[ToolResult]:
        if not env_params:
            env_params = [None * len(tool_calls)]
        if not extra_kwargs or len(extra_kwargs) == 0:
            extra_kwargs = [{} * len(tool_calls)]
        
        tasks = [self.execute_tool_call(c, env_param, **extra_kwarg) for c, env_param, extra_kwarg in zip(tool_calls, env_params, extra_kwargs)]
        return await asyncio.gather(*tasks)

    async def sequential_tool_call(self, tool_calls: List[ToolCall], env_params: List[ToolCall]=[], extra_kwargs: List[Dict]=[]) -> list[ToolResult]:
        if not env_params:
            env_params = [{} * len(tool_calls)]
        if not extra_kwargs or len(extra_kwargs) == 0:
            extra_kwargs = [{} * len(tool_calls)]
        
        results: list[ToolResult] = []
        for c, env_param, extra_kwarg in zip(tool_calls, env_params, extra_kwargs):
            res = await self.execute_tool_call(c, env_param, **extra_kwarg)
            results.append(res)
        return results

class ToolHandler:
    """    
    - Call each tool's parse_from_str / custom_parse one by one.
    - Each tool decides how to extract its parameters from the LLM output.
    """

    def __init__(self, tools: list[Tool]):
        self.tool_map = {t.name.lower(): t for t in tools}

    def parse_and_match_tool(self, llm_output: str) -> Optional[List[ToolCall]]:
        """
        Try to find a tool that can be successfully parsed from the LLM output.
        Each tool defines its own parsing rules.
        """        
        all_parsed_tools = []
        for tool_name, tool in self.tool_map.items():
            try:
                parsed_args: List[ToolCall] = tool.custom_parse(llm_output)
                parsed_args = parsed_args if isinstance(parsed_args, List) else \
                    [parsed_args]
                if not parsed_args:
                    continue
                for idx, parsed_arg in enumerate(parsed_args):
                    if not parsed_arg:
                        continue
                    if self._validate_arguments(tool, parsed_arg):
                        all_parsed_tools.append(
                            ToolCall(
                                name=tool.name,
                                call_id=f"call_{tool_name}_idx_{idx + 1}",
                                arguments=parsed_arg
                            )
                        )
            except Exception as e:
                print(f"[ToolHandler] {tool_name}.custom_parse() error: {e}")
        if len(all_parsed_tools) == 0:
            print("[ToolHandler] No tool could parse this output.")
        return all_parsed_tools

    def _validate_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> bool:
        """Validate whether the arguments conform to the ParamModel."""
        try:
            if tool.ParamModel:
                _ = tool.ParamModel(**arguments)
            return True
        except Exception as e:
            print(f"[ToolHandler] Argument validation error ({tool.name}): {e}")
            return False

    def register_tool(self, tool: Tool):
        """Dynamically register a new tool."""
        self.tool_map[tool.name.lower()] = tool

    def unregister_tool(self, name: str):
        """Remove a tool."""
        self.tool_map.pop(name.lower(), None)

    def list_registered(self) -> list[str]:
        """List registered tools."""
        return list(self.tool_map.keys())

    def describe_registered_tools(self) -> str:
        """
        Return the names and description info of currently registered tools,
        for displaying to the LLM or for logging.
        """
        if not self.tool_map:
            return "No tools registered."

        lines = []
        for name, tool in self.tool_map.items():
            desc = tool.description
            lines.append(f"{desc}")

        return "\n".join(lines)