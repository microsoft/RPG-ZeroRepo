from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Optional, Union
import json
import logging
from copy import deepcopy
from ..env import Env
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.rpg import RPG
from zerorepo.utils.repo import is_test_file
from zerorepo.rpg_gen.base.tools import (
    Tool,
    ToolCallArguments,
    ToolExecResult
)

TERMINATE = """
### Tool Name: terminate
#### Description
- Permanently end the task and submit the final verified code locations.
- Once this tool succeeds, **the task ends immediately and cannot be resumed or corrected**.
- Use when you are reasonably confident about your findings based on gathered evidence.
- **All results must be ordered by importance to fixing the issue** — the most critical code location comes first.
- Partially valid submissions are accepted: valid items are saved, invalid ones are reported as warnings.
#### Parameters
{
    "tool_name": "terminate",
    "parameters": {
        "results": [
            {
                "file_path": "<File path of the code entity (str)>",
                "func_name": "<String: Function / class / method name (e.g., func, Class, Class.method)>",
                "line_nums": "<Two valid integers [start, end] indicating the exact lines inside the function where the issue manifests>"
            }
        ]
    }
}
#### How to choose `func_name`
- For a standalone function: use `function_name`
- For a class method: use `ClassName.methodName` (add one entry per method if multiple)
- For class-level code (attributes / class body): use `ClassName` (separate entries if also changing methods)
#### Returns
- Confirmation of the final valid code locations, or structured feedback if any are invalid.
#### Example Calls
##### Example 1: Validating multiple code locations
{
  "tool_name": "terminate",
  "parameters": {
    "results": [
      {
        "file_path": "src/requests/adapters.py",
        "func_name": "BaseAdapter.send",
        "line_nums": [50, 70]
      }
    ]
  }
}
"""


class ResultParam(BaseModel):
    """
    Represents a single predicted function location in the codebase.
    Each result includes the file path, function name, and the predicted line range.
    """

    file_path: str = Field(
        ...,
        description=(
            "The absolute or relative path of the file containing the target function. "
            "For example: `'src/requests/adapters.py'`."
        ),
        examples=["src/requests/adapters.py"]
    )

    func_name: str = Field(
        ...,
        description=(
            "The fully qualified name of the function or method to validate. "
            "Can include class prefix if applicable, e.g. `'BaseAdapter.send'`."
        ),
        examples=["BaseAdapter.send"]
    )

    line_nums: List[int] = Field(
        ...,
        description=(
            "The predicted line number range representing the specific segment "
            "within the function. Should be a list of two integers: `[start, end]`. "
            "For example: `[50, 70]`."
        ),
        examples=[[50, 70]]
    )

    @field_validator('line_nums', mode='before')
    @classmethod
    def coerce_int_to_list(cls, v):
        if isinstance(v, (int, float)):
            return [int(v)]
        return v


class TerminateParam(BaseModel):
    """
    Input schema for the `terminate` tool.
    Contains the list of predicted code locations that should be validated before ending the exploration.
    """

    results: Optional[List[ResultParam]] = Field(
        default=None,
        description=(
            "A list of predicted function locations (each defined by `ResultParam`). "
            "Each entry should include a file path, function name, and a pair of start and end lines "
            "indicating the predicted segment to validate. Example:\n"
            "[\n"
            "  {\n"
            "    'file_path': 'src/requests/adapters.py',\n"
            "    'func_name': 'BaseAdapter.send',\n"
            "    'line_nums': [50, 70]\n"
            "  }\n"
            "]"
        )
    )

    @field_validator('results', mode='before')
    @classmethod
    def coerce_dict_to_list(cls, v):
        if isinstance(v, dict):
            return [v]
        return v

    

class Terminate(Tool):
    
    ParamModel: BaseModel=TerminateParam
    
    name: str = "terminate"
    description: str = TERMINATE
    
    
    @classmethod
    def custom_parse(cls, raw: str) -> ToolCallArguments:
        """
        Parse a raw JSON input, validate it against SearchCodeParam,
        and return a validated dictionary of parameters.
        """
        try:
            # Step 1: Clean markdown fences if present
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()
                raw = json.loads(raw)
            # Step 2: Tool name check
            tool_name = raw.get("tool_name", "")
            if tool_name.lower().strip() != cls.get_name().lower():
                logging.warning(f"Warning: tool_name '{tool_name}' does not match expected '{cls.get_name()}'")
                return None
            # Step 3: Extract parameters
            params = raw.get("parameters", raw)
            # Step 4: Validate
            parsed = cls.ParamModel(**params).model_dump()
            return parsed

        except json.JSONDecodeError as e:
            logging.error(f"[custom_parse] Invalid JSON: {e.msg}")
            return None
        except ValidationError as e:
            logging.error(f"[custom_parse] Parameter validation failed: {e.errors()}")
            return None

    @classmethod
    async def execute(cls, arguments: Dict[str, Union[ToolCallArguments, Dict]], env=None, **kwargs) -> ToolExecResult:

        from itertools import islice
        from zerorepo.rpg_encoder.rpg_agent.env import RepoEntitySearcher

        action_dict = arguments
        env_dict = env or {}

        rpg: RPG = env_dict.get("rpg")
        agent_env: Env = env_dict.get("environment")
        entity_searcher: RepoEntitySearcher = env_dict.get("entity_searcher")

        # Create entity_searcher from RPG if not provided
        if not entity_searcher and rpg:
            entity_searcher = RepoEntitySearcher(rpg=rpg)

        if not agent_env.final_results and agent_env.step_count < 1:
            return ToolExecResult(
                error=(
                    "Terminate blocked: no verified code locations yet.\n"
                    "Use search/retrieve/graph tools to find real entities and line ranges first."
                ),
                error_code=1
            )

        terminate_results = action_dict.get("results", [])
        if not terminate_results:
            return ToolExecResult(
                error=(
                    "Terminate rejected: `results` is empty.\n"
                    "Each item must include: file_path(str), func_name(str), line_nums([start,end] ints)."
                ),
                error_code=1
            )

        success_results: List[Dict] = []
        error_items: List[Dict] = []
        seen_entities = set()

        def err_item(idx: int, stage: str, code: str, message: str, ctx: Dict = None) -> Dict:
            item = {"index": idx, "stage": stage, "error_code": code, "message": message}
            if ctx:
                item["context"] = ctx
            return item

        def hint(code: str) -> str:
            return {
                "E_DUPLICATE_ENTITY": "Fix: remove duplicates.",
                "E_PARAM_INVALID": "Fix: line_nums must be two integers [start,end].",
                "E_RANGE_ORDER": "Fix: ensure start <= end.",
                "E_FUNC_NOT_FOUND": "Fix: verify exact file_path/func_name exists in graph (no placeholders).",
                "E_RANGE_OUTSIDE": "Fix: choose a subrange within graph start_line/end_line.",
            }.get(code, "Fix: verify this item via tools.")

        def ctx_str(ctx: Dict) -> str:
            if not ctx:
                return ""
            keys = ("node_id", "file_path", "func_name", "predicted_range", "expected_range", "available_in_file")
            parts = [f"{k}={ctx[k]}" for k in keys if k in ctx and ctx[k] is not None]
            return "; ".join(parts)

        TOLERANCE = 3  # Allow 3 lines of tolerance for range checks

        # Validation loop
        for i, found in enumerate(terminate_results):
            file_path = found.get("file_path")
            func_name = found.get("func_name")
            line_nums = found.get("line_nums")

            entity_id = f"{file_path}:{func_name}"
            if entity_id in seen_entities:
                error_items.append(err_item(
                    idx=i, stage="parse", code="E_DUPLICATE_ENTITY",
                    message="Duplicate entity.",
                    ctx={"file_path": file_path, "func_name": func_name}
                ))
                continue
            seen_entities.add(entity_id)

            if not (isinstance(file_path, str) and file_path and
                    isinstance(func_name, str) and func_name and
                    isinstance(line_nums, list) and len(line_nums) == 2 and
                    all(isinstance(x, int) for x in line_nums)):
                error_items.append(err_item(
                    idx=i, stage="parse", code="E_PARAM_INVALID",
                    message="Bad parameters.",
                    ctx={"received": found}
                ))
                continue

            pred_start, pred_end = line_nums
            if pred_start > pred_end:
                error_items.append(err_item(
                    idx=i, stage="parse", code="E_RANGE_ORDER",
                    message="Invalid range: start > end.",
                    ctx={"line_nums": line_nums}
                ))
                continue

            node_id = f"{file_path}:{func_name}"

            if not entity_searcher.has_node(node_id):
                # List available entities in the file for better error messages
                available = [nid for nid in entity_searcher.G.nodes
                             if nid.startswith(file_path + ":")][:10]
                available_str = ", ".join(available) if available else "none found"
                error_items.append(err_item(
                    idx=i, stage="node_lookup", code="E_FUNC_NOT_FOUND",
                    message="Entity not found in graph.",
                    ctx={"node_id": node_id, "file_path": file_path,
                         "func_name": func_name, "available_in_file": available_str}
                ))
                continue

            node_list = entity_searcher.get_node_data([node_id], return_code_content=True)
            if not node_list or not isinstance(node_list, list):
                error_items.append(err_item(
                    idx=i, stage="node_data", code="E_NODE_DATA_EMPTY",
                    message="Graph returned no node data.",
                    ctx={"node_id": node_id}
                ))
                continue

            node_data = node_list[0]
            if "start_line" not in node_data or "end_line" not in node_data:
                error_items.append(err_item(
                    idx=i, stage="node_data", code="E_NODE_DATA_MISSING_FIELDS",
                    message="Node missing start/end line.",
                    ctx={"node_id": node_id, "node_data_keys": list(node_data.keys())}
                ))
                continue

            start_line = node_data["start_line"]
            end_line = node_data["end_line"]

            # Range check with tolerance
            if not (start_line - TOLERANCE <= pred_start and pred_end <= end_line + TOLERANCE):
                error_items.append(err_item(
                    idx=i, stage="range_check", code="E_RANGE_OUTSIDE",
                    message="Range outside entity boundary.",
                    ctx={
                        "file_path": file_path,
                        "func_name": func_name,
                        "predicted_range": [pred_start, pred_end],
                        "expected_range": [start_line, end_line]
                    }
                ))
                continue

            success_results.append(found)

        # Partial accept: always save successful items
        if success_results:
            agent_env.final_results = success_results

        if error_items and not success_results:
            # All items failed — reject entirely
            lines: List[str] = []
            lines.append("Terminate rejected: fix the items below.")
            lines.append(f"Valid: {len(success_results)} | Invalid: {len(error_items)}")
            lines.append("")

            lines.append("How to fix:")
            lines.append("1) Ensure (file_path, func_name) exists in the graph (exact match).")
            lines.append("2) Use graph start_line/end_line; choose line_nums within that range.")
            lines.append("3) No placeholders; line_nums must be two integers.")
            lines.append("")

            max_show = 8
            for e in islice(error_items, max_show):
                code = e.get("error_code")
                msg = (e.get("message") or "").strip()
                ctx = e.get("context") or {}

                line = f"[{e.get('index')}] {e.get('stage')}/{code}: {msg} {hint(code)}"
                s = ctx_str(ctx)
                if s:
                    line += f" | {s}"
                lines.append(line)

            remaining = len(error_items) - max_show
            if remaining > 0:
                lines.append(f"... and {remaining} more.")
            lines.append("")
            lines.append("Retry after verifying entities and ranges via tools.")

            return ToolExecResult(error="\n".join(lines), error_code=1)

        if error_items and success_results:
            # Partial success: accept valid items, warn about invalid ones
            output = json.dumps(success_results, ensure_ascii=False, indent=2)
            warning = (
                f"\nWarning: {len(error_items)} item(s) were invalid and excluded:\n"
            )
            max_show = 5
            for e in islice(error_items, max_show):
                code = e.get("error_code")
                msg = (e.get("message") or "").strip()
                ctx = e.get("context") or {}
                s = ctx_str(ctx)
                warning += f"  [{e.get('index')}] {code}: {msg}"
                if s:
                    warning += f" | {s}"
                warning += "\n"
            remaining = len(error_items) - max_show
            if remaining > 0:
                warning += f"  ... and {remaining} more.\n"
            return ToolExecResult(output=output + warning)

        return ToolExecResult(output=json.dumps(success_results, ensure_ascii=False, indent=2))
