from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Optional, Union
import json
import logging
from copy import deepcopy
from zerorepo.rpg_gen.base.rpg import RPG
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.tools import (
    Tool,
    ToolCallArguments,
    ToolExecResult
)
from ..ops.explore_rpg import explore_tree_structure


EXPLORE_RPG = """
### Tool Name: explore_rpg_structure
#### Description
- Explore call chains and functional paths in the Repository Planning Graph (RPG).
- Starting from known code or feature entities, traverse upstream/downstream to discover related functions, files, and feature nodes.
#### Parameters
{
  "tool_name": "explore_rpg_structure",
  "parameters": {
    "start_code_entities": "An optional list of existing code entities in the current repository (file paths, classes, or functions); non-existent or speculative entities may be ignored or rejected.",
    "start_feature_entities": "An optional list of existing feature paths in the current repository; non-existent entries may be ignored or rejected.",    "direction": "Specifies the direction of graph traversal: 'upstream' (dependencies), 'downstream' (dependents), or 'both'. Defaults to 'downstream'.",
    "traversal_depth": "The maximum depth of traversal. Defaults to 2. Use -1 for unlimited depth.",
    "entity_type_filter": "Optional filter that restricts traversal to specific node types. Valid values: 'directory', 'file', 'class', 'function', 'method'.",
    "dependency_type_filter": "Optional filter that restricts traversal to specific dependency types. Valid values: 'composes', 'contains', 'inherits', 'invokes', 'imports'."
  }
}
#### Returns
- Connected nodes and edges (code or feature view)
- Hints for invalid or fuzzy matches
#### Example JSON Calls
##### Example 1: Exploring downstream dependencies for a specific function
> You suspect this function triggers the failure and want to see what it calls next.
{
    "tool_name": "explore_rpg_structure",
    "parameters": {
        "start_code_entities": ["src/main.py:my_function"],
        "direction": "downstream",
        "traversal_depth": 3,
        "entity_type_filter": ["method", "class"],
        "dependency_type_filter": ["invokes", "imports"]
    }
}
"""


class ExploreParam(BaseModel):
    """
    Parameters for the `explore_rpg_structure` tool.
    Defines how to explore code or feature dependencies by specifying start entities, direction, depth, and filters.
    """

    start_code_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of code entities (files, methods, or classes) to start exploration from. "
            "Each string may be a file path or method identifier like `'src/module/file.py:Class.method'`. "
            "If not provided, exploration will start from `start_feature_entities` instead."
        ),
        examples=[["src/main.py:my_function"], ["src/requests/adapters.py:BaseAdapter.send"]]
    )

    start_feature_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of feature entities (functional paths) that define where to start exploration. "
            "Each feature path represents a functional hierarchy, for example: "
            "`['CoreUtilitiesAndErrorHandling/string utilities/string compatibility/check if string is ascii']`."
        ),
        examples=[["src/requests/adapters.py:BaseAdapter.send"]]
    )

    direction: str = Field(
        default="downstream",
        description=(
            "Traversal direction for graph exploration. "
            "Possible values: `'downstream'` (explore dependents), `'upstream'` (explore dependencies), `'both'` (explore both directions). "
            "Defaults to `'downstream'`."
        ),
        examples=["downstream", "upstream", "both"]
    )

    traversal_depth: int = Field(
        default=2,
        description=(
            "Maximum number of dependency levels to traverse. "
            "A depth of `-1` removes any limit (explore all levels). "
            "Default is `2`."
        ),
        examples=[1, 2, -1]
    )

    entity_type_filter: Optional[List[str]] = Field(
        default=[],
        description=(
            "Optional filter specifying which entity types to include in the exploration. "
            "Examples include `'class'`, `'method'`, `'file'`, etc. "
            "If omitted, all entity types will be considered."
        ),
        examples=[["method", "class"], ["file"]]
    )

    dependency_type_filter: Optional[List[str]] = Field(
        default=[],
        description=(
            "Optional filter specifying which dependency types to explore. "
            "Examples: `'calls'`, `'imports'`, `'inherits'`. "
            "If omitted, all dependency types are included."
        ),
        examples=[["calls", "imports"]]
    )

    @field_validator('start_code_entities', 'start_feature_entities', 'entity_type_filter', 'dependency_type_filter', mode='before')
    @classmethod
    def coerce_str_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class ExploreRPG(Tool):
    
    ParamModel: BaseModel=ExploreParam
    
    name: str = "explore_rpg_structure"
    description: str = EXPLORE_RPG
    
    
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
        """Run the tool with validated arguments."""
        action_dict = arguments
        env_dict = env or {}
         
         
        repo_rpg: RPG = env_dict.get("rpg")

        if not repo_rpg:
            return ToolExecResult(
                error="RPG not available in environment",
                error_code=1
            )

        start_code_entites = action_dict.get("start_code_entities") or []
        start_feature_entites = action_dict.get("start_feature_entities") or []
        direction = action_dict.get("direction", "downstream")
        traversal_depth = action_dict.get("traversal_depth", 2)
        entity_type_filter = action_dict.get("entity_type_filter") or None
        dependency_type_filter = action_dict.get("dependency_type_filter") or None
        
         
        search_result, suc = explore_tree_structure(
            start_code_entites=start_code_entites,
            start_feature_entites=start_feature_entites,
            direction=direction,
            traversal_depth=traversal_depth,
            entity_type_filter=entity_type_filter,
            dependency_type_filter=dependency_type_filter,
            rpg=repo_rpg
        )
        
        if suc:
            return ToolExecResult(
                output=search_result
            )
        else:
            return ToolExecResult(
                error=search_result,
                error_code=1
            )
