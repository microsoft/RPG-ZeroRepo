#!/usr/bin/env python3
"""Explore RPG Structure Tool — traverse call chains and functional paths in the Repository Planning Graph (RPG).

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/explore_rpg.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from common.tools import Tool, ToolCallArguments, ToolExecResult
from rpg_agent.ops.explore import explore_tree_structure
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Description
# ============================================================================

EXPLORE_RPG_DESC = """
### Tool Name: explore_rpg_structure
#### Description
- Explore call chains and functional paths in the Repository Planning Graph (RPG).
- Starting from known code or feature entities, traverse upstream/downstream to discover related functions, files, and feature nodes.
#### Parameters
{
  "tool_name": "explore_rpg_structure",
  "parameters": {
    "start_code_entities": "An optional list of existing code entities in the current repository (file paths, classes, or functions); non-existent or speculative entities may be ignored or rejected.",
    "start_feature_entities": "An optional list of existing feature paths in the current repository; non-existent entries may be ignored or rejected.",
    "direction": "Specifies the direction of graph traversal: 'upstream' (dependencies), 'downstream' (dependents), or 'both'. Defaults to 'downstream'.",
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


# ============================================================================
# Parameter Model
# ============================================================================

class ExploreParam(BaseModel):
    """Parameters for the ``explore_rpg_structure`` tool."""

    start_code_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of code entities (files, methods, or classes) to start exploration from."
        ),
    )
    start_feature_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of feature entities (functional paths) that define where to start exploration."
        ),
    )
    direction: str = Field(
        default="downstream",
        description=(
            "Traversal direction: 'downstream' (dependents), 'upstream' (dependencies), 'both'."
        ),
    )
    traversal_depth: int = Field(
        default=2,
        description="Maximum number of dependency levels to traverse. -1 for unlimited.",
    )
    entity_type_filter: Optional[List[str]] = Field(
        default=[],
        description="Optional filter for entity types: 'class', 'method', 'file', etc.",
    )
    dependency_type_filter: Optional[List[str]] = Field(
        default=[],
        description="Optional filter for dependency types: 'calls', 'imports', 'inherits', etc.",
    )

    @field_validator(
        "start_code_entities",
        "start_feature_entities",
        "entity_type_filter",
        "dependency_type_filter",
        mode="before",
    )
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v


# ============================================================================
# Tool Class
# ============================================================================

class ExploreRPG(Tool):
    """Explore RPG graph structure (call chains, feature paths).

    Source: RPG-ZeroRepo tools/explore_rpg.py ``ExploreRPG``
    """

    ParamModel: type[BaseModel] = ExploreParam
    name: str = "explore_rpg_structure"
    description: str = EXPLORE_RPG_DESC

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        """Parse raw JSON input and validate against ExploreParam."""
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()
                raw = json.loads(raw)

            tool_name = raw.get("tool_name", "")
            if tool_name.lower().strip() != cls.get_name().lower():
                return None

            params = raw.get("parameters", raw)
            parsed = cls.ParamModel(**params).model_dump()
            return parsed

        except json.JSONDecodeError:
            return None
        except ValidationError:
            return None

    @classmethod
    async def execute(
        cls,
        arguments: Union[ToolCallArguments, Dict[str, Any]],
        env: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolExecResult:
        """Run the explore_rpg_structure tool with validated arguments."""
        action_dict = arguments
        env_dict = env or {}

        repo_rpg: Optional[RPG] = env_dict.get("rpg")

        if not repo_rpg:
            return ToolExecResult(
                error="RPG not available in environment",
                error_code=1,
            )

        start_code_entities = action_dict.get("start_code_entities") or []
        start_feature_entities = action_dict.get("start_feature_entities") or []
        direction = action_dict.get("direction", "downstream")
        traversal_depth = action_dict.get("traversal_depth", 2)
        entity_type_filter = action_dict.get("entity_type_filter") or None
        dependency_type_filter = action_dict.get("dependency_type_filter") or None

        search_result, suc = explore_tree_structure(
            start_code_entities=start_code_entities,
            start_feature_entities=start_feature_entities,
            direction=direction,
            traversal_depth=traversal_depth,
            entity_type_filter=entity_type_filter,
            dependency_type_filter=dependency_type_filter,
            rpg=repo_rpg,
        )

        if suc:
            return ToolExecResult(output=search_result)
        return ToolExecResult(error=search_result, error_code=1)
