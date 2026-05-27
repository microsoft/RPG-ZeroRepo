#!/usr/bin/env python3
"""Fetch Node Tool — retrieve precise metadata and source context for known code or feature entities.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/fetch_node.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from common.tools import Tool, ToolCallArguments, ToolExecResult
from rpg_agent.ops.fetch import fetch_node
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Description
# ============================================================================

FETCH_NODE_DESC = """
### Tool Name: fetch_node
#### Description
- Retrieve precise metadata and source context for known code or feature entities.
- Use this tool to verify candidate code locations after identifying them through searches or graph exploration.
- Provides exact file path, entity type, start/end lines, mapped feature information, and code preview.
#### Parameters
{
    "tool_name": "fetch_node",
    "parameters": {
        "code_entities": "<List of existing and validated code entities in the current repository; non-existent paths or speculative entities may be ignored. Optional.>",
        "feature_entities": "<List of existing and validated feature paths in the current repository; non-existent entries may be ignored. Optional.>",
    }
}
#### Returns
- Entity type (file/class/method/feature), Feature paths and Code Content
#### Example Calls
##### Example 1: Retrieving metadata for a code entity
{
    "tool_name": "fetch_node",
    "parameters": {
        "code_entities": ["src/module/my_method.py"]
    }
}
"""


# ============================================================================
# Parameter Model
# ============================================================================

class RetrieveParam(BaseModel):
    """Parameters for the ``fetch_node`` tool."""

    code_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of code entities to retrieve information about. "
            "Each entry can be a file path, class name, or method identifier, "
            "for example: ['src/module/my_method.py'] or "
            "['src/requests/adapters.py:BaseAdapter.send']."
        ),
    )
    feature_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of feature entities (functional paths) to retrieve information about. "
            "These describe specific functionality hierarchies within the codebase."
        ),
    )

    @field_validator("code_entities", "feature_entities", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v


# ============================================================================
# Tool Class
# ============================================================================

class FetchNode(Tool):
    """Fetch metadata and source for known code / feature entities.

    Source: RPG-ZeroRepo tools/fetch_node.py ``FetchNode``
    """

    ParamModel: type[BaseModel] = RetrieveParam
    name: str = "fetch_node"
    description: str = FETCH_NODE_DESC

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        """Parse raw JSON input and validate against RetrieveParam."""
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()
                raw = json.loads(raw)

            tool_name = raw.get("tool_name", "")
            if tool_name.lower().strip() != cls.name.lower():
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
        """Run the fetch_node tool with validated arguments."""
        action_dict = arguments
        env_dict = env or {}

        repo_rpg: Optional[RPG] = env_dict.get("rpg")
        entity_searcher = env_dict.get("entity_searcher")

        code_entities = action_dict.get("code_entities", [])
        feature_entities = action_dict.get("feature_entities", [])

        if not repo_rpg or not entity_searcher:
            return ToolExecResult(
                error="RPG or entity_searcher not available in environment",
                error_code=1,
            )

        search_result, suc = fetch_node(
            rpg=repo_rpg,
            entity_searcher=entity_searcher,
            code_entities=code_entities,
            feature_entities=feature_entities,
        )

        if suc:
            return ToolExecResult(output=search_result)
        return ToolExecResult(error=search_result, error_code=1)
