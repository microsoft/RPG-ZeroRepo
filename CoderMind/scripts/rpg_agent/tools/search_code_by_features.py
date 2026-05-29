#!/usr/bin/env python3
"""Feature Search Tool — search the repository using functional or behavioral descriptions to map high-level feature terms to concrete code entities.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/search_code_by_features.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from common.tools import Tool, ToolCallArguments, ToolExecResult
from rpg_agent.env.searcher import RepoEntitySearcher
from rpg_agent.ops.search_by_feature import search_features_by_keywords
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Description
# ============================================================================

FEATURE_SEARCH_DESC = """
## Tool Name: search_code_by_features
### Description
- Search the repository using functional or behavioral descriptions when you do not know the exact class, function, or file name.
- This tool matches high-level feature or behavior terms to concrete code entities using the Functionality SubGraph -> Code SubGraph mapping.
- Tip: If the exact file/class/function name isn't obvious from the issue, use this tool (and try a few alternative feature_terms / nearby search_scopes) to map behavioral descriptions to concrete code entities before narrowing down.
### Parameters
{
    "tool_name": "search_code_by_features",
    "parameters": {
        "feature_terms": "<List of feature names to search in RPG>",
        "search_scopes": "<List of feature paths to restrict search scope>"
    }
}
### Returns
Matched feature nodes and their linked code entities based on search parameters.
### Example JSON Calls
#### Example 1: Search for features by name
{
    "tool_name": "search_code_by_features",
    "parameters": {
        "feature_terms": ["abstract send prepared request"],
        "search_scopes": ["ConnectionAndAdapterManagement/adapter management/http adapters"]
    }
}
#### Example 2: Broad feature search
{
    "tool_name": "search_code_by_features",
    "parameters": {
        "feature_terms": ["error handling", "retry logic"]
    }
}
"""


# ============================================================================
# Parameter Model
# ============================================================================

class FeatureSearchParam(BaseModel):
    """Parameters for feature-based search in RPG."""

    feature_terms: Optional[List[str]] = Field(
        default=[],
        description="List of feature names to search in RPG",
    )
    search_scopes: Optional[List[str]] = Field(
        default=[],
        description="Feature paths to restrict search scope",
    )

    @field_validator("feature_terms", "search_scopes", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v


# ============================================================================
# Tool Class
# ============================================================================

class SearchCodeByFeatures(Tool):
    """Feature search tool for finding code entities via behavioral descriptions.

    Source: RPG-ZeroRepo tools/search_code_by_features.py ``SearchCodeByFeatures``
    """

    ParamModel: type[BaseModel] = FeatureSearchParam
    name: str = "search_code_by_features"
    description: str = FEATURE_SEARCH_DESC

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        """Parse raw JSON input and validate against FeatureSearchParam."""
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
        """Execute the feature search tool with validated arguments."""
        action_dict = arguments
        env_dict = env or {}

        rpg: Optional[RPG] = env_dict.get("rpg")
        entity_searcher: Optional[RepoEntitySearcher] = env_dict.get("entity_searcher")

        if not rpg:
            return ToolExecResult(
                error="RPG not available in environment",
                error_code=1,
            )

        if not entity_searcher:
            try:
                entity_searcher = RepoEntitySearcher(rpg=rpg)
            except Exception as exc:
                return ToolExecResult(
                    error=f"Failed to create entity searcher: {exc}",
                    error_code=1,
                )

        feature_terms = action_dict.get("feature_terms") or []
        search_scopes = action_dict.get("search_scopes") or []
        top_k = 5

        if not feature_terms:
            return ToolExecResult(
                error="No feature_terms provided for feature search.",
                error_code=1,
            )

        try:
            result, success = search_features_by_keywords(
                rpg=rpg,
                entity_searcher=entity_searcher,
                keywords=feature_terms,
                search_scopes=search_scopes,
                top_k=top_k,
            )
            if success:
                return ToolExecResult(output=result)
            return ToolExecResult(error=result, error_code=1)
        except Exception as exc:
            logger.exception("Feature search failed")
            return ToolExecResult(error=f"Error: {exc}", error_code=1)
