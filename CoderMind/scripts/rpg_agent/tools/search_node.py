#!/usr/bin/env python3
"""Unified Search Node Tool.

Integrates feature-based search (from search_by_feature.py) and
code-based search (from search_by_meta.py) into a single tool interface.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/search_node.py
Adaptations:
  - CoderMind imports (scripts.common.tools, scripts.rpg_agent)
  - search_code_snippets takes file2code dict instead of repo_skeleton/bm25_retriever
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from common.tools import Tool, ToolCallArguments, ToolExecResult
from rpg_agent.env.searcher import RepoDependencySearcher, RepoEntitySearcher
from rpg_agent.ops.search_by_feature import search_features_by_keywords
from rpg_agent.ops.search_by_meta import search_code_snippets
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Description
# ============================================================================

SEARCH_NODE_DESC = """
### Tool Name: search_node
#### Description
Unified search tool for finding code entities in the repository. Use ONE of the two search modes:
1. Code Search:
- Search and retrieve concrete code from the repository using file paths, qualified names (file:Class.method), or raw text keywords.
- Supports direct symbol lookup and keyword-based code search, and can return full files, specific functions, or targeted lines.
- Do NOT search, open, or return any code from test-related files or directories (e.g., `tests/`, `test/`, `testing/`, `*_test.py`, `test_*.py`), unless the user explicitly requests it.
2. Feature Search:
- Search the repository using functional or behavioral descriptions when you do not know the exact class, function, or file name.
- This tool matches high-level feature or behavior terms to concrete code entities using the Functionality SubGraph -> Code SubGraph mapping.
- Tip: If the exact file/class/function name isn't obvious from the issue, use this tool (and try a few alternative feature_terms / nearby search_scopes) to map behavioral descriptions to concrete code entities before narrowing down.
#### Parameters
{
    "tool_name": "search_node",
    "parameters": {
        "code_search": {
            "search_terms": "<List of file paths, qualified code entities, or text keywords; for text search, use specific, high-signal keywords (e.g. identifiers or unique code fragments), as low-information tokens like '\\n' are not meaningful.>",
            "line_nums": "<List of two integers [start, end] to extract lines from a specific file. Requires an exact file path. Optional.>",
            "file_path_or_pattern": "<File path or glob pattern to restrict search scope. Default: '**/*.py'>"
        },
        "feature_search": {
            "feature_terms": "<List of feature names to search in RPG>",
            "search_scopes": "<List of feature paths to restrict search scope>"
        }
    }
}
#### Returns
Matched code entities, feature nodes, or code snippets based on search parameters.
#### Example JSON Calls
##### Example 1: Search for a specific code entity
{
    "tool_name": "search_code_snippets",
    "parameters": {
        "search_terms": ["src/my_file.py:MyClass.func_name"]
    }
}
##### Example 2: Search for features by name
{
    "tool_name": "search_node",
    "parameters": {
        "feature_search": {
            "feature_terms": ["abstract send prepared request"],
            "search_scopes": ["ConnectionAndAdapterManagement/adapter management/http adapters"]
        }
    }
}
"""


# ============================================================================
# Parameter Models
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


class CodeSearchParam(BaseModel):
    """Parameters for code-based search."""

    search_terms: Optional[List[str]] = Field(
        default=[],
        description="List of file paths, qualified code entities, or text keywords",
    )
    file_path_or_pattern: Optional[str] = Field(
        default="**/*.py",
        description="File path or glob pattern to restrict search scope",
    )
    line_nums: Optional[List[int]] = Field(
        default=[],
        description="List of two integers [start, end] to extract lines from a specific file",
    )

    @field_validator("search_terms", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("line_nums", mode="before")
    @classmethod
    def coerce_int_to_list(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            return [int(v)]
        return v


class SearchNodeParam(BaseModel):
    """Parameters for unified node search."""

    feature_search: Optional[FeatureSearchParam] = Field(
        default=None,
        description="Feature-based search parameters (search in RPG feature graph)",
    )
    code_search: Optional[CodeSearchParam] = Field(
        default=None,
        description="Code-based search parameters (search in code entities)",
    )


# ============================================================================
# Tool Class
# ============================================================================

class SearchNode(Tool):
    """Unified search tool for finding code entities in the repository.

    Source: RPG-ZeroRepo tools/search_node.py ``SearchNode``
    """

    ParamModel: type[BaseModel] = SearchNodeParam
    name: str = "search_node"
    description: str = SEARCH_NODE_DESC

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        """Parse raw JSON input and validate against SearchNodeParam."""
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
        """Execute the search tool with validated arguments."""
        action_dict = arguments
        env_dict = env or {}

        rpg: Optional[RPG] = env_dict.get("rpg")
        file2code: Dict[str, str] = env_dict.get("file2code", {})
        entity_searcher: Optional[RepoEntitySearcher] = env_dict.get("entity_searcher")
        dep_searcher: Optional[RepoDependencySearcher] = env_dict.get("dep_searcher")

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

        feature_search_dict = action_dict.get("feature_search")
        code_search_dict = action_dict.get("code_search")
        top_k = 5

        if not feature_search_dict and not code_search_dict:
            return ToolExecResult(
                error="No search parameters provided. Use 'feature_search' or 'code_search'.",
                error_code=1,
            )

        results: List[tuple] = []
        all_success: List[bool] = []

        # --- Feature search ---
        if feature_search_dict:
            feature_params = FeatureSearchParam(**feature_search_dict)
            feature_terms = feature_params.feature_terms or []
            search_scopes = feature_params.search_scopes or []

            if feature_terms:
                try:
                    result, success = search_features_by_keywords(
                        rpg=rpg,
                        entity_searcher=entity_searcher,
                        keywords=feature_terms,
                        search_scopes=search_scopes,
                        top_k=top_k,
                    )
                    results.append(("Feature Search", result))
                    all_success.append(success)
                except Exception as exc:
                    logger.exception("Feature search failed")
                    results.append(("Feature Search", f"Error: {exc}"))
                    all_success.append(False)
            else:
                results.append(("Feature Search", "No feature_terms provided for feature search."))
                all_success.append(False)

        # --- Code search ---
        if code_search_dict:
            code_params = CodeSearchParam(**code_search_dict)
            search_terms = code_params.search_terms or []
            file_path_or_pattern = code_params.file_path_or_pattern or "**/*.py"
            line_nums = code_params.line_nums or []

            if search_terms or line_nums:
                try:
                    result, success = search_code_snippets(
                        file2code=file2code,
                        entity_searcher=entity_searcher,
                        dep_searcher=dep_searcher,
                        search_terms=search_terms if search_terms else None,
                        line_nums=line_nums if line_nums else None,
                        file_path_or_pattern=file_path_or_pattern,
                    )
                    results.append(("Code Search", result))
                    all_success.append(success)
                except Exception as exc:
                    logger.exception("Code search failed")
                    results.append(("Code Search", f"Error: {exc}"))
                    all_success.append(False)
            else:
                results.append(("Code Search", "No search_terms or line_nums provided for code search."))
                all_success.append(False)

        # --- Format output ---
        if not results:
            return ToolExecResult(error="No results found.", error_code=1)

        if len(results) == 1:
            _title, content = results[0]
            if all_success[0]:
                return ToolExecResult(output=content)
            return ToolExecResult(error=content, error_code=1)

        combined = []
        for title, content in results:
            if content and content.strip():
                combined.append(f"=== {title} ===\n{content}")

        output = "\n\n".join(combined)
        if any(all_success):
            return ToolExecResult(output=output)
        return ToolExecResult(error=output, error_code=1)
