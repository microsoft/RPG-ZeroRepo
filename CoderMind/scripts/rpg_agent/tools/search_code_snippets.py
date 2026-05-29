#!/usr/bin/env python3
"""Code Search Tool — search and retrieve concrete code snippets from the repository using file paths, qualified names (file:Class.method), or raw text keywords.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/search_code_snippets.py
Adaptations:
  - search_code_snippets takes file2code dict instead of repo_skeleton/bm25_retriever
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from common.tools import Tool, ToolCallArguments, ToolExecResult
from rpg_agent.env.searcher import RepoDependencySearcher, RepoEntitySearcher
from rpg_agent.ops.search_by_meta import search_code_snippets
from rpg import RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Description
# ============================================================================

CODE_SEARCH_DESC = """
## Tool Name: search_code_snippets
### Description
- Search and retrieve concrete code from the repository using file paths, qualified names (file:Class.method), or raw text keywords.
- Supports direct symbol lookup and keyword-based code search, and can return full files, specific functions, or targeted lines.
- Do NOT search, open, or return any code from test-related files or directories (e.g., `tests/`, `test/`, `testing/`, `*_test.py`, `test_*.py`), unless the user explicitly requests it.
### Parameters
{
    "tool_name": "search_code_snippets",
    "parameters": {
        "search_terms": "<List of file paths, qualified code entities, or text keywords; for text search, use specific, high-signal keywords (e.g. identifiers or unique code fragments), as low-information tokens like '\\n' are not meaningful.>",
        "line_nums": "<List of two integers [start, end] to extract lines from a specific file. Requires an exact file path. Optional.>",
        "file_path_or_pattern": "<File path or glob pattern to restrict search scope. Default: '**/*.py'>"
    }
}
### Returns
Matched code entities or code snippets based on search parameters.
### Example JSON Calls
#### Example 1: Search for a specific code entity
{
    "tool_name": "search_code_snippets",
    "parameters": {
        "search_terms": ["src/auth/login.py:LoginHandler.authenticate"]
    }
}
#### Example 2: Search by keywords in a specific file
{
    "tool_name": "search_code_snippets",
    "parameters": {
        "search_terms": ["parse_response", "validate"],
        "file_path_or_pattern": "src/utils/*.py"
    }
}
"""


# ============================================================================
# Parameter Model
# ============================================================================

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


# ============================================================================
# Tool Class
# ============================================================================

class SearchCodeSnippets(Tool):
    """Code search tool for retrieving code snippets from the repository.

    Source: RPG-ZeroRepo tools/search_code_snippets.py ``SearchCodeSnippets``
    """

    ParamModel: type[BaseModel] = CodeSearchParam
    name: str = "search_code_snippets"
    description: str = CODE_SEARCH_DESC

    @classmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        """Parse raw JSON input and validate against CodeSearchParam."""
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
        """Execute the code search tool with validated arguments."""
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

        search_terms = action_dict.get("search_terms") or []
        file_path_or_pattern = action_dict.get("file_path_or_pattern") or "**/*.py"
        line_nums = action_dict.get("line_nums") or []

        if not search_terms and not line_nums:
            return ToolExecResult(
                error="No search_terms or line_nums provided for code search.",
                error_code=1,
            )

        try:
            result, success = search_code_snippets(
                file2code=file2code,
                entity_searcher=entity_searcher,
                dep_searcher=dep_searcher,
                search_terms=search_terms if search_terms else None,
                line_nums=line_nums if line_nums else None,
                file_path_or_pattern=file_path_or_pattern,
            )
            if success:
                return ToolExecResult(output=result)
            return ToolExecResult(error=result, error_code=1)
        except Exception as exc:
            logger.exception("Code search failed")
            return ToolExecResult(error=f"Error: {exc}", error_code=1)
