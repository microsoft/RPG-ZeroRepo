from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Optional, Union
import json
import logging
from copy import deepcopy
from zerorepo.rpg_gen.base.node import RepoSkeleton
from zerorepo.rpg_gen.base.rpg import RPG
from zerorepo.rpg_gen.base.tools import (
    Tool,
    ToolCallArguments,
    ToolExecResult
)
from ..ops.fetch_node import fetch_node


FETCH_NODE = """
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



class RetrieveParam(BaseModel):
    """
    Parameters for the `fetch_node` tool.
    Defines the input arguments that specify which entities to retrieve from the codebase.
    """

    code_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of code entities to retrieve information about. "
            "Each entry can be a file path, class name, or method identifier, "
            "for example: `['src/module/my_method.py']` or "
            "`['src/requests/adapters.py:BaseAdapter.send']`. "
            "If not provided, only `feature_entities` will be searched."
        )
    )

    feature_entities: Optional[List[str]] = Field(
        default=[],
        description=(
            "List of feature entities (functional paths) to retrieve information about. "
            "These describe specific functionality hierarchies within the codebase, "
            "e.g. `['CoreUtilitiesAndErrorHandling/string utilities/string compatibility/ensure string compatibility/check if string is ascii']`. "
            "If not provided, only `code_entities` will be searched."
        )
    )

    @field_validator('code_entities', 'feature_entities', mode='before')
    @classmethod
    def coerce_str_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class FetchNode(Tool):
    
    ParamModel: BaseModel=RetrieveParam
    
    name: str = "fetch_node"
    
    description: str = FETCH_NODE

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
            if tool_name.lower().strip() != cls.name.lower():
                logging.warning(f"Warning: tool_name '{tool_name}' does not match expected '{cls.name}'")
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
        entity_searcher = env_dict.get("entity_searcher")

        code_entities = action_dict.get("code_entities", [])
        feature_entities = action_dict.get("feature_entities", [])

        if not repo_rpg or not entity_searcher:
            return ToolExecResult(
                error="RPG or entity_searcher not available in environment",
                error_code=1
            )

        search_result, suc = fetch_node(
            rpg=repo_rpg,
            entity_searcher=entity_searcher,
            code_entities=code_entities,
            feature_entities=feature_entities
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