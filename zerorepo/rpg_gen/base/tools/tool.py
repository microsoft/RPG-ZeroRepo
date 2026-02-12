
import asyncio
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type, Union
import json
from pydantic import BaseModel, ValidationError
from .error import ToolValidationError

ToolCallArguments = Dict[str, Any]


@dataclass
class ToolExecResult:
    """Intermediate result of a tool execution."""

    output: Optional[str] = None
    error: Optional[str] = None
    error_code: int = 0
    state: Optional[Dict[str, Any]] = None

@dataclass
class ToolResult:
    """Final result of a tool call, surfaced by the executor."""
    name: str
    success: bool
    call_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    # Compatibility field if you need to carry an API-level id (e.g., OpenAI):
    id: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

@dataclass
class ToolCall:
    """Represents a parsed tool call input."""

    name: str
    call_id: str
    arguments: ToolCallArguments
    # Optional cross-provider identifier (kept for compatibility):
    id: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover
        return f"ToolCall(name={self.name}, call_id={self.call_id}, arguments={self.arguments}, id={self.id})"

    def to_dict(self) -> Dict:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments
        }
        
# ---------------------------------------------------------------------------
# Parameter model base (for documentation / shared mixins if needed)
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    """Abstract base for tool argument models.

    Note: This class is intentionally empty; concrete tools should define their own
    Pydantic models by subclassing :class:`pydantic.BaseModel` or this class.
    Keeping this around makes it easy to add shared mixins/validators later.
    """
    pass


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Abstract base for tools/actions with runtime Pydantic validation.

    Each concrete Tool may set ``ParamModel`` to a Pydantic model type.
    If ``ParamModel`` is None, the raw ``dict`` is passed through to ``execute``.
    """

    # Override in subclasses to enforce validation on input arguments.
    ParamModel: Optional[Type[BaseModel]] = None
    name: str = ""
    description: str = ""
    
    # --- Required metadata ---
    @classmethod
    def get_name(cls) -> str:
        """Return the canonical tool name."""
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return a concise human-readable description of the tool."""
        return cls.get_description
        
    # --- Core execution ---
    @classmethod
    @abstractmethod
    async def execute(cls, arguments: Dict[str, Union[ToolCallArguments, BaseModel]], env: Optional[Any]=None) -> ToolExecResult:
        """Run the tool with validated arguments.

        Implementations can type-narrow ``arguments`` to their custom ``ParamModel``.
        """
        pass
    
    
    # --- Validation hook ---
    @classmethod
    async def check(cls, arguments: ToolCallArguments) -> ToolCallArguments:
        """Validate/normalize input arguments using ``ParamModel`` if provided.

        Returns the validated/normalized payload (a Pydantic model instance or dict).
        Raise :class:`ToolValidationError` on failure.
        """
        if cls.ParamModel is None:
            return arguments
        try:
            return cls.ParamModel(**arguments).model_dump()
        except ValidationError as e:  # pragma: no cover - exercised in integration
            raise ToolValidationError(str(e)) from e

    # --- Optional lifecycle hooks ---
    @classmethod
    async def before_execute(cls, payload: Union[BaseModel, ToolCallArguments], env:Optional[Any]=None, **kwargs) -> None:
        """Hook called right before ``execute``. Override if needed."""
        return None

    @classmethod
    async def after_execute(cls, payload: ToolCallArguments, result: ToolExecResult, env:Optional[Any]=None, **kwargs) -> None:
        """Hook called right after ``execute``. Override if needed."""
        return None

    @classmethod
    # --- Resource cleanup ---
    async def close(cls) -> None:
        """Override to release resources if necessary."""
        return None

    @classmethod
    @abstractmethod
    def custom_parse(cls, raw: str) -> Optional[ToolCallArguments]:
        pass

