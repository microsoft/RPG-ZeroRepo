from __future__ import annotations
# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Base class for tool-related errors."""

class ToolNotFoundError(ToolError):
    pass

class ToolValidationError(ToolError):
    pass

class ToolExecutionError(ToolError):
    pass
