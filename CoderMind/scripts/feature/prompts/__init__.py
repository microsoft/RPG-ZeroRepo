"""LLM prompt templates for feature-related stages.

This package collects prompt templates grouped by stage.  Compatibility
prompts live in ``feature.prompts.legacy`` and are re-exported here so
existing imports (``from feature.prompts import PROMPT_TEMPLATE_BUILD_FEATURE``
etc.) continue to work unchanged.

New stages add their prompts in dedicated submodules and re-export from
here as needed.
"""

# Back-compat: re-export every public symbol from the legacy module.
from .legacy import *  # noqa: F401,F403

# New, dedicated prompt modules.
from .spec import (  # noqa: F401
    PROMPT_TEMPLATE_FEATURE_SPEC_SYSTEM,
    PROMPT_TEMPLATE_FEATURE_SPEC_USER,
)
