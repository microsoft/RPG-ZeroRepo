#!/usr/bin/env python3
"""Feature Module for CoderMind.

This module provides prompt templates for feature tree operations:
- Feature build (expansion and review)
- Feature edit (planning and review)
- Feature refactor (planning and organization)
"""

from .prompts import (
    # Feature Build Prompts
    PROMPT_TEMPLATE_BUILD_REVIEW,
    PROMPT_TEMPLATE_BUILD_FEATURE,
    PROMPT_TEMPLATE_BUILD_DIRECTED,
    PROMPT_TEMPLATE_SUGGEST_DIRECTIONS,
    # Feature Edit Prompts
    PROMPT_TEMPLATE_EDIT_PLAN,
    PROMPT_TEMPLATE_EDIT_REVIEW,
    # Feature Refactor Prompts
    PROMPT_TEMPLATE_SUBTREE_PLANNING,
    PROMPT_TEMPLATE_FEATURE_ORGANIZATION,
)

__all__ = [
    # Feature Build
    "PROMPT_TEMPLATE_BUILD_REVIEW",
    "PROMPT_TEMPLATE_BUILD_FEATURE",
    "PROMPT_TEMPLATE_BUILD_DIRECTED",
    "PROMPT_TEMPLATE_SUGGEST_DIRECTIONS",
    # Feature Edit
    "PROMPT_TEMPLATE_EDIT_PLAN",
    "PROMPT_TEMPLATE_EDIT_REVIEW",
    # Feature Refactor
    "PROMPT_TEMPLATE_SUBTREE_PLANNING",
    "PROMPT_TEMPLATE_FEATURE_ORGANIZATION",
]
