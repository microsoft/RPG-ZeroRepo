"""Project-level task prompt templates owned by language backends."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectTaskContext:
    """Context used to render language-specific project tasks."""

    repo_name: str
    repo_info: str
    package_name: str


@dataclass(frozen=True)
class ProjectTaskTemplates:
    """Rendered project-level task prompts for a target language."""

    dependencies: str
    main_entry: str
    readme: str