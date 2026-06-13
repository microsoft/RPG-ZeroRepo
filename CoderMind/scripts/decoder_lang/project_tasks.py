"""Project-level task prompt templates owned by language backends."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectTaskContext:
    """Context used to render language-specific project tasks."""

    repo_name: str
    repo_info: str
    package_name: str
    # Reconciled program entry-point path. The planner resolves this from
    # the already-designed interfaces (e.g. reusing an existing
    # ``cmd/<name>/main.go`` instead of generating a second command
    # package). When ``None``, the backend falls back to its canonical
    # entry path. Keeps entry *format* in the backend while letting the
    # planner — which alone sees the full interface set — decide *which*
    # file, so a single entry source is preserved.
    entry_point_path: str | None = None


@dataclass(frozen=True)
class ProjectTaskTemplates:
    """Rendered project-level task prompts for a target language."""

    dependencies: str
    main_entry: str
    readme: str