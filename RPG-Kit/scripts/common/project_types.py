"""Project-type tokens carried in ``feature_spec.json`` ``meta``.

The LLM that generates ``feature_spec`` declares which user-facing surfaces
the project exposes. Downstream stages (skeleton design, plan_tasks
UI_POLISH, run_batch sub-agent prompts, rpg_edit visual recon) read these
tokens to decide whether to inject web-specific guidance, GUI tooling,
data-pipeline checks, etc.


This module is intentionally tiny — no dependency on RPG/dataflow code so
it stays cheap to import from validation utilities.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

logger = logging.getLogger(__name__)


# 8-token whitelist (UPPERCASE). Multiple may be selected per project.
ALLOWED_PROJECT_TYPES: frozenset[str] = frozenset({
    "WEB",       # HTTP endpoints rendering HTML pages for browsers
    "API",       # JSON / GraphQL endpoints, no HTML rendering
    "SERVICE",   # long-running daemon / worker / bot / scheduler
    "PIPELINE",  # batch ETL / DAG / Spark job / ML training (clear start+end)
    "CLI",       # command-line entry point with subcommands
    "GUI",       # desktop window with widgets
    "GAME",      # interactive real-time application with rendering loop
    "LIBRARY",   # importable package, no end-user interface
})

MAX_PROJECT_NOTES_LEN = 500


class ProjectTypesError(ValueError):
    """Raised when ``meta.project_types`` cannot be normalized to a non-empty set of whitelisted tokens. Callers (e.g. feature_build_validation) treat this as a hard stop — feature_spec must be regenerated."""


def validate_project_types(meta: dict) -> Tuple[List[str], str]:
    """Normalize ``meta.project_types`` and ``meta.project_notes``.

    Behaviour:
      * Tokens are upper-cased and trimmed.
      * Tokens not in :data:`ALLOWED_PROJECT_TYPES` are dropped with a warning.
      * Notes longer than :data:`MAX_PROJECT_NOTES_LEN` are truncated.
      * Empty notes log a warning but do not fail.
      * Empty/missing ``project_types`` (or all rejected) raises
        :class:`ProjectTypesError`.

    Returns:
    -------
    ``(types, notes)`` — ``types`` is a deduplicated, alphabetically sorted
    list of valid uppercase tokens; ``notes`` is the (possibly truncated)
    free-form description.
    """
    raw_types: Iterable = meta.get("project_types") or []
    if isinstance(raw_types, str):
        # Tolerate "WEB,CLI" string form even though the spec asks for a list.
        raw_types = [s for s in raw_types.replace(";", ",").split(",")]

    normalized: List[str] = []
    for tok in raw_types:
        if not isinstance(tok, str):
            continue
        upper = tok.strip().upper()
        if upper:
            normalized.append(upper)

    accepted = sorted({t for t in normalized if t in ALLOWED_PROJECT_TYPES})
    rejected = sorted({t for t in normalized if t not in ALLOWED_PROJECT_TYPES})

    if rejected:
        logger.warning(
            "meta.project_types contains unknown tokens (ignored): %s",
            rejected,
        )

    if not accepted:
        raise ProjectTypesError(
            "meta.project_types must contain at least one of "
            f"{sorted(ALLOWED_PROJECT_TYPES)}; got {list(raw_types)!r}. "
            "Re-run feature_spec or fix the source documents."
        )

    notes_raw = meta.get("project_notes")
    notes = (notes_raw or "").strip() if isinstance(notes_raw, str) else ""
    if len(notes) > MAX_PROJECT_NOTES_LEN:
        logger.warning(
            "meta.project_notes truncated to %d chars (was %d)",
            MAX_PROJECT_NOTES_LEN, len(notes),
        )
        notes = notes[:MAX_PROJECT_NOTES_LEN]
    if not notes:
        logger.warning(
            "meta.project_notes is empty; downstream prompts will lack context"
        )

    return accepted, notes


def has_type(meta: dict, token: str) -> bool:
    """Convenience: True iff ``meta.project_types`` includes ``token`` (case- insensitive). Returns False on any validation failure so callers don't need to handle exceptions when probing optional behaviour."""
    target = token.strip().upper()
    if target not in ALLOWED_PROJECT_TYPES:
        return False
    try:
        types, _ = validate_project_types(meta)
    except Exception:
        return False
    return target in types
