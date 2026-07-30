"""Validation and defensive sanitization for dashboard snapshot v1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_DROP_KEYS = {
    "prompt",
    "response",
    "raw_response",
    "parsed_result",
    "stdout",
    "stderr",
    "content",
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
}

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
)


def _sanitize_text(value: str) -> str:
    sanitized = value
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)(api"):
            sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
        else:
            sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized[:20_000] + ("...[truncated]" if len(sanitized) > 20_000 else "")


def sanitize_snapshot(value: Any) -> Any:
    """Return a JSON-safe copy with prompts, source bodies, and secrets removed."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_snapshot(item)
            for key, item in value.items()
            if str(key).lower() not in _DROP_KEYS
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value))


def validate_snapshot(snapshot: Any) -> list[str]:
    """Validate the stable structural contract consumed by the future renderer."""
    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["snapshot must be an object"]
    if snapshot.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    expected_types = {
        "generated_at": str,
        "workspace": dict,
        "current_state": dict,
        "pipeline": list,
        "rpg": dict,
        "graph": dict,
        "tasks": dict,
        "rpg_edit": dict,
        "artifacts": list,
        "telemetry": dict,
        "verification": dict,
        "runs": list,
        "trends": dict,
        "source_health": list,
    }
    for key, expected_type in expected_types.items():
        if not isinstance(snapshot.get(key), expected_type):
            errors.append(f"{key} must be {expected_type.__name__}")

    workspace = snapshot.get("workspace")
    if isinstance(workspace, dict):
        for key in ("name", "path", "mode"):
            if not isinstance(workspace.get(key), str):
                errors.append(f"workspace.{key} must be str")

    runs = snapshot.get("runs")
    if isinstance(runs, list):
        for run_index, run in enumerate(runs):
            prefix = f"runs[{run_index}]"
            if not isinstance(run, dict):
                errors.append(f"{prefix} must be object")
                continue
            for key in ("run_id", "command", "status", "display_status"):
                if not isinstance(run.get(key), str):
                    errors.append(f"{prefix}.{key} must be str")
            if not isinstance(run.get("stages"), list):
                errors.append(f"{prefix}.stages must be list")
                continue
            for stage_index, stage in enumerate(run["stages"]):
                stage_prefix = f"{prefix}.stages[{stage_index}]"
                if not isinstance(stage, dict):
                    errors.append(f"{stage_prefix} must be object")
                    continue
                for key in ("stage_id", "name", "status"):
                    if not isinstance(stage.get(key), str):
                        errors.append(f"{stage_prefix}.{key} must be str")

    forbidden = _find_forbidden_keys(snapshot)
    errors.extend(f"forbidden sensitive key: {path}" for path in forbidden)
    return errors


def _find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in _DROP_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return found


def assert_valid_snapshot(snapshot: Any) -> None:
    errors = validate_snapshot(snapshot)
    if errors:
        raise ValueError("Invalid dashboard snapshot: " + "; ".join(errors))