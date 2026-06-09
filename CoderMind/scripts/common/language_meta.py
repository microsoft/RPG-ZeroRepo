"""Helpers for language metadata in generated decoder artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PRIMARY_LANGUAGE_FIELD = "primary_language"
TARGET_LANGUAGES_FIELD = "target_languages"

_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "cplusplus": "cpp",
    "cc": "cpp",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
}


def canonical_language_name(value: str) -> str:
    """Return the decoder's canonical language key for a raw name."""
    cleaned = value.strip().lower()
    return _LANGUAGE_ALIASES.get(cleaned, cleaned)


def normalize_language_metadata(
    primary: Any = None,
    languages: Any = None,
) -> tuple[str | None, list[str]]:
    """Returns a primary language and normalized ordered language list."""
    normalized: list[str] = []
    if isinstance(languages, list):
        for language in languages:
            if isinstance(language, str):
                cleaned = canonical_language_name(language)
                if cleaned and cleaned not in normalized:
                    normalized.append(cleaned)

    clean_primary = None
    if isinstance(primary, str):
        candidate = canonical_language_name(primary)
        if candidate:
            clean_primary = candidate

    if clean_primary:
        if clean_primary in normalized:
            normalized.remove(clean_primary)
        normalized.insert(0, clean_primary)
    elif normalized:
        clean_primary = normalized[0]

    return clean_primary, normalized


def extract_language_metadata(data: Any) -> tuple[str | None, list[str]]:
    """Reads canonical language metadata from ``meta``."""
    meta = _get_value(data, "meta")
    primary = _get_value(meta, PRIMARY_LANGUAGE_FIELD)
    languages = _get_value(meta, TARGET_LANGUAGES_FIELD)
    return normalize_language_metadata(primary, languages)


def metadata_with_languages(data: Any, base_meta: Any = None) -> dict[str, Any]:
    """Returns a metadata object with canonical language fields."""
    meta = _get_value(data, "meta") if base_meta is None else base_meta
    result = dict(meta) if isinstance(meta, Mapping) else {}
    primary, languages = extract_language_metadata(data)
    result[PRIMARY_LANGUAGE_FIELD] = primary
    result[TARGET_LANGUAGES_FIELD] = languages
    return result


def _get_value(data: Any, key: str) -> Any:
    if isinstance(data, Mapping):
        return data.get(key)
    return getattr(data, key, None)