"""Helpers for injecting a language-specific preamble into decoder prompts.

Phase 5 ships only the wiring \u2014 a small ``language_directive`` builder
that the prompt-rendering call sites can prepend to any system prompt
when the target language is not Python. Bulk substitution of literal
``"Python"`` / ``".py"`` / ``"pytest"`` strings across the prompt
files is deferred to Phase 6 (real cobra/Go decoder run) so each
edit is driven by an actual quality signal rather than a textual
diff exercise.

Design:

* When the resolved language is ``"python"`` the directive is the
  empty string \u2014 prompts render byte-identically to the pre-Phase-5
  output, so existing Python pipelines are zero-impact.
* When the language differs, a short directive (display name,
  one-line style note, markdown fence reminder) is prepended so the
  LLM at least *knows* the target language even before per-prompt
  rewrites land.
"""
from __future__ import annotations

from typing import Optional

from .backend import LanguageBackend


_PYTHON_DEFAULT_NAME = "python"


def language_directive(backend: Optional[LanguageBackend]) -> str:
    """Return a short preamble appropriate for ``backend``'s language.

    Empty string for Python (and for ``backend is None``) so that
    callers can unconditionally prepend the return value without
    introducing any diff to the existing Python prompt output. For
    every other language the preamble carries the display name, a
    style directive, and a markdown-fence reminder so the LLM emits
    the right kind of code.
    """
    if backend is None or backend.name == _PYTHON_DEFAULT_NAME:
        return ""
    hints = backend.prompt_hints()
    # Compact, neutral-tone preamble. Two newlines after the block so
    # it visibly separates from whatever the caller appends next.
    lines = [
        f"### Target language: {hints.display_name}",
        hints.style_directive.strip(),
        (
            f"Emit all code fences as ```{hints.markdown_fence} \u2026 ```. "
            f"Source files use the ``{hints.file_extension}`` extension. "
            f"Test framework: {hints.test_framework_name}."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def with_language_directive(
    system_prompt: str,
    backend: Optional[LanguageBackend],
) -> str:
    """Convenience: prepend ``language_directive(backend)`` to a system
    prompt body. Returns ``system_prompt`` unchanged when the
    directive is empty (Python or no backend supplied)."""
    directive = language_directive(backend)
    if not directive:
        return system_prompt
    return directive + system_prompt
