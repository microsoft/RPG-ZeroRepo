"""Prompt-hint dataclass used by :meth:`LanguageBackend.prompt_hints`.

Holds the strings the decoder injects into LLM prompts so a single
prompt template can render correctly for any target language. Prompt
renderers use these fields for display names, markdown fences, file
extensions, test framework names, and language-specific style guidance.

Kept deliberately small: only fields a prompt template can reference
verbatim. Anything that needs computation (e.g. signature extraction)
stays on :class:`LanguageBackend` as a method.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptHints:
    """Per-language prompt-template fill values.

    All fields are short literals safe to substitute into any prompt
    body. Backends construct one instance at module load and return it
    unchanged from :meth:`LanguageBackend.prompt_hints` so callers can
    cache it without worrying about identity.
    """

    # Identity / display
    display_name: str                 # "Python" / "Go"
    markdown_fence: str               # "python" / "go" (after ```)
    file_extension: str               # ".py" / ".go"

    # Layout / naming guidance
    module_naming_rule: str           # single-sentence rule for file names
    package_layout_example: str       # short ASCII tree of idiomatic layout
    entrypoint_example: str           # e.g. "src/main.py" / "cmd/<name>/main.go"

    # Test framework
    test_framework_name: str          # "pytest" / "go test"

    # Free-form preamble injected at the top of every code-gen prompt
    # to nudge the LLM toward idiomatic style. Keep concise (≤ 3
    # sentences) to avoid prompt bloat.
    style_directive: str
