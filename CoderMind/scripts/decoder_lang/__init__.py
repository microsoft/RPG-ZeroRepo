"""Decoder language abstraction layer.

This package introduces a :class:`LanguageBackend` strategy interface
that lets the decoder pipeline (skeleton / func_design / code_gen)
treat the target programming language as a parameter rather than
a hard-coded ``.py`` / ``ast`` / ``pytest`` assumption.

The registry currently ships a full :class:`PythonBackend` plus a
Go backend with the skeleton-stage subset implemented. Decoder stages
resolve the backend from explicit feature-spec language, RPG metadata,
or source-file dominant language.

Public API (see :mod:`decoder_lang.backend` for full signatures):

* :class:`LanguageBackend` — Protocol every backend implements.
* :class:`PythonBackend` — production backend used by the existing
  Python decoder pipeline.
* :func:`get_backend` — registry lookup; falls back to Python with a
  single WARNING log when the requested language is unknown.
* :func:`register_backend` — decorator used by backend modules to
  self-register on import.
* :class:`PromptHints`, :class:`EnvHandle`, :class:`TestRunResult`,
  :class:`TestFailure` — value types passed across the interface.
"""
from __future__ import annotations

from .backend import (
    LanguageBackend,
    ToolchainUnavailable,
    get_backend,
    list_backends,
    register_backend,
    resolve_decoder_language,
    resolve_target_language,
)
from .go_backend import GoBackend
from .prompt_directive import language_directive, with_language_directive
from .prompt_hints import PromptHints
from .python_backend import PythonBackend
from .test_result import EnvHandle, TestFailure, TestRunResult

# Side-effect: register backends on package import so the registry is
# populated even when callers only ``import decoder_lang``. Python is
# the decoder's default; Go provides the skeleton-stage subset and
# raises ``NotImplementedError`` for unsupported code-analysis and
# test-runner operations.
register_backend(PythonBackend)
register_backend(GoBackend)

__all__ = [
    "EnvHandle",
    "GoBackend",
    "LanguageBackend",
    "PromptHints",
    "PythonBackend",
    "TestFailure",
    "TestRunResult",
    "ToolchainUnavailable",
    "get_backend",
    "language_directive",
    "list_backends",
    "register_backend",
    "resolve_decoder_language",
    "resolve_target_language",
    "with_language_directive",
]
