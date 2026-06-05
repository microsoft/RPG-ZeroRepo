"""Decoder language abstraction layer.

This package introduces a :class:`LanguageBackend` strategy interface
that lets the decoder pipeline (skeleton / func_design / code_gen)
treat the target programming language as a parameter rather than
a hard-coded ``.py`` / ``ast`` / ``pytest`` assumption.

Phase 0 (current): the abstraction exists but the decoder still routes
exclusively through :class:`PythonBackend`, whose behaviour matches the
pre-existing Python-only logic byte-for-byte. Later phases add
``GoBackend`` etc. and migrate decoder call sites to look up the
backend via :func:`get_backend` based on the project's target language
(resolved from RPG ``meta.language``).

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
# the decoder's default; Go is the first non-Python backend (Phase 2
# ships the skeleton-stage subset; AST and test-runner methods raise
# NotImplementedError until Phase 3/4).
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
