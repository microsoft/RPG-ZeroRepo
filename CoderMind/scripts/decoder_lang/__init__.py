"""Decoder language abstraction layer.

This package introduces a :class:`LanguageBackend` strategy interface
that lets the decoder pipeline (skeleton / func_design / code_gen)
treat the target programming language as a parameter rather than
a hard-coded ``.py`` / ``ast`` / ``pytest`` assumption.

The registry currently ships :class:`PythonBackend`, :class:`GoBackend`,
:class:`RustBackend`, :class:`TypeScriptBackend`, :class:`CBackend`, and
:class:`CppBackend` implementations. Decoder stages resolve the backend
from explicit feature-spec language, RPG metadata, or source-file
dominant language.

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
    default_find_existing_entry,
    get_backend,
    list_backends,
    register_backend,
    resolve_decoder_language,
    resolve_repo_backend,
    resolve_target_language,
    scan_repo_source_files,
)
from .c_backend import CBackend
from .cpp_backend import CppBackend
from .file_deps import FileDependencyEdge, infer_language_from_path
from .go_backend import GoBackend
from .javascript_backend import JavaScriptBackend
from .prompt_directive import language_directive, with_language_directive
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .python_backend import PythonBackend
from .rust_backend import RustBackend
from .test_result import EnvHandle, TestFailure, TestRunResult
from .typescript_backend import TypeScriptBackend

# Side-effect: register backends on package import so the registry is
# populated even when callers only ``import decoder_lang``. Python is
# the decoder's default; Go provides parser-backed code-structure and
# basic Go toolchain/test-runner behavior.
register_backend(PythonBackend)
register_backend(GoBackend)
register_backend(RustBackend)
register_backend(TypeScriptBackend)
register_backend(JavaScriptBackend)
register_backend(CBackend)
register_backend(CppBackend)

__all__ = [
    "EnvHandle",
    "FileDependencyEdge",
    "CBackend",
    "CppBackend",
    "GoBackend",
    "JavaScriptBackend",
    "LanguageBackend",
    "PromptHints",
    "ProjectTaskContext",
    "ProjectTaskTemplates",
    "PythonBackend",
    "RustBackend",
    "TestFailure",
    "TestRunResult",
    "ToolchainUnavailable",
    "TypeScriptBackend",
    "default_find_existing_entry",
    "get_backend",
    "infer_language_from_path",
    "language_directive",
    "list_backends",
    "register_backend",
    "resolve_decoder_language",
    "resolve_repo_backend",
    "resolve_target_language",
    "scan_repo_source_files",
    "with_language_directive",
]
