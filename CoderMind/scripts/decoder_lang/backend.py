"""Strategy interface every per-language decoder backend implements.

A backend bundles all language-specific behaviour the decoder needs:

* file / package layout conventions (extension, package marker file,
  identifier rules);
* code-structure extraction (signatures, classes, functions,
  placeholder detection) — implementations delegate to
  :mod:`lang_parser` so encoder and decoder agree on AST semantics;
* build / test environment handling (venv / go mod / cargo);
* test command + native-output parsing;
* prompt fill values.

The interface is a runtime-checkable :class:`Protocol`; backends may
either subclass it or just match the structural shape. Implementations
are plain classes that satisfy the protocol so static type-checking and
``isinstance`` both work. Backends can implement a useful subset while
raising :class:`NotImplementedError` for operations a decoder stage does
not call for that language.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Protocol,
    runtime_checkable,
)

from common.language_meta import extract_language_metadata

from .prompt_hints import PromptHints
from .test_result import EnvHandle, TestRunResult

# Re-exported for convenience of method signatures. Callers that want
# to consume the AST output can import LPCodeUnit / LPDependency
# directly from ``lang_parser`` — backends produce instances of those
# types so the decoder doesn't depend on ``ast``.
try:  # pragma: no cover - lang_parser is a peer package
    from lang_parser import LPCodeUnit, LPDependency  # type: ignore
except ImportError:  # pragma: no cover
    LPCodeUnit = Any  # type: ignore[assignment,misc]
    LPDependency = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolchainUnavailable(RuntimeError):
    """Raised by :meth:`LanguageBackend.ensure_env` when the host machine
    lacks the compiler / runtime needed for the language (e.g. no ``go``
    binary on PATH for :class:`GoBackend`).

    Callers in the decoder must translate this into a
    :class:`~decoder_lang.test_result.TestRunResult` with
    ``status="skipped"`` so the verification step becomes a non-fatal
    WARN rather than a crash. The :class:`PythonBackend` never raises
    this; if the Python interpreter that imported the decoder is
    running, the toolchain is by definition available.
    """


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


@runtime_checkable
class LanguageBackend(Protocol):
    """Per-language behaviour bundle consumed by the decoder pipeline.

    Implementations are expected to be stateless (call-safe from
    multiple threads), cheap to construct, and idempotent. The
    registry caches one instance per language; backends MUST NOT
    store per-project state on ``self``.
    """

    # --- Identity --------------------------------------------------------

    name: str                          # registry key, e.g. "python"
    display_name: str                  # human-readable, e.g. "Python"
    file_extension: str                # primary source extension, e.g. ".py"
    markdown_fence: str                # code-fence tag, e.g. "python"

    # --- 1. File & package layout ---------------------------------------

    def is_source_file(self, path: str) -> bool:
        """Return True when ``path`` is a source file the decoder should
        treat as compilable for this language. Should exclude test
        files (callers use :meth:`is_test_file` for those)."""

    def is_test_file(self, path: str) -> bool:
        """Return True for test files (e.g. ``tests/test_*.py``,
        ``*_test.go``)."""

    def package_marker_filename(self) -> str | None:
        """File that signals a directory is a package in this language
        (``__init__.py`` for Python; ``None`` for Go / Rust / TypeScript
        where directories are packages by virtue of containing source
        files)."""

    def package_marker_content(self, pkg_path: str) -> str | None:
        """Initial body for the marker file, or ``None`` if the marker
        does not exist for this language."""

    def is_valid_module_identifier(self, segment: str) -> bool:
        """Check whether ``segment`` is a legal module / package name
        component for this language (used by skeleton path validation)."""

    def sanitize_module_identifier(self, segment: str) -> str:
        """Convert an arbitrary string into a legal module name
        component, e.g. replacing hyphens with underscores for Python.
        Idempotent: ``sanitize(sanitize(s)) == sanitize(s)``."""

    # --- 2. Code structure (delegates to lang_parser) -------------------

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        """Return True when ``code`` contains a return-placeholder
        pattern (TODO / PLACEHOLDER / NOT IMPLEMENTED string return)."""

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        """Return ``(ok, error_message)``. ``ok=True`` means the source
        parses cleanly; ``error_message`` is human-readable diagnostic
        text when ``ok=False`` (caller surfaces it to the LLM)."""

    def list_code_units(
        self,
        code: str,
        path: str = "<string>",
    ) -> list[Any]:
        """Return every class / function / method declaration in
        ``code`` as :class:`lang_parser.LPCodeUnit` instances (flat
        list including nested declarations).

        Backends implement this with whatever parsing infrastructure
        they have (stdlib ``ast`` for Python, tree-sitter via
        :mod:`lang_parser` for Go / Rust / etc.). On syntax errors,
        backends return an empty list rather than raising — callers
        already tolerate empty results and rendering the error to the
        LLM is the orchestrator's responsibility, not the backend's.
        """

    def format_signature(self, unit: Any) -> str:
        """Format a function / method :class:`LPCodeUnit` into a
        concise one-line signature, e.g.
        ``parse(data: bytes) -> Result``. Backends that can't extract
        a precise signature fall back to ``unit.name``."""

    def list_imports(
        self,
        code: str,
        path: str = "<string>",
    ) -> list[Any]:
        """Return :class:`lang_parser.LPDependency` records for every
        import statement in ``code``. On syntax error returns ``[]``."""

    # --- 3. Build / test environment ------------------------------------

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        """Return an :class:`EnvHandle` describing an existing
        environment (e.g. a ``.venv_dev`` directory for Python, a
        ``go.mod`` + ``go`` binary on PATH for Go). Never raises.
        Returns ``None`` when no usable env is present."""

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        """Detect-or-create the environment. Idempotent. Raises
        :class:`ToolchainUnavailable` when the language toolchain is
        missing from the host (compiler / runtime)."""

    def test_command(
        self,
        env: EnvHandle,
        selectors: list[str] | None = None,
    ) -> list[str]:
        """Return the ``subprocess`` argv to run the project's tests.
        ``selectors`` is a backend-specific filter (pytest ``-k``
        expression, ``go test -run`` regex, etc.); backends ignore it
        if not supported."""

    def install_deps_command(
        self,
        env: EnvHandle,
        deps: list[str],
    ) -> list[str] | None:
        """Return the argv to install third-party dependencies, or
        ``None`` when the language manages deps implicitly (Go modules
        auto-fetch on build)."""

    # --- 4. Test-output parsing -----------------------------------------

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        """Parse the native test-tool output (pytest text, ``go test``
        JSON, etc.) into the parser-agnostic
        :class:`TestRunResult`. Backends that fail to extract anything
        useful must still return a valid result (``raw_output=raw``)
        so callers can fall back to LLM-driven analysis."""

    # --- 5. Prompt hints ------------------------------------------------

    def prompt_hints(self) -> PromptHints:
        """Return the per-language strings the decoder injects into
        LLM prompts (see :class:`PromptHints` for the field list)."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, LanguageBackend] = {}
_DEFAULT_BACKEND_NAME = "python"


def register_backend(cls: type[LanguageBackend]) -> type[LanguageBackend]:
    """Register a backend class. Instantiates it once and stores the
    singleton under its ``name`` attribute.

    Can be used either as a decorator on the class definition or as a
    plain function call from package ``__init__``. Re-registering a
    backend silently replaces the previous entry (supports test
    fixtures that swap in fakes)."""
    instance = cls()  # type: ignore[call-arg]
    if not getattr(instance, "name", None):
        raise ValueError(f"backend {cls.__name__} has no .name attribute")
    _REGISTRY[instance.name] = instance
    logger.debug("Registered decoder backend: %s", instance.name)
    return cls


def get_backend(language: str | None) -> LanguageBackend:
    """Look up a backend by language name.

    Falls back to :class:`PythonBackend` with a single WARNING log
    when ``language`` is unrecognised. ``None`` maps to the default
    backend without warning because older artefacts may not carry an
    explicit language field.
    """
    if language is None:
        return _REGISTRY[_DEFAULT_BACKEND_NAME]
    backend = _REGISTRY.get(language)
    if backend is None:
        logger.warning(
            "No decoder backend for language=%r; falling back to %s",
            language,
            _DEFAULT_BACKEND_NAME,
        )
        return _REGISTRY[_DEFAULT_BACKEND_NAME]
    return backend


def list_backends() -> list[str]:
    """Return the names of all registered backends (test helper)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Convenience: resolve project language from RPG / fall back to dominant
# ---------------------------------------------------------------------------


def resolve_target_language(
    rpg_obj: Any,
    valid_files: Iterable[str] | None = None,
) -> str:
    """Determine the project's target language using a three-tier chain.

    1. ``rpg_obj["root"]["meta"]["language"]`` (written by encoder on
       :mod:`rpg_encoder.rpg_encoding`).
    2. ``lang_parser.dominant_language(valid_files)`` when the field is
       missing (older RPG artefacts) and ``valid_files`` is supplied.
    3. ``"python"`` as a last-resort default, with a WARNING log so
       silent fallbacks are visible during debugging.

    The helper lives in this package — rather than in
    :mod:`lang_parser` — because the "default to python" rule is
    decoder-specific. The encoder never assumes a default language.
    """
    # Tier 1: read from RPG root meta
    if isinstance(rpg_obj, dict):
        root = rpg_obj.get("root")
        if isinstance(root, dict):
            meta = root.get("meta")
            if isinstance(meta, dict):
                lang = meta.get("language")
                if isinstance(lang, str) and lang:
                    return lang

    # Tier 2: dominant_language over the provided file list
    if valid_files is not None:
        try:
            # Local import to avoid forcing lang_parser as a hard dep
            # of decoder_lang at import time. In practice the decoder
            # already imports lang_parser elsewhere, so this is just
            # defensive.
            from lang_parser import dominant_language  # type: ignore
        except ImportError:
            dominant_language = None  # type: ignore[assignment]
        if dominant_language is not None:
            inferred = dominant_language(valid_files)
            if inferred:
                return inferred

    # Tier 3: default to python with a single warning so the silent
    # fallback path is visible in logs.
    logger.warning(
        "resolve_target_language: no language on RPG root and no "
        "dominant_language fallback; defaulting to 'python'",
    )
    return _DEFAULT_BACKEND_NAME


def resolve_decoder_language(
    feature_spec: Any = None,
    rpg_obj: Any = None,
    valid_files: Iterable[str] | None = None,
) -> str:
    """Determine the target language for a *decoder* stage.

    Extends :func:`resolve_target_language` with a higher-priority
     tier 0: explicit language metadata on the loaded ``feature_spec.json``.
     Falls through to the same RPG-then-default
    chain when the field is absent.

    Tier order:

     0. ``feature_spec["meta"]["primary_language"]`` or the first
         ``feature_spec["meta"]["target_languages"]`` item.
    1. ``rpg_obj["root"]["meta"]["language"]``.
    2. ``lang_parser.dominant_language(valid_files)``.
    3. ``"python"`` default with WARNING.

    Returns the resolved language name; callers feed it directly to
    :func:`get_backend`. Never raises.
    """
    # Tier 0: explicit override on feature_spec
    if feature_spec is not None:
        primary, languages = extract_language_metadata(feature_spec)
        if primary:
            return primary
        if languages:
            return languages[0]
    # Tier 1-3 share the same logic as resolve_target_language.
    return resolve_target_language(rpg_obj, valid_files=valid_files)
