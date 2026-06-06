"""Production :class:`LanguageBackend` implementation for Go.

This backend currently implements the skeleton-relevant subset needed
for ``FileDesigner`` to emit ``.go`` files and skip ``__init__.py``
package markers. Code-structure, test-runner, and output-parser
methods raise :class:`NotImplementedError` until the decoder stages use
Go-specific implementations for those behaviours.

Reference for Go conventions consulted:
* ``$GOROOT/src`` and Go's effective package guide — directories *are*
  packages, no marker file required.
* ``go test`` convention — sibling ``*_test.go`` files; no separate
  ``tests/`` tree by default.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .prompt_hints import PromptHints
from .test_result import EnvHandle, TestRunResult

logger = logging.getLogger(__name__)

# Go identifier rule: ASCII letters / digits / underscore; cannot start
# with a digit. Hyphens are illegal (unlike many tools' file-name
# conventions, Go's *package* names must be valid identifiers).
_GO_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GO_IDENT_INVALID = re.compile(r"[^A-Za-z0-9_]")

# A short list of Go reserved words. Used only for identifier
# validation; not a parser. Source: Go language spec §"Keywords".
_GO_KEYWORDS = frozenset({
    "break", "case", "chan", "const", "continue", "default", "defer",
    "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
    "interface", "map", "package", "range", "return", "select",
    "struct", "switch", "type", "var",
})


class GoBackend:
    """Skeleton-stage :class:`LanguageBackend` for Go.

    See :class:`decoder_lang.backend.LanguageBackend` for method
    contracts. Implemented methods cover file/test classification, the
    no-op package marker, identifier rules, and prompt hints. Code
    analysis and test-runner methods raise :class:`NotImplementedError`
    so unsupported paths fail explicitly.
    """

    name = "go"
    display_name = "Go"
    file_extension = ".go"
    markdown_fence = "go"

    # ------------------------------------------------------------------
    # 1. File & package layout
    # ------------------------------------------------------------------

    def is_source_file(self, path: str) -> bool:
        # ``*_test.go`` are still .go files but the caller is expected
        # to use :meth:`is_test_file` to separate tests from sources;
        # to mirror :class:`PythonBackend.is_source_file` (which does
        # NOT exclude tests), we keep the same convention here.
        return path.endswith(".go")

    def is_test_file(self, path: str) -> bool:
        normalised = path.replace("\\", "/")
        basename = normalised.rsplit("/", 1)[-1]
        return basename.endswith("_test.go")

    def package_marker_filename(self) -> str | None:
        # Go packages are directories; no marker file required.
        return None

    def package_marker_content(self, pkg_path: str) -> str | None:
        # Returning None makes call sites skip creation entirely.
        return None

    def is_valid_module_identifier(self, segment: str) -> bool:
        if not segment or segment in _GO_KEYWORDS:
            return False
        return bool(_GO_IDENT_RE.match(segment))

    def sanitize_module_identifier(self, segment: str) -> str:
        if not segment:
            return "_"
        cleaned = _GO_IDENT_INVALID.sub("_", segment)
        if cleaned[:1].isdigit():
            cleaned = f"_{cleaned}"
        # Avoid clashing with a Go keyword by suffixing an underscore;
        # never strip user content.
        if cleaned in _GO_KEYWORDS:
            cleaned = f"{cleaned}_"
        return cleaned

    # ------------------------------------------------------------------
    # 2. Code structure — not implemented for Go yet
    # ------------------------------------------------------------------

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        raise NotImplementedError(
            "GoBackend.has_placeholder is not implemented; "
            "the current Go backend supports skeleton-stage behaviour only.",
        )

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        raise NotImplementedError(
            "GoBackend.syntax_check is not implemented.",
        )

    def list_code_units(self, code: str, path: str = "<string>") -> list:
        raise NotImplementedError(
            "GoBackend.list_code_units is not implemented.",
        )

    def format_signature(self, unit) -> str:  # type: ignore[override]
        raise NotImplementedError(
            "GoBackend.format_signature is not implemented.",
        )

    def list_imports(self, code: str, path: str = "<string>") -> list:
        raise NotImplementedError(
            "GoBackend.list_imports is not implemented.",
        )

    # ------------------------------------------------------------------
    # 3. Build / test environment — not implemented for Go yet
    # ------------------------------------------------------------------

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        raise NotImplementedError(
            "GoBackend.detect_env is not implemented.",
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        raise NotImplementedError(
            "GoBackend.ensure_env is not implemented.",
        )

    def test_command(
        self,
        env: EnvHandle,
        selectors: list[str] | None = None,
    ) -> list[str]:
        raise NotImplementedError(
            "GoBackend.test_command is not implemented.",
        )

    def install_deps_command(
        self,
        env: EnvHandle,
        deps: list[str],
    ) -> list[str] | None:
        raise NotImplementedError(
            "GoBackend.install_deps_command is not implemented.",
        )

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        raise NotImplementedError(
            "GoBackend.parse_test_output is not implemented.",
        )

    # ------------------------------------------------------------------
    # 4. Prompt hints
    # ------------------------------------------------------------------

    _PROMPT_HINTS_SINGLETON: PromptHints | None = None

    def prompt_hints(self) -> PromptHints:
        cached = GoBackend._PROMPT_HINTS_SINGLETON
        if cached is not None:
            return cached
        hints = PromptHints(
            display_name=self.display_name,
            markdown_fence=self.markdown_fence,
            file_extension=self.file_extension,
            module_naming_rule=(
                "Use short, lowercase package directory names with no "
                "underscores; tests live next to source as <name>_test.go."
            ),
            package_layout_example=(
                "cmd/\n"
                "  myapp/\n"
                "    main.go\n"
                "internal/\n"
                "  core/\n"
                "    core.go\n"
                "    core_test.go\n"
                "go.mod\n"
            ),
            entrypoint_example="cmd/<name>/main.go",
            test_framework_name="go test",
            style_directive=(
                "Write idiomatic Go: short, lowercase package names; "
                "explicit error returns; small interfaces consumed at "
                "the call site rather than declared up-front."
            ),
        )
        GoBackend._PROMPT_HINTS_SINGLETON = hints
        return hints


__all__ = ["GoBackend"]
