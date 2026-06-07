"""Production :class:`LanguageBackend` implementation for Go."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .prompt_hints import PromptHints
from .test_result import EnvHandle, TestFailure, TestRunResult

logger = logging.getLogger(__name__)

# Go identifier rule: ASCII letters / digits / underscore; cannot start
# with a digit. Hyphens are illegal (unlike many tools' file-name
# conventions, Go's *package* names must be valid identifiers).
_GO_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GO_IDENT_INVALID = re.compile(r"[^A-Za-z0-9_]")
_PLACEHOLDER_RE = re.compile(
    r"(?is)\b(?:return|panic\s*\()\s*(?:\"[^\"]*|`[^`]*|'[^']*)"
    r"(?:TODO|PLACEHOLDER|NOT IMPLEMENTED)"
)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_GO_TEST_RUN_RE = re.compile(r"^===\s+RUN\s+(\S+)")
_GO_TEST_EVENT_RE = re.compile(r"^---\s+(PASS|FAIL|SKIP):\s+(\S+)(?:\s+\(([^)]*)\))?")
_GO_TEST_PACKAGE_RE = re.compile(r"^(ok|FAIL)\s+\S+\s+([0-9.]+)s\b")
_GO_TEST_FILE_LINE_RE = re.compile(r"^\s*([^\s:]+_test\.go):(\d+):\s*(.*)$")

# A short list of Go reserved words. Used only for identifier
# validation; not a parser. Source: Go language spec §"Keywords".
_GO_KEYWORDS = frozenset({
    "break", "case", "chan", "const", "continue", "default", "defer",
    "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
    "interface", "map", "package", "range", "return", "select",
    "struct", "switch", "type", "var",
})


class GoBackend:
    """:class:`LanguageBackend` for Go source."""

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
    # 2. Code structure
    # ------------------------------------------------------------------

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        ok, _ = self.syntax_check(code, path)
        if not ok:
            return False
        stripped = _BLOCK_COMMENT_RE.sub("", _LINE_COMMENT_RE.sub("", code))
        return bool(_PLACEHOLDER_RE.search(stripped))

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        parser = self._parser()
        return parser.validate_syntax(self._parse_path(path), code)

    def list_code_units(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(code, path)
        if result is None or result.syntax_error:
            return []
        return [
            unit for unit in result.units
            if unit.unit_type in {"struct", "interface", "function", "method"}
        ]

    def format_signature(self, unit: Any) -> str:
        if unit is None:
            return ""
        name = getattr(unit, "name", None) or ""
        if getattr(unit, "unit_type", None) not in {"function", "method"}:
            return name
        code = (getattr(unit, "code", "") or "").strip()
        if not code:
            return name
        signature_lines: list[str] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "{" in stripped:
                stripped = stripped.split("{", 1)[0].rstrip()
                if stripped:
                    signature_lines.append(stripped)
                break
            signature_lines.append(stripped)
            if getattr(unit, "unit_type", None) in {"struct", "interface"}:
                break
            if stripped.endswith(")") or stripped.endswith(") error"):
                break
        if not signature_lines:
            return name
        return " ".join(signature_lines)

    def list_imports(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(code, path)
        if result is None or result.syntax_error:
            return []
        return [dep for dep in result.dependencies if dep.relation == "imports"]

    # ------------------------------------------------------------------
    # 3. Build / test environment
    # ------------------------------------------------------------------

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        go_exe = shutil.which("go")
        if not go_exe:
            return None
        root = repo_root.resolve()
        module_file = root / "go.mod"
        return EnvHandle(
            project_root=root,
            runtime_executable=go_exe,
            extra={
                "module_file": str(module_file) if module_file.exists() else None,
                "module": self._read_module_name(module_file),
            },
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        env = self.detect_env(repo_root)
        if env is None:
            raise ToolchainUnavailable("Go toolchain is not available on PATH")
        module_file = env.project_root / "go.mod"
        if not module_file.exists():
            module_name = self._default_module_name(env.project_root)
            module_file.write_text(
                f"module {module_name}\n\ngo 1.22\n",
                encoding="utf-8",
            )
            return EnvHandle(
                project_root=env.project_root,
                runtime_executable=env.runtime_executable,
                extra={"module_file": str(module_file), "module": module_name},
            )
        return env

    def test_command(
        self,
        env: EnvHandle,
        selectors: list[str] | None = None,
    ) -> list[str]:
        go_exe = env.runtime_executable or "go"
        cmd = [go_exe, "test"]
        if selectors:
            cmd.extend(["-run", "|".join(selectors)])
        cmd.append("./...")
        return cmd

    def install_deps_command(
        self,
        env: EnvHandle,
        deps: list[str],
    ) -> list[str] | None:
        if not deps:
            return None
        go_exe = env.runtime_executable or "go"
        return [go_exe, "get", *deps]

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        duration_sec = 0.0
        failures: list[TestFailure] = []
        current_test: str | None = None
        output_by_test: dict[str, list[str]] = {}

        for line in raw.splitlines():
            started = _GO_TEST_RUN_RE.match(line)
            if started:
                current_test = started.group(1)
                output_by_test.setdefault(current_test, [])
                continue

            event = _GO_TEST_EVENT_RE.match(line)
            if event:
                kind, test_name, duration_text = event.groups()
                if kind == "PASS":
                    passed_count += 1
                elif kind == "SKIP":
                    skipped_count += 1
                elif kind == "FAIL":
                    failed_count += 1
                    long_message = "\n".join(output_by_test.get(test_name, [])).strip()
                    file_path, line_number, message = self._failure_location(long_message)
                    short_message = message or f"{test_name} failed"
                    failures.append(TestFailure(
                        test_id=test_name,
                        short_message=short_message,
                        long_message=long_message,
                        file_path=file_path,
                        line=line_number,
                    ))
                duration_sec += self._parse_duration(duration_text)
                current_test = None
                continue

            package = _GO_TEST_PACKAGE_RE.match(line)
            if package:
                duration_sec = max(duration_sec, self._parse_duration(package.group(2)))
                continue

            if current_test:
                output_by_test.setdefault(current_test, []).append(line)

        if exit_code == 0:
            status = "passed"
        elif failed_count:
            status = "failed"
        else:
            status = "errored"

        return TestRunResult(
            status=status,
            exit_code=exit_code,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=0 if status != "errored" else 1,
            skipped_count=skipped_count,
            duration_sec=duration_sec,
            failures=failures,
            raw_output=raw,
            extra={"tool": "go test"},
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

    @staticmethod
    def _parser() -> Any:
        from lang_parser import get_parser  # type: ignore

        return get_parser("go")

    @staticmethod
    def _parse_path(path: str) -> str:
        if path == "<string>" or not path.endswith(".go"):
            return "main.go"
        return path

    def _parse(self, code: str, path: str):
        parser = self._parser()
        try:
            return parser.parse_file(self._parse_path(path), code)
        except Exception:
            logger.exception("Failed to parse Go source: %s", path)
            return None

    @staticmethod
    def _parse_duration(duration_text: str | None) -> float:
        if not duration_text:
            return 0.0
        text = duration_text.rstrip("s")
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _read_module_name(module_file: Path) -> str | None:
        try:
            for line in module_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("module "):
                    return stripped.split(None, 1)[1]
        except OSError:
            return None
        return None

    @staticmethod
    def _default_module_name(repo_root: Path) -> str:
        raw_name = repo_root.name.lower()
        module_leaf = re.sub(r"[^a-z0-9._/-]+", "-", raw_name).strip("-./")
        return f"codermind.local/{module_leaf or 'module'}"

    @staticmethod
    def _failure_location(text: str) -> tuple[str | None, int | None, str | None]:
        for line in text.splitlines():
            match = _GO_TEST_FILE_LINE_RE.match(line)
            if match:
                file_path, line_number, message = match.groups()
                return file_path, int(line_number), message or None
        return None, None, None


__all__ = ["GoBackend"]
