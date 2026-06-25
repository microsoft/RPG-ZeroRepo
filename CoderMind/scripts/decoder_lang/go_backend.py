"""Production :class:`LanguageBackend` implementation for Go."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .file_deps import FileDependencyEdge, resolved_go_edges
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .unit_kind import classify_unit_kind
from .test_result import EnvHandle, TestFailure, TestRunResult, ran_no_tests

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

    def file_dependency_edges(
        self,
        files: list[str],
        file_sources: dict[str, str],
    ) -> list[FileDependencyEdge]:
        return resolved_go_edges(files, file_sources)

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

    def list_inheritance(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(code, path)
        if result is None or result.syntax_error:
            return []
        return [dep for dep in result.dependencies if dep.relation == "inherits"]

    def unit_kind(self, unit_name: str) -> str:
        return classify_unit_kind(unit_name)

    def is_callable_unit(self, unit_name: str) -> bool:
        return classify_unit_kind(unit_name) == "callable"

    def entry_point_path(self, module: str) -> str:
        slug = self.sanitize_module_identifier(module) if module else "app"
        return f"cmd/{slug}/main.go"

    def find_existing_entry(self, interfaces: dict) -> str | None:
        """Reuse an existing ``cmd/<name>/main.go`` from the skeleton.

        Go places the program entry under ``cmd/<name>/main.go``. The
        canonical slug (``cmd/<repo>/main.go``) rarely matches the name
        the skeleton actually chose (``cmd/todoapp/main.go``), so a plain
        filename match is not enough: this returns the first designed
        ``cmd/<name>/main.go`` three-segment path so the synthetic
        MAIN_ENTRY task extends it instead of creating a second
        ``package main``. Falls back to ``None`` (→ backend default) when
        the skeleton declared no command package.
        """
        if not isinstance(interfaces, dict):
            return None
        for subtree in interfaces.get("subtrees", {}).values():
            if not isinstance(subtree, dict):
                continue
            container = subtree.get("interfaces", subtree.get("files", {}))
            if not isinstance(container, dict):
                continue
            for file_path in container:
                norm = str(file_path).replace("\\", "/")
                parts = norm.split("/")
                if len(parts) == 3 and parts[0] == "cmd" and parts[2] == "main.go":
                    return norm
        return None

    def entry_point_candidates(self) -> list[str]:
        # Go's entry lives under cmd/<name>/main.go; accept any such
        # command package, not only the canonical slug.
        return ["cmd/*/main.go"]

    def prepare_test_env(self, env: EnvHandle) -> None:
        return None

    def entry_run_command(self, repo_root: Path, entry: str) -> list[str] | None:
        if not (repo_root / entry).is_file():
            return None
        pkg_dir = str(Path(entry).parent).replace("\\", "/")
        return ["go", "run", f"./{pkg_dir}", "--help"]

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
        # ``-v`` makes ``go test`` emit per-test ``=== RUN`` / ``--- PASS``
        # lines that :meth:`parse_test_output` counts. Without it the output
        # is only the ``ok <pkg>`` package summary, so passed_count stays 0 —
        # a real run would then look indistinguishable from a zero-test no-op.
        cmd = [go_exe, "test", "-v"]
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
            # ``go test ./...`` exits 0 even when it matched no packages
            # (empty output) — a no-op that must not pass a gate. A real run
            # always emits output, so the empty-output signal catches the
            # no-op. Only trust a POSITIVE parsed count as proof tests ran;
            # a parsed 0 is ambiguous (the ``-json`` stream may not match the
            # text-format regexes), so fall back to the output signal rather
            # than false-failing a real run.
            observed = passed_count + failed_count + skipped_count
            if ran_no_tests(exit_code, raw, observed_tests=observed or None):
                status = "errored"
            else:
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

    def project_task_templates(self, context: ProjectTaskContext) -> ProjectTaskTemplates:
        module_name = context.package_name
        # Reuse the planner-reconciled entry path when provided (avoids a
        # second command package); else fall back to the canonical path.
        command_path = context.entry_point_path or f"cmd/{module_name}/main.go"
        # Directory passed to ``go run ./<dir>`` — derived from the actual
        # command path so docs and verify steps match the real entry.
        command_dir = command_path.rsplit("/", 1)[0] if "/" in command_path else "."
        return ProjectTaskTemplates(
            dependencies=f"""Generate or update Go module dependency files for the repository: {context.repo_name}

**Files to create/update:**
1. `go.mod` - Go module declaration using module path `{module_name}`
2. `go.sum` - Only if external dependencies are introduced

**Instructions:**
1. Prefer the Go standard library. Do not add third-party dependencies unless the implemented code already requires them.
2. If there are no external dependencies, create a minimal `go.mod` with a current Go version.
3. If dependencies are needed, run `go mod tidy` after adding imports.
4. Verify the module with `go test ./...`.

**Important:**
- Do NOT create Python dependency files for a Go project.
- Keep the module compact and local to this repository.
- The fixture expects standard-library-only code unless the implementation proves otherwise.
""",
            main_entry=f"""Create the Go command entry point for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Create a production-quality Go CLI entry point that lets users run the complete product through documented commands.

**Go entry-point rules:**
1. The executable package MUST be `package main`.
2. It MUST define `func main()`.
3. Keep `main.go` thin: parse CLI arguments, handle user-facing errors, and delegate business logic to existing project packages.
4. Do NOT duplicate domain logic in `main.go`.

**Files to create/update:**
1. `{command_path}` - command entry file.
   - Prefer the conventional `cmd/<name>/main.go` layout for CLI applications.
   - If `{command_path}` already exists, extend it in place.
   - Do NOT create a second command package unless the project explicitly requires multiple commands.

**Critical Rules:**
- Every import must reference real packages and symbols from this module.
- Use idiomatic Go error handling with explicit non-zero exits on user-facing failures.
- Keep output plain text unless the requirements explicitly ask otherwise.

**Requirements:**
1. Use `package main` and define `func main()`.
2. Provide `--help` output and documented commands/options that expose all major CLI features.
3. Import and call real packages/symbols from this module.
4. Handle invalid commands, invalid ids, missing arguments, and runtime errors clearly.
5. Verify with `go run ./{command_dir} --help` and `go test ./...`.

**Important:**
- Read `docs/` first and faithfully expose the requested behavior.
- Reuse `{command_path}`; do NOT create Python package entry points or another Go command package for this Go project.
""",
            readme=f"""Update the README.md for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Replace the placeholder README with comprehensive documentation for the actual Go CLI implementation.

**Sections to include:**

## 1. Project Title & Description
- Clear, concise description of what the CLI does
- Key commands and capabilities

## 2. Installation
- Go version prerequisite
- Clone/build instructions
- Module setup using `go mod tidy` when needed

## 3. Usage
- How to run the CLI with `go run ./{command_dir} --help`
- Common command examples with expected plain-text output
- Data file options and local persistence behavior if applicable

## 4. Project Structure
- Brief overview of `cmd/`, `internal/`, and test files
- Key packages and their purposes

## 5. Development
- How to run tests with `go test ./...`
- How to format code with `gofmt`

**Instructions:**
1. Read the `docs/` directory for the original requirements.
2. Explore the actual Go codebase to understand what was implemented.
3. Run `go run ./{command_dir} --help` if the command exists.
4. Reference actual package names, types, and functions.

**Important:**
- Do NOT document Python commands, Python test runners, or Python dependency files for this Go project.
- Base everything on the actual implemented code, not assumptions.
- Keep the tone professional and concise.
""",
        )

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
