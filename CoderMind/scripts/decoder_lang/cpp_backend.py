"""Production :class:`LanguageBackend` implementation for C++."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from common.utils import path_has_skip_dir

from .backend import ToolchainUnavailable
from .file_deps import FileDependencyEdge, resolved_c_family_edges
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .unit_kind import classify_unit_kind
from .test_result import EnvHandle, TestFailure, TestRunResult, ran_no_tests

_CPP_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CPP_IDENT_INVALID = re.compile(r"[^A-Za-z0-9_]")
_PLACEHOLDER_RE = re.compile(
    r"(?is)\b(?:TODO|PLACEHOLDER|NOT IMPLEMENTED|throw\s+std::logic_error|abort\s*\()"
)
_COMPILE_ONLY_COMMAND_RE = re.compile(
    r"(?m)^\s*\S*(?:c\+\+|g\+\+|clang\+\+)(?:-[\w.]+)?(?=\s).*\s-c\s"
)
_TEST_EXECUTION_RE = re.compile(
    r"(?im)(^|\s)(?:PASS|FAIL)(?:\s|:)|^\s*ok\b|^\s*1\.\.|"
    r"test result:|\btests? passed\b|^\s*running\s+\d+\s+tests?"
)
_CPP_SOURCE_EXTENSIONS = (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h")
_CPP_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "asm", "auto", "bool", "break",
    "case", "catch", "char", "class", "const", "constexpr", "continue",
    "decltype", "default", "delete", "do", "double", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend",
    "goto", "if", "inline", "int", "long", "namespace", "new", "noexcept",
    "operator", "private", "protected", "public", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "template", "this",
    "throw", "true", "try", "typedef", "typename", "union", "unsigned",
    "using", "virtual", "void", "volatile", "while",
})


class CppBackend:
    """:class:`LanguageBackend` for C++ source."""

    name = "cpp"
    display_name = "C++"
    file_extension = ".cpp"
    markdown_fence = "cpp"

    def is_source_file(self, path: str) -> bool:
        return path.endswith(_CPP_SOURCE_EXTENSIONS)

    def is_test_file(self, path: str) -> bool:
        normalised = path.replace("\\", "/")
        basename = normalised.rsplit("/", 1)[-1]
        return (
            f"/{normalised}".startswith("/tests/")
            or "/tests/" in f"/{normalised}"
            or basename.endswith(("_test.cpp", "_test.cc", "_test.cxx"))
            or basename.startswith("test_") and basename.endswith((".cpp", ".cc", ".cxx"))
        )

    def file_dependency_edges(
        self,
        files: list[str],
        file_sources: dict[str, str],
    ) -> list[FileDependencyEdge]:
        return resolved_c_family_edges(files, file_sources, language=self.name)

    def package_marker_filename(self) -> str | None:
        return None

    def package_marker_content(self, pkg_path: str) -> str | None:
        return None

    def is_valid_module_identifier(self, segment: str) -> bool:
        if not segment or segment in _CPP_KEYWORDS:
            return False
        return bool(_CPP_IDENT_RE.match(segment))

    def sanitize_module_identifier(self, segment: str) -> str:
        if not segment:
            return "_"
        cleaned = _CPP_IDENT_INVALID.sub("_", segment)
        if cleaned[:1].isdigit():
            cleaned = f"_{cleaned}"
        if cleaned in _CPP_KEYWORDS:
            cleaned = f"{cleaned}_"
        return cleaned

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        ok, _ = self.syntax_check(code, path)
        return ok and bool(_PLACEHOLDER_RE.search(code))

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        return self._parser().validate_syntax(self._parse_path(path), code)

    def list_code_units(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(code, path)
        if result is None or result.syntax_error:
            return []
        return [
            unit for unit in result.units
            if unit.unit_type in {"class", "struct", "enum", "function", "method"}
        ]

    def format_signature(self, unit: Any) -> str:
        if unit is None:
            return ""
        code = (getattr(unit, "code", "") or "").strip()
        if not code:
            return getattr(unit, "name", "") or ""
        first = code.split("{", 1)[0].split(";", 1)[0].strip()
        return " ".join(first.split()) or (getattr(unit, "name", "") or "")

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
        return "src/main.cpp"

    def find_existing_entry(self, interfaces: dict) -> str | None:
        from .backend import default_find_existing_entry

        return default_find_existing_entry(self, interfaces)

    def entry_point_candidates(self) -> list[str]:
        return [self.entry_point_path("")]

    def prepare_test_env(self, env: EnvHandle) -> None:
        from .backend import cmake_reconfigure

        cmake_reconfigure(env)

    def entry_run_command(self, repo_root: Path, entry: str) -> list[str] | None:
        # Compiled CLI: binary name is project-specific; the smoke layer
        # locates the built artifact rather than guessing here.
        return None

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        cxx = self._find_compiler()
        make = shutil.which("make")
        cmake = shutil.which("cmake")
        ctest = shutil.which("ctest")
        if not cxx and not make and not cmake:
            return None
        root = repo_root.resolve()
        return EnvHandle(
            project_root=root,
            runtime_executable=cmake or make or cxx,
            extra={
                "cxx": cxx,
                "make": make,
                "cmake": cmake,
                "ctest": ctest,
                "cmake_lists": str(root / "CMakeLists.txt") if (root / "CMakeLists.txt").exists() else None,
                "makefile": str(root / "Makefile") if (root / "Makefile").exists() else None,
            },
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        env = self.detect_env(repo_root)
        if env is None or not env.extra.get("cxx"):
            raise ToolchainUnavailable("C++ compiler is not available on PATH")
        return env

    def test_command(self, env: EnvHandle, selectors: list[str] | None = None) -> list[str]:
        ctest = env.extra.get("ctest") if env.extra else None
        if ctest and (env.project_root / "CMakeLists.txt").exists():
            return [ctest, "--test-dir", str(env.project_root / "build"), "--output-on-failure"]
        make = env.extra.get("make") if env.extra else None
        if make and (env.project_root / "Makefile").exists():
            return [make, "test"]
        cxx = env.extra.get("cxx") if env.extra else None
        if not cxx:
            raise ToolchainUnavailable("C++ compiler is not available on PATH")
        sources = sorted(
            str(path)
            for ext in ("*.cpp", "*.cc", "*.cxx")
            for path in env.project_root.rglob(ext)
            if not path_has_skip_dir(path.relative_to(env.project_root).as_posix())
        )
        return [
            cxx,
            "-std=c++17",
            "-I",
            str(env.project_root),
            "-Wall",
            "-Wextra",
            "-fsyntax-only",
            *sources,
        ]

    def install_deps_command(self, env: EnvHandle, deps: list[str]) -> list[str] | None:
        return None

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        # The C++ test command is ctest/make when a harness exists, else a
        # bare ``-fsyntax-only`` compile check that legitimately emits no
        # output. So empty output is NOT a no-op here (a clean compile is a
        # real signal); only ctest's explicit "no tests" / "out of 0" is.
        out_of = re.search(r"out of (\d+)", raw)
        observed = int(out_of.group(1)) if out_of else None
        if ran_no_tests(
            exit_code, raw, observed_tests=observed,
            no_tests_markers=("No tests were found", "Nothing to be done for 'test'"),
            empty_output_is_no_op=False,
        ) or self._looks_like_compile_only_make_test(raw):
            status = "errored"
        else:
            status = "passed" if exit_code == 0 else "failed"
        failures = [] if status != "failed" else [TestFailure(
            test_id="c++ test",
            short_message="C++ test command failed",
            long_message=raw,
        )]
        fail_match = re.search(r"(\d+)\s+tests?\s+failed", raw)
        return TestRunResult(
            status=status,
            exit_code=exit_code,
            passed_count=0,
            failed_count=(
                int(fail_match.group(1)) if fail_match
                else (1 if status == "failed" else 0)
            ),
            error_count=0,
            skipped_count=0,
            duration_sec=0.0,
            failures=failures,
            raw_output=raw,
            extra={"tool": "ctest, make test, or compiler syntax check"},
        )

    @staticmethod
    def _looks_like_compile_only_make_test(raw: str) -> bool:
        """Return True when make output compiled tests but ran none."""
        if not _COMPILE_ONLY_COMMAND_RE.search(raw):
            return False
        return not _TEST_EXECUTION_RE.search(raw)

    _PROMPT_HINTS_SINGLETON: PromptHints | None = None

    def prompt_hints(self) -> PromptHints:
        cached = CppBackend._PROMPT_HINTS_SINGLETON
        if cached is not None:
            return cached
        hints = PromptHints(
            display_name=self.display_name,
            markdown_fence=self.markdown_fence,
            file_extension=self.file_extension,
            module_naming_rule=(
                "Use lowercase snake_case or short descriptive C++ file names; "
                "public declarations live in headers and implementations in matching source files."
            ),
            package_layout_example=(
                "CMakeLists.txt\n"
                "src/\n"
                "  main.cpp\n"
                "  task.cpp\n"
                "  task.hpp\n"
                "tests/\n"
                "  task_test.cpp\n"
            ),
            entrypoint_example="src/main.cpp",
            test_framework_name="ctest or make test",
            style_directive=(
                "Write idiomatic C++17: RAII for ownership, const-correct APIs, "
                "standard library containers, and clear header/source separation."
            ),
        )
        CppBackend._PROMPT_HINTS_SINGLETON = hints
        return hints

    def project_task_templates(self, context: ProjectTaskContext) -> ProjectTaskTemplates:
        # Reuse the planner-reconciled entry path when provided (avoids a
        # second ``main`` file); else fall back to the canonical path.
        entry = context.entry_point_path or "src/main.cpp"
        return ProjectTaskTemplates(
            dependencies=f"""Generate or update C++ build metadata for the repository: {context.repo_name}

**Files to create/update:**
1. `CMakeLists.txt` - C++17 project, executable, and test target definitions.

**Instructions:**
1. Prefer the C++ standard library.
2. Use C++17 unless implemented code requires a newer standard.
3. Provide build and test instructions compatible with CMake and ctest.
4. The test target must build and execute real test binaries. For Makefile
    fallback projects, `make test` must not only compile `.o` files or print
    "Nothing to be done".
5. Keep generated binaries and build directories out of source control.

**Important:**
- Do NOT create Python dependency files for a C++ project.
- Do NOT introduce third-party dependencies unless the implemented code requires them.
""",
            main_entry=f"""Create the C++ command entry point for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Create a production-quality C++ CLI entry point that exposes the documented product behavior.

**Files to create:**
1. `{entry}` - CLI entry point.
2. Headers only when needed to call implemented modules.

**Critical Rules:**
- Do NOT re-implement business logic in the entry file; delegate to implemented modules.
- Include real project headers and call real symbols.
- Validate arguments and return non-zero for user-facing failures.
- Keep output plain text unless requirements say otherwise.
- This is the ONLY program entry point in the repository. If `{entry}` already exists, extend it in place — do NOT create a second entry file.

**Requirements:**
1. Provide `int main(int argc, char **argv)`.
2. Implement `--help` and documented commands/options.
3. Delegate storage and task lifecycle behavior to existing C++ modules.
4. Verify with a CMake build and `ctest`, or with `make test` when the project uses a Makefile. The verification command must execute tests and return non-zero when any test fails.

**Important:**
- Read `docs/` first and faithfully expose the requested behavior.
- Do NOT create Python package entry points for this C++ project.
""",
            readme=f"""Update the README.md for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Replace the placeholder README with comprehensive documentation for the actual C++ CLI implementation.

**Sections to include:**

## 1. Project Title & Description
- Clear, concise description of what the CLI does
- Key commands and capabilities

## 2. Build
- C++ compiler and CMake prerequisites
- Configure/build commands

## 3. Usage
- How to run the compiled binary and `--help`
- Common command examples with expected plain-text output

## 4. Project Structure
- Brief overview of `src/`, headers, tests, and CMake targets

## 5. Development
- How to run tests with `ctest` or `make test`
- How to clean build artefacts

**Important:**
- Do NOT document Python commands, Python test runners, or Python dependency files for this C++ project.
- Base everything on the actual implemented code, not assumptions.
""",
        )

    @staticmethod
    def _parser() -> Any:
        from lang_parser import get_parser  # type: ignore

        return get_parser("cpp")

    @staticmethod
    def _parse_path(path: str) -> str:
        if path == "<string>" or not path.endswith(_CPP_SOURCE_EXTENSIONS):
            return "src/main.cpp"
        if path.endswith(".h"):
            return f"{path[:-2]}.hpp"
        return path

    def _parse(self, code: str, path: str):
        try:
            return self._parser().parse_file(self._parse_path(path), code)
        except Exception:
            return None

    @staticmethod
    def _find_compiler() -> str | None:
        return shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")


__all__ = ["CppBackend"]