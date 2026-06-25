"""Production :class:`LanguageBackend` implementation for TypeScript."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .file_deps import FileDependencyEdge, resolved_ts_js_edges
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .unit_kind import classify_unit_kind
from .test_result import EnvHandle, TestFailure, TestRunResult, ran_no_tests

_TS_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_$-]+$")
_TS_SEGMENT_INVALID = re.compile(r"[^A-Za-z0-9_$-]")
_TS_INTERFACE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)\b"
)
_TS_TYPE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)\b"
)
_TS_DECLARE_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?declare\s+function\s+([A-Za-z_$][\w$]*)\b"
)
_TS_DECLARE_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?declare\s+(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)\b"
)
_TS_ENUM_RE = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?enum\s+([A-Za-z_$][\w$]*)\b"
)
_PLACEHOLDER_RE = re.compile(
    r"(?is)\b(?:TODO|PLACEHOLDER|NOT IMPLEMENTED|throw\s+new\s+Error\s*\()"
)
_TS_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
_TS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


class TypeScriptBackend:
    """:class:`LanguageBackend` for TypeScript source."""

    name = "typescript"
    display_name = "TypeScript"
    file_extension = ".ts"
    markdown_fence = "typescript"

    def is_source_file(self, path: str) -> bool:
        return (path.endswith(".ts") or path.endswith(".tsx")) and not path.endswith(".d.ts")

    def is_test_file(self, path: str) -> bool:
        normalised = path.replace("\\", "/")
        basename = normalised.rsplit("/", 1)[-1]
        return (
            "/tests/" in f"/{normalised}"
            or basename.endswith(".test.ts")
            or basename.endswith(".spec.ts")
            or basename.endswith(".test.tsx")
            or basename.endswith(".spec.tsx")
        )

    def file_dependency_edges(
        self,
        files: list[str],
        file_sources: dict[str, str],
    ) -> list[FileDependencyEdge]:
        return resolved_ts_js_edges(files, file_sources, language=self.name)

    def package_marker_filename(self) -> str | None:
        return None

    def package_marker_content(self, pkg_path: str) -> str | None:
        return None

    def is_valid_module_identifier(self, segment: str) -> bool:
        return bool(segment and _TS_SEGMENT_RE.match(segment))

    def sanitize_module_identifier(self, segment: str) -> str:
        if not segment:
            return "module"
        cleaned = _TS_SEGMENT_INVALID.sub("-", segment.strip())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        return cleaned or "module"

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        ok, _ = self.syntax_check(code, path)
        return ok and bool(_PLACEHOLDER_RE.search(code))

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        return self._parser().validate_syntax(
            self._parse_path(path),
            self._parse_source(code),
        )

    def list_code_units(self, code: str, path: str = "<string>") -> list[Any]:
        parse_path = self._parse_path(path)
        result = self._parse(self._parse_source(code), parse_path)
        units = [] if result is None or result.syntax_error else [
            unit for unit in result.units
            if unit.unit_type in {"class", "function", "method"}
        ]
        units.extend(self._declaration_units(code, parse_path))
        return units

    def format_signature(self, unit: Any) -> str:
        if unit is None:
            return ""
        code = (getattr(unit, "code", "") or "").strip()
        if not code:
            return getattr(unit, "name", "") or ""
        first = code.split("{", 1)[0].split(";", 1)[0].strip()
        return " ".join(first.split()) or (getattr(unit, "name", "") or "")

    def list_imports(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(self._parse_source(code), path)
        if result is None or result.syntax_error:
            return []
        return [dep for dep in result.dependencies if dep.relation == "imports"]

    def list_inheritance(self, code: str, path: str = "<string>") -> list[Any]:
        result = self._parse(self._parse_source(code), path)
        if result is None or result.syntax_error:
            return []
        return [dep for dep in result.dependencies if dep.relation == "inherits"]

    def unit_kind(self, unit_name: str) -> str:
        return classify_unit_kind(unit_name)

    def is_callable_unit(self, unit_name: str) -> bool:
        return classify_unit_kind(unit_name) == "callable"

    def entry_point_path(self, module: str) -> str:
        return "src/index.ts"

    def find_existing_entry(self, interfaces: dict) -> str | None:
        from .backend import default_find_existing_entry

        return default_find_existing_entry(self, interfaces)

    def entry_point_candidates(self) -> list[str]:
        return [self.entry_point_path("")]

    def prepare_test_env(self, env) -> None:
        return None

    def entry_run_command(self, repo_root: Path, entry: str) -> list[str] | None:
        # Prefer the package-defined start script (handles tsc build/dist).
        if (repo_root / "package.json").is_file():
            return ["npm", "start", "--", "--help"]
        return None

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        npm = shutil.which("npm")
        node = shutil.which("node")
        if not npm and not node:
            return None
        root = repo_root.resolve()
        return EnvHandle(
            project_root=root,
            runtime_executable=npm or node,
            extra={"package_json": str(root / "package.json")},
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        env = self.detect_env(repo_root)
        if env is None:
            raise ToolchainUnavailable("Node.js/npm toolchain is not available on PATH")
        package_json = env.project_root / "package.json"
        if not package_json.exists():
            name = self._default_package_name(env.project_root)
            package_json.write_text(
                json.dumps({
                    "name": name,
                    "version": "0.1.0",
                    "type": "module",
                    "scripts": {"test": "node --test"},
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        return env

    def test_command(self, env: EnvHandle, selectors: list[str] | None = None) -> list[str]:
        executable = env.runtime_executable or "npm"
        if Path(executable).name == "node":
            return [executable, "--test", *(selectors or [])]
        return [executable, "test", *(selectors or [])]

    def install_deps_command(self, env: EnvHandle, deps: list[str]) -> list[str] | None:
        if not deps:
            return None
        executable = env.runtime_executable or "npm"
        if Path(executable).name == "node":
            return None
        return [executable, "install", *deps]

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        # node:test prints a TAP summary ("# tests N", "# pass N",
        # "# fail N"). Use the test count to tell a real run from a no-op
        # that exits 0 without running anything (which must not pass a gate).
        tests_match = re.search(r"(?m)^#?\s*tests\s+(\d+)\b", raw)
        observed = int(tests_match.group(1)) if tests_match else None
        if ran_no_tests(exit_code, raw, observed_tests=observed):
            status = "errored"
        else:
            status = "passed" if exit_code == 0 else "failed"
        failures = [] if status != "failed" else [TestFailure(
            test_id="npm test",
            short_message="npm test failed",
            long_message=raw,
        )]
        pass_match = re.search(r"(?m)^#?\s*pass\s+(\d+)\b", raw)
        fail_match = re.search(r"(?m)^#?\s*fail\s+(\d+)\b", raw)
        return TestRunResult(
            status=status,
            exit_code=exit_code,
            passed_count=int(pass_match.group(1)) if pass_match else 0,
            failed_count=(
                int(fail_match.group(1)) if fail_match
                else (1 if status == "failed" else 0)
            ),
            error_count=0,
            skipped_count=0,
            duration_sec=0.0,
            failures=failures,
            raw_output=raw,
            extra={"tool": "npm test"},
        )

    _PROMPT_HINTS_SINGLETON: PromptHints | None = None

    def prompt_hints(self) -> PromptHints:
        cached = TypeScriptBackend._PROMPT_HINTS_SINGLETON
        if cached is not None:
            return cached
        hints = PromptHints(
            display_name=self.display_name,
            markdown_fence=self.markdown_fence,
            file_extension=self.file_extension,
            module_naming_rule=(
                "Use kebab-case or short lowercase directory names; source "
                "files live under src/ and tests under tests/ or *.test.ts."
            ),
            package_layout_example=(
                "package.json\n"
                "tsconfig.json\n"
                "src/\n"
                "  index.ts\n"
                "  cli.ts\n"
                "tests/\n"
                "  cli.test.ts\n"
            ),
            entrypoint_example="src/index.ts",
            test_framework_name="npm test",
            style_directive=(
                "Write idiomatic TypeScript: explicit exported types, narrow "
                "interfaces, async-aware APIs, and Node.js standard modules "
                "for local CLI/file operations."
            ),
        )
        TypeScriptBackend._PROMPT_HINTS_SINGLETON = hints
        return hints

    def project_task_templates(self, context: ProjectTaskContext) -> ProjectTaskTemplates:
        # Reuse the planner-reconciled entry path when provided (avoids a
        # second entry file); else fall back to the canonical path.
        entry = context.entry_point_path or "src/index.ts"
        return ProjectTaskTemplates(
            dependencies=f"""Generate or update Node.js/TypeScript dependency files for the repository: {context.repo_name}

**Files to create/update:**
1. `package.json` - Package metadata, scripts, and dependencies using package name `{context.package_name}`
2. `tsconfig.json` - TypeScript compiler configuration for Node.js
3. `package-lock.json` - Only if dependency installation creates it

**Instructions:**
1. Prefer Node.js standard APIs for local file and CLI behavior.
2. Add TypeScript tooling and a minimal test runner only when needed by the implemented code.
3. Provide scripts for `npm start`, `npm test`, and type checking when appropriate.
4. Run `npm test` after updating dependencies.

**Important:**
- Do NOT create Python dependency files for a TypeScript project.
- Keep dependencies minimal and aligned with actual imports.
""",
            main_entry=f"""Create the TypeScript command entry point for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Create a production-quality Node.js CLI entry point that lets users run the complete product through documented commands.

**Files to create:**
1. `{entry}` - CLI entry point exported or referenced by package scripts.
2. `src/cli.ts` (optional) - Command parsing and dispatch separated from domain logic.

**Critical Rules:**
- Do NOT re-implement business logic in the entry file. Import and delegate to implemented modules.
- This is the ONLY program entry point in the repository. If `{entry}` already exists, extend it in place — do NOT create a second entry file.
- Every import must reference real files and exported symbols.
- Use explicit error handling and non-zero process exits for user-facing failures.
- Keep output plain text unless the requirements explicitly ask otherwise.

**Requirements:**
1. Expose all major CLI commands and options described in `docs/`.
2. Wire `package.json` scripts so users can run the CLI with `npm start -- --help`.
3. Delegate storage and task lifecycle behavior to implemented modules.
4. Handle invalid commands, invalid ids, missing arguments, and runtime errors clearly.
5. Verify with `npm start -- --help` and `npm test`.

**Important:**
- Read `docs/` first and faithfully expose the requested behavior.
- Do NOT create Python package entry points for this TypeScript project.
""",
            readme=f"""Update the README.md for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Replace the placeholder README with comprehensive documentation for the actual TypeScript CLI implementation.

**Sections to include:**

## 1. Project Title & Description
- Clear, concise description of what the CLI does
- Key commands and capabilities

## 2. Installation
- Node.js/npm prerequisite
- Clone/install instructions using `npm install`
- TypeScript build or runtime notes if applicable

## 3. Usage
- How to run the CLI with `npm start -- --help`
- Common command examples with expected plain-text output
- Data file options and local persistence behavior if applicable

## 4. Project Structure
- Brief overview of `src/`, `tests/`, and configuration files
- Key modules and their purposes

## 5. Development
- How to run tests with `npm test`
- How to run type checks and build commands if scripts exist

**Instructions:**
1. Read the `docs/` directory for the original requirements.
2. Explore the actual TypeScript codebase to understand what was implemented.
3. Run `npm start -- --help` if package scripts exist.
4. Reference actual exported types, functions, and modules.

**Important:**
- Do NOT document Python commands, Python test runners, or Python dependency files for this TypeScript project.
- Base everything on the actual implemented code, not assumptions.
- Keep the tone professional and concise.
""",
        )

    @staticmethod
    def _parser() -> Any:
        from lang_parser import get_parser  # type: ignore

        return get_parser("typescript")

    @staticmethod
    def _parse_path(path: str) -> str:
        if path == "<string>" or not (path.endswith(".ts") or path.endswith(".tsx")):
            return "src/index.ts"
        return path

    @staticmethod
    def _parse_source(code: str) -> str:
        return _TS_LINE_COMMENT_RE.sub("", _TS_BLOCK_COMMENT_RE.sub("", code))

    def _parse(self, code: str, path: str):
        try:
            return self._parser().parse_file(self._parse_path(path), code)
        except Exception:
            return None

    def _declaration_units(self, code: str, path: str) -> list[Any]:
        from lang_parser import LPCodeUnit  # type: ignore

        units: list[Any] = []
        lines = code.splitlines()
        for index, line in enumerate(lines):
            match = (
                _TS_INTERFACE_RE.match(line)
                or _TS_TYPE_RE.match(line)
                or _TS_DECLARE_FUNCTION_RE.match(line)
                or _TS_DECLARE_CLASS_RE.match(line)
                or _TS_ENUM_RE.match(line)
            )
            if match is None:
                continue
            unit_type = self._declaration_unit_type(line)
            end = self._declaration_end(lines, index)
            units.append(LPCodeUnit(
                name=match.group(1),
                unit_type=unit_type,
                file_path=path,
                parent=None,
                line_start=index + 1,
                line_end=end + 1,
                code="\n".join(lines[index:end + 1]),
                language=self.name,
                extra={"kind": unit_type},
            ))
        return units

    @staticmethod
    def _declaration_unit_type(line: str) -> str:
        if "interface" in line:
            return "interface"
        if "type" in line:
            return "type"
        if "function" in line:
            return "function"
        if "class" in line:
            return "class"
        if "enum" in line:
            return "enum"
        return "declaration"

    def _declaration_end(self, lines: list[str], start: int) -> int:
        depth = 0
        for index in range(start, len(lines)):
            depth += lines[index].count("{") - lines[index].count("}")
            if depth <= 0 and (";" in lines[index] or "}" in lines[index]):
                return index
        return start

    def _default_package_name(self, repo_root: Path) -> str:
        raw = repo_root.name.lower().replace("_", "-").replace(" ", "-")
        return self.sanitize_module_identifier(raw)
