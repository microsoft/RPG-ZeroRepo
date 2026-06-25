"""Production :class:`LanguageBackend` implementation for Rust."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .file_deps import FileDependencyEdge, resolved_rust_edges
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .unit_kind import classify_unit_kind
from .test_result import EnvHandle, TestFailure, TestRunResult, ran_no_tests

_RUST_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUST_IDENT_INVALID = re.compile(r"[^A-Za-z0-9_]")
_PLACEHOLDER_RE = re.compile(
    r"(?is)\b(?:todo!|unimplemented!|panic!\s*\(|TODO|PLACEHOLDER|NOT IMPLEMENTED)"
)
_RUST_KEYWORDS = frozenset({
    "as", "async", "await", "break", "const", "continue", "crate",
    "dyn", "else", "enum", "extern", "false", "fn", "for", "if",
    "impl", "in", "let", "loop", "match", "mod", "move", "mut",
    "pub", "ref", "return", "self", "Self", "static", "struct",
    "super", "trait", "true", "type", "unsafe", "use", "where", "while",
})


class RustBackend:
    """:class:`LanguageBackend` for Rust source."""

    name = "rust"
    display_name = "Rust"
    file_extension = ".rs"
    markdown_fence = "rust"

    def is_source_file(self, path: str) -> bool:
        return path.endswith(".rs")

    def is_test_file(self, path: str) -> bool:
        normalised = path.replace("\\", "/")
        basename = normalised.rsplit("/", 1)[-1]
        return "/tests/" in f"/{normalised}" or basename.endswith("_test.rs")

    def file_dependency_edges(
        self,
        files: list[str],
        file_sources: dict[str, str],
    ) -> list[FileDependencyEdge]:
        return resolved_rust_edges(files, file_sources)

    def package_marker_filename(self) -> str | None:
        return None

    def package_marker_content(self, pkg_path: str) -> str | None:
        return None

    def is_valid_module_identifier(self, segment: str) -> bool:
        if not segment or segment in _RUST_KEYWORDS:
            return False
        return bool(_RUST_IDENT_RE.match(segment))

    def sanitize_module_identifier(self, segment: str) -> str:
        if not segment:
            return "_"
        cleaned = _RUST_IDENT_INVALID.sub("_", segment)
        if cleaned[:1].isdigit():
            cleaned = f"_{cleaned}"
        if cleaned in _RUST_KEYWORDS:
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
            if unit.unit_type in {"struct", "enum", "trait", "function", "method"}
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
        return "src/main.rs"

    def find_existing_entry(self, interfaces: dict) -> str | None:
        from .backend import default_find_existing_entry

        return default_find_existing_entry(self, interfaces)

    def entry_point_candidates(self) -> list[str]:
        return [self.entry_point_path("")]

    def prepare_test_env(self, env) -> None:
        return None

    def entry_run_command(self, repo_root: Path, entry: str) -> list[str] | None:
        if not (repo_root / "Cargo.toml").is_file():
            return None
        return ["cargo", "run", "--", "--help"]

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        cargo = shutil.which("cargo")
        if not cargo:
            return None
        root = repo_root.resolve()
        manifest = root / "Cargo.toml"
        return EnvHandle(
            project_root=root,
            runtime_executable=cargo,
            extra={"manifest": str(manifest) if manifest.exists() else None},
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        env = self.detect_env(repo_root)
        if env is None:
            raise ToolchainUnavailable("Rust toolchain is not available on PATH")
        manifest = env.project_root / "Cargo.toml"
        if not manifest.exists():
            name = self._default_package_name(env.project_root)
            manifest.write_text(
                f"[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            return EnvHandle(
                project_root=env.project_root,
                runtime_executable=env.runtime_executable,
                extra={"manifest": str(manifest)},
            )
        return env

    def test_command(self, env: EnvHandle, selectors: list[str] | None = None) -> list[str]:
        cmd = [env.runtime_executable or "cargo", "test"]
        if selectors:
            cmd.extend(selectors)
        return cmd

    def install_deps_command(self, env: EnvHandle, deps: list[str]) -> list[str] | None:
        if not deps:
            return None
        return [env.runtime_executable or "cargo", "add", *deps]

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        # cargo prints "test result: ok. N passed; M failed; K ignored" per
        # test binary. Summing across binaries tells a real run from a no-op
        # exit-0 (which must not pass a gate).
        totals = re.findall(
            r"test result:\s*\w+\.\s*(\d+)\s+passed;\s*(\d+)\s+failed"
            r"(?:;\s*(\d+)\s+ignored)?",
            raw,
        )
        if totals:
            passed = sum(int(p) for p, _f, _i in totals)
            failed = sum(int(f) for _p, f, _i in totals)
            ignored = sum(int(i or 0) for _p, _f, i in totals)
            observed: int | None = passed + failed + ignored
        else:
            passed = failed = ignored = 0
            observed = None
        if ran_no_tests(exit_code, raw, observed_tests=observed):
            status = "errored"
        else:
            status = "passed" if exit_code == 0 else "failed"
        failures = [] if status != "failed" else [TestFailure(
            test_id="cargo test",
            short_message="cargo test failed",
            long_message=raw,
        )]
        return TestRunResult(
            status=status,
            exit_code=exit_code,
            passed_count=passed,
            failed_count=failed if failed else (1 if status == "failed" else 0),
            error_count=0,
            skipped_count=ignored,
            duration_sec=0.0,
            failures=failures,
            raw_output=raw,
            extra={"tool": "cargo test"},
        )

    _PROMPT_HINTS_SINGLETON: PromptHints | None = None

    def prompt_hints(self) -> PromptHints:
        cached = RustBackend._PROMPT_HINTS_SINGLETON
        if cached is not None:
            return cached
        hints = PromptHints(
            display_name=self.display_name,
            markdown_fence=self.markdown_fence,
            file_extension=self.file_extension,
            module_naming_rule=(
                "Use snake_case Rust module file names; Cargo entrypoints live "
                "in src/main.rs or src/lib.rs."
            ),
            package_layout_example=(
                "Cargo.toml\n"
                "src/\n"
                "  main.rs\n"
                "  lib.rs\n"
                "  store.rs\n"
                "tests/\n"
                "  integration_test.rs\n"
            ),
            entrypoint_example="src/main.rs",
            test_framework_name="cargo test",
            style_directive=(
                "Write idiomatic Rust: explicit Result-based error handling, "
                "small modules, ownership-conscious APIs, and structs/enums "
                "for domain data."
            ),
        )
        RustBackend._PROMPT_HINTS_SINGLETON = hints
        return hints

    def project_task_templates(self, context: ProjectTaskContext) -> ProjectTaskTemplates:
        # Reuse the planner-reconciled entry path when provided (avoids a
        # second ``main`` file); else fall back to the canonical path.
        entry = context.entry_point_path or "src/main.rs"
        return ProjectTaskTemplates(
            dependencies=f"""Generate or update Rust Cargo dependency files for the repository: {context.repo_name}

**Files to create/update:**
1. `Cargo.toml` - Cargo package declaration using package name `{context.package_name}`
2. `Cargo.lock` - Only if dependency resolution creates it

**Instructions:**
1. Prefer the Rust standard library for CLI parsing and file handling unless implemented code already requires a crate.
2. Include `serde` and `serde_json` when JSON serialization is implemented.
3. Use edition `2021` unless the implemented code requires another stable edition.
4. Run `cargo test` after updating dependencies.

**Important:**
- Do NOT create Python dependency files for a Rust project.
- Keep dependency choices minimal and justified by actual imports.
""",
            main_entry=f"""Create the Rust command entry point for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Create a production-quality Cargo CLI entry point that lets users run the complete product through documented commands.

**Files to create:**
1. `{entry}` - Binary entry point for the CLI.
2. `src/lib.rs` (optional) - Library module that exposes reusable task/store logic.

**Critical Rules:**
- Do NOT re-implement business logic in the entry file. Delegate to modules already defined in the crate.
- This is the ONLY binary entry point in the repository. If `{entry}` already exists, extend it in place — do NOT create a second entry file.
- Every `use` path must reference real modules and symbols.
- Use idiomatic `Result`-based error handling and explicit non-zero exits for user-facing failures.
- Keep output plain text unless the requirements explicitly ask otherwise.

**Requirements:**
1. Provide a `main()` function in `src/main.rs`.
2. Expose all major CLI commands and options described in `docs/`.
3. Delegate storage and task lifecycle behavior to implemented modules.
4. Handle invalid commands, invalid ids, missing arguments, and runtime errors clearly.
5. Verify with `cargo run -- --help` and `cargo test`.

**Important:**
- Read `docs/` first and faithfully expose the requested behavior.
- Do NOT create Python package entry points for this Rust project.
""",
            readme=f"""Update the README.md for the repository: {context.repo_name}
Repository purpose: {context.repo_info}

**Goal:** Replace the placeholder README with comprehensive documentation for the actual Rust CLI implementation.

**Sections to include:**

## 1. Project Title & Description
- Clear, concise description of what the CLI does
- Key commands and capabilities

## 2. Installation
- Rust/Cargo prerequisite
- Clone/build instructions
- Dependency setup with `cargo build`

## 3. Usage
- How to run the CLI with `cargo run -- --help`
- Common command examples with expected plain-text output
- Data file options and local persistence behavior if applicable

## 4. Project Structure
- Brief overview of `src/`, modules, and tests
- Key modules and their purposes

## 5. Development
- How to run tests with `cargo test`
- How to format code with `cargo fmt`

**Instructions:**
1. Read the `docs/` directory for the original requirements.
2. Explore the actual Rust codebase to understand what was implemented.
3. Run `cargo run -- --help` if the binary exists.
4. Reference actual module names, structs, traits, enums, and functions.

**Important:**
- Do NOT document Python commands, Python test runners, or Python dependency files for this Rust project.
- Base everything on the actual implemented code, not assumptions.
- Keep the tone professional and concise.
""",
        )

    @staticmethod
    def _parser() -> Any:
        from lang_parser import get_parser  # type: ignore

        return get_parser("rust")

    @staticmethod
    def _parse_path(path: str) -> str:
        if path == "<string>" or not path.endswith(".rs"):
            return "src/lib.rs"
        return path

    def _parse(self, code: str, path: str):
        try:
            return self._parser().parse_file(self._parse_path(path), code)
        except Exception:
            return None

    def _default_package_name(self, repo_root: Path) -> str:
        raw = repo_root.name.lower().replace("-", "_")
        return self.sanitize_module_identifier(raw)
