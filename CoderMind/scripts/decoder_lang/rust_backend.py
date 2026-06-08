"""Production :class:`LanguageBackend` implementation for Rust."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .prompt_hints import PromptHints
from .test_result import EnvHandle, TestFailure, TestRunResult

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
        status = "passed" if exit_code == 0 else "failed"
        failures = [] if exit_code == 0 else [TestFailure(
            test_id="cargo test",
            short_message="cargo test failed",
            long_message=raw,
        )]
        return TestRunResult(
            status=status,
            exit_code=exit_code,
            passed_count=0,
            failed_count=0 if exit_code == 0 else 1,
            error_count=0,
            skipped_count=0,
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
