"""Production :class:`LanguageBackend` implementation for TypeScript."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .prompt_hints import PromptHints
from .test_result import EnvHandle, TestFailure, TestRunResult

_TS_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_$-]+$")
_TS_SEGMENT_INVALID = re.compile(r"[^A-Za-z0-9_$-]")
_TS_INTERFACE_RE = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)\b")
_TS_TYPE_RE = re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\b")
_PLACEHOLDER_RE = re.compile(
    r"(?is)\b(?:TODO|PLACEHOLDER|NOT IMPLEMENTED|throw\s+new\s+Error\s*\()"
)


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
        return self._parser().validate_syntax(self._parse_path(path), code)

    def list_code_units(self, code: str, path: str = "<string>") -> list[Any]:
        parse_path = self._parse_path(path)
        result = self._parse(code, parse_path)
        units = [] if result is None or result.syntax_error else [
            unit for unit in result.units
            if unit.unit_type in {"class", "function", "method"}
        ]
        units.extend(self._type_units(code, parse_path))
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
        result = self._parse(code, path)
        if result is None or result.syntax_error:
            return []
        return [dep for dep in result.dependencies if dep.relation == "imports"]

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
        status = "passed" if exit_code == 0 else "failed"
        failures = [] if exit_code == 0 else [TestFailure(
            test_id="npm test",
            short_message="npm test failed",
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

    @staticmethod
    def _parser() -> Any:
        from lang_parser import get_parser  # type: ignore

        return get_parser("typescript")

    @staticmethod
    def _parse_path(path: str) -> str:
        if path == "<string>" or not (path.endswith(".ts") or path.endswith(".tsx")):
            return "src/index.ts"
        return path

    def _parse(self, code: str, path: str):
        try:
            return self._parser().parse_file(self._parse_path(path), code)
        except Exception:
            return None

    def _type_units(self, code: str, path: str) -> list[Any]:
        from lang_parser import LPCodeUnit  # type: ignore

        units: list[Any] = []
        lines = code.splitlines()
        for index, line in enumerate(lines):
            match = _TS_INTERFACE_RE.match(line) or _TS_TYPE_RE.match(line)
            if match is None:
                continue
            unit_type = "interface" if "interface" in line else "type"
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
