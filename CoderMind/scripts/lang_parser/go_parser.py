from __future__ import annotations

import re

from .base import BaseLanguageParser
from .config.go import GO_CONFIG
from .extractors.fallback import (
    block_end_for_braces,
    delimiter_syntax_error,
    dependency_from_import,
    make_unit,
    strip_string_literals,
)
from .models import LPCodeUnit, LPDependency, LPFileResult
from .tree_sitter_backend import TreeSitterBackend


_PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_]\w*)\b")
_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+(?:(?P<alias>[\w._]+)\s+)?["`](?P<path>[^"`]+)["`]')
_IMPORT_SPEC_RE = re.compile(r'^\s*(?:(?P<alias>[\w._]+)\s+)?["`](?P<path>[^"`]+)["`]')
_GO_IDENTIFIER = r"[A-Za-z_]\w*"
_GO_TYPE_PARAMS = r"\[[^\]]+\]"
_STRUCT_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+struct\b")
_INTERFACE_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+interface\b")
_METHOD_RE = re.compile(
    rf"^\s*func\s*\(\s*(?:{_GO_IDENTIFIER}\s+)?\*?\s*(?P<receiver>{_GO_IDENTIFIER})"
    rf"\s*(?:{_GO_TYPE_PARAMS})?\s*\)\s*(?P<name>{_GO_IDENTIFIER})\s*(?:{_GO_TYPE_PARAMS})?\s*\("
)
_FUNCTION_RE = re.compile(rf"^\s*func\s+(?P<name>{_GO_IDENTIFIER})\s*(?:{_GO_TYPE_PARAMS})?\s*\(")
_SELECTOR_CALL_RE = re.compile(r"(?<![\w.])(?P<qualifier>[A-Za-z_]\w*)\.(?P<name>[A-Za-z_]\w*)\s*\(")
_DIRECT_CALL_RE = re.compile(r"(?<![\w.])(?P<name>[A-Za-z_]\w*)\s*\(")
_GO_CALL_KEYWORDS = frozenset({
    "if", "for", "switch", "select", "go", "defer", "return", "func", "range",
})
_GO_BUILTINS = frozenset({
    "append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len",
    "make", "max", "min", "new", "panic", "print", "println", "real", "recover",
})


class GoParser(BaseLanguageParser):
    language = "go"

    def __init__(self) -> None:
        self.backend = TreeSitterBackend(GO_CONFIG.tree_sitter_language)

    def parse_file(self, path: str, source: str) -> LPFileResult:
        lines = source.splitlines()
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []
        import_aliases: dict[str, str] = {}
        index = 0
        while index < len(lines):
            line = lines[index]
            package_match = _PACKAGE_RE.match(line)
            if package_match:
                units.append(
                    make_unit(
                        name=package_match.group(1),
                        unit_type="package",
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=index + 1,
                        language=self.language,
                        node_type="package_clause",
                    )
                )
                index += 1
                continue

            stripped = line.strip()
            if stripped.startswith("import ("):
                block_start = index
                index += 1
                while index < len(lines):
                    if lines[index].strip().startswith(")"):
                        break
                    spec_match = _IMPORT_SPEC_RE.match(lines[index])
                    if spec_match:
                        import_path = spec_match.group("path")
                        alias = spec_match.group("alias")
                        qualifier = self._import_qualifier(import_path, alias)
                        if qualifier:
                            import_aliases[qualifier] = import_path
                        units.append(
                            make_unit(
                                name=import_path,
                                unit_type="import",
                                file_path=path,
                                parent=None,
                                lines=lines,
                                line_start=index + 1,
                                line_end=index + 1,
                                language=self.language,
                                node_type="import_spec",
                                extra={"alias": alias, "import_path": import_path, "qualifier": qualifier},
                            )
                        )
                        dep = dependency_from_import(
                            path=path,
                            module=import_path,
                            symbol=alias or import_path,
                            line=index + 1,
                            language=self.language,
                            import_kind="go_import",
                        )
                        dep.extra.update({"alias": alias, "qualifier": qualifier})
                        dependencies.append(dep)
                    index += 1
                index = max(index + 1, block_start + 1)
                continue

            import_match = _IMPORT_SINGLE_RE.match(line)
            if import_match:
                import_path = import_match.group("path")
                alias = import_match.group("alias")
                qualifier = self._import_qualifier(import_path, alias)
                if qualifier:
                    import_aliases[qualifier] = import_path
                units.append(
                    make_unit(
                        name=import_path,
                        unit_type="import",
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=index + 1,
                        language=self.language,
                        node_type="import_declaration",
                        extra={"alias": alias, "import_path": import_path, "qualifier": qualifier},
                    )
                )
                dep = dependency_from_import(
                    path=path,
                    module=import_path,
                    symbol=alias or import_path,
                    line=index + 1,
                    language=self.language,
                    import_kind="go_import",
                )
                dep.extra.update({"alias": alias, "qualifier": qualifier})
                dependencies.append(dep)
                index += 1
                continue

            struct_match = _STRUCT_RE.match(line)
            if struct_match:
                end = block_end_for_braces(lines, index)
                units.append(
                    make_unit(
                        name=struct_match.group(1),
                        unit_type="struct",
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=end + 1,
                        language=self.language,
                        node_type="struct_type",
                        extra={"kind": "struct"},
                    )
                )
                index = end + 1
                continue

            interface_match = _INTERFACE_RE.match(line)
            if interface_match:
                end = block_end_for_braces(lines, index)
                units.append(
                    make_unit(
                        name=interface_match.group(1),
                        unit_type="interface",
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=end + 1,
                        language=self.language,
                        node_type="interface_type",
                        extra={"kind": "interface"},
                    )
                )
                index = end + 1
                continue

            method_match = _METHOD_RE.match(line)
            if method_match:
                end = block_end_for_braces(lines, index)
                receiver_type = method_match.group("receiver").replace("*", "").strip()
                units.append(
                    make_unit(
                        name=method_match.group("name"),
                        unit_type="method",
                        file_path=path,
                        parent=receiver_type,
                        lines=lines,
                        line_start=index + 1,
                        line_end=end + 1,
                        language=self.language,
                        node_type="method_declaration",
                        extra={"receiver_type": receiver_type},
                    )
                )
                index = end + 1
                continue

            function_match = _FUNCTION_RE.match(line)
            if function_match:
                end = block_end_for_braces(lines, index)
                units.append(
                    make_unit(
                        name=function_match.group("name"),
                        unit_type="function",
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=end + 1,
                        language=self.language,
                        node_type="function_declaration",
                    )
                )
                index = end + 1
                continue

            index += 1

        dependencies.extend(self._extract_invokes(path, lines, units, import_aliases))

        syntax_error = self._syntax_error(source)
        return LPFileResult(
            file_path=path,
            language=self.language,
            units=units,
            dependencies=dependencies,
            syntax_error=syntax_error,
        )

    def validate_syntax(self, path: str, source: str) -> tuple[bool, str | None]:
        syntax_error = self._syntax_error(source)
        return (syntax_error is None, syntax_error)

    def _import_qualifier(self, import_path: str, alias: str | None) -> str | None:
        if alias in {".", "_"}:
            return None
        if alias:
            return alias
        return import_path.rsplit("/", 1)[-1]

    def _extract_invokes(
        self,
        path: str,
        lines: list[str],
        units: list[LPCodeUnit],
        import_aliases: dict[str, str],
    ) -> list[LPDependency]:
        import_ranges = [
            (unit.line_start, unit.line_end)
            for unit in units
            if unit.unit_type == "import"
            and unit.line_start is not None
            and unit.line_end is not None
        ]
        dependencies: list[LPDependency] = []
        seen: set[tuple[str, str, int, str, str | None]] = set()

        for line_number, line in enumerate(lines, start=1):
            if any(start <= line_number <= end for start, end in import_ranges):
                continue
            clean = strip_string_literals(line).split("//", 1)[0]
            if clean.lstrip().startswith("func "):
                if "{" not in clean:
                    continue
                clean = clean.split("{", 1)[1]
            source_ref = self._source_reference_for_line(path, units, line_number)

            for match in _SELECTOR_CALL_RE.finditer(clean):
                qualifier = match.group("qualifier")
                name = match.group("name")
                import_path = import_aliases.get(qualifier)
                if not import_path:
                    continue
                key = (source_ref, name, line_number, "selector", qualifier)
                if key in seen:
                    continue
                seen.add(key)
                dependencies.append(
                    LPDependency(
                        src=source_ref,
                        dst=import_path,
                        relation="invokes",
                        symbol=name,
                        line=line_number,
                        confidence="high",
                        extra={
                            "language": self.language,
                            "call_kind": "selector",
                            "qualifier": qualifier,
                            "module": import_path,
                        },
                    )
                )

            for match in _DIRECT_CALL_RE.finditer(clean):
                name = match.group("name")
                if name in _GO_CALL_KEYWORDS or name in _GO_BUILTINS:
                    continue
                if self._is_direct_call_declaration(clean, match.start("name")):
                    continue
                key = (source_ref, name, line_number, "direct", None)
                if key in seen:
                    continue
                seen.add(key)
                dependencies.append(
                    LPDependency(
                        src=source_ref,
                        dst=name,
                        relation="invokes",
                        symbol=name,
                        line=line_number,
                        confidence="high",
                        extra={"language": self.language, "call_kind": "direct"},
                    )
                )
        return dependencies

    def _source_reference_for_line(self, path: str, units: list[LPCodeUnit], line_number: int) -> str:
        candidates = [
            unit
            for unit in units
            if unit.unit_type in {"function", "method"}
            and unit.line_start is not None
            and unit.line_end is not None
            and unit.line_start <= line_number <= unit.line_end
        ]
        if not candidates:
            return path
        candidates.sort(key=lambda unit: (unit.line_end or line_number) - (unit.line_start or line_number))
        unit = candidates[0]
        if unit.parent and unit.name:
            return f"{path}:{unit.parent}.{unit.name}"
        if unit.name:
            return f"{path}:{unit.name}"
        return path

    def _is_direct_call_declaration(self, clean_line: str, match_start: int) -> bool:
        prefix = clean_line[:match_start].rstrip()
        return prefix.endswith("func") or prefix.endswith("go") or prefix.endswith("defer")

    def _syntax_error(self, source: str) -> str | None:
        backend_result = self.backend.validate_syntax(source)
        if backend_result is not None:
            valid, error = backend_result
            if not valid:
                return error
        delimiter_error = delimiter_syntax_error(source)
        if delimiter_error:
            return delimiter_error
        if not re.search(r"(?m)^\s*package\s+[A-Za-z_]\w*\b", source.strip()):
            return "Go source is missing a package clause"
        return None
