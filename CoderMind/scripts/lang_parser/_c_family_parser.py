from __future__ import annotations

import re

from .base import BaseLanguageParser
from .extractors.fallback import (
    block_end_for_braces,
    delimiter_syntax_error,
    dependency_from_import,
    line_end_for_statement,
    make_unit,
    strip_string_literals,
)
from .models import LanguageConfig, LPCodeUnit, LPDependency, LPFileResult
from .tree_sitter_backend import TreeSitterBackend


_IDENTIFIER = r"[A-Za-z_]\w*"
_INCLUDE_QUOTE_RE = re.compile(r'^\s*#\s*include\s+"(?P<path>[^"]+)"')
_INCLUDE_ANGLE_RE = re.compile(r"^\s*#\s*include\s+<(?P<path>[^>]+)>")
_CLASS_LIKE_RE = re.compile(
    rf"^\s*(?:template\s*<[^>]+>\s*)?(?:typedef\s+)?(?P<kind>class|struct)\s+"
    rf"(?P<name>{_IDENTIFIER})\b[^;{{}}]*\{{"
)
_METHOD_RE = re.compile(
    rf"^\s*(?:(?:virtual|static|inline|constexpr|consteval|explicit|friend)\s+)*"
    rf"(?:(?:[\w:<>~*&,.\[\]\s]+)\s+)?"
    rf"(?P<name>~?{_IDENTIFIER})\s*\([^;{{}}]*\)\s*"
    rf"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?(?:override\s*)?(?:final\s*)?"
    rf"(?:=\s*(?:0|default|delete)\s*)?(?::[^{{;]+)?\s*(?:\{{|;)"
)
_FUNCTION_DEF_RE = re.compile(
    rf"^\s*(?:(?:extern\s+\"C\"\s+)?(?P<prefix>(?:[\w:<>~*&,.\[\]\s]+)\s+))?"
    rf"(?:(?P<parent>{_IDENTIFIER})::)?(?P<name>~?{_IDENTIFIER})\s*\([^;{{}}]*\)\s*"
    rf"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?(?:override\s*)?(?:final\s*)?"
    rf"(?:->\s*[^{{;]+)?(?::[^{{;]+)?\s*\{{"
)
_FUNCTION_DECL_RE = re.compile(
    rf"^\s*(?:(?:extern\s+\"C\"\s+)?(?P<prefix>(?:[\w:<>~*&,\.\[\]\s]+)\s+))?"
    rf"(?:(?P<parent>{_IDENTIFIER})::)?(?P<name>~?{_IDENTIFIER})\s*\([^;{{}}]*\)\s*"
    rf"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?(?:override\s*)?(?:final\s*)?"
    rf"(?:->\s*[^{{;]+)?(?::[^{{;]+)?\s*;"
)
_STATIC_CALL_RE = re.compile(rf"(?<![\w:])(?P<qualifier>{_IDENTIFIER})::(?P<name>~?{_IDENTIFIER})\s*\(")
_NEW_EXPRESSION_RE = re.compile(rf"\bnew\s+(?P<name>{_IDENTIFIER})\s*\(")
_DIRECT_CALL_RE = re.compile(rf"(?<![\w:.>])(?P<name>{_IDENTIFIER})\s*\(")
_ACCESS_SPECIFIER_RE = re.compile(r"^\s*(?:public|private|protected)\s*:\s*$")

_C_OPENING_KEYWORDS = frozenset({
    "if",
    "else",
    "while",
    "for",
    "do",
    "switch",
    "try",
    "catch",
    "namespace",
    "extern",
})
_C_CALL_KEYWORDS = frozenset({
    "if",
    "while",
    "for",
    "switch",
    "return",
    "sizeof",
    "alignof",
    "typeof",
    "offsetof",
    "new",
    "delete",
    "catch",
    "throw",
    "decltype",
    "static_assert",
})
_C_BUILTINS = frozenset({
    "printf",
    "fprintf",
    "sprintf",
    "snprintf",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "memcpy",
    "memmove",
    "memset",
    "strlen",
    "strcpy",
    "strncpy",
    "strcmp",
    "strncmp",
    "strcat",
    "puts",
    "putchar",
    "fopen",
    "fclose",
    "fread",
    "fwrite",
})


class CFamilyParser(BaseLanguageParser):
    language = "c"

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.backend = TreeSitterBackend(config.tree_sitter_language)

    def parse_file(self, path: str, source: str) -> LPFileResult:
        lines = source.splitlines()
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []

        units.extend(self._extract_includes(path, lines, dependencies))
        class_units, class_ranges = self._extract_class_like_units(path, lines)
        units.extend(class_units)
        units.extend(self._extract_functions(path, lines, class_ranges))
        dependencies.extend(self._extract_invokes(path, lines, units))

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

    def _extract_includes(
        self,
        path: str,
        lines: list[str],
        dependencies: list[LPDependency],
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        for index, line in enumerate(lines):
            match = _INCLUDE_QUOTE_RE.match(line)
            include_style = "quote"
            if match is None:
                match = _INCLUDE_ANGLE_RE.match(line)
                include_style = "angle"
            if match is None:
                continue

            include_path = match.group("path")
            units.append(
                make_unit(
                    name=include_path,
                    unit_type="import",
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=index + 1,
                    line_end=index + 1,
                    language=self.language,
                    node_type="preproc_include",
                    extra={
                        "module": include_path,
                        "import_path": include_path,
                        "include_style": include_style,
                    },
                )
            )
            dep = dependency_from_import(
                path=path,
                module=include_path,
                symbol=include_path,
                line=index + 1,
                language=self.language,
                import_kind=f"{self.language}_include",
            )
            dep.extra.update({"include_style": include_style, "import_path": include_path})
            dependencies.append(dep)
        return units

    def _extract_class_like_units(self, path: str, lines: list[str]) -> tuple[list[LPCodeUnit], list[tuple[int, int]]]:
        units: list[LPCodeUnit] = []
        ranges: list[tuple[int, int]] = []
        index = 0
        while index < len(lines):
            match = _CLASS_LIKE_RE.match(self._clean_line(lines[index]))
            if match is None:
                index += 1
                continue

            name = match.group("name")
            kind = match.group("kind")
            unit_type = "class" if kind == "class" else "struct"
            end = block_end_for_braces(lines, index)
            ranges.append((index, end))
            units.append(
                make_unit(
                    name=name,
                    unit_type=unit_type,
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=index + 1,
                    line_end=end + 1,
                    language=self.language,
                    node_type=f"{kind}_specifier",
                    extra={"kind": kind},
                )
            )
            units.extend(self._extract_methods(path, lines, name, index, end))
            index = end + 1
        return units, ranges

    def _extract_methods(
        self,
        path: str,
        lines: list[str],
        class_name: str,
        class_start: int,
        class_end: int,
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        depth = self._brace_delta(lines[class_start])
        index = class_start + 1
        while index < class_end:
            line = lines[index]
            clean = self._clean_line(line)
            if depth == 1 and not _ACCESS_SPECIFIER_RE.match(clean):
                match = _METHOD_RE.match(clean)
                if match is not None:
                    method_end = block_end_for_braces(lines, index) if "{" in clean else line_end_for_statement(lines, index)
                    units.append(
                        make_unit(
                            name=match.group("name"),
                            unit_type="method",
                            file_path=path,
                            parent=class_name,
                            lines=lines,
                            line_start=index + 1,
                            line_end=method_end + 1,
                            language=self.language,
                            node_type="method_definition" if "{" in clean else "method_declaration",
                        )
                    )
            depth += self._brace_delta(line)
            if depth < 0:
                depth = 0
            index += 1
        return units

    def _extract_functions(
        self,
        path: str,
        lines: list[str],
        excluded_ranges: list[tuple[int, int]],
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        index = 0
        while index < len(lines):
            if self._in_ranges(index, excluded_ranges):
                index += 1
                continue

            match, signature_end, has_body = self._match_function(lines, index)
            if match is None:
                index += 1
                continue

            name = match.group("name")
            parent = match.group("parent")
            if name in _C_OPENING_KEYWORDS or (not parent and not match.group("prefix")):
                index += 1
                continue

            end = block_end_for_braces(lines, signature_end) if has_body else line_end_for_statement(lines, signature_end)
            unit_type = "method" if parent else "function"
            extra = {"qualified_parent": parent} if parent else None
            units.append(
                make_unit(
                    name=name,
                    unit_type=unit_type,
                    file_path=path,
                    parent=parent,
                    lines=lines,
                    line_start=index + 1,
                    line_end=end + 1,
                    language=self.language,
                    node_type=(
                        "method_definition" if parent and has_body
                        else "method_declaration" if parent
                        else "function_definition" if has_body
                        else "function_declaration"
                    ),
                    extra=extra,
                )
            )
            index = end + 1
        return units

    def _match_function(
        self,
        lines: list[str],
        start_index: int,
    ) -> tuple[re.Match[str] | None, int, bool]:
        if not self._clean_line(lines[start_index]).strip():
            return None, start_index, False

        statement_parts: list[str] = []
        max_end = min(len(lines), start_index + 8)
        for end_index in range(start_index, max_end):
            clean = self._clean_line(lines[end_index]).strip()
            if not clean:
                continue
            statement_parts.append(clean)
            statement = " ".join(statement_parts)
            open_index = statement.find("{")
            semi_index = statement.find(";")
            if semi_index != -1 and (open_index == -1 or semi_index < open_index):
                match = _FUNCTION_DECL_RE.match(statement)
                if match is None:
                    return None, start_index, False
                return match, end_index, False
            if open_index == -1:
                continue
            match = _FUNCTION_DEF_RE.match(statement)
            if match is None:
                return None, start_index, False
            return match, end_index, True
        return None, start_index, False

    def _extract_invokes(self, path: str, lines: list[str], units: list[LPCodeUnit]) -> list[LPDependency]:
        import_ranges = [
            (unit.line_start, unit.line_end)
            for unit in units
            if unit.unit_type == "import" and unit.line_start is not None and unit.line_end is not None
        ]
        unit_start_lines = {
            unit.line_start
            for unit in units
            if unit.unit_type in {"class", "struct", "function", "method"} and unit.line_start is not None
        }
        dependencies: list[LPDependency] = []
        seen: set[tuple[str, str, int, str, str | None]] = set()

        for line_number, line in enumerate(lines, start=1):
            if any(start <= line_number <= end for start, end in import_ranges):
                continue
            clean = self._clean_line(line)
            if line_number in unit_start_lines:
                clean = clean.split("{", 1)[1] if "{" in clean else ""
            if not clean:
                continue
            source_ref = self._source_reference_for_line(path, units, line_number)

            static_spans: list[tuple[int, int]] = []
            for match in _STATIC_CALL_RE.finditer(clean):
                qualifier = match.group("qualifier")
                name = match.group("name")
                static_spans.append(match.span("name"))
                if name in _C_CALL_KEYWORDS:
                    continue
                self._append_invoke_dependency(
                    dependencies,
                    seen,
                    source_ref,
                    name,
                    line_number,
                    "static",
                    qualifier,
                )

            constructor_spans: list[tuple[int, int]] = []
            for match in _NEW_EXPRESSION_RE.finditer(clean):
                name = match.group("name")
                constructor_spans.append(match.span("name"))
                self._append_invoke_dependency(
                    dependencies,
                    seen,
                    source_ref,
                    name,
                    line_number,
                    "constructor",
                    None,
                )

            for match in _DIRECT_CALL_RE.finditer(clean):
                name = match.group("name")
                if name in _C_CALL_KEYWORDS or name in _C_BUILTINS:
                    continue
                if any(start <= match.start("name") < end for start, end in static_spans + constructor_spans):
                    continue
                if self._is_declaration_call_context(clean, match.start("name")):
                    continue
                self._append_invoke_dependency(
                    dependencies,
                    seen,
                    source_ref,
                    name,
                    line_number,
                    "direct",
                    None,
                )
        return dependencies

    def _append_invoke_dependency(
        self,
        dependencies: list[LPDependency],
        seen: set[tuple[str, str, int, str, str | None]],
        source_ref: str,
        name: str,
        line_number: int,
        call_kind: str,
        qualifier: str | None,
    ) -> None:
        key = (source_ref, name, line_number, call_kind, qualifier)
        if key in seen:
            return
        seen.add(key)
        extra = {"language": self.language, "call_kind": call_kind}
        if qualifier:
            extra["qualifier"] = qualifier
        dependencies.append(
            LPDependency(
                src=source_ref,
                dst=qualifier or name,
                relation="invokes",
                symbol=name,
                line=line_number,
                confidence="high",
                extra=extra,
            )
        )

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

    def _is_declaration_call_context(self, clean_line: str, match_start: int) -> bool:
        prefix = clean_line[:match_start].rstrip()
        if prefix.endswith(("#define", "typedef")):
            return True
        if re.search(r"(?:^|\b)(?:class|struct|enum|if|for|while|switch|catch)\s*$", prefix):
            return True
        return False

    def _syntax_error(self, source: str) -> str | None:
        backend_result = self.backend.validate_syntax(source)
        if backend_result is not None:
            valid, error = backend_result
            if not valid:
                return error
        return delimiter_syntax_error(source)

    def _clean_line(self, line: str) -> str:
        return strip_string_literals(line).split("//", 1)[0]

    def _brace_delta(self, line: str) -> int:
        clean = self._clean_line(line)
        return clean.count("{") - clean.count("}")

    def _in_ranges(self, index: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= index <= end for start, end in ranges)
