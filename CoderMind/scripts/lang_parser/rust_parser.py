from __future__ import annotations

import re

from .base import BaseLanguageParser
from .config.rust import RUST_CONFIG
from .extractors.fallback import (
    block_end_for_braces,
    delimiter_syntax_error,
    line_end_for_statement,
    make_unit,
    strip_string_literals,
)
from .models import LPCodeUnit, LPDependency, LPFileResult
from .tree_sitter_backend import TreeSitterBackend


_IDENT = r"[A-Za-z_]\w*"
_VIS = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
_CFG_TEST_RE = re.compile(r"^\s*#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")
_MOD_DECL_RE = re.compile(rf"^\s*{_VIS}mod\s+(?P<name>{_IDENT})\s*;")
_MOD_INLINE_RE = re.compile(rf"^\s*{_VIS}mod\s+(?P<name>{_IDENT})\s*\{{")
_USE_START_RE = re.compile(rf"^\s*{_VIS}use\s+")
_USE_RE = re.compile(rf"^\s*{_VIS}use\s+(?P<body>[\s\S]+?)\s*;")
_STRUCT_RE = re.compile(rf"^\s*{_VIS}struct\s+(?P<name>{_IDENT})\b")
_ENUM_RE = re.compile(rf"^\s*{_VIS}enum\s+(?P<name>{_IDENT})\b")
_TRAIT_RE = re.compile(rf"^\s*{_VIS}(?:unsafe\s+)?trait\s+(?P<name>{_IDENT})\b")
_IMPL_HEADER_RE = re.compile(r"^\s*impl\b(?P<header>[^{]*)\{")
_FN_RE = re.compile(
    rf"^\s*{_VIS}(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
    rf"(?:extern\s+\"[^\"]+\"\s+)?fn\s+(?P<name>{_IDENT})\b"
)
_PATH_CALL_RE = re.compile(rf"(?<![\w:])(?P<path>{_IDENT}(?:::{_IDENT})+)\s*(?:::<[^>]+>\s*)?\(")
_DIRECT_CALL_RE = re.compile(rf"(?<![\w:!])(?P<name>{_IDENT})\s*(?:::<[^>]+>\s*)?\(")
_ALIAS_RE = re.compile(rf"\bas\s+{_IDENT}$")
_TYPE_IDENT_RE = re.compile(_IDENT)

_RUST_CALL_KEYWORDS = frozenset({
    "as",
    "async",
    "await",
    "const",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "pub",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
})
_RUST_BUILTINS = frozenset({
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
    "drop",
    "eprint",
    "eprintln",
    "format",
    "panic",
    "print",
    "println",
    "todo",
    "unimplemented",
    "unreachable",
    "Box",
    "Err",
    "None",
    "Ok",
    "Some",
    "String",
    "Vec",
})
_EXTERNAL_QUALIFIER_PREFIXES = ("std::", "core::", "alloc::")


class RustParser(BaseLanguageParser):
    language = "rust"

    def __init__(self) -> None:
        self.backend = TreeSitterBackend(RUST_CONFIG.tree_sitter_language)

    def parse_file(self, path: str, source: str) -> LPFileResult:
        lines = source.splitlines()
        clean_lines = self._clean_source_lines(source)
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []

        index = 0
        skip_next_test_item = False
        while index < len(lines):
            clean = clean_lines[index].strip()
            if not clean:
                index += 1
                continue

            if _CFG_TEST_RE.match(clean):
                skip_next_test_item = True
                index += 1
                continue

            if skip_next_test_item:
                skip_next_test_item = False
                if "{" in clean:
                    index = block_end_for_braces(clean_lines, index) + 1
                else:
                    index = line_end_for_statement(clean_lines, index) + 1
                continue

            mod_match = _MOD_DECL_RE.match(clean)
            if mod_match is not None:
                self._append_import(
                    units,
                    dependencies,
                    path,
                    lines,
                    mod_match.group("name"),
                    index,
                    index,
                    "rust_mod_decl",
                    "mod_item",
                )
                index += 1
                continue

            if _USE_START_RE.match(clean):
                statement, end_index = self._collect_statement(clean_lines, index)
                use_match = _USE_RE.match(statement)
                if use_match is not None:
                    use_paths = self._expand_use_paths(use_match.group("body"))
                    multiple = len(use_paths) > 1
                    for import_index, use_path in enumerate(use_paths, start=1):
                        self._append_import(
                            units,
                            dependencies,
                            path,
                            lines,
                            use_path,
                            index,
                            end_index,
                            "rust_use",
                            "use_declaration",
                            import_index=import_index if multiple else None,
                        )
                index = end_index + 1
                continue

            inline_mod_match = _MOD_INLINE_RE.match(clean)
            if inline_mod_match is not None:
                index = block_end_for_braces(clean_lines, index) + 1
                continue

            for unit_type, node_type, pattern in (
                ("struct", "struct_item", _STRUCT_RE),
                ("enum", "enum_item", _ENUM_RE),
            ):
                match = pattern.match(clean)
                if match is None:
                    continue
                end = block_end_for_braces(clean_lines, index) if "{" in clean else line_end_for_statement(clean_lines, index)
                units.append(
                    make_unit(
                        name=match.group("name"),
                        unit_type=unit_type,
                        file_path=path,
                        parent=None,
                        lines=lines,
                        line_start=index + 1,
                        line_end=end + 1,
                        language=self.language,
                        node_type=node_type,
                        extra={"kind": unit_type},
                    )
                )
                index = end + 1
                break
            else:
                trait_match = _TRAIT_RE.match(clean)
                if trait_match is not None:
                    end = block_end_for_braces(clean_lines, index) if "{" in clean else line_end_for_statement(clean_lines, index)
                    trait_name = trait_match.group("name")
                    units.append(
                        make_unit(
                            name=trait_name,
                            unit_type="trait",
                            file_path=path,
                            parent=None,
                            lines=lines,
                            line_start=index + 1,
                            line_end=end + 1,
                            language=self.language,
                            node_type="trait_item",
                            extra={"kind": "trait"},
                        )
                    )
                    units.extend(self._extract_trait_methods(path, lines, clean_lines, trait_name, index, end))
                    index = end + 1
                    continue

                impl_target = self._impl_target_from_line(clean)
                if impl_target is not None:
                    target_type, trait_name = impl_target
                    end = block_end_for_braces(clean_lines, index)
                    if trait_name:
                        dependencies.append(
                            LPDependency(
                                src=target_type,
                                dst=trait_name,
                                relation="inherits",
                                symbol=trait_name,
                                line=index + 1,
                                confidence="high",
                                extra={
                                    "language": self.language,
                                    "relation_kind": "trait_impl",
                                    "type": target_type,
                                    "trait": trait_name,
                                },
                            )
                        )
                    units.extend(self._extract_impl_methods(path, lines, clean_lines, target_type, index, end))
                    index = end + 1
                    continue

                fn_match = _FN_RE.match(clean)
                if fn_match is not None:
                    end = self._function_end(clean_lines, index)
                    units.append(
                        make_unit(
                            name=fn_match.group("name"),
                            unit_type="function",
                            file_path=path,
                            parent=None,
                            lines=lines,
                            line_start=index + 1,
                            line_end=end + 1,
                            language=self.language,
                            node_type="function_item",
                        )
                    )
                    index = end + 1
                    continue

                if self._is_unparsed_block_start(clean):
                    index = block_end_for_braces(clean_lines, index) + 1
                    continue

                index += 1
                continue

        dependencies.extend(self._extract_invokes(path, lines, clean_lines, units))
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

    def _append_import(
        self,
        units: list[LPCodeUnit],
        dependencies: list[LPDependency],
        path: str,
        lines: list[str],
        module: str,
        start_index: int,
        end_index: int,
        import_kind: str,
        node_type: str,
        import_index: int | None = None,
    ) -> None:
        extra = {"module": module, "import_path": module, "import_kind": import_kind}
        if import_index is not None:
            extra["import_index"] = import_index
        units.append(
            make_unit(
                name=module,
                unit_type="import",
                file_path=path,
                parent=None,
                lines=lines,
                line_start=start_index + 1,
                line_end=end_index + 1,
                language=self.language,
                node_type=node_type,
                extra=extra,
            )
        )
        dep = LPDependency(
            src=path,
            dst=module,
            relation="imports",
            symbol=self._symbol_from_use_path(module),
            line=start_index + 1,
            confidence="unresolved",
            extra={"language": self.language, "import_kind": import_kind, "import_path": module},
        )
        if import_index is not None:
            dep.extra["import_index"] = import_index
        dependencies.append(dep)

    def _extract_trait_methods(
        self,
        path: str,
        lines: list[str],
        clean_lines: list[str],
        trait_name: str,
        trait_start: int,
        trait_end: int,
    ) -> list[LPCodeUnit]:
        return self._extract_parented_functions(path, lines, clean_lines, trait_name, trait_start, trait_end, "trait_method")

    def _extract_impl_methods(
        self,
        path: str,
        lines: list[str],
        clean_lines: list[str],
        target_type: str,
        impl_start: int,
        impl_end: int,
    ) -> list[LPCodeUnit]:
        return self._extract_parented_functions(path, lines, clean_lines, target_type, impl_start, impl_end, "impl_method")

    def _extract_parented_functions(
        self,
        path: str,
        lines: list[str],
        clean_lines: list[str],
        parent_name: str,
        block_start: int,
        block_end: int,
        node_type: str,
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        depth = self._brace_delta(clean_lines[block_start])
        index = block_start + 1
        while index < block_end:
            clean = clean_lines[index].strip()
            if depth == 1:
                fn_match = _FN_RE.match(clean)
                if fn_match is not None:
                    end = self._function_end(clean_lines, index)
                    units.append(
                        make_unit(
                            name=fn_match.group("name"),
                            unit_type="method",
                            file_path=path,
                            parent=parent_name,
                            lines=lines,
                            line_start=index + 1,
                            line_end=end + 1,
                            language=self.language,
                            node_type=node_type,
                            extra={"receiver_type": parent_name},
                        )
                    )
                    index = end + 1
                    depth = 1
                    continue
            depth += self._brace_delta(clean_lines[index])
            if depth < 0:
                depth = 0
            index += 1
        return units

    def _extract_invokes(
        self,
        path: str,
        lines: list[str],
        clean_lines: list[str],
        units: list[LPCodeUnit],
    ) -> list[LPDependency]:
        import_ranges = [
            (unit.line_start, unit.line_end)
            for unit in units
            if unit.unit_type == "import" and unit.line_start is not None and unit.line_end is not None
        ]
        unit_start_lines = {
            unit.line_start
            for unit in units
            if unit.unit_type in {"struct", "enum", "trait", "function", "method"} and unit.line_start is not None
        }
        dependencies: list[LPDependency] = []
        seen: set[tuple[str, str, int, str, str | None]] = set()

        for line_number, raw_line in enumerate(clean_lines, start=1):
            if any(start <= line_number <= end for start, end in import_ranges):
                continue
            clean = self._clean_line(raw_line)
            if line_number in unit_start_lines:
                clean = clean.split("{", 1)[1] if "{" in clean else ""
            if not clean:
                continue
            source_ref = self._source_reference_for_line(path, units, line_number)

            path_spans: list[tuple[int, int]] = []
            for match in _PATH_CALL_RE.finditer(clean):
                call_path = match.group("path")
                path_spans.append(match.span("path"))
                qualifier, symbol = call_path.rsplit("::", 1)
                if qualifier.startswith(_EXTERNAL_QUALIFIER_PREFIXES):
                    continue
                if symbol in _RUST_CALL_KEYWORDS or symbol in _RUST_BUILTINS:
                    continue
                self._append_invoke_dependency(
                    dependencies,
                    seen,
                    source_ref,
                    symbol,
                    line_number,
                    "path",
                    qualifier,
                )

            for match in _DIRECT_CALL_RE.finditer(clean):
                name = match.group("name")
                if name in _RUST_CALL_KEYWORDS or name in _RUST_BUILTINS:
                    continue
                if any(start <= match.start("name") < end for start, end in path_spans):
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

    def _collect_statement(self, clean_lines: list[str], start_index: int) -> tuple[str, int]:
        parts: list[str] = []
        for index in range(start_index, len(clean_lines)):
            parts.append(clean_lines[index].strip())
            statement = " ".join(parts)
            if ";" in statement:
                return statement, index
        return " ".join(parts), start_index

    def _expand_use_paths(self, body: str) -> list[str]:
        text = re.sub(r"\s+", "", body.strip())
        expanded = [self._normalize_use_path(path) for path in self._expand_use_expr(text)]
        deduped: list[str] = []
        seen: set[str] = set()
        for path in expanded:
            if not path or path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _expand_use_expr(self, expr: str) -> list[str]:
        brace_index = expr.find("{")
        if brace_index == -1:
            return [expr]
        close_index = self._matching_brace(expr, brace_index)
        if close_index == -1:
            return [expr]

        prefix = expr[:brace_index]
        suffix = expr[close_index + 1:]
        body = expr[brace_index + 1:close_index]
        results: list[str] = []
        for part in self._split_top_level_commas(body):
            if not part:
                continue
            if part == "self":
                combined = prefix[:-2] if prefix.endswith("::") else prefix
            else:
                combined = f"{prefix}{part}"
            results.extend(self._expand_use_expr(f"{combined}{suffix}"))
        return results

    def _split_top_level_commas(self, text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        depth = 0
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(text[start:index])
                start = index + 1
        parts.append(text[start:])
        return [part for part in parts if part]

    def _matching_brace(self, text: str, open_index: int) -> int:
        depth = 0
        for index in range(open_index, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    def _normalize_use_path(self, path: str) -> str:
        normalized = _ALIAS_RE.sub("", path).strip(":")
        return normalized.rstrip(",")

    def _symbol_from_use_path(self, path: str) -> str | None:
        if not path:
            return None
        if path.endswith("::*"):
            return "*"
        return path.rsplit("::", 1)[-1]

    def _impl_target_from_line(self, clean_line: str) -> tuple[str, str | None] | None:
        match = _IMPL_HEADER_RE.match(clean_line)
        if match is None:
            return None
        header = match.group("header").strip()
        if not header:
            return None
        header = self._strip_leading_generics(header)
        if " for " in header:
            trait_part, target_part = header.rsplit(" for ", 1)
            target = self._type_name_from_fragment(target_part)
            trait_name = self._type_name_from_fragment(trait_part)
            if target and trait_name:
                return target, trait_name
            return None
        target = self._type_name_from_fragment(header)
        if target:
            return target, None
        return None

    def _strip_leading_generics(self, header: str) -> str:
        stripped = header.strip()
        if not stripped.startswith("<"):
            return stripped
        depth = 0
        for index, char in enumerate(stripped):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    return stripped[index + 1:].strip()
        return stripped

    def _type_name_from_fragment(self, fragment: str) -> str | None:
        text = self._remove_angle_groups(fragment)
        text = text.replace("&", " ").replace("'", " ")
        identifiers = _TYPE_IDENT_RE.findall(text)
        ignored = {"as", "const", "dyn", "for", "impl", "mut", "Self", "self", "unsafe", "where"}
        candidates = [identifier for identifier in identifiers if identifier not in ignored]
        if not candidates:
            return None
        return candidates[-1]

    def _remove_angle_groups(self, text: str) -> str:
        result: list[str] = []
        depth = 0
        for char in text:
            if char == "<":
                depth += 1
                result.append(" ")
            elif char == ">" and depth > 0:
                depth -= 1
                result.append(" ")
            elif depth == 0:
                result.append(char)
        return "".join(result)

    def _function_end(self, clean_lines: list[str], start_index: int) -> int:
        max_end = min(len(clean_lines), start_index + 12)
        statement_parts: list[str] = []
        for end_index in range(start_index, max_end):
            statement_parts.append(clean_lines[end_index].strip())
            statement = " ".join(statement_parts)
            if "{" in statement:
                return block_end_for_braces(clean_lines, end_index)
            if ";" in statement:
                return end_index
        return line_end_for_statement(clean_lines, start_index)

    def _is_declaration_call_context(self, clean_line: str, match_start: int) -> bool:
        prefix = clean_line[:match_start].rstrip()
        return re.search(r"(?:^|\b)(?:fn|struct|enum|trait|impl|mod|use|if|for|while|match)\s*$", prefix) is not None

    def _is_unparsed_block_start(self, clean_line: str) -> bool:
        if "{" not in clean_line:
            return False
        stripped = clean_line.lstrip()
        return stripped.startswith(("extern ", "macro_rules!", "const ", "static "))

    def _syntax_error(self, source: str) -> str | None:
        backend_result = self.backend.validate_syntax(source)
        if backend_result is not None:
            valid, error = backend_result
            if not valid:
                return error
            return None
        return delimiter_syntax_error(source)

    def _clean_source_lines(self, source: str) -> list[str]:
        without_block_comments = self._strip_block_comments(source)
        return [self._clean_line(line) for line in without_block_comments.splitlines()]

    def _strip_block_comments(self, source: str) -> str:
        result_lines: list[str] = []
        in_block = False
        for line in source.splitlines():
            index = 0
            chars: list[str] = []
            while index < len(line):
                if in_block:
                    end = line.find("*/", index)
                    if end == -1:
                        break
                    index = end + 2
                    in_block = False
                    continue
                start = line.find("/*", index)
                if start == -1:
                    chars.append(line[index:])
                    break
                chars.append(line[index:start])
                index = start + 2
                in_block = True
            result_lines.append("".join(chars))
        return "\n".join(result_lines)

    def _clean_line(self, line: str) -> str:
        return strip_string_literals(line).split("//", 1)[0]

    def _brace_delta(self, line: str) -> int:
        clean = self._clean_line(line)
        return clean.count("{") - clean.count("}")
