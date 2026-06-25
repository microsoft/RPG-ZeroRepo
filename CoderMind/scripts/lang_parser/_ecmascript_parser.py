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


_IDENTIFIER = r"[A-Za-z_$][\w$]*"
_IMPORT_RE = re.compile(r"^\s*import\b")
_EXPORT_FROM_RE = re.compile(r"^\s*export\b[\s\S]*\bfrom\s+['\"]([^'\"]+)['\"]")
_IMPORT_FROM_RE = re.compile(r"\bfrom\s+['\"]([^'\"]+)['\"]")
_IMPORT_SIDE_EFFECT_RE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]")
_DEFAULT_EXPORT_RE = re.compile(r"^\s*export\s+default\b")
_CLASS_RE = re.compile(
    rf"^\s*(?:export\s+default\s+|export\s+)?(?:abstract\s+)?class\s+(?P<name>{_IDENTIFIER})\b"
)
_METHOD_RE = re.compile(
    rf"^\s*(?:(?:public|private|protected|static|async|override|readonly|abstract)\s+)*"
    rf"(?:get\s+|set\s+)?(?P<name>constructor|#?{_IDENTIFIER})\s*"
    rf"(?:<[^>{{}}]+>)?\([^)]*\)\s*(?::[^={{;]+)?(?:\{{|;)"
)
_FIELD_METHOD_RE = re.compile(
    rf"^\s*(?:(?:public|private|protected|static|async|override|readonly)\s+)*"
    rf"(?P<name>#?{_IDENTIFIER})\s*=\s*(?:async\s*)?(?:\([^)]*\)|{_IDENTIFIER})\s*"
    rf"(?::[^=]+)?=>\s*\{{"
)
_FUNCTION_RE = re.compile(
    rf"^\s*(?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+(?P<name>{_IDENTIFIER})\s*\("
)
_ARROW_FUNCTION_RE = re.compile(
    rf"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>{_IDENTIFIER})\s*=\s*"
    rf"(?:async\s*)?(?:\([^)]*\)|{_IDENTIFIER})\s*(?::[^=]+)?=>"
)
_COMMONJS_FUNCTION_RE = re.compile(
    rf"^\s*(?:module\.exports\.|exports\.)(?P<name>{_IDENTIFIER})\s*=\s*(?:async\s+)?function\b"
)
_NEW_EXPRESSION_RE = re.compile(rf"\bnew\s+(?P<name>{_IDENTIFIER})\s*(?:<[^>]+>)?\(")
_MEMBER_CALL_RE = re.compile(rf"(?<![\w$])(?P<qualifier>{_IDENTIFIER})\.(?P<name>{_IDENTIFIER})\s*(?:<[^>]+>)?\(")
_DIRECT_CALL_RE = re.compile(rf"(?<![\w$.])(?P<name>{_IDENTIFIER})\s*(?:<[^>]+>)?\(")
_CALL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "function", "return", "typeof",
    "void", "delete", "await", "new", "super", "import", "require",
})
_DECLARATION_PREFIX_RE = re.compile(
    r"(?:^|\b)(?:function|class|interface|if|for|while|switch|catch|with|const|let|var)\s*$"
)


class ECMAScriptParser(BaseLanguageParser):
    language = "javascript"

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.backend = TreeSitterBackend(config.tree_sitter_language)

    def parse_file(self, path: str, source: str) -> LPFileResult:
        lines = source.splitlines()
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []

        units.extend(self._extract_imports(path, lines, dependencies))
        class_units, class_ranges = self._extract_classes(path, lines)
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

    def _extract_imports(
        self,
        path: str,
        lines: list[str],
        dependencies: list[LPDependency],
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not (_IMPORT_RE.match(line) or self._looks_like_export_from_start(line)):
                index += 1
                continue

            end = self._line_end_for_import_export(lines, index)
            statement = "\n".join(lines[index:end + 1])
            module = self._module_from_import(statement)
            if module is None:
                index += 1
                continue

            bindings = self._import_bindings_from_statement(statement, module)
            extra = {"module": module}
            if bindings:
                extra["bindings"] = bindings
            units.append(
                make_unit(
                    name=module or statement.strip(),
                    unit_type="import",
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=index + 1,
                    line_end=end + 1,
                    language=self.language,
                    node_type="import_statement",
                    extra=extra,
                )
            )
            dep = dependency_from_import(
                path=path,
                module=module,
                symbol=module,
                line=index + 1,
                language=self.language,
                import_kind=f"{self.language}_import",
            )
            if bindings:
                dep.extra["bindings"] = bindings
            dependencies.append(dep)
            index = end + 1
        return units

    def _looks_like_export_from_start(self, line: str) -> bool:
        stripped = line.lstrip()
        return (
            stripped.startswith("export {")
            or stripped.startswith("export type {")
            or stripped.startswith("export *")
            or _EXPORT_FROM_RE.match(line) is not None
        )

    def _line_end_for_import_export(self, lines: list[str], start_index: int) -> int:
        for index in range(start_index, len(lines)):
            statement = "\n".join(lines[start_index:index + 1])
            if self._module_from_import(statement) is not None:
                return index
            if lines[index].strip().endswith(";"):
                return index
        return len(lines) - 1

    def _import_bindings_from_statement(self, statement: str, module: str) -> dict[str, dict[str, str]]:
        stripped = statement.strip()
        if not _IMPORT_RE.match(stripped) or _IMPORT_SIDE_EFFECT_RE.match(stripped):
            return {}
        if re.match(r"^import\s+type\b", stripped):
            return {}

        before_from = re.split(r"\bfrom\b", stripped, maxsplit=1)[0]
        before_from = re.sub(r"^import\s+", "", before_from, count=1).strip()
        bindings: dict[str, dict[str, str]] = {}

        namespace_match = re.match(rf"\*\s+as\s+(?P<local>{_IDENTIFIER})$", before_from)
        if namespace_match:
            local = namespace_match.group("local")
            bindings[local] = {"module": module, "imported": "*", "kind": "namespace"}
            return bindings

        named_match = re.search(r"\{(?P<body>[\s\S]*)\}", before_from)
        default_part = before_from
        if named_match:
            default_part = before_from[:named_match.start()].strip().rstrip(",")
            for item in named_match.group("body").split(","):
                item = item.strip()
                if not item or item.startswith("type "):
                    continue
                if " as " in item:
                    imported, local = [part.strip() for part in item.split(" as ", 1)]
                else:
                    imported = local = item
                if re.fullmatch(_IDENTIFIER, imported) and re.fullmatch(_IDENTIFIER, local):
                    bindings[local] = {"module": module, "imported": imported, "kind": "named"}

        default_part = default_part.strip()
        if default_part:
            default_name = default_part.split(",", 1)[0].strip()
            if re.fullmatch(_IDENTIFIER, default_name):
                bindings[default_name] = {"module": module, "imported": default_name, "kind": "default"}

        return bindings

    def _extract_classes(self, path: str, lines: list[str]) -> tuple[list[LPCodeUnit], list[tuple[int, int]]]:
        units: list[LPCodeUnit] = []
        ranges: list[tuple[int, int]] = []
        index = 0
        while index < len(lines):
            match = _CLASS_RE.match(lines[index])
            if not match:
                index += 1
                continue

            class_name = match.group("name")
            end = block_end_for_braces(lines, index)
            ranges.append((index, end))
            extra = {"export_default": True} if _DEFAULT_EXPORT_RE.match(lines[index]) else None
            units.append(
                make_unit(
                    name=class_name,
                    unit_type="class",
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=index + 1,
                    line_end=end + 1,
                    language=self.language,
                    node_type="class_declaration",
                    extra=extra,
                )
            )
            units.extend(self._extract_methods(path, lines, class_name, index, end))
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
        depth = strip_string_literals(lines[class_start]).count("{") - strip_string_literals(lines[class_start]).count("}")
        index = class_start + 1
        while index < class_end:
            line = lines[index]
            clean = strip_string_literals(line)
            if depth == 1:
                match = _METHOD_RE.match(line) or _FIELD_METHOD_RE.match(line)
                if match:
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
                            node_type="method_definition",
                        )
                    )
            depth += clean.count("{") - clean.count("}")
            index += 1
        return units

    def _extract_functions(
        self,
        path: str,
        lines: list[str],
        excluded_ranges: list[tuple[int, int]],
    ) -> list[LPCodeUnit]:
        units: list[LPCodeUnit] = []
        depth = 0
        index = 0
        while index < len(lines):
            if any(start <= index <= end for start, end in excluded_ranges):
                index += 1
                continue

            line = lines[index]
            clean = strip_string_literals(line)
            if depth == 0:
                match = _FUNCTION_RE.match(line) or _ARROW_FUNCTION_RE.match(line) or _COMMONJS_FUNCTION_RE.match(line)
                if match:
                    end = block_end_for_braces(lines, index) if "{" in clean else line_end_for_statement(lines, index)
                    extra = {"export_default": True} if _DEFAULT_EXPORT_RE.match(line) else None
                    units.append(
                        make_unit(
                            name=match.group("name"),
                            unit_type="function",
                            file_path=path,
                            parent=None,
                            lines=lines,
                            line_start=index + 1,
                            line_end=end + 1,
                            language=self.language,
                            node_type="function_declaration",
                            extra=extra,
                        )
                    )
                    index = end + 1
                    continue
            depth += clean.count("{") - clean.count("}")
            if depth < 0:
                depth = 0
            index += 1
        return units

    def _extract_invokes(
        self,
        path: str,
        lines: list[str],
        units: list[LPCodeUnit],
    ) -> list[LPDependency]:
        local_symbols = {
            unit.name
            for unit in units
            if unit.unit_type in {"class", "function"} and unit.name
        }
        import_bindings: dict[str, dict[str, str]] = {}
        import_ranges: list[tuple[int, int]] = []
        for unit in units:
            if unit.unit_type != "import":
                continue
            if unit.line_start is not None and unit.line_end is not None:
                import_ranges.append((unit.line_start, unit.line_end))
            for local, binding in unit.extra.get("bindings", {}).items():
                import_bindings[local] = dict(binding)

        dependencies: list[LPDependency] = []
        seen: set[tuple[str, str, int, str]] = set()
        for line_number, line in enumerate(lines, start=1):
            if any(start <= line_number <= end for start, end in import_ranges):
                continue
            clean = strip_string_literals(line).split("//", 1)[0]
            source_ref = self._source_reference_for_line(path, units, line_number)

            constructor_spans: list[tuple[int, int]] = []
            for match in _NEW_EXPRESSION_RE.finditer(clean):
                name = match.group("name")
                constructor_spans.append(match.span("name"))
                if name in local_symbols or name in import_bindings:
                    self._append_invoke_dependency(
                        dependencies,
                        seen,
                        source_ref,
                        name,
                        line_number,
                        "constructor",
                        import_bindings.get(name),
                    )

            for match in _MEMBER_CALL_RE.finditer(clean):
                qualifier = match.group("qualifier")
                name = match.group("name")
                binding = import_bindings.get(qualifier)
                if not binding or binding.get("kind") != "namespace":
                    continue
                namespace_binding = dict(binding)
                namespace_binding["imported"] = name
                namespace_binding["namespace"] = qualifier
                self._append_invoke_dependency(
                    dependencies,
                    seen,
                    source_ref,
                    name,
                    line_number,
                    "namespace_member",
                    namespace_binding,
                )

            for match in _DIRECT_CALL_RE.finditer(clean):
                name = match.group("name")
                if name in _CALL_KEYWORDS:
                    continue
                if any(start <= match.start("name") < end for start, end in constructor_spans):
                    continue
                if self._is_declaration_call_context(clean, match.start("name")):
                    continue
                if name in local_symbols or name in import_bindings:
                    self._append_invoke_dependency(
                        dependencies,
                        seen,
                        source_ref,
                        name,
                        line_number,
                        "function",
                        import_bindings.get(name),
                    )
        return dependencies

    def _append_invoke_dependency(
        self,
        dependencies: list[LPDependency],
        seen: set[tuple[str, str, int, str]],
        source_ref: str,
        name: str,
        line_number: int,
        call_kind: str,
        import_binding: dict[str, str] | None,
    ) -> None:
        key = (source_ref, name, line_number, call_kind)
        if key in seen:
            return
        seen.add(key)
        extra = {"language": self.language, "call_kind": call_kind, "local": name}
        dst = name
        if import_binding:
            extra.update(import_binding)
            dst = import_binding["module"]
        dependencies.append(
            LPDependency(
                src=source_ref,
                dst=dst,
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
            if unit.unit_type in {"function", "method", "class"}
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
        if prefix.endswith("new"):
            return True
        return _DECLARATION_PREFIX_RE.search(prefix) is not None

    def _module_from_import(self, statement: str) -> str | None:
        side_effect = _IMPORT_SIDE_EFFECT_RE.search(statement)
        if side_effect:
            return side_effect.group(1)
        import_from = _IMPORT_FROM_RE.search(statement)
        if import_from:
            return import_from.group(1)
        export_from = _EXPORT_FROM_RE.search(statement)
        if export_from:
            return export_from.group(1)
        return None

    def _syntax_error(self, source: str) -> str | None:
        backend_result = self.backend.validate_syntax(source)
        if backend_result is not None:
            valid, error = backend_result
            if not valid:
                return error
        return delimiter_syntax_error(source)
