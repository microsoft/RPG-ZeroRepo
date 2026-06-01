from __future__ import annotations

import ast
from typing import Optional

from .base import BaseLanguageParser
from .models import LPCodeUnit, LPDependency, LPFileResult


class PythonParser(BaseLanguageParser):
    language = "python"

    def parse_file(self, path: str, source: str) -> LPFileResult:
        result, _, _ = self.parse_file_with_ast(path, source)
        return result

    def parse_file_with_ast(
        self,
        path: str,
        source: str,
    ) -> tuple[LPFileResult, ast.Module, SyntaxError | None]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            empty_tree = ast.Module(body=[], type_ignores=[])
            return (
                LPFileResult(
                    file_path=path,
                    language=self.language,
                    units=[],
                    dependencies=[],
                    syntax_error=str(exc),
                ),
                empty_tree,
                exc,
            )

        return self._result_from_tree(path, source, tree), tree, None

    def validate_syntax(self, path: str, source: str) -> tuple[bool, str | None]:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return False, str(exc)
        return True, None

    def _result_from_tree(self, path: str, source: str, tree: ast.Module) -> LPFileResult:
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                units.append(self._make_unit(ast.unparse(node).strip(), "import", path, None, source, node))
                dependencies.extend(self._dependencies_from_import(path, node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(self._make_unit(node.name, "function", path, None, source, node))
            elif isinstance(node, ast.ClassDef):
                units.append(self._make_unit(node.name, "class", path, None, source, node))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        units.append(self._make_unit(child.name, "method", path, node.name, source, child))
                    elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                        units.append(
                            self._make_unit(
                                self._extract_assignment_name(child),
                                "assignment",
                                path,
                                node.name,
                                source,
                                child,
                            )
                        )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                units.append(
                    self._make_unit(
                        self._extract_assignment_name(node),
                        "assignment",
                        path,
                        None,
                        source,
                        node,
                    )
                )

        return LPFileResult(
            file_path=path,
            language=self.language,
            units=units,
            dependencies=dependencies,
            syntax_error=None,
        )

    def _make_unit(
        self,
        name: str | None,
        unit_type: str,
        path: str,
        parent: str | None,
        source: str,
        node: ast.AST,
    ) -> LPCodeUnit:
        line_start = getattr(node, "lineno", None)
        line_end = getattr(node, "end_lineno", line_start)
        return LPCodeUnit(
            name=name,
            unit_type=unit_type,
            file_path=path,
            parent=parent,
            line_start=line_start,
            line_end=line_end,
            code=self._source_for_node(source, node, line_start, line_end),
            language=self.language,
            extra={"ast_node": node, "node_type": type(node).__name__},
        )

    def _source_for_node(
        self,
        source: str,
        node: ast.AST,
        line_start: Optional[int],
        line_end: Optional[int],
    ) -> str:
        if line_start is not None and line_end is not None:
            lines = source.splitlines()
            if 1 <= line_start <= line_end <= len(lines):
                return "\n".join(lines[line_start - 1:line_end])
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ""

    def _dependencies_from_import(self, path: str, node: ast.Import | ast.ImportFrom) -> list[LPDependency]:
        dependencies: list[LPDependency] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependencies.append(
                    LPDependency(
                        src=path,
                        dst=alias.name,
                        relation="imports",
                        symbol=alias.asname or alias.name,
                        line=getattr(node, "lineno", None),
                        confidence="unresolved",
                        extra={"module": alias.name, "alias": alias.asname},
                    )
                )
        else:
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                dependencies.append(
                    LPDependency(
                        src=path,
                        dst=module or None,
                        relation="imports",
                        symbol=alias.asname or alias.name,
                        line=getattr(node, "lineno", None),
                        confidence="unresolved",
                        extra={"module": module, "imported": alias.name, "alias": alias.asname},
                    )
                )
        return dependencies

    def _extract_assignment_name(self, node: ast.Assign | ast.AnnAssign) -> str | None:
        if isinstance(node, ast.Assign):
            if node.targets and isinstance(node.targets[0], ast.Name):
                return node.targets[0].id
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                return node.target.id
        return None
