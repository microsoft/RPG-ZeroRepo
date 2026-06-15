#!/usr/bin/env python3
"""
Tests for PythonParser and ParsedFile parity with the existing AST semantics.
"""

import ast
import inspect
import os
import sys
import textwrap
from types import SimpleNamespace
from typing import Optional

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax
from lang_parser.python_parser import PythonParser
from rpg.code_unit import CodeSnippetBuilder, CodeUnit, ParsedFile


SAMPLE_CODE = textwrap.dedent(
    """\
    import os
    from pathlib import Path as P

    CONSTANT = 1
    typed_value: int = 2

    def top_function(x: int = 1) -> int:
        return x + CONSTANT

    async def fetch_data():
        return None

    class Example(Base):
        class_attr = "value"
        typed_attr: str = "typed"

        def __init__(self, value):
            self.value = value

        async def run(self):
            return self.value
    """
)


def _extract_assignment_name(node) -> Optional[str]:
    if isinstance(node, ast.Assign):
        if node.targets and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return node.target.id
    return None


def _legacy_units(code: str, file_path: str) -> list[CodeUnit]:
    tree = ast.parse(code)
    units: list[CodeUnit] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            units.append(CodeUnit(ast.unparse(node).strip(), node, "import", file_path))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(CodeUnit(node.name, node, "function", file_path))
        elif isinstance(node, ast.ClassDef):
            units.append(CodeUnit(node.name, node, "class", file_path))
            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    units.append(CodeUnit(sub_node.name, sub_node, "method", file_path, parent=node.name))
                elif isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
                    units.append(
                        CodeUnit(
                            _extract_assignment_name(sub_node),
                            sub_node,
                            "assignment",
                            file_path,
                            parent=node.name,
                        )
                    )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            units.append(CodeUnit(_extract_assignment_name(node), node, "assignment", file_path))
    return units


def _unit_summary(units):
    return [
        (
            unit.name,
            unit.unit_type,
            unit.parent,
            unit.lineno,
            unit.end_lineno,
            unit.unparse(),
        )
        for unit in units
    ]


def _lp_summary(units):
    return [
        (
            unit.name,
            unit.unit_type,
            unit.parent,
            unit.line_start,
            unit.line_end,
            unit.code,
            unit.language,
        )
        for unit in units
    ]


class TestPythonParserResult:
    def test_valid_python_parser_result(self):
        result = PythonParser().parse_file("pkg/sample.py", SAMPLE_CODE)
        assert result.file_path == "pkg/sample.py"
        assert result.language == "python"
        assert result.syntax_error is None

        unit_types = [unit.unit_type for unit in result.units]
        assert unit_types == [
            "import",
            "import",
            "assignment",
            "assignment",
            "function",
            "function",
            "class",
            "assignment",
            "assignment",
            "method",
            "method",
        ]
        assert all(unit.language == "python" for unit in result.units)
        assert all(unit.line_start is not None and unit.line_end is not None for unit in result.units)
        assert result.units[0].code == "import os"
        assert result.units[1].code == "from pathlib import Path as P"
        assert result.units[4].code.startswith("def top_function")
        assert result.units[6].code.startswith("class Example")
        assert len(result.dependencies) == 2
        assert [dependency.relation for dependency in result.dependencies] == ["imports", "imports"]

    def test_public_parse_file_matches_parser(self):
        direct = PythonParser().parse_file("pkg/sample.py", SAMPLE_CODE)
        public = parse_file("pkg/sample.py", SAMPLE_CODE)
        assert _lp_summary(public.units) == _lp_summary(direct.units)

    def test_invalid_python_parser_result(self):
        invalid_code = "def broken(\n"
        result = PythonParser().parse_file("bad.py", invalid_code)
        assert result.file_path == "bad.py"
        assert result.language == "python"
        assert result.syntax_error is not None
        assert result.units == []
        assert result.dependencies == []
        assert validate_syntax("bad.py", invalid_code)[0] is False

    def test_validate_syntax_matches_ast_parse(self):
        assert PythonParser().validate_syntax("ok.py", SAMPLE_CODE) == (True, None)
        invalid_code = "def broken(\n"
        parser_valid, parser_error = PythonParser().validate_syntax("bad.py", invalid_code)
        try:
            ast.parse(invalid_code)
        except SyntaxError as exc:
            ast_error = str(exc)
        else:
            ast_error = None
        assert parser_valid is False
        assert parser_error == ast_error


class TestParsedFileParity:
    def test_constructor_signature_is_unchanged(self):
        signature = inspect.signature(ParsedFile.__init__)
        assert list(signature.parameters) == ["self", "code", "file_path"]
        assert signature.parameters["code"].annotation is str
        assert signature.parameters["file_path"].annotation is str

    def test_parsed_file_units_match_legacy_extraction(self):
        parsed = ParsedFile(SAMPLE_CODE, "pkg/sample.py")
        legacy = _legacy_units(SAMPLE_CODE, "pkg/sample.py")
        assert parsed.has_error() is False
        assert isinstance(parsed.tree, ast.Module)
        assert _unit_summary(parsed.units) == _unit_summary(legacy)

    def test_parsed_file_queries_match_legacy_extraction(self):
        parsed = ParsedFile(SAMPLE_CODE, "pkg/sample.py")
        legacy = _legacy_units(SAMPLE_CODE, "pkg/sample.py")
        legacy_by_name = {unit.name: unit for unit in legacy if unit.name is not None}

        for name in ["CONSTANT", "typed_value", "top_function", "fetch_data", "Example", "__init__", "run"]:
            parsed_unit = parsed.get_unit_by_name(name)
            assert parsed_unit is not None
            assert parsed_unit.unit_type == legacy_by_name[name].unit_type
            assert parsed_unit.parent == legacy_by_name[name].parent

        assert [unit.name for unit in parsed.get_units_by_type("method")] == ["__init__", "run"]
        assert [unit.name for unit in parsed.get_units_by_type("assignment")] == [
            "CONSTANT",
            "typed_value",
            "class_attr",
            "typed_attr",
        ]

    def test_snippet_and_count_line_behavior_match_legacy_extraction(self):
        path = "pkg/sample.py"
        parsed = ParsedFile(SAMPLE_CODE, path)
        legacy_units = _legacy_units(SAMPLE_CODE, path)

        parsed_function = parsed.get_unit_by_name("top_function")
        legacy_function = next(unit for unit in legacy_units if unit.name == "top_function")
        assert parsed_function.count_lines(original=True, return_code=True) == legacy_function.count_lines(
            original=True,
            return_code=True,
        )
        assert parsed_function.count_lines(original=False, return_code=True) == legacy_function.count_lines(
            original=False,
            return_code=True,
        )

        parsed_builder = CodeSnippetBuilder({path: SAMPLE_CODE}, {path: parsed})
        legacy_builder = CodeSnippetBuilder({path: SAMPLE_CODE}, {path: SimpleNamespace(units=legacy_units)})
        assert parsed_builder.generate_code_snippet(SAMPLE_CODE, parsed.units) == legacy_builder.generate_code_snippet(
            SAMPLE_CODE,
            legacy_units,
        )

    def test_invalid_python_matches_existing_error_behavior(self):
        parsed = ParsedFile("def broken(\n", "bad.py")
        assert parsed.has_error() is True
        assert isinstance(parsed.error, SyntaxError)
        assert isinstance(parsed.tree, ast.Module)
        assert parsed.tree.body == []
        assert parsed.units == []
