#!/usr/bin/env python3
"""Tests for the C language parser."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


C_SOURCE = textwrap.dedent(
    """\
    #include "math_utils.h"
    #include <string.h>

    struct Point { int x; int y; };

    static int add(int a, int b) {
        return a + b;
    }

    int run(void) {
        return add(1, 2);
    }
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name): unit for unit in result.units}


class TestCParser:
    def test_extracts_includes_struct_and_functions(self):
        result = parse_file("src/math.c", C_SOURCE)
        assert result.file_path == "src/math.c"
        assert result.language == "c"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("import", "math_utils.h") in units
        assert ("import", "string.h") in units
        assert units[("import", "math_utils.h")].extra["include_style"] == "quote"
        assert units[("import", "string.h")].extra["include_style"] == "angle"
        assert ("struct", "Point") in units
        assert ("function", "add") in units
        assert ("function", "run") in units

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("src/math.c", C_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "c"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "c"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_dependencies_are_recorded_for_includes_and_invokes(self):
        result = parse_file("src/math.c", C_SOURCE)
        imports = [dep for dep in result.dependencies if dep.relation == "imports"]
        assert [(dep.dst, dep.extra["include_style"]) for dep in imports] == [
            ("math_utils.h", "quote"),
            ("string.h", "angle"),
        ]

        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        assert [(dep.src, dep.symbol, dep.dst, dep.extra["call_kind"]) for dep in invokes] == [
            ("src/math.c:run", "add", "add", "direct"),
        ]

    def test_builtin_calls_are_not_emitted_as_invokes(self):
        source = textwrap.dedent(
            """\
            #include <stdio.h>

            int run(void) {
                printf("hello");
                return 0;
            }
            """
        )
        result = parse_file("src/main.c", source)
        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        assert invokes == []

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.c", "int broken(\n")
        assert result.language == "c"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.c", "int broken(\n")
        assert valid is False
        assert error is not None
