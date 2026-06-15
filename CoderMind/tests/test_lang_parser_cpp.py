#!/usr/bin/env python3
"""Tests for the C++ language parser."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


CPP_SOURCE = textwrap.dedent(
    """\
    #include "model.hpp"
    #include <vector>

    class Widget {
    public:
        Widget();
        int value() const { return helper(); }
    private:
        int helper() const { return 1; }
    };

    int Widget::make() {
        Widget* widget = new Widget();
        return value();
    }

    int run() {
        return Widget::make();
    }
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name, unit.parent): unit for unit in result.units}


class TestCppParser:
    def test_extracts_includes_class_methods_and_functions(self):
        result = parse_file("src/model.cpp", CPP_SOURCE)
        assert result.file_path == "src/model.cpp"
        assert result.language == "cpp"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("import", "model.hpp", None) in units
        assert ("import", "vector", None) in units
        assert ("class", "Widget", None) in units
        assert ("method", "Widget", "Widget") in units
        assert ("method", "value", "Widget") in units
        assert ("method", "helper", "Widget") in units
        assert ("method", "make", "Widget") in units
        assert ("function", "run", None) in units

    def test_dependencies_include_constructor_static_and_direct_calls(self):
        result = parse_file("src/model.cpp", CPP_SOURCE)
        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        observed = {(dep.src, dep.symbol, dep.dst, dep.extra["call_kind"]) for dep in invokes}
        assert ("src/model.cpp:Widget.value", "helper", "helper", "direct") in observed
        assert ("src/model.cpp:Widget.make", "Widget", "Widget", "constructor") in observed
        assert ("src/model.cpp:Widget.make", "value", "value", "direct") in observed
        assert ("src/model.cpp:run", "make", "Widget", "static") in observed

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("src/model.cpp", CPP_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "cpp"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "cpp"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_struct_definition_is_class_like_unit(self):
        source = textwrap.dedent(
            """\
            struct Packet {
                int size() const { return 1; }
            };
            """
        )
        result = parse_file("include/packet.hpp", source)
        units = _unit_map(result)
        assert ("struct", "Packet", None) in units
        assert ("method", "size", "Packet") in units

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.cpp", "class Broken {\n")
        assert result.language == "cpp"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.cpp", "class Broken {\n")
        assert valid is False
        assert error is not None
