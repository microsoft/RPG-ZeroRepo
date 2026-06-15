#!/usr/bin/env python3
"""Tests for the JavaScript language parser."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


JS_SOURCE = textwrap.dedent(
    """\
    import fs from "fs";

    export class Loader {
      load(path) {
        return fs.readFileSync(path, "utf8");
      }

      static create() {
        return new Loader();
      }
    }

    export function makeLoader() {
      return Loader.create();
    }

    const normalizePath = (path) => path.trim();
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name): unit for unit in result.units}


class TestJavaScriptParser:
    def test_extracts_imports_classes_functions_and_methods(self):
        result = parse_file("src/loader.js", JS_SOURCE)
        assert result.file_path == "src/loader.js"
        assert result.language == "javascript"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("import", "fs") in units
        assert ("class", "Loader") in units
        assert ("method", "load") in units
        assert ("method", "create") in units
        assert ("function", "makeLoader") in units
        assert ("function", "normalizePath") in units
        assert units[("method", "load")].parent == "Loader"

    def test_jsx_extension_uses_javascript_language(self):
        result = parse_file("src/view.jsx", "import React from 'react';\nexport function View() { return <div />; }\n")
        assert result.language == "javascript"
        assert any(unit.unit_type == "function" and unit.name == "View" for unit in result.units)

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("src/loader.js", JS_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "javascript"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "javascript"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.js", "export function broken(\n")
        assert result.language == "javascript"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.js", "export function broken(\n")
        assert valid is False
        assert error is not None
