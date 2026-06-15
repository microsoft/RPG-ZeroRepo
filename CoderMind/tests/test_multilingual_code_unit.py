#!/usr/bin/env python3
"""Tests for multilingual ParsedFile and CodeSnippetBuilder behavior."""

import os
import sys
from unittest.mock import patch

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from rpg.code_unit import CodeSnippetBuilder, ParsedFile


def test_parsed_file_handles_go_without_ast_parse_crash():
    code = "package main\n\nimport \"fmt\"\n\ntype Server struct {}\nfunc (s *Server) Handle() { fmt.Println(\"ok\") }\n"
    parsed = ParsedFile(code, "main.go")

    assert parsed.has_error() is False
    # Go struct is normalised to ``class`` so semantic_parsing.py's
    # class-vs-function grouping picks it up; the original kind is kept
    # in ``extra['lp_kind']`` for callers that need the raw taxonomy.
    assert [(unit.unit_type, unit.name, unit.parent) for unit in parsed.units] == [
        ("package", "main", None),
        ("import", "fmt", None),
        ("class", "Server", None),
        ("method", "Handle", "Server"),
    ]
    struct_unit = parsed.get_unit_by_name("Server")
    assert struct_unit.extra["lp_kind"] == "struct"
    method = parsed.get_unit_by_name("Handle")
    assert method.lineno == 6
    assert method.end_lineno == 6
    assert method.extra["language"] == "go"


def test_snippet_builder_uses_go_fence_and_skips_ast_parse_for_go():
    path = "main.go"
    code = "package main\n\nimport \"fmt\"\n\ntype Server struct {}\nfunc (s *Server) Handle() { fmt.Println(\"ok\") }\n"
    parsed = ParsedFile(code, path)
    builder = CodeSnippetBuilder({path: code}, {path: parsed})
    units = [unit for unit in parsed.units if unit.unit_type in {"class", "method"}]

    with patch("ast.parse", side_effect=AssertionError("ast.parse should not run for Go")):
        snippet = builder.build(units)

    assert snippet.startswith("```go")
    assert "type Server struct" in snippet
    assert "func (s *Server) Handle" in snippet


def test_snippet_builder_uses_typescript_and_javascript_fences():
    ts_path = "src/app.ts"
    ts_code = "import { x } from './x';\nexport function run(): number { return x; }\n"
    js_path = "src/app.jsx"
    js_code = "import React from 'react';\nexport function View() { return <div />; }\n"
    ts_parsed = ParsedFile(ts_code, ts_path)
    js_parsed = ParsedFile(js_code, js_path)
    builder = CodeSnippetBuilder(
        {ts_path: ts_code, js_path: js_code},
        {ts_path: ts_parsed, js_path: js_parsed},
    )

    assert builder.build(ts_parsed.units).startswith("```typescript")
    assert builder.build(js_parsed.units).startswith("```javascript")


def test_python_snippet_behavior_still_uses_python_fence():
    path = "pkg/mod.py"
    code = "import os\n\ndef helper():\n    return os.getcwd()\n"
    parsed = ParsedFile(code, path)
    builder = CodeSnippetBuilder({path: code}, {path: parsed})
    snippet = builder.build(parsed.units)

    assert snippet.startswith("```python")
    assert "def helper" in snippet
