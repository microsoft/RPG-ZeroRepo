#!/usr/bin/env python3
"""Tests for the TypeScript language parser."""

import multiprocessing as mp
import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


def _strip_string_literals_worker(line, result_queue):
    from lang_parser.extractors.fallback import strip_string_literals

    result_queue.put(strip_string_literals(line))


TS_SOURCE = textwrap.dedent(
    """\
    import { User } from "./models";

    export class Greeter {
      constructor(private user: User) {}

      greet(name: string): string {
        return `hello ${name}`;
      }
    }

    export function makeGreeter(user: User): Greeter {
      return new Greeter(user);
    }

    export const formatName = (name: string): string => {
      return name.trim();
    };
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name): unit for unit in result.units}


class TestTypeScriptParser:
    def test_extracts_imports_classes_functions_and_methods(self):
        result = parse_file("src/greeter.ts", TS_SOURCE)
        assert result.file_path == "src/greeter.ts"
        assert result.language == "typescript"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("import", "./models") in units
        assert ("class", "Greeter") in units
        assert ("method", "constructor") in units
        assert ("method", "greet") in units
        assert ("function", "makeGreeter") in units
        assert ("function", "formatName") in units
        assert units[("method", "greet")].parent == "Greeter"

    def test_tsx_extension_uses_typescript_language(self):
        result = parse_file("src/component.tsx", "import React from 'react';\nexport function View() { return <div />; }\n")
        assert result.language == "typescript"
        assert any(unit.unit_type == "function" and unit.name == "View" for unit in result.units)

    def test_consecutive_semicolonless_imports_stay_separate(self):
        source = textwrap.dedent(
            """\
            import { A } from "./a"
            import B from "./b"
            export { C } from "./c"
            """
        )
        result = parse_file("src/app.ts", source)

        import_units = [unit for unit in result.units if unit.unit_type == "import"]
        assert [unit.extra["module"] for unit in import_units] == ["./a", "./b", "./c"]
        assert [dep.dst for dep in result.dependencies if dep.relation == "imports"] == ["./a", "./b", "./c"]

    def test_multiline_import_stays_single_dependency(self):
        source = textwrap.dedent(
            """\
            import {
              A,
              B,
            } from "./types"
            import { C } from "./c"
            """
        )
        result = parse_file("src/app.ts", source)

        import_units = [unit for unit in result.units if unit.unit_type == "import"]
        assert [unit.extra["module"] for unit in import_units] == ["./types", "./c"]
        assert import_units[0].line_start == 1
        assert import_units[0].line_end == 4

    def test_invokes_include_imported_function_and_constructor_calls(self):
        source = textwrap.dedent(
            """\
            import { getDebugOption } from "./debug"
            import { ChromeRemote } from "./remote"

            export function boot() {
              getDebugOption();
              return new ChromeRemote();
            }
            """
        )
        result = parse_file("src/app.ts", source)

        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        assert [(dep.symbol, dep.extra["module"], dep.extra["call_kind"]) for dep in invokes] == [
            ("getDebugOption", "./debug", "function"),
            ("ChromeRemote", "./remote", "constructor"),
        ]

    def test_default_exported_class_and_function_units_are_marked(self):
        class_result = parse_file("src/local.ts", "export default class ActualClass {}\n")
        class_units = _unit_map(class_result)
        assert ("class", "ActualClass") in class_units
        assert class_units[("class", "ActualClass")].extra["export_default"] is True

        function_result = parse_file("src/factory.ts", "export default function createActual() { return true; }\n")
        function_units = _unit_map(function_result)
        assert ("function", "createActual") in function_units
        assert function_units[("function", "createActual")].extra["export_default"] is True

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("src/greeter.ts", TS_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "typescript"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "typescript"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.ts", "export function broken(\n")
        assert result.language == "typescript"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.ts", "export function broken(\n")
        assert valid is False
        assert error is not None

    def test_comment_with_unterminated_quote_and_many_escapes_does_not_hang(self):
        zod_like_line = (
            "// const emailRegex = /^([!#\\$%&'"
            + ("\\d" * 20)
            + "_`{|}~]/"
        )
        result_queue = mp.Queue()
        process = mp.Process(
            target=_strip_string_literals_worker,
            args=(zod_like_line, result_queue),
        )

        process.start()
        process.join(3)

        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError(
                "strip_string_literals hung on zod-like escaped regex comment"
            )

        assert process.exitcode == 0
        assert result_queue.get_nowait() == zod_like_line
