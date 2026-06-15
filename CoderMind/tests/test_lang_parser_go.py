#!/usr/bin/env python3
"""Tests for the Go language parser."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


GO_SOURCE = textwrap.dedent(
    """\
    package server

    import (
        "fmt"
        nethttp "net/http"
    )

    type Server struct {
        Name string
    }

    func NewServer(name string) *Server {
        return &Server{Name: name}
    }

    func (s *Server) Handle() {
        fmt.Println(s.Name)
    }
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name): unit for unit in result.units}


class TestGoParser:
    def test_extracts_package_import_function_struct_and_receiver_method(self):
        result = parse_file("internal/server/main.go", GO_SOURCE)
        assert result.file_path == "internal/server/main.go"
        assert result.language == "go"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("package", "server") in units
        assert ("import", "fmt") in units
        assert ("import", "net/http") in units
        assert ("struct", "Server") in units
        assert ("function", "NewServer") in units
        assert ("method", "Handle") in units
        assert units[("method", "Handle")].parent == "Server"

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("main.go", GO_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "go"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "go"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_dependencies_are_recorded_for_imports(self):
        result = parse_file("main.go", GO_SOURCE)
        imports = [dep for dep in result.dependencies if dep.relation == "imports"]
        assert [dep.dst for dep in imports] == ["fmt", "net/http"]

    def test_invokes_include_same_package_direct_and_imported_selector_calls(self):
        source = textwrap.dedent(
            """\
            package app

            import "github.com/example/project/constraints"

            func Run() {
                AllC()
                constraints.Check()
            }

            func AllC() bool {
                return true
            }
            """
        )
        result = parse_file("cmd/app/app.go", source)

        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        assert [(dep.symbol, dep.dst, dep.extra["call_kind"]) for dep in invokes] == [
            ("AllC", "AllC", "direct"),
            ("Check", "github.com/example/project/constraints", "selector"),
        ]
        assert invokes[1].extra["qualifier"] == "constraints"
        assert invokes[1].extra["module"] == "github.com/example/project/constraints"

    def test_generic_functions_and_receiver_methods_are_parsed(self):
        source = textwrap.dedent(
            """\
            package collections

            func All[T any](items []T) bool {
                return AllC(items)
            }

            func AllC[T any](items []T) bool {
                return true
            }

            func (s Set[T]) Add(value T) {}
            func (s *Set[T]) Remove(value T) {}
            func (s Set[T]) Map[U any](f func(T) U) []U { return nil }
            """
        )
        result = parse_file("collections/set.go", source)

        units = _unit_map(result)
        assert ("function", "All") in units
        assert ("function", "AllC") in units
        assert ("method", "Add") in units
        assert ("method", "Remove") in units
        assert ("method", "Map") in units
        assert units[("method", "Add")].parent == "Set"
        assert units[("method", "Remove")].parent == "Set"
        assert units[("method", "Map")].parent == "Set"

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.go", "package main\nfunc broken(\n")
        assert result.language == "go"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.go", "package main\nfunc broken(\n")
        assert valid is False
        assert error is not None
