"""Tests for unit-name classification (callable vs type-like).

Covers the shared :func:`decoder_lang.unit_kind.classify_unit_kind`
helper and the ``unit_kind`` / ``is_callable_unit`` backend methods.
Classification feeds orphan detection: callable units are subject to
the "no incoming edge => dead code" heuristic; type-like units are
exempt (a data structure legitimately has no incoming invocation edge).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import get_backend, list_backends  # noqa: E402
from decoder_lang.unit_kind import classify_unit_kind  # noqa: E402


class TestClassifyUnitKind(unittest.TestCase):
    def test_callable_prefixes(self):
        for name in ("function f", "method m", "class C", "fn g", "func h"):
            self.assertEqual(classify_unit_kind(name), "callable", name)

    def test_type_prefixes(self):
        for name in (
            "struct S", "enum E", "interface I", "trait T",
            "type Alias", "union U", "typedef Td",
        ):
            self.assertEqual(classify_unit_kind(name), "type", name)

    def test_unknown(self):
        for name in ("", "weird X", "constant K", "noprefix"):
            self.assertEqual(classify_unit_kind(name), "unknown", name)

    def test_case_insensitive(self):
        self.assertEqual(classify_unit_kind("Function F"), "callable")
        self.assertEqual(classify_unit_kind("STRUCT S"), "type")

    def test_custom_prefix_sets(self):
        self.assertEqual(
            classify_unit_kind(
                "widget W",
                callable_prefixes=frozenset({"widget"}),
                type_prefixes=frozenset(),
            ),
            "callable",
        )


class TestBackendUnitKind(unittest.TestCase):
    """Every registered backend exposes unit_kind / is_callable_unit."""

    def test_all_backends_have_methods(self):
        for name in list_backends():
            b = get_backend(name)
            self.assertTrue(hasattr(b, "unit_kind"), name)
            self.assertTrue(hasattr(b, "is_callable_unit"), name)

    def test_python_class_is_callable(self):
        # Decision: Python class stays callable (zero regression — the
        # encoder records Foo() instantiation as an invocation edge).
        b = get_backend("python")
        self.assertEqual(b.unit_kind("class JsonTodoStore"), "callable")
        self.assertTrue(b.is_callable_unit("class JsonTodoStore"))

    def test_go_struct_is_type(self):
        # The Go false-positive case: struct Store / struct PageData must
        # be exempt from orphan detection.
        b = get_backend("go")
        self.assertEqual(b.unit_kind("struct Store"), "type")
        self.assertFalse(b.is_callable_unit("struct Store"))
        self.assertTrue(b.is_callable_unit("function main"))
        self.assertTrue(b.is_callable_unit("method ServeHTTP"))

    def test_rust_struct_enum_are_types(self):
        b = get_backend("rust")
        self.assertFalse(b.is_callable_unit("struct Config"))
        self.assertFalse(b.is_callable_unit("enum Command"))
        self.assertTrue(b.is_callable_unit("fn main"))

    def test_typescript_interface_is_type(self):
        b = get_backend("typescript")
        self.assertFalse(b.is_callable_unit("interface Todo"))
        self.assertTrue(b.is_callable_unit("function render"))

    def test_cpp_class_callable_struct_type(self):
        b = get_backend("cpp")
        self.assertTrue(b.is_callable_unit("class Evaluator"))
        self.assertFalse(b.is_callable_unit("struct Token"))


if __name__ == "__main__":
    unittest.main()
