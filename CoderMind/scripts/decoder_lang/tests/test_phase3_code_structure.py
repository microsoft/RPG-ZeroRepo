"""Tests for Phase 3 of decoder multi-language: code-structure
methods added to ``PythonBackend`` (``list_code_units``,
``format_signature``, ``list_imports``, ``find_main_block_lineno``).

Each test cross-checks against the stdlib ``ast`` behaviour the
caller in ``func_design/`` currently relies on, so the upcoming
mechanical refactor in Phase 3b can ride on top with confidence.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

# Make ``scripts/`` importable for direct invocation.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import GoBackend, PythonBackend, get_backend  # noqa: E402


# Sample Python source that exercises top-level functions, classes
# with methods, nested functions, async, and decorators.
_SAMPLE_SRC = '''\
"""Module docstring."""
import os
from typing import Optional, List
from .util import helper as h

CONST = 1

def top_level_func(a: int, b: str = "x") -> bool:
    """Top-level."""
    return True

async def top_level_async(data: bytes) -> None:
    pass

class Parser:
    """Parser class."""

    def __init__(self, path: str) -> None:
        self.path = path

    def parse(self, data: bytes, *, strict: bool = False) -> List[int]:
        return []

    async def parse_async(self, x, y, z, w, extra) -> None:
        pass

    def _private(self):
        pass

def outer():
    def inner():
        pass
    return inner

if __name__ == "__main__":
    main()
'''


class ListCodeUnitsTests(unittest.TestCase):
    """``list_code_units`` walks all nesting; matches ast.walk semantics
    used by ``func_design/interface_agent``."""

    def setUp(self) -> None:
        self.backend: PythonBackend = get_backend("python")  # type: ignore
        self.units = self.backend.list_code_units(_SAMPLE_SRC, "demo.py")

    def test_returns_units_for_every_decl(self) -> None:
        names = [u.name for u in self.units]
        # Order is ast.walk's BFS; we only assert membership.
        for expected in (
            "top_level_func", "top_level_async", "Parser",
            "__init__", "parse", "parse_async", "_private",
            "outer", "inner",
        ):
            with self.subTest(decl=expected):
                self.assertIn(expected, names)

    def test_unit_types_assigned_correctly(self) -> None:
        by_name = {u.name: u for u in self.units}
        self.assertEqual(by_name["top_level_func"].unit_type, "function")
        self.assertEqual(by_name["top_level_async"].unit_type, "function")
        self.assertEqual(by_name["Parser"].unit_type, "class")
        self.assertEqual(by_name["__init__"].unit_type, "method")
        self.assertEqual(by_name["parse"].unit_type, "method")
        self.assertEqual(by_name["parse_async"].unit_type, "method")
        # Nested function is NOT a method (its parent is a function,
        # not a class) — matches the original ast.walk + isinstance
        # logic where ``outer`` and ``inner`` are both "function".
        self.assertEqual(by_name["outer"].unit_type, "function")
        self.assertEqual(by_name["inner"].unit_type, "function")

    def test_parent_populated_for_methods(self) -> None:
        by_name = {u.name: u for u in self.units}
        self.assertEqual(by_name["__init__"].parent, "Parser")
        self.assertEqual(by_name["parse"].parent, "Parser")
        # Top-level decls have no parent.
        self.assertIsNone(by_name["top_level_func"].parent)
        # Nested function has no class parent.
        self.assertIsNone(by_name["inner"].parent)

    def test_line_numbers_populated(self) -> None:
        by_name = {u.name: u for u in self.units}
        for n in ("top_level_func", "Parser", "parse"):
            with self.subTest(decl=n):
                self.assertIsNotNone(by_name[n].line_start)
                self.assertIsNotNone(by_name[n].line_end)
                self.assertGreaterEqual(by_name[n].line_end, by_name[n].line_start)

    def test_ast_node_escape_hatch_preserved(self) -> None:
        # PythonBackend stuffs the raw ast node into extra["ast_node"]
        # so format_signature can use ast.unparse without re-parsing.
        by_name = {u.name: u for u in self.units}
        parse_unit = by_name["parse"]
        node = parse_unit.extra.get("ast_node")
        self.assertIsInstance(node, ast.FunctionDef)
        self.assertEqual(node.name, "parse")

    def test_empty_on_syntax_error(self) -> None:
        # Callers in func_design tolerate empty results; backend must
        # not raise even on garbled source.
        self.assertEqual(self.backend.list_code_units("def f(:\n  pass\n"), [])

    def test_file_path_propagated(self) -> None:
        # File path on every unit matches the path argument so callers
        # can index by file without re-passing it.
        for u in self.units:
            self.assertEqual(u.file_path, "demo.py")


class FormatSignatureTests(unittest.TestCase):
    """Signature formatting matches the historical
    ``GlobalInterfaceRegistry._format_func_signature`` output."""

    def setUp(self) -> None:
        self.backend: PythonBackend = get_backend("python")  # type: ignore
        self.units = self.backend.list_code_units(_SAMPLE_SRC, "demo.py")
        self.by_name = {u.name: u for u in self.units}

    def test_simple_function(self) -> None:
        self.assertEqual(
            self.backend.format_signature(self.by_name["top_level_func"]),
            "top_level_func(a: int, b: str) -> bool",
        )

    def test_async_function(self) -> None:
        self.assertEqual(
            self.backend.format_signature(self.by_name["top_level_async"]),
            "top_level_async(data: bytes) -> None",
        )

    def test_method_skips_self(self) -> None:
        # ``self`` is excluded from rendered params (per historical
        # _format_func_signature behaviour).
        self.assertEqual(
            self.backend.format_signature(self.by_name["__init__"]),
            "__init__(path: str) -> None",
        )

    def test_method_with_keyword_only(self) -> None:
        # Note: the original helper only walks node.args.args (no
        # kwonly handling); ``strict`` is kwonly so it does NOT appear.
        # PythonBackend preserves this exact behaviour for parity.
        sig = self.backend.format_signature(self.by_name["parse"])
        self.assertIn("data: bytes", sig)
        self.assertNotIn("strict", sig)
        self.assertTrue(sig.endswith(" -> List[int]"))

    def test_truncation_when_more_than_4_params(self) -> None:
        # parse_async has 5 positional params after dropping ``self``.
        sig = self.backend.format_signature(self.by_name["parse_async"])
        self.assertIn(", ...", sig)
        self.assertTrue(sig.endswith(" -> None"))

    def test_non_function_returns_name(self) -> None:
        self.assertEqual(
            self.backend.format_signature(self.by_name["Parser"]),
            "Parser",
        )

    def test_none_safe(self) -> None:
        self.assertEqual(self.backend.format_signature(None), "")


class ListImportsTests(unittest.TestCase):
    """``list_imports`` matches lang_parser's dependency shape."""

    def setUp(self) -> None:
        self.backend: PythonBackend = get_backend("python")  # type: ignore
        self.deps = self.backend.list_imports(_SAMPLE_SRC, "demo.py")

    def test_all_imports_emitted(self) -> None:
        # 3 statements → 1 + 2 + 1 = 4 entries (typing imports List + Optional).
        modules = [d.extra.get("module") for d in self.deps]
        self.assertIn("os", modules)
        self.assertIn("typing", modules)
        self.assertIn(".util", modules)

    def test_relation_is_imports(self) -> None:
        for dep in self.deps:
            self.assertEqual(dep.relation, "imports")

    def test_alias_recorded(self) -> None:
        # ``from .util import helper as h`` → alias=h.
        util_deps = [d for d in self.deps if d.extra.get("module") == ".util"]
        self.assertEqual(len(util_deps), 1)
        self.assertEqual(util_deps[0].extra.get("alias"), "h")
        self.assertEqual(util_deps[0].extra.get("imported"), "helper")

    def test_empty_on_syntax_error(self) -> None:
        self.assertEqual(self.backend.list_imports("import"), [])


class FindMainBlockLinenoTests(unittest.TestCase):
    """``find_main_block_lineno`` is the Python-only hook
    ``interface_review`` will call (others get None via getattr)."""

    def setUp(self) -> None:
        self.backend: PythonBackend = get_backend("python")  # type: ignore

    def test_finds_main_block(self) -> None:
        ln = self.backend.find_main_block_lineno(_SAMPLE_SRC)
        # The ``if __name__ == "__main__":`` line in the fixture is the
        # 2nd-to-last line. We don't pin it absolutely — just check it
        # points at an ``if`` line in the source.
        self.assertIsNotNone(ln)
        line_text = _SAMPLE_SRC.splitlines()[ln - 1]
        self.assertIn("__name__", line_text)

    def test_none_when_absent(self) -> None:
        src = "def foo():\n    return 1\n"
        self.assertIsNone(self.backend.find_main_block_lineno(src))

    def test_none_on_syntax_error(self) -> None:
        self.assertIsNone(self.backend.find_main_block_lineno("def f(:"))

    def test_not_in_protocol(self) -> None:
        # Documented as a Python-only hook; non-Python backends don't
        # expose it. Feature detection via getattr is the contract.
        self.assertFalse(hasattr(get_backend("go"), "find_main_block_lineno"))


class GoBackendStubsTests(unittest.TestCase):
    """Phase 3 new methods on GoBackend still raise until Phase 4."""

    def test_list_code_units_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            get_backend("go").list_code_units("package main")

    def test_format_signature_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            get_backend("go").format_signature(None)

    def test_list_imports_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            get_backend("go").list_imports("package main")


if __name__ == "__main__":
    unittest.main()
