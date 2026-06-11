"""Tests for PythonBackend code-structure helpers.

The suite covers ``list_code_units``, ``format_signature``,
``list_imports``, ``list_inheritance``, and ``find_main_block_lineno``.
Assertions focus on the shapes consumed by ``func_design`` and
code-generation prompts.
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
        # Nested function is NOT a method: its parent is a function,
        # not a class. Both ``outer`` and ``inner`` are functions.
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
    """Signature formatting matches interface-registry expectations."""

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
        # ``self`` is excluded from rendered params.
        self.assertEqual(
            self.backend.format_signature(self.by_name["__init__"]),
            "__init__(path: str) -> None",
        )

    def test_method_with_keyword_only(self) -> None:
        # Keyword-only args are omitted from the rendered prompt
        # signature, so ``strict`` does not appear.
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


class ListInheritanceTests(unittest.TestCase):
    """``list_inheritance`` yields uniform ``inherits`` edges per language."""

    def test_python_derives_edges_from_class_bases(self) -> None:
        backend = get_backend("python")
        code = (
            "class Base:\n    pass\n\n"
            "class Mixin:\n    pass\n\n"
            "class Child(Base, Mixin):\n    pass\n"
        )
        edges = backend.list_inheritance(code, "m.py")
        pairs = {(d.src, d.symbol) for d in edges}
        self.assertEqual(pairs, {("Child", "Base"), ("Child", "Mixin")})
        for dep in edges:
            self.assertEqual(dep.relation, "inherits")

    def test_python_empty_on_syntax_error(self) -> None:
        self.assertEqual(get_backend("python").list_inheritance("class"), [])

    def test_rust_trait_impl_is_inheritance(self) -> None:
        backend = get_backend("rust")
        code = "struct Store;\ntrait Repo {}\nimpl Repo for Store {}\n"
        edges = backend.list_inheritance(code, "m.rs")
        pairs = {(d.src, d.symbol) for d in edges}
        self.assertIn(("Store", "Repo"), pairs)
        for dep in edges:
            self.assertEqual(dep.relation, "inherits")

    def test_go_without_inheritance_is_empty(self) -> None:
        backend = get_backend("go")
        code = "package m\n\ntype S struct{}\n"
        self.assertEqual(backend.list_inheritance(code, "m.go"), [])


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


class GoBackendCodeStructureTests(unittest.TestCase):
    """Go backend code-structure helpers delegate to ``lang_parser``."""

    SAMPLE_GO = """\
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

    def setUp(self) -> None:
        self.backend: GoBackend = get_backend("go")  # type: ignore

    def test_list_code_units(self) -> None:
        units = self.backend.list_code_units(self.SAMPLE_GO, "server.go")
        by_name = {unit.name: unit for unit in units}
        self.assertEqual(by_name["Server"].unit_type, "struct")
        self.assertEqual(by_name["NewServer"].unit_type, "function")
        self.assertEqual(by_name["Handle"].unit_type, "method")
        self.assertEqual(by_name["Handle"].parent, "Server")

    def test_list_code_units_empty_on_syntax_error(self) -> None:
        self.assertEqual(self.backend.list_code_units("func broken(\n", "bad.go"), [])

    def test_format_signature(self) -> None:
        units = self.backend.list_code_units(self.SAMPLE_GO, "server.go")
        by_name = {unit.name: unit for unit in units}
        self.assertEqual(
            self.backend.format_signature(by_name["NewServer"]),
            "func NewServer(name string) *Server",
        )
        self.assertEqual(
            self.backend.format_signature(by_name["Handle"]),
            "func (s *Server) Handle()",
        )
        self.assertEqual(self.backend.format_signature(by_name["Server"]), "Server")
        self.assertEqual(self.backend.format_signature(None), "")

    def test_list_imports(self) -> None:
        imports = self.backend.list_imports(self.SAMPLE_GO, "server.go")
        self.assertEqual([dep.dst for dep in imports], ["fmt", "net/http"])
        self.assertEqual(imports[1].extra.get("alias"), "nethttp")

    def test_list_imports_empty_on_syntax_error(self) -> None:
        self.assertEqual(self.backend.list_imports("func broken(\n", "bad.go"), [])

    def test_list_inheritance_empty_for_plain_struct(self) -> None:
        self.assertEqual(self.backend.list_inheritance(self.SAMPLE_GO, "server.go"), [])


if __name__ == "__main__":
    unittest.main()
