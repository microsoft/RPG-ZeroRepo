"""Tests for the decoder backend registry and Python backend contract.

These tests focus on invariants relied on by code paths that already
route through :mod:`decoder_lang`. Unsupported methods are asserted to
raise ``NotImplementedError`` so accidental partial implementations are
visible.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make ``scripts/`` importable when these tests are run directly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    LanguageBackend,
    PromptHints,
    PythonBackend,
    ToolchainUnavailable,
    get_backend,
    list_backends,
    register_backend,
)
from decoder_lang.backend import resolve_target_language  # noqa: E402


class RegistryTests(unittest.TestCase):
    """Backend registry behaviour."""

    def test_python_backend_registered_by_default(self) -> None:
        self.assertIn("python", list_backends())

    def test_get_backend_returns_singleton(self) -> None:
        a = get_backend("python")
        b = get_backend("python")
        self.assertIs(a, b)

    def test_unknown_language_falls_back_to_python_with_warning(self) -> None:
        with self.assertLogs("decoder_lang.backend", level="WARNING") as cm:
            backend = get_backend("nonexistent-language")
        self.assertEqual(backend.name, "python")
        self.assertTrue(
            any("falling back" in msg for msg in cm.output),
            f"expected fallback warning, got: {cm.output}",
        )

    def test_none_language_returns_default_silently(self) -> None:
        # None is the explicit "no info" case; not a misconfiguration,
        # so no warning expected.
        backend = get_backend(None)
        self.assertEqual(backend.name, "python")

    def test_python_backend_satisfies_protocol(self) -> None:
        backend = get_backend("python")
        # Runtime Protocol check confirms every required attribute exists.
        self.assertIsInstance(backend, LanguageBackend)

    def test_register_backend_replaces_existing(self) -> None:
        # Roundtrip: register a fake then restore.
        class _FakePython(PythonBackend):
            name = "python"

        try:
            register_backend(_FakePython)
            self.assertIsInstance(get_backend("python"), _FakePython)
        finally:
            register_backend(PythonBackend)
            self.assertNotIsInstance(get_backend("python"), _FakePython)


class FileLayoutTests(unittest.TestCase):
    """Behaviour-preservation for the trial-wired
    ``is_source_file`` path and surrounding layout helpers."""

    def setUp(self) -> None:
        self.backend = get_backend("python")

    # --- is_source_file equivalence with old suffix check -----------

    def test_is_source_file_accepts_py(self) -> None:
        self.assertTrue(self.backend.is_source_file("foo/bar.py"))

    def test_is_source_file_rejects_non_py(self) -> None:
        for path in ("README.md", "data.json", "foo.pyc", "Makefile",
                     "src/no_ext", "foo.PY"):
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_source_file(path))

    # Cross-check: ``not is_source_file(p)`` == ``Path(p).suffix != ".py"``
    # — the exact predicate the original ``static_completeness_check``
    # used. Equivalence here is what makes the trial wiring safe.
    def test_is_source_file_equivalent_to_old_suffix_check(self) -> None:
        from pathlib import PurePosixPath
        for path in ("a.py", "a.PY", "a.pyi", "x/y.py", "Makefile",
                     "tests/test_x.py", "weird.py.bak"):
            with self.subTest(path=path):
                old = PurePosixPath(path).suffix != ".py"
                new = not self.backend.is_source_file(path)
                self.assertEqual(old, new, f"divergent for {path}")

    # --- is_test_file ---------------------------------------------

    def test_is_test_file_matches_pytest_conventions(self) -> None:
        for path in (
            "tests/test_foo.py",
            "src/pkg/tests/test_inner.py",
            "test_root.py",
            "foo_test.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(self.backend.is_test_file(path))

    def test_is_test_file_rejects_regular_sources(self) -> None:
        for path in ("src/pkg/core.py", "main.py", "tester.py"):
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_test_file(path))

    # --- package markers -----------------------------------------

    def test_package_marker(self) -> None:
        self.assertEqual(self.backend.package_marker_filename(), "__init__.py")
        # Empty body matches the pre-existing skeleton emitter.
        self.assertEqual(self.backend.package_marker_content("pkg/sub"), "")

    # --- identifier rules ----------------------------------------

    def test_is_valid_module_identifier(self) -> None:
        self.assertTrue(self.backend.is_valid_module_identifier("auth"))
        self.assertTrue(self.backend.is_valid_module_identifier("auth_utils"))
        for bad in ("", "1auth", "auth-utils", "auth utils", "class", "def"):
            with self.subTest(seg=bad):
                self.assertFalse(self.backend.is_valid_module_identifier(bad))

    def test_sanitize_module_identifier_is_idempotent(self) -> None:
        cases = [
            ("auth-utils", "auth_utils"),
            ("1stage", "_1stage"),
            ("foo bar", "foo_bar"),
            ("ok_name", "ok_name"),
            ("", "_"),
        ]
        for raw, want in cases:
            with self.subTest(raw=raw):
                got = self.backend.sanitize_module_identifier(raw)
                self.assertEqual(got, want)
                # Idempotency: a second pass changes nothing.
                self.assertEqual(
                    self.backend.sanitize_module_identifier(got), got,
                )


class CodeStructureTests(unittest.TestCase):
    """``has_placeholder`` + ``syntax_check`` mirror the original
    semantics inside ``static_completeness_check``."""

    def setUp(self) -> None:
        self.backend = get_backend("python")

    def test_has_placeholder_true_on_todo_return(self) -> None:
        code = (
            "def f():\n"
            "    return 'TODO: implement me'\n"
        )
        self.assertTrue(self.backend.has_placeholder(code))

    def test_has_placeholder_true_on_placeholder_marker(self) -> None:
        code = "def f():\n    return 'PLACEHOLDER value'\n"
        self.assertTrue(self.backend.has_placeholder(code))

    def test_has_placeholder_true_on_not_implemented_string(self) -> None:
        code = "def f():\n    return 'Not implemented yet'\n"
        self.assertTrue(self.backend.has_placeholder(code))

    def test_has_placeholder_false_on_normal_code(self) -> None:
        code = (
            "def add(a, b):\n"
            "    '''A docstring mentioning TODO is fine.'''\n"
            "    return a + b\n"
        )
        self.assertFalse(self.backend.has_placeholder(code))

    def test_has_placeholder_false_on_non_string_return(self) -> None:
        self.assertFalse(self.backend.has_placeholder("def f(): return 42"))

    def test_has_placeholder_false_on_syntax_error(self) -> None:
        # Garbled source must NOT be reported as containing a placeholder.
        self.assertFalse(self.backend.has_placeholder("def f(:\n    pass\n"))

    def test_syntax_check(self) -> None:
        ok, err = self.backend.syntax_check("x = 1\n")
        self.assertTrue(ok)
        self.assertIsNone(err)
        ok, err = self.backend.syntax_check("def f(:\n    pass\n")
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        self.assertIn("SyntaxError", err or "")


class StubbedMethodsTests(unittest.TestCase):
    """Unsupported methods must raise instead of returning bad data."""

    def setUp(self) -> None:
        self.backend = get_backend("python")

    def test_detect_env_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.backend.detect_env(Path("."))

    def test_ensure_env_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.backend.ensure_env(Path("."))

    def test_test_command_stub(self) -> None:
        from decoder_lang.test_result import EnvHandle
        with self.assertRaises(NotImplementedError):
            self.backend.test_command(EnvHandle(project_root=Path(".")))

    def test_install_deps_command_stub(self) -> None:
        from decoder_lang.test_result import EnvHandle
        with self.assertRaises(NotImplementedError):
            self.backend.install_deps_command(
                EnvHandle(project_root=Path(".")), deps=["x"],
            )

    def test_parse_test_output_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.backend.parse_test_output("foo", 0)


class PromptHintsTests(unittest.TestCase):
    """Prompt-hint fields are populated and the instance is cached."""

    def test_prompt_hints_shape(self) -> None:
        hints = get_backend("python").prompt_hints()
        self.assertIsInstance(hints, PromptHints)
        self.assertEqual(hints.display_name, "Python")
        self.assertEqual(hints.markdown_fence, "python")
        self.assertEqual(hints.file_extension, ".py")
        self.assertEqual(hints.test_framework_name, "pytest")
        # Non-empty guidance strings ensure templates don't render blanks.
        self.assertTrue(hints.style_directive.strip())
        self.assertTrue(hints.module_naming_rule.strip())
        self.assertTrue(hints.package_layout_example.strip())

    def test_prompt_hints_is_cached(self) -> None:
        a = get_backend("python").prompt_hints()
        b = get_backend("python").prompt_hints()
        self.assertIs(a, b)


class ResolveTargetLanguageTests(unittest.TestCase):
    """Three-tier target-language fallback chain."""

    def test_tier_1_reads_root_meta_language(self) -> None:
        rpg = {"root": {"meta": {"language": "go"}}}
        self.assertEqual(resolve_target_language(rpg), "go")

    def test_tier_2_uses_dominant_language_when_root_missing(self) -> None:
        # Without root.meta.language, fall back to dominant_language()
        # over the provided file list. Use a Python-heavy list so we
        # don't depend on whatever lang_parser ships for non-Python.
        result = resolve_target_language(
            rpg_obj={"root": {}},
            valid_files=["a.py", "b.py", "c.py"],
        )
        self.assertEqual(result, "python")

    def test_tier_3_defaults_to_python_with_warning(self) -> None:
        with self.assertLogs("decoder_lang.backend", level="WARNING") as cm:
            result = resolve_target_language({}, valid_files=None)
        self.assertEqual(result, "python")
        self.assertTrue(
            any("defaulting to 'python'" in msg for msg in cm.output),
        )

    def test_handles_bad_input_gracefully(self) -> None:
        # None / non-dict shouldn't crash.
        with self.assertLogs("decoder_lang.backend", level="WARNING"):
            self.assertEqual(resolve_target_language(None), "python")
        with self.assertLogs("decoder_lang.backend", level="WARNING"):
            self.assertEqual(resolve_target_language("garbage"), "python")


class ToolchainUnavailableTests(unittest.TestCase):
    """:class:`ToolchainUnavailable` is a real exception type callers
    can catch by name."""

    def test_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(ToolchainUnavailable, RuntimeError))


if __name__ == "__main__":
    unittest.main()
