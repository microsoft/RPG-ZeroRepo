"""Tests for Phase 2 of decoder multi-language: skeleton stage.

Covers:

* :class:`decoder_lang.GoBackend` registration + skeleton-relevant methods.
* :func:`skeleton.file_designer.validate_directory_structure` honours
  the supplied backend's identifier rules; behaviour is unchanged when
  ``backend=None`` (legacy callers).
* :meth:`skeleton_models.RepoSkeleton.add_init_files` is a no-op for
  backends whose :meth:`package_marker_filename` returns ``None``
  (Go / Rust / TypeScript), and bit-equivalent to the pre-Phase-2
  Python path otherwise.
* :class:`FileDesigner.backend` is the registered backend for the
  resolved language (Go instance for a Go RPG, Python instance for a
  Python RPG).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Make ``scripts/`` importable for direct invocation.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    GoBackend,
    PythonBackend,
    get_backend,
    list_backends,
)


class GoBackendRegistrationTests(unittest.TestCase):
    """Go backend is in the registry and returns the same instance."""

    def test_go_backend_registered(self) -> None:
        self.assertIn("go", list_backends())

    def test_get_backend_go_returns_singleton(self) -> None:
        a = get_backend("go")
        b = get_backend("go")
        self.assertIs(a, b)
        self.assertIsInstance(a, GoBackend)


class GoBackendBehaviourTests(unittest.TestCase):
    """The skeleton-relevant subset of GoBackend behaves correctly."""

    def setUp(self) -> None:
        self.backend = get_backend("go")

    # --- file classification -----------------------------------------

    def test_is_source_file(self) -> None:
        self.assertTrue(self.backend.is_source_file("cmd/myapp/main.go"))
        self.assertTrue(self.backend.is_source_file("internal/core/core_test.go"))
        for path in ("README.md", "main.py", "main.GO", "main"):
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_source_file(path))

    def test_is_test_file(self) -> None:
        self.assertTrue(self.backend.is_test_file("foo_test.go"))
        self.assertTrue(self.backend.is_test_file("internal/x/y_test.go"))
        for path in ("foo.go", "tests/foo.go", "test_foo.go"):
            # Note: Go convention is *_test.go, NOT test_*.go
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_test_file(path))

    # --- package marker ----------------------------------------------

    def test_no_package_marker(self) -> None:
        self.assertIsNone(self.backend.package_marker_filename())
        self.assertIsNone(self.backend.package_marker_content("any/path"))

    # --- identifier rules --------------------------------------------

    def test_valid_identifiers(self) -> None:
        for seg in ("auth", "auth_utils", "_internal", "Foo123"):
            with self.subTest(seg=seg):
                self.assertTrue(self.backend.is_valid_module_identifier(seg))

    def test_invalid_identifiers(self) -> None:
        for seg in ("", "1auth", "auth-utils", "auth utils", "package", "func"):
            with self.subTest(seg=seg):
                self.assertFalse(self.backend.is_valid_module_identifier(seg))

    def test_sanitize(self) -> None:
        self.assertEqual(self.backend.sanitize_module_identifier("auth-utils"), "auth_utils")
        self.assertEqual(self.backend.sanitize_module_identifier("1auth"), "_1auth")
        # Keyword collision avoided by suffix.
        self.assertEqual(self.backend.sanitize_module_identifier("func"), "func_")
        # Idempotency
        s = self.backend.sanitize_module_identifier("a-b-c")
        self.assertEqual(self.backend.sanitize_module_identifier(s), s)

    # --- stubbed methods raise ---------------------------------------

    def test_ast_methods_stub(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.backend.has_placeholder("package main")
        with self.assertRaises(NotImplementedError):
            self.backend.syntax_check("package main")

    def test_test_methods_stub(self) -> None:
        from decoder_lang.test_result import EnvHandle
        with self.assertRaises(NotImplementedError):
            self.backend.test_command(EnvHandle(project_root=Path(".")))
        with self.assertRaises(NotImplementedError):
            self.backend.detect_env(Path("."))

    # --- prompt hints ------------------------------------------------

    def test_prompt_hints(self) -> None:
        hints = self.backend.prompt_hints()
        self.assertEqual(hints.display_name, "Go")
        self.assertEqual(hints.markdown_fence, "go")
        self.assertEqual(hints.file_extension, ".go")
        self.assertEqual(hints.test_framework_name, "go test")
        self.assertIn("idiomatic Go", hints.style_directive)


class ValidateDirectoryStructureTests(unittest.TestCase):
    """Backend-aware identifier validation in ``validate_directory_structure``."""

    def setUp(self) -> None:
        from skeleton.file_designer import validate_directory_structure  # noqa
        self.validate = validate_directory_structure

    def test_python_default_unchanged(self) -> None:
        # No backend → historical behaviour: hyphens are rejected.
        ok, msg = self.validate(
            {"comp": "src/my-pkg/utils"}, ["comp"],
        )
        self.assertFalse(ok)
        self.assertIn("my-pkg", msg)
        self.assertIn("Python identifier", msg)

    def test_go_backend_accepts_lowercase_underscored(self) -> None:
        ok, msg = self.validate(
            {"comp": "internal/auth_utils/token"}, ["comp"],
            backend=get_backend("go"),
        )
        self.assertTrue(ok, msg)

    def test_go_backend_rejects_hyphen(self) -> None:
        ok, msg = self.validate(
            {"comp": "internal/auth-utils"}, ["comp"],
            backend=get_backend("go"),
        )
        self.assertFalse(ok)
        self.assertIn("auth-utils", msg)
        self.assertIn("Go identifier", msg)

    def test_go_backend_rejects_keyword(self) -> None:
        ok, msg = self.validate(
            {"comp": "internal/func"}, ["comp"],
            backend=get_backend("go"),
        )
        self.assertFalse(ok)
        self.assertIn("func", msg)


class AddInitFilesTests(unittest.TestCase):
    """Verify the behaviour-preservation contract on ``add_init_files``.

    Uses a small in-memory ``RepoSkeleton`` so the test runs without
    touching the LLM pipeline.
    """

    def _make_skeleton(self):
        from skeleton.skeleton_models import RepoSkeleton  # noqa: E402

        # RepoSkeleton accepts a flat ``{file_path: source_code}`` map
        # and builds the directory tree automatically. We only need a
        # single source file under a sub-directory so that
        # ``add_init_files`` has at least one candidate directory.
        return RepoSkeleton({"src/foo.py": ""})

    def test_default_behaviour_unchanged_no_backend(self) -> None:
        # backend=None preserves pre-Phase-2 Python __init__.py emission.
        skel = self._make_skeleton()
        added = skel.add_init_files()
        self.assertEqual(added, 1)
        self.assertIn("src/__init__.py", skel.path_to_node)

    def test_python_backend_matches_no_backend(self) -> None:
        # Passing PythonBackend explicitly produces the same result as
        # not passing one. (Documents the back-compat invariant.)
        skel_a = self._make_skeleton()
        a = skel_a.add_init_files()

        skel_b = self._make_skeleton()
        b = skel_b.add_init_files(backend=get_backend("python"))

        self.assertEqual(a, b)
        self.assertEqual(
            set(skel_a.path_to_node), set(skel_b.path_to_node),
        )

    def test_go_backend_is_noop(self) -> None:
        # backend whose package_marker_filename() is None makes the
        # whole method a no-op: zero files added, registry unchanged.
        skel = self._make_skeleton()
        before = set(skel.path_to_node)
        added = skel.add_init_files(backend=get_backend("go"))
        self.assertEqual(added, 0)
        self.assertEqual(set(skel.path_to_node), before)


class FileDesignerBackendInstanceTests(unittest.TestCase):
    """``FileDesigner.backend`` is the right instance for the language
    resolved from the RPG. Already covered structurally in Phase 1
    tests; Phase 2 adds the Go-specific assertion now that GoBackend
    exists."""

    def _make_designer(self, root_language):
        from skeleton.file_designer import FileDesigner  # noqa

        rpg = MagicMock()
        rpg.repo_node = MagicMock()
        rpg.repo_node.meta = MagicMock()
        rpg.repo_node.meta.language = root_language
        return FileDesigner(rpg=rpg, llm_client=MagicMock())

    def test_python_rpg_gets_python_backend(self) -> None:
        d = self._make_designer("python")
        self.assertIs(d.backend, get_backend("python"))
        self.assertIsInstance(d.backend, PythonBackend)

    def test_go_rpg_gets_go_backend(self) -> None:
        d = self._make_designer("go")
        self.assertIs(d.backend, get_backend("go"))
        self.assertIsInstance(d.backend, GoBackend)

    def test_fallback_filename_uses_backend_extension(self) -> None:
        d = self._make_designer("go")
        # We don't run the full designer pipeline; just assert the
        # backend extension is what the misc-fallback code uses.
        self.assertEqual(d.backend.file_extension, ".go")


if __name__ == "__main__":
    unittest.main()
