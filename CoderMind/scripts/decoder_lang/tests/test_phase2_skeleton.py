"""Tests for backend-aware skeleton behaviour.

Covers:

* :class:`decoder_lang.GoBackend` registration + backend methods.
* :func:`skeleton.file_designer.validate_directory_structure` honors
    the supplied backend's identifier rules; Python defaults apply when
    ``backend=None`` (Python default).
* :meth:`skeleton_models.RepoSkeleton.add_init_files` is a no-op for
  backends whose :meth:`package_marker_filename` returns ``None``
    (Go / Rust / TypeScript), and equivalent to the Python default
    path otherwise.
* :class:`FileDesigner.backend` is the registered backend for the
  resolved language (Go instance for a Go RPG, Python instance for a
  Python RPG).
"""
from __future__ import annotations

import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make ``scripts/`` importable for direct invocation.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    GoBackend,
    PythonBackend,
    ToolchainUnavailable,
    get_backend,
    list_backends,
)
from decoder_lang.test_result import EnvHandle  # noqa: E402


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
    """GoBackend behaviour exposed through the decoder backend contract."""

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

    # --- code structure ----------------------------------------------

    def test_syntax_check(self) -> None:
        ok, error = self.backend.syntax_check("package main\nfunc Run() {}\n")
        self.assertTrue(ok, error)
        ok, error = self.backend.syntax_check("func Run() {}\n")
        self.assertFalse(ok)
        self.assertIn("package", error or "")

    def test_has_placeholder(self) -> None:
        code = 'package main\nfunc Run() string { return "TODO: implement" }\n'
        self.assertTrue(self.backend.has_placeholder(code))
        self.assertFalse(
            self.backend.has_placeholder('package main\nfunc Run() string { return "ok" }\n')
        )

    # --- test environment --------------------------------------------

    def test_detect_env_none_when_go_missing(self) -> None:
        with patch("decoder_lang.go_backend.shutil.which", return_value=None):
            self.assertIsNone(self.backend.detect_env(Path(".")))

    def test_ensure_env_raises_when_go_missing(self) -> None:
        with patch("decoder_lang.go_backend.shutil.which", return_value=None):
            with self.assertRaises(ToolchainUnavailable):
                self.backend.ensure_env(Path("."))

    def test_ensure_env_creates_go_mod_when_toolchain_exists(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("decoder_lang.go_backend.shutil.which", return_value="/usr/bin/go"):
                env = self.backend.ensure_env(root)
            self.assertEqual(env.runtime_executable, "/usr/bin/go")
            self.assertEqual(env.extra.get("module"), f"codermind.local/{root.name.lower()}")
            self.assertTrue((root / "go.mod").exists())
            self.assertIn("module codermind.local", (root / "go.mod").read_text())

    def test_test_command(self) -> None:
        cmd = self.backend.test_command(
            EnvHandle(project_root=Path("."), runtime_executable="/usr/bin/go"),
            selectors=["TestRun", "TestStop"],
        )
        self.assertEqual(cmd, ["/usr/bin/go", "test", "-v", "-run", "TestRun|TestStop", "./..."])

    def test_install_deps_command(self) -> None:
        env = EnvHandle(project_root=Path("."), runtime_executable="/usr/bin/go")
        self.assertIsNone(self.backend.install_deps_command(env, []))
        self.assertEqual(
            self.backend.install_deps_command(env, ["github.com/acme/lib"]),
            ["/usr/bin/go", "get", "github.com/acme/lib"],
        )

    def test_parse_test_output(self) -> None:
        raw = "\n".join([
            "=== RUN   TestRun",
            "--- PASS: TestRun (0.01s)",
            "=== RUN   TestBroken",
            "    service_test.go:12: expected true",
            "--- FAIL: TestBroken (0.02s)",
            "FAIL\texample.com/demo\t0.03s",
        ])
        result = self.backend.parse_test_output(raw, 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.passed_count, 1)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(result.failures[0].test_id, "TestBroken")
        self.assertEqual(result.failures[0].file_path, "service_test.go")
        self.assertEqual(result.failures[0].line, 12)

    def test_parse_test_output_without_test_failure_is_error(self) -> None:
        result = self.backend.parse_test_output("FAIL\texample.com/demo\n", 1)
        self.assertEqual(result.status, "errored")
        self.assertEqual(result.error_count, 1)

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

    def test_python_default_identifier_rules(self) -> None:
        # No backend → Python identifier rules: hyphens are rejected.
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

    def test_python_marker_added_without_backend(self) -> None:
        # backend=None uses Python __init__.py emission.
        skel = self._make_skeleton()
        added = skel.add_init_files()
        self.assertEqual(added, 1)
        self.assertIn("src/__init__.py", skel.path_to_node)

    def test_python_backend_matches_no_backend(self) -> None:
        # Passing PythonBackend explicitly produces the same package
        # markers as default backend resolution.
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
    resolved from the RPG, including the registered Go backend."""

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
