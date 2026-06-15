"""Tests for the JavaScript decoder backend.

Run from ``scripts/`` (e.g. ``python -m pytest decoder_lang/tests``) so the
sibling ``common`` / ``lang_parser`` packages are importable.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    JavaScriptBackend,
    ToolchainUnavailable,
    get_backend,
    list_backends,
)
from decoder_lang.test_result import EnvHandle  # noqa: E402


class JavaScriptBackendRegistrationTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIn("javascript", list_backends())

    def test_get_backend_returns_singleton(self) -> None:
        a = get_backend("javascript")
        b = get_backend("javascript")
        self.assertIs(a, b)
        self.assertIsInstance(a, JavaScriptBackend)


class JavaScriptBackendBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = get_backend("javascript")

    # --- identity ----------------------------------------------------

    def test_identity_fields(self) -> None:
        self.assertEqual(self.backend.name, "javascript")
        self.assertEqual(self.backend.display_name, "JavaScript")
        self.assertEqual(self.backend.file_extension, ".js")
        self.assertEqual(self.backend.markdown_fence, "javascript")

    # --- file classification -----------------------------------------

    def test_is_source_file(self) -> None:
        for path in ("src/index.js", "src/cli.mjs", "lib/store.cjs", "ui/app.jsx"):
            with self.subTest(path=path):
                self.assertTrue(self.backend.is_source_file(path))
        for path in ("README.md", "main.py", "src/app.ts", "main"):
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_source_file(path))

    def test_is_test_file(self) -> None:
        for path in ("tests/cli.js", "src/store.test.js", "src/cli.spec.mjs"):
            with self.subTest(path=path):
                self.assertTrue(self.backend.is_test_file(path))
        for path in ("src/index.js", "lib/store.cjs"):
            with self.subTest(path=path):
                self.assertFalse(self.backend.is_test_file(path))

    # --- package marker / identifiers --------------------------------

    def test_no_package_marker(self) -> None:
        self.assertIsNone(self.backend.package_marker_filename())
        self.assertIsNone(self.backend.package_marker_content("any/path"))

    def test_identifier_rules(self) -> None:
        self.assertTrue(self.backend.is_valid_module_identifier("task-store"))
        self.assertTrue(self.backend.is_valid_module_identifier("cli"))
        self.assertFalse(self.backend.is_valid_module_identifier(""))
        self.assertFalse(self.backend.is_valid_module_identifier("a/b"))

    def test_sanitize(self) -> None:
        self.assertEqual(self.backend.sanitize_module_identifier("my mod"), "my-mod")
        self.assertEqual(self.backend.sanitize_module_identifier("a/b/c"), "a-b-c")
        s = self.backend.sanitize_module_identifier("x--y  z")
        self.assertEqual(self.backend.sanitize_module_identifier(s), s)  # idempotent

    # --- code structure ----------------------------------------------

    def test_syntax_check_ok(self) -> None:
        ok, err = self.backend.syntax_check(
            "// user's data — doesn't break\nexport function f() { return 1; }\n",
            "src/a.js",
        )
        self.assertTrue(ok, err)

    def test_syntax_check_failure(self) -> None:
        ok, err = self.backend.syntax_check("export function broken(\n", "src/b.js")
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_list_code_units(self) -> None:
        code = "export function foo() {}\nclass Bar { run() {} }\n"
        units = self.backend.list_code_units(code, "src/c.js")
        kinds = {(u.unit_type, u.name) for u in units}
        self.assertIn(("function", "foo"), kinds)
        self.assertIn(("class", "Bar"), kinds)

    def test_has_placeholder(self) -> None:
        self.assertTrue(self.backend.has_placeholder(
            'export function f() { throw new Error("not implemented"); }\n'
        ))
        self.assertFalse(self.backend.has_placeholder(
            "export function f() { return 42; }\n"
        ))

    def test_list_imports(self) -> None:
        code = "import { store } from './store.js';\nexport function f() {}\n"
        imports = self.backend.list_imports(code, "src/c.js")
        self.assertTrue(any(getattr(d, "relation", "") == "imports" for d in imports))

    # --- test environment --------------------------------------------

    def test_detect_env_none_when_node_missing(self) -> None:
        with patch("decoder_lang.javascript_backend.shutil.which", return_value=None):
            self.assertIsNone(self.backend.detect_env(Path(".")))

    def test_ensure_env_raises_when_node_missing(self) -> None:
        with patch("decoder_lang.javascript_backend.shutil.which", return_value=None):
            with self.assertRaises(ToolchainUnavailable):
                self.backend.ensure_env(Path("."))

    def test_ensure_env_creates_package_json(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "decoder_lang.javascript_backend.shutil.which",
                return_value="/usr/bin/npm",
            ):
                self.backend.ensure_env(root)
            pkg = root / "package.json"
            self.assertTrue(pkg.exists())
            self.assertIn('"type": "module"', pkg.read_text())
            self.assertNotIn("tsconfig", pkg.read_text())

    def test_test_command_npm_vs_node(self) -> None:
        npm_env = EnvHandle(project_root=Path("."), runtime_executable="/usr/bin/npm")
        self.assertEqual(self.backend.test_command(npm_env), ["/usr/bin/npm", "test"])
        node_env = EnvHandle(project_root=Path("."), runtime_executable="/usr/bin/node")
        self.assertEqual(self.backend.test_command(node_env), ["/usr/bin/node", "--test"])

    # --- prompt hints / templates ------------------------------------

    def test_prompt_hints_are_javascript(self) -> None:
        hints = self.backend.prompt_hints()
        self.assertEqual(hints.markdown_fence, "javascript")
        self.assertIn(".js", hints.entrypoint_example)
        # Must steer away from TypeScript.
        self.assertIn("TypeScript", hints.style_directive)

    def test_project_task_templates_avoid_typescript(self) -> None:
        from decoder_lang.project_tasks import ProjectTaskContext

        ctx = ProjectTaskContext(
            repo_name="tasklite",
            repo_info="A small task CLI",
            package_name="tasklite",
        )
        templates = self.backend.project_task_templates(ctx)
        self.assertIsNotNone(templates)
        self.assertIn("package.json", templates.dependencies)
        self.assertIn("tsconfig", templates.dependencies)  # mentioned as a "do NOT"
        self.assertIn("src/index.js", templates.main_entry)


if __name__ == "__main__":
    unittest.main()
