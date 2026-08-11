"""Tests for the language-aware smoke-test entry probe.

Covers the multilang `check_entry_point` path: the run command comes
from the backend, runs in a CLEAN subprocess (no PYTHONPATH bridging),
and the Python-only import/stub layers are skipped for other languages.
"""
from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import smoke_test  # noqa: E402


class TestEntryPointCleanEnv(unittest.TestCase):
    def test_src_layout_without_bridge_fails(self):
        # Reproduces the P3 bug: main.py imports a package under src/ but
        # there is no pyproject/path bridge → ModuleNotFoundError in a
        # clean env. The smoke entry probe must catch it.
        with TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "src" / "pkg").mkdir(parents=True)
            (repo / "src" / "pkg" / "__init__.py").write_text("")
            (repo / "src" / "pkg" / "app.py").write_text("def run():\n    return 0\n")
            (repo / "main.py").write_text(
                textwrap.dedent(
                    """\
                    import argparse
                    from pkg.app import run

                    def main():
                        argparse.ArgumentParser().parse_args()
                        return run()

                    if __name__ == "__main__":
                        raise SystemExit(main())
                    """
                )
            )
            result = smoke_test.SmokeResult()
            layer = smoke_test.check_entry_point(repo, result)
            self.assertFalse(layer.get("passed"))
            self.assertTrue(
                any(f.check == "help_fails" for f in result.findings),
                [f.check for f in result.findings],
            )

    def test_src_layout_with_path_bridge_passes(self):
        # Same layout, but main.py adds the sys.path bridge → --help works.
        with TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "src" / "pkg").mkdir(parents=True)
            (repo / "src" / "pkg" / "__init__.py").write_text("")
            (repo / "src" / "pkg" / "app.py").write_text("def run():\n    return 0\n")
            (repo / "main.py").write_text(
                textwrap.dedent(
                    """\
                    import sys, pathlib
                    sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
                    import argparse
                    from pkg.app import run

                    def main():
                        argparse.ArgumentParser().parse_args()
                        return run()

                    if __name__ == "__main__":
                        raise SystemExit(main())
                    """
                )
            )
            result = smoke_test.SmokeResult()
            layer = smoke_test.check_entry_point(repo, result)
            self.assertTrue(layer.get("passed"), [f.message for f in result.findings])


class TestPythonOnlyLayersSkipped(unittest.TestCase):
    def test_non_python_skips_import_and_stub_layers(self):
        # A Go-flagged repo must skip the ast-based import/stub layers.
        with TemporaryDirectory() as d:
            repo = Path(d)
            (repo / ".cmind" / "data").mkdir(parents=True)
            (repo / ".cmind" / "data" / "rpg.json").write_text(
                '{"root": {"meta": {"language": "go"}}}'
            )
            res = smoke_test.run_smoke_test(repo_path=repo)
            self.assertTrue(res.layers["imports"].get("skipped"))
            self.assertTrue(res.layers["stubs"].get("skipped"))


class TestPythonSourceDiscovery(unittest.TestCase):
    def test_setup_py_is_not_imported_as_application_source(self):
        with TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "src" / "pkg").mkdir(parents=True)
            module = repo / "src" / "pkg" / "app.py"
            module.write_text("VALUE = 1\n")
            setup = repo / "setup.py"
            setup.write_text("raise RuntimeError('must not import setup.py')\n")

            discovered = smoke_test._find_source_files(repo)

            self.assertIn(module, discovered)
            self.assertNotIn(setup, discovered)


class TestSmokeCliModes(unittest.TestCase):
    def test_advisory_mode_preserves_findings_but_exits_successfully(self):
        result = smoke_test.SmokeResult(success=False, project_type="python")
        result.add_finding(smoke_test.SmokeFinding(
            layer="stubs",
            severity="error",
            check="stub_detected",
            message="baseline stub",
        ))

        with patch.object(sys, "argv", ["smoke_test.py", "--json", "--advisory"]), patch.object(
            smoke_test, "run_smoke_test", return_value=result,
        ), patch("common.logging_setup.setup_file_logging"):
            self.assertEqual(smoke_test.main(), 0)

    def test_strict_mode_still_fails_for_findings(self):
        result = smoke_test.SmokeResult(success=False, project_type="python")

        with patch.object(sys, "argv", ["smoke_test.py", "--json"]), patch.object(
            smoke_test, "run_smoke_test", return_value=result,
        ), patch("common.logging_setup.setup_file_logging"):
            self.assertEqual(smoke_test.main(), 1)


if __name__ == "__main__":
    unittest.main()
