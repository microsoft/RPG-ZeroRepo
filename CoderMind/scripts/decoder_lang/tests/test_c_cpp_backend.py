"""Tests for C and C++ decoder language backends."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    CBackend,
    CppBackend,
    EnvHandle,
    ProjectTaskContext,
    ToolchainUnavailable,
    get_backend,
    language_directive,
    list_backends,
)


class CBackendTests(unittest.TestCase):
    """C backend registry and parser-backed behaviour."""

    def setUp(self) -> None:
        self.backend = get_backend("c")

    def test_registered(self) -> None:
        self.assertIn("c", list_backends())
        self.assertIsInstance(self.backend, CBackend)

    def test_file_classification(self) -> None:
        self.assertTrue(self.backend.is_source_file("src/store.c"))
        self.assertTrue(self.backend.is_source_file("include/store.h"))
        self.assertFalse(self.backend.is_source_file("src/store.cpp"))
        self.assertTrue(self.backend.is_test_file("tests/test_store.c"))
        self.assertTrue(self.backend.is_test_file("src/store_test.c"))

    def test_identifier_rules(self) -> None:
        self.assertTrue(self.backend.is_valid_module_identifier("task_store"))
        self.assertFalse(self.backend.is_valid_module_identifier("struct"))
        self.assertEqual(self.backend.sanitize_module_identifier("task-store"), "task_store")
        self.assertEqual(self.backend.sanitize_module_identifier("1task"), "_1task")

    def test_code_units_imports_and_signature(self) -> None:
        code = """
        #include "store.h"

        struct Task { int id; };

        int load_task(int id);

        int add_task(int id) {
            return id + 1;
        }
        """
        ok, error = self.backend.syntax_check(code, "src/store.c")
        self.assertTrue(ok, error)
        units = self.backend.list_code_units(code, "src/store.c")
        names = {(unit.unit_type, unit.name) for unit in units}
        self.assertIn(("struct", "Task"), names)
        self.assertIn(("function", "load_task"), names)
        self.assertIn(("function", "add_task"), names)
        function = next(unit for unit in units if unit.name == "add_task")
        self.assertIn("add_task", self.backend.format_signature(function))
        imports = self.backend.list_imports(code, "src/store.c")
        self.assertEqual([dep.dst for dep in imports], ["store.h"])

    def test_prompt_hints_and_project_tasks(self) -> None:
        hints = self.backend.prompt_hints()
        self.assertEqual(hints.display_name, "C")
        self.assertEqual(hints.markdown_fence, "c")
        self.assertIn("C99", hints.style_directive)
        self.assertIn("Target language: C", language_directive(self.backend))
        templates = self.backend.project_task_templates(
            ProjectTaskContext(repo_name="tasklite", repo_info="task cli", package_name="tasklite")
        )
        self.assertIn("Makefile", templates.dependencies)
        self.assertIn("must build and execute real test binaries", templates.dependencies)
        self.assertIn("must not only", templates.dependencies)
        self.assertIn("src/main.c", templates.main_entry)
        self.assertIn("must execute tests", templates.main_entry)
        self.assertIn("C CLI", templates.readme)

    def test_missing_toolchain_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("decoder_lang.c_backend.shutil.which", return_value=None):
                with self.assertRaises(ToolchainUnavailable):
                    self.backend.ensure_env(Path(temp_dir))

    def test_syntax_fallback_skips_git_refs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "main.c").write_text("int main(void) { return 0; }\n")
            (root / ".git" / "refs" / "heads" / "batch").mkdir(parents=True)
            (root / ".git" / "refs" / "heads" / "batch" / "main.c").write_text("not c")
            env = EnvHandle(project_root=root, extra={"cc": "/usr/bin/cc"})

            cmd = self.backend.test_command(env)

            self.assertIn(str(root / "src" / "main.c"), cmd)
            self.assertNotIn(str(root / ".git" / "refs" / "heads" / "batch" / "main.c"), cmd)

    def test_make_test_compile_only_output_is_not_pass(self) -> None:
        raw = "cc -Isrc -std=c99 -Wall -Wextra -c tests/test_engine.c -o build/tests/test_engine.o\n"

        result = self.backend.parse_test_output(raw, 0)

        self.assertEqual(result.status, "errored")

    def test_make_test_nothing_to_do_is_not_pass(self) -> None:
        result = self.backend.parse_test_output("make: Nothing to be done for 'test'.\n", 0)

        self.assertEqual(result.status, "errored")


class CppBackendTests(unittest.TestCase):
    """C++ backend registry and parser-backed behaviour."""

    def setUp(self) -> None:
        self.backend = get_backend("cpp")

    def test_registered(self) -> None:
        self.assertIn("cpp", list_backends())
        self.assertIsInstance(self.backend, CppBackend)

    def test_file_classification(self) -> None:
        self.assertTrue(self.backend.is_source_file("src/store.cpp"))
        self.assertTrue(self.backend.is_source_file("include/store.hpp"))
        self.assertTrue(self.backend.is_source_file("include/store.h"))
        self.assertFalse(self.backend.is_source_file("src/store.c"))
        self.assertTrue(self.backend.is_test_file("tests/store_test.cpp"))
        self.assertTrue(self.backend.is_test_file("src/test_store.cc"))

    def test_identifier_rules(self) -> None:
        self.assertTrue(self.backend.is_valid_module_identifier("TaskStore"))
        self.assertFalse(self.backend.is_valid_module_identifier("class"))
        self.assertEqual(self.backend.sanitize_module_identifier("task-store"), "task_store")
        self.assertEqual(self.backend.sanitize_module_identifier("1task"), "_1task")

    def test_code_units_imports_and_signature(self) -> None:
        code = """
        #include "store.hpp"

        int run_task(int id);

        class TaskStore {
        public:
            int add(int id) { return id + 1; }
        };

        int run() {
            TaskStore store;
            return store.add(1);
        }
        """
        ok, error = self.backend.syntax_check(code, "src/store.cpp")
        self.assertTrue(ok, error)
        units = self.backend.list_code_units(code, "src/store.cpp")
        names = {(unit.unit_type, unit.name) for unit in units}
        self.assertIn(("class", "TaskStore"), names)
        self.assertIn(("function", "run_task"), names)
        self.assertIn(("function", "run"), names)
        run = next(unit for unit in units if unit.name == "run")
        self.assertIn("run", self.backend.format_signature(run))
        imports = self.backend.list_imports(code, "src/store.cpp")
        self.assertEqual([dep.dst for dep in imports], ["store.hpp"])

    def test_h_header_parses_as_cpp(self) -> None:
        code = "class Reader { public: int value() const { return 1; } };\n"
        ok, error = self.backend.syntax_check(code, "include/reader.h")
        self.assertTrue(ok, error)
        units = self.backend.list_code_units(code, "include/reader.h")
        self.assertTrue(any(unit.name == "Reader" for unit in units))

    def test_prompt_hints_and_project_tasks(self) -> None:
        hints = self.backend.prompt_hints()
        self.assertEqual(hints.display_name, "C++")
        self.assertEqual(hints.markdown_fence, "cpp")
        self.assertIn("C++17", hints.style_directive)
        self.assertIn("Target language: C++", language_directive(self.backend))
        templates = self.backend.project_task_templates(
            ProjectTaskContext(repo_name="tasklite", repo_info="task cli", package_name="tasklite")
        )
        self.assertIn("CMakeLists.txt", templates.dependencies)
        self.assertIn("must build and execute real test binaries", templates.dependencies)
        self.assertIn("must not only", templates.dependencies)
        self.assertIn("src/main.cpp", templates.main_entry)
        self.assertIn("must execute tests", templates.main_entry)
        self.assertIn("C++ CLI", templates.readme)

    def test_cmake_test_command_runs_ctest_in_build_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
            env = EnvHandle(
                project_root=root,
                extra={"ctest": "/usr/bin/ctest"},
            )

            self.assertEqual(
                self.backend.test_command(env),
                [
                    "/usr/bin/ctest",
                    "--test-dir",
                    str(root / "build"),
                    "--output-on-failure",
                ],
            )

    def test_prepare_test_env_configures_and_builds_cmake_project(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
            env = EnvHandle(
                project_root=root,
                extra={"cmake": "/usr/bin/cmake"},
            )
            calls = []

            def fake_run(args, **kwargs):
                calls.append(args)

                class Result:
                    returncode = 0

                return Result()

            with patch("subprocess.run", side_effect=fake_run):
                self.backend.prepare_test_env(env)

            self.assertEqual(
                calls,
                [
                    ["/usr/bin/cmake", "-S", str(root), "-B", str(root / "build")],
                    ["/usr/bin/cmake", "--build", str(root / "build")],
                ],
            )

    def test_syntax_fallback_skips_git_refs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "main.cpp").write_text("int main() { return 0; }\n")
            (root / ".git" / "refs" / "heads" / "batch").mkdir(parents=True)
            (root / ".git" / "refs" / "heads" / "batch" / "main.cpp").write_text("not cpp")
            env = EnvHandle(project_root=root, extra={"cxx": "/usr/bin/c++"})

            cmd = self.backend.test_command(env)

            self.assertIn(str(root / "src" / "main.cpp"), cmd)
            self.assertNotIn(str(root / ".git" / "refs" / "heads" / "batch" / "main.cpp"), cmd)

    def test_make_test_compile_only_output_is_not_pass(self) -> None:
        raw = "c++ -std=c++17 -c tests/parser_test.cpp -o build/tests/parser_test.o\n"

        result = self.backend.parse_test_output(raw, 0)

        self.assertEqual(result.status, "errored")

    def test_make_test_nothing_to_do_is_not_pass(self) -> None:
        result = self.backend.parse_test_output("make: Nothing to be done for 'test'.\n", 0)

        self.assertEqual(result.status, "errored")

    def test_missing_toolchain_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("decoder_lang.cpp_backend.shutil.which", return_value=None):
                with self.assertRaises(ToolchainUnavailable):
                    self.backend.ensure_env(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()