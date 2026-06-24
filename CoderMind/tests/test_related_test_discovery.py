import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from code_gen.test_runner import find_related_test_files


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_related_go_test_file(tmp_path):
    _write(tmp_path / "internal" / "store" / "store.go")
    _write(tmp_path / "internal" / "store" / "store_test.go")

    assert find_related_test_files("internal/store/store.go", tmp_path) == [
        "internal/store/store_test.go"
    ]


def test_find_related_typescript_test_file(tmp_path):
    _write(tmp_path / "src" / "client.ts")
    _write(tmp_path / "src" / "client.test.ts")

    assert find_related_test_files("src/client.ts", tmp_path) == [
        "src/client.test.ts"
    ]


def test_find_related_c_test_file(tmp_path):
    _write(tmp_path / "src" / "task.c")
    _write(tmp_path / "tests" / "test_task.c")

    assert find_related_test_files("src/task.c", tmp_path) == [
        "tests/test_task.c"
    ]


def test_find_related_cpp_test_file(tmp_path):
    _write(tmp_path / "src" / "task.cpp")
    _write(tmp_path / "tests" / "task_test.cpp")

    assert find_related_test_files("src/task.cpp", tmp_path) == [
        "tests/task_test.cpp"
    ]


def test_find_related_rust_test_file(tmp_path):
    _write(tmp_path / "src" / "lib.rs")
    _write(tmp_path / "tests" / "lib_test.rs")

    assert find_related_test_files("src/lib.rs", tmp_path) == [
        "tests/lib_test.rs"
    ]
