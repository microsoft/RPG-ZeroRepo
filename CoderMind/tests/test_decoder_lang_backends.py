import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from decoder_lang.cpp_backend import CppBackend
from decoder_lang.test_result import EnvHandle


def test_cpp_cmake_ctest_command_targets_build_dir(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(sample)\nenable_testing()\n",
        encoding="utf-8",
    )
    env = EnvHandle(project_root=tmp_path, extra={"ctest": "/usr/bin/ctest"})

    assert CppBackend().test_command(env) == [
        "/usr/bin/ctest",
        "--test-dir",
        str(tmp_path / "build"),
        "--output-on-failure",
    ]
