import os
import sys
from dataclasses import dataclass

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from code_gen import post_verify as post_verify_module
from code_gen.test_runner import TestResult as RunnerTestResult
from common.task_batch import PlannedTask


@dataclass
class _Backend:
    name: str = "go"
    display_name: str = "Go"


def test_post_verify_keeps_non_python_project_tests_unscoped(monkeypatch, tmp_path):
    source = tmp_path / "internal" / "store" / "store.go"
    test = tmp_path / "internal" / "store" / "store_test.go"
    source.parent.mkdir(parents=True)
    source.write_text("package store\n", encoding="utf-8")
    test.write_text("package store\n", encoding="utf-8")

    task = PlannedTask(
        task="Implement store",
        file_path="internal/store/store.go",
        units_key=["Store"],
        unit_to_code={"Store": ""},
        unit_to_features={"Store": []},
    )

    calls = {}

    def fake_resolve_test_backend(valid_files=None, repo_path=None):
        calls["valid_files"] = valid_files
        return _Backend()

    def fake_run_project_tests(repo_root, test_files=None, timeout=300, extra_args=None, env=None, backend=None):
        calls["run_test_files"] = test_files
        calls["backend"] = backend
        return RunnerTestResult(success=True, return_code=0, output="ok", test_files=test_files or [])

    monkeypatch.setattr(post_verify_module, "resolve_test_backend", fake_resolve_test_backend)
    monkeypatch.setattr(post_verify_module, "run_project_tests", fake_run_project_tests)

    passed, summary = post_verify_module.post_verify(tmp_path, task)

    assert passed is True
    assert "passed=0" in summary
    assert calls["valid_files"] == ["internal/store/store_test.go"]
    assert calls["run_test_files"] is None
    assert calls["backend"].name == "go"
