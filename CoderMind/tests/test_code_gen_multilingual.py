from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from code_gen import batch_prompts  # noqa: E402
from code_gen import context_collector  # noqa: E402
from code_gen import static_checks  # noqa: E402
from code_gen import test_runner  # noqa: E402
from common.execution_state import BatchExecutionState  # noqa: E402
from common.task_batch import PlannedTask  # noqa: E402
from decoder_lang import EnvHandle, TestRunResult as BackendTestRunResult  # noqa: E402
import run_batch  # noqa: E402


def _task(file_path: str) -> PlannedTask:
    return PlannedTask(
        task="Implement the target unit.",
        file_path=file_path,
        units_key=["Unit"],
        unit_to_code={"Unit": "interface code"},
        unit_to_features={"Unit": ["Feature/path"]},
        subtree="Core",
    )


def _state(task: PlannedTask) -> BatchExecutionState:
    state = BatchExecutionState(
        batch_id=task.task_id,
        file_path=task.file_path,
        subtree=task.subtree,
    )
    state.test_prompt = "Write focused tests."
    state.code_prompt = "Implement the code."
    return state


def _set_language(monkeypatch, tmp_path: Path, language: str) -> None:
    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(
        json.dumps({"meta": {"primary_language": language, "target_languages": [language]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_prompts, "FEATURE_SPEC_FILE", spec_path)
    monkeypatch.setattr(batch_prompts, "REPO_RPG_FILE", tmp_path / "missing_rpg.json")
    monkeypatch.setattr(static_checks, "FEATURE_SPEC_FILE", spec_path)
    monkeypatch.setattr(static_checks, "REPO_RPG_FILE", tmp_path / "missing_rpg.json")
    monkeypatch.setattr(test_runner, "FEATURE_SPEC_FILE", spec_path)
    monkeypatch.setattr(test_runner, "REPO_RPG_FILE", tmp_path / "missing_rpg.json")


def test_python_codegen_prompt_keeps_pytest_command(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "python")
    task = _task("src/app/service.py")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "Language: Python" in prompt
    assert "python3 -m pytest" in prompt
    assert "Read `requirements.txt`" in prompt
    assert "Use snake_case file and directory names" in prompt


def test_go_codegen_prompt_uses_go_test(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "go")
    task = _task("internal/task/store.go")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "Language: Go" in prompt
    assert "go test ./..." in prompt
    assert "Read `go.mod`" in prompt
    assert "go get <module>" in prompt
    assert "python3 -m pytest" not in prompt
    assert "requirements.txt" not in prompt


def test_cpp_codegen_prompt_injects_cpp_context(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "cpp")
    task = _task("src/tasklite_cli/task.cpp")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "Language: C++" in prompt
    assert "Source extension: `.cpp`" in prompt
    assert "C++17" in prompt
    assert "Do NOT introduce Python-specific files" in prompt
    assert "python3 -m pytest" not in prompt


def test_run_project_tests_uses_backend_command(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class FakeBackend:
        """Minimal backend for exercising generic test execution."""

        name = "go"
        display_name = "Go"

        def detect_env(self, repo_root: Path) -> EnvHandle:
            return EnvHandle(project_root=repo_root, runtime_executable="fake-go")

        def test_command(self, env: EnvHandle) -> list[str]:
            return [env.runtime_executable or "go", "test", "./..."]

        def parse_test_output(self, raw: str, exit_code: int) -> BackendTestRunResult:
            return BackendTestRunResult(
                status="passed" if exit_code == 0 else "failed",
                exit_code=exit_code,
                passed_count=1,
                raw_output=raw,
            )

    class FakeProcess:
        """Subprocess stand-in that records the command and succeeds."""

        returncode = 0
        pid = 12345

        def __init__(self, cmd, **kwargs):
            seen["cmd"] = cmd
            seen["cwd"] = kwargs.get("cwd")

        def communicate(self, timeout=None):
            seen["timeout"] = timeout
            return "ok\n", ""

    monkeypatch.setattr(test_runner.subprocess, "Popen", FakeProcess)

    result = test_runner.run_project_tests(
        tmp_path,
        timeout=12,
        backend=FakeBackend(),
    )

    assert result.success
    assert result.passed == 1
    assert seen["cmd"] == ["fake-go", "test", "./..."]
    assert seen["cwd"] == tmp_path
    assert seen["timeout"] == 12


def test_static_completeness_uses_c_backend(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "c")
    source = tmp_path / "src" / "task.c"
    source.parent.mkdir()
    source.write_text("int task_count(void) { return 0; }\n", encoding="utf-8")

    assert static_checks.static_completeness_check(["src/task.c"], tmp_path) == []

    source.write_text("int task_count(void) { abort(); }\n", encoding="utf-8")

    issues = static_checks.static_completeness_check(["src/task.c"], tmp_path)

    assert issues == ["PLACEHOLDER: src/task.c contains placeholder code"]


def test_write_interface_skeletons_keeps_c_code_unchanged(tmp_path: Path) -> None:
    interfaces_path = tmp_path / "interfaces.json"
    interfaces_path.write_text(
        json.dumps({
            "meta": {"primary_language": "c", "target_languages": ["c"]},
            "subtrees": {
                "Core": {
                    "interfaces": {
                        "src/task.c": {"file_code": "int task_count(void);\n"}
                    }
                }
            },
        }),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"

    result = context_collector.write_interface_skeletons(interfaces_path, repo)

    assert result == {"written": ["src/task.c"], "skipped": []}
    assert (repo / "src" / "task.c").read_text(encoding="utf-8") == "int task_count(void);\n"


def test_run_batch_skips_python_env_for_non_python(monkeypatch, tmp_path: Path) -> None:
    class FakeBackend:
        name = "go"
        display_name = "Go"

    monkeypatch.setattr(run_batch, "resolve_test_backend", lambda: FakeBackend())
    monkeypatch.setattr(
        run_batch,
        "ensure_dev_venv",
        lambda _repo: (_ for _ in ()).throw(AssertionError("venv should not run")),
    )
    monkeypatch.setattr(
        run_batch,
        "ensure_deps_installed",
        lambda _repo: (_ for _ in ()).throw(AssertionError("deps should not run")),
    )

    run_batch._setup_codegen_environment(tmp_path)


def test_run_batch_keeps_python_env_setup(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeBackend:
        name = "python"
        display_name = "Python"

    monkeypatch.setattr(run_batch, "resolve_test_backend", lambda: FakeBackend())
    monkeypatch.setattr(
        run_batch,
        "ensure_dev_venv",
        lambda _repo: (calls.append("venv") or False, tmp_path / ".venv_dev"),
    )
    monkeypatch.setattr(
        run_batch,
        "ensure_deps_installed",
        lambda _repo: calls.append("deps"),
    )

    run_batch._setup_codegen_environment(tmp_path)

    assert calls == ["venv", "deps"]
