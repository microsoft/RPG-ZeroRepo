from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from code_gen import batch_prompts  # noqa: E402
from code_gen import context_collector  # noqa: E402
from code_gen import static_checks  # noqa: E402
from code_gen import test_runner  # noqa: E402
from common.execution_state import BatchExecutionState  # noqa: E402
from common.task_batch import PlannedTask  # noqa: E402
from decoder_lang import EnvHandle, TestRunResult as BackendTestRunResult, get_backend  # noqa: E402
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
    assert "go test -v ./..." in prompt
    assert "Read `go.mod`" in prompt
    assert "go get <module>" in prompt
    assert "python3 -m pytest" not in prompt
    # Go must not be instructed to manage Python deps. requirements.txt may now
    # appear only inside the explicit FORBIDDEN list, never as an instruction.
    assert "Append the package to `requirements.txt`" not in prompt
    assert "Update `requirements.txt`" not in prompt


def test_cpp_codegen_prompt_injects_cpp_context(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "cpp")
    task = _task("src/tasklite_cli/task.cpp")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "Language: C++" in prompt
    assert "Source extension: `.cpp`" in prompt
    assert "C++17" in prompt
    assert "mapfile -d" in prompt
    assert "sources < <(find ." in prompt
    assert "No C++ source files found" in prompt
    assert "PYTEST_SUMMARY: syntax check passed" in prompt
    # Non-Python projects get the strengthened prohibition, not the legacy line.
    assert "NOT Python" in prompt
    assert "Do NOT create ANY `.py` file" in prompt
    assert "conftest.py" in prompt
    assert "standalone translation units" in prompt
    assert "create or update a matching header" in prompt
    assert "Do NOT edit or commit generated build, dependency, cache" in prompt
    assert "python3 -m pytest" not in prompt


def test_cpp_codegen_prompt_aligns_cmake_command_with_post_verify(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "cpp")
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
    task = _task("src/tasklite_cli/task.cpp")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "cmake -S . -B build" in prompt
    assert "cmake --build build" in prompt
    assert "ctest --test-dir build --output-on-failure" in prompt


def test_cpp_syntax_prompt_expands_repo_include_path(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "cpp")
    task = _task("src/tasklite_cli/task.cpp")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert '-I "$PWD"' in prompt
    assert "-I '$PWD'" not in prompt


def test_resume_prompt_includes_agent_pytest_summary() -> None:
    prompt = batch_prompts.build_resume_prompt(
        "Original prompt",
        attempt_number=2,
        failure_reason="Post-verification failed",
        last_test_output="1 failed in 0.10s",
        sub_agent_claimed_pass=True,
        agent_pytest_summary="1 passed in 0.10s",
    )

    assert "PYTEST_SUMMARY line `1 passed in 0.10s`" in prompt
    assert "{agent_summary_repr}" not in prompt


def test_non_python_integration_prompt_uses_native_entry_point(monkeypatch, tmp_path: Path) -> None:
    # Regression for the bug where every language was told "Do NOT create
    # main.py", planting a Python file name into Go/JS/C projects.
    _set_language(monkeypatch, tmp_path, "go")
    task = PlannedTask(
        task="Add the cross-module integration tests.",
        file_path="<INTEGRATION_TEST>",
        units_key=["Core_integration_tests"],
        unit_to_code={"Core_integration_tests": ""},
        unit_to_features={"Core_integration_tests": ["Feature/path"]},
        subtree="Core",
        task_type="integration_test",
    )

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "main.go" in prompt          # native entry point referenced
    assert "create main.py" not in prompt  # no Python file name planted


def test_javascript_codegen_prompt_forbids_python_files(monkeypatch, tmp_path: Path) -> None:
    _set_language(monkeypatch, tmp_path, "javascript")
    task = _task("src/store.js")

    prompt = batch_prompts.build_tdd_prompt(_state(task), task, tmp_path)

    assert "Language: JavaScript" in prompt
    assert "npm test" in prompt
    assert "Do NOT create ANY `.py` file" in prompt
    assert "python3 -m pytest" not in prompt


def test_api_summary_uses_backend_for_non_python(monkeypatch, tmp_path: Path) -> None:
    # Regression: _build_api_summary previously hardcoded the Python backend,
    # so a Go/Rust/TS project's API signatures (used by test-writing batches)
    # came back empty. It must resolve the project backend and render via
    # backend.format_signature for non-Python.
    _set_language(monkeypatch, tmp_path, "go")
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "store.go").write_text(
        "package store\n\n"
        "type Store struct{ path string }\n\n"
        "func NewStore(path string) *Store { return &Store{path: path} }\n\n"
        "func (s *Store) Save(id int) error { return nil }\n",
        encoding="utf-8",
    )

    summary = batch_prompts._build_api_summary(tmp_path, ["internal/store.go"])

    assert "internal/store.go" in summary
    # Go declarations surface (not an empty Python-parsed result).
    assert "Store" in summary
    assert "NewStore" in summary
    # No Python "def " rendering leaked in.
    assert "def NewStore" not in summary


def test_dependency_context_base_class_summary_uses_backend(tmp_path: Path) -> None:
    # Regression: _format_dependency_context previously parsed base-class code
    # with the Python backend, so a Go/Rust base class surfaced as a
    # "parse error — read file directly" line instead of its real
    # struct/method summary. The backend must be resolved from the file path.
    from code_gen import prompts  # noqa: PLC0415

    ctx = {
        "base_classes": {
            "base_classes": [
                {
                    "file_path": "internal/base.go",
                    "code": (
                        "package store\n\n"
                        "type Store struct{ path string }\n\n"
                        "func (s *Store) Save(id int) error { return nil }\n"
                    ),
                    "subclasses": {},
                }
            ]
        }
    }

    summary = prompts._format_dependency_context(ctx)

    assert "`Store` in `internal/base.go`" in summary
    assert "Save" in summary
    # The Python-backend fallback line must not appear for valid Go code.
    assert "parse error" not in summary


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


def test_c_backend_syntax_command_includes_repo_root(tmp_path: Path) -> None:
    source = tmp_path / "src" / "task.c"
    source.parent.mkdir()
    source.write_text('#include "src/task.h"\nint task_count(void) { return 0; }\n', encoding="utf-8")
    (tmp_path / "src" / "task.h").write_text("int task_count(void);\n", encoding="utf-8")
    env = EnvHandle(project_root=tmp_path, extra={"cc": "cc"})

    command = get_backend("c").test_command(env)

    assert command[:4] == ["cc", "-std=c99", "-I", str(tmp_path)]
    assert str(source) in command


def test_cpp_backend_syntax_command_includes_repo_root(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "repository_layout.cpp"
    source.parent.mkdir()
    source.write_text(
        '#include "configs/repository_layout.hpp"\nint layout_count() { return 0; }\n',
        encoding="utf-8",
    )
    (tmp_path / "configs" / "repository_layout.hpp").write_text(
        "int layout_count();\n",
        encoding="utf-8",
    )
    env = EnvHandle(project_root=tmp_path, extra={"cxx": "c++"})

    command = get_backend("cpp").test_command(env)

    assert command[:4] == ["c++", "-std=c++17", "-I", str(tmp_path)]
    assert str(source) in command


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

    monkeypatch.setattr(run_batch, "resolve_test_backend", lambda *_a, **_k: FakeBackend())
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

    monkeypatch.setattr(run_batch, "resolve_test_backend", lambda *_a, **_k: FakeBackend())
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


def test_run_batch_loop_honors_max_batches(monkeypatch) -> None:
    calls: list[int] = []

    def fake_run_batch(**_kwargs):
        calls.append(len(calls) + 1)
        return {
            "success": True,
            "type": "batch_complete",
            "batch_id": f"batch-{len(calls)}",
            "attempts_used": 1,
            "total_duration": 0,
            "stats": {"completed": len(calls), "total": 10, "failed": 0},
        }

    monkeypatch.setattr(run_batch, "run_batch", fake_run_batch)
    args = SimpleNamespace(
        merge_file=False,
        max_units=0,
        agent_timeout=1,
        max_batches=2,
        json=True,
    )

    assert run_batch._run_loop(args) == 0
    assert calls == [1, 2]
