from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from code_gen import post_verify  # noqa: E402
from code_gen.git_ops import merge_batch_branch  # noqa: E402
from code_gen.test_runner import TestResult as CodegenTestResult  # noqa: E402
from common.generated_artifacts import (  # noqa: E402
    ensure_generated_artifact_excludes,
    find_persisted_generated_artifact_changes,
    generated_artifact_prompt_rule,
    is_generated_artifact_path,
)
from common.git_utils import GitRunner  # noqa: E402
from common.task_batch import PlannedTask  # noqa: E402


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(repo_path: Path) -> Path:
    repo_path.mkdir(parents=True, exist_ok=True)
    _run_git(repo_path, "init", "-b", "main")
    _run_git(repo_path, "config", "user.email", "coder@example.com")
    _run_git(repo_path, "config", "user.name", "CoderMind Test")
    source = repo_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    _run_git(repo_path, "add", "src/app.py")
    _run_git(repo_path, "commit", "-m", "init")
    return repo_path


def _task() -> PlannedTask:
    return PlannedTask(
        task="Implement the source unit.",
        file_path="src/app.py",
        units_key=["function run"],
        unit_to_code={"function run": "def run(): ..."},
        unit_to_features={"function run": ["Core/run"]},
        subtree="Core",
    )


def test_generated_artifact_path_policy_covers_common_outputs() -> None:
    blocked = [
        "build/CTestTestfile.cmake",
        "src/CMakeFiles/app.dir/main.cpp.o",
        "target/debug/libapp.rlib",
        "node_modules/pkg/index.js",
        "app.pyc",
        "__pycache__/app.cpython-312.pyc",
        "compile_commands.json",
        "dist/bundle.js",
        "build/generated_source.c",
    ]
    allowed = [
        "src/app.py",
        "src/build_config.py",
        "src/env/config.py",
        "src/venv/settings.py",
        "src/.venv/config.py",
        "configs/build/build_contract.c",
        "configs/build/build_contract.h",
        "CMakeLists.txt",
        "Makefile",
        "package.json",
        "Cargo.toml",
    ]

    assert all(is_generated_artifact_path(path) for path in blocked)
    assert not any(is_generated_artifact_path(path) for path in allowed)


def test_generated_artifact_prompt_rule_uses_policy_examples() -> None:
    rule = generated_artifact_prompt_rule("Change `CMakeLists.txt` instead.")

    assert "build/" in rule
    assert "target/" in rule
    assert "node_modules/" in rule
    assert "CTestTestfile.cmake" in rule
    assert "Change `CMakeLists.txt` instead." in rule


def test_local_excludes_prevent_git_add_a_from_staging_outputs(tmp_path: Path) -> None:
    repo_path = _init_repo(tmp_path / "repo")

    assert ensure_generated_artifact_excludes(repo_path)
    (repo_path / "build").mkdir()
    (repo_path / "build" / "CTestTestfile.cmake").write_text("generated\n", encoding="utf-8")
    source = repo_path / "src" / "feature.py"
    source.write_text("def feature():\n    return 2\n", encoding="utf-8")
    build_contract = repo_path / "configs" / "build" / "build_contract.c"
    build_contract.parent.mkdir(parents=True)
    build_contract.write_text("int build_contract(void) { return 0; }\n", encoding="utf-8")

    _run_git(repo_path, "add", "-A")
    status = _run_git(repo_path, "status", "--porcelain").stdout

    assert "src/feature.py" in status
    assert "configs/build/build_contract.c" in status
    assert "build/CTestTestfile.cmake" not in status


def test_local_excludes_upgrade_existing_managed_block(tmp_path: Path) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    exclude_path = repo_path / ".git" / "info" / "exclude"
    exclude_path.write_text(
        "# keep\n"
        "# BEGIN CoderMind generated artifact hygiene\n"
        "old-output/\n"
        "# END CoderMind generated artifact hygiene\n"
        "# tail\n",
        encoding="utf-8",
    )

    assert ensure_generated_artifact_excludes(repo_path)
    content = exclude_path.read_text(encoding="utf-8")

    assert "# keep" in content
    assert "# tail" in content
    assert "old-output/" not in content
    assert "build/" in content
    assert "target/" in content


def test_persisted_generated_artifact_changes_are_reported(tmp_path: Path) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    _run_git(repo_path, "checkout", "-b", "batch/generated-artifact")
    (repo_path / "build").mkdir()
    (repo_path / "build" / "CTestTestfile.cmake").write_text("generated\n", encoding="utf-8")

    _run_git(repo_path, "add", "-f", "build/CTestTestfile.cmake")
    staged = find_persisted_generated_artifact_changes(repo_path, base_ref="main")

    assert [(change.path, change.scope) for change in staged] == [
        ("build/CTestTestfile.cmake", "staged")
    ]

    _run_git(repo_path, "commit", "-m", "bad artifact")
    committed = find_persisted_generated_artifact_changes(repo_path, base_ref="main")

    assert [(change.path, change.scope) for change in committed] == [
        ("build/CTestTestfile.cmake", "branch")
    ]


def test_post_verify_rejects_persisted_generated_artifacts_before_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    _run_git(repo_path, "checkout", "-b", "batch/generated-artifact")
    (repo_path / "build").mkdir()
    (repo_path / "build" / "CTestTestfile.cmake").write_text("generated\n", encoding="utf-8")
    _run_git(repo_path, "add", "-f", "build/CTestTestfile.cmake")
    _run_git(repo_path, "commit", "-m", "bad artifact")

    monkeypatch.setattr(
        post_verify,
        "run_project_tests",
        lambda *_args, **_kwargs: pytest.fail("tests should not run"),
    )

    passed, output = post_verify.post_verify(repo_path, _task())

    assert not passed
    assert "Generated build/dependency/cache artifacts" in output
    assert "build/CTestTestfile.cmake" in output


def test_post_verify_allows_untracked_ignored_build_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    _run_git(repo_path, "checkout", "-b", "batch/generated-artifact")
    ensure_generated_artifact_excludes(repo_path)
    (repo_path / "build").mkdir()
    (repo_path / "build" / "CTestTestfile.cmake").write_text("generated\n", encoding="utf-8")

    class FakeBackend:
        name = "go"
        display_name = "Go"

    monkeypatch.setattr(
        post_verify,
        "resolve_test_backend",
        lambda **_kwargs: FakeBackend(),
    )
    monkeypatch.setattr(
        post_verify,
        "run_project_tests",
        lambda *_args, **_kwargs: CodegenTestResult(
            success=True,
            return_code=0,
            output="ok\n",
            test_files=[],
            passed=1,
        ),
    )

    passed, output = post_verify.post_verify(repo_path, _task())

    assert passed
    assert output == "passed=1 failed=0 errors=0 skipped=0"


def test_post_verify_runs_project_tests_for_docs_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    docs_task = PlannedTask(
        task="Update README.",
        file_path="README.md",
        units_key=["README"],
        unit_to_code={"README": "# Demo"},
        unit_to_features={"README": ["Docs/readme"]},
        task_type="project_docs",
    )

    class FakeBackend:
        name = "cpp"
        display_name = "C++"

    monkeypatch.setattr(
        post_verify,
        "resolve_test_backend",
        lambda **_kwargs: FakeBackend(),
    )

    calls = {"count": 0}

    def fake_run_project_tests(*_args, **_kwargs):
        calls["count"] += 1
        return CodegenTestResult(
            success=False,
            return_code=8,
            output="ctest failed after README update",
            test_files=[],
            failed=1,
        )

    monkeypatch.setattr(post_verify, "run_project_tests", fake_run_project_tests)

    passed, output = post_verify.post_verify(repo_path, docs_task)

    assert calls["count"] == 1
    assert not passed
    assert "ctest failed after README update" in output


def test_merge_batch_rejects_committed_generated_artifacts(tmp_path: Path) -> None:
    repo_path = _init_repo(tmp_path / "repo")
    _run_git(repo_path, "checkout", "-b", "batch/generated-artifact")
    (repo_path / "build").mkdir()
    (repo_path / "build" / "CTestTestfile.cmake").write_text("generated\n", encoding="utf-8")
    _run_git(repo_path, "add", "-f", "build/CTestTestfile.cmake")
    _run_git(repo_path, "commit", "-m", "bad artifact")

    success, error = merge_batch_branch(
        GitRunner(str(repo_path)),
        "batch/generated-artifact",
        "task-id",
        file_path="src/app.py",
        units=["function run"],
    )

    assert not success
    assert error is not None
    assert "Generated build/dependency/cache artifacts" in error
    assert (
        _run_git(repo_path, "branch", "--show-current").stdout.strip()
        == "batch/generated-artifact"
    )