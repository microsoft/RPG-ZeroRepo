from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _assert_atomic_result(path: Path, expected: dict) -> None:
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_validate_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import validate

    path = tmp_path / "rpg_edit_validate.json"
    monkeypatch.setattr(validate, "RPG_EDIT_VALIDATE_FILE", path)
    result = {"type": "error", "error_code": "rpg_not_found", "message": "missing"}
    validate._write_validate_result(result)
    _assert_atomic_result(path, result)


def test_locate_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import locate

    path = tmp_path / "rpg_edit_locate.json"
    monkeypatch.setattr(locate, "RPG_EDIT_LOCATE_FILE", path)
    result = {"type": "candidates", "query": "service", "results": []}
    locate._write_locate_result(result)
    _assert_atomic_result(path, result)


def test_code_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import code

    path = tmp_path / "rpg_edit_code_result.json"
    monkeypatch.setattr(code, "RPG_EDIT_CODE_RESULT_FILE", path)
    result = {"type": "code_applied", "success": True, "files_modified": ["src/a.py"]}
    code._write_code_result(result)
    _assert_atomic_result(path, result)


def test_apply_result_merges_split_phases_atomically(tmp_path, monkeypatch):
    from rpg_edit import apply

    path = tmp_path / "rpg_edit_apply_result.json"
    monkeypatch.setattr(apply, "RPG_EDIT_APPLY_RESULT_FILE", path)
    first = apply._record_apply_result(
        {"type": "rpg_updated"},
        backup_timestamp="123",
        backups={"rpg": "/tmp/rpg.backup"},
        applied_features=[{"node_id": "feature-1", "action": "modified"}],
        before_state={"head_branch": "rpg-edit/demo", "head_commit": "abc"},
    )
    second = apply._record_apply_result(
        {"type": "dep_refreshed", "dep_graph_refreshed": True},
        backup_timestamp="123",
    )

    assert first["rollback_path"] == "/tmp/rpg.backup"
    assert second["applied_features"] == first["applied_features"]
    assert second["backups"] == first["backups"]
    assert second["before_state"] == first["before_state"]
    assert "--rollback 123" in second["rollback_command"]
    assert "--rollback-branch rpg-edit/demo" in second["rollback_command"]
    _assert_atomic_result(path, second)


def test_review_result_is_atomic(tmp_path, monkeypatch):
    from rpg_edit import review

    path = tmp_path / "rpg_edit_review_result.json"
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", path)
    result = {"type": "skipped", "reason": "impact too small"}
    review._write_review_result(result)
    _assert_atomic_result(path, result)


def test_impact_review_skips_full_pytest_when_no_affected_tests(tmp_path, monkeypatch):
    from rpg_edit import review
    import code_gen.test_runner as test_runner
    import run_batch
    import smoke_test

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "code_changes": [{"file_path": "src/demo/new_module.py", "change_type": "add"}],
    }))
    events = []
    prompts = []

    monkeypatch.setattr(run_batch, "_setup_codegen_environment", lambda path: events.append("setup"))
    monkeypatch.setattr(
        test_runner,
        "run_pytest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full pytest must not run")),
    )
    monkeypatch.setattr(
        smoke_test,
        "run_smoke_test",
        lambda path: SimpleNamespace(to_dict=lambda: {"success": True, "layers": {}}),
    )

    def dispatch(prompt, repo_path, **kwargs):
        events.append("dispatch")
        prompts.append(prompt)
        return "REVIEW_RESULT: PASS", None

    monkeypatch.setattr(run_batch, "dispatch_sub_agent", dispatch)

    result = review.impact_review(plan_path, None, tmp_path, max_iterations=3)

    assert result["success"] is True
    assert len(result["iterations"]) == 1
    assert events == ["setup", "dispatch"]
    assert "No affected tests discovered; skip pytest" in prompts[0]
    assert "Do not perform mobile checks" in prompts[0]
    assert "The controller already ran the advisory smoke scan" in prompts[0]
    assert "smoke_test.py" not in prompts[0]


def test_impact_review_runs_only_existing_affected_tests(tmp_path, monkeypatch):
    from rpg_edit import review
    import code_gen.test_runner as test_runner
    import run_batch
    import smoke_test

    source = tmp_path / "src" / "demo" / "widget.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    dev_python = tmp_path / ".venv_dev" / "bin" / "python"
    dev_python.parent.mkdir(parents=True)
    dev_python.write_text("")
    test_file = tmp_path / "tests" / "test_widget.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_widget():\n    assert True\n")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "code_changes": [{"file_path": "src/demo/widget.py", "change_type": "modify"}],
    }))
    pytest_calls = []

    monkeypatch.setattr(run_batch, "_setup_codegen_environment", lambda path: None)

    def run_targeted(repo_path, *, test_files, **kwargs):
        pytest_calls.append(test_files)
        return test_runner.TestResult(True, 0, "1 passed", test_files, passed=1)

    monkeypatch.setattr(test_runner, "run_pytest", run_targeted)
    monkeypatch.setattr(
        smoke_test,
        "run_smoke_test",
        lambda path: SimpleNamespace(to_dict=lambda: {"success": True, "layers": {}}),
    )
    prompts = []

    def dispatch(prompt, *args, **kwargs):
        prompts.append(prompt)
        return "REVIEW_RESULT: PASS", None

    monkeypatch.setattr(run_batch, "dispatch_sub_agent", dispatch)

    result = review.impact_review(plan_path, None, tmp_path, max_iterations=1)

    assert result["success"] is True
    assert pytest_calls == [["tests/test_widget.py"], ["tests/test_widget.py"]]
    assert f"{dev_python} -m pytest" in prompts[0]
    assert "The controller already ran the advisory smoke scan" in prompts[0]
    assert "smoke_test.py" not in prompts[0]


def test_code_agent_defers_smoke_to_impact_review():
    from rpg_edit.code import _build_initial_prompt

    prompt = _build_initial_prompt({}, "nodes", "impact", [])

    assert "smoke_test.py" not in prompt
    assert "python3 -m pytest" in prompt


def test_dev_venv_installs_packaging_tools(tmp_path, monkeypatch):
    import code_gen.test_runner as test_runner

    installed = []
    monkeypatch.setattr(test_runner, "get_dev_python", lambda path: None)
    monkeypatch.setattr(
        test_runner, "get_dev_venv_path", lambda path: tmp_path / ".venv_dev",
    )
    monkeypatch.setattr(test_runner.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        test_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        test_runner,
        "install_packages_into_venv",
        lambda packages, path: installed.extend(packages) or (True, packages),
    )

    created, _ = test_runner.ensure_dev_venv(tmp_path)

    assert created is True
    assert installed == ["pytest", "pytest-timeout", "setuptools"]