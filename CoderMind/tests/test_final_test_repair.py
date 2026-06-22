from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from code_gen import final_validation as fv  # noqa: E402
from code_gen.test_runner import TestResult as _TestResult  # noqa: E402


def _fail_result(output: str = "FAILED tests/test_x.py::t - assert ...") -> _TestResult:
    return _TestResult(
        success=False,
        return_code=1,
        output=output,
        test_files=[],
        passed=10,
        failed=1,
    )


def _pass_result() -> _TestResult:
    return _TestResult(
        success=True,
        return_code=0,
        output="",
        test_files=[],
        passed=11,
        failed=0,
    )


class _Backend:
    name = "python"
    display_name = "Python"


def _patch_common(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fv, "ensure_on_main", lambda *_a, **_k: None)
    monkeypatch.setattr(fv, "GitRunner", lambda *_a, **_k: object())
    monkeypatch.setattr(fv, "resolve_test_backend", lambda *_a, **_k: _Backend())
    monkeypatch.setattr(fv, "ensure_deps_installed", lambda *_a, **_k: None)
    monkeypatch.setattr(fv, "get_dev_python", lambda *_a, **_k: "python3")
    # build_batch_pytest_cmd is imported lazily inside the loop.
    import code_gen.batch_prompts as bp
    monkeypatch.setattr(bp, "build_batch_pytest_cmd", lambda *_a, **_k: "pytest -q")


def test_final_test_repairs_failing_suite(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    runs = {"n": 0}

    def fake_run_tests(*_a, **_k):
        runs["n"] += 1
        # First run fails, second (post-repair) passes.
        return _fail_result() if runs["n"] == 1 else _pass_result()

    dispatched = {"n": 0, "prompt": None}

    def fake_dispatch(prompt, repo_path, timeout=0, purpose=""):
        dispatched["n"] += 1
        dispatched["prompt"] = prompt
        dispatched["purpose"] = purpose
        return "BATCH_RESULT: PASS", None

    monkeypatch.setattr(fv, "run_project_tests", fake_run_tests)
    monkeypatch.setattr(fv, "dispatch_sub_agent", fake_dispatch)
    # Skip smoke step on the success path for this unit test.
    monkeypatch.setattr(fv, "save_stage_result", lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "smoke_test", type(sys)("smoke_test"))
    sys.modules["smoke_test"].run_smoke_test = lambda: type(
        "S", (), {"findings": [], "to_dict": lambda self: {"findings": [], "error_count": 0}}
    )()

    out = fv.final_test(repo_path=tmp_path, max_repair_iters=2)

    assert out["success"] is True
    assert out["final_test_repair_attempts"] == 1
    assert out["final_test_repaired"] is True
    assert dispatched["n"] == 1
    assert dispatched["purpose"] == "final_test_repair"
    assert "Do NOT delete, skip, or weaken any test." in dispatched["prompt"]


def test_final_test_fails_loudly_on_zero_tests_executed(monkeypatch, tmp_path):
    # A no-op final test (the go-test-found-no-packages case): exit-0 but zero
    # tests executed. It must fail with a clear diagnostic and must NOT dispatch
    # a code-repair agent (which cannot fix a "no tests ran" state).
    _patch_common(monkeypatch, tmp_path)

    def fake_run_tests(*_a, **_k):
        return _TestResult(
            success=False, return_code=0, output="",
            test_files=[], passed=0, failed=0,
        )

    dispatched = {"n": 0}

    def fake_dispatch(*_a, **_k):
        dispatched["n"] += 1
        return "BATCH_RESULT: PASS", None

    monkeypatch.setattr(fv, "run_project_tests", fake_run_tests)
    monkeypatch.setattr(fv, "dispatch_sub_agent", fake_dispatch)
    monkeypatch.setattr(fv, "save_stage_result", lambda *_a, **_k: None)

    out = fv.final_test(repo_path=tmp_path, max_repair_iters=2)

    assert out["success"] is False
    assert out["no_tests_executed"] is True
    assert dispatched["n"] == 0


def test_final_test_repair_bounded_when_still_failing(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    def fake_run_tests(*_a, **_k):
        return _fail_result()  # always fails

    dispatched = {"n": 0}

    def fake_dispatch(prompt, repo_path, timeout=0, purpose=""):
        dispatched["n"] += 1
        return "BATCH_RESULT: PASS", None

    monkeypatch.setattr(fv, "run_project_tests", fake_run_tests)
    monkeypatch.setattr(fv, "dispatch_sub_agent", fake_dispatch)
    monkeypatch.setattr(fv, "save_stage_result", lambda *_a, **_k: None)

    out = fv.final_test(repo_path=tmp_path, max_repair_iters=2)

    assert out["success"] is False
    # Bounded: exactly max_repair_iters dispatches, no infinite loop.
    assert dispatched["n"] == 2
    assert out["final_test_repair_attempts"] == 2
    assert out["final_test_repaired"] is False


def test_final_test_no_repair_when_first_pass(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(fv, "run_project_tests", lambda *_a, **_k: _pass_result())
    monkeypatch.setattr(fv, "save_stage_result", lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "smoke_test", type(sys)("smoke_test"))
    sys.modules["smoke_test"].run_smoke_test = lambda: type(
        "S", (), {"findings": [], "to_dict": lambda self: {"findings": [], "error_count": 0}}
    )()

    def fake_dispatch(*_a, **_k):
        raise AssertionError("repair must not be dispatched when tests pass")

    monkeypatch.setattr(fv, "dispatch_sub_agent", fake_dispatch)

    out = fv.final_test(repo_path=tmp_path, max_repair_iters=2)

    assert out["success"] is True
    assert "final_test_repair_attempts" not in out


def test_final_test_fails_when_smoke_test_crashes(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(fv, "run_project_tests", lambda *_a, **_k: _pass_result())

    saved = {}

    def fake_save_stage_result(name, data):
        saved[name] = data

    monkeypatch.setattr(fv, "save_stage_result", fake_save_stage_result)
    monkeypatch.setitem(sys.modules, "smoke_test", type(sys)("smoke_test"))

    def crash_smoke_test():
        raise FileNotFoundError("python")

    sys.modules["smoke_test"].run_smoke_test = crash_smoke_test

    out = fv.final_test(repo_path=tmp_path, max_repair_iters=2)

    assert out["success"] is False
    assert out["errors"] == 1
    assert "Smoke test failed to run" in out["smoke_test_error"]
    assert saved["final_test"]["success"] is False
    assert saved["smoke_test"]["error_count"] == 1
