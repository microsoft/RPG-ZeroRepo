"""Unit tests for the planning orchestrator's pure logic.

Covers the decision rules of ``scripts/plan.py``:

* ``decide()`` cascade behaviour
* probe-result parsing (``_extract_last_json_object``)
* CLI flag wiring for max-iteration overrides
* checker JSON field contracts used by the orchestrator

The build sub-scripts themselves are *not* exercised here because they
would require real LLM calls; this test focuses on deterministic logic.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ``plan.py`` is shipped under ``scripts/`` and not installed as a
# package, so load it via importlib.
_SPEC = importlib.util.spec_from_file_location("plan_orchestrator", _SCRIPTS / "plan.py")
assert _SPEC is not None and _SPEC.loader is not None
plan = importlib.util.module_from_spec(_SPEC)
sys.modules["plan_orchestrator"] = plan
_SPEC.loader.exec_module(plan)


@pytest.fixture(autouse=True)
def activity_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from common.activity_events import ActivityWriter
    monkeypatch.setattr(plan, "ACTIVITY_WRITER", ActivityWriter(tmp_path / "activity", workspace_id="ws_test"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _states(types: list[str]) -> list["plan.StageState"]:
    """Build a list of StageState objects from the given type sequence."""
    assert len(types) == len(plan.STAGES)
    return [
        plan.StageState(stage=stage, type=t, done=(t == "update"))
        for stage, t in zip(plan.STAGES, types)
    ]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# decide() cascade rule
# ---------------------------------------------------------------------------

class TestDecideCascade:
    def test_all_update_means_nothing_runs(self) -> None:
        states = _states(["update"] * 5)
        plan.decide(states, force=False)
        assert [s.will_run for s in states] == [False] * 5

    def test_fresh_workspace_runs_everything(self) -> None:
        states = _states(["init"] * 5)
        plan.decide(states, force=False)
        assert [s.will_run for s in states] == [True] * 5

    def test_partial_resume_runs_from_first_non_update(self) -> None:
        # skeleton + data_flow done, base_classes init, rest init
        states = _states(["update", "update", "init", "init", "init"])
        plan.decide(states, force=False)
        assert [s.will_run for s in states] == [False, False, True, True, True]

    def test_cascade_forces_downstream_even_if_update(self) -> None:
        # Inconsistent state: skeleton needs rebuild but base_classes
        # is "update" (e.g. user manually deleted skeleton.json).
        # Cascade rule must force base_classes to rebuild anyway.
        states = _states(["init", "update", "update", "update", "update"])
        plan.decide(states, force=False)
        assert [s.will_run for s in states] == [True, True, True, True, True]
        # downstream reasons should mention cascade
        assert "upstream" in states[1].reason

    def test_warning_is_treated_as_incomplete(self) -> None:
        # A warning means the artefact violates a cross-stage contract.
        # Rebuild from that stage so bench cannot report a false PASS for
        # a partial plan.
        states = _states(["update", "warning", "update", "update", "update"])
        plan.decide(states, force=False)
        assert [s.will_run for s in states] == [False, True, True, True, True]
        assert states[1].reason == "warning: cross-stage contract violation; rebuild stage and downstream"

    def test_force_runs_everything(self) -> None:
        states = _states(["update"] * 5)
        plan.decide(states, force=True)
        assert all(s.will_run for s in states)
        assert all(s.reason == "forced" for s in states)


# ---------------------------------------------------------------------------
# _extract_last_json_object — tolerant JSON parsing.
# ---------------------------------------------------------------------------

class TestExtractLastJsonObject:
    def test_pure_json(self) -> None:
        obj = plan._extract_last_json_object('{"type": "init"}')
        assert obj == {"type": "init"}

    def test_json_with_trailing_text(self) -> None:
        text = '{"type": "update", "ok": true}\n📸 snapshot abc123'
        obj = plan._extract_last_json_object(text)
        assert obj == {"type": "update", "ok": True}

    def test_json_with_leading_text(self) -> None:
        text = 'Running ...\n{"type": "init"}'
        obj = plan._extract_last_json_object(text)
        assert obj == {"type": "init"}

    def test_takes_last_object_when_multiple(self) -> None:
        text = '{"first": 1}{"type": "update"}'
        obj = plan._extract_last_json_object(text)
        assert obj == {"type": "update"}

    def test_returns_none_on_garbage(self) -> None:
        assert plan._extract_last_json_object("no braces here") is None
        assert plan._extract_last_json_object("{not json}") is None


# ---------------------------------------------------------------------------
# Per-stage max-iter flag wiring.
# ---------------------------------------------------------------------------

class TestBuildArgs:
    def test_max_iter_skeleton_uses_max_iterations_flag(self) -> None:
        ns = plan._parse_args(["--max-iter-skeleton", "7"])
        skeleton = plan.STAGES[0]
        assert skeleton.name == "skeleton"
        args = plan._build_args_for(skeleton, ns)
        assert "--max-iterations" in args
        assert args[args.index("--max-iterations") + 1] == "7"

    def test_max_iter_interfaces_uses_max_file_iterations_flag(self) -> None:
        # design_interfaces.py has a different flag name than the others.
        ns = plan._parse_args(["--max-iter-interfaces", "4"])
        interfaces = next(s for s in plan.STAGES if s.name == "interfaces")
        args = plan._build_args_for(interfaces, ns)
        assert "--max-file-iterations" in args
        assert "--max-iterations" not in args
        assert args[args.index("--max-file-iterations") + 1] == "4"

    def test_tasks_stage_has_no_max_iter_flag(self) -> None:
        # plan_tasks.py takes no iteration count; --max-iter-* must not be
        # forwarded even if some other stage's flag is set.
        ns = plan._parse_args(["--max-iter-skeleton", "9"])
        tasks = next(s for s in plan.STAGES if s.name == "tasks")
        args = plan._build_args_for(tasks, ns)
        assert all(not a.startswith("--max") for a in args)

    def test_verbose_forwarded(self) -> None:
        ns = plan._parse_args(["--verbose"])
        args = plan._build_args_for(plan.STAGES[0], ns)
        assert "--verbose" in args

    def test_no_trajectory_forwarded(self) -> None:
        ns = plan._parse_args(["--no-trajectory"])
        args = plan._build_args_for(plan.STAGES[0], ns)
        assert "--no-trajectory" in args

    def test_stage_timeout_defaults_and_overrides(self) -> None:
        defaults = plan._parse_args([])
        assert plan._stage_timeout_for(plan.STAGES[0], defaults) == 2700
        assert plan._stage_timeout_for(plan.STAGES[3], defaults) == 5400
        assert defaults.no_progress_timeout_sec == 1200

        overrides = plan._parse_args([
            "--stage-timeout-sec", "30",
            "--interfaces-timeout-sec", "60",
            "--terminate-grace-sec", "2",
        ])
        assert plan._stage_timeout_for(plan.STAGES[0], overrides) == 30
        assert plan._stage_timeout_for(plan.STAGES[3], overrides) == 60
        assert overrides.terminate_grace_sec == 2

    @pytest.mark.parametrize(
        "flag",
        ["--stage-timeout-sec", "--interfaces-timeout-sec", "--terminate-grace-sec"],
    )
    def test_timeout_options_must_be_positive(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            plan._parse_args([flag, "0"])

    def test_workspace_config_and_environment_supply_execution_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cmind_dir = tmp_path / ".cmind"
        cmind_dir.mkdir()
        (cmind_dir / "config.toml").write_text(
            "[execution]\n"
            "llm_timeout_sec = 40\n"
            "llm_max_attempts = 1\n"
            "terminate_grace_sec = 4\n"
            "no_progress_timeout_sec = 55\n"
            "[plan]\n"
            "stage_timeout_sec = 80\n"
            "interfaces_timeout_sec = 160\n"
        )
        monkeypatch.setattr(plan, "CMIND_DIR", cmind_dir)
        configured = plan._parse_args([])

        assert configured.llm_timeout_sec == 40
        assert configured.llm_max_attempts == 1
        assert configured.terminate_grace_sec == 4
        assert configured.no_progress_timeout_sec == 55
        assert configured.stage_timeout_sec == 80
        assert configured.interfaces_timeout_sec == 160

        monkeypatch.setenv("CMIND_PLAN_STAGE_TIMEOUT_SEC", "25")
        monkeypatch.setenv("CMIND_LLM_TIMEOUT_SEC", "15")
        overridden = plan._parse_args([])
        assert overridden.stage_timeout_sec == 25
        assert overridden.llm_timeout_sec == 15


class TestPlanLock:
    def test_rejects_second_writer_and_releases_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / ".plan.lock"
        with plan.PlanLock(path) as first:
            first.update(stage="interfaces")
            with pytest.raises(plan.PlanAlreadyRunning) as caught:
                with plan.PlanLock(path):
                    pass
            assert caught.value.metadata["pid"] == os.getpid()
            assert caught.value.metadata["stage"] == "interfaces"

        with plan.PlanLock(path):
            pass

        assert plan.PlanLock.active_metadata(path) is None

    def test_status_probe_reads_active_owner(self, tmp_path: Path) -> None:
        path = tmp_path / ".plan.lock"
        with plan.PlanLock(path) as lock:
            lock.update(stage="tasks", stage_status="running")
            active = plan.PlanLock.active_metadata(path)

        assert active is not None
        assert active["pid"] == os.getpid()
        assert active["stage"] == "tasks"

    def test_direct_interface_stage_rejects_duplicate_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ) -> None:
        design_interfaces = _load_script("design_interfaces")
        lock_path = tmp_path / ".design_interfaces.lock"
        monkeypatch.setattr(design_interfaces, "_INTERFACES_LOCK_PATH", lock_path)
        calls = []

        @design_interfaces._exclusive_interfaces_run
        def guarded() -> int:
            calls.append("ran")
            return 0

        with design_interfaces.ProcessLock(lock_path):
            assert guarded() == design_interfaces._LOCKED_EXIT_CODE

        assert calls == []
        assert "already active" in capsys.readouterr().err


class TestStageSupervision:
    def test_proc_parent_parser_ignores_process_name_spacing(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "123"
        proc_dir.mkdir()
        (proc_dir / "status").write_text(
            "Name:\tworker name with spaces\nState:\tS (sleeping)\nPPid:\t42\n"
        )

        assert plan._read_proc_parent_pid(proc_dir) == 42

    def test_process_signal_falls_back_without_killpg(self, monkeypatch) -> None:
        class FakeProcess:
            pid = 123

            def __init__(self):
                self.actions = []

            def terminate(self):
                self.actions.append("terminate")

            def kill(self):
                self.actions.append("kill")

            def send_signal(self, signum):
                self.actions.append(("signal", signum))

        process = FakeProcess()
        monkeypatch.setattr(plan.os, "name", "nt")

        plan._signal_process_group(process, signal.SIGTERM)
        plan._signal_process_group(process, signal.SIGTERM, force=True)

        assert process.actions == ["terminate", "kill"]

    def test_no_progress_watchdog_stops_stage(self, monkeypatch) -> None:
        class HungProcess:
            pid = 123

            def wait(self, timeout):
                raise subprocess.TimeoutExpired("stage", timeout)

        clock = iter((0.0, 0.0, 2.0, 2.0))
        terminated = []
        monkeypatch.setattr(plan.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(plan.subprocess, "Popen", lambda *args, **kwargs: HungProcess())
        monkeypatch.setattr(
            plan,
            "_terminate_process_group",
            lambda proc, grace: terminated.append((proc.pid, grace)),
        )

        result = plan._run_stage(
            ["cmind", "script"],
            "plan_tasks.py",
            [],
            timeout_sec=100,
            terminate_grace_sec=3,
            no_progress_timeout_sec=1,
        )

        assert result.returncode == 124
        assert result.timed_out is True
        assert result.timeout_reason == "no_progress"
        assert terminated == [(123, 3)]

    def test_termination_escalates_to_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProcess:
            pid = 123
            returncode = None
            wait_calls = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("stage", timeout)
                self.returncode = -signal.SIGKILL
                return self.returncode

            def send_signal(self, signum):
                raise AssertionError(f"unexpected fallback signal {signum}")

            def kill(self):
                raise AssertionError("unexpected direct kill fallback")

        signals = []
        monkeypatch.setattr(plan.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(plan.os, "killpg", lambda pgid, signum: signals.append(signum))

        plan._terminate_process_group(FakeProcess(), 0.1)

        assert signals == [signal.SIGTERM, signal.SIGKILL]

    def test_termination_signals_detached_descendant_groups(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeProcess:
            pid = 123
            returncode = None
            wait_calls = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("stage", timeout)
                self.returncode = -signal.SIGKILL
                return self.returncode

            def kill(self):
                raise AssertionError("unexpected direct kill fallback")

        snapshots = iter(({123, 456}, {456}))
        signals = []
        monkeypatch.setattr(
            plan,
            "_descendant_process_groups",
            lambda pid: set(next(snapshots)),
        )
        monkeypatch.setattr(
            plan,
            "_signal_groups",
            lambda groups, signum: signals.append((set(groups), signum)),
        )

        plan._terminate_process_group(FakeProcess(), 0.1)

        assert signals == [
            ({123, 456}, signal.SIGTERM),
            ({123, 456}, signal.SIGKILL),
        ]

    def test_timeout_with_new_valid_artifact_recovers(self, monkeypatch, capsys) -> None:
        states = _states(["update", "update", "update", "update", "init"])
        monkeypatch.setattr(plan, "probe", lambda invoker: states)
        monkeypatch.setattr(plan, "POST_STEPS", ())
        monkeypatch.setattr(
            plan,
            "_run_stage",
            lambda *args, **kwargs: plan.StageRunResult(124, True, 1.0),
        )
        monkeypatch.setattr(
            plan,
            "_run_check",
            lambda invoker, script: {"type": "update", "message": "valid"},
        )

        args = plan._parse_args(["--stage-timeout-sec", "1"])
        assert plan._run_pipeline(args, ["cmind", "script"], None) == 0
        assert "artifact is valid; continuing" in capsys.readouterr().out

    def test_timeout_with_invalid_artifact_fails(self, monkeypatch) -> None:
        states = _states(["update", "update", "update", "update", "init"])
        monkeypatch.setattr(plan, "probe", lambda invoker: states)
        monkeypatch.setattr(
            plan,
            "_run_stage",
            lambda *args, **kwargs: plan.StageRunResult(124, True, 1.0),
        )
        monkeypatch.setattr(
            plan,
            "_run_check",
            lambda invoker, script: {"type": "init", "message": "missing"},
        )

        args = plan._parse_args(["--stage-timeout-sec", "1"])
        assert plan._run_pipeline(args, ["cmind", "script"], None) == 124

    def test_force_does_not_recover_from_preexisting_artifact(self, monkeypatch) -> None:
        states = _states(["update"] * 5)
        monkeypatch.setattr(plan, "probe", lambda invoker: states)
        monkeypatch.setattr(
            plan,
            "_run_stage",
            lambda *args, **kwargs: plan.StageRunResult(124, True, 1.0),
        )
        monkeypatch.setattr(
            plan,
            "_run_check",
            lambda invoker, script: {"type": "update", "message": "old artifact"},
        )

        args = plan._parse_args(["--force", "--stage-timeout-sec", "1"])
        assert plan._run_pipeline(args, ["cmind", "script"], None) == 124

    def test_pipeline_forwards_llm_execution_budget(self, monkeypatch) -> None:
        states = _states(["update", "update", "update", "update", "init"])
        monkeypatch.setattr(plan, "probe", lambda invoker: states)
        monkeypatch.setattr(plan, "POST_STEPS", ())
        observed = []

        def run_stage(*args, **kwargs):
            observed.append(kwargs["env_overrides"])
            return plan.StageRunResult(0, False, 1.0)

        monkeypatch.setattr(plan, "_run_stage", run_stage)
        monkeypatch.setattr(
            plan,
            "_run_check",
            lambda invoker, script: {"type": "update", "message": "valid"},
        )

        args = plan._parse_args([
            "--llm-timeout-sec", "45",
            "--llm-max-attempts", "1",
        ])
        assert plan._run_pipeline(args, ["cmind", "script"], None) == 0
        assert observed == [{
            "CMIND_LLM_TIMEOUT_SEC": "45",
            "CMIND_LLM_MAX_ATTEMPTS": "1",
        }]

    def test_json_probe_includes_active_run(self, capsys) -> None:
        active = {"pid": 123, "stage": "interfaces", "stage_status": "running"}
        plan._emit_check_only_json(_states(["update"] * 5), active)

        payload = json.loads(capsys.readouterr().out)
        assert payload["active_run"] == active


# ---------------------------------------------------------------------------
# Stage table sanity — guard against silent registry drift.
# ---------------------------------------------------------------------------

class TestCheckerContracts:
    @pytest.mark.parametrize(
        ("script_name", "args"),
        [
            ("check_data_flow", (Path("missing-data-flow.json"), Path("missing-skeleton.json"))),
            ("check_base_classes", (Path("missing-base-classes.json"),)),
        ],
    )
    def test_plan_checkers_emit_type_not_state(self, script_name: str, args: tuple[Path, ...]) -> None:
        checker = _load_script(script_name)
        result = checker.inspect_state(*args)
        assert result["type"] == "init"
        assert "state" not in result

    def test_interfaces_checker_rejects_failed_global_review(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = _load_script("check_interfaces")
        skeleton = tmp_path / "skeleton.json"
        interfaces = tmp_path / "interfaces.json"
        rpg = tmp_path / "rpg.json"
        skeleton.write_text(json.dumps({
            "root": {
                "type": "directory",
                "children": [{
                    "type": "file",
                    "path": "src/app.py",
                    "feature_paths": ["App/run"],
                }],
            },
        }))
        interfaces.write_text(json.dumps({
            "subtrees": {
                "App": {
                    "interfaces": {
                        "src/app.py": {
                            "units": ["function run"],
                            "units_to_features": {"function run": ["App/run"]},
                            "units_to_code": {"function run": "def run(): ..."},
                        },
                    },
                },
            },
            "global_review": {
                "passed": False,
                "feature_orphans_count": 3,
                "orphan_units_count": 2,
            },
        }))
        rpg.write_text(json.dumps({"root": {"node_type": "root", "children": []}}))
        monkeypatch.setattr(checker, "REPO_RPG_FILE", rpg)

        result = checker.check_state(skeleton, interfaces)

        assert result["type"] == "warning"
        assert result["output_valid"] is True
        assert "3 orphan feature(s), 2 orphan unit(s)" in result["message"]


class TestStageRegistry:
    def test_five_stages_in_canonical_order(self) -> None:
        assert [s.name for s in plan.STAGES] == [
            "skeleton",
            "data_flow",
            "base_classes",
            "interfaces",
            "tasks",
        ]

    @pytest.mark.parametrize("stage", plan.STAGES)
    def test_every_stage_has_a_build_and_check_script(self, stage: plan.Stage) -> None:
        assert (_SCRIPTS / stage.build_script).is_file(), stage.build_script
        assert (_SCRIPTS / stage.check_script).is_file(), stage.check_script

    @pytest.mark.parametrize("post_script", plan.POST_STEPS)
    def test_post_step_scripts_exist(self, post_script: str) -> None:
        assert (_SCRIPTS / post_script).is_file(), post_script
