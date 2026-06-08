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
import sys
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
        assert states[1].reason == "type=warning"

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
