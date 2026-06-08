"""Unit tests for the feature construction orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "feature_construct_orchestrator",
    _SCRIPTS / "feature_construct.py",
)
assert _SPEC is not None and _SPEC.loader is not None
feature_construct = importlib.util.module_from_spec(_SPEC)
sys.modules["feature_construct_orchestrator"] = feature_construct
_SPEC.loader.exec_module(feature_construct)


@pytest.fixture
def artifact_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {
        "feature_spec": tmp_path / "feature_spec.json",
        "feature_build": tmp_path / "feature_build.json",
        "feature_refactor": tmp_path / "feature_tree.json",
    }
    monkeypatch.setattr(feature_construct, "FEATURE_SPEC_FILE", paths["feature_spec"])
    monkeypatch.setattr(feature_construct, "FEATURE_BUILD_FILE", paths["feature_build"])
    monkeypatch.setattr(feature_construct, "FEATURE_TREE_FILE", paths["feature_refactor"])
    return paths


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_feature_spec() -> dict[str, object]:
    return {
        "meta": {
            "generated_at": "2026-05-25",
            "project_types": ["CLI"],
            "primary_language": "python",
            "target_languages": ["python"],
        },
        "repository_name": "sample-cli",
        "repository_purpose": "Build a sample CLI.",
        "background_and_overview": [{"id": "BG-001", "description": "Users need a CLI."}],
        "functional_requirements": [{"id": "FT-001", "name": "CLI", "children": []}],
        "non_functional_requirements": [{"id": "NFR-001", "description": "Fast startup."}],
    }


def _valid_feature_build(language: str = "python") -> dict[str, object]:
    return {
        "feature_tree": {},
        "meta": {"primary_language": language, "target_languages": [language]},
    }


def _valid_feature_tree(
    language: str = "python",
    component_name: str = "core",
) -> dict[str, object]:
    return {
        "components": [{"name": component_name}],
        "meta": {"primary_language": language, "target_languages": [language]},
    }


def _states(types: list[str]) -> list["feature_construct.StageState"]:
    assert len(types) == len(feature_construct.STAGES)
    return [
        feature_construct.StageState(stage=stage, type=t, done=(t == "update"))
        for stage, t in zip(feature_construct.STAGES, types)
    ]


class TestStageRegistry:
    def test_three_stages_in_canonical_order(self) -> None:
        assert [stage.name for stage in feature_construct.STAGES] == [
            "feature_spec",
            "feature_build",
            "feature_refactor",
        ]

    @pytest.mark.parametrize("stage", feature_construct.STAGES)
    def test_every_stage_has_a_build_script(self, stage: "feature_construct.Stage") -> None:
        assert (_SCRIPTS / stage.build_script).is_file(), stage.build_script


class TestCompletionDetection:
    def test_missing_artifacts_are_incomplete(self, artifact_paths: dict[str, Path]) -> None:
        states = feature_construct.probe()
        assert [state.type for state in states] == ["init", "init", "init"]
        assert [state.done for state in states] == [False, False, False]

    def test_valid_artifacts_are_complete(self, artifact_paths: dict[str, Path]) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_json(artifact_paths["feature_build"], _valid_feature_build())
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree())

        states = feature_construct.probe()
        assert [state.type for state in states] == ["update", "update", "update"]
        assert [state.done for state in states] == [True, True, True]

    def test_feature_spec_requires_downstream_fields(self, artifact_paths: dict[str, Path]) -> None:
        spec = _valid_feature_spec()
        spec.pop("functional_requirements")
        _write_json(artifact_paths["feature_spec"], spec)

        state = feature_construct.probe()[0]
        assert state.type == "warning"
        assert state.done is False
        assert "functional_requirements" in state.message

    def test_feature_spec_requires_language_fields(self, artifact_paths: dict[str, Path]) -> None:
        spec = _valid_feature_spec()
        meta = spec["meta"]
        assert isinstance(meta, dict)
        meta.pop("primary_language")
        meta.pop("target_languages")
        _write_json(artifact_paths["feature_spec"], spec)

        state = feature_construct.probe()[0]

        assert state.type == "warning"
        assert state.done is False
        assert "meta.primary_language" in state.message
        assert "meta.target_languages" in state.message

    def test_feature_build_preserves_feature_spec_language(self, artifact_paths: dict[str, Path]) -> None:
        spec = _valid_feature_spec()
        spec["meta"] = {**spec["meta"], "primary_language": "go", "target_languages": ["go"]}
        _write_json(artifact_paths["feature_spec"], spec)
        _write_json(
            artifact_paths["feature_build"],
            {
                "feature_tree": {},
                "meta": {"primary_language": "python", "target_languages": ["python"]},
            },
        )

        state = feature_construct.probe()[1]

        assert state.type == "warning"
        assert state.done is False
        assert "expected 'go'" in state.message

    def test_feature_refactor_preserves_feature_spec_language(self, artifact_paths: dict[str, Path]) -> None:
        spec = _valid_feature_spec()
        spec["meta"] = {**spec["meta"], "primary_language": "go", "target_languages": ["go"]}
        _write_json(artifact_paths["feature_spec"], spec)
        _write_json(
            artifact_paths["feature_refactor"],
            {
                "components": [{"name": "core"}],
                "meta": {"primary_language": "go", "target_languages": []},
            },
        )

        state = feature_construct.probe()[2]

        assert state.type == "warning"
        assert state.done is False
        assert "target_languages" in state.message

    def test_feature_refactor_requires_non_empty_components(self, artifact_paths: dict[str, Path]) -> None:
        _write_json(artifact_paths["feature_refactor"], {"components": []})

        state = feature_construct.probe()[2]
        assert state.type == "warning"
        assert state.done is False
        assert "components" in state.message


class TestCheckOnlyJson:
    def test_json_payload_reports_progress(self, artifact_paths: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_json(artifact_paths["feature_build"], _valid_feature_build())

        rc = feature_construct.main(["--check-only", "--json"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert rc == 0
        assert payload["total"] == 3
        assert payload["done"] == 2
        assert payload["completed"] == 2
        assert payload["next"] == "feature_refactor"
        assert [stage["name"] for stage in payload["stages"]] == [
            "feature_spec",
            "feature_build",
            "feature_refactor",
        ]
        assert [stage["done"] for stage in payload["stages"]] == [True, True, False]
        assert payload["stages"][0]["details"]["primary_language"] == "python"


class TestExecutionReset:
    def test_force_removes_stale_output_sensitive_artifacts_before_stage_invocation(
        self,
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_json(artifact_paths["feature_build"], {"stale": "build"})
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="stale"))
        calls: list[str] = []

        def fake_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            calls.append(script_name)
            if script_name == "feature_spec.py":
                _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
            elif script_name == "feature_build.py":
                assert not artifact_paths["feature_build"].exists()
                _write_json(artifact_paths["feature_build"], _valid_feature_build())
            elif script_name == "feature_refactor.py":
                assert not artifact_paths["feature_refactor"].exists()
                _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="fresh"))
            return 0

        monkeypatch.setattr(feature_construct, "_run_stage", fake_run_stage)

        rc = feature_construct.main(["--force"])

        assert rc == 0
        assert calls == ["feature_spec.py", "feature_build.py", "feature_refactor.py"]

    def test_cascade_removes_stale_downstream_artifacts_before_stage_invocation(
        self,
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec = _valid_feature_spec()
        spec.pop("repository_purpose")
        _write_json(artifact_paths["feature_spec"], spec)
        _write_json(artifact_paths["feature_build"], {"stale": "build"})
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="stale"))
        calls: list[str] = []

        def fake_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            calls.append(script_name)
            if script_name == "feature_spec.py":
                _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
            elif script_name == "feature_build.py":
                assert not artifact_paths["feature_build"].exists()
                _write_json(artifact_paths["feature_build"], _valid_feature_build())
            elif script_name == "feature_refactor.py":
                assert not artifact_paths["feature_refactor"].exists()
                _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="fresh"))
            return 0

        monkeypatch.setattr(feature_construct, "_run_stage", fake_run_stage)

        rc = feature_construct.main([])

        assert rc == 0
        assert calls == ["feature_spec.py", "feature_build.py", "feature_refactor.py"]

    def test_invalid_output_sensitive_artifact_is_removed_before_stage_invocation(
        self,
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_text(artifact_paths["feature_build"], "{")
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="stale"))
        calls: list[str] = []

        def fake_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            calls.append(script_name)
            if script_name == "feature_build.py":
                assert not artifact_paths["feature_build"].exists()
                _write_json(artifact_paths["feature_build"], _valid_feature_build())
            elif script_name == "feature_refactor.py":
                assert not artifact_paths["feature_refactor"].exists()
                _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="fresh"))
            return 0

        monkeypatch.setattr(feature_construct, "_run_stage", fake_run_stage)

        rc = feature_construct.main([])

        assert rc == 0
        assert calls == ["feature_build.py", "feature_refactor.py"]

    def test_all_up_to_date_skip_path_does_not_remove_artifacts(
        self,
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_json(artifact_paths["feature_build"], _valid_feature_build())
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree())

        def fail_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            pytest.fail(f"unexpected stage run: {script_name}")

        monkeypatch.setattr(feature_construct, "_run_stage", fail_run_stage)

        rc = feature_construct.main([])

        assert rc == 0
        assert artifact_paths["feature_build"].exists()
        assert artifact_paths["feature_refactor"].exists()

    @pytest.mark.parametrize("argv", [["--check-only"], ["--check-only", "--json"]])
    def test_check_only_does_not_remove_artifacts_or_run_stages(
        self,
        argv: list[str],
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_text(artifact_paths["feature_spec"], "{")
        _write_json(artifact_paths["feature_build"], {"stale": "build"})
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="stale"))

        def fail_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            pytest.fail(f"unexpected stage run: {script_name}")

        monkeypatch.setattr(feature_construct, "_run_stage", fail_run_stage)

        rc = feature_construct.main(argv)

        assert rc == 0
        assert artifact_paths["feature_spec"].exists()
        assert artifact_paths["feature_build"].exists()
        assert artifact_paths["feature_refactor"].exists()

    def test_dry_run_does_not_remove_artifacts_or_run_stages(
        self,
        artifact_paths: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_json(artifact_paths["feature_spec"], _valid_feature_spec())
        _write_text(artifact_paths["feature_build"], "{")
        _write_json(artifact_paths["feature_refactor"], _valid_feature_tree(component_name="stale"))

        def fail_run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
            pytest.fail(f"unexpected stage run: {script_name}")

        monkeypatch.setattr(feature_construct, "_run_stage", fail_run_stage)

        rc = feature_construct.main(["--dry-run"])
        captured = capsys.readouterr()

        assert rc == 0
        assert "DRY-RUN >" in captured.out
        assert artifact_paths["feature_build"].exists()
        assert artifact_paths["feature_refactor"].exists()


class TestDecideCascade:
    def test_all_update_means_nothing_runs(self) -> None:
        states = _states(["update", "update", "update"])
        feature_construct.decide(states, force=False)
        assert [state.will_run for state in states] == [False, False, False]

    def test_fresh_workspace_runs_everything(self) -> None:
        states = _states(["init", "init", "init"])
        feature_construct.decide(states, force=False)
        assert [state.will_run for state in states] == [True, True, True]

    def test_partial_resume_runs_from_first_incomplete_stage(self) -> None:
        states = _states(["update", "init", "update"])
        feature_construct.decide(states, force=False)
        assert [state.will_run for state in states] == [False, True, True]
        assert "upstream" in states[2].reason

    def test_upstream_warning_cascades_to_downstream_update(self) -> None:
        states = _states(["warning", "update", "update"])
        feature_construct.decide(states, force=False)
        assert [state.will_run for state in states] == [True, True, True]
        assert "upstream" in states[1].reason

    def test_force_runs_everything(self) -> None:
        states = _states(["update", "update", "update"])
        feature_construct.decide(states, force=True)
        assert [state.will_run for state in states] == [True, True, True]
        assert [state.reason for state in states] == ["forced", "forced", "forced"]


class TestBuildArgs:
    def test_feature_build_forwards_review_options(self) -> None:
        ns = feature_construct._parse_args([
            "--review-threshold",
            "99",
            "--review-max-iterations",
            "4",
        ])
        stage = next(stage for stage in feature_construct.STAGES if stage.name == "feature_build")
        args = feature_construct._build_args_for(stage, ns)

        assert args[:2] == ["--mode", "step1"]
        assert args[args.index("--review-threshold") + 1] == "99"
        assert args[args.index("--review-max-iterations") + 1] == "4"

    def test_feature_refactor_maps_facade_iteration_flag(self) -> None:
        ns = feature_construct._parse_args(["--max-iter-refactor", "7"])
        stage = next(stage for stage in feature_construct.STAGES if stage.name == "feature_refactor")
        args = feature_construct._build_args_for(stage, ns)

        assert "--max-iterations" in args
        assert args[args.index("--max-iterations") + 1] == "7"
        assert "--max-iter-refactor" not in args

    def test_verbose_and_no_trajectory_use_native_stage_names(self) -> None:
        ns = feature_construct._parse_args(["--verbose", "--no-trajectory"])
        build_stage = next(stage for stage in feature_construct.STAGES if stage.name == "feature_build")
        refactor_stage = next(stage for stage in feature_construct.STAGES if stage.name == "feature_refactor")

        build_args = feature_construct._build_args_for(build_stage, ns)
        refactor_args = feature_construct._build_args_for(refactor_stage, ns)

        assert "--verbose" in build_args
        assert "--no-trajectory" in build_args
        assert "--log-level" in refactor_args
        assert refactor_args[refactor_args.index("--log-level") + 1] == "DEBUG"
        assert "--no-trajectory" in refactor_args

    def test_feature_spec_receives_supported_facade_options(self) -> None:
        # ``feature_spec.py`` (the new Python+LLMClient pipeline) accepts
        # the standard facade flags --force / --verbose / --no-trajectory.
        ns = feature_construct._parse_args(["--force", "--verbose", "--no-trajectory"])
        stage = next(stage for stage in feature_construct.STAGES if stage.name == "feature_spec")
        args = feature_construct._build_args_for(stage, ns)
        assert "--force" in args
        assert "--verbose" in args
        assert "--no-trajectory" in args
        # Build/refactor-only options must NOT leak into feature_spec.
        ns2 = feature_construct._parse_args([
            "--review-threshold", "99",
            "--review-max-iterations", "3",
            "--max-iter-refactor", "5",
        ])
        args2 = feature_construct._build_args_for(stage, ns2)
        assert "--review-threshold" not in args2
        assert "--review-max-iterations" not in args2
        assert "--max-iter-refactor" not in args2
        assert "--max-iterations" not in args2
