#!/usr/bin/env python3
"""Phase 1 feature construction facade orchestrator."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from common.paths import FEATURE_BUILD_FILE as _FEATURE_BUILD_FILE
from common.paths import FEATURE_SPEC_FILE as _FEATURE_SPEC_FILE
from common.paths import FEATURE_TREE_FILE as _FEATURE_TREE_FILE

_SCRIPTS_DIR = Path(__file__).resolve().parent

FEATURE_SPEC_FILE = _FEATURE_SPEC_FILE
FEATURE_BUILD_FILE = _FEATURE_BUILD_FILE
FEATURE_TREE_FILE = _FEATURE_TREE_FILE

_LOGICAL_PATHS = {
    "feature_spec": ".cmind/data/feature_spec.json",
    "feature_build": ".cmind/data/feature_build.json",
    "feature_refactor": ".cmind/data/feature_tree.json",
}

_RESET_REASONS = {"forced", "upstream rebuilt"}

_REQUIRED_FEATURE_SPEC_FIELDS = (
    "meta",
    "repository_name",
    "repository_purpose",
    "background_and_overview",
    "functional_requirements",
    "non_functional_requirements",
)


@dataclass(frozen=True)
class Stage:
    name: str
    build_script: str


STAGES: tuple[Stage, ...] = (
    Stage(name="feature_spec", build_script="feature_spec.py"),
    Stage(name="feature_build", build_script="feature_build.py"),
    Stage(name="feature_refactor", build_script="feature_refactor.py"),
)


@dataclass
class StageState:
    stage: Stage
    type: str = "init"
    message: str = ""
    done: bool = False
    will_run: bool = False
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _resolve_invoker() -> list[str]:
    cmind = shutil.which("cmind")
    if cmind:
        return [cmind, "script"]
    return [sys.executable]


def _script_argv(invoker: list[str], script_name: str) -> list[str]:
    # Use ``.stem`` so the check matches both ``cmind`` (POSIX) and
    # ``cmind.exe`` (Windows packaging) uniformly.
    if Path(invoker[0]).stem == "cmind":
        return [*invoker, script_name]
    return [*invoker, str(_SCRIPTS_DIR / script_name)]


def _run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
    argv = [*_script_argv(invoker, script_name), *extra]
    proc = subprocess.run(argv, check=False)
    return proc.returncode


def _load_json_object(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    except OSError as exc:
        return None, f"cannot read file: {exc}"

    if not isinstance(data, dict):
        return None, "JSON root must be an object"
    return data, None


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _state(stage: Stage, type_: str, message: str, raw: Optional[dict[str, Any]] = None) -> StageState:
    return StageState(
        stage=stage,
        type=type_,
        message=message,
        done=(type_ == "update"),
        raw=raw or {},
    )


def _check_feature_spec(stage: Stage) -> StageState:
    data, error = _load_json_object(FEATURE_SPEC_FILE)
    logical = _LOGICAL_PATHS[stage.name]
    if error == "missing":
        return _state(stage, "init", f"{logical} is missing")
    if error:
        return _state(stage, "warning", f"{logical} is not complete: {error}")

    missing = [field for field in _REQUIRED_FEATURE_SPEC_FIELDS if not _has_content(data.get(field))]
    if missing:
        return _state(
            stage,
            "warning",
            f"{logical} is missing required fields: {', '.join(missing)}",
            {"missing_fields": missing},
        )
    return _state(stage, "update", f"{logical} is valid")


def _check_feature_build(stage: Stage) -> StageState:
    data, error = _load_json_object(FEATURE_BUILD_FILE)
    logical = _LOGICAL_PATHS[stage.name]
    if error == "missing":
        return _state(stage, "init", f"{logical} is missing")
    if error:
        return _state(stage, "warning", f"{logical} is not complete: {error}")
    return _state(stage, "update", f"{logical} is valid JSON", {"keys": sorted(data.keys())})


def _check_feature_refactor(stage: Stage) -> StageState:
    data, error = _load_json_object(FEATURE_TREE_FILE)
    logical = _LOGICAL_PATHS[stage.name]
    if error == "missing":
        return _state(stage, "init", f"{logical} is missing")
    if error:
        return _state(stage, "warning", f"{logical} is not complete: {error}")

    components = data.get("components")
    if isinstance(components, (list, dict)) and components:
        return _state(stage, "update", f"{logical} has components")
    return _state(stage, "warning", f"{logical} has no non-empty components collection")


def _check_stage(stage: Stage) -> StageState:
    if stage.name == "feature_spec":
        return _check_feature_spec(stage)
    if stage.name == "feature_build":
        return _check_feature_build(stage)
    if stage.name == "feature_refactor":
        return _check_feature_refactor(stage)
    return _state(stage, "error", f"unknown stage: {stage.name}")


def probe() -> list[StageState]:
    return [_check_stage(stage) for stage in STAGES]


def decide(states: list[StageState], force: bool) -> None:
    cascade = False
    for state in states:
        if force:
            state.will_run = True
            state.reason = "forced"
            continue
        if cascade:
            state.will_run = True
            state.reason = "upstream rebuilt"
            continue
        if state.type == "update":
            state.will_run = False
            state.reason = "up-to-date"
        else:
            state.will_run = True
            state.reason = f"type={state.type}"
            cascade = True


_GLYPH = {"update": "✓", "init": "·", "warning": "!", "error": "✗"}


def _format_table(states: list[StageState]) -> str:
    rows = ["Stage             Type      Done   Action"]
    rows.append("-" * 52)
    for state in states:
        glyph = _GLYPH.get(state.type, "?")
        action = "run" if state.will_run else "skip"
        rows.append(
            f"{state.stage.name:<16}  {glyph} {state.type:<8}  "
            f"{'yes' if state.done else 'no ':<3}   {action}"
        )
    return "\n".join(rows)


def _print_probe_summary(states: list[StageState]) -> None:
    done = sum(1 for state in states if state.done)
    total = len(states)
    first_pending = next((state.stage.name for state in states if not state.done), None)
    print(f"Feature construction progress: {done}/{total} stages complete.")
    if first_pending:
        print(f"Next pending stage: {first_pending}")
    else:
        print("All Phase 1 stages are up-to-date.")
    print()
    print(_format_table(states))


def _check_only_payload(states: list[StageState]) -> dict[str, Any]:
    done = sum(1 for state in states if state.done)
    total = len(states)
    next_pending = next((state.stage.name for state in states if not state.done), None)
    return {
        "total": total,
        "done": done,
        "completed": done,
        "next": next_pending,
        "stages": [
            {
                "name": state.stage.name,
                "type": state.type,
                "message": state.message,
                "done": state.done,
                "will_run": state.will_run,
                "reason": state.reason,
            }
            for state in states
        ],
    }


def _emit_check_only_json(states: list[StageState]) -> None:
    print(json.dumps(_check_only_payload(states), indent=2))


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _build_args_for(stage: Stage, args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if stage.name == "feature_spec":
        # ``feature_spec.py`` accepts the standard facade flags only;
        # inline requirement text reaches it via the slash-command layer
        # (which calls ``feature_spec.py --input-text "..."`` directly
        # before resuming the orchestrator).
        if args.force:
            extra.append("--force")
        if args.verbose:
            extra.append("--verbose")
        if args.no_trajectory:
            extra.append("--no-trajectory")
    elif stage.name == "feature_build":
        extra.extend(["--mode", "step1"])
        if args.review_threshold is not None:
            extra.extend(["--review-threshold", _format_number(args.review_threshold)])
        if args.review_max_iterations is not None:
            extra.extend(["--review-max-iterations", str(args.review_max_iterations)])
        if args.verbose:
            extra.append("--verbose")
        if args.no_trajectory:
            extra.append("--no-trajectory")
    elif stage.name == "feature_refactor":
        if args.max_iter_refactor is not None:
            extra.extend(["--max-iterations", str(args.max_iter_refactor)])
        if args.verbose:
            extra.extend(["--log-level", "DEBUG"])
        if args.no_trajectory:
            extra.append("--no-trajectory")
    return extra


def _debug_args_for(stage: Stage) -> list[str]:
    if stage.name == "feature_build":
        return ["--verbose"]
    if stage.name == "feature_refactor":
        return ["--log-level", "DEBUG"]
    return []


def _target_output_for(stage: Stage) -> Optional[Path]:
    return {
        "feature_spec": FEATURE_SPEC_FILE,
        "feature_build": FEATURE_BUILD_FILE,
        "feature_refactor": FEATURE_TREE_FILE,
    }.get(stage.name)


def _should_reset_output(state: StageState) -> bool:
    if not state.will_run or _target_output_for(state.stage) is None:
        return False
    return state.reason in _RESET_REASONS or state.type not in {"init", "update"}


def _reset_output_if_needed(state: StageState) -> None:
    target = _target_output_for(state.stage)
    if target is not None and _should_reset_output(state):
        target.unlink(missing_ok=True)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="feature_construct.py",
        description="Run the Phase 1 feature construction pipeline with automatic resume.",
    )
    parser.add_argument("--check-only", action="store_true", help="Probe all stages and exit.")
    parser.add_argument("--json", action="store_true", help="With --check-only, emit JSON.")
    parser.add_argument("--force", action="store_true", help="Rebuild all Phase 1 stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--verbose", action="store_true", help="Forward native verbose logging flags.")
    parser.add_argument("--no-trajectory", action="store_true", help="Disable trajectory recording where supported.")
    parser.add_argument(
        "--max-iter-refactor",
        type=int,
        default=None,
        metavar="N",
        help="Override feature_refactor.py --max-iterations.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        metavar="N",
        help="Forward to feature_build.py --review-threshold.",
    )
    parser.add_argument(
        "--review-max-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Forward to feature_build.py --review-max-iterations.",
    )
    return parser.parse_args(argv)


def _install_sigint_handler() -> None:
    def _handle(signum: int, frame: Any) -> None:
        print("\n[feature_construct] interrupted — rerun `cmind script feature_construct.py` to resume.")
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle)


def _print_failure_hint(
    invoker: list[str],
    stage: Stage,
    rc: int,
    *,
    phase: str,
) -> None:
    """Print recovery hints to stderr after a stage fails.

    Commands are rendered using the current *invoker* so the hint stays
    correct whether ``cmind`` is on ``$PATH`` (``cmind script <name>``)
    or the Python fallback is in use (``python <abspath>``).
    """
    debug_args = _debug_args_for(stage)

    def _fmt(name: str, *extra: str) -> str:
        # Render the command using the basename of the invoker so
        # users see e.g. ``cmind script ...`` or ``python ...`` rather
        # than the resolved absolute path that subprocess actually uses.
        argv = _script_argv(invoker, name)
        display = [Path(argv[0]).name, *argv[1:], *extra]
        return " ".join(display)

    print(file=sys.stderr)
    print(f"X {stage.name} {phase} failed (exit {rc})", file=sys.stderr)
    print(f"  Resume :  {_fmt('feature_construct.py')}", file=sys.stderr)
    print(f"  Debug  :  {_fmt(stage.build_script, *debug_args)}", file=sys.stderr)
    print(f"  Status :  {_fmt('feature_construct.py', '--check-only')}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _install_sigint_handler()
    invoker = _resolve_invoker()

    states = probe()
    decide(states, force=args.force)

    if args.check_only:
        if args.json:
            _emit_check_only_json(states)
        else:
            _print_probe_summary(states)
        return 0

    runnable = [state for state in states if state.will_run]
    if not runnable:
        print("All Phase 1 stages are already complete — nothing to do.")
        print("Use `--force` to rebuild from scratch.")
        print("Next: `/cmind.plan` to build the Repository Planning Graph (RPG).")
        return 0

    print(f"Feature construction pipeline: {len(runnable)} of {len(states)} stages to run.")
    print(_format_table(states))
    print()

    if args.dry_run:
        for state in runnable:
            cmd = _script_argv(invoker, state.stage.build_script)
            cmd += _build_args_for(state.stage, args)
            print("DRY-RUN >", " ".join(cmd))
        return 0

    started = time.monotonic()
    for state in states:
        if not state.will_run:
            print(f"skip {state.stage.name:<16} ({state.reason})")
            continue

        stage_started = time.monotonic()
        print(f"run  {state.stage.name:<16} {state.stage.build_script} ...")
        _reset_output_if_needed(state)
        rc = _run_stage(invoker, state.stage.build_script, _build_args_for(state.stage, args))
        if rc != 0:
            _print_failure_hint(invoker, state.stage, rc, phase="build")
            return rc

        verify = _check_stage(state.stage)
        if verify.type != "update":
            print(
                f"   verification failed: {verify.type} — {verify.message}",
                file=sys.stderr,
            )
            _print_failure_hint(invoker, state.stage, 1, phase="check")
            return 1

        elapsed = time.monotonic() - stage_started
        print(f"done {state.stage.name:<16} in {elapsed:.1f}s")

    total_elapsed = time.monotonic() - started
    print()
    print(f"Feature construct complete in {total_elapsed:.1f}s.")
    print("Next: `/cmind.plan` to build the Repository Planning Graph (RPG).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
