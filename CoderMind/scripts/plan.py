#!/usr/bin/env python3
"""Plan Orchestrator Script.

Run the full RPG planning pipeline in one shot, replacing the
five sequential slash-commands ``/cmind.build_skeleton`` →
``/cmind.build_data_flow`` → ``/cmind.design_base_classes`` →
``/cmind.design_interfaces`` → ``/cmind.plan_tasks``.

Design contract
---------------

This script is intentionally non-interactive.  All user-facing
"continue / restart / exit" decisions belong to the slash-command
template (``templates/commands/plan.md``); this script only
implements the three execution modes the template chooses from:

* ``--check-only [--json]`` — probe every stage's ``check_*.py``
  script and print a progress report, then exit 0.  This is how
  the template inspects the workspace before prompting the user.

* (default) — *resume mode*: skip stages whose check returns
  ``type == "update"``; run every other stage in dependency order.
  Once any stage gets (re)built, all downstream stages are forced
  to rebuild too, so up- and down-stream artifacts never drift
  apart.

* ``--force`` — discard the current progress and rebuild all five
  stages from scratch.

Only one mutating Plan run may own a workspace at a time.  Every stage
runs in a supervised process group with a hard deadline; if a timed-out
stage produced an artifact that changes from incomplete to valid, the
pipeline records the recovery and continues from that checkpoint.

Sub-scripts are invoked via ``cmind script <name>`` when the
``cmind`` CLI is on ``$PATH`` (so each stage gets its own
``logs/<stage>.log`` and inner-git snapshot, courtesy of the
dispatcher).  When ``cmind`` is missing, the script falls back
to a direct ``python <bundled-script-path>`` invocation.

Exit codes
----------

* 0    — pipeline finished successfully (or nothing to do)
* 2    — argument error
* 75   — another Plan run owns the workspace lock
* 124  — a stage exceeded its deadline without a newly valid artifact
* 130  — interrupted with Ctrl-C
* N    — exit code of the first failing sub-stage (passed through)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common.activity_events import ActivityWriter, activity_environment, record_activity
from common.paths import CMIND_DIR, LOGS_DIR
from common.process_lock import LockHeldError as PlanAlreadyRunning
from common.process_lock import ProcessLock as PlanLock
from common.progress import PROGRESS_FILE_ENV, read_progress, update_progress

ACTIVITY_WRITER: ActivityWriter | None = None

# Sub-scripts live in the same directory as this file (bundled under
# cmind_cli/core_pack/scripts/ in the installed wheel).
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PLAN_LOCK_PATH = LOGS_DIR / ".plan.lock"
_PLAN_PROGRESS_PATH = LOGS_DIR / "plan_progress.json"
_CHECK_TIMEOUT_SEC = 120
_STAGE_TIMEOUT_EXIT_CODE = 124
_LOCKED_EXIT_CODE = 75
_DEFAULT_STAGE_TIMEOUT_SEC = 2700
_DEFAULT_INTERFACES_TIMEOUT_SEC = 5400
_DEFAULT_TERMINATE_GRACE_SEC = 15
_DEFAULT_LLM_TIMEOUT_SEC = 900
_DEFAULT_LLM_MAX_ATTEMPTS = 2
_DEFAULT_NO_PROGRESS_TIMEOUT_SEC = 1200


# ---------------------------------------------------------------------------
# Stage table — single source of truth for the pipeline.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    """One step of the planning pipeline."""

    name: str                       # short id used by the user and the template
    build_script: str               # the .py runner under scripts/
    check_script: str               # the .py probe under scripts/
    max_iter_flag: Optional[str]    # CLI flag used by the build script, if any


STAGES: tuple[Stage, ...] = (
    Stage(
        name="skeleton",
        build_script="build_skeleton.py",
        check_script="check_skeleton.py",
        max_iter_flag="--max-iterations",
    ),
    Stage(
        name="data_flow",
        build_script="build_data_flow.py",
        check_script="check_data_flow.py",
        max_iter_flag="--max-iterations",
    ),
    Stage(
        name="base_classes",
        build_script="design_base_classes.py",
        check_script="check_base_classes.py",
        max_iter_flag="--max-iterations",
    ),
    Stage(
        name="interfaces",
        build_script="design_interfaces.py",
        check_script="check_interfaces.py",
        # design_interfaces uses a different flag name than the others.
        max_iter_flag="--max-file-iterations",
    ),
    Stage(
        name="tasks",
        build_script="plan_tasks.py",
        check_script="check_tasks.py",
        max_iter_flag=None,  # plan_tasks.py takes no iteration count.
    ),
)

# Post-pipeline helper scripts.  Always run on a successful pipeline
# so the user gets an up-to-date summary + visualization.
POST_STEPS: tuple[str, ...] = (
    "summary_skeleton.py",
    "generate_viz.py",
)


# ---------------------------------------------------------------------------
# Subprocess helpers.
# ---------------------------------------------------------------------------

def _resolve_invoker() -> list[str]:
    """Return the argv prefix used to invoke a sub-script.

    Prefer the ``cmind`` CLI so the dispatcher tees each stage's
    output to ``~/.cmind/workspaces/<hash>/logs/<stem>.log`` and
    snapshots the inner git repo automatically.  Fall back to a
    direct python invocation when ``cmind`` is not on ``$PATH``.
    """
    cmind = shutil.which("cmind")
    if cmind:
        return [cmind, "script"]
    return [sys.executable]  # script path appended by caller


def _script_argv(invoker: list[str], script_name: str) -> list[str]:
    """Build the argv needed to invoke ``script_name`` via ``invoker``."""
    if Path(invoker[0]).stem == "cmind":
        return [*invoker, script_name]
    return [*invoker, str(_SCRIPTS_DIR / script_name)]


def _run_check(invoker: list[str], script_name: str) -> dict[str, Any]:
    """Run a check_*.py script and parse its JSON stdout.

    The check scripts print exactly one JSON object on stdout when
    invoked with ``--json``.  We capture it without printing to the
    parent terminal so the user is not flooded by 5 raw JSON blobs
    during probing.  ``--json`` is the unified contract across all
    ``check_*.py`` scripts; ``check_skeleton.py`` accepts it as a
    no-op for compatibility.
    """
    argv = [sys.executable, str(_SCRIPTS_DIR / script_name), "--json"]
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **activity_environment()},
            check=False,
            timeout=_CHECK_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        return {"type": "error", "message": f"cannot invoke {argv[0]}: {exc}"}
    except subprocess.TimeoutExpired:
        return {
            "type": "error",
            "message": f"{script_name} timed out after {_CHECK_TIMEOUT_SEC}s",
        }

    text = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if not text:
        return {
            "type": "error",
            "message": f"{script_name} produced no output (exit {proc.returncode})",
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some checks may emit human-readable lines before the JSON
        # object; take the last brace-balanced block.
        last_obj = _extract_last_json_object(text)
        if last_obj is not None:
            return last_obj
        return {
            "type": "error",
            "message": f"{script_name} returned non-JSON output",
        }


def _extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort: pull the last ``{...}`` block out of ``text``.

    Robust to unmatched ``}`` characters that may appear in surrounding
    log lines (e.g. error messages quoting JSON fragments). When depth
    would go negative, reset the parser state so a later, well-formed
    object can still be captured.
    """
    depth = 0
    start = -1
    last: Optional[str] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                # Stray ``}`` outside any object — ignore and stay reset.
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                last = text[start : i + 1]
    if last is None:
        return None
    try:
        obj = json.loads(last)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class StageRunResult:
    """Outcome from one supervised planning stage."""

    returncode: int
    timed_out: bool
    duration_sec: float
    timeout_reason: str | None = None


def _signal_process_group(
    proc: subprocess.Popen[Any], signum: int, *, force: bool = False,
) -> None:
    if os.name != "nt" and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(proc.pid), signum)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        if force:
            proc.kill()
        elif os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signum)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _descendant_process_groups(root_pid: int) -> set[int]:
    """Snapshot POSIX process groups rooted below ``root_pid``."""
    if os.name == "nt" or not Path("/proc").is_dir():
        return set()
    parents: dict[int, int] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        parent_pid = _read_proc_parent_pid(proc_dir)
        if parent_pid is None:
            continue
        parents[int(proc_dir.name)] = parent_pid
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid in parents.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    own_group = os.getpgrp()
    groups: set[int] = set()
    for pid in descendants:
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
        if pgid != own_group:
            groups.add(pgid)
    return groups


def _read_proc_parent_pid(proc_dir: Path) -> int | None:
    """Read the stable ``PPid`` field without parsing space-sensitive stat."""
    try:
        lines = (proc_dir / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("PPid:"):
            continue
        try:
            return int(line.split(":", 1)[1].strip())
        except ValueError:
            return None
    return None


def _signal_groups(groups: set[int], signum: int) -> None:
    if os.name == "nt" or not hasattr(os, "killpg"):
        return
    for pgid in sorted(groups, reverse=True):
        try:
            os.killpg(pgid, signum)
        except (ProcessLookupError, PermissionError, OSError):
            continue


def _terminate_process_group(
    proc: subprocess.Popen[Any], grace_sec: float,
) -> None:
    """Terminate a stage process group without any unbounded waits."""
    if proc.poll() is not None:
        return
    wait_timeout = max(float(grace_sec), 0.1)
    groups = _descendant_process_groups(proc.pid)
    if groups:
        _signal_groups(groups, signal.SIGTERM)
    else:
        _signal_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=wait_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    remaining_groups = groups | _descendant_process_groups(proc.pid)
    if remaining_groups:
        _signal_groups(
            remaining_groups,
            getattr(signal, "SIGKILL", signal.SIGTERM),
        )
    else:
        _signal_process_group(
            proc,
            getattr(signal, "SIGKILL", signal.SIGTERM),
            force=True,
        )
    try:
        proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _run_stage(
    invoker: list[str],
    script_name: str,
    extra: list[str],
    *,
    timeout_sec: int,
    terminate_grace_sec: int,
    env_overrides: dict[str, str] | None = None,
    progress_path: Path | None = None,
    no_progress_timeout_sec: int | None = None,
) -> StageRunResult:
    """Run a stage in its own process group with a hard deadline."""
    argv = [*_script_argv(invoker, script_name), *extra]
    started = time.monotonic()
    process_options: dict[str, Any] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    proc = subprocess.Popen(
        argv,
        env={
            **os.environ,
            **activity_environment(),
            **(env_overrides or {}),
        },
        **process_options,
    )
    try:
        deadline = started + timeout_sec
        last_progress_at = started
        last_progress_mtime = _progress_mtime_ns(progress_path)
        timeout_reason: str | None = None
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                timeout_reason = "hard_timeout"
                break
            if (
                no_progress_timeout_sec is not None
                and now - last_progress_at >= no_progress_timeout_sec
            ):
                timeout_reason = "no_progress"
                break
            wait_for = min(5.0, remaining)
            if no_progress_timeout_sec is not None:
                wait_for = min(
                    wait_for,
                    max(0.1, no_progress_timeout_sec - (now - last_progress_at)),
                )
            try:
                returncode = proc.wait(timeout=wait_for)
                return StageRunResult(
                    returncode=returncode,
                    timed_out=False,
                    duration_sec=time.monotonic() - started,
                )
            except subprocess.TimeoutExpired:
                current_mtime = _progress_mtime_ns(progress_path)
                if current_mtime != last_progress_mtime:
                    last_progress_mtime = current_mtime
                    last_progress_at = time.monotonic()
        _terminate_process_group(proc, terminate_grace_sec)
        return StageRunResult(
            returncode=_STAGE_TIMEOUT_EXIT_CODE,
            timed_out=True,
            duration_sec=time.monotonic() - started,
            timeout_reason=timeout_reason,
        )
    except BaseException:
        _terminate_process_group(proc, terminate_grace_sec)
        raise


def _progress_mtime_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Progress probing and decision logic.
# ---------------------------------------------------------------------------

@dataclass
class StageState:
    stage: Stage
    type: str = "error"     # init | update | warning | error
    message: str = ""
    done: bool = False
    will_run: bool = False
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def probe(invoker: list[str]) -> list[StageState]:
    """Run every check_*.py and return a parallel list of states."""
    states: list[StageState] = []
    for stage in STAGES:
        result = _run_check(invoker, stage.check_script)
        type_ = str(result.get("type", "error"))
        states.append(
            StageState(
                stage=stage,
                type=type_,
                message=str(result.get("message", "")),
                done=(type_ == "update"),
                raw=result,
            )
        )
    return states


def decide(states: list[StageState], force: bool) -> None:
    """Mark each state's ``will_run`` / ``reason`` in place.

    Rule: only ``type == "update"`` is complete. ``type == "warning"`` is
    explicitly NOT a completed state: it means the artifact exists but
    violates a cross-stage contract, so this stage is rerun and downstream
    artifacts are rebuilt from it. ``--force`` flips every stage to
    ``will_run``.
    """
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
            if state.type == "warning":
                state.reason = "warning: cross-stage contract violation; rebuild stage and downstream"
            else:
                state.reason = f"type={state.type}"
            cascade = True


# ---------------------------------------------------------------------------
# Pretty-printing.
# ---------------------------------------------------------------------------

_GLYPH = {"update": "✓", "init": "·", "warning": "!", "error": "✗"}


def _format_table(states: list[StageState]) -> str:
    rows = ["Stage           Type     Done   Action"]
    rows.append("-" * 50)
    for s in states:
        glyph = _GLYPH.get(s.type, "?")
        action = "run" if s.will_run else "skip"
        rows.append(
            f"{s.stage.name:<14}  {glyph} {s.type:<7}  "
            f"{'yes' if s.done else 'no ':<3}   {action}"
        )
    return "\n".join(rows)


def _print_probe_summary(
    states: list[StageState], active_run: dict[str, Any] | None = None,
) -> None:
    done = sum(1 for s in states if s.done)
    total = len(states)
    first_pending = next((s.stage.name for s in states if not s.done), None)
    print(f"Planning progress: {done}/{total} stages complete.")
    if first_pending:
        print(f"Next pending stage: {first_pending}")
    else:
        print("All stages are up-to-date.")
    if active_run:
        print(
            "Active pipeline: "
            f"pid={active_run.get('pid', 'unknown')}, "
            f"stage={active_run.get('stage', 'unknown')}, "
            f"status={active_run.get('stage_status', active_run.get('status', 'unknown'))}"
        )
    print()
    print(_format_table(states))


def _emit_check_only_json(
    states: list[StageState], active_run: dict[str, Any] | None = None,
) -> None:
    done = sum(1 for s in states if s.done)
    total = len(states)
    next_pending = next((s.stage.name for s in states if not s.done), None)
    payload = {
        "total": total,
        "done": done,
        "next": next_pending,
        "active_run": active_run,
        "stages": [
            {
                "name": s.stage.name,
                "type": s.type,
                "message": s.message,
                "done": s.done,
            }
            for s in states
        ],
    }
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Build-args assembly.
# ---------------------------------------------------------------------------

def _build_args_for(stage: Stage, args: argparse.Namespace) -> list[str]:
    """Collect CLI args to forward to ``stage.build_script``."""
    extra: list[str] = []
    if stage.max_iter_flag is not None:
        value = getattr(args, f"max_iter_{stage.name}", None)
        if value is not None:
            extra.extend([stage.max_iter_flag, str(value)])
    if args.verbose:
        extra.append("--verbose")
    if args.no_trajectory:
        extra.append("--no-trajectory")
    return extra


def _stage_timeout_for(stage: Stage, args: argparse.Namespace) -> int:
    if stage.name == "interfaces":
        return args.interfaces_timeout_sec
    return args.stage_timeout_sec


def _config_default(
    section: str, key: str, env_name: str, fallback: int,
) -> int:
    raw = os.environ.get(env_name)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    try:
        with (CMIND_DIR / "config.toml").open("rb") as handle:
            config = tomllib.load(handle)
        value = (config.get(section) or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return fallback


def _execution_default(key: str, env_name: str, fallback: int) -> int:
    return _config_default("execution", key, env_name, fallback)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plan.py",
        description=(
            "Run the full RPG planning pipeline (skeleton → data_flow → "
            "base_classes → interfaces → tasks) with automatic resume."
        ),
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Probe every stage and print progress, then exit. No build runs.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="With --check-only, emit a machine-readable JSON progress report.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore current progress and rebuild every stage from scratch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Forward --verbose to every sub-script.",
    )
    p.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Forward --no-trajectory to every sub-script that supports it.",
    )
    p.add_argument(
        "--stage-timeout-sec",
        type=int,
        default=_config_default(
            "plan",
            "stage_timeout_sec",
            "CMIND_PLAN_STAGE_TIMEOUT_SEC",
            _DEFAULT_STAGE_TIMEOUT_SEC,
        ),
        metavar="SECONDS",
        help="Hard timeout for each planning stage (default: 2700).",
    )
    p.add_argument(
        "--interfaces-timeout-sec",
        type=int,
        default=_config_default(
            "plan",
            "interfaces_timeout_sec",
            "CMIND_PLAN_INTERFACES_TIMEOUT_SEC",
            _DEFAULT_INTERFACES_TIMEOUT_SEC,
        ),
        metavar="SECONDS",
        help="Hard timeout for interface design (default: 5400).",
    )
    p.add_argument(
        "--terminate-grace-sec",
        type=int,
        default=_execution_default(
            "terminate_grace_sec",
            "CMIND_TERMINATE_GRACE_SEC",
            _DEFAULT_TERMINATE_GRACE_SEC,
        ),
        metavar="SECONDS",
        help="Wait after SIGTERM before escalating to SIGKILL (default: 15).",
    )
    p.add_argument(
        "--llm-timeout-sec",
        type=int,
        default=_execution_default(
            "llm_timeout_sec", "CMIND_LLM_TIMEOUT_SEC", _DEFAULT_LLM_TIMEOUT_SEC,
        ),
        metavar="SECONDS",
        help="Timeout for each LLM attempt (default: 900).",
    )
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=_execution_default(
            "llm_max_attempts", "CMIND_LLM_MAX_ATTEMPTS", _DEFAULT_LLM_MAX_ATTEMPTS,
        ),
        metavar="N",
        help="Maximum attempts for each LLM request (default: 2).",
    )
    p.add_argument(
        "--no-progress-timeout-sec",
        type=int,
        default=_execution_default(
            "no_progress_timeout_sec", "CMIND_NO_PROGRESS_TIMEOUT_SEC", 0,
        ),
        metavar="SECONDS",
        help=(
            "Stop a stage after this many seconds without structured progress "
            "(default: max(1200, llm timeout + 300))."
        ),
    )
    # Per-stage iteration overrides (only the four stages that take one).
    for stage in STAGES:
        if stage.max_iter_flag is None:
            continue
        p.add_argument(
            f"--max-iter-{stage.name.replace('_', '-')}",
            dest=f"max_iter_{stage.name}",
            type=int,
            default=None,
            metavar="N",
            help=f"Override iteration count for the '{stage.name}' stage.",
        )
    args = p.parse_args(argv)
    if args.no_progress_timeout_sec <= 0:
        args.no_progress_timeout_sec = max(
            _DEFAULT_NO_PROGRESS_TIMEOUT_SEC,
            args.llm_timeout_sec + 300,
        )
    for name in (
        "stage_timeout_sec",
        "interfaces_timeout_sec",
        "terminate_grace_sec",
        "llm_timeout_sec",
        "llm_max_attempts",
        "no_progress_timeout_sec",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be greater than zero")
    return args


def _install_sigint_handler() -> None:
    def _handle(signum: int, frame: Any) -> None:  # noqa: ARG001
        print("\n[plan] interrupted — rerun `cmind script plan.py` to resume.")
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle)


def _run_pipeline(
    args: argparse.Namespace,
    invoker: list[str],
    plan_lock: PlanLock | None,
) -> int:
    progress_path = _PLAN_PROGRESS_PATH if plan_lock is not None else None
    if progress_path is not None:
        update_progress(
            progress_path,
            pid=os.getpid(),
            status="running",
            stage="probe",
            stage_status="running",
            llm_timeout_sec=args.llm_timeout_sec,
            llm_max_attempts=args.llm_max_attempts,
            no_progress_timeout_sec=args.no_progress_timeout_sec,
        )
    # --- Step: probe ------------------------------------------------------
    states = probe(invoker)
    decide(states, force=args.force)

    if args.check_only:
        active_run = PlanLock.active_metadata(_PLAN_LOCK_PATH)
        progress = read_progress(_PLAN_PROGRESS_PATH)
        if (
            active_run
            and progress
            and progress.get("pid") == active_run.get("pid")
        ):
            active_run = {**active_run, **progress}
        if args.json:
            _emit_check_only_json(states, active_run)
        else:
            _print_probe_summary(states, active_run)
        return 0

    # --- Step: prerequisite check -----------------------------------------
    # If the very first stage cannot even start (its input is missing or
    # invalid), abort cleanly so the user gets a helpful pointer instead
    # of a confusing failure from the build script itself.  ``--dry-run``
    # bypasses this so users can preview commands without an initialised
    # workspace.
    head = states[0]
    if head.type == "error" and not args.dry_run:
        if progress_path is not None:
            update_progress(
                progress_path,
                status="failed",
                stage="probe",
                stage_status="failed",
                error=head.message,
            )
        print(
            f"Cannot start the planning pipeline: {head.message}",
            file=sys.stderr,
        )
        print(
            "Run `/cmind.feature_construct` first to produce "
            "`feature_tree.json`, then re-run `/cmind.plan`.",
            file=sys.stderr,
        )
        return 2

    # --- Step: short-circuit when nothing to do ---------------------------
    runnable = [s for s in states if s.will_run]
    if not runnable:
        if progress_path is not None:
            update_progress(
                progress_path,
                status="complete",
                stage="complete",
                stage_status="nothing_to_do",
                activity="complete",
            )
        print("All 5 planning stages are already complete — nothing to do.")
        print("Use `cmind script plan.py --force` to rebuild from scratch.")
        return 0

    # --- Step: announce plan ----------------------------------------------
    print(f"Planning pipeline: {len(runnable)} of {len(states)} stages to run.")
    print(_format_table(states))
    print()

    if args.dry_run:
        for s in runnable:
            cmd = _script_argv(invoker, s.stage.build_script)
            cmd += _build_args_for(s.stage, args)
            print("DRY-RUN ▸", " ".join(cmd))
        for post in POST_STEPS:
            print("DRY-RUN ▸", " ".join(_script_argv(invoker, post)))
        return 0

    # --- Step: execute ----------------------------------------------------
    with record_activity(
        "workflow", "plan", logical_key="decoder-plan", trigger="command",
        writer=ACTIVITY_WRITER,
    ) as activity:
        activity.note(planned_stages=len(runnable), total_stages=len(states))
        started = time.monotonic()
        recovered_stages: list[str] = []
        llm_env = {
            "CMIND_LLM_TIMEOUT_SEC": str(args.llm_timeout_sec),
            "CMIND_LLM_MAX_ATTEMPTS": str(args.llm_max_attempts),
        }
        if progress_path is not None:
            llm_env[PROGRESS_FILE_ENV] = str(progress_path)
        for s in states:
            if not s.will_run:
                print(f"⏭  {s.stage.name:<14} skip ({s.reason})")
                continue

            stage_started = time.monotonic()
            print(f"▶  {s.stage.name:<14} running {s.stage.build_script} ...")
            if plan_lock is not None:
                plan_lock.update(
                    stage=s.stage.name,
                    stage_status="running",
                    stage_started_at=datetime.now(timezone.utc).isoformat(),
                )
            build_extra = _build_args_for(s.stage, args)
            stage_timeout = _stage_timeout_for(s.stage, args)
            if progress_path is not None:
                update_progress(
                    progress_path,
                    stage=s.stage.name,
                    stage_status="running",
                    activity="stage",
                    stage_timeout_sec=stage_timeout,
                    stage_deadline_epoch=time.time() + stage_timeout,
                )
            result = _run_stage(
                invoker,
                s.stage.build_script,
                build_extra,
                timeout_sec=stage_timeout,
                terminate_grace_sec=args.terminate_grace_sec,
                env_overrides=llm_env,
                progress_path=progress_path,
                no_progress_timeout_sec=args.no_progress_timeout_sec,
            )
            verify: dict[str, Any] | None = None
            if result.timed_out:
                verify = _run_check(invoker, s.stage.check_script)
                can_recover = (
                    not args.force
                    and s.type != "update"
                    and verify.get("type") == "update"
                )
                if can_recover:
                    recovered_stages.append(s.stage.name)
                    print(
                        f"⚠  {s.stage.name:<14} timed out after {stage_timeout}s, "
                        "but its artifact is valid; continuing"
                    )
                else:
                    activity.status = "failed"
                    activity.error = {
                        "type": "PlanStageTimeout",
                        "message": (
                            f"{s.stage.name} stopped after "
                            f"{result.timeout_reason or 'timeout'}"
                        ),
                    }
                    if plan_lock is not None:
                        plan_lock.update(
                            stage_status="timed_out",
                            timeout_reason=result.timeout_reason,
                            verification=verify.get("type", "error"),
                        )
                    if progress_path is not None:
                        update_progress(
                            progress_path,
                            status="failed",
                            stage=s.stage.name,
                            stage_status="timed_out",
                            timeout_reason=result.timeout_reason,
                            verification=verify.get("type", "error"),
                        )
                    _print_failure_hint(
                        invoker, s.stage, _STAGE_TIMEOUT_EXIT_CODE, phase="build",
                    )
                    return _STAGE_TIMEOUT_EXIT_CODE
            elif result.returncode != 0:
                activity.status = "failed"
                activity.error = {
                    "type": "PlanStageError",
                    "message": f"{s.stage.name} exited {result.returncode}",
                }
                if plan_lock is not None:
                    plan_lock.update(
                        stage_status="failed",
                        returncode=result.returncode,
                    )
                if progress_path is not None:
                    update_progress(
                        progress_path,
                        status="failed",
                        stage=s.stage.name,
                        stage_status="failed",
                        returncode=result.returncode,
                    )
                _print_failure_hint(invoker, s.stage, result.returncode, phase="build")
                return result.returncode

        # Re-run the check to confirm the artifact came out valid.  Parse
        # its JSON quietly; surface details only when the verification
        # fails, otherwise the user would see a JSON dump after every
        # stage.
        #
        # ``update`` -> stage is fully valid; continue.
        # Any other type means the artifact is missing, unusable, or
        # violates a cross-stage contract; fail so bench cannot report a
        # false PASS for partial plans.
            if verify is None:
                verify = _run_check(invoker, s.stage.check_script)
            verify_type = verify.get("type", "error")
            if verify_type != "update":
                activity.status = "failed"
                activity.error = {"type": "PlanVerificationError", "message": f"{s.stage.name}: {verify_type}"}
                print(
                    f"   verification failed: {verify_type} — "
                    f"{verify.get('message', 'no message')}",
                    file=sys.stderr,
                )
                for err in verify.get("validation_errors", [])[:5]:
                    print(f"     - {err}", file=sys.stderr)
                if plan_lock is not None:
                    plan_lock.update(
                        stage_status="verification_failed",
                        verification=verify_type,
                    )
                if progress_path is not None:
                    update_progress(
                        progress_path,
                        status="failed",
                        stage=s.stage.name,
                        stage_status="verification_failed",
                        verification=verify_type,
                    )
                _print_failure_hint(invoker, s.stage, 1, phase="check")
                return 1

            elapsed = time.monotonic() - stage_started
            print(f"✓  {s.stage.name:<14} done in {elapsed:.1f}s")
            if plan_lock is not None:
                plan_lock.update(
                    stage_status="recovered" if result.timed_out else "complete",
                    verification=verify_type,
                )
            if progress_path is not None:
                update_progress(
                    progress_path,
                    stage=s.stage.name,
                    stage_status="recovered" if result.timed_out else "complete",
                    activity="stage_complete",
                    verification=verify_type,
                )

    # --- Step: post-pipeline helpers --------------------------------------
        print()
        print("Running post-pipeline helpers ...")
        for post in POST_STEPS:
            print(f"▶  {post}")
            if progress_path is not None:
                update_progress(
                    progress_path,
                    stage=post,
                    stage_status="running",
                    activity="post_step",
                )
            result = _run_stage(
                invoker,
                post,
                [],
                timeout_sec=args.stage_timeout_sec,
                terminate_grace_sec=args.terminate_grace_sec,
                env_overrides=llm_env,
                progress_path=progress_path,
                no_progress_timeout_sec=args.no_progress_timeout_sec,
            )
            if result.returncode != 0:
                detail = "timed out" if result.timed_out else f"exited with {result.returncode}"
                print(f"   warning: {post} {detail} (continuing)")

        total_elapsed = time.monotonic() - started
        activity.note(
            duration_ms=round(total_elapsed * 1000, 3),
            recovered_stages=recovered_stages,
        )
        if plan_lock is not None:
            plan_lock.update(stage="complete", stage_status="complete")
        if progress_path is not None:
            update_progress(
                progress_path,
                status="complete",
                stage="complete",
                stage_status="complete",
                activity="complete",
            )
    print()
    print(f"Plan complete in {total_elapsed:.1f}s.")
    print("Next: `/cmind.code_gen` to generate source code.")
    print("Graph: see the 'Writing visualization to:' line above for the generated HTML path.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _install_sigint_handler()
    invoker = _resolve_invoker()
    if args.check_only or args.dry_run:
        return _run_pipeline(args, invoker, None)
    try:
        with PlanLock(_PLAN_LOCK_PATH) as plan_lock:
            return _run_pipeline(args, invoker, plan_lock)
    except PlanAlreadyRunning as exc:
        owner = exc.metadata
        print(
            "Another planning pipeline is already active in this workspace "
            f"(pid={owner.get('pid', 'unknown')}, "
            f"stage={owner.get('stage', 'unknown')}, "
            f"started_at={owner.get('started_at', 'unknown')}).",
            file=sys.stderr,
        )
        print(
            "Use `cmind script plan.py --check-only` to inspect progress.",
            file=sys.stderr,
        )
        return _LOCKED_EXIT_CODE
    except BaseException as exc:
        update_progress(
            _PLAN_PROGRESS_PATH,
            status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            stage_status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=type(exc).__name__,
        )
        raise


def _print_failure_hint(
    invoker: list[str],
    stage: Stage,
    rc: int,
    *,
    phase: str,
) -> None:
    """Print recovery hints to stderr after a stage fails.

    ``phase`` is ``"build"`` or ``"check"``; the debug command points at
    the script that actually failed so the user can reproduce. Commands
    are rendered using the current *invoker* so the hint stays correct
    whether ``cmind`` is on ``$PATH`` (``cmind script <name>``) or the
    Python fallback is in use (``python <abspath>``).
    """
    debug_script = stage.build_script if phase == "build" else stage.check_script

    def _fmt(name: str, *extra: str) -> str:
        # Render the command using the basename of the invoker so
        # users see e.g. ``cmind script ...`` or ``python ...`` rather
        # than the resolved absolute path that subprocess actually uses.
        argv = _script_argv(invoker, name)
        display = [Path(argv[0]).name, *argv[1:], *extra]
        return " ".join(display)

    print(file=sys.stderr)
    print(f"✗ {stage.name} {phase} failed (exit {rc})", file=sys.stderr)
    print(f"  Resume :  {_fmt('plan.py')}", file=sys.stderr)
    print(f"  Debug  :  {_fmt(debug_script, '--verbose')}", file=sys.stderr)
    print(f"  Status :  {_fmt('plan.py', '--check-only')}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
