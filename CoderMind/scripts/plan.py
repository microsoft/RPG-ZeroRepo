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

Sub-scripts are invoked via ``cmind script <name>`` when the
``cmind`` CLI is on ``$PATH`` (so each stage gets its own
``logs/<stage>.log`` and inner-git snapshot, courtesy of the
dispatcher).  When ``cmind`` is missing, the script falls back
to a direct ``python <bundled-script-path>`` invocation.

Exit codes
----------

* 0    — pipeline finished successfully (or nothing to do)
* 2    — argument error
* 130  — interrupted with Ctrl-C
* N    — exit code of the first failing sub-stage (passed through)
"""

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

# Sub-scripts live in the same directory as this file (bundled under
# cmind_cli/core_pack/scripts/ in the installed wheel).
_SCRIPTS_DIR = Path(__file__).resolve().parent


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
    argv = [*_script_argv(invoker, script_name), "--json"]
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"type": "error", "message": f"cannot invoke {argv[0]}: {exc}"}

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


def _run_stage(invoker: list[str], script_name: str, extra: list[str]) -> int:
    """Run a build_*/design_* script and stream its output live."""
    argv = [*_script_argv(invoker, script_name), *extra]
    proc = subprocess.run(argv, check=False)
    return proc.returncode


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


def _print_probe_summary(states: list[StageState]) -> None:
    done = sum(1 for s in states if s.done)
    total = len(states)
    first_pending = next((s.stage.name for s in states if not s.done), None)
    print(f"Planning progress: {done}/{total} stages complete.")
    if first_pending:
        print(f"Next pending stage: {first_pending}")
    else:
        print("All stages are up-to-date.")
    print()
    print(_format_table(states))


def _emit_check_only_json(states: list[StageState]) -> None:
    done = sum(1 for s in states if s.done)
    total = len(states)
    next_pending = next((s.stage.name for s in states if not s.done), None)
    payload = {
        "total": total,
        "done": done,
        "next": next_pending,
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
    return p.parse_args(argv)


def _install_sigint_handler() -> None:
    def _handle(signum: int, frame: Any) -> None:  # noqa: ARG001
        print("\n[plan] interrupted — rerun `cmind script plan.py` to resume.")
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _install_sigint_handler()
    invoker = _resolve_invoker()

    # --- Step: probe ------------------------------------------------------
    states = probe(invoker)
    decide(states, force=args.force)

    if args.check_only:
        if args.json:
            _emit_check_only_json(states)
        else:
            _print_probe_summary(states)
        return 0

    # --- Step: prerequisite check -----------------------------------------
    # If the very first stage cannot even start (its input is missing or
    # invalid), abort cleanly so the user gets a helpful pointer instead
    # of a confusing failure from the build script itself.  ``--dry-run``
    # bypasses this so users can preview commands without an initialised
    # workspace.
    head = states[0]
    if head.type == "error" and not args.dry_run:
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
    started = time.monotonic()
    for s in states:
        if not s.will_run:
            print(f"⏭  {s.stage.name:<14} skip ({s.reason})")
            continue

        stage_started = time.monotonic()
        print(f"▶  {s.stage.name:<14} running {s.stage.build_script} ...")
        build_extra = _build_args_for(s.stage, args)
        rc = _run_stage(invoker, s.stage.build_script, build_extra)
        if rc != 0:
            _print_failure_hint(invoker, s.stage, rc, phase="build")
            return rc

        # Re-run the check to confirm the artifact came out valid.  Parse
        # its JSON quietly; surface details only when the verification
        # fails, otherwise the user would see a JSON dump after every
        # stage.
        #
        # ``update`` -> stage is fully valid; continue.
        # Any other type means the artifact is missing, unusable, or
        # violates a cross-stage contract; fail so bench cannot report a
        # false PASS for partial plans.
        verify = _run_check(invoker, s.stage.check_script)
        verify_type = verify.get("type", "error")
        if verify_type != "update":
            print(
                f"   verification failed: {verify_type} — "
                f"{verify.get('message', 'no message')}",
                file=sys.stderr,
            )
            for err in verify.get("validation_errors", [])[:5]:
                print(f"     - {err}", file=sys.stderr)
            _print_failure_hint(invoker, s.stage, 1, phase="check")
            return 1

        elapsed = time.monotonic() - stage_started
        print(f"✓  {s.stage.name:<14} done in {elapsed:.1f}s")

    # --- Step: post-pipeline helpers --------------------------------------
    print()
    print("Running post-pipeline helpers ...")
    for post in POST_STEPS:
        print(f"▶  {post}")
        rc = _run_stage(invoker, post, [])
        if rc != 0:
            print(f"   warning: {post} exited with {rc} (continuing)")

    total_elapsed = time.monotonic() - started
    print()
    print(f"Plan complete in {total_elapsed:.1f}s.")
    print("Next: `/cmind.code_gen` to generate source code.")
    print("Graph: see the 'Writing visualization to:' line above for the generated HTML path.")
    return 0


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
