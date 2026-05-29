#!/usr/bin/env python3
"""Apply EditPlan code_changes via a dedicated SubAgent (RPG-Driven).

This script implements Step 5c of /cmind.rpg_edit. Instead of the main
Agent freely editing code, it dispatches a SubAgent with a constrained
prompt that treats the updated RPG nodes as authoritative ground truth.

Workflow:
  1. Load EditPlan + RPG (updated by Step 5b) + impact data
  2. Build a SubAgent prompt with RPG target nodes + code_changes
  3. Dispatch SubAgent, parse CODE_STATUS from its response
  4. If PARTIAL, iterate with only the remaining changes (no RPG re-send)
  5. After completion (or max_iterations), commit all changes once

Output JSON:
  {
    "type": "code_applied" | "error",
    "success": bool,
    "files_modified": [...],
    "iterations": [{"iteration", "prompt_len", "parsed_status", ...}],
    "last_status": "complete" | "partial" | "failed" | "llm_error" | "unknown",
    "last_error": str | null,
    "commit_sha": str | null
  }
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import (  # noqa: E402
    RPG_FILE,
    REPO_DIR,
    RPG_EDIT_PLAN_FILE,
    DATA_DIR,
    WORKSPACE_ROOT,
    cmd_for,
)
from common.logging_setup import setup_file_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CODE_PROMPT_INITIAL = Template("""\
# RPG-Driven Code Modification

You are a code implementation agent. Your job is to faithfully implement
the changes described below. The RPG nodes are the AUTHORITY — your code
MUST match what they describe.

## RPG Target State (Authority)
$RPG_TARGET_NODES

## Code Changes to Apply
$CODE_CHANGES

## Impact Context (read-only — these depend on your changes)
$IMPACT_CONTEXT

## Instructions
For each entry in "Code Changes to Apply":
1. Read the target file
2. Implement the change as described
3. Ensure the result matches the RPG target state above

After all changes are applied:
1. Run: $SMOKE_TEST_CMD
2. Run: $PYTEST_CMD
3. If tests fail, fix the code and re-run

## Constraints
- Do NOT modify files outside "Code Changes to Apply"
- Do NOT add features not described in the plan
- The RPG describes the TARGET state — make code match it
- Do NOT commit. The driver script will commit after verifying status.

## Exit Protocol
On the LAST line of your response, output exactly one of:
- CODE_STATUS: COMPLETE
- CODE_STATUS: PARTIAL | <json_array_of_completed_files>
- CODE_STATUS: FAILED | <reason>

Examples:
  CODE_STATUS: COMPLETE
  CODE_STATUS: PARTIAL | ["src/a.py", "src/b.py"]
  CODE_STATUS: FAILED | unable to locate function X in file Y
""")


CODE_PROMPT_CONTINUE = Template("""\
# RPG-Driven Code Modification (Iteration $ITERATION)

## Progress So Far
Completed ($DONE_COUNT/$TOTAL_COUNT files):
$DONE_FILES

## Remaining Changes
$REMAINING_CHANGES
$ERROR_SECTION

## Instructions
Continue from where the previous iteration left off.
Only modify files listed in "Remaining Changes" above.

After all changes:
1. Run: $SMOKE_TEST_CMD
2. Run: $PYTEST_CMD
3. If tests fail, fix the code

## Constraints
- Do NOT commit. The driver script will commit.
- Do NOT re-modify already-completed files unless fixing test failures.

## Exit Protocol
- CODE_STATUS: COMPLETE
- CODE_STATUS: PARTIAL | <json_array_of_completed_files>
- CODE_STATUS: FAILED | <reason>
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_test_files(code_changes: List[dict]) -> List[str]:
    """Derive pytest -k patterns from code_changes file paths.

    Mirrors the logic in :mod:`scripts.rpg_edit.review` to keep behavior consistent.
    """
    seen: set = set()
    patterns: List[str] = []
    for cc in code_changes:
        fp = cc.get("file_path", "")
        if not fp.endswith(".py"):
            continue
        p = Path(fp)
        parts = list(p.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        while parts and parts[0] in ("src", "lib"):
            parts.pop(0)
        if not parts:
            continue
        if len(parts) > 1:
            parts = parts[1:]  # drop top-level package dir
        key_parts = parts[-3:] if len(parts) > 3 else parts
        pattern = "test_" + "_".join(key_parts)
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    return patterns


def _build_validation_cmds(code_changes: List[dict]) -> Tuple[str, str]:
    """Build absolute-path smoke_test and pytest commands.

    SubAgent inherits the parent's cwd (workspace root == REPO_DIR), but
    we still use absolute paths to keep the prompt cwd-agnostic — it
    must work no matter where the user runs the slash command from.
    """
    smoke = f"{cmd_for('smoke_test.py')} --json"

    patterns = _derive_test_files(code_changes)
    if patterns:
        pattern_expr = " or ".join(patterns)
        pytest_cmd = (
            f'python3 -m pytest -x -q -k "{pattern_expr}" --timeout=30'
        )
    else:
        pytest_cmd = "python3 -m pytest -x -q --timeout=30"
    return smoke, pytest_cmd


def _derive_summary(plan: dict) -> str:
    """Derive a one-line commit summary from the plan."""
    changes = plan.get("code_changes", [])
    if changes:
        desc = (changes[0].get("description") or "").strip()
        if desc:
            return desc[:60]
    feats = plan.get("feature_changes", [])
    if feats:
        patch = feats[0].get("patch", {}) or {}
        name = patch.get("name") or feats[0].get("node_id", "")
        if name:
            return f"update {name[:60]}"
    return "apply EditPlan"


def _format_rpg_target_nodes(
    plan: dict,
    rpg_path: Path,
    max_chars: int = 5000,
) -> str:
    """Extract and format affected RPG nodes as compact text lines.

    Output one line per affected node:
      - <name> [<node_type>] @ <meta.path>  (feature_path)

    Falls back gracefully if RPG load fails or affected_nodes is missing.
    """
    affected = plan.get("affected_nodes") or []
    if not affected:
        # Fall back to feature_changes node IDs (skip "add" without node_id)
        affected = [
            fc.get("node_id") for fc in plan.get("feature_changes", [])
            if fc.get("node_id")
        ]
    if not affected:
        return "(no affected_nodes listed in plan)"

    try:
        from rpg.service import RPGService
        svc = RPGService.load(str(rpg_path))
    except Exception as exc:
        return f"(failed to load RPG: {exc})"

    lines: List[str] = []
    for nid in affected:
        node = svc.rpg._node_index.get(nid)
        if node is None:
            lines.append(f"- {nid}  (NOT FOUND in RPG)")
            continue
        name = node.name or "?"
        ntype = node.node_type or "?"
        meta_path = ""
        if node.meta and node.meta.path:
            meta_path = (
                node.meta.path
                if isinstance(node.meta.path, str)
                else " | ".join(node.meta.path)
            )
        try:
            fp = node.feature_path()
        except Exception:
            fp = ""
        line = f"- `{name}` [{ntype}]"
        if meta_path:
            line += f" @ {meta_path}"
        if fp:
            line += f"  (path: {fp})"
        lines.append(line)

    text = "\n".join(lines)
    if len(text) > max_chars:
        # Truncate by line count to stay within budget
        truncated = []
        size = 0
        for ln in lines:
            if size + len(ln) + 1 > max_chars - 80:
                break
            truncated.append(ln)
            size += len(ln) + 1
        omitted = len(lines) - len(truncated)
        truncated.append(f"... ({omitted} more nodes omitted)")
        text = "\n".join(truncated)
    return text


def _format_impact_context(
    plan: dict,
    max_chars: int = 2000,
) -> str:
    """Format callers/callees from rpg_edit_impact.json (if available)."""
    impact_path = DATA_DIR / "rpg_edit_impact.json"
    if not impact_path.exists():
        return "(no impact data available)"

    try:
        impact = json.loads(impact_path.read_text())
    except Exception as exc:
        return f"(failed to load impact: {exc})"

    results = impact.get("results", {}) if isinstance(impact, dict) else {}

    callers: List[str] = []
    callees: List[str] = []
    for _nid, data in results.items():
        if not isinstance(data, dict):
            continue
        for c in (data.get("callers") or [])[:20]:
            name = c.get("name") if isinstance(c, dict) else None
            if name and name not in callers:
                callers.append(name)
        for c in (data.get("callees") or [])[:20]:
            name = c.get("name") if isinstance(c, dict) else None
            if name and name not in callees:
                callees.append(name)

    parts: List[str] = []
    if callers:
        parts.append("Callers (depend on the modified code): "
                     + ", ".join(callers[:20]))
    else:
        parts.append("Callers: (none)")
    if callees:
        parts.append("Callees (invoked by the modified code): "
                     + ", ".join(callees[:20]))
    else:
        parts.append("Callees: (none)")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars - 20] + "\n... (truncated)"
    return text


def _format_code_changes(changes: List[dict], max_chars: int = 3000) -> str:
    """Format code_changes as a numbered list."""
    if not changes:
        return "(no changes)"
    lines: List[str] = []
    for i, cc in enumerate(changes, 1):
        fp = cc.get("file_path", "?")
        ct = cc.get("change_type", "modify")
        desc = (cc.get("description") or "").strip()
        line = f"{i}. [{ct}] {fp}"
        if desc:
            line += f"\n   {desc}"
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars - 20] + "\n... (truncated)"
    return text


def _parse_code_status(response: str) -> Tuple[str, Any]:
    """Parse CODE_STATUS line from the last 20 lines of response.

    Returns:
        ("complete", None)
        ("partial", ["file1.py", ...])
        ("failed", "reason")
        ("unknown", raw_last_line_snippet)
    """
    if not response:
        return "unknown", ""
    lines = response.strip().splitlines()
    tail = lines[-20:]
    for line in reversed(tail):
        # Strip leading bullet/markdown markers first so we can detect
        # lines like "- CODE_STATUS: COMPLETE".
        s = line.strip().lstrip("-*`> ").strip()
        if not s.startswith("CODE_STATUS"):
            continue
        # Format: CODE_STATUS: STATUS [| detail]
        try:
            _, rest = s.split(":", 1)
        except ValueError:
            continue
        rest = rest.strip()
        if "|" in rest:
            status_part, detail_part = rest.split("|", 1)
            status_part = status_part.strip().lower()
            detail_part = detail_part.strip()
        else:
            status_part = rest.strip().lower()
            detail_part = ""

        if status_part == "complete":
            return "complete", None
        if status_part == "partial":
            # Detail is expected to be a JSON array
            try:
                parsed = json.loads(detail_part) if detail_part else []
                if isinstance(parsed, list):
                    return "partial", [str(x) for x in parsed]
                return "partial", parsed  # mark malformed
            except json.JSONDecodeError:
                return "partial", detail_part  # malformed
        if status_part == "failed":
            return "failed", detail_part or "no reason"
        # Unknown status keyword — continue searching
    return "unknown", (tail[-1] if tail else "")[:120]


def _commit_changes(
    repo_path: Path,
    summary: str,
    status: str,
) -> Optional[str]:
    """Stage all changes and create a single commit. Returns commit SHA."""
    # Check if there is anything to commit
    try:
        st = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as exc:
        logger.warning("git status failed: %s", exc)
        return None
    if not st.stdout.strip():
        logger.info("No changes to commit in %s", repo_path)
        return None

    msg = f"rpg_edit: {summary}"
    if status != "complete":
        msg += f" [{status}]"
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "add", "-A"],
            check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "commit", "-m", msg],
            check=True, capture_output=True, timeout=30,
        )
        sha = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return sha.stdout.strip() or None
    except Exception as exc:
        logger.error("git commit failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_initial_prompt(
    plan: dict,
    rpg_nodes_text: str,
    impact_text: str,
    remaining: List[dict],
) -> str:
    smoke_cmd, pytest_cmd = _build_validation_cmds(remaining)
    return CODE_PROMPT_INITIAL.safe_substitute(
        RPG_TARGET_NODES=rpg_nodes_text,
        CODE_CHANGES=_format_code_changes(remaining),
        IMPACT_CONTEXT=impact_text,
        SMOKE_TEST_CMD=smoke_cmd,
        PYTEST_CMD=pytest_cmd,
    )


def _build_continue_prompt(
    done_files: List[str],
    remaining: List[dict],
    last_error: Optional[str],
    iteration: int,
    total: int,
) -> str:
    smoke_cmd, pytest_cmd = _build_validation_cmds(remaining)
    done_text = "\n".join(f"- {f}" for f in done_files) or "(none)"
    error_section = ""
    if last_error:
        snippet = last_error[:1000]
        error_section = (
            f"\n## Previous Iteration Error\n{snippet}\n"
        )
    return CODE_PROMPT_CONTINUE.safe_substitute(
        ITERATION=str(iteration),
        DONE_COUNT=str(len(done_files)),
        TOTAL_COUNT=str(total),
        DONE_FILES=done_text,
        REMAINING_CHANGES=_format_code_changes(remaining),
        ERROR_SECTION=error_section,
        SMOKE_TEST_CMD=smoke_cmd,
        PYTEST_CMD=pytest_cmd,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def apply_code_changes(
    plan_path: Path,
    rpg_path: Path,
    repo_path: Path,
    max_iterations: int = 3,
    timeout: int = 900,
) -> Dict[str, Any]:
    """Apply EditPlan code_changes via SubAgent with iterative completion."""
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "type": "error",
            "success": False,
            "error": f"failed to load plan: {exc}",
        }

    all_changes = plan.get("code_changes") or []
    if not all_changes:
        return {
            "type": "code_applied",
            "success": True,
            "files_modified": [],
            "iterations": [],
            "last_status": "complete",
            "message": "no code_changes in plan",
            "commit_sha": None,
        }

    # Build static context once (reused across iterations)
    rpg_nodes_text = _format_rpg_target_nodes(plan, rpg_path)
    impact_text = _format_impact_context(plan)
    summary = _derive_summary(plan)
    all_files = [c["file_path"] for c in all_changes if c.get("file_path")]
    total_files = len(set(all_files))

    # Lazy import to avoid circular dependency at module load time
    from run_batch import dispatch_sub_agent

    done_files: List[str] = []
    last_error: Optional[str] = None
    last_status = "unknown"
    iterations_data: List[Dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        remaining = [
            c for c in all_changes
            if c.get("file_path") and c["file_path"] not in done_files
        ]
        if not remaining:
            last_status = "complete"
            break

        if iteration == 1:
            prompt = _build_initial_prompt(
                plan, rpg_nodes_text, impact_text, remaining,
            )
        else:
            prompt = _build_continue_prompt(
                done_files, remaining, last_error, iteration, total_files,
            )

        logger.info(
            "Iteration %d/%d: prompt_len=%d remaining=%d",
            iteration, max_iterations, len(prompt), len(remaining),
        )

        response, error = dispatch_sub_agent(
            prompt, repo_path,
            timeout=timeout,
            purpose=f"rpg_edit_code_{iteration}",
        )

        iter_info: Dict[str, Any] = {
            "iteration": iteration,
            "prompt_len": len(prompt),
            "response_len": len(response) if response else 0,
            "error": error,
        }
        iterations_data.append(iter_info)

        if error:
            last_error = f"LLM call failed: {error}"
            last_status = "llm_error"
            iter_info["parsed_status"] = "llm_error"
            continue

        status, detail = _parse_code_status(response or "")
        last_status = status
        iter_info["parsed_status"] = status

        if status == "complete":
            # Agent declares all remaining changes done.  Mark current
            # `remaining` files as completed so files_modified reflects
            # actual work even when SubAgent doesn't list them explicitly.
            for c in remaining:
                fp = c.get("file_path")
                if fp and fp not in done_files:
                    done_files.append(fp)
            iter_info["detail"] = None
            break
        elif status == "partial":
            if not isinstance(detail, list):
                last_error = f"PARTIAL detail not a list: {detail!r}"
                iter_info["detail"] = detail
                continue
            new_files = [f for f in detail if f not in done_files]
            done_files.extend(new_files)
            iter_info["new_files"] = new_files
            if not new_files:
                # SubAgent claims partial progress but reported no new
                # files (or only already-done files).  Treat as a stall
                # to avoid infinite loops within max_iterations.
                last_error = (
                    "PARTIAL with no new files; SubAgent stalled."
                )
                last_status = "stalled"
                iter_info["parsed_status"] = "stalled"
                break
            last_error = None
        elif status == "failed":
            last_error = str(detail)
            iter_info["fail_reason"] = last_error
        else:  # unknown
            last_error = (
                f"no CODE_STATUS line in response; last line: {detail!r}"
            )
            iter_info["fail_reason"] = last_error

    # If loop exhausted iterations without completion, annotate last_error.
    if last_status not in ("complete",) and not last_error:
        last_error = (
            f"max_iterations ({max_iterations}) reached without COMPLETE"
        )

    # Commit if anything was actually modified
    commit_sha: Optional[str] = None
    if last_status == "complete" or done_files:
        commit_sha = _commit_changes(repo_path, summary, last_status)

    success = last_status == "complete"
    return {
        "type": "code_applied",
        "success": success,
        "iterations": iterations_data,
        "files_modified": done_files,
        "last_status": last_status,
        "last_error": last_error,
        "commit_sha": commit_sha,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply EditPlan code_changes via SubAgent (RPG-driven)",
    )
    parser.add_argument(
        "--plan", type=Path, default=RPG_EDIT_PLAN_FILE,
        help="Path to rpg_edit_plan.json (default: %(default)s)",
    )
    parser.add_argument(
        "--rpg", type=Path, default=RPG_FILE,
        help="Path to updated rpg.json (default: %(default)s)",
    )
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="Repository root (default: common.paths.REPO_DIR)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Max SubAgent iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Per-iteration SubAgent timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output result as JSON",
    )
    args = parser.parse_args()

    # Capture log records for post-mortem inspection
    setup_file_logging("rpg_edit")

    repo_path = args.repo or REPO_DIR

    result = apply_code_changes(
        plan_path=args.plan,
        rpg_path=args.rpg,
        repo_path=repo_path,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        ok = "OK" if result.get("success") else "FAIL"
        print(f"[{ok}] {result.get('type')}: "
              f"last_status={result.get('last_status')}, "
              f"files_modified={len(result.get('files_modified', []))}, "
              f"iterations={len(result.get('iterations', []))}")
        if result.get("last_error"):
            print(f"last_error: {result['last_error']}")
        if result.get("commit_sha"):
            print(f"commit_sha: {result['commit_sha']}")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
