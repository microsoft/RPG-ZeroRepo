#!/usr/bin/env python3
"""Impact-scoped review for rpg_edit — verify affected functionality via sub-agent.

Dispatches a sub-agent to verify that code changes made by rpg_edit
actually work correctly. The review scope is driven by impact analysis
data (callers, affected_files), NOT a full global review.

Usage:
    cmind script rpg_edit/review.py \
      --plan .cmind/data/rpg_edit_plan.json \
      --impact .cmind/data/rpg_edit_impact.json \
      --json

The sub-agent will:
  1. Run pytest on affected test files
  2. Run smoke_test for import/entry verification
  3. Start the application and verify affected functionality paths
  4. Fix any issues found and re-verify
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import REPO_DIR, cmd_for, RPG_EDIT_PLAN_FILE, RPG_EDIT_IMPACT_FILE  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Review prompt template
# ---------------------------------------------------------------------------

IMPACT_REVIEW_PROMPT = Template("""\
# Impact Review: Verify Modified Functionality

You are a QA engineer verifying a **specific code modification** — NOT a full
project review. Focus ONLY on the affected functionality listed below.

## What Changed

**Modified files:**
$CODE_CHANGES

**Affected RPG feature nodes:**
$AFFECTED_NODES

**Callers of modified code (must be verified):**
$CALLERS

**All affected files:**
$AFFECTED_FILES

## Pre-Check Results

**pytest (affected tests):**
$PYTEST_STATUS

**smoke_test (imports/entry):**
$SMOKE_STATUS

## Your Workflow

### 1. Read the modified code
Read each modified file to understand what changed.

### 2. Run targeted tests
```bash
$PYTEST_CMD
```

### 3. Run smoke test
```bash
$SMOKE_TEST_CMD
```

### 4. Start the application and verify affected paths

$START_INSTRUCTIONS

For EACH caller listed above:
- Determine what user action triggers that caller
- Execute that action (HTTP request, CLI command, GUI interaction)
- Verify the result is correct

### 5. Visual Verification (MANDATORY for web/GUI projects)

**This step is NOT optional.** You MUST use the provided tools to visually
verify the project. Verifying only via curl/API is insufficient — real users
interact through the browser or GUI.

#### 5a. Inspect every affected page

For **web apps**, use `inspect` on EVERY affected route to capture
screenshots and saved HTML:
```bash
$BROWSER_TOOL inspect http://localhost:<PORT>/
$BROWSER_TOOL inspect http://localhost:<PORT>/<affected_route>
```
Read the saved HTML files to understand the full page content, CSS layout,
and element structure. Check for:
- Fixed pixel widths that should be responsive
- Elements overflowing or being cut off
- Broken layout at different conceptual viewport sizes
- Missing or misaligned visual elements

#### 5b. Simulate real user interactions

Don't just view pages — **interact** with them like a real user:
```bash
$BROWSER_TOOL run-script http://localhost:<PORT>/<page> --script '
page.click("a:has-text(\\"Some Link\\")")
page.wait_for_load_state("networkidle")
'
```
After each interaction, read the saved [After] HTML to verify the result.

For **GUI apps**, use the GUI tool:
```bash
$GUI_TOOL start-display
$GUI_TOOL launch "python main.py" --wait 3
$GUI_TOOL status
$GUI_TOOL screenshot
```
Click every relevant button, fill forms, and screenshot after each action.

#### 5c. Visual quality check

After inspecting pages / taking screenshots:
- Check that content renders correctly (not blank, not broken)
- Verify layout adapts properly (no horizontal scrollbar, no overflow)
- For style/CSS/layout changes: verify the visual result matches the intent
- If the visual result is poor (misaligned, cut off, ugly), this is a
  **FAIL** even if tests pass

### 6. Fix any issues found
If a test fails, functionality doesn't work, or **visual quality is poor**:
- Fix the code
- Re-run the failing test
- Re-inspect the affected pages to verify the visual fix
- Re-verify the affected path

### 7. Commit fixes (if any)
```bash
git add -A && git commit -m "review: fix issues found in impact review"
```

## Exit Protocol

After verifying ALL affected callers AND visual inspection, output your
result on the LAST line:

- `REVIEW_RESULT: PASS` — all affected functionality works AND looks correct
- `REVIEW_RESULT: FAIL | <reason>` — unfixable issues remain
- `REVIEW_RESULT: PASS_WITH_FIXES` — issues found and fixed

**Before the REVIEW_RESULT line**, if you noticed any related issues that
are **outside the scope of this plan** but worth addressing, list them
in a `SUGGESTIONS` block:

```
SUGGESTIONS:
- src/flask_blog/views/errors.py: still has max-width:600px hardcoded
- src/flask_blog/models/view_engine.py: .sidebar width:260px is fixed px
- (any other patterns you noticed while inspecting)
```

These will be shown to the user as follow-up recommendations.
Do NOT fix these — they are out of scope. Just report them.

$PREVIOUS_ISSUES

## Critical Rules
- Only verify functionality connected to the modified code — NOT all features
- Actually RUN the code — don't just read it
- **MUST use browser.py/gui.py tools** — curl alone is NOT sufficient
- For layout/style changes: visual inspection is the PRIMARY verification
- After taking a screenshot, check it shows meaningful content (not blank)
- Create test data through the project's own interfaces
- Kill background processes before finishing
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_test_files(code_changes: List[dict]) -> List[str]:
    """Derive test file patterns from code_changes.

    Uses directory context to build discriminating patterns.
    e.g. ``src/flask_blog/views/posts/misc.py`` → ``test_views_posts_misc``
    instead of the generic ``test_misc`` which matches nothing.
    """
    seen: set = set()
    patterns: List[str] = []
    for cc in code_changes:
        fp = cc.get("file_path", "")
        if not fp.endswith(".py"):
            continue
        p = Path(fp)
        stem = p.stem

        # Build a path-aware pattern: take up to 3 trailing path segments
        # e.g. views/posts/misc.py → test_views_posts_misc
        parts = list(p.with_suffix("").parts)
        # Drop __init__ — use parent directory name instead
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        # Strip common prefixes like "src", package name
        while parts and parts[0] in ("src", "lib"):
            parts.pop(0)
        if not parts:
            continue
        # Skip the top-level package dir (e.g. "flask_blog")
        if len(parts) > 1:
            parts = parts[1:]  # drop package name
        # Use last 3 segments max
        key_parts = parts[-3:] if len(parts) > 3 else parts
        pattern = "test_" + "_".join(key_parts)

        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    return patterns


def _format_code_changes(code_changes: List[dict]) -> str:
    lines = []
    for cc in code_changes:
        fp = cc.get("file_path", "?")
        ct = cc.get("change_type", "?")
        desc = cc.get("description", "")
        lines.append(f"- `{fp}` ({ct}): {desc}")
    return "\n".join(lines) or "(no code changes)"


def _format_callers(impact_results: dict) -> str:
    seen: set = set()
    lines: List[str] = []
    for node_id, data in impact_results.items():
        for caller in data.get("callers", []):
            nid = caller.get("node_id", "?")
            if nid in seen:
                continue
            seen.add(nid)
            name = caller.get("name", "?")
            lines.append(f"- `{name}` ({nid})")
    return "\n".join(lines) or "(no callers — isolated change)"


def _format_affected_files(impact_results: dict) -> str:
    files = set()
    for node_id, data in impact_results.items():
        files.update(data.get("affected_files", []))
    return "\n".join(f"- `{f}`" for f in sorted(files)) or "(none)"


def _format_affected_nodes(plan: dict) -> str:
    nodes = plan.get("affected_nodes", [])
    return "\n".join(f"- `{n}`" for n in nodes) or "(none)"


def _count_impact(impact_results: dict) -> Tuple[int, int]:
    """Return (unique_callers, affected_file_count)."""
    caller_ids: set = set()
    files: set = set()
    for data in impact_results.values():
        for c in data.get("callers", []):
            caller_ids.add(c.get("node_id") or c.get("name", ""))
        files.update(data.get("affected_files", []))
    return len(caller_ids), len(files)


def _parse_review_result(response: Optional[str]) -> Tuple[bool, str]:
    """Parse REVIEW_RESULT from sub-agent response."""
    if not response:
        return False, "No response from sub-agent"

    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if line.startswith("REVIEW_RESULT:"):
            result = line[len("REVIEW_RESULT:"):].strip()
            if result == "PASS" or result == "PASS_WITH_FIXES":
                return True, result
            elif result.startswith("FAIL"):
                return False, result
    return False, "REVIEW_RESULT not found in response"


def _parse_suggestions(response: Optional[str]) -> List[str]:
    """Extract SUGGESTIONS block from sub-agent response."""
    if not response:
        return []
    suggestions: List[str] = []
    in_block = False
    for line in response.splitlines():
        stripped = line.strip()
        if stripped == "SUGGESTIONS:":
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                suggestions.append(stripped[2:])
            elif stripped.startswith("```") or stripped.startswith("REVIEW_RESULT"):
                break
            elif not stripped:
                continue
            else:
                break
    return suggestions


def build_impact_review_prompt(
    plan: dict,
    impact_results: dict,
    pytest_status: str,
    smoke_status: str,
    previous_issues: str = "",
) -> str:
    """Build the impact-scoped review prompt."""
    code_changes = plan.get("code_changes", [])
    test_patterns = _derive_test_files(code_changes)

    pytest_cmd = "python3 -m pytest -x -q"
    if test_patterns:
        pattern = " or ".join(test_patterns)
        pytest_cmd += f' -k "{pattern}" --timeout=30'

    # Tool invocations route through the global ``cmind`` CLI (the
    # scripts no longer live in the workspace).  See ``cmind script``
    # in docs/cli-reference.md.
    browser_tool = cmd_for("tools/browser.py")
    gui_tool = cmd_for("tools/gui.py")
    smoke_test_cmd = f"{cmd_for('smoke_test.py')} --json"

    # Start instructions depend on project type
    start_instructions = (
        "Start the application in the background and verify it's running:\n"
        "```bash\n"
        "# Read main.py or app.py to find the start command\n"
        "python3 main.py &\n"
        "# Wait and verify\n"
        "sleep 2 && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/\n"
        "```\n"
        "Adjust the port based on what the app actually uses."
    )

    return IMPACT_REVIEW_PROMPT.safe_substitute(
        CODE_CHANGES=_format_code_changes(code_changes),
        AFFECTED_NODES=_format_affected_nodes(plan),
        CALLERS=_format_callers(impact_results),
        AFFECTED_FILES=_format_affected_files(impact_results),
        PYTEST_STATUS=pytest_status,
        SMOKE_STATUS=smoke_status,
        PYTEST_CMD=pytest_cmd,
        BROWSER_TOOL=browser_tool,
        GUI_TOOL=gui_tool,
        SMOKE_TEST_CMD=smoke_test_cmd,
        START_INSTRUCTIONS=start_instructions,
        PREVIOUS_ISSUES=previous_issues or "",
    )


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def impact_review(
    plan_path: Path,
    impact_path: Optional[Path],
    repo_path: Path,
    max_iterations: int = 3,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run impact-scoped review with iterative repair."""
    from run_batch import dispatch_sub_agent
    from code_gen.test_runner import run_pytest
    from smoke_test import run_smoke_test

    # 1. Load data
    plan = json.loads(plan_path.read_text())

    if impact_path and impact_path.exists():
        impact_data = json.loads(impact_path.read_text())
        impact_results = impact_data.get("results", {})
    else:
        impact_results = {}
        logger.warning("No impact data provided, review scope may be incomplete")

    # 2. Pre-check: pytest on affected test files
    test_patterns = _derive_test_files(plan.get("code_changes", []))
    try:
        pre_pytest = run_pytest(
            repo_path,
            test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
            timeout=120,
            extra_args=["--timeout=30"],
        )
        pytest_status = (
            f"{'PASS' if pre_pytest.success else 'FAIL'}: "
            f"{pre_pytest.passed} passed, {pre_pytest.failed} failed, "
            f"{pre_pytest.errors} errors"
        )
    except Exception as e:
        pytest_status = f"ERROR: {e}"
        pre_pytest = None

    # 3. Pre-check: smoke_test
    try:
        smoke = run_smoke_test(repo_path)
        # ``run_smoke_test`` returns a ``SmokeResult`` dataclass, not a dict.
        smoke_dict = smoke.to_dict()
        # SmokeResult has no "summary" key; surface a compact per-layer
        # pass/fail map so the agent sees what actually failed.
        layer_summary = {
            name: bool(info.get("passed", False)) if isinstance(info, dict) else None
            for name, info in (smoke_dict.get("layers") or {}).items()
        }
        smoke_status = (
            f"{'PASS' if smoke_dict.get('success') else 'FAIL'}: "
            f"{json.dumps(layer_summary)}"
        )
    except Exception as e:
        smoke_status = f"ERROR: {e}"

    results: Dict[str, Any] = {
        "type": "impact_review",
        "iterations": [],
        "success": False,
        "total_duration": 0.0,
    }
    start_time = time.time()
    previous_issues = ""

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        logger.info("━━━ Impact Review: iteration %d/%d ━━━", iteration, max_iterations)

        # 4. Build prompt (re-compute pytest_status for iteration 2+
        #    so the sub-agent sees post-fix state, not stale pre-fix state)
        if iteration > 1:
            try:
                re_pytest = run_pytest(
                    repo_path,
                    test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
                    timeout=120,
                    extra_args=["--timeout=30"],
                )
                pytest_status = (
                    f"{'PASS' if re_pytest.success else 'FAIL'}: "
                    f"{re_pytest.passed} passed, {re_pytest.failed} failed, "
                    f"{re_pytest.errors} errors"
                )
            except Exception as e:
                pytest_status = f"ERROR: {e}"

        prompt = build_impact_review_prompt(
            plan, impact_results, pytest_status, smoke_status,
            previous_issues=(
                f"\n## Previous Issues (iteration {iteration - 1})\n{previous_issues}"
                if previous_issues else ""
            ),
        )

        # 5. Dispatch sub-agent
        response, error = dispatch_sub_agent(
            prompt, repo_path,
            timeout=timeout,
            purpose=f"impact_review_{iteration}",
            max_retries=2,
        )

        if error:
            results["iterations"].append({
                "iteration": iteration,
                "error": error,
            })
            logger.warning("Sub-agent error on iteration %d: %s", iteration, error[:120])
            continue

        # 6. Parse result
        passed, detail = _parse_review_result(response)
        suggestions = _parse_suggestions(response)

        # 7. Post-verify (independent — don't trust sub-agent)
        post_passed = True  # default: no relevant tests = not a failure
        try:
            post_pytest = run_pytest(
                repo_path,
                test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
                timeout=120,
                extra_args=["--timeout=30"],
            )
            # 0 tests collected = no relevant tests exist → not a failure
            total = post_pytest.passed + post_pytest.failed + post_pytest.errors
            if total == 0:
                post_passed = True
            else:
                post_passed = post_pytest.success
        except Exception:
            post_passed = True  # pytest infra failure ≠ code failure

        iter_result = {
            "iteration": iteration,
            "agent_passed": passed,
            "agent_detail": detail,
            "post_pytest_passed": post_passed,
            "duration": time.time() - iter_start,
            "suggestions": suggestions,
        }
        results["iterations"].append(iter_result)

        # Early exit: agent says PASS and post-verify agrees
        if passed and post_passed:
            results["success"] = True
            break

        # Extract issues for next iteration
        if response:
            # Take last 2000 chars as context for next iteration
            previous_issues = response[-2000:]

    results["total_duration"] = time.time() - start_time
    # Aggregate suggestions from all iterations (deduplicated)
    all_suggestions: List[str] = []
    seen_suggestions: set = set()
    for it in results["iterations"]:
        for s in it.get("suggestions", []):
            if s not in seen_suggestions:
                seen_suggestions.add(s)
                all_suggestions.append(s)
    if all_suggestions:
        results["suggestions"] = all_suggestions
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Impact-scoped review for rpg_edit changes"
    )
    parser.add_argument("--plan", type=Path, default=RPG_EDIT_PLAN_FILE,
                        help="Path to rpg_edit_plan.json (default: %(default)s)")
    parser.add_argument("--impact", type=Path, default=RPG_EDIT_IMPACT_FILE,
                        help="Path to rpg_edit_impact.json (default: %(default)s)")
    parser.add_argument("--repo", type=Path, default=None,
                        help="Repository root path")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum review+repair iterations (default: 3)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Sub-agent timeout per iteration in seconds (default: 600)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    if not args.plan.exists():
        result = {"type": "error", "message": f"Plan not found: {args.plan}"}
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    # Resolve repo path: workspace root is the project repo root.
    # ``--repo`` override stays for tests / brownfield setups.
    repo_path = args.repo or REPO_DIR

    # Check if review is needed based on impact scale
    if args.impact and args.impact.exists():
        impact_data = json.loads(args.impact.read_text())
        impact_results = impact_data.get("results", {})
        total_callers, affected_files = _count_impact(impact_results)

        if total_callers == 0 and affected_files <= 1:
            result = {
                "type": "skipped",
                "reason": f"Impact too small for sub-agent review "
                          f"(callers={total_callers}, files={affected_files}). "
                          f"Agent self-review is sufficient.",
            }
            print(json.dumps(result, indent=2) if args.json else
                  f"Skipped: {result['reason']}")
            return 0

    result = impact_review(
        plan_path=args.plan,
        impact_path=args.impact,
        repo_path=repo_path,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
    )

    print(json.dumps(result, indent=2) if args.json else
          f"Review {'PASSED' if result['success'] else 'FAILED'} "
          f"({len(result['iterations'])} iterations, "
          f"{result['total_duration']:.1f}s)")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
