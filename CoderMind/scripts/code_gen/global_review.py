#!/usr/bin/env python3
"""Global feature review — final QA pass of the codegen pipeline.

This module hosts the helpers extracted from ``scripts/run_batch.py``
Module 6b ("Global Review"):

* The :data:`GLOBAL_REVIEW_PROMPT` template (≈470 lines of sub-agent prompt).
* :class:`_HeartbeatLogger` — periodic-progress log for long-running calls.
* :func:`global_review` — iterative review-and-repair loop.
* A dozen private helpers (``_load_feature_spec`` / ``_build_review_prompt`` /
  ``_extract_review_checklist`` / ``_parse_review_result`` / etc.).

The orchestrator (``scripts.run_batch``) calls :func:`global_review`
from its ``--global-review`` CLI mode.  No external (non-``run_batch``)
caller imports from this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.paths import (
    FEATURE_SPEC_FILE,
    LOGS_DIR as _LOGS_DIR,
    REPO_DIR,
    TOOLS_DIR,
)
from code_gen.batch_prompts import _build_backend_test_cmd
from code_gen.stage_io import (
    save_stage_result as _save_stage_result,
    load_stage_result as _load_stage_result,
)
from code_gen.sub_agent import dispatch_sub_agent
from code_gen.test_runner import (
    ensure_deps_installed,
    get_dev_python,
    resolve_test_backend,
    run_project_tests,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Shared timeout constants
# ============================================================================

from code_gen._constants import (  # noqa: E402
    DEFAULT_PYTEST_OVERALL_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
)


# ============================================================================
# Global Review prompt template
# ============================================================================

GLOBAL_REVIEW_PROMPT = """\
# Global Review: Full Feature Verification & Quality Check

You are a QA engineer performing a comprehensive review of a completed Python
project. Your job is to:
1. Verify every planned feature works correctly by **actually running** the project
2. **Simulate real user interactions** (click buttons, fill forms, navigate pages)
3. Fix any bugs or missing functionality you find
4. **Improve visual quality** if the UI looks rough or unprofessional

## Your Workflow

### A. Read & Understand
1. Read main.py and understand how to start the project
2. Read the key source files to understand the architecture
3. Read the feature requirements listed below carefully — these are **the plan**;
   your job is to verify every planned feature is actually implemented and working

### B. Start the Project
4. Set up the environment:
   - Use the virtual environment at .venv_dev/ if it exists
   - Set environment variables as needed (e.g., DATABASE_URL=sqlite:///test_review.db)
   - Initialize the database if applicable
5. Start the project in the background
6. Verify it's running:
   - Read the startup output to find the actual port (e.g., "Running on http://127.0.0.1:5000")
   - Use that port for ALL subsequent commands (do NOT hardcode a port)
   - Bind to 127.0.0.1 only (never 0.0.0.0)
   - If the default port is occupied, use a different one and note which port you chose
   - Verify with: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:<PORT>/`

### C. Create Test Data
7. Create realistic test data through the project's own interfaces:
   - For web apps: use the app's UI or CLI commands to create users, sample
     content, and any data needed for testing (check main.py for seed commands)
   - For CLI tools: prepare input files or data fixtures
   - For GUI apps: use the GUI tools to create content interactively

### D. Verify Every Feature (Functional Completeness)
8. For EACH functional requirement listed below:
   a. Determine how to test it (HTTP request, CLI command, GUI interaction, etc.)
   b. **Actually execute** the test — don't just read the code, run it
   c. Check the response/output matches expectations
   d. Record the result as PASS or FAIL with details

### E. Visual Verification & Interactive Testing

**This step is NOT optional for web and GUI projects.** You MUST use the
provided tools to visually verify the project. Verifying only via curl/API
is insufficient — real users interact through the browser or GUI.

#### Step E1: Discover all pages/screens

For **web apps**:
- Read the source code to find all registered routes (e.g., grep for
  `@app.route`, `@bp.route`, `url_map`, router definitions)
- Start from the homepage: `inspect` it to see all links and navigation
- Follow every link found to discover all reachable pages
- Also check routes from source code that may not be linked from nav

For **GUI apps**:
- Launch the app and take an initial screenshot
- Identify all visible buttons, menus, tabs, and panels

For **CLI tools**:
- Run the help command to list all subcommands and options

#### Step E2: Inspect every page/screen

For **web apps**, use `inspect` on EVERY distinct page:
```bash
# Start from homepage — discover links, forms, navigation
python $BROWSER_TOOL inspect http://localhost:<PORT>/
# Then inspect every link found, and every route from source code
# Read the saved HTML files to understand full page content
```

For **GUI apps**, screenshot after each action:
```bash
python $GUI_TOOL start-display
python $GUI_TOOL launch "python main.py" --wait 3
python $GUI_TOOL status    # IMPORTANT: verify a window actually appeared
python $GUI_TOOL screenshot
```

**After `launch`, check the output carefully:**
- If it says "Top-level windows: N" with window titles → the GUI opened correctly
- If it says "[WARNING] No visible window detected" → the app did NOT create
  a GUI window. It may only print text to console. This is a **bug in the app**
  that you must fix before continuing. The GUI code needs to actually create
  a window (e.g., `tkinter.Tk()`, `QApplication`, `pygame.display.set_mode`).

#### Step E3: Simulate real user interactions

Don't just view pages — **interact** with them like a real user:

For **web apps**:
- Click every link and button on each page
- Fill and submit every form
- After each interaction, read the saved HTML to verify the result
- Check: did the action succeed? Did it redirect correctly? Does the
  response page show the expected content?
```bash
# Example: interact with a form (login, search, create, etc.)
python $BROWSER_TOOL run-script http://localhost:<PORT>/some-page --script '
page.click("a:has-text(\"Some Link\")")
page.wait_for_load_state("networkidle")
'
# Then read the [After] HTML file to verify the result
```

For **GUI apps**:
- Click every button, try every tool, fill every input
- After each action, take a screenshot and verify the result
- **Keep the app running** — do NOT close and relaunch between tests.
  Use one launch session for all interaction testing.
- For multi-step workflows (e.g., select tool → click canvas → set params):
  use `run-script` to chain actions reliably with proper timing.

**Simple interactions** (one action at a time):
```bash
python $GUI_TOOL click 200 150
python $GUI_TOOL screenshot
python $GUI_TOOL type "test input"
python $GUI_TOOL key "Return"
python $GUI_TOOL screenshot
```

**Complex multi-step interactions** — write to a file first, then run via
`--file`. This guarantees the script is reusable in future review iterations:
```bash
# Write the test script to the reusable scripts directory
mkdir -p .cmind/tmp/gui_test_scripts
cat > .cmind/tmp/gui_test_scripts/01_create_shape.py << 'PYEOF'
import time
# Verify: selecting a tool and using it on the canvas
gui.click(120, 45)       # open dropdown menu
time.sleep(0.3)          # wait for menu to appear
gui.click(120, 120)      # select an option
time.sleep(0.3)
gui.click(400, 300)      # click on canvas/main area
time.sleep(0.5)
gui.screenshot()         # verify result
PYEOF
# Run it
python $GUI_TOOL run-script --file .cmind/tmp/gui_test_scripts/01_create_shape.py
```
This way the script file persists and can be replayed in the next iteration.

#### Step E4: Visual quality check

Review screenshots, saved HTML files, or GUI state for EACH page/screen:

**Screenshot sanity check (critical — do this FIRST):**
- After taking a screenshot, check that it actually shows meaningful content.
  A completely black, white, or solid-color screenshot means the application
  did NOT render correctly — do NOT mark this as "expected" or "empty scene".
- For GUI apps: if the screenshot is all black, the app either crashed,
  failed to initialize its renderer, or Xvfb didn't work. This is a FAILURE,
  not "an empty scene". Investigate and fix the startup issue.
- For web apps: if the screenshot is blank, the page didn't load. Check for
  500 errors, missing templates, or broken routes.
- **Never rationalize a blank/black screen as acceptable.** A working app
  must show visible UI elements (menus, toolbars, canvas, content).

**Layout verification (critical):**
- For web: read saved HTML and check CSS layout structure. For grid/flex
  layouts, count child elements vs column/row definitions. Common bug:
  extra elements inside a grid push content into wrong columns.
- For GUI: verify widgets are positioned correctly — main content area
  should not be empty while data is squeezed into a sidebar or toolbar.
- For all: verify the PRIMARY content (data lists, forms, visualizations)
  occupies the MAIN area of the screen, not a narrow secondary panel.
  Large empty areas next to content indicate a layout bug.

**Visual quality:**
- Check for: broken layouts, missing styles, overlapping elements,
  unreadable text, inconsistent spacing, unstyled defaults
- If the UI looks rough or unprofessional, improve it:
  - Web: fix CSS (fonts, colors, spacing, borders, hover states)
  - GUI: fix widget styling (padding, alignment, font sizes, colors)
  - Ensure proper alignment and visual hierarchy across all screens
  - Make error/success feedback visually distinct

**Content rendering:**
- Verify text content displays correctly (not showing raw HTML tags,
  raw markdown syntax, or escaped entities)
- Check that formatted content (markdown, rich text) renders properly
- Ensure long content doesn't overflow its container or get clipped

After fixing visual issues, re-inspect to verify the fix.

### F. Additional Checks
9. After interactive testing, also verify:
   a. All routes/pages are reachable (no 500 errors or broken links)
   b. Forms submit correctly and show success/error feedback
   c. Navigation links lead to real pages
   d. Error handling works (invalid input, 404, unauthorized access)
   e. Data operations are consistent (create/edit/delete don't break related data)

### G. Fix Issues
10. After completing ALL verification (not after the first failure):
    - List all issues found (functional bugs, visual issues, missing features)
    - Fix them in the production code
    - Do NOT modify test files unless the test itself is clearly wrong
    - After fixing, run the full test suite to verify no regressions:
      ```
      $PYTEST_CMD
      ```
    - **After fixing, re-inspect affected pages** to confirm the fix

### H. Report Results

**Build the checklist incrementally** as you work through steps D–G:
- After verifying each FR, immediately record it in the checklist
- When you discover an issue not covered by any FR (e.g., a broken link found
  while navigating, a 500 error on an unlisted route, a visual glitch), add it
  to "Discovered Issues" right away
- This ensures nothing is forgotten even if you hit context limits

11. Clean up:
    - Stop any background project processes you started
    - Delete any test databases you created (e.g., test_review.db)
    - For GUI apps: run `gui.py close` then `gui.py stop-display`
    - For GUI apps: your test scripts in `.cmind/tmp/gui_test_scripts/`
      are already saved (you wrote them to files before running via `--file`).
      Do NOT delete them — future review iterations will replay them.
12. Output the **Review Checklist** you've been building. Use this exact format:

```
## Review Checklist

### Functional Verification
- [x] FR1: [description] — [how you verified it]
- [~] FR2: [description] — [was broken, what you fixed, verified fix]
- [ ] FR3: [description] — [what's wrong, what you tried]
- [-] FR4: [description] — [why not tested]

### Visual Quality
- [x] /page-url — clean layout, consistent nav, properly styled
- [~] /login — was unstyled, added form CSS, verified
- [ ] /search — text overlaps sidebar on narrow viewport

### Discovered Issues
- [~] /admin/users returned 500 — missing import, fixed
- [ ] Footer links point to # — not yet fixed
- [x] Missing favicon — added default icon

### Tool Usage
- Pages inspected: [N]
- Forms tested: [N]
- Screenshots taken: [N]
- Code fixes applied: [N files changed]
```

**Checklist symbols:**
- `[x]` = verified, works correctly (no action needed)
- `[~]` = was broken, you fixed it AND verified the fix
- `[ ]` = broken, could not fix (explain why)
- `[-]` = not tested (context limit, dependency on failed item, etc.)

**Rules:**
- Every FR MUST appear in "Functional Verification" (one symbol each)
- "Visual Quality" lists PAGES or SCREENS you inspected (web/GUI projects).
  For each, verify: (a) primary content occupies the main area (not squeezed
  into a sidebar or secondary panel), (b) no large empty areas next to content,
  (c) text is readable and properly styled, (d) layout matches the screen's
  purpose (e.g., main listing is prominent, not hidden or misaligned)
- "Discovered Issues" captures problems found OUTSIDE the FR list — anything
  you noticed while navigating, testing, or inspecting that isn't covered by
  a specific FR. These are just as important as FR failures.
- "Tool Usage" = counts of actual tool invocations

13. Then on the **LAST line** of your response, output EXACTLY ONE result:
- `REVIEW_RESULT: DONE | functional=N/T, visual=V/P, fixed=M, discovered=D`
  — ALL FRs verified or fixed; all pages visually checked; discovered issues resolved.
- `REVIEW_RESULT: CONTINUE | functional=N/T, visual=V/P, fixed=M, failed=K, remaining=R`
  — Made progress but some items still failed or not tested.
    The next iteration will pick up where you left off.
- `REVIEW_RESULT: BLOCKED | reason=...`
  — Cannot proceed at all (project won't start, critical crash, etc.)

**When to use each:**
- `DONE`: ONLY when ALL of these are true:
  1. Every FR is `[x]` or `[~]` in Functional Verification
  2. Every page is `[x]` or `[~]` in Visual Quality (web/GUI only)
  3. All Discovered Issues are `[x]` or `[~]` (no unresolved `[ ]`)
  4. You actually used browser/GUI tools (not just curl)
- `CONTINUE`: Whenever there are `[ ]` or `[-]` items in any section.
  This is NOT a failure — it means the next iteration will continue.
- `BLOCKED`: Only for showstopper issues that prevent ANY verification.

14. Commit your changes (if any fixes were made):
```
git add -A && git commit -m "review: fix issues found in global review"
```

## Critical Rules
- Verify ALL features, not just the first one that fails
- **Actually run and interact** with the project — don't just read source code
- **For web/GUI projects: you MUST use browser.py or gui.py tools.** Verifying
  only via curl is NOT acceptable — use `inspect` and `run-script` to simulate
  real user interactions, and read saved HTML files to analyze results
- **When gui.py reports "[WARNING] No visible window detected"**, this means
  the application has a BUG — it did not create a real GUI window. You MUST
  fix the code (e.g., add `tkinter.Tk()` to the window/display class). Do NOT
  rationalize this as "abstract API", "visualization framework", or "expected
  design". A DisplayWindow that only sets `self._is_open = True` without
  creating a real OS window is a bug that must be fixed.
- **NEVER mark GUI visual quality as N/A or "not applicable"** if the project
  specification mentions "interactive window", "GUI mode", or "visual interactive".
  The project IS a GUI application — verify it creates and renders a real window.
- Discover pages/routes from both navigation AND source code — don't assume
  you know all pages; some may not be linked from the homepage
- Create test data through the project's own interfaces (not direct DB writes)
- If the project won't start, fix the startup issue first
- Do not create new test files — only fix production code
- Run pytest after every batch of fixes to catch regressions
- Kill any background processes before finishing
- Use a separate test database (e.g., test_review.db), not the default one

## Context Limit Handling
If you are running low on context/tokens and cannot finish all FRs:
1. Complete and report as many FRs as you can
2. Commit any fixes you've already made
3. Fill out the Review Checklist — put finished items in their sections, mark
   unfinished FRs as `[-]` in Functional Verification
4. Output `REVIEW_RESULT: CONTINUE | functional=N/T, visual=V/P, fixed=M, failed=0, remaining=R`
   The next iteration will receive your checklist as context and skip verified items.

## Browser Tools Reference (for web projects)

These tools use headless Chromium via Playwright. Use them in Step E above.

**Primary command — `inspect` (recommended):**
One call = screenshot + HTML + links + forms + page structure. Use this instead
of calling screenshot/list-links/list-forms separately.
```bash
python $BROWSER_TOOL inspect http://localhost:<PORT>/
python $BROWSER_TOOL inspect http://localhost:<PORT>/login
```
Output includes: request URL, actual URL, title, screenshot path, HTML path,
all visible links, all forms with fields, and page structure (headings/nav/buttons).
**Read the saved HTML file** to analyze full page content.

**Other read-only commands** (for specific needs):
```bash
# Get page structure as text (headings, links, forms, buttons)
python $BROWSER_TOOL accessibility-tree http://localhost:<PORT>/

# Get rendered HTML of a specific element
python $BROWSER_TOOL get-html http://localhost:<PORT>/ --selector "nav"
```

**Interactive command — `run-script`** (for multi-step flows):

After run-script completes, it automatically prints:
- `[Before] URL` and `[After] URL` — track page navigation
- `[After] Screenshot` — path to auto-saved screenshot
- `[After] HTML` — path to auto-saved HTML (**read this file** to analyze the result)

```bash
# Example: fill a form and submit (adapt selectors to the actual page)
python $BROWSER_TOOL run-script http://localhost:<PORT>/some-form --script '
page.fill("input[name=field1]", "value1")
page.fill("input[name=field2]", "value2")
page.click("button[type=submit]")
page.wait_for_load_state("networkidle")
'
# After this runs, read the [After] HTML file to verify the page content
```

In run-script, you have access to: `page`, `browser`, `Path`, `json`, `print`.
Use `inspect` to analyze any page (links, forms, structure, HTML).
Use `run-script` for multi-step flows (login → navigate → verify).
After each action, **read the saved HTML file** to verify the page content.

**Quoting tip:** If your script contains single quotes (e.g., `has-text('...')`),
write the script to a temp file and use `--file` instead of `--script`:
```bash
cat > /tmp/_review_script.py << 'SCRIPT'
page.click("a:has-text('Some Link')")
page.wait_for_load_state("networkidle")
SCRIPT
python $BROWSER_TOOL run-script http://localhost:<PORT>/ --file /tmp/_review_script.py
```

## GUI Tools Reference (for desktop applications)

These tools use Xvfb + xdotool for virtual display interaction. Use them in Step E above.

**Display and app management:**
```bash
python $GUI_TOOL start-display
python $GUI_TOOL launch "python main.py" --wait 3
python $GUI_TOOL status    # verify window exists — if "No visible window", fix the app
python $GUI_TOOL screenshot
```
**CRITICAL:** After `launch`, read its output. If you see `[WARNING] No visible
window detected`, the app's GUI code is broken — it runs but doesn't create a
window. Fix the GUI initialization code before taking screenshots.

**IMPORTANT:** Keep the app running for all your testing. Do NOT close and
re-launch between each test — use one continuous session. Only restart if
you modified the app's code and need to verify the fix.

**Simple interactions** (one action at a time):
```bash
python $GUI_TOOL click 300 200
python $GUI_TOOL type "Hello World"
python $GUI_TOOL key "Return"
python $GUI_TOOL key "ctrl+s"
python $GUI_TOOL scroll -3
python $GUI_TOOL screenshot
```

**Multi-step interactions** — always write to a file first, then run via
`--file`. Scripts saved under `.cmind/tmp/gui_test_scripts/` persist across
review iterations so the next agent can replay them:
```bash
mkdir -p .cmind/tmp/gui_test_scripts
cat > .cmind/tmp/gui_test_scripts/02_form_fill.py << 'PYEOF'
import time
# Verify: dropdown selection + form fill + submit
wid = gui.find_window("My App")
if wid:
    gui.focus_window(wid)
gui.click(120, 45)       # open dropdown/menu
time.sleep(0.3)
gui.click(120, 120)      # select option
time.sleep(0.3)
gui.click(200, 150)      # click input field
gui.type_text("my value")
gui.key("Tab")
gui.type_text("another value")
gui.key("Return")
time.sleep(0.5)
gui.screenshot()         # verify result
PYEOF
python $GUI_TOOL run-script --file .cmind/tmp/gui_test_scripts/02_form_fill.py
```

**Simple one-off scripts** (no need to persist):
```bash
python $GUI_TOOL run-script --script 'gui.click(100, 200); import time; time.sleep(0.3); gui.screenshot()'
```

In run-script: `gui` (GuiHelper with click/type_text/key/scroll/screenshot/
find_window/focus_window), `subprocess`, `Path`, `time`, `print`.
**Always clean up when finished:** `gui.py close` then `gui.py stop-display`.

---

## Project Context

**Repository:** $REPO_PATH

### Functional Requirements (from feature spec)
$REQUIREMENTS_TEXT

### Source Files
$FILE_LIST

### Reusable GUI Interaction Scripts (if any)
$GUI_SCRIPT_REUSE_CONTEXT

### Previous Issues (unresolved from last iteration)
$PREVIOUS_ISSUES
"""


# ============================================================================
# Helpers and main entry point
# ============================================================================


def _load_feature_spec() -> Dict[str, Any]:
    """Load feature_spec.json for review prompt."""
    if FEATURE_SPEC_FILE.exists():
        try:
            with open(FEATURE_SPEC_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_all_stage_findings_for_prompt() -> str:
    """Build a compact summary of failed stages with file paths.

    Only includes stages that have errors. Provides status + one-line
    summary + absolute file path so the agent can read details on demand,
    keeping the prompt small.
    """
    parts: List[str] = []

    # 1. Pytest failures
    ft = _load_stage_result("final_test")
    ft_path = _LOGS_DIR / "codegen_final_test.json"
    if ft and not ft.get("success", True):
        parts.append(
            f"- **pytest**: FAILED ({ft.get('failed', 0)} failures, "
            f"{ft.get('errors', 0)} errors). "
            f"Fix these FIRST. Details: `cat {ft_path}`"
        )

    # 2. Smoke test errors
    st = _load_stage_result("smoke_test")
    st_path = _LOGS_DIR / "codegen_smoke_test.json"
    if st:
        error_count = len([f for f in st.get("findings", []) if f.get("severity") == "error"])
        if error_count > 0:
            repair_note = ""
            if st.get("repair_attempted"):
                repair_note = f" (repair attempted, {st.get('repair_remaining', '?')} remaining)"
            parts.append(
                f"- **smoke_test**: {error_count} error(s){repair_note}. "
                f"Details: `cat {st_path}`"
            )

    # 3. Previous global review
    gr = _load_stage_result("global_review")
    gr_path = _LOGS_DIR / "codegen_global_review.json"
    if gr:
        iters = gr.get("iterations", [])
        if iters and not iters[-1].get("review_passed"):
            detail = iters[-1].get("detail", "")[:80]
            parts.append(
                f"- **global_review**: {len(iters)} iteration(s), "
                f"last: {detail}. Details: `cat {gr_path}`"
            )

    if not parts:
        return ""
    return (
        "**Prior stage failures** (read the file for full details):\n"
        + "\n".join(parts)
    )


def _format_requirements_for_review(feature_spec: Dict[str, Any]) -> str:
    """Format functional requirements for the review prompt.

    Handles the feature_spec.json structure:
      { "functional_requirements": [
          { "id": "FT-001", "name": "...", "description": "...",
            "children": [ { "id": "FT-001-001", "name": "...", ... } ] }
      ] }
    """
    frs = feature_spec.get("functional_requirements", [])
    if not frs:
        return "(No functional requirements found)"

    parts = []
    for i, fr in enumerate(frs, 1):
        name = fr.get("name", fr.get("title", ""))
        desc = fr.get("description", "")
        text = f"{name}: {desc}" if desc else name

        # Recursively collect children (supports arbitrary depth)
        children = fr.get("children", fr.get("sub_features", fr.get("features", [])))
        if children:
            sub_texts = _collect_children(children, depth=1)
            text += "\n" + "\n".join(sub_texts)
        parts.append(f"FR{i}: {text}")

    return "\n\n".join(parts)


def _collect_children(children: list, depth: int = 1, max_depth: int = 3) -> List[str]:
    """Recursively collect child feature descriptions."""
    lines = []
    indent = "  " * depth
    for child in children:
        name = child.get("name", child.get("title", ""))
        desc = child.get("description", "")
        line = f"{indent}- {name}"
        if desc and desc != name:
            line += f": {desc}"
        lines.append(line)
        if depth < max_depth:
            sub = child.get("children", child.get("sub_features", []))
            if sub:
                lines.extend(_collect_children(sub, depth + 1, max_depth))
    return lines


def _load_gui_script_reuse_context(repo_path: Path) -> str:
    """Load reusable GUI interaction scripts for review prompt context.

    Scripts are stored under ``repo/.cmind/tmp/gui_test_scripts`` and
    capture stable interaction flows that the review prompt can reuse.
    """
    scripts_dir = repo_path / ".cmind" / "tmp" / "gui_test_scripts"
    if not scripts_dir.is_dir():
        return "(No reusable GUI scripts found yet)"

    files = sorted(p for p in scripts_dir.glob("*.py") if p.is_file())
    if not files:
        return "(No reusable GUI scripts found yet)"

    # Keep prompt size bounded.
    max_chars = 8000
    used = 0
    chunks: List[str] = [
        "Use these scripts first when testing matching GUI flows. "
        "If a script fails due to UI changes, update it and continue."
    ]

    for idx, f in enumerate(files):
        rel = f.relative_to(repo_path)
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            content = "# (failed to read script)"

        block = f"\n- {rel}\n```python\n{content}\n```\n"
        if used + len(block) > max_chars:
            remaining = len(files) - idx
            if remaining > 0:
                chunks.append(f"\n... ({remaining} more scripts omitted for size)\n")
            break

        chunks.append(block)
        used += len(block)

    return "\n".join(chunks)


def _build_review_prompt(repo_path: Path, previous_issues: str = "") -> str:
    """Build the global review prompt.

    Uses $VAR substitution instead of .format() to avoid conflicts with
    curly braces in example code (e.g., f-string {path} in GUI examples).
    """
    feature_spec = _load_feature_spec()
    requirements_text = _format_requirements_for_review(feature_spec)

    # Reuse smoke_test's file finder for consistency
    try:
        from smoke_test import _find_source_files
        source_files = _find_source_files(repo_path)
        file_lines = []
        for f in source_files:
            rel = f.relative_to(repo_path)
            size = f.stat().st_size
            file_lines.append(f"  {rel} ({size} bytes)")
        file_list = "\n".join(file_lines)
    except ImportError:
        file_list = "(file listing unavailable)"

    venv_python = get_dev_python(repo_path) or "python3"
    backend = resolve_test_backend(repo_path=repo_path)
    pytest_cmd = _build_backend_test_cmd(backend, repo_path, [], venv_python)
    gui_script_reuse_context = _load_gui_script_reuse_context(repo_path)

    # Load accumulated findings from all pipeline stages
    findings_text = _load_all_stage_findings_for_prompt()
    if findings_text:
        if previous_issues:
            previous_issues = findings_text + "\n\n---\n\n" + previous_issues
        else:
            previous_issues = findings_text

    # Absolute paths for tools so the prompt is cwd-agnostic.
    browser_tool = str(TOOLS_DIR / "browser.py")
    gui_tool = str(TOOLS_DIR / "gui.py")

    # Use string.Template for safe substitution ($VAR style)
    from string import Template
    tmpl = Template(GLOBAL_REVIEW_PROMPT)
    return tmpl.safe_substitute(
        REPO_PATH=str(repo_path),
        REQUIREMENTS_TEXT=requirements_text,
        FILE_LIST=file_list,
        GUI_SCRIPT_REUSE_CONTEXT=gui_script_reuse_context,
        PREVIOUS_ISSUES=previous_issues or "(First iteration — no previous issues)",
        PYTEST_CMD=pytest_cmd,
        BROWSER_TOOL=browser_tool,
        GUI_TOOL=gui_tool,
    )


def _extract_review_checklist(response: str) -> Dict[str, Any]:
    """Extract the structured Review Checklist from a review response.

    Parses three content sections (Functional Verification, Visual Quality,
    Discovered Issues) and Tool Usage metrics. Each item is classified by
    its checkbox symbol: [x] verified, [~] fixed, [ ] failed, [-] not tested.

    Returns dict with:
        functional: dict with verified/fixed/failed/not_tested lists
        visual: dict with verified/fixed/failed/not_tested lists
        discovered: dict with verified/fixed/failed/not_tested lists
        pages_inspected: int
        forms_tested: int
        screenshots_taken: int
    """
    def _empty_section() -> Dict[str, list]:
        return {"verified": [], "fixed": [], "failed": [], "not_tested": []}

    _symbol_to_key = {"x": "verified", "~": "fixed", " ": "failed", "-": "not_tested"}

    result: Dict[str, Any] = {
        "functional": _empty_section(),
        "visual": _empty_section(),
        "discovered": _empty_section(),
        "pages_inspected": 0,
        "forms_tested": 0,
        "screenshots_taken": 0,
    }

    # Map heading text → which content section to fill
    section = None  # current content section key or "_tools"
    heading_map = {
        "functional verification": "functional",
        "functional": "functional",
        "visual quality": "visual",
        "visual": "visual",
        "discovered issues": "discovered",
        "discovered": "discovered",
        "tool usage": "_tools",
        # Legacy headings (backward compat)
        "verified": "functional",
        "fixed": "functional",
        "failed": "functional",
        "not tested": "functional",
        "not_tested": "functional",
    }
    # For legacy single-section format, track which symbol bucket to force
    _legacy_force_key: Optional[str] = None
    _legacy_key_map = {
        "verified": "verified", "fixed": "fixed",
        "failed": "failed", "not tested": "not_tested",
    }

    for line in response.splitlines():
        stripped = line.strip()

        # Detect section headers: ### Functional Verification, ### Visual Quality, etc.
        if stripped.startswith("###"):
            heading = stripped.lstrip("# ").strip().lower()
            matched = heading_map.get(heading)
            if matched:
                section = matched
                # Legacy: if heading is "Verified"/"Fixed"/etc., force that key;
                # otherwise reset to None so new-format sections use symbol-based classification
                _legacy_force_key = _legacy_key_map.get(heading)  # None for new-format headings
            continue

        # Parse checklist items: - [x], - [~], - [ ], - [-]
        if section and section != "_tools":
            m = re.match(r'^-\s*\[([ x~-])\]\s*(.+)', stripped)
            if m:
                symbol = m.group(1)
                text = m.group(2).strip()
                if _legacy_force_key:
                    # Legacy format: heading determines the key
                    result[section][_legacy_force_key].append(text)
                else:
                    # New format: symbol determines the key
                    key = _symbol_to_key.get(symbol, "failed")
                    result[section][key].append(text)
                continue

        # Parse Tool Usage metrics
        if section == "_tools":
            m = re.match(r'^-\s*[Pp]ages inspected:\s*(\d+)', stripped)
            if m:
                result["pages_inspected"] = int(m.group(1))
                continue
            m = re.match(r'^-\s*[Ff]orms tested:\s*(\d+)', stripped)
            if m:
                result["forms_tested"] = int(m.group(1))
                continue
            m = re.match(r'^-\s*[Ss]creenshots taken:\s*(\d+)', stripped)
            if m:
                result["screenshots_taken"] = int(m.group(1))
                continue

    # Fallback: also try old-style FR lines (FR1: ... — PASS/FAIL)
    func = result["functional"]
    if not func["verified"] and not func["fixed"] and not func["failed"]:
        for line in response.splitlines():
            stripped = line.strip()
            if re.match(r'^FR\d+:', stripped):
                if 'PASS' in stripped:
                    func["verified"].append(stripped)
                elif 'FAIL' in stripped:
                    func["failed"].append(stripped)

    return result


def _parse_review_result(response: Optional[str]) -> Tuple[bool, str]:
    """Parse the sub-agent's review result.

    Handles both new format (DONE/CONTINUE/BLOCKED) and legacy format
    (PASS/PARTIAL/FAIL/CONTEXT_LIMIT). Handles markdown formatting.
    Returns (passed, detail).
    """
    if not response:
        return False, "no response"

    lines = response.strip().splitlines()
    search_lines = lines[-30:] if len(lines) > 30 else lines
    for line in reversed(search_lines):
        # Strip whitespace and markdown bold/italic markers
        line = line.strip().strip("*").strip("_").strip()
        if line.startswith("REVIEW_RESULT:"):
            rest = line[len("REVIEW_RESULT:"):].strip()
            # New format
            if rest.startswith("DONE"):
                return True, rest
            elif rest.startswith("CONTINUE"):
                return False, rest
            elif rest.startswith("BLOCKED"):
                return False, rest
            # Legacy format (backward compat)
            elif rest == "PASS":
                return True, "DONE (legacy PASS)"
            elif rest.startswith("PARTIAL"):
                return False, f"CONTINUE (legacy {rest})"
            elif rest.startswith("FAIL"):
                return False, rest
            elif rest.startswith("CONTEXT_LIMIT"):
                return False, f"CONTINUE (legacy {rest})"

    return False, "no REVIEW_RESULT found in response"


def _build_review_retry_context(
    response: str,
    post_pytest: Any,
    post_stubs: List[str],
    detail: str = "",
    checklist: Optional[Dict[str, Any]] = None,
) -> str:
    """Build context from previous iteration for retry prompt.

    Uses the structured checklist (functional/visual/discovered) to pass
    completed/failed items to the next iteration.
    """
    parts: List[str] = []

    # Check if previous iteration didn't finish or was blocked
    if "BLOCKED" in detail:
        parts.append(
            "IMPORTANT: The previous iteration was BLOCKED.\n"
            f"Reason: {detail}\n"
            "Diagnose and fix the blocker, then continue verification.\n"
        )
    elif "CONTINUE" in detail:
        parts.append(
            "IMPORTANT: The previous iteration did not finish all items.\n"
            "Continue reviewing from where it left off.\n"
            "DO NOT re-verify items already marked [x] or [~] below.\n"
        )

    # Use structured checklist if available
    if checklist:
        for section_name, label in [
            ("functional", "Functional Verification"),
            ("visual", "Visual Quality"),
            ("discovered", "Discovered Issues"),
        ]:
            sec = checklist.get(section_name, {})
            if not sec:
                continue
            verified = sec.get("verified", [])
            fixed = sec.get("fixed", [])
            failed = sec.get("failed", [])
            not_tested = sec.get("not_tested", [])
            if not any([verified, fixed, failed, not_tested]):
                continue

            parts.append(f"\n### {label} (from previous iteration)")
            if verified:
                parts.append(f"Already verified ({len(verified)}) — skip:")
                parts.extend(f"  [x] {item}" for item in verified[:30])
            if fixed:
                parts.append(f"Fixed ({len(fixed)}):")
                parts.extend(f"  [~] {item}" for item in fixed[:20])
            if failed:
                parts.append(f"Still FAILED ({len(failed)}) — fix these:")
                parts.extend(f"  [ ] {item}" for item in failed[:20])
            if not_tested:
                parts.append(f"Not yet tested ({len(not_tested)}) — verify these:")
                parts.extend(f"  [-] {item}" for item in not_tested[:20])

        pages = checklist.get("pages_inspected", 0)
        forms = checklist.get("forms_tested", 0)
        if pages or forms:
            parts.append(f"\nPrevious tool usage: pages_inspected={pages}, forms_tested={forms}")
    else:
        # Fallback: extract FR lines from raw response
        fr_lines = [
            line.strip() for line in response.splitlines()
            if line.strip().startswith("FR") and ("PASS" in line or "FAIL" in line)
        ]
        if fr_lines:
            pass_lines = [fr for fr in fr_lines if "PASS" in fr]
            fail_lines = [fr for fr in fr_lines if "FAIL" in fr]
            if pass_lines:
                parts.append(f"Features already verified PASS ({len(pass_lines)}):")
                parts.extend(f"  {line}" for line in pass_lines[:30])
            if fail_lines:
                parts.append(f"\nFeatures that FAIL ({len(fail_lines)}):")
                parts.extend(f"  {line}" for line in fail_lines[:20])
        else:
            fail_lines = [
                line.strip() for line in response.splitlines()
                if "FAIL" in line and "FR" in line
            ]
            if fail_lines:
                parts.append("Features that still FAIL:")
                parts.extend(f"  {line}" for line in fail_lines[:20])

    # Post-pytest failures
    if not post_pytest.success:
        parts.append(
            f"\npytest regressions: {post_pytest.failed} failed, "
            f"{post_pytest.errors} errors"
        )
        output_tail = "\n".join(post_pytest.output.splitlines()[-30:])
        parts.append(f"pytest output (tail):\n{output_tail}")

    # Remaining stubs
    if post_stubs:
        parts.append(f"\nRemaining stubs ({len(post_stubs)}):")
        parts.extend(f"  {s}" for s in post_stubs[:10])

    return "\n".join(parts)


def _cleanup_background_processes(repo_path: Path) -> None:
    """Best-effort cleanup of Python processes started in repo_path."""
    import signal
    import subprocess as _sp
    try:
        # Use lsof to find processes with files open in repo_path
        # This is more targeted than pgrep -f which matches too broadly
        result = _sp.run(
            ["lsof", "+D", str(repo_path), "-t"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            my_pid = os.getpid()
            my_ppid = os.getppid()
            for pid_str in result.stdout.strip().splitlines():
                try:
                    pid = int(pid_str.strip())
                except ValueError:
                    continue
                if pid in (my_pid, my_ppid):
                    continue  # Don't kill ourselves or parent
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info("Cleaned up process %d", pid)
                except OSError:
                    pass
    except FileNotFoundError:
        # lsof not available — skip cleanup, prompt already asks agent to clean up
        logger.debug("lsof not found, skipping process cleanup")
    except Exception as exc:
        logger.debug("Process cleanup failed (non-fatal): %s", exc)

    # Also clean up any Xvfb processes on the default display
    try:
        result = _sp.run(
            ["pgrep", "-f", "Xvfb :99( |$)"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().splitlines():
                try:
                    pid = int(pid_str.strip())
                    os.kill(pid, signal.SIGTERM)
                    logger.info("Cleaned up Xvfb process %d", pid)
                except (ValueError, OSError):
                    pass
    except Exception:
        pass


# ============================================================================
# E2 — Heartbeat helper for long-running sub-agent calls
# ============================================================================

class _HeartbeatLogger:
    """Emit a periodic ``...still running, elapsed=Xs`` log line.

    Designed to wrap a single long-running blocking call (typically
    ``dispatch_sub_agent`` inside a global_review iteration). Exits
    cleanly via context manager — the daemon thread stops as soon as
    ``__exit__`` runs, even when the wrapped call raises.
    """

    def __init__(self, label: str, interval_s: int = 60) -> None:
        self._label = label
        self._interval = max(1, int(interval_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_HeartbeatLogger":
        start = time.time()

        def _beat() -> None:
            while not self._stop.wait(self._interval):
                elapsed = int(time.time() - start)
                logger.info("%s ...still running, elapsed=%ds", self._label, elapsed)

        self._thread = threading.Thread(
            target=_beat, name=f"heartbeat:{self._label}", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        # Don't join longer than one tick — the thread is daemon and will
        # exit on its own at process shutdown if join fails.
        if self._thread is not None:
            self._thread.join(timeout=2)


def global_review(
    repo_path: Optional[Path] = None,
    max_iterations: int = 10,
    timeout_per_iteration: int = 1800,
) -> Dict[str, Any]:
    """Run global feature review with iterative repair.

    Dispatches a sub-agent as QA engineer to verify every feature
    against the feature_spec, fix issues, and iterate until all pass.

    Args:
        repo_path: Project repo path.
        max_iterations: Max review+repair cycles.
        timeout_per_iteration: Sub-agent timeout per iteration (seconds).

    Returns:
        Result dict with review findings and statistics.
    """
    repo_path = repo_path or REPO_DIR

    results: Dict[str, Any] = {
        "type": "global_review",
        "iterations": [],
        "success": False,
        "total_duration": 0.0,
    }
    start_time = time.time()
    previous_issues = ""
    backend = resolve_test_backend(repo_path=repo_path)

    for iteration in range(1, max_iterations + 1):
        logger.info("━━━ Global Review: iteration %d/%d ━━━", iteration, max_iterations)

        # Clean screenshots from previous iteration so size check is fresh
        try:
            screenshots_dir = repo_path / ".cmind" / "tmp" / "screenshots"
            if screenshots_dir.is_dir():
                shutil.rmtree(screenshots_dir)
        except Exception:
            pass

        # 1. Pre-check: run the project's tests to know current state
        if backend.name == "python":
            try:
                ensure_deps_installed(repo_path)
            except Exception:
                pass
        pre_pytest = run_project_tests(
            repo_path,
            timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
            extra_args=[f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread"],
            backend=backend,
        )
        # Update stage file so _build_review_prompt sees fresh state
        _save_stage_result("final_test", {
            "success": pre_pytest.success,
            "passed": pre_pytest.passed,
            "failed": pre_pytest.failed,
            "errors": pre_pytest.errors,
            "output_tail": "\n".join(pre_pytest.output.splitlines()[-40:]) if not pre_pytest.success else "",
        })
        if not pre_pytest.success:
            logger.warning(
                "pytest pre-check: %d failures, %d errors — agent will fix",
                pre_pytest.failed, pre_pytest.errors,
            )

        # 2. Build prompt
        prompt = _build_review_prompt(repo_path, previous_issues=previous_issues)

        # 3. Dispatch sub-agent (with retries for transient failures).
        #    Wrap with a heartbeat so the operator sees the iteration is
        #    still alive even if the sub-agent runs for many minutes
        #    without producing output.
        with _HeartbeatLogger(
            label=f"global_review[{iteration}/{max_iterations}]",
            interval_s=60,
        ):
            response, error = dispatch_sub_agent(
                prompt, repo_path,
                timeout=timeout_per_iteration,
                purpose=f"global_review_{iteration}",
                max_retries=3,
            )

        if error:
            results["iterations"].append({
                "iteration": iteration,
                "error": error,
            })
            # Don't break — transient LLM failures shouldn't abort the
            # entire review.  Continue to the next iteration which will
            # rebuild the prompt and retry.
            logger.warning(
                "Dispatch error on iteration %d, will retry next iteration: %s",
                iteration, error[:120],
            )
            if iteration == max_iterations:
                logger.error("Dispatch error on final iteration — giving up")
            continue

        # 4. Parse sub-agent result and extract checklist
        review_passed, detail = _parse_review_result(response)
        checklist = _extract_review_checklist(response) if response else None

        # 5. Post-verify (independent — don't trust sub-agent)
        _cleanup_background_processes(repo_path)

        post_pytest = run_project_tests(
            repo_path,
            timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
            extra_args=["-v", "--tb=short", f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread"],
            backend=backend,
        )

        # Stub check
        post_stubs: List[str] = []
        try:
            from code_gen.static_checks import static_completeness_check
            from smoke_test import _find_source_files
            source_files = _find_source_files(repo_path)
            file_paths = [str(f.relative_to(repo_path)) for f in source_files]
            post_stubs = [
                s for s in static_completeness_check(file_paths, repo_path)
                if s.startswith("STUB:")
            ]
        except Exception as exc:
            logger.debug("Stub check during review post-verify failed: %s", exc)

        # Framework-level tool usage validation:
        # If agent claims DONE but never inspected any pages, override to CONTINUE
        tools_used = False
        if checklist:
            pages = checklist.get("pages_inspected", 0)
            screenshots = checklist.get("screenshots_taken", 0)
            tools_used = pages > 0 or screenshots > 0
            if review_passed and not tools_used:
                logger.warning(
                    "Agent reported DONE but pages_inspected=%d, screenshots=%d "
                    "— overriding to CONTINUE (visual verification required)",
                    pages, screenshots,
                )
                review_passed = False
                detail = (
                    "CONTINUE (overridden: agent claimed DONE but did not use "
                    "browser/GUI tools for visual verification)"
                )

        # Override DONE if post-pytest failed (agent's fixes may have broken tests)
        if review_passed and not post_pytest.success:
            logger.warning(
                "Agent reported DONE but post-pytest FAILED (%d failures, %d errors) "
                "— overriding to CONTINUE",
                post_pytest.failed, post_pytest.errors,
            )
            review_passed = False
            detail = (
                f"CONTINUE (overridden: post-pytest failed with "
                f"{post_pytest.failed} failures, {post_pytest.errors} errors)"
            )

        # Framework-level screenshot content validation:
        # Check if screenshots are suspiciously small (black/empty = ~250 bytes).
        # A normal screenshot with actual content is at least a few KB.
        # This check uses filesystem evidence directly, not the checklist's
        # self-reported metrics (which may be missing or malformed).
        if review_passed:
            try:
                screenshots_dir = repo_path / ".cmind" / "tmp" / "screenshots"
                if screenshots_dir.is_dir():
                    png_files = list(screenshots_dir.glob("*.png"))
                    if png_files:
                        small_count = sum(
                            1 for f in png_files if f.stat().st_size < 1000
                        )
                        # Fail if majority of screenshots are blank
                        if small_count > 0 and small_count >= len(png_files) * 0.5:
                            logger.warning(
                                "%d/%d screenshots are < 1KB (likely blank/black) "
                                "— overriding to CONTINUE",
                                small_count, len(png_files),
                            )
                            review_passed = False
                            detail = (
                                f"CONTINUE (overridden: {small_count}/{len(png_files)} "
                                f"screenshots are < 1KB, likely blank/black — "
                                f"GUI may not have rendered)"
                            )
            except Exception as exc:
                logger.debug("Screenshot size check failed: %s", exc)

        # Check for unresolved items in any checklist section
        has_unresolved = False
        if checklist:
            for sec_name in ("functional", "visual", "discovered"):
                sec = checklist.get(sec_name, {})
                if sec.get("failed"):
                    has_unresolved = True
                    break
                # For functional/visual: not_tested items are blockers.
                # For discovered: not_tested items are best-effort (bonus
                # findings the agent couldn't verify), not blockers.
                if sec_name != "discovered" and sec.get("not_tested"):
                    has_unresolved = True
                    break
            if review_passed and has_unresolved:
                logger.warning(
                    "Agent reported DONE but checklist has unresolved items "
                    "— overriding to CONTINUE"
                )
                review_passed = False
                detail = "CONTINUE (overridden: checklist has unresolved [ ] or [-] items)"

        # Build iteration result with checklist stats per section
        cl_stats: Dict[str, Any] = {}
        if checklist:
            for sec_name in ("functional", "visual", "discovered"):
                sec = checklist.get(sec_name, {})
                total = sum(len(sec.get(k, [])) for k in ("verified", "fixed", "failed", "not_tested"))
                if total > 0:
                    cl_stats[sec_name] = {
                        "verified": len(sec.get("verified", [])),
                        "fixed": len(sec.get("fixed", [])),
                        "failed": len(sec.get("failed", [])),
                        "not_tested": len(sec.get("not_tested", [])),
                    }
            cl_stats["pages_inspected"] = checklist.get("pages_inspected", 0)
            cl_stats["forms_tested"] = checklist.get("forms_tested", 0)
            cl_stats["screenshots_taken"] = checklist.get("screenshots_taken", 0)

        iteration_result = {
            "iteration": iteration,
            "review_passed": review_passed,
            "post_pytest_pass": post_pytest.success,
            "post_stub_count": len(post_stubs),
            "detail": detail,
            "tools_used": tools_used,
            "checklist": cl_stats,
        }
        results["iterations"].append(iteration_result)

        logger.info(
            "Review iteration %d: agent=%s, pytest=%s, stubs=%d, "
            "tools=%s, checklist=%s, detail=%s",
            iteration,
            "DONE" if review_passed else "CONTINUE",
            "PASS" if post_pytest.success else "FAIL",
            len(post_stubs),
            "yes" if tools_used else "NO",
            cl_stats if cl_stats else "none",
            detail[:80],
        )

        # 6. Decision
        if review_passed and post_pytest.success and len(post_stubs) == 0:
            results["success"] = True
            logger.info("Global review DONE on iteration %d", iteration)
            break

        # Build context for next iteration (pass checklist for structured retry)
        if response:
            previous_issues = _build_review_retry_context(
                response, post_pytest, post_stubs,
                detail=detail, checklist=checklist,
            )
        else:
            previous_issues = "Previous iteration produced no response."

        if iteration == max_iterations:
            logger.warning(
                "Global review reached max iterations (%d) without full pass",
                max_iterations,
            )

    results["total_duration"] = round(time.time() - start_time, 1)

    # Persist results for cross-stage context
    try:
        _save_stage_result("global_review", results)
    except Exception:
        pass

    return results


