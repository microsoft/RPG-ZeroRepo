---
description: Edit RPG feature graph + code + dep_graph in sync, driven by natural language.
name: cmind.rpg_edit
---

## User Input

```text
$ARGUMENTS
```

**Important:** You **MUST** consider the user input before proceeding.
If the input is empty, respond immediately:

> **Edit instructions required**
>
> Please describe what you want to change. Examples:
>
> - "Add a last_login field to the User model, update it on login"
> - "Add rate limiting (10 req/s) to all API endpoints"
> - "Refactor the auth module, split registration and login into separate files"
>
> Usage: `/cmind.rpg_edit <your edit instructions>`

## Overview

`/cmind.rpg_edit` is an **independent command** that uses the RPG feature
graph as the entry point to locate modification targets, then drives
synchronized changes across **code + RPG + dep_graph**.

- Does NOT go through `feature_tree.json`.
- Does NOT depend on `/cmind.feature_edit` or `/cmind.update_rpg`.
- The RPG feature graph is the authoritative source for code modifications.

## Workflow

The text after `/cmind.rpg_edit` is the edit instruction, available as `$ARGUMENTS`.

**Working Directory**: All relative paths are based on the project root.

### Step 1: Pre-check

```bash
cmind script rpg_edit/validate.py --json
```

Inspect the `type` field:

- **`"error"`**: Display the error message and **stop**.
- **`"ready"`**: Proceed to Step 2.

### Step 2: Locate Target Nodes

```bash
cmind script rpg_edit/locate.py --query "$ARGUMENTS" --json
```

> **Note:** If `$ARGUMENTS` contains double quotes, escape them before passing.

The script returns top-K candidate feature nodes ranked by relevance,
plus a `tree_summary` showing the full RPG structure for orientation.

**Agent action**:

1. Review the candidates against the edit instruction.
2. If the edit involves **existing features** (good matches with high scores):
   select those nodes and proceed to Step 3.
3. If the edit involves **new features** that don't exist yet (all scores low,
   or the feature isn't in the tree): use the `tree_summary` to identify the
   **nearest parent node** where the new feature should be added, select that
   parent, and proceed to Step 3. In Step 4, use `"action": "add"` in
   `feature_changes`.
4. Do NOT manually explore the RPG JSON — the `tree_summary` provides the
   full structure. Do NOT ask the user to confirm which nodes.

### Step 3: Analyze Impact

For each selected node, run impact analysis and persist the result so
the Step 5d review step can pick it up automatically:

```bash
cmind script rpg_edit/impact.py --node-id <id1> [--node-id <id2> ...] --json --save
```

The `--save` flag persists `rpg_edit_impact.json` for downstream stages;
stdout still carries the JSON for you to read. Do NOT present it
separately — incorporate the results directly into Step 4.

### Step 3.5: Visual Reconnaissance (optional, before EditPlan)

This step is relevant for any edit that affects **what the user sees** —
UI, layout, styles, pages, forms, or any visual component. Failure modes
short-circuit to Step 4 with no user-visible error.

**Trigger:** Run this step when the edit instruction relates to visual
or layout concerns. Match case-insensitively against:

```text
redesign, restyle, style, css, theme, color, layout, look, look-and-feel,
ui, ux, page, screen, view, form, button, modal, navbar, sidebar,
responsive, width, height, display, adapt, resize, overflow, scroll,
grid, flex, margin, padding, font, align, position
```

If no keyword matches, skip directly to Step 4.

**Step 3.5a — Probe tool availability (≤ 5s):**

```bash
cmind script tools/browser.py check >/dev/null 2>&1 \
    && BROWSER_OK=1 || BROWSER_OK=0
```

If `BROWSER_OK=0` (Playwright not installed, headless launch failed,
no display), skip the rest of this step with a one-line note like
`Note: visual recon skipped — playwright unavailable.` and proceed to
Step 4.

**Step 3.5b — Decide what to capture:**

- For **web** projects: check if the app is running by probing common
  ports (5000, 8000, 3000, 8080). Use the first responding port.
  If the app is not running, try to start it (read `app.py` / `main.py`
  for the start command). If it cannot be started, skip this step.
- For **GUI** projects: skip — taking a screenshot requires the app to
  be launched under Xvfb, which is out of scope here.
- For **CLI / API / library** projects: skip.

**Step 3.5c — Run inspect:**

```bash
cmind script tools/browser.py inspect <url>
```

The command prints paths to the saved HTML and screenshot. Read the
HTML file to extract real class names, element structure, and any
inline styles. Cite these concretely in Step 4's `code_changes`
descriptions instead of generic phrases like "polish the form".

**Failure handling:** If `inspect` exits non-zero (server down, 404,
network error), record the failure in your reasoning and proceed to
Step 4 without recon — never block the edit on a failed probe.

### Step 4: Generate EditPlan and Confirm

**Before generating the plan, you MUST perform code reconnaissance.**
The plan must be based on **what the code actually contains**, not
assumptions from node names. Poor plans come from skipping this step.

#### Step 4a — Code Reconnaissance (MANDATORY)

1. **Read all affected source files** from Step 3's `affected_files`
   and Step 2's `meta_path` file paths.

2. **Search for related patterns** across the entire codebase. Choose
   grep patterns based on the edit type:
   - Responsive/layout: `grep -rn 'width.*[0-9]*px\|max-width.*px\|style=.*width\|height.*[0-9]*px' . --include="*.py" | grep -v __pycache__`
   - Refactor: list all files in the target module
   - Add feature: read the module where the feature will be added
   - Bug fix: read the function and all its callers

3. **For UI/visual edits**: use the screenshot and HTML from Step 3.5
   (if available) to understand the current page structure. If Step 3.5
   was skipped but the app is running, take a screenshot now:

   ```bash
   cmind script tools/browser.py inspect http://localhost:<PORT>/
   ```

4. **Collect all files that need changes** — not just the ones from
   impact analysis. The grep results reveal files the impact analysis
   may have missed (e.g., inline styles in files with no dep_graph edges).
   If grep finds files not mapped to any RPG node, include them in
   `code_changes` anyway — the code-level change is still needed even
   if no `feature_changes` entry applies. Do NOT create fake RPG nodes
   just to cover them.

#### Step 4b — Generate EditPlan

Based on the edit instruction, located nodes, impact analysis, **and
the code/visual reconnaissance above**, generate an `EditPlan` JSON.
Every file found in Step 4a that needs modification MUST be included
in `code_changes`:

```json
{
  "feature_changes": [
    {"node_id": "...", "action": "add|modify|delete", "patch": {"name": "...", "meta.path": "..."}}
  ],
  "code_changes": [
    {"file_path": "...", "change_type": "add|modify|delete", "description": "..."}
  ],
  "affected_nodes": ["node_id_1", "node_id_2"]
}
```

**Quality checklist before saving the plan:**

- [ ] Every file from the grep results that needs changes is listed
- [ ] Each `description` references specific functions/classes/lines found in the code
- [ ] No generic descriptions like "update styles" — cite exact CSS properties or function names

Save the plan via the dedicated helper, which persists
`rpg_edit_plan.json` for downstream stages and prints the absolute
path on stdout. Do NOT use the Write tool for `.cmind/` paths:

```bash
cat << 'PLAN_EOF' | cmind script rpg_edit/save_plan.py
<paste the JSON here>
PLAN_EOF
```

#### Step 4c — Present and Confirm

Then present a consolidated summary to the user:

```markdown
### Edit Plan

**Target nodes:** <node names> (from impact analysis: N callers, M callees, K affected files)

**Feature changes:**
- Modify `<node>`: ...
- Add `<node>`: ...

**Code changes:**
- `file.py`: Add field `last_login` to class `User`
- `auth.py`: Update `login()` to set `last_login`

Reply with one of:
* `Y` / `yes` / `apply`        → Continue to Step 5 (RPG-First apply)
* `N` / `no` / `cancel`        → Abort, repo and RPG unchanged
* `revise: <feedback>`         → Revise this plan using your feedback, then ask again
* `show: <node_id>`            → Expand full impact / detail for this node, then ask again
```

**This is the only user confirmation point.** Wait for one of the four
replies above before proceeding. Treat any other free-form reply as
`revise: <free-form text>`.

### Step 5: Apply Changes (RPG-First, on a dedicated branch)

All work in this step happens on a fresh `rpg-edit/<short-id>` branch
in the project repo (workspace root), never directly on `main`.  The
branch is merged into `main` only after Step 5e tests pass, so a
failed run leaves `main` clean and the branch preserved for inspection.

`<short-id>` should be derived from the plan filename or the first
affected node id (e.g. last 8 chars of `feature_changes[0].node_id`).
A timestamp suffix (`-<HHMMSS>`) is acceptable when no node id is
suitable.

**Step 5a — Pre-flight: ensure the working tree is clean and create branch:**

```bash
# Refuse to start if the working tree has uncommitted changes (avoids
# carrying user edits into the rpg-edit branch by accident).
test -z "$(git status --porcelain)" || {
  echo "Error: working tree has uncommitted changes. Commit or stash first."; exit 1;
}

git checkout -b rpg-edit/<short-id>
```

If the precondition fails, surface the error to the user and stop —
do **not** silently `git stash`, as that would hide their work.

**Step 5b — Update RPG feature graph:**

```bash
cmind script rpg_edit/apply.py --phase rpg-only --json
```

This applies `feature_changes` to the RPG and saves it (reading the plan
from the default home-dir location). Note the `backup_timestamp` from
the output — you'll need it in Step 5c and on rollback.

**Step 5c — Apply code changes via dedicated SubAgent + refresh dep_graph + commit on the branch:**

The RPG is now updated. Dispatch the code-modification SubAgent — it reads
the updated RPG and EditPlan, implements all `code_changes` in RPG-driven
mode, and the driver script creates a single commit on the current branch
(even when multiple SubAgent iterations are needed).

```bash
cmind script rpg_edit/code.py --json
```

Inspect the result `success` field:

- `true`: code applied, single commit made on the branch (SHA in `commit_sha`).
  The full result JSON is on stdout; the script also persists
  `rpg_edit_code_result.json` for later inspection. Continue to refresh
  dep_graph below.
- `false`: report `last_error` to user, do NOT refresh dep_graph,
  leave the rpg-edit branch for inspection.

If success, refresh the dep_graph and amend the existing commit so that
code + dep_graph land together:

```bash
cmind script rpg_edit/apply.py --phase dep-refresh \
    --backup-ts <timestamp_from_5b> --json

git add -A && git commit --amend --no-edit
```

**Step 5d — Test and review (still on the branch):**

1. **Smoke test** — verify imports and entry point:

```bash
cmind script smoke_test.py --json
```

1. **Impact review** — run targeted tests and verify affected functionality:

```bash
cmind script rpg_edit/review.py --json
```

The review script reads the plan and impact JSON from their default
home-dir locations and automatically:

- Derives test patterns from `code_changes` in the plan
- Runs pytest on matching test files
- Dispatches a sub-agent to verify affected callers (if impact is large enough)

Check the output `type` field:

- `"skipped"`: impact is small, pytest passed or no tests — review not needed
- `"impact_review"`: sub-agent review completed, check `success` field

If the output contains a `"suggestions"` array, save it for Step 6 —
these are related issues the review agent noticed but are outside the
current plan's scope. Present them as follow-up recommendations.

If any test step fails: fix the code on the branch, re-run dep-refresh
(Step 5c command), `git commit --amend --no-edit` to fold the
fix into the same branch commit, then re-test.

**Step 5e — Merge into `main` (only after Step 5d is green):**

```bash
git checkout main
git merge --no-ff rpg-edit/<short-id> -m "rpg_edit: merge <short-id>"
git branch -d rpg-edit/<short-id>
```

`--no-ff` preserves the merge commit so the rpg_edit boundary is
visible in `git log --graph`.

### Step 6: Report Results

- **Success path** (Step 5e completed):

  > Merged `rpg-edit/<short-id>` into `main` (commit `<merge-SHA>`).
  > To revert later:
  > - Code:  `git revert -m 1 <merge-SHA>`
  > - Graphs: `cmind script rpg_edit/apply.py --rollback <timestamp> --json`

  If the review output contained `suggestions`, append:

  > **Follow-up recommendations** (noticed during review, not in this edit's scope):
  > - <suggestion 1>
  > - <suggestion 2>
  >
  > You can address these with another `/cmind.rpg_edit` command.

- **Failure path** (Step 5d failed, Step 5e skipped):

  Restore `main` and preserve the branch for the user to inspect:

  ```bash
  git checkout main
  ```

  Report to the user:

  > Tests failed.  Branch `rpg-edit/<short-id>` preserved for inspection.
  > `main` is clean.  Choose one of:
  > - Inspect:  `git diff main rpg-edit/<short-id>`
  > - Discard code + graphs together:
  >     `cmind script rpg_edit/apply.py --rollback <timestamp> --rollback-branch rpg-edit/<short-id> --json`
  > - Discard code only:  `git branch -D rpg-edit/<short-id>`
  > - Continue editing on the branch and re-run from Step 5d.

## Key Principles

1. **RPG is the anchor** — all modifications start from RPG feature graph nodes, not from files.
2. **Three-way sync** — code, RPG, and dep_graph must stay consistent after every edit.
3. **User confirmation** — always confirm the plan before applying changes. Never auto-apply.
4. **Branch isolation** — `main` is touched only after tests pass. Failed runs leave the work on a `rpg-edit/<id>` branch for inspection.
5. **Coordinated rollback** — `--rollback <ts> --rollback-branch <name>` reverts RPG, dep_graph, and the dedicated branch in one step.
6. **Independent command** — does not depend on or invoke any other `/cmind.*` command.
