---
description: Build the complete Phase 2 Repository Planning Graph (RPG) from the feature tree in one step, with automatic resume on failure
name: cmind.plan
---

## User Input

```text
$ARGUMENTS
```

`$ARGUMENTS` is forwarded verbatim to `cmind script plan.py` (for
example, `--verbose`, `--max-iter-skeleton 15`, or `--force`).
If empty, proceed with default behavior.

## **Outline**

Given the feature tree produced by `/cmind.feature_construct`, this
command builds the complete Repository Planning Graph (RPG) in a single
non-interactive run with automatic resume on failure.

> [!WARNING]
> A full pipeline run can take from a few minutes to over an hour
> depending on project size. Set your terminal timeout to at least
> **240 minutes** before running. Do **not** interrupt it; if you
> must, re-run this command and it will resume from where it stopped.

### Step 1: Probe progress

Run the orchestrator in probe mode and capture the JSON report:

```bash
cmind script plan.py --check-only --json
```

Parse the JSON. The fields you need:

* `total` — total number of stages (always 5)
* `done`  — count of stages whose `type` is `update`
* `next`  — name of the first not-done stage (or `null` if all done)
* `stages[*].name`, `stages[*].type` (`update` / `warning` / `init` /
  `error`), `stages[*].done`

Treat `warning` as **not done**. A warning means the artifact exists but
violates a cross-stage contract (for example, `interfaces.json` does not
cover all `skeleton.json` features). Do not skip the stage, do not run a
later stage directly, and do not create downstream artifacts from a
warning-state input.

### Step 2: One decision (the only prompt of this command)

Choose **exactly one** case based on `done` vs `total`:

**Case A — Everything already done (`done == total`):**

Display this prompt and wait for the user's choice:

```text
All 5 planning stages are already complete:
  ✓ skeleton
  ✓ data_flow
  ✓ base_classes
  ✓ interfaces
  ✓ tasks

What would you like to do?
  [O] Overwrite — regenerate everything from scratch
  [E] Exit      — keep existing artifacts and proceed to /cmind.code_gen
```

* `O` → execute: `cmind script plan.py --force $ARGUMENTS`
* `E` → terminate this command; remind the user that `/cmind.code_gen`
  is the next step.

**Case B — Fresh workspace (`done == 0`):**

Do **not** prompt. Briefly inform the user and proceed:

```text
Starting the full planning pipeline (5 stages). This may take a while.
```

Then execute: `cmind script plan.py $ARGUMENTS`

**Case C — Partial progress (`0 < done < total`):**

Display this prompt. Glyph per stage:

* `stages[*].done == true` → `✓`
* the first not-done stage → `▸`
* every other not-done stage → `·`

(Per-stage warning details, if any, are surfaced by `plan.py`'s own
stdout when it runs; the user-facing prompt only conveys done/not-done.)

```text
Planning is partially complete: <done>/<total> stages done.
  <glyph> skeleton
  <glyph> data_flow
  <glyph> base_classes
  <glyph> interfaces
  <glyph> tasks

Last completed stage: <last_done or "(none)">
Stopped at: <next>

What would you like to do?
  [C] Continue — resume from `<next>` and finish the pipeline
  [R] Restart  — discard progress and regenerate everything
  [E] Exit     — do nothing
```

* `C` → execute: `cmind script plan.py $ARGUMENTS`
* `R` → execute: `cmind script plan.py --force $ARGUMENTS`
* `E` → terminate this command.

### Step 3: Stream the orchestrator's output

When you execute the orchestrator (cases A → O, B, C → C/R above),
stream its stdout/stderr to the user as-is. The orchestrator already
prints one progress line per stage and a final summary; do not add
your own commentary on top of every line.

### Step 4: On failure

If the orchestrator exits non-zero, it has already printed a
`✗ <stage> ... failed` line plus three recovery hints. Surface those
hints verbatim. The most common follow-ups are:

```bash
# Re-check progress (no side effects).
cmind script plan.py --check-only

# Resume from where it failed (default behavior).
cmind script plan.py

# Debug a single stage interactively.
cmind script <stage>.py --verbose
```

### Step 5: On success

Tell the user:

```text
Planning pipeline complete.

Next:
  /cmind.code_gen        — generate source code from the plan
```
