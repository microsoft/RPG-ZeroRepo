---
name: cmind.code_gen
description: Implement code using TDD workflow with iterative test-code-fix cycles
---

# Code Gen

Execute TDD-based code implementation using `run_batch.py`.
Each batch dispatches a sub-agent that autonomously writes tests, implements code,
runs pytest, and fixes issues — up to 5 iterations per attempt, 2 attempts per batch.

**Remember to re-read this document after context compact!**

## Workflow

### Step 1: Pre-Check

Run the check script to determine current state:

```bash
cmind script check_code_gen.py --json
```

**If type is "error"**:

* Fix the reported issues or run the prerequisite command first

**If type is "init"**:

* Proceed to Step 2 to initialize the codebase

**If type is "in_progress"**:

* Run `cmind script run_batch.py --resume --json` to resume

**If type is "complete"**:

* Run `--final-test`, then `--global-review`, then show summary.

### Step 2: Initialize Codebase (First Run Only)

**This step is only needed once**, before the first batch.

```bash
cmind script init_codebase.py --json
```

This creates README.md, .gitignore, base classes, and an initial commit.
Then proceed to Step 3.

### Step 3: Choose Batch Mode

Ask the user to choose a batch mode:

```text
Choose batch mode:
- S: Single-batch mode (one batch at a time)
- F: File-merge mode (merge batches per file, fewer iterations)
```

**If File-merge mode (F)**, ask:

```text
Max units per batch (0 = no limit):
```

Remember both choices for the session.

---

## Main Loop: Step 4

> **⚠️ CRITICAL: Autonomous Continuous Operation**
> This workflow MUST run autonomously in a tight loop. **NEVER stop, pause, or wait
> for user input between batches.** After completing one batch, IMMEDIATELY proceed
> to the next. The only acceptable reasons to stop are:
>
> * Unrecoverable errors (merge conflicts, script crashes)
> * All tasks completed
>
> Brief one-line status is OK (e.g., "Batch 5 done, 15/179 tasks, continuing...")
> but you MUST immediately run the next command.

### Step 4: Run Batches

`run_batch.py` handles the entire lifecycle of each batch:

1. Loads the next task and creates a git branch from main
2. Sets up the dev venv (`.venv_dev/`) with pytest + pytest-timeout
3. Dispatches a sub-agent that autonomously runs the TDD cycle
4. Post-verifies by running its own independent pytest
5. On failure: auto-retries once with resume context
6. On success: merges branch into main, updates state
7. On final failure: preserves branch, marks failed, continues

**Single-batch mode:**

```bash
cmind script run_batch.py --next --json
```

**File-merge mode (no unit limit):**

```bash
cmind script run_batch.py --next --merge-file --json
```

**File-merge mode (with unit limit):**

```bash
cmind script run_batch.py --next --merge-file --max-units <N> --json
```

**Read the JSON output:**

| Field                       | Meaning                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `type: "batch_complete"`    | Batch passed. Check `next_action` and continue.                      |
| `type: "batch_failed"`      | Batch failed after 2 attempts. Branch preserved. Continue to next.   |
| `type: "complete"`          | All tasks done. Proceed to Step 5.                                   |
| `success: false` + `error`  | Script error. Fix and retry.                                         |

**After each batch completes, IMMEDIATELY run the same command again for the next batch.**

Continue until `type` is `"complete"` or no tasks remain.

### Step 5: Final Validation

When all batches are processed:

```bash
cmind script run_batch.py --final-test --json
```

This runs pytest (full suite) and smoke test (import check, entry point, stub detection).
If smoke test reports errors, a repair agent is dispatched automatically.

### Step 5b: Global Review

After final test passes, run the global review:

```bash
cmind script run_batch.py --global-review --json
```

This dispatches a sub-agent that:

* Starts the application (web server, GUI, or CLI)
* Verifies every planned feature by actually running it
* Takes screenshots and inspects pages/screens
* Fixes bugs found during verification
* Iterates up to 10 times until all features pass

This step can be re-run independently without re-running `--final-test`.

### Step 6: Completion Summary

```text
╔══════════════════════════════════════════════════════════════╗
║                  Implementation Summary                       ║
╚══════════════════════════════════════════════════════════════╝

   Progress:
      • Completed: X/Y tasks
      • Failed: Z tasks
      • Success rate: XX.X%

   Next steps:
      • Review failed batches (branches preserved for inspection)
      • Run: cmind script run_batch.py --retry <batch_id> --json
```

---

## Additional Commands

```bash
# Resume an interrupted batch
cmind script run_batch.py --resume --json

# Retry a specific failed batch
cmind script run_batch.py --retry <batch_id> --json

# Run a specific batch by ID
cmind script run_batch.py --batch-id <id> --json

# Run a bounded smoke sample of the next N batches
cmind script run_batch.py --loop --max-batches <N> --json

# Repo validation (pytest + smoke)
cmind script run_batch.py --final-test --json

# Full feature review + visual QA
cmind script run_batch.py --global-review --json
```

## Recovery

To resume from any state:

```bash
cmind script check_code_gen.py --json
```

Follow the `next_action` field — it always tells you the exact command to run.
State is persisted in `.cmind/data/code_gen_state.jsonl`.

## Notes

* Each batch runs on its own git branch created from main
* Failed batches preserve their branch for manual inspection
* The dev venv at `.venv_dev/` is shared across all batches
* Sub-agents can install dependencies and update requirements.txt incrementally
* `run_batch.py` does NOT require manual intervention between steps
