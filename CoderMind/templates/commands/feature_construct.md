---
description: Build the complete Phase 1 feature tree from requirements in one step, with automatic resume on failure
name: cmind.feature_construct
---

## User Input

```text
$ARGUMENTS
```

`/cmind.feature_construct` is the **default entry point** for Phase 1.
It orchestrates three stages and produces three artefacts in the
CoderMind data store.

| Stage | Artefact |
| --- | --- |
| `feature_spec` | `feature_spec.json` |
| `feature_build` | `feature_build.json` |
| `feature_refactor` | `feature_tree.json` |

Each stage is a standalone Python helper that drives an LLM through
`LLMClient.call_structured(...)` with a Pydantic-validated schema — no
intermediate Markdown artefacts are produced.

## Argument Parsing

Supported facade options (forward as-is to the orchestrator):

- `--check-only` — probe stage status and exit.
- `--json` — with `--check-only`, emit JSON.
- `--force` — rebuild every stage from scratch.
- `--dry-run` — list commands without executing.
- `--verbose` — DEBUG logging.
- `--no-trajectory` — skip trajectory recording.
- `--max-iter-refactor N` — override `feature_refactor.py --max-iterations`.
- `--review-threshold N` — forward to `feature_build.py`.
- `--review-max-iterations N` — forward to `feature_build.py`.

If options and requirement text are both present, split them with `--`:

```text
/cmind.feature_construct --review-threshold 99 -- Build a CLI for managing containers
```

If no options are present, treat the whole argument string as requirement
text. Requirement text is **not** forwarded to the orchestrator; instead,
when the spec stage runs the slash command passes it via
`--input-text "<text>"` to `feature_spec.py`.

### Check-only short-circuits

If the user invoked `/cmind.feature_construct --check-only` (optionally
with `--json`), run exactly:

```bash
cmind script feature_construct.py --check-only [--json]
```

Show the orchestrator output verbatim and stop. Do not inspect `docs/`,
do not ask for requirements, do not run any stage.

## Workflow

> [!WARNING]
> A full Phase 1 run typically takes a few minutes (todo-list scale)
> up to ~30 minutes for larger projects. Do not interrupt it; if you
> must, re-run this command and it will resume from the first
> incomplete stage.

### Step 1: Probe progress

```bash
cmind script feature_construct.py --check-only --json
```

Parse:

- `total` — always 3.
- `done` — count of stages whose per-stage `done` flag is `true` (artefact present and valid). The script also emits a `completed` alias with the same value for convenience.
- `next` — first incomplete stage name, or `null` if all done.
- `stages[*]` — per-stage `type` (`update` / `skip` / `run`), `message`, `done` (boolean), `will_run`, `reason`.

### Step 2: Determine requirement source

Only relevant when `feature_spec` is incomplete or the user chose to
overwrite. Priority:

1. **Requirement text after the command** — capture into a variable to
   pass via `--input-text` when the spec stage runs.
2. **`docs/*.md` files** — auto-detected by `feature_spec.py`; no flag
   needed.
3. **Inline prompt** — if neither exists, ask the user once in this
   chat for requirements, capture the text, then continue. Do not ask
   them to rerun the slash command.

If the user supplied new requirement text while any Phase 1 artefact
already exists and `--force` was **not** supplied, ask one real decision
before changing anything:

```text
Phase 1 artefacts already exist, and new requirements were provided.

What would you like to do?
  [R] Restart  — regenerate Phase 1 from the new requirements (--force)
  [E] Exit     — keep existing artefacts unchanged
```

`R` → continue with `--force`. `E` → terminate after Step 7.

### Step 3: Choose execution mode

**Case A — all three stages already complete (`done == total`)**

```text
All Phase 1 stages are already complete:
  ✓ feature_spec
  ✓ feature_build
  ✓ feature_refactor

What would you like to do?
  [X] Expand features — suggest expansion directions, expand selected ones, then rerun refactor
  [O] Overwrite       — regenerate Phase 1 from scratch (--force)
  [E] Exit            — keep existing artefacts and proceed to /cmind.plan
```

- `X` → go to **Step 6** (expansion flow).
- `O` → ensure a requirement source exists per Step 2, then rerun the
  orchestrator with `--force`.
- `E` → show the completion guidance in Step 7 and stop.

**Case B — fresh or incomplete workspace (`done < total`)**

If `done == 0`, inform the user and start immediately:

```text
Starting Phase 1 feature construction (3 stages). This may take a while.
```

If `0 < done < total`, resume from `next` automatically. Do not ask for
stage parameters unless the user explicitly requested an overwrite in
Step 2.

### Step 4: Run the orchestrator

The orchestrator (`feature_construct.py`) runs each stage in order
(`feature_spec` → `feature_build` → `feature_refactor`), skipping
already-complete stages, and validates each artefact after the stage
finishes.

If requirement text was supplied (Step 2 case 1 or 3), the slash command
must invoke `feature_spec.py` directly for the spec stage so that
`--input-text` can be passed — then resume the rest of the pipeline:

```bash
# Spec stage with inline text
cmind script feature_spec.py --input-text "<requirement text>" [--force] [other facade flags]

# Remaining stages (orchestrator skips feature_spec since it now exists)
cmind script feature_construct.py [facade flags]
```

Otherwise (auto-detected `docs/*.md` or pure resume) just run:

```bash
cmind script feature_construct.py [facade flags]
```

If restart was chosen in Step 2 or 3 (or `--force` was supplied), append
`--force` to the orchestrator command.

If `--dry-run` was supplied, stream the dry-run command list and stop
without modifying artefacts.

### Step 5: Stream output and handle failures

Stream stdout/stderr verbatim. The orchestrator prints one progress
line per stage and validates each generated artefact.

If a stage exits non-zero, surface the orchestrator's recovery hints.
Typical recovery commands:

```bash
cmind script feature_construct.py --check-only           # see which stage failed
cmind script feature_construct.py                        # resume from failed stage
cmind script feature_spec.py --verbose                   # debug spec stage
cmind script feature_build.py --verbose                  # debug build stage
cmind script feature_refactor.py --log-level DEBUG       # debug refactor stage
```

For spec-stage failures, the most common cause is an LLM call that
failed to produce a schema-valid JSON after retries. Re-run with
`--verbose` to surface the trajectory file location in the script's
own log output; do not attempt to locate it manually.

### Step 6: Optional feature expansion

Offered both after normal completion and from Case A in Step 3:

```text
Would you like to expand the feature tree beyond the current specification?
  [Y] Yes — suggest expansion directions
  [N] No  — finish here
```

If `Y`, perform one or more *expansion rounds*. Each round is a closed
cycle:

1. **Suggest directions:**

   ```bash
   cmind script feature_build.py --mode suggest-directions
   ```

   Add `--verbose` / `--no-trajectory` only if the user supplied those
   facade options.

2. Parse the JSON output and show directions as a numbered Markdown
   table.

3. Ask for comma-separated direction numbers (or `N` to finish this
   round). Normalise numeric input before passing it on.

4. **Run directed expansion:**

   ```bash
   cmind script feature_build.py --mode step2 --direction "<normalized indices>"
   ```

   Forward `--review-max-iterations`, `--verbose`, `--no-trajectory` if
   supplied. Do not forward `--review-threshold` to `step2`.

5. **Run refactor immediately** so `feature_tree.json` reflects the
   expanded `feature_build.json` and the user can inspect the actual
   tree before deciding on further rounds:

   ```bash
   cmind script feature_refactor.py
   ```

   Forward `--max-iterations <N>` when `--max-iter-refactor N` was
   supplied, `--log-level DEBUG` when `--verbose` was supplied, and
   `--no-trajectory` when supplied.

6. **Ask whether to start another round:**

   ```text
   Expansion round complete. The feature tree has been refactored.

   Would you like another expansion round?
     [Y] Yes — suggest more directions
     [N] No  — finish here
   ```

   If `Y`, repeat from step 1. If `N`, continue to Step 7.

### Step 7: Completion message

On success or when the all-complete case exits without changes:

```text
Feature construct complete.

Next step:
  /cmind.plan                        — build the Repository Planning Graph (RPG)

Optional refinements:
  Rerun this command and choose [X]  — expand the feature tree further
  /cmind.feature_edit <instructions> — adjust the final tree if it is unsatisfactory
```
