---
description: Run Phase 1 feature specification, feature building, and feature refactoring in one step with automatic resume
name: cmind.feature_construct
---

## User Input

```text
$ARGUMENTS
```

This command consolidates `/cmind.feature_spec`, `/cmind.feature_build`, and `/cmind.feature_refactor` into the recommended one-step Phase 1 flow. The granular commands remain available for debugging, surgical reruns, and single-stage recovery.

## Argument Parsing

Supported facade options:

- `--check-only`
- `--json`
- `--force`
- `--dry-run`
- `--verbose`
- `--no-trajectory`
- `--max-iter-refactor N`
- `--review-threshold N`
- `--review-max-iterations N`

If options and requirement text are both present, split them with `--`:

```text
/cmind.feature_construct --review-threshold 99 -- Build a CLI for managing containers
```

If no options are present, treat the whole argument string as requirement text:

```text
/cmind.feature_construct Build a CLI for managing containers
```

Forward only the supported options to `cmind script feature_construct.py`; do not forward requirement text to the script helper.

If the user invoked `/cmind.feature_construct --check-only`, run:

```bash
cmind script feature_construct.py --check-only
```

If the user invoked `/cmind.feature_construct --check-only --json`, run:

```bash
cmind script feature_construct.py --check-only --json
```

In both check-only cases, show the script status output and stop without asking for requirements, inspecting `docs/` for generation, or running the pipeline.

## Outline

> [!WARNING]
> A full Phase 1 run can take from a few minutes to over an hour depending on project size. Do not interrupt it; if you must, re-run this command and it will resume from the first incomplete stage.

### Step 1: Probe progress

Run the orchestrator in probe mode and capture JSON:

```bash
cmind script feature_construct.py --check-only --json
```

Parse these fields:

- `total` — total stages, always 3
- `done` — count of stages whose `type` is `update`
- `next` — first incomplete stage, or `null` if all done
- `stages[*].name`, `stages[*].done`, `stages[*].will_run`, `stages[*].reason`

### Step 2: Determine requirement source

Requirement source is needed only when `feature_spec` is not complete or when the user chooses to overwrite/restart from the beginning.

Use this priority order:

1. **Requirement text after the command** — use it directly.
2. **Usable `docs/*.md` files** — if no requirement text was provided and Markdown files exist under `docs/`, use them automatically as the source. Do not ask the old `/cmind.feature_spec` confirmation prompt.
3. **Inline requirement prompt** — if there is neither requirement text nor usable `docs/*.md`, pause in this same command flow and ask the user to provide requirements. After the user supplies them, continue; do not ask them to rerun the slash command.

If the user supplied new requirement text while any Phase 1 artifact already exists and `--force` was not supplied, ask one real overwrite decision before changing artifacts:

```text
Phase 1 artifacts already exist, and new requirements were provided.

What would you like to do?
  [R] Restart  — regenerate Phase 1 from the new requirements
  [E] Exit     — keep existing artifacts unchanged
```

`R` continues with `--force`; `E` terminates the command.

### Step 3: Create or refresh feature specification artifacts when needed

If `feature_spec` is incomplete, or if the user chose restart/overwrite from Step 2, generate the feature specification artifacts using the existing `/cmind.feature_spec` workflow, but without its avoidable docs confirmation prompt:

- For direct requirement text, create `.cmind/data/feature_spec/evidence/user_input.md`, `.cmind/data/feature_spec/feature_spec.md`, and `.cmind/data/feature_spec/features/FT-*.md` following the rules in `/cmind.feature_spec`.
- For `docs/*.md`, process each document one by one into evidence, then generate the main spec and feature-domain files following the rules in `/cmind.feature_spec`.
- On restart, overwrite, or `--force` from the beginning, replace the selected input source's feature-spec working tree contents before conversion. Remove stale `.cmind/data/feature_spec/evidence/*.md` files and stale `.cmind/data/feature_spec/features/FT-*.md` files that are no longer part of the regenerated spec; do not leave markdown artifacts from a previous source in place.
- Preserve the existing quality requirements: English output, evidence line numbers, feature IDs, project type metadata, and JSON conversion readiness.
- Run `cmind script feature_spec_to_json.py` only after the markdown working tree reflects the selected source and stale generated markdown has been removed.

Do not prompt for review thresholds, review iteration counts, or refactor iteration counts in the default path. Use existing script defaults unless the user supplied explicit facade options.

### Step 4: Run the one-step Phase 1 helper

Choose exactly one execution mode:

**Case A — all three stages already complete (`done == total`)**

Display:

```text
All 3 Phase 1 stages are already complete:
  ✓ feature_spec
  ✓ feature_build
  ✓ feature_refactor

What would you like to do?
  [X] Expand features — suggest expansion directions, expand selected ones, then rerun refactor
  [O] Overwrite       — regenerate Phase 1 from scratch
  [E] Exit            — keep existing artifacts and proceed to /cmind.plan
```

- `X` → go to Step 6.
- `O` → ensure a requirement source exists using Step 2, then run `cmind script feature_construct.py --force <options>`.
- `E` → terminate after showing the completion guidance in Step 7.

**Case B — fresh or incomplete workspace (`done < total`)**

If `done == 0`, do not prompt. Briefly inform the user:

```text
Starting Phase 1 feature construction (3 stages). This may take a while.
```

If `0 < done < total`, do not ask for stage parameters. Continue automatically from `next` unless there is a real overwrite decision from Step 2.

Run:

```bash
cmind script feature_construct.py <options>
```

If restart was chosen or `--force` was supplied, run:

```bash
cmind script feature_construct.py --force <options>
```

If `--dry-run` was supplied, stream the dry-run command list and stop without modifying artifacts.

### Step 5: Stream output and handle failures

Stream stdout/stderr from `feature_construct.py` as-is. The helper prints one progress line per stage and validates each generated artifact after the stage runs.

If the helper exits non-zero, surface its recovery hints verbatim. Typical recovery commands:

```bash
cmind script feature_construct.py --check-only
cmind script feature_construct.py
cmind script feature_build.py --verbose
cmind script feature_refactor.py --log-level DEBUG
```

### Step 6: Optional feature expansion

After normal completion, and also from the already-complete case, offer optional expansion:

```text
Feature construction is complete.

Would you like to expand the feature tree beyond the current specification?
  [Y] Yes — suggest expansion directions
  [N] No  — finish here
```

If `Y`:

1. Run direction suggestion:

   ```bash
   cmind script feature_build.py --mode suggest-directions
   ```

   Include `--verbose` or `--no-trajectory` only if those facade options were supplied.

2. Parse the JSON output and show directions as a numbered Markdown table.

3. Ask for comma-separated direction numbers or `N` to finish. Normalize numeric input before passing it to the script.

4. Run directed expansion:

   ```bash
   cmind script feature_build.py --mode step2 --direction "<normalized indices>"
   ```

   Forward `--review-max-iterations`, `--verbose`, and `--no-trajectory` if supplied. Do not forward `--review-threshold` to `step2`; that mode uses the existing lightweight review flow.

5. Immediately rerun refactor so `.cmind/data/feature_tree.json` reflects the expanded `.cmind/data/feature_build.json`:

   ```bash
   cmind script feature_refactor.py
   ```

   Forward `--max-iterations <N>` when the facade option was `--max-iter-refactor N`, `--log-level DEBUG` when `--verbose` was supplied, and `--no-trajectory` when supplied.

6. Ask whether the user wants another expansion round. If yes, repeat Step 6 from direction suggestion; if no, continue to Step 7.

### Step 7: Completion message

On success or when the all-complete case exits without changes, tell the user:

```text
Feature construct complete.

Default next step:
  /cmind.plan                     — run the full Phase 2 RPG planning pipeline

Optional refinements:
  Expand features                  — rerun this command and choose expansion, or answer Y when prompted
  /cmind.feature_edit <instructions> — adjust the final feature tree if it is unsatisfactory

Granular/debug commands remain available:
  /cmind.feature_spec, /cmind.feature_build, /cmind.feature_refactor

Phase 2 granular fallback:
  /cmind.build_skeleton           — debug/surgical fallback only; /cmind.plan is the default next step
```
