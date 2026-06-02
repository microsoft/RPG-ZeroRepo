---
description: Extract a structured feature specification from requirement docs or inline text
name: cmind.feature_spec
---

## User Input

```text
$ARGUMENTS
```

This command is a thin wrapper around `cmind script feature_spec.py`.
For the recommended end-to-end Phase 1 flow (spec → build → refactor),
use `/cmind.feature_construct` instead. The granular command remains
available for debugging and single-stage reruns.

## Argument Parsing

Supported facade options (forward as-is to the helper):

- `--check-only` — probe output state without invoking the LLM.
- `--json` — with `--check-only`, emit JSON.
- `--force` — overwrite an existing valid `feature_spec.json`.
- `--verbose` — DEBUG logging.
- `--no-trajectory` — skip trajectory recording.

If options and requirement text are both present, split them with `--`:

```text
/cmind.feature_spec --force -- Build a CLI for managing Docker containers
```

If no options are present, treat the whole argument string as requirement
text.

## Workflow

### Step 1: Check-only short-circuits

If `--check-only` was supplied, run:

```bash
cmind script feature_spec.py --check-only [--json]
```

Show the output verbatim and stop without inspecting `docs/` or
generating anything.

### Step 2: Determine requirement source

Priority:

1. **Requirement text** after the command (and after `--` if options are
   present). Pass via `--input-text "<text>"`.
2. **`docs/*.md` files**. If text is empty and `docs/` contains usable
   Markdown files, no flag is needed — `feature_spec.py` auto-detects.
3. **Inline prompt**. If neither exists, ask the user for requirements in
   this conversation, then pass the captured text via `--input-text`.

Do **not** display the legacy "Use these documents? (Y/N)" confirmation —
auto-detection already covers it.

### Step 3: Overwrite decision

If a **valid** `feature_spec.json` already exists and `--force` was not
supplied, the helper exits with `[SKIP]` without calling the LLM.
(Missing or schema-invalid existing files do not trigger `[SKIP]`;
the helper regenerates them.) To regenerate a valid spec, ask the user:

```text
feature_spec.json already exists. Regenerate?
  [F] Force regenerate    [E] Keep existing
```

`F` → rerun the command with `--force`; `E` → stop.

### Step 4: Run

```bash
cmind script feature_spec.py [--input-text "<text>"] [--force] [--no-trajectory] [--verbose]
```

Stream stdout / stderr from the helper. The helper handles:

- Reading docs / inline text.
- Calling the LLM with a strict Pydantic schema (`FeatureSpecOutput`).
- Validating the output against the schema (no markdown intermediaries).
- Writing `feature_spec.json` atomically (on failure, no partial file).

### Step 5: Recovery hints on failure

If the helper exits non-zero:

- Exit code `2` (`NoInputAvailable`) — neither `--input-text` nor
  `docs/*.md` was found. Re-invoke with either inline text or after
  populating `docs/`.
- Exit code `1` (LLM failure / schema validation failure) — re-run with
  `--verbose` to surface the trajectory file location in the script's
  own log output; do not attempt to locate it manually.

  ```bash
  cmind script feature_spec.py --verbose
  ```

### Step 6: Completion

On success, the helper prints summary stats (repository, top-level
feature count, BG / NFR counts) and writes the spec to the home-side
data store.

For the standard end-to-end Phase 1 flow, use
`/cmind.feature_construct` — it runs this stage plus the remaining
Phase 1 steps in one command.
