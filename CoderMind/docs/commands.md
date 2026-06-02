# /cmind Commands Reference

CoderMind provides 15 slash commands that work in three paths:

- **Forward pipeline:** Requirements → Repository Planning Graph (RPG) → Code
- **Reverse encoder:** Existing code → RPG
- **Surgical edit:** Natural-language changes applied to code, RPG, and dependency graph together

> **Note on data paths.** Throughout this document, paths shown as `.cmind/data/...` and `.cmind/logs/...` are stable logical names. The actual files live **outside the workspace** under `~/.cmind/workspaces/<workspace-id>/{data,logs}/`, where `<workspace-id>` is a slug-based identifier and may include an overflow `-<hash6>` suffix, so that runtime artefacts never enter the user's git repository. Reports (`rpg.html`, review HTML, etc.) stay in the workspace at `<workspace>/.cmind/reports/` because they are small user-facing artefacts users may want to commit. Run `cmind version` from inside the workspace to see the resolved Data / Logs paths. See [project-structure.md](project-structure.md) for the full layout.

## Command Overview

### Phase 1: Feature Specification

| Command | Description |
| ------- | ----------- |
| `/cmind.feature_construct <desc>` | Run Phase 1 feature specification, build, and refactor in one step — recommended |
| `/cmind.feature_spec <desc>` | Create structured feature specifications from user input or `docs/` files |
| `/cmind.feature_build` | Generate and expand the feature tree from specifications |
| `/cmind.feature_refactor` | Refactor feature tree into modular component architecture |
| `/cmind.feature_edit <instr>` | Edit feature tree nodes before skeleton planning — optional |

> `/cmind.feature_construct` is the simplest way to drive Phase 1 end-to-end. Use
> the individual Phase 1 commands only when you want to debug or re-run a
> specific stage.

### Phase 2: RPG Construction and Planning

| Command | Description |
| ------- | ----------- |
| `/cmind.plan` | Run all five Phase-2 stages in one step with automatic resume — recommended |
| `/cmind.build_skeleton` | Build repository file skeleton from component architecture; creates `.cmind/data/rpg.json` |
| `/cmind.build_data_flow` | Build inter-component data flow DAG and update the RPG |
| `/cmind.design_base_classes` | Design shared base classes and data structures |
| `/cmind.design_interfaces` | Design function/class interfaces with type hints and docstrings |
| `/cmind.plan_tasks` | Plan dependency-ordered implementation task batches |

> `/cmind.plan` is the simplest way to drive Phase 2 end-to-end. Use
> the individual commands above only when you want to debug or
> re-run a specific stage.

### Phase 3: Code Generation and Surgical Edits

| Command | Description |
| ------- | ----------- |
| `/cmind.code_gen` | TDD-based implementation with iterative test-code-fix cycles |
| `/cmind.rpg_edit <instr>` | Surgical edit of RPG graph, code, and dependency graph from a natural-language instruction — optional |

### RPG Encoder: Code to RPG

| Command | Description |
| ------- | ----------- |
| `/cmind.encode` | Encode an existing repository into `.cmind/data/rpg.json` |
| `/cmind.update_rpg` | Manually run incremental RPG update when the automatic hook is skipped or fails |

Both directions produce the same RPG structure at `.cmind/data/rpg.json`, enabling AI agents to query the graph via the **MCP server** (`search_rpg`, `explore_rpg`, `get_node_detail`, `list_rpg_tree`). See [configuration.md](configuration.md) for MCP details.

---

## Phase 1: Feature Specification

### `/cmind.feature_construct`

Run the full Phase 1 pipeline (`feature_spec` → `feature_build` → `feature_refactor`) in one step. This is the recommended entry point for creating the feature tree that feeds `/cmind.plan`.

**Input modes:**

- **Direct input:** provide requirements after the command.
- **Auto-detect:** omit input to use existing `docs/*.md` files automatically.
- **Inline prompt:** if neither direct input nor usable docs exist, the command asks for requirements and then continues in the same flow.

**Output:** the three Phase 1 JSON artefacts — `feature_spec.json`, `feature_build.json`, and `feature_tree.json` — in the CoderMind data store.

**Process:**

1. **Probe progress** — runs `cmind script feature_construct.py --check-only --json` to see which Phase 1 stages already have valid artifacts.
2. **Generate requirements artifacts when needed** — follows the `/cmind.feature_spec` workflow for direct text or `docs/*.md` sources.
3. **Run/resume** — executes `cmind script feature_construct.py`, skipping completed stages and cascading downstream rebuilds when an upstream stage reruns.
4. **Optional expansion** — after completion, the user can expand features through the existing `feature_build --mode suggest-directions` and `--mode step2 --direction <indices>` flow; refactor is rerun afterward so `feature_tree.json` stays aligned.

**CLI flags forwarded after `$ARGUMENTS`:**

- `--check-only` — show Phase 1 progress without modifying artifacts or running stages.
- `--json` — with `--check-only`, emit the progress report as JSON.
- `--force` — rebuild all Phase 1 stages.
- `--dry-run` — print the commands that would run without modifying artifacts.
- `--verbose` — forward native verbose logging flags.
- `--no-trajectory` — disable trajectory recording where supported.
- `--max-iter-refactor N` — forward to `feature_refactor.py` as `--max-iterations N`.
- `--review-threshold N` — forward to `feature_build.py --mode step1`.
- `--review-max-iterations N` — forward to `feature_build.py --mode step1`.

Use `--` to separate options from requirement text:

```text
/cmind.feature_construct --check-only
/cmind.feature_construct --check-only --json
/cmind.feature_construct --review-threshold 99 -- Build a CLI tool for managing Docker containers
/cmind.feature_construct Build a CLI tool for managing Docker containers
/cmind.feature_construct                  # Auto-detect docs/ files
```

**Next step:** `/cmind.plan` is the default handoff after Phase 1. If the final tree needs small adjustments, run `/cmind.feature_edit <instructions>`. The granular Phase 1 commands remain available for debug and surgical reruns; `/cmind.build_skeleton` is a Phase 2 granular fallback, not the default next step.

---

### `/cmind.feature_spec`

Generate a structured feature specification from user input or
documentation files.

> **Recommended workflow:** Use `/cmind.feature_construct` for the
> end-to-end Phase 1 flow. `/cmind.feature_spec` is the granular
> single-stage command, kept available for debugging and surgical reruns.

**Input modes:**

- **Direct input:** provide a description after the command
- **Auto-detect:** omit input to auto-detect `docs/*.md` files

**Output:** `.cmind/data/feature_spec.json` — a Pydantic-validated JSON
document containing `meta`, `repository_name`, `repository_purpose`,
`background_and_overview`, `non_functional_requirements`, and a
recursive `functional_requirements` tree.

**Examples:**

```text
/cmind.feature_spec Build a CLI tool for managing Docker containers
/cmind.feature_spec                  # Auto-detect docs/ files
```

---

### `/cmind.feature_build`

Generate and iteratively refine the feature tree from `.cmind/data/feature_spec.json`.

**Input:** `.cmind/data/feature_spec.json`

**Output:** `.cmind/data/feature_build.json`

**Current workflow:**

1. **Validate status** — runs `cmind script feature_build_validation.py` to verify that `feature_spec.json` exists and decide whether this is a first build or an expansion.
2. **Build or expand** — runs `cmind script feature_build.py --mode step1`.
   - If `feature_build.json` does not exist, CoderMind builds the feature tree from the specification and iterates until requirements are covered.
   - If `feature_build.json` already exists, CoderMind switches to beyond-spec expansion mode and adds production-relevant features not described by the original spec.
3. **Review** — validates coverage, duplicates, and MIU constraints. Coverage review uses a default threshold of `98.0` and up to `3` review iterations.
4. **Optional user-guided expansion** — the agent can ask whether to suggest additional expansion directions, then run `--mode suggest-directions` and `--mode step2 --direction <indices>`.

The spec-driven expansion loop has a hard safety cap of 20 iterations; the model self-terminates when it determines the spec is covered.

**Examples:**

```text
/cmind.feature_build
```

---

### `/cmind.feature_refactor`

Refactor the feature tree into a modular component architecture.

**Input:** `.cmind/data/feature_build.json`

**Output:** `.cmind/data/feature_tree.json`

**Process:**

1. **Plan** — analyze domains and plan subtree structure.
2. **Assign** — iteratively assign features to planned subtrees. The default assignment budget is 10 iterations and stops early when assignment reaches at least 99%.

**Example:**

```text
/cmind.feature_refactor
```

---

### `/cmind.feature_edit`

Edit feature tree nodes before repository planning begins.

**Input/Output:** `.cmind/data/feature_tree.json`

**Supported edits:** add, delete, modify, expand, move, or merge feature tree nodes.

**Process:**

1. **Plan** — generate an edit plan from the user's instruction.
2. **Execute** — apply the planned changes.
3. **Review** — verify and auto-fix if needed, up to 3 rounds.

**Examples:**

```text
/cmind.feature_edit Delete the 'cloud integration' component
/cmind.feature_edit Add logging features under 'cli operations'
/cmind.feature_edit Expand the 'security' component with encryption options
/cmind.feature_edit Merge 'analytics telemetry' into 'monitoring observability'
```

---

## Phase 2: RPG Construction and Planning

### `/cmind.plan`

Run the full Phase-2 pipeline (`build_skeleton` → `build_data_flow` →
`design_base_classes` → `design_interfaces` → `plan_tasks`) in one
step. This is the recommended entry point for Phase 2.

**Input:** `~/.cmind/workspaces/<workspace-id>/data/feature_tree.json` (produced by
`/cmind.feature_construct` or `/cmind.feature_refactor`)

**Output:** every artifact produced by the five individual commands —
`skeleton.json`, `data_flow.json`, `base_classes.json`,
`interfaces.json`, `tasks.json`, plus `rpg.json` and the
`data_flow_viz.html` visualization.

**Process:**

1. **Probe progress** — runs `cmind script plan.py --check-only --json`
   to see which stages already have valid artifacts.
2. **Decide** — based on the probe result, the command prompts you
   **once** with one of three options:
    - All five stages already done → `Overwrite` or `Exit`.
    - Partial progress (some stages done) → `Continue`, `Restart`, or `Exit`.
    - Fresh workspace → no prompt; runs the full pipeline.
3. **Run** — executes the chosen mode through `cmind script plan.py`.
   Each stage's stdout is streamed live and also written to a per-stage
   log under `~/.cmind/workspaces/<workspace-id>/logs/`.
4. **Verify** — after every stage's build script, the corresponding
   `check_*.py` script re-runs to validate the produced artifact. If
   verification fails the pipeline stops and prints recovery hints.

**Resume semantics:** if you press Ctrl-C halfway through, running
`/cmind.plan` again automatically resumes from the first incomplete
stage. When any earlier stage is re-run, every downstream stage is
rebuilt too so artifacts never drift apart.

**CLI flags forwarded after `$ARGUMENTS`:**

- `--force` — discard existing artifacts and rebuild every stage.
- `--max-iter-skeleton N`, `--max-iter-data-flow N`,
  `--max-iter-base-classes N`, `--max-iter-interfaces N` —
  override iteration counts for the corresponding stage.
- `--verbose` — forward `--verbose` to every sub-script.
- `--no-trajectory` — forward `--no-trajectory` where supported.

**Examples:**

```text
/cmind.plan
/cmind.plan --verbose
/cmind.plan --force                    # rebuild everything
/cmind.plan --max-iter-skeleton 15
```

To inspect progress without running anything:

```bash
cmind script plan.py --check-only
```

---

### `/cmind.build_skeleton`

Build the repository file skeleton from the component architecture. This is where the forward pipeline first creates the RPG.

**Input:** `.cmind/data/feature_tree.json`

**Output:**

- `.cmind/data/skeleton.json` — file skeleton
- `.cmind/data/skeleton_summary.txt` — human-readable skeleton summary
- `.cmind/data/rpg.json` — initial Repository Planning Graph with file and feature nodes

**Process:**

1. **Directory design** — design directory structure for each component.
2. **File assignment** — assign features to source files. The default assignment budget is 10 iterations.

**Examples:**

```text
/cmind.build_skeleton
/cmind.build_skeleton Prefer flat directory structure
```

---

### `/cmind.build_data_flow`

Build inter-component data flow as a directed acyclic graph (DAG).

**Input:** `.cmind/data/skeleton.json`, `.cmind/data/feature_tree.json`

**Output:**

- `.cmind/data/data_flow.json` — data flow DAG
- `.cmind/data/data_flow_viz.html` — interactive visualization
- Updates `.cmind/data/rpg.json` — adds data-flow edges

**Process:**

1. **Pre-check** — verifies whether data flow is missing, valid, or mismatched with the skeleton.
2. **Iteration choice** — asks for max iterations:
   - `Y` uses the default of 5 iterations.
   - A number sets a custom iteration budget.
3. **DAG design** — runs `cmind script build_data_flow.py --max-iterations <N>`.
4. **Validation** — runs `cmind script check_data_flow.py --verbose`.
5. **Visualization** — runs `cmind script generate_viz.py` when a new data flow is built.

**Example:**

```text
/cmind.build_data_flow
/cmind.build_data_flow Make the ingestion layer independent from reporting
```

---

### `/cmind.design_base_classes`

Design shared base classes and global data structures to improve modularity and reuse.

**Input:** `.cmind/data/skeleton.json`, `.cmind/data/data_flow.json`

**Output:**

- `.cmind/data/base_classes.json` — base class and global data structure definitions
- Updates `.cmind/data/rpg.json` — adds base-class relationship edges

**Process:**

1. **Functional base classes** — design behavioral abstractions.
2. **Global data structures** — design shared data formats.

**Options:**

| Input | Description |
| ----- | ----------- |
| `Y` | Use defaults, 5 iterations |
| Number | Set a custom iteration count |

**Example:**

```text
/cmind.design_base_classes
```

---

### `/cmind.design_interfaces`

Design function and class interfaces with type hints and docstrings for all planned repository files.

**Input:** `.cmind/data/skeleton.json`, `.cmind/data/data_flow.json`, `.cmind/data/base_classes.json`

**Output:**

- `.cmind/data/interfaces.json` — function/class interface definitions
- Updates `.cmind/data/rpg.json` — adds fine-grained dependency edges such as inheritance, invocation, and references

**Process:**

1. Read skeleton, data flow, and base classes for context.
2. Process components in dependency order from the data flow DAG.
3. Design functions and classes with type-hinted signatures.
4. Map each unit to the features it implements.

**Example:**

```text
/cmind.design_interfaces
```

---

### `/cmind.plan_tasks`

Plan implementation tasks from interface definitions, organized into dependency-ordered batches.

**Input:** `.cmind/data/interfaces.json`, `.cmind/data/data_flow.json`, `.cmind/data/rpg.json`

**Output:** `.cmind/data/tasks.json`

**Process:**

1. Analyze dependencies between units using the RPG.
2. Sort units topologically.
3. Group units into implementation batches.
4. Add auxiliary file tasks such as `requirements.txt`, `main.py`, `README.md`, and `.gitignore`.

**Example:**

```text
/cmind.plan_tasks
```

---

## Phase 3: Code Generation and Surgical Edits

### `/cmind.code_gen`

Execute TDD-based code implementation with iterative test-code-fix cycles.

**Input:** `.cmind/data/tasks.json`, `.cmind/data/interfaces.json`, `.cmind/data/base_classes.json`, `.cmind/data/data_flow.json`, `.cmind/data/rpg.json`

**Output:** complete tested source code, `.cmind/data/code_gen_state.jsonl`, and updated `.cmind/data/rpg.json`

**Batch modes:**

| Mode | Description |
| ---- | ----------- |
| `S` | Single-batch mode: one batch at a time |
| `F` | File-merge mode: merge batches per file, optionally limited by max units |

**TDD cycle:**

1. Initialize the codebase if needed.
2. Create a branch from `main` for the next batch.
3. Dispatch a sub-agent to write tests, implement code, run pytest, and fix failures.
4. Independently verify the batch.
5. Merge successful batches into `main`; preserve failed branches for inspection.
6. Continue autonomously until all tasks are processed.
7. Run final test and global review.

**Auxiliary files:**

| File | Test method |
| ---- | ----------- |
| `requirements.txt` | Import validation in an isolated virtual environment |
| `main.py` | Execution test, usually `--help` |
| `README.md` | No direct test |
| `.gitignore` | No direct test |

**Example:**

```text
/cmind.code_gen
```

---

### `/cmind.rpg_edit`

Apply a natural-language edit to code, RPG, and dependency graph in sync.

This command is independent from `/cmind.feature_edit` and `/cmind.update_rpg`. It does not edit `feature_tree.json`; it uses the current RPG feature graph as the authoritative entry point for code modifications.

**Input:** edit instruction after the command

**Input files:** `.cmind/data/rpg.json`, `.cmind/data/dep_graph.json`

**Generated files:**

- `.cmind/data/rpg_edit_impact.json` — impact analysis output
- `.cmind/data/rpg_edit_plan.json` — user-confirmed edit plan
- `.cmind/data/rpg_edit_code_result.json` — code application result

**Workflow:**

1. **Pre-check** — runs `cmind script rpg_edit/validate.py --json` and stops if the RPG or dependency graph is unavailable.
2. **Locate target nodes** — runs `cmind script rpg_edit/locate.py --query "<instruction>" --json` and selects existing nodes or nearest parent nodes for new features.
3. **Analyze impact** — runs `cmind script rpg_edit/impact.py --node-id ... --json` to identify affected nodes, callers, callees, and files.
4. **Optional visual reconnaissance** — for UI/layout/style edits, probes the app with the browser helper when available.
5. **Mandatory code reconnaissance** — reads affected files and searches related patterns before producing a plan.
6. **Generate and confirm plan** — writes `.cmind/data/rpg_edit_plan.json` and asks the user to apply, cancel, revise, or inspect a node.
7. **Apply on a branch** — creates a `rpg-edit/<short-id>` branch only after a clean working-tree preflight.
8. **RPG-first apply** — updates RPG feature changes first, then dispatches code changes, refreshes `dep_graph.json`, and folds graph updates into the branch commit.
9. **Test and review** — runs smoke tests and impact review.
10. **Merge or preserve** — merges into `main` only after tests pass; failed runs leave the branch for inspection.

**Examples:**

```text
/cmind.rpg_edit Add a last_login field to the User model and update it on login
/cmind.rpg_edit Add rate limiting to all API endpoints
/cmind.rpg_edit Refactor auth into separate registration and login modules
```

---

## RPG Encoder: Code to RPG

The encoder works in the reverse direction from the forward pipeline. It takes an existing codebase and produces the same Repository Planning Graph structure used by CoderMind's planning, editing, and MCP tooling.

### `/cmind.encode`

Encode the current repository into an RPG from scratch.

**Output:**

- `.cmind/data/rpg.json` — Repository Planning Graph
- `.cmind/data/dep_graph.json` — code dependency graph used for incremental sync and edits

**Process:**

1. **Pre-check** — runs `cmind script rpg_encoder/check_encode.py --json`.
2. **Full encode** — runs `cmind script rpg_encoder/run_encode.py --json`.
3. **Next steps** — suggests `/cmind.update_rpg` for incremental updates and MCP tools for exploration.

If `rpg.json` already exists, the command asks whether to full re-encode, switch to `/cmind.update_rpg`, or quit.

**Example:**

```text
/cmind.encode
```

---

### `/cmind.update_rpg`

Manually trigger an incremental RPG update when the automatic hook did not run or when the user wants an immediate foreground update.

Under normal use, CoderMind installs a post-commit hook that updates the RPG in the background after each commit. This command is the manual fallback.

**Input:** existing `.cmind/data/rpg.json` and a git repository with at least two commits

**Output:** updated `.cmind/data/rpg.json` and `.cmind/data/dep_graph.json`

**Process:**

1. **Pre-check** — runs `cmind script rpg_encoder/check_encode.py --json` and stops if `rpg.json` is missing or corrupt.
2. **Commit baseline check** — verifies `HEAD~1` exists. If there is no previous commit, run `/cmind.encode` instead.
3. **Incremental update** — runs `cmind script update_graphs.py update-rpg --json`, comparing the current workspace against `HEAD~1`, the same baseline used by the hook.
4. **Report result** — displays node/edge deltas, functional areas, alignment status, and output path.

Use this command when:

- The post-commit hook failed or was skipped.
- `.cmind/logs/update_rpg.log` shows an error.
- The RPG seems stale and you want to force a synchronous update.

**Example:**

```text
/cmind.update_rpg
```

---

## MCP Server Tools

CoderMind registers an MCP server named `rpg-tools` so AI agents can query `.cmind/data/rpg.json` during chat. The server exposes four read-only tools:

| Tool | Description |
| ---- | ----------- |
| `search_rpg` | Search code entities or features by keyword, path, class, function, or feature name |
| `explore_rpg` | Traverse dependencies and call chains from a starting node |
| `get_node_detail` | Fetch full details for a function, class, file, or feature node |
| `list_rpg_tree` | Render the functional architecture as a tree |

If `.cmind/data/rpg.json` is not available yet, the tools return an `rpg_unavailable` response that asks the agent to run `/cmind.encode`.

See [configuration.md](configuration.md) for MCP registration, auto-approval, hooks, and initialization options.

---

## Data Files

All intermediate data is stored in `.cmind/data/` (a logical name; the actual files live under `~/.cmind/workspaces/<workspace-id>/data/` — see the note at the top of this document).

| File | Produced by | Description |
| ---- | ----------- | ----------- |
| `feature_spec.json` | `feature_construct` / `feature_spec` | Structured feature specification |
| `feature_build.json` | `feature_construct` / `feature_build` | Expanded feature tree |
| `feature_tree.json` | `feature_construct` / `feature_refactor` / `feature_edit` | Component architecture |
| `skeleton.json` | `build_skeleton` | File skeleton |
| `skeleton_summary.txt` | `build_skeleton` | Human-readable skeleton summary |
| `rpg.json` | `build_skeleton` / `encode`, then updated by later commands | Repository Planning Graph |
| `dep_graph.json` | `encode` / `update_rpg` / `rpg_edit` | Code dependency graph used for incremental sync and edits |
| `data_flow.json` | `build_data_flow` | Inter-component data flow DAG |
| `data_flow_viz.html` | `build_data_flow` | Data flow visualization |
| `base_classes.json` | `design_base_classes` | Shared base class definitions |
| `interfaces.json` | `design_interfaces` | Function/class interface definitions |
| `tasks.json` | `plan_tasks` | Dependency-ordered implementation batches |
| `code_gen_state.jsonl` | `code_gen` | Code generation progress state, append-only JSONL |
| `rpg_edit_impact.json` | `rpg_edit` | Impact analysis for a surgical edit |
| `rpg_edit_plan.json` | `rpg_edit` | Confirmed surgical edit plan |
| `rpg_edit_code_result.json` | `rpg_edit` | Code application result for a surgical edit |
| `trajectory/` | All scripts | Execution trajectory logs |

### `rpg.json` — The Repository Planning Graph

`rpg.json` is the central artifact that ties the pipeline together. It can be created in either direction:

1. **Forward:** `/cmind.build_skeleton` creates it from `feature_tree.json`; later planning and generation commands enrich it.
2. **Reverse:** `/cmind.encode` creates it from an existing codebase; `/cmind.update_rpg` keeps it aligned after commits.

Subsequent commands update the same file:

1. **`build_data_flow`** — adds data-flow edges.
2. **`design_base_classes`** — adds base-class relationship edges.
3. **`design_interfaces`** — adds fine-grained dependency edges.
4. **`code_gen`** — updates implementation status as code is generated.
5. **`rpg_edit`** — applies targeted feature graph edits together with code and dependency graph changes.
