# /rpgkit Commands Reference

RPG-Kit provides 13 slash commands that work in three paths:

- **Forward pipeline:** Requirements → Repository Planning Graph (RPG) → Code
- **Reverse encoder:** Existing code → RPG
- **Surgical edit:** Natural-language changes applied to code, RPG, and dependency graph together

> **Note on data paths.** Throughout this document, paths shown as `.rpgkit/data/...` and `.rpgkit/logs/...` are stable logical names. The actual files live **outside the workspace** under `~/.rpgkit/workspaces/<workspace-id>/{data,logs}/`, where `<workspace-id>` is a slug-based identifier and may include an overflow `-<hash6>` suffix, so that runtime artefacts never enter the user's git repository. Reports (`rpg.html`, review HTML, etc.) stay in the workspace at `<workspace>/.rpgkit/reports/` because they are small user-facing artefacts users may want to commit. Run `rpgkit version` from inside the workspace to see the resolved Data / Logs paths. See [project-structure.md](project-structure.md) for the full layout.

## Command Overview

### Phase 1: Feature Specification

| Command | Description |
| ------- | ----------- |
| `/rpgkit.feature_spec <desc>` | Create structured feature specifications from user input or `docs/` files |
| `/rpgkit.feature_build` | Generate and expand the feature tree from specifications |
| `/rpgkit.feature_refactor` | Refactor feature tree into modular component architecture |
| `/rpgkit.feature_edit <instr>` | Edit feature tree nodes before skeleton planning — optional |

### Phase 2: RPG Construction and Planning

| Command | Description |
| ------- | ----------- |
| `/rpgkit.build_skeleton` | Build repository file skeleton from component architecture; creates `.rpgkit/data/rpg.json` |
| `/rpgkit.build_data_flow` | Build inter-component data flow DAG and update the RPG |
| `/rpgkit.design_base_classes` | Design shared base classes and data structures |
| `/rpgkit.design_interfaces` | Design function/class interfaces with type hints and docstrings |
| `/rpgkit.plan_tasks` | Plan dependency-ordered implementation task batches |

### Phase 3: Code Generation and Surgical Edits

| Command | Description |
| ------- | ----------- |
| `/rpgkit.code_gen` | TDD-based implementation with iterative test-code-fix cycles |
| `/rpgkit.rpg_edit <instr>` | Surgical edit of RPG graph, code, and dependency graph from a natural-language instruction — optional |

### RPG Encoder: Code to RPG

| Command | Description |
| ------- | ----------- |
| `/rpgkit.encode` | Encode an existing repository into `.rpgkit/data/rpg.json` |
| `/rpgkit.update_rpg` | Manually run incremental RPG update when the automatic hook is skipped or fails |

Both directions produce the same RPG structure at `.rpgkit/data/rpg.json`, enabling AI agents to query the graph via the **MCP server** (`search_rpg`, `explore_rpg`, `get_node_detail`, `list_rpg_tree`). See [configuration.md](configuration.md) for MCP details.

---

## Phase 1: Feature Specification

### `/rpgkit.feature_spec`

Create structured feature specifications from user input or documentation files.

**Input modes:**

- **Direct input:** provide a description after the command
- **Auto-detect:** omit input to auto-detect `docs/*.md` files

**Output:**

```text
.rpgkit/data/feature_spec/
├── evidence/                # Source evidence files
│   ├── user_input.md        # From direct user input, or
│   ├── 01_project_charter.md
│   └── ...
├── feature_spec.md          # Meta + Background + NFR
└── features/                # Feature tree documents
    ├── FT-001.md
    ├── FT-002.md
    └── ...
```

Also generates `.rpgkit/data/feature_spec.json`.

**Examples:**

```text
/rpgkit.feature_spec Build a CLI tool for managing Docker containers
/rpgkit.feature_spec                  # Auto-detect docs/ files
```

---

### `/rpgkit.feature_build`

Generate and iteratively refine the feature tree from `.rpgkit/data/feature_spec.json`.

**Input:** `.rpgkit/data/feature_spec.json`

**Output:** `.rpgkit/data/feature_build.json`

**Current workflow:**

1. **Validate status** — runs `rpgkit script feature_build_validation.py` to verify that `feature_spec.json` exists and decide whether this is a first build or an expansion.
2. **Build or expand** — runs `rpgkit script feature_build.py --mode step1`.
   - If `feature_build.json` does not exist, RPG-Kit builds the feature tree from the specification and iterates until requirements are covered.
   - If `feature_build.json` already exists, RPG-Kit switches to beyond-spec expansion mode and adds production-relevant features not described by the original spec.
3. **Review** — validates coverage, duplicates, and MIU constraints. Coverage review uses a default threshold of `98.0` and up to `3` review iterations.
4. **Optional user-guided expansion** — the agent can ask whether to suggest additional expansion directions, then run `--mode suggest-directions` and `--mode step2 --direction <indices>`.

The spec-driven expansion loop has a hard safety cap of 20 iterations; the model self-terminates when it determines the spec is covered.

**Examples:**

```text
/rpgkit.feature_build
```

---

### `/rpgkit.feature_refactor`

Refactor the feature tree into a modular component architecture.

**Input:** `.rpgkit/data/feature_build.json`

**Output:** `.rpgkit/data/feature_tree.json`

**Process:**

1. **Plan** — analyze domains and plan subtree structure.
2. **Assign** — iteratively assign features to planned subtrees. The default assignment budget is 10 iterations and stops early when assignment reaches at least 99%.

**Example:**

```text
/rpgkit.feature_refactor
```

---

### `/rpgkit.feature_edit`

Edit feature tree nodes before repository planning begins.

**Input/Output:** `.rpgkit/data/feature_tree.json`

**Supported edits:** add, delete, modify, expand, move, or merge feature tree nodes.

**Process:**

1. **Plan** — generate an edit plan from the user's instruction.
2. **Execute** — apply the planned changes.
3. **Review** — verify and auto-fix if needed, up to 3 rounds.

**Examples:**

```text
/rpgkit.feature_edit Delete the 'cloud integration' component
/rpgkit.feature_edit Add logging features under 'cli operations'
/rpgkit.feature_edit Expand the 'security' component with encryption options
/rpgkit.feature_edit Merge 'analytics telemetry' into 'monitoring observability'
```

---

## Phase 2: RPG Construction and Planning

### `/rpgkit.build_skeleton`

Build the repository file skeleton from the component architecture. This is where the forward pipeline first creates the RPG.

**Input:** `.rpgkit/data/feature_tree.json`

**Output:**

- `.rpgkit/data/skeleton.json` — file skeleton
- `.rpgkit/data/skeleton_summary.txt` — human-readable skeleton summary
- `.rpgkit/data/rpg.json` — initial Repository Planning Graph with file and feature nodes

**Process:**

1. **Directory design** — design directory structure for each component.
2. **File assignment** — assign features to source files. The default assignment budget is 10 iterations.

**Examples:**

```text
/rpgkit.build_skeleton
/rpgkit.build_skeleton Prefer flat directory structure
```

---

### `/rpgkit.build_data_flow`

Build inter-component data flow as a directed acyclic graph (DAG).

**Input:** `.rpgkit/data/skeleton.json`, `.rpgkit/data/feature_tree.json`

**Output:**

- `.rpgkit/data/data_flow.json` — data flow DAG
- `.rpgkit/data/data_flow_viz.html` — interactive visualization
- Updates `.rpgkit/data/rpg.json` — adds data-flow edges

**Process:**

1. **Pre-check** — verifies whether data flow is missing, valid, or mismatched with the skeleton.
2. **Iteration choice** — asks for max iterations:
   - `Y` uses the default of 5 iterations.
   - A number sets a custom iteration budget.
3. **DAG design** — runs `rpgkit script build_data_flow.py --max-iterations <N>`.
4. **Validation** — runs `rpgkit script check_data_flow.py --verbose`.
5. **Visualization** — runs `rpgkit script generate_viz.py` when a new data flow is built.

**Example:**

```text
/rpgkit.build_data_flow
/rpgkit.build_data_flow Make the ingestion layer independent from reporting
```

---

### `/rpgkit.design_base_classes`

Design shared base classes and global data structures to improve modularity and reuse.

**Input:** `.rpgkit/data/skeleton.json`, `.rpgkit/data/data_flow.json`

**Output:**

- `.rpgkit/data/base_classes.json` — base class and global data structure definitions
- Updates `.rpgkit/data/rpg.json` — adds base-class relationship edges

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
/rpgkit.design_base_classes
```

---

### `/rpgkit.design_interfaces`

Design function and class interfaces with type hints and docstrings for all planned repository files.

**Input:** `.rpgkit/data/skeleton.json`, `.rpgkit/data/data_flow.json`, `.rpgkit/data/base_classes.json`

**Output:**

- `.rpgkit/data/interfaces.json` — function/class interface definitions
- Updates `.rpgkit/data/rpg.json` — adds fine-grained dependency edges such as inheritance, invocation, and references

**Process:**

1. Read skeleton, data flow, and base classes for context.
2. Process components in dependency order from the data flow DAG.
3. Design functions and classes with type-hinted signatures.
4. Map each unit to the features it implements.

**Example:**

```text
/rpgkit.design_interfaces
```

---

### `/rpgkit.plan_tasks`

Plan implementation tasks from interface definitions, organized into dependency-ordered batches.

**Input:** `.rpgkit/data/interfaces.json`, `.rpgkit/data/data_flow.json`, `.rpgkit/data/rpg.json`

**Output:** `.rpgkit/data/tasks.json`

**Process:**

1. Analyze dependencies between units using the RPG.
2. Sort units topologically.
3. Group units into implementation batches.
4. Add auxiliary file tasks such as `requirements.txt`, `main.py`, `README.md`, and `.gitignore`.

**Example:**

```text
/rpgkit.plan_tasks
```

---

## Phase 3: Code Generation and Surgical Edits

### `/rpgkit.code_gen`

Execute TDD-based code implementation with iterative test-code-fix cycles.

**Input:** `.rpgkit/data/tasks.json`, `.rpgkit/data/interfaces.json`, `.rpgkit/data/base_classes.json`, `.rpgkit/data/data_flow.json`, `.rpgkit/data/rpg.json`

**Output:** complete tested source code, `.rpgkit/data/code_gen_state.jsonl`, and updated `.rpgkit/data/rpg.json`

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
/rpgkit.code_gen
```

---

### `/rpgkit.rpg_edit`

Apply a natural-language edit to code, RPG, and dependency graph in sync.

This command is independent from `/rpgkit.feature_edit` and `/rpgkit.update_rpg`. It does not edit `feature_tree.json`; it uses the current RPG feature graph as the authoritative entry point for code modifications.

**Input:** edit instruction after the command

**Input files:** `.rpgkit/data/rpg.json`, `.rpgkit/data/dep_graph.json`

**Generated files:**

- `.rpgkit/data/rpg_edit_impact.json` — impact analysis output
- `.rpgkit/data/rpg_edit_plan.json` — user-confirmed edit plan
- `.rpgkit/data/rpg_edit_code_result.json` — code application result

**Workflow:**

1. **Pre-check** — runs `rpgkit script rpg_edit/validate.py --json` and stops if the RPG or dependency graph is unavailable.
2. **Locate target nodes** — runs `rpgkit script rpg_edit/locate.py --query "<instruction>" --json` and selects existing nodes or nearest parent nodes for new features.
3. **Analyze impact** — runs `rpgkit script rpg_edit/impact.py --node-id ... --json` to identify affected nodes, callers, callees, and files.
4. **Optional visual reconnaissance** — for UI/layout/style edits, probes the app with the browser helper when available.
5. **Mandatory code reconnaissance** — reads affected files and searches related patterns before producing a plan.
6. **Generate and confirm plan** — writes `.rpgkit/data/rpg_edit_plan.json` and asks the user to apply, cancel, revise, or inspect a node.
7. **Apply on a branch** — creates a `rpg-edit/<short-id>` branch only after a clean working-tree preflight.
8. **RPG-first apply** — updates RPG feature changes first, then dispatches code changes, refreshes `dep_graph.json`, and folds graph updates into the branch commit.
9. **Test and review** — runs smoke tests and impact review.
10. **Merge or preserve** — merges into `main` only after tests pass; failed runs leave the branch for inspection.

**Examples:**

```text
/rpgkit.rpg_edit Add a last_login field to the User model and update it on login
/rpgkit.rpg_edit Add rate limiting to all API endpoints
/rpgkit.rpg_edit Refactor auth into separate registration and login modules
```

---

## RPG Encoder: Code to RPG

The encoder works in the reverse direction from the forward pipeline. It takes an existing codebase and produces the same Repository Planning Graph structure used by RPG-Kit's planning, editing, and MCP tooling.

### `/rpgkit.encode`

Encode the current repository into an RPG from scratch.

**Output:**

- `.rpgkit/data/rpg.json` — Repository Planning Graph
- `.rpgkit/data/dep_graph.json` — code dependency graph used for incremental sync and edits

**Process:**

1. **Pre-check** — runs `rpgkit script rpg_encoder/check_encode.py --json`.
2. **Full encode** — runs `rpgkit script rpg_encoder/run_encode.py --json`.
3. **Next steps** — suggests `/rpgkit.update_rpg` for incremental updates and MCP tools for exploration.

If `rpg.json` already exists, the command asks whether to full re-encode, switch to `/rpgkit.update_rpg`, or quit.

**Example:**

```text
/rpgkit.encode
```

---

### `/rpgkit.update_rpg`

Manually trigger an incremental RPG update when the automatic hook did not run or when the user wants an immediate foreground update.

Under normal use, RPG-Kit installs a post-commit hook that updates the RPG in the background after each commit. This command is the manual fallback.

**Input:** existing `.rpgkit/data/rpg.json` and a git repository with at least two commits

**Output:** updated `.rpgkit/data/rpg.json` and `.rpgkit/data/dep_graph.json`

**Process:**

1. **Pre-check** — runs `rpgkit script rpg_encoder/check_encode.py --json` and stops if `rpg.json` is missing or corrupt.
2. **Commit baseline check** — verifies `HEAD~1` exists. If there is no previous commit, run `/rpgkit.encode` instead.
3. **Incremental update** — runs `rpgkit script update_graphs.py update-rpg --json`, comparing the current workspace against `HEAD~1`, the same baseline used by the hook.
4. **Report result** — displays node/edge deltas, functional areas, alignment status, and output path.

Use this command when:

- The post-commit hook failed or was skipped.
- `.rpgkit/logs/update_rpg.log` shows an error.
- The RPG seems stale and you want to force a synchronous update.

**Example:**

```text
/rpgkit.update_rpg
```

---

## MCP Server Tools

RPG-Kit registers an MCP server named `rpg-tools` so AI agents can query `.rpgkit/data/rpg.json` during chat. The server exposes four read-only tools:

| Tool | Description |
| ---- | ----------- |
| `search_rpg` | Search code entities or features by keyword, path, class, function, or feature name |
| `explore_rpg` | Traverse dependencies and call chains from a starting node |
| `get_node_detail` | Fetch full details for a function, class, file, or feature node |
| `list_rpg_tree` | Render the functional architecture as a tree |

If `.rpgkit/data/rpg.json` is not available yet, the tools return an `rpg_unavailable` response that asks the agent to run `/rpgkit.encode`.

See [configuration.md](configuration.md) for MCP registration, auto-approval, hooks, and initialization options.

---

## Data Files

All intermediate data is stored in `.rpgkit/data/`:

| File | Produced by | Description |
| ---- | ----------- | ----------- |
| `feature_spec/` | `feature_spec` | Evidence and feature specification documents |
| `feature_spec.json` | `feature_spec` | Structured feature specification |
| `feature_build.json` | `feature_build` | Expanded feature tree |
| `feature_tree.json` | `feature_refactor` / `feature_edit` | Component architecture |
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

1. **Forward:** `/rpgkit.build_skeleton` creates it from `feature_tree.json`; later planning and generation commands enrich it.
2. **Reverse:** `/rpgkit.encode` creates it from an existing codebase; `/rpgkit.update_rpg` keeps it aligned after commits.

Subsequent commands update the same file:

1. **`build_data_flow`** — adds data-flow edges.
2. **`design_base_classes`** — adds base-class relationship edges.
3. **`design_interfaces`** — adds fine-grained dependency edges.
4. **`code_gen`** — updates implementation status as code is generated.
5. **`rpg_edit`** — applies targeted feature graph edits together with code and dependency graph changes.
