# Project Structure

## Workspace == Repo

RPG-Kit installs alongside your project code: the directory you run `rpgkit init` in, also called the workspace root, **is** the project repository root. There is no separate `repo/` subdirectory. This means:

- `rpgkit init my-project` creates `my-project/` containing both your source code (`src/`, `tests/`, `docs/`) and RPG-Kit's in-workspace configuration files (`.rpgkit/config.toml`, `.claude/`, `.github/`, `.vscode/`, depending on the selected agent).
- `rpgkit init --here` inside an existing git repository adds RPG-Kit on top of the existing code without moving the repository.
- A single `.git` repository tracks user-owned code and any RPG-Kit files the user chooses to commit. **Runtime data, logs, and the inner-git snapshot repo all live outside the workspace** under `~/.rpgkit/workspaces/<workspace-id>/`, so generated artefacts don't pollute your repo or accidentally get committed. Only a small set of user-facing files (`.rpgkit/config.toml`, `.rpgkit/reports/*.html`) stay inside the workspace.

## After `rpgkit init`

Running `rpgkit init` downloads a template and creates a structure like this:

```text
my-project/
├── docs/                               # Optional requirement docs for /rpgkit.feature_spec
│   ├── project_charter.md              # Auto-detected when no description is provided
│   └── ...
├── .claude/                            # Claude Code configuration when --ai claude
│   ├── commands/                       # /rpgkit.* command definitions
│   │   ├── rpgkit.feature_spec.md
│   │   ├── rpgkit.feature_build.md
│   │   ├── rpgkit.feature_refactor.md
│   │   ├── rpgkit.feature_edit.md
│   │   ├── rpgkit.build_skeleton.md
│   │   ├── rpgkit.build_data_flow.md
│   │   ├── rpgkit.design_base_classes.md
│   │   ├── rpgkit.design_interfaces.md
│   │   ├── rpgkit.plan_tasks.md
│   │   ├── rpgkit.code_gen.md
│   │   ├── rpgkit.rpg_edit.md
│   │   ├── rpgkit.encode.md
│   │   └── rpgkit.update_rpg.md
│   └── settings.json                   # Permissions and MCP auto-approval
├── .github/                            # Copilot configuration when --ai copilot
│   ├── agents/                         # rpgkit.* agent definitions
│   └── prompts/                        # companion prompts
├── .vscode/                            # Copilot/VS Code configuration when applicable
│   ├── mcp.json                        # MCP server registration
│   └── tasks.json                      # Optional workspace tasks
└── .rpgkit/
│   ├── config.toml                     # Workspace AI / config (committed). See docs/configuration.md
│   ├── .source                         # Provisioning channel marker: "bundle" or "legacy"
│   └── reports/                        # User-facing HTML reports (rpg.html, review HTML, ...)
└── .git/                               # Your existing git repo
    └── hooks/                          # Installed by `rpgkit init`
        ├── post-commit                 # Single line: `rpgkit hook post-commit`
        └── post-merge                  # Single line: `rpgkit hook post-merge`
```

### Out-of-workspace runtime store

Starting from the global-install layout, all runtime state lives under your home directory, keyed by a path-derived **slug** (the workspace's absolute path, lowercased, with non-alphanumeric runs collapsed to `-`):

```text
~/.rpgkit/workspaces/<workspace-id>/
├── .git/        # Inner-git snapshot repo (per-stage auto-commits)
├── .gitignore   # Excludes logs/copilot/ only — other logs are tracked for debug
├── .meta.toml   # Back-pointer to the workspace path + metadata
├── data/        # Runtime artifacts (rpg.json, dep_graph.json, ...)
└── logs/        # Per-stage logs (tracked by inner-git; LLM session traces under logs/copilot/ are excluded)
```

Reports (`rpg.html`, review HTML, …) stay **inside** the workspace at `<workspace>/.rpgkit/reports/` because they are small, user-facing artefacts that benefit from sitting next to the code (and may be committed).

`<workspace-id>` is normally the slug itself (e.g. `home-hys-projects-myrepo`); paths whose slug exceeds 200 characters are truncated and given a 6-char base36 SHA-256 suffix so the directory name fits comfortably under POSIX `NAME_MAX` (255). Same shape as Claude Code's `~/.claude/projects/`. Moving or renaming the workspace yields a different id, so each clone has independent state. Run `rpgkit version` from inside the workspace to see the resolved paths (the **Data**, **Logs**, and **Inner git** lines). For backward compatibility, workspaces created before 0.1.4 (which used a 12-hex-char SHA-256 hash directory) continue to resolve correctly.

> Pipeline scripts (formerly materialised into `.rpgkit/scripts/`) now live inside the installed `rpgkit-cli` wheel under `rpgkit_cli/core_pack/scripts/` and are invoked via the global [`rpgkit script <name>`](cli-reference.md) command. They are no longer copied into each workspace, so `rpgkit init` produces a much smaller footprint and a single source of truth per CLI install.

The agent configuration directory varies by the selected AI assistant and release package. For the verified CLI path, `--ai claude` installs `.claude/commands/`, while `--ai copilot` installs `.github/agents/`, `.github/prompts/`, and `.vscode/mcp.json`.

Command definitions are installed into the AI-agent-specific folder. Normal users should not need to inspect `~/.rpgkit/workspaces/<workspace-id>/data/` directly—run `rpgkit version` from the workspace to see all relevant paths.

### Quick reference: where does each file live?

| Artefact | Location |
|---|---|
| Your source code | `<workspace>/` |
| Workspace AI config | `<workspace>/.rpgkit/config.toml` |
| User-facing HTML reports (`rpg.html`, …) | `<workspace>/.rpgkit/reports/` |
| Agent command definitions | `<workspace>/.claude/` or `<workspace>/.github/` |
| MCP / VS Code config | `<workspace>/.vscode/` |
| Git hooks (`post-commit`, `post-merge`) | `<workspace>/.git/hooks/` |
| Generated data (`rpg.json`, `dep_graph.json`, …) | `~/.rpgkit/workspaces/<workspace-id>/data/` |
| Per-stage logs | `~/.rpgkit/workspaces/<workspace-id>/logs/` |
| Inner-git snapshot repo | `~/.rpgkit/workspaces/<workspace-id>/.git/` |
| Pipeline scripts (read-only) | inside the installed `rpgkit-cli` wheel |

To see the resolved paths for the current workspace, run `rpgkit version` from anywhere inside it.

## Generated Data Files

As you run `/rpgkit.*` commands, `~/.rpgkit/workspaces/<workspace-id>/data/` is progressively populated (paths below are shown relative to that directory):

| Generated file | Command | Description |
| -------------- | ------- | ----------- |
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
| `tasks.json` | `plan_tasks` | Implementation task batches |
| `code_gen_state.jsonl` | `code_gen` | Code generation progress state, append-only JSONL |
| `rpg_edit_impact.json` | `rpg_edit` | Impact analysis for a surgical edit |
| `rpg_edit_plan.json` | `rpg_edit` | Confirmed surgical edit plan |
| `rpg_edit_code_result.json` | `rpg_edit` | Code application result for a surgical edit |
| `trajectory/` | All scripts | Execution trajectory logs |

## `rpg.json` — The Repository Planning Graph

`rpg.json` is the central graph artifact used by the forward pipeline, reverse encoder, MCP tools, incremental update hooks, and `/rpgkit.rpg_edit`.

It can be created in either direction:

1. **Forward pipeline:** `/rpgkit.build_skeleton` creates `rpg.json` from `feature_tree.json`.
2. **Reverse encoder:** `/rpgkit.encode` creates `rpg.json` from an existing codebase.

Later commands enrich or maintain the same file:

1. **`build_data_flow`** — adds data-flow edges between components.
2. **`design_base_classes`** — adds base-class relationship edges.
3. **`design_interfaces`** — adds fine-grained dependency edges such as inheritance, invocation, and references.
4. **`code_gen`** — updates implementation status as code is generated.
5. **`update_rpg`** — incrementally updates the RPG after commits when the hook is skipped or needs to be run manually.
6. **`rpg_edit`** — applies targeted feature graph edits together with code and dependency graph changes.

## `dep_graph.json` — Code Dependency Graph

`dep_graph.json` stores the code-level dependency graph used by the encoder, incremental update path, and surgical edit path. It is maintained alongside `rpg.json` so RPG-Kit can keep feature-level structure and code-level dependencies aligned.

Typical producers and updaters:

- `/rpgkit.encode` creates the initial dependency graph when encoding an existing codebase.
- The post-commit hook and `/rpgkit.update_rpg` refresh it after code changes.
- `/rpgkit.rpg_edit` refreshes it after applying targeted code edits.

## Runtime Logs and Reports

Runtime logs are written under `~/.rpgkit/workspaces/<workspace-id>/logs/`, for example:

- `~/.rpgkit/workspaces/<workspace-id>/logs/encode.log`
- `~/.rpgkit/workspaces/<workspace-id>/logs/update_rpg.log`
- `~/.rpgkit/workspaces/<workspace-id>/logs/feature_build.log`
- `~/.rpgkit/workspaces/<workspace-id>/logs/build_data_flow.log`

Execution traces are written under `~/.rpgkit/workspaces/<workspace-id>/data/trajectory/`. Review or diagnostic artifacts may be written under `<workspace>/.rpgkit/reports/` when a command generates them.

To discover the home-side paths (data / logs / inner-git) for the current workspace, run `rpgkit version` from anywhere inside it—the relevant lines are labelled **Workspace**, **Data**, **Logs**, and **Inner git**.
