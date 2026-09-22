<h1 align="center">CoderMind</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

> [!NOTE]
> CoderMind is at an early stage (v0.1.x). Interfaces may still change.

## Make coding agents plan before they edit

Coding agents are strong at local edits, but repo-level tasks often fail without a stable planning structure. Requirements drift, architecture decisions disappear, multi-file generation becomes inconsistent, and updates can miss hidden dependencies.

CoderMind gives Claude Code and GitHub Copilot a **persistent RPG workspace** for repository-level coding. The workspace is built around a Repository Planning Graph (RPG) that connects requirements, features, architecture, files, code entities, and dependencies.

With CoderMind, agents work through graph-driven workflows:

- **Build**: turn requirements into an RPG plan, then generate a multi-file repository.
- **Understand**: map an existing repo into RPG, then search, explore, and explain it.
- **Update**: locate affected RPG nodes, plan the edit, and update code and graph together.

### Choose your workflow

| Goal | Workflow | Start here |
|---|---|---|
| Build a new repository from requirements | Build workflow (requirements → RPG → code) | [`Quick Start: New Repository`](#quick-start-new-repository) |
| Understand an existing repository | Understand workflow (repository → RPG → search/explore) | [`Quick Start: Existing Repository`](#quick-start-existing-repository) |
| Update an existing repository | Update workflow (change request → affected RPG nodes → edit plan → code/RPG update) | [`Quick Start: Existing Repository`](#quick-start-existing-repository) |

### Detailed pipeline

New users can skip this and start from the Quick Start sections below.

<details>
<summary>Full command-level workflow diagram</summary>

```text
Forward Direction: Requirements → RPG → Code

 Phase 1: Feature Specification       Phase 2: RPG Construction & Planning                             Phase 3
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ feature  │ │ feature  │ │ feature  │ │  build   │ │  build   │ │ design   │ │ design   │ │  plan    │ │          │
│  _spec   ├─▶  _build  ├─▶_refactor ├─▶ skeleton ├─▶  data    ├─▶  base    ├─▶interfaces├─▶  tasks  ├─▶ code_gen │
│          │ │          │ │          │ │          │ │  flow    │ │ classes  │ │          │ │          │ │   (TDD)  │
└──────────┘ └──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────┬─────┘
 feature_     feature_        │        skeleton     data_flow    base_        interfaces   tasks        source
 spec/        build           │        .json        .json        classes      .json        .json        code
 feature_     .json           │        skeleton_    data_flow    .json
 spec.json                    │        summary.txt  _viz.html
                              │
                       ┌──────▼──────┐
                       │ feature_edit│ optional pre-planning edits to feature_tree.json
                       └─────────────┘
                                        ╰───── rpg.json (created → progressively enriched) ─────╯
                                                                            │
                                                                            ▼
                                                                     ┌──────────┐
Surgical edit workflow: Requirements -> RPG update -> Code Update    │ rpg_edit │ optional synchronized RPG + code + dep_graph edits
                                                                     └──▲────▲──┘
                                                                        │    │
Reverse Direction: Code → RPG                                           │    │
                                                                        │    │
┌──────────────────┐         ┌──────────┐       ┌──────────┐            │    │
│ Existing Codebase│────────▶│  encode  │──────▶│update_rpg│────────────┘    │
│                  │         │  (full)  │       │ (explicit│                 │
└──────────────────┘         └────┬─────┘       │ update)  │                 │
                              rpg.json          └──────────┘                 │
                              dep_graph.json     rpg.json / dep_graph.json   │
                                  │                                          │
                                  └──────────────────────────────────────────┘
                                                  ▲
                                                  │ opt-in post-commit hook: deterministic sync only (no AI update)

MCP Server: search_rpg / explore_rpg / get_node_detail / list_rpg_tree
```

</details>

### CoderMind in action

Below is part of the graph visualization generated for this repository. After running `/cmind.encode`, you can open `<workspace>/.cmind/reports/rpg.html` to browse the full interactive graph. Run `cmind version` to see the resolved paths for the current workspace.

![CoderMind repository graph visualization](../docs/cmind_visualized_graph.png)

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- An installed and authenticated AI coding agent CLI: [GitHub Copilot](https://docs.github.com/en/copilot) or [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### Install CoderMind

```bash
# For persistent installation (Recommended)
uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
cmind check

# For one-time usage
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name> --ai claude
```

Since `0.1.3`, the wheel ships the pipeline scripts and slash-command templates as packaged assets, so the template provisioning step of `cmind init` works offline. Optional initial encoding (`--encode`) may invoke AI services and require network access; self-upgrade during `cmind update` may also require network access.

The examples use `--ai claude`; replace it with `--ai copilot` for GitHub Copilot. Interactive init can prompt if `--ai` is omitted; non-TTY init requires explicit `--ai`, even with an environment override.

## Quick Start: New Repository

Use this path when you want CoderMind to turn requirements into a new codebase.

> [!WARNING]
> For projects with a large amount of generated code, `/cmind.design_interfaces` and `/cmind.code_gen` can take a long time to run. As a typical example: 100 features take about 30 minutes.

1. Initialize a new project:

   ```bash
   cmind init my-project --ai claude
   cd my-project
   ```

   Common variants:

   ```bash
   cmind init my-project --ai claude --script sh
   cmind init my-project --ai copilot
   ```

2. **[Optional]** place your requirement documents in `my-project/docs/`.

3. Launch your AI coding agent in the project directory.

4. Run the forward pipeline:

   ```text
   /cmind.feature_construct <feature description>
   [Optional] /cmind.feature_edit <edit instructions>
   /cmind.plan
   /cmind.code_gen
   [Optional] /cmind.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **Coding Agents are invoked slightly differently**:
>
> - **Claude Code**: type `/cmind.feature_construct ...` directly in the chat — slash commands are recognised and dispatch the matching workflow.
> - **GitHub Copilot CLI**: slash commands are not supported (custom agents are), so first run `/agent cmind.feature_construct` to switch to the target agent, then type `start` to run its built-in workflow.

CoderMind progressively builds `rpg.json` in the home-side runtime directory (`~/.cmind/workspaces/<workspace-id>/data/rpg.json`) and uses it to keep requirements, planning artifacts, generated code, and dependency information aligned. Bulk RPG data stays outside the repo; generated reports remain in the workspace.

## Quick Start: Existing Repository

Use this path when you already have a repository and want an AI agent to understand or edit it with RPG context.

> [!WARNING]
> For larger projects, `cmind init . --ai claude --encode` and `/cmind.encode` can take a long time to run. As a typical example: 200 source files take about 100 minutes.

1. Initialize CoderMind in the repository root and build the initial graph:

   ```bash
   cd existing-repo/
   cmind init . --ai claude --encode    # --encode builds the RPG from the current code
   ```

   If you want to skip the confirmation prompt for a non-empty directory:

   ```bash
   cmind init . --ai claude --force --encode
   ```

2. Launch your AI coding agent in the repository.

3. **[Optional]** Use the generated RPG through MCP tools and slash commands. The following commands are only needed when run manually:

   ```text
   /cmind.encode                                  # rebuild the full RPG when needed
   /cmind.update_rpg                              # explicitly request an AI-driven incremental update
   /cmind.rpg_edit <edit instructions>            # graph-aware code edit
   ```

4. Request AI-driven graph updates explicitly with `/cmind.update_rpg`. Git hooks are off by default; opting in enables deterministic, foreground sync only, not AI updates or background workers.

## What happens after `cmind init`

`cmind init` provisions command definitions, a workspace marker/configuration, MCP registration, and status integrations without editing your source files. Generated reports live in the workspace; bulk runtime data (RPG artifacts and logs) lives under `~/.cmind/workspaces/<workspace-id>/`, where `<workspace-id>` is a slug derived from the workspace's absolute path (e.g. `home-hys-projects-myrepo`).

```text
my-project/
├── docs/                 # Optional requirement docs for /cmind.feature_construct
├── .github/ or .claude/  # Coding Agent command definitions and settings
├── .vscode/              # Copilot/VS Code MCP and status integration when applicable
├── .cmind/              # Workspace marker/configuration and generated reports
└── .git/hooks/           # Optional post-commit / post-merge deterministic sync only
```

CoderMind Git hooks are **OFF by default**. Pass `--git-hooks` on **each** init/update invocation to opt into deterministic sync only. Omitting it or using `--no-git-hooks` removes recognized CoderMind-owned blocks while preserving other user/team hook content. CoderMind-managed Git hooks do not launch AI or background workers. Claude `SessionStart` and Copilot/VS Code `folderOpen` remain **status-only**, separate from Git hooks and unaffected by `--no-git-hooks`.

See [docs/project-structure.md](docs/project-structure.md) for the full layout and data file reference.

### Execution configuration

Repository `recommended_provider` and valid legacy `ai_provider` / exact built-in `ai_cli_cmd` hints are **not runtime authority**. Init saves your explicit choice outside the repo at `~/.cmind/execution/<workspace-hash>/selection.json`, bound to the canonical workspace path and separate from slug-based RPG data. See [local execution selection](docs/configuration.md#user-local-execution-selection).

A clone at another path, a moved workspace, or a new user must choose again. Trusted CI may use `CMIND_AI_PROVIDER` for process execution; non-TTY init still requires `--ai`. `cmind update --ai claude` explicitly changes the saved choice; update without `--ai` preserves it (or leaves it absent), never authorizing execution from repository hints or detected integrations.

Unsafe raw-command or invalid configuration makes init/update fail preflight, before provisioning or hook migration. `--ai`, `--force`, and environment overrides do not bypass validation. Correct the configuration as described in the [migration checklist](docs/configuration.md#updating-an-existing-codermind-project).

## Updating CoderMind

First install the repaired CLI using your installation method, for example:

```bash
uv tool install cmind-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
   --force \
   --reinstall
```

Then migrate **each existing workspace**: review and correct invalid configuration before running the explicit selection below (replace `claude` with `copilot` if intended). Installing the CLI alone does not clean old workspace hooks.

```bash
cd <your-workspace>
cmind update --ai claude --no-upgrade --no-git-hooks

# Routine updates preserve the local choice; they do not create missing consent
cmind update
```

Review unrecognized legacy hooks manually, preserving unrelated user/team hooks. Use `--git-hooks` on each update only if deterministic sync is wanted. See the [migration checklist](docs/configuration.md#updating-an-existing-codermind-project).

## Supported Platforms

**Coding Agent support**:

| Agent          | CLI usage | VS Code extension usage |
| -------------- | --------- | ----------------------- |
| Claude Code    | ✅        | ✅                      |
| GitHub Copilot | ✅        | ✅                      |
| Codex          | ⌛        | ⌛                      |

**Operating system support**:

| Operating system | Status |
| ---------------- | ------ |
| Linux            | ✅     |
| macOS            | ⌛     |
| Windows          | ⌛     |

Windows support remains partial: templates are `sh`-only (`ps` is not supported). AI launches use installed native `.exe` files or supported, known Claude/Copilot npm entries; `.cmd`, `.bat`, and `.ps1` wrappers are never executed. This is not a live-release compatibility guarantee. Normal provider approvals apply, with no broad permission bypass; read-only MCP preapproval is separate. See [executable policy](docs/configuration.md#executable-and-permission-policy) and [MCP permissions](docs/configuration.md#assistant-permissions-and-scope).

## Documentation

- [Slash command reference](docs/commands.md) — every `/cmind.*` command, inputs, outputs, and examples.
- [CLI reference](docs/cli-reference.md) — `cmind init`, `cmind update`, `cmind check`, `cmind version`, and all options.
- [Configuration](docs/configuration.md) — local provider selection, MCP permissions, opt-in hooks, migration, and troubleshooting.
- [Project structure](docs/project-structure.md) — files and directories created by CoderMind.

## Upcoming Features

- **Simpler generation commands:** merge the current multi-step generation flow into fewer commands, such as `/cmind.generate_repo` and `/cmind.generate_feature`. `/cmind.plan` has shipped in 0.1.4.
- **Multi-language support:** add support for Go, C++, Rust, JavaScript/TypeScript, and more.
- **More platform integrations:** support CoderMind across CLI and VS Code extension workflows for different AI coding agents on different systems.

## Troubleshooting

**AI assistant CLI not found:** run `cmind check`, install and authenticate the selected assistant CLI, then rerun `cmind init` or `cmind update`.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgements

Based on [GitHub Spec-Kit](https://github.com/github/spec-kit).
