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
│                  │         │  (full)  │       │ (manual  │                 │
└──────────────────┘         └────┬─────┘       │ fallback)│                 │
                              rpg.json          └──────────┘                 │
                           (includes dep_graph)          rpg.json              │
                                  │                                          │
                                  └──────────────────────────────────────────┘
                                                  ▲
                                                  │ post-commit hook normally runs incremental updates

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
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name>
```

Since `0.1.3`, the wheel ships the pipeline scripts and slash-command templates as packaged assets, so `cmind init` works offline (for example in air-gapped or corporate proxy environments).

## Quick Start: New Repository

Use this path when you want CoderMind to turn requirements into a new codebase.

> [!WARNING]
> For projects with a large amount of generated code, `/cmind.design_interfaces` and `/cmind.code_gen` can take a long time to run. As a typical example: 100 features take about 30 minutes.

1. Initialize a new project:

   ```bash
   cmind init my-project
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

CoderMind progressively builds `rpg.json` in the home-side runtime directory (`~/.cmind/workspaces/<workspace-id>/data/rpg.json`) and uses it to keep requirements, planning artifacts, generated code, and dependency information aligned. Your workspace source files are not polluted.

## Quick Start: Existing Repository

Use this path when you already have a repository and want an AI agent to understand or edit it with RPG context.

> [!WARNING]
> For larger projects, `cmind init . --encode` and `/cmind.encode` can take a long time to run. As a typical example: 200 source files take about 100 minutes.

1. Initialize CoderMind in the repository root and build the initial graph:

   ```bash
   cd existing-repo/
   cmind init . --encode    # --encode builds the RPG from the current code
   ```

   If you want to skip the confirmation prompt for a non-empty directory:

   ```bash
   cmind init . --force --encode
   ```

2. Launch your AI coding agent in the repository.

3. **[Optional]** Use the generated RPG through MCP tools and slash commands. The following commands are only needed when run manually:

   ```text
   /cmind.encode                                  # rebuild the full RPG when needed
   /cmind.update_rpg                              # manual incremental update fallback
   /cmind.rpg_edit <edit instructions>            # graph-aware code edit
   ```

4. After each commit, the git hook installed by CoderMind automatically calls the `cmind hook <name>` dispatcher to update the RPG and keep it aligned with code changes. If the hook fails or is skipped, run `/cmind.update_rpg` manually.

## What happens after `cmind init`

`cmind init` does not modify your source files, **and it does not write runtime state into your workspace**. It only adds command definitions, MCP configuration, and hooks to your workspace. CoderMind runtime data (artifacts and logs) lives under the home-side directory `~/.cmind/workspaces/<workspace-id>/`, where `<workspace-id>` is a slug derived from the workspace's absolute path (e.g. `home-hys-projects-myrepo`).

```text
my-project/
├── docs/                 # Optional requirement docs for /cmind.feature_construct
├── .github/ or .claude/  # Coding Agent command definitions and settings
├── .vscode/              # Copilot/VS Code MCP configuration when applicable
├── .cmind/              # Generated reports and configuration files
└── .git/hooks/           # post-commit / post-merge installed by cmind init (each hook is one line: `cmind hook <name>`)
```

See [docs/project-structure.md](docs/project-structure.md) for the full layout and data file reference.

## Updating CoderMind

```bash
uv tool install cmind-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
   --force \
   --reinstall

# Update an existing workspace
cd <your-workspace>
cmind update
```

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

## Documentation

- [Slash command reference](docs/commands.md) — every `/cmind.*` command, inputs, outputs, and examples.
- [CLI reference](docs/cli-reference.md) — `cmind init`, `cmind update`, `cmind check`, `cmind version`, and all options.
- [Configuration](docs/configuration.md) — AI assistant setup, MCP registration, hooks, auto-approval, and troubleshooting.
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
