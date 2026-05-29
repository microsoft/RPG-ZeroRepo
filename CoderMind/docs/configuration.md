# Configuration

This document covers CoderMind configuration that is useful after installation: AI assistant setup, MCP registration, auto-approval, hooks, and initial encoding.

> **Data paths.** References below such as `.cmind/data/rpg.json` and `.cmind/logs/...` are logical names. Runtime files actually live under `~/.cmind/workspaces/<workspace-id>/{data,logs}/` so they stay outside your git repo, where `<workspace-id>` is the slug-based workspace identifier used by the home-side store (with an optional `-<hash6>` suffix when needed). Reports stay in the workspace at `<workspace>/.cmind/reports/`. The MCP server, hooks, and pipeline scripts all resolve the home-dir location automatically from the workspace root. Run `cmind version` from inside the workspace to see the resolved Data / Logs paths; see [project-structure.md](project-structure.md) for the full layout.

## AI Assistant CLI Requirements

CoderMind slash commands are executed by an AI coding agent. Before running `cmind init`, install and authenticate at least one supported AI assistant CLI.

Currently verified assistants:

| Agent | `--ai` value | Generated configuration | Requirement |
| ----- | ------------ | ----------------------- | ----------- |
| GitHub Copilot | `copilot` | `.github/`, `.vscode/` | Copilot CLI available and authenticated |
| Claude Code | `claude` | `.claude/` | Claude Code CLI available and authenticated |

Use `cmind check` to verify required local tools.

```bash
cmind check
```

If the selected AI assistant is not found, install and authenticate it, then rerun `cmind init` or `cmind update`.

## Workspace Configuration (`.cmind/config.toml`)

Since `0.1.3`, every workspace owns a `.cmind/config.toml` file that records which AI CLI command the pipeline scripts should invoke. This decouples the scripts from a single AI at packaging time — the same packaged scripts now serve any AI you pick.

```toml
# .cmind/config.toml
[cmind]
ai_cli_cmd = "claude"
```

The file is created automatically by `cmind init --ai <agent>`. Edit it any time to switch the workspace to a different AI; no need to re-run `init`.

### Resolution priority

When a pipeline script (or a hook, or the MCP server) needs to invoke the AI CLI, it resolves the command via the following chain. The first non-empty value wins:

| # | Source | Use case |
| - | ------ | -------- |
| P1 | `LLMClient(tool="...")` constructor argument | Programmatic override (rare) |
| P2 | `CMIND_AI_CLI_CMD` environment variable | CI runs, one-off experiments |
| P3 | `.cmind/config.toml` `[cmind].ai_cli_cmd` | Normal default (per workspace) |
| P4 | Release-zip baked-in literal | Legacy workspaces provisioned before v0.1.4 |

If all four resolve to empty, the next `LLMClient.generate()` call raises a `RuntimeError` instructing the user to run `cmind init` or set the env var.

### Supported AI CLI commands

The values written to `ai_cli_cmd` mirror the per-AI substitutions performed by the GitHub release-zip CI:

| `--ai` value | `ai_cli_cmd` |
| ------------ | ------------ |
| `copilot` | `copilot` |
| `claude` | `claude` |
| `gemini` | `gemini -p` |
| `qwen` | `qwen -p` |
| `cursor-agent` | `agent -p` |
| `auggie` | `augment -p` |
| `codex` | `codex exec` |
| `codebuddy` | `codebuddy -p` |
| `qoder` | `qodercli -p` |
| `opencode` | `opencode run` |
| `amp` | `amp --execute` |

Only `copilot` and `claude` are currently verified end-to-end; the others are scaffolded but may need integration adjustments.

### Other config keys

The `[cmind]` table currently holds only `ai_cli_cmd`. Future releases will add timeouts, retry budgets, and model overrides under the same namespace; older keys remain forward-compatible.

## Initialization Options

### AI assistant selection

```bash
cmind init my-project --ai claude
cmind init my-project --ai copilot
```

If `--ai` is omitted in an interactive terminal, CoderMind prompts for a supported assistant.

### Script type

```bash
cmind init my-project --script sh
cmind init my-project --script ps
```

`sh` installs POSIX shell-oriented command snippets. `ps` installs PowerShell-oriented snippets.

### MCP registration

By default, `cmind init` registers the CoderMind MCP server for the selected assistant.

```bash
cmind init my-project
```

Pass `--no-mcp` to skip MCP registration:

```bash
cmind init my-project --no-mcp
cmind update --no-mcp
```

Skipping MCP means the slash-command pipeline still works, but the AI assistant will not get the `rpg-tools` graph-query tools automatically.

### Initial encode

The MCP tools query `.cmind/data/rpg.json`. For existing codebases, that file is created by the encoder.

`cmind init` supports:

```bash
cmind init --here --encode
cmind init --here --no-encode
```

Behavior:

- `--encode` runs the encoder at the end of init without prompting.
- `--no-encode` skips the encoder prompt.
- If neither flag is provided, CoderMind may prompt in an interactive terminal when Python code is present.

You can always run the encoder later from the AI assistant:

```text
/cmind.encode
```

## MCP Server

CoderMind's MCP server is named `rpg-tools`. It reads `.cmind/data/rpg.json` and exposes read-only graph-query tools to the AI assistant.

| Tool | Purpose |
| ---- | ------- |
| `search_rpg` | Search code entities or features by keyword, path, function, class, or feature name |
| `explore_rpg` | Traverse dependencies and call chains from a starting node |
| `get_node_detail` | Fetch details for a specific node, optionally including source code |
| `list_rpg_tree` | Render the functional architecture as a tree |

If `.cmind/data/rpg.json` does not exist yet, the tools return an `rpg_unavailable` response with a next step telling the agent to run `/cmind.encode`.

## Assistant Configuration Files

### Claude Code

For Claude Code, CoderMind writes command definitions and settings under `.claude/`:

```text
.claude/
├── commands/              # /cmind.* command definitions
└── settings.json          # permissions and MCP auto-approval
```

The settings file grants project-scoped permissions needed by CoderMind commands, including access to the `rpg-tools` MCP server. Review `.claude/settings.json` if your team wants stricter local permission prompts.

### GitHub Copilot / VS Code

For Copilot, CoderMind writes agent instructions under `.github/` and VS Code MCP configuration under `.vscode/`:

```text
.github/
├── agents/                # cmind.* agent definitions
└── prompts/               # companion prompts
.vscode/
└── mcp.json               # rpg-tools registration
```

Open the project in VS Code after initialization so the workspace MCP configuration is available to Copilot.

## Auto-approval and Scope

CoderMind pre-authorizes the `rpg-tools` MCP server where the selected assistant supports project-scoped permissions. The goal is to avoid prompting on every graph query during chat.

Scope rules:

- Configuration is written into the project that ran `cmind init` or `cmind update`.
- User-level assistant settings are not modified.
- Passing `--no-mcp` skips MCP registration and related auto-approval entries.

## Git Hooks and Incremental Updates

CoderMind installs local git hooks to keep the RPG aligned with code changes.

The important hook behavior is:

- After commits, CoderMind can run an incremental update in the background.
- The update refreshes `.cmind/data/rpg.json` and `.cmind/data/dep_graph.json`.
- Logs are written to `.cmind/logs/update_rpg.log`.

Manual fallback:

```text
/cmind.update_rpg
```

Use `/cmind.update_rpg` when:

- The hook failed.
- The hook was skipped.
- You want to force a foreground update and inspect the result.

If the RPG seems significantly stale or corrupted, run a full encode instead:

```text
/cmind.encode
```

## Updating an Existing CoderMind Project

Run `cmind update` from the project root to refresh scripts, command definitions, MCP configuration, gitignore rules, and hooks.

```bash
cmind update
cmind update --ai claude
cmind update --no-upgrade
cmind update --no-mcp
```

`cmind update` auto-detects the existing assistant configuration when possible.

## Troubleshooting

### AI assistant CLI not found

Run:

```bash
cmind check
```

Install and authenticate the missing assistant CLI, or rerun init with the assistant you want:

```bash
cmind init my-project --ai claude
cmind init my-project --ai copilot
```

### MCP tools say `rpg_unavailable`

The MCP server is configured, but `.cmind/data/rpg.json` has not been created yet. Run:

```text
/cmind.encode
```

### Incremental update failed

Check:

```bash
tail -n 200 .cmind/logs/update_rpg.log
```

Then run:

```text
/cmind.update_rpg
```

If the graph is corrupted or too stale, run `/cmind.encode` for a full rebuild.

### Template download hits rate limits or private repo access errors

As of v0.1.4 `cmind init` and `cmind update` no longer fetch templates
from GitHub releases — templates are bundled inside the installed
`cmind-cli` wheel, so this class of error should no longer occur during
provisioning.  To pick up newer templates, upgrade the CLI itself:

```bash
uv tool upgrade cmind-cli
```

`cmind update` does this automatically by default; pass `--no-upgrade`
to opt out.
