# CLI Reference

This document covers the `cmind` command-line interface. Use the CLI to install templates, initialize projects, update CoderMind files, and verify local tool availability.

## `cmind init`

Initialize a new project from the latest template, or add CoderMind to an existing repository.

```bash
cmind init <project-name> [options]
cmind init --here [options]
cmind init . [options]
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | Default encoder/decoder backend: `copilot`, `claude`, or `codex` |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--here` | Initialize in current directory |
| `--force` | Skip confirmation for non-empty current directory |
| `--no-git` | Skip git initialization |
| `--no-mcp` | Skip MCP server configuration |
| `--ignore-agent-tools` | Skip checks for AI agent CLI tools |
| `--encode/--no-encode` | Run or skip initial RPG encoding at the end of init |
| `--debug` | Show verbose diagnostic output |

### Supported AI Assistants

| Agent | Folder | Invocation | Status |
| ----- | ------ | ---------- | ------ |
| `copilot` | `.github/`, `.vscode/` | `/cmind.*` | Verified |
| `claude` | `.claude/` | `/cmind.*` | Verified |
| `codex` | `.agents/skills/`, `.codex/` | `$cmind-*` or `/skills` | CLI verified |

All three integrations are generated together. `--ai` selects only the default
LLM backend used by encoder/decoder pipeline calls.

### Examples

```bash
cmind init my-project
cmind init my-project --ai claude --script sh
cmind init my-project --ai codex --script sh
cmind init . --force
cmind init . --encode
cmind init . --force --encode
cmind init --here --ai copilot
```

## `cmind update`

Update CoderMind template files, all agent integrations, MCP configuration,
gitignore rules, and hooks. The active LLM backend is preserved unless `--ai`
is passed.

```bash
cmind update
cmind update --ai claude
cmind update --no-mcp
cmind update --no-upgrade
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | Change the active encoder/decoder backend |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--no-upgrade` | Skip the default-on CLI self-upgrade and only sync workspace files. |
| `--no-mcp` | Skip MCP server configuration |
| `--debug` | Show verbose diagnostic output |

### Auto-upgrade behaviour

Since the global-install layout, `cmind update` performs a **best-effort silent self-upgrade by default** when the install source is safe to refresh (git+URL or PyPI). After upgrading the CLI it re-executes itself once to continue the workspace sync with the new code. Editable installs, local-file installs, and unknown sources are skipped silently.

- Pass `--no-upgrade` to skip the upgrade entirely (useful for offline or pinned environments).
- A loop guard environment variable (`CMIND_UPGRADE_DONE`) is set across the re-exec to guarantee at most one upgrade attempt per invocation.

### Provisioning sources

As of `0.1.4`, `cmind init` and `cmind update` provision exclusively
from the **packaged assets bundle** shipped inside the installed
`cmind-cli` wheel (under `cmind_cli/core_pack/`).  No network access
is required at provisioning time.

To pick up newer prompts and templates, upgrade the CLI itself
(e.g. `uv tool upgrade cmind-cli`).  `cmind update` does this
automatically by default (see *Auto-upgrade behaviour* above); pass
`--no-upgrade` to opt out.

## `cmind check`

Verify that the local environment has the tools CoderMind relies on.

```bash
cmind check
```

Probes for Git, the supported AI assistant CLIs (GitHub Copilot,
Claude Code), and optional editors (VS Code / VS Code Insiders), and
prints a tree of which ones are available.  Run this after
installation to confirm the environment is ready, or whenever a
pipeline step complains about a missing tool.

## `cmind config`

Inspect or switch the active encoder/decoder LLM backend without changing the
installed Claude, Copilot, or Codex integrations.

```bash
cmind config show
cmind config set-agent codex
cmind config set-agent claude
cmind config set-agent copilot
```

## `cmind version`

Display version and system information.

```bash
cmind version
```

## `cmind script`

Execute one of the bundled CoderMind pipeline scripts.  After install
(`uv tool install cmind-cli`) the scripts live inside the wheel under
`cmind_cli/core_pack/scripts/` and are no longer copied into each
workspace; this command is the supported way to invoke them.

```bash
cmind script <relpath> [args...]
```

Arguments after `<relpath>` are forwarded verbatim to the target
script.  Standard input/output/error and exit code are inherited.

### Options

- `--list` — print every available script (relative path) and exit.
- `--where <name>` — print the absolute filesystem path of one script
  and exit; pipeable into `$(...)` for ad-hoc inspection.

The `.py` suffix on `<relpath>` is optional.  Path traversal (`..`)
and absolute paths are rejected for safety.

### Examples

```bash
cmind script smoke_test.py --json
cmind script rpg_edit/validate.py
cmind script --list
cmind script --where mcp_server.py
```

The Claude/Copilot commands and Codex skills installed by `cmind init` all use
`cmind script …` under the hood, so every agent invokes the same pipeline
contract.

A companion console script, `cmind-mcp`, is the MCP server entry
point and is what `.mcp.json`, `.vscode/mcp.json`, and
`.codex/config.toml` register as the `rpg-tools` command.
