# CLI Reference

This document covers the `rpgkit` command-line interface. Use the CLI to install templates, initialize projects, update RPG-Kit files, and verify local tool availability.

## `rpgkit init`

Initialize a new project from the latest template, or add RPG-Kit to an existing repository.

```bash
rpgkit init <project-name> [options]
rpgkit init --here [options]
rpgkit init . [options]
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | AI assistant: `copilot` or `claude` |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--here` | Initialize in current directory |
| `--force` | Skip confirmation for non-empty current directory |
| `--no-git` | Skip git initialization |
| `--no-mcp` | Skip MCP server configuration |
| `--ignore-agent-tools` | Skip checks for AI agent CLI tools |
| `--encode/--no-encode` | Run or skip initial RPG encoding at the end of init |
| `--debug` | Show verbose diagnostic output |

### Supported AI Assistants

| Agent | Folder | Description | Status |
| ----- | ------ | ----------- | ------ |
| `copilot` | `.github/`, `.vscode/` | GitHub Copilot | Verified |
| `claude` | `.claude/` | Claude Code | Verified |

RPG-Kit currently supports only **GitHub Copilot** and **Claude Code** in the CLI. Additional agents may be adapted in future releases.

### Examples

```bash
rpgkit init my-project
rpgkit init my-project --ai claude --script sh
rpgkit init . --force
rpgkit init . --encode
rpgkit init . --force --encode
rpgkit init --here --ai copilot
```

## `rpgkit update`

Update RPG-Kit template files, scripts, command definitions, MCP configuration, gitignore rules, and hooks in an existing project. The AI assistant is auto-detected from existing project configuration when possible.

```bash
rpgkit update
rpgkit update --ai claude
rpgkit update --no-mcp
rpgkit update --no-upgrade
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | AI assistant, auto-detected if not specified |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--no-upgrade` | Skip the default-on CLI self-upgrade and only sync workspace files. |
| `--no-mcp` | Skip MCP server configuration |
| `--debug` | Show verbose diagnostic output |

### Auto-upgrade behaviour

Since the global-install layout, `rpgkit update` performs a **best-effort silent self-upgrade by default** when the install source is safe to refresh (git+URL or PyPI). After upgrading the CLI it re-executes itself once to continue the workspace sync with the new code. Editable installs, local-file installs, and unknown sources are skipped silently.

- Pass `--no-upgrade` to skip the upgrade entirely (useful for offline or pinned environments).
- A loop guard environment variable (`RPGKIT_UPGRADE_DONE`) is set across the re-exec to guarantee at most one upgrade attempt per invocation.

### Provisioning sources

As of `0.1.4`, `rpgkit init` and `rpgkit update` provision exclusively
from the **packaged assets bundle** shipped inside the installed
`rpgkit-cli` wheel (under `rpgkit_cli/core_pack/`).  No network access
is required at provisioning time.

To pick up newer prompts and templates, upgrade the CLI itself
(e.g. `uv tool upgrade rpgkit-cli`).  `rpgkit update` does this
automatically by default (see *Auto-upgrade behaviour* above); pass
`--no-upgrade` to opt out.

## `rpgkit check`

Verify that the local environment has the tools RPG-Kit relies on.

```bash
rpgkit check
```

Probes for Git, the supported AI assistant CLIs (GitHub Copilot,
Claude Code), and optional editors (VS Code / VS Code Insiders), and
prints a tree of which ones are available.  Run this after
installation to confirm the environment is ready, or whenever a
pipeline step complains about a missing tool.

## `rpgkit version`

Display version and system information.

```bash
rpgkit version
```

## `rpgkit script`

Execute one of the bundled RPG-Kit pipeline scripts.  After install
(`uv tool install rpgkit-cli`) the scripts live inside the wheel under
`rpgkit_cli/core_pack/scripts/` and are no longer copied into each
workspace; this command is the supported way to invoke them.

```bash
rpgkit script <relpath> [args...]
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
rpgkit script smoke_test.py --json
rpgkit script rpg_edit/validate.py
rpgkit script --list
rpgkit script --where mcp_server.py
```

The slash-command templates installed by `rpgkit init` (in
`.claude/commands/` or `.github/agents/`) all use `rpgkit script …`
under the hood, so AI agents invoke the pipeline through the same
contract.

A companion console script, `rpgkit-mcp`, is the MCP server entry
point and is what `.mcp.json` / `.vscode/mcp.json` register as the
`rpg-tools` command — no absolute paths in the config, no per-machine
edits.
