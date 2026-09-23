# CLI Reference

This document covers the `cmind` command-line interface. Use the CLI to install templates, initialize projects, update CoderMind files, and verify local tool availability.

## `cmind init`

Initialize a new project using templates bundled with the installed CLI, or add CoderMind to an existing repository. Template provisioning needs no download; optional initial encoding is a separate AI operation and may use the network.

```bash
cmind init <project-name> [options]
cmind init --here [options]
cmind init . [options]
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | Explicit user-local provider and integration: `copilot` or `claude`; required for non-TTY init |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--here` | Initialize in current directory |
| `--force` | Skip confirmation for non-empty current directory |
| `--no-git` | Skip git initialization |
| `--no-mcp` | Skip MCP server configuration |
| `--no-copilot-cli-mcp` | For Copilot, skip registration in `~/.copilot/mcp-config.json`; retain workspace MCP setup unless `--no-mcp` is also set. Does not remove existing registrations. |
| `--no-cmind-git` | Skip creating the private snapshot repo at `~/.cmind/workspaces/<workspace-id>/.git/` for this init only; does not delete or disable an existing repo or control project Git hooks. |
| `--git-hooks/--no-git-hooks` | Default OFF: remove only recognized CoderMind-managed Git hook blocks, preserving other content. Opt in to deterministic sync only. |
| `--ignore-agent-tools` | Skip availability checks for AI agent CLI tools, not AI provider validation or execution policy |
| `--encode/--no-encode` | Run or skip initial RPG encoding at the end of init |
| `--debug` | Show verbose diagnostic output |

### Supported AI Assistants

| Agent | Folder | Description | Status |
| ----- | ------ | ----------- | ------ |
| `copilot` | `.github/`, `.vscode/` | GitHub Copilot | CLI integration; known npm adapter layouts |
| `claude` | `.claude/` | Claude Code | CLI integration; known npm adapter layouts |

CLI `--ai` selection exposes only **GitHub Copilot** and **Claude Code**. The runtime policy retains 11 historical provider IDs; the other nine have native-executable/scaffold compatibility only, no Windows npm adapters or verified integration support. See the [provider enum](configuration.md#provider-enum-and-legacy-compatibility). Adapter fixtures are not live npm-release tests; see the [Windows executable/npm requirements](configuration.md#executable-and-permission-policy).

### AI configuration and execution

New workspace configuration uses `[cmind].recommended_provider`: a tracked hint and workspace-discovery marker, **never execution authority**. Valid existing hints, including legacy `ai_provider` and exact built-in `ai_cli_cmd`, are preserved byte-for-byte. Only one hint key is allowed; raw commands are never executed. `--ai` does not overwrite a valid hint or import it into local state.

Runtime authority is P1 explicit trusted `LLMClient(tool=...)` (exact legacy command), P2 `CMIND_AI_PROVIDER` (enum) **or** legacy `CMIND_AI_CLI_CMD` (exact legacy command; mutually exclusive), then P3 user-local selection. Every consulted source must validate; malformed values fail closed without fallback. Repository hints are always preflighted, **even with environment/constructor overrides**. Legacy release-baked commands are validated but never authorize a call. No paths, extra flags, quoting, whitespace variants, or shell syntax are accepted as provider values.

Local consent lives under the resolved home directory at `~/.cmind/execution/<full-sha256-of-canonical-workspace-path>/selection.json`, separate from RPG metadata at `~/.cmind/workspaces/<workspace-id>/.meta.toml`. Its exact schema requires integer `schema_version: 1`, matching canonical `workspace` identity, and valid `ai_provider`. A clone/move/new user needs explicit selection; copying metadata or selection records is not a consent migration mechanism. See [local storage and validation](configuration.md#user-local-execution-selection).

Init requires `--ai` or an interactive provider choice independent of the recommendation. Non-TTY init without `--ai` fails before provisioning, even if the environment selects a provider. The explicit choice is saved only after provisioning and Git hook reconciliation succeed. Git hook cleanup/installation or local-save failure means no success message or initial encode; a failed atomic save preserves the prior selection. Separate `SessionStart` / `folderOpen` status integration installation remains best-effort.

Execution is shell-free with absolute external executables. Windows prioritizes direct `.exe` files; the adapter recognizes only exact Claude/Copilot npm package names and fixed native/JS entries, with canonical paths outside workspace/cwd and external absolute `node.exe` for JS. Wrappers are neither parsed nor executed; arbitrary user scripts and signature/authenticity guarantees are not supported. See the [exact layouts and trust boundary](configuration.md#executable-and-permission-policy).

The default Claude `--dangerously-skip-permissions` and Copilot `--allow-all` flags remain removed, with no opt-in bypass. Normal permissions may require approval or halt noninteractive runs. Both init and update preflight repository configuration before provisioning or hook migration; `--ai`, `--force`, and `--ignore-agent-tools` do not auto-trust invalid hints. [Trusted CI process selection](configuration.md#trusted-ci-process-selection) needs no persistent local consent, but still requires normal workspace lookup and validation; it is not a sandbox for untrusted PR workflows.

On configuration, authentication, access, or approval rejection, stop and show the exact error. Do not automatically retry, run init/update, rewrite configuration or local consent, or grant trust/permission overrides; wait for the user's explicit resolution and retry decision.

### Examples

```bash
cmind init my-project --ai claude
cmind init my-project --ai claude --script sh
cmind init . --ai claude --force
cmind init . --ai claude --encode
cmind init . --ai claude --force --encode
cmind init --here --ai copilot
cmind init --here --ai claude --git-hooks
```

## `cmind update`

Refresh workspace command definitions, MCP configuration, and gitignore rules from the installed CLI's bundle, and reconcile hooks in an existing project. Pipeline scripts stay in the installed wheel, not workspace copies. Update attempts a best-effort CLI self-upgrade by default, which may use the network; `--no-upgrade` skips it. Git hooks are OFF unless explicitly requested on this invocation.

With `--ai`, save that explicit provider locally only after provisioning and hook reconciliation succeed; valid repository hints remain unchanged. Without `--ai`, preserve the local record byte-for-byte (or leave it absent), even when an integration is auto-detected or chosen interactively. Template selection prefers a supported local provider, then detected folders, then an interactive integration-only choice. Malformed local state fails when consulted; non-TTY update without a determinable integration requires `--ai`. Neither detection nor recommendations create consent, and local-save failure must not report success.

```bash
cmind update
cmind update --ai claude
cmind update --no-mcp
cmind update --no-upgrade
cmind update --no-upgrade --no-git-hooks
cmind update --git-hooks
```

### Options

| Option | Description |
| ------ | ----------- |
| `--ai <agent>` | Explicitly save `copilot` or `claude` locally after success and refresh its integration; omission refreshes integrations without changing local consent |
| `--script <type>` | Script type: `sh` (POSIX). `ps` (PowerShell) is not yet supported and will be added in a future release. |
| `--no-upgrade` | Skip the default-on CLI self-upgrade and only sync workspace files. |
| `--no-mcp` | Skip MCP server configuration |
| `--no-copilot-cli-mcp` | For Copilot, skip registration in `~/.copilot/mcp-config.json`; retain workspace MCP setup unless `--no-mcp` is also set. Does not remove existing registrations. |
| `--no-cmind-git` | Skip backfilling a missing private snapshot repo at `~/.cmind/workspaces/<workspace-id>/.git/`; leaves existing snapshot repos unchanged and does not control project Git hooks. |
| `--git-hooks/--no-git-hooks` | Default OFF: remove only recognized CoderMind-managed Git hook blocks, preserving other content. `--git-hooks` installs deterministic sync only; repeat on each init/update to retain it. |
| `--debug` | Show verbose diagnostic output |

### Git hooks and workspace migration

- Default / `--no-git-hooks`: remove recognized managed `pre-commit`, `post-commit`, and `post-merge` blocks, including recognized legacy bodies; preserve unrelated or unrecognized content.
- `--git-hooks`: install foreground, deterministic `post-commit` / `post-merge` sync only; remove retired managed `pre-commit` blocks. Post-commit background LLM updates are removed entirely, not an optional mode.
- LLM-driven graph updates require explicit `cmind script update_graphs.py update-rpg --json` or `/cmind.update_rpg`, outside hook context.
- Git hook cleanup/installation failures abort init/update before saving a new local selection or initial encoding. Installation of Claude `SessionStart` and Copilot / VS Code `folderOpen` status-only integrations remains best-effort; `--no-git-hooks` does not disable them.

Upgrade the CLI, then reconcile **each** old workspace. To establish local consent, use `cmind update --ai claude --no-upgrade --no-git-hooks` (or `copilot`); omit `--ai` only to preserve existing local state, including absence. A wheel upgrade alone does not clean legacy inline-script hooks. Valid provider-only hints may stay tracked; no `git rm` is required. Invalid raw commands fail preflight even with environment overrides: manually remove them or replace them with one valid `recommended_provider`, then explicitly select a provider. Review unrecognized hooks, including `core.hooksPath`, without deleting unrelated content. See the [upgrade checklist](configuration.md#updating-an-existing-codermind-project).

### Auto-upgrade behaviour

Since the global-install layout, `cmind update` performs a **best-effort silent self-upgrade by default** when the install source is safe to refresh (git+URL or PyPI). After upgrading the CLI it re-executes itself once to continue the workspace sync with the new code. Editable installs, local-file installs, and unknown sources are skipped silently.

- Pass `--no-upgrade` to skip the upgrade entirely (useful for offline or pinned environments).
- A loop guard environment variable (`CMIND_UPGRADE_DONE`) is set across the re-exec to guarantee at most one upgrade attempt per invocation.

### Provisioning sources

`cmind init` and `cmind update` provision exclusively
from the **packaged assets bundle** shipped inside the installed
`cmind-cli` wheel (under `cmind_cli/core_pack/`). No network access
is required for this template-copying step, not a guarantee that the
entire init/update workflow is offline: encoding may call a provider,
and update's default self-upgrade may fetch packages.

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

This availability check does not replace execution-policy validation: a detected Windows wrapper alone is insufficient. See the [native/npm requirements](configuration.md#executable-and-permission-policy). It also does not establish user-local consent.

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
cmind script update_graphs.py status
cmind script update_graphs.py update-rpg --json
cmind script --list
cmind script --where mcp_server.py
```

`status` displays read-only graph status. `update-rpg` explicitly requests an LLM-driven update against `HEAD~1`; it requires an existing RPG and at least two commits, a trusted runtime provider selection, and normal provider permissions. Repository recommendations alone are insufficient. Neither enabling Git sync hooks nor opening a session requests this update.

`HEAD~1` must resolve locally, including in shallow clones. The comparison uses the current working tree, so uncommitted changes are allowed; it is not limited to committed changes. It does not automatically select the graph's last synced commit. If the graph spans older history or another branch, ask the user whether to perform a full `/cmind.encode` instead; do not rebuild or modify history automatically.

The slash-command templates installed by `cmind init` (in
`.claude/commands/` or `.github/agents/`) all use `cmind script …`
under the hood, so AI agents invoke the pipeline through the same
contract.

A companion console script, `cmind-mcp`, is the MCP server entry
point and is what `.mcp.json` / `.vscode/mcp.json` register as the
`rpg-tools` command — no absolute paths in the config, no per-machine
edits.
