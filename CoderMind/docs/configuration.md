# Configuration

This document covers AI provider selection, assistant permissions, MCP registration, hooks, and workspace upgrades.

> **Data paths.** References below such as `.cmind/data/rpg.json` and `.cmind/logs/...` are logical names. Runtime files actually live under `~/.cmind/workspaces/<workspace-id>/{data,logs}/` so they stay outside your git repo, where `<workspace-id>` is the slug-based workspace identifier used by the home-side store (with an optional `-<hash6>` suffix when needed). Reports stay in the workspace at `<workspace>/.cmind/reports/`. The MCP server, hooks, and pipeline scripts all resolve the home-dir location automatically from the workspace root. Run `cmind version` from inside the workspace to see the resolved Data / Logs paths; see [project-structure.md](project-structure.md) for the full layout.

## AI Assistant CLI Requirements

CoderMind slash commands are executed by an AI coding agent. Before running `cmind init`, install and authenticate at least one supported AI assistant CLI.

CLI integration choices (not a claim that live npm releases were tested):

| Agent | `--ai` value | Generated configuration | Requirement |
| ----- | ------------ | ----------------------- | ----------- |
| GitHub Copilot | `copilot` | `.github/`, `.vscode/` | Copilot CLI available and authenticated |
| Claude Code | `claude` | `.claude/` | Claude Code CLI available and authenticated |

Use `cmind check` to check local tool availability. AI execution also applies the stricter [executable policy](#executable-and-permission-policy) below.

```bash
cmind check
```

If the selected AI assistant is not found, install and authenticate it, then rerun `cmind init` or `cmind update`.

## Workspace Configuration (`.cmind/config.toml`)

The workspace's `.cmind/config.toml` is a **workspace-discovery marker and provider recommendation only**, never execution authority. Keeping this provider-only hint tracked is intentional; no `git rm` is needed. It cannot select an executable, authorize AI calls, or grant auto-approval.

```toml
# .cmind/config.toml
[cmind]
recommended_provider = "claude"
```

`cmind init` and `cmind update` create this file if missing and preserve valid existing files byte-for-byte. Legacy `ai_provider` and exact built-in `ai_cli_cmd` values remain valid **hints**, never authority. Only one of the three keys is allowed. Changing a recommendation does not change execution consent; an explicit user choice is independent of the hint and is never imported from it.

### Resolution priority

AI calls use the closed policy in [../scripts/common/ai_cli_policy.py](../scripts/common/ai_cli_policy.py):

| # | Source | Accepted value |
| - | ------ | -------------- |
| P1 | Explicit, trusted `LLMClient(tool="...")` constructor argument | Exact legacy command from the table below; not copied from repository metadata |
| P2 | `CMIND_AI_PROVIDER` **or** legacy `CMIND_AI_CLI_CMD` | Provider enum for the former; exact legacy command for the latter; mutually exclusive |
| P3 | User-local workspace selection | Validated provider and workspace identity from the record described below |

Every consulted configuration must be valid: unknown, non-string, empty, malformed, unreadable, or conflicting settings fail closed, with **no fallthrough**. Workspace configuration is always validated before resolution, **even with a constructor or environment override**. Init/update also preflight it before provisioning, hook migration, or self-upgrade. Nonempty legacy release-baked commands are validated but **never authorize execution or provide a fallback**.

A higher-priority authority does not read or rewrite lower-priority local state. If no authority is present, `LLMClient.generate()` reports a configuration error despite any workspace or baked recommendation.

### User-local execution selection

Explicit choices are saved under `Path.home().resolve() / ".cmind" / "execution" / <full-sha256> / "selection.json"`. The hash is the full SHA-256 of the UTF-8 canonical workspace identity: resolve symlinks, normalize Windows extended-path prefixes, then apply `os.path.normcase`. It is not the slug used by the RPG data store.

The JSON record has exactly three fields: `schema_version` (integer `1`), `workspace` (that exact canonical identity), and `ai_provider` (a valid provider enum). Invalid schemas or mismatched identities fail closed when read; redirected/symlinked state paths and storage inside the workspace are rejected. This record is separate from RPG data and provisioning metadata; copying those files cannot authorize execution.

A clone at another path, a moved workspace, or a new user needs an explicit selection (or trusted process environment). There is no automatic import from recommendations, detected integrations, or metadata. Use `cmind init --here --ai claude` or, for an existing CoderMind workspace, `cmind update --ai claude`; choose `copilot` instead if intended.

### Provider enum and legacy compatibility

The policy accepts exactly these 11 historical providers, case-sensitively. Legacy command values must match the right column exactly: no executable paths, extra flags, quote characters inside the value, leading/trailing or altered whitespace, or shell syntax. The fixed arguments and single spaces shown below are the only accepted legacy forms.

| Provider enum (`recommended_provider`, `ai_provider`, `CMIND_AI_PROVIDER`) | Exact legacy form (`ai_cli_cmd`, `CMIND_AI_CLI_CMD`, `tool`, baked command) |
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

Only `copilot` and `claude` are exposed by CLI `--ai` selection and have Windows npm adapters. The other nine retain historical native-executable/scaffold integration only, not verified integration support or npm adapters. Enum acceptance and mocked adapter coverage are not live-release compatibility guarantees.

### Executable and permission policy

- CoderMind builds argv internally and invokes the provider without a shell. It resolves an absolute executable using only absolute `PATH` entries outside both the workspace and current working directory, checking resolved symlink targets too. Empty and relative entries are ignored; installed tools and those external `PATH` directories remain trusted.
- On Windows, direct provider `.exe` files take priority across eligible `PATH` directories. The fallback adapter recognizes only the Claude/Copilot npm layouts below, under an external search directory's `node_modules`. It never reads, parses, or executes `.cmd`, `.bat`, or `.ps1` wrappers, and does not accept arbitrary user scripts or invoke `npm`/`npx`.
- The adapter requires an exact manifest `name` and a `bin` object containing exactly the provider key and an allowlisted entry. Package, manifest, entry, and interpreter paths must resolve canonically outside the workspace and current directory; manifest/entry links must stay inside the package. JS entries require an external `node.exe`, returned as an absolute path from the same eligible search directories. Native Claude needs no Node.
- The default Claude `--dangerously-skip-permissions` and Copilot `--allow-all` arguments have been removed, with **no new opt-in bypass**. Standard provider permissions apply: a tool action may require user approval or halt a noninteractive workflow. Review approvals through the provider's normal controls; provider selection grants none.

| Exact npm package name | Exact `bin` key | Allowed entry and public manifest evidence |
| --- | --- | --- |
| `@anthropic-ai/claude-code` | `claude` | `bin/claude.exe` ([2.1.278](https://unpkg.com/@anthropic-ai/claude-code@2.1.278/package.json)); `cli.js` ([2.0.0](https://unpkg.com/@anthropic-ai/claude-code@2.0.0/package.json)) |
| `@github/copilot` | `copilot` | `index.js` ([0.0.369](https://unpkg.com/@github/copilot@0.0.369/package.json)); `npm-loader.js` ([1.0.87](https://unpkg.com/@github/copilot@1.0.87/package.json)) |

These versioned manifests document layouts, not version pins, compatibility guarantees, or locally tested installations. The [adapter](../scripts/common/windows_ai_cli.py) checks filesystem paths and metadata, **not signatures or publisher authenticity**. External installations, Node, and loader dependencies remain user-trusted; these checks do not secure a tampered installation.

### Trusted CI process selection

For an already initialized workspace (or a checkout with its valid workspace marker), a trusted POSIX CI step can select a provider for that process and its children only:

```bash
CMIND_AI_PROVIDER=claude cmind script rpg_encoder/run_encode.py --json
```

This changes no system setting and does not require persisting a local selection. Normal pipeline workspace-marker lookup still applies; this example does not bootstrap an uninitialized checkout. A noninteractive `cmind init` still requires `--ai`, even with the environment set. Invalid repository hints still fail preflight, and normal executable and provider-permission rules still apply.

Only a trusted operator/workflow may supply execution authority. Running untrusted upstream or pull-request workflows is not an authorization bypass or sandbox; the provider policy does not make arbitrary repository code safe.

## Initialization Options

### AI assistant selection

```bash
cmind init my-project --ai claude
cmind init my-project --ai copilot
```

`--ai` selects both the integration and the provider to save locally. If omitted in an interactive terminal, CoderMind asks for that explicit choice, independently of the recommendation. Non-TTY init without `--ai` fails early, before provisioning; environment overrides or detected integrations do not satisfy this init requirement.

The local choice is saved atomically only after successful provisioning and hook reconciliation, before the success message or optional initial encode. A hook or local-save error aborts init/update without reporting success; init does not proceed to encoding. A failed atomic replacement preserves the previous selection.

### Script type

```bash
cmind init my-project --script sh
```

`sh` installs POSIX shell-oriented command snippets. `ps` (PowerShell) is not yet supported; this is separate from Windows native/npm executable resolution for AI calls.

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

## Assistant Permissions and Scope

Provider configuration is not a permission grant. Claude's status integration separately adds a project-scoped `mcp__rpg-tools` allow rule for the four read-only graph queries; Copilot / VS Code manages MCP approvals through its own controls. Review generated and existing assistant permissions separately: removing CLI bypass flags does not revoke settings you already granted.

- `--no-mcp` skips MCP registration, not status integrations or blanket removal of existing permissions.
- Copilot initialization/update also registers MCP in the user's Copilot CLI configuration by default. Use `--no-copilot-cli-mcp` to skip that registration while retaining workspace MCP setup.

## Git Hooks and Incremental Updates

Git hooks are **OFF by default** for both `cmind init` and `cmind update`.

- Omit the option or pass `--no-git-hooks` to remove only recognized CoderMind-managed `pre-commit`, `post-commit`, and `post-merge` blocks, including recognized legacy bodies. Other hook content is preserved; unfamiliar or incomplete old bodies require manual review.
- If a hook cannot be read or cleaned, reconciliation still attempts the other hooks, then fails explicitly. Malformed bytes are preserved; init/update must not report successful migration or proceed to encoding after that failure.
- Pass `--git-hooks` on **each** init/update invocation to install deterministic, foreground `post-commit` / `post-merge` sync only. Retired managed `pre-commit` blocks are removed in either mode. Sync writes `hooks.log` under the home-side Logs directory shown by `cmind version`.
- The post-commit background LLM update has been removed entirely. Git hooks do not launch AI or background workers; sync does not replace an LLM-driven feature graph update.

```bash
cmind init --here --ai claude --no-git-hooks
cmind update --git-hooks
cmind update --no-git-hooks
```

For LLM-driven updates, explicitly invoke the following from the workspace, outside Git hooks, or use `/cmind.update_rpg`:

```bash
cmind script update_graphs.py update-rpg --json
```

This requires an existing RPG and `HEAD~1` (at least two commits). If the graph is missing or needs a full rebuild, use `/cmind.encode`.

### Session and workspace startup status

Claude's `SessionStart` integration and Copilot / VS Code's `folderOpen` task remain **status-only**: they run `cmind script update_graphs.py status` to display graph status and guidance, without an LLM update. They are not Git hooks and are not disabled by `--no-git-hooks`.

## Updating an Existing CoderMind Project

Upgrade the installed CLI **and reconcile every existing workspace**. Upgrading the wheel alone does not rewrite old hook files, especially legacy inline-script hooks.

1. Upgrade `cmind-cli` using your installation method (for example, `uv tool upgrade cmind-cli`). Verify the installed CLI with `cmind version`; do not rely on a workspace update's best-effort self-upgrade alone.
2. Review each workspace's `.cmind/config.toml`. Valid provider-only hints, including exact legacy `ai_provider`/`ai_cli_cmd`, may remain tracked and are preserved; no `git rm` is needed. Old raw commands are never executed. Manually remove unsupported command settings or replace them with one valid `recommended_provider`. **Invalid configuration fails preflight before provisioning, self-upgrade, or hook migration**, even with `--ai`, `--force`, or an environment override.
3. From each workspace root, explicitly choose the intended provider with `cmind update --ai claude --no-upgrade --no-git-hooks` (or `--ai copilot`); for first-time setup use `cmind init --here --ai ...`. This refreshes integrations, reconciles hooks, then saves the explicit choice locally. Use `--git-hooks` instead only if deterministic sync is wanted. No choice is imported from workspace or RPG metadata.
4. Review remaining hooks in the active hooks directory, including any `core.hooksPath` override. Manually migrate unrecognized legacy inline-script bodies; preserve unrelated user/team hooks. Do not assume a wheel upgrade cleaned every workspace.
5. Request feature graph updates explicitly with `cmind script update_graphs.py update-rpg --json` or `/cmind.update_rpg`, allowing for normal provider approval requirements.

Without `--ai`, `cmind update` refreshes integrations only and preserves the local selection byte-for-byte (or leaves it absent). It prefers an existing supported local provider for templates, then detects integration folders or prompts interactively for an integration only; none of these paths saves new consent. Malformed local state fails when read; retry with an explicit choice after checking local-store permissions/identity. `--ai` saves the selected provider only after hooks succeed and leaves valid workspace hints unchanged. `--no-mcp` skips MCP registration, not hook reconciliation.

### Repository regression checks

For maintainers, [../tests/run_security_tests.py](../tests/run_security_tests.py) includes expanded mocked policy, user-local state, Windows npm-layout/path, and hook-migration regressions. It uses an existing Python environment with dependencies already installed and redirects home/config/temp locations into a disposable repository directory. Its audit guard forbids real process launches, network access, and writes outside that directory. No dependency installation or machine-setting changes are needed. The guard prevents accidental test side effects, not hostile Python code.

The security CI workflow is configured for disposable Linux and Windows runners: source and freshly installed wheel tests, module-origin/byte checks, and verification of all eleven provider Release ZIPs. Stable and pre-release publishing depend on it. The local runner does not install packages or build releases; `--installed` is for CI after a fresh wheel installation. Neither this runner nor that workflow installs or executes real provider npm releases; mocked fixtures and artifact checks are not live compatibility tests or evidence of a completed CI run.

## Troubleshooting

### AI assistant CLI not found

Run:

```bash
cmind check
```

Install and authenticate the intended CLI outside the workspace, with an absolute `PATH` directory. Availability detection is not execution-policy validation. On Windows, wrappers alone are insufficient: review the [direct-executable/npm adapter requirements](#executable-and-permission-policy), rather than adding paths or flags to configuration. A tracked recommendation alone also does not authorize execution; make an explicit local or trusted process selection.

### MCP tools say `rpg_unavailable`

The MCP server is configured, but `.cmind/data/rpg.json` has not been created yet. Run:

```text
/cmind.encode
```

### Incremental update failed

Inspect the foreground error and use `cmind version` to locate the actual Logs directory. Check for invalid provider configuration, unsupported launchers, or required provider approval before retrying explicitly:

```text
/cmind.update_rpg
```

If the graph is corrupted or too stale, run `/cmind.encode` for a full rebuild.

### Template download hits rate limits or private repo access errors

`cmind init` and `cmind update` do not fetch templates
from GitHub releases — templates are bundled inside the installed
`cmind-cli` wheel, so this class of error should no longer occur during
provisioning.  To pick up newer templates, upgrade the CLI itself:

```bash
uv tool upgrade cmind-cli
```

`cmind update` attempts a best-effort self-upgrade by default; pass
`--no-upgrade` to opt out. Existing workspaces still need the reconciliation
described in the [upgrade checklist](#updating-an-existing-codermind-project).
