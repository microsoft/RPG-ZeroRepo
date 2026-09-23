---
name: cmind.update_rpg
description: Explicitly run an AI-driven incremental RPG update in the foreground
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

## Outline

This command explicitly requests an LLM-driven feature graph diff and
dependency graph rebuild in the foreground. It is **not a Git hook
fallback**: Git hooks are OFF by default, and `--git-hooks` installs only
deterministic foreground sync, never AI calls or background workers.

The comparison is the **current working tree against `HEAD~1`**.
Uncommitted changes are allowed; do not require a clean tree or commit/stash
the user's changes. `HEAD~1` is fixed, not the graph's last synced commit.
If the graph is stale across several commits or a branch switch, ask the
user whether a full `/cmind.encode` is more appropriate; do not rebuild
automatically. An existing usable RPG and locally available parent commit
are required.

### Execution prerequisites and stop rule

AI calls require a trusted explicit constructor choice, trusted process
environment selection, or valid user-local selection. Tracked
`recommended_provider` and valid legacy hints are not execution authority.
Normal provider approvals apply. On configuration, authentication, access,
or approval blocks at any step, surface the exact error and any reported
diagnostic artifact path, preserve artifacts, and pause for the user.
Do not retry, run init/update, rewrite configuration/local selection, or
grant trust/permission overrides to recover. Continue only after explicit
user resolution and authorization to retry.

### Step 1: Pre-Check

Run the check script:

```bash
cmind script rpg_encoder/check_encode.py --json
```

Inspect the `type` field in the JSON output:

* **`error`** → display the exact `message` and any reported artifact
  path, preserve the graph/reports, and stop. Do not assume every error
  means corruption or delete/rebuild the graph automatically; ask the user
  how to proceed.
* **`init`** → no `rpg.json` yet. Explain that `/cmind.encode` can create
  the baseline graph and ask the user to request it, then terminate.
* **`update`** → display `result.stats.repo_name`, Feature graph
  `node_count` / `edge_count`, and Dependency graph `dep_nodes` /
  `dep_edges`, then proceed to Step 2.

Also verify the parent commit is available locally (at least two commits;
shallow history may not contain the needed parent):

```bash
git rev-parse --verify HEAD~1
```

If this fails, show the exact error and explain that the required baseline
is unavailable. Suggest `/cmind.encode` as a user-chosen alternative and
stop; do not fetch history, make commits, or run encoding automatically.

### Step 2: Run the Update

Explicitly invoke the AI update script, independently of Git hooks. It
creates and cleans up its own temporary worktree internally — **you do
not need to manage `git worktree` manually**.

```bash
cmind script update_graphs.py update-rpg --json
```

The full JSON result is printed on stdout (single `{...}` block). The
script also writes a structured log automatically; you do
not need to redirect output.

### Step 3: Display Result

**If `status` is `"success"`** (top-level field of the JSON):

```text
RPG update complete!
  Repository: <repo_name>
  Previous ref: <prev_ref>
  Feature graph Nodes: <node_count> (delta: <nodes_delta>)
  Feature graph Edges: <edge_count> (delta: <edges_delta>)
  Dependency graph Nodes: <dep_nodes> (delta: <dep_nodes_delta>)
  Dependency graph Edges: <dep_edges> (delta: <dep_edges_delta>)
  Aligned to dep_graph: <aligned>
  Functional areas: <functional_areas>
  Saved to: <output_path>
```

**If `status` is `"error"`, an `error` field is present, or the process exits
non-zero**:

* Show the exact `error` / stderr and any reported diagnostic artifact path.
  Preserve the existing graph and reports; stop rather than claiming success.
* Tell the user to run `cmind version` to locate the logs directory
  and inspect `update_rpg.log` for the full trace.
* Configuration, authentication, executable/access, or provider approval
  blocks follow the stop rule above, not an automatic retry or repair path.
* Other possible causes include network failures or errors creating the
  temporary worktree. Uncommitted workspace changes alone are not a blocker.

### Step 4: Next Steps (optional)

```text
Tips:
  - /cmind.update_rpg explicitly requests an AI update; Git sync
    hooks and session startup do not request it for you.
  - /cmind.encode — Choose a full re-encode if the graph needs a
    new baseline; rebuilding requires the user's explicit decision.
  - The latest `update_rpg.log` (path shown by `cmind version`) keeps
    the most recent run output.
```
