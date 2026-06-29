---
name: cmind.update_rpg
description: Manually trigger an incremental RPG update (fallback for when the post-commit hook didn't run)
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

## Outline

After every `git commit`, the **post-commit hook** automatically runs an
incremental RPG update in the background — so under normal use **you do
not need to run this command**.

This slash command is a **manual fallback** for the few cases where the
automatic update didn't happen, e.g.:

* You committed with `git commit --no-verify` (skipping hooks).
* The background hook errored out (network blip, LLM timeout) — run
  `cmind version` to locate the workspace's logs directory and tail
  the latest `update_rpg.log` there.
* You want to force a fresh update synchronously and see the result
  immediately instead of waiting for the async hook.

It compares the workspace against `HEAD~1` (same baseline the hook
uses) and runs the LLM-driven feature graph diff + dep_graph rebuild.

### Step 1: Pre-Check

Run the check script:

```bash
cmind script rpg_encoder/check_encode.py --json
```

Inspect the `type` field in the JSON output:

* **`error`** → display `message` and stop. The `rpg.json` file is
  corrupt; the user may need to delete it and rerun `/cmind.encode`.
* **`init`** → no `rpg.json` yet. Tell the user to run `/cmind.encode`
  first to create the baseline graph, then terminate.
* **`update`** → display `result.stats.repo_name`, Feature graph
  `node_count` / `edge_count`, and Dependency graph `dep_nodes` /
  `dep_edges`, then proceed to Step 2.

Also verify there is at least one previous commit (the update needs
`HEAD~1` as baseline):

```bash
git rev-list --count HEAD
```

If the count is `< 2`, tell the user there is no previous commit to
diff against, and suggest running `/cmind.encode` instead. Terminate.

### Step 2: Run the Update

Invoke the same script the post-commit hook uses. It creates and cleans
up its own temporary worktree internally — **you do not need to manage
`git worktree` manually**.

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

**If `status` is `"error"`**:

* Show the `error` field.
* Tell the user to run `cmind version` to locate the logs directory
  and inspect `update_rpg.log` for the full trace.
* Common causes: LLM API misconfigured, network failure, dirty worktree
  blocking `git worktree add`.

### Step 4: Next Steps (optional)

```text
Tips:
  - The post-commit hook runs this same update automatically after
    every commit; you only need to invoke this command when the
    automatic update failed or was skipped.
  - /cmind.encode — Run a full re-encode if the RPG seems stale or
    has drifted significantly from the codebase.
  - The latest `update_rpg.log` (path shown by `cmind version`) keeps
    the most recent run output.
```
