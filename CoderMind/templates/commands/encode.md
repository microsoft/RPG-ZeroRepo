---
name: cmind.encode
description: Encode a repository into an RPG (Repository Program Graph)
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

## Outline

Encode the current repository into an RPG structure. The RPG captures the
codebase's functional architecture as a graph of nodes (features, modules,
code entities) and edges (dependencies, containment).

### Step 1: Pre-Check

Run the check script to determine the current encode state:

```bash
cmind script rpg_encoder/check_encode.py --json
```

Inspect the `type` field in the output:

**If type is "error"**:

* Display the error message and the `rpg_file` path. Explain that the
  existing RPG file appears corrupted and blocks encoding.
* Ask the user whether to delete the damaged RPG file and run a full encode now:

  ```text
  The RPG file appears to be corrupted:
    <rpg_file>

  Delete this file and run a full /cmind.encode now?
  - Y: Delete <rpg_file> and run full encode
  - N: Quit without deleting anything
  ```

* If user chooses **Y**: delete exactly the returned `rpg_file` path with
  `rm -- "<rpg_file>"`, then proceed to Step 2 in this same command run.
  Do not delete directories, globs, or any other candidate RPG paths.
* If user chooses **N**: terminate without deleting anything.
* If deletion fails: display the deletion error and stop.

**If type is "init"**:

* No RPG file exists yet. Proceed to Step 2 for a full encode.

**If type is "update"**:

* An RPG file already exists. Display the current stats and ask:

  ```text
  An RPG already exists for this repository:
    Nodes: <node_count>
    Edges: <edge_count>
    Repo:  <repo_name>

  Choose an action:
  - R: Full re-encode (rebuild RPG from scratch)
  - U: Incremental update (use /cmind.update_rpg instead)
  - Q: Quit
  ```

* If user chooses **R**: proceed to Step 2.
* If user chooses **U**: instruct user to run `/cmind.update_rpg` instead. Terminate.
* If user chooses **Q**: terminate.

### Step 2: Full Encode

Run the full encode script:

```bash
cmind script rpg_encoder/run_encode.py --json
```

This may take several minutes depending on repository size and LLM response times.
The script prints a JSON summary on stdout and writes a structured
log automatically.
Inspect the JSON `status` field to decide next steps.

**If status is "success"**:

* Display the encoding statistics:

  ```text
  RPG encoding complete!
    Repository: <repo_name>
    Nodes: <node_count>
    Edges: <edge_count>
    Functional areas: <functional_areas>
    Saved to: <output_path>
  ```

* Proceed to Step 3.

**If status is "error"**:

* Display the error message.
* Suggest checking LLM API key configuration and repository structure.

### Step 3: Next Steps

Display suggestions for what the user can do next:

```text
Next steps:
  - /cmind.update_rpg  — Incrementally update after code changes
  - The MCP server exposes search_rpg and explore_rpg tools
    for AI agents to query the RPG interactively.
  - RPG data is saved at .cmind/data/rpg.json
```
