---
description: Edit the feature tree, nodes can be deleted, modified, added, or expanded as needed.
name: cmind.feature_edit
---


## User Input

```text
$ARGUMENTS
```

**Important:** You **MUST** consider the user input before proceeding.
If the input is empty, respond immediately:

> **Edit instructions required**
>
> Please provide a description of how you want to edit the feature tree. Examples:
>
> - "Delete the 'cloud integration' component"
> - "Add error handling features under 'cli operations'"
> - "Expand the 'security' component with more encryption options"
> - "Merge 'analytics telemetry' into 'monitoring observability'"
>
> Usage: `/cmind.feature_edit <your edit instructions>`

## Workflow

The text typed by the user after `/cmind.feature_edit` **is the edit instruction**. You can assume it is always available as `$ARGUMENTS`, unless explicitly empty. Do **not** ask the user to repeat it otherwise.

**File:** `.cmind/data/feature_tree.json` (both input and output)

**Working Directory**: All relative paths are based on the project root directory.

### Step 1: Run Pre-check Script

Execute from repository root:

```bash
cmind script feature_edit_validation.py --edit_instruction "$ARGUMENTS"
```

**Important:** If `$ARGUMENTS` contains a double quote (`"`), it MUST be escaped before being passed to the script.

Inspect the `type` field in the output:

1. **If `type` is `"error"`**: Display the error message and stop execution.

   Based on `error_code`:

   - `file_not_found`:

     ```markdown
     > **Error**: No feature tree found.
     > 
     > Please run `/cmind.feature_construct` first to build the feature tree.
     ```

   - `field_empty` or `field_missing`:

     ```markdown
     > **Error**: The feature tree exists but is incomplete.
     > 
     > Please run `/cmind.feature_construct` to rebuild a valid feature tree.
     ```

2. **If `type` is `"ready"`**: Proceed to Step 2.

### Step 2: Confirm Execution

**Before executing the edit script, the agent must wait for user input and must not proceed in advance.**

Display the following prompt and wait for user confirmation:

```markdown
The script `cmind script feature_edit.py` will be executed to edit the feature tree based on your instructions.

**File:** `.cmind/data/feature_tree.json`

**Edit Instructions:** 
> $ARGUMENTS

**Process:**
1. **Plan** - Generate edit plan based on your instructions
2. **Execute** - Apply the planned changes
3. **Review** - Verify changes and auto-fix if needed (up to 3 review rounds)

Please confirm to proceed:

* Input: **Y** → Execute the edit script
* Input: **N** → Cancel and exit
```

### Step 3: Execute Edit Script

Execute the following command:

```bash
cmind script feature_edit.py
```

The script writes a structured log automatically; stdout
carries the summary you need below.

### Step 4: Summarize Results

After the script completes:

1. Analyze and summarize the information printed during script execution.

2. Present the edit summary in a clear format showing:
   - Operations performed (DELETE, MODIFY, ADD, MOVE, etc.)
   - Number of features affected
   - Before/after feature count comparison
   - Review status and any fixes applied
   - Final summary from LLM

3. Prompt the user to review the output file:

   ```markdown
   **Edit Complete**
   
   The edited feature tree has been saved to `.cmind/data/feature_tree.json`.
   
   Please review the changes to verify they match your expectations.
   
   If further adjustments are needed, run:

   ```text
   /cmind.feature_edit <additional edit instructions>
   ```

   If you are satisfied with the feature tree and ready to proceed to the **next step**, run:

   ```text
   /cmind.plan
   ```
