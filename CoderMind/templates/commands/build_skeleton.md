---
description: Build repository file skeleton from component architecture
name: cmind.build_skeleton
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

## **Outline**

The text entered by the user after `/cmind.build_skeleton` **is the adjustment suggestion**.
Unless it is explicitly empty, you may assume it is always available as `$ARGUMENTS`.
**Do not** ask the user to repeat the input.

### Step 1: Pre-check

Run the script `cmind script check_skeleton.py` to verify the current state.

1. Inspect the `type` field in the output:

   * `error` → Display the error message and stop. Instruct user to run `/cmind.feature_refactor` first. Terminate this command.
   * `init` → Proceed to Step 2.
   * `warning` → Display the following prompt and wait for user confirmation:

     ```text
     The skeleton file exists but has feature mismatches:
     - Files: <total_files>
     - Features in skeleton: <skeleton_feature_count>
     - Features in input: <input_feature_count>
     - Mismatches: <warning_count>
     
     Missing in skeleton: <list first 5 from in_input_not_skeleton>
     Extra in skeleton: <list first 5 from in_skeleton_not_input>
     
     Do you want to regenerate?
     * Y: Regenerate the skeleton
     * N: Cancel and keep existing
     ```

     If user choose to regenerate, proceed to Step 2; if to cancel and keep existing, jump to Step 4.
   * `update` → Display the following prompt and wait for user confirmation:

     ```text
     The skeleton file already exists and is consistent.
     ```

     Then proceed to Step 4.

### Step 2: Build Skeleton

1. Display the following prompt and wait for user confirmation:

   ```text
   Description: Run the script `cmind script build_skeleton.py` to:
     - Step 1: Design directory structure for components
     - Step 2: Assign features to Python files
   
   Select max iterations for file assignment:
     - [Y] → use default (10)
     - [Number] → specify a custom iteration count
   ```

2. Execute the following command with the selected iteration count:

   ```bash
   cmind script build_skeleton.py --max-iterations <default_or_user_defined>
   ```

   The script writes a structured log automatically;
   stdout carries the human-readable summary you need below.

3. From the captured stdout, find the section containing:

   ```text
   SKELETON BUILDING COMPLETE
   ```

   Display the summary information in a Markdown table format showing:
     * Total components
     * Total features
     * Total files created
     * File assignments (path and feature count)

### Step 3: Validation

Run the validation script:

```bash
cmind script check_skeleton.py --verbose
```

Display the validation results to the user:

* If `output_valid` is `true` and `cross_validation.is_consistent` is `true`: Report success with full consistency
* If `output_valid` is `true` but `cross_validation.is_consistent` is `false`:
  * Display warning about feature mismatches
  * Show `cross_validation.in_input_not_skeleton` (features missing in skeleton)
  * Show `cross_validation.in_skeleton_not_input` (extra features in skeleton)
  * Suggest re-running if mismatches are significant
* If `output_valid` is `false`: Display `validation_errors` and suggest re-running

### Step 4: Completion & Handoff

Run the summary script to generate a formatted report and save to file:

```bash
cmind script summary_skeleton.py
```

The summary (including directory structure, component paths, and statistics) is
printed on stdout by `summary_skeleton.py`; the script also persists it
to the workspace's state directory for later inspection.

Then prompt the user:

```text
Skeleton has been generated.

Outputs (managed by the script; consumed by downstream stages):
  skeleton.json          - Skeleton data (JSON format)
  skeleton_summary.txt   - Human-readable summary

To proceed with data flow design, run:
  /cmind.build_data_flow

To regenerate with adjustments, run:
  /cmind.build_skeleton <adjustment instructions>
```
