---
description: Build inter-component data flow graph (DAG)
name: cmind.build_data_flow
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

All the bash command timeout is set to 1 hour.

## **Outline**

The text entered by the user after `/cmind.build_data_flow` **is the adjustment suggestion**.
Unless it is explicitly empty, you may assume it is always available as `$ARGUMENTS`.
**Do not** ask the user to repeat the input.

### Step 1: Pre-check

Run the script `cmind script check_data_flow.py` to verify the current state.

1. Inspect the `type` field in the output:

   * `error` → Display the error message and stop. Instruct user to fix the error or regenerate. Terminate this command.
   * `init` → Proceed to Step 2.
   * `warning` → Display the following prompt and wait for user confirmation:

     ```text
     The data flow file exists but has component mismatches:
     - Skeleton components: <skeleton_components>
     - Data flow components: <data_flow_components>
     - Matched: <matched>
     
     Missing in data flow: <list first 5 from in_skeleton_only>
     Extra in data flow: <list first 5 from in_data_flow_only>
     
     Do you want to regenerate?
     * Y: Regenerate the data flow
     * N: Cancel and keep existing
     ```

     If user choose to regenerate, proceed to Step 2; if to cancel and keep existing, jump to Step 4.
   * `update` → Display the following prompt and wait for user confirmation:

     ```text
     The data flow file already exists and is valid:
     - Data Flow Edges: <edge_count>
     - Components: <component_count>
     - Subtree Order: <subtree_order>
     
     Do you want to regenerate?
     * Y: Regenerate the data flow
     * N: Cancel and keep existing
     ```

     If user choose to regenerate, proceed to Step 2; if to cancel and keep existing, jump to Step 4.

### Step 2: Build Data Flow

> This command may run for a long time depending on project size.
> **Set your terminal timeout to at least 60 minutes** before running.
> Do **NOT** interrupt or re-run this command.

1. Display the following prompt and wait for user confirmation:

   ```text
   Description: Run the script `cmind script build_data_flow.py` to:
     - Design inter-component data flow as a DAG
     - Generate subtree processing order
   
   Select max iterations for valid design:
     - [Y] → use default (5)
     - [Number] → specify a custom iteration count
   ```

2. Execute the following command with the selected iteration count:

   ```bash
   cmind script build_data_flow.py --max-iterations <default_or_user_defined>
   ```

   The script writes a structured log automatically;
   stdout carries the summary you need below.

3. Upon successful completion, display:

   ```text
   ✓ Data flow built successfully
   
   Summary:
   - Data Flow Edges: <edge_count>
   - Components: <component_count>
   - Subtree Order: <show subtree order>
   
   Output: .cmind/data/data_flow.json
   ```

### Step 3: Validation

Run the validation script:

```bash
cmind script check_data_flow.py --verbose
```

Display the validation results to the user:

* If `output_valid` is `true` and `cross_validation.is_consistent` is `true`: Report success with full consistency
* If `output_valid` is `true` but `cross_validation.is_consistent` is `false`:
  * Display warning about component mismatches
  * Show `cross_validation.in_skeleton_only` (components missing in data flow)
  * Show `cross_validation.in_data_flow_only` (extra components in data flow)
  * Suggest re-running if mismatches are significant
* If `output_valid` is `false`: Display `validation_errors` and suggest re-running

### Step 4: Completion & Handoff

**If data flow was built (Step 2 was executed):**

Run the visualization script:

```bash
cmind script generate_viz.py
```

Report:

* Status of data flow building
* Summary of edges and subtree order
* Preparedness for next stage (`/cmind.design_base_classes` or `/cmind.design_interfaces`)

Prompt the user:

```text
Data flow has been generated. Review the file structure at:
.cmind/data/data_flow.json

Visualization generated at:
.cmind/data/data_flow_viz.html (Open in browser to inspect)

To proceed with base class design, run:
/cmind.design_base_classes

To regenerate with adjustments, run:
/cmind.build_data_flow <adjustment instructions>
```

**If keeping existing data flow:**

Display the current data flow information:

```text
Current data flow:
- Data Flow Edges: <edge_count>
- Components: <component_count>
- Subtree Order: <subtree_order>

Next step: Run /cmind.design_base_classes
```
