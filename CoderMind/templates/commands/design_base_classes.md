---
description: Design shared base classes and data structures
name: cmind.design_base_classes
---

## User Input

```text
$ARGUMENTS
```

You **MAY** consider additional user input if provided. If empty,
proceed with default behavior.

## **Outline**

The text entered by the user after `/cmind.design_base_classes` **is the adjustment suggestion**.
Unless it is explicitly empty, you may assume it is always available as `$ARGUMENTS`.
**Do not** ask the user to repeat the input.

### Step 1: Pre-check

Run the script `cmind script check_base_classes.py` to verify the current state.

1. Inspect the `type` field in the output:

   * `error` → Display the error message and stop. Instruct user to fix the error or regenerate. Terminate this command.
   * `init` → Proceed to Step 2.
   * `update` → Display the following prompt and wait for user confirmation:

     ```text
     The base classes file already exists:
     - Base Classes: <class_count>
     - Files: <file_count>
     
     Classes:
     <list classes with names and types>
     
     Do you want to regenerate?
     * Y: Regenerate base classes
     * N: Cancel and keep existing
     ```

     If user choose to regenerate, proceed to Step 2; if to cancel, jump to Step 4.

### Step 2: Design Base Classes

> This command may run for a long time depending on project size.
> **Set your terminal timeout to at least 60 minutes** before running.
> Do **NOT** interrupt or re-run this command.

1. Display the following prompt and wait for user confirmation:

   ```text
   Description: Run the script `cmind script design_base_classes.py` to:
     - Design functional base classes (behavioral abstractions)
     - Design global data structures (shared data formats)
   
   Base classes help improve modularity and code reuse.
   
   Select max iterations for valid design:
     - [Y] → use default (5)
     - [Number] → specify a custom iteration count
   ```

2. Execute the following command with the selected iteration count:

   ```bash
   cmind script design_base_classes.py --max-iterations <default_or_user_defined>
   ```

   The script writes a structured log automatically;
   stdout carries the summary the next step needs.

3. Upon successful completion, display:

   ```text
   ✓ Base classes designed successfully
   
   Summary:
   - Base Classes: <class_count>
   - Files: <file_count>
   
   Classes:
   <list class names, types, and file paths>
   
   Output: .cmind/data/base_classes.json
   ```

### Step 3: Validation

Run the validation script:

```bash
cmind script check_base_classes.py --verbose
```

Display the validation results to the user:

* If `output_valid` is `true` and `syntax_valid` is `true`: Report success
* If `output_valid` is `true` but `syntax_valid` is `false`:
  * Display warning about syntax errors
  * Show `syntax_errors` list with class names and error details
  * Suggest re-running to fix syntax issues
* If `output_valid` is `false`: Display `validation_errors` and suggest re-running

### Step 4: Completion & Handoff

**If base classes were designed (Step 2 was executed):**

Report:

* Status of base class design
* Summary of classes and files
* Preparedness for next stage (`/cmind.design_interfaces`)

Prompt the user:

```text
Base classes have been designed. Review the file at:
.cmind/data/base_classes.json

To proceed with interface design, run:
/cmind.design_interfaces

To regenerate with adjustments, run:
/cmind.design_base_classes <adjustment instructions>
```

**If keeping existing:**

If base classes file exists, display:

```text
Current base classes:
- Base Classes: <class_count>
- Files: <file_count>

Next step: Run /cmind.design_interfaces
```

If no base classes file, display an error:

```text
✗ Error: base_classes.json not found.

This step is required before proceeding to interface design.
Please run /cmind.design_base_classes to generate base classes.
```
