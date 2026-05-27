---
name: rpgkit.design_interfaces
description: Design interfaces (functions/classes) for repository files
---

All the bash command timeout is set to 1 hour.

# Design Interfaces

Design function and class interfaces for your repository files based on the skeleton structure.

## Workflow

### Step 1: Pre-Check

Run the check script to determine current state:

```bash
rpgkit script check_interfaces.py --json
```

**If type is "error"**:

* Fix the reported issues or run the prerequisite command first

**If type is "init"**:

* Proceed to Step 2 to design interfaces

**If type is "warning"**:

* Display the feature mismatches:

  ```text
  interfaces.json exists but has feature mismatches:
  - Features in skeleton: <skeleton_feature_count>
  - Features in interfaces: <interfaces_feature_count>
  - Mismatches: <warning_count>
  
  Missing in interfaces: <list first 5 from in_skeleton_not_interfaces>
  Extra in interfaces: <list first 5 from in_interfaces_not_skeleton>
  
  Do you want to redesign?
  * Y: Redesign interfaces
  * N: Keep existing
  ```

  If user choose to regenerate, proceed to Step 2; if to cancel and keep existing, jump to Step 4.

**If type is "update"**:

* Ask user: "Valid interfaces.json already exists and is consistent. Do you want to redesign?"
* If yes, proceed to Step 2
* If no, skip to completion

### Step 2: Design Interfaces

> This command may run for a long time depending on project size.
> **Set your terminal timeout to at least 180 minutes** before running.
> Do **NOT** interrupt or re-run this command.

Run the interface designer:

```bash
rpgkit script design_interfaces.py
```

The script writes a structured log automatically; stdout carries the
summary you need below.

This will:

1. Read the skeleton.json file structure
2. Read data_flow.json for subtree processing order (if available)
3. Read base_classes.json for context (if available)
4. For each file, design the appropriate functions and classes
5. Generate signatures with type hints and comprehensive docstrings
6. Map each unit to the features it implements
7. Save the results to `.rpgkit/data/interfaces.json`

Note: If data_flow.json exists, components are processed in the subtree order
defined by the data flow DAG. This ensures dependencies are resolved correctly.

### Step 3: Validation

After generation, run the check script again:

```bash
rpgkit script check_interfaces.py --json
```

Verify:

* `output_valid` is `true`
* `cross_validation.is_consistent` is `true` (all features mapped correctly)
* `stats.units` shows the number of designed units
* `stats.features_mapped` shows feature coverage

If `cross_validation.is_consistent` is `false`, display warnings about unmapped features.

### Step 4: Completion

Summarize what was designed:

* Number of components processed
* Total number of units (functions + classes)
* Feature coverage

Guide user to the next step:

```text
> Interface design complete! Your next step is:
> 
> **/rpgkit.plan_tasks** - Create implementation tasks
```
