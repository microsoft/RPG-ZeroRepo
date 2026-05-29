---
name: cmind.plan_tasks
description: Plan implementation tasks from interface definitions
---

# Plan Tasks

Create implementation tasks from the interface definitions.

## Workflow

### Step 1: Pre-Check

Run the check script to determine current state:

```bash
cmind script check_tasks.py --json
```

**If type is "error"**:

* Fix the reported issues or run the prerequisite command first

**If type is "init"**:

* Proceed to Step 2 to plan tasks

**If type is "warning"**:

* Display the unit mismatches:

  ```text
  tasks.json exists but has unit mismatches:
  - Units in interfaces: <interfaces_unit_count>
  - Units in tasks: <tasks_unit_count>
  - Mismatches: <warning_count>
  
  Missing in tasks: <list first 5 from in_interfaces_not_tasks>
  Extra in tasks: <list first 5 from in_tasks_not_interfaces>
  
  Do you want to replan?
  * Y: Replan tasks
  * N: Keep existing
  ```

  If user choose to regenerate, proceed to Step 2; if to cancel and keep existing, jump to Step 4.

**If type is "update"**:

* Ask user: "Valid tasks.json already exists and is consistent. Do you want to replan?"
* If yes, proceed to Step 2
* If no, skip to completion

### Step 2: Plan Tasks

> This command may run for a long time depending on project size.
> **Set your terminal timeout to at least 60 minutes** before running.
> Do **NOT** interrupt or re-run this command at background.

Run the task planner:

```bash
cmind script plan_tasks.py
```

The script writes a structured log automatically; stdout carries the
summary you need below.

This will:

1. Read all units from interfaces.json
2. Analyze dependencies between units
3. Sort units topologically
4. Group into implementation tasks
5. **Append the main entry point task** (main.py) as the final core task
6. **Append project file tasks** (requirements.txt, README.md) as post-implementation tasks
7. Save the ordered tasks to `.cmind/data/tasks.json`

### Step 3: Validation

After generation, run the check script again:

```bash
cmind script check_tasks.py --json
```

Verify:

* `output_valid` is `true`
* `cross_validation.is_consistent` is `true` (all units included)
* `stats.total_tasks` shows the number of tasks
* `stats.total_units` shows total units to implement

If `cross_validation.is_consistent` is `false`, display warnings about missing units.

### Step 4: Completion

Summarize the task plan:

* Total number of tasks (including project file tasks)
* Total number of units
* Files to be touched
* Project files to be generated

Guide user on implementation:

```text
> Task planning complete! The implementation order is defined in `tasks.json`.
> 
> The plan includes:
> - Code implementation tasks (ordered by dependencies)
> - Integration test tasks (one per module)
> - Comprehensive end-to-end test task
> - Main entry point task (task_type: main_entry)
>     - Creates the program's entry point (main.py)
>     - Tested via execution (--help flag)
> - Project file tasks (run after all code and main entry are complete):
>   - `requirements.txt` (task_type: project_requirements)
>     - Dependencies based on actual imports
>     - Tested via import validation in isolated environment
>   - `README.md` (task_type: project_docs)
>     - Documentation based on actual code
>     - No testing required
>
> **All tasks must be executed one by one. Do not skip any task.**
```
