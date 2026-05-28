---
description: Generate and iteratively refine the feature tree based on functional requirements.
name: cmind.feature_build
---

## Workflow

**Working Directory**: All relative paths and command executions are based on the project root directory

This workflow has four steps:

- **Step 1 (Validate Status)**: Check input/output file readiness before starting.
- **Step 2 (Build or Expand)**: If the output file does not exist, build the feature tree from the specification. If it already exists, automatically switch to beyond-spec expansion mode to add features the spec does not describe but are practically necessary.
- **Step 3 (Optional User-Guided Expansion)**: Expand the feature tree beyond the spec in user-chosen directions. Only adds reasonable and necessary features.
- **Step 4 (Completion)**: Report results and hand off to the next phase.

### Step 1: Validate Status

Execute the following command to check the current state of input/output files:

```bash
cmind script feature_build_validation.py
```

**After execution, parse the JSON output and display a user-friendly summary.**

**Display format based on results:**

1. **If `status` is `error`:**

   - Inform the user about the issues based on the error information in the output

   - Remind the user to run `/cmind.feature_spec` first to create a valid `.cmind/data/feature_spec.json`

2. **If `status` is `ready`:**

   1. If message = "Output exists", display the following information for user decision:

      ```markdown
      Output file `.cmind/data/feature_build.json` already exists.
      Continuing will expand the feature tree beyond the specification, adding features not described in the spec but practically necessary for production use.
      
      Please enter your choice:
      **Y**: Continue expanding
      **N**: Exit
      ```

      If user inputs Y, proceed to `Step 2`

      If user inputs N, stop the agent execution

   2. If message = "Ready to create", proceed to `Step 2`

### Step 2: Spec-Driven Feature Tree Construction

The script automatically detects whether the output file (`feature_build.json`) already exists:

- **Output file does NOT exist**: Builds the feature tree from the specification. The model iterates autonomously until it determines all spec requirements are covered, then a full review phase validates coverage (gaps + MIU + duplicates).
- **Output file already exists** (with a non-empty feature tree): Assumes the spec-based features are already complete. Automatically switches to **beyond-spec expansion mode**, adding features that the specification does not describe but are genuinely necessary for a production-quality implementation. Uses a lightweight review (MIU + duplicates only, no coverage gap check).

1. **Execute the command:**

   ```bash
   cmind script feature_build.py --mode step1
   ```

   The script prints its full output on stdout and also writes a
   structured log automatically. Inspect the stdout to capture the
   `FEATURE EXPANSION SUMMARY` section described below.

   **Available parameters for Step 2:**

   | Parameter                 | Default | Description                                                    |
   | ------------------------- | ------- | -------------------------------------------------------------- |
   | `--mode step1`            | step1   | Spec-driven build mode                                         |
   | `--review-threshold`      | 98.0    | Coverage percentage threshold for review (0–100)               |
   | `--review-max-iterations` | 3       | Maximum iterations for review phase                            |

   **Note:** The expansion loop has a hard safety cap of 20 iterations. The model self-terminates when it determines all spec requirements are covered.

2. **After command executes successfully:**

   - Capture the **complete standard output (stdout)** from the script

   - Look for the section containing the following marker:

     ```text
     FEATURE EXPANSION SUMMARY
     ```

   - If this section exists, display the result information in **Markdown table** format.

3. **Proceed to Step 3.**

### Step 3: User-Guided Expansion (Optional)

After the spec-driven build is complete, ask the user whether they want to expand the feature tree beyond the documented specification.

1. **Display the following prompt to the user:**

   ```markdown
   Spec-driven feature tree construction is complete.
   
   Would you like to expand the feature tree beyond the documented specification?
   This will add features not covered in the original spec — only reasonable and necessary features will be added.
   
   **Y**: Yes, suggest expansion directions
   **N**: No, finish here
   ```

2. **If user inputs `N`:** Proceed to `Step 4: Completion`.

3. **If user inputs `Y`:**

   a. **Get expansion direction suggestions:**

      ```bash
      cmind script feature_build.py --mode suggest-directions
      ```

      The JSON payload is printed on stdout (and the full log is
      written automatically).

   b. **Parse the JSON output** and display the directions as a numbered list to the user:

      ```markdown
      Suggested expansion directions:
      
      | # | Direction | Description | Rationale |
      |---|-----------|-------------|-----------|
      | 1 | {name}    | {description} | {rationale} |
      | 2 | {name}    | {description} | {rationale} |
      | ... | ... | ... | ... |
      
      Enter the numbers of the directions you want to expand (comma-separated, e.g. `1,3,5`), or **N** to finish:
      ```

   c. **If user inputs `N`:** Proceed to `Step 4: Completion`.

   d. **If user selects direction numbers (single or multiple):**

      **Normalize the user's input** before passing to the script: extract only the numeric values and join them with commas (no spaces). For example:
      - User enters `1, 3, 5` → normalize to `1,3,5`
      - User enters `1 3 5` → normalize to `1,3,5`
      - User enters `1、3、5` (Chinese commas) → normalize to `1,3,5`
      - User enters `2` → pass as `2`

      Then pass the normalized indices to the script:

      ```bash
      cmind script feature_build.py \
       --mode step2 \
       --direction "<normalized indices>"
      ```

      For example, if the user enters `1,3,5`:

      ```bash
      cmind script feature_build.py \
       --mode step2 \
       --direction "1,3,5"
      ```

      **What happens inside the script:**
      1. The script reads each index and looks up the corresponding direction from the **latest round** of `expansion_directions[]` saved in `feature_build.json` by the `suggest-directions` step
      2. For each direction, the script resolves the full direction context (name + description + rationale) and passes it as part of the expansion prompt
      3. The script expands all selected directions sequentially — each direction runs against the latest tree (including features added by previous directions)
      4. After each direction's expansion, a lightweight review checks for duplicate leaf nodes and MIU violations
      5. The selected directions are recorded in the corresponding round's `selected` array for history tracking

      Display all `FEATURE EXPANSION SUMMARY` sections in **Markdown table** format (if present).

   e. **After all selected directions are expanded:**

      - Display a brief summary of which directions were expanded.

      - **Ask the user again:**

        ```markdown
        All selected directions have been expanded. Would you like to continue expanding in more directions?
        
        **Y**: Yes, show directions again
        **N**: No, finish here
        ```

      - If `Y`: Go back to step 3a (re-run `suggest-directions` to get updated suggestions based on the now-expanded tree and full expansion history from all previous rounds).
      - If `N`: Proceed to `Step 4: Completion`.

### Note on Expansion History

The `expansion_directions` field in `feature_build.json` is an **array of rounds**. Each time `suggest-directions` runs, a new round is appended. Each round records:

- The generated directions for that round
- Which directions the user selected

This history is automatically fed back to the model when generating new directions, helping it produce more contextually relevant suggestions that build on previous decisions.

### Step 4: Completion and Handoff

Report includes:

- Feature tree generation status
- Total feature count
- Whether ready to proceed to the next phase (`/cmind.feature_refactor`)
