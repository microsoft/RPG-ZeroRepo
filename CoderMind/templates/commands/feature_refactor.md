---
description: Refactor feature tree into modular component architecture
name: cmind.feature_refactor
---

## Workflow

**Working Directory**: All relative paths are based on the project root directory.

1. Run the validation script to verify input and check output file status:

   ```bash
   cmind script feature_refactor_validation.py
   ```

   The script outputs a JSON object. Determine the next action based on the `status` and `action` fields:

   1. **If `status` is `"error"`**: The input file `.cmind/data/feature_build.json` is missing or invalid. Display the error message, prompt the user to rerun the `/cmind.feature_build` command, and then exit.

   2. **If `status` is `"ready"` and `action` is `"create"`**: The output file does not exist or has no valid content. Proceed directly to the next step.

   3. **If `status` is `"ready"` and `action` is `"overwrite_or_skip"`**: The output file already exists with content. Display the following prompt and wait for user confirmation:

      ```markdown
      Note: The output file `.cmind/data/feature_tree.json` already exists and is not empty. Please confirm the operation:
      
      - **Y**: Regenerate the feature tree and overwrite the existing output file.
      - **N**: Cancel and exit the agent.
      ```

2. Run script.

   1. Must display the following information and prompt the user to confirm the maximum number of iterations (default: 10).

      ```markdown
      **description**: Run the script `cmind script feature_refactor.py` to perform a two-step process:
        - Step 1: Plan the structure and number of subtrees
        - Step 2: Iteratively assign features to the planned subtrees
      
      **Note**: Iteration will stop when the maximum number of iterations is reached, or when the feature assignment rate is ≥ 99%.
      
      Select max iterations for feature assignment:
        - **[Y]** → use default (10)
        - **[Number]** → specify a custom iteration count
      ```

   2. Execute the following command with the selected max iteration count (default: 10 or user-defined):

      ```bash
      cmind script feature_refactor.py --max-iterations <default_or_user_defined_iterations>
      ```

      The script writes a structured log automatically;
      stdout carries the summary you need below.

   3. Analyze and summarize the information printed during script execution, and present the results in a Markdown table format.

3. Prompt the user to review the output file `.cmind/data/feature_tree.json`, paying particular attention to the `components` field, which represents the final feature tree.

   - If the user determines that **minor adjustments** are needed, instruct them to run:

     ```text
     /cmind.feature_edit
     ```

   - If the user wants to **regenerate the entire feature tree**, instruct them to run:

     ```text
     /cmind.feature_refactor
     ```

   - If the user is satisfied and wants to proceed to the **next step**, instruct them to run:

     ```text
     /cmind.build_skeleton
     ```
