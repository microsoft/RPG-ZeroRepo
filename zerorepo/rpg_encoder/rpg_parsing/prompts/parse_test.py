PARSE_TEST_CLASS = """
## Instructions
You are a code analysis assistant specialized in test organization and static analysis.
Your task is to analyze a single Python **test class** and group its test methods by the **core algorithm, function, or logical component** they target.

Each method starts with `test_` and usually reveals the target algorithm/component/behavior through its name, docstring, assertions, or the primary function it invokes.

## Grouping Principles (Merge Bias)
1. **Semantic first, names second**: Identify the **central algorithm/function/feature** under test. Use method names as signals, but prioritize what the method is actually validating.
2. **Avoid over-splitting**: If multiple methods test different aspects of the **same target** (input types, edge cases, exceptions, output correctness, parameter/config handling, intermediate states), **group them together**.
3. **Merge wrapper/indirect tests**: If a method tests a wrapper/helper/typing utility that directly supports the same core function, **merge into the core function’s group**, unless the target is clearly unrelated.
4. **Class name as a hint**: If the class name clearly indicates the target (e.g., `TestMinkowski`), prefer a **single consolidated group** unless strong evidence shows distinct targets.

## Category Naming Rules
- Use concise, stable, **snake_case** names (e.g., `quick_sort`, `minkowski_metric`, `http_get` or `get` — be consistent).
- Prefer the **public API or canonical algorithm name** as the group key.
- Normalize superficial variations/synonyms (e.g., `get`, `http_get` → pick one).
- Keep names **general enough** to cover all grouped methods, but not so broad as to mix unrelated algorithms.

## Coverage & Constraints
- **Include every test method exactly once** (no omissions, no duplicates).
- **Only** include methods from this class whose names start with `test_`. Ignore fixtures (`setup`, `teardown`), helpers, or nested functions.
- Parameterized tests count as **one method** (use the method’s declared name).
- If the class truly targets only **one** algorithm/function, return a **single group** containing all methods.
- Do **not** create vague buckets like `"others"`, `"misc"`, or `"general"`.

## Output Format
Return **two blocks**:

1. `<think>` (optional, hidden from end users)  
   Briefly explain: identified targets, why certain methods were merged, any non-trivial inferences.
2. `<solution>` (required)  
   A **valid JSON object** only.  
   Keys = group names; values = lists of method names.

### Example Output:
<think>
- All `test_quick_sort_...` methods validate different inputs/behaviors of the same algorithm → merged under "quick_sort".
- `test_get_sort_key` and `test_sort_key_default_behavior` are utility checks tied to the same sorting pipeline → also grouped into "quick_sort".
</think>
<solution>
{{
  "quick_sort": [
    "test_quick_sort_empty_input",
    "test_quick_sort_sorted_input",
    "test_quick_sort_invalid_type",
    "test_get_sort_key",
    "test_sort_key_default_behavior"
  ]
}}
</solution>

## Input Context
### Repo Skeleton
{skeleton_info}
"""


PARSE_TEST_FUNCTION = """
## Instructions
You are a code analysis assistant specialized in test organization and static analysis.
Your task is to analyze a list of Python test functions and group them based on the **core algorithm or functionality** they are testing.

Each function starts with `test_` and typically contains a keyword that reflects the algorithm, operation, or component being tested (e.g., `quick_sort`, `get`, `minkowski`, `dtype`, etc.).

## Requirements
You must:
1. Identify the **core algorithm, function, or behavior** each test function is validating.  
   - Prioritize **semantic understanding** over simple keyword matching. Consider what each test is actually checking rather than just the name.
2. **Group related tests** into a **single, unified category** wherever possible. For example:
   - Methods testing different aspects of the same algorithm (e.g., input validation, exception handling, output correctness) should all belong to the same group.
   - Helper functions like `get_metric_dtype` that support larger algorithms (e.g., `minkowski_metric`) should be grouped with the main algorithm.
3. The goal is to produce **meaningful, broad categories** that represent the core function or behavior being tested — avoid treating every function as a unique test.
4. Output a **single JSON object**:
   - Each key should be a **high-level** category name (representing an algorithm, feature, or behavior).
   - Each value should be a list of test function names that belong to that category.
5. **Do not** use vague categories like `"others"`, `"misc"`, or `"general"`. Every method must be classified into a clear and meaningful group.
6. **Avoid over-splitting** categories. Only create new groups if they truly represent distinct functionality. A well-organized classification will typically have **fewer categories than the number of functions**.
7. Group names should be **concise, consistent, and representative**. Use conceptual, general terms whenever possible (e.g., `sorting`, `http_requests`, `data_validation`).

## Output Format
You must respond with two blocks:

1. A `<think>` block  
   In this block, briefly explain:
   - Which core concepts or functions you identified.
   - How you grouped the test functions together.
   - Any nuances or reasoning that led to merging or categorizing functions.
   This block is optional and will not be shown to users.

2. A `<solution>` block  
   Output the final classification in a **valid JSON object** format only — no extra commentary.

### Example Output:
<think>
- The `test_minkowski_metric_...` methods validate different aspects of the Minkowski metric (e.g., weight validation, handling bad input) — they clearly belong in a single group: `minkowski_metric`.
- `test_get_metric_dtype` and `test_get_metric_bad_dtype` both validate the behavior of the same algorithm, checking its handling of different data types. These tests are grouped under the same category `minkowski_metric`.
</think>
<solution>
{{
  "minkowski_metric": [
    "test_minkowski_metric_validate_weights_values",
    "test_minkowski_metric_validate_weights_size",
    "test_minkowski_metric_validate_bad_p_parameter",
    "test_get_metric_dtype",
    "test_get_metric_bad_dtype"
  ]
}}
</solution>

## Input Context
### Repo Skeleton
{skeleton_info}
"""