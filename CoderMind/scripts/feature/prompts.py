#!/usr/bin/env python3
"""Prompt Templates for Feature Tree Operations.

Contains all prompt templates used in feature tree workflows:
- Feature build prompts (expansion and review)
- Feature edit prompts (planning and review)
- Feature refactor prompts (subtree planning and organization)
"""

# ============================================================================
# Feature Build Prompts
# ============================================================================

PROMPT_TEMPLATE_BUILD_REVIEW = r"""
## Instruction
You are a Feature Coverage Review Assistant. Your task is to analyze whether ALL described capabilities from the repository_specification have been properly converted into feature paths in the feature tree.

Review coverage across ALL specification sections:
- **functional_requirements**: Every leaf-level requirement must have a corresponding feature path
- **background_and_overview**: Page structures, routes, data models, and technology integrations described here are implementable features
- **non_functional_requirements**: Security measures, performance constraints, etc. should have concrete feature paths

Perform a **semantic matching** analysis - do not require exact text matches, but verify that the **intent and functionality** of each described item is represented in the feature tree.

Your primary responsibility is to ensure completeness — the feature tree should cover WHAT the system does across all spec sections.

## Review Process
1. Extract distinct capabilities from ALL sections of the repository_specification (functional_requirements, background_and_overview, non_functional_requirements)
2. For each capability, semantically match it against existing feature paths
3. Identify any requirements that are NOT adequately covered
4. For uncovered requirements, generate new feature paths that would cover them
5. Calculate: coverage_percentage = (covered_requirements / total_requirements) * 100
6. **Validate leaf nodes against the Minimum Implementable Unit (MIU) principle**

## IMPORTANT: Coverage Consistency Rule
- If this is a follow-up review (previous_review_result is provided), you MUST maintain consistency:
  - Any requirement/capability that was previously marked as COVERED should remain covered (unless the feature tree was modified to remove relevant paths)
  - Coverage percentage should NOT decrease unless paths were removed
  - Only focus on the previously identified missing functionalities
  - The new coverage should be: previous_coverage + (newly_covered / total_requirements * 100)

## MINIMUM IMPLEMENTABLE UNIT (MIU) PRINCIPLE
**This validation applies ONLY to LEAF NODES (the final segment of each path), NOT to intermediate nodes or top-level categories.**

Each leaf node MUST be a **Minimum Implementable Unit** — independently implementable, testable, and deliverable.

**MIU Criteria:**
1. **Single Action**: One verb + one object; no "and"/"or"
2. **Testable**: Clear input → output or observable state change
3. **Atomic**: One function/method scope; assignable as one dev task
4. **Behavior-focused**: Describes WHAT the system does, not internal execution steps

**[WARNING] CONSERVATIVE DELETION PRINCIPLE (VERY IMPORTANT):**
- **ONLY flag nodes that are CLEARLY and OBVIOUSLY invalid** - no borderline cases
- **When in doubt, KEEP the node** - do not delete
- **Domain-specific terms are usually VALID** - do not flag them as implementation details
- **Prefer suggesting improvements over deletion** - if a node is marginally acceptable, keep it

**CATEGORY-SPECIFIC GUIDANCE (BE LENIENT):**
- Mathematical/statistical operations ARE VALID (e.g., "calculate percentage", "compute average", "aggregate counts", "sum values")
- Algorithm-related operations ARE VALID (e.g., "sort by priority", "topological ordering", "detect cycles", "traverse graph")
- Data structure operations ARE VALID (e.g., "index lookup", "cache retrieval", "queue insertion", "hash mapping")
- Transformation operations ARE VALID (e.g., "parse markers", "normalize paths", "extract values", "filter items")
- Domain-specific operations ARE VALID - respect the repository's domain terminology

**Only flag as INVALID if ALL of these are true:**
1. The node describes a pure internal loop/recursion step (e.g., "iterate items", "recurse children")
2. The node has NO observable outcome or testable result
3. The node is NOT a domain-specific operation
4. You are 100% confident it should be removed

**IMPORTANT DISTINCTIONS:**
- Top-level categories are organizational and should NOT be flagged as MIU violations
- Intermediate path segments are organizational and should NOT be validated against MIU
- Only the FINAL segment (leaf node) of each path is subject to MIU validation


When reviewing, if any leaf nodes violate MIU:
- **Only add to `invalid_leaf_nodes` if you are 100% certain** it is invalid
- Provide `suggested_replacements` that are valid MIUs
- **If unsure, do NOT add to invalid_leaf_nodes** - leave it alone

## FEATURE PATH NAMING STANDARD (for suggested paths)
- TOP-LEVEL CATEGORIES: Use categories that best fit the repository's domain (e.g., workflow, functionality, api, service, module, component, feature, etc.)
- CHARACTERS: lowercase letters a–z, digits 0–9; single spaces allowed inside segments
- SEPARATOR: forward slash "/" ONLY
- PATH DEPTH: 4–7 segments
- VALIDATION: First segment is the top-level category, followed by 3-6 hierarchical segments

## DUPLICATE LEAF NODE RESOLUTION
If duplicate leaf names are detected in the current feature tree, you MUST resolve them by renaming.

**Rules:**
- For each set of duplicate leaf names, keep ONE path unchanged
- Rename the other duplicate(s) to make leaf names unique across the entire tree
- Use more specific or contextual names that reflect the path's location

**Output format for renames:**
```
"duplicate_leaf_renames": [
  "old/full/path/duplicate_name -> more_specific_name",
  "another/path/duplicate_name -> contextual_name"
]
```

## Output Format
Your response MUST contain exactly one <think></think> block and exactly one <result_json></result_json> block.
**IMPORTANT: You MUST ALWAYS output the <result_json> block, even if there are no changes to suggest.**

<think>
1. List all requirements/capabilities found in the repository_specification
2. For each, explain which feature paths cover it (or note if uncovered)
3. Validate leaf nodes against MIU and WHAT-only rules
4. Check for duplicate leaf names and plan renames if needed
5. Provide your coverage calculation methodology
</think>

<result_json>
{{
  "coverage_percentage": 85.5,
  "has_gaps": true,
  "missing_functionalities": [
    "Filter Books by Publication Date - date range filtering not covered"
  ],
  "suggested_paths": [
    {{"path": "workflow/date filtering/ui controls/range selection", "description": "Allow users to select a date range for filtering books"}}
  ],
  "invalid_leaf_nodes": [
    "workflow/user/manage account - too broad"
  ],
  "suggested_replacements": [
    {{"path": "workflow/user/account/update email address", "description": "Allow users to change their email address"}}
  ],
  "duplicate_leaf_renames": [
    "path/to/duplicate -> unique_name"
  ]
}}
</result_json>

**When everything is perfect (100% coverage, no MIU violations, no duplicates), output:**
<result_json>
{{
  "coverage_percentage": 100.0,
  "has_gaps": false,
  "missing_functionalities": [],
  "suggested_paths": [],
  "invalid_leaf_nodes": [],
  "suggested_replacements": [],
  "duplicate_leaf_renames": []
}}
</result_json>

## Inputs

### Previous Review Result (if any):
{previous_review}

### Duplicate Leaf Names Detected:
{duplicate_leaves_info}

### Current Feature Tree:
{current_tree}

### Repository Information:
{repo_info}

"""

PROMPT_TEMPLATE_BUILD_FEATURE = r"""
## Instruction
You are a GitHub Repository Feature Tree Builder. Your task is to convert the repository specification into a comprehensive, structured feature tree. Extract features from **all sections** of the specification:

- **functional_requirements**: The primary source — convert every described capability into feature paths
- **background_and_overview**: Extract implementable features from architecture descriptions, page structures, data models, technology integrations, and routing definitions
- **non_functional_requirements**: Convert security measures, performance constraints, and other cross-cutting concerns into concrete feature paths

**GUIDING PRINCIPLE: Comprehensive Specification Coverage**
- Cover ALL sections of the specification, not just functional_requirements
- If the spec describes pages, routes, or UI structure (even in background sections), create features for them
- If the spec defines data models, create features for the model layer
- If the spec lists security measures, create features for each measure
- Each feature should trace back to the specification where possible
- When the spec implies but does not fully detail a capability (e.g., mentions a page but not its sub-features), you may reasonably expand it into concrete implementable features
- Use domain knowledge to fill in standard supporting features that the spec assumes but does not enumerate (e.g., data models, form handling, error responses)

## LEAF NODE GRANULARITY (MIU Principle)
Each leaf node (final path segment) MUST be a **Minimum Implementable Unit** - independently implementable, testable, and deliverable.

**MIU Criteria:**
1. **Single Action**: One verb + one object; no "and"/"or"
2. **Testable**: Clear input → output or observable state change
3. **Atomic**: One function/method scope; assignable as one dev task
4. **Behavior-focused**: Describes WHAT the system does, not pure control flow

**[WARNING] AVOID THESE PATTERNS (will be flagged in review):**
- Pure loop/iteration steps: "iterate items", "loop through elements", "traverse nodes"
- Pure recursion steps: "recurse children", "recurse subtree"
- Internal state only: "set flag", "increment counter", "mark visited"
- No observable outcome: operations with no return value or side effect

**[OK] VALID leaf node patterns:**
- Returns a value: "calculate X", "compute Y", "get Z"
- Produces output: "generate X", "create Y", "format Z"
- Performs action with result: "detect X", "validate Y", "resolve Z"
- Transforms data: "parse X", "convert Y", "normalize Z"
- Stores/retrieves: "cache X", "lookup Y", "index Z"

**[OK] VALID Examples by Category:**

`computation operation` (mathematical/statistical operations):
- `calculate coverage percentage` → returns a number
- `compute execution duration` → returns time value
- `aggregate test counts` → returns summary statistics
- `measure memory usage` → returns memory metrics

`algorithm` (algorithm-related operations):
- `sort tests by priority` → returns ordered list
- `detect circular dependencies` → returns cycle info
- `resolve fixture ordering` → returns ordered fixtures
- `match keyword expression` → returns matching items

`data structures` (data structure operations):
- `index test by nodeid` → enables lookup
- `cache fixture result` → stores for reuse
- `enqueue test item` → adds to queue
- `lookup parent node` → retrieves parent

`data processing` (transformation operations):
- `parse marker expression` → extracts marker info
- `normalize file path` → standardizes path format
- `extract parameter values` → retrieves param data
- `format error message` → produces readable output

**[FAIL] INVALID Examples (will be deleted in review):**
- `iterate test items` → pure loop step, no outcome
- `recurse into children` → pure recursion step
- `visit graph node` → traversal step only
- `increment failure count` → internal state only

## LEAF NODE UNIQUENESS REQUIREMENT
**Do NOT generate paths whose leaf node name already exists in the Current Feature Tree.**
If a leaf name is taken, use a more specific or different name.

## FEATURE PATH NAMING AND SELECTION STANDARD
- TOP-LEVEL CATEGORIES: Use categories that best fit the repository's domain and existing tree structure
  - Analyze the current feature tree and repository context to determine appropriate categories
  - Common examples: workflow, functionality, api, service, module, component, feature, core, util, etc.
- CHARACTERS: lowercase letters a–z, digits 0–9; single spaces allowed inside segments
- PATH AS NODE SEQUENCE:
  - Each feature path MUST be a sequence of nodes, with each node as one segment.
  - Segments MUST be separated **only** by "/" (forward slash).
  - No other separators are allowed.
- SEPARATOR: forward slash "/" ONLY
  - No leading or trailing "/"
  - No double slashes
  - No spaces around "/"
- DISALLOWED: underscores, hyphens, commas, parentheses, dots, emojis, special symbols
- SEGMENT LENGTH: 1–5 words (prefer 2–4)
- PATH DEPTH: 4–7 segments (minimum: category / subsystem / feature group / specific feature)
- GROUPING: cluster related leaves under shared parents; avoid scattering similar concepts
- NORMALIZATION (apply before validation):
  1) trim leading/trailing spaces
  2) collapse multiple spaces into one
  3) convert underscores and hyphens to spaces
  4) remove non-essential noise phrases
  5) enforce exactly one "/" between segments
  6) remove leading/trailing "/"
  7) convert to lowercase
- HARD VALIDATION (post-normalization):
    - Path must have 4-7 segments (3-6 slashes)
    - Each segment: 1-5 words, lowercase alphanumeric with single spaces
    - REGEX for segment format: ^[a-z0-9]+(?: [a-z0-9]+){{0,4}}$
- SELECTION POLICY:
  - Prefer extending existing branches that map to spec requirements before creating new trunks
  - Only create new top-level categories when no existing category fits a spec requirement
- INTERMEDIATE SEGMENTS MUST BE SELF-DESCRIBING:
  - Each non-leaf segment is the only carrier of that level's meaning (intermediate nodes do NOT have a description field). Choose names that a reader can understand without surrounding context.
  - Source intent / Resulting segment:
      "user-facing display surfaces" → "article display"
      "front-end pages"              → "frontend pages"
      "authentication routes"        → "auth routes"
  - A segment name reads correctly when read alone, in any path it appears in.

## Task
Convert the repository specification into a comprehensive feature tree that covers everything needed for a working implementation.

## Specification Mapping Guidelines
1. **Read ALL sections of the specification** (functional_requirements, background_and_overview, non_functional_requirements) and extract every described capability
2. **For each described capability**, create concrete feature paths needed to implement it
3. **Reasonably expand implied features** — if the spec describes a page, also create features for the forms, navigation, and data handling that page requires
4. **Include standard supporting features** that any working implementation would need (e.g., data model definitions, input validation, error handling at system boundaries)
5. **Do NOT copy generic examples** — design the hierarchy based on the actual repository's domain
6. **Maintain consistency** with any existing tree structure (if current_tree is not empty)

### Path Format Reference (structure only, NOT content to copy)
```
{{top_level_category}}/{{subsystem}}/{{feature_group}}/{{specific_action}}
```
- 4-7 segments deep
- Each segment: 1-5 lowercase words
- Separator: "/" only

## Output Format
Your response MUST contain exactly one <think></think> block and exactly one <result_json></result_json> block.
**IMPORTANT: You MUST ALWAYS output the <result_json> block with valid JSON, even if you have no new paths to add.**

<think>
Describe your analysis approach:
- List requirements from ALL spec sections: functional_requirements, background_and_overview, and non_functional_requirements
- For each new feature path, reference which spec section and item it comes from
- Which spec items still need to be converted into features
- How you ensured no duplicate leaf names with existing tree
</think>

<result_json>
{{
  "add_new_feature_paths": [
    {{"path": "path/to/feature1", "description": "Brief 1-2 sentence description of what this feature does"}},
    {{"path": "path/to/feature2", "description": "Brief 1-2 sentence description of what this feature does"}}
  ],
  "is_complete": false,
  "completion_reason": ""
}}
</result_json>

**When the feature tree adequately covers all functional requirements from the specification, output:**
<result_json>
{{
  "add_new_feature_paths": [],
  "is_complete": true,
  "completion_reason": "All functional requirements from the specification are adequately covered by the feature tree."
}}
</result_json>

Constraints for add_new_feature_paths:
- Each item MUST be a JSON object with "path" and "description" keys.
- The "path" value MUST be a single feature path string composed of multiple nodes separated by "/" (e.g., "functionality/memory management/pooling/adaptive scheduling").
- The "description" value MUST be a concise 1-2 sentence explanation of what this feature does.
- Do NOT return nested structures or any other separators inside paths.
- All paths MUST follow the naming and validation rules above.
- Leaf node names must NOT duplicate existing ones in Current Feature Tree.

Notes:
- The <result_json> block MUST contain valid JSON - this is mandatory.
- No additional comments or text outside the two blocks.

## Completion Judgment
- Set `is_complete: true` when ALL sections of the specification have been covered:
  - Every functional_requirement has corresponding feature paths
  - Every page/route/UI element from background_and_overview has feature paths
  - Every data model from background_and_overview has feature paths
  - Every security/non-functional measure has feature paths
- Avoid generating features that are completely unrelated to the repository's domain
- Ask yourself: "Is there anything described in ANY section of the spec, or reasonably implied by it, that doesn't have a feature path yet?"

## Output Expectations
- Generate feature paths covering ALL spec sections (no fixed minimum or maximum)
- Every feature path uses "/" to separate nodes/segments
- Depth between 4–7 segments
- Grouped siblings with shared prefixes
- No duplicate leaf names with Current Feature Tree

## Inputs

### Current Feature Tree:
{current_tree}

### Repository Information:
{repo_info}
"""

PROMPT_TEMPLATE_BUILD_EXPAND = r"""
## Instruction
You are a GitHub Repository Feature Expansion Assistant. The current feature tree already covers all explicitly described requirements from the repository specification. Your task is to expand the feature tree **beyond the existing specification** by adding features that are **genuinely necessary and reasonable** for a complete, production-quality implementation.

**CRITICAL PRINCIPLE: Beyond-Spec Expansion Only**
- The existing tree already covers the spec — do NOT re-implement or duplicate existing features
- Only add features that the spec does NOT describe but that are **practically necessary**
- Each feature must serve a clear, concrete purpose — explain why the repository would be incomplete without it
- Do NOT add features that are merely speculative, decorative, or "nice to have"
- Fewer, well-justified features are always better than many loosely related ones

## LEAF NODE GRANULARITY (MIU Principle)
Each leaf node (final path segment) MUST be a **Minimum Implementable Unit** - independently implementable, testable, and deliverable.

**MIU Criteria:**
1. **Single Action**: One verb + one object; no "and"/"or"
2. **Testable**: Clear input → output or observable state change
3. **Atomic**: One function/method scope; assignable as one dev task
4. **Behavior-focused**: Describes WHAT the system does, not pure control flow

**[WARNING] AVOID THESE PATTERNS (will be flagged in review):**
- Pure loop/iteration steps: "iterate items", "loop through elements", "traverse nodes"
- Pure recursion steps: "recurse children", "recurse subtree"
- Internal state only: "set flag", "increment counter", "mark visited"
- No observable outcome: operations with no return value or side effect

**[OK] VALID leaf node patterns:**
- Returns a value: "calculate X", "compute Y", "get Z"
- Produces output: "generate X", "create Y", "format Z"
- Performs action with result: "detect X", "validate Y", "resolve Z"
- Transforms data: "parse X", "convert Y", "normalize Z"
- Stores/retrieves: "cache X", "lookup Y", "index Z"

## LEAF NODE UNIQUENESS REQUIREMENT
**Do NOT generate paths whose leaf node name already exists in the Current Feature Tree.**
If a leaf name is taken, use a more specific or different name.

## FEATURE PATH NAMING AND SELECTION STANDARD
- TOP-LEVEL CATEGORIES: Use categories that best fit the repository's domain and existing tree structure
  - Analyze the current feature tree and repository context to determine appropriate categories
  - Common examples: workflow, functionality, api, service, module, component, feature, core, util, etc.
- CHARACTERS: lowercase letters a–z, digits 0–9; single spaces allowed inside segments
- PATH AS NODE SEQUENCE:
  - Each feature path MUST be a sequence of nodes, with each node as one segment.
  - Segments MUST be separated **only** by "/" (forward slash).
  - No other separators are allowed.
- SEPARATOR: forward slash "/" ONLY
  - No leading or trailing "/"
  - No double slashes
  - No spaces around "/"
- DISALLOWED: underscores, hyphens, commas, parentheses, dots, emojis, special symbols
- SEGMENT LENGTH: 1–5 words (prefer 2–4)
- PATH DEPTH: 4–7 segments (minimum: category / subsystem / feature group / specific feature)
- GROUPING: cluster related leaves under shared parents; avoid scattering similar concepts
- NORMALIZATION (apply before validation):
  1) trim leading/trailing spaces
  2) collapse multiple spaces into one
  3) convert underscores and hyphens to spaces
  4) remove non-essential noise phrases
  5) enforce exactly one "/" between segments
  6) remove leading/trailing "/"
  7) convert to lowercase
- HARD VALIDATION (post-normalization):
    - Path must have 4-7 segments (3-6 slashes)
    - Each segment: 1-5 words, lowercase alphanumeric with single spaces
    - REGEX for segment format: ^[a-z0-9]+(?: [a-z0-9]+){{0,4}}$
- SELECTION POLICY:
  - Prefer extending existing branches before creating new trunks
  - Only create new top-level categories when no existing category fits
- INTERMEDIATE SEGMENTS MUST BE SELF-DESCRIBING:
  - Each non-leaf segment is the only carrier of that level's meaning (intermediate nodes do NOT have a description field). Choose names that a reader can understand without surrounding context.
  - Source intent / Resulting segment:
      "user-facing display surfaces" → "article display"
      "front-end pages"              → "frontend pages"
      "authentication routes"        → "auth routes"
  - A segment name reads correctly when read alone, in any path it appears in.

## Task
Analyze the current feature tree and repository specification to identify functional areas that are NOT covered by the spec but are **practically necessary** for a production-quality implementation. Add features for these areas.

## Analysis Approach
1. **Review the spec-based tree** — understand what is already covered
2. **Identify practical gaps** — what functionality would a real implementation need that the spec doesn't mention?
3. **Prioritize by necessity** — focus on features the repository genuinely cannot work without
4. **Consider common patterns** for this type of repository — error handling, edge cases, data validation at system boundaries, performance-critical operations
5. **Maintain consistency** with the existing tree structure

### Path Format Reference (structure only, NOT content to copy)
```
{{top_level_category}}/{{subsystem}}/{{feature_group}}/{{specific_action}}
```
- 4-7 segments deep
- Each segment: 1-5 lowercase words
- Separator: "/" only

## Output Format
Your response MUST contain exactly one <think></think> block and exactly one <result_json></result_json> block.
**IMPORTANT: You MUST ALWAYS output the <result_json> block with valid JSON, even if you have no new paths to add.**

<think>
Describe your analysis approach:
- What functionality is already covered by the spec-based tree
- What practical gaps exist that a real implementation would need
- For each new feature, explain why the repository would be noticeably incomplete without it
- How you ensured no duplicate leaf names with existing tree
- Confirm you are NOT duplicating spec-covered features
</think>

<result_json>
{{
  "add_new_feature_paths": [
    {{"path": "path/to/feature1", "description": "Brief 1-2 sentence description of what this feature does"}},
    {{"path": "path/to/feature2", "description": "Brief 1-2 sentence description of what this feature does"}}
  ],
  "is_complete": false,
  "completion_reason": ""
}}
</result_json>

**When all genuinely necessary beyond-spec features have been added:**
<result_json>
{{
  "add_new_feature_paths": [],
  "is_complete": true,
  "completion_reason": "All genuinely necessary features beyond the specification have been added."
}}
</result_json>

Constraints for add_new_feature_paths:
- Each item MUST be a JSON object with "path" and "description" keys.
- The "path" value MUST be a single feature path string composed of multiple nodes separated by "/" (e.g., "functionality/memory management/pooling/adaptive scheduling").
- The "description" value MUST be a concise 1-2 sentence explanation of what this feature does.
- Do NOT return nested structures or any other separators inside paths.
- All paths MUST follow the naming and validation rules above.
- Leaf node names must NOT duplicate existing ones in Current Feature Tree.

Notes:
- The <result_json> block MUST contain valid JSON - this is mandatory.
- No additional comments or text outside the two blocks.

## Completion Judgment
- Set `is_complete: true` when all **genuinely necessary** beyond-spec features have been added
- Do NOT over-expand: if the tree is already comprehensive enough for production use, stop immediately
- Only add features the repository genuinely needs — not speculative ones
- Ask yourself: "Would this repository be noticeably incomplete or broken without this feature?" — if no, do not add it

## Output Expectations
- Generate only features that are genuinely necessary beyond the spec (no fixed minimum or maximum)
- Every feature path uses "/" to separate nodes/segments
- Depth between 4–7 segments
- Grouped siblings with shared prefixes
- No duplicate leaf names with Current Feature Tree

## Inputs

### Current Feature Tree:
{current_tree}

### Repository Information:
{repo_info}
"""

PROMPT_TEMPLATE_BUILD_DIRECTED = r"""
## Instruction
You are a GitHub Repository Feature Expansion Assistant. Your task is to expand the feature tree **beyond the existing specification** in a specific direction chosen by the user. The current tree already covers the spec requirements — your job is to add features that the spec does NOT describe but that are **genuinely necessary and reasonable** for a complete, production-quality implementation.

IMPORTANT CONSTRAINTS:
- Only add features that are **genuinely necessary** for the repository to work well in practice
- Each feature must serve a clear, concrete purpose — you must be able to explain why the repository would be incomplete without it
- Do NOT add features that are merely speculative, decorative, or "nice to have"
- Do NOT duplicate or overlap with features already in the tree (those already cover the spec)
- Fewer, well-justified features are always better than many loosely related ones

## Expansion Direction
{direction}

## LEAF NODE GRANULARITY (MIU Principle)
Each leaf node (final path segment) MUST be a **Minimum Implementable Unit** - independently implementable, testable, and deliverable.

**MIU Criteria:**
1. **Single Action**: One verb + one object; no "and"/"or"
2. **Testable**: Clear input → output or observable state change
3. **Atomic**: One function/method scope; assignable as one dev task
4. **Behavior-focused**: Describes WHAT the system does, not pure control flow

## LEAF NODE UNIQUENESS REQUIREMENT
**Do NOT generate paths whose leaf node name already exists in the Current Feature Tree.**
If a leaf name is taken, use a more specific or different name.

## FEATURE PATH NAMING AND SELECTION STANDARD
- TOP-LEVEL CATEGORIES: Use categories that best fit the repository's domain and existing tree structure
- CHARACTERS: lowercase letters a–z, digits 0–9; single spaces allowed inside segments
- SEPARATOR: forward slash "/" ONLY
- PATH DEPTH: 4–7 segments
- GROUPING: cluster related leaves under shared parents; avoid scattering similar concepts
- NORMALIZATION: trim spaces, collapse multiple spaces, convert underscores/hyphens to spaces, lowercase
- HARD VALIDATION:
    - Path must have 4-7 segments (3-6 slashes)
    - Each segment: 1-5 words, lowercase alphanumeric with single spaces
    - REGEX for segment format: ^[a-z0-9]+(?: [a-z0-9]+){{0,4}}$
- INTERMEDIATE SEGMENTS MUST BE SELF-DESCRIBING:
  - Each non-leaf segment is the only carrier of that level's meaning (intermediate nodes do NOT have a description field). Choose names that a reader can understand without surrounding context.
  - Source intent / Resulting segment:
      "user-facing display surfaces" → "article display"
      "front-end pages"              → "frontend pages"
      "authentication routes"        → "auth routes"
  - A segment name reads correctly when read alone, in any path it appears in.

## Completion Judgment
- Set `is_complete: true` when the expansion direction has been **sufficiently covered** with all genuinely necessary features
- Do NOT over-expand: if the direction is well-covered, stop immediately
- Only add features that the repository genuinely needs — not speculative ones
- Ask yourself: "Would this repository be noticeably incomplete or broken without this feature?" — if no, do not add it

## Output Format
Your response MUST contain exactly one <think></think> block and exactly one <result_json></result_json> block.

<think>
Describe your analysis:
- What the expansion direction requires that is NOT already in the spec-based tree
- For each feature, explain concretely why the repository would be incomplete without it
- Which areas of the direction still need coverage
- Why you consider the expansion complete or incomplete
</think>

<result_json>
{{
  "add_new_feature_paths": [
    {{"path": "path/to/feature1", "description": "Brief 1-2 sentence description of what this feature does"}},
    {{"path": "path/to/feature2", "description": "Brief 1-2 sentence description of what this feature does"}}
  ],
  "is_complete": false,
  "completion_reason": ""
}}
</result_json>

**When the direction is sufficiently expanded:**
<result_json>
{{
  "add_new_feature_paths": [],
  "is_complete": true,
  "completion_reason": "The expansion direction has been fully covered with all necessary features."
}}
</result_json>

Constraints for add_new_feature_paths:
- Each item MUST be a JSON object with "path" and "description" keys.
- The "path" value MUST be a single feature path string with "/" separators.
- The "description" value MUST be a concise 1-2 sentence explanation of what this feature does.
- All paths MUST follow the naming and validation rules above.
- Leaf node names must NOT duplicate existing ones in Current Feature Tree.

## Inputs

### Current Feature Tree:
{current_tree}

### Repository Information:
{repo_info}
"""

PROMPT_TEMPLATE_SUGGEST_DIRECTIONS = r"""
## Instruction
You are a Feature Tree Analysis Assistant. The current feature tree already covers all explicitly described requirements from the specification. Your task is to suggest **4 to 6 expansion directions** for features that go **beyond the spec** but are **genuinely necessary** for a complete, production-quality repository.

Each direction should represent a coherent functional area or capability that is:
- **NOT already covered** by the existing feature tree (which covers the spec)
- **Genuinely necessary** for the repository to work well in practice
- Concrete enough to guide feature expansion
- Something the repository would be noticeably incomplete without
- NOT speculative, generic, or merely "nice to have"

Do NOT suggest directions that duplicate what the spec already covers.

## Analysis Process
1. Understand the repository's purpose and what the spec-based tree already covers
2. **Review the expansion history** (previously generated directions and user selections) to understand what has already been explored and what the user considered important
3. Identify functional areas that the spec does not describe but that are **practically necessary**
4. Focus on areas where the repository would be incomplete or impractical without them
5. Rank directions by how essential they are to a working, production-quality repository

## IMPORTANT: Expansion History Awareness
- **Previously selected directions** indicate areas the user found important — use them to understand the user's priorities and suggest complementary directions
- **Previously generated but NOT selected directions** may still be relevant — you may suggest them again if they remain genuinely necessary, but consider whether the user intentionally skipped them
- **Do NOT suggest directions that have already been expanded** (i.e., previously selected and expanded into the feature tree)
- Use the history to generate **progressively more refined and contextually relevant** suggestions

## Output Format
Your response MUST contain exactly one <think></think> block and exactly one <result_json></result_json> block.

<think>
Analyze the current tree structure, review expansion history, and identify underrepresented or missing functional areas.
</think>

<result_json>
{{
  "directions": [
    {{
      "name": "Short direction name",
      "description": "2-3 sentence description of what this direction covers",
      "rationale": "Why this direction is important for the repository"
    }}
  ]
}}
</result_json>

## Inputs

### Current Feature Tree:
{current_tree}

### Repository Information:
{repo_info}

### Expansion History:
{expansion_history}
"""


# ============================================================================
# Feature Edit Prompts
# ============================================================================

PROMPT_TEMPLATE_EDIT_PLAN = """You are an expert software architect. Your task is to create a precise edit plan for modifying a feature tree.

## User Edit Instructions

```
{edit_instruction}
```

## Repository Information

- **Repository Name**: {repository_name}
- **Repository Purpose**: {repository_purpose}

## Available Components

The feature tree is organized into the following components. Each component has a `refactored_subtree` containing feature paths.

{components_summary}

## Your Task

Analyze the user's instructions and create a detailed edit plan that specifies EXACTLY which paths to add/remove from which components.

### Supported Operation Types

1. **ADD** - Add new features to a component
   - For new top-level feature: `paths_to_add: ["new_category/new_feature"]`
   - For adding under existing path: `paths_to_add: ["existing_category/existing_subcategory/new_feature"]`
   - Example: Add "support parquet format" under "IO Operations/file formats":
     ```json
     {{
       "component_name": "IO & Serialization",
       "operation_type": "ADD",
       "paths_to_remove": [],
       "paths_to_add": ["IO Operations/file formats/support parquet format"],
       "reason": "Add parquet format support"
     }}
     ```

2. **DELETE** - Remove features from a component
   - `paths_to_remove: ["path/to/feature"]` removes the leaf
   - `paths_to_remove: ["path/to/category"]` removes entire category with all children
   - Example:
     ```json
     {{
       "component_name": "Data Structures",
       "operation_type": "DELETE",
       "paths_to_remove": ["deprecated/old_feature"],
       "paths_to_add": [],
       "reason": "Remove deprecated feature"
     }}
     ```

3. **MOVE** - Move features between components
   - Requires TWO separate ComponentOperation entries:
     - First: DELETE from source component
     - Second: ADD to target component
   - Example: Move "rolling window" from Component A to Component B:
     ```json
     [
       {{
         "component_name": "Component A",
         "operation_type": "DELETE",
         "paths_to_remove": ["windowing/rolling window"],
         "paths_to_add": [],
         "reason": "Move rolling window to Component B"
       }},
       {{
         "component_name": "Component B", 
         "operation_type": "ADD",
         "paths_to_remove": [],
         "paths_to_add": ["windowing/rolling window"],
         "reason": "Receive rolling window from Component A"
       }}
     ]
     ```

4. **RENAME** - Rename a feature (same component)
   - DELETE old path + ADD new path in ONE ComponentOperation
   - Example: Rename "feature A" to "feature B":
     ```json
     {{
       "component_name": "X",
       "operation_type": "MODIFY",
       "paths_to_remove": ["category/feature A"],
       "paths_to_add": ["category/feature B"],
       "reason": "Rename feature A to feature B"
     }}
     ```

5. **EXTEND** - Expand an existing leaf into a category with children
   - DELETE the original leaf, ADD new sub-features
   - Example: Expand "data validation" into multiple specific validators:
     ```json
     {{
       "component_name": "Data Processing",
       "operation_type": "MODIFY",
       "paths_to_remove": ["validation/data validation"],
       "paths_to_add": [
         "validation/data validation/type checking",
         "validation/data validation/range validation", 
         "validation/data validation/null handling"
       ],
       "reason": "Expand data validation into specific validators"
     }}
     ```

6. **MERGE** - Combine multiple features into one
   - DELETE multiple old paths, ADD one consolidated path
   - Example: Merge similar features:
     ```json
     {{
       "component_name": "Analytics",
       "operation_type": "MODIFY",
       "paths_to_remove": ["stats/mean calculation", "stats/average calculation"],
       "paths_to_add": ["stats/mean and average calculation"],
       "reason": "Merge duplicate statistics features"
     }}
     ```

7. **SPLIT** - Split one feature into multiple
   - DELETE one path, ADD multiple new paths
   - Example: Split a complex feature:
     ```json
     {{
       "component_name": "IO",
       "operation_type": "MODIFY", 
       "paths_to_remove": ["file operations"],
       "paths_to_add": ["file operations/read operations", "file operations/write operations"],
       "reason": "Split file operations into read and write"
     }}
     ```

### Important Rules

1. **Paths must be exact** - use the exact path format shown in the component summaries
2. **One operation per component** - combine all changes for each component into one operation
3. **Validate paths exist** - only DELETE paths that actually exist in the component
4. **Use consistent naming** - new paths should follow the existing naming conventions
5. **Cross-component moves** require separate DELETE and ADD operations

### Path Format

- Paths use "/" as delimiter
- Example: "dataframe/windowing/create rolling window"
- The path represents the hierarchy from root to leaf

## Output Format

Your response must contain exactly one <think> block and exactly one <result_json> block.

<think>
1. Understand what the user wants to do
2. Identify source and target components
3. List exact paths to remove and add
4. Verify the plan is consistent
</think>

<result_json>
{{
  "summary": "Brief description of the edit plan",
  "operations": [
    {{
      "component_name": "Component Name Here",
      "operation_type": "DELETE|ADD|MODIFY",
      "paths_to_remove": ["path/to/remove1", "path/to/remove2"],
      "paths_to_add": ["path/to/add1", "path/to/add2"],
      "reason": "Why this operation is needed"
    }}
  ],
  "is_valid": true,
  "validation_notes": "Any notes about the plan"
}}
</result_json>
"""


PROMPT_TEMPLATE_EDIT_REVIEW = """You are an expert software architect reviewing the results of a feature tree edit operation.

## Original User Instructions

```
{edit_instruction}
```

## Edit Plan That Was Generated

Summary: {plan_summary}

Operations planned:
{plan_operations}

## Execution Results

{execution_results}

## State Before Edit

{state_before}

## State After Edit

{state_after}

## Duplicate Features Detected

{duplicate_features}

## Model Analysis Context (if available)

{model_analysis}

## Your Task

Review whether the edit operation was executed correctly and achieved the user's intent. If there are issues or incomplete changes, generate fix operations.

### Review Criteria

1. **Plan Accuracy**: Did the plan correctly interpret the user's intent?
2. **Execution Accuracy**: Was the plan executed correctly?
3. **No Side Effects**: Are there any unintended changes?
4. **Completeness**: Were all requested changes made?
5. **Consistency**: Is the resulting tree structure consistent and logical?
6. **No Duplicates**: For MOVE operations, ensure features only exist in the TARGET component, not in both source and target.

### If Issues Found

If you find issues that need to be fixed, set `needs_fix` to `true` and provide `fix_operations`.

## Output Format

IMPORTANT: Output ONLY a valid JSON object inside <result_json> tags. Do NOT include any text before or after the JSON. Keep string values on single lines without line breaks.

<result_json>
{{
  "thinking": "Brief analysis in one line",
  "summary": "What was edited and the outcome in one line",
  "execution_matches_plan": true,
  "execution_matches_intent": true,
  "issues_found": [],
  "suggestions": [],
  "overall_success": true,
  "confidence_score": 0.95,
  "needs_fix": false,
  "fix_operations": []
}}
</result_json>

If fixes are needed, use this format for fix_operations:
{{
  "needs_fix": true,
  "fix_operations": [
    {{"component_name": "Name", "operation_type": "DELETE", "paths_to_remove": ["path"], "paths_to_add": [], "reason": "Why"}}
  ]
}}

RULES:
1. All string values must be on a single line (no newlines inside strings)
2. Use double quotes for all strings
3. Boolean values must be lowercase: true or false
4. Arrays can be empty: []
5. No trailing commas
"""


# ============================================================================
# Feature Refactor Prompts
# ============================================================================

PROMPT_TEMPLATE_SUBTREE_PLANNING = """You are an expert software architect specializing in feature tree organization and modular system design.

Your task is to analyze feature trees and design logical subtree structures that represent coherent functional components.

## Definition of Subtrees
In this task, a subtree does not refer to an arbitrary internal tree node.
1. A subtree represents a top-level functional area of the repository.  
2. Each subtree should correspond to a distinct, self-contained domain of functionality that contributes to the overall system.
3. Subtrees must not overlap.  
  - No subtree may conceptually contain another subtree.  
  - No feature should reasonably belong to more than one subtree.
4. Think of subtrees as the primary architectural divisions of the entire repository.
5. They describe how the system is logically partitioned at the functional level, not how individual nodes are arranged in the feature tree.

## Expertise
You are expected to rely on the following knowledge areas:

- Software architecture and modular design principles
- Feature clustering and functional decomposition
- Domain-driven design concepts
- System organization best practices

Guidelines for Subtree Planning

## Functional Cohesion
- Group features that work together to achieve the same functional objective
- Consider data flow and dependencies when determining boundaries
- Separate concerns that serve clearly different purposes

## Modularity Principles
- Determine the appropriate number of subtrees based on the actual complexity and domain structure of the feature tree
- Each subtree should have a specific and focused role in the system
- Minimize dependencies between different subtrees
- Maximize cohesion inside each subtree

## Adaptive Subtree Count Guidelines
The number of subtrees should emerge from semantic analysis of the domain, NOT from feature count formulas.

### Primary Principle: Domain-Driven Division
1. **Identify natural functional boundaries first** - What are the distinct responsibility areas in this system?
2. **Each subtree = one coherent domain** - A subtree should answer "what does this part of the system do?" with a clear, focused answer
3. **Let the domain dictate the count** - If the system naturally has 2 major areas, use 2. If it has 12, use 12.

### Quality Indicators (use these to validate your division, not to determine count):
- **Cohesion check**: Features within a subtree should be more related to each other than to features in other subtrees
- **Naming check**: If you struggle to name a subtree clearly, it may lack coherent purpose
- **Size balance check**: Subtrees with vastly different sizes (e.g., one has 50 features, another has 2) may indicate poor boundary placement
- **Dependency check**: Subtrees should have minimal cross-dependencies

### Red Flags to Avoid:
- **Forced splits**: Creating subtrees just to reduce size, not because of semantic difference
- **Catch-all subtrees**: Names like "Utilities", "Misc", "Other" suggest poor domain analysis
- **Single-feature subtrees**: Unless it represents a truly distinct concern (e.g., "Authentication" might be small but distinct)
- **Overlapping responsibilities**: If a feature could reasonably belong to multiple subtrees, the boundaries are unclear

### Reference Boundaries (soft guidelines, not rules):
- A subtree with fewer than 1-5 features: Consider if it should merge with a related subtree
- A subtree with more than 100-200 features: Consider if it should be split into sub-domains
- These are sanity checks, not targets

## Naming Conventions
- Use clear and descriptive subtree names
- Names should reflect the primary function or domain
- Avoid vague, abstract or overly technical naming
- Consider clarity for readers unfamiliar with implementation details

## Size Considerations
- Balance sizes so no subtree becomes disproportionately large
- Allow variation where complexity demands it
- Larger subtrees may later be subdivided internally
- Avoid trivial subtrees that serve no meaningful standalone purpose
- A subtree with only 1-2 features should be merged into a related subtree unless it represents a truly distinct concern

## Output Requirements
Your output should provide:

- A list of subtrees with names and functional purposes
- A concise explanation of organizational decisions
- An estimated number of features belonging to each subtree
- Coverage that accounts for the entire feature tree space

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block, with no other content outside these two blocks.
<think>
Your internal reasoning and drafts. Treat this as architectural design notes.
Include:
1. Analysis of the feature tree's size and complexity
2. Identification of natural domain boundaries
3. Justification for the chosen number of subtrees
</think>
<result_json>
{
  "total_subtrees": "<integer, determined by domain analysis>",
  "subtree_plans": [
    {
      "name": "<concise descriptive subtree name>",
      "purpose": "<high level description of this subtree's functional role or theme>",
      "estimate_size": "<integer estimate of how many feature paths belong to this subtree>"
    }
  ],
  "reasoning": "<coherent explanation of why this subtree organization is appropriate for the repository and feature tree, including why this specific number of subtrees was chosen>"
}
</result_json>
"""

PROMPT_TEMPLATE_FEATURE_ORGANIZATION = """
## CRITICAL: Path Format Requirement [WARNING]

Every path in assigned_paths MUST have **2 to 8 segments** separated by "/" (i.e., 1-7 slashes).

**Required Format:** `<segment1>/.../<leaf_name>` (minimum 2 segments, maximum 8 segments)

**Format Guidelines:**
- Minimum: 2 segments (e.g., `<category>/<leaf_feature>`)
- Recommended: 3-5 segments for balanced hierarchy
- Maximum: 8 segments for deeply nested structures
- Final segment: Leaf feature name (MUST match exactly from source tree)

**Valid Format Examples:**
- 2 segments: `<domain>/<leaf_feature>`
- 3 segments: `<domain>/<group>/<leaf_feature>`
- 4 segments: `<domain>/<subdomain>/<group>/<leaf_feature>`
- 5+ segments: Use when semantic grouping requires deeper hierarchy

**Invalid Formats (DO NOT USE):**
- `<leaf_only>` [FAIL] (only 1 segment, needs at least 2)
- `<s1>/<s2>/<s3>/<s4>/<s5>/<s6>/<s7>/<s8>/<s9>` [FAIL] (9 segments, exceeds maximum of 8)

**Derive appropriate segment names from the repository's domain and subtree purposes.**

---

## Instruction
You are acting as a senior software architecture engineer responsible for refactoring a complex five-level feature tree into a clean, modular, and semantically consistent architecture.

Your objective is to reorganize all functionality into well-defined modules that are:
- semantically meaningful,
- non-overlapping,
- internally coherent,
- aligned with the natural structure and intent of the system.

## Subtree Definition
A Subtree represents a distinct functional area with a flexible hierarchical structure (2-8 levels deep):

{
  "name": "<root_name>",
  "refactored_subtree": {
    "<category>": {
      "<subcategory>": [
        "<feature1>",
        "<feature2>"
      ]
    }
  }
}

Explanation:
- name: concise label summarizing the scope of this functional area.
- refactored_subtree: structured hierarchy with 2-8 segments:
  - Minimum 2 segments: `<category>/<leaf>`
  - Recommended 3-5 segments for balanced organization
  - Maximum 8 segments for complex nested structures
  - Final segment (Leaf): concrete features originating directly from the original feature tree.

## Leaf Assignment Rules
All assignment actions operate only on leaf nodes of the remaining feature tree.

1. Every value in assigned_paths must correspond to a leaf node currently present in the remaining feature tree.
2. Intermediate categories, internal nodes, or partially expanded paths must never be assigned directly.
3. Leaf labels must remain exactly as they appear in the source feature tree.
4. A leaf that has already been assigned in previous steps must not be reused.

If a value does not exist as a leaf in the remaining feature tree, it must not appear in assigned_paths.

## Path Refactoring Rules
The original feature tree structure is input only for meaning, not for target layout.

1. Your job is to refactor paths, not to preserve them.
2. Middle-level categories may be regrouped or recombined where appropriate.
3. Leaf names must remain unchanged, but the path leading to them may change.
4. Simply copying the original full path for a leaf is considered a failure of refactoring.
5. **CRITICAL:** Each refactored path must have 2-8 segments, with the leaf name as the final segment.

## Path Composition

Each `assigned_paths` entry has two distinct kinds of segments:

1. Intermediate segments — the new component hierarchy you design.
   These follow the Naming Guidance below and do NOT need to mirror
   the source tree's intermediate names. You are free to regroup
   leaves under any meaningful hierarchy.

2. Leaf segment (the final segment) — exactly the leaf's `name` value
   from the source tree, copied verbatim. The leaf name is the stable
   identifier downstream stages use to locate the feature; never
   modify, abbreviate, expand, or annotate it.

Source leaf:
  {"name": "user model definition",
   "description": "User model with username, password_hash, ..."}

Resulting path:
  "user system/data model/user model definition"
                          ^^^^^^^^^^^^^^^^^^^^^^^^
                          leaf segment = source leaf's `name` (verbatim)

## Naming Guidance
When defining categories within a subtree:
- Prefer names that describe real functionality rather than abstract taxonomy labels.
- Avoid generic buckets such as misc, utilities, general, etc.
- Names should plausibly map to real modules, packages, or directories, while still conveying business or system meaning.

## **Requirements**
- Each subtree represents a self-contained functional domain.
- Every valid leaf node appears exactly once across all subtrees.
- Collectively, all subtrees must cover the complete set of valid feature leaves.
- Leaf names must remain exactly as they appear in the source tree (no renaming).
- Each path in assigned_paths has 2-8 segments (leaf name as the final segment).
  - The subtree_name field identifies which subtree receives these paths.
  - The path string itself does NOT include the subtree name as a prefix.

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block, with no other content outside these sections.
<think>
Explain your reasoning process:
1. How you evaluated grouping options and identified natural clusters
2. How you handled features that could belong to multiple subtrees
3. How you ensured balanced distribution across subtrees

Self-check before submission (MANDATORY):
- [ ] Count "/" in each path: every path must have 1-7 slashes (2-8 segments)
- [ ] Verify each leaf_name exists in the remaining feature tree (no invented names)
- [ ] Confirm no leaf appears in more than one assigned_path
- [ ] Ensure intermediate segments form meaningful, non-generic hierarchies
</think>
<result_json>
{
  "assignments": [
    {
      "subtree_name": "<name of the subtree this group belongs to>",
      "assigned_paths": [
        "level1/level2/level3/leaf1",
        "level1/level2/level3/leaf2"
      ]
    }
  ]
}
</result_json>
"""
