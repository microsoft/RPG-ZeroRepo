REFACTOR_TREE = """
You are an expert software architect and large-scale repository refactoring specialist.

## Goal
Reorganize and enrich the repository’s parsed feature tree by assigning each **top-level feature group**
(e.g., "data_loader", "model_trainer", "metrics") to the most semantically appropriate location
within the target architecture.

## Hard Requirements
You must:
- Ensure exhaustive coverage: **every** top-level feature group in <parsed_folder_tree> appears in the output.
- Assign each top-level group to **exactly one** target path (no omissions, no duplicates).
- Keep assignments meaningful, expressive, and domain-aligned; avoid terse or vague labels.
- Operate strictly at **top-level group** granularity — descendants automatically follow their parent group.

## Output
Return exactly **one** JSON object mapping target paths to lists of assigned top-level feature groups.

- JSON keys: target architectural paths.
- JSON values: lists of **top-level group names only** (the keys of <parsed_folder_tree>).
- Do not include child nodes, files, or functions.
- Do not invent new top-level group names.

## Target Path Format (STRICT)
Each target path must have **exactly three levels**:

`<functional_area>/<category_level_1>/<subcategory_level_2>`
- `functional_area` must be one of the provided <functional_areas>.
- `category_level_1` expresses broader purpose or lifecycle role.
- `subcategory_level_2` adds precise specialization or context.
- Each segment: concise (2–5 words), semantically meaningful, intent-focused.

Examples:
- "data ingestion/pipeline orchestration/task scheduling"
- "model training/optimization strategy/hyperparameter tuning"

Avoid filler labels (e.g., "misc", "others", "core", "general").

## Semantic Naming Rules
When creating or adjusting semantic labels (categories/subcategories), follow:
1. Use **"verb + object"** phrasing  
   - e.g., `load config`, `validate token`
2. Use **lowercase English only**.
3. Describe **purpose**, not implementation.
4. One label = one responsibility.
5. If something has multiple distinct roles, prefer multiple precise labels over one overloaded label.
6. Avoid vague verbs (avoid: `handle`, `process`, `deal with`).
7. Avoid implementation details (no control-flow / data-structure references).
8. Avoid mentioning specific libraries/frameworks/formats  
   - Correct: `serialize data`  
   - Incorrect: `pickle object`, `save to json`
9. Prefer domain/system semantics over low-level actions  
   - Correct: `manage session`  
   - Incorrect: `update dict`

## Assignment Principles
Group and assign top-level groups by functional coherence:

Signals to use:
- Functional purpose (load, preprocess, train, evaluate, serve)
- Pipeline stage (ingest → prepare → train → evaluate → deploy)
- Shared domain/modality (image, text, audio, etc.)
- Strong affinity (naming patterns, dependencies, conceptual linkage)

Balance:
- Do not over-cluster into generic buckets.
- Do not over-fragment into many tiny one-item paths without strong justification.
- Prefer cohesive, well-named groupings that preserve architectural clarity.

Coverage:
- Each top-level group must map to **one and only one** target path.
- If no perfect match exists, choose the closest appropriate path rather than omitting the group.

## Scope Constraints
- Only assign **top-level groups** (keys of <parsed_folder_tree>).
- Exclude docs/examples/tests/vendor code unless essential to core functionality.
- Do not invent new functional areas; use only those in <functional_areas>.
- You may define new categories/subcategories as needed, but they must remain meaningful and consistent.

## Output Format (STRICT)
Return **only** the JSON object wrapped exactly as:

<solution>
{
  "<functional_area>/<category>/<subcategory>": ["top_level_group_1", "top_level_group_2", ...],
  "<functional_area>/<category>/<subcategory>": ["top_level_group_3", ...]
}
</solution>
"""



FUNCTIONAL_AREA = """
You are an expert software architect and repository analyst.

Your task:
Given the following information about a software repository:
- Repository name
- Overview or description
- Repository skeleton (folder/file structure)
- Parsed feature or component summaries

Analyze the repository holistically and identify its main functional areas — coherent, high-level modules or subsystems that reflect the repository’s architecture and purpose.

### Core Objective
Produce a SMALL, disciplined set of functional areas that form a clean architectural decomposition:
- Each area is a top-level responsibility (subsystem), not a grab-bag of parts.
- Areas must be mutually exclusive (no overlap) and collectively cover the repo at a high level.

### Hard Constraints (must follow)
1) Be conservative: output **1–8** areas by default.
   - Only exceed 8 if the repository is clearly very large and multi-product, and you can justify strict non-overlap anyway.
2) **No overlap**: each file/folder/concept should map to exactly one area.
   - If two areas could both claim the same components, you must merge them or redefine boundaries until ambiguity is removed.
3) Each area must be **single-responsibility** and stable over time (architectural layer / subsystem).
4) Avoid “layer slicing” and “duplicate abstractions”:
   - Do NOT produce both “DataProcessing” and “DataPipeline” unless you can draw a crisp, non-overlapping boundary.
   - Do NOT produce both “Training” and “Optimization” unless optimization is a separate subsystem with distinct ownership and artifacts.
5) Avoid vague buckets: do NOT use names like Core/Misc/Other/Common/Utils.
6) Do not list tests, docs, CI/build, third-party/vendor code as functional areas.

### Process Guidance (how to stay disciplined)
- Start from the repo’s purpose: identify the **minimum** set of subsystems needed to explain how the repo works end-to-end.
- Prefer merging over splitting. Split only when:
  - responsibilities are clearly different,
  - inputs/outputs are different,
  - and there is minimal shared ownership/code.
- When in doubt, choose fewer areas and broaden boundaries slightly, but keep them precise.

### Naming Principles
- Use PascalCase for names (e.g., "FeatureExtraction", "EvaluationMetrics").
- Names should be concrete and domain-relevant (e.g., "RepoParsing", "TaskOrchestration", "CodeGeneration"), not generic (e.g., "Core").

## Output Format
Your response must contain exactly one <think> block and exactly one <solution> block, with no other content outside these two blocks.

<think>
Architectural notes:
- List candidate subsystems.
- Merge candidates until you reach a conservative set (5–8).
- For each final area, explicitly state boundary rules (what belongs vs. what does not) to ensure no overlap.
- If any overlap remains, merge or rename until the partition is unambiguous.
</think>
<solution>
[
"FunctionalArea1",
"FunctionalArea2",
"FunctionalArea3"
]
</solution>
"""