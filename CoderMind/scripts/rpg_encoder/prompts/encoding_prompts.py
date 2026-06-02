"""RPG Encoding Prompt Templates.

LLM prompt templates for RPG encoding tasks — repo info generation,
file exclusion, functional area planning, tree refactoring, and
optional data-flow analysis.

**All prompts are verbatim copies from RPG-ZeroRepo** (no edits) to
preserve prompt engineering fidelity.

Sources:
    - zerorepo/rpg_encoder/rpg_parsing/prompts/repo.py
    - zerorepo/rpg_encoder/rpg_parsing/prompts/refactor.py
"""

# ============================================================================
# Repo-level Prompts  (from prompts/repo.py)
# ============================================================================

GENERATE_REPO_INFO = """
You are an expert software engineer and repository analyst.

Goal:
Analyze the provided information about a code repository — including its name, directory structure, and README content —
and generate a concise, accurate, and informative Repository Overview.

The overview should clearly summarize:
1. Project Purpose & Domain — What the repository is for, what problem it solves, or what system it implements.
2. Core Functionalities — The main features, algorithms, or components provided by the codebase.
3. Architectural Composition — How the repository is structured logically (e.g., data layer, model layer, interface layer, utilities, etc.).
4. Key Technologies or Frameworks — Important libraries, frameworks, or tools mentioned or implied.
5. Typical Usage or Workflow — (if available) how a user might run or interact with the project.
6. Dependencies and External Integrations — Any notable external APIs, packages, or systems used.

Guidelines:
- Write in clear professional English, as if writing for a technical report.
- Avoid generic statements like "this repository contains code"; instead, infer specific purposes from names and files.
- If information is missing, use reasonable inference based on structure and naming conventions.
- Keep it factual, concise, and neutral in tone.

Output format:
Return your answer strictly inside a <solution> block as follows:
<solution>
```
your generated repository overview text here
```
</solution>
"""

EXCLUDE_FILES = """
You are an expert in large-scale software repository auditing.

## Goal
Exclude paths that clearly do NOT contribute to core library logic or functionality.

## Key Policy
- Default: **keep code unless exclusion is obvious**
- Only exclude paths that are clearly documentation, demos, or benchmark material
- Err on the side of keeping — conservative filtering

## Scope
Consider only source files in supported languages and the directories that
contain them. Ignore folders with no recognised source files.

## Exclude when it is obvious the content is non-core
Examples of clearly non-functional areas:
- Documentation generators / doc build helpers: `docs/`
- Benchmarks / performance tests: `bench/`, `benchmarks/`
- Examples, tutorials, demos: `examples/`, `example/`, `demo/`, `tutorials/`

## Do NOT exclude just because a folder name looks generic
Do **not** remove:
- utilities, tools, helpers
- internal scripts
- development infrastructure
- anything that may support runtime behavior, plugins, CLI, or internal workflows
If there is any plausible chance the code contributes to operation,
**keep it**.

## Precision
- Prefer excluding specific subfolders/files inside docs/examples/bench when possible
- Never exclude top-level packages that look like core code
- Do not guess or invent paths

## Output
Return only excluded paths in this exact format:
<solution>
```
path/to/excluded_dir/
some/other/irrelevant
third_party/
tests/
...
```
</solution>
"""

ANALYZE_DATA_FLOW = """
You are a system architect tasked with EXTRACTING the inter-subtree (functional area) data flows for a source repository, based solely on the provided context.

## Task
From the repository context below, infer a directed data-flow graph between functional subtrees. Each edge represents a data object moving from one subtree to another.

## Output
Return ONLY a JSON array of edges inside <solution> ... </solution>. Each edge is a dict:
[
  {{
    "source": "<source subtree name>",
    "target": "<target subtree name>",
    "data_id": "<unique name or short description of the exchanged data>",
    "data_type": <one type string OR a list of alternative types>,
    "transformation": "<how the data is transformed en route; use 'none' if unchanged>"
  }},
  ...
]

### Validity constraints
1) Subtree names: "source" and "target" MUST be chosen from: {trees_names}
2) Full connectivity: every subtree in {trees_names} must appear at least once (producer or consumer).
3) DAG: no cycles, and no self-loops (from != to).
4) Topology: one-to-many and many-to-one are allowed if semantically sound and acyclic.

### Data typing guidance
- The "data_type" field can be:
  - a single precise type string, e.g. "UserRecord"
  - OR an array of alternatives, e.g. ["UserRecord", "User"] to indicate acceptable forms.
- Container types are allowed and should be explicit, e.g. "list[Sample]", "dict[str, MetricValue]", "tuple[Header, bytes]".
- Prefer consistent, reusable type labels across edges when representing the same logical payload.

### Evidence & inference
- Base your edges on the provided functional layout, cross-area invokes, and code skeleton. Favor edges supported by clear cross-area interactions.
- If multiple plausible flows exist, prefer the minimal acyclic set that connects all subtrees. Merge equivalent flows by reusing the same data_type labels.

## Output Format
You must respond with two blocks: a `<think>` block and a `<solution>` block:
<think>
Your scratchpad: describe how you interpret the function, its goals, and internal reasoning.
</think>
<solution>
[
  {{
    "source": "...",
    "target": "...",
    "data_id": "...",
    "data_type": "..." or ["...", "...", ...],
    "transformation": "..."
  }},
  ...
]
</solution>

## Repository Context
<repo_name>
{repo_name}
</repo_name>

<repo_info>
{repo_info}
</repo_info>

<repo_skeleton>
{repo_skeleton}
</repo_skeleton>

<trees_info>
{trees_info}
</trees_info>

<invokes>
{summary_invokes}
</invokes>

<cross_code>
{cross_code}
</cross_code>
"""

# ============================================================================
# Refactor Prompts  (from prompts/refactor.py)
# ============================================================================

REFACTOR_TREE = """
You are an expert software architect and large-scale repository refactoring specialist.

## Goal
Reorganize and enrich the repository's parsed feature tree by assigning each **top-level feature group**
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


REFACTOR_MODIFIED = """
You are an expert software architect.

## Goal
Some files in the repository have been modified and their feature names (L4) have changed.
Review each file's original placement and decide whether the intermediate path levels (L2, L3)
should also be adjusted to match the updated functionality.

## Input
You will receive:
1. **Functional areas** — the L1 nodes. Do NOT change these.
2. **Current refactored tree** — the existing architecture (for context).
3. **Modified files** — a JSON object keyed by each file's original full path (L1/L2/L3/old_L4):
   - `new_name`: the updated L4 name (mandatory in output).
   - `features`: the new feature descriptions after modification.

## Decision Rules (priority order)
For each file, apply the **first** matching rule:

1. **Keep original L2-L3** — if they still semantically fit the updated features, just replace old L4 with new L4. This is the **default and strongly preferred** choice.
2. **Move to an existing L2-L3 branch** — if the original L2-L3 no longer fit, look at the <current_refactored_tree> and pick an **already-existing** L2/L3 path (under the same L1) that better matches the new features.
3. **Create new L2-L3 only as last resort** — only if NO existing branch in the tree is a reasonable fit, you may propose new L2-L3 names. This should be rare.

In short: **reuse before creating**. Do not invent new branch names when a suitable path already exists in the tree.

## Path Format (STRICT)
Every path (both key and value) has **exactly four levels**:

`L1/L2/L3/L4`

- **L1** (functional area): MUST stay the same as the original — do NOT change it.
- **L2** (category): broader purpose or lifecycle role.
- **L3** (subcategory): precise specialization or context.
- **L4** (file name): MUST be the `new_name` provided in the input.

## Naming Rules for L2-L3
- Use **lowercase English only**.
- Use **"verb + object"** phrasing (e.g., `load config`, `validate token`).
- Describe **purpose**, not implementation.
- Avoid vague labels (`misc`, `core`, `general`, `handle`, `process`).

## Output Format (STRICT)
Return **only** a JSON object mapping each original path to its new path, wrapped as:

<solution>
{
  "L1/old_L2/old_L3/old_L4": "L1/new_L2/new_L3/new_L4",
  ...
}
</solution>

Rules:
- Every original path from the input must appear exactly once as a key.
- Values must be valid 4-level paths with the correct `new_name` as L4.
- If no L2-L3 change is needed, the value path keeps the same L2-L3 with the new L4.
"""


FUNCTIONAL_AREA = """
You are an expert software architect and repository analyst.

Your task:
Given the following information about a software repository:
- Repository name
- Overview or description
- Repository skeleton (folder/file structure)
- Parsed feature or component summaries

Analyze the repository holistically and identify its main functional areas — coherent, high-level modules or subsystems that reflect the repository's architecture and purpose.

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
4) Avoid "layer slicing" and "duplicate abstractions":
   - Do NOT produce both "DataProcessing" and "DataPipeline" unless you can draw a crisp, non-overlapping boundary.
   - Do NOT produce both "Training" and "Optimization" unless optimization is a separate subsystem with distinct ownership and artifacts.
5) Avoid vague buckets: do NOT use names like Core/Misc/Other/Common/Utils.
6) Do not list tests, docs, CI/build, third-party/vendor code as functional areas.

### Process Guidance (how to stay disciplined)
- Start from the repo's purpose: identify the **minimum** set of subsystems needed to explain how the repo works end-to-end.
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
