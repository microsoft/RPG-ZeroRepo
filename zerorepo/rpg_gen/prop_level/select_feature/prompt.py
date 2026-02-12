"""
Prompt templates for property-level feature generation.
Extracted and unified from feature_gen.py and simple_gen.py.
"""

# Top-level features used across all modes
TOP_FEATURES = [
    'workflow', 'implementation style', 'functionality', 'resource usage', 
    'computation operation', 'user interaction', 'data processing', 
    'file operation', 'dependency relations', 'algorithm', "data structures"
]

# ================== FEATURE MODE PROMPTS ==================
PROMPT_TEMPLATE_SELECT_EXPLOITATION = """
You are a Feature Selection Assistant helping to build a comprehensive feature tree for a software repository.

## Your Role
You are the EXPLOITATION agent. Select feature paths from the provided "Sample Exploit Feature Tree" that:

1. Deepen and extend existing branches in the current repository feature tree
2. Add implementation-level detail to already-established categories
3. Represent concrete technical functionality that logically follows from what already exists

## Exploitation Strategy
- Prefer depth over breadth
- Elaborate on existing paths rather than introducing entirely new domains
- Focus on code-level mechanisms rather than conceptual categories
- Think of these as “the next realistic layer of implementation detail”

## Selection Criteria
1. **Direct Extension**: The path must extend a branch that already exists in the repository tree
2. **Technical Depth**: Features should describe concrete mechanisms, logic, or algorithms
3. **Logical Progression**: Each selection should naturally follow from what is already implemented
4. **Non-redundant**: Avoid anything already present or trivially equivalent

## Exclusion Criteria — DO NOT SELECT:
- vague goals (e.g., "improve accuracy", "enhance usability")
- generic infrastructure or meta tasks (e.g., "logging", "unit testing", "S3 connector")
- container labels without specific logic (e.g., "monitoring", "bias correction")
- high-level orchestration placeholders (e.g., "training loop", "checkpointing")
- renamed duplicates of existing features
- generic “industry standard features” that are not clearly implied by the current tree

## Coverage Expectation
Within these constraints, **select as many useful and relevant paths as possible** — aim for complete, meaningful coverage of all valid exploitation opportunities.

## Output Format
Your response must contain exactly one <think> block and exactly one <tool_call> block, with no other content outside these two blocks:
<think>
Your internal reasoning and drafts—treat this like architectural scratch work.  
</think>
<tool_call>
{
  "all_selected_feature_paths": [
    "feature/path/one",
    "feature/path/two",
    ...
  ]
}
</tool_call>
""".strip()


PROMPT_TEMPLATE_SELECT_EXPLORATION = """
You are a Feature Selection Assistant helping to build a comprehensive feature tree for a software repository.

## Role
You are the EXPLORATION agent. Your task is to select feature paths from the provided Sample Explore Feature Tree that meet the following conditions:
1. They introduce new functional categories that are not currently present.
2. They add forward-looking and domain-relevant capabilities.
3. They complement, but do not overlap with, existing functionality.
4. They expand into adjacent technical areas that remain aligned with the repository’s purpose.

## Definition of Selection
Selection means choosing complete feature paths exactly as they appear in the Sample Explore Feature Tree.
You must always output full hierarchical paths. For example:
Correct: "analytics/modeling/time_series/forecasting"  
Incorrect: "forecasting"
Do not invent, rename, shorten, or modify paths.

## Exploration Strategy
1. Aim for realistic and meaningful expansion.  
2. Consider new use cases that the repository could plausibly support.

## Selection Criteria
1. Novelty. The path introduces a new top-level or mid-level branch.
2. Innovation. The path provides modern, distinctive or advanced capabilities.
3. Complementarity. The path enhances the system without duplicating existing features.
4. Strategic value. The path expands real, practical usage scenarios for the repository.

## Exclusion Criteria
Do not select any of the following:
1. Features unrelated to the repository’s domain or purpose.
2. Trend-driven or fashionable additions with no clear functional connection.
3. Generic platform or ecosystem features such as marketplaces, plugin ecosystems or monetization layers.
4. DevOps, infrastructure or meta tasks including CI/CD, logging, deployment, testing or connectors.
5. Very broad conceptual categories that lack clear implementation direction.
6. Anything already implicitly covered by existing branches.
7. Features that would logically belong to an entirely different type of product.

## General Guidance
When uncertain, prioritize relevance and realism. New categories are acceptable only when they clearly fit within the repository’s conceptual and technical scope.

## Output Format
Your response must contain exactly one <think> block and exactly one <tool_call> block, with no other content outside these two blocks.
<think>
Your internal reasoning and drafts—treat this like architectural scratch work. 
</think>
<tool_call>
{
  "all_selected_feature_paths": [
    "feature/path/one",
    "feature/path/two"
  ]
}
</tool_call>
""".strip()

PROMPT_TEMPLATE_SELF_CHECK = """
You are a Feature Validation Assistant responsible for ensuring the quality and relevance of selected features.

## Your Role
Review the selected feature paths and validate which ones should be added to the repository's feature tree. Your validation should ensure:
1. Features are technically sound and implementable
2. No redundancy with existing features
3. Alignment with repository's purpose and scope
4. Proper naming conventions and path structure

## Validation Criteria
1. **Technical Validity**: Can this feature be concretely implemented in code?
2. **Non-redundancy**: Does this feature already exist in a similar form?
3. **Alignment**: Does this feature support the repository's stated purpose?
4. **Naming**: Is the path properly structured and clearly named?
5. **Scope**: Is this feature at an appropriate level of abstraction?

## Output Format
Your response must contain exactly one <think> block and exactly one <tool_call> block, with no other content outside these two blocks:
<think>
Your internal reasoning and drafts—treat this like architectural scratch work.  
</think>
<tool_call>
{
  "validated_feature_paths": [
    "path/that/passed/validation",
    "another/valid/path",
    ...
  ]
}
</tool_call>
""".strip()

PROMPT_TEMPLATE_MISSING_FEATURES = """
## Instruction
You are a GitHub project assistant designing a functionally complete, production-grade repository.
Your job is to propose **missing functional capabilities / algorithms** the project should reasonably include
based on its real-world purpose, scope, category, and domain expectations — NOT based on the current Feature Tree.

## Objective
Identify up to 50 groups of **concrete, code-level features** that:
1) Fit the repository’s domain and real usage.
2) Are missing or only weakly represented.
3) Map cleanly to implementable modules/classes/functions/algorithms.

## What counts as a feature
Include real behavior and computation:
- parsing, transformation, modeling, inference, optimization, evaluation, scheduling, routing, ranking, etc.
- standard and advanced algorithms, if within scope.

## What NOT to include
Do NOT propose:
- vague intentions (e.g., "improve accuracy", "better UX")
- meta/infrastructure work (e.g., "logging", "unit testing", "S3 connector", "CI")
- container-only concepts without logic (e.g., "monitoring" with no concrete algorithm)
- high-level placeholders (e.g., "training loop", "checkpointing")
- duplicates / simple renames of existing features

## Naming rules (STRICT)
Leaf nodes must match Feature Tree style:
- **2–4 lowercase words** (prefer 2–3)
- words separated by single spaces
- must describe a **specific behavior/algorithm**
- no long sentences, no clauses, no punctuation, no camelCase, no snake_case
- if an idea requires a long description, **split it into multiple short leaves**

Examples (good):
- "beam search"
- "prompt caching"
- "topk sampling"
- "kalman filter"
- "graph shortest path"

Examples (bad):
- "perform beam search decoding with length penalty and repetition avoidance"
- "improve retrieval accuracy"
- "model training pipeline"

## Structure guidelines
- Organize features into a coherent hierarchy reflecting computational architecture.
- **Depth is flexible**: you may use **more than 4 levels** when it improves clarity.
- Prefer small, meaningful subgroups over overly broad buckets.

## Response Format
Respond with ONLY a <think> and an <action> block.
<think>
Review the Repository Information and consider what capabilities a user or developer would reasonably expect from a complete implementation.
Think in three layers:
1. Are there **entire functional domains** missing?
2. Within known domains, are there **entire workflows or categories** that are underdeveloped or absent?
3. Within existing categories, are there **missing algorithms or code-level techniques** that should be implemented?

Only suggest features that represent **concrete, functional implementations** — not abstract ideas or structural placeholders.
</think>
<tool_call>
{
  "missing_features": {
    "category1": {
      "subcategory1": ["specific_feature1", "specific_feature2"],
      "subcategory2": ["specific_feature3"]
    },
    "category2": {
      "subcategory3": ["specific_feature4"],
      "subcategory4": {
        "sub-subcategory1": [
          "specific_feature5",
          ...
        ]
      },
      ...
    }
  }
}
</tool_call>
""".strip()




# ================== SIMPLE MODE PROMPTS ==================
PROMPT_TEMPLATE_FEATURE_GEN = """
## Instruction
You are a GitHub Repository Expansion Assistant. Your goal is to generate a large number of **meaningful, implementable, and domain-aligned feature paths** to extend and future-proof the repository. Follow the Feature Path Naming and Selection Standard rigorously.

Each feature path must correspond to a capability that can be concretely implemented in code (e.g., algorithms, data structures, data transformations, computation routines, or file/data operations).  
Exclude any paths related to tests, logging, monitoring, analytics, visualization, documentation, configuration, or UI decoration.

## FEATURE PATH NAMING AND SELECTION STANDARD
- TOP-LEVEL CATEGORIES (first segment must be one of):
  workflow; functionality; computation operation; data processing; file operation; algorithm; data structures
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
  REGEX: ^(workflow|functionality|computation operation|data processing|file operation|algorithm|data structures)(?:\/[a-z0-9]+(?: [a-z0-9]+){0,4}){3,6}$
  - Total slashes: 3–6
- SELECTION POLICY:
  - If current tree ≥25 paths: ~70% extensions, ~30% new trunks
  - If current tree <25 paths: ~40% extensions, ~60% new trunks
  - Prefer extending shallow branches and underrepresented subsystems
- ANCHOR STRATEGY:
  - Select ≥3 anchor prefixes and add ≥3 children or grandchildren to each
  - Prefer anchors aligned with core goals, low branching, and high impact

## Inputs
1) Repository Information (name, description, purpose, category, usage, scope)
2) Repository Feature Tree (existing structure)

## Task
Generate new feature paths that do not currently exist and that strengthen the architecture through extension and targeted innovation.

## Quality Criteria
- Align with repository objectives and users
- Improve modularity, extensibility, maintainability, or performance
- Fill architectural gaps and support future scale
- Avoid vague or placeholder concepts
- Ensure logical sibling groupings

## Exclusions
- No duplication or trivial rewording
- No test/logging/monitoring/analytics/visualization/config paths
- No invalid separators or structures

### Examples (valid style)
- functionality/memory management/pooling/adaptive scheduling
- functionality/configuration management/schema validation/json schema validation
- data processing/http header parsing/extract values/tokens and cookies
- workflow/batch processing/parallel execution/task distribution

## Output Format
Your response must contain exactly one <think> block and exactly one <tool_call> block, with no other content outside these two blocks:

<think>
Provide 3–6 sentences describing how you analyzed the repository, selected anchor prefixes, introduced new trunks, and ensured compliance with standards.
</think>
<tool_call>
{
  "add_new_feature_paths": [
    "path/to/feature1",
    "path/to/feature2"
  ]
}
</tool_call>

Constraints for add_new_feature_paths:
- Each item MUST be a single feature path string composed of multiple nodes separated by "/" (e.g., "functionality/memory management/pooling/adaptive scheduling").
- Do NOT return nested structures, objects, or any other separators inside paths.
- All paths MUST follow the naming and validation rules above.

Notes:
- The <tool_call> block must be valid JSON.
- No additional comments or text outside the two blocks.

## Output Expectations
- At least 30 unique feature paths (preferably up to 50+ if meaningful)
- Every feature path uses "/" to separate nodes/segments
- Balanced distribution across categories
- Depth between 4–7 segments
- Grouped siblings with shared prefixes
""".strip()
