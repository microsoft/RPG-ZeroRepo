#!/usr/bin/env python3
"""Data Flow Design Prompts.

This module contains prompts for data flow design stage.
"""

# ============================================================================
# Data Flow Design Prompts
# ============================================================================

DATA_FLOW_PROMPT = """
You are a system architect designing the **inter-subtree data flow** for a software repository.

Your goal is to describe **how data moves** between functional subtrees as a **directed acyclic graph (DAG)** of edges.
Each edge represents one data object passed from one subtree to another.

## Hard Constraint: The Graph MUST Be Acyclic
- The final data flow **must be a strict DAG**:
  There must be **no path** that starts from a subtree and eventually returns to the **same** subtree via one or more edges.
- In particular, you **must not** create:
  - Direct 2-node cycles, e.g. `A → B` and `B → A`
  - Longer cycles, e.g. `A → B → C → A`
  - Any chain that, when followed, returns to an earlier subtree.
- If the natural design seems to require feedback or iteration, you **must instead**:
  - Introduce explicit, one-directional stages (e.g. `Trainer → MetricsCollector → Reporting`), or
  - Model the feedback as a new, downstream component instead of sending data "backwards".
- Before producing output, mentally verify that your proposed `data_flow` is a DAG with **no cycles of any length**.

## Data Flow Guidelines
- Treat each edge as a meaningful data handoff between two **distinct** subtrees (no self-loops like `A → A`).
- Every subtree defined in the system must appear **at least once** as a producer or consumer.
- Reuse logical data types across edges when they represent the same structure.
- Prefer explicit, schema-aware data descriptions; do not use pandas-style tabular types.
- Ensure naming is consistent and domain-aware.

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block:

<think>
Architectural scratch work:
- Enumerate the subtrees and their responsibilities.
- Propose candidate edges and check whether they introduce any cycles.
- If you detect a potential cycle, refactor until the graph is acyclic.
</think>
<result_json>
{{
  "data_flow": [
    {{
      "source": "source_subtree_name",
      "target": "target_subtree_name",
      "data_id": "unique name or description of the data exchange",
      "data_type": "logical type or structure of the data (e.g., 'FeatureBatch', 'InferenceResult')",
      "transformation": "1–2 sentences describing how the data is processed / validated / serialized / enriched during this transfer"
    }}
  ]
}}
</result_json>

Constraints:
- source != target (no self-loops)
- No cycles of any length in the overall graph
- Every required subtree should appear at least once as producer or consumer
- transformation must not be empty
""".strip()


DATA_FLOW_REVIEW_PROMPT = """
You are reviewing the cross-subsystem data architecture of the repository.

The submitted data flow graph defines how subtrees collaborate, what data contracts they expose, and how responsibilities are split.  
If this graph is incorrect, vague, or overcomplicated, the entire system will suffer from tight coupling and unclear interfaces.

Review this as a strategic decision about how information moves across architectural boundaries.

## Constraints
- Every subtree must appear at least once as a producer ("source") or consumer ("target").
- The graph must be a Directed Acyclic Graph (no cycles, no self-loops).
- Data edges should be semantically plausible (realistic producer → consumer relationships).
- Prefer clear, reusable data types over ad-hoc labels; avoid vague types like "object" or "any".

## Review Dimensions
1. Data Integrity
   - Are data types and contracts consistent and believable across edges?
   - Are there obvious type mismatches or broken assumptions between producer and consumer?
2. Flow Logic
   - Do the directions of edges make sense given each subtree's role?
   - Is the graph acyclic and free of self-loops and obviously redundant or unjustified flows?
3. Transformation Clarity
   - Is it clear what happens to data at each hop (transformation field)?
   - Do transformations align with the roles of the involved subtrees, or are they vague/hand-wavy?
4. Coverage
   - Are all subtrees from {trees_names} represented, with no missing or extraneous names?
   - Are there isolated or under-connected subtrees that indicate gaps or unclear responsibilities?

## Output Format
Return **only** a valid JSON object in the following format:
{
  "review": {
    "Data Integrity": {
      "feedback": "<Your feedback here>",
      "pass": true/false
    },
    "Flow Logic": {
      "feedback": "<Your feedback here>",
      "pass": true/false
    },
    "Transformation Clarity": {
      "feedback": "<Your feedback here>",
      "pass": true/false
    },
    "Coverage": {
      "feedback": "<Your feedback here>",
      "pass": true/false
    }
  },
  "final_pass": true/false
}

Rules:
- `final_pass` should be `true` only if all four dimensions pass, or if remaining issues are minor and easily fixable.
- All `feedback` fields must provide concrete, actionable guidance.
- Do not add new fields or categories beyond the four listed.
""".strip()


# ============================================================================
# Utility Functions for Prompt Building
# ============================================================================

def format_functional_areas(functional_areas: list, component_dirs: dict = None) -> str:
    """Format functional areas for prompt display."""
    lines = []
    for area in functional_areas:
        if component_dirs and area in component_dirs:
            lines.append(f"- {area} [{component_dirs[area]}]")
        else:
            lines.append(f"- {area}")
    return "\n".join(lines)
