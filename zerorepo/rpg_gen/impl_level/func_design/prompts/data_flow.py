COMPONENT_DATA_FLOW = """
You are a system architect designing the **inter-subtree data flow** for a Python repository.

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
  - Model the feedback as a new, downstream component instead of sending data “backwards”.
- Before calling any tool, mentally verify that your proposed `data_flow` is a DAG with **no cycles of any length**.

## Data Flow Guidelines
- Treat each edge as a meaningful data handoff between two **distinct** subtrees (no self-loops like `A → A`).
- The overall graph must be **connected enough** that:
  - Every subtree defined in the system appears **at least once** as a producer or consumer.
- Reuse logical data types across edges when they represent the same structure (for example, a shared batch or result type), instead of inventing many ad-hoc labels.
- Prefer explicit, schema-aware data descriptions; do not use or reference pandas-style tabular types.
- Ensure naming is consistent and domain-aware so the flow is understandable as an architectural diagram.

## Action Space
{Tool_Description}

## Output Format
Your response must contain exactly one <think> block and exactly one <tool_call> block, with no other content outside these two blocks:

<think>
Your internal reasoning and drafts—treat this like architectural scratch work.
Carefully design a **cycle-free** (DAG) data flow:  
- Enumerate the subtrees and their responsibilities.
- Propose candidate edges and check whether they introduce any cycles.
- If you detect a potential cycle, refactor the design (e.g., by inserting intermediate stages or changing responsibilities) until the graph is acyclic.
Only once you are confident the graph is a DAG, prepare the final tool call.
</think>
<tool_call>
{{
  "tool_name": "...",
  "parameters": {{...}}
}}
</tool_call>
""".strip()


COMPONENT_DATA_FLOW_REVIEW = """
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
   - Do the directions of edges make sense given each subtree’s role?
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


GENERATE_DATA_FLOW_TOOL = """
## Tool Name: generate_data_flow

### Description
Generate and validate a **data flow graph** between functional subtrees, ensuring:
- All edges are well-formed.
- All referenced subtrees are valid.
- Every required subtree participates in the flow.
- **The overall graph is a Directed Acyclic Graph (DAG)** — i.e., there are **no cycles** of any length.

### When to Use
Use this tool whenever defining or updating the inter-subtree data flow (who sends what to whom) and you need to check that:

- All referenced subtrees exist in the functional architecture.
- No required subtree is completely unused.
- Each flow unit contains complete and meaningful information.
- Transformations are non-empty and clearly described.
- **All `source -> target` edges together form an acyclic graph** (no feedback loops).

If the natural design seems to require feedback or iteration, you must restructure it into one-directional stages so that the resulting graph remains a DAG.

### Parameters (JSON Schema)

{
  "tool_name": "generate_data_flow",
  "parameters": {
    "data_flow": [
      {
        "source": "source_subtree_name",
        "target": "target_subtree_name",
        "data_id": "unique name or description of the data exchange",
        "data_type": "logical type or structure of the data (e.g., 'FeatureBatch', 'InferenceResult')",
        "transformation": "1–2 sentences describing how the data is processed / validated / serialized / enriched during this transfer"
      }
    ]
  }
}

#### Field Summary
- **data_flow** (list of objects)  
  Non-empty list of directed edges in the data flow graph that together must form a DAG (no cycles).
- **source** (string)  
  Name of the subtree that produces this data; must be a valid functional subtree and different from `target`.
- **target** (string)  
  Name of the subtree that consumes this data; must be a valid functional subtree and different from `source`.
- **data_id** (string)  
  Human-readable identifier for this specific data exchange (e.g., `"processed_features"`, `"training_batches"`).
- **data_type** (string)  
  Logical type or schema of the data being transferred (e.g., `"FeatureBatch"`, `"ModelArtifact"`, `"InferenceResult"`).
- **transformation** (string)  
  One–two sentences describing what happens to the data on this edge, and it must not be empty or `"none"`.

### Graph Constraints
- For all (`source`, `target`) edges:
  - No self-loops: `source != target`.
  - No cycles of any length:
    - No 2-node cycle: `A -> B` **and** `B -> A`.
    - No longer cycle: `A -> B -> C -> ... -> A`.
- Every required subtree should appear at least once as a producer or consumer.

### Example Tool Call
{
  "tool_name": "generate_data_flow",
  "parameters": {
    "data_flow": [
      {
        "source": "Data Processing",
        "target": "Algorithms",
        "data_id": "processed_features",
        "data_type": "FeatureBatch",
        "transformation": "Raw inputs are validated, normalized, and batched into feature tensors ready for algorithm consumption."
      },
      {
        "source": "Algorithms",
        "target": "Machine Learning",
        "data_id": "training_batches",
        "data_type": "FeatureBatch",
        "transformation": "Prepared feature batches are scheduled and streamed into the training loop."
      },
      {
        "source": "Machine Learning",
        "target": "Core Systems",
        "data_id": "trained_model_artifact",
        "data_type": "ModelArtifact",
        "transformation": "The trained model is serialized into a versioned artifact suitable for storage and deployment."
      }
    ]
  }
}
""".strip()