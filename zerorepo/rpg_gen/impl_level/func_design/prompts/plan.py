PLAN_FILE = """
You are an expert software architect assisting in planning feature implementation within a Python codebase.

Your task is to construct an **implementation dependency graph** across a set of files that collectively realize a functional subtree of the system.  
Each file corresponds to one or more feature paths. These features may have logical dependencies derived from the feature hierarchy and standard software layering principles.

## Repository Context
### High-Level Repository Description
{repo_info}

### Abstract Feature Tree (Omitting Low-Level Detail)
{trees_info}

### Files to be planned
{files_to_planned}

## Planning Guidelines
You must output a **directed acyclic graph (DAG)** over the given file paths, where:
- Each node represents a file (specified as a file path string).
- An edge from A to B means **file A must be implemented before file B**.
- The graph must include **all provided file paths** — do not invent or omit file names.
- The graph must **not contain cycles**.
- Favor bottom-up ordering, respecting typical architecture layering  
  (e.g., utilities before logic, logic before interface layers).

## Output Format (Strict Requirement)
You must output **only** the graph in the following exact JSON structure — no explanations, no commentary, no formatting text:
{{
  "file_implementation_graph": [
    {{"from": "path/to/file1.py", "to": "path/to/file2.py"}},
    {{"from": "path/to/file2.py", "to": "path/to/file3.py"}}
  ]
}}

### Strict structural rules:
1. The top-level object must contain **exactly one key**: `"file_implementation_graph"`.
2. `"file_implementation_graph"` must be a JSON array.
3. Each element of the array must be an object with **exactly two fields**:
   - `"from"` : a string equal to one of the provided file paths  
   - `"to"`   : a string equal to one of the provided file paths  
4. No other keys or fields are permitted.  
5. No file path may appear that was not provided in the input.  
6. The JSON must be valid and parseable — **no trailing commas**, no comments, no text outside the JSON.  
7. The graph must be a **DAG**: no cycles, no self-loops (`"from": X, "to": X"`), no implicit cycles.

## Notice
- Your output must be **only** the JSON object matching the required structure.
- If dependencies are unclear, choose the most reasonable bottom-up ordering — but still obey DAG constraints.
- Do not wrap the JSON in markdown (no ```json or ```).
"""


PLAN_FILE_LIST = """
You are an expert software architect assisting in planning feature implementation within a Python codebase.

Your task is to **reorder the provided file paths** into a single, bottom-up implementation sequence.
The resulting sequence should reflect logical dependencies and standard software layering.

## CRITICAL REQUIREMENTS (Zero Tolerance for Violations)
- The output list MUST be an EXACT reordering of the provided file paths.
- INCLUDE ALL and ONLY the provided file paths.
- Each file path must appear EXACTLY ONCE (no duplicates, no omissions).
- Do NOT invent new paths. Do NOT omit any path. Do NOT repeat any path.
- Do NOT normalize, modify, or rewrite paths; output them VERBATIM.
- Count the input files and ensure your output has the SAME COUNT.

## Planning Guidelines
Prefer bottom-up layering:
utilities/primitives/data models/helpers → core logic → orchestration/workflows → interfaces/adapters → entry points.

If dependencies are uncertain, choose the most reasonable bottom-up order while still including every provided file exactly once.

## Output Format (STRICT)
Your response must contain exactly one <think> block and exactly one <tool_call> block, with NO other text outside them.
<think>
Think through the problem step by step by following the instructions below:

1) Determine INPUT_COUNT as the total number of input file paths.
2) Derive a bottom-up implementation order for all files based on logical dependencies and layering.
3) Verify the candidate order:
   - Every input file appears exactly once.
   - No extra or invented file is included.
   - The list length equals INPUT_COUNT.
4) If any discrepancy is found, correct the list and re-run the verification.
5) Once verification passes, stop reasoning and produce only the final <tool_call> block.
</think>
<tool_call>
{{
  "implementation_order": [
    "path/to/file1.py",
    "path/to/file2.py"
  ]
}}
</tool_call>

## Structural Rules (MUST follow)
1) The top-level object must contain exactly one key: "implementation_order".
2) "implementation_order" must be a JSON array.
3) Every element must be a string matching a provided file path EXACTLY.
4) Each provided file path must appear exactly once.
5) Do NOT use placeholders (e.g., "...") and do NOT include explanation text.
6) The array length must equal the number of input files.

## Error Prevention
- BEFORE outputting, double-check that your list contains exactly the same files as the input.
- If you find any discrepancy (missing, extra, or duplicate files), you MUST fix it.
- Remember: This is a REORDERING task, not a filtering or selection task.


====Task Beigin=====
## Repository Context
### High-Level Repository Description
{repo_info}

### Abstract Feature Tree (Omitting Low-Level Detail)
{trees_info}

### Files to be planned (Authoritative List)
{files_to_planned}
"""