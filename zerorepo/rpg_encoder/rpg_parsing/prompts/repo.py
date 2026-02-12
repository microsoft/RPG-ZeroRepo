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
Exclude Python paths that clearly do NOT contribute to core library logic or functionality.

## Key Policy
- Default: **keep code unless exclusion is obvious**
- Only exclude paths that are clearly documentation, demos, or benchmark material
- Err on the side of keeping — conservative filtering

## Scope
Consider only:
1) `.py` files  
2) Directories containing `.py` files  
Ignore folders with no `.py`.

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
some/other/irrelevant.py
third_party/
tests/
...
```
</solution>
"""

ANALYZE_DATA_FLOW = """
You are a system architect tasked with EXTRACTING the inter-subtree (functional area) data flows for a Python repository, based solely on the provided context.

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
  - a single precise type string, e.g. "pandas.DataFrame"
  - OR an array of alternatives, e.g. ["pandas.DataFrame", "pyarrow.Table"] to indicate acceptable forms.
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