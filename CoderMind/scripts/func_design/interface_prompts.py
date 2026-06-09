#!/usr/bin/env python3
"""Interface Design Prompts.

This module contains prompts for interface design stage.
"""

# ============================================================================
# Interface Design Prompts
# ============================================================================

INTERFACE_PROMPT = """
You are designing interfaces (functions, classes, structs, interfaces, or methods) for a large, production-oriented target-language repository.
The goal is not to write arbitrary APIs, but to define interfaces that integrate cleanly into the repository's architecture, respect existing data flows, and follow established conventions for modules, base classes, and shared data structures.

## Objective
For each invocation:
1. Select exactly one assigned feature, or a small group of closely related features.
2. Define exactly one public target-language interface for it.
3. Provide the following elements:
   - All required imports:
     - standard library imports
     - external dependency imports
     - internal project imports
   - The interface definition:
     - target-language declaration stubs only
     - no implementation logic; function and method bodies must use a parseable target-language placeholder
   - Precise target-language documentation comments or docstrings documenting:
     - purpose and intended usage context within the repository
     - parameters, including names, types, and semantics
     - return type and meaning
     - assumptions, constraints, error conditions, and edge cases
4. Do not generate implementation logic or pseudo-implementation.
5. Interface design is incremental. Each round may define one or a small number of interfaces, but each must be self-contained and justified.

## Repository Context and Constraints
All interfaces must:
1. Align with the repository's data flow patterns.
2. Use existing shared data structures or typed DTOs where applicable, instead of inventing new ad hoc structures.
3. Inherit from existing base classes when the feature conceptually fits into existing extensibility hierarchies.
4. Call or integrate with internal utility components when appropriate rather than duplicating behavior.
5. Avoid speculative abstractions that are unrelated to the repository's direction.

Interfaces should feel like natural extensions of the repository, not isolated standalone utilities.
## Interface Shape Decision Rules
A function or free operation is appropriate when:
- the operation is conceptually a single computation or transformation,
- the logic is stateless,
- configuration is provided entirely by parameters,
- the operation does not manage lifecycle or persistent state.
- Helper functions are permitted, but only when they clearly support higher-level components rather than replacing them.

A class, struct, interface, trait, type, or receiver-backed method set is appropriate when:
- configuration persists across multiple calls,
- internal state influences behavior,
- multiple related operations belong together,
- subclassing, strategy replacement, or pluggable behavior is expected.

## Cohesion, Scope, and Grouping
1. Interfaces must correspond to a single coherent responsibility.
2. Do not merge unrelated features merely to reduce the number of interfaces.
3. Group features only when they share state, configuration, contract expectations, or lifecycle.
4. If an interface cannot be described in one clear sentence, it likely needs to be split.

## Type Requirements
1. Type annotations are mandatory for all parameters and return values.
2. The type `Any` is strictly prohibited.
3. Use one of the following instead when uncertainty exists:
   - concrete application data structures already defined in the repository,
   - generic types such as `Optional[T]`, `Union[T1, T2]`, `Mapping[...]`, `Iterable[...]`,
   - well-defined protocol or typed base class.

Types must be meaningful, stable, and reflect real usage.

## Architectural Fit
Design each interface so that it:
1. can be extended without breaking current callers,
2. avoids unnecessary coupling between unrelated modules,
3. does not bypass existing architectural layers,
4. uses names consistent with repository modules, packages, and conventions.

Prefer explicit, predictable contracts over overly generic APIs.

## Integration & Wiring Requirements
Every interface must have a clear integration story within the repository's call graph.

1. **Caller/callee awareness**: For each interface, consider:
   - WHO will call this interface? (another interface in this file, an upstream/downstream module, or external invocation)
   - WHAT does this interface call? (other interfaces from upstream context or base classes)
2. **No islands**: An interface that is never called by anyone AND calls nothing is dead code. The only exceptions are top-level interfaces — units not expected to be called by other internal modules (e.g., application entry points, standalone submodules, externally-invoked APIs, framework callbacks).
3. **Upstream integration**: When upstream context provides interfaces that produce data you need, import and use them (or accept their output types as parameters). Don't redefine what already exists upstream.
4. **Execution path test**: For each interface, ask: "Can I trace a plausible execution path from a top-level orchestrator to this code?" If not, the interface is likely orphaned.

## Action Space
For each interface, you MUST declare its dependencies:
- **inherits_from**: List of base class names this class inherits from (from base_classes or upstream modules). Empty list [] if none.
- **calls**: List of function/method names this interface expects to call during execution. Empty list [] if none.
- **uses_types**: List of type names used in parameters, return values, or internally (from base_classes or data structures). Empty list [] if none.

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block:
<think>
For each interface you design, reason through:
1. Evaluate alternative interface shapes and justify your decision against the repository architecture.
2. WHO will call this interface? Name the specific caller, or explain why it is a top-level interface with no internal caller.
3. WHAT does it call from upstream context or base classes? Name specific interfaces you will import.
4. Do the input/output types align with the data flow contracts?
</think>
<result_json>
{{
  "interfaces": [
    {{
      "features": [
        "fully/qualified/feature/path_1",
        "fully/qualified/feature/path_2"
      ],
      "code": "Target-language code string with imports and declaration stubs",
      "dependencies": {{
        "inherits_from": ["BaseClassName"],
        "calls": ["function_or_method_name"],
        "uses_types": ["TypeName"]
      }}
    }}
  ]
}}
</result_json>

Constraints:
- One interface per code string, covering one feature or a tight group of related features.
- The code must define exactly one cohesive target-language declaration group.
- Function and method bodies must use a parseable target-language placeholder and contain no implementation logic.
- Public declarations must have target-language documentation comments or docstrings.
- Prefer explicit, custom containers and typed structures; do not use pandas.DataFrame or other third-party tabular types.
""".strip()


# ============================================================================
# Subtree-Level Interface Design Prompts
# ============================================================================

SUBTREE_INTERFACE_PROMPT = """
You are designing interfaces (functions, classes, structs, interfaces, or methods) for a large, production-oriented target-language repository.
The goal is to define interfaces that integrate cleanly into the repository's architecture, respect existing data flows, and follow established conventions.

## Objective
You are given files within the same subtree (functional area), listed in implementation dependency order.
Design interfaces for **ALL** files, processing them **sequentially** in the given order.
Later files may depend on and reference interfaces from earlier files in this batch.

For each file:
1. Cover ALL assigned feature paths — no feature left uncovered.
2. Each interface covers one feature or a small group of closely related features.
3. For each interface, provide:
   - Required imports (standard library, external, internal project)
  - The interface definition: target-language declaration stubs with no implementation logic
  - Target-language documentation comments or docstrings covering: purpose, parameters with types and semantics, return type, and notable constraints or edge cases
4. You MAY import and reuse symbols from upstream context, base classes, and earlier files in this batch.
5. **Glue/Orchestration Code**: If you need to create orchestrator classes, manager facades, or data structures that integrate multiple features but don't map to any assigned feature, you MAY create NEW feature paths for them. Simply include these new feature paths in the `features` field. New feature paths should follow the same naming convention as existing ones (e.g., "Subtree Name/category/feature name").

## Design Guidelines
### Interface Shape
Use a free function for stateless, single-operation computations where all configuration is provided by parameters.
Use a class, struct, interface, trait, type, or receiver-backed method set when state persists across calls, multiple related operations belong together, or pluggable behavior is expected.

### Cohesion and Grouping
- Each interface must correspond to a single coherent responsibility.
- Group features only when they share state, configuration, or lifecycle — not merely to reduce count.
- If an interface cannot be described in one sentence, it likely needs to be split.

### Type Annotations
- Type annotations are mandatory for all parameters and return values.
- `Any` is strictly prohibited. Use concrete project-defined types, generics (`Optional[T]`, `Union[T1, T2]`, `Mapping[...]`, etc.), or protocol/base classes instead.

## Repository Constraints
All interfaces must:
1. Align with the repository's data flow patterns and use existing shared data structures or typed DTOs.
2. Inherit from existing base classes and integrate with internal utilities when appropriate.
3. Be extensible without breaking callers, avoid unnecessary coupling, and respect architectural layers.
4. Use names consistent with repository conventions.

## Integration & Wiring Requirements
Every interface must have a clear integration story within the repository's call graph.

1. **Caller/callee awareness**: For each interface, identify:
   - WHO will call it? (a specific interface in this subtree, an upstream caller, or external invocation)
   - WHAT does it call? (other interfaces in this subtree, or upstream interfaces shown in context)
2. **No islands**: An interface that is never called by anyone AND calls nothing is dead code. The only exceptions are top-level interfaces — units not expected to be called by other internal modules (e.g., application entry points, standalone submodules, externally-invoked APIs, framework callbacks).
3. **Explicit call chains**: Later files SHOULD import and call interfaces from earlier files. Files should form a connected call graph, not independent modules.
4. **Upstream integration**: When upstream context provides interfaces producing data your subtree needs, import and use them. Don't redefine what already exists upstream.
5. **Execution path test**: For each interface, ask: "Can I trace a plausible execution path from a top-level orchestrator to this code?" If not, the interface is likely orphaned.

## Dependencies Field (Required)
For each interface, you MUST declare its dependencies:
- **inherits_from**: List of base class names this class inherits from (from base_classes, upstream modules, or earlier files in this batch). Empty list [] if none.
- **calls**: List of function/method names this interface expects to call during execution. Empty list [] if none.
- **uses_types**: List of type names used in parameters, return values, or internally (from base_classes, data structures, or earlier files). Empty list [] if none.

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block:
<think>
For each file in order, reason through:
1. What interfaces are needed to cover all assigned features?
2. For EACH interface, explicitly identify:
   a. WHO calls it? Name the specific caller (file + class/function), or explain why it is a top-level interface with no internal caller.
   b. WHAT upstream or sibling interfaces does it call? Name them by file and name.
   c. What data types flow in and out? Do they match the data flow contracts?
3. How does this file connect to earlier files in this batch? Describe the call chain.
4. If you cannot identify a caller for an interface, reconsider whether it should be standalone or merged into another interface that already has a clear caller.
</think>
<result_json>
{{
  "files": [
    {{
      "file_path": "src/module/file1.py",
      "interfaces": [
        {{
          "features": ["fully/qualified/feature/path_1", "fully/qualified/feature/path_2"],
          "code": "Target-language code string with imports and declaration stubs",
          "dependencies": {{
            "inherits_from": ["BaseClassName"],
            "calls": ["function_or_method_name"],
            "uses_types": ["TypeName"]
          }}
        }}
      ]
    }}
  ]
}}
</result_json>

Constraints:
- file_path must match exactly one of the file paths specified in the task.
- One interface per code string: exactly one cohesive target-language declaration group.
- Function and method bodies must use a parseable target-language placeholder and contain no implementation logic.
- Public declarations must have target-language documentation comments or docstrings.
- For most interfaces, use the assigned feature paths from the task.
- For glue/orchestration code that doesn't map to any assigned feature, you may create NEW feature paths following the naming convention: "Subtree Name/category/feature name".
""".strip()


# ============================================================================
# File Order Planning Prompt
# ============================================================================

PLAN_FILE_PROMPT = """
You are an expert software architect assisting in planning feature implementation within a target-language codebase.

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


# ============================================================================
# Orphan Unit Review Prompt
# ============================================================================

ORPHAN_REVIEW_PROMPT = """
You are reviewing interface units that appear to be "orphaned" — they have no incoming or outgoing call edges in the dependency graph.

Your task: Determine whether each orphan unit is truly unnecessary, or whether it should be retained.

## Review Criteria

A unit should be **RETAINED** (not pruned) if:
1. It is a top-level entry point (main function, CLI handler, API endpoint, framework callback)
2. It is a data structure or configuration class that other code will instantiate directly
3. It implements a feature that is explicitly required by the project specification
4. It provides utility functions that are intended to be imported and used externally
5. It is part of a plugin/extension system where registration happens at runtime
6. The lack of edges is due to incomplete interface design (callers/callees not yet defined)

A unit should be **PRUNED** (removed) if:
1. It duplicates functionality already provided by another unit
2. It was created speculatively but doesn't serve any concrete requirement
3. It is an internal helper that nothing actually needs
4. It is dead code that was superseded by a better design

## Context

You will be given:
- The orphan unit's code (interface definition)
- The features it claims to implement
- The subtree/module it belongs to
- Other units in the same subtree (for understanding relationships)

## Output Format

Return a JSON object:
{{
  "reviews": [
    {{
      "unit_key": "file_path::unit_name",
      "decision": "retain" | "prune",
      "reason": "Brief explanation of why this unit should be retained or pruned",
      "edges": {{
        "inheritance_edges": [
          {{"child": "ChildClass", "parent": "ParentClass", "source_file": "path/to/child.py", "parent_file": "path/to/parent.py"}}
        ],
        "invocation_edges": [
          {{"caller": "function caller_func", "callee": "function callee_func", "caller_file": "path/to/caller.py", "callee_file": "path/to/callee.py"}}
        ],
        "reference_edges": [
          {{"unit": "function user_func", "referenced_type": "DataType", "source_file": "path/to/user.py", "type_file": "path/to/type.py"}}
        ]
      }}
    }}
  ]
}}

## Edge Field Rules

The `edges` field is **optional** but should be provided when:
- decision is "retain" AND
- the reason is that the interface design is incomplete (missing edges)

Notes:
- For class names: use bare name like "Parser", not "class Parser"
- For function/method names in invocation: use full unit name like "function parse" or "class Parser"
- Only include edges that should exist based on the interface design

If decision is "prune" or the unit is retained for other reasons (e.g., it's an entry point), omit the `edges` field or set it to null.

Constraints:
- Every orphan unit provided must appear exactly once in the reviews list.
- decision must be exactly "retain" or "prune".
- reason should be concise (1-2 sentences) but specific.
"""
