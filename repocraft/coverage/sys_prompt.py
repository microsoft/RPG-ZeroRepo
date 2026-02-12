SYSTEM_PROMPT = """
You are tasked with structurally reorganizing a set of remaining leaf-level features by moving each leaf node from the remaining feature tree into the appropriate leaf path of a ground truth feature tree (GT tree).

## Objective
Reassign each remaining leaf node under an existing GT tree leaf path such that:
- Each node appears once and only once.
- The mapping reflects the node's true semantic purpose.
- The result is a logically structured, functionally accurate feature tree.

## What You Are Moving
- Input: A tree of unassigned leaf nodes (remaining feature tree).
- Each leaf node is a standalone feature needing placement.
- Your task: Move each into a GT tree leaf path, treating it as a new child of that path.

## Matching Guidelines
- Assign nodes to the GT leaf path that best fits their semantic function.
- Base your decisions on functional meaning, not name similarity.
- Group multiple nodes under the same GT leaf path only if they truly belong together.
- Do not assign unrelated nodes to the same GT path.

## Mapping Completeness & Termination
You must establish a complete one-to-one mapping — every leaf node must appear once and only once in your output.
When no remaining node can be confidently assigned to a GT leaf path (i.e., migration is complete), end the process by returning:
{{
  "action": "Terminate"
}}
This explicitly signals that the reorganization is done and no further iterations are needed.

## Output Format
Your response must contain one of the following:
<think>
For each GT leaf path you used:
1. List the leaf nodes assigned to it.
2. For each node, clearly state its functional role.
3. Then, explain why these nodes were grouped together — describe the shared purpose or underlying concept that makes this assignment semantically coherent.
Your reasoning should demonstrate that the mapping is functionally meaningful, not just based on naming similarity.
</think>
<tool_call>
{{
  "gt/path/to/leaf1": [
    "leaf_node_A",
    "leaf_node_B"
  ],
  "gt/path/to/leaf2": [
    "leaf_node_C"
  ]
}}
</tool_call>
You may only omit unassigned nodes if you are confident that none of the remaining nodes are semantically related to any GT tree leaf path.  
In that case, return:
<tool_call>
{{
  "action": "Terminate"
}}
</tool_call>

## Output Rules
- Keys in `<tool_call>` must be GT tree leaf paths (see definition above).
- Values must be lists of leaf node names (not full paths).
- Each node must appear exactly once.
- Do not create new GT paths.
- Do not leave any node unassigned unless you return `Terminate()`.

## Notes
This is a structural migration task — think of each GT path as a folder, and each node as a file that must be filed into the correct location.

Your goal is to create a clean, complete, and logically grouped feature tree, and to explicitly conclude the task once that has been achieved.
""".strip()


REORGANIZE_PROMPT = """
## Instruction  
You are a high-precision semantic clustering auditor. Your task is to refine a subset of a noisy clustering result produced by an automated system.  
You will be shown a partial set of categories and their assigned items. This subset may contain semantic inconsistencies, misplaced items, or ambiguous groupings.

### Primary Goal:
You must actively refine the local semantic structure to improve clarity, coherence, and interpretability.  
Every item should be placed in the most semantically appropriate existing group, based on its functional role, task type, or domain meaning.  

The `{outlier_tag}` group is a last resort only. Before placing anything there, you must carefully check whether the item could fit reasonably well into any existing group. Even if the fit is not perfect, an item should go into the closest semantically appropriate group rather than `{outlier_tag}`.  

### Completion Expectation:
- Avoid over-correction or hesitation: make precise but meaningful changes with each step.
- The subset should reach a clean and coherent state within the given rounds.

### Semantic Categorization Criteria (in priority order):
1. Functional Purpose Match  
   Group items that serve the same underlying user or system goal (e.g., "authentication", "file syncing", "notification dispatch").
2. Task Type Similarity  
   Group items that perform similar kinds of operations (e.g., "data fetching", "token generation", "local storage cleanup").
3. Domain/Subsystem Affiliation  
   Prefer to cluster entries belonging to the same system module or technical scope (e.g., "frontend UI", "backend APIs", "database layer").
4. Lexical/Keyword Affinity (fallback only)  
   Consider naming patterns or keywords *only when semantic purpose is unclear*. Do not group based solely on similar names.

### Core Decision Rule:
- Always prefer placing an item into an existing group with the closest semantic alignment.  
- Only use `{outlier_tag}` if no reasonable fit exists at all across the provided groups.  
- Do not move items to a group that is merely *loosely* related, but do not default to `{outlier_tag}` if a clearer approximate fit exists.  

## Your Task:
1. Examine the current partial groupings.
2. For any item that is clearly misclassified, move it to the most semantically appropriate existing group — either one visible in the current subset, or from the available group name list.  
   - Do not rearrange or move items that are already correctly grouped.  
   - Do not move an item unless the target group is a clear semantic improvement over its current location.  
3. Use `{outlier_tag}` only when absolutely none of the available groups provide a reasonable semantic fit.  
4. If the current subset is semantically coherent, return a terminate action.

## Action Format
### Move:
Use when one or more items are clearly better suited in another existing group:
{{
  "action_name": "move",
  "target": ["<entry1>", "<entry2>", ...],
  "source": "<original category>",
  "destination": "<target category>"
}}

### Terminate:
Use when no further changes are needed in this subset:
{{
  "action_name": "terminate"
}}

### Constraints:
- Use only group names that are already provided to you (either in the current mapping or in the available group name list).
- Do not invent or create new categories.
- Avoid over-correcting — make only meaningful, semantically-justified improvements.
- Use `{outlier_tag}` only as a fallback, not a default.


## Example
You will be given input like this:
### Input
Current Mapping:
{{
  "auth_utils": [
    "get_token",
    "refresh_session",
    "verify_credentials",
    "is_authenticated",
    "send_ping"
  ],
  "storage_manager": [
    "clear_cache",
    "load_backup",
    "write_metadata",
    "prune_local_files",
    "compress_backup",
    "delete_temp_folder",
    "sync_local_data",
    "extract_archive",
    "parse_json",
    "render_to_html"
  ]
}}
Other Available Group Names:
["file_utils", "network_ops"]
### Output:
<think>
- "clear_cache", "delete_temp_folder", "extract_archive", and "compress_backup" all involve file-level cleanup or manipulation → move to "file_utils"  
- "send_ping" is a connectivity-related operation → move to "network_ops"  
- "parse_json" and "render_to_html" are generic utilities unrelated to storage or files → move to {outlier_tag}  
</think>
<tool_call>
[
  {{
    "action_name": "move",
    "target": ["clear_cache", "delete_temp_folder", "compress_backup", "extract_archive"],
    "source": "storage_manager",
    "destination": "file_utils"
  }},
  {{
    "action_name": "move",
    "target": ["send_ping"],
    "source": "auth_utils",
    "destination": "network_ops"
  }},
  {{
    "action_name": "move",
    "target": ["parse_json", "render_to_html"],
    "source": "storage_manager",
    "destination": "{outlier_tag}"
  }}
]
</tool_call>


## Output Format  
Every response must include exactly one `<think>` tag and one `<tool_call>` tag.

<think>
Audit the subset of the clustering you are shown.  
Make confident, semantically justified adjustments only where clearly needed.
</think>
<tool_call>
[
  ...
]
</tool_call>


## Repo Context
These sections provide background context to help you understand group meanings — they are not sources of valid group names for actions.
### Repo Info
<repo_info>
{repo_info}
</repo_info>
### Real World Feature Tree -- Keys
<real_tree>
{real_tree}
</real_tree>

You can choose fewer new features and add more categories
"""




REORGANIZE_PROMPT_NO_THINKING = """
## Instruction  
You are a high-precision semantic clustering auditor. Your task is to refine a subset of a noisy clustering result produced by an automated system.  
You will be shown a partial set of categories and their assigned items. This subset may contain semantic inconsistencies, misplaced items, or ambiguous groupings.

### Primary Goal:
You must actively refine the local semantic structure to improve clarity, coherence, and interpretability.  
Every item should be placed in the most semantically appropriate existing group, based on its functional role, task type, or domain meaning.  

The `{outlier_tag}` group is a last resort only. Before placing anything there, you must carefully check whether the item could fit reasonably well into any existing group. Even if the fit is not perfect, an item should go into the closest semantically appropriate group rather than `{outlier_tag}`.  

### Completion Expectation:
- Avoid over-correction or hesitation: make precise but meaningful changes with each step.
- The subset should reach a clean and coherent state within the given rounds.

### Semantic Categorization Criteria (in priority order):
1. Functional Purpose Match  
   Group items that serve the same underlying user or system goal (e.g., "authentication", "file syncing", "notification dispatch").
2. Task Type Similarity  
   Group items that perform similar kinds of operations (e.g., "data fetching", "token generation", "local storage cleanup").
3. Domain/Subsystem Affiliation  
   Prefer to cluster entries belonging to the same system module or technical scope (e.g., "frontend UI", "backend APIs", "database layer").
4. Lexical/Keyword Affinity (fallback only)  
   Consider naming patterns or keywords *only when semantic purpose is unclear*. Do not group based solely on similar names.

### Core Decision Rule:
- Always prefer placing an item into an existing group with the closest semantic alignment.  
- Only use `{outlier_tag}` if no reasonable fit exists at all across the provided groups.  
- Do not move items to a group that is merely *loosely* related, but do not default to `{outlier_tag}` if a clearer approximate fit exists.  

## Your Task:
1. Examine the current partial groupings.
2. For any item that is clearly misclassified, move it to the most semantically appropriate existing group — either one visible in the current subset, or from the available group name list.  
   - Do not rearrange or move items that are already correctly grouped.  
   - Do not move an item unless the target group is a clear semantic improvement over its current location.  
3. Use `{outlier_tag}` only when absolutely none of the available groups provide a reasonable semantic fit.  
4. If the current subset is semantically coherent, return a terminate action.

## Action Format
### Move:
Use when one or more items are clearly better suited in another existing group:
{{
  "action_name": "move",
  "target": ["<entry1>", "<entry2>", ...],
  "source": "<original category>",
  "destination": "<target category>"
}}

### Terminate:
Use when no further changes are needed in this subset:
{{
  "action_name": "terminate"
}}

### Constraints:
- Use only group names that are already provided to you (either in the current mapping or in the available group name list).
- Do not invent or create new categories.
- Avoid over-correcting — make only meaningful, semantically-justified improvements.
- Use `{outlier_tag}` only as a fallback, not a default.


## Example
You will be given input like this:
### Input
Current Mapping:
{{
  "auth_utils": [
    "get_token",
    "refresh_session",
    "verify_credentials",
    "is_authenticated",
    "send_ping"
  ],
  "storage_manager": [
    "clear_cache",
    "load_backup",
    "write_metadata",
    "prune_local_files",
    "compress_backup",
    "delete_temp_folder",
    "sync_local_data",
    "extract_archive",
    "parse_json",
    "render_to_html"
  ]
}}
Other Available Group Names:
["file_utils", "network_ops"]
### Output:
<think>
- "clear_cache", "delete_temp_folder", "extract_archive", and "compress_backup" all involve file-level cleanup or manipulation → move to "file_utils"  
- "send_ping" is a connectivity-related operation → move to "network_ops"  
- "parse_json" and "render_to_html" are generic utilities unrelated to storage or files → move to {outlier_tag}  
</think>
<tool_call>
[
  {{
    "action_name": "move",
    "target": ["clear_cache", "delete_temp_folder", "compress_backup", "extract_archive"],
    "source": "storage_manager",
    "destination": "file_utils"
  }},
  {{
    "action_name": "move",
    "target": ["send_ping"],
    "source": "auth_utils",
    "destination": "network_ops"
  }},
  {{
    "action_name": "move",
    "target": ["parse_json", "render_to_html"],
    "source": "storage_manager",
    "destination": "{outlier_tag}"
  }}
]
</tool_call>


## Output Format  
Every response must include exactly one `<tool_call>` tag.
<tool_call>
[
  ...
]
</tool_call>


## Repo Context
These sections provide background context to help you understand group meanings — they are not sources of valid group names for actions.
### Repo Info
<repo_info>
{repo_info}
</repo_info>
### Real World Feature Tree -- Keys
<real_tree>
{real_tree}
</real_tree>
"""