#!/usr/bin/env python3
"""Skeleton Prompts.

This module contains professional prompts for skeleton generation.

Key prompts:
- RAW_SKELETON: Design initial directory structure
- GROUP_SKELETON: Assign features to files
"""

# ============================================================================
# Raw Skeleton Generation Prompt
# ============================================================================

RAW_SKELETON_PROMPT = """You are a repository architect responsible for designing the initial project structure of a software repository in its early development stage.

You will be provided with:
- A summary describing the repository's purpose, domain, and scope.
- A list of functional components, each representing a major grouping within the repository.

Your task is to propose a clean, modular file-system skeleton that organizes the repository into appropriate top-level folders.

## Requirements
1. The structure must clearly separate each functional component and reflect logical domain boundaries.
2. Folder names must be concise, meaningful, and follow the target language's naming conventions.
3. Component names serve as functional descriptions, not required folder names.
   - Rename folders as needed for clarity and readability.
   - Include a mapping from folder names to the original component names.
4. You may choose a flat layout (folders at root) or a nested layout (e.g., under `src`) depending on what best enhances clarity, maintainability, and scalability.
5. Include standard auxiliary folders when appropriate, such as:
   - `utils` — shared utilities
   - `tests` — test code
   - `docs` — documentation
   - `configs`, `scripts` — configuration and automation assets
6. Avoid unnecessary complexity or deep nesting. The structure should be intuitive and developer-friendly.

## Naming Guidelines
- Use short, semantically precise names that clearly indicate a folder's purpose.
- Do not reuse component names verbatim; translate them into practical module or folder names.
- Avoid vague names such as `module`, `misc`, `feature1`, or `temp`.

## Output Format
Your response must contain exactly one <think> block and exactly one <result_json> block:

<think>
Your internal reasoning and drafts—treat this like architectural scratch work.
Feel free to explore options, debate trade-offs, sketch out intermediate designs, or work step-by-step until you're confident in your final direction.
</think>

<result_json>
{
  "assignments": [
    {
      "component_name": "component1",
      "directory_path": "src/project/area",
      "reasoning": "Brief explanation for this assignment"
    }
  ],
  "overall_reasoning": "Overall design rationale"
}
</result_json>"""

# ============================================================================
# Group Skeleton Generation Prompt
# ============================================================================

GROUP_SKELETON_PROMPT = """You are a repository architect responsible for incrementally assigning features from a functional component into a production-grade target-language repository structure.

Your primary goals are clarity, modularity, and long-term maintainability. The resulting layout should resemble a modern, well-structured repository in the target language rather than a direct projection of the feature tree.

You may:
- Group related features into shared modules,
- Introduce or adjust folders when semantically appropriate,
- Refine or reorganize previous design decisions as needed.

Your task is to assign each feature to a target-language source file path that:
- Begins with the designated folder,
- Groups semantically related features together (even if they originate from different branches of the feature tree),
- Reflects realistic target-language module/package organization,
- Uses folders where helpful to express higher-level structure.

## Rules
- Assign each feature exactly once (no duplication or omission).
- Only assign leaf-level features.
- All assigned files must reside under the designated folder in a coherent, maintainable structure.

### File and Folder Structure
- Group features into meaningful modules based on real-world development practices, not solely on the original tree layout.
- By default, prefer placing a reasonably large set of closely related features in the same file rather than splitting into many very small files.
- Aim for moderate-to-substantial module sizes when possible: for cohesive groups, assigning on the order of 10–100 leaf features per file is acceptable if they share a clear theme.
- Use single-feature files only for particularly complex, central, or architecturally significant features.
- Keep files reasonably scoped; avoid mixing clearly unrelated features in the same module.
- When a folder becomes crowded, introduce semantically meaningful subfolders rather than scattering features into many tiny modules.

### Naming and Organization Guidelines
1. Use clear, concise, semantically meaningful names that follow target-language conventions. Each file or folder should represent a well-scoped functional area.
2. Names should reflect functional purpose without redundancy. Avoid repeating folder context in filenames when it is obvious.
3. Avoid vague or purely placeholder names such as `module_part1` or `other_module`.
4. Utility-style modules are allowed when they are clearly scoped. Examples:
   - `vector_utils`, `io_utils`, or `text_utils` (with the target language's file extension) inside appropriately named folders,
   - a `util`/`utils` module within a well-defined domain folder, where the utility code is narrowly focused on that domain.
   These should not become unbounded catch-all modules.
5. It is acceptable to place features originating from multiple original subtrees into the same file if they form a coherent functional unit in the repository architecture.

## Output Format
Your response must include exactly one `<think>` block and exactly one `<result_json>` block, and you **MUST** follow the structure below:

<think>
Internal reasoning and drafts — use this area for exploration, planning, and structural considerations.
</think>

<result_json>
{
  "assignments": [
    {
      "file_path": "src/project/component/module.ext",
      "features": ["feature1", "feature2"],
      "purpose": "Brief description of file purpose"
    }
  ]
}
</result_json>"""

# ============================================================================
# Review Prompts
# ============================================================================

RAW_SKELETON_REVIEW_PROMPT = """You are a senior reviewer responsible for evaluating a proposed raw project skeleton for a software repository. Your goal is to verify that the directory layout forms a clean, scalable, and well-structured foundation aligned with the provided functional subtrees.

## Review Objective
Assess the skeleton across four dimensions and provide detailed, actionable, category-specific feedback.

## Evaluation Dimensions
1. The structure should demonstrate thoughtful functional grouping rather than a direct 1:1 mapping from each subtree, with clear opportunities for consolidation or abstraction.
2. Lightweight or utility-style bridging components should be placed appropriately without unnecessary nesting or over-isolation.
3. Each subtree should appear exactly once in the structure, without duplication, fragmentation, or ambiguous ownership.
4. Folder names should be clear, specific, consistent, and aligned with common software naming conventions.

## Output Format
Return **only** a valid JSON object in the following format:
{
  "review": {
    "Functional Grouping": {
      "feedback": "<Your critical feedback here>",
      "pass": true/false
    },
    "Simplified Bridging Components": {
      "feedback": "<Your evaluation here>",
      "pass": true/false
    },
    "Exclusive Assignment": {
      "feedback": "<Your evaluation here>",
      "pass": true/false
    },
    "Semantic Naming": {
      "feedback": "<Your evaluation here>",
      "pass": true/false
    }
  },
  "final_pass": true/false
}

Rules:
- `final_pass` should be `true` only if all four dimensions pass, or if remaining issues are minor and easily fixable.
- All `feedback` fields must provide concrete, actionable guidance.
- Do not add new fields or categories beyond the four listed.
"""

GROUP_SKELETON_REVIEW_PROMPT = """You are a senior software architect reviewing the feature-to-file assignments proposed by an architecture assistant. Your role is to critically evaluate the structural quality of the resulting target-language module layout across the five criteria below.

## Review Criteria
### 1. File Scope Appropriateness
- Each file must have a clear, focused responsibility.
- Group features only when they share meaningful semantic or functional alignment.
- Split files when they accumulate unrelated logic or become overloaded.
- Complex features generally merit isolation; simple, tightly related ones may be grouped.

### 2. File Structure Organization
- The folder hierarchy should reflect clean separations of concern and meaningful domain boundaries.
- Introduce subfolders when a directory becomes crowded or mixes distinct types of functionality.
- Avoid excessively flat or deeply nested layouts.
- Detect filename clusters with shared prefixes and organize them into subfolders; avoid redundant naming (e.g., a `nlp/` folder whose files repeat the `nlp_` prefix).

### 3. Modularity & Cohesion
- Modules should exhibit strong internal cohesion and minimal coupling.
- Each module should map to a single clear abstraction.
- Flag mixed-purpose, catch-all, or poorly scoped modules for redesign.
### 4. Naming Quality
- Names must be clear, concise, meaningful, and consistent with target-language naming conventions.
- Avoid redundancy between folder and file names.
- Reject vague, generic, placeholder, or suffix-based names.
- Prefer succinct, expressive names that accurately reflect functionality.

### 5. Structural Soundness & Scalability
- The architecture should support clean layering (data, logic, interface) and long-term scalability.
- Shared logic should be abstracted into appropriate modules.
- Avoid structural bottlenecks, overloaded directories, or ambiguous boundaries.

### Special Emphasis
- Apply strict scrutiny to both naming and structural decisions.
- Placeholder or incremental naming patterns must be rejected.
- When flagging an issue, always recommend specific, meaningful alternatives.

## Output Format
Return **only valid JSON**, with no extra comments or text:
{
  "review": {
    "File Scope Appropriateness": {
      "feedback": "<Your detailed feedback here>",
      "pass": true/false
    },
    "File Structure Organization": {
      "feedback": "<Your detailed feedback here>",
      "pass": true/false
    },
    "Modularity & Cohesion": {
      "feedback": "<Your detailed feedback here>",
      "pass": true/false
    },
    "Naming Quality": {
      "feedback": "<Your detailed feedback here>",
      "pass": true/false
    },
    "Structural Soundness": {
      "feedback": "<Your detailed feedback here>",
      "pass": true/false
    }
  },
  "final_pass": true/false
}"""

# ============================================================================
# Utility Functions
# ============================================================================

def build_component_summary(components: list) -> str:
    """Build formatted component summary for prompts."""
    summary_lines = []
    for i, comp in enumerate(components, 1):
        name = comp.get("name", f"Component {i}")
        desc = comp.get("description", "No description")

        # Count features in refactored_subtree
        feature_count = count_features_in_subtree(comp.get("refactored_subtree", {}))

        summary_lines.append(f"{i}. **{name}**")
        summary_lines.append(f"   Description: {desc}")
        summary_lines.append(f"   Features: {feature_count}")
        summary_lines.append("")

    return "\n".join(summary_lines)


def count_features_in_subtree(subtree) -> int:
    """Count total features in a component's subtree."""
    if isinstance(subtree, dict):
        total = 0
        for key, value in subtree.items():
            if key == "description":
                continue
            total += count_features_in_subtree(value)
        return total
    elif isinstance(subtree, list):
        return len([item for item in subtree if item])
    else:
        return 1 if subtree else 0


def extract_features_from_subtree(subtree, prefix=""):
    """Extract all feature paths from a subtree structure."""
    features = []

    if isinstance(subtree, dict):
        for key, value in subtree.items():
            if key == "description":
                continue

            current_path = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                # Check if this is just a description wrapper
                if set(value.keys()) == {"description"}:
                    # This is a leaf feature with only description metadata
                    features.append(current_path)
                else:
                    # Nested structure - extract sub-features with full path
                    features.extend(extract_features_from_subtree(value, current_path))
            elif isinstance(value, list):
                # List of leaf features - each item gets full path
                for item in value:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        if name:
                            features.append(f"{current_path}/{name}")
                    elif item:
                        features.append(f"{current_path}/{item}")
            else:
                # Single feature value - this is a leaf node
                if value:
                    # If value is the same as key, it means this is a leaf feature
                    if isinstance(value, str) and value == key:
                        features.append(current_path)
                    else:
                        # Otherwise it's a nested feature
                        features.append(current_path)

    elif isinstance(subtree, list):
        for item in subtree:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    feature_path = f"{prefix}/{name}" if prefix else name
                    features.append(feature_path)
            elif item:
                feature_path = f"{prefix}/{item}" if prefix else str(item)
                features.append(feature_path)
    else:
        # This is a leaf feature - use the current prefix as the full path
        if subtree:
            features.append(prefix if prefix else str(subtree))

    return features


def extract_leaf_descriptions_from_subtree(subtree, prefix=""):
    """Extract descriptions from dict-format leaf nodes in a subtree.

    Returns:
        Dict mapping full feature paths to their descriptions
    """
    descriptions = {}
    if isinstance(subtree, dict):
        for key, value in subtree.items():
            if key == "description":
                continue
            current_path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if set(value.keys()) != {"description"}:
                    descriptions.update(extract_leaf_descriptions_from_subtree(value, current_path))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        desc = item.get("description", "")
                        if name and desc:
                            descriptions[f"{current_path}/{name}"] = desc
    elif isinstance(subtree, list):
        for item in subtree:
            if isinstance(item, dict):
                name = item.get("name", "")
                desc = item.get("description", "")
                if name and desc:
                    path = f"{prefix}/{name}" if prefix else name
                    descriptions[path] = desc
    return descriptions


def format_feature_list(features: list, descriptions: dict = None) -> str:
    """Format feature list for prompts, optionally with descriptions."""
    if not features:
        return "No features found"

    formatted_lines = []
    for i, feature in enumerate(features, 1):
        desc = descriptions.get(feature, "") if descriptions else ""
        if desc:
            formatted_lines.append(f"{i}. {feature}: {desc}")
        else:
            formatted_lines.append(f"{i}. {feature}")

    return "\n".join(formatted_lines)


if __name__ == "__main__":
    # Test prompt utilities
    test_component = {
        "name": "parser",
        "description": "Text parsing functionality",
        "refactored_subtree": {
            "tokenizer": ["tokenize_text", "handle_whitespace"],
            "validator": {
                "syntax": ["check_syntax", "report_errors"],
                "semantic": ["validate_meaning"]
            }
        }
    }

    features = extract_features_from_subtree(test_component["refactored_subtree"])
    print("Extracted features:")
    print(format_feature_list(features))

    print(f"\nFeature count: {count_features_in_subtree(test_component['refactored_subtree'])}")