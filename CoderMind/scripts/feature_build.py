#!/usr/bin/env python3
"""Feature Tree Expansion Script Uses AI Assistant CLI tool instead of API calls."""

import copy
import json
import argparse
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from feature.prompts import (
    PROMPT_TEMPLATE_BUILD_REVIEW,
    PROMPT_TEMPLATE_BUILD_FEATURE,
    PROMPT_TEMPLATE_BUILD_EXPAND,
    PROMPT_TEMPLATE_BUILD_DIRECTED,
    PROMPT_TEMPLATE_SUGGEST_DIRECTIONS,
)
from common.paths import (
    FEATURE_SPEC_FILE,
    FEATURE_BUILD_FILE,
)
from common import print_unicode_table, get_all_leaf_paths, get_leaf_name, get_all_leaf_descriptions
from common.llm_client import LLMClient
from common.language_meta import extract_language_metadata, metadata_with_languages
from common.trajectory import load_or_create_trajectory

# ======================== Configuration ========================

MAX_ITERATIONS = 20  # Hard safety cap for both Step 1 and Step 2
MAX_CONSECUTIVE_FAILURES = 3  # Break after N consecutive empty/error responses

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======================== Pydantic Models ========================


class AddPathsOutput(BaseModel):
    """Output model for adding new feature paths."""

    add_new_feature_paths: List[Dict[str, str]] = Field(
        description="List of new feature paths to add, each with 'path' and 'description'",
        default_factory=list,
    )
    is_complete: bool = Field(
        description="Whether the feature tree is sufficiently complete for the current objective",
        default=False,
    )
    completion_reason: str = Field(
        description="Reason for declaring completion",
        default="",
    )


class ReviewOutput(BaseModel):
    """Output model for review results - combines gap analysis and suggested paths."""

    coverage_percentage: float = Field(
        description="Estimated percentage of FRD requirements covered by feature tree (0-100)",
        ge=0,
        le=100,
    )
    has_gaps: bool = Field(
        description="Whether there are uncovered functional requirements"
    )
    missing_functionalities: List[str] = Field(
        description="List of functional requirements from FRD that are not yet covered",
        default_factory=list,
    )
    suggested_paths: List[Dict[str, str]] = Field(
        description="List of new feature paths to cover the missing functionalities, each with 'path' and 'description'",
        default_factory=list,
    )
    invalid_leaf_nodes: List[str] = Field(
        description="List of leaf nodes that violate the Minimum Implementable Unit principle",
        default_factory=list,
    )
    suggested_replacements: List[Dict[str, str]] = Field(
        description="List of replacement paths for invalid leaf nodes (split into proper MIU), each with 'path' and 'description'",
        default_factory=list,
    )
    duplicate_leaf_renames: List[str] = Field(
        description="List of rename operations for duplicate leaf names, format: 'old_full_path -> new_leaf_name'",
        default_factory=list,
    )


class DirectionItem(BaseModel):
    """A single expansion direction suggestion."""

    name: str = Field(description="Short direction name")
    description: str = Field(
        description="2-3 sentence description of what this direction covers"
    )
    rationale: str = Field(
        description="Why this direction is important for the repository"
    )


class SuggestDirectionsOutput(BaseModel):
    """Output model for expansion direction suggestions."""

    directions: List[DirectionItem] = Field(
        description="List of 4-6 expansion direction suggestions",
        default_factory=list,
    )


# ======================== Utility Functions ========================


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {path}, error: {e}")
        sys.exit(1)


def save_json(data: Dict[str, Any], path: Path):
    """Save data to JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {path}, error: {e}")
        sys.exit(1)


def build_repo_info(repo_data: Dict[str, Any]) -> str:
    """Build repository information string.

    Extracts repo metadata and spec fields. If background_and_overview or
    functional_requirements are not at the top level, they are parsed from
    the repository_specification JSON string (which contains the full
    feature_spec.json content).
    """
    # Build a merged view: top-level keys + parsed repository_specification
    merged = dict(repo_data)
    spec_keys = ["background_and_overview", "functional_requirements",
                 "non_functional_requirements"]

    # If spec fields are missing at top level, extract from repository_specification
    if any(k not in merged for k in spec_keys):
        raw_spec = repo_data.get("repository_specification", "")
        if isinstance(raw_spec, str) and raw_spec.strip():
            try:
                parsed_spec = json.loads(raw_spec)
                for k in spec_keys:
                    if k not in merged and k in parsed_spec:
                        merged[k] = parsed_spec[k]
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(raw_spec, dict):
            for k in spec_keys:
                if k not in merged and k in raw_spec:
                    merged[k] = raw_spec[k]

    info = ""
    for key in [
        "repository_name",
        "repository_purpose",
        "background_and_overview",
        "non_functional_requirements",
        "functional_requirements",
    ]:
        if key in merged:
            formatted_key = key.replace("_", " ").capitalize()
            value = merged[key]
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2, ensure_ascii=False)
            info += f"{formatted_key}: {value}\n"
    return info.strip()


def find_duplicate_leaf_names(tree: Dict[str, Any]) -> Dict[str, List[str]]:
    """Find duplicate leaf node names in the feature tree.

    Args:
        tree: Feature tree dictionary

    Returns:
        Dictionary mapping duplicate leaf names to their full paths
    """
    leaf_to_paths: Dict[str, List[str]] = {}
    all_paths = get_all_leaf_paths(tree)

    for path in all_paths:
        leaf_name = path.split("/")[-1]
        if leaf_name not in leaf_to_paths:
            leaf_to_paths[leaf_name] = []
        leaf_to_paths[leaf_name].append(path)

    # Filter to only duplicates (more than one path per leaf name)
    duplicates = {k: v for k, v in leaf_to_paths.items() if len(v) > 1}
    return duplicates


def format_duplicate_leaves_info(duplicates: Dict[str, List[str]]) -> str:
    """Format duplicate leaf information for the prompt.

    Args:
        duplicates: Dictionary from find_duplicate_leaf_names()

    Returns:
        Formatted string for the prompt
    """
    if not duplicates:
        return "No duplicate leaf nodes detected."

    lines = [
        f"Found {len(duplicates)} duplicate leaf name(s) that need to be renamed:\n"
    ]
    for leaf_name, paths in duplicates.items():
        lines.append(f'- Leaf name "{leaf_name}" appears {len(paths)} times:')
        for path in paths:
            lines.append(f"  - {path}")
    lines.append(
        "\nFor each duplicate, keep ONE path unchanged and rename the others to make leaf names unique."
    )
    return "\n".join(lines)


def convert_leaves_to_list(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Convert leaf nodes to lists.

    Rules:
    - Empty dict {} becomes empty list []
    - Non-empty dict is recursively processed
    - List values have whitespace trimmed from string elements
    - Other values are wrapped in a list
    """
    result = {}
    for key, value in tree.items():
        clean_key = key.strip()
        if isinstance(value, dict):
            if len(value) == 0:
                # Empty dict should become empty list (leaf node with no children)
                result[clean_key] = []
            else:
                result[clean_key] = convert_leaves_to_list(value)
        elif isinstance(value, list):
            # Trim whitespace from string elements in list; preserve dict items
            cleaned = []
            for item in value:
                if isinstance(item, str):
                    cleaned.append(item.strip())
                elif isinstance(item, dict):
                    # Dict leaf node: strip the name field
                    stripped = dict(item)
                    if "name" in stripped and isinstance(stripped["name"], str):
                        stripped["name"] = stripped["name"].strip()
                    cleaned.append(stripped)
                else:
                    cleaned.append(item)
            result[clean_key] = cleaned
        else:
            # If value is string, trim and wrap in list
            result[clean_key] = [value.strip() if isinstance(value, str) else value]
    return result


def _list_leaf_to_branch(item: Any) -> tuple[str | None, Any]:
    """Converts a list leaf into a branch entry."""
    if isinstance(item, str):
        return item.strip(), []
    if isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            children = item.get("children", [])
            if isinstance(children, (dict, list)):
                return name.strip(), copy.deepcopy(children)
            return name.strip(), []
        if len(item) == 1:
            key, value = next(iter(item.items()))
            if isinstance(key, str) and key.strip():
                if isinstance(value, (dict, list)):
                    return key.strip(), copy.deepcopy(value)
                return key.strip(), []
    if item is None:
        return None, []
    return str(item).strip(), []


def _list_leaves_to_branch(items: List[Any]) -> Dict[str, Any]:
    """Promotes list leaves to a branch map."""
    branch: Dict[str, Any] = {}
    for item in items:
        key, value = _list_leaf_to_branch(item)
        if key:
            branch.setdefault(key, value)
    return branch


def apply_changes(tree: Dict[str, Any], paths: List[str]) -> Dict[str, Any]:
    """Apply path list to tree structure.

    For a path like "a/b/c/d/leaf", creates:
    {
        "a": {
            "b": {
                "c": {
                    "d": ["leaf"]
                }
            }
        }
    }
    """
    new_tree = copy.deepcopy(tree)

    for path in paths:
        # Trim whitespace from each path segment
        parts = [part.strip() for part in path.split("/")]

        if len(parts) < 2:
            continue  # Skip invalid paths

        # Navigate/create the tree structure up to the second-to-last element
        current = new_tree
        for part in parts[:-2]:
            if part not in current:
                current[part] = {}
            elif isinstance(current[part], list):
                old_list = current[part]
                current[part] = _list_leaves_to_branch(old_list)
            elif not isinstance(current[part], dict):
                # Unexpected type, convert to dict
                current[part] = {}
            current = current[part]

        # Now handle the last two parts: parent and leaf
        parent_key = parts[-2]
        leaf = parts[-1]

        # Ensure parent_key exists in current
        if parent_key not in current:
            # Create a new list with the leaf
            current[parent_key] = [leaf]
        elif isinstance(current[parent_key], list):
            # Append to existing list if not already present
            if leaf not in current[parent_key]:
                current[parent_key].append(leaf)
        elif isinstance(current[parent_key], dict):
            # This path segment is used both as a leaf parent and as a
            # branch. Keep the branch shape and add the leaf key.
            if leaf not in current[parent_key]:
                current[parent_key][leaf] = []
        else:
            # Unexpected type, replace with list
            current[parent_key] = [leaf]

    return new_tree


def remove_paths(tree: Dict[str, Any], paths: List[str]) -> Dict[str, Any]:
    """Remove paths from tree structure.

    For a path like "a/b/c/d/leaf", removes the leaf from the tree.
    If the parent becomes empty after removal, it will be cleaned up.

    Note: The paths in invalid_leaf_nodes may contain explanations like:
    "workflow/user/manage account - too broad and not WHAT-only"
    We need to extract just the path part before the explanation.

    Args:
        tree: Current feature tree
        paths: List of paths to remove (may contain explanations after ' - ')

    Returns:
        Updated tree with paths removed
    """
    new_tree = copy.deepcopy(tree)

    for path in paths:
        # Extract path part (before any explanation marker like " - ")
        path_part = path.split(" - ")[0].strip()

        # Trim whitespace from each path segment
        parts = [part.strip() for part in path_part.split("/")]

        if len(parts) < 2:
            logger.warning(f"Invalid path to remove (too short): {path}")
            continue

        # Navigate to the parent of the leaf
        current = new_tree
        parent_stack = []  # Track parents for cleanup

        # Navigate to the second-to-last element
        valid_path = True
        for part in parts[:-2]:
            if part not in current or not isinstance(current[part], dict):
                logger.warning(f"Path not found in tree: {path} (missing: {part})")
                valid_path = False
                break
            parent_stack.append((current, part))
            current = current[part]

        if not valid_path:
            continue

        # Now handle the last two parts: parent_key and leaf
        parent_key = parts[-2]
        leaf = parts[-1]

        if parent_key not in current:
            logger.warning(f"Parent key not found: {parent_key} in path {path}")
            continue

        parent_stack.append((current, parent_key))

        # Remove the leaf
        if isinstance(current[parent_key], list):
            # Find and remove the leaf, handling both str and dict items
            removed = False
            for idx, item in enumerate(current[parent_key]):
                item_name = get_leaf_name(item)
                if item_name == leaf:
                    current[parent_key].pop(idx)
                    removed = True
                    logger.info(f"Removed leaf '{leaf}' from list at '{parent_key}'")
                    break
            if not removed:
                logger.warning(f"Leaf '{leaf}' not found in list at '{parent_key}'")
        elif isinstance(current[parent_key], dict):
            if leaf in current[parent_key]:
                del current[parent_key][leaf]
                logger.info(f"Removed leaf '{leaf}' from dict at '{parent_key}'")
            else:
                logger.warning(f"Leaf '{leaf}' not found in dict at '{parent_key}'")
        else:
            logger.warning(
                f"Unexpected type at '{parent_key}': {type(current[parent_key])}"
            )
            continue

        # Clean up empty parents (bottom-up)
        for parent, key in reversed(parent_stack):
            if key in parent:
                value = parent[key]
                # Remove if empty list or empty dict
                if (isinstance(value, list) and len(value) == 0) or (
                    isinstance(value, dict) and len(value) == 0
                ):
                    del parent[key]
                    logger.debug(f"Cleaned up empty node: {key}")
                else:
                    break  # Stop cleanup if not empty

    return new_tree


def count_features_recursive(tree: Dict[str, Any]) -> int:
    """Recursively count only leaf features in the tree.

    Args:
        tree: Feature tree or subtree

    Returns:
        Total number of leaf features (items in lists)
    """
    count = 0
    for key, value in tree.items():
        if isinstance(value, dict):
            # Recursively count features in subtree
            count += count_features_recursive(value)
        elif isinstance(value, list):
            # Count leaf features (items in the list)
            count += len(value)
        # Don't count intermediate nodes, only leaf nodes
    return count


def attach_descriptions(tree: Dict[str, Any], desc_map: Dict[str, str], prefix: str = "") -> Dict[str, Any]:
    """Attach descriptions to leaf nodes in the tree.

    Traverses the tree and replaces string leaf items with {"name": ..., "description": ...}
    when a matching description is found in desc_map.

    Args:
        tree: Feature tree dictionary
        desc_map: Mapping of full leaf paths to descriptions
        prefix: Current path prefix (for recursion)

    Returns:
        Updated tree with descriptions attached to leaf nodes
    """
    result = {}
    for key, value in tree.items():
        new_prefix = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result[key] = attach_descriptions(value, desc_map, new_prefix)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                name = get_leaf_name(item)
                path = f"{new_prefix}/{name}" if new_prefix else name
                desc = desc_map.get(path, "")
                if not desc and isinstance(item, dict):
                    # Preserve existing description if no new one
                    desc = item.get("description", "")
                if desc:
                    new_list.append({"name": name, "description": desc})
                else:
                    new_list.append(item)
            result[key] = new_list
        else:
            result[key] = value
    return result


def analyze_tree_statistics(tree: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Analyze feature tree statistics for each top-level category.

    Args:
        tree: Feature tree

    Returns:
        Dictionary with statistics for each category
    """
    stats = {}
    for category, subtree in tree.items():
        if isinstance(subtree, dict):
            feature_count = count_features_recursive(subtree)
            stats[category] = {"total_features": feature_count}
        elif isinstance(subtree, list):
            stats[category] = {"total_features": len(subtree)}
        else:
            stats[category] = {"total_features": 1}
    return stats




def count_paths_by_category(paths: List[str]) -> Dict[str, int]:
    """Count paths by their top-level category.

    Args:
        paths: List of feature paths

    Returns:
        Dictionary mapping category to count
    """
    category_counts = {}
    for path in paths:
        parts = [part.strip() for part in path.split("/")]
        if parts:
            category = parts[0]
            category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts


def print_summary_tables(
    iteration_logs: List[Dict[str, Any]],
    final_tree: Dict[str, Any],
    review_logs: List[Dict[str, Any]] = None,
    previous_tree: Dict[str, Any] = None,
):
    """Print summary tables with Unicode borders.

    Args:
        iteration_logs: List of iteration logs
        final_tree: Final feature tree
        review_logs: List of review logs (optional)
        previous_tree: Initial feature tree before expansion (optional)
    """
    print("\n" + "=" * 70)
    print("FEATURE EXPANSION SUMMARY")
    print("=" * 70)

    # Iteration Summary Table
    iteration_rows = []
    total_paths = 0
    all_new_paths = []

    for log in iteration_logs:
        paths_count = log.get("paths_count", 0)
        total_paths += paths_count
        raw_paths = log.get("new_paths", [])
        extracted, _ = _extract_paths_and_descs(raw_paths)
        all_new_paths.extend(extracted)
        iteration_rows.append([log["iteration"], paths_count])

    # Add total row
    iteration_rows.append(["Total", total_paths])

    print_unicode_table(
        headers=["Iteration", "New Feature Paths Added"], rows=iteration_rows, title=""
    )

    # Collect Review phase paths if available
    review_added_paths = []
    review_removed_paths = []
    if review_logs:
        for log in review_logs:
            if log.get("status") in ("paths_applied", "threshold_met"):
                # threshold_met only applies MIU fixes, not coverage gap paths
                if log.get("status") != "threshold_met":
                    paths, _ = _extract_paths_and_descs(log.get("suggested_paths", []))
                    review_added_paths.extend(paths)
                repl, _ = _extract_paths_and_descs(log.get("suggested_replacements", []))
                review_added_paths.extend(repl)
                review_removed_paths.extend(log.get("invalid_leaf_nodes", []))

    # Module Statistics Table
    # Count initial features (before expansion)
    initial_stats = analyze_tree_statistics(previous_tree) if previous_tree else {}

    # Count new added paths by category (expansion phase only)
    expansion_category_counts = count_paths_by_category(all_new_paths)

    # Count review added paths by category
    review_added_category_counts = count_paths_by_category(review_added_paths)

    # Count review removed paths by category (extract path part before explanation)
    review_removed_clean_paths = []
    for path in review_removed_paths:
        path_part = path.split(" - ")[0].strip()
        review_removed_clean_paths.append(path_part)
    review_removed_category_counts = count_paths_by_category(review_removed_clean_paths)

    # Count total features in final tree
    final_stats = analyze_tree_statistics(final_tree)

    # Get all categories
    all_categories = sorted(
        set(
            list(initial_stats.keys())
            + list(expansion_category_counts.keys())
            + list(review_added_category_counts.keys())
            + list(review_removed_category_counts.keys())
            + list(final_stats.keys())
        )
    )

    module_rows = []
    total_initial = 0
    total_expansion_added = 0
    total_review_added = 0
    total_review_removed = 0
    total_features = 0

    for category in all_categories:
        initial_feat = initial_stats.get(category, {}).get("total_features", 0)
        expansion_added = expansion_category_counts.get(category, 0)
        review_added = review_added_category_counts.get(category, 0)
        review_removed = review_removed_category_counts.get(category, 0)
        total_feat = final_stats.get(category, {}).get("total_features", 0)

        total_initial += initial_feat
        total_expansion_added += expansion_added
        total_review_added += review_added
        total_review_removed += review_removed
        total_features += total_feat

        # Format: Initial | Final | Expansion Added | Review (+added/-removed = net)
        if review_logs:
            if review_added or review_removed:
                net = review_added - review_removed
                net_str = f"+{net}" if net >= 0 else str(net)
                review_change = f"+{review_added}/-{review_removed}={net_str}"
            else:
                review_change = "-"
            module_rows.append(
                [category, initial_feat, total_feat, expansion_added, review_change]
            )
        else:
            module_rows.append([category, initial_feat, total_feat, expansion_added])

    # Add total row
    if review_logs:
        total_net = total_review_added - total_review_removed
        total_net_str = f"+{total_net}" if total_net >= 0 else str(total_net)
        review_total_change = (
            f"+{total_review_added}/-{total_review_removed}={total_net_str}"
        )
        module_rows.append(
            [
                "TOTAL",
                total_initial,
                total_features,
                total_expansion_added,
                review_total_change,
            ]
        )
        headers = [
            "Top-Level Category",
            "Initial",
            "Final",
            "Expansion Added",
            "Review Change",
        ]
    else:
        module_rows.append(
            ["TOTAL", total_initial, total_features, total_expansion_added]
        )
        headers = ["Top-Level Category", "Initial", "Final", "New Added Features"]

    print_unicode_table(
        headers=headers,
        rows=module_rows,
        title="\n",
    )

    # Print detailed new paths grouped by category
    if all_new_paths:
        print("\n   New Feature Paths:")
        grouped: Dict[str, List[str]] = {}
        for path in all_new_paths:
            parts = [p.strip() for p in path.split("/")]
            category = parts[0] if parts else "unknown"
            grouped.setdefault(category, []).append(path)
        for category in sorted(grouped.keys()):
            paths = grouped[category]
            noun = "path" if len(paths) == 1 else "paths"
            print(f"\n   [{category}] ({len(paths)} {noun})")
            for p in paths:
                print(f"     -  {p}")

    print("\n" + "=" * 70)


# ======================== Review Functions ========================


def review_feature_coverage(
    llm: LLMClient,
    repo_info: str,
    current_tree: Dict[str, Any],
    previous_review: Optional[Dict[str, Any]] = None,
    review_iter: int = 0,
) -> Tuple[Optional[ReviewOutput], str, str]:
    """Review feature tree coverage against FRD requirements.

    Uses AI semantic matching to check if all functional requirements
    from the repository information are covered by the feature tree.

    Args:
        llm: LLMClient instance
        repo_info: The repository information content
        current_tree: Current feature tree
        previous_review: Previous review result (to maintain consistency)

    Returns:
        Tuple of (ReviewOutput, think_content, raw_response)
    """
    logger.info("Starting feature coverage review...")

    current_tree_json = json.dumps(current_tree, indent=2) if current_tree else "{}"

    # Detect duplicate leaf names
    duplicates = find_duplicate_leaf_names(current_tree)
    duplicate_leaves_info = format_duplicate_leaves_info(duplicates)
    if duplicates:
        logger.info(f"Detected {len(duplicates)} duplicate leaf name(s)")

    # Format previous review info for the prompt
    if previous_review:
        previous_review_str = f"""Previous coverage: {previous_review.get("coverage_percentage", 0):.1f}%
Previously identified missing functionalities:
{json.dumps(previous_review.get("missing_functionalities", []), indent=2)}
Paths that were added to address gaps:
{json.dumps(previous_review.get("suggested_paths", []), indent=2)}

IMPORTANT: The coverage should NOT decrease. Only the previously missing functionalities need to be re-evaluated."""
    else:
        previous_review_str = (
            "This is the first review iteration. No previous review data."
        )

    prompt = PROMPT_TEMPLATE_BUILD_REVIEW.format(
        repo_info=repo_info,
        current_tree=current_tree_json,
        previous_review=previous_review_str,
        duplicate_leaves_info=duplicate_leaves_info,
    )

    think, result, response = llm.call_structured(
        system_prompt=prompt, user_prompt="",
        response_model=ReviewOutput, max_retries=3,
        purpose=f"review_{review_iter + 1}",
    )

    if result:
        logger.info(f"Review complete: {result.coverage_percentage:.1f}% coverage")
        logger.info(f"Has gaps: {result.has_gaps}")
        if result.missing_functionalities:
            logger.info(
                f"Missing functionalities: {len(result.missing_functionalities)}"
            )
        if result.suggested_paths:
            logger.info(f"Suggested paths: {len(result.suggested_paths)}")
        if result.invalid_leaf_nodes:
            logger.info(
                f"Invalid leaf nodes (MIU violations): {len(result.invalid_leaf_nodes)}"
            )
        if result.suggested_replacements:
            logger.info(f"Suggested replacements: {len(result.suggested_replacements)}")
        if result.duplicate_leaf_renames:
            logger.info(f"Duplicate leaf renames: {len(result.duplicate_leaf_renames)}")

    return result, think, response


def print_review_summary(review_result: ReviewOutput, iteration: int):
    """Print a formatted summary of the review results."""
    print(f"\n{'=' * 60}")
    print(f"REVIEW ITERATION {iteration} SUMMARY")
    print(f"{'=' * 60}")
    print(f"Coverage: {review_result.coverage_percentage:.1f}%")
    print(f"Has Gaps: {review_result.has_gaps}")

    if review_result.missing_functionalities:
        print(
            f"\nMissing Functionalities ({len(review_result.missing_functionalities)}):"
        )
        for i, missing in enumerate(review_result.missing_functionalities[:10], 1):
            print(f"  {i}. {missing[:80]}{'...' if len(missing) > 80 else ''}")
        if len(review_result.missing_functionalities) > 10:
            print(f"  ... and {len(review_result.missing_functionalities) - 10} more")

    if review_result.suggested_paths:
        print(f"\nSuggested Paths ({len(review_result.suggested_paths)}):")
        for i, item in enumerate(review_result.suggested_paths[:10], 1):
            if isinstance(item, dict):
                print(f"  {i}. {item.get('path', '')}: {item.get('description', '')}")
            else:
                print(f"  {i}. {item}")
        if len(review_result.suggested_paths) > 10:
            print(f"  ... and {len(review_result.suggested_paths) - 10} more")

    if review_result.invalid_leaf_nodes:
        print(f"\n[WARNING]  MIU Violations ({len(review_result.invalid_leaf_nodes)}):")
        for i, node in enumerate(review_result.invalid_leaf_nodes[:10], 1):
            print(f"  {i}. {node[:80]}{'...' if len(node) > 80 else ''}")
        if len(review_result.invalid_leaf_nodes) > 10:
            print(f"  ... and {len(review_result.invalid_leaf_nodes) - 10} more")

    if review_result.suggested_replacements:
        print(
            f"\nSuggested Replacements ({len(review_result.suggested_replacements)}):"
        )
        for i, item in enumerate(review_result.suggested_replacements[:10], 1):
            if isinstance(item, dict):
                print(f"  {i}. {item.get('path', '')}: {item.get('description', '')}")
            else:
                print(f"  {i}. {item}")
        if len(review_result.suggested_replacements) > 10:
            print(f"  ... and {len(review_result.suggested_replacements) - 10} more")

    if review_result.duplicate_leaf_renames:
        print(
            f"\nDuplicate Leaf Renames ({len(review_result.duplicate_leaf_renames)}):"
        )
        for i, rename in enumerate(review_result.duplicate_leaf_renames[:10], 1):
            print(f"  {i}. {rename}")
        if len(review_result.duplicate_leaf_renames) > 10:
            print(f"  ... and {len(review_result.duplicate_leaf_renames) - 10} more")

    print(f"{'=' * 60}\n")


# ======================== Helper Functions ========================


def _extract_paths_and_descs(items: List) -> Tuple[List[str], Dict[str, str]]:
    """Extract path strings and description map from AddPathsOutput items.

    Handles both old format (List[str]) and new format (List[Dict[str, str]]).

    Returns:
        Tuple of (path_list, desc_map)
    """
    paths = []
    desc_map = {}
    for item in items:
        if isinstance(item, dict):
            p = item.get("path", "")
            d = item.get("description", "")
            if p:
                paths.append(p)
                if d:
                    desc_map[p] = d
        elif isinstance(item, str):
            paths.append(item)
    return paths, desc_map


def _target_languages(data: Dict[str, Any]) -> List[str]:
    """Returns normalized target language list from feature data."""
    return extract_language_metadata(data)[1]


def _save_intermediate(
    feature_tree: Dict[str, Any],
    current_tree: Dict[str, Any],
    previous_feature_tree: Dict[str, Any],
    iteration_logs: List[Dict[str, Any]],
    review_logs: List[Dict[str, Any]],
    output_file: Path,
):
    """Save intermediate results to file."""
    intermediate_result = {
        "repository_name": feature_tree.get("repository_name", "unknown"),
        "repository_purpose": feature_tree.get("repository_purpose", ""),
        "repository_specification": feature_tree.get("repository_specification", ""),
        "meta": metadata_with_languages(feature_tree),
        "feature_tree": current_tree,
        "previous_feature_tree": previous_feature_tree,
        "iteration_logs": iteration_logs,
        "review_logs": review_logs,
        "expansion_directions": feature_tree.get("expansion_directions") or [],
    }
    try:
        save_json(intermediate_result, output_file)
    except Exception as e:
        logger.warning(f"Failed to save intermediate results: {e}")


def _apply_duplicate_renames(
    current_tree: Dict[str, Any],
    duplicate_leaf_renames: List[str],
) -> Tuple[Dict[str, Any], int]:
    """Apply duplicate leaf rename operations to the tree.

    Returns:
        Tuple of (updated tree, number of renames applied)
    """
    renames_applied = 0
    if not duplicate_leaf_renames:
        return current_tree, 0

    logger.info(
        f"Applying {len(duplicate_leaf_renames)} leaf renames for duplicates..."
    )
    for rename_str in duplicate_leaf_renames:
        if " -> " not in rename_str:
            logger.warning(f"Invalid rename format (missing ' -> '): {rename_str}")
            continue

        parts = rename_str.split(" -> ")
        if len(parts) != 2:
            logger.warning(f"Invalid rename format: {rename_str}")
            continue

        old_path = parts[0].strip()
        new_leaf_name = parts[1].strip()

        path_parts = [p.strip() for p in old_path.split("/")]
        if len(path_parts) < 2:
            logger.warning(f"Path too short for rename: {old_path}")
            continue

        new_path = "/".join(path_parts[:-1] + [new_leaf_name])

        existing_paths = get_all_leaf_paths(current_tree)
        if new_path in existing_paths:
            logger.warning(
                f"Cannot rename '{old_path}' -> '{new_path}': target path already exists"
            )
            continue

        # Collect description before removing
        old_descs = get_all_leaf_descriptions(current_tree)
        old_desc = old_descs.get(old_path, "")

        current_tree = remove_paths(current_tree, [old_path])
        current_tree = apply_changes(current_tree, [new_path])

        # Re-attach description to renamed leaf
        if old_desc:
            current_tree = attach_descriptions(current_tree, {new_path: old_desc})

        logger.info(f"Renamed: '{old_path}' -> '{new_path}'")
        renames_applied += 1

    current_tree = convert_leaves_to_list(current_tree)
    return current_tree, renames_applied


def _print_review_summary(review_logs: List[Dict[str, Any]]):
    """Print review phase summary."""
    print("\n" + "=" * 60)
    print("REVIEW PHASE SUMMARY")
    print("=" * 60)

    applied_logs = [
        log for log in review_logs
        if log.get("status") in ("paths_applied", "threshold_met")
    ]

    total_invalid_removed = sum(
        len(log.get("invalid_leaf_nodes", [])) for log in applied_logs
    )
    # threshold_met only applies MIU fixes, not coverage gap paths
    total_suggested_paths = sum(
        len(log.get("suggested_paths", []))
        for log in applied_logs
        if log.get("status") != "threshold_met"
    )
    total_replacements = sum(
        len(log.get("suggested_replacements", [])) for log in applied_logs
    )
    total_paths_added = total_suggested_paths + total_replacements

    print(f"Review iterations: {len(review_logs)} (applied: {len(applied_logs)})")
    print()
    print("[IN] PATHS ADDED:")
    print(f"  Total paths added: {total_paths_added}")
    print(f"    ├─ For coverage gaps: {total_suggested_paths}")
    print(f"    └─ For MIU replacements: {total_replacements}")
    print()
    print("[OUT] NODES REMOVED:")
    print(f"  Invalid nodes removed (MIU violations): {total_invalid_removed}")
    print()
    print(
        f"Net change: +{total_paths_added} added, -{total_invalid_removed} removed"
    )

    if review_logs:
        last_review = review_logs[-1]
        if "coverage_percentage" in last_review:
            print()
            print(f"Final coverage: {last_review['coverage_percentage']:.1f}%")
            print(f"Final status: {last_review.get('status', 'unknown')}")
    print("=" * 60)


# ======================== Review Phase ========================


def _run_review_phase(
    llm: LLMClient,
    feature_tree: Dict[str, Any],
    current_tree: Dict[str, Any],
    previous_feature_tree: Dict[str, Any],
    iteration_logs: List[Dict[str, Any]],
    output_file: Optional[Path],
    review_max_iterations: int = 3,
    review_threshold: float = 98.0,
    skip_coverage_gaps: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run the review phase on the current feature tree.

    Args:
        llm: LLMClient instance
        feature_tree: Full feature tree data (with repo info)
        current_tree: Current feature tree structure
        previous_feature_tree: Initial feature tree before expansion
        iteration_logs: Expansion iteration logs
        output_file: Output file path
        review_max_iterations: Max review iterations
        review_threshold: Coverage threshold (step1 only)
        skip_coverage_gaps: If True, only check duplicates and MIU (step2)

    Returns:
        Tuple of (updated current_tree, review_logs)
    """
    review_logs = []
    phase_name = "Lightweight Review (Duplicates + MIU)" if skip_coverage_gaps else "Full Review"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting {phase_name} Phase")
    if not skip_coverage_gaps:
        logger.info(f"Review threshold: {review_threshold}%")
    logger.info(f"Max review iterations: {review_max_iterations}")
    logger.info(f"{'=' * 60}")

    repository_specification = feature_tree.get("repository_specification", "")
    previous_review_data = None

    for review_iter in range(review_max_iterations):
        logger.info(
            f"\n{'=' * 20} Review Iteration {review_iter + 1}/{review_max_iterations} {'=' * 20}"
        )

        # Perform review with retry on parse failure
        parse_retries = 2
        review_result = None
        review_think = ""
        review_response = ""

        for parse_attempt in range(parse_retries):
            if parse_attempt > 0:
                logger.info(
                    f"Retrying review parse (attempt {parse_attempt + 1}/{parse_retries})..."
                )

            review_result, review_think, review_response = review_feature_coverage(
                llm=llm,
                repo_info=repository_specification,
                current_tree=current_tree,
                previous_review=previous_review_data,
                review_iter=review_iter,
            )

            if review_result is not None:
                break
            else:
                logger.warning(
                    f"Parse attempt {parse_attempt + 1}/{parse_retries} failed"
                )

        if review_result is None:
            error_msg = f"Failed to parse review response after {parse_retries} attempts"
            logger.warning(f"Review iteration {review_iter + 1}: {error_msg}")
            review_logs.append({
                "review_iteration": review_iter + 1,
                "status": "error",
                "error": error_msg,
                "think": review_think,
                "response": review_response,
            })
            continue

        # Print review summary
        print_review_summary(review_result, review_iter + 1)

        # Apply duplicate leaf renames
        current_tree, renames_applied = _apply_duplicate_renames(
            current_tree, review_result.duplicate_leaf_renames
        )

        if skip_coverage_gaps:
            # Lightweight review (step2): only handle MIU violations and duplicates
            repl_paths, repl_descs = _extract_paths_and_descs(review_result.suggested_replacements)

            if review_result.invalid_leaf_nodes or repl_paths:
                # Remove invalid nodes (even if no replacements provided)
                if review_result.invalid_leaf_nodes:
                    logger.info(
                        f"Removing {len(review_result.invalid_leaf_nodes)} invalid leaf nodes (MIU violations)..."
                    )
                    current_tree = remove_paths(current_tree, review_result.invalid_leaf_nodes)
                if repl_paths:
                    logger.info(f"Applying {len(repl_paths)} MIU replacement paths...")
                    current_tree = apply_changes(current_tree, repl_paths)
                current_tree = convert_leaves_to_list(current_tree)
                if repl_descs:
                    current_tree = attach_descriptions(current_tree, repl_descs)

                review_logs.append({
                    "review_iteration": review_iter + 1,
                    "status": "paths_applied",
                    "coverage_percentage": review_result.coverage_percentage,
                    "has_gaps": review_result.has_gaps,
                    "missing_functionalities": [],
                    "suggested_paths": [],
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "suggested_replacements": review_result.suggested_replacements,
                    "duplicate_leaf_renames": review_result.duplicate_leaf_renames,
                    "renames_applied": renames_applied,
                    "paths_applied": len(repl_paths),
                    "think": review_think,
                    "response": review_response,
                })

                if output_file:
                    _save_intermediate(
                        feature_tree, current_tree, previous_feature_tree,
                        iteration_logs, review_logs, output_file,
                    )
                    logger.info(f"[OK] Review iteration {review_iter + 1} results saved")
            else:
                status = "renames_only" if renames_applied > 0 else "clean"
                logger.info(f"Lightweight review complete: {status}")
                review_logs.append({
                    "review_iteration": review_iter + 1,
                    "status": status,
                    "coverage_percentage": review_result.coverage_percentage,
                    "has_gaps": review_result.has_gaps,
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "duplicate_leaf_renames": review_result.duplicate_leaf_renames,
                    "renames_applied": renames_applied,
                    "paths_applied": 0,
                    "think": review_think,
                    "response": review_response,
                })
                break  # Lightweight review: one pass unless MIU fixes were applied
        else:
            # Full review (step1): check coverage threshold
            if review_result.coverage_percentage >= review_threshold:
                logger.info(
                    f"[OK] Coverage threshold met: {review_result.coverage_percentage:.1f}% >= {review_threshold}%"
                )

                # Even when coverage is met, still apply MIU fixes and replacements
                miu_paths, miu_descs = _extract_paths_and_descs(review_result.suggested_replacements)
                if review_result.invalid_leaf_nodes:
                    logger.info(
                        f"Removing {len(review_result.invalid_leaf_nodes)} invalid leaf nodes (MIU violations)..."
                    )
                    current_tree = remove_paths(current_tree, review_result.invalid_leaf_nodes)
                if miu_paths:
                    logger.info(
                        f"Applying {len(miu_paths)} MIU replacement paths..."
                    )
                    current_tree = apply_changes(current_tree, miu_paths)
                if review_result.invalid_leaf_nodes or miu_paths:
                    current_tree = convert_leaves_to_list(current_tree)
                    if miu_descs:
                        current_tree = attach_descriptions(current_tree, miu_descs)

                if renames_applied > 0:
                    logger.info(f"   (Applied {renames_applied} leaf renames)")

                review_logs.append({
                    "review_iteration": review_iter + 1,
                    "status": "threshold_met",
                    "coverage_percentage": review_result.coverage_percentage,
                    "has_gaps": review_result.has_gaps,
                    "missing_functionalities": review_result.missing_functionalities,
                    "suggested_paths": review_result.suggested_paths,
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "suggested_replacements": review_result.suggested_replacements,
                    "duplicate_leaf_renames": review_result.duplicate_leaf_renames,
                    "renames_applied": renames_applied,
                    "paths_applied": len(miu_paths),
                    "think": review_think,
                    "response": review_response,
                })
                break

            # Collect all paths to apply
            sugg_paths, sugg_descs = _extract_paths_and_descs(review_result.suggested_paths)
            repl_paths2, repl_descs2 = _extract_paths_and_descs(review_result.suggested_replacements)
            all_paths_to_apply = sugg_paths + repl_paths2
            all_descs = {**sugg_descs, **repl_descs2}

            if all_paths_to_apply or review_result.invalid_leaf_nodes:
                if all_paths_to_apply:
                    logger.info(
                        f"Applying {len(all_paths_to_apply)} paths from review "
                        f"(suggested: {len(sugg_paths)}, "
                        f"MIU replacements: {len(repl_paths2)})..."
                    )

                if review_result.invalid_leaf_nodes:
                    logger.info(
                        f"Removing {len(review_result.invalid_leaf_nodes)} invalid leaf nodes..."
                    )
                    current_tree = remove_paths(
                        current_tree, review_result.invalid_leaf_nodes
                    )

                if all_paths_to_apply:
                    current_tree = apply_changes(current_tree, all_paths_to_apply)
                current_tree = convert_leaves_to_list(current_tree)
                if all_descs:
                    current_tree = attach_descriptions(current_tree, all_descs)

                previous_review_data = {
                    "coverage_percentage": review_result.coverage_percentage,
                    "missing_functionalities": review_result.missing_functionalities,
                    "suggested_paths": review_result.suggested_paths,
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "suggested_replacements": review_result.suggested_replacements,
                }

                review_logs.append({
                    "review_iteration": review_iter + 1,
                    "status": "paths_applied",
                    "coverage_percentage": review_result.coverage_percentage,
                    "has_gaps": review_result.has_gaps,
                    "missing_functionalities": review_result.missing_functionalities,
                    "suggested_paths": review_result.suggested_paths,
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "suggested_replacements": review_result.suggested_replacements,
                    "duplicate_leaf_renames": review_result.duplicate_leaf_renames,
                    "renames_applied": renames_applied,
                    "paths_applied": len(all_paths_to_apply),
                    "think": review_think,
                    "response": review_response,
                })

                if output_file:
                    _save_intermediate(
                        feature_tree, current_tree, previous_feature_tree,
                        iteration_logs, review_logs, output_file,
                    )
                    logger.info(f"[OK] Review iteration {review_iter + 1} results saved")
            else:
                if renames_applied > 0:
                    logger.info(
                        f"No more gaps, but applied {renames_applied} leaf renames"
                    )
                    status = "renames_only"
                else:
                    logger.info("No more gaps identified or no suggestions provided")
                    status = "no_gaps"
                review_logs.append({
                    "review_iteration": review_iter + 1,
                    "status": status,
                    "coverage_percentage": review_result.coverage_percentage,
                    "has_gaps": review_result.has_gaps,
                    "missing_functionalities": review_result.missing_functionalities,
                    "suggested_paths": review_result.suggested_paths,
                    "invalid_leaf_nodes": review_result.invalid_leaf_nodes,
                    "suggested_replacements": review_result.suggested_replacements,
                    "duplicate_leaf_renames": review_result.duplicate_leaf_renames,
                    "renames_applied": renames_applied,
                    "paths_applied": 0,
                    "think": review_think,
                    "response": review_response,
                })
                break

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{phase_name} Phase Completed")
    logger.info(f"{'=' * 60}")

    if review_logs:
        last_review = review_logs[-1]
        if "coverage_percentage" in last_review:
            logger.info(f"Final coverage: {last_review['coverage_percentage']:.1f}%")

    return current_tree, review_logs


# ======================== Feature Tree Building ========================


def build_from_spec(
    feature_tree: Dict[str, Any],
    output_file: Path = None,
    review_max_iterations: int = 3,
    review_threshold: float = 98.0,
    llm: LLMClient = None,
) -> Dict[str, Any]:
    """Step 1: Build or expand feature tree.

    If output_file already exists with a non-empty feature tree, assumes spec
    features are complete and switches to beyond-spec expansion mode (adds
    features the spec does not describe but are practically necessary).
    Otherwise, builds the feature tree from spec requirements.

    Args:
        feature_tree: Initial feature tree structure with repo info
        output_file: Output file path for saving intermediate results
        review_max_iterations: Maximum review iterations
        review_threshold: Coverage threshold to stop review (spec mode only)
        llm: LLMClient instance

    Returns:
        Dictionary containing final feature tree and logs
    """
    if llm is None:
        llm = LLMClient()

    # Detect expand mode: output file exists with non-empty feature tree
    expand_mode = (
        output_file is not None
        and output_file.exists()
        and bool(feature_tree.get("feature_tree"))
    )

    if expand_mode:
        logger.info("=" * 60)
        logger.info("Step 1: Beyond-Spec Expansion (output file exists, spec features assumed complete)")
        logger.info(f"Max iterations: {MAX_ITERATIONS}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Build from Spec (Model Self-Termination)")
        logger.info(f"Max iterations: {MAX_ITERATIONS}")
        logger.info("=" * 60)

    iteration_logs = []
    previous_feature_tree = feature_tree.get("feature_tree", {})
    current_tree = previous_feature_tree
    repo_info = build_repo_info(feature_tree)
    consecutive_failures = 0

    for i in range(MAX_ITERATIONS):
        logger.info(f"\n{'=' * 20} Iteration {i + 1}/{MAX_ITERATIONS} {'=' * 20}")

        current_tree_json = json.dumps(current_tree, indent=2) if current_tree else "{}"
        if expand_mode:
            prompt = PROMPT_TEMPLATE_BUILD_EXPAND.format(
                repo_info=repo_info, current_tree=current_tree_json
            )
        else:
            prompt = PROMPT_TEMPLATE_BUILD_FEATURE.format(
                repo_info=repo_info, current_tree=current_tree_json
            )

        think, result, response = llm.call_structured(
            system_prompt=prompt, user_prompt="",
            response_model=AddPathsOutput, max_retries=3,
            purpose=f"step1_{'expand' if expand_mode else 'build'}_{i + 1}",
        )

        if result is None:
            error_msg = "Failed to parse AI response or AI call failed"
            logger.warning(f"Iteration {i + 1}: {error_msg}")
            iteration_logs.append({
                "iteration": i + 1,
                "status": "error",
                "error": error_msg,
                "new_paths": [],
                "paths_count": 0,
                "is_complete": False,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"[WARNING] {MAX_CONSECUTIVE_FAILURES} consecutive failures, stopping expansion"
                )
                break
            continue

        # Check completion signal
        if result.is_complete:
            logger.info(
                f"[OK] Model declared completion at iteration {i + 1}: {result.completion_reason}"
            )
            # Apply any remaining paths before stopping
            if result.add_new_feature_paths:
                new_paths, new_descs = _extract_paths_and_descs(result.add_new_feature_paths)
                logger.info(f"Applying {len(new_paths)} final paths before completion")
                current_tree = apply_changes(current_tree, new_paths)
                current_tree = convert_leaves_to_list(current_tree)
                if new_descs:
                    current_tree = attach_descriptions(current_tree, new_descs)
            iteration_logs.append({
                "iteration": i + 1,
                "status": "complete",
                "new_paths": result.add_new_feature_paths,
                "paths_count": len(result.add_new_feature_paths),
                "is_complete": True,
                "completion_reason": result.completion_reason,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            break

        if not result.add_new_feature_paths:
            logger.info(f"Iteration {i + 1}: No new paths returned (continuing)")
            iteration_logs.append({
                "iteration": i + 1,
                "status": "empty",
                "new_paths": [],
                "paths_count": 0,
                "is_complete": False,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"[WARNING] {MAX_CONSECUTIVE_FAILURES} consecutive empty/error responses, stopping expansion"
                )
                break
            continue

        # Reset counter on successful path addition
        consecutive_failures = 0
        new_paths, new_descs = _extract_paths_and_descs(result.add_new_feature_paths)
        logger.info(f"Received {len(new_paths)} new paths")

        current_tree = apply_changes(current_tree, new_paths)
        current_tree = convert_leaves_to_list(current_tree)
        if new_descs:
            current_tree = attach_descriptions(current_tree, new_descs)

        iteration_logs.append({
            "iteration": i + 1,
            "status": "success",
            "new_paths": result.add_new_feature_paths,
            "paths_count": len(result.add_new_feature_paths),
            "is_complete": False,
            "think": think,
            "current_tree": current_tree,
            "response": response,
        })

        if output_file:
            _save_intermediate(
                feature_tree, current_tree, previous_feature_tree,
                iteration_logs, [], output_file,
            )
            logger.info(f"[OK] Iteration {i + 1} results saved to {output_file}")
    else:
        logger.warning(
            f"[WARNING] Reached maximum iterations ({MAX_ITERATIONS}) without model declaring completion"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("Expansion phase completed")
    logger.info(f"{'=' * 60}")

    # Review phase:
    # - Expand mode: lightweight review (duplicates + MIU only, no coverage gaps)
    # - Spec mode: full review (coverage gaps + MIU + duplicates)
    current_tree, review_logs = _run_review_phase(
        llm=llm,
        feature_tree=feature_tree,
        current_tree=current_tree,
        previous_feature_tree=previous_feature_tree,
        iteration_logs=iteration_logs,
        output_file=output_file,
        review_max_iterations=review_max_iterations,
        review_threshold=review_threshold,
        skip_coverage_gaps=expand_mode,
    )

    result = {
        "repository_name": feature_tree.get("repository_name", "unknown"),
        "repository_purpose": feature_tree.get("repository_purpose", ""),
        "repository_specification": feature_tree.get("repository_specification", ""),
        "meta": metadata_with_languages(feature_tree),
        "feature_tree": current_tree,
        "previous_feature_tree": previous_feature_tree,
        "iteration_logs": iteration_logs,
        "review_logs": review_logs,
        "expansion_directions": feature_tree.get("expansion_directions") or [],
    }
    return result


def expand_with_direction(
    feature_tree: Dict[str, Any],
    direction: str,
    output_file: Path = None,
    review_max_iterations: int = 3,
    llm: LLMClient = None,
) -> Dict[str, Any]:
    """Step 2: Expand feature tree in a user-chosen direction with model self-termination.

    Only expands reasonable and necessary features for the given direction.
    Does NOT add uncertain, unclear, or meaningless features.
    After expansion, runs lightweight review (duplicates + MIU only).

    Args:
        feature_tree: Feature tree data (with repo info and existing tree)
        direction: User-chosen expansion direction description
        output_file: Output file path for saving intermediate results
        review_max_iterations: Max review iterations for lightweight review
        llm: LLMClient instance

    Returns:
        Dictionary containing final feature tree and logs
    """
    if llm is None:
        llm = LLMClient()

    # Resolve short direction name to full direction context
    resolved_direction = _resolve_direction(feature_tree, direction)

    logger.info("=" * 60)
    logger.info("Step 2: Directed Expansion")
    logger.info(f"Direction: {direction}")
    if resolved_direction != direction:
        logger.info(f"Resolved to full context ({len(resolved_direction)} chars)")
    logger.info(f"Max iterations: {MAX_ITERATIONS}")
    logger.info("=" * 60)

    # Record user selection
    _record_selected_direction(feature_tree, direction)

    iteration_logs = []
    previous_feature_tree = feature_tree.get("feature_tree", {})
    current_tree = previous_feature_tree
    repo_info = build_repo_info(feature_tree)
    consecutive_failures = 0

    for i in range(MAX_ITERATIONS):
        logger.info(f"\n{'=' * 20} Iteration {i + 1}/{MAX_ITERATIONS} {'=' * 20}")

        current_tree_json = json.dumps(current_tree, indent=2) if current_tree else "{}"
        prompt = PROMPT_TEMPLATE_BUILD_DIRECTED.format(
            repo_info=repo_info,
            current_tree=current_tree_json,
            direction=resolved_direction,
        )

        think, result, response = llm.call_structured(
            system_prompt=prompt, user_prompt="",
            response_model=AddPathsOutput, max_retries=3,
            purpose=f"step2_directed_{i + 1}",
        )

        if result is None:
            error_msg = "Failed to parse AI response or AI call failed"
            logger.warning(f"Iteration {i + 1}: {error_msg}")
            iteration_logs.append({
                "iteration": i + 1,
                "status": "error",
                "error": error_msg,
                "new_paths": [],
                "paths_count": 0,
                "is_complete": False,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"[WARNING] {MAX_CONSECUTIVE_FAILURES} consecutive failures, stopping expansion"
                )
                break
            continue

        # Check completion signal
        if result.is_complete:
            logger.info(
                f"[OK] Model declared completion at iteration {i + 1}: {result.completion_reason}"
            )
            if result.add_new_feature_paths:
                new_paths, new_descs = _extract_paths_and_descs(result.add_new_feature_paths)
                logger.info(f"Applying {len(new_paths)} final paths before completion")
                current_tree = apply_changes(current_tree, new_paths)
                current_tree = convert_leaves_to_list(current_tree)
                if new_descs:
                    current_tree = attach_descriptions(current_tree, new_descs)
            iteration_logs.append({
                "iteration": i + 1,
                "status": "complete",
                "new_paths": result.add_new_feature_paths,
                "paths_count": len(result.add_new_feature_paths),
                "is_complete": True,
                "completion_reason": result.completion_reason,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            break

        if not result.add_new_feature_paths:
            logger.info(f"Iteration {i + 1}: No new paths returned (continuing)")
            iteration_logs.append({
                "iteration": i + 1,
                "status": "empty",
                "new_paths": [],
                "paths_count": 0,
                "is_complete": False,
                "think": think,
                "current_tree": current_tree,
                "response": response,
            })
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"[WARNING] {MAX_CONSECUTIVE_FAILURES} consecutive empty/error responses, stopping expansion"
                )
                break
            continue

        # Reset counter on successful path addition
        consecutive_failures = 0
        new_paths, new_descs = _extract_paths_and_descs(result.add_new_feature_paths)
        logger.info(f"Received {len(new_paths)} new paths")

        current_tree = apply_changes(current_tree, new_paths)
        current_tree = convert_leaves_to_list(current_tree)
        if new_descs:
            current_tree = attach_descriptions(current_tree, new_descs)

        iteration_logs.append({
            "iteration": i + 1,
            "status": "success",
            "new_paths": result.add_new_feature_paths,
            "paths_count": len(result.add_new_feature_paths),
            "is_complete": False,
            "think": think,
            "current_tree": current_tree,
            "response": response,
        })

        if output_file:
            _save_intermediate(
                feature_tree, current_tree, previous_feature_tree,
                iteration_logs, [], output_file,
            )
            logger.info(f"[OK] Iteration {i + 1} results saved to {output_file}")
    else:
        logger.warning(
            f"[WARNING] Reached maximum iterations ({MAX_ITERATIONS}) without model declaring completion"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("Directed expansion phase completed")
    logger.info(f"{'=' * 60}")

    # Lightweight review: duplicates + MIU only (no coverage gap analysis)
    current_tree, review_logs = _run_review_phase(
        llm=llm,
        feature_tree=feature_tree,
        current_tree=current_tree,
        previous_feature_tree=previous_feature_tree,
        iteration_logs=iteration_logs,
        output_file=output_file,
        review_max_iterations=review_max_iterations,
        review_threshold=100.0,  # Not used when skip_coverage_gaps=True
        skip_coverage_gaps=True,
    )

    result = {
        "repository_name": feature_tree.get("repository_name", "unknown"),
        "repository_purpose": feature_tree.get("repository_purpose", ""),
        "repository_specification": feature_tree.get("repository_specification", ""),
        "meta": metadata_with_languages(feature_tree),
        "feature_tree": current_tree,
        "previous_feature_tree": previous_feature_tree,
        "iteration_logs": iteration_logs,
        "review_logs": review_logs,
        "expansion_directions": feature_tree.get("expansion_directions") or [],
    }
    return result


def suggest_directions(
    feature_tree: Dict[str, Any],
    output_file: Path = None,
    llm: LLMClient = None,
) -> Dict[str, Any]:
    """Suggest 4-6 expansion directions based on current feature tree.

    Saves the suggested directions as a new round in the expansion_directions array.

    Args:
        feature_tree: Feature tree data (with repo info and existing tree)
        output_file: Output file path to persist directions
        llm: LLMClient instance

    Returns:
        Dictionary with directions list
    """
    if llm is None:
        llm = LLMClient()

    logger.info("=" * 60)
    logger.info("Suggesting Expansion Directions")
    logger.info("=" * 60)

    current_tree = feature_tree.get("feature_tree", {})
    repo_info = build_repo_info(feature_tree)
    current_tree_json = json.dumps(current_tree, indent=2) if current_tree else "{}"

    # Build expansion history from previous rounds
    expansion_history = _build_expansion_history(feature_tree)

    prompt = PROMPT_TEMPLATE_SUGGEST_DIRECTIONS.format(
        repo_info=repo_info,
        current_tree=current_tree_json,
        expansion_history=expansion_history,
    )

    think, result, response = llm.call_structured(
        system_prompt=prompt, user_prompt="",
        response_model=SuggestDirectionsOutput, max_retries=3,
        purpose="suggest_directions",
    )

    if result is None:
        logger.error("Failed to get direction suggestions")
        return {"directions": [], "error": "Failed to parse AI response"}

    generated_at = datetime.now(timezone.utc).isoformat()
    directions = []
    for d in result.directions:
        directions.append({
            "name": d.name,
            "description": d.description,
            "rationale": d.rationale,
        })

    # Save directions as a new round in the expansion_directions array
    if output_file and output_file.exists():
        try:
            existing_data = load_json(output_file)
            expansion_dirs = existing_data.get("expansion_directions") or []

            # Migrate old object format to array format if needed
            if isinstance(expansion_dirs, dict):
                expansion_dirs = _migrate_expansion_directions(expansion_dirs)

            # Determine new round number
            new_round = len(expansion_dirs) + 1

            # Append new round
            expansion_dirs.append({
                "round": new_round,
                "generated_at": generated_at,
                "directions": directions,
                "selected": [],
            })

            existing_data["expansion_directions"] = expansion_dirs
            save_json(existing_data, output_file)
            logger.info(f"Saved {len(directions)} directions as round {new_round} to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save directions to {output_file}: {e}")

    logger.info(f"Generated {len(directions)} expansion directions")
    return {"directions": directions}


def _migrate_expansion_directions(old_format: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Migrate old single-object expansion_directions format to the new array-of-rounds format.

    Old format:
        {"generated_at": "...", "directions": [...], "selected": [...]}

    New format:
        [{"round": 1, "generated_at": "...", "directions": [...], "selected": [...]}]

    Args:
        old_format: The old-style expansion_directions dict

    Returns:
        List of round dicts in the new format
    """
    if not old_format or not old_format.get("directions"):
        return []

    return [{
        "round": 1,
        "generated_at": old_format.get("generated_at", ""),
        "directions": old_format.get("directions", []),
        "selected": old_format.get("selected", []),
    }]


def _build_expansion_history(feature_tree: Dict[str, Any]) -> str:
    """Build a formatted expansion history string from all previous rounds for inclusion in the suggest-directions prompt.

    Args:
        feature_tree: Feature tree data containing expansion_directions

    Returns:
        Formatted string describing previous rounds of direction generation and selection
    """
    expansion_dirs = feature_tree.get("expansion_directions") or []

    # Support old dict format
    if isinstance(expansion_dirs, dict):
        expansion_dirs = _migrate_expansion_directions(expansion_dirs)

    if not expansion_dirs:
        return "No previous expansion history. This is the first time suggesting directions."

    lines = [f"Total previous rounds: {len(expansion_dirs)}\n"]

    for round_data in expansion_dirs:
        round_num = round_data.get("round", "?")
        generated_at = round_data.get("generated_at", "unknown")
        directions = round_data.get("directions", [])
        selected = round_data.get("selected", [])

        selected_names = {s.get("name", "").strip().lower() for s in selected}
        direction_names = {d.get("name", "").strip().lower() for d in directions}

        lines.append(f"### Round {round_num} (generated: {generated_at})")
        lines.append(f"Generated {len(directions)} directions, user selected {len(selected)}:\n")

        for i, d in enumerate(directions, 1):
            name = d.get("name", "")
            desc = d.get("description", "")
            rationale = d.get("rationale", "")
            was_selected = name.strip().lower() in selected_names
            status = "[SELECTED & EXPANDED]" if was_selected else "[NOT SELECTED]"

            lines.append(f"  {i}. {status} **{name}**")
            if desc:
                lines.append(f"     Description: {desc}")
            if rationale:
                lines.append(f"     Rationale: {rationale}")
            lines.append("")

        # List selected directions that don't match any generated direction in this round
        # (can happen with migrated data from old format)
        orphaned_selected = [
            s for s in selected
            if s.get("name", "").strip().lower() not in direction_names
        ]
        if orphaned_selected:
            lines.append("  Also expanded (from earlier sessions):")
            for s in orphaned_selected:
                lines.append(f"    - [SELECTED & EXPANDED] **{s.get('name', '')}**")
            lines.append("")

    return "\n".join(lines)


def _resolve_direction(
    feature_tree: Dict[str, Any],
    direction_name: str,
) -> str:
    """Resolve a short direction name to a full direction string for the prompt.

    Looks up saved directions across all rounds in feature_tree data. If found,
    returns 'name: description (rationale: ...)'. Otherwise returns the name as-is.

    Args:
        feature_tree: Feature tree data (may contain expansion_directions)
        direction_name: Short direction name from CLI

    Returns:
        Full direction string for the prompt
    """
    saved = feature_tree.get("expansion_directions") or []

    # Support both old dict format and new array format
    if isinstance(saved, dict):
        saved = _migrate_expansion_directions(saved)

    direction_lower = direction_name.strip().lower()

    # Search across all rounds, latest first
    for round_data in reversed(saved):
        saved_directions = round_data.get("directions", [])
        for d in saved_directions:
            if d.get("name", "").strip().lower() == direction_lower:
                parts = [d["name"]]
                if d.get("description"):
                    parts.append(d["description"])
                if d.get("rationale"):
                    parts.append(f"Rationale: {d['rationale']}")
                logger.info(f"Resolved direction '{direction_name}' from saved data")
                return "\n".join(parts)

    logger.info(f"Direction '{direction_name}' not found in saved data, using as-is")
    return direction_name


def _record_selected_direction(
    feature_tree: Dict[str, Any],
    direction_name: str,
):
    """Record that the user selected a direction for expansion.

    Updates the in-memory feature_tree dict (persisted via normal save flow).
    Records the selection in the latest round of expansion_directions.

    Args:
        feature_tree: Feature tree data (modified in-place)
        direction_name: The direction name selected by user
    """
    expansion_dirs = feature_tree.get("expansion_directions") or []

    # Migrate old dict format if needed
    if isinstance(expansion_dirs, dict):
        expansion_dirs = _migrate_expansion_directions(expansion_dirs)
        feature_tree["expansion_directions"] = expansion_dirs

    if not expansion_dirs:
        return

    # Record in the latest round
    latest_round = expansion_dirs[-1]
    selected = latest_round.get("selected", [])

    # Avoid duplicate entries for same direction name within the same round
    already_selected = any(
        s.get("name", "").strip().lower() == direction_name.strip().lower()
        for s in selected
    )
    if not already_selected:
        selected.append({
            "name": direction_name,
            "selected_at": datetime.now(timezone.utc).isoformat(),
        })
        latest_round["selected"] = selected
        logger.info(f"Recorded selected direction: {direction_name} (round {latest_round.get('round', '?')})")


# ======================== Data Loading ========================


def _load_feature_data(feature_build_path: Path, feature_spec_path: Path) -> Dict[str, Any]:
    """Load feature tree data from feature_build.json or feature_spec.json."""
    use_feature_build = False
    if feature_build_path.exists():
        try:
            feature_build_data = load_json(feature_build_path)
            repository_name = feature_build_data.get("repository_name", "")
            repository_purpose = feature_build_data.get("repository_purpose", "")
            repository_specification = feature_build_data.get(
                "repository_specification", ""
            )
            feature_tree_data = feature_build_data.get("feature_tree", {})

            if (
                isinstance(repository_name, str)
                and repository_name.strip()
                and isinstance(repository_purpose, str)
                and repository_purpose.strip()
                and isinstance(repository_specification, str)
                and repository_specification.strip()
                and isinstance(feature_tree_data, dict)
                and feature_tree_data
            ):
                use_feature_build = True
                logger.info(
                    "feature_build.json has all required fields, using it as input"
                )
            else:
                logger.info(
                    "feature_build.json exists but has empty required fields, will use feature_spec.json"
                )
        except Exception as e:
            logger.warning(
                f"Failed to validate feature_build.json: {e}, will use feature_spec.json"
            )

    if use_feature_build:
        feature_tree = feature_build_data
        logger.info("Loaded from feature_build.json:")
        logger.info(
            f"  repository_name: {feature_tree.get('repository_name', 'unknown')}"
        )
        logger.info(
            f"  repository_purpose: {len(feature_tree.get('repository_purpose', ''))} chars"
        )
        logger.info(
            f"  repository_specification: {len(feature_tree.get('repository_specification', ''))} chars"
        )
        logger.info(
            f"  feature_tree: {len(feature_tree.get('feature_tree', {}))} top-level categories"
        )
    else:
        if not feature_spec_path.exists():
            logger.error(f"feature_spec.json not found: {feature_spec_path}")
            sys.exit(1)

        try:
            feature_spec = load_json(feature_spec_path)

            feature_tree = {
                "repository_name": "",
                "repository_purpose": "",
                "repository_specification": "",
                "meta": {},
                "feature_tree": {},
            }

            spec_repo_name = feature_spec.get("repository_name", "").strip()
            if spec_repo_name:
                feature_tree["repository_name"] = spec_repo_name
                logger.info(
                    f"Loaded repository_name from feature_spec.json: {spec_repo_name}"
                )

            spec_repo_purpose = feature_spec.get("repository_purpose", "").strip()
            if spec_repo_purpose:
                feature_tree["repository_purpose"] = spec_repo_purpose
                logger.info(
                    f"Loaded repository_purpose from feature_spec.json ({len(spec_repo_purpose)} chars)"
                )

            feature_tree["meta"] = metadata_with_languages(feature_spec)
            languages = _target_languages(feature_tree)
            if languages:
                logger.info(
                    "Loaded target languages from feature_spec.json: %s",
                    ", ".join(languages),
                )

            spec_content = feature_spec_path.read_text(encoding="utf-8").strip()
            if spec_content:
                feature_tree["repository_specification"] = spec_content
                logger.info(
                    f"Loaded repository_specification from feature_spec.json ({len(spec_content)} chars)"
                )

            if feature_build_path.exists():
                try:
                    existing_build = load_json(feature_build_path)
                    existing_tree = existing_build.get("feature_tree", {})
                    if isinstance(existing_tree, dict) and existing_tree:
                        feature_tree["feature_tree"] = existing_tree
                        logger.info(
                            f"Loaded existing feature_tree from feature_build.json ({len(existing_tree)} top-level categories)"
                        )
                    # Preserve expansion_directions from existing build
                    existing_dirs = existing_build.get("expansion_directions")
                    if existing_dirs:
                        feature_tree["expansion_directions"] = existing_dirs
                        logger.info("Loaded existing expansion_directions from feature_build.json")
                except Exception as e:
                    logger.warning(
                        f"Could not load feature_tree from existing feature_build.json: {e}"
                    )

        except Exception as e:
            logger.error(f"Failed to read feature_spec.json: {e}")
            sys.exit(1)

    # Validate required fields
    if not feature_tree.get("repository_name"):
        logger.error(
            f"repository_name is required. Ensure {feature_spec_path} exists and has repository_name field."
        )
        sys.exit(1)

    if not feature_tree.get("repository_specification"):
        logger.error(
            f"repository_specification is required. Ensure {feature_spec_path} exists and has content."
        )
        sys.exit(1)

    return feature_tree


# ======================== Main Function ========================


def main():
    parser = argparse.ArgumentParser(
        description="Feature tree expansion script - Two-step workflow"
    )

    parser.add_argument(
        "--mode",
        choices=["step1", "step2", "suggest-directions"],
        default="step1",
        help="Operation mode: step1 (spec-driven build), step2 (directed expansion), suggest-directions",
    )

    parser.add_argument(
        "--direction",
        type=str,
        default="",
        help="Comma-separated direction indices from suggest-directions output, e.g. '1,3,5' (required for --mode step2)",
    )

    parser.add_argument(
        "--feature-tree",
        type=Path,
        required=False,
        default=FEATURE_BUILD_FILE,
        help=f"Path to feature tree JSON file (default: {FEATURE_BUILD_FILE})",
    )

    parser.add_argument(
        "--feature-spec",
        type=Path,
        required=False,
        default=FEATURE_SPEC_FILE,
        help=f"Path to feature spec JSON file (default: {FEATURE_SPEC_FILE})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURE_BUILD_FILE,
        help=f"Output file path (default: {FEATURE_BUILD_FILE})",
    )

    parser.add_argument("--verbose", action="store_true", help="Show verbose logging")

    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording",
    )

    # Review arguments (used in step1; step2 uses lightweight review automatically)
    parser.add_argument(
        "--review-max-iterations",
        type=int,
        default=3,
        help="Maximum number of review iterations (default: 3)",
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=98.0,
        help="Coverage percentage threshold to stop review (default: 98.0, step1 only)",
    )

    args = parser.parse_args()

    # Validate step2 requires --direction
    if args.mode == "step2" and not args.direction.strip():
        parser.error("--direction is required when --mode is step2")

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load feature data
    logger.info("Loading input data...")
    feature_tree = _load_feature_data(args.feature_tree, args.feature_spec)

    logger.info(f"Repository: {feature_tree.get('repository_name', 'unknown')}")
    logger.info(f"Mode: {args.mode}")

    # Initialize trajectory
    trajectory = None
    step_id = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("feature_build")
        trajectory.start(metadata={
            "mode": args.mode,
            "direction": args.direction if args.mode == "step2" else "",
            "output_file": str(args.output),
        })
        step_desc = {
            "step1": "Build feature tree from specification",
            "step2": f"Expand feature tree: directions {args.direction}",
            "suggest-directions": "Suggest expansion directions",
        }[args.mode]
        step = trajectory.add_step("feature_build", step_desc)
        trajectory.start_step(step.step_id)
        step_id = step.step_id

    # Execute
    try:
        llm = LLMClient(trajectory=trajectory, step_id=step_id)

        if args.mode == "step1":
            result = build_from_spec(
                feature_tree,
                output_file=args.output,
                review_max_iterations=args.review_max_iterations,
                review_threshold=args.review_threshold,
                llm=llm,
            )
            save_json(result, args.output)

            # Print summary
            iteration_logs = result.get("iteration_logs", [])
            review_logs = result.get("review_logs", [])
            final_tree = result.get("feature_tree", {})
            previous_tree = result.get("previous_feature_tree", {})
            print_summary_tables(iteration_logs, final_tree, review_logs, previous_tree)
            if review_logs:
                _print_review_summary(review_logs)

        elif args.mode == "step2":
            # Parse direction indices (comma-separated)
            try:
                direction_indices = [int(x.strip()) for x in args.direction.split(",")]
            except ValueError:
                logger.error("--direction must be comma-separated integers (e.g., '1,3,5')")
                sys.exit(1)

            # Deduplicate while preserving order
            seen = set()
            unique_indices = []
            for idx in direction_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            if len(unique_indices) < len(direction_indices):
                logger.info(
                    f"Deduplicated direction indices: {direction_indices} -> {unique_indices}"
                )
            direction_indices = unique_indices

            # Resolve indices to direction names from saved data
            saved = feature_tree.get("expansion_directions") or []

            # Support both old dict format and new array format
            if isinstance(saved, dict):
                saved = _migrate_expansion_directions(saved)

            if not saved:
                logger.error("No saved expansion directions found. Run --mode suggest-directions first.")
                sys.exit(1)

            # Use the latest round's directions for index resolution
            latest_round = saved[-1]
            saved_directions = latest_round.get("directions", [])

            if not saved_directions:
                logger.error("No saved expansion directions found in latest round. Run --mode suggest-directions first.")
                sys.exit(1)

            direction_names = []
            for idx in direction_indices:
                if idx < 1 or idx > len(saved_directions):
                    logger.error(
                        f"Invalid direction index: {idx} (valid range: 1-{len(saved_directions)})"
                    )
                    sys.exit(1)
                direction_names.append(saved_directions[idx - 1]["name"])

            logger.info(f"Expanding {len(direction_names)} direction(s): {direction_names}")

            for dir_i, direction_name in enumerate(direction_names):
                logger.info(f"\n{'#' * 60}")
                logger.info(f"Direction {dir_i + 1}/{len(direction_names)}: {direction_name}")
                logger.info(f"{'#' * 60}")

                # Reload feature tree between directions (previous expansion saved to args.output)
                if dir_i > 0:
                    feature_tree = _load_feature_data(args.output, args.feature_spec)

                result = expand_with_direction(
                    feature_tree,
                    direction=direction_name,
                    output_file=args.output,
                    review_max_iterations=args.review_max_iterations,
                    llm=llm,
                )
                save_json(result, args.output)

                # Print summary for this direction
                iteration_logs = result.get("iteration_logs", [])
                review_logs = result.get("review_logs", [])
                final_tree = result.get("feature_tree", {})
                previous_tree = result.get("previous_feature_tree", {})
                print_summary_tables(iteration_logs, final_tree, review_logs, previous_tree)
                if review_logs:
                    _print_review_summary(review_logs)

        elif args.mode == "suggest-directions":
            result = suggest_directions(feature_tree, output_file=args.output, llm=llm)
            # Output JSON to stdout for agent parsing
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        if trajectory:
            if step_id is not None:
                trajectory.fail_step(step_id, str(e))
            trajectory.fail(str(e))
        raise

    # Mark trajectory as complete
    if trajectory:
        completion_metadata = {"mode": args.mode}
        if args.mode in ("step1", "step2"):
            completion_metadata["feature_count"] = len(result.get("feature_tree", {}))
            completion_metadata["review_iterations"] = len(result.get("review_logs", []))
        elif args.mode == "suggest-directions":
            completion_metadata["directions_count"] = len(result.get("directions", []))

        if step_id is not None:
            trajectory.complete_step(step_id, completion_metadata)
        trajectory.complete(metadata=completion_metadata)
        logger.info(f"[OK] Trajectory saved to: {trajectory.trajectory_file}")

    logger.info(f"\n[OK] Done (mode={args.mode})")


if __name__ == "__main__":
    main()
