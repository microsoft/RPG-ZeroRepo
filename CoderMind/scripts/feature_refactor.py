#!/usr/bin/env python3
"""Simplified Feature Tree Refactoring Script.

Function: Refactor feature tree into modular component architecture
- Step 1: Plan subtree structure based on domain analysis
- Step 2: Iteratively assign features to planned subtrees
"""

import json
import logging
import argparse
import copy
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from feature.prompts import (
    PROMPT_TEMPLATE_SUBTREE_PLANNING,
    PROMPT_TEMPLATE_FEATURE_ORGANIZATION,
)
from common.paths import FEATURE_BUILD_FILE, FEATURE_TREE_FILE
from common import print_unicode_table, get_all_leaf_paths, get_leaf_name, get_all_leaf_descriptions
from common.llm_client import LLMClient
from common.language_meta import metadata_with_languages
from common.trajectory import load_or_create_trajectory


# ============================================================================
# Utility Functions
# ============================================================================


def count_all_nodes(tree: Dict[str, Any]) -> int:
    """Recursively count all nodes in the tree (including intermediate and leaf nodes).

    Args:
        tree: Feature tree or subtree

    Returns:
        Total node count
    """
    if not tree:
        return 0

    count = 0
    if isinstance(tree, dict):
        for key, value in tree.items():
            count += 1  # Count current key
            if isinstance(value, dict):
                count += count_all_nodes(value)  # Recursively count subtree
            elif isinstance(value, list):
                count += len(value)  # Count leaf nodes
    elif isinstance(tree, list):
        count = len(tree)

    return count


# ============================================================================
# Pydantic Data Models
# ============================================================================


class SubtreePlan(BaseModel):
    """Subtree planning."""

    name: str = Field(description="Subtree/component name")
    purpose: str = Field(description="High-level purpose or theme of the subtree")
    estimate_size: int = Field(description="Estimated feature count")


class SubtreePlanningOutput(BaseModel):
    """Planning step output."""

    total_subtrees: int = Field(description="Total number of planned subtrees")
    subtree_plans: List[SubtreePlan] = Field(description="List of subtree plans")
    reasoning: str = Field(description="Organizational rationale")


class FeatureAssignment(BaseModel):
    """Feature assignment."""

    subtree_name: str = Field(description="Target subtree name")
    assigned_paths: List[str] = Field(
        description="List of feature paths assigned to this subtree"
    )


class FeatureOrganizationOutput(BaseModel):
    """Organization step output."""

    assignments: List[FeatureAssignment] = Field(
        description="Feature assignments for each subtree"
    )


# ============================================================================
# Tree Operation Utility Functions (Inline Version)
# ============================================================================


def extract_leaf_nodes(tree: Dict[str, Any]) -> List[str]:
    """Extract all leaf node names."""
    leaf_names = set()
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                if not value:
                    leaf_names.add(key)
                else:
                    leaf_names.update(extract_leaf_nodes(value))
            elif isinstance(value, list):
                for item in value:
                    leaf_names.add(get_leaf_name(item))
            else:
                leaf_names.add(value)
    return list(leaf_names)


def insert_path(tree: Dict[str, Any], path: str, delimiter: str = "/") -> None:
    """Insert a single path in place."""
    parts = [p.strip() for p in path.split(delimiter) if p.strip()]
    parent, key_in_parent = None, None
    node = tree
    i = 0

    while i < len(parts):
        part, last = parts[i], i == len(parts) - 1

        if isinstance(node, dict):
            mk = next((k for k in node if k.lower() == part.lower()), None)
            if last:
                if mk is None:
                    node[part] = []
                break
            else:
                if mk is None:
                    node[part] = {}
                    mk = part
                elif isinstance(node[mk], list):
                    node[mk] = {x: [] for x in node[mk]}
                elif not isinstance(node[mk], dict):
                    node[mk] = {}
                parent, key_in_parent = node, mk
                node = node[mk]
                i += 1
                continue

        elif isinstance(node, list):
            if last:
                if part.lower() not in (x.lower() for x in node):
                    node.append(part)
                break
            else:
                upgraded = {x: [] for x in node}
                parent[key_in_parent] = upgraded
                node = upgraded
                continue
        else:
            upgraded = {}
            parent[key_in_parent] = upgraded
            node = upgraded
            continue


def _collapse_leaf_dicts(node: Union[Dict, List]) -> Union[Dict, List]:
    """Collapse pure leaf dicts into lists."""
    if isinstance(node, dict):
        if not node:
            return {}
        collapsed = {k: _collapse_leaf_dicts(v) for k, v in node.items()}
        if all(isinstance(v, list) and len(v) == 0 for v in collapsed.values()):
            return list(collapsed.keys())
        return collapsed
    elif isinstance(node, list):
        return [_collapse_leaf_dicts(v) for v in node]
    else:
        return node


def apply_changes(
    tree: Dict[str, Any],
    changes: List[str],
    delimiter: str = "/",
    inplace: bool = True,
    auto_collapse: bool = True,
) -> Dict[str, Any]:
    """Batch insert paths."""
    target = tree if inplace else copy.deepcopy(tree)
    for p in changes:
        insert_path(target, p, delimiter)
    if auto_collapse:
        collapsed = _collapse_leaf_dicts(target)
        if inplace:
            tree.clear()
            tree.update(collapsed)
            return tree
        else:
            return collapsed
    return target


def convert_leaves_to_list(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure leaves are in list format."""
    if isinstance(tree, dict):
        return {k: convert_leaves_to_list(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return tree if tree else {}
    else:
        return tree


def find_leaf_paths_by_node(
    tree: Dict[str, Any], target_leaf_names: List[str], prefix: str = ""
) -> List[str]:
    """Find complete paths for specified leaf nodes."""
    matches = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if not value and key in target_leaf_names:
                    matches.append(new_prefix)
                else:
                    matches.extend(
                        find_leaf_paths_by_node(value, target_leaf_names, new_prefix)
                    )
            elif isinstance(value, list):
                for item in value:
                    if item in target_leaf_names:
                        matches.append(f"{new_prefix}/{item}")
            else:
                if value in target_leaf_names:
                    matches.append(new_prefix)
    return matches


def remove_paths(
    tree: Dict[str, Any], paths: List[str], inplace: bool = False
) -> Dict[str, Any]:
    """Remove specified paths from tree."""
    if not inplace:
        tree = copy.deepcopy(tree)

    def delete_path(node, path_parts):
        if not path_parts:
            return False
        key = path_parts[0]
        if isinstance(node, dict):
            if key not in node:
                return False
            if len(path_parts) == 2 and isinstance(node[key], list):
                value_to_remove = path_parts[1]
                if value_to_remove in node[key]:
                    node[key].remove(value_to_remove)
                    if not node[key]:
                        del node[key]
                    return True
            if len(path_parts) == 1:
                del node[key]
                return True
            child_deleted = delete_path(node[key], path_parts[1:])
            if isinstance(node[key], dict) and not node[key]:
                del node[key]
                return True
            elif isinstance(node[key], list) and not node[key]:
                del node[key]
                return True
            return child_deleted
        return False

    for path in paths:
        if not path or not isinstance(path, str):
            continue
        path_parts = [p for p in path.split("/") if p]
        delete_path(tree, path_parts)

    return tree


def pre_order_traversal_to_list(feature_dict: Dict[str, Any]) -> List[str]:
    """Pre-order traverse tree to list."""
    result = []
    if not isinstance(feature_dict, dict):
        return result
    for key, value in feature_dict.items():
        result.append(key)
        if isinstance(value, dict):
            result.extend(pre_order_traversal_to_list(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result.extend(pre_order_traversal_to_list(item))
                elif isinstance(item, (str, int)):
                    result.append(item)
    return result


def build_repo_info(repo_data: Dict[str, Any]) -> str:
    """Build repository information string.

    Extracts repo metadata and spec fields. If background_and_overview or
    functional_requirements are not at the top level, they are parsed from
    the repository_specification field. List/dict values are serialized as
    readable JSON.
    """
    merged = dict(repo_data)
    spec_keys = ["background_and_overview", "functional_requirements",
                 "non_functional_requirements"]

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


def convert_component_to_features(
    component_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert component list to feature tree format.

    Extracts refactored_subtree from each component and merges them into a single tree.
    Removes purpose, estimate_size, actual_size, util_percent fields.

    Args:
        component_list: List of component dictionaries with refactored_subtree

    Returns:
        Merged feature tree dictionary
    """
    if not component_list or not isinstance(component_list, list):
        return {}

    merged_tree = {}

    for component in component_list:
        if not isinstance(component, dict):
            continue

        # Get the refactored_subtree from component
        subtree = component.get("refactored_subtree", {})

        if not subtree or not isinstance(subtree, dict):
            continue

        # Merge subtree into merged_tree
        # Use component name as top-level key if subtree content should be namespaced
        # Otherwise, merge directly (which may cause key conflicts)
        component_name = component.get("name", "")

        # Option 1: Merge directly (may cause conflicts if keys overlap)
        # merged_tree.update(subtree)

        # Option 2: Use component name as wrapper (safer, preserves structure)
        if component_name:
            # Create a clean key name from component name
            # clean_name = component_name.replace(" ", "_").replace("&", "and").lower()
            merged_tree[component_name] = subtree
        else:
            # If no name, merge directly
            merged_tree.update(subtree)

    return merged_tree


# ============================================================================
# Core Refactoring Class
# ============================================================================


class FeatureTreeRefactor:
    """Feature tree refactorer (simplified version)."""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 20):
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
        self.subtree_plans = []

        # Tracking variables
        self.current_feature_tree = {}
        self.total_leaf_paths = 0  # Total paths count (primary tracking metric)
        self.assigned_paths_count = 0  # Assigned paths count
        self.feature_tree_len = 0

        # Iteration history
        self.iteration_history = []

    def refactor(
        self, feature_tree: Dict[str, Any], repo_data: dict = None
    ) -> Dict[str, Any]:
        """Execute complete two-step refactoring workflow."""
        self.logger.info("=" * 70)
        self.logger.info("Starting feature tree refactoring")
        self.logger.info("=" * 70)

        # Initialize tracking variables
        self.current_feature_tree = feature_tree
        all_leaf_paths = get_all_leaf_paths(feature_tree)
        self.total_leaf_paths = len(all_leaf_paths)
        self.assigned_paths_count = 0
        self.feature_tree_len = len(set(pre_order_traversal_to_list(feature_tree)))

        self.logger.info(
            f"Feature tree statistics: {self.total_leaf_paths} leaf paths, "
            f"{self.feature_tree_len} total nodes"
        )

        # Step 1: Plan subtrees
        planning = self._step1_plan_subtrees(feature_tree, repo_data)
        if not planning:
            return {"error": "Planning step failed", "Features": feature_tree}

        # Step 2: Organize features
        components = self._step2_organize_features(feature_tree, repo_data)
        if not components:
            return {"error": "Organization step failed", "Features": feature_tree}

        # Build result (maintaining Features and Component key compatibility)
        # Inherit descriptions from input feature tree to refactored subtrees
        input_descs = get_all_leaf_descriptions(feature_tree)
        if input_descs:
            for comp in components:
                subtree = comp.get("refactored_subtree", {})
                if subtree:
                    from feature_build import attach_descriptions
                    comp["refactored_subtree"] = attach_descriptions(subtree, input_descs)

        statistics = self._calculate_statistics(components)

        repo_name = repo_data.get("repository_name", "Unknown")
        result = {
            "repository_name": repo_name,
            "repository_purpose": repo_data.get("repository_purpose", ""),
            "repository_specification": json.dumps(
                repo_data.get("repository_specification", {}), indent=2
            ),
            "meta": metadata_with_languages(repo_data),
            "features": feature_tree,
            "components": components,
            # "components_format": convert_component_to_features(components),
            "planning_result": {
                "total_subtrees": planning.total_subtrees,
                "subtree_plans": [p.model_dump() for p in planning.subtree_plans],
                "reasoning": planning.reasoning,
            },
            "statistics": statistics,
            "llm_call_history": [r.to_dict() for r in self.llm.get_call_history()],
            "iteration_history": self.iteration_history,
        }

        # Print detailed statistics table
        self._print_statistics_table(components, statistics)

        return result

    def _step1_plan_subtrees(
        self, feature_tree: Dict[str, Any], repo_data: Dict[str, Any]
    ) -> Optional[SubtreePlanningOutput]:
        """Step 1: Plan subtree structure."""
        self.logger.info("\n" + "-" * 70)
        self.logger.info("[Step 1] Planning subtree structure...")
        self.logger.info("-" * 70)

        # Build user prompt
        all_paths = get_all_leaf_paths(feature_tree)
        feature_tree_json = json.dumps(feature_tree, indent=2)

        user_prompt = f"""## Repository Information:
{build_repo_info(repo_data)}

## Feature Tree to Refactor:
**Total Features**: {len(all_paths)}

```json
{feature_tree_json}
```

## Task:
Analyze the feature tree and design a logical organization into functional subtrees/components.
Each subtree should represent a coherent functional area or module.

Provide your subtree planning with:
1. Appropriate number of subtrees (determined by domain analysis, not fixed rules)
2. Clear names for each subtree
3. Purpose/theme for each subtree
4. Estimated feature count for each subtree

Consider:
- Natural domain boundaries in the system
- Functional cohesion (related features together)
- Repository purpose and domain
- Clear separation of concerns
- Quality indicators: cohesion, naming clarity, balanced sizes
"""

        # Call LLM
        _, result, _ = self.llm.call_structured(
            system_prompt=PROMPT_TEMPLATE_SUBTREE_PLANNING,
            user_prompt=user_prompt,
            response_model=SubtreePlanningOutput,
            purpose="step1_planning",
        )

        if result:
            self.subtree_plans = result.subtree_plans
            self.logger.info(f"\n[OK] Planned {len(self.subtree_plans)} subtrees:")
            for i, plan in enumerate(self.subtree_plans, 1):
                self.logger.info(f"  {i}. {plan.name}")
                self.logger.info(f"     Purpose: {plan.purpose}")
                self.logger.info(f"     Estimated: ~{plan.estimate_size} features")
            return result

        return None

    def _step2_organize_features(
        self, feature_tree: Dict[str, Any], repo_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Step 2: Iteratively organize features into subtrees."""
        self.logger.info("\n" + "-" * 70)
        self.logger.info("[Step 2] Organizing features (iterative)...")
        self.logger.info("-" * 70)

        if not self.subtree_plans:
            self.logger.error("[FAIL] No subtree plans")
            return None

        # Initialize
        all_paths = set(get_all_leaf_paths(feature_tree))
        remaining_paths = all_paths.copy()

        # Initialize subtree structure
        subtrees = [
            {
                "name": plan.name,
                "purpose": plan.purpose,
                # "estimate_size": plan.estimate_size,
                "refactored_subtree": {},
                "actual_size": 0,
                "util_percent": 0.0,
            }
            for plan in self.subtree_plans
        ]

        # Error handling: consecutive failure count
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Iterative assignment
        for iteration in range(1, self.max_iterations + 1):
            if not remaining_paths:
                self.logger.info("\n[OK] All paths assigned")
                break

            # Calculate utilization based on paths (not leaf names)
            utilization = self.assigned_paths_count / self.total_leaf_paths
            if utilization >= 0.99:
                self.logger.info("\n[OK] Reached 99% utilization, stopping iteration")
                break

            if consecutive_failures >= max_consecutive_failures:
                self.logger.warning(
                    f"\n[FAIL] {max_consecutive_failures} consecutive iterations failed, stopping"
                )
                break

            remaining_count = len(remaining_paths)
            self.logger.info(f"\n>>> Iteration {iteration}/{self.max_iterations}")
            self.logger.info(f"    Remaining: {remaining_count} paths")
            self.logger.info(f"    Utilization rate: {utilization:.1%}")
            # Progress estimation
            if iteration > 1:
                avg_assigned_per_iter = self.assigned_paths_count / (iteration - 1)
                remaining_to_assign = self.total_leaf_paths - self.assigned_paths_count
                estimated_iters = (
                    int(remaining_to_assign / avg_assigned_per_iter) + 1
                    if avg_assigned_per_iter > 0
                    else self.max_iterations - iteration
                )
                self.logger.info(
                    f"    Estimated still needed: ~{estimated_iters} iterations"
                )

            # Record iteration start state
            iteration_record = {
                "iteration": iteration,
                "remaining_paths_count": remaining_count,
                "assigned_paths_before": self.assigned_paths_count,
                "utilization_before": utilization,
            }

            # Build current feature tree
            current_tree = apply_changes({}, list(remaining_paths), inplace=False)
            current_tree = convert_leaves_to_list(current_tree)

            # Build user prompt
            user_prompt = self._build_organization_prompt(
                repo_data, subtrees, current_tree, len(remaining_paths), utilization
            )

            # Call LLM
            _, result, _ = self.llm.call_structured(
                system_prompt=PROMPT_TEMPLATE_FEATURE_ORGANIZATION,
                user_prompt=user_prompt,
                response_model=FeatureOrganizationOutput,
                purpose=f"step2_iteration_{iteration}",
            )

            if not result:
                consecutive_failures += 1
                self.logger.warning(
                    f"    [FAIL] LLM call failed (consecutive failures: {consecutive_failures})"
                )
                continue

            # Process assignment results
            iteration_assigned, remaining_paths = self._process_assignments(
                result, subtrees, remaining_paths
            )

            if iteration_assigned > 0:
                consecutive_failures = 0  # Reset failure count
                self.assigned_paths_count += iteration_assigned
                self.logger.info(
                    f"    [OK] Assigned this iteration: {iteration_assigned} paths"
                )
                # Update iteration record
                new_utilization = self.assigned_paths_count / self.total_leaf_paths
                iteration_record["assigned_this_iteration"] = iteration_assigned
                iteration_record["remaining_paths_after"] = len(remaining_paths)
                iteration_record["assigned_paths_after"] = self.assigned_paths_count
                iteration_record["utilization_after"] = new_utilization
                iteration_record["status"] = "success"
            else:
                consecutive_failures += 1
                self.logger.warning(
                    f"    [FAIL] No progress (consecutive failures: {consecutive_failures})"
                )
                iteration_record["assigned_this_iteration"] = 0
                iteration_record["remaining_paths_after"] = iteration_record[
                    "remaining_paths_count"
                ]
                iteration_record["assigned_paths_after"] = iteration_record[
                    "assigned_paths_before"
                ]
                iteration_record["utilization_after"] = iteration_record[
                    "utilization_before"
                ]
                iteration_record["status"] = "failed"

            # Save iteration record
            self.iteration_history.append(iteration_record)

        # Calculate final statistics
        self._finalize_subtrees(subtrees)

        final_util = self.assigned_paths_count / self.total_leaf_paths
        unassigned_count = self.total_leaf_paths - self.assigned_paths_count
        self.logger.info(f"\n[OK] Organization completed ({iteration} iterations)")
        self.logger.info(f"  Final utilization rate: {final_util:.1%}")
        self.logger.info(f"  Remaining unassigned: {unassigned_count} paths")

        return subtrees

    def _build_organization_prompt(
        self,
        repo_data: Dict[str, Any],
        subtrees: List[Dict],
        current_tree: Dict,
        remaining_count: int,
        utilization: float,
    ) -> str:
        """Build prompt for organization step."""
        # Filter out estimate_size to avoid misleading LLM to "match the numbers"
        # Only pass fields useful for assignment decisions
        subtrees_for_prompt = [
            {
                "name": s["name"],
                "purpose": s["purpose"],
                "actual_size": s["actual_size"],
                "refactored_subtree": s["refactored_subtree"],
            }
            for s in subtrees
        ]
        subtrees_json = json.dumps(subtrees_for_prompt, indent=2)
        current_tree_json = json.dumps(current_tree, indent=2)

        return f"""## Repository Information:
{build_repo_info(repo_data)}

## Current Subtrees Status:
{subtrees_json}

## Remaining Feature Tree ({remaining_count} paths to organize):
Current Utilization: {utilization:.1%}

```json
{current_tree_json}
```

## Task
Refactor the remaining feature leaves by rebuilding their paths so they fit the repository subtrees,
the existing feature subgraphs, and the actual semantics of the repo.
The original feature tree is provided only as semantic context and must not be preserved as-is.

Rules:
1. Operate only on leaf nodes from the remaining feature tree.
2. For each leaf, construct a new full path under the appropriate subtree.
   The refactored path must have 2-8 segments joined by '/', with the leaf as the final segment.
3. The new full path for a leaf must not be identical to its original full path string
   from the remaining feature tree.
4. Do not follow or mirror the original intermediate hierarchy; reorganize leaves according to
   subtree purposes, repository architecture, and feature subgraph relationships.
5. Each leaf must appear exactly once across all assigned paths.
6. Do not invent, rename, or modify leaf names.

Return only the new subtree path assignments.
"""

    def _process_assignments(
        self,
        result: FeatureOrganizationOutput,
        subtrees: List[Dict],
        remaining_paths: set,
    ) -> tuple[int, set]:
        """Process assignment results returned by LLM."""
        iteration_assigned = 0
        total_proposed = 0
        total_rejected_depth = 0
        total_rejected_leaf = 0

        # Calculate remaining leaf nodes
        remaining_tree = apply_changes({}, list(remaining_paths), inplace=False)
        remaining_leaf_nodes_set = set(extract_leaf_nodes(remaining_tree))

        for assignment in result.assignments:
            valid_paths = []
            rejected_depth = []
            rejected_leaf = []

            # Validate each path
            for path in assignment.assigned_paths:
                total_proposed += 1
                parts = path.split("/")

                # Check 1: Depth must be 2-8 segments
                if len(parts) < 2 or len(parts) > 8:
                    rejected_depth.append(path)
                    total_rejected_depth += 1
                    continue

                # Check 2: Leaf node must be in remaining tree (and not already assigned in this iteration)
                leaf_name = parts[-1]
                if leaf_name not in remaining_leaf_nodes_set:
                    rejected_leaf.append(path)
                    total_rejected_leaf += 1
                    continue

                valid_paths.append(path)
                # Remove from remaining set to prevent duplicate assignment within same iteration
                remaining_leaf_nodes_set.discard(leaf_name)

            # Record rejection information
            if rejected_depth:
                self.logger.info(
                    f"      [FAIL] Rejected {len(rejected_depth)} paths with depth errors (expected 2-8 segments)"
                )
                for p in rejected_depth[:5]:
                    self.logger.info(f"        - {p} (depth: {len(p.split('/'))})")
                if len(rejected_depth) > 5:
                    self.logger.info(f"        ... and {len(rejected_depth) - 5} more")
            if rejected_leaf:
                self.logger.info(
                    f"      [FAIL] Rejected {len(rejected_leaf)} paths with non-existent/already assigned leaves"
                )
                for p in rejected_leaf[:5]:
                    leaf = p.split("/")[-1]
                    self.logger.info(f"        - {p} (leaf '{leaf}' not in remaining)")
                if len(rejected_leaf) > 5:
                    self.logger.info(f"        ... and {len(rejected_leaf) - 5} more")

            if not valid_paths:
                continue

            # Update corresponding subtree
            for subtree in subtrees:
                if subtree["name"] == assignment.subtree_name:
                    current_paths = get_all_leaf_paths(subtree["refactored_subtree"])
                    current_paths.extend(valid_paths)
                    subtree["refactored_subtree"] = apply_changes({}, current_paths)
                    subtree["refactored_subtree"] = convert_leaves_to_list(
                        subtree["refactored_subtree"]
                    )
                    break

            iteration_assigned += len(valid_paths)

            self.logger.info(
                f"      → {assignment.subtree_name}: +{len(valid_paths)} paths"
            )

        # Output iteration statistics
        if total_proposed > 0:
            acceptance_rate = iteration_assigned / total_proposed * 100
            self.logger.info(
                f"    Iteration stats: proposed {total_proposed} paths, "
                f"accepted {iteration_assigned} ({acceptance_rate:.1f}%), "
                f"rejected {total_rejected_depth + total_rejected_leaf} "
                f"(depth: {total_rejected_depth}, leaf: {total_rejected_leaf})"
            )

        # Recalculate remaining paths (based on leaf nodes)
        if iteration_assigned > 0:
            selected_feature_paths = []
            for subtree in subtrees:
                sub_tree = subtree.get("refactored_subtree", {})
                if sub_tree:
                    parsed_leaf_nodes = extract_leaf_nodes(sub_tree)
                    valid_leaf_paths = find_leaf_paths_by_node(
                        self.current_feature_tree, target_leaf_names=parsed_leaf_nodes
                    )
                    selected_feature_paths.extend(valid_leaf_paths)

            selected_feature_paths = list(set(selected_feature_paths))
            filter_feature_tree = remove_paths(
                self.current_feature_tree, selected_feature_paths, inplace=False
            )
            remaining_paths = set(get_all_leaf_paths(filter_feature_tree))

        return iteration_assigned, remaining_paths

    def _finalize_subtrees(self, subtrees: List[Dict]) -> None:
        """Calculate final subtree statistics."""
        total_leaf_paths = self.total_leaf_paths

        for subtree in subtrees:
            # Use path count instead of unique leaf names for accurate statistics
            subtree_paths = (
                get_all_leaf_paths(subtree["refactored_subtree"])
                if subtree["refactored_subtree"]
                else []
            )
            subtree["actual_size"] = len(subtree_paths)
            subtree["util_percent"] = (
                len(subtree_paths) / total_leaf_paths if total_leaf_paths > 0 else 0.0
            )

    def _calculate_statistics(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics."""
        # Use path count for accurate statistics (handles duplicate leaf names)
        total_paths = self.total_leaf_paths
        # Calculate assigned paths by summing actual paths in refactored subtrees
        assigned_paths = sum(
            len(get_all_leaf_paths(c["refactored_subtree"]))
            for c in components
            if c["refactored_subtree"]
        )

        return {
            "original_leaf_count": total_paths,
            "assigned_leaf_count": assigned_paths,
            "unassigned_leaf_count": total_paths - assigned_paths,
            "coverage_rate": assigned_paths / total_paths if total_paths > 0 else 0,
            "subtree_count": len(components),
        }

    def _print_statistics_table(
        self, components: List[Dict[str, Any]], statistics: Dict[str, Any]
    ) -> None:
        """Print statistics in table format."""
        print("\n" + "=" * 80)
        print("REFACTORING SUMMARY")
        print("=" * 80)

        # 1. Structure comparison - original vs refactored
        # Count top-level categories in original feature tree using path count (consistent with statistics)
        original_categories = {}
        for top_category, subtree in self.current_feature_tree.items():
            # Use get_all_leaf_paths for consistent counting
            leaf_count = len(get_all_leaf_paths({top_category: subtree}))
            original_categories[top_category] = leaf_count

        # Sort by category name
        sorted_original = sorted(original_categories.items(), key=lambda x: x[0])

        # Sort by component name
        sorted_components = sorted(components, key=lambda c: c["name"])

        # Build comparison rows - show side by side without implying correspondence
        comparison_rows = []
        max_rows = max(len(sorted_original), len(sorted_components))

        for i in range(max_rows):
            original_cat = sorted_original[i][0] if i < len(sorted_original) else ""
            original_count = sorted_original[i][1] if i < len(sorted_original) else ""

            # Add arrow separator only in the middle
            separator = "→" if i == max_rows // 2 else ""

            component_name = (
                sorted_components[i]["name"] if i < len(sorted_components) else ""
            )
            component_count = (
                sorted_components[i]["actual_size"]
                if i < len(sorted_components)
                else ""
            )

            comparison_rows.append(
                [
                    original_cat,
                    original_count,
                    separator,
                    component_name,
                    component_count,
                ]
            )

        # Add total row
        comparison_rows.append(
            [
                "TOTAL",
                sum(original_categories.values()),
                "→",
                "TOTAL",
                statistics["assigned_leaf_count"],
            ]
        )

        print("\n")
        print_unicode_table(
            headers=["Original Category", "Count", "", "Refactored Component", "Count"],
            rows=comparison_rows,
            title="Feature Tree Refactored",
        )

        # 2. Iteration process
        if self.iteration_history:
            iteration_rows = []
            for log in self.iteration_history:
                status = "[OK]" if log.get("status") == "success" else "[FAIL]"
                assigned = log.get("assigned_this_iteration", 0)
                # Use remaining count after iteration ends
                remaining_after = log.get(
                    "remaining_paths_after",
                    log.get("remaining_paths_count", 0),
                )
                progress_after = log.get(
                    "utilization_after", log.get("utilization_before", 0)
                )

                iteration_rows.append(
                    [
                        log["iteration"],
                        status,
                        assigned,
                        remaining_after,
                        f"{progress_after:.1%}",
                    ]
                )

            print("\n")
            print_unicode_table(
                headers=["Iteration", "Status", "Assigned", "Remaining", "Progress"],
                rows=iteration_rows,
                title="Iteration Process",
            )

        print("\n" + "=" * 80)


# ============================================================================
# Main Program
# ============================================================================


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Feature Tree Refactoring Tool - Organize flat feature tree into modular components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python refactor_simple.py --input feature_build.json --output refactored.json
  python refactor_simple.py --input feature_build.json --max-iterations 15
        """,
    )

    parser.add_argument(
        "--input",
        help="Input feature tree JSON file (feature_build.json format)",
        default=str(FEATURE_BUILD_FILE),
    )
    parser.add_argument(
        "--output",
        default=str(FEATURE_TREE_FILE),
        help=f"Output result file (default: {FEATURE_TREE_FILE})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        # Load input data
        logger.info(f"Reading input file: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            repo_specification_data = json.load(f)

        try:
            with open(args.output, "r", encoding="utf-8") as f:
                output_data = json.load(f)
        except Exception:
            output_data = {}

        # Extract feature tree and repository information
        components = output_data.get("components", [])
        if len(components) == 0:
            feature_tree = repo_specification_data.get("feature_tree")
        else:
            # Convert component list to feature_tree format
            component_list = output_data.get("components", [])
            feature_tree = convert_component_to_features(component_list)
            logger.info(
                f"Converting feature tree from existing {len(component_list)} components"
            )

        if not feature_tree:
            logger.error("[FAIL] Input file missing 'feature_tree' key")
            return 1

        repo_name = repo_specification_data.get("repository_name", "Unknown")
        repository_purpose = repo_specification_data.get("repository_purpose", "")
        background_and_overview = repo_specification_data.get(
            "background_and_overview", []
        )
        functional_requirements = repo_specification_data.get(
            "functional_requirements", []
        )

        repo_data = {
            "repository_name": repo_name,
            "repository_purpose": repository_purpose,
            "background_and_overview": background_and_overview,
            "functional_requirements": functional_requirements,
            "repository_specification": repo_specification_data,
            "meta": metadata_with_languages(repo_specification_data),
        }
        # Create LLM client
        # Initialize trajectory
        trajectory = None
        step_id = None
        if not args.no_trajectory:
            trajectory = load_or_create_trajectory("feature_refactor")
            trajectory.start(metadata={
                "input_file": args.input,
                "output_file": args.output,
                "max_iterations": args.max_iterations,
            })
            step = trajectory.add_step("feature_refactor", "Refactor flat feature tree into modular components")
            trajectory.start_step(step.step_id)
            step_id = step.step_id

        llm_client = LLMClient(trajectory=trajectory, step_id=step_id)

        # Create refactorer and execute
        refactor = FeatureTreeRefactor(llm_client, max_iterations=args.max_iterations)
        result = refactor.refactor(feature_tree, repo_data)

        # Save results
        logger.info(f"\nSaving results to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'=' * 70}")
        logger.info("[OK] Refactoring complete!")
        logger.info(f"{'=' * 70}\n")

        # Mark trajectory as complete
        if trajectory:
            components = result.get("components", [])
            if step_id is not None:
                trajectory.complete_step(step_id, {
                    "components_count": len(components),
                })
            trajectory.complete(metadata={
                "components_count": len(components),
            })
            logger.info(f"[OK] Trajectory saved to: {trajectory.trajectory_file}")

        return 0

    except FileNotFoundError:
        logger.error(f"[FAIL] Input file not found: {args.input}")
        if trajectory:
            if step_id is not None:
                trajectory.fail_step(step_id, "Input file not found")
            trajectory.fail("Input file not found")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"[FAIL] JSON parsing error: {e}")
        if trajectory:
            if step_id is not None:
                trajectory.fail_step(step_id, str(e))
            trajectory.fail(str(e))
        return 1
    except Exception as e:
        logger.error(f"[FAIL] Execution error: {e}", exc_info=True)
        if trajectory:
            if step_id is not None:
                trajectory.fail_step(step_id, str(e))
            trajectory.fail(str(e))
        return 1


if __name__ == "__main__":
    exit(main())
