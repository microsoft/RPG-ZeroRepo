#!/usr/bin/env python3
"""Edit Feature Tree Script.

Workflow:
1. Planning - Analyze all components and generate an edit plan.
2. Execution - Apply the plan precisely to each component.
3. Review - Verify changes and auto-fix if needed (up to 3 rounds).

Input/Output: .cmind/data/feature_tree.json
"""

import json
import logging
import argparse
import copy
import time
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from feature.prompts import (
    PROMPT_TEMPLATE_EDIT_PLAN,
    PROMPT_TEMPLATE_EDIT_REVIEW,
)
from common.paths import FEATURE_TREE_FILE
from common import print_unicode_table, get_all_leaf_paths
from common.llm_client import LLMClient
from common.trajectory import load_or_create_trajectory

# ============================================================================
# Utility Functions
# ============================================================================


def count_leaf_nodes(tree: Dict[str, Any]) -> int:
    """Count only leaf nodes in the tree."""
    if not tree:
        return 0

    count = 0
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                if not value:
                    count += 1
                else:
                    count += count_leaf_nodes(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
    elif isinstance(tree, list):
        count = len(tree)

    return count


def find_duplicate_features(components: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Find features that exist in multiple components (potential issues from move operations)."""
    # Extract leaf names (last part of path) from each component
    feature_locations = {}  # feature_name -> list of component names

    for comp in components:
        name = comp.get("name", "Unknown")
        subtree = comp.get("refactored_subtree", {})
        paths = get_all_leaf_paths(subtree)

        for path in paths:
            # Get the leaf name (last part)
            leaf_name = path.split("/")[-1].lower() if "/" in path else path.lower()
            if leaf_name not in feature_locations:
                feature_locations[leaf_name] = []
            feature_locations[leaf_name].append((name, path))

    # Find duplicates (same leaf name in multiple components)
    duplicates = {}
    for feature, locations in feature_locations.items():
        if len(locations) > 1:
            # Check if it's the same feature (not just same name)
            unique_components = set(loc[0] for loc in locations)
            if len(unique_components) > 1:
                duplicates[feature] = [(comp, path) for comp, path in locations]

    return duplicates


# ============================================================================
# Tree Operation Functions
# ============================================================================


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
        return [
            _collapse_leaf_dicts(v) if isinstance(v, (dict, list)) else v for v in node
        ]
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


def remove_paths(
    tree: Dict[str, Any], paths: List[str], inplace: bool = False
) -> tuple[Dict[str, Any], Dict[str, bool]]:
    """Remove specified paths from tree.

    Returns:
        tuple: (modified_tree, removal_results)
               removal_results is a dict mapping path -> bool (True if actually removed)
    """
    if not inplace:
        tree = copy.deepcopy(tree)

    removal_results = {}

    def delete_path(node, path_parts):
        if not path_parts:
            return False
        key = path_parts[0]
        if isinstance(node, dict):
            matched_key = next((k for k in node if k.lower() == key.lower()), None)
            if matched_key is None:
                return False

            if len(path_parts) == 2 and isinstance(node[matched_key], list):
                value_to_remove = path_parts[1]
                found = False
                for item in node[matched_key]:
                    item_name = item.get("name", "") if isinstance(item, dict) else item
                    if isinstance(item_name, str) and item_name.lower() == value_to_remove.lower():
                        node[matched_key].remove(item)
                        found = True
                        break
                if not found:
                    return False
                if not node[matched_key]:
                    del node[matched_key]
                return True
            if len(path_parts) == 1:
                del node[matched_key]
                return True
            child_deleted = delete_path(node[matched_key], path_parts[1:])
            if isinstance(node[matched_key], dict) and not node[matched_key]:
                del node[matched_key]
                return True
            elif isinstance(node[matched_key], list) and not node[matched_key]:
                del node[matched_key]
                return True
            return child_deleted
        return False

    for path in paths:
        if not path or not isinstance(path, str):
            removal_results[path] = False
            continue
        path_parts = [p for p in path.split("/") if p]
        was_deleted = delete_path(tree, path_parts)
        removal_results[path] = was_deleted

    return tree, removal_results


# ============================================================================
# Pydantic Data Models
# ============================================================================


class ComponentOperation(BaseModel):
    """Single operation for a specific component."""

    component_name: str = Field(description="Name of the component to modify")
    operation_type: str = Field(description="Type: DELETE, ADD, or MODIFY")
    paths_to_remove: List[str] = Field(
        default_factory=list, description="Paths to remove from this component"
    )
    paths_to_add: List[str] = Field(
        default_factory=list, description="Paths to add to this component"
    )
    reason: str = Field(description="Brief explanation of why this operation is needed")


class EditPlan(BaseModel):
    """Complete edit plan generated by the planning step."""

    summary: str = Field(description="Overall summary of the edit plan")
    operations: List[ComponentOperation] = Field(
        description="List of operations to perform"
    )
    is_valid: bool = Field(description="Whether the plan is valid and can be executed")
    validation_notes: str = Field(default="", description="Notes about plan validation")


class ReviewResult(BaseModel):
    """Review result generated by the review step."""

    thinking: str = Field(description="Detailed thinking process of the review")
    summary: str = Field(
        description="Human-readable summary of what was edited and the result"
    )
    execution_matches_plan: bool = Field(
        description="Whether execution result matches the plan"
    )
    execution_matches_intent: bool = Field(
        description="Whether execution result matches user's intent"
    )
    issues_found: List[str] = Field(
        default_factory=list, description="List of issues found during review"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )
    overall_success: bool = Field(description="Overall success of the edit operation")
    confidence_score: float = Field(description="Confidence score 0.0-1.0")
    needs_fix: bool = Field(
        default=False, description="Whether fix operations are needed"
    )
    fix_operations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Operations to fix the issues found"
    )


# ============================================================================
# Feature Tree Editor
# ============================================================================


class FeatureTreeEditor:
    """Feature tree editor with planning, execution, and review steps."""

    def __init__(self, llm_client: LLMClient, enable_review: bool = True):
        self.llm = llm_client
        self.enable_review = enable_review
        self.logger = logging.getLogger(__name__)

        # Tracking
        self.operations_executed = []
        self.paths_deleted = []
        self.paths_added = []

        # State snapshots for review
        self.state_before = {}
        self.state_after = {}

    def edit(
        self,
        components: List[Dict[str, Any]],
        edit_instruction: str,
        repo_data: Dict[str, Any],
        model_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the planning, execution, and review workflow."""
        # Capture state before edit
        self.state_before = self._capture_state(components)

        self.logger.info("=" * 70)
        self.logger.info("STEP: PLANNING")
        self.logger.info("=" * 70)

        # Build components summary for planning
        components_summary = self._build_components_summary(components)

        # Generate edit plan.
        plan = self._generate_edit_plan(components_summary, edit_instruction, repo_data)

        if plan is None:
            self.logger.error("[FAIL] Failed to generate edit plan")
            return {"success": False, "error": "Failed to generate edit plan"}

        if not plan.is_valid:
            self.logger.error(f"[FAIL] Plan is invalid: {plan.validation_notes}")
            return {"success": False, "error": f"Invalid plan: {plan.validation_notes}"}

        # Display the plan
        self._display_plan(plan)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP: EXECUTION")
        self.logger.info("=" * 70)

        # Execute the plan.
        execution_results = self._execute_plan(plan, components)

        # Capture state after edit
        self.state_after = self._capture_state(components)

        # Build result
        result = {
            "success": True,
            "plan": plan,
            "plan_summary": plan.summary,
            "operations_executed": self.operations_executed,
            "paths_deleted": self.paths_deleted,
            "paths_added": self.paths_added,
            "execution_results": execution_results,
        }

        # Review with auto-fix loop (max 3 iterations).
        if self.enable_review:
            MAX_REVIEW_ITERATIONS = 3
            review_iterations = []
            final_status = "UNKNOWN"
            review_result = None  # Initialize to avoid reference before assignment

            for review_round in range(1, MAX_REVIEW_ITERATIONS + 1):
                self.logger.info("\n" + "=" * 70)
                self.logger.info(
                    f"PHASE 3: REVIEW (Round {review_round}/{MAX_REVIEW_ITERATIONS})"
                )
                self.logger.info("=" * 70)

                # Update state snapshot before review
                self.state_after = self._capture_state(components)

                review_result = self._review_execution(
                    edit_instruction=edit_instruction,
                    plan=plan,
                    execution_results=execution_results,
                    model_analysis=model_analysis,
                    components=components,  # Pass components for duplicate detection
                )

                if review_result is None:
                    self.logger.warning("[WARNING] Review failed, skipping...")
                    final_status = "REVIEW_FAILED"
                    break

                # Store review iteration info
                iteration_info = {
                    "round": review_round,
                    "execution_matches_plan": review_result.execution_matches_plan,
                    "execution_matches_intent": review_result.execution_matches_intent,
                    "issues_found": review_result.issues_found,
                    "overall_success": review_result.overall_success,
                    "confidence_score": review_result.confidence_score,
                    "needs_fix": review_result.needs_fix,
                    "fix_operations_count": len(review_result.fix_operations)
                    if review_result.fix_operations
                    else 0,
                }
                review_iterations.append(iteration_info)

                # Display review results
                self._display_review(review_result, review_round)

                # Case 1: Success - no issues, no fixes needed
                if review_result.overall_success and not review_result.needs_fix:
                    final_status = "SUCCESS"
                    print("\n" + "=" * 70)
                    print("[OK] REVIEW COMPLETE - ALL CHANGES VERIFIED")
                    print("=" * 70)
                    print(f"\n  Review passed after {review_round} round(s)")
                    print(f"  Confidence: {review_result.confidence_score:.2f}")
                    if review_result.summary:
                        print("\n  Final Summary:")
                        print(f"     {review_result.summary}")
                    print("\n" + "=" * 70)
                    break

                # Case 2: Issues found, fixes needed
                if review_result.needs_fix and review_result.fix_operations:
                    print("\n  [WARNING]  Status: Issues detected, applying fixes...")
                    self.logger.info(
                        f"\n  Applying {len(review_result.fix_operations)} fix operations..."
                    )

                    # Execute fix operations
                    fix_results = self._execute_fix_operations(
                        review_result.fix_operations, components
                    )

                    # Update execution results
                    execution_results.extend(fix_results)
                    result["execution_results"] = execution_results

                    print("  [OK] Fix operations completed")

                    # Check if this is the last round
                    if review_round < MAX_REVIEW_ITERATIONS:
                        print(
                            f"  → Proceeding to verification round {review_round + 1}..."
                        )
                        continue
                    else:
                        final_status = "MAX_ITERATIONS_REACHED"
                        print(
                            f"\n  [WARNING]  Maximum review iterations ({MAX_REVIEW_ITERATIONS}) reached"
                        )
                        break

                # Case 3: Issues found but no fix operations provided
                if review_result.issues_found and not review_result.fix_operations:
                    if review_result.overall_success:
                        # Minor issues that don't affect success
                        final_status = "SUCCESS_WITH_WARNINGS"
                        print("\n" + "=" * 70)
                        print(
                            "[OK] REVIEW COMPLETE - CHANGES VERIFIED (with minor notes)"
                        )
                        print("=" * 70)
                    else:
                        final_status = "ISSUES_UNRESOLVED"
                        print("\n" + "=" * 70)
                        print("[WARNING]  REVIEW COMPLETE - UNRESOLVED ISSUES")
                        print("=" * 70)
                    break

                # Case 4: No issues, but overall_success is False (edge case)
                if not review_result.issues_found and not review_result.overall_success:
                    final_status = "UNCERTAIN"
                    print("\n  [WARNING]  Review uncertain, stopping...")
                    break

            # Final summary after all review rounds
            print("\n" + "─" * 70)
            print("REVIEW PROCESS SUMMARY")
            print("─" * 70)
            print(f"\n  Total Review Rounds: {len(review_iterations)}")
            print(f"  Final Status: {final_status}")

            # Show iteration history
            if len(review_iterations) > 1:
                print("\n  Iteration History:")
                for it in review_iterations:
                    status_icon = (
                        "[OK]"
                        if it["overall_success"]
                        else "[FAIL]"
                        if it["needs_fix"]
                        else "?"
                    )
                    fix_info = (
                        f" → {it['fix_operations_count']} fixes applied"
                        if it["fix_operations_count"] > 0
                        else ""
                    )
                    print(
                        f"    Round {it['round']}: {status_icon} (confidence: {it['confidence_score']:.2f}){fix_info}"
                    )

            print("─" * 70)

            # Store final review result
            if review_result:
                result["review"] = {
                    "thinking": review_result.thinking,
                    "summary": review_result.summary,
                    "execution_matches_plan": review_result.execution_matches_plan,
                    "execution_matches_intent": review_result.execution_matches_intent,
                    "issues_found": review_result.issues_found,
                    "suggestions": review_result.suggestions,
                    "overall_success": review_result.overall_success,
                    "confidence_score": review_result.confidence_score,
                    "review_iterations": review_iterations,
                    "total_rounds": len(review_iterations),
                    "final_status": final_status,
                }

                # Update success based on final review
                if final_status not in ["SUCCESS", "SUCCESS_WITH_WARNINGS"]:
                    result["success"] = False
                    result["review_failed"] = True

        # Print summary
        self._print_summary(result, components)

        return result

    def _execute_fix_operations(
        self, fix_operations: List[Dict[str, Any]], components: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute fix operations from review phase."""
        # Build component lookup
        comp_by_name = {comp.get("name"): comp for comp in components}

        fix_results = []

        for op_data in fix_operations:
            # Convert dict to ComponentOperation-like structure
            comp_name = op_data.get("component_name", "")
            comp = comp_by_name.get(comp_name)

            if comp is None:
                self.logger.warning(f"  [WARNING] Fix: Component not found: {comp_name}")
                fix_results.append(
                    {
                        "component": comp_name,
                        "status": "skipped",
                        "reason": "Component not found",
                        "is_fix": True,
                    }
                )
                continue

            subtree = comp.get("refactored_subtree", {})
            initial_count = count_leaf_nodes(subtree)

            paths_to_remove = op_data.get("paths_to_remove", [])
            paths_to_add = op_data.get("paths_to_add", [])
            operation_type = op_data.get("operation_type", "FIX")
            reason = op_data.get("reason", "Fix from review")

            self.logger.info(f"  → Fix: {comp_name} - {operation_type}")
            self.logger.info(f"    Reason: {reason}")

            # Execute DELETE operations
            deleted_count = 0
            failed_deletions = []
            if paths_to_remove:
                for path in paths_to_remove:
                    _, removal_results = remove_paths(subtree, [path], inplace=True)
                    if removal_results.get(path, False):
                        self.paths_deleted.append(f"{comp_name}: {path} (fix)")
                        deleted_count += 1
                        self.logger.info(f"    [FAIL] Removed: {path}")
                    else:
                        failed_deletions.append(path)
                        self.logger.warning(f"    [WARNING] Path not found: {path}")

            # Execute ADD operations
            added_count = 0
            if paths_to_add:
                apply_changes(subtree, paths_to_add, inplace=True)
                for path in paths_to_add:
                    self.paths_added.append(f"{comp_name}: {path} (fix)")
                    added_count += 1
                    self.logger.info(f"    [OK] Added: {path}")

            final_count = count_leaf_nodes(subtree)
            comp["actual_size"] = final_count

            # Determine status
            has_failures = len(failed_deletions) > 0
            if has_failures and deleted_count == 0 and len(paths_to_remove) > 0:
                status = "partial_failure"
            elif has_failures:
                status = "partial_success"
            else:
                status = "success"

            fix_results.append(
                {
                    "component": comp_name,
                    "status": status,
                    "operation_type": f"FIX_{operation_type}",
                    "paths_removed": deleted_count,
                    "paths_added": added_count,
                    "failed_deletions": failed_deletions,
                    "initial_leaf_count": initial_count,
                    "final_leaf_count": final_count,
                    "is_fix": True,
                    "reason": reason,
                }
            )

            # Build fix operation record with new optimized structure
            fix_op_record = {
                "component": comp_name,
                "operation_type": f"FIX_{operation_type}",
                "removed": paths_to_remove,
                "added": paths_to_add,
                "reason": reason,
                "status": status,
                "leaf_count": {"before": initial_count, "after": final_count},
                "is_fix": True,
            }
            # Only add failed field if there are failures
            if failed_deletions:
                fix_op_record["failed"] = failed_deletions

            self.operations_executed.append(fix_op_record)

        return fix_results

    def _capture_state(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Capture current state of all components for comparison."""
        state = {}
        for comp in components:
            name = comp.get("name", "Unknown")
            subtree = comp.get("refactored_subtree", {})
            state[name] = {
                "leaf_count": count_leaf_nodes(subtree),
                "paths": get_all_leaf_paths(subtree),
            }
        return state

    def _build_components_summary(self, components: List[Dict[str, Any]]) -> str:
        """Build a summary of all components with their paths."""
        summary_parts = []

        for comp in components:
            name = comp.get("name", "Unknown")
            purpose = comp.get("purpose", "")[:200]
            subtree = comp.get("refactored_subtree", {})

            # Get all paths in this component
            paths = get_all_leaf_paths(subtree)
            leaf_count = len(paths)

            summary_parts.append(f"### {name}")
            summary_parts.append(f"**Purpose**: {purpose}")
            summary_parts.append(f"**Leaf Count**: {leaf_count}")
            summary_parts.append("**Paths**:")

            # Show paths (limit to 50 for readability)
            for path in paths[:50]:
                summary_parts.append(f"  - {path}")
            if len(paths) > 50:
                summary_parts.append(f"  - ... and {len(paths) - 50} more paths")

            summary_parts.append("")

        return "\n".join(summary_parts)

    def _generate_edit_plan(
        self,
        components_summary: str,
        user_instructions: str,
        repo_data: Dict[str, Any],
    ) -> Optional[EditPlan]:
        """Generate an edit plan using the LLM."""
        prompt = PROMPT_TEMPLATE_EDIT_PLAN.format(
            edit_instruction=user_instructions,
            repository_name=repo_data.get("repository_name", "Unknown"),
            repository_purpose=repo_data.get("repository_purpose", "")[:500],
            components_summary=components_summary,
        )

        self.logger.info("Generating edit plan...")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")

        _, plan, _ = self.llm.call_structured(
            system_prompt=prompt,
            user_prompt="",
            response_model=EditPlan,
            purpose="generate_plan",
        )

        return plan

    def _display_plan(self, plan: EditPlan):
        """Display the generated edit plan."""
        print("\n" + "─" * 70)
        print("EDIT PLAN")
        print("─" * 70)
        print(f"\nSummary: {plan.summary}")
        print(f"\nOperations ({len(plan.operations)}):")

        for i, op in enumerate(plan.operations, 1):
            print(f"\n  [{i}] {op.component_name} - {op.operation_type}")
            print(f"      Reason: {op.reason}")
            if op.paths_to_remove:
                print(f"      Remove ({len(op.paths_to_remove)}):")
                for path in op.paths_to_remove[:5]:
                    print(f"        [FAIL] {path}")
                if len(op.paths_to_remove) > 5:
                    print(f"        ... and {len(op.paths_to_remove) - 5} more")
            if op.paths_to_add:
                print(f"      Add ({len(op.paths_to_add)}):")
                for path in op.paths_to_add[:5]:
                    print(f"        [OK] {path}")
                if len(op.paths_to_add) > 5:
                    print(f"        ... and {len(op.paths_to_add) - 5} more")

        if plan.validation_notes:
            print(f"\nValidation Notes: {plan.validation_notes}")

        print("\n" + "─" * 70)

    def _execute_plan(
        self, plan: EditPlan, components: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute the edit plan."""
        # Build component lookup
        comp_by_name = {comp.get("name"): comp for comp in components}

        execution_results = []

        for op in plan.operations:
            comp_name = op.component_name
            comp = comp_by_name.get(comp_name)

            if comp is None:
                self.logger.warning(f"[WARNING] Component not found: {comp_name}")
                execution_results.append(
                    {
                        "component": comp_name,
                        "status": "skipped",
                        "reason": "Component not found",
                    }
                )
                continue

            subtree = comp.get("refactored_subtree", {})
            initial_count = count_leaf_nodes(subtree)

            self.logger.info(f"\nProcessing: {comp_name}")

            # Execute DELETE operations
            deleted_count = 0
            failed_deletions = []
            if op.paths_to_remove:
                self.logger.info(f"  Removing {len(op.paths_to_remove)} paths...")
                for path in op.paths_to_remove:
                    _, removal_results = remove_paths(subtree, [path], inplace=True)
                    if removal_results.get(path, False):
                        self.paths_deleted.append(f"{comp_name}: {path}")
                        deleted_count += 1
                        self.logger.info(f"    [OK] Removed: {path}")
                    else:
                        failed_deletions.append(path)
                        self.logger.warning(f"    [WARNING] Path not found: {path}")

            # Execute ADD operations
            added_count = 0
            if op.paths_to_add:
                self.logger.info(f"  Adding {len(op.paths_to_add)} paths...")
                apply_changes(subtree, op.paths_to_add, inplace=True)
                for path in op.paths_to_add:
                    self.paths_added.append(f"{comp_name}: {path}")
                    added_count += 1

            final_count = count_leaf_nodes(subtree)

            # Update component's actual_size
            comp["actual_size"] = final_count

            # Determine actual status
            has_failures = len(failed_deletions) > 0
            if has_failures and deleted_count == 0:
                status = "partial_failure"
            elif has_failures:
                status = "partial_success"
            else:
                status = "success"

            execution_results.append(
                {
                    "component": comp_name,
                    "status": status,
                    "operation_type": op.operation_type,
                    "paths_removed": deleted_count,
                    "paths_requested_remove": len(op.paths_to_remove),
                    "paths_added": added_count,
                    "failed_deletions": failed_deletions,
                    "initial_leaf_count": initial_count,
                    "final_leaf_count": final_count,
                }
            )

            # Build operation record with new optimized structure
            op_record = {
                "component": comp_name,
                "operation_type": op.operation_type,
                "removed": op.paths_to_remove,
                "added": op.paths_to_add,
                "reason": op.reason,
                "status": status,
                "leaf_count": {"before": initial_count, "after": final_count},
            }
            # Only add failed field if there are failures
            if failed_deletions:
                op_record["failed"] = failed_deletions

            self.operations_executed.append(op_record)

            # Log with appropriate status
            if has_failures:
                self.logger.warning(
                    f"  [WARNING] Partial: -{deleted_count}/{len(op.paths_to_remove)} +{added_count} (leaves: {initial_count} → {final_count})"
                )
                self.logger.warning(f"    Failed to remove: {failed_deletions}")
            else:
                self.logger.info(
                    f"  [OK] Done: -{deleted_count} +{added_count} (leaves: {initial_count} → {final_count})"
                )

        return execution_results

    def _review_execution(
        self,
        edit_instruction: str,
        plan: EditPlan,
        execution_results: List[Dict[str, Any]],
        model_analysis: Optional[Dict[str, Any]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ReviewResult]:
        """Review the execution results."""
        # Format plan operations for prompt
        plan_ops_str = ""
        for i, op in enumerate(plan.operations, 1):
            plan_ops_str += f"\n{i}. {op.component_name} ({op.operation_type}):\n"
            plan_ops_str += f"   - Remove: {op.paths_to_remove}\n"
            plan_ops_str += f"   - Add: {op.paths_to_add}\n"
            plan_ops_str += f"   - Reason: {op.reason}\n"

        # Format execution results - include failed deletions
        exec_str = ""
        for er in execution_results:
            exec_str += f"\n- {er['component']}: {er['status']}"
            if er["status"] in ["success", "partial_success", "partial_failure"]:
                requested = er.get("paths_requested_remove", er["paths_removed"])
                exec_str += f" (removed {er['paths_removed']}/{requested}, added {er['paths_added']})"
                exec_str += (
                    f" leaves: {er['initial_leaf_count']} → {er['final_leaf_count']}"
                )
                # Highlight failed deletions
                failed = er.get("failed_deletions", [])
                if failed:
                    exec_str += f"\n  [WARNING] FAILED TO REMOVE: {failed}"

        # Collect affected component names for focused review
        affected_components = set()
        for op in plan.operations:
            affected_components.add(op.component_name)

        # Format state - only show affected components with full paths
        # For others, just show leaf count
        state_before_str = ""
        for name, state in self.state_before.items():
            state_before_str += f"\n### {name}: {state['leaf_count']} leaves"
            if name in affected_components:
                state_before_str += "\nPaths:\n"
                for path in state["paths"]:
                    state_before_str += f"  - {path}\n"

        state_after_str = ""
        for name, state in self.state_after.items():
            state_after_str += f"\n### {name}: {state['leaf_count']} leaves"
            if name in affected_components:
                state_after_str += "\nPaths:\n"
                for path in state["paths"]:
                    state_after_str += f"  - {path}\n"

        # Detect duplicate features across components
        duplicate_features_str = "No duplicates detected."
        if components:
            duplicates = find_duplicate_features(components)
            if duplicates:
                duplicate_features_str = "[WARNING] DUPLICATES FOUND - These features exist in multiple components:\n"
                for feature, locations in duplicates.items():
                    duplicate_features_str += f"\n- '{feature}' found in:\n"
                    for comp_name, path in locations:
                        duplicate_features_str += f"    -  {comp_name}: {path}\n"
                duplicate_features_str += "\nFor MOVE operations, duplicates should be removed from the source component."

        # Format model analysis if available
        model_analysis_str = "No previous model analysis available."
        if model_analysis:
            model_analysis_str = json.dumps(
                model_analysis, indent=2, ensure_ascii=False
            )[:2000]

        prompt = PROMPT_TEMPLATE_EDIT_REVIEW.format(
            edit_instruction=edit_instruction,
            plan_summary=plan.summary,
            plan_operations=plan_ops_str,
            execution_results=exec_str,
            state_before=state_before_str,
            state_after=state_after_str,
            duplicate_features=duplicate_features_str,
            model_analysis=model_analysis_str,
        )

        self.logger.info("Reviewing execution results...")

        _, review, _ = self.llm.call_structured(
            system_prompt=prompt,
            user_prompt="",
            response_model=ReviewResult,
            purpose="review_execution",
        )

        return review

    def _display_review(self, review: ReviewResult, review_round: int = 1):
        """Display the review results."""
        print("\n" + "─" * 70)
        print(f"REVIEW RESULTS (Round {review_round})")
        print("─" * 70)

        # 1. Display thinking process
        if review.thinking:
            print("\n  Analysis:")
            print("  " + "-" * 40)
            # Format thinking with proper indentation
            for line in review.thinking.split("\n"):
                if line.strip():
                    print(f"    {line.strip()}")
            print("  " + "-" * 40)

        # 2. Display summary (most important for user)
        if review.summary:
            print("\n  Summary:")
            print(f"    {review.summary}")

        # 3. Status indicators
        plan_status = "[OK]" if review.execution_matches_plan else "[FAIL]"
        intent_status = "[OK]" if review.execution_matches_intent else "[FAIL]"
        overall_status = (
            "[OK] SUCCESS"
            if review.overall_success
            else "[FAIL] NEEDS FIX"
            if review.needs_fix
            else "[FAIL] FAILED"
        )

        print("\n  Verification:")
        print(f"    Execution matches plan:   {plan_status}")
        print(f"    Execution matches intent: {intent_status}")
        print(f"    Confidence score:         {review.confidence_score:.2f}")
        print(f"\n  [TARGET] Overall: {overall_status}")

        # 4. Issues found
        if review.issues_found:
            print(f"\n  [WARNING]  Issues Found ({len(review.issues_found)}):")
            for issue in review.issues_found:
                print(f"      -  {issue}")

        # 5. Suggestions
        if review.suggestions:
            print(f"\n  Suggestions ({len(review.suggestions)}):")
            for suggestion in review.suggestions:
                print(f"      → {suggestion}")

        # 6. Fix operations (if any)
        if review.needs_fix and review.fix_operations:
            print(f"\n  Fix Operations Required ({len(review.fix_operations)}):")
            for i, op in enumerate(review.fix_operations, 1):
                comp_name = op.get("component_name", "Unknown")
                op_type = op.get("operation_type", "FIX")
                reason = op.get("reason", "")
                paths_remove = op.get("paths_to_remove", [])
                paths_add = op.get("paths_to_add", [])

                print(f"      [{i}] {comp_name} - {op_type}")
                print(f"          Reason: {reason}")
                if paths_remove:
                    print(
                        f"          Remove: {paths_remove[:3]}{'...' if len(paths_remove) > 3 else ''}"
                    )
                if paths_add:
                    print(
                        f"          Add: {paths_add[:3]}{'...' if len(paths_add) > 3 else ''}"
                    )

        print("\n" + "─" * 70)

    def _print_summary(self, result: Dict[str, Any], components: List[Dict[str, Any]]):
        """Print execution summary."""
        print("\n")
        print("=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)

        # 1. Operation Statistics
        exec_results = result.get("execution_results", [])
        if exec_results:
            rows = []
            total_removed = 0
            total_added = 0
            for er in exec_results:
                if er["status"] == "success":
                    rows.append(
                        [
                            er["component"][:30],
                            er["operation_type"],
                            er["paths_removed"],
                            er["paths_added"],
                            f"{er['initial_leaf_count']} → {er['final_leaf_count']}",
                        ]
                    )
                    total_removed += er["paths_removed"]
                    total_added += er["paths_added"]

            if rows:
                rows.append(["TOTAL", "", total_removed, total_added, ""])
                print_unicode_table(
                    headers=["Component", "Operation", "Removed", "Added", "Leaves"],
                    rows=rows,
                    title="Operations Executed",
                )

        # 2. Deleted Paths
        if self.paths_deleted:
            print(f"\n   Deleted Paths ({len(self.paths_deleted)}):")
            for path in self.paths_deleted[:10]:
                print(f"   [FAIL] {path}")
            if len(self.paths_deleted) > 10:
                print(f"   ... and {len(self.paths_deleted) - 10} more")

        # 3. Added Paths
        if self.paths_added:
            print(f"\n   Added Paths ({len(self.paths_added)}):")
            for path in self.paths_added[:10]:
                print(f"   [OK] {path}")
            if len(self.paths_added) > 10:
                print(f"   ... and {len(self.paths_added) - 10} more")

        # 4. Final Component Stats
        print("\n   Final Component Statistics:")
        for comp in components:
            name = comp.get("name", "Unknown")
            leaf_count = count_leaf_nodes(comp.get("refactored_subtree", {}))
            print(f"   -  {name}: {leaf_count} leaves")

        print("\n" + "=" * 80)


# ============================================================================
# Main Program
# ============================================================================


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Edit Feature Tree - Three-phase approach (Plan + Execute + Review)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python feature_edit.py
  python feature_edit.py --log-level DEBUG
  python feature_edit.py --no-review  # Skip review phase (not recommended)
        """,
    )

    parser.add_argument(
        "--file",
        help="Feature tree JSON file (input and output)",
        default=str(FEATURE_TREE_FILE),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the review phase (not recommended)",
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
        # Load feature_tree.json
        logger.info(f"Reading feature tree file: {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)

        components = data.get("components")
        if not components:
            logger.error("[FAIL] File missing 'components' field")
            return 1

        # Extract data
        repository_name = data.get("repository_name", "Unknown")
        repository_purpose = data.get("repository_purpose", "")
        edit_instruction = data.get("edit_instruction", "")
        edit_history = data.get("edit_history", [])

        if not edit_instruction:
            logger.error("[FAIL] No user edit instructions found")
            return 1

        logger.info(f"[OK] Loaded {len(components)} components")
        logger.info(f"[OK] User instructions: {edit_instruction[:100]}...")

        repo_data = {
            "repository_name": repository_name,
            "repository_purpose": repository_purpose,
        }

        # Extract model analysis from previous build/refactor if available
        model_analysis = data.get("model_analysis", None)
        if not model_analysis:
            # Try to extract from components metadata
            model_analysis = {
                "source": "feature_tree.json",
                "components_count": len(components),
                "total_features": sum(
                    comp.get("actual_size", 0) for comp in components
                ),
            }

        # Initialize trajectory
        trajectory = None
        step_id = None
        if not args.no_trajectory:
            trajectory = load_or_create_trajectory("feature_edit")
            trajectory.start(metadata={
                "input_file": args.file,
                "edit_instruction": edit_instruction[:200],
                "components_count": len(components),
            })
            step = trajectory.add_step("feature_edit", "Edit feature tree based on user instructions")
            trajectory.start_step(step.step_id)
            step_id = step.step_id

        # Create LLM client and editor
        llm_client = LLMClient(trajectory=trajectory, step_id=step_id)
        editor = FeatureTreeEditor(
            llm_client,
            enable_review=not args.no_review,
        )

        # Execute three-phase editing
        result = editor.edit(
            components,
            edit_instruction,
            repo_data,
            model_analysis=model_analysis,
        )

        if not result.get("success"):
            logger.error(f"[FAIL] Edit failed: {result.get('error', 'Unknown error')}")
            return 1

        # Build result summary
        review_data = result.get("review", {})
        result_summary = {
            "status": review_data.get(
                "final_status", "SUCCESS" if result.get("success") else "FAILED"
            ),
            "total_removed": len(result.get("paths_deleted", [])),
            "total_added": len(result.get("paths_added", [])),
            "review": {
                "summary": review_data.get("summary", ""),
                "confidence": review_data.get("confidence_score", 0.0),
                "iterations": review_data.get("total_rounds", 1),
            }
            if review_data
            else None,
        }

        # Update edit history with new optimized structure
        edit_record = {
            "instruction": edit_instruction,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "plan_summary": result.get("plan_summary", ""),
            "operations": result.get("operations_executed", []),
            "result": result_summary,
        }

        # Ensure history is a list of dicts
        if not isinstance(edit_history, list):
            edit_history = []

        edit_history.append(edit_record)
        data["edit_history"] = edit_history

        # Remove last_edit_result if exists (no longer needed)
        if "last_edit_result" in data:
            del data["last_edit_result"]

        # Check if review flagged issues
        if review_data and not review_data.get("overall_success", True):
            logger.warning("[WARNING] Review identified issues with the edit operation")
            if review_data.get("issues_found"):
                for issue in review_data["issues_found"]:
                    logger.warning(f"  - {issue}")

        # Reorder output: put edit_instruction before edit_history
        # Build ordered output dict
        ordered_data = {}
        # First, add all keys except edit_instruction and edit_history
        for key in data:
            if key not in ("edit_instruction", "edit_history"):
                ordered_data[key] = data[key]
        # Then add edit_instruction and edit_history in desired order
        ordered_data["edit_instruction"] = data.get("edit_instruction", "")
        ordered_data["edit_history"] = data.get("edit_history", [])

        # Save results back to the same file
        logger.info(f"\nSaving results to: {args.file}")
        with open(args.file, "w", encoding="utf-8") as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'=' * 70}")
        logger.info("[OK] Edit complete!")
        logger.info(f"{'=' * 70}\n")

        # Mark trajectory as complete
        if trajectory:
            if step_id is not None:
                trajectory.complete_step(step_id, {
                    "total_removed": len(result.get("paths_deleted", [])),
                    "total_added": len(result.get("paths_added", [])),
                })
            trajectory.complete(metadata={
                "total_removed": len(result.get("paths_deleted", [])),
                "total_added": len(result.get("paths_added", [])),
            })
            logger.info(f"[OK] Trajectory saved to: {trajectory.trajectory_file}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"[FAIL] File not found: {e}")
        if trajectory:
            if step_id is not None:
                trajectory.fail_step(step_id, str(e))
            trajectory.fail(str(e))
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
