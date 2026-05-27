#!/usr/bin/env python3
"""Global Interface Review Module.

Implements the Global Review phase for interface design, including:
- Entry point identification via LLM semantic reasoning
- Wiring completeness / call graph connectivity checks
- Cross-module type compatibility validation
- Automatic fix suggestions and application

This module is invoked AFTER all per-subtree interface designs are complete,
but BEFORE the final interfaces.json is saved.
"""

import json
import logging
import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import LLMClient

from .interface_agent import (
    GlobalInterfaceRegistry,
    DependencyCollector,
    cross_validate_imports_vs_calls,
)
from .interface_prompts import ORPHAN_REVIEW_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# Global Review Prompt
# ============================================================================

GLOBAL_INTERFACE_REVIEW_PROMPT = """
You are a senior software engineer reviewing the COMPLETE set of interfaces for an entire Python repository.

All subtrees have been designed. Your task is to review the interfaces holistically,
focusing on CROSS-MODULE integration — not individual interface quality.

## Input
- All designed interfaces (grouped by subtree)
- Data flow DAG (subtree-level dependencies)
- Import cross-validation warnings (symbols imported but not declared as calls)

## Review Tasks

### Task 1: Identify Top-Level Interfaces
Identify which units (classes/functions) are **top-level interfaces** — those that
are not expected to be called by other internal modules within this repository.

Top-level interfaces are NOT limited to "files named main.py". They are units whose
role in the architecture means they don't need an internal caller. This includes:
- Application entry points: a `MainLoop` class, a CLI `main()` function, an `Application` class
- Standalone submodules: components that can function independently (e.g., a `TestRunner`, a `Benchmark` harness)
- Externally-invoked APIs: interfaces designed to be called by external code, plugins, or frameworks
- Framework callbacks: handlers registered with an event system or framework

Use semantic judgment based on the module's role and the project's architecture.

### Task 2: Wiring Completeness
- Does every non-top-level module's output have at least one consumer?
- Are there "island" modules that neither call nor are called by anyone?
- Do the identified top-level interfaces actually invoke the key subsystems?
- Are there missing orchestration layers?

### Task 3: Call Chain Realism
- Can you trace a realistic execution path from each top-level interface to leaf modules?
- Are the parameter/return types compatible across call boundaries?

### Task 4: Dependency Direction Consistency
- Do dependencies flow in the direction specified by the data_flow DAG?
- Are there undeclared reverse dependencies?

## Output
You must return ONLY a valid JSON object with the following structure (no other text):
{
  "entry_points": [
    {
      "file_path": "...",
      "unit_name": "...",
      "rationale": "..."
    }
  ],
  "orphan_modules": [
    {
      "file_path": "...",
      "unit_name": "...",
      "reason": "..."
    }
  ],
  "missing_wiring": [
    {
      "from_unit": "...",
      "from_file": "...",
      "to_unit": "...",
      "to_file": "...",
      "description": "..."
    }
  ],
  "type_mismatches": [
    {
      "file_path": "...",
      "unit_name": "...",
      "description": "..."
    }
  ],
  "orchestration_gaps": [
    {
      "description": "...",
      "suggested_location": "..."
    }
  ],
  "recommended_fixes": [
    {
      "action": "add_dependency",
      "file_path": "...",
      "unit_name": "...",
      "description": "...",
      "calls_to_add": [
        {"callee": "...", "callee_file": "...", "purpose": "..."}
      ]
    }
  ],
  "pass": true
}

Important:
- "pass" should be true only if there are no orphan modules, no missing wiring,
  and no orchestration gaps.
- recommended_fixes should contain concrete, actionable fixes.
- Each fix action must be one of: "add_dependency", "add_interface", "modify_interface"
- For "add_dependency" fixes, include "calls_to_add" with callee name and file.
""".strip()


# ============================================================================
# Code-Based Structural Checks
# ============================================================================

def build_call_graph(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    """Build a directed call graph from interfaces and enhanced_data_flow.
    
    Returns:
        - outgoing: {unit_key -> set of callee unit_keys}
        - incoming: {unit_key -> set of caller unit_keys}
        - unit_to_file: {unit_key -> file_path}
    """
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    unit_to_file = {}
    
    # Collect all units
    subtrees = interfaces_data.get("subtrees", {})
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            for unit_name in file_data.get("units", []):
                unit_key = f"{file_path}::{unit_name}"
                unit_to_file[unit_key] = file_path
    
    # Build a lookup: bare_name -> unit_key(s)
    name_to_keys = defaultdict(list)
    for unit_key in unit_to_file:
        parts = unit_key.split("::", 1)
        if len(parts) == 2:
            unit_name = parts[1]
            # e.g., "class Foo" -> bare name "Foo"
            name_parts = unit_name.split(" ", 1)
            bare_name = name_parts[1] if len(name_parts) == 2 else unit_name
            name_to_keys[bare_name].append(unit_key)
            name_to_keys[unit_name].append(unit_key)
    
    # Process invocation edges from enhanced_data_flow
    for edge in enhanced_data_flow.get("invocation_edges", []):
        caller = edge.get("caller", "")
        callee = edge.get("callee", "")
        caller_file = edge.get("caller_file", "")
        callee_file = edge.get("callee_file", "")
        
        # Resolve caller key
        caller_key = f"{caller_file}::{caller}" if caller_file else None
        if caller_key and caller_key not in unit_to_file:
            # Try to find by name
            candidates = name_to_keys.get(caller, [])
            if candidates:
                caller_key = candidates[0]
            else:
                caller_key = None
        
        # Resolve callee key
        callee_key = None
        if callee_file:
            callee_key = f"{callee_file}::{callee}"
            if callee_key not in unit_to_file:
                # Try matching just by callee name
                for key in name_to_keys.get(callee, []):
                    if unit_to_file.get(key) == callee_file:
                        callee_key = key
                        break
                else:
                    candidates = name_to_keys.get(callee, [])
                    callee_key = candidates[0] if candidates else None
        else:
            candidates = name_to_keys.get(callee, [])
            callee_key = candidates[0] if candidates else None
        
        if caller_key and callee_key:
            outgoing[caller_key].add(callee_key)
            incoming[callee_key].add(caller_key)
    
    # Process inheritance edges
    for edge in enhanced_data_flow.get("inheritance_edges", []):
        child = edge.get("child", "")
        parent = edge.get("parent", "")
        source_file = edge.get("source_file", "")
        parent_file = edge.get("parent_file", "")
        
        child_candidates = name_to_keys.get(child, [])
        parent_candidates = name_to_keys.get(parent, [])
        
        if child_candidates and parent_candidates:
            child_key = child_candidates[0]
            parent_key = parent_candidates[0]
            outgoing[child_key].add(parent_key)
            incoming[parent_key].add(child_key)
    
    # Process reference edges
    for edge in enhanced_data_flow.get("reference_edges", []):
        unit = edge.get("unit", "")
        ref_type = edge.get("referenced_type", "")
        source_file = edge.get("source_file", "")
        
        unit_candidates = name_to_keys.get(unit, [])
        type_candidates = name_to_keys.get(ref_type, [])
        
        if unit_candidates and type_candidates:
            unit_key = unit_candidates[0]
            type_key = type_candidates[0]
            outgoing[unit_key].add(type_key)
            incoming[type_key].add(unit_key)
    
    return dict(outgoing), dict(incoming), unit_to_file


def check_call_graph_connectivity(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    entry_points: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a directed graph of all invocation edges and check connectivity.

    Identifies orphan units (non-entry-point units with no incoming edges).

    Returns:
        Dict with keys: orphan_units, total_units, entry_point_count
    """
    outgoing, incoming, unit_to_file = build_call_graph(interfaces_data, enhanced_data_flow)

    all_units = set(unit_to_file.keys())

    # Build entry point key set
    entry_point_keys = set()
    for ep in entry_points:
        ep_file = ep.get("file_path", "")
        ep_unit = ep.get("unit_name", "")
        ep_key = f"{ep_file}::{ep_unit}"
        if ep_key in all_units:
            entry_point_keys.add(ep_key)
        else:
            # Try fuzzy match
            for uk in all_units:
                if uk.endswith(f"::{ep_unit}"):
                    entry_point_keys.add(uk)
                    break

    non_entry_units = all_units - entry_point_keys

    # Units with no incoming edges (excluding entry points)
    orphan_units = []
    for unit_key in non_entry_units:
        if unit_key not in incoming or len(incoming[unit_key]) == 0:
            orphan_units.append({
                "unit_key": unit_key,
                "file_path": unit_to_file.get(unit_key, ""),
            })

    return {
        "orphan_units": orphan_units,
        "total_units": len(all_units),
        "entry_point_count": len(entry_point_keys),
    }


def check_feature_dependency_coverage(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    entry_points: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check that every feature-bearing unit is either an entry point or has at least one incoming dependency edge.
    
    Returns: list of orphan features (feature paths without incoming edges
             and not in entry points)
    """
    _, incoming, unit_to_file = build_call_graph(interfaces_data, enhanced_data_flow)
    
    # Build entry point key set
    entry_point_keys = set()
    for ep in entry_points:
        ep_file = ep.get("file_path", "")
        ep_unit = ep.get("unit_name", "")
        ep_key = f"{ep_file}::{ep_unit}"
        entry_point_keys.add(ep_key)
        # Also add bare match
        for uk in unit_to_file:
            if uk.endswith(f"::{ep_unit}"):
                entry_point_keys.add(uk)
    
    orphan_features = []
    subtrees = interfaces_data.get("subtrees", {})
    
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            units_to_features = file_data.get("units_to_features", {})
            for unit_name, features in units_to_features.items():
                unit_key = f"{file_path}::{unit_name}"
                
                # Skip entry points
                if unit_key in entry_point_keys:
                    continue
                
                # Check if has any incoming edge
                if unit_key not in incoming or len(incoming[unit_key]) == 0:
                    orphan_features.append({
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "features": features,
                        "subtree": subtree_name,
                    })
    
    return orphan_features


# ============================================================================
# Interface Reviewer
# ============================================================================

class InterfaceReviewer:
    """Global interface reviewer that performs holistic review after all subtrees are designed.
    
    Combines:
    1. LLM-based semantic review (entry point identification, wiring, consistency)
    2. Code-based structural checks (call graph connectivity, feature coverage)
    3. Automatic fix application (add missing dependencies / interfaces)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None,
    ):
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
        self.logger = logging.getLogger(__name__)
    
    def review_and_fix(
        self,
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        import_warnings: List[Dict[str, str]],
        data_flow_edges: List[Dict[str, Any]],
        dependency_collector: Optional[DependencyCollector] = None,
        max_fix_iterations: int = 2,
    ) -> Dict[str, Any]:
        """Run the full global review and fix cycle.
        
        Steps:
        1. LLM global review (entry point identification + wiring + consistency)
        2. Code-based checks using LLM-identified entry points
        3. If issues found: apply recommended_fixes from LLM
        4. Re-run code checks
        5. Repeat until pass or max iterations
        
        Args:
            interfaces_data: The full interfaces result dict
            enhanced_data_flow: The enhanced_data_flow dict from DependencyCollector
            global_registry: The GlobalInterfaceRegistry with all designed interfaces
            import_warnings: List of import cross-validation warnings
            data_flow_edges: Original data flow DAG edges
            dependency_collector: DependencyCollector for adding new edges
            max_fix_iterations: Maximum number of review-fix cycles
            
        Returns:
            Dict with review results, applied fixes, and updated interfaces_data
        """
        self.logger.info("[InterfaceReviewer] Starting global interface review")
        
        review_history = []
        
        for iteration in range(max_fix_iterations):
            self.logger.info(f"[InterfaceReviewer] Review iteration {iteration + 1}/{max_fix_iterations}")
            
            # Step 1: LLM global review
            llm_review = self._run_llm_review(
                interfaces_data=interfaces_data,
                enhanced_data_flow=enhanced_data_flow,
                global_registry=global_registry,
                import_warnings=import_warnings,
                data_flow_edges=data_flow_edges,
                iteration=iteration,
                previous_reviews=review_history,
            )
            
            if not llm_review:
                self.logger.warning("[InterfaceReviewer] LLM review returned empty result")
                break
            
            entry_points = llm_review.get("entry_points", [])
            self.logger.info(
                f"[InterfaceReviewer] LLM identified {len(entry_points)} entry points"
            )
            for ep in entry_points:
                self.logger.info(
                    f"  Entry point: {ep.get('unit_name', '?')} in {ep.get('file_path', '?')} "
                    f"— {ep.get('rationale', '')}"
                )
            
            # Step 2: Code-based structural checks
            connectivity = check_call_graph_connectivity(
                interfaces_data, enhanced_data_flow, entry_points
            )
            feature_orphans = check_feature_dependency_coverage(
                interfaces_data, enhanced_data_flow, entry_points
            )

            self.logger.info(
                f"[InterfaceReviewer] Connectivity: "
                f"{connectivity['total_units']} total units, "
                f"{connectivity['entry_point_count']} entry points, "
                f"{len(connectivity['orphan_units'])} orphan units"
            )
            self.logger.info(
                f"[InterfaceReviewer] Feature coverage: {len(feature_orphans)} orphan features"
            )

            review_result = {
                "iteration": iteration + 1,
                "llm_review": llm_review,
                "orphan_units": connectivity["orphan_units"],
                "feature_orphans": feature_orphans,
                "entry_points": entry_points,
            }
            review_history.append(review_result)
            
            # Step 3: Check if passed
            llm_passed = llm_review.get("pass", False)
            code_passed = (
                len(connectivity["orphan_units"]) == 0
                and len(feature_orphans) == 0
            )
            
            if llm_passed and code_passed:
                self.logger.info("[InterfaceReviewer] [OK] Global review PASSED")
                break
            
            # Step 4: Apply fixes
            recommended_fixes = llm_review.get("recommended_fixes", [])
            if recommended_fixes:
                applied_count = self._apply_fixes(
                    fixes=recommended_fixes,
                    interfaces_data=interfaces_data,
                    enhanced_data_flow=enhanced_data_flow,
                    global_registry=global_registry,
                    dependency_collector=dependency_collector,
                )
                self.logger.info(
                    f"[InterfaceReviewer] Applied {applied_count}/{len(recommended_fixes)} fixes"
                )
            else:
                self.logger.info("[InterfaceReviewer] No fixes recommended, stopping iteration")
                break
        
        # Compile final summary
        final_result = {
            "review_history": review_history,
            "final_entry_points": review_history[-1]["entry_points"] if review_history else [],
            "final_feature_orphans": review_history[-1]["feature_orphans"] if review_history else [],
            "iterations_run": len(review_history),
            "passed": (
                review_history[-1]["llm_review"].get("pass", False)
                if review_history else False
            ),
        }

        return final_result
    
    def _run_llm_review(
        self,
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        import_warnings: List[Dict[str, str]],
        data_flow_edges: List[Dict[str, Any]],
        iteration: int = 0,
        previous_reviews: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the LLM global review.
        
        Builds a comprehensive prompt with all designed interfaces and asks
        LLM to identify entry points, orphan modules, missing wiring, etc.
        """
        # Build the interface summary for the prompt
        interface_summary = self._build_interface_summary(interfaces_data, global_registry)
        
        # Build data flow summary
        data_flow_summary = self._build_data_flow_summary(data_flow_edges)
        
        # Build import warnings summary
        import_warnings_summary = self._build_import_warnings_summary(import_warnings)
        
        # Build dependency summary
        dep_summary = self._build_dependency_summary(enhanced_data_flow)
        
        # Build previous review context (for iteration > 0)
        prev_context = ""
        if previous_reviews:
            last_review = previous_reviews[-1]
            prev_llm = last_review.get("llm_review", {})
            prev_orphan_units = last_review.get("orphan_units", [])
            prev_orphan_count = len(last_review.get("feature_orphans", []))

            prev_context = f"""
## Previous Review Results (iteration {last_review.get('iteration', '?')})
- Entry points identified: {len(prev_llm.get('entry_points', []))}
- Orphan modules from LLM: {len(prev_llm.get('orphan_modules', []))}
- Orphan units (no incoming edges): {len(prev_orphan_units)}
- Orphan features: {prev_orphan_count}
- Fixes applied: {len(prev_llm.get('recommended_fixes', []))}

Please review the CURRENT state after fixes were applied and provide updated analysis.
"""
        
        user_prompt = f"""
## All Designed Interfaces (grouped by subtree)
{interface_summary}

## Data Flow DAG
{data_flow_summary}

## Current Dependency Edges
{dep_summary}

## Import Cross-Validation Warnings
{import_warnings_summary}
{prev_context}

Please perform the review tasks and return the JSON result.
""".strip()
        
        combined_prompt = f"{GLOBAL_INTERFACE_REVIEW_PROMPT}\n\n{user_prompt}"
        
        try:
            response = self.llm.generate(
                combined_prompt,
                purpose=f"global_interface_review_{iteration + 1}"
            )
            
            # Parse JSON from response
            result = self.llm.parse_json_block(response)
            
            if result:
                return result
            
            # Try to extract JSON directly
            try:
                # Find JSON in the response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(response[start:end])
                    return result
            except json.JSONDecodeError:
                pass
            
            self.logger.warning("[InterfaceReviewer] Failed to parse LLM review response")
            return None
            
        except Exception as e:
            self.logger.error(f"[InterfaceReviewer] LLM review failed: {e}")
            return None
    
    def _apply_fixes(
        self,
        fixes: List[Dict[str, Any]],
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        dependency_collector: Optional[DependencyCollector] = None,
    ) -> int:
        """Apply recommended fixes from the LLM review.
        
        Supported actions:
        - add_dependency: Add a call dependency edge
        - add_interface: (logged as warning — requires manual or future LLM action)
        - modify_interface: (logged as warning — requires manual or future LLM action)
        
        Returns:
            Number of fixes successfully applied
        """
        applied = 0
        
        for fix in fixes:
            action = fix.get("action", "")
            file_path = fix.get("file_path", "")
            unit_name = fix.get("unit_name", "")
            description = fix.get("description", "")
            
            if action == "add_dependency":
                calls_to_add = fix.get("calls_to_add", [])
                for call_info in calls_to_add:
                    callee = call_info.get("callee", "")
                    callee_file = call_info.get("callee_file", "")
                    
                    if not callee:
                        continue
                    
                    # Resolve callee_file from global registry if not provided
                    if not callee_file:
                        callee_file = global_registry.resolve_callee(callee)
                    
                    if not callee_file:
                        self.logger.warning(
                            f"[InterfaceReviewer] Cannot resolve callee '{callee}' "
                            f"for fix on {file_path}::{unit_name}"
                        )
                        continue
                    
                    # Add to enhanced_data_flow
                    inv_edges = enhanced_data_flow.get("invocation_edges", [])
                    
                    # Check if edge already exists
                    exists = any(
                        e.get("caller") == unit_name
                        and e.get("callee") == callee
                        and e.get("caller_file") == file_path
                        for e in inv_edges
                    )
                    
                    if not exists:
                        new_edge = {
                            "caller": unit_name,
                            "callee": callee,
                            "caller_file": file_path,
                            "callee_file": callee_file,
                            "edge_type": "invokes",
                            "generator": "global_review",
                        }
                        inv_edges.append(new_edge)
                        enhanced_data_flow["invocation_edges"] = inv_edges
                        
                        # Also add to dependency_collector if available
                        if dependency_collector:
                            dependency_collector.add_invocation(
                                caller=unit_name,
                                callee=callee,
                                caller_file=file_path,
                                callee_file=callee_file,
                            )
                        
                        self.logger.info(
                            f"[InterfaceReviewer] Added dependency: "
                            f"{unit_name} ({file_path}) -> {callee} ({callee_file})"
                        )
                        applied += 1
            
            elif action == "add_interface":
                self.logger.warning(
                    f"[InterfaceReviewer] add_interface fix requested but not auto-applied: "
                    f"{description} (file: {file_path})"
                )
            
            elif action == "modify_interface":
                self.logger.warning(
                    f"[InterfaceReviewer] modify_interface fix requested but not auto-applied: "
                    f"{description} (file: {file_path}, unit: {unit_name})"
                )
            
            else:
                self.logger.warning(
                    f"[InterfaceReviewer] Unknown fix action: {action}"
                )
        
        return applied
    
    def _build_interface_summary(
        self,
        interfaces_data: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
    ) -> str:
        """Build a comprehensive interface summary for the LLM review prompt."""
        parts = []
        subtrees = interfaces_data.get("subtrees", {})
        subtree_order = interfaces_data.get("subtree_order", [])
        
        for subtree_name in subtree_order:
            subtree_data = subtrees.get(subtree_name, {})
            file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
            
            if not file_interfaces:
                continue
            
            parts.append(f"\n### Subtree: {subtree_name}")
            
            for file_path, file_data in file_interfaces.items():
                units = file_data.get("units", [])
                units_to_features = file_data.get("units_to_features", {})
                file_code = file_data.get("file_code", "")
                
                if not units:
                    continue
                
                parts.append(f"\n**{file_path}**")
                
                for unit_name in units:
                    features = units_to_features.get(unit_name, [])
                    features_str = ", ".join(features[:5])
                    if len(features) > 5:
                        features_str += f" (+{len(features) - 5} more)"
                    parts.append(f"  - `{unit_name}` → features: {features_str}")
                
                # Include abbreviated code (first 20 lines)
                if file_code:
                    code_lines = file_code.split("\n")
                    if len(code_lines) > 25:
                        code_preview = "\n".join(code_lines[:25]) + "\n    # ... (truncated)"
                    else:
                        code_preview = file_code
                    parts.append(f"  ```python\n{code_preview}\n  ```")
        
        return "\n".join(parts) if parts else "No interfaces designed."
    
    def _build_data_flow_summary(self, data_flow_edges: List[Dict[str, Any]]) -> str:
        """Build a data flow summary for the prompt."""
        if not data_flow_edges:
            return "No data flow edges."
        
        parts = []
        for edge in data_flow_edges:
            source = edge.get("source", "?")
            target = edge.get("target", "?")
            desc = edge.get("description", "")
            parts.append(f"  {source} → {target}" + (f": {desc}" if desc else ""))
        
        return "\n".join(parts)
    
    def _build_import_warnings_summary(self, warnings: List[Dict[str, str]]) -> str:
        """Build import warnings summary for the prompt."""
        if not warnings:
            return "No import cross-validation warnings."
        
        parts = [f"Found {len(warnings)} potential issues:"]
        for w in warnings[:20]:  # Limit to 20
            parts.append(f"  - {w.get('message', '?')}")
        
        if len(warnings) > 20:
            parts.append(f"  ... and {len(warnings) - 20} more warnings")
        
        return "\n".join(parts)
    
    def _build_dependency_summary(self, enhanced_data_flow: Dict[str, Any]) -> str:
        """Build a dependency edge summary for the prompt."""
        if not enhanced_data_flow:
            return "No dependency edges collected."
        
        parts = []
        
        inv_edges = enhanced_data_flow.get("invocation_edges", [])
        inh_edges = enhanced_data_flow.get("inheritance_edges", [])
        ref_edges = enhanced_data_flow.get("reference_edges", [])
        
        parts.append(
            f"Total: {len(inv_edges)} invocation, {len(inh_edges)} inheritance, "
            f"{len(ref_edges)} reference edges"
        )
        
        # Cross-file invocations
        cross_file = [e for e in inv_edges if e.get("caller_file") != e.get("callee_file")]
        same_file = [e for e in inv_edges if e.get("caller_file") == e.get("callee_file")]
        no_callee = [e for e in inv_edges if not e.get("callee_file")]
        
        parts.append(
            f"Invocations: {len(cross_file)} cross-file, {len(same_file)} same-file, "
            f"{len(no_callee)} unresolved callee"
        )
        
        # Show cross-file edges
        if cross_file:
            parts.append("\nCross-file invocations:")
            for e in cross_file[:30]:
                parts.append(
                    f"  {e.get('caller', '?')} ({e.get('caller_file', '?')}) "
                    f"→ {e.get('callee', '?')} ({e.get('callee_file', '?')})"
                )
            if len(cross_file) > 30:
                parts.append(f"  ... and {len(cross_file) - 30} more")
        
        return "\n".join(parts)


# ============================================================================
# Orphan Pruning
# ============================================================================

def prune_orphan_interfaces(
    interfaces_data: Dict[str, Any],
    review_result: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Remove orphan interfaces from interfaces_data after global review.

    An interface unit is considered a **true orphan** when it has **no incoming
    edges AND no outgoing edges** in the call graph and is not an entry point.
    Units that participate in any edge (caller or callee) are preserved even if
    they are not reachable from entry points — their connected components are
    valid code that just lacks proper wiring to the top-level entry flow.

    For each pruned unit the function:
    - Removes it from ``units``, ``units_to_features``, ``units_to_code``
    - Regenerates ``file_code`` from the remaining units
    - If all units in a file are removed, removes the entire file entry
    - Removes related edges from ``enhanced_data_flow``

    Returns a summary dict::

        {
            "pruned_units": [...],
            "pruned_files": [...],
            "orphan_feature_paths": set of feature paths whose ALL implementing
                                    units were pruned,
            "surviving_feature_paths": set of feature paths that still have at
                                       least one surviving unit,
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ---- 0. Build call graph to find truly isolated units ----
    entry_points = review_result.get("final_entry_points", [])
    entry_point_keys: Set[str] = set()
    for ep in entry_points:
        ep_file = ep.get("file_path", "")
        ep_unit = ep.get("unit_name", "")
        if ep_file and ep_unit:
            entry_point_keys.add(f"{ep_file}::{ep_unit}")

    outgoing, incoming, unit_to_file = build_call_graph(
        interfaces_data, enhanced_data_flow
    )
    all_units = set(unit_to_file.keys())

    # Truly isolated: no incoming AND no outgoing AND not entry point
    isolated_keys: Set[str] = set()
    for u in all_units:
        if u in entry_point_keys:
            continue
        has_in = u in incoming and len(incoming[u]) > 0
        has_out = u in outgoing and len(outgoing[u]) > 0
        if not has_in and not has_out:
            isolated_keys.add(u)

    if not isolated_keys:
        logger.info("[prune_orphan_interfaces] No truly isolated units — nothing to prune")
        # Still compute surviving features for RPG pruning
        surviving = _collect_surviving_features(interfaces_data)
        return {
            "pruned_units": [],
            "pruned_files": [],
            "orphan_feature_paths": set(),
            "surviving_feature_paths": surviving,
        }

    logger.info(
        f"[prune_orphan_interfaces] {len(isolated_keys)} truly isolated units "
        f"(out of {len(all_units)} total) to prune"
    )

    pruned_units: List[Dict[str, Any]] = []
    pruned_files: List[Dict[str, str]] = []

    # ---- 1. Build a global map: feature_path → set of unit_keys that implement it ----
    feature_to_all_unit_keys: Dict[str, Set[str]] = defaultdict(set)
    subtrees = interfaces_data.get("subtrees", {})
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            for unit_name, features in file_data.get("units_to_features", {}).items():
                unit_key = f"{file_path}::{unit_name}"
                for fp in features:
                    feature_to_all_unit_keys[fp].add(unit_key)

    # ---- 2. Prune units from interfaces_data ----
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        files_to_remove: List[str] = []

        for file_path in list(file_interfaces.keys()):
            file_data = file_interfaces[file_path]
            units: List[str] = file_data.get("units", [])
            units_to_features: Dict[str, List[str]] = file_data.get("units_to_features", {})
            units_to_code: Dict[str, str] = file_data.get("units_to_code", {})

            units_to_remove: List[str] = []
            for unit_name in units:
                unit_key = f"{file_path}::{unit_name}"
                if unit_key in isolated_keys:
                    units_to_remove.append(unit_name)
                    pruned_units.append({
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "subtree": subtree_name,
                        "features": units_to_features.get(unit_name, []),
                    })

            if not units_to_remove:
                continue

            # Remove the units
            for uname in units_to_remove:
                if uname in units:
                    units.remove(uname)
                units_to_features.pop(uname, None)
                units_to_code.pop(uname, None)
                logger.info(f"[prune_orphan_interfaces] Pruned unit: {file_path}::{uname}")

            file_data["units"] = units
            file_data["units_to_features"] = units_to_features
            file_data["units_to_code"] = units_to_code

            if not units:
                # All units pruned → remove the entire file entry
                files_to_remove.append(file_path)
            else:
                # Regenerate file_code from surviving units
                code_parts = []
                for uname in units:
                    code = units_to_code.get(uname, "")
                    if code:
                        code_parts.append(code)
                file_data["file_code"] = "\n\n".join(code_parts)

        for fp in files_to_remove:
            del file_interfaces[fp]
            pruned_files.append({"file_path": fp, "subtree": subtree_name})
            logger.info(f"[prune_orphan_interfaces] Pruned entire file: {fp} (all units removed)")

    # ---- 3. Remove edges for pruned units from enhanced_data_flow ----
    pruned_unit_names = {pu["unit_name"] for pu in pruned_units}
    pruned_file_paths = {pu["file_path"] for pu in pruned_units}

    def _edge_involves_pruned_unit(edge: Dict[str, Any]) -> bool:
        """Return True if the edge references a pruned unit."""
        for role_name, role_file in [
            ("caller", "caller_file"), ("callee", "callee_file"),
            ("child", "child_file"), ("parent", "parent_file"),
            ("unit", "unit_file"),
        ]:
            name_val = edge.get(role_name, "")
            file_val = edge.get(role_file, "")
            if name_val in pruned_unit_names:
                # Double-check file to avoid false positives on common names
                if not file_val or file_val in pruned_file_paths:
                    return True
        return False

    for edge_list_key in ("invocation_edges", "inheritance_edges", "reference_edges"):
        edges = enhanced_data_flow.get(edge_list_key, [])
        before = len(edges)
        edges[:] = [e for e in edges if not _edge_involves_pruned_unit(e)]
        after = len(edges)
        if before != after:
            logger.info(
                f"[prune_orphan_interfaces] Removed {before - after} edges from {edge_list_key}"
            )

    # ---- 4. Identify features that are now fully orphaned ----
    pruned_key_set = {f"{pu['file_path']}::{pu['unit_name']}" for pu in pruned_units}
    orphan_feature_paths: Set[str] = set()
    for feature_path, all_keys in feature_to_all_unit_keys.items():
        if all_keys and all_keys.issubset(pruned_key_set):
            orphan_feature_paths.add(feature_path)

    if orphan_feature_paths:
        logger.info(
            f"[prune_orphan_interfaces] {len(orphan_feature_paths)} features fully orphaned: "
            + ", ".join(sorted(orphan_feature_paths)[:10])
        )

    # ---- 5. Collect surviving feature paths for RPG pruning ----
    surviving = _collect_surviving_features(interfaces_data)

    return {
        "pruned_units": pruned_units,
        "pruned_files": pruned_files,
        "orphan_feature_paths": orphan_feature_paths,
        "surviving_feature_paths": surviving,
    }


def _collect_surviving_features(interfaces_data: Dict[str, Any]) -> Set[str]:
    """Collect all feature paths that still have at least one interface unit."""
    surviving: Set[str] = set()
    for st_data in interfaces_data.get("subtrees", {}).values():
        file_interfaces = st_data.get("interfaces", st_data.get("files", {}))
        for file_data in file_interfaces.values():
            for features in file_data.get("units_to_features", {}).values():
                surviving.update(features)
    return surviving


def print_review_summary(review_result: Dict[str, Any]):
    """Print a human-readable summary of the global review results."""
    print("\n" + "=" * 60)
    print("GLOBAL INTERFACE REVIEW SUMMARY")
    print("=" * 60)

    iterations = review_result.get("iterations_run", 0)
    passed = review_result.get("passed", False)

    print(f"Iterations: {iterations}")
    print(f"Final Status: {'[OK] PASSED' if passed else '[FAIL] NEEDS ATTENTION'}")

    # Entry points
    entry_points = review_result.get("final_entry_points", [])
    if entry_points:
        print(f"\nEntry Points ({len(entry_points)}):")
        for ep in entry_points:
            print(f"  -  {ep.get('unit_name', '?')} in {ep.get('file_path', '?')}")
            if ep.get("rationale"):
                print(f"    Reason: {ep['rationale']}")
    
    # Feature orphans
    feature_orphans = review_result.get("final_feature_orphans", [])
    if feature_orphans:
        print(f"\nOrphan Features ({len(feature_orphans)}):")
        for fo in feature_orphans[:10]:
            print(
                f"  -  {fo.get('unit_name', '?')} in {fo.get('file_path', '?')} "
                f"({fo.get('subtree', '?')})"
            )
        if len(feature_orphans) > 10:
            print(f"  ... and {len(feature_orphans) - 10} more")

    print("=" * 60)


# ============================================================================
# Orphan Unit Review
# ============================================================================


@dataclass
class OrphanReviewResult:
    """Result of orphan unit review."""
    decisions: Dict[str, str] = field(default_factory=dict)  # unit_key -> "retain" | "prune"
    completed_edges: Dict[str, Dict[str, List[Dict]]] = field(default_factory=dict)  # unit_key -> edges dict

    @property
    def keys_to_prune(self) -> List[str]:
        return [k for k, d in self.decisions.items() if d == "prune"]

    @property
    def keys_to_retain(self) -> List[str]:
        return [k for k, d in self.decisions.items() if d == "retain"]

    def get_all_edges(self) -> Dict[str, List[Dict]]:
        """Aggregate all completed edges by type."""
        result: Dict[str, List[Dict]] = {
            "inheritance_edges": [],
            "invocation_edges": [],
            "reference_edges": [],
        }
        for edges_dict in self.completed_edges.values():
            for edge_type, edges in edges_dict.items():
                if edge_type in result and edges:
                    result[edge_type].extend(edges)
        return result


def review_orphan_units(
    orphan_details: List[Dict[str, Any]],
    repo_info: str,
    subtree_interfaces: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
) -> OrphanReviewResult:
    """Review orphan units using LLM to determine which should be retained or pruned.

    Units are grouped by subtree for better context during review.

    Args:
        orphan_details: List of orphan unit details from InterfacesStore.get_orphan_unit_details()
        repo_info: Repository description for context
        subtree_interfaces: Optional dict mapping subtree -> interfaces data for context
        llm_client: LLM client to use (creates new one if not provided)

    Returns:
        OrphanReviewResult with decisions and completed edges
    """
    if not orphan_details:
        logger.info("[review_orphan_units] No orphan units to review")
        return OrphanReviewResult()

    llm = llm_client or LLMClient()
    result = OrphanReviewResult()

    # Group orphans by subtree
    orphans_by_subtree: Dict[str, List[Dict[str, Any]]] = {}
    for detail in orphan_details:
        subtree = detail.get("subtree", "unknown")
        orphans_by_subtree.setdefault(subtree, []).append(detail)

    # Review each subtree's orphans together
    for subtree, subtree_orphans in orphans_by_subtree.items():
        # Get subtree context if available
        subtree_context = None
        if subtree_interfaces and subtree in subtree_interfaces:
            subtree_context = subtree_interfaces[subtree]

        batch_result = _review_orphan_batch(
            subtree_orphans, repo_info, subtree, subtree_context, llm
        )
        result.decisions.update(batch_result.decisions)
        result.completed_edges.update(batch_result.completed_edges)

    logger.info(
        f"[review_orphan_units] Reviewed {len(orphan_details)} orphan units across "
        f"{len(orphans_by_subtree)} subtrees: "
        f"{len(result.keys_to_retain)} retain, "
        f"{len(result.keys_to_prune)} prune, "
        f"{len(result.completed_edges)} with completed edges"
    )

    return result


def _review_orphan_batch(
    batch: List[Dict[str, Any]],
    repo_info: str,
    subtree_name: str,
    subtree_context: Optional[Dict[str, Any]],
    llm: LLMClient,
) -> OrphanReviewResult:
    """Review orphan units from a single subtree."""
    # Build user prompt with orphan details
    orphan_summaries = []
    for detail in batch:
        summary = f"""
### Unit: {detail['unit_key']}
- File: {detail['file_path']}
- Features: {', '.join(detail['features']) if detail['features'] else '(none)'}

Code:
```python
{detail['code']}
```
"""
        orphan_summaries.append(summary)

    user_prompt = f"""## Repository Context
{repo_info}

## Subtree: {subtree_name}

## Orphan Units to Review
The following {len(batch)} interface units in subtree "{subtree_name}" have no incoming or outgoing call edges.
Determine whether each should be retained or pruned.

{''.join(orphan_summaries)}
"""

    if subtree_context:
        # Extract other unit names in the subtree for context
        other_units = []
        interfaces = subtree_context.get("interfaces", {})
        for file_path, file_data in interfaces.items():
            units_to_code = file_data.get("units_to_code", {})
            for unit_name in units_to_code.keys():
                unit_key = f"{file_path}::{unit_name}"
                # Exclude current orphans from context
                if not any(d["unit_key"] == unit_key for d in batch):
                    other_units.append(unit_key)

        if other_units:
            user_prompt += f"""
## Other Units in This Subtree (for context)
{', '.join(other_units[:20])}{'...' if len(other_units) > 20 else ''}
"""

    combined_prompt = f"{ORPHAN_REVIEW_PROMPT}\n\n{user_prompt}"

    result = OrphanReviewResult()

    try:
        response = llm.generate(combined_prompt, purpose="orphan_review")

        # Parse JSON response using LLMClient's built-in method
        parsed = llm.parse_json_block(response)
        if not parsed:
            logger.error("[orphan_review] Failed to parse LLM response as JSON")
            for detail in batch:
                result.decisions[detail["unit_key"]] = "retain"
            return result

        reviews = parsed.get("reviews", [])

        for review in reviews:
            unit_key = review.get("unit_key", "")
            decision = review.get("decision", "retain").lower()
            reason = review.get("reason", "")
            edges = review.get("edges")

            if decision not in ("retain", "prune"):
                decision = "retain"  # Default to retain if unclear

            result.decisions[unit_key] = decision

            # Collect completed edges if provided
            if edges and isinstance(edges, dict):
                valid_edges = {}
                for edge_type in ("inheritance_edges", "invocation_edges", "reference_edges"):
                    if edge_type in edges and edges[edge_type]:
                        valid_edges[edge_type] = edges[edge_type]
                if valid_edges:
                    result.completed_edges[unit_key] = valid_edges
                    logger.info(
                        f"[orphan_review] {unit_key}: {decision} - {reason} "
                        f"(+{sum(len(e) for e in valid_edges.values())} edges)"
                    )
                else:
                    logger.info(f"[orphan_review] {unit_key}: {decision} - {reason}")
            else:
                logger.info(f"[orphan_review] {unit_key}: {decision} - {reason}")

        # Ensure all units in batch have a decision (default to retain)
        for detail in batch:
            if detail["unit_key"] not in result.decisions:
                result.decisions[detail["unit_key"]] = "retain"
                logger.warning(
                    f"[orphan_review] {detail['unit_key']}: defaulting to retain (missing from LLM response)"
                )

        return result

    except Exception as e:
        logger.error(f"[orphan_review] Error during review: {e}")
        # Default all to retain on error
        for detail in batch:
            result.decisions[detail["unit_key"]] = "retain"
        return result
