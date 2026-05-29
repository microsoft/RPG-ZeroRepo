#!/usr/bin/env python3
"""Check Data Flow Script.

Function: Validate data_flow.json state and cross-validate with skeleton.json
- Checks if data_flow.json exists (init state)
- Validates data flow structure (error state if invalid)
- Cross-validates components between skeleton and data flow (warning state)
- Returns update state if valid

Input: .cmind/data_flow.json
Reference: .cmind/skeleton.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

from common.paths import DATA_FLOW_FILE, SKELETON_FILE


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def get_components_from_skeleton(skeleton: Dict[str, Any]) -> Set[str]:
    """Extract component names from skeleton."""
    components = set()
    
    def traverse(node: Dict[str, Any]):
        if node.get("type") == "file":
            feature_paths = node.get("feature_paths", [])
            for fp in feature_paths:
                if "/" in fp:
                    component = fp.split("/")[0]
                    components.add(component)
            comp = node.get("component", "")
            if comp:
                components.add(comp)
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    
    return components


def get_components_from_data_flow(data_flow: Dict[str, Any]) -> Set[str]:
    """Extract component names from data flow."""
    components = set()
    
    # From components list
    for comp in data_flow.get("components", []):
        components.add(comp)
    
    # From subtree_order
    for comp in data_flow.get("subtree_order", []):
        components.add(comp)
    
    # From data flow edges
    for edge in data_flow.get("data_flow", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source:
            components.add(source)
        if target:
            components.add(target)
    
    return components


def validate_data_flow_structure(data_flow: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate data flow structure."""
    errors = []
    
    edges = data_flow.get("data_flow", [])
    
    if not isinstance(edges, list):
        errors.append("'data_flow' must be a list")
        return False, errors
    
    # Check each edge
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"Edge {i}: must be a dictionary")
            continue
        
        # Required fields
        for field in ["source", "target", "data_id", "data_type", "transformation"]:
            if field not in edge:
                errors.append(f"Edge {i}: missing required field '{field}'")
            elif not edge[field]:
                errors.append(f"Edge {i}: field '{field}' is empty")
        
        # No self-loops
        if edge.get("source") == edge.get("target"):
            errors.append(f"Edge {i}: self-loop detected ({edge.get('source')} -> {edge.get('source')})")
    
    # Check for cycles
    graph = defaultdict(list)
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source and target:
            graph[source].append(target)
    
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: str, path: List[str]) -> Tuple[bool, List[str]]:
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                found, cycle_path = has_cycle(neighbor, path + [neighbor])
                if found:
                    return True, cycle_path
            elif neighbor in rec_stack:
                return True, path + [neighbor]
        
        rec_stack.remove(node)
        return False, []
    
    for node in graph:
        if node not in visited:
            found, cycle_path = has_cycle(node, [node])
            if found:
                errors.append(f"Cycle detected: {' -> '.join(cycle_path)}")
                break
    
    return len(errors) == 0, errors


def cross_validate_components(
    skeleton_components: Set[str],
    data_flow_components: Set[str]
) -> Tuple[bool, Dict[str, Any]]:
    """Cross-validate components between skeleton and data flow.
    
    Returns:
        (is_consistent, details)
    """
    in_skeleton_only = skeleton_components - data_flow_components
    in_data_flow_only = data_flow_components - skeleton_components
    matched = skeleton_components & data_flow_components
    
    is_consistent = len(in_skeleton_only) == 0 and len(in_data_flow_only) == 0
    
    return is_consistent, {
        "skeleton_components": len(skeleton_components),
        "data_flow_components": len(data_flow_components),
        "matched": len(matched),
        "in_skeleton_only": sorted(list(in_skeleton_only)),
        "in_data_flow_only": sorted(list(in_data_flow_only))
    }


def inspect_state(data_flow_path: Path, skeleton_path: Path) -> Dict[str, Any]:
    """Inspect current state and determine action needed.
    
    Returns dict with:
    - type: "error" | "init" | "warning" | "update"
    - message: description
    - details: additional info
    """
    # Check if data_flow.json exists
    if not data_flow_path.exists():
        return {
            "type": "init",
            "message": "data_flow.json not found - need to run build_data_flow",
            "details": {}
        }
    
    # Load and validate
    try:
        with open(data_flow_path, 'r', encoding='utf-8') as f:
            data_flow = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "type": "error",
            "message": f"Invalid JSON in data_flow.json: {e}",
            "details": {}
        }
    
    # Check for error field
    if "error" in data_flow:
        return {
            "type": "error",
            "message": f"Data flow has error: {data_flow['error']}",
            "details": {}
        }
    
    # Validate structure
    is_valid, errors = validate_data_flow_structure(data_flow)
    if not is_valid:
        return {
            "type": "error",
            "message": "Data flow structure is invalid",
            "details": {"errors": errors}
        }
    
    # Cross-validate with skeleton if available
    if skeleton_path.exists():
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeleton = json.load(f)
            
            skeleton_components = get_components_from_skeleton(skeleton)
            data_flow_components = get_components_from_data_flow(data_flow)
            
            is_consistent, xval_details = cross_validate_components(
                skeleton_components, data_flow_components
            )
            
            if not is_consistent:
                return {
                    "type": "warning",
                    "message": "Component mismatch between skeleton and data flow",
                    "details": xval_details
                }
            
            # All good
            return {
                "type": "update",
                "message": "Data flow is valid and consistent",
                "details": {
                    "edge_count": len(data_flow.get("data_flow", [])),
                    "component_count": len(data_flow_components),
                    "subtree_order": data_flow.get("subtree_order", [])
                }
            }
            
        except Exception as e:
            # Skeleton load failed, just validate data flow
            return {
                "type": "update",
                "message": f"Data flow is valid (skeleton check skipped: {e})",
                "details": {
                    "edge_count": len(data_flow.get("data_flow", [])),
                    "component_count": len(get_components_from_data_flow(data_flow))
                }
            }
    
    # No skeleton to compare
    return {
        "type": "update",
        "message": "Data flow is valid (no skeleton to cross-validate)",
        "details": {
            "edge_count": len(data_flow.get("data_flow", [])),
            "component_count": len(get_components_from_data_flow(data_flow))
        }
    }


def print_state(result: Dict[str, Any]) -> None:
    """Print state information."""
    state = result["type"]
    message = result["message"]
    details = result.get("details", {})
    
    state_icons = {
        "error": "[FAIL]",
        "init": "[-]",
        "warning": "[WARNING]",
        "update": "[OK]"
    }
    
    icon = state_icons.get(state, "[?]")
    print(f"\n{icon} State: {state.upper()}")
    print(f"   {message}")
    
    if state == "error" and "errors" in details:
        print("\n   Errors:")
        for err in details["errors"][:10]:
            print(f"   - {err}")
        if len(details.get("errors", [])) > 10:
            print(f"   ... and {len(details['errors']) - 10} more")
    
    elif state == "warning":
        if details.get("in_skeleton_only"):
            print("\n   Components in skeleton but not in data flow:")
            for comp in details["in_skeleton_only"][:5]:
                print(f"   - {comp}")
        if details.get("in_data_flow_only"):
            print("\n   Components in data flow but not in skeleton:")
            for comp in details["in_data_flow_only"][:5]:
                print(f"   - {comp}")
    
    elif state == "update":
        if "edge_count" in details:
            print(f"\n   Data Flow Edges: {details['edge_count']}")
        if "component_count" in details:
            print(f"   Components: {details['component_count']}")
        if details.get("subtree_order"):
            print(f"   Subtree Order: {' → '.join(details['subtree_order'][:5])}")
            if len(details.get("subtree_order", [])) > 5:
                print(f"                  ... and {len(details['subtree_order']) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Check data flow state"
    )
    parser.add_argument(
        "--data-flow", "-d",
        type=Path,
        default=DATA_FLOW_FILE,
        help="Data flow file to check"
    )
    parser.add_argument(
        "--skeleton", "-s",
        type=Path,
        default=SKELETON_FILE,
        help="Skeleton file for cross-validation"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed edge list and component information"
    )
    
    args = parser.parse_args()
    
    result = inspect_state(args.data_flow, args.skeleton)
    
    # In verbose mode, include all edges and component details
    if args.verbose and result.get("type") == "update":
        data_flow_data = load_json(args.data_flow)
        if data_flow_data:
            result["edges"] = data_flow_data.get("data_flow", [])
            result["subtree_order"] = data_flow_data.get("subtree_order", [])
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 50)
        print("DATA FLOW CHECK")
        print("=" * 50)
        print_state(result)
        
        # Print verbose details
        if args.verbose and result.get("type") == "update":
            edges = result.get("edges", [])
            if edges:
                print("\nData Flow Edges:")
                for edge in edges:
                    print(f"   {edge.get('source', '?')} → {edge.get('target', '?')}: {edge.get('data_id', '?')} ({edge.get('data_type', '?')})")
            
            subtree_order = result.get("subtree_order", [])
            if subtree_order:
                print(f"\nSubtree Order: {' → '.join(subtree_order)}")
    
    # Return exit code based on state
    if result["type"] == "error":
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
