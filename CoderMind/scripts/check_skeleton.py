#!/usr/bin/env python3
"""Check Skeleton Script.

Inspect .cmind/skeleton.json and validate its structure.
Also cross-validate feature paths between refactor_feature.json and skeleton.json.

Decision rules:
- Check if input file (refactor_feature.json) exists
- Check if output file (skeleton.json) exists and has required fields
- Validate skeleton tree structure
- Cross-validate feature paths between input and output

The script prints EXACTLY ONE JSON object to stdout.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Import centralized paths
from common.paths import FEATURE_TREE_FILE as INPUT_FILE, SKELETON_FILE as OUTPUT_FILE


def load_json(path: Path) -> Dict[str, Any] | None:
    """Load JSON file safely."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                return data
    except Exception:
        pass
    return None


# ============================================================================
# Feature Path Extraction from refactor_feature.json
# ============================================================================

def get_all_feature_paths_from_subtree(subtree: Dict[str, Any], prefix: str = "") -> List[str]:
    """Extract all feature paths from a refactored_subtree.

    Features are the leaf nodes in the tree structure.
    This mirrors the logic in build_skeleton.py.
    """
    paths = []
    if isinstance(subtree, dict):
        for key, value in subtree.items():
            # Skip 'description' keys as they are metadata
            if key in ("description",):
                continue
            
            new_prefix = f"{prefix}/{key}" if prefix else key
            
            if key == "children":
                # 'children' is a container, recurse into it
                paths.extend(get_all_feature_paths_from_subtree(value, prefix))
            elif isinstance(value, dict):
                # Check if this is a leaf node (only has 'description')
                if set(value.keys()) <= {"description"}:
                    paths.append(new_prefix)
                else:
                    # Has more content, recurse
                    paths.extend(get_all_feature_paths_from_subtree(value, new_prefix))
            elif isinstance(value, list):
                # List of leaf features
                for item in value:
                    if isinstance(item, str):
                        paths.append(f"{new_prefix}/{item}" if new_prefix else item)
                    elif isinstance(item, dict):
                        paths.extend(get_all_feature_paths_from_subtree(item, new_prefix))
            else:
                paths.append(new_prefix)
    elif isinstance(subtree, list):
        for item in subtree:
            if isinstance(item, str):
                paths.append(f"{prefix}/{item}" if prefix else item)
            elif isinstance(item, dict):
                paths.extend(get_all_feature_paths_from_subtree(item, prefix))
    
    return paths


def get_features_from_refactor(data: Dict[str, Any]) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Extract all feature paths from refactor_feature.json.
    
    Returns:
        - Set of all feature paths
        - Dict mapping component name to its feature paths
    """
    all_features = set()
    features_by_component = {}
    
    components = data.get("components", [])
    if not isinstance(components, list):
        return all_features, features_by_component
    
    for comp in components:
        comp_name = comp.get("name", "unknown")
        subtree = comp.get("refactored_subtree", {})
        
        # Get features with component prefix (as build_skeleton does)
        comp_features = get_all_feature_paths_from_subtree(subtree, prefix=comp_name)
        
        features_by_component[comp_name] = comp_features
        all_features.update(comp_features)
    
    return all_features, features_by_component


# ============================================================================
# Feature Path Extraction from skeleton.json
# ============================================================================

def get_all_feature_paths_from_skeleton(node: Dict[str, Any]) -> Set[str]:
    """Extract all feature paths from skeleton tree."""
    features = set()
    
    if node.get("type") == "file":
        for fp in node.get("feature_paths", []):
            features.add(fp)
    else:
        for child in node.get("children", []):
            features.update(get_all_feature_paths_from_skeleton(child))
    
    return features


# ============================================================================
# Cross Validation
# ============================================================================

def cross_validate_features(
    input_features: Set[str], 
    skeleton_features: Set[str]
) -> Dict[str, Any]:
    """Cross-validate features between input (refactor_feature) and output (skeleton).
    
    Returns dict with:
        - in_input_not_skeleton: features in refactor_feature but not in skeleton
        - in_skeleton_not_input: features in skeleton but not in refactor_feature
        - matched_count: number of matched features
        - warnings: list of warning messages
    """
    in_input_not_skeleton = input_features - skeleton_features
    in_skeleton_not_input = skeleton_features - input_features
    matched = input_features & skeleton_features
    
    warnings = []
    
    # Generate warnings for missing features
    for feat in sorted(in_input_not_skeleton):
        warnings.append({
            "type": "missing_in_skeleton",
            "feature": feat,
            "message": f"Feature '{feat}' exists in refactor_feature.json but not in skeleton.json"
        })
    
    for feat in sorted(in_skeleton_not_input):
        warnings.append({
            "type": "missing_in_input",
            "feature": feat,
            "message": f"Feature '{feat}' exists in skeleton.json but not in refactor_feature.json"
        })
    
    return {
        "in_input_not_skeleton": sorted(list(in_input_not_skeleton)),
        "in_skeleton_not_input": sorted(list(in_skeleton_not_input)),
        "matched_count": len(matched),
        "input_feature_count": len(input_features),
        "skeleton_feature_count": len(skeleton_features),
        "warnings": warnings,
        "is_consistent": len(warnings) == 0
    }


# ============================================================================
# Skeleton Structure Validation
# ============================================================================


# ============================================================================
# Skeleton Structure Validation
# ============================================================================

def count_files_in_tree(node: Dict[str, Any]) -> int:
    """Count total files in skeleton tree."""
    if node.get("type") == "file":
        return 1
    
    count = 0
    for child in node.get("children", []):
        count += count_files_in_tree(child)
    
    return count


def count_features_in_tree(node: Dict[str, Any]) -> int:
    """Count total features in skeleton tree."""
    if node.get("type") == "file":
        return len(node.get("feature_paths", []))
    
    count = 0
    for child in node.get("children", []):
        count += count_features_in_tree(child)
    
    return count


def get_all_files(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all file nodes from skeleton tree."""
    files = []
    
    if node.get("type") == "file":
        files.append({
            "path": node.get("path", ""),
            "feature_count": len(node.get("feature_paths", [])),
            "component": node.get("component", "")
        })
    else:
        for child in node.get("children", []):
            files.extend(get_all_files(child))
    
    return files


def validate_tree_structure(node: Dict[str, Any], errors: List[str], path: str = "") -> bool:
    """Validate skeleton tree structure recursively."""
    # Check required fields
    if "type" not in node:
        errors.append(f"Missing 'type' at {path or 'root'}")
        return False
    
    if "name" not in node:
        errors.append(f"Missing 'name' at {path or 'root'}")
        return False
    
    node_type = node.get("type")
    node_path = node.get("path", path)
    
    if node_type == "directory":
        # Directories should have children
        children = node.get("children", [])
        if not isinstance(children, list):
            errors.append(f"Invalid 'children' at {node_path}")
            return False
        
        # Recursively validate children
        for child in children:
            validate_tree_structure(child, errors, f"{node_path}/{child.get('name', '?')}")
    
    elif node_type == "file":
        # Files should have feature_paths
        features = node.get("feature_paths")
        if features is not None and not isinstance(features, list):
            errors.append(f"Invalid 'feature_paths' at {node_path}")
            return False
    
    else:
        errors.append(f"Unknown type '{node_type}' at {node_path}")
        return False
    
    return True


def inspect_state() -> Dict[str, Any]:
    """Inspect skeleton state and return decision object."""
    # Check input file
    input_exists = INPUT_FILE.exists()
    input_data = load_json(INPUT_FILE) if input_exists else None
    input_valid = input_data is not None and "components" in input_data
    
    # Extract features from input
    input_features = set()
    features_by_component = {}
    if input_valid and input_data:
        input_features, features_by_component = get_features_from_refactor(input_data)
    
    # Check output file
    output_exists = OUTPUT_FILE.exists()
    output_data = load_json(OUTPUT_FILE) if output_exists else None
    
    # Validate output structure
    output_valid = False
    validation_errors = []
    statistics = {}
    files = []
    skeleton_features = set()
    cross_validation = None
    
    if output_data:
        # Check required fields
        required_fields = ["repository_name", "root"]
        missing_fields = [f for f in required_fields if f not in output_data]
        
        if not missing_fields:
            root = output_data.get("root", {})
            
            # Validate tree structure
            validate_tree_structure(root, validation_errors)
            
            if not validation_errors:
                output_valid = True
                
                # Extract features from skeleton
                skeleton_features = get_all_feature_paths_from_skeleton(root)
                
                # Collect statistics
                statistics = {
                    "total_files": count_files_in_tree(root),
                    "total_features": count_features_in_tree(root),
                    "components": list(output_data.get("component_directories", {}).keys())
                }
                
                # Get file list
                files = get_all_files(root)
                
                # Cross-validate features if both input and output are valid
                if input_valid:
                    cross_validation = cross_validate_features(input_features, skeleton_features)
        else:
            validation_errors.append(f"Missing required fields: {missing_fields}")
    
    # Determine type and message
    if not input_valid:
        type_value = "error"
        message = "Input file missing or invalid. Run /cmind.feature_refactor first."
    elif not output_exists or not output_valid:
        type_value = "init"
        message = "Ready to build skeleton."
    else:
        # Check cross-validation results
        if cross_validation and not cross_validation["is_consistent"]:
            type_value = "warning"
            warning_count = len(cross_validation["warnings"])
            message = f"Skeleton exists but has {warning_count} feature mismatches."
        else:
            type_value = "update"
            message = "Skeleton exists and is consistent. Regenerate?"
    
    result = {
        "type": type_value,
        "message": message,
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_exists": input_exists,
        "input_valid": input_valid,
        "output_exists": output_exists,
        "output_valid": output_valid,
        "validation_errors": validation_errors,
        "statistics": statistics,
        "files": files[:10],  # First 10 files for preview
        "files_total": len(files),
        "cross_validation": cross_validation,
    }

    # Add next_action for clear guidance
    if type_value == "init":
        result["next_action"] = "cmind script build_skeleton.py --max-iterations 10"
    elif type_value == "warning":
        result["next_action"] = "cmind script build_skeleton.py --patch"
    else:
        result["next_action"] = "Skeleton is consistent. Proceed to next step."
    
    # Add input feature count for reference
    if input_valid:
        result["input_statistics"] = {
            "total_features": len(input_features),
            "components": list(features_by_component.keys()),
            "features_by_component": {
                comp: len(feats) for comp, feats in features_by_component.items()
            }
        }
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check skeleton file state"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed file list and all feature mismatches"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Accepted for compatibility with the unified check_*.py contract used by "
            "plan.py; this script already prints JSON unconditionally."
        ),
    )

    args = parser.parse_args()
    
    result = inspect_state()
    
    # In verbose mode, include all files and feature details
    if args.verbose and result.get("output_valid"):
        output_data = load_json(OUTPUT_FILE)
        if output_data:
            result["files"] = get_all_files(output_data.get("root", {}))
        
        # Include full feature lists in verbose mode
        if result.get("input_valid"):
            input_data = load_json(INPUT_FILE)
            if input_data:
                input_features, features_by_component = get_features_from_refactor(input_data)
                result["input_features_detail"] = {
                    comp: sorted(feats) for comp, feats in features_by_component.items()
                }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
