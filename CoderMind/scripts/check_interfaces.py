#!/usr/bin/env python3
"""Check Interfaces Script - Validation for interfaces.json.

Validates the interfaces.json file and determines the execution state:
- "error": Input file missing or invalid
- "init": No interfaces.json exists or it's invalid
- "warning": interfaces.json exists but has feature mismatches with skeleton
- "update": Valid interfaces.json exists and is consistent

Cross-validates feature paths between skeleton.json and interfaces.json.
Also validates RPG feature nodes have proper meta.path assignments.

Returns JSON with validation status and statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

# Import centralized paths
from common.paths import SKELETON_FILE, INTERFACES_FILE, REPO_RPG_FILE


def validate_skeleton(skeleton_path: Path) -> Tuple[bool, List[str]]:
    """Validate that skeleton.json exists and is valid."""
    errors = []
    
    if not skeleton_path.exists():
        errors.append(f"Input file not found: {skeleton_path}")
        return False, errors
    
    try:
        with open(skeleton_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors
    
    # Check required structure
    if "root" not in data:
        errors.append("Missing required field: 'root'")
        return False, errors
    
    root = data["root"]
    if not isinstance(root, dict):
        errors.append("'root' must be an object")
        return False, errors
    
    if root.get("type") != "directory":
        errors.append("'root.type' must be 'directory'")
        return False, errors
    
    return True, errors


def get_files_from_skeleton(skeleton_path: Path) -> List[Dict[str, Any]]:
    """Extract all files from skeleton tree."""
    with open(skeleton_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def collect_files(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        files = []
        if node.get("type") == "file":
            files.append({
                "path": node.get("path", ""),
                "feature_paths": node.get("feature_paths", []),
                "component": node.get("component", "")
            })
        else:
            for child in node.get("children", []):
                files.extend(collect_files(child))
        return files
    
    return collect_files(data.get("root", {}))


def get_all_features_from_skeleton(skeleton_path: Path) -> Set[str]:
    """Extract all feature paths from skeleton.json."""
    files = get_files_from_skeleton(skeleton_path)
    features = set()
    for f in files:
        features.update(f.get("feature_paths", []))
    return features


def get_all_features_from_interfaces(interfaces_path: Path) -> Set[str]:
    """Extract all feature paths from interfaces.json."""
    with open(interfaces_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = set()
    # Support both "subtrees" (new format) and "components" (old format)
    containers = data.get("subtrees", data.get("components", {}))
    for comp_name, comp_data in containers.items():
        # Support both "interfaces" (reference format) and "files" (old format) as container key
        file_container = comp_data.get("interfaces", comp_data.get("files", {}))
        for file_path, file_data in file_container.items():
            # Reference format: units_to_features at file level
            units_to_features = file_data.get("units_to_features", {})
            for unit_name, unit_features in units_to_features.items():
                if isinstance(unit_features, list):
                    features.update(unit_features)
    return features


def cross_validate_features(skeleton_features: Set[str], interfaces_features: Set[str]) -> Dict[str, Any]:
    """Cross-validate features between skeleton and interfaces.
    
    Returns dict with:
        - in_skeleton_not_interfaces: features in skeleton but not in interfaces
        - in_interfaces_not_skeleton: features in interfaces but not in skeleton
        - matched_count: number of matched features
        - warnings: list of warning messages
    """
    in_skeleton_not_interfaces = skeleton_features - interfaces_features
    in_interfaces_not_skeleton = interfaces_features - skeleton_features
    matched = skeleton_features & interfaces_features
    
    warnings = []
    
    for feat in sorted(in_skeleton_not_interfaces):
        warnings.append({
            "type": "missing_in_interfaces",
            "feature": feat,
            "message": f"Feature '{feat}' exists in skeleton.json but not mapped in interfaces.json"
        })
    
    for feat in sorted(in_interfaces_not_skeleton):
        warnings.append({
            "type": "missing_in_skeleton",
            "feature": feat,
            "message": f"Feature '{feat}' mapped in interfaces.json but not in skeleton.json"
        })
    
    return {
        "in_skeleton_not_interfaces": sorted(list(in_skeleton_not_interfaces)),
        "in_interfaces_not_skeleton": sorted(list(in_interfaces_not_skeleton)),
        "matched_count": len(matched),
        "skeleton_feature_count": len(skeleton_features),
        "interfaces_feature_count": len(interfaces_features),
        "warnings": warnings,
        "is_consistent": len(warnings) == 0
    }


def validate_interfaces(interfaces_path: Path, skeleton_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate interfaces.json structure and content."""
    errors = []
    stats = {
        "components": 0,
        "files": 0,
        "units": 0,
        "features_mapped": 0
    }
    
    if not interfaces_path.exists():
        errors.append(f"Output file not found: {interfaces_path}")
        return False, errors, stats
    
    try:
        with open(interfaces_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors, stats
    
    # Check required structure - support both "subtrees" and "components"
    containers = data.get("subtrees", data.get("components", {}))
    if not containers:
        errors.append("Missing required field: 'subtrees' or 'components'")
        return False, errors, stats
    
    if not isinstance(containers, dict):
        errors.append("'subtrees'/'components' must be an object")
        return False, errors, stats
    
    stats["components"] = len(containers)
    
    # Get expected files from skeleton
    expected_files = set()
    if skeleton_path.exists():
        skeleton_files = get_files_from_skeleton(skeleton_path)
        expected_files = {f["path"] for f in skeleton_files if f.get("feature_paths")}
    
    # Validate each component
    all_features = set()
    designed_files = set()
    
    for comp_name, comp_data in containers.items():
        if not isinstance(comp_data, dict):
            errors.append(f"Component '{comp_name}' must be an object")
            continue
        
        # Support both "interfaces" (reference format) and "files" (old format) as container key
        file_container = comp_data.get("interfaces", comp_data.get("files", {}))
        if not isinstance(file_container, dict):
            errors.append(f"Component '{comp_name}.interfaces/files' must be an object")
            continue
        
        for file_path, file_data in file_container.items():
            stats["files"] += 1
            designed_files.add(file_path)
            
            if not isinstance(file_data, dict):
                errors.append(f"File '{file_path}' data must be an object")
                continue
            
            # Reference format: units, units_to_features, units_to_code at file level
            units = file_data.get("units", [])
            if not isinstance(units, list):
                errors.append(f"File '{file_path}.units' must be a list")
            else:
                stats["units"] += len(units)
            
            units_to_features = file_data.get("units_to_features", {})
            if not isinstance(units_to_features, dict):
                errors.append(f"File '{file_path}.units_to_features' must be an object")
            else:
                for unit_name, features in units_to_features.items():
                    if isinstance(features, list):
                        all_features.update(features)
            
            units_to_code = file_data.get("units_to_code", {})
            if not isinstance(units_to_code, dict):
                errors.append(f"File '{file_path}.units_to_code' must be an object")
    
    stats["features_mapped"] = len(all_features)
    
    # Check coverage
    missing_files = expected_files - designed_files
    if missing_files:
        # This is a warning, not an error
        pass
    
    is_valid = len(errors) == 0
    return is_valid, errors, stats


def validate_rpg_feature_paths(rpg_path: Path) -> Dict[str, Any]:
    """Validate that feature nodes in RPG have proper meta.path assignments.
    
    Returns:
        Dict with:
        - features_with_path: count of features with valid meta.path
        - features_without_path: count of features missing meta.path
        - same_unit_edges: count of SAME_UNIT edges
        - warnings: list of validation warnings
    """
    result = {
        "features_with_path": 0,
        "features_without_path": 0,
        "same_unit_edges": 0,
        "warnings": [],
        "is_valid": True
    }
    
    if not rpg_path.exists():
        result["warnings"].append(f"RPG file not found: {rpg_path}")
        result["is_valid"] = False
        return result
    
    try:
        with open(rpg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["warnings"].append(f"Invalid RPG JSON: {e}")
        result["is_valid"] = False
        return result
    
    # Count SAME_UNIT edges
    for edge in data.get("edges", []):
        if edge.get("relation") == "same_unit":
            result["same_unit_edges"] += 1
    
    # Traverse tree to find feature nodes
    def check_node(node_data: Dict[str, Any]):
        node_type = node_data.get("node_type")
        meta = node_data.get("meta", {})
        name = node_data.get("name", "")
        
        # Check if this is a feature node (leaf level or node_type == "feature")
        if node_type == "feature":
            path = meta.get("path") if meta else None
            if path:
                result["features_with_path"] += 1
            else:
                result["features_without_path"] += 1
                result["warnings"].append(f"Feature '{name}' missing meta.path")
        
        # Recurse into children
        for child in node_data.get("children", []):
            check_node(child)
    
    root = data.get("root")
    if root:
        check_node(root)
    
    # Mark as invalid if there are features without paths
    if result["features_without_path"] > 0:
        result["is_valid"] = False
    
    return result


def check_state(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Check the current state and return execution guidance."""
    result = {
        "type": "error",
        "message": "",
        "input_exists": input_path.exists(),
        "input_valid": False,
        "output_exists": output_path.exists(),
        "output_valid": False,
        "validation_errors": [],
        "stats": {},
        "cross_validation": None
    }
    
    # Check input (skeleton.json)
    if not result["input_exists"]:
        result["type"] = "error"
        result["message"] = f"Input file not found: {input_path}. Please run /cmind.build_skeleton first."
        return result
    
    input_valid, input_errors = validate_skeleton(input_path)
    result["input_valid"] = input_valid
    
    if not input_valid:
        result["type"] = "error"
        result["message"] = "Invalid skeleton.json"
        result["validation_errors"] = input_errors
        return result
    
    # Get skeleton features for cross-validation
    skeleton_features = get_all_features_from_skeleton(input_path)
    result["input_statistics"] = {
        "total_features": len(skeleton_features)
    }
    
    # Check output (interfaces.json)
    if not result["output_exists"]:
        result["type"] = "init"
        result["message"] = "Ready to design interfaces. No existing interfaces.json found."
        return result
    
    output_valid, output_errors, stats = validate_interfaces(output_path, input_path)
    result["output_valid"] = output_valid
    result["stats"] = stats
    
    if not output_valid:
        result["type"] = "init"
        result["message"] = "Existing interfaces.json is invalid. Will regenerate."
        result["validation_errors"] = output_errors
        return result
    
    # Cross-validate features
    interfaces_features = get_all_features_from_interfaces(output_path)
    cross_validation = cross_validate_features(skeleton_features, interfaces_features)
    result["cross_validation"] = cross_validation
    
    # Validate RPG feature paths
    rpg_validation = validate_rpg_feature_paths(REPO_RPG_FILE)
    result["rpg_validation"] = rpg_validation
    
    # Determine type based on cross-validation and RPG validation
    if not cross_validation["is_consistent"]:
        warning_count = len(cross_validation["warnings"])
        result["type"] = "warning"
        result["message"] = f"interfaces.json exists but has {warning_count} feature mismatches with skeleton."
    elif not rpg_validation["is_valid"]:
        missing_count = rpg_validation["features_without_path"]
        result["type"] = "warning"
        result["message"] = f"interfaces.json valid but {missing_count} features in RPG missing meta.path."
    else:
        result["type"] = "update"
        result["message"] = (f"Valid interfaces.json exists with {stats['units']} units across {stats['files']} files. "
                            f"RPG has {rpg_validation['features_with_path']} features with paths, "
                            f"{rpg_validation['same_unit_edges']} SAME_UNIT edges.")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check interfaces.json validity and state"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=SKELETON_FILE,
        help=f"Input skeleton file (default: {SKELETON_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=INTERFACES_FILE,
        help=f"Output interfaces file to check (default: {INTERFACES_FILE})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only"
    )
    
    args = parser.parse_args()
    
    result = check_state(args.input, args.output)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nState: {result['type']}")
        print(f"Message: {result['message']}")
        
        if result['validation_errors']:
            print("\nValidation Errors:")
            for err in result['validation_errors']:
                print(f"  - {err}")
        
        if result['stats']:
            print("\nStatistics:")
            for key, value in result['stats'].items():
                print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    exit(main())
