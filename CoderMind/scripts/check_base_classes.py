#!/usr/bin/env python3
"""Check Base Classes Script.

Function: Validate base_classes.json state and target-language syntax
- Checks if base_classes.json exists (init state)
- Validates JSON structure (error state if invalid)
- Validates source syntax (error state if syntax errors)
- Returns update state if valid

Input: .cmind/base_classes.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

from common.language_meta import extract_language_metadata
from decoder_lang import get_backend
from func_design.base_class_agent import extract_declaration_names

# Import centralized paths
from common.paths import BASE_CLASSES_FILE


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def validate_base_classes_structure(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate base classes structure."""
    errors = []
    backend = get_backend(extract_language_metadata(data)[0])
    
    base_classes = data.get("base_classes", [])
    
    if not isinstance(base_classes, list):
        errors.append("'base_classes' must be a list")
        return False, errors
    
    for i, bc in enumerate(base_classes):
        if not isinstance(bc, dict):
            errors.append(f"Base class {i}: must be a dictionary")
            continue
        
        # Required fields: file_path, code, and scope
        for field in ["file_path", "code", "scope"]:
            if field not in bc:
                errors.append(f"Base class {i}: missing required field '{field}'")
            elif not bc[field]:
                errors.append(f"Base class {i}: field '{field}' is empty")
        
        code = bc.get("code", "")
        if code:
            is_valid, error = backend.syntax_check(code, bc.get("file_path", ""))
            if not is_valid:
                # Try to get name from bc or extract from code
                name = bc.get("name", "")
                if not name:
                    declarations = extract_declaration_names(code, backend)
                    name = declarations[0] if declarations else "unknown"
                errors.append(f"Base class {i} ({name}): syntax error - {error}")
    
    # Also validate data_structures if present
    data_structures = data.get("data_structures", [])
    if data_structures and not isinstance(data_structures, list):
        errors.append("'data_structures' must be a list")
    elif isinstance(data_structures, list):
        for i, ds in enumerate(data_structures):
            if not isinstance(ds, dict):
                errors.append(f"Data structure {i}: must be a dictionary")
                continue
            
            # code and subtree are required; file_path is optional (assigned later)
            for field in ["code", "subtree"]:
                if field not in ds:
                    errors.append(f"Data structure {i}: missing required field '{field}'")
                elif not ds[field]:
                    errors.append(f"Data structure {i}: field '{field}' is empty")
            
            # subtree must NOT be 'global'
            subtree = ds.get("subtree", "")
            if subtree.lower() == "global":
                errors.append(f"Data structure {i}: subtree cannot be 'global'")
            
            # data_flow_types is required and must be non-empty
            df_types = ds.get("data_flow_types", [])
            if not isinstance(df_types, list) or not df_types:
                errors.append(f"Data structure {i}: 'data_flow_types' must be a non-empty list")
            
            code = ds.get("code", "")
            if code:
                is_valid, error = backend.syntax_check(
                    code,
                    ds.get("file_path", f"data_structure{backend.file_extension}"),
                )
                if not is_valid:
                    declarations = extract_declaration_names(code, backend)
                    name = declarations[0] if declarations else "unknown"
                    errors.append(f"Data structure {i} ({name}): syntax error - {error}")
    
    return len(errors) == 0, errors


def inspect_state(base_classes_path: Path) -> Dict[str, Any]:
    """Inspect current state and determine action needed.
    
    Returns dict with:
    - type: "error" | "init" | "update"
    - message: description
    - details: additional info
    """
    # Check if base_classes.json exists
    if not base_classes_path.exists():
        return {
            "type": "init",
            "message": "base_classes.json not found - need to run design_base_classes",
            "details": {}
        }
    
    # Load and validate
    try:
        with open(base_classes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "type": "error",
            "message": f"Invalid JSON in base_classes.json: {e}",
            "details": {}
        }
    
    # Check for error field
    if "error" in data:
        return {
            "type": "error",
            "message": f"Base classes has error: {data['error']}",
            "details": {}
        }
    
    # Validate structure and syntax
    is_valid, errors = validate_base_classes_structure(data)
    if not is_valid:
        return {
            "type": "error",
            "message": "Base classes structure or syntax is invalid",
            "details": {"errors": errors}
        }
    
    # Gather details
    base_classes = data.get("base_classes", [])
    class_names = data.get("class_names", [])
    data_structures = data.get("data_structures", [])
    ds_class_names = data.get("data_structure_names", [])
    
    # Collect file paths from base_classes
    file_paths = [bc.get("file_path", "") for bc in base_classes if bc.get("file_path")]
    # Collect subtrees from data_structures (file_path may not be assigned yet)
    ds_subtrees = [ds.get("subtree", "") for ds in data_structures if ds.get("subtree")]
    ds_file_paths = [ds.get("file_path", "") for ds in data_structures if ds.get("file_path")]
    
    return {
        "type": "update",
        "message": "Base classes are valid",
        "details": {
            "file_count": len(base_classes),
            "class_count": len(class_names),
            "file_paths": file_paths,
            "class_names": class_names,
            "data_structure_count": len(data_structures),
            "data_structure_names": ds_class_names,
            "data_structure_subtrees": ds_subtrees,
            "data_structure_file_paths": ds_file_paths,
            "language": extract_language_metadata(data)[0],
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
    
    elif state == "update":
        if "file_count" in details:
            print(f"\n   Base Class Files: {details['file_count']}")
        if "class_count" in details:
            print(f"   Base Classes: {details['class_count']}")
        if details.get("data_structure_count"):
            print(f"   Data Structure Files: {details['data_structure_count']}")
            print(f"   Data Structures: {len(details.get('data_structure_names', []))}")
        
        file_paths = details.get("file_paths", [])
        if file_paths:
            print("\n   Base Class File Paths:")
            for fp in file_paths[:5]:
                print(f"   - {fp}")
            if len(file_paths) > 5:
                print(f"   ... and {len(file_paths) - 5} more")
        
        class_names = details.get("class_names", [])
        if class_names:
            print("\n   Base Classes:")
            for cn in class_names[:10]:
                print(f"   - {cn}")
            if len(class_names) > 10:
                print(f"   ... and {len(class_names) - 10} more")
        
        ds_names = details.get("data_structure_names", [])
        if ds_names:
            print("\n   Data Flow Data Structures:")
            for dn in ds_names[:10]:
                print(f"   - {dn}")
            if len(ds_names) > 10:
                print(f"   ... and {len(ds_names) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Check base classes state"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_CLASSES_FILE,
        help="Base classes file to check"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include detailed class list and syntax check results"
    )
    
    args = parser.parse_args()
    
    result = inspect_state(args.input)
    
    # In verbose mode, include raw base_classes data
    if args.verbose and result.get("type") == "update":
        base_classes_data = load_json(args.input)
        if base_classes_data:
            result["base_classes"] = base_classes_data.get("base_classes", [])
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 50)
        print("BASE CLASSES CHECK")
        print("=" * 50)
        print_state(result)
    
    # Return exit code based on state
    if result["type"] == "error":
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
