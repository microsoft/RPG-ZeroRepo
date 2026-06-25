#!/usr/bin/env python3
"""Check Tasks Script - Validation for tasks.json.

Validates the tasks.json file and determines the execution state:
- "error": Input file missing or invalid
- "init": No tasks.json exists or it's invalid
- "warning": tasks.json exists but has unit mismatches with interfaces
- "update": Valid tasks.json exists and is consistent

Cross-validates units between interfaces.json and tasks.json.

Returns JSON with validation status and statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

# Import centralized paths
from common.paths import INTERFACES_FILE as INPUT_FILE, TASKS_FILE as OUTPUT_FILE


def validate_interfaces(interfaces_path: Path) -> Tuple[bool, List[str]]:
    """Validate that interfaces.json exists and is valid."""
    errors = []
    
    if not interfaces_path.exists():
        errors.append(f"Input file not found: {interfaces_path}")
        return False, errors
    
    try:
        with open(interfaces_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors
    
    # Check required structure - support both "subtrees" (new) and "components" (old)
    containers = data.get("subtrees", data.get("components", {}))
    if not containers:
        errors.append("Missing required field: 'subtrees' or 'components'")
        return False, errors
    
    if not isinstance(containers, dict):
        errors.append("'subtrees'/'components' must be an object")
        return False, errors
    
    return True, errors


def get_all_units_from_interfaces(interfaces_path: Path) -> Set[str]:
    """Extract all unit identifiers from interfaces.json (file_path::unit_name)."""
    with open(interfaces_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    units = set()
    # Support both "subtrees" (new format) and "components" (old format)
    containers = data.get("subtrees", data.get("components", {}))
    for comp_name, comp_data in containers.items():
        # Support both "interfaces" (new format) and "files" (old format)
        file_container = comp_data.get("interfaces", comp_data.get("files", {}))
        for file_path, file_data in file_container.items():
            for unit_name in file_data.get("units", []):
                units.add(f"{file_path}::{unit_name}")
    return units


def get_all_units_from_tasks(tasks_path: Path) -> Set[str]:
    """Extract all unit identifiers from tasks.json (file_path::unit_name).
    
    Supports both formats:
    - planned_tasks_dict: {component: {file_path: [task, ...]}}
    - batches: [{batch_id, units: [{file_path, unit_name}, ...]}]
    """
    with open(tasks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    units = set()

    # ``plan_tasks.py`` emits auxiliary scaffolding tasks (README,
    # requirements.txt, integration tests, cross-module wiring, UI
    # polish, main entry, comprehensive tests) using synthetic file-path
    # placeholders wrapped in angle-brackets such as ``<README>`` or
    # ``<INTEGRATION_TEST>_Foo``. These are by design not present in
    # ``interfaces.json``; skipping them prevents false-positive
    # ``missing_in_interfaces`` warnings from the cross-validation step.
    def _is_synthetic(file_path: str) -> bool:
        return file_path.startswith("<") and ">" in file_path

    # Support planned_tasks_dict format
    if "planned_tasks_dict" in data:
        for component_name, files_dict in data["planned_tasks_dict"].items():
            for file_path, task_list in files_dict.items():
                if _is_synthetic(file_path):
                    continue
                for task in task_list:
                    # units_key contains the unit names
                    for unit_name in task.get("units_key", []):
                        units.add(f"{file_path}::{unit_name}")
    # Support batches format (legacy)
    elif "batches" in data:
        for batch in data.get("batches", []):
            for unit in batch.get("units", []):
                file_path = unit.get("file_path", "")
                unit_name = unit.get("unit_name", "")
                if file_path and unit_name and not _is_synthetic(file_path):
                    units.add(f"{file_path}::{unit_name}")

    return units


def cross_validate_units(interfaces_units: Set[str], tasks_units: Set[str]) -> Dict[str, Any]:
    """Cross-validate units between interfaces and tasks.
    
    Returns dict with:
        - in_interfaces_not_tasks: units in interfaces but not in tasks
        - in_tasks_not_interfaces: units in tasks but not in interfaces
        - matched_count: number of matched units
        - warnings: list of warning messages
    """
    in_interfaces_not_tasks = interfaces_units - tasks_units
    in_tasks_not_interfaces = tasks_units - interfaces_units
    matched = interfaces_units & tasks_units
    
    warnings = []
    
    for unit in sorted(in_interfaces_not_tasks):
        warnings.append({
            "type": "missing_in_tasks",
            "unit": unit,
            "message": f"Unit '{unit}' exists in interfaces.json but not in tasks.json"
        })
    
    for unit in sorted(in_tasks_not_interfaces):
        warnings.append({
            "type": "missing_in_interfaces",
            "unit": unit,
            "message": f"Unit '{unit}' exists in tasks.json but not in interfaces.json"
        })
    
    return {
        "in_interfaces_not_tasks": sorted(list(in_interfaces_not_tasks)),
        "in_tasks_not_interfaces": sorted(list(in_tasks_not_interfaces)),
        "matched_count": len(matched),
        "interfaces_unit_count": len(interfaces_units),
        "tasks_unit_count": len(tasks_units),
        "warnings": warnings,
        "is_consistent": len(warnings) == 0
    }


def validate_tasks(tasks_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate tasks.json structure and content.
    
    Supports both formats:
    - planned_tasks_dict: {component: {file_path: [task, ...]}}
    - batches: [{batch_id, units: [{file_path, unit_name}, ...]}]
    """
    errors = []
    stats = {
        "total_tasks": 0,
        "total_units": 0,
        "files_touched": 0,
        "components": []  # Use list instead of set for JSON serialization
    }
    components_set = set()  # Track components internally
    all_files = set()
    total_units = 0
    
    if not tasks_path.exists():
        errors.append(f"Output file not found: {tasks_path}")
        return False, errors, stats
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors, stats
    
    # Support planned_tasks_dict format (primary)
    if "planned_tasks_dict" in data:
        planned_tasks = data["planned_tasks_dict"]
        if not isinstance(planned_tasks, dict):
            errors.append("'planned_tasks_dict' must be an object")
            return False, errors, stats
        
        stats["total_tasks"] = len(planned_tasks)
        
        for component_name, files_dict in planned_tasks.items():
            components_set.add(component_name)
            
            if not isinstance(files_dict, dict):
                errors.append(f"Component '{component_name}' must contain a files object")
                continue
            
            for file_path, task_list in files_dict.items():
                all_files.add(file_path)
                
                if not isinstance(task_list, list):
                    errors.append(f"Tasks for '{file_path}' must be a list")
                    continue
                
                for i, task in enumerate(task_list):
                    if not isinstance(task, dict):
                        errors.append(f"Task {i+1} in '{file_path}' must be an object")
                        continue
                    
                    # Check required fields in task
                    if "task_id" not in task:
                        errors.append(f"Task {i+1} in '{file_path}' missing 'task_id'")
                    
                    if "units_key" not in task:
                        errors.append(f"Task {i+1} in '{file_path}' missing 'units_key'")
                    else:
                        units_key = task.get("units_key", [])
                        if isinstance(units_key, list):
                            total_units += len(units_key)
        
        stats["total_units"] = total_units
    
    # Support batches format (legacy)
    elif "batches" in data:
        batches = data.get("batches", [])
        if not isinstance(batches, list):
            errors.append("'batches' must be a list")
            return False, errors, stats
        
        stats["total_tasks"] = len(batches)
        stats["total_units"] = data.get("total_units", 0)
        
        for i, batch in enumerate(batches):
            if not isinstance(batch, dict):
                errors.append(f"Batch {i+1} must be an object")
                continue
            
            if "batch_id" not in batch:
                errors.append(f"Batch {i+1} missing 'batch_id'")
            
            if "units" not in batch:
                errors.append(f"Batch {i+1} missing 'units'")
                continue
            
            units = batch.get("units", [])
            if not isinstance(units, list):
                errors.append(f"Batch {i+1} 'units' must be a list")
                continue
            
            for j, unit in enumerate(units):
                if not isinstance(unit, dict):
                    errors.append(f"Batch {i+1}, unit {j+1} must be an object")
                    continue
                
                if "unit_name" not in unit:
                    errors.append(f"Batch {i+1}, unit {j+1} missing 'unit_name'")
                
                if "file_path" not in unit:
                    errors.append(f"Batch {i+1}, unit {j+1} missing 'file_path'")
                else:
                    all_files.add(unit["file_path"])
                
                if "component" in unit:
                    components_set.add(unit["component"])
            
            if "files" in batch:
                files = batch.get("files", [])
                if isinstance(files, list):
                    all_files.update(files)
    else:
        errors.append("Missing required field: 'planned_tasks_dict' or 'batches'")
        return False, errors, stats
    
    stats["files_touched"] = len(all_files)
    stats["components"] = sorted(list(components_set))
    
    is_valid = len(errors) == 0
    return is_valid, errors, stats


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
    
    # Check input (interfaces.json)
    if not result["input_exists"]:
        result["type"] = "error"
        result["message"] = f"Input file not found: {input_path}. Please run /cmind.design_interfaces first."
        return result
    
    input_valid, input_errors = validate_interfaces(input_path)
    result["input_valid"] = input_valid
    
    if not input_valid:
        result["type"] = "error"
        result["message"] = "Invalid interfaces.json"
        result["validation_errors"] = input_errors
        return result
    
    # Get interfaces units for cross-validation
    interfaces_units = get_all_units_from_interfaces(input_path)
    result["input_statistics"] = {
        "total_units": len(interfaces_units)
    }
    
    # Check output (tasks.json)
    if not result["output_exists"]:
        result["type"] = "init"
        result["message"] = "Ready to plan tasks. No existing tasks.json found."
        return result
    
    output_valid, output_errors, stats = validate_tasks(output_path)
    result["output_valid"] = output_valid
    result["stats"] = stats
    
    if not output_valid:
        result["type"] = "init"
        result["message"] = "Existing tasks.json is invalid. Will regenerate."
        result["validation_errors"] = output_errors
        return result
    
    # Cross-validate units
    tasks_units = get_all_units_from_tasks(output_path)
    cross_validation = cross_validate_units(interfaces_units, tasks_units)
    result["cross_validation"] = cross_validation
    
    # Determine type based on cross-validation
    if not cross_validation["is_consistent"]:
        warning_count = len(cross_validation["warnings"])
        result["type"] = "warning"
        result["message"] = (
            f"tasks.json exists but has {warning_count} unit mismatches with interfaces. "
            "This is a cross-stage contract violation; plan.py will rebuild tasks and downstream stages."
        )
    else:
        result["type"] = "update"
        result["message"] = f"Valid tasks.json exists with {stats['total_tasks']} tasks and {stats['total_units']} units."
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check tasks.json validity and state"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input interfaces.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output tasks.json file to check"
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
