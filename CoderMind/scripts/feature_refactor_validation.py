#!/usr/bin/env python3
"""Validate feature_build.json (input) and feature_tree.json (output) for /cmind.feature_refactor command.

This script checks:
1. Input file: .cmind/data/feature_build.json
   - File existence
   - Required fields: repository_name, repository_purpose, feature_tree
   - Fields must exist and not be empty

2. Output file: .cmind/data/feature_tree.json
   - File existence
   - Fields status: repository_name, repository_purpose, feature_tree, components

Output:
- Status messages are printed to stderr (user-friendly progress info)
- JSON result is printed to stdout (for agent parsing)

Exit codes:
- 0: Input file is valid (output file status is informational only)
- 1: Input file has errors (missing or invalid)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

from common.paths import FEATURE_BUILD_FILE, FEATURE_TREE_FILE

# File paths
INPUT_FILE = FEATURE_BUILD_FILE
OUTPUT_FILE = FEATURE_TREE_FILE

# Required fields for input file
INPUT_REQUIRED_FIELDS = ["repository_name", "repository_purpose", "feature_tree"]

# Fields to check in output file
OUTPUT_CHECK_FIELDS = [
    "repository_name",
    "repository_purpose",
    "feature_tree",
    "components",
]


def print_status(message: str) -> None:
    """Print status message to stderr to keep stdout clean for JSON."""
    print(message, file=sys.stderr)


def load_json(path: Path) -> Dict[str, Any] | None:
    """Load JSON file and return data if valid, None otherwise."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError as e:
        print_status(f"  [FAIL] JSON parse error: {e}")
    except Exception as e:
        print_status(f"  [FAIL] Failed to read file: {e}")
    return None


def is_field_valid(data: Dict[str, Any], field: str) -> bool:
    """Check if a field exists and is not empty."""
    if field not in data:
        return False

    value = data[field]

    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False

    return True


def count_feature_tree_leaves(tree: Dict[str, Any]) -> int:
    """Recursively count all leaf nodes in the feature tree."""
    count = 0
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, dict):
                if not value:
                    count += 1
                else:
                    count += count_feature_tree_leaves(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
    return count


def validate_input_file() -> Dict[str, Any]:
    """Validate the input file (.cmind/data/feature_build.json)."""
    result = {
        "valid": False,
        "exists": False,
        "errors": [],
    }

    if not INPUT_FILE.exists():
        result["errors"].append(f"Input file not found: {INPUT_FILE}")
        print_status(f"[INPUT] [FAIL] {INPUT_FILE} not found")
        return result

    result["exists"] = True

    data = load_json(INPUT_FILE)
    if data is None:
        result["errors"].append("Failed to parse JSON or file is empty")
        print_status("[INPUT] [FAIL] Invalid JSON")
        return result

    all_fields_valid = True
    missing_fields = []
    for field in INPUT_REQUIRED_FIELDS:
        if not is_field_valid(data, field):
            all_fields_valid = False
            missing_fields.append(field)
            if field not in data:
                result["errors"].append(f"Missing required field: {field}")
            else:
                result["errors"].append(f"Field is empty: {field}")

    if is_field_valid(data, "feature_tree"):
        leaf_count = count_feature_tree_leaves(data.get("feature_tree", {}))
        result["feature_tree_leaf_count"] = leaf_count

    if all_fields_valid:
        result["valid"] = True
        print_status(
            f"[INPUT] [OK] Valid ({result.get('feature_tree_leaf_count', 0)} leaves)"
        )
    else:
        print_status(f"[INPUT] [FAIL] Missing: {', '.join(missing_fields)}")

    return result


def check_output_file() -> Dict[str, Any]:
    """Check the output file (.cmind/data/feature_tree.json) status."""
    result = {
        "exists": False,
        "has_content": False,
        "errors": [],
    }

    if not OUTPUT_FILE.exists():
        print_status("[OUTPUT] [-] Not exists (will create)")
        return result

    result["exists"] = True

    data = load_json(OUTPUT_FILE)
    if data is None:
        result["errors"].append("Invalid JSON or empty file")
        print_status("[OUTPUT] [-] Exists but invalid JSON")
        return result

    # Check if output has valid content (components field with content)
    if is_field_valid(data, "components"):
        result["has_content"] = True
        print_status("[OUTPUT] [OK] Exists with content")
    else:
        print_status("[OUTPUT] [-] Exists but no valid content")

    return result


def main() -> None:
    input_result = validate_input_file()
    output_result = check_output_file()

    # Build simplified result (validation status only, no file content)
    result = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input": {
            "valid": input_result["valid"],
            "exists": input_result["exists"],
            "errors": input_result["errors"],
        },
        "output": {
            "exists": output_result["exists"],
            "has_content": output_result["has_content"],
            "errors": output_result["errors"],
        },
    }

    if not input_result["valid"]:
        result["status"] = "error"
        result["message"] = "Input invalid"
        result["action"] = "none"
    elif output_result["exists"] and output_result["has_content"]:
        result["status"] = "ready"
        result["message"] = "Output exists"
        result["action"] = "overwrite_or_skip"
    else:
        result["status"] = "ready"
        result["message"] = "Ready to create"
        result["action"] = "create"

    print_status(f"[RESULT] status={result['status']}, action={result['action']}")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["status"] == "error":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
