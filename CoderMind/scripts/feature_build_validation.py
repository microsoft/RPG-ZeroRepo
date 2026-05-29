#!/usr/bin/env python3
"""Validate feature_spec.json (input) and feature_build.json (output) for /cmind.feature_build command.

This script checks:
1. Input file: .cmind/data/feature_spec.json
   - File existence
   - Required fields: meta, background_and_overview, functional_requirements
   - Fields must exist and not be empty

2. Output file: .cmind/data/feature_build.json
   - File existence
   - Fields status: repository_name, repository_purpose, repository_specification, feature_tree

Output:
- Status messages are printed to stderr (user-friendly progress info)
- JSON result is printed to stdout (for agent parsing)

Exit codes:
- 0: Input file is valid (output file status is informational only)
- 1: Input file has errors (missing or invalid)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from common.paths import FEATURE_SPEC_FILE, FEATURE_BUILD_FILE

# File paths
INPUT_FILE = FEATURE_SPEC_FILE
OUTPUT_FILE = FEATURE_BUILD_FILE

# Required fields for input file
INPUT_REQUIRED_FIELDS = [
    "meta",
    "repository_name",
    "repository_purpose",
    "background_and_overview",
    "functional_requirements",
    "non_functional_requirements",
]

# Fields to check in output file
OUTPUT_CHECK_FIELDS = [
    "repository_name",
    "repository_purpose",
    "repository_specification",
    "feature_tree",
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


def count_nodes(nodes: List[Dict[str, Any]]) -> int:
    """Recursively count all nodes in the tree."""
    count = 0
    for node in nodes:
        count += 1
        if "children" in node and isinstance(node["children"], list):
            count += count_nodes(node["children"])
    return count


def validate_input_file() -> Dict[str, Any]:
    """Validate the input file (.cmind/data/feature_spec.json)."""
    result = {
        "valid": False,
        "exists": False,
        "errors": [],
        "fields": {field: False for field in INPUT_REQUIRED_FIELDS},
        "meta": None,
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
        if is_field_valid(data, field):
            result["fields"][field] = True
        else:
            all_fields_valid = False
            missing_fields.append(field)
            if field not in data:
                result["errors"].append(f"Missing required field: {field}")
            else:
                result["errors"].append(f"Field is empty: {field}")

    if "meta" in data and isinstance(data["meta"], dict):
        meta_dict = data["meta"]
        result["meta"] = {
            "repository_name": data.get("repository_name"),
            "repository_purpose": data.get("repository_purpose"),
            "generated_at": meta_dict.get("generated_at"),
            "source_documents": meta_dict.get("source_documents"),
            "project_types": meta_dict.get("project_types"),
            "project_notes": meta_dict.get("project_notes"),
        }

        # Validate project_types / project_notes. Soft-fail with
        # an error entry so the operator regenerates feature_spec, but
        # don't prevent legacy specs (without these fields) from running
        # through downstream stages — they will simply miss the project-
        # specific prompt branches.
        try:
            from common.project_types import validate_project_types
            types, notes = validate_project_types(meta_dict)
            result["meta"]["project_types"] = types
            result["meta"]["project_notes"] = notes
        except Exception as exc:
            # Only treat as error when the field is present but invalid;
            # missing field is treated as a warning so legacy spec files
            # still load.
            if "project_types" in meta_dict or "project_notes" in meta_dict:
                result["errors"].append(f"meta validation: {exc}")
                all_fields_valid = False
            else:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "feature_spec.meta is missing project_types/project_notes "
                    "; downstream prompts will lack project-type context"
                )

    if result["fields"]["functional_requirements"]:
        total_nodes = count_nodes(data.get("functional_requirements", []))
        result["functional_requirements_count"] = total_nodes

    if all_fields_valid:
        result["valid"] = True
        print_status(
            f"[INPUT] [OK] Valid ({result.get('functional_requirements_count', 0)} nodes)"
        )
    else:
        print_status(f"[INPUT] [FAIL] Missing: {', '.join(missing_fields)}")

    return result


def check_output_file() -> Dict[str, Any]:
    """Check the output file (.cmind/data/feature_build.json) status."""
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

    result["has_content"] = True

    print_status("[OUTPUT] [OK] Exists")
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
