from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from func_design.interface_agent import GlobalInterfaceRegistry, InterfaceOrchestrator

_SPEC = importlib.util.spec_from_file_location(
    "design_interfaces_script",
    _SCRIPTS / "design_interfaces.py",
)
assert _SPEC is not None and _SPEC.loader is not None
design_interfaces = importlib.util.module_from_spec(_SPEC)
sys.modules["design_interfaces_script"] = design_interfaces
_SPEC.loader.exec_module(design_interfaces)


def test_file_coverage_success_requires_all_features() -> None:
    coverage = InterfaceOrchestrator._new_coverage_status()
    InterfaceOrchestrator._record_file_coverage(
        coverage_status=coverage,
        subtree_name="Task Store",
        file_node={
            "path": "src/store.ts",
            "feature_paths": ["Task Store/Add", "Task Store/List"],
        },
        result={
            "units": ["class TaskStore"],
            "units_to_features": {
                "class TaskStore": ["Task Store/Add", "Task Store/List"],
            },
        },
    )

    assert coverage["expected_files"] == 1
    assert coverage["successful_files"] == 1
    assert coverage["covered_features"] == 2
    assert coverage["missing_features"] == 0
    assert coverage["issues"] == []


def test_file_coverage_records_partial_result() -> None:
    coverage = InterfaceOrchestrator._new_coverage_status()
    InterfaceOrchestrator._record_file_coverage(
        coverage_status=coverage,
        subtree_name="Task Store",
        file_node={
            "path": "src/store.rs",
            "feature_paths": ["Task Store/Add", "Task Store/List"],
        },
        result={
            "units": ["struct TaskStore"],
            "units_to_features": {"struct TaskStore": ["Task Store/Add"]},
        },
    )

    assert coverage["successful_files"] == 0
    assert coverage["covered_features"] == 1
    assert coverage["missing_features"] == 1
    assert coverage["failed_files"] == ["src/store.rs"]
    assert coverage["issues"] == [
        {
            "subtree": "Task Store",
            "file_path": "src/store.rs",
            "reason": "missing features",
            "missing_features": ["Task Store/List"],
        }
    ]


def test_build_result_marks_coverage_issues_unsuccessful() -> None:
    coverage = InterfaceOrchestrator._new_coverage_status()
    InterfaceOrchestrator._record_missing_subtree(coverage, "CLI")
    orchestrator = InterfaceOrchestrator(
        llm_client=object(),
        target_language="typescript",
    )

    result = orchestrator._build_result({}, ["CLI"], {}, coverage)

    assert result["success"] is False
    assert result["coverage"]["missing_subtrees"] == ["CLI"]


def test_design_interfaces_main_fails_on_incomplete_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skeleton_path = tmp_path / "skeleton.json"
    data_flow_path = tmp_path / "data_flow.json"
    base_classes_path = tmp_path / "base_classes.json"
    output_path = tmp_path / "interfaces.json"
    skeleton_path.write_text(json.dumps({"root": {"type": "directory", "children": []}}))
    data_flow_path.write_text(json.dumps({}))
    base_classes_path.write_text(json.dumps({}))

    class FakeDesigner:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, skeleton, data_flow, base_classes):
            return {
                "success": False,
                "subtrees": {},
                "subtree_order": [],
                "coverage": {
                    "issues": [
                        {
                            "subtree": "CLI",
                            "file_path": "src/main.ts",
                            "reason": "no units",
                            "missing_features": ["CLI/Run"],
                        }
                    ]
                },
            }

        def print_summary(self, result):
            pass

    monkeypatch.setattr(design_interfaces, "InterfaceDesigner", FakeDesigner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "design_interfaces.py",
            "--skeleton",
            str(skeleton_path),
            "--data-flow",
            str(data_flow_path),
            "--base-classes",
            str(base_classes_path),
            "--output",
            str(output_path),
            "--no-trajectory",
        ],
    )

    assert design_interfaces.main() == 1
    saved = json.loads(output_path.read_text())
    assert saved["success"] is False


def test_restore_completed_subtrees_reuses_only_complete_prefix(tmp_path: Path) -> None:
    output_path = tmp_path / "interfaces.json"
    output_path.write_text(json.dumps({
        "subtrees": {
            "Core": {
                "files_order": ["core.go"],
                "interfaces": {
                    "core.go": {
                        "file_code": "package core\n\ntype Core struct{}\n",
                        "units": ["struct Core"],
                        "units_to_features": {"struct Core": ["Core/run"]},
                    }
                },
            },
            "Store": {
                "files_order": ["store.go"],
                "interfaces": {},
            },
        }
    }))
    skeleton = {
        "root": {
            "type": "directory",
            "children": [
                {"type": "file", "path": "core.go", "feature_paths": ["Core/run"]},
                {"type": "file", "path": "store.go", "feature_paths": ["Store/load"]},
            ],
        }
    }
    orchestrator = InterfaceOrchestrator(
        llm_client=object(),
        output_path=str(output_path),
        target_language="go",
    )
    all_interfaces = {}
    implemented_subtrees = {}
    coverage = InterfaceOrchestrator._new_coverage_status()
    registry = GlobalInterfaceRegistry()

    restored = orchestrator._restore_completed_subtrees(
        skeleton=skeleton,
        subtree_order=["Core", "Store"],
        all_interfaces=all_interfaces,
        implemented_subtrees=implemented_subtrees,
        coverage_status=coverage,
        global_registry=registry,
    )

    assert restored == {"Core"}
    assert list(all_interfaces) == ["Core"]
    assert implemented_subtrees["Core"][0]["path"] == "core.go"
    assert coverage["expected_features"] == 1
    assert coverage["covered_features"] == 1


def test_subtree_complete_allows_cross_file_feature_mapping() -> None:
    file_nodes = [
        {"path": "cmd/main.go", "feature_paths": ["CLI/run"]},
        {"path": "cmd/usage.go", "feature_paths": ["CLI/help"]},
    ]
    file_container = {
        "cmd/main.go": {
            "units": ["function Run"],
            "units_to_features": {"function Run": ["CLI/run", "CLI/help"]},
        },
        "cmd/usage.go": {"units": [], "units_to_features": {}},
    }

    assert InterfaceOrchestrator._subtree_interfaces_complete(
        file_nodes,
        file_container,
    )