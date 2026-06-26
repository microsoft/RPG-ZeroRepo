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
from decoder_lang import get_backend as get_backend_for

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


def _callable_by_prefix(unit_name: str) -> bool:
    return unit_name.split(" ", 1)[0] in {"function", "method", "class"}


def test_global_review_reconciles_retained_orphans() -> None:
    # A single isolated callable unit (no edges) that the orphan review
    # explicitly RETAINED must not fail the verdict.
    interfaces_data = {
        "subtrees": {
            "App": {
                "interfaces": {
                    "src/app.py": {
                        "units": ["function main"],
                        "units_to_features": {"function main": ["App/run"]},
                    }
                }
            }
        }
    }
    global_review = {
        "feature_orphans_count": 1,
        "orphan_units_count": 1,
        "blocking_unapplied_fixes_count": 0,
        "passed": False,
    }

    design_interfaces._finalize_global_review_verdict(
        global_review=global_review,
        interfaces_data=interfaces_data,
        enhanced_data_flow={"invocation_edges": []},
        entry_points=[],
        is_callable=_callable_by_prefix,
        retained_keys={"src/app.py::function main"},
    )

    assert global_review["passed"] is True
    assert global_review["orphan_units_count"] == 0
    assert global_review["feature_orphans_count"] == 0


def test_global_review_keeps_unresolved_orphans_failing() -> None:
    # Two isolated callable units; only one is retained, so the other
    # remains an orphan and the verdict stays failing.
    interfaces_data = {
        "subtrees": {
            "App": {
                "interfaces": {
                    "src/app.py": {
                        "units": ["function main", "function unused"],
                        "units_to_features": {
                            "function main": ["App/run"],
                            "function unused": ["App/unused"],
                        },
                    }
                }
            }
        }
    }
    global_review = {
        "feature_orphans_count": 2,
        "orphan_units_count": 2,
        "blocking_unapplied_fixes_count": 0,
        "passed": False,
    }

    design_interfaces._finalize_global_review_verdict(
        global_review=global_review,
        interfaces_data=interfaces_data,
        enhanced_data_flow={"invocation_edges": []},
        entry_points=[],
        is_callable=_callable_by_prefix,
        retained_keys={"src/app.py::function main"},
    )

    assert global_review["passed"] is False
    assert global_review["orphan_units_count"] == 1
    assert global_review["orphan_units_count"] == 1
    assert global_review["feature_orphans_count"] == 1
    assert global_review["unresolved_orphan_units"] == ["src/app.py::function unused"]


def _store_skeleton_and_interfaces():
    """A skeleton feature missing from interfaces, plus its file block."""
    skeleton = {
        "root": {
            "type": "directory",
            "name": "root",
            "path": ".",
            "children": [
                {
                    "type": "file",
                    "name": "schema.js",
                    "path": "src/store/schema.js",
                    "feature_paths": [
                        "Data/schema/define store structure",
                        "Data/schema/define todo object schema",
                    ],
                }
            ],
        }
    }
    interfaces = {
        "subtrees": {
            "Data": {
                "interfaces": {
                    "src/store/schema.js": {
                        "units": ["function parseTodoRecord"],
                        "units_to_features": {
                            "function parseTodoRecord": [
                                "Data/schema/define store structure"
                            ]
                        },
                    }
                }
            }
        }
    }
    return skeleton, interfaces


def test_backfill_attributes_missing_feature() -> None:
    skeleton, interfaces = _store_skeleton_and_interfaces()
    audit = design_interfaces.backfill_uncovered_features(skeleton, interfaces)

    # The orphan feature is attributed to the file's existing unit.
    assert len(audit["backfilled"]) == 1
    assert audit["backfilled"][0]["feature"] == "Data/schema/define todo object schema"
    assert audit["backfilled"][0]["file_path"] == "src/store/schema.js"
    assert audit["unbackfilled"] == []

    covered = design_interfaces._collect_interface_features(interfaces)
    assert "Data/schema/define todo object schema" in covered
    # Coverage now equals the skeleton (the bench consistency gate passes).
    assert design_interfaces.collect_skeleton_features(skeleton) - covered == set()


def test_backfill_noop_when_fully_covered() -> None:
    skeleton, interfaces = _store_skeleton_and_interfaces()
    # Pre-attribute the missing feature so nothing is uncovered.
    u2f = interfaces["subtrees"]["Data"]["interfaces"]["src/store/schema.js"]["units_to_features"]
    u2f["function parseTodoRecord"].append("Data/schema/define todo object schema")

    audit = design_interfaces.backfill_uncovered_features(skeleton, interfaces)
    assert audit["backfilled"] == []
    assert audit["unbackfilled"] == []


def test_backfill_reports_unbackfillable_when_file_absent() -> None:
    skeleton, interfaces = _store_skeleton_and_interfaces()
    # Remove the interface file block so the feature has nowhere to attach.
    interfaces["subtrees"]["Data"]["interfaces"] = {}

    audit = design_interfaces.backfill_uncovered_features(skeleton, interfaces)
    assert audit["backfilled"] == []
    reasons = {item["reason"] for item in audit["unbackfilled"]}
    assert reasons == {"file not in interfaces"}


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


# ---------------------------------------------------------------------------
# Global interface review — multilingual (G4 stage 1)
# ---------------------------------------------------------------------------

class _StubLLM:
    """Minimal LLMClient stand-in (review fixes are applied directly)."""


def _make_reviewer(language: str):
    from func_design.interface_review import InterfaceReviewer

    return InterfaceReviewer(llm_client=_StubLLM(), target_language=language)


def test_apply_fixes_add_dependency_is_language_agnostic() -> None:
    reviewer = _make_reviewer("go")
    enhanced_data_flow: dict = {"invocation_edges": []}
    registry = GlobalInterfaceRegistry(backend=get_backend_for("go"))

    stats = reviewer._apply_fixes(
        fixes=[{
            "action": "add_dependency",
            "file_path": "internal/cli.go",
            "unit_name": "function Run",
            "calls_to_add": [
                {"callee": "NewStore", "callee_file": "internal/store.go"},
            ],
        }],
        interfaces_data={"subtrees": {}},
        enhanced_data_flow=enhanced_data_flow,
        global_registry=registry,
    )

    assert stats["applied_edges"] == 1
    assert stats["unapplied"] == []
    assert enhanced_data_flow["invocation_edges"][0]["callee"] == "NewStore"


def test_apply_fixes_records_advisory_add_interface_for_non_python() -> None:
    reviewer = _make_reviewer("go")
    registry = GlobalInterfaceRegistry(backend=get_backend_for("go"))
    interfaces_data = {
        "subtrees": {
            "Core": {
                "interfaces": {
                    "internal/cli.go": {"units": [], "units_to_features": {}, "file_code": ""},
                }
            }
        }
    }

    stats = reviewer._apply_fixes(
        fixes=[{
            "action": "add_interface",
            "file_path": "internal/cli.go",
            "unit_name": "function Run",
            "signature": "func Run() error",
            "docstring": "Run the CLI.",
            "feature_path": "CLI/run",
        }],
        interfaces_data=interfaces_data,
        enhanced_data_flow={"invocation_edges": []},
        global_registry=registry,
        skeleton_features={"CLI/run"},
        rpg_features={"CLI/run"},
    )

    # add_interface stub synthesis is Python-only. For other languages the
    # request is recorded as an advisory manual follow-up rather than silently
    # dropped, so the review can still pass on structural grounds.
    assert stats["applied_fixes"] == 0
    assert stats["applied_edges"] == 0
    assert len(stats["unapplied"]) == 1
    advisory = stats["unapplied"][0]
    assert advisory["action"] == "add_interface"
    assert advisory["unit_name"] == "function Run"
    assert advisory["advisory"] is True
    assert advisory["manual_follow_up"] is True
    assert advisory["unsupported_for_language"] == "go"
    # No Go stub was injected into the interface file.
    cli = interfaces_data["subtrees"]["Core"]["interfaces"]["internal/cli.go"]
    assert cli["units"] == []


def test_apply_fixes_applies_add_interface_for_python() -> None:
    reviewer = _make_reviewer("python")
    registry = GlobalInterfaceRegistry(backend=get_backend_for("python"))
    interfaces_data = {
        "subtrees": {
            "Core": {
                "interfaces": {
                    "src/cli.py": {"units": [], "units_to_features": {}, "file_code": ""},
                }
            }
        }
    }

    stats = reviewer._apply_fixes(
        fixes=[{
            "action": "add_interface",
            "file_path": "src/cli.py",
            "unit_name": "function run",
            "signature": "def run() -> None:",
            "docstring": "Run the CLI.",
            "feature_path": "CLI/run",
        }],
        interfaces_data=interfaces_data,
        enhanced_data_flow={"invocation_edges": []},
        global_registry=registry,
        skeleton_features={"CLI/run"},
        rpg_features={"CLI/run"},
    )

    assert stats["applied_fixes"] == 1
    cli = interfaces_data["subtrees"]["Core"]["interfaces"]["src/cli.py"]
    assert "function run" in cli["units"]


def test_apply_fixes_can_create_python_interface_file_in_feature_subtree() -> None:
    reviewer = _make_reviewer("python")
    registry = GlobalInterfaceRegistry(backend=get_backend_for("python"))
    interfaces_data = {
        "subtrees": {
            "Todo Display": {
                "files_order": ["src/todo_web_app/views/todo_list.py"],
                "interfaces": {
                    "src/todo_web_app/views/todo_list.py": {
                        "units": ["function render_todo_items"],
                        "units_to_features": {
                            "function render_todo_items": [
                                "Todo Display/list rendering/items/render all items"
                            ]
                        },
                        "units_to_code": {},
                        "file_code": "",
                    }
                },
            }
        }
    }
    feature_path = "Todo Display/list rendering/page/render complete page"

    stats = reviewer._apply_fixes(
        fixes=[{
            "action": "add_interface",
            "file_path": "src/todo_web_app/views/render.py",
            "unit_name": "function render_todo_page",
            "signature": "def render_todo_page(todos: list[dict]) -> str:",
            "docstring": "Render the complete todo page.",
            "feature_path": feature_path,
            "incoming_calls_from": ["list_todos"],
        }],
        interfaces_data=interfaces_data,
        enhanced_data_flow={"invocation_edges": []},
        global_registry=registry,
        skeleton_features={feature_path},
        rpg_features={feature_path},
    )

    assert stats["applied_fixes"] == 1
    files = interfaces_data["subtrees"]["Todo Display"]["interfaces"]
    render_file = files["src/todo_web_app/views/render.py"]
    assert render_file["units"] == ["function render_todo_page"]
    assert render_file["units_to_features"]["function render_todo_page"] == [
        feature_path
    ]
    assert "def render_todo_page(todos: list[dict]) -> str:" in render_file["file_code"]
    assert "src/todo_web_app/views/render.py" in interfaces_data["subtrees"][
        "Todo Display"
    ]["files_order"]


def test_interface_orchestrator_writes_partial_resume_file(tmp_path) -> None:
    output_path = tmp_path / "interfaces.json"
    orchestrator = InterfaceOrchestrator(output_path=str(output_path))
    partial_result = {
        "subtrees": {"Core": {"interfaces": {}}},
        "subtree_order": ["Core", "UI"],
        "implemented_subtrees": {"Core": []},
        "coverage": {"issues": [{"subtree": "UI"}]},
        "success": False,
    }
    final_result = {
        "subtrees": {"Core": {"interfaces": {}}, "UI": {"interfaces": {}}},
        "subtree_order": ["Core", "UI"],
        "implemented_subtrees": {"Core": [], "UI": []},
        "coverage": {"issues": []},
        "success": True,
    }

    orchestrator._save_interfaces(partial_result, partial=True)

    assert not output_path.exists()
    assert Path(f"{output_path}.partial").exists()

    orchestrator._save_interfaces(final_result)

    assert output_path.exists()
    assert not Path(f"{output_path}.partial").exists()
