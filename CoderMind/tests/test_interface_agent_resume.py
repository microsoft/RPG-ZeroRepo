import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from func_design.interface_agent import InterfaceOrchestrator
import func_design.func_designer as func_designer


def test_subtree_complete_when_no_features_and_all_files_present():
    file_nodes = [{"path": "src/config.go", "feature_paths": []}]
    file_container = {"src/config.go": {"units": [], "file_code": ""}}

    assert InterfaceOrchestrator._subtree_interfaces_complete(
        file_nodes, file_container,
    ) is True


def test_subtree_incomplete_when_no_features_and_file_missing():
    file_nodes = [{"path": "src/config.go", "feature_paths": []}]

    assert InterfaceOrchestrator._subtree_interfaces_complete(
        file_nodes, {},
    ) is False


def test_subtree_complete_when_no_file_nodes():
    assert InterfaceOrchestrator._subtree_interfaces_complete([], {}) is True


def test_subtree_complete_ignores_new_features_sentinel():
    file_nodes = [{"path": "src/lib.go", "feature_paths": ["Lib/init"]}]
    file_container = {
        "src/lib.go": {
            "units": ["function Init"],
            "units_to_features": {"function Init": ["Lib/init"]},
        },
        "__new_features__": [{"feature_path": "Lib/glue"}],
    }

    assert InterfaceOrchestrator._subtree_interfaces_complete(
        file_nodes, file_container,
    ) is True


def _designer_with_metadata(skeleton, data_flow=None, base_classes=None):
    designer = func_designer.FuncDesigner(max_interface_iterations=1)
    designer.skeleton = skeleton
    designer.data_flow = data_flow or {"data_flow": []}
    designer.base_classes = base_classes or {"base_classes": []}
    designer.repo_info = "repo"
    return designer


def _capture_orchestrator_language(monkeypatch, tmp_path):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured["target_language"] = kwargs.get("target_language")

        def design_all_interfaces(self, **kwargs):
            return {"success": True, "subtrees": {}}

    monkeypatch.setattr(func_designer, "InterfaceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(func_designer, "OUTPUT_INTERFACES", tmp_path / "interfaces.json")
    return captured


def test_run_interface_phase_extracts_language_from_skeleton(monkeypatch, tmp_path):
    captured = _capture_orchestrator_language(monkeypatch, tmp_path)
    designer = _designer_with_metadata({
        "meta": {"primary_language": "go", "target_languages": ["go"]},
        "root": {"type": "directory", "children": []},
    })

    result = designer.run_interface_phase()

    assert result["success"] is True
    assert captured["target_language"] == "go"


def test_run_interface_phase_falls_back_to_data_flow_language(monkeypatch, tmp_path):
    captured = _capture_orchestrator_language(monkeypatch, tmp_path)
    designer = _designer_with_metadata(
        {"root": {"type": "directory", "children": []}},
        data_flow={
            "data_flow": [],
            "meta": {"primary_language": "rust", "target_languages": ["rust"]},
        },
    )

    result = designer.run_interface_phase()

    assert result["success"] is True
    assert captured["target_language"] == "rust"
