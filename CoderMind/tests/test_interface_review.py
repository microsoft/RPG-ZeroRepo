import json
import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from decoder_lang import get_backend
from func_design.interface_agent import GlobalInterfaceRegistry
from func_design.interface_review import (
    InterfaceReviewer,
    prune_orphan_interfaces,
    review_orphan_units,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate(self, prompt, purpose=None):
        self.prompts.append(prompt)
        return json.dumps(self.payload)

    def parse_json_block(self, response):
        return json.loads(response)


def _minimal_interfaces(unit_name="function main", file_path="main.go", code="package main\n\nfunc main() {}\n"):
    return {
        "subtree_order": ["core"],
        "subtrees": {
            "core": {
                "interfaces": {
                    file_path: {
                        "units": [unit_name],
                        "units_to_features": {unit_name: ["CLI/run"]},
                        "units_to_code": {unit_name: code},
                        "file_code": code,
                    }
                }
            }
        },
        "enhanced_data_flow": {
            "original_edges": [],
            "invocation_edges": [],
            "inheritance_edges": [],
            "reference_edges": [],
        },
    }


def _registry(language="go"):
    return GlobalInterfaceRegistry(backend=get_backend(language))


def test_non_python_add_interface_is_visible_advisory_and_does_not_mutate_interfaces():
    reviewer = InterfaceReviewer(target_language="go")
    interfaces_data = _minimal_interfaces()
    original = json.loads(json.dumps(interfaces_data))
    enhanced_data_flow = interfaces_data["enhanced_data_flow"]

    stats = reviewer._apply_fixes(
        fixes=[{
            "action": "add_interface",
            "file_path": "main.go",
            "unit_name": "function helper",
            "signature": "func helper() {}",
            "docstring": "Help main.",
            "feature_path": "CLI/run",
            "description": "Add missing helper.",
        }],
        interfaces_data=interfaces_data,
        enhanced_data_flow=enhanced_data_flow,
        global_registry=_registry("go"),
    )

    assert stats["requested_fixes"] == 1
    assert stats["applied_fixes"] == 0
    assert stats["applied_edges"] == 0
    assert len(stats["unapplied"]) == 1
    unapplied = stats["unapplied"][0]
    assert unapplied["action"] == "add_interface"
    assert unapplied["advisory"] is True
    assert unapplied["manual_follow_up"] is True
    assert unapplied["unsupported_for_language"] == "go"
    assert "Python-only" in unapplied["reason"]
    assert interfaces_data == original


def test_non_python_add_interface_advisory_does_not_block_review_and_fix():
    llm = FakeLLM({
        "entry_points": [{
            "file_path": "main.go",
            "unit_name": "main",
            "rationale": "CLI entry point",
        }],
        "orphan_modules": [],
        "missing_wiring": [],
        "type_mismatches": [],
        "orchestration_gaps": [],
        "recommended_fixes": [{
            "action": "add_interface",
            "file_path": "main.go",
            "unit_name": "function helper",
            "signature": "func helper() {}",
            "docstring": "Help main.",
            "feature_path": "CLI/run",
            "description": "Add missing helper.",
        }],
        "pass": False,
    })
    reviewer = InterfaceReviewer(llm_client=llm, target_language="go")
    interfaces_data = _minimal_interfaces()

    result = reviewer.review_and_fix(
        interfaces_data=interfaces_data,
        enhanced_data_flow=interfaces_data["enhanced_data_flow"],
        global_registry=_registry("go"),
        import_warnings=[],
        data_flow_edges=[],
        max_fix_iterations=1,
    )

    assert len(result["unapplied_fixes"]) == 1
    assert len(result["advisory_fixes"]) == 1
    assert len(result["blocking_unapplied_fixes"]) == 0
    assert result["passed"] is True


def test_unresolved_add_dependency_remains_blocking():
    llm = FakeLLM({
        "entry_points": [{
            "file_path": "main.go",
            "unit_name": "main",
            "rationale": "CLI entry point",
        }],
        "orphan_modules": [],
        "missing_wiring": [],
        "type_mismatches": [],
        "orchestration_gaps": [],
        "recommended_fixes": [{
            "action": "add_dependency",
            "file_path": "main.go",
            "unit_name": "function main",
            "description": "Wire missing callee.",
            "calls_to_add": [{"callee": "missing", "purpose": "missing"}],
        }],
        "pass": False,
    })
    reviewer = InterfaceReviewer(llm_client=llm, target_language="go")
    interfaces_data = _minimal_interfaces()

    result = reviewer.review_and_fix(
        interfaces_data=interfaces_data,
        enhanced_data_flow=interfaces_data["enhanced_data_flow"],
        global_registry=_registry("go"),
        import_warnings=[],
        data_flow_edges=[],
        max_fix_iterations=1,
    )

    assert len(result["unapplied_fixes"]) == 1
    assert len(result["advisory_fixes"]) == 0
    assert len(result["blocking_unapplied_fixes"]) == 1
    assert result["passed"] is False


def test_modify_interface_remains_advisory():
    llm = FakeLLM({
        "entry_points": [{
            "file_path": "main.go",
            "unit_name": "main",
            "rationale": "CLI entry point",
        }],
        "orphan_modules": [],
        "missing_wiring": [],
        "type_mismatches": [],
        "orchestration_gaps": [],
        "recommended_fixes": [{
            "action": "modify_interface",
            "file_path": "main.go",
            "unit_name": "function main",
            "description": "Manual architecture cleanup.",
        }],
        "pass": False,
    })
    reviewer = InterfaceReviewer(llm_client=llm, target_language="go")
    interfaces_data = _minimal_interfaces()

    result = reviewer.review_and_fix(
        interfaces_data=interfaces_data,
        enhanced_data_flow=interfaces_data["enhanced_data_flow"],
        global_registry=_registry("go"),
        import_warnings=[],
        data_flow_edges=[],
        max_fix_iterations=1,
    )

    assert len(result["unapplied_fixes"]) == 1
    assert len(result["advisory_fixes"]) == 1
    assert len(result["blocking_unapplied_fixes"]) == 0
    assert result["passed"] is True


def test_build_interface_summary_uses_backend_fence():
    reviewer = InterfaceReviewer(target_language="go")
    summary = reviewer._build_interface_summary(_minimal_interfaces(), _registry("go"))

    assert "```go" in summary
    assert "```python" not in summary


def test_review_orphan_units_uses_target_language_fence():
    llm = FakeLLM({
        "reviews": [{
            "unit_key": "main.go::function helper",
            "decision": "retain",
            "reason": "needed",
        }]
    })

    result = review_orphan_units(
        orphan_details=[{
            "unit_key": "main.go::function helper",
            "unit_name": "function helper",
            "file_path": "main.go",
            "subtree": "core",
            "code": "func helper() {}",
            "features": ["CLI/run"],
        }],
        repo_info="repo",
        subtree_interfaces={},
        llm_client=llm,
        target_language="go",
    )

    assert result.decisions["main.go::function helper"] == "retain"
    assert "```go" in llm.prompts[0]
    assert "```python" not in llm.prompts[0]


def test_review_orphan_units_defaults_to_python_fence():
    llm = FakeLLM({
        "reviews": [{
            "unit_key": "app.py::function helper",
            "decision": "retain",
            "reason": "needed",
        }]
    })

    review_orphan_units(
        orphan_details=[{
            "unit_key": "app.py::function helper",
            "unit_name": "function helper",
            "file_path": "app.py",
            "subtree": "core",
            "code": "def helper():\n    pass\n",
            "features": ["CLI/run"],
        }],
        repo_info="repo",
        subtree_interfaces={},
        llm_client=llm,
    )

    assert "```python" in llm.prompts[0]


def test_prune_orphan_interfaces_docstring_marks_outdated():
    assert "Outdated legacy helper" in prune_orphan_interfaces.__doc__
    assert "InterfacesStore.find_orphan_units" in prune_orphan_interfaces.__doc__


def test_extract_signature_summary_go_struct_registered_as_class():
    backend = get_backend("go")
    code = "package store\n\ntype Repository struct {\n\tdb string\n}\n"

    result = GlobalInterfaceRegistry._extract_signature_summary(
        code,
        unit_type="class",
        bare_name="Repository",
        backend=backend,
    )

    assert result != "Repository"
    assert "Repository" in result
    assert "struct" in result


def test_extract_signature_summary_go_interface_registered_as_class():
    backend = get_backend("go")
    code = "package store\n\ntype Storer interface {\n\tLoad() string\n}\n"

    result = GlobalInterfaceRegistry._extract_signature_summary(
        code,
        unit_type="class",
        bare_name="Storer",
        backend=backend,
    )

    assert result != "Storer"
    assert "Storer" in result
    assert "interface" in result


def test_extract_signature_summary_unknown_type_resolves_by_name():
    backend = get_backend("go")
    code = "package main\n\nfunc Connect() error { return nil }\n"

    result = GlobalInterfaceRegistry._extract_signature_summary(
        code,
        unit_type="unknown",
        bare_name="Connect",
        backend=backend,
    )

    assert result != "Connect"
    assert "Connect" in result
    assert "error" in result


def test_extract_signature_summary_unknown_name_returns_bare_name():
    backend = get_backend("go")
    code = "package main\n\nfunc Connect() error { return nil }\n"

    result = GlobalInterfaceRegistry._extract_signature_summary(
        code,
        unit_type="class",
        bare_name="DoesNotExist",
        backend=backend,
    )

    assert result == "DoesNotExist"
