from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from decoder_lang import get_backend  # noqa: E402
from func_design.base_class_agent import (  # noqa: E402
    BaseClassOutput,
    validate_base_classes_model,
    validate_data_structures,
)
from func_design.interface_agent import validate_interface  # noqa: E402
from plan_tasks import TaskPlanner  # noqa: E402


def test_base_class_validation_accepts_go_source() -> None:
    backend = get_backend("go")
    model = BaseClassOutput.model_validate({
        "base_classes": [
            {
                "file_path": "internal/task/store.go",
                "code": "package task\n\ntype Store interface {\n\tLoad() error\n}\n",
                "scope": "Task Store",
                "subclasses": {"Store": ["FileStore", "MemoryStore"]},
            }
        ],
        "data_structures": [],
    })

    ok, error = validate_base_classes_model(
        model,
        valid_subtrees=["Task Store"],
        backend=backend,
    )

    assert ok, error


def test_data_structure_validation_accepts_go_source() -> None:
    backend = get_backend("go")
    ok, error = validate_data_structures(
        [
            {
                "code": "package task\n\ntype TaskRecord struct {\n\tID int\n}\n",
                "subtree": "Task Store",
                "data_flow_types": ["TaskRecord"],
            }
        ],
        ["TaskRecord"],
        valid_subtrees=["Task Store"],
        backend=backend,
    )

    assert ok, error


def test_interface_validation_accepts_go_declaration() -> None:
    backend = get_backend("go")
    ok, error, info = validate_interface(
        {
            "features": ["Task Lifecycle Management/task/create"],
            "code": "package task\n\ntype Task struct {\n\tTitle string\n}\n",
        },
        {"Task Lifecycle Management/task/create"},
        set(),
        backend=backend,
    )

    assert ok, error
    assert "struct Task" in info["declarations"]


def test_task_planner_project_tasks_use_go_conventions() -> None:
    planner = TaskPlanner(
        interfaces={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        data_flow={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        repo_name="tasklite",
        repo_info="Go CLI task tracker.",
    )

    requirements = planner._build_requirements_task()
    main_entry = planner._build_main_entry_task()
    readme = planner._build_readme_task()

    assert "go.mod" in requirements
    assert "requirements.txt" not in requirements
    assert "cmd/tasklite/main.go" in main_entry
    assert "main.py" not in main_entry
    assert "go test ./..." in readme
    assert "pytest" not in readme