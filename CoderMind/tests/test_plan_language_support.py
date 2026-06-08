from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from decoder_lang import ProjectTaskTemplates, get_backend  # noqa: E402
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


def test_task_planner_prefers_backend_project_task_templates(monkeypatch) -> None:
    planner = TaskPlanner(
        interfaces={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        data_flow={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        repo_name="tasklite",
        repo_info="Go CLI task tracker.",
    )

    def fake_templates(context):
        return ProjectTaskTemplates(
            dependencies=f"deps for {context.package_name}",
            main_entry=f"main for {context.package_name}",
            readme=f"readme for {context.package_name}",
        )

    monkeypatch.setattr(planner.backend, "project_task_templates", fake_templates)

    assert planner._build_requirements_task() == "deps for tasklite"
    assert planner._build_main_entry_task() == "main for tasklite"
    assert planner._build_readme_task() == "readme for tasklite"


def test_task_planner_project_tasks_use_rust_conventions() -> None:
    planner = TaskPlanner(
        interfaces={"meta": {"primary_language": "rust", "target_languages": ["rust"]}},
        data_flow={"meta": {"primary_language": "rust", "target_languages": ["rust"]}},
        repo_name="tasklite",
        repo_info="Rust CLI task tracker.",
    )

    requirements = planner._build_requirements_task()
    main_entry = planner._build_main_entry_task()
    readme = planner._build_readme_task()

    assert "Cargo.toml" in requirements
    assert "requirements.txt" not in requirements
    assert "src/main.rs" in main_entry
    assert "main.py" not in main_entry
    assert "cargo test" in readme
    assert "pytest" not in readme


def test_task_planner_project_tasks_use_typescript_conventions() -> None:
    planner = TaskPlanner(
        interfaces={
            "meta": {
                "primary_language": "typescript",
                "target_languages": ["typescript"],
            }
        },
        data_flow={
            "meta": {
                "primary_language": "typescript",
                "target_languages": ["typescript"],
            }
        },
        repo_name="tasklite",
        repo_info="TypeScript CLI task tracker.",
    )

    requirements = planner._build_requirements_task()
    main_entry = planner._build_main_entry_task()
    readme = planner._build_readme_task()

    assert "package.json" in requirements
    assert "requirements.txt" not in requirements
    assert "src/index.ts" in main_entry
    assert "main.py" not in main_entry
    assert "npm test" in readme
    assert "pytest" not in readme


def test_task_planner_special_tasks_are_language_neutral() -> None:
    planner = TaskPlanner(
        interfaces={"meta": {"primary_language": "rust", "target_languages": ["rust"]}},
        data_flow={
            "meta": {"primary_language": "rust", "target_languages": ["rust"]},
            "data_flow": [
                {"source": "Core", "target": "CLI", "data_type": "Payload"},
            ],
        },
        repo_name="tasklite",
        repo_info="Rust CLI task tracker.",
    )
    planned_tasks: dict = {"Core": {}}
    agent_results: dict = {"Core": {}}

    planner._add_special_tasks(planned_tasks, agent_results, ["Core"])
    text = "\n".join(
        task["task"]
        for files in planned_tasks.values()
        for tasks in files.values()
        for task in tasks
    )

    assert "main.py" not in text
    assert "styles.py" not in text


def test_rust_backend_accepts_basic_declarations() -> None:
    backend = get_backend("rust")
    code = "pub struct Task {\n    pub title: String,\n}\n\npub fn run() {}\n"

    ok, error = backend.syntax_check(code, "src/lib.rs")
    units = backend.list_code_units(code, "src/lib.rs")

    assert ok, error
    assert {unit.unit_type for unit in units} >= {"struct", "function"}
    assert backend.prompt_hints().test_framework_name == "cargo test"


def test_typescript_backend_accepts_basic_declarations() -> None:
    backend = get_backend("typescript")
    code = "export interface Task { title: string }\nexport function run(): void {}\n"

    ok, error = backend.syntax_check(code, "src/index.ts")
    units = backend.list_code_units(code, "src/index.ts")

    assert ok, error
    assert "interface Task" in [f"{unit.unit_type} {unit.name}" for unit in units]
    assert any(unit.name == "run" for unit in units)
    assert backend.prompt_hints().test_framework_name == "npm test"


def test_interface_validation_accepts_typescript_interface() -> None:
    backend = get_backend("typescript")
    ok, error, info = validate_interface(
        {
            "features": ["Task Domain Model/task schema"],
            "code": "export interface Task { title: string }\n",
        },
        {"Task Domain Model/task schema"},
        set(),
        backend=backend,
    )

    assert ok, error
    assert "interface Task" in info["declarations"]