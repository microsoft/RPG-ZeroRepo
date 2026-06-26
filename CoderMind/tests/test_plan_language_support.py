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
from func_design.interface_agent import (  # noqa: E402
    DependencyCollector,
    SubtreeInterfaceAgent,
    SubtreeInterfaceOutput,
    validate_interface,
)
from func_design.interface_prompts import SUBTREE_INTERFACE_PROMPT  # noqa: E402
from plan_tasks import TaskPlanner  # noqa: E402


def test_dependency_collector_extracts_rust_inheritance() -> None:
    # Regression for G3: non-Python inheritance edges must be extracted via
    # the backend's list_inheritance (Rust trait impls emit `inherits`),
    # not the Python-only AST path that silently produced nothing before.
    collector = DependencyCollector(
        known_base_classes={"Repo"},
        known_types=set(),
        target_language="rust",
    )
    code = "struct Store;\ntrait Repo {}\nimpl Repo for Store {}\n"
    collector.analyze_code_dependencies(
        code=code,
        file_path="src/store.rs",
        base_class_files={"Repo": "src/repo.rs"},
    )
    assert any(
        e["child"] == "Store" and e["parent"] == "Repo"
        and e["parent_file"] == "src/repo.rs"
        for e in collector.inheritance_edges
    ), collector.inheritance_edges


def test_dependency_collector_python_inheritance_still_works() -> None:
    # The Python AST-derived path keeps producing inheritance edges.
    collector = DependencyCollector(
        known_base_classes={"Base"},
        known_types=set(),
        target_language="python",
    )
    code = "class Base:\n    pass\n\nclass Child(Base):\n    pass\n"
    collector.analyze_code_dependencies(
        code=code,
        file_path="pkg/child.py",
        base_class_files={"Base": "pkg/base.py"},
    )
    assert any(
        e["child"] == "Child" and e["parent"] == "Base"
        for e in collector.inheritance_edges
    ), collector.inheritance_edges


def test_dependency_collector_python_same_file_method_calls() -> None:
    collector = DependencyCollector(
        known_base_classes=set(),
        known_types=set(),
        target_language="python",
    )
    code = """
class RecordFactory:
    def create_record(self, title: str) -> dict:
        return {"title": title}


class Planner:
    def __init__(self, record_factory: RecordFactory) -> None:
        self._record_factory = record_factory or RecordFactory()

    def resolve_action(self, action: str) -> str:
        return action.strip()

    def _require_action(self, action: str) -> str:
        return self.resolve_action(action)

    def plan_add(self, title: str) -> dict:
        action = self._require_action("add")
        record = self._record_factory.create_record(title)
        return {"action": action, "record": record}
"""

    collector.analyze_code_dependencies(
        code=code,
        file_path="src/domain/todo.py",
        base_class_files={},
    )

    assert {
        (edge["caller"], edge["callee"])
        for edge in collector.invocation_edges
    } >= {
        ("method __init__", "class RecordFactory"),
        ("class Planner", "class RecordFactory"),
        ("method plan_add", "method resolve_action"),
        ("method plan_add", "method create_record"),
    }


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


def test_subtree_interface_output_accepts_common_file_aliases() -> None:
    model = SubtreeInterfaceOutput.model_validate({
        "files": [
            {
                "path": "src/tasklite_cli/task/task.c",
                "features": ["Task Domain Model/task schema/define record"],
                "code": "int task_record_init(void);\n",
            }
        ]
    })

    assert model.files[0].file_path == "src/tasklite_cli/task/task.c"
    assert model.files[0].interfaces[0].features == [
        "Task Domain Model/task schema/define record"
    ]


def test_interface_validation_filters_non_target_and_duplicate_features() -> None:
    backend = get_backend("c")
    interface = {
        "features": [
            "Task Domain Model/task schema/define record",
            "Task Domain Model/glue/generated helper",
            "Task Domain Model/task schema/already covered",
        ],
        "code": "int task_record_init(void);\n",
    }

    ok, error, info = validate_interface(
        interface,
        {
            "Task Domain Model/task schema/define record",
            "Task Domain Model/task schema/already covered",
        },
        {"Task Domain Model/task schema/already covered"},
        backend=backend,
    )

    assert ok, error
    assert interface["features"] == ["Task Domain Model/task schema/define record"]
    assert "function task_record_init" in info["declarations"]


def test_subtree_agent_adds_c_fallback_for_remaining_features() -> None:
    agent = SubtreeInterfaceAgent(target_language="c")
    feature = "Task Domain Model/task schema/status representation/encode completion flag"
    state = {
        "target_features": {feature},
        "covered_features": set(),
        "all_interfaces": [],
        "all_code_blocks": [],
    }

    agent._complete_remaining_c_family_features("src/tasklite_cli/task/task.c", state)
    result, _new_features = agent._build_file_result(
        file_path="src/tasklite_cli/task/task.c",
        all_interfaces=state["all_interfaces"],
        all_code_blocks=state["all_code_blocks"],
        target_features=state["target_features"],
        covered_features=state["covered_features"],
    )

    assert result["success"]
    assert feature in next(iter(result["units_to_features"].values()))
    assert "int task" in result["file_code"]


def test_subtree_agent_adds_cpp_fallback_for_empty_file_result() -> None:
    agent = SubtreeInterfaceAgent(target_language="cpp")
    features = {
        "CLI Entry and Dispatch/storage/options/use local tasks file",
        "CLI Entry and Dispatch/storage/options/resolve store path",
    }
    state = {
        "target_features": features,
        "covered_features": set(),
        "all_interfaces": [],
        "all_code_blocks": [],
    }

    agent._complete_remaining_c_family_features(
        "src/tasklite_cli/cli/store_path_options.cpp",
        state,
    )
    result, _new_features = agent._build_file_result(
        file_path="src/tasklite_cli/cli/store_path_options.cpp",
        all_interfaces=state["all_interfaces"],
        all_code_blocks=state["all_code_blocks"],
        target_features=state["target_features"],
        covered_features=state["covered_features"],
    )

    assert result["success"]
    assert set(next(iter(result["units_to_features"].values()))) == features
    assert "namespace tasklite" in result["file_code"]


def test_subtree_agent_uses_cpp_fallback_for_verification_subtree() -> None:
    class FailingLLM:
        def call_structured(self, **_kwargs):
            raise AssertionError("LLM should not run for C++ verification fallback")

    agent = SubtreeInterfaceAgent(
        llm_client=FailingLLM(),
        target_language="cpp",
    )
    files = [
        {
            "path": "tests/store_test.cpp",
            "feature_paths": [
                "Verification and Test Isolation/store/loading coverage/verify missing file loading",
                "Verification and Test Isolation/store/corruption coverage/verify corrupt json handling",
            ],
        },
        {
            "path": "tests/cli_test.cpp",
            "feature_paths": [
                "Verification and Test Isolation/cli/list coverage/verify task list output",
            ],
        },
    ]

    result = agent.design_subtree_interfaces(
        file_nodes=files,
        file_order=["tests/store_test.cpp", "tests/cli_test.cpp"],
        repo_info="TaskLite C++ CLI",
        data_flow_str="",
        base_classes_str="",
        upstream_context="",
        subtree_name="Verification and Test Isolation",
    )

    assert result["tests/store_test.cpp"]["success"]
    assert result["tests/cli_test.cpp"]["success"]
    assert len(result["tests/store_test.cpp"]["units"]) == 1
    assert len(result["tests/cli_test.cpp"]["units"]) == 1


def test_interface_validation_strips_markdown_fence() -> None:
    backend = get_backend("go")
    ok, error, info = validate_interface(
        {
            "features": ["Runtime Architecture Constraints/layout/packages/use fixed package layout"],
            "code": "```go\npackage app\n\ntype AppLayout struct {\n\tStorePath string\n}\n```",
        },
        {"Runtime Architecture Constraints/layout/packages/use fixed package layout"},
        set(),
        backend=backend,
    )

    assert ok, error
    assert "struct AppLayout" in info["declarations"]


def test_interface_validation_accepts_python_backend_docstring() -> None:
    backend = get_backend("python")
    ok, error, info = validate_interface(
        {
            "features": ["Application Infrastructure/server bootstrap/application factory setup"],
            "code": (
                "from flask import Flask\n\n"
                "def create_app() -> Flask:\n"
                "    \"\"\"Create and configure the Flask application.\"\"\"\n"
                "    ...\n"
            ),
        },
        {"Application Infrastructure/server bootstrap/application factory setup"},
        set(),
        backend=backend,
    )

    assert ok, error
    assert "function create_app" in info["declarations"]


def test_subtree_interface_prompt_is_language_neutral() -> None:
    assert "with `pass` bodies" not in SUBTREE_INTERFACE_PROMPT
    assert "All function/method bodies must use `pass`" not in SUBTREE_INTERFACE_PROMPT
    assert "target-language declaration stubs" in SUBTREE_INTERFACE_PROMPT


def test_typescript_subtree_prompt_omits_python_import_convention() -> None:
    agent = SubtreeInterfaceAgent(target_language="typescript")
    prompt = agent._build_subtree_user_prompt(
        remaining_files=["src/tasklite-cli/cli/main.ts"],
        file_states={
            "src/tasklite-cli/cli/main.ts": {
                "target_features": {"CLI Application/startup/process bootstrap"},
                "covered_features": set(),
                "all_code_blocks": [],
            }
        },
        file_info_map={
            "src/tasklite-cli/cli/main.ts": {
                "path": "src/tasklite-cli/cli/main.ts",
                "feature_paths": ["CLI Application/startup/process bootstrap"],
            }
        },
        repo_info="TypeScript CLI task tracker.",
        data_flow_str="No data flow.",
        base_classes_str="No base classes.",
        upstream_context="No upstream interfaces.",
        last_error="",
    )

    assert "Import Convention" not in prompt
    assert "from src.tasklite-cli" not in prompt


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


def test_go_main_entry_reuses_existing_command_package() -> None:
    # The skeleton already placed the entry under cmd/todo/main.go. The
    # MAIN_ENTRY task must reuse that path, not generate a second
    # cmd/<repo-slug>/main.go (which would yield two func main()).
    interfaces = {
        "meta": {"primary_language": "go", "target_languages": ["go"]},
        "subtrees": {
            "Server": {
                "interfaces": {
                    "cmd/todo/main.go": {"units": ["function main"]},
                    "internal/store/store.go": {"units": ["struct Store"]},
                }
            }
        },
    }
    planner = TaskPlanner(
        interfaces=interfaces,
        data_flow={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        repo_name="demo-go-web-todo",
        repo_info="Go web todo.",
    )

    # Go command-path resolution moved to the backend
    # (``go_backend.find_existing_entry``); the planner reuses it when
    # building the synthetic MAIN_ENTRY task.
    assert get_backend("go").find_existing_entry(interfaces) == "cmd/todo/main.go"
    main_entry = planner._build_main_entry_task()
    assert "cmd/todo/main.go" in main_entry
    assert "cmd/demo-go-web-todo/main.go" not in main_entry


def test_go_main_entry_falls_back_when_no_command_package() -> None:
    # No cmd/*/main.go in the skeleton → fall back to the canonical
    # cmd/<module>/main.go from the backend.
    interfaces = {
        "meta": {"primary_language": "go", "target_languages": ["go"]},
        "subtrees": {
            "Core": {"interfaces": {"internal/store/store.go": {"units": ["struct Store"]}}}
        },
    }
    planner = TaskPlanner(
        interfaces=interfaces,
        data_flow={"meta": {"primary_language": "go", "target_languages": ["go"]}},
        repo_name="tasklite",
        repo_info="Go CLI.",
    )

    # No cmd/*/main.go in the skeleton → the backend reports no existing entry
    # and the planner falls back to the canonical cmd/<module>/main.go.
    assert get_backend("go").find_existing_entry(interfaces) is None
    assert "cmd/tasklite/main.go" in planner._build_main_entry_task()


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
    code = """
export interface Task { title: string }
export type TaskId = number;
export declare function run(task: Task): void;
export declare class TaskCli { run(task: Task): void; }
"""

    ok, error = backend.syntax_check(code, "src/index.ts")
    units = backend.list_code_units(code, "src/index.ts")

    assert ok, error
    declarations = [f"{unit.unit_type} {unit.name}" for unit in units]
    assert "interface Task" in declarations
    assert "type TaskId" in declarations
    assert "function run" in declarations
    assert "class TaskCli" in declarations
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


def test_interface_validation_accepts_typescript_declare_function() -> None:
    backend = get_backend("typescript")
    ok, error, info = validate_interface(
        {
            "features": ["CLI Application/startup/process bootstrap/bootstrap main command"],
            "code": "export declare function runTasklite(argv: readonly string[]): Promise<number>;\n",
        },
        {"CLI Application/startup/process bootstrap/bootstrap main command"},
        set(),
        backend=backend,
    )

    assert ok, error
    assert "function runTasklite" in info["declarations"]


def test_interface_validation_accepts_typescript_jsdoc_backticks() -> None:
    backend = get_backend("typescript")
    ok, error, info = validate_interface(
        {
            "features": [
                "CLI Application/store file/path selection/select default file",
                "CLI Application/store file/path selection/select override file",
            ],
            "code": """```typescript
import { homedir } from 'node:os';
import { join } from 'node:path';

/**
 * Resolve the filesystem path for the JSON task store.
 *
 * When an override path is supplied, it is returned as-is.
 * Otherwise the default path is computed as `~/.tasklite.json`.
 * Empty-string values are treated as "no override".
 *
 * @param override - Optional explicit path to the store file.
 * @returns Absolute filesystem path to the JSON store file.
 */
export function resolveStorePath(override?: string): string;
```""",
        },
        {
            "CLI Application/store file/path selection/select default file",
            "CLI Application/store file/path selection/select override file",
        },
        set(),
        backend=backend,
    )

    assert ok, error
    assert "function resolveStorePath" in info["declarations"]


def test_file_ordering_uses_imports_for_go() -> None:
    # Regression: non-Python file ordering previously degraded to the raw LLM
    # order because dependency extraction used Python AST only. Go imports must
    # now drive the topological sort (store before its cli importer). The
    # backend resolves module imports through go.mod, so the module manifest
    # is part of the ordered file set.
    from plan_tasks import correct_intra_subtree_file_order

    interfaces = {
        "go.mod": {"file_code": "module tasklite\n\ngo 1.21\n"},
        "internal/store/store.go": {
            "file_code": "package store\n\ntype Store struct{}\nfunc New() *Store { return &Store{} }\n",
        },
        "cmd/app/cli.go": {
            "file_code": "package main\n\nimport \"tasklite/internal/store\"\n\nfunc main(){ _ = store.New() }\n",
        },
    }
    corrected, diag = correct_intra_subtree_file_order(
        subtree_name="Core",
        files_order=["cmd/app/cli.go", "internal/store/store.go", "go.mod"],
        subtree_interfaces=interfaces,
        language="go",
    )

    assert corrected == ["internal/store/store.go", "cmd/app/cli.go", "go.mod"]
    assert diag["changed"] is True
    assert diag["reason"] == "backend_file_dependencies"


def test_file_ordering_keeps_python_dotted_module_path() -> None:
    from plan_tasks import correct_intra_subtree_file_order

    interfaces = {
        "src/app/store.py": {"file_code": "class Store:\n    pass\n"},
        "src/app/cli.py": {"file_code": "from app.store import Store\n"},
    }
    corrected, diag = correct_intra_subtree_file_order(
        subtree_name="Core",
        files_order=["src/app/cli.py", "src/app/store.py"],
        subtree_interfaces=interfaces,
        language="python",
    )

    assert corrected == ["src/app/store.py", "src/app/cli.py"]
    assert diag["reason"] == "backend_file_dependencies"

