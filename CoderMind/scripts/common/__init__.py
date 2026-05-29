#!/usr/bin/env python3
"""Common Module.

This module contains common utilities shared across all scripts.
"""

from .trajectory import (
    Trajectory,
    StepStatus,
    LLMInteraction,
    load_or_create_trajectory,
)

from .llm_client import (
    LLMClient,
    LLMCallRecord,
)

from .task_batch import (
    PlannedTask,
    load_tasks_from_tasks_json,
    get_task_by_id,
    get_next_pending_task,
)

from .execution_state import (
    BatchExecutionState,
    CodeGenState,
    WorkflowPhase,
    FailureType,
    WorkflowType,
    load_code_gen_state,
    save_code_gen_state,
    update_batch_state,
    complete_batch,
    reset_rpg_backup_tracking,
)

from .git_utils import (
    GitRunner,
    GitResult,
    create_task_branch,
    complete_task_branch,
)

from .paths import (
    CMIND_DIR,

    SKELETON_FILE,
    DATA_FLOW_FILE,
    INTERFACES_FILE,
    BASE_CLASSES_FILE,
    RPG_FILE,
    REPO_RPG_FILE,
    DEP_GRAPH_FILE,
    REPO_INFO_FILE,
    TASKS_FILE,
    CODE_GEN_STATE_FILE,
    TRAJECTORY_DIR,
    SKELETON_SUMMARY_FILE,
    ensure_cmind_dir,
    get_trajectory_file,
)

from .utils import (
    print_unicode_table,
    get_skeleton_tree_string,
    extract_functional_areas_from_skeleton,
    format_functional_graph_overview,
    extract_component_directories,
    validate_python_syntax,
    format_data_flow_edges,
    format_base_classes,
    format_data_structures,
    format_base_classes_and_data_structures,
    get_repo_info_from_files,
    get_project_background_context,
    get_all_leaf_paths,
    get_leaf_name,
    get_leaf_description,
    get_all_leaf_descriptions,
    extract_class_names,
    # --- M4 Utils: newly ported functions ---
    normalize_path,
    is_test_file,
    merge_intervals,
    filter_excluded_files,
    parse_solution_output,
    parse_code_blocks,
    get_skeleton,
    transfer_parsed_tree,
    format_parsed_tree,
    iterative_by_folder,
    get_node_range_robust,
    extract_source_by_lines,
    # Utils (M6: token counting)
    calculate_tokens,
    truncate_by_token,
)

from .tools import (
    Tool,
    ToolCall,
    ToolCallArguments,
    ToolError,
    ToolExecResult,
    ToolExecutionError,
    ToolExecutor,
    ToolHandler,
    ToolNotFoundError,
    ToolParameter,
    ToolResult,
    ToolValidationError,
)

from .llm_types import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
    Memory,
)

__all__ = [
    # Trajectory
    "Trajectory",
    "StepStatus",
    "LLMInteraction",
    "load_or_create_trajectory",
    # LLM Client
    "LLMClient",
    "LLMCallRecord",
    # Task Batch
    "PlannedTask",
    "load_tasks_from_tasks_json",
    "get_task_by_id",
    "get_next_pending_task",
    # Execution State
    "BatchExecutionState",
    "CodeGenState",
    "WorkflowPhase",
    "FailureType",
    "WorkflowType",
    "load_code_gen_state",
    "save_code_gen_state",
    "update_batch_state",
    "complete_batch",
    "reset_rpg_backup_tracking",
    # Git Utils
    "GitRunner",
    "GitResult",
    "create_task_branch",
    "complete_task_branch",
    # Paths
    "CMIND_DIR",

    "SKELETON_FILE",
    "DATA_FLOW_FILE",
    "INTERFACES_FILE",
    "BASE_CLASSES_FILE",
    "RPG_FILE",
    "REPO_RPG_FILE",
    "DEP_GRAPH_FILE",
    "REPO_INFO_FILE",
    "TASKS_FILE",
    "CODE_GEN_STATE_FILE",
    "TRAJECTORY_DIR",
    "SKELETON_SUMMARY_FILE",
    "ensure_cmind_dir",
    "get_trajectory_file",
    # Utils
    "print_unicode_table",
    "get_skeleton_tree_string",
    "extract_functional_areas_from_skeleton",
    "format_functional_graph_overview",
    "extract_component_directories",
    "validate_python_syntax",
    "format_data_flow_edges",
    "format_base_classes",
    "format_data_structures",
    "format_base_classes_and_data_structures",
    "get_repo_info_from_files",
    "get_project_background_context",
    "get_all_leaf_paths",
    "get_leaf_name",
    "get_leaf_description",
    "get_all_leaf_descriptions",
    "extract_class_names",
    # Utils (M4: ported from RPG-ZeroRepo)
    "normalize_path",
    "is_test_file",
    "merge_intervals",
    "filter_excluded_files",
    "parse_solution_output",
    "parse_code_blocks",
    "get_skeleton",
    "transfer_parsed_tree",
    "format_parsed_tree",
    "iterative_by_folder",
    "get_node_range_robust",
    "extract_source_by_lines",
    # Utils (M6: token counting)
    "calculate_tokens",
    "truncate_by_token",
    # Tools
    "Tool",
    "ToolCall",
    "ToolCallArguments",
    "ToolError",
    "ToolExecResult",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolHandler",
    "ToolNotFoundError",
    "ToolParameter",
    "ToolResult",
    "ToolValidationError",
    # LLM Types (M5: ported from RPG-ZeroRepo)
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "Message",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "Memory",
]
