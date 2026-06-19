#!/usr/bin/env python3
"""Plan Tasks Script - Implementation Level Step 5.

Function: Create implementation tasks from interfaces using LLM
- Reads interfaces.json, data_flow.json, repo_rpg.json
- Uses LLM to plan implementation tasks for each file
- Validates that all units are covered without duplicates
- Generates tasks.json with ordered implementation tasks

Input: .cmind/interfaces.json, .cmind/data_flow.json, .cmind/repo_rpg.json
Output: .cmind/tasks.json (ordered implementation tasks)
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict, deque

from common.trajectory import Trajectory, load_or_create_trajectory
from common import LLMClient
from common.code_dedup import dedup_file_code
from common.language_meta import extract_language_metadata, metadata_with_languages
from decoder_lang import FileDependencyEdge, ProjectTaskContext, get_backend, infer_language_from_path
from rpg import uuid8

# Import centralized paths
from common.paths import (
    DATA_FLOW_FILE,
    INTERFACES_FILE,
    SKELETON_FILE,
    REPO_RPG_FILE as RPG_FILE,
    REPO_INFO_FILE,
    TASKS_FILE as OUTPUT_FILE,
    REPO_DIR,
)
import re as _re
from os.path import commonpath, dirname


# ============================================================================
# Prompts (ZeroRepo compatible)
# ============================================================================

TASK_PLANNER_PROMPT = """
## Instruction
You are an Implementation Task Planner. Your job is to decide the implementation order for all files and units within a subtree (module) by grouping them into dependency-aware tasks that feel natural for real development and code review.

Think like someone organizing GitHub PRs:
each task should represent a meaningful, reviewable feature step — not just a list of functions.

## Planning Principles
- Implement prerequisites before dependents, both within and across files.
- Prefer tasks that deliver a complete functional milestone.
- Favor fewer, clearer tasks over many tiny ones.
- Each task targets units within a single file; task ordering reflects cross-file dependencies.
- A task should be implementable without needing code from future tasks.

## Task Scope
Combine units when they naturally belong together:
- data model + validation/normalization
- helper functions + main logic that uses them
- core class + tightly coupled behavior methods

Separate units when they:
- are foundational utilities reused across many places
- represent orchestration or entry-point logic
- clearly belong to a higher-level layer

Split only when it improves clarity or dependency flow, not for symmetry or size.

## Task Description Style
Each task description should convey:
1) What capability this task delivers
2) Why it matters right now
3) What scope is included (and what is not)

Focus on functional milestone + intent. Avoid vague summaries or simply restating function names.

## Output Format
Your response must include exactly one `<think>` block and exactly one `<result_json>` block:
<think>
Analyze dependencies across all files, identify functional steps, decide task grouping and order.
</think>
<result_json>
{{
  "tasks": [
    {{
      "file_path": "<file_path>",
      "units": ["<unit_key_1>", "<unit_key_2>", ...],
      "task": "<GitHub-style task description>"
    }}
  ]
}}
</result_json>

Constraints:
- Every unit in every file must appear exactly once across all tasks.
- Each task's units must belong to the specified file.
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlannedTask:
    """Represents a planned implementation task."""
    task_id: str = field(init=False)
    task: str
    file_path: str
    units_key: List[str]
    unit_to_code: Dict[str, str]
    unit_to_features: Dict[str, List]
    priority: int = 0
    subtree: str = ""
    task_type: str = "implementation"

    def __post_init__(self):
        unique_suffix = uuid8()
        self.task_id = f"{self.file_path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_suffix}"

        if not isinstance(self.units_key, list) or not self.units_key:
            raise ValueError("PlannedTask validation error: 'units_key' must be a non-empty list.")

        missing_in_code = [k for k in self.units_key if k not in self.unit_to_code]
        if missing_in_code:
            raise ValueError(
                f"PlannedTask validation error: units_key contains keys not present "
                f"in unit_to_code: {missing_in_code}"
            )

        # Auto-fill missing unit_to_features keys (informational only)
        for k in self.units_key:
            if k not in self.unit_to_features:
                self.unit_to_features[k] = []

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "file_path": self.file_path,
            "units_key": self.units_key,
            "unit_to_code": self.unit_to_code,
            "unit_to_features": self.unit_to_features,
            "priority": self.priority,
            "subtree": self.subtree,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlannedTask":
        obj = cls(
            task=data["task"],
            file_path=data["file_path"],
            units_key=data["units_key"],
            unit_to_code=data["unit_to_code"],
            unit_to_features=data["unit_to_features"],
            priority=data.get("priority", 0),
            subtree=data.get("subtree", ""),
            task_type=data.get("task_type", "implementation"),
        )
        if "task_id" in data:
            obj.task_id = data["task_id"]
        return obj


def _load_dependency_source_code(file_path: str, interface_file_code: str) -> str:
    """Load source code for dependency analysis, combining repo and interface inputs."""
    code_parts: List[str] = []
    repo_file_path = REPO_DIR / file_path
    if repo_file_path.exists():
        try:
            repo_code = repo_file_path.read_text(encoding="utf-8")
            if repo_code.strip():
                code_parts.append(repo_code)
        except OSError:
            pass
    if interface_file_code.strip():
        code_parts.append(interface_file_code)
    return "\n\n".join(code_parts)


def _infer_backend_name_for_file(file_path: str, fallback_language: Optional[str] = None) -> str:
    """Infer the backend name for a file without relying on subtree primary language."""
    inferred = infer_language_from_path(file_path)
    if inferred:
        return inferred
    return (fallback_language or "python").lower()


def _subtree_file_sources(
    files: List[str],
    subtree_interfaces: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Build source-code map used by backend file-dependency resolvers."""
    sources: Dict[str, str] = {}
    for file_path in files:
        interface_file_code = subtree_interfaces.get(file_path, {}).get("file_code", "")
        sources[file_path] = _load_dependency_source_code(file_path, interface_file_code)
    return sources


def _edges_to_toposort_input(edges: List[FileDependencyEdge]) -> Dict[str, Set[str]]:
    dependency_edges: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        dependency_edges[edge.dependency].add(edge.dependent)
    return dependency_edges


def _serialize_dependency_edges(edges: List[FileDependencyEdge]) -> List[Dict[str, Any]]:
    return [
        {
            "dependency": edge.dependency,
            "dependent": edge.dependent,
            "reason": edge.reason,
            "confidence": edge.confidence,
            "detail": edge.detail,
        }
        for edge in edges
    ]


def _language_summary(language_to_files: Dict[str, List[str]]) -> Dict[str, int]:
    return {language: len(files) for language, files in sorted(language_to_files.items())}


def _sort_files_with_backend_edges(
    files: List[str],
    backend_name: str,
    file_sources: Dict[str, str],
) -> tuple[List[str], List[FileDependencyEdge], bool]:
    """Sort a same-language file group with its backend's dependency edges."""
    backend = get_backend(backend_name)
    edges = backend.file_dependency_edges(files, file_sources)
    corrected = _topologically_sort_files(files, _edges_to_toposort_input(edges))
    if corrected is None:
        return files, edges, True
    return corrected, edges, False


def _merge_sorted_language_groups(
    original_files: List[str],
    file_to_language: Dict[str, str],
    sorted_by_language: Dict[str, List[str]],
) -> List[str]:
    """Fill each original language slot with the next sorted file from that language."""
    queues = {language: deque(files) for language, files in sorted_by_language.items()}
    merged: List[str] = []
    for file_path in original_files:
        language = file_to_language[file_path]
        merged.append(queues[language].popleft())
    return merged

def _topologically_sort_files(
    files_order: List[str],
    dependency_edges: Dict[str, Set[str]],
) -> Optional[List[str]]:
    """Stable topological sort preserving original order for unrelated files."""
    order_index = {file_path: index for index, file_path in enumerate(files_order)}
    adjacency: Dict[str, Set[str]] = {file_path: set() for file_path in files_order}
    indegree: Dict[str, int] = {file_path: 0 for file_path in files_order}

    for dependency_file, dependent_files in dependency_edges.items():
        if dependency_file not in adjacency:
            continue
        for dependent_file in dependent_files:
            if dependent_file not in adjacency:
                continue
            if dependent_file not in adjacency[dependency_file]:
                adjacency[dependency_file].add(dependent_file)
                indegree[dependent_file] += 1

    queue = deque(sorted(
        [file_path for file_path, degree in indegree.items() if degree == 0],
        key=lambda file_path: order_index[file_path],
    ))
    resolved_order: List[str] = []

    while queue:
        file_path = queue.popleft()
        resolved_order.append(file_path)

        for neighbor in sorted(adjacency[file_path], key=lambda item: order_index[item]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                insert_pos = 0
                while insert_pos < len(queue) and order_index[queue[insert_pos]] <= order_index[neighbor]:
                    insert_pos += 1
                queue.insert(insert_pos, neighbor)

    if len(resolved_order) != len(files_order):
        return None

    return resolved_order


def correct_intra_subtree_file_order(
    subtree_name: str,
    files_order: List[str],
    subtree_interfaces: Dict[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    language: Optional[str] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """Correct file order with per-file language backend dependency rules."""
    logger = logger or logging.getLogger(__name__)
    available_files = [file_path for file_path in files_order if file_path in subtree_interfaces]
    file_to_language = {
        file_path: _infer_backend_name_for_file(file_path, language)
        for file_path in available_files
    }
    language_to_files: Dict[str, List[str]] = defaultdict(list)
    for file_path in available_files:
        language_to_files[file_to_language[file_path]].append(file_path)
    languages = _language_summary(language_to_files)

    if len(available_files) <= 1:
        return available_files, {
            "original_files_order": list(available_files),
            "corrected_files_order": list(available_files),
            "changed": False,
            "dependency_edges": [],
            "reason": "single_file_or_empty_subtree",
            "languages": languages,
            "language_groups": languages,
            "cycle_detected": False,
            "has_unresolved_or_ambiguous_edges": False,
            "has_weak_edges": False,
        }

    file_sources = _subtree_file_sources(available_files, subtree_interfaces)
    all_edges: List[FileDependencyEdge] = []
    cycles_by_language: Dict[str, bool] = {}

    if len(language_to_files) == 1:
        backend_name = next(iter(language_to_files))
        corrected_order, edges, cycle_detected = _sort_files_with_backend_edges(
            available_files,
            backend_name,
            file_sources,
        )
        all_edges.extend(edges)
        if cycle_detected:
            logger.warning(
                "[TaskPlanner] Detected cyclic or invalid %s file dependencies in '%s'; keeping original file order.",
                backend_name,
                subtree_name,
            )
            corrected_order = available_files
        changed = corrected_order != available_files
        if changed:
            logger.info(
                "[TaskPlanner] Corrected files_order for subtree '%s' using %s backend dependencies: %s -> %s",
                subtree_name,
                backend_name,
                available_files,
                corrected_order,
            )
        return corrected_order, {
            "original_files_order": list(available_files),
            "corrected_files_order": list(corrected_order),
            "changed": changed,
            "dependency_edges": _serialize_dependency_edges(all_edges),
            "reason": "backend_file_dependencies",
            "languages": languages,
            "language_groups": languages,
            "cycle_detected": cycle_detected,
            "has_unresolved_or_ambiguous_edges": any(
                edge.confidence in {"unresolved", "ambiguous"}
                for edge in all_edges
            ),
            "has_weak_edges": any(edge.confidence == "weak" for edge in all_edges),
        }

    sorted_by_language: Dict[str, List[str]] = {}
    for backend_name, language_files in language_to_files.items():
        sorted_group, edges, cycle_detected = _sort_files_with_backend_edges(
            language_files,
            backend_name,
            file_sources,
        )
        all_edges.extend(edges)
        cycles_by_language[backend_name] = cycle_detected
        sorted_by_language[backend_name] = language_files if cycle_detected else sorted_group
        if cycle_detected:
            logger.warning(
                "[TaskPlanner] Detected cyclic or invalid %s file dependencies in mixed-language subtree '%s'; keeping that language group's original order.",
                backend_name,
                subtree_name,
            )

    corrected_order = _merge_sorted_language_groups(
        available_files,
        file_to_language,
        sorted_by_language,
    )
    changed = corrected_order != available_files
    if changed:
        logger.info(
            "[TaskPlanner] Corrected files_order for mixed-language subtree '%s' within language groups only: %s -> %s",
            subtree_name,
            available_files,
            corrected_order,
        )

    return corrected_order, {
        "original_files_order": list(available_files),
        "corrected_files_order": list(corrected_order),
        "changed": changed,
        "dependency_edges": _serialize_dependency_edges(all_edges),
        "reason": "mixed_language_grouped_sort",
        "languages": languages,
        "language_groups": languages,
        "cycles_by_language": cycles_by_language,
        "cycle_detected": any(cycles_by_language.values()),
        "has_unresolved_or_ambiguous_edges": any(
            edge.confidence in {"unresolved", "ambiguous"}
            for edge in all_edges
        ),
        "has_weak_edges": any(edge.confidence == "weak" for edge in all_edges),
        "cross_language_policy": "preserve_original_language_slots_no_cross_language_edges",
    }


# ============================================================================
# Validation Functions
# ============================================================================

def validate_tasks(
    tasks: List[Dict],
    file_unit_keys: Dict[str, List[str]]
) -> tuple[bool, str, Optional[List[Dict]]]:
    """Validate planned tasks for a subtree (ZeroRepo compatible validation).

    Args:
        tasks: List of task dicts with file_path, units, task
        file_unit_keys: Dict mapping file_path -> list of unit keys

    Returns: (success, error_message, validated_tasks)
    """
    if not tasks:
        return False, "Invalid: 'tasks' is empty. You must provide a complete list of ALL tasks.", None

    # Validate task structure + collect unit keys per file
    file_got_units: Dict[str, List[str]] = {}

    for i, task_entry in enumerate(tasks):
        required_keys = ["file_path", "units", "task"]
        missing = [k for k in required_keys if k not in task_entry]
        if missing:
            return False, f"Invalid task at index {i}: missing required keys {missing}.", None

        fp = task_entry["file_path"]
        task_units = task_entry["units"]

        if fp not in file_unit_keys:
            return False, (
                f"Invalid task at index {i}: unknown file_path '{fp}'. "
                f"Valid files: {list(file_unit_keys.keys())}"
            ), None

        if not isinstance(task_units, list) or not task_units:
            return False, f"Invalid task at index {i}: 'units' must be a non-empty list.", None

        file_got_units.setdefault(fp, []).extend(task_units)

    # Validate per file
    errors = []
    for fp, expected_units in file_unit_keys.items():
        expected = set(expected_units)
        got = set(file_got_units.get(fp, []))

        counter = Counter(file_got_units.get(fp, []))
        duplicates = sorted([k for k, c in counter.items() if c > 1])

        if got != expected:
            missing_u = sorted(list(expected - got))
            extra_u = sorted(list(got - expected))
            errors.append(
                f"File '{fp}': expected {len(expected)} units, got {len(got)}. "
                f"Missing: {missing_u}, Extra: {extra_u}, Duplicates: {duplicates}"
            )
        elif duplicates:
            errors.append(f"File '{fp}': duplicate unit keys: {duplicates}")

    if errors:
        all_units_info = {fp: units for fp, units in file_unit_keys.items()}
        return False, (
            "Unit key mismatch:\n" + "\n".join(errors) + "\n\n"
            f"IMPORTANT: Re-plan ALL tasks. Required units per file: {json.dumps(all_units_info)}"
        ), None

    total_units = sum(len(u) for u in file_unit_keys.values())
    return True, f"Planned {len(tasks)} tasks covering all {total_units} units across {len(file_unit_keys)} files.", tasks


def _dedup_interface_source(fdata: Dict[str, Any]) -> str:
    """Return a file's interface source with per-unit duplication collapsed.

    ``interfaces.json`` stores ``file_code`` as ``"\n\n".join(unit codes)``.
    For non-Python units the ``count_lines`` call in interface synthesis
    raises (``LPCodeUnit`` has no such method), and the fallback stores the
    whole interface block as *every* unit's code, so a file with N units
    embeds the entire file N times — an O(units x file_size) blow-up that
    pushes the planner prompt past the 128 KB single-argument limit on large
    modules.

    Collapsing identical blocks reconstructs the original single file: because
    each duplicate copy carries the module header and imports, keeping exactly
    one copy yields a valid, complete file rather than a header-less
    concatenation of bodies. Genuinely distinct per-unit slices are preserved
    unchanged. This is a consumer-side safety net; freshly serialized
    ``interfaces.json`` is already deduplicated at the source.
    """
    unit_codes = fdata.get("units_to_code", {}).values()
    return dedup_file_code(unit_codes, fallback=fdata.get("file_code", ""))


# ============================================================================
# Task Planner Agent (per subtree)
# ============================================================================

class TaskPlannerAgent:
    """Agent for planning implementation tasks for all files in a subtree."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        subtree: str,
        files_data: Dict[str, Dict[str, Any]],
        files_order: List[str],
        repo_name: str = "",
        repo_info: str = "",
        logger: Optional[logging.Logger] = None
    ):
        self.llm = llm_client
        self.subtree = subtree
        self.files_data = files_data  # file_path -> {file_code, units_to_code, units_to_features}
        self.files_order = files_order  # authoritative implementation order
        self.repo_name = repo_name
        self.repo_info = repo_info
        self.logger = logger or logging.getLogger(__name__)
    
    def plan_subtree_tasks(
        self,
        max_retry: int = 5,
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """Plan tasks for all files in a subtree using a single LLM call."""
        # Build file_unit_keys for validation
        file_unit_keys = {
            fp: list(fdata["units_to_code"].keys())
            for fp, fdata in self.files_data.items()
        }
        total_units = sum(len(v) for v in file_unit_keys.values())
        self.logger.info(
            f"[TaskPlannerAgent] Planning {total_units} units across "
            f"{len(self.files_data)} files for subtree '{self.subtree}'"
        )

        # Build system prompt
        system_prompt = TASK_PLANNER_PROMPT

        # Build per-file context (in files_order)
        files_context_parts = []
        for i, fp in enumerate(self.files_order):
            if fp not in self.files_data:
                continue
            fdata = self.files_data[fp]
            unit_keys = list(fdata["units_to_code"].keys())
            files_context_parts.append(
                f"### File {i + 1}: {fp}\n"
                f"Units ({len(unit_keys)}): {json.dumps(unit_keys)}\n\n"
                f"Source code (interfaces only):\n{_dedup_interface_source(fdata)}\n"
            )
        files_context = "\n---\n".join(files_context_parts)
        
        # Build all units summary (in files_order)
        ordered_file_unit_keys = {fp: file_unit_keys[fp] for fp in self.files_order if fp in file_unit_keys}
        all_units_summary = json.dumps(ordered_file_unit_keys, indent=2)
        
        # Build files_order hint
        files_order_hint = json.dumps(self.files_order, indent=2)
        
        # Build task prompt
        task_prompt = f"""Plan the implementation tasks for the repository: {self.repo_name}
Repository description: {self.repo_info}

Context:
- You are planning the implementation order for the subtree / module: {self.subtree}
- Total files: {len(self.files_data)}
- Total units: {total_units}

**CRITICAL — Mandatory file implementation order (files_order):**
{files_order_hint}
You MUST output tasks following this file order strictly.
All tasks for file N must appear BEFORE any task for file N+1.
Within each file, order tasks by internal dependency.

Files and their source code (listed in files_order):
{files_context}

All units per file (must ALL be covered exactly once):
{all_units_summary}
"""
        
        combined_prompt = f"{system_prompt}\n\n{task_prompt}"
        last_error = ""
        planned_tasks = []
        
        for step in range(max_steps):
            self.logger.info(f"[TaskPlannerAgent] Step {step + 1}/{max_steps} for subtree '{self.subtree}'")
            
            current_prompt = combined_prompt
            if last_error:
                current_prompt += f"\n\n[Tool Execution Feedback - Please fix and retry]:\n{last_error}"
            
            try:
                response = self.llm.generate(current_prompt, purpose=f"plan_{self.subtree}_{step + 1}")
                parsed = self.llm.parse_result_json(response)

                if not parsed:
                    # Fallback: try to find {"tasks": [...]} directly in the response
                    tasks_match = _re.search(r'\{\s*"tasks"\s*:\s*\[', response)
                    if tasks_match:
                        # Found "tasks" key — try brace-counting extraction
                        start = tasks_match.start()
                        brace_count = 0
                        for i, ch in enumerate(response[start:], start):
                            if ch == '{':
                                brace_count += 1
                            elif ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        parsed = json.loads(response[start:i+1])
                                        self.logger.info(
                                            "[TaskPlannerAgent] Recovered tasks JSON via fallback extraction"
                                        )
                                    except json.JSONDecodeError:
                                        pass
                                    break

                if not parsed:
                    # Show the LLM what it actually returned so it can fix the format
                    response_tail = response[-500:] if len(response) > 500 else response
                    last_error = (
                        "Failed to parse result_json from your response.\n"
                        "You MUST wrap your JSON output in <result_json></result_json> tags.\n"
                        "Your response ended with:\n"
                        f"```\n{response_tail}\n```\n\n"
                        "Expected format:\n"
                        "<result_json>\n"
                        '{"tasks": [{"file_path": "...", "units": [...], "task": "..."}]}\n'
                        "</result_json>"
                    )
                    continue
                
                tasks = parsed.get("tasks", [])
                
                # Validate
                success, message, validated_tasks = validate_tasks(tasks, file_unit_keys)
                
                if success:
                    self.logger.info(f"[TaskPlannerAgent] [OK] {message}")
                    planned_tasks = validated_tasks
                    break
                else:
                    self.logger.warning(f"[TaskPlannerAgent] Validation failed: {message}")
                    last_error = message
                    
            except Exception as e:
                self.logger.error(f"[TaskPlannerAgent] Error: {e}")
                last_error = str(e)
        
        # Convert to PlannedTask objects, organized by file_path
        # Enforce files_order: re-sort tasks so that file order is respected,
        # while preserving LLM's within-file task ordering.
        file_order_index = {fp: i for i, fp in enumerate(self.files_order)}
        sorted_tasks = sorted(
            enumerate(planned_tasks),
            key=lambda pair: (
                file_order_index.get(pair[1]["file_path"], 999),
                pair[0],  # preserve original LLM order within same file
            ),
        )

        tasks_by_file: Dict[str, List[PlannedTask]] = {}
        for priority_idx, (_orig_idx, task_data) in enumerate(sorted_tasks):
            fp = task_data["file_path"]
            t_unit_keys = task_data["units"]
            fdata = self.files_data[fp]
            t_unit_to_code = {u: fdata["units_to_code"][u] for u in t_unit_keys}
            t_unit_to_features = {u: fdata["units_to_features"][u] for u in t_unit_keys}
            
            planned_task = PlannedTask(
                task=task_data["task"],
                file_path=fp,
                units_key=t_unit_keys,
                unit_to_code=t_unit_to_code,
                unit_to_features=t_unit_to_features,
                priority=priority_idx,
                subtree=self.subtree
            )
            tasks_by_file.setdefault(fp, []).append(planned_task)
        
        total_tasks = sum(len(t) for t in tasks_by_file.values())
        return {
            "planned_tasks": tasks_by_file,
            "success": total_tasks > 0,
            "subtree": self.subtree
        }


# ============================================================================
# Helpers (B4 — entry-file hint based on project_types)
# ============================================================================

def _format_entry_file_hint(project_types: List[str]) -> str:
    """Render a per-project-type hint about the entry filename.

    Returns a single-line string appended to the "main.py" bullet of the
    main-entry task description. The wording stays advisory — the agent
    keeps freedom to choose ``main.py`` for any project — but flags more
    idiomatic alternatives for SERVICE / API / PIPELINE / GAME / GUI.
    Empty ``project_types`` falls back to a neutral hint.
    """
    if not project_types:
        return (
            "If a different filename better expresses the project's purpose "
            "(e.g. `app.py`, `server.py`, `pipeline.py`), you may use it "
            "instead — name the file by intent."
        )

    types = set(project_types)
    if "SERVICE" in types or "API" in types:
        return (
            "For service/API projects, `app.py` or `server.py` is often a "
            "more conventional name — pick whichever expresses intent best."
        )
    if "PIPELINE" in types:
        return (
            "For data pipelines / batch jobs, name the file after the job "
            "(e.g. `pipeline.py`, `dag.py`, `train.py`) instead of `main.py`."
        )
    if "GAME" in types:
        return (
            "For games, `main.py` is fine, but `game.py` or `play.py` are "
            "also acceptable. Pick what reads naturally to a new contributor."
        )
    if "LIBRARY" in types and not (types & {"CLI", "WEB", "GUI", "GAME"}):
        return (
            "This project is primarily a library; an entry point is "
            "optional. If you create one, prefer a thin CLI demonstrator "
            "(e.g. `examples/run.py`)."
        )
    # WEB / CLI / GUI all use main.py idiomatically.
    return "`main.py` is the right choice for this project."


# ============================================================================
# Task Planner (orchestrator)
# ============================================================================

class TaskPlanner:
    """Plans implementation tasks from interfaces using LLM."""
    
    def __init__(
        self,
        interfaces: Dict[str, Any],
        data_flow: Dict[str, Any],
        repo_name: str = "",
        repo_info: str = "",
        debug: bool = False,
        trajectory: Optional[Trajectory] = None
    ):
        self.interfaces = interfaces
        self.data_flow = data_flow
        self.repo_name = repo_name
        self.repo_info = repo_info
        self.debug = debug
        self.trajectory = trajectory
        self.primary_language = (
            extract_language_metadata(interfaces)[0]
            or extract_language_metadata(data_flow)[0]
        )
        self.backend = get_backend(self.primary_language)
        self.llm: Optional[LLMClient] = None
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.planned_tasks_dict: Dict[str, Dict[str, List[PlannedTask]]] = {}
        self.agent_results_dict: Dict[str, Dict[str, Any]] = {}
        self.file_order_diagnostics: Dict[str, Dict[str, Any]] = {}
    
    def plan(self) -> Dict[str, Any]:
        """Create implementation task plan using LLM."""
        # Ensure repo directory exists (LLMClient needs it for session management,
        # but plan_tasks may run before init_codebase creates it)
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        
        # Add step to trajectory
        step_id = None
        if self.trajectory:
            step_id = self.trajectory.add_step(
                "plan_tasks",
                description="Create implementation tasks from interfaces using LLM"
            )
            self.trajectory.start_step(step_id)
        
        self.llm = LLMClient(trajectory=self.trajectory, step_id=step_id)
        
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   PLANNING IMPLEMENTATION TASKS (LLM-based)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Get subtree order from data_flow
        subtree_order = self.data_flow.get("subtree_order", [])
        print(f"\n   Subtree order: {subtree_order}")
        
        # Get subtrees data from interfaces
        subtrees_data = self.interfaces.get("subtrees", {})
        
        total_files = 0
        total_units = 0
        
        for subtree in subtree_order:
            if subtree not in subtrees_data:
                self.logger.warning(f"Subtree {subtree} not found in interfaces")
                continue
            
            subtree_dict = subtrees_data[subtree]
            self.planned_tasks_dict[subtree] = {}
            self.agent_results_dict[subtree] = {}
            
            # Get files order and interfaces
            subtree_interfaces = subtree_dict.get("interfaces", {})
            files_order = subtree_dict.get("files_order", list(subtree_interfaces.keys()))
            corrected_files_order, order_diagnostics = correct_intra_subtree_file_order(
                subtree_name=subtree,
                files_order=files_order,
                subtree_interfaces=subtree_interfaces,
                logger=self.logger,
                language=self.primary_language,
            )
            self.file_order_diagnostics[subtree] = order_diagnostics
            
            print(f"\n   Subtree: {subtree} ({len(corrected_files_order)} files)")
            if order_diagnostics["changed"]:
                print("      ↺ Corrected file order from interface imports")
            
            # Collect all files data for this subtree
            files_data: Dict[str, Dict[str, Any]] = {}
            for file_path in corrected_files_order:
                if file_path not in subtree_interfaces:
                    continue
                
                file_dict = subtree_interfaces[file_path]
                file_code = file_dict.get("file_code", "")
                units_to_code = file_dict.get("units_to_code", {})
                units_to_features = file_dict.get("units_to_features", {})
                
                if not units_to_code:
                    self.logger.warning(f"No units found for {file_path}")
                    continue
                
                total_files += 1
                total_units += len(units_to_code)
                
                print(f"      -  {file_path}: {len(units_to_code)} units")
                
                files_data[file_path] = {
                    "file_code": file_code,
                    "units_to_code": units_to_code,
                    "units_to_features": units_to_features,
                }
            
            if not files_data:
                self.logger.warning(f"No files with units found for subtree {subtree}")
                continue
            
            # Create agent and plan entire subtree at once
            agent = TaskPlannerAgent(
                llm_client=self.llm,
                subtree=subtree,
                files_data=files_data,
                files_order=[fp for fp in corrected_files_order if fp in files_data],
                repo_name=self.repo_name,
                repo_info=self.repo_info,
                logger=self.logger
            )
            
            result = agent.plan_subtree_tasks(
                max_retry=5,
                max_steps=5
            )
            
            if result["success"]:
                # Reassign priorities based on topological file order so that
                # files depended-on by others are implemented first.
                file_order_map = {
                    fp: idx
                    for idx, fp in enumerate(corrected_files_order)
                    if fp in files_data
                }
                for fp, tasks in result["planned_tasks"].items():
                    base_priority = file_order_map.get(fp, 999)
                    for i, task in enumerate(tasks):
                        task.priority = base_priority * 100 + i
                    self.planned_tasks_dict[subtree][fp] = tasks
                total_planned = sum(len(t) for t in result["planned_tasks"].values())
                print(f"      [OK] {total_planned} tasks planned for subtree")
            else:
                self.logger.error(f"Failed to plan tasks for subtree {subtree}")
                print("      [FAIL] Planning failed for subtree")
        
        # Serialize results (ZeroRepo compatible format)
        # Filter out tasks with empty units_key (e.g., __init__.py files
        # that the LLM planned but have no units to implement)
        planned_tasks_serializable = {}
        for subtree, files_dict in self.planned_tasks_dict.items():
            planned_tasks_serializable[subtree] = {}
            for file_path, tasks_list in files_dict.items():
                valid_tasks = [
                    task.to_dict() for task in tasks_list
                    if task.units_key  # skip empty
                ]
                if valid_tasks:
                    planned_tasks_serializable[subtree][file_path] = valid_tasks
        
        result = {
            "meta": metadata_with_languages(
                self.interfaces
                if extract_language_metadata(self.interfaces)[0]
                else self.data_flow
            ),
            "planned_tasks_dict": planned_tasks_serializable,
            "agent_results_dict": self.agent_results_dict,
            "file_order_diagnostics": self.file_order_diagnostics,
            "subtree_order": subtree_order,
            "success": True
        }
        
        # Count total tasks
        total_tasks = sum(
            len(tasks) 
            for files_dict in self.planned_tasks_dict.values() 
            for tasks in files_dict.values()
        )
        
        # Complete step
        if self.trajectory and step_id:
            self.trajectory.complete_step(step_id, metadata={
                "total_tasks": total_tasks,
                "total_units": total_units,
                "total_files": total_files
            })
        
        print(f"\n   Planned {total_tasks} tasks for {total_units} units across {total_files} files")
        
        # Add integration tests and documentation tasks
        self._add_special_tasks(planned_tasks_serializable, self.agent_results_dict, subtree_order)
        
        # Add project file tasks (after all core implementation)
        self._add_project_file_tasks(planned_tasks_serializable, self.agent_results_dict)
        
        # Update subtree order to include special and project files
        updated_subtree_order = subtree_order + ["FINAL_TASKS", "PROJECT_FILES"]
        
        result = {
            "meta": metadata_with_languages(
                self.interfaces
                if extract_language_metadata(self.interfaces)[0]
                else self.data_flow
            ),
            "planned_tasks_dict": planned_tasks_serializable,
            "agent_results_dict": self.agent_results_dict,
            "file_order_diagnostics": self.file_order_diagnostics,
            "subtree_order": updated_subtree_order,
            "success": True
        }
        
        # Recount total tasks including project files
        total_tasks = sum(
            len(tasks) 
            for files_dict in planned_tasks_serializable.values() 
            for tasks in files_dict.values()
        )
        
        print(f"   [OK] Added project file tasks (total tasks: {total_tasks})")
        
        return result
    
    def _add_special_tasks(
        self,
        planned_tasks: Dict,
        agent_results: Dict,
        subtree_order: List[str]
    ):
        """Add integration test and documentation tasks (ZeroRepo compatible)."""
        print("\n   Adding integration test and documentation tasks...")
        
        # Add integration test task for each subtree
        for subtree in subtree_order:
            if subtree in planned_tasks:
                # Get subtree path information
                subtree_path = self._get_subtree_path(subtree)
                
                integration_test_task = PlannedTask(
                    task=(
                        f"Write comprehensive integration tests for the {subtree} module "
                        f"{f'located in {subtree_path}' if subtree_path else ''}. "
                        f"Test the interactions between all components in this module, "
                        f"verify data flow, error handling, and edge cases, and ensure all public APIs "
                        f"work correctly together. "
                        f"Focus on testing the integration points between different files in this module. "
                        f"In addition to the integration tests, create small, focused usage examples for this module "
                        f"(e.g., example scripts or functions) that demonstrate typical end-to-end usage of its main APIs. "
                        f"Create appropriate test files and example files in the module directory or the test/example "
                        f"directory as needed, following the existing project conventions."
                    ),
                    file_path="<INTEGRATION_TEST>",  # Special marker - let agent decide placement
                    units_key=[f"{subtree}_integration_tests"],
                    unit_to_code={f"{subtree}_integration_tests": f"# Integration tests for {subtree} module"},
                    unit_to_features={f"{subtree}_integration_tests": [f"{subtree} integration testing"]},
                    priority=1000,  # Lower priority (higher number) - run after regular implementation
                    subtree=subtree,
                    task_type="integration_test",
                )
                
                # Add integration test to the subtree
                integration_file_path = f"<INTEGRATION_TEST>_{subtree}"
                planned_tasks[subtree][integration_file_path] = [integration_test_task.to_dict()]
                agent_results[subtree][integration_file_path] = {"success": True, "type": "integration_test"}
                
                self.logger.info(f"Added integration test task for subtree: {subtree} (path: {subtree_path})")
                print(f"      -  Added integration test task for subtree: {subtree}")
        
        # Create a special subtree for final tasks
        final_subtree = "FINAL_TASKS"
        planned_tasks[final_subtree] = {}
        agent_results[final_subtree] = {}
        
        # === Cross-module wiring verification task ===
        edges = self.data_flow.get("data_flow", [])
        if edges:
            edges_desc = "\n".join(
                f"  - {e.get('source', '?')} → {e.get('target', '?')}: "
                f"{e.get('data_type', 'N/A')}"
                for e in edges
            )
            wiring_task = PlannedTask(
                task=(
                    "Verify and fix cross-module wiring for all data flow edges.\n\n"
                    "The following data flow edges are defined in the system design. "
                    "For EACH edge, you must:\n"
                    "1. Read the source module's actual code (not just the skeleton)\n"
                    "2. Read the target module's actual code\n"
                    "3. Check if code exists that calls the source and passes results "
                    "to the target\n"
                    "4. If NOT connected or connected incorrectly, fix the production code\n"
                    "5. Write a test that verifies the connection works\n\n"
                    "Data Flow Edges (CHECKLIST):\n"
                    f"{edges_desc}\n\n"
                    "Common wiring bugs to fix:\n"
                    "- Route handler returning placeholder string instead of calling "
                    "the real handler\n"
                    "- Module A defines function but Module B never imports/calls it\n"
                    "- Data format mismatch at module boundary\n"
                    "- CSS class names in templates not matching stylesheet definitions\n"
                    "\nDo NOT create the main entry point — it will be created in a later task."
                ),
                file_path="<WIRING>",
                units_key=["cross_module_wiring"],
                unit_to_code={"cross_module_wiring": "# Cross-module wiring verification"},
                unit_to_features={"cross_module_wiring": [
                    "cross-module data flow wiring"
                ]},
                priority=1500,
                subtree=final_subtree,
                task_type="wiring",
            )
            planned_tasks[final_subtree]["<WIRING>"] = [wiring_task.to_dict()]
            agent_results[final_subtree]["<WIRING>"] = {
                "success": True, "type": "wiring"
            }
            self.logger.info(
                "Added cross-module wiring task with %d data flow edges", len(edges)
            )
            print(f"      -  Added cross-module wiring verification task "
                  f"({len(edges)} edges)")

        # === UI Polish task ===
        ui_polish_task = PlannedTask(
            task=(
                "Review and improve the user-facing interface of this application.\n\n"
                "FIRST: Determine what type of user interface this project has:\n"
                "- Web application (HTML pages, templates, CSS)\n"
                "- GUI application (tkinter, PyQt, pygame, etc.)\n"
                "- CLI tool (terminal output, argument parsing)\n"
                "- Library with no direct UI\n\n"
                "If it is a pure library with no user-facing output, skip this "
                "task — commit an empty change and report PASS.\n\n"
                "For ALL other project types, follow these steps:\n\n"
                "## Step 1: Inventory existing assets\n"
                "List all files related to user-facing output:\n"
                "- Style, theme, or presentation files when the project type uses them\n"
                "- Template/page/view files\n"
                "- Layout/component files\n"
                "- Static assets directory\n"
                "If any necessary files are MISSING (e.g., no CSS file exists "
                "but HTML references styles), CREATE them.\n\n"
                "## Step 2: Audit every user-facing output\n"
                "**For web apps:**\n"
                "- Does every page use the shared layout (head+CSS, nav, footer)?\n"
                "- Do HTML class names match the CSS definitions exactly?\n"
                "- Is content in proper containers? Are forms styled?\n"
                "- Are all navigation links correct and complete?\n\n"
                "**For GUI apps:**\n"
                "- Is there consistent widget styling and layout?\n"
                "- Are windows properly sized with sensible defaults?\n"
                "- Is there a menu bar or toolbar for navigation?\n\n"
                "**For CLI tools:**\n"
                "- Is output well-formatted with aligned columns?\n"
                "- Are error messages clear and helpful?\n"
                "- Does --help show all commands with descriptions?\n"
                "- Are long operations showing progress?\n\n"
                "## Step 3: Fix all issues\n"
                "- Create missing style/template/static files if needed\n"
                "- Fix class name mismatches between HTML and CSS\n"
                "- Add missing layout wrapping to bare pages\n"
                "- Replace placeholder/stub responses with real renderers\n"
                "- Ensure consistent look across all pages/screens/commands\n\n"
                "## Step 4: Verify\n"
                "- Web: test client requests → check for <style>/<link>, "
                "<nav>, response >500 bytes\n"
                "- GUI: verify window opens without errors\n"
                "- CLI: verify --help output and a basic command run\n"
                "- Write tests that assert key structural elements\n\n"
                "Do NOT create the main entry point — it will be created in a later task."
            ),
            file_path="<UI_POLISH>",
            units_key=["ui_polish"],
            unit_to_code={"ui_polish": "# UI polish and consistency"},
            unit_to_features={"ui_polish": ["UI consistency and styling"]},
            priority=1800,
            subtree=final_subtree,
            task_type="ui_polish",
        )
        planned_tasks[final_subtree]["<UI_POLISH>"] = [
            ui_polish_task.to_dict()
        ]
        agent_results[final_subtree]["<UI_POLISH>"] = {
            "success": True, "type": "ui_polish"
        }
        self.logger.info("Added UI polish task")
        print("      -  Added UI polish task")

        # Add final comprehensive test and documentation task
        comprehensive_test_task = PlannedTask(
            task=(
                "Design and implement comprehensive end-to-end tests for the entire repository. "
                "Cover system tests, integration tests across all major modules, performance scenarios, "
                "and critical edge cases. "
                "Verify the complete workflow from input to output and validate interactions between modules "
                "so the system works reliably as a whole. "
                "In addition, create clear usage examples (e.g., example scripts or notebooks) that demonstrate "
                "typical end-to-end workflows. "
                "Place the new test files and examples in appropriate locations in the project structure. "
                "NOTE: The main entry point will be created in the next task — "
                "do NOT create it here."
            ),
            file_path="<COMPREHENSIVE_TEST>",  # Special marker - let agent decide placement
            units_key=["comprehensive_tests"],
            unit_to_code={"comprehensive_tests": "# Comprehensive end-to-end tests"},
            unit_to_features={"comprehensive_tests": ["comprehensive system testing"]},
            priority=2000,  # Lowest priority - run last
            subtree=final_subtree,
            task_type="final_test_docs",
        )
        
        # Add final tasks
        planned_tasks[final_subtree]["<COMPREHENSIVE_TEST>"] = [comprehensive_test_task.to_dict()]
        agent_results[final_subtree]["<COMPREHENSIVE_TEST>"] = {"success": True, "type": "final_test_docs"}
        
        self.logger.info("Added final comprehensive test and documentation tasks")
        print("      -  Added final comprehensive test and documentation task")

        # Main entry point task (part of core implementation, after comprehensive tests)
        main_task = PlannedTask(
            task=self._build_main_entry_task(),
            file_path="<MAIN_ENTRY>",
            units_key=["main_entry_generation"],
            unit_to_code={"main_entry_generation": "# Generate main entry point"},
            unit_to_features={"main_entry_generation": ["program entry point"]},
            priority=2100,  # After comprehensive tests (2000), before project files (3000)
            subtree=final_subtree,
            task_type="main_entry",  # Needs run test
        )
        planned_tasks[final_subtree]["<MAIN_ENTRY>"] = [main_task.to_dict()]
        agent_results[final_subtree]["<MAIN_ENTRY>"] = {"success": True, "type": "main_entry"}
        print("      -  Added main entry point task (with run test)")
    
    def _get_subtree_path(self, subtree_name: str) -> str:
        """Get the directory path for a subtree from available metadata."""
        try:
            # Try to infer from interfaces data - look at file paths in this subtree
            subtrees_data = self.interfaces.get("subtrees", {})
            if subtree_name in subtrees_data:
                subtree_dict = subtrees_data[subtree_name]
                subtree_interfaces = subtree_dict.get("interfaces", {})
                if subtree_interfaces:
                    # Get common prefix from file paths
                    file_paths = list(subtree_interfaces.keys())
                    if file_paths:
                        try:
                            common = commonpath(file_paths)
                            return common if common and common != "." else dirname(file_paths[0])
                        except ValueError:
                            return dirname(file_paths[0]) if file_paths else ""
            
            # Try to load from RPG file
            if RPG_FILE.exists():
                with open(RPG_FILE, 'r', encoding='utf-8') as f:
                    rpg_data = json.load(f)
                nodes = rpg_data.get("nodes", {})
                for node in nodes.values():
                    node_data = node if isinstance(node, dict) else {}
                    if (node_data.get("name") == subtree_name and
                            node_data.get("level") == 1):
                        meta = node_data.get("meta", {})
                        if meta and isinstance(meta, dict):
                            path = meta.get("path", "")
                            if path and path != ".":
                                return path if isinstance(path, str) else ", ".join(path)
                        break
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Failed to get subtree path for {subtree_name}: {e}")
            return ""
    
    def _add_project_file_tasks(
        self,
        planned_tasks: Dict,
        agent_results: Dict
    ):
        """Add project file tasks after all implementation and main entry tasks.
        
        These tasks are scheduled to run AFTER all core implementation is complete,
        so they can reference the actual code content.
        
        Task types:
        - project_requirements: language-specific dependency metadata
        - project_docs: README.md
        """
        print("\n   Adding project file tasks...")
        
        # Create a subtree for project files
        pf_subtree = "PROJECT_FILES"
        planned_tasks[pf_subtree] = {}
        agent_results[pf_subtree] = {}
        
        # 1. Requirements/Dependencies task (with import validation test)
        requirements_task = PlannedTask(
            task=self._build_requirements_task(),
            file_path="<REQUIREMENTS>",
            units_key=["requirements_generation"],
            unit_to_code={"requirements_generation": "# Generate dependency metadata"},
            unit_to_features={"requirements_generation": ["dependency management"]},
            priority=3000,  # After main_entry (2100)
            subtree=pf_subtree,
            task_type="project_requirements",  # Needs import validation test
        )
        planned_tasks[pf_subtree]["<REQUIREMENTS>"] = [requirements_task.to_dict()]
        agent_results[pf_subtree]["<REQUIREMENTS>"] = {"success": True, "type": "project_requirements"}
        print("      -  Added dependency metadata task (with validation test)")
        
        # 2. README documentation task (no tests needed)
        readme_task = PlannedTask(
            task=self._build_readme_task(),
            file_path="<README>",
            units_key=["readme_generation"],
            unit_to_code={"readme_generation": "# Generate README.md"},
            unit_to_features={"readme_generation": ["project documentation"]},
            priority=3100,  # After requirements
            subtree=pf_subtree,
            task_type="project_docs",  # No tests needed
        )
        planned_tasks[pf_subtree]["<README>"] = [readme_task.to_dict()]
        agent_results[pf_subtree]["<README>"] = {"success": True, "type": "project_docs"}
        print("      -  Added README.md generation task (no test)")
        
        self.logger.info("Added 2 project file tasks")

    def _backend_project_task_templates(self):
        """Return backend-owned project task templates when available."""
        return self.backend.project_task_templates(
            ProjectTaskContext(
                repo_name=self.repo_name,
                repo_info=self.repo_info,
                package_name=self._package_slug(separator="-"),
                entry_point_path=self._reconciled_entry_point_path(),
            )
        )

    def _reconciled_entry_point_path(self) -> Optional[str]:
        """Resolve the program entry path from already-designed interfaces.

        Reuses an existing language-appropriate entry file when the
        skeleton already placed one (e.g. C++ ``src/cli/main.cpp`` off the
        canonical ``src/main.cpp``, or Go ``cmd/<name>/main.go``), so the
        synthetic MAIN_ENTRY task extends it instead of generating a
        SECOND entry — the dual-``main`` bug. The per-language matching
        rule lives in ``backend.find_existing_entry`` (filename match by
        default; Go encodes the ``cmd/*/main.go`` shape). Returns ``None``
        when the skeleton declared no entry, letting the backend use its
        canonical path.
        """
        return self.backend.find_existing_entry(self.interfaces)
    
    def _build_requirements_task(self) -> str:
        """Build task description for dependency metadata generation."""
        templates = self._backend_project_task_templates()
        if templates is not None:
            return templates.dependencies

        if self.backend.name != "python":
            raise RuntimeError(f"{self.backend.name} backend must provide project task templates")

        return f"""Generate or update the dependency management files for the repository: {self.repo_name}

**Files to create/update:**
1. `requirements.txt` - Main dependencies
2. `requirements-dev.txt` (optional) - Development dependencies if needed

**Instructions:**
1. Analyze ALL Python files in the repository to identify:
   - Third-party imports (external packages)
   - Standard library imports (exclude these)
   - Internal project imports (exclude these)

2. For each third-party package:
   - Determine the correct PyPI package name
   - Specify version constraints if necessary (prefer `>=` for flexibility)
   - Group related packages together

3. Separate development dependencies if present:
   - Testing frameworks (pytest, unittest-mock, etc.)
   - Linting/formatting tools (black, flake8, mypy, etc.)
   - Documentation tools (sphinx, mkdocs, etc.)

4. Add comments explaining package purposes

**Output format for requirements.txt:**
```
# Core dependencies
package1>=1.0.0
package2>=2.0.0

# Optional dependencies
package3>=3.0.0  # For feature X
```

**Important:**
- Do NOT include standard library modules
- Do NOT include the project's own modules
- Verify package names match PyPI exactly
- If no external dependencies are found, create an empty requirements.txt with a comment
"""
    
    def _build_main_entry_task(self) -> str:
        """Build task description for main entry point generation."""
        templates = self._backend_project_task_templates()
        if templates is not None:
            return templates.main_entry

        if self.backend.name != "python":
            raise RuntimeError(f"{self.backend.name} backend must provide project task templates")

        # Infer the main package name from the interfaces subtree structure
        package_name = self._get_package_name()

        # Read project_types so we can hint at the right entry-file shape
        # without locking the agent into "main.py" for SERVICE/PIPELINE
        # projects. Failure to load is non-fatal — fall back to
        # the generic guidance.
        project_types = self._load_project_types()
        entry_hint = _format_entry_file_hint(project_types)

        return f"""Create the main entry point for the repository: {self.repo_name}
Repository purpose: {self.repo_info}

**Goal:** Create a production-quality entry point that lets users run the COMPLETE
software product — not a toy example or demo script.  If this is an interactive
project, users should be able to interact with every feature through this entry
point.

**Files to create:**
1. `main.py` — Default entry point in the project root.
   {entry_hint}
2. `{package_name}/__main__.py` (optional) — For `python -m {package_name}` usage

**Critical Rules:**
- **Do NOT re-implement any functionality or data structures.** Import and use the
  classes, functions, and data structures already defined in the project modules.
- Analyze the actual implemented code thoroughly before writing anything.
- Every import must reference real, existing modules and symbols.
- All features exposed through the entry point must delegate to existing module
  APIs — do not duplicate logic.

**Requirements:**
1. Import core functionality from the implemented modules
2. Provide a clean CLI interface (argparse or similar) with intuitive commands
   and options that expose ALL major features of the project
3. Include `--help` text with usage examples
4. Handle common error cases gracefully (missing args, invalid input, runtime
   errors) with user-friendly messages
5. If the project is interactive (game, REPL, UI), launch the full interactive
   session by default — not a one-shot demo
6. If the project is a processing tool or library, expose the primary pipeline
   end-to-end, reading input and producing output

**Suggested structure:**
```python
#!/usr/bin/env python3
\"\"\"{{repo_name}} — Main Entry Point

{{repo_info}}
\"\"\"\n
import argparse
import sys
from typing import Optional

# Import core modules — use actual module names from the codebase
# from {package_name} import ...

def main(args: Optional[list] = None) -> int:
    \"\"\"Main entry point.\"\"\"\n
    parser = argparse.ArgumentParser(
        description='{{repo_info}}'
    )
    # Add subcommands / arguments based on real functionality
    # ...

    parsed_args = parser.parse_args(args)
    # Delegate to existing module APIs
    # ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Important:**
- Reference ONLY actual module names, classes, and functions from the codebase
- Provide meaningful default behaviors so `python main.py` does something useful
- The entry point should feel like a finished product, not a scaffold
- **Make `python main.py` work from a clean checkout.** If the package lives
  under `src/` (e.g. `src/{package_name}/`), a bare `python main.py` will raise
  `ModuleNotFoundError` because `src/` is not on `sys.path`. You MUST make the
  import resolvable by ONE of:
    1. Adding a `pyproject.toml` with `[tool.setuptools] packages` discovery
       under `src` (`package-dir = {{"" = "src"}}`), so an editable/normal
       install exposes the package; OR
    2. Inserting a path bridge at the very top of `main.py`, before importing
       the package:
       `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))`
  Prefer (1) for installable projects; (2) is the minimal always-works bridge.
  Do NOT rely on the caller exporting `PYTHONPATH`.
- **Read the `docs/` directory first** — it contains the user's original requirements
  and feature specifications. Make sure the entry point faithfully exposes
  all requested features and does NOT deviate from the intended purpose.
"""

    def _load_project_types(self) -> List[str]:
        """Load ``feature_spec.meta.project_types`` if available.

        Returns an empty list when the file is missing, malformed, or has
        no valid tokens. Callers must handle the empty case.
        """
        try:
            from common.paths import FEATURE_SPEC_FILE
            from common.project_types import validate_project_types
            if not FEATURE_SPEC_FILE.exists():
                return []
            with open(FEATURE_SPEC_FILE, "r", encoding="utf-8") as f:
                spec = json.load(f)
            meta = spec.get("meta") or {}
            types, _ = validate_project_types(meta)
            return types
        except Exception as exc:
            self.logger.debug("project_types lookup failed: %s", exc)
            return []

    def _get_package_name(self) -> str:
        """Infer the main Python package name from actual file paths in interfaces."""
        subtrees_data = self.interfaces.get("subtrees", {})

        # Collect all source file paths across all subtrees
        all_paths: List[str] = []
        for st_data in subtrees_data.values():
            for fpath in st_data.get("interfaces", {}):
                all_paths.append(fpath)

        if all_paths:
            # Find the top-level package: take the first directory component
            # after 'src/' (if present), or the first directory component.
            # e.g. "src/personal_blog_system/models/post.py" → "personal_blog_system"
            # e.g. "myapp/core/engine.py" → "myapp"
            candidates: List[str] = []
            for p in all_paths:
                parts = p.replace("\\", "/").split("/")
                # Skip 'src' prefix if present
                if parts and parts[0] == "src":
                    parts = parts[1:]
                if len(parts) >= 2:  # need at least package/file.py
                    candidates.append(parts[0])

            if candidates:
                # Return the most common top-level package name
                counter = Counter(candidates)
                return counter.most_common(1)[0][0]

        # Fallback to repo name in snake_case
        if self.repo_name:
            return self.repo_name.lower().replace("-", "_").replace(" ", "_")
        return "project"

    def _package_slug(self, separator: str = "-") -> str:
        """Infer a compact package name from repository metadata."""
        raw = self.repo_name or "project"
        candidate = raw.lower().replace(" ", separator).replace("_", separator)
        candidate = _re.sub(rf"[^a-z0-9{_re.escape(separator)}]+", separator, candidate)
        candidate = _re.sub(rf"{_re.escape(separator)}+", separator, candidate)
        return candidate.strip(separator) or "project"

    def _build_readme_task(self) -> str:
        """Build task description for README.md generation."""
        templates = self._backend_project_task_templates()
        if templates is not None:
            return templates.readme

        if self.backend.name != "python":
            raise RuntimeError(f"{self.backend.name} backend must provide project task templates")

        return f"""Update the README.md for the repository: {self.repo_name}
Repository purpose: {self.repo_info}

**Goal:** Replace the placeholder README with comprehensive project documentation
based on the actual implemented code.

**Sections to include:**

## 1. Project Title & Description
- Clear, concise description of what the software does
- Key features and capabilities

## 2. Installation
- Prerequisites (Python version, system dependencies)
- Installation steps using requirements.txt
- Virtual environment setup instructions

## 3. Usage
- How to run the program (`python main.py` with actual flags/subcommands)
- Common usage examples with expected output
- Configuration options if applicable

## 4. Project Structure
- Brief overview of the directory layout
- Key modules and their purposes

## 5. Development
- How to run tests (`pytest`)
- Contributing guidelines (brief)

**Instructions:**
1. Read the `docs/` directory for the original requirements and feature specifications
2. Explore the actual codebase to understand what was implemented
3. Run `python main.py --help` to document the real CLI interface
4. Reference actual module names, classes, and function signatures
5. Ensure all examples are accurate and runnable

**Important:**
- Do NOT copy the placeholder README content — write fresh documentation
- Base everything on the ACTUAL implemented code, not assumptions
- Keep the tone professional and concise
- Use proper Markdown formatting
"""


# ============================================================================
# Utility Functions
# ============================================================================

def load_repo_info() -> tuple[str, str]:
    """Load repository name and info from available sources."""
    repo_name = ""
    repo_info = ""
    
    # Try repo_rpg.json first
    if RPG_FILE.exists():
        try:
            with open(RPG_FILE, 'r', encoding='utf-8') as f:
                rpg_data = json.load(f)
                repo_name = rpg_data.get("repo_name", "")
                repo_info = rpg_data.get("repo_info", "")
        except Exception:
            pass
    
    # Fallback to repo_info.json
    if not repo_info and REPO_INFO_FILE.exists():
        try:
            with open(REPO_INFO_FILE, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                repo_name = repo_name or info_data.get("name", "")
                repo_info = info_data.get("description", "")
        except Exception:
            pass
    
    return repo_name, repo_info


def _collect_skeleton_features(skeleton: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_features = node.get("feature_paths") or []
        if isinstance(node_features, list):
            features.update(str(feature) for feature in node_features if feature)
        for child in node.get("children") or []:
            visit(child)

    visit(skeleton.get("root"))
    return features


def _collect_interface_features(interfaces: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    subtrees = interfaces.get("subtrees") or interfaces.get("components") or {}
    if not isinstance(subtrees, dict):
        return features
    for subtree_data in subtrees.values():
        if not isinstance(subtree_data, dict):
            continue
        files = subtree_data.get("interfaces") or subtree_data.get("files") or {}
        if not isinstance(files, dict):
            continue
        for file_data in files.values():
            if not isinstance(file_data, dict):
                continue
            units_to_features = file_data.get("units_to_features") or {}
            if not isinstance(units_to_features, dict):
                continue
            for unit_features in units_to_features.values():
                if isinstance(unit_features, list):
                    features.update(str(feature) for feature in unit_features if feature)
    return features


def _validate_interfaces_cover_skeleton_features(
    interfaces: Dict[str, Any],
    skeleton_path: Path = SKELETON_FILE,
) -> None:
    if not skeleton_path.exists():
        return
    with open(skeleton_path, "r", encoding="utf-8") as f:
        skeleton = json.load(f)
    missing = sorted(
        _collect_skeleton_features(skeleton) - _collect_interface_features(interfaces)
    )
    if missing:
        preview = ", ".join(repr(feature) for feature in missing[:5])
        raise ValueError(
            "interfaces.json is incomplete for skeleton.json: "
            f"{len(missing)} feature(s) missing from interfaces.json: "
            f"[{preview}]. Re-run design_interfaces before plan_tasks."
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plan implementation tasks from interfaces using LLM"
    )
    parser.add_argument(
        "--interfaces", "-i",
        type=Path,
        default=INTERFACES_FILE,
        help=f"Input interfaces file (default: {INTERFACES_FILE})"
    )
    parser.add_argument(
        "--data-flow", "-d",
        type=Path,
        default=DATA_FLOW_FILE,
        help=f"Input data flow file (default: {DATA_FLOW_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output tasks file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording"
    )
    
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       PLAN TASKS - Implementation Level                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Load interfaces
    if not args.interfaces.exists():
        print(f"\n   Error: Interfaces file not found: {args.interfaces}")
        print("   Please run /cmind.design_interfaces first.")
        return 1
    
    # Load data_flow
    if not args.data_flow.exists():
        print(f"\n   Error: Data flow file not found: {args.data_flow}")
        print("   Please run /cmind.build_data_flow first.")
        return 1
    
    print(f"\n   Loading interfaces from: {args.interfaces}")
    print(f"   Loading data flow from: {args.data_flow}")
    
    with open(args.interfaces, 'r', encoding='utf-8') as f:
        interfaces = json.load(f)
    _validate_interfaces_cover_skeleton_features(interfaces)
    
    with open(args.data_flow, 'r', encoding='utf-8') as f:
        data_flow = json.load(f)
    
    # Load repo info
    repo_name, repo_info = load_repo_info()
    print(f"   Repository: {repo_name or '(unknown)'}")
    
    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("plan_tasks")
        trajectory.start(metadata={
            "interfaces_file": str(args.interfaces),
            "data_flow_file": str(args.data_flow),
            "output_file": str(args.output)
        })
    
    try:
        # Plan tasks
        planner = TaskPlanner(
            interfaces=interfaces,
            data_flow=data_flow,
            repo_name=repo_name,
            repo_info=repo_info,
            debug=args.verbose,
            trajectory=trajectory
        )
        result = planner.plan()
        
        # Save output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n   [OK] Tasks saved to: {args.output.name}")
        
        # Complete trajectory
        if trajectory:
            trajectory.complete(metadata={"success": True})
            print("   [OK] Trajectory saved")
        
        # Final summary
        total_tasks = sum(
            len(tasks) 
            for files_dict in result.get("planned_tasks_dict", {}).values() 
            for tasks in files_dict.values()
        )
        
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              TASK PLANNING COMPLETE                          ║")
        print(f"║  Total Tasks: {total_tasks:>4}                                        ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        print("\n   [PIN] Implementation can now begin!")
        print("   Use the tasks.json to guide the implementation order.")
        
        return 0

    except Exception as e:
        if trajectory:
            trajectory.fail(str(e))
        raise


if __name__ == "__main__":
    exit(main())
