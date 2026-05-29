#!/usr/bin/env python3
"""PlannedTask Data Class for CoderMind.

Represents a single planned implementation task.
Each task contains one or more units from a single file to be
implemented together. Multiple tasks may be grouped into one
execution batch at runtime (see file-merge mode).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from rpg import uuid8


@dataclass
class PlannedTask:
    """Represents a single planned implementation task.
    
    This is the core unit of work in the code generation phase.
    Each task contains one or more units from a single file that
    should be implemented together.  At runtime, one or more
    PlannedTasks may be grouped into a single execution batch.
    """
    task_id: str = field(init=False)
    task: str                           # Task description (GitHub-style)
    file_path: str                      # Target file path
    units_key: List[str]                # List of unit keys to implement
    unit_to_code: Dict[str, str]        # Unit key -> interface code
    unit_to_features: Dict[str, List]   # Unit key -> feature list
    priority: int = 0                   # Execution priority (lower = earlier)
    subtree: str = ""                   # Subtree/component name
    task_type: str = "implementation"   # Task type:
                                        #   "implementation" - Core code implementation
                                        #   "integration_test" - Integration testing
                                        #   "final_test_docs" - Final tests and documentation
                                        #   "main_entry" - main.py entry point (run test)
                                        #   Project file types (after core + main entry):
                                        #   "project_requirements" - requirements.txt
                                        #       Needs import validation test
                                        #   "project_docs" - README.md
                                        #       No tests needed
    
    def __post_init__(self):
        """Generate unique task_id and validate inputs."""
        unique_suffix = uuid8()
        safe_path = self.file_path.replace('/', '_').replace('\\', '_')
        self.task_id = f"{safe_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_suffix}"
        
        # Validation
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedTask":
        """Create PlannedTask from dictionary."""
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
        # Restore original task_id if present
        if "task_id" in data:
            obj.task_id = data["task_id"]
        return obj
    
    def get_units_summary(self) -> str:
        """Get a summary of units in this task."""
        return ", ".join(self.units_key)
    
    def get_interface_code(self) -> str:
        """Get combined interface code for all units."""
        code_parts = []
        for unit_key in self.units_key:
            code = self.unit_to_code.get(unit_key, "")
            if code:
                code_parts.append(f"# {unit_key}\n{code}")
        return "\n\n".join(code_parts)




def load_tasks_from_tasks_json(tasks_path: Path) -> List[PlannedTask]:
    """Load all PlannedTask objects from tasks.json file.
    
    Args:
        tasks_path: Path to tasks.json file
        
    Returns:
        List of PlannedTask objects in execution order
    """
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")
    
    with open(tasks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    planned_tasks_dict = data.get("planned_tasks_dict", {})
    subtree_order = data.get("subtree_order", list(planned_tasks_dict.keys()))
    
    all_tasks = []
    
    for subtree in subtree_order:
        if subtree not in planned_tasks_dict:
            continue
        
        files_dict = planned_tasks_dict[subtree]
        
        # Get files in order (if available) or use dict order
        for file_path, task_list in files_dict.items():
            for task_data in task_list:
                # Skip tasks with empty units_key (e.g., __init__.py package
                # files that the LLM planned but have nothing to implement)
                if not task_data.get("units_key"):
                    continue
                try:
                    task = PlannedTask.from_dict(task_data)
                    all_tasks.append(task)
                except Exception as e:
                    print(f"Warning: Failed to load task from {file_path}: {e}")
    
    return all_tasks


def get_task_by_id(tasks_path: Path, task_id: str) -> Optional[PlannedTask]:
    """Get a specific PlannedTask by its task_id.
    
    Args:
        tasks_path: Path to tasks.json file
        task_id: The task_id to find
        
    Returns:
        PlannedTask if found, None otherwise
    """
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    for t in all_tasks:
        if t.task_id == task_id:
            return t
    return None


def get_next_pending_task(
    tasks_path: Path,
    completed_ids: List[str]
) -> Optional[PlannedTask]:
    """Get the next task that hasn't been completed yet.
    
    Args:
        tasks_path: Path to tasks.json file
        completed_ids: List of already completed task IDs
        
    Returns:
        Next pending PlannedTask, or None if all completed
    """
    all_tasks = load_tasks_from_tasks_json(tasks_path)
    completed_set = set(completed_ids)
    
    for t in all_tasks:
        if t.task_id not in completed_set:
            return t
    
    return None
