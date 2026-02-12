"""
TaskBatch data class for representing a batch of tasks
"""
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import uuid

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class TaskBatch:
    """Represents a batch of tasks to be implemented together"""
    # Auto-generated attribute: not required during construction; accessible via self.task_id after initialization
    task_id: str = field(init=False)

    task: str
    file_path: str
    units_key: List[str]
    unit_to_code: Dict[str, str]
    unit_to_features: Dict[str, List]
    priority: int = 0  # Higher priority = implement earlier
    subtree: str = ""  # Which functional area this belongs to (e.g., "Algorithms", "Data Processing")
    task_type: str = "implementation"  # Task type: "implementation", "integration_test", "final_test_docs"

    def __post_init__(self):
      
        unique_suffix = uuid.uuid4().hex[:8]
        self.task_id = f"{self.file_path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_suffix}"

        # basic type sanity (optional but helpful)
        if not isinstance(self.units_key, list) or not self.units_key:
            raise ValueError("TaskBatch validation error: 'units_key' must be a non-empty list.")

        missing_in_code = [k for k in self.units_key if k not in self.unit_to_code]
        missing_in_features = [k for k in self.units_key if k not in self.unit_to_features]

        if missing_in_code or missing_in_features:
            msg_parts = ["TaskBatch validation error: units_key contains keys not present in mappings."]
            if missing_in_code:
                msg_parts.append(f"Missing in unit_to_code: {missing_in_code}")
            if missing_in_features:
                msg_parts.append(f"Missing in unit_to_features: {missing_in_features}")
            raise ValueError(" ".join(msg_parts))

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
    def from_dict(cls, input: Dict):
      
        obj = cls(
            task=input["task"],
            file_path=input["file_path"],
            units_key=input["units_key"],
            unit_to_code=input["unit_to_code"],
            unit_to_features=input["unit_to_features"],
            priority=input.get("priority", 0),
            subtree=input.get("subtree", ""),
            task_type=input.get("task_type", "implementation"),
        )
        
        if "task_id" in input:
            obj.task_id = input["task_id"]
        return obj