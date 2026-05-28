#!/usr/bin/env python3
"""Trajectory Recording Module for CoderMind.

This module provides utilities for recording command execution trajectories,
including:
- Step-by-step execution status (pending, in_progress, completed, failed)
- Script invocations with commands and outputs
- LLM interactions (prompts and responses)
- Component/target states
- Resume support for interrupted executions

Each command (build_feature, refactor_feature, build_skeleton, etc.) 
maintains its own trajectory file in .cmind/trajectory/
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from .paths import TRAJECTORY_DIR
from .paths import WORKSPACE_ROOT


# ============================================================================
# Enums and Constants
# ============================================================================

class StepStatus(str, Enum):
    """Status of a step in the trajectory."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CommandStatus(str, Enum):
    """Status of the overall command execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ScriptCall:
    """Record of a script/command invocation."""
    command: str
    started_at: str
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScriptCall":
        return cls(**data)


@dataclass
class LLMInteraction:
    """Record of a single LLM interaction."""
    interaction_id: int
    timestamp: str
    purpose: str  # e.g., "generate_structure", "assign_features"
    prompt: str
    response: Optional[str] = None
    parsed_result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMInteraction":
        return cls(**data)


@dataclass
class Step:
    """Record of a step in the command execution."""
    step_id: int
    name: str
    description: str = ""
    status: str = StepStatus.PENDING.value
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    script_calls: List[ScriptCall] = field(default_factory=list)
    llm_interactions: List[LLMInteraction] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "script_calls": [sc.to_dict() for sc in self.script_calls],
            "llm_interactions": [li.to_dict() for li in self.llm_interactions],
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        step = cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data.get("description", ""),
            status=data.get("status", StepStatus.PENDING.value),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )
        step.script_calls = [ScriptCall.from_dict(sc) for sc in data.get("script_calls", [])]
        step.llm_interactions = [LLMInteraction.from_dict(li) for li in data.get("llm_interactions", [])]
        return step


@dataclass
class TargetState:
    """State of a target (component, file, etc.)."""
    name: str
    status: str = StepStatus.PENDING.value
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetState":
        return cls(**data)


@dataclass
class ResumePoint:
    """Information needed to resume an interrupted execution."""
    step_id: int
    step_name: str
    target: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumePoint":
        return cls(**data)


# ============================================================================
# Main Trajectory Class
# ============================================================================

class Trajectory:
    """Manages trajectory recording for a single command execution.
    
    Usage:
        traj = Trajectory("build_skeleton")
        traj.start()
        
        step = traj.add_step("generate_structure", "Generate directory structure")
        traj.start_step(step.step_id)
        
        # Record LLM interaction
        interaction_id = traj.start_llm_interaction(step.step_id, "generate_structure", prompt)
        traj.complete_llm_interaction(step.step_id, interaction_id, response, parsed_result)
        
        traj.complete_step(step.step_id)
        traj.complete()
    """
    
    def __init__(self, command_name: str, base_dir: Path = None):
        """Initialize a trajectory for a command.
        
        Args:
            command_name: Name of the command (e.g., "build_skeleton")
            base_dir: Base directory for trajectory files (default: current dir)
        """
        self.command_name = command_name
        self.base_dir = Path(base_dir) if base_dir else WORKSPACE_ROOT
        self.trajectory_dir = self.base_dir / TRAJECTORY_DIR
        
        # Generate filename with timestamp (human-readable, to seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_file = self.trajectory_dir / f"{command_name}_trajectory_{timestamp}.json"
        
        # Trajectory data
        self.status: str = CommandStatus.NOT_STARTED.value
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.steps: List[Step] = []
        self.targets_state: Dict[str, TargetState] = {}
        self.resume_point: Optional[ResumePoint] = None
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
        # Runtime tracking
        self._llm_interaction_counter = 0
        self._step_counter = 0
        self.logger = logging.getLogger(__name__)
    
    # ========================================================================
    # File Operations
    # ========================================================================
    
    def exists(self) -> bool:
        """Check if trajectory file already exists."""
        return self.trajectory_file.exists()
    
    def load(self) -> bool:
        """Load existing trajectory from file.
        
        Returns:
            True if loaded successfully, False if file doesn't exist or is invalid
        """
        if not self.trajectory_file.exists():
            return False
        
        try:
            with open(self.trajectory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.status = data.get("status", CommandStatus.NOT_STARTED.value)
            self.started_at = data.get("started_at")
            self.finished_at = data.get("finished_at")
            self.error = data.get("error")
            self.metadata = data.get("metadata", {})
            
            self.steps = [Step.from_dict(s) for s in data.get("steps", [])]
            self.targets_state = {
                k: TargetState.from_dict(v) 
                for k, v in data.get("targets_state", {}).items()
            }
            
            if data.get("resume_point"):
                self.resume_point = ResumePoint.from_dict(data["resume_point"])
            
            # Restore counters
            if self.steps:
                self._step_counter = max(s.step_id for s in self.steps)
            for step in self.steps:
                if step.llm_interactions:
                    max_id = max(li.interaction_id for li in step.llm_interactions)
                    self._llm_interaction_counter = max(self._llm_interaction_counter, max_id)
            
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to load trajectory: {e}")
            return False
    
    def save(self) -> None:
        """Save current trajectory to file."""
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "command": self.command_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
            "targets_state": {k: v.to_dict() for k, v in self.targets_state.items()},
            "resume_point": self.resume_point.to_dict() if self.resume_point else None
        }
        
        with open(self.trajectory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def delete(self) -> bool:
        """Delete the trajectory file."""
        if self.trajectory_file.exists():
            self.trajectory_file.unlink()
            return True
        return False
    
    # ========================================================================
    # Command Lifecycle
    # ========================================================================
    
    def start(self, metadata: Dict[str, Any] = None) -> None:
        """Start the command execution."""
        self.status = CommandStatus.IN_PROGRESS.value
        self.started_at = datetime.now().isoformat()
        self.finished_at = None
        self.error = None
        if metadata:
            self.metadata.update(metadata)
        self.save()
    
    def complete(self, metadata: Dict[str, Any] = None) -> None:
        """Mark command as successfully completed."""
        self.status = CommandStatus.COMPLETED.value
        self.finished_at = datetime.now().isoformat()
        self.resume_point = None
        if metadata:
            self.metadata.update(metadata)
        self.save()
    
    def fail(self, error: str, metadata: Dict[str, Any] = None) -> None:
        """Mark command as failed."""
        self.status = CommandStatus.FAILED.value
        self.finished_at = datetime.now().isoformat()
        self.error = error
        if metadata:
            self.metadata.update(metadata)
        self.save()
    
    def is_resumable(self) -> bool:
        """Check if this trajectory can be resumed."""
        return (
            self.status == CommandStatus.IN_PROGRESS.value and 
            self.resume_point is not None
        )
    
    # ========================================================================
    # Step Management
    # ========================================================================
    
    def add_step(self, name: str, description: str = "", metadata: Dict[str, Any] = None) -> Step:
        """Add a new step to the trajectory."""
        self._step_counter += 1
        step = Step(
            step_id=self._step_counter,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        self.steps.append(step)
        self.save()
        return step
    
    def get_step(self, step_id: int) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_step_by_name(self, name: str) -> Optional[Step]:
        """Get a step by its name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def start_step(self, step_id: int) -> None:
        """Mark a step as in progress."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.IN_PROGRESS.value
            step.started_at = datetime.now().isoformat()
            self.save()
    
    def complete_step(self, step_id: int, metadata: Dict[str, Any] = None) -> None:
        """Mark a step as completed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.COMPLETED.value
            step.finished_at = datetime.now().isoformat()
            if metadata:
                step.metadata.update(metadata)
            self.save()
    
    def fail_step(self, step_id: int, error: str) -> None:
        """Mark a step as failed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.FAILED.value
            step.finished_at = datetime.now().isoformat()
            step.error = error
            self.save()
    
    def skip_step(self, step_id: int, reason: str = "") -> None:
        """Mark a step as skipped."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.SKIPPED.value
            step.finished_at = datetime.now().isoformat()
            if reason:
                step.metadata["skip_reason"] = reason
            self.save()
    
    # ========================================================================
    # Script Call Recording
    # ========================================================================
    
    def record_script_start(self, step_id: int, command: str) -> int:
        """Record the start of a script call.
        
        Returns:
            Index of the script call in the step's script_calls list
        """
        step = self.get_step(step_id)
        if not step:
            return -1
        
        script_call = ScriptCall(
            command=command,
            started_at=datetime.now().isoformat()
        )
        step.script_calls.append(script_call)
        self.save()
        return len(step.script_calls) - 1
    
    def record_script_end(
        self, 
        step_id: int, 
        call_index: int,
        exit_code: int,
        stdout: str = "",
        stderr: str = ""
    ) -> None:
        """Record the completion of a script call."""
        step = self.get_step(step_id)
        if step and 0 <= call_index < len(step.script_calls):
            sc = step.script_calls[call_index]
            sc.finished_at = datetime.now().isoformat()
            sc.exit_code = exit_code
            sc.stdout = stdout
            sc.stderr = stderr
            self.save()
    
    # ========================================================================
    # LLM Interaction Recording
    # ========================================================================
    
    def start_llm_interaction(
        self, 
        step_id: int, 
        purpose: str, 
        prompt: str
    ) -> int:
        """Record the start of an LLM interaction.
        
        Returns:
            The interaction_id for this interaction
        """
        step = self.get_step(step_id)
        if not step:
            return -1
        
        self._llm_interaction_counter += 1
        interaction = LLMInteraction(
            interaction_id=self._llm_interaction_counter,
            timestamp=datetime.now().isoformat(),
            purpose=purpose,
            prompt=prompt
        )
        step.llm_interactions.append(interaction)
        self.save()
        return self._llm_interaction_counter
    
    def complete_llm_interaction(
        self,
        step_id: int,
        interaction_id: int,
        response: str,
        parsed_result: Dict[str, Any] = None,
        success: bool = True,
        error: str = None,
        duration_seconds: float = None
    ) -> None:
        """Record the completion of an LLM interaction."""
        step = self.get_step(step_id)
        if not step:
            return
        
        for interaction in step.llm_interactions:
            if interaction.interaction_id == interaction_id:
                interaction.response = response
                interaction.parsed_result = parsed_result
                interaction.success = success
                interaction.error = error
                interaction.duration_seconds = duration_seconds
                self.save()
                return
    
    # ========================================================================
    # Target State Management
    # ========================================================================
    
    def set_target_state(
        self, 
        target_name: str, 
        status: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Set the state of a target (component, file, etc.)."""
        self.targets_state[target_name] = TargetState(
            name=target_name,
            status=status,
            details=details or {}
        )
        self.save()
    
    def get_target_state(self, target_name: str) -> Optional[TargetState]:
        """Get the state of a target."""
        return self.targets_state.get(target_name)
    
    def update_target_details(self, target_name: str, details: Dict[str, Any]) -> None:
        """Update details for a target."""
        if target_name in self.targets_state:
            self.targets_state[target_name].details.update(details)
            self.save()
    
    # ========================================================================
    # Resume Point Management
    # ========================================================================
    
    def set_resume_point(
        self,
        step_id: int,
        step_name: str,
        target: str = None,
        context: Dict[str, Any] = None
    ) -> None:
        """Set a resume point for potential recovery."""
        self.resume_point = ResumePoint(
            step_id=step_id,
            step_name=step_name,
            target=target,
            context=context or {}
        )
        self.save()
    
    def clear_resume_point(self) -> None:
        """Clear the resume point."""
        self.resume_point = None
        self.save()
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the trajectory."""
        completed_steps = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED.value)
        failed_steps = sum(1 for s in self.steps if s.status == StepStatus.FAILED.value)
        total_llm_interactions = sum(len(s.llm_interactions) for s in self.steps)
        total_script_calls = sum(len(s.script_calls) for s in self.steps)
        
        return {
            "command": self.command_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_steps": len(self.steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_llm_interactions": total_llm_interactions,
            "total_script_calls": total_script_calls,
            "targets_count": len(self.targets_state),
            "is_resumable": self.is_resumable(),
            "error": self.error
        }
    
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Trajectory: {summary['command']}")
        print(f"{'='*60}")
        print(f"Status: {summary['status']}")
        print(f"Started: {summary['started_at'] or 'N/A'}")
        print(f"Finished: {summary['finished_at'] or 'N/A'}")
        print(f"Steps: {summary['completed_steps']}/{summary['total_steps']} completed")
        if summary['failed_steps'] > 0:
            print(f"Failed steps: {summary['failed_steps']}")
        print(f"LLM interactions: {summary['total_llm_interactions']}")
        print(f"Script calls: {summary['total_script_calls']}")
        if summary['is_resumable']:
            print(f"[WARNING] Can be resumed from: {self.resume_point.step_name}")
        if summary['error']:
            print(f"Error: {summary['error']}")
        print(f"{'='*60}\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def find_latest_trajectory(command_name: str, base_dir: Path = None) -> Optional[Path]:
    """Find the most recent trajectory file for a command.
    
    Returns the path to the latest trajectory file, or None if not found.
    """
    base = Path(base_dir) if base_dir else WORKSPACE_ROOT
    traj_dir = base / TRAJECTORY_DIR
    
    if not traj_dir.exists():
        return None
    
    # Find all trajectory files matching the pattern
    pattern = f"{command_name}_trajectory_*.json"
    files = list(traj_dir.glob(pattern))
    
    if not files:
        # Also check for old-style filename (without timestamp)
        old_file = traj_dir / f"{command_name}_trajectory.json"
        if old_file.exists():
            return old_file
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]


def load_or_create_trajectory(command_name: str, base_dir: Path = None) -> Trajectory:
    """Load an existing trajectory or create a new one.
    
    If an in-progress trajectory exists, it will be loaded for potential resume.
    Otherwise, a fresh trajectory will be created.
    """
    # First, check if there's an existing in-progress trajectory
    latest_file = find_latest_trajectory(command_name, base_dir)
    
    if latest_file:
        # Try to load and check if it's resumable
        traj = Trajectory(command_name, base_dir)
        traj.trajectory_file = latest_file  # Override with found file
        if traj.load() and traj.is_resumable():
            return traj
    
    # Create a new trajectory (with timestamp)
    return Trajectory(command_name, base_dir)


def get_trajectory_status(command_name: str, base_dir: Path = None) -> Dict[str, Any]:
    """Get the status of a trajectory without fully loading it."""
    traj = Trajectory(command_name, base_dir)
    if traj.load():
        return traj.get_summary()
    return {"command": command_name, "status": "not_found"}


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Demo/test usage
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        traj = Trajectory("test_command", Path(tmpdir))
        
        # Start command
        traj.start({"version": "1.0"})
        
        # Add and execute steps
        step1 = traj.add_step("check_input", "Validate input files")
        traj.start_step(step1.step_id)
        
        # Record script call
        call_idx = traj.record_script_start(step1.step_id, "python check.py --json")
        traj.record_script_end(step1.step_id, call_idx, 0, '{"valid": true}', "")
        
        traj.complete_step(step1.step_id)
        
        # Step with LLM interaction
        step2 = traj.add_step("generate", "Generate output")
        traj.start_step(step2.step_id)
        
        interaction_id = traj.start_llm_interaction(step2.step_id, "generate_code", "Write hello world")
        traj.complete_llm_interaction(
            step2.step_id, 
            interaction_id, 
            'print("Hello, World!")',
            {"language": "python"},
            success=True,
            duration_seconds=2.5
        )
        
        # Set target state
        traj.set_target_state("component_a", StepStatus.COMPLETED.value, {"files": 3})
        
        traj.complete_step(step2.step_id)
        
        # Complete command
        traj.complete()
        
        # Print summary
        traj.print_summary()
        
        # Load and verify
        traj2 = Trajectory("test_command", Path(tmpdir))
        traj2.load()
        print("Loaded trajectory summary:")
        traj2.print_summary()
