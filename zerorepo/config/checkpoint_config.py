"""
Centralized Checkpoint File Configuration

This module provides a unified approach to managing all checkpoint files
used throughout the ZeroRepo system. It eliminates hardcoded paths and
provides a single source of truth for checkpoint file management.
"""

from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
import json


@dataclass
class CheckpointFiles:
    """Configuration for all checkpoint files used in ZeroRepo"""
    
    # Core state files
    repo_data: str = "repo_data.json"
    task_manager_state: str = "task_manager_state.json" 
    
    # Design phase files (ImplBuilder)
    skeleton: str = "skeleton.json"
    skeleton_trajectory: str = "skeleton_traj.json"
    graph: str = "graph.json"
    tasks: str = "tasks.json"
    
    # RPG files
    global_repo_rpg: str = "global_repo_rpg.json"
    current_repo_rpg: str = "cur_repo_rpg.json"

    # Dependency graph file
    repo_dep: str = "repo_dep.json"
    
    # Code generation files (IterativeCodeGenerator) 
    iteration_state: str = "iteration_state.json"
    execution_history: str = "execution_history.json"
    batch_trajectory: str = "batch_trajectory.json"
    
    # Property level files (PropBuilder)
    feature_selection: str = "feature_selection.json"
    feature_refactoring: str = "feature_refactoring.json"
    
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class CheckpointManager:
    """
    Centralized manager for all checkpoint file operations
    
    Provides a unified interface for:
    - File path resolution
    - Directory creation
    - File existence checking
    - Path standardization
    """
    
    def __init__(
        self, 
        checkpoint_dir: Union[str, Path],
        file_config: Optional[CheckpointFiles] = None
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Base directory for all checkpoint files
            file_config: Custom file configuration (uses default if None)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.files = file_config or CheckpointFiles()
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, file_key: str) -> Path:
        """
        Get full path for a checkpoint file
        
        Args:
            file_key: Key from CheckpointFiles dataclass
            
        Returns:
            Full path to the checkpoint file
        """
        if not hasattr(self.files, file_key):
            raise ValueError(f"Unknown checkpoint file key: {file_key}")
        
        filename = getattr(self.files, file_key)
        return self.checkpoint_dir / filename
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get dictionary of all checkpoint file paths"""
        return {
            key: self.get_path(key) 
            for key in self.files.to_dict().keys()
        }
    
    def exists(self, file_key: str) -> bool:
        """Check if a checkpoint file exists"""
        return self.get_path(file_key).exists()
    
    def create_subdirectory(self, subdir: str) -> Path:
        """Create and return path to subdirectory"""
        sub_path = self.checkpoint_dir / subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        return sub_path
    
    def save_config(self, config_path: Optional[Path] = None) -> Path:
        """
        Save current checkpoint configuration to file
        
        Args:
            config_path: Where to save config (defaults to checkpoint_dir/checkpoint_config.json)
        """
        if config_path is None:
            config_path = self.checkpoint_dir / "checkpoint_config.json"
        
        config_data = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "files": self.files.to_dict()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return config_path
    
    @classmethod
    def from_config_file(cls, config_path: Path) -> 'CheckpointManager':
        """Load checkpoint manager from configuration file"""
        with open(config_path) as f:
            config_data = json.load(f)
        
        files_config = CheckpointFiles(**config_data.get("files", {}))
        return cls(
            checkpoint_dir=config_data["checkpoint_dir"],
            file_config=files_config
        )
    
    # Convenience methods for commonly used paths
    @property 
    def repo_data_path(self) -> Path:
        return self.get_path("repo_data")
    
    @property
    def task_state_path(self) -> Path: 
        return self.get_path("task_manager_state")
    
    @property
    def skeleton_path(self) -> Path:
        return self.get_path("skeleton")
    
    @property 
    def graph_path(self) -> Path:
        return self.get_path("graph")
    
    @property
    def tasks_path(self) -> Path:
        return self.get_path("tasks")
    
    @property
    def global_rpg_path(self) -> Path:
        return self.get_path("global_repo_rpg")
    
    @property 
    def current_rpg_path(self) -> Path:
        return self.get_path("current_repo_rpg")
    
    @property
    def iteration_state_path(self) -> Path:
        return self.get_path("iteration_state")
    
    @property
    def execution_history_path(self) -> Path:
        return self.get_path("execution_history")
    
    @property
    def batch_trajectory_path(self) -> Path:
        return self.get_path("batch_trajectory")

    @property
    def repo_dep_path(self) -> Path:
        return self.get_path("repo_dep")


# Global default instance - can be overridden
_default_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance"""
    if _default_manager is None:
        raise RuntimeError(
            "No global checkpoint manager configured. "
            "Call set_checkpoint_manager() or create_default_manager() first."
        )
    return _default_manager


def set_checkpoint_manager(manager: CheckpointManager) -> None:
    """Set the global checkpoint manager instance"""
    global _default_manager
    _default_manager = manager


def create_default_manager(checkpoint_dir: Union[str, Path]) -> CheckpointManager:
    """Create and set a default checkpoint manager"""
    manager = CheckpointManager(checkpoint_dir)
    set_checkpoint_manager(manager)
    return manager


def reset_checkpoint_manager() -> None:
    """Reset the global checkpoint manager"""
    global _default_manager
    _default_manager = None