#!/usr/bin/env python3
"""
Refactored CodeGenerator with three distinct workflows:
1. Test Generation - generates test patches
2. Code Generation - generates code patches  
3. Environment Setup - configures execution environment

Features iterative execution loop with intelligent failure analysis.
"""
import json
import logging
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path
from zerorepo.utils.api import truncate_by_token
from zerorepo.rpg_gen.base import RPG, RepoSkeleton
from zerorepo.rpg_gen.impl_level.plan_tasks import TaskBatch, TaskManager
from zerorepo.config.checkpoint_config import CheckpointManager, get_checkpoint_manager
from .ct_builder import TraeAgentContainerBuilder, docker_exec
from .runner import ContinuousTraeAgentRunner
from .util import build_pytest_command_from_test_patch, is_test_successful
from .prompts.general_response import analyze_failure_detailed, generate_commit_message_detailed
from .prompts.gen_prompt import (
    init_test_gen_prompt,
    init_code_gen_prompt,
    test_gen_prompt,
    code_gen_prompt,
    env_gen_prompt
)
from ..utils.git_command_runner import GitCommandRunner

class WorkflowType(Enum):
    """Types of workflows in the code generation process."""
    # Test-related workflows
    TEST_DEVELOPMENT = "test_development"      # Initial test creation for new features
    TEST_FIX = "test_fix"                     # Fixing existing broken tests
    
    # Code-related workflows  
    CODE_INCREMENTAL = "code_incremental"     # Incremental feature development
    CODE_BUG_FIX = "code_bug_fix"            # Bug fixes and error corrections
    
    # Environment workflows
    ENV_SETUP = "env_setup"                   # Environment configuration and dependencies
    
    # Legacy support (for backward compatibility)
    TEST_GENERATION = "test_development"      # Alias for TEST_DEVELOPMENT
    CODE_GENERATION = "code_incremental"     # Alias for CODE_INCREMENTAL


class FailureType(Enum):
    """Types of failures that can occur during execution."""
    TEST_ERROR = "test_error"          # Test itself is wrong
    CODE_ERROR = "code_error"          # Code implementation is wrong
    ENV_ERROR = "env_error"            # Environment/setup issues
    UNKNOWN_ERROR = "unknown_error"    # Unknown/unclassified error

@dataclass
class ExecutionResult:
    """Result of executing a workflow step."""
    workflow_type: WorkflowType
    success: bool
    patch_file: Optional[str] = None
    trajectory_file: Optional[str] = None
    output: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    test_output: Optional[str] = None
    
@dataclass
class IterationState:
    """State of the current iteration."""
    iteration: int
    initial_base_commit: str  # 最开始的commit，用于test验证
    current_commit: str       # 当前commit，用于判断任务进度和生成patch
    current_workflow: WorkflowType
    test_patch: Optional[str] = None
    code_patch: Optional[str] = None
    env_patch: Optional[str] = None
    last_test_output: Optional[str] = None
    failure_history: List[FailureType] = None
    
    def __post_init__(self):
        if self.failure_history is None:
            self.failure_history = []

@dataclass
class TrajectoryRecord:
    """Single trajectory record in a batch."""
    trajectory_id: str
    workflow_type: WorkflowType
    trajectory_file: str
    execution_time: float
    success: bool
    timestamp: str
    task_description: str
    output: Optional[str] = None
    error: Optional[str] = None

@dataclass
class CommitRecord:
    """Single commit record."""
    commit_hash: str
    workflow_type: WorkflowType
    commit_message: str
    timestamp: str
    iteration: int
    subtask_id: str
    batch_id: str
    trajectory_id: Optional[str] = None
    patch_file: Optional[str] = None
    parent_commit: Optional[str] = None

@dataclass
class FailureAnalysisRecord:
    """Single failure analysis record."""
    iteration: int
    failure_type: FailureType
    test_output: str
    analysis_description: str
    timestamp: str
    subtask_id: str
    batch_id: str
    failed_workflow_type: WorkflowType
    trajectory_id: Optional[str] = None
    related_commit_hash: Optional[str] = None
    test_patch_file: Optional[str] = None
    code_patch_file: Optional[str] = None

@dataclass
class BatchTrajectoryRecord:
    """Complete trajectory record for a batch execution."""
    batch_id: str
    batch_description: str
    start_time: str
    end_time: Optional[str] = None
    total_iterations: int = 0
    success: bool = False
    final_result: Optional[str] = None
    trajectories: List[TrajectoryRecord] = None
    commits: List[CommitRecord] = None
    failure_analyses: List[FailureAnalysisRecord] = None
    llm_interactions: List[Dict[str, Any]] = None  # Store LLM interaction records
    initial_base_commit: Optional[str] = None
    final_commit: Optional[str] = None
    execution_summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.trajectories is None:
            self.trajectories = []
        if self.commits is None:
            self.commits = []
        if self.failure_analyses is None:
            self.failure_analyses = []
        if self.llm_interactions is None:
            self.llm_interactions = []
        if self.execution_summary is None:
            self.execution_summary = {}


class WorkflowContextAnalyzer:
    """Analyzes context to determine appropriate workflow type."""

    # Keywords that indicate a fix/repair task (use word boundaries to avoid false matches)
    # Avoid matching words like "NotImplementedError" or "Error handling"
    FIX_KEYWORDS = ['fix ', 'fix:', 'fixing', 'repair', 'debug', 'resolve', 'bug', 'bugfix']

    @staticmethod
    def _contains_fix_intent(text: str) -> bool:
        """Check if text indicates a fix/repair intent using careful matching."""
        text_lower = text.lower()
        # Check for explicit fix-related phrases
        fix_phrases = [
            'fix ', 'fix:', 'fixing ', 'to fix', 'need fix', 'needs fix',
            'repair ', 'repairing', 'debug ', 'debugging',
            'resolve ', 'resolving', 'bug ', 'bugfix', 'bug fix',
            'broken ', 'not working', 'fails to', 'failure in',
            'incorrect ', 'wrong ', 'issue with', 'problem with'
        ]
        return any(phrase in text_lower for phrase in fix_phrases)

    @staticmethod
    def determine_test_workflow_type(
        iteration: int,
        failure_history: List[FailureType],
        test_output: Optional[str] = None,
        batch_description: str = ""
    ) -> WorkflowType:
        """Determine if this is test development or test fix."""
        # If we have test failures in recent history, this is a test fix
        # This takes precedence over iteration number
        if failure_history and FailureType.TEST_ERROR in failure_history[-3:]:
            return WorkflowType.TEST_FIX

        # For first iteration, check if task description indicates a fix
        if iteration == 1:
            # Check batch description for fix-related keywords with careful matching
            if WorkflowContextAnalyzer._contains_fix_intent(batch_description):
                return WorkflowType.TEST_FIX
            return WorkflowType.TEST_DEVELOPMENT

        # If iteration > 1 and we have test output indicating test issues, it's a fix
        if iteration > 1 and test_output and any(keyword in test_output.lower()
            for keyword in ['test failed', 'assertion error', 'test error']):
            return WorkflowType.TEST_FIX

        # Default to test development
        return WorkflowType.TEST_DEVELOPMENT

    @staticmethod
    def determine_code_workflow_type(
        iteration: int,
        failure_history: List[FailureType],
        test_output: Optional[str] = None,
        batch_description: str = ""
    ) -> WorkflowType:
        """Determine if this is incremental development or bug fix."""
        # Check batch description for fix-related keywords first (even for iteration 1)
        # This ensures bug fixes are properly categorized from the start
        if WorkflowContextAnalyzer._contains_fix_intent(batch_description):
            return WorkflowType.CODE_BUG_FIX

        # If we have code/unknown errors in recent history, this is likely a bug fix
        recent_failures = failure_history[-3:] if failure_history else []
        if any(failure in recent_failures for failure in [FailureType.CODE_ERROR, FailureType.UNKNOWN_ERROR]):
            return WorkflowType.CODE_BUG_FIX

        # If iteration > 1 and we have test failures, likely fixing bugs
        if iteration > 1 and test_output and any(keyword in test_output.lower()
            for keyword in ['error', 'exception', 'failed', 'traceback']):
            return WorkflowType.CODE_BUG_FIX

        # Default to incremental development
        return WorkflowType.CODE_INCREMENTAL


class IterativeCodeGenerator:
    """
    Code generator with iterative test-code-env workflow.
    
    This generator implements a smart loop:
    1. Generate tests for the requirements
    2. Generate code to pass the tests  
    3. Execute tests and analyze failures
    4. Based on failure analysis, decide next action:
       - Test error: Regenerate tests
       - Code error: Regenerate code
       - Env error: Fix environment setup
    """
    
    def __init__(
        self,
        repo_path: Union[str, Path],
        ckpt_dir: Optional[Union[str, Path]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: logging.Logger = None,
        # Docker-related parameters
        use_docker: bool = False,
        trae_config_file: str = "trae_config.yaml",
        docker_image: str = "ubuntu:22.04", 
        docker_container_name: Optional[str] = None,
        docker_env_config: Optional[Dict[str, str]] = None,
        docker_volumes: Optional[Dict[str, Dict[str, str]]] = None,
        container_workspace_path: str = "/trae-workspace",
        force_rebuild: bool = False,
        # Iteration control parameters
        max_iterations: int = 5,
        max_retries_per_workflow: int = 3,
        # LLM configuration parameters
        commit_llm_config: Optional[Dict[str, Any]] = None,
        failure_llm_config: Optional[Dict[str, Any]] = None,
        # Feature control parameters
        enable_commit_changes: bool = True,
        enable_failure_analysis: bool = True
    ):
        """
        Initialize IterativeCodeGenerator with checkpoint management
        
        Args:
            repo_path: Target repository path
            ckpt_dir: (Deprecated) Checkpoint directory - use checkpoint_manager instead
            checkpoint_manager: CheckpointManager instance, if None uses global instance
            logger: Logger instance
            ... (other parameters remain the same)
        """
        self.repo_path = repo_path if isinstance(repo_path, Path) else Path(repo_path)
        self.logger = logger or logging.getLogger(__name__)
        self.git_runner = GitCommandRunner(str(self.repo_path))

        self.checkpoint_manager = checkpoint_manager        
        # Legacy support - keep ckpt_dir for backward compatibility
        self.ckpt_dir = self.checkpoint_manager.checkpoint_dir
        
        # Issue deprecation warning if ckpt_dir is used
        if ckpt_dir is not None:
            self.logger.warning(
                "ckpt_dir parameter is deprecated. Use checkpoint_manager parameter instead."
            )
        
        self.task_manager = None
        
        # Initialize additional tracking files
        self.batch_trajectory_file = self.checkpoint_manager.get_path("batch_trajectory")
        
        self.logger.info(f"IterativeCodeGenerator initialized with checkpoint dir: {self.checkpoint_manager.checkpoint_dir}")
        
        # Docker-related configuration
        self.use_docker = use_docker
        self.trae_config_file = trae_config_file
        self.docker_image = docker_image
        self.docker_container_name = docker_container_name
        self.docker_env_config = docker_env_config or {}
        self.docker_volumes = docker_volumes or {}
        self.container_workspace_path = container_workspace_path
        self.force_rebuild = force_rebuild
        
        # Iteration control
        self.max_iterations = max_iterations
        self.max_retries_per_workflow = max_retries_per_workflow
        
        # LLM configuration
        self.commit_llm_config = commit_llm_config
        self.failure_llm_config = failure_llm_config
        
        # Feature control
        self.enable_commit_changes = enable_commit_changes
        self.enable_failure_analysis = enable_failure_analysis
        
        # Docker components (initialized when needed)
        self.container_builder = None
        self.agent_runner = None
        self.container = None
        self.container_config_path = None
        
        # Current state
        self.current_state: Optional[IterationState] = None
        self.execution_history: List[ExecutionResult] = []
        
        # Batch trajectory tracking
        self.batch_trajectory_record: Optional[BatchTrajectoryRecord] = None
        self.current_trajectory_id: Optional[str] = None  # Track current trajectory for linking
        
        # Auto-repair trajectory file on initialization
        self.repair_batch_trajectory_file()
    
    # Properties for accessing checkpoint paths - eliminates hardcoded paths!
    @property
    def state_file(self) -> Path:
        """Get TaskManager state file path"""
        return self.checkpoint_manager.get_path("task_manager_state")
    
    @property
    def cur_rpg_path(self) -> Path:
        """Get current RPG file path"""
        return self.checkpoint_manager.get_path("current_repo_rpg")
    
    @property
    def global_rpg_path(self) -> Path:
        """Get global RPG file path"""
        return self.checkpoint_manager.get_path("global_repo_rpg")
    
    @property
    def tasks_result_path(self) -> Path:
        """Get tasks result file path"""
        return self.checkpoint_manager.get_path("tasks")
    
    @property
    def graph_result_path(self) -> Path:
        """Get graph result file path"""
        return self.checkpoint_manager.get_path("graph")
    
    @property
    def iteration_state_file(self) -> Path:
        """Get iteration state file path"""
        return self.checkpoint_manager.get_path("iteration_state")
    
    @property
    def execution_history_file(self) -> Path:
        """Get execution history file path"""
        return self.checkpoint_manager.get_path("execution_history")
        
    def _setup_docker_components(self):
        """Initialize Docker components if not already done."""
        if not self.use_docker:
            return
            
        if self.container_builder is None:
            self.container_builder = TraeAgentContainerBuilder(
                trae_config_file=self.trae_config_file,
                working_dir=str(self.repo_path),
                docker_image=self.docker_image,
                docker_container_name=self.docker_container_name,
                docker_env_config=self.docker_env_config,
                docker_volumes=self.docker_volumes,
                container_workspace_path=self.container_workspace_path
            )
            
        if self.container is None:
            if self.force_rebuild:
                self.container_builder.clean_artifacts()
            
            self.container, self.container_config_path = self.container_builder.build_container(
                force_rebuild=self.force_rebuild
            )
            
        if self.agent_runner is None:
            self.agent_runner = ContinuousTraeAgentRunner(
                container=self.container,
                container_config_path=self.container_config_path,
                container_workspace_path=self.container_workspace_path,
                results_dir=str(self.container_builder.results_dir),
                cleanup_interval=3,  # Cleanup zombie processes every 3 tasks
            )
            self.agent_runner.start()

    def _init_batch_trajectory_record(self, batch: TaskBatch) -> None:
        """Initialize batch trajectory record for a new batch execution."""
        start_time = datetime.now().isoformat()
        
        self.batch_trajectory_record = BatchTrajectoryRecord(
            batch_id=batch.task_id,
            batch_description=batch.task,
            start_time=start_time,
            initial_base_commit=self.current_state.initial_base_commit if self.current_state else None
        )
        
        # Log batch start
        self.logger.info(f"Started batch trajectory recording for {batch.task_id}")

    def _add_trajectory_record(
        self, 
        execution_result: ExecutionResult, 
        task_description: str
    ) -> None:
        """Add a trajectory record to the current batch."""
        if not self.batch_trajectory_record:
            return
            
        timestamp = datetime.now().isoformat()
        trajectory_id = f"{execution_result.workflow_type.value}_{len(self.batch_trajectory_record.trajectories) + 1}"
        
        # Store current trajectory_id for linking with commits and failures
        self.current_trajectory_id = trajectory_id
        
        trajectory_record = TrajectoryRecord(
            trajectory_id=trajectory_id,
            workflow_type=execution_result.workflow_type,
            trajectory_file=execution_result.trajectory_file or "",
            execution_time=execution_result.execution_time,
            success=execution_result.success,
            timestamp=timestamp,
            task_description=task_description,
            output=execution_result.output,
            error=execution_result.error
        )
        
        self.batch_trajectory_record.trajectories.append(trajectory_record)
        
        if self.logger:
            self.logger.info(f"Added trajectory record: {trajectory_id}")

    def _add_commit_record(
        self, 
        workflow_type: WorkflowType, 
        commit_hash: str,
        commit_message: str,
        iteration: int,
        subtask_id: str,
        patch_file: Optional[str] = None,
        parent_commit: Optional[str] = None
    ) -> None:
        """Add a commit record to the current batch."""
        if not self.batch_trajectory_record:
            return
            
        timestamp = datetime.now().isoformat()
        
        commit_record = CommitRecord(
            commit_hash=commit_hash,
            workflow_type=workflow_type,
            commit_message=commit_message,
            timestamp=timestamp,
            iteration=iteration,
            subtask_id=subtask_id,
            batch_id=self.batch_trajectory_record.batch_id,
            trajectory_id=self.current_trajectory_id,
            patch_file=patch_file,
            parent_commit=parent_commit
        )
        
        self.batch_trajectory_record.commits.append(commit_record)
        
        if self.logger:
            self.logger.info(f"Added commit record: {commit_hash[:8]} (iteration {iteration}, subtask: {subtask_id}, trajectory: {self.current_trajectory_id})")

    def _add_failure_analysis_record(
        self,
        iteration: int,
        failure_type: FailureType,
        test_output: str,
        analysis_description: str,
        subtask_id: str,
        failed_workflow_type: WorkflowType,
        related_commit_hash: Optional[str] = None,
        test_patch_file: Optional[str] = None,
        code_patch_file: Optional[str] = None
    ) -> None:
        """Add a failure analysis record to the current batch."""
        if not self.batch_trajectory_record:
            return
            
        timestamp = datetime.now().isoformat()
        
        failure_record = FailureAnalysisRecord(
            iteration=iteration,
            failure_type=failure_type,
            test_output=test_output,
            analysis_description=analysis_description,
            timestamp=timestamp,
            subtask_id=subtask_id,
            batch_id=self.batch_trajectory_record.batch_id,
            failed_workflow_type=failed_workflow_type,
            trajectory_id=self.current_trajectory_id,
            related_commit_hash=related_commit_hash,
            test_patch_file=test_patch_file,
            code_patch_file=code_patch_file
        )
        
        self.batch_trajectory_record.failure_analyses.append(failure_record)
        
        if self.logger:
            self.logger.info(f"Added failure analysis record: {failure_type.value} at iteration {iteration}, subtask: {subtask_id}, workflow: {failed_workflow_type.value}, trajectory: {self.current_trajectory_id}")

    def _add_llm_interaction_record(self, llm_record) -> None:
        """Add an LLM interaction record to the current batch."""
        if not self.batch_trajectory_record or not llm_record:
            return
        
        # Convert LLMInteractionRecord to dict for JSON serialization
        try:
            from dataclasses import is_dataclass
            if is_dataclass(llm_record):
                llm_record_dict = asdict(llm_record)
            elif hasattr(llm_record, '__dict__'):
                llm_record_dict = llm_record.__dict__
            else:
                llm_record_dict = dict(llm_record)
        except Exception:
            # Fallback to dict conversion
            llm_record_dict = dict(llm_record) if llm_record else {}
        
        # Add trajectory_id to the LLM interaction record
        llm_record_dict['trajectory_id'] = self.current_trajectory_id
        
        self.batch_trajectory_record.llm_interactions.append(llm_record_dict)
        
        if self.logger:
            self.logger.info(f"Added LLM interaction record: {llm_record_dict.get('function_name', 'unknown')} (trajectory: {self.current_trajectory_id})")

    def _validate_batch_data(self, batch_dict: Dict[str, Any]) -> bool:
        """Validate batch data integrity before saving."""
        required_fields = ['batch_id', 'batch_description', 'start_time']
        
        for field in required_fields:
            if field not in batch_dict or not batch_dict[field]:
                if self.logger:
                    self.logger.warning(f"Missing or empty required field: {field}")
                return False
        
        # Validate list fields exist and are lists
        list_fields = ['trajectories', 'commits', 'failure_analyses', 'llm_interactions']
        for field in list_fields:
            if field in batch_dict and not isinstance(batch_dict[field], list):
                if self.logger:
                    self.logger.warning(f"Field {field} should be a list, got {type(batch_dict[field])}")
                return False
                
        # Validate trajectory records
        if 'trajectories' in batch_dict:
            for i, traj in enumerate(batch_dict['trajectories']):
                if not isinstance(traj, dict) or 'workflow_type' not in traj:
                    if self.logger:
                        self.logger.warning(f"Invalid trajectory record at index {i}")
                    return False
        
        return True

    def _batch_to_dict(self, batch_record: BatchTrajectoryRecord) -> Dict[str, Any]:
        """Convert BatchTrajectoryRecord to dict with proper enum serialization."""
        try:
            # Use asdict first
            batch_dict = asdict(batch_record)
            
            # Convert enums to strings in trajectories
            if 'trajectories' in batch_dict:
                for trajectory in batch_dict['trajectories']:
                    if 'workflow_type' in trajectory and hasattr(trajectory['workflow_type'], 'value'):
                        trajectory['workflow_type'] = trajectory['workflow_type'].value
            
            # Convert enums to strings in commits
            if 'commits' in batch_dict:
                for commit in batch_dict['commits']:
                    if 'workflow_type' in commit and hasattr(commit['workflow_type'], 'value'):
                        commit['workflow_type'] = commit['workflow_type'].value
            
            # Convert enums to strings in failure_analyses
            if 'failure_analyses' in batch_dict:
                for failure in batch_dict['failure_analyses']:
                    if 'failure_type' in failure and hasattr(failure['failure_type'], 'value'):
                        failure['failure_type'] = failure['failure_type'].value
                    if 'failed_workflow_type' in failure and hasattr(failure['failed_workflow_type'], 'value'):
                        failure['failed_workflow_type'] = failure['failed_workflow_type'].value
            
            return batch_dict
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error converting batch to dict: {e}")
            # Fallback to basic conversion
            return {
                'batch_id': getattr(batch_record, 'batch_id', ''),
                'batch_description': getattr(batch_record, 'batch_description', ''),
                'start_time': getattr(batch_record, 'start_time', ''),
                'end_time': getattr(batch_record, 'end_time', None),
                'success': getattr(batch_record, 'success', False),
                'final_result': getattr(batch_record, 'final_result', None),
                'trajectories': [],
                'commits': [],
                'failure_analyses': [],
                'llm_interactions': getattr(batch_record, 'llm_interactions', []),
                'initial_base_commit': getattr(batch_record, 'initial_base_commit', None),
                'final_commit': getattr(batch_record, 'final_commit', None),
                'execution_summary': getattr(batch_record, 'execution_summary', {}),
                'total_iterations': getattr(batch_record, 'total_iterations', 0)
            }

    def _load_existing_batch_trajectories(self) -> List[Dict[str, Any]]:
        """Load existing batch trajectories from JSON file."""
        try:
            if self.batch_trajectory_file.exists():
                with open(self.batch_trajectory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'batches' in data:
                        return data['batches']
                    else:
                        # Handle old single batch format
                        return [data] if data else []
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load existing batch trajectories: {e}")
                
        return []

    def _save_batch_trajectory_record(self) -> None:
        """Save/append current batch trajectory record to JSON list."""
        if not self.batch_trajectory_record:
            return
            
        try:
            # Ensure directory exists
            self.batch_trajectory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing trajectories
            existing_batches = self._load_existing_batch_trajectories()
            
            # Convert current batch to dict for JSON serialization
            current_batch_dict = self._batch_to_dict(self.batch_trajectory_record)
            current_batch_dict["saved_at"] = datetime.now().isoformat()
            
            # Validate data before saving
            if not self._validate_batch_data(current_batch_dict):
                if self.logger:
                    self.logger.error("Cannot save batch trajectory: data validation failed")
                return
            
            # Check if this batch already exists (by batch_id) and replace it
            batch_found = False
            for i, batch in enumerate(existing_batches):
                if batch.get('batch_id') == current_batch_dict['batch_id']:
                    existing_batches[i] = current_batch_dict
                    batch_found = True
                    break
            
            # If not found, append new batch
            if not batch_found:
                existing_batches.append(current_batch_dict)
            
            # Create a backup before writing
            backup_file = self.batch_trajectory_file.with_suffix('.json.backup')
            if self.batch_trajectory_file.exists():
                import shutil
                shutil.copy2(self.batch_trajectory_file, backup_file)
            
            # Save the list back to file with atomic write
            temp_file = self.batch_trajectory_file.with_suffix('.json.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_batches, f, indent=2, ensure_ascii=False)
                    f.flush()  # Ensure data is written to disk
                    
                # Atomic move
                import os
                if os.name == 'nt':  # Windows
                    if self.batch_trajectory_file.exists():
                        os.replace(str(temp_file), str(self.batch_trajectory_file))
                    else:
                        temp_file.rename(self.batch_trajectory_file)
                else:  # Unix/Linux
                    temp_file.rename(self.batch_trajectory_file)
                
                # Verify file was written correctly
                if self.batch_trajectory_file.exists():
                    file_size = self.batch_trajectory_file.stat().st_size
                    if self.logger:
                        self.logger.info(f"Successfully wrote {file_size} bytes to {self.batch_trajectory_file}")
                else:
                    if self.logger:
                        self.logger.error(f"File {self.batch_trajectory_file} was not created!")
                    
            except Exception as e:
                # Clean up temp file on failure
                if temp_file.exists():
                    temp_file.unlink()
                if self.logger:
                    self.logger.error(f"Failed during atomic write: {e}")
                raise e
                
            if self.logger:
                action = "Updated" if batch_found else "Added"
                self.logger.info(f"{action} batch trajectory for {current_batch_dict['batch_id']} (total: {len(existing_batches)})")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save batch trajectory record: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

    # TODO: 加上 commit 的 link 信息
    def _finalize_batch_trajectory_record(self, success: bool, final_result: Optional[str] = None) -> None:
        """Finalize the batch trajectory record with completion information."""
        if not self.batch_trajectory_record:
            return
            
        end_time = datetime.now().isoformat()
        
        self.batch_trajectory_record.end_time = end_time
        self.batch_trajectory_record.success = success
        self.batch_trajectory_record.final_result = final_result
        self.batch_trajectory_record.total_iterations = self.current_state.iteration if self.current_state else 0
        self.batch_trajectory_record.final_commit = self.current_state.current_commit if self.current_state else None
        
        # Add execution summary
        self.batch_trajectory_record.execution_summary = {
            "total_trajectories": len(self.batch_trajectory_record.trajectories),
            "total_commits": len(self.batch_trajectory_record.commits),
            "total_failures": len(self.batch_trajectory_record.failure_analyses),
            "total_llm_interactions": len(self.batch_trajectory_record.llm_interactions),
            "workflow_counts": {
                workflow.value: len([t for t in self.batch_trajectory_record.trajectories 
                                   if (hasattr(t.workflow_type, 'value') and t.workflow_type.value == workflow.value) or
                                      (isinstance(t.workflow_type, str) and t.workflow_type == workflow.value)])
                for workflow in WorkflowType
            },
            "failure_type_counts": {
                failure.value: len([f for f in self.batch_trajectory_record.failure_analyses 
                                   if (hasattr(f.failure_type, 'value') and f.failure_type.value == failure.value) or
                                      (isinstance(f.failure_type, str) and f.failure_type == failure.value)])
                for failure in FailureType
            },
            "llm_interaction_counts": {
                func_name: len([i for i in self.batch_trajectory_record.llm_interactions 
                               if i.get('function_name') == func_name])
                for func_name in ['analyze_failure', 'generate_commit_message']
            },
            "llm_success_rate": {
                func_name: (
                    len([i for i in self.batch_trajectory_record.llm_interactions 
                         if i.get('function_name') == func_name and i.get('success', False)]) /
                    max(len([i for i in self.batch_trajectory_record.llm_interactions 
                             if i.get('function_name') == func_name]), 1)
                ) for func_name in ['analyze_failure', 'generate_commit_message']
            }
        }
        
        # Save final record
        self._save_batch_trajectory_record()
        
        if self.logger:
            self.logger.info(f"Finalized batch trajectory record for {self.batch_trajectory_record.batch_id}")

    def generate_tests(self, task: str, repo_skeleton: Optional[RepoSkeleton] = None, repo_rpg: Optional[RPG] = None) -> ExecutionResult:
        """
        Generate test patch for the given task batch.
        
        Args:
            task(str): Task Description For Tests
            
        Returns:
            ExecutionResult with test generation results
        """
        start_time = time.time()
        self.logger.info(f"Generating tests with task: {task}")
        
        try:
            if self.use_docker:
                self._setup_docker_components()
                result = self.agent_runner.run_single_task(
                    task=task,
                    task_id=f"test_gen_{int(time.time())}",
                    must_patch=True,
                    timeout=3600,  # 60 minutes
                )
                success = result.get("status") == "success"
                return ExecutionResult(
                    workflow_type=WorkflowType.TEST_GENERATION,
                    success=success,
                    patch_file=result.get("patch_file"),
                    trajectory_file=result.get("trajectory_file"),
                    output=result.get("output", ""),
                    error=result.get("error") if not success else None,
                    execution_time=time.time() - start_time
                )
            else:
                # Non-docker fallback
                self.logger.warning("Non-docker test generation not implemented")
                return ExecutionResult(
                    workflow_type=WorkflowType.TEST_GENERATION,
                    success=False,
                    error="Non-docker execution not implemented",
                    execution_time=time.time() - start_time
                )
        except Exception as e:
            self.logger.error(f"Error in test generation: {e}")
            return ExecutionResult(
                workflow_type=WorkflowType.TEST_GENERATION,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def generate_code(self, task: str, repo_skeleton: Optional[RepoSkeleton] = None, repo_rpg: Optional[RPG] = None) -> ExecutionResult:
        """
        Generate code patch for the given task batch.
        
        Args:
            task(str): Task Description For Code Generation
            
        Returns:
            ExecutionResult with code generation results
        """
        start_time = time.time()
        
        self.logger.info(f"Generating code with task: {task}")
        
        try:
            if self.use_docker:
                self._setup_docker_components()
                result = self.agent_runner.run_single_task(
                    task=task,
                    task_id=f"code_gen_{int(time.time())}",
                    must_patch=True,
                    timeout=3600,  # 60 minutes
                )
                
                success = result.get("status") == "success"
                return ExecutionResult(
                    workflow_type=WorkflowType.CODE_GENERATION,
                    success=success,
                    patch_file=result.get("patch_file"),
                    trajectory_file=result.get("trajectory_file"),
                    output=result.get("output", ""),
                    error=result.get("error") if not success else None,
                    execution_time=time.time() - start_time
                )
            else:
                # Non-docker fallback
                self.logger.warning("Non-docker code generation not implemented")
                return ExecutionResult(
                    workflow_type=WorkflowType.CODE_GENERATION,
                    success=False,
                    error="Non-docker execution not implemented",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"Error in code generation: {e}")
            return ExecutionResult(
                workflow_type=WorkflowType.CODE_GENERATION,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def setup_environment(self, task: str) -> ExecutionResult:
        """
        Setup or fix environment for the given task batch.
        
        Args:
            task(str): task description
        Returns:
            ExecutionResult with environment setup results
        """
        start_time = time.time()

        self.logger.info(f"Setting up environment with task: {task}")
        
        try:
            if self.use_docker:
                self._setup_docker_components()
                result = self.agent_runner.run_single_task(
                    task=task,
                    task_id=f"env_setup_{int(time.time())}",
                    must_patch=True,
                    timeout=3600,  # 30 minutes
                )
                
                success = result.get("status") == "success"
                return ExecutionResult(
                    workflow_type=WorkflowType.ENV_SETUP,
                    success=success,
                    patch_file=result.get("patch_file"),
                    trajectory_file=result.get("trajectory_file"),
                    output=result.get("output", ""),
                    error=result.get("error") if not success else None,
                    execution_time=time.time() - start_time
                )
            else:
                # Non-docker fallback
                self.logger.warning("Non-docker environment setup not implemented")
                return ExecutionResult(
                    workflow_type=WorkflowType.ENV_SETUP,
                    success=False,
                    error="Non-docker execution not implemented",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"Error in environment setup: {e}")
            return ExecutionResult(
                workflow_type=WorkflowType.ENV_SETUP,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    
    def commit_single_patch(self, workflow_type: WorkflowType, patch_file: str, batch: TaskBatch, iteration: int, subtask_id: str) -> bool:
        """
        Commit a single patch file to git with auto-generated commit message.
        
        Args:
            workflow_type: Type of workflow that generated the patch
            patch_file: Path to the patch file
            batch: Task batch being processed
            iteration: Current iteration number
            subtask_id: Unique identifier for this subtask
            
        Returns:
            True if commit was successful, False otherwise
        """
        # Check if commit changes is enabled
        if not self.enable_commit_changes:
            if self.logger:
                self.logger.info("Commit changes disabled, skipping commit")
            return True
            
        try:
            with open(patch_file, 'r') as f:
                patch_content = f.read()

            # Stage all changes
            self.git_runner.run(
                ["git", "add", "-A"],
                cwd=self.repo_path,
                capture_output=False,
                text=False,
                check=True,
            )

            # Check if there are changes to commit
            result = self.git_runner.run(
                ["git", "diff", "--staged", "--quiet"],
                cwd=self.repo_path,
                capture_output=False,
                text=False,
                check=False,
            )

            if result.success:
                # No changes to commit
                if self.logger:
                    self.logger.info("No changes to commit")
                return True

            # Use configured commit LLM or fallback to main function
            if self.commit_llm_config:
                # Pass LLM config to the commit message generation function
                commit_result = generate_commit_message_detailed(
                    workflow_type,
                    patch_content,
                    batch,
                    llm_config=self.commit_llm_config
                )
            else:
                commit_result = generate_commit_message_detailed(workflow_type, patch_content, batch)

            commit_msg = commit_result.commit_message

            # Record LLM interaction if available
            if commit_result.llm_record:
                self._add_llm_interaction_record(commit_result.llm_record)

            # Commit changes
            result = self.git_runner.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.repo_path,
                check=False,
            )

            if result.success:
                # Update current_commit but keep initial_base_commit
                new_head = self._get_head_commit()
                if new_head and self.current_state:
                    self.current_state.current_commit = new_head
                    if self.logger:
                        self.logger.info(f"Updated current_commit -> {new_head}")
                
                # Add commit record to batch trajectory
                if new_head:
                    parent_commit = self.current_state.current_commit if self.current_state else None
                    self._add_commit_record(
                        workflow_type=workflow_type,
                        commit_hash=new_head,
                        commit_message=commit_msg,
                        iteration=iteration,
                        subtask_id=subtask_id,
                        patch_file=patch_file,
                        parent_commit=parent_commit
                    )
    
                return True
            else:
                if self.logger:
                    self.logger.error(f"Failed to commit: {result.stderr}")
                
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during commit: {e}")
            
            return False    
    
    def execute_tests(
        self, 
        use_initial_base: bool = True
    ) -> Tuple[int, str]:
        """
        Execute tests and capture output for analysis with comprehensive test detection.
        
        Args:
            use_initial_base: If True, use initial_base_commit to get full diff for testing
                             If False, use current test patches
            
        Returns:
            Tuple of (return_code, test_output)
        """
        from .util import build_comprehensive_pytest_command_from_patch
        
        if use_initial_base and self.current_state and self.current_state.initial_base_commit:
            # 使用 initial_base_commit 和当前HEAD的完整diff来运行测试
            try:
                # 获取当前 HEAD commit
                current_head = self.git_runner.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.repo_path,
                    check=True,
                ).stdout.strip()

                self.logger.info(f"============== Generating Diff for Testing ==============")
                self.logger.info(f"Initial base commit: {self.current_state.initial_base_commit[:12]}")
                self.logger.info(f"Current HEAD commit: {current_head[:12]}")
                self.logger.info(f"Git diff command: git diff {self.current_state.initial_base_commit} HEAD")

                # 生成从 initial_base_commit 到 HEAD 的完整 diff
                full_diff = self.git_runner.run(
                    ["git", "diff", self.current_state.initial_base_commit, "HEAD"],
                    cwd=self.repo_path,
                    check=True,
                ).stdout
                
                # Log diff statistics
                diff_lines = full_diff.splitlines()
                files_changed = [line for line in diff_lines if line.startswith('diff --git')]
                self.logger.info(f"Diff contains {len(diff_lines)} lines, {len(files_changed)} files changed")
                
                if files_changed:
                    self.logger.info(f"Files in diff:")
                    for file_line in files_changed:  # Show first 10 files
                        # Extract file paths from "diff --git a/path b/path"
                        parts = file_line.split()
                        if len(parts) >= 4:
                            file_path = parts[3].replace('b/', '')
                            self.logger.info(f"  - {file_path}")
                    if len(files_changed) > 10:
                        self.logger.info(f"  ... and {len(files_changed) - 10} more files")
                else:
                    self.logger.info("No files changed in diff")
                
                # 使用新的综合测试检测功能
                py_cmd, analysis_info = build_comprehensive_pytest_command_from_patch(
                    patch_content=full_diff,
                    repo_root=str(self.repo_path)
                )
                
                # 记录详细分析信息
                self.logger.info(f"============== Test Analysis Results ==============")
                self.logger.info(f"Test Analysis - Patch test files: {len(analysis_info['patch_test_files'])}")
                if analysis_info['patch_test_files']:
                    self.logger.info(f"Patch test files: {analysis_info['patch_test_files']}")
                    
                self.logger.info(f"Test Analysis - Modified source files: {len(analysis_info['patch_source_files'])}")
                if analysis_info['patch_source_files']:
                    self.logger.info(f"Modified source files: {analysis_info['patch_source_files']}")
                    
                self.logger.info(f"Test Analysis - Related test files: {len(analysis_info['related_test_files'])}")
                if analysis_info['related_test_files']:
                    self.logger.info(f"Related test files: {analysis_info['related_test_files']}")
                    
                self.logger.info(f"Test Analysis - Total test files to run: {len(analysis_info['all_test_files'])}")
                if analysis_info['all_test_files']:
                    self.logger.info(f"All test files to run: {analysis_info['all_test_files']}")
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to get full diff, using fallback: {e}")
                # fallback to basic approach
                if self.current_state.test_patch:
                    with open(self.current_state.test_patch, 'r') as f:
                        test_patch = f.read()
                    py_cmd = build_pytest_command_from_test_patch(
                        test_patch_content=test_patch
                    )
                else:
                    py_cmd = ["python", "-m", "pytest"]
        else:
            # 使用当前的test patch文件
            if self.current_state and self.current_state.test_patch:
                self.logger.info(f"============== Using Test Patch File ==============")
                self.logger.info(f"Test patch file: {self.current_state.test_patch}")
                
                with open(self.current_state.test_patch, 'r') as f:
                    test_patch = f.read()
                
                # Log patch statistics
                patch_lines = test_patch.splitlines()
                files_changed = [line for line in patch_lines if line.startswith('diff --git')]
                self.logger.info(f"Test patch contains {len(patch_lines)} lines, {len(files_changed)} files changed")
                
                if files_changed:
                    self.logger.info(f"Files in test patch:")
                    for file_line in files_changed:  # Show first 10 files
                        # Extract file paths from "diff --git a/path b/path"
                        parts = file_line.split()
                        if len(parts) >= 4:
                            file_path = parts[3].replace('b/', '')
                            self.logger.info(f"  - {file_path}")
                    if len(files_changed) > 10:
                        self.logger.info(f"  ... and {len(files_changed) - 10} more files")
                else:
                    self.logger.info("No files changed in test patch")
                
                try:
                    # 使用新的综合测试检测功能
                    py_cmd, analysis_info = build_comprehensive_pytest_command_from_patch(
                        patch_content=test_patch,
                        repo_root=str(self.repo_path)
                    )
                    
                    # 记录详细分析信息
                    self.logger.info(f"============== Test Analysis Results ==============")
                    self.logger.info(f"Test Analysis - Patch test files: {len(analysis_info['patch_test_files'])}")
                    if analysis_info['patch_test_files']:
                        self.logger.info(f"Patch test files: {analysis_info['patch_test_files']}")
                        
                    self.logger.info(f"Test Analysis - Modified source files: {len(analysis_info['patch_source_files'])}")
                    if analysis_info['patch_source_files']:
                        self.logger.info(f"Modified source files: {analysis_info['patch_source_files']}")
                        
                    self.logger.info(f"Test Analysis - Related test files: {len(analysis_info['related_test_files'])}")
                    if analysis_info['related_test_files']:
                        self.logger.info(f"Related test files: {analysis_info['related_test_files']}")
                        
                    self.logger.info(f"Test Analysis - Total test files to run: {len(analysis_info['all_test_files'])}")
                    if analysis_info['all_test_files']:
                        self.logger.info(f"All test files to run: {analysis_info['all_test_files']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed comprehensive test analysis, using basic approach: {e}")
                    py_cmd = build_pytest_command_from_test_patch(
                        test_patch_content=test_patch
                    )
            else:
                py_cmd = ["python", "-m", "pytest"]
        
        # 转换为字符串命令
        if isinstance(py_cmd, list):
            py_cmd_str = " ".join(py_cmd)
        else:
            py_cmd_str = py_cmd
        
        py_cmd_str = f"PYTHONPATH={self.container_workspace_path} {py_cmd_str}"
        self.logger.info(f"============== Executing Test ==============")
        self.logger.info(f"Executing Test with command: {py_cmd_str}")
        return_code, test_result = docker_exec(
            container=self.container,
            command=py_cmd_str
        )
        self.logger.info(f"Executed Result: {test_result}")
        
        return return_code, test_result
    
    def _get_head_commit(self) -> Optional[str]:
        try:
            r = self.git_runner.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                check=True,
            )
            return r.stdout.strip()
        except Exception as e:
            return None

    def _get_main_branch_commit(self) -> Optional[str]:
        """Get commit hash of the main branch"""
        try:
            # Try common main branch names
            for branch_name in ["main", "master"]:
                try:
                    r = self.git_runner.run(
                        ["git", "rev-parse", branch_name],
                        cwd=self.repo_path,
                        check=True,
                    )
                    return r.stdout.strip()
                except Exception:
                    continue
            return None
        except Exception:
            return None
        
    def task_executor(
        self,
        batch: TaskBatch,
        repo_skeleton: RepoSkeleton,
        repo_rpg: RPG,
        **kwargs  # Additional arguments (currently unused)
    ) -> Tuple[bool, str]:
        """
        Returns:
            (success, error_msg)
            - success=True  -> error_msg="" (empty)
            - success=False -> error_msg contains the reason (prefer last test output / exception)
        """
        
        self.logger.info(f"Starting iterative workflow for batch {batch.task_id}")

        # Initialize batch trajectory recording
        self._init_batch_trajectory_record(batch)

        # Initialize state - we need to get the main branch commit as initial_base_commit
        # because TaskManager switches to a task branch, but we want to test against main
        head_commit = self._get_head_commit()
        if not head_commit:
            self.logger.error("Failed to get current HEAD commit")
            return False, "Failed to get current HEAD commit"
            
        # Use initial_commit from TaskManager if available (task start commit)
        # This provides a more precise baseline than main branch
        task_initial_commit = getattr(batch, 'initial_commit', None)
        
        if task_initial_commit:
            initial_base_commit = task_initial_commit
            self.logger.info(f"Using task start commit {task_initial_commit[:8]} as test baseline")
            self.logger.info(f"Working on task branch commit {head_commit[:8]} for development")
        else:
            # Fallback: use main branch commit for test validation
            main_branch_commit = self._get_main_branch_commit()
            if main_branch_commit:
                initial_base_commit = main_branch_commit
                self.logger.info(f"Using main branch commit {main_branch_commit[:8]} as test baseline")
                self.logger.info(f"Working on task branch commit {head_commit[:8]} for development")
            else:
                # Final fallback: use HEAD for both
                initial_base_commit = head_commit
                self.logger.warning(f"Could not find main branch, using HEAD {head_commit[:8]} for both baseline and development")
        
        self.current_state = IterationState(
            iteration=0,
            initial_base_commit=initial_base_commit,  # 任务开始时的commit，用于test验证
            current_commit=head_commit,               # 任务分支当前commit，用于开发和patch生成
            current_workflow=WorkflowType.TEST_DEVELOPMENT  # Start with test development
        )


        test_task = init_test_gen_prompt(batch=batch)
        code_task = init_code_gen_prompt(batch=batch)
        env_task = ""
        
        last_error_msg = ""
        try:
            while self.current_state.iteration < self.max_iterations:
                self.current_state.iteration += 1
                self.logger.info(f"=== Iteration {self.current_state.iteration} ===")
                
                # 1) tests
                if not self.current_state.test_patch:
                    # Always use the analyzer to determine workflow type
                    # It will consider iteration, failure history, and task description
                    test_workflow_type = WorkflowContextAnalyzer.determine_test_workflow_type(
                        iteration=self.current_state.iteration,
                        failure_history=self.current_state.failure_history,
                        test_output=self.current_state.last_test_output,
                        batch_description=batch.task
                    )
                    
                    test_result = self.generate_tests(test_task, repo_skeleton, repo_rpg)
                    # Update the workflow type in the result
                    test_result.workflow_type = test_workflow_type
                    self.execution_history.append(test_result)
                    
                    # Add trajectory record for test generation
                    self._add_trajectory_record(test_result, test_task)

                    if test_result.success and test_result.patch_file:
                        # Store file path (not content), verify file is readable first
                        try:
                            with open(test_result.patch_file, 'r') as f:
                                f.read()  # Verify file is readable
                            self.current_state.test_patch = test_result.patch_file
                        except:
                            self.current_state.test_patch = None
                        self.logger.info(f"Test patch generated: {test_result.patch_file} (type: {test_workflow_type.value})")
                        if self.current_state.test_patch:
                            # 立即commit test patch with correct workflow type
                            subtask_type = "test_dev" if test_workflow_type == WorkflowType.TEST_DEVELOPMENT else "test_fix"
                            subtask_id = f"{subtask_type}_iter{self.current_state.iteration}_{batch.task_id}"
                            commit_success = self.commit_single_patch(
                                test_workflow_type, 
                                test_result.patch_file, 
                                batch,
                                self.current_state.iteration,
                                subtask_id
                            )
                            if commit_success:
                                self.logger.info("Test patch committed successfully")
                            else:
                                self.logger.warning("Failed to commit test patch")
                        
                    else:
                        last_error_msg = test_result.error or "Test generation failed"
                        continue
                # 2) code
                if not self.current_state.test_patch or not self.current_state.code_patch:
                    # Always use the analyzer to determine workflow type
                    # It will consider iteration, failure history, and task description
                    code_workflow_type = WorkflowContextAnalyzer.determine_code_workflow_type(
                        iteration=self.current_state.iteration,
                        failure_history=self.current_state.failure_history,
                        test_output=self.current_state.last_test_output,
                        batch_description=batch.task
                    )
                    
                    code_result = self.generate_code(code_task, repo_skeleton, repo_rpg)
                    # Update the workflow type in the result
                    code_result.workflow_type = code_workflow_type
                    self.execution_history.append(code_result)
                    
                    # Add trajectory record for code generation
                    self._add_trajectory_record(code_result, code_task)

                    if code_result.success and code_result.patch_file:
                        # Store file path (not content), verify file is readable first
                        try:
                            with open(code_result.patch_file, 'r') as f:
                                f.read()  # Verify file is readable
                            self.current_state.code_patch = code_result.patch_file
                        except:
                            self.current_state.code_patch = None
                        self.logger.info(f"Code patch generated: {code_result.patch_file} (type: {code_workflow_type.value})")
    
                        if self.current_state.code_patch:
                            # 立即commit code patch with correct workflow type
                            subtask_type = "code_dev" if code_workflow_type == WorkflowType.CODE_INCREMENTAL else "code_fix"
                            subtask_id = f"{subtask_type}_iter{self.current_state.iteration}_{batch.task_id}"
                            commit_success = self.commit_single_patch(
                                code_workflow_type,
                                code_result.patch_file,
                                batch,
                                self.current_state.iteration,
                                subtask_id
                            )
                            if commit_success:
                                self.logger.info("Code patch committed successfully")
                            else:
                                self.logger.warning("Failed to commit code patch")
                    else:
                        last_error_msg = code_result.error or "Code generation failed"
                        continue
                # 3) execute tests
                if self.current_state.test_patch and self.current_state.code_patch:
                    return_code, test_output = self.execute_tests(
                        use_initial_base=True  # 使用initial_base_commit和HEAD的完整diff进行测试
                    )
                
                    # 关键：把最后的测试输出保留下来，作为失败时的 error_msg
                    self.current_state.last_test_output = test_output.strip() if test_output else ""

                    test_success = is_test_successful(
                        return_code=return_code,
                        test_output=test_output
                    )
                    self.logger.info(f"Test execution success: {test_success}")
                    # 测试通过
                    if test_success:
                        # 测试成功，不需要再次commit（已经在每个步骤commit过了）
                        self.logger.info("🎉 All tests passed! Workflow completed successfully.")
                        
                        # Finalize batch trajectory record on success
                        self._finalize_batch_trajectory_record(
                            success=True, 
                            final_result="All tests passed successfully"
                        )
                        
                        return True, ""

                    # Use detailed failure analysis with LLM recording (if enabled)
                    if self.enable_failure_analysis:
                        test_output = truncate_by_token(
                            text=self.current_state.last_test_output,
                            max_tokens=50000,  # Truncate to 3000 tokens for analysis
                            model=self.failure_llm_config.model if self.failure_llm_config else "gpt-4o"
                        )
                        # Read patch content from file paths for LLM analysis
                        test_patch_content = ""
                        code_patch_content = ""
                        if self.current_state.test_patch:
                            try:
                                with open(self.current_state.test_patch, 'r') as f:
                                    test_patch_content = f.read()
                            except Exception:
                                test_patch_content = ""
                        if self.current_state.code_patch:
                            try:
                                with open(self.current_state.code_patch, 'r') as f:
                                    code_patch_content = f.read()
                            except Exception:
                                code_patch_content = ""

                        # Use configured failure LLM or fallback to main function
                        if self.failure_llm_config:
                            failure_result = analyze_failure_detailed(
                                test_patch=test_patch_content,
                                code_patch=code_patch_content,
                                test_output=test_output,  # 使用字符串不是ExecutionResult
                                batch=batch,
                                llm_config=self.failure_llm_config
                            )
                        else:
                            failure_result = analyze_failure_detailed(
                                test_patch=test_patch_content,
                                code_patch=code_patch_content,
                                test_output=test_output,  # 使用字符串不是ExecutionResult
                                batch=batch
                            )
                        
                        fail_task = failure_result.task_description
                        failure_type = failure_result.failure_type
                        
                        # Record LLM interaction if available
                        if failure_result.llm_record:
                            self._add_llm_interaction_record(failure_result.llm_record)
                    else:
                        # Simple failure analysis without LLM
                        self.logger.info("Failure analysis disabled, using simple heuristics")
                        if "test" in test_output.lower() and ("failed" in test_output.lower() or "error" in test_output.lower()):
                            if any(keyword in test_output.lower() for keyword in ["assertion", "expect"]):
                                failure_type = FailureType.TEST_ERROR
                                fail_task = f"Fix failing tests based on output: {test_output[:200]}..."
                            else:
                                failure_type = FailureType.CODE_ERROR
                                fail_task = f"Fix code issues based on test failures: {test_output[:200]}..."
                        elif "import" in test_output.lower() or "module" in test_output.lower():
                            failure_type = FailureType.ENV_ERROR
                            fail_task = f"Fix environment/dependency issues: {test_output[:200]}..."
                        else:
                            failure_type = FailureType.UNKNOWN_ERROR
                            fail_task = f"Investigate and fix unknown error: {test_output[:200]}..."
                    
                    self.current_state.failure_history.append(failure_type)
                    self.logger.warning(f"Tests failed with failure type: {failure_type.value}")
                    
                    # Add failure analysis record to batch trajectory
                    # Determine which workflow failed based on the failure type with granular types
                    if failure_type == FailureType.TEST_ERROR:
                        # Use the actual test workflow type that was used
                        test_workflow_type = WorkflowContextAnalyzer.determine_test_workflow_type(
                            iteration=self.current_state.iteration,
                            failure_history=self.current_state.failure_history,
                            test_output=test_output,
                            batch_description=batch.task
                        )
                        failed_workflow = test_workflow_type
                        workflow_name = "test_dev" if test_workflow_type == WorkflowType.TEST_DEVELOPMENT else "test_fix"
                        subtask_id = f"{workflow_name}_analysis_iter{self.current_state.iteration}_{batch.task_id}"
                    elif failure_type == FailureType.CODE_ERROR or failure_type == FailureType.UNKNOWN_ERROR:
                        # Use the actual code workflow type that was used
                        code_workflow_type = WorkflowContextAnalyzer.determine_code_workflow_type(
                            iteration=self.current_state.iteration,
                            failure_history=self.current_state.failure_history,
                            test_output=test_output,
                            batch_description=batch.task
                        )
                        failed_workflow = code_workflow_type
                        workflow_name = "code_dev" if code_workflow_type == WorkflowType.CODE_INCREMENTAL else "code_fix"
                        subtask_id = f"{workflow_name}_analysis_iter{self.current_state.iteration}_{batch.task_id}"
                    else:  # ENV_ERROR
                        failed_workflow = WorkflowType.ENV_SETUP
                        subtask_id = f"env_analysis_iter{self.current_state.iteration}_{batch.task_id}"
                        
                    self._add_failure_analysis_record(
                        iteration=self.current_state.iteration,
                        failure_type=failure_type,
                        test_output=test_output,
                        analysis_description=fail_task,
                        subtask_id=subtask_id,
                        failed_workflow_type=failed_workflow,
                        related_commit_hash=self.current_state.current_commit,
                        test_patch_file=self.current_state.test_patch if hasattr(self.current_state, 'test_patch') else None,
                        code_patch_file=self.current_state.code_patch if hasattr(self.current_state, 'code_patch') else None
                    )
                    
                    # 4) decide next action
                    if failure_type == FailureType.TEST_ERROR:
                        # 清除test patch状态，下次循环会重新生成
                        self.current_state.test_patch = None
                        test_task = test_gen_prompt(
                            test_result=test_output,
                            task=fail_task
                        )
                    elif failure_type == FailureType.CODE_ERROR or failure_type == FailureType.UNKNOWN_ERROR:
                        self.current_state.code_patch = None
                        code_task = code_gen_prompt(
                            test_result=test_output,
                            task=fail_task
                        )
                    elif failure_type == FailureType.ENV_ERROR:
                        env_task = env_gen_prompt(
                            test_result=test_output,
                            task=fail_task
                        )
                        env_result = self.setup_environment(env_task)
                        self.execution_history.append(env_result)
                        
                        # Add trajectory record for environment setup
                        self._add_trajectory_record(env_result, env_task)
                        if env_result.success and env_result.patch_file:
                            # Store file path (not content), verify file is readable first
                            try:
                                with open(env_result.patch_file, 'r') as f:
                                    f.read()  # Verify file is readable
                                self.current_state.env_patch = env_result.patch_file
                            except:
                                self.current_state.env_patch = None

                            if self.current_state.env_patch:
                                # 立即commit env patch
                                subtask_id = f"env_setup_iter{self.current_state.iteration}_{batch.task_id}"
                                commit_success = self.commit_single_patch(
                                    WorkflowType.ENV_SETUP,
                                    env_result.patch_file,
                                    batch,
                                    self.current_state.iteration,
                                    subtask_id
                                )
                                if commit_success:
                                    self.logger.info("Environment patch committed successfully")
                                else:
                                    self.logger.warning("Failed to commit environment patch")
            # max iterations reached
            self.logger.error(f"Max iterations ({self.max_iterations}) reached without success")
            
            # Finalize batch trajectory record on max iterations failure
            final_error_msg = last_error_msg or "max_iterations_reached"
            self._finalize_batch_trajectory_record(
                success=False, 
                final_result=f"Failed after {self.max_iterations} iterations: {final_error_msg}"
            )
            
            # 返回最后一次我们记录到的错误信息（尽量有内容）
            return False, final_error_msg

        except Exception as e:
            self.logger.error(f"Error in task executor: {e}")
            
            # Finalize batch trajectory record on exception
            self._finalize_batch_trajectory_record(
                success=False, 
                final_result=f"Exception during execution: {str(e)}"
            )
        
            return False, str(e)

        finally:
            self._save_execution_history()
    
    def cleanup_docker(self):
        """Clean up Docker resources."""
        if self.agent_runner:
            self.agent_runner.stop()
            
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except Exception as e:
                self.logger.warning(f"Error cleaning up container: {e}")
                    
        # Reset Docker components
        self.container_builder = None
        self.agent_runner = None
        self.container = None
        self.container_config_path = None
        
    def _save_execution_history(self):
        """Save execution history to file."""
        history_data = []
        for result in self.execution_history:
            history_data.append({
                "workflow_type": result.workflow_type.value,
                "success": result.success,
                "patch_file": result.patch_file,
                "trajectory_file": result.trajectory_file,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time,
                "test_output": result.test_output
            })
        
        with open(self.execution_history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def get_batch_trajectory_record(self) -> Optional[Dict[str, Any]]:
        """Get the current batch trajectory record as a dictionary."""
        if not self.batch_trajectory_record:
            return None
        return self._batch_to_dict(self.batch_trajectory_record)
    
    def load_all_batch_trajectories(self) -> List[Dict[str, Any]]:
        """Load all batch trajectory records from JSONL file."""
        return self._load_existing_batch_trajectories()
    
    def get_batch_trajectory_by_id(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific batch trajectory by batch_id."""
        trajectories = self.load_all_batch_trajectories()
        for batch in trajectories:
            if batch.get('batch_id') == batch_id:
                return batch
        return None
    
    def get_all_llm_interactions(self) -> List[Dict[str, Any]]:
        """Get all LLM interactions from all batches."""
        trajectories = self.load_all_batch_trajectories()
        all_interactions = []
        
        for batch in trajectories:
            batch_id = batch.get('batch_id', 'unknown')
            interactions = batch.get('llm_interactions', [])
            
            # Add batch_id to each interaction for context
            for interaction in interactions:
                interaction_with_batch = interaction.copy()
                interaction_with_batch['batch_id'] = batch_id
                all_interactions.append(interaction_with_batch)
                
        return all_interactions
    
    def get_llm_interactions_by_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get LLM interactions for a specific batch."""
        batch_data = self.get_batch_trajectory_by_id(batch_id)
        if batch_data:
            return batch_data.get('llm_interactions', [])
        return []
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all batches."""
        trajectories = self.load_all_batch_trajectories()
        
        total_batches = len(trajectories)
        total_llm_interactions = 0
        total_commits = 0
        total_failures = 0
        successful_batches = 0
        
        workflow_counts = {}
        failure_type_counts = {}
        
        for batch in trajectories:
            if batch.get('success', False):
                successful_batches += 1
                
            total_llm_interactions += len(batch.get('llm_interactions', []))
            total_commits += len(batch.get('commits', []))
            total_failures += len(batch.get('failure_analyses', []))
            
            # Count workflows
            for trajectory in batch.get('trajectories', []):
                workflow_type = trajectory.get('workflow_type', 'unknown')
                workflow_counts[workflow_type] = workflow_counts.get(workflow_type, 0) + 1
                
            # Count failure types
            for failure in batch.get('failure_analyses', []):
                failure_type = failure.get('failure_type', 'unknown')
                failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        
        return {
            "total_batches": total_batches,
            "successful_batches": successful_batches,
            "success_rate": successful_batches / max(total_batches, 1),
            "total_llm_interactions": total_llm_interactions,
            "total_commits": total_commits,
            "total_failures": total_failures,
            "workflow_counts": workflow_counts,
            "failure_type_counts": failure_type_counts,
            "last_updated": datetime.now().isoformat()
        }
    
    def repair_batch_trajectory_file(self) -> bool:
        """Repair corrupted batch trajectory file by removing incomplete records."""
        try:
            if not self.batch_trajectory_file.exists():
                if self.logger:
                    self.logger.info("No batch trajectory file to repair")
                return True
            
            # Try to read the file
            with open(self.batch_trajectory_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                # Empty file, create new empty list
                with open(self.batch_trajectory_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                if self.logger:
                    self.logger.info("Repaired empty batch trajectory file")
                return True
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # Validate each batch record
                    valid_batches = []
                    for i, batch in enumerate(data):
                        if isinstance(batch, dict) and self._validate_batch_data(batch):
                            valid_batches.append(batch)
                        else:
                            if self.logger:
                                self.logger.warning(f"Removed invalid batch record at index {i}")
                    
                    # Write back valid batches
                    with open(self.batch_trajectory_file, 'w', encoding='utf-8') as f:
                        json.dump(valid_batches, f, indent=2, ensure_ascii=False)
                    
                    if self.logger:
                        self.logger.info(f"Repaired batch trajectory file: kept {len(valid_batches)} valid records")
                    return True
                    
            except json.JSONDecodeError:
                # File is corrupted, try to salvage what we can
                if self.logger:
                    self.logger.warning("Batch trajectory file is corrupted, attempting to salvage data")
                
                # Create backup of corrupted file
                backup_file = self.batch_trajectory_file.with_suffix('.json.corrupted')
                import shutil
                shutil.copy2(self.batch_trajectory_file, backup_file)
                
                # Start fresh with empty list
                with open(self.batch_trajectory_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                
                if self.logger:
                    self.logger.info(f"Created fresh batch trajectory file. Corrupted backup saved to {backup_file}")
                
                return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to repair batch trajectory file: {e}")
            return False
        
        return True

    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Unused parameters are required by context manager protocol
        _ = exc_type, exc_val, exc_tb
        self.cleanup_docker()
        return False  # Don't suppress exceptions
