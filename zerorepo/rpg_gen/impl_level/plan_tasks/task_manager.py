"""
Task Manager with Git-based version control and PR workflow
Each task is managed as a separate branch/PR with rollback capability
"""

import json
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from zerorepo.utils.logs import setup_logger
from zerorepo.utils.git_command_runner import GitCommandRunner
from zerorepo.config.checkpoint_config import CheckpointManager, get_checkpoint_manager
from .task_batch import TaskBatch
from zerorepo.rpg_gen.base import (
    RepoSkeleton, RPG, Node
)

@dataclass
class TaskState:
    """Tracks the state of a task/PR"""
    task_id: str
    batch: TaskBatch
    branch_name: str
    status: str  # 'pending', 'in_progress', 'success', 'failed'
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    commit_hash: Optional[str] = None
    initial_commit: Optional[str] = None  # Commit hash when task branch was created
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskState':
        batch_data = data.pop('batch')
        batch = TaskBatch.from_dict(batch_data)
        return cls(batch=batch, **data)


class TaskManager:
    """
    Manages tasks with Git version control
    - Each task is a branch/PR
    - Only successful tasks are merged
    - Failed tasks are rolled back
    - Supports resuming from checkpoint
    - Now uses CheckpointManager for file path management
    """
    
    def __init__(
        self,
        repo_path: str,
        state_file: Optional[Union[str, Path]] = None,
        rpg_path: Optional[Union[str, Path]] = None,
        global_rpg_path: Optional[Union[str, Path]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        graph_data: Optional[Dict] = None,
        main_branch: str = "master",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TaskManager with checkpoint management
        
        Args:
            repo_path: Git repository path
            state_file: (Deprecated) State file path - now managed by CheckpointManager
            rpg_path: (Deprecated) RPG file path - now managed by CheckpointManager  
            global_rpg_path: (Deprecated) Global RPG file path - now managed by CheckpointManager
            checkpoint_manager: CheckpointManager instance, if None uses global instance
            graph_data: Optional graph data
            main_branch: Main branch name
            logger: Logger instance
        """
        self.repo_path = Path(repo_path)
        self.main_branch = main_branch
        self.logger = logger or setup_logger()
        self.graph_data = graph_data
        
        # Get checkpoint manager
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            try:
                self.checkpoint_manager = get_checkpoint_manager()
            except RuntimeError:
                # If no global manager, create a default one
                from config.checkpoint_config import create_default_manager
                self.checkpoint_manager = create_default_manager(
                    self.repo_path / "checkpoints"
                )
        
        # Issue deprecation warnings for old parameters
        if any([state_file, rpg_path, global_rpg_path]):
            self.logger.warning(
                "state_file, rpg_path, global_rpg_path parameters are deprecated. "
                "Paths are now managed by CheckpointManager."
            )
        
        # Initialize task manager components
        self._initialize_task_manager()
    
    # Properties for accessing checkpoint paths
    @property
    def state_file(self) -> Path:
        """Get TaskManager state file path"""
        return self.checkpoint_manager.get_path("task_manager_state")
    
    @property
    def rpg_path(self) -> Path:
        """Get current RPG file path"""
        return self.checkpoint_manager.get_path("current_repo_rpg")
    
    @property
    def global_rpg_path(self) -> Path:
        """Get global RPG file path"""
        return self.checkpoint_manager.get_path("global_repo_rpg")
    
    def _initialize_task_manager(self):
        """Initialize TaskManager after checkpoint manager is set up"""
        # Task execution history
        self.completed_tasks: List[TaskState] = []
        self.failed_tasks: List[TaskState] = []
        self.current_task: Optional[TaskState] = None
        self.git_runner: GitCommandRunner = GitCommandRunner(str(self.repo_path))

        # stash 记录（避免 stash 后不恢复）
        self._active_stash_ref: Optional[str] = None

        # Initialize git repository if needed
        self._ensure_git_repository()

        # Clean up stale task branches before loading state
        self._cleanup_stale_task_branches()

        # Initialize skeleton and RPG if not provided
        self._initialize_repo_state()
        
        # Load previous state if exists
        self._load_state()
        
        # Initialize base classes if this is a fresh start
        self._initialize_base_classes_if_needed()
        
        self.logger.info(f"TaskManager initialized with checkpoint manager: {self.checkpoint_manager.checkpoint_dir}")
    
    def _cleanup_stale_task_branches(self):
        """Clean up stale task/* branches before starting new tasks.

        This ensures a clean state by removing any leftover task branches
        that may have been created in previous runs but not properly cleaned up
        (e.g., due to rollback failures or interrupted executions).
        """
        # First ensure we're on main branch
        self._force_switch_to_main()

        # List all branches
        branch_result = self.git_runner.run(['git', 'branch'], check=False)
        if not branch_result.success:
            self.logger.warning("Failed to list git branches for cleanup")
            return

        # Find task/* branches
        task_branches = []
        for line in branch_result.stdout.strip().split('\n'):
            branch = line.strip().lstrip('* ')
            if branch.startswith('task/'):
                task_branches.append(branch)

        if not task_branches:
            self.logger.debug("No stale task branches found")
            return

        self.logger.info(f"Found {len(task_branches)} stale task branch(es), cleaning up...")

        # Delete each task branch
        for branch in task_branches:
            self.logger.info(f"Deleting stale branch: {branch}")
            delete_result = self.git_runner.run(['git', 'branch', '-D', branch], check=False)
            if not delete_result.success:
                self.logger.warning(f"Failed to delete branch {branch}: {delete_result.stderr}")
            else:
                self.logger.info(f"Deleted branch: {branch}")

        self.logger.info("Stale task branches cleanup completed")

    def _force_switch_to_main(self):
        """Helper: Force switch to main branch safely"""
        # 1. 获取当前分支
        branch_result = self.git_runner.run(['git', 'branch', '--show-current'], check=False)
        current = branch_result.stdout.strip()

        if current == self.main_branch:
            return

        self.logger.info(f"Switching from '{current}' back to '{self.main_branch}'...")

        # 2. 尝试直接切换
        checkout_result = self.git_runner.run(['git', 'checkout', self.main_branch], check=False)

        if not checkout_result.success:
            self.logger.warning(f"Direct checkout to {self.main_branch} failed ({checkout_result.stderr}), attempting to cleanup...")
            # 如果因为有未提交的更改导致无法切换，先强制提交
            self._ensure_clean_workspace()
            # 再次尝试切换
            checkout_result = self.git_runner.run(['git', 'checkout', self.main_branch], check=False)
            if not checkout_result.success:
                self.logger.error(f"CRITICAL: Failed to return to {self.main_branch}. Stuck on {current}.")

    def _ensure_git_repository(self):
        """Initialize git repository properly ensuring main branch exists with at least one commit"""
        git_dir = self.repo_path / ".git"

        # 1. 确保 Git 初始化
        if not git_dir.exists():
            self.logger.info("Initializing git repository...")
            self.repo_path.mkdir(parents=True, exist_ok=True)
            init_success = self.git_runner.run(['git', 'init', '-b', self.main_branch], check=False).success
            if not init_success:
                self.git_runner.run(['git', 'init'], check=False)
        else:
            self.logger.debug("Git repository already exists")

        # Fix ownership and permissions for git operations
        self.git_runner._fix_git_ownership_and_permissions()

        # Remove stale git lock files if they exist
        lock_file = self.repo_path / ".git" / "index.lock"
        if lock_file.exists():
            try:
                lock_file.unlink()
                self.logger.info(f"Removed stale git lock file: {lock_file}")
            except Exception as e:
                self.logger.warning(f"Could not remove git lock file: {e}")

        # Configure git to trust this directory
        try:
            self.git_runner.run([
                'git', 'config', '--global', '--add',
                'safe.directory', str(self.repo_path.resolve())
            ], check=False)
            self.logger.info(f"Added {self.repo_path} to git safe directories")
        except Exception as e:
            self.logger.warning(f"Could not add directory to git safe.directory: {e}")

        # Also try local git config
        try:
            self.git_runner.run([
                'git', 'config', '--add',
                'safe.directory', str(self.repo_path.resolve())
            ], check=False)
        except Exception as e:
            self.logger.warning(f"Could not add directory to local git safe.directory: {e}")

        email = self.git_runner.run(['git', 'config', '--get', 'user.email'], check=False).stdout
        name = self.git_runner.run(['git', 'config', '--get', 'user.name'], check=False).stdout

        if not (email and email.strip()) or not (name and name.strip()):
            self.logger.info("Configuring default git user identity")
            self.git_runner.run(['git', 'config', 'user.email', 'zerorepo@bot.com'], check=False)
            self.git_runner.run(['git', 'config', 'user.name', 'ZeroRepo Bot'], check=False)

        main_exists = self.git_runner.run(
            ['git', 'rev-parse', '--verify', f'refs/heads/{self.main_branch}']
        ).success

        if not main_exists:
            alt_branch = "main" if self.main_branch == "master" else "master"
            alt_exists = self.git_runner.run(
                ['git', 'rev-parse', '--verify', f'refs/heads/{alt_branch}']
            ).success

            if alt_exists:
                self.logger.info(f"Main branch '{self.main_branch}' not found, using existing '{alt_branch}' instead")
                self.main_branch = alt_branch
                main_exists = True

        if not main_exists:
            self.logger.info(f"Main branch '{self.main_branch}' does not exist, creating...")
            has_any_commit = self.git_runner.run(['git', 'rev-parse', 'HEAD'], check=False).success

            if has_any_commit:
                self.logger.info(f"Creating '{self.main_branch}' branch from current HEAD")
                create_result = self.git_runner.run(['git', 'branch', self.main_branch], check=False)
                if not create_result.success:
                    self.logger.warning(f"Failed to create branch '{self.main_branch}': {create_result.stderr}")
            else:
                self.logger.info("No commits found, creating initial commit...")
                readme_path = self.repo_path / "README.md"
                if not readme_path.exists():
                    content = f"# {self.repo_path.name}\n\nManaged by ZeroRepo.\n"
                    readme_path.write_text(content, encoding='utf-8')

                self.git_runner.run(['git', 'checkout', '-b', self.main_branch], check=False)
                self.git_runner.run(['git', 'add', '-A'], check=True)
                commit_result = self.git_runner.run(
                    ['git', 'commit', '-m', 'Initial commit', '--allow-empty'], check=False
                )
                if not commit_result.success:
                    self.logger.error(f"Initial commit failed: {commit_result.stderr}")
                    raise RuntimeError(f"Could not create initial commit: {commit_result.stderr}")
                
        status_result = self.git_runner.run(['git', 'status', '--porcelain'], check=False)
        has_changes = bool(status_result.stdout and status_result.stdout.strip())

        if has_changes:
            self.logger.info("Found uncommitted changes, stashing before checkout...")
            stash_result = self.git_runner.run(['git', 'stash', 'push', '-m', 'Auto-stash before checkout to main'], check=False)
            if stash_result.success:
                self.logger.info("Changes stashed successfully")
            else:
                self.logger.warning(f"Failed to stash changes: {stash_result.stderr}")

        checkout_result = self.git_runner.run(['git', 'checkout', self.main_branch], check=False)
        if not checkout_result.success:
            self.logger.error(f"Failed to checkout '{self.main_branch}': {checkout_result.stderr}")
            raise RuntimeError(f"Could not checkout main branch '{self.main_branch}': {checkout_result.stderr}")

        verify_result = self.git_runner.run(
            ['git', 'rev-parse', f'{self.main_branch}^{{commit}}']
        )
        commit_hash = verify_result.stdout
        if not verify_result.success:
            raise RuntimeError(f"Failed to ensure main branch '{self.main_branch}' has commits. STDERR: {verify_result.stderr}")

        self.logger.info(f"Git repository initialized with branch '{self.main_branch}' at {commit_hash.strip()[:8]}")

    def _ensure_main_branch_exists(self):
        """Ensure the main branch exists and is checked out.

        Handles the following scenarios:
        1. Brand-new repo on an unborn branch -> create an initial commit, then rename the branch
        2. Repo with existing commits, currently on master/main -> rename to the target main branch
        3. Repo with existing commits, currently on a task/* branch, and master does not exist -> create master from HEAD
        4. Repo with existing commits, master exists -> simply check out master
        """

        branch_result = self.git_runner.run(['git', 'branch', '--show-current'], check=False)
        current_branch = branch_result.stdout.strip() if branch_result.success else ''

        if current_branch == self.main_branch:
            self.logger.debug(f"Already on main branch '{self.main_branch}'")
            return

        branch_exists = self.git_runner.run(
            ['git', 'rev-parse', '--verify', f'refs/heads/{self.main_branch}']
        ).success

        if branch_exists:
            self.logger.info(f"Switching to existing main branch '{self.main_branch}'")
            self.git_runner.run(['git', 'checkout', self.main_branch], check=True)
        else:
            is_safe_to_rename = current_branch in ('master', 'main', '')

            if is_safe_to_rename:
                self.logger.info(f"Renaming current branch '{current_branch or '(default)'}' to '{self.main_branch}'")
                rename_result = self.git_runner.run(['git', 'branch', '-m', self.main_branch], check=False)

                if not rename_result.success:
                    self.logger.warning(f"Rename failed ({rename_result.stderr}), trying checkout -b")
                    self.git_runner.run(['git', 'checkout', '-b', self.main_branch], check=True)
            else:
                self.logger.info(
                    f"Currently on '{current_branch}', creating main branch '{self.main_branch}' from HEAD"
                )
                create_result = self.git_runner.run(['git', 'branch', self.main_branch], check=False)
                if create_result.success:
                    self.logger.info(f"Created main branch '{self.main_branch}', switching to it")
                    self.git_runner.run(['git', 'checkout', self.main_branch], check=True)
                else:
                    self.logger.warning(f"git branch failed ({create_result.stderr}), trying checkout -b")
                    checkout_result = self.git_runner.run(['git', 'checkout', '-b', self.main_branch], check=False)
                    if not checkout_result.success:
                        self.logger.error(
                            f"Failed to create main branch '{self.main_branch}': {checkout_result.stderr}"
                        )
                        raise RuntimeError(
                            f"Could not create main branch '{self.main_branch}'. "
                            f"Current branch: '{current_branch}', Error: {checkout_result.stderr}"
                        )
                

    def _initialize_repo_state(self):
        """Initialize repository skeleton and RPG from existing files if not provided"""
        if self.rpg_path.exists():
            try:
                with open(self.rpg_path, 'r', encoding="utf-8") as f:
                    rpg_data = json.load(f)
                self.repo_rpg = RPG.from_dict(rpg_data)
                self.logger.info(f"Loaded RPG from {self.rpg_path}")
            except Exception as e:
                self.logger.error(f"Failed to load RPG: {e}")
                self.repo_rpg = RPG(repo_name=self.repo_path.name)
        else:
            self.repo_rpg = RPG(repo_name=self.repo_path.name)
            self.logger.info("Created new RPG instance")

        if self.global_rpg_path.exists():
            try:
                with open(self.global_rpg_path, 'r', encoding="utf-8") as f:
                    global_rpg_data = json.load(f)
                self.global_rpg = RPG.from_dict(data=global_rpg_data)
            except Exception as e:
                self.logger.error(f"Failed to load global RPG: {e}")
                self.global_rpg = None
        else:
            self.logger.warning(f"Global RPG file not found: {self.global_rpg_path}")
            self.global_rpg = None
 
        try:
            self.repo_skeleton = RepoSkeleton.from_workspace(self.repo_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize repo skeleton: {e}")
            self.repo_skeleton = None
        
        
    def _update_rpg_for_batch(self, batch: TaskBatch):
        """Update RPG with implemented units for a batch"""
        if not self.repo_rpg or not batch or not self.global_rpg:
            return
        
        file_path = batch.file_path
        units_key = batch.units_key
        unit_to_features = batch.unit_to_features
        
        self.logger.info(f"Updating RPG for batch: {file_path} with {len(units_key)} units")
        
        # Extract all relevant features from the batch
        all_features = []
        for unit_name in units_key:
            if unit_name in unit_to_features:
                features = unit_to_features[unit_name]
                if isinstance(features, list):
                    all_features.extend(features)
                else:
                    all_features.append(str(features))
        
        if not all_features:
            self.logger.warning(f"No features found for units: {units_key}")
            return
        
        # Find matching nodes in global_rpg by feature path or name
        nodes_to_copy = []
        edges_to_copy = []
        
        for feature in all_features:
            feature_str = str(feature).strip()
            if not feature_str:
                continue
                
            # Try to find nodes by feature path first
            node = self.global_rpg.get_node_by_feature_path(feature_str)
            
            assert node, f"Could not find node for {feature_str}"
        
            if node.id not in [n.id for n in nodes_to_copy]:
                nodes_to_copy.append(node)
                self.logger.debug(f"Found matching node: {node.name} (ID: {node.id})")
    
        if not nodes_to_copy:
            self.logger.warning(f"No matching nodes found in global RPG for features: {all_features}")
            return
        
        # Collect relevant edges - include edges between copied nodes and their ancestors
        copied_node_ids = {node.id for node in nodes_to_copy}
        
        # Add ancestor nodes and paths to root for context
        ancestor_ids_to_add = set()
        for node in nodes_to_copy:
            path_to_root = self.global_rpg.get_path_to_root(node.id)
            for ancestor_id in path_to_root:
                if ancestor_id not in copied_node_ids:
                    ancestor_ids_to_add.add(ancestor_id)
        
        # Get ancestor nodes and add them to nodes to copy
        ancestors_to_add = []
        for ancestor_id in ancestor_ids_to_add:
            ancestor_node = self.global_rpg.get_node_by_id(ancestor_id)
            if ancestor_node:
                ancestors_to_add.append(ancestor_node)
        
        nodes_to_copy.extend(ancestors_to_add)
        copied_node_ids.update(ancestor_ids_to_add)
        
        # Find edges involving copied nodes
        for edge in self.global_rpg.edges:
            if edge.src in copied_node_ids and edge.dst in copied_node_ids:
                edges_to_copy.append(edge)
        
        # Copy nodes to repo_rpg (avoid duplicates)
        added_nodes = 0
        for node in nodes_to_copy:
            if node.id not in self.repo_rpg.nodes:
                new_node = Node(
                    id=node.id,
                    name=node.name,
                    node_type=node.node_type,
                    level=node.level,
                    unit=node.unit,
                    meta=node.meta
                )
                self.repo_rpg.add_node(new_node)
                added_nodes += 1
                self.logger.debug(f"Added node to repo RPG: {node.name}")
        
        # Copy edges to repo_rpg (avoid duplicates)
        added_edges = 0
        existing_edges = {(edge.src, edge.dst) for edge in self.repo_rpg.edges}
        
        for edge in edges_to_copy:
            edge_key = (edge.src, edge.dst)
            if edge_key not in existing_edges:
                self.repo_rpg.add_edge(
                    src=edge.src,
                    dst=edge.dst, 
                    relation=edge.relation,
                    meta=edge.meta
                )
                existing_edges.add(edge_key)
                added_edges += 1
                self.logger.debug(f"Added edge to repo RPG: {edge.src} -> {edge.dst}")
        
        self.repo_rpg.update_all_metadata_bottom_up()
        self.logger.info(f"Updated repo RPG: added {added_nodes} nodes and {added_edges} edges")
        
    def _load_state(self):
        """Load task manager state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding="utf-8") as f:
                    state = json.load(f)
                
                self.completed_tasks = [
                    TaskState.from_dict(t) for t in state.get('completed_tasks', [])
                ]
                self.failed_tasks = [
                    TaskState.from_dict(t) for t in state.get('failed_tasks', [])
                ]
                if state.get('current_task'):
                    self.current_task = TaskState.from_dict(state['current_task'])
                    
                self.logger.info(f"Loaded state: {len(self.completed_tasks)} completed, "
                               f"{len(self.failed_tasks)} failed tasks")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save task manager state to file"""
        state = {
            "feature_selection": True,
            "feature_refactoring": True,
            "build_skeleton": True,
            "build_function": True,
            "plan_tasks": True,
            "build_skeleton": True,
            "plan_tasks": True,
            "code_generation": False,
            'completed_tasks': [t.to_dict() for t in self.completed_tasks],
            'failed_tasks': [t.to_dict() for t in self.failed_tasks],
            'current_task': self.current_task.to_dict() if self.current_task else None,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w', encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
        # Also save RPG state
        if self.repo_rpg:
            try:
                with open(self.rpg_path, 'w', encoding="utf-8") as f:
                    json.dump(self.repo_rpg.to_dict(), f, indent=2)
                self.logger.debug(f"Saved RPG to {self.rpg_path}")
            except Exception as e:
                self.logger.error(f"Failed to save RPG: {e}")
    
    def _ensure_clean_workspace(self) -> bool:
        """Ensure git workspace is clean before switching branches

        Strategy: before starting each task, commit any uncommitted changes.
        This ensures previous work is not lost and also avoids stash conflict issues.
        """
        status_output = self.git_runner.run(['git', 'status', '--porcelain'], check=True).stdout
        if not status_output.strip():
            return True  
        
        current_branch = self.git_runner.run(['git', 'branch', '--show-current'], check=True).stdout
        current_branch = current_branch.strip() if current_branch else 'unknown'

        self.logger.info(f"Uncommitted changes detected on '{current_branch}', committing...")

        self.git_runner.run(['git', 'add', '-A'], check=True)
        commit_result = self.git_runner.run(
            ['git', 'commit', '-m', f'[Auto] Save changes before new task on {current_branch}'],
        )

        if commit_result.success:
            self.logger.info(f"Auto-committed changes on branch '{current_branch}'")
        else:
            if 'nothing to commit' in (commit_result.stderr or '').lower():
                self.logger.debug("No changes to commit")
            else:
                self.logger.warning(f"Auto-commit failed: {commit_result.stderr[:100] if commit_result.stderr else 'unknown'}")

        self._active_stash_ref = None
        return True

    def _restore_stash_if_any(self):
        """Restore previously stashed changes (legacy support)
        """
        if not self._active_stash_ref:
            return

        self.logger.info(f"Restoring legacy stash {self._active_stash_ref}...")

        apply_ok = self.git_runner.run(['git', 'stash', 'apply', self._active_stash_ref], check=False).success
        if apply_ok:
            self.git_runner.run(['git', 'stash', 'drop', self._active_stash_ref], check=False)
            self.logger.info("Stash restored successfully")
        else:
            self.logger.warning(
                f"Failed to restore stash {self._active_stash_ref}. "
                f"Stash preserved for manual recovery."
            )

        self._active_stash_ref = None
        
    def _create_task_branch(self, task_id: str) -> str:
        """Create a new branch for the task.

        Ensure that:
        1. The main branch exists and has at least one commit.
        2. The task branch is created from the main branch.
        """
        branch_name = f"task/{task_id}"

        self._ensure_main_branch_exists()

        has_commit = self.git_runner.run(['git', 'rev-parse', f'{self.main_branch}^{{commit}}'], check=False).success
        if not has_commit:
            self.logger.error(f"Main branch '{self.main_branch}' has no commits, cannot create task branch")
            raise RuntimeError(f"Main branch '{self.main_branch}' has no commits")

        checkout_result = self.git_runner.run(['git', 'checkout', self.main_branch], check=False)
        if not checkout_result.success:
            self.logger.error(f"Failed to checkout branch '{self.main_branch}': {checkout_result.stderr}")
            raise RuntimeError(f"Failed to checkout main branch '{self.main_branch}'")

        checkout_b_success = self.git_runner.run(['git', 'checkout', '-b', branch_name], check=False).success
        if not checkout_b_success:
            self.logger.info(f"Branch '{branch_name}' may already exist, trying to checkout")
            checkout_task_success = self.git_runner.run(['git', 'checkout', branch_name], check=False).success
            if not checkout_task_success:
                self.logger.error(f"Failed to create or checkout task branch '{branch_name}'")
                raise RuntimeError(f"Failed to create task branch '{branch_name}'")

        return branch_name
    
    def _checkout_main_branch_safe(self) -> bool:
        """
        Safely checkout main branch.
        Strategy:
        1) Abort in-progress git operations (merge/rebase/cherry-pick/revert) if detected.
        2) Ensure main branch exists (and repo has at least one commit).
        3) Try checkout main; if blocked by local changes, auto-commit them then retry.
        4) If still fails, do NOT force discard by default (avoid data loss). Return False / raise.
        """
        git_dir = self.repo_path / ".git"

        if not git_dir.exists():
            self._ensure_git_repository()

        in_progress_flags = {
            "merge": (git_dir / "MERGE_HEAD").exists(),
            "cherry_pick": (git_dir / "CHERRY_PICK_HEAD").exists(),
            "revert": (git_dir / "REVERT_HEAD").exists(),
            "rebase": (git_dir / "rebase-apply").exists() or (git_dir / "rebase-merge").exists(),
        }

        if any(in_progress_flags.values()):
            self.logger.warning(f"Git operation in progress detected: {in_progress_flags}. Trying to abort safely...")

            if in_progress_flags["merge"]:
                self.git_runner.run(["git", "merge", "--abort"], check=False)
            if in_progress_flags["rebase"]:
                self.git_runner.run(["git", "rebase", "--abort"], check=False)
            if in_progress_flags["cherry_pick"]:
                self.git_runner.run(["git", "cherry-pick", "--abort"], check=False)
            if in_progress_flags["revert"]:
                self.git_runner.run(["git", "revert", "--abort"], check=False)
                
            still_bad = (
                (git_dir / "MERGE_HEAD").exists()
                or (git_dir / "CHERRY_PICK_HEAD").exists()
                or (git_dir / "REVERT_HEAD").exists()
                or (git_dir / "rebase-apply").exists()
                or (git_dir / "rebase-merge").exists()
            )
            if still_bad:
                msg = "Failed to abort in-progress git operation; refusing to checkout main to avoid corruption."
                self.logger.error(msg)
                return False

        try:
            self._ensure_main_branch_exists()
        except Exception as e:
            self.logger.warning(f"_ensure_main_branch_exists failed ({e}), falling back to _ensure_git_repository...")
            self._ensure_git_repository()

        current = self.git_runner.run(["git", "branch", "--show-current"], check=False).stdout
        current = (current or "").strip()
        if current == self.main_branch:
            return True

        checkout_result = self.git_runner.run(["git", "checkout", self.main_branch], check=False)
        if checkout_result.success:
            return True

        self.logger.warning(f"Checkout '{self.main_branch}' failed: {checkout_result.stderr.strip() if checkout_result.stderr else checkout_result.stderr}. Trying auto-commit and retry...")
        self._ensure_clean_workspace()

        checkout_result = self.git_runner.run(["git", "checkout", self.main_branch], check=False)
        if checkout_result.success:
            return True

        self.logger.error(f"Failed to checkout '{self.main_branch}' after retry: {checkout_result.stderr.strip() if checkout_result.stderr else checkout_result.stderr}")
        return False

    def _commit_changes(self, message: str) -> Optional[str]:
        """Commit current changes and return commit hash"""
        self.git_runner.run(['git', 'add', '-A'], check=True)

        status_output = self.git_runner.run(['git', 'status', '--porcelain'], check=True).stdout
        if not status_output.strip():
            self.logger.info("No changes to commit")
            return None

        commit_success = self.git_runner.run(['git', 'commit', '-m', message], check=False).success
        if not commit_success:
            return None

        verify_result = self.git_runner.run(['git', 'rev-parse', 'HEAD'], check=False)
        commit_hash = verify_result.stdout
        return commit_hash.strip() if verify_result.success else None

    def _rollback_task(self, task_state: TaskState):
        """Rollback a failed task"""
        self.logger.info(f"Rolling back task {task_state.task_id}")

        checkout_success = self.git_runner.run(['git', 'checkout', self.main_branch], check=False).success
        if not checkout_success:
            self.logger.warning(
                f"Failed to checkout branch '{self.main_branch}' during rollback; "
                f"will NOT delete branch '{task_state.branch_name}' to avoid data loss"
            )
        else:
            if task_state.branch_name != self.main_branch:
                self.logger.info(f"Deleting task branch '{task_state.branch_name}'")
                self.git_runner.run(['git', 'branch', '-D', task_state.branch_name], check=False)
            else:
                self.logger.error(
                    f"Refusing to delete branch '{self.main_branch}' in rollback "
                    f"for task {task_state.task_id}"
                )
        
        task_state.status = 'failed'
        self.failed_tasks.append(task_state)
        self._save_state()


    def _merge_task(self, task_state: TaskState) -> bool:
        """Merge successful task to main branch"""
        self.logger.info(f"Merging task {task_state.task_id}")

        checkout_success = self.git_runner.run(['git', 'checkout', self.main_branch], check=False).success
        if not checkout_success:
            self.logger.error(f"Failed to checkout branch '{self.main_branch}' for merge")
            return False

        merge_success = self.git_runner.run(
            ['git', 'merge', '--no-ff', task_state.branch_name, '-m',
             f'Merge task {task_state.task_id}: {task_state.batch.file_path}'],
            check=False,
        ).success

        if merge_success:
            self.git_runner.run(['git', 'branch', '-d', task_state.branch_name], check=False)
            task_state.status = 'success'
            task_state.completed_at = datetime.now().isoformat()
            self.completed_tasks.append(task_state)

            try:
                self.repo_skeleton = RepoSkeleton.from_workspace(self.repo_path)
            except Exception as e:
                self.logger.error(f"Failed to refresh repo skeleton after merge: {e}")

            self._save_state()

        return merge_success

    def _init_batch_to_branch(self, batch: TaskBatch):
        # Check task type - skip writing code for special task types
        task_type = getattr(batch, 'task_type', 'implementation')
        
        # Check for special file path markers
        if batch.file_path.startswith('<') and batch.file_path.endswith('>'):
            self.logger.info(f"Skipping code write for special task with marker {batch.file_path}: {batch.task_id}")
            return
        
        if task_type in ['integration_test', 'final_test_docs']:
            self.logger.info(f"Skipping code write for {task_type} task: {batch.task_id}")
            return
        
        file_path = self.repo_path / batch.file_path
        unit_to_code = batch.unit_to_code

        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = "\n".join(unit_to_code.values())
        if content and not content.endswith("\n"):
            content += "\n"

        if file_path.exists():
            with file_path.open("a", encoding="utf-8") as f:
                try:
                    if file_path.stat().st_size > 0:
                        with file_path.open("rb") as bf:
                            bf.seek(-1, os.SEEK_END)
                            last = bf.read(1)
                        if last not in (b"\n", b"\r"):
                            f.write("\n\n\n")
                except Exception:
                    pass
                f.write(content)
        else:
            file_path.write_text(content, encoding="utf-8")
            

    def start_task(self, batch: TaskBatch) -> TaskState:
        """Start a new task with optional skeleton and RPG updates
        """
        task_id = getattr(batch, "task_id", None)
        if not task_id:
            task_id = f"{batch.file_path.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            setattr(batch, "task_id", task_id)

        self._ensure_clean_workspace()
        self._checkout_main_branch_safe()

        current_branch = self.git_runner.run(['git', 'branch', '--show-current'], check=False).stdout
        if current_branch.strip() != self.main_branch:
            self._force_switch_to_main()
            final_check = self.git_runner.run(['git', 'branch', '--show-current'], check=False).stdout
            if final_check.strip() != self.main_branch:
                raise RuntimeError(f"Failed to switch to main branch '{self.main_branch}' before starting task.")

        branch_name = self._create_task_branch(task_id)

        # Record the initial commit hash after creating the task branch
        # This will be used as the baseline for testing (instead of main branch)
        verify_result = self.git_runner.run(['git', 'rev-parse', 'HEAD'], check=False)
        if not verify_result.success:
            initial_commit = None
            self.logger.warning(f"Failed to get initial commit for task {task_id}")
        else:
            initial_commit = verify_result.stdout.strip()
            self.logger.info(f"Task {task_id} initial commit: {initial_commit[:8]}")
        
        # Add initial_commit to batch object for IterativeCodeGenerator to use
        setattr(batch, 'initial_commit', initial_commit)
        
        self._init_batch_to_branch(batch)

        task_state = TaskState(
            task_id=task_id,
            batch=batch,
            branch_name=branch_name,
            status='in_progress',
            created_at=datetime.now().isoformat(),
            initial_commit=initial_commit
        )
        
        self.current_task = task_state
        self._save_state()
        
        self.logger.info(f"Started task {task_id} on branch {branch_name}")

        try:
            self.repo_skeleton = RepoSkeleton.from_workspace(self.repo_path)
        except Exception as e:
            self.logger.error(f"Failed to refresh repo skeleton at task start: {e}")

        return task_state
        
    def complete_task(
        self, 
        success: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """Complete current task with success/failure status and sync repository state"""
        if not self.current_task:
            self.logger.error("No active task to complete")
            self._force_switch_to_main()
            return False

        try:
            commit_message = f"Implement {self.current_task.batch.file_path}"
            if getattr(self.current_task.batch, "units_key", None):
                unit_names = self.current_task.batch.units_key[:3]
                commit_message += f": {', '.join(unit_names)}"
                if len(self.current_task.batch.units_key) > 3:
                    commit_message += f" and {len(self.current_task.batch.units_key) - 3} more"
                    
            commit_hash = self._commit_changes(commit_message)
            self.current_task.commit_hash = commit_hash
            
            if success:
                merge_success = self._merge_task(self.current_task)
                if not merge_success:
                    success = False
                    error_message = "Failed to merge task"
                else:
                    # Check if this is a special task type that should skip RPG updates
                    task_type = getattr(self.current_task.batch, 'task_type', 'implementation')
                    file_path = self.current_task.batch.file_path
                    
                    # Skip RPG update for integration tests, final tests, and special marker tasks
                    if (task_type in ['integration_test', 'final_test_docs'] or 
                        (file_path.startswith('<') and file_path.endswith('>'))):
                        self.logger.info(f"Skipping RPG update for {task_type} task: {self.current_task.task_id}")
                    else:
                        self._update_rpg_for_batch(self.current_task.batch)
            
            if not success:
                self.current_task.error_message = error_message
                self._rollback_task(self.current_task)
                
            self.current_task = None
            self._save_state()
            return success

        finally:
            self._restore_stash_if_any()
            self._force_switch_to_main()
            
    
    def _has_executed_tasks(self) -> bool:
        """Check if any tasks have been executed before"""
        return len(self.completed_tasks) > 0 or len(self.failed_tasks) > 0 or self.current_task is not None
    
    def _initialize_base_classes_if_needed(self):
        """Initialize base classes and repository info if no tasks have been executed"""
        if not self._has_executed_tasks():
            self.logger.info("No previous tasks found, initializing repository as starting point")
            self._setup_initial_repository()
    
    def _setup_initial_repository(self):
        """Setup initial repository with README and base classes from graph data"""
        # Ensure we're on main branch and clean
        self._ensure_clean_workspace()
        checkout_success = self.git_runner.run(['git', 'checkout', self.main_branch], check=False).success
        if not checkout_success:
            self.logger.warning(f"Failed to checkout branch '{self.main_branch}' for repository setup")
        
        has_changes = False
        
        # Create README.md with repository info from RPG
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            repo_name = self.repo_rpg.repo_name if self.repo_rpg else "Unknown Repository"
            repo_info = getattr(self.repo_rpg, 'repo_info', None) if self.repo_rpg else None
            
            readme_content = f"# {repo_name}\n\n"
            if repo_info:
                readme_content += f"{repo_info}\n\n"
            # readme_content += "This repository is managed by ZeroRepo.\n"
            
            readme_path.write_text(readme_content, encoding='utf-8')
            self.logger.info(f"Created README.md for repository: {repo_name}")
            has_changes = True
        
        # Add ignore rules for Python bytecode / caches
        gitignore_path = self.repo_path / ".gitignore"
        ignore_block = (
            "\n"
            "# --- Python bytecode / cache ---\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            "*$py.class\n"
        )

        def _ensure_gitignore_block(path, block) -> bool:
            """Return True if file changed."""
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if block.strip() in content:
                    return False
                path.write_text(content.rstrip() + block, encoding="utf-8")
                return True
            else:
                path.write_text(block.lstrip("\n"), encoding="utf-8")
                return True

        if _ensure_gitignore_block(gitignore_path, ignore_block):
            self.logger.info("Updated .gitignore to ignore __pycache__ and *.pyc")
            has_changes = True

        # If any pyc files were already tracked, untrack them (keep files on disk)
        # This prevents future merges from bringing them back.
        self.git_runner.run(
            ['git', 'rm', '-r', '--cached', '--ignore-unmatch', '**/__pycache__', '*.pyc', '*.pyo'],
            check=False,
        )

        # Setup base classes from graph data if available
        if self.graph_data and 'base_classes_phase' in self.graph_data:
            base_classes = self.graph_data['base_classes_phase'].get('base_classes', [])
            if base_classes:
                self.logger.info(f"Setting up {len(base_classes)} base classes")

                # Write base classes to files
                for base_class in base_classes:
                    file_path = self.repo_path / base_class['file_path']
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    self.logger.info(f"Writing base class to {file_path}")
                    file_path.write_text(base_class['code'], encoding='utf-8')
                    has_changes = True
            else:
                self.logger.info("No base classes found in graph data")
        else:
            self.logger.info("No graph data or base_classes_phase available")

        # Commit the changes if any
        if has_changes:
            self.git_runner.run(['git', 'add', '-A'], check=True)
            commit_success = self.git_runner.run([
                'git', 'commit', '-m',
                'Initial repository setup: Add README and base classes'
            ], check=False).success

            if commit_success:
                self.logger.info(f"Repository initialized successfully on branch {self.main_branch}")
            else:
                self.logger.warning("Failed to commit initial setup or no changes to commit")
        else:
            self.logger.info("Repository already initialized, skipping setup")
        
        # Restore any stashed changes
        self._restore_stash_if_any()
    
    def execute_task_batch(
        self,
        batch: TaskBatch,
        task_executor  # Function that executes a batch and returns (success, error_msg)
    ) -> bool:
        """Execute a single task batch"""
        
        # For Exp Run
        repo_name = self.global_rpg.repo_name if self.global_rpg else "unknown_repo"
        task_desc = (
            f"## Task Requirements\nYou are working on reconstructing the repository '{repo_name}' from scratch. "
            "Do NOT import, vendor, wrap, or call the original external library you are re-implementing "
            "(directly or indirectly). Treat this as a fully standalone local implementation.\n\n"
            "Hard requirement: ALL required abstractions and algorithms must be implemented end-to-end from zero. "
            "Even if the target library provides high-level objects (e.g., DataFrame/Series/Index), "
            "you must implement their core data structures, behaviors, and algorithms yourself.\n\n"
            "No placeholders: do NOT leave stubs, TODOs, or partial implementations. "
            "Do NOT use `pass`, `...`, `raise NotImplementedError`, `raise Exception('not implemented')`, "
            "or any equivalent 'not supported' fallback for required APIs. "
            "Every specified API must be fully functional, testable, and produce correct outputs."
        )
        batch.task = task_desc + "\n\n" + batch.task
        
        task_state = self.start_task(batch)
        try:
            success, error_msg = task_executor(batch, self.repo_skeleton, self.repo_rpg)
            self.complete_task(success, error_msg)
            
            if success:
                self.logger.info(f"Task {task_state.task_id} completed successfully")
            else:
                self.logger.error(f"Task {task_state.task_id} failed: {error_msg}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Unexpected error in task {task_state.task_id}: {e}")
            self.complete_task(False, str(e))
            return False
        finally:
            try:
                self._ensure_main_branch_exists()
            except Exception as e:
                self.logger.error(f"Failed to ensure {self.main_branch} branch after task {task_state.task_id}: {e}")
    
    def execute_batch_list(
        self,
        batches: List[TaskBatch],
        task_executor
    ) -> Dict[str, int]:
        """Execute a list of pre-generated task batches"""
        
        self.logger.info(f"🚀 Starting batch execution: {len(batches)} batches queued")

        stats = {
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'total': len(batches)
        }

        start_time = datetime.now()
        
        executed_keys = set()
        for t in (self.completed_tasks + self.failed_tasks):
            key = (t.batch.file_path, tuple(t.batch.units_key))
            executed_keys.add(key)

        for i, batch in enumerate(batches, 1):
            self.logger.info(f"\n📋 Processing batch {i}/{len(batches)}: {batch.file_path}")
            
            key = (batch.file_path, tuple(batch.units_key))
            if key in executed_keys:
                self.logger.info(
                    f"⏭️ Skipping already executed task: {batch.file_path} with units {batch.units_key}"
                )
                stats['skipped'] += 1
                continue
        
            try:
                batch_start = datetime.now()
            
                success = self.execute_task_batch(batch, task_executor)
                batch_duration = datetime.now() - batch_start
                
                if success:
                    stats['completed'] += 1
    
                    self.logger.info(f"✅ Batch {i} completed successfully in {batch_duration.total_seconds():.2f}s")
                else:
                    stats['failed'] += 1
                    self.logger.error(f"❌ Batch {i} failed after {batch_duration.total_seconds():.2f}s")

                # Progress update
                progress = (stats['completed'] + stats['failed'] + stats['skipped']) / len(batches) * 100
                self.logger.info(f"📊 Progress: {progress:.1f}% ({stats['completed']} completed, {stats['failed']} failed, {stats['skipped']} skipped)")
                
            except KeyboardInterrupt:
                self.logger.info("⚠️ Process interrupted by user")
                self._save_state()
                raise
        
        total_duration = datetime.now() - start_time
        self.logger.info(f"\n🏁 Batch execution completed in {total_duration.total_seconds():.2f}s")
        self.logger.info(f"📈 Final stats: {stats}")
        
        return stats
        
    def get_repo_skeleton(self) -> Optional[RepoSkeleton]:
        """Get current repository skeleton"""
        return self.repo_skeleton
    
    def get_repo_rpg(self) -> Optional[RPG]:
        """Get current repository RPG"""
        return self.repo_rpg
    
    def update_repo_state(self, skeleton: Optional[RepoSkeleton] = None, 
                         rpg: Optional[RPG] = None):
        """Update repository state manually"""
        if skeleton:
            self.repo_skeleton = skeleton
            self.logger.info("Updated repository skeleton")
        if rpg:
            self.repo_rpg = rpg
            self.logger.info("Updated repository RPG")
        self._save_state()