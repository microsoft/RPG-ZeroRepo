import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import os
from datetime import datetime

from .rpg_gen.base import LLMConfig
from .rpg_gen.base import RPG, RepoSkeleton
from .rpg_gen.prop_level.prop_builder import PropBuilder
from .rpg_gen.impl_level.impl_builder import ImplBuilder
from .rpg_gen.impl_level.plan_tasks.task_manager import TaskManager
from .rpg_gen.impl_level.plan_tasks.task_batch import TaskBatch
from .code_gen.code_gen import IterativeCodeGenerator
from .utils.logs import setup_logger
from .config.checkpoint_config import (
    CheckpointManager, CheckpointFiles, create_default_manager, set_checkpoint_manager
)


class PersistentTaskExecutor:
    """Task executor that maintains a single IterativeCodeGenerator instance"""
    
    def __init__(
        self,
        repo_path: Path,
        checkpoint_manager: CheckpointManager,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        self.repo_path = repo_path
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.logger = logger
        self.generator = None
    
    def __enter__(self):
        """Initialize the IterativeCodeGenerator"""
        self.logger.info("🐳 Initializing IterativeCodeGenerator container...")
        
        code_gen_cfg = self.config["code_generation"]

        self.generator = IterativeCodeGenerator(
            repo_path=self.repo_path,
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger,
            use_docker=code_gen_cfg.get("docker", {}).get("use_docker", True),
            docker_container_name=code_gen_cfg.get("docker", {}).get("container_name", "zerorepo"),
            container_workspace_path=code_gen_cfg.get("docker", {}).get("workspace", "/tare_workspace"),
            trae_config_file=code_gen_cfg.get("trae_agent", {}).get("trae_config", "./code_gen/trae-agent/trae_config.yaml"),
            docker_image=code_gen_cfg.get("docker", {}).get("image", "ubuntu:22.04"),
            force_rebuild=code_gen_cfg.get("docker", {}).get("force_rebuild", False),
            max_iterations=code_gen_cfg.get("trae_agent", {}).get("max_iterations", 5),
            max_retries_per_workflow=code_gen_cfg.get("trae_agent", {}).get("max_retries_per_workflow", 3),
            commit_llm_config=LLMConfig.from_dict(self.config["llm"]),
            failure_llm_config=LLMConfig.from_dict(self.config["llm"])
        )
        
        # Setup Docker components once
        if self.config.get("use_docker", True):
            self.generator._setup_docker_components()
            self.logger.info("✅ Container initialized and ready for batch processing")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the IterativeCodeGenerator"""
        if self.generator:
            self.logger.info("🧹 Cleaning up IterativeCodeGenerator container...")
            self.generator.cleanup_docker()
            self.generator = None
    
    def task_executor(self, batch, repo_skeleton, repo_rpg, **kwargs):
        """Execute a single task batch using the persistent generator"""
        if not self.generator:
            error_msg = "Task executor not properly initialized"
            self.logger.error(error_msg)
            return False, error_msg
        
        # Execute the batch using iterative approach
        self.logger.info(f"📋 Executing batch with persistent container")
        
        try:
            success, error_msg = self.generator.task_executor(
                batch=batch,
                repo_skeleton=repo_skeleton,
                repo_rpg=repo_rpg,
                **kwargs
            )
            
            if success:
                self.logger.info(f"✅ Batchcompleted successfully")
            else:
                self.logger.error(f"❌ Batch failed: {error_msg}")
            
            return success, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error in batch: {e}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False, str(e)


class ZeroRepo:
    """
    Main orchestrator class that chains together the complete ZeroRepo pipeline:
    prop_level (PropBuilder) -> impl_level (ImplBuilder) -> code_gen (IterativeCodeGenerator)
    
    Provides checkpoint management, state persistence, and configuration-driven execution.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        checkpoint_dir: Union[str, Path],
        repo_path: Union[str, Path],
        checkpoint_config: Optional[Union[CheckpointFiles, Dict]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ZeroRepo orchestrator with checkpoint management
        
        Args:
            config_path: Path to zerorepo_config.yaml
            checkpoint_dir: Directory for saving checkpoints and state
            repo_path: Target repository path for code generation
            checkpoint_config: Optional custom checkpoint configuration
            logger: Optional logger instance
        """
        self.config_path = Path(config_path)
        self.target_repo_path = Path(repo_path)
        
        # Create directories
        self.target_repo_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        base_logger = logger or logging.getLogger("ZeroRepo")
        self.logger = setup_logger(
            logger=base_logger,
            file_path=None,  # 或者 self.checkpoint_dir / "zerorepo.log"
            level=logging.INFO,
        )
        
        # Initialize checkpoint manager
        if isinstance(checkpoint_config, dict):
            checkpoint_config = CheckpointFiles(**checkpoint_config)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            file_config=checkpoint_config
        )
        
        # Set as global checkpoint manager
        set_checkpoint_manager(self.checkpoint_manager)
        
        # Load configuration
        self.config = self._load_config()
        
        # Save checkpoint configuration
        self.checkpoint_manager.save_config()
        
        # Initialize components - now don't need to pass checkpoint paths!
        self.prop_builder = None
        self.impl_builder = None
        self.code_generator = None
        
        # Initialize pipeline state - will be loaded properly after method definitions
        self.pipeline_state = {}
        
        self.logger.info(f"ZeroRepo initialized with config: {config_path}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_manager.checkpoint_dir}")
        self.logger.info(f"Target repository: {repo_path}")
        self.logger.info(f"Checkpoint files: {list(self.checkpoint_manager.get_all_paths().keys())}")
        
        # Load existing pipeline state after all methods are defined
        self.pipeline_state = self._load_state()
    
    # Properties for accessing checkpoint paths
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory"""
        return self.checkpoint_manager.checkpoint_dir
    
    @property 
    def repo_data_file(self) -> Path:
        """Get repo data file path"""
        return self.checkpoint_manager.repo_data_path
    
    @property
    def task_manager_state_file(self) -> Path:
        """Get task manager state file path"""
        return self.checkpoint_manager.get_path("task_manager_state")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")

    def _load_state(self) -> Dict[str, Any]:
        if not self.task_manager_state_file.exists():
            self.logger.info("No existing task manager state found. Starting fresh.")
            return {}
        with open(self.task_manager_state_file, 'r') as f:
            return json.load(f)
    
    def _save_pipeline_state(self) -> None:
        """Save current pipeline state to checkpoint file"""
        with open(self.task_manager_state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
        self.logger.debug(f"Pipeline state saved: {self.pipeline_state}")
    
    def _get_llm_config(self) -> LLMConfig:
        """Get LLM configuration for specific purpose"""
        llm_configs = self.config.get('llm', {})
        return LLMConfig.from_dict(llm_configs)
        
    def _initialize_prop_builder(self) -> None:
        """Initialize PropBuilder with configuration"""
        if self.prop_builder is None:
            prop_config = self.config.get('prop_level', {})
            
            self.prop_builder = PropBuilder(
                llm_cfg=self._get_llm_config(),
                logger=self.logger,
                checkpoint_manager=self.checkpoint_manager,  # Pass checkpoint manager
                **prop_config.get('feature_selection', {}),
                **prop_config.get('feature_refactoring', {})
            )
            self.logger.info("✅ PropBuilder initialized")
    
    def _initialize_impl_builder(self) -> None:
        """Initialize ImplBuilder with configuration"""
        if self.impl_builder is None:
            impl_config = self.config.get('impl_level', {})
            
            skeleton_cfg = impl_config['file_design_cfg_path']
            graph_cfg = impl_config['func_design_cfg_path']
            
            self.impl_builder = ImplBuilder(
                llm_cfg=self._get_llm_config(),
                skeleton_cfg=skeleton_cfg,
                graph_cfg=graph_cfg,
                repo_path=self.target_repo_path,
                ckpt_dir=self.checkpoint_dir,  # Keep for backward compatibility
                logger=self.logger,
                checkpoint_manager=self.checkpoint_manager  # Pass checkpoint manager
            )
            self.logger.info("✅ ImplBuilder initialized")
        
    
    def save_repo_data(self, repo_data: Dict[str, Any]) -> None:
        """Save repository data to checkpoint file"""
        try:
            with open(self.repo_data_file, 'w', encoding='utf-8') as f:
                json.dump(repo_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved repo data to {self.repo_data_file}")
        except Exception as e:
            self.logger.error(f"Failed to save repo data: {e}")
            raise
    
    def load_repo_data(self) -> Optional[Dict[str, Any]]:
        """Load repository data from checkpoint file"""
        if not self.repo_data_file.exists():
            return None
        
        try:
            with open(self.repo_data_file, 'r', encoding='utf-8') as f:
                repo_data = json.load(f)
            self.logger.info(f"Loaded repo data from {self.repo_data_file}")
            return repo_data
        except Exception as e:
            self.logger.error(f"Failed to load repo data: {e}")
            return None

    def run_prop_level(
        self, 
        repo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run complete property level processing (feature selection + refactoring)
        
        Args:
            repo_data: Repository data containing initial feature tree
            force_rerun: Whether to force rerun even if already completed
            
        Returns:
            Updated repo_data with refactored components
        """
        # Check if already completed
        
        cur_state = self._load_state()
        if cur_state.get("feature_selection", False) and cur_state.get("feature_refactoring", False):
            self.logger.info("Proposal-Level have completed, skip...")
            return repo_data
        
        # Initialize PropBuilder if not already initialized
        self._initialize_prop_builder()
        
        try:
            repo_data = self.prop_builder.build_feature_tree()
            if "error" in repo_data:
                raise ValueError(f"Error Existing in running proposal-level: {repo_data['error']}")
            if repo_data.get("Feature_tree", {}) or repo_data.get("Component", []):
                self.save_repo_data(repo_data)
            return repo_data
        except Exception as e:
            self.logger.error(f"Property level processing failed: {e}")
            raise
    
    def run_impl_level(self) -> Tuple[Any, Any, Any, Any]:
        """
        Run complete implementation level processing (skeleton + function design + task planning)
        
        Args:
            force_rerun: Whether to force rerun even if already completed
            
        Returns:
            Tuple of (plan_batches, skeleton, rpg, graph_data)
        """
        
        # Check if already completed
        if (self.pipeline_state.get("build_skeleton", False) and 
            self.pipeline_state.get("build_function", False) and 
            self.pipeline_state.get("plan_tasks", False)
        ):
            self.logger.info("Implementation level already completed, loading existing results")
            # Load from checkpoints
            with open(self.checkpoint_manager.skeleton_path, 'r') as f:
                skeleton_dict = json.load(f)
            with open(self.checkpoint_manager.global_rpg_path, 'r') as f:
                rpg_dict = json.load(f)
            with open(self.checkpoint_manager.graph_path, 'r') as f:
                graph_data = json.load(f)
            with open(self.checkpoint_manager.tasks_path, 'r') as f:
                batch_dict = json.load(f)
                
            skeleton = RepoSkeleton.from_dict(skeleton_dict)
            rpg = RPG.from_dict(rpg_dict)
            
            # Load task batches using shared utility function
            from .utils.task_loader import load_batches_from_dict
            planned_batches_dict = batch_dict.get("planned_batches_dict", {})
            plan_batches = load_batches_from_dict(planned_batches_dict, graph_data, self.logger)

            return plan_batches, skeleton, rpg, graph_data
        
        # Initialize ImplBuilder if not already initialized
        self._initialize_impl_builder()
        
        self.logger.info("=== Starting Implementation Level Processing ===")
        try:
            plan_batches, skeleton, rpg, graph_data = self.impl_builder.run()
            return plan_batches, skeleton, rpg, graph_data
        except Exception as e:
            self.logger.error(f"Implementation level processing failed: {e}")
            raise
    
    def run_code_generation(
        self, 
        plan_batches: Optional[List[Any]] = None,
        skeleton: Optional[Any] = None,
        rpg: Optional[Any] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        force_rerun: bool = False
    ) -> bool:
        """
        Run code generation using TaskManager and IterativeCodeGenerator with persistent container
        
        Args:
            plan_batches: Task batches from implementation level (optional, will be loaded if None)
            skeleton: Repository skeleton (optional, will be loaded if None)
            rpg: Repository Planning Graph (optional, will be loaded if None)
            graph_data: Graph data for task management (optional, will be loaded if None)
            force_rerun: Whether to force rerun even if already completed
            
        Returns:
            True if successful, False otherwise
        """
        if self.pipeline_state.get("code_generation", False) and not force_rerun:
            self.logger.info("Code generation already completed")
            return True
        
        self.logger.info("=== Starting Code Generation with TaskManager ===")
        
        try:
            # Load task batches and graph data if not provided
            if plan_batches is None or graph_data is None:
                tasks_file = Path(self.checkpoint_manager.tasks_path)
                graph_file = Path(self.checkpoint_manager.graph_path)
                
                
                if not tasks_file.exists():
                    self.logger.error(f"❌ No saved tasks found at {tasks_file}")
                    self.logger.error("Run implementation phase first")
                    return False
                
                if not graph_file.exists():
                    self.logger.error(f"❌ No graph data found at {graph_file}")
                    self.logger.error("Graph data is required for proper task ordering")
                    return False
                
                # Load tasks and graph data
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                with open(graph_file, 'r') as f:
                    graph_data = json.load(f)
                
                # Recreate task batches using shared utility function
                from .utils.task_loader import load_batches_from_file
                
                self.logger.info("Loading task batches with proper ordering...")
                plan_batches = load_batches_from_file(tasks_file, graph_data, self.logger)
                
                if not plan_batches:
                    self.logger.error("❌ No task batches found in saved data")
                    return False
                    
                self.logger.info(f"✅ Loaded {len(plan_batches)} task batches")
            
            # Setup TaskManager for coordinated execution
            state_file = self.checkpoint_manager.task_state_path
            rpg_path = self.checkpoint_manager.current_rpg_path
            global_rpg_path = self.checkpoint_manager.global_rpg_path
            
            task_manager = TaskManager(
                repo_path=str(self.target_repo_path),
                state_file=state_file,
                rpg_path=rpg_path,
                global_rpg_path=global_rpg_path,
                graph_data=graph_data,
                main_branch="master",
                logger=self.logger
            )
            
            # Create persistent task executor that reuses container
            with PersistentTaskExecutor(
                repo_path=self.target_repo_path,
                checkpoint_manager=self.checkpoint_manager,
                config=self.config,
                logger=self.logger
            ) as persistent_executor:
                
                # Execute all task batches with persistent container
                self.logger.info(f"Executing {len(plan_batches)} task batches with persistent container...")
                stats = task_manager.execute_batch_list(
                    batches=plan_batches,
                    task_executor=persistent_executor.task_executor
                )
            
            # ===== RESULTS =====
            self.logger.info("=== Code Generation Results ===")
            self.logger.info(f"📊 Final Statistics:")
            self.logger.info(f"   ✅ Completed: {stats['completed']}")
            self.logger.info(f"   ❌ Failed: {stats['failed']}")
            self.logger.info(f"   ⏭️ Skipped: {stats['skipped']}")
            self.logger.info(f"   📝 Total: {stats['total']}")
            
            success_rate = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            self.logger.info(f"   📈 Success Rate: {success_rate:.1f}%")
            
            # Update pipeline state
            if stats['completed'] == stats['total']:
                self.pipeline_state["code_generation"] = True
                self.pipeline_state["current_stage"] = "completed"
                self._save_pipeline_state()
                self.logger.info("🎉 All code generation tasks completed successfully!")
                return True
            else:
                self.logger.warning(f"⚠️ {stats['failed']} tasks failed, check logs for details")
                # Don't mark as completed if there were failures
                return False
            
        except KeyboardInterrupt:
            self.logger.info("⚠️ Code generation interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Code generation failed: {e} {e.__traceback__}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def run(
        self,
        repo_data: Dict=None
    ) -> bool:
        """
        Run the complete ZeroRepo pipeline from start to finish
        
        Args:
            repo_data: Initial repository data with feature tree
            force_rerun: Whether to force rerun all stages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=== Starting Complete ZeroRepo Pipeline ===")
            
            if not repo_data:
                if self.repo_data_file.exists():
                    with open(self.repo_data_file, 'r')  as f:
                        repo_data = json.load(f)
                else:
                    raise ValueError(f"Please Make Sure {self.repo_data_file} exists")
                
            # Stage 1: Property Level Processing
            self.run_prop_level(repo_data)
            
            # Stage 2: Implementation Level Processing
            plan_batches, skeleton, rpg, graph_data = self.run_impl_level()
            
            # Stage 3: Code Generation
            success = self.run_code_generation(plan_batches, skeleton, rpg, graph_data)
            
            if success:
                self.logger.info("=== Complete ZeroRepo Pipeline Completed Successfully ===")
            else:
                self.logger.error("=== Complete ZeroRepo Pipeline Failed ===")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Complete pipeline failed: {e}")
            return False
    