"""
Property-Level Builder

Main class that orchestrates the complete property-level workflow:
1. Feature Selection: Generate comprehensive feature trees from repository data
2. Feature Refactoring: Organize selected features into coherent subtrees/components

This provides a simple interface to run the complete select -> refactor pipeline.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from copy import deepcopy
from zerorepo.utils.tree import get_all_leaf_paths
from zerorepo.utils.logs import setup_logger
from zerorepo.config.checkpoint_config import CheckpointManager, get_checkpoint_manager
from ..base.llm_client import LLMConfig
from ..prop_level.select_feature import FeatureSelectAgent
from ..prop_level.refactor_feature import FeatureRefactorAgent

class PropBuilder:
    """
    Main property-level builder that orchestrates the complete workflow:
    
    1. Feature Selection: Use PropLevelAgent to generate comprehensive feature trees
    2. Feature Refactoring: Use FeatureRefactorAgent to organize features into subtrees
    
    This provides a unified interface for the select -> refactor pipeline.
    """
    
    def __init__(
        self,
        llm_cfg: LLMConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize the property builder with checkpoint management.
        
        Args:
            llm_cfg: LLM configuration for both selection and refactoring
            checkpoint_manager: CheckpointManager instance, if None uses global instance
            logger: Optional logger instance
            **kwargs: Additional configuration passed to agents
        """
        self.llm_cfg = llm_cfg
        self.logger = logger or setup_logger()
        self.kwargs = kwargs
        
        # Get checkpoint manager
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            try:
                self.checkpoint_manager = get_checkpoint_manager()
            except RuntimeError:
                # If no global manager, create a default one
                from config.checkpoint_config import create_default_manager
                self.checkpoint_manager = create_default_manager("./checkpoints")
        
        # Initialize agents (will be created when needed)
        self.selector_agent: Optional[FeatureSelectAgent] = None
        self.refactor_agent: Optional[FeatureRefactorAgent] = None
        
        self.logger.info(f"PropBuilder initialized with checkpoint dir: {self.checkpoint_manager.checkpoint_dir}")
    
    # Properties for accessing checkpoint paths
    @property
    def feature_analysis_path(self) -> Path:
        """Get feature analysis result file path"""
        return self.checkpoint_manager.get_path("feature_analysis")
    
    @property
    def refactor_plan_path(self) -> Path:
        """Get refactor plan file path"""
        return self.checkpoint_manager.get_path("refactor_plan")
    
    def _get_selector_agent(self) -> FeatureSelectAgent:
        """Get or create feature selection agent"""
        if self.selector_agent is None:
            # Extract feature selection specific kwargs
            feature_tree_path = self.kwargs.get('feature_tree_path')
            frequency_path = self.kwargs.get('frequency_path')
            mode = self.kwargs.get('mode', 'simple')

            # feature_tree_path and frequency_path are only required in "feature" mode
            if mode == "feature" and (not feature_tree_path or not frequency_path):
                raise ValueError(
                    "feature_tree_path and frequency_path are required when mode='feature'. "
                    "Set mode='simple' to skip these, or provide valid paths."
                )

            # Extract other optional parameters
            selection_kwargs = {
                'vec_db': self.kwargs.get('vec_db'),
                'vec_db_path': self.kwargs.get('vec_db_path'),
                'mode': mode,
                'explore_conditions': self.kwargs.get('explore_conditions', [5, 8, 5, 4, 5]),
                'select_max_iterations': self.kwargs.get('select_max_iterations', 2),
                'select_context_window': self.kwargs.get('select_context_window', 0),
                'temperature': self.kwargs.get('temperature', 10),
                'overlap_pct': self.kwargs.get('overlap_pct', 0)
            }

            self.selector_agent = FeatureSelectAgent(
                feature_tree_path=feature_tree_path,
                frequency_path=frequency_path,
                llm_cfg=self.llm_cfg,
                checkpoint_manager=self.checkpoint_manager,
                logger=self.logger,
                **selection_kwargs
            )
            self.logger.info("✅ FeatureSelectAgent initialized")
        return self.selector_agent
    
    def _get_refactor_agent(self) -> FeatureRefactorAgent:
        """Get or create feature refactoring agent"""
        if self.refactor_agent is None:
            # Extract only the parameters that FeatureRefactorAgent accepts
            refactor_kwargs = {}
            if 'refactor_context_window' in self.kwargs:
                refactor_kwargs['refactor_context_window'] = self.kwargs['refactor_context_window']
            if 'refactor_max_iterations' in self.kwargs:
                refactor_kwargs['refactor_max_iterations'] = self.kwargs['refactor_max_iterations']
                
            self.refactor_agent = FeatureRefactorAgent(
                llm_cfg=self.llm_cfg,
                checkpoint_manager=self.checkpoint_manager,
                logger=self.logger,
                **refactor_kwargs
            )
            self.logger.info("✅ FeatureRefactorAgent initialized")
        return self.refactor_agent
    
    def build_feature_tree(
        self
    ) -> Dict[str, Any]:
        """
        Complete property-level workflow: feature selection -> refactoring.
        
        Args:
            repo_data: Repository metadata 
            feature_gen_iterations: Max iterations for feature generation            
        Returns:
            Complete results with generated and refactored features
        """
        # Load or initialize repo data
        repo_file = Path(self.checkpoint_manager.repo_data_path)
        if repo_file.exists():
            with open(repo_file, 'r') as f:
                repo_data = json.load(f)
        else:
            self.logger.error(f"Repository data file not found: {repo_file}")
            raise FileNotFoundError(f"Repository data file not found: {repo_file}")
            
        repo_name = repo_data.get("repository_name", "unknown")
        self.logger.info(f"Starting property-level workflow for repository: {repo_name}")
        
        # Initialize or load checkpoint state
        ckpt_file = Path(self.checkpoint_manager.task_state_path)
        if ckpt_file.exists():
            with open(ckpt_file, 'r') as f:
                cur_state = json.load(f)
        else:
            # Initialize empty state with all stages set to False
            cur_state = {
                "feature_selection": False,
                "feature_refactoring": False,
                "build_skeleton": False,
                "build_function": False,
                "plan_tasks": False,
                "code_generation": False,
                "current_stage": "proposal_level"
            }
            with open(ckpt_file, 'w') as f:
                json.dump(cur_state, f, indent=4)
            self.logger.info(f"Created new checkpoint state file: {ckpt_file}")
        
        if not cur_state.get("feature_selection", False):
            # Step 1: Feature Selection
            self.logger.info("=== STEP 1: Feature Selection ===")
            selector = self._get_selector_agent()
            try:
                selection_result, repo_data = selector.iterate_feature_tree(
                    repo_data=repo_data
                )
                # Extract generated feature tree
                generated_feature_tree = selection_result.get("Feature_tree", {})
                if not generated_feature_tree:
                    raise ValueError("No feature tree generated during selection phase")
                self.logger.info(f"Feature selection completed. Generated {len(get_all_leaf_paths(generated_feature_tree))} feature paths")
                
                cur_state["feature_selection"] = True
                with open(ckpt_file, 'w') as f:
                    json.dump(cur_state, f, indent=4)
                with open(repo_file, 'w') as f:
                    json.dump(repo_data, f, indent=4)
            except Exception as e:
                self.logger.error(f"Feature selection failed: {e}")
                raise  ValueError(f"Feature selection failed: {str(e)}")
        
        # Step 2: Feature Refactoring        
        if not cur_state.get("feature_refactoring", False):
            self.logger.info("=== STEP 2: Feature Refactoring ===")
            generated_feature_tree = repo_data["Feature_tree"]
            refactor = self._get_refactor_agent()
            
            try:
                refactoring_result, repo_data = refactor.refactor_feature_tree(
                    feature_tree=generated_feature_tree,
                    repo_data=repo_data
                )
                
                # Extract organized components
                organized_components = refactoring_result.get("Component", [])
                refactor_statistics = refactoring_result.get("statistics", {})
                
                if not organized_components:
                    raise ValueError("No feature graph generated during refactoring phase")
                
                self.logger.info(f"Feature refactoring completed. Created {len(organized_components)} components "
                            f"with {refactor_statistics.get('coverage_rate', 0):.1%} coverage")
               
                cur_state["feature_refactoring"] = True
                with open(ckpt_file, 'w') as f:
                    json.dump(cur_state, f, indent=4)
                    
                with open(repo_file, 'w') as f:
                    json.dump(repo_data, f, indent=4)
            except Exception as e:
                self.logger.error(f"Feature refactoring failed: {e}")
                raise ValueError(f"Feature refactoring failed: {str(e)}")
            
            # Step 3: Combine Results
            self.logger.info("=== Combining Results ===")
                    
        return repo_data
    
