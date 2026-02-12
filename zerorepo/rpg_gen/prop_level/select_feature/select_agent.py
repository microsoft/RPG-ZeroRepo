"""
Property-Level Agent for Feature Tree Generation

This module implements a unified agent architecture without tools, using LLMClient's 
call_with_structure_output for all LLM interactions. Supports two modes:
- 'feature' mode: Complex feature generation with multiple specialized agents
- 'simple' mode: Simple direct feature generation with single agent
"""

import json
import os
import logging
import threading
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient, LLMConfig, Memory,
    SystemMessage, UserMessage, AssistantMessage
)
from zerorepo.utils.tree import (
    get_all_leaf_paths,
    convert_leaves_to_list,
    apply_changes,
    prune_tree
)
from zerorepo.utils.sample import (
    sample_feature
)
from zerorepo.config.checkpoint_config import CheckpointManager

from .prompt import (
    PROMPT_TEMPLATE_FEATURE_GEN,
    PROMPT_TEMPLATE_SELF_CHECK,
    PROMPT_TEMPLATE_SELECT_EXPLOITATION,
    PROMPT_TEMPLATE_SELECT_EXPLORATION,
    PROMPT_TEMPLATE_MISSING_FEATURES
)
from .faiss_db import FaissDocDB

# Global settings
TOP_FEATURES = [
    'workflow', 'implementation style', 'functionality', 'resource usage', 
    'computation operation', 'user interaction', 'data processing', 
    'file operation', 'dependency relations', 'algorithm', "data structures"
]

BATCH_SIZE = 100
_API_SEMA = threading.Semaphore(6)

# Pydantic models for structured outputs
class FeaturePathsOutput(BaseModel):
    """Output schema for feature path selection"""
    all_selected_feature_paths: List[str] = Field(
        description="List of selected feature paths",
        default_factory=list
    )

class ValidatedPathsOutput(BaseModel):
    """Output schema for path validation"""
    validated_feature_paths: List[str] = Field(
        description="List of validated feature paths",
        default_factory=list
    )



class MissingFeaturesOutput(BaseModel):
    """Output schema for missing features identification"""
    missing_features: Dict[str, Any] = Field(
        description="Hierarchical structure of missing features",
        default_factory=dict
    )


class AddPathsOutput(BaseModel):
    """Output schema for adding new feature paths (simple mode)"""
    add_new_feature_paths: List[str] = Field(
        description="List of new feature paths to add",
        default_factory=list
    )


class PropLevelEnv:
    """
    Environment for property-level feature tree generation.
    Manages the state of feature trees, sampling, and validation.
    """
    
    def __init__(
        self,
        feature_tree: Union[str, Path],
        frequency: Union[str, Path],
        vec_db: Union[str, Path],
        mode: Literal["feature", "simple"] = "feature",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.feature_tree = feature_tree
        self.frequency = frequency
            
        self.vec_db = vec_db
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # State management
        self.current_tree = {}
        self.current_refactored = {}
        self.selected_tree = {}
        self.selected_refactored = {}
        self.sampled_elements_paths = set()
        self.iteration_logs = []
        self.final_results = []
        
        # Feature mode specific
        if mode == "feature":
            self.fixed_features = []
            self.missing_paths = []
            self.q_feature_paths = []
            self._vecdb_lock = threading.Lock()
    
    def reset(self):
        """Reset environment state"""
        self.current_tree = {}
        self.current_refactored = {}
        self.selected_tree = {}
        self.selected_refactored = {}
        self.sampled_elements_paths = set()
        self.iteration_logs = []
        self.final_results = []
        
        if self.mode == "feature":
            self.fixed_features = []
            self.missing_paths = []
            self.q_feature_paths = []
    
    def get_feature_tree_state(self) -> Dict[str, Any]:
        """Get current state of feature trees"""
        return {
            "current_tree": self.current_refactored,
            "selected_tree": self.selected_refactored,
            "sampled_paths": list(self.sampled_elements_paths),
            "iteration_count": len(self.iteration_logs)
        }


class ActionHandler:
    """
    Handles action execution logic separated from LLM interactions.
    Each action type has its own processing method.
    """
    
    def __init__(self, env: PropLevelEnv):
        self.env = env
        self.logger = env.logger
    
    def handle_exploitation_selection(self, paths: List[str]) -> Dict[str, Any]:
        """Handle exploitation path selection"""
        self.logger.info(f"Processing exploitation selection: {len(paths)} paths")
        
        # Update selected tree
        self.env.selected_tree = apply_changes(self.env.selected_tree, paths)
        self.env.selected_refactored = convert_leaves_to_list(self.env.selected_tree)
        
        result = {
            "type": "exploitation_selection",
            "paths": paths,
            "updated_tree": self.env.selected_refactored
        }
        
        self.env.final_results.append(result)
        return result
    
    def handle_exploration_selection(self, paths: List[str]) -> Dict[str, Any]:
        """Handle exploration path selection"""
        self.logger.info(f"Processing exploration selection: {len(paths)} paths")
        
        # Update selected tree
        self.env.selected_tree = apply_changes(self.env.selected_tree, paths)
        self.env.selected_refactored = convert_leaves_to_list(self.env.selected_tree)
        
        result = {
            "type": "exploration_selection", 
            "paths": paths,
            "updated_tree": self.env.selected_refactored
        }
        
        self.env.final_results.append(result)
        return result
    
    def handle_path_validation(self, validated_paths: List[str]) -> Dict[str, Any]:
        """Handle path validation"""
        self.logger.info(f"Processing path validation: {len(validated_paths)} paths")
        
        # Update current tree with validated paths
        self.env.current_tree = apply_changes(self.env.current_tree, validated_paths)
        self.env.current_refactored = convert_leaves_to_list(self.env.current_tree)
        
        result = {
            "type": "path_validation",
            "validated": validated_paths,
            "updated_tree": self.env.current_refactored
        }
        
        self.env.final_results.append(result)
        return result
    
    
    def handle_missing_features(self, missing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing features identification with vector DB processing"""
        self.logger.info("Processing missing features identification")
        
        # Process missing features with vector DB if available
        q_feature_paths = get_all_leaf_paths(missing_features)
        missing_paths = []
        
        if self.env.vec_db and q_feature_paths:
            q_features = [q_path.split("/")[-1] for q_path in q_feature_paths]
            
            with self.env._vecdb_lock:
                try:
                    retrieval_results = self.env.vec_db.search(q_features)
                    
                    for q_results in retrieval_results:
                        for result in q_results:
                            docs = result.get("doc", [])
                            if docs:
                                sampled = random.choices(docs, k=min(len(docs), 3))
                                missing_paths.extend(sampled)
                                
                except Exception as e:
                    self.logger.warning(f"Vector DB search failed: {e}")
        
        # Update environment state
        self.env.missing_paths = missing_paths[:600]
        self.env.q_feature_paths = q_feature_paths
        
        result = {
            "type": "missing_features",
            "missing": missing_features,
            "missing_paths": missing_paths,
            "q_feature_paths": q_feature_paths
        }
        
        self.env.final_results.append(result)
        return result
    
    def handle_add_paths(self, new_paths: List[str]) -> Dict[str, Any]:
        """Handle adding new feature paths (simple mode)"""
        self.logger.info(f"Processing add paths: {len(new_paths)} paths")
        
        # Update current tree
        self.env.current_tree = apply_changes(self.env.current_tree, new_paths)
        self.env.current_refactored = convert_leaves_to_list(self.env.current_tree)
        
        result = {
            "type": "add_paths",
            "paths": new_paths,
            "updated_tree": self.env.current_refactored
        }
        
        self.env.final_results.append(result)
        return result


class PromptBuilder:
    """Unified prompt builder for both modes"""
    
    @staticmethod
    def build_repo_info(repo_data: Dict[str, Any]) -> str:
        """Format repository information for prompts"""
        repo_info = ""
        for key, value in repo_data.items():
            if key not in ["repository_name", "repository_purpose", "scope"]:
                continue
            key = key.replace("_", "")
            words = key.split()
            if words:
                words[0] = words[0].capitalize()
            new_key = " ".join(words)
            repo_info += f"{new_key}: {value}\n"
        return repo_info
    
    @staticmethod
    def build_feature_prompt(repo_data: Dict[str, Any], **kwargs) -> str:
        """Build prompt from template with repo data and additional kwargs"""
        repo_info = PromptBuilder.build_repo_info(repo_data)
        prompt = f"Repo Info: {repo_info}\n"
        
        # Replace additional placeholders
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
                
            prompt +=f"{key}: {value}\n\n" 
        
        return prompt


class FeatureSelectAgent:
    """
    Property-level agent for feature tree generation.
    Uses structured output instead of tools for all LLM interactions.
    """
    
    def __init__(
        self,
        feature_tree_path: Optional[Union[str, Path]] = None,
        frequency_path: Optional[Union[str, Path]] = None,
        vec_db: Optional[Union[str, Path, FaissDocDB]] = None,
        vec_db_path: Optional[Union[str, Path]] = None,
        mode: Literal["feature", "simple"] = "feature",
        llm_cfg: Union[str, Dict, LLMConfig] = None,
        explore_conditions: List[int] = [5, 8, 5, 4, 5],
        select_max_iterations: int = 2,
        select_context_window: int = 0,
        temperature: int = 10,
        overlap_pct: float = 0,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize FeatureSelectAgent with vector database support.

        Args:
            feature_tree_path: Path to feature tree file (required for 'feature' mode, optional for 'simple')
            frequency_path: Path to frequency file (required for 'feature' mode, optional for 'simple')
            vec_db: Vector database instance, path to existing db, or None
            vec_db_path: Path where to save/load vector database
            mode: Agent mode ('feature' or 'simple')
            llm_cfg: LLM configuration
            explore_conditions: Exploration conditions for feature selection
            select_max_iterations: Maximum selection iterations
            select_context_window: Context window for selection
            temperature: Temperature for sampling
            overlap_pct: Overlap percentage threshold
            logger: Logger instance
            **kwargs: Additional keyword arguments
        """
        self.logger = logger or logging.getLogger(__name__)

        # Load feature tree and frequency data if paths are provided
        feature_tree = {}
        frequency = {}
        if feature_tree_path:
            with open(feature_tree_path, 'r') as f:
                feature_tree = json.load(f)
        if frequency_path:
            with open(frequency_path, 'r') as f:
                frequency = json.load(f)

        # Validate that feature mode has the required data
        if mode == "feature" and (not feature_tree or not frequency):
            raise ValueError(
                "feature_tree_path and frequency_path are required when mode='feature'. "
                "Set mode='simple' to skip these, or provide valid paths."
            )

        # Initialize or load vector database
        self.vec_db = self._initialize_vector_database(
            vec_db=vec_db,
            vec_db_path=vec_db_path,
            feature_tree=feature_tree
        ) if mode == "feature" else None

        # Initialize environment
        self.env = PropLevelEnv(
            feature_tree=feature_tree,
            frequency=frequency,
            vec_db=self.vec_db,
            mode=mode,
            logger=self.logger
        )
        
        # Initialize action handler
        self.action_handler = ActionHandler(self.env)
        # Initialize LLM client
        self.llm_client = LLMClient(llm_cfg)
        # Store configuration
        self.mode = mode
        self.feature_tree = feature_tree
        self.frequency = frequency
        self.explore_conditions = explore_conditions
        self.temperature = temperature
        self.overlap_pct = overlap_pct
        self.context_window = select_context_window
        self.max_iterations = select_max_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_manager = checkpoint_manager
        
        # Prompt builder
        self.prompt_builder = PromptBuilder()
        
        # Feature mode specific: multiple agent memories
        if mode == "feature":
            self.exploit_memory = Memory(context_window=self.context_window)
            self.explore_memory = Memory(context_window=self.context_window)
            self.missing_paths_memory = Memory(context_window=self.context_window)
            self.self_check_memory = Memory(context_window=self.context_window)
            self.topic_features = self._load_topic_features()
        else:
            # Simple mode uses single memory
            self.memory = Memory(context_window=self.context_window)
    
    def _load_topic_features(self) -> Dict[str, List[str]]:
        """Load secondary features from feature tree (feature mode only)"""
        topic_features = {}
        for topic in self.feature_tree.keys():
            if topic not in TOP_FEATURES:
                continue
            if isinstance(self.feature_tree[topic], dict):
                topic_features[topic] = list(self.feature_tree[topic].keys())
            else:
                topic_features[topic] = list(self.feature_tree[topic])
        return topic_features
    
    def reset_memories(self):
        """Reset all agent memories"""
        if self.mode == "feature":
            self.exploit_memory.clear_memory()
            self.explore_memory.clear_memory()
            self.missing_paths_memory.clear_memory()
            self.self_check_memory.clear_memory()
        else:
            self.memory.clear_memory()
    
    def _call_llm_structured(
        self, 
        memory: Memory, 
        prompt: str, 
        output_schema: BaseModel, 
        max_retries: int = 3
    ) -> Tuple[Optional[BaseModel], str]:
        """
        Call LLM with structured output and handle retries
        
        Args:
            memory: Memory context for the conversation
            prompt: User prompt to send
            output_schema: Pydantic model for expected output structure
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (parsed structured output or None if all attempts fail, raw response)
        """
        memory.add_message(UserMessage(prompt))
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"LLM call attempt {attempt + 1}")
                parsed_result, response = self.llm_client.call_with_structure_output(
                    memory=memory,
                    response_model=output_schema
                )
                self.logger.info(f"LLM call response: {response}")
                if parsed_result:
                    # Add successful response to memory
                    parsed_result = output_schema(**parsed_result)
                    memory.add_message(AssistantMessage(str(response)))
                    return parsed_result, response
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} LLM call attempts failed")          
        return None, ""
    
    
    def run_simple_mode(
        self,
        repo_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run agent in simple mode"""
        self.env.reset()
        
        repo_name = repo_data.get("repository_name", "unknown")
        self.logger.info(f"Running simple mode for {repo_name}")
        
        # Clean repo data
        repo_data = {k: v for k, v in repo_data.items() 
                    if k not in ["usage_scenario", "keywords"]}
        
        # Initialize memory with system prompt
        system_prompt = PROMPT_TEMPLATE_FEATURE_GEN
        self.memory.add_message(SystemMessage(system_prompt))
        
        for i in range(self.max_iterations):
            self.logger.info(f"--------- Iteration {i+1} ---------")
            
            iter_traj = []
            
            # Generate prompt for new features
            prompt = self.prompt_builder.build_feature_prompt(
                repo_data,
                CurrentFeature=self.env.current_refactored
            )
            
            # Collect multiple batches
            all_paths = []

            result, response = self._call_llm_structured(
                memory=self.memory,
                prompt=prompt,
                output_schema=AddPathsOutput
            )
            
            iter_traj.append({
                "role": "simple_agent",
                "input": prompt,
                "output": response if response else ""
            })
                
            if result and result.add_new_feature_paths:
                all_paths.extend(result.add_new_feature_paths)
            
            if not all_paths:
                continue
                  
            # Process with action handler
            self.action_handler.handle_add_paths(all_paths)
            
            # Log iteration
            self.env.iteration_logs.append({
                "iteration": i + 1,
                "selected_paths": all_paths,
                "current_tree": self.env.current_refactored,
                "iter_traj": iter_traj
            })
        
        return {
            **repo_data,
            "Feature_tree": self.env.current_refactored,
            "Iterations": self.env.iteration_logs
        }
    
    def run_feature_mode(
        self,
        repo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run agent in feature mode with multiple specialized agents"""
        self.env.reset()
        self.reset_memories()
        
        repo_name = repo_data.get("repository_name", "unknown")
        self.logger.info(f"Running feature mode for {repo_name}")
        
        # Clean repo data
        repo_data = {k: v for k, v in repo_data.items() 
                    if k not in ["usage_scenario", "keywords"]}
        
        # Initialize agent memories with system prompts
        self._init_feature_mode_memories()
        
        # Step 1: Initialize with missing features approach
        iter_traj = []
        initial_missing_paths, initial_q_feature_paths = self._get_initial_missing_features(repo_data, iter_traj)
        self.env.iteration_logs.append({
            "iteration": 0,
            "retrieval_paths": initial_missing_paths,
            "iter_traj": iter_traj
        })
        
        self.env.sampled_elements_paths = set(initial_missing_paths)
        missing_tree = apply_changes(
            tree={}, changes=initial_missing_paths
        )
        missing_tree = convert_leaves_to_list(missing_tree)
        

        # Main iteration loop
        for it in range(1, self.max_iterations + 1):
            self.logger.info(f"--------- Iteration {it} ---------")
            
            iter_traj = []
            # Build exploit_tree from vector retrieval instead of sampling
            exploit_tree = missing_tree
            # Use entire feature tree as basis (no fixed/random split)
            # Sample exploration tree from remaining features
            explore_tree = sample_feature(
                feature_tree=self.feature_tree, 
                frequencies=self.frequency, 
                conditions=self.explore_conditions,
                temperature=self.temperature, 
                rejected_paths=self.env.sampled_elements_paths, 
                top_features=TOP_FEATURES,
                fixed_features=[], 
                overlap_pct=self.overlap_pct,
                max_sample_times=100
            )
            self.env.sampled_elements_paths |= set(get_all_leaf_paths(explore_tree))
            
            # Select paths using specialized agents
            exploit_paths = self._select_with_memory(
                self.exploit_memory, repo_data, self.env.current_refactored, 
                exploit_tree, "Exploit", iter_traj
            )
            
            explore_paths = self._select_with_memory(
                self.explore_memory, repo_data, self.env.current_refactored,
                explore_tree, "Explore", iter_traj
            )
            
            # Combine all paths
            all_paths = list(set(self.env.q_feature_paths + exploit_paths + explore_paths))
            
            # Validate paths in batches
            validated_paths = self._validate_paths_batch(
                repo_data, all_paths, self.env.current_tree, iter_traj
            )
            
            # Process with action handler
            self.action_handler.handle_path_validation(validated_paths)
            self.action_handler.handle_exploitation_selection(exploit_paths)
            self.action_handler.handle_exploration_selection(explore_paths)
            
            # Find missing features for next iteration
            retrieve_paths, q_paths = self._find_missing_with_memory(repo_data, iter_traj)
            missing_tree = apply_changes({}, retrieve_paths)
            missing_tree = convert_leaves_to_list(missing_tree)

            # Log iteration
            self.env.iteration_logs.append({
                "iteration": it,
                "selected_paths": all_paths,
                "validated_paths": validated_paths,
                "exploit_subtree": exploit_tree,  # Built from vector retrieval
                "explore_subtree": explore_tree,  # Built from sampling
                "current_tree": self.env.current_refactored,
                "iter_traj": iter_traj
            })
        
        return {
            **repo_data,
            "Feature_tree": self.env.current_refactored,
            "Iterations": self.env.iteration_logs,
            "Exploition_Agent": self.exploit_memory.to_dict(),
            "Explore_Agent": self.explore_memory.to_dict(),
            "Self_Check_Agent": self.self_check_memory.to_dict(),
            "Missing_Feature_Agent": self.missing_paths_memory.to_dict()
        }
    
    def _init_feature_mode_memories(self):
        """Initialize all specialized agent memories with system prompts"""
        # Exploitation agent
        self.exploit_memory.add_message(SystemMessage(PROMPT_TEMPLATE_SELECT_EXPLOITATION))
        # Exploration agent
        self.explore_memory.add_message(SystemMessage(PROMPT_TEMPLATE_SELECT_EXPLORATION))
        # Missing paths agent
        self.missing_paths_memory.add_message(SystemMessage(PROMPT_TEMPLATE_MISSING_FEATURES))
        # Self check agent
        self.self_check_memory.add_message(SystemMessage(PROMPT_TEMPLATE_SELF_CHECK))

    def _select_with_memory(
        self,
        memory: Memory,
        repo_data: Dict[str, Any],
        current_tree: Dict[str, Any],
        sample_tree: Dict[str, Any],
        select_type: str,
        iter_traj: List[Dict]
    ) -> List[str]:
        """Select paths using a specific agent memory with structured output"""
        
        self.logger.info(f"============ Selecting {select_type} Feature Paths ===============")
        
        prompt = self.prompt_builder.build_feature_prompt(
            repo_data,
            CurrentTree=current_tree,
            SampleTree=sample_tree
        )
        
        result, response = self._call_llm_structured(
            memory=memory,
            prompt=prompt,
            output_schema=FeaturePathsOutput
        )
        
        iter_traj.append({
            "role": f"{select_type.lower()}_agent",
            "input": prompt,
            "output": response if response else ""
        })
        
        if result and result.all_selected_feature_paths:
            return result.all_selected_feature_paths
        
        return []
    
    def _validate_paths_batch(
        self,
        repo_data: Dict[str, Any],
        all_paths: List[str],
        current_tree: Dict[str, Any],
        iter_traj: List[Dict]
    ) -> List[str]:
        """Validate paths in batches using structured output"""
        validated_paths = []
        
        self.logger.info("============ Validating Feature Paths ===============")
        
        for j in range(0, len(all_paths), BATCH_SIZE):
            batch = all_paths[j:j + BATCH_SIZE]
            
            prompt = self.prompt_builder.build_feature_prompt(
                repo_data,
                CandidatePaths=batch,
                CurrentTree=current_tree
            )
            
            result, response = self._call_llm_structured(
                memory=self.self_check_memory,
                prompt=prompt,
                output_schema=ValidatedPathsOutput
            )
            
            iter_traj.append({
                "role": "self_check_agent",
                "input": prompt,
                "output": response if response else ""
            })
            
            if result and result.validated_feature_paths:
                validated_paths.extend(result.validated_feature_paths)
        
        return list(set(validated_paths))
    
    def _find_missing_with_memory(
        self,
        repo_data: Dict[str, Any],
        iter_traj: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """Find missing features using dedicated memory with structured output"""
        
        self.logger.info("============ Finding Missing Feature Paths ===============")
        env_prompt = self.prompt_builder.build_feature_prompt(
            repo_data,
            CurrentTree=self.env.current_refactored
        )
    
        result, response = self._call_llm_structured(
            memory=self.missing_paths_memory,
            prompt=env_prompt,
            output_schema=MissingFeaturesOutput
        )
        
        if result and result.missing_features:
            iter_traj.append({
                "role": "missing_agent",
                "input": env_prompt,
                "output": response if response else ""
            })
            
            # Process with action handler
            action_result = self.action_handler.handle_missing_features(result.missing_features)
            return action_result["missing_paths"], action_result["q_feature_paths"]
        
        return [], []
    
    def _get_initial_missing_features(self, repo_data: Dict[str, Any], iter_traj) -> List[str]:
        """Get initial missing features to guide the first iteration"""
        env_prompt = self.prompt_builder.build_feature_prompt(
            repo_data,
            CurrentTree={}  # Empty tree for initial assessment
        )

        result, response = self._call_llm_structured(
            memory=self.missing_paths_memory,
            prompt=env_prompt,
            output_schema=MissingFeaturesOutput
        )
        
        
        if result and result.missing_features:
            iter_traj.append({
                "role": "missing_agent",
                "input": env_prompt,
                "output": response if response else ""
            })
            # Process with action handler
            action_result = self.action_handler.handle_missing_features(result.missing_features)
            self.env.missing_paths = action_result["missing_paths"]
            self.env.q_feature_paths = action_result["q_feature_paths"]
            return action_result["missing_paths"], action_result["q_feature_paths"]
        
        return [], []
    
    def iterate_feature_tree(
        self,
        repo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for feature tree iteration.
        
        Args:
            repo_data: Repository information
            result_dir: Directory to save results
        """
        mode = self.mode
        
        if mode == "simple":
            result = self.run_simple_mode(repo_data)
        else:
            result = self.run_feature_mode(repo_data)
        
        result_file = self.checkpoint_manager.get_path("feature_selection")
            
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)
            
        repo_data["Feature_tree"] = result["Feature_tree"]
        
        self.logger.info(f"Results saved to {result_file}")
        
        return result, repo_data
    
    def _initialize_vector_database(
        self, 
        vec_db: Optional[Union[str, Path, FaissDocDB]] = None,
        vec_db_path: Optional[Union[str, Path]] = None,
        feature_tree: Optional[Union[Dict[str, Any], str, Path]] = None
    ) -> Optional[FaissDocDB]:
        """
        Initialize vector database from existing file or create from feature tree.
        
        Args:
            vec_db: Vector database instance, path to existing db, or None
            vec_db_path: Path where to save/load vector database 
            feature_tree: Feature tree data to build database from
            
        Returns:
            FaissDocDB instance or None if no database needed
        """
        # If vec_db is already a FaissDocDB instance, return it
        if isinstance(vec_db, FaissDocDB):
            self.logger.info("Using provided FaissDocDB instance")
            return vec_db
        
        # If vec_db is None, no vector database needed
        if vec_db is None and vec_db_path is None:
            self.logger.info("No vector database specified, running without vector search")
            return None
        
        # Determine database path
        db_path = None
        if isinstance(vec_db, (str, Path)):
            db_path = Path(vec_db)
        elif vec_db_path is not None:
            db_path = Path(vec_db_path)
        else:
            self.logger.warning("No vector database path specified")
            return None
        
        # Try to load existing database
        if db_path.exists():
            # Check if it's a directory with index files or a single file path
            if db_path.is_dir():
                index_path = db_path / "index.faiss"
                doc_path = db_path / "id2doc.json"
            else:
                # Assume db_path is the base path, append file extensions
                index_path = db_path.with_suffix('.faiss')
                doc_path = db_path.with_suffix('.json')
            
            if index_path.exists() and doc_path.exists():
                try:
                    self.logger.info(f"Loading existing vector database from: {db_path}")
                    vector_db = FaissDocDB()
                    vector_db.load(str(index_path), str(doc_path))
                    self.logger.info(f"Successfully loaded vector database with {len(vector_db.id2doc)} documents")
                    return vector_db
                except Exception as e:
                    self.logger.warning(f"Failed to load existing vector database: {e}")
                    self.logger.info("Will create new database from feature tree")
            else:
                self.logger.info(f"Vector database files not found at: {index_path}, {doc_path}")
                self.logger.info("Will create new database from feature tree")
        else:
            self.logger.info(f"Vector database path not found at: {db_path}")
            self.logger.info("Will create new database from feature tree")
        
        # Create new database from feature tree using hash table with multiprocessing
        return self._create_vector_database_from_feature_tree(
            feature_tree=feature_tree, 
            save_path=db_path,
            use_multiprocessing=True,
            num_workers=None,  # Use default (CPU count)
            show_progress=True
        )
    
    def _create_vector_database_from_feature_tree(
        self,
        feature_tree: Union[Dict[str, Any], str, Path],
        save_path: Optional[Path] = None,
        use_multiprocessing: bool = True,
        num_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> Optional[FaissDocDB]:
        """
        Create vector database from feature tree using build_feature_path_hash_table.
        Now with enhanced multiprocessing and progress tracking.
        
        Args:
            feature_tree: Feature tree data or path to feature tree file
            save_path: Path to save the created database
            use_multiprocessing: Whether to use multiprocessing for hash table building
            num_workers: Number of worker processes for hash table building
            show_progress: Whether to show progress bars during creation
            
        Returns:
            FaissDocDB instance or None if creation failed
        """
        try:
            # Load feature tree if it's a path
            tree_data = feature_tree
            if isinstance(feature_tree, (str, Path)):
                self.logger.info("Loading feature tree from file...")
                with open(feature_tree, 'r') as f:
                    tree_data = json.load(f)
            elif feature_tree is None:
                self.logger.error("No feature tree provided for database creation")
                return None
            
            self.logger.info("🚀 Creating vector database from feature tree...")
            
            # Use the existing build_feature_path_hash_table function with multiprocessing
            from .faiss_db import build_feature_path_hash_table
            
            self.logger.info("Building feature path hash table from feature tree...")
            hash_table = build_feature_path_hash_table(
                tree_data, 
                use_multiprocessing=use_multiprocessing,
                num_workers=num_workers,
                show_progress=show_progress
            )
            
            if not hash_table:
                self.logger.warning("No feature paths extracted from tree")
                return None
            
            # Initialize vector database
            vector_db = FaissDocDB(
                use_parallel_encoding=True,
                default_num_workers=num_workers
            )
            
            # Convert hash table to documents for vector database with progress tracking
            keys = []
            docs = []
            
            self.logger.info(f"Converting {len(hash_table)} hash table entries to documents...")
            
            if show_progress:
                # Show progress for converting hash table to documents
                with tqdm(
                    total=len(hash_table),
                    desc="📄 Creating Documents", 
                    unit="docs",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    dynamic_ncols=True,
                    leave=True
                ) as pbar:
                    
                    for feature_name, paths in hash_table.items():
                        # Use feature name as key
                        keys.append(feature_name)
                        # Use list of paths as doc
                        docs.append(paths)  # Store the entire list of paths
                        pbar.update(1)
                        
                        pbar.set_postfix({
                            'Features': len(hash_table),
                            'Total Docs': len(keys)
                        }, refresh=True)
            else:
                # No progress bar, direct conversion
                for feature_name, paths in hash_table.items():
                    # Use feature name as key
                    keys.append(feature_name)
                    # Use list of paths as doc
                    docs.append(paths)  # Store the entire list of paths
            
            if not keys:
                self.logger.warning("No feature documents generated from hash table")
                return None
            
            # Build the database with progress tracking enabled
            self.logger.info(f"🔨 Building vector database with {len(keys)} feature paths...")
            vector_db.build(
                keys=keys, 
                docs=docs,
                batch_size=64,  # Larger batch size for better performance
                num_workers=num_workers,
                show_progress=show_progress
            )
            
            # Save database if path provided
            if save_path:
                try:
                    self.logger.info("💾 Saving vector database to disk...")
                    
                    # Determine save paths based on whether save_path is a directory or file
                    if save_path.suffix in ['.faiss', '.json'] or '.' not in save_path.name:
                        # save_path is a base path, create index and doc paths
                        index_path = save_path.with_suffix('.faiss') if save_path.suffix != '.faiss' else save_path
                        doc_path = save_path.with_suffix('.json')
                        
                        # Ensure parent directory exists
                        index_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        # Treat as directory path
                        save_path.mkdir(parents=True, exist_ok=True)
                        index_path = save_path / "index.faiss"
                        doc_path = save_path / "id2doc.json"
                    
                    if show_progress:
                        # Show a simple progress indicator for saving
                        with tqdm(
                            total=2, 
                            desc="💾 Saving Database", 
                            unit="files",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
                            leave=True
                        ) as save_pbar:
                            
                            # Save in steps to show progress
                            vector_db.save(str(index_path), str(doc_path))
                            save_pbar.update(2)
                    else:
                        vector_db.save(str(index_path), str(doc_path))
                    
                    self.logger.info(f"✅ Vector database saved to: {index_path} and {doc_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save vector database: {e}")
            
            # Log final statistics
            stats = vector_db.get_stats()
            self.logger.info(
                f"✅ Successfully created vector database:\n"
                f"   📊 Total documents: {stats.get('total_documents', len(keys))}\n"
                f"   🧠 Model: {stats.get('model_name', 'unknown')}\n"
                f"   ⚡ GPU enabled: {stats.get('use_gpu', False)}\n"
                f"   🔧 Multiprocessing: {use_multiprocessing}"
            )
            
            return vector_db
            
        except Exception as e:
            self.logger.error(f"Failed to create vector database from feature tree: {e}")
            return None
