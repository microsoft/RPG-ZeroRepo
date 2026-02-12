import json
import logging
import yaml
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from zerorepo.rpg_gen.base import (
    RPG, LLMConfig,
    RepoNode, DirectoryNode,
    FileNode, RepoSkeleton
)
from zerorepo.rpg_gen.base.node import show_project_structure_from_tree
from .util import convert_raw_skeleton_to_repo_node_tree
from .prompts import (
    GROUP_SKELETON,
    GROUP_SKELETON_REVIEW,
    RAW_SKELETON,
    RAW_SKELETON_REVIEW
)
from .agents import (
    RawSkeletonAgent,
    GroupSkeletonAgent
)

class FileDesigner:
    """Main orchestrator for skeleton building using separate agents"""
    
    def __init__(
        self,
        file_map: Optional[Dict[str, str]] = None,
        llm_config: Optional[Union[str, Dict, LLMConfig]] = None,
        rpg: Optional[RPG] = None,
        config_path: Optional[str] = None
    ):
        self.file_map = file_map or {}
        self.skeleton = RepoSkeleton(self.file_map)
        self.rpg = rpg
        
        self.llm_config = llm_config
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize agents when repo_data is available
        if self.rpg:
            self._init_agents()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"[FileDesigner] Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logging.warning(f"[FileDesigner] Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                'raw_skeleton': {
                    'max_steps': 20,
                    'max_review_times': 0,
                    'design_context_window': 20,
                    'review_context_window': 3
                },
                'group_skeleton': {
                    'max_steps': 20,
                    'max_retry': 3,
                    'max_review_times': 0,
                    'design_context_window': 20,
                    'review_context_window': 3
                },
                'general': {
                    'log_level': 'INFO'
                }
            }
    
    def _init_agents(self):
        """Initialize the specialized agents"""
        if not self.rpg:
            logging.warning("RPG not available, skipping agent initialization")
            return
            
        # Raw skeleton agent
        raw_system_prompt = RAW_SKELETON
        raw_review_prompt = RAW_SKELETON_REVIEW
        
        # Get configuration
        raw_config = self.config.get('raw_skeleton', {})
        
        self.raw_agent = RawSkeletonAgent(
            llm_cfg=self.llm_config,
            design_system_prompt=raw_system_prompt,
            design_context_window=raw_config.get('design_context_window', 30),
            review_system_prompt=raw_review_prompt,
            review_context_window=raw_config.get('review_context_window', 10),
            max_review_times=raw_config.get('max_review_times', 1),
            repo_rpg=self.rpg
        )
        
    def generate_raw_skeleton(
        self
    ) -> Tuple[Dict, RPG, Dict]:
        """Generate raw skeleton using RawSkeletonAgent
        
        Returns:
            Tuple[Dict, RPG, Dict]: (raw_skeleton_dict, updated_rpg, raw_agent_results)
        """
        logging.info("[FileDesigner] === Generating raw skeleton ===")
        
        # Use config value if not provided
        max_steps = self.config.get('raw_skeleton', {}).get('max_steps', 20)
        max_retry = self.config.get('raw_skeleton', {}).get('max_retry', 3)
        
        result = self.raw_agent.generate_raw_skeleton(
            max_steps=max_steps,
            max_retry=max_retry
        )
        
        if not result.get("raw_skeleton"):
            logging.error("[FileDesigner] Raw skeleton generation failed.")
            return {}, self.rpg, {"error": "Raw skeleton generation failed", "result": result}
        
        # Extract components
        raw_skeleton_dict = result["raw_skeleton"]
        updated_rpg = result.get("repo_rpg", self.rpg)
        
        
        # Raw agent results (everything except the main outputs)
        result.pop("raw_skeleton")
        result.pop("repo_rpg")
        
        return raw_skeleton_dict, updated_rpg, result
    
    def group_skeleton(
        self,
        raw_skeleton_dict: Dict, 
        name_to_path_map: Dict[str, List[str]], 
    ) -> Tuple[DirectoryNode, RPG, Dict]:
        """Group features into files using GroupSkeletonAgent
        
        Returns:
            Tuple[DirectoryNode, RPG, Dict]: (tree_root, updated_rpg, group_agent_results)
        """
        logging.info("[FileDesigner] === Starting feature grouping ===")
        
        # Use config values if not provided
        group_config = self.config.get('group_skeleton', {})
       
        max_steps = group_config.get('max_steps', 20)
        max_retry = group_config.get('max_retry', 3)
        
        # Convert raw skeleton to tree
        tree_root = convert_raw_skeleton_to_repo_node_tree(
            raw_skeleton_dict,
            self.file_map,
            root_name=self.rpg.repo_name if self.rpg else "repo"
        )
        
        
        trajectories = {}
        subtree_results = {}
        
        # Process each subtree
        for subtree_name, top_paths in name_to_path_map.items():
            
            logging.info(f"[Subtree] Processing '{subtree_name}'")
            
            group_system_prompt = GROUP_SKELETON
            group_review_prompt = GROUP_SKELETON_REVIEW
            
            group_config = self.config.get("group_skeleton", {})
            # Group skeleton agent (initialized lazily when needed)
            
            group_agent = GroupSkeletonAgent(
                llm_cfg=self.llm_config,
                design_system_prompt=group_system_prompt,
                design_context_window=group_config.get('design_context_window', 30),
                review_system_prompt=group_review_prompt,
                review_context_window=group_config.get('review_context_window', 10),
                max_review_times=group_config.get('max_review_times', 1),
                tree_root=tree_root,
                subtree_name=subtree_name,
                file_map=self.file_map,
                repo_rpg=self.rpg
            )
        
            # Use GroupSkeletonAgent to process this subtree
            result = group_agent.group_subtree_features(
                max_steps=max_steps,
                max_retry=max_retry
            )
            
            trajectories[f"build_skeleton_{subtree_name}"] = result.get("all_traj", [])
            
            # Update tree_root and rpg with result
            tree_root = result["tree_root"]
            self.rpg = result["repo_rpg"]
            
            # Get utilization from the agent's environment
            util = group_agent._env.util if hasattr(group_agent, '_env') else 0.0
            logging.info(f"[Subtree] {subtree_name} completed with utilization: {util:.2%}")
            
            result.pop("tree_root", None)
            result.pop("repo_rpg", None)
            
            # Store subtree-specific results
            subtree_results[subtree_name] = {
                "utilization": util,
                "completed": group_agent._env.is_completed if hasattr(group_agent, '_env') else False,
                "agent_result": result
            }
        
        # Group agent results
        group_agent_results = {
            "trajectories": trajectories,
            "subtree_results": subtree_results,
            "total_subtrees": len(name_to_path_map),
            "success": True
        }
        
        return tree_root, self.rpg, group_agent_results
    
    
    def run(
        self,
        skeleton_result_path: str,
        agent_result_path: str, 
    ) -> Tuple[RepoSkeleton, RPG, Dict[str, Any]]:
        """Full pipeline to build RepoSkeleton from LLM-guided structure design.
        
        Returns:
            Tuple[RepoSkeleton, RPG, Dict]: (final_skeleton, final_rpg, all_results)
        """
        
        logging.info("[FileDesigner] === Start building repository skeleton ===")
        
        # Phase 1: Generate raw high-level layout
        raw_dict, updated_rpg, raw_results = self.generate_raw_skeleton()
        
        if not raw_dict:
            logging.error("[FileDesigner] Raw skeleton generation failed. Exiting.")
            error_results = {
                "error": "Raw skeleton generation failed", 
                "raw_results": raw_results,
                "phase": "raw_skeleton"
            }
            return RepoSkeleton({}), self.rpg, error_results
        
        # Update RPG from raw skeleton result
        self.rpg = updated_rpg
        name_to_path_map = raw_results.get("name_to_path_map", {})
        
        logging.info(f"[FileDesigner] Raw layout parsed. Starting feature grouping with {len(name_to_path_map)} subtrees...")
        logging.info(f"[FileDesigner] === Start Grouping repository skeleton ===")
        
        # Phase 2: Group feature paths into file tree
        grouped_root, final_rpg, group_results = self.group_skeleton(
            raw_skeleton_dict=raw_dict,
            name_to_path_map=name_to_path_map
        )
        
        # Update RPG from grouping result
        self.rpg = final_rpg
        
        logging.info("[FileDesigner] Grouping phase completed. Extracting final file map...")
        
        # Phase 3: Build final RepoSkeleton from tree
        file_map: Dict[str, str] = {}
        def _collect_code(n: RepoNode):
            if n.is_file and isinstance(n, FileNode):
                file_map[n.path] = n.code
            elif n.is_dir:
                for c in n.children():
                    _collect_code(c)
        _collect_code(grouped_root)
        
        skeleton = RepoSkeleton({})
        skeleton.root = grouped_root
        
        skeleton.path_to_node = {}
        def _collect_nodes(n: RepoNode):
            skeleton.path_to_node[n.path] = n
            if n.is_dir:
                for c in n.children():
                    _collect_nodes(c)
        _collect_nodes(skeleton.root)
        
        # RPG updates are now handled directly inside the agents
        
        logging.info(f"[FileDesigner] Final RepoSkeleton ready. Root: {skeleton.root.path}, Files: {len(file_map)}, Nodes: {len(skeleton.path_to_node)}")
        
        # Compile all results
        all_results = {
            "raw_skeleton_phase": raw_results,
            "grouping_phase": group_results,
            "final_file_count": len(file_map),
            "final_node_count": len(skeleton.path_to_node),
            "config_used": self.config,
            "success": True
        }
        
        logging.info("----------------------------Final Mapping: Show RepoSkeleton without Feature--------------------------------")
        logging.info(show_project_structure_from_tree(node=skeleton.root, show_features=False))
        logging.info("----------------------------Final Mapping: Show full path----------------------------")
        logging.info(show_project_structure_from_tree(node=skeleton.root, show_leaves_only=False))
        logging.info("----------------------------Final Mapping: Show Only Leaves----------------------------")
        logging.info(show_project_structure_from_tree(node=skeleton.root, show_leaves_only=True))
        
        skeleton.add_missing_init_files()
        skeleton.save_json(skeleton_result_path)
        
        with open(agent_result_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        return skeleton, final_rpg, all_results
        