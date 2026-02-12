from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from zerorepo.rpg_encoder.rpg_parsing.rpg_encoding import RPGParser
from zerorepo.rpg_gen.base.rpg.dep_graph import DependencyGraph
from zerorepo.utils.compress import get_skeleton
from zerorepo.rpg_gen.base.unit import CodeUnit, ParsedFile
from zerorepo.rpg_gen.base.node import RepoSkeleton, FileNode
from zerorepo.rpg_gen.base.rpg import RPG, Node, NodeMetaData, NodeType
from zerorepo.rpg_gen.impl_level.plan_tasks.task_batch import TaskBatch
from zerorepo.rpg_gen.impl_level.impl_builder import ImplBuilder
from zerorepo.rpg_gen.base import LLMConfig, LLMClient, Memory, UserMessage, AssistantMessage
from zerorepo.config.checkpoint_config import CheckpointManager, CheckpointFiles, create_default_manager, set_checkpoint_manager
from zerorepo.utils.logs import setup_logger
from zerorepo.utils.tree import get_all_leaf_paths, apply_changes, convert_leaves_to_list
from zerorepo.rpg_gen.impl_level.func_design.util import (
    validate_file_implementation_graph,
    validate_file_implementation_list,
    topo_sort_file_graph
)
from zerorepo.rpg_gen.impl_level.func_design.prompts.plan import PLAN_FILE, PLAN_FILE_LIST


class RebuildMode(Enum):
    """Rebuild modes for different levels of information preservation"""
    FEATURE_ONLY = "feature_only"  # Only preserve feature graph, redesign files and functions
    FEATURE_FILE = "feature_file"  # Preserve feature and file info, redesign functions
    FULL_PRESERVE = "full_preserve"  # Preserve all info, only plan data flow and file order


@dataclass 
class RebuildConfig:
    """Configuration for rebuild process"""
    mode: RebuildMode = RebuildMode.FEATURE_ONLY
    llm_config: Optional[LLMConfig] = None
    skeleton_cfg_path: str = ""
    graph_cfg_path: str = ""
    
    # RPG parsing parameters
    max_repo_info_iters: int = 3
    max_exclude_votes: int = 3
    max_parse_iters: int = 10
    min_batch_tokens: int = 10_000
    max_batch_tokens: int = 50_000
    summary_min_batch_tokens: int = 10_000
    summary_max_batch_tokens: int = 50_000
    class_context_window: int = 10
    func_context_window: int = 10
    max_parse_workers: int = 8
    refactor_context_window: int = 10
    refactor_max_iters: int = 10
    
    # Data flow analysis parameters
    run_data_flow_analysis: bool = True
    data_flow_max_results: int = 3
    
    # Parse control
    skip_parse: bool = False


class Rebuild:
    """
    Rebuild RPG from repository with different preservation modes.
    
    Provides checkpoint management, state persistence, and configuration-driven execution.
    Similar to ZeroRepo orchestrator but focused on rebuilding existing repositories.
    """
    
    def __init__(
        self,
        repo_dir: Union[str, Path],
        repo_name: str,
        checkpoint_dir: Union[str, Path],
        config: Optional[RebuildConfig] = None,
        checkpoint_config: Optional[Union[CheckpointFiles, Dict]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Rebuild orchestrator with checkpoint management.
        
        Args:
            repo_dir: Source repository directory to analyze
            repo_name: Name of the repository
            checkpoint_dir: Directory for saving checkpoints and state
            config: Rebuild configuration
            checkpoint_config: Optional custom checkpoint configuration
            logger: Optional logger instance
        """
        self.repo_dir = Path(repo_dir)
        self.repo_name = repo_name
        self.config = config or RebuildConfig()
        
        # Setup logging - use provided logger if available
        if logger:
            self.logger = logger
        else:
            # Fallback to default logger setup
            base_logger = logging.getLogger(f"Rebuild[{repo_name}]")
            self.logger = setup_logger(
                logger=base_logger,
                file_path=None,  # Could be checkpoint_dir / "rebuild.log"
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
        
        # Save checkpoint configuration
        self.checkpoint_manager.save_config()
        
        # Initialize components (lazy initialization)
        self.rpg_parser = None
        self.impl_builder = None
        self.repo_rpg = None
        self.repo_skeleton = None
        
        # Initialize rebuild state
        self.rebuild_state = self._load_rebuild_state()
        
        self.logger.info(f"Rebuild initialized for repository: {repo_name}")
        self.logger.info(f"Source repository: {self.repo_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_manager.checkpoint_dir}")
        self.logger.info(f"Rebuild mode: {self.config.mode.value}")
        self.logger.info(f"Checkpoint files: {list(self.checkpoint_manager.get_all_paths().keys())}")
    
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
    def rebuild_state_file(self) -> Path:
        """Get rebuild state file path"""
        return self.checkpoint_manager.get_path("task_manager_state")  # Reuse task_state for rebuild state
    
    def _load_rebuild_state(self) -> Dict[str, Any]:
        """Load rebuild state from checkpoint file"""
        if not self.rebuild_state_file.exists():
            self.logger.info("No existing rebuild state found. Starting fresh.")
            return {}
        try:
            with open(self.rebuild_state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load rebuild state: {e}. Starting fresh.")
            return {}
    
    def _save_rebuild_state(self) -> None:
        """Save current rebuild state to checkpoint file"""
        try:
            with open(self.rebuild_state_file, 'w') as f:
                json.dump(self.rebuild_state, f, indent=2)
            self.logger.debug(f"Rebuild state saved: {self.rebuild_state}")
        except Exception as e:
            self.logger.error(f"Failed to save rebuild state: {e}")
    
    def _initialize_rpg_parser(self) -> None:
        """Initialize RPGParser with configuration"""
        if self.rpg_parser is None:
            self.rpg_parser = RPGParser(
                repo_dir=str(self.repo_dir),
                repo_name=self.repo_name,
                logger=self.logger,
                llm_config=self.config.llm_config
            )
            self.logger.info("RPGParser initialized")
    
    def _initialize_impl_builder(self) -> None:
        """Initialize ImplBuilder with configuration"""
        if self.impl_builder is None:
            self.impl_builder = ImplBuilder(
                llm_cfg=self.config.llm_config or LLMConfig(),
                skeleton_cfg=self.config.skeleton_cfg_path,
                graph_cfg=self.config.graph_cfg_path,
                repo_path=self.checkpoint_dir,  # Use checkpoint dir as working directory
                ckpt_dir=self.checkpoint_dir,  # Keep for backward compatibility
                logger=self.logger,
                checkpoint_manager=self.checkpoint_manager
            )
            self.logger.info("ImplBuilder initialized")
    
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
    
    def extract_rpg_from_repo(self) -> Dict[str, Any]:
        """
        Extract RPG from repository and save to checkpoint.
        
        Returns:
            Repository data dictionary with RPG information
        """
        # Check if already completed
        if self.rebuild_state.get("rpg_extraction", False):
            self.logger.info("RPG extraction already completed, loading from checkpoint...")
            return self.load_repo_data() or {}
        
        self.logger.info("=== Starting RPG extraction ===")
        self.logger.info(f"Extracting RPG from {self.repo_dir}...")
        
        # Initialize RPGParser
        self._initialize_rpg_parser()

        try:
            base_rpg, feature_tree, skeleton = self.rpg_parser.parse_rpg_from_repo(
                max_repo_info_iters=self.config.max_repo_info_iters,
                max_exclude_votes=self.config.max_exclude_votes,
                max_parse_iters=self.config.max_parse_iters,
                min_batch_tokens=self.config.min_batch_tokens,
                max_batch_tokens=self.config.max_batch_tokens,
                summary_min_batch_tokens=self.config.summary_min_batch_tokens,
                summary_max_batch_tokens=self.config.summary_max_batch_tokens,
                class_context_window=self.config.class_context_window,
                func_context_window=self.config.func_context_window,
                max_parse_workers=self.config.max_parse_workers,
                refactor_context_window=self.config.refactor_context_window,
                refactor_max_iters=self.config.refactor_max_iters,
            )
        except Exception as e:
            self.logger.error(f"RPG extraction failed: {e}")
            raise
        
        self.repo_rpg = base_rpg
        self.repo_skeleton = skeleton

        components = base_rpg.get_functionality_graph()

        # Build feature tree from components
        all_paths = []
        for cmpt in components:
            name = cmpt["name"]
            refactored_tree = cmpt.get("refactored_subtree", {})

            subtree_paths = get_all_leaf_paths(refactored_tree)
            subtree_paths = [
                name + "/" + path for path in subtree_paths
            ]
            all_paths.extend(subtree_paths)

        feature_tree_dict = apply_changes({}, all_paths)
        feature_tree_dict = convert_leaves_to_list(feature_tree_dict)

        repo_info = base_rpg.repo_info

        repo_data = {
            "repository_name": self.repo_name,
            "repository_purpose": repo_info,
            "Feature_tree": feature_tree_dict,
            "Component": components
        }

        # Use skeleton from parser output
        repo_skeleton_dict = skeleton.to_dict() if skeleton else {}

        # Save to checkpoint files
        with open(self.checkpoint_manager.skeleton_path, 'w') as f:
            json.dump(repo_skeleton_dict, f, indent=4)

        # Save RPG in flat format (includes dep_graph if available)
        rpg_dict = base_rpg.to_dict(include_dep_graph=True)
        with open(self.checkpoint_manager.global_rpg_path, 'w') as f:
            json.dump(rpg_dict, f, indent=2, ensure_ascii=False)

        # Save repo data
        self.save_repo_data(repo_data)

        # Update rebuild state
        self.rebuild_state["rpg_extraction"] = True
        self._save_rebuild_state()

        self.logger.info("RPG extraction completed successfully")
        return repo_data
        
    
    def rebuild(self) -> List[TaskBatch]:
        """
        Run the complete rebuild process based on configured mode.
        
        Returns:
            List of TaskBatch objects for task execution
        """
        try:
            self.logger.info("=== Starting complete rebuild process ===")
            self.logger.info(f"Rebuild mode: {self.config.mode.value}")
            
            # Step 1: Extract RPG from repository or load from checkpoint
            if self.config.skip_parse:
                self.logger.info("Skip parse enabled, loading existing repo data...")
                repo_data = self.load_repo_data()
                if not repo_data:
                    raise ValueError("No existing repo data found. Cannot skip parse.")
                self.logger.info("Loaded existing repo data from checkpoint")
            else:
                repo_data = self.extract_rpg_from_repo()
            
            # Step 2: Rebuild based on mode
            if self.config.mode == RebuildMode.FEATURE_ONLY:
                return self._rebuild_feature_only(repo_data)
            elif self.config.mode == RebuildMode.FEATURE_FILE:
                return self._rebuild_feature_file(repo_data)
            elif self.config.mode == RebuildMode.FULL_PRESERVE:
                return self._rebuild_full_preserve(repo_data)
            else:
                raise ValueError(f"Unknown rebuild mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Rebuild process failed: {e}")
            raise
        
        
    def _rebuild_feature_only(self, repo_data: Dict[str, Any]) -> List[TaskBatch]:
        """Mode 1: Only preserve feature graph, redesign files and functions"""
        self.logger.info("=== Rebuilding with FEATURE_ONLY mode ===")
        
        if not os.path.exists(self.rebuild_state_file):
            # Set task state to start from scratch with ImplBuilder
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": False,
                    "build_function": False,
                    "plan_tasks": False,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
        
        # Initialize ImplBuilder
        self._initialize_impl_builder()
        
        try:
            # Run the complete ImplBuilder pipeline
            plan_batches, skeleton, rpg, graph_data = self.impl_builder.run()
            
            self.logger.info(f"Feature-only rebuild completed. Generated {len(plan_batches)} task batches.")
            
            # Set task state to preserve skeleton but rebuild functions
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": True,
                    "build_function": True,
                    "plan_tasks": True,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
                
            return plan_batches
            
        except Exception as e:
            self.logger.error(f"Feature-only rebuild failed: {e}")
            raise
        
    def _rebuild_feature_file(self, repo_data: Dict[str, Any]) -> List[TaskBatch]:
        """Mode 2: Preserve feature and file info, redesign functions"""
        self.logger.info("=== Rebuilding with FEATURE_FILE mode ===")
        
        if not os.path.exists(self.rebuild_state_file):
            # Set task state to preserve skeleton but rebuild functions
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": True,
                    "build_function": False,
                    "plan_tasks": False,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
            
        # Initialize ImplBuilder
        self._initialize_impl_builder()
    
        try:
            # Load existing skeleton and RPG from checkpoints
            with open(self.checkpoint_manager.skeleton_path, 'r') as f:
                skeleton_dict = json.load(f)
            
            with open(self.checkpoint_manager.global_rpg_path, 'r') as f:
                rpg_dict = json.load(f)
            
            skeleton = RepoSkeleton.from_dict(skeleton_dict)
            rpg = RPG.from_dict(rpg_dict)
            
            # Clear all file content to force regeneration
            all_file_nodes = skeleton.get_all_file_nodes()
            for file_node in all_file_nodes:
                file_node.code = ""
            
            rpg.data_flow = []
            # Remove low-level node metadata to force function redesign
            for node_id, node in rpg.nodes.items():
                if node.meta and node.meta.type_name in [NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                    node.meta = None
            
            # Run function design (build_graph)
            self.logger.info("Running function design...")
            skeleton, rpg, graph_data = self.impl_builder.build_graph(
                skeleton=skeleton,
                rpg=rpg
            )
            
            # Run task planning
            self.logger.info("Running task planning...")
            plan_batches = self.impl_builder.plan_all_tasks(rpg, graph_data)
            
            
            # Set task state to preserve skeleton but rebuild functions
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": True,
                    "build_function": True,
                    "plan_tasks": True,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
            
            self.logger.info(f"Feature-file rebuild completed. Generated {len(plan_batches)} task batches.")
            return plan_batches
            
        except Exception as e:
            self.logger.error(f"Feature-file rebuild failed: {e}")
            raise
        
        
    def _rebuild_full_preserve(self, repo_data: Dict[str, Any]) -> List[TaskBatch]:
        """Mode 3: Preserve all info, only plan data flow and file order"""
        self.logger.info("=== Rebuilding with FULL_PRESERVE mode ===")

        if not os.path.exists(self.rebuild_state_file):
            # Set task state to preserve everything except task planning
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": True,
                    "build_function": True,
                    "plan_tasks": False,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
            
        # Initialize ImplBuilder
        self._initialize_impl_builder()

        try:
            # Load existing skeleton and RPG (they should exist from extraction step)
            if not self.checkpoint_manager.skeleton_path.exists() or not self.checkpoint_manager.global_rpg_path.exists():
                raise FileNotFoundError("Full preserve mode requires existing skeleton and RPG files")

            self.logger.info("Loading existing skeleton and RPG...")
            with open(self.checkpoint_manager.skeleton_path, 'r') as f:
                skeleton_dict = json.load(f)
            with open(self.checkpoint_manager.global_rpg_path, 'r') as f:
                rpg_dict = json.load(f)

            skeleton = RepoSkeleton.from_dict(skeleton_dict)
            rpg = RPG.from_dict(rpg_dict)

            # Get dependency graph from RPG (now embedded in RPG)
            dep_graph = rpg.dep_graph
            if dep_graph:
                self.logger.info(f"Loaded dependency graph from RPG: {dep_graph.G.number_of_nodes()} nodes, {dep_graph.G.number_of_edges()} edges")
            else:
                self.logger.info("No dependency graph in RPG. Will use LLM-based file ordering.")

            # 获取 data_flow（如果存在）
            data_flow = getattr(rpg, 'data_flow', None) or []

            # Store original file codes before extracting interfaces
            original_file_codes = {}
            all_file_nodes = skeleton.get_all_file_nodes()
            for file_node in all_file_nodes:
                if file_node.code and file_node.path.endswith(".py"):
                    original_file_codes[file_node.path] = file_node.code
                    # Extract classes and functions from file, replacing implementations with pass
                    file_node.code = self._extract_interface_from_file_code(file_node.code)


            # Note: Methods are no longer processed for planning - only functions and classes are retained

            # Get functionality graph and map relation between subgraph and feature path
            components = rpg.get_functionality_graph()
            cmpt_to_paths = {}
            
            for cmpt in components:
                cmpt_name = cmpt["name"]
                subtree = cmpt["refactored_subtree"]
                
                cmpt_to_paths[cmpt_name] = [
                    cmpt_name + "/" + path for path in get_all_leaf_paths(subtree)
                ]
            
            # Build subtree order from data_flow
            subtree_order = self._build_subtree_order_from_data_flow(data_flow)
            # If no data_flow, get subtree names from RPG
            if not subtree_order:
                subtree_order = self._get_subtrees_from_rpg(rpg)

            self.logger.info(f"Subtree order: {subtree_order}")

            # Plan file order and extract interfaces for each subtree
            all_interfaces = {}
            file_order_map = {}

            for subtree_name in subtree_order:
                self.logger.info(f"Processing subtree: {subtree_name}")

                # Find files belonging to this subtree
                feature_paths = cmpt_to_paths.get(subtree_name, [])
                
                if not feature_paths:
                    self.logger.warning(f"No feature paths found for subtree: {subtree_name}")
                    continue
                
                file_level_feature_paths = set(
                    "/".join(path.split("/")[:-1]) for path in feature_paths
                )
                
                file_nodes = []
                for feature_path in file_level_feature_paths:
                    feature_node: Node = rpg.get_node_by_feature_path(feature_path)
                    if feature_node:
                        file_path = feature_node.meta.path
                        file_node = skeleton.find_file(path=file_path)
                        if file_node:
                            file_nodes.append(file_node)
            
                # Plan file order using dependency graph if available, otherwise LLM
                file_order = self._plan_file_order(file_nodes, rpg=rpg, dep_graph=dep_graph)
                file_order_map[subtree_name] = file_order

                self.logger.info(f"File order for {subtree_name}: {file_order}")

                # Extract interface information for each file
                subtree_interfaces = {}
                for file_path in file_order:
                    self.logger.debug(f"Extracting interface for file: {file_path}")
                    file_interface = self._extract_file_interface_from_rpg(
                        file_path=file_path,
                        rpg=rpg,
                        skeleton=skeleton,
                        original_file_codes=original_file_codes
                    )
                    if file_interface:
                        subtree_interfaces[file_path] = file_interface
                        self.logger.debug(f"Interface extracted for {file_path}: {len(file_interface.get('units', []))} units")
                    else:
                        self.logger.warning(f"No interface extracted for file: {file_path}")

                all_interfaces[subtree_name] = {
                    "files_order": file_order,
                    "interfaces": subtree_interfaces
                }

            # Build graph_data format (consistent with FuncDesigner.run() output)
            graph_data = {
                "data_flow_phase": {
                    "data_flow": data_flow,
                    "subtree_order": subtree_order
                },
                "base_classes_phase": {
                    "base_classes": {}
                },
                "interfaces_phase": {
                    "interfaces": {
                        "subtrees": all_interfaces,
                        "implemented_subtrees": {
                            subtree: file_order_map.get(subtree, [])
                            for subtree in subtree_order
                        },
                        "processing_order": subtree_order
                    }
                },
                "final_node_count": len(rpg.nodes),
                "final_edge_count": len(rpg.edges),
                "success": True
            }

            # Save graph_data to checkpoint
            with open(self.checkpoint_manager.graph_path, 'w') as f:
                json.dump(graph_data, f, indent=2)

            self.logger.info(f"Graph data saved to {self.checkpoint_manager.graph_path}")

            # Run task planning
            self.logger.info("Running task planning...")
            plan_batches = self.impl_builder.plan_all_tasks(rpg, graph_data)

            self.logger.info(f"Full-preserve rebuild completed. Generated {len(plan_batches)} task batches.")
            
            # Set task state to preserve skeleton but rebuild functions
            with open(self.rebuild_state_file, 'w') as f:
                task_state = {   
                    "feature_selection": True,
                    "feature_refactoring": True,
                    "build_skeleton": True,
                    "build_function": True,
                    "plan_tasks": True,
                    "code_generation": False
                }
                json.dump(task_state, f, indent=4)
                
            return plan_batches
            
        except Exception as e:
            self.logger.error(f"Full-preserve rebuild failed: {e}")
            raise
    
    def _extract_interface_from_file_code(self, file_code: str) -> str:
        """
        Extract interfaces from file code, replacing all implementations with pass.
        
        Args:
            file_code: Original file code
            
        Returns:
            Interface-only code with implementations replaced by pass
        """
        import ast
        try:
            # Try to parse the code, handling encoding issues
            if isinstance(file_code, bytes):
                file_code = file_code.decode('utf-8', errors='ignore')
            
            # Clean up common file content issues
            file_code = file_code.strip()
            if not file_code:
                return ""
            
            # Handle files that might have BOM or other encoding artifacts
            file_code = file_code.replace('\ufeff', '')  # Remove BOM
            
            tree = ast.parse(file_code)
            result_lines = []
            lines = file_code.split('\n')
            
            # Keep imports and module-level variables
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_line = lines[node.lineno - 1]
                    result_lines.append(import_line)
                elif isinstance(node, ast.Assign) and node.lineno <= 50:  # Module level variables (approximate)
                    assign_line = lines[node.lineno - 1]
                    result_lines.append(assign_line)
            
            if result_lines:
                result_lines.append('')  # Empty line after imports/variables
            
            # Extract classes and functions with interfaces only
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_interface = self._extract_class_from_code(file_code, node.name)
                    if class_interface:
                        result_lines.append(class_interface)
                        result_lines.append('')  # Empty line after class
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0:  # Top-level functions only
                    func_interface = self._extract_function_from_code(file_code, node.name)
                    if func_interface:
                        result_lines.append(func_interface)
                        result_lines.append('')  # Empty line after function
                        
            return '\n'.join(result_lines).strip()
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in file code, line {e.lineno}: {e.msg}")
            # Fallback: return original code for files with syntax errors
            return file_code
        except Exception as e:
            self.logger.warning(f"Failed to extract interface from file code: {e}")
            # Fallback: return original code
            return file_code

    def _build_subtree_order_from_data_flow(self, data_flow: List[Dict]) -> List[str]:
        """从 data_flow 构建 subtree 的拓扑顺序"""
        from collections import defaultdict, deque

        if not data_flow:
            return []

        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_names = []
        seen = set()

        for edge in data_flow:
            src = edge.get("source")
            tgt = edge.get("target")
            if src and tgt:
                graph[src].append(tgt)
                in_degree[tgt] += 1
                in_degree.setdefault(src, 0)

                for name in (src, tgt):
                    if name not in seen:
                        seen.add(name)
                        all_names.append(name)

        # Kahn's algorithm
        queue = deque([n for n in all_names if in_degree[n] == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 处理可能的环
        if len(topo_order) < len(all_names):
            remaining = [n for n in all_names if n not in topo_order]
            topo_order.extend(remaining)

        return topo_order

    def _get_subtrees_from_rpg(self, rpg: RPG) -> List[str]:
        """从 RPG 获取所有 subtree 名称（level 1 的节点）"""
        subtrees = []
        for node in rpg.nodes.values():
            if node.level == 1 and node.name:
                subtrees.append(node.name)
        return subtrees

    def _find_files_for_subtree_from_skeleton(
        self,
        skeleton: RepoSkeleton,
        subtree_name: str
    ) -> List:
        """从 skeleton 中找到属于指定 subtree 的文件"""
        from zerorepo.rpg_gen.base.node import FileNode, DirectoryNode

        file_nodes = []

        def traverse_node(node):
            if isinstance(node, FileNode):
                feature_paths = getattr(node, 'feature_paths', [])
                file_subtree_names = [fp.split("/")[0] for fp in feature_paths]
                if subtree_name in file_subtree_names:
                    file_nodes.append(node)
            elif isinstance(node, DirectoryNode):
                for child in node.children():
                    traverse_node(child)

        if skeleton and skeleton.root:
            traverse_node(skeleton.root)

        return file_nodes

    def _plan_file_order(
        self,
        file_nodes: List[FileNode],
        rpg: RPG = None,
        max_retry: int = 5,
        dep_graph: DependencyGraph = None
    ) -> List[str]:
        """
        Plan the order of files within a subtree.

        Strategy:
        1. Use dep_graph to identify entry points (isolated files + connected roots)
        2. Use LLM to plan the order of entry points (with file content context)
        3. Expand each entry point: isolated -> itself, connected root -> root + descendants
        4. Final order follows LLM-planned entry point order

        Example: [1, 2->3, 4->6<-7, 8]
        - Entry points: [1, 2, 4, 7, 8]
        - LLM plans: [1, 8, 2, 4, 7]
        - Expand: [1, 8, 2, 3, 4, 7, 6]

        Args:
            file_nodes: List of FileNode objects to order
            rpg: RPG object for context
            max_retry: Maximum number of retries for LLM calls
            dep_graph: DependencyGraph object for dependency-based ordering

        Returns:
            List of file paths in planned order
        """
        file_paths = [node.path for node in file_nodes]
        path_to_node = {node.path: node for node in file_nodes}

        # If only one file, return directly
        if len(file_paths) <= 1:
            self.logger.info(f"[_plan_file_order] Only {len(file_paths)} file(s), returning directly without planning")
            return file_paths

        # If dep_graph is provided, use it to identify entry points
        if dep_graph is not None:
            self.logger.info(f"[_plan_file_order] Using DEPENDENCY GRAPH for file ordering ({len(file_paths)} files)")
            isolated_paths, connected_roots, root_to_descendants = self._topo_sort_files_from_dep_graph(file_paths, dep_graph)

            # All entry points = isolated files + connected root nodes
            all_entry_points = isolated_paths + connected_roots
            self.logger.info(f"[_plan_file_order] Entry points: {len(isolated_paths)} isolated + {len(connected_roots)} connected roots = {len(all_entry_points)} total")

            if not all_entry_points:
                self.logger.warning(f"[_plan_file_order] No entry points found, falling back to full LLM planning")
            else:
                # Use LLM to plan the order of entry points
                entry_point_nodes = [path_to_node[p] for p in all_entry_points if p in path_to_node]

                if entry_point_nodes:
                    self.logger.info(f"[_plan_file_order] Using LLM to plan order of {len(entry_point_nodes)} entry points")
                    planned_entry_order = self._plan_file_order_with_llm(entry_point_nodes, rpg, max_retry)
                    self.logger.info(f"[_plan_file_order] LLM planned entry points order: {planned_entry_order}")

                    # Expand entry points to final order
                    final_order = []
                    connected_roots_set = set(connected_roots)

                    for entry_point in planned_entry_order:
                        if entry_point in connected_roots_set:
                            # Connected root: add root + its descendants
                            final_order.append(entry_point)
                            descendants = root_to_descendants.get(entry_point, [])
                            final_order.extend(descendants)
                            self.logger.debug(f"[_plan_file_order] Expanded root '{entry_point}' -> +{len(descendants)} descendants")
                        else:
                            # Isolated file: add directly
                            final_order.append(entry_point)

                    if final_order:
                        self.logger.info(f"[_plan_file_order] Final order: {len(final_order)} files (expanded from {len(planned_entry_order)} entry points)")
                        return final_order

        # Fallback to LLM-based planning if no dep_graph or empty result
        self.logger.info(f"[_plan_file_order] Using LLM for full file ordering ({len(file_paths)} files)")
        return self._plan_file_order_with_llm(file_nodes, rpg, max_retry)

    def _topo_sort_files_from_dep_graph(
        self,
        file_paths: List[str],
        dep_graph: DependencyGraph
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Analyze dependency graph and prepare for LLM-based ordering of entry points.

        Returns:
        - isolated_files: files with no dependencies (entry points themselves)
        - connected_root_nodes: entry points of connected components
        - root_to_descendants: mapping from each root node to its descendants (topo sorted)

        The caller should:
        1. Combine isolated_files + connected_root_nodes as "entry points"
        2. Use LLM to plan the order of entry points
        3. For each entry point in planned order:
           - If isolated: add directly
           - If connected root: add root + its descendants from root_to_descendants

        Args:
            file_paths: List of file paths to sort
            dep_graph: DependencyGraph with import/invoke/inherit edges

        Returns:
            Tuple of (isolated_files, connected_root_nodes, root_to_descendants)
        """
        from collections import defaultdict, deque

        # Normalize file paths to match dep_graph node format
        def normalize_to_dep_graph_path(path: str) -> str:
            if path.startswith("./"):
                path = path[2:]
            return path

        # Build a set of file paths we care about (normalized)
        target_files = set()
        path_mapping = {}  # normalized -> original
        for fp in file_paths:
            norm_path = normalize_to_dep_graph_path(fp)
            target_files.add(norm_path)
            path_mapping[norm_path] = fp

        # Build adjacency list and in-degree from dep_graph
        # An edge src -> dst in imports means: src depends on dst
        # So for build order, dst should come before src
        graph = defaultdict(set)  # dst -> [src1, src2, ...] (dst needs to be built before src)
        reverse_graph = defaultdict(set)  # src -> [dst1, dst2, ...] (src depends on these)
        in_degree = defaultdict(int)

        # Initialize all target files with 0 in-degree
        for fp in target_files:
            in_degree[fp] = 0

        # Process edges from dep_graph
        for u, v, data in dep_graph.G.edges(data=True):
            edge_type = data.get("type")
            if edge_type not in ["imports", "invokes", "inherits"]:
                continue

            src_file = u.split(":")[0] if ":" in u else u
            dst_file = v.split(":")[0] if ":" in v else v

            if src_file in target_files and dst_file in target_files and src_file != dst_file:
                if src_file not in graph[dst_file]:
                    graph[dst_file].add(src_file)
                    reverse_graph[src_file].add(dst_file)
                    in_degree[src_file] += 1

        self.logger.debug(f"Dependency graph edges: {dict(graph)}")
        self.logger.debug(f"In-degree: {dict(in_degree)}")

        # Identify connected files (have dependencies or are depended upon)
        connected_files = set()
        for dst, srcs in graph.items():
            if srcs:
                connected_files.add(dst)
                connected_files.update(srcs)

        # Isolated files: no dependencies and not depended upon
        isolated_files = [fp for fp in target_files if fp not in connected_files]

        # Find root nodes in connected component (in_degree = 0)
        connected_root_nodes = [fp for fp in connected_files if in_degree[fp] == 0]
        connected_root_nodes = sorted(connected_root_nodes)

        self.logger.info(f"[_topo_sort] Found {len(isolated_files)} isolated files: {isolated_files}")
        self.logger.info(f"[_topo_sort] Found {len(connected_files)} connected files with {len(connected_root_nodes)} root nodes: {connected_root_nodes}")

        # For each root node, find its descendants via BFS/topo sort
        # We need to handle the case where multiple roots share descendants
        root_to_descendants = {}
        visited_globally = set()

        for root in connected_root_nodes:
            if root in visited_globally:
                root_to_descendants[root] = []
                continue

            # BFS from this root to find all reachable nodes
            descendants = []
            local_in_degree = {fp: in_degree[fp] for fp in connected_files}

            # Find nodes reachable from this root
            reachable = set()
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if node in reachable:
                    continue
                reachable.add(node)
                for dependent in graph.get(node, set()):
                    if dependent not in reachable:
                        queue.append(dependent)

            # Topo sort only the reachable nodes from this root
            # Reset in_degree for reachable nodes
            local_in_degree = defaultdict(int)
            for node in reachable:
                for dep in reverse_graph.get(node, set()):
                    if dep in reachable:
                        local_in_degree[node] += 1

            # Start from nodes with 0 local in_degree (should include root)
            queue = deque([n for n in reachable if local_in_degree[n] == 0])

            while queue:
                node = queue.popleft()
                if node not in visited_globally:
                    descendants.append(node)
                    visited_globally.add(node)

                for dependent in sorted(graph.get(node, set())):
                    if dependent in reachable:
                        local_in_degree[dependent] -= 1
                        if local_in_degree[dependent] == 0:
                            queue.append(dependent)

            # Remove the root itself from descendants (it will be added separately)
            if descendants and descendants[0] == root:
                root_to_descendants[root] = descendants[1:]
            else:
                root_to_descendants[root] = [d for d in descendants if d != root]

        # Convert to original paths
        isolated_result = [path_mapping.get(fp, fp) for fp in isolated_files]
        root_nodes_result = [path_mapping.get(fp, fp) for fp in connected_root_nodes]
        root_to_descendants_result = {
            path_mapping.get(root, root): [path_mapping.get(d, d) for d in descs]
            for root, descs in root_to_descendants.items()
        }

        self.logger.info(f"[_topo_sort] Result: {len(isolated_result)} isolated, {len(root_nodes_result)} connected roots")
        for root, descs in root_to_descendants_result.items():
            self.logger.info(f"[_topo_sort]   Root '{root}' -> {len(descs)} descendants: {descs}")

        return isolated_result, root_nodes_result, root_to_descendants_result

    def _plan_root_nodes_order_with_llm(
        self,
        root_nodes: List[str],
        path_mapping: Dict[str, str],
        rpg: RPG,
        max_retry: int = 5
    ) -> List[str]:
        """
        Use LLM to determine the order of root nodes (files with no dependencies).

        Args:
            root_nodes: List of normalized file paths with no dependencies
            path_mapping: Mapping from normalized to original paths
            rpg: RPG for context
            max_retry: Max retries for LLM calls

        Returns:
            Ordered list of root nodes
        """
        from zerorepo.rpg_gen.base import LLMClient, Memory, UserMessage, AssistantMessage, LLMConfig

        if len(root_nodes) <= 1:
            return root_nodes

        # Get original paths for display
        original_paths = [path_mapping.get(fp, fp) for fp in root_nodes]

        # Build prompt for LLM
        trees_info = rpg.visualize_dir_map(max_depth=2, feature_only=True)
        repo_info = rpg.repo_info

        prompt = f"""You are planning the implementation order for a code repository.

## Repository Information
{repo_info}

## Repository Structure
{trees_info}

## Task
The following files have no direct import dependencies on each other.
Please determine the best order to implement them based on:
1. Foundational/utility files should come first
2. Files that define base classes or core abstractions should come before files that use them
3. Configuration and constant files should come early

## Files to Order
{chr(10).join(f"- {p}" for p in original_paths)}

Please return ONLY these file paths in the recommended implementation order.
Output a JSON object with an "implementation_order" field containing the file paths array.
Example: {{"implementation_order": ["path1", "path2", "path3"]}}
"""

        llm_client = LLMClient(self.config.llm_config or LLMConfig())
        memory = Memory(context_window=max_retry)

        class RootNodeOrder(BaseModel):
            implementation_order: List[str] = Field(default=[])

        for i in range(max_retry):
            try:
                memory.add_message(UserMessage(content=prompt))

                planning_result, response = llm_client.call_with_structure_output(
                    memory=memory,
                    response_model=RootNodeOrder,
                    retry_delay=3
                )

                memory.add_message(AssistantMessage(content=response))

                ordered_paths = planning_result.get("implementation_order", [])

                # Validate result
                ordered_set = set(ordered_paths)
                original_set = set(original_paths)

                if ordered_set == original_set:
                    # Convert back to normalized paths
                    reverse_mapping = {v: k for k, v in path_mapping.items()}
                    return [reverse_mapping.get(p, p) for p in ordered_paths]
                else:
                    self.logger.warning(f"[Retry {i + 1}] LLM returned invalid file list. Expected: {original_paths}, Got: {ordered_paths}")
                    prompt = f"Your previous response was invalid. Please only include these exact files: {original_paths}"

            except Exception as e:
                self.logger.warning(f"[Retry {i + 1}] Failed to get LLM response for root node ordering: {e}")

        # Fallback: return sorted alphabetically
        self.logger.info("Using alphabetical order for root nodes as fallback")
        return sorted(root_nodes)

    def _plan_file_order_with_llm(
        self,
        file_nodes: List[FileNode],
        rpg: RPG = None,
        max_retry: int = 5
    ) -> List[str]:
        """
        Plan file order using LLM (original implementation).
        Used as fallback when dep_graph is not available.

        Args:
            file_nodes: List of FileNode objects to order
            rpg: RPG object for context
            max_retry: Maximum number of retries for LLM calls

        Returns:
            List of file paths in planned order
        """
        file_paths = [node.path for node in file_nodes]

        if len(file_paths) <= 1 or not self.config.llm_config:
            return sorted(file_paths)

        file_info_map = {}

        for node in file_nodes:
            code = get_skeleton(
                node.code,
                total_lines=20,
                prefix_lines=10,
                suffix_lines=10,
                keep_constant=False,
                compress_assign=True,
                keep_docstring=False
            )
            if len(code.split("\n")) >= 40:
                code = "\n".join(code.split("\n")[:30])
            file_info_map[node.path] = code

        file_names = list(file_info_map.keys())
        # Get RPG context
        trees_info = rpg.visualize_dir_map(max_depth=2, feature_only=True)
        repo_info = rpg.repo_info

        # Build "files_to_planned" section for the prompt
        lines = []
        lines.append("")
        lines.append("#### File Paths")
        lines.append("You MUST include ONLY the following file paths in the implementation order (no additions, no omissions, no duplicates).")
        for p in file_info_map.keys():
            lines.append(f"- {p}")

        files_to_planned = "\n".join(lines)

        env_prompt = PLAN_FILE_LIST.format(
            repo_info=repo_info,
            trees_info=trees_info,
            files_to_planned=files_to_planned
        )

        # Create LLM client and memory
        llm_client = LLMClient(self.config.llm_config or LLMConfig())
        memory = Memory(context_window=max_retry)

        # Pydantic model for structured output
        class FileImplementationOrder(BaseModel):
            implementation_order: List[str] = Field(default=[])

        final_order = []
        for i in range(max_retry):
            try:
                memory.add_message(UserMessage(content=env_prompt))

                planning_result, response = llm_client.call_with_structure_output(
                    memory=memory,
                    response_model=FileImplementationOrder,
                    retry_delay=3
                )

                memory.add_message(AssistantMessage(content=response))

                ordered_files = planning_result.get("implementation_order", [])

                env_prompt, flag = validate_file_implementation_list(
                    file_list=ordered_files,
                    file_names=file_names
                )

                if not flag:
                    logging.warning(f"[Retry {i + 1}] File list validation failed: {env_prompt}")
                    final_order = []
                    continue
                else:
                    final_order = ordered_files
                    break

            except Exception as e:
                logging.warning(f"[Retry {i + 1}] Failed to parse or validate LLM response: {e}")
                if i == max_retry - 1:
                    logging.error("Exceeded maximum retries. Using fallback ordering.")

        # If we got a valid order, return it
        if final_order:
            return final_order

        # Fallback: return alphabetically sorted file paths
        logging.info("Using fallback file ordering (alphabetical)")
        return sorted(file_paths)

    def _extract_interfaces_from_rpg(
        self,
        rpg: RPG,
        skeleton: RepoSkeleton,
        file_order_map: Dict[str, List[str]],
        subtree_order: List[str]
    ) -> Dict[str, Any]:
        """Extract interface information from RPG and skeleton"""
        
        interfaces_data = {
            "interfaces": {
                "subtrees": {}
            }
        }
        
        # Process each subtree in order
        for subtree_name in subtree_order:
            if subtree_name not in file_order_map:
                continue
                
            subtree_data = {
                "files_order": file_order_map[subtree_name],
                "interfaces": {}
            }
            
            # Process each file in the subtree
            for file_path in file_order_map[subtree_name]:
                file_interface = self._extract_file_interface_from_rpg(
                    file_path=file_path,
                    rpg=rpg,
                    skeleton=skeleton,
                    original_file_codes=None  # This method is not in the _rebuild_full_preserve context
                )
                
                if file_interface:
                    subtree_data["interfaces"][file_path] = file_interface
            
            interfaces_data["interfaces"]["subtrees"][subtree_name] = subtree_data
        
        return interfaces_data
    
    def _extract_file_interface_from_rpg(
        self,
        file_path: str,
        rpg: RPG,
        skeleton: RepoSkeleton,
        original_file_codes: Dict[str, str] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract interface information for a single file from RPG"""
        
        # Find the file node in skeleton
        file_node = skeleton.path_to_node.get(file_path)
        if not file_node or not hasattr(file_node, 'code'):
            self.logger.debug(f"File node not found or no code attribute for {file_path}")
            return None
        
        # Get file code - prefer original code if available
        if original_file_codes and file_path in original_file_codes:
            file_code = original_file_codes[file_path]
        else:
            file_code = getattr(file_node, 'code', '')
        
        if not file_code:
            self.logger.debug(f"File code is empty for {file_path}")
            return None
        
        # Use ParsedFile to extract units (functions and classes) from the file
        try:
            parsed_file = ParsedFile(code=file_code, file_path=file_path)
            parsed_units = parsed_file.units
        except Exception as e:
            self.logger.debug(f"Failed to parse file {file_path} with ParsedFile: {e}")
            return None
        
        units = []
        units_to_code = {}
        units_to_features = {}
        
        # Create a mapping of unit names to RPG nodes for feature path resolution
        unit_name_to_rpg_node = {}
        class_to_method_features = {}  # To aggregate method features to classes
        
        for node in rpg.nodes.values():
            if (hasattr(node, 'meta') and node.meta and 
                hasattr(node.meta, 'path') and 
                node.meta.type_name in [NodeType.FUNCTION, NodeType.CLASS, NodeType.METHOD]):
                node_path = node.meta.path
                # Check if this node belongs to our file
                if (isinstance(node_path, str) and node_path.startswith(file_path + ":")):
                    # Extract the unit name from the path
                    # Format: "file_path:unit_name" or "file_path:class.method"
                    unit_identifier = node_path.split(":", 1)[1]
                    
                    if node.meta.type_name == NodeType.FUNCTION:
                        unit_key = f"function {unit_identifier}"
                        unit_name_to_rpg_node[unit_key] = node
                    elif node.meta.type_name == NodeType.CLASS:
                        unit_key = f"class {unit_identifier}"
                        unit_name_to_rpg_node[unit_key] = node
                    elif node.meta.type_name == NodeType.METHOD:
                        # For methods, extract class name and store features
                        if '.' in unit_identifier:
                            class_name = unit_identifier.split('.')[0]
                            class_key = f"class {class_name}"
                            if class_key not in class_to_method_features:
                                class_to_method_features[class_key] = []
                            # Store the node for later feature extraction
                            class_to_method_features[class_key].append(node)
                    
        self.logger.debug(f"Found {len(unit_name_to_rpg_node)} unit mappings for file {file_path}")
        
        # Process each parsed unit (only functions and classes for planning)
        for unit in parsed_units:
            if unit.unit_type == "function":
                unit_name = f"function {unit.name}"
                unit_code = self._extract_function_from_code(file_code, unit.name)
            elif unit.unit_type == "class":
                unit_name = f"class {unit.name}"
                unit_code = self._extract_class_from_code(file_code, unit.name)
            elif unit.unit_type == "method":
                # Skip methods - only keep functions and classes for planning
                continue
            else:
                continue
            
            units.append(unit_name)
            units_to_code[unit_name] = unit_code or ''
            
            # Map unit to feature paths using RPG nodes
            feature_paths = []
            rpg_node = unit_name_to_rpg_node.get(unit_name)
            if rpg_node:
                # Traverse up the RPG tree to build feature path
                current = rpg_node
                path_parts = []
                while current and current != rpg.repo_node:
                    path_parts.insert(0, current.name)
                    # Find parent via _parents dict
                    parent_id = rpg._parents.get(current.id)
                    current = rpg.nodes.get(parent_id) if parent_id else None

                if path_parts:
                    # Build feature path like "subtree/feature/subfeature"
                    feature_path = "/".join(path_parts)
                    feature_paths.append(feature_path)
                else:
                    self.logger.debug(f"No path parts found for unit {unit_name}")
            else:
                self.logger.debug(f"No RPG node found for unit {unit_name}")

            # For classes, also aggregate features from their methods
            if unit.unit_type == "class" and unit_name in class_to_method_features:
                for method_node in class_to_method_features[unit_name]:
                    # Traverse up from method node to get its feature path
                    current = method_node
                    path_parts = []
                    while current and current != rpg.repo_node:
                        path_parts.insert(0, current.name)
                        parent_id = rpg._parents.get(current.id)
                        current = rpg.nodes.get(parent_id) if parent_id else None
                    
                    if path_parts:
                        feature_path = "/".join(path_parts)
                        if feature_path not in feature_paths:
                            feature_paths.append(feature_path)
                            self.logger.debug(f"Added method feature to class {unit_name}: {feature_path}")
            
            units_to_features[unit_name] = feature_paths
        
        if not units:
            self.logger.debug(f"No units found for file {file_path}")
            return None
        
        self.logger.debug(f"Returning interface for {file_path} with {len(units)} units: {units}")
        # Extract interface-only version of file code
        file_interface_code = self._extract_interface_from_file_code(file_code)
        
        return {
            "file_code": file_interface_code,
            "units": units,
            "units_to_features": units_to_features,
            "units_to_code": units_to_code
        }
    
    def _extract_function_from_code(self, file_code: str, function_name: str) -> str:
        """Extract function definition from file code, replacing implementation with pass"""
        import ast
        try:
            tree = ast.parse(file_code)
            lines = file_code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                    result_lines = []
                    
                    # Extract decorators (they come before the function)
                    if hasattr(node, 'decorator_list') and node.decorator_list:
                        # Find the first decorator line
                        first_decorator_line = node.decorator_list[0].lineno - 1
                        func_def_line = node.lineno - 1
                        
                        # Include all lines from first decorator to function definition
                        for i in range(first_decorator_line, func_def_line):
                            result_lines.append(lines[i])
                    
                    # Extract function definition line(s) until colon
                    func_def_start = node.lineno - 1
                    current_line = func_def_start
                    
                    # Handle multi-line function definitions
                    paren_count = 0
                    in_string = False
                    string_char = None
                    
                    while current_line < len(lines):
                        line = lines[current_line]
                        result_lines.append(line)
                        
                        # Track parentheses and string literals to find the real end
                        i = 0
                        while i < len(line):
                            char = line[i]
                            if not in_string:
                                if char in ('"', "'"):
                                    # Check for triple quotes
                                    if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):
                                        in_string = True
                                        string_char = line[i:i+3]
                                        i += 2
                                    else:
                                        in_string = True
                                        string_char = char
                                elif char == '(':
                                    paren_count += 1
                                elif char == ')':
                                    paren_count -= 1
                            else:
                                if (len(string_char) == 3 and i + 2 < len(line) and 
                                    line[i:i+3] == string_char):
                                    in_string = False
                                    string_char = None
                                    i += 2
                                elif (len(string_char) == 1 and char == string_char and 
                                      (i == 0 or line[i-1] != '\\')):
                                    in_string = False
                                    string_char = None
                            i += 1
                        
                        # Stop when we find colon at the right nesting level
                        if ':' in line and paren_count == 0 and not in_string:
                            break
                        current_line += 1
                    
                    # Check if this is a single-line function (def ... : return ...)
                    last_line = result_lines[-1] if result_lines else ""
                    is_single_line = ':' in last_line and ('return ' in last_line or 'pass' in last_line or '=' in last_line)
                    
                    # Extract docstring if present (only for multi-line functions)
                    if (not is_single_line and len(node.body) > 0 and 
                        isinstance(node.body[0], ast.Expr)):
                        
                        # Handle different Python versions for string constants
                        docstring_value = None
                        if hasattr(ast, 'Str') and isinstance(node.body[0].value, ast.Str):
                            docstring_value = node.body[0].value.s
                        elif isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            docstring_value = node.body[0].value.value
                        
                        if docstring_value:
                            # Find docstring in source - handle both single and multi-line
                            doc_start_line = current_line + 1
                            doc_end_line = None
                            
                            # Look for docstring start
                            for i in range(doc_start_line, len(lines)):
                                line = lines[i].strip()
                                if line.startswith('"""') or line.startswith("'''"):
                                    quote_type = '"""' if line.startswith('"""') else "'''"
                                    
                                    # Check if it's a single-line docstring
                                    if line.count(quote_type) >= 2:
                                        result_lines.append(lines[i])
                                        doc_end_line = i
                                        break
                                    else:
                                        # Multi-line docstring
                                        result_lines.append(lines[i])
                                        # Find closing quotes
                                        for j in range(i + 1, len(lines)):
                                            result_lines.append(lines[j])
                                            if quote_type in lines[j]:
                                                doc_end_line = j
                                                break
                                        break
                                elif not line:  # Empty line
                                    result_lines.append(lines[i])
                                else:
                                    break
                    
                    # Add pass with proper indentation (only for multi-line functions)
                    if not is_single_line:
                        base_indent = 0
                        for line in result_lines:
                            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                                base_indent = len(line) - len(line.lstrip())
                                break
                        
                        result_lines.append(' ' * (base_indent + 4) + 'pass')
                    else:
                        # For single-line functions, replace the implementation with pass
                        last_line = result_lines[-1]
                        if ':' in last_line:
                            colon_pos = last_line.find(':')
                            func_def_part = last_line[:colon_pos + 1]
                            base_indent = len(last_line) - len(last_line.lstrip())
                            result_lines[-1] = func_def_part
                            result_lines.append(' ' * (base_indent + 4) + 'pass')
                    
                    return '\n'.join(result_lines)
        except SyntaxError as e:
            self.logger.debug(f"Syntax error extracting function {function_name}: {e}")
            return ""
        except Exception as e:
            self.logger.debug(f"Error extracting function {function_name}: {e}")
            return ""
        return ""
    
    def _extract_class_from_code(self, file_code: str, class_name: str) -> str:
        """Extract class definition from file code, replacing method implementations with pass"""
        import ast
        try:
            tree = ast.parse(file_code)
            lines = file_code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    result_lines = []
                    
                    # Extract class decorators if present
                    if hasattr(node, 'decorator_list') and node.decorator_list:
                        first_decorator_line = node.decorator_list[0].lineno - 1
                        class_def_line = node.lineno - 1
                        
                        # Include all lines from first decorator to class definition
                        for i in range(first_decorator_line, class_def_line):
                            result_lines.append(lines[i])
                    
                    # Extract class definition line(s) until colon
                    class_def_start = node.lineno - 1
                    current_line = class_def_start
                    while current_line < len(lines):
                        line = lines[current_line]
                        result_lines.append(line)
                        if ':' in line:
                            break
                        current_line += 1
                    
                    base_indent = 0
                    for line in result_lines:
                        if line.strip().startswith('class '):
                            base_indent = len(line) - len(line.lstrip())
                            break
                    
                    # Extract class docstring if present
                    if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                        # Handle different Python versions for string constants
                        docstring_value = None
                        if hasattr(ast, 'Str') and isinstance(node.body[0].value, ast.Str):
                            docstring_value = node.body[0].value.s
                        elif isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            docstring_value = node.body[0].value.value
                            
                        if docstring_value:
                            doc_start_line = current_line + 1
                            # Look for docstring start
                            for i in range(doc_start_line, len(lines)):
                                line = lines[i].strip()
                                if line.startswith('"""') or line.startswith("'''"):
                                    quote_type = '"""' if line.startswith('"""') else "'''"
                                    
                                    # Check if it's a single-line docstring
                                    if line.count(quote_type) >= 2:
                                        result_lines.append(lines[i])
                                        break
                                    else:
                                        # Multi-line docstring
                                        result_lines.append(lines[i])
                                        # Find closing quotes
                                        for j in range(i + 1, len(lines)):
                                            result_lines.append(lines[j])
                                            if quote_type in lines[j]:
                                                break
                                        break
                                elif not line:  # Empty line
                                    result_lines.append(lines[i])
                                else:
                                    break
                    
                    # Process class body nodes in order
                    for body_node in node.body:
                        if isinstance(body_node, (ast.AnnAssign, ast.Assign)):
                            # Class variables and type annotations
                            var_line = body_node.lineno - 1
                            if var_line < len(lines):
                                result_lines.append(lines[var_line])
                                
                        elif isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Add empty line before method if not the first item
                            if result_lines and not result_lines[-1].strip() == '':
                                result_lines.append('')
                            
                            # Extract method decorators if present  
                            if hasattr(body_node, 'decorator_list') and body_node.decorator_list:
                                first_decorator_line = body_node.decorator_list[0].lineno - 1
                                method_def_line = body_node.lineno - 1
                                
                                # Include all lines from first decorator to method definition
                                for i in range(first_decorator_line, method_def_line):
                                    result_lines.append(lines[i])
                            
                            # Extract method definition line(s) until colon
                            method_def_start = body_node.lineno - 1
                            current_method_line = method_def_start
                            
                            # Handle multi-line method definitions
                            method_paren_count = 0
                            method_in_string = False
                            method_string_char = None
                            
                            while current_method_line < len(lines):
                                line = lines[current_method_line]
                                result_lines.append(line)
                                
                                # Track parentheses and string literals to find the real end
                                i = 0
                                while i < len(line):
                                    char = line[i]
                                    if not method_in_string:
                                        if char in ('"', "'"):
                                            # Check for triple quotes
                                            if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):
                                                method_in_string = True
                                                method_string_char = line[i:i+3]
                                                i += 2
                                            else:
                                                method_in_string = True
                                                method_string_char = char
                                        elif char == '(':
                                            method_paren_count += 1
                                        elif char == ')':
                                            method_paren_count -= 1
                                    else:
                                        if (len(method_string_char) == 3 and i + 2 < len(line) and 
                                            line[i:i+3] == method_string_char):
                                            method_in_string = False
                                            method_string_char = None
                                            i += 2
                                        elif (len(method_string_char) == 1 and char == method_string_char and 
                                              (i == 0 or line[i-1] != '\\')):
                                            method_in_string = False
                                            method_string_char = None
                                    i += 1
                                
                                # Stop when we find colon at the right nesting level
                                if ':' in line and method_paren_count == 0 and not method_in_string:
                                    break
                                current_method_line += 1
                            
                            # Extract method docstring if present
                            if len(body_node.body) > 0 and isinstance(body_node.body[0], ast.Expr):
                                # Handle different Python versions for string constants
                                method_docstring_value = None
                                if hasattr(ast, 'Str') and isinstance(body_node.body[0].value, ast.Str):
                                    method_docstring_value = body_node.body[0].value.s
                                elif isinstance(body_node.body[0].value, ast.Constant) and isinstance(body_node.body[0].value.value, str):
                                    method_docstring_value = body_node.body[0].value.value
                                
                                if method_docstring_value:
                                    doc_start_line = current_method_line + 1
                                    # Look for docstring start
                                    for i in range(doc_start_line, len(lines)):
                                        line = lines[i].strip()
                                        if line.startswith('"""') or line.startswith("'''"):
                                            quote_type = '"""' if line.startswith('"""') else "'''"
                                            
                                            # Check if it's a single-line docstring
                                            if line.count(quote_type) >= 2:
                                                result_lines.append(lines[i])
                                                break
                                            else:
                                                # Multi-line docstring
                                                result_lines.append(lines[i])
                                                # Find closing quotes
                                                for j in range(i + 1, len(lines)):
                                                    result_lines.append(lines[j])
                                                    if quote_type in lines[j]:
                                                        break
                                                break
                                        elif not line:  # Empty line
                                            result_lines.append(lines[i])
                                        else:
                                            break
                            
                            # Add pass for method implementation
                            method_indent = 0
                            for line in reversed(result_lines):
                                if line.strip().startswith('def '):
                                    method_indent = len(line) - len(line.lstrip())
                                    break
                            
                            result_lines.append(' ' * (method_indent + 4) + 'pass')
                    
                    # If class has no content, add pass
                    has_content = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)) 
                                    for n in node.body if not (isinstance(n, ast.Expr) and 
                                    isinstance(n.value, (ast.Str, ast.Constant))))
                    if not has_content:
                        result_lines.append(' ' * (base_indent + 4) + 'pass')
                    
                    return '\n'.join(result_lines)
        except SyntaxError as e:
            self.logger.debug(f"Syntax error extracting class {class_name}: {e}")
            return ""
        except Exception as e:
            self.logger.debug(f"Error extracting class {class_name}: {e}")
            return ""
        return ""