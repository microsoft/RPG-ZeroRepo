import json
import logging
import yaml
import os
from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict, deque
from zerorepo.rpg_gen.base import (
    RPG, DirectoryNode, FileNode, RepoSkeleton,
    NodeType,
    LLMConfig, SystemMessage,
    UserMessage, AssistantMessage,
    LLMClient, Memory
)
from .agents import (
    DataFlowAgent, 
    BaseClassAgent,
    InterfaceAgent,
    GenerateBaseClassesTool,
    GenerateDataFlowTool,
    DesignInterfacesTool
)
from .prompts.plan import PLAN_FILE, PLAN_FILE_LIST
from .util import (
    validate_file_implementation_graph,
    validate_file_implementation_list,
    topo_sort_file_graph
)
from zerorepo.utils.logs import setup_logger

class FileImplementationGraph(BaseModel):
    file_implementation_graph: List[Dict] = Field(default=[])

class FileImplementationOrder(BaseModel):
    implementation_order: List[str] = Field(default=[])
        
class FuncDesigner:
    """Main orchestrator for graph building using separate agents with RPG consolidation"""
    
    def __init__(
        self,
        repo_skeleton: Optional[RepoSkeleton] = None,
        llm_config: Optional[Union[str, Dict, LLMConfig]] = None,
        repo_rpg: Optional[RPG] = None,
        config_path: Optional[str] = None,
        logger: logging.Logger = None
    ):
        self.skeleton = repo_skeleton
        self.repo_rpg = repo_rpg
        self.llm_config = llm_config
        # Load configuration
        self.config = self._load_config(config_path)
            
        self.logger = setup_logger() if not logger \
            else logger

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"[FuncDesigner] Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logging.warning(f"[FuncDesigner] Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                'dependency_graph': {
                    'max_steps': 10,
                    'max_review_times': 1,
                    'design_context_window': 30,
                    'review_context_window': 10
                },
                'data_flow': {
                    'max_steps': 10,
                    'max_review_times': 1,
                    'design_context_window': 30,
                    'review_context_window': 10
                },
                'base_classes': {
                    'max_steps': 10,
                    'max_review_times': 1,
                    'design_context_window': 30,
                    'review_context_window': 10
                },
                'interfaces': {
                    'max_steps': 10,
                    'max_review_times': 1,
                    'design_context_window': 30,
                    'review_context_window': 10
                },
                'general': {
                    'log_level': 'INFO'
                }
            }
    
    def _init_agents(self):
        """Initialize the specialized agents"""
        if not self.skeleton:
            logging.warning("Repository skeleton not available, skipping agent initialization")
            return
            
        # Note: DependencyGraphAgent not implemented yet, skipping for now
        
        # Data Flow Agent
        flow_config = self.config.get('data_flow', {})
        self.flow_agent = DataFlowAgent(
            llm_cfg=self.llm_config,
            design_context_window=flow_config.get('design_context_window', 30),
            review_context_window=flow_config.get('review_context_window', 10),
            max_review_times=flow_config.get('max_review_times', 1),
            repo_rpg=self.repo_rpg
        )
        
        # Base Class Agent
        class_config = self.config.get('base_classes', {})
        self.class_agent = BaseClassAgent(
            llm_cfg=self.llm_config,
            design_context_window=class_config.get('design_context_window', 30),
            review_context_window=class_config.get('review_context_window', 10),
            max_review_times=class_config.get('max_review_times', 1),
            repo_rpg=self.repo_rpg
        )
        
        # Interface Agent
        interface_config = self.config.get('interfaces', {})
        self.interface_agent = InterfaceAgent(
            llm_cfg=self.llm_config,
            design_context_window=interface_config.get('design_context_window', 30),
            review_context_window=interface_config.get('review_context_window', 10),
            max_review_times=interface_config.get('max_review_times', 1),
            repo_rpg=self.repo_rpg
        )
    
    
    def build_data_flow(
        self,
        max_steps: Optional[int] = None
    ) -> Tuple[List[Dict], RPG, Dict]:
        """
        Build component data flow
        
        Returns:
            Tuple[List[Dict], RPG, Dict]: (data_flow, updated_rpg, agent_results)
        """
        logging.info("[FuncDesigner] === Building data flow ===")
    

        self.flow_agent = DataFlowAgent(
            llm_cfg=self.llm_config,
            design_context_window=self.config.get('data_flow', {}).get('design_context_window', 30),
            review_context_window=self.config.get('data_flow', {}).get('review_context_window', 10),
            max_review_times=self.config.get('data_flow', {}).get('max_review_times', 0),
            repo_rpg=self.repo_rpg,
            repo_skeleton=self.skeleton,
            register_tools=[GenerateDataFlowTool],
            logger=self.logger
        )
        
        # Use config value if not provided
        max_steps = self.config.get('data_flow', {}).get('max_steps', 5)
        max_retry = self.config.get('data_flow', {}).get('max_retry', 3)
        
        result = self.flow_agent.generate_data_flow(
            max_retry=max_retry,
            max_steps=max_steps
        )
        
        if not result.get("success", False):
            logging.error("[FuncDesigner] Data flow building failed.")
            raise Exception("Data flow building failed.")
        
        # Extract components
        data_flow = result["data_flow"]
        updated_rpg = result["repo_rpg"]
        
        # Update local RPG reference
        self.repo_rpg = updated_rpg
        
        # Agent results
        agent_results = result.get("agent_results", {})
        
        return data_flow, updated_rpg, agent_results
    
    def design_base_classes(
        self
    ) -> Tuple[Dict, RPG, Dict]:
        """
        Design base classes
        
        Returns:
            Tuple[Dict, RPG, Dict]: (base_classes, updated_rpg, agent_results)
        """
        logging.info("[FuncDesigner] === Designing base classes ===")
        
      
        max_steps = self.config.get('base_classes', {}).get('max_steps', 10)
        max_retry = self.config.get('base_classes', {}).get('max_retry', 10)
        
        self.class_agent = BaseClassAgent(
            llm_cfg=self.llm_config,
            design_context_window=self.config.get('data_flow', {}).get('design_context_window', 30),
            review_context_window=self.config.get('data_flow', {}).get('review_context_window', 10),
            max_review_times=self.config.get('data_flow', {}).get('max_review_times', 0),
            repo_rpg=self.repo_rpg,
            repo_skeleton=self.skeleton,
            register_tools=[GenerateBaseClassesTool],
            logger=self.logger
        )
        
        result = self.class_agent.generate_base_classes(
            max_retry=max_retry,
            max_steps=max_steps
        )
        
        if not result.get("success", False):
            raise Exception("[FuncDesigner] Base class design failed.")
        
        # Extract components
        base_classes = result["base_classes"]
        updated_rpg = result["repo_rpg"]
        
        # Update local RPG reference
        self.repo_rpg = updated_rpg
        
        # Agent results
        agent_results = result.get("agent_results", {})
        
        return base_classes, updated_rpg, agent_results
    
    def design_interfaces(
        self,
        data_flow: List[Dict]
    ) -> Tuple[Dict, RPG, Dict]:
        """
        Design component interfaces with full subtree and file orchestration.
        
        This method replicates the traversal logic from the original InterfacesDesigner:
        1. Build subtree dependency order from data flow
        2. For each subtree in topological order:
           - Find files in the subtree
           - Plan file implementation order 
           - Design interface for each file in order
           - Track implemented files for context
        
        Returns:
            Tuple[Dict, RPG, Dict]: (interfaces, updated_rpg, agent_results)
        """
        logging.info("[FuncDesigner] === Designing interfaces with full orchestration ===")
        
        # Use config value if not provided
        max_steps = self.config.get('interfaces', {}).get('max_steps', 10)
        max_retry = self.config.get('interfaces', {}).get('max_retry', 3)

        # 1. Build subtree dependency order using data flow
        subtree_order = self._build_subtree_dependency_order(data_flow)
        logging.info(f"[FuncDesigner] Subtree processing order: {subtree_order}")
        
        # Track results and state across subtrees
        all_interfaces = {}              # 纯接口信息（对外用）
        implemented_subtrees = {}        # 已实现文件列表
        all_agent_results = {}           # 每个文件的 agent_results 全部收集
        
        # 2. Process each subtree in dependency order
        for subtree_name in subtree_order:
            logging.info(f"[FuncDesigner] === Processing subtree: {subtree_name} ===")
            
            try:
                # Find files for this subtree
                file_nodes = self._find_files_for_subtree(subtree_name)
                if not file_nodes:
                    logging.warning(f"No files found for subtree: {subtree_name}")
                    continue
                
                # Plan file order within subtree
                file_order = self._plan_file_order_for_subtree(
                    file_nodes=file_nodes,
                    max_retry=max_retry
                )

                file_node_map = {node.path: node for node in file_nodes}
                
                logging.info(f"[FuncDesigner] File processing order for {subtree_name}: {file_order}")
                
                # Track implemented files for this subtree
                subtree_implemented: List[FileNode] = []
                subtree_interfaces: Dict[str, Dict] = {}  # Will store the merged interface info
                subtree_agent_results: Dict[str, Dict] = {}  # ✅ 每个 file 的 agent_results
                
                # 3. Process each file in the planned order
                for file_path in file_order:
                    file_node = file_node_map.get(file_path)
                    if not file_node:
                        logging.warning(f"File node not found for path: {file_path}")
                        continue
                    
                    logging.info(f"[FuncDesigner] Designing interface for file: {file_path}")
                    
                    self.interface_agent = InterfaceAgent(
                        llm_cfg=self.llm_config,
                        design_context_window=self.config.get('interfaces', {}).get('design_context_window', 30),
                        review_context_window=self.config.get('interfaces', {}).get('review_context_window', 10),
                        max_review_times=self.config.get('interfaces', {}).get('max_review_times', 0),
                        repo_rpg=self.repo_rpg,
                        repo_skeleton=self.skeleton,
                        register_tools=[DesignInterfacesTool],
                        file_node=file_node,
                        tgt_subgraph_name=subtree_name,
                        logger=self.logger
                    )
                    
                    try:
                        result = self.interface_agent.design_file_interface(
                            tgt_subtree_implemented=deepcopy(subtree_implemented),
                            all_implemented_subtrees=deepcopy(implemented_subtrees),
                            max_steps=max_steps,
                            max_retries=max_retry
                        )
                        
                        # 无论成功失败，都把这一轮的 agent_results 收集起来
                        file_agent_results = result.get("agent_results", {})
                        subtree_agent_results[file_path] = {
                            "success": result.get("success", False),
                            "agent_results": file_agent_results
                        }
                        
                        if result.get("success", False):
                            # Create merged interface info for this file
                            file_interface_info = {}
                            
                            # Update file with new interface code
                            if "code" in result:
                                file_node.code = result["code"]
                                # Store the final code for this file
                                file_interface_info["file_code"] = result["code"]
                            
                            # Track this file as implemented
                            subtree_implemented.append(file_node)
                            
                            # Extract units (classes and functions)
                            units = []
                            units_to_code = {}
                            units_to_features = {}
                            
                            # Store interface mapping
                            if "feature_interface_map" in result:
                                units_to_features = result["feature_interface_map"]
                                units = list(units_to_features.keys())
                            
                            # Store unit-specific code if available (only class and function, not method)
                            if "designed_interfaces" in result:
                                # designed_interfaces is a dict with interface keys
                                for interface_key, interface_data in result["designed_interfaces"].items():
                                    if "unit" in interface_data:
                                        unit = interface_data["unit"]
                                        # Only include class and function units, exclude methods
                                        if hasattr(unit, 'unit_type') and unit.unit_type in ["class", "function"]:
                                            # Extract the code from the unit object using count_lines method
                                            try:
                                                _, unit_code = unit.count_lines(original=True, return_code=True)
                                                units_to_code[interface_key] = unit_code
                                            except Exception as e:
                                                logging.warning(f"Failed to extract code for {interface_key}: {e}")
                                                # Fallback to string representation
                                                units_to_code[interface_key] = str(unit)
                                        else:
                                            logging.debug(f"Skipping unit {interface_key} of type {getattr(unit, 'unit_type', 'unknown')} (not class/function)")
                            
                            # Store all info in the merged structure
                            file_interface_info["units"] = units
                            file_interface_info["units_to_features"] = units_to_features
                            file_interface_info["units_to_code"] = units_to_code
                            
                            subtree_interfaces[file_path] = file_interface_info
                            
                            # Update RPG if returned
                            if "repo_rpg" in result:
                                self.repo_rpg = result["repo_rpg"]
        
                            logging.info(f"[FuncDesigner] Successfully designed interface for: {file_path}")
                        else:
                            logging.error(f"[FuncDesigner] Failed to design interface for: {file_path}")
                            
                    except Exception as e:
                        logging.error(f"[FuncDesigner] Exception designing interface for {file_path}: {e}")
                        # 发生异常就标记失败，agent_results 已经在上面收集
                        continue
                    
                # Store results for this subtree
                implemented_subtrees[subtree_name] = subtree_implemented
                all_interfaces[subtree_name] = {
                    "files_order": file_order,
                    "interfaces": subtree_interfaces
                }
                all_agent_results[subtree_name] = {    # ✅ 记录这一棵 subtree 下所有文件的结果
                    "files_order": file_order,
                    "files": subtree_agent_results
                }
                
                logging.info(f"[FuncDesigner] Completed subtree {subtree_name}: {len(subtree_implemented)} files implemented")
                
            except Exception as e:
                logging.error(f"[FuncDesigner] Failed to process subtree {subtree_name}: {e}")
                continue
        
        # Compile final results
        final_interfaces = {
            "subtrees": all_interfaces,
            "implemented_subtrees": {
                subtree: [fn.path for fn in implemented_subtrees[subtree]]
                for subtree in implemented_subtrees
            },
            "processing_order": subtree_order
        }
        
        total_files = sum(
            len(subtree_info.get("files", {}))
            for subtree_info in all_agent_results.values()
        )
        
        agent_results = {
            # 把全部 agent_results 按 subtree → file 组织好返回
            "subtrees": all_agent_results,
            "subtree_count": len(all_agent_results),
            "total_files": total_files,
            "success": True
        }
        
        logging.info(f"[FuncDesigner] Interface design complete: {len(all_interfaces)} subtrees processed")
        
        return final_interfaces, self.repo_rpg, agent_results
    

    def _build_subtree_dependency_order(self, data_flow: List[Dict]) -> List[str]:
        """Build best-effort ordering of subtrees based on data flow graph.
        
        If cycles are present, returns a partial topological order followed by
        the remaining cyclic nodes in arbitrary (but stable) order.
        """
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Build dependency graph from data flow
        for edge in data_flow:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt:
                graph[src].append(tgt)
                in_degree[tgt] += 1
                in_degree.setdefault(src, 0)
        
        # Collect all subtree names
        all_names = []
        seen = set()
        for edge in data_flow:
            for key in ("source", "target"):
                name = edge.get(key)
                if name and name not in seen:
                    seen.add(name)
                    all_names.append(name)
        
        # Initialize in_degree for all nodes
        for name in all_names:
            in_degree.setdefault(name, 0)
        
        # Topological sort (Kahn)
        queue = deque([n for n in all_names if in_degree[n] == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(topo_order) < len(all_names):
            logging.warning(
                "Cycle detected in data flow graph. "
                "Returning partial topological order plus remaining cyclic nodes."
            )
        
            remaining = [n for n in all_names if n not in topo_order]
            topo_order.extend(remaining)
        
        return topo_order
        
    def _find_files_for_subtree(self, subtree_name: str) -> List[FileNode]:
        """Find all file nodes that belong to a specific subtree by getting files from RPG"""
        
        file_nodes = []
    
        if not self.repo_rpg:
            logging.warning(f"No RPG available, cannot find files for subtree: {subtree_name}")
            return file_nodes
            
        # First, find the subtree node in RPG by name
        subtree_node = None
        for node in self.repo_rpg.nodes.values():
            if node.name == subtree_name:
                subtree_node = node
                break
                
        if not subtree_node:
            logging.warning(f"Subtree node '{subtree_name}' not found in RPG")
            return file_nodes
            
        # Get all descendant nodes recursively
        descendant_nodes = subtree_node.children(recursive=True)
        
        # Filter for file nodes and extract their paths
        file_paths = []
        for node in descendant_nodes:
            if node.meta and node.meta.type_name == NodeType.FILE and node.meta.path:
                if isinstance(node.meta.path, str):
                    file_paths.append(node.meta.path)
                elif isinstance(node.meta.path, list):
                    file_paths.extend(node.meta.path)
        
        file_paths = list(set(file_paths))  # Deduplicate
        # Now find the corresponding FileNode objects in skeleton using these paths
        if not self.skeleton or not self.skeleton.root:
            logging.warning("No skeleton available, cannot retrieve FileNode objects")
            return file_nodes
        
        def traverse_skeleton_node(skel_node):
            nonlocal file_nodes
            
            if isinstance(skel_node, FileNode):
                if skel_node.path in file_paths:
                    file_nodes.append(skel_node)
            elif isinstance(skel_node, DirectoryNode):
                for child in skel_node.children():
                    traverse_skeleton_node(child)
        
        traverse_skeleton_node(self.skeleton.root)
        
        logging.info(f"Found {len(file_nodes)} files for subtree '{subtree_name}' from RPG")
        return file_nodes
    
    def _plan_file_order_for_subtree(
        self,
        file_nodes: List[FileNode],
        max_retry: int=5,
    ) -> List[str]:
        """Plan the order of files within a subtree using a simple list format"""
        file_paths = [node.path for node in file_nodes]

        # If only one file, return directly
        if len(file_paths) <= 1:
            return file_paths

        file_info_map = {
            file_node.path: file_node.feature_paths for file_node in file_nodes
        }

        file_names = list(file_info_map.keys())
        trees_info = self.repo_rpg.visualize_dir_map(max_depth=2)

        files_to_planned = ""
        for path, features in file_info_map.items():
            feature_lines = "\n  - ".join(features) if features else "  (no feature paths recorded)"
            files_to_planned += f"- {path}:\n  - {feature_lines}\n\n"

        env_prompt = PLAN_FILE_LIST.format(
            repo_info=self.repo_rpg.repo_info,
            trees_info=trees_info,
            files_to_planned=files_to_planned
        )

        memory = Memory(context_window=max_retry)

        llm_client = LLMClient(self.llm_config)

        final_order = []
        for i in range(max_retry):
            try:
                memory.add_message(
                    UserMessage(content=env_prompt)
                )
                planning_result, response = llm_client.call_with_structure_output(
                    memory=memory,
                    response_model=FileImplementationOrder,
                    retry_delay=3
                )
                memory.add_message(
                    AssistantMessage(content=response)
                )

                ordered_files = planning_result.get("implementation_order", [])

                env_prompt, flag = validate_file_implementation_list(
                    file_list=ordered_files,
                    file_names=file_names
                )

                if not flag:
                    final_order = []
                    continue
                else:
                    final_order = ordered_files
                    break
            except Exception as e:
                logging.warning(f"[Retry {i + 1}] Failed to parse or validate GPT response: {e}")
                if i == max_retry - 1:
                    logging.error("Exceeded maximum retries. Using fallback ordering.")

        # Fallback: return alphabetically sorted file paths
        if not final_order:
            final_order = sorted(file_paths)

        return final_order

    def run(
        self,
        result_path: str,
    ) -> Tuple[RPG, Dict[str, Any]]:
        """
        Full pipeline to build comprehensive repository graph design.
        
        Returns:
            Tuple[RPG, Dict]: (final_rpg, all_results)
        """
        
        logging.info("[FuncDesigner] === Start building repository graph design ===")
        
        # Phase 1: Build data flow
        data_flow, updated_rpg, flow_results = self.build_data_flow()
        
        if not data_flow and "error" in flow_results:
            raise Exception("[FuncDesigner] Data flow building failed. Continuing with available data.")
        
        logging.info(f"[FuncDesigner] Data flow built with {len(data_flow)} flow units")
        
        # Phase 3: Design base classes
        base_classes, updated_rpg, class_results = self.design_base_classes()
        
        if not base_classes and "error" in class_results:
            raise Exception("[FuncDesigner] Base class design failed. Continuing with available data.")
            # base_classes = {}
        
        logging.info(f"[FuncDesigner] Base classes designed: {len(base_classes)} classes")
    
        # Phase 4: Design interfaces
        interfaces, final_rpg, interface_results = self.design_interfaces(
            data_flow=data_flow
        )
        
        if not interfaces and "error" in interface_results:
            raise Exception("[FuncDesigner] Interface design failed. Continuing with available data.")
        
        logging.info(f"[FuncDesigner] Interfaces designed: {len(interfaces)} interfaces")
        
        # Final RPG consolidation
        self.repo_rpg = final_rpg
        self.repo_rpg.update_all_metadata_bottom_up()
        
        logging.info(f"[FuncDesigner] Final RPG ready. Nodes: {len(self.repo_rpg.nodes)}, Edges: {len(self.repo_rpg.edges)}")
        
        # Extract subtree order from data flow
        subtree_order = []
        seen_subtrees = set()
        for flow in data_flow:
            source = flow.get("source", "")
            target = flow.get("target", "")
            
            # Add source subtree if not seen
            if source and source not in seen_subtrees:
                subtree_order.append(source)
                seen_subtrees.add(source)
            
            # Add target subtree if not seen
            if target and target not in seen_subtrees:
                subtree_order.append(target)
                seen_subtrees.add(target)
        
        # Compile all results
        all_results = {
            "data_flow_phase": {
                "data_flow": data_flow,
                "subtree_order": subtree_order,
                "agent_results": flow_results
            },
            "base_classes_phase": {
                "base_classes": base_classes,
                "agent_results": class_results
            },
            "interfaces_phase": {
                "interfaces": interfaces,
                "agent_results": interface_results
            },
            "final_node_count": len(self.repo_rpg.nodes),
            "final_edge_count": len(self.repo_rpg.edges),
            "config_used": self.config,
            "success": True
        }
        
        # Save RPG to result path
        self._save_rpg_results(result_path, all_results)
        
        return self.skeleton, self.repo_rpg, all_results
    
    def _save_rpg_results(self, result_path: str, all_results: Dict):
        """Save RPG and results to files"""
        try:
            # Save RPG structure
            # rpg_path = result_path.replace('.json', '_rpg.json')
            # with open(rpg_path, 'w') as f:
            #    json.dump(self.repo_rpg.to_dict(), f, indent=2)
       
            # Save comprehensive results
            with open(result_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logging.info(f"[FuncDesigner] Results saved to {result_path}")
            # logging.info(f"[FuncDesigner] RPG saved to {rpg_path}")
            
        except Exception as e:
            logging.error(f"[FuncDesigner] Failed to save results: {e}")