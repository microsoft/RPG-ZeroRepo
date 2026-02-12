import yaml
import json
import logging
from typing import Union, Dict, Tuple, Optional, Any
from pathlib import Path
from zerorepo.rpg_gen.base import (
    RPG, NodeType, Node, NodeMetaData,
    RepoSkeleton, LLMConfig
)
from zerorepo.utils.logs import setup_logger
from zerorepo.config.checkpoint_config import CheckpointManager, CheckpointFiles, get_checkpoint_manager
from .file_design import FileDesigner
from .func_design import FuncDesigner
from .plan_tasks import TaskBatch, TaskPlanner

# 封装完整的类去构建repo
class ImplBuilder:
    
    def __init__(
        self,
        llm_cfg: Optional[Union[str, Dict, LLMConfig]],
        skeleton_cfg: Union[str, Path],
        graph_cfg: Union[str, Path],
        repo_path: Union[str, Path],
        ckpt_dir: Union[str, Path],
        logger: logging.Logger,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize ImplBuilder with checkpoint management
        
        Args:
            llm_cfg: LLM configuration
            skeleton_cfg: skeleton configuration file path
            graph_cfg: graph configuration file path  
            repo_path: target repository path
            ckpt_dir: checkpoint directory
            logger: logger instance
            checkpoint_manager: optional CheckpointManager instance
            checkpoint_config: optional custom checkpoint configuration
        """
        self.llm_cfg = llm_cfg
        self.skeleton_cfg = skeleton_cfg if isinstance(skeleton_cfg, Path) \
            else Path(skeleton_cfg)
        self.graph_cfg = graph_cfg if isinstance(graph_cfg, Path) \
            else Path(graph_cfg)
            
        self.repo_path = repo_path if isinstance(repo_path, Path) \
            else Path(repo_path)
        self.repo_path.mkdir(exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = checkpoint_manager
        # Legacy support - keep ckpt_dir for backward compatibility
        self.ckpt_dir = self.checkpoint_manager.checkpoint_dir
        
        # Save checkpoint configuration
        self.checkpoint_manager.save_config()
        
        self.logger = logger or setup_logger()
        self.logger.info(f"ImplBuilder initialized with checkpoint dir: {self.checkpoint_manager.checkpoint_dir}")
    
    # Properties for accessing checkpoint paths - eliminates hardcoded paths!
    @property
    def repo_data_path(self) -> Path:
        """Get repo data file path"""
        return self.checkpoint_manager.repo_data_path
    
    @property 
    def skeleton_result_path(self) -> Path:
        """Get skeleton result file path"""
        return self.checkpoint_manager.get_path("skeleton")
    
    @property
    def skeleton_agent_result_path(self) -> Path:
        """Get skeleton agent trajectory file path"""
        return self.checkpoint_manager.get_path("skeleton_trajectory")
    
    @property
    def graph_result_path(self) -> Path:
        """Get graph result file path"""
        return self.checkpoint_manager.get_path("graph")
    
    @property
    def tasks_result_path(self) -> Path:
        """Get tasks result file path"""
        return self.checkpoint_manager.get_path("tasks")
    
    @property
    def state_file(self) -> Path:
        """Get task manager state file path"""
        return self.checkpoint_manager.get_path("task_manager_state")
    
    @property
    def global_rpg_path(self) -> Path:
        """Get global RPG file path"""
        return self.checkpoint_manager.get_path("global_repo_rpg")
    
    @property 
    def cur_rpg_path(self) -> Path:
        """Get current RPG file path"""
        return self.checkpoint_manager.get_path("current_repo_rpg")
    
    def load_repo_data(self) -> Dict[str, Any]:
        """Load and validate JSON data from file"""
        try:
            with open(self.repo_data_path, 'r', encoding='utf-8') as f:
                self.repo_data = json.load(f)
            self.logger.info(f"Successfully loaded JSON data from {self.repo_data_path}")
            # Validate required fields
            required_fields = ["repository_name", "Component"]
            missing_fields = [field for field in required_fields if field not in self.repo_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON data from {self.repo_data_path}: {e}")
            raise

    def create_initial_rpg(self) -> RPG:
        """Create initial RPG from JSON repository data"""
        repo_name = self.repo_data.get("repository_name", "repo")
        repo_info = self.repo_data.get("repository_purpose", "")
        repo_cmpt = self.repo_data.get("Component", [])

        # 先建 RPG，再往里面塞节点
        rpg = RPG(
            repo_name=repo_name,
            repo_info=repo_info,
            excluded_files=[]
        )

        # --- 小工具: 统一生成 id ---
        def _new_id(name: str) -> str:
            return f"{name}_{rpg._rand8()}"

        # --- 小工具: 在 parent 下按名字找/建子节点 ---
        def _ensure_child(parent: Node, name: str, is_leaf: bool) -> Node:
            """
            is_leaf=True  -> 视作 feature 节点 (FUNCTION)
            is_leaf=False -> 视作中间分组节点 (DIRECTORY)
            """
            existing = rpg.find_child_by_name(parent.id, name)
            if existing:
                return existing

            node = Node(
                id=_new_id(name),
                name=name,
                # level 这里可以不指定，让后面 recalculate_levels_topdown 来统一收拾
                meta=None
            )
            rpg.add_node(node)
            rpg.add_edge(parent, node)
            return node

        # --- 递归: 把 refactored_subtree 变成节点树 ---
        def _build_from_subtree(parent: Node, subtree: Any):
            """
            subtree 的几种形态：
            - dict:      {"ChildA": {...}, "ChildB": [...]}
            - list:      ["feature1", "feature2"]
            """
            if isinstance(subtree, dict):
                for key, child in subtree.items():
                    if isinstance(child, list):
                        # key 是一层分组，list 里的每个元素是叶子 feature
                        group_node = _ensure_child(parent, key, is_leaf=False)
                        for feat in child:
                            if not feat:
                                continue
                            _ensure_child(group_node, str(feat), is_leaf=True)
                    elif isinstance(child, dict):
                        # 还是 dict -> 中间分组，继续下钻
                        node = _ensure_child(parent, key, is_leaf=False)
                        _build_from_subtree(node, child)
                    else:
                        # 其他类型直接当成 leaf
                        _ensure_child(parent, str(key), is_leaf=True)

            elif isinstance(subtree, list):
                # parent 下直接是一堆 feature
                for feat in subtree:
                    if not feat:
                        continue
                    _ensure_child(parent, str(feat), is_leaf=True)

            else:
                # 单一叶子
                _ensure_child(parent, str(subtree), is_leaf=True)

        # --- 主逻辑: 遍历所有 Component ---
        for component in repo_cmpt:
            cmpt_name = component.get("name", "") or "Component"
            re_tree = component.get("refactored_subtree", {})

            if not re_tree:
                continue

            # 这里：name 节点是 refactored_subtree 的父节点
            # -> 作为 repo 的直接子节点，level=1
            cmpt_node = rpg.find_child_by_name(rpg.repo_node.id, cmpt_name)
            if not cmpt_node:
                cmpt_node = Node(
                    id=_new_id(cmpt_name),
                    name=cmpt_name,
                    level=1,
                    meta=NodeMetaData(
                        type_name=NodeType.DIRECTORY,
                        path=None
                    )
                )
                rpg.add_node(cmpt_node)
                rpg.add_edge(rpg.repo_node, cmpt_node)

            # 把整个 refactored_subtree 都挂到这个 name 节点下面
            _build_from_subtree(cmpt_node, re_tree)

        # 统一按“自顶向下”规则重算 level 和 node_type
        rpg.recalculate_levels_topdown()

        self.logger.info(f"Created initial RPG for repository: {repo_name}")
        return rpg
    
    def build_skeleton(
        self,
        rpg: RPG
    ) -> Tuple[RepoSkeleton, RPG]:
        
        self.logger.info("=== Building Repository Skeleton ===")
        
        skeleton_designer = FileDesigner(
            file_map={},
            llm_config=self.llm_cfg,
            config_path=self.skeleton_cfg,
            rpg=rpg
        )
        
        skeleton, updated_rpg, results = skeleton_designer.run(
            skeleton_result_path=self.skeleton_result_path,
            agent_result_path=self.skeleton_agent_result_path
        )
        
        if not results.get("success", False):
            self.logger.error("Skeleton generation failed")
            raise Exception("Failed to generate repository skeleton")
        
        self.logger.info(f"Successfully built skeleton with {len(skeleton.path_to_node)} nodes")
        
        with open(self.global_rpg_path, 'w') as f:
            json.dump(updated_rpg.to_dict(), f, indent=4)
        return skeleton, updated_rpg
    
    
    def build_graph(
        self,
        skeleton: RepoSkeleton,
        rpg: RPG
    ) -> Tuple[RPG, Dict[str, Any]]:
        """Build repository graph using GraphDesigner"""
        self.logger.info("=== Building Repository Graph ===")
        graph_designer = FuncDesigner(
            repo_skeleton=skeleton,
            llm_config=self.llm_cfg,
            repo_rpg=rpg,
            config_path=str(self.graph_cfg),
            logger=self.logger
        )
        
        final_skeleton, final_rpg, graph_results = graph_designer.run(
            result_path=self.graph_result_path
        )
        
        if not graph_results.get("success", False):
            self.logger.error("Graph generation failed")
            raise Exception("Failed to generate repository graph")
        
        self.logger.info(f"Successfully built graph with {len(final_rpg.nodes)} nodes and {len(final_rpg.edges)} edges")
        
        with open(self.global_rpg_path, 'w') as f:
            json.dump(final_rpg.to_dict(), f, indent=4)

        return final_skeleton, final_rpg, graph_results
    
    def plan_all_tasks(self, rpg, graph_data):    
        planner = TaskPlanner(
            rpg=rpg,
            llm_cfg=self.llm_cfg,
            graph_data=graph_data,
            logger=self.logger
        )
        
        task_batches = planner.plan_all_batches(
            result_path=self.tasks_result_path
        )
        
        return task_batches
    
    
    def run(self):
        state_dict = {}
        if not self.state_file.exists():
            with open(self.state_file, 'w') as f:
                json.dump({}, f, indent=4)
        else:
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
        
        self.load_repo_data()
        if not state_dict.get("build_skeleton", False):
            rpg = self.create_initial_rpg()
            skeleton, rpg = self.build_skeleton(
                rpg
            )
            state_dict["build_skeleton"] = True
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=4)
        else:
            with open(self.skeleton_result_path, 'r') as f:
                skeleton_dict = json.load(f)
            with open(self.global_rpg_path, 'r') as f:
                rpg_dict = json.load(f)
            skeleton = RepoSkeleton.from_dict(skeleton_dict)
            rpg = RPG.from_dict(rpg_dict)
    
        if not state_dict.get("build_function", False):
            skeleton, rpg, _ = self.build_graph(
                skeleton=skeleton,
                rpg=rpg
            )
            state_dict["build_function"] = True
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=4)
        else:
            with open(self.skeleton_result_path, 'r') as f:
                skeleton_dict = json.load(f)
            with open(self.global_rpg_path, 'r') as f:
                rpg_dict = json.load(f)
            skeleton = RepoSkeleton.from_dict(skeleton_dict)
            rpg = RPG.from_dict(rpg_dict)
        
        with open(self.graph_result_path, 'r') as f:
            graph_data = json.load(f)
        
        if not state_dict.get("plan_tasks", False):
            plan_batches = self.plan_all_tasks(
                rpg,
                graph_data
            )
            state_dict["plan_tasks"] = True
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=4)
        else:
            with open(self.tasks_result_path, 'r') as f:
                tasks_data = json.load(f)
            
            # Extract flattened list of task batches using shared utility function
            from zerorepo.utils.task_loader import load_batches_from_dict
            
            planned_batches_dict = tasks_data.get("planned_batches_dict", {})
            plan_batches = load_batches_from_dict(planned_batches_dict, graph_data, self.logger)

        # 设计阶段完成，返回任务批次供代码生成阶段使用
        self.logger.info(f"Design phase completed. Generated {len(plan_batches)} task batches.")
        return plan_batches, skeleton, rpg, graph_data
        
    
        
        
        