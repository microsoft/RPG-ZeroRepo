from typing import Union, Dict, Optional, List, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from copy import deepcopy
from zerorepo.rpg_gen.base import (
    RPG, Node, NodeMetaData, NodeType,
    Tool, ToolResult, ToolExecResult, ToolCall, ToolCallArguments,
    LLMClient, LLMConfig, Memory,
    AgentwithReview, ReviewEnv,
    DirectoryNode
)
from zerorepo.rpg_gen.base.node import extract_feature_list_from_tree, show_project_structure_from_tree
from zerorepo.utils.tree import remove_paths, get_all_leaf_paths
from ..util import process_grouping_assignments, add_file_node_with_features
from ..prompts.group_skeleton import ( 
    GROUP_SKELETON, 
    GROUP_SKELETON_REVIEW, 
    ASSIGN_FEATURES_TO_FILES_TOOL
)

class ReviewCategory(BaseModel):
    """Single review dimension block."""
    feedback: str = Field(description="Concrete, actionable guidance for this dimension")
    pass_: bool = Field(alias="pass", description="Whether this dimension passed review")

    class Config:
        populate_by_name = True


class GroupReviewBlock(BaseModel):
    """Full review section containing all five required dimensions."""
    
    File_Scope_Appropriateness: ReviewCategory = Field(
        alias="File Scope Appropriateness"
    )
    File_Structure_Organization: ReviewCategory = Field(
        alias="File Structure Organization"
    )
    Modularity_Cohesion: ReviewCategory = Field(
        alias="Modularity & Cohesion"
    )
    Naming_Quality: ReviewCategory = Field(
        alias="Naming Quality"
    )
    Structural_Soundness: ReviewCategory = Field(
        alias="Structural Soundness"
    )

    class Config:
        populate_by_name = True


class GroupSkeletonReviewOutput(BaseModel):
    """Final JSON output for the group skeleton review process."""
    
    review: GroupReviewBlock
    final_pass: bool = Field(
        description="True only if all dimensions passed or remaining issues are minor"
    )


class AssignFeaturesToFilesToolParamModel(BaseModel):
    assignments: Dict[str, List[str]] = Field(
        description="Mapping of file paths to list of features to assign"
    )
        
class AssignFeaturesToFilesTool(Tool):
    name = "assign_features_to_files"
    description = ASSIGN_FEATURES_TO_FILES_TOOL
        
    ParamModel = AssignFeaturesToFilesToolParamModel

       
    @classmethod
    def custom_parse(cls, raw: str) -> List[ToolCallArguments]:
        """
        Parse a raw JSON input, validate it against SearchCodeParam,
        and return a validated dictionary of parameters.
        """
        valid_tools = []  # always return a list

        # --- Step 1: Parse JSON safely ---
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()

                raw_tools = json.loads(raw)
                raw_tools = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        except json.JSONDecodeError as e:
            logging.error(f"[custom_parse] Invalid JSON: {e.msg}")
            return valid_tools  # return empty list, but at least valid type

        # --- Step 2: Process each tool independently ---
        for raw_tool in raw_tools:
            try:
                # Tool name check
                tool_name = raw_tool.get("tool_name", "")
                if tool_name.lower().strip() != cls.name.lower():
                    logging.warning(
                        f"Warning: tool_name '{tool_name}' does not match expected '{cls.name}'"
                    )
                    continue

                # Extract parameters
                params = raw_tool.get("parameters", raw_tool)

                # Validate
                parsed = cls.ParamModel(**params).model_dump()
                valid_tools.append(parsed)

            except ValidationError as e:
                logging.error(f"[custom_parse] Parameter validation failed: {e.errors()}")
                # Do NOT return; just skip this tool and continue
                continue
            except Exception as e:
                logging.error(f"[custom_parse] Unexpected error: {e}")
                continue

        return valid_tools
    
    @classmethod
    async def execute(cls, arguments: Dict, env: Optional[Any] = None) -> ToolResult:
        try:
            
            
            assignments = arguments.get("assignments", {})
            top_paths = deepcopy(env.top_paths)
            remain_subtree = deepcopy(env.remain_tree)
            
            suc_mappings, err_msg, ok = process_grouping_assignments(
                assignments=assignments,
                top_paths=top_paths,
                subtree_dict=remain_subtree
            )
            
            if not ok:
                import pdb
                pdb.set_trace()
                top_paths_str = ", ".join(top_paths)
                err_msg = (
                    "Your design for this round is invalid, so the skeleton was not updated. Details:\n"
                    f"{err_msg}\n"
                    f"All file paths in your design must be located under the target folder: {top_paths_str}"
                )
                return ToolExecResult(
                    output=None,
                    error=err_msg,
                    error_code=1,
                    state={}
                )
            else:
                return ToolExecResult(
                    output=err_msg,
                    error_code=0,
                    state={"suc_mappings": suc_mappings}
                )
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=str(e),
                error_code=1,
                state={}
            )
            


class GroupSkeletonEnv(ReviewEnv):
    """Environment for grouping phase"""
    
    def __init__(
        self,
        review_llm: LLMClient,
        review_format: BaseModel,
        review_memory: Memory,
        logger: logging.Logger,
        
        review_times: int = 1,
        register_tools: List = [],
        
        file_map: Dict={},
        subtree_name: str = "",
        tree_root: DirectoryNode=None,
        repo_rpg: RPG = None,
    ):
        
        super().__init__(
            review_llm, review_format,
            review_memory, review_times,
            register_tools, logger
        )
        self.tree_root = tree_root   
        self.repo_rpg = repo_rpg
    
        # Initialize subtree data
        self.subtree_name = subtree_name
    
        subtree_node = self.repo_rpg.get_node_by_feature_path(self.subtree_name)
        assert subtree_node, ""
        top_paths = subtree_node.meta.path
        self.top_paths = top_paths if isinstance(top_paths, list) else [top_paths]
        
        functional_graph = self.repo_rpg.get_functionality_graph()
        subtree = next(
            (
                subtree_dict["refactored_subtree"]
                for subtree_dict in functional_graph
                if subtree_dict["name"] == subtree_name
            ),
            None
        )
        assert subtree, ""
        self.subtree = subtree        
        self.orig_tree = deepcopy(self.subtree) if self.subtree else {}
        self.remain_tree = deepcopy(self.orig_tree)
        self.all_paths = get_all_leaf_paths(self.orig_tree)
        self.all_paths = ["/".join([subtree_name] + path.split("/")) for path in self.all_paths]
        self.file_map = file_map
        
        self.util = 0.0
        self.is_completed = False
    
    def post_feedback(self):
        if self.last_action_suc:
            skeleton_view = show_project_structure_from_tree(self.tree_root, show_features=False)
            remain_tree = json.dumps(self.remain_tree, indent=2)
            
            return (
                "Current Status:\n"
                f"Current Skeleton:\n{skeleton_view}\n"
                f"Remaining Subtrees:\n{remain_tree}\n"
                "Please continue designing feature paths for the remaining subtrees "
                "and map them to appropriate locations in the skeleton. "
                "Do not modify paths or feature assignments that have already been placed."
            )
        else:
            return (
                "All of your tool calls in this round have failed. Please reflect on the errors, correct your approach, and try again."
            )
            
    def load_review_message(self, tool_calls: List[ToolCall]):
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info

        all_paths_str = ", ".join(self.top_paths)
        cur_skeleton_info = show_project_structure_from_tree(self.tree_root, show_features=False)
        
        review_input = (
            f"Repository Name: {repo_name}\n"
            f"Repository Information: {repo_info}\n"
            f"Current Skeleton: {cur_skeleton_info}\n"
            f"Designated Functional Folder(s): {all_paths_str}.\n"
            f"Agent’s Design in This Round: {json.dumps([tool_call.to_dict() for tool_call in tool_calls], indent=2)}\n"
        )
        
        return review_input
        
    def should_terminate(self, tool_calls, tool_results):
        """Custom termination logic - check if all features are assigned or max utilization reached"""
        return self.is_completed or self.util >= 0.99
    
    def update_env(self, tool_call, result):
        if tool_call.name == "assign_features_to_files" and result.success:
            try:
                suc_mappings = result.state.get("suc_mappings", [])
                
                # Create trial copies for testing
                trial_root = deepcopy(self.tree_root)
                trial_remain = deepcopy(self.remain_tree)
                file_map = deepcopy(self.file_map)
                
                for (file_path, feature_paths) in suc_mappings:
                    
                    new_feature_paths = []
                    for feature_path in feature_paths:
                        features = feature_path.split("/")
                        features = [self.subtree_name] + features
                        file_feature_path = "/".join(features)
                        new_feature_paths.append(file_feature_path)
                    
                    feature_paths = new_feature_paths
                    for feature_path in feature_paths:      
                        feature_path = feature_path.split("/")[:-1]
                        feature_path = "/".join(feature_path)  
                        feature_node = self.repo_rpg.get_node_by_feature_path(
                            feature_path
                        )
                        if not feature_node:
                            continue
                        
                        if feature_node.meta is None:
                            feature_node.meta = NodeMetaData(
                                type_name=NodeType.FILE,
                                path=file_path
                            )
                        else:
                            feature_node.meta.type_name = NodeType.FILE
                            if not feature_node.meta.path:
                                feature_node.meta.path = file_path
                            else:
                                if isinstance(feature_node.meta.path, str):
                                    feature_node.meta.path = file_path if file_path == feature_node.meta.path \
                                        else [feature_node.meta.path, file_path]
                                else:
                                    feature_node.meta.path.append(file_path)
                                    feature_node.meta.path = list(set(feature_node.meta.path))

                    add_file_node_with_features(
                        trial_root,
                        file_path,
                        feature_paths,
                        file_map
                    )
                    
                # Update actual state if successful
                self.tree_root = trial_root
                self.remain_tree = trial_remain
                
                # Update utilization
                features = set(extract_feature_list_from_tree(self.tree_root))
                covered = features.intersection(set(self.all_paths))
                self.util = len(covered) / len(self.all_paths) if self.all_paths else 1.0
                covered_cut = ["/".join(path.split("/")[1:]) for path in covered]
                self.remain_tree = remove_paths(self.orig_tree, list(covered_cut), inplace=False)
                logging.info(f"[GroupSkeleton] Updated utilization: {self.util:.2%}")
                # Check if we're done
                if self.util >= 0.99 or len(covered) == len(self.all_paths) or not self.remain_tree:
                    self.is_completed = True
                    logging.info(f"[GroupSkeleton] Subtree {self.subtree_name} completed")
            except Exception as e:
                logging.error(f"Failed to update grouping: {e}")
    

class GroupSkeletonAgent(AgentwithReview):
    """Agent responsible for grouping features into files"""
    
    def __init__(
        self, 
        llm_cfg: Union[str, Dict, LLMConfig],
        design_system_prompt: str = "",
        design_context_window: int = 30,
        review_system_prompt: str = "",
        review_context_window: int = 10,
        max_review_times: int = 3,
        
        tree_root: DirectoryNode=None,
        subtree_name: str="",
        file_map: Dict={},
        repo_rpg: RPG = None,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        if register_tools is None:
            register_tools = [AssignFeaturesToFilesTool()]
        
        super().__init__(
            llm_cfg=llm_cfg,
            design_system_prompt=design_system_prompt,
            design_context_window=design_context_window,
            review_system_prompt=review_system_prompt,
            review_format=GroupSkeletonReviewOutput,
            max_review_times=max_review_times,
            register_tools=register_tools,
            logger=logger,
            review_context_window=review_context_window,
            **kwargs
        )
        
        self._env = GroupSkeletonEnv(
            review_llm=self.review_llm,
            review_format=GroupSkeletonReviewOutput,
            review_memory=self.review_memory,
            logger=logger or logging.getLogger(__name__),
            review_times=max_review_times,
            register_tools=register_tools,
            
            tree_root=tree_root,
            subtree_name=subtree_name,
            file_map=file_map,
            repo_rpg=repo_rpg
        )
    
        self.repo_rpg = repo_rpg
    
    def group_subtree_features(
        self, 
        max_steps: int = 20,
        max_retry: int = 3
    ) -> Dict:
        """Group features from a specific subtree into the skeleton"""
        
        # Build initial task
        all_paths_str = ", ".join(self._env.top_paths)
        cur_skeleton_str = show_project_structure_from_tree(self._env.tree_root, show_features=False)
        task = (
            f"Designated Functional Folder(s): {all_paths_str}.\n"
            f"You need to design the file structure for these top-level folders.\n\n"
            f"## Current Repository Skeleton Overview:\n{cur_skeleton_str}\n"
            f"## Remaining Features to Assign from Subtree:\n"
            f"Remaining Features (JSON structure):\n"
            f"{json.dumps(self._env.remain_tree, indent=2)}\n"
            f"You must assign **all** remaining features to appropriate files or directories in the proposed structure.\n"
        )
        # Run the agent
        result = self.run(task=task, max_error_times=max_retry, max_steps=max_steps)
        
        return {
            "tree_root": self._env.tree_root,
            "repo_rpg": self._env.repo_rpg,
            **result
        }