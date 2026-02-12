from typing import Union, Dict, Optional, List, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from zerorepo.rpg_gen.base import (
    RPG, NodeType,
    Tool, ToolResult, ToolCall,
    ToolExecResult, ToolCallArguments,
    LLMClient, LLMConfig, Memory,
    AgentwithReview, ReviewEnv
)
from ..prompts.raw_skeleton import(
    RAW_SKELETON,
    RAW_SKELETON_REVIEW, 
    RAW_SKELETON_TOOL
)
from ..util import extract_all_strings


class ReviewCategory(BaseModel):
    """Single review dimension block"""
    feedback: str = Field(description="Concrete, actionable guidance for this dimension")
    pass_: bool = Field(alias="pass", description="Whether this dimension passed review")

class ReviewBlock(BaseModel):
    """Full review section containing all four required dimensions"""
    Functional_Grouping: ReviewCategory = Field(alias="Functional Grouping")
    Simplified_Bridging_Components: ReviewCategory = Field(alias="Simplified Bridging Components")
    Exclusive_Assignment: ReviewCategory = Field(alias="Exclusive Assignment")
    Semantic_Naming: ReviewCategory = Field(alias="Semantic Naming")

    class Config:
        allow_population_by_field_name = True

class RawSkeletonReviewOutput(BaseModel):
    """Final JSON output for the skeleton review process"""
    review: ReviewBlock
    final_pass: bool = Field(description="True only if all passes or remaining issues are minor")
    

class GenerateRawSkeletonToolParamModel(BaseModel):
    raw_skeleton: Dict = Field(description="The raw skeleton structure")
    
    
class GenerateRawSkeletonTool(Tool):
    name = "generate_raw_skeleton"
    description = RAW_SKELETON_TOOL
    ParamModel = GenerateRawSkeletonToolParamModel
        
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
            raw_skeleton = arguments.get("raw_skeleton", {})
            repo_rpg: RPG = env.repo_rpg
            
            parsed_area_names = extract_all_strings(raw_skeleton)
            gt_area_names = repo_rpg.get_functional_areas()
            
            p_area_names_set, gt_area_names_set = set(parsed_area_names), \
                set(gt_area_names)
                
            missing = gt_area_names_set - p_area_names_set
            imagined = p_area_names_set- gt_area_names_set
            
            passed = True
            notes = []
            
            if not parsed_area_names:
                notes.append("No subtree names were found.")
                passed = False
            if missing:
                notes.append(f"Missing: {', '.join(sorted(missing))}.")
                passed = False
            if imagined:
                notes.append(f"Unrecognized: {', '.join(sorted(imagined))}.")
                passed = False
            if passed:
                notes.append("All subtrees present and valid.")
            
            if passed:
                return ToolExecResult(
                    output=json.dumps(raw_skeleton),
                    error=None,
                    error_code=0,
                    state={"raw_skeleton": raw_skeleton}
                )    
            else:
                feedback = "\n".join(notes)
                return ToolExecResult(
                    output=None,
                    error=feedback,
                    error_code=1,
                    state={}   
                )
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=str(e),
                error_code=1,
                state={}
            )


class RawSkeletonEnv(ReviewEnv):
    """Environment for raw skeleton generation phase"""
    
    def __init__(
        self,
        review_llm: LLMClient,
        review_format: BaseModel,
        review_memory: Memory,
        review_times: int = 1,
        register_tools: List = [],
        repo_rpg: RPG = None,
        logger: logging.Logger=None
    ):
        super().__init__(
            review_llm, review_format,
            review_memory, review_times,
            register_tools
        )
        self.repo_rpg = repo_rpg
        self.name_to_path_map = {}
        self.raw_skeleton_dict = {}
        self.is_completed = False
        self.logger = logger
    
    def load_review_message(self, tool_calls: List[ToolCall]):
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info  
        repo_tree = self.repo_rpg.visualize_dir_map(max_depth=3, feature_only=True)
        
   
        trees_names = self.repo_rpg.get_functional_areas()
        
        tool_str = json.dumps([tool_call.to_dict() for tool_call in tool_calls], indent=2)
        # Build task description
        review_input = (
            "You should review a high-level repository skeleton design based on the following information:\n"
            f"Repository Name: {repo_name}\n"
            f"Description: {repo_info}\n"
            f"Functional Graph Overview:\n{json.dumps(repo_tree, indent=2)}\n"
            f"Available Functional Areas: {', '.join(trees_names)}\n"
            f"Here are the proposed skeleton designs:\n{tool_str}\n"
            "Please evaluate the designs according to the review criteria defined in the system prompt and produce a single consolidated review.\n"
            "Respond with a valid JSON object that strictly follows the required output format in the system prompt.\n"
            "Do not include any text or comments outside of the JSON object."
        )
            
        return review_input
    
    def should_terminate(self, tool_calls, tool_results):
        """Custom termination logic - check if raw skeleton is complete"""
        return self.is_completed
    
    def update_env(self, tool_call, result):
        cmpt_names = self.repo_rpg.get_functional_areas()
        if tool_call.name == "generate_raw_skeleton" and result.success:
            try:
                result.state = result.state if result.state else {}
                raw_skeleton = result.state.get("raw_skeleton", {})
                self.raw_skeleton_dict = raw_skeleton
                # Update name_to_path_map
                self.name_to_path_map = self._reverse_skeleton_map(
                    raw_skeleton, 
                    cmpt_names=cmpt_names
                )
                # Check if all components are covered
                if all(cmpt in self.name_to_path_map for cmpt in cmpt_names):
                    self.is_completed = True
                    
                    for cmpt, dir_paths in self.name_to_path_map.items():
                        feature_node = self.repo_rpg.get_node_by_feature_path(cmpt)
                        if not feature_node:
                            self.logger.info(f"Not Found the Feature Node for {cmpt}, skip...")
                            continue
                        feature_node.meta.type_name = NodeType.DIRECTORY
                        feature_node.meta.path = dir_paths
                    logging.info("[RawSkeleton] All components covered, marking as completed")
            except Exception as e:
                logging.error(f"Failed to update raw skeleton: {e}")
    
    def _reverse_skeleton_map(self, skeleton_dict: dict, prefix: str = "", 
                           cmpt_names: Optional[List[str]] = None) -> dict:
        result = {}
        for key, value in skeleton_dict.items():
            current_path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, list):
                for cmpt in value:
                    if cmpt_names is None or cmpt in cmpt_names:
                        result.setdefault(cmpt, []).append(current_path)
            elif isinstance(value, dict):
                sub_result = self._reverse_skeleton_map(value, current_path, cmpt_names)
                for cmpt, paths_list in sub_result.items():
                    result.setdefault(cmpt, []).extend(paths_list)
        return result


class RawSkeletonAgent(AgentwithReview):
    """Agent responsible for generating raw skeleton structure"""
    
    def __init__(
        self, 
        llm_cfg: Union[str, Dict, LLMConfig],
        design_system_prompt: str = "",
        design_context_window: int = 30,
        review_system_prompt: str = "",
        review_context_window: int = 10,
        max_review_times: int = 1,
        repo_rpg: RPG = None,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        if register_tools is None:
            register_tools = [GenerateRawSkeletonTool()]
        
        super().__init__(
            llm_cfg=llm_cfg,
            design_system_prompt=design_system_prompt,
            design_context_window=design_context_window,
            review_system_prompt=review_system_prompt,
            review_format=RawSkeletonReviewOutput,
            max_review_times=max_review_times,
            register_tools=register_tools,
            logger=logger,
            review_context_window=review_context_window,
            **kwargs
        )

        self.repo_rpg = repo_rpg
        
        # Create custom environment
        self._env = RawSkeletonEnv(
            review_llm=self.review_llm,
            review_format=RawSkeletonReviewOutput,
            review_memory=self.review_memory,
            review_times=max_review_times,
            register_tools=register_tools,
            repo_rpg=repo_rpg
        )
    
    def generate_raw_skeleton(
        self, 
        max_steps: int = 2,
        max_retry: int=3
    ) -> Dict:
        """Generate raw skeleton structure"""
        
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info
        
        repo_tree = self.repo_rpg.visualize_dir_map(max_depth=3, feature_only=True)
        
        trees_names = self.repo_rpg.get_functional_areas()
        
        # Build task description
        task = (
            f"Design a high-level skeleton structure for the repository based on the following information:\n"
            f"Repository Name: {repo_name}\n"
            f"Description: {repo_info}\n"
            f"Functional Graph Overview: {json.dumps(repo_tree, indent=2)}\n"
            f"Available Functional Areas: {','.join(trees_names)}\n"
            "Please generate a comprehensive directory structure that can accommodate all the functional areas listed above."
        )
        
        result = self.run(
            task=task, 
            max_error_times=max_retry,
            max_steps=max_steps
        )
        
        return {
            "raw_skeleton": self._env.raw_skeleton_dict,
            "repo_rpg": self._env.repo_rpg,
            "name_to_path_map": self._env.name_to_path_map,
            **result
        }