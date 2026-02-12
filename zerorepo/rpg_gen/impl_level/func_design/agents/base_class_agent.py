import logging
import json, json5
import ast
from copy import deepcopy
from typing import Union, Dict, Optional, List, Any
from pydantic import BaseModel, Field, ValidationError
from zerorepo.rpg_gen.base import (
    RPG, Tool, ToolCall,
    ToolExecResult, ToolCallArguments,
    LLMClient, LLMConfig, Memory,
    AgentwithReview, ReviewEnv,
    RepoSkeleton
)
from ..prompts import (
    DESIGN_BASE_CLASSES, 
    DESIGN_BASE_CLASSES_REVIEW, 
    GENERATE_BASE_CLASS_TOOL
)

# ========================
# 1. REVIEW MODELS
# ========================
class BaseClassReviewCategory(BaseModel):
    """Single review dimension block"""
    feedback: str = Field(description="Concrete, actionable guidance for this dimension")
    pass_: bool = Field(alias="pass", description="Whether this dimension passed review")


class BaseClassReviewBlock(BaseModel):
    """Full review section for base class design"""
    Design_Quality: BaseClassReviewCategory = Field(alias="Design Quality")
    Reusability: BaseClassReviewCategory
    Abstraction_Level: BaseClassReviewCategory = Field(alias="Abstraction Level")
    Interface_Clarity: BaseClassReviewCategory = Field(alias="Interface Clarity")
    
    class Config:
        populate_by_name = True


class BaseClassReviewOutput(BaseModel):
    """Final JSON output for the base class review process"""
    review: BaseClassReviewBlock
    final_pass: bool = Field(description="True only if all passes or remaining issues are minor")


# ========================
# 2. AGENT TOOL
# ========================
class BaseClassDefinition(BaseModel):
    file_path: str = Field(..., description="file path of the base classes")
    code: str = Field(..., description="Code of Base Classes")
    
    
class GenerateBaseClassesToolParamModel(BaseModel):
    base_classes: List[BaseClassDefinition] = Field(
        ...,
        description="List of base class definitions",
    )
    

class GenerateBaseClassesTool(Tool):
    name = "generate_base_classes"
    description = GENERATE_BASE_CLASS_TOOL
    
    ParamModel = GenerateBaseClassesToolParamModel
    
    @classmethod
    def custom_parse(cls, raw: str) -> List[ToolCallArguments]:
        """Parse raw JSON input for base classes"""
        valid_tools = []
        
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()
                
                raw_tools = json5.loads(raw)
                raw_tools = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        except json.JSONDecodeError as e:
            logging.error(f"[custom_parse] Invalid JSON: {e}")
            return valid_tools
        
        for raw_tool in raw_tools:
            try:
                tool_name = raw_tool.get("tool_name", "")
                if tool_name.lower().strip() != cls.name.lower():
                    logging.warning(f"Tool name '{tool_name}' does not match expected '{cls.name}'")
                    continue
                
                params = raw_tool.get("parameters", raw_tool)
                parsed = cls.ParamModel(**params).model_dump()
                valid_tools.append(parsed)
            except ValidationError as e:
                logging.error(f"[custom_parse] Parameter validation failed: {e.errors()}")
                continue
            except Exception as e:
                logging.error(f"[custom_parse] Unexpected error: {e}")
                continue
        
        return valid_tools
    
    @classmethod
    async def execute(cls, arguments: Dict, env: Optional[Any] = None) -> ToolExecResult:
        try:
            base_classes = arguments.get("base_classes", [])
            
            if not base_classes:
                return ToolExecResult(
                    output=None,
                    error="Empty base classes provided",
                    error_code=1,
                    state={}
                )

            feedbacks = []
            no_error = True
            for base_class in base_classes:
                file_path = base_class["file_path"]
                code = base_class["code"]
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    error_message = (
                        f"There is a syntax error in the base class code you designed in file '{file_path}': "
                        f"line {e.lineno}, column {e.offset}: {e.msg}"
                    )
                    feedbacks.append(
                       error_message
                    )
                    no_error = False
            
            if not no_error:
                return ToolExecResult(
                    output=None,
                    error="\n".join(feedbacks),
                    error_code=1
                )
            
            return ToolExecResult(
                output="You have successfully designed these base classes, and we have pushed them to the project.",
                error=None,
                error_code=0,
                state={"base_classes": base_classes}
            )
            
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=str(e),
                error_code=1,
                state={}
            )


# ========================
# 3. ENVIRONMENT
# ========================

class BaseClassEnv(ReviewEnv):
    """Environment for base class generation phase"""
    
    def __init__(
        self,
        review_llm: LLMClient,
        review_format: BaseModel,
        review_memory: Memory,
        review_times: int = 1,
        register_tools: List = [],
        repo_skeleton: RepoSkeleton = None,
        repo_rpg: RPG = None,
        logger: logging.Logger = None
    ):
        super().__init__(
            review_llm, review_format,
            review_memory, review_times,
            register_tools,
            logger
        )
        self.repo_rpg = repo_rpg
        self.repo_skeleton = repo_skeleton
        self.base_classes = []
        self.is_completed = False
        self.logger = logger or logging.getLogger(__name__)
    
    def load_review_message(self, tool_calls: List[ToolCall]):
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info
        data_flow = self.repo_rpg.data_flow
        
        data_flow_str = json.dumps(data_flow, indent=2)
        repo_skeleton_str = self.repo_skeleton.to_tree_string()
        
        tree_overview = self.repo_rpg.visualize_dir_map(max_depth=3, feature_only=False)
        tool_str = json.dumps([tool_call.to_dict() for tool_call in tool_calls], indent=2)
        
        review_input = (
            "You should review a base class design based on the following information:\n"
            f"Repository Name: {repo_name}\n"
            f"Description: {repo_info}\n"
            f"Repository Skeleton: {repo_skeleton_str}\n"
            f"Functional Graph Overview: {tree_overview}\n"
            f"Data Flow for the Functional Graph: {data_flow_str}\n"
            f"Proposed base classes:\n{tool_str}\n"
            "Please evaluate the base classes according to the review criteria and produce a consolidated review.\n"
            "Respond with a valid JSON object that follows the required output format in the system prompt.\n"
            "Do not include any text outside of the JSON object."
        )
        
        return review_input
    
    def should_terminate(self, tool_calls, tool_results):
        """Check if base class generation is complete"""    
        self.is_completed = all(tool_result.success for tool_result in tool_results) \
            and tool_results and self.last_action_suc
        if self.is_completed:
            self.repo_rpg.base_classes = self.base_classes
        return self.is_completed
    
    def update_env(self, tool_call, result):
        """Update environment state after tool execution"""
        if tool_call.name == "generate_base_classes" and result.success:
            try:
                base_classes = result.state.get("base_classes", [])
                base_file_paths = []
                if base_classes:
                    for base_class in base_classes:
                        file_path = base_class['file_path']
                        base_file_paths.append(file_path)
                        code = base_class['code']
                        file_node = self.repo_skeleton.find_file(path=file_path)
                        if not file_node:
                            self.repo_skeleton.insert_file(
                                file_path=file_path,
                                code=code
                            )    
                        else:
                            file_node.code = file_node.code + "\n" + code
                            file_node.code = file_node.code.strip()

                update_base_classes = []
                base_file_paths = list(set(base_file_paths))
                for file_path in base_file_paths:
                    file_node = self.repo_skeleton.find_file(path=file_path)
                    if not file_node:
                        continue
                    code = file_node.code
                    update_base_classes.append(
                        {
                            "file_path": file_path,
                            "code": code
                        }
                    )
                
                if not self.base_classes:
                    self.base_classes = update_base_classes
                else:
                    to_base_classes = deepcopy(self.base_classes)        
                    
                    for update_base_class in update_base_classes:
                        update_file_path = update_base_class["file_path"]
                        update_code = update_base_class["code"]
                        
                        file_path_exists = False
                        for to_base_class in to_base_classes:
                            file_path = to_base_class["file_path"]
                            code = to_base_class["code"]

                            if update_file_path == file_path:
                                file_path_exists = True
                                to_base_class["code"] = update_code
                                break
                        
                        if not file_path_exists:
                            to_base_classes.append(
                                {
                                    "file_path": update_file_path,
                                    "code": update_code
                                }
                            )
                    
                    self.base_classes = to_base_classes
            except Exception as e:
                self.logger.error(f"Failed to update base classes: {e}")


# ========================
# 4. AGENT
# ========================niy
class BaseClassAgent(AgentwithReview):
    """Agent responsible for generating base class definitions"""
    
    def __init__(
        self,
        llm_cfg: Union[str, Dict, LLMConfig],
        design_system_prompt: str = DESIGN_BASE_CLASSES,
        design_context_window: int = 30,
        review_system_prompt: str = DESIGN_BASE_CLASSES_REVIEW,
        review_context_window: int = 10,
        max_review_times: int = 1,
        repo_skeleton: RepoSkeleton=None,
        repo_rpg: RPG = None,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        if register_tools is None:
            register_tools = [GenerateBaseClassesTool()]
        
        super().__init__(
            llm_cfg=llm_cfg,
            design_system_prompt=design_system_prompt,
            design_context_window=design_context_window,
            review_system_prompt=review_system_prompt,
            review_format=BaseClassReviewOutput,
            max_review_times=max_review_times,
            register_tools=register_tools,
            logger=logger,
            review_context_window=review_context_window,
            **kwargs
        )
        
        self.repo_skeleton = repo_skeleton
        self.base_classes = []
        self.repo_rpg = repo_rpg
        
        # Create custom environment
        self._env = BaseClassEnv(
            review_llm=self.review_llm,
            review_format=BaseClassReviewOutput,
            review_memory=self.review_memory,
            review_times=max_review_times,
            register_tools=register_tools,
            repo_rpg=repo_rpg,
            repo_skeleton=repo_skeleton,
            logger=logger
        )
        
    
    def generate_base_classes(
        self,
        max_retry: int = 3,
        max_steps: int = 10
    ) -> Dict:
        """Generate base class definitions"""
        
        data_flow = self.repo_rpg.data_flow
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info
        skeleton_info = self.repo_skeleton.to_tree_string()
        
        tree_overview = self.repo_rpg.visualize_dir_map(max_depth=2, feature_only=False)
        functional_areas = self.repo_rpg.get_functional_areas()
        
        # Build task description
        task = (
            f"Based on the repository structure and functional areas, generate base class definitions:\n"
            f"Repository Name: {repo_name}\n"
            f"Repository Info: {repo_info}\n"
            f"Repository Skeleton: {skeleton_info}\n"
            f"Functional Graph Overview: {tree_overview}\n\n"
            f"Functional Areas: {', '.join(functional_areas)}\n"
            f"Data Flow: {json.dumps(data_flow, indent=2)}"
            "Please use the generate_base_classes tool to create base class definitions."
        )
        
        result = self.run(
            task=task,
            max_error_times=max_retry,
            max_steps=max_steps
        )
        
        # TODO: 设计基类后优化Data Flow信息
        
        return {
            "base_classes": self._env.base_classes,
            "repo_rpg": self._env.repo_rpg,
            "success": self._env.is_completed,
            "agent_results": result
        }
