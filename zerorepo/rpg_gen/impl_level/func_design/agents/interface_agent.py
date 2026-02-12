from typing import Union, Dict, Optional, List, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import json, json5
import uuid
import ast
from zerorepo.rpg_gen.base import (
    RPG, NodeType,
    NodeMetaData,
    Tool, ToolResult,
    ToolCall,
    ToolExecResult, ToolCallArguments,
    LLMClient, LLMConfig, Memory,
    AgentwithReview, ReviewEnv,
    RepoSkeleton, FileNode,
    ParsedFile
)
from zerorepo.rpg_gen.base.unit import render_codeunits_as_file
from zerorepo.utils.compress import get_skeleton  
from ..prompts import (
    DESIGN_ITF,
    DESIGN_ITF_REVIEW,
    DESIGN_INTERFACE_TOOL
)

def _format_base_classes(base_classes: List) -> str:
    """Format base classes information"""
    if not base_classes:
        return "No base classes available."
    
    base_classes_lines = []
    for classes_list in (base_classes.items() if isinstance(base_classes, dict) else [base_classes]):
        for base_class in classes_list:
            if isinstance(base_class, dict):
                file_path = base_class.get("file_path", "unknown")
                code = base_class.get("code", "")
                base_classes_lines.append(
                    f"### {file_path}\n"
                    f"```python\n{code}\n```\n"
                )
    
    return "\n".join(base_classes_lines)


# ========================
# 1. REVIEW MODELS
# ========================
class InterfaceReviewCategory(BaseModel):
    """Single review dimension block"""
    feedback: str = Field(description="Concrete, actionable guidance for this dimension")
    pass_: bool = Field(alias="pass", description="Whether this dimension passed review")


class InterfaceReviewBlock(BaseModel):
    """Full review section for interface design"""
    Feature_Alignment: InterfaceReviewCategory = Field(alias="Feature Alignment")
    Structural_Completeness: InterfaceReviewCategory = Field(alias="Structural Completeness")
    Docstring_Quality: InterfaceReviewCategory = Field(alias="Docstring Quality")
    Interface_Style_Granularity: InterfaceReviewCategory = Field(alias="Interface Style & Granularity")
    Scalability_Maintainability: InterfaceReviewCategory = Field(alias="Scalability & Maintainability")
    Data_Flow_Consistency: InterfaceReviewCategory = Field(alias="Data Flow Consistency")
    
    class Config:
        allow_population_by_field_name = True


class InterfaceReviewOutput(BaseModel):
    """Final JSON output for the interface review process"""
    review: InterfaceReviewBlock
    final_pass: bool = Field(description="True only if all passes or remaining issues are minor")


# ========================
# 2. AGENT TOOL
# ========================
class InterfaceDefinition(BaseModel):
    features: List[str] = Field(..., description="List of feature paths this interface handles")
    code: str = Field(..., description="Python code for the interface (functions/classes with docstrings)")

class DesignInterfacesToolParamModel(BaseModel):
    interfaces: List[InterfaceDefinition] = Field(
        ...,
        description="List of interface definitions, each covering specific feature paths",
    )

class DesignInterfacesTool(Tool):
    name = "design_itfs_for_feature"
    description = DESIGN_INTERFACE_TOOL
    ParamModel = DesignInterfacesToolParamModel
    
    @classmethod
    def custom_parse(cls, raw: str) -> List[ToolCallArguments]:
        """Parse raw JSON input for data flow"""
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
            interfaces = arguments.get("interfaces", [])
            
            # Access target features from environment if available
            tgt_features = getattr(env, "tgt_features", None)
            covered_features = set(getattr(env, "covered_features", set()))
            
            # Debug logging
            if hasattr(env, 'logger'):
                env.logger.debug(f"Tool execute: tgt_features={len(tgt_features) if tgt_features else 0}, covered_features={len(covered_features)}")
            
            if not interfaces:
                return ToolExecResult(
                    output=None,
                    error="Empty interfaces provided",
                    error_code=1,
                    state={}
                )

            feedbacks: List[str] = []
            parsed_units = []
            all_features: set = set()
            valid_interfaces: List[Dict] = []
            unit_features_map: Dict[str, List[str]] = {}  # unit_id -> features mapping

            for interface in interfaces:
                features = interface.get("features", [])
                code = interface.get("code", "")
                interface_errors: List[str] = []

                # ====== 校验 features ======
                if not features or not isinstance(features, list):
                    interface_errors.append(
                        f"Interface must have valid features list, got: {features}"
                    )
                else:
                    feature_set = set(features)

                    # Check if features are in target features
                    if tgt_features and not feature_set.issubset(tgt_features):
                        invalid_features = feature_set - set(tgt_features)
                        interface_errors.append(
                            f"Features {list(invalid_features)} are not in target features {list(tgt_features)}"
                        )

                    # Check for feature overlap with already accepted features
                    overlap = all_features & feature_set
                    if overlap:
                        interface_errors.append(
                            f"Features {list(overlap)} are already covered by another interface"
                        )
                if interface_errors:
                    feedbacks.extend(interface_errors)
                    continue

                feature_set = set(features)

                try:
                    ast.parse(code)
                except SyntaxError as e:
                    interface_errors.append(
                        f"Syntax error in interface for features {features}: "
                        f"line {e.lineno}, column {e.offset}: {e.msg}"
                    )
                    feedbacks.extend(interface_errors)
                    continue

                parsed_file = ParsedFile(code=code, file_path="temp_interface.py")
                # Only include class and function, exclude method for units_to_code
                interface_units = [
                    unit for unit in parsed_file.units
                    if unit.unit_type in ["function", "class"]
                ]

                if not interface_units:
                    interface_errors.append(
                        f"No valid functions/classes found for features {features}"
                    )
                    feedbacks.extend(interface_errors)
                    continue

                for unit in interface_units:
                    if not unit.docstring and unit.unit_type in ["function", "class"]:
                        interface_errors.append(
                            f"Missing docstring for {unit.unit_type} '{unit.name}' "
                            f"in features {features}"
                        )

                if interface_errors:
                    feedbacks.extend(interface_errors)
                    continue

                for unit in interface_units:
                    unit_key = f"{unit.unit_type}:{unit.name}:{unit.parent or ''}"
                    unit_features_map[unit_key] = features


                parsed_units.extend(interface_units)
                valid_interfaces.append(interface)
                all_features.update(feature_set)


            feedback_str = ""
            for feedback in feedbacks:
                if isinstance(feedback, str):
                    feedback_str += f"{feedback}\n"
                elif isinstance(feedback, list):
                    feedback_str += "\n".join(feedback) + "\n"
                else:
                    feedback_str += f"{str(feedback)}\n"
            
            # Calculate progress and provide specific guidance with complete context
            if not feedback_str.strip():
                total_features = len(tgt_features) if tgt_features else 0
                # Current batch features + previously covered features
                newly_covered = covered_features.union(set(all_features))
                remaining_features = (set(tgt_features) - newly_covered) if tgt_features else set()
                covered_count = len(newly_covered)
                
                if remaining_features:
                    feedback_str = (
                        f"SUCCESS: {len(valid_interfaces)} interface(s) successfully registered.\n"
                        f"Progress: {covered_count}/{total_features} features covered.\n"
                    )
                    feedback_str += (
                        f"\nPLANNING GUIDANCE:\n"
                        f"  - Continue to design interfaces in your next call\n"
                        f"  - Group logically related features (shared state/config/lifecycle)\n"
                        f"  - Each interface should have ONE clear responsibility\n"
                        f"  - Avoid cramming all features into one large class/function\n"
                        f"  - follow the instruction and output format in the system prompt\n"
                    )
                else:
                    feedback_str = (
                        f"SUCCESS: All {total_features} features have been covered!\n"
                        f"Interface design is complete. You can now proceed."
                    )
            else:
                # There were errors - provide specific guidance for re-planning with complete context
                feedback_str = feedback_str.strip()
                total_features = len(tgt_features) if tgt_features else 0
                # Only subtract actually covered features, not the failed batch
                remaining_features = (set(tgt_features) - covered_features) if tgt_features else set()
                
                feedback_str += (
                    f"\n\nRE-PLANNING NEEDED:\n"
                    f"Current status: {len(covered_features)}/{total_features} features covered.\n"
                    f"Please fix the issues above and re-design interfaces.\n\n"
                    f"PLANNING GUIDANCE:\n"
                    f"  - Review the error messages and fix each issue\n"
                    f"  - Plan interfaces for the remaining features\n"
                    f"  - Ensure each interface has ONE clear responsibility\n"
                    f"  - Double-check feature paths match the target list exactly\n"
                    f"  - follow the instruction and output format in the system prompt\n"
                )
            
            if not valid_interfaces:
                feedback_str += f"\n\nAll provided interfaces had issues - none were accepted."
                # Add format requirement if not already present
                feedback_str = feedback_str.strip()
                return ToolExecResult(
                    output=None,
                    error=feedback_str.strip(),
                    error_code=1,
                    state={}    
                )
                       
            return ToolExecResult(
                output=feedbacks,
                error_code=0,
                state={
                    "interfaces": valid_interfaces,
                    "parsed_units": parsed_units,
                    "covered_features": list(all_features),
                    "unit_features_map": unit_features_map
                }
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

class InterfaceEnv(ReviewEnv):
    """Environment for interface design phase"""
    
    def __init__(
        self,
        review_llm: LLMClient,
        review_format: BaseModel,
        review_memory: Memory,
        review_times: int = 1,
        register_tools: List = [],
        repo_skeleton: RepoSkeleton = None,
        repo_rpg: RPG = None,
        tgt_file_path: str = "",
        tgt_subgraph_name: str = "", 
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
        self.tgt_file_path = tgt_file_path
        self.tgt_subgraph_name = tgt_subgraph_name
        
        self.imports_units = []
        self.designed_interfaces = {}
        self.covered_features = set()
        self.is_completed = False
        self.logger = logger or logging.getLogger(__name__)
        
        self.tgt_features = []
        self.tgt_subtree_implemented = []
        self.all_implemented_subtrees = {}
    
    def post_feedback(self) -> str:
        """Provide complete context information as post-feedback after tool execution"""

        # Check if there are remaining features or if we need context
        if not self.tgt_features:
            self.logger.debug("post_feedback: No target features, returning empty")
            return ""
            
        covered_features = getattr(self, 'covered_features', set())
        remaining_features = set(self.tgt_features) - covered_features
        
        self.logger.info(f"post_feedback: {len(covered_features)}/{len(self.tgt_features)} features covered, {len(remaining_features)} remaining")
        
        # Only provide context if there are remaining features
        if not remaining_features:
            self.logger.info("post_feedback: All features covered, no context needed")
            return ""
        
        self.logger.debug(f"post_feedback: Building complete context for {len(remaining_features)} remaining features")
        context_info = self.build_complete_context_info()
        self.logger.info(f"[Post Feedback]: COMPLETE CONTEXT FOR NEXT ITERATION:\n{context_info}")
        
        return f"\n\nCOMPLETE CONTEXT FOR NEXT ITERATION:\n{context_info}"
    
    def build_complete_context_info(self) -> str:
        """Build complete context information including current implementation status"""
        if not self.repo_rpg:
            return ""
            
        # Calculate remaining features
        covered_features = getattr(self, 'covered_features', set())
        remaining_features = set(self.tgt_features) - covered_features
        
        # Current file code
        current_code = ""
        if hasattr(self, 'designed_interfaces') and self.designed_interfaces:
            imports_units = getattr(self, 'imports_units', [])
            interface_units = [info['unit'] for info in self.designed_interfaces.values()]
            current_code = render_codeunits_as_file(units=imports_units + interface_units)
        
        # Prepare all context information
        repo_info = self.repo_rpg.repo_info or "Repository information not available"
        data_flow_str = json.dumps(self.repo_rpg.data_flow, indent=2) if self.repo_rpg.data_flow else "No data flow available"
        
        # Format upstream context (reuse existing logic from agent)
        from zerorepo.utils.compress import get_skeleton
        upstream_context = self._extract_upstream_context(
            self.repo_rpg.data_flow or [], 
            self.tgt_subgraph_name, 
            self.all_implemented_subtrees
        )
        
        # Format implemented summary
        implemented_summary = self._format_implemented_summary(
            self.tgt_subtree_implemented, 
            self.tgt_file_path
        )
        
        # Format base classes
        base_classes_str = _format_base_classes(self.repo_rpg.base_classes)
        
        # Format features lists
        all_features_str = '\n'.join([f"- {f}" for f in self.tgt_features])
        remaining_features_str = '\n'.join([f"- {f}" for f in remaining_features]) if remaining_features else "All features covered!"
        
        # Current code section
        current_code_section = ""
        if current_code.strip():
            current_code_section = (
                "=== Current File Implementation Status ===\n"
                f"Current implementation of `{self.tgt_file_path}` after previous iterations:\n"
                f"```python\n{current_code}\n```\n\n"
            )
        
        # Build comprehensive context
        context_info = (
            f"{repo_info}\n\n"
            "=== Data Flow Graph ===\n"
            f"{data_flow_str}\n\n"
            "=== Upstream Context ===\n"
            f"{upstream_context}\n\n"
            "=== Implemented Summary ===\n"
            f"{implemented_summary}\n\n"
            "=== Available Base Classes ===\n"
            f"{base_classes_str}\n\n"
            f"{current_code_section}"
            "=== Remaining Features To Design ===\n"
            f"{remaining_features_str}\n\n"
            "Continue designing interfaces for the REMAINING features listed above. "
            "Use the complete context information provided to make informed design decisions."
        )
        
        return context_info
    
    def _extract_upstream_context(self, data_flow_graph, tgt_subgraph, all_implemented_subtrees, top_n=5):
        """Extract upstream interface context"""
        upstream_nodes = set()
        
        # Find upstream dependencies
        for edge in data_flow_graph:
            from_node = edge.get("source")
            to_node = edge.get("target")
            if to_node and to_node.startswith(tgt_subgraph):
                upstream_nodes.add(from_node)
        
        if not upstream_nodes:
            return "No upstream modules connected to this subtree."
        
        # Format upstream context
        upstream_summary = []
        for node in sorted(upstream_nodes):
            impl_files = all_implemented_subtrees.get(node, [])
            if not isinstance(impl_files, list):
                continue
            impl_files = impl_files[:top_n]
            if not impl_files:
                continue
            for file_node in impl_files:
                path = file_node.path
                features = ', '.join(file_node.feature_paths)
                code_skeleton = get_skeleton(file_node.code, keep_indent=True)
                upstream_summary.append(
                    f"### From module: `{node}`\nFile: `{path}`\nFeatures: {features}\nCode:\n```python\n{code_skeleton}\n```\n"
                )
            upstream_summary.append('...some files have been omitted due to context length constraints...')

        
        return "\n".join(upstream_summary) if upstream_summary else "No implemented upstream interfaces found."
    
    def _format_implemented_summary(self, tgt_subtree_implemented, current_file, latest_k=5):
        """Format summary of already implemented files in subtree"""
        if not tgt_subtree_implemented:
            return "No files implemented yet in this subtree."
        
        summary_lines = []
        for impl_file in tgt_subtree_implemented[-latest_k:]:
            if impl_file.path != current_file:
                features_str = ', '.join(impl_file.feature_paths[:2])
                if len(impl_file.feature_paths) > 2:
                    features_str += f", ...{len(impl_file.feature_paths) - 2} more"
                
                skeleton = get_skeleton(raw_code=impl_file.code, keep_indent=True)
                block = (
                    f"#### Related Implemented File: `{impl_file.path}`\n"
                    f"**Associated Features:** {features_str}\n"
                    f"**Code Skeleton:**\n"
                    f"```python\n{skeleton}\n```\n"
                )
                summary_lines.append(block)
        
        return "\n".join(summary_lines)
    
    def load_review_message(self, tool_calls: List[ToolCall]):
        """Load review message for interface design"""
        if not self.repo_rpg or not self.repo_skeleton:
            return "Missing repository context for review"
        
        repo_info = self.repo_rpg.repo_info
        # skeleton_str = self.repo_skeleton.to_tree_string()
        data_flow = self.repo_rpg.data_flow
        base_classes = _format_base_classes(self.repo_rpg.base_classes)
        data_flow_str = json.dumps(data_flow, indent=2) if data_flow else "No data flow available"
        
        # Extract interface information from tool calls
        interfaces_info = []
        for tool_call in tool_calls:
            if hasattr(tool_call, 'arguments') and 'interfaces' in tool_call.arguments:
                interfaces = tool_call.arguments['interfaces']
                for interface in interfaces:
                    features = interface.get('features', [])
                    code = interface.get('code', '')
                    interfaces_info.append(f"Features: {features}\nCode:\n{code}\n")
        
        interfaces_str = "\n---\n".join(interfaces_info) if interfaces_info else "No interfaces found"
        
        review_input = (
            "Please review the following interface design:\n\n"
            f"Repository Info: {repo_info}\n\n"
            # f"Repository Skeleton:\n{skeleton_str}\n\n"
            f"Data Flow:\n{data_flow_str}\n\n"
            f"Base Classes:\n{base_classes}"
            f"Designed Interfaces:\n{interfaces_str}\n\n"
            "Evaluate the interfaces according to the review criteria and provide detailed feedback.\n"
            "Respond with a valid JSON object following the required output format."
        )
        
        return review_input
    
    def should_terminate(self, tool_calls, tool_results):
        """Check if interface design is complete"""
        # Don't update state here - that's handled in update_env
        # Just check if all target features are covered
        if not self.tgt_features:
            return False
            
        tgt_features_set = set(self.tgt_features)
        covered_features_set = set(self.covered_features)
        
        # Debug logging
        self.logger.debug(f"should_terminate: tgt_features={len(tgt_features_set)}, covered_features={len(covered_features_set)}")
        self.logger.debug(f"Missing features: {tgt_features_set - covered_features_set}")
        
        # Check if all target features are covered
        self.is_completed = tgt_features_set.issubset(covered_features_set)
        
        if self.is_completed:
            self.logger.info(f"All {len(self.tgt_features)} target features have been covered. Interface design complete.")
        else:
            remaining = len(tgt_features_set - covered_features_set)
            self.logger.debug(f"Still need to cover {remaining} features out of {len(self.tgt_features)} total.")
        
        return self.is_completed
    
    def update_env(self, tool_call, result: ToolResult):
        """Update environment state after tool execution"""
        if tool_call.name == "design_itfs_for_feature" and result.success:
            try:
                valid_interfaces = result.state.get("interfaces", [])
                parsed_units = result.state.get("parsed_units", [])
                unit_features_map = result.state.get("unit_features_map", {})
                covered_features = result.state.get("covered_features", [])
                
                # Extract imports from valid interfaces for import handling
                for v_itf in valid_interfaces:
                    code = v_itf.get("code", "")
                    temp_units = ParsedFile(code=code, file_path="temp_interface.py").units
                    self.imports_units.extend([unit for unit in temp_units if unit.unit_type in ["import", "assignment"]])
                
                # Update covered features (ensure it's a set)
                if isinstance(covered_features, list):
                    self.covered_features.update(covered_features)
                    self.logger.debug(f"Updated covered_features: added {len(covered_features)} features, total now {len(self.covered_features)}")
                else:
                    self.covered_features.update(covered_features or [])
                    self.logger.debug(f"Updated covered_features from set/other: total now {len(self.covered_features)}")
                
                # Store interface information and update RPG
                for unit in parsed_units:
                    # Only include class and function units, exclude methods
                    if unit.unit_type in ["class", "function"]:
                        unit_key = f"{unit.unit_type}:{unit.name}:{unit.parent or ''}"
                        features = unit_features_map.get(unit_key, [])
                        interface_key = f"{unit.unit_type} {unit.name}"
                        self.designed_interfaces[interface_key] = {
                            'unit': unit,
                            'features': features
                        }
                        
                        # Debug logging
                        self.logger.debug(f"Interface {unit.name} ({unit.unit_type}) has features: {features}")
                        
                        # Update RPG with interface metadata
                        self._update_rpg_metadata(unit, features)

                self.logger.info(f"Updated interfaces: {len(self.designed_interfaces)} total, "
                               f"covered features: {len(self.covered_features)}/{len(self.tgt_features)}")
                
            except Exception as e:
                self.logger.error(f"Failed to update interface environment: {e}")
    
    def _update_rpg_metadata(self, unit, features):
        """Update RPG node metadata with interface information"""
    
        if not self.repo_rpg:
            return
        
        if not unit.unit_type in ["function", "class"]:
            return 
        
        for feature_path in features:
            try:
                self.logger.debug(f"Processing feature path: {feature_path}")
                
                # Add subgraph prefix if not already present
                if not feature_path.startswith(self.tgt_subgraph_name + "/"):
                    feature_path = self.tgt_subgraph_name + "/" + feature_path 
                
                self.logger.debug(f"Final feature path: {feature_path}")
                
                # Find the feature node in RPG
                node = self.repo_rpg.get_node_by_feature_path(feature_path)
                
                self.logger.debug(f"Found node: {node.name if node else 'None'}")

                
                if node:
                    node_type = None
                    if unit.unit_type == "function":
                        node_type = NodeType.FUNCTION
                    elif unit.unit_type == "class":
                        node_type = NodeType.CLASS
                    
                    # Ensure node has metadata object
                    if node.meta is None:
                        node.meta = NodeMetaData()
                    
                    if node.meta.type_name == None:
                        node.meta.type_name = node_type
                    new_path = f"{self.tgt_file_path}:{unit.name}"
                    if node.meta.path == None:
                        node.meta.path = new_path
                    else:
                        if isinstance(node.meta.path, str):
                            if node.meta.path != new_path:
                                node.meta.path = [node.meta.path, new_path]
                        else:
                            if new_path not in node.meta.path:
                                node.meta.path.append(new_path)
                            
                    self.logger.info(f"Updated RPG node {node.name} with interface {unit.name}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to update RPG metadata for feature {feature_path}: {e}")


# ========================
# 4. AGENT
# ========================
class InterfaceAgent(AgentwithReview):
    """Agent responsible for designing component interfaces"""
    
    def __init__(
        self,
        llm_cfg: Union[str, Dict, LLMConfig],
        design_system_prompt: str = DESIGN_ITF,
        design_context_window: int = 30,
        review_system_prompt: str = DESIGN_ITF_REVIEW,
        review_context_window: int = 10,
        max_review_times: int = 1,
        register_tools: Optional[List[Tool]] = None,
        logger: Optional[logging.Logger] = None,
    
        file_node: FileNode=None,
        tgt_subgraph_name: str=None,
        repo_skeleton: RepoSkeleton = None,
        repo_rpg: RPG = None,
        **kwargs
    ):
        if register_tools is None:
            register_tools = [DesignInterfacesTool()]
        
        super().__init__(
            llm_cfg=llm_cfg,
            design_system_prompt=design_system_prompt,
            design_context_window=design_context_window,
            review_system_prompt=review_system_prompt,
            review_format=InterfaceReviewOutput,
            max_review_times=max_review_times,
            register_tools=register_tools,
            logger=logger,
            review_context_window=review_context_window,
            **kwargs
        )
        
        self.tgt_file_node = file_node
        self.tgt_file_path = file_node.path if file_node else ""
        self.tgt_subgraph_name = tgt_subgraph_name
        self.repo_skeleton = repo_skeleton
        self.repo_rpg = repo_rpg
        
        # Create custom environment
        self._env = InterfaceEnv(
            review_llm=self.review_llm,
            review_format=InterfaceReviewOutput,
            review_memory=self.review_memory,
            review_times=max_review_times,
            register_tools=register_tools,
            tgt_file_path=file_node.path if file_node else "",
            tgt_subgraph_name=tgt_subgraph_name if tgt_subgraph_name else "",
            repo_skeleton=repo_skeleton,
            repo_rpg=repo_rpg,
            logger=logger
        )
    
    def design_file_interface(
        self,
        tgt_subtree_implemented: List[FileNode],
        all_implemented_subtrees: Dict[str, List[FileNode]],
        max_steps: int = 10,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Design interface for a specific file, following the original InterfacesDesigner pattern
        
        Args:
            file_node: File to design interfaces for
            data_flow_graph: Component data flow relationships
            tgt_subtree: Target subtree containing the file
            tgt_subtree_implemented: Already implemented files in subtree
            all_implemented_subtrees: All implemented subtrees
            base_classes: Available base classes
            max_steps: Maximum design iterations
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with designed interfaces, code, and metadata
        """
        file_node = self.tgt_file_node
        tgt_subgraph_name = self.tgt_subgraph_name
        
        self.logger.info(f"[InterfaceAgent] Designing interface for {file_node.path}")
        
        file_path = file_node.path
        file_features = set(file_node.feature_paths)
        
        self._env.tgt_features = file_features
        # Store context information in environment for dynamic injection
        self._env.tgt_subtree_implemented = tgt_subtree_implemented
        self._env.all_implemented_subtrees = all_implemented_subtrees
        
        file_features_str = '\n'.join([f"- {f}" for f in file_features])
        
        # Prepare context information
        repo_info = self.repo_rpg.repo_info if self.repo_rpg else "Repository information not available"
        # skeleton_info = self.repo_skeleton.to_tree_string() if self.repo_skeleton else "Skeleton not available"
        
        # Format data flow
        data_flow_str = json.dumps(self.repo_rpg.data_flow, indent=2)
        
        # Format upstream context
        upstream_context = self._extract_upstream_context(
            self.repo_rpg.data_flow if self.repo_rpg else [], 
            tgt_subgraph_name, 
            all_implemented_subtrees
        )
        
        # Format implemented summary
        implemented_summary = self._format_implemented_summary(
            tgt_subtree_implemented, file_node.path
        )
        
        # Format base classes
        base_classes_str = _format_base_classes(self.repo_rpg.base_classes)
        
        # Build task description
        task = (
            "[Begin Iteration]\n"
            f"Design interfaces for file: `{file_node.path}`.\n\n"
            "Requirements:\n"
            "- ONLY cover the following feature paths:\n"
            f"{file_features_str}\n"
            "- When calling `design_itfs_for_feature`, ONLY use feature paths listed above.\n"
            "- Do NOT introduce new/unspecified feature paths.\n"
            "- Define interfaces only (imports + signature + docstring + `pass`).\n\n"
            "**Interface Planning Strategy:**\n"
            "- **Design interfaces per call**: Create several focused interfaces in each tool call.\n"
            "- **Avoid large monolithic interfaces**: Don't put all features in one big class or function.\n"
            "- **Group logically related features**: Only group features that share state, configuration, or lifecycle.\n"
            "- **One responsibility per interface**: Each interface should have a single, clear purpose.\n"
            "- **Iterative approach**: After each successful call, the tool will show your progress and suggest next steps.\n\n"
            "- You MAY import and reuse symbols (classes, functions, data structures) described in "
            "`Upstream Context`, `Implemented Summary`, and `Available Base Classes` via normal Python imports.\n\n"
            "Global context you can use:\n"
            "=== Repository Info ===\n"
            f"{repo_info}\n\n"
            "=== Data Flow Graph ===\n"
            f"{data_flow_str}\n\n"
            "=== Upstream Context ===\n"  # Interface information across functional areas
            f"{upstream_context}\n\n"
            "=== Implemented Summary ===\n"  # Interface information already planned within the current area
            f"{implemented_summary}\n\n"
            "=== Available Base Classes ===\n"  # Base classes and fundamental data structures
            f"{base_classes_str}\n"
            "Please follow the instructions and output format in the system prompt carefully and design the interfaces accordingly.\n"
        )
        
        # Execute the design process
        result = self.run(
            task=task,
            max_error_times=max_retries,
            max_steps=max_steps
        )
        
        # Process results
        designed_interfaces = self._env.designed_interfaces
        covered_features = self._env.covered_features
        
        # Generate final code
        final_code = render_codeunits_as_file(
            units=self._env.imports_units + [info['unit'] for info in designed_interfaces.values()]
        )
        
        # Create feature interface map
        feature_interface_map = self._create_feature_interface_map(designed_interfaces)
        
        # Log feature map like original implementation
        self.logger.info(f"Feature Map for file `{file_node.path}`: {json.dumps(feature_interface_map, indent=4)}")
        self.logger.info(f"[InterfaceAgent] Designed {len(designed_interfaces)} interfaces for {file_node.path}")
        
        return {
            "code": final_code,
            "feature_interface_map": feature_interface_map,
            "designed_interfaces": designed_interfaces,
            "update_repo_rpg": self._env.repo_rpg,
            "covered_features": list(covered_features),
            "success": self._env.is_completed,
            "agent_results": result
        }
    
    def _extract_upstream_context(
        self, 
        data_flow_graph: List[Dict],
        tgt_subgraph: str,
        all_implemented_subtrees: Dict[str, List[FileNode]],
        top_n: int=5
    ) -> str:
        """Extract upstream interface context"""
        upstream_nodes = set()
        
        # Find upstream dependencies
        for edge in data_flow_graph:
            from_node = edge.get("source")
            to_node = edge.get("target")
            if to_node and to_node.startswith(tgt_subgraph):
                upstream_nodes.add(from_node)
        
        if not upstream_nodes:
            return "No upstream modules connected to this subtree."
        
        # Format upstream context
        upstream_summary = []
        for node in sorted(upstream_nodes):
            impl_files = all_implemented_subtrees.get(node, [])
            if not isinstance(impl_files, list):
                logging.warning(f"Expected list of FileNode, but got {type(impl_files)} for node={node}")
                continue
            impl_files = impl_files[:top_n]
            if not impl_files:
                continue
            for file_node in impl_files:
                path = file_node.path
                features = ', '.join(file_node.feature_paths)
                code_skeleton = get_skeleton(file_node.code, keep_indent=True)
                upstream_summary.append(
                    f"### From module: `{node}`\nFile: `{path}`\nFeatures: {features}\nCode:\n```python\n{code_skeleton}\n```\n"
                )
            upstream_summary.append('...some files have been omitted due to context length constraints...')

        
        return "\n".join(upstream_summary) if upstream_summary else "No implemented upstream interfaces found."
    
    def _format_implemented_summary(
        self, 
        tgt_subtree_implemented: List[FileNode], 
        current_file: str,
        latest_k: int =5
    ) -> str:
        """Format summary of already implemented files in subtree"""
        if not tgt_subtree_implemented:
            return "No files implemented yet in this subtree."
        
        summary_lines = []
        for impl_file in tgt_subtree_implemented[-latest_k:]:
            if impl_file.path != current_file:
                features_str = ', '.join(impl_file.feature_paths[:2])
                if len(impl_file.feature_paths) > 2:
                    features_str += f", ...{len(impl_file.feature_paths) - 2} more"
                
                skeleton = get_skeleton(raw_code=impl_file.code, keep_indent=True)
                block = (
                    f"#### Related Implemented File: `{impl_file.path}`\n"
                    f"**Associated Features:** {features_str}\n"
                    f"**Code Skeleton:**\n"
                    f"```python\n{skeleton}\n```\n"
                )
                summary_lines.append(block)
        
        return "\n".join(summary_lines)
    
    def _create_feature_interface_map(self, designed_interfaces: Dict) -> Dict[str, List[str]]:
        """
        Create mapping from interface key to features, following the original implementation logic
        
        Args:
            designed_interfaces: Dict with interface information
            
        Returns:
            Dict mapping interface key (e.g. "function login_user") to list of feature paths
        """
        feature_interface_map = {}
        
        for interface_key, interface_info in designed_interfaces.items():
            unit = interface_info['unit']
            features = interface_info.get('features', [])
            
            # Only include class and function interfaces, following original logic
            if unit.unit_type in ["class", "function"]:
                # Create key in format "{unit_type} {unit_name}" consistent with interface_key
                map_key = f"{unit.unit_type} {unit.name}"
                feature_interface_map[map_key] = features
        
        return feature_interface_map
    
