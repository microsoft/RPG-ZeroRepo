"""
Feature Tree Refactoring Agent

Simplified two-step workflow to refactor feature trees into subgraphs:
1. Step 1: Subtree Planning - determines subtree names and count
2. Step 2: Feature Organization - reorganizes features into planned subtrees
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Type, Union
from pydantic import BaseModel, Field
from copy import deepcopy
from zerorepo.rpg_gen.base.llm_client import (
    LLMClient, LLMConfig, Memory,
    SystemMessage, UserMessage, AssistantMessage
)

from zerorepo.utils.tree import (
    get_all_leaf_paths,
    convert_leaves_to_list,
    apply_changes,
    extract_leaf_nodes,
    find_leaf_paths_by_node,
    filter_tree_by_leaf_nodes,
    remove_paths,
    pre_order_traversal_to_list
)

from zerorepo.config.checkpoint_config import CheckpointManager

from .prompt import (
    PROMPT_TEMPLATE_SUBTREE_PLANNING,
    PROMPT_TEMPLATE_FEATURE_ORGANIZATION
)


# Pydantic models for structured outputs
class SubtreePlan(BaseModel):
    """Schema for a single subtree plan"""
    name: str = Field(description="Name of the subtree/component")
    purpose: str = Field(description="High-level purpose or theme of this subtree")
    estimate_size: int=Field(description="Estimize of each subtree's size")
    
class SubtreePlanningOutput(BaseModel):
    """Output schema for subtree planning step"""
    total_subtrees: int = Field(description="Total number of planned subtrees")
    subtree_plans: List[SubtreePlan] = Field(description="List of planned subtrees")
    reasoning: str = Field(description="Reasoning for the subtree organization")


class FeatureAssignment(BaseModel):
    """Schema for assigning features to a subtree"""
    subtree_name: str = Field(description="Name of the target subtree")
    refactored_paths: List[str] = Field(description="List of refactored feature paths assigned to this subtree")

class FeatureOrganizationOutput(BaseModel):
    """Output schema for feature organization step"""
    assignments: List[FeatureAssignment] = Field(description="Feature assignments for each subtree")

class RefactoredSubtree(BaseModel):
    """Final refactored subtree structure"""
    name: str
    purpose: str
    feature_tree: Dict[str, Any]
    feature_paths: List[str]
    feature_count: int

class FeatureRefactorAgent:
    """
    Simplified feature tree refactoring agent using a two-step workflow.
    """
    
    def __init__(
        self,
        llm_cfg: LLMConfig,
        refactor_context_window: int = 3,
        refactor_max_iterations: int = 20,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the refactoring agent.
        
        Args:
            llm_cfg: LLM configuration
            context_window: Context window for memory management
            logger: Logger instance
        """
        self.llm_client = LLMClient(llm_cfg)
        self.context_window = refactor_context_window
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_manager = checkpoint_manager
        
        # Memory for the two-step process
        self.planning_memory = Memory(context_window=self.context_window)
        self.organization_memory = Memory(context_window=self.context_window)
        
        # Results storage
        self.subtree_plans: List[SubtreePlan] = []
        self.refactored_subtrees: List[RefactoredSubtree] = []
        self.unassigned_paths: List[str] = []
        self.max_iterations = refactor_max_iterations
    
    def _call_llm_structured(
        self,
        memory: Memory,
        prompt: str,
        output_schema: Type[BaseModel],
        max_retries: int = 3
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Call LLM with structured output and handle retries"""
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
                    parsed_result = output_schema(**parsed_result)
                    memory.add_message(AssistantMessage(str(response)))
                    return parsed_result, response
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} LLM call attempts failed")
        
        return None, ""
    
    def process_refactored_subtree(self, subtree: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
        """Process and validate subtree using leaf-node based approach"""
        if not hasattr(self, 'available_leaf_nodes'):
            self.available_leaf_nodes = extract_leaf_nodes(self.current_feature_tree)
            
        subtree_leaf_nodes = extract_leaf_nodes(subtree)
        
        # Filter to only real leaf nodes that exist in the feature tree
        real_leaf_nodes = list(set(self.available_leaf_nodes) & set(subtree_leaf_nodes))
        # Identify fabricated leaf nodes
        img_leaf_nodes = list(set(subtree_leaf_nodes) - set(self.available_leaf_nodes))

        self.logger.info(f"Imagine Rate: {len(img_leaf_nodes)/max(len(subtree_leaf_nodes), 1): .2%}")
        
        # Clean the subtree by filtering with real leaf nodes
        cleaned_sub_tree = filter_tree_by_leaf_nodes(subtree, target_leaf_names=real_leaf_nodes)

        if len(real_leaf_nodes) <= 20:
            msg = (
                "Subtrees with 20 or fewer valid leaf nodes are too small for `add_subtree`. "
                "If this is an extension of an existing area, please use `edit_subtree` instead"  
            )
            return {}, False, msg
        
        if not cleaned_sub_tree:
            img_leaf_nodes_str = ', '.join(img_leaf_nodes)
            msg = (
                "The provided subtree could not be processed because none of its leaf nodes are valid.\\n"
                "The following leaf nodes were not found in the feature tree and appear to be fabricated:\\n"
                f"{img_leaf_nodes_str}\\n"
                "Please use only existing leaf nodes from the feature tree when defining a cluster.\\n"
                "Arbitrary or non-existent nodes cannot be used."
            )
            return {}, False, msg
                    
        # Get all leaf paths and validate structure
        all_leaf_paths = get_all_leaf_paths(cleaned_sub_tree)
        filter_leaf_paths = list(filter(lambda path: len(path.split("/")) == 4, all_leaf_paths))
        
        if len(filter_leaf_paths) == 0:
            leaf_paths_str = "\\n".join(all_leaf_paths)
            msg = (
                "The cleaned subtree does not contain any valid leaf paths with a depth of 4.\\n"
                "Each leaf path must follow the format: 'Level1/Level2/Level3/Feature'.\\n"
                f"Current leaf paths:\\n{leaf_paths_str}\\n"
                "Please ensure your subtree includes paths that match this structure."
            )
            return {}, False, msg
        
        # Rebuild subtree from valid paths
        cleaned_sub_tree = apply_changes({}, filter_leaf_paths)
        cleaned_sub_tree = convert_leaves_to_list(cleaned_sub_tree)
        
        return cleaned_sub_tree, True, ""
    
    def calculate_current_utilization(self, all_subtrees=None) -> float:
        """Calculate current utilization based on assigned leaf nodes"""
        if not hasattr(self, 'feature_tree_len') or self.feature_tree_len == 0:
            return 0.0

        if not hasattr(self, 'assigned_leaf_nodes') or not hasattr(self, 'current_feature_tree'):
            return 0.0

        # Use assigned_leaf_nodes to find original paths and calculate utilization
        assigned_leaf_nodes_list = list(self.assigned_leaf_nodes)
        if not assigned_leaf_nodes_list:
            return 0.0

        # Find original paths corresponding to assigned leaf nodes
        selected_feature_paths = find_leaf_paths_by_node(
            self.current_feature_tree,
            target_leaf_names=assigned_leaf_nodes_list
        )

        selected_feature_paths = list(set(selected_feature_paths))
        filter_feature_tree = remove_paths(self.current_feature_tree, selected_feature_paths, inplace=False)

        # Calculate utilization as in refactored_iteration.py
        util_percent = 1 - len(set(pre_order_traversal_to_list(filter_feature_tree))) / self.feature_tree_len
        return util_percent
    
    def get_remaining_leaf_count(self) -> int:
        """Get count of remaining unassigned leaf nodes"""
        if not hasattr(self, 'available_leaf_nodes') or not hasattr(self, 'assigned_leaf_nodes'):
            return 0
        return len(self.available_leaf_nodes) - len(self.assigned_leaf_nodes)
    
    def _build_repo_info(self, repo_data: Dict[str, Any]) -> str:
        """Format repository information for prompts"""
        repo_info = ""
        for key, value in repo_data.items():
            if key in ["repository_name", "repository_purpose", "scope", "main_purpose", "programming_language", "domain"]:
                readable_key = key.replace("_", " ").capitalize()
                repo_info += f"{readable_key}: {value}\n"
        return repo_info
    
    def step1_plan_subtrees(
        self,
        feature_tree: Dict[str, Any],
        repo_data: Dict[str, Any]
    ) -> Optional[SubtreePlanningOutput]:
        """
        Step 1: Plan the subtree structure and names.
        
        Args:
            feature_tree: The input feature tree to refactor
            repo_data: Repository metadata
            
        Returns:
            Subtree planning output or None if failed
        """
        self.logger.info("=== Step 1: Planning Subtrees ===")
        # Initialize planning memory with system prompt
        self.planning_memory.add_message(SystemMessage(PROMPT_TEMPLATE_SUBTREE_PLANNING))
        
        # Get all feature paths for analysis
        all_paths = get_all_leaf_paths(feature_tree)
        feature_count = len(all_paths)
        
        # Build prompt
        repo_info = self._build_repo_info(repo_data)
        feature_tree_json = json.dumps(feature_tree, indent=2)
        
        prompt = (
            "## Repository Information:\n"
            f"{repo_info}\n\n"
            "## Feature Tree to Refactor:\n"
            f"**Total Features**: {feature_count}\n\n"
            "```json\n"
            f"{feature_tree_json}\n"
            "```\n\n"
            "## Task:\n"
            "Analyze the feature tree and design a logical organization into functional subtrees/components.\n"
            "Each subtree should represent a coherent functional area or module.\n\n"
            "Provide your subtree planning with:\n"
            "1. Appropriate number of subtrees (typically 3-8 for good modularity)\n"
            "2. Clear names for each subtree\n"
            "3. Purpose/theme for each subtree\n"
            "4. Estimated feature count for each subtree\n\n"
            "Consider:\n"
            "- Functional cohesion (related features together)\n"
            "- Repository purpose and domain\n"
            "- Balanced subtree sizes\n"
            "- Clear separation of concerns"
        )
                
        planning_output, response = self._call_llm_structured(
            memory=self.planning_memory,
            prompt=prompt,
            output_schema=SubtreePlanningOutput
        )
        
        if planning_output:
            # Convert dict result to Pydantic models
            self.subtree_plans = planning_output.subtree_plans
            self.logger.info(f"Successfully planned {len(self.subtree_plans)} subtrees:")
            for i, plan in enumerate(self.subtree_plans):
                self.logger.info(f"  {i+1}. {plan.name}: {plan.purpose} (~{plan.estimate_size} features)")
            return planning_output
        else:
            self.logger.error("Failed to generate subtree planning")
            return None

    def step2_organize_features(
        self,
        feature_tree: Dict[str, Any],
        repo_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Step 2: Organize features into the planned subtrees using iterative process.
        Continues until all features are processed or max iterations reached.
        
        Args:
            feature_tree: The input feature tree
            repo_data: Repository metadata
 
        Returns:
            List of component subtrees (similar to Component key structure) or None if failed
        """
        self.logger.info("=== Step 2: Organizing Features (Iterative) ===")
        
        if not self.subtree_plans:
            self.logger.error("No subtree plans available. Run step1_plan_subtrees first.")
            return None
        
        # Initialize organization memory with system prompt
        self.organization_memory.add_message(SystemMessage(PROMPT_TEMPLATE_FEATURE_ORGANIZATION))
        
        # Initialize available leaf nodes and tracking for supervision
        self.current_feature_tree = feature_tree
        self.available_leaf_nodes = extract_leaf_nodes(feature_tree)
        self.assigned_leaf_nodes = set()
        
        # Calculate initial feature tree length (following refactored_iteration.py approach)
        self.feature_tree_len = len(set(pre_order_traversal_to_list(feature_tree)))
        
        # Get all feature paths that need to be organized
        all_paths = get_all_leaf_paths(feature_tree)
        remaining_paths = set(all_paths)
        
        self.logger.info(f"Total paths: {len(all_paths)}, Feature tree length: {self.feature_tree_len}")
        
        all_subtrees = [
            {
                "refactored_subtree": {},
                "name": subtree_plan.name,
                "purpose": subtree_plan.purpose,
                "estimate_size": subtree_plan.estimate_size
            } for subtree_plan in self.subtree_plans
        ]
        
        # Track assignments and progress
        all_assignments = []
        all_unassigned = []  # List to collect unassigned paths
        
        # Build subtree plans summary (for all iterations)
        plans_summary = ""
        for i, plan in enumerate(self.subtree_plans):
            plans_summary += f"{i+1}. **{plan.name}**: {plan.purpose}\n"
        
        repo_info = self._build_repo_info(repo_data)
        
        iteration = 0
        # Enhanced termination condition: check both paths and leaf node utilization
        while remaining_paths and iteration < self.max_iterations and self.calculate_current_utilization(all_subtrees) < 0.99:
            iteration += 1
            current_util = self.calculate_current_utilization(all_subtrees)
            remaining_leaf_nodes_count = self.get_remaining_leaf_count()
            self.logger.info(f"--- Organization Iteration {iteration}: {len(remaining_paths)} paths remaining, "
                           f"{remaining_leaf_nodes_count} leaf nodes unassigned, {current_util:.1%} utilization ---")
            
            # Calculate remaining leaf nodes for better utilization tracking
            remaining_leaf_nodes = set(self.available_leaf_nodes) - self.assigned_leaf_nodes
            
            # Build current feature tree from remaining paths
            current_tree = apply_changes({}, list(remaining_paths))
            current_tree = convert_leaves_to_list(current_tree)
            feature_tree_json = json.dumps(current_tree)
            
            # Show current assignments status
            assignments_status = json.dumps(all_subtrees)
                        
            prompt = (
                f"## Repository Context:\n{repo_info}\n\n"
                f"## Target Architecture (Planned Subtrees):\n{plans_summary}\n\n"
                f"## Progress (Iteration {iteration}):\n"
                f"- Remaining paths: {len(remaining_paths)}\n"
                f"- Remaining leaf nodes: {len(remaining_leaf_nodes)}\n"
                f"- Current utilization: {current_util:.1%}\n\n"
                f"## Already Assigned:\n```json\n{assignments_status}\n```\n\n"
                f"## Remaining Feature Tree (Source):\n"
                f"```json\n{feature_tree_json}\n```\n\n"
                "## Your Task\n"
                "Reorganize ALL remaining leaf nodes into the planned subtrees following these requirements:\n\n"
                "### Path Format (STRICT)\n"
                "Each path MUST have EXACTLY 4 segments: `Level1/Level2/Level3/LeafName`\n"
                "- Level1-3: Redesign hierarchy to match real code organization for this repo type\n"
                "- LeafName: Copy EXACTLY from remaining tree (character for character, no modifications)\n\n"
                "### Requirements\n"
                "1. Assign as many leaves as possible in this iteration\n"
                "2. Redesign the hierarchy - do NOT copy original paths\n"
                "3. Group related features by functional cohesion\n"
                "4. Use concrete module names (avoid `utils`, `helpers`, `misc`)\n\n"
                "### Example\n"
                "Original: `parsers/url/query/parse_query_string`\n"
                "- GOOD: `http/request/url_handling/parse_query_string` (redesigned, exact leaf)\n"
                "- BAD: `parsers/url/query/parse_query_string` (copied original)\n"
                "- BAD: `http/request/parse_query_string` (only 3 segments)\n"
                "- BAD: `http/request/url_handling/parseQueryString` (leaf modified)\n\n"
                "### Goal\n"
                f"Reorganize all {len(remaining_leaf_nodes)} remaining leaves with valid 4-segment paths."
            )
            org_output, response = self._call_llm_structured(
                memory=self.organization_memory,
                prompt=prompt,
                output_schema=FeatureOrganizationOutput
            )
            
            if not org_output:
                self.logger.warning(f"Failed to organize features at iteration {iteration}, skipping to next iteration...")
                continue
            
            # Process assignments from this iteration
            iteration_assigned = set()
            iteration_assigned_leaf_nodes = set()  # Track leaf nodes assigned this iteration

            # Calculate remaining leaf nodes from current remaining_paths
            remaining_tree = apply_changes({}, list(remaining_paths))
            remaining_leaf_nodes_set = set(extract_leaf_nodes(remaining_tree))

            for assignment in org_output.assignments:
                # Filter refactored_paths: keep only depth-4 paths whose leaf nodes exist in remaining feature tree
                valid_paths = []
                invalid_paths = []
                for p in assignment.refactored_paths:
                    if len(p.split("/")) == 4:  # Only consider depth-4 paths
                        leaf_name = p.split("/")[-1]  # Get the leaf node name
                        if leaf_name in remaining_leaf_nodes_set:
                            valid_paths.append(p)
                        else:
                            invalid_paths.append(p)
                    else:
                        invalid_paths.append(p)

                if invalid_paths:
                    self.logger.info(f"  Subtree '{assignment.subtree_name}': Rejected {len(invalid_paths)} invalid paths (leaf not in remaining tree or wrong depth)")

                if valid_paths:
                    # Track assigned leaf nodes for utilization calculation
                    assigned_leaf_nodes_this_batch = set(p.split("/")[-1] for p in valid_paths)
                    self.assigned_leaf_nodes.update(assigned_leaf_nodes_this_batch)
                    iteration_assigned_leaf_nodes.update(assigned_leaf_nodes_this_batch)

                    # Find existing assignment for this subtree or create new
                    existing_assignment = next(
                        (a for a in all_assignments if a.subtree_name == assignment.subtree_name),
                        None
                    )
                    if existing_assignment:
                        existing_assignment.refactored_paths.extend(valid_paths)
                    else:
                        all_assignments.append(FeatureAssignment(
                            subtree_name=assignment.subtree_name,
                            refactored_paths=valid_paths
                        ))

                    # Update the corresponding subtree in all_subtrees
                    for subtree in all_subtrees:
                        if subtree["name"] == assignment.subtree_name:
                            # Build refactored_subtree from assigned paths
                            current_paths = get_all_leaf_paths(subtree["refactored_subtree"])
                            current_paths.extend(valid_paths)
                            subtree["refactored_subtree"] = apply_changes({}, current_paths)
                            subtree["refactored_subtree"] = convert_leaves_to_list(subtree["refactored_subtree"])
                            break

                    iteration_assigned.update(valid_paths)
                    self.logger.info(f"  Assigned {len(valid_paths)} paths ({len(assigned_leaf_nodes_this_batch)} leaf nodes) to '{assignment.subtree_name}'")
            
            # Update remaining paths based on verified leaf nodes from this iteration
            if iteration_assigned:
                # Use the verified leaf nodes from this iteration to find and remove original paths
                # This ensures we only remove paths for leaf nodes that were actually validated
                paths_to_remove = find_leaf_paths_by_node(
                    self.current_feature_tree,
                    target_leaf_names=list(iteration_assigned_leaf_nodes)
                )

                if paths_to_remove:
                    # Remove the original paths corresponding to assigned leaf nodes
                    remaining_paths = remaining_paths - set(paths_to_remove)
                    self.logger.info(f"  Processed {len(iteration_assigned)} refactored paths, removed {len(paths_to_remove)} original paths, {len(remaining_paths)} remaining")
                else:
                    self.logger.warning(f"  Processed {len(iteration_assigned)} paths but no original paths found to remove")
            else:
                self.logger.warning(f"No progress made in iteration {iteration}, skipping to next iteration...")
                continue
        
        # Handle any remaining paths as unassigned
        if remaining_paths:
            all_unassigned.extend(list(remaining_paths))
            self.logger.warning(f"  {len(remaining_paths)} paths remain unassigned after {iteration} iterations")
        
        # Calculate utilization percentages based on leaf nodes instead of paths
        total_leaf_nodes = len(self.available_leaf_nodes)
        for subtree in all_subtrees:
            subtree_leaf_nodes = extract_leaf_nodes(subtree["refactored_subtree"])
            subtree["util_percent"] = len(subtree_leaf_nodes) / total_leaf_nodes if total_leaf_nodes > 0 else 0.0
            subtree["actual_size"] = len(subtree_leaf_nodes)
        
        # Log final summary with leaf node information
        total_assigned_leaf_nodes = sum(subtree["actual_size"] for subtree in all_subtrees)
        final_utilization = self.calculate_current_utilization(all_subtrees)
        self.logger.info(f"Organization completed after {iteration} iterations:")
        self.logger.info(f"  Total assigned: {total_assigned_leaf_nodes} leaf nodes ({final_utilization:.1%})")
        self.logger.info(f"  Total unassigned: {len(remaining_paths)} paths, {self.get_remaining_leaf_count()} leaf nodes")
        for subtree in all_subtrees:
            self.logger.info(f"    {subtree['name']}: {subtree['actual_size']} leaf nodes ({subtree['util_percent']:.1%} util)")
        
        # Filter out components with empty refactored_subtree before returning
        filtered_subtrees = []
        for subtree in all_subtrees:
            if subtree.get("refactored_subtree") and subtree["refactored_subtree"]:
                filtered_subtrees.append(subtree)
            else:
                self.logger.info(f"Removing empty component: {subtree['name']}")
        
        # Return filtered subtrees (similar to Component structure)
        return filtered_subtrees
    
    
    def refactor_feature_tree(
        self,
        feature_tree: Dict[str, Any],
        repo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete two-step refactoring workflow.
        
        Args:
            feature_tree: Input feature tree to refactor
            repo_data: Repository metadata
            
        Returns:
            Complete refactoring results
        """
        self.logger.info(f"Starting feature tree refactoring for {repo_data.get('repository_name', 'unknown')}")
        
        # Step 1: Plan subtrees
        planning_result = self.step1_plan_subtrees(feature_tree, repo_data)
        if not planning_result:
            return {"error": "Failed at subtree planning step"}
        
        # Step 2: Organize features
        component_subtrees = self.step2_organize_features(feature_tree, repo_data)
        if not component_subtrees:
            return {"error": "Failed at feature organization step"}
        
        # Calculate statistics based on leaf nodes for more accurate tracking
        original_leaf_nodes = len(extract_leaf_nodes(feature_tree))
        assigned_leaf_nodes_count = len(self.assigned_leaf_nodes)
        unassigned_leaf_nodes_count = original_leaf_nodes - assigned_leaf_nodes_count
        coverage_rate = assigned_leaf_nodes_count / original_leaf_nodes if original_leaf_nodes > 0 else 0
        
        result = {
            **repo_data,
            "Features": feature_tree,  # Original feature tree (matching checkpoint format)
            "Component": component_subtrees,  # Component-style subtrees
            "planning_result": {
                "total_subtrees": planning_result.total_subtrees,
                "subtree_plans": [plan.dict() for plan in planning_result.subtree_plans],
                "reasoning": planning_result.reasoning
            },
            "statistics": {
                "original_leaf_node_count": original_leaf_nodes,
                "assigned_leaf_node_count": assigned_leaf_nodes_count,
                "unassigned_leaf_node_count": unassigned_leaf_nodes_count,
                "coverage_rate": coverage_rate,
                "subtree_count": len(component_subtrees)
            },
            "memories": {
                "planning_memory": self.planning_memory.to_dict(),
                "organization_memory": self.organization_memory.to_dict()
            }
        }
        
        result_file = self.checkpoint_manager.get_path("feature_refactoring")
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
            
        repo_data["Component"] = result["Component"]
        self.logger.info(f"Refactoring completed: {len(component_subtrees)} subtrees, "
                        f"{coverage_rate:.1%} coverage, {unassigned_leaf_nodes_count} unassigned leaf nodes")
        
        return result, repo_data
    