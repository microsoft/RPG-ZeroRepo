#!/usr/bin/env python3
"""Data Flow Agent.

This module provides the DataFlowAgent for designing inter-component data flow
as a directed acyclic graph (DAG).

Key components:
- DataFlowAgent: Orchestrates data flow generation with validation
- Validation functions for DAG properties
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from pydantic import BaseModel, Field

from .data_flow_prompts import (
    DATA_FLOW_PROMPT,
    DATA_FLOW_REVIEW_PROMPT,
    format_functional_areas
)

# Import common LLMClient with trajectory support
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import LLMClient
from decoder_lang import get_backend
from decoder_lang.prompt_directive import with_language_directive


# ============================================================================
# Data Models
# ============================================================================

class DataFlowEdge(BaseModel):
    """Single data flow edge between components."""
    source: str = Field(..., description="Name of the source component that produces this data")
    target: str = Field(..., description="Name of the target component that consumes this data")
    data_id: str = Field(..., description="Unique identifier for this data exchange")
    data_type: str = Field(..., description="Logical type or schema of the data")
    transformation: str = Field(default="", description="Description of how data is processed during transfer")


class DataFlowOutput(BaseModel):
    """Output from LLM for data flow design."""
    data_flow: List[DataFlowEdge] = Field(..., min_length=1, description="List of data flow edges (must not be empty)")


# ============================================================================
# Validation Functions
# ============================================================================

def validate_data_flow(
    data_flow: List[Dict[str, Any]], 
    required_components: List[str]
) -> Tuple[bool, str]:
    """Validate data flow graph: 1. All source/target are valid components 2. No self-loops 3. No cycles (must be a DAG) 4. All components are covered (appear at least once).
    
    Returns: (is_valid, error_message_or_success)
    """
    errors = []
    component_set = set(required_components)
    used_components = set()
    
    # Build graph for cycle detection
    graph = defaultdict(list)
    edge_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    
    for i, edge in enumerate(data_flow):
        source = edge.get("source", "")
        target = edge.get("target", "")
        
        # Check source validity
        if not source:
            errors.append(f"Item {i}: 'source' is missing.")
            continue
        if source not in component_set:
            errors.append(f"Item {i}: 'source' node '{source}' is not in required_keys.")
        
        # Check target validity
        if not target:
            errors.append(f"Item {i}: 'target' is missing.")
            continue
        if target not in component_set:
            errors.append(f"Item {i}: 'target' node '{target}' is not in required_keys.")
        
        # Check self-loop
        if source == target:
            errors.append(f"Item {i}: self-loop detected ({source} -> {source})")
        
        # Add to graph
        if source and target and source != target:
            graph[source].append(target)
            used_components.add(source)
            used_components.add(target)
            edge_indices[(source, target)].append(i)
    
    # Check for unused components
    unused = component_set - used_components
    if unused:
        errors.append(
            f"Unused nodes from required_keys (i.e., no data flow defined): {sorted(unused)}"
        )
    
    # Check for cycles using DFS if no basic errors
    if not errors:
        visited = set()
        rec_stack = set()
        cycle_found = False
        cycle_path = []
        
        def has_cycle(node: str, path: List[str]) -> bool:
            nonlocal cycle_found, cycle_path
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path + [neighbor]):
                        return True
                elif neighbor in rec_stack:
                    cycle_path = path + [neighbor]
                    cycle_found = True
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in list(graph.keys()):
            if node not in visited:
                if has_cycle(node, [node]):
                    break
        
        if cycle_found:
            errors.append(f"Cycle detected in data flow: {' -> '.join(cycle_path)}")
    
    if errors:
        return False, "\n".join(errors)
    return True, "All data flow checks passed."


def compute_topological_order(data_flow: List[Dict[str, Any]]) -> List[str]:
    """Compute topological order of components based on data flow.

    Returns components in dependency order (sources before targets).
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()
    
    for edge in data_flow:
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source and target:
            graph[source].append(target)
            in_degree[target] += 1
            all_nodes.add(source)
            all_nodes.add(target)
    
    # Initialize in_degree for all nodes
    for node in all_nodes:
        if node not in in_degree:
            in_degree[node] = 0
    
    # Kahn's algorithm
    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Handle remaining nodes (cycle case - should not happen after validation)
    remaining = [n for n in all_nodes if n not in result]
    result.extend(remaining)
    
    return result


# ============================================================================
# Data Flow Agent
# ============================================================================

class DataFlowAgent:
    """Agent for designing inter-component data flow."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = 5,
        logger: Optional[logging.Logger] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None,
        target_language: Optional[str] = None,
    ):
        # Create LLMClient with trajectory support if not provided
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
            # Update trajectory info on existing client
            if trajectory is not None:
                self.llm.set_trajectory(trajectory, step_id)
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.backend = get_backend(target_language)
    
    def build_data_flow(
        self,
        repo_name: str,
        repo_info: str,
        functional_areas: List[str],
        component_dirs: Optional[Dict[str, str]] = None,
        skeleton_tree: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build data flow between functional areas.
        
        Args:
            repo_name: Repository name
            repo_info: Repository description
            functional_areas: List of functional area names (components)
            component_dirs: Optional mapping of component to directory
            skeleton_tree: Optional tree string of skeleton
            
        Returns:
            Dict containing:
            - data_flow: List of data flow edges
            - subtree_order: Topological order of components
            - success: Whether the operation succeeded
        """
        self.logger.info(f"[DataFlowAgent] Building data flow for {len(functional_areas)} components")

        # Build system prompt (tool description is now integrated)
        system_prompt = with_language_directive(DATA_FLOW_PROMPT, self.backend)

        # Build user prompt
        areas_str = format_functional_areas(functional_areas, component_dirs)
        
        user_prompt = f"""[Step 1 User Query]: Based on the repository structure and dependency relationships, generate data flow between components:
Repository Name: {repo_name}
Repository Info: {repo_info}
Functional Graph Overview: {areas_str}
Component Names: {', '.join(functional_areas)}
Please use the generate_data_flow tool to create comprehensive data flow definitions.
Focus on:
1. What data flows between components
2. Data types and formats
3. Any transformations applied
4. Direction of data flow"""

        if skeleton_tree:
            user_prompt += f"\n\nRepository Skeleton:\n{skeleton_tree}"
        
        # Iterate until valid or max iterations
        last_error = ""
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"[DataFlowAgent] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Build prompt with error feedback if needed
            current_user_prompt = user_prompt
            if last_error:
                current_user_prompt += f"\n\n[Validation Failed]\nError: {last_error}\nPlease fix the issues and try again."
            
            # Call LLM with Pydantic validation
            _, result_model, _ = self.llm.call_structured(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                response_model=DataFlowOutput,
                purpose=f"data_flow_design_{iteration + 1}",
                max_retries=1  # Handle retries at this level
            )
            
            if not result_model:
                last_error = "Failed to parse LLM response or Pydantic validation failed."
                continue
            
            # Convert to dict list for custom validation
            data_flow = [edge.model_dump() for edge in result_model.data_flow]
            
            # Custom DAG validation
            is_valid, error_msg = validate_data_flow(data_flow, functional_areas)
            
            if is_valid:
                # Compute subtree order
                subtree_order = compute_topological_order(data_flow)
                
                self.logger.info(f"[DataFlowAgent] Data flow validated successfully with {len(data_flow)} edges")
                return {
                    "data_flow": data_flow,
                    "subtree_order": subtree_order,
                    "success": True,
                    "iterations": iteration + 1
                }
            else:
                self.logger.warning(f"[DataFlowAgent] Validation failed: {error_msg}")
                last_error = error_msg
        
        # Failed after all iterations
        self.logger.error(f"[DataFlowAgent] Failed after {self.max_iterations} iterations")
        return {
            "data_flow": [],
            "subtree_order": [],
            "success": False,
            "error": last_error,
            "iterations": self.max_iterations
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    agent = DataFlowAgent()
    result = agent.build_data_flow(
        repo_name="test-repo",
        repo_info="A test repository",
        functional_areas=["ComponentA", "ComponentB", "ComponentC"]
    )
    print(json.dumps(result, indent=2))
