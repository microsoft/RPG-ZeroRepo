"""
Task loading utilities for converting planned_batches_dict to properly ordered TaskBatch lists.

This module provides a centralized way to handle the complex task structure including:
- Regular implementation tasks (in subtree and file order)
- Integration tests (after each subtree)
- Final comprehensive tests and documentation (sorted by priority)
"""

from typing import Dict, List, Any
from ..rpg_gen.impl_level.plan_tasks.task_batch import TaskBatch


def load_batches_from_dict(
    planned_batches_dict: Dict, 
    graph_data: Dict,
    logger=None
) -> List[TaskBatch]:
    """
    Convert planned_batches_dict to a properly ordered list of TaskBatch objects.
    
    This function handles the complete task structure including:
    - Regular implementation tasks (in subtree and file order from graph_data)
    - Integration tests (after each subtree, identified by <INTEGRATION_TEST> file paths)
    - Final comprehensive tests and documentation (under FINAL_TASKS, sorted by priority)
    
    Args:
        planned_batches_dict: Dictionary containing task batches organized by subtree and file
        graph_data: Graph data containing subtree_order and files_order information
        logger: Optional logger for debugging information
        
    Returns:
        List of TaskBatch objects in proper execution order
    """
    if not planned_batches_dict:
        return []
        
    flattened_batches = []
    subtree_order = graph_data.get("data_flow_phase", {}).get("subtree_order", [])
    
    if logger:
        logger.debug(f"Loading task batches: {len(subtree_order)} subtrees, {len(planned_batches_dict)} total subtree groups")
    
    # Process regular subtrees first (in order)
    for subtree in subtree_order:
        if subtree not in planned_batches_dict:
            if logger:
                logger.debug(f"Subtree {subtree} not found in planned batches")
            continue
            
        subtree_files = planned_batches_dict[subtree]
        
        # Get file order from graph data for regular implementation files
        subtrees_data = graph_data.get('interfaces_phase', {}).get("interfaces", {}).get("subtrees", {})
        if subtree in subtrees_data:
            files_order = subtrees_data[subtree].get("files_order", [])
            
            if logger:
                logger.debug(f"Processing subtree {subtree}: {len(files_order)} regular files, {len(subtree_files)} total files")
            
            # Add regular implementation batches first
            for file_path in files_order:
                if file_path in subtree_files:
                    file_batches = subtree_files[file_path]
                    if file_batches and isinstance(file_batches[0], dict):
                        # Convert from dict format
                        for batch_dict in file_batches:
                            flattened_batches.append(TaskBatch.from_dict(batch_dict))
                    else:
                        # Already TaskBatch objects
                        flattened_batches.extend(file_batches)
            
            # Add integration test batches for this subtree (special file paths)
            integration_count = 0
            for file_path, file_batches in subtree_files.items():
                if file_path.startswith("<INTEGRATION_TEST>"):
                    integration_count += len(file_batches)
                    if file_batches and isinstance(file_batches[0], dict):
                        # Convert from dict format
                        for batch_dict in file_batches:
                            flattened_batches.append(TaskBatch.from_dict(batch_dict))
                    else:
                        # Already TaskBatch objects
                        flattened_batches.extend(file_batches)
            
            if integration_count > 0 and logger:
                logger.debug(f"Added {integration_count} integration test batches for subtree {subtree}")
    
    # Process FINAL_TASKS last (comprehensive tests and documentation)
    if "FINAL_TASKS" in planned_batches_dict:
        final_tasks = planned_batches_dict["FINAL_TASKS"]
        
        if logger:
            logger.debug(f"Processing FINAL_TASKS with {len(final_tasks)} task groups")
        
        # Sort by priority to ensure correct order
        all_final_batches = []
        for file_batches in final_tasks.values():
            if file_batches and isinstance(file_batches[0], dict):
                # Convert from dict format
                for batch_dict in file_batches:
                    all_final_batches.append(TaskBatch.from_dict(batch_dict))
            else:
                # Already TaskBatch objects
                all_final_batches.extend(file_batches)
        
        all_final_batches.sort(key=lambda b: b.priority)
        flattened_batches.extend(all_final_batches)
        
        if logger:
            logger.debug(f"Added {len(all_final_batches)} final task batches")
    
    if logger:
        logger.info(f"Loaded total {len(flattened_batches)} task batches in proper execution order")
    
    return flattened_batches


def load_batches_from_file(
    tasks_file_path: str,
    graph_data: Dict,
    logger=None
) -> List[TaskBatch]:
    """
    Load task batches from a saved tasks JSON file.
    
    Args:
        tasks_file_path: Path to the saved tasks JSON file
        graph_data: Graph data containing ordering information
        logger: Optional logger for debugging
        
    Returns:
        List of TaskBatch objects in proper execution order
    """
    import json
    
    with open(tasks_file_path, 'r') as f:
        tasks_data = json.load(f)
    
    planned_batches_dict = tasks_data.get("planned_batches_dict", {})
    
    if not planned_batches_dict:
        # Fallback to simple batches format
        if logger:
            logger.warning("Using fallback task loading (simple batches format)")
        return [TaskBatch.from_dict(task) for task in tasks_data.get("batches", [])]
    else:
        return load_batches_from_dict(planned_batches_dict, graph_data, logger)