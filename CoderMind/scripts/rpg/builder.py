#!/usr/bin/env python3
"""RPG Builder.

This module provides functionality to build RPG (Repository Program Graph)
from CoderMind's refactor_feature.json input format.

Key functions:
- create_initial_rpg: Build RPG from component architecture
"""

import json
import logging
from typing import Dict, Any, Union
from pathlib import Path

from common.language_meta import extract_language_metadata

from .models import RPG, Node, NodeMetaData, NodeType, uuid8


def create_initial_rpg(repo_data: Dict[str, Any]) -> RPG:
    """Create initial RPG from CoderMind's refactor_feature.json data.

    Args:
        repo_data: Dictionary from refactor_feature.json containing:
            - repository_name: Project name
            - repository_purpose: Project description
            - component: List of components with refactored_subtree

    Returns:
        RPG: Constructed graph with hierarchical node structure
    """
    repo_name = repo_data.get("repository_name", "repo")
    repo_info = repo_data.get("repository_purpose", "")
    repo_cmpt = repo_data.get("components", [])

    # Create RPG with basic info
    rpg = RPG(
        repo_name=repo_name,
        repo_info=repo_info,
        excluded_files=[]
    )
    
    # Set generator for repo node (created in RPG.__init__)
    if rpg.repo_node:
        rpg.repo_node.meta.generator = "build_skeleton"
        target_language = extract_language_metadata(repo_data)[0]
        if target_language:
            rpg.repo_node.meta.language = target_language

    logging.info(f"Creating initial RPG for repository: {repo_name}")
    logging.info(f"Found {len(repo_cmpt)} components to process")

    # Utility: Generate unique ID
    def _new_id(name: str) -> str:
        return f"{name}_{uuid8()}"

    # Utility: Determine node_type based on parent and child type
    def _get_node_type(parent: Node, is_leaf: bool, children_are_leaves: bool = False) -> str:
        """Determine node_type based on parent's node_type and node characteristics.
        
        Feature tree hierarchy:
        - repo -> functional_area
        - functional_area -> category
        - category -> subcategory (if children are groupings) or feature_group (if children are features)
        - subcategory -> subcategory (if children are groupings) or feature_group (if children are features)
        - feature_group -> feature
        
        Args:
            parent: Parent node
            is_leaf: Whether this node is a leaf (feature)
            children_are_leaves: Whether this node's children will be leaves (features)
        """
        parent_type = parent.node_type if parent else "repo"
        
        if is_leaf:
            # Leaf nodes are features
            return "feature"
        
        # Non-leaf nodes (groupings)
        if children_are_leaves:
            # This node's children are features -> it's a feature_group
            return "feature_group"
        
        # This node's children are more groupings
        type_progression = {
            "repo": "functional_area",
            "functional_area": "category",
            "category": "subcategory",
            "subcategory": "subcategory",  # Stay at subcategory for deeper nesting
            "feature_group": "feature",  # Should not happen, but fallback
        }
        return type_progression.get(parent_type, "subcategory")

    # Utility: Find or create child node under parent
    def _ensure_child(parent: Node, name: str, is_leaf: bool, children_are_leaves: bool = False) -> Node:
        """Create child node under parent if not exists.

        Uses signature matching (name and id prefix) to detect existing nodes.
        
        Args:
            parent: Parent node
            name: Node name
            is_leaf: Whether this node is a leaf (feature)
            children_are_leaves: Whether this node's children will be leaves (features)
        """
        # First check by name (simple match)
        existing = rpg.find_child_by_name(parent.id, name)
        if existing:
            return existing
        
        # Also check by signature (name + id prefix) in case of regeneration
        id_prefix = name  # ID format is "{name}_{uuid}"
        existing_by_sig = rpg.find_node_by_signature(name, id_prefix, parent.id)
        if existing_by_sig:
            return existing_by_sig

        node_type = _get_node_type(parent, is_leaf, children_are_leaves)
        
        node = Node(
            id=_new_id(name),
            name=name,
            node_type=node_type,
            # Level will be calculated by recalculate_levels_topdown
            meta=NodeMetaData(
                generator="build_skeleton"
            )
        )
        rpg.add_node(node)
        rpg.add_edge(parent, node, meta=NodeMetaData(generator="build_skeleton"))
        return node

    # Recursive: Convert refactored_subtree to node tree
    def _build_from_subtree(parent: Node, subtree: Any):
        """Convert refactored_subtree to nodes.

        subtree formats:
        - dict:      {"ChildA": {...}, "ChildB": [...]}
        - list:      ["feature1", "feature2"]
        - string:    "single_feature"
        """
        if isinstance(subtree, dict):
            for key, child in subtree.items():
                if isinstance(child, list):
                    # key is a grouping whose children are leaf features -> feature_group
                    group_node = _ensure_child(parent, key, is_leaf=False, children_are_leaves=True)
                    for feat in child:
                        if not feat:
                            continue
                        _ensure_child(group_node, str(feat), is_leaf=True)
                elif isinstance(child, dict):
                    # Still dict -> intermediate grouping with more groupings
                    node = _ensure_child(parent, key, is_leaf=False, children_are_leaves=False)
                    _build_from_subtree(node, child)
                else:
                    # Other type, treat as leaf
                    _ensure_child(parent, str(key), is_leaf=True)

        elif isinstance(subtree, list):
            # parent has direct list of features (parent should be feature_group)
            # Note: parent's node_type should already be set correctly by caller
            for feat in subtree:
                if not feat:
                    continue
                _ensure_child(parent, str(feat), is_leaf=True)

        else:
            # Single leaf
            _ensure_child(parent, str(subtree), is_leaf=True)

    # Main logic: Process all components
    components_processed = 0
    features_added = 0

    for component in repo_cmpt:
        cmpt_name = component.get("name", "") or "Component"
        re_tree = component.get("refactored_subtree", {})

        if not re_tree:
            logging.warning(f"Component '{cmpt_name}' has empty refactored_subtree")
            continue

        logging.debug(f"Processing component: {cmpt_name}")

        # Create component node as direct child of repo (level=1, node_type=functional_area)
        cmpt_node = rpg.find_child_by_name(rpg.repo_node.id, cmpt_name)
        if not cmpt_node:
            cmpt_node = Node(
                id=_new_id(cmpt_name),
                name=cmpt_name,
                node_type="functional_area",
                level=1,
                meta=NodeMetaData(
                    type_name=NodeType.DIRECTORY,
                    path=None,
                    generator="build_skeleton"
                )
            )
            rpg.add_node(cmpt_node)
            rpg.add_edge(rpg.repo_node, cmpt_node, meta=NodeMetaData(generator="build_skeleton"))

        # Convert entire refactored_subtree under this component node
        initial_node_count = len(rpg.nodes)
        _build_from_subtree(cmpt_node, re_tree)
        component_features = len(rpg.nodes) - initial_node_count

        features_added += component_features
        components_processed += 1
        logging.debug(f"  Added {component_features} nodes for component '{cmpt_name}'")

    # Recalculate levels and node types using topdown approach
    rpg.recalculate_levels_topdown()

    logging.info("RPG creation completed:")
    logging.info(f"  - Components processed: {components_processed}")
    logging.info(f"  - Total nodes: {len(rpg.nodes)}")
    logging.info(f"  - Total edges: {len(rpg.edges)}")
    logging.info(f"  - Features added: {features_added}")

    return rpg


def load_refactor_feature_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate refactor_feature.json data.

    Args:
        file_path: Path to refactor_feature.json

    Returns:
        Dict containing validated repository data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required fields are missing
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["repository_name", "components"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields in {file_path}: {missing_fields}")

    if not data.get("components"):
        raise ValueError(f"No components found in {file_path}")

    logging.info(f"Loaded repository data: {data.get('repository_name', 'unknown')}")
    logging.info(f"Found {len(data.get('components', []))} components")

    return data


def count_features_in_component_tree(subtree: Any) -> int:
    """Count total features in a component's refactored_subtree.

    Args:
        subtree: The refactored_subtree structure

    Returns:
        int: Total number of leaf features
    """
    if isinstance(subtree, dict):
        total = 0
        for key, value in subtree.items():
            if key == "description":  # Skip metadata
                continue
            total += count_features_in_component_tree(value)
        return total
    elif isinstance(subtree, list):
        return len([item for item in subtree if item])  # Count non-empty items
    else:
        return 1 if subtree else 0


def get_rpg_statistics(rpg: RPG) -> Dict[str, Any]:
    """Get comprehensive statistics about an RPG.

    Args:
        rpg: The RPG instance

    Returns:
        Dict containing various statistics
    """
    stats = {
        "total_nodes": len(rpg.nodes),
        "total_edges": len(rpg.edges),
        "repo_name": rpg.repo_name,
        "levels": {},
        "node_types": {},
    }

    # Count by level
    for node in rpg.nodes.values():
        level = f"L{node.level}" if node.level is not None else "L?"
        stats["levels"][level] = stats["levels"].get(level, 0) + 1

        # Count by node type
        node_type = node.node_type or "unknown"
        stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

    return stats


if __name__ == "__main__":
    # Test functionality with default input
    logging.basicConfig(level=logging.INFO)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.paths import FEATURE_TREE_FILE

    if FEATURE_TREE_FILE.exists():
        try:
            data = load_refactor_feature_data(FEATURE_TREE_FILE)
            rpg = create_initial_rpg(data)
            stats = get_rpg_statistics(rpg)

            print("\nRPG Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            logging.error(f"Test failed: {e}")
    else:
        logging.info(f"No test file found at {FEATURE_TREE_FILE}")