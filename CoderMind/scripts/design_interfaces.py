#!/usr/bin/env python3
"""Design Interfaces Script - Implementation Level Step 4.

Function: Design function/class interfaces for each file in the repository
- Reads skeleton.json, data_flow.json, and base_classes.json for context
- For each subtree (in data flow order), plans file implementation order
- For each file, designs interfaces with signatures, docstrings, and feature mappings
- Validates Python syntax and docstring presence
- Collects fine-grained dependencies (inheritance, invocation, type references)
- Updates repo_rpg.json with dependency edges

Input: 
  - .cmind/skeleton.json (file structure with feature assignments)
  - .cmind/data_flow.json (data flow with subtree order)
  - .cmind/base_classes.json (base classes for context)
Output: 
  - .cmind/interfaces.json (interfaces organized by subtree and file, with enhanced_data_flow)
  - .cmind/repo_rpg.json (updated with fine-grained dependency edges)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import trajectory module
from common.trajectory import Trajectory, load_or_create_trajectory
from common.llm_client import LLMClient

# Import the InterfaceOrchestrator and DependencyCollector
from func_design.interface_agent import InterfaceOrchestrator, DependencyCollector

# Import Global Interface Reviewer
from func_design.interface_review import (
    InterfaceReviewer,
    print_review_summary,
)

# Import unified InterfacesStore
from func_design.interfaces_store import InterfacesStore

# Import RPG models for updating repo_rpg.json
from rpg import RPG, Node, NodeType, Edge, EdgeType, NodeMetaData, strip_uuid8

# Import centralized paths
from common.paths import (
    SKELETON_FILE as INPUT_SKELETON,
    DATA_FLOW_FILE as INPUT_DATA_FLOW,
    BASE_CLASSES_FILE as INPUT_BASE_CLASSES,
    INTERFACES_FILE as OUTPUT_FILE,
    REPO_RPG_FILE,
)
from common import print_unicode_table, get_repo_info_from_files
import ast
from common import get_project_background_context
from func_design.interface_review import review_orphan_units


def count_total_files(skeleton: Dict[str, Any]) -> int:
    """Count total files in skeleton."""
    count = 0
    
    def traverse(node):
        nonlocal count
        if node.get("type") == "file":
            if node.get("feature_paths"):  # Only count files with features
                count += 1
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    return count


def count_total_features(skeleton: Dict[str, Any]) -> int:
    """Count total features in skeleton."""
    return len(collect_skeleton_features(skeleton))


def collect_skeleton_features(skeleton: Dict[str, Any]) -> set:
    """Return the canonical set of feature paths declared in skeleton.json.

    Used by ``InterfaceReviewer._apply_add_interface`` to validate that an
    LLM-requested ``feature_path`` corresponds to a real feature, not an
    invented one.
    """
    features: set = set()

    def traverse(node):
        if node.get("type") == "file":
            for fp in node.get("feature_paths", []):
                features.add(fp)
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)

    root = skeleton.get("root", skeleton)
    traverse(root)
    return features


def collect_rpg_feature_paths(rpg_path: Path) -> set:
    """Return the set of feature paths present in the current repo_rpg.json.

    Walking the RPG tree (not just ``skeleton``) is required because earlier
    pipeline phases may have **pruned** orphan feature nodes. The
    ``add_interface`` handler must reject requests for pruned features
    because ``update_rpg`` silently drops ``meta.path`` write-backs that
    target missing nodes.

    Returns an empty set on any I/O / parsing error (handler then degrades
    gracefully — ``rpg_features=set()`` disables the RPG-presence check).
    """
    if not rpg_path or not Path(rpg_path).is_file():
        return set()
    try:
        rpg = RPG.load_json(str(rpg_path))
    except Exception:  # noqa: BLE001 — best-effort, degraded mode on any failure
        return set()
    paths: set = set()
    for node in rpg.nodes.values():
        if node.node_type == "feature":
            fp = node.feature_path()
            if fp:
                paths.add(fp)
    return paths


def extract_known_classes_and_types(base_classes: Dict[str, Any]) -> tuple:
    """Extract known base class names and type names from base_classes.json.
    
    Returns:
        Tuple of (known_base_classes: Set[str], known_types: Set[str])
    """
    known_base_classes = set()
    known_types = set()
    
    base_classes_list = base_classes.get("base_classes", [])
    
    for bc in base_classes_list:
        code = bc.get("code", "")
        if not code:
            continue
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    known_base_classes.add(node.name)
                    # Classes can also be used as types
                    known_types.add(node.name)
        except SyntaxError:
            continue
    
    # Also add class_names if provided
    for name in base_classes.get("class_names", []):
        known_base_classes.add(name)
        known_types.add(name)
    
    # Also process data_structures - these are known types (not base classes)
    data_structures_list = base_classes.get("data_structures", [])
    
    for ds in data_structures_list:
        code = ds.get("code", "")
        if code:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        known_types.add(node.name)
            except SyntaxError:
                continue
        
        # Also add data_flow_types names as known types
        for dt_name in ds.get("data_flow_types", []):
            known_types.add(dt_name)
    
    # Add data_structure_names if provided
    for name in base_classes.get("data_structure_names", []):
        known_types.add(name)
    
    return known_base_classes, known_types


# ============================================================================
# RPG Update Function
# ============================================================================

def update_rpg_with_interfaces(
    interfaces_data: Dict[str, Any],
    base_classes: Dict[str, Any],
    rpg_path: Path
):
    """.. deprecated:: This standalone function is NOT called at runtime.

        The actual RPG update is done by ``InterfacesStore.update_rpg()``
        (in func_design/interfaces_store.py, called at line ~899).
        Use ``rpg.service.RPGService`` for new code.

    Update RPG with interface design results.
    
    This function does NOT create new nodes. Instead, it:
    1. Updates existing feature nodes' meta.path with implementation location
    2. Adds SAME_UNIT edges when multiple features share the same implementation unit
    3. Adds fine-grained dependency edges (INHERITS, INVOKES, REFERENCES)
    
    Each feature maps to at most one implementation unit (class/function/method).
    
    Args:
        interfaces_data: Result dict containing subtrees with interfaces
        base_classes: Base classes data (for context)
        rpg_path: Path to the repo_rpg.json file
    """
    if not rpg_path.exists():
        logging.warning(f"RPG file not found: {rpg_path}")
        return
    
    try:
        rpg = RPG.load_json(str(rpg_path))
    except Exception as e:
        logging.error(f"Failed to load RPG: {e}")
        return
    
    # Remove old edges by generator
    rpg.remove_edges_by_generator("design_interfaces")
    
    updated_features = 0
    added_same_unit_edges = 0
    added_dependency_edges = 0
    skipped = 0
    
    # Build feature name -> node mapping for quick lookup
    feature_nodes: Dict[str, Node] = {}
    for node in rpg.nodes.values():
        # Feature nodes are typically at node_type "feature" or leaf nodes
        if node.node_type == "feature" or node.level == rpg.MAX_FEATURE_LEVEL:
            feature_nodes[node.name] = node
            # Also index by feature_path
            feature_path = node.feature_path()
            if feature_path:
                feature_nodes[feature_path] = node
    
    # Track unit -> list of feature nodes mapping for SAME_UNIT edges
    unit_to_features: Dict[str, List[Node]] = {}
    
    # Process interfaces from subtrees
    subtrees = interfaces_data.get("subtrees", interfaces_data.get("components", {}))
    
    for subtree_name, subtree_data in subtrees.items():
        # Support both "interfaces" and "files" format
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        
        for file_path, file_data in file_interfaces.items():
            # Get units_to_features mapping: {unit_name: [feature_paths]}
            units_to_features = file_data.get("units_to_features", {})
            
            for unit_name, feature_list in units_to_features.items():
                if not isinstance(feature_list, list):
                    continue
                
                # Build the implementation path for this unit
                # Format: file_path::unit_name (e.g., "src/parser.py::Parser::parse")
                impl_path = f"{file_path}::{unit_name}"
                
                # Track features that share this unit
                features_for_unit = []
                
                for feature_path in feature_list:
                    # Find the feature node
                    feature_node = feature_nodes.get(feature_path)
                    if not feature_node:
                        # Try finding by name (last part of path)
                        feature_name = feature_path.split("/")[-1] if "/" in feature_path else feature_path
                        feature_node = feature_nodes.get(feature_name)
                    
                    if not feature_node:
                        logging.warning(f"Feature node not found: {feature_path}")
                        skipped += 1
                        continue
                    
                    # Update the feature node's meta.path with implementation location
                    if feature_node.meta is None:
                        feature_node.meta = NodeMetaData()
                    
                    feature_node.meta.path = impl_path
                    # Note: Do NOT modify generator - it should reflect the step that first created the node
                    
                    # Infer type_name from unit_name prefix ("class Foo" or "function bar")
                    if unit_name.startswith("class "):
                        feature_node.meta.type_name = NodeType.CLASS
                    elif unit_name.startswith("function "):
                        feature_node.meta.type_name = NodeType.FUNCTION
                    elif "::" in impl_path:
                        # Fallback: infer from path structure
                        parts = impl_path.split("::")
                        if len(parts) >= 3:
                            feature_node.meta.type_name = NodeType.METHOD
                    
                    updated_features += 1
                    features_for_unit.append(feature_node)
                    logging.debug(f"Updated feature '{feature_path}' with path: {impl_path}")
                
                # Record for SAME_UNIT edge creation
                if impl_path not in unit_to_features:
                    unit_to_features[impl_path] = []
                unit_to_features[impl_path].extend(features_for_unit)
    
    # Add SAME_UNIT edges for features sharing the same implementation unit
    for impl_path, feature_list in unit_to_features.items():
        if len(feature_list) < 2:
            continue
        
        # Create edges between all pairs (only one direction to avoid duplicates)
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                node_a = feature_list[i]
                node_b = feature_list[j]
                
                # Check if edge already exists
                existing = rpg.find_edge_by_signature(
                    strip_uuid8(node_a.id),
                    strip_uuid8(node_b.id),
                    EdgeType.SAME_UNIT
                )
                if existing:
                    continue
                
                # Also check reverse direction
                existing_rev = rpg.find_edge_by_signature(
                    strip_uuid8(node_b.id),
                    strip_uuid8(node_a.id),
                    EdgeType.SAME_UNIT
                )
                if existing_rev:
                    continue
                
                edge = Edge(
                    src=node_a.id,
                    dst=node_b.id,
                    relation=EdgeType.SAME_UNIT,
                    meta=NodeMetaData(
                        description=f"Share implementation: {impl_path}",
                        generator="design_interfaces"
                    )
                )
                rpg.edges.append(edge)
                added_same_unit_edges += 1
                logging.debug(f"Added SAME_UNIT edge: {node_a.name} <-> {node_b.name}")
    
    # Process enhanced_data_flow for dependency edges (INHERITS, INVOKES, REFERENCES)
    enhanced_data_flow = interfaces_data.get("enhanced_data_flow", {})
    
    if enhanced_data_flow:
        # Process inheritance edges
        for edge_data in enhanced_data_flow.get("inheritance_edges", []):
            child = edge_data.get("child", "")
            parent = edge_data.get("parent", "")
            
            if not child or not parent:
                continue
            
            # Find nodes by name
            child_node = _find_node_by_name(rpg, child)
            parent_node = _find_node_by_name(rpg, parent)
            
            if child_node and parent_node:
                existing = rpg.find_edge_by_signature(
                    strip_uuid8(child_node.id),
                    strip_uuid8(parent_node.id),
                    EdgeType.INHERITS
                )
                if not existing:
                    edge = Edge(
                        src=child_node.id,
                        dst=parent_node.id,
                        relation=EdgeType.INHERITS,
                        meta=NodeMetaData(
                            description=f"{child} inherits from {parent}",
                            generator="design_interfaces"
                        )
                    )
                    rpg.edges.append(edge)
                    added_dependency_edges += 1
        
        # Process invocation edges
        for edge_data in enhanced_data_flow.get("invocation_edges", []):
            caller = edge_data.get("caller", "")
            callee = edge_data.get("callee", "")
            
            if not caller or not callee:
                continue
            
            caller_node = _find_node_by_name(rpg, caller)
            callee_node = _find_node_by_name(rpg, callee)
            
            if caller_node and callee_node:
                existing = rpg.find_edge_by_signature(
                    strip_uuid8(caller_node.id),
                    strip_uuid8(callee_node.id),
                    EdgeType.INVOKES
                )
                if not existing:
                    edge = Edge(
                        src=caller_node.id,
                        dst=callee_node.id,
                        relation=EdgeType.INVOKES,
                        meta=NodeMetaData(
                            description=f"{caller} invokes {callee}",
                            generator="design_interfaces"
                        )
                    )
                    rpg.edges.append(edge)
                    added_dependency_edges += 1
        
        # Process reference edges
        for edge_data in enhanced_data_flow.get("reference_edges", []):
            unit = edge_data.get("unit", "")
            ref_type = edge_data.get("referenced_type", "")
            
            if not unit or not ref_type:
                continue
            
            unit_node = _find_node_by_name(rpg, unit)
            type_node = _find_node_by_name(rpg, ref_type)
            
            if unit_node and type_node:
                existing = rpg.find_edge_by_signature(
                    strip_uuid8(unit_node.id),
                    strip_uuid8(type_node.id),
                    EdgeType.REFERENCES
                )
                if not existing:
                    edge = Edge(
                        src=unit_node.id,
                        dst=type_node.id,
                        relation=EdgeType.REFERENCES,
                        meta=NodeMetaData(
                            description=f"{unit} references type {ref_type}",
                            generator="design_interfaces"
                        )
                    )
                    rpg.edges.append(edge)
                    added_dependency_edges += 1
    
    # Process global review: mark entry points on RPG nodes
    global_review = interfaces_data.get("global_review", {})
    entry_points = global_review.get("entry_points", [])
    marked_entry_points = 0
    
    if entry_points:
        for ep in entry_points:
            ep_unit = ep.get("unit_name", "")
            ep_file = ep.get("file_path", "")
            ep_rationale = ep.get("rationale", "")
            
            if not ep_unit:
                continue
            
            # Find the node by unit name
            ep_node = _find_node_by_name(rpg, ep_unit)
            
            if not ep_node:
                # Try matching by file_path::unit_name in meta.path
                expected_path = f"{ep_file}::{ep_unit}" if ep_file else ""
                if expected_path:
                    for node in rpg.nodes.values():
                        if node.meta and node.meta.path == expected_path:
                            ep_node = node
                            break
            
            if ep_node:
                if ep_node.meta is None:
                    ep_node.meta = NodeMetaData()
                # Append entry_point marker to description
                ep_marker = f"[ENTRY_POINT] {ep_rationale}".strip()
                if ep_node.meta.description:
                    if "[ENTRY_POINT]" not in ep_node.meta.description:
                        ep_node.meta.description += f" | {ep_marker}"
                else:
                    ep_node.meta.description = ep_marker
                marked_entry_points += 1
                logging.debug(f"Marked entry point: {ep_unit} in {ep_file}")
            else:
                logging.debug(f"Entry point node not found in RPG: {ep_unit} ({ep_file})")
    
    # Save RPG
    rpg.save_json(str(rpg_path))
    
    total_changes = updated_features + added_same_unit_edges + added_dependency_edges + marked_entry_points
    if total_changes > 0:
        print(f"[OK] RPG updated: {updated_features} features updated, "
              f"{added_same_unit_edges} SAME_UNIT edges, "
              f"{added_dependency_edges} dependency edges, "
              f"{marked_entry_points} entry points marked. Skipped: {skipped}")
    else:
        print(f"No interface updates applied. Skipped: {skipped}")


def _find_node_by_name(rpg: RPG, name: str) -> Optional[Node]:
    """Find a node by name (class, function, or feature name).

    .. deprecated::
        Use ``rpg.service.RPGService.find_node_by_unit_name()`` instead.
    
    Searches by:
    1. Exact node.name match
    2. meta.path match (e.g., "src/file.py::class Foo" matches "class Foo")
    3. Qualified name suffix (e.g., "ClassName.method" -> "method")
    """
    # Try exact name match first
    for node in rpg.nodes.values():
        if node.name == name:
            return node
    
    # Try matching by meta.path (e.g., "src/file.py::class ClassName")
    for node in rpg.nodes.values():
        if node.meta and node.meta.path:
            path_str = node.meta.path if isinstance(node.meta.path, str) else ""
            # Extract unit part from path like "src/file.py::class Foo"
            if "::" in path_str:
                unit_part = path_str.split("::", 1)[-1]
                # Match "class Foo" == "class Foo" or "function bar" == "function bar"
                if unit_part == name:
                    return node
                # Also try matching the bare name (e.g., "Foo" from "class Foo")
                if " " in unit_part:
                    bare_name = unit_part.split(" ", 1)[-1]
                    if bare_name == name:
                        return node
    
    # Try matching by the last part of a qualified name (e.g., "ClassName.method" -> "method")
    if "." in name:
        short_name = name.rsplit(".", 1)[-1]
        for node in rpg.nodes.values():
            if node.name == short_name:
                return node
    
    return None


def prune_orphan_features_from_rpg(
    surviving_feature_paths: set,
    rpg_path: Path,
) -> Dict[str, Any]:
    """.. deprecated:: This standalone function is NOT called at runtime.

        The actual pruning is done by ``InterfacesStore._prune_rpg_orphan_features()``.
        Use ``rpg.service.RPGService.prune_orphan_features()`` for new code.

    Remove features from repo_rpg.json that have **no surviving interface unit**.

    After interface pruning, ``surviving_feature_paths`` contains the set of
    feature paths that still map to at least one interface unit.  Any RPG
    feature node whose ``feature_path()`` is NOT in this set is removed,
    along with any edges referencing the removed nodes.

    Empty parent nodes (feature_group / category / subcategory) whose children
    have all been removed are also pruned to keep the tree clean.

    Args:
        surviving_feature_paths: Feature path strings that still have at least
            one implementing interface unit.
        rpg_path: Path to the ``repo_rpg.json`` file.

    Returns:
        Summary dict with pruned_node_count, pruned_node_names, pruned_edge_count.
    """
    empty = {"pruned_node_count": 0, "pruned_node_names": [], "pruned_edge_count": 0}

    if not rpg_path.exists():
        logging.warning(f"RPG file not found for orphan pruning: {rpg_path}")
        return empty

    try:
        rpg = RPG.load_json(str(rpg_path))
    except Exception as e:
        logging.error(f"Failed to load RPG for orphan pruning: {e}")
        return empty

    if not surviving_feature_paths:
        logging.info("No surviving features at all — skipping RPG pruning to avoid wiping entire tree")
        return empty

    # ---- 1. Identify feature nodes to remove ----
    #   A feature node is removed when its feature_path() is NOT in the
    #   surviving set.  We only consider leaf-level feature nodes (the ones
    #   that correspond to actual implementation units).
    nodes_to_remove: Dict[str, Node] = {}

    for node in rpg.nodes.values():
        if node.node_type != "feature" and node.level != rpg.MAX_FEATURE_LEVEL:
            continue
        fp = node.feature_path()
        if fp and fp in surviving_feature_paths:
            continue  # this feature survives
        if node.name in surviving_feature_paths:
            continue  # fallback match by name
        # This feature has no surviving interface unit → mark for removal
        nodes_to_remove[node.id] = node

    if not nodes_to_remove:
        logging.info("All RPG feature nodes have surviving interface units — nothing to prune")
        return empty

    removed_ids = set(nodes_to_remove.keys())
    pruned_names = [n.name for n in nodes_to_remove.values()]

    # ---- 2. Remove the feature nodes (dict + tree) ----
    for nid in removed_ids:
        node = rpg.nodes.pop(nid, None)
        if node:
            parent = node.parent()
            if parent:
                parent.remove_child(node)

    # ---- 3. Prune empty parent nodes (bottom-up) ----
    #   After removing feature leaves, some parent nodes (feature_group,
    #   category, subcategory, functional_area) may have zero remaining
    #   children.  Remove those iteratively.
    pruned_parents = 0
    parent_types = {"feature_group", "category", "subcategory", "functional_area"}
    changed = True
    while changed:
        changed = False
        to_remove_parents = []
        for nid, node in rpg.nodes.items():
            if str(node.node_type) not in parent_types:
                continue
            if not node.children():
                to_remove_parents.append(nid)
        for nid in to_remove_parents:
            removed_ids.add(nid)
            node = rpg.nodes.pop(nid, None)
            if node:
                parent = node.parent()
                if parent:
                    parent.remove_child(node)
            pruned_parents += 1
            changed = True

    # ---- 4. Remove edges referencing removed nodes ----
    edges_before = len(rpg.edges)
    rpg.edges = [
        e for e in rpg.edges
        if e.src not in removed_ids and e.dst not in removed_ids
    ]
    pruned_edge_count = edges_before - len(rpg.edges)

    # ---- 5. Save ----
    rpg.save_json(str(rpg_path))

    feat_count = len(nodes_to_remove)
    logging.info(
        f"Pruned {feat_count} orphan feature nodes, {pruned_parents} empty parent nodes, "
        f"{pruned_edge_count} edges from RPG"
    )
    print(
        f"[OK] RPG pruned: removed {feat_count} orphan feature nodes"
        + (f", {pruned_parents} empty parent nodes" if pruned_parents else "")
        + f", {pruned_edge_count} edges"
        + f" (surviving: {len(rpg.nodes)} nodes, {len(rpg.edges)} edges)"
    )

    return {
        "pruned_node_count": feat_count + pruned_parents,
        "pruned_node_names": pruned_names,
        "pruned_edge_count": pruned_edge_count,
    }


# ============================================================================
# Interface Designer
# ============================================================================

class InterfaceDesigner:
    """Design interfaces using InterfaceOrchestrator."""

    def __init__(
        self,
        max_file_iterations: int = 10,
        max_planning_retries: int = 3,
        trajectory: Optional[Trajectory] = None,
        output_path: Optional[str] = None
    ):
        self.max_file_iterations = max_file_iterations
        self.max_planning_retries = max_planning_retries
        self.trajectory = trajectory
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)
        self._current_step_id: Optional[int] = None
        self.llm: Optional[LLMClient] = None  # Created lazily when step_id is known
    
    def build(
        self,
        skeleton: Dict[str, Any],
        data_flow: Dict[str, Any],
        base_classes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design interfaces for all files in the skeleton.
        
        Args:
            skeleton: The skeleton.json data
            data_flow: The data_flow.json data
            base_classes: The base_classes.json data
            
        Returns:
            Dict containing all interfaces organized by subtree
        """
        # Get repository info
        repo_name, repo_info = get_repo_info_from_files()
        
        # Enrich repo_info with project background / technology context
        # so it flows through to all interface design prompts.
        project_background = get_project_background_context()
        if project_background and project_background.strip():
            enriched_repo_info = (
                f"{repo_info}\n\n{project_background}\n"
                "When the project specifies a concrete technology stack, design interfaces "
                "that are idiomatic for those technologies (e.g., Flask route patterns, "
                "SQLAlchemy model methods, etc.)."
            )
        else:
            enriched_repo_info = repo_info
        
        # Get statistics
        total_files = count_total_files(skeleton)
        total_features = count_total_features(skeleton)
        subtree_order = data_flow.get("subtree_order", [])
        
        print("\n" + "=" * 70)
        print("INTERFACE DESIGN")
        print("=" * 70)
        print(f"Repository: {repo_name}")
        print(f"Total Files: {total_files}")
        print(f"Total Features: {total_features}")
        print(f"Subtrees: {len(subtree_order)}")
        if subtree_order:
            print("Processing Order:")
            for i, st in enumerate(subtree_order, 1):
                print(f"  {i}. {st}")
        print("=" * 70)
        
        # Record step start
        if self.trajectory:
            step = self.trajectory.add_step(
                "design_interfaces",
                f"Design interfaces for {total_files} files"
            )
            self._current_step_id = step.step_id
            self.trajectory.start_step(step.step_id)

        # Create LLMClient with trajectory support
        self.llm = LLMClient(trajectory=self.trajectory, step_id=self._current_step_id)

        # Get base classes list
        base_classes_list = base_classes.get("base_classes", [])
        data_structures_list = base_classes.get("data_structures", [])
        
        # Extract known classes and types for dependency analysis
        known_base_classes, known_types = extract_known_classes_and_types(base_classes)
        
        # Initialize dependency collector
        dependency_collector = DependencyCollector(
            known_base_classes=known_base_classes,
            known_types=known_types
        )
        
        # Store original data flow edges
        dependency_collector.set_original_edges(data_flow.get("data_flow", []))
        
        # Initialize orchestrator and run
        orchestrator = InterfaceOrchestrator(
            max_file_iterations=self.max_file_iterations,
            max_planning_retries=self.max_planning_retries,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=self._current_step_id,
            output_path=self.output_path
        )
        
        result = orchestrator.design_all_interfaces(
            skeleton=skeleton,
            data_flow=data_flow,
            base_classes=base_classes_list,
            repo_info=enriched_repo_info,
            dependency_collector=dependency_collector,
            data_structures=data_structures_list
        )
        
        # =====================================================================
        # Phase 1.5: Post-process invocation edges (normalise + resolve)
        # =====================================================================
        global_registry = result.get("_global_registry")
        if global_registry:
            dependency_collector.post_process_edges(global_registry)
            self.logger.info(
                f"Post-processed invocation edges: {dependency_collector.get_summary()}"
            )

        # Add enhanced data flow to result
        result["enhanced_data_flow"] = dependency_collector.to_dict()
        
        # Log dependency summary
        dep_summary = dependency_collector.get_summary()
        self.logger.info(f"Collected dependencies: {dep_summary}")
        
        # =====================================================================
        # Phase 2: Global Interface Review (entry points + wiring + auto-fix)
        # =====================================================================
        global_registry = result.pop("_global_registry", None)
        import_warnings = result.pop("_import_warnings", [])
        
        if global_registry and result.get("success"):
            self.logger.info("Starting global interface review phase...")
            print("\n" + "=" * 70)
            print("GLOBAL INTERFACE REVIEW")
            print("=" * 70)
            
            reviewer = InterfaceReviewer(
                trajectory=self.trajectory,
                step_id=self._current_step_id,
            )
            
            review_result = reviewer.review_and_fix(
                interfaces_data=result,
                enhanced_data_flow=result["enhanced_data_flow"],
                global_registry=global_registry,
                import_warnings=import_warnings,
                data_flow_edges=data_flow.get("data_flow", []),
                dependency_collector=dependency_collector,
                max_fix_iterations=2,
                skeleton_features=collect_skeleton_features(skeleton),
                rpg_features=collect_rpg_feature_paths(REPO_RPG_FILE),
            )
            
            # Update enhanced_data_flow in result (may have been modified by fixes)
            result["enhanced_data_flow"] = dependency_collector.to_dict()
            
            # Store review results in interfaces output
            result["global_review"] = {
                "entry_points": review_result.get("final_entry_points", []),
                "feature_orphans_count": len(review_result.get("final_feature_orphans", [])),
                "orphan_units_count": len(review_result.get("final_orphan_units", [])),
                "unapplied_fixes_count": len(review_result.get("unapplied_fixes", [])),
                "unapplied_fixes": review_result.get("unapplied_fixes", []),
                "iterations_run": review_result.get("iterations_run", 0),
                "passed": review_result.get("passed", False),
            }
            
            # Store import warnings summary
            if import_warnings:
                result["import_warnings_count"] = len(import_warnings)
            
            # Print review summary
            print_review_summary(review_result)

            # =================================================================
            # Phase 3: Create InterfacesStore and prune orphans
            # =================================================================
            # Create unified store from current result
            store = InterfacesStore.from_legacy_format(
                interfaces_data=result,
                enhanced_data_flow=result["enhanced_data_flow"],
                global_review={
                    "entry_points": review_result.get("final_entry_points", []),
                },
            )

            # =================================================================
            # Phase 3b: Review and prune orphan units
            # =================================================================
            # First, find orphan units
            orphan_keys = store.find_orphan_units()
            prune_summary = None  # Initialize to None

            if orphan_keys:
                print(f"\nFound {len(orphan_keys)} orphan interface units (no call edges)")

                # Get details for review
                orphan_details = store.get_orphan_unit_details(orphan_keys)

                # Review orphans using LLM (grouped by subtree)
                print("   Reviewing orphan units with LLM (by subtree)...")

                orphan_review_result = review_orphan_units(
                    orphan_details=orphan_details,
                    repo_info=repo_info,
                    subtree_interfaces=result.get("subtrees", {}),
                    llm_client=self.llm,
                )

                # Apply completed edges first (before pruning, so retained units get connected)
                if orphan_review_result.completed_edges:
                    all_edges = orphan_review_result.get_all_edges()
                    edges_added = store.add_edges(all_edges)
                    if edges_added:
                        print(f"   [OK] Added {edges_added} missing edges (design completion)")

                # Report retained units
                if orphan_review_result.keys_to_retain:
                    print(f"   [OK] Retaining {len(orphan_review_result.keys_to_retain)} units (deemed necessary)")
                    for key in orphan_review_result.keys_to_retain[:5]:
                        print(f"      -  {key}")
                    if len(orphan_review_result.keys_to_retain) > 5:
                        print(f"      ... and {len(orphan_review_result.keys_to_retain) - 5} more")

                # Prune only the units that LLM confirmed should be pruned
                if orphan_review_result.keys_to_prune:
                    prune_summary = store.prune_units(orphan_review_result.keys_to_prune)

                    pruned_count = len(prune_summary.pruned_units)
                    pruned_file_count = len(prune_summary.pruned_files)
                    orphan_feat_count = len(prune_summary.orphan_features)
                    print(
                        f"\n[OK] Pruned {pruned_count} orphan interface units"
                        + (f", {pruned_file_count} empty files removed" if pruned_file_count else "")
                        + (f", {orphan_feat_count} features orphaned" if orphan_feat_count else "")
                    )

                    # Record pruning info in global_review
                    result["global_review"]["pruned_units_count"] = pruned_count
                    result["global_review"]["pruned_files_count"] = pruned_file_count
                    result["global_review"]["orphan_features"] = prune_summary.get_orphan_features_list()
                    result["global_review"]["retained_orphans_count"] = len(orphan_review_result.keys_to_retain)
                    result["global_review"]["pruned_units"] = [
                        f"{unit.file_path}::{unit.name}" for unit in prune_summary.pruned_units
                    ]
                else:
                    print("\n[OK] All orphan units retained after review")
                    result["global_review"]["retained_orphans_count"] = len(orphan_review_result.keys_to_retain)

            # =================================================================
            # Phase 4: Update result from store and update RPG
            # =================================================================
            # Update result with store's current state (reflects pruning)
            store_export = store.to_interfaces_json()
            result["subtrees"] = store_export["subtrees"]
            result["enhanced_data_flow"] = store_export["enhanced_data_flow"]
            result["implemented_subtrees"] = store_export["implemented_subtrees"]

            # Store surviving feature paths for potential later use
            if prune_summary:
                result["_surviving_feature_paths"] = prune_summary.surviving_feature_paths

            # Update RPG using the store
            rpg_summary = store.update_rpg(REPO_RPG_FILE)

            # Record RPG pruning in global_review
            if rpg_summary.pruned_feature_nodes > 0:
                result["global_review"]["rpg_pruned_nodes"] = (
                    rpg_summary.pruned_feature_nodes + rpg_summary.pruned_parent_nodes
                )

            # Update dependency summary
            dep_summary = store.get_stats()
            self.logger.info(f"Final store stats: {dep_summary}")
        else:
            if not global_registry:
                self.logger.info("GlobalInterfaceRegistry not available, skipping global review")
        
        # Update trajectory
        if self.trajectory and self._current_step_id:
            if result.get("success"):
                # Count successful files
                success_count = 0
                total_count = 0
                for subtree_data in result.get("subtrees", {}).values():
                    # Support both "interfaces" (reference format) and "files" (old format)
                    file_container = subtree_data.get("interfaces", subtree_data.get("files", {}))
                    for file_data in file_container.values():
                        total_count += 1
                        # Check if units exist (success indicator)
                        if file_data.get("units"):
                            success_count += 1
                
                self.trajectory.complete_step(
                    self._current_step_id,
                    {"success_files": success_count, "total_files": total_count}
                )
            else:
                self.trajectory.fail_step(
                    self._current_step_id,
                    result.get("error", "Unknown error")
                )
        
        return result
    
    def print_summary(self, result: Dict[str, Any]) -> None:
        """Print summary of interface design.

        Organised as three explicit stages so the user can tell which check
        a failure came from:

          Stage 1 (Generation) — did the LLM produce signatures for every
            file? Counts files attempted vs files with at least one unit.
          Stage 2 (Coverage)   — is every skeleton feature mapped to some
            unit? Schema-level completeness, no graph reasoning.
          Stage 3 (Global Review) — is the call graph connected and
            self-consistent? This is the strictest check and the source of
            the canonical ``passed`` verdict.

        The overall "Overall: PASS/FAIL" line at the bottom mirrors Stage
        3's ``passed`` value.
        """
        print("\n" + "=" * 60)
        print("INTERFACE DESIGN SUMMARY")
        print("=" * 60)

        subtrees = result.get("subtrees", {})
        subtree_order = result.get("subtree_order", [])

        # ------------------------------------------------------------------
        # Pre-compute counters used by the three stages.
        # ------------------------------------------------------------------
        total_files = 0
        total_success = 0
        total_interfaces = 0
        failed_files: List[str] = []
        rows: List[List[Any]] = []
        all_unit_features: set = set()

        for subtree_name in subtree_order:
            subtree_data = subtrees.get(subtree_name, {})
            file_container = subtree_data.get("interfaces", subtree_data.get("files", {}))
            file_count = len(file_container)
            success_count = sum(1 for f in file_container.values() if f.get("units"))
            interface_count = sum(len(f.get("units", [])) for f in file_container.values())

            total_files += file_count
            total_success += success_count
            total_interfaces += interface_count
            for file_path, file_data in file_container.items():
                if not file_data.get("units"):
                    failed_files.append(file_path)
                for unit_features in (file_data.get("units_to_features", {}) or {}).values():
                    if isinstance(unit_features, list):
                        all_unit_features.update(unit_features)

            status = "[OK]" if success_count == file_count else f"[WARNING] {success_count}/{file_count}"
            rows.append([subtree_name[:25], file_count, interface_count, status])

        # ------------------------------------------------------------------
        # Stage 1: Generation
        # ------------------------------------------------------------------
        print("\n[Stage 1] Generation — did the LLM produce signatures for every file?")
        if rows:
            print_unicode_table(
                headers=["Subtree", "Files", "Interfaces", "Status"],
                rows=rows,
                title="Per-Subtree Summary"
            )
        print(f"  Files attempted: {total_files}")
        print(f"  Files successful: {total_success}"
              + (f"  ({(total_success / total_files) * 100:.1f}%)" if total_files else ""))
        print(f"  Total interfaces (units): {total_interfaces}")
        if failed_files:
            print(f"  [WARNING] {len(failed_files)} file(s) produced no units:")
            for f in failed_files[:10]:
                print(f"    - {f}")
            if len(failed_files) > 10:
                print(f"    ... and {len(failed_files) - 10} more")

        # Dependency edge summary — diagnostic, belongs with Stage 1.
        enhanced_data_flow = result.get("enhanced_data_flow", {}) or {}
        inheritance_count = len(enhanced_data_flow.get("inheritance_edges", []))
        invocation_count = len(enhanced_data_flow.get("invocation_edges", []))
        reference_count = len(enhanced_data_flow.get("reference_edges", []))
        if inheritance_count or invocation_count or reference_count:
            print("  Dependency edges collected:")
            print(f"    - Inheritance: {inheritance_count}")
            print(f"    - Invocation: {invocation_count}")
            print(f"    - Reference:  {reference_count}")
            cross_file = sum(
                1 for e in enhanced_data_flow.get("invocation_edges", [])
                if e.get("caller_file") != e.get("callee_file") and e.get("callee_file")
            )
            no_callee = sum(
                1 for e in enhanced_data_flow.get("invocation_edges", [])
                if not e.get("callee_file")
            )
            print(f"    - Cross-file invocations: {cross_file}")
            if no_callee:
                print(f"    - Unresolved callee_file: {no_callee}")

        # ------------------------------------------------------------------
        # Stage 2: Coverage
        # ------------------------------------------------------------------
        print("\n[Stage 2] Coverage — is every skeleton feature mapped to some unit?")
        # NOTE: We can compute the skeleton-feature universe from
        # interfaces.json alone (every unit declares its features); for a
        # canonical count we'd need skeleton.json again. The cross-validate
        # path already runs `check_interfaces.py` for that; here we just
        # report what the produced data carries.
        print(f"  Distinct features mapped to a unit: {len(all_unit_features)}")
        if not all_unit_features:
            print("  [WARNING] no feature mappings at all — likely Stage 1 failed")

        # ------------------------------------------------------------------
        # Stage 3: Global Review
        # ------------------------------------------------------------------
        global_review = result.get("global_review", {}) or {}
        print("\n[Stage 3] Global Review — is the call graph connected and self-consistent?")
        if not global_review:
            print("  (no global review was performed)")
        else:
            print(f"  Entry points: {len(global_review.get('entry_points', []))}")
            orphan_units = global_review.get("orphan_units_count", 0)
            orphan_features = global_review.get("feature_orphans_count", 0)
            unapplied_count = global_review.get("unapplied_fixes_count", 0)
            print(f"  Orphan units (no incoming edges): {orphan_units}")
            print(f"  Orphan features (no unit reachable from entry): {orphan_features}")
            print(f"  Unapplied fix requests: {unapplied_count}")
            for u in global_review.get("unapplied_fixes", [])[:3]:
                print(f"    - [{u.get('action','?')}] "
                      f"{u.get('file_path','?')}::{u.get('unit_name','?')}"
                      f" — {u.get('reason','?')}")
            if unapplied_count > 3:
                print(f"    ... and {unapplied_count - 3} more")

            if result.get("import_warnings_count"):
                print(f"  Import cross-validation warnings: {result['import_warnings_count']}")

            # Handler-added units (review-time interface completion).
            handler_added: List[str] = []
            for st_data in subtrees.values():
                file_container = st_data.get("interfaces", st_data.get("files", {}))
                for file_path, file_data in file_container.items():
                    for uname in file_data.get("_handler_added", []) or []:
                        handler_added.append(f"{file_path}::{uname}")
            if handler_added:
                print(f"  Handler-added units (review-time interface completion): {len(handler_added)}")
                for k in handler_added[:5]:
                    print(f"    + {k}")
                if len(handler_added) > 5:
                    print(f"    ... and {len(handler_added) - 5} more")

            # Pruning info (post-Stage-3 housekeeping).
            pruned_units_count = global_review.get("pruned_units_count", 0)
            pruned_files_count = global_review.get("pruned_files_count", 0)
            if pruned_units_count:
                print(f"  Pruned orphan units: {pruned_units_count}")
            if pruned_files_count:
                print(f"  Pruned empty files: {pruned_files_count}")
            pruned_feats = global_review.get("orphan_features", [])
            if pruned_feats:
                print(f"  Orphan features pruned from RPG: {len(pruned_feats)}")
                for of in pruned_feats[:5]:
                    print(f"    - {of['feature_path']} ({of['unit_key']})")
                if len(pruned_feats) > 5:
                    print(f"    ... and {len(pruned_feats) - 5} more")
            rpg_pruned = global_review.get("rpg_pruned_nodes", 0)
            if rpg_pruned:
                print(f"  RPG nodes pruned (cascading): {rpg_pruned}")

        # ------------------------------------------------------------------
        # Verdict — mirrors Stage 3's `passed`.
        # ------------------------------------------------------------------
        passed = bool(global_review.get("passed"))
        verdict = "✓ PASS" if passed else "✗ FAIL — see Stage 3 above"
        print(f"\nOverall: {verdict}")
        print("(Stage 3 is the strictest; PASS requires Stages 1+2 also clean.)")
        print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Design function/class interfaces for repository files"
    )
    parser.add_argument(
        "--skeleton", "-s",
        type=str,
        default=str(INPUT_SKELETON),
        help=f"Skeleton input file (default: {INPUT_SKELETON})"
    )
    parser.add_argument(
        "--data-flow", "-d",
        type=str,
        default=str(INPUT_DATA_FLOW),
        help=f"Data flow input file (default: {INPUT_DATA_FLOW})"
    )
    parser.add_argument(
        "--base-classes", "-b",
        type=str,
        default=str(INPUT_BASE_CLASSES),
        help=f"Base classes input file (default: {INPUT_BASE_CLASSES})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--max-file-iterations", "-m",
        type=int,
        default=10,
        help="Max iterations per file (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load skeleton
    skeleton_path = Path(args.skeleton)
    if not skeleton_path.exists():
        logger.error(f"Skeleton file not found: {skeleton_path}")
        print(f"ERROR: Skeleton file not found: {skeleton_path}")
        print("Please run /cmind.build_skeleton first.")
        return 1

    with open(skeleton_path, "r", encoding="utf-8") as f:
        skeleton = json.load(f)

    # Load data flow
    data_flow_path = Path(args.data_flow)
    data_flow = {}
    if data_flow_path.exists():
        try:
            with open(data_flow_path, "r", encoding="utf-8") as f:
                data_flow = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load data flow: {e}")
    else:
        logger.warning(f"Data flow file not found: {data_flow_path}")
        print(f"[WARNING] Warning: Data flow file not found: {data_flow_path}")
        print("  Run /cmind.build_data_flow first for better results.")

    # Load base classes
    base_classes_path = Path(args.base_classes)
    base_classes = {}
    if base_classes_path.exists():
        try:
            with open(base_classes_path, "r", encoding="utf-8") as f:
                base_classes = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load base classes: {e}")
    else:
        logger.warning(f"Base classes file not found: {base_classes_path}")
        print(f"[WARNING] Warning: Base classes file not found: {base_classes_path}")
        print("  Run /cmind.design_base_classes first for better results.")

    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("design_interfaces")
        
        if trajectory.is_resumable():
            print(f"\n[WARNING] Found in-progress execution from {trajectory.started_at}")
            print(f"  Resume point: {trajectory.resume_point.step_name}")
            print("  (Use --no-trajectory to start fresh)")
        
        trajectory.start(metadata={
            "skeleton_file": str(skeleton_path),
            "data_flow_file": str(data_flow_path),
            "base_classes_file": str(base_classes_path),
            "output_file": str(args.output),
            "max_file_iterations": args.max_file_iterations
        })

    try:
        # Design interfaces
        designer = InterfaceDesigner(
            max_file_iterations=args.max_file_iterations,
            trajectory=trajectory,
            output_path=str(args.output)
        )
        
        result = designer.build(skeleton, data_flow, base_classes)

        # Extract internal keys before JSON serialisation
        result.pop("_surviving_feature_paths", None)

        # Save output (interfaces.json)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] Interfaces saved to: {output_path}")
        designer.print_summary(result)
        print(f"\n[OK] Interfaces saved to: {output_path.name}")

        # RPG update is now handled inside InterfaceDesigner.build() via InterfacesStore

        if not result.get("success", True) and "error" in result:
            if trajectory:
                trajectory.fail(result["error"])
            return 1
        
        # Mark trajectory as complete
        if trajectory:
            subtrees = result.get("subtrees", {})
            total_files = sum(
                len(st.get("files", {}))
                for st in subtrees.values()
            )
            # Include dependency summary in metadata
            enhanced_data_flow = result.get("enhanced_data_flow", {})
            trajectory.complete(metadata={
                "subtrees": len(subtrees),
                "total_files": total_files,
                "inheritance_edges": len(enhanced_data_flow.get("inheritance_edges", [])),
                "invocation_edges": len(enhanced_data_flow.get("invocation_edges", [])),
                "reference_edges": len(enhanced_data_flow.get("reference_edges", []))
            })
            print(f"[OK] Trajectory saved to: {trajectory.trajectory_file.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Design failed: {e}")
        if trajectory:
            trajectory.fail(str(e))
        raise


if __name__ == "__main__":
    exit(main())
