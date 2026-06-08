#!/usr/bin/env python3
"""Design Base Classes Script - Implementation Level Step 3.

Function: Design shared base classes and data structures for the repository
- Reads skeleton.json and data_flow.json for context
- Designs functional base classes (behavioral abstractions)
- Designs global data structures (shared data formats)
- Validates Python code syntax

Input: 
  - .cmind/skeleton.json (file structure)
  - .cmind/data_flow.json (data flow between components)
Output: .cmind/base_classes.json (base class definitions with code)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import trajectory module
from common.trajectory import Trajectory, load_or_create_trajectory

# Import common utils
from common import (
    get_skeleton_tree_string,
    extract_functional_areas_from_skeleton,
    format_functional_graph_overview,
    print_unicode_table,
    get_repo_info_from_files,
)

# Import the BaseClassAgent
from func_design.base_class_agent import (
    BaseClassAgent,
    extract_declaration_names,
)
from decoder_lang import get_backend
from rpg import (
    RPG, Node, NodeType, EdgeType, NodeMetaData, strip_uuid8, uuid8,
    class_node_path,
)

# Import centralized paths
from common.paths import (
    SKELETON_FILE as INPUT_SKELETON,
    DATA_FLOW_FILE as INPUT_DATA_FLOW,
    BASE_CLASSES_FILE as OUTPUT_FILE,
    REPO_RPG_FILE
)
from common import get_project_background_context
from common.language_meta import extract_language_metadata, metadata_with_languages


def load_data_flow() -> Dict[str, Any]:
    """Load data flow configuration if available."""
    if INPUT_DATA_FLOW.exists():
        try:
            with open(INPUT_DATA_FLOW, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ============================================================================
# RPG Update Function
# ============================================================================

def update_rpg_with_base_classes(base_classes_data: Dict[str, Any], rpg_path: Path):
    """Update RPG with newly designed base classes.
    
    Creates File nodes and Class nodes for base classes and mounts them based on scope:
    - "global": directly under repo_node (L0)
    - "<subtree_name>": directly under the specified L1 subtree node
    
    NOTE: Does NOT create intermediate directory nodes. File nodes are mounted
    directly to the scope parent regardless of file path depth.
    
    Args:
        base_classes_data: Result dict containing base_classes list (with scope field)
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
    
    # Cleanup old data first
    rpg.remove_nodes_by_generator("design_base_classes")
    
    base_classes = base_classes_data.get("base_classes", [])
    backend = get_backend(extract_language_metadata(base_classes_data)[0])

    if not base_classes:
        rpg.save_json(str(rpg_path))  # Save to persist cleanup
        return

    added_nodes = 0
    added_edges = 0
    skipped_nodes = 0
    skipped_edges = 0

    # Build L1 subtree name -> node mapping for scope resolution
    subtree_nodes = {}
    for node in rpg.nodes.values():
        if node.level == 1 and node.id != rpg.repo_node.id:
            subtree_nodes[node.name] = node
            # Also add lowercase version for case-insensitive matching
            subtree_nodes[node.name.lower()] = node

    # Index existing file nodes by normalized path
    file_nodes = {}
    for node_id, node in rpg.nodes.items():
        if node.meta and node.meta.type_name == NodeType.FILE and node.meta.path:
            p = str(Path(node.meta.path))
            file_nodes[p] = node

    for bc_file in base_classes:
        file_path = bc_file.get("file_path")
        code = bc_file.get("code", "")
        scope = bc_file.get("scope")
        
        if not file_path or not code or not scope:
            logging.warning("Skipping base class: missing file_path, code, or scope")
            continue
        
        # Determine parent node based on scope
        if scope == "global" or scope.lower() == "global":
            scope_parent = rpg.repo_node
            logging.info(f"Base class file '{file_path}' scope: global (L0)")
        else:
            # Find the L1 subtree node by name - must be exact match
            scope_parent = subtree_nodes.get(scope)
            if not scope_parent:
                # Get unique subtree names (avoid duplicates from lowercase mapping)
                unique_subtrees = sorted(node.name for node in subtree_nodes.values())
                error_msg = (
                    f"ERROR: Scope '{scope}' for base class '{file_path}' does not match any L1 subtree node. "
                    f"Available L1 subtrees: {unique_subtrees}"
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
            else:
                logging.info(f"Base class file '{file_path}' scope: {scope} (L1)")
            
        norm_file_path = str(Path(file_path))
        
        # Build a composite key that includes scope to distinguish files with same path
        # but different scopes (e.g., base.py under different subtrees)
        scope_file_key = f"{scope}::{norm_file_path}"
        file_node = file_nodes.get(scope_file_key)
        
        # If file node does not exist, create it
        if not file_node:
            file_name = Path(file_path).name
            file_id_prefix = f"file_{file_name.replace('.', '_')}"
            
            # Check if file node with same signature exists UNDER THIS SCOPE PARENT
            existing_file = rpg.find_node_by_signature(file_name, file_id_prefix, scope_parent.id)
            if existing_file:
                file_node = existing_file
                file_nodes[scope_file_key] = file_node
            else:
                file_id = f"{file_id_prefix}_{uuid8()}"
                file_node = Node(
                    id=file_id,
                    name=file_name,
                    node_type="feature_group",  # Base class files are feature_group level
                    level=None,  # Will be set by add_edge based on parent
                    meta=NodeMetaData(
                        type_name=NodeType.FILE,
                        path=file_path,
                        description=f"Base Class Definition File (scope: {scope})",
                        generator="design_base_classes"
                    )
                )
                rpg.add_node(file_node)
                
                # Mount file node directly to scope_parent (no intermediate directories)
                rpg.add_edge(scope_parent.id, file_node.id, EdgeType.CONTAINS)
                file_nodes[scope_file_key] = file_node
                added_nodes += 1
        
        # Extract declarations from target-language code.
        class_names = extract_declaration_names(code, backend)
        
        for class_name in class_names:
            # Check if class node with same signature already exists under this file
            class_id_prefix = f"class_{class_name}"
            existing_class = rpg.find_node_by_signature(class_name, class_id_prefix, file_node.id)
            if existing_class:
                skipped_nodes += 1
                logging.info(f"Class node already exists: {class_name}")
                continue
            
            # Create Class Node with canonical RPG path format
            class_id = f"{class_id_prefix}_{uuid8()}"
            class_path = class_node_path(file_path, class_name)
            
            class_node = Node(
                id=class_id,
                name=class_name,
                node_type="feature",  # Base classes are feature level
                level=None,  # Will be set by add_edge based on parent
                meta=NodeMetaData(
                    type_name=NodeType.CLASS,
                    path=class_path,  # Precise path: file::class
                    description=f"Base Class: {class_name} (scope: {scope})",
                    content=code,
                    generator="design_base_classes"
                )
            )
            rpg.add_node(class_node)
            added_nodes += 1
            
            # Check if edge with same signature exists
            edge_src_prefix = strip_uuid8(file_node.id)
            edge_dst_prefix = class_id_prefix
            existing_edge = rpg.find_edge_by_signature(edge_src_prefix, edge_dst_prefix, EdgeType.CONTAINS_BASE_CLASS)
            if existing_edge:
                skipped_edges += 1
                logging.info(f"Edge already exists: {file_node.name} -> {class_name}")
                continue
            
            # Add CONTAINS_BASE_CLASS edge
            rpg.add_edge(
                src=file_node.id,
                dst=class_node.id,
                relation=EdgeType.CONTAINS_BASE_CLASS,
                meta=NodeMetaData(
                    description=f"Defines Base Class (scope: {scope})",
                    generator="design_base_classes"
                )
            )
            added_edges += 1
    
    if added_nodes > 0 or added_edges > 0:
        rpg.save_json(str(rpg_path))
        print(f"[OK] RPG updated: Added {added_nodes} nodes, {added_edges} edges. Skipped {skipped_nodes} nodes, {skipped_edges} edges.")
    else:
        rpg.save_json(str(rpg_path))  # Save to persist cleanup
        print(f"No new base classes added to RPG. Skipped {skipped_nodes} existing nodes.")


# ============================================================================
# Base Class Designer
# ============================================================================

class BaseClassDesigner:
    """Design base classes using BaseClassAgent."""
    
    def __init__(
        self,
        max_iterations: int = 5,
        trajectory: Optional[Trajectory] = None
    ):
        self.max_iterations = max_iterations
        self.trajectory = trajectory
        self.logger = logging.getLogger(__name__)
        self._current_step_id: Optional[int] = None
    
    def build(
        self,
        skeleton: Dict[str, Any],
        data_flow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design base classes from skeleton and data flow context.
        
        Args:
            skeleton: The skeleton.json data
            data_flow: The data_flow.json data
            
        Returns:
            Dict containing base_classes, class_names, etc.
        """
        # Get repository info
        repo_name, repo_info = get_repo_info_from_files()
        primary_language, _ = extract_language_metadata(skeleton)
        if not primary_language:
            primary_language = extract_language_metadata(data_flow)[0]
        
        # Get project background / technology context
        project_background = get_project_background_context()
        
        # Extract functional areas from skeleton
        functional_areas = extract_functional_areas_from_skeleton(skeleton)
        
        # Get hierarchical functional areas overview
        functional_areas_overview = format_functional_graph_overview(skeleton)
        
        # Get skeleton tree for context
        skeleton_tree = get_skeleton_tree_string(skeleton, max_depth=3)
        
        # Get data flow edges
        data_flow_edges = data_flow.get("data_flow", [])
        
        print("\n" + "=" * 70)
        print("BASE CLASS DESIGN")
        print("=" * 70)
        print(f"Repository: {repo_name}")
        print(f"Functional Areas: {len(functional_areas)}")
        print(f"Data Flow Edges: {len(data_flow_edges)}")
        print("=" * 70)
        
        # Record step start
        if self.trajectory:
            step = self.trajectory.add_step(
                "design_base_classes",
                "Design shared base classes and data structures"
            )
            self._current_step_id = step.step_id
            self.trajectory.start_step(step.step_id)
        
        # Initialize agent and run
        agent = BaseClassAgent(
            max_iterations=self.max_iterations,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=self._current_step_id,
            target_language=primary_language,
        )
        
        result = agent.design_base_classes(
            repo_name=repo_name,
            repo_info=repo_info,
            data_flow=data_flow_edges,
            skeleton_tree=skeleton_tree,
            functional_areas=functional_areas,
            functional_areas_overview=functional_areas_overview,
            project_background=project_background,
        )
        result["meta"] = metadata_with_languages(
            skeleton if extract_language_metadata(skeleton)[0] else data_flow
        )
        
        # Update trajectory
        if self.trajectory and self._current_step_id:
            if result.get("success"):
                self.trajectory.complete_step(
                    self._current_step_id,
                    {"class_count": len(result.get("base_classes", []))}
                )
            else:
                self.trajectory.fail_step(
                    self._current_step_id,
                    result.get("error", "Unknown error")
                )
        
        return result
    
    def print_summary(self, result: Dict[str, Any]) -> None:
        """Print summary of base class design."""
        print("\n" + "=" * 60)
        print("BASE CLASS DESIGN SUMMARY")
        print("=" * 60)
        
        base_classes = result.get("base_classes", [])
        backend = get_backend(extract_language_metadata(result)[0])

        class_names = result.get("class_names", [])
        data_structures = result.get("data_structures", [])
        ds_class_names = result.get("data_structure_names", [])
        
        print(f"\nBase Class Files: {len(base_classes)}")
        print(f"Total Base Classes: {len(class_names)}")
        print(f"Data Structure Entries: {len(data_structures)}")
        print(f"Total Data Structures: {len(ds_class_names)}")
        
        if base_classes:
            rows = []
            for bc in base_classes:
                file_path = bc.get("file_path", "")[:40]
                code = bc.get("code", "")
                classes = extract_declaration_names(code, backend)
                class_str = ", ".join(classes[:3])
                if len(classes) > 3:
                    class_str += f" (+{len(classes) - 3})"
                rows.append([file_path, class_str])
            
            print_unicode_table(
                headers=["File Path", "Classes"],
                rows=rows,
                title="Base Class Definitions"
            )
        
        if data_structures:
            rows = []
            for ds in data_structures:
                subtree = ds.get("subtree", "")[:30]
                code = ds.get("code", "")
                classes = extract_declaration_names(code, backend)
                class_str = ", ".join(classes[:3])
                if len(classes) > 3:
                    class_str += f" (+{len(classes) - 3})"
                df_types = ", ".join(ds.get("data_flow_types", [])[:3])
                rows.append([subtree, class_str, df_types])
            
            print_unicode_table(
                headers=["Subtree", "Data Structures", "Covers Data Flow Types"],
                rows=rows,
                title="Data Flow Data Structure Stubs"
            )
        
        uncovered = result.get("uncovered_data_flow_types", [])
        if uncovered:
            print(f"\n[WARNING] Uncovered data flow types: {', '.join(uncovered)}")
        
        if result.get("note"):
            print(f"\nℹ Note: {result['note']}")
        if result.get("error"):
            print(f"\n[WARNING] Error: {result['error']}")
        
        print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Design shared base classes and data structures"
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
        "--output", "-o",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=5,
        help="Max iterations for valid design (default: 5)"
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

    # Load data flow (optional, but recommended)
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

    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("design_base_classes")
        
        if trajectory.is_resumable():
            print(f"\n[WARNING] Found in-progress execution from {trajectory.started_at}")
            print(f"  Resume point: {trajectory.resume_point.step_name}")
            print("  (Use --no-trajectory to start fresh)")
        
        trajectory.start(metadata={
            "skeleton_file": str(skeleton_path),
            "data_flow_file": str(data_flow_path),
            "output_file": str(args.output),
            "max_iterations": args.max_iterations
        })

    try:
        # Design base classes
        designer = BaseClassDesigner(
            max_iterations=args.max_iterations,
            trajectory=trajectory
        )
        
        result = designer.build(skeleton, data_flow)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] Base classes saved to: {output_path}")
        designer.print_summary(result)
        print(f"\n[OK] Base classes saved to: {output_path.name}")

        # Update RPG with base classes
        if result.get("success", True):
            update_rpg_with_base_classes(result, REPO_RPG_FILE)

        if not result.get("success", True) and "error" in result:
            if trajectory:
                trajectory.fail(result["error"])
            return 1
        
        # Mark trajectory as complete
        if trajectory:
            trajectory.complete(metadata={
                "base_class_files": len(result.get("base_classes", [])),
                "class_names": result.get("class_names", []),
                "data_structure_files": len(result.get("data_structures", [])),
                "data_structure_names": result.get("data_structure_names", []),
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
