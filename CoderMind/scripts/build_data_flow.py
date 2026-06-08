#!/usr/bin/env python3
"""Build Data Flow Script - Implementation Level Step 2.

Function: Design inter-component data flow as a directed acyclic graph (DAG)
- Reads skeleton.json to get component information
- Designs how data flows between components (what data, types, transformations)
- Validates the data flow graph is acyclic
- Generates subtree processing order for later steps
- Adds data flow dependencies as edges to repo_rpg.json

Input: .cmind/skeleton.json (file structure with component info)
Output: .cmind/data_flow.json (data flow edges and subtree order)
        .cmind/repo_rpg.json (updated with data flow edges)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import trajectory module
from common.trajectory import Trajectory, load_or_create_trajectory
from common import (
    get_skeleton_tree_string,
    extract_functional_areas_from_skeleton,
    extract_component_directories,
    print_unicode_table,
    get_repo_info_from_files,
)

# Import the DataFlowAgent
from func_design.data_flow_agent import DataFlowAgent

# Import RPG models for adding edges to repo_rpg.json
from rpg import EdgeType

# Import centralized paths
from common.paths import SKELETON_FILE, DATA_FLOW_FILE, REPO_RPG_FILE
from common import get_project_background_context
from common.language_meta import extract_language_metadata, metadata_with_languages


# ============================================================================
# RPG Update Function
# ============================================================================

def update_rpg_with_data_flow(data_flow_data: Dict[str, Any], rpg_path: Path):
    """Update RPG with data flow edges.
    
    Args:
        data_flow_data: Result dict containing data_flow list
        rpg_path: Path to the repo_rpg.json file
    """
    if not rpg_path.exists():
        logging.info(f"Skipping repo_rpg.json update: file not found at {rpg_path}")
        return
    
    data_flow = data_flow_data.get("data_flow", [])
    if not data_flow:
        return

    from rpg.service import RPGService

    try:
        svc = RPGService.load(rpg_path)
    except Exception as e:
        logging.error(f"Failed to load RPG: {e}")
        return
    
    # Cleanup old edges first
    svc.refresh_stage_edges("build_data_flow")

    added = 0
    
    for edge_data in data_flow:
        source_name = edge_data.get("source", "")
        target_name = edge_data.get("target", "")
        data_id = edge_data.get("data_id", "")
        data_type = edge_data.get("data_type", "")
        
        # Find source node
        src_node = svc.find_functional_area_by_name(source_name)
        if not src_node:
            logging.warning(f"Source component not found: {source_name}")
            continue
        
        # Find target node
        dst_node = svc.find_functional_area_by_name(target_name)
        if not dst_node:
            logging.warning(f"Target component not found: {target_name}")
            continue
        
        # Add data flow edge with dedup
        was_added = svc.add_dependency_edge(
            src_node, dst_node,
            EdgeType.REFERENCES,
            "build_data_flow",
            description=edge_data.get("transformation", ""),
            content=f"data_id={data_id}, data_type={data_type}",
        )
        if was_added:
            added += 1
            logging.info(f"Added data flow edge: {source_name} -> {target_name} ({data_id})")
        else:
            logging.info(f"Edge already exists (by signature): {source_name} -> {target_name}")
    
    svc.save(rpg_path)
    if added > 0:
        print(f"[OK] Added {added} data flow edges to: {rpg_path.name}")
    else:
        print(f"No new data flow edges to add to: {rpg_path.name}")


# ============================================================================
# Data Flow Builder
# ============================================================================

class DataFlowBuilder:
    """Build data flow using DataFlowAgent."""
    
    def __init__(
        self,
        max_iterations: int = 5,
        trajectory: Optional[Trajectory] = None
    ):
        self.max_iterations = max_iterations
        self.trajectory = trajectory
        self.logger = logging.getLogger(__name__)
        self._current_step_id: Optional[int] = None
    
    def build(self, skeleton: Dict[str, Any]) -> Dict[str, Any]:
        """Build data flow from skeleton.
        
        Args:
            skeleton: The skeleton.json data
            
        Returns:
            Dict containing data_flow, subtree_order, components, etc.
        """
        # Get repository info
        repo_name, repo_info = get_repo_info_from_files()
        primary_language, _ = extract_language_metadata(skeleton)
        
        # Enrich repo_info with project background / technology context
        project_background = get_project_background_context()
        if project_background and project_background.strip():
            repo_info = f"{repo_info}\n\n{project_background}"
        
        # Extract functional areas (components) from skeleton
        functional_areas = extract_functional_areas_from_skeleton(skeleton)
        component_dirs = extract_component_directories(skeleton)
        
        if len(functional_areas) < 2:
            self.logger.warning("Less than 2 components found, skipping data flow design")
            return {
                "data_flow": [],
                "subtree_order": functional_areas,
                "components": functional_areas,
                "warning": "Not enough components for data flow"
            }
        
        print("\n" + "=" * 70)
        print("DATA FLOW DESIGN")
        print("=" * 70)
        print(f"Repository: {repo_name}")
        print(f"Components: {len(functional_areas)}")
        for area in functional_areas:
            dir_info = f" [{component_dirs.get(area, '')}]" if area in component_dirs else ""
            print(f"  -  {area}{dir_info}")
        print("=" * 70)
        
        # Record step start
        if self.trajectory:
            step = self.trajectory.add_step(
                "design_data_flow",
                f"Design data flow for {len(functional_areas)} components"
            )
            self._current_step_id = step.step_id
            self.trajectory.start_step(step.step_id)
        
        # Get skeleton tree for context
        skeleton_tree = get_skeleton_tree_string(skeleton, max_depth=3)
        
        # Initialize agent and run
        agent = DataFlowAgent(
            max_iterations=self.max_iterations,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=self._current_step_id,
            target_language=primary_language,
        )
        
        result = agent.build_data_flow(
            repo_name=repo_name,
            repo_info=repo_info,
            functional_areas=functional_areas,
            component_dirs=component_dirs,
            skeleton_tree=skeleton_tree
        )
        
        # Add components to result
        result["components"] = functional_areas
        result["meta"] = metadata_with_languages(skeleton)
        
        # Update trajectory
        if self.trajectory and self._current_step_id:
            if result.get("success"):
                self.trajectory.complete_step(
                    self._current_step_id,
                    {"edge_count": len(result.get("data_flow", []))}
                )
            else:
                self.trajectory.fail_step(
                    self._current_step_id,
                    result.get("error", "Unknown error")
                )
        
        return result
    
    def print_summary(self, result: Dict[str, Any]) -> None:
        """Print summary of data flow design."""
        print("\n" + "=" * 60)
        print("DATA FLOW DESIGN SUMMARY")
        print("=" * 60)
        
        components = result.get("components", [])
        data_flow = result.get("data_flow", [])
        subtree_order = result.get("subtree_order", [])
        
        print(f"\nComponents: {len(components)}")
        print(f"Data Flow Edges: {len(data_flow)}")
        
        if subtree_order:
            print(f"\nSubtree Processing Order:")
            for i, comp in enumerate(subtree_order, 1):
                print(f"  {i}. {comp}")
        
        if data_flow:
            rows = []
            for edge in data_flow:
                source = str(edge.get("source", ""))[:20]
                target = str(edge.get("target", ""))[:20]
                data_id = str(edge.get("data_id", ""))[:25]
                data_type = str(edge.get("data_type", ""))[:20]
                rows.append([source, "→", target, data_id, data_type])
            
            print_unicode_table(
                headers=["Source", "", "Target", "Data ID", "Data Type"],
                rows=rows,
                title="Data Flow Edges"
            )
        
        if result.get("error"):
            print(f"\n[WARNING] Error: {result['error']}")
        if result.get("warning"):
            print(f"\n[WARNING] Warning: {result['warning']}")
        
        print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build inter-component data flow graph"
    )
    parser.add_argument(
        "--skeleton", "-s",
        type=str,
        default=str(SKELETON_FILE),
        help=f"Input skeleton file (default: {SKELETON_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DATA_FLOW_FILE),
        help=f"Output data flow file (default: {DATA_FLOW_FILE})"
    )
    parser.add_argument(
        "--repo-rpg", "-r",
        type=str,
        default=str(REPO_RPG_FILE),
        help=f"Repo RPG file to update with data flow edges (default: {REPO_RPG_FILE})"
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

    # Load input
    input_path = Path(args.skeleton)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        print(f"ERROR: Input file not found: {input_path}")
        print("Please run /cmind.build_skeleton first.")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        skeleton = json.load(f)

    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("build_data_flow")
        
        if trajectory.is_resumable():
            print(f"\n[WARNING] Found in-progress execution from {trajectory.started_at}")
            print(f"  Resume point: {trajectory.resume_point.step_name}")
            print("  (Use --no-trajectory to start fresh)")
        
        trajectory.start(metadata={
            "input_file": str(input_path),
            "output_file": str(args.output),
            "max_iterations": args.max_iterations
        })

    try:
        # Build data flow
        builder = DataFlowBuilder(
            max_iterations=args.max_iterations,
            trajectory=trajectory
        )
        
        result = builder.build(skeleton)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] Data flow saved to: {output_path}")
        builder.print_summary(result)
        print(f"\n[OK] Data flow saved to: {output_path.name}")

        # Add data flow edges to repo_rpg.json
        update_rpg_with_data_flow(result, Path(args.repo_rpg))

        if not result.get("success", True) and "error" in result:
            if trajectory:
                trajectory.fail(result["error"])
            return 1
        
        # Mark trajectory as complete
        if trajectory:
            trajectory.complete(metadata={
                "components": len(result.get("components", [])),
                "edges": len(result.get("data_flow", []))
            })
            print(f"[OK] Trajectory saved to: {trajectory.trajectory_file.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        if trajectory:
            trajectory.fail(str(e))
        raise


if __name__ == "__main__":
    exit(main())
