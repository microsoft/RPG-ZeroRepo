#!/usr/bin/env python3
"""Func Designer - Unified Entry Point.

This module provides a unified interface for running the complete
func_design workflow:
1. Data Flow Design - Design inter-component data flow (DAG)
2. Base Class Design - Design shared base classes and data structures
3. Interface Design - Design function/class interfaces for each file

Can be run as a complete pipeline or individual phases.
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.trajectory import Trajectory, load_or_create_trajectory
from common.language_meta import extract_language_metadata
from common import (
    get_skeleton_tree_string,
    extract_functional_areas_from_skeleton,
    format_functional_graph_overview,
    extract_component_directories,
)

# Import agents
from .data_flow_agent import DataFlowAgent
from .base_class_agent import BaseClassAgent
from .interface_agent import (
    InterfaceOrchestrator
)

# Import centralized paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.paths import (
    SKELETON_FILE as INPUT_SKELETON,
    DATA_FLOW_FILE as OUTPUT_DATA_FLOW,
    BASE_CLASSES_FILE as OUTPUT_BASE_CLASSES,
    INTERFACES_FILE as OUTPUT_INTERFACES,
)


# ============================================================================
# Func Designer
# ============================================================================

class FuncDesigner:
    """Main orchestrator for func_design workflow.
    
    Manages three phases:
    1. Data Flow: Design inter-component data flow as a DAG
    2. Base Classes: Design shared base classes and data structures
    3. Interfaces: Design function/class interfaces for each file
    """
    
    def __init__(
        self,
        max_data_flow_iterations: int = 5,
        max_base_class_iterations: int = 5,
        max_interface_iterations: int = 10,
        trajectory: Optional[Trajectory] = None
    ):
        self.max_data_flow_iterations = max_data_flow_iterations
        self.max_base_class_iterations = max_base_class_iterations
        self.max_interface_iterations = max_interface_iterations
        self.trajectory = trajectory
        self.logger = logging.getLogger(__name__)
        self._current_step_id: Optional[int] = None
        
        # State
        self.skeleton = None
        self.data_flow = None
        self.base_classes = None
        self.interfaces = None
        self.repo_name = ""
        self.repo_info = ""
    
    def load_skeleton(self, skeleton_path: Path = INPUT_SKELETON) -> bool:
        """Load skeleton from file."""
        if not skeleton_path.exists():
            self.logger.error(f"Skeleton file not found: {skeleton_path}")
            return False
        
        try:
            with open(skeleton_path, "r", encoding="utf-8") as f:
                self.skeleton = json.load(f)
            
            self.repo_name = self.skeleton.get("repository_name", "project")
            self.repo_info = self.skeleton.get("repository_purpose", "")
            
            self.logger.info(f"Loaded skeleton for {self.repo_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load skeleton: {e}")
            return False
    
    def run_data_flow_phase(self) -> Dict[str, Any]:
        """Run data flow design phase."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: DATA FLOW DESIGN")
        self.logger.info("=" * 70)
        
        if not self.skeleton:
            return {"success": False, "error": "Skeleton not loaded"}
        
        # Extract functional areas
        functional_areas = extract_functional_areas_from_skeleton(self.skeleton)
        component_dirs = extract_component_directories(self.skeleton)
        skeleton_tree = get_skeleton_tree_string(self.skeleton)
        
        self.logger.info(f"Found {len(functional_areas)} functional areas")
        
        if len(functional_areas) < 2:
            self.logger.warning("Less than 2 components, skipping data flow")
            self.data_flow = {
                "data_flow": [],
                "subtree_order": functional_areas,
                "components": functional_areas,
                "success": True
            }
            return self.data_flow
        
        # Add step to trajectory
        step_id = None
        if self.trajectory:
            step = self.trajectory.add_step("data_flow_design", "Design inter-component data flow")
            step_id = step.step_id
            self._current_step_id = step_id
            self.trajectory.start_step(step_id)
        
        # Initialize agent with trajectory
        agent = DataFlowAgent(
            max_iterations=self.max_data_flow_iterations,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=step_id
        )
        
        # Run
        result = agent.build_data_flow(
            repo_name=self.repo_name,
            repo_info=self.repo_info,
            functional_areas=functional_areas,
            component_dirs=component_dirs,
            skeleton_tree=skeleton_tree
        )
        
        result["components"] = functional_areas
        self.data_flow = result
        
        # Complete trajectory step
        if self.trajectory and step_id:
            if result.get("success"):
                self.trajectory.complete_step(step_id, {"edge_count": len(result.get("data_flow", []))})
            else:
                self.trajectory.fail_step(step_id, result.get("error", "Data flow design failed"))
        
        # Save
        OUTPUT_DATA_FLOW.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DATA_FLOW, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data flow saved to {OUTPUT_DATA_FLOW}")
        return result
    
    def run_base_class_phase(self) -> Dict[str, Any]:
        """Run base class design phase."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: BASE CLASS DESIGN")
        self.logger.info("=" * 70)
        
        if not self.skeleton:
            return {"success": False, "error": "Skeleton not loaded"}
        
        if not self.data_flow:
            # Try to load from file
            if OUTPUT_DATA_FLOW.exists():
                with open(OUTPUT_DATA_FLOW, "r", encoding="utf-8") as f:
                    self.data_flow = json.load(f)
            else:
                return {"success": False, "error": "Data flow not available"}
        
        # Extract info
        functional_areas = extract_functional_areas_from_skeleton(self.skeleton)
        functional_areas_overview = format_functional_graph_overview(self.skeleton)
        skeleton_tree = get_skeleton_tree_string(self.skeleton)
        data_flow_edges = self.data_flow.get("data_flow", [])
        
        # Add step to trajectory
        step_id = None
        if self.trajectory:
            step = self.trajectory.add_step("base_class_design", "Design shared base classes")
            step_id = step.step_id
            self._current_step_id = step_id
            self.trajectory.start_step(step_id)
        
        # Initialize agent with trajectory
        agent = BaseClassAgent(
            max_iterations=self.max_base_class_iterations,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=step_id
        )
        
        # Run
        result = agent.design_base_classes(
            repo_name=self.repo_name,
            repo_info=self.repo_info,
            data_flow=data_flow_edges,
            skeleton_tree=skeleton_tree,
            functional_areas=functional_areas,
            functional_areas_overview=functional_areas_overview
        )
        
        self.base_classes = result
        
        # Complete trajectory step
        if self.trajectory and step_id:
            if result.get("success"):
                self.trajectory.complete_step(step_id, {"class_count": len(result.get("base_classes", []))})
            else:
                self.trajectory.fail_step(step_id, result.get("error", "Base class design failed"))
        
        # Save
        OUTPUT_BASE_CLASSES.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_BASE_CLASSES, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Base classes saved to {OUTPUT_BASE_CLASSES}")
        return result
    
    def run_interface_phase(self) -> Dict[str, Any]:
        """Run interface design phase."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: INTERFACE DESIGN")
        self.logger.info("=" * 70)
        
        if not self.skeleton:
            return {"success": False, "error": "Skeleton not loaded"}
        
        # Load data flow if not available
        if not self.data_flow:
            if OUTPUT_DATA_FLOW.exists():
                with open(OUTPUT_DATA_FLOW, "r", encoding="utf-8") as f:
                    self.data_flow = json.load(f)
            else:
                return {"success": False, "error": "Data flow not available"}
        
        # Load base classes if not available
        if not self.base_classes:
            if OUTPUT_BASE_CLASSES.exists():
                with open(OUTPUT_BASE_CLASSES, "r", encoding="utf-8") as f:
                    self.base_classes = json.load(f)
            else:
                self.base_classes = {"base_classes": []}
        
        # Get base classes list
        base_classes_list = self.base_classes.get("base_classes", [])

        primary_language, _ = extract_language_metadata(self.skeleton)
        if not primary_language:
            primary_language = extract_language_metadata(self.data_flow)[0]
        if not primary_language:
            primary_language = extract_language_metadata(self.base_classes)[0]

        # Add step to trajectory
        step_id = None
        if self.trajectory:
            step = self.trajectory.add_step("interface_design", "Design function/class interfaces")
            step_id = step.step_id
            self._current_step_id = step_id
            self.trajectory.start_step(step_id)
        
        # Initialize orchestrator with trajectory
        orchestrator = InterfaceOrchestrator(
            max_file_iterations=self.max_interface_iterations,
            logger=self.logger,
            trajectory=self.trajectory,
            step_id=step_id,
            target_language=primary_language,
        )
        
        # Run
        result = orchestrator.design_all_interfaces(
            skeleton=self.skeleton,
            data_flow=self.data_flow,
            base_classes=base_classes_list,
            repo_info=self.repo_info
        )
        
        self.interfaces = result
        
        # Complete trajectory step
        if self.trajectory and step_id:
            if result.get("success"):
                self.trajectory.complete_step(step_id, {"interface_count": len(result.get("subtrees", {}))})
            else:
                self.trajectory.fail_step(step_id, result.get("error", "Interface design failed"))
        
        # Save
        OUTPUT_INTERFACES.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_INTERFACES, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Interfaces saved to {OUTPUT_INTERFACES}")
        return result
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete func_design pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("FUNC DESIGNER - FULL PIPELINE")
        self.logger.info("=" * 70)
        
        results = {
            "data_flow_phase": None,
            "base_classes_phase": None,
            "interfaces_phase": None,
            "success": True
        }
        
        # Data-flow design step.
        data_flow_result = self.run_data_flow_phase()
        results["data_flow_phase"] = data_flow_result
        
        if not data_flow_result.get("success", False):
            self.logger.warning("Data flow phase had issues, continuing...")
        
        # Base-class design step.
        base_class_result = self.run_base_class_phase()
        results["base_classes_phase"] = base_class_result
        
        if not base_class_result.get("success", False):
            self.logger.warning("Base class phase had issues, continuing...")
        
        # Interface design step.
        interface_result = self.run_interface_phase()
        results["interfaces_phase"] = interface_result
        
        if not interface_result.get("success", False):
            self.logger.warning("Interface phase had issues")
            results["success"] = False
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of all phases."""
        print("\n" + "=" * 70)
        print("FUNC DESIGNER - SUMMARY")
        print("=" * 70)
        
        # Data Flow
        df = results.get("data_flow_phase", {})
        df_status = "[OK]" if df.get("success") else "[FAIL]"
        df_edges = len(df.get("data_flow", []))
        print(f"\n[{df_status}] Data Flow: {df_edges} edges")
        
        # Base Classes
        bc = results.get("base_classes_phase", {})
        bc_status = "[OK]" if bc.get("success") else "[FAIL]"
        bc_count = len(bc.get("base_classes", []))
        print(f"[{bc_status}] Base Classes: {bc_count} files")
        
        # Interfaces
        itf = results.get("interfaces_phase", {})
        itf_status = "[OK]" if itf.get("success") else "[FAIL]"
        subtrees = itf.get("subtrees", {})
        # Support both "interfaces" (reference format) and "files" (old format)
        total_files = sum(
            len(st.get("interfaces", st.get("files", {}))) 
            for st in subtrees.values()
        )
        success_files = sum(
            sum(1 for f in st.get("interfaces", st.get("files", {})).values() if f.get("units"))
            for st in subtrees.values()
        )
        print(f"[{itf_status}] Interfaces: {success_files}/{total_files} files")
        
        print("\n" + "=" * 70)
        
        if results.get("success"):
            print("[OK] All phases completed successfully!")
        else:
            print("[WARNING] Some phases had issues. Check logs for details.")
        
        print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run func_design workflow (data flow, base classes, interfaces)"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "data_flow", "base_classes", "interfaces"],
        default="all",
        help="Which phase(s) to run (default: all)"
    )
    parser.add_argument(
        "--skeleton", "-s",
        type=str,
        default=str(INPUT_SKELETON),
        help=f"Skeleton input file (default: {INPUT_SKELETON})"
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=5,
        help="Max iterations per phase (default: 5)"
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

    # Initialize trajectory
    trajectory = None
    if not args.no_trajectory:
        trajectory = load_or_create_trajectory("func_designer")
        trajectory.start(metadata={
            "phase": args.phase,
            "skeleton_file": args.skeleton,
            "max_iterations": args.max_iterations
        })

    try:
        # Initialize designer
        designer = FuncDesigner(
            max_data_flow_iterations=args.max_iterations,
            max_base_class_iterations=args.max_iterations,
            max_interface_iterations=args.max_iterations * 2,
            trajectory=trajectory
        )
        
        # Load skeleton
        if not designer.load_skeleton(Path(args.skeleton)):
            print(f"ERROR: Could not load skeleton from {args.skeleton}")
            return 1
        
        # Run appropriate phase(s)
        if args.phase == "all":
            result = designer.run_full_pipeline()
        elif args.phase == "data_flow":
            result = designer.run_data_flow_phase()
        elif args.phase == "base_classes":
            result = designer.run_base_class_phase()
        elif args.phase == "interfaces":
            result = designer.run_interface_phase()
        else:
            print(f"Unknown phase: {args.phase}")
            return 1
        
        # Check result
        if not result.get("success", False):
            if trajectory:
                trajectory.fail(result.get("error", "Phase failed"))
            return 1
        
        # Mark trajectory as complete
        if trajectory:
            trajectory.complete(metadata={"phase": args.phase})
        
        return 0
        
    except Exception as e:
        logger.error(f"Func designer failed: {e}")
        if trajectory:
            trajectory.fail(str(e))
        raise


if __name__ == "__main__":
    exit(main())
