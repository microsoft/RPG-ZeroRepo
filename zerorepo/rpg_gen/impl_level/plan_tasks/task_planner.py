"""
Task Planning Agent for organizing implementation tasks
Based on the agent pattern from graph/refactored_agents
"""
import logging
import json
import uuid
from typing import Union, Dict, Optional, List, Any
from pydantic import BaseModel, Field
from collections import Counter
from datetime import datetime
from zerorepo.rpg_gen.base import (
    RPG,
    Tool, ToolCall,
    ToolExecResult, ToolCallArguments,
    LLMClient, LLMConfig, Memory,
    BaseAgent, BaseEnv
)
from .prompts.task_planner import PLAN_TASK_BATCHES_TOOL, BATCH_PLANNER
from .task_batch import TaskBatch


# ========================
# 2. AGENT TOOL
# ========================

class PlanTaskBatchesTool(Tool):
    name = "plan_task_batches"
    description = PLAN_TASK_BATCHES_TOOL
    
    class ParamModel(BaseModel):
        batches: List[Dict] = Field(..., description="List of task batches for this file")
    
    @classmethod
    def custom_parse(cls, raw: str) -> List[ToolCallArguments]:
        """Parse raw JSON input for task batches"""
        valid_tools = []
        
        try:
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`")
                raw = raw.replace("```json", "").replace("```", "").strip()
                
                raw_tools = json.loads(raw)
                raw_tools = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        except json.JSONDecodeError as e:
            logging.error(f"[custom_parse] Invalid JSON: {e}")
            return valid_tools
        
        for raw_tool in raw_tools:
            try:
                tool_name = raw_tool.get("tool_name", "")
                if tool_name.lower().strip() != cls.name.lower():
                    continue
                
                params = raw_tool.get("parameters", raw_tool)
                parsed = cls.ParamModel(**params).model_dump()
                valid_tools.append(parsed)
            except Exception as e:
                logging.error(f"[custom_parse] Error: {e}")
                continue
        
        return valid_tools
    
    @classmethod
    async def execute(cls, arguments: Dict, env: Optional[Any] = None) -> ToolExecResult:
        try:
            file_unit_keys = list(env.units_to_code.keys())
            batches = arguments.get("batches", [])

            if not batches:
                return ToolExecResult(
                    output=None,
                    error="Invalid arguments: 'batches' is missing or empty. You must provide a complete list of ALL task batches for this file, not partial updates. Re-plan all units from scratch.",
                    error_code=1,
                    state=None
                )

            # Validate batch structure + collect unit keys
            all_unit_keys: List[str] = []

            for i, batch in enumerate(batches):
                required_keys = ["units", "task"]
                missing = [k for k in required_keys if k not in batch]
                if missing:
                    return ToolExecResult(
                        output=None,
                        error=f"Invalid batch at index {i}: missing required keys {missing}. Each batch must include: {required_keys}. You must re-plan ALL batches for this file, not just fix the problematic ones. Start over with a complete plan.",
                        error_code=1,
                        state=None
                    )

                batch_units = batch["units"]
                if not isinstance(batch_units, list) or not batch_units:
                    return ToolExecResult(
                        output=None,
                        error=f"Invalid batch at index {i}: 'units' must be a non-empty list of unit keys. You must re-plan ALL batches for this file with correct format. Each batch should have: {{'units': [...], 'task': '...'}}",
                        error_code=1,
                        state=None
                    )

                all_unit_keys.extend(batch_units)

            expected = set(file_unit_keys)
            got = set(all_unit_keys)

            # Detect duplicates across all batches in current call (same unit key appears multiple times)
            counter = Counter(all_unit_keys)
            duplicates = sorted([k for k, c in counter.items() if c > 1])

            # Reject duplicates within current call
            if duplicates:
                return ToolExecResult(
                    output=None,
                    error=(
                        "Duplicate unit keys detected across batches. "
                        f"Each unit key should appear exactly once. Duplicates: {duplicates}.\n\n"
                        "IMPORTANT: You must re-plan ALL batches for this file from scratch. "
                        "Ensure each unit appears in exactly one batch. Start with a fresh complete plan."
                    ),
                    error_code=1,
                    state=None
                )

            # Check if this is not a complete re-planning (we require complete re-planning each time)
            if got != expected:
                missing = sorted(list(expected - got))
                extra = sorted(list(got - expected))
                return ToolExecResult(
                    output=None,
                    error=(
                        "Unit key mismatch: expected "
                        f"{len(expected)} unique unit keys from units in file, got {len(got)} unique keys from batches.\n"
                        f"Missing units: {missing}\n"
                        f"Extra units: {extra}\n\n"
                        "IMPORTANT: You must re-plan ALL batches for this file from scratch. Do not try to add only the missing units. "
                        "Create a complete new plan that includes ALL units: " + str(file_unit_keys)
                    ),
                    error_code=1,
                    state=None
                )

            # All units must be covered - success case
            return ToolExecResult(
                output=f"Planned {len(batches)} batches covering all {len(expected)} file units.",
                error=None,
                error_code=0,
                state={"batches": batches}
            )
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=f"Unexpected error while planning task batches: {type(e).__name__}: {e}",
                error_code=1,
                state={"batches": arguments.get("batches", [])}
            )


# ========================  
# 3. ENVIRONMENT
# ========================

class TaskPlanningEnv(BaseEnv):
    """Environment for task planning phase"""

    def __init__(
        self,
        cur_subtree: str,
        repo_rpg: RPG,
        file_path: str,
        units_to_feature: Dict[str, List],
        units_to_code: Dict[str, str],
        register_tools: List = [PlanTaskBatchesTool],
        logger: logging.Logger = None
    ):
        super().__init__(
            register_tools,
            logger
        )

        self.repo_rpg=repo_rpg
        self.cur_subtree = cur_subtree
        self.file_path = file_path
        self.units_to_feature = units_to_feature
        self.units_to_code = units_to_code

        self.planned_batches = []
        self.is_completed = False
        self.logger = logger or logging.getLogger(__name__)

    def should_terminate(self, tool_calls, tool_results):
        """Check if planning is complete"""
        # If already completed, remain terminated
        if self.is_completed:
            return True
            
        # Check if there are successful results
        if tool_results and all(result.success for result in tool_results):
            # Check if it's the plan_task_batches tool
            for tool_call, result in zip(tool_calls, tool_results):
                if tool_call.name == "plan_task_batches" and result.success:
                    # The actual completion will be set in update_env
                    # after validating the batches are complete
                    pass
        
        return self.is_completed

    def update_env(self, tool_call, result):
        """Update environment after tool execution"""
        if tool_call.name == "plan_task_batches" and result.success:
            try:
                batches_data = result.state.get("batches", [])
                
                # Convert to TaskBatch objects
                batches = []
                for idx, batch_data in enumerate(batches_data):
                    task = batch_data["task"]
                    b_unit_keys = batch_data["units"]
                    b_unit_keys = b_unit_keys if isinstance(
                        b_unit_keys, list
                    ) else [b_unit_keys]
        
                    b_unit_to_code = {
                        unit: self.units_to_code[unit] for unit in b_unit_keys
                    }
                    b_unit_to_features = {
                        unit: self.units_to_feature[unit] for unit in b_unit_keys
                    }
                    
                    batch = TaskBatch(
                        task=task,
                        file_path=self.file_path,
                        units_key=b_unit_keys,
                        unit_to_code=b_unit_to_code,
                        unit_to_features=b_unit_to_features,
                        priority=idx,
                        subtree=self.cur_subtree
                    )
                    batches.append(batch)
                
                # Only update planned_batches if the planning is complete and correct
                # The tool execution already validated that all units are covered
                self.planned_batches = batches
                self.is_completed = True  # Mark as completed immediately after successful planning
                self.logger.info(f"Successfully planned {len(batches)} batches for {self.file_path}")
                
            except Exception as e:
                # Don't save partial results on error
                self.logger.error(f"Failed to update planning: {e}")
                self.planned_batches = []
                self.is_completed = False


# ========================
# 4. AGENT
# ========================

class TaskPlannerAgent(BaseAgent):
    """Agent for planning implementation tasks"""
    
    def __init__(
        self,
        llm_cfg: Union[str, Dict, LLMConfig],
        file_path: str,
        repo_rpg: RPG,
        cur_subtree: str,
        units_to_feature: Dict[str, List],
        units_to_code: Dict[str, str],
        system_prompt: str =BATCH_PLANNER,
        context_window: int = 30,
        register_tools: Optional[List[Tool]] = [PlanTaskBatchesTool],
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        super().__init__(
            llm_cfg=llm_cfg,
            system_prompt=system_prompt,
            context_window=context_window,
            register_tools=register_tools,
            logger=logger,
            **kwargs
        )
        self.repo_rpg = repo_rpg
        
        self._env = TaskPlanningEnv(
            repo_rpg=repo_rpg,
            cur_subtree=cur_subtree,
            file_path=file_path,
            units_to_feature=units_to_feature,
            units_to_code=units_to_code,
            register_tools=register_tools,
            logger=logger
        )        
        
    def plan_all_tasks(
        self,
        file_code: str,
        max_retry: int = 3,
        max_steps: int = 20
    ) -> Dict:
        """Plan tasks for all files in the repository"""
        
        repo_name = self.repo_rpg.repo_name
        repo_info = self.repo_rpg.repo_info
        
        # Build comprehensive task description
        task = (
            f"Plan the implementation tasks for the repository: {repo_name}\n"
            f"Repository description: {repo_info}\n\n"
            f"Context:\n"
            f"- You are planning the implementation order for the file: {self._env.file_path}\n"
            f"- Functional area / subtree: {self._env.cur_subtree}\n\n"
            f"Source code of the file:\n"
            f"{file_code}\n\n"
            f"Units to plan (to be implemented):\n"
            f"{json.dumps(list(self._env.units_to_code.keys()), indent=2)}"
        )
        
        result = self.run(
            task=task,
            max_error_times=max_retry,
            max_steps=max_steps
        )
        
        return {
            "planned_batches": self._env.planned_batches,
            "success": self._env.is_completed,
            "agent_result": result
        }


# ========================
# 5. LEGACY INTERFACE  
# ========================

class TaskPlanner:
    """Legacy interface wrapping the new agent-based planner"""
    
    def __init__(
        self,
        rpg: RPG,
        llm_cfg: Union[Dict, LLMConfig],
        graph_data: Union[str, Dict],
        logger: logging.Logger
    ):
        self.repo_rpg = rpg
        
        if isinstance(graph_data, str):
            with open(graph_data, 'r') as f:
                graph_data = json.load(f)
        
        self.llm_cfg = llm_cfg        
        self.graph_data = graph_data

        self.logger = logger
        self.planned_batches_dict = {}
        self.agent_results_dict = {}
    
    @staticmethod
    def load_batches_from_dict(
        planned_batches_dict: Dict, 
        graph_data: Dict
    ) -> List[TaskBatch]:
        """
        Utility function to convert planned_batches_dict to a properly ordered list of TaskBatch objects.
        (Deprecated: Use zerorepo.utils.task_loader.load_batches_from_dict instead)
        """
        from ....utils.task_loader import load_batches_from_dict
        return load_batches_from_dict(planned_batches_dict, graph_data)

    def plan_all_batches(self, result_path) -> List[TaskBatch]:
        """Plan all implementation batches"""
        subtree_order = self.graph_data["data_flow_phase"]["subtree_order"]
        
        planned_batches = {}
        agent_results = {}
        
        subtrees_data = self.graph_data['interfaces_phase']["interfaces"]["subtrees"]
        for subtree in subtree_order:
            if subtree not in subtrees_data:
                continue
                
            subtree_dict = subtrees_data[subtree]
            planned_batches[subtree] = {}
            agent_results[subtree] = {}
            
            subtree_interfaces = subtree_dict["interfaces"]
            files_order = subtree_dict["files_order"]
            
            for file_path in files_order:
                if not file_path in subtree_interfaces:
                    continue
                file_dict = subtree_interfaces[file_path]
                file_code = file_dict["file_code"]
                units_to_code = file_dict["units_to_code"]
                units_to_feature = file_dict["units_to_features"]
                
                plan_agent = TaskPlannerAgent(
                    llm_cfg=self.llm_cfg,
                    file_path=file_path,
                    repo_rpg=self.repo_rpg,
                    cur_subtree=subtree,
                    units_to_feature=units_to_feature,
                    units_to_code=units_to_code,
                    logger=self.logger
                )
                
                results = plan_agent.plan_all_tasks(
                    file_code=file_code,
                    max_retry=5,
                    max_steps=5
                )
                assert results["success"]

                file_plan_batch = results["planned_batches"]
                file_agent_result = results["agent_result"]
                
                planned_batches[subtree][file_path] = file_plan_batch
                agent_results[subtree][file_path] = file_agent_result
        
        # Add integration tests and documentation batches
        self._add_special_batches(planned_batches, agent_results)
                
        # Convert TaskBatch objects to dictionaries for JSON serialization
        planned_batches_serializable = {}
        for subtree, files_dict in planned_batches.items():
            planned_batches_serializable[subtree] = {}
            for file_path, batches_list in files_dict.items():
                # Convert list of TaskBatch objects to list of dictionaries
                planned_batches_serializable[subtree][file_path] = [
                    batch.to_dict() for batch in batches_list
                ]
        
        # Save results
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                "planned_batches_dict": planned_batches_serializable,
                "agent_results_dict": agent_results,
                "success": True
            }, f, indent=2)
            
        self.planned_batches_dict = planned_batches
        self.agent_results_dict = agent_results
        self.logger.info(f"Planned batches and agent results saved to {result_path}")
        
        return self.get_batches_as_list()
    
    def get_batches_as_dict(self) -> Dict[str, Dict[str, List[TaskBatch]]]:
        """Return batches organized by subtree and file path (dictionary version)"""
        if not hasattr(self, 'planned_batches_dict'):
            raise RuntimeError("Must call plan_all_batches first")
        return self.planned_batches_dict
    
    def get_agent_results_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return agent results organized by subtree and file path (dictionary version)"""
        if not hasattr(self, 'agent_results_dict'):
            raise RuntimeError("Must call plan_all_batches first")
        return self.agent_results_dict
    
    def get_batches_as_list(self) -> List[TaskBatch]:
        """Return all batches in a flattened list with proper ordering (list version)"""
        if not hasattr(self, 'planned_batches_dict'):
            raise RuntimeError("Must call plan_all_batches first")
            
        return self.load_batches_from_dict(self.planned_batches_dict, self.graph_data)
    
    def _add_special_batches(self, planned_batches: Dict, agent_results: Dict):
        """Add integration test and documentation batches"""
        
        # Get subtree order from graph data  
        subtree_order = self.graph_data.get("data_flow_phase", {}).get("subtree_order", [])
        
        # Add integration test batch for each subtree
        for subtree in subtree_order:
            if subtree in planned_batches:
                # Check if the subtree has any actual batches (non-empty)
                has_batches = any(
                    len(batches) > 0 
                    for batches in planned_batches.get(subtree, {}).values()
                )
                
                if not has_batches:
                    self.logger.info(f"Skipping integration test for subtree {subtree} - no implementation batches")
                    continue
                
                # Get subtree path information from RPG
                subtree_path = self._get_subtree_path_from_rpg(subtree)
                
                integration_test_batch = TaskBatch(
                task=(
                    f"Write comprehensive integration tests for the {subtree} module "
                    f"{f'located in {subtree_path}' if subtree_path else ''}. "
                    f"Test the interactions between all components in this module, "
                    f"verify data flow, error handling, and edge cases, and ensure all public APIs "
                    f"work correctly together. "
                    f"Focus on testing the integration points between different files in this module. "
                    f"In addition to the integration tests, create small, focused usage examples for this module "
                    f"(e.g., example scripts or functions) that demonstrate typical end-to-end usage of its main APIs. "
                    f"Create appropriate test files and example files in the module directory or the test/example "
                    f"directory as needed, following the existing project conventions."
                ),
                file_path="<INTEGRATION_TEST>",  # Special marker - let agent decide placement
                units_key=[f"{subtree}_integration_tests"],
                unit_to_code={f"{subtree}_integration_tests": f"# Integration tests for {subtree} module"},
                unit_to_features={f"{subtree}_integration_tests": [f"{subtree} integration testing"]},
                priority=1000,  # Lower priority (higher number) - run after regular implementation
                subtree=subtree,
                task_type="integration_test",
            )
                
                # Add integration test to the subtree
                integration_file_path = f"<INTEGRATION_TEST>_{subtree}"
                planned_batches[subtree][integration_file_path] = [integration_test_batch]
                agent_results[subtree][integration_file_path] = {"success": True, "type": "integration_test"}
                
                self.logger.info(f"Added integration test batch for subtree: {subtree} (path: {subtree_path})")
        
        # Create a special subtree for final tasks
        final_subtree = "FINAL_TASKS"
        planned_batches[final_subtree] = {}
        agent_results[final_subtree] = {}
        
        # Add final comprehensive test and documentation batches
        comprehensive_test_batch = TaskBatch(
            task=(
                "Design and implement comprehensive end-to-end tests for the entire repository. "
                "Cover system tests, integration tests across all major modules, performance scenarios, "
                "and critical edge cases. "
                "Verify the complete workflow from input to output and validate interactions between modules "
                "so the system works reliably as a whole. "
                "In addition, create clear usage examples (e.g., example scripts or notebooks) that demonstrate "
                "typical end-to-end workflows, and add or update an executable main entry point (e.g., main.py) "
                "that can run a representative full pipeline using the implemented tests and examples. "
                "Place the new test files, examples, and main entry point in appropriate locations in the project structure."
            ),
            file_path="<COMPREHENSIVE_TEST>",  # Special marker - let agent decide placement
            units_key=["comprehensive_tests"],
            unit_to_code={"comprehensive_tests": "# Comprehensive end-to-end tests"},
            unit_to_features={"comprehensive_tests": ["comprehensive system testing"]},
            priority=2000,  # Lowest priority - run last
            subtree=final_subtree,
            task_type="final_test_docs",
        )
        # Add final batches
        planned_batches[final_subtree]["<COMPREHENSIVE_TEST>"] = [comprehensive_test_batch]
        self.logger.info("Added final comprehensive test and documentation batches")
    
    def _get_subtree_path_from_rpg(self, subtree_name: str) -> Optional[str]:
        """Get the directory path for a subtree from RPG metadata"""
        try:
            # Look for nodes with matching names at level 1 (direct children of repo)
            for node in self.repo_rpg.nodes.values():
                if (node.name == subtree_name and 
                    node.level == 1):
                    
                    if node.meta and node.meta.path and node.meta.path != ".":
                        final_path = node.meta.path
                        final_path = final_path if isinstance(final_path, str) else ", ".join(final_path)
                        return final_path
                    break
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Failed to get subtree path from RPG for {subtree_name}: {e}")
            return None
    
    def get_next_batch(self, processed_files: set = None) -> Optional[TaskBatch]:
        """Get next unprocessed batch"""
        processed_files = processed_files or set()
        
        for batch in self.get_batches_as_list():
            if batch.file_path not in processed_files:
                return batch
        
        return None
    
    def get_batch_count(self) -> Dict[str, int]:
        """Get statistics about planned batches"""
        if not hasattr(self, 'planned_batches_dict'):
            return {"total_batches": 0, "total_files": 0, "total_subtrees": 0}
            
        total_batches = 0
        total_files = 0
        total_subtrees = len(self.planned_batches_dict)
        
        for subtree, files_dict in self.planned_batches_dict.items():
            total_files += len(files_dict)
            for file_batches in files_dict.values():
                total_batches += len(file_batches)
                
        return {
            "total_batches": total_batches,
            "total_files": total_files, 
            "total_subtrees": total_subtrees
        }
    