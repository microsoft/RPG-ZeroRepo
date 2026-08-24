"""Coverage validation for base-class generation."""
from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from func_design.base_class_agent import (  # noqa: E402
    BaseClassAgent,
    BaseClassOutput,
    DataStructureDefinition,
)


class PartialCoverageLLM:
    def __init__(self) -> None:
        self.calls = 0

    def call_structured(self, **kwargs):
        self.calls += 1
        result = BaseClassOutput(
            data_structures=[
                DataStructureDefinition(
                    code="class TypeA:\n    pass\n",
                    subtree="Area",
                    data_flow_types=["TypeA"],
                )
            ]
        )
        return None, result, None


def test_uncovered_data_flow_types_exhaust_retries() -> None:
    llm = PartialCoverageLLM()
    agent = BaseClassAgent(
        llm_client=llm,
        max_iterations=2,
        target_language="python",
    )

    result = agent.design_base_classes(
        repo_name="fixture",
        repo_info="fixture",
        data_flow=[
            {"data_type": "TypeA"},
            {"data_type": "TypeB"},
        ],
        skeleton_tree="fixture",
        functional_areas=["Area"],
        functional_areas_overview="Area",
        project_background="",
    )

    assert llm.calls == 2
    assert result["success"] is False
    assert "TypeB" in result["error"]
