#!/usr/bin/env python3
"""Func Design Module.

This module provides the complete func_design workflow:
1. Data Flow Design - Design inter-component data flow (DAG)
2. Base Class Design - Design shared base classes and data structures
3. Interface Design - Design function/class interfaces for each file
"""

# Data Flow
from .data_flow_agent import (
    DataFlowAgent,
    LLMClient as DataFlowLLMClient,
    DataFlowEdge,
    DataFlowOutput,
    validate_data_flow,
    compute_topological_order,
)
from .data_flow_prompts import (
    DATA_FLOW_PROMPT,
    DATA_FLOW_REVIEW_PROMPT,
    format_functional_areas,
)

# Base Class
from .base_class_agent import (
    BaseClassAgent,
    LLMClient as BaseClassLLMClient,
    extract_declaration_names,
    validate_base_classes,
    DataStructureDefinition,
    extract_data_flow_types,
    validate_data_structures,
)
from .base_class_prompts import (
    BASE_CLASS_PROMPT,
    BASE_CLASS_REVIEW_PROMPT,
)

# Interface
from .interface_agent import (
    InterfaceAgent,
    SubtreeInterfaceAgent,
    InterfaceOrchestrator,
    DependencyCollector,
    GlobalInterfaceRegistry,
    cross_validate_imports_vs_calls,
    LLMClient as InterfaceLLMClient,
)
from .interface_prompts import (
    INTERFACE_PROMPT,
    PLAN_FILE_PROMPT,
    SUBTREE_INTERFACE_PROMPT,
    ORPHAN_REVIEW_PROMPT,
)

# Global Interface Review
from .interface_review import (
    InterfaceReviewer,
    check_call_graph_connectivity,
    check_feature_dependency_coverage,
    print_review_summary,
    prune_orphan_interfaces,
    review_orphan_units,
)

# Unified InterfacesStore
from .interfaces_store import (
    InterfacesStore,
    InterfaceUnit,
    InheritanceEdge,
    InvocationEdge,
    ReferenceEdge,
    PruneSummary,
    OrphanFeature,
    RPGUpdateSummary,
)

# Unified Entry Point
from .func_designer import FuncDesigner

__all__ = [
    # Data Flow
    "DataFlowAgent",
    "DataFlowLLMClient",
    "DataFlowEdge",
    "DataFlowOutput",
    "validate_data_flow",
    "compute_topological_order",
    "DATA_FLOW_PROMPT",
    "DATA_FLOW_REVIEW_PROMPT",
    "format_functional_areas",
    # Base Class
    "BaseClassAgent",
    "BaseClassLLMClient",
    "extract_declaration_names",
    "validate_base_classes",
    "DataStructureDefinition",
    "extract_data_flow_types",
    "validate_data_structures",
    "BASE_CLASS_PROMPT",
    "BASE_CLASS_REVIEW_PROMPT",
    # Interface
    "InterfaceAgent",
    "SubtreeInterfaceAgent",
    "InterfaceOrchestrator",
    "DependencyCollector",
    "GlobalInterfaceRegistry",
    "cross_validate_imports_vs_calls",
    "InterfaceLLMClient",
    "INTERFACE_PROMPT",
    "PLAN_FILE_PROMPT",
    "SUBTREE_INTERFACE_PROMPT",
    "ORPHAN_REVIEW_PROMPT",
    # Global Review
    "InterfaceReviewer",
    "check_call_graph_connectivity",
    "check_feature_dependency_coverage",
    "print_review_summary",
    "prune_orphan_interfaces",
    "review_orphan_units",
    # InterfacesStore
    "InterfacesStore",
    "InterfaceUnit",
    "InheritanceEdge",
    "InvocationEdge",
    "ReferenceEdge",
    "PruneSummary",
    "OrphanFeature",
    "RPGUpdateSummary",
    # Unified Entry Point
    "FuncDesigner",
]
