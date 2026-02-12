# Refactored agents following tool-based pattern

from .base_class_agent import (
    BaseClassAgent,
    GenerateBaseClassesTool,
    BaseClassEnv,
    BaseClassReviewOutput
)

from .data_flow_agent import (
    DataFlowAgent,
    GenerateDataFlowTool,
    DataFlowEnv,
    DataFlowReviewOutput
)

from .interface_agent import (
    InterfaceAgent,
    DesignInterfacesTool,
    InterfaceEnv,
    InterfaceReviewOutput
)