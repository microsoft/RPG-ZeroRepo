"""RPG (Repository Program Graph) — core graph model package.

Provides the data structures, builders, serializers, and operation
interfaces for the RPG graph.

Note: ``DependencyGraph`` is intentionally NOT exported at this top
level (it pulls in ``networkx`` and ``common.utils``).  When needed,
import it explicitly via ``from rpg.dep_graph import DependencyGraph``.
"""

from .models import (
    RPG, Node, Edge,
    NodeMetaData, NodeType, EdgeType,
    strip_uuid8, uuid8,
    infer_type_name_from_path,
    MAX_LEVEL, MAX_FEATURE_LEVEL,
)
from .builder import (
    create_initial_rpg,
    load_refactor_feature_data,
    get_rpg_statistics,
)
from .code_unit import (
    CodeUnit, ParsedFile,
    ParsedWorkspace, ParsedModule,
    CodeSnippetBuilder, merge_codeunits,
    class_ast_to_header_str, compare_code_units,
)
from .path_format import (
    file_node_path,
    function_node_path,
    class_node_path,
    method_node_path,
    parse_node_path,
    to_dep_graph_id,
    from_dep_graph_id,
    desc_key_function,
    desc_key_class,
    desc_key_method,
)
# DependencyGraph is intentionally NOT eagerly imported here — it depends
# on networkx + common.utils.  Use: from rpg.dep_graph import DependencyGraph

__all__ = [
    "RPG", "Node", "Edge",
    "NodeMetaData", "NodeType", "EdgeType",
    "strip_uuid8", "uuid8", "infer_type_name_from_path",
    "MAX_LEVEL", "MAX_FEATURE_LEVEL",
    "create_initial_rpg", "load_refactor_feature_data", "get_rpg_statistics",
    "CodeUnit", "ParsedFile", "ParsedWorkspace", "ParsedModule",
    "CodeSnippetBuilder", "merge_codeunits",
    "class_ast_to_header_str", "compare_code_units",
    "file_node_path", "function_node_path", "class_node_path",
    "method_node_path", "parse_node_path",
    "to_dep_graph_id", "from_dep_graph_id",
    "desc_key_function", "desc_key_class", "desc_key_method",
]
