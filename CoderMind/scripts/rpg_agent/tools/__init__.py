"""RPG Agent tools — Tool wrappers for agent actions.

Each tool inherits from ``scripts.common.tools.Tool`` (M3 Tool ABC) and
bridges to the underlying ops functions (M9 Agent Ops).

Source: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/tools/
"""

from rpg_agent.tools.search_node import SearchNode
from rpg_agent.tools.search_code_snippets import SearchCodeSnippets
from rpg_agent.tools.search_code_by_features import SearchCodeByFeatures
from rpg_agent.tools.fetch_node import FetchNode
from rpg_agent.tools.explore_rpg import ExploreRPG
from rpg_agent.tools.terminate import Terminate

# Convenience list for registering all tools at once.
ALL_TOOLS = [
    SearchNode,
    SearchCodeSnippets,
    SearchCodeByFeatures,
    FetchNode,
    ExploreRPG,
    Terminate,
]

__all__ = [
    "SearchNode",
    "SearchCodeSnippets",
    "SearchCodeByFeatures",
    "FetchNode",
    "ExploreRPG",
    "Terminate",
    "ALL_TOOLS",
]
