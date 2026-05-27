"""Agent operations — search, fetch, and explore RPG nodes.

Public API:
    search_code_snippets    — search by entity name, code path, line numbers
    search_features_by_keywords — search by functional-feature keywords
    fetch_node              — retrieve code by entity ID or feature path
    explore_tree_structure  — traverse the RPG graph as a tree / JSON
"""

from rpg_agent.ops.search_by_meta import search_code_snippets
from rpg_agent.ops.search_by_feature import search_features_by_keywords
from rpg_agent.ops.fetch import fetch_node
from rpg_agent.ops.explore import explore_tree_structure

__all__ = [
    "search_code_snippets",
    "search_features_by_keywords",
    "fetch_node",
    "explore_tree_structure",
]
