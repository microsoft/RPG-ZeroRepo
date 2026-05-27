#!/usr/bin/env python3
"""Repository Entity and Dependency Searchers for RPG Agent.

Provides:
- ``RepoEntitySearcher`` — Looks up node data and feature paths via RPG + dep_graph.
- ``RepoDependencySearcher`` — Traverses dependency edges in the dep_graph.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/env/searcher.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from common.utils import is_test_file, wrap_code_snippet
from rpg import EdgeType, Node, NodeType, RPG

logger = logging.getLogger(__name__)


# ============================================================================
# RepoEntitySearcher
# ============================================================================

class RepoEntitySearcher:
    """Search entities in repository using RPG and its dependency graph.

    The RPG now contains:
    - ``dep_graph``: DependencyGraph with the underlying networkx graph (``dep_graph.G``)
    - ``_dep_to_rpg_map``: Mapping from dep graph nodes to RPG nodes

    Source: RPG-ZeroRepo env/searcher.py ``RepoEntitySearcher``
    """

    def __init__(self, rpg: RPG):
        """Initialize searcher with an RPG instance.

        Args:
            rpg: RPG instance containing ``dep_graph`` and ``_dep_to_rpg_map``.
        """
        self.rpg = rpg
        # Get dep graph from RPG
        self.G = rpg.dep_graph.G if rpg.dep_graph else None
        # Get dep2rpg mapping from RPG
        self.dep2rpg: Dict[str, List[str]] = rpg._dep_to_rpg_map or {}

        self._global_name_dict: Optional[Dict[str, List[str]]] = None
        self._global_name_dict_lowercase: Optional[Dict[str, List[str]]] = None
        self._etypes_dict = {etype: i for i, etype in enumerate(EdgeType)}

    @classmethod
    def from_components(
        cls,
        dep_graph,
        rpg: RPG,
        dep2rpg: Dict[str, List[str]],
    ) -> "RepoEntitySearcher":
        """Create searcher from individual components (backward-compat).

        Args:
            dep_graph: NetworkX graph of dependencies.
            rpg: RPG instance.
            dep2rpg: Mapping from dep graph nodes to RPG nodes.

        Returns:
            RepoEntitySearcher instance.
        """
        from rpg.dep_graph import DependencyGraph

        if rpg.dep_graph is None:
            rpg.dep_graph = DependencyGraph.__new__(DependencyGraph)
            rpg.dep_graph.G = dep_graph
            rpg.dep_graph.repo_dir = ""
        if not rpg._dep_to_rpg_map:
            rpg._dep_to_rpg_map = dep2rpg

        return cls(rpg)

    # --- Lazy computed name dicts ----------------------------------------

    @property
    def global_name_dict(self) -> Dict[str, List[str]]:
        """Case-sensitive name -> [nid, ...] mapping."""
        if self._global_name_dict is None:
            d: Dict[str, List[str]] = defaultdict(list)
            if self.G is not None:
                for nid in self.G.nodes():
                    if nid.endswith(".py"):
                        fname = nid.split("/")[-1]
                        d[fname].append(nid)
                        name = nid[: -len(".py")].split("/")[-1]
                        d[name].append(nid)
                    elif ":" in nid:
                        name = nid.split(":")[-1].split(".")[-1]
                        d[name].append(nid)
            self._global_name_dict = d
        return self._global_name_dict

    @property
    def global_name_dict_lowercase(self) -> Dict[str, List[str]]:
        """Case-insensitive (lowered) name -> [nid, ...] mapping."""
        if self._global_name_dict_lowercase is None:
            d: Dict[str, List[str]] = defaultdict(list)
            if self.G is not None:
                for nid in self.G.nodes():
                    if nid.endswith(".py"):
                        fname = nid.split("/")[-1].lower()
                        d[fname].append(nid)
                        name = nid[: -len(".py")].split("/")[-1].lower()
                        d[name].append(nid)
                    elif ":" in nid:
                        name = nid.split(":")[-1].split(".")[-1].lower()
                        d[name].append(nid)
            self._global_name_dict_lowercase = d
        return self._global_name_dict_lowercase

    # --- Node queries ----------------------------------------------------

    def has_node(self, nid: str, include_test: bool = False) -> bool:
        """Check whether *nid* exists in the dependency graph."""
        if self.G is None:
            return False
        if not include_test and is_test_file(nid):
            return False
        return nid in self.G

    def get_feature_paths_for_node(self, nid: str) -> List[str]:
        """Return RPG feature paths mapped to *nid*."""
        rpg_node_ids: List[str] = self.dep2rpg.get(nid, [])
        feature_paths: List[str] = []

        for rpg_node_id in rpg_node_ids:
            rpg_node: Optional[Node] = self.rpg.get_node_by_id(rpg_node_id)
            if not rpg_node:
                continue
            if rpg_node.level == 1:
                continue
            fp = rpg_node.feature_path()
            if fp:
                feature_paths.append(fp)
        return feature_paths

    def get_node_data(
        self,
        nids: List[str],
        return_code_content: bool = False,
        wrap_with_ln: bool = True,
    ) -> List[Dict]:
        """Return structured data dicts for the given *nids*.

        Each dict has keys: ``node_id``, ``type``, ``feature_paths``,
        and optionally ``start_line``, ``end_line``, ``code_content``.
        """
        if self.G is None:
            return []

        rtn: List[Dict] = []
        for nid in nids:
            if nid not in self.G.nodes:
                continue
            node_data = self.G.nodes[nid]
            feature_paths = self.get_feature_paths_for_node(nid)

            formatted: Dict = {
                "node_id": nid,
                "type": node_data["type"],
                "feature_paths": feature_paths,
            }

            code = node_data.get("code", "")
            if code:
                # start_line
                if "start_line" in node_data:
                    start_line = node_data["start_line"]
                elif formatted["type"] == NodeType.FILE:
                    start_line = 1
                else:
                    start_line = 1
                formatted["start_line"] = start_line

                # end_line
                if "end_line" in node_data:
                    end_line = node_data["end_line"]
                elif formatted["type"] == NodeType.FILE:
                    end_line = len(code.split("\n"))
                else:
                    end_line = 1
                formatted["end_line"] = end_line

                if return_code_content:
                    if wrap_with_ln:
                        formatted["code_content"] = wrap_code_snippet(
                            code, start_line, end_line,
                        )
                    else:
                        formatted["code_content"] = code

            rtn.append(formatted)
        return rtn

    def get_all_nodes_by_type(self, type: NodeType) -> List[Dict]:
        """Return formatted data dicts for all nodes of *type* (excluding tests)."""
        if self.G is None:
            return []

        nodes: List[Dict] = []
        for nid in self.G.nodes():
            if is_test_file(nid):
                continue
            if self.G.nodes[nid]["type"] != type:
                continue

            node_data = self.G.nodes[nid]
            if type == NodeType.FILE:
                formatted: Dict = {
                    "name": nid,
                    "type": node_data["type"],
                    "content": node_data.get("code", "").split("\n"),
                }
            elif type in (NodeType.METHOD, NodeType.FUNCTION):
                formatted = {
                    "name": nid.split(":")[-1],
                    "file": nid.split(":")[0],
                    "type": node_data["type"],
                    "content": node_data.get("code", "").split("\n"),
                    "start_line": node_data.get("start_line", 0),
                    "end_line": node_data.get("end_line", 0),
                }
            elif type == NodeType.CLASS:
                formatted = {
                    "name": nid.split(":")[-1],
                    "file": nid.split(":")[0],
                    "type": node_data["type"],
                    "content": node_data.get("code", "").split("\n"),
                    "start_line": node_data.get("start_line", 0),
                    "end_line": node_data.get("end_line", 0),
                    "methods": [],
                }
                # Resolve method children
                dp_searcher = RepoDependencySearcher(self.G)
                methods, _ = dp_searcher.get_neighbors(
                    nid, "forward",
                    ntype_filter=[NodeType.METHOD],
                    etype_filter=[EdgeType.CONTAINS],
                )
                for mid in methods:
                    mnode = self.G.nodes[mid]
                    formatted["methods"].append({
                        "name": mid.split(".")[-1],
                        "start_line": mnode.get("start_line", 0),
                        "end_line": mnode.get("end_line", 0),
                    })
            else:
                continue

            feature_paths = self.get_feature_paths_for_node(nid)
            formatted["feature_paths"] = feature_paths
            nodes.append(formatted)

        return nodes


# ============================================================================
# RepoDependencySearcher
# ============================================================================

class RepoDependencySearcher:
    """Traverse dependency edges in the dep_graph.

    Source: RPG-ZeroRepo env/searcher.py ``RepoDependencySearcher``
    """

    def __init__(self, graph):
        self.G = graph
        self._etypes_dict = {etype: i for i, etype in enumerate(EdgeType)}

    @classmethod
    def from_rpg(cls, rpg: RPG) -> "RepoDependencySearcher":
        """Create searcher from RPG instance."""
        if rpg.dep_graph is None:
            raise ValueError("RPG does not have a dependency graph")
        return cls(rpg.dep_graph.G)

    def subgraph(self, nids):
        """Return the subgraph induced by *nids*."""
        return self.G.subgraph(nids)

    def get_neighbors(
        self,
        nid: str,
        direction: str = "forward",
        ntype_filter=None,
        etype_filter=None,
        ignore_test_file: bool = True,
    ):
        """Return (nodes, edges) reachable from *nid* in *direction*.

        Args:
            nid: Starting node ID.
            direction: ``'forward'`` (successors) or ``'backward'`` (predecessors).
            ntype_filter: Optional list of NodeType values to include.
            etype_filter: Optional list of EdgeType values to include.
            ignore_test_file: Skip test-file nodes.

        Returns:
            Tuple of (node_ids, edge_tuples).
        """
        nodes, edges = [], []

        if direction == "forward":
            for sn in self.G.successors(nid):
                if ntype_filter and self.G.nodes[sn]["type"] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(sn):
                    continue
                for _key, edge_data in self.G.get_edge_data(nid, sn).items():
                    etype = edge_data["type"]
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append(
                        (nid, sn, self._etypes_dict.get(etype, 0), {"type": etype})
                    )
                    nodes.append(sn)

        elif direction == "backward":
            for pn in self.G.predecessors(nid):
                if ntype_filter and self.G.nodes[pn]["type"] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(pn):
                    continue
                for _key, edge_data in self.G.get_edge_data(pn, nid).items():
                    etype = edge_data["type"]
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append(
                        (pn, nid, self._etypes_dict.get(etype, 0), {"type": etype})
                    )
                    nodes.append(pn)

        return nodes, edges
