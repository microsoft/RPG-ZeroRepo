"""Graph query functions for RPG and dep_graph.

Provides search, explore, detail, and tree browsing over the RPG feature
graph and AST-based dependency graph.  These are pure Python functions
usable both as a Python API (by codegen pipeline) and as MCP tool backends.

Usage (Python API)::

    from rpg.graph_query import GraphQueryEngine

    engine = GraphQueryEngine.from_rpg_file("rpg.json")
    results = engine.search("pagination")
    detail = engine.get_node_detail("utils/pagination.py:paginate")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.rpg_io import safe_load_rpg

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

# Maximum edges returned by explore before truncation
_MAX_EXPLORE_EDGES = 20


class GraphQueryEngine:
    """Stateful query engine over RPG + dep_graph data.

    Loads and indexes the graph data once, then provides fast queries.
    Thread-safe for read-only operations.
    """

    def __init__(self, rpg_data: dict, dep_graph_data: dict):
        self._rpg = rpg_data

        # Path prefix for normalising RPG meta.path.  In the unified
        # workspace==repo layout this is always ``""``; only legacy data
        # produced under the old ``<workspace>/repo`` layout has a
        # non-empty code_dir that needs stripping.
        code_dir = rpg_data.get("_dep_graph_code_dir", "")
        self._code_dir_prefix = code_dir + "/" if code_dir else ""

        # Indexes built on init
        self._dep_nodes: Dict[str, dict] = dep_graph_data.get("nodes", {})
        self._dep_edges: List[dict] = dep_graph_data.get("edges", [])
        self._dep_to_rpg: Dict[str, list] = rpg_data.get("_dep_to_rpg_map", {})
        self._feature_to_dep: Dict[str, list] = rpg_data.get("_feature_to_dep_map", {})

        # Adjacency lists for fast explore (P4)
        self._outgoing: Dict[str, List[dict]] = defaultdict(list)
        self._incoming: Dict[str, List[dict]] = defaultdict(list)
        for edge in self._dep_edges:
            self._outgoing[edge["src"]].append(edge)
            self._incoming[edge["dst"]].append(edge)

        # RPG tree index: id -> node dict (flattened) + parent map (P3)
        self._rpg_nodes: Dict[str, dict] = {}
        self._rpg_children: Dict[str, List[str]] = defaultdict(list)
        self._parent_map: Dict[str, Optional[str]] = {}
        root = rpg_data.get("root")
        if root:
            self._index_rpg_tree(root, parent_id=None)

        # Prebuilt lists for rapidfuzz fuzzy matching
        self._dep_node_ids: List[str] = list(self._dep_nodes.keys())
        self._dep_node_names: List[str] = [
            self._dep_nodes[nid].get("name", "") for nid in self._dep_node_ids
        ]
        self._rpg_node_ids: List[str] = list(self._rpg_nodes.keys())
        self._rpg_node_names: List[str] = [
            self._rpg_nodes[nid].get("name", "") for nid in self._rpg_node_ids
        ]

    def _index_rpg_tree(self, node: dict, parent_id: Optional[str]) -> None:
        """Recursively index RPG tree nodes."""
        nid = node.get("id", "")
        self._rpg_nodes[nid] = node
        self._parent_map[nid] = parent_id
        if parent_id:
            self._rpg_children[parent_id].append(nid)
        for child in node.get("children", []):
            self._index_rpg_tree(child, parent_id=nid)

    @classmethod
    def from_rpg_file(cls, rpg_path: str) -> "GraphQueryEngine":
        """Load from a single rpg.json file.

        Handles both embedded dep_graph and external dep_graph_file reference.

        Uses :func:`common.rpg_io.safe_load_rpg` so a corrupted ``rpg.json``
        (e.g. an encoder that was killed mid-write) is silently recovered
        from the inner-git snapshot history rather than blocking every
        downstream read with ``JSONDecodeError``.
        """
        rpg_dir = Path(rpg_path).resolve().parent
        rpg_data = safe_load_rpg(rpg_path)

        # Try embedded dep_graph first, then external file
        dep_graph_data = rpg_data.get("dep_graph", {})
        if not dep_graph_data.get("nodes"):
            dep_graph_file = rpg_data.get("dep_graph_file", "")
            if dep_graph_file:
                dep_path = rpg_dir / dep_graph_file
                if dep_path.is_file():
                    dep_graph_data = safe_load_rpg(dep_path)
                    logger.info("Loaded dep_graph from %s", dep_path)
                else:
                    logger.warning("dep_graph_file not found: %s", dep_path)

        return cls(rpg_data, dep_graph_data)

    @classmethod
    def from_files(cls, rpg_path: str, dep_graph_path: str = "") -> "GraphQueryEngine":
        """Load from JSON files. If dep_graph_path is empty, uses embedded dep_graph."""
        rpg_data = safe_load_rpg(rpg_path)
        if dep_graph_path:
            dep_graph_data = safe_load_rpg(dep_graph_path)
        else:
            dep_graph_data = rpg_data.get("dep_graph", {})
        return cls(rpg_data, dep_graph_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_path(self, meta_path: str) -> str:
        """Strip redundant code_dir prefix from RPG meta.path.

        For legacy data where ``_dep_graph_code_dir`` was ``"repo"``,
        this converts e.g. ``repo/routes/auth.py`` to ``routes/auth.py``
        so it aligns with dep_graph node IDs.  In the unified
        workspace==repo layout the prefix is empty and this is a no-op.
        """
        if self._code_dir_prefix and meta_path.startswith(self._code_dir_prefix):
            return meta_path[len(self._code_dir_prefix):]
        return meta_path

    def _get_feature_path(self, node_id: str) -> str:
        """Build ancestor chain for an RPG node (excluding repo root)."""
        parts: list[str] = []
        current: Optional[str] = node_id
        while current:
            node = self._rpg_nodes.get(current, {})
            parts.append(node.get("name", current))
            current = self._parent_map.get(current)
        # Reverse and drop the repo root
        if len(parts) > 1:
            return " > ".join(reversed(parts[:-1]))
        return parts[0] if parts else ""

    def _node_summary(self, node_id: str) -> Dict[str, Any]:
        """Compact summary of a dep_graph node."""
        attrs = self._dep_nodes.get(node_id, {})
        summary: Dict[str, Any] = {
            "type": attrs.get("type"),
            "name": attrs.get("name"),
        }
        module = attrs.get("module")
        if module:
            summary["module"] = module
        sig = attrs.get("signature")
        if sig:
            summary["signature"] = sig
        rpg = attrs.get("rpg_nodes", [])
        if rpg:
            summary["rpg_nodes"] = rpg
        return summary

    # ------------------------------------------------------------------
    # Tool 1: search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        scope: str = "all",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search nodes by keyword in feature graph and/or dep_graph.

        Args:
            query: Search keyword (name, path fragment, or keyword).
            scope: ``"feature"`` (RPG tree), ``"code"`` (dep_graph), or ``"all"``.
            top_k: Max results to return.

        Returns:
            List of result dicts with id, name, type, path, score, source.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        if scope not in ("code", "feature", "all"):
            return [{"error": f"Invalid scope: {scope!r}. Expected 'code', 'feature', or 'all'."}]

        results: List[Dict[str, Any]] = []

        if scope in ("code", "all"):
            results.extend(self._search_dep_graph(query_lower, top_k))

        if scope in ("feature", "all"):
            results.extend(self._search_rpg_tree(query_lower, top_k))

        # Deduplicate by id, keep highest score
        seen: Dict[str, Dict] = {}
        for r in results:
            rid = r["id"]
            if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                seen[rid] = r
        results = sorted(seen.values(), key=lambda x: -x.get("score", 0))

        return results[:top_k]

    def _search_dep_graph(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search dep_graph nodes by ID, name, path substring, then fuzzy match."""
        results = []
        seen_ids: set[str] = set()

        for nid, attrs in self._dep_nodes.items():
            ntype = attrs.get("type", "")
            if ntype in ("directory",):
                continue

            name = attrs.get("name", "")
            score = 0

            if query == nid.lower():
                score = 100
            elif query == name.lower():
                score = 95
            elif query in name.lower():
                score = 80
            elif query in nid.lower():
                score = 70
            elif query in (attrs.get("signature") or "").lower():
                score = 60

            if score > 0:
                file_path = nid.split(":")[0] if ":" in nid else nid
                results.append({
                    "id": nid,
                    "name": name,
                    "type": ntype,
                    "file": file_path,
                    "score": score,
                    "source": "dep_graph",
                    "signature": attrs.get("signature"),
                })
                seen_ids.add(nid)

        # Fuzzy matching fallback when substring yields few results
        if _HAS_RAPIDFUZZ and len(results) < top_k:
            needed = top_k - len(results)
            fuzzy_hits = self._fuzzy_match_dep(query, top_k=needed * 2, score_cutoff=55)
            for nid, fscore in fuzzy_hits:
                if nid in seen_ids:
                    continue
                attrs = self._dep_nodes[nid]
                ntype = attrs.get("type", "")
                if ntype in ("directory",):
                    continue
                file_path = nid.split(":")[0] if ":" in nid else nid
                results.append({
                    "id": nid,
                    "name": attrs.get("name", ""),
                    "type": ntype,
                    "file": file_path,
                    "score": fscore,
                    "source": "dep_graph",
                    "signature": attrs.get("signature"),
                    "match": "fuzzy",
                })
                seen_ids.add(nid)

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def _search_rpg_tree(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search RPG feature tree nodes by name, meta.path, then fuzzy match."""
        results = []
        seen_ids: set[str] = set()

        for nid, node in self._rpg_nodes.items():
            name = node.get("name", "")
            meta = node.get("meta") or {}
            meta_path = meta.get("path") or ""
            node_type = node.get("node_type", "")
            score = 0

            description = (meta.get("description") or "").lower()

            if query == name.lower():
                score = 95
            elif query in name.lower():
                score = 75
            else:
                file_part = meta_path.split(":")[0] if ":" in meta_path else meta_path
                if query in file_part.lower():
                    score = 60
                elif query in nid.lower():
                    score = 55
                elif description and query in description:
                    score = 50

            if score > 0:
                results.append({
                    "id": nid,
                    "name": name,
                    "type": node_type,
                    "path": self._normalize_path(meta_path),
                    "score": score,
                    "source": "rpg_tree",
                    "level": node.get("level"),
                })
                seen_ids.add(nid)

        # Fuzzy matching fallback
        if _HAS_RAPIDFUZZ and len(results) < top_k:
            needed = top_k - len(results)
            fuzzy_hits = self._fuzzy_match_rpg(query, top_k=needed * 2, score_cutoff=55)
            for nid, fscore in fuzzy_hits:
                if nid in seen_ids:
                    continue
                node = self._rpg_nodes[nid]
                meta = node.get("meta") or {}
                meta_path = meta.get("path") or ""
                results.append({
                    "id": nid,
                    "name": node.get("name", ""),
                    "type": node.get("node_type", ""),
                    "path": self._normalize_path(meta_path),
                    "score": fscore,
                    "source": "rpg_tree",
                    "level": node.get("level"),
                    "match": "fuzzy",
                })
                seen_ids.add(nid)

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    # ------------------------------------------------------------------
    # Tool 2: explore
    # ------------------------------------------------------------------

    def explore(
        self,
        node_id: str,
        direction: str = "both",
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Explore dep_graph from a starting node.

        Args:
            node_id: Starting dep_graph node ID.
            direction: ``"downstream"`` (callees), ``"upstream"`` (callers),
                or ``"both"``.
            depth: Max traversal depth.
            edge_types: Filter edges (e.g. ``["invokes", "inherits"]``).
                Default: all non-contains edges.

        Returns:
            Dict with start node, connected nodes, and edges.
        """
        if node_id not in self._dep_nodes:
            suggestions = self._suggest_dep_nodes(node_id)
            return {
                "error": f"Node not found: {node_id}",
                "suggestions": suggestions,
            }

        if direction not in ("downstream", "upstream", "both"):
            return {
                "error": f"Invalid direction: {direction!r}. Expected 'downstream', 'upstream', or 'both'.",
            }

        allowed_types = set(edge_types) if edge_types else {"invokes", "inherits", "imports"}
        visited = {node_id}
        result_nodes = {node_id: self._node_summary(node_id)}
        result_edges: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()  # (src, dst, type) dedup
        truncated = False
        frontier = [node_id]

        for hop in range(depth):
            next_frontier: list[str] = []
            for current in frontier:
                edges_to_check: list[dict] = []
                if direction in ("downstream", "both"):
                    edges_to_check.extend(self._outgoing.get(current, []))
                if direction in ("upstream", "both"):
                    edges_to_check.extend(self._incoming.get(current, []))

                for edge in edges_to_check:
                    etype = (edge.get("attrs") or {}).get("type", "")
                    if etype not in allowed_types:
                        continue

                    # Determine neighbor and direction (B1: use elif)
                    neighbor = None
                    edge_dir = None
                    if edge["src"] == current and direction in ("downstream", "both"):
                        neighbor = edge["dst"]
                        edge_dir = "downstream"
                    elif edge["dst"] == current and direction in ("upstream", "both"):
                        neighbor = edge["src"]
                        edge_dir = "upstream"

                    if not neighbor:
                        continue

                    # Always record edges for complete subgraph (B2), deduplicate
                    edge_key = (edge["src"], edge["dst"], etype)
                    if edge_key not in seen_edges and (
                        neighbor in result_nodes or neighbor not in visited
                    ):
                        seen_edges.add(edge_key)
                        if len(result_edges) < _MAX_EXPLORE_EDGES:
                            result_edges.append({
                                "src": edge["src"],
                                "dst": edge["dst"],
                                "type": etype,
                                "direction": edge_dir,
                                "hop": hop + 1,
                            })
                        else:
                            truncated = True

                    # Discover new nodes
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        result_nodes[neighbor] = self._node_summary(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        result: Dict[str, Any] = {
            "start": node_id,
            "direction": direction,
            "depth": depth,
            "total_nodes": len(result_nodes),
            "total_edges": len(result_edges),
            "nodes": result_nodes,
            "edges": result_edges,
        }
        if truncated:
            result["truncated"] = True
        return result

    # ------------------------------------------------------------------
    # Tool 3: get_node_detail
    # ------------------------------------------------------------------

    def get_node_detail(
        self,
        node_id: str,
        include_code: bool = False,
    ) -> Dict[str, Any]:
        """Get detailed information about a dep_graph or RPG node.

        Args:
            node_id: dep_graph node ID or RPG node ID.
            include_code: Include source code (reads from disk).

        Returns:
            Dict with all node attributes.
        """
        # Try dep_graph first
        if node_id in self._dep_nodes:
            attrs = dict(self._dep_nodes[node_id])
            result: Dict[str, Any] = {
                "id": node_id,
                "type": attrs.get("type"),
                "name": attrs.get("name"),
                "module": attrs.get("module"),
                "signature": attrs.get("signature"),
                "start_line": attrs.get("start_line"),
                "end_line": attrs.get("end_line"),
                "imports_from": attrs.get("imports_from"),
                "calls": attrs.get("calls"),
                "called_by": attrs.get("called_by"),
                "inherits": attrs.get("inherits"),
                "inherited_by": attrs.get("inherited_by"),
                "rpg_nodes": attrs.get("rpg_nodes", []),
                "source": "dep_graph",
            }
            # Strip None values
            result = {k: v for k, v in result.items() if v is not None}

            # For file nodes, aggregate all child features (Q1/P10)
            if attrs.get("type") == "file":
                prefix = node_id + ":"
                all_features_set: set[str] = set()
                for nid2, attrs2 in self._dep_nodes.items():
                    if nid2.startswith(prefix):
                        all_features_set.update(attrs2.get("rpg_nodes", []))
                if all_features_set:
                    result["all_features"] = sorted(all_features_set)

            if include_code:
                code_path = attrs.get("code_path", "")
                start = attrs.get("start_line")
                end = attrs.get("end_line")
                if code_path and start and end:
                    try:
                        with open(code_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        result["code"] = "".join(lines[start - 1:end])
                    except (OSError, IndexError):
                        result["code"] = None

            return result

        # Try RPG node
        if node_id in self._rpg_nodes:
            node = self._rpg_nodes[node_id]
            meta = node.get("meta") or {}
            meta_path = meta.get("path") or ""
            children = node.get("children", [])

            result = {
                "id": node_id,
                "name": node.get("name"),
                "node_type": node.get("node_type"),
                "level": node.get("level"),
                "meta_type_name": meta.get("type_name"),
                "code_path": self._normalize_path(meta_path) if meta_path else None,
                "feature_path": self._get_feature_path(node_id),
                "dep_node_ids": self._feature_to_dep.get(node_id, []),
                "children_count": len(children),
                "source": "rpg_tree",
            }
            if children:
                result["children_names"] = [c.get("name") for c in children[:20]]
            # Strip None values
            return {k: v for k, v in result.items() if v is not None}

        # Not found — suggest
        suggestions = self._suggest_dep_nodes(node_id) + self._suggest_rpg_nodes(node_id)
        return {
            "error": f"Node not found: {node_id}",
            "suggestions": suggestions[:5],
        }

    # ------------------------------------------------------------------
    # Fuzzy matching helpers (rapidfuzz)
    # ------------------------------------------------------------------

    def _fuzzy_match_dep(
        self, query: str, top_k: int = 10, score_cutoff: float = 55,
    ) -> List[Tuple[str, float]]:
        """Fuzzy match query against dep_graph node names and IDs.

        Uses rapidfuzz token_set_ratio + WRatio, returns (node_id, score) pairs.
        """
        if not _HAS_RAPIDFUZZ or not self._dep_node_ids:
            return []

        best: Dict[str, float] = {}

        # Match against names (token_set_ratio)
        for name, score, idx in rf_process.extract(
            query, self._dep_node_names, scorer=fuzz.token_set_ratio,
            limit=top_k, score_cutoff=score_cutoff,
        ):
            nid = self._dep_node_ids[idx]
            if nid not in best or score > best[nid]:
                best[nid] = score

        # Match against names (WRatio, often better for partial matches)
        for name, score, idx in rf_process.extract(
            query, self._dep_node_names, scorer=fuzz.WRatio,
            limit=top_k, score_cutoff=score_cutoff,
        ):
            nid = self._dep_node_ids[idx]
            if nid not in best or score > best[nid]:
                best[nid] = score

        # Match against node IDs (path-level fuzzy)
        for nid_str, score, idx in rf_process.extract(
            query, self._dep_node_ids, scorer=fuzz.ratio,
            limit=top_k, score_cutoff=score_cutoff,
        ):
            nid = self._dep_node_ids[idx]
            if nid not in best or score > best[nid]:
                best[nid] = score

        # Normalize scores to 0-55 range (below exact/substring tier)
        results = []
        for nid, raw_score in best.items():
            normalized = round(raw_score * 0.55, 1)  # max 55, below substring=60
            results.append((nid, normalized))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _fuzzy_match_rpg(
        self, query: str, top_k: int = 10, score_cutoff: float = 55,
    ) -> List[Tuple[str, float]]:
        """Fuzzy match query against RPG tree node names."""
        if not _HAS_RAPIDFUZZ or not self._rpg_node_ids:
            return []

        best: Dict[str, float] = {}

        # Match against RPG node names
        for name, score, idx in rf_process.extract(
            query, self._rpg_node_names, scorer=fuzz.token_set_ratio,
            limit=top_k, score_cutoff=score_cutoff,
        ):
            nid = self._rpg_node_ids[idx]
            if nid not in best or score > best[nid]:
                best[nid] = score

        for name, score, idx in rf_process.extract(
            query, self._rpg_node_names, scorer=fuzz.WRatio,
            limit=top_k, score_cutoff=score_cutoff,
        ):
            nid = self._rpg_node_ids[idx]
            if nid not in best or score > best[nid]:
                best[nid] = score

        results = []
        for nid, raw_score in best.items():
            normalized = round(raw_score * 0.55, 1)
            results.append((nid, normalized))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _suggest_dep_nodes(self, query: str, limit: int = 5) -> List[str]:
        """Suggest dep_graph node IDs for a failed lookup."""
        query_lower = query.lower()
        # Substring match on IDs first
        subs = [nid for nid in self._dep_nodes if query_lower in nid.lower()]
        if len(subs) >= limit:
            return subs[:limit]

        if _HAS_RAPIDFUZZ:
            seen = set(subs)
            # Fuzzy match on IDs (path-level)
            for nid_str, score, idx in rf_process.extract(
                query, self._dep_node_ids, scorer=fuzz.ratio,
                limit=limit, score_cutoff=40,
            ):
                nid = self._dep_node_ids[idx]
                if nid not in seen:
                    subs.append(nid)
                    seen.add(nid)
            # Fuzzy match on names (e.g. "user_model" -> node named "User")
            for name, score, idx in rf_process.extract(
                query, self._dep_node_names, scorer=fuzz.WRatio,
                limit=limit, score_cutoff=50,
            ):
                nid = self._dep_node_ids[idx]
                if nid not in seen:
                    subs.append(nid)
                    seen.add(nid)
        return subs[:limit]

    def _suggest_rpg_nodes(self, query: str, limit: int = 5) -> List[str]:
        """Suggest RPG node IDs for a failed lookup."""
        query_lower = query.lower()
        subs = [nid for nid in self._rpg_nodes if query_lower in nid.lower()]
        if len(subs) >= limit:
            return subs[:limit]

        if _HAS_RAPIDFUZZ:
            seen = set(subs)
            # Fuzzy match on IDs
            for nid_str, score, idx in rf_process.extract(
                query, self._rpg_node_ids, scorer=fuzz.ratio,
                limit=limit, score_cutoff=40,
            ):
                nid = self._rpg_node_ids[idx]
                if nid not in seen:
                    subs.append(nid)
                    seen.add(nid)
            # Fuzzy match on names
            for name, score, idx in rf_process.extract(
                query, self._rpg_node_names, scorer=fuzz.WRatio,
                limit=limit, score_cutoff=50,
            ):
                nid = self._rpg_node_ids[idx]
                if nid not in seen:
                    subs.append(nid)
                    seen.add(nid)
        return subs[:limit]

    # ------------------------------------------------------------------
    # Tool 4: list_tree
    # ------------------------------------------------------------------

    def list_tree(
        self,
        root_id: Optional[str] = None,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """List RPG feature tree structure.

        Args:
            root_id: Start node ID (default: repo root).
            max_depth: How many levels to show.

        Returns:
            Dict with tree structure.
        """
        if root_id:
            if root_id in self._rpg_nodes:
                root = self._rpg_nodes[root_id]
            else:
                suggestions = self._suggest_rpg_nodes(root_id)
                return {
                    "error": f"Node not found: {root_id}",
                    "suggestions": suggestions,
                }
        elif self._rpg.get("root"):
            root = self._rpg["root"]
        else:
            return {"error": "No RPG tree loaded"}

        subtree_count = [0]

        def build(node: dict, current_depth: int) -> dict:
            subtree_count[0] += 1
            entry: Dict[str, Any] = {
                "id": node.get("id"),
                "name": node.get("name"),
                "node_type": node.get("node_type"),
            }
            meta = node.get("meta") or {}
            if meta.get("path"):
                entry["path"] = self._normalize_path(meta["path"])
            if meta.get("type_name"):
                entry["type_name"] = meta["type_name"]

            children = node.get("children", [])
            if children and current_depth < max_depth:
                entry["children"] = [
                    build(child, current_depth + 1) for child in children
                ]
            elif children:
                entry["children_count"] = len(children)

            return entry

        tree = build(root, 0)
        if root_id:
            tree["subtree_nodes"] = subtree_count[0]
        tree["total_nodes"] = len(self._rpg_nodes)
        return tree
