#!/usr/bin/env python3
"""Fetch RPG node content — retrieve code entities and feature entities.

Given code entity IDs (e.g., ``src/auth/login.py:LoginManager``) or feature
paths (e.g., ``authentication/login``), fetches node data and returns
formatted search results.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/ops/fetch_node.py
(the ``fetch_node`` function at the bottom of that file)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from rpg_agent.env.query import QueryInfo, QueryResult
from rpg_agent.env.searcher import RepoEntitySearcher
from rpg_agent.ops.search_by_meta import fuzzy_retrieve
from rpg import Node, NodeType, RPG


# ============================================================================
# TF-IDF feature path fuzzy matching (simplified)
# ============================================================================

def _collect_all_feature_paths(rpg: RPG) -> List[str]:
    """Return all unique feature paths from the RPG."""
    paths: List[str] = []
    for node in rpg.nodes.values():
        try:
            fp = node.feature_path()
        except TypeError:
            if hasattr(node, "feature_path") and isinstance(node.feature_path, str):
                fp = node.feature_path
            else:
                fp = None
        if fp:
            paths.append(fp)
    return sorted(set(paths))


def _get_deps_from_feature_id(feature_id: str, rpg: RPG) -> List[str]:
    """Return dep_graph node IDs mapped to the given RPG feature node ID."""
    dep2rpg = rpg._dep_to_rpg_map or {} if rpg else {}
    dep_ids: List[str] = []
    for dep_id, rpg_ids in dep2rpg.items():
        ids = rpg_ids if isinstance(rpg_ids, list) else [rpg_ids]
        if feature_id in ids:
            dep_ids.append(dep_id)
    return dep_ids


def _fuzzy_feature_paths(
    rpg: RPG,
    keyword: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Fuzzy match *keyword* against all feature paths using rapidfuzz.

    This is a simplified replacement for the TF-IDF approach in ZeroRepo,
    avoiding the ``scikit-learn`` / ``numpy`` dependency.

    Source: RPG-ZeroRepo search_node_by_feature.py ``_fuzzy_feature_paths``
    """
    from common.utils import normalize_text

    keyword = normalize_text(keyword or "")
    if not keyword:
        return []

    all_paths = _collect_all_feature_paths(rpg)
    if not all_paths:
        return []

    matches = process.extract(
        keyword,
        all_paths,
        scorer=fuzz.token_set_ratio,
        limit=top_k,
    )

    results: List[Tuple[str, float]] = []
    for path, score, _ in matches:
        if score > 0:
            results.append((path, float(score)))

    return results


# ============================================================================
# Main fetch function
# ============================================================================

def fetch_node(
    rpg: RPG,
    entity_searcher: RepoEntitySearcher,
    code_entities: Optional[List[str]] = None,
    feature_entities: Optional[List[str]] = None,
    similarity_top_k: int = 5,
) -> Tuple[str, bool]:
    """Return search result string (markdown) and success flag.

    Supports two kinds of input:
    - ``code_entities``: dep_graph node IDs (exact or fuzzy matched).
    - ``feature_entities``: RPG feature paths (exact or fuzzy matched).

    Args:
        rpg: RPG instance (contains dep_graph and _dep_to_rpg_map).
        entity_searcher: RepoEntitySearcher instance.
        code_entities: List of code entity IDs to retrieve.
        feature_entities: List of feature paths to retrieve.
        similarity_top_k: Number of fuzzy match candidates.

    Returns:
        (formatted_result, success) tuple.

    Source: RPG-ZeroRepo search_node_by_feature.py ``fetch_node``
    """
    if code_entities is None:
        code_entities = []
    if feature_entities is None:
        feature_entities = []

    dep_graph = rpg.dep_graph.G if rpg and rpg.dep_graph else None
    results: List[QueryResult] = []

    # ==================== Code Entities ====================
    for nid in code_entities:
        nid = (nid or "").strip()
        if nid.endswith(".__init__"):
            nid = nid[: -len(".__init__")]
        if not nid:
            continue

        # Exact match
        if entity_searcher.has_node(nid):
            ndata_list = entity_searcher.get_node_data([nid], return_code_content=True)
            if not ndata_list:
                continue
            ndata = ndata_list[0]
            qr = QueryResult(
                query_info=QueryInfo(term=nid),
                format_mode="complete",
                nid=nid,
                ntype=ndata.get("type"),
                start_line=ndata.get("start_line"),
                end_line=ndata.get("end_line"),
                retrieve_src="Exact entity match in repo graph",
            )
            results.append(qr)
            continue

        # Fuzzy match candidates
        try:
            matches = fuzzy_retrieve(
                keyword=nid,
                rpg=rpg,
                search_scope="all",
                include_files=None,
                similarity_top_k=similarity_top_k,
                return_score=True,
            )
        except Exception:
            all_nids = list(dep_graph.nodes()) if dep_graph else []
            matches = process.extract(
                nid, all_nids, scorer=fuzz.token_set_ratio, limit=similarity_top_k
            )

        if matches:
            normalized: List[Tuple[str, float]] = []
            for m in matches:
                try:
                    cand, score, *_ = m
                except Exception:
                    cand = m[0]
                    score = m[1] if len(m) > 1 else 0
                normalized.append((str(cand), float(score)))

            for cand, score in normalized:
                if not entity_searcher.has_node(cand):
                    continue
                ndata_list = entity_searcher.get_node_data([cand], return_code_content=True)
                if not ndata_list:
                    continue
                ndata = ndata_list[0]
                qr = QueryResult(
                    query_info=QueryInfo(term=nid),
                    format_mode="preview",
                    nid=cand,
                    ntype=ndata.get("type"),
                    start_line=ndata.get("start_line"),
                    end_line=ndata.get("end_line"),
                    retrieve_src=f"Fuzzy match (score {score:.1f}) -- refine query to get full code",
                )
                results.append(qr)

    # ==================== Feature Entities ====================
    for fpath in feature_entities:
        raw = (fpath or "").strip()
        if not raw:
            continue
        if raw.startswith("/"):
            raw = raw[1:]

        # Exact feature path -> node -> dep_id(s)
        fnode: Optional[Node] = rpg.get_node_by_feature_path(raw)

        if fnode:
            mapped_dep_ids = _get_deps_from_feature_id(fnode.id, rpg)
            for did in mapped_dep_ids:
                if not entity_searcher.has_node(did):
                    continue
                ndata_list = entity_searcher.get_node_data([did], return_code_content=True)
                if not ndata_list:
                    continue
                ndata = ndata_list[0]
                qr = QueryResult(
                    query_info=QueryInfo(term=raw),
                    format_mode="complete",
                    nid=did,
                    ntype=ndata.get("type"),
                    start_line=ndata.get("start_line"),
                    end_line=ndata.get("end_line"),
                    retrieve_src="Feature -> code mapping",
                )
                results.append(qr)
        else:
            # Fuzzy feature match
            cands = _fuzzy_feature_paths(rpg, raw, top_k=similarity_top_k)
            for cand_path, score in cands:
                cand_node = rpg.get_node_by_feature_path(cand_path)
                if not cand_node:
                    continue
                dep_ids = _get_deps_from_feature_id(cand_node.id, rpg)
                for did in dep_ids:
                    if not entity_searcher.has_node(did):
                        continue
                    ndata_list = entity_searcher.get_node_data([did], return_code_content=True)
                    if not ndata_list:
                        continue
                    ndata = ndata_list[0]
                    qr = QueryResult(
                        query_info=QueryInfo(term=raw),
                        format_mode="preview",
                        nid=did,
                        ntype=ndata.get("type"),
                        start_line=ndata.get("start_line"),
                        end_line=ndata.get("end_line"),
                        retrieve_src=(
                            f"Fuzzy feature match `{cand_path}` "
                            f"(similarity {score:.1f}%)"
                        ),
                    )
                    results.append(qr)

    # ==================== Format Output ====================
    if not results:
        return "No entities provided or no match found.", False

    out = "## Search Results\n\n"
    last_src = None
    for qr in results:
        text = qr.format_output(entity_searcher)
        if qr.retrieve_src != last_src:
            out += f"### Source: {qr.retrieve_src}\n"
            last_src = qr.retrieve_src
        out += text + "\n\n"

    return out.strip(), True
