#!/usr/bin/env python3
"""BM25 Search Model for RPG Agent.

Provides BM25-based retrieval for code entities and code content,
using `rank_bm25` (already a CoderMind dependency) instead of the
heavier llama_index-based implementation in RPG-ZeroRepo.

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/ops/bm25_model.py
Adaptations:
  - Replaced llama_index + Stemmer with rank_bm25.BM25Okapi
  - Removed multiprocessing document splitting (not needed for module search)
  - Kept the same public API surface

Key functions:
  - build_module_retriever: Build a BM25 index over RPG entity node IDs
  - bm25_search: Search the index with a query string
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from common.utils import is_test_file
from rpg import NodeType, RPG

logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Split text into tokens on underscores, hyphens, slashes, colons, dots, and camelCase boundaries, then lowercase."""
    # Insert spaces before uppercase letters (camelCase split)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Replace separators with spaces
    text = re.sub(r"[/_.\-:]+", " ", text)
    tokens = text.lower().split()
    return [t for t in tokens if t]


# ============================================================================
# Module Retriever
# ============================================================================

class ModuleRetriever:
    """Lightweight BM25 retriever over RPG dependency-graph node IDs.

    Each 'document' is the node ID string (e.g., ``src/auth/login.py:LoginManager``)
    tokenized into path/name segments.
    """

    def __init__(self, nids: List[str], similarity_top_k: int = 10):
        self._nids = nids
        self._top_k = similarity_top_k
        corpus = [_tokenize(nid) for nid in nids]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Return up to ``top_k`` (nid, score) pairs sorted by relevance."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Pair with nids and sort descending
        scored = sorted(zip(self._nids, scores), key=lambda x: -x[1])
        return [(nid, score) for nid, score in scored[:self._top_k] if score > 0]


def build_module_retriever(
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[object] = None,
    search_scope: str = "all",
    similarity_top_k: int = 10,
) -> ModuleRetriever:
    """Build a BM25 retriever for module / entity search.

    Args:
        rpg: RPG instance (preferred). Will use dep_graph from it.
        entity_searcher: RepoEntitySearcher instance (alternative to rpg).
            If provided, uses its ``G`` graph directly.
        search_scope: ``'all'`` or a specific ``NodeType`` value string.
        similarity_top_k: Number of top results to return.

    Returns:
        A ``ModuleRetriever`` instance.

    Source: RPG-ZeroRepo bm25_model.py ``build_module_retriever``
    """
    # Determine graph
    G = None
    if entity_searcher is not None and hasattr(entity_searcher, "G"):
        G = entity_searcher.G
    elif rpg is not None and rpg.dep_graph is not None:
        G = rpg.dep_graph.G
    if G is None:
        raise ValueError("No dependency graph available. Provide rpg or entity_searcher.")

    # Select node IDs
    selected_nids: List[str] = []
    scope_types = {
        NodeType.FILE, NodeType.CLASS, NodeType.METHOD, NodeType.FUNCTION,
    }

    for nid in G.nodes():
        if is_test_file(nid):
            continue
        ndata = G.nodes[nid]
        ntype = ndata.get("type")

        if search_scope == "all":
            if ntype in scope_types:
                selected_nids.append(nid)
        elif ntype == search_scope:
            selected_nids.append(nid)

    return ModuleRetriever(selected_nids, similarity_top_k)
