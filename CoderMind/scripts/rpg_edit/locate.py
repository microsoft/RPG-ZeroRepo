#!/usr/bin/env python3
"""Locate candidate RPG feature nodes matching a natural language query.

Uses BM25-style keyword matching against node names, descriptions,
and dep_graph identifiers to find relevant feature nodes.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import REPO_RPG_FILE  # noqa: E402


def _build_tree_summary(svc, max_depth: int = 3, max_lines: int = 150) -> List[str]:
    """Build a compact indented tree of the RPG for orientation.

    Shows the first few levels of the feature tree so the agent can
    quickly understand the project structure without manually parsing
    the RPG JSON.  Leaf nodes include their node_id for direct use
    in ``--node-id`` arguments.

    For large repos, automatically reduces depth to stay within
    *max_lines* to avoid excessive prompt size.
    """
    lines: List[str] = []

    def _get_children(node):
        ch = node.children
        return ch() if callable(ch) else (ch or [])

    def _walk(node, depth: int = 0, effective_max: int = max_depth):
        if depth > effective_max:
            return
        indent = "  " * depth
        children = _get_children(node)
        name = getattr(node, "name", "?")
        nid = getattr(node, "id", "")
        if children and depth < effective_max:
            lines.append(f"{indent}▸ {name}")
            for child in children:
                _walk(child, depth + 1, effective_max)
        else:
            suffix = f"  [{nid}]" if nid else ""
            extra = ""
            if children:
                extra = f"  (+{len(children)} children)"
            lines.append(f"{indent}• {name}{extra}{suffix}")

    root = svc.rpg.repo_node
    if not root:
        return lines

    # Try with requested depth; if too many lines, reduce depth
    for depth in range(max_depth, 0, -1):
        lines.clear()
        _walk(root, 0, depth)
        if len(lines) <= max_lines:
            break

    if len(lines) > max_lines:
        lines[max_lines:] = [f"  ... (tree truncated, {len(svc.rpg._node_index)} total nodes)"]

    return lines


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r'[^a-z0-9_]+', text.lower()) if len(t) > 1]


def build_node_docs(svc) -> Dict[str, str]:
    """Build a searchable text document for each RPG node."""
    docs = {}
    for nid, node in svc.rpg._node_index.items():
        parts = [node.name]
        if node.meta:
            if node.meta.path:
                p = node.meta.path if isinstance(node.meta.path, str) else " ".join(node.meta.path)
                parts.append(p)
            if node.meta.description:
                parts.append(node.meta.description)
            if node.meta.type_name:
                tn = node.meta.type_name.value if hasattr(node.meta.type_name, 'value') else str(node.meta.type_name)
                parts.append(tn)
        # Add dep_graph entity names if mapped
        dep_nids = svc.rpg._feature_to_dep_map.get(nid, [])
        for dep_nid in dep_nids:
            parts.append(dep_nid.split(":")[-1] if ":" in dep_nid else dep_nid)
        # Add feature path
        try:
            fp = node.feature_path()
            if fp:
                parts.append(fp)
        except Exception:
            pass
        docs[nid] = " ".join(parts)
    return docs


def bm25_search(query: str, docs: Dict[str, str], top_k: int = 10) -> List[Dict]:
    """Simple BM25 ranking."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Build IDF
    N = len(docs)
    df = Counter()
    doc_tokens = {}
    doc_lengths = {}
    for nid, text in docs.items():
        tokens = tokenize(text)
        doc_tokens[nid] = tokens
        doc_lengths[nid] = len(tokens)
        for t in set(tokens):
            df[t] += 1

    avg_dl = sum(doc_lengths.values()) / max(N, 1)
    k1 = 1.5
    b = 0.75

    scores = {}
    for nid, tokens in doc_tokens.items():
        tf_map = Counter(tokens)
        score = 0.0
        dl = doc_lengths[nid]
        for qt in query_tokens:
            if qt not in df:
                continue
            idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
            tf = tf_map.get(qt, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        if score > 0:
            scores[nid] = score

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Locate RPG feature nodes by query")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--rpg", type=Path,
                        default=REPO_RPG_FILE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    from rpg.service import RPGService
    svc = RPGService.load(str(args.rpg))

    docs = build_node_docs(svc)
    ranked = bm25_search(args.query, docs, top_k=args.top_k)

    results = []
    for nid, score in ranked:
        node = svc.rpg._node_index[nid]
        entry = {
            "node_id": nid,
            "name": node.name,
            "score": round(score, 4),
            "level": node.level,
            "node_type": node.node_type,
        }
        if node.meta:
            entry["meta_path"] = node.meta.path
            if node.meta.type_name:
                entry["type_name"] = node.meta.type_name.value if hasattr(node.meta.type_name, 'value') else str(node.meta.type_name)
        try:
            entry["feature_path"] = node.feature_path()
        except Exception:
            pass
        # Include dep_graph node IDs if mapped
        dep_nids = svc.rpg._feature_to_dep_map.get(nid, [])
        if dep_nids:
            entry["dep_nodes"] = dep_nids
        results.append(entry)

    output = {"type": "candidates", "query": args.query, "results": results}

    # Add tree summary so the agent doesn't need to manually explore RPG JSON
    # when search results are poor (e.g. editing features that don't exist yet).
    tree_lines = _build_tree_summary(svc)
    output["tree_summary"] = tree_lines

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Query: {args.query}")
        print(f"Found {len(results)} candidates:\n")
        for r in results:
            print(f"  [{r['score']:.2f}] {r['name']} ({r.get('type_name', '?')})")
            print(f"         id: {r['node_id']}")
            if r.get("meta_path"):
                print(f"         path: {r['meta_path']}")
            print()
        if tree_lines:
            print("--- RPG Tree Overview ---")
            print("\n".join(tree_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
