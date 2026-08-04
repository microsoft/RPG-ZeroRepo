"""Semantic RPG version reads and diffs backed by the workspace meta-git."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _git(meta_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=meta_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout


def read_rpg_version(meta_root: Path, relative_path: str, ref: str) -> dict[str, Any] | None:
    """Read one RPG JSON blob from the meta-git without touching the worktree."""
    payload = _git(meta_root, "show", f"{ref}:{relative_path}")
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def previous_rpg_version(meta_root: Path, relative_path: str, commit: str) -> str | None:
    """Return the previous commit that contains a different version of RPG_FILE."""
    previous = _git(
        meta_root,
        "log",
        "--max-count=1",
        "--format=%H",
        f"{commit}^",
        "--",
        relative_path,
    )
    return previous.strip() if previous and previous.strip() else None


def _public_node(
    node_id: str,
    value: dict[str, Any],
    *,
    parent_id: str | None = None,
) -> dict[str, Any]:
    meta = value.get("meta") if isinstance(value.get("meta"), dict) else {}
    return {
        "node_id": node_id,
        "name": value.get("name"),
        "node_type": value.get("node_type") or value.get("type"),
        "path": meta.get("path") or value.get("path"),
        "parent_id": parent_id,
    }


def _feature_index(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    values: dict[str, dict[str, Any]] = {}
    public: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any], parent_id: str | None) -> None:
        raw_id = node.get("id")
        if raw_id is None:
            return
        node_id = str(raw_id)
        own = {key: value for key, value in node.items() if key != "children"}
        own["parent_id"] = parent_id
        values[node_id] = own
        public[node_id] = _public_node(node_id, node, parent_id=parent_id)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, node_id)

    root = data.get("root")
    if isinstance(root, dict):
        walk(root, None)
    return values, public


def _dep_index(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dep_graph = data.get("dep_graph") or {}
    raw_nodes = dep_graph.get("nodes")
    values: dict[str, dict[str, Any]] = {}
    if isinstance(raw_nodes, dict):
        values = {
            str(node_id): dict(value) if isinstance(value, dict) else {"value": value}
            for node_id, value in raw_nodes.items()
        }
    elif isinstance(raw_nodes, list):
        for value in raw_nodes:
            if isinstance(value, dict) and value.get("id") is not None:
                values[str(value["id"])] = dict(value)
    parents: dict[str, str] = {}
    for edge in dep_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        attrs = edge.get("attrs") if isinstance(edge.get("attrs"), dict) else {}
        relation = edge.get("relation") or edge.get("type") or attrs.get("type")
        if str(relation).lower() == "contains" and edge.get("src") is not None and edge.get("dst") is not None:
            parents[str(edge["dst"])] = str(edge["src"])
    for node_id, value in values.items():
        value["parent_id"] = parents.get(node_id)
    public = {
        node_id: _public_node(node_id, value, parent_id=parents.get(node_id))
        for node_id, value in values.items()
    }
    return values, public


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


def _node_diff(
    before_values: dict[str, dict[str, Any]],
    before_public: dict[str, dict[str, Any]],
    after_values: dict[str, dict[str, Any]],
    after_public: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before_ids = set(before_values)
    after_ids = set(after_values)
    modified = []
    for node_id in sorted(before_ids & after_ids):
        changed = _changed_fields(before_values[node_id], after_values[node_id])
        if changed:
            modified.append({
                **after_public[node_id],
                "previous_parent_id": before_public[node_id].get("parent_id"),
                "changed_fields": changed,
            })
    added = [after_public[node_id] for node_id in sorted(after_ids - before_ids)]
    removed = [before_public[node_id] for node_id in sorted(before_ids - after_ids)]
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "counts": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "total": len(after_ids),
        },
    }


def _edge_key(value: dict[str, Any]) -> str:
    attrs = value.get("attrs") if isinstance(value.get("attrs"), dict) else {}
    source = value.get("src") if value.get("src") is not None else value.get("source")
    target = value.get("dst") if value.get("dst") is not None else value.get("target")
    relation = value.get("relation") or value.get("type") or attrs.get("type")
    return json.dumps([source, target, relation], ensure_ascii=False, separators=(",", ":"))


def _edge_index(values: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        return {}
    return {_edge_key(value): value for value in values if isinstance(value, dict)}


def _public_edge(value: dict[str, Any]) -> dict[str, Any]:
    attrs = value.get("attrs") if isinstance(value.get("attrs"), dict) else {}
    return {
        "source": value.get("src") if value.get("src") is not None else value.get("source"),
        "target": value.get("dst") if value.get("dst") is not None else value.get("target"),
        "relation": value.get("relation") or value.get("type") or attrs.get("type"),
    }


def _edge_diff(before: Any, after: Any) -> dict[str, Any]:
    before_values = _edge_index(before)
    after_values = _edge_index(after)
    before_keys = set(before_values)
    after_keys = set(after_values)
    return {
        "added": [_public_edge(after_values[key]) for key in sorted(after_keys - before_keys)],
        "removed": [_public_edge(before_values[key]) for key in sorted(before_keys - after_keys)],
        "counts": {
            "added": len(after_keys - before_keys),
            "removed": len(before_keys - after_keys),
            "total": len(after_keys),
        },
    }


def _hierarchy_edges(public_nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": node["parent_id"], "target": node_id, "relation": "contains"}
        for node_id, node in public_nodes.items()
        if node.get("parent_id") is not None
    ]


def _mapping_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("_dep_to_rpg_map")
    if not isinstance(raw, dict):
        return []
    return [
        {"source": str(dep_id), "target": str(feature_id), "relation": "maps_to"}
        for dep_id, feature_ids in raw.items()
        for feature_id in (feature_ids if isinstance(feature_ids, list) else [])
        if feature_id not in (None, "")
    ]


def semantic_rpg_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    commit: str,
    parent_commit: str | None,
) -> dict[str, Any]:
    """Return compact, renderer-ready node and edge changes between two RPGs."""
    before_feature, before_feature_public = _feature_index(before)
    after_feature, after_feature_public = _feature_index(after)
    before_dep, before_dep_public = _dep_index(before)
    after_dep, after_dep_public = _dep_index(after)
    feature_nodes = _node_diff(
        before_feature,
        before_feature_public,
        after_feature,
        after_feature_public,
    )
    dependency_nodes = _node_diff(before_dep, before_dep_public, after_dep, after_dep_public)
    semantic_edges = _edge_diff(before.get("edges"), after.get("edges"))
    dependency_edges = _edge_diff(
        (before.get("dep_graph") or {}).get("edges"),
        (after.get("dep_graph") or {}).get("edges"),
    )
    feature_hierarchy_edges = _edge_diff(
        _hierarchy_edges(before_feature_public),
        _hierarchy_edges(after_feature_public),
    )
    mapping_edges = _edge_diff(_mapping_edges(before), _mapping_edges(after))
    return {
        "available": True,
        "quality": "measured",
        "source": "meta-git adjacent rpg.json versions",
        "commit": commit,
        "short_commit": commit[:8],
        "parent_commit": parent_commit,
        "parent_short": parent_commit[:8] if parent_commit else None,
        "feature_nodes": feature_nodes,
        "dependency_nodes": dependency_nodes,
        "semantic_edges": semantic_edges,
        "dependency_edges": dependency_edges,
        "feature_hierarchy_edges": feature_hierarchy_edges,
        "mapping_edges": mapping_edges,
        "summary": {
            "feature_nodes_added": feature_nodes["counts"]["added"],
            "feature_nodes_removed": feature_nodes["counts"]["removed"],
            "feature_nodes_modified": feature_nodes["counts"]["modified"],
            "dependency_nodes_added": dependency_nodes["counts"]["added"],
            "dependency_nodes_removed": dependency_nodes["counts"]["removed"],
            "dependency_nodes_modified": dependency_nodes["counts"]["modified"],
            "semantic_edges_added": semantic_edges["counts"]["added"],
            "semantic_edges_removed": semantic_edges["counts"]["removed"],
            "dependency_edges_added": dependency_edges["counts"]["added"],
            "dependency_edges_removed": dependency_edges["counts"]["removed"],
            "feature_hierarchy_edges_added": feature_hierarchy_edges["counts"]["added"],
            "feature_hierarchy_edges_removed": feature_hierarchy_edges["counts"]["removed"],
            "mapping_edges_added": mapping_edges["counts"]["added"],
            "mapping_edges_removed": mapping_edges["counts"]["removed"],
        },
    }