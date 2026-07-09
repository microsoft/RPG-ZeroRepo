#!/usr/bin/env python3
"""Unified graph update tool — update dep_graph, feature graph, or both.

Subcommands:
    dep         Rebuild the AST dependency graph and embed it in rpg.json
  enrich      Enrich feature graph from actual code (align paths + fill missing)
  sync        Full sync: dep + enrich + mappings
  update-rpg  Full RPG update (dep_graph + feature graph via LLM) against
              the previous git commit. Designed to run in the background
              from post-commit hooks.
  mapping     Rebuild dep_graph + dep↔rpg mappings (legacy)
  feature     Load existing dep_graph, rebuild mappings (legacy)
  full        AST scan + mappings + edges (legacy, use 'sync' instead)

Usage:
  cmind script update_graphs.py dep --json
  cmind script update_graphs.py enrich --json
  cmind script update_graphs.py enrich --file models/user.py --dry-run --json
  cmind script update_graphs.py sync --json
  cmind script update_graphs.py update-rpg --json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import REPO_RPG_FILE, DEP_GRAPH_FILE, RPG_HTML_FILE, HOOK_CALLS_LOG  # noqa: E402
from common.rpg_io import atomic_write_rpg, safe_load_rpg  # noqa: E402
from common.run_events import (  # noqa: E402
    ArtifactEvent,
    CodeDeltaEvent,
    CommandRun,
    DepGraphDeltaEvent,
    RPGDeltaEvent,
    StepEvent,
    VerificationEvent,
)
from common.run_report import write_command_report  # noqa: E402


# Shared message used by every subcommand that requires an existing
# ``rpg.json`` (sync, update-rpg, ...).  Surfaces in two places:
#   * ``.cmind/logs/update_rpg.log`` for the asynchronous post-commit
#     phase — where it's the user's only diagnostic.
#   * stdout / JSON output for direct CLI invocations.
# Keep the message single-line so it survives JSON serialisation cleanly
# and stays easy to grep.
_RPG_MISSING_MSG = (
    "rpg.json not found at {rpg_path}. Run /cmind.encode in your AI agent "
    "to generate it; the post-commit hook will resume keeping it in sync "
    "on the next commit."
)


def _log_hook_call(hook_type: str, result: dict) -> None:
    """Append a single-line JSON record to the hook calls log.

    Best-effort: never raises.
    """
    try:
        from datetime import datetime, timezone
        HOOK_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hook": hook_type,
            "mode": result.get("mode", ""),
            "reason": result.get("reason", ""),
            "dep_nodes": result.get("dep_nodes"),
            "dep_edges": result.get("dep_edges"),
            "modified": result.get("modified"),
            "added": result.get("added"),
            "deleted": result.get("deleted"),
            "rpg_nodes": result.get("rpg_nodes"),
            "duration_ms": int(result.get("duration", 0) * 1000),
        }
        # Strip None values to keep lines compact
        record = {k: v for k, v in record.items() if v is not None}
        with open(HOOK_CALLS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _change_count(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _diff_summary(result: dict) -> dict:
    summary = result.get("diff_summary")
    if isinstance(summary, dict):
        return {
            "added": _change_count(summary.get("added")),
            "deleted": _change_count(summary.get("deleted")),
            "modified": _change_count(summary.get("modified")),
            "renamed": _change_count(summary.get("renamed")),
        }
    return {
        key: _change_count(result.get(key))
        for key in ("added", "deleted", "modified", "renamed")
    }


def _format_count_delta(value: object, delta: object) -> object:
    if value in (None, ""):
        return ""
    if isinstance(delta, int):
        return f"{value} (delta: {delta:+d})"
    return value


def _format_diff_summary(summary: dict) -> str:
    total = sum(summary.values())
    parts = [f"{total} semantic files"]
    for key in ("added", "deleted", "modified", "renamed"):
        count = summary.get(key, 0)
        if count:
            parts.append(f"{key}={count}")
    return ", ".join(parts)


def _git_change_type(status: object) -> str:
    code = str(status or "").upper()
    if code.startswith("A"):
        return "added"
    if code.startswith("D"):
        return "deleted"
    if code.startswith("R"):
        return "renamed"
    if code.startswith("C"):
        return "copied"
    if code.startswith("T"):
        return "typechanged"
    if code.startswith("U"):
        return "unmerged"
    return "modified" if code.startswith("M") else (code.lower() or "changed")


def _git_diff_text(prev_ref: str, workspace_root: str, paths: list[str]) -> str:
    import subprocess

    if not paths:
        return ""
    try:
        return subprocess.check_output(
            ["git", "diff", "--relative", f"{prev_ref}..HEAD", "--", *paths],
            cwd=workspace_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_delta_files(prev_ref: str, workspace_root: str) -> list[dict[str, str]]:
    import subprocess

    try:
        output = subprocess.check_output(
            [
                "git",
                "diff",
                "--relative",
                "--name-status",
                "--find-renames",
                f"{prev_ref}..HEAD",
                "--",
                ".",
            ],
            cwd=workspace_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    files: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[-1]
        before = parts[1] if status.upper().startswith(("R", "C")) and len(parts) >= 3 else path
        diff_paths = [before, path] if before != path else [path]
        row = {
            "status": status,
            "change_type": _git_change_type(status),
            "path": path,
            "file": path,
            "diff": _git_diff_text(prev_ref, workspace_root, diff_paths),
        }
        if before != path:
            row["before"] = before
            row["after"] = path
        elif row["change_type"] == "added":
            row["after"] = path
        elif row["change_type"] == "deleted":
            row["before"] = path
        files.append(row)
    return files


def _listify(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _ordered_unique_text(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _code_delta_rows(result: dict) -> list[dict[str, Any]]:
    rows = result.get("code_deltas") or result.get("git_delta") or []
    normalized: list[dict[str, Any]] = []
    for row in _listify(rows):
        if isinstance(row, dict):
            path = row.get("file") or row.get("path") or row.get("after") or row.get("before")
            if path in (None, ""):
                continue
            item = dict(row)
            item["file"] = path
            item["path"] = path
            item["change_type"] = item.get("change_type") or _git_change_type(item.get("status"))
            item.setdefault("diff", "")
            normalized.append(item)
        elif row not in (None, ""):
            normalized.append({"file": str(row), "path": str(row), "change_type": "changed", "diff": ""})
    return normalized


def _diff_files(result: dict) -> list[str]:
    diff_files = result.get("diff_files")
    files: list[Any] = []
    if isinstance(diff_files, dict):
        for value in diff_files.values():
            if isinstance(value, dict):
                files.extend(value.keys())
            else:
                files.extend(_listify(value))
    return _ordered_unique_text(files)


def _changed_report_files(result: dict, code_deltas: list[dict[str, Any]]) -> list[str]:
    files: list[Any] = _diff_files(result)
    for row in code_deltas:
        files.append(row.get("file") or row.get("path") or row.get("after") or row.get("before"))
    return _ordered_unique_text(files)


def _load_report_rpg(result: dict) -> dict[str, Any]:
    rpg_path = result.get("output_path") or result.get("rpg_path")
    if not rpg_path:
        return {}
    try:
        return safe_load_rpg(Path(str(rpg_path)))
    except Exception:
        return {}


def _flatten_rpg_tree(root: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    nodes: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}

    def visit(node: Any, ancestors: list[str]) -> None:
        if not isinstance(node, dict):
            return
        node_id = node.get("id") or node.get("node_id")
        name = node.get("name") or node_id
        next_ancestors = ancestors + ([str(name)] if name not in (None, "") else [])
        if node_id not in (None, ""):
            node_id_text = str(node_id)
            nodes[node_id_text] = node
            paths[node_id_text] = " / ".join(next_ancestors)
        for child in node.get("children") or []:
            visit(child, next_ancestors)

    visit(root, [])
    return nodes, paths


def _dep_node_items(dep_nodes: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(dep_nodes, dict):
        return [(str(node_id), attrs if isinstance(attrs, dict) else {}) for node_id, attrs in dep_nodes.items()]
    if isinstance(dep_nodes, list):
        rows = []
        for index, attrs in enumerate(dep_nodes):
            if not isinstance(attrs, dict):
                continue
            node_id = attrs.get("id") or attrs.get("node_id") or attrs.get("dep_node_id") or index
            rows.append((str(node_id), attrs))
        return rows
    return []


def _dep_node_path(dep_id: str, attrs: dict[str, Any]) -> str:
    for key in ("path", "file", "module", "code_path"):
        value = attrs.get(key)
        if value not in (None, ""):
            text = str(value)
            if key == "code_path" and os.path.isabs(text):
                try:
                    return os.path.relpath(text, os.getcwd())
                except ValueError:
                    return text
            return text
    return dep_id.split(":", 1)[0]


def _dep_node_matches_file(dep_id: str, attrs: dict[str, Any], file_path: str) -> bool:
    candidates = {dep_id, dep_id.split(":", 1)[0], _dep_node_path(dep_id, attrs)}
    for key in ("path", "file", "module", "code_path"):
        value = attrs.get(key)
        if value not in (None, ""):
            candidates.add(str(value))
    for candidate in candidates:
        if candidate == file_path or candidate.endswith("/" + file_path):
            return True
        if candidate.startswith(file_path + ":"):
            return True
    return False


def _slug_id(prefix: str, value: Any) -> str:
    raw = str(value or prefix)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-_")
    return f"{prefix}-{slug or 'node'}"


def _node_link_id(kind: str, node_id: Any) -> str:
    return _slug_id(kind, node_id)


def _set_if_present(row: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        row[key] = value


def _code_delta_file(delta: dict[str, Any]) -> str:
    return str(delta.get("file") or delta.get("path") or delta.get("after") or delta.get("before") or "")


def _diff_anchor_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "change"


def _diff_anchor_map(code_deltas: list[dict[str, Any]]) -> dict[str, str]:
    anchors: dict[str, str] = {}
    used: dict[str, int] = {}
    for index, delta in enumerate(code_deltas, start=1):
        file_path = _code_delta_file(delta) or f"change-{index}"
        base = _diff_anchor_slug(f"diff-{file_path}")
        count = used.get(base, 0) + 1
        used[base] = count
        if file_path not in anchors:
            anchors[file_path] = base if count == 1 else f"{base}-{count}"
    return anchors


def _changed_file_refs(files: list[Any], anchors: dict[str, str]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for file_path in _ordered_unique_text(files):
        row: dict[str, Any] = {"path": file_path}
        _set_if_present(row, "diff_anchor", anchors.get(file_path))
        refs.append(row)
    return refs


def _node_state(row: dict[str, Any], default: str = "available") -> str:
    status = row.get("state") or row.get("status") or row.get("mapping_status")
    if status in (None, ""):
        return default
    if status == "missing":
        return "missing_mapping"
    return str(status)


def _line_range_from(row: dict[str, Any]) -> dict[str, Any]:
    start = row.get("start_line") or row.get("lineno") or row.get("line") or row.get("start")
    end = row.get("end_line") or row.get("end") or start
    if start in (None, ""):
        return {}
    return {"start": start, "end": end}


def _symbol_from_dep(dep_id: str, attrs: dict[str, Any]) -> str:
    for key in ("symbol", "name", "qualname"):
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    if ":" in dep_id:
        return dep_id.rsplit(":", 1)[-1]
    path = attrs.get("path") or attrs.get("file") or dep_id
    return Path(str(path)).name


def _edge_endpoint_link_id(node_id: Any, rpg_nodes: dict[str, dict[str, Any]], code_nodes: dict[str, dict[str, Any]]) -> str:
    node_text = str(node_id or "")
    if node_text in rpg_nodes:
        return str(rpg_nodes[node_text].get("link_id") or _node_link_id("rpg", node_text))
    if node_text in code_nodes:
        return str(code_nodes[node_text].get("link_id") or _node_link_id("code", node_text))
    return _node_link_id("context", node_text)


def _warning_link_fields(warning: dict[str, Any], rpg_nodes: dict[str, dict[str, Any]], code_nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = dict(warning)
    node_id = warning.get("node_id") or warning.get("rpg_node_id")
    dep_id = warning.get("dep_node_id") or warning.get("code_node_id")
    if node_id not in (None, "") and str(node_id) in rpg_nodes:
        row["node_link_id"] = _edge_endpoint_link_id(node_id, rpg_nodes, code_nodes)
    if dep_id not in (None, "") and str(dep_id) in code_nodes:
        row["code_link_id"] = _edge_endpoint_link_id(dep_id, rpg_nodes, code_nodes)
    return row


def _hierarchy_segments(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    text = str(value)
    separator = " / " if " / " in text else "/"
    return [part.strip() for part in text.split(separator) if part.strip()]


def _hierarchy_child(parent: dict[str, Any], child_id: str, name: str, kind: str) -> dict[str, Any]:
    children = parent.setdefault("children", [])
    for child in children:
        if isinstance(child, dict) and child.get("id") == child_id:
            return child
    child = {"id": child_id, "name": name, "kind": kind, "children": []}
    children.append(child)
    return child


def _append_hierarchy_leaf(parent: dict[str, Any], leaf: dict[str, Any]) -> None:
    children = parent.setdefault("children", [])
    leaf_id = leaf.get("id")
    if any(isinstance(child, dict) and child.get("id") == leaf_id for child in children):
        return
    children.append(leaf)


def _semantic_hierarchy_parts(node: dict[str, Any]) -> list[str]:
    for key in ("breadcrumb", "breadcrumb_path", "feature_path", "path"):
        parts = _hierarchy_segments(node.get(key))
        if parts:
            return parts[:-1] if len(parts) > 1 else []
    return []


def _focused_graph_hierarchy(
    semantic_nodes: list[dict[str, Any]],
    code_nodes: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    hidden_counts: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    root: dict[str, Any] = {
        "id": "focused-graph-root",
        "name": "Focused graph",
        "kind": "root",
        "feature_name": "Focused graph",
        "feature_path": "Focused graph",
        "meta": {"hidden_counts": hidden_counts, "warnings": len(warnings), "edges": len(edges)},
        "children": [],
    }
    attached_endpoint_ids: set[str] = set()
    code_by_id = {
        str(node.get("node_id") or node.get("dep_node_id")): node
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }

    def feature_path_text(node: dict[str, Any], group_parts: list[str], feature_name: str) -> str:
        for key in ("feature_path", "breadcrumb_path", "breadcrumb", "path"):
            parts = _hierarchy_segments(node.get(key))
            if parts:
                return " / ".join(parts)
        return " / ".join(group_parts + ([feature_name] if feature_name else []))

    def endpoint_group(parent: dict[str, Any], owner_id: str, name: str, kind: str) -> dict[str, Any]:
        return _hierarchy_child(parent, _node_link_id(kind, owner_id), name, kind)

    def make_code_leaf(ref: dict[str, Any]) -> dict[str, Any]:
        code_id = str(ref.get("node_id") or ref.get("dep_node_id") or "")
        code = code_by_id.get(code_id, {})
        path = ref.get("path") or code.get("path") or code.get("file") or code.get("module") or _dep_node_path(code_id, code)
        symbol = ref.get("symbol") or ref.get("name") or code.get("symbol") or code.get("name") or _symbol_from_dep(code_id, code)
        leaf: dict[str, Any] = {
            "id": str(ref.get("link_id") or code.get("link_id") or _node_link_id("code", code_id)),
            "node_id": code_id,
            "dep_node_id": code_id,
            "name": symbol or path or code_id,
            "symbol": symbol,
            "path": path,
            "kind": "code",
            "state": ref.get("state") or code.get("state") or _node_state(code, "mapped"),
            "aliases": _ordered_unique_text([code_id, ref.get("link_id"), code.get("link_id")]),
        }
        for key in ("type", "line_range", "source", "changed", "changed_files", "diff_anchor"):
            _set_if_present(leaf, key, ref.get(key) or code.get(key))
        return leaf

    def append_endpoint(parent: dict[str, Any], leaf: dict[str, Any]) -> None:
        leaf_id = str(leaf.get("id") or "")
        if not leaf_id or leaf_id in attached_endpoint_ids:
            return
        attached_endpoint_ids.add(leaf_id)
        _append_hierarchy_leaf(parent, leaf)

    for node in semantic_nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        link_id = str(node.get("link_id") or _node_link_id("rpg", node_id))
        feature_name = str(node.get("name") or node.get("symbol") or node_id or "feature")
        group_parts = _semantic_hierarchy_parts(node)
        parent = root
        trail: list[str] = []
        for part in group_parts:
            trail.append(part)
            group = _hierarchy_child(parent, _node_link_id("feature-path", "/".join(trail)), part, "feature_group")
            group["feature_name"] = part
            group["feature_path"] = " / ".join(trail)
            parent = group
        leaf: dict[str, Any] = {
            "id": link_id,
            "node_id": node_id,
            "name": feature_name,
            "feature_name": feature_name,
            "feature_path": feature_path_text(node, group_parts, feature_name),
            "kind": "feature",
            "state": node.get("state"),
            "mapping_status": node.get("mapping_status"),
        }
        for key in (
            "type",
            "node_type",
            "path",
            "breadcrumb",
            "breadcrumb_path",
            "changed_files",
            "hidden_counts",
            "warning_types",
            "source",
            "mapped_code",
            "mapped_code_node_ids",
            "mapped_code_link_ids",
            "mapped_code_paths",
            "mapped_code_symbols",
            "mapped_code_path",
            "mapped_code_symbol",
            "mapped_code_count",
        ):
            _set_if_present(leaf, key, node.get(key))
        code_group: Optional[dict[str, Any]] = None
        for ref in _listify(node.get("mapped_code")):
            if not isinstance(ref, dict):
                continue
            if code_group is None:
                code_group = endpoint_group(leaf, link_id, "Mapped code", "code_group")
            append_endpoint(code_group, make_code_leaf(ref))
        _append_hierarchy_leaf(parent, leaf)

    root_code_group: Optional[dict[str, Any]] = None
    for code in code_nodes:
        code_id = str(code.get("node_id") or code.get("dep_node_id") or "")
        if not code_id:
            continue
        leaf = make_code_leaf({"node_id": code_id, "link_id": code.get("link_id")})
        if str(leaf.get("id") or "") in attached_endpoint_ids:
            continue
        if root_code_group is None:
            root_code_group = endpoint_group(root, "unassigned", "Additional code context", "code_group")
        append_endpoint(root_code_group, leaf)

    root_context_group: Optional[dict[str, Any]] = None
    known_links = {str(node.get("link_id") or "") for node in semantic_nodes + code_nodes}
    for edge in edges:
        for side in ("source", "target"):
            link_id = str(edge.get(f"{side}_link_id") or "")
            node_id = str(edge.get(f"{side}_node_id") or link_id)
            if not link_id or link_id in known_links or link_id in attached_endpoint_ids:
                continue
            leaf = {
                "id": link_id,
                "node_id": node_id,
                "name": edge.get("name") or edge.get("path") or node_id or "context",
                "kind": "context",
                "state": "context",
                "relation": edge.get("relation"),
                "source": edge.get("source") or edge.get("source_graph") or edge.get("edge_source"),
                "aliases": _ordered_unique_text([node_id, link_id]),
            }
            for key in ("path", "reason", "source_graph", "edge_source", "relation_source"):
                _set_if_present(leaf, key, edge.get(key))
            if root_context_group is None:
                root_context_group = endpoint_group(root, "unassigned", "Additional relation endpoints", "context_group")
            append_endpoint(root_context_group, leaf)
    return root


def _focused_graph_default_focus(
    semantic_nodes: list[dict[str, Any]],
    code_nodes: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic_links = [str(node.get("link_id") or _node_link_id("rpg", node.get("node_id"))) for node in semantic_nodes]
    focused_semantic_links = [
        str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        for node in semantic_nodes
        if node.get("changed") or node.get("changed_files") or node.get("diff_anchor") or node.get("change")
    ]
    code_links = [str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id"))) for node in code_nodes]
    rpg_link_by_node_id = {
        str(node.get("node_id")): str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        for node in semantic_nodes
        if node.get("node_id") not in (None, "")
    }
    code_link_by_node_id = {
        str(node.get("node_id") or node.get("dep_node_id")): str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id")))
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }
    code_to_feature_links: dict[str, list[str]] = {}
    hierarchy_paths_by_link: dict[str, list[str]] = {}

    def remember_code_feature(code_link: Any, feature_link: Any) -> None:
        if code_link in (None, "") or feature_link in (None, ""):
            return
        values = code_to_feature_links.setdefault(str(code_link), [])
        feature_text = str(feature_link)
        if feature_text not in values:
            values.append(feature_text)

    def remember_path(alias: Any, path_ids: list[str]) -> None:
        if alias in (None, "") or not path_ids:
            return
        hierarchy_paths_by_link.setdefault(str(alias), path_ids)

    semantic_path_ids_by_link: dict[str, list[str]] = {}
    for node in semantic_nodes:
        link_id = str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        trail: list[str] = []
        path_ids: list[str] = ["focused-graph-root"]
        for part in _semantic_hierarchy_parts(node):
            trail.append(part)
            path_ids.append(_node_link_id("feature-path", "/".join(trail)))
        path_ids.append(link_id)
        semantic_path_ids_by_link[link_id] = path_ids
        remember_path(link_id, path_ids)
        remember_path(node.get("node_id"), path_ids)

    for mapping in mappings:
        source = mapping.get("source_link_id") or rpg_link_by_node_id.get(str(mapping.get("rpg_node_id") or mapping.get("node_id") or ""))
        target_node_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        target = mapping.get("target_link_id") or code_link_by_node_id.get(str(target_node_id or "")) or _node_link_id("code", target_node_id)
        remember_code_feature(target, source)
        remember_code_feature(target_node_id, source)
    for node in code_nodes:
        node_id = node.get("node_id") or node.get("dep_node_id")
        code_link = str(node.get("link_id") or _node_link_id("code", node_id))
        for rpg_link in _listify(node.get("mapped_rpg_link_ids")):
            remember_code_feature(code_link, rpg_link)
            remember_code_feature(node_id, rpg_link)

    for code_id, code_link in code_link_by_node_id.items():
        feature_links = code_to_feature_links.get(code_link) or code_to_feature_links.get(code_id) or []
        if feature_links:
            base_path = semantic_path_ids_by_link.get(str(feature_links[0]))
            if base_path:
                code_path = base_path + [_node_link_id("code_group", feature_links[0]), code_link]
            else:
                code_path = ["focused-graph-root", _node_link_id("code_group", "unassigned"), code_link]
        else:
            code_path = ["focused-graph-root", _node_link_id("code_group", "unassigned"), code_link]
        remember_path(code_link, code_path)
        remember_path(code_id, code_path)

    endpoint_link_ids: list[Any] = []
    context_link_ids: list[Any] = []
    semantic_link_set = set(semantic_links)
    code_link_set = set(code_links)
    for edge in edges:
        for side in ("source", "target"):
            link_id = str(edge.get(f"{side}_link_id") or "")
            node_id = edge.get(f"{side}_node_id")
            if not link_id:
                continue
            endpoint_link_ids.append(link_id)
            endpoint_link_ids.append(node_id)
            if link_id in semantic_link_set or link_id in code_link_set or str(node_id or "") in code_link_by_node_id:
                continue
            context_link_ids.append(link_id)
            context_path = ["focused-graph-root", _node_link_id("context_group", "unassigned"), link_id]
            remember_path(link_id, context_path)
            remember_path(node_id, context_path)

    changed_feature_links: list[Any] = []
    changed_code_links: list[str] = []
    for node in code_nodes:
        link_id = str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id")))
        if node.get("changed") or node.get("changed_files") or node.get("diff_anchor"):
            changed_code_links.append(link_id)
            changed_feature_links.extend(_listify(node.get("mapped_rpg_link_ids")) + code_to_feature_links.get(link_id, []))
    node_link_ids = _ordered_unique_text(focused_semantic_links + changed_feature_links + changed_code_links)

    expanded_node_ids: list[Any] = ["focused-graph-root"]
    focused_path_node_ids: list[Any] = []
    focused_tree_node_ids: list[Any] = []
    for link_id in node_link_ids:
        path_ids = hierarchy_paths_by_link.get(str(link_id or ""))
        if not path_ids:
            continue
        expanded_node_ids.extend(path_ids[:-1])
        focused_path_node_ids.extend(path_ids)
        focused_tree_node_ids.append(path_ids[-1])

    return {
        "node_link_ids": node_link_ids,
        "focused_node_ids": node_link_ids,
        "focused_tree_node_ids": _ordered_unique_text(focused_tree_node_ids),
        "focused_code_link_ids": _ordered_unique_text(changed_code_links + [link_id for link_id in node_link_ids if link_id in code_link_set]),
        "expanded_node_ids": _ordered_unique_text(expanded_node_ids),
        "default_expanded_node_ids": _ordered_unique_text(expanded_node_ids),
        "focused_path_node_ids": _ordered_unique_text(focused_path_node_ids),
        "semantic_node_ids": _ordered_unique_text([node.get("node_id") for node in semantic_nodes]),
        "code_node_ids": _ordered_unique_text([node.get("node_id") or node.get("dep_node_id") for node in code_nodes]),
        "mapping_link_ids": _ordered_unique_text([mapping.get("link_id") for mapping in mappings]),
        "edge_link_ids": _ordered_unique_text([edge.get("link_id") for edge in edges]),
        "relation_endpoint_link_ids": _ordered_unique_text(endpoint_link_ids),
        "context_node_ids": _ordered_unique_text(context_link_ids),
        "edge_depth": 1,
        "show_edges": True,
    }


def _focused_graph_metadata(
    semantic_nodes: list[dict[str, Any]],
    code_nodes: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    hidden_counts: dict[str, Any],
    warnings: list[dict[str, Any]],
    graph_context: dict[str, Any],
) -> dict[str, Any]:
    focused_graph: dict[str, Any] = {
        "schema": "cmind.focused_graph.v1",
        "hierarchy": _focused_graph_hierarchy(semantic_nodes, code_nodes, mappings, edges, hidden_counts, warnings),
        "default_focus": _focused_graph_default_focus(semantic_nodes, code_nodes, mappings, edges, warnings),
    }
    if graph_context:
        focused_graph["graph_context"] = graph_context
    if hidden_counts:
        focused_graph["hidden_counts"] = hidden_counts
    if warnings:
        focused_graph["warning_count"] = len(warnings)
    return focused_graph


def _build_nodes_view(
    primary_rpg_nodes: list[dict[str, Any]],
    primary_code_nodes: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    hidden_counts: dict[str, Any],
    warnings: list[dict[str, Any]],
    changed_files: list[str],
    diff_anchors: dict[str, str],
    graph_context: dict[str, Any],
) -> dict[str, Any]:
    rpg_by_id = {str(row.get("node_id")): row for row in primary_rpg_nodes if row.get("node_id") not in (None, "")}
    code_by_id = {str(row.get("node_id") or row.get("dep_node_id")): row for row in primary_code_nodes if (row.get("node_id") or row.get("dep_node_id")) not in (None, "")}
    code_ids_by_rpg: dict[str, list[str]] = {}
    warnings_by_rpg: dict[str, list[dict[str, Any]]] = {}
    warnings_by_code: dict[str, list[dict[str, Any]]] = {}
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        if rpg_id not in (None, "") and code_id not in (None, ""):
            code_ids_by_rpg.setdefault(str(rpg_id), []).append(str(code_id))
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        rpg_id = warning.get("node_id") or warning.get("rpg_node_id")
        code_id = warning.get("dep_node_id") or warning.get("code_node_id")
        if rpg_id not in (None, ""):
            warnings_by_rpg.setdefault(str(rpg_id), []).append(warning)
        if code_id not in (None, ""):
            warnings_by_code.setdefault(str(code_id), []).append(warning)

    semantic_nodes: list[dict[str, Any]] = []
    for node in primary_rpg_nodes:
        node_id = str(node.get("node_id") or "")
        row: dict[str, Any] = {
            "node_id": node_id,
            "link_id": node.get("link_id") or _node_link_id("rpg", node_id),
            "state": _node_state(node),
            "mapping_status": node.get("mapping_status") or node.get("status"),
        }
        for key in ("name", "symbol", "type", "node_type", "path", "feature_path", "breadcrumb", "breadcrumb_path", "change", "source"):
            _set_if_present(row, key, node.get(key))
        changed_refs = _changed_file_refs(_listify(node.get("changed_files")) or _listify(node.get("affected_files")), diff_anchors)
        if changed_refs:
            row["changed_files"] = changed_refs
        mapped_code_ids = _ordered_unique_text(code_ids_by_rpg.get(node_id, []))
        if mapped_code_ids:
            mapped_code_refs: list[dict[str, Any]] = []
            for code_id in mapped_code_ids:
                code = code_by_id.get(code_id, {})
                path = code.get("path") or code.get("file") or code.get("module") or _dep_node_path(code_id, code)
                symbol = code.get("symbol") or code.get("name") or _symbol_from_dep(code_id, code)
                ref: dict[str, Any] = {
                    "node_id": code_id,
                    "dep_node_id": code_id,
                    "link_id": _edge_endpoint_link_id(code_id, rpg_by_id, code_by_id),
                    "path": path,
                    "symbol": symbol,
                }
                for key in ("type", "kind", "line_range", "state", "source", "diff_anchor"):
                    _set_if_present(ref, key, code.get(key))
                mapped_code_refs.append(ref)
            row["mapped_code"] = mapped_code_refs
            row["mapped_code_node_ids"] = mapped_code_ids
            row["mapped_code_link_ids"] = [ref["link_id"] for ref in mapped_code_refs]
            row["mapped_code_paths"] = _ordered_unique_text([ref.get("path") for ref in mapped_code_refs])
            row["mapped_code_symbols"] = _ordered_unique_text([ref.get("symbol") for ref in mapped_code_refs])
            if row["mapped_code_paths"]:
                row["mapped_code_path"] = row["mapped_code_paths"][0]
            if row["mapped_code_symbols"]:
                row["mapped_code_symbol"] = row["mapped_code_symbols"][0]
            row["mapped_code_count"] = len(mapped_code_refs)
        if warnings_by_rpg.get(node_id):
            row["warning_types"] = _ordered_unique_text([warning.get("type") for warning in warnings_by_rpg[node_id]])
        semantic_nodes.append(row)

    code_nodes: list[dict[str, Any]] = []
    for node in primary_code_nodes:
        code_id = str(node.get("node_id") or node.get("dep_node_id") or "")
        path = node.get("path") or _dep_node_path(code_id, node)
        row: dict[str, Any] = {
            "node_id": code_id,
            "dep_node_id": code_id,
            "link_id": node.get("link_id") or _node_link_id("code", code_id),
            "state": _node_state(node, "mapped"),
        }
        for key in ("name", "symbol", "type", "kind", "path", "module", "file", "signature", "source", "change"):
            _set_if_present(row, key, node.get(key))
        _set_if_present(row, "path", path)
        line_range = node.get("line_range") if isinstance(node.get("line_range"), dict) else _line_range_from(node)
        if line_range:
            row["line_range"] = line_range
        changed_refs = _changed_file_refs(_listify(node.get("changed_files")) or ([path] if path in diff_anchors else []), diff_anchors)
        if changed_refs:
            row["changed"] = True
            row["changed_files"] = changed_refs
            row["diff_anchor"] = changed_refs[0].get("diff_anchor")
        rpg_ids = _ordered_unique_text(
            _listify(node.get("mapped_rpg_node_ids"))
            + [mapping.get("rpg_node_id") for mapping in mappings if (mapping.get("code_node_id") or mapping.get("dep_node_id")) == code_id]
        )
        if rpg_ids:
            mapped_rpg_refs: list[dict[str, Any]] = []
            for rpg_id in rpg_ids:
                rpg = rpg_by_id.get(rpg_id, {})
                ref: dict[str, Any] = {
                    "node_id": rpg_id,
                    "link_id": _edge_endpoint_link_id(rpg_id, rpg_by_id, code_by_id),
                    "name": rpg.get("name") or rpg_id,
                }
                for key in ("type", "node_type", "path", "feature_path", "breadcrumb_path", "state", "source"):
                    _set_if_present(ref, key, rpg.get(key))
                mapped_rpg_refs.append(ref)
            row["mapped_rpg"] = mapped_rpg_refs
            row["mapped_rpg_node_ids"] = rpg_ids
            row["mapped_rpg_link_ids"] = [ref["link_id"] for ref in mapped_rpg_refs]
        if warnings_by_code.get(code_id):
            row["warning_types"] = _ordered_unique_text([warning.get("type") for warning in warnings_by_code[code_id]])
        code_nodes.append(row)

    mapping_rows: list[dict[str, Any]] = []
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        row: dict[str, Any] = {
            "link_id": _node_link_id("map", f"{rpg_id}-{code_id or 'missing'}"),
            "rpg_node_id": rpg_id,
            "source_link_id": _edge_endpoint_link_id(rpg_id, rpg_by_id, code_by_id),
            "status": mapping.get("status") or "mapped",
            "state": _node_state(mapping, "mapped"),
        }
        if code_id not in (None, ""):
            row["code_node_id"] = code_id
            row["dep_node_id"] = code_id
            row["target_link_id"] = _edge_endpoint_link_id(code_id, rpg_by_id, code_by_id)
        else:
            row["target_state"] = "missing_mapping"
        for key in ("source", "path", "reason"):
            _set_if_present(row, key, mapping.get(key))
        changed_refs = _changed_file_refs(_listify(mapping.get("changed_files")), diff_anchors)
        if changed_refs:
            row["changed_files"] = changed_refs
        mapping_rows.append(row)

    edge_rows: list[dict[str, Any]] = []
    for edge in edges:
        source_id = edge.get("source_node_id")
        target_id = edge.get("target_node_id")
        relation = edge.get("relation") or "dependency"
        row: dict[str, Any] = {
            "link_id": _node_link_id("edge", f"{source_id}-{relation}-{target_id}"),
            "source_node_id": source_id,
            "target_node_id": target_id,
            "source_link_id": _edge_endpoint_link_id(source_id, rpg_by_id, code_by_id),
            "target_link_id": _edge_endpoint_link_id(target_id, rpg_by_id, code_by_id),
            "relation": relation,
            "state": edge.get("status") or "visible",
        }
        for key in ("direction", "source", "source_graph", "edge_source", "relation_source", "path", "reason", "rpg_node_id", "neighbor_node_id", "name"):
            _set_if_present(row, key, edge.get(key))
        edge_rows.append(row)

    warning_rows = [_warning_link_fields(warning, rpg_by_id, code_by_id) for warning in warnings if isinstance(warning, dict)]
    focused_graph = _focused_graph_metadata(
        semantic_nodes,
        code_nodes,
        mapping_rows,
        edge_rows,
        hidden_counts,
        warning_rows,
        graph_context,
    )
    return {
        "summary": {
            "semantic_nodes": len(semantic_nodes),
            "code_nodes": len(code_nodes),
            "mappings": len(mapping_rows),
            "edges": len(edge_rows),
            "warnings": len(warning_rows),
            "changed_files": len(changed_files),
        },
        "semantic_nodes": semantic_nodes,
        "code_nodes": code_nodes,
        "mappings": mapping_rows,
        "edges": edge_rows,
        "hidden_counts": hidden_counts,
        "warnings": warning_rows,
        "changed_files": _changed_file_refs(changed_files, diff_anchors),
        "hierarchy": focused_graph["hierarchy"],
        "default_focus": focused_graph["default_focus"],
        "focused_graph": focused_graph,
        "graph_context": graph_context,
    }


def _rpg_node_row(node_id: str, node: dict[str, Any], paths: dict[str, str], changed_files: list[str]) -> dict[str, Any]:
    meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
    name = node.get("name") or node_id
    return {
        "node_id": node_id,
        "link_id": _node_link_id("rpg", node_id),
        "name": name,
        "symbol": name,
        "type": node.get("node_type") or node.get("type") or meta.get("type_name"),
        "node_type": node.get("node_type") or node.get("type") or meta.get("type_name"),
        "path": meta.get("path") or node.get("path"),
        "feature_path": paths.get(node_id, ""),
        "breadcrumb_path": paths.get(node_id, ""),
        "change": "semantic",
        "changed_files": changed_files,
        "mapping_status": "mapped",
        "state": "mapped",
    }


def _dep_node_row(dep_id: str, attrs: dict[str, Any], changed_files: list[str], mapped_rpg_ids: list[str]) -> dict[str, Any]:
    symbol = attrs.get("name") or dep_id.rsplit(":", 1)[-1]
    row: dict[str, Any] = {
        "node_id": dep_id,
        "dep_node_id": dep_id,
        "link_id": _node_link_id("code", dep_id),
        "path": _dep_node_path(dep_id, attrs),
        "name": symbol,
        "symbol": symbol,
        "type": attrs.get("type") or attrs.get("kind"),
        "change": "code",
        "changed_files": changed_files,
        "source": "dep_graph",
        "state": "mapped" if mapped_rpg_ids else "missing_mapping",
    }
    if attrs.get("signature") not in (None, ""):
        row["signature"] = attrs.get("signature")
    start = attrs.get("start_line") or attrs.get("lineno") or attrs.get("line")
    end = attrs.get("end_line") or start
    if start not in (None, ""):
        row["line_range"] = {"start": start, "end": end}
    if mapped_rpg_ids:
        row["mapped_rpg_node_ids"] = mapped_rpg_ids
    return row


def _edge_rows(dep_edges: Any, code_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(dep_edges, list):
        return rows
    for edge in dep_edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("src") or edge.get("source") or edge.get("from")
        target = edge.get("dst") or edge.get("target") or edge.get("to")
        if source not in code_ids and target not in code_ids:
            continue
        attrs = edge.get("attrs") if isinstance(edge.get("attrs"), dict) else {}
        rows.append({
            "source_node_id": source,
            "target_node_id": target,
            "relation": edge.get("relation") or edge.get("type") or attrs.get("type") or "dependency",
            "source_graph": "dep_graph",
            "edge_source": "dep_graph",
            "reason": "adjacent to changed code",
        })
    return rows[:50]


def _build_update_focus(result: dict, code_deltas: list[dict[str, Any]], semantic_total: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    changed_files = _changed_report_files(result, code_deltas)
    diff_anchors = _diff_anchor_map(code_deltas)
    rpg_data = _load_report_rpg(result)
    dep_graph = rpg_data.get("dep_graph") if isinstance(rpg_data.get("dep_graph"), dict) else {}
    dep_to_rpg = rpg_data.get("_dep_to_rpg_map") if isinstance(rpg_data.get("_dep_to_rpg_map"), dict) else {}
    rpg_nodes, rpg_paths = _flatten_rpg_tree(rpg_data.get("root"))
    matched_dep_ids: set[str] = set()
    code_nodes: list[dict[str, Any]] = []
    semantic_by_id: dict[str, dict[str, Any]] = {}
    mappings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if semantic_total == 0:
        warnings.append({
            "type": "zero_semantic_delta",
            "message": "RPG semantic delta 为 0",
            "reason": "diff_summary contains no added, deleted, modified, or renamed semantic files",
        })

    dep_items = _dep_node_items(dep_graph.get("nodes") if isinstance(dep_graph, dict) else {})
    for changed_file in changed_files:
        file_matched = False
        for dep_id, attrs in dep_items:
            if not _dep_node_matches_file(dep_id, attrs, changed_file):
                continue
            file_matched = True
            if dep_id in matched_dep_ids:
                continue
            mapped_rpg_ids = _ordered_unique_text(_listify(attrs.get("rpg_nodes")) + _listify(dep_to_rpg.get(dep_id)))
            code_nodes.append(_dep_node_row(dep_id, attrs, [changed_file], mapped_rpg_ids))
            matched_dep_ids.add(dep_id)
            for rpg_id in mapped_rpg_ids:
                node = rpg_nodes.get(rpg_id)
                if not isinstance(node, dict):
                    continue
                semantic_by_id.setdefault(rpg_id, _rpg_node_row(rpg_id, node, rpg_paths, [changed_file]))
                mappings.append({
                    "rpg_node_id": rpg_id,
                    "code_node_id": dep_id,
                    "dep_node_id": dep_id,
                    "status": "mapped",
                    "state": "mapped",
                    "source": "rpg_json",
                    "path": _dep_node_path(dep_id, attrs),
                    "changed_files": [changed_file],
                })
        if not file_matched:
            warnings.append({
                "type": "unmapped_changed_file",
                "message": "changed file has no dep_graph node in updated RPG JSON",
                "path": changed_file,
            })

    semantic_nodes = list(semantic_by_id.values())
    dep_edges = _edge_rows(dep_graph.get("edges") if isinstance(dep_graph, dict) else [], matched_dep_ids)
    unmatched_files = set(changed_files)
    for node in code_nodes:
        unmatched_files.difference_update(str(path) for path in _listify(node.get("changed_files")))
    unmatched_code_deltas = [row for row in code_deltas if (row.get("file") or row.get("path")) in unmatched_files]
    hidden_counts: dict[str, Any] = {}
    graph_context = {
        "rpg_nodes": result.get("node_count", result.get("rpg_nodes")),
        "rpg_edges": result.get("edge_count", result.get("rpg_edges")),
        "rpg_nodes_delta": result.get("nodes_delta"),
        "rpg_edges_delta": result.get("edges_delta"),
        "dep_nodes": result.get("dep_nodes"),
        "dep_edges": result.get("dep_edges"),
        "dep_nodes_delta": result.get("dep_nodes_delta"),
        "dep_edges_delta": result.get("dep_edges_delta"),
        "dep_to_rpg_map_size": result.get("dep_to_rpg_map_size"),
        "changed_files": len(changed_files),
        "semantic_delta": semantic_total,
    }
    graph_context = {key: value for key, value in graph_context.items() if value not in (None, "")}
    nodes_view = _build_nodes_view(
        semantic_nodes,
        code_nodes,
        mappings,
        dep_edges,
        hidden_counts,
        warnings,
        changed_files,
        diff_anchors,
        graph_context,
    )
    enriched_semantic_nodes = nodes_view["semantic_nodes"]
    enriched_code_nodes = nodes_view["code_nodes"]
    enriched_mappings = nodes_view["mappings"]
    enriched_edges = nodes_view["edges"]
    enriched_warnings = nodes_view["warnings"]
    summary = {
        "selected_feature_groups": len(enriched_semantic_nodes),
        "primary_rpg_nodes": len(enriched_semantic_nodes),
        "primary_code_nodes": len(enriched_code_nodes),
        "mapped_code_relations": len(enriched_mappings),
        "missing_mappings": len([node for node in enriched_code_nodes if not node.get("mapped_rpg_node_ids")]),
        "edges": len(enriched_edges),
        "warnings": len(enriched_warnings),
        "changed_files": len(changed_files),
        "semantic_delta": semantic_total,
    }
    focused_view = {
        "summary": summary,
        "nodes_view": nodes_view,
        "primary_rpg_nodes": enriched_semantic_nodes,
        "primary_code_nodes": enriched_code_nodes,
        "mappings": enriched_mappings,
        "edges": enriched_edges,
        "hidden_counts": hidden_counts,
        "warnings": enriched_warnings,
        "changed_files": nodes_view["changed_files"],
        "unmatched_code_deltas": unmatched_code_deltas,
    }
    return focused_view, enriched_semantic_nodes, enriched_code_nodes


def _hook_context(result: dict) -> dict[str, Any]:
    return {
        "CMIND_HOOK": result.get("CMIND_HOOK", os.environ.get("CMIND_HOOK", "")),
        "hook_calls_log": result.get("hook_calls_log") or str(HOOK_CALLS_LOG),
        "update_rpg_log": result.get("update_rpg_log") or str(HOOK_CALLS_LOG.parent / "update_rpg.log"),
    }


def _commit_range(result: dict) -> dict[str, Any]:
    return {
        "prev_ref": result.get("prev_ref"),
        "previous_commit": result.get("previous_commit"),
        "new_commit": result.get("new_commit"),
    }


def _commit_range_reason(commit_range: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in commit_range.items() if value not in (None, "")]
    return ", ".join(parts) if parts else "not recorded"


def _dep_delta_detail(result: dict) -> str:
    parts = []
    if result.get("dep_nodes_delta") not in (None, ""):
        parts.append(f"nodes_delta={result.get('dep_nodes_delta'):+d}" if isinstance(result.get("dep_nodes_delta"), int) else f"nodes_delta={result.get('dep_nodes_delta')}")
    if result.get("dep_edges_delta") not in (None, ""):
        parts.append(f"edges_delta={result.get('dep_edges_delta'):+d}" if isinstance(result.get("dep_edges_delta"), int) else f"edges_delta={result.get('dep_edges_delta')}")
    if result.get("dep_to_rpg_map_size") not in (None, ""):
        parts.append(f"dep_to_rpg_map_size={result.get('dep_to_rpg_map_size')}")
    return ", ".join(parts)


def _attach_update_report(result: dict) -> dict:
    try:
        semantic_summary = _diff_summary(result)
        semantic_total = sum(semantic_summary.values())
        code_deltas = _code_delta_rows(result)
        git_delta = result.get("git_delta")
        git_total = _change_count(git_delta) if git_delta is not None else _change_count(code_deltas)
        node_count = result.get("node_count", result.get("rpg_nodes", ""))
        rpg_path = result.get("output_path") or result.get("rpg_path")
        dep_summary = ""
        if result.get("dep_nodes") not in (None, ""):
            dep_summary = "nodes={}".format(
                _format_count_delta(
                    result.get("dep_nodes"),
                    result.get("dep_nodes_delta"),
                )
            )
            if result.get("dep_edges") not in (None, ""):
                dep_summary += ", edges={}".format(
                    _format_count_delta(
                        result.get("dep_edges"),
                        result.get("dep_edges_delta"),
                    )
                )
        semantic_detail = "RPG semantic delta 为 0" if semantic_total == 0 else _format_diff_summary(semantic_summary)
        dep_detail = _dep_delta_detail(result)
        focused_view, rpg_delta_rows, dep_delta_rows = _build_update_focus(result, code_deltas, semantic_total)
        hook_context = _hook_context(result)
        commit_range = _commit_range(result)
        viz_status = result.get("viz_error") or (
            "ok" if result.get("viz_path") else "not recorded"
        )
        status = result.get("status") or ("error" if result.get("error") else result.get("mode"))
        evidence = dict(result)
        evidence.update({
            "code_deltas": code_deltas,
            "semantic_summary": semantic_summary,
            "semantic_total": semantic_total,
            "focused_view_summary": focused_view.get("summary", {}),
            "commit_range": commit_range,
            "commit_range_reason": _commit_range_reason(commit_range),
            "hook_context": hook_context,
            "artifact_paths": [
                {"label": "rpg_json", "path": rpg_path},
                {"label": "rpg_html", "path": result.get("viz_path")},
                {"label": "hook_calls_log", "path": hook_context["hook_calls_log"]},
                {"label": "update_rpg_log", "path": hook_context["update_rpg_log"]},
            ],
        })
        report_run = CommandRun(
            command="update_rpg",
            title="CoderMind update_rpg Explain View",
            status=status,
            summary=[
                {"label": "mode", "value": result.get("mode", "")},
                {
                    "label": "reason",
                    "value": result.get("reason") or result.get("error", ""),
                },
                {"label": "git files", "value": git_total},
                {"label": "semantic files", "value": semantic_total, "detail": semantic_detail},
                {
                    "label": "RPG nodes",
                    "value": _format_count_delta(
                        node_count,
                        result.get("nodes_delta"),
                    ),
                    "detail": f"edges={_format_count_delta(result.get('edge_count'), result.get('edges_delta'))}" if result.get("edge_count") not in (None, "") else "",
                },
                {"label": "dep graph", "value": dep_summary, "detail": dep_detail},
                {
                    "label": "visualization",
                    "value": result.get("viz_path") or result.get("viz_error", ""),
                },
            ],
            steps=[
                StepEvent(
                    name="git delta",
                    status=result.get("mode", ""),
                    reason=(
                        f"{git_total} changed files"
                        if git_total != ""
                        else "not recorded"
                    ),
                ),
                StepEvent(
                    name="semantic delta",
                    status="warning" if semantic_total == 0 else result.get("mode", ""),
                    reason=semantic_detail,
                ),
                StepEvent(
                    name="commit range",
                    status=status,
                    reason=_commit_range_reason(commit_range),
                ),
                StepEvent(
                    name="hook context",
                    status="recorded",
                    reason=(
                        f"CMIND_HOOK={hook_context['CMIND_HOOK'] or 'not set'}, "
                        f"hook_calls_log={hook_context['hook_calls_log']}, "
                        f"update_rpg_log={hook_context['update_rpg_log']}"
                    ),
                ),
                StepEvent(
                    name="sync graph",
                    status=status,
                    reason=result.get("reason", ""),
                ),
                StepEvent(
                    name="visualize",
                    status=(
                        "ok"
                        if result.get("viz_path")
                        else "error"
                        if result.get("viz_error")
                        else "skipped"
                    ),
                    reason=result.get("viz_path") or result.get("viz_error", ""),
                ),
            ],
            artifacts=[
                ArtifactEvent(label="rpg_json", path=rpg_path),
                ArtifactEvent(label="rpg_html", path=result.get("viz_path")),
                ArtifactEvent(label="hook_calls_log", path=hook_context["hook_calls_log"]),
                ArtifactEvent(label="update_rpg_log", path=hook_context["update_rpg_log"]),
            ],
            rpg_deltas=[
                RPGDeltaEvent(
                    node_id=row.get("node_id"),
                    name=row.get("name"),
                    type=row.get("node_type") or row.get("type"),
                    path=row.get("path") or row.get("feature_path"),
                    change=row.get("change"),
                )
                for row in rpg_delta_rows
            ],
            dep_graph_deltas=[
                DepGraphDeltaEvent(
                    dep_node_id=row.get("dep_node_id") or row.get("node_id"),
                    path=row.get("path"),
                    source_feature=", ".join(_ordered_unique_text(_listify(row.get("mapped_rpg_node_ids")))),
                    change=row.get("change"),
                )
                for row in dep_delta_rows
            ],
            code_deltas=[
                CodeDeltaEvent(
                    file=row.get("file") or row.get("path"),
                    change_type=row.get("change_type"),
                    before=row.get("before"),
                    after=row.get("after"),
                    diff=row.get("diff"),
                )
                for row in code_deltas
            ],
            verification=[
                VerificationEvent(name="update_rpg", status=status, detail=result.get("reason") or result.get("error")),
                VerificationEvent(name="viz", status=viz_status, detail=result.get("viz_path") or result.get("viz_error")),
                VerificationEvent(name="semantic_delta", status="warning" if semantic_total == 0 else "recorded", detail=semantic_detail),
                VerificationEvent(name="commit_range", status="recorded", detail=_commit_range_reason(commit_range)),
            ],
            evidence=evidence,
        ).to_dict()
        report_run["focused_view"] = focused_view
        report_path = write_command_report(report_run)
        result["report_path"] = str(report_path)
    except Exception as exc:
        result["report_error"] = str(exc)
    return result


def _refresh_rpg_html(rpg_path: Path) -> dict:
    """Regenerate ``rpg.html`` next to ``rpg.json`` after a hook update.

    The encoder's ``run_encode.py`` already produces ``rpg.html`` via
    :mod:`rpg_visualize` during the initial full encode, but the
    pre-/post-commit hooks only re-write ``rpg.json``.  Without this
    refresh, the interactive visualisation drifts behind the graph
    until the next full encode.

    Best-effort: any failure (missing rpg.json, parse error, write
    permission) is swallowed so a slow / broken viz never blocks a
    commit.  The returned dict surfaces ``viz_path`` on success or
    ``viz_error`` on failure so callers can include it in the hook
    output for debugging.
    """
    result: dict = {}
    if not rpg_path.is_file():
        # Nothing to render — caller should already have surfaced
        # _RPG_MISSING_MSG, so stay quiet here.
        return result
    try:
        from rpg_visualize import load_rpg, generate_html  # noqa: WPS433

        data = load_rpg(str(rpg_path))
        html_content = generate_html(data)
        # rpg.html is a user-facing artefact: write it to the
        # workspace's .cmind/reports/ (the home-side data/ holds
        # only machine-consumed JSON).  This mirrors run_encode.py.
        RPG_HTML_FILE.parent.mkdir(parents=True, exist_ok=True)
        RPG_HTML_FILE.write_text(html_content, encoding="utf-8")
        result["viz_path"] = str(RPG_HTML_FILE)
    except Exception as exc:  # pragma: no cover — defensive
        result["viz_error"] = str(exc)
    return result


def update_dep_only(code_dir: str, workspace_root: str, dep_graph_path: Path,
                    rpg_path: Optional[Path] = None) -> dict:
    """Mode: dep — Rebuild dep_graph from AST and persist into rpg.json.

    In the embedded-dep_graph world the dep_graph lives inside
    ``rpg.json`` (see ``RPG.to_dict(include_dep_graph=True)``).  This
    mode therefore reads the current ``rpg.json``, swaps in the freshly
    rebuilt dep_graph, and writes ``rpg.json`` back out.  When
    ``rpg_path`` is ``None`` (or the file is missing) we fall back to
    the legacy standalone ``dep_graph.json`` write so that environments
    which haven't run the encoder yet still get a useful artefact —
    this is the path the very-first pre-commit hook hits on a fresh
    workspace before any RPG exists.

    ``dep_graph_path`` is preserved as a parameter for CLI back-compat
    but is now used only in the legacy fallback path.
    """
    from rpg.dep_graph import DependencyGraph

    t0 = time.time()
    dg = DependencyGraph(code_dir)
    dg.build()
    dg.parse()

    _rel = os.path.relpath(code_dir, workspace_root)
    code_dir_rel = "" if _rel == "." else _rel

    # Preferred path: dep_graph rides inside rpg.json (single source of truth).
    if rpg_path is not None and rpg_path.is_file():
        from rpg.service import RPGService
        svc = RPGService.load(str(rpg_path))
        svc.rpg.dep_graph = dg
        svc.rpg._dep_graph_code_dir = code_dir_rel
        svc.rpg._dep_to_rpg_map = svc.rpg._build_dep_to_rpg_map()
        svc.rpg.rebuild_cross_maps()
        # Drop the legacy external pointer so RPGService.load doesn't
        # override the embedded dep_graph on the next read.
        svc.rpg._dep_graph_file = None
        svc.save(str(rpg_path))

        return {
            "mode": "dep",
            "dep_nodes": len(dg.G.nodes()),
            "dep_edges": len(dg.G.edges()),
            "rpg_path": str(rpg_path),
            "duration": round(time.time() - t0, 3),
        }

    # Legacy fallback: write standalone dep_graph.json for environments
    # without an rpg.json yet (rare in practice — the pre-commit hook
    # exits early on workspaces that never ran the encoder).
    raw = dg.to_dict()
    raw["code_dir"] = code_dir_rel
    from datetime import datetime, timezone
    raw["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_rpg(str(dep_graph_path), raw, ensure_ascii=False, indent=2)

    return {
        "mode": "dep",
        "dep_nodes": len(dg.G.nodes()),
        "dep_edges": len(dg.G.edges()),
        "dep_graph_path": str(dep_graph_path),
        "duration": round(time.time() - t0, 3),
    }


def update_mapping(rpg_path: Path, code_dir: str, workspace_root: str,
                   dep_graph_path: Path) -> dict:
    """Mode: mapping — Rebuild dep_graph + dep↔rpg mappings, persist into rpg.json."""
    from rpg.service import RPGService

    t0 = time.time()
    svc = RPGService.load(str(rpg_path))
    # ``save_path=None``: dep_graph rides inside rpg.json (no standalone file)
    svc.refresh_dep_graph(code_dir, workspace_root=workspace_root)
    # Drop any stale external pointer left by older runs.
    svc.rpg._dep_graph_file = None
    svc.save(str(rpg_path))

    return {
        "mode": "mapping",
        "dep_nodes": len(svc.rpg.dep_graph.G.nodes()),
        "dep_edges": len(svc.rpg.dep_graph.G.edges()),
        "dep_to_rpg": len(svc.rpg._dep_to_rpg_map),
        "feature_to_dep": len(svc.rpg._feature_to_dep_map),
        "rpg_nodes": len(svc.rpg._node_index),
        "rpg_edges": len(svc.rpg.edges),
        "rpg_path": str(rpg_path),
        "duration": round(time.time() - t0, 3),
    }


def update_feature(rpg_path: Path, dep_graph_path: Path) -> dict:
    """Mode: feature — Load existing dep_graph, rebuild mappings + edges only.

    Reads dep_graph from rpg.json's embedded copy (the new contract); only
    falls back to the standalone ``dep_graph.json`` for legacy workspaces
    that haven't been re-encoded since the embed migration.
    """
    from rpg.service import RPGService
    from rpg.models import RPG

    t0 = time.time()
    svc = RPGService.load(str(rpg_path))

    # Prefer the embedded dep_graph that RPGService.load already
    # attached.  Only touch the standalone file when the embedded copy
    # is absent (legacy on-disk rpg.json from before the embed
    # migration).
    if svc.rpg.dep_graph is None:
        if not dep_graph_path.exists():
            return {
                "mode": "feature",
                "error": (
                    f"rpg.json has no embedded dep_graph and no standalone "
                    f"dep_graph.json found at {dep_graph_path}. "
                    "Run `cmind script update_graphs.py sync` to rebuild it."
                ),
            }
        # Legacy compat path
        dg = RPG.load_dep_graph(dep_graph_path)
        svc.rpg.dep_graph = dg

    # Rebuild mappings
    svc.rpg._dep_to_rpg_map = svc.rpg._build_dep_to_rpg_map()
    svc.rpg.rebuild_cross_maps()

    # Save RPG (edges will be merged from dep_graph via to_dict)
    svc.rpg._dep_graph_file = None
    svc.save(str(rpg_path))

    return {
        "mode": "feature",
        "dep_to_rpg": len(svc.rpg._dep_to_rpg_map),
        "feature_to_dep": len(svc.rpg._feature_to_dep_map),
        "rpg_edges": len(svc.rpg.edges),
        "rpg_path": str(rpg_path),
        "duration": round(time.time() - t0, 3),
    }


def update_full(rpg_path: Path, code_dir: str, workspace_root: str,
                dep_graph_path: Path) -> dict:
    """Mode: full — AST scan + mappings + edges, persist into rpg.json."""
    from rpg.service import RPGService

    t0 = time.time()
    svc = RPGService.load(str(rpg_path))

    # Rebuild dep_graph from code; ``save_path=None`` so dep_graph rides
    # inside rpg.json only.
    svc.refresh_dep_graph(code_dir, workspace_root=workspace_root)

    # Count dep_graph semantic edges that will merge into RPG edges
    dep_semantic_edges = [
        e for e in svc.rpg.get_dep_edges_for_rpg()
    ]

    svc.rpg._dep_graph_file = None
    svc.save(str(rpg_path))

    return {
        "mode": "full",
        "dep_nodes": len(svc.rpg.dep_graph.G.nodes()),
        "dep_edges": len(svc.rpg.dep_graph.G.edges()),
        "dep_to_rpg": len(svc.rpg._dep_to_rpg_map),
        "feature_to_dep": len(svc.rpg._feature_to_dep_map),
        "dep_semantic_edges_merged": len(dep_semantic_edges),
        "rpg_nodes": len(svc.rpg._node_index),
        "rpg_edges": len(svc.rpg.edges),
        "rpg_path": str(rpg_path),
        "duration": round(time.time() - t0, 3),
    }


def cmd_enrich(rpg_path: Path, code_dir: str, workspace_root: str,
               dep_graph_path: Path, files: list, align_only: bool,
               dry_run: bool) -> dict:
    """Subcommand: enrich — Align + fill feature graph from actual code."""
    from rpg.service import RPGService

    t0 = time.time()
    svc = RPGService.load(str(rpg_path))

    # Rebuild dep_graph first for accuracy (embedded only — single source).
    svc.refresh_dep_graph(code_dir, workspace_root=workspace_root)

    # Run enrichment (skip_dep_rebuild since refresh_dep_graph already did it)
    enrich_result = svc.enrich_from_code(
        code_dir,
        files=files or None,
        align_only=align_only,
        dry_run=dry_run,
        skip_dep_rebuild=True,
    )

    if not dry_run:
        svc.rpg._dep_graph_file = None
        svc.save(str(rpg_path))

    enrich_result.update({
        "mode": "enrich",
        "dry_run": dry_run,
        "rpg_path": str(rpg_path),
        "duration": round(time.time() - t0, 3),
    })
    return enrich_result


def cmd_sync(
    rpg_path: Path,
    code_dir: str,
    workspace_root: str,
    dep_graph_path: Path,
    *,
    staged_only: bool = False,
    force_full: bool = False,
    file_limit: int = 50,
) -> dict:
    """Subcommand: sync — commit-aware incremental refresh.

    Pre-commit hook path: pass ``staged_only=True`` so only ``git add``'d
    files contribute to the diff (working-tree-but-not-staged changes
    are out of scope for the imminent commit).

    Manual CLI path: omit ``staged_only`` (default ``False``) and the
    full working tree is considered dirty.

    Falls back to full rebuild whenever:

    * the workspace isn't a git repo, or
    * RPG has no ``meta.git`` baseline yet, or
    * history was rewritten (rebase / amend / reset / branch fork), or
    * the changed-file count exceeds ``file_limit`` (default 50).

    The "enrich" pass (path alignment + feature node fill) runs only
    when sync actually mutated the graph (``mode != "noop"``).
    """
    from rpg.service import RPGService

    t0 = time.time()

    # Fail-soft when the workspace hasn't run the encoder yet.  Without
    # this guard, ``RPGService.load`` raises ``FileNotFoundError`` which
    # the post-commit hook's ``|| true`` would swallow silently — making
    # the failure invisible during debugging.  Emit a structured error
    # instead so the hook log shows exactly what's wrong and how to fix
    # it.
    if not rpg_path.is_file():
        return _attach_update_report({
            "mode": "sync",
            "error": _RPG_MISSING_MSG.format(rpg_path=rpg_path),
            "rpg_path": str(rpg_path),
            "duration": round(time.time() - t0, 3),
        })

    svc = RPGService.load(str(rpg_path))

    # ``save_path=None``: dep_graph rides inside rpg.json (single source).
    # The caller's ``svc.save(rpg_path)`` below embeds it.
    sync_result = svc.sync_from_commit_diff(
        code_dir=code_dir,
        workspace_root=workspace_root,
        file_limit=file_limit,
        staged_only=staged_only,
        force_full=force_full,
    )

    # Run enrichment only when the graph actually changed.  ``noop``
    # means dep_graph is byte-identical to the previous state — there's
    # nothing for ``enrich_from_code`` to align.
    enrich_result: dict = {}
    if sync_result.get("mode") != "noop":
        enrich_result = svc.enrich_from_code(code_dir, skip_dep_rebuild=True)

    svc.rpg._dep_graph_file = None
    svc.save(str(rpg_path))

    # Keep ``rpg.html`` aligned with the freshly-saved ``rpg.json``.
    # The encoder produces both files during the initial full encode,
    # but earlier hook revisions only refreshed the JSON — leaving the
    # visualisation silently stale.  Best-effort: ``_refresh_rpg_html``
    # swallows its own errors so a broken viz can never block a commit.
    viz_result = _refresh_rpg_html(rpg_path)

    sync_out = {
        "mode": sync_result.get("mode", "sync"),
        "reason": sync_result.get("reason", ""),
        "last_commit": sync_result.get("last_commit"),
        "current_commit": sync_result.get("current_commit"),
        "meta_git_advanced_to": sync_result.get("meta_git_advanced_to"),
        "dep_nodes": sync_result.get("dep_nodes"),
        "dep_edges": sync_result.get("dep_edges"),
        # Incremental-only diagnostics (None for full / noop).
        "unchanged_hash": sync_result.get("unchanged_hash"),
        "modified": sync_result.get("modified"),
        "added": sync_result.get("added"),
        "deleted": sync_result.get("deleted"),
        "renamed": sync_result.get("renamed"),
        "edges_resemanticised": sync_result.get("edges_resemanticised"),
        # Enrich diagnostics (only populated when enrich ran).
        "aligned": enrich_result.get("aligned", 0),
        "groups_pathed": enrich_result.get("groups_pathed", 0),
        "l1_pathed": enrich_result.get("l1_pathed", 0),
        "filled": enrich_result.get("filled", 0),
        "groups_created": enrich_result.get("groups_created", 0),
        "rpg_nodes": len(svc.rpg._node_index),
        "rpg_path": str(rpg_path),
        "viz_path": viz_result.get("viz_path"),
        "viz_error": viz_result.get("viz_error"),
        "duration": round(time.time() - t0, 3),
    }
    _attach_update_report(sync_out)
    _log_hook_call("sync", sync_out)
    return sync_out


def cmd_update_rpg(
    rpg_path: Path,
    dep_graph_path: Path,
    workspace_root: str,
) -> dict:
    """Subcommand: update-rpg — full RPG update (dep_graph + feature graph).

    Creates a temporary git worktree at ``HEAD~1`` as the "previous version",
    runs ``run_update_rpg`` (LLM-driven feature tree diff + dep_graph rebuild),
    and cleans up the worktree.

    Designed for post-commit background invocation via ``setsid``::

        setsid env -u GIT_INDEX_FILE -u GIT_DIR sh -c \
            "cd <workspace>; cmind script update_graphs.py update-rpg --json >> log 2>&1" &

    Requires:
        - rpg.json exists (encode has been run)
        - git history has at least 2 commits (needs HEAD~1)
    """
    import shutil
    import subprocess
    import tempfile

    t0 = time.time()

    if not rpg_path.is_file():
        return _attach_update_report({
            "mode": "update-rpg",
            "error": _RPG_MISSING_MSG.format(rpg_path=rpg_path),
            "rpg_path": str(rpg_path),
        })

    # Check git has enough history
    try:
        prev_ref = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD~1"],
            cwd=workspace_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _attach_update_report({
            "mode": "update-rpg",
            "error": "Need at least 2 commits for incremental update (no HEAD~1)",
            "rpg_path": str(rpg_path),
        })

    # Prune orphaned worktrees from previous runs that were killed.
    subprocess.call(
        ["git", "worktree", "prune"],
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Create temporary worktree for previous commit.
    worktree_dir = tempfile.mkdtemp(prefix="cmind_prev_")
    try:
        wt_proc = subprocess.run(
            ["git", "worktree", "add", worktree_dir, prev_ref, "--detach", "-q"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )
        if wt_proc.returncode != 0:
            return _attach_update_report({
                "mode": "update-rpg",
                "error": f"git worktree add failed for {prev_ref}: {wt_proc.stderr.strip()}",
                "rpg_path": str(rpg_path),
                "prev_ref": prev_ref,
            })

        from common.git_utils import git_workspace_prefix
        from rpg_encoder.run_update_rpg import run_update_rpg

        git_prefix = git_workspace_prefix(workspace_root)
        last_repo_dir = (
            os.path.join(worktree_dir, git_prefix)
            if git_prefix
            else worktree_dir
        )
        git_delta = _git_delta_files(prev_ref, workspace_root)

        result = run_update_rpg(
            rpg_file=str(rpg_path),
            last_repo_dir=last_repo_dir,
            cur_repo_dir=workspace_root,
            dep_graph_path=str(dep_graph_path),
        )

        result["mode"] = "update-rpg"
        result["prev_ref"] = prev_ref
        result["git_delta"] = git_delta
        result["code_deltas"] = git_delta
        result["CMIND_HOOK"] = os.environ.get("CMIND_HOOK", "")
        result["hook_calls_log"] = str(HOOK_CALLS_LOG)
        result["update_rpg_log"] = str(HOOK_CALLS_LOG.parent / "update_rpg.log")

        # Refresh ``rpg.html`` whenever the JSON was actually rewritten.
        # ``run_update_rpg`` returns ``status="success"`` on a normal
        # write; skip the regen when it failed so we don't paper over
        # a broken graph with a stale-but-pretty HTML page.
        if result.get("status") == "success":
            viz_result = _refresh_rpg_html(rpg_path)
            if "viz_path" in viz_result:
                result["viz_path"] = viz_result["viz_path"]
            if "viz_error" in viz_result:
                result["viz_error"] = viz_result["viz_error"]

        result["duration"] = round(time.time() - t0, 3)
        _attach_update_report(result)
        _log_hook_call("update-rpg", result)
        return result

    finally:
        # Clean up worktree
        try:
            subprocess.call(
                ["git", "worktree", "remove", worktree_dir, "--force"],
                cwd=workspace_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        # Belt and suspenders: remove the directory if worktree remove failed
        if os.path.isdir(worktree_dir):
            shutil.rmtree(worktree_dir, ignore_errors=True)


def _auto_detect_code_dir(workspace_root: str, code_dir_arg: str = None) -> str:
    """Resolve the code directory to scan.

    Returns the workspace root by default — matching the encoder
    entry points (``run_encode.py`` / ``run_update_rpg.py``) which
    also default to ``WORKSPACE_ROOT``.  This keeps all 3 entry
    points consistent in encoder mode (``cmind init --here`` inside
    an existing repo).

    An explicit ``code_dir_arg`` always wins; pass it when scanning
    a non-default subdirectory (e.g. the decoder layout's ``repo/``).
    """
    if code_dir_arg:
        return os.path.abspath(code_dir_arg)
    return os.path.abspath(workspace_root)


def cmd_status(rpg_path: Path, dep_graph_path: Path) -> dict:
    """Subcommand: status — Read-only summary + AI-agent guidance.

    Designed for invocation from a Claude Code ``SessionStart`` hook or a
    VS Code ``runOn: folderOpen`` task (the Copilot analogue).  The stdout
    is consumed by the AI agent as context, so it should:

    1. Be cheap (no AST scan, no LLM calls).
    2. Confirm the RPG is available and how big it is.
    3. Remind the agent to prefer the ``rpg-tools`` MCP server over raw
       file scans when locating code.

    Returns a small dict for ``--json`` mode; emits plain text otherwise.
    """
    status = {
        "mode": "status",
        "rpg_path": str(rpg_path),
        "dep_graph_path": str(dep_graph_path),
        "rpg_exists": rpg_path.exists(),
        "legacy_dep_graph_exists": dep_graph_path.exists(),
        "dep_graph_exists": False,
        "dep_graph_source": "none",
    }

    def _count_graph(graph_data: dict) -> None:
        nodes = graph_data.get("nodes") or []
        edges = graph_data.get("edges") or []
        status["dep_nodes"] = len(nodes) if isinstance(nodes, (list, dict)) else 0
        status["dep_edges"] = len(edges) if isinstance(edges, (list, dict)) else 0

    if rpg_path.exists():
        try:
            # Use safe_load_rpg so a corrupted rpg.json doesn't crash
            # the cheap status command — it'll silently restore from
            # inner-git history when possible.
            rpg_data = safe_load_rpg(rpg_path)
            # RPG stores features in a hierarchical tree rooted at "root".
            # Walk it lazily to count nodes without loading the full
            # rpg.service module (the status command must stay cheap).
            def _walk(node):
                if not isinstance(node, dict):
                    return 0
                count = 1
                for child in node.get("children", []) or []:
                    count += _walk(child)
                return count

            root = rpg_data.get("root")
            status["rpg_nodes"] = _walk(root) if root else 0
            status["rpg_edges"] = len(rpg_data.get("edges", []) or [])
            status["repo_name"] = rpg_data.get("repo_name") or "unknown"
            repo_info = rpg_data.get("repo_info") or {}
            if not isinstance(repo_info, dict):
                repo_info = {}
            status["generated_at"] = repo_info.get("generated_at")
            # Extract meta.git (added Step 1 of commit-based sync plan).
            # Legacy RPGs without ``meta`` produce ``last_synced_*`` = None.
            meta = rpg_data.get("meta") or {}
            git_meta = meta.get("git") if isinstance(meta, dict) else None
            if isinstance(git_meta, dict) and git_meta.get("head_commit"):
                status["last_synced_commit"] = git_meta.get("head_commit")
                status["last_synced_short"] = git_meta.get("head_short")
                status["last_synced_branch"] = git_meta.get("head_branch")
                status["last_synced_at"] = git_meta.get("head_timestamp")
            embedded_dep = rpg_data.get("dep_graph")
            if isinstance(embedded_dep, dict) and (
                embedded_dep.get("nodes") or embedded_dep.get("edges")
            ):
                status["dep_graph_exists"] = True
                status["dep_graph_source"] = "embedded"
                _count_graph(embedded_dep)
        except (OSError, json.JSONDecodeError) as exc:
            status["rpg_error"] = str(exc)

    # Compare RPG's recorded git state against the current HEAD so the
    # agent / user can see when the RPG is stale.  Silent-fail in
    # non-git workspaces or when git is unavailable — that's the
    # "no info" case, not an error.
    try:
        from common.git_utils import read_head  # type: ignore
        current_head = read_head(os.getcwd())
    except Exception:
        current_head = None
    if current_head:
        status["current_commit"] = current_head.get("head_commit")
        status["current_short"] = current_head.get("head_short")
        status["current_branch"] = current_head.get("head_branch")
        last = status.get("last_synced_commit")
        if last and current_head.get("head_commit"):
            status["rpg_in_sync_with_head"] = last == current_head["head_commit"]

    if status["dep_graph_source"] == "none" and dep_graph_path.exists():
        try:
            with open(dep_graph_path, "r", encoding="utf-8") as f:
                dg_data = json.load(f)
            status["dep_graph_exists"] = True
            status["dep_graph_source"] = "legacy_file"
            _count_graph(dg_data)
            status["dep_generated_at"] = dg_data.get("generated_at")
        except (OSError, json.JSONDecodeError) as exc:
            status["dep_graph_error"] = str(exc)

    return status


def _format_status_for_agent(status: dict) -> str:
    """Render ``status`` as text guidance for the AI agent (stdout).

    For Claude Code ``SessionStart`` hooks, stdout is injected verbatim
    into the agent's context.  For VS Code tasks running on folderOpen,
    the user sees this text in a terminal; Copilot can read it on
    request.
    """
    lines = []
    rpg_broken = "rpg_error" in status
    rpg_available = status.get("rpg_exists") and not rpg_broken
    if rpg_available:
        nodes = status.get("rpg_nodes", "?")
        edges = status.get("rpg_edges", "?")
        repo = status.get("repo_name") or "unknown"
        lines.append(
            f"[CoderMind] Repository Program Graph is available "
            f"(repo={repo}, nodes={nodes}, edges={edges})."
        )
        if status.get("dep_graph_exists") and "dep_graph_error" not in status:
            dn = status.get("dep_nodes", "?")
            de = status.get("dep_edges", "?")
            lines.append(
                f"[CoderMind] Dependency graph: {dn} nodes, {de} edges."
            )
        elif "dep_graph_error" in status:
            lines.append(
                f"[CoderMind] Dependency graph unavailable (parse error: "
                f"{status['dep_graph_error']})."
            )

        # Git sync state — present only when both the RPG's recorded
        # ``meta.git`` and the workspace's current HEAD are readable.
        last_short = status.get("last_synced_short") or status.get("last_synced_commit")
        cur_short = status.get("current_short") or status.get("current_commit")
        last_branch = status.get("last_synced_branch")
        cur_branch = status.get("current_branch")
        in_sync = status.get("rpg_in_sync_with_head")

        # Build optional ``on branch X'' suffixes (kept off the message
        # when the branch is unknown so detached-HEAD output stays tidy).
        def _branch_suffix(branch):
            return f" on branch '{branch}'" if branch else ""

        if last_short and cur_short:
            if in_sync:
                lines.append(
                    f"[CoderMind] Last synced at commit {last_short}"
                    f"{_branch_suffix(cur_branch or last_branch)} "
                    "(in sync with current HEAD)."
                )
            else:
                # Diverged-branch hint: when the recorded branch and
                # current branch disagree, surface it — the user has
                # almost certainly switched branches and the figure
                # changes count comes mostly from that switch.
                branch_note = ""
                if last_branch and cur_branch and last_branch != cur_branch:
                    branch_note = (
                        f" (branch changed: '{last_branch}' → '{cur_branch}')"
                    )
                lines.append(
                    f"[CoderMind] Last synced at commit {last_short}"
                    f"{_branch_suffix(last_branch)}; "
                    f"current HEAD is {cur_short}"
                    f"{_branch_suffix(cur_branch)}{branch_note}. "
                    "Run /cmind.update_rpg "
                    "(or commit to trigger the pre-commit sync hook) to "
                    "refresh the graph."
                )
        elif last_short and not cur_short:
            lines.append(
                f"[CoderMind] Last synced at commit {last_short}"
                f"{_branch_suffix(last_branch)}; git status "
                "for the current workspace is unavailable."
            )
        lines.append("")
        lines.append(
            "When locating, navigating, or generating code in this "
            "workspace, prefer the rpg-tools MCP server over raw file "
            "scans:"
        )
        lines.append(
            "  - search_rpg(query, scope)        find a function, "
            "class, file, or feature by name/keyword."
        )
        lines.append(
            "  - explore_rpg(node_id, direction) walk callers/callees, "
            "inheritance, imports up to N hops."
        )
        lines.append(
            "  - get_node_detail(node_id)        full signature, "
            "call sites, optional source code."
        )
        lines.append(
            "  - list_rpg_tree(root_id)          browse the RPG "
            "feature tree (functional areas → groups → features)."
        )
        lines.append("")
        lines.append(
            "Fall back to Grep/Glob/Read only when the graph does not "
            "cover what you need. This saves tokens by avoiding a "
            "full-codebase scan."
        )
    elif rpg_broken:
        # File present but unreadable: tell the agent the graph is
        # NOT available so it doesn't waste a turn calling rpg-tools.
        lines.append(
            f"[CoderMind] RPG file at {status.get('rpg_path')} could not "
            f"be parsed (error: {status['rpg_error']}). Graph-powered "
            "navigation is unavailable until it is rebuilt. Run "
            "/cmind.encode to regenerate it."
        )
    else:
        lines.append(
            "[CoderMind] No RPG found at "
            f"{status.get('rpg_path')}. Run /cmind.encode to build the "
            "Repository Program Graph and enable graph-powered code "
            "navigation via the rpg-tools MCP server."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Update dep_graph and/or feature graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Common args helper
    def _add_common(p):
        p.add_argument("--rpg", type=Path, default=REPO_RPG_FILE,
                        help="Path to RPG file (repo_rpg.json)")
        p.add_argument("--dep-graph", type=Path, default=DEP_GRAPH_FILE,
                        help=(
                            "Legacy standalone dep_graph path used only "
                            "when rpg.json has no embedded dep_graph"
                        ))
        p.add_argument("--code-dir", type=str, default=None,
                        help="Code directory (default: auto-detect)")
        p.add_argument("--json", action="store_true", help="JSON output")

    # dep
    p_dep = sub.add_parser("dep", help="Rebuild dep_graph from AST into rpg.json")
    _add_common(p_dep)

    # enrich
    p_enrich = sub.add_parser("enrich", help="Enrich feature graph from actual code")
    _add_common(p_enrich)
    p_enrich.add_argument("--file", action="append", dest="files", default=[],
                          help="Only enrich specified file(s) (repeatable)")
    p_enrich.add_argument("--align-only", action="store_true",
                          help="Only align meta.path, don't add new nodes")
    p_enrich.add_argument("--dry-run", action="store_true",
                          help="Show what would change without saving")

    # sync
    p_sync = sub.add_parser(
        "sync",
        help="Commit-aware incremental dep_graph sync + enrich",
    )
    _add_common(p_sync)
    p_sync.add_argument(
        "--staged-only",
        action="store_true",
        help=(
            "Only consider ``git add``'d files (pre-commit hook scope). "
            "Without this flag, the entire working tree is considered."
        ),
    )
    p_sync.add_argument(
        "--force-full",
        action="store_true",
        help="Skip incremental decision tree; rebuild dep_graph from scratch.",
    )
    p_sync.add_argument(
        "--file-limit",
        type=int,
        default=50,
        help=(
            "Maximum changed files for incremental mode; above this we "
            "fall back to a full rebuild. Default: 50."
        ),
    )

    # status (read-only; for SessionStart hooks / folderOpen tasks)
    p_status = sub.add_parser(
        "status",
        help="Read-only RPG status + AI-agent MCP usage guidance",
    )
    _add_common(p_status)

    # update-rpg (full RPG update via LLM; background post-commit)
    p_update_rpg = sub.add_parser(
        "update-rpg",
        help="Full RPG update (dep_graph + feature graph via LLM). "
             "Creates a worktree for HEAD~1, runs process_diff, cleans up.",
    )
    _add_common(p_update_rpg)

    # Legacy: mapping, feature, full
    p_mapping = sub.add_parser("mapping", help="(legacy) dep + mappings")
    _add_common(p_mapping)
    p_feature = sub.add_parser("feature", help="(legacy) load dep_graph + mappings")
    _add_common(p_feature)
    p_full = sub.add_parser("full", help="(legacy) AST + mappings + edges")
    _add_common(p_full)

    # Backward compat: --mode flag
    parser.add_argument("--mode", choices=["dep", "mapping", "feature", "full"],
                        help="(deprecated) Use subcommands instead")
    parser.add_argument("--rpg", type=Path, default=REPO_RPG_FILE)
    parser.add_argument("--dep-graph", type=Path, default=DEP_GRAPH_FILE)
    parser.add_argument("--code-dir", type=str, default=None)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    # Resolve command from subcommand or --mode
    command = args.command or args.mode
    if not command:
        parser.print_help()
        sys.exit(1)

    workspace_root = os.getcwd()

    # For background hook processes (setsid) or any caller whose cwd is
    # not the workspace root, infer the workspace.  Earlier versions of
    # this fallback walked up from ``args.rpg`` assuming a layout of
    # ``<workspace>/.cmind/data/rpg.json``, which became wrong once the
    # default ``rpg.json`` moved into the home-side store
    # (``~/.cmind/workspaces/<id>/data/rpg.json``) in v0.1.3.  Use the
    # storage helper that already knows how to find the live workspace,
    # plus the ``CMIND_WORKSPACE`` env var as a final hint.
    if not os.path.isdir(os.path.join(workspace_root, ".cmind")):
        try:
            from cmind_cli._storage import find_workspace_root_from
        except Exception:
            find_workspace_root_from = None

        inferred: Path | None = None
        if find_workspace_root_from is not None:
            try:
                inferred = find_workspace_root_from(Path.cwd())
            except Exception:
                inferred = None
        if inferred is None:
            env = os.environ.get("CMIND_WORKSPACE")
            if env:
                cand = Path(env).expanduser()
                if cand.is_dir():
                    inferred = cand
        if inferred is not None and (inferred / ".cmind").is_dir():
            workspace_root = str(inferred)
            os.chdir(workspace_root)

    code_dir = _auto_detect_code_dir(workspace_root, args.code_dir)

    # Dispatch
    if command == "dep":
        # ``rpg_path`` is preferred (embedded dep_graph); falls back to
        # writing a legacy standalone dep_graph when the workspace has no
        # rpg.json yet (very first commit before /cmind.encode).
        result = update_dep_only(
            code_dir, workspace_root, args.dep_graph,
            rpg_path=args.rpg,
        )
    elif command == "mapping":
        result = update_mapping(args.rpg, code_dir, workspace_root, args.dep_graph)
    elif command == "feature":
        result = update_feature(args.rpg, args.dep_graph)
    elif command == "full":
        result = update_full(args.rpg, code_dir, workspace_root, args.dep_graph)
    elif command == "enrich":
        result = cmd_enrich(
            args.rpg, code_dir, workspace_root, args.dep_graph,
            files=getattr(args, "files", []),
            align_only=getattr(args, "align_only", False),
            dry_run=getattr(args, "dry_run", False),
        )
    elif command == "sync":
        result = cmd_sync(
            args.rpg,
            code_dir,
            workspace_root,
            args.dep_graph,
            staged_only=getattr(args, "staged_only", False),
            force_full=getattr(args, "force_full", False),
            file_limit=getattr(args, "file_limit", 50),
        )
    elif command == "status":
        result = cmd_status(args.rpg, args.dep_graph)
    elif command == "update-rpg":
        result = cmd_update_rpg(args.rpg, args.dep_graph, workspace_root)
    else:
        parser.print_help()
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    elif command == "status":
        # Plain-text mode: emit AI-agent guidance to stdout so Claude
        # SessionStart hooks and VS Code folderOpen tasks can surface it.
        print(_format_status_for_agent(result))
    else:
        print(f"Mode: {result['mode']}")
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        for k, v in result.items():
            if k not in ("mode",):
                print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
