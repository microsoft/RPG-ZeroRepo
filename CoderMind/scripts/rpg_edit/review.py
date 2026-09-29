#!/usr/bin/env python3
"""Impact-scoped review for rpg_edit — verify affected functionality via sub-agent.

Dispatches a sub-agent to verify that code changes made by rpg_edit
actually work correctly. The review scope is driven by impact analysis
data (callers, affected_files), NOT a full global review.

Usage:
    cmind script rpg_edit/review.py \
      --plan .cmind/data/rpg_edit_plan.json \
      --impact .cmind/data/rpg_edit_impact.json \
      --json

The sub-agent will:
  1. Run pytest on affected test files
  2. Run smoke_test for import/entry verification
  3. Start the application and verify affected functionality paths
  4. Fix any issues found and re-verify
"""

import argparse
import json
import logging
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Mapping, Optional, Tuple

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.diff_ranges import (  # noqa: E402
    changed_line_ranges_by_file,
    is_file_level_node,
    line_range_from_mapping,
    row_overlaps_changed_lines,
)
from common.paths import (  # noqa: E402
    REPO_DIR,
    REPORTS_DIR,
    REPO_RPG_FILE,
    cmd_for,
    RPG_EDIT_PLAN_FILE,
    RPG_EDIT_IMPACT_FILE,
    RPG_EDIT_VALIDATE_FILE,
    RPG_EDIT_LOCATE_FILE,
    RPG_EDIT_CODE_RESULT_FILE,
    RPG_EDIT_APPLY_RESULT_FILE,
    RPG_EDIT_REVIEW_RESULT_FILE,
)
from common.run_events import (  # noqa: E402
    ArtifactEvent,
    CodeDeltaEvent,
    CommandRun,
    DepGraphDeltaEvent,
    RPGDeltaEvent,
    RetrievalEvent,
    StepEvent,
    UserDecisionEvent,
    VerificationEvent,
)
from common.git_utils import file_diffs_between, read_head  # noqa: E402
from common.run_report import write_command_report  # noqa: E402

logger = logging.getLogger(__name__)


def _write_review_result(result: Dict[str, Any]) -> None:
    RPG_EDIT_REVIEW_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RPG_EDIT_REVIEW_RESULT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_json_artifact(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc), "_path": str(path)}


def _load_review_artifacts(plan_path: Path, impact_path: Optional[Path]) -> Dict[str, Any]:
    return {
        "validate": _load_json_artifact(RPG_EDIT_VALIDATE_FILE),
        "locate": _load_json_artifact(RPG_EDIT_LOCATE_FILE),
        "plan": _load_json_artifact(plan_path),
        "impact": _load_json_artifact(impact_path),
        "code_result": _load_json_artifact(RPG_EDIT_CODE_RESULT_FILE),
        "apply_result": _load_json_artifact(RPG_EDIT_APPLY_RESULT_FILE),
    }


_REPORT_SCOPES = {"final", "internal", "none"}


def _normalize_report_scope(report_scope: str) -> str:
    scope = str(report_scope or "final").strip().lower()
    if scope not in _REPORT_SCOPES:
        raise ValueError(f"report_scope must be one of {sorted(_REPORT_SCOPES)}, got {report_scope!r}")
    return scope


def _report_target_dir(report_scope: str, report_dir: Optional[Path]) -> Path:
    base_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    return base_dir / "internal" if report_scope == "internal" else base_dir


def _report_filename_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug or "command"


def _report_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _expected_report_path(report_dir: Path, report_timestamp: str) -> Path:
    return report_dir / f"cmind_run_rpg_edit_{_report_filename_slug(report_timestamp)}.html"


def _load_existing_review_result() -> Dict[str, Any]:
    result = _load_json_artifact(RPG_EDIT_REVIEW_RESULT_FILE)
    return result if isinstance(result, dict) else {}


def _existing_internal_report_paths(result: Dict[str, Any]) -> List[str]:
    paths = _listify(result.get("internal_report_paths"))
    if result.get("report_scope") == "internal" and result.get("report_path"):
        paths.extend(_listify(result.get("report_path")))
    return _ordered_unique(paths)


def _artifact_links(
    plan_path: Path,
    impact_path: Optional[Path],
    internal_report_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    paths = {
        "validate": RPG_EDIT_VALIDATE_FILE,
        "locate": RPG_EDIT_LOCATE_FILE,
        "plan": plan_path,
        "impact": impact_path,
        "code_result": RPG_EDIT_CODE_RESULT_FILE,
        "apply_result": RPG_EDIT_APPLY_RESULT_FILE,
        "review_result": RPG_EDIT_REVIEW_RESULT_FILE,
    }
    links: List[Dict[str, Any]] = []
    for label, path in paths.items():
        if path is None:
            continue
        status = "available" if path.exists() or label == "review_result" else "missing"
        links.append({"label": label, "path": str(path), "status": status})
    for index, path in enumerate(_ordered_unique(internal_report_paths or []), start=1):
        try:
            status = "available" if Path(path).exists() else "missing"
        except (OSError, ValueError):
            status = "missing"
        links.append({"label": f"internal_report_{index}", "path": path, "status": status})
    return links


def _impact_results(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    impact = artifacts.get("impact") if isinstance(artifacts.get("impact"), dict) else {}
    results = impact.get("results") if isinstance(impact.get("results"), dict) else {}
    return results


def _selected_candidate_rows(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else {}
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    affected = [node_id for node_id in plan.get("affected_nodes") or [] if node_id]
    candidates = [c for c in locate.get("results") or [] if isinstance(c, dict)]
    if not affected:
        return candidates

    impact_results = _impact_results(artifacts)
    candidates_by_id = {c.get("node_id"): c for c in candidates if c.get("node_id") in affected}
    rows: List[Dict[str, Any]] = []
    for node_id in affected:
        impact = impact_results.get(node_id) if isinstance(impact_results.get(node_id), dict) else {}
        located = candidates_by_id.get(node_id)
        candidate = dict(located or {"node_id": node_id})
        if located:
            candidate.setdefault("locate_state", "selected")
        else:
            candidate["locate_state"] = "missing"
            candidate["reason_state"] = "reconstructed_from_plan_impact"
        if impact:
            candidate.setdefault("name", impact.get("name", ""))
            if not candidate.get("dep_nodes"):
                candidate["dep_nodes"] = impact.get("dep_nodes") or []
                candidate["dep_nodes_source"] = "impact"
            if not candidate.get("status") and (impact.get("error") or impact.get("message")):
                candidate["status"] = impact.get("error") or impact.get("message")
        rows.append(candidate)
    return rows


def _listify(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _ordered_unique(values: List[Any]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _candidate_dep_nodes(candidate: Dict[str, Any], impact: Dict[str, Any]) -> List[str]:
    return _ordered_unique(_listify(candidate.get("dep_nodes")) + _listify(impact.get("dep_nodes")))


def _retrieval_hit_reason(candidate: Dict[str, Any], impact: Dict[str, Any]) -> str:
    parts: List[str] = []
    if candidate.get("locate_state") == "missing":
        parts.append("locate missing; reconstructed from plan/impact")
    elif candidate.get("score") not in (None, ""):
        parts.append(f"locate score={candidate.get('score')}")
    elif candidate.get("node_id"):
        parts.append("selected feature")
    if candidate.get("feature_path"):
        parts.append(f"feature path={candidate.get('feature_path')}")
    dep_count = len(_candidate_dep_nodes(candidate, impact))
    if dep_count:
        parts.append(f"{dep_count} mapped code relations ({dep_count} dep nodes)")
    else:
        parts.append("missing dep_graph mapping")
    if not impact:
        parts.append("missing impact result")
    elif impact.get("error"):
        parts.append(f"impact error={impact.get('error')}")
    elif impact.get("message"):
        parts.append(f"impact state={impact.get('message')}")
    summary = impact.get("impact_summary") if isinstance(impact.get("impact_summary"), dict) else {}
    callers = summary.get("total_callers", len(impact.get("callers") or []))
    callees = summary.get("total_callees", len(impact.get("callees") or []))
    inheritance = summary.get("total_inheritance", len(impact.get("inheritance") or []))
    files = summary.get("affected_file_count", len(impact.get("affected_files") or []))
    if callers or callees or inheritance or files:
        parts.append(f"impact callers={callers}, affected_files={files}")
        if callees or inheritance:
            parts.append(f"impact callees={callees}, inheritance={inheritance}")
    return "; ".join(parts) or "selected by review plan"


def _retrieval_rows(artifacts: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else {}
    impact_results = _impact_results(artifacts)
    if locate or candidates:
        hits: List[Dict[str, Any]] = []
        for candidate in candidates:
            node_id = candidate.get("node_id")
            impact = impact_results.get(node_id) if isinstance(impact_results.get(node_id), dict) else {}
            dep_nodes = _candidate_dep_nodes(candidate, impact)
            locate_state = candidate.get("locate_state") or ("selected" if node_id else "missing")
            impact_state = "available" if impact else "missing"
            if impact.get("error"):
                impact_state = "error"
            elif impact and not dep_nodes:
                impact_state = "missing_mapping"
            hits.append({
                "node_id": node_id,
                "name": candidate.get("name"),
                "path": candidate.get("path") or candidate.get("meta_path"),
                "score": candidate.get("score"),
                "locate_state": locate_state,
                "impact_state": impact_state,
                "mapping_state": "mapped" if dep_nodes else "missing_mapping",
                "mapped_code_relations": len(dep_nodes),
                "reason": _retrieval_hit_reason(candidate, impact),
            })
        rows.append({
            "query": locate.get("query", ""),
            "tool": str(RPG_EDIT_LOCATE_FILE),
            "reason": f"{len(hits)} selected feature groups from {len(locate.get('results') or [])} locate candidates",
            "hits": hits,
        })
    if impact_results:
        hits = []
        for node_id, impact in impact_results.items():
            impact = impact if isinstance(impact, dict) else {}
            dep_nodes = _candidate_dep_nodes({"node_id": node_id}, impact)
            hits.append({
                "node_id": node_id,
                "name": impact.get("name"),
                "impact_state": "available" if dep_nodes else "missing_mapping",
                "mapped_code_relations": len(dep_nodes),
                "reason": _retrieval_hit_reason({"node_id": node_id}, impact),
            })
        rows.append({
            "query": ", ".join(str(node_id) for node_id in impact_results),
            "tool": str(RPG_EDIT_IMPACT_FILE),
            "reason": f"{len(impact_results)} impact result sets with mapped code relations",
            "hits": hits,
        })
    return rows


def _code_delta_rows(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    files = [str(path) for path in _listify(code_result.get("files_modified"))]
    commit_sha = code_result.get("commit_sha")
    rows: List[Dict[str, Any]] = []
    if commit_sha:
        try:
            rows = file_diffs_between(
                REPO_DIR,
                to_commit=str(commit_sha),
                files=files or None,
                py_only=False,
            )
        except Exception as exc:
            rows = [{"file": path, "change_type": "modify", "diff": "", "error": str(exc)} for path in files]
    seen = {row.get("file") for row in rows}
    for path in files:
        if path not in seen:
            rows.append({"file": path, "change_type": "modify", "diff": ""})
    return rows


def _dep_node_path(dep_id: Any) -> str:
    if dep_id in (None, ""):
        return ""
    dep_id_text = str(dep_id)
    return dep_id_text.split(":", 1)[0] if ":" in dep_id_text else dep_id_text


def _modified_dep_ids(
    dep_ids: List[str],
    relation_by_dep: Mapping[str, Dict[str, Any]],
    current_dep_nodes: Mapping[str, Dict[str, Any]],
    changed_ranges: Mapping[str, List[Tuple[int, int]]],
    changed_files: List[str],
) -> List[str]:
    changed_file_set = set(changed_files)
    ranged_matches: List[str] = []
    fallback_matches: List[str] = []
    for dep_id in dep_ids:
        relation = relation_by_dep.get(dep_id, {})
        current = current_dep_nodes.get(dep_id, {})
        row = {**relation, **current}
        path = str(current.get("path") or relation.get("path") or _dep_node_path(dep_id))
        if path not in changed_file_set:
            continue
        if line_range_from_mapping(row):
            if row_overlaps_changed_lines(row, path, changed_ranges):
                ranged_matches.append(dep_id)
        elif is_file_level_node(dep_id, row, path):
            fallback_matches.append(dep_id)
    return _ordered_unique(ranged_matches or fallback_matches[:1])


def _mapped_code_relations(candidate: Dict[str, Any], impact: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_dep = {str(dep_id) for dep_id in _listify(candidate.get("dep_nodes"))}
    impact_dep = {str(dep_id) for dep_id in _listify(impact.get("dep_nodes"))}
    candidate_dep_source = candidate.get("dep_nodes_source")
    rows: List[Dict[str, Any]] = []
    for dep_id in _candidate_dep_nodes(candidate, impact):
        sources: List[str] = []
        if dep_id in candidate_dep and candidate_dep_source != "impact":
            sources.append("locate")
        if dep_id in impact_dep or (dep_id in candidate_dep and candidate_dep_source == "impact"):
            sources.append("impact")
        rows.append({
            "node_id": dep_id,
            "dep_node_id": dep_id,
            "source_feature": candidate.get("node_id"),
            "path": _dep_node_path(dep_id) or candidate.get("meta_path") or candidate.get("path"),
            "relation": "feature_to_dep",
            "source": "+".join(sources) or "selected_feature",
            "status": "mapped",
        })
    return rows


def _dep_node_rows(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(_mapped_code_relations(candidate, {}))
    return rows


def _code_delta_file(delta: Dict[str, Any]) -> str:
    return str(delta.get("file") or delta.get("path") or "")


def _diff_anchor_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "change"


def _diff_anchor_map(code_deltas: List[Dict[str, Any]]) -> Dict[str, str]:
    anchors: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for index, delta in enumerate(code_deltas, start=1):
        file_path = _code_delta_file(delta) or f"change-{index}"
        base = _diff_anchor_slug(f"diff-{file_path}")
        count = used.get(base, 0) + 1
        used[base] = count
        anchor = base if count == 1 else f"{base}-{count}"
        if file_path not in anchors:
            anchors[file_path] = anchor
    return anchors


def _changed_file_refs(files: List[Any], anchors: Dict[str, str]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for file_path in _ordered_unique(files):
        row = {"path": file_path}
        _set_if_present(row, "diff_anchor", anchors.get(file_path))
        refs.append(row)
    return refs


def _node_state(row: Dict[str, Any], default: str = "available") -> str:
    status = row.get("state") or row.get("status") or row.get("mapping_status")
    if status in (None, ""):
        return default
    if status == "missing":
        return "missing_mapping"
    return str(status)


def _node_link_id(kind: str, node_id: Any) -> str:
    return _slug_id(kind, node_id)


def _edge_endpoint_link_id(node_id: Any, rpg_nodes: Dict[str, Dict[str, Any]], code_nodes: Dict[str, Dict[str, Any]]) -> str:
    node_text = str(node_id or "")
    if node_text in rpg_nodes:
        return str(rpg_nodes[node_text].get("link_id") or _node_link_id("rpg", node_text))
    if node_text in code_nodes:
        return str(code_nodes[node_text].get("link_id") or _node_link_id("code", node_text))
    return _node_link_id("context", node_text)


def _warning_link_fields(warning: Dict[str, Any], rpg_nodes: Dict[str, Dict[str, Any]], code_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(warning)
    node_id = warning.get("node_id") or warning.get("rpg_node_id")
    dep_id = warning.get("dep_node_id") or warning.get("code_node_id")
    if node_id not in (None, "") and str(node_id) in rpg_nodes:
        row["node_link_id"] = _edge_endpoint_link_id(node_id, rpg_nodes, code_nodes)
    if dep_id not in (None, "") and str(dep_id) in code_nodes:
        row["code_link_id"] = _edge_endpoint_link_id(dep_id, rpg_nodes, code_nodes)
    return row


def _hierarchy_segments(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    text = str(value)
    separator = " / " if " / " in text else "/"
    return [part.strip() for part in text.split(separator) if part.strip()]


def _hierarchy_child(parent: Dict[str, Any], child_id: str, name: str, kind: str) -> Dict[str, Any]:
    children = parent.setdefault("children", [])
    for child in children:
        if isinstance(child, dict) and child.get("id") == child_id:
            return child
    child = {"id": child_id, "name": name, "kind": kind, "children": []}
    children.append(child)
    return child


def _append_hierarchy_leaf(parent: Dict[str, Any], leaf: Dict[str, Any]) -> None:
    children = parent.setdefault("children", [])
    leaf_id = leaf.get("id")
    if any(isinstance(child, dict) and child.get("id") == leaf_id for child in children):
        return
    children.append(leaf)


def _add_hierarchy_path(root: Dict[str, Any], parts: List[str], leaf: Dict[str, Any], group_kind: str) -> None:
    parent = root
    trail: List[str] = []
    for part in parts:
        trail.append(part)
        parent = _hierarchy_child(parent, _node_link_id(group_kind, "/".join(trail)), part, group_kind)
    _append_hierarchy_leaf(parent, leaf)


def _semantic_hierarchy_parts(node: Dict[str, Any]) -> List[str]:
    for key in ("breadcrumb_path", "feature_path"):
        parts = _hierarchy_segments(node.get(key))
        if parts:
            return parts[:-1] if len(parts) > 1 else []
    return []


def _code_hierarchy_parts(node: Dict[str, Any]) -> List[str]:
    for key in ("path", "module", "file"):
        parts = _hierarchy_segments(node.get(key))
        if parts:
            return parts[:-1]
    return []


def _focused_graph_hierarchy(
    semantic_nodes: List[Dict[str, Any]],
    code_nodes: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    hidden_counts: Dict[str, Any],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    root: Dict[str, Any] = {
        "id": "focused-graph-root",
        "name": "Focused graph",
        "kind": "root",
        "feature_name": "Focused graph",
        "feature_path": "Focused graph",
        "meta": {"hidden_counts": hidden_counts, "warnings": len(warnings), "edges": len(edges)},
        "children": [],
    }
    code_by_id = {
        str(node.get("node_id") or node.get("dep_node_id")): node
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }
    semantic_link_by_id = {
        str(node.get("node_id")): str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        for node in semantic_nodes
        if node.get("node_id") not in (None, "")
    }
    code_link_by_id = {
        code_id: str(node.get("link_id") or _node_link_id("code", code_id))
        for code_id, node in code_by_id.items()
    }
    mapped_ids_by_rpg: Dict[str, List[str]] = {}
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        if rpg_id not in (None, "") and code_id not in (None, ""):
            mapped_ids_by_rpg.setdefault(str(rpg_id), []).append(str(code_id))

    def code_refs_for(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        rpg_id = str(node.get("node_id") or "")
        refs: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def append_ref(ref: Dict[str, Any]) -> None:
            key = str(ref.get("node_id") or ref.get("link_id") or ref.get("path") or ref.get("symbol") or "")
            if not key or key in seen:
                return
            seen.add(key)
            refs.append(ref)

        for item in _listify(node.get("mapped_code")):
            if not isinstance(item, dict):
                continue
            code_id = item.get("node_id") or item.get("dep_node_id")
            code_id_text = str(code_id or "")
            code = code_by_id.get(code_id_text, {})
            path = item.get("path") or code.get("path") or item.get("file") or code.get("file") or code.get("module") or _dep_node_path(code_id_text)
            symbol = item.get("symbol") or item.get("name") or code.get("symbol") or code.get("name") or _symbol_from_dep(code_id_text, code)
            row: Dict[str, Any] = {
                "node_id": code_id_text,
                "link_id": item.get("link_id") or code.get("link_id") or _node_link_id("code", code_id_text),
                "path": path,
                "symbol": symbol,
            }
            for key in ("type", "kind", "line_range", "state", "source"):
                _set_if_present(row, key, item.get(key) or code.get(key))
            append_ref(row)

        code_ids = _ordered_unique(_listify(node.get("mapped_code_node_ids")) + mapped_ids_by_rpg.get(rpg_id, []))
        for code_id in code_ids:
            code = code_by_id.get(code_id, {})
            path = code.get("path") or code.get("file") or code.get("module") or _dep_node_path(code_id)
            symbol = code.get("symbol") or code.get("name") or _symbol_from_dep(code_id, code)
            row = {
                "node_id": code_id,
                "link_id": code.get("link_id") or _node_link_id("code", code_id),
                "path": path,
                "symbol": symbol,
            }
            for key in ("type", "kind", "line_range", "state", "source"):
                _set_if_present(row, key, code.get(key))
            append_ref(row)
        return refs

    def merge_code_metadata(row: Dict[str, Any], refs: List[Dict[str, Any]]) -> None:
        if not refs:
            return
        existing = [item for item in _listify(row.get("mapped_code")) if isinstance(item, dict)]
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for ref in existing + refs:
            key = str(ref.get("node_id") or ref.get("link_id") or ref.get("path") or ref.get("symbol") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(ref)
        row["mapped_code"] = merged
        row["mapped_code_node_ids"] = _ordered_unique([ref.get("node_id") for ref in merged])
        row["mapped_code_link_ids"] = _ordered_unique([ref.get("link_id") for ref in merged])
        row["mapped_code_paths"] = _ordered_unique([ref.get("path") for ref in merged])
        row["mapped_code_symbols"] = _ordered_unique([ref.get("symbol") for ref in merged])
        if row["mapped_code_paths"]:
            row["mapped_code_path"] = row["mapped_code_paths"][0]
        if row["mapped_code_symbols"]:
            row["mapped_code_symbol"] = row["mapped_code_symbols"][0]
        row["mapped_code_count"] = len(merged)

    def feature_path_text(node: Dict[str, Any], group_parts: List[str], feature_name: str) -> str:
        for key in ("breadcrumb_path", "feature_path"):
            value = node.get(key)
            parts = _hierarchy_segments(value)
            if parts:
                return " / ".join(parts)
        return " / ".join(group_parts + ([feature_name] if feature_name else []))

    def semantic_tree_kind(node: Dict[str, Any]) -> str:
        node_type = str(node.get("node_type") or node.get("type") or "").lower()
        if node_type in {"feature_group", "category", "functional_area", "root"}:
            return node_type
        return "feature"

    def semantic_full_path_parts(node: Dict[str, Any], group_parts: List[str], feature_name: str) -> List[str]:
        return _hierarchy_segments(feature_path_text(node, group_parts, feature_name))

    semantic_node_by_path: Dict[str, Dict[str, Any]] = {}
    for semantic_node in semantic_nodes:
        semantic_name = str(semantic_node.get("name") or semantic_node.get("symbol") or semantic_node.get("node_id") or "")
        semantic_parts = semantic_full_path_parts(semantic_node, _semantic_hierarchy_parts(semantic_node), semantic_name)
        if semantic_parts:
            semantic_node_by_path.setdefault(" / ".join(semantic_parts), semantic_node)

    def merge_semantic_metadata(row: Dict[str, Any], node: Dict[str, Any], feature_name: str, feature_path: str) -> None:
        node_id = str(node.get("node_id") or "")
        link_id = str(node.get("link_id") or _node_link_id("rpg", node_id))
        row["name"] = feature_name
        row["feature_name"] = feature_name
        row["feature_path"] = feature_path
        row["kind"] = semantic_tree_kind(node)
        if node_id:
            row["node_id"] = node_id
        row["aliases"] = _ordered_unique(_listify(row.get("aliases")) + [node_id, link_id, _node_link_id("feature-path", feature_path)])
        _set_if_present(row, "state", node.get("state"))
        _set_if_present(row, "mapping_status", node.get("mapping_status"))
        for key in (
            "type",
            "node_type",
            "path",
            "breadcrumb",
            "breadcrumb_path",
            "locate_status",
            "score",
            "reason",
            "apply_action",
            "changed_files",
            "hidden_counts",
            "warning_types",
            "source",
        ):
            _set_if_present(row, key, node.get(key))

    def append_or_merge_hierarchy_leaf(parent: Dict[str, Any], leaf: Dict[str, Any]) -> None:
        children = parent.setdefault("children", [])
        leaf_id = leaf.get("id")
        for child in children:
            if not isinstance(child, dict) or child.get("id") != leaf_id:
                continue
            incoming_children = [item for item in _listify(leaf.get("children")) if isinstance(item, dict)]
            for key, value in leaf.items():
                if key == "children" or value in (None, ""):
                    continue
                child[key] = value
            for grandchild in incoming_children:
                _append_hierarchy_leaf(child, grandchild)
            return
        children.append(leaf)

    attached_endpoint_ids: set[str] = set()

    def make_code_leaf(ref: Dict[str, Any]) -> Dict[str, Any]:
        code_id = str(ref.get("node_id") or ref.get("dep_node_id") or "")
        code = code_by_id.get(code_id, {})
        path = ref.get("path") or code.get("path") or code.get("file") or code.get("module") or _dep_node_path(code_id)
        symbol = ref.get("symbol") or ref.get("name") or code.get("symbol") or code.get("name") or _symbol_from_dep(code_id, code)
        leaf: Dict[str, Any] = {
            "id": str(ref.get("link_id") or code.get("link_id") or _node_link_id("code", code_id)),
            "node_id": code_id,
            "dep_node_id": code_id,
            "name": symbol or path or code_id,
            "symbol": symbol,
            "path": path,
            "kind": "code",
            "state": ref.get("state") or code.get("state") or _node_state(code, "mapped"),
            "aliases": _ordered_unique([code_id, ref.get("link_id"), code.get("link_id")]),
        }
        for key in ("type", "line_range", "source", "changed", "changed_files", "diff_anchor"):
            _set_if_present(leaf, key, ref.get(key) or code.get(key))
        return leaf

    def endpoint_leaf(edge: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
        node_id = edge.get(f"{side}_node_id")
        node_text = str(node_id or "")
        link_id = edge.get(f"{side}_link_id") or semantic_link_by_id.get(node_text) or code_link_by_id.get(node_text)
        if link_id in (None, "") and node_text:
            link_id = _node_link_id("context", node_text)
        link_text = str(link_id or "")
        if not link_text or link_text in set(semantic_link_by_id.values()):
            return None
        if node_text in code_by_id and link_text == code_link_by_id.get(node_text):
            return None
        leaf: Dict[str, Any] = {
            "id": link_text,
            "node_id": node_text or link_text,
            "name": edge.get(f"{side}_name") or edge.get("name") or edge.get(f"{side}_path") or edge.get("path") or node_text or "context",
            "kind": "context",
            "state": edge.get("state") or "context",
            "relation": edge.get("relation"),
            "direction": edge.get("direction"),
            "source": edge.get("source") or edge.get("source_graph") or edge.get("edge_source"),
            "aliases": _ordered_unique([node_text, link_text]),
        }
        path = edge.get(f"{side}_path") or edge.get("path")
        _set_if_present(leaf, "path", path)
        for key in ("reason", "source_graph", "edge_source", "relation_source"):
            _set_if_present(leaf, key, edge.get(key))
        return leaf

    def append_endpoint(parent: Dict[str, Any], leaf: Dict[str, Any]) -> None:
        leaf_id = str(leaf.get("id") or "")
        if not leaf_id or leaf_id in attached_endpoint_ids:
            return
        attached_endpoint_ids.add(leaf_id)
        _append_hierarchy_leaf(parent, leaf)

    def relation_endpoint_refs(node_id: str, link_id: str, code_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        feature_tokens = set(_ordered_unique([node_id, link_id]))
        for ref in code_refs:
            feature_tokens.update(_ordered_unique([ref.get("node_id"), ref.get("link_id")]))
        refs: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for edge in edges:
            edge_tokens = _ordered_unique([
                edge.get("rpg_node_id"),
                edge.get("source_node_id"),
                edge.get("target_node_id"),
                edge.get("source_link_id"),
                edge.get("target_link_id"),
            ])
            if not feature_tokens.intersection(edge_tokens):
                continue
            for side in ("source", "target"):
                leaf = endpoint_leaf(edge, side)
                if not leaf:
                    continue
                leaf_id = str(leaf.get("id") or "")
                if not leaf_id or leaf_id in feature_tokens or leaf_id in seen:
                    continue
                seen.add(leaf_id)
                refs.append(leaf)
        return refs

    def endpoint_group(parent: Dict[str, Any], owner_id: str, name: str, kind: str) -> Dict[str, Any]:
        return _hierarchy_child(parent, _node_link_id(kind, owner_id), name, kind)

    for node in semantic_nodes:
        node_id = str(node.get("node_id") or "")
        link_id = str(node.get("link_id") or _node_link_id("rpg", node_id))
        feature_name = str(node.get("name") or node.get("symbol") or node_id or "feature")
        group_parts = _semantic_hierarchy_parts(node)
        code_refs = code_refs_for(node)
        parent = root
        trail: List[str] = []
        for part in group_parts:
            trail.append(part)
            group_path = " / ".join(trail)
            group_node = semantic_node_by_path.get(group_path)
            if group_node:
                group_node_id = str(group_node.get("node_id") or "")
                group = _hierarchy_child(parent, str(group_node.get("link_id") or _node_link_id("rpg", group_node_id)), part, semantic_tree_kind(group_node))
                merge_semantic_metadata(group, group_node, part, group_path)
            else:
                group = _hierarchy_child(parent, _node_link_id("feature-path", "/".join(trail)), part, "feature_group")
                group["feature_name"] = part
                group["feature_path"] = group_path
            merge_code_metadata(group, code_refs)
            parent = group
        leaf_feature_path = feature_path_text(node, group_parts, feature_name)
        leaf: Dict[str, Any] = {
            "id": link_id,
            "node_id": node_id,
            "name": feature_name,
            "feature_name": feature_name,
            "feature_path": leaf_feature_path,
            "kind": semantic_tree_kind(node),
            "state": node.get("state"),
            "mapping_status": node.get("mapping_status"),
            "aliases": _ordered_unique([node_id, link_id, _node_link_id("feature-path", leaf_feature_path)]),
        }
        for key in (
            "type",
            "node_type",
            "path",
            "breadcrumb",
            "breadcrumb_path",
            "locate_status",
            "score",
            "reason",
            "apply_action",
            "changed_files",
            "hidden_counts",
            "warning_types",
            "source",
        ):
            _set_if_present(leaf, key, node.get(key))
        merge_code_metadata(leaf, code_refs)
        context_group: Optional[Dict[str, Any]] = None
        for ref in relation_endpoint_refs(node_id, link_id, code_refs):
            if context_group is None:
                context_group = endpoint_group(leaf, link_id, "Relation endpoints", "context_group")
            append_endpoint(context_group, ref)
        append_or_merge_hierarchy_leaf(parent, leaf)

    root_context_group: Optional[Dict[str, Any]] = None
    for edge in edges:
        for side in ("source", "target"):
            leaf = endpoint_leaf(edge, side)
            if not leaf or str(leaf.get("id") or "") in attached_endpoint_ids:
                continue
            if root_context_group is None:
                root_context_group = endpoint_group(root, "unassigned", "Additional relation endpoints", "context_group")
            append_endpoint(root_context_group, leaf)
    return root


def _focused_graph_default_focus(
    semantic_nodes: List[Dict[str, Any]],
    code_nodes: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    semantic_links = [str(node.get("link_id") or _node_link_id("rpg", node.get("node_id"))) for node in semantic_nodes]
    focused_semantic_links = [
        str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        for node in semantic_nodes
        if node.get("changed")
        or node.get("changed_files")
        or node.get("diff_anchor")
        or node.get("change")
        or node.get("apply_action")
    ]
    code_links = [str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id"))) for node in code_nodes]
    semantic_link_set = set(semantic_links)
    code_to_feature_links: Dict[str, List[str]] = {}
    hierarchy_paths_by_link: Dict[str, List[str]] = {}

    def remember_code_feature(code_link: Any, feature_link: Any) -> None:
        if code_link in (None, "") or feature_link in (None, ""):
            return
        code_text = str(code_link)
        feature_text = str(feature_link)
        values = code_to_feature_links.setdefault(code_text, [])
        if feature_text not in values:
            values.append(feature_text)

    def remember_path(alias: Any, path_ids: List[str]) -> None:
        if alias in (None, "") or not path_ids:
            return
        alias_text = str(alias)
        if alias_text not in hierarchy_paths_by_link:
            hierarchy_paths_by_link[alias_text] = path_ids

    rpg_link_by_node_id = {
        str(node.get("node_id")): str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        for node in semantic_nodes
        if node.get("node_id") not in (None, "")
    }
    semantic_link_by_path: Dict[str, str] = {}
    for node in semantic_nodes:
        link_id = str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        feature_name = str(node.get("name") or node.get("symbol") or node.get("node_id") or "")
        full_parts: List[str] = []
        for key in ("breadcrumb_path", "feature_path"):
            full_parts = _hierarchy_segments(node.get(key))
            if full_parts:
                break
        if not full_parts:
            full_parts = _semantic_hierarchy_parts(node) + ([feature_name] if feature_name else [])
        if full_parts:
            semantic_link_by_path.setdefault(" / ".join(full_parts), link_id)

    semantic_path_ids_by_link: Dict[str, List[str]] = {}
    for node in semantic_nodes:
        link_id = str(node.get("link_id") or _node_link_id("rpg", node.get("node_id")))
        trail: List[str] = []
        path_ids: List[str] = ["focused-graph-root"]
        for part in _semantic_hierarchy_parts(node):
            trail.append(part)
            group_path = " / ".join(trail)
            path_ids.append(semantic_link_by_path.get(group_path) or _node_link_id("feature-path", "/".join(trail)))
        path_ids.append(link_id)
        semantic_path_ids_by_link[link_id] = path_ids
        remember_path(link_id, path_ids)
        remember_path(node.get("node_id"), path_ids)

    code_link_by_node_id = {
        str(node.get("node_id") or node.get("dep_node_id")): str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id")))
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }

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
            for feature_link in feature_links:
                base_path = semantic_path_ids_by_link.get(str(feature_link))
                if not base_path:
                    continue
                code_path = base_path + [_node_link_id("code_group", feature_link), code_link]
                remember_path(code_link, code_path)
                remember_path(code_id, code_path)
        else:
            code_path = ["focused-graph-root", _node_link_id("code_group", "unassigned"), code_link]
            remember_path(code_link, code_path)
            remember_path(code_id, code_path)

    endpoint_link_ids: List[Any] = []
    context_link_ids: List[Any] = []

    def endpoint_link(edge: Dict[str, Any], side: str) -> str:
        node_id = edge.get(f"{side}_node_id")
        node_text = str(node_id or "")
        return str(
            edge.get(f"{side}_link_id")
            or rpg_link_by_node_id.get(node_text)
            or code_link_by_node_id.get(node_text)
            or _node_link_id("context", node_text)
        )

    def owner_feature_links(edge: Dict[str, Any]) -> List[str]:
        owners: List[Any] = []
        owners.append(rpg_link_by_node_id.get(str(edge.get("rpg_node_id") or "")))
        for side in ("source", "target"):
            node_id = edge.get(f"{side}_node_id")
            link_id = endpoint_link(edge, side)
            if str(node_id or "") in rpg_link_by_node_id:
                owners.append(rpg_link_by_node_id[str(node_id)])
            owners.extend(code_to_feature_links.get(str(node_id or ""), []))
            owners.extend(code_to_feature_links.get(link_id, []))
        return _ordered_unique(owners)

    for edge in edges:
        owners = owner_feature_links(edge)
        for side in ("source", "target"):
            link_id = endpoint_link(edge, side)
            node_id = edge.get(f"{side}_node_id")
            endpoint_link_ids.append(link_id)
            endpoint_link_ids.append(node_id)
            if link_id in semantic_link_set:
                continue
            if link_id in code_links:
                continue
            context_link_ids.append(link_id)
            context_path_base = None
            for feature_link in owners:
                base_path = semantic_path_ids_by_link.get(str(feature_link))
                if base_path:
                    context_path_base = base_path + [_node_link_id("context_group", feature_link)]
                    break
            if context_path_base is None:
                context_path_base = ["focused-graph-root", _node_link_id("context_group", "unassigned")]
            context_path = context_path_base + [link_id]
            remember_path(link_id, context_path)
            remember_path(node_id, context_path)

    changed_feature_links: List[Any] = []
    changed_code_links: List[str] = []
    for node in code_nodes:
        link_id = str(node.get("link_id") or _node_link_id("code", node.get("node_id") or node.get("dep_node_id")))
        if node.get("changed") or node.get("changed_files") or node.get("diff_anchor"):
            changed_code_links.append(link_id)
            changed_feature_links.extend(_listify(node.get("mapped_rpg_link_ids")) + code_to_feature_links.get(link_id, []))
    node_link_ids = _ordered_unique(focused_semantic_links + changed_feature_links + context_link_ids)

    expanded_node_ids: List[Any] = ["focused-graph-root"]
    focused_path_node_ids: List[Any] = []
    focused_tree_node_ids: List[Any] = []

    def focus_tree_link(link_id: Any) -> None:
        link_text = str(link_id or "")
        if not link_text:
            return
        path_ids = hierarchy_paths_by_link.get(link_text)
        if path_ids:
            expanded_node_ids.extend(path_ids[:-1])
            focused_path_node_ids.extend(path_ids)
            focused_tree_node_ids.append(path_ids[-1])
            return
        feature_links = [link_text] if link_text in semantic_link_set else code_to_feature_links.get(link_text, [])
        if not feature_links:
            feature_links = [link_text] if link_text.startswith("feature-path-") or link_text == "focused-graph-root" else []
        for feature_link in feature_links:
            feature_path = semantic_path_ids_by_link.get(feature_link, [feature_link])
            expanded_node_ids.extend(feature_path[:-1])
            focused_path_node_ids.extend(feature_path)
            focused_tree_node_ids.append(feature_path[-1])

    for link_id in node_link_ids:
        focus_tree_link(link_id)

    expanded_node_ids = _ordered_unique(expanded_node_ids)
    focused_path_node_ids = _ordered_unique(focused_path_node_ids)
    focused_tree_node_ids = _ordered_unique(focused_tree_node_ids)
    focused_code_link_ids = _ordered_unique([link_id for link_id in node_link_ids if link_id in code_links])
    return {
        "node_link_ids": node_link_ids,
        "focused_node_ids": node_link_ids,
        "focused_tree_node_ids": focused_tree_node_ids,
        "focused_code_link_ids": focused_code_link_ids,
        "expanded_node_ids": expanded_node_ids,
        "default_expanded_node_ids": expanded_node_ids,
        "focused_path_node_ids": focused_path_node_ids,
        "semantic_node_ids": _ordered_unique([node.get("node_id") for node in semantic_nodes]),
        "code_node_ids": _ordered_unique([node.get("node_id") or node.get("dep_node_id") for node in code_nodes]),
        "mapping_link_ids": _ordered_unique([mapping.get("link_id") for mapping in mappings]),
        "edge_link_ids": _ordered_unique([edge.get("link_id") for edge in edges]),
        "relation_endpoint_link_ids": _ordered_unique(endpoint_link_ids),
        "context_node_ids": _ordered_unique(context_link_ids),
        "edge_depth": 1,
        "show_edges": True,
    }


def _focused_graph_metadata(
    semantic_nodes: List[Dict[str, Any]],
    code_nodes: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    hidden_counts: Dict[str, Any],
    warnings: List[Dict[str, Any]],
    graph_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    hierarchy = _focused_graph_hierarchy(semantic_nodes, code_nodes, mappings, edges, hidden_counts, warnings)
    default_focus = _focused_graph_default_focus(semantic_nodes, code_nodes, mappings, edges, warnings)
    focused_graph: Dict[str, Any] = {
        "schema": "cmind.focused_graph.v1",
        "hierarchy": hierarchy,
        "default_focus": default_focus,
    }
    if graph_context:
        focused_graph["graph_context"] = graph_context
    if hidden_counts:
        focused_graph["hidden_counts"] = hidden_counts
    if warnings:
        focused_graph["warning_count"] = len(warnings)
    return focused_graph


def _build_nodes_view(
    primary_rpg_nodes: List[Dict[str, Any]],
    primary_code_nodes: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    hidden_counts: Dict[str, Any],
    warnings: List[Dict[str, Any]],
    changed_files: List[str],
    diff_anchors: Dict[str, str],
    *,
    graph_context: Optional[Dict[str, Any]] = None,
    current_rpg_nodes: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    selected_rpg_ids = {str(row.get("node_id")) for row in primary_rpg_nodes if row.get("node_id") not in (None, "")}
    all_rpg_by_id: Dict[str, Dict[str, Any]] = {}
    for node_id, node in (current_rpg_nodes or {}).items():
        node_id_text = str(node_id)
        row = dict(node)
        row["node_id"] = node_id_text
        row.setdefault("link_id", _node_link_id("rpg", node_id_text))
        all_rpg_by_id[node_id_text] = row
    for node in primary_rpg_nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        all_rpg_by_id[node_id] = {**all_rpg_by_id.get(node_id, {}), **node, "node_id": node_id}
    rpg_by_id = all_rpg_by_id or {str(row.get("node_id")): row for row in primary_rpg_nodes if row.get("node_id") not in (None, "")}
    code_by_id = {str(row.get("node_id") or row.get("dep_node_id")): row for row in primary_code_nodes if (row.get("node_id") or row.get("dep_node_id")) not in (None, "")}
    code_ids_by_rpg: Dict[str, List[str]] = {}
    warnings_by_rpg: Dict[str, List[Dict[str, Any]]] = {}
    warnings_by_code: Dict[str, List[Dict[str, Any]]] = {}
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

    semantic_nodes: List[Dict[str, Any]] = []
    semantic_source_nodes = primary_rpg_nodes
    for node in semantic_source_nodes:
        node_id = str(node.get("node_id") or "")
        is_selected = node_id in selected_rpg_ids
        row: Dict[str, Any] = {
            "node_id": node_id,
            "link_id": node.get("link_id") or _node_link_id("rpg", node_id),
            "state": _node_state(node) if is_selected else _node_state(node, "context"),
            "mapping_status": node.get("mapping_status") or node.get("status"),
        }
        if is_selected:
            row["selected"] = True
        for key in ("name", "symbol", "type", "node_type", "path", "feature_path", "breadcrumb", "breadcrumb_path", "locate_status", "score", "reason", "apply_action"):
            _set_if_present(row, key, node.get(key))
        changed_refs = _changed_file_refs(_listify(node.get("changed_files")) or _listify(node.get("affected_files")), diff_anchors)
        if changed_refs:
            row["changed_files"] = changed_refs
        if isinstance(node.get("hidden_counts"), dict):
            row["hidden_counts"] = node.get("hidden_counts")
        mapped_code_ids = _ordered_unique(code_ids_by_rpg.get(node_id, []))
        if mapped_code_ids:
            mapped_code_refs: List[Dict[str, Any]] = []
            for code_id in mapped_code_ids:
                code = code_by_id.get(code_id, {})
                path = code.get("path") or code.get("file") or code.get("module") or _dep_node_path(code_id)
                symbol = code.get("symbol") or code.get("name") or _symbol_from_dep(code_id, code)
                ref: Dict[str, Any] = {
                    "node_id": code_id,
                    "link_id": _edge_endpoint_link_id(code_id, rpg_by_id, code_by_id),
                    "path": path,
                    "symbol": symbol,
                }
                for key in ("type", "kind", "line_range", "state", "source"):
                    _set_if_present(ref, key, code.get(key))
                mapped_code_refs.append(ref)
            row["mapped_code"] = mapped_code_refs
            row["mapped_code_node_ids"] = mapped_code_ids
            row["mapped_code_link_ids"] = [ref["link_id"] for ref in mapped_code_refs]
            row["mapped_code_paths"] = _ordered_unique([ref.get("path") for ref in mapped_code_refs])
            row["mapped_code_symbols"] = _ordered_unique([ref.get("symbol") for ref in mapped_code_refs])
            if row["mapped_code_paths"]:
                row["mapped_code_path"] = row["mapped_code_paths"][0]
            if row["mapped_code_symbols"]:
                row["mapped_code_symbol"] = row["mapped_code_symbols"][0]
            row["mapped_code_count"] = len(mapped_code_refs)
        if warnings_by_rpg.get(node_id):
            row["warning_types"] = _ordered_unique([warning.get("type") for warning in warnings_by_rpg[node_id]])
        semantic_nodes.append(row)

    code_nodes: List[Dict[str, Any]] = []
    for node in primary_code_nodes:
        code_id = str(node.get("node_id") or node.get("dep_node_id") or "")
        path = node.get("path") or _dep_node_path(code_id)
        row = {
            "node_id": code_id,
            "dep_node_id": code_id,
            "link_id": node.get("link_id") or _node_link_id("code", code_id),
            "state": _node_state(node, "mapped"),
        }
        for key in ("name", "symbol", "type", "kind", "path", "module", "file", "signature", "source", "source_feature", "source_features", "breadcrumb"):
            _set_if_present(row, key, node.get(key))
        _set_if_present(row, "path", path)
        line_range = node.get("line_range") if isinstance(node.get("line_range"), dict) else _line_range_from(node)
        if line_range:
            row["line_range"] = line_range
        changed_refs = _changed_file_refs([path] if path in diff_anchors else [], diff_anchors)
        if changed_refs:
            row["changed"] = True
            row["changed_files"] = changed_refs
            row["diff_anchor"] = changed_refs[0].get("diff_anchor")
        rpg_ids = _ordered_unique([mapping.get("rpg_node_id") for mapping in mappings if (mapping.get("code_node_id") or mapping.get("dep_node_id")) == code_id])
        if rpg_ids:
            row["mapped_rpg_node_ids"] = rpg_ids
            row["mapped_rpg_link_ids"] = [_edge_endpoint_link_id(rpg_id, rpg_by_id, code_by_id) for rpg_id in rpg_ids]
        if warnings_by_code.get(code_id):
            row["warning_types"] = _ordered_unique([warning.get("type") for warning in warnings_by_code[code_id]])
        code_nodes.append(row)

    mapping_rows: List[Dict[str, Any]] = []
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        row: Dict[str, Any] = {
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

    edge_rows: List[Dict[str, Any]] = []
    for edge in edges:
        source_id = edge.get("source_node_id")
        target_id = edge.get("target_node_id")
        relation = edge.get("relation") or "dependency"
        row: Dict[str, Any] = {
            "link_id": _node_link_id("edge", f"{source_id}-{relation}-{target_id}"),
            "source_node_id": source_id,
            "target_node_id": target_id,
            "source_link_id": edge.get("source_link_id") or _edge_endpoint_link_id(source_id, rpg_by_id, code_by_id),
            "target_link_id": edge.get("target_link_id") or _edge_endpoint_link_id(target_id, rpg_by_id, code_by_id),
            "relation": relation,
            "state": edge.get("status") or "visible",
        }
        for key in ("direction", "source", "source_graph", "edge_source", "relation_source", "path", "reason", "rpg_node_id", "neighbor_node_id", "name", "source_path", "target_path", "source_name", "target_name"):
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
        graph_context or {},
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
        "caps": dict(_LEGACY_FOCUSED_VIEW_CAPS),
        "graph_context": graph_context or {},
    }


_LEGACY_FOCUSED_VIEW_CAPS = {"primary_rpg_nodes": 20, "primary_code_nodes": 50, "edges": 80}
_FOCUSED_WARNING_TYPES = {"missing_mapping", "missing_reason", "stale_graph"}
_HIDDEN_CONTEXT_REASON = "selected by plan/impact, hidden because no modified mapped code"


def _set_if_present(row: Dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        row[key] = value


def _focus_reason(candidate: Dict[str, Any], impact: Dict[str, Any]) -> str:
    for source in (candidate, impact):
        for key in ("reason", "hit_reason", "rationale", "explanation"):
            value = source.get(key) if isinstance(source, dict) else None
            if value not in (None, ""):
                return str(value)
    return ""


def _slug_id(prefix: str, value: Any) -> str:
    raw = str(value or prefix)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-_")
    return f"{prefix}-{slug or 'node'}"


def _line_range_from(row: Dict[str, Any]) -> Dict[str, Any]:
    start = row.get("line_start") or row.get("start_line") or row.get("lineno") or row.get("line")
    end = row.get("line_end") or row.get("end_line") or start
    line_range: Dict[str, Any] = {}
    _set_if_present(line_range, "start", start)
    _set_if_present(line_range, "end", end)
    return line_range


def _symbol_from_dep(dep_id: str, row: Dict[str, Any]) -> str:
    for key in ("symbol", "name", "qualname", "qualified_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    dep_text = str(dep_id)
    return dep_text.rsplit(":", 1)[-1] if ":" in dep_text else dep_text


def _rpg_node_entry(node_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    row: Dict[str, Any] = {"node_id": node_id, "link_id": _slug_id("rpg", node_id)}
    name = raw.get("name") or raw.get("title")
    node_type = raw.get("node_type") or raw.get("type") or raw.get("type_name")
    path = raw.get("path") or raw.get("file") or meta.get("path")
    feature_path = raw.get("feature_path") or meta.get("feature_path") or raw.get("breadcrumb") or meta.get("breadcrumb")
    _set_if_present(row, "name", name)
    _set_if_present(row, "symbol", raw.get("symbol") or name)
    _set_if_present(row, "node_type", node_type)
    _set_if_present(row, "type", node_type)
    _set_if_present(row, "path", path)
    _set_if_present(row, "feature_path", feature_path)
    _set_if_present(row, "breadcrumb", feature_path or path)
    return row


def _dep_node_entry(dep_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"node_id": dep_id, "dep_node_id": dep_id, "link_id": _slug_id("code", dep_id)}
    for key in (
        "name",
        "symbol",
        "qualname",
        "qualified_name",
        "type",
        "kind",
        "module",
        "path",
        "file",
        "signature",
        "line_start",
        "line_end",
        "start_line",
        "end_line",
        "lineno",
        "line",
    ):
        _set_if_present(row, key, raw.get(key))
    if not row.get("path"):
        _set_if_present(row, "path", row.get("file") or _dep_node_path(dep_id) or row.get("module"))
    row["symbol"] = _symbol_from_dep(dep_id, row)
    _set_if_present(row, "breadcrumb", row.get("path"))
    line_range = _line_range_from(row)
    if line_range:
        row["line_range"] = line_range
    return row


_CONTAINMENT_RELATIONS = {"contains", "CONTAINS", "composes", "COMPOSES"}


def _edge_relation(edge: Dict[str, Any], default: str) -> str:
    attrs = edge.get("attrs") if isinstance(edge.get("attrs"), dict) else {}
    relation = (
        edge.get("relation")
        or edge.get("type")
        or edge.get("edge_type")
        or edge.get("kind")
        or attrs.get("relation")
        or attrs.get("type")
        or attrs.get("edge_type")
        or attrs.get("kind")
        or default
    )
    return str(relation)


def _coerce_rpg_edge(edge: Any) -> Optional[Dict[str, Any]]:
    if isinstance(edge, dict):
        source = edge.get("source_node_id") or edge.get("source") or edge.get("from") or edge.get("src")
        target = edge.get("target_node_id") or edge.get("target") or edge.get("to") or edge.get("dst")
        relation = _edge_relation(edge, "semantic")
        if relation in _CONTAINMENT_RELATIONS or source in (None, "") or target in (None, ""):
            return None
        row = {
            "source_node_id": str(source),
            "target_node_id": str(target),
            "relation": relation,
            "source": "rpg_semantic",
            "source_graph": "rpg",
            "relation_source": "rpg_semantic",
            "edge_source": "rpg",
        }
        for key in ("direction", "name", "path", "reason", "status"):
            _set_if_present(row, key, edge.get(key))
        return row
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        relation = str(edge[2]) if len(edge) >= 3 else "semantic"
        if relation in _CONTAINMENT_RELATIONS:
            return None
        return {"source_node_id": str(edge[0]), "target_node_id": str(edge[1]), "relation": relation, "source": "rpg_semantic", "source_graph": "rpg", "relation_source": "rpg_semantic", "edge_source": "rpg"}
    return None


def _coerce_dep_edge(edge: Any) -> Optional[Dict[str, Any]]:
    if isinstance(edge, dict):
        source = edge.get("source_node_id") or edge.get("source") or edge.get("from") or edge.get("caller") or edge.get("src")
        target = edge.get("target_node_id") or edge.get("target") or edge.get("to") or edge.get("callee") or edge.get("dst")
        relation = _edge_relation(edge, "dep_graph")
        if relation in _CONTAINMENT_RELATIONS or source in (None, "") or target in (None, ""):
            return None
        row = {
            "source_node_id": str(source),
            "target_node_id": str(target),
            "relation": relation,
            "source": "dep_graph",
            "source_graph": "dep_graph",
            "relation_source": "dep_graph",
            "edge_source": "dep_graph",
        }
        _set_if_present(row, "reason", edge.get("reason"))
        return row
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        relation = str(edge[2]) if len(edge) >= 3 else "dep_graph"
        if relation in _CONTAINMENT_RELATIONS:
            return None
        return {"source_node_id": str(edge[0]), "target_node_id": str(edge[1]), "relation": relation, "source": "dep_graph", "source_graph": "dep_graph", "relation_source": "dep_graph", "edge_source": "dep_graph"}
    return None


def _current_rpg_context() -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[Dict[str, Any]],
]:
    rpg_data = _load_json_artifact(REPO_RPG_FILE)
    if not isinstance(rpg_data, dict) or rpg_data.get("_error"):
        return {}, {}, [], [], {}, {}, [{"type": "stale_graph", "message": f"RPG file not available: {REPO_RPG_FILE}"}]

    rpg_nodes: Dict[str, Dict[str, Any]] = {}

    def visit_rpg_node(raw: Any, breadcrumb: List[str]) -> None:
        if not isinstance(raw, dict):
            return
        name = raw.get("name") or raw.get("title")
        node_breadcrumb = breadcrumb + ([str(name)] if name not in (None, "") else [])
        node_id = raw.get("id") or raw.get("node_id")
        if node_id not in (None, ""):
            row = _rpg_node_entry(str(node_id), raw)
            if node_breadcrumb:
                row["breadcrumb"] = node_breadcrumb
                row["breadcrumb_path"] = " / ".join(node_breadcrumb)
            rpg_nodes[str(node_id)] = row
        children = raw.get("children") or []
        if isinstance(children, dict):
            children = list(children.values())
        if isinstance(children, (list, tuple)):
            for child in children:
                visit_rpg_node(child, node_breadcrumb)

    visit_rpg_node(rpg_data.get("root"), [])
    raw_rpg_nodes = rpg_data.get("nodes")
    if isinstance(raw_rpg_nodes, dict):
        for node_id, raw in raw_rpg_nodes.items():
            raw_dict = raw if isinstance(raw, dict) else {}
            existing = rpg_nodes.get(str(node_id), {})
            rpg_nodes[str(node_id)] = {**_rpg_node_entry(str(node_id), raw_dict), **existing}
    elif isinstance(raw_rpg_nodes, (list, tuple)):
        for raw in raw_rpg_nodes:
            if isinstance(raw, dict):
                node_id = raw.get("id") or raw.get("node_id")
                if node_id not in (None, ""):
                    existing = rpg_nodes.get(str(node_id), {})
                    rpg_nodes[str(node_id)] = {**_rpg_node_entry(str(node_id), raw), **existing}

    dep_graph = rpg_data.get("dep_graph") if isinstance(rpg_data.get("dep_graph"), dict) else {}
    dep_nodes: Dict[str, Dict[str, Any]] = {}
    dep_to_rpg: Dict[str, List[str]] = {}
    rpg_to_dep: Dict[str, List[str]] = {}
    raw_dep_nodes = dep_graph.get("nodes")
    if isinstance(raw_dep_nodes, dict):
        dep_items = raw_dep_nodes.items()
    elif isinstance(raw_dep_nodes, (list, tuple)):
        dep_items = []
        for raw in raw_dep_nodes:
            if isinstance(raw, dict):
                dep_id = raw.get("id") or raw.get("node_id") or raw.get("dep_node_id")
                if dep_id not in (None, ""):
                    dep_items.append((dep_id, raw))
    else:
        dep_items = []
    for dep_id, raw in dep_items:
        dep_id_text = str(dep_id)
        raw_dict = raw if isinstance(raw, dict) else {}
        dep_nodes[dep_id_text] = _dep_node_entry(dep_id_text, raw_dict)
        linked = _ordered_unique([str(item) for item in _listify(raw_dict.get("rpg_nodes") or raw_dict.get("features") or raw_dict.get("source_features"))])
        if linked:
            dep_to_rpg[dep_id_text] = linked
            for rpg_id in linked:
                rpg_to_dep.setdefault(rpg_id, []).append(dep_id_text)

    raw_dep_to_rpg = rpg_data.get("_dep_to_rpg_map") if isinstance(rpg_data.get("_dep_to_rpg_map"), dict) else {}
    for dep_id, rpg_ids in raw_dep_to_rpg.items():
        dep_id_text = str(dep_id)
        linked = _ordered_unique([str(item) for item in _listify(rpg_ids)])
        if linked:
            dep_to_rpg[dep_id_text] = _ordered_unique((dep_to_rpg.get(dep_id_text) or []) + linked)
            for rpg_id in linked:
                rpg_to_dep.setdefault(rpg_id, []).append(dep_id_text)

    raw_rpg_to_dep = rpg_data.get("_rpg_to_dep_map") if isinstance(rpg_data.get("_rpg_to_dep_map"), dict) else {}
    for rpg_id, dep_ids in raw_rpg_to_dep.items():
        rpg_id_text = str(rpg_id)
        linked = _ordered_unique([str(item) for item in _listify(dep_ids)])
        if linked:
            rpg_to_dep[rpg_id_text] = _ordered_unique((rpg_to_dep.get(rpg_id_text) or []) + linked)
            for dep_id in linked:
                dep_to_rpg.setdefault(dep_id, []).append(rpg_id_text)

    for rpg_id, dep_ids in list(rpg_to_dep.items()):
        rpg_to_dep[rpg_id] = _ordered_unique(dep_ids)
    for dep_id, rpg_ids in list(dep_to_rpg.items()):
        dep_to_rpg[dep_id] = _ordered_unique(rpg_ids)

    def edge_values(value: Any) -> List[Any]:
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return []

    rpg_edges = []
    for raw_edge in edge_values(rpg_data.get("edges")) + edge_values(rpg_data.get("semantic_edges")) + edge_values(rpg_data.get("feature_edges")):
        edge = _coerce_rpg_edge(raw_edge)
        if edge:
            rpg_edges.append(edge)

    dep_edges = []
    for raw_edge in edge_values(dep_graph.get("edges")) + edge_values(dep_graph.get("syntax_edges")):
        edge = _coerce_dep_edge(raw_edge)
        if edge:
            dep_edges.append(edge)

    warnings: List[Dict[str, Any]] = []
    if not rpg_nodes and not dep_nodes:
        warnings.append({"type": "stale_graph", "message": "Current RPG contains no indexed feature or dependency nodes"})
    return rpg_nodes, dep_nodes, rpg_edges, dep_edges, rpg_to_dep, dep_to_rpg, warnings


def _feature_evidence_groups(
    artifacts: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    code_deltas: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else {}
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    impact_results = _impact_results(artifacts)
    current_rpg_nodes, current_dep_nodes, current_rpg_edges, current_dep_edges, current_rpg_to_dep, current_dep_to_rpg, graph_warnings = _current_rpg_context()
    current_graph_available = bool(current_rpg_nodes or current_dep_nodes)
    locate_ids = {str(row.get("node_id")) for row in locate.get("results") or [] if isinstance(row, dict) and row.get("node_id")}
    applied_by_id = {
        str(row.get("node_id")): row
        for row in apply_result.get("applied_features") or []
        if isinstance(row, dict) and row.get("node_id")
    }
    changed_files = _ordered_unique(
        [_code_delta_file(delta) for delta in code_deltas]
        + [str(path) for path in _listify(code_result.get("files_modified"))]
        + [str(change.get("file_path")) for change in plan.get("code_changes") or [] if isinstance(change, dict) and change.get("file_path")]
    )
    changed_ranges = changed_line_ranges_by_file(code_deltas)
    warnings: List[Dict[str, Any]] = []
    warning_keys: set[str] = set()

    def add_warning(warning_type: str, message: str, **context: Any) -> None:
        if warning_type not in _FOCUSED_WARNING_TYPES:
            return
        row: Dict[str, Any] = {"type": warning_type, "message": message}
        for key, value in context.items():
            _set_if_present(row, key, value)
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in warning_keys:
            warning_keys.add(key)
            warnings.append(row)

    for warning in graph_warnings:
        if isinstance(warning, dict):
            add_warning(str(warning.get("type") or "stale_graph"), str(warning.get("message") or "Current RPG graph may be stale"))

    if apply_result and apply_result.get("dep_graph_refreshed") is False:
        add_warning("stale_graph", "Apply result says the dependency graph was not refreshed", apply_status=_apply_status(apply_result))

    primary_rpg_nodes_all: List[Dict[str, Any]] = []
    mapping_rows_all: List[Dict[str, Any]] = []
    code_node_by_id: Dict[str, Dict[str, Any]] = {}
    all_edges: List[Dict[str, Any]] = []
    impact_hidden_counts: Dict[str, int] = {}

    def count_value(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def add_code_node(dep_id: Any, *, source_feature: Any = None, source: str = "mapping", fallback_path: Any = None) -> None:
        if dep_id in (None, ""):
            return
        dep_id_text = str(dep_id)
        if dep_id_text in code_node_by_id:
            row = code_node_by_id[dep_id_text]
            row["source"] = "+".join(_ordered_unique(_listify(row.get("source")) + [source]))
            if source_feature not in (None, ""):
                row["source_features"] = _ordered_unique(_listify(row.get("source_features")) + [str(source_feature)])
            return
        current = current_dep_nodes.get(dep_id_text, {})
        row = dict(current) if current else {"node_id": dep_id_text, "dep_node_id": dep_id_text}
        row.setdefault("node_id", dep_id_text)
        row.setdefault("dep_node_id", dep_id_text)
        row.setdefault("link_id", _node_link_id("code", dep_id_text))
        row.setdefault("symbol", _symbol_from_dep(dep_id_text, row))
        _set_if_present(row, "path", fallback_path)
        row["status"] = "mapped"
        row["graph_state"] = "available" if dep_id_text in current_dep_nodes else "unavailable"
        row["source"] = source
        if source_feature not in (None, ""):
            row["source_feature"] = str(source_feature)
            row["source_features"] = [str(source_feature)]
        if row.get("path") in changed_files:
            row["changed"] = True
        if current_graph_available and dep_id_text not in current_dep_nodes:
            row["status"] = "stale_graph"
            row["graph_state"] = "stale_graph"
            add_warning("stale_graph", "Mapped code node is absent from the current dependency graph", dep_node_id=dep_id_text)
        code_node_by_id[dep_id_text] = row

    neighbor_specs = [
        ("callers", "caller", "upstream", "total_callers"),
        ("callees", "callee", "downstream", "total_callees"),
        ("imports", "imports", "downstream", "total_imports"),
        ("inheritance", "inherits", "downstream", "total_inheritance"),
    ]
    edge_keys: set[tuple[str, str, str]] = set()

    def mapped_feature_id(dep_id: Any) -> Optional[str]:
        dep_text = str(dep_id or "")
        if not dep_text:
            return None
        mapped_ids = _ordered_unique(_listify(current_dep_to_rpg.get(dep_text)) + _listify(current_dep_nodes.get(dep_text, {}).get("rpg_nodes")))
        return str(mapped_ids[0]) if mapped_ids else None

    def neighbor_id_from(item: Any, impact_key: str) -> str:
        if isinstance(item, dict):
            side_keys = {
                "callers": ("source_node_id", "caller_node_id", "caller", "from", "src"),
                "callees": ("target_node_id", "callee_node_id", "callee", "to", "dst"),
                "imports": ("target_node_id", "import_node_id", "imported_node_id", "import", "to", "dst"),
                "inheritance": ("target_node_id", "parent_node_id", "base_node_id", "inherits", "to", "dst"),
            }
            for key in side_keys.get(impact_key, ()):
                value = item.get(key)
                if value not in (None, ""):
                    return str(value)
            for key in ("rpg_node_id", "feature_node_id", "node_id", "dep_node_id", "code_node_id", "id", "path", "file"):
                value = item.get(key)
                if value not in (None, ""):
                    return str(value)
            return ""
        if item not in (None, ""):
            return str(item)
        return ""

    def neighbor_metadata(item: Any, neighbor_id: str) -> Dict[str, Any]:
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = {}
        dep = current_dep_nodes.get(neighbor_id, {})
        rpg = current_rpg_nodes.get(neighbor_id, {})
        path = row.get("path") or row.get("file") or dep.get("path") or dep.get("file") or dep.get("module") or rpg.get("path")
        name = row.get("name") or row.get("symbol") or dep.get("name") or dep.get("symbol") or rpg.get("name") or rpg.get("symbol")
        return {"path": path, "name": name}

    def add_impact_edges(node_id: str, impact: Dict[str, Any], focus_reason: str) -> None:
        for impact_key, relation, direction, _total_key in neighbor_specs:
            for item in _listify(impact.get(impact_key)):
                raw_neighbor_id = neighbor_id_from(item, impact_key)
                if not raw_neighbor_id:
                    continue
                neighbor_id = mapped_feature_id(raw_neighbor_id) or raw_neighbor_id
                if neighbor_id == node_id:
                    continue
                if impact_key == "callers":
                    source_id, target_id = neighbor_id, node_id
                else:
                    source_id, target_id = node_id, neighbor_id
                key = (str(source_id), str(target_id), relation)
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                metadata = neighbor_metadata(item, raw_neighbor_id)
                edge_row: Dict[str, Any] = {
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "relation": relation,
                    "direction": direction,
                    "source": "impact",
                    "source_graph": "impact",
                    "edge_source": "impact",
                    "relation_source": impact_key,
                    "rpg_node_id": node_id,
                    "neighbor_node_id": neighbor_id,
                    "reason": focus_reason or f"impact {impact_key}",
                }
                if neighbor_id not in current_rpg_nodes:
                    side = "source" if impact_key == "callers" else "target"
                    edge_row[f"{side}_link_id"] = _node_link_id("context", raw_neighbor_id)
                _set_if_present(edge_row, "path", metadata.get("path"))
                _set_if_present(edge_row, "name", metadata.get("name"))
                side = "source" if impact_key == "callers" else "target"
                _set_if_present(edge_row, f"{side}_path", metadata.get("path"))
                _set_if_present(edge_row, f"{side}_name", metadata.get("name"))
                all_edges.append(edge_row)

    for candidate in candidates:
        raw_node_id = candidate.get("node_id")
        if raw_node_id in (None, ""):
            continue
        node_id = str(raw_node_id)
        impact = impact_results.get(node_id) if isinstance(impact_results.get(node_id), dict) else {}
        current_node = current_rpg_nodes.get(node_id, {})
        relations = _mapped_code_relations(candidate, impact)
        relation_by_dep = {str(row.get("dep_node_id") or row.get("node_id")): row for row in relations if row.get("dep_node_id") or row.get("node_id")}
        mapped_dep_ids = _ordered_unique(list(relation_by_dep) + (current_rpg_to_dep.get(node_id) or []))
        modified_dep_ids = _modified_dep_ids(mapped_dep_ids, relation_by_dep, current_dep_nodes, changed_ranges, changed_files)
        relation_paths = _ordered_unique(
            [row.get("path") for row in relations]
            + [current_dep_nodes.get(dep_id, {}).get("path") for dep_id in mapped_dep_ids]
        )
        affected_files = _ordered_unique(_listify(impact.get("affected_files")) + relation_paths)
        relevant_files = set(_ordered_unique(affected_files + [
            candidate.get("path"),
            candidate.get("meta_path"),
            current_node.get("path"),
        ]))
        relevant_deltas = [delta for delta in code_deltas if _code_delta_file(delta) in relevant_files]
        changed_for_node = _ordered_unique([_code_delta_file(delta) for delta in relevant_deltas])
        focus_reason = _focus_reason(candidate, impact)
        if not focus_reason:
            add_warning("missing_reason", "Focused view has no explicit selection reason", node_id=node_id)
        if not mapped_dep_ids:
            add_warning("missing_mapping", "Selected RPG node has no mapped code node", node_id=node_id)
        for dep_id in mapped_dep_ids:
            if current_graph_available and dep_id not in current_dep_nodes:
                add_warning("stale_graph", "Mapped code node is absent from the current dependency graph", node_id=node_id, dep_node_id=dep_id)
        if current_graph_available and node_id not in current_rpg_nodes:
            add_warning("stale_graph", "Selected RPG node is absent from the current RPG graph", node_id=node_id)

        impact_summary = impact.get("impact_summary") if isinstance(impact.get("impact_summary"), dict) else {}
        node_hidden_counts: Dict[str, int] = {}
        for impact_key, _relation, _direction, total_key in neighbor_specs:
            items = _listify(impact.get(impact_key))
            total = count_value(impact_summary.get(total_key), len(items))
            hidden = max(0, total - len(items))
            if hidden:
                node_hidden_counts[impact_key] = hidden
                impact_hidden_counts[impact_key] = impact_hidden_counts.get(impact_key, 0) + hidden
        add_impact_edges(node_id, impact, focus_reason)

        apply_row = applied_by_id.get(node_id, {})
        rpg_row: Dict[str, Any] = dict(current_node) if current_node else {"node_id": node_id, "link_id": _node_link_id("rpg", node_id)}
        rpg_row["node_id"] = node_id
        rpg_row.setdefault("link_id", _node_link_id("rpg", node_id))
        rpg_row["status"] = "mapped" if mapped_dep_ids else "missing"
        rpg_row["mapping_status"] = "mapped" if mapped_dep_ids else "missing"
        rpg_row["graph_state"] = "available" if node_id in current_rpg_nodes else "unavailable"
        if current_graph_available and node_id not in current_rpg_nodes:
            rpg_row["graph_state"] = "stale_graph"
        _set_if_present(rpg_row, "name", candidate.get("name") or impact.get("name") or current_node.get("name"))
        _set_if_present(rpg_row, "symbol", candidate.get("symbol") or impact.get("symbol") or rpg_row.get("name"))
        _set_if_present(rpg_row, "node_type", candidate.get("node_type") or candidate.get("type_name") or candidate.get("type") or current_node.get("node_type"))
        _set_if_present(rpg_row, "type", rpg_row.get("node_type"))
        _set_if_present(rpg_row, "path", candidate.get("path") or candidate.get("meta_path") or current_node.get("path"))
        _set_if_present(rpg_row, "feature_path", candidate.get("feature_path") or current_node.get("feature_path"))
        _set_if_present(rpg_row, "score", candidate.get("score"))
        _set_if_present(rpg_row, "reason", focus_reason)
        if node_id not in locate_ids and locate:
            rpg_row["locate_status"] = candidate.get("locate_state") or "missing"
        if affected_files:
            rpg_row["affected_files"] = affected_files
        if changed_for_node:
            rpg_row["changed_files"] = changed_for_node
        if node_hidden_counts:
            rpg_row["hidden_counts"] = node_hidden_counts
        _set_if_present(rpg_row, "apply_action", apply_row.get("action") or apply_row.get("change"))
        primary_rpg_nodes_all.append(rpg_row)

        if mapped_dep_ids:
            for dep_id in modified_dep_ids:
                relation = relation_by_dep.get(dep_id, {})
                current_dep = current_dep_nodes.get(dep_id, {})
                path = current_dep.get("path") or relation.get("path")
                source_parts = _ordered_unique(_listify(relation.get("source")) + (["current_rpg"] if dep_id in (current_rpg_to_dep.get(node_id) or []) else []))
                mapping_status = "mapped"
                if current_graph_available and dep_id not in current_dep_nodes:
                    mapping_status = "stale_graph"
                    add_warning("stale_graph", "Mapped code node is absent from the current dependency graph", node_id=node_id, dep_node_id=dep_id)
                mapping_row: Dict[str, Any] = {
                    "rpg_node_id": node_id,
                    "code_node_id": dep_id,
                    "dep_node_id": dep_id,
                    "status": mapping_status,
                    "source": "+".join(source_parts) or "selected_feature",
                }
                _set_if_present(mapping_row, "path", path)
                _set_if_present(mapping_row, "reason", focus_reason)
                if changed_for_node:
                    mapping_row["changed_files"] = changed_for_node
                mapping_rows_all.append(mapping_row)
                add_code_node(dep_id, source_feature=node_id, source=mapping_row["source"], fallback_path=path)
        else:
            mapping_row = {"rpg_node_id": node_id, "status": "missing"}
            _set_if_present(mapping_row, "reason", focus_reason)
            if changed_for_node:
                mapping_row["changed_files"] = changed_for_node
            mapping_rows_all.append(mapping_row)

    selected_rpg_ids_all = {str(row.get("node_id") or "") for row in primary_rpg_nodes_all if row.get("node_id") not in (None, "")}
    modified_mapped_rpg_ids = _ordered_unique([row.get("rpg_node_id") for row in mapping_rows_all if row.get("code_node_id")])
    edge_endpoint_rpg_ids = _ordered_unique([
        endpoint
        for edge in all_edges
        for endpoint in (edge.get("source_node_id"), edge.get("target_node_id"))
        if str(endpoint or "") in current_rpg_nodes or str(endpoint or "") in selected_rpg_ids_all
    ])
    visible_rpg_ids = _ordered_unique(modified_mapped_rpg_ids + edge_endpoint_rpg_ids)
    visible_rpg_id_set = set(visible_rpg_ids)
    primary_rpg_by_id = {
        str(row.get("node_id")): row
        for row in primary_rpg_nodes_all
        if row.get("node_id") not in (None, "")
    }
    primary_rpg_nodes: List[Dict[str, Any]] = []
    for node_id in visible_rpg_ids:
        row = primary_rpg_by_id.get(node_id)
        if row is None and node_id in current_rpg_nodes:
            row = dict(current_rpg_nodes[node_id])
            row["node_id"] = node_id
            row.setdefault("link_id", _node_link_id("rpg", node_id))
            row.setdefault("status", "context")
            row.setdefault("graph_state", "available")
        if row is not None:
            primary_rpg_nodes.append(row)
    visible_primary_rpg_ids = {str(row.get("node_id") or "") for row in primary_rpg_nodes}
    hidden_context_nodes: List[Dict[str, Any]] = []
    for row in primary_rpg_nodes_all:
        node_id = str(row.get("node_id") or "")
        if not node_id or node_id in visible_primary_rpg_ids:
            continue
        hidden_row = dict(row)
        hidden_row["hidden_reason"] = _HIDDEN_CONTEXT_REASON
        hidden_row["reason"] = _HIDDEN_CONTEXT_REASON
        hidden_context_nodes.append(hidden_row)
    primary_code_nodes_all = list(code_node_by_id.values())
    visible_code_ids = {
        str(row.get("code_node_id") or row.get("dep_node_id") or "")
        for row in mapping_rows_all
        if str(row.get("rpg_node_id") or "") in visible_primary_rpg_ids and (row.get("code_node_id") or row.get("dep_node_id")) not in (None, "")
    }
    primary_code_nodes = [row for row in primary_code_nodes_all if str(row.get("node_id") or row.get("dep_node_id") or "") in visible_code_ids]
    mappings = [
        row
        for row in mapping_rows_all
        if str(row.get("rpg_node_id") or "") in visible_primary_rpg_ids
    ]
    edges = all_edges
    hidden_counts: Dict[str, Any] = {}
    for key, count in impact_hidden_counts.items():
        if count:
            hidden_counts[key] = hidden_counts.get(key, 0) + count

    matched_files = {file_path for node in primary_rpg_nodes_all for file_path in node.get("changed_files") or []}
    unmatched_code_deltas = [delta for delta in code_deltas if _code_delta_file(delta) not in matched_files]
    mapped_code_relations = sum(1 for row in mappings if row.get("code_node_id"))
    missing_mappings = sum(1 for row in mappings if row.get("status") == "missing")
    diff_anchors = _diff_anchor_map(code_deltas)
    graph_context = {
        "current_graph_available": current_graph_available,
        "current_rpg_nodes": len(current_rpg_nodes),
        "current_dep_nodes": len(current_dep_nodes),
        "current_rpg_edges": len(current_rpg_edges),
        "current_dep_edges": len(current_dep_edges),
    }
    nodes_view = _build_nodes_view(
        primary_rpg_nodes,
        primary_code_nodes,
        mappings,
        edges,
        hidden_counts,
        warnings,
        changed_files,
        diff_anchors,
        graph_context=graph_context,
        current_rpg_nodes=current_rpg_nodes,
    )
    summary = {
        "selected_feature_groups": len(primary_rpg_nodes_all),
        "primary_rpg_nodes": len(primary_rpg_nodes),
        "primary_code_nodes": len(primary_code_nodes),
        "mapped_code_relations": mapped_code_relations,
        "missing_mappings": missing_mappings,
        "edges": len(edges),
        "warnings": len(warnings),
        "changed_files": len(changed_files),
        "hidden_context_nodes": len(hidden_context_nodes),
        "review_status": result.get("type", "review"),
        "apply_status": _apply_status(apply_result),
        "verification_status": _test_status(result, code_result, apply_result),
    }
    nodes_view["hidden_context_nodes"] = hidden_context_nodes
    nodes_summary = dict(nodes_view.get("summary", {}))
    for key in ("selected_feature_groups", "mapped_code_relations", "missing_mappings", "hidden_context_nodes", "review_status", "apply_status", "verification_status"):
        _set_if_present(nodes_summary, key, summary.get(key))
    nodes_view["summary"] = nodes_summary
    return {
        "summary": summary,
        "nodes_view": nodes_view,
        "primary_rpg_nodes": primary_rpg_nodes,
        "primary_code_nodes": primary_code_nodes,
        "mappings": mappings,
        "edges": edges,
        "hidden_counts": hidden_counts,
        "hidden_context_nodes": hidden_context_nodes,
        "warnings": warnings,
        "changed_files": changed_files,
        "unmatched_code_deltas": unmatched_code_deltas,
        "apply": {
            "status": _apply_status(apply_result),
            "dep_graph_refreshed": apply_result.get("dep_graph_refreshed"),
        },
        "review": {
            "status": result.get("type", "review"),
            "success": result.get("success", result.get("type") == "skipped"),
            "iterations": len(result.get("iterations") or []),
            "suggestions": len(result.get("suggestions") or []),
        },
    }


def _review_summary_cards(
    result: Dict[str, Any],
    artifacts: Dict[str, Any],
    focused_view: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    summary = focused_view.get("summary") if isinstance(focused_view, dict) else {}
    result_value = "passed" if result.get("success", result.get("type") == "skipped") else "failed"
    changed_files = summary.get("changed_files", len(code_result.get("files_modified") or []))
    return [
        {"label": "Review", "value": result.get("type", "review"), "detail": result_value},
        {"label": "Selected features", "value": summary.get("selected_feature_groups", len(plan.get("affected_nodes") or []))},
        {"label": "Mapped code relations", "value": summary.get("mapped_code_relations", 0)},
        {"label": "Missing mappings", "value": summary.get("missing_mappings", 0)},
        {"label": "Changed files", "value": changed_files},
        {"label": "Verification", "value": summary.get("verification_status") or _test_status(result, code_result, apply_result)},
    ]


def _review_timeline(result: Dict[str, Any], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    validate = artifacts.get("validate") if isinstance(artifacts.get("validate"), dict) else None
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else None
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else None
    impact = artifacts.get("impact") if isinstance(artifacts.get("impact"), dict) else None
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else None
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else None
    apply_reason = "artifact not found"
    if apply_result is not None:
        apply_reason = (
            f"{len(apply_result.get('applied_features') or [])} applied features; "
            f"dep_graph_refreshed={apply_result.get('dep_graph_refreshed')}"
        )
    return [
        {"name": "validate", "status": validate.get("type") if validate else "missing", "reason": validate.get("message", "") if validate else "artifact not found"},
        {"name": "locate", "status": locate.get("type") if locate else "missing", "reason": f"{len(locate.get('results') or [])} candidates" if locate else "artifact not found"},
        {"name": "plan", "status": "available" if plan else "missing", "reason": f"{len(plan.get('code_changes') or [])} code changes" if plan else "artifact not found"},
        {"name": "impact", "status": impact.get("type", "available") if impact else "missing", "reason": f"{len((impact.get('results') or {}))} impact result sets" if impact else "artifact not found"},
        {"name": "code", "status": code_result.get("last_status") if code_result else "missing", "reason": code_result.get("last_error") or f"success={code_result.get('success')}" if code_result else "artifact not found"},
        {"name": "apply/dep-refresh", "status": _apply_status(apply_result or {}), "reason": apply_reason},
        {"name": "review", "status": result.get("type"), "reason": result.get("reason") or f"success={result.get('success')}"},
    ]


def _review_verification(result: Dict[str, Any], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    validate = artifacts.get("validate") if isinstance(artifacts.get("validate"), dict) else None
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else None
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else None
    if validate:
        checks.append({"name": "validate", "status": validate.get("type"), "detail": validate.get("message", "")})
    if code_result:
        checks.append({"name": "code", "status": code_result.get("success"), "detail": code_result.get("last_error") or code_result.get("last_status")})
    if apply_result is None:
        checks.append({"name": "apply", "status": "missing", "detail": "artifact not found"})
    else:
        checks.append({
            "name": "apply",
            "status": _apply_status(apply_result),
            "detail": f"{len(apply_result.get('applied_features') or [])} applied features",
        })
    test_result = apply_result.get("test_result") if isinstance(apply_result, dict) and isinstance(apply_result.get("test_result"), dict) else {}
    checks.append({
        "name": "test",
        "status": _test_status(result, code_result or {}, apply_result or {}),
        "detail": "apply test_result" if test_result else "review/code status fallback",
    })
    checks.append({
        "name": "dep_graph refresh",
        "status": apply_result.get("dep_graph_refreshed") if apply_result is not None else "missing",
        "detail": f"apply status={_apply_status(apply_result or {})}" if apply_result is not None else "artifact not found",
    })
    checks.append({"name": "review", "status": result.get("success", result.get("type") == "skipped"), "detail": result.get("reason") or result.get("type")})
    for iteration in result.get("iterations") or []:
        checks.append({
            "name": f"review iteration {iteration.get('iteration')}",
            "status": iteration.get("post_pytest_passed"),
            "detail": iteration.get("agent_detail", ""),
        })
    return checks


def _status_from_bool(value: Any) -> Optional[str]:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return None


def _apply_status(apply_result: Dict[str, Any]) -> Any:
    return apply_result.get("type") or apply_result.get("status") or "missing"


def _test_status(result: Dict[str, Any], code_result: Dict[str, Any], apply_result: Dict[str, Any]) -> Any:
    test_result = apply_result.get("test_result") if isinstance(apply_result.get("test_result"), dict) else {}
    status = _status_from_bool(test_result.get("passed"))
    if status:
        return status
    for iteration in reversed(result.get("iterations") or []):
        if isinstance(iteration, dict):
            status = _status_from_bool(iteration.get("post_pytest_passed"))
            if status:
                return status
    if code_result.get("last_status"):
        return code_result.get("last_status")
    return _status_from_bool(result.get("success"))


def _rollback_path(apply_result: Dict[str, Any]) -> Any:
    if apply_result.get("rollback_path"):
        return apply_result.get("rollback_path")
    backups = apply_result.get("backups") if isinstance(apply_result.get("backups"), dict) else {}
    return backups.get("rpg") or backups.get("dep_graph") or apply_result.get("rollback_command")


def _artifact_path_pointers(artifact_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pointers: List[Dict[str, Any]] = []
    for row in artifact_rows:
        pointer: Dict[str, Any] = {}
        for key in ("label", "path", "status"):
            if row.get(key) not in (None, ""):
                pointer[key] = row.get(key)
        if pointer:
            pointers.append(pointer)
    return pointers


def _compact_plan_audit(plan: Dict[str, Any]) -> Dict[str, Any]:
    changes = []
    for change in plan.get("code_changes") or []:
        if not isinstance(change, dict):
            continue
        row: Dict[str, Any] = {}
        for key in ("file_path", "change_type", "action"):
            if change.get(key) not in (None, ""):
                row[key] = change.get(key)
        if row:
            changes.append(row)
    return {
        "affected_nodes": [str(node_id) for node_id in _listify(plan.get("affected_nodes"))],
        "code_changes": changes,
    }


def _compact_impact_audit(impact: Dict[str, Any]) -> Dict[str, Any]:
    results = impact.get("results") if isinstance(impact.get("results"), dict) else {}
    affected_files: List[Any] = []
    mapped_relations = 0
    node_summaries = []
    for node_id, row in results.items():
        row = row if isinstance(row, dict) else {}
        dep_nodes = _listify(row.get("dep_nodes"))
        files = _listify(row.get("affected_files"))
        affected_files.extend(files)
        mapped_relations += len(dep_nodes)
        summary = row.get("impact_summary") if isinstance(row.get("impact_summary"), dict) else {}
        node_summaries.append({
            "node_id": node_id,
            "mapped_code_relations": len(dep_nodes),
            "affected_files": len(files),
            "total_callers": summary.get("total_callers", len(row.get("callers") or [])),
            "total_callees": summary.get("total_callees", len(row.get("callees") or [])),
        })
    return {
        "type": impact.get("type"),
        "result_count": len(results),
        "affected_files": _ordered_unique(affected_files),
        "mapped_code_relations": mapped_relations,
        "results": node_summaries,
    }


def _compact_code_audit(code_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": code_result.get("success"),
        "last_status": code_result.get("last_status"),
        "commit_sha": code_result.get("commit_sha"),
        "files_modified": [str(path) for path in _listify(code_result.get("files_modified"))],
        "iterations": len(code_result.get("iterations") or []),
    }


def _compact_apply_audit(apply_result: Dict[str, Any]) -> Dict[str, Any]:
    applied = []
    for feature in apply_result.get("applied_features") or []:
        if not isinstance(feature, dict):
            continue
        row: Dict[str, Any] = {}
        for key in ("node_id", "action", "change"):
            if feature.get(key) not in (None, ""):
                row[key] = feature.get(key)
        if row:
            applied.append(row)
    test_result = apply_result.get("test_result") if isinstance(apply_result.get("test_result"), dict) else {}
    audit = {
        "status": _apply_status(apply_result),
        "dep_graph_refreshed": apply_result.get("dep_graph_refreshed"),
        "applied_features": applied,
        "rollback_path": _rollback_path(apply_result),
        "test_status": _status_from_bool(test_result.get("passed")),
    }
    for key in ("backup_timestamp", "backups", "confirmed", "before_state", "rollback_command"):
        if apply_result.get(key) not in (None, "", [], {}):
            audit[key] = apply_result.get(key)
    return audit


def _compact_review_audit(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("type", "review"),
        "success": result.get("success", result.get("type") == "skipped"),
        "iterations": len(result.get("iterations") or []),
        "suggestions": len(result.get("suggestions") or []),
        "reason": result.get("reason"),
    }


def _compact_review_evidence(
    artifacts: Dict[str, Any],
    artifact_rows: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    validate = artifacts.get("validate") if isinstance(artifacts.get("validate"), dict) else {}
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else {}
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    impact = artifacts.get("impact") if isinstance(artifacts.get("impact"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    evidence = {
        "artifact_paths": _artifact_path_pointers(artifact_rows),
        "audit_summary": {
            "validate": {"type": validate.get("type"), "message": validate.get("message")},
            "locate": {
                "type": locate.get("type"),
                "query": locate.get("query"),
                "candidate_count": len(locate.get("results") or []),
            },
            "plan": _compact_plan_audit(plan),
            "impact": _compact_impact_audit(impact),
            "code": _compact_code_audit(code_result),
            "apply": _compact_apply_audit(apply_result),
            "review": _compact_review_audit(result),
        },
    }
    for key in ("run_id", "parent_run_id", "is_final", "report_scope", "published_to"):
        evidence[key] = result.get(key)
    return evidence


def _user_decision(result: Dict[str, Any], artifacts: Dict[str, Any]) -> UserDecisionEvent:
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    current_head = read_head(REPO_DIR)
    before_state = apply_result.get("before_state") if apply_result.get("before_state") else current_head
    branch = before_state.get("head_branch") if isinstance(before_state, dict) else None
    if not branch and isinstance(current_head, dict):
        branch = current_head.get("head_branch")
    confirmed = apply_result.get("confirmed") if "confirmed" in apply_result else None
    return UserDecisionEvent(
        decision="apply",
        branch=branch,
        before_state=before_state,
        rollback_path=_rollback_path(apply_result),
        confirmed=confirmed,
        apply_status=_apply_status(apply_result),
        test_status=_test_status(result, code_result, apply_result),
    )


class _ReportPayload:
    def __init__(self, run: CommandRun, focused_view: Dict[str, Any], report_dir: Optional[Path] = None):
        self.run = run
        self.focused_view = focused_view
        self.report_dir = report_dir

    def to_dict(self) -> Dict[str, Any]:
        data = self.run.to_dict()
        if self.focused_view:
            data["focused_view"] = self.focused_view
        if self.report_dir is not None:
            data["report_dir"] = str(self.report_dir)
        return data


def _publish_review_report(
    result: Dict[str, Any],
    plan_path: Path,
    impact_path: Optional[Path],
    *,
    report_scope: str = "final",
    report_dir: Optional[Path] = None,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    report_scope = _normalize_report_scope(report_scope)
    previous_result = _load_existing_review_result()
    internal_report_paths = _existing_internal_report_paths(previous_result)
    run_id = str(result.get("run_id") or uuid.uuid4().hex)
    if parent_run_id is None:
        parent_run_id = result.get("parent_run_id")

    result["run_id"] = run_id
    result["parent_run_id"] = parent_run_id
    result["is_final"] = report_scope == "final"
    result["report_scope"] = report_scope
    result["internal_report_paths"] = internal_report_paths

    if report_scope == "none":
        result["published_to"] = None
        result.pop("report_path", None)
        _write_review_result(result)
        return result

    target_report_dir = _report_target_dir(report_scope, report_dir)
    report_timestamp = _report_timestamp()
    result["published_to"] = str(_expected_report_path(target_report_dir, report_timestamp))
    _write_review_result(result)

    artifacts = _load_review_artifacts(plan_path, impact_path)
    candidates = _selected_candidate_rows(artifacts)
    code_deltas = _code_delta_rows(artifacts)
    focused_view = _feature_evidence_groups(artifacts, candidates, code_deltas, result)
    visible_rpg_ids = {
        str(row.get("node_id"))
        for row in focused_view.get("primary_rpg_nodes", [])
        if isinstance(row, dict) and row.get("node_id") not in (None, "")
    }
    rpg_delta_rows = [row for row in candidates if str(row.get("node_id") or "") in visible_rpg_ids] if visible_rpg_ids else candidates
    artifact_rows = _artifact_links(plan_path, impact_path, internal_report_paths)
    evidence = _compact_review_evidence(artifacts, artifact_rows, result)
    try:
        report_run = CommandRun(
            command="rpg_edit",
            title="CoderMind rpg_edit Explain View",
            status=str(result.get("type", "review")),
            timestamp=report_timestamp,
            summary=_review_summary_cards(result, artifacts, focused_view),
            steps=[
                StepEvent(name=row.get("name", "stage"), status=row.get("status"), reason=row.get("reason", ""))
                for row in _review_timeline(result, artifacts)
            ],
            rpg_deltas=[
                RPGDeltaEvent(
                    node_id=row.get("node_id"),
                    name=row.get("name"),
                    type=row.get("type"),
                    path=row.get("path") or row.get("meta_path"),
                    score=row.get("score"),
                )
                for row in rpg_delta_rows
            ],
            dep_graph_deltas=[
                DepGraphDeltaEvent(
                    dep_node_id=row.get("node_id"),
                    path=row.get("path"),
                    source_feature=row.get("source_feature"),
                    change=row.get("change"),
                )
                for row in _dep_node_rows(rpg_delta_rows)
            ],
            retrievals=[
                RetrievalEvent(query=row.get("query"), tool=row.get("tool"), hits=row.get("hits"), reason=row.get("reason"))
                for row in _retrieval_rows(artifacts, candidates)
            ],
            artifacts=[
                ArtifactEvent(label=row["label"], path=row["path"], status=row.get("status"))
                for row in artifact_rows
            ],
            code_deltas=[
                CodeDeltaEvent(
                    file=row.get("file"),
                    change_type=row.get("change_type"),
                    before=row.get("before"),
                    after=row.get("after"),
                    diff=row.get("diff"),
                )
                for row in code_deltas
            ],
            verification=[
                VerificationEvent(name=row.get("name", "verification"), status=row.get("status"), detail=row.get("detail"))
                for row in _review_verification(result, artifacts)
            ],
            user_decisions=[_user_decision(result, artifacts)],
            evidence=evidence,
        )
        report_path = write_command_report(_ReportPayload(report_run, focused_view, target_report_dir))
        result["report_path"] = str(report_path)
        result["published_to"] = str(report_path)
        if report_scope == "internal":
            result["internal_report_paths"] = _ordered_unique(internal_report_paths + [str(report_path)])
    except Exception as exc:
        result["report_error"] = str(exc)
    _write_review_result(result)
    return result


# ---------------------------------------------------------------------------
# Review prompt template
# ---------------------------------------------------------------------------

IMPACT_REVIEW_PROMPT = Template("""\
# Impact Review: Verify Modified Functionality

You are a QA engineer verifying a **specific code modification** — NOT a full
project review. Focus ONLY on the affected functionality listed below.

## What Changed

**Modified files:**
$CODE_CHANGES

**Affected RPG feature nodes:**
$AFFECTED_NODES

**Callers of modified code (must be verified):**
$CALLERS

**All affected files:**
$AFFECTED_FILES

## Pre-Check Results

**pytest (affected tests):**
$PYTEST_STATUS

**smoke_test (imports/entry):**
$SMOKE_STATUS

## Your Workflow

### 1. Read the modified code
Read each modified file to understand what changed.

### 2. Run targeted tests
```bash
$PYTEST_CMD
```

### 3. Run smoke test
```bash
$SMOKE_TEST_CMD
```

### 4. Start the application and verify affected paths

$START_INSTRUCTIONS

For EACH caller listed above:
- Determine what user action triggers that caller
- Execute that action (HTTP request, CLI command, GUI interaction)
- Verify the result is correct

### 5. Visual Verification (MANDATORY for web/GUI projects)

**This step is NOT optional.** You MUST use the provided tools to visually
verify the project. Verifying only via curl/API is insufficient — real users
interact through the browser or GUI.

#### 5a. Inspect every affected page

For **web apps**, use `inspect` on EVERY affected route to capture
screenshots and saved HTML:
```bash
$BROWSER_TOOL inspect http://localhost:<PORT>/
$BROWSER_TOOL inspect http://localhost:<PORT>/<affected_route>
```
Read the saved HTML files to understand the full page content, CSS layout,
and element structure. Check for:
- Fixed pixel widths that should be responsive
- Elements overflowing or being cut off
- Broken layout at different conceptual viewport sizes
- Missing or misaligned visual elements

#### 5b. Simulate real user interactions

Don't just view pages — **interact** with them like a real user:
```bash
$BROWSER_TOOL run-script http://localhost:<PORT>/<page> --script '
page.click("a:has-text(\\"Some Link\\")")
page.wait_for_load_state("networkidle")
'
```
After each interaction, read the saved [After] HTML to verify the result.

For **GUI apps**, use the GUI tool:
```bash
$GUI_TOOL start-display
$GUI_TOOL launch "python main.py" --wait 3
$GUI_TOOL status
$GUI_TOOL screenshot
```
Click every relevant button, fill forms, and screenshot after each action.

#### 5c. Visual quality check

After inspecting pages / taking screenshots:
- Check that content renders correctly (not blank, not broken)
- Verify layout adapts properly (no horizontal scrollbar, no overflow)
- For style/CSS/layout changes: verify the visual result matches the intent
- If the visual result is poor (misaligned, cut off, ugly), this is a
  **FAIL** even if tests pass

### 6. Fix any issues found
If a test fails, functionality doesn't work, or **visual quality is poor**:
- Fix the code
- Re-run the failing test
- Re-inspect the affected pages to verify the visual fix
- Re-verify the affected path

### 7. Commit fixes (if any)
```bash
git add -A && git commit -m "review: fix issues found in impact review"
```

## Exit Protocol

After verifying ALL affected callers AND visual inspection, output your
result on the LAST line:

- `REVIEW_RESULT: PASS` — all affected functionality works AND looks correct
- `REVIEW_RESULT: FAIL | <reason>` — unfixable issues remain
- `REVIEW_RESULT: PASS_WITH_FIXES` — issues found and fixed

**Before the REVIEW_RESULT line**, if you noticed any related issues that
are **outside the scope of this plan** but worth addressing, list them
in a `SUGGESTIONS` block:

```
SUGGESTIONS:
- src/flask_blog/views/errors.py: still has max-width:600px hardcoded
- src/flask_blog/models/view_engine.py: .sidebar width:260px is fixed px
- (any other patterns you noticed while inspecting)
```

These will be shown to the user as follow-up recommendations.
Do NOT fix these — they are out of scope. Just report them.

$PREVIOUS_ISSUES

## Critical Rules
- Only verify functionality connected to the modified code — NOT all features
- Actually RUN the code — don't just read it
- **MUST use browser.py/gui.py tools** — curl alone is NOT sufficient
- For layout/style changes: visual inspection is the PRIMARY verification
- After taking a screenshot, check it shows meaningful content (not blank)
- Create test data through the project's own interfaces
- Kill background processes before finishing
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_test_files(code_changes: List[dict]) -> List[str]:
    """Derive test file patterns from code_changes.

    Uses directory context to build discriminating patterns.
    e.g. ``src/flask_blog/views/posts/misc.py`` → ``test_views_posts_misc``
    instead of the generic ``test_misc`` which matches nothing.
    """
    seen: set = set()
    patterns: List[str] = []
    for cc in code_changes:
        fp = cc.get("file_path", "")
        if not fp.endswith(".py"):
            continue
        p = Path(fp)
        stem = p.stem

        # Build a path-aware pattern: take up to 3 trailing path segments
        # e.g. views/posts/misc.py → test_views_posts_misc
        parts = list(p.with_suffix("").parts)
        # Drop __init__ — use parent directory name instead
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        # Strip common prefixes like "src", package name
        while parts and parts[0] in ("src", "lib"):
            parts.pop(0)
        if not parts:
            continue
        # Skip the top-level package dir (e.g. "flask_blog")
        if len(parts) > 1:
            parts = parts[1:]  # drop package name
        # Use last 3 segments max
        key_parts = parts[-3:] if len(parts) > 3 else parts
        pattern = "test_" + "_".join(key_parts)

        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    return patterns


def _format_code_changes(code_changes: List[dict]) -> str:
    lines = []
    for cc in code_changes:
        fp = cc.get("file_path", "?")
        ct = cc.get("change_type", "?")
        desc = cc.get("description", "")
        lines.append(f"- `{fp}` ({ct}): {desc}")
    return "\n".join(lines) or "(no code changes)"


def _format_callers(impact_results: dict) -> str:
    seen: set = set()
    lines: List[str] = []
    for node_id, data in impact_results.items():
        for caller in data.get("callers", []):
            nid = caller.get("node_id", "?")
            if nid in seen:
                continue
            seen.add(nid)
            name = caller.get("name", "?")
            lines.append(f"- `{name}` ({nid})")
    return "\n".join(lines) or "(no callers — isolated change)"


def _format_affected_files(impact_results: dict) -> str:
    files = set()
    for node_id, data in impact_results.items():
        files.update(data.get("affected_files", []))
    return "\n".join(f"- `{f}`" for f in sorted(files)) or "(none)"


def _format_affected_nodes(plan: dict) -> str:
    nodes = plan.get("affected_nodes", [])
    return "\n".join(f"- `{n}`" for n in nodes) or "(none)"


def _count_impact(impact_results: dict) -> Tuple[int, int]:
    """Return (unique_callers, affected_file_count)."""
    caller_ids: set = set()
    files: set = set()
    for data in impact_results.values():
        for c in data.get("callers", []):
            caller_ids.add(c.get("node_id") or c.get("name", ""))
        files.update(data.get("affected_files", []))
    return len(caller_ids), len(files)


def _parse_review_result(response: Optional[str]) -> Tuple[bool, str]:
    """Parse REVIEW_RESULT from sub-agent response."""
    if not response:
        return False, "No response from sub-agent"

    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if line.startswith("REVIEW_RESULT:"):
            result = line[len("REVIEW_RESULT:"):].strip()
            if result == "PASS" or result == "PASS_WITH_FIXES":
                return True, result
            elif result.startswith("FAIL"):
                return False, result
    return False, "REVIEW_RESULT not found in response"


def _parse_suggestions(response: Optional[str]) -> List[str]:
    """Extract SUGGESTIONS block from sub-agent response."""
    if not response:
        return []
    suggestions: List[str] = []
    in_block = False
    for line in response.splitlines():
        stripped = line.strip()
        if stripped == "SUGGESTIONS:":
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                suggestions.append(stripped[2:])
            elif stripped.startswith("```") or stripped.startswith("REVIEW_RESULT"):
                break
            elif not stripped:
                continue
            else:
                break
    return suggestions


def build_impact_review_prompt(
    plan: dict,
    impact_results: dict,
    pytest_status: str,
    smoke_status: str,
    previous_issues: str = "",
) -> str:
    """Build the impact-scoped review prompt."""
    code_changes = plan.get("code_changes", [])
    test_patterns = _derive_test_files(code_changes)

    pytest_cmd = "python3 -m pytest -x -q"
    if test_patterns:
        pattern = " or ".join(test_patterns)
        pytest_cmd += f' -k "{pattern}" --timeout=30'

    # Tool invocations route through the global ``cmind`` CLI (the
    # scripts no longer live in the workspace).  See ``cmind script``
    # in docs/cli-reference.md.
    browser_tool = cmd_for("tools/browser.py")
    gui_tool = cmd_for("tools/gui.py")
    smoke_test_cmd = f"{cmd_for('smoke_test.py')} --json"

    # Start instructions depend on project type
    start_instructions = (
        "Start the application in the background and verify it's running:\n"
        "```bash\n"
        "# Read main.py or app.py to find the start command\n"
        "python3 main.py &\n"
        "# Wait and verify\n"
        "sleep 2 && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/\n"
        "```\n"
        "Adjust the port based on what the app actually uses."
    )

    return IMPACT_REVIEW_PROMPT.safe_substitute(
        CODE_CHANGES=_format_code_changes(code_changes),
        AFFECTED_NODES=_format_affected_nodes(plan),
        CALLERS=_format_callers(impact_results),
        AFFECTED_FILES=_format_affected_files(impact_results),
        PYTEST_STATUS=pytest_status,
        SMOKE_STATUS=smoke_status,
        PYTEST_CMD=pytest_cmd,
        BROWSER_TOOL=browser_tool,
        GUI_TOOL=gui_tool,
        SMOKE_TEST_CMD=smoke_test_cmd,
        START_INSTRUCTIONS=start_instructions,
        PREVIOUS_ISSUES=previous_issues or "",
    )


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def impact_review(
    plan_path: Path,
    impact_path: Optional[Path],
    repo_path: Path,
    max_iterations: int = 3,
    timeout: int = 1200,
    report_scope: str = "final",
    report_dir: Optional[Path] = None,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run impact-scoped review with iterative repair."""
    from run_batch import dispatch_sub_agent
    from code_gen.test_runner import run_pytest
    from smoke_test import run_smoke_test

    # 1. Load data
    plan = json.loads(plan_path.read_text())

    if impact_path and impact_path.exists():
        impact_data = json.loads(impact_path.read_text())
        impact_results = impact_data.get("results", {})
    else:
        impact_results = {}
        logger.warning("No impact data provided, review scope may be incomplete")

    # 2. Pre-check: pytest on affected test files
    test_patterns = _derive_test_files(plan.get("code_changes", []))
    try:
        pre_pytest = run_pytest(
            repo_path,
            test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
            timeout=120,
            extra_args=["--timeout=30"],
        )
        pytest_status = (
            f"{'PASS' if pre_pytest.success else 'FAIL'}: "
            f"{pre_pytest.passed} passed, {pre_pytest.failed} failed, "
            f"{pre_pytest.errors} errors"
        )
    except Exception as e:
        pytest_status = f"ERROR: {e}"
        pre_pytest = None

    # 3. Pre-check: smoke_test
    try:
        smoke = run_smoke_test(repo_path)
        # ``run_smoke_test`` returns a ``SmokeResult`` dataclass, not a dict.
        smoke_dict = smoke.to_dict()
        # SmokeResult has no "summary" key; surface a compact per-layer
        # pass/fail map so the agent sees what actually failed.
        layer_summary = {
            name: bool(info.get("passed", False)) if isinstance(info, dict) else None
            for name, info in (smoke_dict.get("layers") or {}).items()
        }
        smoke_status = (
            f"{'PASS' if smoke_dict.get('success') else 'FAIL'}: "
            f"{json.dumps(layer_summary)}"
        )
    except Exception as e:
        smoke_status = f"ERROR: {e}"

    results: Dict[str, Any] = {
        "type": "impact_review",
        "iterations": [],
        "success": False,
        "total_duration": 0.0,
    }
    start_time = time.time()
    previous_issues = ""

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        logger.info("━━━ Impact Review: iteration %d/%d ━━━", iteration, max_iterations)

        # 4. Build prompt (re-compute pytest_status for iteration 2+
        #    so the sub-agent sees post-fix state, not stale pre-fix state)
        if iteration > 1:
            try:
                re_pytest = run_pytest(
                    repo_path,
                    test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
                    timeout=120,
                    extra_args=["--timeout=30"],
                )
                pytest_status = (
                    f"{'PASS' if re_pytest.success else 'FAIL'}: "
                    f"{re_pytest.passed} passed, {re_pytest.failed} failed, "
                    f"{re_pytest.errors} errors"
                )
            except Exception as e:
                pytest_status = f"ERROR: {e}"

        prompt = build_impact_review_prompt(
            plan, impact_results, pytest_status, smoke_status,
            previous_issues=(
                f"\n## Previous Issues (iteration {iteration - 1})\n{previous_issues}"
                if previous_issues else ""
            ),
        )

        # 5. Dispatch sub-agent
        response, error = dispatch_sub_agent(
            prompt, repo_path,
            timeout=timeout,
            purpose=f"impact_review_{iteration}",
            max_retries=2,
        )

        if error:
            results["iterations"].append({
                "iteration": iteration,
                "error": error,
            })
            logger.warning("Sub-agent error on iteration %d: %s", iteration, error[:120])
            continue

        # 6. Parse result
        passed, detail = _parse_review_result(response)
        suggestions = _parse_suggestions(response)

        # 7. Post-verify (independent — don't trust sub-agent)
        post_passed = True  # default: no relevant tests = not a failure
        try:
            post_pytest = run_pytest(
                repo_path,
                test_files=[f"*{p}*" for p in test_patterns] if test_patterns else None,
                timeout=120,
                extra_args=["--timeout=30"],
            )
            # 0 tests collected = no relevant tests exist → not a failure
            total = post_pytest.passed + post_pytest.failed + post_pytest.errors
            if total == 0:
                post_passed = True
            else:
                post_passed = post_pytest.success
        except Exception:
            post_passed = True  # pytest infra failure ≠ code failure

        iter_result = {
            "iteration": iteration,
            "agent_passed": passed,
            "agent_detail": detail,
            "post_pytest_passed": post_passed,
            "duration": time.time() - iter_start,
            "suggestions": suggestions,
        }
        results["iterations"].append(iter_result)

        # Early exit: agent says PASS and post-verify agrees
        if passed and post_passed:
            results["success"] = True
            break

        # Extract issues for next iteration
        if response:
            # Take last 2000 chars as context for next iteration
            previous_issues = response[-2000:]

    results["total_duration"] = time.time() - start_time
    # Aggregate suggestions from all iterations (deduplicated)
    all_suggestions: List[str] = []
    seen_suggestions: set = set()
    for it in results["iterations"]:
        for s in it.get("suggestions", []):
            if s not in seen_suggestions:
                seen_suggestions.add(s)
                all_suggestions.append(s)
    if all_suggestions:
        results["suggestions"] = all_suggestions
    return _publish_review_report(
        results,
        plan_path,
        impact_path,
        report_scope=report_scope,
        report_dir=report_dir,
        parent_run_id=parent_run_id,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Impact-scoped review for rpg_edit changes"
    )
    parser.add_argument("--plan", type=Path, default=RPG_EDIT_PLAN_FILE,
                        help="Path to rpg_edit_plan.json (default: %(default)s)")
    parser.add_argument("--impact", type=Path, default=RPG_EDIT_IMPACT_FILE,
                        help="Path to rpg_edit_impact.json (default: %(default)s)")
    parser.add_argument("--repo", type=Path, default=None,
                        help="Repository root path")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum review+repair iterations (default: 3)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Sub-agent timeout per iteration in seconds (default: 1200)")
    parser.add_argument("--report-scope", choices=sorted(_REPORT_SCOPES), default="final",
                        help="HTML report publication scope (default: %(default)s)")
    parser.add_argument("--no-report", action="store_true",
                        help="Persist review JSON without writing an HTML report")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Base report directory (default: .cmind/reports)")
    parser.add_argument("--parent-run-id", default=None,
                        help="Parent command run ID to record in report evidence")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()
    report_scope = "none" if args.no_report else args.report_scope

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    if not args.plan.exists():
        result = {"type": "error", "message": f"Plan not found: {args.plan}"}
        result = _publish_review_report(
            result,
            args.plan,
            args.impact,
            report_scope=report_scope,
            report_dir=args.report_dir,
            parent_run_id=args.parent_run_id,
        )
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    # Resolve repo path: workspace root is the project repo root.
    # ``--repo`` override stays for tests / brownfield setups.
    repo_path = args.repo or REPO_DIR

    # Check if review is needed based on impact scale
    if args.impact and args.impact.exists():
        impact_data = json.loads(args.impact.read_text())
        impact_results = impact_data.get("results", {})
        total_callers, affected_files = _count_impact(impact_results)

        if total_callers == 0 and affected_files <= 1:
            result = {
                "type": "skipped",
                "success": True,
                "reason": f"Impact too small for sub-agent review "
                          f"(callers={total_callers}, files={affected_files}). "
                          f"Agent self-review is sufficient.",
            }
            result = _publish_review_report(
                result,
                args.plan,
                args.impact,
                report_scope=report_scope,
                report_dir=args.report_dir,
                parent_run_id=args.parent_run_id,
            )
            print(json.dumps(result, indent=2) if args.json else
                  f"Skipped: {result['reason']}\nReport: {result.get('report_path', '')}")
            return 0

    result = impact_review(
        plan_path=args.plan,
        impact_path=args.impact,
        repo_path=repo_path,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
        report_scope=report_scope,
        report_dir=args.report_dir,
        parent_run_id=args.parent_run_id,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Review {'PASSED' if result['success'] else 'FAILED'} "
            f"({len(result['iterations'])} iterations, "
            f"{result['total_duration']:.1f}s)"
        )
        if result.get("report_path"):
            print(f"Report: {result['report_path']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
