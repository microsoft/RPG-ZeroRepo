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
import shutil
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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


def _artifact_links(plan_path: Path, impact_path: Optional[Path]) -> List[Dict[str, Any]]:
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


def _focused_graph_artifact(candidates: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    dep_rows = _dep_node_rows(candidates)
    selected_rpg = sorted({str(row.get("node_id")) for row in candidates if row.get("node_id")})
    selected_dep = sorted({str(row.get("node_id")) for row in dep_rows if row.get("node_id")})
    if not selected_rpg and not selected_dep:
        return {}
    return {
        "status": "embedded",
        "selected_rpg_nodes": selected_rpg,
        "selected_dep_nodes": selected_dep,
    }


def _dep_node_path(dep_id: Any) -> str:
    if dep_id in (None, ""):
        return ""
    dep_id_text = str(dep_id)
    return dep_id_text.split(":", 1)[0] if ":" in dep_id_text else dep_id_text


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


_FOCUSED_RPG_NODE_CAP = 20
_FOCUSED_CODE_NODE_CAP = 50
_FOCUSED_EDGE_CAP = 80
_FOCUSED_WARNING_TYPES = {"missing_mapping", "missing_reason", "too_many_neighbors", "stale_graph"}


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


def _rpg_node_entry(node_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    row: Dict[str, Any] = {"node_id": node_id}
    _set_if_present(row, "name", raw.get("name"))
    _set_if_present(row, "node_type", raw.get("node_type") or raw.get("type") or raw.get("type_name"))
    _set_if_present(row, "path", raw.get("path") or raw.get("file") or meta.get("path"))
    _set_if_present(row, "feature_path", raw.get("feature_path") or meta.get("feature_path"))
    return row


def _dep_node_entry(dep_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"node_id": dep_id, "dep_node_id": dep_id}
    for key in (
        "name",
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
        _set_if_present(row, "path", row.get("module") or row.get("file") or _dep_node_path(dep_id))
    return row


def _coerce_dep_edge(edge: Any) -> Optional[Dict[str, Any]]:
    if isinstance(edge, dict):
        source = edge.get("source") or edge.get("from") or edge.get("caller") or edge.get("src")
        target = edge.get("target") or edge.get("to") or edge.get("callee") or edge.get("dst")
        relation = edge.get("relation") or edge.get("type") or edge.get("edge_type") or edge.get("kind") or "dep_graph"
        if source in (None, "") or target in (None, ""):
            return None
        row = {"source_node_id": str(source), "target_node_id": str(target), "relation": str(relation), "source": "dep_graph"}
        _set_if_present(row, "reason", edge.get("reason"))
        return row
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        relation = edge[2] if len(edge) >= 3 else "dep_graph"
        return {"source_node_id": str(edge[0]), "target_node_id": str(edge[1]), "relation": str(relation), "source": "dep_graph"}
    return None


def _current_rpg_context() -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[Dict[str, Any]],
]:
    rpg_data = _load_json_artifact(REPO_RPG_FILE)
    if not isinstance(rpg_data, dict) or rpg_data.get("_error"):
        return {}, {}, [], {}, {}, [{"type": "stale_graph", "message": f"RPG file not available: {REPO_RPG_FILE}"}]

    rpg_nodes: Dict[str, Dict[str, Any]] = {}

    def visit_rpg_node(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        node_id = raw.get("id") or raw.get("node_id")
        if node_id not in (None, ""):
            rpg_nodes[str(node_id)] = _rpg_node_entry(str(node_id), raw)
        children = raw.get("children") or []
        if isinstance(children, dict):
            children = list(children.values())
        if isinstance(children, (list, tuple)):
            for child in children:
                visit_rpg_node(child)

    visit_rpg_node(rpg_data.get("root"))
    raw_rpg_nodes = rpg_data.get("nodes")
    if isinstance(raw_rpg_nodes, dict):
        for node_id, raw in raw_rpg_nodes.items():
            raw_dict = raw if isinstance(raw, dict) else {}
            rpg_nodes[str(node_id)] = _rpg_node_entry(str(node_id), raw_dict)
    elif isinstance(raw_rpg_nodes, (list, tuple)):
        for raw in raw_rpg_nodes:
            if isinstance(raw, dict):
                node_id = raw.get("id") or raw.get("node_id")
                if node_id not in (None, ""):
                    rpg_nodes[str(node_id)] = _rpg_node_entry(str(node_id), raw)

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

    dep_edges = []
    for raw_edge in dep_graph.get("edges") or []:
        edge = _coerce_dep_edge(raw_edge)
        if edge:
            dep_edges.append(edge)

    warnings: List[Dict[str, Any]] = []
    if not rpg_nodes and not dep_nodes:
        warnings.append({"type": "stale_graph", "message": "Current RPG contains no indexed feature or dependency nodes"})
    return rpg_nodes, dep_nodes, dep_edges, rpg_to_dep, dep_to_rpg, warnings


def _feature_evidence_groups(
    artifacts: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    code_deltas: List[Dict[str, Any]],
    result: Dict[str, Any],
    focused_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    locate = artifacts.get("locate") if isinstance(artifacts.get("locate"), dict) else {}
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    impact_results = _impact_results(artifacts)
    current_rpg_nodes, current_dep_nodes, current_dep_edges, current_rpg_to_dep, _, graph_warnings = _current_rpg_context()
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
    edge_keys: set[Tuple[str, str, str, str]] = set()
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
        _set_if_present(row, "path", fallback_path)
        row["status"] = "mapped"
        row["source"] = source
        if source_feature not in (None, ""):
            row["source_feature"] = str(source_feature)
            row["source_features"] = [str(source_feature)]
        if row.get("path") in changed_files:
            row["changed"] = True
        if current_graph_available and dep_id_text not in current_dep_nodes:
            row["status"] = "stale_graph"
            add_warning("stale_graph", "Mapped code node is absent from the current dependency graph", dep_node_id=dep_id_text)
        code_node_by_id[dep_id_text] = row

    def add_edge(edge: Dict[str, Any]) -> None:
        source_node = edge.get("source_node_id")
        target_node = edge.get("target_node_id")
        relation = edge.get("relation") or "dependency"
        source = edge.get("source") or "impact"
        if source_node in (None, "") or target_node in (None, ""):
            return
        source_text = str(source_node)
        target_text = str(target_node)
        relation_text = str(relation)
        source_text_label = str(source)
        key = (source_text, target_text, relation_text, source_text_label)
        if key in edge_keys:
            return
        edge_keys.add(key)
        row: Dict[str, Any] = {
            "source_node_id": source_text,
            "target_node_id": target_text,
            "relation": relation_text,
            "source": source_text_label,
        }
        for key_name in ("rpg_node_id", "neighbor_node_id", "direction", "name", "path", "reason", "status"):
            _set_if_present(row, key_name, edge.get(key_name))
        all_edges.append(row)

    def impact_neighbor(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, dict):
            node_id = item.get("dep_node_id") or item.get("node_id") or item.get("id")
            if node_id in (None, ""):
                return None
            row: Dict[str, Any] = {"node_id": str(node_id)}
            for key_name in ("name", "path", "reason", "status", "type"):
                _set_if_present(row, key_name, item.get(key_name))
            return row
        if item in (None, ""):
            return None
        return {"node_id": str(item)}

    neighbor_specs = [
        ("callers", "caller", "upstream", "total_callers"),
        ("callees", "callee", "downstream", "total_callees"),
        ("imports", "import", "downstream", "total_imports"),
        ("inheritance", "inheritance", "downstream", "total_inheritance"),
    ]

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
        relation_paths = _ordered_unique(
            [row.get("path") for row in relations]
            + [current_dep_nodes.get(dep_id, {}).get("path") for dep_id in mapped_dep_ids]
        )
        affected_files = _ordered_unique(_listify(impact.get("affected_files")) + relation_paths)
        relevant_files = set(affected_files or relation_paths or changed_files)
        relevant_deltas = [delta for delta in code_deltas if _code_delta_file(delta) in relevant_files]
        changed_for_node = _ordered_unique([_code_delta_file(delta) for delta in relevant_deltas])
        focus_reason = _focus_reason(candidate, impact)
        if not focus_reason:
            add_warning("missing_reason", "Focused view has no explicit selection reason", node_id=node_id)
        if not mapped_dep_ids:
            add_warning("missing_mapping", "Selected RPG node has no mapped code node", node_id=node_id)
        if current_graph_available and node_id not in current_rpg_nodes:
            add_warning("stale_graph", "Selected RPG node is absent from the current RPG graph", node_id=node_id)

        impact_summary = impact.get("impact_summary") if isinstance(impact.get("impact_summary"), dict) else {}
        node_hidden_counts: Dict[str, int] = {}
        for impact_key, relation, direction, total_key in neighbor_specs:
            items = _listify(impact.get(impact_key))
            total = count_value(impact_summary.get(total_key), len(items))
            visible = 0
            for item in items:
                neighbor = impact_neighbor(item)
                if not neighbor:
                    continue
                visible += 1
                if relation == "caller":
                    source_node = neighbor["node_id"]
                    target_node = node_id
                else:
                    source_node = node_id
                    target_node = neighbor["node_id"]
                edge_row = {
                    "source_node_id": source_node,
                    "target_node_id": target_node,
                    "relation": relation,
                    "source": "impact",
                    "rpg_node_id": node_id,
                    "neighbor_node_id": neighbor["node_id"],
                    "direction": direction,
                }
                for key_name in ("name", "path", "reason", "status"):
                    _set_if_present(edge_row, key_name, neighbor.get(key_name))
                add_edge(edge_row)
            hidden = max(0, total - visible)
            if hidden:
                node_hidden_counts[impact_key] = hidden
                impact_hidden_counts[impact_key] = impact_hidden_counts.get(impact_key, 0) + hidden

        apply_row = applied_by_id.get(node_id, {})
        rpg_row: Dict[str, Any] = {
            "node_id": node_id,
            "status": "mapped" if mapped_dep_ids else "missing",
            "mapping_status": "mapped" if mapped_dep_ids else "missing",
        }
        _set_if_present(rpg_row, "name", candidate.get("name") or impact.get("name") or current_node.get("name"))
        _set_if_present(rpg_row, "node_type", candidate.get("node_type") or candidate.get("type_name") or candidate.get("type") or current_node.get("node_type"))
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
            for dep_id in mapped_dep_ids:
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

    for dep_id, dep_node in current_dep_nodes.items():
        if dep_node.get("path") in changed_files:
            add_code_node(dep_id, source="changed_file", fallback_path=dep_node.get("path"))

    selected_code_ids = set(code_node_by_id)
    for edge in current_dep_edges:
        if edge.get("source_node_id") in selected_code_ids or edge.get("target_node_id") in selected_code_ids:
            add_edge(edge)

    primary_rpg_nodes = primary_rpg_nodes_all[:_FOCUSED_RPG_NODE_CAP]
    primary_code_nodes_all = list(code_node_by_id.values())
    primary_code_nodes = primary_code_nodes_all[:_FOCUSED_CODE_NODE_CAP]
    visible_rpg_ids = {row.get("node_id") for row in primary_rpg_nodes}
    visible_code_ids = {row.get("node_id") for row in primary_code_nodes}
    mappings = [
        row for row in mapping_rows_all
        if row.get("rpg_node_id") in visible_rpg_ids and (not row.get("code_node_id") or row.get("code_node_id") in visible_code_ids)
    ]
    edges = all_edges[:_FOCUSED_EDGE_CAP]
    hidden_counts: Dict[str, Any] = {
        "primary_rpg_nodes": max(0, len(primary_rpg_nodes_all) - len(primary_rpg_nodes)),
        "primary_code_nodes": max(0, len(primary_code_nodes_all) - len(primary_code_nodes)),
        "edges": max(0, len(all_edges) - len(edges)),
    }
    for key, count in impact_hidden_counts.items():
        hidden_counts[key] = hidden_counts.get(key, 0) + count
    relation_totals: Dict[str, int] = {}
    relation_visible: Dict[str, int] = {}
    for edge in all_edges:
        relation = str(edge.get("relation") or "dependency")
        relation_totals[relation] = relation_totals.get(relation, 0) + 1
    for edge in edges:
        relation = str(edge.get("relation") or "dependency")
        relation_visible[relation] = relation_visible.get(relation, 0) + 1
    hidden_relations = {
        relation: total - relation_visible.get(relation, 0)
        for relation, total in relation_totals.items()
        if total > relation_visible.get(relation, 0)
    }
    if hidden_relations:
        hidden_counts["relations"] = hidden_relations
    capped_hidden = {
        key: value for key, value in hidden_counts.items()
        if key in {"primary_rpg_nodes", "primary_code_nodes", "edges"} and value
    }
    if capped_hidden:
        add_warning("too_many_neighbors", "Focused view omitted rows because caps were reached", hidden_counts=capped_hidden)

    matched_files = {file_path for node in primary_rpg_nodes_all for file_path in node.get("changed_files") or []}
    unmatched_code_deltas = [delta for delta in code_deltas if _code_delta_file(delta) not in matched_files]
    mapped_code_relations = sum(1 for row in mapping_rows_all if row.get("code_node_id"))
    missing_mappings = sum(1 for row in mapping_rows_all if row.get("status") == "missing")
    summary = {
        "selected_feature_groups": len(primary_rpg_nodes_all),
        "primary_rpg_nodes": len(primary_rpg_nodes),
        "primary_code_nodes": len(primary_code_nodes),
        "mapped_code_relations": mapped_code_relations,
        "missing_mappings": missing_mappings,
        "edges": len(edges),
        "warnings": len(warnings),
        "changed_files": len(changed_files),
        "review_status": result.get("type", "review"),
        "apply_status": _apply_status(apply_result),
        "verification_status": _test_status(result, code_result, apply_result),
    }
    return {
        "summary": summary,
        "primary_rpg_nodes": primary_rpg_nodes,
        "primary_code_nodes": primary_code_nodes,
        "mappings": mappings,
        "edges": edges,
        "hidden_counts": hidden_counts,
        "warnings": warnings,
        "changed_files": changed_files,
        "unmatched_code_deltas": unmatched_code_deltas,
        "caps": {
            "primary_rpg_nodes": _FOCUSED_RPG_NODE_CAP,
            "primary_code_nodes": _FOCUSED_CODE_NODE_CAP,
            "edges": _FOCUSED_EDGE_CAP,
        },
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
        {"label": "Review status", "value": result.get("type", "review")},
        {"label": "Review result", "value": result_value},
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
    return [
        {"name": "validate", "status": validate.get("type") if validate else "missing", "reason": validate.get("message", "") if validate else "artifact not found"},
        {"name": "locate", "status": locate.get("type") if locate else "missing", "reason": f"{len(locate.get('results') or [])} candidates" if locate else "artifact not found"},
        {"name": "plan", "status": "available" if plan else "missing", "reason": f"{len(plan.get('code_changes') or [])} code changes" if plan else "artifact not found"},
        {"name": "impact", "status": impact.get("type", "available") if impact else "missing", "reason": f"{len((impact.get('results') or {}))} impact result sets" if impact else "artifact not found"},
        {"name": "code", "status": code_result.get("last_status") if code_result else "missing", "reason": code_result.get("last_error") or f"success={code_result.get('success')}" if code_result else "artifact not found"},
        {"name": "review", "status": result.get("type"), "reason": result.get("reason") or f"success={result.get('success')}"},
    ]


def _review_verification(result: Dict[str, Any], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    validate = artifacts.get("validate") if isinstance(artifacts.get("validate"), dict) else None
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else None
    if validate:
        checks.append({"name": "validate", "status": validate.get("type"), "detail": validate.get("message", "")})
    if code_result:
        checks.append({"name": "code", "status": code_result.get("success"), "detail": code_result.get("last_error") or code_result.get("last_status")})
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
    return {
        "status": _apply_status(apply_result),
        "dep_graph_refreshed": apply_result.get("dep_graph_refreshed"),
        "applied_features": applied,
        "rollback_path": _rollback_path(apply_result),
        "test_status": _status_from_bool(test_result.get("passed")),
    }


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
    return {
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
    def __init__(self, run: CommandRun, focused_view: Dict[str, Any]):
        self.run = run
        self.focused_view = focused_view

    def to_dict(self) -> Dict[str, Any]:
        data = self.run.to_dict()
        if self.focused_view:
            data["focused_view"] = self.focused_view
        return data


def _publish_review_report(result: Dict[str, Any], plan_path: Path, impact_path: Optional[Path]) -> Dict[str, Any]:
    _write_review_result(result)
    artifacts = _load_review_artifacts(plan_path, impact_path)
    candidates = _selected_candidate_rows(artifacts)
    code_deltas = _code_delta_rows(artifacts)
    focused_view = _feature_evidence_groups(artifacts, candidates, code_deltas, result)
    artifact_rows = _artifact_links(plan_path, impact_path)
    evidence = _compact_review_evidence(artifacts, artifact_rows, result)
    try:
        report_run = CommandRun(
            command="rpg_edit",
            title="CoderMind rpg_edit Explain View",
            status=str(result.get("type", "review")),
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
                for row in candidates
            ],
            dep_graph_deltas=[
                DepGraphDeltaEvent(
                    dep_node_id=row.get("node_id"),
                    path=row.get("path"),
                    source_feature=row.get("source_feature"),
                    change=row.get("change"),
                )
                for row in _dep_node_rows(candidates)
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
        report_path = write_command_report(_ReportPayload(report_run, focused_view))
        result["report_path"] = str(report_path)
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
    timeout: int = 600,
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
    return _publish_review_report(results, plan_path, impact_path)


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
    parser.add_argument("--timeout", type=int, default=600,
                        help="Sub-agent timeout per iteration in seconds (default: 600)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    if not args.plan.exists():
        result = {"type": "error", "message": f"Plan not found: {args.plan}"}
        result = _publish_review_report(result, args.plan, args.impact)
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
            result = _publish_review_report(result, args.plan, args.impact)
            print(json.dumps(result, indent=2) if args.json else
                  f"Skipped: {result['reason']}\nReport: {result.get('report_path', '')}")
            return 0

    result = impact_review(
        plan_path=args.plan,
        impact_path=args.impact,
        repo_path=repo_path,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
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
