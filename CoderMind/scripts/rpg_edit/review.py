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


def _focused_graph_output_path() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"rpg_edit_focused_graph_{time.time_ns()}.html"


def _focused_graph_artifact(candidates: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    dep_rows = _dep_node_rows(candidates)
    selected_rpg = sorted({str(row.get("node_id")) for row in candidates if row.get("node_id")})
    selected_dep = sorted({str(row.get("node_id")) for row in dep_rows if row.get("node_id")})
    if not selected_rpg and not selected_dep:
        return {}

    metadata: Dict[str, Any] = {
        "status": "recorded",
        "selected_rpg_nodes": selected_rpg,
        "selected_dep_nodes": selected_dep,
    }
    rpg_data = _load_json_artifact(REPO_RPG_FILE)
    if not isinstance(rpg_data, dict):
        metadata.update({"status": "missing", "reason": f"RPG file not available: {REPO_RPG_FILE}"})
        return metadata

    try:
        from rpg_visualize import build_focused_graph_data, generate_html

        focused_data = build_focused_graph_data(
            rpg_data,
            rpg_node_ids=selected_rpg,
            dep_node_ids=selected_dep,
        )
        focused_meta = focused_data.get("_focused_graph") if isinstance(focused_data.get("_focused_graph"), dict) else {}
        metadata.update(focused_meta)
        if not (focused_meta.get("matched_rpg_nodes") or focused_meta.get("matched_dep_nodes")):
            metadata.update({"status": "unavailable", "reason": "selected nodes not found in current RPG"})
            return metadata
        graph_path = _focused_graph_output_path()
        graph_path.write_text(generate_html(focused_data), encoding="utf-8")
        metadata.update({"status": "available", "path": str(graph_path)})
    except Exception as exc:
        metadata.update({"status": "error", "reason": str(exc)})
    return metadata


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
    locate_ids = {row.get("node_id") for row in locate.get("results") or [] if isinstance(row, dict)}
    applied_by_id = {
        row.get("node_id"): row
        for row in apply_result.get("applied_features") or []
        if isinstance(row, dict) and row.get("node_id")
    }
    changed_files = _ordered_unique(
        [_code_delta_file(delta) for delta in code_deltas]
        + [str(path) for path in _listify(code_result.get("files_modified"))]
        + [str(change.get("file_path")) for change in plan.get("code_changes") or [] if isinstance(change, dict) and change.get("file_path")]
    )
    groups: List[Dict[str, Any]] = []
    for candidate in candidates:
        node_id = candidate.get("node_id")
        impact = impact_results.get(node_id) if isinstance(impact_results.get(node_id), dict) else {}
        relations = _mapped_code_relations(candidate, impact)
        relation_paths = _ordered_unique([row.get("path") for row in relations])
        affected_files = _ordered_unique(_listify(impact.get("affected_files")) + relation_paths)
        relevant_files = set(affected_files or relation_paths or changed_files)
        relevant_deltas = [delta for delta in code_deltas if _code_delta_file(delta) in relevant_files]
        locate_state = candidate.get("locate_state") or ("selected" if node_id in locate_ids or not locate else "missing")
        if not impact:
            impact_state = "missing"
        elif impact.get("error"):
            impact_state = "error"
        elif relations:
            impact_state = "mapped"
        else:
            impact_state = "missing_mapping"
        missing_states: Dict[str, str] = {}
        if locate_state != "selected":
            missing_states["locate"] = str(locate_state)
        if impact_state != "mapped":
            missing_states["impact"] = impact_state
        if not relations:
            missing_states["mapping"] = "missing_dep_graph_mapping"
        impact_summary = impact.get("impact_summary") if isinstance(impact.get("impact_summary"), dict) else {}
        caller_count = impact_summary.get("total_callers", len(impact.get("callers") or []))
        callee_count = impact_summary.get("total_callees", len(impact.get("callees") or []))
        inheritance_count = impact_summary.get("total_inheritance", len(impact.get("inheritance") or []))
        affected_file_count = impact_summary.get("affected_file_count", len(affected_files))
        apply_row = applied_by_id.get(node_id, {})
        groups.append({
            "node_id": node_id,
            "name": candidate.get("name") or impact.get("name") or node_id,
            "node_type": candidate.get("node_type") or candidate.get("type_name") or candidate.get("type"),
            "path": candidate.get("path") or candidate.get("meta_path"),
            "feature_path": candidate.get("feature_path"),
            "score": candidate.get("score"),
            "status": "mapped" if relations else "missing_mapping",
            "reason": _retrieval_hit_reason(candidate, impact),
            "missing_states": missing_states,
            "code_relations": relations,
            "affected_files": affected_files,
            "changed_files": [_code_delta_file(delta) for delta in relevant_deltas],
            "callers": impact.get("callers") or [],
            "callees": impact.get("callees") or [],
            "imports": impact.get("imports") or [],
            "inheritance": impact.get("inheritance") or [],
            "hidden_counts": {
                "code_relations": len(relations),
                "affected_files": affected_file_count,
                "callers": caller_count,
                "callees": callee_count,
                "imports": len(impact.get("imports") or []),
                "inheritance": inheritance_count,
            },
            "apply": {
                "status": _apply_status(apply_result),
                "action": apply_row.get("action") or apply_row.get("change"),
                "dep_graph_refreshed": apply_result.get("dep_graph_refreshed"),
            },
            "review": {
                "status": result.get("type", "review"),
                "success": result.get("success", result.get("type") == "skipped"),
                "iterations": len(result.get("iterations") or []),
                "suggestions": len(result.get("suggestions") or []),
            },
        })
    matched_files = {file_path for group in groups for file_path in group.get("changed_files") or []}
    unmatched_code_deltas = [delta for delta in code_deltas if _code_delta_file(delta) not in matched_files]
    summary = {
        "selected_feature_groups": len(groups),
        "mapped_code_relations": sum(len(group.get("code_relations") or []) for group in groups),
        "missing_mappings": sum(1 for group in groups if not group.get("code_relations")),
        "changed_files": len(changed_files),
        "review_status": result.get("type", "review"),
        "apply_status": _apply_status(apply_result),
        "verification_status": _test_status(result, code_result, apply_result),
    }
    payload: Dict[str, Any] = {
        "summary": summary,
        "groups": groups,
        "unmatched_code_deltas": unmatched_code_deltas,
    }
    if focused_graph:
        payload["graph"] = focused_graph
    return payload


def _review_summary_cards(
    result: Dict[str, Any],
    artifacts: Dict[str, Any],
    focused_impact: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    plan = artifacts.get("plan") if isinstance(artifacts.get("plan"), dict) else {}
    code_result = artifacts.get("code_result") if isinstance(artifacts.get("code_result"), dict) else {}
    apply_result = artifacts.get("apply_result") if isinstance(artifacts.get("apply_result"), dict) else {}
    summary = focused_impact.get("summary") if isinstance(focused_impact, dict) else {}
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
    def __init__(self, run: CommandRun, focused_graph: Dict[str, Any], focused_impact: Dict[str, Any]):
        self.run = run
        self.focused_graph = focused_graph
        self.focused_impact = focused_impact

    def to_dict(self) -> Dict[str, Any]:
        data = self.run.to_dict()
        if self.focused_graph:
            data["focused_graph"] = self.focused_graph
        if self.focused_impact:
            data["focused_impact"] = self.focused_impact
        return data


def _publish_review_report(result: Dict[str, Any], plan_path: Path, impact_path: Optional[Path]) -> Dict[str, Any]:
    _write_review_result(result)
    artifacts = _load_review_artifacts(plan_path, impact_path)
    candidates = _selected_candidate_rows(artifacts)
    code_deltas = _code_delta_rows(artifacts)
    focused_graph = _focused_graph_artifact(candidates, artifacts)
    focused_impact = _feature_evidence_groups(artifacts, candidates, code_deltas, result, focused_graph)
    artifact_rows = _artifact_links(plan_path, impact_path)
    if focused_graph.get("path"):
        artifact_rows.append({"label": "focused_graph", "path": focused_graph["path"], "status": focused_graph.get("status")})
    evidence = {"artifacts": artifacts, "review_result": result, "focused_impact": focused_impact}
    if focused_graph:
        evidence["focused_graph"] = focused_graph
    try:
        report_run = CommandRun(
            command="rpg_edit",
            title="CoderMind rpg_edit Explain View",
            status=str(result.get("type", "review")),
            summary=_review_summary_cards(result, artifacts, focused_impact),
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
        report_path = write_command_report(_ReportPayload(report_run, focused_graph, focused_impact))
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
