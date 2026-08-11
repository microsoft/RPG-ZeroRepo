"""Generate the static CoderMind dashboard report from a validated snapshot."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.paths import REPORTS_DIR
from rpg_visualize import generate_html

_ASSET_SOURCE_DIR = Path(__file__).resolve().parent / "report_assets"


@dataclass(frozen=True)
class DashboardReportOutputs:
    report_html: Path
    report_data_js: Path
    rpg_html: Path
    report_css: Path
    report_js: Path
    history_index_js: Path
    history_detail_files: tuple[Path, ...]
    d3_js: Path | None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _asset(name: str) -> str:
    return (_ASSET_SOURCE_DIR / name).read_text(encoding="utf-8")


def _d3_source() -> Path | None:
    configured = os.environ.get("CMIND_D3_PATH")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(_ASSET_SOURCE_DIR / "d3.v7.min.js")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _rpg_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    graph = snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {}
    rpg = snapshot.get("rpg") if isinstance(snapshot.get("rpg"), dict) else {}
    workspace = snapshot.get("workspace") if isinstance(snapshot.get("workspace"), dict) else {}
    repo_name = rpg.get("repo_name") or workspace.get("name") or "repository"
    root = graph.get("feature_root")
    if not isinstance(root, dict):
        root = {
            "id": "__empty__",
            "name": repo_name,
            "node_type": "repo",
            "children": [],
        }
    return {
        "repo_name": repo_name,
        "root": root,
        "edges": graph.get("semantic_edges") if isinstance(graph.get("semantic_edges"), list) else [],
        "dep_graph": graph.get("dependency_graph") if isinstance(graph.get("dependency_graph"), dict) else {},
        "_dep_to_rpg_map": graph.get("dep_to_rpg_map") if isinstance(graph.get("dep_to_rpg_map"), dict) else {},
    }


def _history_summary(root: dict[str, Any], detail_path: str) -> dict[str, Any]:
    keys = (
        "span_id", "trace_id", "kind", "logical_key", "name", "status",
        "started_at", "finished_at", "duration_ms", "trigger", "attempt",
        "sequence", "quality", "source", "error", "recovery", "recovered_attempts",
    )
    summary = {key: root.get(key) for key in keys if root.get(key) is not None}
    if isinstance(root.get("metrics"), dict):
        summary["metrics"] = root["metrics"]
    if isinstance(root.get("details"), dict):
        summary["details"] = root["details"]
    all_children = root.get("children") if isinstance(root.get("children"), list) else []
    children = [
        child
        for child in all_children
        if isinstance(child, dict)
        and (
            child.get("kind") not in {
                "artifact.write", "command.script", "tool.llm", "tool.script", "report.snapshot",
            }
            or (child.get("details") or {}).get("grouped_as") == "rpg_edit_check"
        )
    ]
    summary_children = []
    for child in children:
        child_summary = {
            key: child.get(key)
            for key in keys
            if child.get(key) is not None
        }
        if isinstance(child.get("metrics"), dict):
            child_summary["metrics"] = child["metrics"]
        if isinstance(child.get("details"), dict):
            child_summary["details"] = child["details"]
        summary_children.append(child_summary)
    summary["children"] = summary_children
    summary["child_count"] = len(children)
    summary["evidence_count"] = len(all_children) - len(children)
    summary["detail_path"] = detail_path
    return summary


def _history_files(snapshot: dict[str, Any], target: Path) -> tuple[str, dict[Path, str], dict[str, Any]]:
    history = snapshot.get("history") if isinstance(snapshot.get("history"), dict) else {}
    roots = history.get("roots") if isinstance(history.get("roots"), list) else []
    details: dict[Path, str] = {}
    summaries: list[dict[str, Any]] = []
    for root in roots:
        if not isinstance(root, dict):
            continue
        identity = f"{root.get('trace_id') or ''}:{root.get('span_id') or ''}"
        filename = hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".js"
        detail_path = f"history/{filename}"
        summaries.append(_history_summary(root, detail_path))
        root_id = json.dumps(str(root.get("span_id") or identity), ensure_ascii=True)
        detail_payload = json.dumps(root, ensure_ascii=False, separators=(",", ":"))
        details[target / detail_path] = (
            "window.CMIND_HISTORY_DETAILS = window.CMIND_HISTORY_DETAILS || {};\n"
            f"window.CMIND_HISTORY_DETAILS[{root_id}] = {detail_payload};\n"
        )
    index = {
        "schema_version": history.get("schema_version", 1),
        "generated_at": history.get("generated_at"),
        "retention": history.get("retention") if isinstance(history.get("retention"), dict) else {},
        "summary": history.get("summary") if isinstance(history.get("summary"), dict) else {},
        "roots": summaries,
    }
    index_payload = (
        "// Generated by CoderMind; do not edit by hand.\n"
        "window.CMIND_HISTORY_DETAILS = window.CMIND_HISTORY_DETAILS || {};\n"
        f"window.CMIND_HISTORY_INDEX = {json.dumps(index, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    compact_history = {key: value for key, value in index.items() if key != "roots"}
    compact_history["available"] = bool(summaries)
    return index_payload, details, compact_history


def write_dashboard_report(
    snapshot: dict[str, Any],
    reports_dir: Path | None = None,
) -> DashboardReportOutputs:
    """Write the complete file://-openable report and return its output paths."""
    target = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    assets_dir = target / "assets"
    d3_source = _d3_source()
    outputs = DashboardReportOutputs(
        report_html=target / "report.html",
        report_data_js=target / "report-data.js",
        rpg_html=target / "rpg.html",
        report_css=assets_dir / "report.css",
        report_js=assets_dir / "report.js",
        history_index_js=target / "history-index.js",
        history_detail_files=(),
        d3_js=assets_dir / "d3.v7.min.js" if d3_source is not None else None,
    )

    change_data = snapshot.get("rpg_latest_change")
    if not isinstance(change_data, dict) or not change_data.get("available"):
        change_data = None
    rpg_html = generate_html(_rpg_input(snapshot), change_data)
    if d3_source is not None:
        rpg_html = rpg_html.replace("https://d3js.org/d3.v7.min.js", "assets/d3.v7.min.js")
    history_index, history_details, compact_history = _history_files(snapshot, target)
    report_snapshot = dict(snapshot)
    report_snapshot["history"] = compact_history
    payload = json.dumps(report_snapshot, ensure_ascii=False, separators=(",", ":"))
    embedded_rpg = json.dumps(rpg_html, ensure_ascii=True)
    report_data = (
        "// Generated by CoderMind; do not edit by hand.\n"
        f"window.CMIND_RPG_HTML = {embedded_rpg};\n"
        f"window.CMIND_REPORT = {payload};\n"
    )

    # Publish dependencies first and the entry point last so readers never see a
    # new report.html before its matching data and assets are available.
    _atomic_write_text(outputs.report_css, _asset("report.css"))
    _atomic_write_text(outputs.report_js, _asset("report.js"))
    if d3_source is not None and outputs.d3_js is not None:
        _atomic_write_text(outputs.d3_js, d3_source.read_text(encoding="utf-8"))
    for path, content in history_details.items():
        _atomic_write_text(path, content)
    _atomic_write_text(outputs.history_index_js, history_index)
    _atomic_write_text(outputs.report_data_js, report_data)
    _atomic_write_text(outputs.rpg_html, rpg_html)
    _atomic_write_text(outputs.report_html, _asset("report.html"))
    history_dir = target / "history"
    if history_dir.is_dir():
        current_details = {path.resolve() for path in history_details}
        for stale in history_dir.glob("*.js"):
            if stale.resolve() not in current_details:
                stale.unlink(missing_ok=True)
    return DashboardReportOutputs(
        report_html=outputs.report_html,
        report_data_js=outputs.report_data_js,
        rpg_html=outputs.rpg_html,
        report_css=outputs.report_css,
        report_js=outputs.report_js,
        history_index_js=outputs.history_index_js,
        history_detail_files=tuple(sorted(history_details)),
        d3_js=outputs.d3_js,
    )
