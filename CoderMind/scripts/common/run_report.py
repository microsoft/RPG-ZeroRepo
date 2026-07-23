"""Shared HTML renderer for CoderMind command run reports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from common.paths import REPORTS_DIR
from common.run_events import _to_plain

_MAX_SUMMARY_CARDS = 7
_D3_ASSET = Path(__file__).resolve().parent / "assets" / "d3.v7.min.js"


def write_command_report(
    run: Any,
    *,
    report_dir: str | Path | None = None,
    timestamp: str | datetime | None = None,
) -> Path:
    """Write a sanitized Explain View HTML report and return its path."""
    if hasattr(run, "to_dict"):
        data = run.to_dict()
    elif isinstance(run, Mapping):
        data = dict(run)
    else:
        raise TypeError("write_command_report() expects a CommandRun or mapping")

    data = _to_plain(data)
    if not isinstance(data, Mapping):
        raise TypeError("CommandRun.to_dict() must return a mapping")

    command = str(data.get("command") or "command")
    status = data.get("status")
    title = data.get("title")
    timestamp = timestamp if timestamp is not None else data.get("timestamp")
    report_dir = report_dir if report_dir is not None else data.get("report_dir")

    generated_at = _display_timestamp(timestamp)
    filename_ts = _filename_timestamp(timestamp)
    safe_command = _slug(command)
    target_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(target_dir / f"cmind_run_{safe_command}_{filename_ts}.html")

    evidence = dict(data)
    evidence_data = evidence.get("evidence") if isinstance(evidence.get("evidence"), Mapping) else {}
    retrievals = data.get("retrievals") or evidence_data.get("retrievals")
    code_deltas = data.get("code_deltas")
    focused_view = data.get("focused_view")
    user_decisions = data.get("user_decisions") or evidence_data.get("user_decisions")

    page_title = title or f"CoderMind {command} Explain View"
    html = _render_page(
        title=page_title,
        command=command,
        generated_at=generated_at,
        status=status,
        summary_cards=_normalize_cards(data.get("summary")),
        stages=_normalize_stages(data.get("steps")),
        retrievals=_normalize_retrievals(retrievals),
        rpg_nodes=_normalize_nodes(data.get("rpg_deltas"), dep_graph=False),
        dep_nodes=_normalize_nodes(data.get("dep_graph_deltas"), dep_graph=True),
        code_deltas=_normalize_code_deltas(code_deltas),
        focused_view=_normalize_focused_view(focused_view),
        artifacts=_normalize_artifacts(data.get("artifacts")),
        verification=_normalize_verification(data.get("verification")),
        user_decisions=_normalize_user_decisions(user_decisions),
        evidence=evidence,
    )
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug or "command"


def _unique_report_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}_{datetime.now(timezone.utc).strftime('%f')}{suffix}")


def _display_timestamp(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    if value is not None:
        return str(value)
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_timestamp(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        raw = value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    elif value is not None:
        raw = str(value)
    else:
        raw = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _slug(raw)


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _normalize_cards(value: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            cards.append({
                "label": item.get("label") or "Summary",
                "value": item.get("value", ""),
                "detail": item.get("detail"),
            })
        else:
            cards.append({"label": "Summary", "value": item})
    return cards[:_MAX_SUMMARY_CARDS]


def _normalize_stages(value: Any) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            stages.append({
                "name": item.get("name") or "stage",
                "status": item.get("status", "recorded"),
                "reason": item.get("reason", ""),
                "duration": item.get("duration"),
            })
        else:
            stages.append({"name": item, "status": "recorded"})
    return stages


def _normalize_retrievals(value: Any) -> list[dict[str, Any]]:
    retrievals: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            hits = item.get("hits")
            retrievals.append({
                "query": item.get("query", ""),
                "tool": item.get("tool", ""),
                "reason": item.get("reason", ""),
                "hits": _as_sequence(hits),
            })
        else:
            retrievals.append({"query": item, "hits": []})
    return retrievals


def _normalize_nodes(value: Any, *, dep_graph: bool = False) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    id_key = "dep_node_id" if dep_graph else "node_id"
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            entry = dict(item)
        else:
            entry = {id_key: item}
        nodes.append(entry)
    return nodes


def _normalize_code_deltas(value: Any) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            deltas.append({
                "file": item.get("file") or item.get("path") or "",
                "change_type": item.get("change_type") or item.get("status") or "",
                "before": item.get("before"),
                "after": item.get("after"),
                "diff": item.get("diff", ""),
            })
        else:
            deltas.append({"file": item, "change_type": "recorded", "diff": ""})
    return deltas


def _normalize_focused_view(value: Any) -> dict[str, Any]:
    if value in (None, "", [], {}):
        return {}
    if not isinstance(value, Mapping):
        return {"detail": value}
    normalized = dict(value)
    normalized["summary"] = dict(value.get("summary") or {}) if isinstance(value.get("summary"), Mapping) else {}
    normalized["primary_rpg_nodes"] = [
        dict(node) if isinstance(node, Mapping) else {"node_id": node}
        for node in _as_sequence(value.get("primary_rpg_nodes"))
    ]
    normalized["primary_code_nodes"] = [
        dict(node) if isinstance(node, Mapping) else {"node_id": node}
        for node in _as_sequence(value.get("primary_code_nodes"))
    ]
    normalized["mappings"] = [
        dict(mapping) if isinstance(mapping, Mapping) else {"detail": mapping}
        for mapping in _as_sequence(value.get("mappings"))
    ]
    normalized["edges"] = [
        dict(edge) if isinstance(edge, Mapping) else {"detail": edge}
        for edge in _as_sequence(value.get("edges"))
    ]
    normalized["hidden_counts"] = dict(value.get("hidden_counts") or {}) if isinstance(value.get("hidden_counts"), Mapping) else {}
    normalized["hidden_context_nodes"] = [
        dict(node) if isinstance(node, Mapping) else {"detail": node}
        for node in _as_sequence(value.get("hidden_context_nodes"))
    ]
    normalized["warnings"] = [
        dict(warning) if isinstance(warning, Mapping) else {"message": warning}
        for warning in _as_sequence(value.get("warnings"))
    ]
    nodes_view = value.get("nodes_view")
    if isinstance(nodes_view, Mapping):
        normalized["nodes_view"] = _normalize_nodes_view(nodes_view)
    normalized["unmatched_code_deltas"] = [
        dict(delta) if isinstance(delta, Mapping) else {"file": delta}
        for delta in _as_sequence(value.get("unmatched_code_deltas"))
    ]
    return normalized


def _normalize_nodes_view(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized["summary"] = dict(value.get("summary") or {}) if isinstance(value.get("summary"), Mapping) else {}
    for key in ("semantic_nodes", "code_nodes", "mappings", "edges", "warnings", "changed_files", "hidden_context_nodes"):
        normalized[key] = [
            dict(item) if isinstance(item, Mapping) else {"detail": item}
            for item in _as_sequence(value.get(key))
        ]
    normalized["hidden_counts"] = dict(value.get("hidden_counts") or {}) if isinstance(value.get("hidden_counts"), Mapping) else {}
    return normalized


def _normalize_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            path = item.get("path")
            artifacts.append({
                "label": item.get("label") or path or "artifact",
                "path": path,
                "status": _artifact_status(path, item.get("status")),
                "detail": item.get("detail"),
            })
        else:
            artifacts.append({"label": Path(str(item)).name or "artifact", "path": item, "status": _artifact_status(item)})
    return artifacts


def _artifact_status(path: Any, status: Any = None) -> Any:
    if status not in (None, ""):
        return status
    if path in (None, ""):
        return "missing"
    try:
        return "available" if Path(str(path)).expanduser().exists() else "missing"
    except (OSError, ValueError):
        return "missing"


def _normalize_verification(value: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            checks.append({
                "name": item.get("name") or "verification",
                "status": item.get("status", ""),
                "detail": item.get("detail"),
            })
        else:
            checks.append({"name": "verification", "status": item})
    return checks


def _normalize_user_decisions(value: Any) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            entry = dict(item)
        else:
            entry = {"decision": item}
        decisions.append({
            "decision": entry.get("decision", ""),
            "before_state": entry.get("before_state"),
            "confirmed": entry.get("confirmed"),
            "branch": entry.get("branch", ""),
            "apply_status": entry.get("apply_status", ""),
            "test_status": entry.get("test_status", ""),
            "rollback_path": entry.get("rollback_path", ""),
        })
    return decisions


def _render_page(
    *,
    title: str,
    command: str,
    generated_at: str,
    status: str | None,
    summary_cards: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    rpg_nodes: list[dict[str, Any]],
    dep_nodes: list[dict[str, Any]],
    code_deltas: list[dict[str, Any]],
    focused_view: dict[str, Any],
    artifacts: list[dict[str, Any]],
    verification: list[dict[str, Any]],
    user_decisions: list[dict[str, Any]],
    evidence: Mapping[str, Any],
) -> str:
    status_html = f"<span class=\"status\">{_h(status)}</span>" if status else ""
    code_delta_anchors = _code_delta_anchors(code_deltas)
    code_file_anchors = _code_file_anchor_map(code_deltas, code_delta_anchors)
    focused_graph_html = _render_focused_graph(focused_view, code_file_anchors)
    code_deltas_html = _render_code_deltas(code_deltas, code_delta_anchors)
    summary_html = _render_summary_cards(summary_cards)
    timeline_html = _render_timeline(stages, verification)
    safety_html = _render_safety_boundary(user_decisions)
    if focused_graph_html:
        primary_sections_html = summary_html + focused_graph_html + code_deltas_html + timeline_html + safety_html
    else:
        primary_sections_html = summary_html + timeline_html + safety_html + code_deltas_html
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{_h(title)}</title>
<style>
:root {{ color-scheme: light; --bg:#f6f8fb; --card:#fff; --text:#1f2937; --muted:#6b7280; --line:#d9e0ea; --accent:#2563eb; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width:1440px; margin:0 auto; padding:32px 20px 48px; }}
header {{ margin-bottom:24px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
.meta {{ color:var(--muted); font-size:14px; display:flex; flex-wrap:wrap; gap:12px; }}
.status {{ border:1px solid var(--line); border-radius:999px; padding:2px 10px; background:#eef4ff; color:#174ea6; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; margin:16px 0; padding:18px; box-shadow:0 1px 2px rgba(15,23,42,.04); overflow-x:auto; }}
h2 {{ margin:0 0 14px; font-size:18px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.card {{ border:1px solid var(--line); border-radius:12px; padding:14px; background:#fbfdff; }}
.card-wide {{ grid-column:span 2; }}
.card-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.card-value {{ margin-top:6px; font-size:22px; font-weight:700; word-break:break-word; overflow-wrap:anywhere; }}
.card-value-long {{ font-size:14px; line-height:1.35; font-weight:600; }}
.card-detail {{ margin-top:5px; color:var(--muted); font-size:13px; }}
.timeline {{ list-style:none; margin:0; padding:0; }}
.timeline li {{ border-left:3px solid var(--accent); padding:0 0 14px 14px; margin-left:6px; }}
.timeline li:last-child {{ padding-bottom:0; }}
.stage-head {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.badge {{ font-size:12px; border-radius:999px; background:#eef2f7; padding:2px 8px; color:#334155; }}
.reason {{ color:var(--muted); margin-top:4px; }}
.delta {{ border:1px solid var(--line); border-radius:12px; padding:12px; margin:10px 0; background:#fbfdff; }}
.delta-head {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }}
.hit-list {{ margin:0; padding-left:18px; }}
table {{ width:100%; min-width:680px; border-collapse:collapse; font-size:14px; table-layout:auto; }}
th, td {{ border-top:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }}
th {{ color:var(--muted); font-weight:600; background:#fbfdff; }}
code {{ white-space:normal; overflow-wrap:anywhere; word-break:break-word; }}
a {{ color:var(--accent); text-decoration:none; overflow-wrap:anywhere; word-break:break-word; }}
a:hover {{ text-decoration:underline; }}
.empty {{ color:var(--muted); font-style:italic; }}
pre {{ white-space:pre-wrap; overflow:auto; background:#0f172a; color:#e5e7eb; border-radius:10px; padding:14px; }}
details summary {{ cursor:pointer; color:var(--accent); font-weight:600; }}
.focus-summary {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }}
.focus-summary .badge {{ background:#eef4ff; color:#174ea6; }}
.warning-list {{ margin:0 0 12px; padding-left:18px; }}
.focus-map {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; align-items:stretch; }}
.focus-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#fbfdff; display:flex; flex-direction:column; gap:8px; min-width:0; }}
.focus-card header {{ margin:0; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.focus-card-title {{ font-weight:700; overflow-wrap:anywhere; word-break:break-word; }}
.focus-card-meta {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; word-break:break-word; }}
.focus-links {{ display:flex; flex-wrap:wrap; gap:6px; }}
.focus-link {{ border:1px solid var(--line); border-radius:999px; padding:2px 8px; background:#fff; font-size:12px; }}
.focused-graph-section {{ overflow-x:hidden; }}
body.focused-graph-fullscreen-active {{ overflow:hidden; }}
.focused-graph-section.focused-graph-fullscreen {{ position:fixed; inset:0; z-index:9999; margin:0; padding:0; background:#020617; overflow:hidden; }}
.focused-graph-section.focused-graph-fullscreen > h2, .focused-graph-section.focused-graph-fullscreen > .focus-summary, .focused-graph-section.focused-graph-fullscreen > details {{ display:none; }}
.focused-graph-section.focused-graph-fullscreen .focused-graph-stage {{ width:100vw; height:100vh; border:0; border-radius:0; }}
.focused-graph-stage {{ border:1px solid #334155; border-radius:12px; background:#0f172a; height:clamp(520px,72vh,820px); position:relative; overflow:hidden; }}
.focused-graph-svg {{ display:block; width:100%; height:100%; cursor:grab; touch-action:none; }}
.focused-graph-svg:active {{ cursor:grabbing; }}
.focused-graph-toolbar {{ position:absolute; top:14px; left:14px; z-index:3; display:flex; flex-wrap:wrap; gap:8px; align-items:center; max-width:calc(100% - 28px); margin:0; padding:10px; border:1px solid #334155; border-radius:12px; background:rgba(15,23,42,.92); color:#e5e7eb; box-shadow:0 14px 32px rgba(2,6,23,.35); }}
.focused-graph-toolbar button, .focused-graph-toolbar input {{ border:1px solid #475569; border-radius:8px; background:#1e293b; color:#e5e7eb; padding:6px 10px; font:inherit; }}
.focused-graph-toolbar button {{ cursor:pointer; }}
.focused-graph-toolbar button:hover, .focused-graph-toolbar button:focus-visible {{ border-color:#60a5fa; background:#334155; }}
.focused-graph-toolbar button[aria-pressed="true"] {{ background:#1d4ed8; border-color:#93c5fd; color:#eff6ff; }}
.focused-graph-toolbar input::placeholder {{ color:#94a3b8; }}
.focused-graph-toolbar label {{ display:inline-flex; gap:6px; align-items:center; color:#cbd5e1; }}
.focused-graph-detail {{ position:absolute; top:14px; right:14px; z-index:2; width:min(320px,calc(100% - 28px)); max-height:calc(100% - 28px); overflow:auto; border:1px solid #334155; border-radius:12px; background:rgba(15,23,42,.94); color:#e5e7eb; padding:12px; box-shadow:0 18px 40px rgba(2,6,23,.35); }}
.focused-graph-detail h3 {{ margin:0 0 8px; font-size:15px; color:#f8fafc; overflow-wrap:anywhere; }}
.focused-graph-detail dl {{ margin:0; display:grid; gap:8px; }}
.focused-graph-detail-row {{ display:grid; gap:3px; }}
.focused-graph-detail dt {{ color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
.focused-graph-detail dd {{ margin:0; font-size:13px; overflow-wrap:anywhere; }}
.focused-graph-detail a {{ color:#93c5fd; }}
.focused-graph-detail code {{ color:#bfdbfe; }}
.focused-graph-detail .empty {{ color:#94a3b8; }}
.focused-graph-detail-badges {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }}
.focused-graph-detail-badges .badge {{ background:#1e293b; color:#cbd5e1; }}
.focused-graph-detail-list {{ margin:0; padding-left:18px; }}
.focused-graph-detail-list span {{ display:block; color:#94a3b8; font-size:12px; }}
.focused-graph-tree-link {{ fill:none; stroke:#64748b; stroke-width:1.4; }}
.focused-graph-link {{ fill:none; opacity:.86; transition:opacity .15s ease, stroke-width .15s ease; }}
.focused-graph-link.active {{ opacity:1; stroke-width:2.8; }}
.focused-graph-link.dimmed {{ opacity:.14; }}
.focused-graph-link.hidden {{ display:none; }}
.focused-graph-link.edge-semantic {{ stroke:#a78bfa; }}
.focused-graph-link.edge-dependency {{ stroke:#fb923c; }}
.focused-graph-link.relation-invokes, .focused-graph-link.relation-caller, .focused-graph-link.relation-callee {{ stroke:#4ade80; }}
.focused-graph-link.relation-imports, .focused-graph-link.relation-import {{ stroke:#fb923c; }}
.focused-graph-link.relation-inherits, .focused-graph-link.relation-inheritance {{ stroke:#c084fc; }}
.focused-graph-link.relation-references, .focused-graph-link.relation-reference {{ stroke:#60a5fa; }}
.focused-graph-link.source-dep-graph {{ stroke-dasharray:5 3; }}
.focused-graph-node {{ cursor:pointer; transition:opacity .15s ease; }}
.focused-graph-node circle {{ fill:#3b82f6; stroke:#e2e8f0; stroke-width:2; transition:stroke .15s ease, stroke-width .15s ease; }}
.focused-graph-node.non-focused circle {{ fill:#cbd5e1; }}
.focused-graph-node.non-focused text {{ fill:#94a3b8; }}
.focused-graph-node.selected circle, .focused-graph-node.active circle, .focused-graph-node.focused circle {{ stroke:#3b82f6; stroke-width:3; }}
.focused-graph-node.search-match circle {{ stroke:#f59e0b; stroke-width:3; }}
.focused-graph-node.dimmed {{ opacity:.18; }}
.focused-graph-node.hidden {{ display:none; }}
.focused-graph-fallback {{ margin:12px; padding:12px; border:1px dashed #475569; border-radius:10px; background:#1e293b; color:#cbd5e1; }}
.focused-graph-legend {{ position:absolute; left:14px; bottom:14px; z-index:3; display:flex; flex-wrap:wrap; gap:8px; max-width:calc(100% - 28px); margin:0; padding:10px; border:1px solid #334155; border-radius:12px; background:rgba(15,23,42,.92); box-shadow:0 14px 32px rgba(2,6,23,.35); }}
.legend-item {{ display:inline-flex; gap:6px; align-items:center; color:#cbd5e1; font-size:13px; }}
.legend-swatch {{ width:10px; height:10px; border-radius:999px; display:inline-block; border:1px solid #475569; flex:0 0 auto; }}
.legend-line {{ width:28px; height:0; border-radius:0; border:0; border-top:2px solid var(--line); background:transparent; }}
.legend-node {{ background:#2563eb; }}
.legend-tree-link {{ border-top-color:#cbd5e1; border-top-width:1.4px; }}
.legend-semantic-edge {{ border-top-color:#7c3aed; }}
.legend-dependency-edge, .legend-imports-edge {{ border-top-color:#ea580c; }}
.legend-invokes-edge {{ border-top-color:#16a34a; }}
.legend-inherits-edge {{ border-top-color:#9333ea; }}
.legend-references-edge {{ border-top-color:#2563eb; }}
.legend-dep-graph-edge {{ border-top-style:dashed; }}
@media (max-width:720px) {{ main {{ padding:22px 12px 36px; }} .focus-map {{ grid-template-columns:1fr; }} table {{ min-width:560px; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>{_h(title)}</h1>
<div class=\"meta\"><span>Command: <strong>{_h(command)}</strong></span><span>Generated: {_h(generated_at)}</span>{status_html}</div>
</header>
{primary_sections_html}
{_render_artifacts(artifacts)}
{_render_evidence(evidence)}
</main>
</body>
</html>
"""


def _render_summary_cards(cards: list[dict[str, Any]]) -> str:
    if not cards:
        body = "<p class=\"empty\">No summary cards recorded.</p>"
    else:
        rendered_cards = []
        for card in cards:
            detail = card.get("detail")
            detail_html = ""
            if detail is not None:
                detail_html = f"<div class=\"card-detail\">{_h(detail)}</div>"
            value = card.get("value", "")
            long_value = len(str(value)) > 48
            card_class = "card card-wide" if long_value else "card"
            value_class = "card-value card-value-long" if long_value else "card-value"
            rendered_cards.append(
                f"<div class=\"{card_class}\"><div class=\"card-label\">{_h(card.get('label', 'Summary'))}</div>"
                f"<div class=\"{value_class}\">{_h(value)}</div>"
                f"{detail_html}</div>"
            )
        body = '<div class="cards">' + "".join(rendered_cards) + "</div>"
    return f"<section><h2>Summary</h2>{body}</section>"


def _render_timeline(stages: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    items = []
    for stage in stages:
        duration = stage.get("duration")
        duration_text = f"<span class=\"badge\">{_h(duration)}s</span>" if duration not in (None, "") else ""
        items.append(
            "<li>"
            f"<div class=\"stage-head\"><strong>{_h(stage.get('name', 'stage'))}</strong>"
            f"<span class=\"badge\">{_h(stage.get('status', 'recorded'))}</span>{duration_text}</div>"
            f"<div class=\"reason\">{_h(stage.get('reason', ''))}</div>"
            "</li>"
        )
    for check in checks:
        items.append(
            "<li>"
            f"<div class=\"stage-head\"><strong>{_h(check.get('name') or 'verification')}</strong>"
            f"<span class=\"badge\">{_h(check.get('status', ''))}</span></div>"
            f"<div class=\"reason\">{_h(check.get('detail', ''))}</div>"
            "</li>"
        )
    if items:
        body = '<ol class="timeline">' + "".join(items) + "</ol>"
    else:
        body = "<p class=\"empty\">No stages recorded.</p>"
    return f"<section><h2>Stage timeline</h2>{body}</section>"


def _render_safety_boundary(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return ""
    rows = []
    for decision in decisions:
        confirmed = decision.get("confirmed")
        if confirmed is True:
            confirmation = "confirmed"
        elif confirmed is False:
            confirmation = "not confirmed"
        else:
            confirmation = ""
        rows.append(
            "<tr>"
            f"<td>{_h(decision.get('decision', ''))}</td>"
            f"<td>{_h(decision.get('before_state'))}</td>"
            f"<td>{_h(confirmation)}</td>"
            f"<td>{_h(decision.get('branch', ''))}</td>"
            f"<td>{_h(decision.get('apply_status', ''))}</td>"
            f"<td>{_h(decision.get('test_status', ''))}</td>"
            f"<td>{_h(decision.get('rollback_path', ''))}</td>"
            "</tr>"
        )
    body = (
        "<table><thead><tr><th>Decision</th><th>Before state</th>"
        "<th>Confirmation</th><th>Branch</th><th>Apply status</th>"
        "<th>Test status</th><th>Rollback path</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return f"<section><h2>Safety boundary</h2>{body}</section>"


def _render_retrievals(retrievals: list[dict[str, Any]], *, title: str = "Retrieval evidence", as_section: bool = True) -> str:
    if not retrievals:
        return "" if as_section else f"<h3>{_h(title)}</h3><p class=\"empty\">No retrieval evidence recorded.</p>"
    rows = []
    for retrieval in retrievals:
        hits = retrieval.get("hits") or []
        hit_items = []
        for hit in hits:
            if isinstance(hit, Mapping):
                label = hit.get("node_id") or hit.get("dep_node_id") or hit.get("path") or hit.get("file") or hit.get("name") or "hit"
                reason = hit.get("reason") or hit.get("score") or hit.get("status") or ""
                hit_items.append(f"<li><code>{_h(label)}</code> {_h(reason)}</li>")
            else:
                hit_items.append(f"<li>{_h(hit)}</li>")
        hits_html = "<span class=\"empty\">No hits recorded.</span>"
        if hit_items:
            hits_html = '<ul class="hit-list">' + "".join(hit_items) + "</ul>"
        rows.append(
            "<tr>"
            f"<td>{_h(retrieval.get('tool', ''))}</td>"
            f"<td>{_h(retrieval.get('query', ''))}</td>"
            f"<td>{_h(retrieval.get('reason', ''))}</td>"
            f"<td>{hits_html}</td>"
            "</tr>"
        )
    body = "<table><thead><tr><th>Tool</th><th>Query</th><th>Reason</th><th>Hits</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    if not as_section:
        return f"<h3>{_h(title)}</h3>{body}"
    return f"<section><h2>{_h(title)}</h2>{body}</section>"


def _delta_file(delta: Mapping[str, Any]) -> str:
    return str(delta.get("file") or delta.get("path") or "")


def _code_delta_anchors(deltas: list[dict[str, Any]]) -> list[str]:
    anchors: list[str] = []
    used: dict[str, int] = {}
    for index, delta in enumerate(deltas, start=1):
        file_path = _delta_file(delta) or f"change-{index}"
        base = _slug(f"diff-{file_path}")
        count = used.get(base, 0) + 1
        used[base] = count
        anchors.append(base if count == 1 else f"{base}-{count}")
    return anchors


def _code_file_anchor_map(deltas: list[dict[str, Any]], anchors: list[str]) -> dict[str, str]:
    file_anchors: dict[str, str] = {}
    for delta, anchor in zip(deltas, anchors):
        file_path = _delta_file(delta)
        if file_path and file_path not in file_anchors:
            file_anchors[file_path] = anchor
    return file_anchors


def _diff_file_link(file_path: Any, file_anchors: Mapping[str, str]) -> str:
    file_text = str(file_path or "")
    anchor = file_anchors.get(file_text)
    if not anchor:
        return f"<code>{_h(file_text)}</code>"
    return f"<a href=\"#{_h_attr(anchor)}\"><code>{_h(file_text)}</code></a>"


def _render_code_deltas(deltas: list[dict[str, Any]], anchors: list[str] | None = None) -> str:
    if not deltas:
        return ""
    anchors = anchors or _code_delta_anchors(deltas)
    blocks = []
    for index, delta in enumerate(deltas):
        anchor = anchors[index] if index < len(anchors) else _slug(f"diff-{index + 1}")
        diff = delta.get("diff", "")
        diff_html = "<p class=\"empty\">No diff recorded.</p>"
        if diff:
            diff_html = f"<details><summary>View diff</summary><pre>{_h(diff)}</pre></details>"
        before_after = ""
        if delta.get("before") is not None or delta.get("after") is not None:
            before_after = (
                "<details><summary>Before/after</summary>"
                f"<pre>{_h({'before': delta.get('before'), 'after': delta.get('after')})}</pre>"
                "</details>"
            )
        blocks.append(
            f"<div class=\"delta\" id=\"{_h_attr(anchor)}\">"
            "<div class=\"delta-head\">"
            f"<code>{_h(delta.get('file', ''))}</code>"
            f"<span class=\"badge\">{_h(delta.get('change_type', ''))}</span>"
            "</div>"
            f"{diff_html}{before_after}"
            "</div>"
        )
    return f"<section><h2>What changed?</h2>{''.join(blocks)}</section>"


def _summary_badges(summary: Mapping[str, Any], labels: list[tuple[str, str, Any]]) -> str:
    badges = []
    for label, key, fallback in labels:
        value = summary.get(key, fallback)
        badges.append(f"<span class=\"badge\">{_h(label)}: <strong>{_h(value)}</strong></span>")
    return "<div class=\"focus-summary\">" + "".join(badges) + "</div>"


def _mapping_changed_files(mapping: Mapping[str, Any], rpg_node: Mapping[str, Any]) -> list[str]:
    return _ordered_texts(_as_sequence(mapping.get("changed_files")) + _as_sequence(rpg_node.get("changed_files")))


def _ordered_texts(values: Sequence[Any]) -> list[str]:
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


def _changed_file_links(files: Sequence[Any], file_anchors: Mapping[str, str]) -> str:
    file_list = _ordered_texts(list(files))
    if not file_list:
        return "<span class=\"empty\">No changed files mapped.</span>"
    return ", ".join(_diff_file_link(file_path, file_anchors) for file_path in file_list)


def _chain_warning_html(warnings: Sequence[Mapping[str, Any]]) -> str:
    if not warnings:
        return "<span class=\"empty\">No warnings.</span>"
    items = []
    for warning in warnings:
        warning_type = warning.get("type", "warning")
        context = {key: value for key, value in warning.items() if key not in {"type", "message"}}
        context_html = f" <code>{_h(context)}</code>" if context else ""
        items.append(f"<li><code>{_h(warning_type)}</code> {_h(warning.get('message', ''))}{context_html}</li>")
    return '<ul class="hit-list">' + "".join(items) + "</ul>"


def _chain_edge_html(edges: Sequence[Mapping[str, Any]]) -> str:
    if not edges:
        return "<span class=\"empty\">No visible neighborhood edges.</span>"
    items = []
    for edge in edges:
        source = edge.get("source_node_id", "")
        target = edge.get("target_node_id", "")
        relation = edge.get("relation") or "dependency"
        direction = edge.get("direction", "")
        path = edge.get("path", "")
        reason = edge.get("reason", "")
        items.append(
            "<li>"
            f"<code>{_h(source)}</code> → <code>{_h(target)}</code>"
            f" <span class=\"badge\">{_h(relation)}</span>"
            f" {_h(direction)}"
            f"<div class=\"reason\">{_h(path)} {_h(reason)}</div>"
            "</li>"
        )
    return '<ul class="hit-list">' + "".join(items) + "</ul>"


def _combined_hidden_counts(hidden_counts: Mapping[str, Any]) -> list[tuple[str, int]]:
    relation_keys = {"caller": "callers", "callee": "callees", "import": "imports", "inheritance": "inheritance"}
    rows: list[tuple[str, int]] = []
    for relation, count_key in relation_keys.items():
        count = hidden_counts.get(count_key) or 0
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            count_int = 0
        if count_int:
            rows.append((relation, count_int))
    return rows


def _hidden_context_html(hidden_counts: Mapping[str, Any], hidden_context_nodes: Sequence[Any] = ()) -> str:
    parts = []
    for relation, count in _combined_hidden_counts(hidden_counts):
        parts.append(f"<p class=\"reason\">Hidden {_h(count)} additional {_h(relation)} neighbors.</p>")
    rows = [node for node in _as_sequence(hidden_context_nodes) if isinstance(node, Mapping)]
    if rows:
        items = []
        for node in rows:
            node_id = node.get("node_id") or node.get("rpg_node_id") or node.get("id") or "hidden context"
            title = node.get("name") or node.get("symbol") or node_id
            reason = node.get("hidden_reason") or node.get("reason") or "hidden context"
            path = node.get("breadcrumb_path") or node.get("feature_path") or node.get("path") or ""
            items.append(
                "<li>"
                f"<code>{_h(node_id)}</code> {_h(title)}"
                f" <span class=\"badge\">hidden context</span>"
                f"<div class=\"reason\">{_h(reason)}</div>"
                f"<div class=\"reason\">{_h(path)}</div>"
                "</li>"
            )
        parts.append("<details><summary>Hidden context nodes</summary><ul class=\"hit-list\">" + "".join(items) + "</ul></details>")
    return "".join(parts)


def _focused_inspector_payload(focused_view: Mapping[str, Any]) -> dict[str, Any]:
    nodes_view = focused_view.get("nodes_view") if isinstance(focused_view.get("nodes_view"), Mapping) else {}
    if nodes_view:
        return {"nodes_view": nodes_view}
    payload: dict[str, Any] = {}
    for key, value in focused_view.items():
        if key in {"unmatched_code_deltas"}:
            unmatched_files = [
                _delta_file(delta)
                for delta in _as_sequence(value)
                if isinstance(delta, Mapping) and _delta_file(delta)
            ]
            if unmatched_files:
                payload["unmatched_changed_files"] = unmatched_files
            continue
        payload[key] = value
    return payload


def _node_view_id(node: Mapping[str, Any]) -> str:
    return str(node.get("link_id") or node.get("node_id") or node.get("dep_node_id") or "")


def _line_range_text(node: Mapping[str, Any]) -> str:
    line_range = node.get("line_range") if isinstance(node.get("line_range"), Mapping) else {}
    start = line_range.get("start") or node.get("line_start") or node.get("start_line") or node.get("lineno") or node.get("line")
    end = line_range.get("end") or node.get("line_end") or node.get("end_line") or start
    if start in (None, ""):
        return "unavailable"
    if end in (None, "") or str(end) == str(start):
        return str(start)
    return f"{start}-{end}"


def _changed_refs_html(refs: Sequence[Any], file_anchors: Mapping[str, str]) -> str:
    links = []
    for ref in _as_sequence(refs):
        if isinstance(ref, Mapping):
            path = ref.get("path") or ref.get("file")
            anchor = ref.get("diff_anchor") or file_anchors.get(str(path or ""))
        else:
            path = ref
            anchor = file_anchors.get(str(path or ""))
        if path in (None, ""):
            continue
        if anchor:
            links.append(f"<a class=\"focus-link\" href=\"#{_h_attr(anchor)}\">{_h(path)}</a>")
        else:
            links.append(f"<span class=\"focus-link\">{_h(path)}</span>")
    if not links:
        return ""
    return '<div class="focus-links">' + "".join(links) + "</div>"


def _focus_card_badges(node: Mapping[str, Any], *keys: str) -> str:
    badges = []
    for key in keys:
        value = node.get(key)
        if value not in (None, "", [], {}):
            badges.append(f"<span class=\"badge\">{_h(value)}</span>")
    return "".join(badges)


def _semantic_card(node: Mapping[str, Any], file_anchors: Mapping[str, str]) -> str:
    node_id = node.get("node_id")
    link_id = _node_view_id(node)
    title = node.get("name") or node.get("symbol") or node_id or "semantic node"
    breadcrumb = node.get("breadcrumb_path") or node.get("breadcrumb") or node.get("feature_path") or node.get("path") or "unavailable"
    mapped = _as_sequence(node.get("mapped_code_node_ids"))
    mapped_html = ""
    if mapped:
        mapped_html = '<div class="focus-card-meta">Mapped code: ' + ", ".join(f"<code>{_h(item)}</code>" for item in mapped) + "</div>"
    elif (node.get("mapping_status") or node.get("state")) in {"missing", "missing_mapping"}:
        mapped_html = '<div class="focus-card-meta">Mapped code: <span class="empty">missing mapping</span></div>'
    warnings = _as_sequence(node.get("warning_types"))
    warning_html = '<div class="focus-card-meta">Warnings: ' + ", ".join(f"<code>{_h(item)}</code>" for item in warnings) + "</div>" if warnings else ""
    return (
        f"<article class=\"focus-card\" id=\"{_h_attr(link_id)}\">"
        f"<header><span class=\"badge\">semantic</span>{_focus_card_badges(node, 'state', 'mapping_status', 'locate_status')}</header>"
        f"<div class=\"focus-card-title\">{_h(title)}</div>"
        f"<div class=\"focus-card-meta\"><code>{_h(node_id)}</code></div>"
        f"<div class=\"focus-card-meta\">Breadcrumb: {_h(breadcrumb)}</div>"
        f"<div class=\"focus-card-meta\">Type: {_h(node.get('node_type') or node.get('type') or 'unavailable')}</div>"
        f"{mapped_html}{warning_html}{_changed_refs_html(_as_sequence(node.get('changed_files')), file_anchors)}"
        "</article>"
    )


def _code_card(node: Mapping[str, Any], file_anchors: Mapping[str, str]) -> str:
    node_id = node.get("node_id") or node.get("dep_node_id")
    link_id = _node_view_id(node)
    path = node.get("path") or node.get("module") or node.get("file") or "unavailable"
    symbol = node.get("symbol") or node.get("name") or "unavailable"
    changed_refs = _as_sequence(node.get("changed_files"))
    if not changed_refs and node.get("diff_anchor") and path not in (None, "", "unavailable"):
        changed_refs = [{"path": path, "diff_anchor": node.get("diff_anchor")}]
    mapped = _as_sequence(node.get("mapped_rpg_node_ids"))
    mapped_html = '<div class="focus-card-meta">Mapped features: ' + ", ".join(f"<code>{_h(item)}</code>" for item in mapped) + "</div>" if mapped else ""
    return (
        f"<article class=\"focus-card\" id=\"{_h_attr(link_id)}\">"
        f"<header><span class=\"badge\">code</span>{_focus_card_badges(node, 'state', 'source')}</header>"
        f"<div class=\"focus-card-title\">{_h(symbol)}</div>"
        f"<div class=\"focus-card-meta\"><code>{_h(node_id)}</code></div>"
        f"<div class=\"focus-card-meta\">Path: {_h(path)}</div>"
        f"<div class=\"focus-card-meta\">Type: {_h(node.get('type') or node.get('kind') or 'unavailable')}</div>"
        f"<div class=\"focus-card-meta\">Lines: {_h(_line_range_text(node))}</div>"
        f"{mapped_html}{_changed_refs_html(changed_refs, file_anchors)}"
        "</article>"
    )


def _mapping_card(mapping: Mapping[str, Any]) -> str:
    target = mapping.get("code_node_id") or mapping.get("dep_node_id") or "missing mapping"
    source_link = mapping.get("source_link_id")
    target_link = mapping.get("target_link_id")
    source_html = f"<a href=\"#{_h_attr(source_link)}\"><code>{_h(mapping.get('rpg_node_id'))}</code></a>" if source_link else f"<code>{_h(mapping.get('rpg_node_id'))}</code>"
    target_html = f"<a href=\"#{_h_attr(target_link)}\"><code>{_h(target)}</code></a>" if target_link else f"<code>{_h(target)}</code>"
    return (
        "<article class=\"focus-card\">"
        f"<header><span class=\"badge\">mapping</span>{_focus_card_badges(mapping, 'state', 'status', 'source')}</header>"
        f"<div class=\"focus-card-title\">{source_html} → {target_html}</div>"
        f"<div class=\"focus-card-meta\">{_h(mapping.get('path') or mapping.get('reason') or '')}</div>"
        "</article>"
    )


def _inline_d3() -> str:
    try:
        return _D3_ASSET.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _json_for_script(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, default=_json_default)
    return (
        data.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _mapped_code_label(row: Mapping[str, Any]) -> str:
    path = row.get("mapped_code_path")
    symbol = row.get("mapped_code_symbol")
    first = next((item for item in _as_sequence(row.get("mapped_code")) if isinstance(item, Mapping)), None)
    if first:
        path = path or first.get("path")
        symbol = symbol or first.get("symbol") or first.get("name")
    if not path:
        paths = _as_sequence(row.get("mapped_code_paths"))
        path = paths[0] if paths else None
    if not symbol:
        symbols = _as_sequence(row.get("mapped_code_symbols"))
        symbol = symbols[0] if symbols else None
    detail = " · ".join(str(item) for item in (path, symbol) if item not in (None, ""))
    if not detail:
        return ""
    count = row.get("mapped_code_count")
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        count_int = 0
    suffix = f" +{count_int - 1}" if count_int > 1 else ""
    return f"{detail}{suffix}"


def _graph_label(row: Mapping[str, Any], fallback: Any) -> str:
    base = ""
    for key in ("feature_name", "name", "symbol", "label", "path", "node_id", "dep_node_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            base = str(value)
            break
    if not base:
        base = str(fallback or "node")
    mapped_code = _mapped_code_label(row)
    return f"{base} — {mapped_code}" if mapped_code else base


def _graph_diff_ref(row: Mapping[str, Any], file_anchors: Mapping[str, str]) -> dict[str, str]:
    refs = _as_sequence(row.get("changed_files"))
    path = row.get("path") or row.get("file") or row.get("module")
    if not refs and row.get("diff_anchor") and path not in (None, ""):
        refs = [{"path": path, "diff_anchor": row.get("diff_anchor")}]
    for ref in refs:
        if isinstance(ref, Mapping):
            ref_path = ref.get("path") or ref.get("file")
            anchor = ref.get("diff_anchor") or file_anchors.get(str(ref_path or ""))
        else:
            ref_path = ref
            anchor = file_anchors.get(str(ref_path or ""))
        if ref_path not in (None, "") and anchor:
            return {"path": str(ref_path), "href": f"#{anchor}"}
    return {}


def _graph_node_id(row: Mapping[str, Any], kind: str) -> str:
    explicit = _node_view_id(row)
    if explicit:
        return explicit
    if kind == "mapping":
        node_id = f"{row.get('rpg_node_id') or 'semantic'}-{row.get('code_node_id') or row.get('dep_node_id') or 'missing'}"
    else:
        node_id = row.get("node_id") or row.get("dep_node_id") or row.get("id") or kind
    return f"{kind}-{_slug(str(node_id))}"


def _append_graph_node(nodes: list[dict[str, Any]], seen: set[str], row: Mapping[str, Any], kind: str, file_anchors: Mapping[str, str]) -> str:
    node_id = _graph_node_id(row, kind)
    if node_id in seen:
        return node_id
    seen.add(node_id)
    payload: dict[str, Any] = {
        "id": node_id,
        "kind": kind,
        "label": _graph_label(row, row.get("node_id") or row.get("dep_node_id") or node_id),
        "node_id": row.get("node_id") or row.get("dep_node_id"),
        "state": row.get("state") or row.get("status") or row.get("mapping_status"),
    }
    for key in (
        "path",
        "feature_path",
        "feature_name",
        "type",
        "node_type",
        "source",
        "mapping_status",
        "locate_status",
        "breadcrumb_path",
        "mapped_code",
        "mapped_code_path",
        "mapped_code_paths",
        "mapped_code_symbol",
        "mapped_code_symbols",
        "mapped_code_count",
    ):
        if row.get(key) not in (None, ""):
            payload[key] = row.get(key)
    diff_ref = _graph_diff_ref(row, file_anchors)
    if diff_ref:
        payload["diff"] = diff_ref
    nodes.append(payload)
    return node_id


def _append_context_node(nodes: list[dict[str, Any]], seen: set[str], link_id: Any, node_id: Any) -> str:
    graph_id = str(link_id or f"context-{_slug(str(node_id or 'node'))}")
    if graph_id not in seen:
        seen.add(graph_id)
        nodes.append({"id": graph_id, "kind": "context", "label": str(node_id or "context"), "node_id": node_id, "state": "context"})
    return graph_id


def _append_hierarchy_nodes(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    seen_nodes: set[str],
    hierarchy: Any,
    file_anchors: Mapping[str, str],
    node_metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(hierarchy, Mapping):
        return
    seen_links = {str(link.get("id")) for link in links if isinstance(link, Mapping) and link.get("id") not in (None, "")}

    def node_id(row: Mapping[str, Any]) -> str:
        return str(row.get("id") or row.get("link_id") or row.get("node_id") or "")

    def visit(row: Any) -> str:
        if not isinstance(row, Mapping):
            return ""
        row_id = node_id(row)
        if row_id and row_id not in seen_nodes:
            seen_nodes.add(row_id)
            metadata = node_metadata_by_id.get(row_id) or node_metadata_by_id.get(str(row.get("node_id") or "")) or {}
            payload = {
                "id": row_id,
                "kind": row.get("kind") or "feature",
                "label": _graph_label(row, row.get("node_id") or row_id),
                "node_id": row.get("node_id") or row_id,
                "state": row.get("state") or row.get("kind") or "hierarchy",
                "type": row.get("kind") or "hierarchy",
            }
            for key in (
                "name",
                "feature_name",
                "feature_path",
                "path",
                "module",
                "file",
                "symbol",
                "dep_node_id",
                "type",
                "node_type",
                "signature",
                "mapping_status",
                "locate_status",
                "breadcrumb",
                "breadcrumb_path",
                "score",
                "source",
                "source_graph",
                "edge_source",
                "relation_source",
                "source_feature",
                "source_features",
                "relation",
                "direction",
                "reason",
                "line_range",
                "apply_action",
                "mapped_code",
                "mapped_code_node_ids",
                "mapped_code_link_ids",
                "mapped_code_path",
                "mapped_code_paths",
                "mapped_code_symbol",
                "mapped_code_symbols",
                "mapped_code_count",
                "mapped_rpg_node_ids",
                "mapped_rpg_link_ids",
                "changed",
                "changed_files",
                "affected_files",
                "diff_anchor",
                "diff",
                "hidden_counts",
                "warning_types",
                "rpg_node_id",
                "neighbor_node_id",
            ):
                value = row.get(key) if row.get(key) not in (None, "") else metadata.get(key)
                if value not in (None, ""):
                    payload[key] = value
            diff_ref = _graph_diff_ref(payload, file_anchors)
            if diff_ref:
                payload["diff"] = diff_ref
            nodes.append(payload)
        for child in [child for child in _as_sequence(row.get("children")) if isinstance(child, Mapping)]:
            child_id = visit(child)
            if row_id in seen_nodes and child_id in seen_nodes:
                link_id = f"hierarchy-{_slug(row_id)}-{_slug(child_id)}"
                if link_id not in seen_links:
                    seen_links.add(link_id)
                    links.append({"id": link_id, "source": row_id, "target": child_id, "kind": "hierarchy", "relation": "contains"})
        return row_id

    visit(hierarchy)


def _focused_graph_payload(focused_view: Mapping[str, Any], file_anchors: Mapping[str, str]) -> dict[str, Any]:
    nodes_view = focused_view.get("nodes_view") if isinstance(focused_view.get("nodes_view"), Mapping) else {}
    summary_source = nodes_view.get("summary") if isinstance(nodes_view.get("summary"), Mapping) else focused_view.get("summary", {})
    summary = dict(summary_source) if isinstance(summary_source, Mapping) else {}
    semantic_nodes = [node for node in _as_sequence(nodes_view.get("semantic_nodes")) if isinstance(node, Mapping)]
    code_nodes = [node for node in _as_sequence(nodes_view.get("code_nodes")) if isinstance(node, Mapping)]
    mappings = [mapping for mapping in _as_sequence(nodes_view.get("mappings")) if isinstance(mapping, Mapping)]
    context_edges = [edge for edge in _as_sequence(nodes_view.get("edges")) if isinstance(edge, Mapping)]
    hidden_counts = nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else focused_view.get("hidden_counts", {})
    hidden_context_nodes = [node for node in _as_sequence(nodes_view.get("hidden_context_nodes") or focused_view.get("hidden_context_nodes")) if isinstance(node, Mapping)]
    warnings = [warning for warning in _as_sequence(nodes_view.get("warnings")) if isinstance(warning, Mapping)]
    focused_graph = nodes_view.get("focused_graph") if isinstance(nodes_view.get("focused_graph"), Mapping) else {}
    hierarchy = nodes_view.get("hierarchy") or focused_graph.get("hierarchy") or {}
    default_focus = nodes_view.get("default_focus") or focused_graph.get("default_focus") or {}

    if not isinstance(hierarchy, Mapping) or not hierarchy or (semantic_nodes and not _as_sequence(hierarchy.get("children"))):
        hierarchy = {
            "id": "focused-graph-root",
            "name": "Focused graph",
            "kind": "root",
            "children": [
                {
                    "id": _graph_node_id(node, "feature"),
                    "node_id": node.get("node_id"),
                    "name": node.get("name") or node.get("symbol") or node.get("node_id") or "feature",
                    "kind": "feature",
                    "feature_name": node.get("name") or node.get("symbol") or node.get("node_id") or "feature",
                    "feature_path": node.get("breadcrumb_path") or node.get("feature_path"),
                    "mapped_code": node.get("mapped_code"),
                    "mapped_code_node_ids": node.get("mapped_code_node_ids"),
                    "mapped_code_link_ids": node.get("mapped_code_link_ids"),
                    "mapped_code_path": node.get("mapped_code_path"),
                    "mapped_code_paths": node.get("mapped_code_paths"),
                    "mapped_code_symbol": node.get("mapped_code_symbol"),
                    "mapped_code_symbols": node.get("mapped_code_symbols"),
                    "mapped_code_count": node.get("mapped_code_count"),
                }
                for node in semantic_nodes
            ],
        }

    semantic_link_by_node = {
        str(node.get("node_id")): str(node.get("link_id") or _graph_node_id(node, "feature"))
        for node in semantic_nodes
        if node.get("node_id") not in (None, "")
    }
    code_link_by_node = {
        str(node.get("node_id") or node.get("dep_node_id")): str(node.get("link_id") or _graph_node_id(node, "code"))
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }
    node_metadata_by_id: dict[str, Mapping[str, Any]] = {}

    def remember_node_metadata(alias: Any, row: Mapping[str, Any]) -> None:
        if alias in (None, ""):
            return
        node_metadata_by_id.setdefault(str(alias), row)

    for node in semantic_nodes:
        node_id = node.get("node_id")
        remember_node_metadata(node.get("link_id") or _graph_node_id(node, "feature"), node)
        remember_node_metadata(node_id, node)
    for node in code_nodes:
        node_id = node.get("node_id") or node.get("dep_node_id")
        remember_node_metadata(node.get("link_id") or _graph_node_id(node, "code"), node)
        remember_node_metadata(node_id, node)

    def endpoint_link_id(edge: Mapping[str, Any], side: str) -> str:
        node_id = edge.get(f"{side}_node_id")
        node_text = str(node_id or "")
        return str(
            edge.get(f"{side}_link_id")
            or semantic_link_by_node.get(node_text)
            or code_link_by_node.get(node_text)
            or _graph_node_id({"node_id": node_text}, "context")
        )

    def collect_hierarchy_ids(row: Any, ids: set[str]) -> None:
        if not isinstance(row, Mapping):
            return
        row_id = row.get("id") or row.get("link_id") or row.get("node_id")
        if row_id not in (None, ""):
            ids.add(str(row_id))
        for child in _as_sequence(row.get("children")):
            collect_hierarchy_ids(child, ids)

    def endpoint_group() -> dict[str, Any]:
        children = hierarchy.setdefault("children", []) if isinstance(hierarchy, dict) else []
        group_id = "focused-graph-relation-endpoints"
        for child in children:
            if isinstance(child, dict) and child.get("id") == group_id:
                return child
        group = {"id": group_id, "name": "Relation endpoints", "kind": "context_group", "children": []}
        children.append(group)
        return group

    hierarchy_ids: set[str] = set()
    collect_hierarchy_ids(hierarchy, hierarchy_ids)
    code_link_values = set(code_link_by_node.values())
    for edge in context_edges:
        for side in ("source", "target"):
            link_id = endpoint_link_id(edge, side)
            if not link_id or link_id in hierarchy_ids:
                continue
            node_id = edge.get(f"{side}_node_id")
            node_text = str(node_id or link_id)
            is_code = link_id in code_link_values
            leaf: dict[str, Any] = {
                "id": link_id,
                "node_id": node_text,
                "name": edge.get(f"{side}_name") or edge.get("name") or edge.get(f"{side}_path") or edge.get("path") or node_text,
                "kind": "code" if is_code else "context",
                "state": "mapped" if is_code else "context",
                "aliases": _ordered_texts([node_text, link_id]),
            }
            if is_code:
                code = next((row for row in code_nodes if str(row.get("node_id") or row.get("dep_node_id") or "") == node_text), {})
                for key in ("path", "symbol", "type", "line_range", "source", "changed_files", "diff_anchor"):
                    if isinstance(code, Mapping) and code.get(key) not in (None, ""):
                        leaf[key] = code.get(key)
            else:
                for key in ("relation", "direction", "reason", "source", "source_graph", "edge_source", "relation_source"):
                    if edge.get(key) not in (None, ""):
                        leaf[key] = edge.get(key)
                path = edge.get(f"{side}_path") or edge.get("path")
                if path not in (None, ""):
                    leaf["path"] = path
            endpoint_group().setdefault("children", []).append(leaf)
            hierarchy_ids.add(link_id)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    _append_hierarchy_nodes(nodes, links, seen_nodes, hierarchy, file_anchors, node_metadata_by_id)

    rpg_links: dict[str, str] = {}
    code_links: dict[str, str] = {}
    node_aliases: dict[str, list[str]] = {}

    def add_alias(alias: Any, target: Any) -> None:
        if alias in (None, "") or target in (None, ""):
            return
        alias_text = str(alias)
        target_text = str(target)
        values = node_aliases.setdefault(alias_text, [])
        if target_text not in values:
            values.append(target_text)

    def visit_hierarchy(row: Any) -> None:
        if not isinstance(row, Mapping):
            return
        row_id = str(row.get("id") or row.get("link_id") or row.get("node_id") or "")
        node_id = row.get("node_id")
        kind = str(row.get("kind") or "")
        if node_id not in (None, "") and row_id:
            if kind in {"code", "context"}:
                add_alias(node_id, row_id)
                add_alias(row_id, row_id)
            else:
                rpg_links.setdefault(str(node_id), row_id)
        for alias in _as_sequence(row.get("aliases")):
            add_alias(alias, row_id)
        for code_id in _as_sequence(row.get("mapped_code_node_ids")):
            add_alias(_graph_node_id({"node_id": code_id}, "code"), row_id)
            add_alias(code_id, row_id)
        for code_link in _as_sequence(row.get("mapped_code_link_ids")):
            add_alias(code_link, row_id)
        for code_ref in _as_sequence(row.get("mapped_code")):
            if isinstance(code_ref, Mapping):
                add_alias(code_ref.get("link_id"), row_id)
                add_alias(code_ref.get("node_id") or code_ref.get("dep_node_id"), row_id)
        for child in _as_sequence(row.get("children")):
            visit_hierarchy(child)

    visit_hierarchy(hierarchy)

    for node in semantic_nodes:
        link_id = str(node.get("link_id") or _graph_node_id(node, "feature"))
        node_id = node.get("node_id")
        if node_id not in (None, ""):
            rpg_links.setdefault(str(node_id), link_id)
    for node in code_nodes:
        link_id = str(node.get("link_id") or _graph_node_id(node, "code"))
        node_id = node.get("node_id") or node.get("dep_node_id")
        if node_id not in (None, ""):
            code_links[str(node_id)] = link_id
            for rpg_link in _as_sequence(node.get("mapped_rpg_link_ids")):
                add_alias(link_id, rpg_link)
                add_alias(node_id, rpg_link)

    for mapping in mappings:
        source = mapping.get("source_link_id") or rpg_links.get(str(mapping.get("rpg_node_id") or mapping.get("node_id") or ""))
        target = mapping.get("target_link_id") or code_links.get(str(mapping.get("code_node_id") or mapping.get("dep_node_id") or ""))
        add_alias(target, source)
        add_alias(mapping.get("code_node_id") or mapping.get("dep_node_id"), source)

    def edge_kind(edge: Mapping[str, Any]) -> str:
        source_parts = " ".join(
            str(edge.get(key) or "")
            for key in ("source_graph", "edge_source", "relation_source", "source")
        ).lower()
        relation = str(edge.get("relation") or "").lower()
        if "rpg" in source_parts or relation in {"semantic", "depends_on", "related", "relates_to"}:
            return "semantic"
        return "dependency"

    def candidates(edge: Mapping[str, Any], side: str) -> list[str]:
        values: list[Any] = []
        values.extend(_as_sequence(edge.get(f"{side}_candidates")))
        values.extend(_as_sequence(edge.get(f"{side}_rpg_link_ids")))
        values.append(edge.get(f"{side}_link_id"))
        node_id = edge.get(f"{side}_node_id")
        values.append(rpg_links.get(str(node_id or "")))
        values.append(code_links.get(str(node_id or "")))
        values.append(node_id)
        expanded: list[str] = []
        for value in values:
            if value in (None, ""):
                continue
            text = str(value)
            expanded.append(text)
            expanded.extend(node_aliases.get(text, []))
        return _ordered_texts(expanded)

    relation_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(context_edges, start=1):
        source_candidates = candidates(edge, "source")
        target_candidates = candidates(edge, "target")
        relation_edges.append({
            "id": str(edge.get("link_id") or f"edge-{index}"),
            "source": source_candidates[0] if source_candidates else "",
            "target": target_candidates[0] if target_candidates else "",
            "source_candidates": source_candidates,
            "target_candidates": target_candidates,
            "kind": edge_kind(edge),
            "relation": edge.get("relation") or "dependency",
            "source_meta": edge.get("source"),
            "source_graph": edge.get("source_graph"),
            "edge_source": edge.get("edge_source"),
            "relation_source": edge.get("relation_source"),
            "direction": edge.get("direction"),
            "reason": edge.get("reason"),
            "path": edge.get("path"),
        })

    summary.update({
        "semantic_nodes": len(semantic_nodes),
        "code_nodes": len(code_nodes),
        "mappings": len(mappings),
        "edges": len(context_edges),
        "relation_edges": len(relation_edges),
        "context_edges": len(context_edges),
    })
    links.extend(relation_edges)
    return {
        "schema": "cmind.focused_graph.render.v1",
        "summary": summary,
        "nodes": nodes,
        "links": links,
        "relation_edges": relation_edges,
        "semantic_nodes": semantic_nodes,
        "code_nodes": code_nodes,
        "mappings": mappings,
        "edges": context_edges,
        "hidden_counts": hidden_counts,
        "hidden_context_nodes": hidden_context_nodes,
        "warnings": warnings,
        "hierarchy": hierarchy,
        "default_focus": default_focus,
        "node_aliases": node_aliases,
    }


def _focused_graph_runtime() -> str:
    return r"""
(function(){
  const section = document.currentScript.closest('[data-focused-graph]');
  if (!section) return;
  const dataEl = section.querySelector('[data-focused-graph-json]');
  const svg = section.querySelector('[data-focused-graph-svg]');
  const stage = section.querySelector('.focused-graph-stage');
  const fallback = section.querySelector('[data-focused-graph-fallback]');
  const statusEl = section.querySelector('[data-focused-graph-status]');
  const detailEl = section.querySelector('[data-focused-graph-detail]');
  const fullscreenButton = section.querySelector('[data-action="fullscreen"]');
  if (!window.d3 || !dataEl || !svg) return;
  if (fallback) fallback.hidden = true;
  const data = JSON.parse(dataEl.textContent || '{}');
  const svgSelection = d3.select(svg);
  let width = 960;
  let height = 680;
  let isFullscreen = false;

  function refreshGraphViewport() {
    const svgBox = svg.getBoundingClientRect();
    const stageBox = stage ? stage.getBoundingClientRect() : {width: 0, height: 0};
    width = Math.max(320, Math.round(svgBox.width || stageBox.width || svg.clientWidth || 960));
    height = Math.max(360, Math.round(svgBox.height || stageBox.height || svg.clientHeight || 680));
    svgSelection.attr('viewBox', `0 0 ${width} ${height}`);
  }

  function updateFullscreenButton() {
    if (!fullscreenButton) return;
    fullscreenButton.textContent = isFullscreen ? 'Restore embedded' : 'Fullscreen';
    fullscreenButton.setAttribute('aria-pressed', isFullscreen ? 'true' : 'false');
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    section.classList.toggle('focused-graph-fullscreen', isFullscreen);
    document.body.classList.toggle('focused-graph-fullscreen-active', isFullscreen);
    updateFullscreenButton();
    scheduleResize();
  }

  refreshGraphViewport();
  const defaultFocus = data.default_focus || {};
  const defaultShowEdges = defaultFocus.show_edges !== false;
  const text = value => value === undefined || value === null ? '' : String(value);
  const list = value => Array.isArray(value) ? value : [];
  const nodePayloadById = {};
  list(data.nodes).forEach(node => {
    if (!node || typeof node !== 'object') return;
    const id = text(node.id);
    const nodeId = text(node.node_id || node.dep_node_id);
    if (id) nodePayloadById[id] = node;
    if (nodeId && !nodePayloadById[nodeId]) nodePayloadById[nodeId] = node;
  });
  const root = d3.hierarchy(data.hierarchy || {id:'focused-graph-root', name:'Focused graph', children:[]}, d => d.children);
  const rootHierarchyId = text(root.data.id || 'focused-graph-root');
  const focusedNodeIds = new Set(list(defaultFocus.focused_tree_node_ids || defaultFocus.focused_node_ids || defaultFocus.node_link_ids).map(text).filter(Boolean));
  const focusedCodeLinkIds = new Set(list(defaultFocus.focused_code_link_ids).map(text).filter(Boolean));
  const expandedNodeIds = new Set(list(defaultFocus.default_expanded_node_ids || defaultFocus.expanded_node_ids).map(text).filter(Boolean));
  const focusedPathNodeIds = new Set(list(defaultFocus.focused_path_node_ids).map(text).filter(Boolean));
  const defaultExpandedIds = new Set([rootHierarchyId, ...expandedNodeIds]);
  const isDefaultFocused = d => {
    const data = d?.data || {};
    return [d?.id, data.id, data.link_id, data.node_id, data.dep_node_id]
      .map(text)
      .filter(Boolean)
      .some(id => focusedNodeIds.has(id) || focusedCodeLinkIds.has(id));
  };
  let showEdges = defaultShowEdges;
  let query = '';
  let selectedId = null;
  const allNodeById = {};
  let nodeById = {};
  let currentNodes = [];
  let currentRelationEdges = [];

  root.x0 = height / 2;
  root.y0 = 0;
  root.descendants().forEach(d => {
    d.id = text(d.data.id || d.data.link_id || d.data.node_id || d.data.name);
    if (!d.id) d.id = `node-${Math.random().toString(36).slice(2)}`;
    d._allChildren = d.children || null;
    allNodeById[d.id] = d;
  });

  function walkAll(d, visit) {
    visit(d);
    list(d._allChildren).forEach(child => walkAll(child, visit));
  }

  function initializeDefaultState() {
    walkAll(root, d => {
      if (!d._allChildren) return;
      const keepOpen = d.depth === 0 || defaultExpandedIds.has(d.id) || focusedPathNodeIds.has(d.id);
      if (keepOpen) {
        d.children = d._allChildren;
        d._children = null;
      } else {
        d._children = d._allChildren;
        d.children = null;
      }
    });
  }
  initializeDefaultState();

  const relationEdges = list(data.relation_edges || data.links).filter(edge => text(edge.relation) !== 'contains');
  const treemap = d3.tree().nodeSize([28, 250]);
  const graphOffsetX = 80;
  const graphOffsetY = 96;
  svgSelection.selectAll('*').remove();
  const graphLayer = svgSelection.append('g').attr('class', 'focused-graph-layer').attr('transform', `translate(${graphOffsetX},${graphOffsetY})`);
  const relationLayer = graphLayer.append('g').attr('class', 'focused-graph-relation-links');
  const treeLinkLayer = graphLayer.append('g').attr('class', 'focused-graph-tree-links');
  const nodeLayer = graphLayer.append('g').attr('class', 'focused-graph-nodes');
  const zoom = d3.zoom().scaleExtent([0.25, 3.5]).on('zoom', event => graphLayer.attr('transform', event.transform));
  svgSelection.call(zoom).on('dblclick.zoom', null).on('click', event => {
    if (event.target === svg) {
      selectedId = null;
      renderFocusedGraphDetail(null);
      update(root);
    }
  });

  function nodeDetailData(d) {
    if (!d) return {};
    const data = d.data || {};
    const payload = nodePayloadById[text(d.id)] || nodePayloadById[text(data.node_id || data.dep_node_id)] || {};
    return {...payload, ...data};
  }

  function nodeLabel(d) {
    const detail = nodeDetailData(d);
    return text(detail.feature_name || detail.name || detail.node_id || detail.dep_node_id || detail.id || d.id || 'node');
  }

  function searchText(value) {
    if (Array.isArray(value)) return value.map(searchText).join(' ');
    if (value && typeof value === 'object') return Object.values(value).map(searchText).join(' ');
    return text(value);
  }

  function nodeSearchText(d) {
    return `${nodeLabel(d)} ${searchText(nodeDetailData(d))}`.toLowerCase();
  }

  function nodeMatches(d) {
    return query && nodeSearchText(d).includes(query);
  }

  function isPresent(value) {
    if (value === undefined || value === null || value === '') return false;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  }

  function escapeHtml(value) {
    return text(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[char]));
  }

  function uniqueTexts(values) {
    const seen = new Set();
    const result = [];
    values.forEach(value => {
      const item = text(value).trim();
      if (!item || seen.has(item)) return;
      seen.add(item);
      result.push(item);
    });
    return result;
  }

  function detailItems(value) {
    if (!isPresent(value)) return [];
    return Array.isArray(value) ? value : [value];
  }

  function lineRangeText(value) {
    if (!isPresent(value)) return '';
    if (Array.isArray(value)) return value.map(text).filter(Boolean).join('-');
    if (value && typeof value === 'object') {
      const start = value.start ?? value.start_line ?? value.line_start;
      const end = value.end ?? value.end_line ?? value.line_end;
      if (isPresent(start) && isPresent(end)) return `${start}-${end}`;
      return searchText(value).trim();
    }
    return text(value);
  }

  function simpleValueHtml(value) {
    if (!isPresent(value)) return '';
    if (Array.isArray(value)) return detailListHtml(value, simpleValueHtml);
    if (value && typeof value === 'object') return `<code>${escapeHtml(searchText(value).trim() || JSON.stringify(value))}</code>`;
    return escapeHtml(value);
  }

  function detailListHtml(value, render) {
    const rows = detailItems(value).map(render).filter(Boolean);
    return rows.length ? `<ul class="focused-graph-detail-list">${rows.map(row => `<li>${row}</li>`).join('')}</ul>` : '';
  }

  function changedFilesHtml(value) {
    return detailListHtml(value, item => {
      if (item && typeof item === 'object') {
        const label = text(item.path || item.file || item.diff_anchor || item.href);
        const href = text(item.href || (item.diff_anchor ? `#${item.diff_anchor}` : ''));
        if (!label) return '';
        return href ? `<a href="${escapeHtml(href)}">${escapeHtml(label)}</a>` : escapeHtml(label);
      }
      return escapeHtml(item);
    });
  }

  function diffHtml(value, detail) {
    if (value && typeof value === 'object') {
      const label = text(value.path || value.href || 'View diff');
      const href = text(value.href || (value.diff_anchor ? `#${value.diff_anchor}` : ''));
      return href ? `<a href="${escapeHtml(href)}">${escapeHtml(label)}</a>` : escapeHtml(label);
    }
    if (detail.diff_anchor) return `<a href="#${escapeHtml(detail.diff_anchor)}">${escapeHtml(detail.path || detail.diff_anchor)}</a>`;
    return '';
  }

  function canonicalMappedCodeRefs(value, detail) {
    let refs = detailItems(value);
    if (!refs.length) {
      const paths = list(detail.mapped_code_paths);
      const symbols = list(detail.mapped_code_symbols);
      const count = Math.max(paths.length, symbols.length, detail.mapped_code_path ? 1 : 0, detail.mapped_code_symbol ? 1 : 0);
      refs = Array.from({length: count}, (_, index) => ({
        path: index === 0 ? (detail.mapped_code_path || paths[index]) : paths[index],
        symbol: index === 0 ? (detail.mapped_code_symbol || symbols[index]) : symbols[index],
      }));
    }
    const seenRefs = new Set();
    const canonical = [];
    refs.forEach(item => {
      const ref = item && typeof item === 'object'
        ? {
            path: text(item.path || item.file || item.module),
            symbol: text(item.symbol || item.name),
            node_id: text(item.node_id || item.dep_node_id || item.link_id),
            type: text(item.type || item.kind),
            line_range: lineRangeText(item.line_range),
          }
        : {path: '', symbol: '', node_id: text(item), type: '', line_range: ''};
      const label = uniqueTexts([ref.path, ref.symbol]).join(' · ') || ref.node_id;
      const key = uniqueTexts([ref.path, ref.symbol]).join('::') || ref.node_id || label;
      if (!label || seenRefs.has(key)) return;
      seenRefs.add(key);
      canonical.push({label, meta: uniqueTexts([ref.type, ref.line_range]).join(' · ')});
    });
    return canonical;
  }

  function mappedCodeHtml(value, detail) {
    return detailListHtml(canonicalMappedCodeRefs(value, detail), ref => `${escapeHtml(ref.label)}${ref.meta ? `<span>${escapeHtml(ref.meta)}</span>` : ''}`);
  }

  function addValueRow(rows, label, value) {
    if (!isPresent(value)) return;
    rows.push(`<div class="focused-graph-detail-row"><dt>${escapeHtml(label)}</dt><dd>${simpleValueHtml(value)}</dd></div>`);
  }

  function addHtmlRow(rows, label, value) {
    if (!value) return;
    rows.push(`<div class="focused-graph-detail-row"><dt>${escapeHtml(label)}</dt><dd>${value}</dd></div>`);
  }

  function isCodeContextDetail(detail) {
    return uniqueTexts([detail.kind, detail.type, detail.node_type])
      .map(item => item.toLowerCase())
      .some(item => ['code', 'context', 'code_group', 'context_group'].includes(item));
  }

  function renderFocusedGraphDetail(d) {
    if (!detailEl) return;
    if (!d) {
      detailEl.innerHTML = '<h3>Node details</h3><p class="empty">Select a node to inspect metadata.</p>';
      return;
    }
    const detail = nodeDetailData(d);
    const rows = [];
    const typeText = uniqueTexts([detail.node_type || detail.type || detail.kind, detail.state, detail.mapping_status, detail.locate_status]).join(' · ');
    const relationText = uniqueTexts([detail.relation, detail.direction]).join(' · ');
    const sourceText = uniqueTexts([detail.source, detail.source_graph, detail.edge_source, detail.relation_source]).join(' · ');
    addValueRow(rows, 'Node id', detail.node_id || detail.dep_node_id || detail.id);
    if (!isCodeContextDetail(detail)) addValueRow(rows, 'Feature path', detail.breadcrumb_path || detail.feature_path);
    addValueRow(rows, 'Path', detail.path || detail.module || detail.file);
    addValueRow(rows, 'Symbol', detail.symbol);
    addValueRow(rows, 'Type', typeText);
    addValueRow(rows, 'Relation', relationText);
    addValueRow(rows, 'Source', sourceText);
    addValueRow(rows, 'Reason', detail.reason);
    addValueRow(rows, 'Lines', lineRangeText(detail.line_range));
    addHtmlRow(rows, 'Mapped code', mappedCodeHtml(detail.mapped_code, detail));
    addHtmlRow(rows, 'Changed files', changedFilesHtml(detail.changed_files || detail.affected_files));
    addHtmlRow(rows, 'Diff', diffHtml(detail.diff, detail));
    const chips = uniqueTexts([detail.kind, detail.state, detail.mapping_status, detail.source]).map(item => `<span class="badge">${escapeHtml(item)}</span>`).join('');
    const body = rows.length ? `<dl>${rows.join('')}</dl>` : '<p class="empty">No additional metadata for this node.</p>';
    detailEl.innerHTML = `<h3>${escapeHtml(nodeLabel(d))}</h3>${chips ? `<div class="focused-graph-detail-badges">${chips}</div>` : ''}${body}`;
  }

  function diagonal(s, d) {
    return `M${s.y},${s.x} C${(s.y + d.y) / 2},${s.x} ${(s.y + d.y) / 2},${d.x} ${d.y},${d.x}`;
  }

  function openAncestors(d) {
    let p = d.parent;
    while (p) {
      if (p._children) { p.children = p._children; p._children = null; }
      p = p.parent;
    }
  }

  function applySearchOpen() {
    if (!query) return;
    walkAll(root, d => { if (nodeMatches(d)) openAncestors(d); });
  }

  function resetDefault() {
    query = '';
    selectedId = null;
    showEdges = defaultShowEdges;
    initializeDefaultState();
    const search = section.querySelector('[data-action="search"]');
    if (search) search.value = '';
    const edges = section.querySelector('[data-action="edges"]');
    if (edges) edges.checked = showEdges;
    renderFocusedGraphDetail(null);
    svgSelection.transition().duration(150).call(zoom.transform, d3.zoomIdentity.translate(graphOffsetX, graphOffsetY));
    update(root);
  }

  function cssToken(value) {
    const token = text(value).toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
    return token || 'unknown';
  }

  function edgeClass(edge) {
    const kind = text(edge.kind || edge.source_graph).toLowerCase().includes('rpg') || text(edge.kind).toLowerCase() === 'semantic' ? 'edge-semantic' : 'edge-dependency';
    const relation = `relation-${cssToken(edge.relation || 'dependency')}`;
    const source = `source-${cssToken(edge.source_graph || edge.edge_source || edge.source_meta || kind)}`;
    return `${kind} ${relation} ${source}`;
  }

  function visibleEndpoint(edge, side) {
    for (const candidate of list(edge[`${side}_candidates`]).map(text)) {
      let node = allNodeById[candidate] || nodeById[candidate];
      while (node) {
        if (nodeById[node.id]) return nodeById[node.id];
        node = node.parent;
      }
    }
    let node = allNodeById[text(edge[side])] || nodeById[text(edge[side])];
    while (node) {
      if (nodeById[node.id]) return nodeById[node.id];
      node = node.parent;
    }
    return null;
  }

  function relationPath(source, target) {
    const sx = source.y + 8, sy = source.x;
    const dx = target.y - 8, dy = target.x;
    const midX = Math.max(sx, dx) + 56 + Math.abs(sy - dy) * 0.18;
    return `M${sx},${sy} Q${midX},${(sy + dy) / 2} ${dx},${dy}`;
  }

  function updateStatus() {
    if (!statusEl) return;
    const visible = showEdges ? currentRelationEdges.length : 0;
    const total = relationEdges.length;
    statusEl.textContent = `Visible relation edges: ${visible}/${total}`;
  }

  function drawRelationEdges() {
    currentRelationEdges = [];
    if (showEdges) {
      relationEdges.forEach(edge => {
        const source = visibleEndpoint(edge, 'source');
        const target = visibleEndpoint(edge, 'target');
        if (!source || !target || source === target) return;
        currentRelationEdges.push({...edge, _source: source, _target: target, _key: `${edge.id || edge.relation}-${source.id}-${target.id}`});
      });
    }
    const rel = relationLayer.selectAll('path.focused-graph-link').data(currentRelationEdges, edge => edge._key);
    const relEnter = rel.enter().append('path')
      .attr('class', edge => `focused-graph-link ${edgeClass(edge)}`)
      .attr('data-link-id', edge => text(edge.id || edge._key))
      .attr('d', edge => relationPath(edge._source, edge._target));
    relEnter.append('title');
    const relUpdate = relEnter.merge(rel)
      .attr('class', edge => `focused-graph-link ${edgeClass(edge)}${selectedId && (edge._source.id === selectedId || edge._target.id === selectedId) ? ' active' : ''}${selectedId && edge._source.id !== selectedId && edge._target.id !== selectedId ? ' dimmed' : ''}`)
      .attr('d', edge => relationPath(edge._source, edge._target));
    relUpdate.select('title').text(edge => `${edge.relation || 'dependency'}\n${edge.source_graph || edge.edge_source || edge.source_meta || edge.kind || ''}\n${edge.path || edge.reason || ''}`);
    rel.exit().remove();
    updateStatus();
  }

  function update(source) {
    applySearchOpen();
    const layout = treemap(root);
    currentNodes = layout.descendants();
    const maxDepth = d3.max(currentNodes, d => d.depth) || 1;
    const depthGap = Math.max(140, Math.min(250, (width - 260) / maxDepth));
    currentNodes.forEach(d => { d.y = d.depth * depthGap; });
    const minX = d3.min(currentNodes, d => d.x) || 0;
    if (minX < 20) currentNodes.forEach(d => { d.x += 20 - minX; });
    nodeById = {};
    currentNodes.forEach(d => { nodeById[d.id] = d; });

    const node = nodeLayer.selectAll('g.focused-graph-node').data(currentNodes, d => d.id);
    const nodeEnter = node.enter().append('g')
      .attr('class', 'focused-graph-node')
      .attr('data-node-id', d => d.id)
      .attr('tabindex', 0)
      .attr('role', 'button')
      .attr('transform', `translate(${source.y0 || 0},${source.x0 || 0})`)
      .on('click', (event, d) => {
        event.stopPropagation();
        selectedId = selectedId === d.id ? null : d.id;
        renderFocusedGraphDetail(selectedId ? d : null);
        update(d);
      })
      .on('dblclick', (event, d) => {
        event.stopPropagation();
        if (d.children) { d._children = d.children; d.children = null; }
        else if (d._children) { d.children = d._children; d._children = null; }
        update(d);
      });
    nodeEnter.append('circle').attr('r', d => d._children ? 6 : (d.children ? 5 : 4));
    nodeEnter.append('text')
      .attr('x', d => d.children || d._children ? 0 : 10)
      .attr('y', d => d.children || d._children ? 14 : 0)
      .attr('dy', d => d.children || d._children ? 8 : 4)
      .attr('text-anchor', d => d.children || d._children ? 'middle' : 'start')
      .attr('font-size', 12)
      .attr('fill', '#e5e7eb')
      .text(d => nodeLabel(d).length > 46 ? nodeLabel(d).slice(0, 44) + '…' : nodeLabel(d));
    nodeEnter.append('title').text(d => {
      const detail = nodeDetailData(d);
      return `${nodeLabel(d)}\n${detail.feature_path || ''}\n${detail.path || ''} ${detail.symbol || ''}`;
    });

    const nodeUpdate = nodeEnter.merge(node);
    nodeUpdate.transition().duration(250).attr('transform', d => `translate(${d.y},${d.x})`);
    nodeUpdate
      .attr('class', d => `focused-graph-node${d.id === selectedId ? ' selected' : ''}${isDefaultFocused(d) ? ' focused' : ' non-focused'}${nodeMatches(d) ? ' search-match' : ''}${selectedId && d.id !== selectedId && !currentRelationEdges.some(edge => edge._source.id === d.id || edge._target.id === d.id) ? ' dimmed' : ''}`);
    nodeUpdate.select('circle').attr('r', d => d._children ? 6 : (d.children ? 5 : 4));
    nodeUpdate.select('text')
      .attr('x', d => d.children || d._children ? 0 : 10)
      .attr('y', d => d.children || d._children ? 14 : 0)
      .attr('dy', d => d.children || d._children ? 8 : 4)
      .attr('text-anchor', d => d.children || d._children ? 'middle' : 'start');
    node.exit().transition().duration(180).attr('transform', `translate(${source.y || 0},${source.x || 0})`).remove();

    const treeLinks = layout.links();
    const link = treeLinkLayer.selectAll('path.focused-graph-tree-link').data(treeLinks, d => d.target.id);
    link.enter().insert('path', 'g')
      .attr('class', 'focused-graph-tree-link')
      .attr('d', () => diagonal({x: source.x0 || 0, y: source.y0 || 0}, {x: source.x0 || 0, y: source.y0 || 0}))
      .merge(link).transition().duration(250)
      .attr('d', d => diagonal(d.source, d.target));
    link.exit().transition().duration(180)
      .attr('d', () => diagonal({x: source.x || 0, y: source.y || 0}, {x: source.x || 0, y: source.y || 0}))
      .remove();

    drawRelationEdges();
    renderFocusedGraphDetail(selectedId ? (nodeById[selectedId] || allNodeById[selectedId]) : null);
    currentNodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
  }

  let resizeFrame = null;
  function scheduleResize() {
    if (resizeFrame !== null) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = null;
      const previousWidth = width;
      const previousHeight = height;
      refreshGraphViewport();
      if (width !== previousWidth || height !== previousHeight) update(root);
    });
  }
  if (window.ResizeObserver) {
    const resizeObserver = new ResizeObserver(scheduleResize);
    resizeObserver.observe(svg);
    if (stage) resizeObserver.observe(stage);
  } else {
    window.addEventListener('resize', scheduleResize);
  }

  updateFullscreenButton();
  section.querySelector('[data-action="reset"]')?.addEventListener('click', resetDefault);
  fullscreenButton?.addEventListener('click', toggleFullscreen);
  section.querySelector('[data-action="edges"]')?.addEventListener('change', event => { showEdges = event.target.checked; update(root); });
  section.querySelector('[data-action="search"]')?.addEventListener('input', event => { query = text(event.target.value).toLowerCase(); update(root); });
  update(root);
})();
"""


def _render_focused_graph(focused_view: dict[str, Any], file_anchors: Mapping[str, str]) -> str:
    nodes_view = focused_view.get("nodes_view") if isinstance(focused_view.get("nodes_view"), Mapping) else {}
    if not focused_view and not nodes_view:
        return ""
    graph_payload = _focused_graph_payload(focused_view, file_anchors)
    summary = graph_payload.get("summary") if isinstance(graph_payload.get("summary"), Mapping) else {}
    summary_html = _summary_badges(summary, [
        ("Semantic nodes", "semantic_nodes", len(_as_sequence(nodes_view.get("semantic_nodes")))),
        ("Code nodes", "code_nodes", len(_as_sequence(nodes_view.get("code_nodes")))),
        ("Mappings", "mappings", len(_as_sequence(nodes_view.get("mappings")))),
        ("Edges", "edges", len(_as_sequence(nodes_view.get("edges")))),
        ("Warnings", "warnings", len(_as_sequence(nodes_view.get("warnings")))),
    ])
    hidden_counts = nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else focused_view.get("hidden_counts", {})
    hidden_context_nodes = [node for node in _as_sequence(nodes_view.get("hidden_context_nodes") or focused_view.get("hidden_context_nodes")) if isinstance(node, Mapping)]
    hidden_html = _hidden_context_html(hidden_counts if isinstance(hidden_counts, Mapping) else {}, hidden_context_nodes)
    warnings = [warning for warning in _as_sequence(nodes_view.get("warnings") or focused_view.get("warnings")) if isinstance(warning, Mapping)]
    warnings_html = f"<details><summary>Warnings</summary>{_chain_warning_html(warnings)}</details>" if warnings else ""
    relation_edges_value = graph_payload.get("relation_edges")
    if relation_edges_value is None:
        relation_edges_value = graph_payload.get("edges")
    relation_edge_total = len(_as_sequence(relation_edges_value))
    graph_json = _json_for_script(graph_payload)
    d3_js = _inline_d3()
    fallback_hidden = " hidden" if d3_js else ""
    d3_missing_note = "" if d3_js else '<p class="reason">Local D3 asset missing; showing the static fallback.</p>'
    scripts = ""
    if d3_js:
        scripts = f"<script>{d3_js}</script><script>{_focused_graph_runtime()}</script>"
    inspector_payload = {"focused_graph": graph_payload, "nodes_view": nodes_view}
    inspector = json.dumps(inspector_payload, indent=2, ensure_ascii=False, default=_json_default)
    controls = (
        '<div class="focused-graph-toolbar">'
        '<button type="button" data-action="reset">Reset default</button>'
        '<button type="button" data-action="fullscreen" aria-pressed="false">Fullscreen</button>'
        '<label><input type="checkbox" data-action="edges" checked> Edges</label>'
        '<input type="search" data-action="search" placeholder="Search nodes" aria-label="Search focused graph nodes">'
        f'<span class="badge" data-focused-graph-status>Visible relation edges: 0/{relation_edge_total}</span>'
        '</div>'
    )
    legend = (
        '<div class="focused-graph-legend" aria-label="Focused graph legend">'
        '<span class="legend-item"><span class="legend-swatch legend-node"></span>Feature tree node</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-tree-link"></span>Tree link</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-semantic-edge"></span>RPG semantic edge</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-dependency-edge legend-dep-graph-edge"></span>dep_graph dependency edge</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-invokes-edge"></span>invokes</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-imports-edge legend-dep-graph-edge"></span>imports</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-inherits-edge"></span>inherits</span>'
        '<span class="legend-item"><span class="legend-swatch legend-line legend-references-edge"></span>references</span>'
        '</div>'
    )
    return (
        '<section class="focused-graph-section" data-focused-graph><h2>Focused graph</h2>'
        f"{summary_html}"
        f"<script type=\"application/json\" data-focused-graph-json>{graph_json}</script>"
        '<div class="focused-graph-stage">'
        f"{controls}{legend}"
        '<svg class="focused-graph-svg" data-focused-graph-svg role="img" aria-label="Focused graph view"></svg>'
        '<aside class="focused-graph-detail" data-focused-graph-detail aria-live="polite"><h3>Node details</h3><p class="empty">Select a node to inspect metadata.</p></aside>'
        f'<div class="focused-graph-fallback" data-focused-graph-fallback{fallback_hidden}>Static focused graph fallback is available when D3 cannot run.{d3_missing_note}</div>'
        '</div>'
        f"{warnings_html}{hidden_html}"
        f"<details><summary>Inspector JSON</summary><pre>{_h(inspector)}</pre></details>"
        f"{scripts}"
        '</section>'
    )


def _legacy_render_focused_nodes_map(focused_view: dict[str, Any], file_anchors: Mapping[str, str]) -> str:
    nodes_view = focused_view.get("nodes_view") if isinstance(focused_view.get("nodes_view"), Mapping) else {}
    if not nodes_view:
        return ""
    summary = nodes_view.get("summary") if isinstance(nodes_view.get("summary"), Mapping) else {}
    summary_html = _summary_badges(summary, [
        ("Semantic nodes", "semantic_nodes", len(_as_sequence(nodes_view.get("semantic_nodes")))),
        ("Code nodes", "code_nodes", len(_as_sequence(nodes_view.get("code_nodes")))),
        ("Mappings", "mappings", len(_as_sequence(nodes_view.get("mappings")))),
        ("Edges", "edges", len(_as_sequence(nodes_view.get("edges")))),
        ("Warnings", "warnings", len(_as_sequence(nodes_view.get("warnings")))),
    ])
    semantic_cards = [
        _semantic_card(node, file_anchors)
        for node in _as_sequence(nodes_view.get("semantic_nodes"))
        if isinstance(node, Mapping)
    ]
    code_cards = [
        _code_card(node, file_anchors)
        for node in _as_sequence(nodes_view.get("code_nodes"))
        if isinstance(node, Mapping)
    ]
    mapping_cards = [
        _mapping_card(mapping)
        for mapping in _as_sequence(nodes_view.get("mappings"))
        if isinstance(mapping, Mapping)
    ]
    hidden_html = _hidden_context_html(
        nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else {},
        _as_sequence(nodes_view.get("hidden_context_nodes")),
    )
    warnings = [warning for warning in _as_sequence(nodes_view.get("warnings")) if isinstance(warning, Mapping)]
    warnings_html = f"<h3>Warnings</h3>{_chain_warning_html(warnings)}" if warnings else ""
    body = summary_html
    if semantic_cards or code_cards or mapping_cards:
        body += '<div class="focus-map">' + "".join(semantic_cards + code_cards + mapping_cards) + "</div>"
    else:
        body += '<p class="empty">No focused nodes map rows recorded.</p>'
    body += hidden_html + warnings_html + _focused_graph_metadata(focused_view)
    return f"<section><h2>Focused nodes map</h2>{body}</section>"


def _render_feature_chain_rows(focused_view: dict[str, Any], file_anchors: Mapping[str, str]) -> str:
    rpg_nodes = [node for node in _as_sequence(focused_view.get("primary_rpg_nodes")) if isinstance(node, Mapping)]
    code_nodes = [node for node in _as_sequence(focused_view.get("primary_code_nodes")) if isinstance(node, Mapping)]
    mappings = [mapping for mapping in _as_sequence(focused_view.get("mappings")) if isinstance(mapping, Mapping)]
    edges = [edge for edge in _as_sequence(focused_view.get("edges")) if isinstance(edge, Mapping)]
    warnings = [warning for warning in _as_sequence(focused_view.get("warnings")) if isinstance(warning, Mapping)]
    if not rpg_nodes and not mappings and not code_nodes:
        return "<p class=\"empty\">No semantic-code impact chain rows recorded.</p>"

    rpg_by_id = {str(node.get("node_id")): node for node in rpg_nodes if node.get("node_id") not in (None, "")}
    code_by_id = {
        str(node.get("node_id") or node.get("dep_node_id")): node
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }
    mappings_by_rpg: dict[str, list[Mapping[str, Any]]] = {}
    mapped_code_ids_by_rpg: dict[str, set[str]] = {}
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id") or ""
        rpg_text = str(rpg_id)
        mappings_by_rpg.setdefault(rpg_text, []).append(mapping)
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        if code_id not in (None, ""):
            mapped_code_ids_by_rpg.setdefault(rpg_text, set()).add(str(code_id))
        if rpg_text and rpg_text not in rpg_by_id:
            rpg_by_id[rpg_text] = {"node_id": rpg_text, "mapping_status": mapping.get("status", "")}
    edge_rows_by_rpg: dict[str, list[Mapping[str, Any]]] = {str(node_id): [] for node_id in rpg_by_id}
    for edge in edges:
        rpg_id = edge.get("rpg_node_id")
        matched = False
        if rpg_id not in (None, "") and str(rpg_id) in edge_rows_by_rpg:
            edge_rows_by_rpg[str(rpg_id)].append(edge)
            matched = True
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        for node_id, code_ids in mapped_code_ids_by_rpg.items():
            if source in code_ids or target in code_ids:
                edge_rows_by_rpg.setdefault(node_id, []).append(edge)
                matched = True
        if not matched and len(edge_rows_by_rpg) == 1:
            only_id = next(iter(edge_rows_by_rpg))
            edge_rows_by_rpg[only_id].append(edge)

    warnings_by_rpg: dict[str, list[Mapping[str, Any]]] = {str(node_id): [] for node_id in rpg_by_id}
    global_warnings = []
    for warning in warnings:
        node_id = warning.get("node_id") or warning.get("rpg_node_id")
        if node_id not in (None, "") and str(node_id) in warnings_by_rpg:
            warnings_by_rpg[str(node_id)].append(warning)
        else:
            global_warnings.append(warning)

    rows = []
    for node_id, rpg_node in rpg_by_id.items():
        node_mappings = mappings_by_rpg.get(node_id) or []
        mapping_items = []
        changed_files: list[Any] = list(_as_sequence(rpg_node.get("changed_files")))
        for mapping in node_mappings:
            code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
            code_node = code_by_id.get(str(code_id)) if code_id not in (None, "") else None
            changed_files.extend(_mapping_changed_files(mapping, rpg_node))
            status = mapping.get("status") or rpg_node.get("mapping_status") or "recorded"
            source = mapping.get("source", "")
            reason = mapping.get("reason") or rpg_node.get("reason") or ""
            path = mapping.get("path") or (code_node or {}).get("path") or ""
            mapping_items.append(
                "<li>"
                f"<span class=\"badge\">{_h(status)}</span> {_focused_node_cell(code_id, code_node)}"
                f"<div class=\"reason\">{_h(source)} {_h(path)} {_h(reason)}</div>"
                "</li>"
            )
        if not mapping_items:
            mapping_items.append(f"<li><span class=\"badge\">{_h(rpg_node.get('mapping_status') or rpg_node.get('status') or 'recorded')}</span> <span class=\"empty\">No mapped code node.</span></li>")
        mapping_html = '<ul class="hit-list">' + "".join(mapping_items) + "</ul>"
        node_hidden = rpg_node.get("hidden_counts") if isinstance(rpg_node.get("hidden_counts"), Mapping) else {}
        hidden_html = _hidden_context_html(node_hidden)
        rows.append(
            "<tr>"
            f"<td>{_focused_node_cell(node_id, rpg_node)}<div class=\"reason\">{_h(rpg_node.get('reason', ''))}</div></td>"
            f"<td>{mapping_html}</td>"
            f"<td>{_changed_file_links(changed_files or _as_sequence(rpg_node.get('affected_files')), file_anchors)}</td>"
            f"<td>{_chain_edge_html(edge_rows_by_rpg.get(node_id, []))}{hidden_html}</td>"
            f"<td>{_chain_warning_html(warnings_by_rpg.get(node_id, []))}</td>"
            "</tr>"
        )

    orphan_code_nodes = [
        node for code_id, node in code_by_id.items()
        if not any(code_id in ids for ids in mapped_code_ids_by_rpg.values())
    ]
    if orphan_code_nodes:
        orphan_items = "".join(
            f"<li>{_focused_node_cell(node.get('node_id') or node.get('dep_node_id'), node)} <span class=\"badge\">{_h(node.get('source', ''))}</span></li>"
            for node in orphan_code_nodes
        )
        rows.append(
            "<tr>"
            "<td><span class=\"empty\">Changed code without selected feature</span></td>"
            f"<td><ul class=\"hit-list\">{orphan_items}</ul></td>"
            "<td><span class=\"empty\">No changed files mapped.</span></td>"
            "<td><span class=\"empty\">No visible neighborhood edges.</span></td>"
            "<td><span class=\"empty\">No warnings.</span></td>"
            "</tr>"
        )

    table = (
        "<table><thead><tr><th>Feature group</th><th>Semantic → code evidence</th>"
        "<th>Changed files</th><th>Neighborhood</th><th>Warnings</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    if global_warnings:
        table += f"<h3>Global warnings</h3>{_chain_warning_html(global_warnings)}"
    return table


def _render_legacy_chain_rows(
    retrievals: list[dict[str, Any]],
    rpg_nodes: list[dict[str, Any]],
    dep_nodes: list[dict[str, Any]],
) -> str:
    blocks = []
    if retrievals:
        blocks.append(_render_retrievals(retrievals, title="Retrieval audit trail", as_section=False))
    if rpg_nodes or dep_nodes:
        rows = []
        dep_by_feature: dict[str, list[dict[str, Any]]] = {}
        for dep_node in dep_nodes:
            source_feature = str(dep_node.get("source_feature") or "")
            dep_by_feature.setdefault(source_feature, []).append(dep_node)
        for node in rpg_nodes or [{"node_id": ""}]:
            node_id = str(node.get("node_id") or "")
            mapped = dep_by_feature.get(node_id) or ([] if node_id else dep_nodes)
            mapping_html = "<span class=\"empty\">No mapped code nodes recorded.</span>"
            if mapped:
                items = []
                for dep_node in mapped:
                    dep_id = dep_node.get("dep_node_id") or dep_node.get("node_id")
                    items.append(
                        "<li>"
                        f"<code>{_h(dep_id)}</code> {_h(dep_node.get('path', ''))}"
                        f" <span class=\"badge\">{_h(dep_node.get('relation') or dep_node.get('change') or dep_node.get('status', ''))}</span>"
                        "</li>"
                    )
                mapping_html = '<ul class="hit-list">' + "".join(items) + "</ul>"
            rows.append(
                "<tr>"
                f"<td>{_focused_node_cell(node_id, node)}</td>"
                f"<td>{mapping_html}</td>"
                "<td><span class=\"empty\">No changed files mapped.</span></td>"
                "<td><span class=\"empty\">No visible neighborhood edges.</span></td>"
                "<td><span class=\"empty\">No warnings.</span></td>"
                "</tr>"
            )
        blocks.append(
            "<table><thead><tr><th>Feature group</th><th>Semantic → code evidence</th>"
            "<th>Changed files</th><th>Neighborhood</th><th>Warnings</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return "".join(blocks)


def _legacy_render_semantic_code_impact_chain(
    retrievals: list[dict[str, Any]],
    rpg_nodes: list[dict[str, Any]],
    dep_nodes: list[dict[str, Any]],
    focused_view: dict[str, Any],
    file_anchors: Mapping[str, str],
) -> str:
    if not focused_view and not retrievals and not rpg_nodes and not dep_nodes:
        return ""
    if focused_view:
        summary = focused_view.get("summary") if isinstance(focused_view.get("summary"), Mapping) else {}
        summary_html = _summary_badges(summary, [
            ("Selected feature groups", "selected_feature_groups", len(_as_sequence(focused_view.get("primary_rpg_nodes")))),
            ("Primary code nodes", "primary_code_nodes", len(_as_sequence(focused_view.get("primary_code_nodes")))),
            ("Mapped relations", "mapped_code_relations", len(_as_sequence(focused_view.get("mappings")))),
            ("Missing mappings", "missing_mappings", 0),
            ("Edges shown", "edges", len(_as_sequence(focused_view.get("edges")))),
            ("Warnings", "warnings", len(_as_sequence(focused_view.get("warnings")))),
        ])
        hidden_counts = focused_view.get("hidden_counts") if isinstance(focused_view.get("hidden_counts"), Mapping) else {}
        hidden_html = _hidden_context_html(hidden_counts)
        inspector_html = "" if isinstance(focused_view.get("nodes_view"), Mapping) else _focused_graph_metadata(focused_view)
        body = f"{summary_html}{_render_feature_chain_rows(focused_view, file_anchors)}{hidden_html}{inspector_html}"
    else:
        body = _render_legacy_chain_rows(retrievals, rpg_nodes, dep_nodes)
    return f"<section><h2>semantic-code impact chain</h2>{body}</section>"


def _focused_graph_metadata(focused_view: dict[str, Any]) -> str:
    data = json.dumps(_focused_inspector_payload(focused_view), indent=2, ensure_ascii=False, default=_json_default)
    return f"<details><summary>Inspector JSON</summary><pre>{_h(data)}</pre></details>"


def _focused_node_cell(node_id: Any, node: Mapping[str, Any] | None) -> str:
    if node_id in (None, "") and not node:
        return "<span class=\"empty\">missing</span>"
    node = node or {}
    parts = []
    if node_id not in (None, ""):
        parts.append(f"<code>{_h(node_id)}</code>")
    if node.get("name"):
        parts.append(f"<div>{_h(node.get('name'))}</div>")
    if node.get("path"):
        parts.append(f"<div class=\"reason\">{_h(node.get('path'))}</div>")
    return "".join(parts) or "<span class=\"empty\">missing</span>"


def _render_artifacts(artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        body = "<p class=\"empty\">No artifact links recorded.</p>"
    else:
        rows = []
        for artifact in artifacts:
            path = artifact.get("path")
            href = _artifact_href(path)
            rows.append(
                "<tr>"
                f"<td>{_h(artifact.get('label', 'artifact'))}</td>"
                f"<td><a href=\"{_h_attr(href)}\">{_h(path or '')}</a></td>"
                f"<td>{_h(_artifact_status(path, artifact.get('status')))}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>Artifact</th><th>Path</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<section><h2>Artifact links</h2>{body}</section>"


def _compact_artifact_pointers(value: Any) -> list[dict[str, Any]]:
    pointers = []
    for item in _as_sequence(value):
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in ("label", "path", "status"):
            if item.get(key) not in (None, ""):
                row[key] = item.get(key)
        if row:
            pointers.append(row)
    return pointers


def _compact_change_summary(value: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _as_sequence(value):
        if not isinstance(item, Mapping):
            continue
        diff = item.get("diff") or ""
        row: dict[str, Any] = {"file": item.get("file") or item.get("path") or ""}
        if item.get("change_type") not in (None, ""):
            row["change_type"] = item.get("change_type")
        row["has_diff"] = bool(diff)
        if diff:
            row["diff_lines"] = len(str(diff).splitlines())
        rows.append(row)
    return rows


def _compact_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, Mapping):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"code_deltas", "focused_view", "nodes_view", "focused_impact", "focused_graph", "evidence"}:
                continue
            if key_text == "diff":
                compacted["has_diff"] = bool(item)
                if item:
                    compacted["diff_lines"] = len(str(item).splitlines())
                continue
            if key_text == "artifacts":
                artifact_pointers = _compact_artifact_pointers(item)
                if artifact_pointers:
                    compacted["artifact_paths"] = artifact_pointers
                continue
            compact_item = _compact_payload(item, depth=depth + 1)
            if compact_item in (None, "", [], {}):
                continue
            compacted[key_text] = compact_item
        return compacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = []
        for item in value:
            compact_item = _compact_payload(item, depth=depth + 1)
            if compact_item not in (None, "", [], {}):
                rows.append(compact_item)
        return rows
    return value


def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    artifacts = _compact_artifact_pointers(evidence.get("artifacts"))
    if artifacts:
        compacted["artifact_paths"] = artifacts
    for key in ("command", "title", "status", "timestamp", "summary", "steps", "retrievals", "rpg_deltas", "dep_graph_deltas", "verification", "user_decisions"):
        value = evidence.get(key)
        if value in (None, "", [], {}):
            continue
        compacted[key] = _compact_payload(value)
    change_summary = _compact_change_summary(evidence.get("code_deltas"))
    if change_summary:
        compacted["changed_files"] = change_summary
    nested = evidence.get("evidence")
    if isinstance(nested, Mapping):
        nested_artifacts = _compact_artifact_pointers(nested.get("artifact_paths") or nested.get("artifacts"))
        if nested_artifacts and not artifacts:
            compacted["artifact_paths"] = nested_artifacts
        audit_source = nested.get("audit_summary") if isinstance(nested.get("audit_summary"), Mapping) else {
            key: value for key, value in nested.items()
            if key not in {"artifact_paths", "artifacts"}
        }
        audit = _compact_payload(audit_source)
        if audit:
            compacted["audit_summary"] = audit
    return compacted


def _render_evidence(evidence: Mapping[str, Any]) -> str:
    data = json.dumps(_compact_evidence(evidence), indent=2, ensure_ascii=False, default=_json_default)
    return f"<section><details><summary>Evidence JSON</summary><pre>{_h(data)}</pre></details></section>"


def _artifact_href(path: Any) -> str:
    if path is None:
        return "#"
    try:
        return Path(str(path)).expanduser().resolve().as_uri()
    except Exception:
        return "file://" + quote(str(path), safe="/._-~")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _h(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=_json_default)
    return escape(str(value), quote=False)


def _h_attr(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)
