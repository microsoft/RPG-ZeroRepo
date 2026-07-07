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
    for key in ("semantic_nodes", "code_nodes", "mappings", "edges", "warnings", "changed_files"):
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
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{_h(title)}</title>
<style>
:root {{ color-scheme: light; --bg:#f6f8fb; --card:#fff; --text:#1f2937; --muted:#6b7280; --line:#d9e0ea; --accent:#2563eb; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width:1120px; margin:0 auto; padding:32px 20px 48px; }}
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
.focused-graph-toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0 12px; }}
.focused-graph-toolbar button, .focused-graph-toolbar input {{ border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--text); padding:6px 10px; font:inherit; }}
.focused-graph-toolbar label {{ display:inline-flex; gap:6px; align-items:center; color:var(--muted); }}
.focused-graph-stage {{ border:1px solid var(--line); border-radius:12px; background:#fbfdff; min-height:480px; position:relative; overflow:hidden; }}
.focused-graph-svg {{ display:block; width:100%; height:480px; cursor:grab; touch-action:none; }}
.focused-graph-svg:active {{ cursor:grabbing; }}
.focused-graph-link {{ transition:opacity .15s ease, stroke-width .15s ease; }}
.focused-graph-link.active {{ opacity:1; stroke:#0f172a; stroke-width:2.6; }}
.focused-graph-link.dimmed {{ opacity:.14; }}
.focused-graph-link.hidden {{ display:none; }}
.focused-graph-node {{ cursor:pointer; transition:opacity .15s ease; }}
.focused-graph-node circle {{ transition:stroke .15s ease, stroke-width .15s ease; }}
.focused-graph-node.selected circle, .focused-graph-node.active circle {{ stroke:#0f172a; stroke-width:3; }}
.focused-graph-node.dragging {{ cursor:grabbing; }}
.focused-graph-node.dimmed {{ opacity:.18; }}
.focused-graph-node.hidden {{ display:none; }}
.focused-graph-fallback {{ margin:12px; padding:12px; border:1px dashed var(--line); border-radius:10px; background:#fff; color:var(--muted); }}
.focused-graph-legend {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 12px; }}
.legend-item {{ display:inline-flex; gap:6px; align-items:center; color:var(--muted); font-size:13px; }}
.legend-swatch {{ width:10px; height:10px; border-radius:999px; display:inline-block; border:1px solid var(--line); }}
.legend-semantic {{ background:#2563eb; }}
.legend-code {{ background:#16a34a; }}
.legend-mapping {{ background:#f59e0b; }}
.legend-context {{ background:#64748b; }}
@media (max-width:720px) {{ main {{ padding:22px 12px 36px; }} .focus-map {{ grid-template-columns:1fr; }} table {{ min-width:560px; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>{_h(title)}</h1>
<div class=\"meta\"><span>Command: <strong>{_h(command)}</strong></span><span>Generated: {_h(generated_at)}</span>{status_html}</div>
</header>
{_render_summary_cards(summary_cards)}
{_render_timeline(stages, verification)}
{_render_safety_boundary(user_decisions)}
{_render_focused_graph(focused_view, code_file_anchors)}
{_render_code_deltas(code_deltas, code_delta_anchors)}
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
    hidden_relations = hidden_counts.get("relations") if isinstance(hidden_counts.get("relations"), Mapping) else {}
    rows: list[tuple[str, int]] = []
    for relation, count_key in relation_keys.items():
        count = hidden_counts.get(count_key) or 0
        if isinstance(hidden_relations, Mapping):
            count += hidden_relations.get(relation, 0) or 0
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            count_int = 0
        if count_int:
            rows.append((relation, count_int))
    return rows


def _hidden_context_html(hidden_counts: Mapping[str, Any]) -> str:
    parts = []
    for relation, count in _combined_hidden_counts(hidden_counts):
        parts.append(f"<p class=\"reason\">Hidden {_h(count)} additional {_h(relation)} neighbors.</p>")
    cap_rows = []
    for key in ("primary_rpg_nodes", "primary_code_nodes", "edges"):
        value = hidden_counts.get(key)
        if value:
            cap_rows.append(f"<tr><th>{_h(key)}</th><td>{_h(value)}</td></tr>")
    if cap_rows:
        parts.append("<table><tbody>" + "".join(cap_rows) + "</tbody></table>")
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


def _graph_label(row: Mapping[str, Any], fallback: Any) -> str:
    for key in ("name", "symbol", "label", "path", "node_id", "dep_node_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback or "node")


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
    for key in ("path", "type", "node_type", "source", "mapping_status", "locate_status", "breadcrumb_path"):
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
        children = [child for child in _as_sequence(row.get("children")) if isinstance(child, Mapping)]
        if row_id and children and row_id not in seen_nodes:
            seen_nodes.add(row_id)
            nodes.append({
                "id": row_id,
                "kind": "hierarchy",
                "label": str(row.get("name") or row.get("node_id") or row_id),
                "node_id": row.get("node_id") or row_id,
                "state": row.get("state") or row.get("kind") or "hierarchy",
                "type": row.get("kind") or "hierarchy",
            })
        for child in children:
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
    summary = nodes_view.get("summary") if isinstance(nodes_view.get("summary"), Mapping) else focused_view.get("summary", {})
    semantic_nodes = [node for node in _as_sequence(nodes_view.get("semantic_nodes")) if isinstance(node, Mapping)]
    code_nodes = [node for node in _as_sequence(nodes_view.get("code_nodes")) if isinstance(node, Mapping)]
    mappings = [mapping for mapping in _as_sequence(nodes_view.get("mappings")) if isinstance(mapping, Mapping)]
    context_edges = [edge for edge in _as_sequence(nodes_view.get("edges")) if isinstance(edge, Mapping)]
    hidden_counts = nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else focused_view.get("hidden_counts", {})
    warnings = [warning for warning in _as_sequence(nodes_view.get("warnings")) if isinstance(warning, Mapping)]
    nodes: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    rpg_links: dict[str, str] = {}
    code_links: dict[str, str] = {}

    for node in semantic_nodes:
        link_id = _append_graph_node(nodes, seen_nodes, node, "semantic", file_anchors)
        node_id = node.get("node_id")
        if node_id not in (None, ""):
            rpg_links[str(node_id)] = link_id
    for node in code_nodes:
        link_id = _append_graph_node(nodes, seen_nodes, node, "code", file_anchors)
        node_id = node.get("node_id") or node.get("dep_node_id")
        if node_id not in (None, ""):
            code_links[str(node_id)] = link_id

    links: list[dict[str, Any]] = []
    for mapping in mappings:
        mapping_id = _append_graph_node(nodes, seen_nodes, mapping, "mapping", file_anchors)
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        source = mapping.get("source_link_id") or rpg_links.get(str(rpg_id or ""))
        target = mapping.get("target_link_id") or code_links.get(str(code_id or ""))
        if source:
            links.append({"id": f"{mapping_id}-semantic", "source": source, "target": mapping_id, "kind": "mapping", "relation": mapping.get("status") or "mapped"})
        if target:
            links.append({"id": f"{mapping_id}-code", "source": mapping_id, "target": target, "kind": "mapping", "relation": mapping.get("source") or "code"})

    for edge in context_edges:
        source = edge.get("source_link_id")
        target = edge.get("target_link_id")
        if not source:
            source = _append_context_node(nodes, seen_nodes, None, edge.get("source_node_id"))
        elif str(source) not in seen_nodes:
            _append_context_node(nodes, seen_nodes, source, edge.get("source_node_id"))
        if not target:
            target = _append_context_node(nodes, seen_nodes, None, edge.get("target_node_id"))
        elif str(target) not in seen_nodes:
            _append_context_node(nodes, seen_nodes, target, edge.get("target_node_id"))
        links.append({
            "id": edge.get("link_id") or f"edge-{len(links) + 1}",
            "source": str(source),
            "target": str(target),
            "kind": "context",
            "relation": edge.get("relation") or "dependency",
            "direction": edge.get("direction"),
        })

    focused_graph = nodes_view.get("focused_graph") if isinstance(nodes_view.get("focused_graph"), Mapping) else {}
    hierarchy = nodes_view.get("hierarchy") or focused_graph.get("hierarchy") or {}
    default_focus = nodes_view.get("default_focus") or focused_graph.get("default_focus") or {}
    _append_hierarchy_nodes(nodes, links, seen_nodes, hierarchy)
    return {
        "schema": "cmind.focused_graph.render.v1",
        "summary": summary,
        "nodes": nodes,
        "links": links,
        "semantic_nodes": semantic_nodes,
        "code_nodes": code_nodes,
        "mappings": mappings,
        "edges": context_edges,
        "hidden_counts": hidden_counts,
        "warnings": warnings,
        "hierarchy": hierarchy,
        "default_focus": default_focus,
    }


def _focused_graph_runtime() -> str:
    return r"""
(function(){
  const section = document.currentScript.closest('[data-focused-graph]');
  if (!section) return;
  const dataEl = section.querySelector('[data-focused-graph-json]');
  const svg = section.querySelector('[data-focused-graph-svg]');
  const fallback = section.querySelector('[data-focused-graph-fallback]');
  if (!window.d3 || !dataEl || !svg) return;
  if (fallback) fallback.hidden = true;
  const data = JSON.parse(dataEl.textContent || '{}');
  const width = Number(svg.getAttribute('width')) || 960;
  const height = Number(svg.getAttribute('height')) || 480;
  const defaultFocus = data.default_focus || {};
  const defaultShowEdges = defaultFocus.show_edges !== false;
  const text = value => value === undefined || value === null ? '' : String(value);
  const list = value => Array.isArray(value) ? value : [];
  let hierarchyDepth = Math.max(0, Number(defaultFocus.hierarchy_depth ?? defaultFocus.edge_depth ?? 1) || 0);
  const defaultHierarchyDepth = hierarchyDepth;
  let showEdges = defaultShowEdges;
  let query = '';
  let selectedId = null;
  let visibleNodeIds = new Set();
  const collapsedHierarchyIds = new Set();
  const nodes = list(data.nodes).map((node, index) => ({
    ...node,
    id: text(node.id || node.node_id || `node-${index + 1}`),
    kind: text(node.kind || 'context'),
    label: text(node.label || node.name || node.node_id || node.id || `node-${index + 1}`),
  }));
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  const links = list(data.links).map((link, index) => {
    const source = text(link.source && link.source.id ? link.source.id : link.source);
    const target = text(link.target && link.target.id ? link.target.id : link.target);
    return {...link, id: text(link.id || `link-${index + 1}`), source, target};
  }).filter(link => nodeById.has(link.source) && nodeById.has(link.target));
  const hierarchyNodeById = new Map();
  const hierarchyParentById = new Map();
  const hierarchyChildrenById = new Map();
  const hierarchyDepthById = new Map();
  const hierarchyAncestorsById = new Map();
  const rootHierarchyId = text(data.hierarchy && data.hierarchy.id);
  const adjacency = new Map(nodes.map(node => [node.id, new Set()]));
  const linksByNode = new Map(nodes.map(node => [node.id, new Set()]));

  function fill(kind) {
    return kind === 'semantic' ? '#2563eb' : kind === 'code' ? '#16a34a' : kind === 'mapping' ? '#f59e0b' : kind === 'hierarchy' ? '#8b5cf6' : '#64748b';
  }
  function linkKey(link) {
    return text(link.id || `${link.source}-${link.target}`);
  }
  function collectHierarchy(row, parentId, depth, ancestors) {
    if (!row || typeof row !== 'object') return;
    const id = text(row.id || row.link_id || row.node_id);
    if (!id) return;
    const children = list(row.children).map(child => text(child && (child.id || child.link_id || child.node_id))).filter(Boolean);
    hierarchyNodeById.set(id, row);
    hierarchyChildrenById.set(id, children);
    hierarchyDepthById.set(id, depth);
    hierarchyAncestorsById.set(id, ancestors);
    if (parentId) hierarchyParentById.set(id, parentId);
    list(row.children).forEach(child => collectHierarchy(child, id, depth + 1, ancestors.concat(id)));
  }
  collectHierarchy(data.hierarchy, '', 0, []);
  let maxHierarchyDepth = Math.max(1, defaultHierarchyDepth);
  hierarchyDepthById.forEach(depth => { maxHierarchyDepth = Math.max(maxHierarchyDepth, depth); });
  const maxDepth = Math.max(maxHierarchyDepth, Math.min(Math.max(nodes.length - 1, 1), 8));

  links.forEach(link => {
    adjacency.get(link.source)?.add(link.target);
    adjacency.get(link.target)?.add(link.source);
    linksByNode.get(link.source)?.add(linkKey(link));
    linksByNode.get(link.target)?.add(linkKey(link));
  });

  const defaultFocusIds = new Set(list(defaultFocus.node_link_ids).map(text).filter(id => nodeById.has(id)));
  const semanticFocusIds = new Set(list(defaultFocus.semantic_node_ids).map(text));
  const codeFocusIds = new Set(list(defaultFocus.code_node_ids).map(text));
  const mappingFocusIds = new Set(list(defaultFocus.mapping_link_ids).map(text));
  nodes.forEach(node => {
    if (semanticFocusIds.has(text(node.node_id)) || codeFocusIds.has(text(node.node_id)) || mappingFocusIds.has(node.id)) {
      defaultFocusIds.add(node.id);
    }
  });

  function layout(rows) {
    const groups = {hierarchy: [], semantic: [], mapping: [], code: [], context: []};
    rows.forEach(node => (groups[node.kind] || groups.context).push(node));
    const x = {hierarchy: 90, semantic: 240, mapping: 455, code: 690, context: 850};
    const marginY = 36;
    const availableY = Math.max(120, height - marginY * 2);
    Object.entries(groups).forEach(([kind, groupRows]) => {
      const gap = groupRows.length > 1 ? Math.min(78, availableY / (groupRows.length - 1)) : 0;
      const start = height / 2 - ((groupRows.length - 1) * gap) / 2;
      groupRows.forEach((node, index) => { node.x = x[kind] || width / 2; node.y = start + index * gap; });
    });
  }

  function expandFromFocus(seedIds, depthLimit) {
    const visible = new Set([...seedIds].filter(id => nodeById.has(id)));
    let frontier = new Set(visible);
    for (let depth = 0; depth < depthLimit && frontier.size; depth += 1) {
      const next = new Set();
      frontier.forEach(id => (adjacency.get(id) || new Set()).forEach(neighbor => {
        if (!visible.has(neighbor)) {
          visible.add(neighbor);
          next.add(neighbor);
        }
      }));
      frontier = next;
    }
    return visible;
  }

  function nodeMatches(node) {
    if (!query) return true;
    return `${node.label} ${node.node_id || ''} ${node.path || ''} ${node.kind || ''}`.toLowerCase().includes(query);
  }

  function isCollapsed(nodeId) {
    return list(hierarchyAncestorsById.get(nodeId)).some(id => collapsedHierarchyIds.has(id));
  }

  function nearestHierarchyToggle(nodeId) {
    return nodeId !== rootHierarchyId && (hierarchyChildrenById.get(nodeId) || []).length ? nodeId : '';
  }

  function computeVisibleNodeIds() {
    const seeds = defaultFocusIds.size ? defaultFocusIds : new Set(nodes.map(node => node.id));
    const visible = expandFromFocus(seeds, hierarchyDepth);
    if (hierarchyNodeById.size) {
      nodes.forEach(node => {
        const depth = hierarchyDepthById.get(node.id);
        if (depth !== undefined && depth <= hierarchyDepth) visible.add(node.id);
      });
    }
    if (query) nodes.forEach(node => { if (nodeMatches(node)) visible.add(node.id); });
    nodes.forEach(node => {
      if (!nodeMatches(node) || isCollapsed(node.id)) visible.delete(node.id);
    });
    if (!visible.size && !query) nodes.forEach(node => { if (!isCollapsed(node.id)) visible.add(node.id); });
    return visible;
  }

  function isLinkVisible(link) {
    return showEdges && visibleNodeIds.has(link.source) && visibleNodeIds.has(link.target);
  }

  layout(nodes);
  const svgSelection = d3.select(svg).attr('viewBox', `0 0 ${width} ${height}`);
  svgSelection.selectAll('*').remove();
  const graphLayer = svgSelection.append('g').attr('class', 'focused-graph-layer');
  const linkLayer = graphLayer.append('g').attr('class', 'focused-graph-links');
  const nodeLayer = graphLayer.append('g').attr('class', 'focused-graph-nodes');
  const zoom = d3.zoom()
    .scaleExtent([0.35, 3.5])
    .on('zoom', event => graphLayer.attr('transform', event.transform));
  svgSelection.call(zoom).on('dblclick.zoom', null).on('click', event => {
    if (event.target === svg) {
      selectedId = null;
      renderVisibility();
    }
  });

  const linkSelection = linkLayer.selectAll('line')
    .data(links, link => linkKey(link))
    .join('line')
    .attr('class', 'focused-graph-link')
    .attr('data-link-id', link => linkKey(link))
    .attr('stroke', link => link.kind === 'mapping' ? '#f59e0b' : '#94a3b8')
    .attr('stroke-width', link => link.kind === 'mapping' ? 2 : 1.4)
    .attr('opacity', .78);

  const drag = d3.drag()
    .container(() => graphLayer.node())
    .on('start', function(event) {
      event.sourceEvent?.stopPropagation();
      d3.select(this).classed('dragging', true);
    })
    .on('drag', function(event, node) {
      node.x = event.x;
      node.y = event.y;
      updatePositions();
    })
    .on('end', function() {
      d3.select(this).classed('dragging', false);
    });

  const nodeSelection = nodeLayer.selectAll('g')
    .data(nodes, node => node.id)
    .join(enter => {
      const group = enter.append('g')
        .attr('class', 'focused-graph-node')
        .attr('data-node-id', node => node.id)
        .attr('tabindex', 0)
        .attr('role', 'button');
      group.append('circle')
        .attr('r', node => node.kind === 'mapping' ? 6 : 8)
        .attr('fill', node => fill(node.kind))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
      group.append('text')
        .attr('x', 13)
        .attr('y', 4)
        .attr('font-size', 12)
        .attr('fill', '#1f2937')
        .text(node => text(node.label || node.id).slice(0, 42));
      group.append('title').text(node => `${node.kind}: ${node.label || node.id}`);
      return group;
    })
    .call(drag)
    .on('click', (event, node) => {
      event.stopPropagation();
      selectedId = selectedId === node.id ? null : node.id;
      renderVisibility();
    })
    .on('dblclick', (event, node) => {
      event.stopPropagation();
      const toggleId = nearestHierarchyToggle(node.id);
      if (!toggleId) return;
      if (collapsedHierarchyIds.has(toggleId)) collapsedHierarchyIds.delete(toggleId);
      else collapsedHierarchyIds.add(toggleId);
      if (selectedId && isCollapsed(selectedId)) selectedId = null;
      renderVisibility();
    });

  function updatePositions() {
    linkSelection
      .attr('x1', link => nodeById.get(link.source)?.x || 0)
      .attr('y1', link => nodeById.get(link.source)?.y || 0)
      .attr('x2', link => nodeById.get(link.target)?.x || 0)
      .attr('y2', link => nodeById.get(link.target)?.y || 0);
    nodeSelection.attr('transform', node => `translate(${node.x},${node.y})`);
  }

  function renderVisibility() {
    visibleNodeIds = computeVisibleNodeIds();
    if (selectedId && !visibleNodeIds.has(selectedId)) selectedId = null;
    const activeNodes = new Set();
    const activeLinks = new Set();
    if (selectedId) {
      activeNodes.add(selectedId);
      (adjacency.get(selectedId) || new Set()).forEach(id => activeNodes.add(id));
      (linksByNode.get(selectedId) || new Set()).forEach(id => activeLinks.add(id));
    }
    nodeSelection
      .classed('hidden', node => !visibleNodeIds.has(node.id))
      .classed('selected', node => node.id === selectedId)
      .classed('active', node => Boolean(selectedId && activeNodes.has(node.id)))
      .classed('dimmed', node => Boolean(selectedId && visibleNodeIds.has(node.id) && !activeNodes.has(node.id)));
    linkSelection
      .classed('hidden', link => !isLinkVisible(link))
      .classed('active', link => Boolean(selectedId && activeLinks.has(linkKey(link))))
      .classed('dimmed', link => Boolean(selectedId && isLinkVisible(link) && !activeLinks.has(linkKey(link))));
  }

  function expandDepth() {
    hierarchyDepth = Math.min(maxDepth, hierarchyDepth + 1);
    renderVisibility();
  }

  function collapseDepth() {
    hierarchyDepth = Math.max(0, hierarchyDepth - 1);
    renderVisibility();
  }

  section.querySelector('[data-action="reset"]')?.addEventListener('click', () => {
    hierarchyDepth = defaultHierarchyDepth;
    showEdges = defaultShowEdges;
    query = '';
    selectedId = null;
    collapsedHierarchyIds.clear();
    const search = section.querySelector('[data-action="search"]');
    if (search) search.value = '';
    const edges = section.querySelector('[data-action="edges"]');
    if (edges) edges.checked = showEdges;
    svgSelection.transition().duration(150).call(zoom.transform, d3.zoomIdentity);
    renderVisibility();
  });
  section.querySelector('[data-action="depth-plus"]')?.addEventListener('click', expandDepth);
  section.querySelector('[data-action="depth-minus"]')?.addEventListener('click', collapseDepth);
  section.querySelector('[data-action="edges"]')?.addEventListener('change', event => { showEdges = event.target.checked; renderVisibility(); });
  section.querySelector('[data-action="search"]')?.addEventListener('input', event => { query = text(event.target.value).toLowerCase(); renderVisibility(); });
  updatePositions();
  renderVisibility();
})();
"""


def _render_focused_graph(focused_view: dict[str, Any], file_anchors: Mapping[str, str]) -> str:
    nodes_view = focused_view.get("nodes_view") if isinstance(focused_view.get("nodes_view"), Mapping) else {}
    if not focused_view and not nodes_view:
        return ""
    summary = nodes_view.get("summary") if isinstance(nodes_view.get("summary"), Mapping) else focused_view.get("summary", {})
    summary_html = _summary_badges(summary, [
        ("Semantic nodes", "semantic_nodes", len(_as_sequence(nodes_view.get("semantic_nodes")))),
        ("Code nodes", "code_nodes", len(_as_sequence(nodes_view.get("code_nodes")))),
        ("Mappings", "mappings", len(_as_sequence(nodes_view.get("mappings")))),
        ("Edges", "edges", len(_as_sequence(nodes_view.get("edges")))),
        ("Warnings", "warnings", len(_as_sequence(nodes_view.get("warnings")))),
    ])
    hidden_counts = nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else focused_view.get("hidden_counts", {})
    hidden_html = _hidden_context_html(hidden_counts if isinstance(hidden_counts, Mapping) else {})
    warnings = [warning for warning in _as_sequence(nodes_view.get("warnings") or focused_view.get("warnings")) if isinstance(warning, Mapping)]
    warnings_html = f"<h3>Warnings</h3>{_chain_warning_html(warnings)}" if warnings else ""
    graph_payload = _focused_graph_payload(focused_view, file_anchors)
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
        '<button type="button" data-action="reset">Reset</button>'
        '<button type="button" data-action="depth-plus" aria-label="Expand hierarchy depth" title="Expand hierarchy depth">+1</button>'
        '<button type="button" data-action="depth-minus" aria-label="Collapse hierarchy depth" title="Collapse hierarchy depth">-1</button>'
        '<label><input type="checkbox" data-action="edges" checked> Edges</label>'
        '<input type="search" data-action="search" placeholder="Search nodes" aria-label="Search focused graph nodes">'
        '</div>'
    )
    legend = (
        '<div class="focused-graph-legend" aria-label="Focused graph legend">'
        '<span class="legend-item"><span class="legend-swatch legend-semantic"></span>Semantic</span>'
        '<span class="legend-item"><span class="legend-swatch legend-code"></span>Code</span>'
        '<span class="legend-item"><span class="legend-swatch legend-mapping"></span>Mapping</span>'
        '<span class="legend-item"><span class="legend-swatch" style="background:#8b5cf6"></span>Hierarchy</span>'
        '<span class="legend-item"><span class="legend-swatch legend-context"></span>Context edge</span>'
        '</div>'
    )
    return (
        '<section class="focused-graph-section" data-focused-graph><h2>Focused graph</h2>'
        f"{summary_html}{controls}{legend}{warnings_html}{hidden_html}"
        f"<script type=\"application/json\" data-focused-graph-json>{graph_json}</script>"
        '<div class="focused-graph-stage">'
        '<svg class="focused-graph-svg" data-focused-graph-svg role="img" aria-label="Focused graph view" width="960" height="480"></svg>'
        f'<div class="focused-graph-fallback" data-focused-graph-fallback{fallback_hidden}>Static focused graph fallback is available when D3 cannot run.{d3_missing_note}</div>'
        '</div>'
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
    hidden_html = _hidden_context_html(nodes_view.get("hidden_counts") if isinstance(nodes_view.get("hidden_counts"), Mapping) else {})
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
