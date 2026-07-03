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
    code_deltas = data.get("code_deltas") or evidence_data.get("code_deltas")
    focused_view = data.get("focused_view") or evidence_data.get("focused_view")
    if not focused_view:
        focused_view = data.get("focused_impact") or evidence_data.get("focused_impact")
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
    normalized["unmatched_code_deltas"] = [
        dict(delta) if isinstance(delta, Mapping) else {"file": delta}
        for delta in _as_sequence(value.get("unmatched_code_deltas"))
    ]
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
</style>
</head>
<body>
<main>
<header>
<h1>{_h(title)}</h1>
<div class=\"meta\"><span>Command: <strong>{_h(command)}</strong></span><span>Generated: {_h(generated_at)}</span>{status_html}</div>
</header>
{_render_summary_cards(summary_cards)}
{_render_timeline(stages)}
{_render_safety_boundary(user_decisions)}
{_render_why_these_nodes(retrievals, rpg_nodes, dep_nodes)}
{_render_focused_impact(focused_view)}
{_render_code_deltas(code_deltas)}
{_render_verification(verification)}
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


def _render_timeline(stages: list[dict[str, Any]]) -> str:
    if not stages:
        body = "<p class=\"empty\">No stages recorded.</p>"
    else:
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
        body = '<ol class="timeline">' + "".join(items) + "</ol>"
    return f"<section><h2>Stage timeline</h2>{body}</section>"


def _render_verification(checks: list[dict[str, Any]]) -> str:
    if not checks:
        body = "<p class=\"empty\">No verification status recorded.</p>"
    else:
        rows = []
        for check in checks:
            rows.append(
                "<tr>"
                f"<td>{_h(check.get('name') or 'verification')}</td>"
                f"<td>{_h(check.get('status', ''))}</td>"
                f"<td>{_h(check.get('detail', ''))}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<section><h2>Verification status</h2>{body}</section>"


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


def _render_code_deltas(deltas: list[dict[str, Any]]) -> str:
    if not deltas:
        return ""
    blocks = []
    for delta in deltas:
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
            "<div class=\"delta\">"
            "<div class=\"delta-head\">"
            f"<code>{_h(delta.get('file', ''))}</code>"
            f"<span class=\"badge\">{_h(delta.get('change_type', ''))}</span>"
            "</div>"
            f"{diff_html}{before_after}"
            "</div>"
        )
    return f"<section><h2>What changed?</h2>{''.join(blocks)}</section>"


def _render_why_these_nodes(
    retrievals: list[dict[str, Any]],
    rpg_nodes: list[dict[str, Any]],
    dep_nodes: list[dict[str, Any]],
) -> str:
    if not retrievals and not rpg_nodes and not dep_nodes:
        return ""
    blocks = []
    if retrievals:
        blocks.append(_render_retrievals(retrievals, title="Retrieval evidence", as_section=False))
    if rpg_nodes:
        blocks.append(_render_node_rows("Selected feature groups", rpg_nodes, dep_graph=False))
    if dep_nodes:
        blocks.append(_render_node_rows("Mapped code relations", dep_nodes, dep_graph=True))
    return f"<section><h2>Why these nodes?</h2>{''.join(blocks)}</section>"


def _render_node_rows(title: str, nodes: list[dict[str, Any]], *, dep_graph: bool) -> str:
    if not nodes:
        return f"<h3>{_h(title)}</h3><p class=\"empty\">No node evidence recorded.</p>"
    rows = []
    if dep_graph:
        for node in nodes:
            dep_id = node.get("dep_node_id") or node.get("node_id")
            rows.append(
                "<tr>"
                f"<td><code>{_h(dep_id)}</code></td>"
                f"<td>{_h(node.get('path', ''))}</td>"
                f"<td><code>{_h(node.get('source_feature', ''))}</code></td>"
                f"<td>{_h(node.get('relation') or node.get('change') or node.get('status', ''))}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>Code node</th><th>Path</th><th>Feature</th><th>Relation/state</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    else:
        for node in nodes:
            rows.append(
                "<tr>"
                f"<td><code>{_h(node.get('node_id', ''))}</code></td>"
                f"<td>{_h(node.get('name', ''))}</td>"
                f"<td>{_h(node.get('type') or node.get('node_type') or '')}</td>"
                f"<td>{_h(node.get('path', ''))}</td>"
                f"<td>{_h(node.get('score') or node.get('status') or '')}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>Feature</th><th>Name</th><th>Type</th><th>Path</th><th>Score/state</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<h3>{_h(title)}</h3>{body}"


def _focused_graph_metadata(focused_view: dict[str, Any]) -> str:
    data = json.dumps(focused_view, indent=2, ensure_ascii=False, default=_json_default)
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


def _render_focused_impact(focused_view: dict[str, Any]) -> str:
    if not focused_view:
        return ""
    summary = focused_view.get("summary") if isinstance(focused_view.get("summary"), Mapping) else {}
    rpg_nodes = [node for node in _as_sequence(focused_view.get("primary_rpg_nodes")) if isinstance(node, Mapping)]
    code_nodes = [node for node in _as_sequence(focused_view.get("primary_code_nodes")) if isinstance(node, Mapping)]
    mappings = [mapping for mapping in _as_sequence(focused_view.get("mappings")) if isinstance(mapping, Mapping)]
    edges = [edge for edge in _as_sequence(focused_view.get("edges")) if isinstance(edge, Mapping)]
    hidden_counts = focused_view.get("hidden_counts") if isinstance(focused_view.get("hidden_counts"), Mapping) else {}
    warnings = [warning for warning in _as_sequence(focused_view.get("warnings")) if isinstance(warning, Mapping)]
    rpg_by_id = {str(node.get("node_id")): node for node in rpg_nodes if node.get("node_id") not in (None, "")}
    code_by_id = {
        str(node.get("node_id") or node.get("dep_node_id")): node
        for node in code_nodes
        if (node.get("node_id") or node.get("dep_node_id")) not in (None, "")
    }
    summary_items = [
        ("Primary RPG nodes", summary.get("primary_rpg_nodes", len(rpg_nodes))),
        ("Primary code nodes", summary.get("primary_code_nodes", len(code_nodes))),
        ("Mapped relations", summary.get("mapped_code_relations", len([row for row in mappings if row.get("code_node_id")]))),
        ("Missing mappings", summary.get("missing_mappings", len([row for row in mappings if row.get("status") == "missing"]))),
        ("Edges shown", summary.get("edges", len(edges))),
        ("Warnings", summary.get("warnings", len(warnings))),
    ]
    summary_html = "<div class=\"focus-summary\">" + "".join(
        f"<span class=\"badge\">{_h(label)}: <strong>{_h(value)}</strong></span>"
        for label, value in summary_items
    ) + "</div>"

    code_rows = []
    for node in code_nodes:
        node_id = node.get("node_id") or node.get("dep_node_id")
        code_rows.append(
            "<tr>"
            f"<td>{_focused_node_cell(node_id, node)}</td>"
            f"<td>{_h(node.get('type') or node.get('kind') or '')}</td>"
            f"<td>{_h(node.get('status', ''))}</td>"
            f"<td>{_h(node.get('source', ''))}</td>"
            "</tr>"
        )
    code_html = "<p class=\"empty\">No primary code nodes recorded.</p>"
    if code_rows:
        code_html = "<table><thead><tr><th>Code node</th><th>Type</th><th>Status</th><th>Source</th></tr></thead><tbody>" + "".join(code_rows) + "</tbody></table>"

    mapping_rows = []
    for mapping in mappings:
        rpg_id = mapping.get("rpg_node_id") or mapping.get("node_id")
        code_id = mapping.get("code_node_id") or mapping.get("dep_node_id")
        rpg_node = rpg_by_id.get(str(rpg_id)) if rpg_id not in (None, "") else None
        code_node = code_by_id.get(str(code_id)) if code_id not in (None, "") else None
        changed_files = ", ".join(str(item) for item in _as_sequence(mapping.get("changed_files")))
        mapping_rows.append(
            "<tr>"
            f"<td>{_focused_node_cell(rpg_id, rpg_node)}</td>"
            f"<td>{_focused_node_cell(code_id, code_node)}</td>"
            f"<td>{_h(mapping.get('status', ''))}</td>"
            f"<td>{_h(mapping.get('path') or (code_node or {}).get('path') or '')}</td>"
            f"<td>{_h(mapping.get('source', ''))}</td>"
            f"<td>{_h(mapping.get('reason', ''))}</td>"
            f"<td>{_h(changed_files)}</td>"
            "</tr>"
        )
    mappings_html = "<p class=\"empty\">No semantic-code mappings recorded.</p>"
    if mapping_rows:
        mappings_html = (
            "<table><thead><tr><th>Semantic node</th><th>Code node</th><th>Status</th>"
            "<th>Path</th><th>Source</th><th>Reason</th><th>Changed files</th></tr></thead><tbody>"
            + "".join(mapping_rows)
            + "</tbody></table>"
        )

    hidden_relations = hidden_counts.get("relations") if isinstance(hidden_counts.get("relations"), Mapping) else {}
    relation_hidden_keys = {"caller": "callers", "callee": "callees", "import": "imports", "inheritance": "inheritance"}
    edges_by_relation: dict[str, list[Mapping[str, Any]]] = {}
    for edge in edges:
        relation = str(edge.get("relation") or "dependency")
        edges_by_relation.setdefault(relation, []).append(edge)
    relation_names = set(edges_by_relation)
    for relation, hidden_key in relation_hidden_keys.items():
        if hidden_counts.get(hidden_key):
            relation_names.add(relation)
    relation_blocks = []
    for relation in sorted(relation_names):
        hidden = hidden_relations.get(relation, 0) if isinstance(hidden_relations, Mapping) else 0
        hidden += hidden_counts.get(relation_hidden_keys.get(relation, ""), 0) or 0
        relation_blocks.append(_render_focused_impact_group({"relation": relation, "edges": edges_by_relation.get(relation, []), "hidden": hidden}))
    neighborhood_html = "".join(relation_blocks) or "<p class=\"empty\">No one-hop neighborhood edges recorded.</p>"

    warning_html = "<p class=\"empty\">No focused warnings recorded.</p>"
    if warnings:
        warning_items = []
        for warning in warnings:
            warning_type = warning.get("type", "warning")
            context = {key: value for key, value in warning.items() if key not in {"type", "message"}}
            context_html = f" <code>{_h(context)}</code>" if context else ""
            warning_items.append(f"<li><code>{_h(warning_type)}</code> {_h(warning.get('message', ''))}{context_html}</li>")
        warning_html = '<ul class="warning-list">' + "".join(warning_items) + "</ul>"

    hidden_html = "<p class=\"empty\">No hidden focused context recorded.</p>"
    if hidden_counts:
        hidden_rows = "".join(f"<tr><th>{_h(key)}</th><td>{_h(value)}</td></tr>" for key, value in hidden_counts.items())
        hidden_html = "<table><tbody>" + hidden_rows + "</tbody></table>"

    return (
        "<section><h2>Focused impact view</h2>"
        f"{summary_html}"
        f"<h3>Primary code nodes</h3>{code_html}"
        f"<h3>Semantic-code mappings</h3>{mappings_html}"
        f"<h3>Capped neighborhood</h3>{neighborhood_html}"
        f"<h3>Warnings</h3>{warning_html}"
        f"<h3>Hidden context</h3>{hidden_html}"
        f"{_focused_graph_metadata(focused_view)}"
        "</section>"
    )


def _render_focused_impact_group(group: Mapping[str, Any]) -> str:
    relation = group.get("relation") or "dependency"
    edges = [edge for edge in _as_sequence(group.get("edges")) if isinstance(edge, Mapping)]
    hidden = group.get("hidden") or 0
    edge_rows = []
    for edge in edges:
        edge_rows.append(
            "<tr>"
            f"<td><code>{_h(edge.get('source_node_id', ''))}</code></td>"
            f"<td><code>{_h(edge.get('target_node_id', ''))}</code></td>"
            f"<td>{_h(edge.get('direction', ''))}</td>"
            f"<td>{_h(edge.get('path', ''))}</td>"
            f"<td>{_h(edge.get('source', ''))}</td>"
            f"<td>{_h(edge.get('reason', ''))}</td>"
            "</tr>"
        )
    rows_html = "<p class=\"empty\">No visible rows for this relation.</p>"
    if edge_rows:
        rows_html = "<table><thead><tr><th>Source</th><th>Target</th><th>Direction</th><th>Path</th><th>Source</th><th>Reason</th></tr></thead><tbody>" + "".join(edge_rows) + "</tbody></table>"
    hidden_html = f"<p class=\"reason\">Hidden { _h(hidden) } additional {_h(relation)} neighbors.</p>" if hidden else ""
    return f"<details><summary>{_h(relation)} neighborhood ({_h(len(edges))} shown)</summary>{hidden_html}{rows_html}</details>"


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


def _render_evidence(evidence: Mapping[str, Any]) -> str:
    data = json.dumps(evidence, indent=2, ensure_ascii=False, default=_json_default)
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
