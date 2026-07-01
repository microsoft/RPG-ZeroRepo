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
    focused_graph = data.get("focused_graph") or evidence_data.get("focused_graph")
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
        focused_graph=_normalize_focused_graph(focused_graph),
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


def _normalize_focused_graph(value: Any) -> dict[str, Any]:
    if value in (None, "", [], {}):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"detail": value}


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
    focused_graph: dict[str, Any],
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
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; margin:16px 0; padding:18px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
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
table {{ width:100%; border-collapse:collapse; font-size:14px; table-layout:fixed; }}
th, td {{ border-top:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }}
th {{ color:var(--muted); font-weight:600; background:#fbfdff; }}
code {{ white-space:normal; overflow-wrap:anywhere; word-break:break-word; }}
a {{ color:var(--accent); text-decoration:none; overflow-wrap:anywhere; word-break:break-word; }}
a:hover {{ text-decoration:underline; }}
.empty {{ color:var(--muted); font-style:italic; }}
pre {{ white-space:pre-wrap; overflow:auto; background:#0f172a; color:#e5e7eb; border-radius:10px; padding:14px; }}
details summary {{ cursor:pointer; color:var(--accent); font-weight:600; }}
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
{_render_verification(verification)}
{_render_retrievals(retrievals)}
{_render_node_table("Focused RPG node evidence", rpg_nodes)}
{_render_node_table("Focused dependency node evidence", dep_nodes)}
{_render_code_deltas(code_deltas)}
{_render_focused_graph(focused_graph)}
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


def _render_retrievals(retrievals: list[dict[str, Any]]) -> str:
    if not retrievals:
        return ""
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
    return f"<section><h2>Retrieval evidence</h2>{body}</section>"


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
    return f"<section><h2>Code deltas</h2>{''.join(blocks)}</section>"


def _render_focused_graph(focused_graph: dict[str, Any]) -> str:
    if not focused_graph:
        return ""
    path = focused_graph.get("path") or focused_graph.get("artifact_path") or focused_graph.get("html_path")
    href = _artifact_href(path) if path else "#"
    rows = [
        ("Status", focused_graph.get("status", "recorded"), False),
        ("Graph artifact", f"<a href=\"{_h_attr(href)}\">{_h(path or '')}</a>" if path else "", True),
        ("Selected RPG nodes", ", ".join(str(v) for v in focused_graph.get("selected_rpg_nodes") or focused_graph.get("rpg_node_ids") or []), False),
        ("Selected dependency nodes", ", ".join(str(v) for v in focused_graph.get("selected_dep_nodes") or focused_graph.get("dep_node_ids") or []), False),
        ("Included RPG nodes", focused_graph.get("rpg_node_count") or focused_graph.get("rpg_nodes") or "", False),
        ("Included dependency nodes", focused_graph.get("dep_node_count") or focused_graph.get("dep_nodes") or "", False),
    ]
    table_rows = []
    for label, value, is_html in rows:
        if value in (None, ""):
            continue
        rendered = str(value) if is_html else _h(value)
        table_rows.append(f"<tr><th>{_h(label)}</th><td>{rendered}</td></tr>")
    table = "<table><tbody>" + "".join(table_rows) + "</tbody></table>" if table_rows else "<p class=\"empty\">No focused graph metadata recorded.</p>"
    metadata = json.dumps(focused_graph, indent=2, ensure_ascii=False, default=_json_default)
    inspector = f"<details><summary>Inspector metadata</summary><pre>{_h(metadata)}</pre></details>"
    return f"<section><h2>Focused graph evidence</h2>{table}{inspector}</section>"


def _render_node_table(title: str, nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        body = "<p class=\"empty\">No node evidence recorded.</p>"
    elif any("dep_node_id" in node for node in nodes):
        rows = []
        for node in nodes:
            rows.append(
                "<tr>"
                f"<td><code>{_h(node.get('dep_node_id', ''))}</code></td>"
                f"<td>{_h(node.get('path', ''))}</td>"
                f"<td>{_h(node.get('source_feature', ''))}</td>"
                f"<td>{_h(node.get('change', ''))}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>ID</th><th>Path</th><th>Source feature</th><th>Change</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    else:
        rows = []
        for node in nodes:
            rows.append(
                "<tr>"
                f"<td><code>{_h(node.get('node_id', ''))}</code></td>"
                f"<td>{_h(node.get('name', ''))}</td>"
                f"<td>{_h(node.get('type', ''))}</td>"
                f"<td>{_h(node.get('path', ''))}</td>"
                f"<td>{_h(node.get('score', ''))}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Path</th><th>Score/status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<section><h2>{_h(title)}</h2>{body}</section>"


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
