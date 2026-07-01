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

_MAX_SUMMARY_CARDS = 7


def write_command_report(
    command: str | None = None,
    payload: Any = None,
    *,
    summary_cards: Any = None,
    summary: Any = None,
    stages: Any = None,
    timeline: Any = None,
    rpg_nodes: Any = None,
    dep_nodes: Any = None,
    artifacts: Any = None,
    artifact_links: Any = None,
    verification: Any = None,
    evidence: Any = None,
    evidence_json: Any = None,
    status: str | None = None,
    title: str | None = None,
    report_dir: str | Path | None = None,
    timestamp: str | datetime | None = None,
    **extra: Any,
) -> Path:
    """Write a sanitized Explain View HTML report and return its path."""
    if isinstance(payload, Mapping):
        extra = {**dict(payload), **extra}
    elif payload is not None:
        extra.setdefault("payload", payload)
    if command is None:
        command = str(extra.pop("command", extra.pop("command_name", "command")))
    summary_cards = summary_cards if summary_cards is not None else extra.pop("summary_cards", None)
    summary = summary if summary is not None else extra.pop("summary", None)
    stages = stages if stages is not None else extra.pop("stages", None)
    timeline = timeline if timeline is not None else extra.pop("timeline", None)
    rpg_nodes = rpg_nodes if rpg_nodes is not None else extra.pop("rpg_nodes", None)
    dep_nodes = dep_nodes if dep_nodes is not None else extra.pop("dep_nodes", None)
    artifacts = artifacts if artifacts is not None else extra.pop("artifacts", None)
    artifact_links = artifact_links if artifact_links is not None else extra.pop("artifact_links", None)
    verification = verification if verification is not None else extra.pop("verification", None)
    evidence = evidence if evidence is not None else extra.pop("evidence", None)
    evidence_json = evidence_json if evidence_json is not None else extra.pop("evidence_json", None)
    status = status if status is not None else extra.pop("status", None)
    title = title if title is not None else extra.pop("title", None)
    report_dir = report_dir if report_dir is not None else extra.pop("report_dir", None)
    timestamp = timestamp if timestamp is not None else extra.pop("timestamp", None)
    summary_cards = summary_cards if summary_cards is not None else summary
    stages = stages if stages is not None else timeline
    artifacts = artifacts if artifacts is not None else artifact_links
    evidence = evidence if evidence is not None else evidence_json

    if rpg_nodes is None:
        rpg_nodes = extra.pop("rpg_node_evidence", None) or extra.pop("rpg_evidence", None)
    if dep_nodes is None:
        dep_nodes = extra.pop("dep_node_evidence", None) or extra.pop("dep_evidence", None)
    if isinstance(evidence, Mapping):
        if rpg_nodes is None:
            rpg_nodes = _evidence_nodes(evidence.get("rpg_nodes")) or _evidence_nodes(evidence.get("rpg_node_evidence"))
        if dep_nodes is None:
            dep_nodes = _evidence_nodes(evidence.get("dep_nodes")) or _evidence_nodes(evidence.get("dep_node_evidence"))

    generated_at = _display_timestamp(timestamp)
    filename_ts = _filename_timestamp(timestamp)
    safe_command = _slug(command)
    target_dir = Path(report_dir) if report_dir is not None else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(target_dir / f"cmind_run_{safe_command}_{filename_ts}.html")

    aggregate_evidence = {
        "command": command,
        "status": status,
        "summary_cards": summary_cards,
        "stages": stages,
        "rpg_nodes": rpg_nodes,
        "dep_nodes": dep_nodes,
        "artifacts": artifacts,
        "verification": verification,
        "evidence": evidence,
    }
    for key, value in extra.items():
        aggregate_evidence[key] = value

    page_title = title or f"CoderMind {command} Explain View"
    html = _render_page(
        title=page_title,
        command=command,
        generated_at=generated_at,
        status=status,
        summary_cards=_normalize_cards(summary_cards),
        stages=_normalize_stages(stages),
        rpg_nodes=_normalize_nodes(rpg_nodes),
        dep_nodes=_normalize_nodes(dep_nodes),
        artifacts=_normalize_artifacts(artifacts),
        verification=_normalize_verification(verification),
        evidence=aggregate_evidence,
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
        return list(value.items())
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _normalize_cards(value: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        iterable = value.items()
    else:
        iterable = _as_sequence(value)
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            label, card_value = item
            cards.append({"label": label, "value": card_value})
        elif isinstance(item, Mapping):
            label = item.get("label") or item.get("title") or item.get("name") or item.get("key") or "Summary"
            card_value = item.get("value", item.get("count", item.get("text", "")))
            detail = item.get("detail") or item.get("description")
            cards.append({"label": label, "value": card_value, "detail": detail})
        else:
            cards.append({"label": "Summary", "value": item})
    return cards[:_MAX_SUMMARY_CARDS]


def _normalize_stages(value: Any) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, tuple) and len(item) == 2:
            name, state = item
            stages.append({"name": name, "status": state})
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("stage") or item.get("id") or "stage"
            status = item.get("status") or item.get("state") or item.get("type") or item.get("action") or "recorded"
            reason = item.get("reason") or item.get("message") or item.get("description") or ""
            duration = item.get("duration") or item.get("elapsed") or item.get("elapsed_seconds")
            stages.append({"name": name, "status": status, "reason": reason, "duration": duration})
        else:
            stages.append({"name": item, "status": "recorded"})
    return stages


def _evidence_nodes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        return value
    return None


def _normalize_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, tuple) and len(item) == 2:
            node_id, node_value = item
            if isinstance(node_value, Mapping):
                entry = {"node_id": node_id, **dict(node_value)}
            else:
                entry = {"node_id": node_id, "value": node_value}
        elif isinstance(item, Mapping):
            entry = dict(item)
        else:
            entry = {"node_id": item}
        nodes.append(entry)
    return nodes


def _normalize_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        iterable = value.items()
    else:
        iterable = _as_sequence(value)
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            label, path = item
            artifacts.append({"label": label, "path": path, "status": _artifact_status(path)})
        elif isinstance(item, Mapping):
            path = item.get("path") or item.get("href") or item.get("url") or item.get("file")
            label = item.get("label") or item.get("name") or item.get("title")
            if path is None:
                path_items = [
                    (key, item_value)
                    for key, item_value in item.items()
                    if key not in {"label", "name", "title", "status"}
                ]
                if len(path_items) == 1:
                    label, path = path_items[0]
            label = label or path or "artifact"
            artifacts.append({"label": label, "path": path, "status": _artifact_status(path, item.get("status"))})
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
    if isinstance(value, Mapping) and not any(isinstance(v, Mapping) for v in value.values()):
        for key, check_value in value.items():
            checks.append({"name": key, "status": check_value})
        return checks
    if isinstance(value, Mapping):
        iterable = value.items()
    else:
        iterable = _as_sequence(value)
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            name, check_value = item
            if isinstance(check_value, Mapping):
                checks.append({"name": name, **dict(check_value)})
            else:
                checks.append({"name": name, "status": check_value})
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("check") or item.get("label") or "verification"
            checks.append({"name": name, **dict(item)})
        else:
            checks.append({"name": "verification", "status": item})
    return checks


def _render_page(
    *,
    title: str,
    command: str,
    generated_at: str,
    status: str | None,
    summary_cards: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    rpg_nodes: list[dict[str, Any]],
    dep_nodes: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    verification: list[dict[str, Any]],
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
{_render_verification(verification)}
{_render_node_table("Focused RPG node evidence", rpg_nodes)}
{_render_node_table("Focused dependency node evidence", dep_nodes)}
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
                f"<td>{_h(check.get('name') or check.get('check') or 'verification')}</td>"
                f"<td>{_h(check.get('status', check.get('success', '')))}</td>"
                f"<td>{_h(check.get('detail') or check.get('message') or check.get('reason') or '')}</td>"
                "</tr>"
            )
        body = "<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<section><h2>Verification status</h2>{body}</section>"


def _render_node_table(title: str, nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        body = "<p class=\"empty\">No node evidence recorded.</p>"
    else:
        rows = []
        for node in nodes:
            node_id = node.get("node_id") or node.get("id") or node.get("dep_node") or ""
            node_type = node.get("type_name") or node.get("node_type") or node.get("type") or ""
            path = node.get("meta_path") or node.get("path") or node.get("file_path") or node.get("feature_path") or ""
            score = node.get("score") or node.get("weight") or node.get("status") or ""
            rows.append(
                "<tr>"
                f"<td><code>{_h(node_id)}</code></td>"
                f"<td>{_h(node.get('name', ''))}</td>"
                f"<td>{_h(node_type)}</td>"
                f"<td>{_h(path)}</td>"
                f"<td>{_h(score)}</td>"
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
