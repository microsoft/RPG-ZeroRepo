#!/usr/bin/env python3
"""
CoderMind Usage Statistics & Report Generator

Reads JSONL telemetry logs from .cmind/logs/ and generates usage reports.

Usage:
    python utils/rpg_stats.py                  # print summary to stdout
    python utils/rpg_stats.py --report         # write markdown report to .cmind/reports/
    python utils/rpg_stats.py --json           # print summary as JSON
    python utils/rpg_stats.py --days 7         # only last 7 days

Log files consumed:
    .cmind/logs/mcp_calls.jsonl    — MCP tool invocations (search, explore, etc.)
    .cmind/logs/hook_calls.jsonl   — git hook invocations (sync, update-rpg)
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Support running from workspace root or from utils/
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from common.paths import MCP_CALLS_LOG, HOOK_CALLS_LOG, REPORTS_DIR
except ImportError:
    # Fallback for standalone usage
    _WS = Path.cwd()
    MCP_CALLS_LOG = _WS / ".cmind" / "logs" / "mcp_calls.jsonl"
    HOOK_CALLS_LOG = _WS / ".cmind" / "logs" / "hook_calls.jsonl"
    REPORTS_DIR = _WS / ".cmind" / "reports"


def _read_jsonl(path: Path, since: Optional[datetime] = None) -> List[dict]:
    """Read a JSONL file, optionally filtering by timestamp."""
    if not path.is_file():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if since and "ts" in rec:
                    ts = datetime.fromisoformat(rec["ts"])
                    if ts < since:
                        continue
                records.append(rec)
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def _compute_mcp_stats(records: List[dict]) -> Dict[str, Any]:
    """Compute MCP tool usage statistics."""
    if not records:
        return {"total_calls": 0}

    tool_counts = Counter(r.get("tool", "?") for r in records)
    tool_durations: Dict[str, list] = defaultdict(list)
    queries: List[str] = []

    for r in records:
        tool = r.get("tool", "?")
        dur = r.get("duration_ms", 0)
        tool_durations[tool].append(dur)
        if tool == "search_rpg" and "params" in r:
            q = r["params"].get("query", "")
            if q:
                queries.append(q)

    tool_stats = {}
    for tool, count in tool_counts.most_common():
        durs = tool_durations[tool]
        tool_stats[tool] = {
            "calls": count,
            "avg_ms": round(sum(durs) / len(durs)) if durs else 0,
            "max_ms": max(durs) if durs else 0,
        }

    top_queries = Counter(queries).most_common(10)

    return {
        "total_calls": len(records),
        "tools": tool_stats,
        "top_queries": [{"query": q, "count": c} for q, c in top_queries],
        "first_call": records[0].get("ts", ""),
        "last_call": records[-1].get("ts", ""),
    }


def _compute_hook_stats(records: List[dict]) -> Dict[str, Any]:
    """Compute hook invocation statistics."""
    if not records:
        return {"total_calls": 0}

    hook_counts = Counter(r.get("hook", "?") for r in records)
    mode_counts = Counter(r.get("mode", "?") for r in records)
    durations: Dict[str, list] = defaultdict(list)
    total_modified = 0
    total_added = 0
    total_deleted = 0

    for r in records:
        hook = r.get("hook", "?")
        dur = r.get("duration_ms", 0)
        durations[hook].append(dur)
        total_modified += r.get("modified", 0) or 0
        total_added += r.get("added", 0) or 0
        total_deleted += r.get("deleted", 0) or 0

    hook_stats = {}
    for hook, count in hook_counts.most_common():
        durs = durations[hook]
        hook_stats[hook] = {
            "calls": count,
            "avg_ms": round(sum(durs) / len(durs)) if durs else 0,
            "max_ms": max(durs) if durs else 0,
        }

    return {
        "total_calls": len(records),
        "hooks": hook_stats,
        "modes": dict(mode_counts.most_common()),
        "total_files_modified": total_modified,
        "total_files_added": total_added,
        "total_files_deleted": total_deleted,
        "first_call": records[0].get("ts", ""),
        "last_call": records[-1].get("ts", ""),
    }


def generate_report(mcp_stats: dict, hook_stats: dict, days: Optional[int] = None) -> str:
    """Generate a Markdown usage report."""
    period = f"last {days} days" if days else "all time"
    lines = [
        f"# CoderMind Usage Report",
        f"",
        f"Period: **{period}**",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
    ]

    # MCP section
    lines.append("## MCP Tool Calls")
    lines.append("")
    if mcp_stats["total_calls"] == 0:
        lines.append("No MCP tool calls recorded.")
    else:
        lines.append(f"Total calls: **{mcp_stats['total_calls']}**")
        lines.append(f"Period: {mcp_stats.get('first_call', '?')} → {mcp_stats.get('last_call', '?')}")
        lines.append("")
        lines.append("| Tool | Calls | Avg (ms) | Max (ms) |")
        lines.append("|------|-------|----------|----------|")
        for tool, s in mcp_stats.get("tools", {}).items():
            lines.append(f"| `{tool}` | {s['calls']} | {s['avg_ms']} | {s['max_ms']} |")
        lines.append("")

        top_q = mcp_stats.get("top_queries", [])
        if top_q:
            lines.append("### Top Search Queries")
            lines.append("")
            lines.append("| Query | Count |")
            lines.append("|-------|-------|")
            for item in top_q:
                lines.append(f"| `{item['query']}` | {item['count']} |")
            lines.append("")

    # Hook section
    lines.append("## Hook Invocations")
    lines.append("")
    if hook_stats["total_calls"] == 0:
        lines.append("No hook invocations recorded.")
    else:
        lines.append(f"Total invocations: **{hook_stats['total_calls']}**")
        lines.append(f"Period: {hook_stats.get('first_call', '?')} → {hook_stats.get('last_call', '?')}")
        lines.append("")
        lines.append("| Hook | Calls | Avg (ms) | Max (ms) |")
        lines.append("|------|-------|----------|----------|")
        for hook, s in hook_stats.get("hooks", {}).items():
            lines.append(f"| `{hook}` | {s['calls']} | {s['avg_ms']} | {s['max_ms']} |")
        lines.append("")

        modes = hook_stats.get("modes", {})
        if modes:
            lines.append("### Sync Modes")
            lines.append("")
            for mode, count in modes.items():
                lines.append(f"- **{mode}**: {count}")
            lines.append("")

        lines.append("### File Changes (cumulative)")
        lines.append("")
        lines.append(f"- Modified: {hook_stats.get('total_files_modified', 0)}")
        lines.append(f"- Added: {hook_stats.get('total_files_added', 0)}")
        lines.append(f"- Deleted: {hook_stats.get('total_files_deleted', 0)}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CoderMind usage statistics")
    parser.add_argument("--report", action="store_true",
                        help="Write Markdown report to .cmind/reports/")
    parser.add_argument("--json", action="store_true",
                        help="Output raw stats as JSON")
    parser.add_argument("--days", type=int, default=None,
                        help="Only include records from the last N days")
    parser.add_argument("--mcp-log", type=Path, default=MCP_CALLS_LOG)
    parser.add_argument("--hook-log", type=Path, default=HOOK_CALLS_LOG)
    args = parser.parse_args()

    since = None
    if args.days:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)

    mcp_records = _read_jsonl(args.mcp_log, since)
    hook_records = _read_jsonl(args.hook_log, since)

    mcp_stats = _compute_mcp_stats(mcp_records)
    hook_stats = _compute_hook_stats(hook_records)

    if args.json:
        print(json.dumps({"mcp": mcp_stats, "hooks": hook_stats}, indent=2))
        return

    if args.report:
        report = generate_report(mcp_stats, hook_stats, args.days)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = REPORTS_DIR / f"rpg_usage_{date_str}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"Report written to {report_path}")
        return

    # Default: print summary to stdout
    print(f"=== MCP Tool Calls ({mcp_stats['total_calls']} total) ===")
    for tool, s in mcp_stats.get("tools", {}).items():
        print(f"  {tool}: {s['calls']} calls, avg {s['avg_ms']}ms")
    top_q = mcp_stats.get("top_queries", [])
    if top_q:
        print(f"  Top queries: {', '.join(q['query'] for q in top_q[:5])}")

    print(f"\n=== Hook Invocations ({hook_stats['total_calls']} total) ===")
    for hook, s in hook_stats.get("hooks", {}).items():
        print(f"  {hook}: {s['calls']} calls, avg {s['avg_ms']}ms")
    modes = hook_stats.get("modes", {})
    if modes:
        print(f"  Modes: {', '.join(f'{m}={c}' for m, c in modes.items())}")
    print(f"  Files: +{hook_stats.get('total_files_added', 0)} "
          f"~{hook_stats.get('total_files_modified', 0)} "
          f"-{hook_stats.get('total_files_deleted', 0)}")


if __name__ == "__main__":
    main()
