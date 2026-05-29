#!/usr/bin/env python3
"""
Analyze a folder of Copilot CLI log files and produce aggregate usage statistics.

Scans all .log files in the specified directory (default: ~/.copilot/logs/),
parses token usage and timing information, and prints a summary report including:
  - Total sessions, premium requests, and tokens
  - Per-model breakdown
  - Total and per-session runtime
  - Date range of sessions
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def _extract_response_blocks(log_path: str) -> list[dict]:
    """Extract JSON response blocks from a single log file."""
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if re.search(r"response \(Request-ID", lines[i]):
            j = i + 1
            while j < len(lines) and "{" not in lines[j]:
                j += 1
            if j >= len(lines):
                i = j
                continue

            json_lines = []
            depth = 0
            started = False
            while j < len(lines):
                raw = lines[j]
                content = re.sub(
                    r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+\[\w+\]\s*", "", raw
                )
                json_lines.append(content)
                depth += content.count("{") - content.count("}")
                if content.count("{") > 0:
                    started = True
                if started and depth <= 0:
                    break
                j += 1

            try:
                block = json.loads("\n".join(json_lines))
                blocks.append(block)
            except json.JSONDecodeError:
                pass
            i = j + 1
        else:
            i += 1
    return blocks


def _extract_session_time(log_path: str) -> tuple[datetime | None, datetime | None]:
    """Extract the first and last timestamp from a log file to compute session duration."""
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)")
    first_ts = None
    last_ts = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = ts_pattern.match(line)
            if m:
                ts_str = m.group(1)
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
    return first_ts, last_ts


_KNOWN_MODELS = [
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "o4-mini",
    "o3",
    "o3-mini",
]


def _normalize_model_name(model: str) -> str:
    for name in _KNOWN_MODELS:
        if name in model:
            return name
    return model


def _format_duration(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def analyze_folder(folder: str) -> dict:
    """
    Analyze all .log files in a folder and return aggregate statistics.

    Returns a dict with:
      - totals: aggregate token/request counts
      - models: per-model breakdown
      - sessions: per-file summary (file, duration, premium_requests, tokens)
      - time_range: earliest and latest session timestamps
      - total_runtime: sum of all session durations
    """
    log_dir = Path(folder)
    log_files = sorted(log_dir.glob("*.log"))

    per_model = defaultdict(lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "premium_requests": 0,
        "nano_aiu": 0,
    })

    sessions = []
    total_runtime = timedelta()
    earliest = None
    latest = None
    total_aux_requests = 0

    for log_file in log_files:
        # Parse token usage
        blocks = _extract_response_blocks(str(log_file))
        session_premium = 0
        session_prompt = 0
        session_completion = 0
        session_cached = 0
        session_nano_aiu = 0
        session_aux = 0

        for block in blocks:
            usage = block.get("usage")
            if not usage:
                continue

            model = block.get("model", "unknown")
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

            copilot_usage = block.get("copilot_usage")
            if copilot_usage:
                display_model = _normalize_model_name(model)
                entry = per_model[display_model]
                entry["prompt_tokens"] += prompt
                entry["completion_tokens"] += completion
                entry["cached_tokens"] += cached
                entry["premium_requests"] += 1
                entry["nano_aiu"] += copilot_usage.get("total_nano_aiu", 0)

                session_premium += 1
                session_prompt += prompt
                session_completion += completion
                session_cached += cached
                session_nano_aiu += copilot_usage.get("total_nano_aiu", 0)
            else:
                session_aux += 1

        total_aux_requests += session_aux

        # Parse timing
        first_ts, last_ts = _extract_session_time(str(log_file))
        duration = timedelta()
        if first_ts and last_ts:
            duration = last_ts - first_ts
            total_runtime += duration
            if earliest is None or first_ts < earliest:
                earliest = first_ts
            if latest is None or last_ts > latest:
                latest = last_ts

        if session_premium > 0 or session_aux > 0:
            sessions.append({
                "file": log_file.name,
                "start": first_ts.isoformat() if first_ts else None,
                "duration": _format_duration(duration),
                "duration_seconds": duration.total_seconds(),
                "premium_requests": session_premium,
                "prompt_tokens": session_prompt,
                "completion_tokens": session_completion,
                "cached_tokens": session_cached,
                "nano_aiu": session_nano_aiu,
            })

    totals = {
        "total_log_files": len(log_files),
        "sessions_with_requests": len(sessions),
        "premium_requests": sum(m["premium_requests"] for m in per_model.values()),
        "auxiliary_requests": total_aux_requests,
        "prompt_tokens": sum(m["prompt_tokens"] for m in per_model.values()),
        "completion_tokens": sum(m["completion_tokens"] for m in per_model.values()),
        "cached_tokens": sum(m["cached_tokens"] for m in per_model.values()),
        "total_nano_aiu": sum(m["nano_aiu"] for m in per_model.values()),
    }

    return {
        "totals": totals,
        "models": dict(per_model),
        "sessions": sessions,
        "time_range": {
            "earliest": earliest.isoformat() if earliest else None,
            "latest": latest.isoformat() if latest else None,
        },
        "total_runtime": _format_duration(total_runtime),
        "total_runtime_seconds": total_runtime.total_seconds(),
    }


def print_report(result: dict) -> None:
    totals = result["totals"]
    models = result["models"]
    sessions = result["sessions"]

    print("=" * 70)
    print("       Copilot CLI Usage Statistics — Folder Summary")
    print("=" * 70)

    # Time range
    time_range = result["time_range"]
    if time_range["earliest"]:
        print(f"\n  Date range:         {time_range['earliest'][:10]}  →  {time_range['latest'][:10]}")
    print(f"  Total runtime:      {result['total_runtime']}")
    print(f"  Log files scanned:  {totals['total_log_files']}")
    print(f"  Sessions w/ usage:  {totals['sessions_with_requests']}")

    # Token totals
    print(f"\n{'─' * 70}")
    print("  Token Totals")
    print(f"{'─' * 70}")
    print(f"  Premium requests:     {totals['premium_requests']:>10,}")
    print(f"  Auxiliary requests:   {totals['auxiliary_requests']:>10,}")
    print(f"  Prompt tokens:        {totals['prompt_tokens']:>10,}  ({_format_tokens(totals['prompt_tokens'])})")
    print(f"  Completion tokens:    {totals['completion_tokens']:>10,}  ({_format_tokens(totals['completion_tokens'])})")
    print(f"  Cached tokens:        {totals['cached_tokens']:>10,}  ({_format_tokens(totals['cached_tokens'])})")
    print(f"  Total nano AIU:       {totals['total_nano_aiu']:>10,}")

    # Per-model breakdown
    if models:
        print(f"\n{'─' * 70}")
        print("  Per-Model Breakdown")
        print(f"{'─' * 70}")
        header = f"  {'Model':<25s} {'Requests':>8s} {'Prompt':>10s} {'Completion':>10s} {'Cached':>10s} {'nano AIU':>14s}"
        print(header)
        print(f"  {'─' * 25} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 14}")
        for model in sorted(models):
            m = models[model]
            print(
                f"  {model:<25s} {m['premium_requests']:>8,} "
                f"{_format_tokens(m['prompt_tokens']):>10s} "
                f"{_format_tokens(m['completion_tokens']):>10s} "
                f"{_format_tokens(m['cached_tokens']):>10s} "
                f"{m['nano_aiu']:>14,}"
            )

    # Top sessions by token usage
    if sessions:
        print(f"\n{'─' * 70}")
        print("  Top 15 Sessions by Premium Requests")
        print(f"{'─' * 70}")
        top = sorted(sessions, key=lambda s: s["premium_requests"], reverse=True)[:15]
        header = f"  {'Log File':<42s} {'Reqs':>5s} {'Prompt':>8s} {'Compl.':>8s} {'Duration':>10s}"
        print(header)
        print(f"  {'─' * 42} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 10}")
        for s in top:
            print(
                f"  {s['file']:<42s} {s['premium_requests']:>5d} "
                f"{_format_tokens(s['prompt_tokens']):>8s} "
                f"{_format_tokens(s['completion_tokens']):>8s} "
                f"{s['duration']:>10s}"
            )

    print(f"\n{'=' * 70}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a folder of Copilot CLI log files for usage statistics."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path.home() / ".copilot" / "logs"),
        help="Path to the folder containing .log files (default: ~/.copilot/logs/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    log_count = len(list(folder.glob("*.log")))
    if log_count == 0:
        print(f"No .log files found in '{folder}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {log_count} log files in {folder} ...", file=sys.stderr)
    result = analyze_folder(str(folder))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
