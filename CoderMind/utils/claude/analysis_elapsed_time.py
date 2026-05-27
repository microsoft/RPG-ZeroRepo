#!/usr/bin/env python3
"""Extract per-API-call token usage and elapsed time to CSV.

Usage:
    # Single file → CSV to stdout
    python extract_calls.py <trajectory.jsonl>

    # Single file → CSV to file
    python extract_calls.py <trajectory.jsonl> -o output.csv

    # Directory batch (all .jsonl with given prefix)
    python extract_calls.py <directory> --prefix <prefix> -o output.csv

For each API call (assistant record with positive token counts), outputs:
    file, call_index, line, model, input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, elapsed_s
"""

import csv
import json
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, IO


@dataclass
class CallRecord:
    """One API call with tokens and timing."""
    file: str
    call_index: int
    line_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    elapsed_s: float           # seconds from last user record to this final chunk
    timestamp: str             # ISO timestamp of the final chunk


CSV_FIELDS = [
    "file", "call_index", "line", "model",
    "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
    "elapsed_s", "timestamp",
]


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle trailing Z
        s = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def extract_calls(filepath: str) -> List[CallRecord]:
    """Extract API call records from a trajectory JSONL file."""
    records: List[CallRecord] = []
    call_idx = 0

    # Track the timestamp of the last user-type record seen.
    # This serves as the "request sent" timestamp for the next API call.
    last_user_ts: Optional[datetime] = None
    # Also track the very first assistant delta timestamp as fallback.
    first_assistant_delta_ts: Optional[datetime] = None
    in_assistant_turn = False

    fname = Path(filepath).name

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_type = rec.get("type", "")
            ts_str = rec.get("timestamp", "")
            ts = _parse_ts(ts_str)

            # queue-operation: just track timestamp
            if rec_type == "queue-operation":
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            usage = msg.get("usage", {})

            # ── user record ───────────────────────────────────────
            if rec_type == "user" or role == "user":
                if ts is not None:
                    last_user_ts = ts
                # Reset assistant turn tracking
                in_assistant_turn = False
                first_assistant_delta_ts = None
                continue

            # ── assistant record ──────────────────────────────────
            if rec_type == "assistant" or role == "assistant":
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)

                # Stream delta (usage all zero)
                if in_tok == 0 and out_tok == 0 and cache_create == 0 and cache_read == 0:
                    if not in_assistant_turn:
                        in_assistant_turn = True
                        first_assistant_delta_ts = ts
                    continue

                # ── API call termination record ───────────────────
                call_idx += 1
                model = msg.get("model", "")

                # Compute elapsed time
                elapsed = 0.0
                if ts is not None and last_user_ts is not None:
                    elapsed = (ts - last_user_ts).total_seconds()
                elif ts is not None and first_assistant_delta_ts is not None:
                    # Fallback: use first streaming delta if no user record
                    elapsed = (ts - first_assistant_delta_ts).total_seconds()

                records.append(CallRecord(
                    file=fname,
                    call_index=call_idx,
                    line_number=line_no,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_creation_input_tokens=cache_create,
                    cache_read_input_tokens=cache_read,
                    elapsed_s=round(elapsed, 3),
                    timestamp=ts_str,
                ))

                # Reset for next turn
                in_assistant_turn = False
                first_assistant_delta_ts = None

    return records


def write_csv(records: List[CallRecord], dest: IO[str]) -> None:
    """Write call records as CSV."""
    writer = csv.writer(dest)
    writer.writerow(CSV_FIELDS)
    for r in records:
        writer.writerow([
            r.file, r.call_index, r.line_number, r.model,
            r.input_tokens, r.output_tokens,
            r.cache_creation_input_tokens, r.cache_read_input_tokens,
            r.elapsed_s, r.timestamp,
        ])


def print_table(records: List[CallRecord]) -> None:
    """Print records as a human-readable table to stdout."""
    if not records:
        print("No API calls found.")
        return

    # Adaptive file column width
    max_name = max(len(r.file) for r in records)
    name_col = min(max(max_name, 10), 55)

    def shorten(s, n):
        return s if len(s) <= n else s[:n - 3] + "..."

    hdr = (f"{'File':<{name_col}}  {'#':>3}  {'Line':>5}  {'Model':>12}  "
           f"{'Input':>10}  {'Output':>10}  {'Cache_Cr':>10}  "
           f"{'Cache_Rd':>10}  {'Elapsed':>9}")
    print(hdr)
    print("-" * len(hdr))

    for r in records:
        short_model = (r.model.replace("claude-", "").replace("opus-", "o")
                       .replace("sonnet-", "s").replace("haiku-", "h") if r.model else "")
        print(f"{shorten(r.file, name_col):<{name_col}}  {r.call_index:>3}  "
              f"{r.line_number:>5}  {short_model:>12}  "
              f"{r.input_tokens:>10,}  {r.output_tokens:>10,}  "
              f"{r.cache_creation_input_tokens:>10,}  "
              f"{r.cache_read_input_tokens:>10,}  "
              f"{r.elapsed_s:>8.1f}s")

    # Summary
    print("-" * len(hdr))
    tot_in = sum(r.input_tokens for r in records)
    tot_out = sum(r.output_tokens for r in records)
    tot_cc = sum(r.cache_creation_input_tokens for r in records)
    tot_cr = sum(r.cache_read_input_tokens for r in records)
    tot_elapsed = sum(r.elapsed_s for r in records)
    print(f"{'TOTAL':<{name_col}}  {len(records):>3}  {'':>5}  {'':>12}  "
          f"{tot_in:>10,}  {tot_out:>10,}  {tot_cc:>10,}  "
          f"{tot_cr:>10,}  {tot_elapsed:>8.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-API-call token usage and elapsed time to CSV."
    )
    parser.add_argument("file", help="Path to a .jsonl file or a directory")
    parser.add_argument("--prefix", default=None,
                        help="When FILE is a directory, only process .jsonl files "
                             "whose name starts with this prefix")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV file path (default: print table to stdout)")
    args = parser.parse_args()

    target = Path(args.file)

    # ── Collect files ─────────────────────────────────────────────────────
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if args.prefix:
            files = [f for f in files if f.name.startswith(args.prefix)]
        if not files:
            pfx_msg = f" with prefix '{args.prefix}'" if args.prefix else ""
            print(f"No .jsonl files found in {target}{pfx_msg}", file=sys.stderr)
            sys.exit(1)
    elif target.is_file():
        files = [target]
    else:
        print(f"Error: not found: {target}", file=sys.stderr)
        sys.exit(1)

    # ── Extract records ───────────────────────────────────────────────────
    all_records: List[CallRecord] = []
    for f in files:
        try:
            all_records.extend(extract_calls(str(f)))
        except Exception as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)

    # ── Output ────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as csvf:
            write_csv(all_records, csvf)
        print(f"Wrote {len(all_records)} records to {args.output}")
    else:
        # No -o: print human-readable table
        print_table(all_records)


if __name__ == "__main__":
    main()
