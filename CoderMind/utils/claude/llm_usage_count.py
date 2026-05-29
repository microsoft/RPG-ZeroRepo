#!/usr/bin/env python3
"""Count LLM token usage from a Claude Code trajectory JSONL file.

Usage:
    # Single file (detailed report)
    python count_tokens.py <trajectory.jsonl> [--detail] [--no-cost]

    # Directory batch (table of all matching files)
    python count_tokens.py <directory> --prefix <prefix> [--no-cost]

Each line in the JSONL is one of:
  - queue-operation  : metadata (enqueue/dequeue), skipped
  - user             : user prompt or tool_result, text estimated only
  - assistant        : LLM response; streaming deltas have usage all-zero,
                       the final chunk of each API call carries real usage

Token accounting (from Anthropic docs):
  total_input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens

  where:
    input_tokens                  = tokens AFTER the last cache breakpoint (uncached)
    cache_creation_input_tokens   = tokens written to cache this call
    cache_read_input_tokens       = tokens read from cache (hit)

Cost multipliers (vs base input price):
    cache write (5-min) : 1.25×
    cache write (1-hour): 2.00×
    cache read          : 0.10×
"""

import json
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


# ── Pricing tables ($ per million tokens) ────────────────────────────────────

PRICING = {
    # model_prefix: (base_input, cache_write_5m, cache_write_1h, cache_read, output)
    "claude-opus-4-6":   (5.0,   6.25,  10.0,  0.50,  25.0),
    "claude-opus-4.6":   (5.0,   6.25,  10.0,  0.50,  25.0),
    "claude-opus-4-5":   (5.0,   6.25,  10.0,  0.50,  25.0),
    "claude-opus-4.5":   (5.0,   6.25,  10.0,  0.50,  25.0),
    "claude-opus-4-1":   (15.0,  18.75, 30.0,  1.50,  75.0),
    "claude-opus-4.1":   (15.0,  18.75, 30.0,  1.50,  75.0),
    "claude-opus-4":     (15.0,  18.75, 30.0,  1.50,  75.0),
    "claude-sonnet-4-6": (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-sonnet-4.6": (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-sonnet-4-5": (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-sonnet-4.5": (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-sonnet-4":   (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-sonnet-3.7": (3.0,   3.75,  6.0,   0.30,  15.0),
    "claude-haiku-4.5":  (1.0,   1.25,  2.0,   0.10,  5.0),
    "claude-haiku-3.5":  (0.80,  1.00,  1.60,  0.08,  4.0),
    "claude-haiku-3":    (0.25,  0.30,  0.50,  0.03,  1.25),
}


def _match_pricing(model: str):
    """Find the best matching pricing entry for a model string."""
    if not model:
        return None
    m = model.lower()
    # Try exact match first
    if m in PRICING:
        return PRICING[m]
    # Try prefix match (handles dated variants like claude-opus-4-5-20251101)
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        if m.startswith(key):
            return PRICING[key]
    return None


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class APICall:
    """One LLM API call (final streaming chunk with real usage)."""
    index: int                          # 1-based API call index
    line_number: int                    # line number in JSONL
    model: str = ""
    input_tokens: int = 0              # uncached input after last breakpoint
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    # Granular cache creation breakdown
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0
    # Server tool use
    web_search_requests: int = 0
    web_fetch_requests: int = 0

    @property
    def total_input(self) -> int:
        return (self.input_tokens
                + self.cache_creation_input_tokens
                + self.cache_read_input_tokens)

    def cost(self, pricing=None) -> Optional[float]:
        """Compute cost in USD.  Returns None if pricing unavailable."""
        p = pricing or _match_pricing(self.model)
        if not p:
            return None
        base_in, cw5m, cw1h, cr, out = p
        per_m = 1_000_000

        # Granular cache-creation split if available
        if self.ephemeral_5m_input_tokens or self.ephemeral_1h_input_tokens:
            cache_write_cost = (
                self.ephemeral_5m_input_tokens * cw5m / per_m
                + self.ephemeral_1h_input_tokens * cw1h / per_m
            )
        else:
            # Default: assume all cache_creation is 5-min
            cache_write_cost = self.cache_creation_input_tokens * cw5m / per_m

        return (
            self.input_tokens * base_in / per_m
            + cache_write_cost
            + self.cache_read_input_tokens * cr / per_m
            + self.output_tokens * out / per_m
        )


@dataclass
class TextEstimate:
    """Estimated tokens from text content (chars / 4)."""
    user_prompt_chars: int = 0
    tool_result_chars: int = 0
    assistant_text_chars: int = 0
    assistant_tool_input_chars: int = 0

    @property
    def user_prompt_tokens_est(self) -> int:
        return self.user_prompt_chars // 4

    @property
    def tool_result_tokens_est(self) -> int:
        return self.tool_result_chars // 4

    @property
    def assistant_text_tokens_est(self) -> int:
        return self.assistant_text_chars // 4

    @property
    def assistant_tool_tokens_est(self) -> int:
        return self.assistant_tool_input_chars // 4


@dataclass
class FileSummary:
    """Aggregated statistics for one trajectory file."""
    filepath: str
    total_lines: int = 0
    queue_operation_count: int = 0
    user_record_count: int = 0
    assistant_record_count: int = 0
    assistant_stream_delta_count: int = 0  # usage all-zero
    api_calls: List[APICall] = field(default_factory=list)
    text_est: TextEstimate = field(default_factory=TextEstimate)
    models_seen: Dict[str, int] = field(default_factory=dict)


# ── Parsing ──────────────────────────────────────────────────────────────────

def _content_char_len(content) -> int:
    """Recursively measure the char length of message content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                # tool_result, tool_use, text, etc.
                if "text" in block:
                    total += len(block["text"])
                if "content" in block:
                    total += _content_char_len(block["content"])
                if "input" in block and isinstance(block["input"], dict):
                    total += len(json.dumps(block["input"]))
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


def parse_file(filepath: str) -> FileSummary:
    """Parse a trajectory JSONL file and extract token statistics."""
    summary = FileSummary(filepath=filepath)
    api_call_idx = 0

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            summary.total_lines += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec_type = rec.get("type", "")

            # ── queue-operation ───────────────────────────────────────
            if rec_type == "queue-operation":
                summary.queue_operation_count += 1
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")
            usage = msg.get("usage", {})

            # ── user record ───────────────────────────────────────────
            if rec_type == "user" or role == "user":
                summary.user_record_count += 1
                clen = _content_char_len(content)
                # Distinguish initial prompt vs tool results
                if isinstance(content, str):
                    summary.text_est.user_prompt_chars += clen
                elif isinstance(content, list):
                    summary.text_est.tool_result_chars += clen
                continue

            # ── assistant record ──────────────────────────────────────
            if rec_type == "assistant" or role == "assistant":
                summary.assistant_record_count += 1

                # Track model
                model = msg.get("model", "")
                if model:
                    summary.models_seen[model] = summary.models_seen.get(model, 0) + 1

                # Text estimation (all assistant chunks, including deltas)
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "text" and "text" in block:
                            summary.text_est.assistant_text_chars += len(block["text"])
                        elif btype == "tool_use" and "input" in block:
                            summary.text_est.assistant_tool_input_chars += len(
                                json.dumps(block["input"])
                            )

                # Is this a streaming delta (usage all-zero) or a real API result?
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)

                if in_tok == 0 and out_tok == 0 and cache_create == 0 and cache_read == 0:
                    summary.assistant_stream_delta_count += 1
                    continue

                # Real API call result
                api_call_idx += 1
                call = APICall(
                    index=api_call_idx,
                    line_number=line_no,
                    model=model,
                    input_tokens=in_tok,
                    cache_creation_input_tokens=cache_create,
                    cache_read_input_tokens=cache_read,
                    output_tokens=out_tok,
                )

                # Granular cache creation
                cache_creation_obj = usage.get("cache_creation", {})
                if isinstance(cache_creation_obj, dict):
                    call.ephemeral_5m_input_tokens = cache_creation_obj.get(
                        "ephemeral_5m_input_tokens", 0
                    )
                    call.ephemeral_1h_input_tokens = cache_creation_obj.get(
                        "ephemeral_1h_input_tokens", 0
                    )

                # Server tool use
                stu = usage.get("server_tool_use", {})
                if isinstance(stu, dict):
                    call.web_search_requests = stu.get("web_search_requests", 0)
                    call.web_fetch_requests = stu.get("web_fetch_requests", 0)

                summary.api_calls.append(call)

    return summary


# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_tokens(n: int) -> str:
    """Format token count with thousands separator."""
    return f"{n:>10,}"


def _fmt_cost(c: Optional[float]) -> str:
    if c is None:
        return "        N/A"
    return f"  ${c:>8.4f}"


# ── Output ───────────────────────────────────────────────────────────────────

def print_summary(summary: FileSummary, show_detail: bool = True,
                  show_cost: bool = True) -> None:
    filepath = summary.filepath
    fname = Path(filepath).name
    if len(fname) > 80:
        fname = fname[:40] + "..." + fname[-37:]

    print("=" * 80)
    print(f"  Token Usage Report: {fname}")
    print("=" * 80)
    print()

    # ── Record counts ─────────────────────────────────────────────────────
    print("Record Counts:")
    print(f"  Total lines            : {summary.total_lines}")
    print(f"  queue-operation        : {summary.queue_operation_count}")
    print(f"  user records           : {summary.user_record_count}")
    print(f"  assistant records      : {summary.assistant_record_count}")
    print(f"    stream deltas (0/0)  : {summary.assistant_stream_delta_count}")
    print(f"    API call results     : {len(summary.api_calls)}")
    if summary.models_seen:
        print(f"  Models                 : {', '.join(summary.models_seen.keys())}")
    print()

    if not summary.api_calls:
        print("  No API calls with token usage found.")
        return

    # ── Per-call detail ───────────────────────────────────────────────────
    if show_detail:
        # Determine primary model for pricing
        primary_model = ""
        if summary.models_seen:
            primary_model = max(summary.models_seen, key=summary.models_seen.get)
        pricing = _match_pricing(primary_model)

        print("Per-API-Call Detail:")
        hdr = (f"  {'#':>3}  {'Line':>5}  {'input':>10}  {'cache_cr':>10}  "
               f"{'cache_rd':>10}  {'total_in':>10}  {'output':>10}")
        if show_cost:
            hdr += f"  {'cost':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for call in summary.api_calls:
            row = (f"  {call.index:>3}  {call.line_number:>5}  "
                   f"{_fmt_tokens(call.input_tokens)}  "
                   f"{_fmt_tokens(call.cache_creation_input_tokens)}  "
                   f"{_fmt_tokens(call.cache_read_input_tokens)}  "
                   f"{_fmt_tokens(call.total_input)}  "
                   f"{_fmt_tokens(call.output_tokens)}")
            if show_cost:
                row += _fmt_cost(call.cost(pricing))
            print(row)
        print()

    # ── Aggregated totals ─────────────────────────────────────────────────
    tot_in = sum(c.input_tokens for c in summary.api_calls)
    tot_cc = sum(c.cache_creation_input_tokens for c in summary.api_calls)
    tot_cr = sum(c.cache_read_input_tokens for c in summary.api_calls)
    tot_total = sum(c.total_input for c in summary.api_calls)
    tot_out = sum(c.output_tokens for c in summary.api_calls)
    tot_5m = sum(c.ephemeral_5m_input_tokens for c in summary.api_calls)
    tot_1h = sum(c.ephemeral_1h_input_tokens for c in summary.api_calls)

    print("Aggregated Token Usage (from API responses):")
    print(f"  API calls              :{_fmt_tokens(len(summary.api_calls))}")
    print(f"  input_tokens (uncached):{_fmt_tokens(tot_in)}")
    print(f"  cache_creation_input   :{_fmt_tokens(tot_cc)}")
    if tot_5m or tot_1h:
        print(f"    └─ 5-min writes      :{_fmt_tokens(tot_5m)}")
        print(f"    └─ 1-hour writes     :{_fmt_tokens(tot_1h)}")
    print(f"  cache_read_input       :{_fmt_tokens(tot_cr)}")
    print("  ─────────────────────────────────────")
    print(f"  total input tokens     :{_fmt_tokens(tot_total)}")
    print(f"  output_tokens          :{_fmt_tokens(tot_out)}")
    print()

    # ── Cost estimate ─────────────────────────────────────────────────────
    if show_cost:
        primary_model = ""
        if summary.models_seen:
            primary_model = max(summary.models_seen, key=summary.models_seen.get)
        pricing = _match_pricing(primary_model)

        if pricing:
            total_cost = sum(c.cost(pricing) or 0 for c in summary.api_calls)
            base_in, cw5m, cw1h, cr, out = pricing

            print(f"Cost Estimate (model: {primary_model}):")
            print(f"  Pricing: input=${base_in}/MTok  cache_write_5m=${cw5m}/MTok  "
                  f"cache_read=${cr}/MTok  output=${out}/MTok")

            per_m = 1_000_000
            cost_uncached_in = tot_in * base_in / per_m
            if tot_5m or tot_1h:
                cost_cache_write = (tot_5m * cw5m + tot_1h * cw1h) / per_m
            else:
                cost_cache_write = tot_cc * cw5m / per_m
            cost_cache_read = tot_cr * cr / per_m
            cost_output = tot_out * out / per_m

            print(f"  Uncached input         :{_fmt_cost(cost_uncached_in)}")
            print(f"  Cache write            :{_fmt_cost(cost_cache_write)}")
            print(f"  Cache read             :{_fmt_cost(cost_cache_read)}")
            print(f"  Output                 :{_fmt_cost(cost_output)}")
            print("  ─────────────────────────────────────")
            print(f"  Total                  :{_fmt_cost(total_cost)}")
        else:
            print(f"Cost Estimate: N/A (unknown model: {primary_model!r})")
        print()

    # ── Text estimation (supplementary) ───────────────────────────────────
    te = summary.text_est
    est_total = (te.user_prompt_tokens_est + te.tool_result_tokens_est
                 + te.assistant_text_tokens_est + te.assistant_tool_tokens_est)
    if est_total > 0:
        print("Text Content Estimates (chars/4, for reference only):")
        print(f"  User prompt text       : ~{te.user_prompt_tokens_est:,} tokens "
              f"({te.user_prompt_chars:,} chars)")
        print(f"  Tool results           : ~{te.tool_result_tokens_est:,} tokens "
              f"({te.tool_result_chars:,} chars)")
        print(f"  Assistant text output   : ~{te.assistant_text_tokens_est:,} tokens "
              f"({te.assistant_text_chars:,} chars)")
        print(f"  Assistant tool inputs   : ~{te.assistant_tool_tokens_est:,} tokens "
              f"({te.assistant_tool_input_chars:,} chars)")
        print()

    # ── Server tool usage ─────────────────────────────────────────────────
    tot_ws = sum(c.web_search_requests for c in summary.api_calls)
    tot_wf = sum(c.web_fetch_requests for c in summary.api_calls)
    if tot_ws or tot_wf:
        print("Server Tool Usage:")
        print(f"  Web search requests    : {tot_ws}")
        print(f"  Web fetch requests     : {tot_wf}")
        print()


# ── Batch table output ───────────────────────────────────────────────────────

def _shorten_name(name: str, maxlen: int = 50) -> str:
    """Shorten a filename for table display."""
    if len(name) <= maxlen:
        return name
    return name[:maxlen - 3] + "..."


def print_batch_table(summaries: List[FileSummary], show_cost: bool = True) -> None:
    """Print a one-row-per-file summary table for multiple trajectory files."""
    if not summaries:
        print("No matching files found.")
        return

    # Determine if any file has cost info
    has_cost = show_cost and any(s.models_seen for s in summaries)

    # Compute column values
    rows = []
    for s in summaries:
        fname = Path(s.filepath).name
        n_calls = len(s.api_calls)
        tot_in = sum(c.input_tokens for c in s.api_calls)
        tot_cc = sum(c.cache_creation_input_tokens for c in s.api_calls)
        tot_cr = sum(c.cache_read_input_tokens for c in s.api_calls)
        tot_total = sum(c.total_input for c in s.api_calls)
        tot_out = sum(c.output_tokens for c in s.api_calls)

        cost = None
        model = ""
        if s.models_seen:
            model = max(s.models_seen, key=s.models_seen.get)
            pricing = _match_pricing(model)
            if pricing:
                cost = sum(c.cost(pricing) or 0 for c in s.api_calls)

        rows.append((fname, model, n_calls, tot_in, tot_cc, tot_cr, tot_total, tot_out, cost))

    # Determine filename column width (adaptive)
    max_name_len = max(len(r[0]) for r in rows)
    name_col = min(max(max_name_len, 20), 60)

    # Header
    hdr_parts = [
        f"{'File':<{name_col}}",
        f"{'Model':>16}",
        f"{'Calls':>5}",
        f"{'Uncached':>10}",
        f"{'Cache_Wr':>10}",
        f"{'Cache_Rd':>10}",
        f"{'Total_In':>10}",
        f"{'Output':>10}",
    ]
    if has_cost:
        hdr_parts.append(f"{'Cost':>10}")
    hdr = "  ".join(hdr_parts)
    sep = "-" * len(hdr)

    print()
    print(hdr)
    print(sep)

    grand_calls = 0
    grand_in = 0
    grand_cc = 0
    grand_cr = 0
    grand_total = 0
    grand_out = 0
    grand_cost = 0.0

    for (fname, model, n_calls, tot_in, tot_cc, tot_cr, tot_total, tot_out, cost) in rows:
        display_name = _shorten_name(fname, name_col)

        # Shorten model name for display
        short_model = model.replace("claude-", "").replace("opus-", "o").replace("sonnet-", "s").replace("haiku-", "h") if model else ""

        parts = [
            f"{display_name:<{name_col}}",
            f"{short_model:>16}",
            f"{n_calls:>5}",
            f"{tot_in:>10,}",
            f"{tot_cc:>10,}",
            f"{tot_cr:>10,}",
            f"{tot_total:>10,}",
            f"{tot_out:>10,}",
        ]
        if has_cost:
            if cost is not None:
                parts.append(f"${cost:>9.4f}")
            else:
                parts.append(f"{'N/A':>10}")
        print("  ".join(parts))

        grand_calls += n_calls
        grand_in += tot_in
        grand_cc += tot_cc
        grand_cr += tot_cr
        grand_total += tot_total
        grand_out += tot_out
        if cost is not None:
            grand_cost += cost

    # Totals row
    print(sep)
    total_parts = [
        f"{'TOTAL':<{name_col}}",
        f"{'':>16}",
        f"{grand_calls:>5}",
        f"{grand_in:>10,}",
        f"{grand_cc:>10,}",
        f"{grand_cr:>10,}",
        f"{grand_total:>10,}",
        f"{grand_out:>10,}",
    ]
    if has_cost:
        total_parts.append(f"${grand_cost:>9.4f}")
    print("  ".join(total_parts))
    print()
    print(f"Files: {len(rows)}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Count LLM token usage from Claude Code trajectory JSONL files."
    )
    parser.add_argument("file", help="Path to a .jsonl file or a directory")
    parser.add_argument("--prefix", default=None,
                        help="When FILE is a directory, only process .jsonl files "
                             "whose name starts with this prefix")
    parser.add_argument("--detail", action="store_true", default=True,
                        help="Show per-API-call detail (default: on for single file)")
    parser.add_argument("--no-detail", action="store_true",
                        help="Hide per-API-call detail")
    parser.add_argument("--no-cost", action="store_true",
                        help="Hide cost estimates")
    parser.add_argument("--sort", default="name",
                        choices=["name", "cost", "output", "input", "calls"],
                        help="Sort order for batch table (default: name)")
    args = parser.parse_args()

    target = Path(args.file)

    # ── Directory / batch mode ────────────────────────────────────────────
    if target.is_dir():
        pattern = "*.jsonl"
        files = sorted(target.glob(pattern))
        if args.prefix:
            files = [f for f in files if f.name.startswith(args.prefix)]
        if not files:
            pfx_msg = f" with prefix '{args.prefix}'" if args.prefix else ""
            print(f"No .jsonl files found in {target}{pfx_msg}", file=sys.stderr)
            sys.exit(1)

        summaries = []
        for f in files:
            try:
                summaries.append(parse_file(str(f)))
            except Exception as e:
                print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)

        # Sort
        sort_key = {
            "name": lambda s: Path(s.filepath).name,
            "cost": lambda s: -(sum(c.cost(_match_pricing(
                max(s.models_seen, key=s.models_seen.get) if s.models_seen else ""
            )) or 0 for c in s.api_calls)),
            "output": lambda s: -sum(c.output_tokens for c in s.api_calls),
            "input": lambda s: -sum(c.total_input for c in s.api_calls),
            "calls": lambda s: -len(s.api_calls),
        }
        summaries.sort(key=sort_key.get(args.sort, sort_key["name"]))

        print_batch_table(summaries, show_cost=not args.no_cost)
        return

    # ── Single file mode ──────────────────────────────────────────────────
    if not target.exists():
        print(f"Error: File not found: {target}", file=sys.stderr)
        sys.exit(1)

    summary = parse_file(str(target))
    print_summary(
        summary,
        show_detail=not args.no_detail,
        show_cost=not args.no_cost,
    )


if __name__ == "__main__":
    main()
