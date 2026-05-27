#!/usr/bin/env python3
"""Coarse-grained LLM token usage summary by workflow stage.

Usage:
    python llm_usage_count_coarse.py <trajectory_directory> [--no-cost]

Groups trajectory files by workflow stage prefixes:
  - feature       : feature_spec, feature_build, feature_refactor, feature_edit
  - file_design   : build_skeleton
  - design_data_flow
  - design_base_classes
  - design_interfaces
  - plan          : plan_tasks
  - gen_code      : code_gen (excluding tests)
  - gen_test      : code_gen tests
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Import from the detailed counter
from llm_usage_count import (
    parse_file,
    FileSummary,
    _match_pricing,
)


# ── Stage definitions ─────────────────────────────────────────────────────────

STAGE_PREFIXES = {
    "feature_design": ["feature_spec", "feature_build", "feature_refactor", "feature_edit"],
    "build_skeleton": ["file_design"],
    "design_data_flow": ["design_data_flow"],
    "design_base_classes": ["design_base_classes"],
    "design_interfaces": ["design_interfaces"],
    "plan_tasks": ["plan_"],
    "gen_code": ["gen_code"],
    "gen_test": ["gen_test"],
}

# Display order
STAGE_ORDER = [
    "feature_design",
    "build_skeleton",
    "design_data_flow",
    "design_base_classes",
    "design_interfaces",
    "plan_tasks",
    "gen_code",
    "gen_test",
]


@dataclass
class StageSummary:
    """Aggregated statistics for one workflow stage."""
    stage: str
    file_count: int = 0
    api_calls: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    models: Dict[str, int] = field(default_factory=dict)

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


def classify_file(filename: str) -> Optional[str]:
    """Determine which stage a trajectory file belongs to."""
    name = filename.lower()

    # Special case: gen_test (code_gen with "test" in name)
    if name.startswith("code_gen") and "test" in name:
        return "gen_test"

    # Match by prefix
    for stage, prefixes in STAGE_PREFIXES.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                return stage

    return None


def aggregate_by_stage(summaries: List[FileSummary]) -> Dict[str, StageSummary]:
    """Group file summaries by workflow stage."""
    stages: Dict[str, StageSummary] = {}

    for stage in STAGE_ORDER:
        stages[stage] = StageSummary(stage=stage)

    for s in summaries:
        fname = Path(s.filepath).name
        stage = classify_file(fname)
        if stage is None:
            continue

        ss = stages[stage]
        ss.file_count += 1
        ss.api_calls += len(s.api_calls)

        for call in s.api_calls:
            ss.input_tokens += call.input_tokens
            ss.cache_creation_tokens += call.cache_creation_input_tokens
            ss.cache_read_tokens += call.cache_read_input_tokens
            ss.output_tokens += call.output_tokens

            # Cost
            pricing = _match_pricing(call.model)
            if pricing:
                c = call.cost(pricing)
                if c:
                    ss.cost += c

            # Model tracking
            if call.model:
                ss.models[call.model] = ss.models.get(call.model, 0) + 1

    return stages


def print_coarse_table(stages: Dict[str, StageSummary], show_cost: bool = True) -> None:
    """Print a summary table grouped by workflow stage."""
    # Header
    hdr_parts = [
        f"{'Stage':<20}",
        f"{'Files':>5}",
        f"{'Calls':>6}",
        f"{'Uncached':>12}",
        f"{'Cache_Wr':>12}",
        f"{'Cache_Rd':>12}",
        f"{'Total_In':>12}",
        f"{'Output':>12}",
    ]
    if show_cost:
        hdr_parts.append(f"{'Cost':>12}")
    hdr = "  ".join(hdr_parts)
    sep = "-" * len(hdr)

    print()
    print("LLM Usage by Workflow Stage")
    print("=" * len(hdr))
    print()
    print(hdr)
    print(sep)

    # Totals
    total_files = 0
    total_calls = 0
    total_in = 0
    total_cc = 0
    total_cr = 0
    total_total = 0
    total_out = 0
    total_cost = 0.0

    for stage in STAGE_ORDER:
        ss = stages[stage]
        if ss.file_count == 0:
            continue

        parts = [
            f"{stage:<20}",
            f"{ss.file_count:>5}",
            f"{ss.api_calls:>6}",
            f"{ss.input_tokens:>12,}",
            f"{ss.cache_creation_tokens:>12,}",
            f"{ss.cache_read_tokens:>12,}",
            f"{ss.total_input:>12,}",
            f"{ss.output_tokens:>12,}",
        ]
        if show_cost:
            parts.append(f"${ss.cost:>11.4f}")
        print("  ".join(parts))

        total_files += ss.file_count
        total_calls += ss.api_calls
        total_in += ss.input_tokens
        total_cc += ss.cache_creation_tokens
        total_cr += ss.cache_read_tokens
        total_total += ss.total_input
        total_out += ss.output_tokens
        total_cost += ss.cost

    # Total row
    print(sep)
    total_parts = [
        f"{'TOTAL':<20}",
        f"{total_files:>5}",
        f"{total_calls:>6}",
        f"{total_in:>12,}",
        f"{total_cc:>12,}",
        f"{total_cr:>12,}",
        f"{total_total:>12,}",
        f"{total_out:>12,}",
    ]
    if show_cost:
        total_parts.append(f"${total_cost:>11.4f}")
    print("  ".join(total_parts))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Coarse-grained LLM token usage summary by workflow stage."
    )
    parser.add_argument("directory", help="Path to trajectory directory")
    parser.add_argument("--no-cost", action="store_true", help="Hide cost estimates")
    args = parser.parse_args()

    target = Path(args.directory)
    if not target.is_dir():
        print(f"Error: Not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    # Find all JSONL files
    files = sorted(target.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {target}", file=sys.stderr)
        sys.exit(1)

    # Parse all files
    summaries = []
    for f in files:
        try:
            summaries.append(parse_file(str(f)))
        except Exception as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)

    if not summaries:
        print("No valid trajectory files found.", file=sys.stderr)
        sys.exit(1)

    # Aggregate by stage
    stages = aggregate_by_stage(summaries)

    # Print table
    print_coarse_table(stages, show_cost=not args.no_cost)

    # Show unclassified files
    unclassified = []
    for s in summaries:
        fname = Path(s.filepath).name
        if classify_file(fname) is None:
            unclassified.append(fname)

    if unclassified:
        print(f"Unclassified files ({len(unclassified)}):")
        for f in unclassified[:10]:
            print(f"  - {f}")
        if len(unclassified) > 10:
            print(f"  ... and {len(unclassified) - 10} more")
        print()


if __name__ == "__main__":
    main()
