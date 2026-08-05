#!/usr/bin/env python3
"""Build the CoderMind dashboard snapshot and publish its static report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from common.dashboard_snapshot import (  # noqa: E402
    build_dashboard_snapshot,
    write_dashboard_snapshot,
)
from common.dashboard_report import write_dashboard_report  # noqa: E402
from common.activity_events import record_completed_activity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the CoderMind dashboard snapshot and static report")
    parser.add_argument("--output", type=Path, default=None, help="Override snapshot output path")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Override static report output directory")
    parser.add_argument("--print", action="store_true", dest="print_snapshot", help="Print full snapshot JSON")
    args = parser.parse_args()

    started = time.perf_counter()
    # Bootstrap report assets first. In particular, write_dashboard_report
    # materializes rpg.html from the snapshot graph. Recollect afterwards so
    # the published snapshot describes the final artifact set, not the state
    # immediately before its own visualization was written.
    snapshot = build_dashboard_snapshot()
    write_dashboard_report(snapshot, args.reports_dir)
    snapshot = build_dashboard_snapshot()
    output = write_dashboard_snapshot(snapshot, args.output)
    report = write_dashboard_report(snapshot, args.reports_dir)
    record_completed_activity(
        "report.snapshot",
        "dashboard snapshot",
        logical_key="dashboard-snapshot-frozen",
        trigger="script",
        duration_ms=(time.perf_counter() - started) * 1000,
        fields={
            "snapshot_path": str(output),
            "report_path": str(report.report_html),
            "run_count": len(snapshot["runs"]),
            "history_root_count": int(snapshot.get("history", {}).get("summary", {}).get("root_count") or 0),
        },
    )
    if args.print_snapshot:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "status": "success",
            "snapshot": str(output),
            "report": str(report.report_html),
            "runs": len(snapshot["runs"]),
            "pipeline_steps": len(snapshot["pipeline"]),
            "source_warnings": sum(
                1
                for source in snapshot["source_health"]
                if source["status"] in {"partial", "invalid", "unreadable"}
            ),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())