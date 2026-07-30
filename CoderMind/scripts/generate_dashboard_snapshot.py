#!/usr/bin/env python3
"""Build the renderer-independent CoderMind dashboard snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from common.dashboard_snapshot import (  # noqa: E402
    build_dashboard_snapshot,
    write_dashboard_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CoderMind dashboard snapshot JSON")
    parser.add_argument("--output", type=Path, default=None, help="Override snapshot output path")
    parser.add_argument("--print", action="store_true", dest="print_snapshot", help="Print full snapshot JSON")
    args = parser.parse_args()

    snapshot = build_dashboard_snapshot()
    output = write_dashboard_snapshot(snapshot, args.output)
    if args.print_snapshot:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "status": "success",
            "snapshot": str(output),
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