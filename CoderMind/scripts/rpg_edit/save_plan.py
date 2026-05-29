#!/usr/bin/env python3
"""Save an EditPlan JSON document to ``RPG_EDIT_PLAN_FILE``.

Reads JSON from stdin, validates that it parses, and writes it to
``~/.cmind/workspaces/<workspace-id>/data/rpg_edit_plan.json``.  Slash-command
templates use this so they never need to know the physical (home-dir)
location of the workspace.

Usage (typical AI-agent invocation)::

    cat << 'PLAN_EOF' | cmind script rpg_edit/save_plan.py
    { "feature_changes": [...], "code_changes": [...] }
    PLAN_EOF

On success prints the absolute path of the saved file (one line) on
stdout and exits 0.  On JSON parse error exits 2 with the parser
message on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import RPG_EDIT_PLAN_FILE  # noqa: E402


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        print(f"Output path: {RPG_EDIT_PLAN_FILE}")
        return 0
    raw = sys.stdin.read()
    if not raw.strip():
        print("save_plan: stdin is empty", file=sys.stderr)
        return 2
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"save_plan: invalid JSON on stdin: {exc}", file=sys.stderr)
        return 2
    RPG_EDIT_PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RPG_EDIT_PLAN_FILE.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False)
    )
    print(str(RPG_EDIT_PLAN_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
