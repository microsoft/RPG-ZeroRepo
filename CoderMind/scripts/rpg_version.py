#!/usr/bin/env python3
"""Read (or diff) a specific historical RPG version from the meta-git on demand.

The dashboard lists version metadata cheaply (``collect_rpg_history``); the full
``rpg.json`` for a given commit is fetched only when requested — here — so the
snapshot stays small while the complete, git-compressed history is retained.

Usage:
    cmind script rpg_version.py --history                  # list versions
  cmind script rpg_version.py --commit <sha>             # print rpg.json at a version
  cmind script rpg_version.py --commit <sha> --output f  # write it to f
  cmind script rpg_version.py --commit <sha> --diff      # node/dep diff vs parent
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from common.paths import DATA_DIR, RPG_FILE  # noqa: E402
from common.rpg_diff import (  # noqa: E402
    previous_rpg_version,
    read_rpg_version,
    semantic_rpg_diff,
)


def _meta_root() -> Path:
    return DATA_DIR.parent


def _rel() -> str:
    try:
        return RPG_FILE.relative_to(_meta_root()).as_posix()
    except ValueError:
        return "data/rpg.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_meta_root()), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _rpg_at(ref: str) -> dict:
    value = read_rpg_version(_meta_root(), _rel(), ref)
    if value is None:
        raise ValueError(f"cannot read {ref}:{_rel()}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or diff a historical RPG version from the meta-git")
    parser.add_argument("--commit", help="meta-git commit sha of the version to read")
    parser.add_argument("--parent", help="ref to diff against (default: <commit>^)")
    parser.add_argument("--diff", action="store_true", help="print a semantic node/edge diff vs the previous RPG version")
    parser.add_argument("--history", "--list", action="store_true", dest="list_versions", help="list available versions and exit")
    parser.add_argument("--output", type=Path, help="write the version's rpg.json to this path instead of stdout")
    args = parser.parse_args()

    rel = _rel()
    if args.list_versions:
        try:
            log = _git("log", "--max-count=200", "--format=%H%x1f%cI%x1f%s", "--", rel)
        except (OSError, subprocess.CalledProcessError):
            print(json.dumps({"status": "unavailable", "reason": "no meta-git history"}))
            return 1
        rows = []
        for line in log.splitlines():
            parts = line.split("\x1f")
            if len(parts) >= 3:
                rows.append({"commit": parts[0], "committed_at": parts[1], "message": parts[2]})
        print(json.dumps({"status": "ok", "versions": rows}, ensure_ascii=False, indent=2))
        return 0

    if not args.commit:
        parser.error("--commit is required unless --list is given")

    try:
        current = _rpg_at(args.commit)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "reason": f"cannot read {args.commit}: {exc}"}))
        return 1

    if args.diff:
        parent_ref = args.parent or previous_rpg_version(_meta_root(), rel, args.commit)
        try:
            parent = _rpg_at(parent_ref) if parent_ref else {}
        except (OSError, ValueError):
            parent = {}
        diff = semantic_rpg_diff(
            parent,
            current,
            commit=args.commit,
            parent_commit=parent_ref,
        )
        print(json.dumps({"status": "ok", **diff}, ensure_ascii=False, indent=2))
        return 0

    payload = json.dumps(current, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "commit": args.commit, "output": str(args.output), "bytes": len(payload)}))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
