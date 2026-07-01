#!/usr/bin/env python3
"""Apply an EditPlan to RPG feature graph and code.

Reads an EditPlan JSON, applies feature_changes to the RPG, applies
code_changes as diffs, refreshes the embedded dep_graph, runs related
tests, and outputs a result JSON. Supports rollback on test failure.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import (  # noqa: E402
    REPO_RPG_FILE,
    DEP_GRAPH_FILE,
    REPO_DIR,
    RPG_EDIT_PLAN_FILE,
    RPG_EDIT_APPLY_RESULT_FILE,
    cmd_for,
)
from common.git_utils import read_head  # noqa: E402


def _backup(rpg_path: Path, dep_graph_path: Path, ts: str) -> Dict[str, str]:
    """Create file-only backups before applying changes.

    Does NOT use ``git stash`` — that would stash all uncommitted work
    in the repository, not just the rpg_edit changes.
    """
    backups = {}
    if rpg_path.exists():
        dst = rpg_path.with_suffix(f".before-edit-{ts}.json")
        shutil.copy2(rpg_path, dst)
        backups["rpg"] = str(dst)
    if dep_graph_path.exists():
        dst = dep_graph_path.with_suffix(f".before-edit-{ts}.json")
        shutil.copy2(dep_graph_path, dst)
        backups["dep_graph"] = str(dst)
    return backups


def _rollback(backups: Dict[str, str], rpg_path: Path, dep_graph_path: Path) -> None:
    """Restore file backups."""
    if "rpg" in backups:
        shutil.copy2(backups["rpg"], rpg_path)
    if "dep_graph" in backups:
        shutil.copy2(backups["dep_graph"], dep_graph_path)


def _rollback_command(timestamp: str | None, before_state: dict | None) -> str | None:
    if not timestamp:
        return None
    command = f"{cmd_for('rpg_edit/apply.py')} --rollback {shlex.quote(str(timestamp))}"
    branch = before_state.get("head_branch") if isinstance(before_state, dict) else None
    if branch:
        command += f" --rollback-branch {shlex.quote(str(branch))}"
    return command


def _rollback_path(backups: Dict[str, str]) -> str | None:
    return backups.get("rpg") or backups.get("dep_graph")


def _persist_apply_result(result: Dict[str, Any]) -> None:
    RPG_EDIT_APPLY_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RPG_EDIT_APPLY_RESULT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _record_apply_result(
    result: Dict[str, Any],
    *,
    backup_timestamp: str | None = None,
    backups: Dict[str, str] | None = None,
    applied_features: list | None = None,
    dep_graph_refreshed: bool | None = None,
    before_state: dict | None = None,
    confirmed: bool | None = None,
) -> Dict[str, Any]:
    backups = backups or {}
    if backup_timestamp is not None:
        result.setdefault("backup_timestamp", backup_timestamp)
        result.setdefault("rollback_command", _rollback_command(backup_timestamp, before_state))
    result.setdefault("backups", backups)
    result.setdefault("applied_features", applied_features or [])
    if dep_graph_refreshed is not None:
        result.setdefault("dep_graph_refreshed", dep_graph_refreshed)
    rollback_path = _rollback_path(backups)
    if rollback_path:
        result.setdefault("rollback_path", rollback_path)
    if before_state is not None:
        result.setdefault("before_state", before_state)
    if confirmed is not None:
        result.setdefault("confirmed", confirmed)
    _persist_apply_result(result)
    return result


def apply_feature_changes(svc, changes: list) -> list:
    """Apply feature_changes to the RPG in memory.

    Returns list of applied change summaries.
    """
    applied = []
    for change in changes:
        node_id = change.get("node_id")
        action = change.get("action")
        patch = change.get("patch", {})

        node = svc.rpg._node_index.get(node_id)

        if action == "modify" and node is not None:
            if "name" in patch:
                node.name = patch["name"]
            if node.meta:
                for k, v in patch.items():
                    if k.startswith("meta."):
                        setattr(node.meta, k[5:], v)
            applied.append({"node_id": node_id, "action": "modified"})

        elif action == "delete" and node is not None:
            parent = node.parent()
            if parent:
                parent.remove_child(node)
            svc.rpg._node_index.pop(node_id, None)
            applied.append({"node_id": node_id, "action": "deleted"})

        elif action == "add":
            from rpg.models import Node, NodeMetaData, uuid8
            parent_id = change.get("parent_id")
            parent = svc.rpg._node_index.get(parent_id)
            if parent is None:
                applied.append({"node_id": node_id, "action": "add_failed",
                                "reason": f"parent {parent_id} not found"})
                continue
            name = patch.get("name", "new_node")
            new_id = f"{name}_{uuid8()}"
            new_node = Node(
                id=new_id, name=name,
                node_type=patch.get("node_type", "feature"),
                level=parent.level + 1 if parent.level is not None else 1,
                meta=NodeMetaData(
                    path=patch.get("meta.path"),
                    type_name=patch.get("meta.type_name"),
                    generator="rpg_edit",
                ),
                _graph=svc.rpg,
            )
            parent.add_child(new_node)
            svc.rpg._node_index[new_id] = new_node
            applied.append({"node_id": new_id, "action": "added"})
        else:
            applied.append({"node_id": node_id, "action": action,
                            "status": "skipped", "reason": "node not found or unknown action"})

    return applied


def main():
    parser = argparse.ArgumentParser(description="Apply EditPlan to RPG + code")
    parser.add_argument("--plan", type=Path, default=RPG_EDIT_PLAN_FILE,
                        help="Path to rpg_edit_plan.json (default: %(default)s)")
    parser.add_argument("--rpg", type=Path,
                        default=REPO_RPG_FILE)
    parser.add_argument("--dep-graph", type=Path,
                        default=DEP_GRAPH_FILE)
    parser.add_argument("--repo", type=Path, default=None,
                        help="Repository root for code changes and dep_graph refresh")
    parser.add_argument("--phase", choices=["rpg-only", "dep-refresh", "all"],
                        default="all",
                        help="Execution phase: rpg-only (apply feature_changes, "
                             "save RPG, no dep_graph refresh), dep-refresh "
                             "(refresh dep_graph only), all (legacy: everything)")
    parser.add_argument("--backup-ts", type=str, default=None,
                        help="Reuse existing backup timestamp (skip new backup)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--confirmed", action="store_const", const=True, default=None,
                        help="Record that the apply boundary was explicitly confirmed")
    parser.add_argument("--rollback", type=str, default=None,
                        help="Rollback to a previous timestamp backup")
    parser.add_argument("--rollback-branch", type=str, default=None,
                        help="Together with --rollback: also force-delete the "
                             "named git branch in the project repo (typically the "
                             "rpg-edit/<id> branch created by /cmind.rpg_edit). "
                             "Has no effect without --rollback.")
    parser.add_argument("--repo-dir", type=Path, default=None,
                        help="Project repo for --rollback-branch operation. "
                             "Defaults to common.paths.REPO_DIR.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    before_state = read_head(REPO_DIR)

    # Handle rollback
    if args.rollback:
        rpg_backup = args.rpg.with_suffix(f".before-edit-{args.rollback}.json")
        dg_backup = args.dep_graph.with_suffix(f".before-edit-{args.rollback}.json")
        restored = []
        if rpg_backup.exists():
            shutil.copy2(rpg_backup, args.rpg)
            restored.append(str(args.rpg))
        if dg_backup.exists():
            shutil.copy2(dg_backup, args.dep_graph)
            restored.append(str(args.dep_graph))

        # Optional companion: drop the rpg-edit/<id> branch left behind by
        # the slash-command's failure path.  Reported per-branch so the
        # caller can surface partial success.
        branch_result: Dict[str, Any] = {}
        if args.rollback_branch:
            repo_dir = args.repo_dir or REPO_DIR
            try:
                proc = subprocess.run(
                    ["git", "-C", str(repo_dir), "branch", "-D", args.rollback_branch],
                    capture_output=True, text=True, timeout=10,
                )
                branch_result = {
                    "name": args.rollback_branch,
                    "deleted": proc.returncode == 0,
                    "message": (proc.stdout + proc.stderr).strip(),
                }
            except Exception as exc:
                branch_result = {
                    "name": args.rollback_branch,
                    "deleted": False,
                    "message": f"git invocation failed: {exc}",
                }

        rollback_backups = {}
        if rpg_backup.exists():
            rollback_backups["rpg"] = str(rpg_backup)
        if dg_backup.exists():
            rollback_backups["dep_graph"] = str(dg_backup)
        result = {"type": "rollback", "restored": restored,
                  "timestamp": args.rollback}
        if branch_result:
            result["branch"] = branch_result
        _record_apply_result(
            result,
            backup_timestamp=args.rollback,
            backups=rollback_backups,
            dep_graph_refreshed=False,
            before_state=before_state,
            confirmed=args.confirmed,
        )
        print(json.dumps(result, indent=2) if args.json else
              f"Rolled back: {restored}" +
              (f"; branch={branch_result}" if branch_result else ""))
        return 0

    # Load plan
    if not args.plan.exists():
        result = {"type": "error", "message": f"Plan not found: {args.plan}"}
        _record_apply_result(
            result,
            dep_graph_refreshed=False,
            before_state=before_state,
            confirmed=args.confirmed,
        )
        print(json.dumps(result) if args.json else f"Error: {result['message']}")
        return 1

    plan = json.loads(args.plan.read_text())

    # Backup: skip if reusing existing timestamp (dep-refresh phase)
    if args.backup_ts:
        ts = args.backup_ts
        backups = {}
    else:
        ts = str(int(time.time()))
        backups = _backup(args.rpg, args.dep_graph, ts)

    from rpg.service import RPGService

    svc = RPGService.load(str(args.rpg))

    # Ensure dep_graph is always embedded in rpg.json (single-file mode)
    svc.rpg._dep_graph_file = None

    # --- Phase: rpg-only or all → apply feature_changes ---
    applied_features = []
    dep_graph_refreshed = False
    if args.phase in ("rpg-only", "all"):
        feature_changes = plan.get("feature_changes", [])
        applied_features = apply_feature_changes(svc, feature_changes) if feature_changes else []
        svc.save(str(args.rpg))

        if args.phase == "rpg-only":
            result = {
                "type": "rpg_updated",
                "applied_features": applied_features,
                "backup_timestamp": ts,
                "backups": backups,
            }
            _record_apply_result(
                result,
                backup_timestamp=ts,
                backups=backups,
                applied_features=applied_features,
                dep_graph_refreshed=dep_graph_refreshed,
                before_state=before_state,
                confirmed=args.confirmed,
            )
            print(json.dumps(result, indent=2) if args.json else
                  f"RPG updated ({len(applied_features)} features). Backup: {ts}")
            return 0

    # --- Phase: dep-refresh or all → refresh dep_graph ---
    if args.phase in ("dep-refresh", "all"):
        # Workspace root is the project repo root.  Explicit ``--repo``
        # still wins for tests / brownfield setups where the code lives
        # somewhere unusual.
        repo_path = args.repo or REPO_DIR

        try:
            svc.refresh_dep_graph(
                str(repo_path),
                workspace_root=str(Path.cwd()),
            )
            dep_graph_refreshed = True
        except Exception as exc:
            dep_graph_refreshed = False
            if args.phase == "dep-refresh":
                result = {
                    "type": "error",
                    "message": f"dep_graph refresh failed: {exc}",
                    "backup_timestamp": ts,
                }
                _record_apply_result(
                    result,
                    backup_timestamp=ts,
                    backups=backups,
                    applied_features=applied_features,
                    dep_graph_refreshed=dep_graph_refreshed,
                    before_state=before_state,
                    confirmed=args.confirmed,
                )
                print(json.dumps(result, indent=2) if args.json else
                      f"Error: {result['message']}")
                return 1

        svc.save(str(args.rpg))

        if args.phase == "dep-refresh":
            result = {
                "type": "dep_refreshed",
                "dep_graph_refreshed": dep_graph_refreshed,
                "backup_timestamp": ts,
            }
            _record_apply_result(
                result,
                backup_timestamp=ts,
                backups=backups,
                applied_features=applied_features,
                dep_graph_refreshed=dep_graph_refreshed,
                before_state=before_state,
                confirmed=args.confirmed,
            )
            print(json.dumps(result, indent=2) if args.json else
                  f"dep_graph refreshed: {dep_graph_refreshed}. Backup: {ts}")
            return 0

    # --- Phase: all → run tests ---
    code_changes = plan.get("code_changes", [])
    test_result = {"passed": True, "output": ""}
    if not args.skip_tests and code_changes:
        test_files = set()
        for cc in code_changes:
            fp = cc.get("file_path", "")
            if fp.endswith(".py"):
                base = Path(fp).stem
                test_files.add(f"test_{base}")

        if test_files:
            pattern = " or ".join(test_files)
            cmd = [sys.executable, "-m", "pytest", "-x", "-q",
                   "-k", pattern, "--timeout=30"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            test_result["passed"] = proc.returncode == 0
            test_result["output"] = (proc.stdout + proc.stderr)[-2000:]

            if not test_result["passed"]:
                _rollback(backups, args.rpg, args.dep_graph)
                result = {
                    "type": "test_failed",
                    "applied_features": applied_features,
                    "test_output": test_result["output"],
                    "test_result": test_result,
                    "rolled_back": True,
                    "backup_timestamp": ts,
                }
                _record_apply_result(
                    result,
                    backup_timestamp=ts,
                    backups=backups,
                    applied_features=applied_features,
                    dep_graph_refreshed=dep_graph_refreshed,
                    before_state=before_state,
                    confirmed=args.confirmed,
                )
                print(json.dumps(result, indent=2) if args.json else
                      f"Tests failed. Rolled back to {ts}.")
                return 1

    result = {
        "type": "success",
        "applied_features": applied_features,
        "code_changes_planned": len(code_changes),
        "dep_graph_refreshed": dep_graph_refreshed,
        "test_result": test_result,
        "backup_timestamp": ts,
        "backups": backups,
    }
    _record_apply_result(
        result,
        backup_timestamp=ts,
        backups=backups,
        applied_features=applied_features,
        dep_graph_refreshed=dep_graph_refreshed,
        before_state=before_state,
        confirmed=args.confirmed,
    )
    print(json.dumps(result, indent=2) if args.json else "EditPlan applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
