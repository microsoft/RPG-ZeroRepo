#!/usr/bin/env python3
"""Analyze impact of modifying RPG feature nodes via dep_graph.

Given one or more node IDs, outputs callers, callees, inheritance,
and imports from the dep_graph to help plan the scope of changes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# This file lives in ``scripts/rpg_edit/``; go up two levels to land
# on ``scripts/`` so ``common.*``, ``rpg.*`` etc. import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.paths import REPO_RPG_FILE, RPG_EDIT_IMPACT_FILE  # noqa: E402


def analyze_impact(svc, node_ids: List[str]) -> Dict:
    """Analyze dep_graph impact for given RPG feature node IDs."""
    if svc.rpg.dep_graph is None:
        return {"error": "No dep_graph loaded"}

    G = svc.rpg.dep_graph.G
    results = {}

    for rpg_nid in node_ids:
        node = svc.rpg._node_index.get(rpg_nid)
        if node is None:
            results[rpg_nid] = {"error": f"Node not found: {rpg_nid}"}
            continue

        dep_nids = svc.rpg._feature_to_dep_map.get(rpg_nid, [])
        if not dep_nids:
            results[rpg_nid] = {
                "name": node.name,
                "dep_nodes": [],
                "message": "No dep_graph mapping for this node",
            }
            continue

        imports = []
        callers = []
        callees = []
        inheritance = []
        affected_files = set()

        for dep_nid in dep_nids:
            if dep_nid not in G.nodes:
                continue

            # Out-edges
            for _, dst, attrs in G.out_edges(dep_nid, data=True):
                etype = attrs.get("type", "")
                dst_attrs = G.nodes.get(dst, {})
                entry = {
                    "name": dst_attrs.get("name", dst),
                    "node_id": dst,
                    "type": dst_attrs.get("type", ""),
                }
                if etype == "imports":
                    entry["module"] = dst_attrs.get("module", "")
                    imports.append(entry)
                elif etype == "invokes":
                    callees.append(entry)
                    # Track affected file
                    file_part = dst.split(":")[0] if ":" in dst else dst
                    affected_files.add(file_part)
                elif etype == "inherits":
                    inheritance.append({"direction": "extends", **entry})

            # In-edges
            for src, _, attrs in G.in_edges(dep_nid, data=True):
                etype = attrs.get("type", "")
                src_attrs = G.nodes.get(src, {})
                entry = {
                    "name": src_attrs.get("name", src),
                    "node_id": src,
                    "type": src_attrs.get("type", ""),
                }
                if etype == "invokes":
                    callers.append(entry)
                    file_part = src.split(":")[0] if ":" in src else src
                    affected_files.add(file_part)
                elif etype == "inherits":
                    inheritance.append({"direction": "extended_by", **entry})
                    file_part = src.split(":")[0] if ":" in src else src
                    affected_files.add(file_part)

        results[rpg_nid] = {
            "name": node.name,
            "dep_nodes": dep_nids,
            "imports": imports,
            "callers": callers,
            "callees": callees,
            "inheritance": inheritance,
            "affected_files": sorted(affected_files),
            "impact_summary": {
                "total_callers": len(callers),
                "total_callees": len(callees),
                "total_inheritance": len(inheritance),
                "affected_file_count": len(affected_files),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze dep_graph impact for RPG nodes")
    parser.add_argument("--node-id", action="append", required=True,
                        help="RPG node ID(s) to analyze (can specify multiple)")
    parser.add_argument("--rpg", type=Path,
                        default=REPO_RPG_FILE)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help=f"Also write the JSON result to "
                             f"{RPG_EDIT_IMPACT_FILE} so downstream "
                             f"steps (review.py) can pick it up.")
    args = parser.parse_args()

    # Capture log records for post-mortem inspection of rpg_edit issues.
    from common.logging_setup import setup_file_logging
    setup_file_logging("rpg_edit")

    from rpg.service import RPGService
    svc = RPGService.load(str(args.rpg))
    results = analyze_impact(svc, args.node_id)

    output = {"type": "impact_analysis", "results": results}
    if args.save:
        RPG_EDIT_IMPACT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RPG_EDIT_IMPACT_FILE.write_text(
            json.dumps(output, indent=2, ensure_ascii=False)
        )
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for nid, info in results.items():
            if "error" in info:
                print(f"[ERROR] {nid}: {info['error']}")
                continue
            print(f"\n=== {info['name']} ({nid}) ===")
            print(f"  Dep nodes: {info['dep_nodes']}")
            if info["callers"]:
                print(f"  Callers ({len(info['callers'])}):")
                for c in info["callers"][:10]:
                    print(f"    - {c['name']} ({c['type']}) @ {c['node_id']}")
            if info["callees"]:
                print(f"  Callees ({len(info['callees'])}):")
                for c in info["callees"][:10]:
                    print(f"    - {c['name']} ({c['type']}) @ {c['node_id']}")
            if info["inheritance"]:
                print("  Inheritance:")
                for inh in info["inheritance"]:
                    print(f"    - {inh['direction']} {inh['name']}")
            print(f"  Affected files: {info['affected_files']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
