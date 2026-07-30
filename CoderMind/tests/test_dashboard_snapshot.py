from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.dashboard_snapshot import (
    DashboardSources,
    build_dashboard_snapshot,
    write_dashboard_snapshot,
)


def _sources(tmp_path: Path) -> DashboardSources:
    return DashboardSources(
        workspace_root=tmp_path / "demo",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        run_events_file=tmp_path / "logs" / "run_events.jsonl",
        rpg_file=tmp_path / "data" / "rpg.json",
        snapshot_file=tmp_path / "data" / "dashboard_snapshot.json",
    )


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_aggregates_run_stage_and_enrichment(tmp_path):
    sources = _sources(tmp_path)
    rows = [
        {"event_type": "run_started", "run_id": "run-1", "command": "encode", "status": "running", "started_at": "2026-07-29T10:00:00Z"},
        {"event_type": "stage_started", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "sequence": 1, "stage": "parse_rpg", "phase": "encoder", "attempt": 1, "status": "running", "started_at": "2026-07-29T10:00:00Z"},
        {"event_type": "stage_progress", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "metrics": {"completed": 2, "total": 3}},
        {"event_type": "stage_finished", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "sequence": 1, "stage": "parse_rpg", "phase": "encoder", "attempt": 1, "status": "success", "started_at": "2026-07-29T10:00:00Z", "finished_at": "2026-07-29T10:00:04Z", "duration_s": 4.2, "metrics": {"node_count": 90}, "error": None},
        {"event_type": "stage_enriched", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "tokens": {"total": 120}, "model": "test-model"},
        {"event_type": "run_finished", "run_id": "run-1", "command": "encode", "status": "success", "started_at": "2026-07-29T10:00:00Z", "finished_at": "2026-07-29T10:00:05Z", "duration_s": 5.0, "metrics": {"node_count": 90}, "error": None},
    ]
    _write_events(sources.run_events_file, rows)

    snapshot = build_dashboard_snapshot(sources, generated_at="2026-07-29T11:00:00Z")
    run = snapshot["runs"][0]
    stage = run["stages"][0]
    assert snapshot["current_state"]["status"] == "success"
    assert stage["status"] == "success"
    assert stage["duration_s"] == 4.2
    assert stage["metrics"] == {"node_count": 90}
    assert stage["progress"] == [{"completed": 2, "total": 3}]
    assert stage["tokens"] == {"total": 120}
    assert stage["model"] == "test-model"
    assert run["changes"]["available"] is False
    assert run["changes"]["quality"] == "missing"
    assert [check["name"] for check in run["verification"]] == ["run lifecycle", "parse_rpg"]
    assert all(check["status"] == "success" for check in run["verification"])
    assert snapshot["trends"]["by_command"]["encode"][0]["tokens_total"] == 120


def test_marks_finished_run_with_unfinished_stage_as_warning(tmp_path):
    sources = _sources(tmp_path)
    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-1", "command": "encode", "status": "running", "started_at": "2026-07-29T10:00:00Z"},
        {"event_type": "stage_started", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "sequence": 1, "stage": "visualize", "status": "running", "started_at": "2026-07-29T10:00:01Z"},
        {"event_type": "run_finished", "run_id": "run-1", "command": "encode", "status": "success", "finished_at": "2026-07-29T10:00:02Z", "duration_s": 2.0},
    ])

    run = build_dashboard_snapshot(sources)["runs"][0]
    assert run["status"] == "success"
    assert run["display_status"] == "completed_with_warnings"
    assert run["stages"][0]["status"] == "interrupted"


def test_reports_corrupt_tail_and_writes_atomically(tmp_path):
    sources = _sources(tmp_path)
    sources.run_events_file.parent.mkdir(parents=True)
    sources.run_events_file.write_text(
        '{"event_type":"run_started","run_id":"run-1","command":"encode"}\n{"event_type":',
        encoding="utf-8",
    )

    snapshot = build_dashboard_snapshot(sources)
    health = snapshot["source_health"][0]
    assert health["status"] == "partial"
    assert health["records"] == 1
    assert health["invalid_records"] == 1

    target = write_dashboard_snapshot(snapshot, sources.snapshot_file)
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not target.with_suffix(".json.tmp").exists()


def test_collects_rpg_artifacts_and_encoder_pipeline(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.rpg_file.parent.mkdir(parents=True)
    sources.rpg_file.write_text(json.dumps({
        "repo_name": "demo",
        "root": {
            "id": "repo",
            "node_type": "repo",
            "meta": {"type_name": "repo", "language": "python"},
            "children": [{
                "id": "feature",
                "node_type": "feature",
                "meta": {"type_name": "function", "language": "python"},
                "children": [],
            }],
        },
        "edges": [{"src": "repo", "dst": "feature", "relation": "invokes"}],
        "dep_graph": {
            "nodes": {
                "a.py:f": {"type": "function", "rpg_nodes": ["feature"]},
                "b.py:g": {"type": "function", "rpg_nodes": []},
            },
            "edges": [{"src": "a.py:f", "dst": "b.py:g", "attrs": {"type": "invokes"}}],
        },
        "_dep_to_rpg_map": {"a.py:f": ["feature"]},
    }), encoding="utf-8")
    sources.reports_dir.mkdir(parents=True)
    (sources.reports_dir / "rpg.html").write_text("<html></html>", encoding="utf-8")

    snapshot = build_dashboard_snapshot(sources)
    assert snapshot["workspace"]["mode"] == "encoder"
    assert snapshot["rpg"]["feature_graph"]["nodes"] == 2
    assert snapshot["rpg"]["dependency_graph"]["nodes"] == 2
    assert snapshot["rpg"]["mapping"] == {
        "mapped_dep_nodes": 1,
        "total_dep_nodes": 2,
        "unmapped_dep_nodes": 1,
        "coverage_percent": 50.0,
        "definition": "dep nodes with at least one RPG mapping / all dep graph nodes",
        "mapping_relations": 1,
    }
    assert [step["status"] for step in snapshot["pipeline"]] == [
        "completed", "completed", "completed", "completed",
    ]
    assert snapshot["current_state"]["pipeline_percent"] == 100.0
    assert snapshot["graph"]["feature_root"]["id"] == "repo"
    assert len(snapshot["graph"]["semantic_edges"]) == 1
    assert len(snapshot["graph"]["dependency_graph"]["nodes"]) == 2
    artifacts = {artifact["label"]: artifact for artifact in snapshot["artifacts"]}
    assert artifacts["rpg_json"]["status"] == "available"
    assert artifacts["rpg_html"]["status"] == "available"


def test_decoder_pipeline_and_code_gen_state(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    for filename in ("feature_spec.json", "feature_build.json", "feature_tree.json", "skeleton.json"):
        (sources.data_dir / filename).write_text("{}", encoding="utf-8")
    (sources.data_dir / "code_gen_state.jsonl").write_text(json.dumps({
        "total_tasks": 10,
        "completed_tasks": 6,
        "failed_tasks": 1,
        "skipped_task_ids": ["skip-1"],
        "current_batch_id": "batch-7",
        "initialized": True,
        "last_updated": "2026-07-29T10:00:00Z",
    }) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(sources)
    assert snapshot["workspace"]["mode"] == "decoder"
    assert [step["status"] for step in snapshot["pipeline"][:5]] == [
        "completed", "completed", "completed", "completed", "not_started",
    ]
    assert snapshot["tasks"]["completed"] == 6
    assert snapshot["tasks"]["pending"] == 2
    assert snapshot["tasks"]["completion_percent"] == 60.0
    assert snapshot["verification"]["next_actions"][0]["command"] == "/cmind.build_data_flow"
    task_check = next(
        check for check in snapshot["verification"]["checks"] if check["name"] == "code generation tasks"
    )
    assert task_check["status"] == "failed"


def test_verification_preserves_reported_checks_and_derives_retry(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-1", "command": "encode", "status": "running"},
        {"event_type": "stage_started", "run_id": "run-1", "command": "encode", "stage_id": "stage-1", "sequence": 1, "stage": "visualize", "status": "running"},
        {
            "event_type": "stage_finished", "run_id": "run-1", "command": "encode", "stage_id": "stage-1",
            "sequence": 1, "stage": "visualize", "status": "failed",
            "error": {"type": "OSError", "message": "cannot write html"},
            "metrics": {
                "verification": [{"name": "html", "passed": False, "detail": "output unavailable"}],
                "next_action": "Check report directory permissions",
            },
        },
        {"event_type": "run_finished", "run_id": "run-1", "command": "encode", "status": "success"},
    ])

    run = build_dashboard_snapshot(sources)["runs"][0]
    assert run["display_status"] == "completed_with_warnings"
    assert [check["name"] for check in run["verification"]] == [
        "run lifecycle", "visualize", "html",
    ]
    assert run["verification"][2]["status"] is False
    assert [action["quality"] for action in run["next_actions"]] == ["reported", "derived"]
    assert run["next_actions"][0]["detail"] == "Check report directory permissions"
    assert run["next_actions"][1]["command"] == "/cmind.encode"


def test_aggregates_exact_api_llm_call_event(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-1", "command": "plan", "status": "running"},
        {"event_type": "stage_started", "run_id": "run-1", "command": "plan", "stage_id": "stage-1", "sequence": 1, "stage": "build_skeleton", "status": "running"},
        {
            "event_type": "llm_call", "event_id": "evt-llm", "timestamp": "2026-07-29T10:00:01Z",
            "run_id": "run-1", "command": "plan", "stage_id": "stage-1", "stage": "build_skeleton",
            "provider": "openai", "model": "gpt-test", "purpose": "generate", "success": True,
            "duration_s": 1.2, "token_status": "measured",
            "tokens": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cache_read_input_tokens": 40},
        },
        {"event_type": "stage_finished", "run_id": "run-1", "command": "plan", "stage_id": "stage-1", "sequence": 1, "stage": "build_skeleton", "status": "success"},
        {"event_type": "run_finished", "run_id": "run-1", "command": "plan", "status": "success"},
    ])

    run = build_dashboard_snapshot(sources)["runs"][0]
    assert run["telemetry"]["llm"]["calls"] == 1
    assert run["telemetry"]["llm"]["tokens"] == {
        "input": 100, "output": 20, "total": 120, "cache_read": 40, "reasoning": 0,
    }
    assert run["stages"][0]["telemetry"]["llm"]["models"][0]["name"] == "gpt-test"


def test_aggregates_mcp_and_hook_telemetry_without_params(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.logs_dir.mkdir(parents=True)
    (sources.logs_dir / "mcp_calls.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-07-29T10:00:00Z", "tool": "search_rpg", "params": {"query": "secret"}, "duration_ms": 10}),
            json.dumps({"ts": "2026-07-29T10:00:01Z", "tool": "search_rpg", "params": {"query": "other"}, "duration_ms": 30}),
        ]) + "\n",
        encoding="utf-8",
    )
    (sources.logs_dir / "hook_calls.jsonl").write_text(json.dumps({
        "ts": "2026-07-29T10:01:00Z",
        "hook": "post-commit",
        "mode": "incremental",
        "added": 1,
        "modified": 2,
        "deleted": 0,
        "rpg_nodes": 90,
        "dep_nodes": 40,
        "dep_edges": 80,
        "duration_ms": 50,
    }) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(sources)
    assert snapshot["telemetry"]["mcp"]["calls"] == 2
    assert snapshot["telemetry"]["mcp"]["average_duration_ms"] == 20.0
    assert snapshot["telemetry"]["mcp"]["tools"] == [{
        "name": "search_rpg",
        "calls": 2,
        "total_duration_ms": 40,
        "average_duration_ms": 20.0,
    }]
    assert "params" not in json.dumps(snapshot["telemetry"])
    assert snapshot["telemetry"]["hooks"]["change_totals"] == {
        "added": 1, "modified": 2, "deleted": 0,
    }
    assert snapshot["telemetry"]["hooks"]["latest_graph"]["dep_edges"] == 80


def test_collects_git_changes_and_focused_impact(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    _git(sources.workspace_root, "init", "-q")
    _git(sources.workspace_root, "config", "user.email", "test@example.com")
    _git(sources.workspace_root, "config", "user.name", "Snapshot Test")
    source_file = sources.workspace_root / "src" / "service.py"
    source_file.parent.mkdir()
    source_file.write_text("def old():\n    return 1\n", encoding="utf-8")
    _git(sources.workspace_root, "add", ".")
    _git(sources.workspace_root, "commit", "-qm", "before")
    base_commit = _git(sources.workspace_root, "rev-parse", "HEAD")
    source_file.write_text("def new():\n    return 2\n", encoding="utf-8")
    _git(sources.workspace_root, "add", ".")
    _git(sources.workspace_root, "commit", "-qm", "after")
    target_commit = _git(sources.workspace_root, "rev-parse", "HEAD")

    sources.rpg_file.parent.mkdir(parents=True)
    sources.rpg_file.write_text(json.dumps({
        "repo_name": "demo",
        "root": {
            "id": "repo", "name": "demo", "node_type": "repo", "children": [{
                "id": "feature-1", "name": "Service", "node_type": "feature",
                "meta": {"path": "src/service.py"}, "children": [],
            }],
        },
        "dep_graph": {
            "nodes": {
                "src/service.py:new": {
                    "type": "function", "path": "src/service.py", "rpg_nodes": ["feature-1"],
                },
            },
            "edges": [],
        },
        "_dep_to_rpg_map": {"src/service.py:new": ["feature-1"]},
    }), encoding="utf-8")
    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-1", "command": "update_rpg", "status": "running"},
        {
            "event_type": "run_finished", "run_id": "run-1", "command": "update_rpg", "status": "success",
            "metrics": {
                "previous_commit": base_commit,
                "new_commit": target_commit,
                "nodes_delta": 1,
                "dep_nodes_delta": 1,
            },
        },
    ])

    changes = build_dashboard_snapshot(sources)["runs"][0]["changes"]
    assert changes["available"] is True
    assert changes["summary"] == {"modified": 1}
    assert changes["files"][0]["path"] == "src/service.py"
    assert changes["files"][0]["lines_added"] == 2
    assert changes["files"][0]["lines_deleted"] == 2
    assert changes["graph_deltas"] == {"nodes_delta": 1, "dep_nodes_delta": 1}
    impact = changes["focused_impact"]
    assert impact["mapped_files"] == ["src/service.py"]
    assert impact["dependency_nodes"][0]["dep_node_id"] == "src/service.py:new"
    assert impact["rpg_nodes"][0]["node_id"] == "feature-1"


def test_collects_rpg_edit_retrieval_decision_and_verification(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    (sources.data_dir / "rpg_edit_plan.json").write_text(json.dumps({
        "affected_nodes": ["feature-1"],
        "feature_changes": [{"node_id": "feature-1", "action": "update", "change": "new behavior"}],
        "code_changes": [{"file_path": "src/service.py", "change_type": "modify", "description": "implement"}],
    }), encoding="utf-8")
    (sources.data_dir / "rpg_edit_locate.json").write_text(json.dumps({
        "type": "located",
        "query": "change service",
        "results": [{"node_id": "feature-1", "name": "Service", "score": 0.91}],
    }), encoding="utf-8")
    (sources.data_dir / "rpg_edit_impact.json").write_text(json.dumps({
        "type": "impact_analysis",
        "results": {
            "feature-1": {
                "name": "Service",
                "dep_nodes": ["src/service.py:run"],
                "callers": [{"node_id": "src/main.py:main", "name": "main", "type": "function"}],
                "callees": [],
                "inheritance": [],
                "imports": [],
                "affected_files": ["src/main.py"],
                "impact_summary": {"total_callers": 1, "affected_file_count": 1},
            },
        },
    }), encoding="utf-8")
    (sources.data_dir / "rpg_edit_code_result.json").write_text(json.dumps({
        "success": True,
        "last_status": "complete",
        "commit_sha": "abc123",
        "files_modified": ["src/service.py"],
        "iterations": [{"iteration": 1}],
    }), encoding="utf-8")
    (sources.data_dir / "rpg_edit_apply_result.json").write_text(json.dumps({
        "type": "success",
        "confirmed": True,
        "before_state": {"head_commit": "before"},
        "applied_features": [{"node_id": "feature-1"}],
        "dep_graph_refreshed": True,
        "test_result": {"passed": True},
        "backups": {"rpg": "/tmp/rpg.backup"},
    }), encoding="utf-8")
    (sources.data_dir / "rpg_edit_review_result.json").write_text(json.dumps({
        "type": "passed",
        "success": True,
        "iterations": [{"iteration": 1}],
        "suggestions": ["keep tests"],
    }), encoding="utf-8")

    context = build_dashboard_snapshot(sources)["rpg_edit"]
    assert context["available"] is True
    assert context["scope"] == "current_workspace"
    assert context["plan"]["affected_nodes"] == ["feature-1"]
    assert [row["kind"] for row in context["retrievals"]] == ["locate", "impact"]
    assert context["retrievals"][1]["summary"]["mapped_code_relations"] == 1
    assert context["retrievals"][1]["hits"][0]["summary"]["callers"] == 1
    assert context["code"]["commit_sha"] == "abc123"
    assert context["decisions"][0]["test_status"] == "passed"
    assert context["decisions"][0]["rollback_path"] == "/tmp/rpg.backup"
    assert [check["name"] for check in context["verification"]] == [
        "code", "apply", "test", "review",
    ]