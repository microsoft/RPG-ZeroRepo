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
    collect_automation_activity,
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
    assert snapshot["history"]["summary"]["root_count"] == 1
    assert snapshot["history"]["roots"][0]["logical_key"] == "encoder-encode"
    assert snapshot["history"]["roots"][0]["children"][0]["logical_key"] == "encoder-encode-parse-rpg"


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
    health = {item["source"]: item for item in snapshot["source_health"]}
    assert health["rpg"]["expectation"] == "required"
    assert health["mcp_calls"]["expectation"] == "optional"
    assert health["activity"]["expectation"] == "optional"
    assert health["rpg_edit_plan"]["expectation"] == "not_expected"


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


def test_codegen_final_test_closes_stale_running_pipeline_stage(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    sources.logs_dir.mkdir(parents=True)
    (sources.data_dir / "code_gen_state.jsonl").write_text(json.dumps({
        "total_tasks": 3,
        "completed_tasks": 3,
        "failed_tasks": 0,
        "started_at": "2026-08-05T00:00:00Z",
        "last_updated": "2026-08-05T00:02:30Z",
    }) + "\n", encoding="utf-8")
    (sources.logs_dir / "codegen_final_test.json").write_text(
        '{"success": true, "passed": 12, "failed": 0, "errors": 0}\n',
        encoding="utf-8",
    )
    _write_events(sources.run_events_file, [
        {
            "event_type": "run_started", "run_id": "run-1",
            "command": "code_gen", "status": "running",
            "started_at": "2026-08-05T00:00:00Z",
        },
        {
            "event_type": "stage_started", "run_id": "run-1",
            "command": "code_gen", "stage_id": "stage-1",
            "stage": "code_gen", "status": "running",
            "started_at": "2026-08-05T00:00:01Z",
        },
    ])

    snapshot = build_dashboard_snapshot(sources)

    codegen_step = next(step for step in snapshot["pipeline"] if step["id"] == "code_gen")
    assert codegen_step["status"] == "completed"
    assert codegen_step["quality"] == "measured"
    assert codegen_step["error"] is None
    assert codegen_step["duration_s"] == 150.0
    final_check = next(
        check for check in snapshot["verification"]["checks"]
        if check["name"] == "code_gen final test"
    )
    assert final_check == {
        "name": "code_gen final test",
        "status": "completed",
        "detail": {"passed": 12, "failed": 0, "errors": 0},
        "source": "codegen_final_test.json",
        "quality": "reported",
    }
    artifacts = {artifact["label"]: artifact for artifact in snapshot["artifacts"]}
    assert artifacts["codegen_final_test"]["status"] == "available"


def test_codegen_duration_remains_unknown_without_state_timestamps(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    (sources.data_dir / "code_gen_state.jsonl").write_text(json.dumps({
        "total_tasks": 3,
        "completed_tasks": 1,
        "failed_tasks": 0,
    }) + "\n", encoding="utf-8")

    snapshot = build_dashboard_snapshot(sources)

    codegen_step = next(step for step in snapshot["pipeline"] if step["id"] == "code_gen")
    assert codegen_step["duration_s"] is None


def test_codegen_tasks_without_final_test_remain_running(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    (sources.data_dir / "code_gen_state.jsonl").write_text(json.dumps({
        "total_tasks": 3,
        "completed_tasks": 3,
        "failed_tasks": 0,
    }) + "\n", encoding="utf-8")
    _write_events(sources.run_events_file, [{
        "event_type": "stage_started", "run_id": "run-1",
        "command": "code_gen", "stage_id": "stage-1",
        "stage": "code_gen", "status": "running",
    }])

    snapshot = build_dashboard_snapshot(sources)

    codegen_step = next(step for step in snapshot["pipeline"] if step["id"] == "code_gen")
    assert codegen_step["status"] == "running"
    assert codegen_step["quality"] == "measured"


def test_workspace_name_prefers_logical_repository_name(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    monkeypatch.setenv("CMIND_REPO_NAME", "todo-list-app")

    snapshot = build_dashboard_snapshot(sources)

    assert snapshot["workspace"]["name"] == "todo-list-app"


def test_pipeline_uses_available_artifact_when_stage_is_stale_pending(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    for filename in (
        "feature_spec.json", "feature_build.json", "feature_tree.json",
        "skeleton.json", "data_flow.json", "base_classes.json", "interfaces.json",
    ):
        (sources.data_dir / filename).write_text("{}", encoding="utf-8")
    (sources.data_dir / "tasks.json").write_text('{"planned_tasks_dict":{"Core":[]}}', encoding="utf-8")
    trajectory_dir = sources.data_dir / "trajectory"
    trajectory_dir.mkdir()
    (trajectory_dir / "plan_tasks.json").write_text(json.dumps({
        "command": "plan_tasks",
        "status": "completed",
        "started_at": "2026-08-05T00:00:00Z",
        "finished_at": "2026-08-05T00:00:05Z",
        "steps": [{"step_id": 1, "name": "other_pending_step", "status": "pending"}],
    }), encoding="utf-8")

    snapshot = build_dashboard_snapshot(sources)

    tasks_step = next(step for step in snapshot["pipeline"] if step["id"] == "plan_tasks")
    assert tasks_step["status"] == "completed"
    assert tasks_step["quality"] == "inferred"
    assert snapshot["verification"]["next_actions"][0]["command"] == "/cmind.code_gen"


def test_pipeline_uses_parent_run_when_internal_stage_names_differ(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    sources.data_dir.mkdir(parents=True)
    (sources.data_dir / "skeleton.json").write_text("{}", encoding="utf-8")
    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-1", "command": "build_skeleton", "status": "running", "started_at": "2026-08-05T00:00:00Z"},
        {"event_type": "stage_started", "run_id": "run-1", "command": "build_skeleton", "stage_id": "stage-1", "stage": "build_rpg", "status": "running", "started_at": "2026-08-05T00:00:01Z"},
        {"event_type": "stage_finished", "run_id": "run-1", "command": "build_skeleton", "stage_id": "stage-1", "stage": "build_rpg", "status": "success", "finished_at": "2026-08-05T00:00:02Z"},
        {"event_type": "run_finished", "run_id": "run-1", "command": "build_skeleton", "status": "success", "finished_at": "2026-08-05T00:00:03Z", "duration_s": 3.0},
    ])

    snapshot = build_dashboard_snapshot(sources)

    skeleton_step = next(step for step in snapshot["pipeline"] if step["id"] == "build_skeleton")
    assert skeleton_step["status"] == "completed"
    assert skeleton_step["quality"] == "measured"
    assert skeleton_step["run_id"] == "run-1"
    assert skeleton_step["duration_s"] == 3.0


def test_collects_mcp_and_hook_automation_separately_from_pipeline() -> None:
    history = {
        "roots": [{
            "kind": "mcp.session", "name": "MCP session", "status": "success",
            "started_at": "2026-08-05T00:00:00Z", "finished_at": "2026-08-05T00:00:01Z",
            "children": [
                {"kind": "tool.mcp", "status": "success"},
                {"kind": "tool.mcp", "status": "degraded"},
            ],
        }, {
            "kind": "hook.workflow", "name": "post-commit", "status": "success",
            "started_at": "2026-08-05T00:00:02Z", "finished_at": "2026-08-05T00:00:04Z",
            "details": {"hook_type": "post-commit", "git_sha": "abc1234"},
            "children": [
                {"kind": "hook.operation", "name": "sync", "status": "success"},
                {"kind": "workflow", "logical_key": "encoder-update-rpg", "status": "success"},
            ],
        }]}

    automation = collect_automation_activity(history)

    assert automation["mcp"] == {
        "sessions": 1, "calls": 2, "succeeded": 1, "degraded": 1, "failed": 0,
    }
    assert automation["hooks"] == {
        "invocations": 1, "post_commit": 1, "post_merge": 0,
        "operations": 1, "updates": 1, "failed": 0,
        "attribution_mismatches": 0,
    }
    assert automation["latest"]["type"] == "hook"
    assert automation["latest"]["git_sha"] == "abc1234"


def test_detects_hook_update_target_mismatch() -> None:
    automation = collect_automation_activity({"roots": [{
        "kind": "hook.workflow", "name": "post-commit", "status": "success",
        "details": {"git_sha": "abc1234", "hook_type": "post-commit"},
        "children": [{
            "kind": "workflow", "logical_key": "encoder-update-rpg",
            "status": "success", "metrics": {"new_commit": "def5678full"},
        }],
    }]})

    assert automation["hooks"]["attribution_mismatches"] == 1


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


def test_latest_failed_run_reason_is_promoted_to_workspace_attention(tmp_path):
    sources = _sources(tmp_path)
    sources.workspace_root.mkdir(parents=True)
    _write_events(sources.run_events_file, [
        {
            "event_type": "run_started", "run_id": "run-1",
            "command": "design_base_classes", "status": "running",
        },
        {
            "event_type": "stage_started", "run_id": "run-1",
            "command": "design_base_classes", "stage_id": "stage-1",
            "sequence": 1, "stage": "design_base_classes", "status": "running",
        },
        {
            "event_type": "stage_finished", "run_id": "run-1",
            "command": "design_base_classes", "stage_id": "stage-1",
            "sequence": 1, "stage": "design_base_classes", "status": "failed",
            "error": {"type": "CoverageError", "message": "six types remain uncovered"},
        },
        {
            "event_type": "run_finished", "run_id": "run-1",
            "command": "design_base_classes", "status": "failed",
            "error": {"type": "CoverageError", "message": "six types remain uncovered"},
        },
    ])

    snapshot = build_dashboard_snapshot(sources)

    action = snapshot["verification"]["next_actions"][0]
    assert action["label"] == "Retry design_base_classes"
    assert action["command"] == "/cmind.design_base_classes"
    assert action["detail"] == "six types remain uncovered"


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


def test_collects_rpg_history_from_meta_git(tmp_path):
    sources = _sources(tmp_path)
    meta_root = tmp_path  # sources.data_dir.parent is the home-side meta-git root
    sources.data_dir.mkdir(parents=True)
    _git(meta_root, "init", "-q")
    _git(meta_root, "config", "user.email", "test@example.com")
    _git(meta_root, "config", "user.name", "Snapshot Test")

    sources.rpg_file.write_text(json.dumps({
        "root": {"id": "repo", "node_type": "repo", "children": [{"id": "a", "children": []}]},
    }), encoding="utf-8")
    _git(meta_root, "add", "data/rpg.json")
    _git(meta_root, "commit", "-qm", "[hook:post-commit @ abc1234] update-rpg --json")

    sources.rpg_file.write_text(json.dumps({
        "root": {"id": "repo", "node_type": "repo", "children": [
            {"id": "a", "children": []}, {"id": "b", "children": []},
        ]},
    }), encoding="utf-8")
    _git(meta_root, "add", "data/rpg.json")
    _git(meta_root, "commit", "-qm", "[hook:post-merge @ def5678] update-rpg --json")

    _write_events(sources.run_events_file, [
        {"event_type": "run_started", "run_id": "run-history", "command": "update_rpg", "status": "running"},
        {
            "event_type": "run_finished",
            "run_id": "run-history",
            "command": "update_rpg",
            "status": "success",
            "metrics": {
                "new_commit": "def5678abcdef0123456789",
                "node_count": 3,
                "edge_count": 2,
                "nodes_delta": 1,
                "edges_delta": 1,
            },
        },
    ])

    history = build_dashboard_snapshot(sources)["rpg_history"]
    assert len(history) == 2
    assert history[0]["operation"] == "update-rpg --json"
    assert history[0]["hook"] == "post-merge"
    assert history[0]["source_commit"] == "def5678"
    assert history[0]["node_count"] == 3
    assert history[0]["nodes_delta"] == 1
    assert history[0]["run_id"] == "run-history"
    assert history[1]["operation"] == "update-rpg --json"
    assert history[1]["source_commit"] == "abc1234"
    assert all(version["commit"] and version["short_commit"] for version in history)
    assert history[0]["previous_version_commit"] == history[1]["commit"]

    latest = build_dashboard_snapshot(sources)["rpg_latest_change"]
    assert latest["quality"] == "measured"
    assert latest["commit"] == history[0]["commit"]
    assert latest["parent_commit"] == history[1]["commit"]
    assert latest["feature_nodes"]["counts"] == {
        "added": 1,
        "removed": 0,
        "modified": 0,
        "total": 3,
    }
    assert latest["feature_nodes"]["added"][0]["node_id"] == "b"


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