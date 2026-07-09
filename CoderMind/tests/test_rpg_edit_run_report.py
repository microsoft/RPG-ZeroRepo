from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_and_locate_results_are_persisted(tmp_path: Path, monkeypatch) -> None:
    validate = _load_script("rpg_edit_validate_test", _SCRIPTS / "rpg_edit" / "validate.py")
    locate = _load_script("rpg_edit_locate_test", _SCRIPTS / "rpg_edit" / "locate.py")

    validate_path = tmp_path / "rpg_edit_validate.json"
    locate_path = tmp_path / "rpg_edit_locate.json"
    monkeypatch.setattr(validate, "RPG_EDIT_VALIDATE_FILE", validate_path)
    monkeypatch.setattr(locate, "RPG_EDIT_LOCATE_FILE", locate_path)

    validate._write_validate_result({"type": "ready", "nodes": 2})
    locate._write_locate_result({"type": "candidates", "results": [{"node_id": "n1"}]})

    assert json.loads(validate_path.read_text(encoding="utf-8"))["type"] == "ready"
    assert json.loads(locate_path.read_text(encoding="utf-8"))["results"][0]["node_id"] == "n1"


def test_code_result_is_persisted(tmp_path: Path, monkeypatch) -> None:
    code = _load_script("rpg_edit_code_test", _SCRIPTS / "rpg_edit" / "code.py")

    result_path = tmp_path / "rpg_edit_code_result.json"
    monkeypatch.setattr(code, "RPG_EDIT_CODE_RESULT_FILE", result_path)

    code._write_code_result({"success": True, "commit_sha": "abc123"})

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data == {"success": True, "commit_sha": "abc123"}


def test_apply_result_preserves_rpg_metadata_when_dep_refresh_reuses_backup_ts(tmp_path: Path, monkeypatch) -> None:
    apply = _load_script("rpg_edit_apply_persist_test", _SCRIPTS / "rpg_edit" / "apply.py")

    result_path = tmp_path / "rpg_edit_apply_result.json"
    monkeypatch.setattr(apply, "RPG_EDIT_APPLY_RESULT_FILE", result_path)
    previous = {
        "type": "rpg_updated",
        "backup_timestamp": "123",
        "backups": {"rpg": "rpg.before-edit-123.json"},
        "applied_features": [{"node_id": "n1", "action": "modified"}],
        "confirmed": True,
        "before_state": {"head_commit": "before123"},
        "rollback_command": "cmind script rpg_edit/apply.py --rollback 123 --rollback-branch rpg-edit/test",
        "rollback_path": "rpg.before-edit-123.json",
    }
    result_path.write_text(json.dumps(previous), encoding="utf-8")

    result = apply._record_apply_result(
        {"type": "dep_refreshed", "backup_timestamp": "123"},
        backup_timestamp="123",
        backups={},
        applied_features=[],
        dep_graph_refreshed=True,
        before_state={"head_commit": "current123"},
        confirmed=True,
    )

    assert result["type"] == "dep_refreshed"
    assert result["applied_features"] == [{"node_id": "n1", "action": "modified"}]
    assert result["backups"] == {"rpg": "rpg.before-edit-123.json"}
    assert result["confirmed"] is True
    assert result["before_state"] == {"head_commit": "before123"}
    assert result["rollback_command"] == "cmind script rpg_edit/apply.py --rollback 123 --rollback-branch rpg-edit/test"
    assert result["rollback_path"] == "rpg.before-edit-123.json"
    assert result["dep_graph_refreshed"] is True
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_code_result_clears_recovered_subagent_error(tmp_path: Path, monkeypatch) -> None:
    code = _load_script("rpg_edit_code_retry_test", _SCRIPTS / "rpg_edit" / "code.py")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"code_changes": [{"file_path": "a.py", "description": "update report"}]}),
        encoding="utf-8",
    )
    calls = iter([
        (None, "Sub-agent failed after 900.1s: timeout"),
        ("CODE_STATUS: COMPLETE", None),
    ])
    fake_run_batch = types.ModuleType("run_batch")
    fake_run_batch.dispatch_sub_agent = lambda *args, **kwargs: next(calls)
    monkeypatch.setitem(sys.modules, "run_batch", fake_run_batch)
    monkeypatch.setattr(code, "_format_rpg_target_nodes", lambda plan, rpg_path: "nodes")
    monkeypatch.setattr(code, "_format_impact_context", lambda plan: "impact")
    monkeypatch.setattr(code, "_commit_changes", lambda repo_path, summary, status: "abc123")

    result = code.apply_code_changes(plan_path, tmp_path / "rpg.json", tmp_path, max_iterations=2, timeout=1)

    assert result["success"] is True
    assert result["last_status"] == "complete"
    assert result["last_error"] is None
    assert result["commit_sha"] == "abc123"
    assert result["iterations"][0]["parsed_status"] == "llm_error"
    assert result["iterations"][1]["parsed_status"] == "complete"


def test_review_publish_report_returns_report_path(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_test", _SCRIPTS / "rpg_edit" / "review.py")

    validate_path = tmp_path / "validate.json"
    locate_path = tmp_path / "locate.json"
    plan_path = tmp_path / "plan.json"
    impact_path = tmp_path / "impact.json"
    code_path = tmp_path / "code.json"
    apply_path = tmp_path / "apply.json"
    review_path = tmp_path / "review.json"
    report_path = tmp_path / "report.html"
    rpg_path = tmp_path / "rpg.json"

    validate_path.write_text(json.dumps({"type": "ready"}), encoding="utf-8")
    locate_path.write_text(
        json.dumps({
            "type": "candidates",
            "query": "a.py",
            "results": [
                {"node_id": "n1", "name": "Node", "score": 1.0, "dep_nodes": ["a.py:f"]},
                {"node_id": "n2", "name": "Missing Node", "score": 0.5},
            ],
        }),
        encoding="utf-8",
    )
    plan_path.write_text(json.dumps({"affected_nodes": ["n1", "n2"], "code_changes": [{"file_path": "a.py"}]}), encoding="utf-8")
    impact_path.write_text(
        json.dumps({
            "type": "impact",
            "results": {
                "n1": {
                    "name": "Node",
                    "dep_nodes": ["a.py:f"],
                    "affected_files": ["a.py"],
                    "callers": [{"node_id": "tests/test_a.py:test_f", "name": "test_f", "path": "tests/test_a.py"}],
                    "callees": [{"node_id": "helpers.py:g", "name": "g", "path": "helpers.py"}],
                    "impact_summary": {"total_callers": 1, "total_callees": 1, "affected_file_count": 1},
                },
                "n2": {
                    "name": "Missing Node",
                    "affected_files": [],
                    "impact_summary": {"total_callers": 0, "affected_file_count": 0},
                }
            },
        }),
        encoding="utf-8",
    )
    code_path.write_text(json.dumps({"success": True, "commit_sha": "abc123", "files_modified": ["a.py"], "last_status": "complete"}), encoding="utf-8")
    backup_path = tmp_path / "rpg.before-edit-123.json"
    apply_path.write_text(
        json.dumps({
            "type": "success",
            "backup_timestamp": "123",
            "backups": {"rpg": str(backup_path)},
            "applied_features": [{"node_id": "n1", "action": "modified"}],
            "dep_graph_refreshed": True,
            "rollback_path": str(backup_path),
            "rollback_command": "cmind script rpg_edit/apply.py --rollback 123",
            "before_state": {
                "head_commit": "before123",
                "head_short": "before1",
                "head_branch": "rpg-edit/test",
                "head_timestamp": "2026-06-30T12:00:00+00:00",
            },
            "confirmed": True,
            "test_result": {"passed": True, "output": ""},
        }),
        encoding="utf-8",
    )
    rpg_path.write_text(
        json.dumps({
            "repo_name": "test",
            "root": {
                "id": "n1",
                "name": "Node",
                "node_type": "feature",
                "meta": {"path": "a.py"},
                "children": [
                    {"id": "n2", "name": "Missing Node", "node_type": "feature", "meta": {"path": "b.py"}, "children": []},
                    {"id": "n3", "name": "Context Node", "node_type": "feature", "meta": {"path": "context.py"}, "children": []},
                ],
            },
            "edges": [],
            "dep_graph": {
                "nodes": {"a.py:f": {"name": "f", "type": "function", "module": "a.py", "line_start": 10, "line_end": 12, "rpg_nodes": ["n1"]}},
                "edges": [],
            },
            "_dep_to_rpg_map": {"a.py:f": ["n1"]},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(review, "RPG_EDIT_VALIDATE_FILE", validate_path)
    monkeypatch.setattr(review, "RPG_EDIT_LOCATE_FILE", locate_path)
    monkeypatch.setattr(review, "RPG_EDIT_IMPACT_FILE", impact_path)
    monkeypatch.setattr(review, "RPG_EDIT_CODE_RESULT_FILE", code_path)
    monkeypatch.setattr(review, "RPG_EDIT_APPLY_RESULT_FILE", apply_path)
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", review_path)
    monkeypatch.setattr(review, "REPO_RPG_FILE", rpg_path)
    monkeypatch.setattr(review, "REPORTS_DIR", tmp_path)

    def fake_file_diffs_between(repo_dir, from_commit=None, to_commit="HEAD", *, files=None, py_only=False):
        assert to_commit == "abc123"
        assert files == ["a.py"]
        assert py_only is False
        return [{"file": "a.py", "change_type": "modify", "diff": "diff --git a/a.py b/a.py\n@@ -10,3 +10,4 @@\n+new <unsafe>"}]

    monkeypatch.setattr(review, "file_diffs_between", fake_file_diffs_between)
    monkeypatch.setattr(
        review,
        "read_head",
        lambda repo_dir: {
            "head_commit": "current123",
            "head_short": "current",
            "head_branch": "current-branch",
            "head_timestamp": "2026-06-30T12:30:00+00:00",
        },
    )

    def fake_write_command_report(run):
        data = run.to_dict()
        review_artifact = next(
            item for item in data["artifacts"] if item["label"] == "review_result"
        )
        apply_artifact = next(
            item for item in data["artifacts"] if item["label"] == "apply_result"
        )
        assert review_artifact["status"] == "available"
        assert apply_artifact["status"] == "available"
        summary_by_label = {row["label"]: row for row in data["summary"]}
        assert summary_by_label["Review"] == {"label": "Review", "value": "skipped", "detail": "passed"}
        assert "Review status" not in summary_by_label
        assert "Review result" not in summary_by_label
        steps_by_name = {row["name"]: row for row in data["steps"]}
        assert steps_by_name["apply/dep-refresh"]["status"] == "success"
        assert "1 applied features" in steps_by_name["apply/dep-refresh"]["reason"]
        assert "dep_graph_refreshed=True" in steps_by_name["apply/dep-refresh"]["reason"]
        verification_by_name = {row["name"]: row for row in data["verification"]}
        assert verification_by_name["apply"]["status"] == "success"
        assert verification_by_name["apply"]["detail"] == "1 applied features"
        assert verification_by_name["test"]["status"] == "passed"
        assert verification_by_name["dep_graph refresh"]["status"] is True
        decision = data["user_decisions"][0]
        assert decision["decision"] == "apply"
        assert decision["branch"] == "rpg-edit/test"
        assert decision["before_state"]["head_commit"] == "before123"
        assert decision["rollback_path"] == str(backup_path)
        assert decision["confirmed"] is True
        assert decision["apply_status"] == "success"
        assert decision["test_status"] == "passed"
        assert data["retrievals"][0]["tool"] == str(locate_path)
        assert data["retrievals"][0]["query"] == "a.py"
        assert "locate score=1.0" in data["retrievals"][0]["hits"][0]["reason"]
        assert "impact callers=1, affected_files=1" in data["retrievals"][0]["hits"][0]["reason"]
        assert data["retrievals"][1]["tool"] == str(impact_path)
        assert data["code_deltas"] == [{"file": "a.py", "change_type": "modify", "diff": "diff --git a/a.py b/a.py\n@@ -10,3 +10,4 @@\n+new <unsafe>"}]
        artifact_paths = {item["label"]: item["path"] for item in data["artifacts"]}
        assert artifact_paths["validate"] == str(validate_path)
        assert artifact_paths["locate"] == str(locate_path)
        assert artifact_paths["plan"] == str(plan_path)
        assert artifact_paths["impact"] == str(impact_path)
        assert artifact_paths["code_result"] == str(code_path)
        assert artifact_paths["apply_result"] == str(apply_path)
        assert artifact_paths["review_result"] == str(review_path)
        evidence = data["evidence"]
        evidence_text = json.dumps(evidence, ensure_ascii=False)
        assert "artifacts" not in evidence
        assert "review_result" not in evidence
        assert "focused_view" not in evidence
        assert "nodes_view" not in evidence_text
        assert "default_focus" not in evidence_text
        assert "hierarchy" not in evidence_text
        assert "focused_graph" not in evidence
        assert "focused_graph" not in evidence_text
        assert "focused_impact" not in evidence
        assert "+new <unsafe>" not in evidence_text
        evidence_paths = {item["label"]: item["path"] for item in evidence["artifact_paths"]}
        assert evidence_paths == artifact_paths
        audit = evidence["audit_summary"]
        assert audit["review"]["status"] == "skipped"
        assert audit["review"]["success"] is True
        assert audit["plan"]["affected_nodes"] == ["n1", "n2"]
        assert audit["plan"]["code_changes"] == [{"file_path": "a.py"}]
        assert audit["impact"]["result_count"] == 2
        assert audit["impact"]["mapped_code_relations"] == 1
        assert audit["code"]["commit_sha"] == "abc123"
        assert audit["code"]["files_modified"] == ["a.py"]
        assert audit["apply"]["status"] == "success"
        assert audit["apply"]["dep_graph_refreshed"] is True
        assert audit["apply"]["applied_features"] == [{"node_id": "n1", "action": "modified"}]
        assert audit["apply"]["rollback_path"] == str(backup_path)
        assert audit["apply"]["backup_timestamp"] == "123"
        assert audit["apply"]["backups"] == {"rpg": str(backup_path)}
        assert audit["apply"]["confirmed"] is True
        assert audit["apply"]["before_state"]["head_commit"] == "before123"
        assert audit["apply"]["rollback_command"] == "cmind script rpg_edit/apply.py --rollback 123"
        assert audit["apply"]["test_status"] == "passed"
        assert "focused_graph" not in data
        assert "focused_impact" not in data
        assert not any(item["label"] == "focused_graph" for item in data["artifacts"])
        assert not list(tmp_path.glob("rpg_edit_focused_graph_*.html"))
        focused = data["focused_view"]
        assert focused["summary"]["selected_feature_groups"] == 2
        assert focused["summary"]["mapped_code_relations"] == 1
        assert focused["summary"]["missing_mappings"] == 1
        assert focused["primary_rpg_nodes"][0]["node_id"] == "n1"
        assert focused["primary_rpg_nodes"][0]["mapping_status"] == "mapped"
        assert focused["primary_rpg_nodes"][0]["changed_files"] == ["a.py"]
        assert focused["primary_code_nodes"][0]["node_id"] == "a.py:f"
        assert focused["primary_code_nodes"][0]["path"] == "a.py"
        mapping_statuses = {row["status"] for row in focused["mappings"]}
        assert {"mapped", "missing"}.issubset(mapping_statuses)
        mapped = next(row for row in focused["mappings"] if row.get("code_node_id") == "a.py:f")
        assert mapped["rpg_node_id"] == "n1"
        assert mapped["changed_files"] == ["a.py"]
        assert any(row["type"] == "missing_mapping" and row["node_id"] == "n2" for row in focused["warnings"])
        nodes_view = focused["nodes_view"]
        assert nodes_view["summary"]["selected_feature_groups"] == 2
        assert nodes_view["summary"]["semantic_nodes"] == 2
        semantic_by_id = {row["node_id"]: row for row in nodes_view["semantic_nodes"]}
        assert semantic_by_id["n1"]["breadcrumb_path"] == "Node"
        assert semantic_by_id["n1"]["mapping_status"] == "mapped"
        assert semantic_by_id["n1"]["selected"] is True
        assert semantic_by_id["n1"]["changed_files"] == [{"path": "a.py", "diff_anchor": "diff-a.py"}]
        assert semantic_by_id["n2"]["breadcrumb_path"] == "Node / Missing Node"
        assert semantic_by_id["n2"]["state"] == "missing_mapping"
        assert semantic_by_id["n2"]["mapping_status"] == "missing"
        assert semantic_by_id["n2"]["selected"] is True
        assert "n3" not in semantic_by_id
        code = nodes_view["code_nodes"][0]
        assert code["dep_node_id"] == "a.py:f"
        assert code["path"] == "a.py"
        assert code["symbol"] == "f"
        assert code["type"] == "function"
        assert code["line_range"] == {"start": 10, "end": 12}
        assert code["diff_anchor"] == "diff-a.py"
        bridge_by_rpg = {row["rpg_node_id"]: row for row in nodes_view["mappings"]}
        assert bridge_by_rpg["n1"]["status"] == "mapped"
        assert bridge_by_rpg["n1"]["changed_files"] == [{"path": "a.py", "diff_anchor": "diff-a.py"}]
        assert bridge_by_rpg["n2"]["state"] == "missing_mapping"
        assert nodes_view["hidden_counts"] == {}
        assert nodes_view["summary"]["edges"] == 2
        assert focused["summary"]["edges"] == 2
        edges_by_relation = {row["relation"]: row for row in nodes_view["edges"]}
        assert edges_by_relation["caller"]["source_node_id"] == "tests/test_a.py:test_f"
        assert edges_by_relation["caller"]["target_node_id"] == "n1"
        assert edges_by_relation["caller"]["source_link_id"] == "context-tests-test_a.py-test_f"
        assert edges_by_relation["callee"]["source_node_id"] == "n1"
        assert edges_by_relation["callee"]["target_node_id"] == "helpers.py:g"
        assert edges_by_relation["callee"]["target_link_id"] == "context-helpers.py-g"
        assert any(row["type"] == "missing_mapping" and row["node_id"] == "n2" for row in nodes_view["warnings"])
        assert nodes_view["hierarchy"]["id"] == "focused-graph-root"
        hierarchy_text = json.dumps(nodes_view["hierarchy"], ensure_ascii=False)
        assert "rpg-n3" not in hierarchy_text
        assert "Node / Context Node" not in hierarchy_text
        assert '"feature_path": "context.py"' not in hierarchy_text
        assert "Mapped code" not in hierarchy_text
        assert "Additional code context" not in hierarchy_text
        default_node_link_ids = nodes_view["default_focus"]["node_link_ids"]
        assert default_node_link_ids == ["rpg-n1", "context-tests-test_a.py-test_f", "context-helpers.py-g"]
        assert "code-a.py-f" not in default_node_link_ids
        assert "rpg-n2" not in default_node_link_ids
        assert "rpg-n3" not in default_node_link_ids
        focused_tree_node_ids = nodes_view["default_focus"]["focused_tree_node_ids"]
        assert "rpg-n2" not in focused_tree_node_ids
        assert "rpg-n3" not in focused_tree_node_ids
        assert nodes_view["default_focus"]["edge_depth"] == 1
        assert nodes_view["default_focus"]["show_edges"] is True
        assert nodes_view["focused_graph"]["schema"] == "cmind.focused_graph.v1"
        assert nodes_view["focused_graph"]["hierarchy"]["id"] == "focused-graph-root"
        assert nodes_view["focused_graph"]["default_focus"] == nodes_view["default_focus"]
        assert nodes_view["caps"] == {"primary_rpg_nodes": 20, "primary_code_nodes": 50, "edges": 80}
        assert nodes_view["graph_context"]["current_graph_available"] is True
        assert nodes_view["graph_context"]["current_rpg_nodes"] == 3
        assert nodes_view["graph_context"]["current_dep_nodes"] == 1
        assert "+new <unsafe>" not in json.dumps(nodes_view, ensure_ascii=False)
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path, report_scope="final")

    assert result["report_path"] == str(report_path)
    persisted = json.loads(review_path.read_text(encoding="utf-8"))
    assert persisted["report_path"] == str(report_path)


def test_rollback_command_only_uses_rpg_edit_branch(tmp_path: Path, monkeypatch) -> None:
    apply_module = _load_script("rpg_edit_apply_test", _SCRIPTS / "rpg_edit" / "apply.py")

    result = apply_module._rollback_command(
        "123",
        {
            "head_commit": "before123",
            "head_short": "before1",
            "head_branch": "feature/my-branch",
            "head_timestamp": "2026-06-30T12:00:00+00:00",
        },
    )
    assert result == "cmind script rpg_edit/apply.py --rollback 123"

    result = apply_module._rollback_command(
        "123",
        {
            "head_commit": "before123",
            "head_short": "before1",
            "head_branch": "rpg-edit/test",
            "head_timestamp": "2026-06-30T12:00:00+00:00",
        },
    )
    assert result == "cmind script rpg_edit/apply.py --rollback 123 --rollback-branch rpg-edit/test"


def test_review_report_reconstructs_affected_node_evidence_from_impact(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_impact_test", _SCRIPTS / "rpg_edit" / "review.py")

    validate_path = tmp_path / "validate.json"
    locate_path = tmp_path / "locate.json"
    plan_path = tmp_path / "plan.json"
    impact_path = tmp_path / "impact.json"
    code_path = tmp_path / "code.json"
    apply_path = tmp_path / "apply.json"
    review_path = tmp_path / "review.json"
    report_path = tmp_path / "report.html"
    rpg_path = tmp_path / "rpg.json"
    dep_id = "scripts/common/run_report.py:_render_artifacts"

    validate_path.write_text(json.dumps({"type": "ready"}), encoding="utf-8")
    locate_path.write_text(json.dumps({"type": "candidates", "results": [{"node_id": "other", "name": "Other"}]}), encoding="utf-8")
    plan_path.write_text(json.dumps({"affected_nodes": ["planned"], "code_changes": [{"file_path": "scripts/common/run_report.py"}]}), encoding="utf-8")
    impact_path.write_text(
        json.dumps({
            "type": "impact",
            "results": {
                "planned": {
                    "name": "Planned Node",
                    "dep_nodes": [dep_id],
                    "affected_files": ["scripts/common/run_report.py"],
                    "callers": [{"node_id": "tests/test_report.py:test_render_artifacts", "name": "test_render_artifacts", "path": "tests/test_report.py"}],
                    "impact_summary": {"total_callers": 1},
                }
            },
        }),
        encoding="utf-8",
    )
    code_path.write_text(json.dumps({"success": True, "commit_sha": "def456", "files_modified": ["scripts/common/run_report.py"], "last_status": "complete"}), encoding="utf-8")
    dep_backup_path = tmp_path / "dep_graph.before-edit-456.json"
    rpg_backup_path = tmp_path / "rpg.before-edit-456.json"
    apply_path.write_text(
        json.dumps({
            "type": "dep_refreshed",
            "backup_timestamp": "456",
            "backups": {"rpg": str(rpg_backup_path), "dep_graph": str(dep_backup_path)},
            "applied_features": [{"node_id": "planned", "action": "modified"}],
            "dep_graph_refreshed": True,
            "rollback_command": "cmind script rpg_edit/apply.py --rollback 456 --rollback-branch rpg-edit/test",
            "before_state": {
                "head_commit": "before456",
                "head_short": "before",
                "head_branch": "rpg-edit/test",
                "head_timestamp": "2026-06-30T12:45:00+00:00",
            },
            "confirmed": True,
            "test_result": {"passed": False, "output": "failing test"},
        }),
        encoding="utf-8",
    )
    rpg_path.write_text(
        json.dumps({
            "repo_name": "test",
            "root": {
                "id": "planned",
                "name": "Planned Node",
                "node_type": "feature",
                "meta": {"path": "scripts/common/run_report.py"},
                "children": [
                    {"id": "background", "name": "Background Node", "node_type": "feature", "meta": {"path": "background.py"}, "children": []}
                ],
            },
            "edges": [],
            "dep_graph": {
                "nodes": {
                    dep_id: {"name": "_render_artifacts", "type": "function", "module": "scripts/common/run_report.py", "line_start": 537, "line_end": 564, "rpg_nodes": ["planned"]},
                    "tests/test_report.py:test_render_artifacts": {"name": "test_render_artifacts", "type": "function", "module": "tests/test_report.py", "line_start": 20, "line_end": 40},
                },
                "edges": [],
            },
            "_dep_to_rpg_map": {dep_id: ["planned"]},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(review, "RPG_EDIT_VALIDATE_FILE", validate_path)
    monkeypatch.setattr(review, "RPG_EDIT_LOCATE_FILE", locate_path)
    monkeypatch.setattr(review, "RPG_EDIT_IMPACT_FILE", impact_path)
    monkeypatch.setattr(review, "RPG_EDIT_CODE_RESULT_FILE", code_path)
    monkeypatch.setattr(review, "RPG_EDIT_APPLY_RESULT_FILE", apply_path)
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", review_path)
    monkeypatch.setattr(review, "REPO_RPG_FILE", rpg_path)

    def fake_file_diffs_between(repo_dir, from_commit=None, to_commit="HEAD", *, files=None, py_only=False):
        assert to_commit == "def456"
        assert files == ["scripts/common/run_report.py"]
        assert py_only is False
        return [{"file": "scripts/common/run_report.py", "change_type": "modify", "diff": "diff --git a/scripts/common/run_report.py b/scripts/common/run_report.py\n@@ -540,3 +540,4 @@\n+new"}]

    monkeypatch.setattr(review, "file_diffs_between", fake_file_diffs_between)
    monkeypatch.setattr(
        review,
        "read_head",
        lambda repo_dir: {
            "head_commit": "current456",
            "head_short": "current",
            "head_branch": "fallback-branch",
            "head_timestamp": "2026-06-30T13:00:00+00:00",
        },
    )

    def fake_write_command_report(run):
        data = run.to_dict()
        steps_by_name = {row["name"]: row for row in data["steps"]}
        assert steps_by_name["apply/dep-refresh"]["status"] == "dep_refreshed"
        assert "1 applied features" in steps_by_name["apply/dep-refresh"]["reason"]
        assert "dep_graph_refreshed=True" in steps_by_name["apply/dep-refresh"]["reason"]
        verification_by_name = {row["name"]: row for row in data["verification"]}
        assert verification_by_name["apply"]["status"] == "dep_refreshed"
        assert verification_by_name["apply"]["detail"] == "1 applied features"
        assert verification_by_name["test"]["status"] == "failed"
        assert verification_by_name["dep_graph refresh"]["status"] is True
        decision = data["user_decisions"][0]
        assert decision["decision"] == "apply"
        assert decision["branch"] == "rpg-edit/test"
        assert decision["before_state"]["head_commit"] == "before456"
        assert decision["confirmed"] is True
        assert decision["apply_status"] == "dep_refreshed"
        assert decision["test_status"] == "failed"
        assert decision["rollback_path"] == str(rpg_backup_path)
        assert any(item["label"] == "apply_result" for item in data["artifacts"])
        assert data["rpg_deltas"] == [{"node_id": "planned", "name": "Planned Node"}]
        assert data["dep_graph_deltas"] == [
            {"dep_node_id": dep_id, "path": "scripts/common/run_report.py", "source_feature": "planned"}
        ]
        assert data["retrievals"][0]["hits"][0]["node_id"] == "planned"
        assert data["retrievals"][0]["hits"][0]["locate_state"] == "missing"
        assert data["retrievals"][0]["hits"][0]["mapping_state"] == "mapped"
        assert "1 mapped code relations" in data["retrievals"][0]["hits"][0]["reason"]
        assert "impact callers=1, affected_files=1" in data["retrievals"][0]["hits"][0]["reason"]
        assert data["retrievals"][1]["tool"] == str(impact_path)
        artifact_paths = {item["label"]: item["path"] for item in data["artifacts"]}
        assert artifact_paths["validate"] == str(validate_path)
        assert artifact_paths["locate"] == str(locate_path)
        assert artifact_paths["plan"] == str(plan_path)
        assert artifact_paths["impact"] == str(impact_path)
        assert artifact_paths["code_result"] == str(code_path)
        assert artifact_paths["apply_result"] == str(apply_path)
        assert artifact_paths["review_result"] == str(review_path)
        evidence = data["evidence"]
        evidence_text = json.dumps(evidence, ensure_ascii=False)
        assert "artifacts" not in evidence
        assert "review_result" not in evidence
        assert "focused_view" not in evidence
        assert "nodes_view" not in evidence_text
        assert "default_focus" not in evidence_text
        assert "hierarchy" not in evidence_text
        assert "focused_graph" not in evidence
        assert "focused_graph" not in evidence_text
        assert "focused_impact" not in evidence
        assert "failing test" not in evidence_text
        evidence_paths = {item["label"]: item["path"] for item in evidence["artifact_paths"]}
        assert evidence_paths == artifact_paths
        audit = evidence["audit_summary"]
        assert audit["review"]["status"] == "skipped"
        assert audit["plan"]["affected_nodes"] == ["planned"]
        assert audit["plan"]["code_changes"] == [{"file_path": "scripts/common/run_report.py"}]
        assert audit["impact"]["result_count"] == 1
        assert audit["impact"]["mapped_code_relations"] == 1
        assert audit["code"]["files_modified"] == ["scripts/common/run_report.py"]
        assert audit["apply"]["status"] == "dep_refreshed"
        assert audit["apply"]["dep_graph_refreshed"] is True
        assert audit["apply"]["applied_features"] == [{"node_id": "planned", "action": "modified"}]
        assert audit["apply"]["test_status"] == "failed"
        assert audit["apply"]["rollback_path"] == str(rpg_backup_path)
        assert audit["apply"]["backup_timestamp"] == "456"
        assert audit["apply"]["backups"] == {"rpg": str(rpg_backup_path), "dep_graph": str(dep_backup_path)}
        assert audit["apply"]["confirmed"] is True
        assert audit["apply"]["before_state"]["head_commit"] == "before456"
        assert audit["apply"]["rollback_command"] == "cmind script rpg_edit/apply.py --rollback 456 --rollback-branch rpg-edit/test"
        assert "focused_graph" not in data
        assert not any(item["label"] == "focused_graph" for item in data["artifacts"])
        focused = data["focused_view"]
        assert focused["summary"]["selected_feature_groups"] == 1
        assert focused["summary"]["mapped_code_relations"] == 1
        assert focused["summary"]["missing_mappings"] == 0
        primary = focused["primary_rpg_nodes"][0]
        assert primary["node_id"] == "planned"
        assert primary["locate_status"] == "missing"
        assert primary["changed_files"] == ["scripts/common/run_report.py"]
        mapping = focused["mappings"][0]
        assert mapping["code_node_id"] == dep_id
        assert "impact" in mapping["source"]
        nodes_view = focused["nodes_view"]
        assert nodes_view["summary"]["selected_feature_groups"] == 1
        assert nodes_view["summary"]["semantic_nodes"] == 1
        semantic_by_id = {row["node_id"]: row for row in nodes_view["semantic_nodes"]}
        semantic = semantic_by_id["planned"]
        assert semantic["breadcrumb_path"] == "Planned Node"
        assert semantic["locate_status"] == "missing"
        assert semantic["mapping_status"] == "mapped"
        assert semantic["selected"] is True
        assert semantic["changed_files"] == [{"path": "scripts/common/run_report.py", "diff_anchor": "diff-scripts_common_run_report.py"}]
        assert "background" not in semantic_by_id
        code = nodes_view["code_nodes"][0]
        assert code["dep_node_id"] == dep_id
        assert code["path"] == "scripts/common/run_report.py"
        assert code["symbol"] == "_render_artifacts"
        assert code["line_range"] == {"start": 537, "end": 564}
        assert code["diff_anchor"] == "diff-scripts_common_run_report.py"
        bridge = nodes_view["mappings"][0]
        assert bridge["status"] == "mapped"
        assert bridge["changed_files"] == [{"path": "scripts/common/run_report.py", "diff_anchor": "diff-scripts_common_run_report.py"}]
        assert nodes_view["summary"]["edges"] == 1
        assert focused["summary"]["edges"] == 1
        edge = nodes_view["edges"][0]
        assert edge["relation"] == "caller"
        assert edge["source_node_id"] == "tests/test_report.py:test_render_artifacts"
        assert edge["target_node_id"] == "planned"
        assert edge["source_link_id"] == "context-tests-test_report.py-test_render_artifacts"
        assert edge["target_link_id"] == "rpg-planned"
        assert any(row["type"] == "missing_reason" and row["node_id"] == "planned" for row in nodes_view["warnings"])
        assert nodes_view["hierarchy"]["id"] == "focused-graph-root"
        hierarchy_text = json.dumps(nodes_view["hierarchy"], ensure_ascii=False)
        assert "rpg-background" not in hierarchy_text
        assert '"feature_path": "Planned Node"' in hierarchy_text
        assert '"feature_path": "scripts/common/run_report.py"' not in hierarchy_text
        assert "Mapped code" not in hierarchy_text
        assert "Additional code context" not in hierarchy_text
        default_node_link_ids = nodes_view["default_focus"]["node_link_ids"]
        assert default_node_link_ids == ["rpg-planned", "context-tests-test_report.py-test_render_artifacts"]
        assert "code-scripts-common-run_report.py-_render_artifacts" not in default_node_link_ids
        assert "rpg-background" not in default_node_link_ids
        assert "rpg-background" not in nodes_view["default_focus"]["focused_tree_node_ids"]
        assert nodes_view["default_focus"]["edge_depth"] == 1
        assert nodes_view["focused_graph"]["schema"] == "cmind.focused_graph.v1"
        assert nodes_view["focused_graph"]["default_focus"] == nodes_view["default_focus"]
        assert nodes_view["caps"] == {"primary_rpg_nodes": 20, "primary_code_nodes": 50, "edges": 80}
        assert nodes_view["graph_context"]["current_graph_available"] is True
        assert nodes_view["graph_context"]["current_rpg_nodes"] == 2
        assert nodes_view["graph_context"]["current_dep_nodes"] == 2
        assert focused["apply"]["status"] == "dep_refreshed"
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path, report_scope="final")

    assert result["report_path"] == str(report_path)


def _patch_minimal_review_report(review, tmp_path: Path, monkeypatch) -> tuple[Path, None, Path]:
    plan_path = tmp_path / "plan.json"
    review_path = tmp_path / "review.json"
    plan_path.write_text(json.dumps({"affected_nodes": [], "code_changes": []}), encoding="utf-8")

    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", review_path)
    monkeypatch.setattr(review, "RPG_EDIT_VALIDATE_FILE", tmp_path / "validate.json")
    monkeypatch.setattr(review, "RPG_EDIT_LOCATE_FILE", tmp_path / "locate.json")
    monkeypatch.setattr(review, "RPG_EDIT_CODE_RESULT_FILE", tmp_path / "code.json")
    monkeypatch.setattr(review, "RPG_EDIT_APPLY_RESULT_FILE", tmp_path / "apply.json")
    monkeypatch.setattr(review, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(review, "_load_review_artifacts", lambda plan, impact: {"plan": {"affected_nodes": [], "code_changes": []}})
    monkeypatch.setattr(review, "_selected_candidate_rows", lambda artifacts: [])
    monkeypatch.setattr(review, "_code_delta_rows", lambda artifacts: [])
    monkeypatch.setattr(review, "_feature_evidence_groups", lambda artifacts, candidates, code_deltas, result: {})
    monkeypatch.setattr(review, "_review_summary_cards", lambda result, artifacts, focused_view: [])
    monkeypatch.setattr(review, "_review_timeline", lambda result, artifacts: [])
    monkeypatch.setattr(review, "_retrieval_rows", lambda artifacts, candidates: [])
    monkeypatch.setattr(review, "_review_verification", lambda result, artifacts: [])
    monkeypatch.setattr(review, "_user_decision", lambda result, artifacts: review.UserDecisionEvent(decision="apply"))
    return plan_path, None, review_path


def test_review_report_scope_none_persists_without_html(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_scope_none_test", _SCRIPTS / "rpg_edit" / "review.py")
    plan_path, impact_path, review_path = _patch_minimal_review_report(review, tmp_path, monkeypatch)

    def fake_write_command_report(run):
        raise AssertionError("write_command_report should not be called for report_scope=none")

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report(
        {"type": "skipped", "success": True},
        plan_path,
        impact_path,
        report_scope="none",
        parent_run_id="parent-1",
    )

    assert "report_path" not in result
    assert result["published_to"] is None
    assert result["report_scope"] == "none"
    assert result["is_final"] is False
    assert result["parent_run_id"] == "parent-1"
    persisted = json.loads(review_path.read_text(encoding="utf-8"))
    assert persisted["report_scope"] == "none"
    assert persisted["published_to"] is None
    assert "report_path" not in persisted


def test_review_report_scope_internal_writes_under_internal_report_dir(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_scope_internal_test", _SCRIPTS / "rpg_edit" / "review.py")
    plan_path, impact_path, review_path = _patch_minimal_review_report(review, tmp_path, monkeypatch)
    captured: dict[str, dict] = {}
    report_timestamp = "20260707T123456Z"
    monkeypatch.setattr(review, "_report_timestamp", lambda: report_timestamp)

    def fake_write_command_report(run):
        data = run.to_dict()
        captured["data"] = data
        target_dir = Path(data["report_dir"])
        assert target_dir == tmp_path / "reports" / "internal"
        report_path = target_dir / "internal-report.html"
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html></html>", encoding="utf-8")
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report(
        {"type": "skipped", "success": True},
        plan_path,
        impact_path,
        report_scope="internal",
        parent_run_id="parent-2",
    )

    assert result["report_path"] == str(tmp_path / "reports" / "internal" / "internal-report.html")
    assert result["internal_report_paths"] == [result["report_path"]]
    assert result["report_scope"] == "internal"
    assert result["is_final"] is False
    data = captured["data"]
    assert data["timestamp"] == report_timestamp
    evidence = data["evidence"]
    assert evidence["report_scope"] == "internal"
    assert evidence["is_final"] is False
    assert evidence["parent_run_id"] == "parent-2"
    expected_published_to = tmp_path / "reports" / "internal" / f"cmind_run_rpg_edit_{report_timestamp}.html"
    assert evidence["published_to"] == str(expected_published_to)
    assert result["run_id"] not in evidence["published_to"]
    persisted = json.loads(review_path.read_text(encoding="utf-8"))
    assert persisted["internal_report_paths"] == [result["report_path"]]


def test_review_final_report_preserves_internal_report_artifacts(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_scope_final_artifacts_test", _SCRIPTS / "rpg_edit" / "review.py")
    plan_path, impact_path, _review_path = _patch_minimal_review_report(review, tmp_path, monkeypatch)
    writes: list[dict] = []

    def fake_write_command_report(run):
        data = run.to_dict()
        writes.append(data)
        target_dir = Path(data["report_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / f"{data['evidence']['report_scope']}-{len(writes)}.html"
        report_path.write_text("<html></html>", encoding="utf-8")
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    first = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path, report_scope="internal")
    second = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path, report_scope="internal")
    final = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path, report_scope="final")

    internal_paths = [first["report_path"], second["report_path"]]
    assert final["internal_report_paths"] == internal_paths
    final_data = writes[-1]
    artifact_paths = {item["label"]: item["path"] for item in final_data["artifacts"]}
    assert artifact_paths["internal_report_1"] == internal_paths[0]
    assert artifact_paths["internal_report_2"] == internal_paths[1]
    evidence_paths = {item["label"]: item["path"] for item in final_data["evidence"]["artifact_paths"]}
    assert evidence_paths["internal_report_1"] == internal_paths[0]
    assert evidence_paths["internal_report_2"] == internal_paths[1]
    assert final_data["report_dir"] == str(tmp_path / "reports")
    assert final_data["evidence"]["report_scope"] == "final"
    assert final_data["evidence"]["is_final"] is True
