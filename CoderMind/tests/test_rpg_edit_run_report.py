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
                    "impact_summary": {"total_callers": 1, "affected_file_count": 1},
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
                    {"id": "n2", "name": "Missing Node", "node_type": "feature", "meta": {"path": "b.py"}, "children": []}
                ],
            },
            "edges": [],
            "dep_graph": {
                "nodes": {"a.py:f": {"name": "f", "type": "function", "module": "a.py", "rpg_nodes": ["n1"]}},
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
        return [{"file": "a.py", "change_type": "modify", "diff": "+new <unsafe>"}]

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
        assert data["code_deltas"] == [{"file": "a.py", "change_type": "modify", "diff": "+new <unsafe>"}]
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
        assert focused["primary_rpg_nodes"][0]["hidden_counts"]["callers"] == 1
        assert focused["primary_code_nodes"][0]["node_id"] == "a.py:f"
        assert focused["primary_code_nodes"][0]["path"] == "a.py"
        mapping_statuses = {row["status"] for row in focused["mappings"]}
        assert {"mapped", "missing"}.issubset(mapping_statuses)
        mapped = next(row for row in focused["mappings"] if row.get("code_node_id") == "a.py:f")
        assert mapped["rpg_node_id"] == "n1"
        assert mapped["changed_files"] == ["a.py"]
        assert any(row["type"] == "missing_mapping" and row["node_id"] == "n2" for row in focused["warnings"])
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path)

    assert result["report_path"] == str(report_path)
    persisted = json.loads(review_path.read_text(encoding="utf-8"))
    assert persisted["report_path"] == str(report_path)


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
        json.dumps({"type": "impact", "results": {"planned": {"name": "Planned Node", "dep_nodes": [dep_id], "affected_files": ["scripts/common/run_report.py"]}}}),
        encoding="utf-8",
    )
    code_path.write_text(json.dumps({"success": True, "files_modified": ["scripts/common/run_report.py"], "last_status": "complete"}), encoding="utf-8")
    apply_path.write_text(
        json.dumps({
            "type": "dep_refreshed",
            "backup_timestamp": "456",
            "backups": {"dep_graph": str(tmp_path / "dep_graph.before-edit-456.json")},
            "applied_features": [],
            "dep_graph_refreshed": True,
            "rollback_command": "cmind script rpg_edit/apply.py --rollback 456",
            "test_result": {"passed": False, "output": "failing test"},
        }),
        encoding="utf-8",
    )
    rpg_path.write_text(
        json.dumps({
            "repo_name": "test",
            "root": {"id": "planned", "name": "Planned Node", "node_type": "feature", "meta": {"path": "scripts/common/run_report.py"}, "children": []},
            "edges": [],
            "dep_graph": {
                "nodes": {dep_id: {"name": "_render_artifacts", "type": "function", "module": "scripts/common/run_report.py", "rpg_nodes": ["planned"]}},
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
        decision = data["user_decisions"][0]
        assert decision["decision"] == "apply"
        assert decision["branch"] == "fallback-branch"
        assert decision["before_state"]["head_commit"] == "current456"
        assert "confirmed" not in decision
        assert decision["apply_status"] == "dep_refreshed"
        assert decision["test_status"] == "failed"
        assert decision["rollback_path"].endswith("dep_graph.before-edit-456.json")
        assert any(item["label"] == "apply_result" for item in data["artifacts"])
        assert data["rpg_deltas"] == [{"node_id": "planned", "name": "Planned Node"}]
        assert data["dep_graph_deltas"] == [
            {"dep_node_id": dep_id, "path": "scripts/common/run_report.py", "source_feature": "planned"}
        ]
        assert data["retrievals"][0]["hits"][0]["node_id"] == "planned"
        assert data["retrievals"][0]["hits"][0]["locate_state"] == "missing"
        assert data["retrievals"][0]["hits"][0]["mapping_state"] == "mapped"
        assert "1 mapped code relations" in data["retrievals"][0]["hits"][0]["reason"]
        assert "impact callers=0, affected_files=1" in data["retrievals"][0]["hits"][0]["reason"]
        assert data["retrievals"][1]["tool"] == str(impact_path)
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
        assert focused["apply"]["status"] == "dep_refreshed"
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path)

    assert result["report_path"] == str(report_path)
