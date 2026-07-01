from __future__ import annotations

import importlib.util
import json
import sys
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


def test_review_publish_report_returns_report_path(tmp_path: Path, monkeypatch) -> None:
    review = _load_script("rpg_edit_review_test", _SCRIPTS / "rpg_edit" / "review.py")

    validate_path = tmp_path / "validate.json"
    locate_path = tmp_path / "locate.json"
    plan_path = tmp_path / "plan.json"
    impact_path = tmp_path / "impact.json"
    code_path = tmp_path / "code.json"
    review_path = tmp_path / "review.json"
    report_path = tmp_path / "report.html"
    rpg_path = tmp_path / "rpg.json"

    validate_path.write_text(json.dumps({"type": "ready"}), encoding="utf-8")
    locate_path.write_text(
        json.dumps({"type": "candidates", "query": "a.py", "results": [{"node_id": "n1", "name": "Node", "score": 1.0, "dep_nodes": ["a.py:f"]}]}),
        encoding="utf-8",
    )
    plan_path.write_text(json.dumps({"affected_nodes": ["n1"], "code_changes": [{"file_path": "a.py"}]}), encoding="utf-8")
    impact_path.write_text(
        json.dumps({
            "type": "impact",
            "results": {
                "n1": {
                    "name": "Node",
                    "dep_nodes": ["a.py:f"],
                    "affected_files": ["a.py"],
                    "impact_summary": {"total_callers": 1, "affected_file_count": 1},
                }
            },
        }),
        encoding="utf-8",
    )
    code_path.write_text(json.dumps({"success": True, "commit_sha": "abc123", "files_modified": ["a.py"], "last_status": "complete"}), encoding="utf-8")
    rpg_path.write_text(
        json.dumps({
            "repo_name": "test",
            "root": {"id": "n1", "name": "Node", "node_type": "feature", "meta": {"path": "a.py"}, "children": []},
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
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", review_path)
    monkeypatch.setattr(review, "REPO_RPG_FILE", rpg_path)
    monkeypatch.setattr(review, "REPORTS_DIR", tmp_path)

    def fake_file_diffs_between(repo_dir, from_commit=None, to_commit="HEAD", *, files=None, py_only=False):
        assert to_commit == "abc123"
        assert files == ["a.py"]
        assert py_only is False
        return [{"file": "a.py", "change_type": "modify", "diff": "+new <unsafe>"}]

    monkeypatch.setattr(review, "file_diffs_between", fake_file_diffs_between)

    def fake_write_command_report(run):
        data = run.to_dict()
        review_artifact = next(
            item for item in data["artifacts"] if item["label"] == "review_result"
        )
        assert review_artifact["status"] == "available"
        assert data["retrievals"][0]["tool"] == str(locate_path)
        assert data["retrievals"][0]["query"] == "a.py"
        assert "locate score=1.0" in data["retrievals"][0]["hits"][0]["reason"]
        assert "impact callers=1, affected_files=1" in data["retrievals"][0]["hits"][0]["reason"]
        assert data["retrievals"][1]["tool"] == str(impact_path)
        assert data["code_deltas"] == [{"file": "a.py", "change_type": "modify", "diff": "+new <unsafe>"}]
        assert data["focused_graph"]["status"] == "available"
        assert data["focused_graph"]["selected_rpg_nodes"] == ["n1"]
        assert data["focused_graph"]["selected_dep_nodes"] == ["a.py:f"]
        assert Path(data["focused_graph"]["path"]).exists()
        assert any(item["label"] == "focused_graph" for item in data["artifacts"])
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
    review_path = tmp_path / "review.json"
    report_path = tmp_path / "report.html"
    dep_id = "scripts/common/run_report.py:_render_artifacts"

    validate_path.write_text(json.dumps({"type": "ready"}), encoding="utf-8")
    locate_path.write_text(json.dumps({"type": "candidates", "results": [{"node_id": "other", "name": "Other"}]}), encoding="utf-8")
    plan_path.write_text(json.dumps({"affected_nodes": ["planned"], "code_changes": [{"file_path": "scripts/common/run_report.py"}]}), encoding="utf-8")
    impact_path.write_text(
        json.dumps({"type": "impact", "results": {"planned": {"name": "Planned Node", "dep_nodes": [dep_id], "affected_files": ["scripts/common/run_report.py"]}}}),
        encoding="utf-8",
    )
    code_path.write_text(json.dumps({"success": True, "files_modified": ["scripts/common/run_report.py"], "last_status": "complete"}), encoding="utf-8")

    monkeypatch.setattr(review, "RPG_EDIT_VALIDATE_FILE", validate_path)
    monkeypatch.setattr(review, "RPG_EDIT_LOCATE_FILE", locate_path)
    monkeypatch.setattr(review, "RPG_EDIT_IMPACT_FILE", impact_path)
    monkeypatch.setattr(review, "RPG_EDIT_CODE_RESULT_FILE", code_path)
    monkeypatch.setattr(review, "RPG_EDIT_REVIEW_RESULT_FILE", review_path)
    monkeypatch.setattr(review, "_focused_graph_artifact", lambda candidates, artifacts: {})

    def fake_write_command_report(run):
        data = run.to_dict()
        assert data["rpg_deltas"] == [{"node_id": "planned", "name": "Planned Node"}]
        assert data["dep_graph_deltas"] == [
            {"dep_node_id": dep_id, "path": "scripts/common/run_report.py", "source_feature": "planned"}
        ]
        assert data["retrievals"][0]["hits"][0]["node_id"] == "planned"
        assert "1 dep nodes" in data["retrievals"][0]["hits"][0]["reason"]
        assert "impact callers=0, affected_files=1" in data["retrievals"][0]["hits"][0]["reason"]
        assert data["retrievals"][1]["tool"] == str(impact_path)
        return report_path

    monkeypatch.setattr(review, "write_command_report", fake_write_command_report)

    result = review._publish_review_report({"type": "skipped", "success": True}, plan_path, impact_path)

    assert result["report_path"] == str(report_path)
