import json
import os
import sys

# Ensure scripts/ is importable when tests run from the project root.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from code_gen import batch_prompts


def _isolate_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_prompts, "FEATURE_SPEC_FILE", tmp_path / "missing_feature_spec.json")
    monkeypatch.setattr(batch_prompts, "REPO_RPG_FILE", tmp_path / "missing_rpg.json")


def test_resolve_codegen_backend_uses_repo_source_scan_without_metadata(monkeypatch, tmp_path):
    _isolate_metadata(monkeypatch, tmp_path)
    (tmp_path / "cmd" / "app").mkdir(parents=True)
    (tmp_path / "cmd" / "app" / "main.go").write_text("package main\n", encoding="utf-8")

    assert batch_prompts._resolve_codegen_backend(tmp_path).name == "go"


def test_resolve_codegen_backend_prefers_feature_spec_metadata(monkeypatch, tmp_path):
    feature_spec = tmp_path / "feature_spec.json"
    feature_spec.write_text(
        json.dumps({"meta": {"primary_language": "python", "target_languages": ["python"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_prompts, "FEATURE_SPEC_FILE", feature_spec)
    monkeypatch.setattr(batch_prompts, "REPO_RPG_FILE", tmp_path / "missing_rpg.json")
    (tmp_path / "cmd" / "app").mkdir(parents=True)
    (tmp_path / "cmd" / "app" / "main.go").write_text("package main\n", encoding="utf-8")

    assert batch_prompts._resolve_codegen_backend(tmp_path).name == "python"
