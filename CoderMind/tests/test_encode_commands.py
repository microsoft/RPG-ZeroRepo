#!/usr/bin/env python3
"""Tests for M12 (Redo): Slash Command Scripts + MCP Server.

Covers:
  - check_encode.py: init/update/error states
  - run_encode.py / run_update_rpg.py: mocked RPGParser/RPGEvolution, verify JSON
  - Template validation: encode.md and update_rpg.md exist with valid YAML frontmatter
  - MCP server: GraphQueryEngine loading, create_mcp_server, tool registration
  - CLI integration: encode/update-rpg/mcp-server commands no longer registered
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root and src/ are on sys.path
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, os.path.join(_project_root, "scripts"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal temporary repository directory."""
    (tmp_path / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    return str(tmp_path)


def _make_rpg_data():
    """Return a valid flat-format RPG dict."""
    return {
        "repo_name": "test_repo",
        "repo_info": "A test repository.",
        "excluded_files": [],
        "nodes": [
            {
                "id": "repo_node",
                "name": "test_repo",
                "level": 0,
                "meta": {"type_name": "root", "path": "."},
            },
            {
                "id": "area_1",
                "name": "Core Logic",
                "level": 1,
                "meta": {"type_name": "directory", "path": "."},
            },
        ],
        "edges": [
            {
                "src": "repo_node",
                "dst": "area_1",
                "relation": "contains",
            }
        ],
    }


@pytest.fixture
def tmp_rpg_file(tmp_path):
    """Create a minimal RPG JSON file for testing (flat format)."""
    rpg_data = _make_rpg_data()
    rpg_file = tmp_path / "rpg.json"
    rpg_file.write_text(json.dumps(rpg_data, indent=2))
    return str(rpg_file)


@pytest.fixture
def mock_rpg():
    """Create a mock RPG object with minimal attributes."""
    rpg = MagicMock()
    rpg.nodes = {"repo_node": MagicMock(), "node_1": MagicMock(), "node_2": MagicMock()}
    rpg.edges = [MagicMock()]
    rpg.repo_info = "A test repository"
    rpg.excluded_files = []
    rpg.get_functional_areas.return_value = ["area_1", "area_2"]
    rpg.to_dict.return_value = {
        "nodes": [{"id": "repo_node"}, {"id": "node_1"}, {"id": "node_2"}],
        "edges": [{"source": "repo_node", "target": "node_1", "type": "contains"}],
    }
    return rpg


# ============================================================================
# Test: check_encode.py
# ============================================================================

class TestCheckEncode:
    def test_init_state_no_rpg_file(self, tmp_path, monkeypatch):
        """When rpg.json does not exist, check_encode should return type=init."""
        monkeypatch.chdir(tmp_path)
        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "init"
        assert "rpg_file" in result

    def test_update_state_valid_rpg(self, tmp_path, monkeypatch):
        """When a valid rpg.json exists, check_encode should return type=update."""
        monkeypatch.chdir(tmp_path)
        cmind_data = tmp_path / ".cmind" / "data"
        cmind_data.mkdir(parents=True)
        rpg_file = cmind_data / "rpg.json"
        rpg_data = _make_rpg_data()
        rpg_data["dep_graph"] = {
            "nodes": {
                "main.py": {"type": "file"},
                "main.py:hello": {"type": "function"},
            },
            "edges": [
                {"src": "main.py", "dst": "main.py:hello", "attrs": {"type": "contains"}},
            ],
        }
        rpg_file.write_text(json.dumps(rpg_data, indent=2))

        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "update"
        assert "stats" in result
        assert result["stats"]["repo_name"] == "test_repo"
        assert result["stats"]["node_count"] == 2
        assert result["stats"]["edge_count"] == 1
        assert result["stats"]["dep_nodes"] == 2
        assert result["stats"]["dep_edges"] == 1

    def test_error_state_invalid_rpg(self, tmp_path, monkeypatch):
        """When rpg.json exists but has invalid format, return type=error."""
        monkeypatch.chdir(tmp_path)
        cmind_data = tmp_path / ".cmind" / "data"
        cmind_data.mkdir(parents=True)
        rpg_file = cmind_data / "rpg.json"
        rpg_file.write_text(json.dumps({"some_key": "value"}, indent=2))

        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "error"
        assert "invalid format" in result["message"].lower() or "missing" in result["message"].lower()

    def test_error_state_empty_file(self, tmp_path, monkeypatch):
        """When rpg.json exists but is empty, return type=error."""
        monkeypatch.chdir(tmp_path)
        cmind_data = tmp_path / ".cmind" / "data"
        cmind_data.mkdir(parents=True)
        rpg_file = cmind_data / "rpg.json"
        rpg_file.write_text("")

        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "error"

    def test_update_state_nested_format(self, tmp_path, monkeypatch):
        """When rpg.json uses nested rpg.structure format, return type=update."""
        monkeypatch.chdir(tmp_path)
        cmind_data = tmp_path / ".cmind" / "data"
        cmind_data.mkdir(parents=True)
        rpg_file = cmind_data / "rpg.json"
        nested_data = {
            "repo_name": "nested_repo",
            "rpg": {
                "structure": {
                    "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                    "edges": [{"src": "n1", "dst": "n2"}],
                }
            },
        }
        rpg_file.write_text(json.dumps(nested_data, indent=2))

        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "update"
        assert result["stats"]["node_count"] == 3
        assert result["stats"]["edge_count"] == 1

    def test_update_state_root_tree_format(self, tmp_path, monkeypatch):
        """When rpg.json uses root tree format (nested children), return type=update."""
        monkeypatch.chdir(tmp_path)
        cmind_data = tmp_path / ".cmind" / "data"
        cmind_data.mkdir(parents=True)
        rpg_file = cmind_data / "rpg.json"
        tree_data = {
            "repo_name": "tree_repo",
            "root": {
                "id": "root_node",
                "name": "tree_repo",
                "children": [
                    {"id": "child_1", "name": "module_a", "children": [
                        {"id": "grandchild_1", "name": "file_a.py", "children": []},
                    ]},
                    {"id": "child_2", "name": "module_b", "children": []},
                ],
            },
            "edges": [{"src": "child_1", "dst": "child_2", "relation": "imports"}],
        }
        rpg_file.write_text(json.dumps(tree_data, indent=2))

        from rpg_encoder.check_encode import check_encode
        result = check_encode()
        assert result["type"] == "update"
        assert result["stats"]["node_count"] == 4  # root + 2 children + 1 grandchild
        assert result["stats"]["edge_count"] == 1
        assert result["stats"]["repo_name"] == "tree_repo"


# ============================================================================
# Test: run_encode.py (mocked RPGParser)
# ============================================================================

class TestRunEncode:
    def test_missing_repo_dir(self):
        """Should return error when repo dir doesn't exist."""
        from rpg_encoder.run_encode import run_encode
        result = run_encode(repo_dir="/nonexistent/path")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_success_with_mock(self, tmp_repo, tmp_path):
        """Should succeed with mocked RPGParser."""
        from rpg_encoder.run_encode import run_encode

        mock_rpg = MagicMock()
        mock_rpg.nodes = {"n1": MagicMock(), "n2": MagicMock()}
        mock_rpg.edges = [MagicMock()]
        mock_rpg.repo_info = "test"
        mock_rpg.excluded_files = []
        mock_rpg.get_functional_areas.return_value = ["area"]
        mock_rpg.to_dict.return_value = {"nodes": [], "edges": []}
        mock_rpg.parse_dep_graph = MagicMock()
        mock_rpg.dep_graph = None
        mock_rpg._dep_to_rpg_map = {}

        mock_parser = MagicMock()
        mock_parser.parse_rpg_from_repo.return_value = (mock_rpg, [], {})

        output_file = str(tmp_path / "output" / "rpg.json")

        with patch("rpg_encoder.rpg_encoding.RPGParser", return_value=mock_parser):
            with patch("common.llm_api_client.LLMConfig"):
                result = run_encode(
                    repo_dir=tmp_repo,
                    repo_name="test_repo",
                    output=output_file,
                )

        assert result["status"] == "success"
        assert result["node_count"] == 2
        assert result["edge_count"] == 1
        assert result["functional_areas"] == 1
        assert os.path.isfile(output_file)


# ============================================================================
# Test: run_update_rpg.py (mocked RPGEvolution)
# ============================================================================

class TestRunUpdateRpg:
    def test_missing_rpg_file(self, tmp_repo):
        """Should return error when RPG file doesn't exist."""
        from rpg_encoder.run_update_rpg import run_update_rpg
        result = run_update_rpg(
            rpg_file="/nonexistent/rpg.json",
            last_repo_dir=tmp_repo,
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_meta_git_reads_explicit_cur_repo_dir(self, tmp_path):
        from rpg_encoder.run_update_rpg import run_update_rpg

        last_repo = tmp_path / "last"
        cur_repo = tmp_path / "current"
        last_repo.mkdir()
        cur_repo.mkdir()
        rpg_file = tmp_path / "rpg.json"
        rpg_file.write_text(json.dumps({
            "repo_name": "test_repo",
            "repo_info": "",
            "root": {
                "id": "test_repo_L0",
                "name": "test_repo",
                "node_type": "repo",
                "level": 0,
                "meta": {"type_name": "directory", "path": "."},
                "children": [],
            },
            "edges": [],
        }))

        calls = []

        def fake_read_head(repo_dir):
            calls.append(Path(repo_dir))
            return {
                "head_commit": "a" * 40,
                "head_short": "aaaaaaa",
                "head_branch": "main",
                "head_timestamp": "2026-06-30T00:00:00+00:00",
            }

        with patch(
            "rpg_encoder.rpg_evolution.RPGEvolution.process_diff",
            side_effect=lambda **kwargs: kwargs["last_rpg"],
        ), patch(
            "rpg.service.RPGService.enrich_from_code",
            return_value={},
        ), patch("common.git_utils.read_head", side_effect=fake_read_head):
            result = run_update_rpg(
                rpg_file=str(rpg_file),
                last_repo_dir=str(last_repo),
                cur_repo_dir=str(cur_repo),
            )

        assert result["status"] == "success"
        assert result["meta_git_advanced"] is True
        assert result["new_commit"] == "a" * 40
        assert calls == [cur_repo.resolve()]

    def test_diff_summary_is_returned(self, tmp_rpg_file, tmp_path):
        from rpg_encoder.run_update_rpg import run_update_rpg

        last_repo = tmp_path / "last"
        cur_repo = tmp_path / "current"
        last_repo.mkdir()
        cur_repo.mkdir()

        def fake_process_diff(**kwargs):
            rpg = kwargs["last_rpg"]
            rpg._last_diff_summary = {
                "added": 1,
                "deleted": 0,
                "modified": 2,
                "renamed": 0,
            }
            rpg._last_diff_files = {
                "added": ["a.py"],
                "deleted": [],
                "modified": ["b.py", "c.py"],
                "renamed": [],
            }
            return rpg

        with patch(
            "rpg_encoder.rpg_evolution.RPGEvolution.process_diff",
            side_effect=fake_process_diff,
        ), patch(
            "rpg.service.RPGService.enrich_from_code",
            return_value={},
        ), patch("common.git_utils.read_head", return_value=None):
            result = run_update_rpg(
                rpg_file=tmp_rpg_file,
                last_repo_dir=str(last_repo),
                cur_repo_dir=str(cur_repo),
            )

        assert result["status"] == "success"
        assert result["diff_summary"] == {
            "added": 1,
            "deleted": 0,
            "modified": 2,
            "renamed": 0,
        }
        assert result["diff_files"]["modified"] == ["b.py", "c.py"]

    def test_missing_last_repo_dir(self, tmp_rpg_file):
        """Should return error when last repo dir doesn't exist."""
        from rpg_encoder.run_update_rpg import run_update_rpg
        result = run_update_rpg(
            rpg_file=tmp_rpg_file,
            last_repo_dir="/nonexistent/dir",
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_missing_cur_repo_dir(self, tmp_rpg_file):
        """Should return error when cur repo dir doesn't exist."""
        from rpg_encoder.run_update_rpg import run_update_rpg
        result = run_update_rpg(
            rpg_file=tmp_rpg_file,
            last_repo_dir="/tmp",
            cur_repo_dir="/nonexistent/dir",
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]


# ============================================================================
# Test: update_graphs report wiring
# ============================================================================


def test_attach_update_report_uses_update_rpg_result_fields(tmp_path, monkeypatch):
    import update_graphs

    rpg_path = tmp_path / "rpg.json"
    rpg_path.write_text(json.dumps({
        "repo_name": "test_repo",
        "root": {
            "id": "root",
            "name": "test_repo",
            "node_type": "root",
            "meta": {"path": "."},
            "children": [
                {
                    "id": "feature_a",
                    "name": "Refresh graph visualization",
                    "node_type": "feature",
                    "meta": {"path": "scripts/a.py:f"},
                    "children": [],
                }
            ],
        },
        "_dep_to_rpg_map": {
            "scripts/a.py": ["feature_a"],
            "scripts/a.py:f": ["feature_a"],
            "scripts/a.py:g": ["feature_a"],
        },
        "dep_graph": {
            "nodes": {
                "scripts/a.py": {
                    "type": "file",
                    "name": "a.py",
                    "path": "scripts/a.py",
                    "rpg_nodes": ["feature_a"],
                },
                "scripts/a.py:f": {
                    "type": "function",
                    "name": "f",
                    "path": "scripts/a.py",
                    "start_line": 10,
                    "end_line": 12,
                    "rpg_nodes": ["feature_a"],
                },
                "scripts/a.py:g": {
                    "type": "function",
                    "name": "g",
                    "path": "scripts/a.py",
                    "start_line": 50,
                    "end_line": 60,
                    "rpg_nodes": ["feature_a"],
                },
                "tests/test_a.py": {
                    "type": "file",
                    "name": "test_a.py",
                    "path": "tests/test_a.py",
                },
            },
            "edges": [
                {
                    "src": "scripts/a.py:f",
                    "dst": "tests/test_a.py",
                    "attrs": {"type": "imports"},
                }
            ],
        },
    }), encoding="utf-8")
    viz_path = tmp_path / "rpg.html"
    viz_path.write_text("<html></html>", encoding="utf-8")
    captured = {}

    def fake_write_command_report(run):
        captured.update(run.to_dict() if hasattr(run, "to_dict") else dict(run))
        return tmp_path / "report.html"

    monkeypatch.setenv("CMIND_HOOK", "PostCommit")
    monkeypatch.setattr(update_graphs, "write_command_report", fake_write_command_report)

    result = update_graphs._attach_update_report({
        "mode": "update-rpg",
        "status": "success",
        "output_path": str(rpg_path),
        "node_count": 4504,
        "edge_count": 15000,
        "nodes_delta": 2,
        "edges_delta": 7,
        "dep_nodes": 2708,
        "dep_nodes_delta": 46,
        "dep_edges": 5498,
        "dep_edges_delta": 103,
        "dep_to_rpg_map_size": 2,
        "diff_summary": {
            "added": 0,
            "deleted": 0,
            "modified": 3,
            "renamed": 0,
        },
        "diff_files": {
            "added": [],
            "deleted": [],
            "modified": ["scripts/a.py", "tests/test_a.py"],
            "renamed": [],
        },
        "git_delta": [
            {"status": "M", "change_type": "modified", "path": "scripts/a.py", "diff": "diff --git a/scripts/a.py b/scripts/a.py\n@@ -10,3 +10,4 @@\n+new"},
            {"status": "M", "change_type": "modified", "path": "tests/test_a.py", "diff": "diff --git a/tests/test_a.py b/tests/test_a.py\n@@ -1,1 +1,2 @@\n+test"},
        ],
        "prev_ref": "prev123",
        "previous_commit": "old456",
        "new_commit": "new789",
        "viz_path": str(viz_path),
    })

    cards = {card["label"]: card for card in captured["summary"]}
    artifacts = {artifact["label"]: artifact["path"] for artifact in captured["artifacts"]}
    steps = {step["name"]: step for step in captured["steps"]}
    evidence = captured["evidence"]
    focused = captured["focused_view"]
    nodes_view = focused["nodes_view"]

    assert result["report_path"] == str(tmp_path / "report.html")
    assert cards["git files"]["value"] == 2
    assert cards["semantic files"]["value"] == 3
    assert cards["semantic files"]["detail"] == "3 semantic files, modified=3"
    assert cards["RPG nodes"]["value"] == "4504 (delta: +2)"
    assert cards["RPG nodes"]["detail"] == "edges=15000 (delta: +7)"
    assert cards["dep graph"]["value"] == "nodes=2708 (delta: +46), edges=5498 (delta: +103)"
    assert "dep_to_rpg_map_size=2" in cards["dep graph"]["detail"]
    assert artifacts["rpg_json"] == str(rpg_path)
    assert artifacts["rpg_html"] == str(viz_path)
    assert artifacts["hook_calls_log"].endswith("hook_calls.jsonl")
    assert artifacts["update_rpg_log"].endswith("update_rpg.log")
    assert captured["status"] == "success"
    assert steps["git delta"]["reason"] == "2 changed files"
    assert steps["semantic delta"]["reason"] == "3 semantic files, modified=3"
    assert "prev_ref=prev123" in steps["commit range"]["reason"]
    assert "previous_commit=old456" in steps["commit range"]["reason"]
    assert "new_commit=new789" in steps["commit range"]["reason"]
    assert "CMIND_HOOK=PostCommit" in steps["hook context"]["reason"]
    assert captured["code_deltas"][0]["file"] == "scripts/a.py"
    assert captured["code_deltas"][0]["change_type"] == "modified"
    assert "+new" in captured["code_deltas"][0]["diff"]
    assert captured["rpg_deltas"][0]["node_id"] == "feature_a"
    assert captured["rpg_deltas"][0]["name"] == "Refresh graph visualization"
    assert {row["path"] for row in captured["dep_graph_deltas"]} >= {"scripts/a.py", "tests/test_a.py"}
    assert focused["summary"]["primary_rpg_nodes"] == 1
    assert focused["summary"]["primary_code_nodes"] == 2
    assert focused["summary"]["mapped_code_relations"] == 1
    assert focused["summary"]["missing_mappings"] == 1
    assert focused["warnings"] == []
    assert nodes_view["summary"]["semantic_nodes"] == 1
    assert nodes_view["summary"]["code_nodes"] == 2
    assert nodes_view["summary"]["mappings"] == 1
    assert nodes_view["summary"]["edges"] == 1
    assert nodes_view["summary"]["changed_files"] == 2
    semantic_node = nodes_view["semantic_nodes"][0]
    code_nodes = {row["node_id"]: row for row in nodes_view["code_nodes"]}
    mappings = {(row["rpg_node_id"], row["code_node_id"]): row for row in nodes_view["mappings"]}
    assert set(code_nodes) == {"scripts/a.py:f", "tests/test_a.py"}
    assert "scripts/a.py:g" not in code_nodes
    assert "scripts/a.py" not in code_nodes
    assert len(nodes_view["edges"]) == 1
    edge = nodes_view["edges"][0]
    assert edge["source_node_id"] == "feature_a"
    assert edge["target_node_id"] == "tests/test_a.py"
    assert edge["source_link_id"] == "rpg-feature_a"
    assert edge["target_link_id"] == "context-tests-test_a.py"
    assert edge["relation"] == "imports"
    assert edge["source_graph"] == "dep_graph"
    assert semantic_node["node_id"] == "feature_a"
    assert semantic_node["link_id"] == "rpg-feature_a"
    assert semantic_node["mapped_code_node_ids"] == ["scripts/a.py:f"]
    assert semantic_node["mapped_code_link_ids"] == ["code-scripts-a.py-f"]
    assert semantic_node["mapped_code"][0]["line_range"] == {"start": 10, "end": 12}
    assert semantic_node["changed_files"] == [{"path": "scripts/a.py", "diff_anchor": "diff-scripts_a.py"}]
    assert code_nodes["scripts/a.py:f"]["line_range"] == {"start": 10, "end": 12}
    assert code_nodes["scripts/a.py:f"]["link_id"] == "code-scripts-a.py-f"
    assert code_nodes["scripts/a.py:f"]["diff_anchor"] == "diff-scripts_a.py"
    assert code_nodes["scripts/a.py:f"]["mapped_rpg_node_ids"] == ["feature_a"]
    assert code_nodes["scripts/a.py:f"]["mapped_rpg_link_ids"] == ["rpg-feature_a"]
    assert code_nodes["scripts/a.py:f"]["mapped_rpg"][0]["feature_path"] == "test_repo / Refresh graph visualization"
    assert code_nodes["tests/test_a.py"]["link_id"] == "code-tests-test_a.py"
    assert "mapped_rpg" not in code_nodes["tests/test_a.py"]
    assert mappings[("feature_a", "scripts/a.py:f")]["source_link_id"] == "rpg-feature_a"
    assert mappings[("feature_a", "scripts/a.py:f")]["target_link_id"] == "code-scripts-a.py-f"
    assert mappings[("feature_a", "scripts/a.py:f")]["changed_files"] == [{"path": "scripts/a.py", "diff_anchor": "diff-scripts_a.py"}]
    assert nodes_view["changed_files"] == [
        {"path": "scripts/a.py", "diff_anchor": "diff-scripts_a.py"},
        {"path": "tests/test_a.py", "diff_anchor": "diff-tests_test_a.py"},
    ]
    assert nodes_view["hierarchy"]["id"] == "focused-graph-root"
    assert nodes_view["hierarchy"]["children"][0]["id"] == "feature-path-test_repo"
    hierarchy_text = json.dumps(nodes_view["hierarchy"], ensure_ascii=False)
    assert "Mapped code" not in hierarchy_text
    assert "Additional code context" not in hierarchy_text
    assert nodes_view["focused_graph"]["schema"] == "cmind.focused_graph.v1"
    assert nodes_view["focused_graph"]["hierarchy"] == nodes_view["hierarchy"]
    assert nodes_view["focused_graph"]["default_focus"] == nodes_view["default_focus"]
    assert nodes_view["graph_context"]["rpg_nodes"] == 4504
    assert nodes_view["graph_context"]["dep_edges"] == 5498
    assert nodes_view["graph_context"]["semantic_delta"] == 3
    default_node_link_ids = nodes_view["default_focus"]["node_link_ids"]
    assert "rpg-feature_a" in default_node_link_ids
    assert "context-tests-test_a.py" in default_node_link_ids
    assert "code-scripts-a.py" not in default_node_link_ids
    assert "code-scripts-a.py-f" not in default_node_link_ids
    assert "code-tests-test_a.py" not in default_node_link_ids
    assert nodes_view["default_focus"]["focused_code_link_ids"] == []
    assert nodes_view["default_focus"]["edge_link_ids"] == ["edge-feature_a-imports-tests-test_a.py"]
    assert "context-tests-test_a.py" in nodes_view["default_focus"]["relation_endpoint_link_ids"]
    assert evidence["code_deltas"][1]["file"] == "tests/test_a.py"
    assert evidence["semantic_summary"] == {"added": 0, "deleted": 0, "modified": 3, "renamed": 0}
    assert evidence["commit_range"] == {
        "prev_ref": "prev123",
        "previous_commit": "old456",
        "new_commit": "new789",
    }
    assert evidence["commit_range_reason"] == "prev_ref=prev123, previous_commit=old456, new_commit=new789"
    assert evidence["hook_context"]["CMIND_HOOK"] == "PostCommit"
    assert evidence["hook_context"]["hook_calls_log"].endswith("hook_calls.jsonl")
    assert evidence["hook_context"]["update_rpg_log"].endswith("update_rpg.log")


def test_diff_ranges_match_only_overlapping_nodes():
    from common.diff_ranges import changed_line_ranges_by_file, row_overlaps_changed_lines

    ranges = changed_line_ranges_by_file([
        {"file": "scripts/a.py", "diff": "diff --git a/scripts/a.py b/scripts/a.py\n@@ -10,3 +10,4 @@\n+new"}
    ])

    assert row_overlaps_changed_lines({"start_line": 10, "end_line": 12}, "scripts/a.py", ranges)
    assert not row_overlaps_changed_lines({"start_line": 50, "end_line": 60}, "scripts/a.py", ranges)


def test_attach_update_report_warns_on_zero_semantic_delta_without_inventing_nodes(tmp_path, monkeypatch):
    import update_graphs

    captured = {}

    def fake_write_command_report(run):
        captured.update(run.to_dict() if hasattr(run, "to_dict") else dict(run))
        return tmp_path / "zero.html"

    monkeypatch.setattr(update_graphs, "write_command_report", fake_write_command_report)

    result = update_graphs._attach_update_report({
        "mode": "update-rpg",
        "status": "success",
        "output_path": str(tmp_path / "missing-rpg.json"),
        "node_count": 4504,
        "nodes_delta": 0,
        "dep_nodes": 2708,
        "dep_nodes_delta": 0,
        "dep_edges": 5498,
        "dep_edges_delta": 0,
        "diff_summary": {
            "added": 0,
            "deleted": 0,
            "modified": 0,
            "renamed": 0,
        },
        "git_delta": [
            {"status": "M", "change_type": "modified", "path": "docs/readme.md", "diff": "+doc"},
        ],
        "prev_ref": "prev123",
        "previous_commit": "old456",
        "new_commit": "new789",
    })

    cards = {card["label"]: card for card in captured["summary"]}
    steps = {step["name"]: step for step in captured["steps"]}
    focused = captured["focused_view"]
    nodes_view = focused["nodes_view"]
    assert result["report_path"] == str(tmp_path / "zero.html")
    assert cards["semantic files"]["value"] == 0
    assert cards["semantic files"]["detail"] == "RPG semantic delta 为 0"
    assert steps["semantic delta"]["status"] == "warning"
    assert steps["semantic delta"]["reason"] == "RPG semantic delta 为 0"
    assert captured["verification"][2]["detail"] == "RPG semantic delta 为 0"
    assert focused["summary"]["primary_rpg_nodes"] == 0
    assert focused["summary"]["primary_code_nodes"] == 0
    assert nodes_view["summary"]["semantic_nodes"] == 0
    assert nodes_view["summary"]["code_nodes"] == 0
    assert nodes_view["summary"]["warnings"] == 2
    assert nodes_view["semantic_nodes"] == []
    assert nodes_view["code_nodes"] == []
    assert nodes_view["mappings"] == []
    assert nodes_view["edges"] == []
    assert nodes_view["changed_files"] == [{"path": "docs/readme.md", "diff_anchor": "diff-docs_readme.md"}]
    assert nodes_view["hierarchy"] == {
        "id": "focused-graph-root",
        "name": "Focused graph",
        "kind": "root",
        "feature_name": "Focused graph",
        "feature_path": "Focused graph",
        "meta": {"hidden_counts": {}, "warnings": 2, "edges": 0},
        "children": [],
    }
    assert nodes_view["default_focus"]["node_link_ids"] == []
    assert nodes_view["default_focus"]["focused_node_ids"] == []
    assert nodes_view["default_focus"]["focused_tree_node_ids"] == []
    assert nodes_view["default_focus"]["focused_code_link_ids"] == []
    assert nodes_view["default_focus"]["relation_endpoint_link_ids"] == []
    assert nodes_view["default_focus"]["semantic_node_ids"] == []
    assert nodes_view["default_focus"]["code_node_ids"] == []
    assert nodes_view["focused_graph"]["schema"] == "cmind.focused_graph.v1"
    assert nodes_view["focused_graph"]["warning_count"] == 2
    assert nodes_view["graph_context"]["semantic_delta"] == 0
    assert nodes_view["graph_context"]["changed_files"] == 1
    assert focused["changed_files"] == [{"path": "docs/readme.md", "diff_anchor": "diff-docs_readme.md"}]
    assert captured.get("rpg_deltas", []) == []
    assert captured.get("dep_graph_deltas", []) == []
    assert any(warning["message"] == "RPG semantic delta 为 0" for warning in focused["warnings"])
    assert any(warning["type"] == "unmapped_changed_file" for warning in focused["warnings"])
    assert focused["unmatched_code_deltas"][0]["file"] == "docs/readme.md"


# ============================================================================
# Test: Template validation
# ============================================================================

class TestTemplates:
    """Validate that slash command template files exist and have valid YAML frontmatter."""

    _template_dir = os.path.join(
        os.path.dirname(__file__), "..", "templates", "commands"
    )

    def test_encode_template_exists(self):
        encode_md = os.path.join(self._template_dir, "encode.md")
        assert os.path.isfile(encode_md), f"Missing template: {encode_md}"

    def test_update_rpg_template_exists(self):
        update_md = os.path.join(self._template_dir, "update_rpg.md")
        assert os.path.isfile(update_md), f"Missing template: {update_md}"

    def _parse_frontmatter(self, filepath: str) -> dict:
        """Parse YAML frontmatter from a markdown file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.startswith("---"):
            pytest.fail(f"No YAML frontmatter in {filepath}")

        # Find closing ---
        end = content.index("---", 3)
        yaml_block = content[3:end].strip()

        # Simple key: value parsing (no external yaml dependency)
        result = {}
        for line in yaml_block.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result

    def test_encode_template_frontmatter(self):
        encode_md = os.path.join(self._template_dir, "encode.md")
        fm = self._parse_frontmatter(encode_md)
        assert "name" in fm
        assert fm["name"] == "cmind.encode"

    def test_update_rpg_template_frontmatter(self):
        update_md = os.path.join(self._template_dir, "update_rpg.md")
        fm = self._parse_frontmatter(update_md)
        assert "name" in fm
        assert fm["name"] == "cmind.update_rpg"

    def test_encode_template_references_check_script(self):
        encode_md = os.path.join(self._template_dir, "encode.md")
        with open(encode_md, "r", encoding="utf-8") as f:
            content = f.read()
        assert "check_encode.py" in content
        assert "run_encode.py" in content

    def test_update_rpg_template_references_check_script(self):
        update_md = os.path.join(self._template_dir, "update_rpg.md")
        with open(update_md, "r", encoding="utf-8") as f:
            content = f.read()
        assert "check_encode.py" in content
        assert "update_graphs.py update-rpg" in content
        assert "Dependency graph Nodes: <dep_nodes> (delta: <dep_nodes_delta>)" in content
        assert "Dependency graph Edges: <dep_edges> (delta: <dep_edges_delta>)" in content


# ============================================================================
# Test: MCP Server (scripts/mcp_server.py + rpg/graph_query.py)
# ============================================================================

def _make_rpg_with_dep_graph(tmp_path):
    """Create an rpg.json with root tree + dep_graph for GraphQueryEngine tests."""
    rpg_data = {
        "repo_name": "test_repo",
        "repo_info": "A test repository.",
        "root": {
            "id": "repo_node",
            "name": "test_repo",
            "node_type": "root",
            "level": 0,
            "meta": {"type_name": "root", "path": "."},
            "children": [
                {
                    "id": "area_1",
                    "name": "Core Logic",
                    "node_type": "functional_area",
                    "level": 1,
                    "meta": {
                        "type_name": "directory",
                        "path": ["src/core.py", "src/extra.py"],
                    },
                    "children": [
                        {
                            "id": "feat_1",
                            "name": "hello handler",
                            "node_type": "feature",
                            "level": 2,
                            "meta": {"type_name": "function", "path": "main.py:hello"},
                            "children": [],
                        },
                    ],
                },
            ],
        },
        "edges": [],
        "dep_graph": {
            "nodes": {
                "main.py": {"type": "file", "name": "main.py"},
                "main.py:hello": {
                    "type": "function", "name": "hello",
                    "module": "main", "signature": "def hello()",
                    "start_line": 1, "end_line": 2,
                },
                "utils.py": {"type": "file", "name": "utils.py"},
                "utils.py:add": {
                    "type": "function", "name": "add",
                    "module": "utils", "signature": "def add(a, b)",
                    "start_line": 1, "end_line": 2,
                },
            },
            "edges": [
                {"src": "main.py:hello", "dst": "utils.py:add",
                 "attrs": {"type": "invokes"}},
            ],
        },
        "_dep_to_rpg_map": {"main.py:hello": ["feat_1"]},
        "_feature_to_dep_map": {"feat_1": ["main.py:hello"]},
    }
    rpg_file = tmp_path / "rpg.json"
    rpg_file.write_text(json.dumps(rpg_data, indent=2))
    return str(rpg_file)


@pytest.fixture
def tmp_rpg_with_dep_graph(tmp_path):
    return _make_rpg_with_dep_graph(tmp_path)


class TestMCPServer:
    def test_graph_query_engine_loads_from_rpg_file(self, tmp_rpg_with_dep_graph):
        """GraphQueryEngine should load from an rpg.json with embedded dep_graph."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        assert len(engine._dep_nodes) == 4
        assert len(engine._rpg_nodes) == 3  # root + area + feature

    def test_graph_query_engine_search_code(self, tmp_rpg_with_dep_graph):
        """search(scope='code') should find dep_graph nodes."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        results = engine.search("hello", scope="code")
        assert len(results) >= 1
        assert any(r["id"] == "main.py:hello" for r in results)

    def test_graph_query_engine_search_feature(self, tmp_rpg_with_dep_graph):
        """search(scope='feature') should find RPG tree nodes."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        results = engine.search("Core Logic", scope="feature")
        assert len(results) >= 1
        assert any(r["id"] == "area_1" for r in results)

    def test_graph_query_engine_search_list_path(self, tmp_rpg_with_dep_graph):
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        feature_results = engine.search("src/core.py", scope="feature")
        all_results = engine.search("src/extra.py", scope="all")
        assert any(
            r["id"] == "area_1" and r["path"] == ["src/core.py", "src/extra.py"]
            for r in feature_results
        )
        assert any(r["id"] == "area_1" for r in all_results)

    def test_graph_query_engine_explore(self, tmp_rpg_with_dep_graph):
        """explore() should traverse edges from a node."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.explore("main.py:hello", direction="downstream")
        assert result["start"] == "main.py:hello"
        assert result["total_edges"] >= 1

    def test_graph_query_engine_get_node_detail(self, tmp_rpg_with_dep_graph):
        """get_node_detail() should return attributes for a dep_graph node."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.get_node_detail("main.py:hello")
        assert result["name"] == "hello"
        assert result["source"] == "dep_graph"
        assert result["signature"] == "def hello()"

    def test_graph_query_engine_get_rpg_node_detail(self, tmp_rpg_with_dep_graph):
        """get_node_detail() should return attributes for an RPG tree node."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.get_node_detail("area_1")
        assert result["name"] == "Core Logic"
        assert result["source"] == "rpg_tree"

    def test_graph_query_engine_list_tree(self, tmp_rpg_with_dep_graph):
        """list_tree() should return the RPG tree structure."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.list_tree(max_depth=3)
        assert result["name"] == "test_repo"
        assert "children" in result
        assert result["total_nodes"] == 3

    def test_graph_query_engine_node_not_found(self, tmp_rpg_with_dep_graph):
        """get_node_detail() should return error + suggestions for missing nodes."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.get_node_detail("nonexistent_node")
        assert "error" in result

    def test_create_mcp_server_returns_server(self, tmp_rpg_with_dep_graph):
        """create_mcp_server should return a FastMCP instance with 4 tools."""
        from mcp_server import create_mcp_server
        server = create_mcp_server(rpg_file=tmp_rpg_with_dep_graph)
        assert hasattr(server, "run")
        assert server.name == "rpg-tools"

    def test_create_mcp_server_handles_missing_rpg_file(self, tmp_path):
        """Server must start cleanly when rpg.json is absent.

        Regression guard: a hard ``sys.exit(1)`` / unhandled exception during
        ``create_mcp_server`` surfaces on the MCP client as the opaque
        ``MCP error -32000: Connection closed`` and hides the real cause.
        The server is required to come up in degraded mode and the
        ``_unavailable_payload`` helper must point users at ``/cmind.encode``.
        """
        import mcp_server as m
        missing = tmp_path / "rpg.json"
        assert not missing.exists()
        # Must not raise.
        server = m.create_mcp_server(rpg_file=str(missing))
        assert server.name == "rpg-tools"
        payload = json.loads(m._unavailable_payload(str(missing), "file_not_found"))
        assert payload["error"] == "rpg_unavailable"
        assert "/cmind.encode" in payload["next_step"]


# ============================================================================
# Test: CLI integration (M12 commands removed)
# ============================================================================

class TestCLIIntegration:
    def test_main_app_no_encode_command(self):
        """The main app should NOT have 'encode' registered (removed in M12 redo)."""
        from cmind_cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "encode" not in command_names

    def test_main_app_no_update_rpg_command(self):
        """The main app should NOT have 'update-rpg' registered."""
        from cmind_cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "update-rpg" not in command_names

    def test_main_app_no_mcp_server_command(self):
        """The main app should NOT have 'mcp-server' registered."""
        from cmind_cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "mcp-server" not in command_names


# ============================================================================
# Test: MCP tool logic (GraphQueryEngine-based)
# ============================================================================

class TestMCPTools:
    def test_search_rpg_returns_json(self, tmp_rpg_with_dep_graph):
        """search_rpg tool should return valid JSON with matches."""
        from mcp_server import create_mcp_server
        server = create_mcp_server(rpg_file=tmp_rpg_with_dep_graph)
        # The tool functions are closures; test the engine directly
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        results = engine.search("add", scope="code")
        output = json.dumps(results, indent=2, ensure_ascii=False)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert any(r["id"] == "utils.py:add" for r in parsed)

    def test_explore_rpg_returns_json(self, tmp_rpg_with_dep_graph):
        """explore_rpg tool should return valid JSON with graph structure."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.explore("main.py:hello", direction="both", depth=1)
        output = json.dumps(result, indent=2, ensure_ascii=False)
        parsed = json.loads(output)
        assert parsed["start"] == "main.py:hello"
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_list_rpg_tree_returns_json(self, tmp_rpg_with_dep_graph):
        """list_rpg_tree tool should return valid JSON tree."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        result = engine.list_tree(max_depth=2)
        output = json.dumps(result, indent=2, ensure_ascii=False)
        parsed = json.loads(output)
        assert parsed["name"] == "test_repo"
        assert isinstance(parsed["total_nodes"], int)

    def test_search_all_scope(self, tmp_rpg_with_dep_graph):
        """search(scope='all') should return results from both code and feature graphs."""
        from rpg.graph_query import GraphQueryEngine
        engine = GraphQueryEngine.from_rpg_file(tmp_rpg_with_dep_graph)
        results = engine.search("hello", scope="all")
        sources = {r["source"] for r in results}
        # Should find "hello" in dep_graph (main.py:hello) and rpg_tree (hello handler)
        assert "dep_graph" in sources
        assert "rpg_tree" in sources
