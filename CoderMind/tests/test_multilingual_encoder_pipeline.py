#!/usr/bin/env python3
"""Tests for multilingual encoder discovery and semantic parsing entry."""

import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from rpg import NodeType
from rpg.code_unit import ParsedFile
from rpg_encoder.refactor_tree import RefactorTree
from rpg_encoder.rpg_encoding import RPGParser
from rpg_encoder.semantic_parsing import ParseFeatures


GO_SOURCE = textwrap.dedent(
    """\
    package main

    type Server struct {}

    func (s *Server) Handle() {}

    func NewServer() *Server {
        return &Server{}
    }
    """
)


TS_SOURCE = textwrap.dedent(
    """\
    import { User } from "./model";

    export class Greeter {
      greet(user: User): string {
        return user.name;
      }
    }

    export function makeGreeter(): Greeter {
      return new Greeter();
    }
    """
)


def _make_parse_features(tmp_path, valid_files, responses):
    mock_llm = MagicMock()
    mock_llm.generate_with_memory.side_effect = responses
    parser = ParseFeatures(
        repo_dir=str(tmp_path),
        repo_info="test repo",
        repo_skeleton="\n".join(valid_files),
        valid_files=valid_files,
        repo_name="test-repo",
        llm_client=mock_llm,
    )
    return parser, mock_llm


def test_rpg_parser_skeleton_includes_supported_languages_and_excludes_tests(tmp_path):
    for rel_path, content in {
        "pkg/mod.py": "def helper():\n    return 1\n",
        "main.go": "package main\nfunc Run() {}\n",
        "src/app.ts": "export function run(): number { return 1; }\n",
        "src/component.tsx": "export function View() { return <div />; }\n",
        "web/app.js": "export function run() { return 1; }\n",
        "web/view.jsx": "export function View() { return <div />; }\n",
        "main_test.go": "package main\nfunc TestRun() {}\n",
        "src/app.test.ts": "export function testRun() {}\n",
        "web/app.spec.js": "export function specRun() {}\n",
        "README.md": "# docs\n",
    }.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    with patch.object(RPGParser, "__init__", lambda self, **kw: None):
        parser = RPGParser.__new__(RPGParser)
        parser.repo_dir = str(tmp_path)
        parser.repo_name = "test"
        parser.logger = MagicMock()
        skeleton, valid_files = parser._load_skeleton_from_repo()

    assert "pkg/mod.py" in valid_files
    assert "main.go" in valid_files
    assert "src/app.ts" in valid_files
    assert "src/component.tsx" in valid_files
    assert "web/app.js" in valid_files
    assert "web/view.jsx" in valid_files
    assert "main_test.go" not in valid_files
    assert "src/app.test.ts" not in valid_files
    assert "web/app.spec.js" not in valid_files
    assert "README.md" not in valid_files
    assert "README.md" in skeleton


def test_go_repo_enters_semantic_parsing_with_non_empty_units(tmp_path):
    (tmp_path / "main.go").write_text(GO_SOURCE)
    abs_path = str(tmp_path / "main.go")
    parsed = ParsedFile(GO_SOURCE, abs_path)
    assert parsed.units

    responses = [
        '<solution>{"Server": {"Handle": ["serve request"]}}</solution>',
        '<solution>{"NewServer": ["create server"]}</solution>',
        f'<solution>{{"{abs_path}": "server runtime"}}</solution>',
    ]
    parser, mock_llm = _make_parse_features(tmp_path, ["main.go", "main_test.go", "README.md"], responses)

    features, trajectories = parser.parse_repo(max_workers=1, max_iterations=1)

    assert "main.go" in features
    assert features["main.go"]["class Server"] == {"Handle": ["serve request"]}
    assert features["main.go"]["function NewServer"] == ["create server"]
    assert features["main.go"]["_file_summary_"] == "server runtime"
    assert trajectories
    assert mock_llm.generate_with_memory.call_count == 3


def test_typescript_repo_enters_semantic_parsing_with_non_empty_units(tmp_path):
    source_path = tmp_path / "src" / "greeter.ts"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(TS_SOURCE)
    (tmp_path / "src" / "greeter.test.ts").write_text("export function testGreeter() {}\n")
    abs_path = str(source_path)
    parsed = ParsedFile(TS_SOURCE, abs_path)
    assert parsed.units

    responses = [
        '<solution>{"Greeter": {"greet": ["format greeting"]}}</solution>',
        '<solution>{"makeGreeter": ["create greeter"]}</solution>',
        f'<solution>{{"{abs_path}": "greeting utilities"}}</solution>',
    ]
    parser, mock_llm = _make_parse_features(
        tmp_path,
        ["src/greeter.ts", "src/greeter.test.ts", "notes.txt"],
        responses,
    )

    features, _ = parser.parse_repo(max_workers=1, max_iterations=1)

    assert "src/greeter.ts" in features
    assert "src/greeter.test.ts" not in features
    assert features["src/greeter.ts"]["class Greeter"] == {"greet": ["format greeting"]}
    assert features["src/greeter.ts"]["function makeGreeter"] == ["create greeter"]
    assert features["src/greeter.ts"]["_file_summary_"] == "greeting utilities"
    assert mock_llm.generate_with_memory.call_count == 3


def test_refactor_tree_assigns_language_metadata_to_go_and_typescript_nodes(tmp_path):
    refactor_go_source = textwrap.dedent(
        """\
        package main

        type Server struct {}
        type Handler struct {}

        func (h *Handler) Handle() {}

        func NewServer() *Server {
            return &Server{}
        }
        """
    )
    refactor_ts_source = textwrap.dedent(
        """\
        export class Greeter {
          greet(): string {
            return "hello";
          }
        }

        export class Helper {}

        export function makeGreeter(): Greeter {
          return new Greeter();
        }
        """
    )

    go_path = tmp_path / "cmd" / "server.go"
    go_path.parent.mkdir(parents=True, exist_ok=True)
    go_path.write_text(refactor_go_source)

    ts_path = tmp_path / "frontend" / "greeter.ts"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text(refactor_ts_source)

    parsed_tree = {
        "cmd/server.go": {
            "_file_summary_": "server runtime",
            "class Server": ["server model"],
            "class Handler": {"Handle": ["handle request"]},
            "function NewServer": ["create server"],
        },
        "frontend/greeter.ts": {
            "_file_summary_": "greeting utilities",
            "class Greeter": {"greet": ["format greeting"]},
            "class Helper": ["helper model"],
            "function makeGreeter": ["create greeter"],
        },
    }

    def fake_process_folder(
        self,
        functional_areas,
        folder_path,
        cur_feature_tree,
        dir_file2node,
        area_update,
        parsed_tree,
        context_window,
        max_iters,
    ):
        area_name = functional_areas[0]
        area_update.setdefault(area_name, {})
        for file_node in dir_file2node.values():
            area_update[area_name][f"{area_name}/Source/{file_node.name}"] = file_node
        return cur_feature_tree, []

    refactor = RefactorTree(
        repo_dir=str(tmp_path),
        repo_info="test repo",
        repo_skeleton="cmd/server.go\nfrontend/greeter.ts",
        repo_name="test-repo",
        llm_client=MagicMock(),
        language="python",
        language_map={"cmd/": "go", "frontend/": "typescript"},
    )

    with patch.object(RefactorTree, "plan_functional_areas", return_value={"final_plan": ["Core"]}), \
         patch.object(RefactorTree, "process_folder", fake_process_folder), \
         patch.object(RefactorTree, "_estimate_batch_tokens_for_process_folder", return_value=1):
        _, _, rpg = refactor.run(parsed_tree, max_iters=1)

    language_by_type = {
        node.meta.type_name: node.meta.language
        for node in rpg.nodes.values()
        if node.meta
        and node.meta.type_name in {NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD}
        and node.meta.path
        and str(node.meta.path).startswith("cmd/")
    }
    assert language_by_type[NodeType.FILE] == "go"
    assert language_by_type[NodeType.CLASS] == "go"
    assert language_by_type[NodeType.FUNCTION] == "go"
    assert language_by_type[NodeType.METHOD] == "go"

    ts_nodes = [
        node
        for node in rpg.nodes.values()
        if node.meta
        and node.meta.type_name in {NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD}
        and node.meta.path
        and str(node.meta.path).startswith("frontend/")
    ]
    assert ts_nodes
    assert {node.meta.language for node in ts_nodes} == {"typescript"}
