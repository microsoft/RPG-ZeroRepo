#!/usr/bin/env python3
"""
Tests for the language parser registry.
"""

import ast
import os
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

import lang_parser
from lang_parser import (
    BaseLanguageParser,
    LPFileResult,
    NotSupported,
    detect_language,
    dominant_language,
    get_config,
    get_config_for_path,
    get_parser,
    get_parser_for_file,
    is_supported_source,
    is_test_file,
    markdown_fence_for_path,
    parse_file,
    validate_syntax,
)


class TestLangParserRegistry:
    def test_import_and_public_api_exports(self):
        assert lang_parser.detect_language is detect_language
        assert lang_parser.parse_file is parse_file
        assert lang_parser.validate_syntax is validate_syntax

    def test_python_config_lookup(self):
        config = get_config("python")
        assert is_dataclass(config)
        assert config.name == "python"
        assert config.display_name == "Python"
        assert config.extensions == (".py",)
        assert config.markdown_fence == "python"
        assert config.tree_sitter_language is None
        assert config.module_path_style == "python"
        assert config.default_test_command == ("uv", "run", "pytest")

    @pytest.mark.parametrize(
        ("language", "extensions", "fence", "tree_sitter_language", "style"),
        [
            ("go", (".go",), "go", "go", "go"),
            ("typescript", (".ts", ".tsx"), "typescript", "typescript", "node"),
            ("javascript", (".js", ".jsx"), "javascript", "javascript", "node"),
            ("c", (".c", ".h"), "c", "c", "c"),
            ("cpp", (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"), "cpp", "cpp", "c"),
            ("rust", (".rs",), "rust", "rust", "rust"),
        ],
    )
    def test_multilingual_config_lookup(self, language, extensions, fence, tree_sitter_language, style):
        config = get_config(language)
        assert is_dataclass(config)
        assert config.name == language
        assert config.extensions == extensions
        assert config.markdown_fence == fence
        assert config.tree_sitter_language == tree_sitter_language
        assert config.module_path_style == style
        assert config.source_globs
        assert config.test_globs

    def test_unknown_config_raises(self):
        with pytest.raises(NotSupported):
            get_config("ruby")

    def test_detects_supported_paths(self):
        assert detect_language("x.py") == "python"
        assert detect_language("pkg/module.py") == "python"
        assert detect_language("./nested/pkg/module.py") == "python"
        assert detect_language("pkg/module.py:SomeClass.method") == "python"
        assert detect_language("main.go") == "go"
        assert detect_language("src/app.ts") == "typescript"
        assert detect_language("src/app.tsx") == "typescript"
        assert detect_language("src/app.js") == "javascript"
        assert detect_language("src/app.jsx") == "javascript"
        assert detect_language("src/math.c") == "c"
        assert detect_language("include/math_utils.h") == "c"
        assert detect_language("src/model.cpp") == "cpp"
        assert detect_language("include/model.hpp") == "cpp"
        assert detect_language("src/main.rs") == "rust"
        assert detect_language("crates/foo/src/lib.rs") == "rust"

    def test_dominant_language_basic_majority(self):
        assert dominant_language(["a.py", "b.py", "c.go"]) == "python"
        assert dominant_language(["a.go"] * 3 + ["b.py"]) == "go"
        assert dominant_language([]) is None
        assert dominant_language(["x.png", "y.md"]) is None

    def test_dominant_language_cpp_with_c_headers(self):
        # A C++ repo that uses .h headers (e.g. googletest): .h detects as C,
        # but the C++-only extensions mean the repo is C++, so C votes fold
        # into C++ rather than letting header count win.
        paths = ["h%d.h" % i for i in range(2018)] + ["s%d.cc" % i for i in range(1062)]
        assert dominant_language(paths) == "cpp"
        # Even a single C++ source flips a header-only-looking repo to C++.
        assert dominant_language(["a.h", "b.h", "c.h", "d.cpp"]) == "cpp"

    def test_dominant_language_pure_c_unaffected(self):
        # No C++ extension present → stays C (no regression).
        assert dominant_language(["a.c", "b.h", "c.h"]) == "c"
        assert dominant_language(["main.c"]) == "c"
        # C alongside an unrelated language must not fold into C++.
        assert dominant_language(["a.c"] * 5 + ["b.go"] * 2) == "c"

    def test_unsupported_paths_are_not_supported_source(self):
        unsupported = [
            "README.md",
            "notes.txt",
            "Makefile",
            "pkg/module",
            "src/app.java",
        ]
        for path in unsupported:
            assert detect_language(path) is None
            assert get_config_for_path(path) is None
            assert get_parser_for_file(path) is None
            assert is_supported_source(path) is False

    def test_supported_source_includes_phase_b_languages(self):
        supported = [
            "main.py",
            "pkg/core.py",
            "pkg/core.py:helper",
            "main.go",
            "src/app.ts",
            "src/app.tsx",
            "src/app.js",
            "src/app.jsx",
            "src/math.c",
            "include/math_utils.h",
            "src/model.cpp",
            "include/model.hpp",
            "src/main.rs",
            "crates/foo/src/lib.rs",
        ]
        for path in supported:
            assert is_supported_source(path) is True

    def test_multilingual_test_file_detection(self):
        test_files = [
            "tests/test_example.py",
            "pkg/foo_test.py",
            "src/test_utils.py",
            "testing/helpers.py",
            "server/server_test.go",
            "tests/helper.go",
            "src/foo.test.ts",
            "src/foo.spec.ts",
            "src/foo.test.tsx",
            "src/foo.spec.tsx",
            "src/foo.test.js",
            "src/foo.spec.js",
            "src/foo.test.jsx",
            "src/foo.spec.jsx",
            "src/__tests__/helper.ts",
            "src/tests/helper.js",
            "server/server_test.c",
            "tests/helper.c",
            "src/app_test.cpp",
            "tests/helper.cpp",
            "tests/helper.rs",
            "crates/foo/tests/integration.rs",
            "examples/demo.rs",
            "benches/bench.rs",
        ]
        for path in test_files:
            assert is_test_file(path) is True

    def test_test_file_detection_avoids_false_positives_and_unsupported(self):
        assert is_test_file("src/contest.py") is False
        assert is_test_file("src/core.py") is False
        assert is_test_file("src/testimonial.ts") is False
        assert is_test_file("src/protest.js") is False
        assert is_test_file("tests/readme.md") is False

    def test_parser_lookup(self):
        for language, path in [
            ("python", "pkg/mod.py"),
            ("go", "main.go"),
            ("typescript", "src/app.ts"),
            ("javascript", "src/app.js"),
            ("c", "src/math.c"),
            ("cpp", "src/model.cpp"),
            ("rust", "src/main.rs"),
        ]:
            parser = get_parser(language)
            assert isinstance(parser, BaseLanguageParser)
            assert get_parser_for_file(path) is parser
        with pytest.raises(NotSupported):
            get_parser("ruby")

    def test_parse_file_public_api_python(self):
        result = parse_file("pkg/mod.py", "import os\n\nx = 1\n")
        assert isinstance(result, LPFileResult)
        assert result.file_path == "pkg/mod.py"
        assert result.language == "python"
        assert result.syntax_error is None
        assert [unit.unit_type for unit in result.units] == ["import", "assignment"]

    def test_validate_syntax_public_api(self):
        assert validate_syntax("pkg/mod.py", "x = 1\n") == (True, None)
        valid, error = validate_syntax("pkg/mod.py", "def broken(\n")
        assert valid is False
        assert error is not None
        unsupported_valid, unsupported_error = validate_syntax("README.md", "text")
        assert unsupported_valid is False
        assert "Unsupported source file" in unsupported_error

    def test_parse_file_unsupported_raises(self):
        with pytest.raises(NotSupported):
            parse_file("README.md", "# docs\n")

    def test_markdown_fence(self):
        assert markdown_fence_for_path("x.py") == "python"
        assert markdown_fence_for_path("main.go") == "go"
        assert markdown_fence_for_path("src/app.ts") == "typescript"
        assert markdown_fence_for_path("src/app.tsx") == "typescript"
        assert markdown_fence_for_path("src/app.js") == "javascript"
        assert markdown_fence_for_path("src/app.jsx") == "javascript"
        assert markdown_fence_for_path("src/math.c") == "c"
        assert markdown_fence_for_path("include/model.hpp") == "cpp"
        assert markdown_fence_for_path("src/main.rs") == "rust"
        assert markdown_fence_for_path("README.md") == "text"

    def test_no_top_level_grammar_package_imports(self):
        forbidden = {
            "tree_sitter_go",
            "tree_sitter_typescript",
            "tree_sitter_javascript",
            "tree_sitter_c",
            "tree_sitter_cpp",
            "tree_sitter_rust",
            "tree_sitter_language_pack",
            "tree_sitter_languages",
        }
        parser_root = Path(_project_root) / "scripts" / "lang_parser"
        for path in parser_root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = {node.module.split(".")[0]}
                else:
                    continue
                assert imported.isdisjoint(forbidden), f"{path} imports {imported & forbidden} at module scope"
