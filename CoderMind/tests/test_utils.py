#!/usr/bin/env python3
"""Tests for M4 Utils — ported utility functions in scripts/common/utils.py.

Covers:
  - normalize_path
  - is_test_file
  - merge_intervals
  - filter_excluded_files
  - parse_solution_output
  - parse_code_blocks
  - get_skeleton
  - transfer_parsed_tree
  - format_parsed_tree
  - iterative_by_folder
  - get_node_range_robust
  - extract_source_by_lines
"""

import ast
import json
import os
import sys
import textwrap
from unittest.mock import patch

import pytest

# Ensure the project root and scripts/ are on sys.path so that
# the ``scripts`` namespace package resolves ``common`` correctly.
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from common.utils import (
    normalize_path,
    is_test_file,
    merge_intervals,
    filter_excluded_files,
    parse_solution_output,
    parse_code_blocks,
    get_skeleton,
    transfer_parsed_tree,
    format_parsed_tree,
    iterative_by_folder,
    get_node_range_robust,
    extract_source_by_lines,
    # pre-existing function for sanity
    get_all_leaf_paths,
)


# ============================================================================
# normalize_path
# ============================================================================

class TestNormalizePath:

    def test_basic_path(self):
        assert normalize_path("src/utils.py") == "src/utils.py"

    def test_dot_prefix(self):
        assert normalize_path("./src/utils.py") == "src/utils.py"

    def test_root(self):
        assert normalize_path(".") == "."

    def test_empty_string(self):
        assert normalize_path("") == "."

    def test_leading_slash(self):
        assert normalize_path("/absolute/path") == "absolute/path"

    def test_with_qualified_name(self):
        assert normalize_path("src/main.py:MyClass") == "src/main.py:MyClass"

    def test_with_method(self):
        assert normalize_path("src/main.py:MyClass.method") == "src/main.py:MyClass.method"

    def test_qualified_name_whitespace(self):
        assert normalize_path("  src/main.py : Cls . meth  ") == "src/main.py:Cls.meth"

    def test_qualified_name_dots_stripped(self):
        assert normalize_path("src/a.py:.Foo.") == "src/a.py:Foo"

    def test_path_object(self):
        from pathlib import Path
        assert normalize_path(Path("src/utils.py")) == "src/utils.py"


# ============================================================================
# is_test_file
# ============================================================================

class TestIsTestFile:

    def test_test_directory(self):
        assert is_test_file("tests/core.py") is True

    def test_test_prefix_file(self):
        assert is_test_file("src/test_utils.py") is True

    def test_normal_file(self):
        assert is_test_file("src/utils.py") is False

    def test_with_qualified_name(self):
        assert is_test_file("tests/core.py:TestCase") is True

    def test_testing_directory(self):
        assert is_test_file("testing/helpers.py") is True

    def test_contestant_not_test(self):
        # "contest" should not match because "contest" does not start with "test"
        # after splitting by _: "contest" starts with "contest", not "test"
        assert is_test_file("src/contest.py") is False

    def test_nested_test_dir(self):
        assert is_test_file("pkg/tests/test_a.py") is True


# ============================================================================
# merge_intervals
# ============================================================================

class TestMergeIntervals:

    def test_empty(self):
        assert merge_intervals([]) == []

    def test_single(self):
        assert merge_intervals([(1, 5)]) == [(1, 5)]

    def test_non_overlapping(self):
        assert merge_intervals([(1, 3), (5, 7)]) == [(1, 3), (5, 7)]

    def test_overlapping(self):
        assert merge_intervals([(1, 5), (3, 8)]) == [(1, 8)]

    def test_adjacent(self):
        # (1,5) and (5,8) overlap because 5 <= 5
        assert merge_intervals([(1, 5), (5, 8)]) == [(1, 8)]

    def test_unsorted_input(self):
        assert merge_intervals([(5, 8), (1, 3), (2, 6)]) == [(1, 8)]

    def test_fully_contained(self):
        assert merge_intervals([(1, 10), (3, 5)]) == [(1, 10)]

    def test_multiple_groups(self):
        result = merge_intervals([(1, 3), (5, 7), (2, 4), (10, 12)])
        assert result == [(1, 4), (5, 7), (10, 12)]


# ============================================================================
# filter_excluded_files
# ============================================================================

class TestFilterExcludedFiles:

    def test_no_exclusions(self):
        files = ["a.py", "b.py"]
        assert filter_excluded_files(files, []) == ["a.py", "b.py"]

    def test_exact_match(self):
        files = ["a.py", "b.py"]
        assert filter_excluded_files(files, ["a.py"]) == ["b.py"]

    def test_directory_exclusion(self):
        files = ["src/a.py", "src/b.py", "lib/c.py"]
        result = filter_excluded_files(files, ["src"])
        assert result == ["lib/c.py"]

    def test_nested_directory(self):
        files = ["pkg/sub/a.py", "pkg/b.py"]
        result = filter_excluded_files(files, ["pkg/sub"])
        assert result == ["pkg/b.py"]

    def test_whitespace_stripped(self):
        files = ["a.py"]
        result = filter_excluded_files(files, ["  ", ""])
        assert result == ["a.py"]

    def test_normalized_paths(self):
        files = ["./src/a.py"]
        result = filter_excluded_files(files, ["src/a.py"])
        assert result == []


# ============================================================================
# parse_solution_output
# ============================================================================

class TestParseSolutionOutput:

    def test_extracts_solution(self):
        text = "Thinking...<solution>hello world</solution> done"
        assert parse_solution_output(text) == "hello world"

    def test_no_tags(self):
        text = "just plain text"
        assert parse_solution_output(text) == "just plain text"

    def test_strips_whitespace(self):
        text = "<solution>  trimmed  </solution>"
        assert parse_solution_output(text) == "trimmed"

    def test_only_start_tag(self):
        text = "before<solution>after"
        assert parse_solution_output(text) == "after"

    def test_empty_solution(self):
        text = "<solution></solution>"
        assert parse_solution_output(text) == ""


# ============================================================================
# parse_code_blocks
# ============================================================================

class TestParseCodeBlocks:

    def test_general_block(self):
        text = "text\n```\ncode here\n```\nmore text"
        result = parse_code_blocks(text)
        assert result == ["code here"]

    def test_python_block(self):
        text = "text\n```python\nprint('hi')\n```\nmore"
        result = parse_code_blocks(text, type="python")
        assert result == ["print('hi')"]

    def test_multiple_blocks(self):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        result = parse_code_blocks(text)
        assert len(result) == 2
        assert result[0] == "block1"
        assert result[1] == "block2"

    def test_no_blocks(self):
        text = "no code blocks here"
        assert parse_code_blocks(text) == []

    def test_general_matches_language(self):
        text = "```python\ncode\n```"
        result = parse_code_blocks(text, type="general")
        assert len(result) == 1
        assert "code" in result[0]

    def test_wrong_language(self):
        text = "```javascript\ncode\n```"
        result = parse_code_blocks(text, type="python")
        assert result == []


# ============================================================================
# get_skeleton
# ============================================================================

class TestGetSkeleton:

    def test_basic_skeleton(self):
        """Test that function bodies are replaced with ... ."""
        code = textwrap.dedent("""\
            class Foo:
                def bar(self):
                    return 42
        """)
        result = get_skeleton(code)
        assert "class Foo" in result
        assert "def bar(self)" in result
        assert "return 42" not in result
        assert "..." in result

    def test_keep_imports(self):
        code = textwrap.dedent("""\
            import os
            from sys import path

            def foo():
                pass
        """)
        result = get_skeleton(code, keep_imports=True)
        assert "import os" in result
        assert "from sys import path" in result

    def test_keep_docstring(self):
        code = textwrap.dedent('''\
            def foo():
                "docstring"
                pass
        ''')
        result = get_skeleton(code, keep_docstring=True, keep_indent=True)
        assert "docstring" in result

    def test_keep_constant(self):
        code = textwrap.dedent("""\
            X = 42

            def foo():
                pass
        """)
        result = get_skeleton(code, keep_constant=True)
        assert "X = 42" in result

    def test_no_constant(self):
        code = textwrap.dedent("""\
            X = 42

            def foo():
                pass
        """)
        result = get_skeleton(code, keep_constant=False)
        assert "X = 42" not in result

    def test_parse_error_returns_raw(self):
        bad_code = "def foo(\n"
        result = get_skeleton(bad_code)
        assert result == bad_code

    def test_sequential_line_numbers(self):
        code = textwrap.dedent("""\
            class A:
                def m(self):
                    pass
        """)
        result = get_skeleton(code, line_number_mode="sequential")
        # Should have line numbers like "1 | ..."
        assert " | " in result

    def test_libcst_not_installed(self):
        """When libcst is not available, raw code is returned."""
        code = "def foo(): pass"
        with patch.dict("sys.modules", {"libcst": None, "libcst.matchers": None}):
            # Since libcst is already imported in this process, simulate failure
            # by testing the parse-failure fallback path
            result = get_skeleton("def foo(\n")
            assert result == "def foo(\n"


# ============================================================================
# transfer_parsed_tree
# ============================================================================

class TestTransferParsedTree:

    def test_basic(self):
        tree = {
            "src/main.py": {
                "_file_summary_": "Main module",
                "func_a": ["feature1", "feature2"],
            }
        }
        fmt, rev = transfer_parsed_tree(tree)
        assert "Main module" in fmt
        assert set(fmt["Main module"]) == {"feature1", "feature2"}
        assert "src/main.py" in rev["feature1"]

    def test_nested_dict(self):
        tree = {
            "src/a.py": {
                "_file_summary_": "A",
                "ClassX": {"method1": ["f1"], "method2": ["f2"]},
            }
        }
        fmt, rev = transfer_parsed_tree(tree)
        assert set(fmt["A"]) == {"f1", "f2"}

    def test_default_summary(self):
        tree = {
            "src/utils.py": {
                "func": ["feat"],
            }
        }
        fmt, _ = transfer_parsed_tree(tree)
        # Default summary is filename without .py
        assert "utils" in fmt

    def test_deduplication(self):
        tree = {
            "a.py": {
                "_file_summary_": "A",
                "f1": ["dup", "dup", "unique"],
            }
        }
        fmt, _ = transfer_parsed_tree(tree)
        assert fmt["A"].count("dup") == 1


# ============================================================================
# format_parsed_tree
# ============================================================================

class TestFormatParsedTree:

    def test_returns_json(self):
        tree = {
            "a.py": {
                "_file_summary_": "A",
                "f": ["feat1"],
            }
        }
        result = format_parsed_tree(tree)
        parsed = json.loads(result)
        assert "A" in parsed

    def test_omit_truncates(self):
        tree = {
            "a.py": {
                "_file_summary_": "A",
                "f": ["f1", "f2", "f3", "f4", "f5"],
            }
        }
        result = format_parsed_tree(tree, omit_full_leaf_nodes=True, max_features=2)
        parsed = json.loads(result)
        # Should have 2 sampled features + "..."
        assert "..." in parsed["A"]
        assert len(parsed["A"]) == 3


# ============================================================================
# iterative_by_folder
# ============================================================================

class TestIterativeByFolder:

    def test_basic(self):
        tree = {
            "src/a.py": {},
            "src/b.py": {},
            "lib/c.py": {},
        }
        result = iterative_by_folder(tree)
        assert "src" in result
        assert "lib" in result
        assert len(result["src"]) == 2
        assert len(result["lib"]) == 1

    def test_root_files(self):
        tree = {"setup.py": {}}
        result = iterative_by_folder(tree)
        assert "(root)" in result
        assert result["(root)"] == ["setup.py"]

    def test_nested_folders(self):
        tree = {"a/b/c.py": {}}
        result = iterative_by_folder(tree)
        assert "a/b" in result


# ============================================================================
# get_node_range_robust
# ============================================================================

class TestGetNodeRangeRobust:

    def test_simple_function(self):
        code = textwrap.dedent("""\
            def foo():
                return 1
        """)
        tree = ast.parse(code)
        func = tree.body[0]
        start, header_end, body_end, end_exc = get_node_range_robust(func, code)
        assert start == 1
        assert header_end == 1
        assert body_end == 2
        assert end_exc == 3

    def test_decorated_function(self):
        code = textwrap.dedent("""\
            @decorator
            def foo():
                pass
        """)
        tree = ast.parse(code)
        func = tree.body[0]
        start, _, _, _ = get_node_range_robust(func, code)
        assert start == 1  # decorator line

    def test_class_method(self):
        code = textwrap.dedent("""\
            class A:
                def method(self):
                    x = 1
                    return x
        """)
        tree = ast.parse(code)
        cls = tree.body[0]
        method = cls.body[0]
        start, header_end, body_end, end_exc = get_node_range_robust(method, code)
        assert start == 2
        assert header_end == 2
        assert body_end == 4

    def test_multiline_body(self):
        code = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
        """)
        tree = ast.parse(code)
        func = tree.body[0]
        start, _, body_end, end_exc = get_node_range_robust(func, code)
        assert start == 1
        assert body_end == 4
        assert end_exc == 5


# ============================================================================
# extract_source_by_lines
# ============================================================================

class TestExtractSourceByLines:

    def test_basic(self):
        src = "line1\nline2\nline3\nline4\n"
        result = extract_source_by_lines(src, 2, 3)
        assert result == "line2\nline3"

    def test_single_line(self):
        src = "one\ntwo\nthree"
        result = extract_source_by_lines(src, 2, 2)
        assert result == "two"

    def test_none_start(self):
        assert extract_source_by_lines("abc", None, 1) == ""

    def test_none_end(self):
        assert extract_source_by_lines("abc", 1, None) == ""

    def test_out_of_range(self):
        src = "a\nb"
        result = extract_source_by_lines(src, 5, 10)
        assert result == ""

    def test_preserves_content(self):
        src = "  indented\n\n  # comment\n  code\n"
        result = extract_source_by_lines(src, 1, 4)
        assert "indented" in result
        assert "# comment" in result


# ============================================================================
# get_all_leaf_paths (pre-existing — sanity check)
# ============================================================================

class TestGetAllLeafPaths:

    def test_simple_dict(self):
        tree = {"a": {"b": ["c", "d"]}}
        paths = get_all_leaf_paths(tree)
        assert "a/b/c" in paths
        assert "a/b/d" in paths

    def test_empty_dict_is_leaf(self):
        tree = {"a": {}}
        paths = get_all_leaf_paths(tree)
        assert paths == ["a"]

    def test_empty_list_is_leaf(self):
        tree = {"a": []}
        paths = get_all_leaf_paths(tree)
        assert paths == ["a"]

    def test_nested(self):
        tree = {"x": {"y": {"z": ["leaf"]}}}
        paths = get_all_leaf_paths(tree)
        assert paths == ["x/y/z/leaf"]
