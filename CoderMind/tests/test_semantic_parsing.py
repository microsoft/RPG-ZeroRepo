#!/usr/bin/env python3
"""Tests for M6 Semantic Parsing.

Covers:
  - ParseFeatures._dedupe_file_summaries (pure logic)
  - ParseFeatures.parse_classes (mocked LLM)
  - ParseFeatures.parse_functions (mocked LLM)
  - ParseFeatures._parse_files_global (mocked LLM)
  - ParseFeatures.parse_repo / parse_partial_repo (mocked LLM + filesystem)
  - Token batching helpers
  - New utils: calculate_tokens, truncate_by_token
  - Code unit extensions: ParsedWorkspace, ParsedModule, CodeSnippetBuilder
  - Prompt templates: PARSE_CLASS, PARSE_FUNCTION
"""

import json
import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root and scripts/ are on sys.path
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from common.utils import calculate_tokens, truncate_by_token
from rpg_encoder.prompts import PARSE_CLASS, PARSE_FUNCTION
from rpg_encoder.semantic_parsing import ParseFeatures
from rpg.code_unit import (
    CodeSnippetBuilder,
    CodeUnit,
    ParsedFile,
    ParsedModule,
    ParsedWorkspace,
    compare_code_units,
    merge_codeunits,
)


# ============================================================================
# Fixtures
# ============================================================================

SAMPLE_CODE_A = textwrap.dedent("""\
    import os

    class DataLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return open(self.path).read()

    def helper():
        return 42
""")

SAMPLE_CODE_B = textwrap.dedent("""\
    class Validator:
        def validate(self, data):
            return bool(data)

    def standalone_func(x):
        return x + 1
""")


@pytest.fixture
def sample_file_code_map(tmp_path):
    """Create temporary Python files and return a code map."""
    a_path = str(tmp_path / "module_a.py")
    b_path = str(tmp_path / "module_b.py")
    with open(a_path, "w") as f:
        f.write(SAMPLE_CODE_A)
    with open(b_path, "w") as f:
        f.write(SAMPLE_CODE_B)
    return {a_path: SAMPLE_CODE_A, b_path: SAMPLE_CODE_B}


def _make_parser(tmp_path=None, llm_client=None):
    """Create a ParseFeatures instance with a mocked LLM client."""
    repo_dir = str(tmp_path) if tmp_path else "/tmp/test_repo"
    with patch("rpg_encoder.semantic_parsing.LLMClient"):
        parser = ParseFeatures(
            repo_dir=repo_dir,
            repo_info="A test project",
            repo_skeleton="<skeleton>",
            valid_files=["module_a.py", "module_b.py"],
            repo_name="test-project",
        )
    if llm_client is not None:
        parser.llm_client = llm_client
    return parser


# ============================================================================
# Prompt Templates
# ============================================================================

class TestPromptTemplates:
    """Verify prompt templates are present and contain expected markers."""

    def test_parse_class_has_placeholders(self):
        assert "{repo_name}" in PARSE_CLASS
        assert "{repo_info}" in PARSE_CLASS

    def test_parse_function_has_placeholders(self):
        assert "{repo_name}" in PARSE_FUNCTION
        assert "{repo_info}" in PARSE_FUNCTION

    def test_parse_class_has_solution_tag(self):
        assert "<solution>" in PARSE_CLASS

    def test_parse_function_has_solution_tag(self):
        assert "<solution>" in PARSE_FUNCTION

    def test_parse_class_has_feature_naming_rules(self):
        assert "Feature Naming Rules" in PARSE_CLASS

    def test_parse_function_has_feature_naming_rules(self):
        assert "Feature Naming Rules" in PARSE_FUNCTION

    def test_parse_class_format_works(self):
        result = PARSE_CLASS.format(repo_name="test", repo_info="A test repo")
        assert "test" in result
        assert "A test repo" in result

    def test_parse_function_format_works(self):
        result = PARSE_FUNCTION.format(repo_name="test", repo_info="A test repo")
        assert "test" in result
        assert "A test repo" in result


# ============================================================================
# calculate_tokens / truncate_by_token
# ============================================================================

class TestTokenUtils:

    def test_calculate_tokens_returns_positive(self):
        tokens = calculate_tokens("Hello, world!")
        assert tokens > 0

    def test_calculate_tokens_empty(self):
        tokens = calculate_tokens("")
        assert tokens == 0

    def test_calculate_tokens_longer_text_more_tokens(self):
        short = calculate_tokens("hi")
        long = calculate_tokens("This is a significantly longer piece of text")
        assert long > short

    def test_truncate_by_token_no_truncation(self):
        text = "short text"
        result = truncate_by_token(text, max_tokens=1000)
        assert result == text

    def test_truncate_by_token_truncates(self):
        text = "word " * 20000  # ~20k tokens
        result = truncate_by_token(text, max_tokens=100)
        assert len(result) < len(text)
        assert "truncated" in result.lower()


# ============================================================================
# ParsedWorkspace / ParsedModule
# ============================================================================

class TestParsedWorkspace:

    def test_all_units(self):
        pw = ParsedWorkspace({"a.py": SAMPLE_CODE_A, "b.py": SAMPLE_CODE_B})
        units = pw.all_units()
        # Should find classes, methods, functions, imports
        assert len(units) > 0
        names = [u.name for u in units if u.name]
        assert "DataLoader" in names
        assert "Validator" in names
        assert "helper" in names
        assert "standalone_func" in names

    def test_find_function(self):
        pw = ParsedWorkspace({"a.py": SAMPLE_CODE_A})
        found = pw.find_function("helper")
        assert found is not None
        assert found.name == "helper"
        assert found.unit_type == "function"

    def test_find_function_not_found(self):
        pw = ParsedWorkspace({"a.py": SAMPLE_CODE_A})
        assert pw.find_function("nonexistent") is None


class TestParsedModule:

    def test_basic_parsing(self):
        pm = ParsedModule(SAMPLE_CODE_A)
        assert len(pm.units) > 0

    def test_get_class(self):
        pm = ParsedModule(SAMPLE_CODE_A)
        cls = pm.get_class("DataLoader")
        assert cls is not None
        assert cls.unit_type == "class"

    def test_get_function(self):
        pm = ParsedModule(SAMPLE_CODE_A)
        fn = pm.get_function("helper")
        assert fn is not None
        assert fn.unit_type == "function"

    def test_get_method(self):
        pm = ParsedModule(SAMPLE_CODE_A)
        m = pm.get_method("DataLoader", "load")
        assert m is not None
        assert m.unit_type == "method"

    def test_get_methods_of_class(self):
        pm = ParsedModule(SAMPLE_CODE_A)
        methods = pm.get_methods_of_class("DataLoader")
        method_names = [m.name for m in methods]
        assert "__init__" in method_names
        assert "load" in method_names


# ============================================================================
# CodeSnippetBuilder
# ============================================================================

class TestCodeSnippetBuilder:

    def test_build_basic(self):
        file_code_map = {"test.py": SAMPLE_CODE_A}
        parsed_files = {"test.py": ParsedFile(SAMPLE_CODE_A, "test.py")}
        builder = CodeSnippetBuilder(file_code_map, parsed_files)

        units = parsed_files["test.py"].units
        result = builder.build(merged=units)
        assert "DataLoader" in result
        assert "helper" in result

    def test_build_empty(self):
        builder = CodeSnippetBuilder({}, {})
        result = builder.build(merged=[])
        assert result == ""

    def test_build_with_file_path(self):
        file_code_map = {"test.py": SAMPLE_CODE_A}
        parsed_files = {"test.py": ParsedFile(SAMPLE_CODE_A, "test.py")}
        builder = CodeSnippetBuilder(file_code_map, parsed_files)

        units = parsed_files["test.py"].units
        result = builder.build(merged=units, with_file_path=True)
        assert "test.py" in result


# ============================================================================
# merge_codeunits
# ============================================================================

class TestMergeCodeunits:

    def test_merge_promotes_complete_methods(self):
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        # Select all methods of DataLoader
        methods = [u for u in parsed.units if u.unit_type == "method" and u.parent == "DataLoader"]
        result = merge_codeunits(
            methods,
            {"test.py": parsed},
        )
        # Should have promoted to the full class (since all methods are selected)
        types = [u.unit_type for u in result]
        assert "class" in types

    def test_merge_functions(self):
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        funcs = [u for u in parsed.units if u.unit_type == "function"]
        result = merge_codeunits(funcs, {"test.py": parsed})
        names = [u.name for u in result]
        assert "helper" in names


# ============================================================================
# compare_code_units
# ============================================================================

class TestCompareCodeUnits:

    def test_identical_functions(self):
        code = "def foo(): return 1"
        pf1 = ParsedFile(code, "a.py")
        pf2 = ParsedFile(code, "b.py")
        u1 = pf1.get_unit_by_name("foo")
        u2 = pf2.get_unit_by_name("foo")
        assert compare_code_units(u1, u2)

    def test_different_functions(self):
        pf1 = ParsedFile("def foo(): return 1", "a.py")
        pf2 = ParsedFile("def foo(): return 2", "b.py")
        u1 = pf1.get_unit_by_name("foo")
        u2 = pf2.get_unit_by_name("foo")
        assert not compare_code_units(u1, u2)

    def test_ignore_docstring(self):
        code1 = 'def foo():\n    """Docstring."""\n    return 1'
        code2 = "def foo():\n    return 1"
        pf1 = ParsedFile(code1, "a.py")
        pf2 = ParsedFile(code2, "b.py")
        u1 = pf1.get_unit_by_name("foo")
        u2 = pf2.get_unit_by_name("foo")
        assert compare_code_units(u1, u2, ignore_docstring=True)


# ============================================================================
# CodeUnit.is_unimplemented_base_class
# ============================================================================

class TestIsUnimplementedBaseClass:

    def test_abstract_class(self):
        code = textwrap.dedent("""\
            class Base:
                def method_a(self):
                    pass
                def method_b(self):
                    ...
        """)
        pf = ParsedFile(code, "test.py")
        cls = pf.get_unit_by_name("Base")
        assert cls.is_unimplemented_base_class is True

    def test_implemented_class(self):
        code = textwrap.dedent("""\
            class Impl:
                def method_a(self):
                    return 42
        """)
        pf = ParsedFile(code, "test.py")
        cls = pf.get_unit_by_name("Impl")
        assert cls.is_unimplemented_base_class is False

    def test_class_without_methods(self):
        code = textwrap.dedent("""\
            class Empty:
                x = 1
        """)
        pf = ParsedFile(code, "test.py")
        cls = pf.get_unit_by_name("Empty")
        assert cls.is_unimplemented_base_class is False

    def test_not_a_class(self):
        code = "def foo(): pass"
        pf = ParsedFile(code, "test.py")
        fn = pf.get_unit_by_name("foo")
        assert fn.is_unimplemented_base_class is False


# ============================================================================
# _dedupe_file_summaries
# ============================================================================

class TestDedupeFileSummaries:

    def test_no_duplicates(self):
        parser = _make_parser()
        repo_map = {
            "a.py": {"_file_summary_": "config loader"},
            "b.py": {"_file_summary_": "data parser"},
        }
        result = parser._dedupe_file_summaries(repo_map)
        assert result["a.py"]["_file_summary_"] == "config loader"
        assert result["b.py"]["_file_summary_"] == "data parser"

    def test_duplicate_summaries_get_suffix(self):
        parser = _make_parser()
        repo_map = {
            "a.py": {"_file_summary_": "utils"},
            "b.py": {"_file_summary_": "utils"},
            "c.py": {"_file_summary_": "utils"},
        }
        result = parser._dedupe_file_summaries(repo_map)
        summaries = [
            result["a.py"]["_file_summary_"],
            result["b.py"]["_file_summary_"],
            result["c.py"]["_file_summary_"],
        ]
        # All must be unique
        assert len(set(summaries)) == 3
        # One should be "utils", others "utils_1" / "utils_2"
        assert "utils" in summaries

    def test_missing_summary_uses_filename(self):
        parser = _make_parser()
        repo_map = {
            "my_module.py": {"class Foo": ["feature1"]},
        }
        result = parser._dedupe_file_summaries(repo_map)
        assert result["my_module.py"]["_file_summary_"] == "my_module"

    def test_slash_replaced(self):
        parser = _make_parser()
        repo_map = {
            "a.py": {"_file_summary_": "input/output handler"},
        }
        result = parser._dedupe_file_summaries(repo_map)
        assert "/" not in result["a.py"]["_file_summary_"]

    def test_empty_map(self):
        parser = _make_parser()
        result = parser._dedupe_file_summaries({})
        assert result == {}


# ============================================================================
# parse_classes (mocked LLM)
# ============================================================================

class TestParseClasses:

    def _make_mock_llm(self, responses):
        """Create a mock LLM client returning pre-defined responses."""
        mock = MagicMock()
        mock.generate_with_memory = MagicMock(side_effect=responses)
        return mock

    def test_parse_classes_single_class(self):
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})

        cls_units = [u for u in parsed.units if u.unit_type in ("class", "method")]

        llm_response = (
            '<solution>\n'
            '{"DataLoader": {"__init__": {"initialize path": "Configures the loader input directory."}, '
            '"load": {"read file": "Reads the source file into memory."}}}\n'
            '</solution>'
        )
        mock_llm = self._make_mock_llm([llm_response])
        parser = _make_parser(llm_client=mock_llm)

        features, descs, messages = parser.parse_classes(
            code_builder=builder,
            cls_units=cls_units,
            max_iterations=1,
        )

        assert "DataLoader" in features
        assert "__init__" in features["DataLoader"]
        assert "load" in features["DataLoader"]
        # Names preserved as list-of-strings (main structure unchanged).
        assert "initialize path" in features["DataLoader"]["__init__"]
        # Descriptions land in composite-key sidecar map.
        assert (
            descs["DataLoader::__init__::initialize path"]
            == "Configures the loader input directory."
        )
        assert (
            descs["DataLoader::load::read file"]
            == "Reads the source file into memory."
        )

    def test_parse_classes_handles_none_response(self):
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})

        cls_units = [u for u in parsed.units if u.unit_type in ("class", "method")]

        mock_llm = self._make_mock_llm([None])
        parser = _make_parser(llm_client=mock_llm)

        features, descs, messages = parser.parse_classes(
            code_builder=builder,
            cls_units=cls_units,
            max_iterations=1,
        )
        # Should not crash, features may be empty
        assert isinstance(features, dict)
        assert isinstance(descs, dict)

    def test_parse_classes_tolerates_legacy_list_schema(self):
        """Legacy ``{ClassName: {method: [feat]}}`` schema still parses (descriptions are simply empty)."""
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})
        cls_units = [u for u in parsed.units if u.unit_type in ("class", "method")]

        legacy_response = (
            '<solution>\n'
            '{"DataLoader": {"__init__": ["initialize path"], '
            '"load": ["read file"]}}\n'
            '</solution>'
        )
        parser = _make_parser(
            llm_client=self._make_mock_llm([legacy_response])
        )
        features, descs, _ = parser.parse_classes(
            code_builder=builder, cls_units=cls_units, max_iterations=1,
        )
        assert "initialize path" in features["DataLoader"]["__init__"]
        assert descs == {}  # legacy format has no descriptions


# ============================================================================
# parse_functions (mocked LLM)
# ============================================================================

class TestParseFunctions:

    def _make_mock_llm(self, responses):
        mock = MagicMock()
        mock.generate_with_memory = MagicMock(side_effect=responses)
        return mock

    def test_parse_functions_basic(self):
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})

        func_units = [u for u in parsed.units if u.unit_type == "function"]

        llm_response = (
            '<solution>\n'
            '{"helper": {"return fixed value": "Returns a hardcoded constant for tests."}}\n'
            '</solution>'
        )
        mock_llm = self._make_mock_llm([llm_response])
        parser = _make_parser(llm_client=mock_llm)

        features, descs, messages = parser.parse_functions(
            code_builder=builder,
            func_units=func_units,
            max_iterations=1,
        )

        assert "helper" in features
        assert isinstance(features["helper"], list)
        assert "return fixed value" in features["helper"]
        assert (
            descs["helper::return fixed value"]
            == "Returns a hardcoded constant for tests."
        )

    def test_parse_functions_slash_replaced(self):
        """Feature *names* containing ``/`` are normalised to ``or``; descriptions, however, keep their ``/`` (e.g. ``"client/server"``).

        The composite desc key MUST use the normalised name so that
        ``_init_feature_tree`` can resolve the description from the
        Node's feature name.
        """
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})
        func_units = [u for u in parsed.units if u.unit_type == "function"]

        llm_response = (
            '<solution>\n'
            '{"helper": {"read/write data": "Reads or writes data on the client/server pair."}}\n'
            '</solution>'
        )
        mock_llm = self._make_mock_llm([llm_response])
        parser = _make_parser(llm_client=mock_llm)

        features, descs, _ = parser.parse_functions(
            code_builder=builder,
            func_units=func_units,
            max_iterations=1,
        )
        # Name slash replaced.
        assert "read or write data" in features["helper"]
        # Description value keeps `/` verbatim (legitimate "client/server").
        assert any("client/server" in v for v in descs.values())
        # Crucially: the desc-map key MUST use the normalised name so
        # downstream consumers looking up by the stored feature name can
        # actually find the description.  This guards against a regression
        # where the key used the raw "/"-form name (bug v3.2 introduced).
        assert (
            descs["helper::read or write data"]
            == "Reads or writes data on the client/server pair."
        )
        # And the raw-form key MUST NOT be present.
        assert "helper::read/write data" not in descs


# ============================================================================
# _parse_files_global (mocked LLM)
# ============================================================================

class TestParseFilesGlobal:

    def _make_mock_llm_for_global(self):
        """Build a mock LLM that returns canned responses for classes, functions, and file summaries."""
        class_resp = (
            '<solution>\n'
            '{"DataLoader": {'
            '"__init__": {"initialize path": "Configures the loader\'s input directory."}, '
            '"load": {"read file data": "Reads the source file into an in-memory buffer."}'
            '}}\n'
            '</solution>'
        )
        func_resp = (
            '<solution>\n'
            '{"helper": {"return fixed value": "Returns a hardcoded constant for tests."}}\n'
            '</solution>'
        )
        summary_resp = (
            '<solution>\n'
            '{"module_a.py": "data loading utilities", '
            '"module_b.py": "validation helpers"}\n'
            '</solution>'
        )

        mock = MagicMock()
        # The mock returns different responses depending on call order;
        # use a list that cycles if needed
        responses = [class_resp, func_resp, summary_resp] * 5
        mock.generate_with_memory = MagicMock(side_effect=responses)
        return mock

    def test_parse_files_global_returns_features(self):
        mock_llm = self._make_mock_llm_for_global()
        parser = _make_parser(llm_client=mock_llm)

        file_code_map = {
            "module_a.py": SAMPLE_CODE_A,
        }

        features, trajectories = parser._parse_files_global(
            file_code_map=file_code_map,
            max_workers=1,
            max_iterations=1,
        )

        assert "module_a.py" in features
        assert isinstance(features["module_a.py"], dict)

    def test_parse_files_global_propagates_descriptions(self):
        """LLM-emitted descriptions end up in the file-level ``_feature_descriptions_`` sidecar under composite keys."""
        mock_llm = self._make_mock_llm_for_global()
        parser = _make_parser(llm_client=mock_llm)

        features, _ = parser._parse_files_global(
            file_code_map={"module_a.py": SAMPLE_CODE_A},
            max_workers=1,
            max_iterations=1,
        )
        file_map = features["module_a.py"]
        assert "_feature_descriptions_" in file_map
        descs = file_map["_feature_descriptions_"]
        # Method-level description (composite key with class + method + feat).
        assert (
            descs.get("DataLoader::__init__::initialize path")
            == "Configures the loader's input directory."
        )
        # Function-level description (composite key with func + feat).
        assert (
            descs.get("helper::return fixed value")
            == "Returns a hardcoded constant for tests."
        )

    def test_parse_files_global_empty_input(self):
        mock_llm = MagicMock()
        parser = _make_parser(llm_client=mock_llm)

        features, trajectories = parser._parse_files_global(
            file_code_map={},
            max_workers=1,
        )
        assert features == {}
        assert trajectories == []


# ============================================================================
# parse_partial_repo (mocked LLM)
# ============================================================================

class TestParsePartialRepo:

    def test_parse_partial_repo(self):
        class_resp = (
            '<solution>\n'
            '{"DataLoader": {"__init__": ["init"], "load": ["load"]}}\n'
            '</solution>'
        )
        func_resp = '<solution>\n{"helper": ["help"]}\n</solution>'
        summary_resp = '<solution>\n{"a.py": "module a"}\n</solution>'

        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(
            side_effect=[class_resp, func_resp, summary_resp] * 5
        )

        parser = _make_parser(llm_client=mock_llm)
        features, trajs = parser.parse_partial_repo(
            file_code_map={"a.py": SAMPLE_CODE_A},
            max_workers=1,
            max_iterations=1,
        )
        assert "a.py" in features


# ============================================================================
# parse_repo (mocked LLM + filesystem)
# ============================================================================

class TestParseRepo:

    def test_parse_repo_with_files(self, tmp_path):
        a_path = tmp_path / "module_a.py"
        a_path.write_text(SAMPLE_CODE_A)

        class_resp = (
            '<solution>\n'
            '{"DataLoader": {"__init__": ["init"], "load": ["load"]}}\n'
            '</solution>'
        )
        func_resp = '<solution>\n{"helper": ["help"]}\n</solution>'
        summary_resp = (
            '<solution>\n'
            f'{{"{str(a_path)}": "data loading module"}}\n'
            '</solution>'
        )

        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(
            side_effect=[class_resp, func_resp, summary_resp] * 5
        )

        with patch("rpg_encoder.semantic_parsing.LLMClient"):
            parser = ParseFeatures(
                repo_dir=str(tmp_path),
                repo_info="Test project",
                repo_skeleton="<skeleton>",
                valid_files=["module_a.py"],
                repo_name="test-project",
            )
        parser.llm_client = mock_llm

        features, _ = parser.parse_repo(max_workers=1, max_iterations=1)
        # Keys should be normalized relative paths
        assert "module_a.py" in features

    def test_parse_repo_excludes_files(self, tmp_path):
        a_path = tmp_path / "module_a.py"
        b_path = tmp_path / "module_b.py"
        a_path.write_text(SAMPLE_CODE_A)
        b_path.write_text(SAMPLE_CODE_B)

        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(
            return_value='<solution>\n{}\n</solution>'
        )

        with patch("rpg_encoder.semantic_parsing.LLMClient"):
            parser = ParseFeatures(
                repo_dir=str(tmp_path),
                repo_info="Test project",
                repo_skeleton="<skeleton>",
                valid_files=["module_a.py", "module_b.py"],
                repo_name="test-project",
            )
        parser.llm_client = mock_llm

        features, _ = parser.parse_repo(
            excluded_files=["module_b.py"],
            max_workers=1,
            max_iterations=1,
        )
        # module_b.py was excluded, so only module_a.py should appear
        assert "module_b.py" not in features


# ============================================================================
# Edge cases & error handling
# ============================================================================

class TestEdgeCases:

    def test_parse_classes_with_json_error(self):
        """When LLM returns invalid JSON, should not crash."""
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})
        cls_units = [u for u in parsed.units if u.unit_type in ("class", "method")]

        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(return_value="not valid json at all")

        parser = _make_parser(llm_client=mock_llm)
        features, _, _ = parser.parse_classes(
            code_builder=builder,
            cls_units=cls_units,
            max_iterations=1,
        )
        # Should return empty features without crashing
        assert isinstance(features, dict)

    def test_parse_functions_with_exception(self):
        """When LLM raises exception, should not crash."""
        parsed = ParsedFile(SAMPLE_CODE_A, "test.py")
        file_code_map = {"test.py": SAMPLE_CODE_A}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})
        func_units = [u for u in parsed.units if u.unit_type == "function"]

        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(side_effect=RuntimeError("API error"))

        parser = _make_parser(llm_client=mock_llm)
        features, _, _ = parser.parse_functions(
            code_builder=builder,
            func_units=func_units,
            max_iterations=1,
        )
        assert isinstance(features, dict)

    def test_syntax_error_file(self):
        """ParsedFile handles syntax errors gracefully."""
        bad_code = "def foo(\n"  # Invalid syntax
        pf = ParsedFile(bad_code, "bad.py")
        assert pf.has_error()
        assert len(pf.units) == 0

    def test_parse_classes_iterative_followup(self):
        """Test that missing-class follow-up mechanism works."""
        code = textwrap.dedent("""\
            class Alpha:
                def run(self):
                    return 1

            class Beta:
                def execute(self):
                    return 2
        """)
        parsed = ParsedFile(code, "test.py")
        file_code_map = {"test.py": code}
        builder = CodeSnippetBuilder(file_code_map, {"test.py": parsed})
        cls_units = [u for u in parsed.units if u.unit_type in ("class", "method")]

        # First response only covers Alpha; second covers Beta
        resp1 = (
            '<solution>\n'
            '{"Alpha": {"run": ["execute run"]}}\n'
            '</solution>'
        )
        resp2 = (
            '<solution>\n'
            '{"Beta": {"execute": ["execute task"]}}\n'
            '</solution>'
        )
        mock_llm = MagicMock()
        mock_llm.generate_with_memory = MagicMock(side_effect=[resp1, resp2])

        parser = _make_parser(llm_client=mock_llm)
        features, descs, messages = parser.parse_classes(
            code_builder=builder,
            cls_units=cls_units,
            max_iterations=3,
        )

        assert "Alpha" in features
        assert "Beta" in features
        assert "run" in features["Alpha"]
        assert "execute" in features["Beta"]
        # Followup-merged descriptions should accumulate across iterations.
        # (Legacy schema in this test = empty desc map, but still a dict.)
        assert isinstance(descs, dict)
