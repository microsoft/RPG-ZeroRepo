#!/usr/bin/env python3
"""Tests for language-neutral encoder prompt wording."""

import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts"))

from rpg_encoder.prompts import EXCLUDE_FILES, PARSE_CLASS, PARSE_FUNCTION


PROMPT_FILES = [
    _project_root / "scripts" / "rpg_encoder" / "prompts" / "parse_prompts.py",
    _project_root / "scripts" / "rpg_encoder" / "prompts" / "encoding_prompts.py",
]

FORBIDDEN_ENCODER_PROMPT_TERMS = [
    ".py only",
    "Python classes",
    "Python repository",
    "__init__",
    "pandas.DataFrame",
    "pyarrow.Table",
]


def test_prompt_files_do_not_contain_forbidden_python_only_terms():
    prompt_text = "\n".join(path.read_text() for path in PROMPT_FILES)
    for term in FORBIDDEN_ENCODER_PROMPT_TERMS:
        assert term not in prompt_text


def test_prompt_files_do_not_scope_exclusion_to_python_extensions():
    prompt_text = "\n".join(path.read_text() for path in PROMPT_FILES)
    assert "Consider only:\n1) `.py` files" not in prompt_text
    assert "Directories containing `.py` files" not in prompt_text


def test_solution_output_schemas_are_preserved():
    # The prompt was updated to emit a richer ``{feature: description}``
    # mapping (instead of the legacy ``[feature1, feature2]`` array).
    # The multilingual scrub must not regress the example payloads —
    # downstream parsers (``semantic_parsing.py``) rely on these exact
    # shapes when validating LLM output.
    assert "<solution>" in PARSE_CLASS
    assert "</solution>" in PARSE_CLASS
    # Class examples: dict-of-dict with method -> {feature: description}.
    assert '"method_1": {{' in PARSE_CLASS
    assert '"feature 1": "description of feature 1"' in PARSE_CLASS
    assert "<solution>" in PARSE_FUNCTION
    assert "</solution>" in PARSE_FUNCTION
    # Function examples: dict-of-dict with func_name -> {feature: description}.
    assert '"func_name_1": {{' in PARSE_FUNCTION
    assert '"feature one": "description of feature one"' in PARSE_FUNCTION
    assert "<solution>" in EXCLUDE_FILES
    assert "</solution>" in EXCLUDE_FILES
