from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.git_utils import sanitize_branch_component  # noqa: E402


def test_trailing_dot_after_truncation_is_removed() -> None:
    # The exact id that produced an invalid 'batch/...state.' ref for C++.
    batch_id = "src_expression_calculator_syntax_expression_state.cpp_20260613_082726_e88325bc"

    safe = sanitize_branch_component(batch_id, max_len=50, fallback="batch")

    assert safe == "src_expression_calculator_syntax_expression_state"
    assert not safe.endswith(".")


def test_empty_and_separator_only_values_use_fallback() -> None:
    assert sanitize_branch_component("", fallback="batch") == "batch"
    assert sanitize_branch_component("   ", fallback="task") == "task"
    assert sanitize_branch_component("///", fallback="review") == "review"
    assert sanitize_branch_component("...", fallback="batch") == "batch"


def test_unsafe_ref_characters_are_replaced() -> None:
    assert sanitize_branch_component("unsafe@{name}") == "unsafe_name"
    assert sanitize_branch_component("a b:c?d*e[f") == "a_b_c_d_e_f"
    assert sanitize_branch_component("abc..def@@@ghi---jkl") == "abc_def_ghi_jkl"


def test_lock_suffix_is_stripped() -> None:
    assert sanitize_branch_component("foo.lock") == "foo"
    assert sanitize_branch_component("only.lock", fallback="batch") == "only"


def test_non_ascii_language_identifiers_stay_git_safe() -> None:
    # Identifiers from non-English task names must still yield a valid ref.
    safe = sanitize_branch_component("模块_state.go", fallback="batch")

    assert safe
    assert ".." not in safe
    assert not safe.endswith(".")
    assert "/" not in safe


def test_result_is_idempotent() -> None:
    once = sanitize_branch_component("Some Mixed/Name..value.lock")
    twice = sanitize_branch_component(once)

    assert once == twice


def test_all_branch_prefixes_consume_the_shared_sanitizer() -> None:
    # Guard against a future call site re-introducing ad-hoc truncation.
    from code_gen import git_ops
    from code_gen import subtree_review
    from common import git_utils

    for module in (git_ops, subtree_review, git_utils):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "sanitize_branch_component" in source
