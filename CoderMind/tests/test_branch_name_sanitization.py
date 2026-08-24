from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from code_gen import git_ops, subtree_review  # noqa: E402
from code_gen.git_ops import setup_batch_branch  # noqa: E402
from common import git_utils  # noqa: E402
from common.git_utils import GitRunner, sanitize_branch_component  # noqa: E402


def test_trailing_dot_after_truncation_is_removed() -> None:
    # The exact id that produced an invalid 'batch/...state.' ref for C++.
    batch_id = "src_expression_calculator_syntax_expression_state.cpp_20260613_082726_e88325bc"

    safe = sanitize_branch_component(batch_id, max_len=50, fallback="batch")

    assert safe.startswith("src_expression_calculator_syntax_")
    assert safe[-9] == "-"
    assert all(char in "0123456789abcdef" for char in safe[-8:])
    assert len(safe) == 50
    assert not safe.endswith(".")


def test_long_batch_ids_with_shared_prefix_remain_unique() -> None:
    first = "src_tasklite_cli_use_cases_manage_tasks.py_20260822_101218_97be5e0e"
    second = "src_tasklite_cli_use_cases_manage_tasks.py_20260822_101218_58a347c9"

    first_safe = sanitize_branch_component(first, max_len=50, fallback="batch")
    second_safe = sanitize_branch_component(second, max_len=50, fallback="batch")

    assert first_safe != second_safe
    assert len(first_safe) <= 50
    assert len(second_safe) <= 50


def test_retry_preserves_failed_branch_and_creates_fresh_recovery_branch(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = GitRunner(str(repo))
    git.run_git(["config", "user.name", "test"])
    git.run_git(["config", "user.email", "test@example.com"])
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git.stage_and_commit("initial")
    original_head = git.get_head_commit()
    batch_id = "docs_policies_runtime.py_20260822_100942_5f60fe86"

    created, failed_branch, _ = setup_batch_branch(git, batch_id, repo)
    assert created
    (repo / "README.md").write_text("failed branch\n", encoding="utf-8")
    git.stage_and_commit("failed attempt")
    assert git.switch_branch("main")

    recovered, recovery_branch, recovery_head = setup_batch_branch(
        git,
        batch_id,
        repo,
        preserve_existing=True,
    )

    assert recovered
    assert recovery_branch == f"{failed_branch}-retry-1"
    assert git.branch_exists(failed_branch)
    assert recovery_head == original_head
    assert git.get_current_branch() == recovery_branch


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
    for module in (git_ops, subtree_review, git_utils):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "sanitize_branch_component" in source
