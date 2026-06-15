"""Regression tests for language-agnostic entry-point reconciliation.

Locks in the fix for the dual-``main`` bug: the program entry was placed
by TWO uncoordinated deciders — the LLM-built skeleton (free to choose a
path) and the synthetic ``<MAIN_ENTRY>`` task (the backend's canonical
path). When they differed (C++ skeleton ``src/cli/main.cpp`` vs canonical
``src/main.cpp``), two ``main`` files were produced. Reconciliation was
only implemented for Go; these tests assert it now works for all 7
languages through the ``backend.find_existing_entry`` /
``entry_point_candidates`` protocol, with no hardcoded ``backend.name``
branch in the planner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from decoder_lang import get_backend  # noqa: E402
from decoder_lang.project_tasks import ProjectTaskContext  # noqa: E402

_LANGS = ["python", "javascript", "typescript", "go", "c", "cpp", "rust"]


def _interfaces_with(file_path: str) -> dict:
    return {
        "subtrees": {
            "Some Subtree": {
                "interfaces": {
                    file_path: {"units": ["function main"], "units_to_features": {}},
                },
            },
        },
    }


class TestProtocolPresence:
    @pytest.mark.parametrize("language", _LANGS)
    def test_all_backends_implement_new_methods(self, language):
        b = get_backend(language)
        assert callable(b.find_existing_entry)
        assert callable(b.entry_point_candidates)
        assert callable(b.prepare_test_env)


class TestDefaultFindExistingEntry:
    def test_reuses_off_canonical_entry_by_filename(self):
        # The cpp case: skeleton placed main.cpp at src/cli/main.cpp,
        # canonical is src/main.cpp. Filename match must reuse the former.
        cpp = get_backend("cpp")
        itf = _interfaces_with("src/cli/main.cpp")
        assert cpp.find_existing_entry(itf) == "src/cli/main.cpp"

    def test_returns_none_when_no_entry_in_skeleton(self):
        cpp = get_backend("cpp")
        itf = _interfaces_with("src/calc/evaluator.cpp")  # not an entry
        assert cpp.find_existing_entry(itf) is None

    def test_empty_interfaces_returns_none(self):
        assert get_backend("rust").find_existing_entry({}) is None
        assert get_backend("c").find_existing_entry({"subtrees": {}}) is None

    @pytest.mark.parametrize(
        ("language", "entry_path"),
        [
            ("python", "app/main.py"),
            ("javascript", "lib/index.js"),
            ("typescript", "lib/index.ts"),
            ("c", "src/cli/main.c"),
            ("rust", "bin/main.rs"),
        ],
    )
    def test_each_language_reuses_off_canonical_entry(self, language, entry_path):
        b = get_backend(language)
        itf = _interfaces_with(entry_path)
        assert b.find_existing_entry(itf) == entry_path


class TestGoEntryReconciliation:
    def test_reuses_existing_cmd_main(self):
        go = get_backend("go")
        itf = _interfaces_with("cmd/todoapp/main.go")
        assert go.find_existing_entry(itf) == "cmd/todoapp/main.go"

    def test_ignores_non_cmd_main_go(self):
        # A main.go NOT under cmd/<name>/ is not a Go command entry.
        go = get_backend("go")
        itf = _interfaces_with("internal/main.go")
        assert go.find_existing_entry(itf) is None

    def test_no_cmd_package_returns_none(self):
        go = get_backend("go")
        itf = _interfaces_with("internal/store/store.go")
        assert go.find_existing_entry(itf) is None


class TestEntryPointCandidates:
    def test_go_uses_glob(self):
        assert get_backend("go").entry_point_candidates() == ["cmd/*/main.go"]

    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("python", "main.py"),
            ("javascript", "src/index.js"),
            ("typescript", "src/index.ts"),
            ("c", "src/main.c"),
            ("cpp", "src/main.cpp"),
            ("rust", "src/main.rs"),
        ],
    )
    def test_fixed_path_languages(self, language, expected):
        assert get_backend(language).entry_point_candidates() == [expected]


class TestTemplatesConsumeReconciledEntry:
    @pytest.mark.parametrize(
        ("language", "off_canonical"),
        [
            ("javascript", "lib/index.js"),
            ("typescript", "lib/index.ts"),
            ("go", "cmd/todoapp/main.go"),
            ("c", "src/cli/main.c"),
            ("cpp", "src/cli/main.cpp"),
            ("rust", "bin/main.rs"),
        ],
    )
    def test_main_entry_template_uses_reconciled_path(self, language, off_canonical):
        # The template must reference the reconciled entry (not the
        # canonical hardcoded path) and forbid a second entry file.
        b = get_backend(language)
        ctx = ProjectTaskContext(
            repo_name="demo",
            repo_info="purpose",
            package_name="demo",
            entry_point_path=off_canonical,
        )
        templates = b.project_task_templates(ctx)
        assert templates is not None
        assert off_canonical in templates.main_entry
        assert "extend it in place" in templates.main_entry

    @pytest.mark.parametrize(
        ("language", "canonical"),
        [
            ("javascript", "src/index.js"),
            ("typescript", "src/index.ts"),
            ("c", "src/main.c"),
            ("cpp", "src/main.cpp"),
            ("rust", "src/main.rs"),
        ],
    )
    def test_main_entry_falls_back_to_canonical_when_none(self, language, canonical):
        b = get_backend(language)
        ctx = ProjectTaskContext(
            repo_name="demo",
            repo_info="purpose",
            package_name="demo",
            entry_point_path=None,
        )
        templates = b.project_task_templates(ctx)
        assert canonical in templates.main_entry


class TestPrepareTestEnvNoOp:
    @pytest.mark.parametrize("language", ["python", "javascript", "typescript", "go", "rust"])
    def test_no_op_for_non_compiled_cmake(self, language):
        # Must not raise even with a bogus env handle.
        get_backend(language).prepare_test_env(object())


class TestNoLanguageNameBranchInPlanner:
    """Guard: the planner's entry reconciliation must not re-introduce a
    per-language ``backend.name == "go"`` branch."""

    def test_reconciled_entry_point_path_has_no_go_branch(self):
        src = (_SCRIPTS / "plan_tasks.py").read_text(encoding="utf-8")
        # Locate the method body and assert it delegates to the backend.
        start = src.index("def _reconciled_entry_point_path")
        end = src.index("def _build_requirements_task", start)
        body = src[start:end]
        assert "find_existing_entry" in body
        assert 'backend.name == "go"' not in body

    def test_check_code_gen_entry_has_no_go_branch(self):
        src = (_SCRIPTS / "check_code_gen.py").read_text(encoding="utf-8")
        # The MAIN_ENTRY artifact check must use entry_point_candidates,
        # not a go-only glob branch.
        assert "entry_point_candidates" in src
        assert 'backend.name == "go"' not in src
