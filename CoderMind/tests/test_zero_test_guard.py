"""Regression tests for the zero-test "no-op pass" guard.

A verification gate that executed zero tests is not a pass — it is a
non-result. Before this guard every non-Python backend reported
``status = "passed"`` whenever the test command exited 0, so a no-op run
(e.g. ``go test ./...`` matching no packages, or a runner invoked before
the sources were in the tree) silently satisfied the final gate. These
tests lock in that an exit-0 run with no executed tests is reported as
``errored`` (non-success), while real passes and real failures are
unaffected, across every language backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from decoder_lang import get_backend  # noqa: E402
from decoder_lang.test_result import ran_no_tests  # noqa: E402


class TestRanNoTestsHelper:
    def test_nonzero_exit_is_never_a_no_op(self):
        # A non-zero exit is already a failure; the no-op concept does not apply.
        assert ran_no_tests(1, "") is False
        assert ran_no_tests(2, "boom") is False

    def test_empty_output_exit0_is_no_op(self):
        assert ran_no_tests(0, "") is True
        assert ran_no_tests(0, "   \n\t ") is True

    def test_marker_phrase_is_no_op(self):
        assert ran_no_tests(
            0, "ctest: No tests were found!!!",
            no_tests_markers=("No tests were found",),
        ) is True

    def test_reliable_zero_count_is_no_op(self):
        assert ran_no_tests(0, "some banner output", observed_tests=0) is True

    def test_positive_count_is_not_a_no_op(self):
        assert ran_no_tests(0, "anything", observed_tests=5) is False

    def test_nonempty_unknown_output_is_not_a_no_op(self):
        # Fail-safe: unrecognized but non-empty output (no count, no marker)
        # must be treated as a real run, never a false failure.
        assert ran_no_tests(0, "weird tool output") is False

    def test_empty_output_opt_out_for_compile_check(self):
        # C / C++ fall back to a clean ``-fsyntax-only`` compile that emits
        # no output; that is a legitimate pass, not a no-op.
        assert ran_no_tests(0, "", empty_output_is_no_op=False) is False


class TestGoVerdict:
    def setup_method(self):
        self.backend = get_backend("go")

    def test_empty_output_exit0_is_errored(self):
        # The exact bench failure: go test matched no packages → no-op.
        result = self.backend.parse_test_output("", 0)
        assert result.status == "errored"

    def test_real_pass_with_events(self):
        raw = "=== RUN   TestAdd\n--- PASS: TestAdd (0.00s)\nok  \tpkg\t0.01s\n"
        result = self.backend.parse_test_output(raw, 0)
        assert result.status == "passed"
        assert result.passed_count == 1

    def test_nonempty_output_without_parsed_counts_still_passes(self):
        # ``-json`` output the text regexes don't parse → 0 counts, but the
        # non-empty stream proves a run happened: must not false-fail.
        raw = '{"Action":"pass","Package":"pkg","Test":"TestAdd"}\n'
        result = self.backend.parse_test_output(raw, 0)
        assert result.status == "passed"

    def test_real_failure(self):
        raw = "=== RUN   TestAdd\n--- FAIL: TestAdd (0.00s)\nFAIL\tpkg\t0.01s\n"
        result = self.backend.parse_test_output(raw, 1)
        assert result.status == "failed"

    def test_test_command_requests_verbose_output(self):
        # ``-v`` is what makes go emit the per-test lines parse_test_output
        # counts; without it a real run reports passed_count 0 and looks like
        # a no-op. Lock the flag into the command.
        from decoder_lang.test_result import EnvHandle

        cmd = self.backend.test_command(EnvHandle(project_root=Path("/tmp/x")))
        assert "-v" in cmd
        assert cmd[-1] == "./..."


class TestNodeBackendsVerdict:
    @pytest.mark.parametrize("language", ["javascript", "typescript"])
    def test_empty_output_exit0_is_errored(self, language):
        result = get_backend(language).parse_test_output("", 0)
        assert result.status == "errored"

    @pytest.mark.parametrize("language", ["javascript", "typescript"])
    def test_real_pass_reports_counts(self, language):
        raw = "# tests 74\n# pass 74\n# fail 0\n"
        result = get_backend(language).parse_test_output(raw, 0)
        assert result.status == "passed"
        assert result.passed_count == 74

    @pytest.mark.parametrize("language", ["javascript", "typescript"])
    def test_zero_tests_summary_is_errored(self, language):
        result = get_backend(language).parse_test_output("# tests 0\n# pass 0\n", 0)
        assert result.status == "errored"

    @pytest.mark.parametrize("language", ["javascript", "typescript"])
    def test_real_failure(self, language):
        raw = "# tests 5\n# pass 4\n# fail 1\n"
        result = get_backend(language).parse_test_output(raw, 1)
        assert result.status == "failed"
        assert result.failed_count == 1


class TestRustVerdict:
    def setup_method(self):
        self.backend = get_backend("rust")

    def test_empty_output_exit0_is_errored(self):
        assert self.backend.parse_test_output("", 0).status == "errored"

    def test_real_pass_sums_counts(self):
        raw = (
            "test result: ok. 5 passed; 0 failed; 0 ignored\n"
            "test result: ok. 3 passed; 0 failed; 1 ignored\n"
        )
        result = self.backend.parse_test_output(raw, 0)
        assert result.status == "passed"
        assert result.passed_count == 8

    def test_zero_tests_result_is_errored(self):
        raw = "test result: ok. 0 passed; 0 failed; 0 ignored\n"
        assert self.backend.parse_test_output(raw, 0).status == "errored"


class TestCompiledBackendsVerdict:
    """C / C++ fall back to a compile check, so empty output is a real pass."""

    @pytest.mark.parametrize("language", ["c", "cpp"])
    def test_empty_output_is_pass_not_no_op(self, language):
        # A clean ``-fsyntax-only`` compile emits nothing and exits 0.
        result = get_backend(language).parse_test_output("", 0)
        assert result.status == "passed"

    @pytest.mark.parametrize("language", ["c", "cpp"])
    def test_ctest_no_tests_marker_is_errored(self, language):
        raw = "Test project /tmp/build\nNo tests were found!!!\n"
        assert get_backend(language).parse_test_output(raw, 0).status == "errored"

    @pytest.mark.parametrize("language", ["c", "cpp"])
    def test_ctest_real_pass(self, language):
        raw = "100% tests passed, 0 tests failed out of 19\n"
        assert get_backend(language).parse_test_output(raw, 0).status == "passed"

    @pytest.mark.parametrize("language", ["c", "cpp"])
    def test_compile_failure_is_failed(self, language):
        raw = "error: expected ';' before '}' token\n"
        assert get_backend(language).parse_test_output(raw, 1).status == "failed"
