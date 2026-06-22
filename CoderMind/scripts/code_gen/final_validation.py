#!/usr/bin/env python3
"""Final test stage of the codegen pipeline.

This module hosts :func:`final_test`, extracted from
``scripts/run_batch.py`` Module 6 ("Final Test").

After all per-task batches complete, the orchestrator runs a single
full-suite pytest pass against the merged code on ``main``.  When
pytest passes, we also run the smoke test (import sweep + entry-point
check + stub detection); if the smoke test reports actionable findings,
a repair sub-agent is dispatched and the full pytest is re-run.

The stage's outcome is persisted to ``.cmind/logs/codegen_final_test.json``
(and ``codegen_smoke_test.json``) via
:mod:`scripts.code_gen.stage_io` so that the global-review stage can
consume the results without re-running pytest.

Internal to the codegen package; no external API contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from common.git_utils import GitRunner
from common.paths import CODE_GEN_STATE_FILE as STATE_FILE, REPO_DIR
from code_gen.batch_prompts import _build_backend_test_cmd
from code_gen.git_ops import ensure_on_main
from code_gen.stage_io import save_stage_result
from code_gen.sub_agent import dispatch_sub_agent
from code_gen.test_runner import (
    ensure_deps_installed,
    get_dev_python,
    resolve_test_backend,
    run_project_tests,
)

logger = logging.getLogger(__name__)


from code_gen._constants import (  # noqa: E402
    DEFAULT_PYTEST_OVERALL_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
)


def _fail_final_test_for_smoke_error(
    result_dict: Dict[str, Any],
    message: str,
    *,
    smoke_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark final validation failed because smoke validation failed."""
    result_dict["success"] = False
    result_dict["errors"] = max(int(result_dict.get("errors", 0) or 0), 1)
    result_dict["output"] = message
    result_dict["next_action"] = (
        "Unit tests passed, but smoke validation failed. Fix the smoke "
        "failure and re-run final validation."
    )
    result_dict["smoke_test_error"] = message
    if smoke_dict is None:
        smoke_dict = {
            "success": False,
            "type": "smoke_test",
            "findings": [{"severity": "error", "message": message}],
            "error_count": 1,
            "warning_count": 0,
        }
    result_dict["smoke_test"] = smoke_dict


def final_test(
    repo_path: Optional[Path] = None,
    state_path: Path = STATE_FILE,
    max_repair_iters: int = 2,
) -> Dict[str, Any]:
    """Run the full test suite against the completed repo.

    Args:
        repo_path: Project repo path.
        state_path: Path to state file.
        max_repair_iters: Bound on repair sub-agent passes when the full
            suite fails. Cross-batch inconsistencies (e.g. a test asserting the
            README documents a symbol another batch produced) only surface here,
            where no per-batch TDD loop can catch them.

    Returns:
        Result dict with test statistics.
    """
    repo_path = repo_path or REPO_DIR
    git = GitRunner(str(repo_path))

    logger.info("━━━ Final Test: full repo validation ━━━")

    try:
        ensure_on_main(git)
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    backend = resolve_test_backend(repo_path=repo_path)
    if backend.name == "python":
        try:
            ensure_deps_installed(repo_path)
        except Exception as exc:
            logger.warning("Dependency install issue: %s", exc)

    # Run full test suite
    result = run_project_tests(
        repo_path,
        timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
        extra_args=[
            "-v", "--tb=short",
            f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread",
        ],
        backend=backend,
    )

    # Guard against a no-op "pass": a verification gate that executed zero
    # tests is not a pass, it is a non-result (e.g. ``go test ./...`` matching
    # no packages, or the runner invoked before sources were in the tree).
    # The backend already reports this as a non-success "errored" status; here
    # we make the final gate fail loudly with a precise diagnostic instead of
    # dispatching a code-repair agent that cannot fix a "no tests ran" state.
    executed = result.passed + result.failed + result.errors + result.skipped
    if not result.success and executed == 0:
        # A toolchain/infra failure (missing tool, timeout, crash →
        # return_code -1) is a different non-result than a command that ran
        # cleanly (exit 0) yet collected zero tests. Neither is a pass and
        # neither is fixable by a code-repair agent, but they need different
        # diagnostics, so report them distinctly.
        toolchain_failure = result.return_code != 0
        if toolchain_failure:
            next_action = (
                f"Final test could not run the {backend.display_name} test "
                "command (toolchain unavailable, timeout, or crash). Install or "
                "repair the language toolchain and re-run — this is an "
                "environment problem, not a code defect."
            )
        else:
            next_action = (
                f"Final test ran the {backend.display_name} test command but "
                "no tests executed (zero collected). This is a verification "
                "no-op, not a pass: confirm the generated test suite is present "
                "on the main branch and the test command discovers it."
            )
        logger.error(
            "Final test executed zero tests for %s backend (return_code=%s) — "
            "treating as a verification failure, not a pass.",
            backend.name, result.return_code,
        )
        no_test_result = {
            "success": False,
            "type": "final_test",
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration": result.duration,
            "output": result.output[:5000],
            "no_tests_executed": not toolchain_failure,
            "toolchain_unavailable": toolchain_failure,
            "next_action": next_action,
        }
        save_stage_result("final_test", {
            "success": False,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "no_tests_executed": not toolchain_failure,
            "toolchain_unavailable": toolchain_failure,
            "output_tail": "\n".join(result.output.splitlines()[-40:]),
        })
        return no_test_result

    # Repair loop for full-suite failures. The per-batch TDD loop only sees one
    # file's tests at a time, so cross-file consistency gaps (a test asserting
    # the README / an example module documents a specific symbol or section that
    # a different batch generated independently) survive to here. Dispatch a
    # bounded repair pass that reconciles the repo against the EXISTING tests
    # rather than letting one such gap fail the whole stage with no recovery.
    repair_attempts = 0
    while not result.success and repair_attempts < max_repair_iters:
        repair_attempts += 1
        venv_python = get_dev_python(repo_path) or "python3"
        repair_verify_cmd = _build_backend_test_cmd(
            backend, repo_path, [], venv_python,
        )
        failure_tail = "\n".join(result.output.splitlines()[-80:])
        repair_prompt = (
            "The full test suite failed after every batch completed. Reconcile "
            "the repository so the EXISTING tests pass. These failures are "
            "usually cross-file consistency gaps — for example a test asserts "
            "that the README or an example module documents a specific symbol "
            "or section, but a different batch generated those files "
            "independently.\n\n"
            f"Failing test output (tail):\n{failure_tail}\n\n"
            "Rules:\n"
            "- Fix production code, documentation, or example files so the "
            "existing tests pass. Do NOT delete, skip, or weaken any test.\n"
            "- Do NOT create new test files.\n\n"
            f"Verify with:\n```\n{repair_verify_cmd}\n```\n\n"
            "When the suite is green, commit:\n"
            "```\ngit add -A && git commit -m "
            '"fix: reconcile final test failures"\n```\n'
            "Then output: BATCH_RESULT: PASS"
        )
        logger.info(
            "Final test failed; dispatching repair agent (attempt %d/%d)",
            repair_attempts, max_repair_iters,
        )
        response, error = dispatch_sub_agent(
            repair_prompt, repo_path, timeout=1800,
            purpose="final_test_repair",
        )
        if not response:
            logger.warning("Final-test repair agent failed: %s", error)
            break
        ensure_on_main(git)
        result = run_project_tests(
            repo_path,
            timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
            extra_args=[
                "-v", "--tb=short",
                f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread",
            ],
            backend=backend,
        )

    result_dict = {
        "success": result.success,
        "type": "final_test",
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "skipped": result.skipped,
        "duration": result.duration,
        "output": result.output[:5000] if not result.success else "",
        "next_action": (
            "All tests passed! The repository is ready."
            if result.success else
            f"Final test failed ({result.failed} failures, {result.errors} errors). "
            f"Review the output above and fix remaining issues."
        ),
    }
    if repair_attempts:
        result_dict["final_test_repair_attempts"] = repair_attempts
        result_dict["final_test_repaired"] = result.success

    # After pytest passes, run smoke test and attempt repair if issues found
    if result.success:
        try:
            # Lazy import: smoke_test pulls in the dep_graph stack, so only
            # load it on the success path where we actually need it.
            from smoke_test import run_smoke_test

            smoke_result = run_smoke_test()
            smoke_dict = smoke_result.to_dict()
            result_dict["smoke_test"] = smoke_dict

            # Collect actionable findings (errors)
            actionable = [f for f in smoke_result.findings if f.severity == "error"]

            if actionable:
                remaining = actionable
                recheck_success = True
                findings_desc = "\n".join(
                    f"- [{f.severity}] {f.message}" for f in actionable
                )
                # Build the language-appropriate verify command for the agent
                venv_python = get_dev_python(repo_path) or "python3"
                repair_verify_cmd = _build_backend_test_cmd(
                    backend, repo_path, [], venv_python,
                )
                repair_prompt = (
                    "The smoke test detected the following issues after all "
                    "unit tests passed. Fix each issue in the production code, "
                    "then run the test suite to verify nothing is broken.\n\n"
                    f"Findings:\n{findings_desc}\n\n"
                    "Common fixes:\n"
                    "- STUB (pass only) → implement the function body\n"
                    "- PLACEHOLDER return → replace with real logic\n"
                    "- Import error → add missing import\n"
                    "- Startup crash → fix initialization code\n\n"
                    "Do NOT create new test files. Only fix production code.\n"
                    "After fixing, run this command to verify:\n"
                    f"```\n{repair_verify_cmd}\n```\n\n"
                    "When done, commit your changes:\n"
                    "```\ngit add -A && git commit -m "
                    '"fix: repair smoke test findings"\n```\n'
                    "Then output: BATCH_RESULT: PASS"
                )
                logger.info(
                    "Smoke test found %d actionable issues, dispatching "
                    "repair agent", len(actionable)
                )
                response, error = dispatch_sub_agent(
                    repair_prompt, repo_path, timeout=1800,
                    purpose="smoke_repair",
                )
                if response:
                    # Verify repair didn't break existing tests
                    recheck = run_project_tests(
                        repo_path,
                        timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
                        extra_args=[
                            "-v", "--tb=short",
                            f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread",
                        ],
                        backend=backend,
                    )
                    if not recheck.success:
                        logger.warning(
                            "Repair agent broke %d tests, results may be degraded",
                            recheck.failed + recheck.errors,
                        )
                    # Re-run smoke test to verify repairs
                    smoke_result_2 = run_smoke_test()
                    result_dict["smoke_test"] = smoke_result_2.to_dict()
                    result_dict["smoke_repair_attempted"] = True
                    result_dict["post_repair_tests_pass"] = recheck.success
                    recheck_success = recheck.success
                    remaining = [
                        f for f in smoke_result_2.findings
                        if f.severity == "error"
                    ]
                    logger.info(
                        "Post-repair: smoke=%d issues remaining (was %d), "
                        "pytest=%s",
                        len(remaining), len(actionable),
                        "PASS" if recheck.success else "FAIL",
                    )
                if remaining or not recheck_success:
                    smoke_dict = result_dict.get("smoke_test")
                    if not isinstance(smoke_dict, dict):
                        smoke_dict = {}
                    message = (
                        "Smoke validation failed after unit tests passed. "
                        f"Remaining smoke errors: {len(remaining)}; "
                        f"post-repair tests pass: {recheck_success}."
                    )
                    _fail_final_test_for_smoke_error(
                        result_dict,
                        message,
                        smoke_dict=smoke_dict,
                    )
        except ImportError:
            logger.debug("smoke_test module not available, skipping")
        except Exception as exc:
            logger.warning("Smoke test / repair failed: %s", exc)
            _fail_final_test_for_smoke_error(
                result_dict,
                f"Smoke test failed to run: {exc}",
            )

    # Save per-stage results for global_review context
    save_stage_result("final_test", {
        "success": bool(result_dict.get("success")),
        "passed": result_dict.get("passed", result.passed),
        "failed": result_dict.get("failed", result.failed),
        "errors": result_dict.get("errors", result.errors),
        "output_tail": (
            "\n".join(str(result_dict.get("output", "")).splitlines()[-40:])
            if not result_dict.get("success") else ""
        ),
    })
    smoke_data = result_dict.get("smoke_test")
    if isinstance(smoke_data, dict):
        smoke_save: Dict[str, Any] = {
            "findings": smoke_data.get("findings", []),
            "error_count": smoke_data.get("error_count", 0),
        }
        if result_dict.get("smoke_repair_attempted"):
            smoke_save["repair_attempted"] = True
            remaining = [
                f for f in smoke_data.get("findings", [])
                if f.get("severity") == "error"
            ]
            smoke_save["repair_remaining"] = len(remaining)
        save_stage_result("smoke_test", smoke_save)

    return result_dict
