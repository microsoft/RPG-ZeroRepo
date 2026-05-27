#!/usr/bin/env python3
"""Final test stage of the codegen pipeline.

This module hosts :func:`final_test`, extracted from
``scripts/run_batch.py`` Module 6 ("Final Test").

After all per-task batches complete, the orchestrator runs a single
full-suite pytest pass against the merged code on ``main``.  When
pytest passes, we also run the smoke test (import sweep + entry-point
check + stub detection); if the smoke test reports actionable findings,
a repair sub-agent is dispatched and the full pytest is re-run.

The stage's outcome is persisted to ``.rpgkit/logs/codegen_final_test.json``
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
from code_gen.git_ops import ensure_on_main
from code_gen.stage_io import save_stage_result
from code_gen.sub_agent import dispatch_sub_agent
from code_gen.test_runner import (
    ensure_deps_installed,
    get_dev_python,
    run_pytest,
)

logger = logging.getLogger(__name__)


from code_gen._constants import (  # noqa: E402
    DEFAULT_PYTEST_OVERALL_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
)


def final_test(
    repo_path: Optional[Path] = None,
    state_path: Path = STATE_FILE,
) -> Dict[str, Any]:
    """Run the full test suite against the completed repo.

    Args:
        repo_path: Project repo path.
        state_path: Path to state file.

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

    # Ensure all deps
    try:
        ensure_deps_installed(repo_path)
    except Exception as exc:
        logger.warning("Dependency install issue: %s", exc)

    # Run full test suite
    result = run_pytest(
        repo_path,
        timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
        extra_args=[
            "-v", "--tb=short",
            f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread",
        ],
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

    # After pytest passes, run smoke test and attempt repair if issues found
    if result.success:
        try:
            # Lazy import: smoke_test pulls in the dep_graph stack, so only
            # load it on the success path where we actually need it.
            from smoke_test import run_smoke_test
            from code_gen.batch_prompts import build_batch_pytest_cmd

            smoke_result = run_smoke_test()
            smoke_dict = smoke_result.to_dict()
            result_dict["smoke_test"] = smoke_dict

            # Collect actionable findings (errors)
            actionable = [f for f in smoke_result.findings if f.severity == "error"]

            if actionable:
                findings_desc = "\n".join(
                    f"- [{f.severity}] {f.message}" for f in actionable
                )
                # Build pytest command for the repair agent
                venv_python = get_dev_python(repo_path) or "python3"
                repair_pytest_cmd = build_batch_pytest_cmd([], venv_python)
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
                    f"```\n{repair_pytest_cmd}\n```\n\n"
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
                    recheck = run_pytest(
                        repo_path,
                        timeout=DEFAULT_PYTEST_OVERALL_TIMEOUT,
                        extra_args=[
                            "-v", "--tb=short",
                            f"--timeout={DEFAULT_TEST_TIMEOUT}", "--timeout-method=thread",
                        ],
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
        except ImportError:
            logger.debug("smoke_test module not available, skipping")
        except Exception as exc:
            logger.warning("Smoke test / repair failed: %s", exc)

    # Save per-stage results for global_review context
    save_stage_result("final_test", {
        "success": result.success,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "output_tail": "\n".join(result.output.splitlines()[-40:]) if not result.success else "",
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
