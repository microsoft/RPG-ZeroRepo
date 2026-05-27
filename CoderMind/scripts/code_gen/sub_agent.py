#!/usr/bin/env python3
"""Sub-agent dispatch helpers used by the codegen pipeline.

This module hosts the four helpers extracted from
``scripts/run_batch.py`` Module 3 ("Sub-agent Dispatch"):

* :func:`dispatch_sub_agent` — launch an LLM sub-agent on a prompt.
* :func:`parse_batch_result` — read the agent's ``BATCH_RESULT:`` marker.
* :func:`parse_pytest_summary` — extract the agent's reported pytest summary line.
* :func:`truncate_test_output` — trim long pytest output for retry prompts.

These helpers are shared across the codegen orchestrator
(``scripts.run_batch``), the post-codegen subtree reviewer
(``scripts.code_gen.subtree_review``), and the RPG-edit pipeline
(``scripts.rpg_edit.review`` / ``scripts.rpg_edit.code``).

``scripts.run_batch`` re-exports these names so the legacy
``from run_batch import dispatch_sub_agent`` imports keep working;
new code should prefer ``from code_gen.sub_agent import …``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from common.llm_client import LLMClient

logger = logging.getLogger(__name__)


from code_gen._constants import DEFAULT_AGENT_TIMEOUT  # noqa: E402


def dispatch_sub_agent(
    prompt: str,
    repo_path: Path,
    timeout: int = DEFAULT_AGENT_TIMEOUT,
    trajectory=None,
    step_id=None,
    purpose: str = "run_batch",
    max_retries: int = 1,
) -> Tuple[Optional[str], Optional[str]]:
    """Dispatch a sub-agent with the given prompt.

    Args:
        prompt: Full prompt string.
        repo_path: Project repo path.
        timeout: Max time for the sub-agent session.
        trajectory: Trajectory instance for recording.
        step_id: Current step ID in trajectory.
        purpose: Purpose string for trajectory/logging.
        max_retries: Number of LLM call attempts (1 = no retry).

    Returns:
        (response_text, error_message) — one of them is None.
    """
    client = LLMClient(trajectory=trajectory, step_id=step_id)
    logger.info(
        "Dispatching sub-agent (purpose=%s, timeout=%ds, prompt_len=%d)",
        purpose, timeout, len(prompt),
    )
    logger.debug("Sub-agent prompt:\n%s", prompt)

    start_time = time.time()
    try:
        response = client.generate(
            prompt,
            purpose=purpose,
            timeout=timeout,
            max_retries=max_retries,
        )
        elapsed = time.time() - start_time
        logger.info("Sub-agent completed in %.1fs (response_len=%d)", elapsed, len(response))
        logger.debug("Sub-agent response:\n%s", response)
        return response, None
    except RuntimeError as exc:
        elapsed = time.time() - start_time
        error_msg = f"Sub-agent failed after {elapsed:.1f}s: {exc}"
        logger.error(error_msg)
        return None, error_msg


def parse_batch_result(response: Optional[str]) -> Tuple[bool, str]:
    """Parse the sub-agent's exit status from its response.

    Looks for ``BATCH_RESULT: PASS`` or ``BATCH_RESULT: FAIL | <reason>``
    in the last 20 lines of the response.

    Args:
        response: Sub-agent response text.

    Returns:
        ``(passed, reason)`` — ``passed`` is ``True`` if ``PASS`` found.
    """
    if not response:
        return False, "No response from sub-agent"

    # Search last 20 lines for the result marker
    lines = response.strip().splitlines()
    search_lines = lines[-20:] if len(lines) > 20 else lines

    for line in reversed(search_lines):
        line = line.strip()
        if line.startswith("BATCH_RESULT: PASS"):
            return True, "Sub-agent reported PASS"
        if line.startswith("BATCH_RESULT: FAIL"):
            reason = line.split("|", 1)[1].strip() if "|" in line else "Unknown failure"
            return False, reason

    # No explicit marker found — treat as failure
    return False, "Sub-agent did not output BATCH_RESULT marker"


def parse_pytest_summary(response: Optional[str]) -> Optional[str]:
    """Extract the sub-agent's claimed pytest summary line, if present.

    The runner's TDD prompt asks the sub-agent to copy the literal pytest
    summary line into a ``PYTEST_SUMMARY: …`` line right above
    ``BATCH_RESULT``.  This helper returns that quoted text (everything
    after the first colon, stripped) so the orchestrator can cross-check
    it against the post-verify rerun.

    Returns ``None`` if the sub-agent did not provide the line.
    """
    if not response:
        return None
    lines = response.strip().splitlines()
    search_lines = lines[-20:] if len(lines) > 20 else lines
    for line in reversed(search_lines):
        stripped = line.strip()
        if stripped.startswith("PYTEST_SUMMARY:"):
            return stripped.split(":", 1)[1].strip()
    return None


def truncate_test_output(text: str, head: int = 20, tail: int = 50) -> str:
    """Trim a long pytest output for safe injection into a retry prompt.

    Keeps the first ``head`` lines (typically: pytest header, collected
    test count, first failure summary) **and** the last ``tail`` lines
    (where pytest places the FAILED/ERROR detail and the summary line),
    inserting ``... <N lines truncated> ...`` between them.

    Returns ``text`` unchanged when it is already shorter than
    ``head + tail + 1`` lines.
    """
    if not text:
        return text
    lines = text.splitlines()
    keep = head + tail
    if len(lines) <= keep + 1:
        return text
    omitted = len(lines) - keep
    body = (
        lines[:head]
        + [f"... <{omitted} lines truncated> ..."]
        + lines[-tail:]
    )
    return "\n".join(body)
