#!/usr/bin/env python3
"""Shared constants for the codegen pipeline.

These were duplicated across ``run_batch.py`` and the extracted
``code_gen.*`` modules (4 copies of ``DEFAULT_TEST_TIMEOUT``,
3 of ``DEFAULT_PYTEST_OVERALL_TIMEOUT``, 2 of ``DEFAULT_AGENT_TIMEOUT``).
Centralising them here makes timeout tuning a single-line change.

Why not ``common.paths``?  ``common.paths`` is shared across the
encoder, decoder, and agent layers; these timeout values are specific
to the codegen / TDD loop and don't belong in that broader namespace.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Sub-agent dispatch timeouts
# ---------------------------------------------------------------------------

# Maximum wall-clock time per sub-agent LLM session (seconds).
# 2700s = 45 minutes — enough for a full TDD iteration loop with retries.
# Overridable per-call via the ``timeout=`` keyword on
# :func:`code_gen.sub_agent.dispatch_sub_agent` and via the
# ``--agent-timeout`` CLI flag in ``run_batch.py``.
DEFAULT_AGENT_TIMEOUT = 2700


# ---------------------------------------------------------------------------
# pytest invocation timeouts
# ---------------------------------------------------------------------------

# Per-test-function timeout passed to ``pytest --timeout=`` (seconds).
# This is the real hang-prevention mechanism — any single test that
# blocks longer than this is killed.
DEFAULT_TEST_TIMEOUT = 30

# Overall pytest-invocation wall-clock budget (seconds).
# Acts as a safety net on top of ``DEFAULT_TEST_TIMEOUT``: kills a
# frozen pytest process even if individual test-level timeouts don't
# fire (rare, but possible on collection / fixture errors).
# 1800s = 30 minutes is generous even for 1000+ test suites.
DEFAULT_PYTEST_OVERALL_TIMEOUT = 1800
