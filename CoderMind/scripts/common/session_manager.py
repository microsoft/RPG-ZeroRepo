#!/usr/bin/env python3
"""Session Manager Module for CoderMind.

Provides a base class and CLI-specific subclasses for managing AI CLI
sessions: injecting tool-specific CLI arguments, preparing prompt
delivery (stdin vs command-line prompt), and capturing session traces
(e.g. JSONL logs) produced during LLM subprocess calls.

The primary interface is the ``trace(prompt, purpose)`` context manager:

    manager = create_session_manager("claude", project_dir=project_dir)
    with manager.trace(prompt, purpose="code_gen") as ctx:
        subprocess.run(cmd + ctx.extra_args, stdin=ctx.stdin, env=ctx.env)
    captured = ctx.captured_path   # Path | None

Each subclass encapsulates the convention of a specific CLI tool for
locating, snapshotting, and copying session files.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .paths import CLAUDE_LOGS_DIR, COPILOT_LOGS_DIR


logger = logging.getLogger(__name__)


# ============================================================================
# Trace context — the object yielded by the ``trace()`` context manager
# ============================================================================

class TraceContext:
    """Holds the result of a session-trace capture.

    Attributes:
        extra_args: Additional CLI arguments the manager wants injected
            into the subprocess command (e.g. ``["--session-id", "<uuid>"]``).
            Populated by ``before()``; the caller should append these to
            the command list.
        env: Complete environment dict for ``subprocess.run(env=...)``. 
            Initialised to ``os.environ.copy()`` by the base-class
            ``before()``; subclasses may modify it further.
        stdin: A file-like object (or *None*) to pass directly as the
            ``stdin`` parameter of ``subprocess.run()``.  When set by
            ``before()``, the caller should use it as-is; when *None*,
            the prompt was placed into ``extra_args`` instead and no
            stdin redirection is needed.
        captured_path: Destination path of the copied session file,
            or *None* if nothing was captured.
    """

    def __init__(self) -> None:
        self.extra_args: List[str] = []
        self.env: Dict[str, str] = os.environ.copy()
        self.stdin: Optional[Any] = None
        self.captured_path: Optional[Path] = None

    def reset_stdin(self) -> None:
        """Reset stdin read position for retry attempts."""
        if self.stdin is not None and hasattr(self.stdin, "seek"):
            self.stdin.seek(0)

    def refresh_for_retry(self) -> None:
        """Refresh context for a retry attempt.

        Resets stdin and calls the optional ``_refresh_hook`` set by the
        session manager (e.g., to regenerate a session ID that the CLI
        tool requires to be unique per invocation).
        """
        self.reset_stdin()
        if hasattr(self, "_refresh_hook") and self._refresh_hook:
            self._refresh_hook(self)


# ============================================================================
# Abstract base class
# ============================================================================

class SessionManager(ABC):
    """Base class for CLI session managers.

    Subclasses must implement two hooks:

    * ``before(ctx, prompt)`` — called **before** the LLM subprocess call.  The
        subclass can inject CLI arguments, prepare prompt delivery, snapshot
      whatever state it needs (e.g. directory listings, timestamps,
      etc.).
    * ``after(purpose)`` — called **after** the LLM subprocess call.
      The subclass compares the current state against whatever was
      saved in ``before()``, performs any copying / archiving, and
      returns the destination ``Path`` (or ``None``).

    The ``trace(prompt, purpose)`` context manager wires those two
    hooks together so that callers get a clean ``with`` block.
    """

    # Default destination for captured traces.  May be relative
    # (interpreted under ``project_dir`` by :meth:`_dest_dir`) or
    # absolute (used as-is — see e.g. :class:`ClaudeSessionManager`).
    DEFAULT_TRAJECTORY_DIR: Path = Path("trajectory")

    def __init__(
        self,
        project_dir: Path,
        trace_filename_builder: Optional[Callable[[str], str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Args: project_dir: Absolute path to the project root.

        trace_filename_builder: Optional callable ``(purpose) -> filename``
            used to name the copied file.  If *None*, a default builder
            based on purpose + timestamp is used.
        logger: Logger instance.
        """
        self.project_dir = project_dir
        self.trajectory_dir = self.DEFAULT_TRAJECTORY_DIR
        self._build_filename = trace_filename_builder or self._default_filename_builder
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def before(self, ctx: TraceContext, prompt: str) -> None:
        """Prepare / snapshot state before the LLM call.

        The subclass may populate ``ctx.extra_args`` with CLI flags
        that need to be injected into the subprocess command, or
        set ``ctx.stdin`` to the prompt content for stdin-based input.

        Args:
            ctx: The trace context to populate.
            prompt: The prompt text to send to the LLM.  Subclasses
                decide whether to add it to ``ctx.extra_args`` or
                ``ctx.stdin`` based on the CLI tool's conventions.
        """
        ...

    @abstractmethod
    def after(self, purpose: str) -> Optional[Path]:
        """Capture session trace after the LLM call.

        Returns:
            Destination path of the captured trace, or *None*.
        """
        ...

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def trace(self, prompt: str, purpose: str = "general") -> Iterator[TraceContext]:
        """Context manager that brackets an LLM call with session tracing.

        Args:
            purpose: A short label describing this LLM call.
            prompt: The prompt text to send to the LLM. Passed to
                ``before()`` so that the subclass can decide how to
                deliver it (via ``extra_args`` or ``stdin``).

        Usage::

            with manager.trace(prompt=my_prompt, purpose="code_gen") as ctx:
                cmd = ["cli"] + ctx.extra_args
                subprocess.run(cmd, stdin=ctx.stdin, env=ctx.env)
            print(ctx.captured_path)
        """
        ctx = TraceContext()
        self.before(ctx, prompt)
        try:
            yield ctx
        finally:
            ctx.captured_path = self.after(purpose)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_filename_builder(purpose: str) -> str:
        """Build ``<purpose>-<YYYYMMDD-HHmmss>.jsonl``."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^\w\-]", "_", purpose) if purpose else "llm_call"
        return f"{safe}-{ts}.jsonl"

    def _dest_dir(self) -> Path:
        """Return (and create) the trajectory destination directory.

        ``trajectory_dir`` may be either:
          * relative — interpreted under ``project_dir`` (legacy default);
          * absolute — used as-is (e.g. ``CLAUDE_LOGS_DIR`` anchored at
            the workspace root via ``common.paths``).
        """
        if self.trajectory_dir.is_absolute():
            d = self.trajectory_dir
        else:
            d = self.project_dir / self.trajectory_dir
        d.mkdir(parents=True, exist_ok=True)
        return d


# ============================================================================
# Null manager (no-op, used when no CLI-specific handling is needed)
# ============================================================================

class NullSessionManager(SessionManager):
    """A no-op manager that never captures anything.

    Used when the CLI tool does not produce session files or when
    session management is not desired.
    """

    def before(self, ctx: TraceContext, prompt: str) -> None:
        pass

    def after(self, purpose: str) -> Optional[Path]:
        return None


# ============================================================================
# Claude CLI manager
# ============================================================================

class ClaudeSessionManager(SessionManager):
    """Captures Claude CLI session JSONL files.

    Uses ``--session-id <uuid>`` to deterministically locate the session
    log file after the subprocess completes, instead of scanning for
    new files.

    The Claude CLI writes per-session JSONL logs under::

        ~/.claude/projects/<encoded-project-path>/<session-id>.jsonl

    where ``<encoded-project-path>`` replaces ``/`` and ``_`` with ``-``.
    """

    # Captured traces live under ``.cmind/logs/claude/`` so all
    # CoderMind-managed artefacts stay inside ``.cmind/`` (single ignore
    # rule, single cleanup target).  ``CLAUDE_LOGS_DIR`` is an absolute
    # path anchored at the workspace root (see ``common.paths``); the
    # base :meth:`_dest_dir` detects this and uses it as-is rather than
    # joining under ``project_dir``.
    DEFAULT_TRAJECTORY_DIR: Path = CLAUDE_LOGS_DIR

    def __init__(
        self,
        project_dir: Path,
        trace_filename_builder: Optional[Callable[[str], str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            project_dir=project_dir,
            trace_filename_builder=trace_filename_builder,
            logger=logger,
        )
        self._sessions_dir: Optional[Path] = self._get_projects_dir()
        # UUID generated by before(), consumed by after()
        self._session_id: Optional[str] = None
        # Temp file path / handle for prompt stdin; cleaned up in after()
        self._tmp_prompt_path: Optional[str] = None
        self._tmp_prompt_fh: Optional[Any] = None

    # ------------------------------------------------------------------
    # Claude-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_path(abs_path: str) -> str:
        """Encode an absolute path to the Claude project-directory name.

        Convention::

            /home/user/My_Project  ->  -home-user-My-Project
        """
        return abs_path.replace("/", "-").replace("_", "-")

    def _get_projects_dir(self) -> Optional[Path]:
        """Return ``~/.claude/projects/<encoded>/`` or *None*."""
        claude_base = Path.home() / ".claude" / "projects"
        encoded = self.encode_path(str(self.project_dir))
        candidate = claude_base / encoded
        return candidate if candidate.is_dir() else None

    # ------------------------------------------------------------------
    # SessionManager interface
    # ------------------------------------------------------------------

    def before(self, ctx: TraceContext, prompt: str) -> None:
        """Generate a UUID session-id and inject ``--session-id`` arg.

        Also removes the ``CLAUDECODE`` env var so that nested Claude
        Code sessions are allowed. The prompt is written to a temp
        file and exposed as ``ctx.stdin``.
        """
        self._session_id = str(uuid.uuid4())
        # --dangerously-skip-permissions: required for autonomous sub-agent
        # execution in the TDD workflow.  The sub-agent must read/write files
        # and run pytest without interactive permission prompts.  This flag
        # should ONLY be used in controlled, single-tenant environments.
        ctx.extra_args.extend([
            "-p", "--session-id", self._session_id,
            "--dangerously-skip-permissions",
        ])
        ctx.env.pop("CLAUDECODE", None)

        # Register a refresh hook so retries get a fresh session ID.
        # Claude CLI rejects a session-id that was already used.
        ctx._refresh_hook = self._regenerate_session_id

        # Clean up any leftover from a previous call, then write prompt
        # to a fresh temp file and open it for reading.
        self._cleanup_tmp_prompt()

        fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="llm_prompt_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(prompt)
            self._tmp_prompt_path = tmp_path
            self._tmp_prompt_fh = open(tmp_path, "r", encoding="utf-8")
        except Exception:
            # Best-effort removal on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            self._tmp_prompt_path = None
            self._tmp_prompt_fh = None
            raise

        ctx.stdin = self._tmp_prompt_fh

    def _regenerate_session_id(self, ctx: TraceContext) -> None:
        """Replace the session ID in extra_args with a fresh UUID.

        Called by ``TraceContext.refresh_for_retry()`` before each retry
        so that the Claude CLI doesn't reject a reused session ID.
        """
        new_id = str(uuid.uuid4())
        # Find and replace the old session-id value in extra_args
        args = ctx.extra_args
        for i, arg in enumerate(args):
            if arg == "--session-id" and i + 1 < len(args):
                args[i + 1] = new_id
                break
        self._session_id = new_id

    def _cleanup_tmp_prompt(self) -> None:
        """Close and remove the temporary prompt file, if any."""
        if self._tmp_prompt_fh is not None:
            try:
                self._tmp_prompt_fh.close()
            except Exception:
                pass
            self._tmp_prompt_fh = None
        if self._tmp_prompt_path is not None:
            try:
                os.unlink(self._tmp_prompt_path)
            except OSError:
                pass
            self._tmp_prompt_path = None

    def after(self, purpose: str) -> Optional[Path]:
        """Locate the JSONL by UUID and copy it to the trajectory directory."""
        self._cleanup_tmp_prompt()

        # Lazy-resolve: the projects dir may not exist at __init__ time
        # (created by the first claude call), so re-check here.
        if self._sessions_dir is None:
            self._sessions_dir = self._get_projects_dir()

        if self._sessions_dir is None or self._session_id is None:
            return None

        source = self._sessions_dir / f"{self._session_id}.jsonl"

        try:
            if not source.exists():
                self.logger.debug(
                    f"Claude session file not found: {source.name}"
                )
                return None

            # Build destination
            dest_dir = self._dest_dir()
            dest_name = self._build_filename(purpose)
            dest = dest_dir / dest_name

            shutil.copy2(source, dest)

            # Also copy companion subagents directory if it exists
            subagent_dir = self._sessions_dir / self._session_id / "subagents"
            if subagent_dir.is_dir():
                dest_sub = dest_dir / dest_name.replace(".jsonl", "") / "subagents"
                if dest_sub.exists():
                    shutil.rmtree(dest_sub)
                shutil.copytree(subagent_dir, dest_sub)

            self.logger.info(
                f"Captured Claude session trace: {source.name} -> {dest}"
            )
            return dest

        except Exception as e:
            self.logger.warning(f"Failed to capture Claude session trace: {e}")
            return None


# ============================================================================
# Copilot CLI manager
# ============================================================================

class CopilotSessionManager(SessionManager):
    """Session manager for the GitHub Copilot CLI.

    Injects Copilot-specific CLI arguments (``--log-dir``,
    ``--log-level``, ``--allow-all``) into ``extra_args`` and appends
    the prompt as the final argument.
    """

    def before(self, ctx: TraceContext, prompt: str) -> None:
        """Inject Copilot-specific CLI flags and append prompt.

        Adds ``--log-dir``, ``--log-level``, ``--allow-all`` and the
        prompt text itself to ``extra_args``.
        """
        log_dir = COPILOT_LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        ctx.extra_args.extend([
            "--log-dir", str(log_dir),
            "--log-level", "all",
            "--allow-all",
            "-p", prompt,
        ])

    def after(self, purpose: str) -> Optional[Path]:
        """No trace capture yet."""
        return None


# ============================================================================
# Factory
# ============================================================================

# Registry mapping agent type names to manager classes.
_MANAGER_REGISTRY: Dict[str, type] = {
    "claude": ClaudeSessionManager,
    "copilot": CopilotSessionManager,
}


def register_manager(agent_type: str, manager_cls: type) -> None:
    """Register a custom manager class for an agent type."""
    _MANAGER_REGISTRY[agent_type] = manager_cls


def create_session_manager(
    agent_type: str,
    project_dir: Path,
    trace_filename_builder: Optional[Callable[[str], str]] = None,
    logger: Optional[logging.Logger] = None,
) -> SessionManager:
    """Factory that returns the appropriate ``SessionManager`` subclass based on the agent type.

    Args:
        agent_type: Canonical agent name returned by ``detect_agent_type()``
            (e.g. ``"claude"``, ``"copilot"``, ``"unknown"``).
        project_dir: Absolute path to the project root.
        trace_filename_builder: Optional custom filename builder.
        logger: Logger instance.

    Returns:
        A ``SessionManager`` subclass instance, or ``NullSessionManager`` if
        the agent type is not recognised.  Each subclass determines its own
        ``trajectory_dir`` via ``DEFAULT_TRAJECTORY_DIR``.
    """
    manager_cls = _MANAGER_REGISTRY.get(agent_type, NullSessionManager)
    return manager_cls(
        project_dir=project_dir,
        trace_filename_builder=trace_filename_builder,
        logger=logger,
    )
