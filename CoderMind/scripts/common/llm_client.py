#!/usr/bin/env python3
"""LLM Client Module for CoderMind.

This module provides a common LLM client with trajectory recording support.
All LLM calls (prompts and responses) are recorded in the trajectory when
a trajectory instance is provided.
"""

import json
import logging
import os as _os
import platform as _platform
import re
import shlex
import signal as _signal
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from common.llm_types import Memory
from common.session_manager import create_session_manager
from . import paths as _paths
from .paths import REPO_DIR as _REPO_DIR, WORKSPACE_ROOT as _WORKSPACE_ROOT

_IS_WINDOWS = _platform.system() == "Windows"


def _set_pdeathsig() -> None:
    """Preexec hook: ask the kernel to send SIGTERM to this child when its parent dies (including SIGKILL).  Called after fork() but before exec() so it runs in the child's address space.  Silently ignored on non-Linux.  Only ever passed as ``preexec_fn`` on POSIX (see ``generate``); Windows never touches this."""
    try:
        import ctypes, signal as _s
        ctypes.CDLL("libc.so.6").prctl(1, _s.SIGTERM)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and its full descendant tree, cross-platform.

    POSIX: the child was launched via ``start_new_session=True`` so it
    leads its own process group; killpg reaches every descendant.
    Windows: the child was launched with ``CREATE_NEW_PROCESS_GROUP``
    (``preexec_fn``/``killpg`` don't exist there); ``taskkill /T`` walks
    the process tree by PID instead.
    """
    if _IS_WINDOWS:
        try:
            res = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            if res.returncode != 0:
                proc.kill()
        except Exception:
            proc.kill()
    else:
        try:
            _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
        except Exception:
            proc.kill()


# ----------------------------------------------------------------------------
# AI CLI command resolution
# ----------------------------------------------------------------------------
#
# Resolution priority:
#   P1. LLMClient(tool="...") constructor argument
#   P2. CMIND_AI_CLI_CMD env var
#   P3. <workspace>/.cmind/config.toml  [cmind].ai_cli_cmd
#   P4. _BAKED_IN_VALUE (release-zip builds substitute it at packaging time;
#                        bundle builds leave the placeholder unchanged)
#
# An unresolved value is reported lazily by LLMClient.generate, not here,
# so importing the module and constructing an LLMClient without calling
# the LLM both succeed.

# Built via concatenation so the release-zip's ``sed s|<AI_CLI_CMD>|...|``
# does not rewrite this sentinel.
_PLACEHOLDER_LITERAL = "<" + "AI_CLI_CMD" + ">"

_BAKED_IN_VALUE = "<AI_CLI_CMD>"


def _load_ai_cli_cmd() -> str:
    """Resolve the AI CLI command string via the P1-P4 priority chain.

    P1 is handled by :class:`LLMClient.__init__` (constructor argument).
    This function implements P2-P4 and returns ``""`` if none of them
    yield a usable value — callers decide how to react.

    The workspace root is *re-resolved at every invocation* via
    :func:`paths._find_workspace_root`, not via the import-frozen
    :data:`paths.WORKSPACE_ROOT` constant.  This matters for long-lived
    processes that may serve more than one workspace (e.g. a future
    global MCP server).
    """
    # P2: env var (highest non-P1 priority — useful in tests and one-off
    # overrides without editing the workspace config).
    env_val = _os.environ.get("CMIND_AI_CLI_CMD", "").strip()
    if env_val:
        return env_val

    # P3: workspace config.toml.
    try:
        workspace = _paths._find_workspace_root()
        cfg_path = workspace / ".cmind" / "config.toml"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                data = tomllib.load(f)
            cfg_val = (data.get("cmind") or {}).get("ai_cli_cmd", "")
            if isinstance(cfg_val, str):
                cfg_val = cfg_val.strip()
                if cfg_val:
                    return cfg_val
    except Exception:
        # paths resolution, missing tomllib, or malformed TOML must not
        # crash LLMClient construction.
        pass

    # P4: legacy baked-in value (release-zip-substituted at build time).
    if _BAKED_IN_VALUE and _BAKED_IN_VALUE != _PLACEHOLDER_LITERAL:
        return _BAKED_IN_VALUE

    return ""


# Resolved once at import for backward-compat with callers that referenced
# the module-level constant directly.  New code should call ``_load_ai_cli_cmd()``
# or use ``LLMClient.tool`` (already populated through the same chain).
AI_CLI_CMD = _load_ai_cli_cmd()


# Mapping from the first token of AI_CLI_CMD to the canonical agent name
_CLI_TO_AGENT = {
    "copilot": "copilot",
    "claude": "claude",
    "gemini": "gemini",
    "qwen": "qwen",
    "agent": "cursor",      # cursor-agent uses "agent -p"
    "augment": "auggie",
    "codex": "codex",
    "codebuddy": "codebuddy",
    "qodercli": "qoder",
    "opencode": "opencode",
    "amp": "amp",
}


def detect_agent_type(cmd: Optional[str] = None) -> str:
    """Detect which AI coding agent is being used.

    Args:
        cmd: Optional explicit CLI command string.  When omitted we
            resolve dynamically via :func:`_load_ai_cli_cmd` so this
            function reflects the current workspace's configuration,
            not whatever was in effect at module import time.

    Returns one of: claude, gemini, copilot, cursor, codex, auggie,
                    amp, opencode, codebuddy, qoder, qwen, unknown
    """
    cmd = cmd if cmd is not None else _load_ai_cli_cmd()
    if not cmd or cmd == _PLACEHOLDER_LITERAL:
        return "unknown"

    try:
        first_token = shlex.split(cmd, posix=False)[0]
    except (IndexError, ValueError):
        parts = cmd.strip().split(maxsplit=1)
        if not parts:
            return "unknown"
        first_token = parts[0]

    # On Windows, cmd may be a quoted, backslash-separated path with an
    # executable suffix (e.g. "C:\tools\claude.cmd" or claude.exe) instead of
    # the bare "claude" that _CLI_TO_AGENT is keyed on.
    executable_name = first_token.strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".cmd", ".exe", ".bat", ".ps1"):
        if executable_name.endswith(suffix):
            executable_name = executable_name[: -len(suffix)]
            break

    return _CLI_TO_AGENT.get(executable_name, "unknown")


# ============================================================================
# LLM Interaction Record (for standalone use without Trajectory)
# ============================================================================

@dataclass
class LLMCallRecord:
    """Record of a single LLM call with full prompt and response.
    
    This is used to track LLM interactions either within a Trajectory
    or independently for debugging/analysis purposes.
    """
    call_id: int
    timestamp: str
    purpose: str
    prompt: str
    response: Optional[str] = None
    parsed_result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMCallRecord":
        return cls(**data)


# ============================================================================
# LLM Client with Trajectory Integration
# ============================================================================

class LLMClient:
    """Client for interacting with AI CLI tools.
    
    Supports trajectory recording of all LLM calls (prompts and responses)
    for debugging, analysis, and reproducibility.
    
    Usage:
        # Without trajectory (calls are still tracked internally)
        client = LLMClient()
        response = client.generate("What is Python?")
        
        # With trajectory recording
        from common.trajectory import Trajectory
        traj = Trajectory("my_command")
        traj.start()
        step = traj.add_step("generate_code", "Generate some code")
        traj.start_step(step.step_id)
        
        client = LLMClient(trajectory=traj, step_id=step.step_id)
        response = client.generate("Write hello world", purpose="code_generation")
        # LLM call is automatically recorded in trajectory
    """
    
    # Workspace root — sourced from common.paths so that symlink-based
    # dev workflows resolve correctly (see paths._find_workspace_root).
    # Used for session trace storage (.cmind/logs/<agent>/) and to
    # express captured-trace paths relative to the workspace.
    _INFERRED_PROJECT_DIR: Path = _WORKSPACE_ROOT

    def __init__(
        self,
        tool: str = None,
        trajectory: Optional[Any] = None,  # Trajectory instance
        step_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize LLM Client.
        
        Args:
            tool: CLI tool command (default: "llm")
            trajectory: Trajectory instance for recording LLM calls
            step_id: Current step ID in the trajectory
            logger: Logger instance
        """
        # P1 (explicit arg) wins; otherwise P2-P4 chain via _load_ai_cli_cmd.
        # The empty-string case is tolerated here so unit tests / utilities
        # that construct an LLMClient without intending to invoke the LLM
        # keep working.  The actual error is raised in :meth:`generate` if
        # the tool is still empty when a call is attempted.
        self.tool = tool if tool is not None else _load_ai_cli_cmd()
        self.trajectory = trajectory
        self.step_id = step_id
        self.logger = logger or logging.getLogger(__name__)
        
        # Session manager — driven by detect_agent_type(self.tool) so the
        # right subclass is chosen even when self.tool came from the
        # workspace config (not the import-time AI_CLI_CMD constant).
        # project_dir must match the subprocess cwd (workspace root == REPO_DIR)
        # so that Claude CLI's session file path
        # (~/.claude/projects/<encoded-cwd>/) can be correctly located by
        # the session manager.
        self._session_manager = create_session_manager(
            agent_type=detect_agent_type(self.tool),
            project_dir=_REPO_DIR,
            trace_filename_builder=self._build_trace_filename,
            logger=self.logger,
        )
        
        # Internal call tracking
        self._call_counter = 0
        self._call_history: List[LLMCallRecord] = []
    
    def set_trajectory(self, trajectory: Any, step_id: int = None) -> None:
        """Set or update the trajectory for recording LLM calls.
        
        Args:
            trajectory: Trajectory instance
            step_id: Current step ID (optional, can be set later)
        """
        self.trajectory = trajectory
        if step_id is not None:
            self.step_id = step_id
    
    def set_step_id(self, step_id: int) -> None:
        """Set the current step ID for trajectory recording."""
        self.step_id = step_id

    # ====================================================================
    # Session tracer helpers
    # ====================================================================

    def _build_trace_filename(self, purpose: str) -> str:
        """Build a semantically meaningful filename for the captured session.

        Format:  ``<YYYYMMDD-HHmmss>-<step_name>-<purpose>.jsonl``

        Date-first layout ensures ``ls`` / ``sort`` orders files chronologically.

        * ``step_name`` is taken from the current trajectory step (if any).
        * ``purpose`` is the value passed to ``generate()`` / ``call_structured()``.
        * The timestamp is the completion time (now).
        """
        parts: List[str] = []

        # Try to get step name from trajectory
        if self.trajectory and self.step_id is not None:
            try:
                step = self.trajectory.get_step(self.step_id)
                if step and step.name:
                    parts.append(step.name)
            except Exception:
                pass

        # Append purpose (avoid duplicating the step name)
        if purpose and purpose not in parts:
            parts.append(purpose)

        if not parts:
            parts.append("llm_call")

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "-".join(parts)
        # Sanitise: keep alphanumerics, hyphens, underscores
        safe = re.sub(r"[^\w\-]", "_", safe)
        return f"{ts}-{safe}.jsonl"

    def generate(
        self,
        prompt: str,
        purpose: str = "general",
        max_retries: int = 3,
        timeout: Optional[int] = 1800,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Generate response from LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            purpose: Description of the purpose of this call (for trajectory)
            max_retries: Number of retry attempts
            timeout: Timeout in seconds
            metadata: Additional metadata to store with the call record
            
        Returns:
            Response text from LLM
            
        Raises:
            RuntimeError: If LLM call fails after all retries
        """
        # Lazy validation: the constructor tolerates an empty/placeholder
        # ``self.tool`` so that tests and tools can build an LLMClient
        # without triggering an LLM invocation, but the moment we are
        # actually asked to call out, the configuration must be valid.
        if not self.tool or self.tool == _PLACEHOLDER_LITERAL:
            raise RuntimeError(
                "AI CLI command not configured.  Run "
                "`cmind init --ai <name>` in this workspace, or set the "
                "CMIND_AI_CLI_CMD environment variable."
            )

        # Create call record
        self._call_counter += 1
        call_record = LLMCallRecord(
            call_id=self._call_counter,
            timestamp=datetime.now().isoformat(),
            purpose=purpose,
            prompt=prompt,
            metadata=metadata or {}
        )
        
        # Record interaction start in trajectory
        interaction_id = None
        if self.trajectory and self.step_id is not None:
            try:
                interaction_id = self.trajectory.start_llm_interaction(
                    self.step_id,
                    purpose,
                    prompt
                )
            except Exception as e:
                self.logger.warning(f"Failed to record LLM interaction start: {e}")
        
        start_time = time.time()
        response = None
        error = None

        with self._session_manager.trace(prompt, purpose=purpose) as trace_ctx:
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"Calling LLM (attempt {attempt + 1})")
                    
                    # Build command with any extra args from session manager.
                    # On retries, refresh_for_retry() regenerates any
                    # single-use tokens (e.g., Claude's --session-id)
                    # and resets stdin for re-reading the prompt.
                    if attempt > 0:
                        trace_ctx.refresh_for_retry()
                    cmd = shlex.split(self.tool) + trace_ctx.extra_args

                    # Sub-agent runs in the project repo directory.
                    # POSIX: start_new_session=True puts the child in its own
                    # process group so killpg kills the whole tree on parent
                    # exit; preexec_fn=_set_pdeathsig handles the SIGKILL case
                    # via PR_SET_PDEATHSIG (kernel sends SIGTERM to child on
                    # parent death). Neither is supported on Windows, where
                    # CREATE_NEW_PROCESS_GROUP + taskkill /T (see
                    # _kill_process_tree) plays the same role.
                    popen_kwargs: Dict[str, Any] = dict(
                        stdin=trace_ctx.stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=trace_ctx.env,
                        cwd=_REPO_DIR,
                    )
                    if _IS_WINDOWS:
                        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                        # Also force UTF-8 stdio inside the CLI subprocess
                        # itself (e.g. Claude/Codex CLI is often a Python
                        # or Node entrypoint). Without this, a non-tty
                        # stdout on Windows falls back to the legacy code
                        # page and the *child* can crash before its output
                        # ever reaches the `encoding="utf-8"` decode above.
                        popen_kwargs["env"] = dict(trace_ctx.env or {})
                        popen_kwargs["env"].setdefault("PYTHONIOENCODING", "utf-8:replace")
                        popen_kwargs["env"].setdefault("PYTHONUTF8", "1")
                    else:
                        popen_kwargs["start_new_session"] = True
                        popen_kwargs["preexec_fn"] = _set_pdeathsig
                    proc = subprocess.Popen(cmd, **popen_kwargs)
                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                    except BaseException:
                        # Kill the entire process tree (agent + any pytest children)
                        _kill_process_tree(proc)
                        proc.wait()
                        raise
                    result = subprocess.CompletedProcess(
                        cmd, proc.returncode, stdout, stderr
                    )
                    
                    if result.returncode != 0:
                        error = f"LLM call failed with return code {result.returncode}: {result.stderr}"
                        self.logger.warning(error)
                        continue
                    
                    response = result.stdout.strip()
                    if response:
                        break
                    else:
                        error = "LLM returned empty response"
                        self.logger.warning(error)
                        
                except subprocess.TimeoutExpired:
                    error = f"LLM call timed out after {timeout}s"
                    self.logger.warning(f"LLM call timed out (attempt {attempt + 1})")
                except Exception as e:
                    error = str(e)
                    self.logger.warning(f"LLM call error: {e}")

        # Session trace captured automatically by context manager
        captured_path = trace_ctx.captured_path
        
        duration = time.time() - start_time
        
        # Update call record
        call_record.response = response
        call_record.duration_seconds = duration
        call_record.success = response is not None
        call_record.error = error if not response else None
        if captured_path:
            try:
                rel = captured_path.relative_to(self._INFERRED_PROJECT_DIR)
                call_record.metadata["session_trace"] = str(rel)
            except ValueError:
                # captured_path lives outside the workspace (e.g. Claude
                # writes traces under ~/.claude/projects/<hash>/sessions/).
                # session_trace is purely informational — never let a
                # bookkeeping error abort the LLM call.
                call_record.metadata["session_trace"] = str(captured_path)
        
        # Store in history
        self._call_history.append(call_record)
        
        # Record interaction completion in trajectory
        if self.trajectory and self.step_id is not None and interaction_id is not None:
            try:
                self.trajectory.complete_llm_interaction(
                    self.step_id,
                    interaction_id,
                    response=response or "",
                    parsed_result=None,
                    success=response is not None,
                    error=error if not response else None,
                    duration_seconds=duration
                )
            except Exception as e:
                self.logger.warning(f"Failed to record LLM interaction completion: {e}")
        
        if not response:
            raise RuntimeError(f"Failed to get response from LLM after {max_retries} retries: {error}")
        
        return response
    
    def generate_with_record(
        self,
        prompt: str,
        purpose: str = "general",
        max_retries: int = 3,
        timeout: Optional[int] = 1800,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, LLMCallRecord]:
        """Generate response from LLM and return both response and call record.
        
        This is useful when you need access to the full call record for
        custom processing or when not using trajectory.
        
        Returns:
            Tuple of (response_text, call_record)
        """
        initial_count = len(self._call_history)
        response = self.generate(
            prompt=prompt,
            purpose=purpose,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata
        )
        # Get the call record that was just added
        if len(self._call_history) > initial_count:
            call_record = self._call_history[-1]
        else:
            # Create a minimal record if something went wrong
            call_record = LLMCallRecord(
                call_id=self._call_counter,
                timestamp=datetime.now().isoformat(),
                purpose=purpose,
                prompt=prompt,
                response=response,
                success=True
            )
        return response, call_record
    
    # ====================================================================
    # Memory-based generation (for Encoder / RPGAgent)
    # ====================================================================

    @staticmethod
    def _flatten_memory(memory: Memory) -> str:
        """Flatten Memory messages into a single prompt string.

        Converts the multi-turn conversation stored in Memory into a single
        prompt suitable for CLI subprocess invocation. System messages are
        placed first without role prefix; user/assistant messages are
        clearly delimited with role headers.
        """
        parts = []
        for msg in memory.history:
            if msg.role == "system":
                parts.append(msg.content)
            elif msg.role == "user":
                parts.append(f"\n[User]\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"\n[Assistant]\n{msg.content}")
        return "\n".join(parts)

    def generate_with_memory(
        self,
        memory: Memory,
        purpose: str = "general",
        max_retries: int = 3,
        timeout: Optional[int] = 1800,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Generate from a Memory object by flattening to a single prompt.

        Encoder and RPGAgent use Memory to manage multi-turn conversations.
        This method flattens the message sequence into a single prompt
        string, then invokes the CLI subprocess via ``generate()``.

        Differences from ``generate()``:
        - Input: Memory object (multiple messages) instead of a single string
        - Return: ``Optional[str]`` (None on failure) instead of raising

        Args:
            memory: Memory instance with SystemMessage / UserMessage /
                AssistantMessage entries.
            purpose, max_retries, timeout, metadata: Same as ``generate()``.

        Returns:
            LLM response text, or None if all retries failed.

        Raises:
            RuntimeError: only when ``self.tool`` is not configured.
                Genuine LLM-call failures (subprocess errors, timeouts,
                bad responses) are swallowed and surface as ``None``;
                missing configuration is an unrecoverable user-action
                error and must not be silently masked.
        """
        # Eagerly surface the "AI CLI not configured" condition.  This is
        # not a transient failure that ``None`` should represent — the
        # user needs an actionable error.  Genuine LLM failures continue
        # to be caught below.
        if not self.tool or self.tool == _PLACEHOLDER_LITERAL:
            raise RuntimeError(
                "AI CLI command not configured.  Run "
                "`cmind init --ai <name>` in this workspace, or set the "
                "CMIND_AI_CLI_CMD environment variable."
            )

        prompt = self._flatten_memory(memory)
        try:
            return self.generate(
                prompt=prompt,
                purpose=purpose,
                max_retries=max_retries,
                timeout=timeout,
                metadata=metadata,
            )
        except RuntimeError:
            return None

    @property
    def last_usage(self) -> Dict[str, int]:
        """Token usage stub (CLI subprocess cannot track tokens)."""
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def generate_and_record_parsed(
        self,
        prompt: str,
        purpose: str = "general",
        max_retries: int = 3,
        timeout: Optional[int] = 1800,
        metadata: Dict[str, Any] = None,
        parsed_result: Dict[str, Any] = None
    ) -> str:
        """Generate response from LLM and update the record with parsed result.
        
        Use this when you parse the response after receiving it and want
        to record the parsed result in the trajectory.
        
        Args:
            prompt: The prompt to send to the LLM
            purpose: Description of the purpose of this call
            parsed_result: Parsed result to store (can be set after parsing)
            
        Returns:
            Response text from LLM
        """
        response = self.generate(
            prompt=prompt,
            purpose=purpose,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata
        )
        
        # Update the last call record with parsed result
        if self._call_history and parsed_result:
            self._call_history[-1].parsed_result = parsed_result
            
            # Also update trajectory if available
            if self.trajectory and self.step_id is not None:
                # Find the interaction and update it
                step = self.trajectory.get_step(self.step_id)
                if step and step.llm_interactions:
                    # Update the last interaction's parsed_result
                    step.llm_interactions[-1].parsed_result = parsed_result
                    self.trajectory.save()
        
        return response
    
    def update_last_parsed_result(self, parsed_result: Dict[str, Any]) -> None:
        """Update the parsed result for the last LLM call.
        
        Call this after parsing the LLM response to store the parsed
        result in both the call history and trajectory.
        """
        if self._call_history:
            self._call_history[-1].parsed_result = parsed_result
            
            if self.trajectory and self.step_id is not None:
                step = self.trajectory.get_step(self.step_id)
                if step and step.llm_interactions:
                    step.llm_interactions[-1].parsed_result = parsed_result
                    self.trajectory.save()
    
    def parse_result_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract <result_json> JSON block from response.

        Includes resilient JSON repair for common LLM output errors:
        trailing commas, Python literals (True/False/None), missing commas,
        and newlines inside strings.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict or None if not found/invalid
        """
        match = re.search(r"<result_json>\s*(.*?)\s*</result_json>", response, re.DOTALL)
        if not match:
            if "<result_json>" in response:
                # <result_json> found but no </result_json> — likely truncated response
                self.logger.warning(
                    "Found <result_json> but no </result_json> - response may be truncated"
                )
                start = response.find("<result_json>") + len("<result_json>")
                raw = response[start:].strip()
                # Use brace counting to find where JSON ends
                brace_count = 0
                json_end = -1
                for i, char in enumerate(raw):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    json_str = raw[:json_end]
                else:
                    self.logger.error("Could not find valid JSON in truncated response")
                    return None
            else:
                # Fallback: try to extract JSON object directly (no result_json tags)
                self.logger.debug(
                    "No <result_json> block found, trying to extract JSON directly..."
                )
                json_match = re.search(
                    r'\{\s*"thinking".*?\}(?=\s*$|\s*```)', response, re.DOTALL
                )
                if not json_match:
                    json_match = re.search(
                        r'\{\s*"summary".*?\}(?=\s*$|\s*```)', response, re.DOTALL
                    )
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None
        else:
            json_str = match.group(1).strip()

        # Clean up markdown code blocks if present
        json_str = re.sub(r'^```json?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        
        # First attempt: try parsing as-is
        try:
            data = json.loads(json_str)
            # Extract parameters if wrapped
            if "parameters" in data:
                return data["parameters"]
            return data
        except json.JSONDecodeError as first_err:
            self.logger.debug(f"First parse attempt failed: {first_err}")

        # ----- JSON repair pass -----
        # Remove trailing commas before closing brackets
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix Python literals -> JSON
        json_str = re.sub(r"\bTrue\b", "true", json_str)
        json_str = re.sub(r"\bFalse\b", "false", json_str)
        json_str = re.sub(r"\bNone\b", "null", json_str)

        # Fix missing commas between key-value pairs
        json_str = re.sub(
            r'(true|false|null|\d+\.?\d*|"[^"]*"|\]|\})\s*\n\s*"', r'\1,\n"', json_str
        )
        json_str = re.sub(
            r'(true|false|null|\d+\.?\d*|"[^"]*"|\]|\})\s+"', r'\1, "', json_str
        )

        # Second attempt after basic repair
        try:
            data = json.loads(json_str)
            if "parameters" in data:
                return data["parameters"]
            return data
        except json.JSONDecodeError as second_err:
            self.logger.debug(f"Second parse attempt failed: {second_err}")

        # Third attempt: handle newlines inside strings by joining split lines
        lines = json_str.split("\n")
        cleaned_lines = []
        in_string = False
        current_line = ""

        for line in lines:
            quote_count = 0
            i = 0
            while i < len(line):
                if line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
                    quote_count += 1
                i += 1

            if in_string:
                current_line += " " + line.strip()
            else:
                if current_line:
                    cleaned_lines.append(current_line)
                current_line = line

            if quote_count % 2 == 1:
                in_string = not in_string

        if current_line:
            cleaned_lines.append(current_line)

        json_str = "\n".join(cleaned_lines)

        try:
            data = json.loads(json_str)
            if "parameters" in data:
                return data["parameters"]
            return data
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON string: {json_str[:500]}")
            self.logger.error(f"JSON parse failed after all repair attempts: {e}")
            return None

    def parse_json_block(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response (looks for ```json blocks or raw JSON).
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dict or None if not found/invalid
        """
        # Try ```json blocks first
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try raw JSON (find first { to last })
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass
        
        return None
    
    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
        max_retries: int = 3,
        purpose: str = "structured_call",
        timeout: Optional[int] = 1800,
    ) -> Tuple[str, Optional[Any], str]:
        """Call LLM and return structured output using a Pydantic model.

        Combines the system and user prompts, sends them to the LLM,
        parses the ``<result_json>`` block, applies ``trim_dict_keys`` and
        validates against the *response_model*.

        Also extracts the optional ``<think>`` block for diagnostics.

        Args:
            system_prompt: System prompt for LLM
            user_prompt: User prompt for LLM
            response_model: Pydantic model class for response validation
            max_retries: Number of retry attempts
            purpose: Purpose of this call for logging/trajectory
            timeout: Timeout in seconds

        Returns:
            ``(think_content, validated_model_or_None, raw_response)``
        """
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        last_think = ""
        last_response = ""

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"LLM structured call attempt {attempt + 1}/{max_retries}")

                # Use generate to get response (with trajectory recording)
                response = self.generate(
                    prompt=combined_prompt,
                    purpose=purpose,
                    max_retries=1,  # Handle retries at this level
                    timeout=timeout
                )

                last_response = response
                last_think = self.extract_think_block(response)

                # Parse the result_json block
                parsed_data = self.parse_result_json(response)
                if parsed_data:
                    result = self.validate_structure(parsed_data, response_model)
                    if result is not None:
                        self.update_last_parsed_result(parsed_data)
                        self.logger.info("[OK] LLM structured call successful")
                        return last_think, result, response
                    else:
                        self.logger.warning(
                            f"[FAIL] Validation failed (attempt {attempt + 1})"
                        )
                else:
                    self.logger.warning(
                        f"[FAIL] Unable to parse <result_json> block (attempt {attempt + 1})"
                    )
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"[FAIL] LLM call timeout (attempt {attempt + 1})")
            except RuntimeError as e:
                self.logger.warning(f"[FAIL] LLM call failed (attempt {attempt + 1}): {e}")
            except Exception as e:
                self.logger.warning(f"[FAIL] Error (attempt {attempt + 1}): {str(e)[:200]}")
        
        self.logger.error(f"[FAIL] All {max_retries} attempts failed")
        return last_think, None, last_response

    # ====================================================================
    # Utility helpers (think extraction, key trimming, validation)
    # ====================================================================

    @staticmethod
    def extract_think_block(response: str) -> str:
        """Extract the optional <think> block from the AI response.

        This does not affect result_json parsing — it is only for
        logging / analysis.

        Returns:
            The text inside <think>…</think>, or "" if not found.
        """
        try:
            if "<think>" in response and "</think>" in response:
                start = response.find("<think>") + len("<think>")
                end = response.find("</think>", start)
                if end != -1:
                    return response[start:end].strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def trim_dict_keys(d):
        """Recursively strip whitespace from all dictionary keys.

        This fixes a common LLM output error where keys contain
        leading / trailing spaces.
        """
        if isinstance(d, dict):
            return {
                (k.strip() if isinstance(k, str) else k): LLMClient.trim_dict_keys(v)
                for k, v in d.items()
            }
        if isinstance(d, list):
            return [LLMClient.trim_dict_keys(item) for item in d]
        return d

    def validate_structure(
        self, data: Dict[str, Any], response_model: type
    ) -> Optional[Any]:
        """Validate parsed data against a Pydantic model.

        Applies ``trim_dict_keys`` before validation so that
        whitespace-padded keys produced by the LLM do not cause
        spurious failures.

        Args:
            data: Parsed dictionary (e.g. from ``parse_result_json``).
            response_model: Pydantic ``BaseModel`` subclass.

        Returns:
            Validated model instance, or ``None`` on failure.
        """
        try:
            data = self.trim_dict_keys(data)
            result = response_model(**data)
            self.logger.info(
                f"[OK] Structure validation passed: {type(result).__name__}"
            )
            return result
        except Exception as e:
            self.logger.warning(f"[FAIL] Structure validation failed: {e}")
            return None

    def get_call_history(self) -> List[LLMCallRecord]:
        """Get all LLM call records."""
        return self._call_history.copy()
    
    def get_last_call(self) -> Optional[LLMCallRecord]:
        """Get the most recent LLM call record."""
        return self._call_history[-1] if self._call_history else None
    
    def get_call_summary(self) -> Dict[str, Any]:
        """Get a summary of all LLM calls."""
        total_duration = sum(
            c.duration_seconds or 0 for c in self._call_history
        )
        successful = sum(1 for c in self._call_history if c.success)
        failed = len(self._call_history) - successful
        
        return {
            "total_calls": len(self._call_history),
            "successful_calls": successful,
            "failed_calls": failed,
            "total_duration_seconds": total_duration,
            "purposes": list(set(c.purpose for c in self._call_history))
        }
    
    def export_call_history(self, filepath: str) -> None:
        """Export call history to a JSON file.
        
        Useful for debugging and analysis when not using trajectory.
        """
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_call_summary(),
            "calls": [c.to_dict() for c in self._call_history]
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear_call_history(self) -> None:
        """Clear the internal call history."""
        self._call_history.clear()
        self._call_counter = 0
