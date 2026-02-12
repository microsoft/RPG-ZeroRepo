from dataclasses import dataclass
import logging
import os
import subprocess
from typing import Any, List

from .logs import setup_logger


@dataclass
class CommandResult:
    success: bool
    stdout: str | bytes | Any | None
    stderr: str | bytes | Any | None


class GitCommandRunner:
    """Utility class to run git commands."""

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.logger = setup_logger(logging.getLogger("GitCommandRunner"), level=logging.DEBUG)

    def run(self, cmd: List[str], capture_output: bool = True, text: bool = True, check: bool = False, cwd: str = None) -> CommandResult:
        """Run a git command and return success status and output

        Args:
            cmd: Git command as list of strings
            capture_output: Default True. If True, capture stdout and stderr.
            text: Default True. If True, output is str; else bytes.
            check: Default False. Set check=True if you want to ensure the command succeeds.
                   If True, raise exception (subprocess.CalledProcessError) on non-zero exit code.
                   If False, return CommandResult with success=False on non-zero exit code.
            cwd: Defaults to self.repo_path. Working directory to run the command in.

        Returns:
            CommandResult
            - success is True only if returncode == 0
            - output is stdout on success, stderr on failure
        """
        self._fix_git_ownership_and_permissions()

        work_dir = cwd if cwd is not None else self.repo_path

        # First attempt
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=capture_output,
                text=text,
                check=check,
            )
            return CommandResult(success=(result.returncode == 0), stdout=result.stdout, stderr=result.stderr)
        except subprocess.CalledProcessError as e:
            # If it's a permissions issue (exit code 128), try to fix permissions and retry
            if e.returncode == 128:
                self.logger.warning("Git command failed with exit code 128, attempting to fix permissions and retry...")
                self._force_fix_git_permissions()
                # Retry
                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    capture_output=capture_output,
                    text=text,
                    check=check,
                )
                return CommandResult(success=(result.returncode == 0), stdout=result.stdout, stderr=result.stderr)
            raise


    def _fix_git_ownership_and_permissions(self):
        """Fix ownership and permissions for git operations to solve 'dubious ownership' issues"""
        # Run only once to avoid doing this on every run()
        if getattr(self, "_ownership_fixed", False):
            return
        self._ownership_fixed = True

        # 1) Remove any stale index.lock file
        index_lock = os.path.join(self.repo_path, ".git", "index.lock")
        if os.path.exists(index_lock):
            try:
                os.remove(index_lock)
                self.logger.info(f"Removed stale git lock file: {index_lock}")
            except OSError as e:
                self.logger.warning(f"Could not remove git lock file: {e}")

        # 2) Configure safe.directory (most reliable approach)
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", self.repo_path],
                check=False,
                capture_output=True
            )
            self.logger.debug(f"Added {self.repo_path} to git safe directories")
        except Exception as e:
            self.logger.warning(f"Could not configure git safe.directory: {e}")

        # 3) Attempt to fix permissions of the .git directory
        self._force_fix_git_permissions()

    def _force_fix_git_permissions(self):
        """Force-fix permissions for the entire repo directory (for files created as root inside containers)"""
        if not os.path.exists(self.repo_path):
            return

        try:
            current_uid = os.getuid()
            current_gid = os.getgid()
            # Fix the entire repo_path, not just the .git directory
            result = subprocess.run(
                ["sudo", "chown", "-R", f"{current_uid}:{current_gid}", self.repo_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.debug(f"Fixed repo ownership with uid={current_uid}, gid={current_gid}")
            else:
                self.logger.debug(f"chown repo failed (non-critical): {result.stderr}")
        except Exception as e:
            self.logger.debug(f"chown not available (non-critical): {e}")