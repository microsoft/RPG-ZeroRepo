"""Load the shared tool resolver only from this installation's trusted assets."""

from functools import lru_cache
import importlib.util
from pathlib import Path
import shlex
import sys

from . import _assets


@lru_cache(maxsize=1)
def _policy():
    path = (_assets.scripts_dir() / "common" / "trusted_tools.py").resolve()
    spec = importlib.util.spec_from_file_location("_cmind_trusted_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the installed tool resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_tool(name: str, workspace: Path | str) -> str:
    return _policy().resolve_tool(name, workspace)


def resolve_git(workspace: Path | str) -> str:
    return resolve_tool("git", workspace)


def cli_argv(*args: str) -> list[str]:
    """Re-enter this CLI using its interpreter and pinned package bootstrap.

    -I prevents cwd/PYTHONPATH from selecting a different cmind_cli module.
    The bootstrap also supports the explicitly installed editable source tree.
    """
    entry = Path(__file__).resolve().with_name("_cli_entry.py")
    if not entry.is_file():
        raise FileNotFoundError("Installed CoderMind CLI entry is missing")
    return [sys.executable, "-I", str(entry), *args]


def cli_shell_command(*args: str) -> str:
    """Quote the pinned entry for Git/Claude's POSIX shell, including Git Bash."""
    command = cli_argv(*args)
    return shlex.join([
        Path(value).as_posix() if index in (0, 2) else value
        for index, value in enumerate(command)
    ])
