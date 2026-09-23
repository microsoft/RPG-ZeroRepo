"""Resolve product tools without searching repository-controlled executables.

The user's external installations and absolute PATH directories are trusted.
Repository/cwd directories, their aliases, and Windows shell wrappers are not.
Standard library only: importing this module does not inspect the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath, PureWindowsPath

_IS_WINDOWS = os.name == "nt"
_TOOLS = frozenset({"git", "uv", "pipx"})


def _comparison_path(path: PurePath) -> PurePath:
    if isinstance(path, PureWindowsPath):
        value = str(path)
        if value[:8].lower() == "\\\\?\\unc\\":
            return PureWindowsPath("\\\\" + value[8:])
        if value.startswith("\\\\?\\"):
            return PureWindowsPath(value[4:])
    return path


def _within(path: Path, root: Path) -> bool:
    return _comparison_path(path).is_relative_to(_comparison_path(root))


def _boundary(path: Path) -> Path:
    """Include the enclosing checkout when called from one of its subfolders."""
    path = path.resolve()
    for parent in (path, *path.parents):
        if ((parent / ".git").exists() or (parent / ".git").is_symlink()
                or (parent / ".cmind" / "config.toml").exists()
                or (parent / ".cmind" / "config.toml").is_symlink()):
            return parent
    return path


def resolve_tool(name: str, workspace: Path | str, *, search_path: str | None = None) -> str:
    """Return an absolute external executable, or fail without launching anything.

    Windows probes .exe only, never PATHEXT/.cmd/.bat/.ps1 or cwd fallback.
    Both lexical and canonical paths are checked to reject redirected entries.
    No cache: a different workspace or PATH must not reuse prior authorization.
    """
    if name not in _TOOLS:
        raise ValueError("Unsupported product tool")
    try:
        roots = (_boundary(Path(workspace)), _boundary(Path.cwd()))
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileNotFoundError("Cannot establish trusted tool search boundaries") from exc
    filename = name + ".exe" if _IS_WINDOWS else name
    path = os.environ.get("PATH", "") if search_path is None else search_path
    for entry in path.split(os.pathsep):
        directory = Path(entry)
        if not entry or not directory.is_absolute():
            continue
        try:
            canonical_dir = directory.resolve(strict=True)
            if any(_within(p, root) for p in (directory, canonical_dir) for root in roots):
                continue
            candidate = (canonical_dir / filename).resolve(strict=True)
            if any(_within(candidate, root) for root in roots):
                continue
            if _IS_WINDOWS and candidate.suffix.lower() != ".exe":
                continue
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except (OSError, RuntimeError, ValueError):
            continue
    raise FileNotFoundError(
        f"No trusted {name} executable found outside the workspace/current checkout. "
        "Install it externally and use an absolute PATH directory."
    )


def resolve_git(workspace: Path | str) -> str:
    return resolve_tool("git", workspace)
