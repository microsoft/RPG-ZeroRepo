"""Locate bundled core_pack assets inside the installed package.

The bundle is created at wheel-build time by hatch's ``force-include``
(see ``pyproject.toml``).  After ``uv tool install cmind-cli``, the
layout is::

    <prefix>/lib/python3.x/site-packages/cmind_cli/
        __init__.py
        _assets.py
        core_pack/
            scripts/         (full CoderMind/scripts/ tree)
            commands/        (full CoderMind/templates/commands/ tree)

``cmind init`` and ``cmind update`` copy from here to the workspace
when bundle mode is active (the default).  When the bundle is absent
(typically in an editable install where ``force-include`` does not run),
:func:`available` returns ``False`` and callers should fall back to the
legacy GitHub-release-zip download path.

Design notes
------------
- Uses :func:`importlib.resources.files` rather than ``__file__``
  arithmetic so non-filesystem packaging formats (zip-imports,
  in-memory loaders) keep working.
- The returned path is a *filesystem* path (not a Traversable) because
  the consumers (``shutil.copytree`` etc.) need real paths.  This works
  for the default wheel layout; if we ever ship as a zipapp this code
  will need ``as_file()`` contexts.
- All functions are pure / side-effect-free.  No mutation of the bundle.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def core_pack_root() -> Path:
    """Absolute path to the bundled ``core_pack/`` directory.

    Returns the path regardless of whether it exists on disk — callers
    should check :func:`available` before using the path.
    """
    return Path(str(files("cmind_cli").joinpath("core_pack")))


def _dev_scripts_dir() -> Path | None:
    """Locate the repo-root ``scripts/`` directory for editable/dev installs.

    When ``cmind-cli`` is installed in editable mode (``pip install -e .``
    or ``uv run cmind ...`` from the source tree), hatch's
    ``force-include`` does not populate ``cmind_cli/core_pack/``.  In
    that case we fall back to the live source at ``<repo>/scripts/``,
    which sits two levels above this file::

        <repo>/
            src/cmind_cli/_assets.py   ← __file__
            scripts/                     ← target
    """
    here = Path(__file__).resolve()
    # src/cmind_cli/_assets.py → repo = parents[2]
    if len(here.parents) >= 3:
        candidate = here.parents[2] / "scripts"
        if candidate.is_dir():
            return candidate
    return None


def _dev_commands_dir() -> Path | None:
    """Counterpart to :func:`_dev_scripts_dir` for slash-command templates."""
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        candidate = here.parents[2] / "templates" / "commands"
        if candidate.is_dir():
            return candidate
    return None


def available() -> bool:
    """True iff a usable scripts source exists.

    Returns ``True`` when either the wheel-bundled ``core_pack/scripts/``
    OR the dev-mode ``<repo>/scripts/`` is present.  Used to decide
    whether the bundle path is viable; callers fall back to the legacy
    GitHub-release-zip download path otherwise.
    """
    return scripts_dir().is_dir()


def scripts_dir() -> Path:
    """Directory containing the CoderMind pipeline scripts.

    Resolution order:
      1. Wheel bundle: ``<site-packages>/cmind_cli/core_pack/scripts/``
      2. Dev/editable fallback: ``<repo>/scripts/``

    Falls back to the wheel path even when missing so error messages
    contain a stable, recognisable location.
    """
    bundled = core_pack_root() / "scripts"
    if bundled.is_dir():
        return bundled
    dev = _dev_scripts_dir()
    if dev is not None:
        return dev
    return bundled  # may not exist; caller decides how to surface


def commands_dir() -> Path:
    """Directory containing the slash-command templates.

    Same resolution order as :func:`scripts_dir`.
    """
    bundled = core_pack_root() / "commands"
    if bundled.is_dir():
        return bundled
    dev = _dev_commands_dir()
    if dev is not None:
        return dev
    return bundled


def mcp_server_path() -> Path:
    """Convenience: path to the MCP server entry script."""
    return scripts_dir() / "mcp_server.py"


def list_scripts() -> list[str]:
    """Return all script relative paths (POSIX-style) under :func:`scripts_dir`.

    Filters to ``.py`` files only, skips ``__pycache__`` directories,
    and sorts alphabetically.  Used by ``cmind script --list``.
    """
    root = scripts_dir()
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        out.append(p.relative_to(root).as_posix())
    out.sort()
    return out

