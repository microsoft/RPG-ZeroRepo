"""Locate bundled core_pack assets inside the installed package.

The bundle is created at wheel-build time by hatch's ``force-include``
(see ``pyproject.toml``).  After ``uv tool install rpgkit-cli``, the
layout is::

    <prefix>/lib/python3.x/site-packages/rpgkit_cli/
        __init__.py
        _assets.py
        core_pack/
            scripts/         (full RPG-Kit/scripts/ tree)
            commands/        (full RPG-Kit/templates/commands/ tree)

``rpgkit init`` and ``rpgkit update`` copy from here to the workspace
when bundle mode is active (the default).  When the bundle is absent
(typically in an editable install where ``force-include`` does not run),
:func:`available` returns ``False`` and callers should fall back to the
legacy GitHub-release-zip download path.

Design notes
------------
- We deliberately use :func:`importlib.resources.files` rather than
  ``__file__`` arithmetic so that future packaging formats (zip-imports,
  in-memory loaders) keep working.
- The returned path is a *filesystem* path (not a Traversable) because
  the consumers (``shutil.copytree`` etc.) need real paths.  This works
  for the default wheel layout; if we ever ship as a zipapp this code
  will need ``as_file()`` contexts.
- All functions are pure / side-effect-free.  No mutation of the bundle.

Plan: ``plans/01-package-bundle-and-ai-config.md``
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def core_pack_root() -> Path:
    """Absolute path to the bundled ``core_pack/`` directory.

    Returns the path regardless of whether it exists on disk — callers
    should check :func:`available` before using the path.
    """
    return Path(str(files("rpgkit_cli").joinpath("core_pack")))


def available() -> bool:
    """True iff a usable bundle exists in the installed package.

    Returns ``False`` for editable installs (where ``force-include`` did
    not run) and for any other situation where the bundle is missing
    or incomplete.  Callers use this to decide whether to fall back to
    the legacy GitHub-release-zip download path.
    """
    root = core_pack_root()
    return root.is_dir() and (root / "scripts").is_dir()


def scripts_dir() -> Path:
    """Directory containing the bundled RPG-Kit pipeline scripts."""
    return core_pack_root() / "scripts"


def commands_dir() -> Path:
    """Directory containing the bundled slash-command templates."""
    return core_pack_root() / "commands"


def mcp_server_path() -> Path:
    """Convenience: path to the bundled MCP server entry script."""
    return scripts_dir() / "mcp_server.py"
