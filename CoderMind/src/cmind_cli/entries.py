"""Console-script entries for ``cmind-cli``.

Currently provides:

* :func:`mcp_main` — the ``cmind-mcp`` console script.  Sets up
  ``sys.path`` so that the bundled ``scripts/`` directory is importable,
  then hands off to ``mcp_server.main()``.

This module stays small and stdout-silent because MCP uses stdio as
its transport: anything written to stdout from import-time code would
corrupt the JSON-RPC stream.  All diagnostics go to stderr.
"""

from __future__ import annotations


def mcp_main() -> None:
    """Console-script entry for MCP clients (stdio transport)."""
    import os
    import sys

    from . import _assets

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    scripts_dir = _assets.scripts_dir()
    if not scripts_dir.is_dir():
        sys.stderr.write(
            "cmind-mcp: packaged scripts directory unavailable. "
            "Try reinstalling: `uv tool install cmind-cli --force`.\n"
        )
        sys.exit(2)

    # Make ``mcp_server`` and its sibling packages (``common``, ``rpg``)
    # importable from the packaged scripts dir.
    sys.path.insert(0, str(scripts_dir))

    try:
        from mcp_server import main as _mcp_server_main  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import-time failure surface
        sys.stderr.write(f"cmind-mcp: failed to import mcp_server: {exc}\n")
        sys.exit(3)

    _mcp_server_main()
