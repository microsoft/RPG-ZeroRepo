"""CoderMind MCP Server.

Exposes RPG graph query tools via MCP (Model Context Protocol), allowing
AI assistants to search, explore, and inspect RPG graphs interactively.

Tools provided:
- ``search_rpg``       -- search nodes by keyword (substring + fuzzy)
- ``explore_rpg``      -- traverse dependency graph from a starting node
- ``get_node_detail``  -- get full attributes and optional source code
- ``list_rpg_tree``    -- browse RPG feature tree structure

The server communicates over stdio (the standard MCP transport for
CLI-based servers).  It ships inside the ``cmind-cli`` wheel and is
launched by MCP clients via the ``cmind-mcp`` console script (which
``.mcp.json`` / ``.vscode/mcp.json`` register as the ``rpg-tools``
command — see ``cmind_cli.entries:mcp_main``).

Run directly (for debugging)::

    cmind-mcp [--rpg-file PATH]
    # or equivalently:
    cmind script mcp_server.py [--rpg-file PATH]
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

# Ensure sibling packages (common/, rpg/) are importable when this script is
# invoked by an absolute path (which is how Claude / VS Code launch it).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common.paths import RPG_FILE, MCP_CALLS_LOG  # noqa: E402
from rpg.graph_query import GraphQueryEngine  # noqa: E402

logger = logging.getLogger(__name__)

# All logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


# ---------------------------------------------------------------------------
# Telemetry: append-only JSONL log of every tool call
# ---------------------------------------------------------------------------

def _log_tool_call(tool_name: str, params: dict, result_summary: dict, duration_ms: int) -> None:
    """Append a single-line JSON record to the MCP calls log.

    Best-effort: never raises; failures are silently ignored so
    telemetry never breaks a tool invocation.
    """
    try:
        MCP_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": tool_name,
            "params": params,
            **result_summary,
            "duration_ms": duration_ms,
        }
        with open(MCP_CALLS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_rpg_path() -> str:
    """Resolve the RPG file path from CLI args, falling back to the default.

    The default (``RPG_FILE``) is provided by
    :mod:`common.paths`, which resolves to
    ``~/.cmind/workspaces/<workspace-id>/data/rpg.json`` for the current
    workspace (discovered by walking up from cwd looking for
    ``.cmind/config.toml``).  Callers running ``cmind-mcp`` from any
    subdirectory of a workspace therefore get the right RPG file
    automatically; ``--rpg-file`` is reserved for explicit overrides
    (test fixtures, alternative graphs, …).
    """
    rpg_path = str(RPG_FILE)
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--rpg-file" and i + 1 < len(args):
            rpg_path = args[i + 1]
    return rpg_path


# Standard message returned to the AI agent when the RPG graph isn't ready
# (e.g. ``cmind init`` ran, but the encoder hasn't been run yet so
# the resolved ``rpg.json`` doesn't exist).  Kept short + actionable so
# the agent will relay it verbatim to the user.  The hint omits the
# concrete directory path; the actual location is reported as the
# ``rpg_file`` field of :func:`_unavailable_payload`.
_ENCODE_HINT = (
    "RPG graph not generated yet. Ask the user to run **`/cmind.encode`** "
    "in this AI agent to build the workspace's `rpg.json`. Once it finishes, "
    "RPG tools will start working automatically on the next call — no need "
    "to restart the MCP server."
)


def _unavailable_payload(rpg_path: str, reason: str) -> str:
    """Render a uniform 'graph not available' JSON response for every tool.

    The shape is identical across all 4 tools so the AI agent
    can reliably detect the condition (``error == "rpg_unavailable"``)
    and surface the ``next_step`` field to the user.
    """
    return json.dumps(
        {
            "error": "rpg_unavailable",
            "rpg_file": rpg_path,
            "reason": reason,
            "next_step": _ENCODE_HINT,
        },
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# MCP Server builder
# ---------------------------------------------------------------------------

def create_mcp_server(rpg_file: str):
    """Create and return a configured MCP server instance.

    Uses ``rpg.graph_query.GraphQueryEngine`` as the query backend.
    Registers 4 MCP tools: search, explore, detail, and tree.

    The engine is loaded **lazily**: if ``rpg_file`` doesn't yet exist
    (typical first-run flow — ``cmind init`` finished but the user
    hasn't run the encoder yet), the server still starts cleanly and
    every tool returns an actionable ``rpg_unavailable`` payload pointing
    the user at ``/cmind.encode``.  Once the encoder writes
    ``rpg.json`` the next tool call picks it up automatically — no
    restart needed.  This avoids the ``MCP error -32000: Connection
    closed`` failure mode that used to happen when the server exited
    during startup.

    Args:
        rpg_file: Path to the RPG JSON file.

    Returns:
        A ``FastMCP`` server instance ready to be run.
    """
    from mcp.server.fastmcp import FastMCP

    # Single-element list used as a mutable box so the per-tool closures
    # below can update the cached engine without needing ``nonlocal`` in
    # each function.
    engine_box: List[Optional[GraphQueryEngine]] = [None]

    def _get_engine() -> Optional[GraphQueryEngine]:
        """Return the cached engine, lazily loading rpg.json on first use.

        Returns ``None`` if the file doesn't exist or fails to load.
        Errors are logged to stderr — never raised — because raising
        from a tool handler closes the MCP transport.
        """
        if engine_box[0] is not None:
            return engine_box[0]
        if not os.path.isfile(rpg_file):
            return None
        try:
            engine_box[0] = GraphQueryEngine.from_rpg_file(rpg_file)
            logger.info("Loaded RPG from %s", rpg_file)
            return engine_box[0]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load RPG from %s: %s", rpg_file, exc)
            return None

    def _unavailable_reason() -> str:
        return (
            "file_not_found"
            if not os.path.isfile(rpg_file)
            else "load_failed"
        )

    mcp = FastMCP(
        "rpg-tools",
        instructions=(
            "This server provides structured access to the Repository "
            "Program Graph (RPG) for the current workspace \u2014 a "
            "pre-computed, queryable index of the codebase built by "
            "`/cmind.encode` and kept in sync with HEAD by a "
            "post-commit hook.\n\n"
            "What the RPG knows about this repository:\n"
            "  \u2022 The feature hierarchy: functional areas \u2192 "
            "feature groups \u2192 individual features, each linked to "
            "the source files that implement it.\n"
            "  \u2022 Every code entity: files, classes, and functions "
            "with their signatures, docstrings, and exact line ranges.\n"
            "  \u2022 Resolved dependency edges between entities: "
            "invokes (call graph), inherits, imports, contains.\n\n"
            "What you can ask it for (and which tool answers it):\n"
            "  \u2022 The project's architecture \u2014 what each "
            "functional area does, without reading any file. "
            "\u2192 `list_rpg_tree`\n"
            "  \u2022 The definition site of any symbol (function, "
            "class, file) by name or keyword. \u2192 `search_rpg`\n"
            "  \u2022 The callers and callees of a function, or its "
            "full reachable subgraph up to N hops. \u2192 `explore_rpg`\n"
            "  \u2022 The full signature, docstring, and optional "
            "source of a specific entity. \u2192 `get_node_detail`\n"
            "  \u2022 The mapping between abstract concerns (e.g. "
            "\"authentication\", \"caching\") and the concrete code "
            "that implements them. \u2192 `search_rpg` with "
            "`scope=\"feature\"`, then `get_node_detail` on the "
            "feature node.\n\n"
            "Tools provided:\n"
            "  \u2022 `list_rpg_tree(root_id, max_depth)` \u2014 "
            "browse the feature tree (functional areas \u2192 groups "
            "\u2192 features). Best entry point for unfamiliar "
            "codebases.\n"
            "  \u2022 `search_rpg(query, scope, top_k)` \u2014 "
            "keyword search over code entities, features, or both; "
            "returns ranked node IDs.\n"
            "  \u2022 `explore_rpg(node_id, direction, depth, "
            "edge_types)` \u2014 traverse the dependency graph "
            "upstream / downstream / both from a node, with edge-type "
            "filtering.\n"
            "  \u2022 `get_node_detail(node_id, include_code)` \u2014 "
            "full attributes of one node: signature, callers, callees, "
            "line ranges, optional source code.\n\n"
            "These tools resolve references semantically and aggregate "
            "them by feature, so they answer structural and "
            "dependency questions far more directly than a text scan. "
            "See each tool's description for parameters and output "
            "shape.\n\n"
            "If a tool returns `error: \"rpg_unavailable\"`, the graph "
            "has not been built yet \u2014 relay the `next_step` field "
            "to the user."
        ),
    )

    # ------------------------------------------------------------------
    # Tool 1: search_rpg
    # ------------------------------------------------------------------
    @mcp.tool()
    def search_rpg(
        query: str,
        scope: str = "all",
        top_k: int = 10,
    ) -> str:
        """Search for code entities or features in this project by keyword.

        Use this when the user asks 'where is X?', 'find the login function',
        'which module handles authentication?', or any question that requires
        locating code or features by name.

        Scope guide:
        - 'code': find functions, classes, files by name or path
        - 'feature': find functional features (e.g. 'authentication', 'data persistence')
        - 'all': search both (recommended when unsure)

        Args:
            query: Search keyword — function name, class name, file path, or feature name.
            scope: 'code' (code entities), 'feature' (functional features), or 'all' (both).
            top_k: Maximum number of results (default 10).

        Returns:
            JSON list of matching nodes with id, name, type, score.
        """
        engine = _get_engine()
        if engine is None:
            return _unavailable_payload(rpg_file, _unavailable_reason())
        t0 = time.monotonic()
        results = engine.search(query, scope=scope, top_k=top_k)
        has_error = bool(results and isinstance(results[0], dict) and "error" in results[0])
        _log_tool_call("search_rpg",
                       {"query": query, "scope": scope, "top_k": top_k},
                       {"results": 0 if has_error else len(results), "error": has_error},
                       int((time.monotonic() - t0) * 1000))
        return json.dumps(results, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool 2: explore_rpg
    # ------------------------------------------------------------------
    @mcp.tool()
    def explore_rpg(
        node_id: str,
        direction: str = "both",
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> str:
        """Explore dependencies and call chains from a code entity.

        Use this when the user asks 'what does X call?', 'who calls X?',
        'what are the dependencies of X?', or 'show me the call chain'.
        Returns the subgraph of connected nodes and edges.

        Args:
            node_id: Starting node ID (from search_rpg results, e.g. 'routes/auth.py:login').
            direction: 'downstream' (what I call), 'upstream' (who calls me), or 'both'.
            depth: Maximum traversal depth in hops (default 2).
            edge_types: Filter by edge types like 'invokes', 'inherits', 'imports'. Default: all.

        Returns:
            JSON with connected nodes and edges.
        """
        engine = _get_engine()
        if engine is None:
            return _unavailable_payload(rpg_file, _unavailable_reason())
        t0 = time.monotonic()
        result = engine.explore(
            node_id, direction=direction, depth=depth, edge_types=edge_types
        )
        _log_tool_call("explore_rpg",
                       {"node_id": node_id, "direction": direction, "depth": depth},
                       {"nodes": result.get("total_nodes", 0), "edges": result.get("total_edges", 0)},
                       int((time.monotonic() - t0) * 1000))
        return json.dumps(result, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool 3: get_node_detail
    # ------------------------------------------------------------------
    @mcp.tool()
    def get_node_detail(
        node_id: str,
        include_code: bool = False,
    ) -> str:
        """Get full details about a specific function, class, or feature.

        Use this when the user asks 'show me the signature of X', 'what does X do?',
        'what are the parameters of X?', or needs the source code of a specific entity.
        Also works for RPG feature nodes (functional areas, feature groups).

        Args:
            node_id: Node ID (from search_rpg or explore_rpg results).
            include_code: If true, include the full source code of the function/class.

        Returns:
            JSON with all node attributes: signature, calls, called_by, line numbers, etc.
        """
        engine = _get_engine()
        if engine is None:
            return _unavailable_payload(rpg_file, _unavailable_reason())
        t0 = time.monotonic()
        result = engine.get_node_detail(node_id, include_code=include_code)
        _log_tool_call("get_node_detail",
                       {"node_id": node_id, "include_code": include_code},
                       {"source": result.get("source", "error"), "found": "error" not in result},
                       int((time.monotonic() - t0) * 1000))
        return json.dumps(result, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool 4: list_rpg_tree
    # ------------------------------------------------------------------
    @mcp.tool()
    def list_rpg_tree(
        root_id: str = "",
        max_depth: int = 2,
    ) -> str:
        """List the project's functional architecture as a tree.

        Shows how the codebase is organized: functional areas (top-level domains),
        feature groups, and individual features — each linked to source files.

        Use this FIRST when the user asks about project structure, module organization,
        or wants an overview of what the codebase does.

        Args:
            root_id: Start from this node ID (empty = full project). Use a functional_area ID to zoom into one domain.
            max_depth: How many levels deep to show (1=areas only, 2=+groups, 3=+features with file paths).

        Returns:
            JSON tree with node names, types, and paths.
        """
        engine = _get_engine()
        if engine is None:
            return _unavailable_payload(rpg_file, _unavailable_reason())
        t0 = time.monotonic()
        result = engine.list_tree(root_id=root_id or None, max_depth=max_depth)
        _log_tool_call("list_rpg_tree",
                       {"root_id": root_id, "max_depth": max_depth},
                       {"total_nodes": result.get("total_nodes", 0)},
                       int((time.monotonic() - t0) * 1000))
        return json.dumps(result, indent=2, ensure_ascii=False)

    return mcp


# ---------------------------------------------------------------------------
# Entry point: ``cmind-mcp`` console script (via cmind_cli.entries:mcp_main)
# or direct ``python <scripts_dir>/mcp_server.py [--rpg-file PATH]`` for
# debugging.
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server over stdio.

    Used by both the ``cmind-mcp`` console-script entry (which sets up
    ``sys.path`` then imports and calls this function) and the direct
    ``python mcp_server.py`` invocation under ``__main__``.
    """
    rpg_path = _resolve_rpg_path()
    # NOTE: do NOT sys.exit when the file is missing.  The MCP transport
    # must stay up so the client can actually receive the
    # ``rpg_unavailable`` hint that tells the user to run
    # ``/cmind.encode``.  Exiting here used to surface as the opaque
    # ``MCP error -32000: Connection closed`` on the client side.
    if not os.path.isfile(rpg_path):
        logger.warning(
            "RPG file not found: %s — server will start in degraded mode "
            "and instruct the user to run /cmind.encode on the first tool call.",
            rpg_path,
        )

    server = create_mcp_server(rpg_file=rpg_path)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
