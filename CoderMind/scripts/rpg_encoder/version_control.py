"""RPG Version Control.

Manages versioned snapshots of RPG state (``rpg.json``) inside the
``.cmind/data/history/`` directory.  Each snapshot is a self-contained
JSON file with metadata (version number, timestamp, message, source).

This is an **original** CoderMind module -- it is NOT ported from
RPG-ZeroRepo.

Key class:
  ``RPGVersionControl`` -- save / rollback / diff operations.

Typical usage::

    vc = RPGVersionControl(cmind_dir=".cmind")
    v = vc.save_version(rpg, message="Initial encode")
    old_rpg = vc.rollback(version=1)
    diff = vc.diff(version1=1, version2=2)
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rpg import RPG
from pathlib import Path
from common.rpg_io import atomic_write_rpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_DIR_NAME = "history"
RPG_FILE_NAME = "rpg.json"
DATA_DIR_NAME = "data"
VERSION_PREFIX = "rpg.v"
VERSION_SUFFIX = ".json"


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------


def _make_version_filename(version: int) -> str:
    """Return the filename for a given version number."""
    return f"{VERSION_PREFIX}{version}{VERSION_SUFFIX}"


def _parse_version_from_filename(filename: str) -> Optional[int]:
    """Extract the version number from a history filename.

    Returns ``None`` if the filename does not match the expected pattern.
    """
    basename = os.path.basename(filename)
    if basename.startswith(VERSION_PREFIX) and basename.endswith(VERSION_SUFFIX):
        version_str = basename[len(VERSION_PREFIX):-len(VERSION_SUFFIX)]
        try:
            return int(version_str)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# RPGVersionControl
# ---------------------------------------------------------------------------


class RPGVersionControl:
    """Manage versioned snapshots of the RPG.

    Versions are stored as ``<cmind_dir>/data/history/rpg.v<N>.json``
    where *N* is a monotonically increasing integer starting from 1.

    Each version file contains:
    - ``version``: int
    - ``message``: user-supplied description
    - ``timestamp``: ISO-8601 UTC datetime
    - ``source``: one of ``"generated"``, ``"encoded"``, ``"mixed"``
    - ``rpg``: the full RPG dict (``RPG.to_dict()``)

    Args:
        cmind_dir: Path to the ``.cmind`` directory.
        max_history: Maximum number of versions to keep (0 = unlimited).
    """

    def __init__(self, cmind_dir: str, max_history: int = 10):
        self.cmind_dir = os.path.abspath(cmind_dir)
        self.data_dir = os.path.join(self.cmind_dir, DATA_DIR_NAME)
        self.history_dir = os.path.join(self.data_dir, HISTORY_DIR_NAME)
        self.max_history = max_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_version(
        self,
        rpg: RPG,
        message: str = "",
        source: str = "encoded",
    ) -> int:
        """Save a new RPG version snapshot.

        Args:
            rpg: The RPG instance to snapshot.
            message: Human-readable description of this version.
            source: Origin label -- ``"generated"``, ``"encoded"``, or
                ``"mixed"``.

        Returns:
            The version number assigned to this snapshot.
        """
        os.makedirs(self.history_dir, exist_ok=True)

        next_version = self._next_version_number()
        filename = _make_version_filename(next_version)
        filepath = os.path.join(self.history_dir, filename)

        payload: Dict[str, Any] = {
            "version": next_version,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "rpg": rpg.to_dict(),
        }

        # Atomic write: a kill mid-save used to leave a truncated history
        # snapshot that ``rollback(version=N)`` could not parse. Aligns
        # with :meth:`rollback` which already uses ``atomic_write_rpg``
        # for the main rpg.json write.
        atomic_write_rpg(Path(filepath), payload, indent=2, ensure_ascii=False)

        logger.info(
            "Saved RPG version %d: %s (%s)",
            next_version,
            message or "(no message)",
            filepath,
        )

        # Enforce max_history by pruning the oldest versions
        if self.max_history > 0:
            self._prune_old_versions()

        return next_version

    def rollback(self, version: int) -> RPG:
        """Restore an RPG from a saved version.

        The restored RPG is also written to the main
        ``<data_dir>/rpg.json`` file so downstream tools can read it.

        Args:
            version: Version number to restore.

        Returns:
            The restored ``RPG`` instance.

        Raises:
            FileNotFoundError: If the requested version does not exist.
        """
        filepath = os.path.join(
            self.history_dir, _make_version_filename(version)
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Version {version} not found at {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        rpg = RPG.from_dict(payload["rpg"])

        # Also write to the main rpg.json so it becomes the "current" RPG.
        # Atomic write: a kill mid-rollback can't leave a half-truncated
        # rpg.json that bricks future reads.
        main_rpg_path = os.path.join(self.data_dir, RPG_FILE_NAME)
        os.makedirs(self.data_dir, exist_ok=True)
        atomic_write_rpg(Path(main_rpg_path), payload["rpg"])

        logger.info(
            "Rolled back to version %d (%s)",
            version,
            payload.get("message", ""),
        )
        return rpg

    def diff(self, version1: int, version2: int) -> Dict[str, Any]:
        """Compare two RPG versions and return a structural diff.

        The diff is a high-level summary, not a line-by-line textual diff.
        It reports:
        - ``nodes_added``: node IDs present in v2 but not v1
        - ``nodes_removed``: node IDs present in v1 but not v2
        - ``edges_added``: edges in v2 but not v1 (as dicts)
        - ``edges_removed``: edges in v1 but not v2 (as dicts)
        - ``metadata_changed``: node IDs whose metadata differs

        Args:
            version1: The base version number.
            version2: The target version number.

        Returns:
            Dictionary describing the structural differences.

        Raises:
            FileNotFoundError: If either version does not exist. Propagated
                from ``_load_version_data`` rather than raised directly here.
        """
        rpg1_data = self._load_version_data(version1)
        rpg2_data = self._load_version_data(version2)

        rpg1_dict = rpg1_data["rpg"]
        rpg2_dict = rpg2_data["rpg"]

        # Collect node IDs from both versions
        nodes1 = _collect_node_ids(rpg1_dict)
        nodes2 = _collect_node_ids(rpg2_dict)

        nodes_added = sorted(nodes2 - nodes1)
        nodes_removed = sorted(nodes1 - nodes2)

        # Collect edges
        edges1 = _collect_edge_tuples(rpg1_dict)
        edges2 = _collect_edge_tuples(rpg2_dict)

        edges_added = sorted(edges2 - edges1)
        edges_removed = sorted(edges1 - edges2)

        # Compare metadata of shared nodes
        metadata_changed = _compare_shared_node_metadata(rpg1_dict, rpg2_dict)

        return {
            "version1": version1,
            "version2": version2,
            "message1": rpg1_data.get("message", ""),
            "message2": rpg2_data.get("message", ""),
            "nodes_added": nodes_added,
            "nodes_removed": nodes_removed,
            "edges_added": [
                {"src": s, "dst": d, "relation": r} for s, d, r in edges_added
            ],
            "edges_removed": [
                {"src": s, "dst": d, "relation": r} for s, d, r in edges_removed
            ],
            "metadata_changed": metadata_changed,
            "summary": {
                "nodes_added": len(nodes_added),
                "nodes_removed": len(nodes_removed),
                "edges_added": len(edges_added),
                "edges_removed": len(edges_removed),
                "metadata_changed": len(metadata_changed),
            },
        }

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all saved versions with metadata (no RPG payload).

        Returns:
            List of dicts with ``version``, ``message``, ``timestamp``,
            ``source`` keys, sorted by version number ascending.
        """
        if not os.path.isdir(self.history_dir):
            return []

        versions: List[Dict[str, Any]] = []
        for fname in os.listdir(self.history_dir):
            v = _parse_version_from_filename(fname)
            if v is None:
                continue
            filepath = os.path.join(self.history_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                versions.append({
                    "version": data.get("version", v),
                    "message": data.get("message", ""),
                    "timestamp": data.get("timestamp", ""),
                    "source": data.get("source", ""),
                })
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping corrupt version file: %s", fname)

        versions.sort(key=lambda x: x["version"])
        return versions

    def get_latest_version(self) -> Optional[int]:
        """Return the highest version number, or ``None`` if no versions exist."""
        versions = self.list_versions()
        if not versions:
            return None
        return versions[-1]["version"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_version_number(self) -> int:
        """Determine the next version number."""
        latest = self.get_latest_version()
        return (latest or 0) + 1

    def _prune_old_versions(self) -> None:
        """Delete oldest versions that exceed ``max_history``."""
        if self.max_history <= 0:
            return

        versions = self.list_versions()
        if len(versions) <= self.max_history:
            return

        to_delete = versions[: len(versions) - self.max_history]
        for v_info in to_delete:
            fname = _make_version_filename(v_info["version"])
            filepath = os.path.join(self.history_dir, fname)
            try:
                os.remove(filepath)
                logger.info("Pruned old version %d", v_info["version"])
            except OSError as exc:
                logger.warning(
                    "Failed to prune version %d: %s", v_info["version"], exc
                )

    def _load_version_data(self, version: int) -> Dict[str, Any]:
        """Load a version's full JSON payload."""
        filepath = os.path.join(
            self.history_dir, _make_version_filename(version)
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Version {version} not found at {filepath}"
            )
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
# Diff helper functions (operate on raw RPG dicts)
# ---------------------------------------------------------------------------


def _collect_node_ids_from_tree(node_dict: Dict[str, Any]) -> set:
    """Recursively collect node IDs from a nested tree dict."""
    ids = set()
    if "id" in node_dict:
        ids.add(node_dict["id"])
    for child in node_dict.get("children", []):
        ids.update(_collect_node_ids_from_tree(child))
    return ids


def _collect_node_ids(rpg_dict: Dict[str, Any]) -> set:
    """Collect all node IDs from an RPG dict (handles both formats)."""
    ids = set()

    # Nested format: "root" key with "children"
    root = rpg_dict.get("root")
    if root:
        ids.update(_collect_node_ids_from_tree(root))

    # Flat format: "nodes" list
    for node in rpg_dict.get("nodes", []):
        if "id" in node:
            ids.add(node["id"])

    return ids


def _collect_edge_tuples(rpg_dict: Dict[str, Any]) -> set:
    """Collect edges as (src, dst, relation) tuples."""
    tuples = set()
    for edge in rpg_dict.get("edges", []):
        tuples.add((
            edge.get("src", ""),
            edge.get("dst", ""),
            str(edge.get("relation", "")),
        ))
    return tuples


def _collect_node_meta_from_tree(
    node_dict: Dict[str, Any],
) -> Dict[str, Optional[Dict]]:
    """Recursively collect {node_id: meta_dict} from a nested tree."""
    result: Dict[str, Optional[Dict]] = {}
    nid = node_dict.get("id")
    if nid:
        result[nid] = node_dict.get("meta")
    for child in node_dict.get("children", []):
        result.update(_collect_node_meta_from_tree(child))
    return result


def _collect_node_meta(rpg_dict: Dict[str, Any]) -> Dict[str, Optional[Dict]]:
    """Collect all node metadata from an RPG dict (both formats)."""
    result: Dict[str, Optional[Dict]] = {}

    root = rpg_dict.get("root")
    if root:
        result.update(_collect_node_meta_from_tree(root))

    for node in rpg_dict.get("nodes", []):
        nid = node.get("id")
        if nid:
            result[nid] = node.get("meta")

    return result


def _compare_shared_node_metadata(
    rpg1_dict: Dict[str, Any],
    rpg2_dict: Dict[str, Any],
) -> List[str]:
    """Return IDs of nodes whose metadata differs between two RPG dicts."""
    meta1 = _collect_node_meta(rpg1_dict)
    meta2 = _collect_node_meta(rpg2_dict)

    shared_ids = set(meta1.keys()) & set(meta2.keys())
    changed: List[str] = []
    for nid in sorted(shared_ids):
        if meta1[nid] != meta2[nid]:
            changed.append(nid)
    return changed
