"""RPGService — high-level RPG operations interface for pipeline stages.

Provides semantic RPG operation methods so individual stages don't have
to manipulate the RPG's internal structure directly.  All mutating ops
have built-in deduplication, validation, and generator tagging.

Typical usage::

    from rpg.service import RPGService

    svc = RPGService.load('.cmind/data/repo_rpg.json')
    svc.refresh_stage_edges("build_data_flow")
    src = svc.find_functional_area_by_name("Data Models")
    dst = svc.find_functional_area_by_name("Auth Routes")
    svc.add_dependency_edge(src, dst, EdgeType.REFERENCES, "build_data_flow",
                            description="User model provides auth data")
    svc.save('.cmind/data/repo_rpg.json')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
from .models import RPG, Node, Edge, EdgeType, NodeType, NodeMetaData, strip_uuid8, uuid8

if TYPE_CHECKING:
    from .dep_graph import DependencyGraph


class RPGService:
    """Operation service layer over an RPG graph.

    Encapsulates the recurring operation patterns each pipeline stage
    needs against the RPG:

    - Idempotent edge refresh.
    - Dependency-edge insertion with deduplication.
    - generator-aware node cleanup.
    - Bulk feature metadata updates.
    - Policy-driven orphan pruning.
    """

    def __init__(self, rpg: RPG):
        self.rpg = rpg
        self._rpg_dir: Path = Path(".")  # set properly by load()

    # === Load / save ===

    @classmethod
    def load(cls, path: str | Path) -> "RPGService":
        """Load an RPG from a file and create a service instance.

        Read order (single-source-of-truth):

        1. **Embedded** dep_graph — ``RPG.load_json`` already restored it
           from ``data["dep_graph"]`` if present.  This is the new
           default since the dep_graph rides inside ``rpg.json``.
        2. **External** dep_graph — only consulted when no embedded copy
           was found AND the legacy ``_dep_graph_file`` pointer is set
           AND the file exists.  Emits a single INFO log so the
           fall-through is visible during debugging.

        The fall-through path keeps pre-embed-migration workspaces
        readable.  New encodes never produce a standalone
        ``dep_graph.json`` so this path naturally goes cold.
        """
        _logger = logging.getLogger(__name__)

        rpg = RPG.load_json(str(path))
        svc = cls(rpg)
        svc._rpg_dir = Path(path).parent

        if rpg.dep_graph is not None:
            # Embedded copy already attached by RPG.load_json — done.
            pass
        elif rpg._dep_graph_file:
            dgp = svc._rpg_dir / rpg._dep_graph_file
            if dgp.exists():
                _logger.info(
                    "Loading dep_graph from legacy external file %s; "
                    "next save will embed it inside rpg.json.",
                    dgp,
                )
                rpg.dep_graph = RPG.load_dep_graph(dgp)
                rpg.rebuild_cross_maps()
        return svc

    def save(self, path: str | Path) -> None:
        """Save the RPG to a file."""
        self.rpg.save_json(str(path))

    def save_snapshot(self, snapshot_path: str | Path) -> None:
        """Save an RPG snapshot to the given path (for cross-stage debug diffing)."""
        self.rpg.save_json(str(snapshot_path))

    # === Edge operations (unified entry points) ===

    def refresh_stage_edges(self, generator: str) -> int:
        """Drop every edge owned by ``generator`` (idempotent refresh).  Returns count removed."""
        before = len(self.rpg.edges)
        self.rpg.remove_edges_by_generator(generator)
        return before - len(self.rpg.edges)

    def refresh_file_edges(self, generator: str, file_path: str) -> int:
        """Drop edges owned by ``generator`` for a single file (used by code_gen).  Returns count removed."""
        return self.rpg.remove_edges_by_generator_and_file(generator, file_path)

    def add_dependency_edge(
        self,
        src_node: Node,
        dst_node: Node,
        relation: EdgeType,
        generator: str,
        description: str = "",
        content: str = "",
        *,
        dedup: bool = True,
        bidirectional_dedup: bool = False,
    ) -> bool:
        """Add a non-hierarchical dependency edge (with deduplication).

        Args:
            src_node: source node
            dst_node: target node
            relation: edge type (REFERENCES / INVOKES / INHERITS / SAME_UNIT, ...)
            generator: creator identifier
            description: edge description
            content: extra content
            dedup: whether to deduplicate by signature (default True)
            bidirectional_dedup: also check the reverse edge (needed for SAME_UNIT)

        Returns:
            True if the edge was actually added.
        """
        if dedup:
            existing = self.rpg.find_edge_by_signature(
                strip_uuid8(src_node.id),
                strip_uuid8(dst_node.id),
                relation,
            )
            if existing:
                return False
            if bidirectional_dedup:
                existing_rev = self.rpg.find_edge_by_signature(
                    strip_uuid8(dst_node.id),
                    strip_uuid8(src_node.id),
                    relation,
                )
                if existing_rev:
                    return False

        edge = Edge(
            src=src_node.id,
            dst=dst_node.id,
            relation=relation,
            meta=NodeMetaData(
                description=description,
                content=content,
                generator=generator,
            ),
        )
        self.rpg.edges.append(edge)
        return True

    # === Node operations ===

    def add_feature_node(
        self,
        name: str,
        parent: Node,
        *,
        impl_path: str = "",
        type_name: Optional[NodeType] = None,
        generator: str = "",
        description: str = "",
    ) -> Node:
        """Create a feature node and attach it to the tree.

        Unified replacement for the combined ``rpg.nodes[id] = node``
        + ``parent.add_child(node)`` pattern.  Generates a unique id
        automatically (format: ``"{name}_{uuid8}"``).

        Args:
            name: node name
            parent: parent node
            impl_path: implementation path (e.g. ``"src/foo.py::class Bar"``)
            type_name: code entity type
            generator: creator identifier
            description: node description

        Returns:
            The freshly created Node.
        """
        node_id = f"{name}_{uuid8()}"
        node = Node(
            id=node_id,
            name=name,
            node_type="feature",
            meta=NodeMetaData(
                path=impl_path or None,
                type_name=type_name,
                description=description,
                generator=generator,
            ),
        )
        node.level = parent.level + 1 if parent.level is not None else self.rpg.MAX_FEATURE_LEVEL
        self.rpg.add_node(node)
        parent.add_child(node)
        return node

    def refresh_stage_nodes(self, generator: str) -> int:
        """Drop every node owned by ``generator`` (idempotent refresh).  Returns count removed."""
        before = len(self.rpg.nodes)
        self.rpg.remove_nodes_by_generator(generator)
        return before - len(self.rpg.nodes)

    # === Node lookup (unified entry points) ===

    def find_node_by_unit_name(self, name: str) -> Optional[Node]:
        """Generic node lookup: match against ``meta.path`` unit part or ``node.name``.

        Unified replacement for the three duplicate implementations in
        ``design_interfaces._find_node_by_name``,
        ``interfaces_store._find_rpg_node`` and
        ``rpg_updater._find_node_by_name``.

        Match order:

        1. Exact match on the unit part of ``meta.path`` after ``::``
           (e.g. ``"class Foo"``).
        2. Bare-name match on the unit part of ``meta.path``
           (e.g. ``"Foo"``).
        3. Exact match on ``node.name``.
        4. Qualified-name suffix match
           (e.g. ``"ClassName.method"`` → ``"method"``).
        """
        normalized = name
        if name.startswith("class "):
            normalized = name[6:]
        elif name.startswith("function "):
            normalized = name[9:]

        # Handle :: format (e.g. "ClassName::method_name")
        if "::" in normalized:
            normalized = normalized.split("::")[-1]

        # 1 & 2: meta.path match
        for node in self.rpg.nodes.values():
            if not node.meta or not node.meta.path:
                continue
            path_str = node.meta.path if isinstance(node.meta.path, str) else ""
            if "::" not in path_str:
                continue
            unit_part = path_str.split("::", 1)[-1]
            # Exact match: "class Foo" == "class Foo"
            if unit_part == name:
                return node
            # Prefix-tagged exact match
            if unit_part == f"class {normalized}" or unit_part == f"function {normalized}":
                return node
            # Bare-name match
            if " " in unit_part:
                bare = unit_part.split(" ", 1)[-1]
                if bare == normalized:
                    return node

        # 3: node.name match
        for node in self.rpg.nodes.values():
            if node.name == normalized:
                return node

        # 4: qualified-name suffix match
        if "." in name:
            short_name = name.rsplit(".", 1)[-1]
            for node in self.rpg.nodes.values():
                if node.name == short_name:
                    return node

        return None

    def find_functional_area_by_name(self, name: str) -> Optional[Node]:
        """Find an L1 functional_area node (case-insensitive fallback)."""
        for node in self.rpg.nodes.values():
            if node.node_type == "functional_area":
                if node.name == name or node.name.lower() == name.lower():
                    return node
        return None

    def find_feature_node(self, feature_path: str) -> Optional[Node]:
        """Look up a feature node by feature_path() or node.name."""
        for node in self.rpg.nodes.values():
            if node.node_type == "feature" or node.level == self.rpg.MAX_FEATURE_LEVEL:
                if node.feature_path() == feature_path or node.name == feature_path:
                    return node
                # Compare only the trailing name segment
                if "/" in feature_path:
                    last = feature_path.rsplit("/", 1)[-1]
                    if node.name == last:
                        return node
        return None

    # === Feature metadata updates ===

    def update_feature_mapping(
        self,
        feature_node: Node,
        impl_path: str,
        type_name: Optional[NodeType] = None,
    ) -> None:
        """Update a feature node's implementation mapping.

        Args:
            feature_node: target feature node
            impl_path: implementation path, e.g. ``"src/foo.py::class Bar"``
            type_name: code entity type (class / function / method)
        """
        if feature_node.meta is None:
            feature_node.meta = NodeMetaData()
        feature_node.meta.path = impl_path
        if type_name is not None:
            feature_node.meta.type_name = type_name

    def mark_implemented(self, feature_node: Node, source_code: str = "") -> None:
        """Mark a feature node as implemented (used by the code_gen stage)."""
        if feature_node.meta is None:
            feature_node.meta = NodeMetaData()
        desc = feature_node.meta.description or ""
        if "[implemented]" not in desc:
            feature_node.meta.description = f"{desc} [implemented]".strip()
        if source_code:
            feature_node.meta.content = source_code

    def mark_entry_point(self, node: Node, rationale: str = "") -> None:
        """Mark a node as an entry point (used by the design_interfaces stage)."""
        if node.meta is None:
            node.meta = NodeMetaData()
        marker = f"[ENTRY_POINT] {rationale}".strip()
        if node.meta.description:
            if "[ENTRY_POINT]" not in node.meta.description:
                node.meta.description += f" | {marker}"
        else:
            node.meta.description = marker

    # === Pruning ===

    def prune_orphan_features(
        self,
        surviving_paths: Set[str],
        protected_generators: Optional[Set[str]] = None,
    ) -> Tuple[int, int, int]:
        """Prune orphan feature nodes that have no surviving interface.

        Args:
            surviving_paths: set of ``feature_path`` values that still
                have implementation units.
            protected_generators: extra ``generator`` values to protect.
                Default behavior: any node whose ``generator !=
                "build_skeleton"`` is protected automatically.

        Returns:
            ``(pruned_features, pruned_parents, pruned_edges)``
        """
        if protected_generators is None:
            protected_generators = set()

        if not surviving_paths:
            return (0, 0, 0)

        # Identify candidates
        nodes_to_remove = {}
        for node in self.rpg.nodes.values():
            if node.node_type != "feature" and node.level != self.rpg.MAX_FEATURE_LEVEL:
                continue
            # Protect nodes not created by build_skeleton
            # (e.g. those produced by design_base_classes)
            if node.meta and node.meta.generator and node.meta.generator != "build_skeleton":
                if node.meta.generator not in protected_generators:
                    continue
            fp = node.feature_path()
            if fp and fp in surviving_paths:
                continue
            if node.name in surviving_paths:
                continue
            nodes_to_remove[node.id] = node

        if not nodes_to_remove:
            return (0, 0, 0)

        removed_ids = set(nodes_to_remove.keys())
        feat_count = len(nodes_to_remove)

        # Remove the nodes
        for nid in list(removed_ids):
            node = self.rpg.nodes.pop(nid, None)
            if node:
                parent = node.parent()
                if parent:
                    parent.remove_child(node)

        # Cascade-clean empty parents (same protection rule applies)
        pruned_parents = 0
        parent_types = {"feature_group", "category", "subcategory", "functional_area"}
        changed = True
        while changed:
            changed = False
            for nid in list(self.rpg.nodes.keys()):
                node = self.rpg.nodes.get(nid)
                if not node or str(node.node_type) not in parent_types:
                    continue
                if node.meta and node.meta.generator and node.meta.generator != "build_skeleton":
                    continue
                if not node.children():
                    removed_ids.add(nid)
                    self.rpg.nodes.pop(nid, None)
                    parent = node.parent()
                    if parent:
                        parent.remove_child(node)
                    pruned_parents += 1
                    changed = True

        # Drop dangling edges
        before = len(self.rpg.edges)
        self.rpg.edges = [
            e for e in self.rpg.edges
            if e.src not in removed_ids and e.dst not in removed_ids
        ]
        pruned_edges = before - len(self.rpg.edges)

        return (feat_count, pruned_parents, pruned_edges)

    # === Dep Graph operations ===

    def refresh_dep_graph(
        self,
        code_dir: str,
        workspace_root: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ) -> "DependencyGraph":
        """Rebuild dep_graph from source code via AST analysis.

        Args:
            code_dir: Absolute path to the directory to scan.
            workspace_root: Base for computing ``_dep_graph_code_dir``
                relative prefix.  Defaults to ``os.getcwd()``.
            save_path: If provided, persist dep_graph to this file and
                set ``_dep_graph_file`` on the RPG accordingly.

        Returns:
            The newly built ``DependencyGraph``.
        """
        import os
        from .dep_graph import DependencyGraph

        if workspace_root is None:
            workspace_root = os.getcwd()

        dg = DependencyGraph(code_dir)
        dg.build()
        dg.parse()

        # ``relpath`` returns ``"."`` when ``code_dir == workspace_root``;
        # normalise to ``""`` so downstream ``if prefix:`` checks behave
        # correctly (otherwise prefix becomes ``"./"`` and we'd strip the
        # wrong number of characters from dep_graph paths).
        _rel = os.path.relpath(code_dir, workspace_root)
        self.rpg._dep_graph_code_dir = "" if _rel == "." else _rel
        self.rpg.set_dep_graph(dg)  # builds _dep_to_rpg_map + rebuild_cross_maps

        if save_path is not None:
            save_path = Path(save_path).resolve()
            # Ensure the parent directory exists so callers can pass an
            # arbitrary location without having to mkdir first.  This
            # was a sharp edge that bit Step 4b (the encoder's
            # ``_update_dep_graph_index`` invokes us with an external
            # tmpdir).
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.rpg.save_dep_graph(save_path)
            # Prefer a relative path from the RPG file's directory so
            # the persisted reference is portable, but fall back to an
            # absolute path when ``save_path`` lives outside the RPG
            # directory (the case when callers haven't called
            # :meth:`RPGService.load` — they get the default
            # ``_rpg_dir = Path(".")`` which is rarely a parent of
            # arbitrary tmpdirs / external save paths).
            try:
                self.rpg._dep_graph_file = str(
                    save_path.relative_to(self._rpg_dir.resolve())
                )
            except ValueError:
                self.rpg._dep_graph_file = str(save_path)

        return dg

    def load_dep_graph(self, dep_graph_path: str | Path) -> None:
        """Load an existing dep_graph from file without re-running AST analysis.

        Only rebuilds ``_feature_to_dep_map`` from the persisted
        ``_dep_to_rpg_map``; does **not** recompute the mapping.
        """
        dep_graph_path = Path(dep_graph_path)
        self.rpg.dep_graph = RPG.load_dep_graph(dep_graph_path)
        self.rpg.rebuild_cross_maps()

    def merge_dep_graph_from(
        self,
        source_rpg_path: str | Path,
        save_dep_graph_path: Optional[str | Path] = None,
    ) -> Dict[str, int]:
        """Import dep_graph from another RPG file (e.g. encoder output).

        Loads the source RPG, extracts its dep_graph, attaches it to
        the current RPG, rebuilds all mappings, and optionally persists
        the dep_graph to an independent file.

        Args:
            source_rpg_path: Path to the source RPG (must contain dep_graph).
            save_dep_graph_path: If provided, save dep_graph to this file
                and set ``_dep_graph_file`` on the current RPG.

        Returns:
            Dict with merge statistics: ``dep_to_rpg``, ``feature_to_dep``,
            ``dep_nodes``, ``dep_edges``.
        """
        source_rpg = RPG.load_json(str(source_rpg_path))
        if source_rpg.dep_graph is None:
            raise ValueError(f"Source RPG {source_rpg_path} has no dep_graph")
        self.rpg._dep_graph_code_dir = source_rpg._dep_graph_code_dir
        self.rpg.set_dep_graph(source_rpg.dep_graph)

        if save_dep_graph_path is not None:
            save_dep_graph_path = Path(save_dep_graph_path).resolve()
            self.rpg.save_dep_graph(save_dep_graph_path)
            try:
                self.rpg._dep_graph_file = str(
                    save_dep_graph_path.relative_to(self._rpg_dir.resolve())
                )
            except ValueError:
                self.rpg._dep_graph_file = str(save_dep_graph_path)

        return {
            "dep_to_rpg": len(self.rpg._dep_to_rpg_map),
            "feature_to_dep": len(self.rpg._feature_to_dep_map),
            "dep_nodes": len(self.rpg.dep_graph.G.nodes()),
            "dep_edges": len(self.rpg.dep_graph.G.edges()),
        }

    # === Commit-based incremental sync ===

    # When more than this many .py files have changed, the cost of
    # cross-file edge re-inference dominates the AST-parse savings, so
    # we fall back to a full rebuild.  Tuned for medium repos (~1k files);
    # callers can override via the ``file_limit`` parameter.
    DEFAULT_INCREMENTAL_FILE_LIMIT: int = 50

    # Environment variable that disables `meta.git` writes during sync.
    # Set in CI environments where the RPG is committed and you don't
    # want every CI run to advance `head_commit` to ephemeral CI commits.
    _NO_GIT_META_ENV = "CMIND_NO_GIT_META"

    def sync_from_commit_diff(
        self,
        code_dir: str,
        workspace_root: str,
        save_path: Optional[str | Path] = None,
        *,
        file_limit: Optional[int] = None,
        staged_only: bool = False,
        force_full: bool = False,
    ) -> Dict:
        """Sync dep_graph against current code, choosing incremental vs full.

        Decision tree (commit-based; ``branch`` is informational only):

        =========================  =========================================
        Condition                  Mode
        =========================  =========================================
        ``force_full=True``        full
        ``meta.git`` missing       full  (baseline)
        ``last == HEAD`` & clean   noop
        ``last == HEAD`` & dirty   incremental (using dirty file set)
        Linear advance             incremental (using ``last..HEAD`` diff)
        Diverged / rebase / amend  full
        ``len(changed) > limit``   full  (fallback safety net)
        =========================  =========================================

        After a successful run, advances ``meta.git`` to the current HEAD
        (unless ``CMIND_NO_GIT_META=1`` or the workspace isn't a git
        repo).  When ``save_path`` is provided the dep_graph is also
        persisted as a standalone JSON (legacy behaviour preserved for
        callers that still want the sidecar); when ``save_path is None``
        the dep_graph lives only in ``self.rpg.dep_graph`` and rides
        inside ``rpg.json`` via the caller's ``svc.save(rpg_path)``.

        Args:
            code_dir: Absolute path to the directory that ``DependencyGraph``
                should scan (typically the workspace root — workspace == repo).
            workspace_root: Absolute path to the git working tree.  Used
                both to read ``meta.git``'s sibling HEAD and to compute
                the relative prefix on dep_graph paths.
            save_path: Optional standalone output path for the dep_graph.
                ``None`` is the new default for callers that rely on the
                embedded dep_graph in ``rpg.json``.
            file_limit: Cap on changed-file count before falling back to
                full.  Defaults to :attr:`DEFAULT_INCREMENTAL_FILE_LIMIT`.
            staged_only: If ``True``, restrict diff to ``git diff --cached``
                — i.e. the pre-commit hook scope.  If ``False``
                (default), include working tree changes as well.
            force_full: Bypass the decision tree and run a full rebuild.

        Returns:
            A diagnostic dict (always includes ``mode``; other keys
            depend on which path ran).  Designed to be JSON-friendly so
            callers can echo it to stdout for hooks.
        """
        from common.git_utils import (
            read_head,
            staged_changes,
            working_tree_changes,
            changed_files_between,
            merge_base,
        )

        limit = file_limit if file_limit is not None else self.DEFAULT_INCREMENTAL_FILE_LIMIT
        save_path = str(save_path) if save_path is not None else None

        # ── Step 1: read current HEAD (silent-fail outside a git repo) ──
        current = read_head(workspace_root)
        # ``last_commit`` may be None for:
        #   * fresh RPG produced by /cmind.build_skeleton
        #   * legacy RPG without meta.git
        #   * workspace not under git (current is None too)
        last_commit = (self.rpg.git_meta or {}).get("head_commit")
        current_commit = current.get("head_commit") if current else None

        # ── Step 2: decide mode ──
        mode_reason = ""
        if force_full:
            mode = "full"
            mode_reason = "force_full"
        elif last_commit is None or current_commit is None:
            mode = "full"
            mode_reason = "baseline" if last_commit is None else "no_git"
        else:
            # Determine the file set we'd act on for incremental.
            if last_commit == current_commit:
                # HEAD unchanged — only dirty / staged files matter.
                if staged_only:
                    changed, renames = staged_changes(workspace_root)
                else:
                    changed, renames = working_tree_changes(workspace_root)
                if not changed and not renames:
                    mode = "noop"
                    mode_reason = "head_unchanged_clean"
                else:
                    mode = "incremental"
                    mode_reason = "head_unchanged_dirty"
            else:
                base = merge_base(workspace_root, last_commit, current_commit)
                if base != last_commit:
                    # Diverged / rebase / amend / reset / shallow.
                    mode = "full"
                    mode_reason = "diverged"
                    changed, renames = [], {}
                else:
                    # Linear advance — diff last..HEAD plus any staged /
                    # dirty extras the caller wants picked up.
                    changed, renames = changed_files_between(
                        workspace_root, last_commit, current_commit
                    )
                    if staged_only:
                        staged_extra, staged_renames = staged_changes(workspace_root)
                    else:
                        staged_extra, staged_renames = working_tree_changes(workspace_root)
                    # Merge without duplicates; renames from the broader
                    # scope take precedence over commit-range ones since
                    # they describe the most recent moves.
                    for p in staged_extra:
                        if p not in changed:
                            changed.append(p)
                    renames.update(staged_renames)
                    mode = "incremental"
                    mode_reason = "linear"

                if mode == "incremental" and len(changed) > limit:
                    mode = "full"
                    mode_reason = f"over_limit_{len(changed)}>{limit}"

        # ── Step 3: execute the chosen mode ──
        result: Dict = {
            "mode": mode,
            "reason": mode_reason,
            "last_commit": last_commit,
            "current_commit": current_commit,
            "save_path": save_path,
        }

        if mode == "noop":
            # Even though no graph edits are needed, the user may have
            # switched branches without moving HEAD (e.g.
            # ``git checkout other_branch_at_same_sha`` or
            # ``git branch -m new-name``), in which case ``meta.git``'s
            # ``head_branch`` / ``head_timestamp`` are stale.  Refresh
            # them in-place so subsequent ``status`` output reflects
            # reality.  ``head_commit`` is unchanged so this never
            # advances the "what code was last synced" signal.
            import os as _os
            if (
                current_commit
                and self.rpg.git_meta is not None
                and not _os.environ.get(self._NO_GIT_META_ENV)
            ):
                old_meta = self.rpg.git_meta
                new_branch = current.get("head_branch") if current else None
                new_timestamp = current.get("head_timestamp") if current else None
                if (
                    old_meta.get("head_branch") != new_branch
                    or old_meta.get("head_timestamp") != new_timestamp
                ):
                    self.rpg.set_git_meta(
                        head_commit=current_commit,
                        head_short=current.get("head_short"),
                        head_branch=new_branch,
                        head_timestamp=new_timestamp,
                    )
                    result["meta_git_refreshed"] = True
            return result

        if mode == "incremental":
            stats = self._apply_incremental_dep_graph_update(
                changed_files=changed,
                renames=renames,
                save_path=save_path,
            )
            result.update(stats)
        else:  # full
            self.refresh_dep_graph(
                code_dir=code_dir,
                workspace_root=workspace_root,
                save_path=save_path,
            )
            result.update({
                "dep_nodes": len(self.rpg.dep_graph.G.nodes()),
                "dep_edges": len(self.rpg.dep_graph.G.edges()),
            })

        # ── Step 4: advance meta.git (unless CI opt-out) ──
        import os
        if current_commit and not os.environ.get(self._NO_GIT_META_ENV):
            self.rpg.set_git_meta(
                head_commit=current_commit,
                head_short=current.get("head_short"),
                head_branch=current.get("head_branch"),
                head_timestamp=current.get("head_timestamp"),
            )
            result["meta_git_advanced_to"] = current_commit

        return result

    def sync_from_file_list(
        self,
        file_paths: List[str],
        code_dir: str,
        workspace_root: str,
        save_path: Optional[str | Path] = None,
        *,
        renames: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Refresh dep_graph for an **explicit** file list (no git involved).

        Used by codegen (``run_batch.py``) where the batch already knows
        exactly which file it just modified — no need to consult git.

        Behaviour matches the incremental branch of
        :meth:`sync_from_commit_diff` but skips the entire decision
        tree.  Does **not** touch ``meta.git`` — the caller is
        responsible for that if they want it.

        Args:
            file_paths: Repo-relative source paths to refresh (any supported
                language: ``.py``/``.go``/``.rs``/``.ts``/``.js``/``.c``/``.cpp``).
            code_dir / workspace_root: As :meth:`refresh_dep_graph`.
            save_path: Optional standalone output path for the dep_graph.
                ``None`` (default) means the caller relies on the embedded
                dep_graph in ``rpg.json`` via a subsequent ``svc.save``.
            renames: Optional ``{old: new}`` pairs (rare in codegen; codegen
                doesn't typically rename files).
        """
        save_path_str = str(save_path) if save_path is not None else None
        # Lazy bootstrap: codegen may call this on an RPG that doesn't
        # have a dep_graph yet (very first batch).  Fall back to full.
        if self.rpg.dep_graph is None:
            self.refresh_dep_graph(
                code_dir=code_dir,
                workspace_root=workspace_root,
                save_path=save_path_str,
            )
            return {
                "mode": "full",
                "reason": "no_existing_dep_graph",
                "dep_nodes": len(self.rpg.dep_graph.G.nodes()),
                "dep_edges": len(self.rpg.dep_graph.G.edges()),
                "save_path": save_path_str,
            }

        stats = self._apply_incremental_dep_graph_update(
            changed_files=list(file_paths),
            renames=renames or {},
            save_path=save_path_str,
        )
        return {
            "mode": "incremental",
            "reason": "explicit_file_list",
            **stats,
            "save_path": save_path_str,
        }

    def _apply_incremental_dep_graph_update(
        self,
        *,
        changed_files: List[str],
        renames: Dict[str, str],
        save_path: Optional[str] = None,
    ) -> Dict:
        """Run ``DependencyGraph.update_files`` + rebuild RPG mappings + save.

        Internal helper shared between :meth:`sync_from_commit_diff` and
        :meth:`sync_from_file_list`.  Re-reading the dep_graph's relative
        prefix ensures rename detection works even if the caller
        normalised paths differently from how the graph stores them.
        """
        from pathlib import Path as _Path

        # Strip the dep_graph_code_dir prefix if callers handed us paths
        # relative to the workspace root rather than the code dir.
        # ``_dep_graph_code_dir`` is normalised to ``""`` for the common
        # workspace==repo case; only legacy ``repo/`` layouts have a
        # non-empty value.
        prefix = self.rpg._dep_graph_code_dir or ""
        def _strip(p: str) -> str:
            if prefix and p.startswith(prefix + "/"):
                return p[len(prefix) + 1 :]
            return p

        rel_files = [_strip(p) for p in changed_files]
        rel_renames = {_strip(k): _strip(v) for k, v in renames.items()}

        update_stats = self.rpg.dep_graph.update_files(
            rel_files,
            renames=rel_renames,
            rebuild_semantic_edges=True,
        )

        # Mappings depend on dep nodes — rebuild after update_files so
        # newly-added file nodes appear in _dep_to_rpg_map and old ones
        # are dropped.
        self.rpg._dep_to_rpg_map = self.rpg._build_dep_to_rpg_map()
        self.rpg.rebuild_cross_maps()

        # ``save_path`` is optional in the embedded-dep_graph world: the
        # dep_graph rides inside rpg.json via ``RPG.to_dict``, so callers
        # that don't need a standalone ``dep_graph.json`` pass ``None`` and
        # rely on ``svc.save(rpg_path)`` afterwards.  Legacy callers that
        # still pass a path keep their standalone artefact.
        if save_path is not None:
            save_path_resolved = _Path(save_path).resolve()
            self.rpg.save_dep_graph(save_path_resolved)
            try:
                self.rpg._dep_graph_file = str(
                    save_path_resolved.relative_to(self._rpg_dir.resolve())
                )
            except ValueError:
                # dep_graph.json lives outside the RPG dir — keep absolute path.
                self.rpg._dep_graph_file = str(save_path_resolved)

        return {
            "dep_nodes": len(self.rpg.dep_graph.G.nodes()),
            "dep_edges": len(self.rpg.dep_graph.G.edges()),
            "dep_to_rpg": len(self.rpg._dep_to_rpg_map),
            **update_stats,
        }

    @classmethod
    def from_encoder(cls, encoder_rpg_path: str | Path) -> "RPGService":
        """Create an RPGService from an encoder-generated RPG file.

        Convenience factory for loading encoder output (``rpg.json``)
        which typically has an embedded dep_graph.
        """
        rpg = RPG.load_json(str(encoder_rpg_path))
        svc = cls(rpg)
        svc._rpg_dir = Path(encoder_rpg_path).parent
        return svc

    def get_dep_context_for_batch(self, file_paths: List[str]) -> Dict[str, Dict]:
        """Extract dep_graph context for a code_gen batch.

        For each file_path, collects imports, callers, callees, and
        inheritance information from the dep_graph.

        Args:
            file_paths: List of file paths (relative to code_dir,
                matching dep_graph node IDs, e.g. ``models/user.py``).

        Returns:
            ``{file_path: {"imports": [...], "callers": [...],
            "callees": [...], "inheritance": [...]}}``.
            Empty dict for files not found in dep_graph.
        """
        if self.rpg.dep_graph is None:
            return {fp: {} for fp in file_paths}

        G = self.rpg.dep_graph.G
        result: Dict[str, Dict] = {}

        for fp in file_paths:
            if fp not in G.nodes:
                result[fp] = {}
                continue

            # Collect all entities belonging to this file
            file_entities = [fp]  # the file node itself
            for nid in G.nodes:
                if nid.startswith(fp + ":"):
                    file_entities.append(nid)

            imports = []
            callers = []
            callees = []
            inheritance = []

            for entity in file_entities:
                # Out-edges from this entity
                for _, dst, attrs in G.out_edges(entity, data=True):
                    etype = attrs.get("type", "")
                    if etype == "imports":
                        dst_attrs = G.nodes.get(dst, {})
                        imports.append({
                            "module": dst_attrs.get("module", dst),
                            "name": dst_attrs.get("name", dst),
                            "node_id": dst,
                        })
                    elif etype == "invokes":
                        dst_attrs = G.nodes.get(dst, {})
                        callees.append({
                            "name": dst_attrs.get("name", dst),
                            "node_id": dst,
                            "type": dst_attrs.get("type", ""),
                        })
                    elif etype == "inherits":
                        dst_attrs = G.nodes.get(dst, {})
                        inheritance.append({
                            "base": dst_attrs.get("name", dst),
                            "node_id": dst,
                            "direction": "extends",
                        })

                # In-edges to this entity
                for src, _, attrs in G.in_edges(entity, data=True):
                    etype = attrs.get("type", "")
                    if etype == "invokes":
                        src_attrs = G.nodes.get(src, {})
                        callers.append({
                            "name": src_attrs.get("name", src),
                            "node_id": src,
                            "type": src_attrs.get("type", ""),
                        })
                    elif etype == "inherits":
                        src_attrs = G.nodes.get(src, {})
                        inheritance.append({
                            "child": src_attrs.get("name", src),
                            "node_id": src,
                            "direction": "extended_by",
                        })

            result[fp] = {
                "imports": imports,
                "callers": callers,
                "callees": callees,
                "inheritance": inheritance,
            }

        return result

    # === Feature Graph Enrichment ===

    def enrich_from_code(
        self,
        code_dir: str,
        *,
        files: Optional[List[str]] = None,
        align_only: bool = False,
        dry_run: bool = False,
        skip_dep_rebuild: bool = False,
    ) -> Dict:
        """Enrich the feature graph by aligning paths and filling missing nodes.

        Phase A: Rebuild dep_graph (unless skip_dep_rebuild).
        Phase B: Align existing feature nodes' meta.path to actual code paths.
        Phase C: Fill new feature nodes for unmapped dep entities.
        Phase D: Rebuild mappings.

        Args:
            code_dir: Absolute path to code directory.
            files: If set, only enrich features related to these files.
            align_only: Only align meta.path, don't add new nodes.
            dry_run: Collect changes but don't apply them.
            skip_dep_rebuild: Skip dep_graph rebuild (used by sync to avoid
                double-rebuild when dep was already refreshed).

        Returns:
            Dict with enrichment statistics.
        """
        import os

        if not skip_dep_rebuild:
            workspace_root = os.getcwd()
            from .dep_graph import DependencyGraph
            dg = DependencyGraph(code_dir)
            dg.build()
            dg.parse()
            # See refresh_dep_graph for the ``.`` → ``""`` rationale.
            _rel = os.path.relpath(code_dir, workspace_root)
            self.rpg._dep_graph_code_dir = "" if _rel == "." else _rel
            self.rpg.set_dep_graph(dg)

        if self.rpg.dep_graph is None:
            return {"error": "No dep_graph available", "aligned": 0, "filled": 0}

        G = self.rpg.dep_graph.G

        # Build file filter set (normalize to dep_graph node ID prefix)
        file_filter: Optional[set] = None
        if files:
            file_filter = set()
            for f in files:
                # Normalize: strip leading workspace prefix if present
                f_norm = f.replace("\\", "/")
                file_filter.add(f_norm)
                # Also try without leading dir
                if "/" in f_norm:
                    file_filter.add(f_norm.split("/", 1)[-1])

        # ── Phase B: Align ──
        align_changes = self._enrich_align(G, file_filter)
        aligned_count = len(align_changes)

        if not dry_run:
            for node_id, new_path in align_changes:
                node = self.rpg._node_index.get(node_id)
                if node and node.meta:
                    node.meta.path = new_path

        # ── Phase C: Fill ──
        filled_count = 0
        groups_created = 0
        fill_changes = []

        if not align_only:
            fill_changes, groups_created = self._enrich_fill(G, file_filter, dry_run=dry_run)
            filled_count = len(fill_changes)

            if not dry_run:
                for parent_id, new_node_dict in fill_changes:
                    parent = self.rpg._node_index.get(parent_id)
                    if not parent:
                        continue
                    self.add_feature_node(
                        name=new_node_dict["name"],
                        parent=parent,
                        impl_path=new_node_dict["path"],
                        type_name=new_node_dict.get("type_name"),
                        generator="enrich",
                    )

        # ── Phase B+: Infer feature_group paths from children ──
        # Runs AFTER Phase C so that newly filled nodes are included.
        group_path_changes = self._enrich_group_paths(G, dry_run=dry_run)
        groups_pathed = len(group_path_changes)

        # ── Phase B++: Infer functional_area/repo paths from groups ──
        l1_path_changes = self._enrich_l1_paths(G, dry_run=dry_run)
        l1_pathed = len(l1_path_changes)

        # ── Phase D: Rebuild mappings ──
        if not dry_run:
            self.rpg._dep_to_rpg_map = self.rpg._build_dep_to_rpg_map()
            self.rpg.rebuild_cross_maps()

        result = {
            "aligned": aligned_count,
            "groups_pathed": groups_pathed,
            "l1_pathed": l1_pathed,
            "filled": filled_count,
            "groups_created": groups_created,
            "mapping_after": len(self.rpg._dep_to_rpg_map) if not dry_run else None,
        }

        if dry_run:
            result["align_details"] = [
                {"node_id": nid, "new_path": p} for nid, p in align_changes
            ]
            result["group_path_details"] = [
                {"node_id": nid, "new_path": p} for nid, p in group_path_changes
            ]
            result["l1_path_details"] = [
                {"node_id": nid, "new_path": p} for nid, p in l1_path_changes
            ]
            result["fill_details"] = [
                {"parent_id": pid, "node": nd} for pid, nd in fill_changes
            ]

        return result

    def _enrich_align(self, G, file_filter: Optional[set]) -> List[Tuple[str, str]]:
        """Phase B: Align existing feature nodes' meta.path to actual dep_graph paths.

        Returns list of (rpg_node_id, new_meta_path) tuples.
        """
        changes: List[Tuple[str, str]] = []

        # Build dep_graph indexes
        # suffix_to_dep: "routes/admin.py" → [dep_node_ids...]
        dep_by_file_suffix: Dict[str, List[str]] = {}
        # name_to_dep: "User" → [dep_node_ids...]
        dep_by_name: Dict[str, List[str]] = {}

        _entity_types = ("class", "function", "method")

        for nid in G.nodes():
            ndata = G.nodes[nid]
            ntype = ndata.get("type", "")
            if ntype not in _entity_types:
                continue
            name = ndata.get("name", "")
            if name:
                dep_by_name.setdefault(name, []).append(nid)
            # Extract file suffix: "routes/admin.py:func" → "routes/admin.py"
            file_part = nid.split(":")[0] if ":" in nid else nid
            parts = file_part.replace("\\", "/").split("/")
            suffix = "/".join(parts[-2:]) if len(parts) >= 2 else file_part
            dep_by_file_suffix.setdefault(suffix, []).append(nid)

        # Track which dep entities are already aligned (to avoid double-alignment)
        already_aligned_dep: set = set()

        # Pre-mark dep entities that are already correctly pointed to by existing
        # feature nodes (idempotency: skip re-aligning what's already correct)
        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.meta and rpg_node.meta.path and isinstance(rpg_node.meta.path, str):
                dep_key = self._rpg_path_to_dep_id(rpg_node.meta.path)
                if dep_key and dep_key in G.nodes:
                    already_aligned_dep.add(dep_key)

        # Process each feature node with a meta.path
        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.meta is None or not rpg_node.meta.path:
                continue
            if not isinstance(rpg_node.meta.path, str):
                continue
            if rpg_node.node_type not in ("feature", None) and rpg_node.level != self.rpg.MAX_FEATURE_LEVEL:
                continue

            old_path = rpg_node.meta.path

            # Apply file filter
            if file_filter:
                matches_filter = any(filt in old_path for filt in file_filter)
                if not matches_filter:
                    continue

            # Extract entity name and file suffix from planned path.
            # Use ``parse_node_path`` so that method paths
            # (``file::Cls::m``) correctly split into file + symbol chain
            # — the previous ``rsplit("::", 1)`` accidentally treated
            # ``file::Cls`` as the file portion and broke alignment for
            # methods.  Tolerates both canonical (``file::Name``) and
            # legacy (``file::class Name``) inputs.
            from .path_format import parse_node_path

            entity_name = ""
            file_part, sym_parts = parse_node_path(old_path)
            if sym_parts:
                last = sym_parts[-1]
                for legacy_prefix in ("class ", "function ", "method "):
                    if last.startswith(legacy_prefix):
                        last = last[len(legacy_prefix):]
                        break
                entity_name = last
            parts = file_part.replace("\\", "/").split("/")
            file_suffix = "/".join(parts[-2:]) if len(parts) >= 2 else file_part

            # Strategy 1: Exact name match → pick best by file suffix
            best_match = None
            if entity_name and entity_name in dep_by_name:
                candidates = dep_by_name[entity_name]
                for dep_nid in candidates:
                    if dep_nid in already_aligned_dep:
                        continue
                    dep_file = dep_nid.split(":")[0] if ":" in dep_nid else dep_nid
                    dep_parts = dep_file.replace("\\", "/").split("/")
                    dep_suffix = "/".join(dep_parts[-2:]) if len(dep_parts) >= 2 else dep_file
                    if dep_suffix == file_suffix:
                        best_match = dep_nid
                        break
                # No cross-file fallback — name alone is ambiguous
                # (e.g. "index" exists in multiple files)

            # Strategy 2: Positional matching within same file
            if not best_match:
                # Find dep entities in same file by suffix
                file_candidates = dep_by_file_suffix.get(file_suffix, [])
                file_candidates = [c for c in file_candidates if c not in already_aligned_dep]
                if file_candidates:
                    # Find sibling features under same parent that share this file suffix
                    parent = rpg_node.parent()
                    if parent:
                        siblings_in_file = []
                        for sib in parent.children():
                            if sib.meta and sib.meta.path and isinstance(sib.meta.path, str):
                                # Use parse_node_path so method paths
                                # (``file::Cls::m``) yield the real file
                                # part instead of ``file::Cls``.
                                sib_file, _ = parse_node_path(sib.meta.path)
                                sib_parts = sib_file.replace("\\", "/").split("/")
                                sib_suffix = "/".join(sib_parts[-2:]) if len(sib_parts) >= 2 else sib_file
                                if sib_suffix == file_suffix:
                                    siblings_in_file.append(sib)
                        # Sort dep candidates by start_line
                        dep_sorted = sorted(file_candidates,
                                          key=lambda nid: G.nodes[nid].get("start_line", 0))
                        # Find position of current node among siblings
                        try:
                            pos = siblings_in_file.index(rpg_node)
                            if pos < len(dep_sorted):
                                best_match = dep_sorted[pos]
                        except ValueError:
                            pass

            # Validate: reject match if type_name disagrees with dep type
            # (e.g. RPG says "class" but dep candidate is "method")
            if best_match:
                dep_type = G.nodes[best_match].get("type", "")
                rpg_type = str(rpg_node.meta.type_name) if rpg_node.meta and rpg_node.meta.type_name else ""
                if rpg_type and dep_type and rpg_type != dep_type:
                    best_match = None

            if best_match:
                # Convert dep node ID to RPG path format
                new_path = self._dep_id_to_rpg_path(best_match, G)
                if new_path != old_path:
                    changes.append((rpg_nid, new_path))
                    already_aligned_dep.add(best_match)

        return changes

    def _enrich_fill(self, G, file_filter: Optional[set], *, dry_run: bool = False) -> Tuple[List[Tuple[str, Dict]], int]:
        """Phase C: Create feature nodes for unmapped dep entities.

        Returns (fill_changes, groups_created) where fill_changes is a list
        of (parent_node_id, new_node_dict) tuples.
        """
        # Find all dep entities already mapped to some feature node
        mapped_dep_ids: set = set()
        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.meta and rpg_node.meta.path and isinstance(rpg_node.meta.path, str):
                path_str = rpg_node.meta.path
                # Normalize RPG path to dep_graph key
                dep_key = self._rpg_path_to_dep_id(path_str)
                if dep_key:
                    mapped_dep_ids.add(dep_key)

        # Also include everything in _dep_to_rpg_map
        mapped_dep_ids.update(self.rpg._dep_to_rpg_map.keys())

        # Build file→L2_parent index
        file_to_parent = self._build_file_to_parent_index()
        # Build directory→L1 index
        dir_to_l1 = self._build_dir_to_l1_index()

        fill_changes: List[Tuple[str, Dict]] = []
        groups_created = 0
        new_groups: Dict[str, str] = {}  # file_key → created group node id

        for nid in G.nodes():
            ndata = G.nodes[nid]
            ntype = ndata.get("type", "")
            if ntype not in ("class", "function", "method"):
                continue
            if nid in mapped_dep_ids:
                continue

            # Apply file filter
            if file_filter:
                matches = any(filt in nid for filt in file_filter)
                if not matches:
                    continue

            name = ndata.get("name", nid.split(":")[-1] if ":" in nid else nid)
            file_part = nid.split(":")[0] if ":" in nid else nid
            rpg_path = self._dep_id_to_rpg_path(nid, G)

            # Resolve NodeType enum from string
            node_type_enum = None
            try:
                node_type_enum = NodeType(ntype)
            except ValueError:
                pass

            # Find parent: first try file-based matching
            parent_id = file_to_parent.get(file_part)

            if not parent_id:
                # Try by file suffix
                parts = file_part.replace("\\", "/").split("/")
                suffix = "/".join(parts[-2:]) if len(parts) >= 2 else file_part
                for fp_key, pid in file_to_parent.items():
                    fp_parts = fp_key.replace("\\", "/").split("/")
                    fp_suffix = "/".join(fp_parts[-2:]) if len(fp_parts) >= 2 else fp_key
                    if fp_suffix == suffix:
                        parent_id = pid
                        break

            if not parent_id:
                # No features reference this file — find/create L2 group
                group_key = file_part
                if group_key in new_groups:
                    parent_id = new_groups[group_key]
                else:
                    # Find L1 by directory
                    dir_path = "/".join(file_part.replace("\\", "/").split("/")[:-1])
                    l1_id = None
                    for d_prefix, l1 in dir_to_l1.items():
                        if dir_path.endswith(d_prefix) or d_prefix in dir_path:
                            l1_id = l1
                            break
                    if not l1_id:
                        # Fallback: first L1
                        for n in self.rpg._node_index.values():
                            if n.node_type == "functional_area":
                                l1_id = n.id
                                break

                    if l1_id:
                        l1_node = self.rpg._node_index[l1_id]
                        # Create L2 group named after file
                        stem = Path(file_part).stem.replace("_", " ").title()
                        group_name = stem

                        # Check if group already exists
                        existing_group = None
                        for child in l1_node.children():
                            if child.name == group_name:
                                existing_group = child
                                break

                        if existing_group:
                            parent_id = existing_group.id
                        else:
                            group_node = Node(
                                id=f"{group_name}_{uuid8()}",
                                name=group_name,
                                node_type="feature_group",
                                meta=NodeMetaData(generator="enrich"),
                            )
                            group_node.level = l1_node.level + 1 if l1_node.level is not None else 2
                            if not dry_run:
                                self.rpg.add_node(group_node)
                                l1_node.add_child(group_node)
                            parent_id = group_node.id
                            groups_created += 1

                        new_groups[group_key] = parent_id

            if parent_id:
                new_node_dict = {
                    "name": name,
                    "path": rpg_path,
                    "type_name": node_type_enum,
                }
                fill_changes.append((parent_id, new_node_dict))
                # Mark as mapped to avoid duplicates within same run
                mapped_dep_ids.add(nid)

        return fill_changes, groups_created

    def _dep_id_to_rpg_path(self, dep_nid: str, G) -> str:
        """Convert dep_graph node ID to canonical RPG meta.path.

        ``models/user.py:User``        -> ``models/user.py::User``
        ``models/user.py:User.login``  -> ``models/user.py::User::login``
        ``models/user.py``             -> ``models/user.py``

        Implementation routes through :func:`rpg.path_format.from_dep_graph_id`
        so the produced path always matches the canonical format used by the
        encoder and code-gen pipelines.  Previously this method emitted the
        legacy ``::class X`` / ``::function X`` prefixed form, which would
        silently revert canonical paths to legacy on every
        ``cmind update`` (a real bug — the prefix-stripped form on next read
        produced a *different* lookup key and broke dep<->RPG correlation).

        The ``G`` argument is kept for backward compatibility with callers
        that pass it; the dep type is no longer needed since canonical paths
        rely on ``NodeMetaData.type_name`` for disambiguation.
        """
        from .path_format import from_dep_graph_id
        return from_dep_graph_id(dep_nid)

    def _rpg_path_to_dep_id(self, rpg_path: str) -> Optional[str]:
        """Convert RPG meta.path (canonical OR legacy) to dep_graph node ID.

        Canonical input examples::

            "models/user.py::User"          -> "models/user.py:User"
            "models/user.py::User::login"   -> "models/user.py:User.login"

        Legacy input is also tolerated so that rpg.json files written by
        older encoder versions can still be aligned during ``cmind update``
        without a separate migration step::

            "models/user.py::class User"           -> "models/user.py:User"
            "models/user.py::function register"    -> "models/user.py:register"

        Returns ``None`` if the path doesn't look like an entity path.
        """
        from .path_format import parse_node_path
        if not rpg_path or "::" not in rpg_path:
            return None
        file_part, parts = parse_node_path(rpg_path)
        cleaned: List[str] = []
        for seg in parts:
            for prefix in ("class ", "function ", "method "):
                if seg.startswith(prefix):
                    seg = seg[len(prefix):]
                    break
            if seg:
                cleaned.append(seg)
        if not cleaned:
            return None
        return f"{file_part}:" + ".".join(cleaned)

    def _build_file_to_parent_index(self) -> Dict[str, str]:
        """Build index: file_path -> parent (L2) node ID.

        For each feature node, extract the file from meta.path and record
        the feature's parent ID.
        """
        from .path_format import parse_node_path

        index: Dict[str, str] = {}
        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.meta is None or not rpg_node.meta.path:
                continue
            if not isinstance(rpg_node.meta.path, str):
                continue
            path_str = rpg_node.meta.path
            # Extract file part via parse_node_path so method paths
            # (``file::Cls::m``) correctly yield the file, not
            # ``file::Cls``.
            file_part, _ = parse_node_path(path_str)
            parent = rpg_node.parent()
            if parent and parent.id not in ("", None):
                index.setdefault(file_part, parent.id)
        return index

    def _build_dir_to_l1_index(self) -> Dict[str, str]:
        """Build index: directory_keyword → L1 node ID.

        Uses L1 name to directory keyword mapping heuristics.
        """
        index: Dict[str, str] = {}
        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.node_type != "functional_area":
                continue
            name_lower = rpg_node.name.lower()
            # Heuristic: map L1 names to directory patterns
            if "model" in name_lower:
                index["models"] = rpg_nid
            if "route" in name_lower or "view" in name_lower:
                index.setdefault("routes", rpg_nid)
                # Also scan descendant feature nodes for directory info
                for desc in rpg_node.children(recursive=True):
                    if desc.meta and desc.meta.path and isinstance(desc.meta.path, str):
                        # Use parse_node_path to get the real file part
                        # (rsplit on "::" mishandles method paths like
                        # ``foo.py::Cls::m`` -> ``foo.py::Cls``).
                        from .path_format import parse_node_path as _ppn
                        file_part, _sym = _ppn(desc.meta.path)
                        dir_part = "/".join(file_part.replace("\\", "/").split("/")[:-1])
                        if dir_part:
                            last_dir = dir_part.split("/")[-1]
                            index.setdefault(last_dir, rpg_nid)
            if "api" in name_lower:
                index.setdefault("api", rpg_nid)
            if "template" in name_lower:
                index.setdefault("templates", rpg_nid)
            if "util" in name_lower:
                index.setdefault("utils", rpg_nid)
            if "core" in name_lower or "app" in name_lower:
                index.setdefault("core", rpg_nid)
            if "static" in name_lower or "css" in name_lower:
                index.setdefault("static", rpg_nid)
        return index

    def _enrich_group_paths(self, G, *, dry_run: bool = False) -> List[Tuple[str, str]]:
        """Phase B+: Infer meta.path for L2 feature_group nodes from children.

        Always re-infers (not skipped if path exists) so that Phase B align
        corrections propagate upward. Only counts as a change if the result
        differs from the current value.

        Returns list of (rpg_node_id, new_path) tuples.
        """
        changes: List[Tuple[str, str]] = []

        # Build set of actual file paths from dep_graph for validation
        dep_file_paths: set = set()
        for nid in G.nodes():
            ndata = G.nodes[nid]
            if ndata.get("type") in ("file", "class", "function", "method"):
                file_part = nid.split(":")[0] if ":" in nid else nid
                dep_file_paths.add(file_part)

        for rpg_nid, rpg_node in self.rpg._node_index.items():
            if rpg_node.node_type != "feature_group":
                continue

            # Collect file paths from children, only if they exist in dep_graph
            child_files: set = set()
            for child in rpg_node.children():
                if child.meta and child.meta.path:
                    paths = child.meta.path if isinstance(child.meta.path, list) else [child.meta.path]
                    for p in paths:
                        if isinstance(p, str):
                            # parse_node_path correctly extracts the file
                            # boundary even for method paths
                            # (``foo.py::Cls::m``); plain ``rsplit`` would
                            # treat ``foo.py::Cls`` as the file.
                            from .path_format import parse_node_path as _ppn
                            file_part, _sym = _ppn(p)
                            if file_part in dep_file_paths:
                                child_files.add(file_part)

            if not child_files:
                continue

            new_path = sorted(child_files)[0] if len(child_files) == 1 else sorted(child_files)
            new_type = NodeType.FILE

            # Check if unchanged (idempotency)
            old_path = rpg_node.meta.path if rpg_node.meta else None
            old_type = rpg_node.meta.type_name if rpg_node.meta else None
            if old_path == new_path and old_type == new_type:
                continue

            changes.append((rpg_nid, new_path))
            if not dry_run:
                if rpg_node.meta is None:
                    rpg_node.meta = NodeMetaData()
                rpg_node.meta.path = new_path
                rpg_node.meta.type_name = new_type

        return changes

    def _enrich_l1_paths(self, G, *, dry_run: bool = False) -> List[Tuple[str, str]]:
        """Phase B++: Infer meta.path for L1 functional_area and L0 repo nodes.

        For L1: collects directories from descendant feature_group paths.
        For L0 repo: sets path to "." for dep_graph root mapping.

        Returns list of (rpg_node_id, new_path) tuples.
        """
        changes: List[Tuple[str, str]] = []

        # Build set of actual directory paths from dep_graph for validation
        dep_dir_paths: set = set()
        for nid in G.nodes():
            ndata = G.nodes[nid]
            if ndata.get("type") == "directory":
                dep_dir_paths.add(nid)

        for rpg_nid, rpg_node in self.rpg._node_index.items():
            # Handle L0 repo node
            if rpg_node.node_type == "repo":
                new_path = "."
                new_type = NodeType.DIRECTORY
                old_path = rpg_node.meta.path if rpg_node.meta else None
                old_type = rpg_node.meta.type_name if rpg_node.meta else None
                if old_path == new_path and old_type == new_type:
                    continue
                changes.append((rpg_nid, new_path))
                if not dry_run:
                    if rpg_node.meta is None:
                        rpg_node.meta = NodeMetaData()
                    rpg_node.meta.path = new_path
                    rpg_node.meta.type_name = new_type
                continue

            # Handle L1 functional_area nodes
            if rpg_node.node_type != "functional_area":
                continue

            # Collect directories from children (feature_group) paths
            dirs: set = set()
            for child in rpg_node.children():
                if child.meta and child.meta.path:
                    paths = child.meta.path if isinstance(child.meta.path, list) else [child.meta.path]
                    for p in paths:
                        if isinstance(p, str):
                            dir_part = "/".join(p.replace("\\", "/").split("/")[:-1])
                            if dir_part and dir_part in dep_dir_paths:
                                dirs.add(dir_part)

            if not dirs:
                continue

            new_path = sorted(dirs)[0] if len(dirs) == 1 else sorted(dirs)
            new_type = NodeType.DIRECTORY

            old_path = rpg_node.meta.path if rpg_node.meta else None
            old_type = rpg_node.meta.type_name if rpg_node.meta else None
            if old_path == new_path and old_type == new_type:
                continue

            changes.append((rpg_nid, new_path))
            if not dry_run:
                if rpg_node.meta is None:
                    rpg_node.meta = NodeMetaData()
                rpg_node.meta.path = new_path
                rpg_node.meta.type_name = new_type

        return changes
