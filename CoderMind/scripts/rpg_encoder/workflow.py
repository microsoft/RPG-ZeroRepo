"""Workflow Integration.

Bridges the *forward* (requirements -> code) and *reverse* (code -> RPG)
pipelines so that they can be composed seamlessly.

This is an **original** CoderMind module -- it is NOT ported from
RPG-ZeroRepo.

Key class:
  ``WorkflowIntegration`` -- stateless helper methods that prepare
  context for code generation and merge generated code back into the RPG.

Supported workflow scenarios:
  1. **Pure forward**:  feature_spec -> build_skeleton -> code_gen
  2. **Pure reverse**:  encode -> search / explore
  3. **Mixed enhance**: encode -> feature_spec(extend) -> code_gen -> update_rpg
  4. **Iterative**:     code_gen <-> update_rpg  (loop)

Typical usage::

    # After encoding an existing repo, prepare context for code generation
    context = WorkflowIntegration.prepare_for_codegen(
        rpg=encoded_rpg,
        target_nodes=["api/endpoints/payment"],
        repo_dir="/path/to/repo",
    )

    # After code generation, merge new files back into the RPG
    updated_rpg = WorkflowIntegration.merge_generated_code(
        rpg=rpg,
        generated_files={"src/payment.py": source_code},
        repo_dir="/path/to/repo",
    )
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rpg import (
    EdgeType,
    Node,
    NodeMetaData,
    NodeType,
    RPG,
    class_node_path,
    function_node_path,
    method_node_path,
)

from .config import CMindConfig
from .version_control import RPGVersionControl, RPG_FILE_NAME
from common.rpg_io import atomic_write_rpg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WorkflowIntegration
# ---------------------------------------------------------------------------


class WorkflowIntegration:
    """Stateless helpers that connect the forward and reverse pipelines.

    All public methods are class-methods or static-methods so that no
    persistent state is required.  Configuration is read from the
    ``CMindConfig`` object when needed.

    Design rationale:
    - **No modifications to existing forward-flow code.**  The forward
      pipeline (``scripts/rpg_gen/``, ``scripts/code_gen/``) continues to
      work as before.
    - **RPG is the shared data structure.**  Both pipelines read/write RPG
      JSON files; this class provides the glue that keeps them consistent.
    """

    # ------------------------------------------------------------------
    # prepare_for_codegen
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_for_codegen(
        rpg: RPG,
        target_nodes: Optional[List[str]] = None,
        repo_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare context for code generation from an encoded RPG.

        This is the key bridge from the *reverse* pipeline (encode) to
        the *forward* pipeline (code_gen).  It extracts the information
        that code generation needs: existing structure, dependency edges,
        interface definitions, and the list of target nodes to extend.

        Args:
            rpg: An RPG instance (typically produced by ``RPGParser``).
            target_nodes: Optional list of node IDs, node names, or
                meta.path values that indicate *where* new code should
                be added.  If omitted, the full RPG context is returned.
            repo_dir: Repository directory (used to resolve file paths).

        Returns:
            A dictionary containing:
            - ``rpg_dict``: Serialized RPG (``rpg.to_dict()``).
            - ``repo_name``: Name of the repository.
            - ``functional_areas``: Top-level L1 area names.
            - ``target_context``: Details about each target node
              (name, path, children, dependencies).
            - ``existing_interfaces``: Map of file paths to their known
              code-level entities (classes, functions) extracted from RPG
              node metadata.
            - ``dependency_edges``: Non-containment edges relevant to the
              target nodes.
            - ``source``: RPG source label (``"encoded"`` / ``"generated"``
              / ``"mixed"``).
        """
        rpg_dict = rpg.to_dict()
        functional_areas = rpg.get_functional_areas()

        # Resolve target nodes
        target_context: List[Dict[str, Any]] = []
        resolved_ids: Set[str] = set()

        if target_nodes:
            for ref in target_nodes:
                node = _resolve_node(rpg, ref)
                if node is None:
                    logger.warning("Target node not found: %s", ref)
                    continue
                resolved_ids.add(node.id)
                target_context.append(_build_node_context(rpg, node))

        # Gather existing interfaces (file-level entities)
        existing_interfaces = _gather_existing_interfaces(rpg)

        # Gather relevant dependency edges
        dependency_edges = _gather_dependency_edges(rpg, resolved_ids)

        # Determine source label
        source = _infer_rpg_source(rpg)

        return {
            "rpg_dict": rpg_dict,
            "repo_name": rpg.repo_name,
            "functional_areas": functional_areas,
            "target_context": target_context,
            "existing_interfaces": existing_interfaces,
            "dependency_edges": dependency_edges,
            "source": source,
        }

    # ------------------------------------------------------------------
    # merge_generated_code
    # ------------------------------------------------------------------

    @staticmethod
    def merge_generated_code(
        rpg: RPG,
        generated_files: Dict[str, str],
        repo_dir: Optional[str] = None,
        source: str = "mixed",
    ) -> RPG:
        """Merge newly generated code back into the RPG.

        After code generation produces new files, this method:
        1. Parses each file to extract code units (classes, functions).
        2. Creates or updates RPG nodes for each code unit.
        3. Updates the dependency graph if ``repo_dir`` is provided.

        The RPG is modified **in place** and also returned.

        Args:
            rpg: The current RPG instance.
            generated_files: Mapping of ``{relative_file_path: source_code}``.
            repo_dir: Repository directory (for dep-graph rebuild).
            source: Source label to set on new nodes' generator field.

        Returns:
            The updated RPG instance (same object, mutated).
        """
        if not generated_files:
            logger.info("No generated files to merge.")
            return rpg

        from rpg.code_unit import ParsedFile

        generator_name = f"workflow_{source}"
        merged_count = 0

        for file_path, code in generated_files.items():
            try:
                parsed = ParsedFile(code=code, file_path=file_path)
            except Exception as exc:
                logger.warning(
                    "Failed to parse generated file %s: %s", file_path, exc
                )
                continue

            # Find or create the parent directory node in the RPG
            parent_node = rpg.find_parent_by_path(
                file_path, create_missing=True, generator=generator_name
            )

            # Create or find the file node
            file_node = rpg.find_node_by_path(file_path)
            if file_node is None:
                file_name = os.path.basename(file_path)
                file_id = f"file_{file_name.replace('.', '_')}_{_short_id()}"
                file_node = Node(
                    id=file_id,
                    name=file_name,
                    node_type="feature",
                    level=None,
                    meta=NodeMetaData(
                        type_name=NodeType.FILE,
                        path=file_path,
                        description=f"Generated file: {file_path}",
                        generator=generator_name,
                    ),
                )
                rpg.add_node(file_node)
                rpg.add_edge(parent_node.id, file_node.id, EdgeType.CONTAINS)

            # Add code-unit child nodes (classes, functions)
            for unit in parsed.units:
                if unit.unit_type not in ("class", "function", "method"):
                    continue

                if unit.unit_type == "method" and unit.parent:
                    unit_path = method_node_path(file_path, unit.parent, unit.name)
                elif unit.unit_type == "class":
                    unit_path = class_node_path(file_path, unit.name)
                else:
                    unit_path = function_node_path(file_path, unit.name)

                existing = rpg.find_node_by_path(unit_path)
                if existing is not None:
                    # Update description to note re-generation
                    if existing.meta:
                        existing.meta.generator = generator_name
                    continue

                unit_id = f"{unit.unit_type}_{unit.name}_{_short_id()}"
                type_name_enum = {
                    "class": NodeType.CLASS,
                    "function": NodeType.FUNCTION,
                    "method": NodeType.METHOD,
                }.get(unit.unit_type, NodeType.FUNCTION)

                unit_node = Node(
                    id=unit_id,
                    name=unit.name,
                    node_type="feature",
                    level=None,
                    meta=NodeMetaData(
                        type_name=type_name_enum,
                        path=unit_path,
                        description=unit.docstring or "",
                        generator=generator_name,
                    ),
                )
                rpg.add_node(unit_node)

                # Methods go under their class node; top-level units under file
                if unit.parent and unit.unit_type == "method":
                    class_path = class_node_path(file_path, unit.parent)
                    class_node = rpg.find_node_by_path(class_path)
                    if class_node:
                        rpg.add_edge(
                            class_node.id, unit_node.id, EdgeType.CONTAINS
                        )
                    else:
                        rpg.add_edge(
                            file_node.id, unit_node.id, EdgeType.CONTAINS
                        )
                else:
                    rpg.add_edge(
                        file_node.id, unit_node.id, EdgeType.CONTAINS
                    )

                merged_count += 1

        logger.info(
            "Merged %d code units from %d generated files into RPG.",
            merged_count,
            len(generated_files),
        )

        # Rebuild dependency graph if repo_dir is available
        if repo_dir:
            try:
                rpg.parse_dep_graph(repo_dir)
                logger.info("Dependency graph rebuilt after merge.")
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild dependency graph: %s", exc
                )

        return rpg

    # ------------------------------------------------------------------
    # save_rpg  (convenience: save + version)
    # ------------------------------------------------------------------

    @staticmethod
    def save_rpg(
        rpg: RPG,
        cmind_dir: str,
        message: str = "",
        source: str = "mixed",
        version_control: bool = True,
    ) -> Dict[str, Any]:
        """Save the RPG to disk and optionally create a version snapshot.

        Args:
            rpg: The RPG instance to save.
            cmind_dir: Path to the ``.cmind`` directory.
            message: Description for the version snapshot.
            source: Source label (``"generated"``/``"encoded"``/``"mixed"``).
            version_control: Whether to also save a versioned snapshot.

        Returns:
            Dictionary with ``rpg_path`` and optional ``version``.
        """
        data_dir = os.path.join(cmind_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        rpg_path = os.path.join(data_dir, RPG_FILE_NAME)
        rpg_dict = rpg.to_dict()
        rpg_dict["repo_name"] = rpg.repo_name
        rpg_dict["repo_info"] = getattr(rpg, "repo_info", "")
        rpg_dict["excluded_files"] = getattr(rpg, "excluded_files", [])

        # Atomic write: a partial encoder run (Ctrl-C, OOM, power loss)
        # can no longer brick the workspace with a truncated rpg.json
        # — we write to <path>.tmp then os.replace into place.  See
        # ``common.rpg_io.atomic_write_rpg`` for the recovery side.
        atomic_write_rpg(Path(rpg_path), rpg_dict)

        result: Dict[str, Any] = {"rpg_path": rpg_path}

        if version_control:
            try:
                config = CMindConfig.load(
                    os.path.dirname(cmind_dir)
                )
                vc = RPGVersionControl(
                    cmind_dir=cmind_dir,
                    max_history=config.workflow.versioning.max_history,
                )
                version = vc.save_version(rpg, message=message, source=source)
                result["version"] = version
            except Exception as exc:
                logger.warning("Version control save failed: %s", exc)

        logger.info("RPG saved to %s", rpg_path)
        return result

    # ------------------------------------------------------------------
    # load_rpg  (convenience: load from .cmind/data/rpg.json)
    # ------------------------------------------------------------------

    @staticmethod
    def load_rpg(cmind_dir: str) -> Optional[RPG]:
        """Load the current RPG from ``<cmind_dir>/data/rpg.json``.

        Args:
            cmind_dir: Path to the ``.cmind`` directory.

        Returns:
            The loaded RPG, or ``None`` if the file does not exist.
        """
        rpg_path = os.path.join(cmind_dir, "data", RPG_FILE_NAME)
        if not os.path.isfile(rpg_path):
            return None

        try:
            with open(rpg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Handle wrapped format (encode command output)
            if "rpg" in data and "structure" in data["rpg"]:
                rpg = RPG.from_dict(data["rpg"]["structure"])
            else:
                rpg = RPG.from_dict(data)

            rpg.repo_info = data.get("repo_info", "")
            rpg.excluded_files = data.get("excluded_files", [])
            return rpg
        except Exception as exc:
            logger.error("Failed to load RPG from %s: %s", rpg_path, exc)
            return None

    # ------------------------------------------------------------------
    # detect_workflow_mode
    # ------------------------------------------------------------------

    @staticmethod
    def detect_workflow_mode(
        rpg: Optional[RPG],
        has_feature_spec: bool = False,
        repo_dir: Optional[str] = None,
    ) -> str:
        """Detect the most appropriate workflow mode.

        Heuristic:
        - No RPG at all -> ``"forward"`` (start from scratch)
        - RPG exists, no feature_spec -> ``"reverse"`` (explore only)
        - RPG exists, feature_spec exists -> ``"mixed"``

        Args:
            rpg: Current RPG (may be ``None``).
            has_feature_spec: Whether a feature_spec.json exists.
            repo_dir: Repository directory (for additional checks).

        Returns:
            One of ``"forward"``, ``"reverse"``, ``"mixed"``.
        """
        if rpg is None:
            return "forward"
        if not has_feature_spec:
            return "reverse"
        return "mixed"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _short_id() -> str:
    """Generate a short random hex ID (8 chars)."""
    import uuid
    return uuid.uuid4().hex[:8]


def _resolve_node(rpg: RPG, ref: str) -> Optional[Node]:
    """Resolve a string reference to an RPG node.

    Tries the following strategies in order:
    1. Direct ID lookup
    2. meta.path lookup
    3. Feature path lookup
    4. Name substring match
    """
    # 1. Direct ID
    node = rpg.get_node_by_id(ref)
    if node is not None:
        return node

    # 2. meta.path
    node = rpg.find_node_by_path(ref)
    if node is not None:
        return node

    # 3. Feature path
    node = rpg.get_node_by_feature_path(ref)
    if node is not None:
        return node

    # 4. Name match (case-insensitive, first match)
    ref_lower = ref.lower()
    for n in rpg.nodes.values():
        if n.name.lower() == ref_lower:
            return n

    return None


def _build_node_context(rpg: RPG, node: Node) -> Dict[str, Any]:
    """Build a context dictionary for a single target node."""
    children_info = []
    for child in node.children():
        children_info.append({
            "id": child.id,
            "name": child.name,
            "type": child.node_type,
            "path": child.meta.path if child.meta else None,
        })

    # Collect dependency edges involving this node
    related_edges = []
    for edge in rpg.edges:
        if edge.src == node.id or edge.dst == node.id:
            related_edges.append(edge.to_dict())

    return {
        "id": node.id,
        "name": node.name,
        "type": node.node_type,
        "level": node.level,
        "path": node.meta.path if node.meta else None,
        "description": node.meta.description if node.meta else "",
        "children": children_info,
        "related_edges": related_edges,
        "feature_path": node.feature_path(),
    }


def _gather_existing_interfaces(rpg: RPG) -> Dict[str, List[Dict[str, str]]]:
    """Extract file-level code entities from RPG node metadata.

    Returns:
        ``{file_path: [{"name": ..., "type": ..., "path": ...}, ...]}``.
    """
    interfaces: Dict[str, List[Dict[str, str]]] = {}

    code_types = {
        NodeType.CLASS,
        NodeType.FUNCTION,
        NodeType.METHOD,
    }

    for node in rpg.nodes.values():
        if not node.meta or not node.meta.type_name:
            continue
        if node.meta.type_name not in code_types:
            continue
        if not node.meta.path or not isinstance(node.meta.path, str):
            continue

        # Extract file path from meta.path (e.g. "src/foo.py::ClassName")
        path_str = node.meta.path
        if "::" in path_str:
            file_path = path_str.split("::")[0]
        else:
            file_path = path_str

        entry = {
            "name": node.name,
            "type": node.meta.type_name.value,
            "path": path_str,
        }
        interfaces.setdefault(file_path, []).append(entry)

    return interfaces


def _gather_dependency_edges(
    rpg: RPG, target_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Collect non-containment edges relevant to the target nodes.

    If ``target_ids`` is empty, returns all non-containment edges.
    """
    if not target_ids:
        return [e.to_dict() for e in rpg.edges]

    # Expand target_ids to include descendants
    expanded: Set[str] = set(target_ids)
    for tid in target_ids:
        expanded.update(rpg.get_children(tid, recursive=True))

    result = []
    for edge in rpg.edges:
        if edge.src in expanded or edge.dst in expanded:
            result.append(edge.to_dict())
    return result


def _infer_rpg_source(rpg: RPG) -> str:
    """Infer the RPG's origin based on node generator metadata.

    Returns ``"generated"``, ``"encoded"``, or ``"mixed"``.
    """
    generators: Set[str] = set()
    for node in rpg.nodes.values():
        if node.meta and node.meta.generator:
            generators.add(node.meta.generator)

    has_forward = any(
        g in generators for g in ("code_gen", "design_base_classes", "feature_spec")
    )
    has_reverse = any(
        g in generators for g in ("rpg_encoder", "rpg_parser", "workflow_encoded")
    )

    if has_forward and has_reverse:
        return "mixed"
    if has_reverse:
        return "encoded"
    return "generated"
