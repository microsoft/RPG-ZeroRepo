#!/usr/bin/env python3
"""Interfaces Store - Unified Data Structure for Interface Design.

This module provides a unified data store for managing all interface-related data
during the design_interfaces workflow. It replaces scattered dict structures with
a single source of truth.

Key components:
- InterfaceUnit: Single interface unit (class/function)
- InheritanceEdge, InvocationEdge, ReferenceEdge: Dependency edge types
- InterfacesStore: Central store managing all units and edges with auto-maintained indexes
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple

from common.code_dedup import dedup_file_code
from decoder_lang.unit_kind import classify_unit_kind

logger = logging.getLogger(__name__)


# ============================================================================
# Edge Data Classes
# ============================================================================

@dataclass
class InheritanceEdge:
    """Inheritance relationship (child extends parent)."""
    child: str              # e.g., "ChildClass"
    parent: str             # e.g., "BaseClass"
    child_file: str         # source file path
    parent_file: Optional[str] = None
    generator: str = "design_interfaces"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format."""
        return {
            "child": self.child,
            "parent": self.parent,
            "source_file": self.child_file,
            "parent_file": self.parent_file,
            "edge_type": "inherits",
            "generator": self.generator,
        }


@dataclass
class InvocationEdge:
    """Invocation relationship (caller calls callee)."""
    caller: str             # e.g., "function parse", "class Parser"
    callee: str             # e.g., "function tokenize"
    caller_file: str
    callee_file: Optional[str] = None
    generator: str = "design_interfaces"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format."""
        return {
            "caller": self.caller,
            "callee": self.callee,
            "caller_file": self.caller_file,
            "callee_file": self.callee_file,
            "edge_type": "invokes",
            "generator": self.generator,
        }


@dataclass
class ReferenceEdge:
    """Type reference relationship (unit references type)."""
    unit: str               # e.g., "function process"
    referenced_type: str    # e.g., "Config"
    source_file: str
    type_file: Optional[str] = None
    generator: str = "design_interfaces"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format."""
        return {
            "unit": self.unit,
            "referenced_type": self.referenced_type,
            "source_file": self.source_file,
            "type_file": self.type_file,
            "edge_type": "references",
            "generator": self.generator,
        }


# ============================================================================
# Interface Unit
# ============================================================================

@dataclass
class InterfaceUnit:
    """A single interface unit (function or class)."""
    name: str               # e.g., "class Parser", "function parse"
    file_path: str
    subtree_name: str
    features: List[str]     # feature paths this unit implements (existing + new combined)
    code: str               # interface source code
    handler_added: bool = False  # True if added by InterfaceReviewer.add_interface handler
                                 # (protected from orphan-prune; carries _handler_added tag)

    @property
    def key(self) -> str:
        """Unique key for this unit: file_path::name."""
        return f"{self.file_path}::{self.name}"

    @property
    def unit_type(self) -> str:
        """Return 'class' or 'function'."""
        if self.name.startswith("class "):
            return "class"
        elif self.name.startswith("function "):
            return "function"
        return "unknown"

    @property
    def bare_name(self) -> str:
        """Return name without 'class ' or 'function ' prefix."""
        parts = self.name.split(" ", 1)
        return parts[1] if len(parts) == 2 else self.name


# ============================================================================
# Summary Data Classes
# ============================================================================

@dataclass
class OrphanFeature:
    """A feature whose implementing unit was pruned."""
    feature_path: str
    unit_key: str  # format: "file_path::unit_name"

    def to_dict(self) -> Dict[str, str]:
        return {
            "feature_path": self.feature_path,
            "unit_key": self.unit_key,
        }


@dataclass
class PruneSummary:
    """Summary of orphan pruning operation."""
    pruned_units: List["InterfaceUnit"] = field(default_factory=list)
    pruned_files: List[str] = field(default_factory=list)
    orphan_features: List[OrphanFeature] = field(default_factory=list)
    surviving_feature_paths: Set[str] = field(default_factory=set)

    def get_orphan_features_list(self) -> List[Dict[str, str]]:
        """Get orphan features as list of dicts for JSON output."""
        return [of.to_dict() for of in self.orphan_features]


@dataclass
class RPGUpdateSummary:
    """Summary of RPG update operation."""
    updated_features: int = 0
    created_new_features: int = 0
    added_same_unit_edges: int = 0
    added_dependency_edges: int = 0
    marked_entry_points: int = 0
    skipped_features: int = 0
    pruned_feature_nodes: int = 0
    pruned_parent_nodes: int = 0
    pruned_edges: int = 0


# ============================================================================
# Interfaces Store
# ============================================================================

class InterfacesStore:
    """Unified store for all interface data.

    Provides:
    - Single source of truth for units and edges
    - Type-safe CRUD operations
    - Auto-maintained indexes for efficient lookups
    - Pruning and RPG update as unified methods
    """

    def __init__(self):
        # Primary data: unit_key -> InterfaceUnit
        self._units: Dict[str, InterfaceUnit] = {}

        # Edge lists
        self._inheritance_edges: List[InheritanceEdge] = []
        self._invocation_edges: List[InvocationEdge] = []
        self._reference_edges: List[ReferenceEdge] = []
        # Preserved original coarse-grained data flow edges
        self._original_data_flow_edges: List[Dict[str, Any]] = []

        # Auto-maintained indexes
        self._file_to_units: Dict[str, List[str]] = defaultdict(list)  # file -> [unit_keys]
        self._subtree_to_files: Dict[str, Set[str]] = defaultdict(set)  # subtree -> {files}
        self._feature_to_units: Dict[str, Set[str]] = defaultdict(set)  # feature -> {unit_keys}
        self._class_to_file: Dict[str, str] = {}   # bare_class_name -> file_path
        self._function_to_file: Dict[str, str] = {}  # bare_function_name -> file_path

        # Entry points (set after global review)
        self._entry_point_keys: Set[str] = set()

        # Subtree ordering
        self.subtree_order: List[str] = []

        # Global review metadata
        self._global_review: Dict[str, Any] = {}

        # New features created during interface design
        self._new_features: Dict[str, str] = {}  # feature_path -> unit_key that created it

    # ========================================================================
    # Unit CRUD Operations
    # ========================================================================

    def add_unit(self, unit: InterfaceUnit) -> None:
        """Add a unit and update all indexes.

        Args:
            unit: The InterfaceUnit to add
        """
        key = unit.key
        self._units[key] = unit

        # Update file index
        if key not in self._file_to_units[unit.file_path]:
            self._file_to_units[unit.file_path].append(key)

        # Update subtree index
        self._subtree_to_files[unit.subtree_name].add(unit.file_path)

        # Update feature index
        for feature_path in unit.features:
            self._feature_to_units[feature_path].add(key)

        # Update symbol resolution indexes
        if unit.unit_type == "class":
            self._class_to_file[unit.bare_name] = unit.file_path
        elif unit.unit_type == "function":
            self._function_to_file[unit.bare_name] = unit.file_path

    def remove_unit(self, key: str) -> Optional[InterfaceUnit]:
        """Remove a unit and clean up all related data.

        Args:
            key: Unit key in format "file_path::unit_name"

        Returns:
            The removed InterfaceUnit, or None if not found
        """
        unit = self._units.pop(key, None)
        if not unit:
            return None

        # Clean file index
        if key in self._file_to_units[unit.file_path]:
            self._file_to_units[unit.file_path].remove(key)

        # Clean feature index
        for feature_path in unit.features:
            self._feature_to_units[feature_path].discard(key)

        # Clean new features that this unit created
        features_to_remove = [fp for fp, uk in self._new_features.items() if uk == key]
        for fp in features_to_remove:
            del self._new_features[fp]

        # Clean symbol index
        if unit.unit_type == "class":
            if self._class_to_file.get(unit.bare_name) == unit.file_path:
                del self._class_to_file[unit.bare_name]
        elif unit.unit_type == "function":
            if self._function_to_file.get(unit.bare_name) == unit.file_path:
                del self._function_to_file[unit.bare_name]

        # Remove related edges
        self._remove_edges_involving_unit(unit)

        # Clean empty file entries
        if not self._file_to_units[unit.file_path]:
            del self._file_to_units[unit.file_path]
            self._subtree_to_files[unit.subtree_name].discard(unit.file_path)

        return unit

    def get_unit(self, key: str) -> Optional[InterfaceUnit]:
        """Get a unit by its key."""
        return self._units.get(key)

    def apply_backfill(self, backfilled: List[Dict[str, Any]]) -> int:
        """Synchronise deterministic feature backfill into the store.

        ``backfill_uncovered_features`` operates on the exported
        ``interfaces.json`` shape and returns audit entries with
        ``feature``, ``file_path``, and ``unit`` fields. This method mirrors
        those metadata-only additions into the canonical store so subsequent
        RPG updates see the same feature ownership and the feature index used
        by ``surviving_feature_paths`` stays current.

        Returns the number of entries that changed the store. Missing units
        are ignored: backfill never creates interface units.
        """
        synced = 0
        for entry in backfilled or []:
            if not isinstance(entry, dict):
                continue
            feature = str(entry.get("feature") or "").strip()
            file_path = str(entry.get("file_path") or "").strip()
            unit_name = str(entry.get("unit") or "").strip()
            if not feature or not file_path or not unit_name:
                continue

            key = f"{file_path}::{unit_name}"
            unit = self._units.get(key)
            if not unit:
                logger.warning(
                    "[InterfacesStore.apply_backfill] Unit not found for %s -> %s",
                    feature,
                    key,
                )
                continue

            changed = False
            if feature not in unit.features:
                unit.features.append(feature)
                changed = True
            if key not in self._feature_to_units[feature]:
                self._feature_to_units[feature].add(key)
                changed = True
            if changed:
                synced += 1

        return synced

    def get_units_for_file(self, file_path: str) -> List[InterfaceUnit]:
        """Get all units in a file."""
        keys = self._file_to_units.get(file_path, [])
        return [self._units[k] for k in keys if k in self._units]

    def get_units_for_subtree(self, subtree_name: str) -> List[InterfaceUnit]:
        """Get all units in a subtree."""
        files = self._subtree_to_files.get(subtree_name, set())
        units = []
        for file_path in files:
            units.extend(self.get_units_for_file(file_path))
        return units

    @property
    def all_units(self) -> Dict[str, InterfaceUnit]:
        """Return all units."""
        return self._units.copy()

    @property
    def all_unit_keys(self) -> Set[str]:
        """Return all unit keys."""
        return set(self._units.keys())

    @property
    def new_features(self) -> Dict[str, str]:
        """Return all new features created during interface design.

        Returns:
            Dict mapping feature_path -> unit_key that created it
        """
        return self._new_features.copy()

    def get_new_features_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all new features for reporting.

        Returns:
            List of dicts with feature info: path, unit_name, file_path, subtree
        """
        summary = []
        for feature_path, unit_key in self._new_features.items():
            unit = self._units.get(unit_key)
            if unit:
                summary.append({
                    "feature_path": feature_path,
                    "unit_name": unit.name,
                    "file_path": unit.file_path,
                    "subtree": unit.subtree_name,
                })
        return summary

    def register_new_feature(self, feature_path: str, unit_key: str) -> None:
        """Register a new feature created during interface design.

        Args:
            feature_path: The new feature path
            unit_key: The unit key that implements this feature
        """
        self._new_features[feature_path] = unit_key
        # Also add to feature index
        self._feature_to_units[feature_path].add(unit_key)

    # ========================================================================
    # Edge Operations
    # ========================================================================

    def add_inheritance_edge(self, edge: InheritanceEdge) -> None:
        """Add an inheritance edge."""
        self._inheritance_edges.append(edge)

    def add_invocation_edge(self, edge: InvocationEdge) -> None:
        """Add an invocation edge (self-calls are filtered)."""
        # Self-call filter
        bare_caller = edge.caller.split(" ", 1)[-1] if " " in edge.caller else edge.caller
        bare_callee = edge.callee.split(" ", 1)[-1] if " " in edge.callee else edge.callee
        if bare_caller == bare_callee and (edge.callee_file is None or edge.callee_file == edge.caller_file):
            return
        self._invocation_edges.append(edge)

    def add_reference_edge(self, edge: ReferenceEdge) -> None:
        """Add a type reference edge."""
        self._reference_edges.append(edge)

    def add_edge(self, edge: Union[InheritanceEdge, InvocationEdge, ReferenceEdge]) -> None:
        """Add any edge type."""
        if isinstance(edge, InheritanceEdge):
            self.add_inheritance_edge(edge)
        elif isinstance(edge, InvocationEdge):
            self.add_invocation_edge(edge)
        elif isinstance(edge, ReferenceEdge):
            self.add_reference_edge(edge)

    def _remove_edges_involving_unit(self, unit: InterfaceUnit) -> int:
        """Remove edges that reference the given unit.

        Returns:
            Number of edges removed
        """
        removed = 0
        unit_name = unit.name
        file_path = unit.file_path

        # Filter inheritance edges
        orig_len = len(self._inheritance_edges)
        self._inheritance_edges = [
            e for e in self._inheritance_edges
            if not self._edge_involves_unit(e, unit_name, file_path, "inheritance")
        ]
        removed += orig_len - len(self._inheritance_edges)

        # Filter invocation edges
        orig_len = len(self._invocation_edges)
        self._invocation_edges = [
            e for e in self._invocation_edges
            if not self._edge_involves_unit(e, unit_name, file_path, "invocation")
        ]
        removed += orig_len - len(self._invocation_edges)

        # Filter reference edges
        orig_len = len(self._reference_edges)
        self._reference_edges = [
            e for e in self._reference_edges
            if not self._edge_involves_unit(e, unit_name, file_path, "reference")
        ]
        removed += orig_len - len(self._reference_edges)

        return removed

    def _edge_involves_unit(
        self,
        edge: Union[InheritanceEdge, InvocationEdge, ReferenceEdge],
        unit_name: str,
        file_path: str,
        edge_type: str
    ) -> bool:
        """Check if an edge involves the specified unit."""
        if edge_type == "inheritance":
            e = edge  # type: InheritanceEdge
            # Child matches
            if e.child == unit_name or (e.child in unit_name):
                if not e.child_file or e.child_file == file_path:
                    return True
            # Parent matches
            if e.parent == unit_name or (e.parent in unit_name):
                if not e.parent_file or e.parent_file == file_path:
                    return True
        elif edge_type == "invocation":
            e = edge  # type: InvocationEdge
            if e.caller == unit_name:
                if not e.caller_file or e.caller_file == file_path:
                    return True
            if e.callee == unit_name:
                if not e.callee_file or e.callee_file == file_path:
                    return True
        elif edge_type == "reference":
            e = edge  # type: ReferenceEdge
            if e.unit == unit_name:
                if not e.source_file or e.source_file == file_path:
                    return True
        return False

    # ========================================================================
    # Entry Points
    # ========================================================================

    def set_entry_points(self, entry_points: List[Dict[str, Any]]) -> None:
        """Set entry point keys from global review result.

        Args:
            entry_points: List of dicts with 'file_path' and 'unit_name' keys
        """
        self._entry_point_keys.clear()
        for ep in entry_points:
            ep_file = ep.get("file_path", "")
            ep_unit = ep.get("unit_name", "")
            if ep_file and ep_unit:
                self._entry_point_keys.add(f"{ep_file}::{ep_unit}")

    def is_entry_point(self, key: str) -> bool:
        """Check if a unit is an entry point."""
        return key in self._entry_point_keys

    # ========================================================================
    # Call Graph Construction
    # ========================================================================

    def build_adjacency(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Build outgoing and incoming adjacency sets from edges.

        Returns:
            Tuple of (outgoing, incoming) where:
            - outgoing: {unit_key -> set of callee unit_keys}
            - incoming: {unit_key -> set of caller unit_keys}
        """
        outgoing: Dict[str, Set[str]] = defaultdict(set)
        incoming: Dict[str, Set[str]] = defaultdict(set)

        # Build name-to-keys lookup
        name_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key, unit in self._units.items():
            name_to_keys[unit.name].append(key)
            name_to_keys[unit.bare_name].append(key)

        # Process invocation edges
        for edge in self._invocation_edges:
            caller_key = self._resolve_edge_unit(edge.caller, edge.caller_file, name_to_keys)
            callee_key = self._resolve_edge_unit(edge.callee, edge.callee_file, name_to_keys)
            if caller_key and callee_key:
                outgoing[caller_key].add(callee_key)
                incoming[callee_key].add(caller_key)

        # Process inheritance edges
        for edge in self._inheritance_edges:
            child_key = self._resolve_edge_unit(f"class {edge.child}", edge.child_file, name_to_keys)
            parent_key = self._resolve_edge_unit(f"class {edge.parent}", edge.parent_file, name_to_keys)
            if child_key and parent_key:
                outgoing[child_key].add(parent_key)
                incoming[parent_key].add(child_key)

        # Process reference edges
        for edge in self._reference_edges:
            unit_key = self._resolve_edge_unit(edge.unit, edge.source_file, name_to_keys)
            type_key = self._resolve_edge_unit(f"class {edge.referenced_type}", edge.type_file, name_to_keys)
            if unit_key and type_key:
                outgoing[unit_key].add(type_key)
                incoming[type_key].add(unit_key)

        return dict(outgoing), dict(incoming)

    def _resolve_edge_unit(
        self,
        name: str,
        file_path: Optional[str],
        name_to_keys: Dict[str, List[str]]
    ) -> Optional[str]:
        """Resolve a unit name to its key."""
        # Try direct key match
        if file_path:
            direct_key = f"{file_path}::{name}"
            if direct_key in self._units:
                return direct_key

        # Try name lookup
        candidates = name_to_keys.get(name, [])
        if candidates:
            if file_path:
                # Prefer matching file
                for key in candidates:
                    if self._units[key].file_path == file_path:
                        return key
            return candidates[0]

        # Try bare name
        if " " in name:
            bare_name = name.split(" ", 1)[1]
            candidates = name_to_keys.get(bare_name, [])
            if candidates:
                return candidates[0]

        return None

    # ========================================================================
    # Pruning
    # ========================================================================

    def find_orphan_units(self) -> List[str]:
        """Find isolated units (no incoming/outgoing edges, not entry point).

        Handler-added units (``InterfaceUnit.handler_added == True``) are
        EXCLUDED from the candidate set: they were deliberately materialised
        by ``InterfaceReviewer._apply_add_interface`` during the global
        review step, and their lack of incoming edges is expected — they
        are the missing pieces the review just installed and the next
        iteration / downstream stages must wire them up.

        Returns:
            List of unit keys that are candidates for pruning
        """
        outgoing, incoming = self.build_adjacency()

        isolated_keys: List[str] = []
        for key, unit in self._units.items():
            if key in self._entry_point_keys:
                continue
            if unit.handler_added:
                continue  # protected: handler-added is treated as required
            # Type-like units (struct / enum / interface / ...) are
            # referenced, not invoked, so a missing edge is not "dead
            # code". Excluding them keeps the pruning detector aligned
            # with the convergence gate (check_call_graph_connectivity).
            if classify_unit_kind(unit.name) != "callable":
                continue
            has_outgoing = key in outgoing and len(outgoing[key]) > 0
            has_incoming = key in incoming and len(incoming[key]) > 0
            if not has_outgoing and not has_incoming:
                isolated_keys.append(key)

        return isolated_keys

    def get_orphan_unit_details(self, orphan_keys: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information about orphan units for review.

        Args:
            orphan_keys: List of unit keys to get details for

        Returns:
            List of dicts with unit details (key, code, features, subtree, file_path)
        """
        details = []
        for key in orphan_keys:
            unit = self._units.get(key)
            if unit:
                details.append({
                    "unit_key": key,
                    "unit_name": unit.name,
                    "file_path": unit.file_path,
                    "subtree": unit.subtree_name,
                    "code": unit.code,
                    "features": unit.features,
                })
        return details

    def prune_units(self, keys_to_prune: List[str]) -> PruneSummary:
        """Remove specified units from the store.

        Args:
            keys_to_prune: List of unit keys to remove

        Returns:
            PruneSummary with details of what was pruned
        """
        if not keys_to_prune:
            logger.info("[InterfacesStore.prune_units] No units to prune")
            return PruneSummary(
                surviving_feature_paths=set(self._feature_to_units.keys())
            )

        logger.info(
            f"[InterfacesStore.prune_units] Pruning {len(keys_to_prune)} units "
            f"out of {len(self._units)} total"
        )

        # Track feature -> unit_key mapping before removal
        feature_to_unit_key: Dict[str, str] = {}
        for fp, keys in self._feature_to_units.items():
            if keys:
                feature_to_unit_key[fp] = next(iter(keys))

        # Remove specified units
        pruned_units: List[InterfaceUnit] = []
        pruned_files: Set[str] = set()
        for key in keys_to_prune:
            unit = self.remove_unit(key)
            if unit:
                pruned_units.append(unit)
                logger.info(f"[InterfacesStore.prune_units] Pruned: {key}")

        # Identify files that were completely emptied
        for unit in pruned_units:
            if unit.file_path not in self._file_to_units:
                pruned_files.add(unit.file_path)
                logger.info(
                    f"[InterfacesStore.prune_units] File fully pruned: {unit.file_path}"
                )

        # Find orphan features
        pruned_key_set = {u.key for u in pruned_units}
        orphan_features: List[OrphanFeature] = []
        for feature_path, unit_key in feature_to_unit_key.items():
            if unit_key in pruned_key_set:
                orphan_features.append(OrphanFeature(
                    feature_path=feature_path,
                    unit_key=unit_key,
                ))

        if orphan_features:
            logger.info(
                f"[InterfacesStore.prune_units] {len(orphan_features)} features orphaned"
            )

        return PruneSummary(
            pruned_units=pruned_units,
            pruned_files=list(pruned_files),
            orphan_features=orphan_features,
            surviving_feature_paths=set(self._feature_to_units.keys()),
        )

    def prune_orphans(self) -> PruneSummary:
        """Remove truly isolated units (no incoming/outgoing edges, not entry point).

        This is a convenience method that finds orphans and prunes them all.

        Returns:
            PruneSummary with details of what was pruned
        """
        isolated_keys = self.find_orphan_units()
        return self.prune_units(isolated_keys)

    def add_edges(self, edges_by_type: Dict[str, List[Dict]]) -> int:
        """Add edges to the store (e.g., from orphan review completion).

        Args:
            edges_by_type: Dict with keys "inheritance_edges", "invocation_edges", "reference_edges"
                          Format matches interfaces.json:
                          - inheritance_edges: {child, parent, source_file, parent_file}
                          - invocation_edges: {caller, callee, caller_file, callee_file}
                          - reference_edges: {unit, referenced_type, source_file, type_file}

        Returns:
            Number of edges added
        """
        added = 0

        # Process inheritance edges
        for edge in edges_by_type.get("inheritance_edges", []):
            child = edge.get("child", "")
            parent = edge.get("parent", "")
            child_file = edge.get("source_file", "")
            parent_file = edge.get("parent_file")
            if child and parent and child_file:
                new_edge = InheritanceEdge(
                    child=child,
                    parent=parent,
                    child_file=child_file,
                    parent_file=parent_file,
                )
                if new_edge not in self._inheritance_edges:
                    self._inheritance_edges.append(new_edge)
                    added += 1
                    logger.info(f"[InterfacesStore.add_edges] Added inheritance: {child} -> {parent}")

        # Process invocation edges
        for edge in edges_by_type.get("invocation_edges", []):
            caller = edge.get("caller", "")
            callee = edge.get("callee", "")
            caller_file = edge.get("caller_file", "")
            callee_file = edge.get("callee_file")
            if caller and callee and caller_file:
                new_edge = InvocationEdge(
                    caller=caller,
                    callee=callee,
                    caller_file=caller_file,
                    callee_file=callee_file,
                )
                if new_edge not in self._invocation_edges:
                    self._invocation_edges.append(new_edge)
                    added += 1
                    logger.info(f"[InterfacesStore.add_edges] Added invocation: {caller} -> {callee}")

        # Process reference edges
        for edge in edges_by_type.get("reference_edges", []):
            unit = edge.get("unit", "")
            referenced_type = edge.get("referenced_type", "")
            source_file = edge.get("source_file", "")
            type_file = edge.get("type_file")
            if unit and referenced_type and source_file:
                new_edge = ReferenceEdge(
                    unit=unit,
                    referenced_type=referenced_type,
                    source_file=source_file,
                    type_file=type_file,
                )
                if new_edge not in self._reference_edges:
                    self._reference_edges.append(new_edge)
                    added += 1
                    logger.info(f"[InterfacesStore.add_edges] Added reference: {unit} -> {referenced_type}")

        return added

    # ========================================================================
    # RPG Update
    # ========================================================================

    def update_rpg(self, rpg_path: Path) -> RPGUpdateSummary:
        """Update RPG with interface design results.

        This method:
        1. Updates existing feature nodes' meta.path with implementation location
        2. Adds SAME_UNIT edges when multiple features share the same unit
        3. Adds dependency edges (INHERITS, INVOKES, REFERENCES)
        4. Marks entry points
        5. Prunes orphan features from RPG

        Args:
            rpg_path: Path to repo_rpg.json

        Returns:
            RPGUpdateSummary with operation counts
        """
        from rpg.service import RPGService
        from rpg.models import Node, NodeType, EdgeType

        summary = RPGUpdateSummary()

        if not rpg_path.exists():
            logger.warning(f"RPG file not found: {rpg_path}")
            return summary

        try:
            svc = RPGService.load(rpg_path)
        except Exception as e:
            logger.error(f"Failed to load RPG: {e}")
            return summary

        rpg = svc.rpg

        # Remove old edges by generator
        svc.refresh_stage_edges("design_interfaces")

        # Build feature name -> node mapping
        feature_nodes: Dict[str, Node] = {}
        for node in rpg.nodes.values():
            if node.node_type == "feature" or node.level == rpg.MAX_FEATURE_LEVEL:
                feature_nodes[node.name] = node
                feature_path = node.feature_path()
                if feature_path:
                    feature_nodes[feature_path] = node

        # Track unit -> list of feature nodes for SAME_UNIT edges
        unit_to_features: Dict[str, List[Node]] = defaultdict(list)

        # Update feature nodes with implementation paths
        for unit in self._units.values():
            impl_path = f"{unit.file_path}::{unit.name}"

            for feature_path in unit.features:
                feature_node = feature_nodes.get(feature_path)
                if not feature_node:
                    # Try by name
                    feature_name = feature_path.split("/")[-1] if "/" in feature_path else feature_path
                    feature_node = feature_nodes.get(feature_name)

                if not feature_node:
                    logger.debug(f"Feature node not found: {feature_path}")
                    summary.skipped_features += 1
                    continue

                # Infer type_name
                inferred_type: Optional[NodeType] = None
                if unit.unit_type == "class":
                    inferred_type = NodeType.CLASS
                elif unit.unit_type == "function":
                    inferred_type = NodeType.FUNCTION

                # Update via service for consistency
                svc.update_feature_mapping(feature_node, impl_path, inferred_type)

                summary.updated_features += 1
                unit_to_features[impl_path].append(feature_node)

        # Create new feature nodes for glue/orchestration code
        summary.created_new_features = self._create_new_feature_nodes(
            svc, rpg, feature_nodes, unit_to_features
        )

        # Add SAME_UNIT edges
        for impl_path, feature_list in unit_to_features.items():
            if len(feature_list) < 2:
                continue

            for i in range(len(feature_list)):
                for j in range(i + 1, len(feature_list)):
                    if svc.add_dependency_edge(
                        feature_list[i], feature_list[j],
                        EdgeType.SAME_UNIT, "design_interfaces",
                        description=f"Share implementation: {impl_path}",
                        bidirectional_dedup=True,
                    ):
                        summary.added_same_unit_edges += 1

        # Add dependency edges
        summary.added_dependency_edges += self._add_rpg_dependency_edges(svc)

        # Mark entry points
        summary.marked_entry_points = self._mark_rpg_entry_points(svc, rpg)

        # Prune orphan features
        prune_result = svc.prune_orphan_features(self.surviving_feature_paths)
        summary.pruned_feature_nodes = prune_result[0]
        summary.pruned_parent_nodes = prune_result[1]
        summary.pruned_edges = prune_result[2]

        # Save RPG
        svc.save(rpg_path)

        total_changes = (
            summary.updated_features + summary.created_new_features +
            summary.added_same_unit_edges + summary.added_dependency_edges +
            summary.marked_entry_points
        )
        if total_changes > 0:
            parts = [f"{summary.updated_features} features updated"]
            if summary.created_new_features > 0:
                parts.append(f"{summary.created_new_features} new features created")
            parts.append(f"{summary.added_same_unit_edges} SAME_UNIT")
            parts.append(f"{summary.added_dependency_edges} dependency edges")
            parts.append(f"{summary.marked_entry_points} entry points")
            print(f"  RPG updated: {', '.join(parts)}")
        if summary.pruned_feature_nodes > 0:
            print(
                f"  RPG pruned: {summary.pruned_feature_nodes} feature nodes, "
                f"{summary.pruned_parent_nodes} parent nodes, "
                f"{summary.pruned_edges} edges"
            )

        return summary

    def _create_new_feature_nodes(
        self,
        svc,
        rpg,
        feature_nodes: Dict[str, Any],
        unit_to_features: Dict[str, List]
    ) -> int:
        """Create new feature nodes in RPG for glue/orchestration code.

        Args:
            svc: RPGService instance for node creation
            rpg: The RPG object (for read-only node lookups)
            feature_nodes: Existing feature name -> node mapping (will be updated)
            unit_to_features: Mapping of impl_path -> feature nodes (will be updated)

        Returns:
            Number of new feature nodes created
        """
        from rpg.models import NodeType

        created = 0

        for feature_path, unit_key in self._new_features.items():
            unit = self._units.get(unit_key)
            if not unit:
                logger.warning(f"Unit not found for new feature: {feature_path} -> {unit_key}")
                continue

            # Skip if already exists
            if feature_path in feature_nodes:
                logger.debug(f"Feature already exists: {feature_path}")
                continue

            impl_path = f"{unit.file_path}::{unit.name}"

            # Parse feature path to determine parent
            # Format: "Subtree Name/category/subcategory/feature name"
            path_parts = feature_path.split("/")
            if len(path_parts) < 2:
                logger.warning(f"Invalid new feature path format: {feature_path}")
                continue

            feature_name = path_parts[-1]
            subtree_name = path_parts[0]

            # Find parent node - try to find the closest existing parent
            # Prefer full path match at each level; fall back to bare name.
            parent_node = None
            for i in range(len(path_parts) - 1, 0, -1):
                parent_path = "/".join(path_parts[:i])
                if parent_path in feature_nodes:
                    parent_node = feature_nodes[parent_path]
                    break
                parent_name = path_parts[i - 1]
                # Skip name lookup when it equals path (i.e., i == 1, single-segment)
                if parent_name != parent_path and parent_name in feature_nodes:
                    parent_node = feature_nodes[parent_name]
                    break

            # Last-resort: scan all nodes (not only features) for subtree_name
            if not parent_node:
                for node in rpg.nodes.values():
                    if node.name == subtree_name:
                        parent_node = node
                        break

            if not parent_node:
                logger.warning(
                    f"Cannot find parent for new feature: {feature_path}. "
                    f"Creating as root-level node."
                )

            # Create new feature node
            type_name = NodeType.CLASS if unit.unit_type == "class" else NodeType.FUNCTION

            if parent_node:
                new_node = svc.add_feature_node(
                    name=feature_name,
                    parent=parent_node,
                    impl_path=impl_path,
                    type_name=type_name,
                    generator="design_interfaces",
                    description=f"Auto-created for glue code: {unit.name}",
                )
            else:
                # No parent found — create as root-level orphan via rpg.add_node
                # to ensure _graph wiring and ID-collision safeguards apply.
                from rpg.models import Node, NodeMetaData, uuid8 as _uuid8
                new_node = Node(
                    id=f"{feature_name}_{_uuid8()}",
                    name=feature_name,
                    node_type="feature",
                    meta=NodeMetaData(
                        path=impl_path,
                        description=f"Auto-created for glue code: {unit.name}",
                        generator="design_interfaces",
                        type_name=type_name,
                    )
                )
                new_node.level = rpg.MAX_FEATURE_LEVEL
                rpg.add_node(new_node)

            # Update feature_nodes mapping
            feature_nodes[feature_path] = new_node
            feature_nodes[feature_name] = new_node

            # Update unit_to_features for SAME_UNIT edges
            unit_to_features[impl_path].append(new_node)

            created += 1
            logger.info(f"Created new feature node: {feature_path} -> {new_node.id}")

        return created

    def _add_rpg_dependency_edges(self, svc) -> int:
        """Add INHERITS, INVOKES, REFERENCES edges to RPG."""
        from rpg.models import EdgeType

        added = 0

        # Inheritance edges
        for edge in self._inheritance_edges:
            child_node = svc.find_node_by_unit_name(edge.child)
            parent_node = svc.find_node_by_unit_name(edge.parent)
            if child_node and parent_node:
                if svc.add_dependency_edge(
                    child_node, parent_node, EdgeType.INHERITS,
                    "design_interfaces",
                    description=f"{edge.child} inherits from {edge.parent}",
                ):
                    added += 1

        # Invocation edges
        for edge in self._invocation_edges:
            caller_node = svc.find_node_by_unit_name(edge.caller)
            callee_node = svc.find_node_by_unit_name(edge.callee)
            if caller_node and callee_node:
                if svc.add_dependency_edge(
                    caller_node, callee_node, EdgeType.INVOKES,
                    "design_interfaces",
                    description=f"{edge.caller} invokes {edge.callee}",
                ):
                    added += 1

        # Reference edges
        for edge in self._reference_edges:
            unit_node = svc.find_node_by_unit_name(edge.unit)
            type_node = svc.find_node_by_unit_name(edge.referenced_type)
            if unit_node and type_node:
                if svc.add_dependency_edge(
                    unit_node, type_node, EdgeType.REFERENCES,
                    "design_interfaces",
                    description=f"{edge.unit} references type {edge.referenced_type}",
                ):
                    added += 1

        return added

    def _mark_rpg_entry_points(self, svc, rpg) -> int:
        """Mark entry points on RPG nodes."""
        marked = 0
        global_review = self._global_review
        entry_points = global_review.get("entry_points", [])

        for ep in entry_points:
            ep_unit = ep.get("unit_name", "")
            ep_file = ep.get("file_path", "")
            ep_rationale = ep.get("rationale", "")

            if not ep_unit:
                continue

            ep_node = svc.find_node_by_unit_name(ep_unit)

            if not ep_node:
                expected_path = f"{ep_file}::{ep_unit}" if ep_file else ""
                if expected_path:
                    for node in rpg.nodes.values():
                        if node.meta and node.meta.path == expected_path:
                            ep_node = node
                            break

            if ep_node:
                svc.mark_entry_point(ep_node, ep_rationale)
                marked += 1

        return marked

    @property
    def surviving_feature_paths(self) -> Set[str]:
        """Get all feature paths that have at least one unit."""
        return set(self._feature_to_units.keys())

    # ========================================================================
    # Symbol Resolution (for GlobalInterfaceRegistry compatibility)
    # ========================================================================

    def resolve_callee(self, callee_name: str) -> Optional[str]:
        """Resolve a callee name to its file_path.

        Compatible with GlobalInterfaceRegistry.resolve_callee().
        """
        # Exact match in class/function indexes
        if callee_name in self._class_to_file:
            return self._class_to_file[callee_name]
        if callee_name in self._function_to_file:
            return self._function_to_file[callee_name]

        # Try with prefix stripped
        stripped = callee_name
        if callee_name.startswith("class "):
            stripped = callee_name[len("class "):]
        elif callee_name.startswith("function "):
            stripped = callee_name[len("function "):]

        if stripped != callee_name:
            if stripped in self._class_to_file:
                return self._class_to_file[stripped]
            if stripped in self._function_to_file:
                return self._function_to_file[stripped]

        # Case-insensitive fallback
        callee_lower = callee_name.lower()
        for name, path in self._class_to_file.items():
            if name.lower() == callee_lower:
                return path
        for name, path in self._function_to_file.items():
            if name.lower() == callee_lower:
                return path

        return None

    def get_all_public_symbols(self) -> Dict[str, str]:
        """Return {symbol_name: file_path} for all registered symbols."""
        symbols = {}
        symbols.update(self._class_to_file)
        symbols.update(self._function_to_file)
        return symbols

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_interfaces_json(self) -> Dict[str, Any]:
        """Export to interfaces.json format.

        Returns:
            Dict compatible with current interfaces.json structure
        """
        subtrees: Dict[str, Any] = {}

        for subtree_name in self.subtree_order:
            files = sorted(self._subtree_to_files.get(subtree_name, set()))
            subtree_interfaces: Dict[str, Any] = {}

            for file_path in files:
                unit_keys = self._file_to_units.get(file_path, [])
                units = [self._units[k] for k in unit_keys if k in self._units]

                if not units:
                    continue

                file_dict: Dict[str, Any] = {
                    "units": [u.name for u in units],
                    "units_to_features": {u.name: u.features for u in units},
                    "units_to_code": {u.name: u.code for u in units},
                    "file_code": dedup_file_code(u.code for u in units),
                }
                # Preserve handler-added tag for downstream diagnostics
                # and so a subsequent ``from_legacy_format`` round-trip
                # keeps the prune protection.
                handler_added_units = [u.name for u in units if u.handler_added]
                if handler_added_units:
                    file_dict["_handler_added"] = handler_added_units
                subtree_interfaces[file_path] = file_dict

            subtrees[subtree_name] = {
                "files_order": files,
                "interfaces": subtree_interfaces,
            }

        result = {
            "subtrees": subtrees,
            "subtree_order": self.subtree_order,
            "implemented_subtrees": {
                st: sorted(self._subtree_to_files.get(st, set()))
                for st in self.subtree_order
            },
            "enhanced_data_flow": {
                "original_edges": self._original_data_flow_edges,
                "inheritance_edges": [e.to_dict() for e in self._inheritance_edges],
                "invocation_edges": [e.to_dict() for e in self._invocation_edges],
                "reference_edges": [e.to_dict() for e in self._reference_edges],
            },
            "success": True,
        }

        # Include new features summary
        if self._new_features:
            result["new_features"] = self.get_new_features_summary()

        # Include global review if set
        if self._global_review:
            result["global_review"] = self._global_review

        return result

    @classmethod
    def from_legacy_format(
        cls,
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Optional[Dict[str, Any]] = None,
        global_review: Optional[Dict[str, Any]] = None,
    ) -> "InterfacesStore":
        """Construct store from current interfaces_data dict format.

        Args:
            interfaces_data: The interfaces.json dict structure
            enhanced_data_flow: The enhanced_data_flow dict (or from interfaces_data)
            global_review: Global review results (or from interfaces_data)

        Returns:
            InterfacesStore populated with units and edges
        """
        store = cls()
        store.subtree_order = interfaces_data.get("subtree_order", [])

        # Use enhanced_data_flow from parameter or from interfaces_data
        if enhanced_data_flow is None:
            enhanced_data_flow = interfaces_data.get("enhanced_data_flow", {})

        store._original_data_flow_edges = enhanced_data_flow.get("original_edges", [])

        # Load units from subtrees
        subtrees = interfaces_data.get("subtrees", {})
        for subtree_name, subtree_data in subtrees.items():
            file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))

            for file_path, file_data in file_interfaces.items():
                units_to_features = file_data.get("units_to_features", {})
                units_to_code = file_data.get("units_to_code", {})
                # Track which units were materialised by the
                # InterfaceReviewer.add_interface handler so we can protect
                # them from orphan-prune in `find_orphan_units`.
                handler_added_set = set(file_data.get("_handler_added", []) or [])

                for unit_name in file_data.get("units", []):
                    unit = InterfaceUnit(
                        name=unit_name,
                        file_path=file_path,
                        subtree_name=subtree_name,
                        features=units_to_features.get(unit_name, []),
                        code=units_to_code.get(unit_name, ""),
                        handler_added=unit_name in handler_added_set,
                    )
                    store.add_unit(unit)

        # Load new features from top-level list
        for nf in interfaces_data.get("new_features", []):
            feature_path = nf.get("feature_path", "")
            file_path = nf.get("file_path", "")
            unit_name = nf.get("unit_name", "")
            if feature_path and file_path and unit_name:
                unit_key = f"{file_path}::{unit_name}"
                store.register_new_feature(feature_path, unit_key)

        # Load edges
        for e in enhanced_data_flow.get("inheritance_edges", []):
            store._inheritance_edges.append(InheritanceEdge(
                child=e.get("child", ""),
                parent=e.get("parent", ""),
                child_file=e.get("source_file", ""),
                parent_file=e.get("parent_file"),
                generator=e.get("generator", "design_interfaces"),
            ))

        for e in enhanced_data_flow.get("invocation_edges", []):
            store._invocation_edges.append(InvocationEdge(
                caller=e.get("caller", ""),
                callee=e.get("callee", ""),
                caller_file=e.get("caller_file", ""),
                callee_file=e.get("callee_file"),
                generator=e.get("generator", "design_interfaces"),
            ))

        for e in enhanced_data_flow.get("reference_edges", []):
            store._reference_edges.append(ReferenceEdge(
                unit=e.get("unit", ""),
                referenced_type=e.get("referenced_type", ""),
                source_file=e.get("source_file", ""),
                type_file=e.get("type_file"),
                generator=e.get("generator", "design_interfaces"),
            ))

        # Load global review
        if global_review is None:
            global_review = interfaces_data.get("global_review", {})
        store._global_review = global_review

        # Set entry points from global review
        entry_points = global_review.get("entry_points", [])
        store.set_entry_points(entry_points)

        return store

    # ========================================================================
    # Debug / Info
    # ========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Get summary statistics."""
        return {
            "units": len(self._units),
            "files": len(self._file_to_units),
            "subtrees": len(self._subtree_to_files),
            "features": len(self._feature_to_units),
            "inheritance_edges": len(self._inheritance_edges),
            "invocation_edges": len(self._invocation_edges),
            "reference_edges": len(self._reference_edges),
            "entry_points": len(self._entry_point_keys),
        }
