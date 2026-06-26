#!/usr/bin/env python3
"""Global Interface Review Module.

Implements the Global Review phase for interface design, including:
- Entry point identification via LLM semantic reasoning
- Wiring completeness / call graph connectivity checks
- Cross-module type compatibility validation
- Automatic fix suggestions and application

This module is invoked AFTER all per-subtree interface designs are complete,
but BEFORE the final interfaces.json is saved.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any, Set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import LLMClient

# AST inspection routes through the Python backend's
# ``find_main_block_lineno`` helper so entry-point splicing shares the
# same parser abstraction as the rest of interface design.
from decoder_lang import get_backend
from decoder_lang.prompt_directive import with_language_directive

from .interface_agent import (
    GlobalInterfaceRegistry,
    DependencyCollector,
)
from .interface_prompts import ORPHAN_REVIEW_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# Non-production feature classification
# ============================================================================

# Top-level feature categories whose units are driven by an EXTERNAL runner
# rather than by repository code: test functions are discovered and invoked
# by the test runner, build targets by ``make`` / ``cmake``. Such a unit
# legitimately has no incoming *invocation* edge in the production call
# graph, so the "no incoming edge => dead code" orphan heuristic is a false
# positive for it — exactly like the type-like case handled by ``is_callable``,
# but along an orthogonal axis (the unit IS callable, it is just called from
# outside the graph).
#
# Categories are matched on the leading segment of a feature path
# (``"Testing/error reporting/..."`` -> ``"testing"``) and on the subtree
# name, both lower-cased. The set is language-agnostic: the planner emits
# these category names independently of the target language.
NON_PRODUCTION_FEATURE_CATEGORIES: frozenset[str] = frozenset({
    "testing", "test", "tests", "test infrastructure", "test suite",
    "build system", "build", "build configuration",
    "tooling", "ci", "cd", "ci/cd",
})


def _is_non_production_feature(
    features: Optional[List[str]],
    subtree: str = "",
) -> bool:
    """Return True when a feature-bearing unit belongs to a test/build category.

    Such units are invoked by an external driver (test runner, ``make``),
    not by repository code, so a missing incoming invocation edge is not a
    coverage gap. Matching is on the leading segment of each feature path
    and on the subtree name, both lower-cased, against
    :data:`NON_PRODUCTION_FEATURE_CATEGORIES`. Language-agnostic.
    """
    if subtree and subtree.strip().lower() in NON_PRODUCTION_FEATURE_CATEGORIES:
        return True
    for feature in features or ():
        if not isinstance(feature, str):
            continue
        head = feature.split("/", 1)[0].strip().lower()
        if head in NON_PRODUCTION_FEATURE_CATEGORIES:
            return True
    return False


# ============================================================================
# Global Review Prompt
# ============================================================================

GLOBAL_INTERFACE_REVIEW_PROMPT = """
You are a senior software engineer reviewing the COMPLETE set of interfaces for an entire repository.

All subtrees have been designed. Your task is to review the interfaces holistically,
focusing on CROSS-MODULE integration — not individual interface quality.

## Input
- All designed interfaces (grouped by subtree)
- Data flow DAG (subtree-level dependencies)
- Import cross-validation warnings (symbols imported but not declared as calls)

## Review Tasks

### Task 1: Identify Top-Level Interfaces
Identify which units (classes/functions) are **top-level interfaces** — those that
are not expected to be called by other internal modules within this repository.

Top-level interfaces are NOT limited to "files named main.py". They are units whose
role in the architecture means they don't need an internal caller. This includes:
- Application entry points: a `MainLoop` class, a CLI `main()` function, an `Application` class
- Standalone submodules: components that can function independently (e.g., a `TestRunner`, a `Benchmark` harness)
- Externally-invoked APIs: interfaces designed to be called by external code, plugins, or frameworks
- Framework callbacks: handlers registered with an event system or framework.
  This explicitly includes (do not omit these):
  * HTTP route handlers (Flask `@route` / FastAPI / Django view functions) —
    even if not decorated in the signature, any unit whose role is "respond
    to an HTTP request" is invoked by the web framework, not by other code
    in the project. Always mark these as entry_points.
  * CLI subcommand handlers (click commands, argparse callbacks).
  * Event / signal subscribers, background workers, scheduled job entry
    functions, message-queue consumers.
  * Test fixtures / hooks that pytest, unittest, or other test runners
    invoke automatically.

Use semantic judgment based on the module's role and the project's architecture.
Lean towards MORE entry points when in doubt — false positives only add an
"externally invoked" tag, but false negatives create spurious orphan reports.

### Task 2: Wiring Completeness
- Does every non-top-level module's output have at least one consumer?
- Are there "island" modules that neither call nor are called by anyone?
- Do the identified top-level interfaces actually invoke the key subsystems?
- Are there missing orchestration layers?

### Task 3: Call Chain Realism
- Can you trace a realistic execution path from each top-level interface to leaf modules?
- Are the parameter/return types compatible across call boundaries?

### Task 4: Dependency Direction Consistency
- Do dependencies flow in the direction specified by the data_flow DAG?
- Are there undeclared reverse dependencies?

## Output
You must return ONLY a valid JSON object with the following structure (no other text):
{
  "entry_points": [
    {
      "file_path": "...",
      "unit_name": "...",
      "rationale": "..."
    }
  ],
  "orphan_modules": [
    {
      "file_path": "...",
      "unit_name": "...",
      "reason": "..."
    }
  ],
  "missing_wiring": [
    {
      "from_unit": "...",
      "from_file": "...",
      "to_unit": "...",
      "to_file": "...",
      "description": "..."
    }
  ],
  "type_mismatches": [
    {
      "file_path": "...",
      "unit_name": "...",
      "description": "..."
    }
  ],
  "orchestration_gaps": [
    {
      "description": "...",
      "suggested_location": "..."
    }
  ],
  "recommended_fixes": [
    {
      "action": "add_dependency",
      "file_path": "...",
      "unit_name": "...",
      "description": "...",
      "calls_to_add": [
        {"callee": "...", "callee_file": "...", "purpose": "..."}
      ]
    },
    {
      "action": "add_interface",
      "file_path": "src/.../foo.py",
      "unit_name": "function bar",
      "signature": "def bar(arg: T) -> R:",
      "docstring": "1-3 sentence summary of what this unit does.",
      "feature_path": "Category/Subcategory/feature name",
      "incoming_calls_from": ["function caller_a", "class CallerB"],
      "description": "Why this new interface is required."
    }
  ],
  "pass": true
}

Important:
- "pass" should be true only if there are no orphan modules, no missing wiring,
  and no orchestration gaps.
- recommended_fixes should contain concrete, actionable fixes.
- Each fix action must be one of: "add_dependency", "add_interface", "modify_interface".

Rules for `add_dependency`:
- include "calls_to_add" with callee name and (if known) file.

Rules for `add_interface` (auto-applied when valid):
- Use ONLY when no existing unit could possibly serve the role (e.g. a route
  handler is missing entirely). If two existing units just need to be wired
  together, use `add_dependency` instead — do NOT invent a new unit.
- `signature` MUST be a single-line language-appropriate interface/stub
  signature (for Python, a def/class header ending with `:`, e.g.
  `def list_todos() -> Response:` or `class TodoView(MethodView):`).
- `docstring` MUST be non-empty (1–3 sentences).
- `unit_name` MUST be prefixed with "function " or "class ".
- `feature_path` MUST be one of the feature paths declared in the skeleton;
  do NOT invent new features.
- `incoming_calls_from` is OPTIONAL but strongly recommended: list the
  existing unit(s) that should call this new interface, so the review can
  install the wiring edge in the same step (otherwise the new unit will
  immediately be reported as orphan).

Rules for `modify_interface`:
- This action has no auto-handler: it is recorded as an advisory
  `unapplied_fix` for manual follow-up and does NOT block `passed`. Use it
  sparingly and only for true architectural issues (e.g. breaking a
  circular import).
- Do NOT use modify_interface for cases solvable by add_dependency or
  add_interface.

Cross-cutting rules:
- Do NOT repeat in this iteration any fix that was reported as unapplied
  in the previous iteration (see "Previous Review Results" above when
  present); switch to a different action or omit it.
""".strip()


# ============================================================================
# Code-Based Structural Checks
# ============================================================================

def build_call_graph(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    """Build a directed call graph from interfaces and enhanced_data_flow.
    
    Returns:
        - outgoing: {unit_key -> set of callee unit_keys}
        - incoming: {unit_key -> set of caller unit_keys}
        - unit_to_file: {unit_key -> file_path}
    """
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    unit_to_file = {}
    
    # Collect all units
    subtrees = interfaces_data.get("subtrees", {})
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            for unit_name in file_data.get("units", []):
                unit_key = f"{file_path}::{unit_name}"
                unit_to_file[unit_key] = file_path
    
    # Build a lookup: bare_name -> unit_key(s)
    name_to_keys = defaultdict(list)
    for unit_key in unit_to_file:
        parts = unit_key.split("::", 1)
        if len(parts) == 2:
            unit_name = parts[1]
            # e.g., "class Foo" -> bare name "Foo"
            name_parts = unit_name.split(" ", 1)
            bare_name = name_parts[1] if len(name_parts) == 2 else unit_name
            name_to_keys[bare_name].append(unit_key)
            name_to_keys[unit_name].append(unit_key)
    
    # Process invocation edges from enhanced_data_flow
    for edge in enhanced_data_flow.get("invocation_edges", []):
        caller = edge.get("caller", "")
        callee = edge.get("callee", "")
        caller_file = edge.get("caller_file", "")
        callee_file = edge.get("callee_file", "")
        
        # Resolve caller key
        caller_key = f"{caller_file}::{caller}" if caller_file else None
        if caller_key and caller_key not in unit_to_file:
            # Try to find by name
            candidates = name_to_keys.get(caller, [])
            if candidates:
                caller_key = candidates[0]
            else:
                caller_key = None
        
        # Resolve callee key
        callee_key = None
        if callee_file:
            callee_key = f"{callee_file}::{callee}"
            if callee_key not in unit_to_file:
                # Try matching just by callee name
                for key in name_to_keys.get(callee, []):
                    if unit_to_file.get(key) == callee_file:
                        callee_key = key
                        break
                else:
                    candidates = name_to_keys.get(callee, [])
                    callee_key = candidates[0] if candidates else None
        else:
            candidates = name_to_keys.get(callee, [])
            callee_key = candidates[0] if candidates else None
        
        if caller_key and callee_key:
            outgoing[caller_key].add(callee_key)
            incoming[callee_key].add(caller_key)
    
    # Process inheritance edges
    for edge in enhanced_data_flow.get("inheritance_edges", []):
        child = edge.get("child", "")
        parent = edge.get("parent", "")
        child_candidates = name_to_keys.get(child, [])
        parent_candidates = name_to_keys.get(parent, [])
        
        if child_candidates and parent_candidates:
            child_key = child_candidates[0]
            parent_key = parent_candidates[0]
            outgoing[child_key].add(parent_key)
            incoming[parent_key].add(child_key)
    
    # Process reference edges
    for edge in enhanced_data_flow.get("reference_edges", []):
        unit = edge.get("unit", "")
        ref_type = edge.get("referenced_type", "")
        unit_candidates = name_to_keys.get(unit, [])
        type_candidates = name_to_keys.get(ref_type, [])
        
        if unit_candidates and type_candidates:
            unit_key = unit_candidates[0]
            type_key = type_candidates[0]
            outgoing[unit_key].add(type_key)
            incoming[type_key].add(unit_key)
    
    return dict(outgoing), dict(incoming), unit_to_file


def _build_unit_feature_index(
    interfaces_data: Dict[str, Any],
) -> Dict[str, Tuple[List[str], str]]:
    """Map ``file_path::unit_name`` -> (feature paths, subtree name).

    Lets the orphan checks consult a unit's feature category without
    re-walking the subtree tree per unit. Units absent from the index
    (no ``units_to_features`` entry) are treated as production code by the
    callers (the conservative default keeps real dead code detectable).
    """
    index: Dict[str, Tuple[List[str], str]] = {}
    subtrees = interfaces_data.get("subtrees", {})
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            units_to_features = file_data.get("units_to_features", {})
            for unit_name, features in units_to_features.items():
                index[f"{file_path}::{unit_name}"] = (
                    features if isinstance(features, list) else [],
                    subtree_name,
                )
    return index


def _unit_name_aliases(unit_name: str) -> Set[str]:
    """Return comparable aliases for an interface unit name."""
    raw = unit_name.strip()
    if not raw:
        return set()

    aliases = {raw}
    if " " in raw:
        aliases.add(raw.split(" ", 1)[1].strip())

    expanded = set()
    for alias in aliases:
        if not alias:
            continue
        expanded.add(alias)
        if "." in alias:
            expanded.add(alias.rsplit(".", 1)[-1])
        if "::" in alias:
            expanded.add(alias.rsplit("::", 1)[-1])
    return {alias for alias in expanded if alias}


def _is_advisory_unapplied_fix(fix: Dict[str, Any]) -> bool:
    """Return True when an unapplied fix is manual follow-up, not a blocker."""
    return (
        fix.get("advisory") is True
        or fix.get("manual_follow_up") is True
        or fix.get("action") == "modify_interface"
    )


def _build_entry_point_keys(
    entry_points: List[Dict[str, Any]],
    unit_to_file: Dict[str, str],
) -> Set[str]:
    """Resolve LLM entry-point records to concrete unit keys."""
    alias_to_keys: Dict[str, Set[str]] = defaultdict(set)
    for unit_key in unit_to_file:
        unit_name = unit_key.split("::", 1)[1] if "::" in unit_key else unit_key
        for alias in _unit_name_aliases(unit_name):
            alias_to_keys[alias].add(unit_key)

    entry_point_keys: Set[str] = set()
    for entry_point in entry_points:
        entry_file = str(entry_point.get("file_path") or "").strip()
        entry_unit = str(entry_point.get("unit_name") or "").strip()
        if entry_file and entry_unit:
            exact_key = f"{entry_file}::{entry_unit}"
            if exact_key in unit_to_file:
                entry_point_keys.add(exact_key)

        for alias in _unit_name_aliases(entry_unit):
            candidate_keys = alias_to_keys.get(alias, set())
            if not entry_file and len(candidate_keys) > 1:
                continue
            for unit_key in candidate_keys:
                if entry_file and unit_to_file.get(unit_key) != entry_file:
                    continue
                entry_point_keys.add(unit_key)
    return entry_point_keys


def _is_isolated_orphan(
    unit_key: str,
    unit_name: str,
    file_path: str,
    features: List[str],
    subtree: str,
    incoming: Dict[str, Set[str]],
    outgoing: Dict[str, Set[str]],
    entry_point_keys: Set[str],
    is_callable: Optional[Callable[[str], bool]],
    is_test_file: Optional[Callable[[str], bool]],
) -> bool:
    """Return True when a unit is a genuinely disconnected production orphan.

    Single source of truth shared by the unit-level connectivity gate and
    the feature-coverage gate so the two can never diverge (a past defect
    had them disagree, leaving stale orphan counts). A unit is an orphan
    only when ALL of the following hold:

    * it is not an entry point;
    * it is callable (type-like units are referenced, not invoked, so a
      missing incoming *invocation* edge is expected);
    * it is production code (test / build units are driven by an external
      runner, so a missing incoming edge is not dead code);
    * it is completely isolated — no incoming AND no outgoing edge.

    Requiring isolation on BOTH directions (rather than "no incoming") is
    what keeps roots / factories / framework callbacks — which have no
    static incoming edge but DO call into the graph — from being mistaken
    for dead code. The rule is identical for every language; all
    language-specific behaviour enters only through the injected
    ``is_callable`` / ``is_test_file`` predicates.
    """
    if unit_key in entry_point_keys:
        return False
    if is_callable is not None and not is_callable(unit_name):
        return False
    if _is_non_production_feature(features, subtree):
        return False
    if is_test_file is not None and is_test_file(file_path):
        return False
    has_incoming = bool(incoming.get(unit_key))
    has_outgoing = bool(outgoing.get(unit_key))
    return not has_incoming and not has_outgoing


def check_call_graph_connectivity(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    entry_points: List[Dict[str, Any]],
    is_callable: Optional[Callable[[str], bool]] = None,
    is_test_file: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """Build a directed graph of all invocation edges and check connectivity.

    Identifies orphan units: *callable* units (functions / methods /
    classes) that are completely isolated — no incoming AND no outgoing
    edges — and are not entry points.

    ``is_callable`` is a per-language predicate (``backend.is_callable_unit``).
    When supplied, type-like units (struct / enum / interface / ...) are
    excluded from orphan candidacy: a data structure legitimately has no
    incoming *invocation* edge, so flagging it is a false positive.
    When ``None`` (legacy callers / tests) every unit is treated as
    callable, preserving the previous behaviour.

    ``is_test_file`` is a per-language predicate (``backend.is_test_file``).
    When supplied, units in test files are excluded from orphan candidacy
    — together with the language-agnostic test/build feature-category
    check — because a test or build unit is driven by an external runner,
    not by repository code, so its lack of an incoming edge is not dead
    code. ``None`` disables the file-level check (the category check still
    applies), preserving legacy behaviour.

    Requiring "no outgoing" as well mirrors
    :meth:`InterfacesStore.find_orphan_units` so the convergence gate and
    the pruning detector share one definition of "orphan".

    Returns:
        Dict with keys: orphan_units, total_units, entry_point_count
    """
    outgoing, incoming, unit_to_file = build_call_graph(interfaces_data, enhanced_data_flow)
    feature_index = _build_unit_feature_index(interfaces_data)

    all_units = set(unit_to_file.keys())

    entry_point_keys = _build_entry_point_keys(entry_points, unit_to_file)

    # Orphan = callable + completely isolated (no incoming, no outgoing).
    orphan_units = []
    for unit_key in all_units:
        unit_name = unit_key.split("::", 1)[1] if "::" in unit_key else unit_key
        features, subtree = feature_index.get(unit_key, ([], ""))
        if _is_isolated_orphan(
            unit_key, unit_name, unit_to_file.get(unit_key, ""),
            features, subtree, incoming, outgoing, entry_point_keys,
            is_callable, is_test_file,
        ):
            orphan_units.append({
                "unit_key": unit_key,
                "file_path": unit_to_file.get(unit_key, ""),
            })

    return {
        "orphan_units": orphan_units,
        "total_units": len(all_units),
        "entry_point_count": len(entry_point_keys),
    }


def check_feature_dependency_coverage(
    interfaces_data: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    entry_points: List[Dict[str, Any]],
    is_callable: Optional[Callable[[str], bool]] = None,
    is_test_file: Optional[Callable[[str], bool]] = None,
) -> List[Dict[str, Any]]:
    """Check that feature-bearing units are not isolated from the graph.

    ``is_callable`` is a per-language predicate (``backend.is_callable_unit``).
    When supplied, type-like feature-bearing units (struct / enum / ...)
    are excluded: a data structure that carries a feature is "used" by
    being referenced, not invoked, so a missing incoming *invocation*
    edge is not a coverage gap. When ``None`` every unit is checked
    (legacy behaviour).

    ``is_test_file`` is a per-language predicate (``backend.is_test_file``).
    When supplied, units in test files are excluded — together with the
    language-agnostic test/build feature-category check — because a test
    or build unit is invoked by an external runner (test framework /
    ``make``), not by repository code, so a missing incoming edge is not a
    coverage gap. ``None`` disables the file-level check (the category
    check still applies), preserving legacy behaviour.

    Returns: list of orphan features attached to isolated units.
    """
    outgoing, incoming, unit_to_file = build_call_graph(interfaces_data, enhanced_data_flow)
    
    entry_point_keys = _build_entry_point_keys(entry_points, unit_to_file)
    
    orphan_features = []
    subtrees = interfaces_data.get("subtrees", {})
    
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            units_to_features = file_data.get("units_to_features", {})
            for unit_name, features in units_to_features.items():
                unit_key = f"{file_path}::{unit_name}"
                if _is_isolated_orphan(
                    unit_key, unit_name, file_path, features, subtree_name,
                    incoming, outgoing, entry_point_keys,
                    is_callable, is_test_file,
                ):
                    orphan_features.append({
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "features": features,
                        "subtree": subtree_name,
                    })
    
    return orphan_features


# ============================================================================
# Handler helpers
# ============================================================================

# Cheap signature sanity check used by ``_apply_add_interface``.
#
# The prompt contract requires ``signature`` to be a single-line Python
# header ending with ``:`` (e.g. ``def foo(arg: int) -> str:`` or
# ``class Bar(Base):``). Enforce both shape and terminator so a
# truncated LLM response like ``def foo(`` cannot be injected as
# syntactically invalid code into ``file_code``.
_SIGNATURE_RE = __import__("re").compile(r"^(def |class )[A-Za-z_]\w*\s*\(.*\)\s*(->\s*.+)?\s*:\s*$|^class\s+[A-Za-z_]\w*\s*:\s*$")


def _insert_unit_into_file_code(file_code: str, stub: str) -> str:
    """Insert a Python ``stub`` into ``file_code`` at a safe location.

    Only reached for Python projects: ``_apply_fixes`` skips the
    ``add_interface`` action (the sole caller) for non-Python backends,
    because stub synthesis emits Python ``def``/``class`` syntax.

    Preferred insertion point is **immediately before** any top-level
    ``if __name__ == "__main__":`` block, so handler-added units do not
    end up after the module's main guard (which would make them look like
    they are defined inside the guard or as dead code, depending on the
    reader's eye).

    Fallbacks (in order):
    - Append to the end of the file with two blank lines of separation.

    Never raises: a ``SyntaxError`` in ``file_code`` falls through to the
    append branch so the handler can never crash the review loop.
    """
    if not file_code.strip():
        return stub

    # Locate the module's main guard via the Python backend so a handler-
    # added unit is spliced before it. A missing guard (or a parse error)
    # falls through to the append branch below.
    backend = get_backend("python")
    find_main = getattr(backend, "find_main_block_lineno", None)
    main_lineno = find_main(file_code) if find_main is not None else None

    if main_lineno is None:
        return file_code.rstrip() + "\n\n\n" + stub

    lines = file_code.splitlines()
    insert_at = max(main_lineno - 1, 0)  # ast.lineno is 1-based
    prefix = lines[:insert_at]
    suffix = lines[insert_at:]
    # Ensure separation: one blank line before stub, two blank lines after.
    return "\n".join(prefix + ["", stub.rstrip(), "", ""] + suffix)


# ============================================================================
# Interface Reviewer
# ============================================================================

class InterfaceReviewer:
    """Global interface reviewer that performs holistic review after all subtrees are designed.
    
    Combines:
    1. LLM-based semantic review (entry point identification, wiring, consistency)
    2. Code-based structural checks (call graph connectivity, feature coverage)
    3. Automatic fix application (add missing dependencies / interfaces)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        trajectory: Optional[Any] = None,
        step_id: Optional[int] = None,
        target_language: Optional[str] = None,
    ):
        if llm_client is None:
            self.llm = LLMClient(trajectory=trajectory, step_id=step_id)
        else:
            self.llm = llm_client
        # Target-language backend. Structural checks and dependency-edge
        # fixes are language-agnostic; only interface-stub synthesis
        # (``add_interface``) is Python-specific and is skipped for other
        # languages. Defaults to Python so standalone callers are unaffected.
        self.backend = get_backend(target_language or "python")
        self.logger = logging.getLogger(__name__)
    
    def review_and_fix(
        self,
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        import_warnings: List[Dict[str, str]],
        data_flow_edges: List[Dict[str, Any]],
        dependency_collector: Optional[DependencyCollector] = None,
        max_fix_iterations: int = 2,
        skeleton_features: Optional[Set[str]] = None,
        rpg_features: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full global review and fix cycle.
        
        Steps:
        1. LLM global review (entry point identification + wiring + consistency)
        2. Code-based checks using LLM-identified entry points
        3. If issues found: apply recommended_fixes from LLM
        4. Re-run code checks
        5. Repeat until pass or max iterations
        
        Args:
            interfaces_data: The full interfaces result dict
            enhanced_data_flow: The enhanced_data_flow dict from DependencyCollector
            global_registry: The GlobalInterfaceRegistry with all designed interfaces
            import_warnings: List of import cross-validation warnings
            data_flow_edges: Original data flow DAG edges
            dependency_collector: DependencyCollector for adding new edges
            max_fix_iterations: Maximum number of review-fix cycles
            skeleton_features: Canonical set of feature paths from skeleton.json.
                Required for ``add_interface`` handler to validate that an
                LLM-requested ``feature_path`` is real (not invented).
            rpg_features: Set of feature paths currently present in
                ``repo_rpg.json``. ``add_interface`` rejects requests for
                features that have been pruned from the RPG, because
                ``update_rpg`` would silently drop them otherwise.

        Returns:
            Dict with review results, applied fixes, and updated interfaces_data
        """
        self.logger.info("[InterfaceReviewer] Starting global interface review")
        
        review_history = []
        
        for iteration in range(max_fix_iterations):
            self.logger.info(f"[InterfaceReviewer] Review iteration {iteration + 1}/{max_fix_iterations}")
            
            # Step 1: LLM global review
            llm_review = self._run_llm_review(
                interfaces_data=interfaces_data,
                enhanced_data_flow=enhanced_data_flow,
                global_registry=global_registry,
                import_warnings=import_warnings,
                data_flow_edges=data_flow_edges,
                iteration=iteration,
                previous_reviews=review_history,
            )
            
            if not llm_review:
                self.logger.warning("[InterfaceReviewer] LLM review returned empty result")
                break
            
            entry_points = llm_review.get("entry_points", [])
            self.logger.info(
                f"[InterfaceReviewer] LLM identified {len(entry_points)} entry points"
            )
            for ep in entry_points:
                self.logger.info(
                    f"  Entry point: {ep.get('unit_name', '?')} in {ep.get('file_path', '?')} "
                    f"— {ep.get('rationale', '')}"
                )
            
            # Step 2: Code-based structural checks
            connectivity = check_call_graph_connectivity(
                interfaces_data, enhanced_data_flow, entry_points,
                is_callable=self.backend.is_callable_unit,
                is_test_file=self.backend.is_test_file,
            )
            feature_orphans = check_feature_dependency_coverage(
                interfaces_data, enhanced_data_flow, entry_points,
                is_callable=self.backend.is_callable_unit,
                is_test_file=self.backend.is_test_file,
            )

            self.logger.info(
                f"[InterfaceReviewer] Connectivity: "
                f"{connectivity['total_units']} total units, "
                f"{connectivity['entry_point_count']} entry points, "
                f"{len(connectivity['orphan_units'])} orphan units"
            )
            self.logger.info(
                f"[InterfaceReviewer] Feature coverage: {len(feature_orphans)} orphan features"
            )

            review_result = {
                "iteration": iteration + 1,
                "llm_review": llm_review,
                "orphan_units": connectivity["orphan_units"],
                "feature_orphans": feature_orphans,
                "entry_points": entry_points,
            }
            review_history.append(review_result)
            
            # Step 3: Check if passed
            llm_passed = llm_review.get("pass", False)
            code_passed = (
                len(connectivity["orphan_units"]) == 0
                and len(feature_orphans) == 0
            )
            
            if llm_passed and code_passed:
                self.logger.info("[InterfaceReviewer] [OK] Global review PASSED")
                break
            
            # Step 4: Apply fixes
            recommended_fixes = llm_review.get("recommended_fixes", [])
            if recommended_fixes:
                fix_stats = self._apply_fixes(
                    fixes=recommended_fixes,
                    interfaces_data=interfaces_data,
                    enhanced_data_flow=enhanced_data_flow,
                    global_registry=global_registry,
                    dependency_collector=dependency_collector,
                    skeleton_features=skeleton_features or set(),
                    rpg_features=rpg_features or set(),
                )
                review_result["fix_stats"] = fix_stats
                unapplied_n = len(fix_stats["unapplied"])
                tail = f" ({unapplied_n} unapplied)" if unapplied_n else ""
                self.logger.info(
                    f"[InterfaceReviewer] Applied {fix_stats['applied_edges']} edge(s) "
                    f"from {fix_stats['applied_fixes']}/{fix_stats['requested_fixes']} fix request(s)"
                    + tail
                )
            else:
                self.logger.info("[InterfaceReviewer] No fixes recommended, stopping iteration")
                break

        # ------------------------------------------------------------------
        # Final re-check: fixes applied in the LAST iteration were never
        # re-evaluated inside the loop, so connectivity / feature_orphans
        # stored in review_history may be stale. Re-run the structural
        # checks once more against the (now patched) enhanced_data_flow
        # and use those numbers for the final verdict.
        # ------------------------------------------------------------------
        final_entry_points = (
            review_history[-1]["entry_points"] if review_history else []
        )
        final_orphan_units: List[Any] = []
        final_feature_orphans: List[Any] = []
        if review_history:
            final_connectivity = check_call_graph_connectivity(
                interfaces_data, enhanced_data_flow, final_entry_points,
                is_callable=self.backend.is_callable_unit,
                is_test_file=self.backend.is_test_file,
            )
            final_feature_orphans = check_feature_dependency_coverage(
                interfaces_data, enhanced_data_flow, final_entry_points,
                is_callable=self.backend.is_callable_unit,
                is_test_file=self.backend.is_test_file,
            )
            final_orphan_units = final_connectivity["orphan_units"]
            self.logger.info(
                f"[InterfaceReviewer] Final re-check: "
                f"{len(final_orphan_units)} orphan unit(s), "
                f"{len(final_feature_orphans)} orphan feature(s)"
            )

        # Collect unapplied fixes from every iteration and classify them:
        #   * advisory  — manual follow-up that does NOT gate the verdict
        #     (``modify_interface`` and unsupported non-Python ``add_interface``).
        #   * blocking  — an automatic repair that the pipeline attempted but
        #     could not install, such as unresolved ``add_dependency`` or a
        #     rejected Python ``add_interface``. These gate ``passed``.
        unapplied_fixes: List[Dict[str, Any]] = []
        for entry in review_history:
            stats = entry.get("fix_stats") or {}
            for u in stats.get("unapplied", []):
                unapplied_fixes.append({**u, "iteration": entry.get("iteration")})

        advisory_fixes = [
            u for u in unapplied_fixes if _is_advisory_unapplied_fix(u)
        ]
        blocking_unapplied_fixes = [
            u for u in unapplied_fixes if not _is_advisory_unapplied_fix(u)
        ]

        last_llm_pass = (
            review_history[-1]["llm_review"].get("pass", False)
            if review_history else False
        )
        code_passed = (
            len(final_orphan_units) == 0
            and len(final_feature_orphans) == 0
            and len(blocking_unapplied_fixes) == 0
        )

        final_result = {
            "review_history": review_history,
            "final_entry_points": final_entry_points,
            "final_feature_orphans": final_feature_orphans,
            "final_orphan_units": final_orphan_units,
            "unapplied_fixes": unapplied_fixes,
            "advisory_fixes": advisory_fixes,
            "blocking_unapplied_fixes": blocking_unapplied_fixes,
            "iterations_run": len(review_history),
            # ``last_llm_pass`` is a snapshot taken BEFORE the LLM's own
            # iteration-N fixes are applied, so it can read FAIL even when
            # those fixes resolved every issue. The structural ``code_passed``
            # check runs against the post-fix graph and is authoritative; we
            # surface ``last_llm_pass`` separately for visibility but do not
            # gate the overall verdict on it.
            "last_llm_pass": last_llm_pass,
            "passed": bool(code_passed),
        }

        return final_result
    
    def _run_llm_review(
        self,
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        import_warnings: List[Dict[str, str]],
        data_flow_edges: List[Dict[str, Any]],
        iteration: int = 0,
        previous_reviews: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the LLM global review.
        
        Builds a comprehensive prompt with all designed interfaces and asks
        LLM to identify entry points, orphan modules, missing wiring, etc.
        """
        # Build the interface summary for the prompt
        interface_summary = self._build_interface_summary(interfaces_data, global_registry)
        
        # Build data flow summary
        data_flow_summary = self._build_data_flow_summary(data_flow_edges)
        
        # Build import warnings summary
        import_warnings_summary = self._build_import_warnings_summary(import_warnings)
        
        # Build dependency summary
        dep_summary = self._build_dependency_summary(enhanced_data_flow)
        
        # Build previous review context (for iteration > 0)
        prev_context = ""
        if previous_reviews:
            last_review = previous_reviews[-1]
            prev_llm = last_review.get("llm_review", {})
            prev_orphan_units = last_review.get("orphan_units", [])
            prev_orphan_count = len(last_review.get("feature_orphans", []))
            prev_fix_stats = last_review.get("fix_stats") or {}

            # Detailed fix accounting so the LLM doesn't blindly repeat
            # requests we already failed to auto-apply last time.
            applied_edges = prev_fix_stats.get("applied_edges", 0)
            applied_fixes = prev_fix_stats.get("applied_fixes", 0)
            requested_fixes = prev_fix_stats.get(
                "requested_fixes", len(prev_llm.get("recommended_fixes", []))
            )
            unapplied = prev_fix_stats.get("unapplied", []) or []

            applied_line = (
                f"- Successfully applied: {applied_edges} dependency edge(s) "
                f"from {applied_fixes}/{requested_fixes} fix request(s)"
            )
            if unapplied:
                # Cap to top 5 to keep prompt small; truncate per-entry desc.
                lines = ["- Could NOT auto-apply (do NOT request these again — switch to a different action or omit):"]
                for u in unapplied[:5]:
                    desc = (u.get("description") or "").strip().replace("\n", " ")
                    if len(desc) > 120:
                        desc = desc[:117] + "..."
                    lines.append(
                        f"  * [{u.get('action','?')}] "
                        f"{u.get('file_path','?')}::{u.get('unit_name','?')} "
                        f"— {u.get('reason','?')}"
                        + (f"  ({desc})" if desc else "")
                    )
                if len(unapplied) > 5:
                    lines.append(f"  * ... and {len(unapplied) - 5} more")
                unapplied_block = "\n".join(lines)
            else:
                unapplied_block = "- Could NOT auto-apply: (none)"

            prev_context = f"""
## Previous Review Results (iteration {last_review.get('iteration', '?')})
- Entry points identified: {len(prev_llm.get('entry_points', []))}
- Orphan modules from LLM: {len(prev_llm.get('orphan_modules', []))}
- Orphan units (no incoming edges): {len(prev_orphan_units)}
- Orphan features: {prev_orphan_count}
{applied_line}
{unapplied_block}

Please review the CURRENT state after the applied fixes and provide updated
analysis. For any issue listed above as "could NOT auto-apply", do NOT
repeat the same request — either pick a different action that the handler
can apply (e.g. switch `modify_interface` to `add_dependency` /
`add_interface`), supply the missing fields (e.g. `signature`,
`docstring`, `feature_path` for `add_interface`), or omit the request and
accept that aspect of the design as-is.
"""
        
        user_prompt = f"""
## All Designed Interfaces (grouped by subtree)
{interface_summary}

## Data Flow DAG
{data_flow_summary}

## Current Dependency Edges
{dep_summary}

## Import Cross-Validation Warnings
{import_warnings_summary}
{prev_context}

Please perform the review tasks and return the JSON result.
""".strip()
        
        # Non-Python projects cannot use the add_interface auto-handler
        # (stub synthesis is Python-only). Steer the LLM toward the
        # language-agnostic actions so it does not waste a fix slot on a
        # request that will be skipped.
        language_note = ""
        if self.backend.name != "python":
            hints = self.backend.prompt_hints()
            language_note = (
                f"\n\n## Target Language: {hints.display_name}\n"
                f"This is a {hints.display_name} project, NOT Python. When "
                "recommending fixes, use `add_dependency` (wire an existing "
                "callee) or `modify_interface` (describe a manual change). "
                "Do NOT use `add_interface`: automatic interface-stub "
                "synthesis is only available for Python and will be skipped."
            )

        combined_prompt = (
            with_language_directive(GLOBAL_INTERFACE_REVIEW_PROMPT, self.backend)
            + f"{language_note}\n\n{user_prompt}"
        )
        
        try:
            response = self.llm.generate(
                combined_prompt,
                purpose=f"global_interface_review_{iteration + 1}"
            )
            
            # Parse JSON from response
            result = self.llm.parse_json_block(response)
            
            if result:
                return result
            
            # Try to extract JSON directly
            try:
                # Find JSON in the response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(response[start:end])
                    return result
            except json.JSONDecodeError:
                pass
            
            self.logger.warning("[InterfaceReviewer] Failed to parse LLM review response")
            return None
            
        except Exception as e:
            self.logger.error(f"[InterfaceReviewer] LLM review failed: {e}")
            return None
    
    def _apply_fixes(
        self,
        fixes: List[Dict[str, Any]],
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        dependency_collector: Optional[DependencyCollector] = None,
        skeleton_features: Optional[Set[str]] = None,
        rpg_features: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Apply recommended fixes from the LLM review.

        Supported actions:
        - add_dependency:    Add a call dependency edge (auto-applied)
        - add_interface:     Materialise a new Python unit (auto-applied via
                             ``_apply_add_interface``; requires
                             ``skeleton_features`` + ``rpg_features``).
                             Non-Python requests are recorded as advisory
                             manual follow-up.
        - modify_interface:  Logged + recorded as advisory manual follow-up

        Returns:
            Stats dict with keys ``requested_fixes`` (top-level fix count),
            ``applied_fixes`` (fixes that produced >=1 edge or unit),
            ``applied_edges`` (total dependency edges added), and
            ``unapplied`` (list of fixes that were not auto-applied).
        """
        applied_edges = 0
        applied_fixes = 0
        unapplied: List[Dict[str, Any]] = []

        for fix in fixes:
            action = fix.get("action", "")
            file_path = fix.get("file_path", "")
            unit_name = fix.get("unit_name", "")
            description = fix.get("description", "")
            
            if action == "add_dependency":
                calls_to_add = fix.get("calls_to_add", [])
                edges_added = 0
                edges_already_existed = 0
                edges_unresolved = 0
                for call_info in calls_to_add:
                    callee = call_info.get("callee", "")
                    callee_file = call_info.get("callee_file", "")
                    
                    if not callee:
                        continue
                    
                    # Resolve callee_file from global registry if not provided
                    if not callee_file:
                        callee_file = global_registry.resolve_callee(callee)
                    
                    if not callee_file:
                        self.logger.warning(
                            f"[InterfaceReviewer] Cannot resolve callee '{callee}' "
                            f"for fix on {file_path}::{unit_name}"
                        )
                        edges_unresolved += 1
                        continue
                    
                    # Add to enhanced_data_flow
                    inv_edges = enhanced_data_flow.get("invocation_edges", [])
                    
                    # Check if edge already exists
                    # (must match all four identity fields: same
                    # callee_name can resolve to different callee_file
                    # when two unrelated units share a name)
                    exists = any(
                        e.get("caller") == unit_name
                        and e.get("callee") == callee
                        and e.get("caller_file") == file_path
                        and e.get("callee_file") == callee_file
                        for e in inv_edges
                    )
                    
                    if exists:
                        edges_already_existed += 1
                        continue

                    new_edge = {
                        "caller": unit_name,
                        "callee": callee,
                        "caller_file": file_path,
                        "callee_file": callee_file,
                        "edge_type": "invokes",
                        "generator": "global_review",
                    }
                    inv_edges.append(new_edge)
                    enhanced_data_flow["invocation_edges"] = inv_edges

                    # Also add to dependency_collector if available
                    if dependency_collector:
                        dependency_collector.add_invocation(
                            caller=unit_name,
                            callee=callee,
                            caller_file=file_path,
                            callee_file=callee_file,
                        )

                    self.logger.info(
                        f"[InterfaceReviewer] Added dependency: "
                        f"{unit_name} ({file_path}) -> {callee} ({callee_file})"
                    )
                    applied_edges += 1
                    edges_added += 1
                # A fix counts as applied if at least one edge was added OR
                # all its edges already existed (LLM repeated a prior fix).
                # Only mark as unapplied when nothing happened AND there were
                # unresolved callees that actually need attention.
                if edges_added > 0 or (edges_already_existed > 0 and edges_unresolved == 0):
                    applied_fixes += 1
                elif edges_unresolved > 0:
                    unapplied.append({
                        "action": action,
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "description": description[:200],
                        "reason": f"{edges_unresolved} callee(s) could not be resolved",
                    })
                # else: empty calls_to_add — silently ignore (LLM bug, not actionable)

            elif action == "add_interface":
                # Interface-stub synthesis is Python-only: it emits a
                # ``def/class`` body with a docstring + ``pass``. For other
                # languages we cannot materialise a syntactically valid stub,
                # so surface the request as advisory manual follow-up instead
                # of silently dropping it or blocking structural convergence.
                if self.backend.name != "python":
                    self.logger.info(
                        "[InterfaceReviewer] Recording add_interface as "
                        "advisory for %s project (stub synthesis is "
                        "Python-only): %s::%s",
                        self.backend.name, file_path, unit_name,
                    )
                    unapplied.append({
                        "action": action,
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "description": description[:200],
                        "reason": (
                            "add_interface auto-handler is Python-only; "
                            f"manual follow-up required for {self.backend.name}"
                        ),
                        "advisory": True,
                        "manual_follow_up": True,
                        "unsupported_for_language": self.backend.name,
                    })
                    continue
                ok, reason, edges_added = self._apply_add_interface(
                    fix=fix,
                    interfaces_data=interfaces_data,
                    enhanced_data_flow=enhanced_data_flow,
                    global_registry=global_registry,
                    dependency_collector=dependency_collector,
                    skeleton_features=skeleton_features or set(),
                    rpg_features=rpg_features or set(),
                )
                if ok:
                    applied_fixes += 1
                    applied_edges += edges_added
                else:
                    self.logger.warning(
                        f"[InterfaceReviewer] add_interface fix rejected: "
                        f"{reason} (file: {file_path}, unit: {unit_name})"
                    )
                    unapplied.append({
                        "action": action,
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "description": description[:200],
                        "reason": reason,
                    })

            elif action == "modify_interface":
                self.logger.warning(
                    f"[InterfaceReviewer] modify_interface fix requested but not auto-applied: "
                    f"{description} (file: {file_path}, unit: {unit_name})"
                )
                unapplied.append({
                    "action": action,
                    "file_path": file_path,
                    "unit_name": unit_name,
                    "description": description[:200],
                    "reason": "modify_interface has no auto-handler",
                })

            else:
                self.logger.warning(
                    f"[InterfaceReviewer] Unknown fix action: {action}"
                )
                unapplied.append({
                    "action": action,
                    "file_path": file_path,
                    "unit_name": unit_name,
                    "description": description[:200],
                    "reason": f"unknown action '{action}'",
                })

        return {
            "requested_fixes": len(fixes),
            "applied_fixes": applied_fixes,
            "applied_edges": applied_edges,
            "unapplied": unapplied,
        }

    def _apply_add_interface(
        self,
        fix: Dict[str, Any],
        interfaces_data: Dict[str, Any],
        enhanced_data_flow: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
        dependency_collector: Optional[DependencyCollector],
        skeleton_features: Set[str],
        rpg_features: Set[str],
    ) -> Tuple[bool, str, int]:
        """Materialise an LLM-requested new interface unit into interfaces_data.

        Required fix fields:
                        - ``file_path``: existing key under some subtree, or a new file
                            whose feature root names an existing subtree
            - ``unit_name``: prefixed with ``"function "`` or ``"class "``
            - ``signature``: full Python signature (e.g. ``def foo() -> None:``)
            - ``docstring``: non-empty, 1-3 sentences
            - ``feature_path``: must be present in BOTH ``skeleton_features``
              and ``rpg_features`` (the latter ensures ``update_rpg`` will
              find a target node for the ``meta.path`` write-back)

        Optional:
            - ``incoming_calls_from``: list of caller unit names; for each
              caller resolvable via ``global_registry.resolve_callee``, an
              invocation edge is added. Unresolvable callers are logged but
              do NOT block the unit from being added.

        Idempotent on ``(file_path, unit_name)``: a second invocation
        returns ``(False, "unit already exists (idempotent skip)", 0)``.

        Returns:
            ``(applied, reason, edges_added)``. ``reason`` is empty on
            success; ``edges_added`` is the number of incoming invocation
            edges materialised (0 if no ``incoming_calls_from``).
        """
        file_path = str(fix.get("file_path", "")).strip()
        unit_name = str(fix.get("unit_name", "")).strip()
        signature = str(fix.get("signature", "")).strip()
        docstring = str(fix.get("docstring", "")).strip()
        feature_path = str(fix.get("feature_path", "")).strip()
        incoming = fix.get("incoming_calls_from") or []
        if not isinstance(incoming, list):
            incoming = []

        # --- Required-field validation ------------------------------------
        missing = [
            name for name, val in (
                ("file_path", file_path),
                ("unit_name", unit_name),
                ("signature", signature),
                ("docstring", docstring),
                ("feature_path", feature_path),
            ) if not val
        ]
        if missing:
            return False, f"missing required field(s): {missing}", 0

        if not (unit_name.startswith("function ") or unit_name.startswith("class ")):
            return (
                False,
                "unit_name must start with 'function ' or 'class '",
                0,
            )

        if not _SIGNATURE_RE.match(signature) or "\n" in signature:
            return False, f"signature must be a single-line Python def/class header ending with ':', got: {signature!r}", 0

        if skeleton_features and feature_path not in skeleton_features:
            return (
                False,
                f"feature_path '{feature_path}' is not present in skeleton.json",
                0,
            )
        if rpg_features and feature_path not in rpg_features:
            return (
                False,
                f"feature_path '{feature_path}' is not present in repo_rpg.json (may have been pruned in a prior run)",
                0,
            )

        # --- Locate target subtree/file -----------------------------------
        subtrees = interfaces_data.get("subtrees") or interfaces_data.get("components") or {}
        target_subtree: Optional[str] = None
        file_entry: Optional[Dict[str, Any]] = None
        file_container: Optional[Dict[str, Any]] = None
        for st_name, st_data in subtrees.items():
            container = st_data.get("interfaces")
            if container is None:
                container = st_data.get("files")
            if container is not None and file_path in container:
                target_subtree = st_name
                file_container = container
                file_entry = container[file_path]
                break
        if file_entry is None:
            feature_root = feature_path.split("/", 1)[0].strip()
            target_data = subtrees.get(feature_root) if feature_root else None
            if target_data is None:
                return (
                    False,
                    f"file_path '{file_path}' not found in any subtree's interfaces",
                    0,
                )

            target_subtree = feature_root
            file_container = target_data.get("interfaces")
            if file_container is None:
                file_container = target_data.get("files")
            if file_container is None:
                file_container = {}
                target_data["interfaces"] = file_container

            file_entry = {
                "units": [],
                "units_to_features": {},
                "units_to_code": {},
                "file_code": "",
            }
            file_container[file_path] = file_entry
            files_order = target_data.setdefault("files_order", [])
            if isinstance(files_order, list) and file_path not in files_order:
                files_order.append(file_path)

        # --- Idempotency check --------------------------------------------
        existing_units = file_entry.setdefault("units", [])
        if unit_name in existing_units:
            return False, "unit already exists (idempotent skip)", 0

        # --- Build stub body (Decision A: signature + docstring + pass) ---
        # Escape any triple-quotes in docstring to keep the stub parseable.
        safe_doc = docstring.replace('"""', '\\"\\"\\"')
        stub = f'{signature}\n    """{safe_doc}"""\n    pass\n'

        new_file_code = _insert_unit_into_file_code(
            file_entry.get("file_code", ""), stub
        )

        # --- Mutate file entry in-place -----------------------------------
        existing_units.append(unit_name)
        file_entry.setdefault("units_to_features", {})[unit_name] = [feature_path]
        file_entry.setdefault("units_to_code", {})[unit_name] = stub
        file_entry["file_code"] = new_file_code
        # Tag so InterfacesStore.find_orphan_units can exclude this unit
        # from prune candidates: it was deliberately added during review and
        # not having incoming edges yet doesn't mean it's dead code.
        file_entry.setdefault("_handler_added", []).append(unit_name)

        # --- Register so subsequent iterations / global review see it -----
        global_registry.register_unit(
            file_path=file_path,
            unit_name=unit_name,
            subtree_name=target_subtree,
            features=[feature_path],
            signature_summary=signature,
        )

        # --- Optional incoming invocation edges ---------------------------
        edges_added = 0
        for caller_name in incoming:
            caller_name = str(caller_name).strip()
            if not caller_name:
                continue
            caller_file = global_registry.resolve_callee(caller_name)
            if not caller_file:
                self.logger.warning(
                    f"[InterfaceReviewer] add_interface: incoming caller "
                    f"'{caller_name}' could not be resolved; skipping that edge"
                )
                continue

            # Normalise caller_name to the "function "/"class " prefixed form
            # the rest of the pipeline uses.
            normalised_caller = caller_name
            if not (caller_name.startswith("function ") or caller_name.startswith("class ")):
                # Best-effort lookup: registry stores prefixed unit_name as key.
                for k in global_registry.units:
                    bare = k.split(" ", 1)[1] if " " in k else k
                    if bare == caller_name:
                        normalised_caller = k
                        break

            # Strip the prefix for the edge "caller" field, matching the
            # convention used elsewhere in enhanced_data_flow.
            caller_bare = normalised_caller.split(" ", 1)[-1]
            callee_bare = unit_name.split(" ", 1)[-1]

            inv_edges = enhanced_data_flow.setdefault("invocation_edges", [])
            already = any(
                e.get("caller") == caller_bare
                and e.get("callee") == callee_bare
                and e.get("caller_file") == caller_file
                and e.get("callee_file") == file_path
                for e in inv_edges
            )
            if already:
                continue

            inv_edges.append({
                "caller": caller_bare,
                "callee": callee_bare,
                "caller_file": caller_file,
                "callee_file": file_path,
                "edge_type": "invokes",
                "generator": "global_review_add_interface",
            })
            if dependency_collector is not None:
                dependency_collector.add_invocation(
                    caller=caller_bare,
                    callee=callee_bare,
                    caller_file=caller_file,
                    callee_file=file_path,
                )
            edges_added += 1
            self.logger.info(
                f"[InterfaceReviewer] add_interface: incoming edge "
                f"{caller_bare} ({caller_file}) -> {callee_bare} ({file_path})"
            )

        self.logger.info(
            f"[InterfaceReviewer] Added interface {unit_name} in {file_path} "
            f"(subtree={target_subtree}, feature={feature_path}, +{edges_added} edge(s))"
        )
        return True, "", edges_added

    def _build_interface_summary(
        self,
        interfaces_data: Dict[str, Any],
        global_registry: GlobalInterfaceRegistry,
    ) -> str:
        """Build a comprehensive interface summary for the LLM review prompt."""
        parts = []
        subtrees = interfaces_data.get("subtrees", {})
        subtree_order = interfaces_data.get("subtree_order", [])
        
        for subtree_name in subtree_order:
            subtree_data = subtrees.get(subtree_name, {})
            file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
            
            if not file_interfaces:
                continue
            
            parts.append(f"\n### Subtree: {subtree_name}")
            
            for file_path, file_data in file_interfaces.items():
                units = file_data.get("units", [])
                units_to_features = file_data.get("units_to_features", {})
                file_code = file_data.get("file_code", "")
                
                if not units:
                    continue
                
                parts.append(f"\n**{file_path}**")
                
                for unit_name in units:
                    features = units_to_features.get(unit_name, [])
                    features_str = ", ".join(features[:5])
                    if len(features) > 5:
                        features_str += f" (+{len(features) - 5} more)"
                    parts.append(f"  - `{unit_name}` → features: {features_str}")
                
                # Include abbreviated code (first 20 lines)
                if file_code:
                    code_lines = file_code.split("\n")
                    if len(code_lines) > 25:
                        code_preview = "\n".join(code_lines[:25]) + "\n    # ... (truncated)"
                    else:
                        code_preview = file_code
                    parts.append(f"  ```{self.backend.markdown_fence}\n{code_preview}\n  ```")
        
        return "\n".join(parts) if parts else "No interfaces designed."
    
    def _build_data_flow_summary(self, data_flow_edges: List[Dict[str, Any]]) -> str:
        """Build a data flow summary for the prompt."""
        if not data_flow_edges:
            return "No data flow edges."
        
        parts = []
        for edge in data_flow_edges:
            source = edge.get("source", "?")
            target = edge.get("target", "?")
            desc = edge.get("description", "")
            parts.append(f"  {source} → {target}" + (f": {desc}" if desc else ""))
        
        return "\n".join(parts)
    
    def _build_import_warnings_summary(self, warnings: List[Dict[str, str]]) -> str:
        """Build import warnings summary for the prompt."""
        if not warnings:
            return "No import cross-validation warnings."
        
        parts = [f"Found {len(warnings)} potential issues:"]
        for w in warnings[:20]:  # Limit to 20
            parts.append(f"  - {w.get('message', '?')}")
        
        if len(warnings) > 20:
            parts.append(f"  ... and {len(warnings) - 20} more warnings")
        
        return "\n".join(parts)
    
    def _build_dependency_summary(self, enhanced_data_flow: Dict[str, Any]) -> str:
        """Build a dependency edge summary for the prompt."""
        if not enhanced_data_flow:
            return "No dependency edges collected."
        
        parts = []
        
        inv_edges = enhanced_data_flow.get("invocation_edges", [])
        inh_edges = enhanced_data_flow.get("inheritance_edges", [])
        ref_edges = enhanced_data_flow.get("reference_edges", [])
        
        parts.append(
            f"Total: {len(inv_edges)} invocation, {len(inh_edges)} inheritance, "
            f"{len(ref_edges)} reference edges"
        )
        
        # Cross-file invocations
        cross_file = [e for e in inv_edges if e.get("caller_file") != e.get("callee_file")]
        same_file = [e for e in inv_edges if e.get("caller_file") == e.get("callee_file")]
        no_callee = [e for e in inv_edges if not e.get("callee_file")]
        
        parts.append(
            f"Invocations: {len(cross_file)} cross-file, {len(same_file)} same-file, "
            f"{len(no_callee)} unresolved callee"
        )
        
        # Show cross-file edges
        if cross_file:
            parts.append("\nCross-file invocations:")
            for e in cross_file[:30]:
                parts.append(
                    f"  {e.get('caller', '?')} ({e.get('caller_file', '?')}) "
                    f"→ {e.get('callee', '?')} ({e.get('callee_file', '?')})"
                )
            if len(cross_file) > 30:
                parts.append(f"  ... and {len(cross_file) - 30} more")
        
        return "\n".join(parts)


# ============================================================================
# Orphan Pruning
# ============================================================================

def prune_orphan_interfaces(
    interfaces_data: Dict[str, Any],
    review_result: Dict[str, Any],
    enhanced_data_flow: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Outdated legacy helper for removing orphan interfaces from raw dicts.

    This function is retained for backward compatibility with older callers.
    The active design_interfaces flow uses ``InterfacesStore.find_orphan_units()``
    followed by ``InterfacesStore.prune_units()`` after LLM orphan review. Do
    not extend this helper for new pruning behavior.

    An interface unit is considered a **true orphan** when it has **no incoming
    edges AND no outgoing edges** in the call graph and is not an entry point.
    Units that participate in any edge (caller or callee) are preserved even if
    they are not reachable from entry points — their connected components are
    valid code that just lacks proper wiring to the top-level entry flow.

    For each pruned unit the function:
    - Removes it from ``units``, ``units_to_features``, ``units_to_code``
    - Regenerates ``file_code`` from the remaining units
    - If all units in a file are removed, removes the entire file entry
    - Removes related edges from ``enhanced_data_flow``

    Returns a summary dict::

        {
            "pruned_units": [...],
            "pruned_files": [...],
            "orphan_feature_paths": set of feature paths whose ALL implementing
                                    units were pruned,
            "surviving_feature_paths": set of feature paths that still have at
                                       least one surviving unit,
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ---- 0. Build call graph to find truly isolated units ----
    entry_points = review_result.get("final_entry_points", [])
    entry_point_keys: Set[str] = set()
    for ep in entry_points:
        ep_file = ep.get("file_path", "")
        ep_unit = ep.get("unit_name", "")
        if ep_file and ep_unit:
            entry_point_keys.add(f"{ep_file}::{ep_unit}")

    outgoing, incoming, unit_to_file = build_call_graph(
        interfaces_data, enhanced_data_flow
    )
    all_units = set(unit_to_file.keys())

    # Truly isolated: no incoming AND no outgoing AND not entry point
    isolated_keys: Set[str] = set()
    for u in all_units:
        if u in entry_point_keys:
            continue
        has_in = u in incoming and len(incoming[u]) > 0
        has_out = u in outgoing and len(outgoing[u]) > 0
        if not has_in and not has_out:
            isolated_keys.add(u)

    if not isolated_keys:
        logger.info("[prune_orphan_interfaces] No truly isolated units — nothing to prune")
        # Still compute surviving features for RPG pruning
        surviving = _collect_surviving_features(interfaces_data)
        return {
            "pruned_units": [],
            "pruned_files": [],
            "orphan_feature_paths": set(),
            "surviving_feature_paths": surviving,
        }

    logger.info(
        f"[prune_orphan_interfaces] {len(isolated_keys)} truly isolated units "
        f"(out of {len(all_units)} total) to prune"
    )

    pruned_units: List[Dict[str, Any]] = []
    pruned_files: List[Dict[str, str]] = []

    # ---- 1. Build a global map: feature_path → set of unit_keys that implement it ----
    feature_to_all_unit_keys: Dict[str, Set[str]] = defaultdict(set)
    subtrees = interfaces_data.get("subtrees", {})
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        for file_path, file_data in file_interfaces.items():
            for unit_name, features in file_data.get("units_to_features", {}).items():
                unit_key = f"{file_path}::{unit_name}"
                for fp in features:
                    feature_to_all_unit_keys[fp].add(unit_key)

    # ---- 2. Prune units from interfaces_data ----
    for subtree_name, subtree_data in subtrees.items():
        file_interfaces = subtree_data.get("interfaces", subtree_data.get("files", {}))
        files_to_remove: List[str] = []

        for file_path in list(file_interfaces.keys()):
            file_data = file_interfaces[file_path]
            units: List[str] = file_data.get("units", [])
            units_to_features: Dict[str, List[str]] = file_data.get("units_to_features", {})
            units_to_code: Dict[str, str] = file_data.get("units_to_code", {})

            units_to_remove: List[str] = []
            for unit_name in units:
                unit_key = f"{file_path}::{unit_name}"
                if unit_key in isolated_keys:
                    units_to_remove.append(unit_name)
                    pruned_units.append({
                        "file_path": file_path,
                        "unit_name": unit_name,
                        "subtree": subtree_name,
                        "features": units_to_features.get(unit_name, []),
                    })

            if not units_to_remove:
                continue

            # Remove the units
            for uname in units_to_remove:
                if uname in units:
                    units.remove(uname)
                units_to_features.pop(uname, None)
                units_to_code.pop(uname, None)
                logger.info(f"[prune_orphan_interfaces] Pruned unit: {file_path}::{uname}")

            file_data["units"] = units
            file_data["units_to_features"] = units_to_features
            file_data["units_to_code"] = units_to_code

            if not units:
                # All units pruned → remove the entire file entry
                files_to_remove.append(file_path)
            else:
                # Regenerate file_code from surviving units
                code_parts = []
                for uname in units:
                    code = units_to_code.get(uname, "")
                    if code:
                        code_parts.append(code)
                file_data["file_code"] = "\n\n".join(code_parts)

        for fp in files_to_remove:
            del file_interfaces[fp]
            pruned_files.append({"file_path": fp, "subtree": subtree_name})
            logger.info(f"[prune_orphan_interfaces] Pruned entire file: {fp} (all units removed)")

    # ---- 3. Remove edges for pruned units from enhanced_data_flow ----
    pruned_unit_names = {pu["unit_name"] for pu in pruned_units}
    pruned_file_paths = {pu["file_path"] for pu in pruned_units}

    def _edge_involves_pruned_unit(edge: Dict[str, Any]) -> bool:
        """Return True if the edge references a pruned unit."""
        for role_name, role_file in [
            ("caller", "caller_file"), ("callee", "callee_file"),
            ("child", "child_file"), ("parent", "parent_file"),
            ("unit", "unit_file"),
        ]:
            name_val = edge.get(role_name, "")
            file_val = edge.get(role_file, "")
            if name_val in pruned_unit_names:
                # Double-check file to avoid false positives on common names
                if not file_val or file_val in pruned_file_paths:
                    return True
        return False

    for edge_list_key in ("invocation_edges", "inheritance_edges", "reference_edges"):
        edges = enhanced_data_flow.get(edge_list_key, [])
        before = len(edges)
        edges[:] = [e for e in edges if not _edge_involves_pruned_unit(e)]
        after = len(edges)
        if before != after:
            logger.info(
                f"[prune_orphan_interfaces] Removed {before - after} edges from {edge_list_key}"
            )

    # ---- 4. Identify features that are now fully orphaned ----
    pruned_key_set = {f"{pu['file_path']}::{pu['unit_name']}" for pu in pruned_units}
    orphan_feature_paths: Set[str] = set()
    for feature_path, all_keys in feature_to_all_unit_keys.items():
        if all_keys and all_keys.issubset(pruned_key_set):
            orphan_feature_paths.add(feature_path)

    if orphan_feature_paths:
        logger.info(
            f"[prune_orphan_interfaces] {len(orphan_feature_paths)} features fully orphaned: "
            + ", ".join(sorted(orphan_feature_paths)[:10])
        )

    # ---- 5. Collect surviving feature paths for RPG pruning ----
    surviving = _collect_surviving_features(interfaces_data)

    return {
        "pruned_units": pruned_units,
        "pruned_files": pruned_files,
        "orphan_feature_paths": orphan_feature_paths,
        "surviving_feature_paths": surviving,
    }


def _collect_surviving_features(interfaces_data: Dict[str, Any]) -> Set[str]:
    """Collect all feature paths that still have at least one interface unit."""
    surviving: Set[str] = set()
    for st_data in interfaces_data.get("subtrees", {}).values():
        file_interfaces = st_data.get("interfaces", st_data.get("files", {}))
        for file_data in file_interfaces.values():
            for features in file_data.get("units_to_features", {}).values():
                surviving.update(features)
    return surviving


def print_review_summary(review_result: Dict[str, Any]):
    """Print a human-readable summary of the global review results."""
    print("\n" + "=" * 60)
    print("GLOBAL INTERFACE REVIEW SUMMARY")
    print("=" * 60)

    iterations = review_result.get("iterations_run", 0)
    passed = review_result.get("passed", False)

    print(f"Iterations: {iterations}")
    print(f"Final Status: {'[OK] PASSED' if passed else '[FAIL] NEEDS ATTENTION'}")

    # Entry points
    entry_points = review_result.get("final_entry_points", [])
    if entry_points:
        print(f"\nEntry Points ({len(entry_points)}):")
        for ep in entry_points:
            print(f"  -  {ep.get('unit_name', '?')} in {ep.get('file_path', '?')}")
            if ep.get("rationale"):
                print(f"    Reason: {ep['rationale']}")
    
    # Feature orphans
    feature_orphans = review_result.get("final_feature_orphans", [])
    if feature_orphans:
        print(f"\nOrphan Features ({len(feature_orphans)}):")
        for fo in feature_orphans[:10]:
            print(
                f"  -  {fo.get('unit_name', '?')} in {fo.get('file_path', '?')} "
                f"({fo.get('subtree', '?')})"
            )
        if len(feature_orphans) > 10:
            print(f"  ... and {len(feature_orphans) - 10} more")

    print("=" * 60)


# ============================================================================
# Orphan Unit Review
# ============================================================================


@dataclass
class OrphanReviewResult:
    """Result of orphan unit review."""
    decisions: Dict[str, str] = field(default_factory=dict)  # unit_key -> "retain" | "prune"
    completed_edges: Dict[str, Dict[str, List[Dict]]] = field(default_factory=dict)  # unit_key -> edges dict

    @property
    def keys_to_prune(self) -> List[str]:
        return [k for k, d in self.decisions.items() if d == "prune"]

    @property
    def keys_to_retain(self) -> List[str]:
        return [k for k, d in self.decisions.items() if d == "retain"]

    def get_all_edges(self) -> Dict[str, List[Dict]]:
        """Aggregate all completed edges by type."""
        result: Dict[str, List[Dict]] = {
            "inheritance_edges": [],
            "invocation_edges": [],
            "reference_edges": [],
        }
        for edges_dict in self.completed_edges.values():
            for edge_type, edges in edges_dict.items():
                if edge_type in result and edges:
                    result[edge_type].extend(edges)
        return result


def review_orphan_units(
    orphan_details: List[Dict[str, Any]],
    repo_info: str,
    subtree_interfaces: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
    target_language: Optional[str] = None,
) -> OrphanReviewResult:
    """Review orphan units using LLM to determine which should be retained or pruned.

    Units are grouped by subtree for better context during review.

    Args:
        orphan_details: List of orphan unit details from InterfacesStore.get_orphan_unit_details()
        repo_info: Repository description for context
        subtree_interfaces: Optional dict mapping subtree -> interfaces data for context
        llm_client: LLM client to use (creates new one if not provided)
        target_language: Optional target language used for prompt code fences.
            Defaults to Python for backward compatibility.

    Returns:
        OrphanReviewResult with decisions and completed edges
    """
    if not orphan_details:
        logger.info("[review_orphan_units] No orphan units to review")
        return OrphanReviewResult()

    llm = llm_client or LLMClient()
    backend = get_backend(target_language or "python")
    result = OrphanReviewResult()

    # Group orphans by subtree
    orphans_by_subtree: Dict[str, List[Dict[str, Any]]] = {}
    for detail in orphan_details:
        subtree = detail.get("subtree", "unknown")
        orphans_by_subtree.setdefault(subtree, []).append(detail)

    # Review each subtree's orphans together
    for subtree, subtree_orphans in orphans_by_subtree.items():
        # Get subtree context if available
        subtree_context = None
        if subtree_interfaces and subtree in subtree_interfaces:
            subtree_context = subtree_interfaces[subtree]

        batch_result = _review_orphan_batch(
            subtree_orphans, repo_info, subtree, subtree_context, llm,
            markdown_fence=backend.markdown_fence,
        )
        result.decisions.update(batch_result.decisions)
        result.completed_edges.update(batch_result.completed_edges)

    logger.info(
        f"[review_orphan_units] Reviewed {len(orphan_details)} orphan units across "
        f"{len(orphans_by_subtree)} subtrees: "
        f"{len(result.keys_to_retain)} retain, "
        f"{len(result.keys_to_prune)} prune, "
        f"{len(result.completed_edges)} with completed edges"
    )

    return result


def _review_orphan_batch(
    batch: List[Dict[str, Any]],
    repo_info: str,
    subtree_name: str,
    subtree_context: Optional[Dict[str, Any]],
    llm: LLMClient,
    markdown_fence: str = "python",
) -> OrphanReviewResult:
    """Review orphan units from a single subtree."""
    # Build user prompt with orphan details
    orphan_summaries = []
    for detail in batch:
        summary = f"""
### Unit: {detail['unit_key']}
- File: {detail['file_path']}
- Features: {', '.join(detail['features']) if detail['features'] else '(none)'}

Code:
```{markdown_fence}
{detail['code']}
```
"""
        orphan_summaries.append(summary)

    user_prompt = f"""## Repository Context
{repo_info}

## Subtree: {subtree_name}

## Orphan Units to Review
The following {len(batch)} interface units in subtree "{subtree_name}" have no incoming or outgoing call edges.
Determine whether each should be retained or pruned.

{''.join(orphan_summaries)}
"""

    if subtree_context:
        # Extract other unit names in the subtree for context
        other_units = []
        interfaces = subtree_context.get("interfaces", {})
        for file_path, file_data in interfaces.items():
            units_to_code = file_data.get("units_to_code", {})
            for unit_name in units_to_code.keys():
                unit_key = f"{file_path}::{unit_name}"
                # Exclude current orphans from context
                if not any(d["unit_key"] == unit_key for d in batch):
                    other_units.append(unit_key)

        if other_units:
            user_prompt += f"""
## Other Units in This Subtree (for context)
{', '.join(other_units[:20])}{'...' if len(other_units) > 20 else ''}
"""

    combined_prompt = f"{ORPHAN_REVIEW_PROMPT}\n\n{user_prompt}"

    result = OrphanReviewResult()

    try:
        response = llm.generate(combined_prompt, purpose="orphan_review")

        # Parse JSON response using LLMClient's built-in method
        parsed = llm.parse_json_block(response)
        if not parsed:
            logger.error("[orphan_review] Failed to parse LLM response as JSON")
            for detail in batch:
                result.decisions[detail["unit_key"]] = "retain"
            return result

        reviews = parsed.get("reviews", [])

        for review in reviews:
            unit_key = review.get("unit_key", "")
            decision = review.get("decision", "retain").lower()
            reason = review.get("reason", "")
            edges = review.get("edges")

            if decision not in ("retain", "prune"):
                decision = "retain"  # Default to retain if unclear

            result.decisions[unit_key] = decision

            # Collect completed edges if provided
            if edges and isinstance(edges, dict):
                valid_edges = {}
                for edge_type in ("inheritance_edges", "invocation_edges", "reference_edges"):
                    if edge_type in edges and edges[edge_type]:
                        valid_edges[edge_type] = edges[edge_type]
                if valid_edges:
                    result.completed_edges[unit_key] = valid_edges
                    logger.info(
                        f"[orphan_review] {unit_key}: {decision} - {reason} "
                        f"(+{sum(len(e) for e in valid_edges.values())} edges)"
                    )
                else:
                    logger.info(f"[orphan_review] {unit_key}: {decision} - {reason}")
            else:
                logger.info(f"[orphan_review] {unit_key}: {decision} - {reason}")

        # Ensure all units in batch have a decision (default to retain)
        for detail in batch:
            if detail["unit_key"] not in result.decisions:
                result.decisions[detail["unit_key"]] = "retain"
                logger.warning(
                    f"[orphan_review] {detail['unit_key']}: defaulting to retain (missing from LLM response)"
                )

        return result

    except Exception as e:
        logger.error(f"[orphan_review] Error during review: {e}")
        # Default all to retain on error
        for detail in batch:
            result.decisions[detail["unit_key"]] = "retain"
        return result
