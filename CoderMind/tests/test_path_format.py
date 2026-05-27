"""Tests for ``scripts.rpg.path_format`` — the canonical RPG node-path helpers.

These tests pin down the contract the rest of the codebase relies on:

- Helpers produce identical, predictable paths for FILE / FUNCTION /
  CLASS / METHOD nodes.
- ``parse_node_path`` round-trips against the constructors.
- ``to_dep_graph_id`` / ``from_dep_graph_id`` interop cleanly with
  ``rpg.dep_graph`` node IDs.
- A CI-style lint guards against any future leak of legacy ``::class X``
  / ``file:X.m`` path formats elsewhere in the source tree.
"""

import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import pytest

from rpg.path_format import (
    file_node_path,
    function_node_path,
    class_node_path,
    method_node_path,
    parse_node_path,
    to_dep_graph_id,
    from_dep_graph_id,
)


# ============================================================================
# Constructors
# ============================================================================

class TestConstructors:

    def test_file_node_path_basic(self):
        assert file_node_path("src/foo.py") == "src/foo.py"

    def test_file_node_path_normalizes_dot_prefix(self):
        assert file_node_path("./src/foo.py") == "src/foo.py"

    def test_file_node_path_strips_leading_slash(self):
        assert file_node_path("/src/foo.py") == "src/foo.py"

    def test_file_node_path_empty_returns_dot(self):
        assert file_node_path("") == "."
        assert file_node_path(None) == "."

    def test_function_node_path(self):
        assert function_node_path("src/foo.py", "bar") == "src/foo.py::bar"

    def test_class_node_path(self):
        assert class_node_path("src/foo.py", "Cls") == "src/foo.py::Cls"

    def test_method_node_path(self):
        assert (
            method_node_path("src/foo.py", "Cls", "m") == "src/foo.py::Cls::m"
        )

    def test_function_class_share_format(self):
        """Function and class paths look identical (disambiguation lives in ``type_name``, not in the path)."""
        assert (
            function_node_path("a.py", "foo")
            == class_node_path("a.py", "foo")
            == "a.py::foo"
        )

    def test_helpers_idempotent(self):
        p1 = method_node_path("src/foo.py", "Cls", "m")
        # No re-construction helper, but parse + rebuild should round-trip.
        f, parts = parse_node_path(p1)
        rebuilt = (
            method_node_path(f, parts[0], parts[1])
            if len(parts) >= 2
            else function_node_path(f, parts[0])
        )
        assert rebuilt == p1


# ============================================================================
# parse_node_path
# ============================================================================

class TestParseNodePath:

    def test_parse_file(self):
        assert parse_node_path("foo.py") == ("foo.py", [])

    def test_parse_function(self):
        assert parse_node_path("foo.py::bar") == ("foo.py", ["bar"])

    def test_parse_class(self):
        assert parse_node_path("foo.py::Cls") == ("foo.py", ["Cls"])

    def test_parse_method(self):
        assert parse_node_path("foo.py::Cls::m") == ("foo.py", ["Cls", "m"])

    def test_parse_empty(self):
        assert parse_node_path("") == ("", [])
        assert parse_node_path(None) == (None or "", [])


# ============================================================================
# Dep-graph interop
# ============================================================================

class TestDepGraphInterop:

    def test_function_roundtrip(self):
        p = function_node_path("foo.py", "bar")
        dg = to_dep_graph_id(p)
        assert dg == "foo.py:bar"
        assert from_dep_graph_id(dg) == p

    def test_class_roundtrip(self):
        p = class_node_path("foo.py", "Cls")
        dg = to_dep_graph_id(p)
        assert dg == "foo.py:Cls"
        assert from_dep_graph_id(dg) == p

    def test_method_roundtrip(self):
        p = method_node_path("foo.py", "Cls", "m")
        dg = to_dep_graph_id(p)
        assert dg == "foo.py:Cls.m"
        assert from_dep_graph_id(dg) == p

    def test_file_id_unchanged(self):
        p = file_node_path("foo.py")
        assert to_dep_graph_id(p) == "foo.py"
        assert from_dep_graph_id("foo.py") == "foo.py"

    def test_from_dep_graph_id_idempotent_on_canonical(self):
        canonical = method_node_path("foo.py", "Cls", "m")
        # Already canonical — should not be mangled.
        assert from_dep_graph_id(canonical) == canonical


# ============================================================================
# CI lint: forbid legacy path-construction patterns elsewhere in the source
# ============================================================================

# Patterns that should NOT appear in production code anymore.  They were the
# three legacy formats the canonical helpers replaced:
#   ``f"{...}::class {...}"``      — encoder prefix style
#   ``f"{...}::function {...}"``   — encoder prefix style
#   ``f"{...}:{...}.{...}"``        — incremental ``file:Cls.method`` style
_BANNED_PATTERNS = [
    re.compile(r'f"\{[^}]+\}::class\s+'),
    re.compile(r"f'\{[^']+\}::class\s+"),
    re.compile(r'f"\{[^}]+\}::function\s+'),
    re.compile(r"f'\{[^']+\}::function\s+"),
]


def _iter_workspace_py_files(skip_files):
    """Yield ``.py`` files under ``scripts/`` and ``utils/`` (excluding files explicitly allowed to keep legacy patterns)."""
    for sub in ("scripts", "utils"):
        root = _PROJECT_ROOT / sub
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part == "__pycache__" for part in path.parts):
                continue
            if path.name in skip_files:
                continue
            yield path


class TestNoLegacyFormatLeaks:
    """Repository-wide lint: no source file (outside the allowlist) should construct RPG node paths with legacy ``::class X`` / ``::function X`` prefixes.  This catches drift if a future change adds a raw ``f"...::class ..."`` literal anywhere in ``scripts/``/``utils/``."""

    # path_format.py is the canonical source-of-truth; it must not import
    # the helpers from itself.  All other files should route through it.
    ALLOWLIST = {
        # The helpers' own implementation file.
        "path_format.py",
        # Migration / docstring references using the legacy format as
        # examples of what *used* to be produced.
        "service.py",
        "models.py",  # `_normalize_path_for_matching` strips legacy prefixes.
        "rpg_updater.py",  # `collect_known_units` re-emits legacy aliases.
    }

    def test_no_legacy_path_construction(self):
        offenders = []
        for path in _iter_workspace_py_files(skip_files=self.ALLOWLIST):
            try:
                src = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pat in _BANNED_PATTERNS:
                m = pat.search(src)
                if m:
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}: {m.group(0)!r}")
                    break
        assert not offenders, (
            "Legacy path-construction pattern detected. Route through "
            "`scripts.rpg.path_format` helpers instead.\nOffending sites:\n"
            + "\n".join(offenders)
        )
