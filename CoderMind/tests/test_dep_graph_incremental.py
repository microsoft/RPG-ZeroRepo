#!/usr/bin/env python3
"""Tests for DependencyGraph incremental update API (Step 2).

Core invariant under test:

    After any sequence of ``add_file`` / ``remove_file`` / ``update_files``
    calls, the resulting DependencyGraph must be **structurally identical**
    to what a fresh ``build() + parse()`` cycle on the same final
    on-disk state would have produced.

Anything weaker risks silent drift between incremental and full updates,
which would mean the pre-commit hook (Step 3) and the codegen path
(Step 4) gradually corrupt the graph in ways nobody notices until a
``/cmind.update_rpg`` full rebuild reveals the discrepancy.

We use small synthetic repos because the equivalence check is O(nodes
+ edges) and we want sub-second tests.  The cross-file semantic-edge
case (imports / inheritance / invokes spanning multiple files) is the
non-trivial part — make sure those tests stay green if you touch
``_rerun_semantic_passes`` or ``_wipe_semantic_edges``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))

from rpg.dep_graph import DependencyGraph, _hash_content  # noqa: E402
from rpg.models import EdgeType, NodeType  # noqa: E402


# ---------------------------------------------------------------------------
# Snapshot / equivalence helpers
# ---------------------------------------------------------------------------

def _node_snapshot(g: DependencyGraph) -> Dict[str, Dict]:
    """Capture each node's stable identity attrs (excludes ``ast`` etc.)."""
    snap: Dict[str, Dict] = {}
    for nid, attrs in g.G.nodes(data=True):
        snap[nid] = {
            "type": attrs.get("type"),
            "name": attrs.get("name"),
            "module": attrs.get("module"),
            "signature": attrs.get("signature"),
            "start_line": attrs.get("start_line"),
            "end_line": attrs.get("end_line"),
        }
    return snap


def _edge_snapshot(g: DependencyGraph) -> Set[Tuple[str, str, str]]:
    """Capture (src, dst, edge_type) tuples — order-independent."""
    return {
        (u, v, attrs.get("type", ""))
        for u, v, attrs in g.G.edges(data=True)
    }


def _build_full(repo_dir: Path) -> DependencyGraph:
    """Build + parse from scratch — the ground-truth comparison target."""
    g = DependencyGraph(str(repo_dir))
    g.build()
    g.parse()
    return g


# ---------------------------------------------------------------------------
# Fixture repos
# ---------------------------------------------------------------------------

@pytest.fixture
def repo_two_files(tmp_path):
    """Tiny repo with a cross-file invoke + inherit + import.

    Layout::

        repo/
          base.py     — class Base, function helper
          consumer.py — class Child(Base), imports helper, calls Base, helper
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "base.py").write_text(
        'class Base:\n'
        '    def greet(self):\n'
        '        return "hi"\n'
        'def helper():\n'
        '    return 42\n'
    )
    (repo / "consumer.py").write_text(
        'from base import Base, helper\n'
        'class Child(Base):\n'
        '    def go(self):\n'
        '        Base().greet()\n'
        '        return helper()\n'
    )
    return repo


@pytest.fixture
def repo_three_files(tmp_path):
    """A repo with 3 files and a small dependency fan-out."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "core.py").write_text(
        'def add(a, b):\n    return a + b\n'
        'def sub(a, b):\n    return a - b\n'
    )
    (repo / "util.py").write_text(
        'from core import add\n'
        'def double(x):\n'
        '    return add(x, x)\n'
    )
    (repo / "main.py").write_text(
        'from util import double\n'
        'from core import sub\n'
        'def run():\n'
        '    return double(5) + sub(3, 2)\n'
    )
    return repo


# ---------------------------------------------------------------------------
# content_hash basics
# ---------------------------------------------------------------------------

def test_build_populates_content_hash(repo_two_files):
    g = _build_full(repo_two_files)
    h = g.G.nodes["base.py"]["content_hash"]
    assert isinstance(h, str) and len(h) == 64  # sha256 hex
    # Recomputing on the same content gives the same hash
    src = (repo_two_files / "base.py").read_text()
    assert h == _hash_content(src)


def test_content_hash_round_trips_through_to_dict(repo_two_files, tmp_path):
    g = _build_full(repo_two_files)
    raw = g.to_dict()
    # content_hash is a regular node attr — must survive serialisation
    assert "content_hash" in raw["nodes"]["base.py"]
    saved = tmp_path / "dep_graph.json"
    saved.write_text(json.dumps(raw))
    restored = DependencyGraph.from_dict(json.loads(saved.read_text()))
    assert restored.G.nodes["base.py"]["content_hash"] == g.G.nodes["base.py"]["content_hash"]


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------

def test_remove_file_drops_file_and_descendants(repo_two_files):
    g = _build_full(repo_two_files)
    # base.py owns Base, Base.greet, helper
    removed = g.remove_file("base.py")
    assert removed >= 3
    assert "base.py" not in g.G
    assert "base.py:Base" not in g.G
    assert "base.py:Base.greet" not in g.G
    assert "base.py:helper" not in g.G
    # Cross-file edges into removed nodes are gone too
    assert not any(
        v.startswith("base.py")
        for _u, v, _attrs in g.G.edges(data=True)
    )


def test_remove_file_is_idempotent(repo_two_files):
    g = _build_full(repo_two_files)
    assert g.remove_file("base.py") > 0
    assert g.remove_file("base.py") == 0  # second call: no-op
    # Removing a file that never existed: also no-op
    assert g.remove_file("does/not/exist.py") == 0


def test_remove_file_preserves_directory_node(repo_two_files):
    g = _build_full(repo_two_files)
    g.remove_file("base.py")
    # "." (root dir) still exists because consumer.py still lives there
    assert "." in g.G


def test_remove_file_refuses_to_remove_directory(repo_three_files):
    """Safety net: a caller passing a directory path must NOT recursively wipe a subtree.  This guards against a real bug found during development where ``update_files(['.'])`` deleted every file node."""
    g = _build_full(repo_three_files)
    nodes_before = g.G.number_of_nodes()
    # "." is a DIRECTORY node, not a FILE → must be rejected
    assert g.remove_file(".") == 0
    assert g.G.number_of_nodes() == nodes_before
    # Code-unit nodes are also not files
    assert g.remove_file("core.py:add") == 0
    assert g.G.number_of_nodes() == nodes_before


# ---------------------------------------------------------------------------
# add_file — fresh add to an empty graph
# ---------------------------------------------------------------------------

def test_add_file_creates_units_for_a_fresh_graph(repo_two_files):
    g = DependencyGraph(str(repo_two_files))
    # No build() call: graph is empty
    assert g.G.number_of_nodes() == 0

    added = g.add_file("base.py")
    assert added is True
    assert "base.py" in g.G
    assert "base.py:Base" in g.G
    assert "base.py:Base.greet" in g.G
    assert "base.py:helper" in g.G
    # content_hash is set
    assert "content_hash" in g.G.nodes["base.py"]
    # But NO semantic edges yet (those need the global pass)
    assert not any(
        attrs.get("type") in (EdgeType.IMPORTS, EdgeType.INHERITS, EdgeType.INVOKES)
        for _u, _v, attrs in g.G.edges(data=True)
    )


def test_add_file_creates_parent_directories(tmp_path):
    """A deeply nested file's parent dir nodes must be auto-created."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a" / "b" / "c").mkdir(parents=True)
    (repo / "a" / "b" / "c" / "deep.py").write_text("def f(): pass\n")

    g = DependencyGraph(str(repo))
    assert g.add_file("a/b/c/deep.py") is True
    assert "a" in g.G and "a/b" in g.G and "a/b/c" in g.G
    assert "a/b/c/deep.py" in g.G


def test_add_file_returns_false_for_missing_path(repo_two_files):
    g = _build_full(repo_two_files)
    # File doesn't exist on disk
    assert g.add_file("ghost.py") is False
    # Filter excludes it (starts with dot)
    (repo_two_files / ".dotfile.py").write_text("x = 1\n")
    assert g.add_file(".dotfile.py") is False


def test_add_file_keeps_node_on_syntax_error(tmp_path):
    """A file with bad syntax: file node exists (with hash) but no units."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "broken.py").write_text("def foo(:\n")  # SyntaxError

    g = DependencyGraph(str(repo))
    assert g.add_file("broken.py") is True
    assert "broken.py" in g.G
    assert "content_hash" in g.G.nodes["broken.py"]
    # No code units were created
    assert not any(
        nid.startswith("broken.py:") for nid in g.G.nodes
    )


# ---------------------------------------------------------------------------
# update_files — equivalence with full rebuild
# ---------------------------------------------------------------------------

def test_update_files_noop_on_unchanged_content(repo_three_files):
    g = _build_full(repo_three_files)
    nodes_before = _node_snapshot(g)
    edges_before = _edge_snapshot(g)

    # Tell update_files all 3 files "changed" — but the disk content
    # is identical to what was just parsed, so the hash check must
    # short-circuit and the graph must be byte-identical.
    stats = g.update_files(["core.py", "util.py", "main.py"])
    assert stats["unchanged_hash"] == 3
    assert stats["modified"] == 0
    assert stats["added"] == 0
    assert stats["deleted"] == 0
    assert _node_snapshot(g) == nodes_before
    assert _edge_snapshot(g) == edges_before


def test_update_files_after_modification_matches_full_rebuild(repo_three_files):
    """The headline correctness invariant: incremental == full rebuild."""
    g_inc = _build_full(repo_three_files)

    # Modify util.py: add a new function that calls into core.sub
    (repo_three_files / "util.py").write_text(
        'from core import add, sub\n'
        'def double(x):\n'
        '    return add(x, x)\n'
        'def triple(x):\n'
        '    return add(x, double(x)) + sub(0, 0)\n'
    )
    stats = g_inc.update_files(["util.py"])
    assert stats["modified"] == 1
    assert stats["unchanged_hash"] == 0

    # Ground truth: a full rebuild from the current on-disk state.
    g_full = _build_full(repo_three_files)
    assert _node_snapshot(g_inc) == _node_snapshot(g_full), (
        "Incremental node set diverged from full rebuild"
    )
    assert _edge_snapshot(g_inc) == _edge_snapshot(g_full), (
        "Incremental edge set diverged from full rebuild"
    )


def test_update_files_handles_deletion(repo_three_files):
    g_inc = _build_full(repo_three_files)
    (repo_three_files / "util.py").unlink()

    stats = g_inc.update_files(["util.py"])
    assert stats["deleted"] == 1
    assert "util.py" not in g_inc.G

    g_full = _build_full(repo_three_files)
    assert _node_snapshot(g_inc) == _node_snapshot(g_full)
    assert _edge_snapshot(g_inc) == _edge_snapshot(g_full)


def test_update_files_handles_addition(repo_three_files):
    g_inc = _build_full(repo_three_files)
    # Brand new file referencing existing ones
    (repo_three_files / "extra.py").write_text(
        'from core import add\n'
        'def quadruple(x):\n'
        '    return add(x, add(x, add(x, x)))\n'
    )
    stats = g_inc.update_files(["extra.py"])
    assert stats["added"] == 1
    assert "extra.py" in g_inc.G
    assert "extra.py:quadruple" in g_inc.G

    g_full = _build_full(repo_three_files)
    assert _node_snapshot(g_inc) == _node_snapshot(g_full)
    assert _edge_snapshot(g_inc) == _edge_snapshot(g_full)


def test_update_files_handles_rename_preserving_cross_file_edges(repo_two_files):
    """``git mv`` must produce an end-state matching a from-scratch build.

    The most important assertion is the edge-equivalence check: a
    rename should not leave orphan edges pointing at the old file
    (which a naive ``delete+add`` would).
    """
    g_inc = _build_full(repo_two_files)

    # Simulate git mv base.py → core.py and update consumer.py to import
    # from the new module.
    (repo_two_files / "base.py").rename(repo_two_files / "core.py")
    (repo_two_files / "consumer.py").write_text(
        'from core import Base, helper\n'
        'class Child(Base):\n'
        '    def go(self):\n'
        '        Base().greet()\n'
        '        return helper()\n'
    )

    stats = g_inc.update_files(
        ["consumer.py"],
        renames={"base.py": "core.py"},
    )
    assert stats["renamed"] == 1
    assert "base.py" not in g_inc.G
    assert "core.py" in g_inc.G
    assert "core.py:Base" in g_inc.G

    g_full = _build_full(repo_two_files)
    assert _node_snapshot(g_inc) == _node_snapshot(g_full)
    assert _edge_snapshot(g_inc) == _edge_snapshot(g_full)


def test_update_files_complex_sequence(repo_three_files):
    """A realistic commit: edit one file, delete another, add a new one."""
    g_inc = _build_full(repo_three_files)

    # Edit core.py
    (repo_three_files / "core.py").write_text(
        'def add(a, b):\n    return a + b\n'
        'def mul(a, b):\n    return a * b\n'  # renamed sub -> mul
    )
    # Delete util.py
    (repo_three_files / "util.py").unlink()
    # Add a new file
    (repo_three_files / "fresh.py").write_text(
        'from core import mul\n'
        'def square(x):\n    return mul(x, x)\n'
    )

    stats = g_inc.update_files(["core.py", "util.py", "fresh.py"])
    assert stats["modified"] == 1
    assert stats["deleted"] == 1
    assert stats["added"] == 1
    # Sub got deleted from core.py; the edge that used to exist
    # (main.py:run → core.py:sub) is no longer represented because
    # main.py still imports sub but it doesn't exist any more — the
    # ground-truth rebuild has the same gap.
    g_full = _build_full(repo_three_files)
    assert _node_snapshot(g_inc) == _node_snapshot(g_full)
    assert _edge_snapshot(g_inc) == _edge_snapshot(g_full)


def test_update_files_with_rebuild_semantic_edges_false_skips_passes(repo_three_files):
    g = _build_full(repo_three_files)
    semantic_edges_before = {
        (u, v, k) for u, v, k, attrs in g.G.edges(keys=True, data=True)
        if attrs.get("type") in (EdgeType.IMPORTS, EdgeType.INHERITS, EdgeType.INVOKES)
    }
    # Touch util.py
    (repo_three_files / "util.py").write_text(
        'def isolated():\n    return 1\n'
    )
    stats = g.update_files(["util.py"], rebuild_semantic_edges=False)
    assert stats["modified"] == 1
    # No re-semanticise was performed: existing edges into util.py:double
    # are stale, but the caller asked for this behaviour explicitly so
    # we don't second-guess them.
    assert stats["edges_resemanticised"] == 0
    # The stale edge set is still present (caller's responsibility to fix).
    semantic_edges_after = {
        (u, v, k) for u, v, k, attrs in g.G.edges(keys=True, data=True)
        if attrs.get("type") in (EdgeType.IMPORTS, EdgeType.INHERITS, EdgeType.INVOKES)
    }
    # We don't claim equality — the contract is just "we didn't run the
    # passes".  Sanity check: edge count didn't increase.
    assert len(semantic_edges_after) <= len(semantic_edges_before)
