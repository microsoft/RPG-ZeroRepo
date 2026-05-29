"""Regression tests for the ``cmind update`` path-format bugs.

After unifying RPG node paths to the canonical codegen format
(``file::Name`` / ``file::Cls::method``) the ``RPGService`` helpers
that bridge between dep_graph node IDs and RPG ``meta.path`` strings
were producing legacy ``::class X`` forms, which would silently revert
canonical paths to legacy on every ``cmind update`` run.  These tests
pin down the canonical behavior so the regression does not re-emerge.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from rpg import RPG, Node, NodeMetaData, NodeType, EdgeType  # noqa: E402
from rpg.service import RPGService  # noqa: E402


# ============================================================================
# _dep_id_to_rpg_path: dep node id -> canonical RPG meta.path
# ============================================================================

class TestDepIdToRpgPath:

    def _svc(self):
        return RPGService(RPG(repo_name="test"))

    def test_function_no_prefix(self):
        svc = self._svc()
        # ``G`` is unused by the new implementation; pass a dummy mock.
        assert (
            svc._dep_id_to_rpg_path("src/foo.py:bar", MagicMock())
            == "src/foo.py::bar"
        )

    def test_class_no_prefix(self):
        svc = self._svc()
        assert (
            svc._dep_id_to_rpg_path("src/foo.py:User", MagicMock())
            == "src/foo.py::User"
        )

    def test_method_uses_double_colon(self):
        """``foo.py:Cls.method`` (dep_graph) must become ``foo.py::Cls::method`` (canonical RPG), NOT ``foo.py::class Cls.method`` or any legacy variant."""
        svc = self._svc()
        result = svc._dep_id_to_rpg_path("src/foo.py:User.login", MagicMock())
        assert result == "src/foo.py::User::login"
        # Legacy prefix MUST NOT leak.
        assert "class " not in result
        assert "function " not in result
        assert "method " not in result

    def test_file_id_unchanged(self):
        svc = self._svc()
        assert (
            svc._dep_id_to_rpg_path("src/foo.py", MagicMock())
            == "src/foo.py"
        )

    def test_directory_id_unchanged(self):
        svc = self._svc()
        assert svc._dep_id_to_rpg_path("src", MagicMock()) == "src"


# ============================================================================
# _rpg_path_to_dep_id: canonical (or legacy) RPG path -> dep node id
# ============================================================================

class TestRpgPathToDepId:

    def _svc(self):
        return RPGService(RPG(repo_name="test"))

    def test_canonical_function(self):
        svc = self._svc()
        assert svc._rpg_path_to_dep_id("src/foo.py::bar") == "src/foo.py:bar"

    def test_canonical_class(self):
        svc = self._svc()
        assert svc._rpg_path_to_dep_id("src/foo.py::User") == "src/foo.py:User"

    def test_canonical_method(self):
        """The previous ``rsplit('::', 1)`` implementation produced ``src/foo.py::Cls:m`` for method paths — a malformed dep_graph id that never matched any real node."""
        svc = self._svc()
        assert (
            svc._rpg_path_to_dep_id("src/foo.py::User::login")
            == "src/foo.py:User.login"
        )

    def test_legacy_class_prefix_tolerated(self):
        """Older rpg.json files may still carry ``::class X`` paths; the conversion must strip the legacy prefix so that align logic can recognise the same dep node and (eventually) rewrite the legacy path to canonical."""
        svc = self._svc()
        assert (
            svc._rpg_path_to_dep_id("src/foo.py::class User")
            == "src/foo.py:User"
        )

    def test_legacy_function_prefix_tolerated(self):
        svc = self._svc()
        assert (
            svc._rpg_path_to_dep_id("src/foo.py::function register")
            == "src/foo.py:register"
        )

    def test_file_only_returns_none(self):
        svc = self._svc()
        assert svc._rpg_path_to_dep_id("src/foo.py") is None

    def test_empty_returns_none(self):
        svc = self._svc()
        assert svc._rpg_path_to_dep_id("") is None
        assert svc._rpg_path_to_dep_id(None) is None


# ============================================================================
# Round-trip: dep -> rpg -> dep should be idempotent for canonical paths
# ============================================================================

class TestRoundTrip:

    def _svc(self):
        return RPGService(RPG(repo_name="test"))

    def test_function_roundtrip(self):
        svc = self._svc()
        dep_id = "src/foo.py:bar"
        rpg_path = svc._dep_id_to_rpg_path(dep_id, MagicMock())
        assert svc._rpg_path_to_dep_id(rpg_path) == dep_id

    def test_class_roundtrip(self):
        svc = self._svc()
        dep_id = "src/foo.py:User"
        rpg_path = svc._dep_id_to_rpg_path(dep_id, MagicMock())
        assert svc._rpg_path_to_dep_id(rpg_path) == dep_id

    def test_method_roundtrip(self):
        svc = self._svc()
        dep_id = "src/foo.py:User.login"
        rpg_path = svc._dep_id_to_rpg_path(dep_id, MagicMock())
        assert rpg_path == "src/foo.py::User::login"
        assert svc._rpg_path_to_dep_id(rpg_path) == dep_id


# ============================================================================
# process_diff accepts max_exclude_votes (the hardcoded 3 was a regression)
# ============================================================================

class TestProcessDiffSignature:

    def test_max_exclude_votes_parameter_exists(self):
        """``cmind update`` should not silently spend 4 LLM calls on exclude_files; ``process_diff`` must accept and propagate the ``max_exclude_votes`` parameter so callers can opt for the single-call default."""
        import inspect
        from rpg_encoder.rpg_evolution import RPGEvolution
        sig = inspect.signature(RPGEvolution.process_diff)
        assert "max_exclude_votes" in sig.parameters
        # Default should match the encoder side (1) — minimal LLM cost.
        assert sig.parameters["max_exclude_votes"].default == 1

    def test_run_update_rpg_propagates_max_exclude_votes(self):
        from rpg_encoder.run_update_rpg import run_update_rpg
        import inspect
        sig = inspect.signature(run_update_rpg)
        assert "max_exclude_votes" in sig.parameters
        assert sig.parameters["max_exclude_votes"].default == 1
