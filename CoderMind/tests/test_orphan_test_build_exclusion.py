"""Regression tests for excluding test/build units from orphan detection.

Reproduces the plan-stage WARN seen across languages
(``global_review.passed=false ... N orphan feature(s)``) where the orphan
heuristic flagged TEST functions and BUILD targets. Those units are
callable, so the type-like (``is_callable``) exclusion does not cover
them, yet they have no incoming *production* invocation edge — they are
invoked by an external runner (test framework / ``make``), so flagging
them as dead code is a false positive.

Exclusion uses two complementary signals:
  * language-agnostic: the feature path / subtree category
    (``Testing`` / ``Build System`` / ...);
  * per-language: ``backend.is_test_file`` on the unit's file.

Real production dead code (a production-category callable with no edges)
must STILL be flagged, so the gate keeps its value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from decoder_lang import get_backend  # noqa: E402
from func_design.interface_review import (  # noqa: E402
    _is_non_production_feature,
    check_call_graph_connectivity,
    check_feature_dependency_coverage,
)


def _interfaces(units_to_features: dict, *, subtree: str, file_path: str) -> dict:
    """Build a minimal interfaces_data with one subtree/file/units block.

    Includes both the ``units`` list (consumed by ``build_call_graph`` to
    register graph nodes) and ``units_to_features`` (consumed by the
    feature-coverage check), so both orphan detectors see the unit.
    """
    return {
        "subtrees": {
            subtree: {
                "interfaces": {
                    file_path: {
                        "units": list(units_to_features.keys()),
                        "units_to_features": units_to_features,
                    },
                },
            },
        },
    }


_EMPTY_FLOW: dict = {
    "invocation_edges": [],
    "inheritance_edges": [],
    "reference_edges": [],
}


class TestIsNonProductionFeature:
    @pytest.mark.parametrize(
        "features,subtree",
        [
            (["Testing/error reporting/verify division by zero"], "Testing"),
            (["Build System/make targets/run test suite"], "Build System"),
            ([], "Test Infrastructure"),
            (["Tooling/lint/run linter"], "Tooling"),
        ],
    )
    def test_test_and_build_categories_are_non_production(self, features, subtree):
        assert _is_non_production_feature(features, subtree) is True

    @pytest.mark.parametrize(
        "features,subtree",
        [
            (["Task Store/add/append todo"], "Task Store"),
            (["Web Routes/handle add"], "Web Routes"),
            ([], "Data Layer"),
        ],
    )
    def test_production_categories_are_production(self, features, subtree):
        assert _is_non_production_feature(features, subtree) is False

    def test_case_insensitive_and_path_head(self):
        assert _is_non_production_feature(["TESTING/x/y"], "") is True
        assert _is_non_production_feature(["tests/unit/foo"], "") is True


class TestFeatureCoverageExcludesTestBuild:
    def test_test_function_not_flagged_by_category(self):
        # A callable test function with no incoming edge: previously an
        # orphan, now excluded by the Testing category (no backend needed).
        data = _interfaces(
            {"function test_division_by_zero": ["Testing/error reporting/div by zero"]},
            subtree="Testing",
            file_path="tests/test_errors.c",
        )
        orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
        )
        assert orphans == []

    def test_build_target_not_flagged_by_category(self):
        data = _interfaces(
            {"function build_run_tests": ["Build System/make targets/run test suite"]},
            subtree="Build System",
            file_path="build/Makefile",
        )
        orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
        )
        assert orphans == []

    def test_test_file_excluded_even_with_production_category(self):
        # Defence in depth: a unit in a test file is excluded via
        # is_test_file even if its feature category were not recognised.
        data = _interfaces(
            {"function helper_in_test": ["Some Category/x/y"]},
            subtree="Some Category",
            file_path="internal/store/store_test.go",
        )
        orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )
        assert orphans == []

    def test_real_production_dead_code_still_flagged(self):
        # A production-category callable with no incoming edge must STILL
        # be an orphan — the gate keeps its value.
        data = _interfaces(
            {"function unused_helper": ["Data Layer/transform/normalize"]},
            subtree="Data Layer",
            file_path="src/data.c",
        )
        orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
            is_test_file=get_backend("c").is_test_file,
        )
        assert len(orphans) == 1
        assert orphans[0]["unit_name"] == "function unused_helper"

    def test_outgoing_production_root_is_not_feature_orphan(self):
        data = _interfaces(
            {
                "function build_app": ["Runtime/bootstrap/wire dependencies"],
                "function make_store": [],
            },
            subtree="Runtime",
            file_path="src/app.c",
        )
        flow = {
            "invocation_edges": [
                {
                    "caller": "function build_app",
                    "caller_file": "src/app.c",
                    "callee": "function make_store",
                    "callee_file": "src/app.c",
                }
            ],
            "inheritance_edges": [],
            "reference_edges": [],
        }

        orphans = check_feature_dependency_coverage(
            data, flow, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
            is_test_file=get_backend("c").is_test_file,
        )

        assert orphans == []

    def test_bare_entry_point_name_matches_prefixed_unit(self):
        data = _interfaces(
            {"function RunMain": ["Runtime/startup/run main package"]},
            subtree="Runtime",
            file_path="internal/app/main.go",
        )
        entry_points = [
            {"file_path": "internal/app/main.go", "unit_name": "RunMain"}
        ]

        connectivity = check_call_graph_connectivity(
            data, _EMPTY_FLOW, entry_points=entry_points,
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )
        feature_orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=entry_points,
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )

        assert connectivity["orphan_units"] == []
        assert feature_orphans == []

    def test_receiver_entry_point_name_matches_method_unit(self):
        data = _interfaces(
            {"method ServeHTTP": ["Web/server/serve request"]},
            subtree="Web",
            file_path="internal/web/handler.go",
        )
        entry_points = [
            {
                "file_path": "internal/web/handler.go",
                "unit_name": "(*HomeHandler).ServeHTTP",
            }
        ]

        connectivity = check_call_graph_connectivity(
            data, _EMPTY_FLOW, entry_points=entry_points,
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )
        feature_orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=entry_points,
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )

        assert connectivity["orphan_units"] == []
        assert feature_orphans == []

    def test_ambiguous_entry_point_alias_without_file_is_not_overmatched(self):
        data = {
            "subtrees": {
                "Web": {
                    "interfaces": {
                        "internal/web/home.go": {
                            "units": ["method ServeHTTP"],
                            "units_to_features": {
                                "method ServeHTTP": ["Web/home/serve request"]
                            },
                        },
                        "internal/web/action.go": {
                            "units": ["method ServeHTTP"],
                            "units_to_features": {
                                "method ServeHTTP": ["Web/action/serve request"]
                            },
                        },
                    },
                },
            },
        }

        connectivity = check_call_graph_connectivity(
            data, _EMPTY_FLOW, entry_points=[{"unit_name": "ServeHTTP"}],
            is_callable=get_backend("go").is_callable_unit,
            is_test_file=get_backend("go").is_test_file,
        )

        assert len(connectivity["orphan_units"]) == 2

    def test_unit_and_feature_detectors_agree_on_same_graph(self):
        # The two orphan detectors must share one definition of "orphan"
        # so the published verdict can never drift from the structural
        # gate. A production root with an outgoing edge is an orphan for
        # NEITHER; a genuinely isolated production unit is an orphan for
        # BOTH.
        data = _interfaces(
            {
                "function wired_root": ["Runtime/bootstrap/wire app"],
                "function reachable_leaf": ["Runtime/bootstrap/make store"],
                "function dead_unit": ["Runtime/extra/never wired"],
            },
            subtree="Runtime",
            file_path="src/app.c",
        )
        flow = {
            "invocation_edges": [
                {
                    "caller": "function wired_root",
                    "caller_file": "src/app.c",
                    "callee": "function reachable_leaf",
                    "callee_file": "src/app.c",
                }
            ],
            "inheritance_edges": [],
            "reference_edges": [],
        }
        backend = get_backend("c")

        unit_orphans = check_call_graph_connectivity(
            data, flow, entry_points=[],
            is_callable=backend.is_callable_unit,
            is_test_file=backend.is_test_file,
        )["orphan_units"]
        feature_orphans = check_feature_dependency_coverage(
            data, flow, entry_points=[],
            is_callable=backend.is_callable_unit,
            is_test_file=backend.is_test_file,
        )

        unit_keys = {o["unit_key"] for o in unit_orphans}
        feature_keys = {
            f"{f['file_path']}::{f['unit_name']}" for f in feature_orphans
        }
        # Only the genuinely isolated unit is an orphan, and both
        # detectors agree on exactly that set.
        assert unit_keys == {"src/app.c::function dead_unit"}
        assert feature_keys == unit_keys

    def test_legacy_no_predicates_preserves_behaviour(self):
        # With no is_callable/is_test_file, the category check still applies
        # but file-level does not; production dead code is still flagged.
        data = _interfaces(
            {"function unused_helper": ["Data Layer/x/y"]},
            subtree="Data Layer",
            file_path="src/data.py",
        )
        orphans = check_feature_dependency_coverage(data, _EMPTY_FLOW, entry_points=[])
        assert len(orphans) == 1


class TestConnectivityExcludesTestBuild:
    def test_isolated_test_function_not_orphan_unit(self):
        data = _interfaces(
            {"function test_addition": ["Testing/eval/verify addition"]},
            subtree="Testing",
            file_path="tests/test_eval.c",
        )
        result = check_call_graph_connectivity(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
            is_test_file=get_backend("c").is_test_file,
        )
        assert result["orphan_units"] == []

    def test_isolated_production_function_still_orphan_unit(self):
        data = _interfaces(
            {"function unused": ["Data Layer/x/y"]},
            subtree="Data Layer",
            file_path="src/data.c",
        )
        result = check_call_graph_connectivity(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=get_backend("c").is_callable_unit,
            is_test_file=get_backend("c").is_test_file,
        )
        assert len(result["orphan_units"]) == 1


class TestPerLanguageTestFileExclusion:
    @pytest.mark.parametrize(
        "language,test_path",
        [
            ("python", "tests/test_store.py"),
            ("go", "internal/store/store_test.go"),
            ("rust", "tests/integration_test.rs"),
            ("javascript", "test/store.test.js"),
            ("typescript", "test/store.test.ts"),
            ("c", "tests/test_eval.c"),
            ("cpp", "tests/test_eval.cpp"),
        ],
    )
    def test_units_in_test_files_excluded(self, language, test_path):
        # Use a production-looking category so ONLY is_test_file can exclude it.
        data = _interfaces(
            {"function some_unit": ["Feature Area/x/y"]},
            subtree="Feature Area",
            file_path=test_path,
        )
        backend = get_backend(language)
        if not backend.is_test_file(test_path):
            pytest.skip(f"{language} backend does not classify {test_path} as a test file")
        orphans = check_feature_dependency_coverage(
            data, _EMPTY_FLOW, entry_points=[],
            is_callable=backend.is_callable_unit,
            is_test_file=backend.is_test_file,
        )
        assert orphans == []
