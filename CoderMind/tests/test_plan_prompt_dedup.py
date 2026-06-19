"""Tests for plan_tasks interface-source deduplication.

The interface synthesis stores the whole-file text as every unit's code, so
``file_code`` (built as ``"\n\n".join(unit codes)``) repeats the entire file
once per unit. On large modules that O(units x file_size) blow-up pushes the
planner prompt past the 128 KB single-argument limit and crashes ``plan_tasks``.
``_dedup_interface_source`` collapses the duplication while preserving genuinely
distinct per-unit slices.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plan_tasks import (  # type: ignore[import-not-found]  # noqa: E402
    _dedup_interface_source,
    _validate_interfaces_cover_skeleton_features,
)


def test_collapses_whole_file_repeated_per_unit():
    whole = "//! mod doc\nuse std::path::Path;\n\nstruct A {}\nfn new() {}\nfn run() {}\n"
    fdata = {
        "units_to_code": {"struct A": whole, "fn new": whole, "fn run": whole},
        "file_code": "\n\n".join([whole, whole, whole]),
    }
    out = _dedup_interface_source(fdata)
    assert out.count("//! mod doc") == 1
    assert out.count("use std::path::Path;") == 1
    assert len(out) < len(fdata["file_code"])


def test_preserves_distinct_unit_slices():
    fdata = {
        "units_to_code": {"a": "fn a() {}", "b": "fn b() {}", "c": "fn c() {}"},
        "file_code": "fn a() {}\n\nfn b() {}\n\nfn c() {}",
    }
    out = _dedup_interface_source(fdata)
    for symbol in ("fn a()", "fn b()", "fn c()"):
        assert symbol in out


def test_empty_units_falls_back_to_file_code():
    fdata = {"units_to_code": {}, "file_code": "raw source"}
    assert _dedup_interface_source(fdata) == "raw source"


def test_rejects_partial_interfaces_against_skeleton(tmp_path):
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(
        """
        {
          "root": {
            "type": "directory",
            "children": [
              {"type": "file", "path": "src/a.cpp", "feature_paths": ["Core/a"]},
              {"type": "file", "path": "src/b.cpp", "feature_paths": ["Core/b"]}
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    interfaces = {
        "subtrees": {
            "Core": {
                "interfaces": {
                    "src/a.cpp": {
                        "units_to_features": {"function a": ["Core/a"]},
                    }
                }
            }
        }
    }

    try:
        _validate_interfaces_cover_skeleton_features(interfaces, skeleton_path)
    except ValueError as exc:
        assert "Core/b" in str(exc)
    else:
        raise AssertionError("partial interfaces should be rejected")


def test_accepts_complete_interfaces_against_skeleton(tmp_path):
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(
        """
        {
          "root": {
            "type": "directory",
            "children": [
              {"type": "file", "path": "src/a.cpp", "feature_paths": ["Core/a"]}
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    interfaces = {
        "subtrees": {
            "Core": {
                "interfaces": {
                    "src/a.cpp": {
                        "units_to_features": {"function a": ["Core/a"]},
                    }
                }
            }
        }
    }

    _validate_interfaces_cover_skeleton_features(interfaces, skeleton_path)
