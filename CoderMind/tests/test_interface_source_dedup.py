"""Tests for shared interface-source dedup and its source-level wiring.

``interfaces.json`` stores ``file_code`` as the join of every unit's code.
Non-Python interface synthesis stores the whole-file text as each unit's
code, so the join repeats the file once per unit. ``code_dedup`` collapses
that duplication; ``InterfacesStore.to_interfaces_json`` applies it at the
source so the serialized artifact (and the code-gen seed file written from
it) is a clean single file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.code_dedup import dedup_code_blocks, dedup_file_code  # noqa: E402
from func_design.interfaces_store import InterfaceUnit, InterfacesStore  # noqa: E402


_WHOLE = "//! mod doc\nuse std::path::Path;\n\nstruct A {}\nfn new() {}\nfn run() {}\n"


class TestDedupHelpers:
    def test_collapses_repeated_whole_file(self):
        assert dedup_file_code([_WHOLE, _WHOLE, _WHOLE]) == _WHOLE.strip()

    def test_preserves_distinct_slices(self):
        out = dedup_file_code(["fn a() {}", "fn b() {}", "fn c() {}"])
        assert out == "fn a() {}\n\nfn b() {}\n\nfn c() {}"

    def test_blank_and_empty(self):
        assert dedup_file_code(["", "   ", "\n"]) == ""
        assert dedup_file_code([], fallback="FB") == "FB"
        assert dedup_file_code(["", ""], fallback="FB") == "FB"

    def test_dedup_code_blocks_order_preserved(self):
        assert dedup_code_blocks(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


class TestSerializationDedup:
    def test_to_interfaces_json_collapses_whole_file_per_unit(self):
        # Non-Python synthesis stores the whole file as every unit's code.
        store = InterfacesStore()
        for name in ("struct A", "function new", "function run"):
            store.add_unit(
                InterfaceUnit(
                    name=name,
                    file_path="src/a.rs",
                    subtree_name="Core",
                    features=["f"],
                    code=_WHOLE,
                )
            )
        store.subtree_order = ["Core"]

        data = store.to_interfaces_json()
        fc = data["subtrees"]["Core"]["interfaces"]["src/a.rs"]["file_code"]

        # file_code is the single file, not three concatenated copies.
        assert fc.count("//! mod doc") == 1
        assert fc.count("use std::path::Path;") == 1
        # units_to_code is untouched (still per-unit entries, valid as stubs).
        utc = data["subtrees"]["Core"]["interfaces"]["src/a.rs"]["units_to_code"]
        assert set(utc) == {"struct A", "function new", "function run"}

    def test_to_interfaces_json_keeps_distinct_unit_slices(self):
        store = InterfacesStore()
        slices = {
            "function a": "fn a() {}",
            "function b": "fn b() {}",
        }
        for name, code in slices.items():
            store.add_unit(
                InterfaceUnit(
                    name=name,
                    file_path="src/b.rs",
                    subtree_name="Core",
                    features=["f"],
                    code=code,
                )
            )
        store.subtree_order = ["Core"]

        data = store.to_interfaces_json()
        fc = data["subtrees"]["Core"]["interfaces"]["src/b.rs"]["file_code"]
        assert "fn a() {}" in fc
        assert "fn b() {}" in fc
