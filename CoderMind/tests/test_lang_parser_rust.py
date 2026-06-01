#!/usr/bin/env python3
"""Tests for the Rust language parser."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser import parse_file, validate_syntax


RUST_SOURCE = textwrap.dedent(
    """\
    use crate::error::GwsError;
    use std::collections::HashMap;

    pub struct Client {
        url: String,
    }

    pub enum Status {
        Ok,
        Error(String),
    }

    pub trait Provider {
        fn get(&self) -> String;
    }

    pub fn build_client() -> Result<Client, GwsError> {
        todo!()
    }

    impl Client {
        pub fn new(url: String) -> Self {
            Client { url }
        }

        pub fn fetch(&self) -> Status {
            build_client().unwrap();
            Status::Ok
        }
    }
    """
)


def _unit_map(result):
    return {(unit.unit_type, unit.name): unit for unit in result.units}


class TestRustParser:
    def test_extracts_struct_enum_trait_function_and_methods(self):
        result = parse_file("src/client.rs", RUST_SOURCE)
        assert result.file_path == "src/client.rs"
        assert result.language == "rust"
        assert result.syntax_error is None

        units = _unit_map(result)
        assert ("import", "crate::error::GwsError") in units
        assert ("import", "std::collections::HashMap") in units
        assert ("struct", "Client") in units
        assert ("enum", "Status") in units
        assert ("trait", "Provider") in units
        assert ("function", "build_client") in units
        assert ("method", "new") in units
        assert ("method", "fetch") in units
        assert units[("method", "new")].parent == "Client"
        assert units[("method", "fetch")].parent == "Client"

    def test_trait_methods_are_parented_to_trait(self):
        result = parse_file("src/client.rs", RUST_SOURCE)
        trait_methods = [unit for unit in result.units if unit.unit_type == "method" and unit.parent == "Provider"]
        assert [unit.name for unit in trait_methods] == ["get"]

    def test_dependencies_are_recorded_for_use_declarations(self):
        result = parse_file("src/client.rs", RUST_SOURCE)
        imports = [dep for dep in result.dependencies if dep.relation == "imports"]
        assert [(dep.dst, dep.extra["import_kind"]) for dep in imports] == [
            ("crate::error::GwsError", "rust_use"),
            ("std::collections::HashMap", "rust_use"),
        ]

    def test_invokes_include_direct_calls_but_not_macros_or_enum_variants(self):
        result = parse_file("src/client.rs", RUST_SOURCE)
        invokes = [dep for dep in result.dependencies if dep.relation == "invokes"]
        invoke_keys = {(dep.src, dep.symbol, dep.extra["call_kind"]) for dep in invokes}
        assert ("src/client.rs:Client.fetch", "build_client", "direct") in invoke_keys
        assert all(dep.symbol != "todo" for dep in invokes)
        assert all(dep.symbol != "Ok" for dep in invokes)

    def test_mod_decl_produces_import_unit_and_dependency(self):
        result = parse_file("src/lib.rs", "mod error;\npub mod services;\n")
        units = _unit_map(result)
        assert ("import", "error") in units
        assert ("import", "services") in units
        imports = [dep for dep in result.dependencies if dep.relation == "imports"]
        assert [(dep.dst, dep.extra["import_kind"]) for dep in imports] == [
            ("error", "rust_mod_decl"),
            ("services", "rust_mod_decl"),
        ]

    def test_grouped_use_imports_expand_to_multiple_units(self):
        result = parse_file("src/lib.rs", "use crate::foo::{A, B};\n")
        imports = [unit.name for unit in result.units if unit.unit_type == "import"]
        assert imports == ["crate::foo::A", "crate::foo::B"]

    def test_trait_impl_emits_high_confidence_inherits_dependency(self):
        source = textwrap.dedent(
            """\
            pub trait Provider {
                fn get(&self) -> String;
            }

            pub struct Client;

            impl Provider for Client {
                fn get(&self) -> String {
                    String::new()
                }
            }
            """
        )
        result = parse_file("src/client.rs", source)
        inherits = [dep for dep in result.dependencies if dep.relation == "inherits"]
        assert len(inherits) == 1
        assert inherits[0].src == "Client"
        assert inherits[0].dst == "Provider"
        assert inherits[0].confidence == "high"

    def test_units_preserve_language_and_line_metadata(self):
        result = parse_file("src/client.rs", RUST_SOURCE)
        assert result.units
        for unit in result.units:
            assert unit.language == "rust"
            assert unit.line_start is not None
            assert unit.line_end is not None
            assert unit.extra["language"] == "rust"
            assert unit.extra["line_start"] == unit.line_start
            assert unit.extra["line_end"] == unit.line_end

    def test_invalid_source_returns_syntax_error_without_crashing(self):
        result = parse_file("bad.rs", "pub fn broken(\n")
        assert result.language == "rust"
        assert result.syntax_error is not None
        valid, error = validate_syntax("bad.rs", "pub fn broken(\n")
        assert valid is False
        assert error is not None
