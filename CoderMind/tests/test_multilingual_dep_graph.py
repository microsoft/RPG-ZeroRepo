#!/usr/bin/env python3
"""Tests for multilingual DependencyGraph parsing."""

import os
import sys
import textwrap

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from rpg import EdgeType, NodeType
from rpg.dep_graph import DependencyGraph, _exclude_irrelevant_for_parse


def _parse_repo(repo_path):
    graph = DependencyGraph(str(repo_path))
    graph.build()
    graph.parse()
    return graph


def _edge_attrs(graph, src, dst, edge_type):
    edge_data = graph.G.get_edge_data(src, dst, default={})
    return [attrs for attrs in edge_data.values() if attrs.get("type") == edge_type]


def test_parse_filter_accepts_supported_sources_and_rejects_tests():
    assert _exclude_irrelevant_for_parse("cmd/server/main.go") is True
    assert _exclude_irrelevant_for_parse("src/app.ts") is True
    assert _exclude_irrelevant_for_parse("src/view.tsx") is True
    assert _exclude_irrelevant_for_parse("src/math.c") is True
    assert _exclude_irrelevant_for_parse("include/math_utils.h") is True
    assert _exclude_irrelevant_for_parse("src/model.cpp") is True
    assert _exclude_irrelevant_for_parse("include/model.hpp") is True
    assert _exclude_irrelevant_for_parse("src/main.rs") is True
    assert _exclude_irrelevant_for_parse("crates/foo/src/lib.rs") is True
    assert _exclude_irrelevant_for_parse("src/config.json") is False
    assert _exclude_irrelevant_for_parse("pkg/server/server_test.go") is False
    assert _exclude_irrelevant_for_parse("src/app.test.ts") is False
    assert _exclude_irrelevant_for_parse("tests/test_main.py") is False
    assert _exclude_irrelevant_for_parse("server/server_test.c") is False
    assert _exclude_irrelevant_for_parse("tests/helper.cpp") is False
    assert _exclude_irrelevant_for_parse("tests/helper.rs") is False
    assert _exclude_irrelevant_for_parse("examples/demo.rs") is False


def test_go_graph_structure_and_receiver_containment(tmp_path):
    source = textwrap.dedent(
        """\
        package server

        import "fmt"

        type Server struct {
            Name string
        }

        func NewServer(name string) *Server {
            return &Server{Name: name}
        }

        func (s *Server) Handle() {
            fmt.Println(s.Name)
        }
        """
    )
    server_dir = tmp_path / "internal" / "server"
    server_dir.mkdir(parents=True)
    (server_dir / "server.go").write_text(source)

    graph = _parse_repo(tmp_path)

    file_id = "internal/server/server.go"
    struct_id = f"{file_id}:Server"
    function_id = f"{file_id}:NewServer"
    method_id = f"{file_id}:Server.Handle"

    assert graph.G.nodes[file_id]["language"] == "go"
    assert graph.G.nodes[file_id]["unit_type"] == "file"
    assert graph.G.nodes[struct_id]["type"] == NodeType.CLASS
    assert graph.G.nodes[struct_id]["unit_type"] == "struct"
    assert graph.G.nodes[struct_id]["language"] == "go"
    assert graph.G.nodes[function_id]["type"] == NodeType.FUNCTION
    assert graph.G.nodes[function_id]["language"] == "go"
    assert graph.G.nodes[method_id]["type"] == NodeType.METHOD
    assert graph.G.nodes[method_id]["language"] == "go"
    assert graph.G.nodes[method_id]["receiver_type"] == "Server"
    assert graph.G.nodes[method_id]["code"].startswith("func (s *Server) Handle")

    assert _edge_attrs(graph, file_id, struct_id, EdgeType.CONTAINS)
    assert _edge_attrs(graph, struct_id, method_id, EdgeType.CONTAINS)
    assert any(
        attrs.get("type") == NodeType.IMPORT and attrs.get("language") == "go"
        for _, attrs in graph.G.nodes(data=True)
    )
    assert any(
        attrs.get("type") == NodeType.PACKAGE and attrs.get("language") == "go"
        for _, attrs in graph.G.nodes(data=True)
    )


def test_typescript_graph_structure_and_resolvable_import_edge(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "models.ts").write_text(
        textwrap.dedent(
            """\
            export class User {
              constructor(public name: string) {}
            }
            """
        )
    )
    (src / "app.ts").write_text(
        textwrap.dedent(
            """\
            import { User } from "./models";

            export class Greeter {
              greet(user: User): string {
                return `hello ${user.name}`;
              }
            }

            export function makeGreeter(): Greeter {
              return new Greeter();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    file_id = "src/app.ts"
    class_id = f"{file_id}:Greeter"
    method_id = f"{file_id}:Greeter.greet"
    function_id = f"{file_id}:makeGreeter"

    assert graph.G.nodes[file_id]["language"] == "typescript"
    assert graph.G.nodes[class_id]["type"] == NodeType.CLASS
    assert graph.G.nodes[class_id]["language"] == "typescript"
    assert graph.G.nodes[method_id]["type"] == NodeType.METHOD
    assert graph.G.nodes[method_id]["language"] == "typescript"
    assert graph.G.nodes[function_id]["type"] == NodeType.FUNCTION
    assert graph.G.nodes[function_id]["language"] == "typescript"
    assert _edge_attrs(graph, class_id, method_id, EdgeType.CONTAINS)

    import_edges = _edge_attrs(graph, file_id, "src/models.ts", EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is True
    assert import_edges[0]["confidence"] == "resolved"
    assert import_edges[0]["import_module"] == "./models"


def test_typescript_imported_and_same_file_invokes_resolve(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "debug.ts").write_text(
        textwrap.dedent(
            """\
            export function getDebugOption(): boolean {
              return true;
            }
            """
        )
    )
    (src / "remote.ts").write_text(
        textwrap.dedent(
            """\
            export class ChromeRemote {
              start(): void {}
            }
            """
        )
    )
    (src / "app.ts").write_text(
        textwrap.dedent(
            """\
            import { getDebugOption } from "./debug"
            import { ChromeRemote } from "./remote"

            export function localHelper(): void {}

            export function boot(): ChromeRemote {
              localHelper();
              getDebugOption();
              return new ChromeRemote();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    source_id = "src/app.ts:boot"
    assert _edge_attrs(graph, source_id, "src/app.ts:localHelper", EdgeType.INVOKES)
    assert _edge_attrs(graph, source_id, "src/debug.ts:getDebugOption", EdgeType.INVOKES)
    assert _edge_attrs(graph, source_id, "src/remote.ts:ChromeRemote", EdgeType.INVOKES)


def test_typescript_default_import_alias_constructor_resolves_to_default_export(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "local.ts").write_text(
        textwrap.dedent(
            """\
            export default class ActualClass {
              start(): void {}
            }
            """
        )
    )
    (src / "app.ts").write_text(
        textwrap.dedent(
            """\
            import LocalAlias from "./local"

            export function boot(): ActualClass {
              return new LocalAlias();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert graph.G.nodes["src/local.ts:ActualClass"]["export_default"] is True
    assert _edge_attrs(graph, "src/app.ts:boot", "src/local.ts:ActualClass", EdgeType.INVOKES)


def test_go_module_prefix_import_and_invokes_resolve(tmp_path):
    (tmp_path / "go.mod").write_text("module github.com/example/project\n")
    constraints_dir = tmp_path / "constraints"
    constraints_dir.mkdir()
    (constraints_dir / "doc.go").write_text("package constraints\n")
    (constraints_dir / "check.go").write_text(
        textwrap.dedent(
            """\
            package constraints

            func Check() bool {
                return true
            }
            """
        )
    )
    cmd_dir = tmp_path / "cmd"
    cmd_dir.mkdir()
    (cmd_dir / "helpers.go").write_text(
        textwrap.dedent(
            """\
            package cmd

            func AllC() bool {
                return true
            }
            """
        )
    )
    (cmd_dir / "app.go").write_text(
        textwrap.dedent(
            """\
            package cmd

            import "github.com/example/project/constraints"

            func Run() {
                AllC()
                constraints.Check()
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert _edge_attrs(graph, "cmd/app.go", "constraints/doc.go", EdgeType.IMPORTS)
    source_id = "cmd/app.go:Run"
    assert _edge_attrs(graph, source_id, "cmd/helpers.go:AllC", EdgeType.INVOKES)
    assert _edge_attrs(graph, source_id, "constraints/check.go:Check", EdgeType.INVOKES)


def test_go_same_package_generic_wrapper_invokes_resolve_across_files(tmp_path):
    channels_dir = tmp_path / "channels"
    channels_dir.mkdir()
    (channels_dir / "channel.go").write_text(
        textwrap.dedent(
            """\
            package channels

            func All[T any](c <-chan T) bool {
                return AllC(c)
            }

            func Any[T any](c <-chan T) bool {
                return AnyC(c)
            }
            """
        )
    )
    (channels_dir / "channel_ctx.go").write_text(
        textwrap.dedent(
            """\
            package channels

            func AllC[T any](c <-chan T) bool {
                return true
            }

            func AnyC[T any](c <-chan T) bool {
                return false
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert _edge_attrs(graph, "channels/channel.go:All", "channels/channel_ctx.go:AllC", EdgeType.INVOKES)
    assert _edge_attrs(graph, "channels/channel.go:Any", "channels/channel_ctx.go:AnyC", EdgeType.INVOKES)


def test_unresolved_typescript_import_is_represented_with_metadata(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.ts").write_text(
        textwrap.dedent(
            """\
            import { Missing } from "./missing";

            export function run(value: Missing): Missing {
              return value;
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    file_id = "src/app.ts"
    import_id = f"{file_id}:import:1"
    assert graph.G.nodes[import_id]["type"] == NodeType.IMPORT
    assert graph.G.nodes[import_id]["language"] == "typescript"
    assert graph.G.nodes[import_id]["resolved"] is False
    assert graph.G.nodes[import_id]["confidence"] == "unresolved"
    assert graph.G.nodes[import_id]["heuristic"] is True

    import_edges = _edge_attrs(graph, file_id, import_id, EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is False
    assert import_edges[0]["confidence"] == "unresolved"
    assert import_edges[0]["heuristic"] is True


def test_incremental_update_keeps_typescript_import_edges(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "models.ts").write_text("export class User {}\n")
    app = src / "app.ts"
    app.write_text(
        textwrap.dedent(
            """\
            import { User } from "./models";

            export function makeUser(): User {
              return new User();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)
    app.write_text(
        textwrap.dedent(
            """\
            import { User } from "./models";

            export function run(): User {
              return new User();
            }
            """
        )
    )

    stats = graph.update_files(["src/app.ts"])

    assert stats["modified"] == 1
    assert "src/app.ts:run" in graph.G
    import_edges = _edge_attrs(graph, "src/app.ts", "src/models.ts", EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is True


def test_non_python_syntax_error_metadata_does_not_abort_parsing(tmp_path):
    (tmp_path / "bad.go").write_text("package main\nfunc broken(\n")
    (tmp_path / "good.go").write_text(
        textwrap.dedent(
            """\
            package main

            func Works() string {
                return "ok"
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert graph.G.nodes["bad.go"]["language"] == "go"
    assert graph.G.nodes["bad.go"]["unit_type"] == "file"
    assert graph.G.nodes["bad.go"].get("syntax_error")
    assert "good.go:Works" in graph.G
    assert graph.G.nodes["good.go:Works"]["language"] == "go"


def test_c_graph_resolves_local_include_and_direct_cross_file_call(tmp_path):
    (tmp_path / "util.h").write_text(
        textwrap.dedent(
            """\
            struct Counter { int value; };
            int add_one(int value);
            """
        )
    )
    (tmp_path / "util.c").write_text(
        textwrap.dedent(
            """\
            #include "util.h"

            int add_one(int value) {
                return value + 1;
            }
            """
        )
    )
    (tmp_path / "main.c").write_text(
        textwrap.dedent(
            """\
            #include "util.h"

            int main(void) {
                return add_one(1);
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert graph.G.nodes["main.c"]["language"] == "c"
    assert graph.G.nodes["util.h"]["language"] == "c"
    assert graph.G.nodes["util.h:Counter"]["type"] == NodeType.CLASS
    assert graph.G.nodes["util.c:add_one"]["type"] == NodeType.FUNCTION
    assert graph.G.nodes["main.c:main"]["type"] == NodeType.FUNCTION

    import_edges = _edge_attrs(graph, "main.c", "util.h", EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is True
    assert import_edges[0]["include_style"] == "quote"
    assert _edge_attrs(graph, "main.c:main", "util.c:add_one", EdgeType.INVOKES)


def test_c_system_include_remains_unresolved_placeholder(tmp_path):
    (tmp_path / "main.c").write_text(
        textwrap.dedent(
            """\
            #include <stdio.h>

            int main(void) {
                return 0;
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    import_id = "main.c:import:1"
    assert graph.G.nodes[import_id]["type"] == NodeType.IMPORT
    assert graph.G.nodes[import_id]["language"] == "c"
    assert graph.G.nodes[import_id]["resolved"] is False
    assert graph.G.nodes[import_id]["confidence"] == "unresolved"
    assert graph.G.nodes[import_id]["heuristic"] is True

    import_edges = _edge_attrs(graph, "main.c", import_id, EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is False
    assert import_edges[0]["include_style"] == "angle"


def test_cpp_graph_resolves_class_methods_constructor_and_static_call(tmp_path):
    (tmp_path / "model.hpp").write_text(
        textwrap.dedent(
            """\
            class Widget {
            public:
                int value() const { return 1; }
            };
            """
        )
    )
    (tmp_path / "model.cpp").write_text(
        textwrap.dedent(
            """\
            #include "model.hpp"

            int Widget::make() {
                Widget* widget = new Widget();
                return 0;
            }
            """
        )
    )
    (tmp_path / "main.cpp").write_text(
        textwrap.dedent(
            """\
            #include "model.hpp"

            int run() {
                return Widget::make();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert graph.G.nodes["model.hpp"]["language"] == "cpp"
    assert graph.G.nodes["model.hpp:Widget"]["type"] == NodeType.CLASS
    assert graph.G.nodes["model.hpp:Widget.value"]["type"] == NodeType.METHOD
    assert graph.G.nodes["model.cpp:Widget.make"]["type"] == NodeType.METHOD
    assert graph.G.nodes["main.cpp:run"]["type"] == NodeType.FUNCTION

    assert _edge_attrs(graph, "main.cpp", "model.hpp", EdgeType.IMPORTS)
    assert _edge_attrs(graph, "model.cpp", "model.hpp", EdgeType.IMPORTS)
    assert _edge_attrs(graph, "model.cpp:Widget.make", "model.hpp:Widget", EdgeType.INVOKES)
    assert _edge_attrs(graph, "main.cpp:run", "model.cpp:Widget.make", EdgeType.INVOKES)


def test_c_syntax_error_metadata_does_not_abort_parsing(tmp_path):
    (tmp_path / "bad.c").write_text("int broken(\n")
    (tmp_path / "good.c").write_text(
        textwrap.dedent(
            """\
            int works(void) {
                return 0;
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert graph.G.nodes["bad.c"]["language"] == "c"
    assert graph.G.nodes["bad.c"].get("syntax_error")
    assert "good.c:works" in graph.G
    assert graph.G.nodes["good.c:works"]["language"] == "c"


def test_rust_graph_structure_trait_impl_and_containment(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        textwrap.dedent(
            """\
            pub trait Provider {
                fn get(&self) -> String;
            }

            pub struct Client;

            pub enum Status {
                Ok,
            }

            impl Provider for Client {
                fn get(&self) -> String {
                    String::new()
                }
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    file_id = "src/lib.rs"
    trait_id = f"{file_id}:Provider"
    struct_id = f"{file_id}:Client"
    enum_id = f"{file_id}:Status"
    trait_method_id = f"{file_id}:Provider.get"
    impl_method_id = f"{file_id}:Client.get"

    assert graph.G.nodes[file_id]["language"] == "rust"
    assert graph.G.nodes[struct_id]["type"] == NodeType.CLASS
    assert graph.G.nodes[struct_id]["unit_type"] == "struct"
    assert graph.G.nodes[enum_id]["type"] == NodeType.CLASS
    assert graph.G.nodes[enum_id]["unit_type"] == "enum"
    assert graph.G.nodes[trait_id]["type"] == NodeType.INTERFACE
    assert graph.G.nodes[trait_id]["unit_type"] == "trait"
    assert graph.G.nodes[trait_method_id]["type"] == NodeType.METHOD
    assert graph.G.nodes[impl_method_id]["type"] == NodeType.METHOD
    assert _edge_attrs(graph, trait_id, trait_method_id, EdgeType.CONTAINS)
    assert _edge_attrs(graph, struct_id, impl_method_id, EdgeType.CONTAINS)
    assert _edge_attrs(graph, struct_id, trait_id, EdgeType.INHERITS)


def test_rust_mod_decl_resolves_to_sibling_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        textwrap.dedent(
            """\
            mod error;

            pub fn run() {
                crate::error::make_error();
            }
            """
        )
    )
    (src / "error.rs").write_text("pub fn make_error() {}\n")

    graph = _parse_repo(tmp_path)

    import_edges = _edge_attrs(graph, "src/lib.rs", "src/error.rs", EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is True
    assert import_edges[0]["import_kind"] == "rust_mod_decl"
    assert _edge_attrs(graph, "src/lib.rs:run", "src/error.rs:make_error", EdgeType.INVOKES)


def test_rust_crate_use_resolves_to_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("pub mod a;\npub mod b;\n")
    (src / "a.rs").write_text(
        textwrap.dedent(
            """\
            use crate::b::Worker;

            pub fn make() {
                crate::b::build();
            }
            """
        )
    )
    (src / "b.rs").write_text(
        textwrap.dedent(
            """\
            pub struct Worker;

            pub fn build() {}
            """
        )
    )

    graph = _parse_repo(tmp_path)

    import_edges = _edge_attrs(graph, "src/a.rs", "src/b.rs", EdgeType.IMPORTS)
    assert import_edges
    assert import_edges[0]["resolved"] is True
    assert import_edges[0]["import_kind"] == "rust_use"
    assert _edge_attrs(graph, "src/a.rs:make", "src/b.rs:build", EdgeType.INVOKES)


def test_rust_grouped_use_import_nodes_are_distinct(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("pub mod foo;\n")
    (src / "foo.rs").write_text("pub struct A;\npub struct B;\n")
    (src / "app.rs").write_text("use crate::foo::{A, B};\n")

    graph = _parse_repo(tmp_path)

    assert "src/app.rs:import:1:1" in graph.G
    assert "src/app.rs:import:1:2" in graph.G
    assert graph.G.nodes["src/app.rs:import:1:1"]["import_module"] == "crate::foo::A"
    assert graph.G.nodes["src/app.rs:import:1:2"]["import_module"] == "crate::foo::B"
    assert _edge_attrs(graph, "src/app.rs", "src/foo.rs", EdgeType.IMPORTS)


def test_rust_direct_invoke_resolves_within_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.rs").write_text(
        textwrap.dedent(
            """\
            fn helper() {}

            fn caller() {
                helper();
            }
            """
        )
    )

    graph = _parse_repo(tmp_path)

    assert _edge_attrs(graph, "src/main.rs:caller", "src/main.rs:helper", EdgeType.INVOKES)
