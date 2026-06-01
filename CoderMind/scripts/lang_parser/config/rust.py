from __future__ import annotations

from ..models import LanguageConfig


RUST_CONFIG = LanguageConfig(
    name="rust",
    display_name="Rust",
    extensions=(".rs",),
    markdown_fence="rust",
    source_globs=("*.rs", "**/*.rs"),
    test_globs=(
        "tests/*.rs",
        "tests/**/*.rs",
        "test/*.rs",
        "test/**/*.rs",
        "**/tests/*.rs",
        "**/tests/**/*.rs",
        "**/test/*.rs",
        "**/test/**/*.rs",
        "benches/*.rs",
        "benches/**/*.rs",
        "examples/*.rs",
        "examples/**/*.rs",
    ),
    tree_sitter_language="rust",
    class_node_types=("struct_item", "enum_item"),
    function_node_types=("function_item",),
    method_node_types=("function_item",),
    import_node_types=("use_declaration", "mod_item"),
    module_path_style="rust",
    default_test_command=("cargo", "test"),
    dependency_files=("Cargo.toml", "Cargo.lock"),
    entrypoint_candidates=("src/main.rs", "src/lib.rs"),
)
