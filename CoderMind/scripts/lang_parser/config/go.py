from __future__ import annotations

from ..models import LanguageConfig


GO_CONFIG = LanguageConfig(
    name="go",
    display_name="Go",
    extensions=(".go",),
    markdown_fence="go",
    source_globs=("*.go", "**/*.go"),
    test_globs=(
        "*_test.go",
        "**/*_test.go",
        "tests/*.go",
        "tests/**/*.go",
        "test/*.go",
        "test/**/*.go",
        "**/tests/*.go",
        "**/tests/**/*.go",
        "**/test/*.go",
        "**/test/**/*.go",
    ),
    tree_sitter_language="go",
    class_node_types=("type_declaration", "struct_type", "interface_type"),
    function_node_types=("function_declaration",),
    method_node_types=("method_declaration",),
    import_node_types=("import_declaration", "import_spec"),
    module_path_style="go",
    default_test_command=("go", "test", "./..."),
    dependency_files=("go.mod", "go.sum"),
    entrypoint_candidates=("main.go", "cmd/main.go"),
)
