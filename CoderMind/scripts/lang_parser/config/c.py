from __future__ import annotations

from ..models import LanguageConfig


_C_TEST_GLOBS = (
    "*_test.c",
    "**/*_test.c",
    "test_*.c",
    "**/test_*.c",
    "tests/*.c",
    "tests/**/*.c",
    "test/*.c",
    "test/**/*.c",
    "**/tests/*.c",
    "**/tests/**/*.c",
    "**/test/*.c",
    "**/test/**/*.c",
)


C_CONFIG = LanguageConfig(
    name="c",
    display_name="C",
    extensions=(".c", ".h"),
    markdown_fence="c",
    source_globs=("*.c", "*.h", "**/*.c", "**/*.h"),
    test_globs=_C_TEST_GLOBS,
    tree_sitter_language="c",
    class_node_types=("struct_specifier",),
    function_node_types=("function_definition",),
    method_node_types=(),
    import_node_types=("preproc_include",),
    module_path_style="c",
    dependency_files=("CMakeLists.txt", "Makefile", "compile_commands.json"),
    entrypoint_candidates=("main.c", "src/main.c"),
)
