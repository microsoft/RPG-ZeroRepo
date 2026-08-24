from __future__ import annotations

from ..models import LanguageConfig


KOTLIN_CONFIG = LanguageConfig(
    name="kotlin",
    display_name="Kotlin",
    extensions=(".kt", ".kts"),
    markdown_fence="kotlin",
    source_globs=("*.kt", "**/*.kt", "*.kts", "**/*.kts"),
    test_globs=(
        "**/src/test/**/*.kt",
        "**/*Test.kt",
        "**/*Tests.kt",
        "**/test/**/*.kt",
    ),
    tree_sitter_language="kotlin",
    class_node_types=("class_declaration", "object_declaration"),
    function_node_types=("function_declaration",),
    method_node_types=("function_declaration",),
    import_node_types=("import", "import_header"),
    module_path_style="go",
    dependency_files=("build.gradle.kts", "build.gradle", "settings.gradle.kts", "pom.xml"),
    entrypoint_candidates=("src/main/kotlin/Main.kt",),
)
