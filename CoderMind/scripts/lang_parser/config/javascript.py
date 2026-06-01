from __future__ import annotations

from ..models import LanguageConfig


_JAVASCRIPT_TEST_GLOBS = (
    "*.test.js",
    "*.spec.js",
    "*.test.jsx",
    "*.spec.jsx",
    "**/*.test.js",
    "**/*.spec.js",
    "**/*.test.jsx",
    "**/*.spec.jsx",
    "tests/*.js",
    "tests/**/*.js",
    "tests/*.jsx",
    "tests/**/*.jsx",
    "test/*.js",
    "test/**/*.js",
    "test/*.jsx",
    "test/**/*.jsx",
    "__tests__/*.js",
    "__tests__/**/*.js",
    "__tests__/*.jsx",
    "__tests__/**/*.jsx",
    "**/tests/*.js",
    "**/tests/**/*.js",
    "**/tests/*.jsx",
    "**/tests/**/*.jsx",
    "**/test/*.js",
    "**/test/**/*.js",
    "**/test/*.jsx",
    "**/test/**/*.jsx",
    "**/__tests__/*.js",
    "**/__tests__/**/*.js",
    "**/__tests__/*.jsx",
    "**/__tests__/**/*.jsx",
)


JAVASCRIPT_CONFIG = LanguageConfig(
    name="javascript",
    display_name="JavaScript",
    extensions=(".js", ".jsx"),
    markdown_fence="javascript",
    source_globs=("*.js", "*.jsx", "**/*.js", "**/*.jsx"),
    test_globs=_JAVASCRIPT_TEST_GLOBS,
    tree_sitter_language="javascript",
    class_node_types=("class_declaration",),
    function_node_types=("function_declaration", "lexical_declaration", "variable_declaration"),
    method_node_types=("method_definition", "public_field_definition"),
    import_node_types=("import_statement", "import_clause"),
    module_path_style="node",
    default_test_command=("npm", "test"),
    dependency_files=("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"),
    entrypoint_candidates=("src/index.js", "src/main.js", "index.js", "main.js"),
)
