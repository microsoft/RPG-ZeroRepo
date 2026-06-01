from __future__ import annotations

from ..models import LanguageConfig


_TYPESCRIPT_TEST_GLOBS = (
    "*.test.ts",
    "*.spec.ts",
    "*.test.tsx",
    "*.spec.tsx",
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/*.test.tsx",
    "**/*.spec.tsx",
    "tests/*.ts",
    "tests/**/*.ts",
    "tests/*.tsx",
    "tests/**/*.tsx",
    "test/*.ts",
    "test/**/*.ts",
    "test/*.tsx",
    "test/**/*.tsx",
    "__tests__/*.ts",
    "__tests__/**/*.ts",
    "__tests__/*.tsx",
    "__tests__/**/*.tsx",
    "**/tests/*.ts",
    "**/tests/**/*.ts",
    "**/tests/*.tsx",
    "**/tests/**/*.tsx",
    "**/test/*.ts",
    "**/test/**/*.ts",
    "**/test/*.tsx",
    "**/test/**/*.tsx",
    "**/__tests__/*.ts",
    "**/__tests__/**/*.ts",
    "**/__tests__/*.tsx",
    "**/__tests__/**/*.tsx",
)


TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    display_name="TypeScript",
    extensions=(".ts", ".tsx"),
    markdown_fence="typescript",
    source_globs=("*.ts", "*.tsx", "**/*.ts", "**/*.tsx"),
    test_globs=_TYPESCRIPT_TEST_GLOBS,
    tree_sitter_language="typescript",
    class_node_types=("class_declaration", "interface_declaration", "type_alias_declaration"),
    function_node_types=("function_declaration", "lexical_declaration", "variable_declaration"),
    method_node_types=("method_definition", "public_field_definition"),
    import_node_types=("import_statement", "import_clause"),
    module_path_style="node",
    default_test_command=("npm", "test"),
    dependency_files=("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "tsconfig.json"),
    entrypoint_candidates=("src/index.ts", "src/main.ts", "index.ts", "main.ts"),
)
