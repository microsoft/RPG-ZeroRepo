from __future__ import annotations

from ..models import LanguageConfig

PYTHON_CONFIG = LanguageConfig(
    name="python",
    display_name="Python",
    extensions=(".py",),
    markdown_fence="python",
    source_globs=("*.py", "**/*.py"),
    test_globs=(
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "tests/**/*.py",
        "test/*.py",
        "test/**/*.py",
        "testing/*.py",
        "testing/**/*.py",
        "**/test_*.py",
        "**/*_test.py",
        "**/tests/*.py",
        "**/tests/**/*.py",
        "**/test/*.py",
        "**/test/**/*.py",
        "**/testing/*.py",
        "**/testing/**/*.py",
    ),
    tree_sitter_language=None,
    class_node_types=("ClassDef",),
    function_node_types=("FunctionDef", "AsyncFunctionDef"),
    method_node_types=("FunctionDef", "AsyncFunctionDef"),
    import_node_types=("Import", "ImportFrom"),
    module_path_style="python",
    default_test_command=("uv", "run", "pytest"),
    dependency_files=("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"),
    entrypoint_candidates=("main.py", "app.py", "__main__.py"),
)
