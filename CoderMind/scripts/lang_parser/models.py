from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    display_name: str
    extensions: tuple[str, ...]
    markdown_fence: str
    source_globs: tuple[str, ...]
    test_globs: tuple[str, ...]
    tree_sitter_language: str | None
    class_node_types: tuple[str, ...]
    function_node_types: tuple[str, ...]
    method_node_types: tuple[str, ...]
    import_node_types: tuple[str, ...]
    module_path_style: str
    default_test_command: tuple[str, ...] | None = None
    dependency_files: tuple[str, ...] = ()
    entrypoint_candidates: tuple[str, ...] = ()


@dataclass
class LPCodeUnit:
    name: str | None
    unit_type: str
    file_path: str
    parent: str | None
    line_start: int | None
    line_end: int | None
    code: str
    language: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LPDependency:
    src: str
    dst: str | None
    relation: str
    symbol: str | None
    line: int | None
    confidence: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LPFileResult:
    file_path: str
    language: str
    units: list[LPCodeUnit]
    dependencies: list[LPDependency]
    syntax_error: str | None = None


class NotSupported(Exception):
    pass
