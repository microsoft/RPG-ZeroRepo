from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


_GRAMMAR_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "go": (("tree_sitter_go", "language"),),
    "typescript": (("tree_sitter_typescript", "language_typescript"),),
    "tsx": (("tree_sitter_typescript", "language_tsx"),),
    "javascript": (("tree_sitter_javascript", "language"),),
    "c": (("tree_sitter_c", "language"),),
    "cpp": (("tree_sitter_cpp", "language"),),
    "rust": (("tree_sitter_rust", "language"),),
}


@dataclass(frozen=True)
class TreeSitterParseResult:
    tree: Any
    source_bytes: bytes


class TreeSitterBackend:
    """Lazy wrapper around optional tree-sitter grammar packages."""

    def __init__(self, language_name: str | None):
        self.language_name = language_name
        self._language: Any | None = None
        self._parser: Any | None = None
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def is_available(self) -> bool:
        return self.get_parser() is not None

    def get_language(self) -> Any | None:
        if self._language is not None:
            return self._language
        if not self.language_name:
            self._load_error = "No tree-sitter language configured"
            return None

        candidates = _GRAMMAR_CANDIDATES.get(self.language_name, ())
        if not candidates:
            self._load_error = f"No grammar candidate configured for {self.language_name}"
            return None

        try:
            from tree_sitter import Language
        except Exception as exc:
            self._load_error = f"tree-sitter is unavailable: {exc}"
            return None

        errors: list[str] = []
        for module_name, factory_name in candidates:
            try:
                module = importlib.import_module(module_name)
                factory = getattr(module, factory_name)
                raw_language = factory()
                if isinstance(raw_language, Language):
                    self._language = raw_language
                else:
                    self._language = Language(raw_language)
                return self._language
            except Exception as exc:
                errors.append(f"{module_name}.{factory_name}: {exc}")

        self._load_error = "; ".join(errors) if errors else "No grammar candidates tried"
        return None

    def get_parser(self) -> Any | None:
        if self._parser is not None:
            return self._parser

        language = self.get_language()
        if language is None:
            return None

        try:
            from tree_sitter import Parser

            parser = Parser()
            if hasattr(parser, "set_language"):
                parser.set_language(language)
            else:
                parser.language = language
            self._parser = parser
            return self._parser
        except Exception as exc:
            self._load_error = f"tree-sitter parser setup failed: {exc}"
            return None

    def parse(self, source: str) -> TreeSitterParseResult | None:
        parser = self.get_parser()
        if parser is None:
            return None
        source_bytes = source.encode("utf-8", errors="replace")
        try:
            return TreeSitterParseResult(tree=parser.parse(source_bytes), source_bytes=source_bytes)
        except Exception as exc:
            self._load_error = f"tree-sitter parse failed: {exc}"
            return None

    def validate_syntax(self, source: str) -> tuple[bool, str | None] | None:
        parsed = self.parse(source)
        if parsed is None:
            return None
        root = getattr(parsed.tree, "root_node", None)
        if root is not None and getattr(root, "has_error", False):
            return False, "tree-sitter reported syntax errors"
        return True, None
