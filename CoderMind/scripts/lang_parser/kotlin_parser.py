"""Kotlin language parser — query-based adapter (architecture v2).

All Kotlin syntax knowledge lives in declarative tree-sitter query files:

* ``queries/kotlin/tags.scm``    — vendored verbatim from
  fwcd/tree-sitter-kotlin @ 1852ea17b7f60fb3f9d84e0b1555d56b46b39fb1
  (definitions and call references, community-maintained upstream)
* ``queries/kotlin/imports.scm`` — cmind extension closing the one known
  upstream gap (import captures + unnamed companion objects); candidate
  for upstreaming.

This adapter contains no node-type names and no declaration regexes. It
knows only capture names, declared in ``CONTRACT``. At load time every
contract capture is verified against the compiled queries; a missing
capture fails loudly instead of silently producing an empty graph.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseLanguageParser
from .config.kotlin import KOTLIN_CONFIG
from .extractors.fallback import dependency_from_import, delimiter_syntax_error, make_unit
from .models import LPDependency, LPFileResult
from .tree_sitter_backend import TreeSitterBackend

_QUERIES_DIR = Path(__file__).parent / "queries" / "kotlin"

# The capture contract: the ONLY syntax knowledge this adapter relies on.
# Derived from the LPFileResult contract (classes, functions/methods,
# type aliases, imports, invoke edges). Parent relationships are derived
# structurally from capture-node ancestry, not from extra captures.
CONTRACT: dict[str, str] = {
    "class": "definition.class",
    "function": "definition.function",
    "type_alias": "definition.type",
    "import": "definition.import",
    "invoke": "reference.call",
}

# Fallback name for unnamed companion objects (captured via imports.scm).
_COMPANION_FALLBACK = "companion"


class KotlinParser(BaseLanguageParser):
    language = "kotlin"

    def __init__(self) -> None:
        self.backend = TreeSitterBackend(KOTLIN_CONFIG.tree_sitter_language)
        self._compiled_queries: dict[str, object] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse_file(self, path: str, source: str) -> LPFileResult:
        parsed = self._parse(source)
        if parsed is None:
            language = KOTLIN_CONFIG.tree_sitter_language or "kotlin"
            raise RuntimeError(
                f"tree-sitter grammar for {language} unavailable: "
                f"{self.backend.load_error}"
            )
        tree_root = parsed.tree.root_node
        matches = self._collect_matches(tree_root)

        lines = source.splitlines()
        units: list[LPCodeUnit] = []
        dependencies: list[LPDependency] = []
        class_nodes: list[object] = []

        # --- classes / objects / companions ---------------------------
        # Class captures live in BOTH query files (tags.scm for named
        # declarations, imports.scm for unnamed companions), so iterate both.
        # Document order guarantees outer classes register before nested ones.
        for _pat, m in [*matches["tags"], *matches["imports"]]:
            if CONTRACT["class"] not in m:
                continue
            node = m[CONTRACT["class"]][0]
            name = self._match_name(m) or _COMPANION_FALLBACK
            parent = self._enclosing_class_name(node, class_nodes)
            units.append(
                make_unit(
                    name=name,
                    unit_type="class",
                    file_path=path,
                    parent=parent,
                    lines=lines,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=self.language,
                    node_type=node.type,
                )
            )
            class_nodes.append(node)

        # --- functions / methods --------------------------------------
        for _pat, m in matches["tags"]:
            if CONTRACT["function"] not in m:
                continue
            node = m[CONTRACT["function"]][0]
            name = self._match_name(m)
            parent = self._enclosing_class_name(node, class_nodes)
            units.append(
                make_unit(
                    name=name,
                    unit_type="method" if parent else "function",
                    file_path=path,
                    parent=parent,
                    lines=lines,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=self.language,
                    node_type=node.type,
                )
            )

        # --- type aliases ---------------------------------------------
        for _pat, m in matches["tags"]:
            if CONTRACT["type_alias"] not in m:
                continue
            node = m[CONTRACT["type_alias"]][0]
            units.append(
                make_unit(
                    name=self._match_name(m),
                    unit_type="typealias",
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=self.language,
                    node_type=node.type,
                )
            )

        # --- imports ----------------------------------------------------
        alias_by_node: dict[int, str] = {}
        for _pat, m in matches["imports"]:
            if CONTRACT["import"] not in m or "import.path" not in m:
                continue
            node = m[CONTRACT["import"]][0]
            import_path = m["import.path"][0].text.decode()
            if "import.alias" in m:
                alias_by_node[node.id] = m["import.alias"][0].text.decode()

        for _pat, m in matches["imports"]:
            if CONTRACT["import"] not in m or "import.path" not in m:
                continue
            node = m[CONTRACT["import"]][0]
            import_path = m["import.path"][0].text.decode()
            alias = alias_by_node.get(node.id)
            qualifier = alias or import_path.rsplit(".", 1)[-1]
            units.append(
                make_unit(
                    name=import_path,
                    unit_type="import",
                    file_path=path,
                    parent=None,
                    lines=lines,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    language=self.language,
                    node_type=node.type,
                    extra={
                        "module": import_path,
                        "import_path": import_path,
                        "alias": alias,
                        "qualifier": qualifier,
                    },
                )
            )
            dep = dependency_from_import(
                path=path,
                module=import_path,
                symbol=qualifier,
                line=node.start_point[0] + 1,
                language=self.language,
                import_kind="kotlin_import",
            )
            dep.extra.update({"alias": alias, "qualifier": qualifier})
            dependencies.append(dep)

        # --- invoke edges ------------------------------------------------
        dependencies.extend(self._extract_invokes(path, tree_root, matches, units))

        return LPFileResult(
            file_path=path,
            language=self.language,
            units=units,
            dependencies=dependencies,
            syntax_error=self._syntax_error(source),
        )

    def validate_syntax(self, path: str, source: str) -> tuple[bool, str | None]:
        syntax_error = self._syntax_error(source)
        return (syntax_error is None, syntax_error)

    # ------------------------------------------------------------------
    # Query loading + contract enforcement
    # ------------------------------------------------------------------

    def _load_queries(self) -> dict[str, object]:
        if self._compiled_queries is not None:
            return self._compiled_queries
        language = self.backend.get_language()
        if language is None:
            raise RuntimeError(
                "tree-sitter-kotlin grammar unavailable: "
                f"{self.backend.load_error}"
            )
        from tree_sitter import Query  # local import: runtime >= 0.24 API

        compiled: dict[str, object] = {}
        available: set[str] = set()
        for name in ("tags", "imports"):
            source = (_QUERIES_DIR / f"{name}.scm").read_text(encoding="utf-8")
            query = Query(language, source)
            compiled[name] = query
            available.update(
                query.capture_name(i) for i in range(query.capture_count)
            )
        missing = sorted(
            cap for cap in CONTRACT.values() if cap not in available
        )
        if missing:
            raise RuntimeError(
                "Kotlin query files do not satisfy the capture contract. "
                f"Missing captures: {missing}. "
                f"The vendored .scm files under {_QUERIES_DIR} are out of sync "
                "with the installed tree-sitter-kotlin grammar — update them "
                "(or the grammar pin) before use."
            )
        self._compiled_queries = compiled
        return compiled

    def _collect_matches(self, tree_root) -> dict[str, list[dict]]:
        from tree_sitter import QueryCursor

        queries = self._load_queries()
        return {
            "tags": QueryCursor(queries["tags"]).matches(tree_root),
            "imports": QueryCursor(queries["imports"]).matches(tree_root),
        }

    # ------------------------------------------------------------------
    # Structural helpers (capture-driven, no node-type knowledge)
    # ------------------------------------------------------------------

    @staticmethod
    def _match_name(match: dict) -> str | None:
        nodes = match.get("name")
        if not nodes:
            return None
        return nodes[0].text.decode().strip("`")

    def _enclosing_class_name(self, node, class_nodes: list) -> str | None:
        """Nearest captured ancestor that is a class-like definition."""
        class_ids = {id(c) for c in class_nodes}
        ancestor = node.parent
        while ancestor is not None:
            if id(ancestor) in class_ids:
                name_child = next(
                    (
                        c
                        for c in ancestor.children
                        if c.type in ("type_identifier", "simple_identifier")
                    ),
                    None,
                )
                if name_child is not None:
                    return name_child.text.decode().strip("`")
                return _COMPANION_FALLBACK
            ancestor = ancestor.parent
        return None

    def _source_reference_for_line(self, path: str, units: list, line: int) -> str:
        innermost = None
        for unit in units:
            if unit.unit_type == "import":
                continue
            if unit.line_start is None or unit.line_end is None:
                continue
            if unit.line_start <= line <= unit.line_end:
                if innermost is None or (
                    (unit.line_end - unit.line_start)
                    <= (innermost.line_end - innermost.line_start)
                ):
                    innermost = unit
        if innermost is not None and innermost.name:
            return f"{path}::{innermost.name}"
        return path

    def _extract_invokes(
        self,
        path: str,
        tree_root,
        matches: dict[str, list[dict]],
        units: list,
    ) -> list[LPDependency]:
        import_aliases: dict[str, str] = {}
        for unit in units:
            if unit.unit_type != "import":
                continue
            qualifier = (unit.extra or {}).get("qualifier")
            module = (unit.extra or {}).get("import_path")
            if qualifier and module:
                first_segment = qualifier.split(".")[0]
                import_aliases.setdefault(first_segment, module)

        dependencies: list[LPDependency] = []
        seen: set[tuple[str, int, str]] = set()
        for _pat, m in matches["tags"]:
            if CONTRACT["invoke"] not in m or "name" not in m:
                continue
            node = m[CONTRACT["invoke"]][0]
            name = m["name"][0].text.decode().strip("`")
            line = node.start_point[0] + 1
            key = (name, line, node.type)
            if key in seen:
                continue
            seen.add(key)
            source_ref = self._source_reference_for_line(path, units, line)
            destination = import_aliases.get(name.split(".")[0])
            dependencies.append(
                LPDependency(
                    src=source_ref,
                    dst=destination,
                    relation="invokes",
                    symbol=name,
                    line=line,
                    confidence="high",
                    extra={
                        "language": self.language,
                        "call_kind": "query_capture",
                        "node_type": node.type,
                    },
                )
            )
        return dependencies

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def _syntax_error(self, source: str) -> str | None:
        parsed = self.backend.parse(source)
        if parsed is not None:
            if self._tree_has_visible_error(parsed.tree.root_node):
                return "tree-sitter reported syntax errors"
            return None
        return delimiter_syntax_error(source)

    def _tree_has_visible_error(self, node) -> bool:
        if getattr(node, "is_error", False) or getattr(node, "is_missing", False):
            return True
        return any(self._tree_has_visible_error(child) for child in node.children)

    def _parse(self, source: str):
        return self.backend.parse(source)
