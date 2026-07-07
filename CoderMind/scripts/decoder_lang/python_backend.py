"""Production :class:`LanguageBackend` implementation for Python.

This is the production Python backend used by the CoderMind decoder
pipeline. It centralizes Python-specific file layout, syntax probes,
code-unit discovery, signature formatting, import extraction, and
prompt hints behind the shared :class:`LanguageBackend` protocol.

Methods that are not wired into the decoder yet raise
:class:`NotImplementedError` so unsupported calls fail loudly instead
of silently producing incomplete language behaviour.
"""
from __future__ import annotations

import ast
import keyword
import logging
from pathlib import Path
from typing import Any

from .backend import ToolchainUnavailable
from .file_deps import FileDependencyEdge, resolved_python_edges
from .prompt_hints import PromptHints
from .project_tasks import ProjectTaskContext, ProjectTaskTemplates
from .unit_kind import classify_unit_kind
from .test_result import EnvHandle, TestRunResult

logger = logging.getLogger(__name__)


# Hyphen → underscore + invalid char strip used by
# ``sanitize_module_identifier``. Cheap; kept module-level so the
# regex (and its compiled state) is constructed once.
import re

_PY_IDENT_INVALID = re.compile(r"[^A-Za-z0-9_]")


# Placeholder markers shared by static completeness checks and backend
# placeholder detection.
_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "TODO",
    "PLACEHOLDER",
    "NOT IMPLEMENTED",
)


class PythonBackend:
    """:class:`LanguageBackend` for Python source.

    See :class:`decoder_lang.backend.LanguageBackend` for method
    contracts. Behaviour matches the pre-existing decoder's Python
    assumptions exactly so dropping this backend into a call site is
    safe by construction.
    """

    name = "python"
    display_name = "Python"
    file_extension = ".py"
    markdown_fence = "python"

    # ------------------------------------------------------------------
    # 1. File & package layout
    # ------------------------------------------------------------------

    def is_source_file(self, path: str) -> bool:
        """A ``*.py`` file that is not under a tests/ tree."""
        # Match the decoder's existing convention: caller is
        # responsible for tests/ filtering via :meth:`is_test_file`;
        # ``is_source_file`` only checks the extension so the union
        # ``is_source_file or is_test_file`` covers everything the
        # encoder considers Python.
        return path.endswith(".py")

    def is_test_file(self, path: str) -> bool:
        """Match common pytest conventions: ``tests/`` directory,
        ``test_*.py`` / ``*_test.py`` file names."""
        normalised = path.replace("\\", "/")
        if "/tests/" in f"/{normalised}/" or normalised.startswith("tests/"):
            return True
        basename = normalised.rsplit("/", 1)[-1]
        return basename.startswith("test_") or basename.endswith("_test.py")

    def file_dependency_edges(
        self,
        files: list[str],
        file_sources: dict[str, str],
    ) -> list[FileDependencyEdge]:
        return resolved_python_edges(files, file_sources)

    def package_marker_filename(self) -> str | None:
        return "__init__.py"

    def package_marker_content(self, pkg_path: str) -> str:
        # The pre-existing skeleton emits an empty package marker; the
        # ``pkg_path`` argument is accepted so future backends (e.g.
        # Go ``doc.go``) can include the package name in the body.
        return ""

    def is_valid_module_identifier(self, segment: str) -> bool:
        """A non-empty Python identifier that is not a reserved keyword."""
        return bool(segment) and segment.isidentifier() and not keyword.iskeyword(segment)

    def sanitize_module_identifier(self, segment: str) -> str:
        """Map an arbitrary path segment to a legal Python identifier.

        Rules:
        * Replace any non-``[A-Za-z0-9_]`` char with ``_``.
        * Prepend ``_`` when the result starts with a digit.
        * Empty input becomes ``"_"`` (caller's job to avoid that).
        Idempotent.
        """
        if not segment:
            return "_"
        cleaned = _PY_IDENT_INVALID.sub("_", segment)
        if cleaned[:1].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned

    # ------------------------------------------------------------------
    # 2. Code structure
    # ------------------------------------------------------------------

    def has_placeholder(self, code: str, path: str = "<string>") -> bool:
        """Detect ``return "TODO..."`` style placeholder returns.

        Mirrors the placeholder-only check used by
        ``static_completeness_check``. Stub-body detection remains in
        ``static_checks.py`` because it needs statement-level context.
        Returns False on syntax errors so an unparseable file isn't
        misreported as containing a placeholder.
        """
        try:
            tree = ast.parse(code, filename=path)
        except (SyntaxError, ValueError):
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return):
                continue
            if not isinstance(node.value, ast.Constant):
                continue
            val = node.value.value
            if not isinstance(val, str):
                continue
            upper = val.upper()
            if any(marker in upper for marker in _PLACEHOLDER_MARKERS):
                return True
        return False

    def syntax_check(self, code: str, path: str = "<string>") -> tuple[bool, str | None]:
        """Quick Python syntax probe via :func:`ast.parse`."""
        try:
            ast.parse(code, filename=path)
        except SyntaxError as exc:
            return False, f"{exc.__class__.__name__}: {exc}"
        return True, None

    def list_code_units(
        self,
        code: str,
        path: str = "<string>",
    ) -> list[Any]:
        """Walk the full AST and return every class / function / method
        declaration as :class:`LPCodeUnit`. The result includes nested
        declarations because ``func_design`` consumers expect a flat
        view of every code unit in the source.

        The raw ``ast`` node is preserved in
        ``unit.extra["ast_node"]`` so callers that need fine-grained
        AST inspection (e.g. signature formatting, decorator lookup)
        can read it without re-parsing.
        """
        # Local imports to avoid forcing ``lang_parser`` as a top-level
        # dependency when this method is unused.
        from lang_parser import LPCodeUnit as _LPCodeUnit  # type: ignore

        try:
            tree = ast.parse(code, filename=path)
        except (SyntaxError, ValueError):
            return []

        units: list[Any] = []
        # Track parent class for ``method`` units. ``ast.walk`` yields
        # nodes in BFS order without parent info, so we precompute a
        # child→parent map first.
        parent_map: dict[ast.AST, ast.AST | None] = {tree: None}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                unit_type = "class"
                parent_name = None
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_node = parent_map.get(node)
                if isinstance(parent_node, ast.ClassDef):
                    unit_type = "method"
                    parent_name = parent_node.name
                else:
                    unit_type = "function"
                    parent_name = None
            else:
                continue
            units.append(
                _LPCodeUnit(
                    name=node.name,
                    unit_type=unit_type,
                    file_path=path,
                    parent=parent_name,
                    line_start=getattr(node, "lineno", None),
                    line_end=getattr(node, "end_lineno", getattr(node, "lineno", None)),
                    code=self._source_for_node(code, node),
                    language=self.name,
                    extra={"ast_node": node, "node_type": type(node).__name__},
                )
            )
        return units

    def format_signature(self, unit: Any) -> str:
        """Render a function / method ``LPCodeUnit`` into a one-line
        signature. Falls back to ``unit.name`` for non-function units
        or when the AST node is unavailable.

        Truncates parameter lists longer than four entries (``..., ...``)
        and renders the return annotation when present.
        """
        if unit is None:
            return ""
        name = getattr(unit, "name", None) or ""
        if getattr(unit, "unit_type", None) not in ("function", "method"):
            return name
        node = (getattr(unit, "extra", {}) or {}).get("ast_node")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return name

        params: list[str] = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            param_str = arg.arg
            if arg.annotation is not None:
                # ``ast.unparse`` is 3.9+; we already require 3.10+ for
                # the decoder but keep the guard for parity with the
                # original helper.
                type_str = ast.unparse(arg.annotation) if hasattr(ast, "unparse") else ""
                if type_str:
                    param_str = f"{arg.arg}: {type_str}"
            params.append(param_str)

        ret_str = ""
        if node.returns is not None:
            ret_type = ast.unparse(node.returns) if hasattr(ast, "unparse") else ""
            if ret_type:
                ret_str = f" -> {ret_type}"

        if len(params) > 4:
            params_str = ", ".join(params[:3]) + ", ..."
        else:
            params_str = ", ".join(params)
        return f"{name}({params_str}){ret_str}"

    def list_imports(
        self,
        code: str,
        path: str = "<string>",
    ) -> list[Any]:
        """Extract import statements as :class:`LPDependency` records.

        Mirrors the dependency shape produced by
        :class:`lang_parser.python_parser.PythonParser._dependencies_from_import`
        so call sites can swap between the two implementations
        without re-keying the dict literals.
        """
        from lang_parser import LPDependency as _LPDependency  # type: ignore

        try:
            tree = ast.parse(code, filename=path)
        except (SyntaxError, ValueError):
            return []

        deps: list[Any] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.append(_LPDependency(
                        src=path,
                        dst=alias.name,
                        relation="imports",
                        symbol=alias.asname or alias.name,
                        line=getattr(node, "lineno", None),
                        confidence="unresolved",
                        extra={"module": alias.name, "alias": alias.asname},
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = "." * (node.level or 0) + (node.module or "")
                for alias in node.names:
                    deps.append(_LPDependency(
                        src=path,
                        dst=module or None,
                        relation="imports",
                        symbol=alias.asname or alias.name,
                        line=getattr(node, "lineno", None),
                        confidence="unresolved",
                        extra={
                            "module": module,
                            "imported": alias.name,
                            "alias": alias.asname,
                        },
                    ))
        return deps

    def list_inheritance(
        self,
        code: str,
        path: str = "<string>",
    ) -> list[Any]:
        """Extract inheritance edges (``class Child(Base)``) as
        :class:`LPDependency` records with ``relation="inherits"``.

        ``src`` is the child class name and ``symbol``/``dst`` the base
        name, mirroring the ``inherits`` shape emitted by the
        tree-sitter backends so consumers treat every language alike.
        """
        from lang_parser import LPDependency as _LPDependency  # type: ignore

        try:
            tree = ast.parse(code, filename=path)
        except (SyntaxError, ValueError):
            return []

        def _base_name(node: ast.expr) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return node.attr
            return None

        deps: list[Any] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases or []:
                parent = _base_name(base)
                if not parent:
                    continue
                deps.append(_LPDependency(
                    src=node.name,
                    dst=parent,
                    relation="inherits",
                    symbol=parent,
                    line=getattr(node, "lineno", None),
                    confidence="high",
                    extra={"language": self.name, "child": node.name, "parent": parent},
                ))
        return deps

    def unit_kind(self, unit_name: str) -> str:
        return classify_unit_kind(unit_name)

    def is_callable_unit(self, unit_name: str) -> bool:
        return classify_unit_kind(unit_name) == "callable"

    def entry_point_path(self, module: str) -> str:
        return "main.py"

    def find_existing_entry(self, interfaces: dict) -> str | None:
        from .backend import default_find_existing_entry

        return default_find_existing_entry(self, interfaces)

    def entry_point_candidates(self) -> list[str]:
        return [self.entry_point_path("")]

    def prepare_test_env(self, env) -> None:
        return None

    def entry_run_command(self, repo_root: Path, entry: str) -> list[str] | None:
        if not (repo_root / entry).is_file():
            return None
        return ["python", entry, "--help"]

    def find_main_block_lineno(self, code: str) -> int | None:
        """Return the 1-based line number of the top-level
        ``if __name__ == "__main__":`` block, or ``None`` when no such
        block exists or the source has a syntax error.

        Python-only helper (not part of the :class:`LanguageBackend`
        Protocol). Other languages don't have this concept; callers
        feature-detect via ``getattr(backend, 'find_main_block_lineno',
        None)`` and skip the splice when absent.
        """
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return None
        for node in tree.body:
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                return getattr(node, "lineno", None)
        return None

    @staticmethod
    def _source_for_node(source: str, node: ast.AST) -> str:
        """Slice ``source`` to the lines covered by ``node`` (matches
        :meth:`lang_parser.python_parser.PythonParser._source_for_node`)."""
        line_start = getattr(node, "lineno", None)
        line_end = getattr(node, "end_lineno", line_start)
        if line_start is not None and line_end is not None:
            lines = source.splitlines()
            if 1 <= line_start <= line_end <= len(lines):
                return "\n".join(lines[line_start - 1:line_end])
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 3. Build / test environment — not wired into the decoder yet
    # ------------------------------------------------------------------

    def detect_env(self, repo_root: Path) -> EnvHandle | None:
        """Return an existing Python test environment when supported."""
        raise NotImplementedError(
            "PythonBackend.detect_env is not wired into the decoder; "
            "callers should keep using code_gen.test_runner.get_dev_python "
            "for now.",
        )

    def ensure_env(self, repo_root: Path) -> EnvHandle:
        """Always available on a host that's already running Python
        (the decoder itself), so this never raises
        :class:`ToolchainUnavailable` once implemented."""
        raise NotImplementedError(
            "PythonBackend.ensure_env is not wired into the decoder.",
        )

    def test_command(
        self,
        env: EnvHandle,
        selectors: list[str] | None = None,
    ) -> list[str]:
        """Return the command used to run Python tests when supported."""
        raise NotImplementedError(
            "PythonBackend.test_command is not wired into the decoder; "
            "callers should keep using code_gen.batch_prompts."
            "build_batch_pytest_cmd for now.",
        )

    def install_deps_command(
        self,
        env: EnvHandle,
        deps: list[str],
    ) -> list[str] | None:
        """Return a dependency-install command when supported."""
        raise NotImplementedError(
            "PythonBackend.install_deps_command is not wired into the decoder.",
        )

    # ------------------------------------------------------------------
    # 4. Test-output parsing — not wired into the decoder yet
    # ------------------------------------------------------------------

    def parse_test_output(self, raw: str, exit_code: int) -> TestRunResult:
        """Parse native Python test output when backend-driven tests run."""
        raise NotImplementedError(
            "PythonBackend.parse_test_output is not wired into the decoder; "
            "callers should keep using code_gen.test_output_parser."
            "analyze_test_output for now.",
        )

    # ------------------------------------------------------------------
    # 5. Prompt hints
    # ------------------------------------------------------------------

    _PROMPT_HINTS_SINGLETON: PromptHints | None = None

    def prompt_hints(self) -> PromptHints:
        """Return cached :class:`PromptHints` for Python. Built lazily
        on first call so the dataclass instance is reused across all
        prompt renders (callers may use it as a dict key safely)."""
        cached = PythonBackend._PROMPT_HINTS_SINGLETON
        if cached is not None:
            return cached
        hints = PromptHints(
            display_name=self.display_name,
            markdown_fence=self.markdown_fence,
            file_extension=self.file_extension,
            module_naming_rule=(
                "Use snake_case file and directory names; every package "
                "directory must contain an __init__.py."
            ),
            package_layout_example=(
                "src/\n"
                "  myproject/\n"
                "    __init__.py\n"
                "    cli.py\n"
                "    core.py\n"
                "tests/\n"
                "  test_cli.py\n"
            ),
            entrypoint_example="src/<project>/main.py",
            test_framework_name="pytest",
            style_directive=(
                "Write idiomatic Python 3.10+ code. Prefer dataclasses "
                "over plain dicts for structured records; use type "
                "annotations on every public function."
            ),
        )
        PythonBackend._PROMPT_HINTS_SINGLETON = hints
        return hints

    def project_task_templates(
        self,
        context: ProjectTaskContext,
    ) -> ProjectTaskTemplates | None:
        """Return None so plan_tasks can use its Python fallback."""
        return None


# Re-export for callers that want the exception without pulling in
# the whole backend module.
__all__ = ["PythonBackend", "ToolchainUnavailable"]
