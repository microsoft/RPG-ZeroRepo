"""Code Unit and Parsed File Module.

Provides AST-based code parsing and extraction capabilities:
- CodeUnit: Represents a single code element (class, function, import, etc.)
- ParsedFile: Parses a Python file and extracts CodeUnits
- ParsedWorkspace: Multi-file workspace of ParsedFile instances
- ParsedModule: Parse a code string into CodeUnits (no file_path)
- CodeSnippetBuilder: Build code snippets from CodeUnit selections
- compare_code_units: Semantic equality for CodeUnits

Source: ZeroRepo rpg_gen/base/unit/ (code_unit.py, snippt_builder.py)
"""

import ast
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Any, Union

__all__ = [
    "CodeUnit", "ParsedFile", "ParsedWorkspace", "ParsedModule",
    "CodeSnippetBuilder", "merge_codeunits",
    "class_ast_to_header_str", "compare_code_units",
]


class CodeUnit:
    """Represents a single code unit extracted from a Python file.
    
    A code unit can be:
    - import: An import statement
    - function: A top-level function definition
    - class: A class definition
    - method: A method inside a class
    - assignment: A variable assignment (top-level or class-level)
    """
    
    def __init__(
        self,
        name: Optional[str],
        node: Any,  # ast.AST or str
        unit_type: str,
        file_path: str = "",
        parent: Optional[str] = None,
        extra: Optional[Dict] = None
    ):
        self.name = name
        self.node = node
        self.unit_type = unit_type
        self.file_path = file_path
        self.parent = parent
        self.extra = extra if extra else {}
        
        # Extract metadata for functions/methods and classes
        if isinstance(self.node, ast.AST):
            if self.unit_type in {"function", "method"} and isinstance(
                self.node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                self.extra.update(self._extract_function_metadata(self.node))
            elif self.unit_type == "class" and isinstance(self.node, ast.ClassDef):
                self.extra.update(self._extract_class_metadata(self.node))
    
    @property
    def docstring(self) -> Optional[str]:
        """Get docstring from the code unit."""
        try:
            if isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                return ast.get_docstring(self.node, clean=True)
        except Exception:
            return None
        return None
    
    @property
    def lineno(self) -> Optional[int]:
        """Get starting line number."""
        if isinstance(self.node, ast.AST):
            return getattr(self.node, "lineno", None)
        # For non-AST nodes (units produced by lang_parser for Go/TS/...),
        # the line info lives in ``extra``.
        return self.extra.get("line_start")
    
    @property
    def end_lineno(self) -> Optional[int]:
        """Get ending line number."""
        if isinstance(self.node, ast.AST):
            return getattr(self.node, "end_lineno", None)
        return self.extra.get("line_end")
    
    @property
    def is_top_level(self) -> bool:
        """Check if this is a top-level unit (not inside a class)."""
        return self.unit_type in {"import", "assignment", "function"} and self.parent is None
    
    def key(self) -> Tuple[str, Optional[str], Optional[str]]:
        """Get unique key for this unit."""
        if self.unit_type == "import":
            code = ast.unparse(self.node).strip() if isinstance(self.node, ast.AST) else str(self.node)
            return (self.unit_type, code, None)
        return (self.unit_type, self.name, self.parent)
    
    def unparse(self) -> str:
        """Convert AST node back to source code."""
        if isinstance(self.node, ast.AST):
            try:
                return ast.unparse(self.node).strip()
            except Exception:
                logging.warning(f"Failed to unparse AST node for {self.name or '<anon>'}")
                return f"# [Unparseable AST node: {self.unit_type} {self.name or '<anon>'}]"
        elif isinstance(self.node, str):
            return self.node.strip()
        return f"# [Unparseable node: {self.unit_type} {self.name or '<anon>'}]"
    
    @property
    def is_unimplemented_base_class(self) -> bool:
        """Determine whether a class is an unimplemented base class.

        A class qualifies when it defines methods but every method body
        contains only ``pass``, ``...``, or a docstring.

        Source: ZeroRepo rpg_gen/base/unit/code_unit.py
        """
        if self.unit_type != "class":
            return False
        if not isinstance(self.node, ast.ClassDef):
            return False

        method_defs = [
            node for node in self.node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not method_defs:
            return False

        for method in method_defs:
            body = method.body
            if not body:
                continue

            idx = 0
            if (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                idx = 1

            remaining = body[idx:]
            for stmt in remaining:
                if not isinstance(stmt, (ast.Pass, ast.Expr)) or (
                    isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant)
                ):
                    return False

        return True

    def semantic_equals(self, other: "CodeUnit", ignore_docstring: bool = True) -> bool:
        """Compare two CodeUnits for semantic equivalence based on AST structure.

        Source: ZeroRepo rpg_gen/base/unit/code_unit.py
        """
        return compare_code_units(self, other, ignore_docstring=ignore_docstring)

    def _extract_function_metadata(self, func: ast.FunctionDef) -> Dict:
        """Extract metadata from function definition."""
        args = [arg.arg for arg in func.args.args]
        return_type = ast.unparse(func.returns).strip() if func.returns else None
        docstring = ast.get_docstring(func)
        return {
            "args": args,
            "return_type": return_type,
            "docstring": docstring,
        }
    
    def _extract_class_metadata(self, cls: ast.ClassDef) -> Dict:
        """Extract metadata from class definition (ZeroRepo compatible format)."""
        docstring = ast.get_docstring(cls)
        methods = []
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append({
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args],
                    "return_type": ast.unparse(item.returns).strip() if item.returns else None,
                    "docstring": ast.get_docstring(item),
                })
        return {
            "docstring": docstring,
            "methods": methods,
        }
    
    def count_lines(self, original: bool = True, return_code: bool = False):
        """Count and optionally return code lines for this CodeUnit.
        
        Parameters
        ----------
        original : bool, default True
            - True  → Physical line count (includes docstring/comments/empty lines)
            - False → Effective code lines (excludes docstring, comments, empty lines)

        return_code : bool, default False
            If True, also return the code string.

        Returns:
        -------
        int or tuple
            Line count, or (count, code_str) if return_code=True
        """
        import os
        import tokenize
        from io import StringIO

        # Get source lines
        if (
            self.lineno is not None
            and self.end_lineno is not None
            and self.file_path
            and os.path.isfile(self.file_path)
        ):
            with open(self.file_path, "r", encoding="utf-8") as f:
                src_lines = f.readlines()[self.lineno - 1 : self.end_lineno]
        else:
            src_lines = self.unparse().splitlines(keepends=True)

        if original:
            code_str = "".join(src_lines)
            return (len(src_lines), code_str) if return_code else len(src_lines)

        remove_mask = [False] * len(src_lines)

        # Filter docstrings
        def _mark_docstrings(node: ast.AST):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                ds = ast.get_docstring(node, clean=False)
                if ds:
                    ds_node = node.body[0]
                    start = ds_node.lineno - 1
                    end = getattr(ds_node, "end_lineno", None)
                    if end is None:
                        end = start + ds.count("\n")
                    else:
                        end -= 1
                    for i in range(start, end + 1):
                        if 0 <= i < len(remove_mask):
                            remove_mask[i] = True
            for child in ast.iter_child_nodes(node):
                _mark_docstrings(child)

        # Filter @doc(...) decorators
        def _mark_doc_decorators(node: ast.AST):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in getattr(node, "decorator_list", []):
                    if (
                        isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name)
                        and dec.func.id == "doc"
                    ):
                        start = dec.lineno - 1
                        end = getattr(dec, "end_lineno", start) - 1
                        for i in range(start, end + 1):
                            if 0 <= i < len(remove_mask):
                                remove_mask[i] = True
            for child in ast.iter_child_nodes(node):
                _mark_doc_decorators(child)

        try:
            tree = ast.parse("".join(src_lines))
            _mark_docstrings(tree)
            _mark_doc_decorators(tree)
        except Exception:
            pass

        # Filter comment lines
        try:
            for tok in tokenize.generate_tokens(StringIO("".join(src_lines)).readline):
                if tok.type == tokenize.COMMENT:
                    idx = tok.start[0] - 1
                    if not src_lines[idx][:tok.start[1]].strip():
                        remove_mask[idx] = True
        except Exception:
            pass

        # Filter empty lines
        for idx, line in enumerate(src_lines):
            if not line.strip():
                remove_mask[idx] = True

        # Output
        kept = [l for i, l in enumerate(src_lines) if not remove_mask[i]]
        code_str = "".join(kept)
        count = len(kept)
        return (count, code_str) if return_code else count
    
    def __repr__(self):
        parent_info = f" in class {self.parent}" if self.parent and self.unit_type == "method" else ""
        name_repr = self.name if self.name else "<anonymous>"
        line_info = ""
        if self.lineno:
            line_info = f" (line {self.lineno})"
        return f"{self.unit_type.upper()} {name_repr}{parent_info} in {self.file_path}{line_info}"
    
    __str__ = __repr__
    
    def __eq__(self, other):
        if not isinstance(other, CodeUnit):
            return False
        return self.key() == other.key() and self.file_path == other.file_path
    
    def __hash__(self):
        return hash((self.key(), self.file_path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (ZeroRepo compatible)."""
        return {
            "name": self.name,
            "unit_type": self.unit_type,
            "file_path": self.file_path,
            "parent": self.parent,
            "extra": self.extra,
            "code": self.unparse(),
            "lineno": self.lineno,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CodeUnit":
        """Create CodeUnit from dictionary (ZeroRepo compatible)."""
        code = data.get("code", "")
        node: Any = code  # Default to string
        
        try:
            parsed = ast.parse(code)
            if parsed.body:
                node = parsed.body[0]
                # Inject lineno if available
                if data.get("lineno"):
                    setattr(node, "lineno", data.get("lineno"))
        except Exception:
            pass  # Keep node as string if parsing fails
        
        return CodeUnit(
            name=data.get("name"),
            node=node,
            unit_type=data.get("unit_type", "unknown"),
            file_path=data.get("file_path", ""),
            parent=data.get("parent"),
            extra=data.get("extra", {})
        )


class ParsedFile:
    """Parses a source file and extracts CodeUnits.

    For Python files, uses the built-in ``ast`` module. For other
    supported source languages (Go, TypeScript/JavaScript,
    C/C++, Rust, ...), delegates to ``lang_parser`` and adapts the result
    into the same ``CodeUnit`` shape so downstream consumers don't need to
    branch on language.

    Handles syntax errors gracefully by storing the error and returning
    an empty units list.
    """

    def __init__(self, code: str, file_path: str):
        self.code = code
        self.file_path = file_path
        self.error: Optional[Exception] = None

        # Try the lang_parser path first; if it doesn't claim the file (i.e.
        # the file is not a registered non-Python source, or lang_parser is
        # unavailable in this environment), fall through to the original
        # Python ast path so existing behaviour is preserved exactly.
        lp_result = self._parse_with_language_parser()
        if lp_result is not None:
            file_result, tree, error = lp_result
            self.tree = tree
            self.error = error
            self.units: List[CodeUnit] = self._code_units_from_parser_result(file_result)
            return

        try:
            self.tree = ast.parse(code)
        except SyntaxError as e:
            self.error = e
            logging.error(f"SyntaxError parsing {file_path}: {e}")
            self.tree = ast.Module(body=[], type_ignores=[])

        self.units: List[CodeUnit] = self._extract_units()

    def _parse_with_language_parser(self):
        """Attempt to parse ``self.code`` via ``lang_parser``.

        Returns ``(file_result, tree, error)`` on success, or ``None`` if
        the file is not recognised by ``lang_parser`` (callers should then
        fall back to the original ast path). Never raises.
        """
        try:
            from lang_parser import get_parser_for_file
        except ImportError:
            return None

        parser = get_parser_for_file(self.file_path)
        if parser is None:
            return None
        # For Python, lang_parser delegates back to ast, but downstream code
        # (notably ``_extract_units``) relies on ``self.tree`` being the raw
        # ast.Module. Keep Python on the original path to avoid divergence.
        try:
            from lang_parser import detect_language
            if detect_language(self.file_path) == "python":
                return None
        except ImportError:
            return None

        if hasattr(parser, "parse_file_with_ast"):
            return parser.parse_file_with_ast(self.file_path, self.code)

        file_result = parser.parse_file(self.file_path, self.code)
        error = Exception(file_result.syntax_error) if file_result.syntax_error else None
        empty_tree = ast.Module(body=[], type_ignores=[])
        return file_result, empty_tree, error

    # Unit kinds that other languages use to express something the rest
    # of the encoder pipeline treats as "a class": Go struct / interface,
    # Rust struct / enum / trait, C/C++ struct / class. Normalising them
    # to ``"class"`` here lets semantic_parsing.py's class-vs-function
    # grouping (which is hard-coded to ``unit_type == "class"``) pick
    # them up without having to learn each parser's per-language taxonomy.
    # The original kind is preserved as ``extra["lp_kind"]`` so callers
    # that care (e.g. RPG rendering) can still recover it.
    _LP_CLASS_LIKE_KINDS = frozenset({
        "struct", "interface", "enum", "trait",
    })

    def _code_units_from_parser_result(self, file_result) -> List["CodeUnit"]:
        """Adapt ``LPFileResult`` units into the ``CodeUnit`` shape.

        ``unit_type == "file"`` entries are skipped (no Python-side equivalent).
        ``unit_type`` values listed in :attr:`_LP_CLASS_LIKE_KINDS` are
        rewritten to ``"class"`` (with the original kind kept in
        ``extra["lp_kind"]``) so downstream code that pivots on
        ``unit_type == "class"`` picks them up.
        The ast-node slot is populated from ``unit.extra['ast_node']`` when
        the parser provides one; otherwise the raw source slice is used so
        downstream code-snippet building still works.
        """
        units: List[CodeUnit] = []
        for unit in file_result.units:
            if unit.unit_type == "file":
                continue
            node = unit.extra.get("ast_node") or unit.code
            extra = dict(unit.extra)
            extra.setdefault("language", unit.language)
            extra.setdefault("line_start", unit.line_start)
            extra.setdefault("line_end", unit.line_end)
            normalised_type = unit.unit_type
            if normalised_type in self._LP_CLASS_LIKE_KINDS:
                extra.setdefault("lp_kind", unit.unit_type)
                normalised_type = "class"
            units.append(CodeUnit(
                unit.name, node, normalised_type, self.file_path,
                unit.parent, extra=extra,
            ))
        return units
    
    def _extract_units(self) -> List[CodeUnit]:
        """Extract all code units from the AST."""
        units = []
        
        if not isinstance(self.tree, ast.Module):
            return units
        
        for node in self.tree.body:
            # Import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = ast.unparse(node).strip()
                units.append(CodeUnit(name, node, "import", self.file_path))
            
            # Top-level function definitions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(CodeUnit(node.name, node, "function", self.file_path))
            
            # Class definitions
            elif isinstance(node, ast.ClassDef):
                units.append(CodeUnit(node.name, node, "class", self.file_path))
                
                # Extract methods and class-level assignments
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        units.append(CodeUnit(
                            sub_node.name, sub_node, "method",
                            self.file_path, parent=node.name
                        ))
                    elif isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
                        name = self._extract_assignment_name(sub_node)
                        units.append(CodeUnit(
                            name, sub_node, "assignment",
                            self.file_path, parent=node.name
                        ))
            
            # Top-level assignments
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                name = self._extract_assignment_name(node)
                units.append(CodeUnit(name, node, "assignment", self.file_path))
        
        return units
    
    def _extract_assignment_name(self, node) -> Optional[str]:
        """Extract variable name from assignment node."""
        if isinstance(node, ast.Assign):
            if node.targets and isinstance(node.targets[0], ast.Name):
                return node.targets[0].id
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                return node.target.id
        return None
    
    def has_error(self) -> bool:
        """Check if there was a parsing error."""
        return self.error is not None
    
    def get_unit_by_name(self, name: str) -> Optional[CodeUnit]:
        """Find a unit by name."""
        for unit in self.units:
            if unit.name == name:
                return unit
        return None
    
    def get_units_by_type(self, unit_type: str) -> List[CodeUnit]:
        """Get all units of a specific type."""
        return [u for u in self.units if u.unit_type == unit_type]


# ============================================================================
# ParsedWorkspace — multi-file workspace
# Source: ZeroRepo rpg_gen/base/unit/code_unit.py
# ============================================================================

class ParsedWorkspace:
    """Parse multiple files and provide a unified query interface."""

    def __init__(self, file_map: Dict[str, str]):
        self.files: Dict[str, ParsedFile] = {
            path: ParsedFile(code, path) for path, code in file_map.items()
        }

    def all_units(self) -> List[CodeUnit]:
        """Return all CodeUnits across every file."""
        return [unit for pf in self.files.values() for unit in pf.units]

    def find_function(self, name: str) -> Optional[CodeUnit]:
        """Find a top-level function by name across all files."""
        for unit in self.all_units():
            if unit.unit_type == "function" and unit.name == name:
                return unit
        return None


# ============================================================================
# ParsedModule — parse a code string without file path
# Source: ZeroRepo rpg_gen/base/unit/code_unit.py
# ============================================================================

class ParsedModule:
    """Parse a code string into CodeUnits (no file path context)."""

    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        self.units: List[CodeUnit] = self._extract_units()

    def _extract_units(self) -> List[CodeUnit]:
        units: List[CodeUnit] = []
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                units.append(CodeUnit(name=None, node=node, unit_type="import"))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(CodeUnit(name=node.name, node=node, unit_type="function"))

            elif isinstance(node, ast.ClassDef):
                units.append(CodeUnit(name=node.name, node=node, unit_type="class"))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        units.append(CodeUnit(
                            name=item.name, node=item,
                            unit_type="method", parent=node.name
                        ))
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        units.append(CodeUnit(
                            name=None, node=item,
                            unit_type="assignment", parent=node.name
                        ))

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                units.append(CodeUnit(name=None, node=node, unit_type="assignment"))

        return units

    def get_units_by_type(self, unit_type: str) -> List[CodeUnit]:
        return [u for u in self.units if u.unit_type == unit_type]

    def get_methods_of_class(self, class_name: str) -> List[CodeUnit]:
        return [u for u in self.units if u.unit_type == "method" and u.parent == class_name]

    def get_class(self, class_name: str) -> Optional[CodeUnit]:
        return next((u for u in self.units if u.unit_type == "class" and u.name == class_name), None)

    def get_function(self, func_name: str) -> Optional[CodeUnit]:
        return next((u for u in self.units if u.unit_type == "function" and u.name == func_name), None)

    def get_method(self, class_name: str, method_name: str) -> Optional[CodeUnit]:
        return next(
            (u for u in self.units
             if u.unit_type == "method" and u.parent == class_name and u.name == method_name),
            None,
        )


# ============================================================================
# CodeSnippetBuilder — build contextual code snippets from unit selections
# Source: ZeroRepo rpg_gen/base/unit/snippt_builder.py
# ============================================================================

# Code omission marker (matches ZeroRepo default)
CODE_OMIT = "\n# ...(code omitted)...\n"


def merge_codeunits(
    units: List[CodeUnit],
    parsed_files: Dict[str, "ParsedFile"],
    keep_top_imports: bool = False,
    keep_top_assignments: bool = False,
) -> List[CodeUnit]:
    """Merge a selection of CodeUnits, promoting methods to full classes when appropriate.

    Source: ZeroRepo rpg_gen/base/unit/snippt_builder.py
    """
    class_selected: set = set()  # (file_path, class_name)
    methods_selected: Dict[Tuple[str, Optional[str]], List[CodeUnit]] = defaultdict(list)
    top_level_units: List[CodeUnit] = []

    for u in units:
        if u.unit_type == "class":
            class_selected.add((u.file_path, u.name))
        elif u.unit_type == "method":
            methods_selected[(u.file_path, u.parent)].append(u)
        elif u.unit_type == "import" and keep_top_imports:
            top_level_units.append(u)
        elif u.unit_type == "assignment" and keep_top_assignments:
            top_level_units.append(u)
        elif u.unit_type == "function":
            top_level_units.append(u)

    result: List[CodeUnit] = []

    for (fp, cls_name) in class_selected:
        parsed = parsed_files.get(fp)
        if not parsed:
            continue
        cls_unit = next(
            (u for u in parsed.units if u.unit_type == "class" and u.name == cls_name), None
        )
        if cls_unit:
            result.append(cls_unit)

    for (fp, cls_name), selected_methods in methods_selected.items():
        if (fp, cls_name) in class_selected:
            continue

        parsed = parsed_files.get(fp)
        if not parsed:
            result.extend(selected_methods)
            continue

        defined_methods = {
            u.name for u in parsed.units
            if u.unit_type == "method" and u.parent == cls_name
        }

        selected_method_names = {u.name for u in selected_methods}

        if selected_method_names >= defined_methods:
            cls_unit = next(
                (u for u in parsed.units if u.unit_type == "class" and u.name == cls_name), None
            )
            if cls_unit:
                result.append(cls_unit)
        else:
            result.extend(selected_methods)

    result.extend(top_level_units)
    return list({(u.unit_type, u.name, u.file_path, u.parent): u for u in result}.values())


def class_ast_to_header_str(node: ast.ClassDef) -> str:
    """Render a class AST node as its header line (e.g. ``class Foo(Base):``)."""
    base_names = [ast.unparse(base) for base in node.bases]
    base_str = f"({', '.join(base_names)})" if base_names else ""
    return f"class {node.name}{base_str}:"


class CodeSnippetBuilder:
    """Build code snippets from file maps and CodeUnit selections.

    Source: ZeroRepo rpg_gen/base/unit/snippt_builder.py
    """

    def __init__(
        self,
        file_code_map: Dict[str, str],
        parsed_files: Dict[str, "ParsedFile"],
    ):
        self.file_code_map = file_code_map
        self.parsed_files = parsed_files

    def ensure_class_headers_for_partial_methods(
        self, units: List[CodeUnit]
    ) -> List[CodeUnit]:
        """Ensure every method has its class header in *units*."""
        result = list(units)
        seen = {(u.unit_type, u.name, u.file_path, u.parent) for u in result}
        methods_by_class: Dict[Tuple[str, Optional[str]], List[CodeUnit]] = defaultdict(list)

        for u in units:
            if u.unit_type == "method":
                methods_by_class[(u.file_path, u.parent)].append(u)

        for (fp, cls_name), _ in methods_by_class.items():
            has_class = any(
                u.unit_type == "class" and u.name == cls_name and u.file_path == fp
                for u in result
            )
            if not has_class:
                parsed = self.parsed_files.get(fp)
                if not parsed:
                    continue
                cls_unit = next(
                    (u for u in parsed.units if u.unit_type == "class" and u.name == cls_name),
                    None,
                )
                if cls_unit and (cls_unit.unit_type, cls_unit.name, cls_unit.file_path, cls_unit.parent) not in seen:
                    result.append(cls_unit)
        return result

    def _language_for_units(self, units: List[CodeUnit]) -> Optional[str]:
        """Best-effort language for a homogeneous-by-file unit list.

        First looks at ``unit.extra['language']`` (set by lang_parser); if
        absent, falls back to detecting from the first ``file_path``. Used
        to skip Python-only ast helpers when the snippet is e.g. Go / TS.
        """
        for unit in units:
            language = unit.extra.get("language")
            if language:
                return language
        for unit in units:
            if not unit.file_path:
                continue
            try:
                from lang_parser import detect_language
                return detect_language(unit.file_path)
            except ImportError:
                return None
        return None

    def generate_code_snippet(
        self,
        source_code: str,
        units: List[CodeUnit],
        keep_imports: bool = True,
        keep_assignments: bool = True,
        with_lineno: bool = False,
        MIN_OMIT_GAP: int = 10,
        KEEP_SMALL_GAP: int = 5,
    ) -> str:
        """Generate a contextual code snippet for a set of CodeUnits."""
        src_lines = source_code.splitlines()
        n = len(src_lines)
        keep = [False] * n

        # 1) Mark CodeUnit core lines
        for u in units:
            if u.lineno is None:
                continue
            s = max(u.lineno - 1, 0)
            e = min((u.end_lineno or u.lineno) - 1, n - 1)
            for i in range(s, e + 1):
                keep[i] = True
            if u.unit_type == "method" and u.parent:
                parsed = self.parsed_files.get(u.file_path)
                if parsed:
                    cls_unit = next(
                        (cu for cu in parsed.units
                         if cu.unit_type == "class" and cu.name == u.parent),
                        None,
                    )
                    if cls_unit and cls_unit.lineno:
                        keep[cls_unit.lineno - 1] = True

        # 2) Import / assignment lines (Python only — non-Python sources
        #    come from lang_parser which already populates the import
        #    unit's line range, so the import lines are kept via step 1).
        if keep_imports or keep_assignments:
            language = self._language_for_units(units)
            if language in (None, "python"):
                try:
                    tree = ast.parse(source_code)
                    for node in tree.body:
                        if keep_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
                            keep[node.lineno - 1] = True
                        if keep_assignments and isinstance(node, ast.Assign):
                            keep[node.lineno - 1] = True
                except SyntaxError:
                    pass

        # 3) Adjacent blank lines near core lines
        core_idx = {i for i, k in enumerate(keep) if k}
        for i in list(core_idx):
            for j in (i - 1, i + 1):
                if 0 <= j < n and not src_lines[j].strip():
                    keep[j] = True

        # 4) Output with truncation markers
        out: List[str] = []
        prev: Optional[int] = None
        for idx, flag in enumerate(keep):
            if not flag:
                continue

            if prev is not None:
                gap = idx - prev - 1
                if gap >= MIN_OMIT_GAP:
                    prev_indent = len(src_lines[prev]) - len(src_lines[prev].lstrip()) if prev is not None else 0
                    next_indent = len(src_lines[idx]) - len(src_lines[idx].lstrip())
                    indent_len = min(prev_indent, next_indent)
                    indent_str = " " * indent_len
                    out.append(f"{indent_str}{CODE_OMIT}")
                elif 0 < gap <= KEEP_SMALL_GAP:
                    for g in range(prev + 1, idx):
                        raw = src_lines[g]
                        out.append(f"{str(g + 1).rjust(4)}| {raw}" if with_lineno else raw)

            raw = src_lines[idx]
            out.append(f"{str(idx + 1).rjust(4)}| {raw}" if with_lineno else raw)
            prev = idx

        return "\n".join(out).rstrip()

    def build(
        self,
        merged: List[CodeUnit],
        keep_imports: bool = True,
        keep_assignments: bool = True,
        with_lineno: bool = False,
        with_file_path: bool = True,
    ) -> str:
        """Build a combined code snippet string from multiple units."""
        grouped: Dict[str, List[CodeUnit]] = defaultdict(list)
        for u in merged:
            if not isinstance(u, CodeUnit):
                logging.info(f"Non-CodeUnit object found: {u} (type: {type(u)})")
                continue
            grouped[u.file_path].append(u)

        sections: List[str] = []
        for file_path, units in grouped.items():
            if not units:
                continue
            src = self.file_code_map.get(file_path, "")
            if not src:
                continue
            body = self.generate_code_snippet(
                source_code=src,
                units=units,
                keep_imports=keep_imports,
                keep_assignments=keep_assignments,
                with_lineno=with_lineno,
            )
            try:
                from lang_parser import markdown_fence_for_path
                fence = markdown_fence_for_path(file_path)
            except ImportError:
                fence = "python"
            if with_file_path:
                sections.append(f"```{fence}\n## File Path: {file_path}\n\n{body}\n```")
            else:
                sections.append(f"```{fence}\n## Tool Block\n\n{body}\n```")
        return "\n\n".join(sections)

    def build_file_map(
        self,
        grouped_units: Dict[str, List[CodeUnit]],
        keep_imports: bool = True,
        keep_assignments: bool = True,
    ) -> Dict[str, str]:
        """Build per-file code snippet mapping."""
        file_map: Dict[str, str] = {}
        for path, units in grouped_units.items():
            if not units:
                continue
            src = self.file_code_map.get(path, "")
            if not src:
                continue
            file_map[path] = self.generate_code_snippet(
                source_code=src,
                units=units,
                keep_imports=keep_imports,
                keep_assignments=keep_assignments,
                with_lineno=False,
            )
        return file_map


# ============================================================================
# Semantic comparison helpers
# Source: ZeroRepo rpg_gen/base/unit/code_unit.py
# ============================================================================

def _is_docstring_node(stmt_node: ast.stmt) -> bool:
    """Check whether an AST statement node is a docstring expression."""
    if not isinstance(stmt_node, ast.Expr):
        return False
    val_node = stmt_node.value
    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
        return True
    if hasattr(val_node, "s") and isinstance(getattr(val_node, "s", None), str):
        return True
    return False


def _sort_class_body(body: list) -> list:
    """Sort class body statements for stable comparison."""
    def sort_key(stmt):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"func:{stmt.name}"
        elif isinstance(stmt, ast.Assign):
            return "assign"
        return str(type(stmt))
    return sorted(body, key=sort_key)


def _normalize_ast_for_comparison(node: ast.AST, ignore_docstring: bool) -> ast.AST:
    """Normalize AST for stable semantic comparison.

    Removes docstrings (if *ignore_docstring*) and sorts class bodies.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        new_body = list(getattr(node, "body", []))
        if ignore_docstring and new_body and _is_docstring_node(new_body[0]):
            new_body = new_body[1:]

        if not new_body and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pass_stmt = ast.Pass()
            ast.copy_location(pass_stmt, node)
            new_body = [pass_stmt]

        if isinstance(node, ast.ClassDef):
            new_body = _sort_class_body(new_body)

        if isinstance(node, ast.FunctionDef):
            new_node = ast.FunctionDef(
                name=node.name, args=node.args, body=new_body,
                decorator_list=node.decorator_list,
                returns=getattr(node, "returns", None),
                type_comment=getattr(node, "type_comment", None),
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            new_node = ast.AsyncFunctionDef(
                name=node.name, args=node.args, body=new_body,
                decorator_list=node.decorator_list,
                returns=getattr(node, "returns", None),
                type_comment=getattr(node, "type_comment", None),
            )
        elif isinstance(node, ast.ClassDef):
            new_node = ast.ClassDef(
                name=node.name, bases=node.bases, keywords=node.keywords,
                body=new_body, decorator_list=node.decorator_list,
            )
        elif isinstance(node, ast.Module):
            new_node = ast.Module(
                body=new_body,
                type_ignores=getattr(node, "type_ignores", []),
            )
        else:
            return node

        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node

    return node


def compare_code_units(
    unit1: CodeUnit,
    unit2: CodeUnit,
    ignore_docstring: bool = True,
) -> bool:
    """Compare two CodeUnits for semantic equivalence based on AST structure.

    Source: ZeroRepo rpg_gen/base/unit/code_unit.py
    """
    node1_is_ast = isinstance(unit1.node, ast.AST)
    node2_is_ast = isinstance(unit2.node, ast.AST)

    if not node1_is_ast and not node2_is_ast:
        return unit1.unparse().strip() == unit2.unparse().strip()
    if not node1_is_ast or not node2_is_ast:
        return False

    try:
        norm1 = _normalize_ast_for_comparison(unit1.node, ignore_docstring)
        norm2 = _normalize_ast_for_comparison(unit2.node, ignore_docstring)
        return ast.dump(norm1, annotate_fields=False) == ast.dump(norm2, annotate_fields=False)
    except Exception as e:
        logging.warning(f"[compare_code_units] Fallback due to error: {e}")
        try:
            return unit1.unparse().strip() == unit2.unparse().strip()
        except Exception:
            return False
