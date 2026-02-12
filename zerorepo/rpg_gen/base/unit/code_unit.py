import ast, logging
from typing import List, Optional, Union, Dict, Tuple
    
class CodeUnit:
    def __init__(self, name, node, unit_type, file_path, parent=None,  extra={}):
        self.name = name # function, class, method 's name
        # function, class, variable, import, method
        self.node = node
        self.unit_type = unit_type
        self.file_path = file_path
        self.parent = parent
        self.extra = extra 

        if isinstance(self.node, ast.AST):
            if self.unit_type in {"function", "method"} and isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.extra.update(self.extract_function_metadata(self.node))
            elif self.unit_type == "class" and isinstance(self.node, ast.ClassDef):
                self.extra.update(self.extract_class_metadata(self.node))
    
    
    @property
    def docstring(self) -> Optional[str]:
        try:
            if isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                return ast.get_docstring(self.node, clean=True)
        except Exception as e:
            return None
        
        return None

    def key(self) -> Tuple[str, str | None, str | None]:
        if self.unit_type == "import":
            code = ast.unparse(self.node).strip() if isinstance(self.node, ast.AST) else str(self.node)
            return (self.unit_type, code, None)
        return (self.unit_type, self.name, self.parent)
    
    def __repr__(self):
        parent_info = f" in class {self.parent}" if self.parent and self.unit_type == "method" else ""
        name_repr = self.name if self.name else "<anonymous>"
        line_num_info = ""
        if hasattr(self.node, 'lineno') and isinstance(self.node, ast.AST): # Check if node is AST
            try: 
                line_num_info = f" (line {self.node.lineno})"
            except AttributeError:
                pass 
        return f"{self.unit_type.upper()} {name_repr}{parent_info} in {self.file_path}{line_num_info}"
    
    __str__ = __repr__ 
     
    def __eq__(self, other):
        if not isinstance(other, CodeUnit):
            return False
        # For __eq__, file_path comparison is important.
        # The original implementation was fine.
        return self.key() == other.key() and self.file_path == other.file_path

    def __hash__(self):
        # For hashing, ensure hash components match __eq__ components
        return hash((self.key(), self.file_path)) # Use key() for consistency with __eq__
    
    @property
    def is_top_level(self) -> bool:
        return self.unit_type in {"import", "assignment", "function"} and self.parent is None
    
    @property
    def lineno(self) -> Optional[int]:
        if isinstance(self.node, ast.AST):
            return getattr(self.node, "lineno", None)
        return None
    
    @property
    def end_lineno(self) -> Optional[int]:
        if isinstance(self.node, ast.AST):
            return getattr(self.node, "end_lineno", None)
        return None
    
    @property
    def is_unimplemented_base_class(self) -> bool:
        """
        Determine whether a class is an "unimplemented base class":
        - The class defines methods;
        - Each method body contains only pass / ... / docstring;
        """
        if self.unit_type != "class":
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


    def unparse(self) -> str:
        if isinstance(self.node, ast.AST):
            try:
                return ast.unparse(self.node).strip()
            except Exception: # Catch errors during unparsing, e.g. malformed AST snippet
                logging.warning(f"Failed to unparse AST node for {self.name or '<anon>'}. Type: {type(self.node)}")
                # Fallback representation for AST nodes that fail to unparse
                return f"# [Unparseable AST node: {self.unit_type} {self.name or '<anon>'}]"
        # If self.node is already a string (e.g. for some import representations)
        elif isinstance(self.node, str):
            return self.node.strip()
        return f"# [Unparseable node: {self.unit_type} {self.name or '<anon>'}]"

    def semantic_equals(self, other: "CodeUnit", ignore_docstring: bool = True) -> bool:
        return compare_code_units(self, other, ignore_docstring=ignore_docstring)

    
    # Metadata extraction functions
    def extract_function_metadata(self, func: ast.FunctionDef) -> Dict:
        args = [arg.arg for arg in func.args.args]

        return_type = ast.unparse(func.returns).strip() if func.returns else None
        docstring = ast.get_docstring(func)
        return {
            "args": args,
            "return_type": return_type,
            "docstring": docstring,
        }

    def extract_class_metadata(self, cls: ast.ClassDef) -> Dict:

        docstring = ast.get_docstring(cls)
        methods = []
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
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
        """
        Count and (optionally) return the code lines for the current CodeUnit.

        Parameters
        ----------
        original : bool, default True
            - True  -> Physical line count (including docstrings / comments / blank lines / @doc decorator blocks)
            - False -> Effective code lines
                    (keeps the class/function signature line, removes docstrings, comments, blank lines,
                    and the @doc(...) decorator along with its arguments)

        return_code : bool, default False
            If True, additionally return the corresponding code string:
            - original=True  -> original source
            - original=False -> normalized source (with non-effective lines removed)

        Returns
        -------
        int
            Line count
        str, optional
            Returned when `return_code` is True: the code snippet
        """
        import ast, os, tokenize
        from io import StringIO

        if (
            self.lineno is not None
            and self.end_lineno is not None
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
        
        try:
            for tok in tokenize.generate_tokens(StringIO("".join(src_lines)).readline):
                if tok.type == tokenize.COMMENT:
                    idx = tok.start[0] - 1
                    if not src_lines[idx][:tok.start[1]].strip():
                        remove_mask[idx] = True
        except Exception:
            pass

        for idx, line in enumerate(src_lines):
            if not line.strip():
                remove_mask[idx] = True

        kept = [l for i, l in enumerate(src_lines) if not remove_mask[i]]
        code_str = "".join(kept)
        count = len(kept)
        return (count, code_str) if return_code else count


    def to_dict(self):
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
    def from_dict(data: Dict) -> "CodeUnit":
        code = data.get("code", "")
        try:
            parsed = ast.parse(code)
            node = parsed.body[0] if parsed.body else ast.Expr()
            setattr(node, "lineno", data.get("lineno"))
        except Exception:
            node = code  # fallback to string if unparsing fails
            
        return CodeUnit(
            name=data["name"],
            node=data.get("code", ""),
            unit_type=data["unit_type"],
            file_path=data["file_path"],
            parent=data.get("parent"),
            extra=data.get("extra", {})
        )
        
        
        
class ParsedFile:
    def __init__(self, code: str, file_path: str):
        self.code = code
        self.file_path = file_path
        self.error = None 
        
        try:
            # Remove null bytes from the code before parsing
            if '\x00' in code:
                logging.warning(f"Null bytes found in {file_path}, removing them before parsing")
                code = code.replace('\x00', '')
            self.tree = ast.parse(code)
        except (SyntaxError, ValueError) as e:
            self.error = e
            logging.error(f"Error parsing {file_path}: {e}")
            self.tree = ast.Module(body=[], type_ignores=[])

        self.units: List[CodeUnit] = self._extract_units()

    # Control flow types whose bodies may contain nested function/class definitions
    _CONTROL_FLOW_TYPES = (ast.If, ast.Try, ast.With, ast.For, ast.While,
                           ast.AsyncWith, ast.AsyncFor)

    @staticmethod
    def _get_control_flow_bodies(node: ast.AST) -> list:
        """Extract all branch bodies from a control flow node."""
        bodies = []
        if isinstance(node, ast.If):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        elif isinstance(node, ast.Try):
            bodies.append(node.body)
            for handler in node.handlers:
                bodies.append(handler.body)
            if node.orelse:
                bodies.append(node.orelse)
            if node.finalbody:
                bodies.append(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            bodies.append(node.body)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        return bodies

    def _extract_from_control_flow(self, stmts: list, units: List[CodeUnit], seen_names: set):
        """Recursively extract CodeUnits from control flow blocks."""
        for node_item in stmts:
            if isinstance(node_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = ("function", node_item.name)
                if key not in seen_names:
                    seen_names.add(key)
                    units.append(CodeUnit(node_item.name, node_item, "function", self.file_path))
            elif isinstance(node_item, ast.ClassDef):
                key = ("class", node_item.name)
                if key not in seen_names:
                    seen_names.add(key)
                    units.append(CodeUnit(node_item.name, node_item, "class", self.file_path))
                    for sub_node in node_item.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            mkey = ("method", sub_node.name, node_item.name)
                            if mkey not in seen_names:
                                seen_names.add(mkey)
                                units.append(CodeUnit(sub_node.name, sub_node, "method", self.file_path, parent=node_item.name))
            # Recurse into nested control flow
            if isinstance(node_item, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node_item):
                    self._extract_from_control_flow(block, units, seen_names)

    def _extract_units(self) -> List[CodeUnit]:
        units = []
        if not isinstance(self.tree, ast.Module):
             return units

        # Track seen names to avoid duplicates from control flow extraction
        seen_names = set()

        for node_item in self.tree.body: # Renamed 'node' to 'node_item'
            if isinstance(node_item, (ast.Import, ast.ImportFrom)):
                name = ast.unparse(node_item).strip()  # 添加这一行
                units.append(CodeUnit(name, node_item, "import", self.file_path))
            elif isinstance(node_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen_names.add(("function", node_item.name))
                units.append(CodeUnit(node_item.name, node_item, "function", self.file_path))
            elif isinstance(node_item, ast.ClassDef):
                seen_names.add(("class", node_item.name))
                units.append(CodeUnit(node_item.name, node_item, "class", self.file_path))
                for sub_node in node_item.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        seen_names.add(("method", sub_node.name, node_item.name))
                        units.append(CodeUnit(sub_node.name, sub_node, "method", self.file_path, parent=node_item.name))
                    elif isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
                        name = None
                        if isinstance(sub_node, ast.Assign) and sub_node.targets and isinstance(sub_node.targets[0], ast.Name):
                            name = sub_node.targets[0].id
                        elif isinstance(sub_node, ast.AnnAssign) and isinstance(sub_node.target, ast.Name):
                            name = sub_node.target.id
                        units.append(CodeUnit(name, sub_node, "assignment", self.file_path, parent=node_item.name))
            elif isinstance(node_item, (ast.Assign, ast.AnnAssign)):
                name = None
                if isinstance(node_item, ast.Assign) and node_item.targets and isinstance(node_item.targets[0], ast.Name):
                    name = node_item.targets[0].id
                elif isinstance(node_item, ast.AnnAssign) and isinstance(node_item.target, ast.Name):
                    name = node_item.target.id
                units.append(CodeUnit(name, node_item, "assignment", self.file_path))

        # Extract function/class definitions from control flow blocks
        for node_item in self.tree.body:
            if isinstance(node_item, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node_item):
                    self._extract_from_control_flow(block, units, seen_names)

        return units



class ParsedWorkspace:
    def __init__(self, file_map: Dict[str, str]):
        self.files = {
            path: ParsedFile(code, path) for path, code in file_map.items()
        }

    def all_units(self) -> List[CodeUnit]:
        return [unit for pf in self.files.values() for unit in pf.units]

    def find_function(self, name: str) -> Optional[CodeUnit]:
        for unit in self.all_units():
            if unit.unit_type == "function" and unit.name == name:
                return unit
        return None



class ParsedModule:
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        self.units = self._extract_units()

    def _extract_units(self) -> List[CodeUnit]:
        units = []
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                units.append(CodeUnit(name=None, node=node, unit_type="import"))

            elif isinstance(node, ast.FunctionDef):
                units.append(CodeUnit(name=node.name, node=node, unit_type="function"))

            elif isinstance(node, ast.ClassDef):
                units.append(CodeUnit(name=node.name, node=node, unit_type="class"))
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        units.append(CodeUnit(name=item.name, node=item, unit_type="method", parent=node.name))
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        # 类变量
                        units.append(CodeUnit(name=None, node=item, unit_type="assignment", parent=node.name))

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
        return next((u for u in self.units if u.unit_type == "method" and u.parent == class_name and u.name == method_name), None)



def _is_docstring_node(stmt_node: ast.stmt) -> bool:
    if not isinstance(stmt_node, ast.Expr):
        return False
    val_node = stmt_node.value
    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
        return True
    if hasattr(val_node, 's') and isinstance(getattr(val_node, 's', None), str):
        return True
    return False

def _sort_class_body(body: list[ast.stmt]) -> list[ast.stmt]:
    def sort_key(stmt):
        if isinstance(stmt, ast.FunctionDef):
            return f"func:{stmt.name}"
        elif isinstance(stmt, ast.Assign):
            return f"assign"
        return str(type(stmt))
    return sorted(body, key=sort_key)

def _normalize_ast_for_comparison(node: ast.AST, ignore_docstring: bool) -> ast.AST:
    """
    Normalize AST to remove docstring (if flag) and sort class body for stable comparison.
    """
    import copy
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        new_body = node.body[:]
        if ignore_docstring and new_body and _is_docstring_node(new_body[0]):
            new_body = new_body[1:]

        # Ensure non-empty body (insert pass if needed)
        if not new_body and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            pass_stmt = ast.Pass()
            ast.copy_location(pass_stmt, node)
            new_body = [pass_stmt]

        # Sort class body if applicable
        if isinstance(node, ast.ClassDef):
            new_body = _sort_class_body(new_body)

        # Construct new node of same type
        if isinstance(node, ast.FunctionDef):
            new_node = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=new_body,
                decorator_list=node.decorator_list,
                returns=getattr(node, 'returns', None),
                type_comment=getattr(node, 'type_comment', None),
            )
        elif isinstance(node, ast.ClassDef):
            new_node = ast.ClassDef(
                name=node.name,
                bases=node.bases,
                keywords=node.keywords,
                body=new_body,
                decorator_list=node.decorator_list,
            )
        elif isinstance(node, ast.Module):
            new_node = ast.Module(
                body=new_body,
                type_ignores=getattr(node, 'type_ignores', []),
            )
        else:
            return node  # fallback

        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node

    return node

def compare_code_units(unit1: CodeUnit, unit2: CodeUnit, ignore_docstring: bool = True) -> bool:
    """
    Compares two CodeUnits for semantic equivalence based on AST structure.

    :param unit1: First CodeUnit
    :param unit2: Second CodeUnit
    :param ignore_docstring: If True, docstrings are ignored in comparison
    :return: True if semantically equivalent, else False
    """
    node1_is_ast = isinstance(unit1.node, ast.AST)
    node2_is_ast = isinstance(unit2.node, ast.AST)

    if not node1_is_ast and not node2_is_ast:
        return unit1.unparse().strip() == unit2.unparse().strip()
    if not node1_is_ast or not node2_is_ast:
        logging.debug(f"AST mismatch types: unit1.ast={node1_is_ast}, unit2.ast={node2_is_ast}")
        return False

    try:
        norm1 = _normalize_ast_for_comparison(unit1.node, ignore_docstring)
        norm2 = _normalize_ast_for_comparison(unit2.node, ignore_docstring)
        return ast.dump(norm1, annotate_fields=False) == ast.dump(norm2, annotate_fields=False)
    except Exception as e:
        logging.warning(f"[compare_code_units] Fallback due to error: {e}")
        try:
            return unit1.unparse().strip() == unit2.unparse().strip()
        except Exception as fallback_e:
            logging.error(f"[compare_code_units] Unparse fallback failed: {fallback_e}")
            return False