from . import CodeUnit, ParsedFile
from typing import List, Dict, Optional
import ast
from collections import defaultdict
import logging 
from zerorepo.utils.envs import CODE_OMITE

def merge_codeunits(
    units: List[CodeUnit],
    parsed_files: Dict[str, ParsedFile],
    keep_top_imports: bool = False,
    keep_top_assignments: bool = False
) -> List[CodeUnit]:
    from collections import defaultdict

    class_selected = set()         # (file_path, class_name)
    methods_selected = defaultdict(list)  # (file_path, class_name) → [CodeUnit]
    top_level_units = []

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

    result = []

    for (fp, cls_name) in class_selected:
        parsed = parsed_files.get(fp)
        if not parsed:
            continue
        cls_unit = next((u for u in parsed.units if u.unit_type == "class" and u.name == cls_name), None)
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
            cls_unit = next((u for u in parsed.units if u.unit_type == "class" and u.name == cls_name), None)
            if cls_unit:
                result.append(cls_unit)
        else:
            result.extend(selected_methods)

    result.extend(top_level_units)

    return list({(u.unit_type, u.name, u.file_path, u.parent): u for u in result}.values())

def class_ast_to_header_str(node: ast.ClassDef) -> str:
    base_names = [ast.unparse(base) for base in node.bases]
    base_str = f"({', '.join(base_names)})" if base_names else ""
    return f"class {node.name}{base_str}:"

class CodeSnippetBuilder:
    def __init__(self, file_code_map: Dict[str, str], parsed_files: Dict[str, ParsedFile]):
        self.file_code_map = file_code_map
        self.parsed_files = parsed_files

    def ensure_class_headers_for_partial_methods(self, units: List[CodeUnit]) -> List[CodeUnit]:
        result = list(units)
        seen = {(u.unit_type, u.name, u.file_path, u.parent) for u in result}
        methods_by_class = defaultdict(list)

        for u in units:
            if u.unit_type == "method":
                methods_by_class[(u.file_path, u.parent)].append(u)

        for (fp, cls_name), method_list in methods_by_class.items():
            has_class = any(u.unit_type == "class" and u.name == cls_name and u.file_path == fp for u in result)
            if not has_class:
                parsed = self.parsed_files.get(fp)
                if not parsed:
                    continue
                cls_unit = next((u for u in parsed.units if u.unit_type == "class" and u.name == cls_name), None)
                if cls_unit and (cls_unit.unit_type, cls_unit.name, cls_unit.file_path, cls_unit.parent) not in seen:
                    result.append(cls_unit)
        return result

    def generate_code_snippet(
        self,
        source_code: str,
        units: List["CodeUnit"],
        keep_imports: bool = True,
        keep_assignments: bool = True,
        with_lineno: bool = False,
        MIN_OMIT_GAP: int = 10,
        KEEP_SMALL_GAP: int = 5
    ) -> str:
        src_lines = source_code.splitlines()
        n = len(src_lines)
        keep = [False] * n

        # ---------- 1) CodeUnit 核心行 ----------
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
                        None
                    )
                    if cls_unit and cls_unit.lineno:
                        keep[cls_unit.lineno - 1] = True

        # ---------- 2) import / assignment ----------
        if keep_imports or keep_assignments:
            tree = ast.parse(source_code)
            for node in tree.body:
                if keep_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
                    keep[node.lineno - 1] = True
                if keep_assignments and isinstance(node, ast.Assign):
                    keep[node.lineno - 1] = True

        # ---------- 3) Only add one layer of blank lines near the "core lines" ----------
        core_idx = {i for i, k in enumerate(keep) if k}         
        for i in list(core_idx):                              
            for j in (i - 1, i + 1):                           
                if 0 <= j < n and not src_lines[j].strip(): 
                    keep[j] = True                          

        # ---------- 4) Output + truncation ----------
        out, prev = [], None
        for idx, flag in enumerate(keep):
            if not flag:
                continue

            if prev is not None:
                gap = idx - prev - 1
                if gap >= MIN_OMIT_GAP:
                    prev_indent = len(src_lines[prev]) - len(src_lines[prev].lstrip()) if prev is not None else 0
                    next_indent = len(src_lines[idx])  - len(src_lines[idx].lstrip())
                    indent_len  = min(prev_indent, next_indent)
                    indent_str  = " " * indent_len
                    out.append(f"{indent_str}{CODE_OMITE}")   
                elif 0 < gap <= KEEP_SMALL_GAP:
                    for g in range(prev + 1, idx):
                        raw = src_lines[g]
                        out.append(f"{str(g+1).rjust(4)}| {raw}" if with_lineno else raw)

            raw = src_lines[idx]
            out.append(f"{str(idx+1).rjust(4)}| {raw}" if with_lineno else raw)
            prev = idx

        return "\n".join(out).rstrip()


    
    def build(
        self,
        merged: List["CodeUnit"],
        keep_imports: bool = True,
        keep_assignments: bool = True,
        with_lineno: bool = False,
        with_file_path: bool = True
    ) -> str:
        from collections import defaultdict

        
        grouped: Dict[str, List["CodeUnit"]] = defaultdict(list)
        for u in merged:
            if not isinstance(u, CodeUnit):
                logging.info(f"Non-CodeUnit object found: {u} (type: {type(u)})")
                continue
            grouped[u.file_path].append(u)

        sections = []
        for file_path, units in grouped.items():
            if not units:
                continue
            src = self.file_code_map[file_path]
            body = self.generate_code_snippet(
                source_code=src,
                units=units,
                keep_imports=keep_imports,
                keep_assignments=keep_assignments,
                with_lineno=with_lineno
            )
            if with_file_path:
                sections.append(f"```python\n## File Path: {file_path}\n\n{body}\n```")
            else:
                sections.append(f"```python\n## Tool Block\n\n{body}\n```")
        return "\n\n".join(sections)

    def build_file_map(
        self,
        grouped_units: Dict[str, List["CodeUnit"]],
        keep_imports: bool = True,
        keep_assignments: bool = True
    ) -> Dict[str, str]:
        file_map: Dict[str, str] = {}
        for path, units in grouped_units.items():
            if not units:
                continue
            src = self.file_code_map[path]
            file_map[path] = self.generate_code_snippet(
                source_code=src,
                units=units,
                keep_imports=keep_imports,
                keep_assignments=keep_assignments,
                with_lineno=False
            )
        return file_map