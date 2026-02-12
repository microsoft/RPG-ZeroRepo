import re
import difflib
import libcst as cst
import libcst.matchers as m


class CompressTransformer(cst.CSTTransformer):
    """
    Compress modules/classes/functions into a "structural skeleton":
    - Module level: by default keep ClassDef / FunctionDef; optionally keep constant assignments,
      module docstring, and import statements.
    - Class level: remove pure docstring lines inside the class; keep other members
      (methods/assignments, etc.). If you want more aggressive filtering, extend here.
    - Function level: optionally keep function docstring and local imports; replace the function
      body with "..." as a placeholder.
    """
    DESCRIPTION = str = "Replaces function body with ..."
    replacement_string = '"__FUNC_BODY_REPLACEMENT_STRING__"'

    def __init__(
        self,
        keep_constant: bool = True,
        keep_indent: bool = False,
        keep_docstring: bool = False,
        keep_imports: bool = False,
    ):
        self.keep_constant = keep_constant
        self.keep_indent = keep_indent
        self.keep_docstring = keep_docstring
        self.keep_imports = keep_imports

    # Check whether a statement is an import (including from ... import ...)
    def _is_import_stmt(self, stmt: cst.CSTNode) -> bool:
        if not m.matches(stmt, m.SimpleStatementLine()):
            return False
        # SimpleStatementLine.body may contain multiple small statements;
        # treat it as import if any is Import / ImportFrom
        return any(
            m.matches(s, m.Import()) or m.matches(s, m.ImportFrom())
            for s in getattr(stmt, "body", [])
        )

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        new_body = []
        for i, stmt in enumerate(updated_node.body):
            if m.matches(stmt, m.ClassDef()) or m.matches(stmt, m.FunctionDef()):
                new_body.append(stmt)

            elif (
                self.keep_constant
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Assign())
            ):
                new_body.append(stmt)

            elif self.keep_imports and self._is_import_stmt(stmt):
                new_body.append(stmt)

            elif (
                self.keep_docstring
                and i == 0
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            ):
                new_body.append(stmt)

        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        new_body = []
        for i, stmt in enumerate(updated_node.body.body):
            # Keep class-level docstring (if enabled)
            if (
                i == 0
                and self.keep_docstring
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            ):
                new_body.append(stmt)
            # Remove "pure docstring lines" inside the class body
            # (they may also appear outside i == 0); keep everything else
            elif not (
                m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            ):
                new_body.append(stmt)

        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        # Optionally keep function docstring
        docstring_stmt = None
        import_stmts = []

        for i, stmt in enumerate(updated_node.body.body):
            if (
                i == 0
                and self.keep_docstring
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            ):
                docstring_stmt = stmt
            else:
                # Optionally keep import statements inside the function body
                if self.keep_imports and self._is_import_stmt(stmt):
                    import_stmts.append(stmt)

        # Replace the function body with "..."
        replacement_expr = cst.Expr(value=cst.SimpleString(value=self.replacement_string))
        replacement_stmt = cst.SimpleStatementLine(body=[replacement_expr])

        if self.keep_indent:
            body = []
            if docstring_stmt:
                body.append(docstring_stmt)
            body.extend(import_stmts)   # put imports first
            body.append(replacement_stmt)
            return updated_node.with_changes(body=cst.IndentedBlock(body=body))

        # If not preserving indentation style, still return multiple statements
        new_body = []
        new_body.extend(import_stmts)
        new_body.append(replacement_stmt)
        return updated_node.with_changes(body=cst.IndentedBlock(tuple(new_body)))


class GlobalVariableVisitor(cst.CSTVisitor):
    """
    Collect the start/end positions of module-level Assign statements,
    used later to fold/compress very large constants.
    """
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.assigns = []

    def leave_Assign(self, original_node: cst.Assign) -> None:
        stmt = original_node
        start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
        end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
        self.assigns.append([stmt, start_pos, end_pos])


def remove_lines(raw_code: str, remove_line_intervals):
    """
    Remove lines in the given intervals and insert '...' at the start of each interval.
    remove_line_intervals: List[Tuple[start, end]], 1-based inclusive intervals.
    """
    new_code = ""
    for i, line in enumerate(raw_code.splitlines(), start=1):
        if not any(start <= i <= end for start, end in remove_line_intervals):
            new_code += line + "\n"
        if any(start == i for start, _ in remove_line_intervals):
            new_code += "...\n"
    return new_code


def compress_assign_stmts(raw_code: str, total_lines=30, prefix_lines=10, suffix_lines=10) -> str:
    """
    For very long (> total_lines) module-level Assign statements,
    keep the first and last few lines and fold the middle with '...'.
    """
    try:
        tree = cst.parse_module(raw_code)
    except Exception:
        return raw_code

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    remove_line_intervals = []
    for _, start, end in visitor.assigns:
        if end.line - start.line > total_lines:
            remove_line_intervals.append(
                (start.line + prefix_lines, end.line - suffix_lines)
            )

    return remove_lines(raw_code, remove_line_intervals)


def add_original_line_numbers(raw_code: str, skeleton_code: str) -> str:
    """
    Annotate each line of skeleton_code with the corresponding original line number,
    and display folded continuous ranges as:
        '  045..120 | ...'
    - Matched skeleton line -> '   123 | actual line'
    - Compressed/missing original line range -> '045..120 | ...'
    """
    orig = raw_code.splitlines()
    skel = skeleton_code.splitlines()

    sm = difflib.SequenceMatcher(None, orig, skel, autojunk=False)
    width = len(str(len(orig)))  # alignment width

    out_lines = []
    prev_orig_end = 0  # end+1 of the previous matched block (0-based)

    def append_gap(start_idx: int, end_idx: int):
        """Original code range [start_idx, end_idx) is folded into '...'."""
        if end_idx <= start_idx:
            return
        left = str(start_idx + 1).rjust(width)
        right = str(end_idx).rjust(width)
        out_lines.append(f"{left}..{right} | ...")

    for (i_orig, j_skel, n) in sm.get_matching_blocks():
        # 1) Gap before the matched block -> '...'
        append_gap(prev_orig_end, i_orig)

        # 2) Output line-by-line mapping for the matched block
        for k in range(n):
            raw_ln = i_orig + k + 1
            line = skel[j_skel + k]
            out_lines.append(f"{str(raw_ln).rjust(width)} | {line}")

        # 3) Update cursor
        prev_orig_end = i_orig + n

    # 4) Trailing gap
    append_gap(prev_orig_end, len(orig))

    return "\n".join(out_lines)


def get_skeleton(
    raw_code: str,
    keep_constant: bool = True,
    keep_indent: bool = False,
    compress_assign: bool = False,
    keep_docstring: bool = False,
    keep_imports: bool = False,
    total_lines: int = 100,
    prefix_lines: int = 50,
    suffix_lines: int = 50,
    line_number_mode: str = "none",  # options: "none" | "original" | "sequential"
) -> str:
    """
    Generate a "structural skeleton" version of the code.
    - keep_constant: keep short module-level constant assignments
    - keep_indent: whether to preserve indentation style when omitting function bodies
    - compress_assign: whether to fold very long module-level assignments
    - keep_docstring: keep module/class/function docstrings
    - keep_imports: keep import / from ... import ...
    - line_number_mode:
        - "none": no line numbers
        - "original": annotate with original line numbers and show folded ranges
        - "sequential": number skeleton lines sequentially from 1..N
    """
    try:
        tree = cst.parse_module(raw_code)
    except Exception:
        # If parsing fails, return the original text
        code = raw_code
    else:
        transformer = CompressTransformer(
            keep_constant=keep_constant,
            keep_indent=keep_indent,
            keep_docstring=keep_docstring,
            keep_imports=keep_imports,
        )
        modified_tree = tree.visit(transformer)
        code = modified_tree.code

    if compress_assign:
        code = compress_assign_stmts(
            code,
            total_lines=total_lines,
            prefix_lines=prefix_lines,
            suffix_lines=suffix_lines,
        )

    # Replace the function body placeholder string with "..."
    if keep_indent:
        code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
        code = code.replace(CompressTransformer.replacement_string, "...\n")
    else:
        pattern = f"\\n[ \\t]*{CompressTransformer.replacement_string}"
        replacement = "\n..."
        code = re.sub(pattern, replacement, code)

    # Line number annotation
    if line_number_mode == "original":
        # Align with original line numbers and show folded ranges
        return add_original_line_numbers(raw_code, code)
    elif line_number_mode == "sequential":
        lines = code.splitlines()
        width = len(str(len(lines)))
        return "\n".join(f"{str(i).rjust(width)} | {ln}" for i, ln in enumerate(lines, 1))

    return code


# ---------------------------
# Simple usage example / self-test entry
# ---------------------------
if __name__ == "__main__":
    sample = '''
"""this is a module docstring"""
import os
import sys

BIG_TABLE = {
    i: i * 2 for i in range(300)
}

class Foo:
    """class doc"""
    x = 1
    def __init__(self, v):
        self.v = v
    def bar(self):
        from math import sqrt
        return sqrt(self.v)

def baz(a, b):
    """func doc"""
    c = a + b
    return c
'''
    print("=== Original LOC:", len(sample.splitlines()))
    skel = get_skeleton(
        sample,
        keep_constant=True,
        keep_indent=True,
        keep_docstring=True,
        keep_imports=True,
        compress_assign=True,
        total_lines=20,
        prefix_lines=5,
        suffix_lines=5,
        line_number_mode="original",  # "none" / "original" / "sequential"
    )
    print(skel)