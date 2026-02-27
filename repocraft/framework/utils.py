import re
from typing import List

from zerorepo.rpg_gen.base.unit import ParsedFile


def extract_python_blocks(text):
    """Extract Python code blocks from markdown-formatted text."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def _extract_import_lines(code: str) -> List[str]:
    """Extract import lines from Python code, handling multi-line imports."""
    lines = code.splitlines()
    imports = []

    in_import_block = False
    current_block = []
    open_parens = 0

    for line in lines:
        stripped = line

        if not in_import_block and (stripped.startswith("import ") or stripped.startswith("from ")):
            current_block = [line]
            open_parens = line.count("(") - line.count(")")
            if open_parens > 0:
                in_import_block = True
            else:
                imports.append(line)

        elif in_import_block:
            current_block.append(line)
            open_parens += line.count("(") - line.count(")")
            if open_parens <= 0:
                imports.append("\n".join(current_block))
                current_block = []
                in_import_block = False

    return imports


def extract_imports_by_line(code: str, file_path: str = ""):
    """Extract import lines using ParsedFile AST parsing, with fallback to regex."""
    file_path = file_path if file_path else "temp.py"
    code_units = ParsedFile(code=code, file_path=file_path).units

    if code_units:
        imports_and_assignments = [
            code_unit for code_unit in code_units
            if code_unit.unit_type in ["import", "assignment"]
        ]
        imports_lines = [code_unit.unparse() for code_unit in imports_and_assignments]
    else:
        imports_lines = _extract_import_lines(code)

    return imports_lines
