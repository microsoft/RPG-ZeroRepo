import re
import ast
from typing import List, Optional, Tuple
from collections import Counter
from pathlib import Path
from enum import Enum
from zerorepo.utils.file import normalize_path


class NodeType(str, Enum):
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    COMPONENT = "component"
    DATA = "data"
    INTERFACE = "interface"
    VARIABLE = "variable"
    IMPORT = "import"
    REPO = "repo"
    MODULE = "module"
    PACKAGE = "package"

    def __str__(self):
        return self.value


class EdgeType(str, Enum):
    
    COMPOSES = "composes"
    CONTAINS = "contains"
    INHERITS = "inherits"
    INVOKES = "invokes"
    IMPORTS = "imports"

    def __str__(self):
        return self.value


def is_test_file(nid):
    # input node id (e.g., 'tests/_core.py:test') and output whether it belongs to a test file
    file_path = nid.split(':')[0]
    word_list = re.split(r" |_|\/", file_path.lower())  # split by ' ', '_', and '/'
    return any([word.startswith('test') for word in word_list])



def get_node_range_robust(node: ast.AST, source: str) -> Tuple[int, int, int, int]:
    lines = source.splitlines()
    start_inclusive = _node_start_with_decorators(node)
    header_end_inclusive = getattr(node, "lineno", start_inclusive)

    body_end_inclusive = getattr(node, "end_lineno", None)
    if not isinstance(body_end_inclusive, int):
        body_end_inclusive = _node_end_by_walk(node, header_end_inclusive)

    first_body_ln = _first_body_lineno(node)
    base_indent = _indent_of_line(lines, first_body_ln) if isinstance(first_body_ln, int) else None

    expanded_end_inclusive = _expand_block_end_strict(lines, body_end_inclusive, base_indent)
    end_exclusive = expanded_end_inclusive + 1
    return start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive


def extract_source_by_lines(source: str, start_inclusive: int, end_inclusive: int) -> str:
    """
    Extract lines [start_inclusive, end_inclusive] (inclusive) from `source` using 1-based line numbers.
    Preserves original blank lines, comments, indentation, and newlines.
    """
    if start_inclusive is None or end_inclusive is None:
        return ""
    lines = source.splitlines(keepends=True)  # 保留换行符
    n = len(lines)
    s = max(1, start_inclusive)
    e = min(n, end_inclusive)
    if s > e:
        return ""
    return "".join(lines[s - 1:e]).strip()


# UNUSED
def infer_type_from_id(node_id: str) -> str:
    if ":" in node_id:
        _, qual = node_id.split(":", 1)
        return NodeType.METHOD if "." in qual else NodeType.CLASS
    else:
        return NodeType.FILE if node_id.endswith(".py") else NodeType.DIRECTORY


# UNUSED: it is more appropriate to implement this in the skeleton, since types can be obtained accurately there
def get_parent(node_id: str) -> Optional[Tuple[str, str]]:
    """
    Return (parent_id, parent_type); return None if there is no parent.
    Note: parent_type here is only inferred and may not be accurate.

    - "a/b.py:Class.method.inner" -> ("a/b.py:Class.method", NodeType.FUNCTION)
    - "a/b.py:Class.method"       -> ("a/b.py:Class", NodeType.CLASS)
    - "a/b.py:Class"              -> ("a/b.py", NodeType.FILE)
    - "a/b.py"                    -> ("a", NodeType.DIRECTORY)
    - "a"                         -> (".", NodeType.DIRECTORY)
    - "."                         -> None
    """
    nid = normalize_path(node_id)
    
    if nid == ".":
        return None
    elif ":" in nid:
        path_part, qual = nid.split(":", 1)
        parts = [p for p in qual.split(".") if p]
        if len(parts) <= 1:
            parent_id = path_part
            parent_type = NodeType.FILE
        else:
            parent_qual = ".".join(parts[:-1])
            parent_id = f"{path_part}:{parent_qual}"
           
            parent_type = NodeType.METHOD if "." in parent_qual else NodeType.CLASS
        return parent_id, parent_type
    else:
        parent_id = normalize_path(Path(nid).parent)
        return parent_id, NodeType.DIRECTORY
        
        
def _indent_of_line(lines, lineno: int) -> int:
    if lineno <= 0 or lineno > len(lines):
        return 0
    s = lines[lineno - 1]
    return len(s.expandtabs(8)) - len(s.lstrip().expandtabs(8))

def _is_blank_or_comment(lines, lineno: int) -> bool:
    if lineno <= 0 or lineno > len(lines):
        return True
    s = lines[lineno - 1].strip()
    return not s or s.startswith("#")

def _first_body_lineno(node: ast.AST) -> Optional[int]:
    body = getattr(node, "body", None)
    if not body:
        return None
    return getattr(body[0], "lineno", None)

def _node_start_with_decorators(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None)
    if decos:
        return min(getattr(d, "lineno", node.lineno) for d in decos) or node.lineno
    return node.lineno

def _node_end_by_walk(node: ast.AST, fallback_start: int) -> int:
    max_line = fallback_start
    for n in ast.walk(node):
        ln = getattr(n, "end_lineno", None) or getattr(n, "lineno", None)
        if isinstance(ln, int):
            max_line = max(max_line, ln)
    return max_line

def _expand_block_end_strict(lines, end_inclusive: int, base_indent: Optional[int]) -> int:
    i = end_inclusive
    n = len(lines)
    if base_indent is None:
        return i
    j = i + 1
    while j <= n:
        if _is_blank_or_comment(lines, j):
            break

        ind = _indent_of_line(lines, j)
        if ind >= base_indent:
            i = j
            j += 1
            continue
        break
    return i

def get_node_range_robust(node: ast.AST, source: str) -> Tuple[int, int, int, int]:
    """
    Return (start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive).

    - The start includes decorators.
    - The end stops strictly at the last effective statement
      (it does not consume trailing whitespace/comments).
    """
    lines = source.splitlines()
    start_inclusive = _node_start_with_decorators(node)
    header_end_inclusive = getattr(node, "lineno", start_inclusive)

    body_end_inclusive = getattr(node, "end_lineno", None)
    if not isinstance(body_end_inclusive, int):
        body_end_inclusive = _node_end_by_walk(node, header_end_inclusive)

    first_body_ln = _first_body_lineno(node)
    base_indent = _indent_of_line(lines, first_body_ln) if isinstance(first_body_ln, int) else None

    expanded_end_inclusive = _expand_block_end_strict(lines, body_end_inclusive, base_indent)
    end_exclusive = expanded_end_inclusive + 1
    return start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive



def backtrack_to_lcp_node(
    paths: List[str], 
    min_support: float = 0.6
) -> List[str]:
    """
    Given a list of paths, this function finds the longest common prefix (LCP) directories
    and returns a list of potential common parent directories. The result contains only those
    directories that have support greater than or equal to min_support.

    Args:
        paths: A list of file paths to be processed.
        min_support: The minimum support threshold (as a fraction of the number of paths) required
                      for a parent directory to be considered as the common parent.
    
    Returns:
        A list of the longest common parent directory paths.
    """
    if not paths:
        return []

    # Step 1: Extract parent directories of the paths
    parent_dirs = [str(Path(p).parent) for p in paths]
    parent_counter = Counter(parent_dirs)

    # Find the most common parent directory
    most_common_parent, count = parent_counter.most_common(1)[0]

    result = []

    # Step 2: If all paths have the same parent, return it as the common parent
    if count == len(paths):
        result.append(most_common_parent)

    # Step 3: If most paths have a common parent directory above a certain support threshold, return it
    elif count / len(paths) >= min_support:
        result.append(most_common_parent)

    # Step 4: Backtrack and find common prefixes among paths
    split_paths = [Path(p).parts for p in paths]
    max_len = min(len(p) for p in split_paths)  # Get the minimum path length

    prefix_candidates = []
    for i in range(max_len):
        tokens = [p[i] for p in split_paths]
        most_common_token, count = Counter(tokens).most_common(1)[0]

        # If the token is sufficiently common, add it to the prefix
        if count / len(paths) >= min_support:
            prefix_candidates.append(most_common_token)
        else:
            break  # Stop if the token is not sufficiently common across paths

    # If a prefix was found, create the path for the common prefix
    if prefix_candidates:
        lcp_path = str(Path(*prefix_candidates))
        result.append(lcp_path)

    return result