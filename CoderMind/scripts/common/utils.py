#!/usr/bin/env python3
"""Common Utility Functions.

This module contains shared utility functions used across multiple scripts:
- Skeleton traversal and formatting functions
- Python code validation functions
- Prompt formatting functions
- Display/printing functions
- Repository info loading functions
- Path normalization and file filtering functions (ported from RPG-ZeroRepo)
- Text / LLM output parsing functions (ported from RPG-ZeroRepo)
- Code skeleton extraction functions (ported from RPG-ZeroRepo)
- AST node range helpers (ported from RPG-ZeroRepo)
"""

import ast
import json
import logging
import os
import random
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple, Union, Any

from .paths import SKELETON_FILE, FEATURE_TREE_FILE, FEATURE_SPEC_FILE
from collections import defaultdict
logger = logging.getLogger(__name__)


# Directory names repository scanners never descend into: version control,
# editor metadata, virtualenvs, dependency installs, and the build-output
# trees of every language the decoder targets (Rust ``target``, CMake build
# dirs, ``build``/``dist``, JS framework caches). Centralized so dep_graph and
# the RPG encoder share one definition instead of each keeping its own list.
#
# Note: any dot-prefixed directory (``.git``, ``.github``, ``.cmind``,
# ``.venv``, ...) is also skipped via :func:`is_skip_dir`; the explicit
# entries below cover the non-dotted build/dependency dirs plus a few common
# dot-dirs kept for readability.
SCAN_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    ".github", ".cmind",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode",
    ".venv", "venv", "env",
    "node_modules", ".next", ".nuxt",
    "target",
    "build", "dist",
    "cmake-build-debug", "cmake-build-release",
})


def is_skip_dir(name: str) -> bool:
    """Return True if a directory ``name`` should never be scanned.

    Skips every dot-prefixed directory (``.git``, ``.github``, ``.cmind``,
    editor/tooling state, virtualenvs) plus the explicit build/dependency
    dirs in :data:`SCAN_SKIP_DIRS`. Centralizes the rule so ``os.walk``
    scanners and the dependency-graph path filter stay consistent.
    """
    return name.startswith(".") or name in SCAN_SKIP_DIRS


def path_has_skip_dir(path: str) -> bool:
    """Return True if any parent directory of ``path`` is a skip dir.

    Used by path-string filters (e.g. the dependency-graph build filter)
    that receive a relative path rather than walking a tree.
    """
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    # The last part is the file name; only directory parts gate inclusion.
    return any(is_skip_dir(part) for part in parts[:-1])


# ============================================================================
# Repository Info Functions
# ============================================================================

def get_repo_info_from_files() -> Tuple[str, str]:
    """Load repository info from available files.
    
    Tries skeleton.json first, then feature_tree.json as backup.
    
    Returns:
        Tuple of (repo_name, repo_info)
    """
    repo_name = "project"
    repo_info = ""
    
    # Try skeleton.json first
    if SKELETON_FILE.exists():
        try:
            with open(SKELETON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            repo_name = data.get("repository_name", repo_name)
            repo_info = data.get("repository_purpose", "")
        except Exception:
            pass
    
    # Also check feature_tree.json for backup
    if FEATURE_TREE_FILE.exists():
        try:
            with open(FEATURE_TREE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not repo_name or repo_name == "project":
                repo_name = data.get("repository_name", repo_name)
            if not repo_info:
                repo_info = data.get("repository_purpose", "")
        except Exception:
            pass
    
    return repo_name, repo_info


def get_project_background_context(
    feature_spec_path=None,
) -> str:
    """Load project background and technology context from feature_spec.json.

    Reads ``background_and_overview`` and ``non_functional_requirements``
    from *feature_spec_path* (defaults to ``FEATURE_SPEC_FILE``).

    The returned string is suitable for direct injection into LLM prompts.
    Returns an empty string when the file does not exist or contains no
    background entries — callers need no special-casing.

    Args:
        feature_spec_path: Optional override for the feature_spec.json location.

    Returns:
        A formatted multi-line string summarising the project background, or "".
    """
    path = Path(feature_spec_path) if feature_spec_path else FEATURE_SPEC_FILE
    if not path.exists():
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""

    parts: List[str] = []

    # Background & overview (contains technology stack, architecture, etc.)
    bg_items = data.get("background_and_overview", [])
    if bg_items:
        bg_lines: List[str] = []
        for item in bg_items:
            title = item.get("title", "")
            desc = item.get("description", "")
            if desc:
                bg_lines.append(f"- **{title}**: {desc}" if title else f"- {desc}")
        if bg_lines:
            parts.append("### Project Background & Technology")
            parts.extend(bg_lines)

    # Non-functional requirements (security, performance, etc.)
    nfr_items = data.get("non_functional_requirements", [])
    if nfr_items:
        nfr_lines: List[str] = []
        for item in nfr_items:
            title = item.get("title", "")
            desc = item.get("description", "")
            if desc:
                nfr_lines.append(f"- **{title}**: {desc}" if title else f"- {desc}")
        if nfr_lines:
            parts.append("")
            parts.append("### Non-Functional Requirements")
            parts.extend(nfr_lines)

    if not parts:
        return ""

    return "\n".join(parts) + "\n"


# ============================================================================
# Tree Traversal Functions
# ============================================================================

def get_leaf_name(item) -> str:
    """Extract feature name from a leaf node item.

    Supports both old format (str) and new format (dict with "name" key).

    Args:
        item: A leaf node item — either a string or a dict like {"name": "...", "description": "..."}

    Returns:
        The feature name string
    """
    if isinstance(item, dict):
        return item.get("name", "")
    return str(item)


def get_leaf_description(item) -> str:
    """Extract description from a leaf node item.

    Args:
        item: A leaf node item — either a string or a dict with "description" key

    Returns:
        The description string, or empty string if not available
    """
    if isinstance(item, dict):
        return item.get("description", "")
    return ""


def get_all_leaf_descriptions(tree: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Collect all leaf descriptions as {full_path: description}.

    Args:
        tree: Feature tree dictionary
        prefix: Current path prefix

    Returns:
        Dict mapping full leaf paths to their descriptions
    """
    descriptions = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            descriptions.update(get_all_leaf_descriptions(value, new_prefix))
    elif isinstance(tree, list):
        for item in tree:
            name = get_leaf_name(item)
            desc = get_leaf_description(item)
            if name and desc:
                path = f"{prefix}/{name}" if prefix else name
                descriptions[path] = desc
    return descriptions


def get_all_leaf_paths(tree: Dict[str, Any], prefix: str = "") -> List[str]:
    """Get all complete paths to leaf nodes.

    Args:
        tree: Feature tree dictionary
        prefix: Current path prefix

    Returns:
        List of full paths to all leaf nodes
    """
    paths = []
    if isinstance(tree, dict):
        if not tree:
            if prefix:
                paths.append(prefix)
        else:
            for key, value in tree.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                paths.extend(get_all_leaf_paths(value, new_prefix))
    elif isinstance(tree, list):
        if not tree:
            if prefix:
                paths.append(prefix)
        else:
            for item in tree:
                name = get_leaf_name(item)
                path = f"{prefix}/{name}" if prefix else name
                paths.append(path)
    else:
        if prefix:
            paths.append(prefix)
    return paths


# ============================================================================
# Code Analysis Functions
# ============================================================================

def extract_class_names(code: str) -> List[str]:
    """Extract class names from Python code.
    
    Args:
        code: Python source code string
        
    Returns:
        List of class names found in the code
    """
    try:
        tree = ast.parse(code)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    except SyntaxError:
        return []


# ============================================================================
# Display Functions
# ============================================================================

def print_unicode_table(headers: List[str], rows: List[List[Any]], title: str = ""):
    """Print a table with Unicode box drawing characters.
    
    Args:
        headers: List of column headers
        rows: List of rows, each row is a list of values
        title: Optional table title
    """
    if not rows:
        return

    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Print title if provided
    if title:
        print(f"\n   {title}")

    # Top border
    print("   ┌" + "┬".join("─" * w for w in col_widths) + "┐")

    # Headers
    header_row = "   │"
    for i, header in enumerate(headers):
        header_row += f" {str(header).ljust(col_widths[i] - 1)}│"
    print(header_row)

    # Separator after headers
    print("   ├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Data rows
    for idx, row in enumerate(rows):
        data_row = "   │"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                data_row += f" {str(cell).ljust(col_widths[i] - 1)}│"
        print(data_row)

        # Add separator between rows (except for last row)
        if idx < len(rows) - 1:
            print("   ├" + "┼".join("─" * w for w in col_widths) + "┤")

    # Bottom border
    print("   └" + "┴".join("─" * w for w in col_widths) + "┘")


# ============================================================================
# Skeleton Utility Functions
# ============================================================================

def get_skeleton_tree_string(skeleton: Dict[str, Any], max_depth: int = 3) -> str:
    """Generate a tree string representation of the skeleton.
    
    Args:
        skeleton: Skeleton dictionary with nested structure
        max_depth: Maximum depth to traverse
        
    Returns:
        Tree-formatted string representation (limited to 50 lines)
    """
    lines = []
    
    def traverse(node: Dict[str, Any], prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        name = node.get("name", "")
        node_type = node.get("type", "")
        
        if node_type == "directory":
            lines.append(f"{prefix}{name}/")
            children = node.get("children", [])
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{connector}{child.get('name', '')}")
                if child.get("type") == "directory":
                    traverse(child, child_prefix, depth + 1)
        else:
            lines.append(f"{prefix}{name}")
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    
    return "\n".join(lines[:50])  # Limit output


def extract_functional_areas_from_skeleton(skeleton: Dict[str, Any]) -> List[str]:
    """Extract functional area names from skeleton by analyzing feature paths.
    
    Args:
        skeleton: Skeleton dictionary with file nodes containing feature_paths
        
    Returns:
        Sorted list of unique functional area names
    """
    components = set()
    
    def traverse(node: Dict[str, Any]):
        if node.get("type") == "file":
            feature_paths = node.get("feature_paths", [])
            for fp in feature_paths:
                # Component is the first part of the feature path
                if "/" in fp:
                    component = fp.split("/")[0]
                    components.add(component)
                else:
                    components.add(fp)
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    
    return sorted(list(components))


def format_functional_graph_overview(skeleton: Dict[str, Any]) -> str:
    """Extract a hierarchical functional graph overview from the skeleton.
    
    Groups feature_paths by component (level 1) and shows unique
    sub-feature categories (level 2) under each component, formatted
    as an indented tree.
    
    Example output::
    
        Functional Graph Overview:
        Expression Processing
        ├─ output
        ├─ parsing
        ├─ representation
        └─ validation
        
        Runtime Environment
        ├─ configuration
        ├─ constants
        └─ persistence
    
    Args:
        skeleton: Skeleton dictionary with file nodes containing feature_paths
        
    Returns:
        Formatted tree string
    """
    tree: Dict[str, set] = defaultdict(set)
    
    def traverse(node: Dict[str, Any]):
        if node.get("type") == "file":
            for fp in node.get("feature_paths", []):
                parts = fp.split("/")
                component = parts[0]
                if len(parts) >= 2:
                    tree[component].add(parts[1])
                else:
                    tree[component]  # ensure key exists
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    
    if not tree:
        return "(no functional areas found)"
    
    lines = []
    for component in sorted(tree):
        lines.append(component)
        subs = sorted(tree[component])
        for i, sub in enumerate(subs):
            prefix = "└─" if i == len(subs) - 1 else "├─"
            lines.append(f"  {prefix} {sub}")
        lines.append("")  # blank line between components
    
    return "\n".join(lines)


def extract_component_directories(skeleton: Dict[str, Any]) -> Dict[str, str]:
    """Extract component to directory mapping from skeleton.
    
    Args:
        skeleton: Skeleton dictionary with file nodes
        
    Returns:
        Dict mapping component names to their directory paths
    """
    component_dirs = {}
    
    def traverse(node: Dict[str, Any]):
        if node.get("type") == "file":
            component = node.get("component", "")
            if component:
                path = node.get("path", "")
                # Get the directory containing this file
                if "/" in path:
                    dir_path = "/".join(path.split("/")[:-1])
                    if component not in component_dirs:
                        component_dirs[component] = dir_path
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                traverse(child)
    
    root = skeleton.get("root", skeleton)
    traverse(root)
    
    return component_dirs


# ============================================================================
# Code Validation Functions
# ============================================================================

def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python code syntax using AST parser.
    
    Args:
        code: Python source code string
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if code parses successfully
        - error_message: Empty string on success, error details on failure
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}, column {e.offset}: {e.msg}"


# ============================================================================
# Prompt Formatting Functions
# ============================================================================

def format_data_flow_edges(data_flow: list) -> str:
    """Format data flow edges for display in prompts.
    
    Args:
        data_flow: List of edge dicts with source, target, data_type keys
        
    Returns:
        Formatted string representation of data flow edges
    """
    if not data_flow:
        return "No data flow defined."
    
    lines = []
    for edge in data_flow:
        source = edge.get("source", "")
        target = edge.get("target", "")
        data_type = edge.get("data_type", "")
        lines.append(f"  {source} → {target}: {data_type}")
    return "\n".join(lines)


def format_base_classes(base_classes: list) -> str:
    """Format base classes for context display in prompts.
    
    Args:
        base_classes: List of base class dicts with file_path and code keys
        
    Returns:
        Formatted markdown string with code blocks
    """
    if not base_classes:
        return "No base classes available."
    
    lines = []
    for bc in base_classes:
        if isinstance(bc, dict):
            file_path = bc.get("file_path", "unknown")
            code = bc.get("code", "")
            lines.append(f"### {file_path}\n```python\n{code}\n```\n")
    
    return "\n".join(lines)


def format_data_structures(data_structures: list) -> str:
    """Format data flow data structures for context display in prompts.
    
    Args:
        data_structures: List of data structure dicts with code, subtree, and data_flow_types keys
        
    Returns:
        Formatted markdown string with code blocks
    """
    if not data_structures:
        return "No data flow data structures available."
    
    lines = []
    for ds in data_structures:
        if isinstance(ds, dict):
            subtree = ds.get("subtree", "unknown")
            code = ds.get("code", "")
            df_types = ds.get("data_flow_types", [])
            types_str = ", ".join(df_types) if df_types else "(unspecified)"
            file_path = ds.get("file_path", "")
            header = f"### Subtree: {subtree}"
            if file_path:
                header += f" | File: {file_path}"
            lines.append(f"{header}\nCovers data flow types: {types_str}\n```python\n{code}\n```\n")
    
    return "\n".join(lines)


def format_base_classes_and_data_structures(base_classes: list, data_structures: list) -> str:
    """Format both base classes and data structures for context display in prompts.

    Args:
        base_classes: List of base class dicts
        data_structures: List of data structure dicts

    Returns:
        Formatted markdown string with code blocks for both sections
    """
    parts = []

    bc_str = format_base_classes(base_classes)
    if base_classes:
        parts.append("## Base Classes\n" + bc_str)

    ds_str = format_data_structures(data_structures)
    if data_structures:
        parts.append("## Data Flow Data Structures\n" + ds_str)

    if not parts:
        return "No base classes or data structures available."

    return "\n\n".join(parts)


# ============================================================================
# Path Normalization Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/file.py and
#  RPG-ZeroRepo/zerorepo/rpg_gen/base/rpg/util.py)
# ============================================================================

def normalize_path(path: Union[str, Path]) -> str:
    """Normalize a node id into a relative POSIX-style format.

    Form: rel/posix/path[:qualname.with.dots]

    Rules:
      - Compatible with Windows/Linux
      - Resolve redundant path components like ".." and "."
      - Remove leading "./" prefix for consistency
      - Treat the part after ":" as a symbol qualified name,
        split by '.', and filter empty segments

    Source: RPG-ZeroRepo/zerorepo/utils/file.py (normalize_path)
    """
    s = str(path).strip()
    if ":" in s:
        left, right = s.split(":", 1)
    else:
        left, right = s, None

    norm = PurePosixPath(str(left).strip()).as_posix()
    norm = norm.removeprefix("./").removeprefix("/")
    if norm == "" or norm == ".":
        base = "."
    else:
        base = norm

    if right is not None:
        segs = [seg.strip() for seg in right.strip().strip(".").split(".") if seg.strip()]
        if segs:
            return f"{base}:{'.'.join(segs)}"
    return base


# ============================================================================
# File Filtering Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/repo.py)
# ============================================================================

def is_test_file(nid: str) -> bool:
    """Check whether a node id belongs to a test file.

    Splits the file path portion by ' ', '_', and '/' and checks if any
    segment starts with 'test'.

    Source: RPG-ZeroRepo/zerorepo/utils/repo.py (is_test_file)
    """
    file_path = nid.split(":")[0]
    word_list = re.split(r" |_|/", file_path.lower())
    return any(word.startswith("test") for word in word_list)


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping inclusive intervals.

    Given a list of (start, end) tuples where both endpoints are inclusive,
    merge overlapping or adjacent intervals and return the merged result
    sorted by start position.

    Args:
        intervals: List of (start, end) tuples, both inclusive.

    Returns:
        Merged list of (start, end) tuples.

    Source: RPG-ZeroRepo/zerorepo/utils/repo.py (merge_intervals)
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def filter_excluded_files(valid_files: List[str], excluded_files: List[str]) -> List[str]:
    """Filter out files that match any path in *excluded_files*.

    *excluded_files* may contain files or directories:
      - If it is a file: remove on exact match.
      - If it is a directory: remove all files under that directory.

    Args:
        valid_files: All valid file paths (typically .py files in the repo).
        excluded_files: List of file or directory paths to exclude.

    Returns:
        The filtered list of valid_files.

    Source: RPG-ZeroRepo/zerorepo/utils/repo.py (filter_excluded_files)
    """
    norm_excluded = [normalize_path(p) for p in excluded_files if p.strip()]
    filtered = []

    for vf in valid_files:
        norm_vf = normalize_path(vf)
        excluded = False
        for excl in norm_excluded:
            if norm_vf == excl or norm_vf.startswith(excl + "/"):
                excluded = True
                break
        if not excluded:
            filtered.append(vf)

    return filtered


# ============================================================================
# LLM Output Parsing Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/api.py)
# ============================================================================

def parse_solution_output(output: str) -> str:
    """Extract the content inside ``<solution>...</solution>`` tags.

    If the tags are not present the full (stripped) output is returned.

    Args:
        output: Raw LLM output string.

    Returns:
        Extracted solution text, stripped of leading/trailing whitespace.

    Source: RPG-ZeroRepo/zerorepo/utils/api.py (parse_solution_output)
    """
    output = output.split("<solution>", 1)[-1]
    output = output.split("</solution>", 1)[0]
    return output.strip()


def parse_code_blocks(output: str, type: str = "general") -> List[str]:
    """Parse markdown fenced code blocks from a string.

    Args:
        output: The text containing code blocks.
        type: The language type to match.
              - ``"general"``: matches any ````` ... ````` block.
              - ``"python"``, ``"javascript"``, etc.: matches only that language.

    Returns:
        A list of extracted code block contents, each stripped of
        leading/trailing whitespace.

    Source: RPG-ZeroRepo/zerorepo/utils/api.py (parse_code_blocks)
    """
    if type == "general":
        pattern = r"```(?:\n)?(.*?)```"
    else:
        pattern = rf"```{type}\s+(.*?)```"

    matches = re.findall(pattern, output, re.DOTALL)
    return [m.strip() for m in matches]


# ============================================================================
# Code Skeleton Extraction Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/compress.py)
# ============================================================================

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
    line_number_mode: str = "none",
) -> str:
    """Generate a structural skeleton version of Python source code.

    Uses ``libcst`` to strip function bodies (replaced with ``...``) while
    keeping class/function signatures, optional constants, docstrings, and
    import statements.  Very long module-level assignments can be compressed
    to keep only their head and tail.

    Args:
        raw_code: Python source code to compress.
        keep_constant: Keep short module-level constant assignments.
        keep_indent: Preserve indentation when omitting function bodies.
        compress_assign: Fold very long module-level assignments.
        keep_docstring: Keep module/class/function docstrings.
        keep_imports: Keep ``import`` / ``from ... import ...`` statements.
        total_lines: Threshold (in lines) above which an assignment is folded.
        prefix_lines: Number of head lines to keep when folding.
        suffix_lines: Number of tail lines to keep when folding.
        line_number_mode: ``"none"`` | ``"original"`` | ``"sequential"``.

    Returns:
        The skeleton code string.

    Source: RPG-ZeroRepo/zerorepo/utils/compress.py (get_skeleton)
    """
    try:
        import libcst as cst
        import libcst.matchers as m
    except ImportError:
        logger.warning(
            "libcst is not installed; get_skeleton() will return raw code. "
            "Install with: pip install libcst"
        )
        return raw_code

    # --- internal transformer (inline to avoid top-level libcst import) ---
    replacement_string = '"__FUNC_BODY_REPLACEMENT_STRING__"'

    class _CompressTransformer(cst.CSTTransformer):
        """Replace function bodies with ``...`` while preserving structure."""

        def __init__(self):
            pass

        def _is_import_stmt(self, stmt: cst.CSTNode) -> bool:
            if not m.matches(stmt, m.SimpleStatementLine()):
                return False
            return any(
                m.matches(s, m.Import()) or m.matches(s, m.ImportFrom())
                for s in getattr(stmt, "body", [])
            )

        def leave_Module(self, original_node, updated_node):
            new_body = []
            for i, stmt in enumerate(updated_node.body):
                if m.matches(stmt, m.ClassDef()) or m.matches(stmt, m.FunctionDef()):
                    new_body.append(stmt)
                elif (
                    keep_constant
                    and m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Assign())
                ):
                    new_body.append(stmt)
                elif keep_imports and self._is_import_stmt(stmt):
                    new_body.append(stmt)
                elif (
                    keep_docstring
                    and i == 0
                    and m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Expr())
                    and m.matches(stmt.body[0].value, m.SimpleString())
                ):
                    new_body.append(stmt)
            return updated_node.with_changes(body=new_body)

        def leave_ClassDef(self, original_node, updated_node):
            new_body = []
            for i, stmt in enumerate(updated_node.body.body):
                if (
                    i == 0
                    and keep_docstring
                    and m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Expr())
                    and m.matches(stmt.body[0].value, m.SimpleString())
                ):
                    new_body.append(stmt)
                elif not (
                    m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Expr())
                    and m.matches(stmt.body[0].value, m.SimpleString())
                ):
                    new_body.append(stmt)
            return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

        def leave_FunctionDef(self, original_node, updated_node):
            docstring_stmt = None
            import_stmts: List = []

            for i, stmt in enumerate(updated_node.body.body):
                if (
                    i == 0
                    and keep_docstring
                    and m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Expr())
                    and m.matches(stmt.body[0].value, m.SimpleString())
                ):
                    docstring_stmt = stmt
                else:
                    if keep_imports and self._is_import_stmt(stmt):
                        import_stmts.append(stmt)

            rep_expr = cst.Expr(value=cst.SimpleString(value=replacement_string))
            rep_stmt = cst.SimpleStatementLine(body=[rep_expr])

            if keep_indent:
                body = []
                if docstring_stmt:
                    body.append(docstring_stmt)
                body.extend(import_stmts)
                body.append(rep_stmt)
                return updated_node.with_changes(body=cst.IndentedBlock(body=body))

            new_body = list(import_stmts) + [rep_stmt]
            return updated_node.with_changes(body=cst.IndentedBlock(tuple(new_body)))

    # --- internal helpers for assignment compression ---
    class _GlobalVariableVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

        def __init__(self):
            self.assigns: list = []

        def leave_Assign(self, original_node):
            start_pos = self.get_metadata(cst.metadata.PositionProvider, original_node).start
            end_pos = self.get_metadata(cst.metadata.PositionProvider, original_node).end
            self.assigns.append([original_node, start_pos, end_pos])

    def _remove_lines(raw: str, remove_intervals):
        new_code = ""
        for i, line in enumerate(raw.splitlines(), start=1):
            if not any(s <= i <= e for s, e in remove_intervals):
                new_code += line + "\n"
            if any(s == i for s, _ in remove_intervals):
                new_code += "...\n"
        return new_code

    def _compress_assign_stmts(raw: str) -> str:
        try:
            tree = cst.parse_module(raw)
        except Exception:
            return raw
        wrapper = cst.metadata.MetadataWrapper(tree)
        visitor = _GlobalVariableVisitor()
        wrapper.visit(visitor)
        intervals = []
        for _, start, end in visitor.assigns:
            if end.line - start.line > total_lines:
                intervals.append((start.line + prefix_lines, end.line - suffix_lines))
        return _remove_lines(raw, intervals)

    def _add_original_line_numbers(raw: str, skel: str) -> str:
        import difflib as _difflib

        orig = raw.splitlines()
        skel_lines = skel.splitlines()
        sm = _difflib.SequenceMatcher(None, orig, skel_lines, autojunk=False)
        width = len(str(len(orig)))
        out: List[str] = []
        prev_orig_end = 0

        def _gap(start_idx, end_idx):
            if end_idx <= start_idx:
                return
            left = str(start_idx + 1).rjust(width)
            right = str(end_idx).rjust(width)
            out.append(f"{left}..{right} | ...")

        for i_orig, j_skel, n in sm.get_matching_blocks():
            _gap(prev_orig_end, i_orig)
            for k in range(n):
                raw_ln = i_orig + k + 1
                line = skel_lines[j_skel + k]
                out.append(f"{str(raw_ln).rjust(width)} | {line}")
            prev_orig_end = i_orig + n

        _gap(prev_orig_end, len(orig))
        return "\n".join(out)

    # --- main logic ---
    try:
        tree = cst.parse_module(raw_code)
    except Exception:
        code = raw_code
    else:
        transformer = _CompressTransformer()
        modified_tree = tree.visit(transformer)
        code = modified_tree.code

    if compress_assign:
        code = _compress_assign_stmts(code)

    if keep_indent:
        code = code.replace(replacement_string + "\n", "...\n")
        code = code.replace(replacement_string, "...\n")
    else:
        pattern_re = f"\\n[ \\t]*{replacement_string}"
        code = re.sub(pattern_re, "\n...", code)

    if line_number_mode == "original":
        return _add_original_line_numbers(raw_code, code)
    elif line_number_mode == "sequential":
        lines = code.splitlines()
        width = len(str(len(lines)))
        return "\n".join(f"{str(i).rjust(width)} | {ln}" for i, ln in enumerate(lines, 1))

    return code


# ============================================================================
# Parsed Feature Tree Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/tree.py)
# ============================================================================

def transfer_parsed_tree(
    input_tree: Dict,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Transform a parsed feature tree into summary and reverse-index mappings.

    Returns:
        A tuple of:
          - format_tree:      ``{ file_summary: [features...] }``
          - feature_to_files: ``{ feature: [file_paths...] }``

    Merges all nested function/class-level descriptions into the file-level
    node and automatically deduplicates feature text.

    Source: RPG-ZeroRepo/zerorepo/utils/tree.py (transfer_parsed_tree)
    """

    def _collect_texts(value: Union[str, List, Dict, None]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            result: List[str] = []
            for v in value:
                result.extend(_collect_texts(v))
            return result
        if isinstance(value, dict):
            result = []
            for v in value.values():
                result.extend(_collect_texts(v))
            return result
        return [str(value)]

    format_tree: Dict[str, List[str]] = {}
    feature_to_files: Dict[str, List[str]] = {}

    for file_path, file_tree in input_tree.items():
        file_summary = file_tree.get(
            "_file_summary_",
            os.path.basename(file_path).replace(".py", ""),
        )

        all_texts: List[str] = []
        for key, value in file_tree.items():
            # Sidecar keys: ``_file_summary_`` is the file's verbal summary
            # (already used as the format-tree key above); ``_feature_descriptions_``
            # stores the LLM-generated descriptions and must NOT be collected
            # as features (otherwise descriptions leak into functional-area
            # planner prompts).
            if key in ("_file_summary_", "_feature_descriptions_"):
                continue
            all_texts.extend(_collect_texts(value))

        deduped_texts = sorted(set(all_texts))
        format_tree[file_summary] = deduped_texts

        for feature in deduped_texts:
            feature_to_files.setdefault(feature, []).append(file_path)

    return format_tree, feature_to_files


def format_parsed_tree(
    input_tree: Dict,
    omit_full_leaf_nodes: bool = False,
    max_features: int = 2,
) -> str:
    """Format a parsed feature tree into a condensed, human-readable JSON string.

    Reuses :func:`transfer_parsed_tree` to build the base mapping, then
    optionally truncates long feature lists for readability.

    Args:
        input_tree: The parsed feature tree (``{ file_path: file_tree }``).
        omit_full_leaf_nodes: If True, truncate feature lists longer than 2.
        max_features: Number of features to sample when truncating.

    Returns:
        A compact JSON string.

    Source: RPG-ZeroRepo/zerorepo/utils/tree.py (format_parsed_tree)
    """
    fmt_tree, _ = transfer_parsed_tree(input_tree)

    for key, features in fmt_tree.items():
        if omit_full_leaf_nodes and len(features) > 2:
            sampled = random.sample(features, min(max_features, len(features)))
            fmt_tree[key] = sampled + ["..."]

    return json.dumps(fmt_tree, ensure_ascii=False, separators=(",", ":"))


def iterative_by_folder(parsed_tree: Dict) -> Dict[str, List[str]]:
    """Group file paths in a parsed tree by their parent folder.

    Args:
        parsed_tree: A dict whose keys are file paths.

    Returns:
        ``{ folder_path: [file_paths...] }``.  Root-level files are
        grouped under ``"(root)"``.

    Source: RPG-ZeroRepo/zerorepo/utils/tree.py (iterative_by_folder)
    """
    file_paths = list(parsed_tree.keys())
    grouped: Dict[str, List[str]] = {}

    for p in file_paths:
        p_norm = p.rstrip("/")
        parent_dir = os.path.dirname(p_norm)
        folder = parent_dir if parent_dir else "(root)"
        grouped.setdefault(folder, []).append(p)

    return grouped


# ============================================================================
# AST Node Range Helpers
# (Ported from RPG-ZeroRepo/zerorepo/rpg_gen/base/rpg/util.py)
# ============================================================================

def _indent_of_line(lines: list, lineno: int) -> int:
    """Return the indentation width (in columns, tab=8) of a 1-based line."""
    if lineno <= 0 or lineno > len(lines):
        return 0
    s = lines[lineno - 1]
    return len(s.expandtabs(8)) - len(s.lstrip().expandtabs(8))


def _is_blank_or_comment(lines: list, lineno: int) -> bool:
    """Return True if the 1-based line is blank or a comment."""
    if lineno <= 0 or lineno > len(lines):
        return True
    s = lines[lineno - 1].strip()
    return not s or s.startswith("#")


def _first_body_lineno(node: ast.AST) -> Optional[int]:
    """Return the line number of the first statement in *node*.body, or None."""
    body = getattr(node, "body", None)
    if not body:
        return None
    return getattr(body[0], "lineno", None)


def _node_start_with_decorators(node: ast.AST) -> int:
    """Return the start line including any decorators."""
    decos = getattr(node, "decorator_list", None)
    if decos:
        return min(getattr(d, "lineno", node.lineno) for d in decos) or node.lineno
    return node.lineno


def _node_end_by_walk(node: ast.AST, fallback_start: int) -> int:
    """Walk *node* to find the maximum end_lineno / lineno."""
    max_line = fallback_start
    for n in ast.walk(node):
        ln = getattr(n, "end_lineno", None) or getattr(n, "lineno", None)
        if isinstance(ln, int):
            max_line = max(max_line, ln)
    return max_line


def _expand_block_end_strict(
    lines: list, end_inclusive: int, base_indent: Optional[int]
) -> int:
    """Expand *end_inclusive* downward while lines have >= base_indent."""
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
    """Return the line range of an AST node including decorators.

    Returns:
        ``(start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive)``

    - The start includes decorators.
    - The end stops strictly at the last effective statement
      (it does not consume trailing whitespace/comments).

    Source: RPG-ZeroRepo/zerorepo/rpg_gen/base/rpg/util.py (get_node_range_robust)
    """
    lines = source.splitlines()
    start_inclusive = _node_start_with_decorators(node)
    header_end_inclusive = getattr(node, "lineno", start_inclusive)

    body_end_inclusive = getattr(node, "end_lineno", None)
    if not isinstance(body_end_inclusive, int):
        body_end_inclusive = _node_end_by_walk(node, header_end_inclusive)

    first_body_ln = _first_body_lineno(node)
    base_indent = (
        _indent_of_line(lines, first_body_ln) if isinstance(first_body_ln, int) else None
    )

    expanded_end_inclusive = _expand_block_end_strict(lines, body_end_inclusive, base_indent)
    end_exclusive = expanded_end_inclusive + 1
    return start_inclusive, header_end_inclusive, body_end_inclusive, end_exclusive


def extract_source_by_lines(
    source: str, start_inclusive: int, end_inclusive: int
) -> str:
    """Extract lines [start_inclusive, end_inclusive] (1-based, inclusive) from *source*.

    Preserves original blank lines, comments, indentation, and newlines.

    Source: RPG-ZeroRepo/zerorepo/rpg_gen/base/rpg/util.py (extract_source_by_lines)
    """
    if start_inclusive is None or end_inclusive is None:
        return ""
    lines = source.splitlines(keepends=True)
    n = len(lines)
    s = max(1, start_inclusive)
    e = min(n, end_inclusive)
    if s > e:
        return ""
    return "".join(lines[s - 1 : e]).strip()


# ============================================================================
# Token Counting and Truncation Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/api.py)
# ============================================================================

def calculate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Calculate the number of tokens in the text using tiktoken.

    Args:
        text: The text to count tokens for.
        model: The tiktoken model to use for encoding.

    Returns:
        Number of tokens.

    Source: RPG-ZeroRepo/zerorepo/utils/api.py (calculate_tokens)
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed; calculate_tokens() will estimate. "
            "Install with: pip install tiktoken"
        )
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4

    model_to_encoding = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "code-davinci-002": "p50k_base",
    }

    encoding_name = model_to_encoding.get(model, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    # Remove special tokens before encoding
    import re as _re
    specials = enc.special_tokens_set
    pattern = _re.compile("|".join(_re.escape(s) for s in specials))
    cleaned_text = pattern.sub("", text)

    tokens = enc.encode(cleaned_text, disallowed_special=())
    return len(tokens)


def truncate_by_token(
    text: str,
    max_tokens: int = 50000,
    model: str = "gpt-4o",
) -> str:
    """Truncate text by token count, keeping head and tail.

    If the token count does not exceed *max_tokens*, returns the text as-is.
    Otherwise keeps head and tail tokens and removes a middle segment.

    Args:
        text: The text to truncate.
        max_tokens: Maximum allowed token count.
        model: The tiktoken model to use for encoding.

    Returns:
        The (possibly truncated) text.

    Source: RPG-ZeroRepo/zerorepo/utils/api.py (truncate_by_token)
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed; truncate_by_token() will return raw text. "
            "Install with: pip install tiktoken"
        )
        return text

    model_to_encoding = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "code-davinci-002": "p50k_base",
    }

    encoding_name = model_to_encoding.get(model, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)

    tokens = enc.encode(text)
    total = len(tokens)

    if total <= max_tokens:
        return text

    keep = max_tokens
    head_keep = keep // 2 + keep % 2
    tail_keep = keep // 2

    if keep >= 2:
        head_keep = max(1, head_keep)
        tail_keep = max(1, tail_keep)

    removed = total - (head_keep + tail_keep)
    if removed <= 0:
        return text

    head_tokens = tokens[:head_keep]
    tail_tokens = tokens[-tail_keep:] if tail_keep > 0 else []

    head_str = enc.decode(head_tokens)
    tail_str = enc.decode(tail_tokens)

    marker = (
        f"\n\n... [output truncated: {removed} tokens omitted in the middle] ...\n\n"
    )

    return head_str + marker + tail_str


# ============================================================================
# Tree Mutation Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/tree.py for M7 RPG Encoding)
# ============================================================================


def convert_leaves_to_list(tree):
    """Recursively convert empty list leaves into empty dicts.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (convert_leaves_to_list)
    """
    if isinstance(tree, dict):
        return {k: convert_leaves_to_list(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        if not tree:
            return {}
        return tree
    else:
        return tree


def _collapse_leaf_dicts(node):
    """Collapse dicts where all values are empty lists into a list of keys.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (_collapse_leaf_dicts)
    """
    if isinstance(node, dict):
        if not node:
            return {}
        collapsed = {k: _collapse_leaf_dicts(v) for k, v in node.items()}
        if all(isinstance(v, list) and len(v) == 0 for v in collapsed.values()):
            return list(collapsed.keys())
        return collapsed
    elif isinstance(node, list):
        return [_collapse_leaf_dicts(v) for v in node]
    else:
        return node


def _split_path_by_delimiters(path: str, delimiters) -> List[str]:
    """Split a path string by one or more delimiters.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (split_path)
    """
    if isinstance(delimiters, str):
        delimiters = [delimiters]
    pattern = "|".join(re.escape(d) for d in delimiters)
    parts = [p.strip() for p in re.split(pattern, path) if p.strip()]
    return parts


def _insert_path(tree: dict, path: str, delimiters="/") -> None:
    """Insert a path into a tree structure, supporting multiple delimiters.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (insert_path)
    """
    if isinstance(delimiters, str):
        parts = [p.strip() for p in path.split(delimiters) if p.strip()]
    else:
        parts = _split_path_by_delimiters(path, delimiters)

    parent, key_in_parent = None, None
    node = tree
    i = 0

    while i < len(parts):
        part = parts[i]
        last = (i == len(parts) - 1)

        if isinstance(node, dict):
            mk = next((k for k in node if k.lower() == part.lower()), None)

            if last:
                if mk is None:
                    node[part] = []
                break
            else:
                if mk is None:
                    node[part] = {}
                    mk = part
                elif isinstance(node[mk], list):
                    node[mk] = {x: [] for x in node[mk]}
                elif not isinstance(node[mk], dict):
                    node[mk] = {}
                parent, key_in_parent = node, mk
                node = node[mk]
                i += 1
                continue

        elif isinstance(node, list):
            if last:
                if part.lower() not in (x.lower() for x in node):
                    node.append(part)
                break
            else:
                upgraded = {x: [] for x in node}
                parent[key_in_parent] = upgraded
                node = upgraded
                continue
        else:
            upgraded = {}
            parent[key_in_parent] = upgraded
            node = upgraded
            continue


def apply_changes(
    tree: dict,
    changes,
    *,
    delimiters="/",
    inplace: bool = True,
    auto_collapse: bool = True,
) -> dict:
    """Batch-insert paths into a tree and optionally normalise leaves.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (apply_changes)
    """
    import copy

    target = tree if inplace else copy.deepcopy(tree)
    if isinstance(changes, str):
        changes = [changes]
    for p in changes:
        _insert_path(target, p, delimiters)
    if auto_collapse:
        collapsed = _collapse_leaf_dicts(target)
        if inplace:
            tree.clear()
            tree.update(collapsed)
            return tree
        else:
            return collapsed
    return target


def get_rpg_info(
    rpg_tree: List[Dict],
    omit_leaf_nodes: bool = True,
    sample_size: int = 2,
    indent: Optional[int] = None,
) -> str:
    """Get a summarised string representation of an RPG tree structure.

    Source: RPG-ZeroRepo ``zerorepo/utils/tree.py`` (get_rpg_info)
    """

    def _prune(node):
        if isinstance(node, list):
            if not omit_leaf_nodes:
                return node
            if sample_size <= 0:
                return {}
            if len(node) > sample_size:
                return random.sample(node, sample_size) + ["..."]
            return node

        if isinstance(node, dict):
            if not node:
                return {}

            out: Dict[str, Any] = {}
            leaf_keys: List[str] = []

            for k, v in node.items():
                pv = _prune(v)
                if isinstance(pv, dict) and not pv:
                    leaf_keys.append(k)
                else:
                    out[k] = pv

            if not out and leaf_keys:
                return leaf_keys

            if leaf_keys:
                out["_"] = leaf_keys

            return out

        return node

    rpg_info: Dict[str, Any] = {}
    for sub_tree in rpg_tree:
        name = sub_tree.get("name")
        tree = sub_tree.get("refactored_subtree", {})
        rpg_info[name] = _prune(tree)

    if indent is None:
        return json.dumps(rpg_info, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(rpg_info, ensure_ascii=False, indent=indent)


def exclude_files(files: List[str]) -> List[str]:
    """Filter out common non-essential files from a file list.

    Returns a list of paths that should be excluded (test files, docs, etc.).

    Source: RPG-ZeroRepo ``zerorepo/utils/repo.py`` (exclude_files)
    """
    excluded: List[str] = []
    exclude_prefixes = (
        "test/", "tests/", "doc/", "docs/",
        "example/", "examples/", "demo/", "demos/",
        "bench/", "benchmarks/",
    )
    exclude_patterns = ("__pycache__", ".egg-info", "node_modules")

    for f in files:
        f_lower = f.lower().replace("\\", "/")
        if any(f_lower.startswith(p) for p in exclude_prefixes):
            excluded.append(f)
        elif any(pat in f_lower for pat in exclude_patterns):
            excluded.append(f)
        elif is_test_file(f):
            excluded.append(f)
    return excluded


# ============================================================================
# Text Normalization Functions
# (Ported from RPG-ZeroRepo/zerorepo/utils/repo.py)
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching: strip extensions, split camelCase, replace separators with spaces, and lowercase.

    Source: RPG-ZeroRepo/zerorepo/utils/repo.py (normalize_text)
    """
    if not text:
        return ""

    # Strip file extension
    text = re.sub(r"\.[a-zA-Z0-9]+$", "", text)
    # Split camelCase
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Replace path/code separators with spaces
    text = re.sub(r"[/_.\-:]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def wrap_code_snippet(code_snippet: str, start_line: int, end_line: int) -> str:
    """Wrap a code snippet with line numbers in a fenced code block.

    Args:
        code_snippet: The raw source code string.
        start_line: The 1-based starting line number.
        end_line: The 1-based ending line number.

    Returns:
        A Markdown fenced code block with line-numbered content.

    Source: RPG-ZeroRepo/zerorepo/utils/repo.py (wrap_code_snippet)
    """
    lines = code_snippet.split("\n")
    max_line_number = start_line + len(lines) - 1

    if not (start_line == end_line == 1):  # which is a file
        assert max_line_number == end_line

    number_width = len(str(max_line_number))
    return (
        "```\n"
        + "\n".join(
            f"{str(i + start_line).rjust(number_width)} | {line}"
            for i, line in enumerate(lines)
        )
        + "\n```"
    )
