#!/usr/bin/env python3
"""Count lines of code in a Python project.

Usage:
    python count_lines.py [project_root]

Reports line counts for two groups:
  - Source: src/ + main.py
  - Tests:  test/ or tests/

Three metrics per group:
  1. Total lines: all lines in all .py files
  2. Functional lines: excluding non-functional files (__init__.py, conftest.py, etc.)
  3. Effective lines: only actual code (no comments, docstrings, blank lines)
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple


# Files considered non-functional (boilerplate, config, etc.)
NON_FUNCTIONAL_FILES = {
    "__init__.py",
    "conftest.py",
    "setup.py",
    "pyproject.toml",
    "__main__.py",
}


@dataclass
class FileStats:
    """Line statistics for a single file."""
    path: Path
    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    docstring_lines: int = 0
    code_lines: int = 0
    is_functional: bool = True

    @property
    def effective_lines(self) -> int:
        """Lines that are actual code (not blank, comment, or docstring)."""
        return self.code_lines


@dataclass
class GroupStats:
    """Aggregated statistics for a group of files (src or test)."""
    name: str
    files: List[FileStats] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def functional_file_count(self) -> int:
        return sum(1 for f in self.files if f.is_functional)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files)

    @property
    def functional_total_lines(self) -> int:
        return sum(f.total_lines for f in self.files if f.is_functional)

    @property
    def effective_lines(self) -> int:
        return sum(f.effective_lines for f in self.files if f.is_functional)

    @property
    def blank_lines(self) -> int:
        return sum(f.blank_lines for f in self.files if f.is_functional)

    @property
    def comment_lines(self) -> int:
        return sum(f.comment_lines for f in self.files if f.is_functional)

    @property
    def docstring_lines(self) -> int:
        return sum(f.docstring_lines for f in self.files if f.is_functional)


def get_docstring_lines(source: str) -> Set[int]:
    """Get line numbers that are part of docstrings (1-indexed)."""
    docstring_lines = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return docstring_lines

    for node in ast.walk(tree):
        # Check for docstrings in modules, classes, and functions
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                docstring_node = node.body[0]
                for line_no in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                    docstring_lines.add(line_no)

    return docstring_lines


def analyze_file(filepath: Path) -> FileStats:
    """Analyze a single Python file for line statistics."""
    stats = FileStats(path=filepath)
    stats.is_functional = filepath.name not in NON_FUNCTIONAL_FILES

    try:
        content = filepath.read_text(encoding="utf-8")
    except (IOError, UnicodeDecodeError):
        return stats

    lines = content.splitlines()
    stats.total_lines = len(lines)

    # Get docstring line numbers
    docstring_lines = get_docstring_lines(content)

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        if not stripped:
            stats.blank_lines += 1
        elif i in docstring_lines:
            stats.docstring_lines += 1
        elif stripped.startswith("#"):
            stats.comment_lines += 1
        else:
            stats.code_lines += 1

    return stats


def collect_python_files(root: Path, dirs: List[str]) -> List[Path]:
    """Collect all .py files from specified directories."""
    files = []
    for d in dirs:
        path = root / d
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def analyze_group(root: Path, name: str, paths: List[str]) -> GroupStats:
    """Analyze a group of paths (directories or files)."""
    group = GroupStats(name=name)
    files = collect_python_files(root, paths)

    for f in files:
        stats = analyze_file(f)
        group.files.append(stats)

    return group


def print_report(source_group: GroupStats, test_group: GroupStats) -> None:
    """Print the line count report."""
    print()
    print("=" * 70)
    print("  Python Line Count Report")
    print("=" * 70)
    print()

    # Header
    hdr = f"{'Metric':<30}  {'Source':>12}  {'Tests':>12}  {'Total':>12}"
    print(hdr)
    print("-" * len(hdr))

    # File counts
    src_files = source_group.file_count
    test_files = test_group.file_count
    print(f"{'Files (all)':<30}  {src_files:>12,}  {test_files:>12,}  {src_files + test_files:>12,}")

    src_func_files = source_group.functional_file_count
    test_func_files = test_group.functional_file_count
    print(f"{'Files (functional)':<30}  {src_func_files:>12,}  {test_func_files:>12,}  {src_func_files + test_func_files:>12,}")

    print()

    # Line counts - Total (all files)
    src_total = source_group.total_lines
    test_total = test_group.total_lines
    print(f"{'Total lines (all files)':<30}  {src_total:>12,}  {test_total:>12,}  {src_total + test_total:>12,}")

    # Line counts - Functional files only
    src_func_total = source_group.functional_total_lines
    test_func_total = test_group.functional_total_lines
    print(f"{'Total lines (functional)':<30}  {src_func_total:>12,}  {test_func_total:>12,}  {src_func_total + test_func_total:>12,}")

    # Line counts - Effective (code only)
    src_effective = source_group.effective_lines
    test_effective = test_group.effective_lines
    print(f"{'Effective lines (code only)':<30}  {src_effective:>12,}  {test_effective:>12,}  {src_effective + test_effective:>12,}")

    print()
    print("-" * len(hdr))

    # Breakdown for functional files
    print()
    print("Breakdown (functional files only):")
    print()

    src_blank = source_group.blank_lines
    test_blank = test_group.blank_lines
    print(f"{'  Blank lines':<30}  {src_blank:>12,}  {test_blank:>12,}  {src_blank + test_blank:>12,}")

    src_comment = source_group.comment_lines
    test_comment = test_group.comment_lines
    print(f"{'  Comment lines':<30}  {src_comment:>12,}  {test_comment:>12,}  {src_comment + test_comment:>12,}")

    src_docstring = source_group.docstring_lines
    test_docstring = test_group.docstring_lines
    print(f"{'  Docstring lines':<30}  {src_docstring:>12,}  {test_docstring:>12,}  {src_docstring + test_docstring:>12,}")

    src_code = source_group.effective_lines
    test_code = test_group.effective_lines
    print(f"{'  Code lines':<30}  {src_code:>12,}  {test_code:>12,}  {src_code + test_code:>12,}")

    print()

    # Ratios
    if src_func_total > 0:
        src_ratio = src_effective / src_func_total * 100
    else:
        src_ratio = 0
    if test_func_total > 0:
        test_ratio = test_effective / test_func_total * 100
    else:
        test_ratio = 0
    total_func = src_func_total + test_func_total
    total_eff = src_effective + test_effective
    if total_func > 0:
        total_ratio = total_eff / total_func * 100
    else:
        total_ratio = 0

    print(f"{'Code density (effective/total)':<30}  {src_ratio:>11.1f}%  {test_ratio:>11.1f}%  {total_ratio:>11.1f}%")
    print()


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    else:
        root = Path.cwd()

    if not root.is_dir():
        print(f"Error: Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Determine source paths
    source_paths = []
    if (root / "src").is_dir():
        source_paths.append("src")
    if (root / "main.py").is_file():
        source_paths.append("main.py")

    # Determine test paths
    test_paths = []
    if (root / "tests").is_dir():
        test_paths.append("tests")
    elif (root / "test").is_dir():
        test_paths.append("test")

    # Analyze
    source_group = analyze_group(root, "Source", source_paths)
    test_group = analyze_group(root, "Tests", test_paths)

    if source_group.file_count == 0 and test_group.file_count == 0:
        print(f"No Python files found in {root}")
        print(f"  Looked for: src/, main.py, tests/, test/")
        sys.exit(1)

    print_report(source_group, test_group)


if __name__ == "__main__":
    main()
