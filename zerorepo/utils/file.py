from typing import Union, List
from pathlib import PurePosixPath, Path

def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize a node id into a relative POSIX-style format prefixed with "./", and support symbol-level ids:
    Form: ./rel/posix/path[:qualname.with.dots]
    Rules:
      - Compatible with Windows/Linux
      - Resolve redundant path components like ".." and "."
      - Remove extra leading "/", and consistently add a "./" prefix
      - Treat the part after ":" as a symbol qualified name (class/function/method chain),
        split by '.', and filter empty segments and whitespace
    """
    s = str(path).strip()
    # Split into file path part and optional symbol qualified name part
    if ":" in s:
        left, right = s.split(":", 1)
    else:
        left, right = s, None

    # Normalize file path to relative "./..."
    norm = PurePosixPath(str(left).strip()).as_posix()
    norm = norm.removeprefix("./").removeprefix("/")
    if norm == "" or norm == ".":
        base = "."
    else:
        base = f"{norm}"

    # Normalize symbol qualified name (contains chain: name1.name2...)
    if right is not None:
        # Strip whitespace, trim extra dots, filter empty segments
        segs = [seg.strip() for seg in right.strip().strip(".").split(".") if seg.strip()]
        if segs:
            return f"{base}:{'.'.join(segs)}"
    return base


def exclude_files(files: List[str]) -> List[str]:
    """
    Filter out irrelevant files from the given list of file paths, including:
    - Directories such as tests, examples, docs, scripts, etc.
    - Specific build/meta files such as __init__.py, __version__.py, setup.py, etc.
    - Hidden files (e.g., .gitignore, .env)
    """

    # Common irrelevant directories or prefix keywords
    exclude_keywords = {
        "docs", "doc",
        "scripts", "script",
        "example", "examples",
        "benchmark", "benchmarks",
        "tests", "testing", "test_",
        "resource", "resources",
        "sandbox", "demo", "demos"
    }

    # Explicitly excluded file names (exact match)
    exclude_exact_files = {
        "__init__.py",
        "__version__.py",
        "__main__.py",
        "setup.py",
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        "MANIFEST.in",
        "LICENSE",
        "COPYING",
        "CONTRIBUTING.md",
        "README.md",
        "readme.md",
    }

    def should_exclude(path: str) -> bool:
        # Normalize path separators
        parts = path.replace("\\", "/").split("/")
        for part in parts:
            part_lower = part.lower().strip()

            # Exclude hidden files or directories
            if part_lower.startswith("."):
                return True

            # Exclude specific file names
            if part_lower in exclude_exact_files:
                return True

            # Exclude directories/files containing certain keywords
            if any(
                part_lower == keyword or part_lower.startswith(keyword)
                for keyword in exclude_keywords
            ):
                return True

        return False

    # Return the filtered file list
    return [f for f in files if should_exclude(f)]