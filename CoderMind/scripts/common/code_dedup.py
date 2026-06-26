"""Shared helpers for collapsing duplicated interface source blocks.

Interface synthesis stores each unit's code as the whole-file text for
non-Python units (``LPCodeUnit`` has no ``count_lines`` slicing), so a
file with N units repeats the entire file N times when those blocks are
joined into ``file_code``. These helpers collapse identical blocks so the
joined source reconstructs the original single file (imports plus each
unit once) instead of an O(units x file_size) blow-up.
"""
from __future__ import annotations

from typing import Iterable, List


def dedup_code_blocks(codes: Iterable[str]) -> List[str]:
    """Return ``codes`` with blank and duplicate blocks removed.

    Order of first appearance is preserved. Duplicates are detected on the
    whitespace-stripped block so trivially different indentation does not
    defeat dedup, but each surviving block keeps its own leading indentation
    (only trailing whitespace is trimmed) so indented unit slices stay valid
    when joined into ``file_code``.
    """
    seen: set[str] = set()
    unique: List[str] = []
    for code in codes:
        key = code.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(code.rstrip())
    return unique


def dedup_file_code(unit_codes: Iterable[str], fallback: str = "") -> str:
    """Build ``file_code`` from per-unit code blocks with duplication removed.

    ``unit_codes`` are the values of ``units_to_code``. When every block is
    an identical whole-file copy, the result is that single file; when
    blocks are genuinely distinct per-unit slices they are all kept. Falls
    back to ``fallback`` when no non-empty block survives.
    """
    unique = dedup_code_blocks(unit_codes)
    if not unique:
        return fallback
    return "\n\n".join(unique)
