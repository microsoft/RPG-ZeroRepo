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
    """Return ``codes`` with blank and duplicate (stripped) blocks removed.

    Order of first appearance is preserved. Comparison is on the
    whitespace-stripped block so trivially different indentation does not
    defeat dedup; the stripped form is returned so the join is clean.
    """
    seen: set[str] = set()
    unique: List[str] = []
    for code in codes:
        stripped = code.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)
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
