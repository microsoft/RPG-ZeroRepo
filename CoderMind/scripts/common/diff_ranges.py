"""Helpers for matching unified diffs to code entity line ranges."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def merge_line_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in ranges if start > 0 and end >= start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def changed_line_ranges_from_diff(diff: Any) -> list[tuple[int, int]]:
    if not isinstance(diff, str) or "@@" not in diff:
        return []
    ranges: list[tuple[int, int]] = []
    new_line: int | None = None
    for raw_line in diff.splitlines():
        match = _HUNK_RE.search(raw_line)
        if match:
            new_line = int(match.group(1))
            continue
        if new_line is None:
            continue
        if raw_line.startswith("\\") or raw_line.startswith("+++") or raw_line.startswith("---"):
            continue
        prefix = raw_line[:1]
        if prefix == "+":
            ranges.append((new_line, new_line))
            new_line += 1
        elif prefix == "-":
            continue
        else:
            new_line += 1
    return merge_line_ranges(ranges)


def changed_line_ranges_by_file(code_deltas: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    ranges_by_file: dict[str, list[tuple[int, int]]] = {}
    for delta in code_deltas:
        file_path = str(delta.get("file") or delta.get("path") or delta.get("after") or delta.get("before") or "")
        if not file_path:
            continue
        ranges = changed_line_ranges_from_diff(delta.get("diff"))
        if file_path in ranges_by_file:
            ranges_by_file[file_path] = merge_line_ranges(ranges_by_file[file_path] + ranges)
        else:
            ranges_by_file[file_path] = ranges
    return ranges_by_file


def line_range_from_mapping(row: Mapping[str, Any]) -> tuple[int, int] | None:
    nested = row.get("line_range") if isinstance(row.get("line_range"), Mapping) else {}
    start = _to_int(nested.get("start") if nested else None)
    end = _to_int(nested.get("end") if nested else None)
    if start is None:
        start = _to_int(
            row.get("line_start")
            or row.get("start_line")
            or row.get("lineno")
            or row.get("line")
            or row.get("start")
        )
    if end is None:
        end = _to_int(row.get("line_end") or row.get("end_line") or row.get("end"))
    if start is None:
        return None
    return (start, end if end is not None else start)


def line_range_overlaps_any(line_range: tuple[int, int] | None, ranges: Sequence[tuple[int, int]]) -> bool:
    if line_range is None or not ranges:
        return False
    start, end = line_range
    return any(start <= range_end and end >= range_start for range_start, range_end in ranges)


def row_overlaps_changed_lines(
    row: Mapping[str, Any],
    file_path: str,
    ranges_by_file: Mapping[str, Sequence[tuple[int, int]]],
) -> bool:
    return line_range_overlaps_any(line_range_from_mapping(row), ranges_by_file.get(file_path, []))


def is_file_level_node(node_id: Any, row: Mapping[str, Any], file_path: str) -> bool:
    node_id_text = str(node_id or "")
    node_type = str(row.get("type") or row.get("kind") or "").lower()
    if node_type in {"file", "module"}:
        return True
    return node_id_text == file_path or (":" not in node_id_text and str(row.get("path") or row.get("file") or row.get("module") or "") == file_path)
