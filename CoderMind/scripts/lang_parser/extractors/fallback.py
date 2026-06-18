from __future__ import annotations

import re
from typing import Iterable

from ..models import LPCodeUnit, LPDependency


_IDENTIFIER = r"[A-Za-z_$][\w$]*"


def strip_comments(line: str) -> str:
    text = line.strip()
    if text.startswith("//") or text.startswith("#"):
        return ""
    return line


def source_slice(lines: list[str], line_start: int | None, line_end: int | None) -> str:
    if line_start is None or line_end is None:
        return ""
    if line_start < 1 or line_end < line_start:
        return ""
    return "\n".join(lines[line_start - 1:line_end])


def block_end_for_braces(lines: list[str], start_index: int) -> int:
    depth = 0
    saw_open = False
    for index in range(start_index, len(lines)):
        line = strip_string_literals(lines[index])
        depth += line.count("{")
        if "{" in line:
            saw_open = True
        depth -= line.count("}")
        if saw_open and depth <= 0:
            return index
    return start_index


def line_end_for_statement(lines: list[str], start_index: int) -> int:
    for index in range(start_index, len(lines)):
        stripped = lines[index].strip()
        if stripped.endswith((";", ")", "}")) or index == len(lines) - 1:
            return index
    return start_index


def strip_string_literals(line: str) -> str:
    # Exclude backslashes from the normal-character branch so escapes have a
    # single matching path.  Otherwise, unterminated quoted text with many
    # escapes can trigger catastrophic backtracking.
    return re.sub(r"(['\"`])(?:\\.|(?!\1)[^\\])*\1", "", line)


def delimiter_syntax_error(source: str) -> str | None:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairs.items()}
    stack: list[tuple[str, int]] = []
    in_string: str | None = None
    escaped = False
    in_block_comment = False

    for line_number, line in enumerate(source.splitlines(), start=1):
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""
            # Block comments (/* ... */) may span multiple lines and frequently
            # contain apostrophes (e.g. "doesn't", "store's") or quotes that
            # would otherwise be mistaken for character/string literals.
            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    index += 2
                    continue
                index += 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                index += 1
                continue
            if char == "/" and next_char == "*":
                in_block_comment = True
                index += 2
                continue
            if char == "/" and next_char == "/":
                break
            if char == "'":
                # A single quote is ambiguous: it may open a C/C++/Go char
                # literal ('x', '\n', '\''), a Rust lifetime ('a, 'static), or
                # appear in prose the scanner already routed here.  Only consume
                # a balanced char literal; otherwise treat the quote as an inert
                # character so lifetimes/apostrophes do not trigger a spurious
                # "unterminated string literal".
                consumed = _char_literal_length(line, index)
                if consumed:
                    index += consumed
                    continue
                index += 1
                continue
            if char in {'"', "`"}:
                in_string = char
            elif char in pairs:
                stack.append((char, line_number))
            elif char in closing:
                if not stack or stack[-1][0] != closing[char]:
                    return f"Unmatched delimiter {char!r} at line {line_number}"
                stack.pop()
            index += 1
        # Double-quoted strings do not span physical lines unless the line ends
        # with a backslash continuation; reset a dangling quote so a genuinely
        # malformed line does not cascade into later lines.  Backtick template
        # literals (JS/TS) legitimately span lines, so leave them open.
        if in_string == '"' and not line.endswith("\\"):
            in_string = None
            escaped = False

    if in_block_comment:
        return "Unterminated block comment"
    if in_string:
        return "Unterminated string literal"
    if stack:
        char, line_number = stack[-1]
        return f"Unclosed delimiter {char!r} opened at line {line_number}"
    return None


def _char_literal_length(line: str, start: int) -> int:
    """Return the length of a balanced char literal at ``line[start]`` or 0.

    ``start`` must point at the opening ``'``.  Recognizes ``'x'`` and escaped
    forms such as ``'\\n'`` / ``'\\''``.  Returns 0 when the quote is not a
    closed single-character literal (e.g. a Rust lifetime or a stray
    apostrophe), signalling the caller to treat it as an inert character.
    """
    if start >= len(line) or line[start] != "'":
        return 0
    index = start + 1
    if index >= len(line):
        return 0
    if line[index] == "\\":
        index += 2  # escape sequence consumes backslash + escaped char
    else:
        index += 1
    if index < len(line) and line[index] == "'":
        return index - start + 1
    return 0


def make_unit(
    *,
    name: str | None,
    unit_type: str,
    file_path: str,
    parent: str | None,
    lines: list[str],
    line_start: int,
    line_end: int,
    language: str,
    node_type: str,
    extra: dict | None = None,
) -> LPCodeUnit:
    metadata = {
        "language": language,
        "line_start": line_start,
        "line_end": line_end,
        "node_type": node_type,
    }
    if extra:
        metadata.update(extra)
    return LPCodeUnit(
        name=name,
        unit_type=unit_type,
        file_path=file_path,
        parent=parent,
        line_start=line_start,
        line_end=line_end,
        code=source_slice(lines, line_start, line_end),
        language=language,
        extra=metadata,
    )


def dependency_from_import(
    *,
    path: str,
    module: str | None,
    symbol: str | None,
    line: int | None,
    language: str,
    import_kind: str,
) -> LPDependency:
    return LPDependency(
        src=path,
        dst=module,
        relation="imports",
        symbol=symbol,
        line=line,
        confidence="unresolved",
        extra={"language": language, "import_kind": import_kind},
    )


def class_like_label(unit_type: str) -> str:
    if unit_type == "struct":
        return "struct"
    if unit_type == "interface":
        return "interface"
    return "class"


def top_level_line_indices(lines: Iterable[str], excluded_ranges: list[tuple[int, int]]) -> Iterable[tuple[int, str]]:
    for index, line in enumerate(lines):
        if any(start <= index <= end for start, end in excluded_ranges):
            continue
        yield index, line


def ts_js_identifier_pattern() -> str:
    return _IDENTIFIER
