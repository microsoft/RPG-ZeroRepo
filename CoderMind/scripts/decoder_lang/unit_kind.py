"""Shared unit-name classification: callable vs type-like.

Several decoder stages — orphan detection most importantly — must tell
*callable* units (functions / methods whose normal use is being
**invoked**) apart from *type-like* units (structs / enums / interfaces
/ data classes that are instantiated or referenced as field / parameter
types, but never "called").

The distinction matters because the orphan heuristic is
"no incoming invocation edge => dead code". That rule only holds for
callables: a data type legitimately has no incoming *invocation* edge
even when it is used, so flagging it as an orphan is a false positive
(this is the Go ``struct Store`` / ``struct PageData`` case).

Interface units carry a leading kind token, e.g. ``"function parse"``,
``"method ServeHTTP"``, ``"struct Store"``, ``"class Parser"``. The
helpers here read that token. The default prefix sets are shared by all
current backends (Python / Go / Rust / TypeScript / JavaScript / C /
C++); a backend may pass custom sets if a language introduces a
callable construct under a different keyword.
"""
from __future__ import annotations

# Units that are normally USED BY BEING CALLED. Orphan detection
# ("no incoming edge => dead") is only meaningful for these.
#
# ``class`` is callable in every OO language the decoder targets
# (Python / JavaScript / TypeScript / C++): the constructor is invoked
# to instantiate it, and the encoder records instantiation as an
# invocation edge, so a used class reliably has an incoming edge.
# Classifying ``class`` as callable keeps dead-class detection working
# with zero false positives on the languages observed.
CALLABLE_UNIT_PREFIXES: frozenset[str] = frozenset({
    "function",
    "func",
    "method",
    "fn",
    "class",
    "constructor",
})

# Units that are TYPES: instantiated, referenced, or used as field /
# parameter types — never "invoked". A type with no incoming
# invocation edge is NOT dead code, so orphan detection must skip it.
TYPE_UNIT_PREFIXES: frozenset[str] = frozenset({
    "struct",
    "enum",
    "interface",
    "trait",
    "type",
    "union",
    "typedef",
    "record",
})


def classify_unit_kind(
    unit_name: str,
    *,
    callable_prefixes: frozenset[str] = CALLABLE_UNIT_PREFIXES,
    type_prefixes: frozenset[str] = TYPE_UNIT_PREFIXES,
) -> str:
    """Classify ``unit_name`` as ``"callable"`` / ``"type"`` / ``"unknown"``.

    ``unit_name`` is expected to carry a leading kind token
    (``"function foo"``, ``"struct Bar"``). When the token is missing or
    unrecognised the result is ``"unknown"``; callers decide how to
    treat it (orphan detection skips non-callable units, staying on the
    false-positive-reducing side).
    """
    if not unit_name:
        return "unknown"
    token = unit_name.split(" ", 1)[0].strip().lower()
    if token in callable_prefixes:
        return "callable"
    if token in type_prefixes:
        return "type"
    return "unknown"
