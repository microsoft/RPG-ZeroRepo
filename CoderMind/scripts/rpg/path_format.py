"""Canonical RPG node path format — single source of truth.

All RPG node ``meta.path`` strings should be constructed via the helpers
in this module to keep formats uniform across encoder, code generation,
incremental update, and design pipelines.

Format::

    FILE       :  "rel/posix/path.py"
    DIRECTORY  :  "rel/posix/dir"
    FUNCTION   :  "rel/posix/path.py::name"
    CLASS      :  "rel/posix/path.py::Name"
    METHOD     :  "rel/posix/path.py::Class::method"

Disambiguation of kind (function vs class) is in ``NodeMetaData.type_name``,
NOT in the path itself.  This keeps path strings canonical and avoids
duplicating kind prefixes such as ``::class Foo`` / ``::function bar``.

Dep-graph nodes use a related but distinct convention
(``"foo.py:Class.method"`` with a single colon and dot separator); the
``to_dep_graph_id`` / ``from_dep_graph_id`` helpers convert between the
two when cross-graph lookups are needed.
"""

from pathlib import PurePosixPath
from typing import List, Tuple

__all__ = [
    "file_node_path",
    "function_node_path",
    "class_node_path",
    "method_node_path",
    "parse_node_path",
    "to_dep_graph_id",
    "from_dep_graph_id",
    "desc_key_function",
    "desc_key_class",
    "desc_key_method",
]


def _norm_file(rel_path: str) -> str:
    """Normalize a file/directory relative path to POSIX form.

    - Strips leading ``./`` or ``/``
    - Returns ``.`` for empty / root inputs
    - Preserves ``..`` segments (no resolution against filesystem)
    """
    s = (rel_path or "").strip()
    if not s:
        return "."
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")
    if not s:
        return "."
    return PurePosixPath(s).as_posix()


def file_node_path(rel_path: str) -> str:
    """Build a canonical FILE / DIRECTORY node path."""
    return _norm_file(rel_path)


def function_node_path(rel_path: str, name: str) -> str:
    """Build a canonical FUNCTION node path."""
    return f"{_norm_file(rel_path)}::{name}"


def class_node_path(rel_path: str, name: str) -> str:
    """Build a canonical CLASS node path."""
    return f"{_norm_file(rel_path)}::{name}"


def method_node_path(rel_path: str, class_name: str, method_name: str) -> str:
    """Build a canonical METHOD node path."""
    return f"{_norm_file(rel_path)}::{class_name}::{method_name}"


def parse_node_path(p: str) -> Tuple[str, List[str]]:
    """Split a node path into ``(file_path, symbol_parts)``.

    Examples::

        "foo.py"                  -> ("foo.py", [])
        "foo.py::bar"             -> ("foo.py", ["bar"])
        "foo.py::Cls"             -> ("foo.py", ["Cls"])
        "foo.py::Cls::m"          -> ("foo.py", ["Cls", "m"])
    """
    if not p or "::" not in p:
        return (p or ""), []
    f, _, s = p.partition("::")
    if not s:
        return f, []
    return f, s.split("::")


# ---------------------------------------------------------------------------
# Dep-graph interop
# ---------------------------------------------------------------------------

def to_dep_graph_id(p: str) -> str:
    """Convert a canonical feature-graph path to a dep-graph node id.

    ``"foo.py::Cls::m"`` -> ``"foo.py:Cls.m"``
    ``"foo.py::bar"``    -> ``"foo.py:bar"``
    ``"foo.py"``         -> ``"foo.py"``
    """
    file, parts = parse_node_path(p)
    if not parts:
        return file
    return f"{file}:" + ".".join(parts)


def from_dep_graph_id(p: str) -> str:
    """Convert a dep-graph node id to a canonical feature-graph path.

    ``"foo.py:Cls.m"`` -> ``"foo.py::Cls::m"``
    ``"foo.py:bar"``   -> ``"foo.py::bar"``
    ``"foo.py"``       -> ``"foo.py"``

    Idempotent on already-canonical inputs.
    """
    if not p or ":" not in p or "::" in p:
        return p
    file, _, sym = p.partition(":")
    if not sym:
        return file
    return f"{file}::" + "::".join(sym.split("."))


# ---------------------------------------------------------------------------
# Feature-description composite keys
# ---------------------------------------------------------------------------
#
# ``ParseFeatures`` stores LLM-generated descriptions in a per-file sidecar
# ``_feature_descriptions_`` mapping.  The keys are composite identifiers
# built from the unit name + feature name so that the same feature text can
# appear under multiple units of the same file without colliding.
#
# Both the producer (``rpg_encoder.semantic_parsing``) and the consumers
# (``rpg.models.RPG.update_from_parsed_tree``, ``refactor_tree._init_feature_tree``)
# MUST agree on the exact key shape.  These helpers are the single source
# of truth — if you change the format here, every consumer automatically
# follows.
#
# Format (matches the canonical node path's symbol-chain layout):
#   FUNCTION feature       :  "{func_name}::{feat}"
#   CLASS-level feature    :  "{class_name}::{feat}"
#   METHOD feature         :  "{class_name}::{method_name}::{feat}"
#
# IMPORTANT: callers MUST pass the *normalized* feature name (with ``/``
# already replaced by ``" or "``) so the key matches the name stored on
# the Node and read at lookup time.

def desc_key_function(func_name: str, feat: str) -> str:
    """Composite description-map key for a top-level function feature."""
    return f"{func_name}::{feat}"


def desc_key_class(class_name: str, feat: str) -> str:
    """Composite description-map key for a class-level feature."""
    return f"{class_name}::{feat}"


def desc_key_method(class_name: str, method_name: str, feat: str) -> str:
    """Composite description-map key for a method-level feature."""
    return f"{class_name}::{method_name}::{feat}"
