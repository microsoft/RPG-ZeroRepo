"""Home-directory workspace storage layout for CoderMind.

Replaces the legacy ``workspace/.cmind/{data,logs,.git}`` layout with a
centralised one rooted at ``~/.cmind/``:

    ~/.cmind/
        workspaces/<workspace-id>/
            .meta.toml          {workspace_path, channel, created_at, last_seen_at}
            .git/               inner git snapshot repo
            data/               rpg.json, dep_graph.json
            logs/               *.log

The workspace itself retains only two minimal items::

    <workspace>/.cmind/
        config.toml             AI configuration (team-shared, committed)
        reports/                user-facing reports (e.g. rpg.html)

Workspace identity
------------------

Each workspace is identified by a **path-derived slug** (the resolved
absolute path with non-alphanumeric runs collapsed to ``-``).  Short
paths produce a readable id like ``home-hys-projects-myrepo``; paths
whose slug would exceed 200 characters are truncated and given a
6-character base36 SHA-256 suffix so the id fits comfortably under
POSIX ``NAME_MAX`` (255).  Same shape as Claude Code's
``~/.claude/projects/`` directory naming, with two improvements: no
leading dash, and a deterministic overflow strategy instead of relying
on the OS to error out.

Slug collisions are theoretically possible (e.g. ``/foo/bar`` and
``/foo-bar`` both slug to ``foo-bar``); they are detected at read time
by comparing ``workspace_path`` recorded in ``.meta.toml``; a
collision aborts cleanly rather than silently overwriting state.

Why a readable slug and not a flat hash?  Users routinely browse
``~/.cmind/workspaces/`` to find logs, delete stale state, or sanity-
check which workspace a process is talking to; a slug makes that ten
times easier than an opaque hex hash.  We accept a small (negligible
in practice) collision risk in exchange.

For backward compatibility, when no slug-named directory exists,
:func:`home_workspace_dir` falls back to the pre-0.1.4 12-char hex
hash layout if one is present on disk.

Resolution
----------

The "workspace root" is discovered by walking up from the caller's
current directory looking for the marker ``.cmind/config.toml``.  Both
the MCP server and ``cmind script <name>`` use the same logic so a user
who ``cd``-s into any subdirectory of a workspace gets the right home
directory automatically.

Public surface
--------------

* :func:`workspace_id` - the slug (or slug+hash suffix) for a workspace path.
* :func:`home_workspace_dir` - ``~/.cmind/workspaces/<workspace_id>/``.
* :func:`workspace_data_dir`, :func:`workspace_logs_dir`,
  :func:`workspace_inner_git_dir`, :func:`workspace_reports_dir` -
  convenience wrappers for the four canonical subdirectories.
* :func:`ensure_workspace_storage` - idempotent: creates the home
  layout and writes/updates ``.meta.toml``.
* :func:`find_workspace_root_from` - walks up from a starting path
  looking for the workspace marker.
* :func:`read_meta`, :func:`write_meta` - typed accessors for
  ``.meta.toml``.

Design constraints
------------------

* No symlinks are created in the workspace (avoids Windows headaches
  and accidental backup-tool double-counting).
* All path inputs are run through :py:meth:`Path.resolve` so symlinked
  workspace roots map to a single canonical id.
* All filesystem mutations are best-effort idempotent so re-running
  ``cmind init`` or ``cmind update`` is safe.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for older Pythons
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Subdirectory of the user's home where cmind keeps all per-workspace data.
HOME_ROOT_RELPATH = Path(".cmind") / "workspaces"

#: Marker file inside the workspace that identifies it as a cmind
#: workspace. ``cmind init`` writes this; cwd-walk-up looks for it.
WORKSPACE_MARKER_RELPATH = Path(".cmind") / "config.toml"

#: Standard subdirectories created under each home workspace dir.
_DATA_SUBDIR = "data"
_LOGS_SUBDIR = "logs"
_INNER_GIT_SUBDIR = ".git"
_META_FILENAME = ".meta.toml"

#: Reports directory inside the workspace (small, user-facing artefacts
#: like ``rpg.html``).
WORKSPACE_REPORTS_SUBDIR = Path(".cmind") / "reports"

#: Channel values written to ``.meta.toml``.
CHANNEL_BUNDLE = "bundle"
CHANNEL_LEGACY = "legacy"
_VALID_CHANNELS = (CHANNEL_BUNDLE, CHANNEL_LEGACY)


# ---------------------------------------------------------------------------
# Workspace ID + path resolution
# ---------------------------------------------------------------------------

def _resolve(path: Path) -> Path:
    """Return the canonical absolute form of ``path``.

    Symlinks are followed so that ``/home/user/proj`` and the underlying
    ``/data/proj`` map to the same workspace ID.  We always operate on
    the resolved path internally; callers shouldn't have to think about
    it.
    """
    return Path(path).resolve()


# Tunables for :func:`workspace_id`.  Picked so the worst-case id
# (truncated slug + ``-`` + 6 base36 chars) stays under ``NAME_MAX``
# (255 on Linux/macOS, 255 UTF-16 code units on Windows NTFS) with
# plenty of headroom for downstream nested paths.
_SLUG_MAX_LEN = 200
_HASH_SUFFIX_LEN = 6
_LEGACY_HASH_LEN = 12

#: Pattern used by :func:`_slugify` to collapse non-alphanumeric runs.
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")

#: Pattern matching the pre-0.1.4 legacy id format (12 lowercase hex chars).
_LEGACY_HASH_RE = re.compile(r"^[0-9a-f]{12}$")

#: Base36 alphabet used for the overflow hash suffix.  Matches Claude
#: Code's convention for ``~/.claude/projects/`` and packs ~5 bits per
#: char (36**6 ≈ 2.2e9 — collision probability negligible at our scale).
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _slugify_path(workspace_path: Path) -> str:
    """Return the slug (no hash) for a resolved workspace path.

    Algorithm: replace every run of non-alphanumeric chars with ``-``,
    strip leading/trailing ``-``, lowercase.  Result is always safe to
    use as a directory name on every filesystem we target.

    Examples:
        ``/home/hys/projects/cmind`` -> ``home-hys-projects-cmind``
        ``C:\\Users\\foo\\bar``       -> ``c-users-foo-bar``
        ``/``                         -> ``root``
    """
    canonical = str(_resolve(workspace_path))
    slug = _NON_ALNUM_RE.sub("-", canonical).strip("-").lower()
    return slug or "root"


def _base36_hash(workspace_path: Path) -> str:
    """Return the 6-char base36 SHA-256 prefix used for overflow ids."""
    canonical = str(_resolve(workspace_path))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    n = int.from_bytes(digest[: _HASH_SUFFIX_LEN], "big")
    out = []
    for _ in range(_HASH_SUFFIX_LEN):
        out.append(_BASE36_ALPHABET[n % 36])
        n //= 36
    return "".join(reversed(out))


def _legacy_workspace_id(workspace_path: Path) -> str:
    """Compute the pre-0.1.4 workspace id (SHA-256 first 12 hex chars).

    Only used by :func:`home_workspace_dir` as a backward-compat probe
    so users who upgraded across the slug-naming change keep finding
    their existing on-disk state.
    """
    canonical = str(_resolve(workspace_path))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_LEGACY_HASH_LEN]


def workspace_id(workspace_path: Path) -> str:
    """Compute the workspace identifier for ``workspace_path``.

    Two formats:

    * **Short (preferred)** — when the path slug is ≤ 200 chars, use
      the slug verbatim, e.g. ``home-hys-projects-cmind``.  Readable
      at a glance; lets users browse ``~/.cmind/workspaces/`` and
      identify their projects without cross-referencing a hash table.
    * **Truncated (overflow)** — when the slug exceeds the budget,
      keep the first ~193 chars and append ``-<hash6>`` where
      ``hash6`` is a 6-character base36 SHA-256 prefix.  Guarantees
      uniqueness across arbitrarily long paths while staying under
      ``NAME_MAX`` (255).

    The identifier is deterministic on a given machine: the same
    resolved absolute path always yields the same id.  Different
    paths (including different clones of the same git repo) yield
    different ids — this is intentional so each clone has independent
    state.
    """
    slug = _slugify_path(workspace_path)
    if len(slug) <= _SLUG_MAX_LEN:
        return slug
    # Reserve room for ``-`` + hash suffix.
    head = slug[: _SLUG_MAX_LEN - _HASH_SUFFIX_LEN - 1].rstrip("-")
    return f"{head}-{_base36_hash(workspace_path)}"


# ---------------------------------------------------------------------------
# Home-side path helpers
# ---------------------------------------------------------------------------

def home_root() -> Path:
    """Return ``~/.cmind/workspaces/``.

    Does not create the directory; callers should use
    :func:`ensure_workspace_storage` when they need it to exist.
    """
    return Path.home() / HOME_ROOT_RELPATH


def home_workspace_dir(workspace_path: Path) -> Path:
    """Return the home directory assigned to ``workspace_path``.

    Normally this is ``~/.cmind/workspaces/<workspace_id>/`` using the
    slug-based id from :func:`workspace_id`.

    Backward compatibility: if a directory under the **legacy** 12-char
    hex id already exists on disk (created by cmind < 0.1.4) and no
    slug-named directory exists for the same path, the legacy directory
    is returned so the user keeps reaching their existing state after
    upgrading.  New workspaces always use the slug-based layout.

    The directory may or may not exist on disk.
    """
    new_dir = home_root() / workspace_id(workspace_path)
    if new_dir.exists():
        return new_dir
    legacy_dir = home_root() / _legacy_workspace_id(workspace_path)
    if legacy_dir.exists():
        return legacy_dir
    return new_dir


def workspace_data_dir(workspace_path: Path) -> Path:
    return home_workspace_dir(workspace_path) / _DATA_SUBDIR


def workspace_logs_dir(workspace_path: Path) -> Path:
    return home_workspace_dir(workspace_path) / _LOGS_SUBDIR


def workspace_inner_git_dir(workspace_path: Path) -> Path:
    """Return the path of the inner-git ``.git/`` directory.

    Note: this is the GIT_DIR itself (a directory named ``.git`` sitting
    inside the home workspace dir).  Callers using ``git -C ...`` should
    pass :func:`home_workspace_dir`; callers using ``--git-dir`` should
    pass this path.
    """
    return home_workspace_dir(workspace_path) / _INNER_GIT_SUBDIR


def workspace_meta_path(workspace_path: Path) -> Path:
    return home_workspace_dir(workspace_path) / _META_FILENAME


def workspace_reports_dir(workspace_path: Path) -> Path:
    """Return the workspace-local ``reports/`` directory.

    This is in the workspace (not home) because reports are small,
    user-facing artefacts users may want to commit or browse alongside
    the source code.
    """
    return _resolve(workspace_path) / WORKSPACE_REPORTS_SUBDIR


# ---------------------------------------------------------------------------
# Marker discovery (cwd-walk-up)
# ---------------------------------------------------------------------------

def _is_live_workspace_root(root: Path) -> bool:
    """Return True iff a candidate workspace root is still live.

    A bare ``.cmind/config.toml`` is enough for a *fresh* workspace
    (the marker may be planted before any home-side state is written),
    so the marker alone is treated as live until proven stale.

    Staleness is detected only when ``.meta.toml`` is present: a moved
    or renamed workspace records its original absolute path there, and
    if that recorded path no longer matches the candidate directory the
    marker is treated as stale.  This guards :func:`find_workspace_root_from`
    against climbing into a renamed parent and misrouting reads/writes,
    while still allowing brand-new (marker-only) workspaces to be
    discovered before they have any home-side state.
    """
    meta = read_meta(root)
    if meta is not None:
        recorded = meta.get("workspace_path")
        if isinstance(recorded, str) and Path(recorded) != _resolve(root):
            return False
    return True


def find_workspace_root_from(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` (default: cwd) looking for an cmind workspace.

    A directory qualifies as a workspace if it contains
    ``.cmind/config.toml`` (see :data:`WORKSPACE_MARKER_RELPATH`)
    **and** passes :func:`_is_live_workspace_root` — i.e. either it
    has no ``.meta.toml`` (fresh workspace), or the recorded
    ``workspace_path`` in meta still matches.  Stale (moved/renamed)
    markers on parent directories are skipped, so the walker continues
    climbing rather than misrouting into a different workspace's state.

    Returns the **resolved** path of the workspace root, or ``None``
    when no live marker is found before reaching the filesystem root.
    """
    cur = _resolve(start if start is not None else Path.cwd())
    while True:
        if (cur / WORKSPACE_MARKER_RELPATH).is_file() and _is_live_workspace_root(cur):
            return cur
        if cur.parent == cur:  # reached / (POSIX) or drive root (Windows)
            return None
        cur = cur.parent


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """ISO-8601 timestamp in UTC, second precision (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _toml_escape(value: str) -> str:
    """Minimal TOML basic-string escape (sufficient for our values).

    Handles the chars TOML's basic-string syntax actually forbids or
    requires escaping: backslash, double quote, and the control chars
    that are common in pathological inputs (newline, carriage return,
    tab).  Other control chars (NUL, vertical tab, etc.) would also be
    invalid but are vanishingly unlikely in our inputs (paths +
    version strings); we accept the small remaining risk rather than
    pull in a full TOML writer dependency.
    """
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def read_meta(workspace_path: Path) -> Optional[Dict[str, Any]]:
    """Read ``.meta.toml`` for ``workspace_path`` or return ``None``.

    Returns ``None`` if the file doesn't exist or fails to parse.
    A separate parse-failure return value isn't useful in practice -
    every caller treats both cases as "no metadata yet".
    """
    meta_file = workspace_meta_path(workspace_path)
    if not meta_file.is_file():
        return None
    try:
        with open(meta_file, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def write_meta(
    workspace_path: Path,
    *,
    channel: str,
    cmind_cli_version: Optional[str] = None,
    preserve_created_at: bool = True,
) -> None:
    """Atomically write the workspace's ``.meta.toml``.

    Args:
        workspace_path: The workspace directory (resolved internally).
        channel: ``"bundle"`` or ``"legacy"`` -- which provisioning
            channel was used.
        cmind_cli_version: The installed cmind-cli version at write
            time.  Stored as ``cmind_cli_version_at_init`` (only on
            first write) and ``cmind_cli_version_last_seen`` (every
            write).
        preserve_created_at: When True (the default), keep the original
            ``created_at`` from any existing meta file; otherwise
            overwrite with ``utc_now()``.

    Raises:
        ValueError: if ``channel`` is not a recognised value.
        OSError: if the file can't be written.
    """
    if channel not in _VALID_CHANNELS:
        raise ValueError(
            f"channel must be one of {_VALID_CHANNELS!r}, got {channel!r}"
        )

    resolved = _resolve(workspace_path)
    meta_file = workspace_meta_path(workspace_path)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    existing = read_meta(workspace_path) or {}
    now = _utc_now_iso()
    if preserve_created_at:
        created_at = existing.get("created_at", now)
        # On preserve, also carry forward the version recorded at init
        # so re-running ``cmind update`` doesn't blow away that history.
        init_version = existing.get(
            "cmind_cli_version_at_init", cmind_cli_version or ""
        )
    else:
        # "Reset" semantics: created_at and init_version both refresh
        # to the values supplied in this call.
        created_at = now
        init_version = cmind_cli_version or ""

    # Serialise by hand - tiny + avoids a TOML writer dep.
    lines = [
        "# CoderMind per-workspace state. Managed by `cmind init/update`.",
        "# Do not commit; recreated automatically if missing.",
        "",
        f'workspace_path = "{_toml_escape(str(resolved))}"',
        f'channel = "{channel}"',
        f'created_at = "{created_at}"',
        f'last_seen_at = "{now}"',
    ]
    if init_version:
        lines.append(f'cmind_cli_version_at_init = "{_toml_escape(init_version)}"')
    if cmind_cli_version:
        lines.append(
            f'cmind_cli_version_last_seen = "{_toml_escape(cmind_cli_version)}"'
        )
    payload = "\n".join(lines) + "\n"

    # Atomic write: .tmp + os.replace
    tmp = meta_file.with_suffix(".toml.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, meta_file)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# Layout bootstrap + integrity check
# ---------------------------------------------------------------------------

class WorkspaceMetaMismatch(RuntimeError):
    """Raised when an existing ``.meta.toml`` points at a different path.

    This indicates either a hash collision (statistically very rare for
    a 48-bit truncated hash on a single machine, but possible) or a
    user manually moving directories under ``~/.cmind/``.  We never
    silently mix two workspaces' data; the user must investigate.
    """


def ensure_workspace_storage(
    workspace_path: Path,
    *,
    channel: str,
    cmind_cli_version: Optional[str] = None,
) -> Path:
    """Create the home layout for ``workspace_path`` (idempotent).

    Creates::

        ~/.cmind/workspaces/<workspace-id>/
            data/
            logs/

    Writes ``.meta.toml`` capturing the workspace path, channel, and
    timestamps.  If an existing ``.meta.toml`` records a *different*
    workspace path (hash collision or manual rename), raises
    :class:`WorkspaceMetaMismatch` -- callers must surface this clearly
    rather than overwriting another workspace's data.

    The inner ``.git/`` directory is NOT created here; that's the
    responsibility of :mod:`cmind_cli._inner_git`, which knows how to
    seed an initial commit message.

    Returns:
        The home workspace directory (``~/.cmind/workspaces/<workspace-id>/``).
    """
    resolved = _resolve(workspace_path)
    home_dir = home_workspace_dir(resolved)

    # Hash-collision / rename guard.
    existing = read_meta(resolved)
    if existing is not None:
        recorded = existing.get("workspace_path")
        if isinstance(recorded, str) and Path(recorded).resolve() != resolved:
            raise WorkspaceMetaMismatch(
                f"Workspace hash collision at {home_dir}: meta points to "
                f"{recorded!r} but caller passed {str(resolved)!r}. "
                f"Resolve manually (e.g., move or delete the offending "
                f"directory) before retrying."
            )

    (home_dir / _DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
    (home_dir / _LOGS_SUBDIR).mkdir(parents=True, exist_ok=True)
    workspace_reports_dir(resolved).mkdir(parents=True, exist_ok=True)

    write_meta(resolved, channel=channel, cmind_cli_version=cmind_cli_version)
    return home_dir


# ---------------------------------------------------------------------------
# Convenience: resolve from cwd in one step
# ---------------------------------------------------------------------------

def resolve_data_from_cwd(start: Optional[Path] = None) -> Optional[Path]:
    """Find the workspace from ``start`` and return its data directory.

    Convenience for scripts that just want the canonical
    ``data/rpg.json`` location without manually chaining
    :func:`find_workspace_root_from` and :func:`workspace_data_dir`.
    Returns ``None`` if no workspace is found.
    """
    root = find_workspace_root_from(start)
    if root is None:
        return None
    return workspace_data_dir(root)
