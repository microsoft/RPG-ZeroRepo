"""Shell-free adapters for trusted Windows npm installs of Claude and Copilot.

Layout evidence supplied from the public, versioned unpkg package.json endpoints:
  @anthropic-ai/claude-code@2.1.278: claude -> bin/claude.exe
  @anthropic-ai/claude-code@2.0.0:   claude -> cli.js
  @github/copilot@1.0.87:           copilot -> npm-loader.js
  @github/copilot@0.0.369:          copilot -> index.js

These versions document layouts, not version pins or publisher authentication.
The external installation directories and their contents remain user-trusted;
local metadata checks cannot detect a tampered installation. Copilot's loader
and its dependencies are part of that trust boundary. In particular, this
module does not infer a native Copilot sibling dependency from its filename.
Claude links escaping its package root are also deliberately unsupported.

Only the standard library is used, with no import-time I/O or policy imports.
No wrappers are read, parsed, or executed, and no process is launched here.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath, PureWindowsPath


_PACKAGES = {
    "claude": ("@anthropic-ai/claude-code", frozenset({"bin/claude.exe", "cli.js"})),
    "copilot": ("@github/copilot", frozenset({"npm-loader.js", "index.js"})),
}
_MAX_MANIFEST_BYTES = 64 * 1024


def _comparison_path(path: PurePath) -> PurePath:
    # Windows resolve() can add an extended-length prefix to only one side of
    # a containment check. Normalize the spelling for comparison, not for I/O.
    if isinstance(path, PureWindowsPath):
        spelling = str(path)
        if spelling[:8].lower() == "\\\\?\\unc\\":
            return PureWindowsPath("\\\\" + spelling[8:])
        if spelling.startswith("\\\\?\\"):
            return PureWindowsPath(spelling[4:])
    return path


def _is_within(path: PurePath, parent: PurePath) -> bool:
    return _comparison_path(path).is_relative_to(_comparison_path(parent))


def _is_excluded(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(_is_within(path, root) for root in roots)


def _safe_path(
    path: Path, roots: tuple[Path, ...], *, parent: Path | None = None
) -> Path | None:
    """Check both the supplied spelling and the strict canonical destination."""
    try:
        if not path.is_absolute() or _is_excluded(path, roots):
            return None
        if parent is not None and not _is_within(path, parent):
            return None
        canonical = path.resolve(strict=True)
        if _is_excluded(canonical, roots):
            return None
        if parent is not None and not _is_within(canonical, parent):
            return None
        return canonical
    except (OSError, RuntimeError, ValueError):
        # Missing/inaccessible paths, symlink loops, and invalid path spellings.
        return None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


def _entry_name(
    manifest: Path, provider: str, package_name: str, allowed: frozenset[str]
) -> str | None:
    try:
        if not manifest.is_file():
            return None
        with manifest.open("rb") as stream:
            raw = stream.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            return None
        metadata = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(metadata, dict) or metadata.get("name") != package_name:
            return None
        bins = metadata.get("bin")
        if not isinstance(bins, dict) or set(bins) != {provider}:
            return None
        entry = bins[provider]
        if not isinstance(entry, str) or entry not in allowed:
            return None
        return entry
    except (OSError, ValueError, RecursionError):
        return None


def _node_executable(directories: list[Path], roots: tuple[Path, ...]) -> Path | None:
    for directory in directories:
        node = _safe_path(directory / "node.exe", roots)
        try:
            # A node.exe symlink must not turn the returned command into a shell
            # wrapper. The native executable's contents are still user-trusted.
            if node is not None and node.suffix.lower() == ".exe" and node.is_file():
                return node
        except OSError:
            continue
    return None


def resolve_windows_argv(
    provider: str, directories: list[Path], excluded_roots: tuple[Path, ...]
) -> list[str] | None:
    """Return only an absolute executable and, for Node, its fixed JS entry.

    The caller first searches ALL trusted directories for direct native provider
    executables (for every provider). This fallback supports only the two npm
    packages above. The caller appends PROVIDER_ARGV[provider][1:] afterwards.

    ``directories`` are ordered, absolute, canonical, user-trusted search paths;
    ``excluded_roots`` are absolute workspace/current-directory roots. Defensive
    checks discard unsafe directories and also check canonical root aliases.
    npm packages live immediately under each directory's node_modules. Node is
    searched only as node.exe in the same supplied directories, including the
    npm prefix itself, never via PATH, cwd, a parent directory, npm, or npx.

    Native Claude needs no Node installation. JavaScript entries require a safe
    node.exe; missing/invalid candidates are skipped and ultimately return None.
    This is filesystem/metadata validation, not executable or publisher
    verification, and does not secure an attacker-writable trusted installation.
    The function is OS-independent for testing; the caller selects it on Windows.
    """
    if not isinstance(provider, str) or provider not in _PACKAGES:
        return None

    roots: list[Path] = []
    try:
        for root in excluded_roots:
            if not root.is_absolute():
                return None
            roots.extend((root, root.resolve()))
    except (OSError, RuntimeError, ValueError):
        # Do not proceed if the exclusion boundary cannot be established.
        return None
    excluded = tuple(roots)

    safe_directories: list[Path] = []
    for directory in directories:
        canonical = _safe_path(directory, excluded)
        try:
            if canonical is not None and canonical.is_dir():
                safe_directories.append(canonical)
        except OSError:
            continue

    package_name, allowed = _PACKAGES[provider]
    for directory in safe_directories:
        package = _safe_path(directory / "node_modules" / package_name, excluded)
        if package is None:
            continue
        try:
            if not package.is_dir():
                continue
            manifest = _safe_path(package / "package.json", excluded, parent=package)
            if manifest is None:
                continue
            name = _entry_name(manifest, provider, package_name, allowed)
            if name is None:
                continue
            entry = _safe_path(package / name, excluded, parent=package)
            if entry is None or not entry.is_file():
                continue
            # Preserve the executable/script kind even through an internal link.
            if entry.suffix.lower() != Path(name).suffix:
                continue
            if name == "bin/claude.exe":
                return [str(entry)]
            node = _node_executable(safe_directories, excluded)
            if node is not None:
                return [str(node), str(entry)]
        except (OSError, RuntimeError, ValueError):
            continue
    return None
