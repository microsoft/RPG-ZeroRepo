"""Closed AI provider policy shared by the CLI and packaged pipeline scripts.

This module uses only the standard library and has no import-time I/O. Repository
configuration can recommend a provider, never authorize execution. Only an
explicit caller, process environment or user-local workspace selection can do so.
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path, PurePath, PureWindowsPath
import tempfile
import tomllib
from typing import Mapping

_IS_WINDOWS = os.name == "nt"


class AICommandPolicyError(ValueError):
    """An AI command could not be authorized by the closed provider policy."""


PROVIDER_ARGV = {
    "copilot": ("copilot",),
    "claude": ("claude",),
    "gemini": ("gemini", "-p"),
    "qwen": ("qwen", "-p"),
    "cursor-agent": ("agent", "-p"),
    "auggie": ("augment", "-p"),
    "codex": ("codex", "exec"),
    "codebuddy": ("codebuddy", "-p"),
    "qoder": ("qodercli", "-p"),
    "opencode": ("opencode", "run"),
    "amp": ("amp", "--execute"),
}
_LEGACY_PROVIDERS = {" ".join(argv): name for name, argv in PROVIDER_ARGV.items()}


def validate_provider(value: object) -> str:
    if not isinstance(value, str) or value not in PROVIDER_ARGV:
        raise AICommandPolicyError(
            "Invalid AI provider. Select a built-in ai_provider; executable paths "
            "and additional arguments are not supported."
        )
    return value


def provider_from_command(value: object) -> str:
    """Accept only exact historical built-in commands, not shell-like syntax."""
    if not isinstance(value, str) or value not in _LEGACY_PROVIDERS:
        raise AICommandPolicyError(
            "Unsupported ai_cli_cmd. Replace it with a built-in ai_provider. "
            "Custom commands, paths and additional arguments are not supported."
        )
    return _LEGACY_PROVIDERS[value]


def read_workspace_provider(workspace: Path) -> str:
    """Validate and return a recommendation only, including legacy spellings.

    Never use this return value as execution authority or copy it to local state
    without an explicit user choice. Keeping the file preserves workspace lookup.
    """
    config = workspace / ".cmind" / "config.toml"
    try:
        with config.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError as exc:
        # A dangling symlink is an invalid configuration, not an absent one.
        if config.is_symlink():
            raise AICommandPolicyError("Cannot read workspace AI configuration.") from exc
        return ""
    except (OSError, ValueError) as exc:
        raise AICommandPolicyError("Cannot read workspace AI configuration.") from exc
    table = data.get("cmind", {})
    if not isinstance(table, dict):
        raise AICommandPolicyError("The cmind configuration must be a TOML table.")
    keys = {"recommended_provider", "ai_provider", "ai_cli_cmd"} & table.keys()
    if len(keys) > 1:
        raise AICommandPolicyError("Use only one workspace provider recommendation.")
    if "recommended_provider" in table:
        return validate_provider(table["recommended_provider"])
    if "ai_provider" in table:
        return validate_provider(table["ai_provider"])
    if "ai_cli_cmd" in table:
        return provider_from_command(table["ai_cli_cmd"])
    return ""


def _comparison_path(path: PurePath) -> PurePath:
    """Normalize Windows extended prefixes for identity/containment, not I/O."""
    if isinstance(path, PureWindowsPath):
        spelling = str(path)
        if spelling[:8].lower() == "\\\\?\\unc\\":
            return PureWindowsPath("\\\\" + spelling[8:])
        if spelling.startswith("\\\\?\\"):
            return PureWindowsPath(spelling[4:])
    return path


def _is_within(path: PurePath, root: PurePath) -> bool:
    return _comparison_path(path).is_relative_to(_comparison_path(root))


def _workspace_identity(workspace: Path) -> str:
    return os.path.normcase(str(_comparison_path(workspace.resolve())))


def local_selection_path(workspace: Path) -> Path:
    """User-owned state, separate from copyable RPG metadata and repository files.

    A full path hash plus an identity field prevents slug collisions or copied
    records from authorizing another workspace. No repository path controls this
    location. Reject redirected state directories and any store in the workspace.
    """
    home = Path.home().resolve()
    digest = hashlib.sha256(_workspace_identity(workspace).encode("utf-8")).hexdigest()
    path = home / ".cmind" / "execution" / digest / "selection.json"
    for part in (home / ".cmind", home / ".cmind" / "execution", path.parent, path):
        resolved = part.resolve()
        if (part.is_symlink() or _comparison_path(resolved) != _comparison_path(part)
            or _is_within(resolved, workspace.resolve())):
            raise AICommandPolicyError("User-local AI selection must not be redirected or stored in the workspace.")
    return path


def read_local_provider(workspace: Path) -> str:
    path = local_selection_path(workspace)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    except (OSError, ValueError) as exc:
        raise AICommandPolicyError("Cannot read user-local AI selection; explicitly select the provider again.") from exc
    if (not isinstance(data, dict)
            or set(data) != {"schema_version", "workspace", "ai_provider"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["workspace"] != _workspace_identity(workspace)):
        raise AICommandPolicyError("User-local AI selection has an invalid schema or workspace identity.")
    return validate_provider(data["ai_provider"])


def write_local_provider(workspace: Path, provider: str) -> None:
    """Persist ONLY an explicit user choice, called after successful init/update.

    This is product behavior, never run against the developer's real profile by
    tests. Atomic publication preserves the prior selection on write failure.
    """
    validate_provider(provider)
    path = local_selection_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    local_selection_path(workspace)  # Recheck after directory creation.
    data = {"schema_version": 1, "workspace": _workspace_identity(workspace), "ai_provider": provider}
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".selection-", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=True, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        local_selection_path(workspace)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def resolve_provider(
    workspace: Path,
    *,
    tool: str | None = None,
    environ: Mapping[str, str] | None = None,
    baked: str = "",
) -> str:
    """Resolve explicit caller > process/CI environment > user-local selection.

    Repository values and legacy release defaults are validated recommendations,
    never executable authority. In particular, absence of local state does not
    fall back to the cloned repository or to a release-baked provider.
    """
    read_workspace_provider(workspace)
    if baked:
        provider_from_command(baked)  # Legacy default is NOT an authorization.
    env = os.environ if environ is None else environ
    if tool is not None:
        return provider_from_command(tool)
    if "CMIND_AI_PROVIDER" in env and "CMIND_AI_CLI_CMD" in env:
        raise AICommandPolicyError("Set only one AI provider environment override.")
    if "CMIND_AI_PROVIDER" in env:
        return validate_provider(env["CMIND_AI_PROVIDER"])
    if "CMIND_AI_CLI_CMD" in env:
        return provider_from_command(env["CMIND_AI_CLI_CMD"])
    return read_local_provider(workspace)


def build_argv(provider: str, workspace: Path, *, search_path: str | None = None) -> list[str]:
    """Resolve a built-in command without shell interpretation of prompt text.

    Direct native executables take priority. On Windows, known external npm
    installations can be launched through their fixed native/Node entry points
    without running the .cmd/.ps1 shim. Arbitrary wrappers remain unsupported.
    The user's installed tools and absolute PATH directories remain trusted.
    """
    base = PROVIDER_ARGV[validate_provider(provider)]
    workspace = workspace.resolve()
    cwd = Path.cwd().resolve()
    path = os.environ.get("PATH", "") if search_path is None else search_path
    filename = base[0] + ".exe" if _IS_WINDOWS else base[0]
    directories = []
    for entry in path.split(os.pathsep):
        directory = Path(entry)
        if not entry or not directory.is_absolute():
            continue
        try:
            canonical_dir = directory.resolve(strict=True)
            if any(_is_within(directory, root) or _is_within(canonical_dir, root)
                   for root in (workspace, cwd)):
                continue
            directories.append(canonical_dir)
            candidate = (canonical_dir / filename).resolve(strict=True)
            if any(_is_within(candidate, root) for root in (workspace, cwd)):
                continue
            if _IS_WINDOWS and candidate.suffix.lower() != ".exe":
                continue
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return [str(candidate), *base[1:]]
        except (OSError, RuntimeError):
            continue
    if _IS_WINDOWS:
        from .windows_ai_cli import resolve_windows_argv

        adapted = resolve_windows_argv(provider, directories, (workspace, cwd))
        if adapted is not None:
            return [*adapted, *base[1:]]
    raise AICommandPolicyError(
        "No trusted AI executable found. Install the selected provider outside "
        "the workspace and use an absolute PATH entry. Windows supports native "
        "executables and recognized Claude/Copilot npm entries (JS entries require "
        "an external node.exe). Arbitrary script wrappers are not executed."
    )
