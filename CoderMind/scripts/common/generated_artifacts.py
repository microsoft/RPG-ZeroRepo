"""Generated artifact hygiene rules for CoderMind-managed repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import subprocess


_EXCLUDE_MARKER_BEGIN = "# BEGIN CoderMind generated artifact hygiene"
_EXCLUDE_MARKER_END = "# END CoderMind generated artifact hygiene"

GIT_LOCAL_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".cmind/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "/.venv/",
    "/venv/",
    "/env/",
    "node_modules/",
    "/.next/",
    "/.nuxt/",
    "/target/",
    "/build/",
    "/dist/",
    "/coverage/",
    "cmake-build-debug/",
    "cmake-build-release/",
    "CMakeFiles/",
    "Testing/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.o",
    "*.obj",
    "*.a",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.rlib",
    "*.rmeta",
    "CMakeCache.txt",
    "CTestTestfile.cmake",
    "cmake_install.cmake",
    "compile_commands.json",
    ".ninja_log",
    ".ninja_deps",
)

GENERATED_ARTIFACT_DIRS = frozenset(
    pattern.rstrip("/")
    for pattern in GIT_LOCAL_EXCLUDE_PATTERNS
    if pattern.endswith("/") and not pattern.startswith("/")
)
ROOT_GENERATED_ARTIFACT_DIRS = frozenset(
    pattern.strip("/")
    for pattern in GIT_LOCAL_EXCLUDE_PATTERNS
    if pattern.startswith("/") and pattern.endswith("/")
)
GENERATED_ARTIFACT_FILES = frozenset(
    pattern
    for pattern in GIT_LOCAL_EXCLUDE_PATTERNS
    if not pattern.endswith("/") and not pattern.startswith("*.")
)
GENERATED_ARTIFACT_SUFFIXES = tuple(
    pattern[1:]
    for pattern in GIT_LOCAL_EXCLUDE_PATTERNS
    if pattern.startswith("*.")
)

_PROMPT_EXAMPLES = (
    "build/",
    "target/",
    "dist/",
    "node_modules/",
    "CMakeFiles/",
    "CTestTestfile.cmake",
    "cmake_install.cmake",
    "compile_commands.json",
    "__pycache__/",
)


@dataclass(frozen=True)
class GeneratedArtifactChange:
    """A generated artifact path found in a persistent git change scope."""

    path: str
    scope: str


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def is_generated_artifact_path(path: str) -> bool:
    """Return True when ``path`` belongs to a generated output/cache area."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False

    parts = tuple(part for part in PurePosixPath(normalized).parts if part not in {"", "."})
    if not parts:
        return False

    if parts[0] in ROOT_GENERATED_ARTIFACT_DIRS:
        return True
    if any(part in GENERATED_ARTIFACT_DIRS for part in parts):
        return True

    name = parts[-1]
    if name in GENERATED_ARTIFACT_FILES:
        return True
    return name.endswith(GENERATED_ARTIFACT_SUFFIXES)


def _run_git(
    repo_path: Path,
    args: list[str],
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _git_changed_artifacts(
    repo_path: Path,
    args: list[str],
    scope: str,
) -> list[GeneratedArtifactChange]:
    result = _run_git(repo_path, args)
    if result is None or result.returncode != 0:
        return []
    changes: list[GeneratedArtifactChange] = []
    for raw_path in result.stdout.splitlines():
        path = _normalize_repo_path(raw_path)
        if path and is_generated_artifact_path(path):
            changes.append(GeneratedArtifactChange(path=path, scope=scope))
    return changes


def find_persisted_generated_artifact_changes(
    repo_path: Path,
    *,
    base_ref: str | None = None,
) -> list[GeneratedArtifactChange]:
    """Find generated artifacts in committed, staged, or tracked diffs.

    Untracked ignored build outputs are intentionally ignored; CMake, Cargo,
    npm, and pytest legitimately create those while tests run. The local git
    exclude policy prevents them from being staged by ordinary ``git add -A``.
    """
    repo_path = Path(repo_path)
    candidates: list[GeneratedArtifactChange] = []
    if base_ref:
        candidates.extend(
            _git_changed_artifacts(
                repo_path,
                ["diff", "--name-only", f"{base_ref}..HEAD"],
                "branch",
            )
        )
    candidates.extend(
        _git_changed_artifacts(repo_path, ["diff", "--name-only", "--cached"], "staged")
    )
    candidates.extend(
        _git_changed_artifacts(repo_path, ["diff", "--name-only"], "worktree")
    )

    unique: dict[str, GeneratedArtifactChange] = {}
    for change in candidates:
        unique.setdefault(change.path, change)
    return list(unique.values())


def format_generated_artifact_violation(changes: list[GeneratedArtifactChange]) -> str:
    """Format a human-readable violation message for codegen agents."""
    shown = changes[:10]
    lines = [
        "Generated build/dependency/cache artifacts are not valid source changes.",
        "Update source, tests, manifests, or build configuration files instead.",
        "Invalid paths:",
    ]
    lines.extend(f"- {change.path} ({change.scope})" for change in shown)
    if len(changes) > len(shown):
        lines.append(f"- ... {len(changes) - len(shown)} more")
    return "\n".join(lines)


def generated_artifact_prompt_rule(extra_guidance: str = "") -> str:
    """Return the prompt rule that mirrors the verification policy."""
    examples = ", ".join(f"`{example}`" for example in _PROMPT_EXAMPLES)
    text = (
        "- Do NOT edit or commit generated build, dependency, cache, or test-output "
        f"artifacts such as {examples}. Update source files, test files, dependency "
        "manifests, or build configuration instead."
    )
    if extra_guidance:
        text += f" {extra_guidance.strip()}"
    return text + "\n"


def _managed_exclude_block() -> str:
    return "\n".join(
        (_EXCLUDE_MARKER_BEGIN, *GIT_LOCAL_EXCLUDE_PATTERNS, _EXCLUDE_MARKER_END)
    )


def ensure_generated_artifact_excludes(repo_path: Path) -> bool:
    """Install CoderMind's generated-artifact ignores in ``.git/info/exclude``.

    Returns True when the local exclude file was changed. Returns False when
    the repository is not a git worktree or the current block is present.
    """
    repo_path = Path(repo_path)
    result = _run_git(repo_path, ["rev-parse", "--git-path", "info/exclude"])
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return False

    exclude_path = Path(result.stdout.strip())
    if not exclude_path.is_absolute():
        exclude_path = repo_path / exclude_path
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    content = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    block = _managed_exclude_block()
    if _EXCLUDE_MARKER_BEGIN in content and _EXCLUDE_MARKER_END in content:
        before, rest = content.split(_EXCLUDE_MARKER_BEGIN, 1)
        _, after = rest.split(_EXCLUDE_MARKER_END, 1)
        updated = f"{before}{block}{after}"
        if updated == content:
            return False
        exclude_path.write_text(updated, encoding="utf-8")
        return True

    prefix = "" if not content or content.endswith("\n") else "\n"
    exclude_path.write_text(f"{content}{prefix}{block}\n", encoding="utf-8")
    return True