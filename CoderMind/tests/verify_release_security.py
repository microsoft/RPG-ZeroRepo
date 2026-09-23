"""CI-only, read-only verification of all 22 provider/script release ZIPs."""

import ast
from collections import Counter
import importlib.util
from pathlib import Path, PurePosixPath
import sys
import zipfile


SCRIPT_TYPES = ("sh", "ps")
SCRIPT_PREFIX = ".cmind/scripts/"


def expected_script_sources(scripts_root: Path) -> dict[str, str]:
    """Discover every reviewed Python script, including future security helpers."""
    sources = {}
    for path in sorted(scripts_root.rglob("*.py")):
        relative = path.relative_to(scripts_root)
        if path.is_file() and "__pycache__" not in relative.parts:
            # Accept only the BOM/CRLF differences allowed for packaged scripts.
            sources[relative.as_posix()] = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    if not sources:
        raise RuntimeError("No source Python scripts found for release verification")
    return sources


def verify_archive(path: Path, sources: dict[str, str], command: str) -> None:
    expected = {
        SCRIPT_PREFIX + name: text.replace("<AI_CLI_CMD>", command)
        for name, text in sources.items()
    }
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        # Check before reading by name: ZipFile.read otherwise hides duplicates.
        duplicates = sorted(name for name, count in Counter(
            member.filename for member in members
        ).items() if count != 1)
        if duplicates:
            raise RuntimeError(f"{path.name}: Duplicate ZIP members: {duplicates}")
        actual_names = {
            member.filename for member in members
            if not member.is_dir() and member.filename.startswith(SCRIPT_PREFIX)
            and member.filename.endswith(".py")
            and "__pycache__" not in PurePosixPath(member.filename).parts
        }
        missing = sorted(expected.keys() - actual_names)
        unexpected = sorted(actual_names - expected.keys())
        if missing or unexpected:
            raise RuntimeError(
                f"{path.name}: Python script catalog mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, text in expected.items():
            try:
                actual = archive.read(name).decode("utf-8-sig").replace("\r\n", "\n")
                if actual != text:
                    raise RuntimeError(f"{path.name}: Release ZIP has stale or altered {name}")
                # Parse packaged text, without importing or executing any script.
                ast.parse(actual, filename=f"{path.name}:{name}")
            except (UnicodeDecodeError, SyntaxError) as exc:
                raise RuntimeError(f"{path.name}: Invalid Python script {name}") from exc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    policy_path = root / "scripts/common/ai_cli_policy.py"
    spec = importlib.util.spec_from_file_location("release_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing provider policy")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    if len(policy.PROVIDER_ARGV) != 11:
        raise RuntimeError("Expected the closed catalog of 11 release providers")
    archives = Path(sys.argv[1])
    archive_paths = {path for path in archives.iterdir() if path.suffix.lower() == ".zip"}
    variants = {}
    for provider, argv in policy.PROVIDER_ARGV.items():
        command = " ".join(argv)
        if policy.provider_from_command(command) != provider:
            raise RuntimeError("Release default does not satisfy the closed policy")
        for script_type in SCRIPT_TYPES:
            prefix = f"cmind-template-{provider}-{script_type}-"
            matches = sorted(path for path in archive_paths if path.name.startswith(prefix))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {script_type} release ZIP for {provider}; found {len(matches)}"
                )
            variants[matches[0]] = command
    unexpected = sorted(path.name for path in archive_paths - variants.keys())
    if len(archive_paths) != 22 or unexpected:
        raise RuntimeError(f"Expected exactly 22 release ZIPs; unexpected ZIPs: {unexpected}")
    sources = expected_script_sources(root / "scripts")
    for path, command in variants.items():
        verify_archive(path, sources, command)
    print(
        f"Verified all {len(sources)} Python scripts against reviewed source in all 22 release ZIPs "
        "(11 providers x sh/ps), including provider substitutions and AST syntax"
    )


if __name__ == "__main__":
    main()
