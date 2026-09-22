"""CI-only, read-only verification of all eleven provider release ZIPs."""

import ast
import importlib.util
from pathlib import Path
import sys
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    policy_path = root / "scripts/common/ai_cli_policy.py"
    spec = importlib.util.spec_from_file_location("release_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing provider policy")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    archives = Path(sys.argv[1])
    for provider, argv in policy.PROVIDER_ARGV.items():
        matches = list(archives.glob(f"cmind-template-{provider}-sh-*.zip"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one sh release ZIP for {provider}")
        command = " ".join(argv)
        with zipfile.ZipFile(matches[0]) as archive:
            for name in ("ai_cli_policy.py", "windows_ai_cli.py", "llm_client.py", "session_manager.py"):
                expected = (root / "scripts/common" / name).read_text(encoding="utf-8")
                expected = expected.replace("<AI_CLI_CMD>", command)
                actual = archive.read(f".cmind/scripts/common/{name}").decode("utf-8-sig")
                # PowerShell packaging may normalize line endings or add a BOM.
                if actual.replace("\r\n", "\n") != expected:
                    raise RuntimeError(f"Release ZIP has stale or altered {name}")
                ast.parse(actual)
        if policy.provider_from_command(command) != provider:
            raise RuntimeError("Release default does not satisfy the closed policy")
    print("Verified policy, client and permission defaults in all 11 release ZIPs")


if __name__ == "__main__":
    main()
