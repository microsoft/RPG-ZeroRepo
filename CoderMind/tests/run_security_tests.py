"""Run mocked security regressions without changing the developer's settings.

Use an existing Python environment. No dependency installation or real child
process is performed. All test writes go into a disposable directory under
CoderMind; home/config/temp variables are redirected only in this process.
This audit guard prevents accidental test side effects, not hostile Python code.
"""

import os
from pathlib import Path
import sys
import tempfile


def main() -> int:
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".security-tests-", dir=root) as tmp:
        sandbox = Path(tmp).resolve()
        home = sandbox / "home"
        home.mkdir()
        # These changes affect this test process only, never the parent shell.
        for key in list(os.environ):
            if key.startswith(("CMIND_", "PYTEST_")):
                del os.environ[key]
        for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR"):
            os.environ[key] = str(home)
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        tempfile.tempdir = str(home)
        os.chdir(sandbox)

        def check_path(value, dir_fd=None):
            if isinstance(value, int):
                return  # Already-open stdout/stderr and temporary file descriptors.
            path = Path(os.fsdecode(value))
            if not path.is_absolute() and dir_fd is not None and dir_fd >= 0:
                # Linux shutil.rmtree uses fd-relative paths during cleanup.
                # Resolve the actual directory rather than trusting a basename.
                fd_path = Path(f"/proc/self/fd/{dir_fd}")
                if not fd_path.exists():
                    raise PermissionError("Cannot verify fd-relative test write")
                path = fd_path.resolve() / path
            path = path.resolve()
            if not path.is_relative_to(sandbox):
                raise PermissionError("Security test attempted a write outside its sandbox")

        def audit(event, args):
            if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                         "os.spawn", "socket.connect", "socket.bind", "socket.getaddrinfo"):
                raise PermissionError("Security tests forbid real processes and network access")
            if event == "open":
                _, mode, flags = args
                if (mode and any(c in mode for c in "wax+")) or flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                ):
                    check_path(args[0])
            elif event in ("os.remove", "os.rmdir"):
                check_path(args[0], args[1])
            elif event in ("os.mkdir", "os.chmod"):
                check_path(args[0], args[2])
            elif event == "os.utime":
                check_path(args[0], args[3])
            elif event == "os.truncate":
                check_path(args[0])
            elif event in ("os.rename", "os.link"):
                check_path(args[0], args[2])
                check_path(args[1], args[3])
            elif event == "os.symlink":
                check_path(args[1], args[2])

        sys.addaudithook(audit)
        try:
            if "--installed" in sys.argv:
                # CI-only mode after installing the freshly built wheel. Preload
                # the modules so test sys.path helpers cannot mask a stale wheel.
                import cmind_cli
                from cmind_cli import _assets

                package = Path(cmind_cli.__file__).resolve().parent
                if package.is_relative_to(root):
                    raise RuntimeError("Artifact tests require an external wheel installation")
                for source, target in [
                    (root / "src/cmind_cli/__init__.py", package / "__init__.py"),
                    *[(root / "scripts/common" / name, _assets.scripts_dir() / "common" / name)
                      for name in ("ai_cli_policy.py", "windows_ai_cli.py", "llm_client.py", "session_manager.py")],
                ]:
                    if source.read_bytes() != target.read_bytes():
                        raise RuntimeError("Installed artifact does not match reviewed source")
                sys.path.insert(0, str(_assets.scripts_dir()))
                from common import ai_cli_policy, windows_ai_cli, llm_client, session_manager

                for module in (ai_cli_policy, windows_ai_cli, llm_client, session_manager):
                    if not Path(module.__file__).resolve().is_relative_to(package):
                        raise RuntimeError("Artifact test imported source instead of wheel")
                print("Verified installed wheel sources and imported module origins")
            else:
                sys.path[:0] = [str(root / "src"), str(root / "scripts")]

            import pytest

            return pytest.main([
                str(root / "tests" / "test_ai_cli_policy.py"),
                str(root / "tests" / "test_llm_client_agent_detect.py"),
                str(root / "tests" / "test_windows_ai_cli.py"),
                str(root / "tests" / "test_hooks_install.py"),
                "-k", "not update_graphs_status",
                "--basetemp", str(sandbox / "pytest"),
                "--log-file", str(sandbox / "pytest.log"),
                "-p", "no:cacheprovider", "--assert=plain", "-q",
            ])
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
