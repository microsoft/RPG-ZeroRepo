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


class SandboxAudit:
    """Guard test writes using each filesystem operation's symlink semantics.

    Content writes follow the final link; directory-entry operations resolve
    only the parent. In particular, unlinking a cyclic or dangling test link
    must not attempt to resolve the target of the link being removed.
    """

    def __init__(self, sandbox: Path) -> None:
        self.sandbox = sandbox.resolve()

    def check_path(self, value, dir_fd=None, *, follow_leaf=True):
        if isinstance(value, int):
            return  # Existing stdout/stderr and temporary file descriptors.
        raw_path = os.fsdecode(value)
        path = Path(raw_path)
        if not path.is_absolute() and dir_fd is not None and dir_fd >= 0:
            # Linux shutil.rmtree uses fd-relative paths during cleanup.
            fd_path = Path(f"/proc/self/fd/{dir_fd}")
            if not fd_path.exists():
                raise PermissionError("Cannot verify fd-relative test write")
            path = fd_path.resolve() / path
        # Trailing separators/dot components request directory traversal; Path
        # normalizes those away, so preserve their semantics from the raw input.
        spelling = raw_path.replace(os.altsep, os.sep) if os.altsep else raw_path
        directory_operand = spelling.endswith((os.sep, os.sep + ".", os.sep + ".."))
        if follow_leaf or directory_operand or path.name in ("", ".", ".."):
            checked = path.resolve()
        else:
            checked = path.parent.resolve() / path.name
        if not checked.is_relative_to(self.sandbox):
            raise PermissionError("Security test attempted a write outside its sandbox")

    def __call__(self, event, args):
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                     "os.spawn", "socket.connect", "socket.bind", "socket.getaddrinfo"):
            raise PermissionError("Security tests forbid real processes and network access")
        if event == "open":
            _, mode, flags = args
            if (mode and any(c in mode for c in "wax+")) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            ):
                self.check_path(args[0])
        elif event in ("os.remove", "os.rmdir"):
            self.check_path(args[0], args[1], follow_leaf=False)
        elif event == "os.mkdir":
            self.check_path(args[0], args[2], follow_leaf=False)
        elif event == "os.chmod":
            self.check_path(args[0], args[2])
        elif event == "os.utime":
            self.check_path(args[0], args[3])
        elif event == "os.truncate":
            self.check_path(args[0])
        elif event == "os.rename":
            self.check_path(args[0], args[2], follow_leaf=False)
            self.check_path(args[1], args[3], follow_leaf=False)
        elif event == "os.link":
            # A hard link must not create an alias to external content.
            self.check_path(args[0], args[2])
            self.check_path(args[1], args[3], follow_leaf=False)
        elif event == "os.symlink":
            self.check_path(args[1], args[2], follow_leaf=False)


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

        sys.addaudithook(SandboxAudit(sandbox))
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
                str(root / "tests" / "test_security_test_sandbox.py"),
                "-k", "not update_graphs_status",
                "--basetemp", str(sandbox / "pytest"),
                "--log-file", str(sandbox / "pytest.log"),
                "-p", "no:cacheprovider", "--assert=plain", "-q",
            ])
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
