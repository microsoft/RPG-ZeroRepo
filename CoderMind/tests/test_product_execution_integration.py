"""Real local subprocess checks; run only with the isolated --integration runner.

No AI tools, downloads, or developer configuration are used. Same-name native
fixtures are unmodified system binaries, not executable attack payloads.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    ws = tmp_path / "workspace with spaces"
    ws.mkdir()
    for key in list(os.environ):
        if key.startswith(("CMIND_", "GIT_")):
            monkeypatch.delenv(key, raising=False)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME",
                "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(key, str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(ws)
    from common.trusted_tools import resolve_git

    template = tmp_path / "empty-template"
    template.mkdir()
    git = resolve_git(ws)
    subprocess.run([git, "init", "--template=" + str(template)], cwd=ws, check=True, capture_output=True)
    return ws


def invoke(args, workspace):
    from cmind_cli._trusted_tools import cli_argv

    return subprocess.run(cli_argv(*args), cwd=workspace, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=90)


def git_run(workspace, *args):
    from common.trusted_tools import resolve_git

    return subprocess.run([resolve_git(workspace), *args], cwd=workspace,
                          capture_output=True, text=True, check=True, timeout=90)


def test_real_init_and_opt_in_commit_use_installed_entry(workspace, monkeypatch):
    import cmind_cli

    # New command names cannot be resolved via PATH/cwd, even when lookalikes
    # exist. These text fixtures are never legitimate executable candidates.
    for name in ("cmind", "cmind.cmd", "cmind.ps1"):
        (workspace / name).write_text("must not execute", encoding="utf-8")
    fake_module = workspace / "cmind_cli"
    fake_module.mkdir()
    (fake_module / "__init__.py").write_text("raise AssertionError('workspace package imported')", encoding="utf-8")
    monkeypatch.setenv("PATH", str(workspace) + os.pathsep + os.environ.get("PATH", ""))

    result = invoke(["init", "--here", "--ai", "claude", "--script", "sh", "--force",
                     "--no-encode", "--ignore-agent-tools", "--no-mcp", "--no-cmind-git",
                     "--git-hooks"], workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    hook = (workspace / ".git/hooks/post-commit").read_text(encoding="utf-8")
    assert "_cli_entry.py" in hook
    assert "command -v cmind" not in hook
    (workspace / "readme.txt").write_text("inert integration fixture\n", encoding="utf-8")
    git_run(workspace, "add", "readme.txt")
    git_run(workspace, "-c", "user.name=Security Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-m", "inert fixture")
    log = cmind_cli._storage.workspace_logs_dir(workspace) / "hooks.log"
    content = log.read_text(encoding="utf-8")
    assert "post-commit fired" in content
    # No graph is seeded: the real packaged sync must report that condition,
    # not import the workspace package or fall back to an AI encoder.
    assert "Error: rpg.json not found at" in content, content
    assert "foreground-sync: done (exit 1)" in content, content
    assert "phase2" not in content
    assert not (log.parent / "update_rpg.log").exists()

    merged = invoke(["hook", "post-merge"], workspace)
    assert merged.returncode == 0
    assert "sync: done (exit 1)" in log.read_text(encoding="utf-8")

    # Default update reconciles the pinned hooks as well, not just old stubs.
    updated = invoke(["update", "--ai", "claude", "--script", "sh", "--no-upgrade",
                      "--no-cmind-git"], workspace)
    assert updated.returncode == 0, updated.stdout + updated.stderr
    assert not (workspace / ".git/hooks/post-commit").exists()
    assert not (workspace / ".git/hooks/post-merge").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows native executable lookup")
def test_windows_process_images_ignore_workspace_binaries(workspace, monkeypatch):
    import _winapi
    import ctypes
    from ctypes import wintypes
    import cmind_cli
    from common.trusted_tools import resolve_git

    system_binary = Path(os.environ["SystemRoot"]) / "System32/whoami.exe"
    for name in ("git.exe", "cmind.exe"):
        shutil.copyfile(system_binary, workspace / name)
    monkeypatch.setenv("PATH", str(workspace) + os.pathsep + os.environ.get("PATH", ""))
    expected_git = Path(resolve_git(workspace)).resolve()
    images = []
    create_process = _winapi.CreateProcess
    query = ctypes.WinDLL("kernel32", use_last_error=True).QueryFullProcessImageNameW
    query.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL

    def record_image(*args, **kwargs):
        handles = create_process(*args, **kwargs)
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if not query(handles[0], 0, buf, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        images.append(Path(buf.value).resolve())
        return handles

    monkeypatch.setattr(_winapi, "CreateProcess", record_image)
    assert cmind_cli.is_git_repo(workspace)
    cmind_cli._short_head_sha(workspace)
    # Non-workspace status command avoids any network/AI or graph mutations.
    log = workspace / "hook-probe.log"
    rc = cmind_cli._hook_run_foreground(workspace, log, os.environ.copy(),
                                       ["update_graphs.py", "status"], "probe")
    assert rc == 0, log.read_text(encoding="utf-8")
    assert images[:2] == [expected_git, expected_git]
    assert images[2] == Path(sys.executable).resolve()
    assert all(not path.is_relative_to(workspace) for path in images)
