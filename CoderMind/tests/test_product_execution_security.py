"""No-process regressions for product Git lookup and installed encoder selection."""

import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    monkeypatch.chdir(workspace)
    for name in ("CMIND_AI_PROVIDER", "CMIND_AI_CLI_CMD", "CMIND_HOOK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("Real process forbidden")))
    return workspace


@pytest.fixture
def tools(isolated):
    from common import trusted_tools

    return trusted_tools


def executable(directory, name="git", *, windows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".exe" if windows else name)
    path.write_text("inert test executable; never launch", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("name", ["git", "uv", "pipx"])
def test_only_external_absolute_tool_is_selected(tmp_path, isolated, tools, monkeypatch, windows, name):
    monkeypatch.setattr(tools, "_IS_WINDOWS", windows)
    executable(isolated, name, windows=windows)
    executable(isolated / "bin", name, windows=windows)
    expected = executable(tmp_path / "external tools", name, windows=windows)
    path = os.pathsep.join(("", ".", "bin", str(isolated), str(isolated / "bin"), str(expected.parent)))
    assert tools.resolve_tool(name, isolated, search_path=path) == str(expected)
    with pytest.raises(FileNotFoundError):
        tools.resolve_tool(name, isolated, search_path=os.pathsep.join(("", ".", str(isolated))))


def test_parent_checkout_is_excluded_from_nested_cwd(tmp_path, isolated, tools, monkeypatch):
    nested = isolated / "src" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    bad = executable(isolated / "tools", windows=tools._IS_WINDOWS)
    home_repo = tmp_path / "home-store"
    home_repo.mkdir()
    with pytest.raises(FileNotFoundError):
        tools.resolve_tool("git", home_repo, search_path=str(bad.parent))


@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".ps1"])
def test_windows_never_executes_script_wrappers(tmp_path, isolated, tools, monkeypatch, suffix):
    monkeypatch.setattr(tools, "_IS_WINDOWS", True)
    directory = tmp_path / "external"
    directory.mkdir()
    (directory / ("git" + suffix)).write_text("never launch", encoding="utf-8")
    monkeypatch.setenv("PATHEXT", suffix)
    with pytest.raises(FileNotFoundError):
        tools.resolve_tool("git", isolated, search_path=str(directory))


def link_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable")


@pytest.mark.parametrize("kind", ["directory-into-workspace", "file-into-workspace", "workspace-link-out"])
def test_redirected_tool_paths_cannot_bypass_boundary(tmp_path, isolated, tools, kind):
    bad = executable(isolated / "bin", windows=tools._IS_WINDOWS)
    external = tmp_path / "external"
    if kind == "directory-into-workspace":
        link_or_skip(external, bad.parent, directory=True)
        directory = external
    elif kind == "file-into-workspace":
        external.mkdir()
        link_or_skip(external / bad.name, bad)
        directory = external
    else:
        executable(external, windows=tools._IS_WINDOWS)
        directory = isolated / "external-link"
        link_or_skip(directory, external, directory=True)
    with pytest.raises(FileNotFoundError):
        tools.resolve_tool("git", isolated, search_path=str(directory))


def test_unknown_tool_is_not_a_command_channel(isolated, tools):
    for name in ("python", "git --version", "../git", "cmd.exe"):
        with pytest.raises(ValueError):
            tools.resolve_tool(name, isolated)


def test_product_git_helpers_use_the_same_external_executable(tmp_path, isolated, tools, monkeypatch):
    import cmind_cli
    from cmind_cli import _inner_git
    from common import git_utils, generated_artifacts

    safe = executable(tmp_path / "external", windows=tools._IS_WINDOWS)
    executable(isolated, windows=tools._IS_WINDOWS)
    monkeypatch.setenv("PATH", os.pathsep.join((str(isolated), str(safe.parent))))
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""))
    monkeypatch.setattr(subprocess, "run", run)
    assert cmind_cli.is_git_repo(isolated)
    assert cmind_cli._short_head_sha(isolated) == "abc123"
    cmind_cli._read_core_hooks_path(isolated)
    _inner_git._run_git(isolated, "status")
    assert git_utils._run_git_readonly(["rev-parse", "HEAD"], isolated) == "abc123"
    runner = git_utils.GitRunner.__new__(git_utils.GitRunner)
    runner.repo_path = isolated
    runner.logger = logging.getLogger(__name__)
    assert runner.run_git(["status"]).success
    assert generated_artifacts._run_git(isolated, ["status"]) is not None
    assert run.call_count == 7
    assert all(call.args[0][0] == str(safe) for call in run.call_args_list)


@pytest.mark.parametrize("installed_exists", [False, True])
def test_initial_encode_never_falls_back_to_workspace(tmp_path, isolated, monkeypatch, installed_exists):
    import cmind_cli
    from cmind_cli import _assets

    legacy = isolated / ".cmind/scripts/rpg_encoder/run_encode.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("raise AssertionError('workspace script must never run')", encoding="utf-8")
    scripts = tmp_path / "trusted-assets"
    scripts.mkdir()
    encoder = scripts / "rpg_encoder/run_encode.py"
    if installed_exists:
        encoder.parent.mkdir()
        encoder.write_text("# installed fixture", encoding="utf-8")
    monkeypatch.setattr(_assets, "scripts_dir", lambda: scripts)
    spawn = Mock(side_effect=OSError("stop before creating process"))
    monkeypatch.setattr(cmind_cli.subprocess, "Popen", spawn)
    assert cmind_cli._run_initial_encode(isolated) is False
    if installed_exists:
        spawn.assert_called_once()
        assert Path(spawn.call_args.args[0][1]) == encoder.resolve()
        assert spawn.call_args.kwargs["cwd"] == str(isolated)
    else:
        spawn.assert_not_called()


def test_cli_self_invocation_pins_interpreter_and_bootstrap(isolated, monkeypatch):
    import cmind_cli
    from cmind_cli import _trusted_tools

    monkeypatch.setenv("PATH", str(isolated))
    command = _trusted_tools.cli_argv("script", "update_graphs.py", "sync")
    assert command[0] == sys.executable
    assert command[1] == "-I"
    assert Path(command[2]) == Path(cmind_cli.__file__).resolve().with_name("_cli_entry.py")
    assert command[3:] == ["script", "update_graphs.py", "sync"]
    assert not Path(command[2]).is_relative_to(isolated)


def test_no_bare_git_or_cmind_subprocess_literals_remain():
    # Guard every pipeline, not just the three originally reported entry points.
    import cmind_cli
    from cmind_cli import _assets

    for root in (Path(cmind_cli.__file__).parent, _assets.scripts_dir()):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"run", "Popen", "call", "check_output", "check_call"}
                        and node.args and isinstance(node.args[0], (ast.List, ast.Tuple))):
                    continue
                items = node.args[0].elts
                if items and isinstance(items[0], ast.Constant):
                    assert items[0].value not in ("git", "cmind"), (str(path), node.lineno)
