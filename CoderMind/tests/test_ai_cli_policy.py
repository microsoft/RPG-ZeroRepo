"""Security regressions: all process execution is mocked, never run a payload."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import Mock
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Pin the closed catalog independently so collection need not import common:
# its package initializer eagerly imports the home-reading LLM client.
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
HINT_KEYS = ("recommended_provider", "ai_provider", "ai_cli_cmd")
RELEASE_MODULES = (
    "common/ai_cli_policy.py", "common/windows_ai_cli.py",
    "common/llm_client.py", "common/session_manager.py",
    "common/trusted_tools.py", "common/git_utils.py",
    "common/generated_artifacts.py", "common/rpg_io.py",
    "code_gen/git_ops.py", "rpg_encoder/version_control.py",
    "rpg_encoder/run_encode.py", "update_graphs.py", "feature_build.py",
    "future/nested/security_helper.py",
)


def write_config(root, contents):
    path = root / ".cmind" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    # A unique sibling home is disposable but is NOT inside this workspace.
    home = tmp_path.parent / (tmp_path.name + "-home")
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in ("CMIND_AI_PROVIDER", "CMIND_AI_CLI_CMD", "CMIND_HOOK"):
        monkeypatch.delenv(name, raising=False)
    # An accidental real spawn fails even when these tests run without the runner.
    spawn = Mock(side_effect=AssertionError("Real subprocess forbidden in policy tests"))
    monkeypatch.setattr(subprocess, "Popen", spawn)
    # Defer imports that derive home paths until isolation is installed. Keep
    # CMIND_WORKSPACE untouched; clients use this explicit workspace finder.
    from common import paths

    monkeypatch.setattr(paths, "_find_workspace_root", lambda: tmp_path)
    return spawn


@pytest.fixture
def policy(isolated_workspace):
    from common import ai_cli_policy

    return ai_cli_policy


@pytest.fixture
def llm_client(isolated_workspace, session_manager, monkeypatch):
    from common import llm_client as module

    monkeypatch.setattr(module, "_BAKED_IN_VALUE", module._PLACEHOLDER_LITERAL)
    return module


@pytest.fixture
def session_manager(isolated_workspace, tmp_path, monkeypatch):
    from common import session_manager as module

    monkeypatch.setattr(module, "COPILOT_LOGS_DIR", tmp_path / "logs" / "copilot")
    monkeypatch.setattr(module.ClaudeSessionManager, "DEFAULT_TRAJECTORY_DIR",
                        tmp_path / "logs" / "claude")
    return module


def selection_record(workspace, provider="claude"):
    return {
        "schema_version": 1,
        "workspace": os.path.normcase(str(workspace.resolve())),
        "ai_provider": provider,
    }


def expected_selection_path(workspace):
    # Compute the contract independently, without policy's private helpers.
    identity = os.path.normcase(str(workspace.resolve()))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path.home().resolve() / ".cmind" / "execution" / digest / "selection.json"


@pytest.fixture
def local_selection(tmp_path, policy):
    write_config(tmp_path, '[cmind]\nrecommended_provider = "copilot"\n')
    policy.write_local_provider(tmp_path, "claude")
    return expected_selection_path(tmp_path)


def assert_invalid_local_rejected(policy, workspace):
    path = expected_selection_path(workspace)
    before = path.read_bytes()
    with pytest.raises(policy.AICommandPolicyError):
        policy.read_local_provider(workspace)
    for baked in ("", "claude", "codex exec"):
        with pytest.raises(policy.AICommandPolicyError):
            policy.resolve_provider(workspace, environ={}, baked=baked)
    assert path.read_bytes() == before


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not allow symlink creation")


def test_provider_catalog_is_closed(policy):
    assert policy.PROVIDER_ARGV == PROVIDER_ARGV


@pytest.mark.parametrize("ordinary,extended", [
    (r"C:\Users\developer\workspace", r"\\?\C:\Users\developer\workspace"),
    (r"\\server\share\workspace", r"\\?\UNC\server\share\workspace"),
])
def test_extended_windows_spelling_shares_identity_and_boundary(policy, ordinary, extended):
    plain = PureWindowsPath(ordinary)
    long = PureWindowsPath(extended)
    assert policy._comparison_path(long) == plain
    assert policy._is_within(long / "nested", plain)
    assert policy._is_within(plain / "nested", long)
    assert policy._workspace_identity(Mock(resolve=lambda: plain)) == policy._workspace_identity(Mock(resolve=lambda: long))



@pytest.mark.parametrize("hint_key", HINT_KEYS)
@pytest.mark.parametrize("provider,argv", PROVIDER_ARGV.items())
def test_closed_provider_and_exact_legacy_mapping(tmp_path, policy, provider, argv, hint_key):
    assert policy.validate_provider(provider) == provider
    command = " ".join(argv)
    assert policy.provider_from_command(command) == provider
    hint = command if hint_key == "ai_cli_cmd" else provider
    write_config(tmp_path, f'[cmind]\n{hint_key} = "{hint}"\n')
    assert policy.read_workspace_provider(tmp_path) == provider
    assert policy.read_local_provider(tmp_path) == ""
    assert policy.resolve_provider(tmp_path, environ={}) == ""
    assert policy.resolve_provider(tmp_path, environ={}, baked=command) == ""
    assert not (Path.home() / ".cmind").exists()


INVALID_COMMANDS = [
    "", " ", "unknown", "python -c PLACEHOLDER", "sh -c PLACEHOLDER",
    "claude --dangerously-skip-permissions", "copilot --allow-all",
    "claude --settings config.json", "claude; ignored", "claude\nignored",
    '"claude"', " claude", "claude ", "CLAUDE", "codex  exec",
    "./claude", "/tmp/claude", r"C:\tools\claude.exe", "claude.cmd",
    "claude.exe", "claude.ps1", None, 1, [], {}, True,
]


@pytest.mark.parametrize("value", INVALID_COMMANDS)
def test_reject_noncanonical_commands(policy, value):
    with pytest.raises(policy.AICommandPolicyError):
        policy.provider_from_command(value)


@pytest.mark.parametrize("contents", [
    "[cmind", "cmind = 7", '[cmind]\nai_provider = "unknown"',
    '[cmind]\nai_provider = "claude -p"', '[cmind]\nai_provider = ""',
    '[cmind]\nai_provider = 7', '[cmind]\nai_cli_cmd = []',
    '[cmind]\nai_cli_cmd = "claude --settings file"',
    '[cmind]\nai_provider = "claude"\nai_cli_cmd = "claude"',
    '[cmind]\nrecommended_provider = "unknown"',
    '[cmind]\nrecommended_provider = "codex exec"',
    '[cmind]\nrecommended_provider = ""',
    '[cmind]\nrecommended_provider = 7',
    '[cmind]\nrecommended_provider = true',
    '[cmind]\nrecommended_provider = []',
    '[cmind]\nrecommended_provider = {}',
    '[cmind]\nrecommended_provider = "claude"\nai_provider = "claude"',
    '[cmind]\nrecommended_provider = "claude"\nai_cli_cmd = "claude"',
    '[cmind]\nrecommended_provider = "claude"\nai_provider = "claude"\nai_cli_cmd = "claude"',
])
def test_invalid_workspace_never_falls_back(tmp_path, policy, contents, isolated_workspace, llm_client):
    write_config(tmp_path, contents)
    policy.write_local_provider(tmp_path, "claude")
    with pytest.raises(policy.AICommandPolicyError):
        policy.read_workspace_provider(tmp_path)
    # Invalid recommendations fail even when CI explicitly chooses the same
    # documented provider, or a constructor/local selection already authorizes it.
    for overrides in (
        {"environ": {}},
        {"environ": {}, "tool": "claude"},
        {"environ": {"CMIND_AI_PROVIDER": "claude"}},
        {"environ": {"CMIND_AI_CLI_CMD": "claude"}},
    ):
        with pytest.raises(policy.AICommandPolicyError):
            policy.resolve_provider(tmp_path, baked="claude", **overrides)
    with pytest.raises(policy.AICommandPolicyError):
        llm_client.LLMClient()
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("source", ["tool", "env", "provider_env", "baked"])
def test_each_command_source_fails_closed(tmp_path, policy, source):
    policy.write_local_provider(tmp_path, "copilot")
    write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"')
    kwargs = {"environ": {}}
    if source == "tool":
        kwargs["tool"] = "claude extra"
    elif source == "env":
        kwargs["environ"] = {"CMIND_AI_CLI_CMD": "claude extra"}
        kwargs["baked"] = "claude"
    elif source == "provider_env":
        kwargs["environ"] = {"CMIND_AI_PROVIDER": "claude extra"}
        kwargs["baked"] = "claude"
    else:
        kwargs["baked"] = "claude extra"
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, **kwargs)


def test_precedence_and_empty_vs_absent(tmp_path, policy):
    assert policy.resolve_provider(tmp_path, environ={}) == ""
    assert policy.resolve_provider(tmp_path, environ={}, baked="codex exec") == ""
    write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"')
    assert policy.resolve_provider(tmp_path, environ={}) == ""
    policy.write_local_provider(tmp_path, "qwen")
    assert policy.resolve_provider(tmp_path, environ={}) == "qwen"
    assert policy.resolve_provider(tmp_path, environ={}, baked="codex exec") == "qwen"
    env = {"CMIND_AI_PROVIDER": "copilot"}
    assert policy.resolve_provider(tmp_path, environ=env) == "copilot"
    assert policy.resolve_provider(tmp_path, environ=env, tool="codex exec") == "codex"
    assert policy.resolve_provider(tmp_path, environ={"CMIND_AI_CLI_CMD": "codex exec"}) == "codex"
    for env in ({"CMIND_AI_PROVIDER": ""}, {"CMIND_AI_CLI_CMD": ""},
                {"CMIND_AI_PROVIDER": "claude", "CMIND_AI_CLI_CMD": "claude"}):
        with pytest.raises(policy.AICommandPolicyError):
            policy.resolve_provider(tmp_path, environ=env)
    assert policy.read_local_provider(tmp_path) == "qwen"


@pytest.mark.parametrize("provider,argv", PROVIDER_ARGV.items())
def test_explicit_and_environment_authority_uses_only_closed_values(tmp_path, policy, provider, argv):
    write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"')
    policy.write_local_provider(tmp_path, "copilot")
    command = " ".join(argv)
    assert policy.resolve_provider(tmp_path, environ={}, tool=command) == provider
    assert policy.resolve_provider(tmp_path, environ={"CMIND_AI_PROVIDER": provider}) == provider
    assert policy.resolve_provider(tmp_path, environ={"CMIND_AI_CLI_CMD": command}) == provider
    assert policy.read_local_provider(tmp_path) == "copilot"


@pytest.mark.parametrize("overrides", [
    {"environ": {}, "tool": "codex exec"},
    {"environ": {"CMIND_AI_PROVIDER": "copilot"}, "tool": "codex exec"},
    {"environ": {"CMIND_AI_CLI_CMD": "copilot"}, "tool": "codex exec"},
    {"environ": {"CMIND_AI_PROVIDER": "codex"}},
    {"environ": {"CMIND_AI_CLI_CMD": "codex exec"}},
])
def test_higher_authority_does_not_read_or_rewrite_local_selection(
    tmp_path, monkeypatch, policy, local_selection, overrides,
):
    before = local_selection.read_bytes()
    read_local = Mock(side_effect=AssertionError("Higher-priority authority must not read local state"))
    monkeypatch.setattr(policy, "read_local_provider", read_local)
    assert policy.resolve_provider(tmp_path, baked="claude", **overrides) == "codex"
    read_local.assert_not_called()
    assert local_selection.read_bytes() == before


@pytest.mark.parametrize("name", ["CMIND_AI_PROVIDER", "CMIND_AI_CLI_CMD"])
@pytest.mark.parametrize("value", INVALID_COMMANDS)
def test_invalid_environment_cannot_fall_back_to_local(tmp_path, policy, local_selection, name, value):
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={name: value}, baked="claude")
    assert policy.read_local_provider(tmp_path) == "claude"


@pytest.mark.parametrize("command", [
    " ".join(argv) for argv in PROVIDER_ARGV.values()
    if " ".join(argv) not in PROVIDER_ARGV
])
def test_provider_environment_does_not_accept_legacy_commands(tmp_path, policy, command):
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={"CMIND_AI_PROVIDER": command})


@pytest.mark.parametrize("hint_key", HINT_KEYS)
@pytest.mark.parametrize("operation", ["clone", "move"])
def test_repository_hint_in_clone_or_move_does_not_authorize(tmp_path, policy, hint_key, operation):
    original = tmp_path / "original"
    hint = "codex exec" if hint_key == "ai_cli_cmd" else "codex"
    config = write_config(original, f'[cmind]\n{hint_key} = "{hint}"\n')
    config_bytes = config.read_bytes()
    policy.write_local_provider(original, "copilot")
    selection = expected_selection_path(original)
    selection_bytes = selection.read_bytes()
    assert policy.resolve_provider(original, environ={}) == "copilot"

    destination = tmp_path / operation
    if operation == "clone":
        shutil.copytree(original, destination)
    else:
        original.rename(destination)

    assert (destination / ".cmind" / "config.toml").read_bytes() == config_bytes
    assert policy.read_workspace_provider(destination) == "codex"
    assert policy.read_local_provider(destination) == ""
    assert policy.resolve_provider(destination, environ={}, baked="codex exec") == ""
    assert not expected_selection_path(destination).exists()
    assert selection.read_bytes() == selection_bytes


@pytest.mark.parametrize("provider", PROVIDER_ARGV)
def test_local_selection_uses_full_canonical_hash_and_strict_record(tmp_path, policy, provider):
    config = write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"\n')
    config_bytes = config.read_bytes()
    policy.write_local_provider(tmp_path, provider)
    path = policy.local_selection_path(tmp_path)
    assert path == expected_selection_path(tmp_path)
    assert len(path.parent.name) == 64
    assert not path.resolve().is_relative_to(tmp_path.resolve())
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == selection_record(tmp_path, provider)
    assert type(record["schema_version"]) is int
    assert policy.read_local_provider(tmp_path) == provider
    assert policy.resolve_provider(tmp_path, environ={}, baked="codex exec") == provider
    assert config.read_bytes() == config_bytes
    assert set(path.parent.iterdir()) == {path}


def test_local_selection_is_separate_from_copyable_storage_metadata(tmp_path, policy):
    from cmind_cli import _storage

    metadata = _storage.workspace_meta_path(tmp_path)
    metadata.parent.mkdir(parents=True)
    metadata.write_text('ai_provider = "copilot"\n', encoding="utf-8")
    metadata_bytes = metadata.read_bytes()
    write_config(tmp_path, '[cmind]\nrecommended_provider = "copilot"\n')
    assert policy.resolve_provider(tmp_path, environ={}, baked="copilot") == ""
    policy.write_local_provider(tmp_path, "claude")
    selection = policy.local_selection_path(tmp_path)
    assert not selection.is_relative_to(metadata.parent)
    assert metadata.read_bytes() == metadata_bytes
    assert policy.read_local_provider(tmp_path) == "claude"


@pytest.mark.parametrize("contents", [
    b"", b"{", b'{"schema_version":', b"\xff", b"null", b"[]", b'"claude"', b"1", b"true",
])
def test_malformed_local_selection_never_falls_back(tmp_path, policy, local_selection, contents):
    local_selection.write_bytes(contents)
    assert_invalid_local_rejected(policy, tmp_path)


@pytest.mark.parametrize("field", ["schema_version", "workspace", "ai_provider"])
def test_local_selection_requires_every_field(tmp_path, policy, local_selection, field):
    record = selection_record(tmp_path)
    del record[field]
    local_selection.write_text(json.dumps(record), encoding="utf-8")
    assert_invalid_local_rejected(policy, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 0), ("schema_version", 2), ("schema_version", True),
    ("schema_version", 1.0), ("schema_version", "1"), ("schema_version", None),
    ("workspace", ""), ("workspace", "different-workspace"),
    ("workspace", None), ("workspace", 1), ("workspace", []),
    ("extra", "not allowed"),
])
def test_local_selection_requires_exact_schema_and_identity(tmp_path, policy, local_selection, field, value):
    record = selection_record(tmp_path)
    record[field] = value
    local_selection.write_text(json.dumps(record), encoding="utf-8")
    assert_invalid_local_rejected(policy, tmp_path)


@pytest.mark.parametrize("provider", [*INVALID_COMMANDS, "codex exec", "agent -p"])
def test_invalid_local_provider_never_falls_back(tmp_path, policy, local_selection, provider):
    local_selection.write_text(json.dumps(selection_record(tmp_path, provider)), encoding="utf-8")
    assert_invalid_local_rejected(policy, tmp_path)


def test_copied_local_selection_cannot_authorize_another_workspace(tmp_path, policy):
    original = tmp_path / "original"
    copied = tmp_path / "copied"
    for workspace in (original, copied):
        write_config(workspace, '[cmind]\nrecommended_provider = "claude"\n')
    policy.write_local_provider(original, "claude")
    source = expected_selection_path(original)
    destination = expected_selection_path(copied)
    assert source != destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    assert_invalid_local_rejected(policy, copied)
    assert policy.read_local_provider(original) == "claude"


@pytest.mark.parametrize("existing_provider", [None, "claude"])
def test_atomic_replace_failure_preserves_selection_and_removes_temporary_files(
    tmp_path, monkeypatch, policy, existing_provider,
):
    if existing_provider is not None:
        policy.write_local_provider(tmp_path, existing_provider)
    selection = expected_selection_path(tmp_path)
    before = selection.read_bytes() if selection.exists() else None

    def fail_replace(source, destination):
        source = Path(source)
        assert Path(destination) == selection
        assert source != selection
        assert source.parent == selection.parent
        assert json.loads(source.read_text(encoding="utf-8")) == selection_record(tmp_path, "copilot")
        assert (selection.read_bytes() if selection.exists() else None) == before
        raise OSError("simulated atomic replace failure")

    replace = Mock(side_effect=fail_replace)
    monkeypatch.setattr(policy.os, "replace", replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        policy.write_local_provider(tmp_path, "copilot")
    replace.assert_called_once()
    assert (selection.read_bytes() if selection.exists() else None) == before
    assert set(selection.parent.iterdir()) == ({selection} if before is not None else set())
    assert policy.read_local_provider(tmp_path) == (existing_provider or "")


@pytest.mark.parametrize("provider", [*INVALID_COMMANDS, "codex exec", "agent -p"])
def test_invalid_write_provider_creates_no_directories(tmp_path, policy, provider):
    root = Path.home() / ".cmind"
    assert not root.exists()
    with pytest.raises(policy.AICommandPolicyError):
        policy.write_local_provider(tmp_path, provider)
    assert not root.exists()
    assert not (tmp_path / ".cmind").exists()


@pytest.mark.parametrize("component", ["home-store", "execution", "workspace", "selection"])
@pytest.mark.parametrize("dangling", [False, True])
def test_local_selection_symlinks_and_directory_redirection_are_rejected(
    tmp_path, policy, component, dangling,
):
    selection = expected_selection_path(tmp_path)
    link = {
        "home-store": selection.parents[2],
        "execution": selection.parents[1],
        "workspace": selection.parent,
        "selection": selection,
    }[component]
    target_root = Path.home().resolve() / "redirect-target"
    directory = component != "selection"
    target = target_root if directory else target_root / "selection.json"
    target_selection = target / selection.relative_to(link) if directory else target
    record = json.dumps(selection_record(tmp_path)).encode("utf-8")
    if not dangling:
        target_selection.parent.mkdir(parents=True)
        target_selection.write_bytes(record)
    link.parent.mkdir(parents=True, exist_ok=True)
    symlink_or_skip(link, target, directory=directory)

    with pytest.raises(policy.AICommandPolicyError):
        policy.local_selection_path(tmp_path)
    with pytest.raises(policy.AICommandPolicyError):
        policy.read_local_provider(tmp_path)
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={}, baked="claude")
    with pytest.raises(policy.AICommandPolicyError):
        policy.write_local_provider(tmp_path, "copilot")
    if dangling:
        assert not target_root.exists()
    else:
        assert target_selection.read_bytes() == record
        assert not list(target_root.rglob(".selection-*.tmp"))


def test_selection_directory_is_invalid_not_missing(tmp_path, policy):
    selection = expected_selection_path(tmp_path)
    selection.mkdir(parents=True)
    write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"\n')
    with pytest.raises(policy.AICommandPolicyError):
        policy.read_local_provider(tmp_path)
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={}, baked="claude")


def test_symlinked_workspace_reuses_canonical_local_selection(tmp_path, policy):
    workspace = tmp_path / "workspace"
    write_config(workspace, '[cmind]\nrecommended_provider = "codex"\n')
    alias = tmp_path / "alias"
    symlink_or_skip(alias, workspace, directory=True)
    policy.write_local_provider(workspace, "claude")
    selection = expected_selection_path(workspace)
    assert policy.local_selection_path(alias) == selection
    assert policy.read_local_provider(alias) == "claude"
    assert policy.resolve_provider(alias, environ={}) == "claude"
    policy.write_local_provider(alias, "copilot")
    assert policy.read_local_provider(workspace) == "copilot"
    assert json.loads(selection.read_text(encoding="utf-8")) == selection_record(workspace, "copilot")
    assert list((Path.home() / ".cmind" / "execution").glob("*/selection.json")) == [selection]


@pytest.mark.parametrize("nested_home", [False, True])
def test_user_local_store_inside_workspace_is_rejected(tmp_path, monkeypatch, policy, nested_home):
    home = tmp_path / "user-home" if nested_home else tmp_path
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    write_config(tmp_path, '[cmind]\nrecommended_provider = "claude"\n')
    with pytest.raises(policy.AICommandPolicyError):
        policy.local_selection_path(tmp_path)
    with pytest.raises(policy.AICommandPolicyError):
        policy.read_local_provider(tmp_path)
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={}, baked="claude")
    with pytest.raises(policy.AICommandPolicyError):
        policy.write_local_provider(tmp_path, "claude")
    assert not (home / ".cmind" / "execution").exists()


def make_cli(directory, name="claude"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".exe" if os.name == "nt" else name)
    path.write_bytes(b"not an executable; test fixture only")
    path.chmod(0o755)
    return path.resolve()


@pytest.mark.parametrize("provider,base", PROVIDER_ARGV.items())
def test_exact_internal_argv_and_absolute_executable(tmp_path, monkeypatch, policy, provider, base):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    executable = make_cli(tmp_path / "trusted tools", base[0])
    assert policy.build_argv(provider, workspace, search_path=str(executable.parent)) == [
        str(executable), *base[1:],
    ]


def test_path_search_ignores_workspace_current_and_relative_entries(tmp_path, monkeypatch, policy):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    make_cli(workspace)
    make_cli(workspace / "bin")
    trusted = make_cli(tmp_path / "trusted")
    path = os.pathsep.join(["", ".", "bin", str(workspace), str(workspace / "bin"), str(trusted.parent)])
    assert policy.build_argv("claude", workspace, search_path=path) == [str(trusted)]
    with pytest.raises(policy.AICommandPolicyError):
        policy.build_argv("claude", workspace, search_path=os.pathsep.join(["", ".", str(workspace)]))


def test_symlink_to_workspace_executable_is_rejected(tmp_path, monkeypatch, policy):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    local = make_cli(workspace)
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    link = trusted / local.name
    try:
        link.symlink_to(local)
    except OSError:
        pytest.skip("OS does not allow symlink creation")
    with pytest.raises(policy.AICommandPolicyError):
        policy.build_argv("claude", workspace, search_path=str(trusted))


@pytest.mark.skipif(os.name != "nt", reason="Windows wrapper policy")
def test_windows_wrappers_never_selected(tmp_path, monkeypatch, policy):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    tools = tmp_path / "tools"
    tools.mkdir()
    for suffix in (".cmd", ".bat", ".ps1"):
        (tools / ("claude" + suffix)).write_text("not executable")
    with pytest.raises(policy.AICommandPolicyError):
        policy.build_argv("claude", workspace, search_path=str(tools))


@pytest.mark.parametrize("value", [v for v in INVALID_COMMANDS if v is not None])
def test_invalid_constructor_cannot_spawn(value, policy, isolated_workspace, llm_client):
    with pytest.raises(policy.AICommandPolicyError):
        llm_client.LLMClient(tool=value)
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("hook", ["post-commit", "post-merge", "pre-commit"])
def test_hook_context_blocks_ai_before_trace(monkeypatch, policy, isolated_workspace, llm_client, hook):
    client = llm_client.LLMClient(tool="claude")
    trace = Mock(side_effect=AssertionError("Trace must not start"))
    monkeypatch.setattr(client._session_manager, "trace", trace)
    monkeypatch.setenv("CMIND_HOOK", hook)
    with pytest.raises(policy.AICommandPolicyError, match="hooks is disabled"):
        client.generate("test")
    trace.assert_not_called()
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("replacement", ["copilot", None])
def test_local_change_between_construction_and_generate_is_rejected(
    tmp_path, monkeypatch, policy, isolated_workspace, llm_client, local_selection, replacement,
):
    client = llm_client.LLMClient()
    assert client.tool == "claude"
    trace = Mock(side_effect=AssertionError("Trace must not start"))
    monkeypatch.setattr(client._session_manager, "trace", trace)
    if replacement is None:
        local_selection.unlink()
    else:
        policy.write_local_provider(tmp_path, replacement)
    with pytest.raises(policy.AICommandPolicyError, match="changed"):
        client.generate("test", max_retries=5)
    trace.assert_not_called()
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("stage", ["construction", "generation"])
def test_invalid_local_client_never_falls_back_to_hint_or_baked(
    tmp_path, monkeypatch, policy, isolated_workspace, llm_client, local_selection, stage,
):
    monkeypatch.setattr(llm_client, "_BAKED_IN_VALUE", "codex exec")
    client = llm_client.LLMClient() if stage == "generation" else None
    trace = Mock(side_effect=AssertionError("Trace must not start"))
    if client is not None:
        assert client.tool == "claude"
        monkeypatch.setattr(client._session_manager, "trace", trace)
    local_selection.write_text('{"ai_provider": "claude"}', encoding="utf-8")
    with pytest.raises(policy.AICommandPolicyError):
        if client is None:
            llm_client.LLMClient()
        else:
            client.generate("test", max_retries=5)
    assert_invalid_local_rejected(policy, tmp_path)
    trace.assert_not_called()
    isolated_workspace.assert_not_called()


def test_unknown_mutated_tool_is_rejected_before_trace(monkeypatch, policy, isolated_workspace, llm_client):
    client = llm_client.LLMClient(tool="claude")
    client.tool = "unknown command"
    trace = Mock(side_effect=AssertionError("Trace must not start"))
    monkeypatch.setattr(client._session_manager, "trace", trace)
    with pytest.raises(policy.AICommandPolicyError):
        client.generate("test")
    trace.assert_not_called()
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("hint_key", HINT_KEYS)
@pytest.mark.parametrize("authority", ["constructor", "local"])
def test_mocked_generation_uses_absolute_internal_argv_despite_hint_changes(
    tmp_path, monkeypatch, policy, llm_client, session_manager, hint_key, authority,
):
    config = write_config(tmp_path, f'[cmind]\n{hint_key} = "claude"\n')
    policy.write_local_provider(tmp_path, "codex")
    selection = expected_selection_path(tmp_path)
    selection_bytes = selection.read_bytes()
    client = llm_client.LLMClient(tool="codex exec" if authority == "constructor" else None)
    config.write_text(f'[cmind]\n{hint_key} = "copilot"\n', encoding="utf-8")
    assert policy.read_workspace_provider(tmp_path) == "copilot"
    assert policy.resolve_provider(tmp_path, environ={}) == "codex"
    assert client.tool == "codex exec"
    argv = [str(tmp_path / "trusted" / "codex"), "exec"]
    monkeypatch.setattr(llm_client, "build_argv", lambda *a: argv)

    @contextmanager
    def trace(*args, **kwargs):
        ctx = session_manager.TraceContext()
        ctx.extra_args = ["--test-internal-flag"]
        yield ctx

    monkeypatch.setattr(client._session_manager, "trace", trace)
    proc = Mock(returncode=0)
    proc.communicate.return_value = ("test response", "")
    spawn = Mock(return_value=proc)
    monkeypatch.setattr(llm_client.subprocess, "Popen", spawn)
    assert client.generate("test prompt") == "test response"
    assert spawn.call_args.args[0] == [*argv, "--test-internal-flag"]
    assert spawn.call_args.kwargs["shell"] is False
    assert spawn.call_args.kwargs["cwd"] == tmp_path.resolve()
    spawn.assert_called_once()
    assert policy.read_local_provider(tmp_path) == "codex"
    assert selection.read_bytes() == selection_bytes


@pytest.mark.parametrize("provider,package,entry", [
    ("claude", "@anthropic-ai/claude-code", "bin/claude.exe"),
    ("claude", "@anthropic-ai/claude-code", "cli.js"),
    ("copilot", "@github/copilot", "index.js"),
    ("copilot", "@github/copilot", "npm-loader.js"),
])
def test_npm_generation_preserves_prompt_as_data(
    tmp_path, monkeypatch, policy, llm_client, provider, package, entry,
):
    tools = Path.home() / "trusted npm 工具"
    installed = tools / "node_modules" / package
    script = installed / entry
    script.parent.mkdir(parents=True)
    script.write_bytes(b"INERT FIXTURE ONLY")
    (installed / "package.json").write_text(json.dumps({
        "name": package, "bin": {provider: entry},
    }))
    node = tools / "node.exe"
    node.write_bytes(b"INERT FIXTURE ONLY")
    monkeypatch.setattr(policy, "_IS_WINDOWS", True)
    monkeypatch.setenv("PATH", str(tools))
    policy.write_local_provider(tmp_path, provider)
    write_config(tmp_path, '[cmind]\nrecommended_provider = "codex"\n')
    client = llm_client.LLMClient()
    prompt = 'Read literally: "quotes" & | %VALUE% !VALUE! ^ $()\n第二行'
    captured = {}

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        if kwargs["stdin"] is not None:
            captured["stdin"] = kwargs["stdin"].read()
        proc = Mock(returncode=0)
        proc.communicate.return_value = ("mocked response", "")
        return proc

    spawn = Mock(side_effect=popen)
    monkeypatch.setattr(llm_client.subprocess, "Popen", spawn)
    assert client.generate(prompt) == "mocked response"
    prefix = [str(script.resolve())] if entry.endswith(".exe") else [str(node.resolve()), str(script.resolve())]
    assert captured["argv"][:len(prefix)] == prefix
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == tmp_path.resolve()
    if provider == "copilot":
        assert captured["argv"][-2:] == ["-p", prompt]
    else:
        assert captured["stdin"] == prompt
    assert "--allow-all" not in captured["argv"]
    assert "--dangerously-skip-permissions" not in captured["argv"]
    spawn.assert_called_once()
    assert policy.read_local_provider(tmp_path) == provider


@pytest.mark.parametrize("command,agent", [
    ("claude --flag", "claude"),
    ('"C:\\tools\\claude.cmd" --flag', "claude"),
    ("C:\\tools\\Gemini.EXE --flag", "gemini"),
])
def test_upstream_agent_identification_does_not_authorize_execution(
    command, agent, policy, llm_client, isolated_workspace,
):
    assert llm_client.detect_agent_type(command) == agent
    with pytest.raises(policy.AICommandPolicyError):
        llm_client.LLMClient(tool=command)
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("explicit_encoding", [False, True])
def test_transplant_preserves_upstream_utf8_and_platform_launch(
    tmp_path, monkeypatch, llm_client, session_manager, windows, explicit_encoding,
):
    client = llm_client.LLMClient(tool="codex exec")
    argv = [str(tmp_path / "external" / "codex.exe"), "exec"]
    monkeypatch.setattr(llm_client, "build_argv", lambda *a: argv)
    monkeypatch.setattr(llm_client, "_IS_WINDOWS", windows)
    monkeypatch.setattr(llm_client.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    child_env = {"UNCHANGED": "yes"}
    if explicit_encoding:
        child_env.update(PYTHONIOENCODING="custom-encoding", PYTHONUTF8="0")

    @contextmanager
    def trace(*args, **kwargs):
        ctx = session_manager.TraceContext()
        ctx.env = child_env
        yield ctx

    monkeypatch.setattr(client._session_manager, "trace", trace)
    proc = Mock(returncode=0)
    proc.communicate.return_value = ("response 中文 →", "")
    spawn = Mock(return_value=proc)
    monkeypatch.setattr(llm_client.subprocess, "Popen", spawn)
    before = child_env.copy()
    assert client.generate("prompt 中文 →") == "response 中文 →"
    kwargs = spawn.call_args.kwargs
    assert spawn.call_args.args[0] == argv
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == tmp_path.resolve()
    assert child_env == before
    if windows:
        assert kwargs["creationflags"] == 0x200
        assert "preexec_fn" not in kwargs and "start_new_session" not in kwargs
        assert kwargs["env"]["PYTHONIOENCODING"] == ("custom-encoding" if explicit_encoding else "utf-8:replace")
        assert kwargs["env"]["PYTHONUTF8"] == ("0" if explicit_encoding else "1")
    else:
        assert kwargs["start_new_session"] is True
        assert kwargs["preexec_fn"] is llm_client._set_pdeathsig
        assert "creationflags" not in kwargs
        assert kwargs["env"] == before


@pytest.mark.parametrize("outcome", ["success", "nonzero", "os-error"])
def test_transplant_preserves_windows_tree_cleanup(monkeypatch, llm_client, outcome):
    monkeypatch.setattr(llm_client, "_IS_WINDOWS", True)
    proc = Mock(pid=12345)
    runner = Mock(return_value=Mock(returncode=0 if outcome == "success" else 1))
    if outcome == "os-error":
        runner.side_effect = OSError("simulated taskkill failure")
    monkeypatch.setattr(llm_client.subprocess, "run", runner)
    llm_client._kill_process_tree(proc)
    runner.assert_called_once_with(
        ["taskkill", "/PID", "12345", "/T", "/F"], capture_output=True, check=False,
    )
    if outcome == "success":
        proc.kill.assert_not_called()
    else:
        proc.kill.assert_called_once()


def test_transplant_timeout_uses_upstream_tree_cleanup(monkeypatch, llm_client):
    client = llm_client.LLMClient(tool="codex exec")
    monkeypatch.setattr(llm_client, "build_argv", lambda *a: ["trusted-codex", "exec"])
    proc = Mock(returncode=0)
    proc.communicate.side_effect = subprocess.TimeoutExpired("trusted-codex", 1)
    monkeypatch.setattr(llm_client.subprocess, "Popen", Mock(return_value=proc))
    cleanup = Mock()
    monkeypatch.setattr(llm_client, "_kill_process_tree", cleanup)
    with pytest.raises(RuntimeError, match="timed out"):
        client.generate("prompt", max_retries=1, timeout=1)
    cleanup.assert_called_once_with(proc)
    proc.wait.assert_called_once()


def test_missing_executable_is_not_retried(monkeypatch, policy, isolated_workspace, llm_client):
    client = llm_client.LLMClient(tool="claude")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(policy.AICommandPolicyError, match="No trusted"):
        client.generate("test", max_retries=5)
    isolated_workspace.assert_not_called()


@pytest.mark.parametrize("agent", ["claude", "copilot"])
def test_session_managers_do_not_bypass_permissions(tmp_path, session_manager, agent):
    # The fixture isolates home BEFORE the manager constructor inspects it.
    assert not Path.home().resolve().is_relative_to(tmp_path.resolve())
    manager = session_manager.create_session_manager(agent, project_dir=tmp_path)
    with manager.trace("test") as ctx:
        assert "--dangerously-skip-permissions" not in ctx.extra_args
        assert "--allow-all" not in ctx.extra_args
        assert "-p" in ctx.extra_args


def test_release_baked_commands_obey_the_same_policy(tmp_path, policy):
    # Historical release substitutions are validated hints, not local choices.
    for argv in policy.PROVIDER_ARGV.values():
        command = " ".join(argv)
        assert policy.resolve_provider(tmp_path, environ={}, baked=command) == ""
    policy.write_local_provider(tmp_path, "copilot")
    for argv in policy.PROVIDER_ARGV.values():
        command = " ".join(argv)
        assert policy.resolve_provider(tmp_path, environ={}, baked=command) == "copilot"
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, environ={}, baked="untrusted --argument")


@pytest.mark.parametrize("overrides", [
    {"environ": {}},
    {"environ": {}, "tool": "claude"},
    {"environ": {"CMIND_AI_PROVIDER": "claude"}},
    {"environ": {"CMIND_AI_CLI_CMD": "claude"}},
])
@pytest.mark.parametrize("baked", ["untrusted --argument", "codex", "claude ", " claude"])
def test_invalid_baked_hint_is_validated_even_with_authority(tmp_path, policy, local_selection, overrides, baked):
    with pytest.raises(policy.AICommandPolicyError):
        policy.resolve_provider(tmp_path, baked=baked, **overrides)


@pytest.mark.parametrize("hint_key", [None, *HINT_KEYS])
@pytest.mark.parametrize("baked", ["", "codex exec"])
def test_unconfigured_client_is_lazy_but_hints_never_authorize_a_launch(
    tmp_path, monkeypatch, isolated_workspace, llm_client, hint_key, baked,
):
    if hint_key is not None:
        write_config(tmp_path, f'[cmind]\n{hint_key} = "claude"\n')
    monkeypatch.setattr(llm_client, "_BAKED_IN_VALUE", baked)
    client = llm_client.LLMClient()
    assert client.tool == ""
    trace = Mock(side_effect=AssertionError("Trace must not start"))
    build = Mock(side_effect=AssertionError("Executable resolution must not start"))
    monkeypatch.setattr(client._session_manager, "trace", trace)
    monkeypatch.setattr(llm_client, "build_argv", build)
    with pytest.raises(RuntimeError, match="not configured"):
        client.generate("test", max_retries=5)
    trace.assert_not_called()
    build.assert_not_called()
    isolated_workspace.assert_not_called()
    assert not (Path.home() / ".cmind").exists()


@pytest.fixture
def release_verifier():
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "tests/verify_release_security.py")
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


@pytest.fixture
def release_packages(tmp_path, monkeypatch, release_verifier):
    # Small synthetic scripts keep the full 22-ZIP matrix cheap. Do not execute
    # production packagers or depend on the size of real pipeline modules.
    sources = {
        name: f'# {name}\nBAKED = "<AI_CLI_CMD>"\n'
              'raise AssertionError("Packaged scripts must not be executed")\n'
        for name in RELEASE_MODULES
    }
    monkeypatch.setattr(release_verifier, "expected_script_sources", lambda root: sources)
    # Use the independently pinned provider catalog and literal script types;
    # a verifier that silently narrows either catalog must fail these fixtures.
    for provider, argv in PROVIDER_ARGV.items():
        for script_type in ("sh", "ps"):
            archive = tmp_path / f"cmind-template-{provider}-{script_type}-v0.0.0-test.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(".cmind/scripts/", b"")
                stream.writestr(".cmind/config.toml", b"[cmind]\n")
                for name, source in sources.items():
                    text = source.replace("<AI_CLI_CMD>", " ".join(argv))
                    if script_type == "ps":
                        text = "\ufeff" + text.replace("\n", "\r\n")
                    stream.writestr(f".cmind/scripts/{name}", text)
    monkeypatch.setattr(sys, "argv", ["verify_release_security.py", str(tmp_path)])
    return sources


def _rewrite_release_fixture(path, *, omit=None, replacements=None, extra=()):
    with zipfile.ZipFile(path) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            if name != omit:
                archive.writestr(name, (replacements or {}).get(name, data))
        for name, data in extra:
            archive.writestr(name, data)


def test_release_verifier_checks_all_22_variants_read_only(
    tmp_path, monkeypatch, capsys, release_verifier, release_packages,
):
    before = {path: path.read_bytes() for path in tmp_path.glob("*.zip")}
    assert len(before) == 22
    parse = Mock(wraps=release_verifier.ast.parse)
    monkeypatch.setattr(release_verifier.ast, "parse", parse)
    release_verifier.main()
    assert "all 22 release ZIPs" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in tmp_path.glob("*.zip")} == before
    assert parse.call_count == 22 * len(release_packages)
    assert {call.kwargs["filename"] for call in parse.call_args_list} == {
        f"{path.name}:.cmind/scripts/{name}"
        for path in before for name in release_packages
    }


@pytest.mark.parametrize("script_type", ["sh", "ps"])
@pytest.mark.parametrize("corrupt", ["stale", "missing", "duplicate"])
@pytest.mark.parametrize("module", RELEASE_MODULES)
def test_release_verifier_detects_stale_missing_or_duplicate_assets(
    tmp_path, release_verifier, release_packages, script_type, corrupt, module,
):
    path = tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-test.zip"
    name = f".cmind/scripts/{module}"
    if corrupt == "stale":
        # Still valid Python: AST validity alone is not source identity.
        _rewrite_release_fixture(path, replacements={name: b"# stale artifact\n"})
        message = "stale or altered"
    elif corrupt == "missing":
        _rewrite_release_fixture(path, omit=name)
        message = "catalog mismatch"
    else:
        with zipfile.ZipFile(path) as archive:
            duplicate = archive.read(name)
        with pytest.warns(UserWarning, match="Duplicate name"):
            _rewrite_release_fixture(path, extra=[(name, duplicate)])
        message = "Duplicate ZIP members"
    with pytest.raises(RuntimeError, match=message) as error:
        release_verifier.main()
    assert path.name in str(error.value)
    assert name in str(error.value)


@pytest.mark.parametrize("script_type", ["sh", "ps"])
@pytest.mark.parametrize("corrupt", ["missing", "duplicate", "balanced"])
def test_release_verifier_requires_exactly_one_zip_per_pair(
    tmp_path, release_verifier, release_packages, script_type, corrupt,
):
    path = tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-test.zip"
    if corrupt == "missing":
        path.unlink()
    else:
        shutil.copyfile(path, tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-old.zip")
        if corrupt == "balanced":
            # Counting 22 archives alone must not hide a duplicate + missing pair.
            (tmp_path / f"cmind-template-amp-{script_type}-v0.0.0-test.zip").unlink()
            assert len(list(tmp_path.glob("*.zip"))) == 22
    with pytest.raises(RuntimeError, match=f"exactly one {script_type} release ZIP for claude"):
        release_verifier.main()


@pytest.mark.parametrize("name", [
    "cmind-template-unknown-sh-v0.0.0-test.zip",
    "cmind-template-claude-other-v0.0.0-test.zip",
    "unrelated.zip",
    "unexpected.ZIP",
])
def test_release_verifier_rejects_extra_archives(tmp_path, release_verifier, release_packages, name):
    with zipfile.ZipFile(tmp_path / name, "w"):
        pass
    with pytest.raises(RuntimeError, match="exactly 22 release ZIPs"):
        release_verifier.main()


@pytest.mark.parametrize("script_type", ["sh", "ps"])
@pytest.mark.parametrize("name", [".cmind/scripts/", ".cmind/config.toml"])
def test_release_verifier_rejects_duplicate_non_python_members(
    tmp_path, release_verifier, release_packages, script_type, name,
):
    path = tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-test.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _rewrite_release_fixture(path, extra=[(name, b"")])
    with pytest.raises(RuntimeError, match="Duplicate ZIP members"):
        release_verifier.main()


@pytest.mark.parametrize("script_type", ["sh", "ps"])
def test_release_verifier_rejects_obsolete_extra_modules(
    tmp_path, release_verifier, release_packages, script_type,
):
    path = tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-test.zip"
    name = ".cmind/scripts/common/removed_module.py"
    _rewrite_release_fixture(path, extra=[(name, b"# obsolete module\n")])
    with pytest.raises(RuntimeError, match="catalog mismatch") as error:
        release_verifier.main()
    assert name in str(error.value)


@pytest.mark.parametrize("script_type", ["sh", "ps"])
@pytest.mark.parametrize("corrupt", ["syntax", "encoding", "substitution"])
def test_release_verifier_validates_packaged_python(
    tmp_path, release_verifier, release_packages, script_type, corrupt,
):
    path = tmp_path / f"cmind-template-claude-{script_type}-v0.0.0-test.zip"
    module = "common/trusted_tools.py"
    sources = dict(release_packages)
    if corrupt == "syntax":
        # Even byte-identical source must parse; checking only identity is insufficient.
        sources[module] = "def invalid(:\n"
        data = sources[module].encode("utf-8")
        message = "Invalid Python script"
    elif corrupt == "encoding":
        data = b"\xff"
        message = "Invalid Python script"
    else:
        data = sources[module].replace("<AI_CLI_CMD>", "copilot").encode("utf-8")
        message = "stale or altered"
    _rewrite_release_fixture(path, replacements={f".cmind/scripts/{module}": data})
    with pytest.raises(RuntimeError, match=message):
        release_verifier.verify_archive(path, sources, "claude")


def test_release_source_catalog_is_recursive_and_excludes_only_caches(tmp_path, release_verifier):
    scripts = tmp_path / "scripts"
    expected = {
        "__init__.py": "",
        "common/trusted_tools.py": 'BAKED = "<AI_CLI_CMD>"\n',
        "new/deep/pipeline.py": "# new pipeline helper\n",
    }
    ignored = {
        "__pycache__/cached.py": "invalid cached Python",
        "new/__pycache__/deep/cached.py": "invalid nested cached Python",
        "common/readme.txt": "not Python",
    }
    for name, text in {**expected, **ignored}.items():
        path = scripts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("\ufeff" + text.replace("\n", "\r\n")).encode("utf-8"))
    (scripts / "directory.py").mkdir()
    assert release_verifier.expected_script_sources(scripts) == expected


def test_release_source_catalog_cannot_be_empty(tmp_path, release_verifier):
    with pytest.raises(RuntimeError, match="No source Python scripts"):
        release_verifier.expected_script_sources(tmp_path)
# All release variants must retain the reviewed Python script catalog.
