"""Inert filesystem fixtures only: never execute a CLI, wrapper, or Node.

Load the standalone module directly to avoid common.__init__ and policy imports.
The Windows resolver itself is platform-independent, so these tests also cover
its path/metadata contract on Linux. Only unavailable symlink creation skips.
"""

import errno
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/common/windows_ai_cli.py"
SPEC = importlib.util.spec_from_file_location("windows_ai_cli_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
windows_ai_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(windows_ai_cli)
resolve_windows_argv = windows_ai_cli.resolve_windows_argv

PACKAGE_NAMES = {
    "claude": "@anthropic-ai/claude-code",
    "copilot": "@github/copilot",
}
OFFICIAL_LAYOUTS = [
    ("claude", "2.1.278", "bin/claude.exe"),
    ("claude", "2.0.0", "cli.js"),
    ("copilot", "1.0.87", "npm-loader.js"),
    ("copilot", "0.0.369", "index.js"),
]
DIRECT_PROVIDERS = [
    ("claude", "claude"), ("copilot", "copilot"), ("gemini", "gemini"),
    ("qwen", "qwen"), ("cursor-agent", "agent"), ("auggie", "augment"),
    ("codex", "codex"), ("codebuddy", "codebuddy"), ("qoder", "qodercli"),
    ("opencode", "opencode"), ("amp", "amp"),
]
INERT = b"INERT TEST FIXTURE - NOT AN EXECUTABLE OR SCRIPT\n"


@pytest.fixture(autouse=True)
def forbid_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Process execution is forbidden in resolver tests")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", forbidden)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    # Normalize Windows temporary-directory short names before comparisons.
    root = tmp_path.resolve()
    tools = root / "trusted tools 工具"
    workspace = root / "workspace"
    cwd = root / "current directory"
    for directory in (tools, workspace, cwd):
        directory.mkdir()
    monkeypatch.chdir(cwd)
    return tools, (workspace, cwd)


def inert_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(INERT)
    return path.resolve()


def write_manifest(package, metadata):
    manifest = package / "package.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return manifest


def make_package(directory, provider="claude", entry="cli.js", version="2.0.0"):
    package = directory / "node_modules" / PACKAGE_NAMES[provider]
    package.mkdir(parents=True, exist_ok=True)
    entry_path = inert_file(package / entry)
    write_manifest(package, {
        "name": PACKAGE_NAMES[provider], "version": version, "bin": {provider: entry},
    })
    return package, entry_path


def symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except NotImplementedError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    except OSError as exc:
        unavailable = {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP}
        if exc.errno in unavailable or getattr(exc, "winerror", None) in {1, 50, 1314}:
            pytest.skip(f"Symlink creation unavailable: {exc}")
        raise


@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
def test_official_layouts_return_only_executable_and_fixed_entry(layout, provider, version, name):
    tools, roots = layout
    _, entry = make_package(tools, provider, name, version)
    expected = [str(entry)]
    if name.endswith(".js"):
        node = inert_file(tools / "node.exe")
        expected.insert(0, str(node))
    directories = [tools]
    assert resolve_windows_argv(provider, directories, roots) == expected
    assert directories == [tools]
    assert all(Path(arg).is_absolute() for arg in expected)


@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
def test_build_argv_reaches_npm_adapter_end_to_end(layout, monkeypatch, provider, version, name):
    # Load the real runtime policy (preloaded from the wheel in artifact mode).
    from common import ai_cli_policy as policy

    tools, roots = layout
    _, entry = make_package(tools, provider, name, version)
    expected = [str(entry)]
    if name.endswith(".js"):
        expected.insert(0, str(inert_file(tools / "node.exe")))
    # The input shim can contain anything: it is never read or launched.
    inert_file(tools / (provider + ".cmd"))
    monkeypatch.setattr(policy, "_IS_WINDOWS", True)
    assert policy.build_argv(provider, roots[0], search_path=str(tools)) == expected


def test_build_argv_prefers_direct_native_over_npm_in_earlier_path(layout, monkeypatch):
    from common import ai_cli_policy as policy

    tools, roots = layout
    make_package(tools)
    inert_file(tools / "node.exe")
    native = inert_file(tools.parent / "native" / "claude.exe")
    native.chmod(0o755)
    monkeypatch.setattr(policy, "_IS_WINDOWS", True)
    assert policy.build_argv("claude", roots[0], search_path=os.pathsep.join([
        str(tools), str(native.parent),
    ])) == [str(native)]


def test_build_argv_does_not_follow_exe_link_to_batch_file(layout, monkeypatch):
    from common import ai_cli_policy as policy

    tools, roots = layout
    target = inert_file(tools / "wrapper.cmd")
    target.chmod(0o755)
    symlink(tools / "claude.exe", target)
    monkeypatch.setattr(policy, "_IS_WINDOWS", True)
    with pytest.raises(policy.AICommandPolicyError):
        policy.build_argv("claude", roots[0], search_path=str(tools))


@pytest.mark.parametrize("provider,base", DIRECT_PROVIDERS)
def test_direct_native_resolution_belongs_to_caller(layout, provider, base):
    tools, roots = layout
    inert_file(tools / (base + ".exe"))
    assert resolve_windows_argv(provider, [tools], roots) is None


@pytest.mark.parametrize("provider", ["", "CLAUDE", "claude -p", "claude.exe", "copilot.cmd", None, [], {}])
def test_invalid_provider_does_not_select_an_adapter(layout, provider):
    tools, roots = layout
    make_package(tools, entry="bin/claude.exe")
    assert resolve_windows_argv(provider, [tools], roots) is None


@pytest.mark.parametrize("provider", ["claude", "copilot"])
@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".ps1"])
def test_wrappers_alone_are_never_read_or_selected(layout, monkeypatch, provider, suffix):
    tools, roots = layout
    inert_file(tools / (provider + suffix))
    inert_file(tools / "node.exe")

    def no_read(*args, **kwargs):
        raise AssertionError("No package manifest exists; no file should be read")

    monkeypatch.setattr(Path, "open", no_read)
    assert resolve_windows_argv(provider, [tools], roots) is None


def test_valid_package_ignores_all_wrappers(layout, monkeypatch):
    tools, roots = layout
    _, entry = make_package(tools)
    node = inert_file(tools / "node.exe")
    for base in ("claude", "node", "npm", "npx"):
        for suffix in (".cmd", ".bat", ".ps1"):
            inert_file(tools / (base + suffix))
    original_open = Path.open

    def checked_open(path, *args, **kwargs):
        assert path.suffix not in {".cmd", ".bat", ".ps1"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    assert resolve_windows_argv("claude", [tools], roots) == [str(node), str(entry)]


@pytest.mark.parametrize("name", ["cli.js", "bin/claude.exe"])
@pytest.mark.parametrize("metadata", [
    None, [], "package", 7, True, {},
    {"bin": {"claude": "cli.js"}},
    {"name": "@github/copilot", "bin": {"claude": "cli.js"}},
    {"name": "@Anthropic-ai/claude-code", "bin": {"claude": "cli.js"}},
    {"name": "@anthropic-ai/claude-code ", "bin": {"claude": "cli.js"}},
    {"name": [], "bin": {"claude": "cli.js"}},
    {"name": "@anthropic-ai/claude-code"},
])
def test_rejects_wrong_package_identity_and_metadata_shape(layout, name, metadata):
    tools, roots = layout
    package, _ = make_package(tools, entry=name)
    inert_file(tools / "node.exe")
    write_manifest(package, metadata)
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("bins", [
    None, "cli.js", [], True, 9, {},
    {"other": "cli.js"}, {"Claude": "cli.js"},
    {"claude": "cli.js", "extra": "cli.js"},
    {"claude": None}, {"claude": True}, {"claude": 9},
    {"claude": ["cli.js"]}, {"claude": {"path": "cli.js"}},
])
def test_rejects_nonexact_bin_key_and_value_types(layout, bins):
    tools, roots = layout
    package, _ = make_package(tools)
    inert_file(tools / "node.exe")
    write_manifest(package, {"name": PACKAGE_NAMES["claude"], "bin": bins})
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
@pytest.mark.parametrize("bad_entry", [
    "", "other.js", "claude.exe", "copilot.exe", "./cli.js", "../cli.js",
    "bin/../cli.js", "/cli.js", "cli.js/", "CLI.JS", "cli.js ",
    "cli.js --settings bad", '"cli.js"', "cli.js;ignored", "cli.js\nignored",
    "claude.cmd", "claude.bat", "claude.ps1", "cli.js:payload", "cli.js\x00",
    "https://example.invalid/cli.js", "file:///cli.js", "bin//claude.exe",
    r"C:\tools\cli.js", r"C:cli.js", r"..\cli.js", r"bin\claude.exe",
    r"\\server\share\cli.js", "bin/claude.exe --extra",
])
def test_rejects_paths_arguments_and_nonallowlisted_entries(layout, provider, version, name, bad_entry):
    tools, roots = layout
    package, _ = make_package(tools, provider, name, version)
    inert_file(tools / "node.exe")
    write_manifest(package, {"name": PACKAGE_NAMES[provider], "bin": {provider: bad_entry}})
    assert resolve_windows_argv(provider, [tools], roots) is None


@pytest.mark.parametrize("raw", [
    b"", b"{", b"\xff", b'\xef\xbb\xbf{}',
    b'{"name":"@anthropic-ai/claude-code","bin":{"claude":"cli.js"},}',
    b'{"name":"wrong","name":"@anthropic-ai/claude-code","bin":{"claude":"cli.js"}}',
    b'{"name":"@anthropic-ai/claude-code","bin":{},"bin":{"claude":"cli.js"}}',
    b'{"name":"@anthropic-ai/claude-code","bin":{"claude":"other.js","claude":"cli.js"}}',
    b'{"name":"@anthropic-ai/claude-code","bin":{"claude":"cli.js"},"extra":NaN}',
    b'{"name":"@anthropic-ai/claude-code","bin":{"claude":"cli.js"},"extra":Infinity}',
    b"[" * 2000 + b"]" * 2000,
    b" " * (64 * 1024 + 1),
], ids=["empty", "truncated", "invalid-utf8", "bom", "trailing-comma",
        "duplicate-name", "duplicate-bin", "duplicate-entry", "nan", "infinity",
        "deep-json", "oversized"])
def test_malformed_manifest_is_a_miss_not_an_exception(layout, raw):
    tools, roots = layout
    package, _ = make_package(tools)
    inert_file(tools / "node.exe")
    (package / "package.json").write_bytes(raw)
    assert resolve_windows_argv("claude", [tools], roots) is None


def test_evidence_versions_are_not_pins_and_other_metadata_is_not_executed(layout):
    tools, roots = layout
    package, entry = make_package(tools, entry="bin/claude.exe", version="99.0.0")
    write_manifest(package, {
        "name": PACKAGE_NAMES["claude"], "version": "99.0.0",
        "bin": {"claude": "bin/claude.exe"},
        "scripts": {"postinstall": "INERT PLACEHOLDER"},
        "main": "ignored.js", "exports": "ignored.js",
    })
    assert resolve_windows_argv("claude", [tools], roots) == [str(entry)]


@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
@pytest.mark.parametrize("broken", ["manifest_missing", "manifest_directory", "entry_missing", "entry_directory"])
def test_manifest_and_entry_must_be_existing_files(layout, provider, version, name, broken):
    tools, roots = layout
    package, entry = make_package(tools, provider, name, version)
    inert_file(tools / "node.exe")
    target = package / "package.json" if broken.startswith("manifest") else entry
    target.unlink()
    if broken.endswith("directory"):
        target.mkdir()
    assert resolve_windows_argv(provider, [tools], roots) is None


@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
def test_native_needs_no_node_but_javascript_does(layout, provider, version, name):
    tools, roots = layout
    _, entry = make_package(tools, provider, name, version)
    expected = [str(entry)] if name.endswith(".exe") else None
    assert resolve_windows_argv(provider, [tools], roots) == expected


def test_node_is_found_in_another_supplied_directory(layout):
    tools, roots = layout
    _, entry = make_package(tools, "copilot", "npm-loader.js", "1.0.87")
    node = inert_file(tools.parent / "node installation" / "node.exe")
    assert resolve_windows_argv("copilot", [tools, node.parent], roots) == [str(node), str(entry)]


def test_never_searches_environment_cwd_or_prefix_parent_for_node(layout, monkeypatch):
    tools, roots = layout
    make_package(tools)
    external = inert_file(tools.parent / "not supplied" / "node.exe")
    inert_file(tools.parent / "node.exe")
    for root in roots:
        inert_file(root / "node.exe")
    monkeypatch.setenv("PATH", str(external.parent))
    monkeypatch.setenv("PATHEXT", ".CMD;.BAT;.PS1;.EXE")
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("name", ["node", "node.cmd", "node.bat", "node.ps1"])
def test_node_wrappers_or_extensionless_files_are_not_interpreters(layout, name):
    tools, roots = layout
    make_package(tools)
    inert_file(tools / name)
    assert resolve_windows_argv("claude", [tools], roots) is None


def test_node_directory_is_not_an_interpreter(layout):
    tools, roots = layout
    make_package(tools)
    (tools / "node.exe").mkdir()
    assert resolve_windows_argv("claude", [tools], roots) is None


def test_unsafe_directories_and_invalid_candidates_do_not_hide_later_install(layout):
    tools, roots = layout
    for root in roots:
        make_package(root, entry="bin/claude.exe")
        make_package(root / "bin", entry="bin/claude.exe")
    broken = tools.parent / "broken"
    package, _ = make_package(broken, entry="bin/claude.exe")
    write_manifest(package, {"name": "wrong", "bin": {"claude": "bin/claude.exe"}})
    wrappers = tools.parent / "wrappers"
    inert_file(wrappers / "claude.cmd")
    _, entry = make_package(tools, entry="bin/claude.exe")
    regular_file = inert_file(tools.parent / "not a directory")
    directories = [
        Path(""), Path("."), Path("bin"), *roots, *(root / "bin" for root in roots),
        tools.parent / "missing", regular_file, wrappers, broken, tools,
    ]
    assert resolve_windows_argv("claude", directories, roots) == [str(entry)]


def test_valid_package_search_preserves_directory_order(layout):
    tools, roots = layout
    _, first = make_package(tools, entry="bin/claude.exe")
    other = tools.parent / "other"
    _, second = make_package(other, entry="bin/claude.exe")
    assert resolve_windows_argv("claude", [other, tools], roots) == [str(second)]
    assert resolve_windows_argv("claude", [tools, other], roots) == [str(first)]


def test_missing_node_in_earlier_js_install_does_not_hide_later_native(layout):
    tools, roots = layout
    make_package(tools)
    other = tools.parent / "native"
    _, entry = make_package(other, entry="bin/claude.exe")
    assert resolve_windows_argv("claude", [tools, other], roots) == [str(entry)]


@pytest.mark.parametrize("root_index", [0, 1])
@pytest.mark.parametrize("component", ["prefix", "package", "manifest", "entry", "node"])
def test_links_into_workspace_or_cwd_are_rejected(layout, root_index, component):
    tools, roots = layout
    excluded = roots[root_index]
    if component == "prefix":
        make_package(excluded, entry="bin/claude.exe")
        link = tools / "prefix-link"
        symlink(link, excluded, directory=True)
        directories = [link]
    elif component == "package":
        target, _ = make_package(excluded, entry="bin/claude.exe")
        link = tools / "node_modules" / PACKAGE_NAMES["claude"]
        link.parent.mkdir(parents=True)
        symlink(link, target, directory=True)
        directories = [tools]
    else:
        package, entry = make_package(tools)
        if component == "node":
            target = inert_file(excluded / "node.exe")
            symlink(tools / "node.exe", target)
        else:
            inert_file(tools / "node.exe")
            link = package / "package.json" if component == "manifest" else entry
            target = excluded / link.name
            target.write_bytes(link.read_bytes())
            link.unlink()
            symlink(link, target)
        directories = [tools]
    assert resolve_windows_argv("claude", directories, roots) is None


@pytest.mark.parametrize("component", ["manifest", "entry"])
@pytest.mark.parametrize("provider,version,name", OFFICIAL_LAYOUTS)
def test_manifest_and_entry_links_may_not_escape_package_even_to_external_tools(
    layout, component, provider, version, name
):
    tools, roots = layout
    package, entry = make_package(tools, provider, name, version)
    inert_file(tools / "node.exe")
    link = package / "package.json" if component == "manifest" else entry
    # In particular, a Claude native link into a sibling platform package is
    # unsupported without a separately verified family/version/architecture rule.
    target = package.parent / "sibling-platform-package" / link.name
    target.parent.mkdir()
    target.write_bytes(link.read_bytes())
    link.unlink()
    symlink(link, target)
    assert resolve_windows_argv(provider, [tools], roots) is None


def test_package_link_to_external_install_is_canonicalized(layout):
    tools, roots = layout
    target, entry = make_package(tools.parent / "external install", entry="bin/claude.exe")
    link = tools / "node_modules" / PACKAGE_NAMES["claude"]
    link.parent.mkdir(parents=True)
    symlink(link, target, directory=True)
    assert resolve_windows_argv("claude", [tools], roots) == [str(entry)]


def test_lexically_excluded_prefix_link_to_external_tools_is_rejected(layout):
    tools, roots = layout
    make_package(tools, entry="bin/claude.exe")
    link = roots[0] / "external-tools"
    symlink(link, tools, directory=True)
    assert resolve_windows_argv("claude", [link], roots) is None


def test_excluded_root_alias_also_excludes_its_canonical_destination(layout):
    tools, roots = layout
    make_package(tools, entry="bin/claude.exe")
    alias = roots[0] / "tools-alias"
    symlink(alias, tools, directory=True)
    assert resolve_windows_argv("claude", [tools], (alias,)) is None


def test_invalid_relative_exclusion_fails_closed(layout):
    tools, _ = layout
    make_package(tools, entry="bin/claude.exe")
    assert resolve_windows_argv("claude", [tools], (Path("workspace"),)) is None


def test_unsafe_node_link_is_skipped_in_favor_of_later_safe_node(layout):
    tools, roots = layout
    _, entry = make_package(tools)
    local_node = inert_file(roots[0] / "node.exe")
    symlink(tools / "node.exe", local_node)
    node = inert_file(tools.parent / "safe node" / "node.exe")
    assert resolve_windows_argv("claude", [tools, node.parent], roots) == [str(node), str(entry)]


@pytest.mark.parametrize("component", ["node", "native_entry"])
@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".ps1"])
def test_exe_links_cannot_change_returned_command_into_wrapper(layout, component, suffix):
    tools, roots = layout
    if component == "native_entry":
        package, link = make_package(tools, entry="bin/claude.exe")
        target = inert_file(package / ("payload" + suffix))
        link.unlink()
    else:
        make_package(tools)
        target = inert_file(tools / ("payload" + suffix))
        link = tools / "node.exe"
    symlink(link, target)
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("kind", ["broken", "loop"])
def test_broken_or_looping_entry_links_are_misses(layout, kind):
    tools, roots = layout
    _, entry = make_package(tools, entry="bin/claude.exe")
    entry.unlink()
    symlink(entry, entry if kind == "loop" else entry.parent / "missing.exe")
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("operation", ["resolve", "open"])
def test_filesystem_errors_skip_candidate_without_raising(layout, monkeypatch, operation):
    tools, roots = layout
    package, _ = make_package(tools, entry="bin/claude.exe")
    other = tools.parent / "fallback"
    _, entry = make_package(other, entry="bin/claude.exe")
    original = getattr(Path, operation)
    blocked = package if operation == "resolve" else package / "package.json"

    def denied(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("synthetic unreadable installation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, denied)
    assert resolve_windows_argv("claude", [tools, other], roots) == [str(entry)]


def test_unresolvable_exclusion_fails_closed(layout, monkeypatch):
    tools, roots = layout
    make_package(tools, entry="bin/claude.exe")
    original_resolve = Path.resolve

    def denied(path, *args, **kwargs):
        if path == roots[0]:
            raise OSError("synthetic exclusion failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", denied)
    assert resolve_windows_argv("claude", [tools], roots) is None


@pytest.mark.parametrize("candidate,parent,expected", [
    (r"\\?\C:\workspace\node.exe", r"C:\workspace", True),
    (r"C:\workspace\node.exe", r"\\?\C:\workspace", True),
    (r"\\?\C:\WORKSPACE\node.exe", r"c:\workspace", True),
    (r"\\?\UNC\server\share\workspace\node.exe", r"\\server\share\workspace", True),
    (r"\\server\share\workspace\node.exe", r"\\?\UNC\server\share\workspace", True),
    (r"\\?\C:\workspace-other\node.exe", r"C:\workspace", False),
    (r"\\?\UNC\server\other\workspace\node.exe", r"\\server\share\workspace", False),
])
def test_windows_extended_path_containment_on_every_os(candidate, parent, expected):
    assert windows_ai_cli._is_within(PureWindowsPath(candidate), PureWindowsPath(parent)) is expected

