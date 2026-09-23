"""Exercise sandbox cleanup without privileges, processes or external writes."""

import errno
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "security_sandbox_under_test", Path(__file__).with_name("run_security_tests.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
SandboxAudit = _RUNNER.SandboxAudit


@pytest.mark.parametrize("form", ["explicit", "argv", "windows-command-line"])
def test_integration_guard_accepts_external_absolute_executable(tmp_path, form):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executable = tmp_path / "external tools" / "python.exe"
    guard = _RUNNER.IntegrationAudit(sandbox)
    if form == "explicit":
        args = (str(executable), [], str(sandbox), {})
    elif form == "argv":
        args = (None, [str(executable), "--version"], str(sandbox), {})
    else:
        args = (None, f'"{executable}" --version', str(sandbox), {})
    guard("subprocess.Popen", args)


@pytest.mark.parametrize("form", ["relative", "inside-sandbox"])
def test_integration_guard_still_rejects_unsafe_executable(tmp_path, form):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    guard = _RUNNER.IntegrationAudit(sandbox)
    executable = "git" if form == "relative" else str(sandbox / "git.exe")
    with pytest.raises(PermissionError):
        guard("subprocess.Popen", (None, [executable], str(sandbox), {}))
    with pytest.raises(PermissionError):
        guard("socket.connect", ())


@pytest.fixture
def layout(tmp_path):
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    return SandboxAudit(sandbox), sandbox, outside


def _symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except NotImplementedError:
        pytest.skip("OS does not implement symlinks")
    except OSError as exc:
        if (exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP)
                or getattr(exc, "winerror", None) in (1, 50, 1314)):
            pytest.skip("Symlink creation requires unavailable OS permissions")
        raise


@pytest.mark.parametrize("operation", ["os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.symlink"])
def test_entry_operations_do_not_resolve_cyclic_leaf(layout, monkeypatch, operation):
    guard, sandbox, _ = layout
    leaf = sandbox / "loop"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == leaf:
            raise RuntimeError("Symlink loop from test leaf")
        return original(path, *args, **kwargs)

    # Works even on Windows machines without permission to create real links.
    monkeypatch.setattr(Path, "resolve", resolve)
    args = {
        "os.remove": (leaf, -1), "os.rmdir": (leaf, -1),
        "os.mkdir": (leaf, 0o700, -1),
        "os.rename": (leaf, sandbox / "renamed", -1, -1),
        "os.symlink": ("arbitrary-link-target", leaf, -1),
    }[operation]
    guard(operation, args)
    with pytest.raises(RuntimeError, match="Symlink loop"):
        guard("open", (leaf, "w", os.O_WRONLY))


@pytest.mark.parametrize("event", ["open", "os.chmod", "os.utime", "os.truncate", "os.link"])
def test_content_operations_still_reject_external_leaf(layout, monkeypatch, event):
    guard, sandbox, outside = layout
    leaf = sandbox / "outward-link"
    target = outside / "protected"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return target if path == leaf else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    args = {
        "open": (leaf, "w", os.O_WRONLY),
        "os.chmod": (leaf, 0o600, -1),
        "os.utime": (leaf, None, None, -1),
        "os.truncate": (leaf, 0),
        "os.link": (leaf, sandbox / "alias", -1, -1),
    }[event]
    with pytest.raises(PermissionError, match="outside"):
        guard(event, args)
    # Unlink affects only the in-sandbox directory entry, not the target.
    guard("os.remove", (leaf, -1))


@pytest.mark.parametrize("operation", ["os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.symlink"])
def test_entry_operations_reject_external_parent(layout, monkeypatch, operation):
    guard, sandbox, outside = layout
    parent = sandbox / "redirect"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return outside if path == parent else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    leaf = parent / "protected"
    args = {
        "os.remove": (leaf, -1), "os.rmdir": (leaf, -1),
        "os.mkdir": (leaf, 0o700, -1),
        "os.rename": (sandbox / "inside", leaf, -1, -1),
        "os.symlink": ("unused-target", leaf, -1),
    }[operation]
    with pytest.raises(PermissionError, match="outside"):
        guard(operation, args)


def test_directory_traversal_is_rejected(layout):
    guard, sandbox, _ = layout
    for path in (sandbox / "..", sandbox / ".." / "protected"):
        with pytest.raises(PermissionError, match="outside"):
            guard("os.remove", (path, -1))


@pytest.mark.parametrize("suffix", [os.sep, os.sep + ".", os.sep + ".."])
def test_directory_operand_cannot_hide_final_symlink(layout, monkeypatch, suffix):
    guard, sandbox, outside = layout
    link = sandbox / "redirect"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == link or path == link / "..":
            return outside
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(PermissionError, match="outside"):
        guard("os.remove", (str(link) + suffix, -1))


@pytest.mark.parametrize("source_external,destination_external", [(True, False), (False, True)])
def test_rename_checks_both_directory_entries(layout, source_external, destination_external):
    guard, sandbox, outside = layout
    source = (outside if source_external else sandbox) / "source"
    dest = (outside if destination_external else sandbox) / "dest"
    with pytest.raises(PermissionError, match="outside"):
        guard("os.rename", (source, dest, -1, -1))


@pytest.mark.parametrize("event", ["subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                                   "os.spawn", "socket.connect", "socket.bind", "socket.getaddrinfo"])
def test_process_and_network_access_remains_blocked(layout, event):
    guard, _, _ = layout
    with pytest.raises(PermissionError, match="processes and network"):
        guard(event, ())


@pytest.mark.parametrize("kind", ["loop", "dangling", "outward", "directory"])
def test_real_fixture_links_can_be_cleaned_without_changing_targets(layout, kind):
    guard, sandbox, outside = layout
    target = outside / "protected.txt"
    target.write_bytes(b"unchanged")
    link = sandbox / "link"
    destination = {
        "loop": link, "dangling": outside / "absent",
        "outward": target, "directory": outside,
    }[kind]
    _symlink_or_skip(link, destination, directory=kind == "directory")
    guard("os.remove", (link, -1))
    # The runner's actual audit hook is also active for this unlink.
    link.unlink()
    assert not link.is_symlink()
    assert target.read_bytes() == b"unchanged"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fd-relative rmtree path")
def test_fd_relative_unlink_uses_directory_handle_not_cwd(layout, monkeypatch):
    guard, sandbox, outside = layout
    link = sandbox / "loop"
    _symlink_or_skip(link, link)
    monkeypatch.chdir(outside)
    descriptor = os.open(sandbox, os.O_RDONLY)
    try:
        guard("os.remove", ("loop", descriptor))
        with pytest.raises(PermissionError, match="outside"):
            guard("os.remove", ("../outside/protected", descriptor))
        os.unlink("loop", dir_fd=descriptor)
    finally:
        os.close(descriptor)


def test_rmtree_cleans_cyclic_links_under_active_runner_guard(tmp_path):
    root = tmp_path / "cleanup"
    root.mkdir()
    _symlink_or_skip(root / "loop", root / "loop")
    shutil.rmtree(root)
    assert not root.exists()
