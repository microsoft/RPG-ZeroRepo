#!/usr/bin/env python3
"""Tests for CoderMind hook installation and status loading.

Verifies:
  - ``_install_claude_hooks`` writes a SessionStart hook that calls
    ``update_graphs.py status`` and merges with existing settings.
  - ``_install_copilot_hooks`` writes a VS Code task with
    ``runOptions.runOn = "folderOpen"``, is idempotent, and preserves
    pre-existing user tasks.
    - ``_install_hooks`` installs status integrations, removes only cmind
        Git blocks by default, and installs sync-only dispatchers on opt-in.
    - Workspace provider configuration is validated before provisioning or
        self-upgrade, without importing policy code from the workspace.
    - Repository recommendations never grant execution consent; explicit local
        choices are saved only after hooks succeed, in an isolated fake home.
  - ``update_graphs.py status`` returns RPG/dep-graph stats + an
    agent-facing MCP-tools reminder, on both populated and empty
    workspaces.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

# Ensure src/ and scripts/ are importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "scripts"))

import cmind_cli  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep real local-selection writes outside both the project and real HOME."""
    home = tmp_path.parent / f"{tmp_path.name}-home"
    home.mkdir()
    home = home.resolve()
    assert not home.is_relative_to(tmp_path.resolve())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A minimal CoderMind workspace with .cmind/scripts/update_graphs.py."""
    # Never follow a developer's global core.hooksPath out of tmp_path.
    monkeypatch.setattr(cmind_cli, "_read_core_hooks_path", lambda _: None)
    scripts_dir = tmp_path / ".cmind" / "scripts"
    scripts_dir.mkdir(parents=True)
    # The installers only need the file to exist; we copy the real script
    # so that subprocess invocations later in the test can actually run.
    src = _project_root / "scripts" / "update_graphs.py"
    (scripts_dir / "update_graphs.py").write_bytes(src.read_bytes())
    # Make `common/` and `rpg/` importable for the copied script.
    for pkg in ("common", "rpg"):
        (scripts_dir / pkg).mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Claude hook
# ---------------------------------------------------------------------------

def test_install_claude_hooks_writes_session_start(project):
    cmind_cli._install_claude_hooks(project)
    data = json.loads((project / ".claude" / "settings.json").read_text())
    assert "hooks" in data
    session_start = data["hooks"]["SessionStart"]
    assert isinstance(session_start, list) and len(session_start) == 1
    cmd = session_start[0]["hooks"][0]["command"]
    # Hook now invokes the global ``cmind`` CLI; no embedded sys.executable.
    assert "cmind script update_graphs.py status" in cmd
    # PATH fallback for GUI-launched session starts (VS Code / IDE git UI).
    assert "command -v cmind" in cmd
    assert cmd.endswith("status 2>/dev/null || echo '[CoderMind] RPG status unavailable'")


def test_install_claude_hooks_is_idempotent_across_python_upgrades(project, monkeypatch):
    """Re-installing must not stack duplicate SessionStart entries.

    Hooks no longer embed ``sys.executable``; they delegate to the
    globally-installed ``cmind`` CLI.  Re-running install therefore
    yields the exact same command and must remain a single entry
    (not a duplicate per invocation).
    """
    cmind_cli._install_claude_hooks(project)
    # Simulate an environment change; the hook body is
    # interpreter-independent so this should be a no-op.
    monkeypatch.setattr(cmind_cli.sys, "executable", "/opt/new-python/bin/python")
    cmind_cli._install_claude_hooks(project)
    data = json.loads((project / ".claude" / "settings.json").read_text())
    session_start = data["hooks"]["SessionStart"]
    cmind_entries = [
        e for e in session_start
        if any("update_graphs.py" in h.get("command", "") for h in e.get("hooks", []))
    ]
    assert len(cmind_entries) == 1
    cmd = cmind_entries[0]["hooks"][0]["command"]
    # Always uses the cmind-script form regardless of interpreter path.
    assert "cmind script update_graphs.py" in cmd
    assert "/opt/new-python/bin/python" not in cmd


def test_install_claude_hooks_shell_escapes_special_chars(project, monkeypatch):
    """Interpreter / workspace paths must not appear in the hook command.

    The hook body invokes the global ``cmind`` CLI directly, so paths
    with special characters cannot end up inside the command string.
    """
    monkeypatch.setattr(
        cmind_cli.sys, "executable", "/path with space/python"
    )
    cmind_cli._install_claude_hooks(project)
    cmd = (
        json.loads((project / ".claude" / "settings.json").read_text())
        ["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    )
    # No path leakage from the interpreter / workspace location.
    assert "/path with space" not in cmd
    assert "cmind script update_graphs.py" in cmd


def test_install_claude_hooks_merges_existing(project):
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {"matcher": "Write", "hooks": [{"type": "command", "command": "echo user"}]}
            ]
        },
        "customField": "preserve me",
    }))

    cmind_cli._install_claude_hooks(project)
    data = json.loads((claude_dir / "settings.json").read_text())
    # Existing event preserved
    assert data["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "echo user"
    # New event added
    assert "SessionStart" in data["hooks"]
    # Non-hooks user fields preserved
    assert data["customField"] == "preserve me"
    # Backup created
    assert (claude_dir / "settings.json.bak").is_file()


# ---------------------------------------------------------------------------
# Copilot hook
# ---------------------------------------------------------------------------

def test_install_copilot_hooks_writes_folder_open_task(project):
    cmind_cli._install_copilot_hooks(project)
    tasks = json.loads((project / ".vscode" / "tasks.json").read_text())
    assert tasks["version"] == "2.0.0"
    assert len(tasks["tasks"]) == 1
    t = tasks["tasks"][0]
    assert t["label"] == "CoderMind: load status"
    assert t["runOptions"] == {"runOn": "folderOpen"}
    # Task now invokes the global ``cmind`` CLI; args carry the
    # dispatcher subcommand + script relpath, with ``status`` last.
    assert t["command"] == "cmind"
    assert t["args"][0] == "script"
    assert t["args"][1] == "update_graphs.py"
    assert t["args"][-1] == "status"
    # Status output should appear silently — we don't want it stealing focus.
    assert t["presentation"]["reveal"] == "silent"
    # NOTE: .gitignore management was moved to `_setup_gitignore` (called
    # earlier in the init flow). `_install_copilot_hooks` no longer touches
    # .gitignore. See test_setup_gitignore_* for ignore-rule coverage.


def test_install_copilot_hooks_is_idempotent(project):
    cmind_cli._install_copilot_hooks(project)
    cmind_cli._install_copilot_hooks(project)
    tasks = json.loads((project / ".vscode" / "tasks.json").read_text())
    labels = [t["label"] for t in tasks["tasks"]]
    assert labels.count("CoderMind: load status") == 1


def test_install_copilot_hooks_preserves_user_tasks(project):
    vscode = project / ".vscode"
    vscode.mkdir()
    (vscode / "tasks.json").write_text(json.dumps({
        "version": "2.0.0",
        "tasks": [
            {"label": "user build", "type": "shell", "command": "make"},
        ],
    }))
    cmind_cli._install_copilot_hooks(project)
    tasks = json.loads((vscode / "tasks.json").read_text())
    labels = [t["label"] for t in tasks["tasks"]]
    assert "user build" in labels
    assert "CoderMind: load status" in labels


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_install_hooks_dispatches_to_copilot(project, monkeypatch):
    (project / ".git" / "hooks").mkdir(parents=True)

    cmind_cli._install_hooks(project, "copilot", tracker=None, git_hooks=True)

    # Copilot tasks.json present, Claude settings.json absent.
    assert (project / ".vscode" / "tasks.json").is_file()
    assert not (project / ".claude" / "settings.json").exists()
    hooks_dir = project / ".git" / "hooks"
    post_commit = (hooks_dir / "post-commit").read_text()
    post_merge = (hooks_dir / "post-merge").read_text()
    assert "CoderMind: post-commit dispatcher" in post_commit
    assert "cmind hook post-commit" in post_commit
    assert "CoderMind: post-merge dispatcher" in post_merge
    assert "cmind hook post-merge" in post_merge
    assert not (hooks_dir / "pre-commit").exists()


def test_install_hooks_dispatches_to_claude(project):
    (project / ".git" / "hooks").mkdir(parents=True)

    cmind_cli._install_hooks(project, "claude", tracker=None, git_hooks=True)

    assert (project / ".claude" / "settings.json").is_file()
    assert not (project / ".vscode" / "tasks.json").exists()
    hooks_dir = project / ".git" / "hooks"
    assert (hooks_dir / "post-commit").is_file()
    assert (hooks_dir / "post-merge").is_file()
    assert not (hooks_dir / "pre-commit").exists()


def test_update_command_invokes_install_hooks():
    """Regression tripwire: ``cmind update`` must call ``_install_hooks``.

    Hook installation belongs in the update flow alongside template,
    gitignore, and MCP config refreshes, so existing workspaces receive
    hook dispatcher fixes when users run ``cmind update``.

    This static assertion complements the mocked CLI tests below:
    deleting the ``_install_hooks(...)`` call must fail loudly without
    running provisioning or the optional CLI self-upgrade.
    """
    import inspect
    source = inspect.getsource(cmind_cli.update)
    assert "_install_hooks(" in source, (
        "cmind update must call _install_hooks(...); "
        "without it, hook upgrades never propagate to existing workspaces"
    )
    # And the tracker must declare a 'hooks' step so the user sees it
    # in the live progress output.
    assert '"hooks"' in source, (
        "cmind update tracker must declare a 'hooks' step"
    )


@pytest.fixture
def no_subprocess(monkeypatch):
    """Policy/installation tests must never launch processes or use the network."""
    blocked = Mock(side_effect=lambda *a, **kw: pytest.fail("Unexpected process or network call"))
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(cmind_cli.subprocess, name, blocked)
    monkeypatch.setattr(cmind_cli.os, "execvp", blocked)
    monkeypatch.setattr(cmind_cli.client, "request", blocked)
    return blocked


@pytest.mark.parametrize("command", ["init", "update"])
@pytest.mark.parametrize("flag,expected", [(None, False), ("--no-git-hooks", False), ("--git-hooks", True)])
def test_cli_passes_explicit_git_hook_choice(tmp_path, monkeypatch, no_subprocess, command, flag, expected):
    """Only the CLI flag enables hooks; all provisioning is stubbed locally."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cmind").mkdir()
    monkeypatch.setattr(cmind_cli, "show_banner", lambda: None)
    for name in (
        "download_and_extract_template", "_setup_gitignore",
        "ensure_cmind_runtime_dirs", "_maybe_offer_initial_encode",
    ):
        monkeypatch.setattr(cmind_cli, name, Mock())
    monkeypatch.setattr(cmind_cli, "_detect_install_method", lambda: "editable")
    monkeypatch.setattr(cmind_cli, "_install_source", lambda: "editable")
    monkeypatch.setattr(cmind_cli.shutil, "which", lambda _: "/installed/cmind")
    installer = Mock()
    monkeypatch.setattr(cmind_cli, "_install_hooks", installer)
    args = [command, "--ai", "copilot", "--script", "sh", "--no-mcp", "--no-cmind-git"]
    if command == "init":
        args += ["--here", "--force", "--no-git", "--no-encode", "--ignore-agent-tools"]
    else:
        args += ["--no-upgrade"]
    if flag:
        args.append(flag)

    result = CliRunner().invoke(cmind_cli.app, args)

    assert result.exit_code == 0, result.output
    installer.assert_called_once()
    assert installer.call_args.kwargs["git_hooks"] is expected
    assert cmind_cli._ai_cli_policy().read_local_provider(tmp_path) == "copilot"
    no_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Sentinel-block upgrade migration
# ---------------------------------------------------------------------------
#
# The installer must replace CoderMind-owned content by sentinel range or
# compatibility marker, while preserving user-authored shell lines.


def _hooks_dir(project):
    hd = project / ".git" / "hooks"
    hd.mkdir(parents=True, exist_ok=True)
    return hd


def test_pre_commit_v1_legacy_is_removed_on_upgrade(project):
    """A CoderMind-owned pre-commit snippet is removed during hook setup."""
    hd = _hooks_dir(project)
    (hd / "pre-commit").write_text(
        "#!/bin/sh\n"
        "# CoderMind: full RPG sync on commit\n"
        "/old/python /old/update_graphs.py sync 2>/dev/null || true\n"
    )

    assert cmind_cli._uninstall_git_pre_commit_hook(project) is True
    assert not (hd / "pre-commit").exists()


def test_post_commit_v1_legacy_is_replaced_on_upgrade(project):
    """A sync-only post-commit snippet upgrades to the dispatcher block."""
    hd = _hooks_dir(project)
    (hd / "post-commit").write_text(
        "#!/bin/sh\n"
        "# CoderMind: advance meta.git after commit\n"
        "/old/python /old/update_graphs.py sync 2>/dev/null || true\n"
    )

    assert cmind_cli._install_git_post_commit_hook(project) is True
    text = (hd / "post-commit").read_text()

    assert "# CoderMind: advance meta.git after commit" not in text
    assert "/old/python" not in text
    assert text.count("# CMIND-BEGIN post-commit") == 1
    assert text.count("# CMIND-END post-commit") == 1
    assert "CoderMind: post-commit dispatcher" in text
    assert "cmind hook post-commit" in text


def test_post_commit_v3_legacy_is_replaced_on_upgrade(project):
    """A multi-line post-commit snippet upgrades to the dispatcher block."""
    hd = _hooks_dir(project)
    old_body = (
        "#!/bin/sh\n"
        "# CoderMind: advance meta.git + background feature graph update\n"
        "/old/python /old/update_graphs.py sync 2>/dev/null || true\n"
        "if [ ! -f /old/.lock ]; then\n"
        '  setsid env -u GIT_INDEX_FILE -u GIT_DIR sh -c "cd /old; sleep 2; touch /old/.lock; '
        '/old/python /old/update_graphs.py update-rpg --json >> /old/log 2>&1; '
        'rm -f /old/.lock" </dev/null >/dev/null 2>&1 &\n'
        "fi\n"
    )
    (hd / "post-commit").write_text(old_body)

    assert cmind_cli._install_git_post_commit_hook(project) is True
    text = (hd / "post-commit").read_text()

    assert "/old/python" not in text
    assert "/old/.lock" not in text
    assert text.count("# CMIND-BEGIN post-commit") == 1
    assert text.count("# CMIND-END post-commit") == 1
    assert text.count("# CoderMind: post-commit dispatcher") == 1
    assert "cmind hook post-commit" in text


def test_install_is_idempotent_under_sentinels(project):
    """Repeated dispatcher installs must not stack sentinel blocks."""
    hd = _hooks_dir(project)
    cmind_cli._install_git_post_commit_hook(project)
    first = (hd / "post-commit").read_text()
    cmind_cli._install_git_post_commit_hook(project)
    cmind_cli._install_git_post_commit_hook(project)
    third = (hd / "post-commit").read_text()

    assert first == third
    assert third.count("# CMIND-BEGIN post-commit") == 1
    assert third.count("# CMIND-END post-commit") == 1


def test_sentinel_block_is_atomically_replaceable(project):
    """The sentinel-pair range is replaced wholesale on install."""
    hd = _hooks_dir(project)
    (hd / "post-commit").write_text(
        "#!/bin/sh\n"
        "\n"
        "# CMIND-BEGIN post-commit\n"
        "# CoderMind: post-commit dispatcher\n"
        "/some/older/path/python /some/older/script.py sync --legacy-flag\n"
        "# CMIND-END post-commit\n"
    )

    assert cmind_cli._install_git_post_commit_hook(project) is True
    text = (hd / "post-commit").read_text()

    assert "/some/older/path/python" not in text
    assert "--legacy-flag" not in text
    assert text.count("# CMIND-BEGIN post-commit") == 1
    assert text.count("# CMIND-END post-commit") == 1
    assert "cmind hook post-commit" in text


def test_user_authored_content_outside_block_is_preserved(project):
    """CoderMind owns only its sentinel block; user-authored shell lines before/after the block must survive an install/upgrade."""
    hd = _hooks_dir(project)
    (hd / "pre-commit").write_text(
        "#!/bin/sh\n"
        "echo 'user-prelude: about to commit' >&2\n"
        "# CMIND-BEGIN pre-commit\n"
        "# CoderMind: incremental RPG sync on commit\n"
        "/old/python /old/update_graphs.py sync --staged-only\n"
        "# CMIND-END pre-commit\n"
        "echo 'user-postlude: still going' >&2\n"
    )

    assert cmind_cli._uninstall_git_pre_commit_hook(project) is True
    text = (hd / "pre-commit").read_text()

    assert "user-prelude" in text
    assert "user-postlude" in text
    assert "/old/python" not in text
    assert "# CMIND-BEGIN pre-commit" not in text
    assert "# CMIND-END pre-commit" not in text


@pytest.mark.parametrize("selected_ai", ["claude", "copilot"])
def test_default_removes_all_cmind_blocks_preserving_user_bytes(project, selected_ai, no_subprocess):
    hd = _hooks_dir(project)
    before = b"#!/bin/sh\r\necho 'user before'\r\n"
    after = b"echo 'user after'\r\n\r\n"
    for name in ("pre-commit", "post-commit", "post-merge"):
        block = (
            f"# CMIND-BEGIN {name}\r\n"
            "cmind script update_graphs.py update-rpg --json &\r\n"
            f"# CMIND-END {name}\r\n"
        ).encode()
        (hd / name).write_bytes(before + block + after)

    cmind_cli._install_hooks(project, selected_ai)
    cmind_cli._install_hooks(project, selected_ai)

    for name in ("pre-commit", "post-commit", "post-merge"):
        assert (hd / name).read_bytes() == before + after
    if selected_ai == "claude":
        assert (project / ".claude" / "settings.json").is_file()
    else:
        assert (project / ".vscode" / "tasks.json").is_file()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("name,body", [
    ("pre-commit", "# CoderMind: full RPG sync on commit\n/old/python /old/update_graphs.py sync 2>/dev/null || true\n"),
    ("pre-commit", "# CoderMind: incremental RPG sync on commit\ncmind script update_graphs.py sync --staged-only\n"),
    ("post-commit", "# CoderMind: advance meta.git after commit\n/old/python /old/update_graphs.py sync 2>/dev/null || true\n"),
    ("post-commit", (
        "# CoderMind: advance meta.git + background feature graph update\n"
        "/old/python /old/update_graphs.py sync 2>/dev/null || true\n"
        "if [ ! -f /old/.lock ]; then\n"
        '  setsid env -u GIT_INDEX_FILE -u GIT_DIR sh -c "cd /old; sleep 2; touch /old/.lock; '
        '/old/python /old/update_graphs.py update-rpg --json >> /old/log 2>&1; '
        'rm -f /old/.lock" </dev/null >/dev/null 2>&1 &\n'
        "fi\n"
    )),
    ("post-merge", "# CoderMind: incremental RPG sync after merge / pull\ncmind script update_graphs.py sync\n"),
    ("post-commit", (
        "# CoderMind: advance meta.git + background feature graph update\n"
        "'/old python/python3.12' '/old scripts/update_graphs.py' sync 2>/dev/null || true\n"
        "if [ ! -f '/old logs/.lock' ]; then\n"
        '  setsid sh -c "\'/old python/python3.12\' \'/old scripts/update_graphs.py\' '
        'update-rpg --json" </dev/null >/dev/null 2>&1 &\n'
        "fi\n"
    )),
    *[(name, f"# CoderMind: {name} dispatcher\n{cmind_cli._HOOK_PATH_FALLBACK}\ncmind hook {name} 2>/dev/null || true\n")
      for name in ("pre-commit", "post-commit", "post-merge")],
])
def test_default_removes_legacy_bodies_without_consuming_user_lines(project, no_subprocess, name, body):
    hd = _hooks_dir(project)
    before = "#!/bin/sh\necho user-before\n"
    after = "echo user-after\necho still-user\n"
    (hd / name).write_bytes((before + body + after).encode())

    cmind_cli._install_hooks(project, "copilot")

    assert (hd / name).read_bytes() == (before + after).encode()
    no_subprocess.assert_not_called()


def test_default_deletes_owned_only_hooks(project, no_subprocess):
    hd = _hooks_dir(project)
    for name in ("pre-commit", "post-commit", "post-merge"):
        (hd / name).write_text(
            f"#!/bin/sh\n# CMIND-BEGIN {name}\ncmind hook {name}\n# CMIND-END {name}\n"
        )
    cmind_cli._install_hooks(project, "copilot")
    assert not any((hd / name).exists() for name in ("pre-commit", "post-commit", "post-merge"))


@pytest.mark.parametrize("body", [
    b"", b"#!/bin/sh\n", b"#!/bin/sh\r\necho user\r\n",
    b"#!/bin/sh\n# CMIND-BEGIN post-commit\necho keep-unmatched-tail\n",
    b"#!/bin/sh\n# CoderMind: advance meta.git after commit\necho not-a-cmind-body\n",
    b"#!/bin/sh\n# user mentions # CoderMind: advance meta.git after commit\necho user\n",
])
def test_default_leaves_unowned_or_ambiguous_hooks_untouched(project, no_subprocess, body):
    hd = _hooks_dir(project)
    path = hd / "post-commit"
    path.write_bytes(body)
    original_mode = path.stat().st_mode
    cmind_cli._install_hooks(project, "copilot")
    assert path.read_bytes() == body
    assert path.stat().st_mode == original_mode


def test_default_does_not_create_git_hooks_directory(project, no_subprocess):
    (project / ".git").mkdir()
    cmind_cli._install_hooks(project, "copilot")
    assert not (project / ".git" / "hooks").exists()


def test_default_cleans_git_hooks_even_if_status_integration_fails(project, monkeypatch, no_subprocess):
    hd = _hooks_dir(project)
    path = hd / "post-commit"
    path.write_text(
        "#!/bin/sh\n# CMIND-BEGIN post-commit\n"
        "cmind script update_graphs.py update-rpg &\n# CMIND-END post-commit\n"
    )
    monkeypatch.setattr(cmind_cli, "_install_copilot_hooks", Mock(side_effect=OSError("read-only tasks")))
    cmind_cli._install_hooks(project, "copilot")
    assert not path.exists()


@pytest.fixture
def legacy_hooks(project, no_subprocess):
    """Owned hooks, including the old AI worker, confined to tmp_path."""
    hd = _hooks_dir(project)
    bodies = {
        "pre-commit": (
            "# CoderMind: full RPG sync on commit\n"
            "cmind script update_graphs.py sync\n"
        ),
        "post-commit": (
            "# CoderMind: advance meta.git + background feature graph update\n"
            "cmind script update_graphs.py sync\n"
            "if [ ! -f .cmind-update.lock ]; then\n"
            "setsid cmind script update_graphs.py update-rpg --json >/dev/null 2>&1 &\n"
            "fi\n"
        ),
        "post-merge": (
            "# CoderMind: incremental RPG sync after merge / pull\n"
            "cmind script update_graphs.py sync\n"
        ),
    }
    for name, body in bodies.items():
        (hd / name).write_bytes(("#!/bin/sh\n" + body).encode("utf-8"))
    return hd


@pytest.mark.parametrize("git_hooks", [False, True])
@pytest.mark.parametrize("with_tracker", [False, True])
def test_invalid_utf8_hook_preserved_while_other_legacy_hooks_are_cleaned(
    project, legacy_hooks, monkeypatch, no_subprocess, git_hooks, with_tracker,
):
    path = legacy_hooks / "pre-commit"
    original = path.read_bytes() + b"# user bytes: \xff\r\n"
    path.write_bytes(original)
    original_mode = path.stat().st_mode
    cleanup = Mock(wraps=cmind_cli._uninstall_git_hook)
    monkeypatch.setattr(cmind_cli, "_uninstall_git_hook", cleanup)
    blocked = Mock()
    for name in (
        "_install_git_post_commit_hook", "_install_git_post_merge_hook",
        "_install_copilot_hooks", "_install_claude_hooks",
    ):
        monkeypatch.setattr(cmind_cli, name, blocked)
    tracker = Mock() if with_tracker else None

    with pytest.raises(RuntimeError) as exc:
        cmind_cli._install_hooks(project, "copilot", tracker=tracker, git_hooks=git_hooks)

    assert str(exc.value) == cmind_cli._GIT_HOOK_CLEANUP_ERROR
    assert [call.args[1] for call in cleanup.call_args_list] == [
        "pre-commit", "post-commit", "post-merge",
    ]
    assert path.read_bytes() == original
    assert path.stat().st_mode == original_mode
    assert not (legacy_hooks / "post-commit").exists()
    assert not (legacy_hooks / "post-merge").exists()
    blocked.assert_not_called()
    if tracker is not None:
        tracker.error.assert_called_once_with("hooks", cmind_cli._GIT_HOOK_CLEANUP_ERROR)
        tracker.complete.assert_not_called()
        tracker.skip.assert_not_called()
    assert not cmind_cli._ai_cli_policy().local_selection_path(project).exists()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("git_hooks", [False, True])
@pytest.mark.parametrize("failed_hook", ["pre-commit", "post-commit", "post-merge"])
@pytest.mark.parametrize("operation", ["read_bytes", "unlink", "write_bytes"])
def test_hook_permission_error_does_not_skip_other_cleanup(
    project, legacy_hooks, monkeypatch, no_subprocess, git_hooks, failed_hook, operation,
):
    names = ("pre-commit", "post-commit", "post-merge")
    originals = {}
    for name in names:
        path = legacy_hooks / name
        if operation == "write_bytes":
            # Retaining user content forces a write rather than an unlink.
            path.write_bytes(path.read_bytes() + b"echo user\r\n")
        originals[name] = path.read_bytes()
    original_operation = getattr(Path, operation)
    attempted = []

    def denied(path, *args, **kwargs):
        if path.parent == legacy_hooks:
            attempted.append(path.name)
            if path.name == failed_hook:
                raise PermissionError("UNTRUSTED_HOOK_ERROR")
        return original_operation(path, *args, **kwargs)

    blocked = Mock()
    for name in (
        "_install_git_post_commit_hook", "_install_git_post_merge_hook",
        "_install_copilot_hooks",
    ):
        monkeypatch.setattr(cmind_cli, name, blocked)
    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, denied)
        with pytest.raises(RuntimeError) as exc:
            cmind_cli._install_hooks(project, "copilot", git_hooks=git_hooks)

    assert str(exc.value) == cmind_cli._GIT_HOOK_CLEANUP_ERROR
    assert attempted == list(names)
    for name in names:
        path = legacy_hooks / name
        if name == failed_hook:
            assert path.read_bytes() == originals[name]
        elif operation == "write_bytes":
            assert path.read_bytes() == b"#!/bin/sh\necho user\r\n"
        else:
            assert not path.exists()
    blocked.assert_not_called()
    assert not cmind_cli._ai_cli_policy().local_selection_path(project).exists()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("failed_hook", ["post-commit", "post-merge"])
def test_opt_in_installer_error_is_fatal_and_stops_further_installation(
    project, legacy_hooks, monkeypatch, no_subprocess, failed_hook,
):
    installers = {}
    for hook_name in ("post-commit", "post-merge"):
        attr = f"_install_git_{hook_name.replace('-', '_')}_hook"
        installer = Mock(wraps=getattr(cmind_cli, attr))
        if hook_name == failed_hook:
            installer.side_effect = PermissionError("UNTRUSTED_HOOK_ERROR")
        monkeypatch.setattr(cmind_cli, attr, installer)
        installers[hook_name] = installer
    status = Mock()
    monkeypatch.setattr(cmind_cli, "_install_copilot_hooks", status)

    with pytest.raises(RuntimeError) as exc:
        cmind_cli._install_hooks(project, "copilot", git_hooks=True)

    assert str(exc.value) == cmind_cli._GIT_HOOK_INSTALL_ERROR
    installers["post-commit"].assert_called_once_with(project)
    if failed_hook == "post-commit":
        installers["post-merge"].assert_not_called()
        assert not (legacy_hooks / "post-commit").exists()
    else:
        installers["post-merge"].assert_called_once_with(project)
        assert "cmind hook post-commit" in (legacy_hooks / "post-commit").read_text()
    assert not (legacy_hooks / "pre-commit").exists()
    assert not (legacy_hooks / "post-merge").exists()
    status.assert_not_called()
    assert not cmind_cli._ai_cli_policy().local_selection_path(project).exists()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("command", ["init", "update"])
@pytest.mark.parametrize("existing_provider", [None, "claude"])
@pytest.mark.parametrize("git_hooks,failure", [
    (False, "invalid-utf8"), (False, "read-error"),
    (True, "invalid-utf8"), (True, "read-error"),
    (True, "post-commit"), (True, "post-merge"),
])
def test_cli_hook_reconciliation_failure_prevents_success_and_encode(
    project, legacy_hooks, monkeypatch, no_subprocess, command, git_hooks, failure,
    existing_provider,
):
    policy = cmind_cli._ai_cli_policy()
    selection = policy.local_selection_path(project)
    if existing_provider:
        policy.write_local_provider(project, existing_provider)
    original_selection = selection.read_bytes() if selection.exists() else None
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)
    monkeypatch.chdir(project)
    monkeypatch.setattr(cmind_cli, "show_banner", lambda: None)
    for name in (
        "download_and_extract_template", "_setup_gitignore", "ensure_cmind_runtime_dirs",
    ):
        monkeypatch.setattr(cmind_cli, name, Mock())
    monkeypatch.setattr(cmind_cli, "_detect_install_method", lambda: "editable")
    monkeypatch.setattr(cmind_cli, "_install_source", lambda: "editable")
    monkeypatch.setattr(cmind_cli.shutil, "which", lambda _: "/installed/cmind")
    encode = Mock()
    monkeypatch.setattr(cmind_cli, "_maybe_offer_initial_encode", encode)
    status = Mock()
    monkeypatch.setattr(cmind_cli, "_install_copilot_hooks", status)
    pre_commit = legacy_hooks / "pre-commit"
    original = pre_commit.read_bytes()
    if failure == "invalid-utf8":
        original += b"# user bytes: \xff\r\n"
        pre_commit.write_bytes(original)
    elif failure == "read-error":
        read_bytes = Path.read_bytes

        def denied(path):
            if path.resolve() == pre_commit.resolve():
                raise PermissionError("UNTRUSTED_HOOK_ERROR")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", denied)
    else:
        monkeypatch.setattr(
            cmind_cli, f"_install_git_{failure.replace('-', '_')}_hook",
            Mock(side_effect=PermissionError("UNTRUSTED_HOOK_ERROR")),
        )
    args = [command, "--ai", "copilot", "--script", "sh", "--no-mcp", "--no-cmind-git"]
    if command == "init":
        args += ["--here", "--force", "--no-git", "--encode", "--ignore-agent-tools"]
    else:
        args += ["--no-upgrade"]
    if git_hooks:
        args.append("--git-hooks")

    result = CliRunner().invoke(cmind_cli.app, args)

    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    if failure in ("invalid-utf8", "read-error"):
        message = "Could not complete CoderMind Git hook cleanup."
        with pre_commit.open("rb") as stream:
            assert stream.read() == original
        assert not (legacy_hooks / "post-commit").exists()
    else:
        message = "Could not install CoderMind Git sync hooks."
    assert message in output
    assert ("Initialization failed:" if command == "init" else "Update failed:") in output
    assert "UNTRUSTED_HOOK_ERROR" not in output
    assert "Project ready." not in output
    assert "updated successfully" not in output
    assert not (legacy_hooks / "post-merge").exists()
    encode.assert_not_called()
    status.assert_not_called()
    save.assert_not_called()
    if original_selection is None:
        assert not selection.exists()
    else:
        assert selection.read_bytes() == original_selection
    no_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Dispatcher: all subprocesses mocked, all log writes under tmp_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["post-commit", "post-merge"])
@pytest.mark.parametrize("sync_result", [0, 1, "os-error"])
def test_dispatcher_only_runs_deterministic_sync(tmp_path, monkeypatch, name, sync_result):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CMIND_AI_CLI_CMD", "UNTRUSTED_CUSTOM_COMMAND")
    monkeypatch.setattr(cmind_cli._storage, "find_workspace_root_from", lambda _: tmp_path)
    logs = tmp_path / "home-store" / "logs"
    monkeypatch.setattr(cmind_cli._storage, "workspace_logs_dir", lambda _: logs)
    blocked = Mock(side_effect=AssertionError("No background or shell process allowed"))
    for entry in ("Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(cmind_cli.subprocess, entry, blocked)

    def run(args, **kwargs):
        if args == ["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="abc123\n")
        assert args == ["cmind", "script", "update_graphs.py", "sync"]
        assert kwargs["env"]["CMIND_HOOK"] == name
        assert kwargs["env"]["CMIND_HOOK_SHA"] == "abc123"
        assert not kwargs.get("shell", False)
        if sync_result == "os-error":
            raise OSError("simulated unavailable cmind")
        return subprocess.CompletedProcess(args, sync_result)

    runner = Mock(side_effect=run)
    monkeypatch.setattr(cmind_cli.subprocess, "run", runner)
    with pytest.raises(cmind_cli.typer.Exit) as exc:
        cmind_cli.hook(name)
    assert exc.value.exit_code == 0
    assert [call.args[0] for call in runner.call_args_list] == [
        ["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"],
        ["cmind", "script", "update_graphs.py", "sync"],
    ]
    assert runner.call_args.kwargs["env"]["CMIND_HOOK"] == name
    assert runner.call_args.kwargs["env"]["CMIND_HOOK_SHA"] == "abc123"
    assert not runner.call_args.kwargs.get("shell", False)
    blocked.assert_not_called()
    assert (logs / "hooks.log").is_file()
    assert not (logs / "update_rpg.log").exists()
    assert not (logs / ".update_rpg.lock").exists()


# ---------------------------------------------------------------------------
# Shared policy loading and repo-only config validation
# ---------------------------------------------------------------------------

def test_policy_loader_ignores_workspace_and_sys_path(tmp_path, monkeypatch, no_subprocess):
    from cmind_cli import _assets

    for parent in (tmp_path, tmp_path / ".cmind" / "scripts"):
        common = parent / "common"
        common.mkdir(parents=True)
        (common / "__init__.py").write_text("raise AssertionError('untrusted common')\n")
        (common / "ai_cli_policy.py").write_text("raise AssertionError('untrusted policy')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "common.ai_cli_policy", ModuleType("common.ai_cli_policy"))
    cmind_cli._ai_cli_policy.cache_clear()
    try:
        policy = cmind_cli._ai_cli_policy()
        assert Path(policy.__file__).resolve() == (_assets.scripts_dir() / "common" / "ai_cli_policy.py").resolve()
        assert policy.validate_provider("copilot") == "copilot"
        assert cmind_cli._ai_cli_policy() is policy
        assert cmind_cli._AI_TO_CLI_CMD == {
            provider: " ".join(argv) for provider, argv in policy.PROVIDER_ARGV.items()
        }
    finally:
        cmind_cli._ai_cli_policy.cache_clear()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("provider", list(cmind_cli._AI_TO_CLI_CMD))
def test_workspace_config_writes_recommendation_only(tmp_path, no_subprocess, provider):
    cmind_cli._write_workspace_config(tmp_path, provider)
    text = (tmp_path / ".cmind" / "config.toml").read_text(encoding="utf-8")
    assert cmind_cli.tomllib.loads(text) == {"cmind": {"recommended_provider": provider}}
    assert "ai_cli_cmd" not in text
    assert "Never execution authority" in text
    policy = cmind_cli._ai_cli_policy()
    assert policy.resolve_provider(tmp_path, environ={}) == ""
    assert not policy.local_selection_path(tmp_path).exists()


@pytest.mark.parametrize("content", [
    b"# recommendation only\r\n[cmind]\r\nrecommended_provider = 'claude'\r\n",
    b"# preserve formatting\r\n[cmind]\r\nai_provider = 'claude'\r\n",
    b"# marker only\n[cmind]\n",
    b"# no provider yet\n[other]\nsetting = 42\n",
    *[f"[cmind]\nai_cli_cmd = '{command}'\n".encode() for command in cmind_cli._AI_TO_CLI_CMD.values()],
])
def test_workspace_config_preserves_valid_existing_bytes(tmp_path, no_subprocess, capsys, content):
    config = tmp_path / ".cmind" / "config.toml"
    config.parent.mkdir()
    config.write_bytes(content)
    cmind_cli._write_workspace_config(tmp_path, "copilot")
    assert config.read_bytes() == content
    output = " ".join(capsys.readouterr().out.split())
    if b"ai_provider" in content or b"ai_cli_cmd" in content:
        assert "legacy workspace AI hint preserved; it is not execution authority" in output
    else:
        assert "legacy workspace AI hint" not in output
    assert not cmind_cli._ai_cli_policy().local_selection_path(tmp_path).exists()


@pytest.mark.parametrize("content", [
    "[cmind", 'cmind = "not a table"\n',
    '[cmind]\nrecommended_provider = "UNTRUSTED_CUSTOM_COMMAND"\n',
    '[cmind]\nrecommended_provider = ["copilot"]\n',
    '[cmind]\nrecommended_provider = "copilot"\nai_provider = "claude"\n',
    '[cmind]\nrecommended_provider = "copilot"\nai_cli_cmd = "copilot"\n',
    '[cmind]\nai_provider = "UNTRUSTED_CUSTOM_COMMAND"\n',
    '[cmind]\nai_provider = ["copilot"]\n',
    '[cmind]\nai_provider = "claude"\nai_cli_cmd = "claude"\n',
    '[cmind]\nai_cli_cmd = "/tmp/claude"\n',
    '[cmind]\nai_cli_cmd = "claude --extra-option"\n',
    '[cmind]\nai_cli_cmd = " copilot "\n',
    '[cmind]\nai_cli_cmd = 42\n',
])
def test_workspace_config_rejects_invalid_without_rewriting(tmp_path, no_subprocess, content):
    config = tmp_path / ".cmind" / "config.toml"
    config.parent.mkdir()
    original = content.encode()
    config.write_bytes(original)
    with pytest.raises(cmind_cli._ai_cli_policy().AICommandPolicyError):
        cmind_cli._write_workspace_config(tmp_path, "copilot")
    assert config.read_bytes() == original


def test_workspace_config_rejects_unknown_selected_provider_without_creating_files(tmp_path, no_subprocess):
    with pytest.raises(cmind_cli._ai_cli_policy().AICommandPolicyError):
        cmind_cli._write_workspace_config(tmp_path, 'custom"\ncommand')
    assert not (tmp_path / ".cmind").exists()


@pytest.mark.parametrize("command", ["init", "update"])
def test_invalid_config_preflight_precedes_any_provisioning_or_upgrade(tmp_path, monkeypatch, no_subprocess, command):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / ".cmind" / "config.toml"
    config.parent.mkdir()
    original = b'[cmind]\nai_cli_cmd = "UNTRUSTED_CUSTOM_COMMAND --flag"\n'
    config.write_bytes(original)
    monkeypatch.setattr(cmind_cli, "show_banner", lambda: None)
    blocked = Mock(side_effect=lambda *a, **kw: pytest.fail("Side effect before policy rejection"))
    for name in (
        "download_and_extract_template", "_write_source_marker", "_write_workspace_config",
        "_setup_gitignore", "_generate_mcp_config", "_register_copilot_cli_global_mcp",
        "_install_hooks", "ensure_cmind_runtime_dirs", "_detect_install_method", "_install_source",
        "_upgrade_command", "check_tool", "select_with_arrows",
    ):
        monkeypatch.setattr(cmind_cli, name, blocked)
    monkeypatch.setattr(cmind_cli.typer, "confirm", blocked)
    monkeypatch.setattr(cmind_cli.os, "execvp", blocked)
    policy = cmind_cli._ai_cli_policy()
    monkeypatch.setattr(policy, "write_local_provider", blocked)
    args = [command, "--ai", "copilot"]
    if command == "init":
        args.append("--here")

    result = CliRunner().invoke(cmind_cli.app, args)

    assert result.exit_code == 1, result.output
    assert " ".join(result.output.split()) == cmind_cli._AI_POLICY_ERROR
    assert "UNTRUSTED_CUSTOM_COMMAND" not in result.output
    assert config.read_bytes() == original
    assert list(tmp_path.iterdir()) == [config.parent]
    assert not policy.local_selection_path(tmp_path).exists()
    blocked.assert_not_called()
    no_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Explicit local consent versus integration-only init/update behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def cli_side_effects(project, monkeypatch, no_subprocess):
    """Keep real policy/file writes, but stub all provisioning and execution."""
    monkeypatch.chdir(project)
    monkeypatch.setattr(cmind_cli, "show_banner", lambda: None)
    # CliRunner replaces sys.stdin. A module-local view lets these tests model
    # TTY consent without changing Click's captured input or running a prompt.
    cli_sys = SimpleNamespace(**vars(sys))
    cli_sys.stdin = Mock()
    cli_sys.stdin.isatty.return_value = False
    monkeypatch.setattr(cmind_cli, "sys", cli_sys)
    monkeypatch.setattr(cmind_cli, "_detect_install_method", Mock(return_value="editable"))
    monkeypatch.setattr(cmind_cli, "_install_source", Mock(return_value="editable"))
    monkeypatch.setattr(cmind_cli.shutil, "which", lambda _: "/installed/cmind")
    mocks = {}
    for name in (
        "download_and_extract_template", "_write_source_marker", "_setup_gitignore",
        "ensure_cmind_runtime_dirs", "_generate_mcp_config", "_register_copilot_cli_global_mcp",
        "_install_hooks", "_maybe_offer_initial_encode", "check_tool", "is_git_repo", "init_git_repo",
    ):
        mocks[name] = Mock()
        monkeypatch.setattr(cmind_cli, name, mocks[name])
    unexpected_prompt = Mock(side_effect=AssertionError("Unexpected interactive prompt"))
    monkeypatch.setattr(cmind_cli, "select_with_arrows", unexpected_prompt)
    monkeypatch.setattr(cmind_cli.typer, "confirm", unexpected_prompt)
    mocks["prompt"] = unexpected_prompt
    return mocks


def _consent_cli_args(command, ai="copilot", *, encode=False):
    args = [command, "--script", "sh", "--no-mcp", "--no-cmind-git"]
    if ai is not None:
        args += ["--ai", ai]
    if command == "init":
        args += ["--here", "--force", "--no-git", "--ignore-agent-tools"]
        args.append("--encode" if encode else "--no-encode")
    else:
        args.append("--no-upgrade")
    return args


@pytest.mark.parametrize("command", ["init", "update"])
@pytest.mark.parametrize("hint_key", ["recommended_provider", "ai_provider", "ai_cli_cmd"])
@pytest.mark.parametrize("selected,hint", [("copilot", "claude"), ("claude", "copilot")])
def test_explicit_cli_provider_wins_over_repository_hint(
    project, monkeypatch, cli_side_effects, no_subprocess, command, hint_key, selected, hint,
):
    policy = cmind_cli._ai_cli_policy()
    config = project / ".cmind" / "config.toml"
    original = f"# keep this hint\r\n[cmind]\r\n{hint_key} = '{hint}'\r\n".encode()
    config.write_bytes(original)
    selection = policy.local_selection_path(project)

    def hooks(*args, **kwargs):
        assert not selection.exists()
        assert config.read_bytes() == original

    cli_side_effects["_install_hooks"].side_effect = hooks
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args(command, selected))

    assert result.exit_code == 0, result.output
    save.assert_called_once()
    assert save.call_args.args[0].resolve() == project.resolve()
    assert save.call_args.args[1] == selected
    assert policy.read_local_provider(project) == selected
    assert policy.resolve_provider(project, environ={}) == selected
    assert config.read_bytes() == original
    assert cli_side_effects["download_and_extract_template"].call_args.args[1] == selected
    assert f"User-local AI provider saved: {selected}" in " ".join(result.output.split())
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("new_directory", [False, True])
@pytest.mark.parametrize("existing_provider", [None, "claude"])
def test_non_tty_init_without_ai_rejects_before_any_write_or_prompt(
    project, isolated_home, monkeypatch, cli_side_effects, no_subprocess,
    new_directory, existing_provider,
):
    policy = cmind_cli._ai_cli_policy()
    config = project / ".cmind" / "config.toml"
    config.write_bytes(b'[cmind]\nrecommended_provider = "copilot"\n')
    target = project / "new-project" if new_directory else project
    if existing_provider:
        policy.write_local_provider(target, existing_provider)
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)
    before = {p.relative_to(project): p.read_bytes() for p in project.rglob("*") if p.is_file()}
    home_before = {p.relative_to(isolated_home): p.read_bytes() for p in isolated_home.rglob("*") if p.is_file()}
    # Do not pass --force: rejection must happen even before the merge prompt.
    args = ["init", "new-project"] if new_directory else ["init", "--here"]

    result = CliRunner().invoke(cmind_cli.app, args)

    assert result.exit_code == 1, result.output
    assert "non-interactive init requires --ai copilot or --ai claude" in " ".join(result.output.split())
    assert {p.relative_to(project): p.read_bytes() for p in project.rglob("*") if p.is_file()} == before
    assert {p.relative_to(isolated_home): p.read_bytes() for p in isolated_home.rglob("*") if p.is_file()} == home_before
    if new_directory:
        assert not target.exists()
    assert policy.read_local_provider(target) == (existing_provider or "")
    if not existing_provider:
        assert not policy.local_selection_path(target).exists()
        assert not (isolated_home / ".cmind").exists()
    for mock in cli_side_effects.values():
        mock.assert_not_called()
    save.assert_not_called()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("hooks_succeed", [False, True])
def test_interactive_init_choice_is_explicit_consent_only_after_hooks(
    project, monkeypatch, cli_side_effects, no_subprocess, hooks_succeed,
):
    policy = cmind_cli._ai_cli_policy()
    selection = policy.local_selection_path(project)
    cmind_cli.sys.stdin.isatty.return_value = True
    choose = Mock(return_value="claude")
    monkeypatch.setattr(cmind_cli, "select_with_arrows", choose)
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    def hooks(*args, **kwargs):
        assert not selection.exists()
        if not hooks_succeed:
            raise RuntimeError(cmind_cli._GIT_HOOK_CLEANUP_ERROR)

    cli_side_effects["_install_hooks"].side_effect = hooks
    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("init", None, encode=True))

    choose.assert_called_once()
    assert "provider to save locally" in choose.call_args.args[1]
    assert result.exit_code == (0 if hooks_succeed else 1), result.output
    if hooks_succeed:
        save.assert_called_once()
        assert policy.read_local_provider(project) == "claude"
    else:
        save.assert_not_called()
        assert not selection.exists()
        assert "Project ready." not in result.output
        cli_side_effects["_maybe_offer_initial_encode"].assert_not_called()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("hint_key", ["recommended_provider", "ai_provider", "ai_cli_cmd"])
def test_update_auto_detection_never_creates_local_consent(
    project, isolated_home, monkeypatch, cli_side_effects, no_subprocess, hint_key,
):
    policy = cmind_cli._ai_cli_policy()
    config = project / ".cmind" / "config.toml"
    original = f"[cmind]\n{hint_key} = 'copilot'\n".encode()
    config.write_bytes(original)
    (project / ".claude").mkdir()
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("update", None))

    assert result.exit_code == 0, result.output
    assert cli_side_effects["download_and_extract_template"].call_args.args[1] == "claude"
    assert policy.resolve_provider(project, environ={}) == ""
    assert not policy.local_selection_path(project).exists()
    assert not (isolated_home / ".cmind" / "execution").exists()
    assert config.read_bytes() == original
    assert "Integration only: user-local AI selection will not be changed" in " ".join(result.output.split())
    save.assert_not_called()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("provider,expected_integration", [("claude", "claude"), ("gemini", "copilot")])
def test_update_without_ai_preserves_local_bytes_and_prefers_supported_local_provider(
    project, monkeypatch, cli_side_effects, no_subprocess, provider, expected_integration,
):
    policy = cmind_cli._ai_cli_policy()
    policy.write_local_provider(project, provider)
    selection = policy.local_selection_path(project)
    # Valid but non-default formatting detects even a same-value rewrite.
    original = (json.dumps(json.loads(selection.read_text(encoding="utf-8")), indent=2) + "\r\n").encode()
    selection.write_bytes(original)
    config = project / ".cmind" / "config.toml"
    hint = b"[cmind]\nrecommended_provider = 'copilot'\n"
    config.write_bytes(hint)
    (project / ".github").mkdir()
    detect = Mock(wraps=cmind_cli._detect_ai_agent)
    monkeypatch.setattr(cmind_cli, "_detect_ai_agent", detect)
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("update", None))

    assert result.exit_code == 0, result.output
    assert cli_side_effects["download_and_extract_template"].call_args.args[1] == expected_integration
    assert selection.read_bytes() == original
    assert policy.read_local_provider(project) == provider
    assert config.read_bytes() == hint
    if provider in cmind_cli.AGENT_CONFIG:
        detect.assert_not_called()
    else:
        detect.assert_called_once()
    save.assert_not_called()
    no_subprocess.assert_not_called()


def test_update_non_tty_with_only_repo_hint_fails_actionably_without_prompt(
    project, monkeypatch, cli_side_effects, no_subprocess,
):
    policy = cmind_cli._ai_cli_policy()
    config = project / ".cmind" / "config.toml"
    original = b"[cmind]\nrecommended_provider = 'copilot'\n"
    config.write_bytes(original)
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("update", None))

    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    assert "cannot determine AI integration in non-interactive update" in output
    assert "--ai copilot or --ai claude" in output
    assert config.read_bytes() == original
    assert not policy.local_selection_path(project).exists()
    for mock in cli_side_effects.values():
        mock.assert_not_called()
    cmind_cli._detect_install_method.assert_not_called()
    save.assert_not_called()
    no_subprocess.assert_not_called()


def test_update_interactive_choice_is_integration_only(
    project, monkeypatch, cli_side_effects, no_subprocess,
):
    policy = cmind_cli._ai_cli_policy()
    cmind_cli.sys.stdin.isatty.return_value = True
    choose = Mock(return_value="copilot")
    monkeypatch.setattr(cmind_cli, "select_with_arrows", choose)
    save = Mock(wraps=policy.write_local_provider)
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("update", None))

    assert result.exit_code == 0, result.output
    choose.assert_called_once()
    assert "integration only" in choose.call_args.args[1]
    assert not policy.local_selection_path(project).exists()
    assert policy.resolve_provider(project, environ={}) == ""
    save.assert_not_called()
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("command", ["init", "update"])
def test_explicit_provider_change_follows_hooks_and_precedes_success_and_encode(
    project, monkeypatch, cli_side_effects, no_subprocess, command,
):
    policy = cmind_cli._ai_cli_policy()
    policy.write_local_provider(project, "claude")
    selection = policy.local_selection_path(project)
    original = selection.read_bytes()
    events = []
    write_local = policy.write_local_provider
    complete = cmind_cli.StepTracker.complete

    def hooks(*args, **kwargs):
        assert selection.read_bytes() == original
        events.append("hooks")

    def save(workspace, provider):
        assert events == ["hooks"]
        assert selection.read_bytes() == original
        write_local(workspace, provider)
        events.append("save")

    def completed(tracker, key, detail=""):
        if key == "final":
            assert events == ["hooks", "save"]
            assert policy.read_local_provider(project) == "copilot"
            events.append("final")
        return complete(tracker, key, detail)

    def encode(*args, **kwargs):
        assert kwargs["encode_choice"] is True
        assert events == ["hooks", "save", "final"]
        assert policy.read_local_provider(project) == "copilot"
        events.append("encode")

    cli_side_effects["_install_hooks"].side_effect = hooks
    cli_side_effects["_maybe_offer_initial_encode"].side_effect = encode
    monkeypatch.setattr(policy, "write_local_provider", save)
    monkeypatch.setattr(cmind_cli.StepTracker, "complete", completed)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args(command, encode=True))

    assert result.exit_code == 0, result.output
    assert events == ["hooks", "save", "final"] + (["encode"] if command == "init" else [])
    assert policy.read_local_provider(project) == "copilot"
    assert selection.read_bytes() != original
    no_subprocess.assert_not_called()


@pytest.mark.parametrize("command", ["init", "update"])
@pytest.mark.parametrize("existing_provider", [None, "claude"])
@pytest.mark.parametrize("failure", ["os-error", "policy-error", "atomic-replace"])
def test_local_selection_write_failure_is_fatal_before_success_or_encode(
    project, monkeypatch, cli_side_effects, no_subprocess, command, existing_provider, failure,
):
    policy = cmind_cli._ai_cli_policy()
    selection = policy.local_selection_path(project)
    if existing_provider:
        policy.write_local_provider(project, existing_provider)
    original = selection.read_bytes() if selection.exists() else None
    save = Mock(wraps=policy.write_local_provider)
    if failure == "atomic-replace":
        monkeypatch.setattr(policy.os, "replace", Mock(side_effect=PermissionError("PRIVATE_LOCAL_ERROR")))
    elif failure == "policy-error":
        save.side_effect = policy.AICommandPolicyError("PRIVATE_LOCAL_ERROR")
    else:
        save.side_effect = PermissionError("PRIVATE_LOCAL_ERROR")
    monkeypatch.setattr(policy, "write_local_provider", save)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args(command, encode=True))

    assert result.exit_code == 1, result.output
    output = " ".join(result.output.replace("│", " ").split())
    assert "Cannot read or save user-local AI selection" in output
    assert "--ai copilot or --ai claude" in output
    assert "PRIVATE_LOCAL_ERROR" not in output
    assert "Project ready." not in output
    assert "updated successfully" not in output
    assert "User-local AI provider saved" not in output
    cli_side_effects["_install_hooks"].assert_called_once()
    cli_side_effects["_maybe_offer_initial_encode"].assert_not_called()
    save.assert_called_once()
    if original is None:
        assert not selection.exists()
    else:
        assert selection.read_bytes() == original
    assert not list(selection.parent.glob(".selection-*.tmp"))
    no_subprocess.assert_not_called()


def test_invalid_local_selection_blocks_implicit_update_without_overwriting(
    project, monkeypatch, cli_side_effects, no_subprocess,
):
    policy = cmind_cli._ai_cli_policy()
    selection = policy.local_selection_path(project)
    selection.parent.mkdir(parents=True)
    original = b'{"ai_provider": "claude"}\n'
    selection.write_bytes(original)
    (project / ".github").mkdir()
    detect = Mock(wraps=cmind_cli._detect_ai_agent)
    monkeypatch.setattr(cmind_cli, "_detect_ai_agent", detect)

    result = CliRunner().invoke(cmind_cli.app, _consent_cli_args("update", None))

    assert result.exit_code == 1, result.output
    assert "Cannot read or save user-local AI selection" in " ".join(result.output.split())
    assert selection.read_bytes() == original
    detect.assert_not_called()
    for mock in cli_side_effects.values():
        mock.assert_not_called()
    cmind_cli._detect_install_method.assert_not_called()
    no_subprocess.assert_not_called()


def test_source_marker_and_storage_updates_cannot_erase_local_selection(project, no_subprocess):
    policy = cmind_cli._ai_cli_policy()
    policy.write_local_provider(project, "claude")
    selection = policy.local_selection_path(project)
    original = selection.read_bytes()

    cmind_cli._write_workspace_config(project, "copilot")
    for channel in (cmind_cli._SOURCE_BUNDLE, cmind_cli._SOURCE_LEGACY):
        cmind_cli._write_source_marker(project, channel)
        assert cmind_cli._read_source_marker(project) == channel
        assert selection.read_bytes() == original
        assert policy.read_local_provider(project) == "claude"
        assert not selection.is_relative_to(cmind_cli._storage.home_workspace_dir(project))
        assert "ai_provider" not in cmind_cli._storage.read_meta(project)
    assert policy.read_workspace_provider(project) == "copilot"
    assert policy.resolve_provider(project, environ={}) == "claude"
    no_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# update_graphs.py status
# ---------------------------------------------------------------------------

def _run_status(workspace: Path, json_mode: bool = False) -> subprocess.CompletedProcess:
    """Run the real source ``update_graphs.py status`` with explicit ``--rpg`` and ``--dep-graph`` paths pointing into ``workspace``.

    We invoke the source script (not the copy in ``workspace/.cmind/
    scripts``) so the test doesn't need to vendor the ``common/`` and
    ``rpg/`` packages alongside it.
    """
    data_dir = workspace / ".cmind" / "data"
    cmd = [
        sys.executable,
        str(_project_root / "scripts" / "update_graphs.py"),
        "status",
        "--rpg", str(data_dir / "rpg.json"),
        "--dep-graph", str(data_dir / "dep_graph.json"),
    ]
    if json_mode:
        cmd.append("--json")
    return subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)


def test_update_graphs_status_empty_workspace(project):
    result = _run_status(project)
    assert result.returncode == 0, result.stderr
    # No RPG yet → guidance points the agent to /cmind.encode.
    assert "No RPG found" in result.stdout
    assert "/cmind.encode" in result.stdout


def test_update_graphs_status_with_rpg(project):
    data_dir = project / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "rpg.json").write_text(json.dumps({
        "repo_name": "demo",
        "edges": [{"src": "a", "dst": "b"}],
        "root": {
            "id": "root",
            "children": [
                {"id": "area1", "children": [{"id": "feat1", "children": []}]},
                {"id": "area2", "children": []},
            ],
        },
    }))
    (data_dir / "dep_graph.json").write_text(json.dumps({
        "nodes": [{"id": "n1"}, {"id": "n2"}],
        "edges": [{"src": "n1", "dst": "n2"}],
        "generated_at": "2026-01-01T00:00:00",
    }))

    text = _run_status(project).stdout
    assert "repo=demo" in text
    # 1 (root) + 2 (areas) + 1 (feat1) = 4 nodes
    assert "nodes=4" in text
    assert "edges=1" in text
    assert "rpg-tools MCP server" in text
    # MCP tool names from mcp_server.py should be in the guidance.
    for tool in ("search_rpg", "explore_rpg", "get_node_detail", "list_rpg_tree"):
        assert tool in text

    payload = json.loads(_run_status(project, json_mode=True).stdout)
    assert payload["mode"] == "status"
    assert payload["rpg_nodes"] == 4
    assert payload["dep_nodes"] == 2
    assert payload["repo_name"] == "demo"


def test_update_graphs_status_handles_corrupt_files(project):
    data_dir = project / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "rpg.json").write_text("{ this is not json")
    (data_dir / "dep_graph.json").write_text("also broken")

    payload = json.loads(_run_status(project, json_mode=True).stdout)
    # Even with broken files, status exits 0 and reports the error fields
    # so the AI agent gets a graceful "graph unavailable" message rather
    # than a hook crash on session start.
    assert payload["rpg_exists"] is True
    assert "rpg_error" in payload
    assert "dep_graph_error" in payload


def test_update_graphs_status_text_on_corrupt_rpg_says_unavailable(project):
    """A corrupt rpg.json must NOT produce 'Repository Program Graph is available' text — that would mislead the AI agent into calling rpg-tools MCP queries that would all fail."""
    data_dir = project / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "rpg.json").write_text("not json at all")

    text = _run_status(project).stdout
    assert "is available" not in text
    assert "could not be parsed" in text
    assert "/cmind.encode" in text


def test_update_graphs_status_diverged_branch_survives_non_ascii_guidance(project):
    """Regression: the diverged-branch guidance line contains "->" (U+2192).

    On Windows, a subprocess whose stdout is piped (exactly what happens
    here via `subprocess.run(..., capture_output=True)`, and what the real
    SessionStart hook / `cmind script` wrapper do too) makes CPython fall
    back to a legacy code page for stdio instead of UTF-8. Printing this
    guidance line used to raise UnicodeEncodeError and crash the whole
    script instead of completing — this test pins that it no longer does.
    """
    subprocess.run(["git", "init", "-q", "-b", "new-branch"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project, check=True)
    (project / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=project, check=True)

    data_dir = project / ".cmind" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "rpg.json").write_text(json.dumps({
        "repo_name": "demo",
        "edges": [],
        "root": {"id": "root", "children": []},
        "meta": {"git": {
            "head_commit": "0" * 40,
            "head_short": "0000000",
            "head_branch": "old-branch",
        }},
    }))

    result = _run_status(project)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "UnicodeEncodeError" not in result.stderr
    assert "branch changed: 'old-branch'" in result.stdout
    assert "'new-branch'" in result.stdout


# ---------------------------------------------------------------------------
# _setup_gitignore — unified .gitignore management
# ---------------------------------------------------------------------------

def test_setup_gitignore_greenfield_writes_full_template(tmp_path):
    """No .git/, no .gitignore → Python standard template + all CoderMind rules."""
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    content = (tmp_path / ".gitignore").read_text()
    # Python conventions (matches github/gitignore/Python.gitignore verbatim)
    assert "__pycache__/" in content
    assert ".venv" in content  # upstream uses ``.venv`` (no trailing slash)
    # Sections that only exist in the full GitHub template.
    assert "PyInstaller" in content
    assert "Jupyter Notebook" in content
    assert ".ipynb_checkpoints" in content
    # CoderMind common (runtime + machine-specific)
    assert ".cmind/*" in content
    assert "!.cmind/config.toml" in content
    assert ".vscode/mcp.json" in content
    assert ".vscode/tasks.json" in content
    assert ".mcp.json" in content
    # Copilot-specific
    assert ".github/agents/" in content
    assert ".github/prompts/" in content
    # Claude rules must NOT leak into copilot project
    assert ".claude/commands/" not in content


def test_setup_gitignore_greenfield_claude(tmp_path):
    """Claude path uses .claude/commands/ instead of .github/*."""
    cmind_cli._setup_gitignore(tmp_path, "claude")
    content = (tmp_path / ".gitignore").read_text()
    assert ".claude/commands/" in content
    # Copilot directories must NOT be ignored on a Claude project
    assert ".github/agents/" not in content
    assert ".github/prompts/" not in content


def test_setup_gitignore_existing_git_no_ignore_writes_cmind_only(tmp_path):
    """Existing .git/, no .gitignore → CoderMind rules only, NO Python template."""
    (tmp_path / ".git").mkdir()
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    content = (tmp_path / ".gitignore").read_text()
    # CoderMind rules present
    assert ".cmind/*" in content
    assert ".github/agents/" in content
    # Python conventions NOT imposed on existing repo
    assert "__pycache__/" not in content
    assert "PyInstaller" not in content
    assert ".ipynb_checkpoints" not in content


def test_setup_gitignore_existing_gitignore_preserves_user_entries(tmp_path):
    """Pre-existing .gitignore content must be preserved verbatim."""
    user_content = "# My custom rules\nnode_modules/\n*.tmp\n"
    (tmp_path / ".gitignore").write_text(user_content)
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    content = (tmp_path / ".gitignore").read_text()
    # User's entries preserved at the top, untouched
    assert content.startswith(user_content)
    assert "node_modules/" in content
    assert "*.tmp" in content
    # CoderMind rules appended
    assert ".cmind/*" in content
    assert ".github/agents/" in content


def test_setup_gitignore_is_idempotent(tmp_path):
    """Running _setup_gitignore twice must not duplicate entries or headers."""
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    first = (tmp_path / ".gitignore").read_text()
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    second = (tmp_path / ".gitignore").read_text()
    assert first == second  # second call is a no-op
    # No duplicate CoderMind header
    assert second.count(cmind_cli._GITIGNORE_CMIND_HEADER) == 1
    # No duplicate runtime-directory glob entry.
    lines = [l.strip() for l in second.splitlines()]
    assert lines.count(".cmind/*") == 1


def test_setup_gitignore_partial_existing_rules_only_appends_missing(tmp_path):
    """If user already has SOME CoderMind rules, only missing ones get appended."""
    # User has manually added the runtime-directory glob but nothing else.
    (tmp_path / ".gitignore").write_text(".cmind/*\n")
    cmind_cli._setup_gitignore(tmp_path, "copilot")
    content = (tmp_path / ".gitignore").read_text()
    # The runtime-directory glob must not be duplicated.
    lines = [l.strip() for l in content.splitlines()]
    assert lines.count(".cmind/*") == 1
    # The new managed config.toml un-ignore line is present
    assert "!.cmind/config.toml" in lines
    # Missing rules are now present
    assert ".vscode/mcp.json" in content
    assert ".github/agents/" in content


# ---------------------------------------------------------------------------
# MCP auto-approval (pre-authorization)
# ---------------------------------------------------------------------------

def test_install_claude_hooks_adds_mcp_rpg_tools_permission(project):
    """Cmind init should pre-authorize mcp__rpg-tools so Claude Code does not prompt for every search_rpg / explore_rpg / get_node_detail / list_rpg_tree call."""
    cmind_cli._install_claude_hooks(project)
    data = json.loads((project / ".claude" / "settings.json").read_text())
    assert "mcp__rpg-tools" in data["permissions"]["allow"]


def test_install_claude_hooks_preserves_existing_permissions(project):
    """User-configured permissions.allow entries must not be wiped, and the mcp rule must not duplicate on repeated init runs."""
    claude_dir = project / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps({
        "permissions": {
            "allow": ["Write", "Edit", "user-custom-rule"],
            "deny": ["WebSearch"],
        }
    }))

    cmind_cli._install_claude_hooks(project)
    data = json.loads((claude_dir / "settings.json").read_text())
    allow = data["permissions"]["allow"]
    # User entries preserved
    assert "Write" in allow
    assert "Edit" in allow
    assert "user-custom-rule" in allow
    # Deny list untouched
    assert data["permissions"]["deny"] == ["WebSearch"]
    # New rule appended
    assert "mcp__rpg-tools" in allow

    # Idempotent: second call must not re-append
    cmind_cli._install_claude_hooks(project)
    data2 = json.loads((claude_dir / "settings.json").read_text())
    assert data2["permissions"]["allow"].count("mcp__rpg-tools") == 1


def test_generate_mcp_config_copilot_omits_sandbox(tmp_path):
    """Copilot ``.vscode/mcp.json`` must NOT include sandbox keys.

    Earlier versions of CoderMind enabled the VS Code MCP sandbox to
    auto-approve tool confirmations, but the sandbox needs ``bwrap``
    and ``socat`` on PATH — missing on WSL, minimal Docker, and stock
    macOS — and missing deps cause the server to crash with a useless
    'Connection closed' error.  We now leave the keys out entirely
    and rely on VS Code's 'Always allow this server' setting for the
    UX win.
    """
    scripts_dir = tmp_path / ".cmind" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "mcp_server.py").write_text("# placeholder\n")

    cmind_cli._generate_mcp_config(tmp_path, "copilot")
    cfg = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
    server = cfg["servers"]["rpg-tools"]
    assert "sandboxEnabled" not in server
    assert "sandbox" not in server
    # The core launch keys must still be present.
    assert "command" in server
    assert "args" in server


def test_generate_mcp_config_claude_has_no_sandbox_field(tmp_path):
    """Claude uses .claude/settings.json permissions, not mcp.json sandbox.  The .mcp.json file should stay clean of Copilot-specific keys to avoid confusion."""
    scripts_dir = tmp_path / ".cmind" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "mcp_server.py").write_text("# placeholder\n")

    cmind_cli._generate_mcp_config(tmp_path, "claude")
    cfg = json.loads((tmp_path / ".mcp.json").read_text())
    server = cfg["mcpServers"]["rpg-tools"]
    assert "sandboxEnabled" not in server
    assert "sandbox" not in server
