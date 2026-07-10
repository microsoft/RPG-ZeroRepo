#!/usr/bin/env python3
"""Tests for CoderMind hook installation and status loading.

Verifies:
  - ``_install_claude_hooks`` writes a SessionStart hook that calls
    ``update_graphs.py status`` and merges with existing settings.
  - ``_install_copilot_hooks`` writes a VS Code task with
    ``runOptions.runOn = "folderOpen"``, is idempotent, and preserves
    pre-existing user tasks.
    - ``_install_hooks`` dispatches the right AI-specific installer,
        installs git post-commit/post-merge dispatchers, and removes
        CoderMind-owned pre-commit blocks when a ``.git`` dir exists.
  - ``update_graphs.py status`` returns RPG/dep-graph stats + an
    agent-facing MCP-tools reminder, on both populated and empty
    workspaces.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src/ and scripts/ are importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "scripts"))

import cmind_cli  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    """A minimal CoderMind workspace with .cmind/scripts/update_graphs.py."""
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

    cmind_cli._install_hooks(project, "copilot", tracker=None)

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

    cmind_cli._install_hooks(project, "claude", tracker=None)

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

    This is a static-source assertion rather than an end-to-end test
    because ``update`` does network I/O (template download) that is
    too heavyweight to mock for a single-bit regression check.  The
    intent is simply: if someone deletes the ``_install_hooks(...)``
    call from ``update``, this test fails loudly.
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
