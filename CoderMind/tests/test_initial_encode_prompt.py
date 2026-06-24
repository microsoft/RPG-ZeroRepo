#!/usr/bin/env python3
"""Tests for the optional initial-encode prompt at the end of ``cmind init``.

Covers:
  * ``_workspace_has_python_code`` correctly ignores ``.cmind/`` (the
    runtime-script tree we just extracted) and other boilerplate dirs.
  * ``_maybe_offer_initial_encode`` skips silently when:
      - rpg.json already exists,
      - the user passed ``--no-encode`` (``encode_choice=False``),
      - stdin is not a TTY (CI / piped invocation),
      - the workspace has no user Python files.
  * ``--encode`` (``encode_choice=True``) bypasses both the TTY and the
    "no python code" checks (the user has explicitly asked).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import cmind_cli  # noqa: E402


# ---------------------------------------------------------------------------
# _workspace_has_python_code
# ---------------------------------------------------------------------------

def test_workspace_has_python_code_finds_user_file(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    assert cmind_cli._workspace_has_python_code(tmp_path) is True


def test_workspace_has_python_code_finds_nested_file(tmp_path):
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("\n")
    assert cmind_cli._workspace_has_python_code(tmp_path) is True


def test_workspace_has_python_code_ignores_cmind_runtime(tmp_path):
    """Every workspace gets ``.cmind/scripts/*.py`` after init.

    Without the prune, this would always return True and make the
    prompt fire even for empty workspaces.
    """
    cmind_scripts = tmp_path / ".cmind" / "scripts"
    cmind_scripts.mkdir(parents=True)
    (cmind_scripts / "mcp_server.py").write_text("\n")
    (cmind_scripts / "update_graphs.py").write_text("\n")
    assert cmind_cli._workspace_has_python_code(tmp_path) is False


def test_workspace_has_python_code_ignores_common_junk_dirs(tmp_path):
    for junk in (".git", ".venv", "__pycache__", "node_modules", "build"):
        sub = tmp_path / junk
        sub.mkdir()
        (sub / "noise.py").write_text("\n")
    assert cmind_cli._workspace_has_python_code(tmp_path) is False


def test_workspace_has_python_code_empty(tmp_path):
    assert cmind_cli._workspace_has_python_code(tmp_path) is False


# ---------------------------------------------------------------------------
# _maybe_offer_initial_encode — short-circuits
# ---------------------------------------------------------------------------

def test_skip_when_rpg_already_exists(tmp_path):
    """If rpg.json is already present, never prompt nor run."""
    (tmp_path / "main.py").write_text("\n")
    rpg_file = tmp_path / ".cmind" / "data" / "rpg.json"
    rpg_file.parent.mkdir(parents=True)
    rpg_file.write_text("{}")

    with patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm") as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=True)
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=False)

    assert run.call_count == 0
    assert confirm.call_count == 0


def test_skip_when_no_encode_flag(tmp_path):
    """``--no-encode`` must skip even when there's Python code."""
    (tmp_path / "main.py").write_text("\n")

    with patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm") as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=False)

    run.assert_not_called()
    confirm.assert_not_called()


def test_skip_when_no_python_code_and_interactive(tmp_path):
    """Empty workspace + interactive → no prompt, no run."""
    with patch("sys.stdin.isatty", return_value=True), \
         patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm") as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)

    run.assert_not_called()
    confirm.assert_not_called()


def test_skip_when_not_a_tty(tmp_path):
    """Non-tty (CI / piped) skips the prompt entirely."""
    (tmp_path / "main.py").write_text("\n")
    with patch("sys.stdin.isatty", return_value=False), \
         patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm") as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)

    run.assert_not_called()
    confirm.assert_not_called()


# ---------------------------------------------------------------------------
# _maybe_offer_initial_encode — happy paths
# ---------------------------------------------------------------------------

def test_explicit_encode_flag_runs_without_prompt(tmp_path):
    """``--encode`` bypasses the prompt even for an empty workspace."""
    with patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm") as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=True)

    run.assert_called_once_with(tmp_path)
    confirm.assert_not_called()


def test_interactive_yes_runs_encoder(tmp_path):
    (tmp_path / "main.py").write_text("\n")
    with patch("sys.stdin.isatty", return_value=True), \
         patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm", return_value=True) as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)

    confirm.assert_called_once()
    run.assert_called_once_with(tmp_path)


def test_interactive_no_skips_encoder(tmp_path):
    (tmp_path / "main.py").write_text("\n")
    with patch("sys.stdin.isatty", return_value=True), \
         patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm", return_value=False) as confirm:
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)

    confirm.assert_called_once()
    run.assert_not_called()


def test_keyboard_interrupt_during_prompt_does_not_propagate(tmp_path):
    """Ctrl-C at the y/N prompt must not crash init."""
    (tmp_path / "main.py").write_text("\n")
    with patch("sys.stdin.isatty", return_value=True), \
         patch.object(cmind_cli, "_run_initial_encode") as run, \
         patch("typer.confirm", side_effect=KeyboardInterrupt):
        cmind_cli._maybe_offer_initial_encode(tmp_path, encode_choice=None)

    run.assert_not_called()


# ---------------------------------------------------------------------------
# _run_initial_encode — missing encoder script
# ---------------------------------------------------------------------------

def test_run_initial_encode_missing_script_returns_false(tmp_path):
    """If .cmind/scripts/rpg_encoder/run_encode.py is absent, we warn
    and return False without raising."""
    assert cmind_cli._run_initial_encode(tmp_path) is False


# ---------------------------------------------------------------------------
# _parse_encoder_line — phase markers drive the progress UI
# ---------------------------------------------------------------------------

def _fresh_state():
    return {
        "phase": "Starting encoder…",
        "kind": None,
        "class_total": 0,
        "class_done": 0,
        "func_total": 0,
        "func_done": 0,
        "total_files": 0,
    }


def test_parse_line_generating_repo_info():
    s = _fresh_state()
    cmind_cli._parse_encoder_line("RPGParser - INFO - Generating repo info (max_iters=3)", s)
    assert "Generating repository overview" in s["phase"]


def test_parse_line_repo_info_iter():
    s = _fresh_state()
    cmind_cli._parse_encoder_line("RPGParser - INFO - LLM call for repo info, iter=2...", s)
    assert "iter 2" in s["phase"]


def test_parse_line_exclude_vote():
    s = _fresh_state()
    cmind_cli._parse_encoder_line("RPGParser - INFO - LLM vote #3 for exclude list...", s)
    assert "vote #3" in s["phase"]


def test_parse_line_excluding_irrelevant_files():
    """Matches the encoder's actual ``Excluding irrelevant files (max_votes=...)``
    log line — not a fabricated marker."""
    s = _fresh_state()
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - Excluding irrelevant files (max_votes=1)...", s)
    assert s["phase"] == "Selecting files to exclude"


def test_parse_line_total_files():
    s = _fresh_state()
    cmind_cli._parse_encoder_line("RPGParser - INFO - Total valid source files to parse: 42", s)
    assert s["total_files"] == 42
    assert "42 files" in s["phase"]


def test_parse_line_class_batches_and_progress():
    s = _fresh_state()
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - [GLOBAL] kind=class, groups=5, batches=7, foo=bar", s)
    assert s["kind"] == "class"
    assert s["class_total"] == 7
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - [GLOBAL] process_class_batch: classes=['A'], units=3", s)
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - [GLOBAL] process_class_batch: classes=['B'], units=2", s)
    assert s["class_done"] == 2


def test_parse_line_function_batches_and_progress():
    s = _fresh_state()
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - [GLOBAL] kind=function, groups=4, batches=6, foo=bar", s)
    assert s["kind"] == "function"
    assert s["func_total"] == 6
    cmind_cli._parse_encoder_line(
        "RPGParser - INFO - [GLOBAL] process_func_batch: functions=['f'], units=1", s)
    assert s["func_done"] == 1


def test_parse_line_refactoring_clears_kind():
    s = _fresh_state()
    s["kind"] = "function"
    cmind_cli._parse_encoder_line("RPGParser - INFO - Refactoring to RPG...", s)
    assert s["kind"] is None
    assert "Refactoring" in s["phase"]


def test_parse_line_unknown_is_ignored():
    """Unrecognised lines must leave state untouched (best-effort parser)."""
    s = _fresh_state()
    cmind_cli._parse_encoder_line("some completely unrelated line", s)
    assert s == _fresh_state()


# ---------------------------------------------------------------------------
# _run_initial_encode — end-to-end with a mocked subprocess
# ---------------------------------------------------------------------------

def _make_fake_encoder(tmp_path: Path, exit_code: int, stderr_lines: list, stdout_text: str = "") -> Path:
    """Write a real Python script that mimics the encoder's IO.

    Using a real subprocess (rather than mocking Popen) keeps the test
    honest: it exercises the actual threaded reader + Progress loop.
    """
    encoder_dir = tmp_path / ".cmind" / "scripts" / "rpg_encoder"
    encoder_dir.mkdir(parents=True)
    script = encoder_dir / "run_encode.py"
    payload = {
        "exit_code": exit_code,
        "stderr_lines": stderr_lines,
        "stdout_text": stdout_text,
    }
    import json as _json
    script.write_text(
        "import json, sys, time\n"
        f"payload = {_json.dumps(payload)}\n"
        "for line in payload['stderr_lines']:\n"
        "    sys.stderr.write(line + '\\n')\n"
        "    sys.stderr.flush()\n"
        "if payload['stdout_text']:\n"
        "    sys.stdout.write(payload['stdout_text'])\n"
        "    sys.stdout.flush()\n"
        "sys.exit(payload['exit_code'])\n"
    )
    return script


def test_run_initial_encode_success_writes_log(tmp_path):
    """A 0-exit encoder is reported as success and its stderr lands in encode.log."""
    _make_fake_encoder(
        tmp_path,
        exit_code=0,
        stderr_lines=[
            "RPGParser - INFO - Generating repo info (max_iters=3)",
            "RPGParser - INFO - LLM call for repo info, iter=1",
            "RPGParser - INFO - Total valid source files to parse: 5",
            "RPGParser - INFO - [GLOBAL] kind=class, groups=1, batches=2, foo=bar",
            "RPGParser - INFO - [GLOBAL] process_class_batch: classes=['A'], units=1",
            "RPGParser - INFO - [GLOBAL] process_class_batch: classes=['B'], units=1",
            "RPGParser - INFO - Refactoring to RPG...",
            "RPGParser - INFO - RPG refactoring done.",
        ],
        stdout_text='{"status": "success"}\n',
    )
    assert cmind_cli._run_initial_encode(tmp_path) is True
    log = cmind_cli._storage.workspace_logs_dir(tmp_path) / "encode.log"
    assert log.is_file()
    contents = log.read_text()
    assert "Generating repo info" in contents
    assert "process_class_batch" in contents


def test_run_initial_encode_failure_returns_false(tmp_path):
    """A non-zero exit is reported as failure and we still get a log file."""
    _make_fake_encoder(
        tmp_path,
        exit_code=1,
        stderr_lines=["RPGParser - ERROR - boom"],
        stdout_text='{"status": "failed", "error": "boom"}\n',
    )
    assert cmind_cli._run_initial_encode(tmp_path) is False
    log = cmind_cli._storage.workspace_logs_dir(tmp_path) / "encode.log"
    assert log.is_file()
    assert "boom" in log.read_text()
