#!/usr/bin/env python3
"""Tests for ``init_codebase.create_gitignore`` dev-env coverage.

A fixture- or hand-authored ``.gitignore`` may already carry ``.cmind/`` and a
Python cache block while predating the throwaway-venv rules.  The updater must
still append ``.venv_dev/`` so codegen scratch environments are never committed.
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

import init_codebase


def _read(path):
    return path.read_text(encoding="utf-8")


class TestCreateGitignoreDevEnv:
    def test_greenfield_includes_venv_dev(self, tmp_path):
        changed = init_codebase.create_gitignore(tmp_path)
        assert changed is True
        content = _read(tmp_path / ".gitignore")
        assert ".venv_dev/" in content
        assert ".cmind/" in content
        assert "__pycache__/" in content

    def test_cmind_and_python_present_but_missing_venv_dev_appends_dev_env(self, tmp_path):
        # Mirrors the fixture-shipped gitignore that fooled the old detection.
        gi = tmp_path / ".gitignore"
        gi.write_text(
            "build/\n*.o\n.cmind/\n__pycache__/\n*.py[cod]\n",
            encoding="utf-8",
        )
        changed = init_codebase.create_gitignore(tmp_path)
        assert changed is True
        content = _read(gi)
        assert ".venv_dev/" in content
        assert ".cmind_dev_env/" in content
        # Existing user entries are preserved.
        assert "build/" in content
        assert "*.o" in content
        # The full CoderMind block is not duplicated (only the dev-env subset).
        assert content.count(".cmind/") == 1

    def test_fully_configured_is_noop(self, tmp_path):
        gi = tmp_path / ".gitignore"
        gi.write_text(
            ".cmind/\n__pycache__/\n.venv_dev/\n.cmind_dev_env/\n",
            encoding="utf-8",
        )
        changed = init_codebase.create_gitignore(tmp_path)
        assert changed is False

    def test_dev_env_detection_accepts_unslashed_form(self, tmp_path):
        gi = tmp_path / ".gitignore"
        gi.write_text(
            ".cmind/\n__pycache__/\n.venv_dev\n",
            encoding="utf-8",
        )
        changed = init_codebase.create_gitignore(tmp_path)
        assert changed is False

    def test_idempotent_after_dev_env_append(self, tmp_path):
        gi = tmp_path / ".gitignore"
        gi.write_text(".cmind/\n__pycache__/\n", encoding="utf-8")
        assert init_codebase.create_gitignore(tmp_path) is True
        # Second run sees venv_dev now present → no further change.
        assert init_codebase.create_gitignore(tmp_path) is False
