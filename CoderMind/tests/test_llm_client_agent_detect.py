"""Regression test for detect_agent_type() Windows path/quote handling.

On Windows the configured AI CLI command is often a quoted, backslash-separated
path with an executable suffix (e.g. `"C:\\tools\\claude.cmd" --flag`) rather
than the bare command name `_CLI_TO_AGENT` is keyed on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from common.llm_client import detect_agent_type


def test_detect_agent_type_bare_command():
    assert detect_agent_type("claude --flag") == "claude"


def test_detect_agent_type_windows_quoted_path_with_suffix():
    assert detect_agent_type('"C:\\tools\\claude.cmd" --flag') == "claude"


def test_detect_agent_type_windows_exe_suffix_no_quotes():
    assert detect_agent_type("C:\\tools\\Gemini.EXE --flag") == "gemini"


def test_detect_agent_type_unknown():
    assert detect_agent_type("some-other-tool --flag") == "unknown"


def test_detect_agent_type_empty():
    assert detect_agent_type("") == "unknown"


if __name__ == "__main__":
    test_detect_agent_type_bare_command()
    test_detect_agent_type_windows_quoted_path_with_suffix()
    test_detect_agent_type_windows_exe_suffix_no_quotes()
    test_detect_agent_type_unknown()
    test_detect_agent_type_empty()
    print("ok")
