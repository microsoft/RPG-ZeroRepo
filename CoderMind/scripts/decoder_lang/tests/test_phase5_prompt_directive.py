"""Tests for language-directive preambles in LLM prompts.

Critical regression invariant: when the target language is Python
the directive is the empty string and prompt text is unchanged.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make ``scripts/`` importable for direct invocation.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from decoder_lang import (  # noqa: E402
    get_backend,
    language_directive,
    with_language_directive,
)


class LanguageDirectiveTests(unittest.TestCase):
    """``language_directive`` produces the right preamble per language."""

    def test_python_directive_is_empty(self) -> None:
        # Critical: Python prompts must render byte-identically.
        self.assertEqual(language_directive(get_backend("python")), "")

    def test_none_backend_directive_is_empty(self) -> None:
        # Defensive: callers without a backend supply None and should
        # see no behavioural change.
        self.assertEqual(language_directive(None), "")

    def test_go_directive_mentions_go(self) -> None:
        d = language_directive(get_backend("go"))
        self.assertTrue(d)
        self.assertIn("Go", d)
        # Markdown fence reminder helps the LLM emit the right code block.
        self.assertIn("```go", d)
        # Extension reminder.
        self.assertIn(".go", d)
        # Test framework hint.
        self.assertIn("go test", d)

    def test_directive_ends_with_blank_line(self) -> None:
        # When a directive is emitted, it must end with a blank line
        # so the system prompt body after it is visually separated.
        d = language_directive(get_backend("go"))
        self.assertTrue(d.endswith("\n"))


class WithLanguageDirectiveTests(unittest.TestCase):
    """``with_language_directive`` prepends correctly + is no-op for Python."""

    def setUp(self) -> None:
        self.body = "You are a helpful assistant.\nFollow the rules."

    def test_python_returns_body_unchanged(self) -> None:
        result = with_language_directive(self.body, get_backend("python"))
        self.assertEqual(result, self.body)
        # ``is`` check confirms no allocation either when nothing to do.
        self.assertIs(result, self.body)

    def test_none_returns_body_unchanged(self) -> None:
        result = with_language_directive(self.body, None)
        self.assertEqual(result, self.body)

    def test_go_prepends_directive(self) -> None:
        result = with_language_directive(self.body, get_backend("go"))
        self.assertTrue(result.endswith(self.body))
        self.assertTrue(result.startswith("### Target language: Go"))
        # The original body is preserved verbatim at the tail.
        self.assertIn(self.body, result)

    def test_empty_body_handled(self) -> None:
        # Edge: empty body + Go directive → just the directive.
        result = with_language_directive("", get_backend("go"))
        self.assertTrue(result.startswith("### Target language: Go"))


if __name__ == "__main__":
    unittest.main()
