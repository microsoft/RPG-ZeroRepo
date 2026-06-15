#!/usr/bin/env python3
"""Tests for the fallback delimiter/syntax scanner.

The scanner in ``lang_parser.extractors.fallback`` is only exercised when a
tree-sitter backend is unavailable, so these cases call it directly to lock in
its handling of block comments, char literals, and Rust lifetimes — historical
sources of spurious "Unterminated string literal" errors.
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_project_root, "scripts"))

from lang_parser.extractors.fallback import delimiter_syntax_error


class TestDelimiterSyntaxError:
    def test_apostrophe_in_block_comment_is_not_an_error(self):
        src = "/* the store's document — doesn't crash */\nint main(void){return 0;}\n"
        assert delimiter_syntax_error(src) is None

    def test_apostrophe_in_line_comment_is_not_an_error(self):
        src = "// user's request — see spec\nint x = 1;\n"
        assert delimiter_syntax_error(src) is None

    def test_valid_char_literals_are_not_errors(self):
        src = "char c = 'a';\nchar n = '\\n';\nchar q = '\\'';\n"
        assert delimiter_syntax_error(src) is None

    def test_rust_lifetimes_are_not_unterminated_strings(self):
        src = "fn foo<'a>(x: &'a str) -> &'a str { x }\n"
        assert delimiter_syntax_error(src) is None

    def test_rust_static_lifetime_is_not_an_error(self):
        src = 'static S: &\'static str = "hi";\n'
        assert delimiter_syntax_error(src) is None

    def test_multiline_block_comment_with_quotes_is_not_an_error(self):
        src = "/*\n * The store's codec — handles \"json\" persistence\n */\nint x = 1;\n"
        assert delimiter_syntax_error(src) is None

    def test_unterminated_block_comment_is_flagged(self):
        src = "/* never closed\nint x = 1;\n"
        assert delimiter_syntax_error(src) == "Unterminated block comment"

    def test_brace_imbalance_is_flagged(self):
        src = "int main(void){ return 0;\n"
        assert delimiter_syntax_error(src) is not None

    def test_unmatched_closing_delimiter_is_flagged(self):
        src = "int main(void){ return 0; }}\n"
        assert delimiter_syntax_error(src) is not None

    def test_balanced_source_is_clean(self):
        src = "int add(int a, int b){ return a + b; }\n"
        assert delimiter_syntax_error(src) is None
