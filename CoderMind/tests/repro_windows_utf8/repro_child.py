"""Minimal stand-in for a bundled CoderMind script.

Mirrors the exact pattern that used to exist in
CoderMind/scripts/update_graphs.py's _format_status_for_agent(): a plain
``print()`` of a string containing a non-ASCII character. That function
prints this unconditionally whenever an RPG exists and is readable (the
"->" in the "branch changed: 'a' -> 'b'" note, and the "->" in the
"functional areas -> groups -> features" guidance line), so it isn't an
exotic edge case -- it's on the common "status looks fine" path.

No LLM call, no network, no CoderMind imports: this reproduces the crash
class with two lines of stdlib print(), on purpose, so it can be run in
under a second to confirm/deny the theory without spending any tokens
on an actual /cmind.encode run.
"""
print("[CoderMind] branch changed: 'old-branch' → 'new-branch'.")
print("status: ok")
