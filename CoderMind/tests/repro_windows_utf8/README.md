# Windows `UnicodeEncodeError` crash — repro, evidence, and what's still a guess

## The bug

On Windows, when a Python script's stdout/stderr is **not a real console**
(it's piped or captured by a parent process — which is exactly what
`cmind script ...`, the SessionStart/post-commit hooks, and
`cmind init`'s optional immediate-encode step all do), CPython falls back
to `locale.getpreferredencoding(False)` for stdio instead of UTF-8. That's
a legacy code page (`cp1252` on most Western-locale Windows installs,
`cp936` on Chinese installs, ...) rather than UTF-8. A bare `print()` of
any character outside that code page then raises `UnicodeEncodeError`
and kills the whole script instead of completing.

## Reproduce it (cheap — no LLM calls, < 1 second)

```
python repro_runner.py         # before the fix: child crashes, exit code 1
python repro_runner.py --fix   # after the fix:  child completes, exit code 0
```

`repro_runner.py` spawns `repro_child.py` (a two-line script that prints
a line containing "→", U+2192) using the same
`subprocess.run(cmd, env=env, stdout=PIPE, stderr=STDOUT)` shape that
`cmind_cli/__init__.py`'s `script()` command uses for every bundled
script. `--fix` adds `PYTHONIOENCODING=utf-8:replace` /
`PYTHONUTF8=1` to the child's env — the same fix now applied at
`CoderMind/scripts/common/paths.py` import time (see the main fix, not
in this directory).

## What's confirmed vs. what's a hypothesis

**Confirmed — this crash is real and already happened on this machine,
in this repo:**

- The SessionStart hook (`cmind script update_graphs.py status`) crashed
  with this exact `UnicodeEncodeError` (character `→`) at the start
  of the session where this fix was written — visible as the hook's
  fallback text, `[CoderMind] RPG status unavailable`.
- `update_graphs.py`'s `_format_status_for_agent()` prints a line
  containing "→" **unconditionally** whenever an RPG exists and is
  readable (not just on the rarer "branch changed" path) — so this
  isn't an edge case, it fires on the ordinary "status looks fine" path.
- Proof: 3 pre-existing tests were silently failing before this fix,
  for exactly this reason (their captured `stdout` was empty because the
  child crashed before finishing) —
  `test_hooks_install.py::test_update_graphs_status_with_rpg`,
  `test_step3_polish.py::test_status_text_omits_branch_when_detached`,
  `test_step3_polish.py::test_status_text_shows_branch_when_in_sync`.
  All three pass after the fix, untouched otherwise.

**Not confirmed — a reasoned hypothesis, not a verified fact:**

- Whether this exact bug is what broke `/cmind.encode` for the reviewer
  who reported "input errors" on Windows. The specific `"→"` print above
  lives in `update_graphs.py` (used by hooks / `/cmind.update_rpg`), which
  is **not** on `/cmind.encode`'s own call path (`run_encode.py` /
  `check_encode.py`). Their own top-level `print(json.dumps(...))` calls
  are ASCII-safe by default (`json.dumps` escapes non-ASCII to `\uXXXX`).
- For `/cmind.encode` to hit the *same class* of crash, some other
  `print()`/`logger.warning()` deeper in the pipeline (`workflow.py`,
  `rpg_encoding.py`, `semantic_parsing.py`, or a log of the raw LLM
  response) would have to emit a non-ASCII character while stdout is
  piped. That's plausible but content-dependent — it doesn't require
  the *repo being encoded* to contain anything unusual. LLMs routinely
  emit "invisible" non-ASCII characters even for plain English content:
  curly quotes (`" "`), em/en dashes (`—`/`–`), arrows (`→`), bullets
  (`•`), ellipses (`…`). Whether a given `/cmind.encode` run trips the
  bug is effectively a coin flip driven by what the LLM happened to
  generate that call, which is consistent with one person's run
  succeeding and another's failing on the same code.
- Because the actual fix (`common/paths.py`, imported by every bundled
  script) isn't tied to any single print site, it protects the whole
  pipeline regardless of which exact line would have crashed — so
  pinpointing the precise trigger for the reviewer's report isn't
  required for the fix to be correct, only for fully explaining their
  report.

## Is "force everything to UTF-8" safe for text that's *supposed* to
   contain special characters?

Yes — this is not a lossy workaround. `encoding="utf-8"` is a universal
encoding: every valid Unicode code point (Vietnamese, Chinese, arrows,
emoji, anything) round-trips through it losslessly. The legacy Windows
code pages (`cp1252`, `cp936`, ...) are the ones that *can't* represent
arbitrary Unicode — UTF-8 is what actually lets special/non-English
content print correctly instead of crashing.

`errors="replace"` only matters for the (rare, genuinely abnormal) case
of a byte sequence that isn't valid UTF-8 at all — e.g. a subprocess
that itself used some other encoding internally. In that case one
unrecognizable byte becomes the replacement character `�` instead of
raising and killing the whole script. That's a deliberate trade-off:
losing the glyph for one malformed byte is preferable to the entire
`/cmind.encode` run (or hook) aborting outright.
