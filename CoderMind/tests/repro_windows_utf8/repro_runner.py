"""Reproduces the exact subprocess launch shape CoderMind uses to run every
bundled script, to confirm the Windows UnicodeEncodeError theory cheaply
(no LLM calls, no CoderMind imports, runs in well under a second):

    CoderMind/src/cmind_cli/__init__.py, function `script()`:
        env = os.environ.copy()
        ...
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

The same shape (piped/captured stdout, not a real console) is used by:
  * the SessionStart / post-commit hooks: `cmind script update_graphs.py status`
  * /cmind.encode: `cmind script rpg_encoder/run_encode.py --json`
  * `cmind init`'s optional immediate encode: `_run_initial_encode()`

Usage::

    python repro_runner.py         # before the fix: child crashes
    python repro_runner.py --fix   # after the fix:  child completes

Before the fix, this prints a UnicodeEncodeError traceback and exit code 1
-- the same traceback that was observed live from the real SessionStart
hook in this repo (see README.md in this directory for the full writeup).
"""
import os
import subprocess
import sys

child = os.path.join(os.path.dirname(__file__), "repro_child.py")

env = os.environ.copy()
env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

if "--fix" in sys.argv:
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("PYTHONUTF8", "1")
    print("=== running WITH the fix (PYTHONIOENCODING=utf-8:replace) ===")
else:
    print("=== running WITHOUT the fix ===")

proc = subprocess.run(
    [sys.executable, child],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
sys.stdout.buffer.write(proc.stdout)
print(f"\n--- child exit code: {proc.returncode} ---")
