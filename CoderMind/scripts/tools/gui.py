#!/usr/bin/env python3
"""GUI Tool — Desktop application screenshot, interaction, and process management.

Provides CLI commands for verifying and interacting with GUI applications
(tkinter, PyQt, pygame, etc.) using Xvfb virtual display + xdotool.

Display management:
    python tools/gui.py start-display [--display :99] [--size 1280x720]
    python tools/gui.py stop-display [--display :99]

Application management:
    python tools/gui.py launch <command> [--display :99] [--wait SECONDS]
    python tools/gui.py status
    python tools/gui.py close [--pid PID]

Screenshot:
    python tools/gui.py screenshot [--display :99] [--output FILE]

Interaction (xdotool):
    python tools/gui.py click <x> <y> [--display :99]
    python tools/gui.py type <text> [--display :99]
    python tools/gui.py key <keys> [--display :99]
    python tools/gui.py scroll <amount> [--display :99]

Interactive command:
    python tools/gui.py run-script --display :99 --script 'gui.click(100, 200)'
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = ".cmind/tmp/screenshots"
DEFAULT_DISPLAY = ":99"
DEFAULT_SCREEN_SIZE = "1280x720x24"
SCRIPT_TIMEOUT = 60
LAUNCH_WAIT = 3          # seconds to wait after launching app
_PID_FILE = ".cmind/tmp/gui_app.pid"   # persist app PID across CLI calls

# Track managed processes for cleanup (in-process only; PID file for cross-process)
_managed_pids: dict = {}  # label -> pid


def _save_app_pid(pid: int) -> None:
    """Persist app PID to file so close can find it across CLI invocations."""
    Path(_PID_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(_PID_FILE).write_text(str(pid))


def _load_app_pid() -> Optional[int]:
    """Load persisted app PID, or None if not found / stale."""
    try:
        pid = int(Path(_PID_FILE).read_text().strip())
        os.kill(pid, 0)  # check if process is alive
        return pid
    except (FileNotFoundError, ValueError, OSError):
        return None


def _clear_app_pid() -> None:
    """Remove the PID file."""
    try:
        Path(_PID_FILE).unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_toplevel_windows(display: str) -> list:
    """Return (wid, title) pairs for top-level windows on the display.

    Toolkits like tkinter create dozens of X11 sub-windows for widgets.
    This filters to only windows with a non-empty title, which are
    typically the top-level application windows.

    Uses a single xdotool call + batch xdotool getwindowname to avoid
    O(N) subprocess overhead for each of the ~36 widget windows.
    """
    env = {"DISPLAY": display, "PATH": os.environ.get("PATH", "")}
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ""],
            capture_output=True, text=True, timeout=3, env=env,
        )
        all_wids = [w for w in result.stdout.strip().splitlines() if w]
    except Exception:
        return []

    if not all_wids:
        return []

    # Batch: get window names for all wids in one xprop call per window
    # is still N calls, but we can use xdotool getwindowname with multiple
    # wids (it doesn't support batch), so we use a single shell pipeline
    # to query all at once via xprop.
    toplevel = []
    try:
        # Use xprop in a batch: query WM_NAME for each window ID
        # This is faster than N separate subprocess calls
        wid_list = " ".join(all_wids[:100])  # cap at 100 to avoid arg overflow
        script = f'for wid in {wid_list}; do echo "$wid $(xdotool getwindowname $wid 2>/dev/null)"; done'
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True, text=True, timeout=5, env=env,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                wid, name = parts[0], parts[1].strip()
                if name:  # non-empty title → top-level window
                    toplevel.append((wid, name))
    except Exception:
        # Fallback: just report that windows exist but can't enumerate
        pass

    # Deduplicate by title (some toolkits give the same title to
    # multiple internal windows)
    seen_titles = set()
    unique = []
    for wid, name in toplevel:
        if name not in seen_titles:
            seen_titles.add(name)
            unique.append((wid, name))

    return unique


# ---------------------------------------------------------------------------
# Display management
# ---------------------------------------------------------------------------

def cmd_start_display(display: str = DEFAULT_DISPLAY,
                      size: str = DEFAULT_SCREEN_SIZE):
    """Start a Xvfb virtual display."""
    # Check if already running (use word-boundary to avoid :99 matching :990)
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"Xvfb {display}( |$)"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().splitlines()[0]
            print(f"Xvfb already running on {display} (pid {pid})")
            return
    except Exception:
        pass

    # Clean stale lock file that may prevent Xvfb from starting
    display_num = display.lstrip(":")
    lock_file = f"/tmp/.X{display_num}-lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", size, "-ac"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    if proc.poll() is not None:
        print(f"Error: Xvfb failed to start on {display}", file=sys.stderr)
        sys.exit(1)

    _managed_pids["xvfb"] = proc.pid
    print(f"Xvfb started on {display} (pid {proc.pid}, size {size})")
    print(f"  Use: export DISPLAY={display}")


def cmd_stop_display(display: str = DEFAULT_DISPLAY):
    """Stop the Xvfb virtual display."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"Xvfb {display}( |$)"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            for pid_str in result.stdout.strip().splitlines():
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGTERM)
                print(f"Stopped Xvfb pid {pid}")
        else:
            print(f"No Xvfb running on {display}")
    except Exception as e:
        print(f"Error stopping Xvfb: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Application management
# ---------------------------------------------------------------------------

def cmd_launch(command: str, display: str = DEFAULT_DISPLAY,
               wait: float = LAUNCH_WAIT):
    """Launch a GUI application on the virtual display."""
    # Close the tracked app process before launching a replacement.
    prev_pid = _load_app_pid()
    if prev_pid is not None:
        print(f"  Closing previous app (pid {prev_pid}) before re-launch")
        _kill_pid(prev_pid)
        _clear_app_pid()
        time.sleep(0.5)

    env = os.environ.copy()
    env["DISPLAY"] = display

    # Use temp files for early-exit diagnostics, then detach so the app
    # doesn't deadlock on full pipes during long runs.
    import tempfile
    stdout_f = tempfile.TemporaryFile()
    stderr_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        command, shell=True, env=env,
        stdout=stdout_f, stderr=stderr_f,
        start_new_session=True,  # create process group so we can kill all children
    )
    time.sleep(wait)

    if proc.poll() is not None:
        stdout_f.seek(0)
        stderr_f.seek(0)
        stdout = stdout_f.read().decode(errors="replace")[:500]
        stderr = stderr_f.read().decode(errors="replace")[:500]
        stdout_f.close()
        stderr_f.close()
        print(f"Error: application exited immediately (exit code {proc.returncode})",
              file=sys.stderr)
        if stdout.strip():
            print(f"  stdout: {stdout}", file=sys.stderr)
        if stderr.strip():
            print(f"  stderr: {stderr}", file=sys.stderr)
        print("  [HINT] Check if the command is correct and dependencies are installed.",
              file=sys.stderr)
        sys.exit(1)

    # App is still running — close temp files (they detach from the FDs
    # the child inherited, so the child continues writing to /dev/null-like FDs)
    stdout_f.close()
    stderr_f.close()

    _managed_pids["app"] = proc.pid
    _save_app_pid(proc.pid)
    print(f"Application launched (pid {proc.pid})")
    print(f"  Command: {command}")
    print(f"  Display: {display}")

    # Wait for a top-level window to appear (up to 10 seconds, polling every 0.5s)
    window_found = False
    for _ in range(20):  # 20 * 0.5s = 10s max
        toplevel = _get_toplevel_windows(display)
        if toplevel:
            window_found = True
            print(f"  Top-level windows: {len(toplevel)}")
            for wid, name in toplevel[:3]:
                print(f"    - {name[:60]}")
            break
        # Check if process died while waiting
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    if not window_found:
        process_alive = proc.poll() is None
        print(f"  [WARNING] No visible window detected after launch!", file=sys.stderr)
        if process_alive:
            print(f"  The process (pid {proc.pid}) is running but did not create a GUI window.",
                  file=sys.stderr)
            print(f"  This likely means the application only prints to console without",
                  file=sys.stderr)
            print(f"  opening a real GUI. Screenshots will show a black screen.",
                  file=sys.stderr)
        else:
            print(f"  The process (pid {proc.pid}) exited with code {proc.returncode}.",
                  file=sys.stderr)
            print(f"  The application crashed or finished before creating a window.",
                  file=sys.stderr)
        print(f"  [HINT] The GUI code may need to be fixed to actually create a window",
              file=sys.stderr)
        print(f"  (e.g., tkinter.Tk(), QApplication, pygame.display.set_mode, etc.)",
              file=sys.stderr)


def cmd_status(display: str = DEFAULT_DISPLAY):
    """Show status of Xvfb and GUI applications."""
    import psutil

    print(f"=== Display {display} ===")

    # Xvfb status
    xvfb_running = False
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            # Match "Xvfb :99" exactly (not :990)
            if len(cmdline) >= 2 and "Xvfb" in cmdline[0] and cmdline[1] == display:
                print(f"  Xvfb: running (pid {proc.pid})")
                xvfb_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not xvfb_running:
        print(f"  Xvfb: not running")

    # Top-level windows on display (filtered from widget sub-windows)
    toplevel = _get_toplevel_windows(display)
    print(f"  Top-level windows: {len(toplevel)}")
    for wid, name in toplevel[:5]:
        print(f"    - Window {wid}: {name[:60]}")
    if not toplevel:
        # Show raw count for debugging
        try:
            env = {"DISPLAY": display, "PATH": os.environ.get("PATH", "")}
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", ""],
                capture_output=True, text=True, timeout=3, env=env,
            )
            raw = [w for w in result.stdout.strip().splitlines() if w]
            if raw:
                print(f"  (raw X11 windows: {len(raw)} — all unnamed/sub-windows)")
        except Exception:
            pass


def cmd_close(pid: Optional[int] = None, display: str = DEFAULT_DISPLAY):
    """Close a GUI application and kill its process."""
    if pid is not None:
        _kill_pid(pid)
    else:
        # 1. Close all visible windows on display via xdotool
        closed = 0
        try:
            env = {"DISPLAY": display, "PATH": os.environ.get("PATH", "")}
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", ""],
                capture_output=True, text=True, timeout=3, env=env,
            )
            windows = [w for w in result.stdout.strip().splitlines() if w]
            for wid in windows:
                subprocess.run(
                    ["xdotool", "windowclose", wid],
                    timeout=3, env=env,
                    capture_output=True,
                )
            closed = len(windows)
        except Exception as e:
            print(f"Error closing windows: {e}", file=sys.stderr)

        # 2. Kill the managed app process (windowclose may not terminate it)
        #    Check in-process dict first, then persisted PID file
        app_pid = _managed_pids.pop("app", None) or _load_app_pid()
        if app_pid:
            _kill_pid(app_pid)
            _clear_app_pid()
            print(f"Closed {closed} window(s) and killed app process {app_pid}")
        else:
            print(f"Closed {closed} window(s) on {display}")


def _kill_pid(pid: int) -> None:
    """Send SIGTERM to a process group, then SIGKILL if it doesn't exit.

    Uses negative PID to kill the entire process group (shell + children)
    when the app was launched with start_new_session=True.
    """
    try:
        # Try killing the process group first (negative PID)
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, PermissionError):
            # Fallback: kill just the process (it may not be a group leader)
            os.kill(pid, signal.SIGTERM)
        # Wait up to 2 seconds for graceful exit
        for _ in range(4):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # check if still alive
            except OSError:
                print(f"Process {pid} terminated")
                return
        # Still alive — force kill
        try:
            os.killpg(pid, signal.SIGKILL)
        except (OSError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        print(f"Force-killed process {pid}")
    except OSError:
        print(f"Process {pid} already terminated")


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def _auto_filename(label: str = "", ext: str = "png") -> str:
    """Generate timestamped screenshot filename."""
    timestamp = time.strftime("%H%M%S")
    parts = [timestamp]
    if label:
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "", label)[:30]
        parts.append(clean)
    else:
        parts.append("gui")
    name = "_".join(parts) + f".{ext}"
    out = f"{DEFAULT_OUTPUT_DIR}/{name}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return out


def cmd_screenshot(display: str = DEFAULT_DISPLAY,
                   output: Optional[str] = None,
                   window: Optional[str] = None):
    """Take a screenshot of the virtual display or a specific window."""
    if not output:
        output = _auto_filename("gui")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DISPLAY"] = display

    if window:
        # Screenshot specific window by name
        try:
            wid_result = subprocess.run(
                ["xdotool", "search", "--name", window],
                capture_output=True, text=True, timeout=3, env=env,
            )
            wids = wid_result.stdout.strip().splitlines()
            if not wids:
                raise RuntimeError(f"no window matching '{window}' found")
            wid = wids[0]
            subprocess.run(
                ["import", "-window", wid, output],
                timeout=5, env=env, check=True,
                capture_output=True,
            )
        except Exception as e:
            print(f"Error: could not capture window '{window}': {e}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        # Full screen capture
        try:
            subprocess.run(
                ["import", "-window", "root", output],
                timeout=5, env=env, check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: screenshot failed: {e}", file=sys.stderr)
            print(f"  [HINT] Is Xvfb running on {display}? Run: gui.py start-display",
                  file=sys.stderr)
            sys.exit(1)

    size = Path(output).stat().st_size
    print(f"Screenshot saved: {output} ({size} bytes)")
    print(f"  Display: {display}")

    # Warn if screenshot is suspiciously small (likely blank/black)
    if size < 1000:
        print(f"  [WARNING] Screenshot is only {size} bytes — likely a blank/black image!",
              file=sys.stderr)
        print(f"  This usually means no GUI window is visible on the display.",
              file=sys.stderr)
        print(f"  Run 'gui.py status' to check if a window exists.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Interaction (xdotool-based — works with abstract sockets on WSL2)
# ---------------------------------------------------------------------------

def _xdotool(display: str, args: list[str], check: bool = True) -> str:
    """Run an xdotool command on the given display."""
    env = {"DISPLAY": display, "PATH": os.environ.get("PATH", "")}
    result = subprocess.run(
        ["xdotool"] + args,
        capture_output=True, text=True, timeout=10, env=env,
    )
    if check and result.returncode != 0:
        msg = result.stderr.strip() or f"xdotool {args[0]} failed"
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def cmd_click(x: int, y: int, display: str = DEFAULT_DISPLAY,
              button: str = "left", clicks: int = 1):
    """Click at coordinates."""
    button_map = {"left": "1", "middle": "2", "right": "3"}
    btn = button_map.get(button, "1")
    _xdotool(display, ["mousemove", str(x), str(y)])
    time.sleep(0.05)
    for _ in range(clicks):
        _xdotool(display, ["click", btn])
    print(f"Clicked ({x}, {y}) button={button} clicks={clicks}")


def cmd_type_text(text: str, display: str = DEFAULT_DISPLAY,
                  interval: float = 0.05):
    """Type text."""
    delay_ms = max(1, int(interval * 1000))
    _xdotool(display, ["type", "--delay", str(delay_ms), "--", text])
    print(f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")


def cmd_key(keys: str, display: str = DEFAULT_DISPLAY):
    """Press key(s). Supports combos like 'ctrl+s', 'Tab', 'Return'."""
    _xdotool(display, ["key", "--", keys])
    print(f"Key pressed: {keys}")


def cmd_scroll(amount: int, display: str = DEFAULT_DISPLAY):
    """Scroll. Positive=up, negative=down."""
    if amount >= 0:
        btn = "4"  # scroll up
    else:
        btn = "5"  # scroll down
    for _ in range(abs(amount)):
        _xdotool(display, ["click", btn])
    print(f"Scrolled: {amount}")


# ---------------------------------------------------------------------------
# Interactive command: run-script
# ---------------------------------------------------------------------------

class _ScriptTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ScriptTimeout("Script execution timed out")


class _GuiHelper:
    """Convenience wrapper around xdotool for use in run-script."""

    def __init__(self, display: str):
        self.display = display

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        button_map = {"left": "1", "middle": "2", "right": "3"}
        btn = button_map.get(button, "1")
        _xdotool(self.display, ["mousemove", str(x), str(y)])
        time.sleep(0.05)
        for _ in range(clicks):
            _xdotool(self.display, ["click", btn])

    def type_text(self, text: str, interval: float = 0.05):
        delay_ms = max(1, int(interval * 1000))
        _xdotool(self.display, ["type", "--delay", str(delay_ms), "--", text])

    def key(self, keys: str):
        _xdotool(self.display, ["key", "--", keys])

    def scroll(self, amount: int):
        btn = "4" if amount >= 0 else "5"
        for _ in range(abs(amount)):
            _xdotool(self.display, ["click", btn])

    def screenshot(self, output: Optional[str] = None) -> str:
        path = output or _auto_filename("script")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        env = {"DISPLAY": self.display, "PATH": os.environ.get("PATH", "")}
        subprocess.run(
            ["import", "-window", "root", path],
            timeout=5, env=env, check=True, capture_output=True,
        )
        return path

    def find_window(self, name: str) -> Optional[str]:
        out = _xdotool(self.display, ["search", "--name", name], check=False)
        lines = out.strip().splitlines()
        return lines[0] if lines else None

    def focus_window(self, wid: str):
        # Use windowfocus (works without WM) with windowraise as fallback
        _xdotool(self.display, ["windowfocus", wid], check=False)
        _xdotool(self.display, ["windowraise", wid], check=False)


def cmd_run_script(display: str, script: str, timeout: int = SCRIPT_TIMEOUT):
    """Execute a Python script with GUI automation helpers.

    Available variables in the script:
        gui        — GuiHelper with click/type_text/key/scroll/screenshot/find_window
        display    — current display string
        subprocess — for running shell commands
        Path       — pathlib.Path
        time       — time module
        print      — standard print

    Safety:
        - Hard timeout (default 60s) via SIGALRM
        - Auto error screenshot on failure
    """
    helper = _GuiHelper(display)

    # Allow importing only safe modules
    _ALLOWED_IMPORTS = frozenset({"time", "json", "re", "math", "pathlib"})
    _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in _ALLOWED_IMPORTS:
            raise ImportError(f"import '{name}' is not allowed in gui scripts")
        return _real_import(name, globals, locals, fromlist, level)

    script_globals = {
        "gui": helper,
        "display": display,
        "subprocess": subprocess,
        "Path": Path,
        "time": time,
        "os": None,  # Blocked
        "print": print,
        "__builtins__": {
            "__import__": _safe_import,
            "print": print, "len": len, "str": str, "int": int,
            "float": float, "bool": bool, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "range": range, "type": type,
            "enumerate": enumerate, "zip": zip, "sorted": sorted,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
            "True": True, "False": False, "None": None,
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError,
        },
    }

    old_handler = None
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        exec(compile(script, "<gui-script>", "exec"), script_globals)
        print("\n[Script completed successfully]")
    except _ScriptTimeout:
        print(f"\n[ERROR] Script timed out after {timeout}s", file=sys.stderr)
        _save_error_screenshot(display)
    except SyntaxError as e:
        print(f"\n[Script syntax error] Line {e.lineno}: {e.msg}", file=sys.stderr)
        if e.text:
            print(f"  Code: {e.text.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"\n[Script error] {type(e).__name__}: {e}", file=sys.stderr)
        tb_lines = traceback.format_exc().splitlines()
        for line in tb_lines:
            if "<gui-script>" in line or "    " == line[:4]:
                print(f"  {line.strip()}", file=sys.stderr)
        _save_error_screenshot(display)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def _save_error_screenshot(display: str):
    """Save error state screenshot."""
    try:
        err_path = _auto_filename("gui_error")
        env = os.environ.copy()
        env["DISPLAY"] = display
        subprocess.run(
            ["import", "-window", "root", err_path],
            timeout=5, env=env, capture_output=True,
        )
        print(f"[Error state screenshot saved: {err_path}]")
    except Exception:
        print("[Could not save error screenshot]", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GUI tool for desktop application verification and interaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start virtual display
  %(prog)s start-display

  # Launch a tkinter app
  %(prog)s launch "python main.py"

  # Check status
  %(prog)s status

  # Screenshot
  %(prog)s screenshot

  # Interact
  %(prog)s click 640 360
  %(prog)s type "Hello World"
  %(prog)s key "ctrl+s"

  # Custom script
  %(prog)s run-script --script '
import time
gui.click(100, 200)
time.sleep(1)
gui.type_text("test", interval=0.05)
gui.key("Return")
'

  # Cleanup
  %(prog)s close
  %(prog)s stop-display
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start-display
    p = sub.add_parser("start-display", help="Start Xvfb virtual display")
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--size", default=DEFAULT_SCREEN_SIZE,
                   help=f"Screen size (default: {DEFAULT_SCREEN_SIZE})")

    # stop-display
    p = sub.add_parser("stop-display", help="Stop Xvfb virtual display")
    p.add_argument("--display", default=DEFAULT_DISPLAY)

    # launch
    p = sub.add_parser("launch", help="Launch a GUI application")
    p.add_argument("cmd", help="Command to run")
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--wait", type=float, default=LAUNCH_WAIT,
                   help=f"Seconds to wait after launch (default: {LAUNCH_WAIT})")

    # status
    p = sub.add_parser("status", help="Show display and application status")
    p.add_argument("--display", default=DEFAULT_DISPLAY)

    # close
    p = sub.add_parser("close", help="Close GUI application(s)")
    p.add_argument("--pid", type=int, help="Specific PID to close")
    p.add_argument("--display", default=DEFAULT_DISPLAY)

    # screenshot
    p = sub.add_parser("screenshot", help="Take a screenshot")
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--window", "-w", help="Window name to capture")

    # click
    p = sub.add_parser("click", help="Click at coordinates")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--button", default="left", choices=["left", "right", "middle"])
    p.add_argument("--clicks", type=int, default=1)

    # type
    p = sub.add_parser("type", help="Type text")
    p.add_argument("text")
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--interval", type=float, default=0.05)

    # key
    p = sub.add_parser("key", help="Press key(s)")
    p.add_argument("keys", help="Key name or combo (e.g. 'tab', 'ctrl+s')")
    p.add_argument("--display", default=DEFAULT_DISPLAY)

    # scroll
    p = sub.add_parser("scroll", help="Scroll mouse wheel")
    p.add_argument("amount", type=int, help="Positive=up, negative=down")
    p.add_argument("--display", default=DEFAULT_DISPLAY)

    # run-script
    p = sub.add_parser("run-script", help="Run custom GUI automation script")
    p.add_argument("--display", default=DEFAULT_DISPLAY)
    p.add_argument("--script", help="Inline Python script")
    p.add_argument("--file", "-f", help="Python script file")
    p.add_argument("--timeout", type=int, default=SCRIPT_TIMEOUT)

    args = parser.parse_args()

    if args.command == "start-display":
        cmd_start_display(args.display, args.size)
    elif args.command == "stop-display":
        cmd_stop_display(args.display)
    elif args.command == "launch":
        cmd_launch(args.cmd, args.display, args.wait)
    elif args.command == "status":
        cmd_status(args.display)
    elif args.command == "close":
        cmd_close(args.pid, args.display)
    elif args.command == "screenshot":
        cmd_screenshot(args.display, args.output, args.window)
    elif args.command == "click":
        cmd_click(args.x, args.y, args.display, args.button, args.clicks)
    elif args.command == "type":
        cmd_type_text(args.text, args.display, args.interval)
    elif args.command == "key":
        cmd_key(args.keys, args.display)
    elif args.command == "scroll":
        cmd_scroll(args.amount, args.display)
    elif args.command == "run-script":
        if args.file:
            script = Path(args.file).read_text(encoding="utf-8")
        elif args.script:
            script = args.script
        else:
            print("Error: --script or --file required", file=sys.stderr)
            sys.exit(1)
        cmd_run_script(args.display, script, timeout=args.timeout)


if __name__ == "__main__":
    main()
