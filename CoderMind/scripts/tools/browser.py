#!/usr/bin/env python3
"""Browser Tool — Web page screenshot, analysis, and interaction.

Provides CLI commands for verifying and interacting with web applications
using headless Chromium via Playwright.

Read-only commands (safe, deterministic):
    python tools/browser.py inspect <url>          (recommended: all-in-one)
    python tools/browser.py screenshot <url> [--output FILE]
    python tools/browser.py accessibility-tree <url>
    python tools/browser.py list-links <url>
    python tools/browser.py list-forms <url>
    python tools/browser.py get-html <url> [--selector CSS]

Interactive command (flexible, user writes Playwright code):
    python tools/browser.py run-script <url> --script 'page.fill(...); ...'
    python tools/browser.py run-script <url> --file script.py

Note: For scripts with single quotes or complex quoting, write the script
to a temp file and use --file instead of --script to avoid shell escaping issues.
"""

import argparse
import atexit
import json
import os
import re
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = ".cmind/tmp/screenshots"
DEFAULT_TIMEOUT = 10000      # 10s per Playwright operation
SCRIPT_TIMEOUT = 60          # 60s hard limit for run-script

# Track browser PIDs for emergency cleanup
_active_browser_pids: list = []


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

@contextmanager
def open_browser(headless: bool = True):
    """Context manager that yields (playwright, browser) and always cleans up.

    Safety guarantees:
    - Browser is always closed, even on exception or SIGTERM
    - Playwright server is always stopped
    - atexit handler kills any leaked chromium processes
    - All cleanup errors are logged (not silently swallowed)
    """
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=headless)

    # Track browser PID for emergency cleanup
    try:
        # Playwright exposes the process via internal API
        process = browser._impl_obj._browser_process
        if process and process.pid:
            _active_browser_pids.append(process.pid)
    except Exception:
        pass  # Not critical — just a safety net

    try:
        yield pw, browser
    finally:
        # Close browser
        try:
            browser.close()
        except Exception as e:
            print(f"[browser.py] Warning: browser.close() failed: {e}",
                  file=sys.stderr)

        # Stop Playwright
        try:
            pw.stop()
        except Exception as e:
            print(f"[browser.py] Warning: pw.stop() failed: {e}",
                  file=sys.stderr)

        # Remove from active PID tracking
        _active_browser_pids.clear()


def _emergency_cleanup():
    """Atexit handler: kill any leaked chromium processes."""
    for pid in _active_browser_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[browser.py] Emergency cleanup: killed chromium pid {pid}",
                  file=sys.stderr)
        except (OSError, ProcessLookupError):
            pass


atexit.register(_emergency_cleanup)


def open_page(browser, url: str, timeout: int = DEFAULT_TIMEOUT,
              exit_on_error: bool = False):
    """Open a page with fallback loading strategies.

    Args:
        browser: Playwright browser instance.
        url: URL to navigate to.
        timeout: Page load timeout in ms.
        exit_on_error: If True, sys.exit(1) on load failure.
    """
    page = browser.new_page()
    page.set_default_timeout(timeout)
    try:
        page.goto(url, wait_until="networkidle", timeout=timeout)
    except Exception:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception as e:
            err_type = type(e).__name__
            print(f"[browser.py] Page load failed for {url}", file=sys.stderr)
            print(f"  Error: {err_type}: {e}", file=sys.stderr)
            if "ERR_CONNECTION_REFUSED" in str(e):
                print("  [HINT] Is the server running? Start it first.", file=sys.stderr)
            elif "TIMEOUT" in str(e).upper():
                print("  [HINT] Server is slow or unresponsive. Try increasing --timeout.", file=sys.stderr)
            elif "ERR_NAME_NOT_RESOLVED" in str(e):
                print("  [HINT] Hostname not found. Check the URL.", file=sys.stderr)

    # Report if page failed to load
    if "chrome-error" in page.url:
        print(f"Error: could not load {url} (connection refused or unreachable)",
              file=sys.stderr)
        print("  [HINT] Make sure the web server is running on the correct port.",
              file=sys.stderr)
        if exit_on_error:
            browser.close()
            sys.exit(1)

    return page


def _ensure_dir(path: str):
    """Create parent directories for an output path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _auto_filename(url: str, label: str = "", ext: str = "png") -> str:
    """Generate a unique screenshot filename from URL + timestamp.

    Naming: HHMMSS_<label>_<url_path>[_q<query_hash>].<ext>
    Examples:
      http://localhost:5000/                → 143025_home.png
      http://localhost:5000/posts/hello     → 143025_posts_hello.png
      http://localhost:5000/search?q=test   → 143025_search_q3a1b.png
      (run-script after)                   → 143025_after_login.png
    """
    import hashlib
    parsed = urlparse(url)
    # Simplify path: /posts/hello-world → posts_hello-world
    path_part = parsed.path.strip("/").replace("/", "_") or "home"
    # Clean non-alphanumeric chars, keep readable
    path_part = re.sub(r"[^a-zA-Z0-9_\-]", "", path_part)[:40]

    # Include query string as short hash to distinguish /search?q=a from /search?q=b
    query_suffix = ""
    if parsed.query:
        qhash = hashlib.md5(parsed.query.encode()).hexdigest()[:5]
        query_suffix = f"_q{qhash}"

    timestamp = time.strftime("%H%M%S")
    parts = [timestamp]
    if label:
        parts.append(label)
    parts.append(path_part + query_suffix)
    name = "_".join(parts) + f".{ext}"
    out = f"{DEFAULT_OUTPUT_DIR}/{name}"
    _ensure_dir(out)
    return out


# ---------------------------------------------------------------------------
# Read-only commands
# ---------------------------------------------------------------------------

def cmd_screenshot(url: str, output: Optional[str] = None,
                   width: int = 1280, height: int = 720):
    """Take a full-page screenshot and save page HTML for analysis."""
    if not output:
        output = _auto_filename(url)
    _ensure_dir(output)
    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)
        page.set_viewport_size({"width": width, "height": height})
        page.screenshot(path=output, full_page=True)

        # Save HTML alongside screenshot for agent analysis
        html_path = output.rsplit(".", 1)[0] + ".html"
        try:
            html_content = page.content()
            Path(html_path).write_text(html_content, encoding="utf-8")
        except Exception:
            html_path = None

        print(f"Screenshot saved: {output}")
        print(f"Page title: {page.title()}")
        print(f"URL: {page.url}")
        if html_path:
            print(f"HTML saved: {html_path}")
            print(f"  (read this file to analyze page structure and content)")


def cmd_accessibility_tree(url: str):
    """Get a structured text representation of the page content."""
    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)

        # Try native accessibility API first
        try:
            snapshot = page.accessibility.snapshot()
            if snapshot:
                _print_a11y_node(snapshot, indent=0)
                return
        except (AttributeError, Exception):
            pass

        # Fallback: extract important elements via JS
        structure = page.evaluate("""() => {
            function walk(el, depth) {
                const lines = [];
                const tag = el.tagName ? el.tagName.toLowerCase() : '';
                if (['script','style','noscript','meta','link'].includes(tag)) return lines;

                const important = ['h1','h2','h3','h4','h5','h6','a','button','input',
                    'textarea','select','form','nav','main','header','footer',
                    'table','img','label','li','th','td'];
                const role = el.getAttribute ? (el.getAttribute('role') || '') : '';
                const isImportant = important.includes(tag) || role;

                if (isImportant && tag) {
                    let desc = tag;
                    if (role) desc += `[role=${role}]`;
                    const attrs = ['href','name','type','action','method','id'];
                    for (const a of attrs) {
                        const v = el.getAttribute ? el.getAttribute(a) : null;
                        if (v && v.length < 100) desc += ` ${a}="${v}"`;
                    }
                    const text = el.textContent ? el.textContent.trim() : '';
                    if (text && text.length < 80 && !el.children.length)
                        desc += ` "${text}"`;
                    lines.push('  '.repeat(depth) + desc);
                }

                if (el.children) {
                    for (const child of el.children)
                        lines.push(...walk(child, isImportant ? depth + 1 : depth));
                }
                return lines;
            }
            return walk(document.body, 0).join('\\n');
        }""")
        print(f"Page structure for {url}:\n")
        print(structure)


def _print_a11y_node(node: dict, indent: int = 0):
    """Recursively print accessibility tree nodes."""
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f'value="{value}"')
    print(f"{prefix}{' '.join(parts)}")
    for child in node.get("children", []):
        _print_a11y_node(child, indent + 1)


def cmd_list_links(url: str):
    """List all links on the page."""
    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)
        links = page.eval_on_selector_all(
            "a[href]",
            """elements => elements.map(el => ({
                text: el.textContent.trim().substring(0, 80),
                href: el.getAttribute('href'),
                visible: el.offsetParent !== null
            }))"""
        )
        print(f"Found {len(links)} links on {url}:\n")
        for link in links:
            vis = "✓" if link["visible"] else "✗"
            text = link["text"][:60] or "(no text)"
            print(f"  [{vis}] {link['href']}  — {text}")


def cmd_list_forms(url: str):
    """List all forms and their fields on the page."""
    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)
        forms = page.eval_on_selector_all(
            "form",
            """forms => forms.map((form, i) => {
                const inputs = Array.from(form.querySelectorAll('input, textarea, select'));
                return {
                    index: i,
                    action: form.getAttribute('action') || '(none)',
                    method: (form.getAttribute('method') || 'GET').toUpperCase(),
                    id: form.id || '(no id)',
                    fields: inputs.map(inp => ({
                        tag: inp.tagName.toLowerCase(),
                        type: inp.getAttribute('type') || inp.tagName.toLowerCase(),
                        name: inp.getAttribute('name') || '(unnamed)',
                        required: inp.hasAttribute('required'),
                        value: inp.value || ''
                    }))
                };
            })"""
        )
        if not forms:
            print(f"No forms found on {url}")
            return

        print(f"Found {len(forms)} form(s) on {url}:\n")
        for form in forms:
            print(f"  Form #{form['index']}: {form['method']} {form['action']} (id={form['id']})")
            for field in form["fields"]:
                req = " *required" if field["required"] else ""
                print(f"    - {field['tag']}[{field['type']}] name=\"{field['name']}\"{req}")
            print()


def cmd_inspect(url: str, width: int = 1280, height: int = 720):
    """Inspect a page: screenshot + HTML + links + forms + structure in one call.

    This is the recommended command for page analysis. It opens the browser once
    and collects all useful information, saving files for later analysis.

    Output:
        - Screenshot (.png) and HTML (.html) saved to .cmind/tmp/screenshots/
        - Prints: request URL, actual URL, title, status
        - Prints: all links with visibility
        - Prints: all forms with fields
        - Prints: page structure (headings, nav, buttons, inputs)
    """
    base_name = _auto_filename(url, ext="png")
    _ensure_dir(base_name)
    html_path = base_name.rsplit(".", 1)[0] + ".html"

    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)
        page.set_viewport_size({"width": width, "height": height})

        actual_url = page.url
        title = page.title()

        # 1. Screenshot
        page.screenshot(path=base_name, full_page=True)

        # 2. Save HTML
        try:
            Path(html_path).write_text(page.content(), encoding="utf-8")
        except Exception:
            html_path = "(failed to save)"

        # 3. Collect links
        links = page.eval_on_selector_all(
            "a[href]",
            """elements => elements.map(el => ({
                text: el.textContent.trim().substring(0, 60),
                href: el.getAttribute('href'),
                visible: el.offsetParent !== null
            }))"""
        )

        # 4. Collect forms
        forms = page.eval_on_selector_all(
            "form",
            """forms => forms.map((form, i) => {
                const inputs = Array.from(form.querySelectorAll('input, textarea, select, button'));
                return {
                    index: i,
                    action: form.getAttribute('action') || '(none)',
                    method: (form.getAttribute('method') || 'GET').toUpperCase(),
                    id: form.id || '',
                    fields: inputs.map(inp => ({
                        tag: inp.tagName.toLowerCase(),
                        type: inp.getAttribute('type') || inp.tagName.toLowerCase(),
                        name: inp.getAttribute('name') || '',
                        required: inp.hasAttribute('required'),
                        placeholder: inp.getAttribute('placeholder') || '',
                        text: inp.textContent ? inp.textContent.trim().substring(0, 30) : ''
                    }))
                };
            })"""
        )

        # 5. Page structure summary (compact)
        structure = page.evaluate("""() => {
            const lines = [];
            const important = ['h1','h2','h3','h4','h5','h6','nav','main','header',
                'footer','form','button','a','input','textarea','select','img','table'];
            function walk(el, depth) {
                const tag = el.tagName ? el.tagName.toLowerCase() : '';
                if (['script','style','noscript','meta','link','svg','path'].includes(tag)) return;
                const isImportant = important.includes(tag);
                if (isImportant) {
                    let desc = tag;
                    const attrs = {
                        'a': ['href'], 'img': ['src','alt'], 'form': ['action','method'],
                        'input': ['type','name','placeholder'], 'button': ['type'],
                        'textarea': ['name'], 'select': ['name']
                    };
                    for (const a of (attrs[tag] || [])) {
                        const v = el.getAttribute(a);
                        if (v && v.length < 80) desc += ' ' + a + '="' + v + '"';
                    }
                    const text = el.textContent ? el.textContent.trim() : '';
                    if (text && text.length < 60 && !el.children.length)
                        desc += ' "' + text + '"';
                    else if (tag.match(/^h[1-6]$/) && text)
                        desc += ' "' + text.substring(0, 60) + '"';
                    lines.push('  '.repeat(Math.min(depth, 6)) + desc);
                }
                if (el.children) {
                    for (const child of el.children)
                        walk(child, isImportant ? depth + 1 : depth);
                }
            }
            walk(document.body, 0);
            return lines.join('\\n');
        }""")

    # --- Print results ---
    print("=" * 60)
    print(f"PAGE INSPECT: {url}")
    print("=" * 60)
    print(f"  Request URL:  {url}")
    print(f"  Actual URL:   {actual_url}")
    if actual_url != url:
        print(f"  ** REDIRECTED from {url}")
    print(f"  Page title:   {title}")
    print(f"  Screenshot:   {base_name}")
    print(f"  HTML file:    {html_path}")
    print("-" * 60)
    print("  >>> Read the HTML file to analyze full page content <<<")
    print("-" * 60)
    print()

    # Links
    visible_links = [l for l in links if l["visible"]]
    hidden_links = [l for l in links if not l["visible"]]
    print(f"LINKS ({len(visible_links)} visible, {len(hidden_links)} hidden):")
    if visible_links:
        for link in visible_links:
            text = link["text"][:50] or "(no text)"
            print(f"  {link['href']:40s}  {text}")
    else:
        print("  (none)")
    print()

    # Forms
    print(f"FORMS ({len(forms)}):")
    if forms:
        for form in forms:
            fid = f" id={form['id']}" if form['id'] else ""
            print(f"  Form #{form['index']}: {form['method']} {form['action']}{fid}")
            for field in form["fields"]:
                parts = [f"{field['tag']}[{field['type']}]"]
                if field["name"]:
                    parts.append(f"name=\"{field['name']}\"")
                if field["required"]:
                    parts.append("*required")
                if field["placeholder"]:
                    parts.append(f"placeholder=\"{field['placeholder']}\"")
                if field["text"] and field["tag"] == "button":
                    parts.append(f"\"{field['text']}\"")
                print(f"    {' '.join(parts)}")
    else:
        print("  (none)")
    print()

    # Structure
    print("PAGE STRUCTURE:")
    if structure.strip():
        struct_lines = structure.strip().split("\n")
        for line in struct_lines[:80]:
            print(f"  {line}")
        if len(struct_lines) > 80:
            print(f"  ... ({len(struct_lines) - 80} more elements)")
    else:
        print("  (empty page)")
    print("=" * 60)


def cmd_get_html(url: str, selector: Optional[str] = None):
    """Get the rendered HTML of a page or a specific element."""
    with open_browser() as (pw, browser):
        page = open_page(browser, url, exit_on_error=True)
        if selector:
            el = page.query_selector(selector)
            if el:
                html = el.inner_html()
                print(f"HTML for '{selector}' on {url} ({len(html)} chars):\n")
            else:
                print(f"Selector '{selector}' not found on {url}")
                return
        else:
            html = page.content()
            print(f"Full HTML for {url} ({len(html)} chars):\n")

        if len(html) > 15000:
            print(html[:15000])
            print(f"\n... (truncated, {len(html)} total chars)")
        else:
            print(html)


# ---------------------------------------------------------------------------
# Interactive command: run-script
# ---------------------------------------------------------------------------

class _ScriptTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ScriptTimeout("Script execution timed out")


def cmd_run_script(url: str, script: str, timeout: int = SCRIPT_TIMEOUT):
    """Execute a Playwright Python script with safety measures.

    Available variables in the script:
        page     — Playwright Page object (already navigated to url)
        browser  — Playwright Browser instance
        Path     — pathlib.Path
        json     — json module
        print    — standard print function

    Safety:
        - Hard timeout (default 60s) via SIGALRM
        - Browser always cleaned up via context manager
        - On error: automatic screenshot saved to .cmind/tmp/screenshots/
        - Restricted builtins (no os, subprocess, sys access)
    """
    with open_browser() as (pw, browser):
        page = open_page(browser, url)

        # Capture initial state
        initial_url = page.url
        initial_title = page.title()
        print(f"[Before] URL: {initial_url}")
        print(f"[Before] Title: {initial_title}")

        # Sandbox context: expose safe variables only
        _ALLOWED_IMPORTS = frozenset({"time", "json", "re", "math", "pathlib"})
        _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name not in _ALLOWED_IMPORTS:
                raise ImportError(f"import '{name}' is not allowed in browser scripts")
            return _real_import(name, globals, locals, fromlist, level)

        script_globals = {
            "page": page,
            "browser": browser,
            "Path": Path,
            "json": json,
            "print": print,
            "__builtins__": {
                "__import__": _safe_import,
                # Basic types and functions
                "print": print, "len": len, "str": str, "int": int,
                "float": float, "bool": bool, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "range": range, "type": type,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "sorted": sorted, "reversed": reversed, "min": min, "max": max,
                "abs": abs, "sum": sum, "any": any, "all": all,
                "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                "setattr": setattr,
                "True": True, "False": False, "None": None,
                # Exceptions
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "AttributeError": AttributeError,
                "RuntimeError": RuntimeError,
            },
        }

        # Set up hard timeout (Unix only)
        old_handler = None
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            exec(compile(script, "<browser-script>", "exec"), script_globals)
            print("\n[Script completed successfully]")

            # Capture final state after script execution
            final_url = page.url
            final_title = page.title()
            print(f"[After] URL: {final_url}")
            print(f"[After] Title: {final_title}")
            if final_url != initial_url:
                print(f"[Navigation] {initial_url} → {final_url}")

            # Auto-save screenshot and HTML for agent analysis
            auto_screenshot = _auto_filename(final_url, label="after")
            try:
                page.screenshot(path=auto_screenshot, full_page=True)
                print(f"[After] Screenshot: {auto_screenshot}")
            except Exception:
                pass

            auto_html = auto_screenshot.rsplit(".", 1)[0] + ".html"
            try:
                Path(auto_html).write_text(page.content(), encoding="utf-8")
                print(f"[After] HTML: {auto_html}")
                print(f"  (read this file to analyze the resulting page)")
            except Exception:
                pass
        except _ScriptTimeout:
            print(f"\n[ERROR] Script timed out after {timeout}s", file=sys.stderr)
            print("[HINT] Possible causes:", file=sys.stderr)
            print("  - Waiting for an element that doesn't exist (use timeout param)", file=sys.stderr)
            print("  - Infinite loop in script", file=sys.stderr)
            print("  - Server not responding", file=sys.stderr)
            print(f"  Current URL: {page.url}", file=sys.stderr)
            _save_error_screenshot(page)
        except SyntaxError as e:
            print(f"\n[Script syntax error] Line {e.lineno}: {e.msg}", file=sys.stderr)
            if e.text:
                print(f"  Code: {e.text.strip()}", file=sys.stderr)
            print("[HINT] Check for unclosed brackets, quotes, or indentation", file=sys.stderr)
        except Exception as e:
            print(f"\n[Script error] {type(e).__name__}: {e}", file=sys.stderr)
            try:
                print(f"  Current URL: {page.url}", file=sys.stderr)
                print(f"  Page title: {page.title()}", file=sys.stderr)
            except Exception:
                pass
            # Include traceback for complex errors
            tb_lines = traceback.format_exc().splitlines()
            # Show only the relevant part (skip exec internals)
            for line in tb_lines:
                if "<browser-script>" in line or "    " == line[:4]:
                    print(f"  {line.strip()}", file=sys.stderr)
            _save_error_screenshot(page)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)


def _save_error_screenshot(page, url: str = ""):
    """Save a screenshot of the current page state on error."""
    try:
        err_path = _auto_filename(url or page.url, label="error")
        page.screenshot(path=err_path, full_page=True)
        print(f"[Error state screenshot saved: {err_path}]")
    except Exception:
        print("[Could not save error screenshot]", file=sys.stderr)


def cmd_check() -> int:
    """Verify Playwright is installed and a headless Chromium can launch.

    Used by templates (e.g. ``rpg_edit.md`` Step 3.5) to decide whether
    the optional visual reconnaissance step can run. Prints a short
    human-readable line on stdout and returns 0 on success, non-zero on
    failure (so callers can use ``if python3 browser.py check; then ...``).

    Failure modes:
      - Playwright not importable → exit 2
      - Chromium browser launch fails (missing binaries, sandbox issue,
        no display) → exit 3
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        print(f"playwright not available: {exc}", file=sys.stderr)
        return 2

    try:
        with open_browser(headless=True) as (_pw, browser):
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_content("<html><body>ok</body></html>")
            page.close()
            ctx.close()
    except Exception as exc:
        print(f"playwright headless launch failed: {exc}", file=sys.stderr)
        return 3

    print("playwright OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browser tool for web page verification and interaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a page (recommended: screenshot + HTML + links + forms + structure)
  %(prog)s inspect http://localhost:5000/
  %(prog)s inspect http://localhost:5000/login

  # Screenshot only
  %(prog)s screenshot http://localhost:5000/ -o .cmind/tmp/home.png

  # Page structure
  %(prog)s accessibility-tree http://localhost:5000/

  # All links
  %(prog)s list-links http://localhost:5000/

  # All forms + fields
  %(prog)s list-forms http://localhost:5000/login

  # HTML of a specific element
  %(prog)s get-html http://localhost:5000/ --selector "nav"

  # Custom interaction script
  %(prog)s run-script http://localhost:5000/login --script '
page.fill("input[name=username]", "admin")
page.fill("input[name=password]", "admin123")
page.click("button[type=submit]")
page.wait_for_load_state("networkidle")
print("URL after login:", page.url)
print("Title:", page.title())
page.screenshot(path=".cmind/tmp/after_login.png", full_page=True)
'
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check (used by templates to detect optional visual recon support)
    sub.add_parser("check",
                   help="Verify Playwright is installed and headless Chromium can launch")

    # inspect (all-in-one)
    p = sub.add_parser("inspect",
                       help="Inspect a page: screenshot + HTML + links + forms + structure")
    p.add_argument("url")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # screenshot
    p = sub.add_parser("screenshot", help="Take a full-page screenshot")
    p.add_argument("url")
    p.add_argument("--output", "-o", default=None,
                   help="Output file (default: auto-generated timestamped name)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # accessibility-tree
    p = sub.add_parser("accessibility-tree", help="Get page structure as text")
    p.add_argument("url")

    # list-links
    p = sub.add_parser("list-links", help="List all links on a page")
    p.add_argument("url")

    # list-forms
    p = sub.add_parser("list-forms", help="List all forms and their fields")
    p.add_argument("url")

    # get-html
    p = sub.add_parser("get-html", help="Get rendered HTML of page or element")
    p.add_argument("url")
    p.add_argument("--selector", "-s", help="CSS selector (default: full page)")

    # run-script
    p = sub.add_parser("run-script", help="Run custom Playwright Python script")
    p.add_argument("url", help="Starting URL (page is pre-navigated)")
    p.add_argument("--script", help="Inline Python script")
    p.add_argument("--file", "-f", help="Python script file to execute")
    p.add_argument("--timeout", type=int, default=SCRIPT_TIMEOUT,
                   help=f"Script timeout in seconds (default: {SCRIPT_TIMEOUT})")

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check())
    elif args.command == "inspect":
        cmd_inspect(args.url, args.width, args.height)
    elif args.command == "screenshot":
        cmd_screenshot(args.url, args.output, args.width, args.height)
    elif args.command == "accessibility-tree":
        cmd_accessibility_tree(args.url)
    elif args.command == "list-links":
        cmd_list_links(args.url)
    elif args.command == "list-forms":
        cmd_list_forms(args.url)
    elif args.command == "get-html":
        cmd_get_html(args.url, args.selector)
    elif args.command == "run-script":
        if args.file:
            script = Path(args.file).read_text(encoding="utf-8")
        elif args.script:
            script = args.script
        else:
            print("Error: --script or --file required for run-script", file=sys.stderr)
            sys.exit(1)
        cmd_run_script(args.url, script, timeout=args.timeout)


if __name__ == "__main__":
    main()
