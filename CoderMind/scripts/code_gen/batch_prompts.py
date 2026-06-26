#!/usr/bin/env python3
"""Per-batch TDD prompt assembly for the codegen pipeline.

This module hosts the prompt-builder helpers extracted from
``scripts/run_batch.py`` Module 1 ("Prompt Builder").  They assemble
the full prompt that ``run_batch``'s sub-agent receives for a single
batch (test code + production code + pytest cmd + dependency context).

Distinct from :mod:`scripts.code_gen.prompts`, which contains the
*pure-template* strings (``init_test_gen_prompt``, ``test_fix_prompt``,
``FAILURE_ANALYSIS_PROMPT``, …).  This module assembles those templates
plus batch-specific runtime context (venv python path, dep_graph,
import conventions, …) into the final TDD batch prompt.

Internal to the codegen package; no external API contract.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.execution_state import BatchExecutionState, load_code_gen_state
from common.generated_artifacts import generated_artifact_prompt_rule
from common.import_normalizer import build_import_convention_snippet
from common.paths import (
    CODE_GEN_STATE_FILE as STATE_FILE,
    FEATURE_SPEC_FILE,
    REPO_RPG_FILE,
    TASKS_FILE,
    get_scripts_dir,
)
from common.task_batch import PlannedTask, load_tasks_from_tasks_json
from code_gen.prompts import (
    _format_dependency_context,
    is_project_docs_batch,
    is_project_file_batch,
)
from code_gen.sub_agent import truncate_test_output
from code_gen.test_runner import (
    find_related_test_files,
    get_dev_python,
    get_dev_venv_path,
)
from decoder_lang import (
    EnvHandle,
    LanguageBackend,
    ToolchainUnavailable,
    get_backend,
    resolve_decoder_language,
    scan_repo_source_files,
)

logger = logging.getLogger(__name__)


from code_gen._constants import DEFAULT_TEST_TIMEOUT  # noqa: E402

# Sub-agent internal TDD-loop iteration cap (enforced inside the
# generated prompt; not used to drive any Python-side loop).
MAX_ITERATIONS = 5

_FALLBACK_TEST_COMMANDS = {
    "go": ["go", "test", "-v", "./..."],
    "rust": ["cargo", "test"],
    "typescript": ["npm", "test"],
    "javascript": ["npm", "test"],
    "c": ["make", "test"],
    "cpp": ["ctest", "--test-dir", "build", "--output-on-failure"],
}


# ============================================================================
# Prompt Templates
# ============================================================================

TDD_BATCH_PREAMBLE = """\
# TDD Batch Implementation

You are an autonomous coding agent completing a single implementation batch
inside a structured TDD workflow. You have **full project access** and must
self-manage the entire write → test → fix cycle.

## ── Workflow ──────────────────────────────────────────────

Follow these steps IN ORDER. Do not skip steps.

### Step 1: Understand Context
- Read the target source file (skeleton may exist with interface stubs).
- Read dependency files listed in the Project Context section below.
- **Explore beyond listed dependencies.** Read any source file that your target
  file imports or will interact with. If a completed module provides functions
  your code should call, read that module to understand its actual API
  (function signatures, return types, class interfaces). Do NOT guess — read
  the real code.
- If your target file produces output consumed by other modules (e.g., generates
  HTML with CSS classes, returns data structures), read those consuming modules
  to ensure compatibility.
- Read existing test files in `tests/` to understand conventions.
- {dependency_manifest_instruction}
- **UI/View code quality:** If you are implementing code that generates HTML,
  renders pages, produces visual output, or defines styles/CSS:
  - Ensure all HTML pages use the shared layout (head, nav, footer) consistently
  - Use the CSS class names already defined in the stylesheet — read the style
    module first and use its exact class names in your HTML
  - Wrap content in proper layout containers (e.g., `.container`, `<main>`)
  - Produce complete, production-quality pages — not minimal stubs
  - Include proper form structure (labels, fieldsets, CSRF tokens where needed)
  - All pages should look like they belong to the same application
  - If the project needs static assets (CSS files, templates, images) that don't
    exist yet, create them. You have permission to create any project files needed.
  - **Layout verification:** After writing layout code (CSS grid/flex for web,
    layout managers for GUI), verify the structure is correct:
    - For CSS grid/flex: count child elements vs column/row definitions.
      Example bug: `grid-template-columns: 1fr 300px` with 3 children (h1,
      content, sidebar) — h1 takes column 1, content gets pushed to 300px column.
      Fix: add `grid-column: 1 / -1` to spanning elements, or restructure HTML.
    - For GUI: verify widgets are placed in the correct parent container and
      pack/grid/place calls produce the intended layout.
  - **Content display:** The primary content area of every screen must show
    meaningful content. Never leave the main area empty while content is
    squeezed into a sidebar, toolbar, or secondary panel.
- **User-facing output quality:** Regardless of project type (web, GUI, CLI),
  all user-facing output should be polished and professional:
  - CLI tools: use clear formatting, aligned columns, colors/bold where helpful,
    progress indicators for long operations, and helpful error messages
  - GUI apps: consistent widget styling, proper layout management, sensible
    defaults, and intuitive navigation
  - Web apps: consistent page layout, working navigation, styled forms, and
    responsive basics (viewport meta tag, flexible widths)

### Step 2: Write Tests
{test_instructions}

### Step 3: Write Implementation
{code_instructions}

### Step 4: Run Tests
Run ONLY this command (no variations):
```
{pytest_cmd}
```
**CRITICAL**: This command runs ALL tests in the `tests/` directory,
not just the ones you wrote in this batch. Your new code must pass
ALL pre-existing tests as well as your new ones.
If pre-existing tests fail after your changes, your code has a bug —
fix YOUR code, not the pre-existing tests (unless the test itself is
clearly wrong based on the skeleton).

### Step 5: Analyze & Fix (if tests fail)
- Read the FULL {test_tool_name} output carefully.
- Determine root cause: test bug, code bug, import error, or dependency issue.
- Fix the appropriate file(s). You MAY fix:
  - Test files (wrong assertions, bad mocks, missing imports)
  - Source files (logic bugs, missing methods, wrong signatures)
    - Other project files (broken imports, missing package markers)
    - Dependency manifests (missing third-party package or module)
- After fixing, re-run the EXACT SAME test command from Step 4.

### Step 6: Repeat Steps 4–5
- Maximum **{max_iterations} iterations** of test → fix → test.
- If tests pass, proceed to Step 7 immediately.
- If after {max_iterations} iterations tests still fail, proceed to Step 7 anyway.

### Step 7: Save & Report
Commit with a conventional-commit message describing what you implemented:
```
git add -A
git commit -m "feat(<module>): <brief description>" \\
  -m "<bullet list of key changes>" \\
  -m "Target: {file_path}" \\
  -m "Units: {units}" \\
  -m "Batch-Id: {batch_id}"
```
The subject line MUST follow this format: `feat(<module>): <what>`
The body MUST include a bullet list of what was implemented/changed.
Examples:
```
git add -A
git commit -m "feat(auth/routes): implement LoginHandler with JWT authentication" \\
  -m "- Add LoginHandler class with login/logout/refresh endpoints
- Implement JWT token generation with configurable expiry
- Add password hashing with bcrypt" \\
  -m "Target: src/personal-blog-system/auth/routes.py" \\
  -m "Units: LoginHandler" \\
  -m "Batch-Id: {batch_id}"
```

## Exit Protocol — How to Report Your Result

The final two lines of your response MUST follow this exact shape so the
runner can verify your claim:

```
PYTEST_SUMMARY: <verbatim last summary line from the test command>
BATCH_RESULT: PASS
```

or on failure:

```
PYTEST_SUMMARY: <verbatim last summary line from the test command>
BATCH_RESULT: FAIL | <one-line reason>
```

The `PYTEST_SUMMARY` marker name is kept for runner compatibility. Its
value must be the *literal* one-line summary printed by the test command,
for example `5 passed in 0.42s`, `ok ./...`, or `test result: ok`. Copy it
verbatim from the run you just performed; do NOT invent it. This lets the
runner cross-check your claim against an independent re-run.
{summary_fallback_rule}

## ── Capabilities ─────────────────────────────────────────

[OK] You CAN:
- Read/write any file under `src/`, `tests/`, `static/`, `templates/`, and `examples/`
  (source files in the target language, plus HTML, CSS, JSON, YAML, config files, etc.)
- Create new directories and files if needed (e.g., `static/css/`, `templates/`)
- Read any file in the repo for context
- Run: `{pytest_cmd}` (this exact command only)
{dependency_install_capability}
- Fix import errors in ANY source file (not just the target)
- Run: `git add -A && git commit -m "<message>"`

[FAIL] You MUST NOT:
- Modify or read files under `.cmind/`
- Run any `cmind script ...` or `cmind-mcp` commands
- Run arbitrary shell commands beyond the test/dependency/git commands listed above
- Install packages that are not genuinely needed by the source code
- Delete files that are not part of your task
{test_timeout_rule}

## ── Test Command Rules (CRITICAL) ─────────────────────────────

1. **Always use the EXACT {test_tool_name} command provided**.
2. **Do not manually run a different test command** — the provided command
    already targets the correct test scope for this batch.
3. If a test times out or hangs, the test is wrong. Fix the test:
   - Remove infinite loops, blocking I/O, or real-time sleeps/waits
   - Mock any external resources (network, filesystem, GPU)
   - Ensure all fixtures have finite setup/teardown
4. **Do not write tests that depend on timing** (real-time waits).
   Mock time-dependent behavior with your target language's test/mocking
   framework (see the Target Language section below).
5. **Do not write tests that spawn subprocesses or servers.**
6. **Output control:** prefer fail-fast and concise tracebacks so the
   FIRST failure stays the focus; follow the exact test command provided.

## ── Test Quality Rules ───────────────────────────────────

- Use spec'd / auto-generated mocks bound to a real type, never an
  unconstrained stand-in, and mock at boundaries (I/O, external deps),
  not internal implementation. Use your target language's idiomatic
  mocking facility.
- For numeric/math operations: use real values, not mocks.
- Keep tests deterministic — no random data without fixed seeds.
- Test count: proportional to task complexity. Small task = 3–8 tests.
  Do NOT over-engineer with 20+ tests for a simple class.

## ── Dependency Management ────────────────────────────────

{dependency_management}

{import_convention}

{language_context}

## ── Project Context ──────────────────────────────────────
{dependency_context}

## ── Task Details ─────────────────────────────────────────

**Batch ID:** {batch_id}
**Target file:** {file_path}
**Units to implement:** [{units}]
**Task type:** {task_type}
"""

TDD_RESUME_PREAMBLE = """\
# TDD Batch — Resume After Previous Failure

A previous attempt at this batch failed.  Code may be partially written.
Your job is to **continue from where it left off** and make tests pass.

## Previous Failure Info
**Attempt:** {attempt_number}
**Failure reason:** {failure_reason}
{post_verify_section}
## Previous Test Output (last test-command run)
```
{last_test_output}
```

## Instructions
1. Review what has already been written (read modified files).
2. Run the exact test command to see current status.
3. If tests fail → fix the **production code** first, then re-run the exact test command.
4. **Do NOT silence failures by editing tests** — the tests in `tests/`
   describe the contract.  Only modify a test if you can show it is
   logically wrong (wrong expected value, wrong fixture, etc.) and
   document the reason in your reply.
5. If tests pass → commit, then exit with the **Exit Protocol** below.

## Exit Protocol (same as the original task)
The final two lines of your response MUST be:
```
PYTEST_SUMMARY: <verbatim last summary line from the test command>
BATCH_RESULT: PASS    # or FAIL | <one-line reason>
```
The `PYTEST_SUMMARY` marker is kept for runner compatibility. Copy the
last summary line from your test-command run verbatim.

All other rules from the original task apply (capabilities, constraints,
test-command rules, etc). The full original task is included below.
"""

TDD_PROJECT_FILE_PREAMBLE = """\
# Project File Generation Task

You are creating a project file as part of a finalization workflow.
The core implementation is already complete.

## Your Capabilities
[OK] You CAN:
- Read any file in the repo to understand the codebase
- Create or update the requested project file(s)
- Run validation commands as specified below

[FAIL] You MUST NOT:
- Modify existing source code or test files
- Modify or read files under `.cmind/`
- Run any `cmind script ...` or `cmind-mcp` commands

## Task Details

**Batch ID:** {batch_id}
**Task type:** {task_type}

{code_prompt}

## Exit Protocol
When finished, on the LAST line of your response write:
- Success: `BATCH_RESULT: PASS`
- Failure: `BATCH_RESULT: FAIL | <one-line reason>`
"""

TDD_DOCS_PREAMBLE = """\
# Documentation Generation Task

You are creating documentation for the project. No tests are needed.

## Your Capabilities
[OK] You CAN:
- Read any file in the repo to understand the codebase
- Create or update documentation files (README.md, docs/, etc.)

[FAIL] You MUST NOT:
- Modify existing source code or test files
- Modify or read files under `.cmind/`

## Task Details

**Batch ID:** {batch_id}

{code_prompt}

## Exit Protocol
When finished, on the LAST line of your response write:
`BATCH_RESULT: PASS`
"""


# ============================================================================
# Builder functions
# ============================================================================

def _load_json_if_exists(path: Path) -> Any:
    """Load JSON from ``path`` or return None when unavailable."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_codegen_backend(repo_path: Path | None = None) -> LanguageBackend:
    """Resolve the target language backend for code generation."""
    feature_spec = _load_json_if_exists(FEATURE_SPEC_FILE)
    rpg_obj = _load_json_if_exists(REPO_RPG_FILE)
    valid_files = None
    if repo_path is not None:
        valid_files = scan_repo_source_files(repo_path) or None
    language = resolve_decoder_language(
        feature_spec=feature_spec,
        rpg_obj=rpg_obj,
        valid_files=valid_files,
    )
    return get_backend(language)


def _shell_join(argv: List[str]) -> str:
    """Return a shell-safe command string for display in prompts."""
    return shlex.join([str(part) for part in argv])


def _fallback_test_command(backend: LanguageBackend) -> List[str]:
    """Return a stable test command when host tool detection is unavailable."""
    return list(_FALLBACK_TEST_COMMANDS.get(backend.name, [backend.prompt_hints().test_framework_name]))


def _dynamic_c_family_syntax_command(
    backend: LanguageBackend,
    command: List[str],
) -> str:
    compiler = shlex.quote(str(command[0]))
    include_flags: List[str] = []
    for index, part in enumerate(command):
        if part == "-I" and index + 1 < len(command):
            include_flags.append('-I "$PWD"')
    standard = "-std=c++17" if backend.name == "cpp" else "-std=c99"
    patterns = (
        r'\( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \)'
        if backend.name == "cpp"
        else r'-name "*.c"'
    )
    include_text = " ".join(include_flags)
    return (
        "bash -lc "
        + shlex.quote(
            "mapfile -d '' sources < <(find . "
            r"\( -path './.git' -o -path './.cmind' -o -path './build' "
            r"-o -path './node_modules' -o -path './target' "
            r"-o -path './dist' -o -path './coverage' -o -path './.venv' "
            r"-o -path './venv' -o -path './CMakeFiles' \) -prune "
            f"-o -type f {patterns} -print0); "
            f"if (( ${{#sources[@]}} == 0 )); then echo 'No {backend.prompt_hints().display_name} source files found' >&2; exit 1; fi; "
            f"{compiler} {standard} {include_text} -Wall -Wextra -fsyntax-only \"${{sources[@]}}\""
        )
    )


def _cmake_c_family_test_command(command: List[str]) -> str:
    ctest = shlex.quote(str(command[0]))
    return (
        "bash -lc "
        + shlex.quote(
            "cmake -S . -B build && "
            "cmake --build build && "
            f"{ctest} --test-dir build --output-on-failure"
        )
    )


def _build_backend_test_cmd(
    backend: LanguageBackend,
    repo_path: Path,
    test_files: List[str],
    venv_python: str,
) -> str:
    """Build the exact test command the codegen agent should run."""
    if backend.name == "python":
        return build_batch_pytest_cmd(test_files, venv_python)

    env = backend.detect_env(repo_path) or EnvHandle(project_root=repo_path.resolve())
    try:
        command = backend.test_command(env)
        if backend.name in {"c", "cpp"} and command and "ctest" in Path(str(command[0])).name:
            return _cmake_c_family_test_command(command)
        if backend.name in {"c", "cpp"} and "-fsyntax-only" in command:
            return _dynamic_c_family_syntax_command(backend, command)
        return _shell_join(command)
    except (ToolchainUnavailable, NotImplementedError, OSError):
        return _shell_join(_fallback_test_command(backend))


def _dependency_manifest_instruction(backend: LanguageBackend) -> str:
    """Return the dependency manifest reading instruction for the backend."""
    manifest_by_language = {
        "python": "Read `requirements.txt` if it exists.",
        "go": "Read `go.mod` if it exists.",
        "rust": "Read `Cargo.toml` if it exists.",
        "typescript": "Read `package.json` and `tsconfig.json` if they exist.",
        "javascript": "Read `package.json` if it exists.",
        "c": "Read `Makefile` if it exists.",
        "cpp": "Read `CMakeLists.txt` or `Makefile` if they exist.",
    }
    return manifest_by_language.get(backend.name, "Read the project's dependency manifest if it exists.")


def _dependency_install_capability(backend: LanguageBackend, repo_path: Path) -> str:
    """Return the allowed dependency-install command bullet."""
    if backend.name == "python":
        return f"- Run: `{_build_pip_install_cmd(repo_path)} install <package>` to install missing packages\n- Update `requirements.txt` when adding new dependencies"
    capability_by_language = {
        "go": "- Run: `go get <module>` only when a non-standard module is genuinely required\n- Update `go.mod` / `go.sum` when adding dependencies",
        "rust": "- Run: `cargo add <crate>` only when a crate is genuinely required\n- Update `Cargo.toml` / `Cargo.lock` when adding dependencies",
        "typescript": "- Run: `npm install <package>` only when a package is genuinely required\n- Update `package.json` / lockfiles when adding dependencies",
        "javascript": "- Run: `npm install <package>` only when a package is genuinely required\n- Update `package.json` / lockfiles when adding dependencies",
        "c": "- Prefer the C standard library; do not install system packages from this workflow\n- Update `Makefile` when build flags or source lists change",
        "cpp": "- Prefer the C++ standard library; do not install system packages from this workflow\n- Update `CMakeLists.txt` or `Makefile` when build flags or source lists change",
    }
    return capability_by_language.get(
        backend.name,
        "- Use the project's native dependency tool only when a dependency is genuinely required",
    )


def _dependency_management_text(backend: LanguageBackend, repo_path: Path) -> str:
    """Return dependency-management instructions for the target language."""
    if backend.name == "python":
        pip_cmd = _build_pip_install_cmd(repo_path)
        return (
            "When you encounter `ModuleNotFoundError` or `ImportError` for a third-party package:\n"
            f"1. Install it: `{pip_cmd} install <package>`\n"
            "2. Verify by re-running the exact test command.\n"
            "3. Append the package to `requirements.txt` (create the file if it doesn't exist)."
        )
    management_by_language = {
        "go": (
            "When a non-standard Go module is genuinely needed:\n"
            "1. Run `go get <module>`.\n"
            "2. Verify by re-running the exact test command.\n"
            "3. Keep `go.mod` and `go.sum` consistent."
        ),
        "rust": (
            "When an external Rust crate is genuinely needed:\n"
            "1. Run `cargo add <crate>`.\n"
            "2. Verify by re-running the exact test command.\n"
            "3. Keep `Cargo.toml` and `Cargo.lock` consistent."
        ),
        "typescript": (
            "When an npm package is genuinely needed:\n"
            "1. Run `npm install <package>`.\n"
            "2. Verify by re-running the exact test command.\n"
            "3. Keep `package.json` and lockfiles consistent."
        ),
        "javascript": (
            "When an npm package is genuinely needed:\n"
            "1. Run `npm install <package>`.\n"
            "2. Verify by re-running the exact test command.\n"
            "3. Keep `package.json` and lockfiles consistent."
        ),
        "c": "Prefer the C standard library. Do not add system dependencies unless the repository already documents them.",
        "cpp": "Prefer the C++ standard library. Do not add system dependencies unless the repository already documents them.",
    }
    return management_by_language.get(
        backend.name,
        "Use the project's native dependency workflow and re-run the exact test command after changes.",
    )


def _test_timeout_rule(backend: LanguageBackend) -> str:
    """Return a timeout-safety rule tailored to the test command."""
    if backend.name == "python":
        return "- Run pytest without `--timeout` flag (already included in the command)"
    return "- Run long-lived servers, watchers, or interactive commands instead of the exact test command"


def _summary_fallback_rule(backend: LanguageBackend, test_command: str) -> str:
    if backend.name in {"c", "cpp"} and "-fsyntax-only" in test_command:
        return (
            "\nFor C/C++ syntax-only commands: if the exact command exits 0 "
            "and prints no summary line, use exactly "
            "`PYTEST_SUMMARY: syntax check passed`.\n"
        )
    return ""


def _build_language_context(backend: LanguageBackend, test_command: str) -> str:
    """Build the target-language prompt section."""
    hints = backend.prompt_hints()
    context = (
        "## ── Target Language ─────────────────────────────────────\n"
        f"- Language: {hints.display_name}\n"
        f"- Source extension: `{hints.file_extension}`\n"
        f"- Code fences: ```{hints.markdown_fence}\n"
        f"- Test command: `{test_command}`\n"
        f"- Test framework/tool: {hints.test_framework_name}\n"
        f"- Module naming: {hints.module_naming_rule}\n"
        f"- Style: {hints.style_directive}\n"
    )
    artifact_extra = ""
    if backend.name in {"c", "cpp"}:
        artifact_extra = (
            "If CTest needs arguments or target wiring, change source files "
            "such as `CMakeLists.txt` or the test source instead."
        )
    context += generated_artifact_prompt_rule(artifact_extra)
    if backend.name != "python":
        # The decoder's defaults are Python-centric; without an explicit
        # prohibition the sub-agent tends to add Python helpers (a main.py
        # launcher wrapper, a pytest conftest.py to drive native tests, a
        # requirements.txt). Forbid them outright so the generated repo stays
        # a pure single-language project.
        context += (
            f"- **This is a {hints.display_name} project, NOT Python.** Every source and test "
            f"file you create MUST use `{hints.file_extension}` (or the language's own test "
            "suffix). Do NOT create ANY `.py` file.\n"
            "- Specifically FORBIDDEN: `main.py` or any Python launcher/wrapper, `conftest.py`, "
            "`pytest.ini`, `setup.py`, `pyproject.toml`, `requirements.txt`, `__init__.py`, or a "
            "`.venv`/pip workflow.\n"
            f"- Run tests ONLY with `{test_command}` ({hints.test_framework_name}). Do NOT wrap, "
            "re-implement, or drive the test suite through pytest or any Python script.\n"
        )
        if backend.name in {"c", "cpp"}:
            context += (
                "- C/C++ tests and examples must be valid standalone translation units. "
                "If a test or example calls a helper implemented in another `.c`/`.cpp` file, "
                "create or update a matching header and include that header; do NOT rely on "
                "transitive `.cpp` inclusion or undeclared functions.\n"
            )
    else:
        context += (
            "- Do NOT introduce Python-specific files, packages, or pytest conventions unless this is a Python project.\n"
        )
    return context

def build_batch_pytest_cmd(
    test_files: List[str],
    venv_python: str,
    per_test_timeout: int = DEFAULT_TEST_TIMEOUT,
) -> str:
    """Build a pytest command with timeout protection.

    Args:
        test_files: Test files to run (empty → tests/).
        venv_python: Path to venv python executable.
        per_test_timeout: Max seconds per individual test function.

    Returns:
        Shell command string ready for the sub-agent to copy-paste.
    """
    files_str = " ".join(test_files) if test_files else "tests/"
    return (
        f"{venv_python} -m pytest {files_str} "
        f"-x --tb=short -q "
        f"--timeout={per_test_timeout} "
        f"--timeout-method=thread "
        f"-W ignore::DeprecationWarning"
    )


def _build_pip_install_cmd(repo_path: Path) -> str:
    """Build the pip install command prefix for the dev venv."""
    venv_path = get_dev_venv_path(repo_path)
    uv = shutil.which("uv")
    if uv:
        py = get_dev_python(repo_path) or str(venv_path / "bin" / "python")
        return f"uv pip --python {py}"
    else:
        if sys.platform == "win32":
            return str(venv_path / "Scripts" / "pip")
        return str(venv_path / "bin" / "pip")


def _build_api_summary(repo_path: Path, source_files: List[str], max_chars: int = 4000) -> str:
    """Extract public API signatures from top-level definitions in source files.

    Used to inject API context into test-writing batches (final_test_docs, wiring)
    so the sub-agent doesn't guess function signatures.

    Args:
        repo_path: Project repo root path.
        source_files: List of source file paths (relative to repo_path).
        max_chars: Maximum output length before truncation.

    Returns:
        Formatted string of file → class/function signatures.
    """
    # Resolve the project's actual backend so signatures are extracted from
    # the right language. Python keeps a precise AST rendering (bare argument
    # names + return annotation); every other language uses the backend's own
    # one-line ``format_signature`` so non-Python test-writing batches still
    # receive real API context instead of nothing.
    import ast as _ast  # local import; only used for unparse(returns)

    backend = _resolve_codegen_backend(repo_path)
    is_python = backend.name == "python"
    summaries = []
    for filepath in sorted(source_files):
        full_path = repo_path / filepath
        if not full_path.exists() or not backend.is_source_file(filepath):
            continue
        try:
            source = full_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue

        units = backend.list_code_units(source, filepath)
        # Walk top-level declarations only (parent is None) and, for
        # classes, list direct public methods. The prompt format keeps
        # bare argument names plus return annotations.
        top_level = [u for u in units if u.parent is None]
        file_sigs = []
        for unit in top_level:
            if not unit.name or unit.name.startswith('_'):
                continue
            if unit.unit_type == 'class':
                methods = [
                    u.name for u in units
                    if u.unit_type == 'method' and u.parent == unit.name
                    and not u.name.startswith('_')
                ]
                methods_str = ', '.join(methods) if methods else '(dataclass)'
                file_sigs.append(f"  class {unit.name}: {methods_str}")
            elif unit.unit_type == 'function':
                if is_python:
                    node = (unit.extra or {}).get('ast_node')
                    if node is None:
                        continue
                    args = [a.arg for a in node.args.args if a.arg != 'self']
                    ret = _ast.unparse(node.returns) if node.returns else ''
                    ret_str = f" -> {ret}" if ret else ""
                    file_sigs.append(f"  def {unit.name}({', '.join(args)}){ret_str}")
                else:
                    # Non-Python: use the backend's own signature renderer.
                    sig = backend.format_signature(unit) or unit.name
                    file_sigs.append(f"  {sig}")

        if file_sigs:
            summaries.append(f"# {filepath}\n" + "\n".join(file_sigs))

    result = "\n\n".join(summaries)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n# ... (truncated)"
    return result


def _build_dep_graph_context_str(file_path: str, repo_path: Path) -> str:
    """Build a dep_graph context string for prompt injection.

    Loads the RPG and dep_graph, extracts dependency info for the file,
    and formats it as a markdown section.  Returns empty string on any error
    or when no dep_graph is available.
    """
    try:
        import os
        scripts_dir = Path(get_scripts_dir())
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from rpg.service import RPGService

        rpg_path = REPO_RPG_FILE
        if not rpg_path.exists():
            return ""

        svc = RPGService.load(str(rpg_path))
        if svc.rpg.dep_graph is None:
            return ""

        # Resolve file_path to a dep_graph node ID.
        # Task file_path may differ from dep_graph node IDs (e.g.
        # task uses 'src/flask_blog/models/user.py' while dep_graph
        # uses 'models/user.py').  Try multiple candidates.
        G = svc.rpg.dep_graph.G
        candidates = [file_path]
        code_dir = svc.rpg._dep_graph_code_dir
        if code_dir:
            candidates.append(code_dir.rstrip("/") + "/" + file_path)
        # Also try matching by filename suffix
        fname = os.path.basename(file_path)
        for nid in G.nodes:
            if G.nodes[nid].get("type") == "file" and nid.endswith("/" + fname):
                candidates.append(nid)

        resolved = None
        for c in candidates:
            if c in G.nodes:
                resolved = c
                break

        if resolved is None:
            return ""

        ctx = svc.get_dep_context_for_batch([resolved])
        info = ctx.get(resolved, {})
        if not any(info.values()):
            return ""

        parts = ["\n\n## Dependency Graph Context (from AST analysis)\n"]

        if info.get("imports"):
            parts.append("### Imports available:")
            for imp in info["imports"][:20]:
                parts.append(f"- `{imp['module']}` ({imp['name']})")

        if info.get("callees"):
            parts.append("\n### Functions/classes this file calls:")
            for c in info["callees"][:15]:
                parts.append(f"- `{c['name']}` ({c['type']}) — `{c['node_id']}`")

        if info.get("callers"):
            parts.append("\n### Called by (external callers):")
            for c in info["callers"][:15]:
                parts.append(f"- `{c['name']}` ({c['type']}) — `{c['node_id']}`")

        if info.get("inheritance"):
            parts.append("\n### Inheritance:")
            for inh in info["inheritance"][:10]:
                if inh["direction"] == "extends":
                    parts.append(f"- extends `{inh['base']}`")
                else:
                    parts.append(f"- extended by `{inh.get('child', '?')}`")

        return "\n".join(parts) + "\n" if len(parts) > 1 else ""
    except Exception:
        return ""


def build_tdd_prompt(
    batch_state: BatchExecutionState,
    task: PlannedTask,
    repo_path: Path,
    merged_tasks: Optional[List[PlannedTask]] = None,
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the complete TDD prompt for a batch.

    Handles all task_type variations:
      - implementation: full TDD preamble
      - integration_test: test-only variant
      - project_docs: docs-only variant
      - project_requirements / main_entry: project file variant

    Args:
        batch_state: Current batch execution state.
        task: Primary PlannedTask for this batch.
        repo_path: Path to the project repo.
        merged_tasks: If file-merge mode, list of merged tasks.
        dependency_context: Dependency context dict from design stages.

    Returns:
        Complete prompt string ready for LLMClient.generate().
    """
    backend = _resolve_codegen_backend(repo_path)
    venv_python = get_dev_python(repo_path) or "python3"
    import_convention = (
        build_import_convention_snippet(repo_path=repo_path)
        if backend.name == "python"
        else ""
    )

    # --- Project docs: simplest path ---
    if is_project_docs_batch(task):
        return TDD_DOCS_PREAMBLE.format(
            batch_id=batch_state.batch_id,
            code_prompt=batch_state.code_prompt,
        )

    # --- Project files (requirements, main_entry): no TDD loop ---
    if is_project_file_batch(task):
        return TDD_PROJECT_FILE_PREAMBLE.format(
            batch_id=batch_state.batch_id,
            task_type=task.task_type,
            code_prompt=batch_state.code_prompt,
        )

    # --- Implementation / integration_test: full TDD ---
    # For marker file paths like <INTEGRATION_TEST>, don't try to find related tests
    if task.file_path.startswith("<") and task.file_path.endswith(">"):
        test_files = []
    else:
        test_files = find_related_test_files(task.file_path, repo_path)
    pytest_cmd = _build_backend_test_cmd(backend, repo_path, test_files, venv_python)

    # Language-aware entry point reference so testing-batch guidance never
    # plants a Python file name (e.g. "main.py") in a non-Python project.
    entry_point = backend.prompt_hints().entrypoint_example or "the main entry point"

    # For testing batches, allow fixing genuine integration bugs
    if task.task_type in ("integration_test", "final_test_docs"):
        code_instructions = (
            "This is primarily a testing batch. Your main deliverable is tests.\n"
            "However, if your tests reveal **genuine integration bugs** in the "
            "production code, you SHOULD fix them. Examples of legitimate fixes:\n"
            "- A route handler returns a placeholder string instead of calling the real handler\n"
            "- CSS class names in a style module don't match those used in page templates\n"
            "- A module defines a function but its consumer never imports/calls it\n"
            "- Data format mismatch at a module boundary\n\n"
            "Do NOT modify production code solely to make a poorly-written test pass.\n"
            "The test should reflect correct behavior; the code should implement it.\n"
            f"Do NOT create the entry point ({entry_point}) — it will be created in a later task.\n\n"
            "**Testing strategy for efficiency:**\n"
            "- After the first full test-command run, use the native tool's "
            "focused rerun option when available. This saves time.\n"
            "- Only run the full provided test command at the very end to "
            "confirm everything passes.\n"
        )
    else:
        code_instructions = batch_state.code_prompt

    # Format dependency context
    dep_ctx_str = _format_dependency_context(dependency_context) if dependency_context else ""

    # Inject dep_graph context (AST-based dependency info)
    dep_graph_ctx = _build_dep_graph_context_str(task.file_path, repo_path)
    if dep_graph_ctx:
        dep_ctx_str += dep_graph_ctx

    # For test-writing batches (wiring, final_test_docs), inject API summary
    # so sub-agent doesn't guess function signatures
    if task.task_type in ("final_test_docs", "wiring"):
        try:
            all_tasks = load_tasks_from_tasks_json(TASKS_FILE)
            global_state_for_api = load_code_gen_state(STATE_FILE)
            completed_files = list(set(
                t.file_path for t in all_tasks
                if t.task_id in global_state_for_api.completed_task_ids
                and not (t.file_path.startswith("<") and t.file_path.endswith(">"))
            ))
            api_summary = _build_api_summary(repo_path, completed_files)
            if api_summary:
                dep_ctx_str += (
                    "\n### Implemented API Signatures\n"
                    "Use these EXACT signatures when writing tests — do NOT guess.\n"
                    f"```\n{api_summary}\n```\n"
                )
        except Exception as exc:
            logger.warning("Failed to build API summary: %s", exc)

    # For WIRING batches, inject subtree review results to avoid redundant testing
    if task.task_type == "wiring":
        try:
            global_state_for_reviews = load_code_gen_state(STATE_FILE)
            reviews = global_state_for_reviews.subtree_reviews
            verified = [
                st for st, rev in reviews.items()
                if rev.get("status") in ("ALL_COMPLETE", "FIXED")
            ]
            if verified:
                code_instructions += (
                    "\n\nThe following subtrees have been individually reviewed "
                    "and their internal + cross-subtree connections verified:\n"
                    + "\n".join(f"- {s}" for s in sorted(verified))
                    + "\n\nFocus your tests on:\n"
                    "1. Global connections NOT covered by subtree reviews "
                    "(e.g., app initialization, route registration)\n"
                    "2. End-to-end flows that span 3+ subtrees\n"
                    "Do NOT re-test connections already verified above.\n"
                    "Keep tests focused and concise — avoid redundancy.\n"
                )
        except Exception as exc:
            logger.warning("Failed to load subtree reviews for WIRING: %s", exc)

    return TDD_BATCH_PREAMBLE.format(
        test_instructions=batch_state.test_prompt,
        code_instructions=code_instructions,
        pytest_cmd=pytest_cmd,
        test_tool_name=backend.prompt_hints().test_framework_name,
        max_iterations=MAX_ITERATIONS,
        batch_id=batch_state.batch_id,
        dependency_manifest_instruction=_dependency_manifest_instruction(backend),
        dependency_install_capability=_dependency_install_capability(backend, repo_path),
        dependency_management=_dependency_management_text(backend, repo_path),
        test_timeout_rule=_test_timeout_rule(backend),
        summary_fallback_rule=_summary_fallback_rule(backend, pytest_cmd),
        import_convention=import_convention,
        language_context=_build_language_context(backend, pytest_cmd),
        dependency_context=dep_ctx_str,
        file_path=task.file_path,
        units=", ".join(task.units_key),
        task_type=task.task_type,
    )


def build_resume_prompt(
    original_prompt: str,
    attempt_number: int,
    failure_reason: str,
    last_test_output: str,
    *,
    sub_agent_claimed_pass: bool = False,
    agent_pytest_summary: Optional[str] = None,
) -> str:
    """Build a resume prompt for auto-retry after failure.

    Args:
        original_prompt: The full TDD prompt from the first attempt.
        attempt_number: Which attempt this is (2 for auto-retry).
        failure_reason: One-line reason from BATCH_RESULT: FAIL,
            or the post-verify mismatch reason if the sub-agent
            self-reported PASS but verification failed.
        last_test_output: Test-command output from post-verification.
        sub_agent_claimed_pass: True if the previous attempt reported
            ``BATCH_RESULT: PASS`` but post-verify rejected it; this
            triggers an extra warning section in the prompt so the
            sub-agent does not repeat the false-positive pattern.
        agent_pytest_summary: The ``PYTEST_SUMMARY:`` line the
            previous attempt produced (verbatim).  Included for
            comparison when ``sub_agent_claimed_pass`` is True.

    Returns:
        Resume prompt string.
    """
    # Smart truncation: keep the first 20 lines (pytest header,
    # collected count, first failure header) and last 50 lines
    # (FAILED/ERROR detail + summary).
    last_test_output = truncate_test_output(last_test_output, head=20, tail=50)

    if sub_agent_claimed_pass:
        agent_summary_repr = (
            f"`{agent_pytest_summary}`"
            if agent_pytest_summary
            else "(missing — you did not include the PYTEST_SUMMARY line)"
        )
        post_verify_section = (
            "\n\n## ⚠ False-positive PASS detected\n"
            "Your previous attempt ended with `BATCH_RESULT: PASS` and the\n"
            f"PYTEST_SUMMARY line {agent_summary_repr}, but the runner's\n"
            "independent test-command re-run reported the failure shown below.\n"
            "Possible causes you must investigate:\n"
            "* You did not actually run the exact test command before declaring PASS.\n"
            "* You ran a different command or selector that\n"
            "  excluded the failing tests.\n"
            "* You modified or deleted tests instead of fixing production code.\n"
            "* Your local changes were not committed before the runner verified.\n"
            "**Do not report PASS again unless the PYTEST_SUMMARY line literally\n"
            "shows zero failures and zero errors.**\n"
        )
    else:
        post_verify_section = ""

    return TDD_RESUME_PREAMBLE.format(
        attempt_number=attempt_number,
        failure_reason=failure_reason,
        last_test_output=last_test_output,
        post_verify_section=post_verify_section,
    ) + "\n---\n\n" + original_prompt
