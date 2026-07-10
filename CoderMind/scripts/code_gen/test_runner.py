#!/usr/bin/env python3
"""Test Runner Utilities for CoderMind Code Generation.

Provides utilities for:
- Finding test files related to source changes
- Building pytest commands
- Executing tests and parsing results
- Determining test success/failure
"""

import os
import re
import subprocess
import sys
import shutil
import importlib.util
import logging
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass
from .test_output_parser import TestOutputAnalysis, _parse_stats, _SUMMARY_RE
from .test_output_parser import analyze_test_output
from common.llm_client import LLMClient
from common.llm_client import _IS_WINDOWS, _kill_process_tree, _set_pdeathsig
import json as _json
from common.import_normalizer import normalize_files
from common.paths import FEATURE_SPEC_FILE, REPO_RPG_FILE
from decoder_lang import (
    EnvHandle,
    LanguageBackend,
    ToolchainUnavailable,
    get_backend,
    resolve_decoder_language,
    scan_repo_source_files,
)




# ============================================================================
# Test File Detection Patterns
# ============================================================================

DEFAULT_TEST_PATTERNS: Tuple[str, ...] = (
    r"(^|/)(tests|test|testing)/.*\.py$",
    r"(^|/)test_.*\.py$",
    r"(^|/).*_test\.py$",
)

DEFAULT_PYTHON_PATTERN = r".*\.py$"


# ============================================================================
# Test Result Data Classes
# ============================================================================

@dataclass
class TestResult:
    """Result of test execution."""
    success: bool
    return_code: int
    output: str
    test_files: List[str]
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "return_code": self.return_code,
            "output": self.output,
            "test_files": self.test_files,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration": self.duration,
        }


# ============================================================================
# Test File Detection
# ============================================================================

def is_test_file(filepath: str, patterns: Tuple[str, ...] = DEFAULT_TEST_PATTERNS) -> bool:
    """Check if a file path matches test file patterns."""
    compiled = [re.compile(p) for p in patterns]
    return any(p.search(filepath) for p in compiled)


def find_test_files_in_directory(
    directory: Path,
    patterns: Tuple[str, ...] = DEFAULT_TEST_PATTERNS
) -> List[str]:
    """Find all test files in a directory."""
    test_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, directory)
                if is_test_file(rel_path, patterns):
                    test_files.append(rel_path)
    
    return sorted(test_files)


def _existing_relative_paths(repo_root: Path, candidates: List[Path]) -> List[str]:
    """Return existing candidate paths relative to ``repo_root`` without duplicates."""
    seen: Set[str] = set()
    found: List[str] = []
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        rel = str(candidate.relative_to(repo_root))
        if rel not in seen:
            seen.add(rel)
            found.append(rel)
    return found


def _find_related_python_tests(source_path: Path, repo_root: Path) -> List[str]:
    """Find Python tests related to ``source_path`` using legacy heuristics."""
    # --- Build canonical signature from source path ---
    # Strip known prefixes: "src", "lib"
    skip_prefixes = {'src', 'lib'}
    parts = list(source_path.parts)

    # Find where meaningful path starts (after src/lib + package root)
    start_idx = 0
    if parts and parts[0] in skip_prefixes:
        start_idx = 1
        # Also skip the project-package root (e.g., "flask_blog")
        # because test file names typically don't include it
        if len(parts) > 1:
            start_idx = 2

    # Collect directory parts (excluding the filename) + stem
    relevant = []
    for part in parts[start_idx:-1]:
        if not part.startswith('.'):
            relevant.append(part)
    relevant.append(source_path.stem)
    src_signature = '_'.join(relevant)

    # --- Search test directories for matching files ---
    related_tests = []
    test_dirs = ['tests', 'test', 'testing']

    for test_dir in test_dirs:
        test_path = repo_root / test_dir
        if not test_path.exists():
            continue
        for test_file in test_path.rglob("test_*.py"):
            test_sig = test_file.stem.replace('test_', '', 1)
            if test_sig == src_signature:
                related_tests.append(str(test_file.relative_to(repo_root)))

    # Fallback: if signature matching found nothing, try simple stem match
    if not related_tests:
        module_name = source_path.stem
        for test_dir in test_dirs:
            test_path = repo_root / test_dir
            if not test_path.exists():
                continue
            test_file = test_path / f"test_{module_name}.py"
            if test_file.exists():
                related_tests.append(str(test_file.relative_to(repo_root)))
            test_file = test_path / f"{module_name}_test.py"
            if test_file.exists():
                related_tests.append(str(test_file.relative_to(repo_root)))

    return related_tests


def _find_related_non_python_tests(source_path: Path, repo_root: Path) -> List[str]:
    """Find likely related non-Python test files using common naming conventions.

    These are discovery hints only.  Non-Python post-verification still runs the
    backend's project-level test command unless a backend-specific selector layer
    explicitly supports scoped execution.
    """
    suffix = source_path.suffix.lower()
    stem = source_path.stem
    parent = repo_root / source_path.parent
    tests_dir = repo_root / "tests"

    if suffix == ".go":
        return _existing_relative_paths(repo_root, [parent / f"{stem}_test.go"])

    if suffix == ".rs":
        return _existing_relative_paths(repo_root, [
            parent / f"{stem}_test.rs",
            tests_dir / f"{stem}.rs",
            tests_dir / f"{stem}_test.rs",
        ])

    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        variants = [
            f"{stem}.test{suffix}",
            f"{stem}.spec{suffix}",
        ]
        return _existing_relative_paths(repo_root, [
            *(parent / name for name in variants),
            *(parent / "__tests__" / name for name in variants),
            *(tests_dir / name for name in variants),
            *(tests_dir / source_path.parent.name / name for name in variants),
        ])

    c_suffixes = {".c": [".c"], ".cpp": [".cpp", ".cc", ".cxx"], ".cc": [".cc", ".cpp", ".cxx"], ".cxx": [".cxx", ".cpp", ".cc"]}
    if suffix in c_suffixes:
        names = []
        for ext in c_suffixes[suffix]:
            names.extend([f"test_{stem}{ext}", f"{stem}_test{ext}"])
        return _existing_relative_paths(repo_root, [
            *(parent / name for name in names),
            *(tests_dir / name for name in names),
        ])

    return []


def find_related_test_files(
    source_file: str,
    repo_root: Path
) -> List[str]:
    """Find test files likely related to a source file.

    Python keeps the legacy path-signature matching. Other languages use
    conservative file-name conventions (Go ``*_test.go``, JS/TS
    ``*.test.*``/``*.spec.*``, C/C++ ``test_*``/``*_test`` and Rust common
    integration-test names). These results are discovery hints; backend-specific
    test execution decides whether scoped execution is safe.

    Args:
        source_file: Path to the source file (relative to repo root)
        repo_root: Repository root path

    Returns:
        List of related test file paths (relative to repo root)
    """
    source_path = Path(source_file)
    if source_path.suffix == '.py':
        return _find_related_python_tests(source_path, repo_root)
    return _find_related_non_python_tests(source_path, repo_root)


def extract_files_from_diff(diff_content: str) -> Tuple[List[str], List[str]]:
    """Extract file paths from a git diff.
    
    Returns:
        Tuple of (source_files, test_files)
    """
    source_files = []
    test_files = []
    
    # Pattern to match file paths in diff
    file_pattern = re.compile(r'^diff --git a/(.+) b/(.+)$', re.MULTILINE)
    
    for match in file_pattern.finditer(diff_content):
        filepath = match.group(2)
        
        if not filepath.endswith('.py'):
            continue
        
        if filepath == '/dev/null':
            continue
        
        if is_test_file(filepath):
            test_files.append(filepath)
        else:
            source_files.append(filepath)
    
    return source_files, test_files


def build_pytest_command(
    test_files: List[str],
    repo_root: Optional[Path] = None,
    verbose: bool = True,
    extra_args: Optional[List[str]] = None,
    python_exe: Optional[str] = None
) -> List[str]:
    """Build a pytest command for running specific test files.
    
    Args:
        test_files: List of test file paths
        repo_root: Repository root (for relative paths)
        verbose: Include verbose flag
        extra_args: Additional pytest arguments
        python_exe: Python executable to use (default: "python3")
        
    Returns:
        Command as list of strings
    """
    py = python_exe or "python3"
    cmd = [py, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add extra args
    if extra_args:
        cmd.extend(extra_args)
    
    # Add test files
    for test_file in test_files:
        if repo_root:
            full_path = repo_root / test_file
            if full_path.exists():
                cmd.append(str(test_file))
        else:
            cmd.append(test_file)
    
    return cmd


def build_comprehensive_test_command(
    diff_content: str,
    repo_root: Path,
    extra_args: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """Build a pytest command that covers all relevant tests for a diff.
    
    This includes:
    1. Test files directly modified in the diff
    2. Test files related to modified source files
    
    Args:
        diff_content: Git diff content
        repo_root: Repository root path
        extra_args: Additional pytest arguments
        
    Returns:
        Tuple of (command, analysis_info)
    """
    source_files, diff_test_files = extract_files_from_diff(diff_content)
    
    # Find related test files for modified source files
    related_test_files = []
    for source_file in source_files:
        related = find_related_test_files(source_file, repo_root)
        related_test_files.extend(related)
    
    # Combine all test files
    all_test_files = list(set(diff_test_files + related_test_files))
    
    # If no specific test files, run all tests
    if not all_test_files:
        cmd = ["python3", "-m", "pytest"]
        if extra_args:
            cmd.extend(extra_args)
    else:
        cmd = build_pytest_command(all_test_files, repo_root, extra_args=extra_args)
    
    analysis_info = {
        "patch_source_files": source_files,
        "patch_test_files": diff_test_files,
        "related_test_files": related_test_files,
        "all_test_files": all_test_files,
    }
    
    return cmd, analysis_info


# ============================================================================
# Test Execution
# ============================================================================

def run_pytest(
    repo_root: Path,
    test_files: Optional[List[str]] = None,
    timeout: int = 300,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
) -> TestResult:
    """Run pytest and return results.
    
    If a dev venv exists at DEV_VENV_DIR, its python is used automatically.
    
    Args:
        repo_root: Repository root path
        test_files: Specific test files to run (None = all)
        timeout: Timeout in seconds
        extra_args: Additional pytest arguments
        env: Environment variables
        
    Returns:
        TestResult with execution details
    """
    # Use dev venv python if available
    python_exe = get_dev_python(repo_root)
    
    # Build command
    if test_files:
        cmd = build_pytest_command(test_files, repo_root, extra_args=extra_args, python_exe=python_exe)
    else:
        cmd = [python_exe or "python3", "-m", "pytest", "-v"]
        if extra_args:
            cmd.extend(extra_args)
    
    # Setup environment
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(repo_root)
    if env:
        run_env.update(env)
    
    popen_kwargs: Dict[str, Any] = dict(
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
    )
    if _IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True   # own process group → killpg kills pytest + children
        popen_kwargs["preexec_fn"] = _set_pdeathsig # PR_SET_PDEATHSIG: killed even when parent SIGKILL'd

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
        except BaseException:
            # Kill the entire pytest process tree (covers forked workers, etc.)
            _kill_process_tree(proc)
            proc.wait()
            raise

        output = stdout_data
        if stderr_data:
            output += "\n\nSTDERR:\n" + stderr_data

        # Parse results
        success = proc.returncode == 0
        stats = parse_pytest_output(output)

        return TestResult(
            success=success,
            return_code=proc.returncode,
            output=output,
            test_files=test_files or [],
            passed=stats.get("passed", 0),
            failed=stats.get("failed", 0),
            errors=stats.get("errors", 0),
            skipped=stats.get("skipped", 0),
            duration=stats.get("duration", 0.0),
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            success=False,
            return_code=-1,
            output=f"Test execution timed out after {timeout} seconds",
            test_files=test_files or [],
        )
    except Exception as e:
        return TestResult(
            success=False,
            return_code=-1,
            output=f"Test execution failed: {str(e)}",
            test_files=test_files or [],
        )


def _load_json_if_exists(path: Path) -> Any:
    """Load JSON from ``path`` or return None when unavailable."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return _json.load(file)
    except (OSError, _json.JSONDecodeError):
        return None


def resolve_test_backend(
    valid_files: Optional[List[str]] = None,
    repo_path: Optional[Path] = None,
) -> LanguageBackend:
    """Resolve the backend that should run codegen verification tests.

    Language is resolved through :func:`resolve_decoder_language`'s tier
    chain (feature_spec meta -> rpg meta -> dominant language of the
    supplied files -> python default). When the caller has no scoped
    ``valid_files`` (e.g. the final-test / global-review / env-setup
    stages operate over the whole repo), pass ``repo_path`` so the
    language can still be inferred from the actual on-disk sources rather
    than silently defaulting to python for a non-python project.
    """
    feature_spec = _load_json_if_exists(FEATURE_SPEC_FILE)
    rpg_obj = _load_json_if_exists(REPO_RPG_FILE)
    if not valid_files and repo_path is not None:
        valid_files = scan_repo_source_files(repo_path) or None
    language = resolve_decoder_language(
        feature_spec=feature_spec,
        rpg_obj=rpg_obj,
        valid_files=valid_files,
    )
    return get_backend(language)


def run_project_tests(
    repo_root: Path,
    test_files: Optional[List[str]] = None,
    timeout: int = 300,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    backend: Optional[LanguageBackend] = None,
) -> TestResult:
    """Run the target language's native project test command."""
    selected_backend = backend or resolve_test_backend(
        valid_files=test_files, repo_path=repo_root
    )
    if selected_backend.name == "python":
        return run_pytest(
            repo_root,
            test_files=test_files,
            timeout=timeout,
            extra_args=extra_args,
            env=env,
        )

    try:
        env_handle = selected_backend.detect_env(repo_root) or EnvHandle(
            project_root=repo_root.resolve(),
        )
        # Settle the build state before testing (no-op for most backends;
        # C/C++ reconfigure cmake so ctest sees the current test set rather
        # than a stale one left from a mid-edit configure).
        prepare = getattr(selected_backend, "prepare_test_env", None)
        if callable(prepare):
            try:
                prepare(env_handle)
            except Exception as exc:  # noqa: BLE001 - best-effort prep
                _logger.debug("prepare_test_env failed (non-fatal): %s", exc)
        cmd = selected_backend.test_command(env_handle)
    except (ToolchainUnavailable, NotImplementedError, OSError) as exc:
        return TestResult(
            success=False,
            return_code=-1,
            output=f"{selected_backend.display_name} test command unavailable: {exc}",
            test_files=test_files or [],
        )

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    popen_kwargs: Dict[str, Any] = dict(
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
    )
    if _IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
        popen_kwargs["preexec_fn"] = _set_pdeathsig

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
        except BaseException:
            _kill_process_tree(proc)
            proc.wait()
            raise

        output = stdout_data
        if stderr_data:
            output += "\n\nSTDERR:\n" + stderr_data
        parsed = selected_backend.parse_test_output(output, proc.returncode)
        return TestResult(
            success=parsed.status == "passed",
            return_code=proc.returncode,
            output=output,
            test_files=test_files or [],
            passed=parsed.passed_count,
            failed=parsed.failed_count,
            errors=parsed.error_count,
            skipped=parsed.skipped_count,
            duration=parsed.duration_sec,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            success=False,
            return_code=-1,
            output=f"Test execution timed out after {timeout} seconds",
            test_files=test_files or [],
        )
    except Exception as exc:
        return TestResult(
            success=False,
            return_code=-1,
            output=f"Test execution failed: {exc}",
            test_files=test_files or [],
        )


def parse_pytest_output(output: str) -> Dict[str, Any]:
    """Parse pytest output to extract statistics.
    
    Returns dict with: passed, failed, errors, skipped, duration
    
    .. note:: The canonical implementation now lives in
       ``code_gen.test_output_parser._parse_stats``.  This wrapper is kept
       for backward compatibility.
    """
    result = TestOutputAnalysis()
    _parse_stats(output, result)
    return {
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "skipped": result.skipped,
        "duration": result.duration,
    }


def is_test_successful(return_code: int, test_output: str) -> bool:
    """Determine if tests passed based on return code and output.
    
    Args:
        return_code: pytest return code
        test_output: pytest output text
        
    Returns:
        True if tests passed
    """
    # Return code 0 means success
    if return_code == 0:
        return True
    
    # Return code 5 means no tests collected (not a failure)
    if return_code == 5:
        # Check if this is expected
        if "no tests ran" in test_output.lower():
            return True
    
    return False


# ============================================================================
# Failure Type Detection (Simple Heuristics)
# ============================================================================

def detect_failure_type_simple(test_output: str) -> str:
    """Detect failure type using simple heuristics (no LLM).
    
    Returns: "TEST_ERROR", "CODE_ERROR", "ENV_ERROR", or "UNKNOWN_ERROR"
    
    .. note:: The canonical implementation now lives in
       ``code_gen.test_output_parser._classify``.  This wrapper is kept
       for backward compatibility and returns upper-case values.
    """
    analysis = analyze_test_output(test_output)
    return analysis.failure_type or "UNKNOWN_ERROR"


# ============================================================================
# Dev Virtual Environment Management
# ============================================================================

# Bare directory name kept locally for backward compatibility with callers
# that build relative paths.  ``common.paths.DEV_VENV_NAME`` is the
# canonical source of truth; the import re-exports it under the original
# name so existing ``from code_gen.test_runner import DEV_VENV_DIR``
# imports keep working (only call sites today are
# ``code_gen.test_runner``-internal anyway).
from common.paths import DEV_VENV_NAME as DEV_VENV_DIR

_logger = logging.getLogger(__name__)

# Stdlib modules — used to filter out standard library imports during scanning.
_STDLIB_TOP_LEVEL = frozenset({
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
    'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
    'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
    'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
    'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
    'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
    'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
    'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
    'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
    'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
    'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
    'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
    'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
    'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil',
    'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
    'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr',
    'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
    'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
    'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
    'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
    'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios',
    'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter',
    'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty',
    'turtle', 'turtledemo', 'types', 'typing', 'unicodedata', 'unittest',
    'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
    'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml',
    'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo',
    '_thread', '__future__', 'typing_extensions',
})

# Common import-name → PyPI-package-name mappings
_IMPORT_TO_PACKAGE: Dict[str, str] = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv',
    'jwt': 'PyJWT',
    'serial': 'pyserial',
    'usb': 'pyusb',
    'git': 'GitPython',
    'skimage': 'scikit-image',
    'attr': 'attrs',
    'wx': 'wxPython',
}


def get_dev_venv_path(repo_root: Path) -> Path:
    """Return the path to the dev venv directory."""
    return repo_root / DEV_VENV_DIR


def get_dev_python(repo_root: Path) -> Optional[str]:
    """Return the dev venv python executable path, or None if venv doesn't exist."""
    venv_path = get_dev_venv_path(repo_root)
    if sys.platform == "win32":
        py = venv_path / "Scripts" / "python.exe"
    else:
        py = venv_path / "bin" / "python"
    if py.exists():
        return str(py)
    return None


def ensure_dev_venv(repo_root: Path) -> Tuple[bool, Path]:
    """Lazily create the dev venv if it doesn't exist.

    Installs pytest into it on creation.
    
    Returns:
        Tuple of (created_new, venv_path)
    """
    venv_path = get_dev_venv_path(repo_root)
    py = get_dev_python(repo_root)
    if py is not None:
        return False, venv_path

    _logger.info("Creating dev venv at %s", venv_path)
    uv = shutil.which("uv")
    try:
        if uv:
            subprocess.run(
                ["uv", "venv", str(venv_path)],
                cwd=repo_root, capture_output=True, text=True, timeout=60,
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                cwd=repo_root, capture_output=True, text=True, timeout=120,
                check=True,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        _logger.error("Failed to create dev venv: %s", exc)
        return False, venv_path

    # Install pytest into the new venv
    install_packages_into_venv(["pytest", "pytest-timeout"], repo_root)
    return True, venv_path


def _build_pip_cmd(
    pip_names: List[str],
    repo_root: Path,
) -> List[str]:
    """Build a complete pip/uv install command for the given packages."""
    venv_path = get_dev_venv_path(repo_root)
    uv = shutil.which("uv")
    if uv:
        py_exe = get_dev_python(repo_root) or str(venv_path / "bin" / "python")
        return ["uv", "pip", "install"] + pip_names + ["--python", py_exe]
    else:
        if sys.platform == "win32":
            pip_exe = str(venv_path / "Scripts" / "pip")
        else:
            pip_exe = str(venv_path / "bin" / "pip")
        return [pip_exe, "install"] + pip_names


def _pip_install_single(pkg: str, repo_root: Path) -> bool:
    """Try to pip-install a single package. Returns True on success."""
    try:
        cmd = _build_pip_cmd([pkg], repo_root)
        result = subprocess.run(
            cmd, cwd=repo_root,
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_pip_names(
    import_names: List[str],
    repo_root: Path,
) -> Dict[str, str]:
    """Resolve Python import names to pip-installable package names.

    Strategy (layered):
      1. Static mapping table (_IMPORT_TO_PACKAGE) for known mismatches.
      2. For unmapped names, batch-ask LLM to resolve import→package.
      3. Fallback: use import name as-is (works for ~80% of packages).

    Returns:
        Dict mapping import_name → pip_package_name.
    """
    resolved: Dict[str, str] = {}
    needs_llm: List[str] = []

    for name in import_names:
        if name in _IMPORT_TO_PACKAGE:
            resolved[name] = _IMPORT_TO_PACKAGE[name]
        else:
            needs_llm.append(name)

    if not needs_llm:
        return resolved

    # Try LLM resolution for unknown mappings
    llm_resolved = _resolve_packages_via_llm(needs_llm)
    for name in needs_llm:
        pip_name = llm_resolved.get(name)
        if pip_name and pip_name != "UNKNOWN":
            resolved[name] = pip_name
        else:
            # Fallback: use import name directly
            resolved[name] = name

    return resolved


def _resolve_packages_via_llm(import_names: List[str]) -> Dict[str, str]:
    """Ask LLM to resolve import names to PyPI package names.

    Uses a single, cheap LLM call (~200 tokens). Falls back to empty dict
    on any error so callers can use the import-name-as-is fallback.

    Returns:
        Dict of {import_name: pip_package_name} for successfully resolved names.
    """
    if not import_names:
        return {}

    try:

        prompt = (
            "Map these Python import names to their PyPI package names.\n"
            "Return ONLY a JSON object: {\"import_name\": \"pip_package_name\"}\n"
            "Rules:\n"
            "- If the import name equals the pip package name, repeat it "
            "(e.g. \"numpy\": \"numpy\").\n"
            "- If you don't know, use \"UNKNOWN\".\n"
            "- Do NOT include any explanation, only the JSON object.\n\n"
            f"Import names: {_json.dumps(import_names)}"
        )

        client = LLMClient()
        response = client.generate(prompt, purpose="resolve_pip_names", timeout=60)
        parsed = client.parse_json_block(response)

        if parsed and isinstance(parsed, dict):
            # Cache successful resolutions for this session
            for k, v in parsed.items():
                if v and v != "UNKNOWN" and k not in _IMPORT_TO_PACKAGE:
                    _IMPORT_TO_PACKAGE[k] = v
            return parsed
    except Exception as exc:
        _logger.warning("LLM package resolution failed: %s", exc)

    return {}


def install_packages_into_venv(
    packages: List[str],
    repo_root: Path,
) -> Tuple[bool, List[str]]:
    """Install packages into the dev venv.

    Resolves import names to pip package names (via mapping table + LLM),
    tries bulk install first, then falls back to per-package install for
    any failures.

    Args:
        packages: List of import names to install
        repo_root: Repository root

    Returns:
        Tuple of (any_succeeded, list of packages actually installed)
    """
    if not packages:
        return True, []

    # Resolve import names → pip package names
    name_map = resolve_pip_names(packages, repo_root)
    pip_names = [name_map.get(p, p) for p in packages]
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_pip_names: List[str] = []
    for n in pip_names:
        if n not in seen:
            seen.add(n)
            unique_pip_names.append(n)
    pip_names = unique_pip_names

    # Try bulk install first
    try:
        cmd = _build_pip_cmd(pip_names, repo_root)
        result = subprocess.run(
            cmd, cwd=repo_root,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return True, pip_names
    except (subprocess.TimeoutExpired, Exception) as exc:
        _logger.warning("Bulk install error: %s", exc)

    # Bulk failed — install individually, collect successes
    _logger.info("Bulk install failed, retrying packages individually...")
    installed: List[str] = []
    for pkg in pip_names:
        if _pip_install_single(pkg, repo_root):
            installed.append(pkg)
        else:
            _logger.warning("Failed to install package: %s", pkg)

    return (len(installed) > 0, installed)


# ---------------------------------------------------------------------------
# Import prefix normalization
# ---------------------------------------------------------------------------

def fix_import_prefixes(repo_root: Path) -> List[str]:
    """Fix inconsistent import prefixes in source files.

    Delegates to :func:`common.import_normalizer.normalize_files`.
    Kept here for backward compatibility with existing callers.

    Returns:
        List of file paths (relative to *repo_root*) that were modified.
    """
    return normalize_files(repo_root)


def scan_missing_imports(repo_root: Path) -> List[str]:
    """Scan all Python files under src/ and tests/ for imports that cannot be resolved in the environment that will run tests.
    
    When a dev venv exists, the check runs inside the venv python so the
    result matches what pytest will actually see.
    
    Returns:
        List of top-level module names that are missing.
    """
    src_dir = repo_root / "src"
    tests_dir = repo_root / "tests"
    
    # Collect project top-level package names (to skip internal imports).
    # Include ALL subdirectories under src/ (not just those with __init__.py)
    # to handle namespace packages correctly.
    project_modules: Set[str] = set()
    for d in [src_dir, tests_dir]:
        if d.is_dir():
            project_modules.add(d.name)
            for child in d.iterdir():
                if child.is_dir() and not child.name.startswith('.'):
                    project_modules.add(child.name)

    # Collect all external imports from source files through the
    # backend import scanner. ``LPDependency.extra["module"]`` carries
    # the dotted module name for both ``import`` and ``from`` statements;
    # use the top-level segment for dependency installation.
    from decoder_lang import get_backend
    backend = get_backend("python")

    external_imports: Set[str] = set()
    scan_dirs = [d for d in [src_dir, tests_dir] if d.is_dir()]

    for scan_dir in scan_dirs:
        for py_file in scan_dir.rglob("*.py"):
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                continue
            for dep in backend.list_imports(source, str(py_file)):
                extra = dep.extra or {}
                module = extra.get("module") or ""
                if not module or module.startswith("."):
                    # Skip relative imports; they refer to project-local
                    # modules rather than installable third-party packages.
                    continue
                mod_name = module.split(".")[0]
                if mod_name in _STDLIB_TOP_LEVEL or mod_name in project_modules:
                    continue
                external_imports.add(mod_name)

    if not external_imports:
        return []

    # Determine which python to check against — dev venv if it exists,
    # otherwise the system python that will run tests.
    py_exe = get_dev_python(repo_root) or sys.executable

    # Check importability in the target python via a single subprocess call
    # to avoid per-module overhead.
    check_script = (
        "import importlib.util, json, sys\n"
        "modules = json.loads(sys.argv[1])\n"
        "missing = [m for m in modules if importlib.util.find_spec(m) is None]\n"
        "print(json.dumps(missing))\n"
    )
    try:
        result = subprocess.run(
            [py_exe, "-c", check_script, _json.dumps(sorted(external_imports))],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(repo_root)},
        )
        if result.returncode == 0 and result.stdout.strip():
            return _json.loads(result.stdout.strip())
    except Exception as exc:
        _logger.warning("Subprocess import check failed, falling back: %s", exc)

    # Fallback: check in current process (may be inaccurate if dev venv exists)
    missing: Set[str] = set()
    for mod_name in external_imports:
        if importlib.util.find_spec(mod_name) is None:
            missing.add(mod_name)
    return sorted(missing)


def ensure_deps_installed(repo_root: Path) -> Tuple[bool, List[str]]:
    """Ensure dev venv exists and all detectable third-party deps are installed.

    This is the single entry point for proactive dependency management.
    Call before running pytest for the first time in a batch.

    Steps:
      1. Create dev venv if it doesn't exist (+ install pytest).
      2. AST-scan src/ and tests/ for third-party imports.
      3. Check which imports are missing in the venv.
      4. Resolve import names → pip package names (mapping table + LLM).
      5. Install missing packages (bulk, with per-package retry on failure).

    Returns:
        (any_installed, list_of_installed_pip_names)
    """
    ensure_dev_venv(repo_root)
    missing = scan_missing_imports(repo_root)
    if not missing:
        return False, []
    _logger.info("Detected missing imports: %s", missing)
    ok, installed = install_packages_into_venv(missing, repo_root)
    if installed:
        _logger.info("Auto-installed packages: %s", installed)
    return ok, installed
