import re
import ast
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict


DEFAULT_TEST_PATTERNS: Tuple[str, ...] = (
    r"(^|/)(tests|test|testing)/.*\.py$",
    r"(^|/)test_.*\.py$",
    r"(^|/).*_test\.py$",
)

DEFAULT_PYTHON_PATTERN = r".*\.py$"

def _strip_ab_prefix(p: str) -> str:
    p = p.strip()
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p

def extract_test_files_from_patch(
    patch_content: str,
    *,
    test_path_patterns: Tuple[str, ...] = DEFAULT_TEST_PATTERNS,
    include_deleted: bool = False,
) -> List[str]:
    """
    Extract repo-relative test file paths from a unified diff.

    Handles:
    - diff --git a/x b/y
    - +++ b/x
    - rename to x
    - ignores /dev/null (unless include_deleted=True and old path exists)
    """
    patterns = [re.compile(p) for p in test_path_patterns]
    test_files: Set[str] = set()

    current_b_path: Optional[str] = None
    current_a_path: Optional[str] = None

    def _maybe_add(path: Optional[str]) -> None:
        if not path:
            return
        path = path.strip()
        if path == "/dev/null":
            return
        path = _strip_ab_prefix(path)
        # normalize any leading "./"
        if path.startswith("./"):
            path = path[2:]
        if any(p.search(path) for p in patterns):
            test_files.add(path)

    for raw in patch_content.splitlines():
        line = raw.rstrip("\n")

        # Start of a new file diff section
        if line.startswith("diff --git "):
            # Example: diff --git a/foo.py b/foo.py
            # Safer parse: take last two tokens as a_path b_path
            parts = line.split()
            if len(parts) >= 4:
                current_a_path = parts[-2]
                current_b_path = parts[-1]
                # prefer b/ path
                _maybe_add(current_b_path)
            continue

        # Git rename/copy headers (within a diff section)
        if line.startswith("rename to "):
            current_b_path = line[len("rename to "):].strip()
            _maybe_add(current_b_path)
            continue

        if line.startswith("copy to "):
            current_b_path = line[len("copy to "):].strip()
            _maybe_add(current_b_path)
            continue

        # Unified diff file markers
        if line.startswith("+++ "):
            # +++ b/path or +++ /dev/null
            path = line[len("+++ "):].strip()
            # prefer explicit marker
            if path != "/dev/null":
                current_b_path = path
            else:
                current_b_path = "/dev/null"
            _maybe_add(current_b_path)
            continue

        if include_deleted and line.startswith("--- "):
            # For deletions, old path might be useful if you want to run deleted tests (usually no)
            path = line[len("--- "):].strip()
            current_a_path = path
            # usually not added; only if include_deleted=True and you want old test file
            if path != "/dev/null":
                _maybe_add(current_a_path)
            continue

        # Also catch "+++ b/..." lines sometimes appear without diff --git in truncated patches
        # already covered above.

    return sorted(test_files)


def build_pytest_command_from_test_patch(
    test_patch_content: str,
    *,
    python: str = "python",
    extra_args: Optional[Iterable[str]] = None,
    fallback: str = "all",  # "all" | "empty"
    base_cmd: Optional[List[str]] = None,
) -> List[str]:
    """
    Build pytest command (argv list) from test patch content.

    - If test files found: run pytest on those files.
    - If none found:
        - fallback="all"   -> run full suite: python -m pytest
        - fallback="empty" -> return [] (caller decides)
    """
    extra_args = list(extra_args or [])
    cmd = (base_cmd[:] if base_cmd else [python, "-m", "pytest"])

    test_files = extract_test_files_from_patch(test_patch_content)

    if test_files:
        cmd.extend(test_files)
    else:
        if fallback == "empty":
            return []
        # fallback == "all": leave cmd as-is (run full suite)

    cmd.extend(extra_args)
    return cmd


def is_test_successful(return_code: int, test_output: Union[str, bytes, None]) -> bool:
    """
    Determine if a test run was successful based on return code and output.
    
    Args:
        return_code: The exit code from the test execution
        test_output: The stdout/stderr output from the test execution (can be None)
        
    Returns:
        bool: True if tests passed, False otherwise
    """
    # Handle None or empty output
    if test_output is None:
        # Only rely on return code if no output
        return return_code == 0
    
    # Convert bytes to string if necessary
    if isinstance(test_output, bytes):
        try:
            test_output = test_output.decode('utf-8', errors='ignore')
        except Exception:
            # If decoding fails, treat as empty
            test_output = ""
    
    # Ensure test_output is string
    test_output = str(test_output)
    
    # Check return code first - non-zero typically means failure
    # But some test runners may return 0 even on failures, so we check output too
    if return_code != 0:
        # Double-check for false positives - some tools return non-zero on warnings
        warning_only_patterns = [
            r"warning[s]?\s*:?\s*\d+",
            r"\d+\s*warning[s]?",
            r"DeprecationWarning",
            r"PendingDeprecationWarning",
            r"FutureWarning",
        ]
        
        # If only warnings and no failures, might still be successful
        has_only_warnings = any(re.search(p, test_output, re.IGNORECASE) for p in warning_only_patterns)
        has_failures = any(re.search(p, test_output, re.IGNORECASE) for p in [
            r"FAILED", r"ERROR", r"FAIL:", r"\d+\s*failed", r"AssertionError", r"Exception:"
        ])
        
        if has_only_warnings and not has_failures:
            # Continue checking, don't immediately return False
            pass
        else:
            return False
    
    # Additional failure patterns for various test frameworks
    failure_patterns = [
        # pytest patterns - specific test output indicators
        r"FAILED\s+\[",  # pytest-style "FAILED [100%]"
        r"::.*FAILED",   # pytest-style "test::path FAILED"
        r"={3,}\s*FAILURES\s*={3,}",  # pytest-style "======= FAILURES ======="
        r"={3,}\s*ERRORS\s*={3,}",    # pytest-style "======= ERRORS ======="
        r"[1-9]\d*\s*failed",  # Only match non-zero failures
        r"[1-9]\d*\s*error",   # Only match non-zero errors
        r"short test summary info",
        
        # unittest patterns
        r"FAIL:",
        r"ERROR:",
        r"failures=[1-9]\d*",  # Only match non-zero failures
        r"errors=[1-9]\d*",    # Only match non-zero errors
        r"unittest.*FAIL",
        
        # Generic patterns - be more specific
        r"AssertionError",
        r"Test.*failed:",   # "Test xyz failed:" style messages
        r"test.*failed:",   # "test xyz failed:" style messages
        r"Test.*error:",    # "Test xyz error:" style messages (be more specific)
        r"test.*error:",    # "test xyz error:" style messages (be more specific)
        r"Traceback\s*\(most recent call last\)",
        r"Exception:",
        r"Failed:",
        r"Failure:",
        r"FAILED:",
        
        # nose patterns
        r"FAIL\s*:",
        r"ERROR\s*:",
        
        # JavaScript test frameworks  
        r"[1-9]\d*\s*failing",  # Only match non-zero failing
        r"✖\s*[1-9]\d*\s*test[s]?\s*failed",  # Only match non-zero failed
        r"×\s*[1-9]\d*\s*test[s]?\s*failed",   # Only match non-zero failed
        
        # Build/compilation errors
        r"compilation failed",
        r"build failed",
        r"error:\s*",
        r"fatal:\s*",
        
        # Timeout/crash indicators
        #r"timeout",
        #r"timed out",
        #r"segmentation fault",
        #r"core dumped",
        #r"aborted",
        # r"killed",
    ]
    
    # Success patterns for various test frameworks
    success_patterns = [
        # pytest patterns
        r"passed",
        r"\d+\s*passed",
        r"={3,}\s*\d+\s*passed",
        r"all\s+tests?\s+passed",
        
        # unittest patterns
        r"OK\s*$",
        r"OK\s*\(",
        r"Ran\s+\d+\s+test[s]?\s+in\s+[\d.]+s\s*OK",
        
        # Generic patterns
        r"Success",
        r"successful",
        r"0\s*failed",
        r"0\s*error[s]?",
        r"test[s]?\s+passed",
        r"Test[s]?\s+passed",
        r"passed\s+\d+/\d+",
        r"✓",  # checkmark
        r"✔",  # checkmark variant
        r"√",  # checkmark variant
        
        # JavaScript test frameworks
        r"\d+\s*passing",
        r"✓\s*\d+\s*test[s]?",
        
        # JUnit/Maven patterns
        r"BUILD\s+SUCCESS",
        r"Tests\s+run:\s*\d+,\s*Failures:\s*0,\s*Errors:\s*0",
        r"Tests\s+run:\s*\d+.*Failures:\s*0.*Errors:\s*0",
        
        # RSpec patterns
        r"\d+\s*examples?,\s*0\s*failures?",
        r"examples?,\s*0\s*failures?",
    ]
    
    # Special handling for empty or very short output
    if len(test_output.strip()) < 10:
        # Very short output - rely more on return code
        return return_code == 0
    
    # Check for failure patterns first (more definitive)
    for pattern in failure_patterns:
        if re.search(pattern, test_output, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Check for explicit success patterns
    found_success = False
    for pattern in success_patterns:
        if re.search(pattern, test_output, re.IGNORECASE | re.MULTILINE):
            found_success = True
            break
    
    # Special case: Check for test count summary patterns
    # e.g., "5 passed, 0 failed" or "Tests: 5 passed, 0 failed" or "8 tests passed, 0 tests failed"
    test_summary_pattern = r"(\d+)\s*(?:test[s]?\s+)?passed[,\s]+(\d+)\s*(?:test[s]?\s+)?failed"
    match = re.search(test_summary_pattern, test_output, re.IGNORECASE)
    if match:
        passed_count = int(match.group(1))
        failed_count = int(match.group(2))
        return failed_count == 0 and passed_count > 0
    
    # If we found explicit success indicators, trust them
    if found_success:
        return True
    
    # No explicit success/failure found - use heuristics
    # Check if output looks like a test run at all
    test_run_indicators = [
        r"test",
        r"Test",
        r"spec",
        r"Spec",
        r"passed",
        r"failed",
        r"ran\s+\d+",
        r"running",
        r"executed",
    ]
    
    looks_like_test_output = any(re.search(p, test_output, re.IGNORECASE) for p in test_run_indicators)
    
    if not looks_like_test_output:
        # Doesn't look like test output - rely on return code
        return return_code == 0
    
    # If return code is 0 and no failure patterns found, consider it successful
    # This handles cases where test runners don't output explicit success messages
    return return_code == 0


def extract_all_python_files_from_patch(
    patch_content: str,
    *,
    include_deleted: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract all Python file paths from a unified diff, categorized by type.

    Returns:
        Dict with keys:
        - "test_files": List of test file paths
        - "source_files": List of non-test Python file paths 
        - "all_files": List of all Python file paths
    """
    test_patterns = [re.compile(p) for p in DEFAULT_TEST_PATTERNS]
    python_pattern = re.compile(DEFAULT_PYTHON_PATTERN)
    
    test_files: Set[str] = set()
    source_files: Set[str] = set()
    all_files: Set[str] = set()

    current_b_path: Optional[str] = None
    current_a_path: Optional[str] = None

    def _maybe_add(path: Optional[str]) -> None:
        if not path:
            return
        path = path.strip()
        if path == "/dev/null":
            return
        path = _strip_ab_prefix(path)
        # normalize any leading "./"
        if path.startswith("./"):
            path = path[2:]
        
        if python_pattern.search(path):
            all_files.add(path)
            if any(p.search(path) for p in test_patterns):
                test_files.add(path)
            else:
                source_files.add(path)

    for raw in patch_content.splitlines():
        line = raw.rstrip("\n")

        # Start of a new file diff section
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_a_path = parts[-2]
                current_b_path = parts[-1]
                _maybe_add(current_b_path)
            continue

        # Git rename/copy headers
        if line.startswith("rename to "):
            current_b_path = line[len("rename to "):].strip()
            _maybe_add(current_b_path)
            continue

        if line.startswith("copy to "):
            current_b_path = line[len("copy to "):].strip()
            _maybe_add(current_b_path)
            continue

        # Unified diff file markers
        if line.startswith("+++ "):
            path = line[len("+++ "):].strip()
            if path != "/dev/null":
                current_b_path = path
            else:
                current_b_path = "/dev/null"
            _maybe_add(current_b_path)
            continue

        if include_deleted and line.startswith("--- "):
            path = line[len("--- "):].strip()
            current_a_path = path
            if path != "/dev/null":
                _maybe_add(current_a_path)
            continue

    return {
        "test_files": sorted(test_files),
        "source_files": sorted(source_files),
        "all_files": sorted(all_files)
    }


def find_imports_in_file(file_path: str, repo_root: Optional[str] = None) -> Set[str]:
    """
    Find all import statements in a Python file and convert them to file paths.
    
    Args:
        file_path: Path to the Python file to analyze
        repo_root: Root directory of the repository (for relative path resolution)
        
    Returns:
        Set of potential file paths that this file imports
    """
    imports = set()
    
    if repo_root:
        full_path = Path(repo_root) / file_path
    else:
        full_path = Path(file_path)
    
    if not full_path.exists():
        return imports
        
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path = alias.name.replace('.', '/')
                    imports.add(f"{module_path}.py")
                    imports.add(f"{module_path}/__init__.py")
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports
                    if node.level > 0:
                        # Relative import, need to resolve based on current file location
                        current_dir = Path(file_path).parent
                        for _ in range(node.level - 1):
                            current_dir = current_dir.parent
                        module_path = str(current_dir / node.module.replace('.', '/'))
                    else:
                        module_path = node.module.replace('.', '/')
                    
                    imports.add(f"{module_path}.py")
                    imports.add(f"{module_path}/__init__.py")
                    
                    # Also check for specific imported names
                    for alias in node.names:
                        if alias.name != '*':
                            imports.add(f"{module_path}/{alias.name}.py")
                            
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        # If we can't parse the file, skip it
        pass
    
    return imports


def find_test_files_that_import_source_files(
    source_files: List[str],
    repo_root: str,
    test_directories: Optional[List[str]] = None
) -> Set[str]:
    """
    Find all test files that import any of the given source files.
    
    Args:
        source_files: List of source file paths that were modified
        repo_root: Root directory of the repository
        test_directories: Optional list of test directories to search in
        
    Returns:
        Set of test file paths that import the modified source files
    """
    if test_directories is None:
        test_directories = ['tests', 'test', 'testing']
    
    repo_path = Path(repo_root)
    related_tests = set()
    
    # Convert source files to module names and potential import patterns
    source_modules = set()
    for source_file in source_files:
        # Convert file path to module path
        module_path = source_file.replace('.py', '').replace('/', '.')
        source_modules.add(module_path)
        
        # Also add the file path itself for direct path imports
        source_modules.add(source_file)
        source_modules.add(source_file.replace('.py', ''))
    
    # Find all test files in the repository
    test_files = []
    for test_dir in test_directories:
        test_dir_path = repo_path / test_dir
        if test_dir_path.exists():
            for test_file in test_dir_path.rglob("*.py"):
                test_files.append(str(test_file.relative_to(repo_path)))
    
    # Also find test files by pattern
    for py_file in repo_path.rglob("*.py"):
        relative_path = str(py_file.relative_to(repo_path))
        test_patterns = [re.compile(p) for p in DEFAULT_TEST_PATTERNS]
        if any(p.search(relative_path) for p in test_patterns):
            test_files.append(relative_path)
    
    # Remove duplicates
    test_files = list(set(test_files))
    
    # Check each test file for imports of the modified source files
    for test_file in test_files:
        imports = find_imports_in_file(test_file, repo_root)
        
        # Check if any import matches our modified source files
        for import_path in imports:
            # Normalize paths for comparison
            import_normalized = import_path.replace('.py', '').replace('/', '.')
            for source_module in source_modules:
                source_normalized = source_module.replace('.py', '').replace('/', '.')
                if (import_normalized == source_normalized or 
                    import_normalized.endswith('.' + source_normalized) or
                    source_normalized.endswith('.' + import_normalized)):
                    related_tests.add(test_file)
                    break
    
    return related_tests


def build_comprehensive_pytest_command_from_patch(
    patch_content: str,
    repo_root: str,
    *,
    python: str = "python",
    extra_args: Optional[Iterable[str]] = None,
    fallback: str = "all",  # "all" | "empty" 
    base_cmd: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build comprehensive pytest command from patch content by analyzing dependencies.
    
    Args:
        patch_content: The git diff patch content
        repo_root: Root directory of the repository
        python: Python executable to use
        extra_args: Extra arguments to pass to pytest
        fallback: What to do if no test files found
        base_cmd: Base command to use
        
    Returns:
        Tuple of (pytest_command, analysis_info)
        where analysis_info contains:
        - "patch_test_files": Test files directly in patch
        - "patch_source_files": Source files modified in patch 
        - "related_test_files": Test files that import modified source files
        - "all_test_files": All test files to run
    """
    extra_args = list(extra_args or [])
    cmd = (base_cmd[:] if base_cmd else [python, "-m", "pytest"])

    # Extract files from patch
    file_analysis = extract_all_python_files_from_patch(patch_content)
    patch_test_files = file_analysis["test_files"]
    patch_source_files = file_analysis["source_files"]
    
    # Find test files that import the modified source files
    related_test_files = find_test_files_that_import_source_files(
        patch_source_files, 
        repo_root
    )
    
    # Combine all test files
    all_test_files = set(patch_test_files) | related_test_files
    all_test_files = sorted(all_test_files)
    
    analysis_info = {
        "patch_test_files": patch_test_files,
        "patch_source_files": patch_source_files,
        "related_test_files": sorted(related_test_files),
        "all_test_files": all_test_files
    }
    
    if all_test_files:
        cmd.extend(all_test_files)
    else:
        if fallback == "empty":
            return [], analysis_info
        # fallback == "all": leave cmd as-is (run full suite)

    cmd.extend(extra_args)
    return cmd, analysis_info