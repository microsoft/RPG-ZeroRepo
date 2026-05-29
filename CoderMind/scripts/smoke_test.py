#!/usr/bin/env python3
"""Smoke Test — Post-codegen integration sanity check.

Runs after all batches complete to verify the assembled project works
as a whole. Unlike unit tests (per-batch TDD), smoke tests verify
cross-module integration: imports resolve, entry point runs, and
no unimplemented stubs remain.

Three layers:
  1. Import completeness — every .py can be imported without error
  2. Entry point — main.py --help works (if main.py exists)
  3. Stub detection — unimplemented functions (pass, ..., NotImplementedError)

Usage:
    python3 smoke_test.py --json                # Run all layers
    python3 smoke_test.py --layer imports       # Import check only
    python3 smoke_test.py --layer entry         # Entry point only
    python3 smoke_test.py --layer stubs         # Stub detection only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from common.paths import DEV_VENV_DIR, REPO_DIR, get_scripts_dir, cmd_for

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SmokeFinding:
    """A single finding from smoke testing."""
    layer: str          # "imports", "entry_point", "stubs"
    severity: str       # "error", "warning", "info"
    check: str          # short check name
    message: str        # human-readable description
    details: str = ""   # optional details (traceback, output, etc.)

    def to_dict(self) -> Dict[str, Any]:
        d = {"layer": self.layer, "severity": self.severity,
             "check": self.check, "message": self.message}
        if self.details:
            d["details"] = self.details[:2000]
        return d


@dataclass
class SmokeResult:
    """Complete smoke test result."""
    success: bool = True
    project_type: str = "unknown"
    duration: float = 0.0
    layers: Dict[str, Any] = field(default_factory=dict)
    findings: List[SmokeFinding] = field(default_factory=list)

    def add_finding(self, finding: SmokeFinding) -> None:
        self.findings.append(finding)
        if finding.severity == "error":
            self.success = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "type": "smoke_test",
            "project_type": self.project_type,
            "duration": round(self.duration, 1),
            "layers": self.layers,
            "findings": [f.to_dict() for f in self.findings],
            "error_count": sum(1 for f in self.findings if f.severity == "error"),
            "warning_count": sum(1 for f in self.findings if f.severity == "warning"),
        }


# ============================================================================
# Helpers
# ============================================================================

def _get_python_exe(repo_path: Path) -> str:
    """Get the dev venv python path, falling back to sys.executable.

    ``repo_path`` is parameterised (not just ``DEV_VENV_DIR``) so tests
    can target an alternative tree; the bare-name part of the venv path
    is sourced from :data:`common.paths.DEV_VENV_NAME` via re-export.
    """
    venv_python = repo_path / DEV_VENV_DIR.name / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _find_source_files(repo_path: Path) -> List[Path]:
    """Find all .py source files (excluding tests, venv, __pycache__)."""
    skip_dirs = {".venv_dev", ".venv", "venv", "__pycache__", ".git",
                 ".cmind", ".rpgkit", ".pytest_cache", "node_modules"}
    result = []
    for py_file in repo_path.rglob("*.py"):
        parts = set(py_file.relative_to(repo_path).parts)
        if parts & skip_dirs:
            continue
        # Skip test files
        name = py_file.name
        if name.startswith("test_") or name.endswith("_test.py"):
            continue
        if any(p in ("tests", "test", "testing") for p in py_file.relative_to(repo_path).parts):
            continue
        result.append(py_file)
    return sorted(result)


def _run_in_repo(repo_path: Path, cmd: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command in the repo directory with the dev venv."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path)
    # Suppress interactive prompts
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(repo_path), env=env,
    )


# ============================================================================
# Layer 1: Import Completeness
# ============================================================================

def check_imports(repo_path: Path, result: SmokeResult) -> Dict[str, Any]:
    """Verify all source files can be imported without errors.

    Runs imports in batched subprocesses to isolate failures while
    keeping the check fast.
    """
    logger.info("Layer 1: Import completeness check")
    python_exe = _get_python_exe(repo_path)
    source_files = _find_source_files(repo_path)

    layer = {"total_files": len(source_files), "importable": 0, "failed": 0, "failures": []}

    # Build module names
    modules = []
    for py_file in source_files:
        rel = py_file.relative_to(repo_path)
        module_parts = list(rel.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        if not module_parts:
            continue
        modules.append(".".join(module_parts))

    if not modules:
        layer["passed"] = True
        return layer

    # Batch check: try importing all at once first
    batch_code = "import sys; sys.path.insert(0,'.'); " + "; ".join(
        f"import {m}" for m in modules
    )
    try:
        proc = _run_in_repo(repo_path, [python_exe, "-c", batch_code], timeout=30)
        if proc.returncode == 0:
            # All imports passed
            layer["importable"] = len(modules)
            layer["passed"] = True
            logger.info("  Imports: %d/%d passed (batch)", len(modules), len(modules))
            return layer
    except subprocess.TimeoutExpired:
        pass  # Fall through to individual checks

    # Batch failed — check individually to find which ones fail
    for module_name in modules:
        import_code = f"import sys; sys.path.insert(0,'.'); import {module_name}"
        try:
            proc = _run_in_repo(repo_path, [python_exe, "-c", import_code], timeout=15)
            if proc.returncode == 0:
                layer["importable"] += 1
            else:
                layer["failed"] += 1
                error_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "Unknown error"
                layer["failures"].append({"module": module_name, "error": error_line})
                result.add_finding(SmokeFinding(
                    layer="imports", severity="error",
                    check="import_fails",
                    message=f"Cannot import {module_name}: {error_line}",
                    details=proc.stderr[-500:] if proc.stderr else "",
                ))
        except subprocess.TimeoutExpired:
            layer["failed"] += 1
            layer["failures"].append({"module": module_name, "error": "import timed out (15s)"})
            result.add_finding(SmokeFinding(
                layer="imports", severity="error",
                check="import_timeout",
                message=f"Import of {module_name} timed out (possible infinite loop at import time)",
            ))

    layer["passed"] = layer["failed"] == 0
    logger.info("  Imports: %d/%d passed", layer["importable"], layer["total_files"])
    return layer


# ============================================================================
# Layer 2: Entry Point Validation
# ============================================================================

def check_entry_point(repo_path: Path, result: SmokeResult) -> Dict[str, Any]:
    """Verify main.py can start and --help works."""
    logger.info("Layer 2: Entry point check")
    main_py = repo_path / "main.py"
    python_exe = _get_python_exe(repo_path)

    if not main_py.exists():
        logger.info("  No main.py found, skipping")
        return {"skipped": True, "reason": "no main.py"}

    layer = {"exists": True, "help_works": False, "help_length": 0, "startup_error": None}

    # Try --help (safe, exits immediately)
    try:
        proc = _run_in_repo(repo_path, [python_exe, "main.py", "--help"], timeout=15)
        if proc.returncode == 0:
            layer["help_works"] = True
            layer["help_length"] = len(proc.stdout)
            if len(proc.stdout) < 30:
                result.add_finding(SmokeFinding(
                    layer="entry_point", severity="warning",
                    check="help_too_short",
                    message=f"main.py --help output is only {len(proc.stdout)} chars (possible stub)",
                ))
        else:
            layer["startup_error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "nonzero exit"
            result.add_finding(SmokeFinding(
                layer="entry_point", severity="error",
                check="help_fails",
                message=f"main.py --help failed: {layer['startup_error']}",
                details=proc.stderr[-1000:] if proc.stderr else "",
            ))
    except subprocess.TimeoutExpired:
        layer["startup_error"] = "timed out (15s)"
        result.add_finding(SmokeFinding(
            layer="entry_point", severity="error",
            check="help_timeout",
            message="main.py --help timed out (15s) — may hang on startup",
        ))

    layer["passed"] = layer["help_works"]
    return layer


# ============================================================================
# Layer 3: Stub/Placeholder Detection
# ============================================================================

def check_stubs(repo_path: Path, result: SmokeResult) -> Dict[str, Any]:
    """Detect unimplemented stubs and placeholders across all source files.

    Uses static_completeness_check from code_gen.static_checks, which
    detects pass-only functions, placeholder returns, NotImplementedError,
    and Ellipsis bodies.
    """
    logger.info("Layer 3: Stub/placeholder detection")
    from code_gen.static_checks import static_completeness_check

    source_files = _find_source_files(repo_path)
    file_paths = [str(f.relative_to(repo_path)) for f in source_files]
    issues = static_completeness_check(file_paths, repo_path)

    layer: Dict[str, Any] = {
        "total_files": len(source_files),
        "stub_count": 0,
        "placeholder_count": 0,
        "stubs": [],
    }

    for issue in issues:
        is_stub = issue.startswith("STUB:")
        is_critical = is_stub or issue.startswith("MISSING:") or issue.startswith("PARSE_ERROR:")
        if is_stub:
            layer["stub_count"] += 1
        else:
            layer["placeholder_count"] += 1

        result.add_finding(SmokeFinding(
            layer="stubs",
            severity="error" if is_critical else "warning",
            check="stub_detected" if is_stub else "placeholder_detected",
            message=issue,
        ))
        layer["stubs"].append(issue)

    layer["passed"] = layer["stub_count"] == 0
    logger.info("  Stubs: %d stubs, %d placeholders",
                layer["stub_count"], layer["placeholder_count"])
    return layer


# ============================================================================
# Main Orchestrator
# ============================================================================

def run_smoke_test(
    repo_path: Optional[Path] = None,
    layers: Optional[List[str]] = None,
) -> SmokeResult:
    """Run smoke tests on the generated repository.

    Args:
        repo_path: Path to the project repo. Defaults to common paths.
        layers: Which layers to run. None = all. Options: imports, entry, stubs

    Returns:
        SmokeResult with findings and per-layer details.
    """
    repo_path = repo_path or REPO_DIR
    run_layers = set(layers) if layers else {"imports", "entry", "stubs"}
    start = time.time()

    result = SmokeResult()

    # Layer 1: Import completeness
    if "imports" in run_layers:
        result.layers["imports"] = check_imports(repo_path, result)

    # Layer 2: Entry point
    if "entry" in run_layers:
        result.layers["entry_point"] = check_entry_point(repo_path, result)

    # Layer 3: Stub/placeholder detection
    if "stubs" in run_layers:
        result.layers["stubs"] = check_stubs(repo_path, result)

    result.duration = time.time() - start
    return result


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke Test — post-codegen integration sanity check",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--layer", choices=["imports", "entry", "stubs"],
                        action="append", help="Run specific layer(s) only")
    parser.add_argument("--repo", type=Path, help="Path to repo (default: auto)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if not args.json else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler for persistent logging via the shared helper
    # (idempotent; degrades gracefully on read-only FS).
    from common.logging_setup import setup_file_logging
    setup_file_logging("smoke_test")

    result = run_smoke_test(
        repo_path=args.repo,
        layers=args.layer,
    )

    r = result.to_dict()

    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if result.success else 1

    icon = "✅" if result.success else "❌"
    print(f"\n  {icon} Smoke Test ({result.project_type}) — {result.duration:.1f}s")

    for layer_name, layer_data in result.layers.items():
        if isinstance(layer_data, dict) and layer_data.get("skipped"):
            print(f"    ⏭  {layer_name}: skipped ({layer_data.get('reason','')})")
        elif isinstance(layer_data, dict):
            passed = layer_data.get("passed", True)
            licon = "✅" if passed else "❌"
            print(f"    {licon} {layer_name}")

    if result.findings:
        print(f"\n  Findings ({len(result.findings)}):")
        for f in result.findings:
            sev_icon = "❌" if f.severity == "error" else "⚠️"
            print(f"    {sev_icon} [{f.layer}] {f.message}")

    scripts = get_scripts_dir()
    if not result.success:
        print("\n  Fix the issues above, then re-run:")
        print(f"    {cmd_for('smoke_test.py')} --json")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
