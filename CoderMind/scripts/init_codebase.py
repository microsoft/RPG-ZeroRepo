#!/usr/bin/env python3
"""Initialize Codebase Script - Setup Initial Repository.

Sets up the initial repository state before TDD implementation:
1. Ensures we're on the main branch
2. Creates README.md with repository info
3. Creates .gitignore with Python cache rules
4. Writes base classes from base_classes.json
5. Creates an initial commit

This matches ZeroRepo's _setup_initial_repository() logic.
Interfaces and __init__.py are created during the TDD loop.

Output: JSON with initialization status

Usage:
    python init_codebase.py             # Initialize codebase
    python init_codebase.py --dry-run   # Preview without writing files
    python init_codebase.py --no-commit # Write files but don't commit
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.git_utils import GitRunner
from common.paths import (
    BASE_CLASSES_FILE,
    INTERFACES_FILE,
    REPO_RPG_FILE,
    FEATURE_BUILD_FILE,
    CODE_GEN_STATE_FILE as STATE_FILE,
    cmd_for,
    REPO_DIR,
)
from common.execution_state import load_code_gen_state, save_code_gen_state
from code_gen.context_collector import write_interface_skeletons


# Default .gitignore content for Python projects.
#
# Split into two logical blocks so ``create_gitignore`` can be smart:
# * ``_GITIGNORE_PYTHON_BLOCK`` — generic Python / OS / IDE ignores.
#   Modeled on the canonical ``github/gitignore/Python.gitignore`` template
#   (trimmed of niche framework sections: Django/Flask/Scrapy/SageMath/
#   Celery/Translations) plus the modern tool-cache entries (ruff, mypy,
#   pyright) and the common OS-junk lines (.DS_Store, Thumbs.db). Written
#   only when the user's existing ``.gitignore`` lacks ``__pycache__/``.
# * ``_GITIGNORE_CMIND_BLOCK`` — CoderMind-specific ignores (the entire
#   ``.cmind/`` runtime tree, the ``.claude`` workspace symlink, and the
#   ``.venv_dev/`` / ``.cmind_dev_env/`` venvs created by the codegen
#   pipeline). Appended whenever the existing ``.gitignore`` lacks
#   ``.cmind/``, regardless of whether Python ignores are already present.
#   This guarantees that an existing Python project getting bootstrapped
#   by ``init_codebase`` still gets the CoderMind runtime files ignored.
_GITIGNORE_PYTHON_BLOCK = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Sphinx / mkdocs documentation
docs/_build/
/site

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# PEP 582
__pypackages__/

# Type checkers
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/
.pytype/
pyrightconfig.json

# Linters / formatters
.ruff_cache/

# Cython debug symbols
cython_debug/

# Environments
.env
.env.local
.env.*.local
env/
venv/
ENV/
.venv/
env.bak/
venv.bak/

# Logs
*.log

# IDE / editors
.idea/
.vscode/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db
ehthumbs.db
desktop.ini
"""

_GITIGNORE_CMIND_BLOCK = """# CoderMind runtime workspace
# The entire .cmind/ tree is internal tooling state: logs, scripts copy,
# state snapshots, trajectory traces, encoder/codegen JSON artifacts.
# Treat it as ephemeral — none of it should be tracked in the project repo.
.cmind/

# CoderMind dev environments (created by codegen pipeline)
.venv_dev/
.cmind_dev_env/

# CoderMind workspace symlink
.claude
"""

# Dev-env-only subset of the CoderMind block.  Appended when a pre-existing
# ``.gitignore`` already carries ``.cmind/`` (so the full block is skipped)
# but predates the throwaway-venv rules.
_GITIGNORE_DEV_ENV_BLOCK = """# CoderMind dev environments (created by codegen pipeline)
.venv_dev/
.cmind_dev_env/
"""

# Kept for backward compatibility with any external import — equivalent to
# the full ``.gitignore`` written for a brand-new project.
GITIGNORE_CONTENT = _GITIGNORE_PYTHON_BLOCK + "\n" + _GITIGNORE_CMIND_BLOCK


def _gitignore_has_python_block(existing: str) -> bool:
    """Heuristic: does an existing .gitignore already cover Python cache?"""
    return "__pycache__/" in existing


def _gitignore_has_cmind_block(existing: str) -> bool:
    """Heuristic: does an existing .gitignore already ignore .cmind/?

    Accepts the line-anchored form ``.cmind/`` or ``.cmind`` (without a
    leading ``#``) so that earlier handwritten variants still count as
    "already configured" and don't get a duplicate block appended.
    """
    for raw in existing.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in (".cmind", ".cmind/", "/.cmind", "/.cmind/"):
            return True
    return False


def _gitignore_has_dev_env(existing: str) -> bool:
    """Heuristic: does an existing .gitignore already ignore ``.venv_dev/``?

    The codegen pipeline materializes a throwaway ``.venv_dev/`` virtual
    environment inside each project.  A fixture- or hand-authored
    ``.gitignore`` can ship ``.cmind/`` without these dev-env rules, so we
    detect them independently to avoid committing scratch venvs.
    """
    for raw in existing.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in (".venv_dev", ".venv_dev/", "/.venv_dev", "/.venv_dev/"):
            return True
    return False


# ============================================================================
# Agent Detection & Persistent Instructions
# ============================================================================
#
# Do not write persistent codegen instructions into the user's repository.
# Claude Code / Copilot auto-load those files for every session, which would
# contaminate unrelated commands (rpg_edit, encode, plain Q&A) with
# codegen-only instructions.  The recovery-after-/compact concern is handled
# by `templates/commands/code_gen.md` itself, which the user re-invokes via
# `/cmind.code_gen`.
#
# `cmind update` cleans up any stale `cmind-codegen.*` files left in older
# user workspaces (see src/cmind_cli/__init__.py).


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file, return empty dict if not found."""
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_repo_info() -> tuple:
    """Get repository name and description from RPG files.
    
    Returns (repo_name, repo_purpose)
    """
    # Try build_feature.json first
    build_feature = load_json_file(FEATURE_BUILD_FILE)
    if build_feature:
        name = build_feature.get("repository_name", "")
        purpose = build_feature.get("repository_purpose", "")
        if name:
            return name, purpose
    
    # Try repo_rpg.json
    repo_rpg = load_json_file(REPO_RPG_FILE)
    if repo_rpg:
        name = repo_rpg.get("repo_name", "")
        info = repo_rpg.get("repo_info", "")
        if name:
            return name, info
    
    # Fallback to directory name
    return REPO_DIR.name, ""


def write_file(file_path: Path, content: str, dry_run: bool = False) -> bool:
    """Write content to a file, creating directories as needed.
    
    Returns True if successful or would succeed (dry_run).
    """
    if dry_run:
        return True
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def create_readme(repo_path: Path, dry_run: bool = False) -> bool:
    """Create README.md if it doesn't exist."""
    readme_path = repo_path / "README.md"
    
    if readme_path.exists():
        return False  # Already exists, no change
    
    repo_name, repo_purpose = get_repo_info()
    
    content = f"# {repo_name}\n\n"
    if repo_purpose:
        content += f"{repo_purpose}\n\n"
    
    return write_file(readme_path, content, dry_run)


def create_gitignore(repo_path: Path, dry_run: bool = False) -> bool:
    """Create or update ``.gitignore`` to cover Python cache and CoderMind runtime.

    Behavior matrix:

    * ``.gitignore`` does not exist        → write the full template (Python + CoderMind blocks).
    * Exists, lacks Python block           → append Python + CoderMind blocks.
    * Exists, has Python block, no CoderMind → append only the CoderMind block.
    * Exists, has both blocks              → no-op.

    Returns True when the file was created/modified, False when nothing changed
    or an error prevented writing.
    """
    gitignore_path = repo_path / ".gitignore"

    if not gitignore_path.exists():
        return write_file(gitignore_path, GITIGNORE_CONTENT, dry_run)

    try:
        existing = gitignore_path.read_text(encoding='utf-8')
    except Exception:
        return False

    has_python = _gitignore_has_python_block(existing)
    has_cmind = _gitignore_has_cmind_block(existing)
    has_dev_env = _gitignore_has_dev_env(existing)

    if has_python and has_cmind and has_dev_env:
        return False  # Already fully configured

    additions = ""
    if not has_python:
        additions += _GITIGNORE_PYTHON_BLOCK
    if not has_cmind:
        # Separate the two blocks with a blank line for readability.
        if additions:
            additions += "\n"
        additions += _GITIGNORE_CMIND_BLOCK
    elif not has_dev_env:
        # The CoderMind block is present but predates the dev-env rules
        # (e.g. a fixture-shipped .gitignore that only carried ``.cmind/``).
        # Append just the dev-env venv ignores so codegen scratch venvs are
        # never committed.
        if additions:
            additions += "\n"
        additions += _GITIGNORE_DEV_ENV_BLOCK

    if not additions:
        return False

    if not dry_run:
        try:
            new_content = existing.rstrip() + "\n\n" + additions
            gitignore_path.write_text(new_content, encoding='utf-8')
        except Exception:
            return False
    return True


def write_base_classes(
    repo_path: Path,
    base_classes_path: Path,
    dry_run: bool = False
) -> List[str]:
    """Write base classes from base_classes.json.
    
    Returns list of files written.
    """
    base_classes_data = load_json_file(base_classes_path)
    if not base_classes_data:
        return []
    
    files_written = []
    
    # Check for "files" field (pre-aggregated file contents)
    if "files" in base_classes_data:
        for file_path, content in base_classes_data["files"].items():
            full_path = repo_path / file_path
            if write_file(full_path, content, dry_run):
                files_written.append(file_path)
    
    # Check for "base_classes" array
    base_class_list = base_classes_data.get("base_classes", [])
    
    # Group by file_path to avoid overwriting
    file_contents: Dict[str, List[str]] = {}
    
    for bc in base_class_list:
        file_path = bc.get("file_path", "")
        code = bc.get("code", "")
        
        if file_path and code:
            if file_path not in file_contents:
                file_contents[file_path] = []
            file_contents[file_path].append(code)
    
    # Write aggregated content
    for file_path, code_blocks in file_contents.items():
        if file_path in files_written:
            continue  # Already written from "files" field
        
        content = "\n\n".join(code_blocks)
        full_path = repo_path / file_path
        if write_file(full_path, content, dry_run):
            files_written.append(file_path)
    
    # Check for "data_structures" array (data flow type stubs)
    # Note: file_path may be empty if not yet assigned by interface designer
    data_structures_list = base_classes_data.get("data_structures", [])
    
    ds_file_contents: Dict[str, List[str]] = {}
    
    for ds in data_structures_list:
        file_path = ds.get("file_path", "")
        code = ds.get("code", "")
        
        if file_path and code:  # Skip entries without file_path
            if file_path not in ds_file_contents:
                ds_file_contents[file_path] = []
            ds_file_contents[file_path].append(code)
    
    # Write data structure stubs - append to existing files or create new
    for file_path, code_blocks in ds_file_contents.items():
        content = "\n\n".join(code_blocks)
        full_path = repo_path / file_path
        
        if file_path in files_written:
            # File was already written by base_classes - append data structures
            if not dry_run:
                existing = full_path.read_text(encoding='utf-8')
                combined = existing.rstrip() + "\n\n\n" + content
                full_path.write_text(combined, encoding='utf-8')
        else:
            if write_file(full_path, content, dry_run):
                files_written.append(file_path)
    
    return files_written


def create_initial_commit(
    repo_path: Path,
    files_written: List[str],
    readme_created: bool,
    gitignore_created: bool
) -> Optional[str]:
    """Stage and commit all written files.

    Returns commit hash if successful, "no-changes" if nothing to commit,
    None on error.
    """
    try:
        git = GitRunner(str(repo_path))

        parts = []
        if readme_created:
            parts.append("README")
        if gitignore_created:
            parts.append(".gitignore")
        if files_written:
            parts.append(f"{len(files_written)} base class files")

        if not parts:
            return "no-changes"

        if not git.has_uncommitted_changes():
            return "no-changes"

        message = "chore: initial repository setup\n\n"
        message += "Add " + ", ".join(parts)

        success, commit_hash = git.stage_and_commit(message)

        if success and commit_hash:
            return commit_hash

        return None

    except Exception:
        return None


def update_code_gen_state(state_path: Path, initial_commit: str) -> None:
    """Update code_gen_state.jsonl with initial commit info (append a line)."""
    state = load_code_gen_state(state_path)
    state.initialized = True
    state.initialized_at = datetime.now().isoformat()
    state.initial_commit = initial_commit
    save_code_gen_state(state, state_path)


def init_codebase(
    repo_path: Path = None,
    base_classes_path: Path = BASE_CLASSES_FILE,
    state_path: Path = STATE_FILE,
    dry_run: bool = False,
    no_commit: bool = False
) -> Dict[str, Any]:
    """Initialize the codebase with README, .gitignore, and base classes.
    
    Matches ZeroRepo's _setup_initial_repository() logic.
    
    Args:
        repo_path: Repository path (defaults to cwd)
        base_classes_path: Path to base_classes.json
        state_path: Path to code_gen_state.jsonl
        dry_run: Preview without writing files
        no_commit: Write files but don't commit
        
    Returns:
        Dict with initialization results
    """
    repo_path = repo_path or REPO_DIR

    # Ensure repo directory exists
    repo_path.mkdir(parents=True, exist_ok=True)

    # Ensure .cmind/ runtime directories exist.  This is normally already
    # done by ``cmind init`` / ``cmind update`` (see
    # ``cmind_cli.ensure_cmind_runtime_dirs``), but we mkdir here too as
    # a safety net: a workspace created by an older cmind may lack
    # ``.cmind/logs/``, in which case stage prompts that redirect with
    # shell ``>`` fail before the Python process can recover.  Creating
    # them here at code_gen bootstrap is harmless and idempotent.
    from common.paths import LOGS_DIR, DATA_DIR, TRAJECTORY_DIR
    for d in (LOGS_DIR, DATA_DIR, TRAJECTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # NOTE: A previous version of this function created a ``.claude``
    # symlink inside ``repo/`` because the project repo lived in a
    # ``<workspace>/repo/`` subdirectory and sub-agents ran with cwd =
    # repo/.  After the workspace==repo unification the workspace root
    # IS the project repo root, so ``.claude`` is already at the right
    # location and the symlink is unnecessary (and would point at
    # ``<workspace.parent>/.claude``, i.e. outside the workspace).
    # Block removed on purpose; do not reintroduce.

    # Check if already initialized
    if state_path.exists():
        try:
            state = load_code_gen_state(state_path)
            if state.initialized:
                return {
                    "success": False,
                    "error": "Codebase already initialized",
                    "initial_commit": state.initial_commit,
                    "initialized_at": state.initialized_at,
                    "suggestion": "Run run_batch.py to start codegen",
                    "next_action": (
                        f"Already initialized. Run: {cmd_for('run_batch.py')} --next --json "
                        f"to start the next batch."
                    )
                }
        except Exception:
            pass
    
    # Ensure on main branch and clean workspace
    if not dry_run:
        git = GitRunner(str(repo_path))
        git.ensure_clean_workspace()
        success, msg = git.ensure_main_branch()
        if not success:
            return {
                "success": False,
                "error": msg,
                "suggestion": "Manually switch to main branch and retry",
                "next_action": "Git branch error. Switch to the main branch manually, then re-run init_codebase.py --json."
            }
    
    # Track changes
    readme_created = False
    gitignore_created = False
    base_files = []

    # 1. Create README.md
    readme_created = create_readme(repo_path, dry_run)

    # 2. Create/update .gitignore
    gitignore_created = create_gitignore(repo_path, dry_run)

    # 3. Write base classes
    if base_classes_path.exists():
        base_files = write_base_classes(repo_path, base_classes_path, dry_run)

    # 4. Write interface skeletons (one-time, from interfaces.json)
    skeletons_written: List[str] = []
    if not dry_run and INTERFACES_FILE.exists():
        try:
            skel_result = write_interface_skeletons(INTERFACES_FILE, repo_path)
            skeletons_written = skel_result.get("written", [])
        except Exception as e:
            print(f"Warning: failed to write interface skeletons: {e}", file=sys.stderr)

    # Check if any changes were made
    has_changes = readme_created or gitignore_created or base_files or skeletons_written
    
    if not has_changes:
        # Mark initialized even if no file changes were needed
        if not dry_run:
            state = load_code_gen_state(state_path)
            if not state.initialized:
                state.initialized = True
                state.initialized_at = datetime.now().isoformat()
                save_code_gen_state(state, state_path)
        return {
            "success": True,
            "message": "Repository already set up, no changes needed",
            "readme_created": False,
            "gitignore_created": False,
            "base_class_files": 0,
            "next_action": (
                f"Codebase already set up. Run: {cmd_for('run_batch.py')} --next --json "
                f"to start the first batch."
            )
        }

    # 5. Create commit
    commit_hash = None
    if not dry_run and not no_commit:
        commit_hash = create_initial_commit(
            repo_path,
            base_files + skeletons_written,
            readme_created,
            gitignore_created
        )

        if commit_hash and commit_hash not in ["no-changes", None]:
            state = load_code_gen_state(state_path)
            state.interfaces_written = bool(skeletons_written)
            save_code_gen_state(state, state_path)
            update_code_gen_state(state_path, commit_hash)

    return {
        "success": True,
        "dry_run": dry_run,
        "readme_created": readme_created,
        "gitignore_created": gitignore_created,
        "base_class_files": len(base_files),
        "base_class_file_list": base_files,
        "skeleton_files": len(skeletons_written),
        "skeleton_file_list": skeletons_written,
        "commit_hash": commit_hash,
        "message": "Repository initialized successfully" if not dry_run else "Dry run complete",
        "next_action": (
            f"Codebase initialized. Run: {cmd_for('run_batch.py')} --next --json "
            f"to start the first batch."
        ) if not dry_run else "Dry run complete. Re-run without --dry-run to apply changes."
    }


def print_result(result: Dict[str, Any], json_output: bool = False):
    """Print the result in a user-friendly format."""
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    if not result.get("success"):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        if result.get("suggestion"):
            print(f"   Suggestion: {result['suggestion']}")
        return
    
    if result.get("dry_run"):
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║                   DRY RUN PREVIEW                           ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    else:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║               REPOSITORY INITIALIZED                        ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    
    print(f"\n   Files created/updated:")
    print(f"      -  README.md: {'[OK] created' if result.get('readme_created') else '[-] already exists'}")
    print(f"      -  .gitignore: {'[OK] created/updated' if result.get('gitignore_created') else '[-] already exists'}")
    
    base_files = result.get("base_class_files", 0)
    if base_files > 0:
        print(f"      -  Base classes: {base_files} files")
        for f in result.get("base_class_file_list", [])[:5]:
            print(f"        - {f}")
        if base_files > 5:
            print(f"        ... and {base_files - 5} more")
    else:
        print(f"      -  Base classes: (none found in base_classes.json)")
    
    if result.get("commit_hash"):
        if result["commit_hash"] == "no-changes":
            print(f"\n   No changes to commit")
        else:
            print(f"\n   Initial commit: {result['commit_hash'][:8]}")
    
    print("\n   " + "─" * 60)
    print(f"   Next step: Run /cmind.code_gen to start TDD")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize codebase with README, .gitignore, and base classes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing files or creating commits"
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Write files but don't create a commit"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--base-classes", "-b",
        type=Path,
        default=BASE_CLASSES_FILE,
        help=f"Input base classes file (default: {BASE_CLASSES_FILE})"
    )
    
    args = parser.parse_args()
    
    result = init_codebase(
        base_classes_path=args.base_classes,
        dry_run=args.dry_run,
        no_commit=args.no_commit
    )
    
    print_result(result, json_output=args.json)
    
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    exit(main())
