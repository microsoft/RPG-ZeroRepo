#!/usr/bin/env python3
"""Prompt Templates for Code Generation.

Contains all prompt templates used in the TDD workflow:
- Test generation prompts
- Code generation prompts  
- Environment setup prompts
- Failure analysis prompts
"""

import sys as _sys
from pathlib import Path as _Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from common.task_batch import PlannedTask

# Ensure scripts dir is on path for common.paths import
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from common.paths import REPO_DIR as _REPO_DIR


_FENCE_BY_SUFFIX = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}


def _markdown_fence_for_path(file_path: str) -> str:
    """Return a markdown code fence language for ``file_path``."""
    return _FENCE_BY_SUFFIX.get(_Path(file_path).suffix.lower(), "text")


# ============================================================================
# Dependency Context Formatter
# ============================================================================

def _read_source_file(file_path: str, max_bytes: int = 8192) -> Optional[str]:
    """Read a source file from disk (skeleton or implementation).

    *file_path* is relative to the project repo root (e.g. ``src/pkg/foo.py``).
    The function resolves it against :data:`common.paths.REPO_DIR` to find
    the actual file.

    Returns the file content (truncated to *max_bytes*) or ``None``
    when the file does not exist or is empty.
    """
    if not file_path:
        return None
    p = _Path(file_path)
    # If relative, resolve against the project repo directory
    if not p.is_absolute():
        p = _REPO_DIR / p
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return None
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n# ... (truncated)\n"
        return text
    except Exception:
        return None


def _format_skeleton_context(file_path: str) -> str:
    """Return a prompt section with the skeleton / interface file content.

    Used in the FIRST iteration when only skeleton code exists on disk.
    If the file doesn't exist or is empty, returns an empty string.
    """
    content = _read_source_file(file_path)
    if not content:
        return ""
    return (
        f"\n## Skeleton / Interface Definitions for `{file_path}`\n"
        "The target file already contains interface definitions (class signatures, method\n"
        "signatures, docstrings, type hints) from the design stage. **Your tests MUST\n"
        "target ONLY the classes, methods, and signatures defined below.** Do NOT invent\n"
        "methods or features that are not present in this skeleton.\n\n"
        f"```{_markdown_fence_for_path(file_path)}\n{content}\n```\n"
    )


def _format_current_source_context(file_path: str) -> str:
    """Return a prompt section with the CURRENT source file content.

    Used in iteration 2+ after the code-generation sub-agent has written
    actual implementation.  The test agent needs to see the real code
    to fix tests accurately.
    """
    content = _read_source_file(file_path)
    if not content:
        return ""
    return (
        f"\n## Current Implementation of `{file_path}`\n"
        "The source file below contains the ACTUAL implementation code generated in the\n"
        "previous step. **Your tests MUST match the real API** (class names, method names,\n"
        "signatures, return types) as shown below. Fix any tests that expect methods or\n"
        "behaviors not present in this implementation.\n\n"
        f"```{_markdown_fence_for_path(file_path)}\n{content}\n```\n"
    )


def _format_dependency_context(ctx: Optional[Dict[str, Any]]) -> str:
    """Format the dependency context dict into a compact prompt section.

    Uses a "map not snapshot" approach: tells the sub-agent *where* to find
    information (file paths, class names) rather than inlining full code.
    The sub-agent has tool access and can read files on demand.

    Returns an empty string when *ctx* is ``None`` or empty.
    """
    if not ctx:
        return ""

    parts: List[str] = []
    parts.append("\n## Project Context (from earlier design stages)\n")

    # --- Project background & technology stack --------------------------------
    project_bg = ctx.get("project_background", "")
    completed = ctx.get("completed", {})
    if project_bg and project_bg.strip():
        if len(completed) == 0:
            # First batch: full background
            parts.append(project_bg)
            parts.append(
                "**Use the technology stack described above** when making implementation "
                "decisions (framework choice, database layer, routing patterns, etc.). "
                "Generate idiomatic code for the specified technologies.\n"
            )
        else:
            # Subsequent batches: one-line summary (sub-agent can read files for full context)
            # Extract first line as a compact summary
            first_line = project_bg.strip().split('\n')[0].strip('#').strip()
            parts.append(
                f"### Project: {first_line}\n"
                "See completed modules below for full architecture context. "
                "Use the same technology stack and patterns as existing code.\n"
            )

    # --- Base classes: compact summary with file pointers ---------------------
    bc_data = ctx.get("base_classes", {})
    base_classes = bc_data.get("base_classes", [])
    if base_classes:
        parts.append("### Base Classes (shared across all modules)\n")
        parts.append("Read these files directly for full API signatures and docstrings.\n")
        for bc in base_classes:
            fp = bc.get("file_path", "")
            code = bc.get("code", "")
            subs = bc.get("subclasses", {})
            if not code:
                continue
            # Extract class and method names through the target-language
            # backend resolved from the file's extension (defaults to
            # Python). Syntax errors yield an empty unit list, so malformed
            # base class snippets simply contribute no class summary here.
            from decoder_lang import get_backend as _get_backend
            from lang_parser import detect_language as _detect_language
            backend = _get_backend(_detect_language(fp) or "python")
            class_like = {"class", "struct", "interface", "type", "enum"}
            units = backend.list_code_units(code, fp)
            classes = [
                u for u in units
                if u.unit_type in class_like and u.parent is None
            ]
            if classes:
                first_class = classes[0]
                methods = [
                    u.name for u in units
                    if u.unit_type == "method" and u.parent == first_class.name
                ]
                parts.append(
                    f"- `{first_class.name}` in `{fp}` — methods: {', '.join(methods)}"
                )
                if subs:
                    for parent, children in subs.items():
                        if parent == first_class.name:
                            parts.append(f"  Subclasses: {', '.join(children)}")
            else:
                parts.append(f"- `{fp}` (parse error — read file directly)")
        parts.append("")

    # --- Data structures: compact summary ------------------------------------
    data_structs = bc_data.get("data_structures", [])
    if data_structs:
        subtree = ctx.get("current_subtree", "")
        parts.append(f"### Data Structures (subtree: {subtree})\n")
        for ds in data_structs:
            fp = ds.get("file_path", "")
            types = ds.get("data_flow_types", [])
            if types and fp:
                parts.append(f"- Types: {', '.join(types)} — read `{fp}`")
            elif types:
                parts.append(f"- Types: {', '.join(types)} — defined in skeleton files")
        parts.append("")

    # --- Data flow edges: compact text format --------------------------------
    df_edges = ctx.get("data_flow_edges", [])
    if df_edges:
        parts.append("### Data Flow (edges involving current subtree)\n")
        for edge in df_edges:
            src = edge.get("source", "?")
            tgt = edge.get("target", "?")
            data_type = edge.get("data_type", "")
            dtype_str = f" ({data_type})" if data_type else ""
            parts.append(f"- {src} → {tgt}{dtype_str}")
        parts.append("")

    # --- Dependency files: deduplicated by file --------------------------------
    deps = ctx.get("dependencies", {})
    dep_files = deps.get("dependent_files", [])
    if dep_files:
        parts.append(
            "### Dependencies of Current File\n"
            "These files are dependencies of your current implementation.\n"
            "**You MUST read these files** before writing code to understand:\n"
            "- What functions/classes they export\n"
            "- What parameters they expect\n"
            "- What they return\n"
            "Do NOT assume or invent APIs — use the actual interface defined in these files.\n"
        )
        # Group by file to deduplicate repeated entries
        from collections import defaultdict as _defaultdict

        inh_grouped = _defaultdict(list)
        for inh in deps.get("inherits_from", []):
            parent = inh['parent']
            if parent not in inh_grouped[inh['parent_file']]:
                inh_grouped[inh['parent_file']].append(parent)
        for f, parents in inh_grouped.items():
            parts.append(f"- `{f}` (inherits: {', '.join(parents)})")

        inv_grouped = _defaultdict(list)
        for inv in deps.get("invokes", []):
            callee = inv['callee']
            if callee not in inv_grouped[inv['callee_file']]:
                inv_grouped[inv['callee_file']].append(callee)
        for f, callees in inv_grouped.items():
            parts.append(f"- `{f}` (invokes: {', '.join(callees)})")

        ref_grouped = _defaultdict(list)
        for ref in deps.get("references", []):
            typ = ref['type']
            if typ not in ref_grouped[ref['type_file']]:
                ref_grouped[ref['type_file']].append(typ)
        for f, types in ref_grouped.items():
            parts.append(f"- `{f}` (references: {', '.join(types)})")
        parts.append("")

    # --- Completed modules: show ALL files, not truncated --------------------
    # 'completed' was already fetched above for project background shortening
    if completed:
        file_list = list(completed.keys())
        parts.append(f"### Already Completed Modules ({len(file_list)} files)\n")
        parts.append(
            "These files have been implemented — import and use them freely.\n"
            "**Read any of these files** if you need to understand their actual API "
            "(function signatures, class interfaces, return types). "
            "Do NOT guess what functions exist — read the source code directly.\n"
        )
        for fp in file_list:  # Show ALL, not truncated
            parts.append(f"- `{fp}`")
        parts.append("")

    # --- ORM Model Registry: cross-file model import requirements -----------
    model_reg = ctx.get("model_registry", {})
    if model_reg and model_reg.get("models"):
        models = model_reg["models"]
        rels = model_reg.get("relationships", [])
        model_files = model_reg.get("model_files", [])

        # Only show this section if there are cross-file relationships
        # (i.e. the mapper-configuration trap is actually possible).
        # Projects with models all in one file don't have this problem.
        cross_file_rels = [r for r in rels if r.get("target_file") and
                           r["target_file"] != r.get("source_file")]

        if cross_file_rels:
            parts.append("### ORM Model Registry\n")
            parts.append("All ORM model classes in this project:\n")
            for cls_name, cls_file in sorted(models.items()):
                parts.append(f"- `{cls_name}` → `{cls_file}`")

            parts.append("\n**Cross-file relationships** (string references resolved at runtime):\n")
            for r in cross_file_rels:
                parts.append(
                    f"- `{r['source_class']}.{r['field']}` → `{r['target_class']}` "
                    f"(in `{r['target_file']}`)"
                )

            parts.append(
                "\n**CRITICAL for tests**: ORM frameworks (SQLAlchemy, etc.) resolve "
                "string-based relationship targets by looking up class names in the "
                "mapper registry. When your test instantiates ANY model, the ORM may "
                "eagerly configure ALL mappers. If model A has a relationship pointing "
                "to model B in another file, class B must be imported — even if your "
                "test never uses B directly.\n"
                "**Import ALL model files** in your test fixture before using any "
                "model or calling `db.create_all()`:\n"
                "```python\n"
            )
            for mf in model_files:
                mod = mf.replace("/", ".").replace(".py", "")
                parts.append(f"import {mod}  # noqa: F401")
            parts.append("```\n")

    # NOTE: ctx may also contain "reverse_deps" (who depends on this file).
    # Not displayed in TDD prompt — value is marginal for codegen since
    # skeleton already defines the fixed API. Reserved for future use in
    # design_interfaces review (P2).

    text = "\n".join(parts)
    return text


# ============================================================================
# Initial Prompts (First Iteration)
# ============================================================================

def init_test_gen_prompt(
    task: str,
    batch_units: str,
    file_path: str,
    task_type: str = "implementation",
    dependency_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate initial test generation prompt for a task batch.

    (This step is for writing/adding tests only.).
    
    Args:
        task: Task description
        batch_units: Comma-separated list of unit keys
        file_path: Target file path
        task_type: Type of task (implementation, integration_test, final_test_docs)
        dependency_context: Dependency context from earlier design stages
    """
    if task_type == "implementation":
        prompt = (
            "You are working in a Test-Driven Development (TDD) workflow.\n"
            "In this step your responsibility is ONLY to write or update tests.\n"
            "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write tests for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- Follow the same testing framework, helpers, fixtures, and style already used.\n"
            "- Cover at minimum: normal behavior, key edge cases, and meaningful failure cases.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
            "- If the expected behavior is unclear, encode the most reasonable interpretation\n"
            "  and add comments explaining your assumptions.\n"
            "- **Only import packages available in the environment.** Use the target language's standard library\n"
            "  and internal project modules freely. For third-party packages, only import\n"
            "  them if they are already used by existing source files. Never add unused imports.\n"
            "- **CRITICAL: Only test classes, methods, and functions that exist in the skeleton\n"
            "  file below (if provided). Do NOT invent or assume additional methods, features,\n"
            "  or APIs beyond what is defined in the skeleton.**\n"
            "**Plan first — output a brief summary** (3–5 sentences) before writing any code:\n"
            "- What test scenarios you intend to write and why.\n"
            "- Key edge cases or design trade-offs you will address.\n"
            "- Any assumptions about expected behavior.\n"
            "This is a small task. **DO NOT over-engineer with too many tests.**\n"
        )
        # Point agent to skeleton file (read on demand, not inlined)
        if file_path:
            prompt += (
                f"\nThe skeleton file `{file_path}` contains interface definitions "
                "(signatures, docstrings, type hints). **Read this file** before "
                "writing tests to understand the exact API.\n"
            )
    elif task_type == "integration_test":
        prompt = (
            "You are working on Integration Testing.\n"
            "Your primary responsibility is to write or update integration tests.\n"
            "If you discover genuine integration bugs in production code while writing tests, "
            "note them — you will have a chance to fix them in the next step.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write integration tests for units [{batch_units}].\n\n"
            "File placement:\n"
            "- There is NO pre-determined target file. You decide where to place test files.\n"
            "- First, explore the repository to understand the existing test directory layout\n"
            "  (e.g., tests/, test/, or co-located test files).\n"
            "- Create new test files following the same naming conventions (e.g., test_<module>.py).\n"
            "- Place integration tests in the most appropriate location per project conventions.\n\n"
            "Requirements:\n"
            "- Focus on testing interactions between components, modules, or systems.\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- Test data flows, API contracts, and cross-module dependencies.\n"
            "- Cover realistic scenarios including success paths and failure modes.\n"
            "- Ensure tests are isolated and can run independently.\n"
            "- Mock external dependencies appropriately.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
        )
    elif task_type == "final_test_docs":
        prompt = (
            "You are working on Final Testing and Documentation.\n"
            "Your primary responsibility is to write comprehensive end-to-end tests AND create documentation.\n"
            "If you discover genuine integration bugs in production code while writing tests, "
            "note them — you will have a chance to fix them in the next step.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: create final tests and documentation for units [{batch_units}].\n\n"
            "File placement:\n"
            "- There is NO pre-determined target file. You decide where to place files.\n"
            "- Explore the repository structure to find the best locations.\n"
            "- Place end-to-end tests in the project's test directory (e.g., tests/e2e/ or tests/).\n"
            "- Place documentation updates in the project root or docs/ directory.\n"
            "- Create example scripts in an examples/ directory if one exists, or create it.\n"
            "- Follow existing project conventions for file naming and organization.\n\n"
            "Requirements:\n"
            "- Write end-to-end tests that validate complete user workflows.\n"
            "- Create or update documentation (README, API docs, usage examples).\n"
            "- Ensure all critical paths and user scenarios are covered.\n"
            "- Document any assumptions, limitations, or known issues.\n"
            "- Provide clear examples and usage instructions.\n"
            "- Validate the entire system works as intended.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
        )
    else:
        # Fallback to implementation behavior
        prompt = (
            "You are working in a Test-Driven Development (TDD) workflow.\n"
            "In this step your responsibility is to write or update tests.\n"
            "If you discover genuine bugs in production code while writing tests, "
            "note them — you will have a chance to fix them in the next step.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: write tests for units [{batch_units}] in {file_path}.\n\n"
            "Requirements:\n"
            "- Use the repository's existing test layout and conventions.\n"
            "- Follow the same testing framework, helpers, fixtures, and style already used.\n"
            "- Cover at minimum: normal behavior, key edge cases, and meaningful failure cases.\n"
            "- Keep tests deterministic, readable, and maintainable.\n"
            "- If the expected behavior is unclear, encode the most reasonable interpretation\n"
            "  and add comments explaining your assumptions.\n"
        )
    
    # NOTE: dependency_context is NOT appended here — it is provided once
    # in the TDD_BATCH_PREAMBLE template to avoid 5x duplication.
    # Only init_project_file_gen_prompt() retains its own dep_context
    # because TDD_PROJECT_FILE_PREAMBLE has no {dependency_context} slot.
    return prompt


def init_code_gen_prompt(
    task: str,
    batch_units: str,
    file_path: str,
    task_type: str = "implementation",
    dependency_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate initial code generation prompt for a task batch.

    (This step is for incremental implementation of production code.).
    
    Args:
        task: Task description
        batch_units: Comma-separated list of unit keys
        file_path: Target file path
        task_type: Type of task
        dependency_context: Dependency context from earlier design stages
    """
    if task_type == "implementation":
        prompt = (
            "You are working in an incremental development workflow.\n"
            "Tests may already exist or may be added later.\n"
            "Your responsibility in this step is to implement or refine production code only.\n"
            "Do NOT modify test files or environment/dependency configuration here.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: implement or refine units [{batch_units}] in {file_path}.\n\n"
            "Guidelines:\n"
            "- Implement behavior consistent with the task description and any existing tests.\n"
            "- Work incrementally: it is fine if not all tests pass yet, as long as your code moves toward correctness.\n"
            "- Prefer small, focused, maintainable changes.\n"
            "- Follow repository architecture, conventions, and abstractions.\n"
            "- **Treat this project as an integrated whole. Prioritize code reuse and leverage existing\n"
            "  implementations. Before writing any new logic, check the dependency context and existing code\n"
            "  for utilities, helpers, or patterns that can be reused. Do NOT reinvent the wheel.**\n"
            "- Do NOT edit or create test files at this stage.\n"
            "- Assume the current file may be missing some imports. Whenever you use a function, class, type, or constant,\n"
            "  you MUST ensure the corresponding import is present at the top of the file.\n"
            "- Before adding new imports, search the repository for existing usage of similar helpers or patterns and\n"
            "  prefer the same modules and import style (to stay consistent with the codebase).\n"
            "- If you introduce new symbols in this file, also add or update the import statements so that the module can be\n"
            "  imported and executed without NameError or ImportError.\n"
            "- **Only import packages available in the environment.** Use the target language's standard library\n"
            "  and internal project modules freely. For third-party packages, only import\n"
            "  them if they are already used by existing source files. Before adding any import,\n"
            "  verify you actually USE the imported name in your code — never add unused imports.\n"
            "\n**Plan first — output a brief summary** (3–5 sentences) before writing any code:\n"
            "- Your implementation approach and key design decisions.\n"
            "- How you will use the dependency context (base classes, data flow, etc.).\n"
            "- Any assumptions or trade-offs to note.\n"
        )
    elif task_type == "integration_test":
        prompt = (
            "You are working on Integration Bug Fixes.\n"
            "Your integration tests (from the previous step) may have revealed \n"
            "genuine bugs in the production code. Your responsibility is to fix those bugs.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: fix integration issues found by tests for [{batch_units}].\n\n"
            "Common issues to look for and fix:\n"
            "- Route handlers returning placeholder strings instead of calling real handler functions\n"
            "- CSS class names in style modules not matching class names used in HTML-generating modules\n"
            "- Missing imports or function calls between modules that should be connected\n"
            "- Data format mismatches at module boundaries\n\n"
            "Guidelines:\n"
            "- Fix only what is needed to make integration tests pass.\n"
            "- Read the actual source files to understand current implementation before changing.\n"
            "- Do NOT refactor working code. Only fix broken connections.\n"
            "- Do NOT create the project entry point \u2014 it will be created in a later task.\n"
            "- Do NOT edit test files at this stage.\n"
        )
    elif task_type == "final_test_docs":
        prompt = (
            "You are working on End-to-End Integration Fixes.\n"
            "Your end-to-end tests (from the previous step) may have revealed integration \n"
            "bugs in the production code. Your responsibility is to fix those bugs.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: fix integration issues found by tests for [{batch_units}].\n\n"
            "Common issues to look for and fix:\n"
            "- Route handlers returning placeholder strings instead of calling real handler functions\n"
            "- CSS class names in style modules not matching class names used in HTML-generating modules\n"
            "- Missing imports or function calls between modules that should be connected\n"
            "- Data format mismatches at module boundaries\n\n"
            "Guidelines:\n"
            "- Fix only what is needed to make end-to-end tests pass.\n"
            "- Read the actual source files to understand current implementation before changing.\n"
            "- Do NOT refactor working code. Only fix broken connections.\n"
            "- Do NOT create the project entry point \u2014 it will be created in the next task.\n"
            "- Do NOT edit test files at this stage.\n"
        )
    else:
        # Fallback
        prompt = (
            "You are working in an incremental development workflow.\n"
            "Tests may already exist or may be added later.\n"
            "Your responsibility in this step is to implement or refine production code only.\n"
            "Do NOT modify test files or environment/dependency configuration here.\n\n"
            "Task description:\n"
            f"{task}\n\n"
            f"Target: implement or refine units [{batch_units}] in {file_path}.\n\n"
            "Guidelines:\n"
            "- Implement behavior consistent with the task description and any existing tests.\n"
            "- Work incrementally: it is fine if not all tests pass yet, as long as your code moves toward correctness.\n"
            "- Prefer small, focused, maintainable changes.\n"
            "- Follow repository architecture, conventions, and abstractions.\n"
            "- Reuse helpers/utilities where possible; introduce small helpers only when justified.\n"
            "- Do NOT edit or create test files at this stage.\n"
        )
    
    # For implementation tasks, remind agent about the skeleton file
    if task_type == "implementation" and file_path:
        prompt += (
            "\n**Skeleton file:** The target file `" + file_path + "` may already contain "
            "interface definitions (signatures, docstrings) written during the design stage. "
            "Use them as your implementation starting point and fill in the function bodies.\n"
        )

    # NOTE: dependency_context is NOT appended here — provided once in TDD_BATCH_PREAMBLE.
    return prompt


def build_test_prompt_from_batch(
    batch: "PlannedTask",
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build test generation prompt from a PlannedTask object."""
    return init_test_gen_prompt(
        task=batch.task,
        batch_units=", ".join(batch.units_key),
        file_path=batch.file_path,
        task_type=batch.task_type,
        dependency_context=dependency_context,
    )


def build_code_prompt_from_batch(
    batch: "PlannedTask",
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build code generation prompt from a PlannedTask object."""
    return init_code_gen_prompt(
        task=batch.task,
        batch_units=", ".join(batch.units_key),
        file_path=batch.file_path,
        task_type=batch.task_type,
        dependency_context=dependency_context,
    )


# ============================================================================
# Merged File-Level Prompts
# ============================================================================

def _format_merged_phases(batches: list) -> str:
    """Format multiple batch tasks into numbered phases for merged prompts."""
    phases = []
    for i, batch in enumerate(batches, 1):
        units_str = ", ".join(batch.units_key)
        phases.append(
            f"### Phase {i}: [{units_str}]\n"
            f"{batch.task}"
        )
    return "\n\n".join(phases)


def build_merged_test_prompt(
    batches: list,
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a combined test generation prompt for multiple batches from the same file.
    
    Used in file-level merge mode when several tasks targeting the same file
    are implemented together.
    
    Args:
        batches: List of PlannedTask objects (all sharing the same file_path)
        dependency_context: Dependency context from earlier design stages
    """
    if len(batches) == 1:
        return build_test_prompt_from_batch(batches[0], dependency_context=dependency_context)
    
    file_path = batches[0].file_path
    all_units = []
    for b in batches:
        all_units.extend(b.units_key)
    all_units_str = ", ".join(all_units)
    
    phases_text = _format_merged_phases(batches)
    
    prompt = (
        "You are working in a Test-Driven Development (TDD) workflow.\n"
        "In this step your responsibility is ONLY to write or update tests.\n"
        "Do NOT modify production/source code and do NOT touch environment or dependency files.\n\n"
        
        f"**File-level batch:** You are implementing `{file_path}` — "
        f"covering the following units in one pass.\n"
        f"**Units in this batch:** [{all_units_str}]\n\n"
        
        "The implementation is organized into ordered phases (by dependency).\n"
        "Write tests that cover ALL phases below.\n\n"
        
        f"{phases_text}\n\n"
        
        "Requirements:\n"
        "- Use the repository's existing test layout and conventions.\n"
        "- Follow the same testing framework, helpers, fixtures, and style already used.\n"
        "- Cover at minimum: normal behavior, key edge cases, and meaningful failure cases.\n"
        "- Keep tests deterministic, readable, and maintainable.\n"
        "- Organize tests logically — you may group by phase or by functional area.\n"
        "- If the expected behavior is unclear, encode the most reasonable interpretation\n"
        "  and add comments explaining your assumptions.\n"
        "- **CRITICAL: Only test classes, methods, and functions that exist in the skeleton\n"
        "  file below (if provided). Do NOT invent or assume additional methods, features,\n"
        "  or APIs beyond what is defined in the skeleton.**\n"
    )
    
    # Point agent to skeleton file (read on demand, not inlined)
    if file_path:
        prompt += (
            f"\nThe skeleton file `{file_path}` contains interface definitions. "
            "**Read this file** for exact API signatures.\n"
        )
    # NOTE: dependency_context is NOT appended here — provided once in TDD_BATCH_PREAMBLE.
    return prompt


def build_merged_code_prompt(
    batches: list,
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a combined code generation prompt for multiple batches from the same file.
    
    Used in file-level merge mode when several tasks targeting the same file
    are implemented together.
    
    Args:
        batches: List of PlannedTask objects (all sharing the same file_path)
        dependency_context: Dependency context from earlier design stages
    """
    if len(batches) == 1:
        return build_code_prompt_from_batch(batches[0], dependency_context=dependency_context)
    
    file_path = batches[0].file_path
    all_units = []
    for b in batches:
        all_units.extend(b.units_key)
    all_units_str = ", ".join(all_units)
    
    phases_text = _format_merged_phases(batches)
    
    prompt = (
        "You are working in an incremental development workflow.\n"
        "Tests may already exist or may be added later.\n"
        "Your responsibility in this step is to implement or refine production code only.\n"
        "Do NOT modify test files or environment/dependency configuration here.\n\n"
        
        f"**File-level batch:** You are implementing `{file_path}` — "
        f"covering the following units in one pass.\n"
        f"**Units in this batch:** [{all_units_str}]\n\n"
        
        "The phases below are ordered by dependency — implement them in order.\n"
        "Earlier phases provide foundations that later phases depend on.\n\n"
        
        f"{phases_text}\n\n"
        
        "Guidelines:\n"
        "- Implement ALL phases listed above in the specified order.\n"
        "- Implement behavior consistent with the task descriptions and any existing tests.\n"
        "- Work incrementally within the file: foundational helpers first, then higher-level logic.\n"
        "- Prefer small, focused, maintainable implementations.\n"
        "- Follow repository architecture, conventions, and abstractions.\n"
        "- Reuse helpers/utilities where possible; introduce small helpers only when justified.\n"
        "- Do NOT edit or create test files at this stage.\n"
        "- Ensure all necessary imports are present at the top of the file.\n"
        "- Before adding new imports, search the repository for existing usage of similar helpers\n"
        "  and prefer the same modules and import style (to stay consistent with the codebase).\n"
    )
    
    # Remind about skeleton file
    if file_path:
        prompt += (
            f"\n**Skeleton file:** The target file `{file_path}` may already contain "
            "interface definitions (signatures, docstrings) written during the design stage. "
            "Use them as your implementation starting point and fill in the function bodies.\n"
        )

    # NOTE: dependency_context is NOT appended here — provided once in TDD_BATCH_PREAMBLE.
    return prompt


# ============================================================================
# Project File Prompts
# ============================================================================

def init_project_file_gen_prompt(
    task: str,
    batch_units: str,
    file_path: str,
    dependency_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate prompt for project file generation.
    
    This is used after all core implementation is complete.
    Project files include dependency manifests, README.md, entry points, etc.
    
    Args:
        task: Task description with detailed instructions
        batch_units: Comma-separated list of unit keys
        file_path: Target file marker (e.g., <REQUIREMENTS>, <README>)
        dependency_context: Dependency context from earlier design stages
    """
    prompt = (
        "You are working on Project Finalization.\n"
        "Your responsibility is to create project files that complete the repository.\n\n"
        
        "**Context:**\n"
        "All core implementation code has been completed.\n"
        "You now need to create the requested file(s) to make the repository complete and usable.\n\n"
        
        "**Important Guidelines:**\n"
        "1. Analyze the ACTUAL implemented code to generate accurate content\n"
        "2. Do NOT guess or assume - reference real module names, functions, and classes\n"
        "3. Ensure all examples and documentation are consistent with the codebase\n"
        "4. Follow standard conventions for each file type\n"
        "5. Read existing files in the repository to understand the structure\n\n"
        
        f"**Target files:** {batch_units}\n\n"
        
        f"**Task description:**\n{task}\n\n"
        
        "**Process:**\n"
        "1. First, explore the repository structure to understand what has been implemented\n"
        "2. Read key source files to understand imports, functions, and classes\n"
        "3. Generate the requested files based on your analysis\n"
        "4. Ensure all references are accurate and all examples are runnable\n"
    )

    prompt += _format_dependency_context(dependency_context)
    return prompt


def build_project_file_prompt_from_batch(
    batch: "PlannedTask",
    dependency_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build project file generation prompt from a PlannedTask object."""
    return init_project_file_gen_prompt(
        task=batch.task,
        batch_units=", ".join(batch.units_key),
        file_path=batch.file_path,
        dependency_context=dependency_context,
    )


def is_project_file_batch(batch: "PlannedTask") -> bool:
    """Check if a batch is for project file generation."""
    return batch.task_type in [
        "project_requirements",     # language dependency metadata
        "project_docs",             # README.md (no tests)
        "main_entry",               # language entry point (needs run test)
    ]


def is_project_docs_batch(batch: "PlannedTask") -> bool:
    """Check if a batch is for documentation files (no tests needed)."""
    return batch.task_type == "project_docs"


def needs_project_file_test(batch: "PlannedTask") -> bool:
    """Check if a project file batch needs testing."""
    return batch.task_type in [
        "project_requirements",  # dependency/import validation
        "main_entry",            # run test
    ]


# ============================================================================
# Iterative Prompts (After Failure)
# ============================================================================

def test_fix_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """Generate iterative test regeneration prompt based on failing tests.

    Used when failure_type == TEST_ERROR.
    """
    prompt = (
        "You are now in the TEST FIX phase.\n"
        "Your responsibility is to correct and improve the TEST CODE only.\n"
        "Assume production code is mostly correct for now.\n\n"
        "Task context:\n"
        f"{task}\n\n"
        "Test failures:\n"
        f"{test_result}\n\n"
        "Your job:\n"
        "- Analyze why the tests fail.\n"
        "- Fix assertions, setups, fixtures, imports, or test logic when they are incorrect.\n"
        "- Ensure the tests describe intended behavior clearly and consistently.\n\n"
        "Rules:\n"
        "- Modify ONLY test-related files.\n"
        "- Do NOT change production code or environment configuration.\n"
        "- Keep tests deterministic and meaningful.\n"
    )
    return prompt


def code_fix_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """Generate iterative code regeneration prompt based on failing tests.

    Used when failure_type == CODE_ERROR.
    """
    prompt = (
        "You are now in the CODE FIX phase.\n"
        "Your responsibility is to fix bugs in production code.\n"
        "Do NOT modify tests or environment configuration here.\n\n"
        "Task context:\n"
        f"{task}\n\n"
        "Test output:\n"
        f"{test_result}\n\n"
        "Your job:\n"
        "- Understand what behavior the failing test expects.\n"
        "- Identify the real root cause in the code.\n"
        "- Apply a minimal, targeted fix aligned with existing architecture.\n\n"
        "Rules:\n"
        "- Modify ONLY production code modules.\n"
        "- Prefer the smallest correct change.\n"
        "- Avoid unnecessary refactors or public API breaks unless unavoidable.\n"
    )
    return prompt


def env_fix_prompt(
    test_result: str,
    task: str,
    **kwargs
) -> str:
    """Generate environment setup prompt based on environment-related failures.

    Used when failure_type == ENV_ERROR.
    """
    prompt = (
        "You are now in the ENVIRONMENT FIX phase.\n"
        "The test failure is caused by importing a third-party package that is not installed.\n\n"
        "Context:\n"
        f"{task}\n\n"
        "Evidence of environment failure:\n"
        f"{test_result}\n\n"
        "Guidelines:\n"
        "- Fix by REMOVING the unused import if the imported name is not actually used in the code,\n"
        "  OR by replacing the third-party functionality with target-language standard library equivalents.\n"
        "- Search the source file for actual usage of the imported name before deciding.\n"
        "- Do NOT attempt to install packages or modify dependency manifests.\n"
        "- Do NOT modify test files.\n"
        "- Prefer minimal, targeted changes.\n"
        "- Logical test failures may remain — that is acceptable.\n"
    )
    return prompt


# ============================================================================
# Failure Analysis Prompt
# ============================================================================

FAILURE_ANALYSIS_PROMPT = """
You are a test failure analysis expert.

Analyze the following test failure. You have FULL access to the test file,
source file, and test output. Your job is to determine the root cause and
produce a concrete fix plan.

## Test Execution Output:
{test_output}

## Source File ({source_file}):
```python
{source_content}
```

## Test File ({test_file}):
```python
{test_content}
```

## Task Context:
{task_context}

## Failure History (previous iterations):
{failure_history}

---

### Step 1: Classify the error

Choose ONE of these categories:
- ENV_ERROR        : Environment issue (missing package, import of uninstalled third-party lib)
- TEST_ERROR       : Only the test code is wrong (wrong assertion, bad fixture, wrong API usage in test)
- CODE_ERROR       : Only the production code is wrong (logic bug, missing method, wrong return value)
- BOTH_ERROR       : Both test AND code have issues that need fixing together

### Step 2: Determine fix_target

Based on the category, choose the fix path:
- "test"            : Only test code needs changes  (for TEST_ERROR)
- "code"            : Only production code needs changes  (for CODE_ERROR)
- "code_then_test"  : Fix code first, then fix tests  (for BOTH_ERROR)
- "env"             : Fix environment/import issue in code  (for ENV_ERROR)

### Step 3: Write a concrete fix plan

For EACH side that needs fixing, describe the SPECIFIC changes needed.
Reference exact function names, line numbers, variable names.
Do NOT write vague instructions like "fix the bug" or "update the test".

### Output Format

Return a JSON object with these fields:

```json
{{
  "category": "CODE_ERROR | TEST_ERROR | ENV_ERROR | BOTH_ERROR",
  "fix_target": "code | test | code_then_test | env",
  "root_cause": "One sentence: the precise technical reason for the failure",
  "fix_plan": {{
    "code_changes": "Specific changes to make in the source file (or null if not needed)",
    "test_changes": "Specific changes to make in the test file (or null if not needed)"
  }},
  "reasoning": "2-3 sentences explaining why you chose this category and fix_target"
}}
```

### Decision Guidelines

- If the test expects behavior X but the code does Y, ask: **which one matches the
  task description / spec?** The one that matches the spec is correct; fix the other.
- If BOTH the test assertion AND the code logic are wrong relative to the spec,
  choose BOTH_ERROR with fix_target "code_then_test".
- If the test uses an API that doesn't exist in the source, check: was the API
  supposed to exist (per the spec)? If yes → CODE_ERROR. If no → TEST_ERROR.
- Prefer CODE_ERROR over TEST_ERROR when the spec is ambiguous — tests represent
  the intended behavior.
- **Mock/patch bugs**: When tests use `@patch`, `MagicMock`, or `side_effect`,
  carefully trace whether the mock setup matches the ACTUAL call sequence in the
  production code. Common test bugs include:
  - `side_effect` list has wrong number of values (too many or too few for the
    actual number of calls the patched function receives)
  - Mock return values don't account for internal helper calls that also invoke
    the patched function
  - If the production code logic is clearly correct but the assertion fails,
    check whether the mock values fed to the code actually produce the expected
    result — the mock setup itself may be wrong → TEST_ERROR
- **ENV_ERROR sub-types**: ENV_ERROR covers three distinct situations.
  Your `fix_plan.code_changes` or `fix_plan.test_changes` MUST specify the
  exact fix — do NOT give vague instructions like "fix the import".
  - **Missing import in source**: `NameError: name 'Enum' is not defined` means
    the source file uses a name without importing it. Fix: add the correct
    import statement (e.g. `from enum import Enum`). fix_target = "code".
  - **Wrong import path**: `ModuleNotFoundError: No module named 'vibeanim'`
    in a project that uses `src.vibeanim.*` means the import path is wrong.
    Fix: change `from vibeanim.x` to `from src.vibeanim.x`. If the error is
    in a test file, fix_target = "test"; if in source, fix_target = "code".
  - **Missing third-party package**: `ModuleNotFoundError` for a non-project
    module means a package is not installed. Fix: remove the import or replace
    with stdlib equivalents. fix_target = "code".
  - Do NOT classify logic errors (AssertionError, TypeError, ValueError) as
    ENV_ERROR — those are CODE_ERROR or TEST_ERROR.
- Look at failure_history: if previous iterations alternated between TEST_ERROR
  and CODE_ERROR, this strongly suggests BOTH_ERROR.
- **Persistent same-error pattern**: If failure_history shows 2+ consecutive
  CODE_ERROR iterations with the same test still failing, seriously consider
  whether the TEST is actually wrong (mock setup, wrong expected value, etc.).
  Repeated code fixes that don't resolve the issue are a strong signal that
  the root cause is in the test, not the code.

### Examples

Example 1 (CODE_ERROR):
{{
  "category": "CODE_ERROR",
  "fix_target": "code",
  "root_cause": "loop(None) sets _loop_count=None which is the same as the default, so build() cannot distinguish 'never called' from 'infinite loop'",
  "fix_plan": {{
    "code_changes": "Add a `_loop_enabled: bool = False` flag to EvolutionSequenceBuilder.__init__. Set it to True in loop(). Use `loop=self._loop_enabled` in build() instead of `loop=(self._loop_count is not None)`. Reset it in clear().",
    "test_changes": null
  }},
  "reasoning": "The test correctly expects loop(None) to produce loop=True per the spec. The code has a sentinel value collision — _loop_count defaults to None and loop(None) also sets it to None."
}}

Example 2 (BOTH_ERROR):
{{
  "category": "BOTH_ERROR",
  "fix_target": "code_then_test",
  "root_cause": "The code returns a list instead of a tuple, AND the test compares against a hardcoded wrong expected value",
  "fix_plan": {{
    "code_changes": "In transform(), change `return [x, y, z]` to `return (x, y, z)` to match the documented return type",
    "test_changes": "In test_transform_origin(), change expected value from (0, 0, 1) to (0, 0, 0) which is the correct origin transform"
  }},
  "reasoning": "The code has a type error (list vs tuple) and the test has a wrong expected value. Both need fixing. Previous iterations alternated between TEST_ERROR and CODE_ERROR, confirming both sides have issues."
}}

Example 3 (TEST_ERROR):
{{
  "category": "TEST_ERROR",
  "fix_target": "test",
  "root_cause": "Test calls entity.get_position() but the API is entity.position (a property, not a method)",
  "fix_plan": {{
    "code_changes": null,
    "test_changes": "Replace all calls to `entity.get_position()` with `entity.position` in test_entity_movement.py (lines 45, 67, 89)"
  }},
  "reasoning": "The source code correctly implements position as a property per the skeleton. The test was generated with a wrong API assumption."
}}
"""


def build_failure_analysis_prompt(
    test_output: str,
    task_context: str,
    source_file: str = "",
    source_content: str = "",
    test_file: str = "",
    test_content: str = "",
    failure_history: str = "",
    max_output_length: int = 3000,
    # Legacy params (kept for backward compat, ignored)
    test_patch: str = "",
    code_patch: str = "",
) -> str:
    """Build the failure analysis prompt with full file context.

    Args:
        test_output: Output from test execution
        task_context: Context about the current task
        source_file: Path to the source file
        source_content: Full content of the source file
        test_file: Path to the test file
        test_content: Full content of the test file
        failure_history: Formatted string of previous failure types
        max_output_length: Maximum length of test output to include
    """
    # Truncate test output if too long
    if len(test_output) > max_output_length:
        test_output = test_output[:max_output_length] + "\n\n... (truncated)"

    # Truncate file contents if too long (keep enough for analysis)
    max_file = 8000
    if len(source_content) > max_file:
        source_content = source_content[:max_file] + "\n# ... (truncated)"
    if len(test_content) > max_file:
        test_content = test_content[:max_file] + "\n# ... (truncated)"

    return FAILURE_ANALYSIS_PROMPT.format(
        test_output=test_output,
        task_context=task_context,
        source_file=source_file or "(unknown)",
        source_content=source_content or "(not available)",
        test_file=test_file or "(unknown)",
        test_content=test_content or "(not available)",
        failure_history=failure_history or "(first iteration)",
    )


# ============================================================================
# Commit Message Prompt
# ============================================================================

COMMIT_MESSAGE_PROMPT = """You are an assistant that writes concise, conventional git commit messages.

You will be given:
- Workflow type (one of: TEST_DEVELOPMENT, TEST_FIX, CODE_INCREMENTAL, CODE_BUG_FIX, ENV_SETUP, or legacy types)
- A unified diff patch
- Patch statistics (lines changed, files changed)
- File path
- Units (logical units or sections implemented)
- A task description

Your job:
1. Decide a good conventional-commit style SUBJECT line based on workflow type:
   - TEST_DEVELOPMENT: Use "test:" prefix for new test creation
   - TEST_FIX: Use "test:" or "fix:" prefix for fixing broken tests
   - CODE_INCREMENTAL: Use "feat:" prefix for new feature implementation
   - CODE_BUG_FIX: Use "fix:" prefix for bug fixes and corrections
   - ENV_SETUP: Use "chore:" prefix for environment/dependency setup
   - Keep it concise (ideally <= 60 characters).
   - No trailing period.
2. Optionally produce a BODY with several lines (each line just plain text, no bullets).
   - You can include information about units, file, lines changed, and task summary.
   - Each item should be a separate string in an array (e.g., ["Units: ...", "File: ..."]).

Return ONLY a JSON object with:
- "subject": string
- "body": either a string or an array of strings. If you don't want a body, you can return an empty string or an empty array.

Examples:
{{
  "subject": "feat: implement user authentication system",
  "body": [
    "Units: login_handler, auth_validator",
    "File: src/auth/auth.py", 
    "Changed: 67 lines in 1 file",
    "Task: implement OAuth2 login flow for new users"
  ]
}}

{{
  "subject": "fix: resolve memory leak in data processor",
  "body": [
    "Units: process_batch, cleanup_resources",
    "File: src/processing/processor.py",
    "Changed: 23 lines in 1 file", 
    "Task: fix memory allocation issues in batch processing"
  ]
}}

{{
  "subject": "test: add unit tests for payment module",
  "body": ""
}}

Now generate a commit message for the following context:

Workflow type: {workflow_type}
Lines changed: {lines_changed}
Files changed: {files_changed}
File path: {file_path}
Units: {units}
Task description: {task_desc}

Unified diff:
{patch_content}
"""


def build_commit_message_prompt(
    workflow_type: str,
    file_path: str,
    units: str,
    task_desc: str,
    patch_content: str = "",
    lines_changed: int = 0,
    files_changed: int = 0
) -> str:
    """Build a prompt for generating commit messages."""
    # Truncate patch content if too long
    if len(patch_content) > 2000:
        patch_content = patch_content[:2000] + "\n... (truncated)"
    
    return COMMIT_MESSAGE_PROMPT.format(
        workflow_type=workflow_type,
        lines_changed=lines_changed,
        files_changed=files_changed,
        file_path=file_path,
        units=units,
        task_desc=task_desc,
        patch_content=patch_content or "(no patch provided)"
    )


def generate_simple_commit_message(
    workflow_type: str,
    file_path: str,
    units: str,
    task: str
) -> str:
    """Generate a simple commit message without LLM.
    
    Used as fallback when LLM is not available.
    """
    prefixes = {
        "test_development": "test",
        "test_fix": "fix(test)",
        "code_incremental": "feat",
        "code_bug_fix": "fix",
        "env_setup": "chore",
    }
    
    prefix = prefixes.get(workflow_type.lower(), "chore")
    
    # Extract filename from path
    filename = file_path.split("/")[-1] if "/" in file_path else file_path
    
    # Truncate task description
    short_task = task[:50] + "..." if len(task) > 50 else task
    
    return f"{prefix}: {filename} - {short_task}"
