"""Semantic Parsing Module.

Extracts semantic features from Python source code using LLM-based analysis.
The core class ``ParseFeatures`` orchestrates the full pipeline:

1. Parse Python files into CodeUnit groups (classes + methods, functions)
2. Batch groups by token budget for efficient LLM usage
3. Call LLM with PARSE_CLASS / PARSE_FUNCTION prompts
4. Generate file-level summaries
5. Deduplicate summaries

Ported from RPG-ZeroRepo ``zerorepo/rpg_encoder/rpg_parsing/semantic_parsing.py``
with the following adaptations for CoderMind:
- Uses ``LLMClient`` from ``scripts.common.llm_client``
- Uses ``Memory`` / message types from ``scripts.common.llm_types``
- Uses utility functions from ``scripts.common.utils``
- Uses ``CodeUnit`` / ``ParsedFile`` / ``CodeSnippetBuilder`` from
  ``scripts.skeleton.code_unit``
"""

import concurrent.futures
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import json5

from common.llm_client import LLMClient
from common.llm_types import (
    AssistantMessage,
    Memory,
    SystemMessage,
    UserMessage,
)
from common.utils import (
    calculate_tokens,
    filter_excluded_files,
    normalize_path,
    parse_solution_output,
)
from lang_parser import is_supported_source, is_test_file
from rpg.code_unit import CodeSnippetBuilder, CodeUnit, ParsedFile
from rpg.path_format import (
    desc_key_class as _desc_key_class,
    desc_key_function as _desc_key_function,
    desc_key_method as _desc_key_method,
)

from .prompts import PARSE_CLASS, PARSE_FUNCTION


# ---------------------------------------------------------------------------
# Helper: build lightweight unit summaries for agent-based prompts
# ---------------------------------------------------------------------------

def _extract_def_line(code: str, lineno: int) -> str:
    """Extract the ``def ...`` line(s) from source at *lineno* (1-based).

    Handles multi-line signatures by collecting continuation lines
    up to the closing ``):``.
    """
    lines = code.splitlines()
    if lineno < 1 or lineno > len(lines):
        return ""
    parts = [lines[lineno - 1].strip()]
    # Collect continuation lines for multi-line signatures
    if ")" not in parts[0]:
        for i in range(lineno, min(lineno + 5, len(lines))):
            part = lines[i].strip()
            parts.append(part)
            if ")" in part:
                break
    return " ".join(parts)


def _build_function_summaries(
    func_units: List[CodeUnit],
    file_code_map: Dict[str, str],
) -> Dict[str, List[str]]:
    """Build ``{file_path: [summary_line, ...]}`` for function units.

    Each summary line looks like:
    ``- _parse_positive_int(raw_value: str, default: int) -> int  # Return a positive integer``
    """
    by_file: Dict[str, List[str]] = defaultdict(list)

    for u in func_units:
        if u.unit_type != "function":
            continue
        code = file_code_map.get(u.file_path, "")
        if code and u.lineno:
            def_line = _extract_def_line(code, u.lineno)
            # Strip "def " prefix for cleaner display
            if def_line.startswith("def "):
                def_line = def_line[4:]
        else:
            def_line = f"{u.name}()"

        doc = getattr(u, "docstring", None) or ""
        doc_short = doc.split("\n")[0].strip()[:80] if doc else ""
        summary = f"- `{def_line}`"
        if doc_short:
            summary += f"  # {doc_short}"
        by_file[u.file_path].append(summary)

    return dict(by_file)


def _build_class_summaries(
    cls_units: List[CodeUnit],
    file_code_map: Dict[str, str],
) -> Dict[str, List[str]]:
    """Build ``{file_path: [summary_block, ...]}`` for class units.

    Each class gets a block like::

        - class `TaxonomyAdminManager`
            methods: get_admin_view(), rename_category(name), ...
    """
    by_file: Dict[str, List[str]] = defaultdict(list)

    # Group methods by (file_path, class_name) to avoid cross-file mixing
    class_methods: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for u in cls_units:
        if u.unit_type == "method" and u.parent:
            class_methods[(u.file_path, u.parent)].append(u.name)

    for u in cls_units:
        if u.unit_type != "class":
            continue
        methods = class_methods.get((u.file_path, u.name), [])
        methods_str = ", ".join(f"{m}()" for m in methods) if methods else "(no methods)"
        doc = getattr(u, "docstring", None) or ""
        doc_short = doc.split("\n")[0].strip()[:80] if doc else ""
        line = f"- class `{u.name}`"
        if doc_short:
            line += f"  # {doc_short}"
        line += f"\n    methods: {methods_str}"
        by_file[u.file_path].append(line)

    return dict(by_file)


# ---------------------------------------------------------------------------
# Helpers: split feature names / descriptions
# ---------------------------------------------------------------------------
#
# Composite description-map key helpers (``_desc_key_function`` /
# ``_desc_key_class`` / ``_desc_key_method``) are imported from
# :mod:`rpg.path_format` so producers (here) and consumers
# (``rpg.models.RPG.update_from_parsed_tree`` /
# ``refactor_tree._init_feature_tree``) share one source of truth.

def _split_features(value: Any) -> Tuple[List[str], Dict[str, str]]:
    """Extract ``(names, desc_map)`` from an LLM-returned feature container.

    Tolerates both schemas so legacy prompt outputs still parse:

    - **New** ``{"feat_name": "description", ...}`` -> names = keys,
      descs = full mapping.
    - **Legacy** ``["feat_name", ...]`` -> names = list, descs = ``{}``.
    - Anything else -> ``([], {})``.

    Non-string description values are coerced to ``""`` (LLM hallucination
    guard).
    """
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)], {}
    if isinstance(value, dict):
        names: List[str] = []
        descs: Dict[str, str] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                continue
            names.append(k)
            descs[k] = v if isinstance(v, str) else ""
        return names, descs
    return [], {}


logger = logging.getLogger(__name__)


class ParseFeatures:
    """Extract semantic features from Python source code via LLM analysis.

    This is the main entry-point for M6 — Semantic Parsing.  It mirrors the
    ``ParseFeatures`` class in RPG-ZeroRepo with minimal interface changes.

    Typical usage::

        parser = ParseFeatures(
            repo_dir="/path/to/repo",
            repo_info="A short description of the repo",
            repo_skeleton="<skeleton string>",
            valid_files=["src/main.py", "src/utils.py"],
            repo_name="my-project",
        )
        features, trajectories = parser.parse_repo()

    Source: RPG-ZeroRepo ``semantic_parsing.py`` :class:`ParseFeatures`
    """

    def __init__(
        self,
        repo_dir: str,
        repo_info: str,
        repo_skeleton: str,
        valid_files: List[str],
        repo_name: str,
        logger: Optional[logging.Logger] = None,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.repo_dir = repo_dir
        self.repo_info = repo_info
        self.repo_skeleton = repo_skeleton
        self.valid_files = valid_files
        self.repo_name = repo_name

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"ParseFeatures[{repo_name}]")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                )
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        # Use shared LLM client if provided, otherwise create a new one
        self.llm_client = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # File-summary deduplication
    # ------------------------------------------------------------------

    def _dedupe_file_summaries(
        self,
        repo_feature_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Ensure every file has a unique ``_file_summary_`` value.

        Uses a global set to track already-claimed summaries, appending a
        numeric suffix when collisions occur.

        Source: ZeroRepo ``semantic_parsing.py`` :meth:`_dedupe_file_summaries`
        """

        def _clean_text(s: str) -> str:
            if not isinstance(s, str):
                return str(s) if s is not None else ""
            s = s.replace("/", "_")
            s = " ".join(s.split())
            return s.strip()

        for path, fmap in repo_feature_map.items():
            if "_file_summary_" in fmap:
                fmap["_file_summary_"] = _clean_text(fmap["_file_summary_"])
            else:
                fmap["_file_summary_"] = _clean_text(
                    os.path.basename(path).replace(".py", "")
                )

        used_summaries: set = set()
        sorted_paths = sorted(repo_feature_map.keys())

        for path in sorted_paths:
            fmap = repo_feature_map[path]
            original_summary = fmap.get("_file_summary_")
            if not original_summary:
                continue

            current_summary = original_summary
            if current_summary in used_summaries:
                counter = 1
                while True:
                    candidate = f"{original_summary}_{counter}"
                    if candidate not in used_summaries:
                        current_summary = candidate
                        break
                    counter += 1

            fmap["_file_summary_"] = current_summary
            used_summaries.add(current_summary)

        return repo_feature_map

    # ------------------------------------------------------------------
    # Core global parsing strategy
    # ------------------------------------------------------------------

    def _parse_files_global(
        self,
        file_code_map: Dict[str, str],
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 1,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict]]:
        """Global parsing strategy.

        Steps:

        1. Parse all files into CodeUnits.
        2. Group classes (class + methods) and functions separately.
        3. Batch groups by token budget.
        4. Call ``parse_classes`` / ``parse_functions`` in parallel.
        5. Re-map results back by file.
        6. Generate file-level summaries in parallel.
        7. Deduplicate summaries.

        Source: ZeroRepo ``semantic_parsing.py`` :meth:`_parse_files_global`
        """
        # --- Step 1: Parse all files ---
        parsed_files: Dict[str, ParsedFile] = {}
        file_units: Dict[str, List[CodeUnit]] = {}

        for path, code in file_code_map.items():
            parsed = ParsedFile(code=code, file_path=path)
            parsed_files[path] = parsed
            file_units[path] = parsed.units

        code_builder = CodeSnippetBuilder(
            file_code_map=file_code_map,
            parsed_files=parsed_files,
        )

        # --- Step 2: Group classes and functions ---
        class_groups: List[List[CodeUnit]] = []
        func_groups: List[List[CodeUnit]] = []

        for path, units in file_units.items():
            cls_units = [u for u in units if u.unit_type == "class"]
            mtd_units = [u for u in units if u.unit_type == "method"]
            fn_units = [u for u in units if u.unit_type == "function"]

            for cls in cls_units:
                group = [cls] + [m for m in mtd_units if m.parent == cls.name]
                class_groups.append(group)

            for fn in fn_units:
                func_groups.append([fn])

        self.logger.info(
            "[GLOBAL] Total class groups: %d, function groups: %d",
            len(class_groups),
            len(func_groups),
        )

        # --- Helper closures ---
        def _units_code(units: List[CodeUnit]) -> str:
            parts = []
            for u in units:
                try:
                    parts.append(u.unparse())
                except Exception:
                    continue
            return "\n\n".join(parts)

        def _units_tokens(units: List[CodeUnit]) -> int:
            code = _units_code(units)
            if not code.strip():
                return 0
            return calculate_tokens(code)

        # --- Step 3: Batch groups by token budget ---
        def _make_token_batches(
            groups: List[List[CodeUnit]], kind: str
        ) -> List[List[CodeUnit]]:
            batches: List[List[CodeUnit]] = []
            cur: List[CodeUnit] = []
            cur_tokens = 0

            for g in groups:
                g_tokens = _units_tokens(g)

                if g_tokens > max_batch_tokens:
                    self.logger.warning(
                        "[GLOBAL] %s group starting at %s exceeds "
                        "max_batch_tokens=%d, tokens=%d, sending it alone.",
                        kind,
                        g[0].name if g else "<unknown>",
                        max_batch_tokens,
                        g_tokens,
                    )
                    if cur:
                        batches.append(cur)
                        cur = []
                        cur_tokens = 0
                    batches.append(list(g))
                    continue

                if cur and cur_tokens + g_tokens > max_batch_tokens:
                    batches.append(cur)
                    cur = list(g)
                    cur_tokens = g_tokens
                else:
                    cur.extend(g)
                    cur_tokens += g_tokens

            if cur:
                batches.append(cur)

            if len(batches) > 1:
                last_tokens = _units_tokens(batches[-1])
                if last_tokens < min_batch_tokens:
                    self.logger.info(
                        "[GLOBAL] %s last batch tokens=%d < "
                        "min_batch_tokens=%d, merging with previous batch.",
                        kind,
                        last_tokens,
                        min_batch_tokens,
                    )
                    batches[-2].extend(batches[-1])
                    batches.pop()

            self.logger.info(
                "[GLOBAL] kind=%s, groups=%d, batches=%d, "
                "min_batch_tokens=%d, max_batch_tokens=%d",
                kind,
                len(groups),
                len(batches),
                min_batch_tokens,
                max_batch_tokens,
            )

            for idx, batch in enumerate(batches):
                batch_tokens = _units_tokens(batch)
                if kind == "class":
                    names = [u.name for u in batch if u.unit_type == "class"]
                else:
                    names = [u.name for u in batch if u.unit_type == "function"]

                preview = names[:20]
                self.logger.info(
                    "[GLOBAL] %s batch #%d: units=%d, tokens=%d, %s=%s",
                    kind,
                    idx,
                    len(batch),
                    batch_tokens,
                    "classes" if kind == "class" else "functions",
                    preview,
                )
            return batches

        class_batches = _make_token_batches(class_groups, "class")
        func_batches = _make_token_batches(func_groups, "function")

        global_feature_map: Dict[str, Any] = {}
        # Composite-key sidecar accumulated across batches.
        # Keys produced by ``_desc_key_function`` / ``_desc_key_class`` /
        # ``_desc_key_method`` (i.e. ``"{unit}::{...}::{feat}"``).
        global_descriptions: Dict[str, str] = {}
        all_trajectories: List[Dict] = []

        # --- Step 4: Process batches in parallel ---
        def process_class_batch(
            batch_units: List[CodeUnit],
        ) -> Tuple[Dict[str, Any], Dict[str, str]]:
            try:
                batch_class_names = [
                    u.name for u in batch_units if u.unit_type == "class"
                ]
                self.logger.info(
                    "[GLOBAL] process_class_batch: classes=%s, units=%d",
                    batch_class_names[:20],
                    len(batch_units),
                )

                cls_features, cls_descs, cls_msgs = self.parse_classes(
                    code_builder=code_builder,
                    cls_units=batch_units,
                    context_window=class_context_window,
                    max_iterations=max_iterations,
                )
                all_trajectories.append(
                    {
                        "type": "class",
                        "chunk_names": list(cls_features.keys()),
                        "messages": cls_msgs,
                    }
                )
                return (
                    {
                        f"class {name}": feats
                        for name, feats in cls_features.items()
                    },
                    dict(cls_descs),
                )
            except Exception as e:
                self.logger.error(
                    "[GLOBAL] process_class_batch error: %s", e, exc_info=True
                )
                return {}, {}
            finally:
                self.logger.info(
                    "[GLOBAL] finished class batch with %d units",
                    len(batch_units),
                )

        def process_func_batch(
            batch_units: List[CodeUnit],
        ) -> Tuple[Dict[str, Any], Dict[str, str]]:
            try:
                batch_func_names = [
                    u.name for u in batch_units if u.unit_type == "function"
                ]
                self.logger.info(
                    "[GLOBAL] process_func_batch: functions=%s, units=%d",
                    batch_func_names[:20],
                    len(batch_units),
                )

                func_features, func_descs, func_msgs = self.parse_functions(
                    code_builder=code_builder,
                    func_units=batch_units,
                    context_window=func_context_window,
                    max_iterations=max_iterations,
                )
                all_trajectories.append(
                    {
                        "type": "function",
                        "chunk_names": list(func_features.keys()),
                        "messages": func_msgs,
                    }
                )
                return (
                    {
                        f"function {name}": feats
                        for name, feats in func_features.items()
                    },
                    dict(func_descs),
                )
            except Exception as e:
                self.logger.error(
                    "[GLOBAL] process_func_batch error: %s", e, exc_info=True
                )
                return {}, {}
            finally:
                self.logger.info(
                    "[GLOBAL] finished function batch with %d units",
                    len(batch_units),
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for b in class_batches:
                if b:
                    futures.append(executor.submit(process_class_batch, b))
            for b in func_batches:
                if b:
                    futures.append(executor.submit(process_func_batch, b))

            for future in as_completed(futures):
                try:
                    feat_result, desc_result = future.result(timeout=600)
                    global_feature_map.update(feat_result)
                    global_descriptions.update(desc_result)
                except concurrent.futures.TimeoutError:
                    self.logger.error("LLM request timed out.")
                except Exception as e:
                    self.logger.error("Error in worker: %s", e, exc_info=True)

        # --- Step 5: Re-map by file ---
        # Main feature map is keyed by per-file unit identity; descriptions
        # are filtered into a sibling ``_feature_descriptions_`` map using
        # composite keys (``"{unit}::..."``).  We bucket descs once by their
        # leading unit-name segment so per-file lookup is O(units) instead of
        # O(units * total_descs) — important for large repos.
        repo_feature_map: Dict[str, Dict[str, Any]] = {
            path: {} for path in file_code_map.keys()
        }

        descs_by_unit: Dict[str, Dict[str, str]] = defaultdict(dict)
        for dk, dv in global_descriptions.items():
            head, _, _ = dk.partition("::")
            if head:
                descs_by_unit[head][dk] = dv

        for path, units in file_units.items():
            file_map: Dict[str, Any] = {}
            file_descs: Dict[str, str] = {}
            for u in units:
                if u.unit_type == "class":
                    key = f"class {u.name}"
                    if key in global_feature_map:
                        file_map[key] = global_feature_map[key]
                    file_descs.update(descs_by_unit.get(u.name, {}))
                elif u.unit_type == "function":
                    key = f"function {u.name}"
                    if key in global_feature_map:
                        file_map[key] = global_feature_map[key]
                    file_descs.update(descs_by_unit.get(u.name, {}))
            self.logger.info(
                "[GLOBAL] file=%s, mapped classes+funcs=%s, descs=%d",
                path,
                list(file_map.keys())[:20],
                len(file_descs),
            )
            repo_feature_map[path] = file_map
            if file_descs:
                repo_feature_map[path]["_feature_descriptions_"] = file_descs

        # --- Step 6: File-level summaries ---
        def _make_file_summary_batches(
            file_feature_items: List[Tuple[str, Dict[str, Any]]],
        ) -> List[List[Tuple[str, Dict[str, Any]]]]:
            batches: List[List[Tuple[str, Dict[str, Any]]]] = []
            cur: List[Tuple[str, Dict[str, Any]]] = []
            cur_tokens = 0

            for path, feature_map in file_feature_items:
                item_str = json.dumps(
                    {path: feature_map}, ensure_ascii=False
                )
                item_tokens = calculate_tokens(item_str)

                if item_tokens > summary_max_batch_tokens:
                    self.logger.warning(
                        "[SUMMARY] file=%s exceeds "
                        "summary_max_batch_tokens=%d, tokens=%d, "
                        "sending it alone.",
                        path,
                        summary_max_batch_tokens,
                        item_tokens,
                    )
                    if cur:
                        batches.append(cur)
                        cur = []
                        cur_tokens = 0
                    batches.append([(path, feature_map)])
                    continue

                if cur and cur_tokens + item_tokens > summary_max_batch_tokens:
                    batches.append(cur)
                    cur = [(path, feature_map)]
                    cur_tokens = item_tokens
                else:
                    cur.append((path, feature_map))
                    cur_tokens += item_tokens

            if cur:
                batches.append(cur)

            if len(batches) > 1:
                last_batch_str = json.dumps(
                    dict(batches[-1]), ensure_ascii=False
                )
                last_tokens = calculate_tokens(last_batch_str)
                if last_tokens < summary_min_batch_tokens:
                    self.logger.info(
                        "[SUMMARY] last batch tokens=%d < "
                        "summary_min_batch_tokens=%d, "
                        "merging with previous batch.",
                        last_tokens,
                        summary_min_batch_tokens,
                    )
                    batches[-2].extend(batches[-1])
                    batches.pop()

            self.logger.info(
                "[SUMMARY] total files=%d, batches=%d",
                len(file_feature_items),
                len(batches),
            )
            for idx, batch in enumerate(batches):
                batch_str = json.dumps(dict(batch), ensure_ascii=False)
                batch_tokens = calculate_tokens(batch_str)
                file_names = [os.path.basename(p) for p, _ in batch][:10]
                self.logger.info(
                    "[SUMMARY] batch #%d: files=%d, tokens=%d, preview=%s",
                    idx,
                    len(batch),
                    batch_tokens,
                    file_names,
                )
            return batches

        def summarize_file_batch(
            batch_items: List[Tuple[str, Dict[str, Any]]],
            context_window: int = 5,
            summary_max_iters: int = 3,
        ) -> Tuple[Dict[str, str], List[Dict]]:
            local_trajs: List[Dict] = []
            summaries: Dict[str, str] = {}
            file_paths = [p for p, _ in batch_items]

            files_info: Dict[str, Any] = {}
            for path, feature_map in batch_items:
                # Strip sidecar keys (descriptions, etc.) — the file-summary
                # prompt only needs the feature name structure, not the
                # potentially verbose descriptions.
                files_info[path] = {
                    k: v
                    for k, v in feature_map.items()
                    if k != "_feature_descriptions_"
                }

            batch_prompt = (
                "You are analyzing multiple Python files. For each file, "
                "summarize its **main functional purpose** in one concise "
                "descriptive phrase (e.g., 'data preprocessing utilities', "
                "'API routing layer').\n\n"
                "### Files to analyze:\n"
                f"```json\n{json.dumps(files_info, indent=2, ensure_ascii=False)}\n```\n\n"
                "### Feature Naming Rules\n"
                '1. Use the "verb + object" format '
                "(e.g., `load config`, `validate token`)\n"
                "2. Use lowercase English only\n"
                "3. Describe purpose, not implementation\n"
                "4. Avoid vague verbs like `handle`, `process`, `deal with`\n"
                "5. Avoid implementation details and specific "
                "libraries/frameworks\n\n"
                "Return a JSON object mapping each file path to its summary, "
                "wrapped in <solution>...</solution>:\n"
                "<solution>\n"
                "{\n"
                '  "<file_path_1>": "<summary_1>",\n'
                '  "<file_path_2>": "<summary_2>",\n'
                "  ...\n"
                "}\n"
                "</solution>\n"
            )

            memory = Memory(context_window=context_window)
            memory.add_message(
                SystemMessage("You are a precise code summarization assistant.")
            )
            memory.add_message(UserMessage(batch_prompt))

            self.logger.info(
                "[SUMMARY] processing batch with %d files: %s",
                len(batch_items),
                [os.path.basename(p) for p, _ in batch_items][:10],
            )

            for i in range(summary_max_iters):
                try:
                    response = self.llm_client.generate_with_memory(memory)
                    if response is None:
                        self.logger.error(
                            "[SUMMARY] LLM returned None at iteration %d",
                            i + 1,
                        )
                        continue
                    self.logger.info("[SUMMARY] Response: %s...", response[:200])
                    memory.add_message(AssistantMessage(response))

                    parsed_response = (
                        parse_solution_output(response)
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    parsed_json = json5.loads(parsed_response)

                    for path in file_paths:
                        if path in parsed_json:
                            summary = parsed_json[path]
                            if summary:
                                summaries[path] = summary.replace("/", "&")
                                self.logger.info(
                                    "[SUMMARY] %s -> %s", path, summaries[path]
                                )

                    missing_paths = [
                        p for p in file_paths if p not in summaries
                    ]
                    if missing_paths:
                        follow_up = (
                            f"You missed the following files: {missing_paths}\n"
                            "Please provide summaries for these files only, "
                            "in the same JSON format."
                        )
                        self.logger.info(
                            "[SUMMARY] Follow-up: missing %d files",
                            len(missing_paths),
                        )
                        memory.add_message(UserMessage(follow_up))
                        continue
                    else:
                        break

                except Exception as e:
                    self.logger.error(
                        "[SUMMARY] batch failed at iteration %d: %s",
                        i + 1,
                        e,
                        exc_info=True,
                    )
                    continue

            local_trajs.append(
                {
                    "type": "file_summary_batch",
                    "files": file_paths,
                    "messages": memory.to_messages(),
                }
            )
            self.logger.info(
                "[SUMMARY] finished batch with %d files",
                len(batch_items),
            )
            return summaries, local_trajs

        file_feature_items = [
            (path, repo_feature_map[path]) for path in repo_feature_map.keys()
        ]
        summary_batches = _make_file_summary_batches(file_feature_items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in summary_batches:
                if batch:
                    futures.append(
                        executor.submit(summarize_file_batch, batch)
                    )

            for future in as_completed(futures):
                try:
                    summaries, local_trajs = future.result(timeout=600)
                    for path, summary in summaries.items():
                        if path in repo_feature_map:
                            repo_feature_map[path]["_file_summary_"] = summary
                    all_trajectories.extend(local_trajs)
                except concurrent.futures.TimeoutError:
                    self.logger.error(
                        "[SUMMARY] file summary batch timed out."
                    )
                except Exception as e:
                    self.logger.error(
                        "[SUMMARY] Error in batch worker: %s", e, exc_info=True
                    )

        # --- Step 7: Deduplicate ---
        repo_feature_map = self._dedupe_file_summaries(repo_feature_map)
        return repo_feature_map, all_trajectories

    # ==================================================================
    # Public API: parse_repo / parse_partial_repo
    # ==================================================================

    def parse_partial_repo(
        self,
        file_code_map: Dict[str, str],
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 1,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict]]:
        """Parse a subset of files (already loaded into *file_code_map*).

        Source: ZeroRepo ``semantic_parsing.py`` :meth:`parse_partial_repo`.

        Normalizes input paths to repo-relative POSIX form so the returned
        ``parsed_tree`` keys are consistent with :meth:`parse_repo` (which
        relativizes against ``self.repo_dir`` after parsing).  Callers that
        pass absolute paths (e.g. ``rpg_evolution``) used to leak the
        original prefix into downstream Node ``meta.path``; this guard
        prevents that.
        """
        tmp_file_code_map: Dict[str, str] = {}
        for path, code in file_code_map.items():
            if os.path.isabs(path):
                norm = normalize_path(os.path.relpath(path, self.repo_dir))
            else:
                norm = normalize_path(path)
            tmp_file_code_map[norm] = code

        self.logger.info(
            "Valid partial files: %s",
            json.dumps(list(tmp_file_code_map.keys()), indent=4),
        )

        return self._parse_files_global(
            file_code_map=tmp_file_code_map,
            max_iterations=max_iterations,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_workers,
        )

    def parse_repo(
        self,
        excluded_files: Optional[List[str]] = None,
        max_iterations: int = 20,
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 20,
        func_context_window: int = 20,
        max_workers: int = 1,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict]]:
        """Parse an entire repository, excluding files in *excluded_files*.

        Source: ZeroRepo ``semantic_parsing.py`` :meth:`parse_repo`
        """
        if excluded_files is None:
            excluded_files = []

        # Step 1: Collect valid source files (any language registered with lang_parser)
        filtered_files = filter_excluded_files(
            valid_files=self.valid_files, excluded_files=excluded_files
        )
        py_files = [
            os.path.join(self.repo_dir, f)
            for f in filtered_files
            if is_supported_source(f) and not is_test_file(f)
        ]

        self.logger.info("Total valid source files to parse: %d", len(py_files))

        file_code_map: Dict[str, str] = {}
        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_code_map[file_path] = f.read()
            except Exception as e:
                self.logger.error("Failed to read file %s: %s", file_path, e)

        repo_feature_map_abs, repo_trajectories = self._parse_files_global(
            file_code_map=file_code_map,
            max_iterations=max_iterations,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_workers,
        )

        # Normalize paths (absolute -> relative POSIX)
        repo_feature_map = {
            normalize_path(os.path.relpath(file_path, self.repo_dir)): value
            for file_path, value in repo_feature_map_abs.items()
        }

        self.logger.info(
            "Successfully parsed: %d files", len(repo_feature_map)
        )

        return repo_feature_map, repo_trajectories

    # ==================================================================
    # LLM parsing: parse_classes / parse_functions
    # ==================================================================

    def parse_classes(
        self,
        code_builder: CodeSnippetBuilder,
        cls_units: List[CodeUnit],
        context_window: int = 10,
        max_iterations: int = 5,
    ) -> Tuple[Dict[str, Any], Dict[str, str], List[Dict[str, str]]]:
        """Extract semantic features for a batch of class definitions.

        The prompt provides file paths and class/method summaries so
        the agent can read the source files itself.

        Returns ``(class_feature_map, descriptions, messages)`` where:

        - ``class_feature_map[class_name]`` is either a dict of method
          feature lists or a flat feature list (preserved for backward
          compatibility with downstream consumers).
        - ``descriptions`` maps composite keys (see ``_desc_key_*``) to
          one-sentence LLM descriptions for every feature.
        """
        memory = Memory(context_window=context_window)
        memory.add_message(
            SystemMessage(
                PARSE_CLASS.format(
                    repo_name=self.repo_name, repo_info=self.repo_info
                )
            )
        )

        class_names = list(
            {cls.name for cls in cls_units if cls.unit_type == "class"}
        )
        class_feature_map: Dict[str, Any] = {
            cls_name: {} for cls_name in class_names
        }
        # Accumulate LLM-generated descriptions across iterations / retries.
        # Keys produced by ``_desc_key_class`` / ``_desc_key_method``.
        local_descs: Dict[str, str] = {}
        processed_classes: set = set()
        processed_methods: Dict[str, set] = {
            cls_name: set() for cls_name in class_names
        }

        valid_class_to_methods: Dict[str, List[str]] = defaultdict(list)
        for cls_name in class_names:
            for cls_unit in cls_units:
                if cls_unit.parent == cls_name and cls_unit.unit_type == "method":
                    valid_class_to_methods[cls_name].append(cls_unit.name)
            valid_class_to_methods[cls_name] = list(
                set(valid_class_to_methods[cls_name])
            )

        # Build file-path + class/method summaries instead of embedding code
        file_code_map = getattr(code_builder, "file_code_map", {})
        summaries = _build_class_summaries(cls_units, file_code_map)

        summary_lines = []
        for file_path, blocks in sorted(summaries.items()):
            summary_lines.append(f"### {file_path}")
            summary_lines.extend(blocks)
            summary_lines.append("")

        env_prompt = (
            "Read the files listed below and extract high-level semantic "
            "features for every class and its methods.\n\n"
            "## Target Classes\n"
            + "\n".join(summary_lines)
            + "\n"
            "## Instructions\n"
            "- Read each file above to understand the class implementations.\n"
            "- For each class, provide features for every method (including "
            "__init__ and special methods).\n"
            "- If a class has no methods, provide class-level features as a "
            "`{feature_name: description}` mapping.\n"
            "- Use verb+object format, 3-8 words per feature name.\n"
            "- Each feature MUST come with a one-sentence (\u226425 words) "
            "description capturing its core purpose. Output `\"\"` if the unit "
            "is a trivial stub and the purpose is unclear.\n\n"
            "## Output Format\n"
            "<solution>\n"
            "{\n"
            '  "ClassName": {\n'
            '    "method_name": {\n'
            '      "feature one": "one-sentence description of feature one",\n'
            '      "feature two": "one-sentence description of feature two"\n'
            "    },\n"
            "    ...\n"
            "  },\n"
            '  "SimpleClass": {\n'
            '    "feature one": "description of class-level feature one"\n'
            "  }\n"
            "}\n"
            "</solution>"
        )
        memory.add_message(UserMessage(env_prompt))

        for i in range(max_iterations):
            try:
                # On retries, build a fresh prompt with only missing items
                # to avoid Memory accumulation exceeding CLI argument limits.
                if i > 0:
                    missing_cls = [
                        cn for cn in class_names
                        if cn not in processed_classes
                    ]
                    missing_mtd: Dict[str, List[str]] = {}
                    for cn in class_names:
                        if cn in processed_classes:
                            all_mtds = valid_class_to_methods.get(cn, [])
                            done_mtds = processed_methods.get(cn, set())
                            left = [m for m in all_mtds if m not in done_mtds]
                            if left:
                                missing_mtd[cn] = left

                    if not missing_cls and not missing_mtd:
                        break

                    # Build targeted retry prompt
                    progress = (
                        f"## Progress\n"
                        f"Completed {len(processed_classes)}/{len(class_names)} classes.\n"
                    )
                    remaining_lines = []
                    # Filter cls_units to only missing items.
                    # Include class units for both fully-missing classes AND
                    # classes with missing methods (so _build_class_summaries
                    # can list them).
                    classes_with_gaps = set(missing_cls) | set(missing_mtd.keys())
                    missing_units = [
                        u for u in cls_units
                        if (u.unit_type == 'class' and u.name in classes_with_gaps)
                        or (u.unit_type == 'method' and u.parent
                            and (u.parent in missing_cls
                                 or u.name in missing_mtd.get(u.parent, [])))
                    ]
                    remaining_summaries = _build_class_summaries(
                        missing_units, file_code_map,
                    )
                    for fp, blocks in sorted(remaining_summaries.items()):
                        remaining_lines.append(f"### {fp}")
                        remaining_lines.extend(blocks)
                        remaining_lines.append("")

                    retry_prompt = (
                        progress
                        + "\n## Remaining Classes/Methods to Analyze\n"
                        + "\n".join(remaining_lines)
                        + "\n## Output Format\n"
                        "Each feature MUST come with a one-sentence "
                        "(\u226425 words) description capturing its core "
                        "purpose. Output `\"\"` if unclear.\n"
                        "<solution>\n"
                        "{\n"
                        '  "ClassName": {\n'
                        '    "method_name": {\n'
                        '      "feature name": "one-sentence description"\n'
                        "    },\n"
                        "    ...\n"
                        "  }\n"
                        "}\n"
                        "</solution>"
                    )
                    memory = Memory(context_window=context_window)
                    memory.add_message(
                        SystemMessage(
                            PARSE_CLASS.format(
                                repo_name=self.repo_name,
                                repo_info=self.repo_info,
                            )
                        )
                    )
                    memory.add_message(UserMessage(retry_prompt))

                response = self.llm_client.generate_with_memory(memory)
                if response is None:
                    self.logger.error(
                        "parse_classes: LLM returned None at iteration %d",
                        i + 1,
                    )
                    continue
                self.logger.info("Response: %s", response[:300])
                parsed_solution = parse_solution_output(response)
                # Do NOT add response to memory — retries use fresh prompts

                parsed_solution = (
                    parsed_solution.replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                    .replace("\t", "")
                )
                parsed_json = json5.loads(parsed_solution)

                # Detect whether ``cls_value`` represents a class-with-methods
                # (its inner values are containers) or class-level features
                # (its inner values are description strings).  Tolerates both
                # the new ``{feat: desc}`` schema and legacy ``[feat]`` lists.
                def _has_method_layer(cv: Any) -> bool:
                    if not isinstance(cv, dict) or not cv:
                        return False
                    for v in cv.values():
                        if isinstance(v, (dict, list)):
                            return True
                        if isinstance(v, str):
                            return False
                    return False

                # Filter: only keep classes present in cls_units
                filtered_parsed_json: Dict[str, Any] = {}
                for cls_name, cls_value in parsed_json.items():
                    if cls_name not in valid_class_to_methods:
                        continue
                    if isinstance(cls_value, list):
                        # Legacy class-level feature list — keep as-is for the
                        # accumulate step to convert.
                        filtered_parsed_json[cls_name] = cls_value
                    elif isinstance(cls_value, dict):
                        if _has_method_layer(cls_value):
                            valid_methods = [
                                m
                                for m in cls_value
                                if m
                                in valid_class_to_methods.get(cls_name, [])
                            ]
                            filtered_parsed_json[cls_name] = {
                                m: cls_value[m] for m in valid_methods
                            }
                        else:
                            # New-schema class-level features: {feat: desc}
                            filtered_parsed_json[cls_name] = cls_value

                # Check for missing classes / methods
                missing_classes = [
                    cn
                    for cn in class_names
                    if cn not in parsed_json and cn not in processed_classes
                ]
                missing_methods: Dict[str, List[str]] = {}

                for cls_name, cls_value in filtered_parsed_json.items():
                    if _has_method_layer(cls_value):
                        methods_in_class = [
                            u.name
                            for u in cls_units
                            if u.unit_type == "method" and u.parent == cls_name
                        ]
                        missing_m = [
                            m
                            for m in methods_in_class
                            if m not in cls_value
                            and m not in processed_methods[cls_name]
                        ]
                        if missing_m:
                            missing_methods[cls_name] = missing_m

                # Accumulate results: split LLM payload into feature names
                # (kept in ``class_feature_map``) and descriptions (kept in
                # ``local_descs`` under composite keys).
                #
                # IMPORTANT: ``/`` in a feature *name* is normalized to
                # ``" or "`` to keep the name compatible with RPG path
                # separators downstream.  The same normalization MUST be
                # applied to the composite-key segment that derives from the
                # name, otherwise ``_init_feature_tree`` will look the desc
                # up under the normalized name and silently miss the entry.
                # Description *values* are left untouched (legitimate text
                # such as ``client/server`` should not be mangled).
                for cls_name, cls_value in filtered_parsed_json.items():
                    if cls_name in processed_classes:
                        continue

                    if _has_method_layer(cls_value):
                        # Class with methods.
                        for method_name, method_features in cls_value.items():
                            names, descs = _split_features(method_features)
                            names = [
                                n.replace("/", " or ") for n in names
                            ]
                            class_feature_map[cls_name][method_name] = names
                            for n, d in descs.items():
                                n_key = n.replace("/", " or ")
                                local_descs[
                                    _desc_key_method(
                                        cls_name, method_name, n_key
                                    )
                                ] = d
                            processed_methods[cls_name].add(method_name)
                    else:
                        # Class-level features (new dict-schema or legacy list).
                        names, descs = _split_features(cls_value)
                        names = [n.replace("/", " or ") for n in names]
                        class_feature_map[cls_name] = names
                        for n, d in descs.items():
                            n_key = n.replace("/", " or ")
                            local_descs[_desc_key_class(cls_name, n_key)] = d

                    processed_classes.add(cls_name)

                if missing_classes or missing_methods:
                    self.logger.info(
                        "Iteration %d: missing_classes=%s, missing_methods=%s",
                        i + 1, missing_classes, list(missing_methods.keys()),
                    )
                    continue
                else:
                    break

            except Exception as e:
                self.logger.error(
                    "parse_classes failed at iteration %d: %s",
                    i + 1,
                    e,
                    exc_info=True,
                )
                continue

        return class_feature_map, local_descs, memory.to_messages()

    def parse_functions(
        self,
        code_builder: CodeSnippetBuilder,
        func_units: List[CodeUnit],
        context_window: int = 5,
        max_iterations: int = 5,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], List[Dict[str, str]]]:
        """Extract semantic features for a batch of standalone functions.

        The prompt provides file paths and function signatures so the
        agent can read the source files itself, avoiding CLI argument
        size limits.

        Returns ``(feature_map, descriptions, messages)`` where:

        - ``feature_map[func_name]`` is the list of feature names (str).
        - ``descriptions`` maps composite keys (see ``_desc_key_function``)
          to one-sentence LLM descriptions.
        """
        memory = Memory(context_window=context_window)
        memory.add_message(
            SystemMessage(
                PARSE_FUNCTION.format(
                    repo_name=self.repo_name, repo_info=self.repo_info
                )
            )
        )

        func_names = list(
            {func.name or "<anonymous>" for func in func_units}
        )
        feature_map: Dict[str, List[str]] = {}
        # Accumulate LLM-generated descriptions across iterations / retries.
        # Keys produced by ``_desc_key_function``.
        local_descs: Dict[str, str] = {}

        # Build file-path + signature summaries instead of embedding code
        file_code_map = getattr(code_builder, "file_code_map", {})
        summaries = _build_function_summaries(func_units, file_code_map)

        summary_lines = []
        for file_path, sigs in sorted(summaries.items()):
            summary_lines.append(f"### {file_path}")
            summary_lines.extend(sigs)
            summary_lines.append("")

        env_prompt = (
            "You are analyzing a set of standalone Python functions.\n"
            "Read each file listed below and extract high-level semantic "
            "features for every function specified.\n\n"
            "## Target Functions\n"
            + "\n".join(summary_lines)
            + "\n"
            "## Instructions\n"
            "- Read each file above to understand the function implementations.\n"
            "- For each function, output a `{feature_name: description}` "
            "mapping. Feature names use verb+object format, 3-8 words each.\n"
            "- Each description is ONE English sentence (\u226425 words) "
            "capturing the core purpose; output `\"\"` for trivial stubs.\n"
            "- If a function performs multiple responsibilities, list "
            "multiple features.\n"
            "- If a function is a stub with no meaningful features, output "
            "an empty mapping `{}`.\n\n"
            "## Output Format\n"
            "<solution>\n"
            "{\n"
            '  "function_name": {\n'
            '    "feature one": "one-sentence description of feature one",\n'
            '    "feature two": "one-sentence description of feature two"\n'
            "  },\n"
            "  ...\n"
            "}\n"
            "</solution>"
        )
        memory.add_message(UserMessage(env_prompt))

        for i in range(max_iterations):
            try:
                # On retries, build a fresh prompt with only missing functions
                # to avoid Memory accumulation exceeding CLI argument limits.
                if i > 0:
                    missing_keys = [
                        name for name in func_names if name not in feature_map
                    ]
                    if not missing_keys:
                        break

                    missing_summaries = _build_function_summaries(
                        [u for u in func_units if u.name in set(missing_keys)],
                        file_code_map,
                    )
                    retry_lines = []
                    for fp, sigs in sorted(missing_summaries.items()):
                        retry_lines.append(f"### {fp}")
                        retry_lines.extend(sigs)
                        retry_lines.append("")

                    progress = (
                        f"## Progress\n"
                        f"Completed {len(feature_map)}/{len(func_names)} functions.\n"
                    )
                    retry_prompt = (
                        progress
                        + "\n## Remaining Functions to Analyze\n"
                        + "\n".join(retry_lines)
                        + "\n## Output Format\n"
                        "Each feature MUST come with a one-sentence "
                        "(\u226425 words) description.\n"
                        "<solution>\n"
                        "{\n"
                        '  "function_name": {\n'
                        '    "feature name": "one-sentence description"\n'
                        "  },\n"
                        "  ...\n"
                        "}\n"
                        "</solution>"
                    )
                    memory = Memory(context_window=context_window)
                    memory.add_message(
                        SystemMessage(
                            PARSE_FUNCTION.format(
                                repo_name=self.repo_name,
                                repo_info=self.repo_info,
                            )
                        )
                    )
                    memory.add_message(UserMessage(retry_prompt))

                response = self.llm_client.generate_with_memory(memory)
                if response is None:
                    self.logger.error(
                        "parse_functions: LLM returned None at iteration %d",
                        i + 1,
                    )
                    continue
                self.logger.info("Response: %s", response[:300])
                parsed_solution = parse_solution_output(response)
                # Do NOT add response to memory — retries use fresh prompts

                parsed_solution = (
                    parsed_solution.replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                    .replace("\t", "")
                )
                parsed_json = json5.loads(parsed_solution)

                # Split LLM payload into names (main map) + descriptions
                # (composite-key sidecar map).  Tolerates both dict-of-desc
                # (new schema) and legacy list-of-name responses.
                #
                # IMPORTANT: ``/`` in a feature name is normalized to
                # ``" or "`` to keep RPG paths separator-safe.  The same
                # normalization MUST be applied to the desc-key segment so
                # ``_init_feature_tree`` can find the description by the
                # name it later stores on the Node.  Description *values*
                # are kept verbatim (``client/server`` etc.).
                valid_feature_map: Dict[str, List[str]] = {}
                for key, value in parsed_json.items():
                    if key not in func_names:
                        continue
                    names, descs = _split_features(value)
                    names = [n.replace("/", " or ") for n in names]
                    valid_feature_map[key] = names
                    for n, d in descs.items():
                        n_key = n.replace("/", " or ")
                        local_descs[_desc_key_function(key, n_key)] = d

                invalid_keys = [
                    key for key in parsed_json.keys() if key not in func_names
                ]
                feature_map.update(valid_feature_map)
                missing_keys = [
                    name for name in func_names if name not in feature_map
                ]

                if missing_keys or invalid_keys:
                    self.logger.info(
                        "Iteration %d: completed=%d, missing=%d, invalid=%d",
                        i + 1, len(feature_map), len(missing_keys), len(invalid_keys),
                    )
                else:
                    break

            except Exception as e:
                self.logger.error(
                    "parse_functions failed at iteration %d: %s",
                    i + 1,
                    e,
                    exc_info=True,
                )
                continue

        return feature_map, local_descs, memory.to_messages()
