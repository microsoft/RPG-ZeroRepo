"""RPG Encoding Module.

Main entry-point for encoding a repository into an RPG structure.
The ``RPGParser`` class orchestrates the full pipeline:

1. Generate repository overview (``repo_info``)
2. Exclude irrelevant files
3. Parse semantic features (via M6 ``ParseFeatures``)
4. Refactor features into a three-level RPG tree (via ``RefactorTree``)
5. (Optional) Analyze data flow

Ported from RPG-ZeroRepo ``zerorepo/rpg_encoder/rpg_parsing/rpg_encoding.py``
with the following adaptations for CoderMind:
- Uses ``LLMClient`` from ``scripts.common.llm_client``
- Uses ``Memory`` / message types from ``scripts.common.llm_types``
- Uses utility functions from ``scripts.common.utils``
- Uses ``RPG`` / ``Node`` / ``NodeMetaData`` / ``NodeType``
  from ``scripts.skeleton.rpg_models``
- DataFlowAgent is **skipped** (CoderMind already has its own data-flow agent)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from common.llm_client import LLMClient
from common.llm_types import (
    AssistantMessage,
    Memory,
    SystemMessage,
    UserMessage,
)
from common.rpg_io import atomic_write_rpg
from common.utils import (
    is_skip_dir,
    exclude_files,
    normalize_path,
    parse_code_blocks,
    parse_solution_output,
    truncate_by_token,
)
from lang_parser import dominant_language, is_supported_source, is_test_file
from rpg import RPG

from .prompts import EXCLUDE_FILES, GENERATE_REPO_INFO
from .refactor_tree import RefactorTree
from .semantic_parsing import ParseFeatures

logger = logging.getLogger(__name__)


class RPGParser:
    """Parse a repository into an RPG (Repository Program Graph).

    This is the main entry-point for M7 -- RPG Encoding.  It mirrors the
    ``RPGParser`` class in RPG-ZeroRepo with minimal interface changes.

    Typical usage::

        parser = RPGParser(
            repo_dir="/path/to/repo",
            repo_name="my-project",
        )
        rpg, feature_tree, skeleton_info = parser.parse_rpg_from_repo()

    Source: RPG-ZeroRepo ``rpg_encoding.py`` :class:`RPGParser`
    """

    def __init__(
        self,
        repo_dir: str,
        repo_name: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.repo_dir = repo_dir
        self.repo_name = repo_name

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"RPGParser[{repo_name}]")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                )
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        # Initialize LLM client
        self.llm_client = LLMClient()

        self.logger.info(
            "Initializing RPGParser: dir=%s, name=%s", repo_dir, repo_name
        )

        # Load skeleton and valid files from repo
        self.skeleton_info, self.valid_files = self._load_skeleton_from_repo()
        self.skeleton_info = truncate_by_token(
            self.skeleton_info, max_tokens=50000
        ).strip()
        self.logger.info("Skeleton loaded: files=%d", len(self.valid_files))

    # ------------------------------------------------------------------
    # Skeleton loading (simplified — no RepoSkeleton dependency)
    # ------------------------------------------------------------------

    def _load_skeleton_from_repo(self) -> Tuple[str, List[str]]:
        """Walk the repo directory and build a text skeleton + valid file list.

        This replaces the ZeroRepo ``load_skeleton_from_repo`` helper
        which depends on ``RepoSkeleton`` / ``FileNode``.  We only need
        two outputs: a tree-formatted string and a list of relative paths
        for all valid Python files.

        Returns:
            (skeleton_info, valid_files)
        """
        valid_files: List[str] = []
        tree_lines: List[str] = []

        for root, dirs, files in os.walk(self.repo_dir):
            # Skip hidden dirs and common non-essential dirs
            dirs[:] = [d for d in dirs if not is_skip_dir(d)]
            dirs.sort()

            rel_root = os.path.relpath(root, self.repo_dir)
            if rel_root == ".":
                rel_root = ""

            for fname in sorted(files):
                if fname.startswith("."):
                    continue
                rel_path = os.path.join(rel_root, fname) if rel_root else fname
                rel_path = rel_path.replace("\\", "/")
                tree_lines.append(rel_path)
                if is_supported_source(rel_path) and not is_test_file(rel_path):
                    valid_files.append(rel_path)

        skeleton_info = "\n".join(tree_lines)
        return skeleton_info, valid_files

    def _find_readme(self) -> str:
        """Try to find and read a README file from the repo."""
        readme_names = [
            "README.md", "readme.md", "README.txt",
            "README", "readme", "README.rst",
        ]
        for r_file in readme_names:
            full_path = os.path.join(self.repo_dir, r_file)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    self.logger.info(
                        "README found: %s (length=%d)", r_file, len(content)
                    )
                    return content
                except Exception:
                    continue
        self.logger.warning(
            "README not found; proceeding with empty README content."
        )
        return ""

    # ------------------------------------------------------------------
    # generate_repo_info
    # ------------------------------------------------------------------

    def generate_repo_info(self, max_iters: int = 3) -> str:
        """Generate a description of the repository using LLM.

        Source: ZeroRepo ``RPGParser.generate_repo_info`` (:71)
        """
        self.logger.info("Generating repo info (max_iters=%d)...", max_iters)

        readme = self._find_readme()
        readme = truncate_by_token(text=readme, max_tokens=50000).strip()

        user_prompt = (
            f"Repository Name:\n<repo_name>\n{self.repo_name}\n</repo_name>\n"
            f"Repository Structure:\n<skeleton>\n{self.skeleton_info}\n</skeleton>\n"
            f"Repository README Content:\n<readme>\n{readme}\n</readme>\n"
            f"Based on the information above, please summarize and generate "
            f"a comprehensive Repository Overview."
        )

        self.logger.info("GENERATE_REPO_INFO user prompt:\n%s", user_prompt[:500])

        memory = Memory(context_window=10)
        memory.add_message(SystemMessage(GENERATE_REPO_INFO))
        memory.add_message(UserMessage(user_prompt))

        repo_info = ""
        for i in range(max_iters):
            try:
                self.logger.info("LLM call for repo info, iter=%d...", i + 1)
                response = self.llm_client.generate_with_memory(memory)
                self.logger.info("Iter %d Response: %s", i + 1, response[:500])
                parsed_response = parse_solution_output(response)
                code_blocks = parse_code_blocks(
                    output=parsed_response, type="general"
                )
                if not code_blocks:
                    self.logger.warning(
                        "No code blocks parsed in iter=%d; skipping.", i + 1
                    )
                    continue
                repo_info = "\n".join(code_blocks)
                self.logger.info(
                    "Repo info received (len=%d) in iter=%d.",
                    len(repo_info),
                    i + 1,
                )
                if repo_info:
                    break
            except Exception as e:
                self.logger.exception(
                    "Error generating repo info at iter=%d: %s", i + 1, e
                )
                continue

        if not repo_info:
            self.logger.warning(
                "Repo info is empty after %d iterations.", max_iters
            )
        return repo_info

    # ------------------------------------------------------------------
    # exclude_irrelevant_files
    # ------------------------------------------------------------------

    def exclude_irrelevant_files(
        self, repo_info: str, max_votes: int = 3
    ) -> List[str]:
        """Identify irrelevant files that should be excluded from parsing.

        Uses multi-vote LLM approach and consolidation.

        Source: ZeroRepo ``RPGParser.exclude_irrelvant_files`` (:131)
        """
        self.logger.info(
            "Excluding irrelevant files (max_votes=%d)...", max_votes
        )

        user_prompt = (
            f"Repository Name:\n<readme>\n{self.repo_name}\n</readme>\n"
            f"Repository Structure:\n<skeleton>\n{self.skeleton_info}\n</skeleton>\n"
            f"Repository Overview:\n<repo_info>\n{repo_info}\n</repo_info>\n"
            f"Based on the information above, please analyze this repository and identify "
            f"the paths that are likely unrelated to the core algorithms or main implementation, "
            f"such as folders or files that appear to be forked from other repositories, "
            f"third-party code, demo data, documentation, or build/test artifacts. "
            f"List these unrelated paths clearly so they can be excluded from further functional extraction."
        )

        self.logger.info(
            "EXCLUDE_FILES user prompt:\n%s", user_prompt[:500]
        )

        memory = Memory(context_window=10)
        memory.add_message(SystemMessage(EXCLUDE_FILES))
        memory.add_message(UserMessage(user_prompt))

        excluded_files_all_rounds: List[List[str]] = []

        for i in range(max_votes):
            try:
                self.logger.info("LLM vote #%d for exclude list...", i + 1)
                response = self.llm_client.generate_with_memory(memory)
                self.logger.info("Iter %d Response: %s", i + 1, response[:500])
                parsed_response = parse_solution_output(response)
                code_blocks = parse_code_blocks(
                    output=parsed_response, type="general"
                )
                code_blocks = (
                    code_blocks
                    if code_blocks
                    else parsed_response.split("\n")
                )
                if i == 0:
                    memory.add_message(AssistantMessage(response))
                if not code_blocks:
                    self.logger.warning(
                        "Vote #%d produced no code block; skipping.", i + 1
                    )
                    continue

                file_text = "\n".join(code_blocks)
                file_paths = [
                    normalize_path(f_line.strip())
                    for f_line in file_text.split("\n")
                    if f_line.strip()
                    and self._path_exists_in_repo(f_line.strip())
                ]
                excluded_files_all_rounds.append(file_paths)

                self.logger.info(
                    "Vote #%d candidates: %d (after existence check).",
                    i + 1,
                    len(file_paths),
                )

            except Exception as e:
                self.logger.exception(
                    "Error during exclude vote #%d: %s", i + 1, e
                )
                continue

        flat_list = [p for sublist in excluded_files_all_rounds for p in sublist]
        self.logger.info(
            "Total raw candidates across votes: %d", len(flat_list)
        )

        # Single-vote fast path: no consolidation LLM call needed.
        if max_votes <= 1:
            final_result: List[str] = sorted(set(flat_list))
            self.logger.info(
                "Single-vote exclude list size: %d (consolidation skipped)",
                len(final_result),
            )
            all_paths = list(self.valid_files)
            filter_paths = exclude_files(files=all_paths)
            return sorted(set(final_result + filter_paths))

        combined_sections: List[str] = []
        for idx, paths in enumerate(excluded_files_all_rounds, start=1):
            if not paths:
                continue
            section = (
                f"### Round {idx} Excluded Paths\n"
                + "\n".join(f"- {p}" for p in paths)
            )
            combined_sections.append(section)

        combined_text = "\n\n".join(combined_sections)
        self.logger.info(
            "Combined exclude candidates text:\n%s", combined_text[:1000]
        )

        summarize_prompt = (
            "Below are multiple rounds of proposed irrelevant paths extracted from the repository:\n"
            f"<round_outputs>\n{combined_text}\n</round_outputs>\n"
            "Please review and consolidate these results into a single, clean list of paths "
            "that are most likely irrelevant or should be excluded from further analysis. "
            "Remove duplicates and keep only the final recommended paths.\n"
            "Output one path per line, no explanations."
        )
        memory.add_message(UserMessage(summarize_prompt))

        self.logger.info("Summarize prompt:\n%s", summarize_prompt[:500])

        final_result: List[str] = []
        try:
            self.logger.info(
                "Consolidating exclude list with a final LLM pass..."
            )
            response = self.llm_client.generate_with_memory(memory)
            parsed_response = parse_solution_output(response)
            code_blocks = parse_code_blocks(
                output=parsed_response, type="general"
            )
            if code_blocks:
                final_text = "\n".join(code_blocks)
                final_result = [
                    normalize_path(line.strip())
                    for line in final_text.split("\n")
                    if line.strip()
                    and self._path_exists_in_repo(line.strip())
                ]

            final_result = sorted(set(final_result))
            self.logger.info("Final exclude list size: %d", len(final_result))
        except Exception as e:
            self.logger.exception("Final consolidation failed: %s", e)

        # Also add standard exclude-files patterns
        all_paths = list(self.valid_files)
        filter_paths = exclude_files(files=all_paths)
        final_result = sorted(set(final_result + filter_paths))

        return final_result

    def _path_exists_in_repo(self, path: str) -> bool:
        """Check if a path exists relative to repo_dir."""
        cleaned = normalize_path(path.strip())
        full = os.path.join(self.repo_dir, cleaned)
        return os.path.exists(full)

    # ------------------------------------------------------------------
    # parse_rpg_from_repo — main pipeline
    # ------------------------------------------------------------------

    def parse_rpg_from_repo(
        self,
        repo_info: Optional[str] = None,
        max_repo_info_iters: int = 3,
        max_exclude_votes: int = 1,
        max_parse_iters: int = 5,
        # Parse features parameters
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 10,
        func_context_window: int = 10,
        max_parse_workers: int = 1,
        # Refactor parameters
        refactor_context_window: int = 10,
        refactor_max_iters: int = 5,
        save_path: str = "",
        # Data flow analysis is skipped (CoderMind has its own)
    ) -> Tuple[RPG, Dict, str]:
        """Run the full RPG parsing pipeline.

        Pipeline steps:
        1. Generate repo info (or use provided)
        2. Exclude irrelevant files
        3. Parse semantic features
        4. Refactor to RPG

        Args:
            repo_info: Pre-computed repo info (skip LLM generation if provided)
            save_path: If set, save intermediate results to this JSON path.

        Returns:
            (rpg, feature_tree, skeleton_info)

        Source: ZeroRepo ``RPGParser.parse_rpg_from_repo`` (:252)
        """
        final_result: dict = {}

        self.logger.info("=== RPG parsing pipeline started ===")
        self.logger.info(
            "Params: repo_info_iters=%d, exclude_votes=%d, parse_iters=%d, "
            "min_batch=%d, max_batch=%d, class_ctx=%d, func_ctx=%d, workers=%d, "
            "refactor_ctx=%d, refactor_iters=%d",
            max_repo_info_iters,
            max_exclude_votes,
            max_parse_iters,
            min_batch_tokens,
            max_batch_tokens,
            class_context_window,
            func_context_window,
            max_parse_workers,
            refactor_context_window,
            refactor_max_iters,
        )

        # 1) Repo overview
        if not repo_info:
            repo_info = self.generate_repo_info(max_iters=max_repo_info_iters)

        final_result["repo_name"] = self.repo_name
        final_result["repo_info"] = repo_info

        # 2) Exclude irrelevant
        excluded_files = self.exclude_irrelevant_files(
            repo_info=repo_info, max_votes=max_exclude_votes
        )
        final_result["excluded_files"] = excluded_files

        self.logger.info("Excluded paths decided: %d", len(excluded_files))

        # 3) Parse features
        feature_parser = ParseFeatures(
            repo_dir=self.repo_dir,
            repo_info=repo_info,
            repo_skeleton=self.skeleton_info,
            valid_files=self.valid_files,
            repo_name=self.repo_name,
            logger=self.logger,
            llm_client=self.llm_client,
        )
        self.logger.info("Parsing features...")

        file2feature, parse_traj = feature_parser.parse_repo(
            excluded_files=excluded_files,
            max_iterations=max_parse_iters,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_workers=max_parse_workers,
        )

        if save_path:
            # Atomic write so a kill mid-dump doesn't truncate the
            # intermediate parsed-tree snapshot (resumed runs read it).
            atomic_write_rpg(
                save_path, final_result,
                indent=4,
                default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o),
            )

        self.logger.info("Features parsed: files=%d", len(file2feature))

        # Determine the repo's dominant language from the files actually
        # scanned by the skeleton so RefactorTree can stamp every
        # NodeMetaData it produces (FILE / CLASS / FUNCTION / METHOD)
        # with the correct ``meta.language``. Without this, the
        # encoder fell back to its (legacy) "python" default and
        # produced misleading meta for Go / Rust / TS / ... repos.
        repo_dominant_language = dominant_language(self.valid_files)
        if repo_dominant_language:
            self.logger.info("Dominant language detected: %s", repo_dominant_language)
        else:
            self.logger.info(
                "Dominant language could not be detected from skeleton; "
                "RefactorTree will leave meta.language unset."
            )

        # 4) Refactor to RPG
        refactor_agent = RefactorTree(
            repo_dir=self.repo_dir,
            repo_info=repo_info,
            repo_skeleton=self.skeleton_info,
            skeleton_info=self.skeleton_info,
            repo_name=self.repo_name,
            logger=self.logger,
            llm_client=self.llm_client,
            language=repo_dominant_language,
        )
        self.logger.info("Refactoring to RPG...")
        final_rpg, refactor_traj, repo_rpg = refactor_agent.run(
            parsed_tree=file2feature,
            context_window=refactor_context_window,
            max_iters=refactor_max_iters,
        )

        repo_rpg.repo_info = repo_info
        repo_rpg.excluded_files = excluded_files

        repo_rpg.update_all_metadata_bottom_up()

        # Remove empty subtrees
        removal_stats = repo_rpg.remove_empty_subtrees()
        if removal_stats["removed_nodes"] > 0:
            self.logger.info(
                "Cleaned up %d empty subtrees after refactoring",
                removal_stats["removed_nodes"],
            )

        final_result["rpg"] = {
            "structure": repo_rpg.to_dict(),
            "feature_tree": final_rpg,
            "traj": refactor_traj,
        }

        if save_path:
            # Atomic write of the final encoder result; see the same
            # note on the parsed-tree snapshot above.
            atomic_write_rpg(
                save_path, final_result,
                indent=4,
                default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o),
            )

        self.logger.info("RPG refactoring done.")
        self.logger.info("=== RPG parsing pipeline finished ===")

        return repo_rpg, final_rpg, self.skeleton_info
