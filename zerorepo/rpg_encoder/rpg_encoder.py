"""
RPG Encoder Module

Unified interface for RPG parsing and evolution. Combines functionality from:
- RPGParser: Parse repository into RPG structure
- RPGEvolution: Handle incremental updates based on diffs
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from zerorepo.rpg_gen.base import (
    RPG, LLMConfig, RepoSkeleton
)
from zerorepo.utils.logs import setup_logger
from .rpg_parsing.rpg_encoding import RPGParser
from .rpg_parsing.rpg_evolution import RPGEvolution

class RPGEncoder:
    """
    Unified interface for encoding a repository into RPG (Repository Program Graph).

    Supports two main workflows:
    1. Initial encoding: Parse a repository from scratch
    2. Incremental update: Update RPG based on repository changes (diffs)

    Example usage:
        # Initial encoding
        encoder = RPGEncoder(repo_dir="/path/to/repo", repo_name="my_repo")
        rpg, feature_tree, skeleton = encoder.encode()
        encoder.save("output.json")

        # Incremental update
        encoder = RPGEncoder.from_saved("output.json", cur_repo_dir="/path/to/updated_repo")
        rpg = encoder.update(last_repo_dir="/path/to/old_repo")
        encoder.save("output_updated.json")
    """

    def __init__(
        self,
        repo_dir: str,
        repo_name: str,
        repo_info: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize RPGEncoder.

        Args:
            repo_dir: Path to the repository directory
            repo_name: Name of the repository
            repo_info: Optional description of the repository
            llm_config: LLM configuration for parsing
            logger: Logger instance
        """
        self.repo_dir = os.path.abspath(repo_dir)
        self.repo_name = repo_name
        self.repo_info = repo_info or ""
        self.llm_config = llm_config or LLMConfig(model="gpt-4o")
        self.logger = logger or setup_logger(logging.getLogger(f"RPGEncoder[{repo_name}]"))

        # RPG and related structures
        self.rpg: Optional[RPG] = None
        self.feature_tree: Optional[List[Dict]] = None
        self.skeleton: Optional[RepoSkeleton] = None
        self.excluded_files: List[str] = []

        # Internal parser instance
        self._parser: Optional[RPGParser] = None

    @classmethod
    def from_saved(
        cls,
        save_path: str,
        cur_repo_dir: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "RPGEncoder":
        """
        Load RPGEncoder from a saved JSON file.

        Args:
            save_path: Path to the saved JSON file
            cur_repo_dir: Current repository directory (for updates)
            llm_config: LLM configuration
            logger: Logger instance

        Returns:
            RPGEncoder instance with loaded state
        """
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        repo_name = data.get("repo_name", "unknown")
        repo_info = data.get("repo_info", "")
        repo_dir = cur_repo_dir or data.get("repo_dir", "")

        encoder = cls(
            repo_dir=repo_dir,
            repo_name=repo_name,
            repo_info=repo_info,
            llm_config=llm_config,
            logger=logger,
        )

        # Restore RPG - supports both flat format and legacy nested format
        rpg_data = data.get("rpg", {})
        if "structure" in rpg_data:
            # Legacy nested format: {rpg: {structure: {...}, feature_tree: [...]}}
            encoder.rpg = RPG.from_dict(rpg_data["structure"])
            encoder.feature_tree = rpg_data.get("feature_tree", [])
        elif "nodes" in data:
            # New flat format: top-level keys contain RPG fields directly
            encoder.rpg = RPG.from_dict(data)
            encoder.feature_tree = encoder.rpg.get_functionality_graph() if encoder.rpg else []
        else:
            encoder.feature_tree = []

        # Restore excluded files
        encoder.excluded_files = data.get("excluded_files", [])

        encoder.logger.info(f"Loaded RPG from {save_path}: {len(encoder.rpg.nodes) if encoder.rpg else 0} nodes")

        return encoder

    def encode(
        self,
        max_repo_info_iters: int = 3,
        max_exclude_votes: int = 3,
        max_parse_iters: int = 10,
        # Parse features parameters
        min_batch_tokens: int = 10_000,
        max_batch_tokens: int = 50_000,
        summary_min_batch_tokens: int = 10_000,
        summary_max_batch_tokens: int = 50_000,
        class_context_window: int = 10,
        func_context_window: int = 10,
        max_parse_workers: int = 8,
        # Refactor parameters
        refactor_context_window: int = 10,
        refactor_max_iters: int = 10,
        update_dep_graph: bool = True,
    ) -> Tuple[RPG, List[Dict], RepoSkeleton]:
        """
        Encode the repository into RPG structure.

        Args:
            max_repo_info_iters: Max iterations for repo info generation
            max_exclude_votes: Max votes for file exclusion
            max_parse_iters: Max iterations for feature parsing
            min_batch_tokens: Min tokens for batch parsing
            max_batch_tokens: Max tokens for batch parsing
            summary_min_batch_tokens: Min tokens for summary batch
            summary_max_batch_tokens: Max tokens for summary batch
            class_context_window: Context window for class parsing
            func_context_window: Context window for function parsing
            max_parse_workers: Max parallel workers for parsing
            refactor_context_window: Context window for refactoring
            refactor_max_iters: Max iterations for refactoring
            update_dep_graph: Whether to update dependency graph

        Returns:
            Tuple of (RPG, feature_tree, skeleton)
        """
        self.logger.info(f"Starting RPG encoding for {self.repo_name}...")

        # Create parser
        self._parser = RPGParser(
            repo_dir=self.repo_dir,
            repo_name=self.repo_name,
            logger=self.logger,
            llm_config=self.llm_config
        )

        # Generate repo info if not provided
        if not self.repo_info:
            self.repo_info = self._parser.generate_repo_info(max_iters=max_repo_info_iters)

        # Parse RPG
        self.rpg, self.feature_tree, self.skeleton = self._parser.parse_rpg_from_repo(
            repo_info=self.repo_info,
            max_repo_info_iters=max_repo_info_iters,
            max_exclude_votes=max_exclude_votes,
            max_parse_iters=max_parse_iters,
            min_batch_tokens=min_batch_tokens,
            max_batch_tokens=max_batch_tokens,
            summary_min_batch_tokens=summary_min_batch_tokens,
            summary_max_batch_tokens=summary_max_batch_tokens,
            class_context_window=class_context_window,
            func_context_window=func_context_window,
            max_parse_workers=max_parse_workers,
            refactor_context_window=refactor_context_window,
            refactor_max_iters=refactor_max_iters,
        )

        # Update excluded files
        self.excluded_files = self.rpg.excluded_files if self.rpg else []

        # Update dependency graph
        if update_dep_graph and self.rpg:
            try:
                self.rpg.parse_dep_graph(self.repo_dir)
                self.logger.info(
                    f"Dependency graph updated: {len(self.rpg._dep_to_rpg_map)} mappings"
                )
            except Exception as e:
                self.logger.warning(f"Failed to update dependency graph: {e}")

        self.logger.info(
            f"RPG encoding complete: {len(self.rpg.nodes)} nodes, {len(self.rpg.edges)} edges"
        )

        return self.rpg, self.feature_tree, self.skeleton

    def update(
        self,
        last_repo_dir: str,
        save_path: Optional[str] = None,
        update_dep_graph: bool = True,
    ) -> RPG:
        """
        Update RPG based on repository changes (diff between last and current version).

        Args:
            last_repo_dir: Path to the previous version of the repository
            save_path: Optional path to save updated results
            update_dep_graph: Whether to update dependency graph after update

        Returns:
            Updated RPG instance
        """
        if self.rpg is None:
            raise ValueError("No existing RPG to update. Call encode() first or load from saved file.")

        self.logger.info(f"Starting RPG update for {self.repo_name}...")
        self.logger.info(f"  Last repo: {last_repo_dir}")
        self.logger.info(f"  Current repo: {self.repo_dir}")

        # Use RPGEvolution for incremental update
        self.rpg = RPGEvolution.process_diff(
            repo_name=self.repo_name,
            repo_info=self.repo_info,
            save_path=save_path or "",
            last_repo_dir=last_repo_dir,
            cur_repo_dir=self.repo_dir,
            last_rpg=self.rpg,
            last_feature_tree=self.feature_tree,
            logger=self.logger,
            llm_config=self.llm_config,
            update_dep_graph=update_dep_graph,
        )

        # Update feature tree from RPG
        self.feature_tree = self.rpg.get_functionality_graph()

        self.logger.info(
            f"RPG update complete: {len(self.rpg.nodes)} nodes, {len(self.rpg.edges)} edges"
        )

        return self.rpg

    def save(self, save_path: str, include_dep_graph: bool = False) -> None:
        """
        Save RPG and related data to a JSON file.

        Args:
            save_path: Path to save the JSON file
            include_dep_graph: Whether to include DependencyGraph data (default False)
        """
        if self.rpg is None:
            raise ValueError("No RPG to save. Call encode() first.")

        # Save in flat format (same as RPG.to_dict output)
        result = self.rpg.to_dict(include_dep_graph=include_dep_graph)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved RPG to {save_path}")

    def get_functional_areas(self) -> List[str]:
        """Get list of functional areas in the RPG."""
        if self.rpg is None:
            return []
        return self.rpg.get_functional_areas()

    def get_file_features(self, file_path: str) -> List[str]:
        """
        Get features associated with a specific file.

        Args:
            file_path: Path to the file (relative to repo)

        Returns:
            List of feature paths for the file
        """
        if self.rpg is None:
            return []

        features = []
        for node in self.rpg.nodes.values():
            if node.meta and node.meta.path:
                node_path = node.meta.path
                if isinstance(node_path, list):
                    node_path = node_path[0] if node_path else ""

                if node_path == file_path or node_path.startswith(f"{file_path}:"):
                    features.append(node.feature_path())

        return features

    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for features matching a query.

        Args:
            query: Search query string

        Returns:
            List of matching features with metadata
        """
        if self.rpg is None:
            return []

        results = []
        query_lower = query.lower()

        for node_id, node in self.rpg.nodes.items():
            if query_lower in node.name.lower():
                results.append({
                    "id": node_id,
                    "name": node.name,
                    "feature_path": node.feature_path(),
                    "type": node.meta.type_name.value if node.meta and node.meta.type_name else None,
                    "path": node.meta.path if node.meta else None,
                })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current RPG."""
        if self.rpg is None:
            return {"status": "no_rpg"}

        return {
            "repo_name": self.repo_name,
            "rpg_nodes": len(self.rpg.nodes),
            "rpg_edges": len(self.rpg.edges),
            "functional_areas": len(self.get_functional_areas()),
            "excluded_files": len(self.excluded_files),
            "dep_graph_nodes": len(self.rpg.dep_graph.G.nodes()) if self.rpg.dep_graph else 0,
            "dep_graph_edges": len(self.rpg.dep_graph.G.edges()) if self.rpg.dep_graph else 0,
            "dep_rpg_edges": len(self.rpg.get_dep_edges_for_rpg()) if self.rpg.dep_graph else 0,
            "dep_mappings": len(self.rpg._dep_to_rpg_map) if self.rpg._dep_to_rpg_map else 0,
        }

    def __repr__(self) -> str:
        status = "loaded" if self.rpg else "empty"
        nodes = len(self.rpg.nodes) if self.rpg else 0
        return f"<RPGEncoder repo='{self.repo_name}' status={status} nodes={nodes}>"
