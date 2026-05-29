#!/usr/bin/env python3
"""Skeleton Module.

Skeleton-specific models, prompts and the file designer:
- skeleton_models: RepoSkeleton, DirectoryNode, FileNode
- skeleton_prompts: Prompts for skeleton generation
- file_designer: FileDesigner, LLMClient

For backward compatibility this package also re-exports the canonical RPG
data model and helpers from the ``rpg`` package:
- rpg.code_unit: CodeUnit, ParsedFile, ...
- rpg.models:    RPG, Node, NodeMetaData, NodeType, Edge, EdgeType
- rpg.builder:   create_initial_rpg, load_refactor_feature_data, get_rpg_statistics
- rpg.dep_graph: DependencyGraph

New code should import these directly from ``rpg.*`` instead of ``skeleton``.
"""

from .skeleton_models import RepoSkeleton, DirectoryNode, FileNode
from rpg.code_unit import (
    CodeUnit, ParsedFile,
    ParsedWorkspace, ParsedModule,
    CodeSnippetBuilder, merge_codeunits, class_ast_to_header_str,
    compare_code_units,
)
from .skeleton_prompts import (
    RAW_SKELETON_PROMPT,
    GROUP_SKELETON_PROMPT,
    RAW_SKELETON_REVIEW_PROMPT,
    GROUP_SKELETON_REVIEW_PROMPT,
    build_component_summary,
    extract_features_from_subtree,
    extract_leaf_descriptions_from_subtree,
    format_feature_list,
)
from rpg.models import RPG, Node, NodeMetaData, NodeType, Edge, EdgeType
from rpg.builder import create_initial_rpg, load_refactor_feature_data, get_rpg_statistics
from rpg.dep_graph import DependencyGraph

# Lazy import: FileDesigner depends on pydantic which may not be available
# in all environments (e.g., project .venv_dev used by run_batch.py)
def __getattr__(name):
    if name in ("FileDesigner", "LLMClient"):
        from .file_designer import FileDesigner, LLMClient
        globals()["FileDesigner"] = FileDesigner
        globals()["LLMClient"] = LLMClient
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # skeleton_models
    "RepoSkeleton",
    "DirectoryNode",
    "FileNode",
    # code_unit
    "CodeUnit",
    "ParsedFile",
    "ParsedWorkspace",
    "ParsedModule",
    "CodeSnippetBuilder",
    "merge_codeunits",
    "class_ast_to_header_str",
    "compare_code_units",
    # skeleton_prompts
    "RAW_SKELETON_PROMPT",
    "GROUP_SKELETON_PROMPT",
    "RAW_SKELETON_REVIEW_PROMPT",
    "GROUP_SKELETON_REVIEW_PROMPT",
    "build_component_summary",
    "extract_features_from_subtree",
    "extract_leaf_descriptions_from_subtree",
    "format_feature_list",
    # rpg_models
    "RPG",
    "Node",
    "NodeMetaData",
    "NodeType",
    "EdgeType",
    "Edge",
    # rpg_builder
    "create_initial_rpg",
    "load_refactor_feature_data",
    "get_rpg_statistics",
    # file_designer
    "FileDesigner",
    "LLMClient",
    # dep_graph
    "DependencyGraph",
]
