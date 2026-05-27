#!/usr/bin/env python3
"""Query data types for RPG Agent search results.

Provides ``QueryInfo`` (search query metadata) and ``QueryResult`` (a single
result entry with formatting logic).

Ported from: RPG-ZeroRepo/zerorepo/rpg_encoder/rpg_agent/env/query.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from common.utils import get_skeleton, wrap_code_snippet
from rpg import NodeType

if TYPE_CHECKING:
    from rpg_agent.env.searcher import RepoEntitySearcher


# ============================================================================
# QueryInfo
# ============================================================================

class QueryInfo:
    """Metadata attached to a single search query / term."""

    query_type: str = "keyword"
    term: Optional[str] = None
    line_nums: Optional[List[int]] = None
    file_path_or_pattern: Optional[str] = None

    def __init__(
        self,
        query_type: str = "keyword",
        term: Optional[str] = None,
        line_nums: Optional[List[int]] = None,
        file_path_or_pattern: Optional[str] = None,
    ):
        self.query_type = query_type
        if term is not None:
            self.term = term
        if line_nums is not None:
            self.line_nums = line_nums
        if file_path_or_pattern is not None:
            self.file_path_or_pattern = file_path_or_pattern

    def __str__(self) -> str:
        parts: List[str] = []
        if self.term is not None:
            parts.append(f"term: {self.term}")
        if self.line_nums is not None:
            parts.append(f"line_nums: {self.line_nums}")
        if self.file_path_or_pattern is not None:
            parts.append(f"file_path_or_pattern: {self.file_path_or_pattern}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return self.__str__()

    # Required for use as dict key (via tuple) --
    def __hash__(self) -> int:
        return hash((self.term, self.query_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryInfo):
            return NotImplemented
        return self.term == other.term and self.query_type == other.query_type


# ============================================================================
# QueryResult
# ============================================================================

class QueryResult:
    """A single search result entry with rich formatting support."""

    file_path: Optional[str] = None
    format_mode: Optional[str] = "complete"
    nid: Optional[str] = None
    ntype: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    query_info_list: Optional[List[QueryInfo]] = None
    desc: Optional[str] = None
    message: Optional[str] = None
    warning: Optional[str] = None
    retrieve_src: Optional[str] = None

    def __init__(
        self,
        query_info: QueryInfo,
        format_mode: str,
        nid: Optional[str] = None,
        ntype: Optional[str] = None,
        file_path: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        desc: Optional[str] = None,
        message: Optional[str] = None,
        warning: Optional[str] = None,
        retrieve_src: Optional[str] = None,
    ):
        self.format_mode = format_mode
        self.query_info_list: List[QueryInfo] = []
        self.insert_query_info(query_info)

        if nid is not None:
            self.nid = nid

        if ntype is not None:
            self.ntype = ntype
            if ntype in [
                NodeType.FILE, NodeType.CLASS,
                NodeType.METHOD, NodeType.FUNCTION,
            ]:
                self.file_path = nid.split(":")[0] if nid else None

        if file_path is not None:
            self.file_path = file_path
        if start_line is not None and end_line is not None:
            self.start_line = start_line
            self.end_line = end_line

        if retrieve_src is not None:
            self.retrieve_src = retrieve_src
        if desc is not None:
            self.desc = desc
        if message is not None:
            self.message = message
        if warning is not None:
            self.warning = warning

    # ------------------------------------------------------------------

    def insert_query_info(self, query_info: QueryInfo) -> None:
        self.query_info_list.append(query_info)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def format_output(self, searcher: "RepoEntitySearcher") -> str:
        """Render this result to a human-readable markdown string.

        The output style depends on ``format_mode``:
        - ``complete``: Full code content (with skeleton fallback for long code).
        - ``preview``: Code for functions/methods; skeleton for large classes/files.
        - ``code_snippet``: Raw line-range extract from the file.
        - ``fold``: Compact one-liner (name + feature paths, no code).
        """
        cur_result = ""

        if self.format_mode == "complete":
            cur_result = self._format_complete(searcher)
        elif self.format_mode == "preview":
            cur_result = self._format_preview(searcher)
        elif self.format_mode == "code_snippet":
            cur_result = self._format_code_snippet(searcher)
        elif self.format_mode == "fold":
            cur_result = self._format_fold(searcher)

        return cur_result

    # --- Private formatters -----------------------------------------------

    def _format_complete(self, searcher: "RepoEntitySearcher") -> str:
        node_data_list = searcher.get_node_data([self.nid], return_code_content=True)
        if not node_data_list:
            return f"Entity `{self.nid}` not found in repository.\n"
        node_data = node_data_list[0]
        ntype = node_data.get("type")
        feature_paths = node_data.get("feature_paths", [])[:2]
        feature_paths_str = "\n".join(feature_paths)

        cur = f"Found {ntype} `{self.nid}`.\n"
        cur += f"Source: {self.retrieve_src}\n"
        if feature_paths_str.strip():
            cur += f"It Functionality Features: {feature_paths_str}\n"

        code = node_data.get("code_content", "")
        if code:
            code_lines = len(code.split("\n"))
            if code_lines > 400:
                new_data = searcher.get_node_data(
                    [self.nid], return_code_content=True, wrap_with_ln=False
                )
                raw_code = new_data[0].get("code_content", "") if new_data else ""
                skeleton = get_skeleton(
                    raw_code,
                    keep_constant=True,
                    keep_indent=True,
                    keep_imports=True,
                    compress_assign=False,
                    keep_docstring=False,
                    total_lines=400,
                    prefix_lines=200,
                    suffix_lines=200,
                    line_number_mode="original",
                )
                cur += (
                    f"Note: The code for `{self.nid}` is very long "
                    f"({code_lines} lines) and exceeds the display limit. "
                    "Only a structural skeleton is shown.\n"
                )
                cur += skeleton + "\n"
            else:
                cur += code + "\n"
        return cur

    def _format_preview(self, searcher: "RepoEntitySearcher") -> str:
        node_data_list = searcher.get_node_data([self.nid], return_code_content=True)
        if not node_data_list:
            return f"Entity `{self.nid}` not found in repository.\n"
        node_data = node_data_list[0]
        ntype = node_data["type"]
        feature_paths = node_data.get("feature_paths", [])[:2]
        feature_paths_str = "\n".join(feature_paths)

        cur = f"Found {ntype} `{self.nid}`.\n"
        cur += f"Source: {self.retrieve_src}\n"
        if feature_paths_str:
            cur += f"It Functionality Features: {feature_paths_str}\n"

        if ntype in (NodeType.FUNCTION, NodeType.METHOD):
            cur += node_data.get("code_content", "") + "\n"
        elif ntype in (NodeType.CLASS, NodeType.FILE):
            start_line = node_data.get("start_line", 0)
            end_line = node_data.get("end_line", 0)
            content_size = (end_line - start_line) if (end_line and start_line) else 0
            if content_size <= 100:
                cur += node_data.get("code_content", "") + "\n"
            else:
                cur += f"Just show the structure of this {ntype} due to response length limitations:\n"
                code_content = searcher.G.nodes[self.nid].get("code", "") if searcher.G and self.nid in searcher.G.nodes else ""
                structure = get_skeleton(
                    code_content,
                    keep_constant=True,
                    keep_indent=True,
                    keep_imports=True,
                    compress_assign=False,
                    keep_docstring=False,
                    total_lines=500,
                    prefix_lines=200,
                    suffix_lines=200,
                    line_number_mode="original",
                )
                cur += "```\n" + structure + "\n```\n"
                cur += f"Hint: Search `{self.nid}` to get the full content if needed.\n"
        return cur

    def _format_code_snippet(self, searcher: "RepoEntitySearcher") -> str:
        cur = ""
        if self.desc:
            cur += self.desc + "\n"
        else:
            cur += f"Found code snippet in file `{self.file_path}`.\n"
        cur += f"Source: {self.retrieve_src}\n"

        node_data_list = searcher.get_node_data([self.file_path], return_code_content=True)
        if not node_data_list:
            return cur + f"File `{self.file_path}` not found in repository.\n"
        node_data = node_data_list[0]
        feature_paths = node_data.get("feature_paths", [])[:2]
        feature_paths_str = "\n".join(feature_paths)
        if feature_paths_str.strip():
            cur += f"It Functionality Features: {feature_paths_str}\n"

        code_content = node_data.get("code_content", "")
        content = code_content.split("\n")[1:-1] if code_content else []

        start = (self.start_line - 1) if self.start_line else 0
        end = self.end_line if self.end_line else len(content)
        snippet = content[start:end]
        cur += "```\n" + "\n".join(snippet) + "\n```\n"

        if self.message and self.message.strip():
            cur += self.message
        return cur

    def _format_fold(self, searcher: "RepoEntitySearcher") -> str:
        node_data_list = searcher.get_node_data([self.nid], return_code_content=False)
        if not node_data_list:
            return f"Entity `{self.nid}` not found in repository.\n"
        node_data = node_data_list[0]
        feature_paths = node_data.get("feature_paths", [])[:2]
        feature_paths_str = "\n".join(feature_paths)
        self.ntype = node_data.get("type")
        cur = f"Found {self.ntype} `{self.nid}`.\n"
        if feature_paths_str:
            cur += f"It Functionality Features: {feature_paths_str}\n"
        return cur

    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"QueryResult(\n"
            f"  query_info_list: {self.query_info_list},\n"
            f"  format_mode: {self.format_mode},\n"
            f"  nid: {self.nid},\n"
            f"  file_path: {self.file_path},\n"
            f"  start_line: {self.start_line},\n"
            f"  end_line: {self.end_line}\n"
            f")"
        )
