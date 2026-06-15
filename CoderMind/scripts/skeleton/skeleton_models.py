#!/usr/bin/env python3
"""Skeleton Models.

This module provides skeleton data structures for representing
the repository file structure.

Key classes:
- RepoNode: Base class for repository nodes
- DirectoryNode: Directory node
- FileNode: File node with feature assignments
- RepoSkeleton: Main skeleton structure
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any
from abc import abstractmethod
from pathlib import PurePosixPath

from rpg.code_unit import CodeUnit, ParsedFile


def normalize_path(path: str) -> str:
    """Normalize file path to unix style."""
    if not path:
        return "."
    # Convert to posix path and normalize
    posix_path = str(PurePosixPath(path))
    # Remove leading "./" if present
    if posix_path.startswith("./"):
        posix_path = posix_path[2:]
    # Handle empty path
    if not posix_path or posix_path == ".":
        return "."
    return posix_path


class RepoNode:
    """Base class for repository nodes."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = normalize_path(path)
        self.parent = None

    @property
    @abstractmethod
    def is_dir(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_file(self) -> bool:
        pass

    def children(self) -> List["RepoNode"]:
        return []

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        return isinstance(other, RepoNode) and self.path == other.path


class DirectoryNode(RepoNode):
    """Directory node in repository skeleton."""

    def __init__(self, name: str, path: str, tags: Optional[List[str]] = None):
        super().__init__(name, path)
        self.tags = tags or []
        self._children: List[RepoNode] = []

    def add_child(self, node: RepoNode):
        """Add child node, preventing duplicates."""
        if not any(child.path == node.path for child in self._children):
            node.parent = self
            self._children.append(node)

    @property
    def is_dir(self) -> bool:
        return True

    @property
    def is_file(self) -> bool:
        return False

    def children(self) -> List[RepoNode]:
        return self._children

    def has_tag(self, tag_name: str) -> bool:
        """Check if directory has a specific tag."""
        return any(tag == tag_name for tag in self.tags)

    def __repr__(self):
        tag_str = f" [tags: {', '.join(self.tags)}]" if self.tags else ""
        return f"DirectoryNode(name='{self.name}', path='{self.path}'){tag_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Remove duplicates while preserving order
        seen = set()
        unique_children = []
        for child in self._children:
            if child.path not in seen:
                unique_children.append(child)
                seen.add(child.path)
            else:
                logging.warning(f"Duplicate child path: {child.path}")

        return {
            "type": "directory",
            "name": self.name,
            "path": self.path,
            "tags": self.tags,
            "children": [child.to_dict() for child in unique_children],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DirectoryNode":
        """Create from dictionary."""
        return DirectoryNode(
            name=data["name"],
            path=data["path"],
            tags=data.get("tags", [])
        )


class FileNode(RepoNode):
    """File node in repository skeleton."""

    def __init__(self, name: str, path: str, code: str = "", feature_paths: Optional[List[str]] = None):
        super().__init__(name, path)
        self.code = code
        self.feature_paths = feature_paths if feature_paths else []
        # Parse code to extract units
        self.parsed = ParsedFile(code, path) if code else None
        self.units: List[CodeUnit] = self.parsed.units if self.parsed else []

    @property
    def is_file(self) -> bool:
        return True

    @property
    def is_dir(self) -> bool:
        return False

    def __repr__(self):
        return f"<FileNode path={self.path}, features={len(self.feature_paths)}>"

    def update_code(self, code: str):
        """Update code and re-parse units."""
        self.code = code
        self.parsed = ParsedFile(code, self.path) if code else None
        self.units = self.parsed.units if self.parsed else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (ZeroRepo compatible)."""
        return {
            "type": "file",
            "name": self.name,
            "path": self.path,
            "code": self.code,
            "feature_paths": self.feature_paths,
            "units": [unit.to_dict() for unit in self.units]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FileNode":
        """Create from dictionary (ZeroRepo compatible)."""
        file_node = FileNode(
            name=data["name"],
            path=data["path"],
            code=data.get("code", "")
        )
        file_node.feature_paths = data.get("feature_paths", [])
        # Restore units from saved data if available
        if data.get("units"):
            file_node.units = [CodeUnit.from_dict(u) for u in data["units"]]
        return file_node


class RepoSkeleton:
    """Repository skeleton structure."""

    def __init__(self, file_map: Optional[Dict[str, str]] = None):
        """Initialize repository skeleton.

        Args:
            file_map: Optional mapping of file_path -> code
        """
        self.root = DirectoryNode(name="project_root", path=".")
        self.path_to_node: Dict[str, RepoNode] = {".": self.root}

        if file_map:
            for file_path, code in sorted(file_map.items()):
                self._insert_file(file_path, code)

    def _insert_file(self, file_path: str, code: str, feature_paths: Optional[List[str]] = None):
        """Insert a file into the skeleton."""
        norm_path = normalize_path(file_path)
        parts = norm_path.split("/")
        current_node = self.root
        current_path = ""

        for i, part in enumerate(parts):
            if not part:
                continue

            is_last = i == len(parts) - 1
            node_path = normalize_path(os.path.join(current_path, part))

            if is_last:
                # Create file node
                if node_path in self.path_to_node:
                    existing = self.path_to_node[node_path]
                    if existing.is_dir:
                        logging.error(f"Path conflict: '{node_path}' exists as directory")
                        return
                    elif existing.is_file:
                        logging.info(f"Overwriting file at: {node_path}")
                        existing.code = code
                        if feature_paths:
                            existing.feature_paths = feature_paths
                        return

                file_node = FileNode(
                    name=part,
                    path=node_path,
                    code=code,
                    feature_paths=feature_paths or []
                )
                self.path_to_node[node_path] = file_node
                current_node.add_child(file_node)
            else:
                # Create or find directory node
                existing_node = self.path_to_node.get(node_path)
                if existing_node is None:
                    dir_node = DirectoryNode(name=part, path=node_path)
                    self.path_to_node[node_path] = dir_node
                    current_node.add_child(dir_node)
                    current_node = dir_node
                elif existing_node.is_dir:
                    current_node = existing_node
                else:
                    logging.error(f"Path conflict: '{node_path}' exists as file, expected directory")
                    return

            current_path = node_path

    def insert_file(self, file_path: str, code: str, feature_paths: Optional[List[str]] = None):
        """Public method to insert file."""
        self._insert_file(file_path, code, feature_paths)

    def find_file(self, path: str) -> Optional[FileNode]:
        """Find file by path."""
        norm_input = normalize_path(path)

        # Exact match first
        for key, node in self.path_to_node.items():
            if isinstance(node, FileNode) and normalize_path(key) == norm_input:
                return node

        # Suffix match fallback
        for key, node in self.path_to_node.items():
            if isinstance(node, FileNode) and normalize_path(key).endswith(norm_input):
                return node

        return None

    def find_dir(self, path: str) -> Optional[DirectoryNode]:
        """Find directory by path."""
        norm_input = normalize_path(path)

        # Exact match first
        for key, node in self.path_to_node.items():
            if isinstance(node, DirectoryNode) and normalize_path(key) == norm_input:
                return node

        # Suffix match fallback
        for key, node in self.path_to_node.items():
            if isinstance(node, DirectoryNode) and normalize_path(key).endswith(norm_input):
                return node

        return None

    def all_paths(self, include_dirs: bool = True, include_files: bool = True) -> List[str]:
        """Get all paths in skeleton."""
        return sorted(
            path for path, node in self.path_to_node.items()
            if (include_dirs and node.is_dir) or (include_files and node.is_file)
        )

    def find_files_by_feature_path(self, feature_path: str) -> List[FileNode]:
        """Find files containing a specific feature path."""
        return [
            node for node in self.path_to_node.values()
            if isinstance(node, FileNode) and feature_path in node.feature_paths
        ]

    def get_all_file_nodes(self) -> List[FileNode]:
        """Get all file nodes."""
        return [
            node for node in self.path_to_node.values()
            if isinstance(node, FileNode)
        ]

    def get_file_code_map(self) -> Dict[str, str]:
        """Get mapping of file paths to code."""
        return {
            node.path: node.code or ""
            for node in self.path_to_node.values()
            if isinstance(node, FileNode)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "root": self.root.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSkeleton":
        """Create from dictionary."""
        skeleton = object.__new__(cls)
        skeleton.path_to_node = {}

        def walk_and_register(node_data: Dict[str, Any], parent: Optional[DirectoryNode] = None) -> RepoNode:
            node_type = node_data["type"]

            if node_type == "directory":
                node = DirectoryNode.from_dict(node_data)
            elif node_type == "file":
                node = FileNode.from_dict(node_data)
            else:
                raise ValueError(f"Unknown node type: {node_type}")

            skeleton.path_to_node[node.path] = node

            if parent:
                # Prevent duplicate children
                if not any(child.path == node.path for child in parent.children()):
                    parent.add_child(node)

            node.parent = parent

            # Process children for directories
            if isinstance(node, DirectoryNode):
                for child_data in node_data.get("children", []):
                    walk_and_register(child_data, node)

            return node

        skeleton.root = walk_and_register(data["root"])
        return skeleton

    def save_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "RepoSkeleton":
        """Load from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_init_files(
        self,
        skip_root: bool = True,
        docstring_template: Optional[str] = None,
        backend: Optional[Any] = None,
    ) -> int:
        """Add package-marker files to all directories in the skeleton.

        When ``backend`` is supplied, the file name, content, and
        per-directory "has source" predicate are sourced from the
        backend. Backends whose :meth:`package_marker_filename` returns
        ``None`` (Go, Rust, TypeScript, …) make this method a no-op
        because directories without marker files are the language
        convention.

        Args:
            skip_root: Whether to skip adding the marker to the root.
            docstring_template: Optional template (``{name}`` /
                ``{path}``). Used only when the backend's
                :meth:`package_marker_content` returns None (i.e. the
                caller wants the built-in marker body).
            backend: Optional :class:`decoder_lang.LanguageBackend`.
                When ``None``, Python package-marker rules are used.

        Returns:
            Number of marker files added (0 for languages that don't
            use a marker file).
        """
        # When no backend is supplied, use Python package-marker rules.
        if backend is None:
            marker_filename: Optional[str] = "__init__.py"
            source_extension: str = ".py"
        else:
            marker_filename = backend.package_marker_filename()
            source_extension = backend.file_extension

        # Languages without a package marker (Go / Rust / TS) → no-op.
        if marker_filename is None:
            logging.debug(
                "add_init_files: backend %s has no package marker; skipping",
                getattr(backend, "name", "?"),
            )
            return 0

        init_files_added = 0

        # Get all directory nodes
        dir_nodes = [n for n in self.path_to_node.values() if isinstance(n, DirectoryNode)]

        for dir_node in dir_nodes:
            # Skip root directory if requested
            if skip_root and (dir_node.path == "." or dir_node == self.root):
                continue

            # Skip directories that contain no source files in this
            # language (mirrors the original heuristic, just
            # parameterised). Sub-directories still count so that an
            # empty package-only directory tree still gets markers
            # placed correctly.
            has_source_content = False
            for child in dir_node.children():
                if isinstance(child, FileNode) and child.name.endswith(source_extension):
                    has_source_content = True
                    break
                if isinstance(child, DirectoryNode):
                    has_source_content = True
                    break

            # Also add if the directory is under a common Python package
            # path. Non-Python backends opt out earlier via
            # ``marker_filename is None``.
            is_python_pkg_path = any(
                dir_node.path.startswith(prefix)
                for prefix in ['src/', 'lib/', 'pkg/', 'packages/']
            ) or '/src/' in dir_node.path

            if not has_source_content and not is_python_pkg_path:
                continue

            # Build marker file path
            init_path = normalize_path(os.path.join(dir_node.path, marker_filename))

            # Skip if marker already exists
            if init_path in self.path_to_node:
                continue

            # Generate content for the marker file
            if backend is not None:
                content = backend.package_marker_content(dir_node.path)
                # Backends that return None for content but emit a
                # marker (rare; not used today) still need *some* body.
                if content is None:
                    content = ""
                code = content
            elif docstring_template:
                code = docstring_template.format(
                    name=dir_node.name,
                    path=dir_node.path,
                )
            else:
                # Default minimal marker docstring.
                code = f'"""Package: {dir_node.name}"""\n'

            # Create marker file node
            init_node = FileNode(
                name=marker_filename,
                path=init_path,
                code=code,
                feature_paths=[],
            )

            # Add to directory and path registry
            dir_node.add_child(init_node)
            self.path_to_node[init_path] = init_node
            init_files_added += 1

            logging.debug(f"Added {marker_filename} to: {dir_node.path}")

        logging.info(f"Added {init_files_added} {marker_filename} files to skeleton")
        return init_files_added

    def get_statistics(self) -> Dict[str, Any]:
        """Get skeleton statistics."""
        total_nodes = len(self.path_to_node)
        file_nodes = [n for n in self.path_to_node.values() if isinstance(n, FileNode)]
        dir_nodes = [n for n in self.path_to_node.values() if isinstance(n, DirectoryNode)]

        total_features = sum(len(f.feature_paths) for f in file_nodes)
        init_files = len([f for f in file_nodes if f.name == "__init__.py"])

        return {
            "total_nodes": total_nodes,
            "file_nodes": len(file_nodes),
            "directory_nodes": len(dir_nodes),
            "total_features": total_features,
            "files_with_features": len([f for f in file_nodes if f.feature_paths]),
            "init_files": init_files,
        }

    def to_tree_string(self, skip_root: bool = True, show_features: bool = False) -> str:
        """Generate tree string representation."""
        def _render_node(node: RepoNode, prefix: str = "", is_last: bool = True) -> str:
            lines = []

            if not (skip_root and node == self.root):
                connector = "└── " if is_last else "├── "
                if node == self.root:
                    lines.append(node.name)
                else:
                    node_str = node.name
                    if show_features and isinstance(node, FileNode) and node.feature_paths:
                        node_str += f" ({len(node.feature_paths)} features)"
                    lines.append(f"{prefix}{connector}{node_str}")

            if isinstance(node, DirectoryNode):
                children = node.children()
                for i, child in enumerate(children):
                    is_child_last = (i == len(children) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ") if not (skip_root and node == self.root) else ""
                    lines.append(_render_node(child, child_prefix, is_child_last))

            return "\n".join(lines)

        return _render_node(self.root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    file_map = {
        "src/main.py": "# Main file",
        "src/utils/helpers.py": "# Helper functions",
        "tests/test_main.py": "# Tests"
    }

    skeleton = RepoSkeleton(file_map)
    print("Created skeleton with files:")
    for path in skeleton.all_paths(include_dirs=False):
        print(f"  {path}")

    print(f"\nSkeleton statistics: {skeleton.get_statistics()}")
    print(f"\nTree structure:\n{skeleton.to_tree_string()}")