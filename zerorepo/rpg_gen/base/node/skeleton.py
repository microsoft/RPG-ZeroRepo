from .node import DirectoryNode, RepoNode, FileNode
from typing import Dict, Optional, List
import logging
import json, os
import difflib
from pathlib import PurePosixPath
from zerorepo.utils.file import normalize_path
from .util import show_project_structure_from_tree

class RepoSkeleton:
    def __init__(self, file_map: Dict[str, str]):
        self.root = DirectoryNode(name="project_root", path=".")
        self.path_to_node: Dict[str, RepoNode] = {".": self.root}
        
        for file_path, code in sorted(file_map.items()):
            self.insert_file(file_path, code)

    @classmethod
    def from_repo_dir(
        cls,
        repo_root: str,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        include_exts: Optional[List[str]] = None
    ) -> "RepoSkeleton":
        """
        Create a RepoSkeleton instance from a local repository directory.

        Args:
            repo_root: Path to the repository root directory.
            exclude_dirs: Optional list of directory names to exclude (e.g., [".git", "__pycache__"]).
            exclude_files: Optional list of file names to exclude (e.g., ["README.md"]).
            include_exts: Optional list of file extensions to include only (e.g., [".py", ".md"]).

        Returns:
            RepoSkeleton: The constructed instance.
        """
        repo_root = os.path.abspath(repo_root)
        file_map: Dict[str, str] = {}

        exclude_dirs = set(exclude_dirs or [".git", "__pycache__", ".idea", ".vscode"])
        exclude_files = set(exclude_files or [])
        include_exts = set(include_exts or []) 

        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue
                if include_exts and not any(file.endswith(ext) for ext in include_exts):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
                rel_path = normalize_path(rel_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, OSError):

                    continue

                file_map[rel_path] = content

        return cls(file_map=file_map)
    
    def path_exists(self, path: str) -> bool:
        """
        Determine whether the given path exists in the current repository structure.

        Args:
            path: A path that can refer to either a file or a directory.
        Returns:
            bool: True if the path exists in the repository (whether file or directory); otherwise False.
        """
        if not path:
            return False

        norm_path = normalize_path(path)
       
        if norm_path in self.path_to_node:
            return True

        alt_path = norm_path.lstrip("./")
        if alt_path in self.path_to_node:
            return True

        for key in self.path_to_node.keys():
            if normalize_path(key).endswith(norm_path):
                return True

        return False
    
    def insert_file(self, file_path: str, code: str):
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
                if node_path in self.path_to_node:
                    existing = self.path_to_node[node_path]
                    if existing.is_dir:
                        logging.error(f"Path conflict: '{node_path}' exists as a directory.")
                        return
                    elif existing.is_file:
                        logging.info(f"Overwriting file at path: {node_path}")
                        existing.code = code
                        return
                file_node = FileNode(name=part, path=node_path, code=code)
                self.path_to_node[node_path] = file_node
                current_node.add_child(file_node)
            else:
                existing_node = self.path_to_node.get(node_path)
                if existing_node is None:
                    dir_node = DirectoryNode(name=part, path=node_path)
                    self.path_to_node[node_path] = dir_node
                    current_node.add_child(dir_node)
                    current_node = dir_node
                elif existing_node.is_dir:
                    current_node = existing_node
                else:
                    logging.error(f"Path conflict: '{node_path}' exists as file, expected directory.")
                    return

            current_path = node_path
            
    def find_file(self, path: str) -> Optional["FileNode"]:
        norm_input = normalize_path(path)

        for key, node in self.path_to_node.items():
            if isinstance(node, FileNode) and normalize_path(key) == norm_input:
                return node

        # Optional fallback: suffix match
        for key, node in self.path_to_node.items():
            if isinstance(node, FileNode) and normalize_path(key).endswith(norm_input):
                return node

        return None

    
    def find_dir(self, path: str) -> Optional["DirectoryNode"]:
        def normalize(p: str) -> str:
            return str(PurePosixPath(p))

        norm_input = normalize(path)

        # First try exact match
        for key, node in self.path_to_node.items():
            if isinstance(node, DirectoryNode) and normalize(key) == norm_input:
                return node

        # Fallback: exact suffix match
        for key, node in self.path_to_node.items():
            if isinstance(node, DirectoryNode) and normalize(key).endswith(norm_input):
                return node

        return None
    
    def tag_dir(self, name_to_path_map: Dict[str, List[str]], name_to_tree):
        for tag_name, paths in name_to_path_map.items():
            subtree = name_to_tree.get(tag_name)
            if not subtree:
                continue
            for path in paths:
                node = self.path_to_node.get(path)
                if isinstance(node, DirectoryNode) and subtree not in node.tags:
                    node.tags.append(subtree)
    
    def find_dir_by_tag(self, tag_name: str) -> List[DirectoryNode]:
        matched = []

        def walk(node: DirectoryNode):
            if isinstance(node, DirectoryNode):
                if node.has_tag(tag_name):
                    matched.append(node)
                for child in node.children():
                    if child.is_dir:
                        walk(child)

        walk(self.root)
        return matched 

    def to_tree_string(self, skip_root: bool = True, show_features=False, filter_func=lambda x: True) -> str:
        return show_project_structure_from_tree(
            node=self.root, 
            skip_root=skip_root, 
            show_features=show_features,
            filter_func=filter_func
        )
    
    def all_paths(self, include_dirs: bool = True, include_files: bool = True) -> List[str]:
        return sorted(
            path for path, node in self.path_to_node.items()
            if (include_dirs and node.is_dir) or (include_files and node.is_file)
        )
        
    def find_files_by_feature_path(self, feature_path: str) -> List[FileNode]:
        return [
            node for node in self.path_to_node.values()
            if isinstance(node, FileNode) and feature_path in node.feature_paths
        ]

    def update_repo_and_patch(self, update_files: Dict[str, str]) -> Dict[str, str]:
 
        patches = {}
        
        for file_path, new_content in update_files.items():
            normalized_path = file_path.strip("/")

            node = self.path_to_node.get(normalized_path)
            old_text = node.code if isinstance(node, FileNode) else ""

            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{file_path}",
                tofile=f"{file_path}"
            )
            patch = ''.join(diff)
            patches[file_path] = patch

            self.insert_file(normalized_path, new_content)

        return patches

    def to_dict(self) -> Dict:
        return {
            "root": self.root.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict) -> "RepoSkeleton":
        skeleton = object.__new__(RepoSkeleton)
        skeleton.path_to_node = {}

        def walk_and_register(data: Dict, parent: Optional[DirectoryNode] = None) -> RepoNode:
            node_type = data["type"]
            if node_type == "directory":
                node = DirectoryNode.from_dict(data)
            elif node_type == "file":
                node = FileNode.from_dict(data)
            else:
                raise ValueError(f"Unknown node type: {node_type}")

            skeleton.path_to_node[node.path] = node
            if parent:
                if not any(child.path == node.path for child in parent.children()):
                    parent.add_child(node)
            node.parent = parent

            if isinstance(node, DirectoryNode):
                for child_data in data.get("children", []):
                    walk_and_register(child_data, node)


            return node

        skeleton.root = walk_and_register(data["root"])

        return skeleton

    @classmethod
    def from_workspace(cls, workspace_root: str, file_extensions: Optional[List[str]] = None) -> "RepoSkeleton":
        from pathlib import Path
        
        workspace_path = Path(workspace_root)
        
        if not workspace_path.exists():
            logging.error(f"Workspace directory does not exist: {workspace_root}")
            return cls({})
        
        if file_extensions is None:
            file_extensions = ['.py'] 
        
        file_map = {}
        
        for file_path in workspace_path.rglob('*'):
            if file_path.is_dir():
                continue
            if file_path.name.startswith('.'):
                continue
            if any(exclude in str(file_path) for exclude in ['__pycache__', '.git', '.venv', 'venv', '.pyc']):
                continue
        
            if file_extensions and file_path.suffix not in file_extensions:
                continue
            
            try:
                relative_path = file_path.relative_to(workspace_path)

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_map[str(relative_path)] = content
                
            except Exception as e:
                logging.warning(f"Failed to read file {file_path}: {e}")
        
        logging.info(f"Created RepoSkeleton from workspace with {len(file_map)} files")
        return cls(file_map)

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(filepath: str) -> "RepoSkeleton":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RepoSkeleton.from_dict(data)
    
    def write_to_workspace(self, workspace_root: str):
    
        for path, node in self.path_to_node.items():
            if node == self.root:
                continue
            abs_path = os.path.join(workspace_root, path)

            if node.is_dir:
                os.makedirs(abs_path, exist_ok=True)

            elif node.is_file:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(node.code or "")
                    
    def get_file_code_map(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping file paths to their code content.
        This includes all files tracked by the RepoSkeleton.
        """
        file_code_map = {}
        for path, node in self.path_to_node.items():
            if isinstance(node, FileNode):
                file_code_map[node.path] = node.code if node.code is not None else ""
        return file_code_map
    
    def get_all_file_nodes(self) -> List[FileNode]:
        """
        Returns a list of all FileNode instances in the repository.
        This excludes directories and only includes valid source files.
        """
        return [
            node for node in self.path_to_node.values()
            if isinstance(node, FileNode)
        ]
    

    def add_missing_init_files(self) -> None:
        """
        Add missing __init__.py files to directories that should be treated as Python packages.

        Rules:
        - Traverse the directory tree bottom-up;
        - A directory is considered a "package directory" if it or any of its descendants
        contains at least one .py file;
        - For each "package directory", if __init__.py does not exist under it, create an empty file;
        - The root directory "." does not receive an __init__.py by default
        (the project root usually does not need to be a package).
        """
        def visit_dir(dir_node: DirectoryNode) -> bool:
            has_python_code = False

            for child in dir_node.children():
                if isinstance(child, DirectoryNode):
                    if visit_dir(child):
                        has_python_code = True
                elif isinstance(child, FileNode):
                    if child.name.endswith(".py"):
                        has_python_code = True

            if has_python_code and dir_node.path != ".":
                init_path = normalize_path(os.path.join(dir_node.path, "__init__.py"))

                if init_path not in self.path_to_node:
    
                    init_node = FileNode(
                        name="__init__.py",
                        path=init_path,
                        code="" 
                    )
                    self.path_to_node[init_path] = init_node
                    dir_node.add_child(init_node)
                    logging.info(f"[RepoSkeleton] Added missing __init__.py at: {init_path}")

            return has_python_code

        if isinstance(self.root, DirectoryNode):
            visit_dir(self.root)