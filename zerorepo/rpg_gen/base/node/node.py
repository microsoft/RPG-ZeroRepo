
from typing import List, Optional, Dict
from abc import abstractmethod
import logging
from ..unit import CodeUnit, ParsedFile
from zerorepo.utils.file import normalize_path

class RepoNode:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = normalize_path(path)  # 相对路径
        self.parent = None
    
    @property
    @abstractmethod
    def is_dir(self) -> bool:
        ...

    @property
    @abstractmethod  
    def is_file(self) -> bool:
        ...

    def children(self) -> List["RepoNode"]:
        return []
    
    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        return isinstance(other, RepoNode) and self.path == other.path

class DirectoryNode(RepoNode):
    def __init__(self, name: str, path: str, tags: Optional[List[str]] = None):
        super().__init__(name, path)
        self.tags = tags or []
        self._children: List[RepoNode] = []

    def add_child(self, node: RepoNode):
        if not any(child.path == node.path for child in self._children):
            node.parent = self
            self._children.append(node)

    @property
    def is_dir(self) -> bool:
        return True

    def children(self) -> List[RepoNode]:
        return self._children

    def has_tag(self, tag_name: str) -> bool:
        return any(tag == tag_name for tag in self.tags)

    def __repr__(self):
        tag_names = [tag for tag in self.tags]
        tag_str = f" [tags: {', '.join(tag_names)}]" if tag_names else ""
        return f"DirectoryNode(name='{self.name}', path='{self.path}'){tag_str}"

    def to_dict(self):
        seen = set()
        unique_children = []
        for child in self._children:
            if child.path not in seen:
                unique_children.append(child)
                seen.add(child.path)
            else:
                logging.warning(f"Duplicate child path in to_dict: {child.path}")
        
        return {
            "type": "directory",
            "name": self.name,
            "path": self.path,
            "tags": self.tags,
            "children": [child.to_dict() for child in unique_children],
        }

    @staticmethod
    def from_dict(data: Dict) -> "DirectoryNode":
        return DirectoryNode(name=data["name"], path=data["path"], tags=data.get("tags", []))
    
class FileNode(RepoNode):
    def __init__(self, name: str, path: str, code: str, feature_paths: Optional[List[str]] = None):
        super().__init__(name, path)
        self.code = code
        self.feature_paths = feature_paths if feature_paths else []
        self.parsed = ParsedFile(code, path)
        self.units: List[CodeUnit] = self.parsed.units

    @property
    def is_file(self) -> bool:
        return True

    @property
    def is_dir(self) -> bool:
        return False

    def __repr__(self):
        return f"<FileNode path={self.path}, features={len(self.feature_paths)}>"

    def to_dict(self):
        return {
            "type": "file",
            "name": self.name,
            "path": self.path,
            "code": self.code,
            "feature_paths": self.feature_paths,
            "units": [unit.to_dict() for unit in self.units]
        }

    @staticmethod
    def from_dict(data: Dict) -> "FileNode":

        file_node = FileNode(
            name=data["name"],
            path=data["path"],
            code=data["code"]
        )
        file_node.feature_paths = data.get("feature_paths", [])  # ✅ 确保明确赋值
        file_node.units = [CodeUnit.from_dict(u) for u in data.get("units", [])]
        return file_node