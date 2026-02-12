import uuid
import json
import logging
import os
from dataclasses import dataclass, field
from typing import (
    Dict, List, Union,
    Optional, Tuple,
    Any, Callable
)
from collections import defaultdict, deque
from .util import NodeType, EdgeType
from .dep_graph import DependencyGraph
from ..unit import CodeUnit


MAX_LEVEL = 5

@dataclass
class NodeMetaData:
    type_name: NodeType = None
    path: Union[str, List[str]] = None
    description: str = ""
    content: str=""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_name": self.type_name.value if self.type_name else None,
            "path": self.path,
            "content": self.content,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["NodeMetaData"]:
        if not d:
            return None
        tn_raw = d.get("type_name")
        type_name = NodeType(tn_raw) if tn_raw in {m.value for m in NodeType} else None
        return cls(
            type_name=type_name,
            path=d.get("path"),
            description=d.get("description", ""),
            content=d.get("content", "")
        )
    
    
LEVEL_LABEL = {0:"L0",1:"L1",2:"L2",3:"L3",4:"L4",5:"L5"}

@dataclass
class Node:
    id: str
    node_type: Optional[str] = None
    name: str = ""
    level: Optional[int] = MAX_LEVEL
    unit: Optional[Tuple] = None
    meta: Optional[NodeMetaData] = None 
    _graph: Optional["RPG"] = field(default=None, repr=False, compare=False)

    def parent(self) -> Optional["Node"]:
        if not self._graph:
            return None
        pid = self._graph._parents.get(self.id)
        return self._graph.nodes.get(pid) if pid else None

    def children(self, recursive=False) -> List["Node"]:
        if not self._graph:
            return []
        child_ids = self._graph.get_children(self.id, recursive)
        return [self._graph.nodes[cid] for cid in child_ids]

    def path_to_root(self) -> List["Node"]:
        if not self._graph:
            return [self]
        ids = self._graph.get_path_to_root(self.id)
        return [self._graph.nodes[i] for i in ids]

    def feature_path(self, sep="/") -> str:
        nodes = self.path_to_root()
        names = [n.name for n in nodes if n.level is None or n.level > 0]
        return sep.join(names)

    def __str__(self):
        return f"<Node {self.name} (L{self.level}, {self.node_type})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "level": self.level,
            "meta": self.meta.to_dict() if self.meta else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        meta = NodeMetaData.from_dict(d.get("meta"))
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            node_type=d.get("node_type"),
            level=d.get("level"),
            meta=meta,
        )

@dataclass
class Edge:
    src: str
    dst: str
    relation: EdgeType = EdgeType.COMPOSES
    meta: NodeMetaData = field(default_factory=NodeMetaData)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "meta": self.meta.to_dict() if self.meta else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        meta = NodeMetaData.from_dict(d.get("meta"))
        return cls(
            src=d["src"],
            dst=d["dst"],
            relation=d.get("relation", EdgeType.COMPOSES),
            meta=meta or NodeMetaData(),
        )


LEVEL_LABEL = {0:"L0",1:"L1",2:"L2",3:"L3",4:"L4",5:"L5"}

class RPG:
    
    MAX_LEVEL = 5  # Default lowest level (feature level)
        
    def __init__(self, repo_name: str, repo_info: str="", excluded_files: List[str]=[]):

        self.repo_info = repo_info
        self.excluded_files = excluded_files
        self.repo_name = repo_name
        self.data_flow = []
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, str] = {}

        self.dep_graph: Optional[DependencyGraph] = None
        self._dep_to_rpg_map: Dict[str, List[str]] = {}  # dep_node_id -> [rpg_node_id, ...]

        repo_id = f"{repo_name}_L0"
        self.repo_node = Node(
            id=repo_id,
            name=repo_name,
            node_type="repo",
            level=0, meta=NodeMetaData(type_name=NodeType.DIRECTORY, path="."))
        self.add_node(self.repo_node)


    def get_nodes_by_type(self, type_name: Union[str, NodeType]) -> List[Node]:
        """Generic version: filter nodes by node_type or the NodeType enum."""
        if isinstance(type_name, NodeType):
            type_name = type_name.value
        return [
            node for node in self.nodes.values()
            if node.meta and node.meta.type_name and node.meta.type_name.value == type_name
        ]
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id, None)

    def get_node_by_feature_path(self, feature_path: str, sep: str = "/") -> Optional[Node]:
        if not feature_path:
            return None

        feature_path = feature_path.strip().strip(sep)

        for node in self.nodes.values():
            try:
                node_path = node.feature_path(sep=sep)
                if node_path == feature_path:
                    return node
            except Exception:
                continue

        return None
    
    def _rand8(self) -> str:
        return uuid.uuid4().hex[:8]

    def get_functional_areas(self) -> List[str]:
        """
        Extract top-level functional areas (first-level groupings under repo_node).

        Returns:
            List[str]: A list of unique functional area names.
        """
        if not hasattr(self, "repo_node") or not self.repo_node:
            logging.warning("RPG has no repo_node defined.")
            return []

        top_level_areas = []
        for edge in self.edges:
           
            if edge.src == self.repo_node.id:
                dst_node = self.nodes.get(edge.dst)
                if not dst_node:
                    continue

                if dst_node.meta and dst_node.meta.type_name == NodeType.DIRECTORY:
                    top_level_areas.append(dst_node.name)

        unique_areas = sorted(set(top_level_areas))
        logging.info(f"RPG functional areas detected: {unique_areas}")
        return unique_areas
        

    def find_child_by_name(self, parent_id: str, name: str) -> Optional[Node]:
        for cid in self._adjacency.get(parent_id, []):
            child = self.nodes.get(cid)
            if child and child.name == name:
                return child
        return None

    def add_node(self, node: Node):
      
        node._graph = self

        if node.level is None:
            node.level = self.MAX_LEVEL
        else:
            node.level = min(node.level, self.MAX_LEVEL)
        
        if node.node_type == "repo":
            node.level = 0
            
        node.node_type = self._infer_node_type(node.level)

        self.nodes[node.id] = node

    def _infer_node_type(self, level: int) -> str:
        mapping = {
            0: "repo",
            1: "functional_area",
            2: "category",
            3: "subcategory",
            4: "feature_group",
            5: "feature",
        }
        return mapping.get(level, "feature")

    def _update_levels_upwards(self, start_node_id: str):
        """
        (Core logic)
        Starting from the specified node, recursively update the level and node_type for all ancestor nodes.
        A parent node's level = max(0, min(children.level) - 1).
        A leaf node's level = MAX_LEVEL (repo_node is always 0).
        """
        current_id = start_node_id
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break  
            child_ids = self._adjacency.get(current_id, [])
            
            new_level = 0  
            if not child_ids:
                if current_id != self.repo_node.id:
                    new_level = self.MAX_LEVEL 
            else:
                try:
                    child_levels = [self.nodes[cid].level for cid in child_ids if cid in self.nodes and self.nodes[cid].level is not None]
                    
                    if not child_levels:
                        if current_id != self.repo_node.id:
                            new_level = self.MAX_LEVEL
                    else:
                        min_child_level = min(child_levels)
                        new_level = max(0, min_child_level - 1)
                        
                except Exception as e:
                    print(f"Warning: Could not infer level for {node.name} ({current_id}) due to child error: {e}. Stopping level update.")
                    break 
                
            if node.level != new_level:
                node.level = new_level
                node.node_type = self._infer_node_type(new_level)
                
                current_id = self._parents.get(current_id)
            else:
             
                break

    def add_edge(self, src: str, dst: str, relation=EdgeType.COMPOSES, meta=None):
        """
        Add an edge.
        - For COMPOSES edges (hierarchical parent-child relationships): update _adjacency/_parents and trigger upward level updates.
        - For other edges (INVOKES, IMPORTS, etc.): only store the edge and do not affect the hierarchical structure.
        """
        if isinstance(src, Node):
            src = src.id
        if isinstance(dst, Node):
            dst = dst.id

        if src not in self.nodes:
            print(f"Error: Source node not found. Src: {src}")
            return
        if dst not in self.nodes:
            print(f"Error: Destination node not found. Dst: {dst}")
            return

        relation_str = str(relation).lower() if relation else "composes"
        is_hierarchy_edge = relation_str in ("composes", "contains")

        if is_hierarchy_edge:
            if dst in self._adjacency.get(src, []):
                return

            anc_path = self.get_path_to_root(src)
            if dst in anc_path:
                print(f"Error: Cannot add edge from {src} to {dst}. Circular dependency detected.")
                return

        meta = meta or None
        edge = Edge(src=src, dst=dst, relation=relation, meta=meta)
        self.edges.append(edge)

        if is_hierarchy_edge:
            self._adjacency[src].append(dst)
            self._parents[dst] = src

            dst_node = self.nodes[dst]
            dst_node.node_type = self._infer_node_type(dst_node.level)

            self._update_levels_upwards(src)

    def get_children(self, node_id: str, recursive=False) -> List[str]:
        if not recursive:
            return [cid for cid in self._adjacency.get(node_id, []) if cid in self.nodes]

        result = []
        queue = deque([node_id])
        while queue:
            cur = queue.popleft()
            for child in self._adjacency.get(cur, []):
                if child not in self.nodes:
                    continue
                result.append(child)
                queue.append(child)
        return result

    def get_path_to_root(self, node_id: str) -> List[str]:
        path = []
        current_id = node_id
        while current_id:
            if current_id in path: 
                print(f"Warning: Circular dependency detected in path for node {node_id}.")
                break
            path.append(current_id)
            current_id = self._parents.get(current_id)
            
        return list(reversed(path))
    
    def to_dict(self, include_dep_graph: bool=True) -> Dict[str, Any]:
        """
        Export as a JSON-serializable dictionary.
        Args:
            include_dep_graph: Whether to include DependencyGraph data
        """

        hierarchy_edges = []
        dep_edges_existing = []
        existing_edge_keys = set() 
        
        for e in self.edges:
            rel_str = str(e.relation).lower()
            edge_dict = e.to_dict()
            edge_key = (e.src, e.dst, rel_str)

            if rel_str in ("composes", "contains"):
                edge_dict["relation"] = "composes"
                hierarchy_edges.append(edge_dict)
            else:
                dep_edges_existing.append(edge_dict)
                existing_edge_keys.add(edge_key)

        result = {
            "repo_name": self.repo_name,
            "repo_info": self.repo_info,
            "data_flow": self.data_flow,
            "excluded_files": self.excluded_files,
            "repo_node_id": self.repo_node.id if self.repo_node else None,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": hierarchy_edges,
            "_dep_to_rpg_map": self._dep_to_rpg_map,
        }

        if include_dep_graph and self.dep_graph is not None:
            # Pass dep_to_rpg_map to include RPG node mappings in dep_graph edges
            result["dep_graph"] = self.dep_graph.to_dict(dep_to_rpg_map=self._dep_to_rpg_map)
            result["edges"].extend(dep_edges_existing)
            dep_edges = self.get_dep_edges_for_rpg()
            for edge in dep_edges:
                edge_key = (edge["src"], edge["dst"], edge.get("relation", "").lower())
                if edge_key not in existing_edge_keys:
                    result["edges"].append(edge)
                    existing_edge_keys.add(edge_key)
        elif dep_edges_existing:
            result["edges"].extend(dep_edges_existing)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RPG":
        repo_name = data.get("repo_name", "repo")
        data_flow = data.get("data_flow", [])
        repo_info = data.get("repo_info", "")
        excluded_files = data.get("excluded_files", [])
        rpg = cls(repo_name=repo_name, repo_info=repo_info,
            excluded_files=excluded_files
        )
        rpg.data_flow = data_flow

        rpg.nodes.clear()
        rpg.edges.clear()
        rpg._adjacency.clear()
        rpg._parents.clear()

        id2node: Dict[str, Node] = {}
        for nd in data.get("nodes", []):
            node = Node.from_dict(nd)
            node._graph = rpg
            id2node[node.id] = node
            rpg.add_node(node)

        repo_node_id = data.get("repo_node_id")
        if repo_node_id and repo_node_id in rpg.nodes:
            rpg.repo_node = rpg.nodes[repo_node_id]
        else:
        
            found_root = None
            for n in rpg.nodes.values():
                if n.name == repo_name and n.level == 0:
                    found_root = n
                    break

            if found_root:
                rpg.repo_node = found_root
            else:
                repo_id = f"{repo_name}_L0"
                rpg.repo_node = Node(id=repo_id, name=repo_name, node_type="repo", level=0)
                rpg.add_node(rpg.repo_node)

        rpg.repo_node.level = 0
        rpg.repo_node.node_type = "repo"

        for ed in data.get("edges", []):
            edge = Edge.from_dict(ed)
            if edge.src not in rpg.nodes or edge.dst not in rpg.nodes:
                continue

            rpg.add_edge(src=edge.src, dst=edge.dst, relation=edge.relation, meta=edge.meta)

        rpg._dep_to_rpg_map = data.get("_dep_to_rpg_map", {})

        dep_graph_data = data.get("dep_graph")
        if dep_graph_data:
            rpg.dep_graph = DependencyGraph.from_dict(dep_graph_data)

        return rpg

    def save_json(self, path: str, ensure_ascii: bool = False, indent: int = 2):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def load_json(cls, path: str) -> "RPG":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    
    def get_functionality_graph(self) -> List[Dict[str, Any]]:
        def build_subtree(node_id: str):
            children_ids = self.get_children(node_id)
            node = self.nodes[node_id]
            
            if not children_ids:
                return [node.name] if node.level == self.MAX_LEVEL else {}

            first_child = self.nodes.get(children_ids[0])
            if first_child and first_child.level == self.MAX_LEVEL:
                return sorted([self.nodes[cid].name for cid in children_ids if cid in self.nodes])

            subtree = {}
            sorted_child_ids = sorted(children_ids, key=lambda cid: self.nodes[cid].name)
            
            for cid in sorted_child_ids:
                child = self.nodes[cid]
                child_sub = build_subtree(cid)
                subtree[child.name] = child_sub
            return subtree


        level1_nodes = [n for n in self.nodes.values() if n.level == 1]
        level1_nodes.sort(key=lambda n: n.name)

        results = []
        for n in level1_nodes:
            subtree = build_subtree(n.id)
            # Filter out empty subtrees (empty dict or empty list)
            if subtree and subtree != {} and subtree != []:
                results.append({
                    "name": n.name,
                    "refactored_subtree": subtree
                })
            else:
                logging.info(f"Filtering out empty subtree: {n.name}")

        return results
    
    def remove_empty_subtrees(self) -> Dict[str, int]:
        """
        Remove empty subtree nodes (level 1 nodes with no descendants).
        This helps clean up the graph after reorganization.
        """
        removed_nodes = 0
        removed_edges = 0
        
        # Find all level 1 nodes that have no children
        level1_nodes = [n for n in self.nodes.values() if n.level == 1]
        nodes_to_remove = []
        
        for node in level1_nodes:
            children = self.get_children(node.id, recursive=True)
            if not children:
                nodes_to_remove.append(node.id)
                logging.info(f"Removing empty subtree node: {node.name} (id: {node.id})")
        
        # Remove the empty nodes
        for node_id in nodes_to_remove:
            # Remove edges
            edges_before = len(self.edges)
            self.edges = [e for e in self.edges if e.src != node_id and e.dst != node_id]
            removed_edges += edges_before - len(self.edges)
            
            # Remove from parent's adjacency
            parent_id = self._parents.pop(node_id, None)
            if parent_id and node_id in self._adjacency.get(parent_id, []):
                self._adjacency[parent_id].remove(node_id)
            
            # Remove node's adjacency list
            self._adjacency.pop(node_id, None)
            
            # Remove the node itself
            if node_id in self.nodes:
                del self.nodes[node_id]
                removed_nodes += 1
        
        if removed_nodes > 0:
            logging.info(f"Removed {removed_nodes} empty subtree nodes and {removed_edges} associated edges")
        
        return {
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges
        }
    
    def delete_file_nodes(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Delete file-type nodes (NodeType.FILE) and all their descendants.
        Then recursively delete parent nodes *only along affected branches*
        if they become empty (no remaining children).
        """
        if not file_paths:
            logging.warning("⚠️ No file paths provided to delete.")
            return {"deleted_nodes": 0, "deleted_edges": 0, "cleaned_parents": 0}

        rel_path_set = set(file_paths)
        logging.info(f"Deleting file nodes for: {rel_path_set}")

        # --- Step 1. Find matching file nodes ---
        target_file_ids = [
            nid for nid, node in self.nodes.items()
            if node.meta
            and node.meta.type_name == NodeType.FILE
            and isinstance(node.meta.path, str)
            and any(node.meta.path.endswith(rp) for rp in rel_path_set)
        ]
        if not target_file_ids:
            logging.info("⚠️ No matching file-type nodes found for deletion.")
            return {"deleted_nodes": 0, "deleted_edges": 0, "cleaned_parents": 0}

        # --- Step 2. Collect descendants to delete ---
        all_delete_ids = set(target_file_ids)
        for fid in target_file_ids:
            for cid in self.get_children(fid, recursive=True):
                all_delete_ids.add(cid)

        # --- Step 3. Delete edges & nodes ---
        before_edges = len(self.edges)
        self.edges = [e for e in self.edges if e.src not in all_delete_ids and e.dst not in all_delete_ids]
        deleted_edges = before_edges - len(self.edges)

        affected_parents = set()
        for nid in all_delete_ids:
            parent_id = self._parents.pop(nid, None)
            if parent_id:
                affected_parents.add(parent_id)
                if nid in self._adjacency.get(parent_id, []):
                    self._adjacency[parent_id].remove(nid)
            self._adjacency.pop(nid, None)
            if nid in self.nodes:
                del self.nodes[nid]

        deleted_nodes = len(all_delete_ids)

        # --- Step 4. Recursively remove empty parent nodes (only along affected paths) ---
        cleaned_parents = 0
        queue = deque(affected_parents)
        visited = set()

        while queue:
            pid = queue.popleft()
            if pid in visited:
                continue
            visited.add(pid)

            children = [c for c in self._adjacency.get(pid, []) if c in self.nodes]
            self._adjacency[pid] = children

            if not children:
                pnode = self.nodes.get(pid)
                if pnode and pnode.meta and pnode.meta.type_name in {NodeType.DIRECTORY, NodeType.CLASS} and pid != self.repo_node.id:
                    logging.debug(f"🧹 Removing empty parent node: {pnode.name}")
                    self._adjacency.pop(pid, None)
                    parent_id = self._parents.pop(pid, None)
                    if pid in self.nodes:
                        del self.nodes[pid]
                        cleaned_parents += 1
                    self.edges = [e for e in self.edges if e.src != pid and e.dst != pid]
                    if parent_id:
                        queue.append(parent_id)

        logging.info(
            f"Deleted {deleted_nodes} nodes, {deleted_edges} edges, "
            f"cleaned {cleaned_parents} empty parent nodes along affected branches."
        )

        return {
            "deleted_nodes": deleted_nodes,
            "deleted_edges": deleted_edges,
            "cleaned_parents": cleaned_parents,
        }
        
    def update_from_parsed_tree(
        self,
        parsed_tree: Dict[str, Dict],
        deleted_units: Optional[Dict[str, List[str]]] = None,
        file2unit: Dict[str, List[CodeUnit]] = {},
        repo_info=""
    ) -> Dict[str, int]:
        """
        Incrementally update RPG graph from a new parsed_tree:
        - Add new functions/methods if not exist.
        - Update existing nodes' name/meta if already exist.
        - Remove specific deleted units (functions or methods).
        - Skip unchanged nodes automatically.

        Handles:
            Empty class → class with methods: remove old empty node, rebuild class+methods
            Class with methods → empty class: remove all method nodes
        """
        added_nodes = 0
        updated_nodes = 0
        added_edges = 0
        deleted_nodes = 0
        deleted_edges = 0
        total_renamed = 0  
        
        updated_file_nodes = set()

        def _remove_node_shallow(nid: str) -> None:
            self.edges = [e for e in self.edges if e.src != nid and e.dst != nid]
            pid = self._parents.pop(nid, None)
            if pid and nid in self._adjacency.get(pid, []):
                self._adjacency[pid].remove(nid)
            self._adjacency.pop(nid, None)
            if nid in self.nodes:
                del self.nodes[nid]

        # === Step 1. Add / update nodes from parsed tree ===
        for rel_path, f_features in parsed_tree.items():
            file_units = file2unit.get(rel_path, [])

            key2unit = {}
            for u in file_units:
                key = (
                    f"{u.unit_type} {u.parent}.{u.name}"
                    if u.unit_type == "method"
                    else f"{u.unit_type} {u.name}"
                )
                key2unit[key] = u

            file_node = None
            for n in self.nodes.values():
                if (
                    n.meta
                    and n.meta.type_name == NodeType.FILE
                    and isinstance(n.meta.path, str)
                    and os.path.normpath(n.meta.path) == rel_path
                ):
                    file_node = n
                    break
            if not file_node:
                continue

            existing_children = {}
            for cid in self.get_children(file_node.id):
                node = self.nodes.get(cid)
                if not node or not node.meta or not node.meta.path:
                    continue
                existing_children[node.meta.path] = cid

            for unit_name, unit_features in f_features.items():
                if unit_name == "_file_summary_":
                    continue

                if unit_name.startswith("function "):
                    func_name = unit_name.replace("function ", "").strip()
                    func_unit = key2unit.get(f"function {func_name}")
                    func_path = f"{rel_path}:{func_name}"

                    for feat in unit_features:
                        feat_name = feat.strip()
                        existing_id = existing_children.get(func_path)
                        if existing_id:
                            func_node = self.nodes[existing_id]
                            if func_node.name != feat_name:
                                old_name = func_node.name
                                func_node.name = feat_name
                                updated_nodes += 1
                                updated_file_nodes.add(file_node.id)
                                logging.debug(
                                    f"Function renamed: {old_name} → {feat_name}"
                                )
                        else:
                            uid = self._rand8()
                            func_node = Node(
                                id=f"{feat_name}_{uid}",
                                name=feat_name,
                                meta=NodeMetaData(
                                    type_name=NodeType.FUNCTION,
                                    path=func_path,
                                ),
                                unit=func_unit.key() if func_unit else None,
                            )
                            self.add_node(func_node)
                            self.add_edge(file_node, func_node)
                            added_nodes += 1
                            added_edges += 1
                            updated_file_nodes.add(file_node.id)
                            logging.debug(f"Added function node: {feat_name}")

                elif unit_name.startswith("class "):
                    class_name = unit_name.replace("class ", "").strip()
                    cls_unit = key2unit.get(f"class {class_name}")
                    class_path = f"{rel_path}:{class_name}"

                    existing_id = next(
                        (
                            cid
                            for cid, node in self.nodes.items()
                            if node.meta and node.meta.path == class_path
                        ),
                        None,
                    )
                    cls_node = self.nodes.get(existing_id) if existing_id else None

                    # ---------- Case A: class with methods ----------
                    if isinstance(unit_features, dict):
                       
                        if cls_node:
                            has_methods = any(
                                self.nodes[cid].meta.type_name == NodeType.METHOD
                                for cid in self.get_children(cls_node.id)
                            )

                            if not has_methods:
                                _remove_node_shallow(cls_node.id)
                                deleted_nodes += 1
                                logging.debug(
                                    f"Removed empty class node '{class_name}' before rebuilding methods."
                                )
                                if class_path in existing_children:
                                    existing_children.pop(class_path, None)
                                cls_node = None

                        for m_name, m_feats in unit_features.items():
                            method_unit = key2unit.get(f"method {class_name}.{m_name}")
                            method_path = f"{rel_path}:{class_name}.{m_name}"
                            for feat in m_feats:
                                feat_name = feat.strip()
                                existing_id = existing_children.get(method_path)
                                if existing_id:
                                    node = self.nodes[existing_id]
                                    if node.name != feat_name:
                                        old = node.name
                                        node.name = feat_name
                                        updated_nodes += 1
                                        updated_file_nodes.add(file_node.id)
                                        logging.debug(
                                            f"Method renamed: {old} → {feat_name}"
                                        )
                                else:
                                    uid = self._rand8()
                                    method_node = Node(
                                        id=f"{feat_name}_{uid}",
                                        name=feat_name,
                                        meta=NodeMetaData(
                                            type_name=NodeType.METHOD,
                                            path=method_path,
                                        ),
                                        unit=(method_unit.key() if method_unit else None),
                                    )
                                    self.add_node(method_node)
                                    self.add_edge(file_node, method_node)
                                    added_nodes += 1
                                    added_edges += 1
                                    updated_file_nodes.add(file_node.id)
                                    existing_children[method_path] = method_node.id
                                    logging.debug(f"Added method node: {feat_name}")

                        updated_file_nodes.add(file_node.id)

                    # ---------- Case B: class without methods ----------
                    elif isinstance(unit_features, list):
                        if cls_node:
                            child_ids = [
                                cid
                                for cid in self.get_children(cls_node.id)
                                if self.nodes[cid].meta
                                and self.nodes[cid].meta.type_name
                                == NodeType.METHOD
                            ]
                            for cid in child_ids:
                                _remove_node_shallow(cid)
                                deleted_nodes += 1
                            logging.debug(
                                f"Removed {len(child_ids)} method nodes from class '{class_name}' (now empty)."
                            )

                        for feat in unit_features:
                            feat_name = feat.strip()
                            if not cls_node:
                                uid = self._rand8()
                                cls_node = Node(
                                    id=f"{feat_name}_{uid}",
                                    name=feat_name,
                                    meta=NodeMetaData(
                                        type_name=NodeType.CLASS,
                                        path=class_path,
                                    ),
                                    unit=cls_unit.key() if cls_unit else None,
                                )
                                self.add_node(cls_node)
                                self.add_edge(file_node, cls_node)
                                added_nodes += 1
                                added_edges += 1
                                logging.debug(
                                    f"Added empty class node: {feat_name}"
                                )
                            elif cls_node.name != feat_name:
                                old = cls_node.name
                                cls_node.name = feat_name
                                updated_nodes += 1
                                logging.debug(
                                    f"Class name updated: {old} → {feat_name}"
                                )
                        updated_file_nodes.add(file_node.id)

        # === Step 2. Clean up deleted units ===
        if deleted_units:
            for rel_path, names in deleted_units.items():
                for name in names:
                    full_path = f"{rel_path}:{name}"
                    to_delete = [
                        nid
                        for nid, node in self.nodes.items()
                        if node.meta and node.meta.path == full_path
                    ]
                    for nid in to_delete:
                        _remove_node_shallow(nid)
                        deleted_nodes += 1
                        logging.debug(f"Deleted node {nid} ({full_path})")

        # === Step 3. Remove empty parent nodes ===
        cleaned = True
        while cleaned:
            cleaned = False
            empty = []
            for pid, children in list(self._adjacency.items()):
                self._adjacency[pid] = [c for c in children if c in self.nodes]
                if not self._adjacency[pid]:
                    node = self.nodes.get(pid)
                    if (
                        node
                        and node.meta
                        and node.meta.type_name
                        in {NodeType.CLASS, NodeType.FILE, NodeType.DIRECTORY}
                        and pid != self.repo_node.id
                    ):
                        empty.append(pid)
            for pid in empty:
                _remove_node_shallow(pid)
                deleted_nodes += 1
                cleaned = True

        logging.info(
            f"RPG updated: +{added_nodes} nodes, Δ{updated_nodes} updated, "
            f"-{deleted_nodes} deleted, +{added_edges} edges, {total_renamed} renamed (semantic rename disabled)."
        )

        return {
            "added_nodes": added_nodes,
            "updated_nodes": updated_nodes,
            "added_edges": added_edges,
            "deleted_nodes": deleted_nodes,
            "deleted_edges": deleted_edges,
            "renamed_upward": total_renamed,
        }

    def visualize_dir_map(
        self,
        start=None,                  
        json_format: bool = False,  
        max_depth: int | None = 3,   
        include_orphans: bool = False,
        use_tree_markers: bool = True, 
        feature_only: bool = True, 
    ) -> str:
        """
        Visualize the entire RPG (starting from L1), including all nodes with meta such as DIRECTORY/FILE/CLASS/FUNCTION/METHOD, etc.
        Always returns a string:
        - json_format=True  -> returns a JSON array string (one object per subtree)
        - json_format=False -> returns a text tree (Node [paths]); multiple subtrees are separated by blank lines

        Notes:
        - Level 0 (repo) is never printed; by default, visualization starts from its L1 children.
        - max_depth is the depth limit measured from the start node (start depth = 0); None means traverse to the bottom.
        - use_tree_markers=True uses '├─/└─/│'; if False, uses indentation only.
        - If feature_only=True, show only functional-relation nodes; nodes are not required to have meta (useful for a pure feature view).
        """
        import json

        if max_depth is not None:
            max_depth = max_depth - 1
            max_depth = min(MAX_LEVEL, max(0, max_depth))

        type_order = {
            getattr(NodeType, "DIRECTORY", None): 0,
            getattr(NodeType, "FILE", None): 1,
            getattr(NodeType, "CLASS", None): 2,
            getattr(NodeType, "FUNCTION", None): 3,
            getattr(NodeType, "METHOD", None): 4,
        }

        def _node_type(n):
            return getattr(n.meta, "type_name", None) if getattr(n, "meta", None) else None

        def _type_rank(n):
            t = _node_type(n)
            return type_order.get(t, 99)

        def _paths_list(n):
            p = getattr(n.meta, "path", None) if getattr(n, "meta", None) else None
            if isinstance(p, list):
                return [str(x) for x in p if x not in (None, "")]
            if isinstance(p, str):
                return [] if p.strip() == "" else [p]
            return []

        def _paths_str(n):
            ps = _paths_list(n)
            if not ps:
                return "."
            if len(ps) == 1:
                return ps[0]
            ps = sorted(ps, key=lambda x: (len(str(x).split("/")), str(x)))
            return ", ".join(ps)

        def _is_connected_to_repo(nid: str) -> bool:
            if nid == self.repo_node.id:
                return True
            current_id = nid
            while current_id:
                parent_id = self._parents.get(current_id)
                if parent_id == self.repo_node.id:
                    return True
                if parent_id is None: 
                    break
                current_id = parent_id
            return False

        if start is None:
            root_id = self.repo_node.id if getattr(self, "repo_node", None) else None
            start_ids = []
            if root_id and root_id in self.nodes:
                start_ids = [
                    cid for cid in self._adjacency.get(root_id, [])
                    if cid in self.nodes and _is_connected_to_repo(cid)
                ]
        else:
            sid = start.id if isinstance(start, Node) else start
            if sid == getattr(self.repo_node, "id", None):
                start_ids = [
                    cid for cid in self._adjacency.get(sid, [])
                    if cid in self.nodes and _is_connected_to_repo(cid)
                ]
            else:
                start_ids = [sid] if (sid in self.nodes and _is_connected_to_repo(sid)) else []
                
        def _children(pid: str):
            cids = [cid for cid in self._adjacency.get(pid, []) if cid in self.nodes]
            if not feature_only:
                cids = [cid for cid in cids if getattr(self.nodes[cid], "meta", None) is not None]
            return sorted(
                cids,
                key=lambda x: (
                    (self.nodes[x].level or 0),
                    _type_rank(self.nodes[x]),
                    self.nodes[x].name or ""
                ),
            )

        def _build_json(nid: str, depth: int = 0):
            if nid not in self.nodes:
                return None
            n = self.nodes[nid]
            if not feature_only and getattr(n, "meta", None) is None:
                return None
            if max_depth is not None and depth > max_depth:
                return None

            children = []
            if max_depth is None or depth < max_depth:
                for cid in _children(nid):
                    j = _build_json(cid, depth + 1)
                    if j:
                        children.append(j)

            if feature_only:
                return {
                    "id": n.id,
                    "name": n.name,
                    "level": n.level,
                    "node_type": n.node_type or "unknown",
                    "children": children,
                }
            else:
                ntype = _node_type(n)
                ntype_val = ntype.value if hasattr(ntype, "value") else (n.node_type or "unknown")
                return {
                    "id": n.id,
                    "name": n.name,
                    "level": n.level,
                    "type": ntype_val,
                    "paths": _paths_list(n) or ["."],
                    "children": children,
                }

        def _render_text(nid: str, prefix: str = "", is_last: bool = True, depth: int = 0, out_lines=None):
            if out_lines is None:
                out_lines = []
            n = self.nodes.get(nid)
            if not n:
                return out_lines
        
            if not feature_only and getattr(n, "meta", None) is None:
                return out_lines
            if max_depth is not None and depth > max_depth:
                return out_lines

            if use_tree_markers:
                connector = "" if depth == 0 else ("└─ " if is_last else "├─ ")
                if feature_only:
                    line = f"{prefix}{connector}{n.name}"
                else:
                    line = f"{prefix}{connector}{n.name} [{_paths_str(n)}]"
            else:
                # 仅缩进
                if feature_only:
                    line = f"{prefix}{n.name}"
                else:
                    line = f"{prefix}{n.name} [{_paths_str(n)}]"
            out_lines.append(line)

            if max_depth is None or depth < max_depth:
                kids = _children(nid)
                for i, cid in enumerate(kids):
                    last = (i == len(kids) - 1)
                    if use_tree_markers:
                        next_prefix = "" if depth == 0 else (prefix + ("   " if is_last else "│  "))
                    else:
                        next_prefix = prefix + "  "
                    _render_text(cid, next_prefix, last, depth + 1, out_lines)
            return out_lines

        results = [] 
        seen = set()

        def _emit_from(sid: str):
            if json_format:
                tree = _build_json(sid, 0)
                if tree:
                    results.append(tree)
            else:
                lines = _render_text(sid, prefix="", is_last=True, depth=0)
                if lines:
                    results.append("\n".join(lines))

            def _collect(nid: str):
                if nid in seen:
                    return
                n = self.nodes.get(nid)
                if not n or getattr(n, "meta", None) is None:
                    return
                seen.add(nid)
                for cid in _children(nid):
                    _collect(cid)
            _collect(sid)

        for sid in start_ids:
            if getattr(self.nodes[sid], "meta", None) is None:
                continue
            _emit_from(sid)

        if include_orphans:
            root_id = self.repo_node.id if getattr(self, "repo_node", None) else None
            forest_roots = []
            for nid, n in self.nodes.items():
                if getattr(n, "meta", None) is None:
                    continue
                if nid == root_id:
                    continue
                if nid in seen:
                    continue
                parent_id = self._parents.get(nid)
                parent_is_visible = (parent_id in self.nodes) and (getattr(self.nodes[parent_id], "meta", None) is not None) if parent_id else False
                if not parent_is_visible:
                    forest_roots.append(nid)
            for rid in sorted(forest_roots, key=lambda x: (self.nodes[x].level or 0, _type_rank(self.nodes[x]), self.nodes[x].name or "")):
                _emit_from(rid)

        if json_format:
            return json.dumps(results, ensure_ascii=False, indent=2)
        else:
            return "\n\n".join(results)
    
    def recalculate_levels_topdown(self) -> None:
        """
        [Nuclear Option - Final Fix]
        1. Resets all levels.
        2. Rebuilds self._adjacency entirely based on self._parents.
        3. Rebuilds self.edges entirely based on self._parents.
        4. Identifies orphans and attaches them to root.
        5. Performs BFS to assign levels and types correctly.
        """
        if not hasattr(self, "repo_node") or not self.repo_node:
            raise RuntimeError("RPG has no repo_node defined.")

        root_id = self.repo_node.id
        
        # 1. Reset Levels
        for n in self.nodes.values():
            n.level = None
        self.repo_node.level = 0
        self.repo_node.node_type = "repo"

        # 2. Rebuild Adjacency & Edges from _parents (Source of Truth)
        new_adjacency = defaultdict(list)
        new_hierarchy_edges = []
        
        non_hierarchy_edges = [
            e for e in self.edges
            if str(e.relation).lower() not in ("composes", "contains")
        ]

        valid_children = set()
        cleaned_parents = {}

        for child_id, parent_id in list(self._parents.items()):
            if child_id not in self.nodes: continue
            if parent_id not in self.nodes: continue

            cleaned_parents[child_id] = parent_id
            new_adjacency[parent_id].append(child_id)
            valid_children.add(child_id)

            new_hierarchy_edges.append(Edge(src=parent_id, dst=child_id, relation=EdgeType.COMPOSES))

        self._parents = cleaned_parents
        self._adjacency = new_adjacency
        self.edges = new_hierarchy_edges + non_hierarchy_edges

        # 3. Handle Orphans (Nodes existing but not in valid_children)
        orphans = [nid for nid in self.nodes if nid != root_id and nid not in valid_children]
        
        if orphans:
            logging.info(f"🕸️ Re-attaching {len(orphans)} orphans to root.")
            for nid in orphans:
                self._parents[nid] = root_id
                self._adjacency[root_id].append(nid)
                self.edges.append(Edge(src=root_id, dst=nid, relation=EdgeType.COMPOSES))
                # Orphan files default to Level 1 (Functional Area equivalent)
                if self.nodes[nid].level is None:
                    self.nodes[nid].level = 1

        # 4. BFS for Levels
        queue = deque([root_id])
        visited = {root_id}

        while queue:
            pid = queue.popleft()
            pnode = self.nodes[pid]
            curr_level = pnode.level if pnode.level is not None else 0
            
            children = self._adjacency.get(pid, [])
            children.sort(key=lambda cid: (
                0 if getattr(self.nodes[cid].meta, 'type_name', '') == NodeType.DIRECTORY else 1,
                self.nodes[cid].name
            ))

            for cid in children:
                if cid in visited: continue
                
                cnode = self.nodes[cid]
                cnode.level = min(curr_level + 1, self.MAX_LEVEL)
                cnode.node_type = self._infer_node_type(cnode.level)
                
                visited.add(cid)
                queue.append(cid)
                
        logging.info("✅ Recalculated levels top-down (Topology & Edges Rebuilt).")
    
    def _iter_bottom_up_ids(self):
        visited = set()
        order = []

        def postorder(start_id):
            stack = [(start_id, False)]
            while stack:
                nid, expanded = stack.pop()

                if nid in visited and expanded:
                    continue
                
                if nid not in self.nodes:
                    visited.add(nid)
                    continue

                if not expanded:
                    stack.append((nid, True))

                    for cid in self._adjacency.get(nid, []):
                        if cid in self.nodes and cid not in visited:
                            stack.append((cid, False))
                else:
                    visited.add(nid)
                    order.append(nid)

        if hasattr(self, "repo_node") and self.repo_node and self.repo_node.id in self.nodes:
            postorder(self.repo_node.id)

        for nid in list(self.nodes.keys()):
            if nid not in visited:
                postorder(nid)
        return order 

    def update_all_metadata_bottom_up(self) -> int:
        """
        Update meta.path for all non-code, non-file nodes bottom-up.
        Each path is recomputed as the minimal common directory (LCA) derived from the directory paths of all FILE-type descendants.

        - FILE/FUNCTION/METHOD/CLASS node paths remain unchanged (to avoid breaking later path-based matching).
        - Aggregation nodes such as DIRECTORY have their paths recomputed.
        """
        
        repo_id = self.repo_node.id if getattr(self, "repo_node", None) else None

        def _is_file(n: Node) -> bool:
            return n.meta and n.meta.type_name == NodeType.FILE

        def _is_code(n: Node) -> bool:
            return (
                n.level == self.MAX_LEVEL
                or (
                    n.meta
                    and n.meta.type_name in {
                        NodeType.FUNCTION,
                        NodeType.METHOD,
                        NodeType.CLASS,
                    }
                )
            )

        def _norm_dir(p: str) -> str:
            d = os.path.normpath(p).replace("\\", "/")
            return "." if d in ("", ".") else d

        def _dir_of_file(fp: str) -> str:
            return _norm_dir(os.path.dirname(fp or ""))

        class _Trie:
            __slots__ = ("children", "terminal")

            def __init__(self):
                self.children: dict[str, "_Trie"] = {}
                self.terminal = False

        def _split(d: str) -> list[str]:
            return [seg for seg in d.split("/") if seg]

        def _insert(root: _Trie, d: str):
            cur = root
            for seg in _split(d):
                cur = cur.children.setdefault(seg, _Trie())
            cur.terminal = True

        def _compress_trie(root: _Trie, prefix: list[str]) -> set[str]:
            branches: list[set[str]] = []
            for seg, child in root.children.items():
                sub = _compress_trie(child, prefix + [seg])
                if sub:
                    branches.append(sub)

            p = "/".join(prefix) if prefix else "."

            if root.terminal:
                if p and p != ".":
                    branches.append({p})

            if len(branches) >= 2 and prefix:
                cur = "/".join(prefix)
                return {cur}

            return set().union(*branches) if branches else set()

        def _dir_lca_merge(dirset: set[str]) -> set[str]:
            if not dirset:
                return set()
            root = _Trie()
            for d in dirset:
                if d and d != ".":
                    _insert(root, d)
            return _compress_trie(root, [])

 
        order = self._iter_bottom_up_ids()
        cover: dict[str, set[str]] = {}
        updated = 0

        for nid in order:
            if nid not in self.nodes:
                continue

            node = self.nodes[nid]

            is_file = _is_file(node)
            is_code_unit = _is_code(node)
            is_repo = (repo_id and nid == repo_id)

            my_cover: set[str] = set()

            if is_file and isinstance(node.meta.path, str):
                my_cover = {node.meta.path}
            elif not is_code_unit:
                for cid in self._adjacency.get(nid, []):
                    my_cover |= cover.get(cid, set())

            cover[nid] = my_cover

            if is_file or is_code_unit:
                continue
            
            if is_repo:
                new_path = "."
            else:
                if not my_cover:
                    continue

                dir_set = {_dir_of_file(p) for p in my_cover if p}
                dir_set.discard(".")

                if not dir_set:
                    continue

                lca_set = _dir_lca_merge(dir_set)

                if not lca_set:
                    continue

                new_path_list = sorted(
                    lca_set, key=lambda x: (len(x.split("/")), x)
                )
                new_path = (
                    new_path_list[0] if len(new_path_list) == 1 else new_path_list
                )

            if node.meta is None:
                node.meta = NodeMetaData(
                    type_name=(
                        NodeType(node.node_type)
                        if node.node_type in {m.value for m in NodeType}
                        else None
                    ),
                    path=new_path,
                )
                updated += 1
            else:
                if node.meta.path != new_path:
                    node.meta.path = new_path
                    updated += 1

        return updated
    
    
    def delete_root_level_file_subtrees(self) -> Dict[str, int]:
        """
        Special cleanup:
        Recursively delete nodes whose meta.type_name == FILE and whose parent is repo_node (L0),
        along with all descendants and related edges.

        Returns stats: deleted_nodes, deleted_edges
        """
        if not getattr(self, "repo_node", None):
            return {"deleted_nodes": 0, "deleted_edges": 0}

        root_id = self.repo_node.id

        # 1) Find FILE nodes directly connected to the repo_node
        root_children = [cid for cid in self._adjacency.get(root_id, []) if cid in self.nodes]
        target_file_ids = []
        for cid in root_children:
            n = self.nodes.get(cid)
            if n and n.meta and n.meta.type_name == NodeType.FILE:
                target_file_ids.append(cid)

        if not target_file_ids:
            return {"deleted_nodes": 0, "deleted_edges": 0}

        # 2) Collect all nodes to delete (including descendants)
        all_delete_ids = set(target_file_ids)
        for fid in target_file_ids:
            for desc in self.get_children(fid, recursive=True):
                all_delete_ids.add(desc)

        # 3) Remove edges (from the edges list)
        before_edges = len(self.edges)
        self.edges = [
            e for e in self.edges
            if e.src not in all_delete_ids and e.dst not in all_delete_ids
        ]
        deleted_edges = before_edges - len(self.edges)

        # 4) Remove nodes + clean up parents/adjacency
        for nid in all_delete_ids:
            pid = self._parents.pop(nid, None)
            if pid and nid in self._adjacency.get(pid, []):
                try:
                    self._adjacency[pid].remove(nid)
                except ValueError:
                    pass

            self._adjacency.pop(nid, None)

            if nid in self.nodes:
                del self.nodes[nid]

        # 5) The root's adjacency may still contain ghost children; filter them out as well
        self._adjacency[root_id] = [c for c in self._adjacency.get(root_id, []) if c in self.nodes]

        return {"deleted_nodes": len(all_delete_ids), "deleted_edges": deleted_edges}
    
    def update_result_to_rpg(self, area_update):
        """
        Structure of area_update:

        {
            "DataProcessingAndTransformation": {
                "DataProcessingAndTransformation/tables_and_arrays/operations/manage tabular column data": <Node(FILE)>,
                "DataProcessingAndTransformation/tables_and_arrays/operations/array manipulation utilities": <Node(FILE)>,
                ...
            },
            "CoordinateSystemsAndTransformations": {
                "CoordinateSystemsAndTransformations/angle_and_value_operations/...": <Node(FILE)>,
                ...
            },
            ...
        }

        Conventions:
        - The functional area node is level 1 and is attached directly under repo_node;
        - Each intermediate segment (e.g., tables_and_arrays / operations) is created as a DIRECTORY node;
        - The final segment (the leaf name) corresponds to the input file node, which already exists and should not be created again;
        - Directory reuse rule: under the same parent, reuse a DIRECTORY node if it has the same name;
        different functional areas have independent roots, so directories are naturally not shared.
        """
        import uuid

        def _uuid():
            return uuid.uuid4().hex[:8]

        def _edge_exists(src_id: str, dst_id: str) -> bool:
            return any(e.src == src_id and e.dst == dst_id for e in self.edges)

        def _ensure_functional_area_node(name: str):
            root_id = self.repo_node.id

            for cid in self._adjacency.get(root_id, []):
                n = self.nodes.get(cid)
                if (
                    n
                    and n.name == name
                    and n.meta
                    and n.meta.type_name == NodeType.DIRECTORY
                ):
                    return n

            for n in self.nodes.values():
                if (
                    n.name == name
                    and n.meta
                    and n.meta.type_name == NodeType.DIRECTORY
                    and n.level == 1
                ):
                    if self._parents.get(n.id) != root_id:
                        self.add_edge(self.repo_node, n)
                    return n

            fa_node = Node(
                id=f"{name}_{_uuid()}",
                name=name,
                meta=NodeMetaData(
                    type_name=NodeType.DIRECTORY,
                    path=".", 
                ),
            )
            self.add_node(fa_node)
            self.add_edge(self.repo_node, fa_node)
            return fa_node

        def _find_or_create_dir(parent_node, name: str):

            for cid in self._adjacency.get(parent_node.id, []):
                n = self.nodes.get(cid)
                if (
                    n
                    and n.name == name
                    and n.meta
                    and n.meta.type_name == NodeType.DIRECTORY
                ):
                    return n

           
            dir_node = Node(
                id=f"{name}_{_uuid()}",
                name=name,
                meta=NodeMetaData(
                    type_name=NodeType.DIRECTORY,
                    path=".",
                ),
            )
            self.add_node(dir_node)
            self.add_edge(parent_node, dir_node)
            return dir_node

      
        for fa_name, path2node in area_update.items():
            fa_node = _ensure_functional_area_node(fa_name)

            for func_path, file_node in path2node.items():
                if file_node.id not in self.nodes:
                    self.add_node(file_node)

                parts = [p for p in func_path.split("/") if p]

                if not parts:
                    parent = fa_node
                else:
                    if parts[0] == fa_name:
                        parts = parts[1:]

                    if len(parts) == 0:
                        parent = fa_node
                    else:
                        middle_parts = parts[:-1] 
                        parent = fa_node
                        for seg in middle_parts:
                            parent = _find_or_create_dir(parent, seg)

                if not _edge_exists(parent.id, file_node.id):
                    self.add_edge(parent, file_node)
                    
        self.recalculate_levels_topdown()       
        self.delete_root_level_file_subtrees()

        self.recalculate_levels_topdown()
        self.update_all_metadata_bottom_up()

    # ============================================================
    #  DependencyGraph
    # ============================================================

    def set_dep_graph(self, dep_graph: DependencyGraph) -> None:
        """
        Set the DependencyGraph and establish the node mapping relationships.

        Args:
            dep_graph: DependencyGraph instance
        """
        self.dep_graph = dep_graph
        self._dep_to_rpg_map = self._build_dep_to_rpg_map()
        logging.info(
            f"DependencyGraph set with {len(self._dep_to_rpg_map)} node mappings"
        )

    def parse_dep_graph(self, repo_dir: str) -> DependencyGraph:
        dep_graph = DependencyGraph(repo_dir)
        dep_graph.build()
        dep_graph.parse()
        self.set_dep_graph(dep_graph)
        return dep_graph

    def _build_dep_to_rpg_map(self) -> Dict[str, List[str]]:
        """
        Establish the mapping from DependencyGraph nodes to RPG nodes.

        Mapping rules:
        - Match by NodeType
        - Match by path (supports rpg_node.meta.path being either a list or a string)

        Returns:
            Dict[str, List[str]]: dep_node_id -> [rpg_node_id, ...]
        """
        if self.dep_graph is None:
            return {}

        dep2rpg_map: Dict[str, List[str]] = defaultdict(list)

        for nid in self.dep_graph.G.nodes():
            dep_node = self.dep_graph.G.nodes[nid]
            dep_node_type: NodeType = dep_node.get("type")

            if dep_node_type is None:
                continue

            for node_id, rpg_node in self.nodes.items():
                rpg_node_meta: NodeMetaData = rpg_node.meta
                if rpg_node_meta is None:
                    continue

                rpg_node_type: NodeType = rpg_node_meta.type_name
                if rpg_node_type != dep_node_type:
                    continue

                rpg_node_paths: Union[List[str], str] = rpg_node_meta.path
                if rpg_node_paths is None:
                    continue

                rpg_node_paths = (
                    rpg_node_paths
                    if isinstance(rpg_node_paths, list)
                    else [rpg_node_paths]
                )

                for rpg_node_path in rpg_node_paths:
                    if rpg_node_path == nid:
                        dep2rpg_map[nid].append(node_id)

        return dict(dep2rpg_map)

    def get_rpg_nodes_for_dep_node(self, dep_node_id: str) -> List[Node]:
        rpg_node_ids = self._dep_to_rpg_map.get(dep_node_id, [])
        return [self.nodes[nid] for nid in rpg_node_ids if nid in self.nodes]

    def get_dep_node_for_rpg_node(self, rpg_node_id: str) -> Optional[str]:
        for dep_id, rpg_ids in self._dep_to_rpg_map.items():
            if rpg_node_id in rpg_ids:
                return dep_id
        return None

    def get_dep_graph_info(self, dep_node_id: str) -> Optional[Dict[str, Any]]:
        if self.dep_graph is None or dep_node_id not in self.dep_graph.G.nodes:
            return None

        node_data = dict(self.dep_graph.G.nodes[dep_node_id])
        node_data.pop("ast", None)
        return node_data

    def get_dep_edges_for_rpg(self) -> List[Dict[str, Any]]:
        """
        Generate edges between RPG nodes based on DependencyGraph edges.

        For each edge in the dep graph, if both source and destination nodes
        have corresponding RPG nodes, create edge records showing the relationship.

        Returns:
            List of edge dicts in the same format as Edge.to_dict():
            {
                "src": rpg_node_id,
                "dst": rpg_node_id,
                "relation": "invokes" | "imports" | "inherits" | "contains",
                "meta": {
                    "src_dep": dep_node_id,
                    "dst_dep": dep_node_id,
                }
            }
        """
        if self.dep_graph is None or not self._dep_to_rpg_map:
            return []

        dep_edges = []
        seen_edges = set()  # Avoid duplicate edges

        for src_dep, dst_dep, edge_data in self.dep_graph.G.edges(data=True):
            edge_type = edge_data.get("type")
            if edge_type is None:
                continue

            # Get edge type as string
            edge_type_str = edge_type.value if hasattr(edge_type, 'value') else str(edge_type)

            # Get RPG nodes for src and dst
            src_rpg_nodes = self._dep_to_rpg_map.get(src_dep, [])
            dst_rpg_nodes = self._dep_to_rpg_map.get(dst_dep, [])

            # Create edges for all combinations of mapped RPG nodes
            for src_rpg in src_rpg_nodes:
                for dst_rpg in dst_rpg_nodes:
                    # Skip self-loops
                    if src_rpg == dst_rpg:
                        continue

                    # Deduplicate edges with same src, dst, relation
                    edge_key = (src_rpg, dst_rpg, edge_type_str)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    dep_edges.append({
                        "src": src_rpg,
                        "dst": dst_rpg,
                        "relation": edge_type_str,
                        "meta": {
                            "src_dep": src_dep,
                            "dst_dep": dst_dep,
                        }
                    })

        return dep_edges