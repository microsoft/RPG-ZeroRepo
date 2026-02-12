import ast
import json
import logging
import os
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Tuple, Any
import networkx as nx
from .util import ( 
    EdgeType, NodeType, 
    normalize_path, 
    is_test_file,
    get_node_range_robust, 
    extract_source_by_lines
)


def _exclude_irrelevant_for_build(file_id: str) -> bool:
    EXT_BLACKLIST = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".mp3", ".mp4", ".zip", ".tar", ".gz",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".exe", ".dll", ".so", ".o", ".a",
        ".log"
    }

    # 目录名
    PATH_BLACKLIST = {
        ".git", "__pycache__", "node_modules",
        ".venv", "venv", ".idea", ".vscode",
        ".pytest_cache", ".mypy_cache", "build", "dist"
    }

    # 文件名
    FILE_BLACKLIST = {
        "Makefile", "CMakeLists.txt",
        "Dockerfile", "LICENSE", "LICENSE.txt",
        "COPYING", "requirements.txt", "environment.yml",
        "pyproject.toml"
    }

    path_obj = PurePosixPath(file_id)

    if path_obj.suffix.lower() in EXT_BLACKLIST:
        return False

    if any(part in PATH_BLACKLIST for part in path_obj.parts):
        return False

    if path_obj.name in FILE_BLACKLIST:
        return False

    if path_obj.name.startswith('.'):
        return False
    
    if is_test_file(file_id):
        return False

    return True


def _exclude_irrelevant_for_parse(file_id: str) -> bool:
   
    if not file_id.endswith(".py"):
        return False

    path_lower = file_id.lower()
    if is_test_file(path_lower):
        return False

    EXCLUDE_FILES = {
        "setup.py", "__main__.py",
        "conftest.py", "requirements.py"
    }
    
    
    if any(path_lower.endswith(f"/{f}") for f in EXCLUDE_FILES):
        return False

    base_name = os.path.basename(file_id)
    if base_name.startswith("test_") or base_name.endswith("_test.py"):
        return False

    return True

def path_to_module(node_id: str) -> str:
 
    s = str(node_id).strip()
    if ":" in s:
        s = s.split(":", 1)[0]

    s = s.removeprefix("./")

    if s == ".":
        return ""

    path = PurePosixPath(s)
    if path.suffix == ".py":
        if path.stem == "__init__":
            parent = path.parent.as_posix()
            return parent.replace("/", ".") if parent != "" else ""
        else:
            mod = path.with_suffix("").as_posix()
            return mod.replace("/", ".")
    else:

        return path.as_posix().replace("/", ".")




class DependencyGraph:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.G = nx.MultiDiGraph()

        self.G_tree = nx.subgraph_view(self.G, filter_edge=lambda u,v,k: self.G.edges[u,v,k].get("type")==EdgeType.CONTAINS)
        self.G_imports = nx.subgraph_view(self.G, filter_edge=lambda u,v,k: self.G.edges[u,v,k].get("type")==EdgeType.IMPORTS)
        self.G_invokes = nx.subgraph_view(self.G, filter_edge=lambda u,v,k: self.G.edges[u,v,k].get("type")==EdgeType.INVOKES)
        self.G_inherits = nx.subgraph_view(self.G, filter_edge=lambda u,v,k: self.G.edges[u,v,k].get("type")==EdgeType.INHERITS)

        self.G_code = nx.subgraph_view(self.G, filter_node=lambda n: self.G.nodes[n].get("ast") is not None)

    def _add_node(self, node_id: str, type: str, name: Optional[str] = None, parent_id: Optional[str] = None, **kwargs) -> None:
        nid = normalize_path(node_id)
        if type not in NodeType:
            logging.warning(f"Unknown node type: {type} at {nid}")
        if not name:
            name = self.get_name(nid, ntype=type, with_badge=False)

        if nid in self.G:
            old_type = self.G.nodes[nid].get("type")
            if old_type != type:
                logging.warning(f"Node type conflict at node_id: {nid}. Existing: {old_type}, New: {type}")

        self.G.add_node(nid, type=type, module=path_to_module(nid), name=name, **kwargs)

        if parent_id is None:
            parent_exists, parent_id = self.get_parent(nid)
        else:
            parent_exists = parent_id in self.G
        if not parent_exists and parent_id != None:
            if type in [NodeType.DIRECTORY, NodeType.FILE]:
                parent_type = NodeType.DIRECTORY
            elif type in [NodeType.CLASS, NodeType.FUNCTION]:
                parent_type = NodeType.FILE
            else:
                parent_type = NodeType.CLASS
            self._add_node(parent_id, parent_type)
        if parent_id != None:
            self._add_edge(parent_id, nid, type=EdgeType.CONTAINS)

    def _ensure_node(self, node_id: str, type: str) -> None:
        nid = normalize_path(node_id)
        if nid not in self.G:
            self._add_node(nid, type=type)

    def _add_edge(self, src: str, dst: str, type: str, **kwargs) -> bool:
        u = normalize_path(src)
        v = normalize_path(dst)

        if u not in self.G or v not in self.G:
            logging.error(f"[RepoSkeleton.add_edge] Missing node(s): '{u}' or '{v}' not in graph")
            return False

        edge_data = self.G.get_edge_data(u, v, default={})
        for _key, data in edge_data.items():
            if data.get("type") == type:
                return False

        self.G.add_edge(u, v, type=type, **kwargs)
        return True


    def build(self, filter_func: Callable[[str], bool] = _exclude_irrelevant_for_build) -> None:
        logging.info(f"Building RepoSkeleton for repo: {self.repo_dir}")

        repo_root = Path(self.repo_dir)
        if not repo_root.exists():
            raise FileNotFoundError(f"Repo root not found: {self.repo_dir}")
        self._add_node(".", type=NodeType.DIRECTORY, code=None)

        for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True, followlinks=False):
            dir_path = Path(dirpath)
            dir_rel = normalize_path(dir_path.relative_to(repo_root))
            if not filter_func(dir_rel):
                dirnames[:] = []
                continue

            self._ensure_node(dir_rel, type=NodeType.DIRECTORY)

            for dname in dirnames:
                subdir_path = dir_path / dname
                subdir_rel = normalize_path(subdir_path.relative_to(repo_root))
                if not filter_func(subdir_rel):
                    continue
                self._add_node(subdir_rel, type=NodeType.DIRECTORY, parent_id=dir_rel, code="")

            for fname in filenames:
                file_path = dir_path / fname
                file_rel = normalize_path(str(file_path.relative_to(repo_root)))
                if not filter_func(file_rel):
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    logging.debug(f"[skip] Cannot read {file_path}: {e}")
                    continue
                self._add_node(file_rel, type=NodeType.FILE, code=content, parent_id=dir_rel)

        logging.info("Finished building RepoSkeleton, now has %d nodes and %d edges", self.G.number_of_nodes(), self.G.number_of_edges())


    def parse(self, filter_func: Callable[[str], bool] = _exclude_irrelevant_for_parse) -> None:
        # 1) parse files to code
        logging.info("Parsing RepoSkeleton to extract code structure")
        for file_id, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE or not filter_func(file_id):
                continue

            content = attrs.get("code") or ""
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                logging.debug(f"[parse:skip] {file_id}: {e}")
                continue

            self.G.nodes[file_id]["ast"] = tree
            self._parse_file(file_id, tree, content)
        logging.info("Finished parsing code structure, now has %d nodes and %d edges", self.G.number_of_nodes(), self.G.number_of_edges())
        
        # 2) parse imports
        logging.info("Parsing RepoSkeleton to extract imports")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._init_alias_map(node_id)

        alias_links = nx.DiGraph()
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._parse_imports(node_id, attrs["ast"], alias_links)
        logging.info("Finished parsing imports, added %d edges", self.G_imports.number_of_edges())

        # 3) parse inherits
        logging.info("Parsing RepoSkeleton to extract inherits")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            if attrs.get("type") == NodeType.CLASS:
                self._parse_inherits(node_id, attrs["ast"])
        logging.info("Finished parsing inherits, added %d edges", self.G_inherits.number_of_edges())

        # 4) parse invokes
        logging.info("Parsing RepoSkeleton to extract invokes")
        for node_id, attrs in list(self.G_code.nodes(data=True)):
            self._parse_invokes(node_id, attrs["ast"])
        logging.info("Finished parsing invokes, added %d edges", self.G_invokes.number_of_edges())

        logging.info("Finished parsing RepoSkeleton, now has %d nodes and %d edges", self.G.number_of_nodes(), self.G.number_of_edges())


    # Control flow types whose bodies may contain nested function/class definitions
    _CONTROL_FLOW_TYPES = (ast.If, ast.Try, ast.With, ast.For, ast.While,
                           ast.AsyncWith, ast.AsyncFor)

    @staticmethod
    def _get_control_flow_bodies(node: ast.AST) -> list:
        """Extract all branch bodies from a control flow node."""
        bodies = []
        if isinstance(node, ast.If):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        elif isinstance(node, ast.Try):
            bodies.append(node.body)
            for handler in node.handlers:
                bodies.append(handler.body)
            if node.orelse:
                bodies.append(node.orelse)
            if node.finalbody:
                bodies.append(node.finalbody)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            bodies.append(node.body)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            bodies.append(node.body)
            if node.orelse:
                bodies.append(node.orelse)
        return bodies

    def _extract_from_control_flow(self, stmts, file_id, source_code, get_range, parent_id):
        """Recursively scan control flow blocks for FunctionDef / ClassDef."""
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_nid = f"{file_id}:{func_name}"
                if func_nid not in self.G:
                    start, end = get_range(node)
                    self._add_node(func_nid, type=NodeType.FUNCTION,
                                   code=extract_source_by_lines(source_code, start, end),
                                   parent_id=parent_id, ast=node,
                                   start_line=start, end_line=end)
            elif isinstance(node, ast.ClassDef):
                cls_name = node.name
                cls_nid = f"{file_id}:{cls_name}"
                if cls_nid not in self.G:
                    start, end = get_range(node)
                    self._add_node(cls_nid, type=NodeType.CLASS,
                                   code=extract_source_by_lines(source_code, start, end),
                                   parent_id=parent_id, ast=node,
                                   start_line=start, end_line=end)
                    # Methods inside the class
                    for body_node in node.body:
                        if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            meth_nid = f"{file_id}:{cls_name}.{body_node.name}"
                            if meth_nid not in self.G:
                                s2, e2 = get_range(body_node)
                                self._add_node(meth_nid, type=NodeType.METHOD,
                                               code=extract_source_by_lines(source_code, s2, e2),
                                               parent_id=cls_nid, ast=body_node,
                                               start_line=s2, end_line=e2)
            # Recurse into nested control flow
            if isinstance(node, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node):
                    self._extract_from_control_flow(block, file_id, source_code, get_range, parent_id)

    def _parse_file(self, file_id: str, tree: ast.AST, source_code: str) -> None:
        """
        Parse definitions at the top level and within control-flow blocks:
        - Top-level class:    ./x/y.py:ClassName
        - Top-level function: ./x/y.py:func
        - Top-level method:   ./x/y.py:ClassName.method
        - Conditionally defined functions/classes (inside if/try/with blocks, etc.)
        """

        def get_range(node):
            start_inc, _, body_end, end_exc = get_node_range_robust(node, source_code)
            return start_inc, body_end

        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_nid  = f"{file_id}:{func_name}"
                start, end = get_range(node)

                self._add_node(
                    func_nid,
                    type=NodeType.FUNCTION,
                    code=extract_source_by_lines(source_code, start, end),
                    parent_id=file_id,
                    ast=node,
                    start_line=start,
                    end_line=end,
                )

            elif isinstance(node, ast.ClassDef):
                cls_name = node.name
                cls_nid  = f"{file_id}:{cls_name}"
                start, end = get_range(node)

                self._add_node(
                    cls_nid,
                    type=NodeType.CLASS,
                    code=extract_source_by_lines(source_code, start, end),
                    parent_id=file_id,
                    ast=node,
                    start_line=start,
                    end_line=end,
                )

                for body in node.body:
                    if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        meth_name = body.name
                        meth_nid  = f"{file_id}:{cls_name}.{meth_name}"
                        start2, end2 = get_range(body)

                        self._add_node(
                            meth_nid,
                            type=NodeType.METHOD,
                            code=extract_source_by_lines(source_code, start2, end2),
                            parent_id=cls_nid,
                            ast=body,
                            start_line=start2,
                            end_line=end2,
                        )

        # Extract function/class definitions from control flow blocks
        for node in getattr(tree, "body", []):
            if isinstance(node, self._CONTROL_FLOW_TYPES):
                for block in self._get_control_flow_bodies(node):
                    self._extract_from_control_flow(block, file_id, source_code, get_range, file_id)

    def _init_alias_map(self, node_id: str) -> dict:
        self.G.nodes[node_id]["alias_to_entity"] = {}
        alias_map = self.G.nodes[node_id]["alias_to_entity"]
        for _, child_id, edata in self.G_tree.out_edges(node_id, data=True):
            if edata.get("type") == EdgeType.CONTAINS:
                child_name = self.G.nodes[child_id].get("name")
                alias_map[child_name] = child_id

        if node_id.endswith("__init__.py"):
            _, parent_id = self.get_parent(node_id) 
            for _, child_id, edata in self.G_tree.out_edges(parent_id, data=True):
                if edata.get("type") == EdgeType.CONTAINS and child_id != node_id:
                    child_name = self.G.nodes[child_id].get("name")
                    child_type = self.G.nodes[child_id].get("type")
                    if child_type == NodeType.DIRECTORY:
                        child_init_id = normalize_path(f"{child_id}/__init__.py")
                        if child_init_id in self.G:
                            alias_map[child_name] = child_init_id
                    else:
                        alias_map[child_name] = child_id
        return alias_map

    def _parse_imports(self, node_id: str, tree: ast.AST, alias_links: nx.DiGraph) -> None:
        if node_id not in self.G or self.G.nodes[node_id].get("type") not in [NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
            logging.warning(f"[imports:skip] invalid src code for parse: {node_id}")
            return

        current_module = self.G.nodes[node_id].get("module", "")

        for node in ast.iter_child_nodes(tree):
            alias_map: dict = self.G.nodes[node_id].get("alias_to_entity")
            if alias_map is None:
                logging.warning(f"missing alias map for: {node_id}")
                alias_map = self._init_alias_map(node_id)

            if isinstance(node, ast.Import):
                for al in node.names:
                    module_id = self._find_module_file(al.name)
                    alias = al.asname or al.name
                    if module_id:
                        self._add_edge(node_id, module_id, type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = module_id
                        self._propagate_aliases(node_id, alias, alias_links)

            elif isinstance(node, ast.ImportFrom):
                level = node.level
                abs_module = self._resolve_relative_module(current_module, node.module, level)
                if not abs_module:
                    continue
                module_id = self._find_module_file(abs_module)
                if module_id is None:
                    continue
                module_alias_map: dict = self.G.nodes[module_id].get("alias_to_entity")
                if module_alias_map is None:
                    logging.warning(f"missing alias map for: {module_id}")
                    module_alias_map = self._init_alias_map(module_id)

                if any((al.name == "*" for al in node.names)):
                    for alias, entity in module_alias_map.items():
                        self._add_edge(node_id, entity, type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = entity
                        self._propagate_aliases(node_id, alias, alias_links)
                    alias_links.add_edge(f'{module_id}:*', f'{node_id}:*')
                    continue

                for al in node.names:
                    target = al.name
                    alias = al.asname or al.name
                    if target in module_alias_map:
                        self._add_edge(node_id, module_alias_map[target], type=EdgeType.IMPORTS, alias=alias)
                        alias_map[alias] = module_alias_map[target]
                        self._propagate_aliases(node_id, alias, alias_links)
                    else:
                        alias_links.add_edge(f'{module_id}:{target}', f'{node_id}:{alias}')

    def _propagate_aliases(self, node_id: str, alias: str, alias_links: nx.DiGraph) -> None:
        alias_link = f'{node_id}:{alias}'
        entity = self.G.nodes[node_id].get("alias_to_entity", {}).get(alias)
        if not entity:
            logging.warning(f"missing entity for alias link: {alias_link}")
            return

        for _, dst in alias_links.out_edges(alias_link):
            dst_node, dst_alias = dst.split(":", 1)
            alias_map: dict = self.G.nodes[dst_node].get("alias_to_entity")
            if alias_map is None:
                logging.warning(f"missing alias map for: {dst_node}")
                alias_map = self._init_alias_map(dst_node)
            if alias_map.get(dst_alias) != entity:
                alias_map[dst_alias] = entity
                self._add_edge(dst_node, entity, type=EdgeType.IMPORTS, alias=dst_alias)
                self._propagate_aliases(dst_node, dst_alias, alias_links)

        star_link = f'{node_id}:*'
        for _, dst in alias_links.out_edges(star_link):
            dst_node, dst_alias = dst.split(":", 1)
            assert dst_alias == "*", f"Expected '*', got {dst_alias}"
            dst_alias = alias  
            alias_map: dict = self.G.nodes[dst_node].get("alias_to_entity")
            if alias_map is None:
                logging.warning(f"missing alias map for: {dst_node}")
                alias_map = self._init_alias_map(dst_node)
            if alias_map.get(dst_alias) != entity:
                alias_map[dst_alias] = entity
                self._add_edge(dst_node, entity, type=EdgeType.IMPORTS, alias=dst_alias)
                self._propagate_aliases(dst_node, dst_alias, alias_links)


    def _parse_inherits(self, node_id: str, tree: ast.AST) -> None:
        if node_id not in self.G or self.G.nodes[node_id].get("type") != NodeType.CLASS or not isinstance(tree, ast.ClassDef):
            logging.warning(f"[inherits:skip] invalid src class for parse: {node_id}")
            return

        for base in tree.bases:
            if isinstance(base, ast.Name) or isinstance(base, ast.Attribute):
                base_name = ast.unparse(base)
                entity = self._find_entity(node_id, base_name)
                if entity:
                    self._add_edge(node_id, entity, type=EdgeType.INHERITS)
                else:
                    logging.debug(f"[inherits:miss] base class not found: {base_name} in {node_id}")

    def _parse_invokes(self, node_id: str, tree: ast.AST) -> None:
        if node_id not in self.G or self.G.nodes[node_id].get("type") not in [NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
            logging.warning(f"[invokes:skip] invalid src code for parse: {node_id}")
            return

        ntype = self.G.nodes[node_id].get("type")
        nodes_to_walk = []
        if ntype == NodeType.FILE:
            for child in ast.iter_child_nodes(tree):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nodes_to_walk.append(child)
        elif ntype == NodeType.CLASS:
            for child in ast.iter_child_nodes(tree):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes_to_walk.append(child)
        else:
            nodes_to_walk.append(tree)

        for subtree in nodes_to_walk:
            for child in ast.walk(subtree):
                if isinstance(child, (ast.Call, ast.Await)):
                    if isinstance(child, ast.Await):
                        child = child.value
                    if not isinstance(child, ast.Call):
                        continue
                    func_name = ast.unparse(child.func)
                    entity = self._find_entity(node_id, func_name)
                    if entity:
                        if entity != node_id:
                            self._add_edge(node_id, entity, type=EdgeType.INVOKES)
                    else:
                        entity = self._find_entity_fuzzy(node_id, func_name)
                        if entity:
    
                            if entity != node_id:
                                self._add_edge(node_id, entity, type=EdgeType.INVOKES)
                        else:
                            logging.debug(f"[invokes:miss] invoked entity not found: {func_name} in {node_id}")

    def _resolve_relative_module(self, current_module: str, module: Optional[str], level: int) -> str:
        if level == 0:
            return module
        base_parts = current_module.split(".") if current_module else []
        base_parts = base_parts[:len(base_parts) - level] if level <= len(base_parts) else []
        if module:
            return ".".join(base_parts + [module]) if base_parts else module
        else:
            return ".".join(base_parts)

    def _find_module_file(self, module_name: str) -> Optional[str]:
        module_path = normalize_path("./" + module_name.replace('.', '/'))
        file_path = normalize_path(module_path + ".py")
        init_path = normalize_path(f"{module_path}/__init__.py")

        if file_path in self.G:
            return file_path
        elif init_path in self.G:
            return init_path
        else:
            return None

    def _find_entity(self, module_id: str, qual_name: str) -> Optional[str]:
        if module_id not in self.G:
            return None

        parts = qual_name.split(".")
        cur_entity = module_id
        for part in parts:
            alias_map: dict = self.G.nodes[cur_entity].get("alias_to_entity", {})
            if alias_map.get(part):
                cur_entity = alias_map[part]
            elif self.G.nodes[module_id].get("type") in [NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                parent_exists, parent_id = self.get_parent(module_id)
                if parent_exists:
                    return self._find_entity(parent_id, qual_name)
            else:
                return None
        return cur_entity

    def _find_entity_fuzzy(self, node_id: str, qual_name: str) -> Optional[str]:
        """
        Fuzzily resolve entities to handle calls that cannot be precisely parsed, such as self.xxx and super().xxx.

        Strategy:
        1. self.method() -> look up method in the current class and its inheritance chain
        2. self.attr.method() -> try to infer the type of attr, or fall back to a global match by method name
        3. super().method() -> look up method in the parent class
        4. var.method() -> try to find a method with the same name in the current class (assuming var may be the same type as self)
        5. var.attr.method() -> take the final method name and search in the current class or globally
        """
        parts = qual_name.split(".")
        if not parts:
            return None

        containing_class = self._get_containing_class(node_id)
        
        if parts[0] == "self" and len(parts) >= 2:
            method_name = parts[-1]  
            
            if len(parts) == 2 and containing_class:
                entity = self._find_method_in_class_hierarchy(containing_class, method_name)
                if entity:
                    return entity

            if len(parts) >= 3 and containing_class:
                attr_name = parts[1]
                attr_type = self._infer_attribute_type(containing_class, attr_name)
                if attr_type:
                    entity = self._find_method_in_class_hierarchy(attr_type, method_name)
                    if entity:
                        return entity
                    
            return self._find_method_by_name_global(method_name)

        if parts[0] == "super()" and len(parts) == 2 and containing_class:
            method_name = parts[1]
            return self._find_method_in_parent_classes(containing_class, method_name)

        if len(parts) >= 2 and containing_class:
            method_name = parts[-1] 
            
            if len(parts) == 2:
                entity = self._find_method_in_class_hierarchy(containing_class, method_name)
                if entity:
                    return entity

            var_name = parts[0]
            var_type = self._infer_local_var_type(node_id, var_name)
            if var_type:
                if len(parts) > 2:
                    type_infer_failed = False
                    for i in range(1, len(parts) - 1):
                        attr_name = parts[i]
                        attr_type = self._infer_attribute_type(var_type, attr_name)
                        if attr_type:
                            var_type = attr_type
                        else:              
                            type_infer_failed = True
                            break

                    if type_infer_failed:
                        return self._find_method_by_name_global(method_name)

                entity = self._find_method_in_class_hierarchy(var_type, method_name)
                if entity:
                    return entity

            return self._find_method_by_name_global(method_name)

        return None

    def _infer_local_var_type(self, node_id: str, var_name: str) -> Optional[str]:
        """
        Infer the type of a local variable.

        Strategy:
        1. Look for assignments like `var = self.method()`, and return the current class
        (assuming it returns `self`'s type).
        2. Look for assignments like `var = SomeClass(...)`.
        3. Look for assignments like `var = some_instance.method()`.
        """
        if node_id not in self.G:
            return None

        node_ast = self.G.nodes[node_id].get("ast")
        if not node_ast:
            return None

        containing_class = self._get_containing_class(node_id)
        _, file_id = self.get_parent(node_id)

        for stmt in ast.walk(node_ast):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if (isinstance(stmt.value, ast.Call) and
                            isinstance(stmt.value.func, ast.Attribute) and
                            isinstance(stmt.value.func.value, ast.Name) and
                            stmt.value.func.value.id == "self"):
                            if containing_class:
                                return containing_class

                        # var = SomeClass(...)
                        type_name = self._extract_type_from_value(stmt.value)
                        if type_name:
                            entity = self._find_entity(file_id, type_name) if file_id else None
                            if entity:
                                return entity
                            if containing_class:
                                entity = self._find_entity(containing_class, type_name)
                                if entity:
                                    return entity

        return None

    def _get_containing_class(self, node_id: str) -> Optional[str]:
        ntype = self.G.nodes.get(node_id, {}).get("type")
        if ntype == NodeType.CLASS:
            return node_id
        elif ntype == NodeType.METHOD:
            _, parent_id = self.get_parent(node_id)
            if parent_id and self.G.nodes.get(parent_id, {}).get("type") == NodeType.CLASS:
                return parent_id
        return None

    def _find_method_in_class_hierarchy(self, class_id: str, method_name: str) -> Optional[str]:
        if class_id not in self.G:
            return None

        alias_map = self.G.nodes[class_id].get("alias_to_entity", {})
        if method_name in alias_map:
            return alias_map[method_name]
        
        for _, dst, edata in self.G.out_edges(class_id, data=True):
            if edata.get("type") == EdgeType.INHERITS:
                result = self._find_method_in_class_hierarchy(dst, method_name)
                if result:
                    return result

        return None

    def _find_method_in_parent_classes(self, class_id: str, method_name: str) -> Optional[str]:
        if class_id not in self.G:
            return None

        for _, dst, edata in self.G.out_edges(class_id, data=True):
            if edata.get("type") == EdgeType.INHERITS:
                alias_map = self.G.nodes[dst].get("alias_to_entity", {})
                if method_name in alias_map:
                    return alias_map[method_name]

                result = self._find_method_in_class_hierarchy(dst, method_name)
                if result:
                    return result

        return None

    def _infer_attribute_type(self, class_id: str, attr_name: str) -> Optional[str]:
        """
        Try to infer the type of a class attribute.

        Strategy:
        1. Look for assignments in __init__ such as `self.attr = SomeClass(...)` or `self._attr = ...`.
        2. Look for type annotations like `attr: SomeClass`.
        3. Look for attributes returned by property getters.
        """
        if class_id not in self.G:
            return None

        class_ast = self.G.nodes[class_id].get("ast")
        if not isinstance(class_ast, ast.ClassDef):
            return None
        
        attr_variants = [attr_name, f"_{attr_name}"]

        _, file_id = self.get_parent(class_id)

        for node in class_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute) and
                                isinstance(target.value, ast.Name) and
                                target.value.id == "self" and
                                target.attr in attr_variants):
                               
                                type_name = self._extract_type_from_value(stmt.value)
                                if type_name:
                                   
                                    entity = self._find_entity(file_id, type_name) if file_id else None
                                    if entity:
                                        return entity
                              
                                    return self._find_entity(class_id, type_name)

        for node in class_ast.body:
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in attr_variants:
                    if node.annotation:
                        type_name = ast.unparse(node.annotation)
                        entity = self._find_entity(file_id, type_name) if file_id else None
                        if entity:
                            return entity
                        return self._find_entity(class_id, type_name)

        return None

    def _extract_type_from_value(self, value_node: ast.AST) -> Optional[str]:
       
        # self.attr = SomeClass(...)
        if isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                return value_node.func.id
            elif isinstance(value_node.func, ast.Attribute):
                return ast.unparse(value_node.func)
   
        if isinstance(value_node, ast.BoolOp) and isinstance(value_node.op, ast.Or):

            for val in value_node.values:
                result = self._extract_type_from_value(val)
                if result:
                    return result


        if isinstance(value_node, ast.IfExp):
            result = self._extract_type_from_value(value_node.orelse)
            if result:
                return result
            result = self._extract_type_from_value(value_node.body)
            if result:
                return result

        return None

    def _find_method_by_name_global(self, method_name: str) -> Optional[str]:
        """
        Search for methods with the same name across the entire repo (a fuzzy-match fallback strategy).
        Return only a unique match; if multiple methods share the same name, return None to avoid false positives.
        """
       
        COMMON_METHODS = {
            "__init__", "__str__", "__repr__", "__eq__", "__hash__",
            "get", "set", "add", "remove", "update", "delete", "save",
            "clone", "copy", "items", "keys", "values", "as_sql",
        }
        if method_name in COMMON_METHODS or method_name.startswith("_"):
            return None

        matches = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == NodeType.METHOD:
                if attrs.get("name") == method_name:
                    matches.append(nid)

  
        if len(matches) == 1:
            return matches[0]

        return None


    def get_parent(self, node_id: str) -> Tuple[bool, Optional[str]]:
        nid = normalize_path(node_id)
        if nid not in self.G:
            return False, None

        if nid == ".":
            return True, None
        elif ":" in nid:
            path_part, qual = nid.split(":", 1)
            parts = qual.split(".")
            if len(parts) <= 1:
                parent_id = path_part
            else:
                parent_qual = ".".join(parts[:-1])
                parent_id = f"{path_part}:{parent_qual}"
            parent_id = normalize_path(parent_id)
        else:
            parent_id = normalize_path(Path(nid).parent)
        return parent_id in self.G, parent_id

    def get_name(self, nid: str, ntype: Optional[str] = None, for_print: bool = False, with_badge: bool = False) -> str:

        badge_map = {
            NodeType.DIRECTORY: "@dir",
            NodeType.FILE: "@file",
            NodeType.CLASS: "@class",
            NodeType.FUNCTION: "@func",
            NodeType.METHOD: "@method",
        }

        if ntype is None:
            ntype = self.G_tree.nodes[nid].get("type")

        name = ""        
        if ntype == NodeType.DIRECTORY or ntype == NodeType.FILE:
            if nid == ".":
                name = "."
            else:
                name = nid.split("/")[-1]
        else:
            qual = nid.split(":", 1)[1]
            parts = [p for p in qual.split(".") if p]
            last = parts[-1] if parts else qual
            name = f".{last}" if for_print and len(parts) > 1 else last
            
        if with_badge and ntype in badge_map:
            name += f" {badge_map[ntype]}"

        if ntype == NodeType.FILE and not for_print:
            name = name.rstrip(".py")

        return name

    def find_node(self, path: str, suffix_match: bool = True) -> Optional[str]:
        norm_input = normalize_path(path)
        if norm_input in self.G:
            return norm_input

        if suffix_match:
            for nid in self.G.nodes:
                if nid.endswith(norm_input):
                    return nid
        return None

    def find_file(self, path: str, suffix_match: bool = True) -> Optional[str]:
        norm_input = normalize_path(path)
        if norm_input in self.G and self.G.nodes[norm_input].get("type") == NodeType.FILE:
            return norm_input

        if suffix_match:
            for nid, attrs in self.G.nodes(data=True):
                if attrs.get("type") == NodeType.FILE and nid.endswith(norm_input):
                    return nid
        return None
    
    def all_paths(self, include_types: List[str]) -> List[str]:
        include_set = set(include_types)
        return sorted(
            nid for nid, attrs in self.G.nodes(data=True)
            if attrs.get("type") in include_set
        )
        
    
    def to_dict(self, dep_to_rpg_map: Optional[Dict[str, List[str]]] = None) -> dict:
        """
        Serialize graph to dict (no AST saved).

        Args:
            dep_to_rpg_map: Optional mapping from dep node IDs to RPG node IDs.
                           If provided, adds rpg_nodes field to each node and
                           src_rpg_nodes/dst_rpg_nodes to each edge.
        """
        data = {
            "repo_dir": self.repo_dir,
            "nodes": {},
            "edges": [],
        }

        for nid, attrs in self.G.nodes(data=True):
            node_data = {k: v for k, v in attrs.items() if k != "ast"}  # drop AST
            # Add RPG node mapping if available
            if dep_to_rpg_map is not None:
                node_data["rpg_nodes"] = dep_to_rpg_map.get(nid, [])
            data["nodes"][nid] = node_data

        for u, v, attrs in self.G.edges(data=True):
            edge_data = {
                "src": u,
                "dst": v,
                "attrs": dict(attrs),
            }
            # Add RPG node mappings for src and dst if available
            if dep_to_rpg_map is not None:
                edge_data["src_rpg_nodes"] = dep_to_rpg_map.get(u, [])
                edge_data["dst_rpg_nodes"] = dep_to_rpg_map.get(v, [])
            data["edges"].append(edge_data)

        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct graph from dict (reparse AST afterwards)"""
        obj = cls(repo_dir=data.get("repo_dir", ""))

        for nid, attrs in data["nodes"].items():
            obj.G.add_node(nid, **attrs)

        for e in data["edges"]:
            u = e["src"]
            v = e["dst"]
            attrs = e["attrs"]
            obj.G.add_edge(u, v, **attrs)

        return obj

    def reparse_ast(self, filter_func=_exclude_irrelevant_for_parse):
        """
        Reparse source code to restore AST and code structure.
        Must be called after from_dict().
        """
        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") != NodeType.FILE or not filter_func(nid):
                continue

            content = attrs.get("code") or ""
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            self.G.nodes[nid]["ast"] = tree

            # rebuild functions / classes nodes
            self._parse_file(nid, tree, content)

        # re-run import / invoke / inherit pass
        alias_links = nx.DiGraph()
        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") in [NodeType.FILE, NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD]:
                self._init_alias_map(nid)
                self._parse_imports(nid, attrs.get("ast"), alias_links)

        for nid, attrs in list(self.G.nodes(data=True)):
            if attrs.get("type") == NodeType.CLASS:
                self._parse_inherits(nid, attrs.get("ast"))

        for nid, attrs in list(self.G.nodes(data=True)):
            self._parse_invokes(nid, attrs.get("ast"))

        logging.info("AST re-parsed & semantic edges reconstructed")
        
