#!/usr/bin/env python3
"""Check whether every class and module-level function definition in a
directory of Python files is actually referenced from somewhere else.

By default skips ``tests/``, ``examples/``, ``__pycache__/``, and any
hidden ``.xxx`` directories."""

import argparse
import ast
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, Optional

DEFAULT_IGNORE_DIRS = {'tests', 'examples', '__pycache__'}
VALID_SORT_KEYS = ('name', 'file', 'type', 'ref_count', 'external_ref')


class DefinitionInfo:
    """Container for a single class or function definition."""
    def __init__(self, name: str, def_type: str, file_path: str, line_no: int, end_line_no: int = None):
        self.name = name
        self.def_type = def_type  # 'class' or 'function'
        self.file_path = file_path
        self.line_no = line_no
        self.end_line_no = end_line_no  # line where the definition ends
        # references is a list of tuples: (source_file, source_def_name, line_no).
        # source_def_name is None for module-level references.
        self.references: List[Tuple[str, Optional[str], int]] = []

    @property
    def ref_count(self) -> int:
        return len(self.references)

    @property
    def is_referenced_by_other_files(self) -> bool:
        """True if this definition is referenced from a file other than its own."""
        for ref_file, _, _ in self.references:
            if ref_file != self.file_path:
                return True
        return False

    def get_external_ref_count(self) -> int:
        """Number of times this definition is referenced from other files."""
        return sum(1 for ref_file, _, _ in self.references if ref_file != self.file_path)

    def get_referencing_defs(self) -> Set[Tuple[str, str]]:
        """Return the set of other definitions that reference this one as ``(file_path, def_name)``."""
        result = set()
        for ref_file, ref_def_name, _ in self.references:
            if ref_def_name is not None:  # exclude module-level references
                result.add((ref_file, ref_def_name))
        return result


class ReferenceChecker:
    """Tool for checking class/function references across a Python codebase."""

    def __init__(self, root_dir: str, ignore_dirs: Set[str] = None):
        self.root_dir = Path(root_dir).resolve()
        self.definitions: Dict[str, DefinitionInfo] = {}
        self.all_py_files: List[Path] = []
        # Default: ignore tests/, examples/ and __pycache__/
        self.ignore_dirs: Set[str] = ignore_dirs if ignore_dirs is not None else {'tests', 'examples', '__pycache__'}

    def collect_py_files(self) -> List[Path]:
        """Walk ``root_dir`` and return every ``.py`` file, honoring ``ignore_dirs`` and skipping hidden dirs."""
        py_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip ignored dirs and any directory starting with '.'
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs and not d.startswith('.')]
            for file in files:
                if file.endswith('.py'):
                    py_files.append(Path(root) / file)
        return py_files

    def get_relative_path(self, file_path: Path) -> str:
        """Return ``file_path`` expressed relative to ``root_dir`` (falls back to absolute)."""
        try:
            return str(file_path.relative_to(self.root_dir))
        except ValueError:
            return str(file_path)

    def extract_definitions(self, file_path: Path) -> List[DefinitionInfo]:
        """Extract every top-level class and standalone function definition from a single file."""
        definitions = []
        rel_path = self.get_relative_path(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: cannot read {file_path}: {e}", file=sys.stderr)
            return definitions

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            print(f"Warning: syntax error in {file_path}: {e}", file=sys.stderr)
            return definitions

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                info = DefinitionInfo(
                    name=node.name,
                    def_type='class',
                    file_path=rel_path,
                    line_no=node.lineno,
                    end_line_no=getattr(node, 'end_lineno', None)
                )
                definitions.append(info)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Standalone function (module-level)
                info = DefinitionInfo(
                    name=node.name,
                    def_type='function',
                    file_path=rel_path,
                    line_no=node.lineno,
                    end_line_no=getattr(node, 'end_lineno', None)
                )
                definitions.append(info)

        return definitions

    def _file_to_module_path(self, file_path: Path) -> str:
        """Convert a file path into a dotted module path relative to ``root_dir``."""
        try:
            rel = file_path.relative_to(self.root_dir)
        except ValueError:
            return ''
        parts = list(rel.parts)
        if not parts:
            return ''
        if parts[-1] == '__init__.py':
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].rsplit('.', 1)[0]  # strip the .py suffix
        return '.'.join(parts)

    def _resolve_relative_import(self, current_file: Path, level: int, module: Optional[str]) -> str:
        """Resolve a relative import to an absolute dotted module path."""
        current_module = self._file_to_module_path(current_file)
        parts = current_module.split('.') if current_module else []
        # Walk up ``level`` package levels
        if level > len(parts):
            return module or ''
        base_parts = parts[:-level] if level > 0 else parts
        if module:
            return '.'.join(base_parts + [module]) if base_parts else module
        return '.'.join(base_parts) if base_parts else ''

    def _build_import_info(self, tree: ast.Module, file_path: Path
                           ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Extract import information from an AST.

        Returns:
            name_import_map:  local_name -> source_module
                e.g. 'from a.b import Foo'         => {'Foo': 'a.b'}
                e.g. 'from a.b import Foo as Bar'   => {'Bar': 'a.b'}
            module_alias_map: alias -> full_module_name
                e.g. 'import a.b'                   => {'a': 'a.b'}
                e.g. 'import a.b as c'              => {'c': 'a.b'}
            alias_to_original: local_alias -> original_name
                e.g. 'from a.b import Foo as Bar'   => {'Bar': 'Foo'}
        """
        name_import_map: Dict[str, str] = {}
        module_alias_map: Dict[str, str] = {}
        alias_to_original: Dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Resolve the module path (handles relative imports)
                if node.level and node.level > 0:
                    resolved_module = self._resolve_relative_import(
                        file_path, node.level, node.module)
                else:
                    resolved_module = node.module or ''
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    name_import_map[local_name] = resolved_module
                    if alias.asname:
                        alias_to_original[alias.asname] = alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        module_alias_map[alias.asname] = alias.name
                    else:
                        top_name = alias.name.split('.')[0]
                        module_alias_map[top_name] = alias.name

        return name_import_map, module_alias_map, alias_to_original

    def _module_path_matches(self, import_source: str, def_module_path: str) -> bool:
        """Return True if the import source matches the module path of a definition (suffix match supported)."""
        if not import_source or not def_module_path:
            return False
        if import_source == def_module_path:
            return True
        # Suffix match: allow the import path to align with the tail of the def module path
        if def_module_path.endswith('.' + import_source):
            return True
        if import_source.endswith('.' + def_module_path):
            return True
        # Tail-segment comparison after splitting on '.'
        import_parts = import_source.split('.')
        def_parts = def_module_path.split('.')
        min_len = min(len(import_parts), len(def_parts))
        if import_parts[-min_len:] == def_parts[-min_len:]:
            return True
        return False

    def _resolve_reference(
        self, name: str, ref_file: str,
        defs: List[DefinitionInfo],
        name_import_map: Dict[str, str],
        module_alias_map: Dict[str, str],
        def_module_paths: Dict[str, str]
    ) -> List[DefinitionInfo]:
        """Disambiguate homonymous definitions using import information.

        Returns the list of most-likely matching definitions (ideally just one).
        """
        # 1. Same-file reference → match definitions in the same file directly
        same_file_defs = [d for d in defs if d.file_path == ref_file]
        if same_file_defs:
            return same_file_defs

        # 2. Imported via ``from ... import ...`` → match by module path
        if name in name_import_map:
            import_source = name_import_map[name]
            matched = [d for d in defs
                       if self._module_path_matches(import_source,
                                                    def_module_paths.get(d.file_path, ''))]
            if matched:
                return matched
            # Imported from a different module but no definition matches → ignore
            return []

        # 3. Import source unknown (e.g. star import) → conservatively count when there is exactly one definition
        if len(defs) == 1:
            return defs
        return []

    def find_references_in_file(
        self, file_path: Path, names_to_find: Set[str],
        file_definitions: List[DefinitionInfo] = None
    ) -> Tuple[Dict[str, List[Tuple[int, Optional[str]]]], Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Find real code references to ``names_to_find`` in ``file_path`` via AST.

        Only counts genuine code references — strings and comments are
        excluded.  Also returns the file's import information so the
        caller can disambiguate same-name definitions.

        Args:
            file_path: file to analyze
            names_to_find: set of names to look up
            file_definitions: definitions in this file (used to attribute each
                reference to its enclosing scope)

        Returns:
            ``(references, name_import_map, module_alias_map, alias_to_original)``
            where ``references`` is ``{name: [(line_no, scope_def_name), ...]}``
            and ``scope_def_name`` is ``None`` when the reference is at module
            top level.
        """
        references: Dict[str, List[Tuple[int, Optional[str]]]] = defaultdict(list)
        empty = references, {}, {}, {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            return empty

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return empty

        name_import_map, module_alias_map, alias_to_original = \
            self._build_import_info(tree, file_path)

        # Reverse mapping: ``from X import Foo as Bar`` → ``Bar -> Foo``.
        # When Foo is in names_to_find, occurrences of Bar in the code
        # should also be recorded as references to Foo.
        alias_lookup: Dict[str, str] = {
            alias: orig for alias, orig in alias_to_original.items()
            if orig in names_to_find
        }

        # Build a line-range → enclosing-definition-name lookup
        def find_scope(line_no: int) -> Optional[str]:
            """Return the name of the definition that contains ``line_no``, or None."""
            if not file_definitions:
                return None
            for d in file_definitions:
                if d.end_line_no and d.line_no <= line_no <= d.end_line_no:
                    return d.name
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in names_to_find:
                    scope = find_scope(node.lineno)
                    references[node.id].append((node.lineno, scope))
                elif node.id in alias_lookup:
                    # Aliased reference → record against the original name
                    scope = find_scope(node.lineno)
                    references[alias_lookup[node.id]].append((node.lineno, scope))

            elif isinstance(node, ast.Attribute) and node.attr in names_to_find:
                # Only count ``module.Name`` form (value must be a known module alias)
                if (isinstance(node.value, ast.Name)
                        and node.value.id in module_alias_map):
                    scope = find_scope(node.lineno)
                    references[node.attr].append((node.lineno, scope))

            elif isinstance(node, ast.ImportFrom) and node.names:
                # ``from ... import Foo`` itself counts as a reference (at module top level)
                for alias in node.names:
                    if alias.name in names_to_find:
                        references[alias.name].append((node.lineno, None))

        return references, name_import_map, module_alias_map, alias_to_original

    def analyze(self) -> List[DefinitionInfo]:
        """Analyze every file, collecting definitions and reference information.

        Uses AST parsing to find real code references (excluding
        strings/comments), and disambiguates same-name definitions via
        import information.
        """
        # Collect all Python files
        self.all_py_files = self.collect_py_files()

        if not self.all_py_files:
            print(f"Warning: no .py files found under {self.root_dir}", file=sys.stderr)
            return []

        # Pass 1: collect every definition
        all_definitions: List[DefinitionInfo] = []
        file_to_defs: Dict[str, List[DefinitionInfo]] = defaultdict(list)
        for py_file in self.all_py_files:
            defs = self.extract_definitions(py_file)
            all_definitions.extend(defs)
            rel_path = self.get_relative_path(py_file)
            file_to_defs[rel_path].extend(defs)

        if not all_definitions:
            print("Warning: no class or function definitions found", file=sys.stderr)
            return []

        # Build the name → [definitions] map (handles homonymous defs)
        name_to_defs: Dict[str, List[DefinitionInfo]] = defaultdict(list)
        for def_info in all_definitions:
            name_to_defs[def_info.name].append(def_info)

        # Compute a dotted module path for each file containing a definition
        # (used for import-based disambiguation)
        def_module_paths: Dict[str, str] = {}
        for def_info in all_definitions:
            if def_info.file_path not in def_module_paths:
                full_path = self.root_dir / def_info.file_path
                def_module_paths[def_info.file_path] = self._file_to_module_path(full_path)

        names_to_find = set(name_to_defs.keys())

        # Pass 2: walk every file's AST to find references and disambiguate them
        for py_file in self.all_py_files:
            rel_path = self.get_relative_path(py_file)
            # Pass the file's own definitions so we can determine the enclosing scope of each reference
            refs, name_import_map, module_alias_map, _ = \
                self.find_references_in_file(py_file, names_to_find, file_to_defs.get(rel_path))

            for name, ref_infos in refs.items():
                defs_for_name = name_to_defs[name]

                for line_no, scope_def_name in ref_infos:
                    # Decide which definition(s) this reference belongs to
                    if len(defs_for_name) == 1:
                        def_info = defs_for_name[0]
                        # Exclude the definition line itself
                        if rel_path == def_info.file_path and line_no == def_info.line_no:
                            continue
                        # For cross-file references, skip if the file explicitly imports the name from another module
                        if rel_path != def_info.file_path and name in name_import_map:
                            import_src = name_import_map[name]
                            def_mod = def_module_paths.get(def_info.file_path, '')
                            if not self._module_path_matches(import_src, def_mod):
                                continue
                        def_info.references.append((rel_path, scope_def_name, line_no))
                    else:
                        # Multiple homonymous definitions → disambiguate
                        matched = self._resolve_reference(
                            name, rel_path, defs_for_name,
                            name_import_map, module_alias_map,
                            def_module_paths
                        )
                        for def_info in matched:
                            if rel_path == def_info.file_path and line_no == def_info.line_no:
                                continue
                            def_info.references.append((rel_path, scope_def_name, line_no))

        return all_definitions

    def _compute_reachability_from_main(self, definitions: List[DefinitionInfo]) -> Tuple[int, int]:
        """Compute how many definitions are reachable starting from ``main.py``.

        Performs a BFS from every definition in ``main.py``, following
        reference edges (call relationships).

        Returns:
            ``(reachable_count, unreachable_count)``
        """
        if not definitions:
            return 0, 0

        # Build the node-key → definition map
        key_to_def: Dict[Tuple[str, str], DefinitionInfo] = {
            (d.file_path, d.name): d for d in definitions
        }

        # Build an adjacency list: from_def -> [to_def] (reference direction: caller -> callee)
        # ``references`` records the *reference site*, so the edge is (scope_def_name, def.name)
        adjacency: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
        for d in definitions:
            target_key = (d.file_path, d.name)
            for ref_file, scope_def_name, _ in d.references:
                if scope_def_name is not None:  # exclude module-level references
                    src_key = (ref_file, scope_def_name)
                    if src_key in key_to_def:
                        adjacency[src_key].add(target_key)

        # All definitions in main.py are BFS start nodes
        start_nodes = set()
        for d in definitions:
            if d.file_path == 'main.py' or d.file_path.endswith('/main.py'):
                start_nodes.add((d.file_path, d.name))

        # BFS traversal
        visited: Set[Tuple[str, str]] = set()
        queue = list(start_nodes)
        for node in queue:
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        reachable_count = len(visited)
        unreachable_count = len(definitions) - reachable_count

        return reachable_count, unreachable_count

    VALID_SORT_KEYS = VALID_SORT_KEYS

    def print_report(self, definitions: List[DefinitionInfo], 
                     show_all: bool = True,
                     sort_by: str = 'file',
                     reverse: bool = False) -> None:
        """Print the analysis report.

        Args:
            definitions: list of definitions to report on
            show_all: when True, include definitions that have references
                (otherwise only print unreferenced ones)
            sort_by: sort key — one of ``'name'``, ``'file'``, ``'ref_count'``,
                ``'type'`` or ``'external_ref'``
            reverse: reverse the sort order
        """
        if not definitions:
            print("No definitions found.")
            return

        # Sort
        sort_keys = {
            'name': lambda x: x.name.lower(),
            'file': lambda x: (x.file_path, x.name.lower()),
            'ref_count': lambda x: x.ref_count,
            'type': lambda x: (x.def_type, x.name.lower()),
            'external_ref': lambda x: x.get_external_ref_count(),
        }
        key_func = sort_keys.get(sort_by, sort_keys['name'])
        definitions.sort(key=key_func, reverse=reverse)

        # Print table header
        print("\n" + "=" * 100)
        print(f"{'Type':<10} {'Name':<40} {'File':<30} {'Refs':<8} {'External'}")
        print("=" * 100)

        # Statistics
        total_classes = 0
        total_functions = 0
        unreferenced_classes = 0
        unreferenced_functions = 0
        no_external_ref_classes = 0
        no_external_ref_functions = 0

        for def_info in definitions:
            type_str = 'class' if def_info.def_type == 'class' else 'function'
            external_ref = 'yes' if def_info.is_referenced_by_other_files else 'no'

            # Tally
            if def_info.def_type == 'class':
                total_classes += 1
                if def_info.ref_count == 0:
                    unreferenced_classes += 1
                if not def_info.is_referenced_by_other_files:
                    no_external_ref_classes += 1
            else:
                total_functions += 1
                if def_info.ref_count == 0:
                    unreferenced_functions += 1
                if not def_info.is_referenced_by_other_files:
                    no_external_ref_functions += 1

            if show_all or def_info.ref_count == 0:
                print(f"{type_str:<10} {def_info.name:<40} {def_info.file_path:<30} {def_info.ref_count:<8} {external_ref}")

        # Compute reachability starting from main.py
        reachable, unreachable = self._compute_reachability_from_main(definitions)

        # Print statistics summary
        print("\n" + "=" * 100)
        print("Summary:")
        print(f"  Classes total: {total_classes}, unreferenced: {unreferenced_classes}, no external refs: {no_external_ref_classes}")
        print(f"  Functions total: {total_functions}, unreferenced: {unreferenced_functions}, no external refs: {no_external_ref_functions}")
        print(f"  Definitions total: {total_classes + total_functions}")
        print(f"  Reachable from main.py: {reachable}, unreachable: {unreachable}")
        print("=" * 100)

    def generate_html_graph(self, definitions: List[DefinitionInfo], output_path: str) -> None:
        """Generate an HTML visualization of the reference graph, grouped by file-path hierarchy.

        Args:
            definitions: list of definitions to render
            output_path: path for the output HTML file
        """
        # Build the file-tree structure
        file_tree: Dict[str, Any] = {}
        for d in definitions:
            parts = d.file_path.split('/')
            current = file_tree
            for part in parts[:-1]:  # directory parts
                if part not in current:
                    current[part] = {'_children': {}, '_files': []}
                current = current[part]['_children']
            # File part
            filename = parts[-1]
            if filename not in current:
                current[filename] = {'_defs': []}
            current[filename]['_defs'].append(d)

        # Build reference edges directly from the ``scope_def_name`` info in references
        ref_edges = []
        for d in definitions:
            target_key = (d.file_path, d.name)
            for ref_file, scope_def_name, _ in d.references:
                if scope_def_name is not None:  # skip module-level references
                    src_key = (ref_file, scope_def_name)
                    ref_edges.append((src_key, target_key))

        # Render the HTML
        html_content = self._generate_html_template(definitions, ref_edges, file_tree)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n✓ Visualization written to: {output_path}")
        print(f"  Definitions: {len(definitions)}, reference edges: {len(ref_edges)}")

    def _generate_html_template(self, definitions: List[DefinitionInfo],
                                 ref_edges: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                                 file_tree: Dict[str, Any]) -> str:
        """Render the HTML template using a file-path hierarchical layout."""

        # Build node and edge JSON payloads
        nodes = []
        edges_data = []
        node_id_map = {}
        node_id = 0

        # First pass: build the node-id mapping
        for d in definitions:
            key = (d.file_path, d.name)
            node_id_map[key] = node_id
            node_id += 1

        # Create edges (source = referrer, target = referenced)
        edge_set: Set[Tuple[int, int]] = set()
        for src_key, target_key in ref_edges:
            if src_key in node_id_map and target_key in node_id_map:
                edge_tuple = (node_id_map[src_key], node_id_map[target_key])
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    edges_data.append({
                        'source': node_id_map[src_key],
                        'target': node_id_map[target_key],
                    })

        # Compute out-degree (references others) and in-degree (referenced by others) per node
        out_degree: Dict[int, int] = defaultdict(int)
        in_degree: Dict[int, int] = defaultdict(int)
        for src_id, tgt_id in edge_set:
            out_degree[src_id] += 1
            in_degree[tgt_id] += 1

        # Create a node entry per definition (color decided by degree)
        for d in definitions:
            key = (d.file_path, d.name)
            nid = node_id_map[key]
            node_in_degree = in_degree.get(nid, 0)

            # Node color — use inDegree to decide whether the node is referenced
            if node_in_degree == 0:
                color = '#e74c3c'  # red: unreferenced
            elif d.def_type == 'class':
                color = '#27ae60'
            else:
                color = '#2980b9'

            nodes.append({
                'id': nid,
                'name': d.name,
                'type': d.def_type,
                'file': d.file_path,
                'outDegree': out_degree.get(nid, 0),
                'inDegree': node_in_degree,
                'color': color,
            })

        # Aggregate statistics from the edge data
        total = len(definitions)
        unreferenced = sum(1 for nid in range(total) if in_degree.get(nid, 0) == 0)
        classes = sum(1 for d in definitions if d.def_type == 'class')
        functions = total - classes

        # Group definitions by file
        files_data: Dict[str, List[dict]] = defaultdict(list)
        for d in definitions:
            nid = node_id_map[(d.file_path, d.name)]
            files_data[d.file_path].append({
                'id': nid,
                'name': d.name,
                'type': d.def_type,
                'outDegree': out_degree.get(nid, 0),
                'inDegree': in_degree.get(nid, 0),
            })

        nodes_json = json.dumps(nodes, ensure_ascii=False)
        edges_json = json.dumps(edges_data, ensure_ascii=False)
        files_json = json.dumps(dict(files_data), ensure_ascii=False)

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Python Reference Graph</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Monaco, monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            display: flex;
            height: 100vh;
        }}

        /* Left-side file tree */
        #file-tree {{
            width: 320px;
            background: #252526;
            border-right: 1px solid #3c3c3c;
            overflow-y: auto;
            padding: 10px 0;
        }}

        .tree-header {{
            padding: 8px 15px;
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            border-bottom: 1px solid #3c3c3c;
            margin-bottom: 5px;
        }}

        .folder {{
            cursor: pointer;
            user-select: none;
        }}

        .folder-header {{
            display: flex;
            align-items: center;
            padding: 3px 8px;
            font-size: 13px;
        }}

        .folder-header:hover {{
            background: #2a2d2e;
        }}

        .folder-icon {{
            width: 16px;
            margin-right: 6px;
            color: #dcb67a;
        }}

        .folder-name {{
            color: #cccccc;
        }}

        .folder-content {{
            padding-left: 16px;
        }}

        .folder.collapsed > .folder-content {{
            display: none;
        }}

        .file {{
            display: flex;
            align-items: center;
            padding: 3px 8px;
            font-size: 13px;
            cursor: pointer;
        }}

        .file:hover {{
            background: #2a2d2e;
        }}

        .file.selected {{
            background: #094771;
        }}

        .file-icon {{
            width: 16px;
            margin-right: 6px;
            color: #519aba;
        }}

        .file-name {{
            color: #cccccc;
            flex: 1;
        }}

        .file-badge {{
            font-size: 10px;
            padding: 1px 5px;
            border-radius: 8px;
            margin-left: 4px;
        }}

        .badge-warn {{
            background: #6c3030;
            color: #f48771;
        }}

        /* Center content area */
        #content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        /* Breadcrumb */
        #breadcrumb {{
            padding: 8px 15px;
            background: #2d2d2d;
            border-bottom: 1px solid #3c3c3c;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .crumb {{
            color: #888;
        }}

        .crumb.current {{
            color: #d4d4d4;
        }}

        .crumb-sep {{
            color: #555;
        }}

        /* Definition list */
        #definitions {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}

        .def-group {{
            margin-bottom: 20px;
        }}

        .def-group-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #3c3c3c;
        }}

        .def-item {{
            display: flex;
            align-items: center;
            padding: 6px 10px;
            margin: 2px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.15s;
        }}

        .def-item:hover {{
            background: #2a2d2e;
        }}

        .def-item.selected {{
            background: #094771;
        }}

        .def-icon {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
            margin-right: 10px;
            color: white;
        }}

        .def-icon.class {{
            background: #27ae60;
        }}

        .def-icon.function {{
            background: #2980b9;
        }}

        .def-icon.unreferenced {{
            background: #e74c3c;
        }}

        .def-name {{
            flex: 1;
            font-size: 13px;
        }}

        .def-refs {{
            font-size: 11px;
            color: #888;
        }}

        /* Right-side detail panel */
        #detail-panel {{
            width: 300px;
            background: #252526;
            border-left: 1px solid #3c3c3c;
            padding: 15px;
            overflow-y: auto;
        }}

        .detail-title {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #3c3c3c;
        }}

        .detail-section {{
            margin-bottom: 15px;
        }}

        .detail-label {{
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 5px;
        }}

        .detail-value {{
            font-size: 13px;
            color: #d4d4d4;
        }}

        .ref-list {{
            max-height: 200px;
            overflow-y: auto;
        }}

        .ref-item {{
            padding: 4px 8px;
            margin: 2px 0;
            background: #2d2d2d;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
        }}

        .ref-item:hover {{
            background: #3c3c3c;
        }}

        /* Status bar */
        #stats-bar {{
            padding: 8px 15px;
            background: #007acc;
            font-size: 12px;
            display: flex;
            gap: 20px;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        /* SVG graph area */
        #graph-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #1e1e1e;
        }}

        #graph-svg {{
            width: 100%;
            height: 100%;
        }}

        .node {{
            cursor: pointer;
        }}

        .node rect, .node ellipse {{
            stroke-width: 2px;
            transition: stroke-width 0.15s;
        }}

        .node:hover rect, .node:hover ellipse {{
            stroke-width: 3px;
        }}

        .node.selected rect, .node.selected ellipse {{
            stroke: #fff;
            stroke-width: 3px;
        }}

        .node text {{
            fill: white;
            font-size: 12px;
            pointer-events: none;
        }}

        .edge {{
            fill: none;
            stroke: #555;
            stroke-width: 1px;
            opacity: 0.6;
        }}

        .edge.highlighted {{
            stroke-width: 1.5px;
            opacity: 1;
        }}

        marker {{
            fill: #555;
        }}

        /* View toggle */
        #view-toggle {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }}

        .view-btn {{
            padding: 6px 12px;
            background: #3c3c3c;
            border: none;
            color: #d4d4d4;
            font-size: 12px;
            cursor: pointer;
            border-radius: 3px;
        }}

        .view-btn.active {{
            background: #007acc;
        }}

        .view-btn:hover {{
            background: #4c4c4c;
        }}

        .view-btn.active:hover {{
            background: #0088e0;
        }}

        /* Legend */
        #legend {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(37, 37, 38, 0.95);
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 10px;
            font-size: 11px;
        }}

        .legend-title {{
            font-size: 10px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
        }}

        .legend-shape {{
            width: 20px;
            height: 14px;
            border-radius: 2px;
        }}

        .legend-shape.ellipse {{
            border-radius: 50%;
        }}

        /* Zoom controls */
        #zoom-controls {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }}

        .zoom-btn {{
            width: 30px;
            height: 30px;
            background: #3c3c3c;
            border: none;
            color: #d4d4d4;
            font-size: 16px;
            cursor: pointer;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .zoom-btn:hover {{
            background: #4c4c4c;
        }}

        #zoom-level {{
            padding: 0 8px;
            background: #2d2d2d;
            border-radius: 3px;
            display: flex;
            align-items: center;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div id="file-tree">
        <div class="tree-header">File Explorer</div>
        <div id="tree-content"></div>
    </div>

    <div id="content">
        <div id="stats-bar">
            <div class="stat-item"><span>Definitions:</span> <strong>{total}</strong></div>
            <div class="stat-item"><span>Classes:</span> <strong>{classes}</strong></div>
            <div class="stat-item"><span>Functions:</span> <strong>{functions}</strong></div>
            <div class="stat-item" style="color:#f48771"><span>Unreferenced:</span> <strong>{unreferenced}</strong></div>
        </div>
        <div id="breadcrumb">
            <span class="crumb current">Select a file to inspect its definitions</span>
        </div>
        <div id="graph-container">
            <svg id="graph-svg">
                <defs>
                    <marker id="arrowhead" viewBox="0 0 10 10" refX="0" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M 10 0 L 0 5 L 10 10 z" />
                    </marker>
                </defs>
            </svg>
            <div id="view-toggle">
                <button class="view-btn active" onclick="setView('hierarchy')">Hierarchy view</button>
                <button class="view-btn" onclick="setView('force')">Force-directed view</button>
            </div>
            <div id="legend">
                <div class="legend-title">Legend</div>
                <div class="legend-item">
                    <div class="legend-shape" style="background:#27ae60;"></div>
                    <span>Class (referenced)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape ellipse" style="background:#2980b9;"></div>
                    <span>Function (referenced)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape" style="background:#e74c3c;"></div>
                    <span>Class (unreferenced)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape ellipse" style="background:#e74c3c;"></div>
                    <span>Function (unreferenced)</span>
                </div>
                <div class="legend-item">
                    <div style="width:20px;height:12px;border:1px solid #007acc;background:rgba(0,122,204,0.15);border-radius:2px;"></div>
                    <span>Current file</span>
                </div>
                <div class="legend-item">
                    <div style="width:20px;height:12px;border:1px dashed #555;background:rgba(60,60,60,0.3);border-radius:2px;"></div>
                    <span>Directory / other file</span>
                </div>
                <div class="legend-item">
                    <div style="width:20px;height:2px;background:#4dd0e1;"></div>
                    <span>Outgoing (references others)</span>
                </div>
                <div class="legend-item">
                    <div style="width:20px;height:2px;background:#ffb74d;"></div>
                    <span>Incoming (referenced by)</span>
                </div>
            </div>
            <div id="zoom-controls">
                <button class="zoom-btn" onclick="zoomGraph(-0.2)">−</button>
                <div id="zoom-level">100%</div>
                <button class="zoom-btn" onclick="zoomGraph(0.2)">+</button>
                <button class="zoom-btn" onclick="resetZoom()" title="Reset">⟲</button>
            </div>
        </div>
    </div>

    <div id="detail-panel">
        <div class="detail-title">Details</div>
        <div id="detail-content">
            <p style="color:#888;font-size:13px;">Click a node to see details</p>
        </div>
    </div>

    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        const fileGroups = {files_json};

        let currentView = 'hierarchy';
        let selectedNode = null;
        let simulation = null;

        // Pan / zoom state
        let scale = 1;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let panStartX = 0;
        let panStartY = 0;
        let draggedNode = null;
        let nodePositions = {{}};
        let nodeWidthsGlobal = {{}};  // Global cache of node widths
        const globalNodeHeight = 24;  // Node height constant
        let currentFilepath = null;   // Currently selected file path

        // Build the file tree
        function buildFileTree() {{
            const tree = {{}};
            Object.keys(fileGroups).sort().forEach(filepath => {{
                const parts = filepath.split('/');
                let current = tree;
                parts.forEach((part, i) => {{
                    if (!current[part]) {{
                        current[part] = i === parts.length - 1 ? {{ _file: filepath, _defs: fileGroups[filepath] }} : {{}};
                    }}
                    current = current[part];
                }});
            }});
            return tree;
        }}

        // Zoom / pan helpers
        function zoomGraph(delta) {{
            const newScale = Math.max(0.2, Math.min(3, scale + delta));
            scale = newScale;
            document.getElementById('zoom-level').textContent = Math.round(scale * 100) + '%';
            applyTransform();
        }}

        function resetZoom() {{
            scale = 1;
            panX = 0;
            panY = 0;
            document.getElementById('zoom-level').textContent = '100%';
            applyTransform();
        }}

        function applyTransform() {{
            const contentGroup = document.getElementById('graph-content');
            if (contentGroup) {{
                contentGroup.setAttribute('transform', `translate(${{panX}},${{panY}}) scale(${{scale}})`);
            }}
        }}

        // Initialize SVG drag-and-zoom event handlers
        function initSvgInteraction() {{
            const svg = document.getElementById('graph-svg');
            let hasMoved = false;  // Track whether the mouse actually moved

            // Mouse-wheel zoom
            svg.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                zoomGraph(delta);
            }});

            // Drag-pan & click-empty-to-deselect
            svg.addEventListener('mousedown', (e) => {{
                if (e.target === svg || e.target.id === 'graph-background') {{
                    isPanning = true;
                    hasMoved = false;
                    panStartX = e.clientX - panX;
                    panStartY = e.clientY - panY;
                    svg.style.cursor = 'grabbing';
                }}
            }});

            svg.addEventListener('mousemove', (e) => {{
                if (isPanning) {{
                    hasMoved = true;  // Mark that movement occurred
                    panX = e.clientX - panStartX;
                    panY = e.clientY - panStartY;
                    applyTransform();
                }} else if (draggedNode) {{
                    // Node drag
                    const rect = svg.getBoundingClientRect();
                    const x = (e.clientX - rect.left - panX) / scale;
                    const y = (e.clientY - rect.top - panY) / scale;
                    nodePositions[draggedNode.id] = {{ x, y }};
                    updateNodePosition(draggedNode.id, x, y);
                }}
            }});

            svg.addEventListener('mouseup', (e) => {{
                // Clear selection only on a true click (no drag movement)
                if ((e.target === svg || e.target.id === 'graph-background') && !hasMoved && !draggedNode) {{
                    clearSelection();
                }}
                isPanning = false;
                draggedNode = null;
                hasMoved = false;
                svg.style.cursor = 'default';
            }});

            svg.addEventListener('mouseleave', () => {{
                isPanning = false;
                draggedNode = null;
                hasMoved = false;
                svg.style.cursor = 'default';
            }});
        }}

        // Clear the selection state
        function clearSelection() {{
            selectedNode = null;
            document.querySelectorAll('.node').forEach(el => el.classList.remove('selected'));
            // Hide all edges
            document.querySelectorAll('.edge').forEach(el => {{
                el.style.display = 'none';
            }});
            // Clear the detail panel
            document.getElementById('detail-content').innerHTML = `
                <p style="color:#888;font-size:13px;">Click a node to see details</p>
            `;
        }}

        function updateNodePosition(nodeId, x, y) {{
            const nodeEl = document.querySelector(`.node[data-node-id="${{nodeId}}"]`);
            if (nodeEl) {{
                nodeEl.setAttribute('transform', `translate(${{x}},${{y}})`);
            }}
            // Update incident edges (compare nodeId as string since dataset returns strings)
            const nodeIdStr = String(nodeId);
            document.querySelectorAll('.edge').forEach(edge => {{
                if (edge.dataset.source === nodeIdStr || edge.dataset.target === nodeIdStr) {{
                    updateEdge(edge);
                }}
            }});
        }}

        function updateEdge(edgeEl) {{
            const sourceId = parseInt(edgeEl.dataset.source, 10);
            const targetId = parseInt(edgeEl.dataset.target, 10);
            const s = nodePositions[sourceId];
            const t = nodePositions[targetId];
            if (s && t) {{
                // Compute the direction vector
                const dx = t.x - s.x;
                const dy = t.y - s.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 1) return;

                // Get node dimensions
                const sWidth = nodeWidthsGlobal[sourceId] || 80;
                const tWidth = nodeWidthsGlobal[targetId] || 80;
                const halfH = globalNodeHeight / 2;

                // Compute the entry point on the rectangle's edge
                function getEdgePoint(cx, cy, halfW, halfH, dirX, dirY) {{
                    if (Math.abs(dirX) < 0.001 && Math.abs(dirY) < 0.001) return {{ x: cx, y: cy }};
                    let t = Infinity;
                    if (dirX > 0) t = Math.min(t, halfW / dirX);
                    else if (dirX < 0) t = Math.min(t, -halfW / dirX);
                    if (dirY > 0) t = Math.min(t, halfH / dirY);
                    else if (dirY < 0) t = Math.min(t, -halfH / dirY);
                    return {{ x: cx + dirX * t, y: cy + dirY * t }};
                }}

                const sEdge = getEdgePoint(s.x, s.y, sWidth / 2, halfH, dx / dist, dy / dist);
                const tEdge = getEdgePoint(t.x, t.y, tWidth / 2, halfH, -dx / dist, -dy / dist);

                if (edgeEl.tagName === 'path') {{
                    // Use a straight line
                    edgeEl.setAttribute('d', `M${{sEdge.x}},${{sEdge.y}} L${{tEdge.x}},${{tEdge.y}}`);
                }} else {{
                    edgeEl.setAttribute('x1', sEdge.x);
                    edgeEl.setAttribute('y1', sEdge.y);
                    edgeEl.setAttribute('x2', tEdge.x);
                    edgeEl.setAttribute('y2', tEdge.y);
                }}
            }}
        }}

        function renderTree(tree, container, path = '') {{
            const sortedKeys = Object.keys(tree).filter(k => !k.startsWith('_')).sort((a, b) => {{
                const aIsFile = tree[a]._file !== undefined;
                const bIsFile = tree[b]._file !== undefined;
                if (aIsFile !== bIsFile) return aIsFile ? 1 : -1;
                return a.localeCompare(b);
            }});

            sortedKeys.forEach(key => {{
                const item = tree[key];
                const fullPath = path ? path + '/' + key : key;

                if (item._file) {{
                    // File — count unreferenced defs (inDegree == 0) for the badge
                    const unreferencedCount = item._defs.filter(d => d.inDegree === 0).length;
                    const fileEl = document.createElement('div');
                    fileEl.className = 'file';
                    fileEl.dataset.filepath = item._file;
                    fileEl.innerHTML = `
                        <span class="file-icon">📄</span>
                        <span class="file-name">${{key}}</span>
                        ${{unreferencedCount > 0 ? `<span class="file-badge badge-warn">${{unreferencedCount}}</span>` : ''}}
                    `;
                    fileEl.onclick = () => selectFile(item._file);
                    container.appendChild(fileEl);
                }} else {{
                    // Folder
                    const folderEl = document.createElement('div');
                    folderEl.className = 'folder';
                    folderEl.innerHTML = `
                        <div class="folder-header">
                            <span class="folder-icon">📁</span>
                            <span class="folder-name">${{key}}</span>
                        </div>
                        <div class="folder-content"></div>
                    `;
                    folderEl.querySelector('.folder-header').onclick = (e) => {{
                        e.stopPropagation();
                        folderEl.classList.toggle('collapsed');
                    }};
                    const content = folderEl.querySelector('.folder-content');
                    renderTree(item, content, fullPath);
                    container.appendChild(folderEl);
                }}
            }});
        }}

        function selectFile(filepath) {{
            // Update the current file path
            currentFilepath = filepath;

            // Update selection state
            document.querySelectorAll('.file').forEach(el => el.classList.remove('selected'));
            document.querySelector(`.file[data-filepath="${{filepath}}"]`)?.classList.add('selected');

            // Update breadcrumb
            const parts = filepath.split('/');
            document.getElementById('breadcrumb').innerHTML = parts.map((p, i) =>
                `<span class="crumb${{i === parts.length - 1 ? ' current' : ''}}">${{p}}</span>` +
                (i < parts.length - 1 ? '<span class="crumb-sep">›</span>' : '')
            ).join('');

            // Clear the previously selected node
            clearSelection();

            // Re-render the graph
            renderGraph(filepath);
        }}

        function renderGraph(filepath = null) {{
            const svg = document.getElementById('graph-svg');
            const container = document.getElementById('graph-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            // Reset transform state (on file switch)
            scale = 1;
            panX = 0;
            panY = 0;
            document.getElementById('zoom-level').textContent = '100%';
            nodePositions = {{}};

            // Clear and add background + content groups
            svg.innerHTML = `
                <defs>
                    <marker id="arrowhead" viewBox="0 0 10 10" refX="0" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M 10 0 L 0 5 L 10 10 z" fill="#555"/>
                    </marker>
                    <marker id="arrowhead-outgoing" viewBox="0 0 10 10" refX="0" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M 10 0 L 0 5 L 10 10 z" fill="#4dd0e1"/>
                    </marker>
                    <marker id="arrowhead-incoming" viewBox="0 0 10 10" refX="0" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M 10 0 L 0 5 L 10 10 z" fill="#ffb74d"/>
                    </marker>
                </defs>
                <rect id="graph-background" width="100%" height="100%" fill="#1e1e1e"/>
                <g id="graph-content"></g>
            `;

            // Initialize interaction handlers
            initSvgInteraction();

            // Filter nodes
            let filteredNodes = filepath ? nodes.filter(n => n.file === filepath) : nodes;
            if (filteredNodes.length === 0) return;

            const nodeIds = new Set(filteredNodes.map(n => n.id));

            // Filter edges: only those incident to the current file
            let filteredEdges = edges.filter(e => nodeIds.has(e.source) || nodeIds.has(e.target));

            // Include referenced nodes that live in other files
            const externalNodeIds = new Set();
            filteredEdges.forEach(e => {{
                if (!nodeIds.has(e.source)) externalNodeIds.add(e.source);
                if (!nodeIds.has(e.target)) externalNodeIds.add(e.target);
            }});

            const externalNodes = nodes.filter(n => externalNodeIds.has(n.id));
            const allNodes = [...filteredNodes, ...externalNodes];

            const contentGroup = document.getElementById('graph-content');
            if (currentView === 'hierarchy') {{
                renderHierarchy(contentGroup, allNodes, filteredEdges, filteredNodes, width, height);
            }} else {{
                renderForce(contentGroup, allNodes, filteredEdges, filteredNodes, width, height);
            }}
        }}

        function renderHierarchy(contentGroup, allNodes, edges, primaryNodes, width, height) {{
            const primaryIds = new Set(primaryNodes.map(n => n.id));
            const nodeHeight = 24;
            const hGap = 50;             // Horizontal gap between sibling nodes
            const vGap = 55;             // Vertical gap between hierarchy levels
            const nodeVGap = 18;         // Vertical gap between nodes inside a file
            const charWidth = 7;         // Estimated character width
            const minNodeWidth = 60;
            const nodePadding = 20;      // Horizontal padding inside a node

            // Compute text width
            function calcTextWidth(text) {{
                return Math.max(minNodeWidth, text.length * charWidth + nodePadding);
            }}

            // Group nodes by file path
            const fileToNodes = {{}};
            allNodes.forEach(n => {{
                if (!fileToNodes[n.file]) fileToNodes[n.file] = [];
                fileToNodes[n.file].push(n);
            }});

            // Pre-compute width of each node
            const nodeWidths = {{}};
            allNodes.forEach(n => {{
                nodeWidths[n.id] = calcTextWidth(n.name);
            }});

            // Build the directory-tree structure
            function buildDirTree(files) {{
                const tree = {{ _children: {{}} }};
                files.forEach(filepath => {{
                    const parts = filepath.split('/');
                    let current = tree._children;
                    parts.forEach((part, i) => {{
                        if (!current[part]) {{
                            current[part] = {{ _children: {{}} }};
                        }}
                        if (i === parts.length - 1) {{
                            current[part]._file = filepath;
                        }}
                        current = current[part]._children;
                    }});
                }});
                return tree;
            }}

            const dirTree = buildDirTree(Object.keys(fileToNodes));

            // Pass 1: compute subtree width of each node
            function calcWidth(node, isRootLevel = false) {{
                const children = Object.keys(node._children || {{}});
                if (children.length === 0) {{
                    // Leaf node (file): width = max width of its definition nodes
                    const defs = node._file ? (fileToNodes[node._file] || []) : [];
                    let maxDefWidth = 80;
                    defs.forEach(n => {{
                        maxDefWidth = Math.max(maxDefWidth, nodeWidths[n.id]);
                    }});
                    node._width = maxDefWidth;
                    node._defCount = defs.length;
                    // Total height of all defs inside this file
                    node._height = defs.length > 0 ? 28 + defs.length * (nodeHeight + nodeVGap) : 30;
                    return node._width;
                }}

                // First, recursively compute widths of all children
                children.forEach(key => {{
                    calcWidth(node._children[key], false);
                }});

                // At the root, split out entry files
                if (isRootLevel) {{
                    const entryFiles = [];
                    const others = [];
                    children.forEach(key => {{
                        const child = node._children[key];
                        const childIsFile = !!child._file;
                        const isEntry = childIsFile && (key === 'main.py' || key.startsWith('__'));
                        if (isEntry) {{
                            entryFiles.push(key);
                        }} else {{
                            others.push(key);
                        }}
                    }});

                    if (entryFiles.length > 0) {{
                        // Width of the entry-file row
                        let entryWidth = 0;
                        entryFiles.forEach((key, i) => {{
                            entryWidth += node._children[key]._width;
                            if (i < entryFiles.length - 1) entryWidth += hGap;
                        }});

                        // Width of the "other nodes" row
                        let othersWidth = 0;
                        others.forEach((key, i) => {{
                            othersWidth += node._children[key]._width;
                            if (i < others.length - 1) othersWidth += hGap;
                        }});

                        // Total width = wider of the two rows
                        node._width = Math.max(entryWidth, othersWidth, 80);
                        return node._width;
                    }}
                }}

                // Common case: all children laid out on a single row
                let totalWidth = 0;
                children.forEach((key, i) => {{
                    totalWidth += node._children[key]._width;
                    if (i < children.length - 1) totalWidth += hGap;
                }});
                node._width = Math.max(totalWidth, 80);
                return node._width;
            }}

            calcWidth(dirTree, true);

            // Pass 2: compute positions
            const positions = {{}};
            const treeNodes = [];  // Tree nodes to draw
            const treeEdges = [];  // Tree edges to draw

            function layoutNode(node, name, x, y, parentPos, isTopLevel = false) {{
                const isFile = !!node._file;
                const isRoot = !name;

                if (!isRoot) {{
                    const nodeInfo = {{
                        name: name,
                        x: x,
                        y: y,
                        isFile: isFile,
                        filepath: node._file,
                        isPrimary: isFile && primaryNodes.some(n => n.file === node._file)
                    }};
                    treeNodes.push(nodeInfo);

                    if (parentPos) {{
                        treeEdges.push({{ from: parentPos, to: {{ x, y }} }});
                    }}
                }}

                const children = Object.keys(node._children || {{}}).sort((a, b) => {{
                    const aIsFile = !!node._children[a]._file;
                    const bIsFile = !!node._children[b]._file;
                    // Directories first, files after (alphabetical within each group)
                    if (aIsFile !== bIsFile) return aIsFile ? 1 : -1;
                    return a.localeCompare(b);
                }});

                if (children.length > 0) {{
                    // Only split out entry files at the top level (direct children of dirTree)
                    if (isRoot) {{
                        const entryFiles = [];
                        const others = [];
                        children.forEach(key => {{
                            const child = node._children[key];
                            const childIsFile = !!child._file;
                            const isEntry = childIsFile && (key === 'main.py' || key.startsWith('__'));
                            if (isEntry) {{
                                entryFiles.push(key);
                            }} else {{
                                others.push(key);
                            }}
                        }});

                        // If there are entry files, lay them out in a row above
                        if (entryFiles.length > 0) {{
                            let entryTotalWidth = 0;
                            entryFiles.forEach((key, i) => {{
                                entryTotalWidth += node._children[key]._width;
                                if (i < entryFiles.length - 1) entryTotalWidth += hGap;
                            }});
                            let entryX = x - entryTotalWidth / 2;
                            const entryY = y + vGap;

                            // Max height of the entry-file row (including its definition nodes)
                            let maxEntryHeight = 30;
                            entryFiles.forEach(key => {{
                                const child = node._children[key];
                                const defs = child._file ? (fileToNodes[child._file] || []) : [];
                                const childHeight = 28 + defs.length * (nodeHeight + nodeVGap) + 20;
                                maxEntryHeight = Math.max(maxEntryHeight, childHeight);
                            }});

                            entryFiles.forEach(key => {{
                                const child = node._children[key];
                                const childCenterX = entryX + child._width / 2;
                                layoutNode(child, key, childCenterX, entryY, null, true);
                                entryX += child._width + hGap;
                            }});

                            // Other nodes go below, after the entry-file row height
                            if (others.length > 0) {{
                                let othersTotalWidth = 0;
                                others.forEach((key, i) => {{
                                    othersTotalWidth += node._children[key]._width;
                                    if (i < others.length - 1) othersTotalWidth += hGap;
                                }});
                                let otherX = x - othersTotalWidth / 2;
                                const otherY = entryY + maxEntryHeight;

                                others.forEach(key => {{
                                    const child = node._children[key];
                                    const childCenterX = otherX + child._width / 2;
                                    layoutNode(child, key, childCenterX, otherY, null, true);
                                    otherX += child._width + hGap;
                                }});
                            }}
                            return;  // Root node already handled
                        }}
                    }}

                    // Standard horizontal layout (non-root, or root without entry files)
                    let totalChildWidth = 0;
                    children.forEach((key, i) => {{
                        totalChildWidth += node._children[key]._width;
                        if (i < children.length - 1) totalChildWidth += hGap;
                    }});

                    let childX = x - totalChildWidth / 2;
                    const childY = y + vGap;

                    children.forEach(key => {{
                        const child = node._children[key];
                        const childCenterX = childX + child._width / 2;
                        layoutNode(child, key, childCenterX, childY, isRoot ? null : {{ x, y }}, false);
                        childX += child._width + hGap;
                    }});
                }}

                // For a file node, lay out its definition nodes
                if (isFile && node._file) {{
                    const defs = fileToNodes[node._file] || [];
                    defs.forEach((n, i) => {{
                        positions[n.id] = {{
                            x: x,
                            y: y + 28 + i * (nodeHeight + nodeVGap)
                        }};
                    }});
                }}
            }}

            // Start layout from the root node
            const rootX = width / 2;
            const rootY = 60;
            layoutNode(dirTree, '', rootX, rootY, null);

            // Persist to the global cache
            Object.assign(nodePositions, positions);
            Object.assign(nodeWidthsGlobal, nodeWidths);

            // Draw tree edges (dimmed)
            const treeEdgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            treeEdges.forEach(e => {{
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', e.from.x);
                line.setAttribute('y1', e.from.y + 10);
                line.setAttribute('x2', e.to.x);
                line.setAttribute('y2', e.to.y - 10);
                line.setAttribute('stroke', '#444');
                line.setAttribute('stroke-width', '1');
                line.setAttribute('stroke-dasharray', '3,2');
                line.setAttribute('opacity', '0.5');
                treeEdgesGroup.appendChild(line);
            }});
            contentGroup.appendChild(treeEdgesGroup);

            // Draw tree nodes (directories and files)
            const treeNodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            treeNodes.forEach(n => {{
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('transform', `translate(${{n.x}},${{n.y}})`);

                const labelText = (n.isFile ? '📄 ' : '📁 ') + n.name;
                const boxWidth = calcTextWidth(labelText);
                const boxHeight = 20;

                if (n.isFile) {{
                    // File node
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', -boxWidth/2);
                    rect.setAttribute('y', -boxHeight/2);
                    rect.setAttribute('width', boxWidth);
                    rect.setAttribute('height', boxHeight);
                    rect.setAttribute('rx', 3);
                    rect.setAttribute('fill', n.isPrimary ? 'rgba(0, 122, 204, 0.3)' : 'rgba(60, 60, 60, 0.4)');
                    rect.setAttribute('stroke', n.isPrimary ? '#007acc' : '#555');
                    rect.setAttribute('stroke-width', n.isPrimary ? '2' : '1');
                    if (!n.isPrimary) rect.setAttribute('stroke-dasharray', '2,1');
                    g.appendChild(rect);

                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('dy', '0.35em');
                    text.setAttribute('fill', n.isPrimary ? '#4fc3f7' : '#888');
                    text.setAttribute('font-size', '11');
                    text.textContent = labelText;
                    g.appendChild(text);

                    // Clicking a file node switches to that file view
                    if (!n.isPrimary && n.filepath) {{
                        g.style.cursor = 'pointer';
                        g.onclick = (e) => {{
                            e.stopPropagation();
                            selectFile(n.filepath);
                        }};
                    }}
                }} else {{
                    // Directory node
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', -boxWidth/2);
                    rect.setAttribute('y', -boxHeight/2);
                    rect.setAttribute('width', boxWidth);
                    rect.setAttribute('height', boxHeight);
                    rect.setAttribute('rx', 3);
                    rect.setAttribute('fill', 'rgba(80, 80, 80, 0.3)');
                    rect.setAttribute('stroke', '#666');
                    rect.setAttribute('stroke-width', '1');
                    rect.setAttribute('stroke-dasharray', '3,1');
                    g.appendChild(rect);

                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('dy', '0.35em');
                    text.setAttribute('fill', '#aaa');
                    text.setAttribute('font-size', '11');
                    text.textContent = labelText;
                    g.appendChild(text);
                }}

                treeNodesGroup.appendChild(g);
            }});
            contentGroup.appendChild(treeNodesGroup);

            // Draw all reference edges (hidden by default; revealed when a node is clicked)
            const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            edges.forEach(e => {{
                // Both endpoints must be positioned
                const sPos = positions[e.source];
                const tPos = positions[e.target];
                if (!sPos || !tPos) return;

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

                // Compute the direction vector
                const dx = tPos.x - sPos.x;
                const dy = tPos.y - sPos.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 1) return;

                // Get node dimensions
                const sWidth = nodeWidths[e.source] || 80;
                const tWidth = nodeWidths[e.target] || 80;
                const halfH = nodeHeight / 2;

                // Compute the entry point on the rectangle/ellipse edge
                // Use ray-vs-rectangle intersection (treat shape as rectangle)
                function getEdgePoint(cx, cy, halfW, halfH, dirX, dirY) {{
                    if (Math.abs(dirX) < 0.001 && Math.abs(dirY) < 0.001) return {{ x: cx, y: cy }};
                    // Find intersection with the four edges; pick the closest
                    const tRight = halfW / Math.abs(dirX);
                    const tLeft = halfW / Math.abs(dirX);
                    const tTop = halfH / Math.abs(dirY);
                    const tBottom = halfH / Math.abs(dirY);

                    let t = Infinity;
                    if (dirX > 0) t = Math.min(t, halfW / dirX);
                    else if (dirX < 0) t = Math.min(t, -halfW / dirX);
                    if (dirY > 0) t = Math.min(t, halfH / dirY);
                    else if (dirY < 0) t = Math.min(t, -halfH / dirY);

                    return {{ x: cx + dirX * t, y: cy + dirY * t }};
                }}

                const sEdge = getEdgePoint(sPos.x, sPos.y, sWidth / 2, halfH, dx / dist, dy / dist);
                const tEdge = getEdgePoint(tPos.x, tPos.y, tWidth / 2, halfH, -dx / dist, -dy / dist);

                // Use a straight line
                path.setAttribute('d', `M${{sEdge.x}},${{sEdge.y}} L${{tEdge.x}},${{tEdge.y}}`);
                path.setAttribute('class', 'edge');
                path.setAttribute('stroke', '#555');
                path.setAttribute('stroke-width', '1');
                path.setAttribute('fill', 'none');
                path.setAttribute('marker-start', 'url(#arrowhead)');
                path.dataset.source = e.source;
                path.dataset.target = e.target;
                path.style.display = 'none';  // Hidden by default
                edgesGroup.appendChild(path);
            }});
            contentGroup.appendChild(edgesGroup);

            // Draw the definition nodes
            const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            allNodes.forEach(n => {{
                if (!positions[n.id]) return;
                const pos = positions[n.id];
                const isPrimary = primaryIds.has(n.id);
                const nWidth = nodeWidths[n.id];

                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('class', 'node');
                g.setAttribute('transform', `translate(${{pos.x}},${{pos.y}})`);
                g.dataset.nodeId = n.id;
                g.style.cursor = 'grab';

                if (n.type === 'class') {{
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', -nWidth/2);
                    rect.setAttribute('y', -nodeHeight/2);
                    rect.setAttribute('width', nWidth);
                    rect.setAttribute('height', nodeHeight);
                    rect.setAttribute('rx', 3);
                    rect.setAttribute('fill', n.color);
                    rect.setAttribute('stroke', isPrimary ? '#fff' : '#555');
                    rect.setAttribute('stroke-width', isPrimary ? '2' : '1');
                    rect.setAttribute('opacity', isPrimary ? 1 : 0.6);
                    g.appendChild(rect);
                }} else {{
                    const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
                    ellipse.setAttribute('rx', nWidth/2);
                    ellipse.setAttribute('ry', nodeHeight/2);
                    ellipse.setAttribute('fill', n.color);
                    ellipse.setAttribute('stroke', isPrimary ? '#fff' : '#555');
                    ellipse.setAttribute('stroke-width', isPrimary ? '2' : '1');
                    ellipse.setAttribute('opacity', isPrimary ? 1 : 0.6);
                    g.appendChild(ellipse);
                }}

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('dy', '0.35em');
                text.setAttribute('font-size', '10');
                text.setAttribute('opacity', isPrimary ? 1 : 0.7);
                text.textContent = n.name;
                g.appendChild(text);

                g.onclick = (e) => {{
                    if (!draggedNode) selectNode(n);
                }};
                g.onmousedown = (e) => {{
                    e.stopPropagation();
                    draggedNode = n;
                    g.style.cursor = 'grabbing';
                }};

                nodesGroup.appendChild(g);
            }});
            contentGroup.appendChild(nodesGroup);
        }}

        function renderForce(contentGroup, allNodes, edges, primaryNodes, width, height) {{
            // Simple force-directed layout
            const primaryIds = new Set(primaryNodes.map(n => n.id));
            const charWidth = 7;
            const minNodeWidth = 60;
            const nodePadding = 20;
            const nodeHeight = 26;

            // Compute text width
            function calcTextWidth(text) {{
                return Math.max(minNodeWidth, text.length * charWidth + nodePadding);
            }}

            // Pre-compute width of each node
            const nodeWidths = {{}};
            allNodes.forEach(n => {{
                nodeWidths[n.id] = calcTextWidth(n.name);
            }});

            // Initial positions
            const positions = {{}};
            allNodes.forEach((n, i) => {{
                const angle = (2 * Math.PI * i) / allNodes.length;
                const radius = Math.min(width, height) * 0.35;
                positions[n.id] = {{
                    x: width/2 + radius * Math.cos(angle),
                    y: height/2 + radius * Math.sin(angle)
                }};
            }});

            // Simple force simulation (a few iterations)
            for (let iter = 0; iter < 50; iter++) {{
                // Repulsion
                allNodes.forEach(n1 => {{
                    allNodes.forEach(n2 => {{
                        if (n1.id >= n2.id) return;
                        const p1 = positions[n1.id];
                        const p2 = positions[n2.id];
                        const dx = p2.x - p1.x;
                        const dy = p2.y - p1.y;
                        const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1);
                        const force = 5000 / (dist * dist);
                        const fx = dx / dist * force;
                        const fy = dy / dist * force;
                        p1.x -= fx; p1.y -= fy;
                        p2.x += fx; p2.y += fy;
                    }});
                }});

                // Attraction along edges
                edges.forEach(e => {{
                    if (!positions[e.source] || !positions[e.target]) return;
                    const p1 = positions[e.source];
                    const p2 = positions[e.target];
                    const dx = p2.x - p1.x;
                    const dy = p2.y - p1.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 1) return;
                    const force = dist * 0.01;
                    const fx = dx / dist * force;
                    const fy = dy / dist * force;
                    p1.x += fx; p1.y += fy;
                    p2.x -= fx; p2.y -= fy;
                }});

                // Centering force
                allNodes.forEach(n => {{
                    const p = positions[n.id];
                    p.x += (width/2 - p.x) * 0.01;
                    p.y += (height/2 - p.y) * 0.01;
                }});
            }}

            // Persist to the global cache
            Object.assign(nodePositions, positions);
            Object.assign(nodeWidthsGlobal, nodeWidths);

            // Draw all edges (hidden by default)
            const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            edges.forEach(e => {{
                const sPos = positions[e.source];
                const tPos = positions[e.target];
                if (!sPos || !tPos) return;

                // Compute the direction vector
                const dx = tPos.x - sPos.x;
                const dy = tPos.y - sPos.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 1) return;

                // Get node dimensions
                const sWidth = nodeWidths[e.source] || 80;
                const tWidth = nodeWidths[e.target] || 80;
                const halfH = nodeHeight / 2;

                // Compute the entry point on the rectangle/ellipse edge
                function getEdgePoint(cx, cy, halfW, halfH, dirX, dirY) {{
                    if (Math.abs(dirX) < 0.001 && Math.abs(dirY) < 0.001) return {{ x: cx, y: cy }};
                    let t = Infinity;
                    if (dirX > 0) t = Math.min(t, halfW / dirX);
                    else if (dirX < 0) t = Math.min(t, -halfW / dirX);
                    if (dirY > 0) t = Math.min(t, halfH / dirY);
                    else if (dirY < 0) t = Math.min(t, -halfH / dirY);
                    return {{ x: cx + dirX * t, y: cy + dirY * t }};
                }}

                const sEdge = getEdgePoint(sPos.x, sPos.y, sWidth / 2, halfH, dx / dist, dy / dist);
                const tEdge = getEdgePoint(tPos.x, tPos.y, tWidth / 2, halfH, -dx / dist, -dy / dist);

                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', sEdge.x);
                line.setAttribute('y1', sEdge.y);
                line.setAttribute('x2', tEdge.x);
                line.setAttribute('y2', tEdge.y);
                line.setAttribute('class', 'edge');
                line.setAttribute('stroke', '#555');
                line.setAttribute('stroke-width', '1');
                line.setAttribute('marker-start', 'url(#arrowhead)');
                line.dataset.source = e.source;
                line.dataset.target = e.target;
                line.style.display = 'none';  // Hidden by default
                edgesGroup.appendChild(line);
            }});
            contentGroup.appendChild(edgesGroup);

            const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            allNodes.forEach(n => {{
                const pos = positions[n.id];
                const isPrimary = primaryIds.has(n.id);
                const nWidth = nodeWidths[n.id];

                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('class', 'node');
                g.setAttribute('transform', `translate(${{pos.x}},${{pos.y}})`);
                g.dataset.nodeId = n.id;
                g.style.cursor = 'grab';

                if (n.type === 'class') {{
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', -nWidth/2);
                    rect.setAttribute('y', -nodeHeight/2);
                    rect.setAttribute('width', nWidth);
                    rect.setAttribute('height', nodeHeight);
                    rect.setAttribute('rx', 4);
                    rect.setAttribute('fill', n.color);
                    rect.setAttribute('stroke', isPrimary ? '#fff' : '#555');
                    rect.setAttribute('opacity', isPrimary ? 1 : 0.7);
                    g.appendChild(rect);
                }} else {{
                    const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
                    ellipse.setAttribute('rx', nWidth/2);
                    ellipse.setAttribute('ry', nodeHeight/2);
                    ellipse.setAttribute('fill', n.color);
                    ellipse.setAttribute('stroke', isPrimary ? '#fff' : '#555');
                    ellipse.setAttribute('opacity', isPrimary ? 1 : 0.7);
                    g.appendChild(ellipse);
                }}

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('dy', '0.35em');
                text.setAttribute('font-size', '10');
                text.textContent = n.name;
                g.appendChild(text);

                // Click selects a node
                g.onclick = (e) => {{
                    if (!draggedNode) selectNode(n);
                }};
                // Drag start
                g.onmousedown = (e) => {{
                    e.stopPropagation();
                    draggedNode = n;
                    g.style.cursor = 'grabbing';
                }};

                nodesGroup.appendChild(g);
            }});
            contentGroup.appendChild(nodesGroup);
        }}

        function selectNode(node) {{
            // If a file is currently selected, restrict selection to nodes belonging to that file
            if (currentFilepath && node.file !== currentFilepath) {{
                return;
            }}

            selectedNode = node;

            // Update selection state
            document.querySelectorAll('.node').forEach(el => el.classList.remove('selected'));
            document.querySelector(`.node[data-node-id="${{node.id}}"]`)?.classList.add('selected');

            // Show only edges incident to the selected node; hide the rest
            // Outgoing edges (from selected node) in blue; incoming edges (to selected node) in orange
            document.querySelectorAll('.edge').forEach(el => {{
                const sourceId = parseInt(el.dataset.source, 10);
                const targetId = parseInt(el.dataset.target, 10);
                const isOutgoing = sourceId === node.id;  // Outgoing from the selected node
                const isIncoming = targetId === node.id;  // Incoming to the selected node
                const isRelated = isOutgoing || isIncoming;

                el.style.display = isRelated ? '' : 'none';
                if (isRelated) {{
                    el.classList.add('highlighted');
                    if (isOutgoing) {{
                        el.style.stroke = '#4dd0e1';  // Bright cyan: outgoing (references others)
                        el.setAttribute('marker-start', 'url(#arrowhead-outgoing)');
                    }} else {{
                        el.style.stroke = '#ffb74d';  // Warm orange: incoming (referenced by)
                        el.setAttribute('marker-start', 'url(#arrowhead-incoming)');
                    }}
                }}
            }});

            // Update the detail panel
            const callers = edges.filter(e => e.target === node.id).map(e => nodes.find(n => n.id === e.source));
            const callees = edges.filter(e => e.source === node.id).map(e => nodes.find(n => n.id === e.target));

            document.getElementById('detail-content').innerHTML = `
                <div class="detail-section">
                    <div class="detail-label">Name</div>
                    <div class="detail-value">${{node.name}}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-label">Type</div>
                    <div class="detail-value">${{node.type === 'class' ? 'class' : 'function'}}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-label">File</div>
                    <div class="detail-value" style="font-size:11px;word-break:break-all;">${{node.file}}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-label">References</div>
                    <div class="detail-value">References: ${{node.outDegree}}, referenced by: ${{node.inDegree}}</div>
                </div>
                ${{callers.length > 0 ? `
                <div class="detail-section">
                    <div class="detail-label">Referenced by (${{callers.length}})</div>
                    <div class="ref-list">
                        ${{callers.map(c => `<div class="ref-item" onclick="selectFile('${{c.file}}')">${{c.name}}<br><small style="color:#888">${{c.file}}</small></div>`).join('')}}
                    </div>
                </div>
                ` : ''}}
                ${{callees.length > 0 ? `
                <div class="detail-section">
                    <div class="detail-label">References (${{callees.length}})</div>
                    <div class="ref-list">
                        ${{callees.map(c => `<div class="ref-item" onclick="selectFile('${{c.file}}')">${{c.name}}<br><small style="color:#888">${{c.file}}</small></div>`).join('')}}
                    </div>
                </div>
                ` : ''}}
            `;
        }}

        function setView(view) {{
            currentView = view;
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent.includes(view === 'hierarchy' ? 'Hierarchy' : 'Force-directed'));
            }});

            const selectedFile = document.querySelector('.file.selected')?.dataset.filepath;
            renderGraph(selectedFile);
        }}

        // Initialize
        const tree = buildFileTree();
        renderTree(tree, document.getElementById('tree-content'));

        // Select the first file by default
        const firstFile = document.querySelector('.file');
        if (firstFile) {{
            firstFile.click();
        }} else {{
            renderGraph();
        }}

        // Redraw on window resize
        window.addEventListener('resize', () => {{
            const selectedFile = document.querySelector('.file.selected')?.dataset.filepath;
            renderGraph(selectedFile);
        }});
    </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check whether every class and module-level function definition in a directory of Python files is actually referenced from somewhere else.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s /path/to/project
  %(prog)s /path/to/project --sort ref_count --reverse
  %(prog)s /path/to/project --unreferenced-only
  %(prog)s /path/to/project --html output.html
  %(prog)s /path/to/project --ignore-dirs tests,examples
  %(prog)s /path/to/project --no-ignore"""
    )
    parser.add_argument('directory', help='Path of the directory to check')
    parser.add_argument(
        '--sort', dest='sort_by', default='file',
        choices=VALID_SORT_KEYS,
        help='Sort key (default: file)'
    )
    parser.add_argument(
        '--reverse', action='store_true',
        help='Reverse the sort order'
    )
    parser.add_argument(
        '--unreferenced-only', action='store_true',
        help='Only show unreferenced definitions'
    )
    parser.add_argument(
        '--html', dest='html_output', default=None,
        help='Write an HTML visualization of the reference graph to this file'
    )

    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument(
        '--ignore-dirs', default=None,
        help=f'Comma-separated list of directories to ignore (default: {",".join(sorted(DEFAULT_IGNORE_DIRS))})'
    )
    ignore_group.add_argument(
        '--no-ignore', action='store_true',
        help='Do not ignore any directory'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    target_dir = args.directory
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    if args.no_ignore:
        ignore_dirs: Set[str] = set()
    elif args.ignore_dirs is not None:
        ignore_dirs = set(d.strip() for d in args.ignore_dirs.split(',') if d.strip())
    else:
        ignore_dirs = DEFAULT_IGNORE_DIRS.copy()

    show_all = not args.unreferenced_only

    print(f"Analyzing directory: {target_dir}")
    print(f"Ignored directories: {ignore_dirs if ignore_dirs else 'none'}")
    print(f"Sort key: {args.sort_by}{' (reversed)' if args.reverse else ''}")
    print("-" * 100)

    checker = ReferenceChecker(target_dir, ignore_dirs=ignore_dirs)
    definitions = checker.analyze()
    checker.print_report(definitions, show_all=show_all, sort_by=args.sort_by, reverse=args.reverse)

    # Generate the HTML visualization
    if args.html_output:
        checker.generate_html_graph(definitions, args.html_output)


if __name__ == '__main__':
    main()
