from collections import defaultdict
from typing import Dict, List, Optional
from networkx import MultiGraph
from zerorepo.rpg_gen.base import RPG, Node
from zerorepo.rpg_gen.base.rpg import EdgeType, NodeType
from zerorepo.utils.repo import is_test_file, wrap_code_snippet

class RepoEntitySearcher:
    """
    Search entities in repository using RPG and its dependency graph.

    The RPG now contains:
    - dep_graph: DependencyGraph with the underlying networkx graph (dep_graph.G)
    - _dep_to_rpg_map: Mapping from dep graph nodes to RPG nodes
    """

    def __init__(self, rpg: RPG):
        """
        Initialize searcher with RPG.

        Args:
            rpg: RPG instance containing dep_graph and dep_to_rpg mapping
        """
        self.rpg = rpg
        # Get dep graph from RPG
        self.G = rpg.dep_graph.G if rpg.dep_graph else None
        # Get dep2rpg mapping from RPG
        self.dep2rpg = rpg._dep_to_rpg_map or {}

        self._global_name_dict = None
        self._global_name_dict_lowercase = None
        self._etypes_dict = {
            etype: i for i, etype in enumerate(EdgeType)
        }

    @classmethod
    def from_components(cls, dep_graph: MultiGraph, rpg: RPG, dep2rpg: Dict[str, List[str]]) -> "RepoEntitySearcher":
        """
        Create searcher from individual components (for backward compatibility).

        Args:
            dep_graph: NetworkX graph of dependencies
            rpg: RPG instance
            dep2rpg: Mapping from dep graph nodes to RPG nodes

        Returns:
            RepoEntitySearcher instance
        """
        # Set the components on RPG if not already set
        if rpg.dep_graph is None:
            from zerorepo.rpg_gen.base.rpg.dep_graph import DependencyGraph
            rpg.dep_graph = DependencyGraph.__new__(DependencyGraph)
            rpg.dep_graph.G = dep_graph
            rpg.dep_graph.repo_dir = ""
        if not rpg._dep_to_rpg_map:
            rpg._dep_to_rpg_map = dep2rpg

        return cls(rpg)
    
    @property
    def global_name_dict(self):
        if self._global_name_dict is None:  # Compute only once
            _global_name_dict = defaultdict(list)
            if self.G is not None:
                for nid in self.G.nodes():
                    if nid.endswith('.py'):
                        fname = nid.split('/')[-1]
                        _global_name_dict[fname].append(nid)

                        name = nid[:-(len('.py'))].split('/')[-1]
                        _global_name_dict[name].append(nid)

                    elif ':' in nid:
                        name = nid.split(':')[-1].split('.')[-1]
                        _global_name_dict[name].append(nid)

            self._global_name_dict = _global_name_dict
        return self._global_name_dict

    @property
    def global_name_dict_lowercase(self):
        if self._global_name_dict_lowercase is None:  # Compute only once
            _global_name_dict_lowercase = defaultdict(list)
            if self.G is not None:
                for nid in self.G.nodes():
                    if nid.endswith('.py'):
                        fname = nid.split('/')[-1].lower()
                        _global_name_dict_lowercase[fname].append(nid)

                        name = nid[:-(len('.py'))].split('/')[-1].lower()
                        _global_name_dict_lowercase[name].append(nid)

                    elif ':' in nid:
                        name = nid.split(':')[-1].split('.')[-1].lower()
                        _global_name_dict_lowercase[name].append(nid)

            self._global_name_dict_lowercase = _global_name_dict_lowercase
        return self._global_name_dict_lowercase

    def has_node(self, nid, include_test=False):
        if self.G is None:
            return False
        if not include_test and is_test_file(nid):
            return False
        return nid in self.G


    def get_feature_paths_for_node(self, nid: str) -> List[str]:
        """
        Retrieve the functional path of RPG nodes associated with the specified node ID.
        """
        rpg_nodes_ids: Optional[List[str]] = self.dep2rpg.get(nid, [])
        feature_paths = []

        for rpg_node_id in rpg_nodes_ids:
            rpg_node: Optional[Node] = self.rpg.get_node_by_id(rpg_node_id)
            if not rpg_node:
                continue  
            if rpg_node.level == 1:
                continue
            feature_path = rpg_node.feature_path()
            feature_paths.append(feature_path)

        feature_paths = [path for path in feature_paths if path]
        return feature_paths
    

    def get_node_data(self, nids, return_code_content=False, wrap_with_ln=True):
        if self.G is None:
            return []
        rtn = []
        for nid in nids:
            if nid not in self.G.nodes:
                continue
            node_data = self.G.nodes[nid]

            feature_paths = self.get_feature_paths_for_node(nid)
    
            formatted_data = {
                'node_id': nid,
                'type': node_data['type'],
                'feature_paths': feature_paths
            }
            if node_data.get('code', ""):
                if 'start_line' in node_data:
                    formatted_data['start_line'] = node_data['start_line']
                    start_line = node_data['start_line']
                elif formatted_data['type'] == NodeType.FILE:
                    start_line = 1
                    formatted_data['start_line'] = start_line
                else:
                    start_line = 1
                    formatted_data['start_line'] = start_line

                if 'end_line' in node_data:
                    formatted_data['end_line'] = node_data['end_line']
                    end_line = node_data['end_line']
                elif formatted_data['type'] == NodeType.FILE:
                    end_line = len(node_data['code'].split("\n")) # - 1
                    formatted_data['end_line'] = end_line
                else:
                    end_line = 1
                    formatted_data['end_line'] = end_line
                # load formatted code data
                if return_code_content and wrap_with_ln:
                    formatted_data['code_content'] = wrap_code_snippet(
                        node_data['code'], start_line, end_line)
                elif return_code_content:
                    formatted_data['code_content'] = node_data['code']
            rtn.append(formatted_data)
        return rtn
    

    def get_all_nodes_by_type(self, type: NodeType):
        if self.G is None:
            return []
        nodes = []
        for nid in self.G.nodes():
            if is_test_file(nid): continue

            if self.G.nodes[nid]['type'] == type:
                node_data = self.G.nodes[nid]
                
                if type == NodeType.FILE:
                    formatted_data = {
                        'name': nid,
                        'type': node_data['type'],
                        'content': node_data.get('code', '').split('\n')
                    }
                elif type == NodeType.METHOD or type == NodeType.FUNCTION:
                    formatted_data = {
                        'name': nid.split(':')[-1],
                        'file': nid.split(':')[0],
                        'type': node_data['type'],
                        'content': node_data.get('code', '').split('\n'),
                        'start_line': node_data.get('start_line', 0),
                        'end_line': node_data.get('end_line', 0)
                    }
                elif type == NodeType.CLASS:
                    formatted_data = {
                        'name': nid.split(':')[-1],
                        'file': nid.split(':')[0],
                        'type': node_data['type'],
                        'content': node_data.get('code', '').split('\n'),
                        'start_line': node_data.get('start_line', 0),
                        'end_line': node_data.get('end_line', 0),
                        'methods': []
                    }
                    dp_searcher = RepoDependencySearcher(self.G)
                    methods = dp_searcher.get_neighbors(nid, 'forward', 
                                                        ntype_filter=[NodeType.METHOD],
                                                        etype_filter=[EdgeType.CONTAINS])[0]
                    formatted_methods = []
                    for mid in methods:
                        mnode = self.G.nodes[mid]
                        
                        m_feature_paths = self.get_all_nodes_by_type(mnode)
                    
                        formatted_methods.append({
                            'name': mid.split('.')[-1],
                            'feature_paths': m_feature_paths,
                            'start_line': mnode.get('start_line', 0),
                            'end_line': mnode.get('end_line', 0),
                        })
                        
                    formatted_data['methods'] = formatted_methods
                    
                feature_paths = self.get_feature_paths_for_node(nid)
                formatted_data["feature_paths"] = feature_paths
                nodes.append(formatted_data)
        return nodes


class RepoDependencySearcher:
    """Traverse Repository Graph"""

    def __init__(self, graph):
        self.G = graph
        self._etypes_dict = {
            etype: i for i, etype in enumerate(EdgeType)
        }

    @classmethod
    def from_rpg(cls, rpg: RPG) -> "RepoDependencySearcher":
        """Create searcher from RPG instance."""
        if rpg.dep_graph is None:
            raise ValueError("RPG does not have a dependency graph")
        return cls(rpg.dep_graph.G)

    def subgraph(self, nids):
        return self.G.subgraph(nids)

    def get_neighbors(self, nid, direction='forward',
                      ntype_filter=None, etype_filter=None, ignore_test_file=True):
        nodes, edges = [], []
        if direction == 'forward':
            for sn in self.G.successors(nid):
                if ntype_filter and self.G.nodes[sn]['type'] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(sn):
                    continue
                for key, edge_data in self.G.get_edge_data(nid, sn).items():
                    etype = edge_data['type']
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((nid, sn, self._etypes_dict[etype], {'type': etype}))
                    nodes.append(sn)

        elif direction == 'backward':
            for pn in self.G.predecessors(nid):
                if ntype_filter and self.G.nodes[pn]['type'] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(pn):
                    continue
                for key, edge_data in self.G.get_edge_data(pn, nid).items():
                    etype = edge_data['type']
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((pn, nid, self._etypes_dict[etype], {'type': etype}))
                    nodes.append(pn)

        return nodes, edges


