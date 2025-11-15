#!/usr/bin/env python3

import logging
from typing import Optional, List, Dict, Any
from core.interfaces import ITopology, TopologyType
from .base_topology import BaseTopology
from .star_topology import StarTopology
from .chain_topology import ChainTopology
from .complete_graph import CompleteGraphTopology
from .random_topology import RandomTopology
from .tree_topology import TreeTopology
from .layered_graph_topology import LayeredGraphTopology

logger = logging.getLogger(__name__)

class TopologyAdapter(ITopology):

    def __init__(self, topology: BaseTopology):
        self.topology = topology

    def add_connection(self, node1: str, node2: str) -> None:

        if hasattr(self.topology, 'add_edge'):
            self.topology.add_edge(node1, node2)
        elif hasattr(self.topology, 'add_connection'):
            self.topology.add_connection(node1, node2)

    def remove_connection(self, node1: str, node2: str) -> None:

        if hasattr(self.topology, 'remove_edge'):
            self.topology.remove_edge(node1, node2)
        elif hasattr(self.topology, 'remove_connection'):
            self.topology.remove_connection(node1, node2)

    def get_neighbors(self, node_id: str) -> List[str]:

        if hasattr(self.topology, 'get_neighbors'):
            return self.topology.get_neighbors(node_id)
        elif hasattr(self.topology, 'neighbors'):
            return list(self.topology.neighbors(node_id))
        return []

    @property
    def graph(self):

        if hasattr(self.topology, 'graph'):
            return self.topology.graph
        return None

    def get_all_nodes(self) -> List[str]:

        if hasattr(self.topology, 'get_all_nodes'):
            return self.topology.get_all_nodes()
        elif hasattr(self.topology, 'graph'):
            try:
                return sorted([str(n) for n in self.topology.graph.nodes()])
            except Exception:
                return list(self.topology.graph.nodes())
        return []

    def get_all_connections(self) -> Dict[str, List[str]]:

        if hasattr(self.topology, 'get_all_connections'):
            return self.topology.get_all_connections()
        elif hasattr(self.topology, 'adjacency_list'):
            return self.topology.adjacency_list
        elif hasattr(self.topology, 'graph'):

            return {str(node): [str(neighbor) for neighbor in self.topology.graph.neighbors(node)] 
                   for node in self.topology.graph.nodes()}
        return {}

    def get_node_count(self) -> int:

        if hasattr(self.topology, 'get_node_count'):
            return self.topology.get_node_count()
        elif hasattr(self.topology, 'num_nodes'):
            return self.topology.num_nodes
        elif hasattr(self.topology, 'graph'):
            return len(self.topology.graph.nodes())
        return 0

    def is_connected(self, node1: str, node2: str) -> bool:

        if hasattr(self.topology, 'is_connected'):
            return self.topology.is_connected(node1, node2)
        elif hasattr(self.topology, 'has_edge'):
            return self.topology.has_edge(node1, node2)
        elif hasattr(self.topology, 'graph'):
            return self.topology.graph.has_edge(node1, node2)
        return False

    @property
    def topology_type(self) -> TopologyType:

        if hasattr(self.topology, 'topology_type'):
            return self.topology.topology_type
        elif hasattr(self.topology, '__class__'):
            class_name = self.topology.__class__.__name__.lower()
            if 'star' in class_name:
                return TopologyType.STAR
            elif 'complete' in class_name:
                return TopologyType.COMPLETE
            elif 'chain' in class_name:
                return TopologyType.CHAIN
            elif 'random' in class_name:
                return TopologyType.RANDOM
            else:
                return TopologyType.RANDOM
        return TopologyType.RANDOM

    @property
    def node_count(self) -> int:

        return self.get_node_count()

    @property
    def num_nodes(self) -> int:

        return self.get_node_count()

    def update_topology(self, **kwargs) -> None:

        if hasattr(self.topology, 'update_topology'):
            self.topology.update_topology(**kwargs)
        elif hasattr(self.topology, 'update'):
            self.topology.update(**kwargs)

    @property
    def center_node_id(self) -> Optional[str]:

        if hasattr(self.topology, 'center_node_id'):
            return getattr(self.topology, 'center_node_id')
        if hasattr(self.topology, 'get_center_node'):
            try:
                return self.topology.get_center_node()
            except Exception:
                return None
        return None

    def get_leaf_nodes(self) -> List[str]:

        if hasattr(self.topology, 'get_leaf_nodes'):
            return self.topology.get_leaf_nodes()

        try:
            import networkx as nx  # noqa: F401
            return [n for n in self.topology.graph.nodes() if self.topology.graph.degree(n) == 1]
        except Exception:
            return []

    @property
    def root_node_id(self) -> Optional[str]:

        if hasattr(self.topology, 'root_node_id'):
            return getattr(self.topology, 'root_node_id')
        return None

    def get_internal_nodes(self) -> List[str]:

        if hasattr(self.topology, 'get_internal_nodes'):
            return self.topology.get_internal_nodes()

        leaves = set(self.get_leaf_nodes())
        try:
            return [n for n in self.topology.graph.nodes() if n not in leaves]
        except Exception:
            return []

    def get_topology_metrics(self) -> Dict[str, Any]:

        if hasattr(self.topology, 'get_topology_metrics'):
            return self.topology.get_topology_metrics()

        metrics = {
            'node_count': self.get_node_count(),
            'edge_count': 0,
            'average_degree': 0.0,
            'density': 0.0
        }

        try:
            connections = self.get_all_connections()
            total_edges = sum(len(neighbors) for neighbors in connections.values()) // 2
            metrics['edge_count'] = total_edges

            if self.get_node_count() > 0:
                metrics['average_degree'] = (2 * total_edges) / self.get_node_count()
                max_edges = self.get_node_count() * (self.get_node_count() - 1) // 2
                if max_edges > 0:
                    metrics['density'] = total_edges / max_edges
        except Exception as e:
            logger.warning(f"计算拓扑指标失败: {e}")

        return metrics

class TopologyFactory:

    @staticmethod
    def create_topology(
        topology_type: TopologyType,
        node_count: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> ITopology:

        try:
            if topology_type == TopologyType.CHAIN:
                topology = ChainTopology(node_count)
                return TopologyAdapter(topology)

            elif topology_type == TopologyType.STAR:
                topology = StarTopology(node_count)
                return TopologyAdapter(topology)
            elif topology_type == TopologyType.COMPLETE:
                topology = CompleteGraphTopology(node_count)
                return TopologyAdapter(topology)
            elif topology_type == TopologyType.RANDOM:
                edge_probability = kwargs.get('edge_probability', 0.3)
                topology = RandomTopology(node_count, edge_probability, seed)
                return TopologyAdapter(topology)
            elif topology_type == TopologyType.TREE:
                topology = TreeTopology(node_count)
                return TopologyAdapter(topology)
            elif topology_type == TopologyType.LAYERED_GRAPH:
                num_layers = kwargs.get('num_layers', 3)
                inter_ratio = kwargs.get('inter_layer_connection_ratio', 0.3)
                topology = LayeredGraphTopology(node_count, num_layers=num_layers, inter_layer_connection_ratio=inter_ratio)
                return TopologyAdapter(topology)
            else:
                raise ValueError(f"不支持的拓扑类型: {topology_type}")
        except Exception as e:
            logger.error(f"创建拓扑失败: {e}")

            logger.warning(f"回退到星形拓扑")
            topology = StarTopology(node_count)
            return TopologyAdapter(topology)

    @staticmethod
    def get_supported_topologies() -> List[TopologyType]:

        return list(TopologyType)
