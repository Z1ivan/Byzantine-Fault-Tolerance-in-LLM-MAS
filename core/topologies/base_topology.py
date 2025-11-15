
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Set
import random
import logging

class BaseTopology(ABC):

    def __init__(self, num_nodes: int, topology_type: str, **kwargs):
        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.graph = nx.Graph()
        self.node_positions = {}
        self.malicious_nodes: Set[str] = set()

        self.max_malicious_ratio = kwargs.get("max_malicious_ratio", 0.33)            
        self.connectivity_threshold = kwargs.get("connectivity_threshold", 0.5)

        self.logger = logging.getLogger(f"Topology_{topology_type}")

        self._init_nodes()

    def _init_nodes(self):

        for i in range(self.num_nodes):
            node_id = f"agent_{i}"
            self.graph.add_node(node_id, 
                              node_type="normal",
                              agent_id=node_id)

    @abstractmethod
    def build_topology(self) -> nx.Graph:

        pass

    def get_neighbors(self, node_id: str) -> List[str]:

        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def get_all_nodes(self) -> List[str]:

        return sorted(list(self.graph.nodes()))

    def set_malicious_nodes(self, malicious_ratio: float = 0.2) -> List[str]:

        if malicious_ratio > self.max_malicious_ratio:
            self.logger.warning(f"恶意节点比例 {malicious_ratio} 超过最大值 {self.max_malicious_ratio}")
            malicious_ratio = self.max_malicious_ratio

        num_malicious = int(self.num_nodes * malicious_ratio)
        if num_malicious == 0 and malicious_ratio > 0:
            num_malicious = 1             

        all_nodes = self.get_all_nodes()
        self.malicious_nodes = set(random.sample(all_nodes, num_malicious))

        for node_id in self.malicious_nodes:
            self.graph.nodes[node_id]["node_type"] = "malicious"

        self.logger.info(f"设置恶意节点: {self.malicious_nodes}")
        return list(self.malicious_nodes)

    def is_malicious(self, node_id: str) -> bool:

        return node_id in self.malicious_nodes

    def calculate_connectivity(self) -> float:

        if self.num_nodes <= 1:
            return 1.0

        total_degree = sum(dict(self.graph.degree()).values())
        max_possible_degree = self.num_nodes * (self.num_nodes - 1)

        return total_degree / max_possible_degree if max_possible_degree > 0 else 0.0

    def calculate_fault_tolerance(self) -> Dict[str, Any]:

        total_nodes = self.num_nodes
        max_tolerable_faults = (total_nodes - 1) // 3             

        connectivity = self.calculate_connectivity()

        actual_fault_tolerance = int(max_tolerable_faults * connectivity)

        return {
            "total_nodes": total_nodes,
            "max_theoretical_faults": max_tolerable_faults,
            "actual_fault_tolerance": actual_fault_tolerance,
            "connectivity": connectivity,
            "fault_tolerance_ratio": actual_fault_tolerance / total_nodes if total_nodes > 0 else 0
        }

    def get_shortest_paths(self) -> Dict[str, Dict[str, int]]:

        try:
            return dict(nx.all_pairs_shortest_path_length(self.graph))
        except:
            return {}

    def analyze_network_properties(self) -> Dict[str, Any]:

        properties = {
            "topology_type": self.topology_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.graph.number_of_edges(),
            "connectivity": self.calculate_connectivity(),
            "is_connected": nx.is_connected(self.graph),
            "diameter": None,
            "average_clustering": None,
            "average_path_length": None
        }

        try:
            if nx.is_connected(self.graph):
                properties["diameter"] = nx.diameter(self.graph)
                properties["average_clustering"] = nx.average_clustering(self.graph)
                properties["average_path_length"] = nx.average_shortest_path_length(self.graph)
        except:
            pass

        fault_tolerance = self.calculate_fault_tolerance()
        properties.update(fault_tolerance)

        return properties

    def simulate_node_failure(self, failed_nodes: List[str]) -> Dict[str, Any]:

        temp_graph = self.graph.copy()

        temp_graph.remove_nodes_from(failed_nodes)

        remaining_nodes = len(temp_graph.nodes())
        is_connected = nx.is_connected(temp_graph) if remaining_nodes > 0 else False

        result = {
            "failed_nodes": failed_nodes,
            "remaining_nodes": remaining_nodes,
            "is_connected_after_failure": is_connected,
            "largest_component_size": 0
        }

        if remaining_nodes > 0:
            components = list(nx.connected_components(temp_graph))
            result["largest_component_size"] = len(max(components, key=len)) if components else 0
            result["num_components"] = len(components)

        return result

    def get_critical_nodes(self) -> List[str]:

        critical_nodes = []

        for node in self.graph.nodes():

            temp_graph = self.graph.copy()
            temp_graph.remove_node(node)

            if not nx.is_connected(temp_graph) and nx.is_connected(self.graph):
                critical_nodes.append(node)

        return critical_nodes

    def visualize(self, save_path: str = None, show_malicious: bool = True) -> None:

        plt.figure(figsize=(12, 8))

        node_colors = []
        for node in self.graph.nodes():
            if show_malicious and self.is_malicious(node):
                node_colors.append('red')           
            else:
                node_colors.append('lightblue')            

        pos = self.node_positions if self.node_positions else nx.spring_layout(self.graph)

        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)

        plt.title(f"{self.topology_type.title()} Topology ({self.num_nodes} nodes)")

        if show_malicious and self.malicious_nodes:
            plt.text(0.02, 0.98, f"恶意节点: {len(self.malicious_nodes)}", 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"拓扑图已保存到: {save_path}")

        plt.show()

    def export_graph_data(self) -> Dict[str, Any]:

        return {
            "topology_type": self.topology_type,
            "num_nodes": self.num_nodes,
            "nodes": [
                {
                    "id": node,
                    "type": self.graph.nodes[node].get("node_type", "normal"),
                    "is_malicious": self.is_malicious(node)
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {"source": edge[0], "target": edge[1]}
                for edge in self.graph.edges()
            ],
            "properties": self.analyze_network_properties()
        }

    def get_topology_summary(self) -> str:

        properties = self.analyze_network_properties()

        summary = f"""
拓扑类型: {self.topology_type}
节点数量: {self.num_nodes}
边数量: {properties['num_edges']}
连通性: {properties['connectivity']:.3f}
是否连通: {properties['is_connected']}
恶意节点数: {len(self.malicious_nodes)}
理论容错上限: {properties['max_theoretical_faults']}
实际容错能力: {properties['actual_fault_tolerance']}
        """.strip()

        return summary 