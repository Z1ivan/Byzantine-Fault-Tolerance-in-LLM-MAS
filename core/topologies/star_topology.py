
import networkx as nx
import numpy as np
from .base_topology import BaseTopology

class StarTopology(BaseTopology):

    def __init__(self, num_nodes: int, center_node_id: str = None, **kwargs):
        self.center_node_id = center_node_id or "agent_0"
        super().__init__(num_nodes, "star", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        if self.center_node_id not in nodes:
            self.center_node_id = nodes[0]

        for node in nodes:
            if node != self.center_node_id:
                self.graph.add_edge(self.center_node_id, node)

        self.node_positions = self._calculate_star_positions()

        self.logger.info(f"星形拓扑构建完成: {self.num_nodes}个节点, {self.graph.number_of_edges()}条边, 中心节点: {self.center_node_id}")
        return self.graph

    def _calculate_star_positions(self) -> dict:

        positions = {}
        nodes = self.get_all_nodes()

        positions[self.center_node_id] = (0, 0)

        other_nodes = [node for node in nodes if node != self.center_node_id]

        if len(other_nodes) == 0:
            return positions

        angle_step = 2 * np.pi / len(other_nodes)
        radius = 1.0

        for i, node in enumerate(other_nodes):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[node] = (x, y)

        return positions

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        star_properties = {
            "center_node": self.center_node_id,
            "center_degree": self.num_nodes - 1,           
            "leaf_degree": 1,           
            "redundancy": "none",         
            "fault_tolerance_level": "critical_center",           
            "single_point_of_failure": self.center_node_id
        }

        properties.update(star_properties)
        return properties

    def calculate_byzantine_resilience(self) -> dict:

        max_faults = 0                 

        return {
            "topology_type": "star",
            "max_byzantine_faults": max_faults,
            "resilience_ratio": max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(n)",
            "consensus_rounds": 2,                   
            "advantages": [
                "通信效率高",
                "中心化控制",
                "简单的路由",
                "低通信延迟"
            ],
            "disadvantages": [
                "单点故障风险极高",
                "中心节点负载重",
                "无容错能力",
                "中心节点成为攻击目标"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        center_is_malicious = self.center_node_id in malicious_nodes

        consensus_simulation = {
            "topology": "star",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "center_node": self.center_node_id,
            "center_is_malicious": center_is_malicious,
            "communication_rounds": 2,
            "messages_per_round": self.num_nodes - 1,
            "consensus_possible": not center_is_malicious,                   
            "isolation_risk": "low" if not center_is_malicious else "total",
            "partition_risk": "none" if not center_is_malicious else "total"
        }

        return consensus_simulation

    def get_center_node(self) -> str:

        return self.center_node_id

    def get_leaf_nodes(self) -> list:

        nodes = self.get_all_nodes()
        return [node for node in nodes if node != self.center_node_id]

    def simulate_center_node_failure(self) -> dict:

        result = super().simulate_node_failure([self.center_node_id])

        leaf_nodes = self.get_leaf_nodes()

        result.update({
            "impact_level": "catastrophic",
            "network_split": True,
            "isolated_nodes": leaf_nodes,
            "remaining_connectivity": 0,
            "recovery_possible": False
        })

        return result

    def analyze_centrality(self) -> dict:

        centrality_analysis = {
            "betweenness_centrality": nx.betweenness_centrality(self.graph),
            "closeness_centrality": nx.closeness_centrality(self.graph),
            "degree_centrality": nx.degree_centrality(self.graph),
            "most_central_node": self.center_node_id,
            "centrality_distribution": "极度不均匀"
        }

        return centrality_analysis

    def simulate_load_balancing(self) -> dict:

        total_communication_load = (self.num_nodes - 1) * 2                  

        load_analysis = {
            "center_node_load": total_communication_load,
            "leaf_node_load": 2,                
            "load_distribution": "极度不均匀",
            "bottleneck_node": self.center_node_id,
            "scalability": "poor"                     
        }

        return load_analysis 