
import networkx as nx
import numpy as np
from .base_topology import BaseTopology

class ChainTopology(BaseTopology):

    def __init__(self, num_nodes: int, bidirectional: bool = True, **kwargs):
        self.bidirectional = bidirectional
        super().__init__(num_nodes, "chain", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        for i in range(len(nodes) - 1):
            self.graph.add_edge(nodes[i], nodes[i + 1])

        self.node_positions = self._calculate_linear_positions()

        self.logger.info(f"链式拓扑构建完成: {self.num_nodes}个节点, {self.graph.number_of_edges()}条边")
        return self.graph

    def _calculate_linear_positions(self) -> dict:

        positions = {}
        nodes = self.get_all_nodes()

        if self.num_nodes == 1:
            positions[nodes[0]] = (0, 0)
        else:
            for i, node in enumerate(nodes):
                x = i / (self.num_nodes - 1) * 2 - 1              
                y = 0
                positions[node] = (x, y)

        return positions

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        chain_properties = {
            "is_linear": True,
            "max_degree": 2,                   
            "end_nodes": [self.get_all_nodes()[0], self.get_all_nodes()[-1]],
            "redundancy": "minimal",         
            "fault_tolerance_level": "poor",          
            "critical_nodes": self.get_all_nodes()[1:-1]              
        }

        properties.update(chain_properties)
        return properties

    def calculate_byzantine_resilience(self) -> dict:

        max_faults = 0 if self.num_nodes > 2 else (self.num_nodes - 1) // 3

        return {
            "topology_type": "chain",
            "max_byzantine_faults": max_faults,
            "resilience_ratio": max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(n)",
            "consensus_rounds": self.num_nodes - 1,                  
            "advantages": [
                "通信开销最小",
                "简单的网络结构",
                "易于实现和维护"
            ],
            "disadvantages": [
                "容错能力极差",
                "单点故障风险高",
                "通信延迟大",
                "易被分割"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        consensus_simulation = {
            "topology": "chain",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "communication_rounds": self.num_nodes - 1,
            "messages_per_round": self.num_nodes - 1,              
            "consensus_possible": len(malicious_nodes) == 0,                  
            "isolation_risk": "high",         
            "partition_risk": "high"          
        }

        return consensus_simulation

    def find_critical_paths(self) -> list:

        nodes = self.get_all_nodes()
        return [nodes]                

    def simulate_node_failure_impact(self, failed_node: str) -> dict:

        result = super().simulate_node_failure([failed_node])

        nodes = self.get_all_nodes()
        if failed_node in nodes:
            node_index = nodes.index(failed_node)

            if node_index == 0 or node_index == len(nodes) - 1:
                result["impact_level"] = "low"
                result["network_split"] = False
            else:

                result["impact_level"] = "critical"
                result["network_split"] = True
                result["split_components"] = [
                    nodes[:node_index],
                    nodes[node_index + 1:]
                ]

        return result 