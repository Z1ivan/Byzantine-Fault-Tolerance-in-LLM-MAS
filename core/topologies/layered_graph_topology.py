
import networkx as nx
import numpy as np
import random
import math
from .base_topology import BaseTopology

class LayeredGraphTopology(BaseTopology):

    def __init__(self, num_nodes: int, num_layers: int = 3, 
                 inter_layer_connection_ratio: float = 0.3, **kwargs):
        self.num_layers = max(2, num_layers)        
        self.inter_layer_connection_ratio = inter_layer_connection_ratio
        self.layer_assignments = {}           
        self.layer_nodes = {}           

        super().__init__(num_nodes, "layered_graph", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        self._assign_nodes_to_layers(nodes)

        self._build_intra_layer_connections()

        self._build_inter_layer_connections()

        self.node_positions = self._calculate_layered_positions()

        self.logger.info(f"分层图拓扑构建完成: {self.num_nodes}个节点, {self.num_layers}层, {self.graph.number_of_edges()}条边")
        return self.graph

    def _assign_nodes_to_layers(self, nodes: list):

        nodes_per_layer = self.num_nodes // self.num_layers
        remaining_nodes = self.num_nodes % self.num_layers

        node_index = 0
        for layer in range(self.num_layers):

            current_layer_size = nodes_per_layer
            if layer < remaining_nodes:
                current_layer_size += 1

            layer_nodes = []
            for _ in range(current_layer_size):
                if node_index < len(nodes):
                    node = nodes[node_index]
                    self.layer_assignments[node] = layer
                    layer_nodes.append(node)
                    node_index += 1

            self.layer_nodes[layer] = layer_nodes

    def _build_intra_layer_connections(self):

        for layer, nodes_in_layer in self.layer_nodes.items():

            for i in range(len(nodes_in_layer)):
                for j in range(i + 1, len(nodes_in_layer)):
                    self.graph.add_edge(nodes_in_layer[i], nodes_in_layer[j])

    def _build_inter_layer_connections(self):

        for layer1 in range(self.num_layers):
            for layer2 in range(layer1 + 1, self.num_layers):
                self._connect_adjacent_layers(layer1, layer2)

    def _connect_adjacent_layers(self, layer1: int, layer2: int):

        nodes1 = self.layer_nodes[layer1]
        nodes2 = self.layer_nodes[layer2]

        max_possible_connections = len(nodes1) * len(nodes2)
        num_connections = int(max_possible_connections * self.inter_layer_connection_ratio)

        connections_made = 0
        attempts = 0
        max_attempts = max_possible_connections * 2

        while connections_made < num_connections and attempts < max_attempts:
            node1 = random.choice(nodes1)
            node2 = random.choice(nodes2)

            if not self.graph.has_edge(node1, node2):
                self.graph.add_edge(node1, node2)
                connections_made += 1

            attempts += 1

    def _calculate_layered_positions(self) -> dict:

        positions = {}

        for layer, nodes_in_layer in self.layer_nodes.items():

            y = layer * 2

            if len(nodes_in_layer) == 1:
                positions[nodes_in_layer[0]] = (0, y)
            else:
                width = len(nodes_in_layer) - 1
                for i, node in enumerate(nodes_in_layer):
                    x = (i - width / 2) * 2 / max(width, 1)
                    positions[node] = (x, y)

        return positions

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        layered_properties = {
            "num_layers": self.num_layers,
            "layer_assignments": self.layer_assignments,
            "layer_sizes": {layer: len(nodes) for layer, nodes in self.layer_nodes.items()},
            "inter_layer_connection_ratio": self.inter_layer_connection_ratio,
            "intra_layer_density": self._calculate_intra_layer_density(),
            "inter_layer_density": self._calculate_inter_layer_density(),
            "redundancy": "high_within_layers",         
            "fault_tolerance_level": "good"           
        }

        properties.update(layered_properties)
        return properties

    def _calculate_intra_layer_density(self) -> dict:

        densities = {}

        for layer, nodes_in_layer in self.layer_nodes.items():
            if len(nodes_in_layer) <= 1:
                densities[layer] = 1.0
                continue

            intra_edges = 0
            for i in range(len(nodes_in_layer)):
                for j in range(i + 1, len(nodes_in_layer)):
                    if self.graph.has_edge(nodes_in_layer[i], nodes_in_layer[j]):
                        intra_edges += 1

            max_possible_edges = len(nodes_in_layer) * (len(nodes_in_layer) - 1) // 2
            densities[layer] = intra_edges / max_possible_edges if max_possible_edges > 0 else 0

        return densities

    def _calculate_inter_layer_density(self) -> dict:

        densities = {}

        for layer1 in range(self.num_layers):
            for layer2 in range(layer1 + 1, self.num_layers):
                nodes1 = self.layer_nodes[layer1]
                nodes2 = self.layer_nodes[layer2]

                inter_edges = 0
                for node1 in nodes1:
                    for node2 in nodes2:
                        if self.graph.has_edge(node1, node2):
                            inter_edges += 1

                max_possible_edges = len(nodes1) * len(nodes2)
                density = inter_edges / max_possible_edges if max_possible_edges > 0 else 0
                densities[f"layer_{layer1}_to_{layer2}"] = density

        return densities

    def get_node_layer(self, node_id: str) -> int:

        return self.layer_assignments.get(node_id, -1)

    def get_layer_nodes(self, layer: int) -> list:

        return self.layer_nodes.get(layer, [])

    def calculate_byzantine_resilience(self) -> dict:

        theoretical_max_faults = (self.num_nodes - 1) // 3

        layer_resilience_bonus = 0.2                 
        actual_max_faults = int(theoretical_max_faults * (1 + layer_resilience_bonus))
        actual_max_faults = min(actual_max_faults, self.num_nodes // 2)          

        return {
            "topology_type": "layered_graph",
            "max_byzantine_faults": actual_max_faults,
            "resilience_ratio": actual_max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(n²) within layers, O(n) between layers",
            "consensus_rounds": self.num_layers + 2,               
            "advantages": [
                "层内高冗余",
                "结构化的容错",
                "可扩展性好",
                "负载分层分布"
            ],
            "disadvantages": [
                "层间依赖性",
                "复杂的路由",
                "可能的层间瓶颈",
                "设计复杂度高"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        layer_malicious_distribution = {}
        for layer in range(self.num_layers):
            layer_nodes = self.get_layer_nodes(layer)
            layer_malicious = [node for node in malicious_nodes if node in layer_nodes]
            layer_malicious_distribution[layer] = {
                "total_nodes": len(layer_nodes),
                "malicious_nodes": len(layer_malicious),
                "malicious_ratio": len(layer_malicious) / len(layer_nodes) if layer_nodes else 0
            }

        consensus_possible = all(
            dist["malicious_ratio"] < 0.5 for dist in layer_malicious_distribution.values()
        )

        consensus_simulation = {
            "topology": "layered_graph",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "num_layers": self.num_layers,
            "layer_malicious_distribution": layer_malicious_distribution,
            "communication_rounds": self.num_layers + 3,                  
            "messages_per_round": self._estimate_messages_per_round(),
            "consensus_possible": consensus_possible,
            "isolation_risk": "low",              
            "partition_risk": "moderate"              
        }

        return consensus_simulation

    def _estimate_messages_per_round(self) -> int:

        intra_layer_messages = 0
        for layer_nodes in self.layer_nodes.values():
            n = len(layer_nodes)
            intra_layer_messages += n * (n - 1)                 

        inter_layer_messages = self.graph.number_of_edges() * 2 - intra_layer_messages

        return intra_layer_messages + inter_layer_messages

    def analyze_layer_connectivity(self) -> dict:

        connectivity_analysis = {}

        for layer1 in range(self.num_layers):
            for layer2 in range(layer1 + 1, self.num_layers):
                nodes1 = self.layer_nodes[layer1]
                nodes2 = self.layer_nodes[layer2]

                connections = 0
                for node1 in nodes1:
                    for node2 in nodes2:
                        if self.graph.has_edge(node1, node2):
                            connections += 1

                connectivity_analysis[f"layer_{layer1}_to_{layer2}"] = {
                    "connections": connections,
                    "max_possible": len(nodes1) * len(nodes2),
                    "connectivity_ratio": connections / (len(nodes1) * len(nodes2)) if nodes1 and nodes2 else 0
                }

        return connectivity_analysis

    def simulate_layer_failure(self, failed_layer: int) -> dict:

        if failed_layer not in self.layer_nodes:
            return {"error": "Invalid layer"}

        failed_nodes = self.layer_nodes[failed_layer]
        result = super().simulate_node_failure(failed_nodes)

        remaining_layers = [layer for layer in range(self.num_layers) if layer != failed_layer]

        result.update({
            "failed_layer": failed_layer,
            "failed_layer_size": len(failed_nodes),
            "remaining_layers": remaining_layers,
            "impact_level": "severe" if len(failed_nodes) > self.num_nodes * 0.3 else "moderate"
        })

        return result 