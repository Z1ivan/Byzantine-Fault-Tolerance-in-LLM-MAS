
import networkx as nx
import numpy as np
import random
from .base_topology import BaseTopology

class RandomTopology(BaseTopology):

    def __init__(self, num_nodes: int, connection_probability: float = 0.3, 
                 min_connectivity: bool = True, seed: int = None, **kwargs):
        self.connection_probability = connection_probability
        self.min_connectivity = min_connectivity           
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        super().__init__(num_nodes, "random", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        self.graph = nx.erdos_renyi_graph(
            self.num_nodes, 
            self.connection_probability, 
            seed=self.seed
        )

        mapping = {i: f"agent_{i}" for i in range(self.num_nodes)}
        self.graph = nx.relabel_nodes(self.graph, mapping)

        if self.min_connectivity and not nx.is_connected(self.graph):
            self._ensure_connectivity()

        self.node_positions = self._calculate_spring_positions()

        self.logger.info(f"随机拓扑构建完成: {self.num_nodes}个节点, {self.graph.number_of_edges()}条边, 连接概率: {self.connection_probability}")
        return self.graph

    def _ensure_connectivity(self):

        components = list(nx.connected_components(self.graph))

        if len(components) <= 1:
            return

        main_component = max(components, key=len)

        for component in components:
            if component != main_component:

                node1 = random.choice(list(main_component))
                node2 = random.choice(list(component))
                self.graph.add_edge(node1, node2)

                main_component = main_component.union(component)

    def _calculate_spring_positions(self) -> dict:

        return nx.spring_layout(self.graph, seed=self.seed)

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())

        random_properties = {
            "connection_probability": self.connection_probability,
            "actual_edge_density": self.graph.number_of_edges() / (self.num_nodes * (self.num_nodes - 1) / 2),
            "degree_distribution": {
                "mean": np.mean(degree_values),
                "std": np.std(degree_values),
                "min": min(degree_values) if degree_values else 0,
                "max": max(degree_values) if degree_values else 0
            },
            "clustering_coefficient": nx.average_clustering(self.graph),
            "small_world_coefficient": self._calculate_small_world_coefficient(),
            "redundancy": "variable",         
            "fault_tolerance_level": "moderate"          
        }

        properties.update(random_properties)
        return properties

    def _calculate_small_world_coefficient(self) -> float:

        try:
            if not nx.is_connected(self.graph):
                return 0.0

            actual_clustering = nx.average_clustering(self.graph)

            actual_path_length = nx.average_shortest_path_length(self.graph)

            expected_clustering = self.connection_probability
            expected_path_length = np.log(self.num_nodes) / np.log(self.num_nodes * self.connection_probability)

            if expected_clustering > 0 and expected_path_length > 0:
                return (actual_clustering / expected_clustering) / (actual_path_length / expected_path_length)

        except:
            pass

        return 0.0

    def calculate_byzantine_resilience(self) -> dict:

        theoretical_max_faults = (self.num_nodes - 1) // 3

        connectivity = self.calculate_connectivity()
        actual_max_faults = int(theoretical_max_faults * connectivity)

        return {
            "topology_type": "random",
            "max_byzantine_faults": actual_max_faults,
            "resilience_ratio": actual_max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(n²)",
            "consensus_rounds": int(np.log(self.num_nodes)) + 1,           
            "advantages": [
                "结构多样性",
                "无单点故障",
                "适应性强",
                "负载分布相对均匀"
            ],
            "disadvantages": [
                "性能不可预测",
                "可能存在孤立节点",
                "通信开销不确定",
                "难以优化"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        connectivity_after_attack = self._analyze_robustness_under_attack(malicious_nodes)

        consensus_simulation = {
            "topology": "random",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "connection_probability": self.connection_probability,
            "communication_rounds": int(np.log(self.num_nodes)) + 2,
            "messages_per_round": self.graph.number_of_edges() * 2,        
            "consensus_possible": connectivity_after_attack["is_connected"],
            "isolation_risk": "low" if connectivity_after_attack["largest_component_ratio"] > 0.8 else "high",
            "partition_risk": "low" if len(connectivity_after_attack["components"]) <= 2 else "high",
            "robustness_score": connectivity_after_attack["robustness_score"]
        }

        return consensus_simulation

    def _analyze_robustness_under_attack(self, malicious_nodes: set) -> dict:

        temp_graph = self.graph.copy()
        temp_graph.remove_nodes_from(malicious_nodes)

        if temp_graph.number_of_nodes() == 0:
            return {
                "is_connected": False,
                "largest_component_ratio": 0,
                "components": [],
                "robustness_score": 0
            }

        components = list(nx.connected_components(temp_graph))
        largest_component_size = len(max(components, key=len)) if components else 0

        return {
            "is_connected": nx.is_connected(temp_graph),
            "largest_component_ratio": largest_component_size / temp_graph.number_of_nodes(),
            "components": components,
            "robustness_score": largest_component_size / self.num_nodes
        }

    def analyze_degree_distribution(self) -> dict:

        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())

        degree_counts = {}
        for degree in degree_values:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1

        return {
            "degree_sequence": degree_values,
            "degree_distribution": degree_counts,
            "statistics": {
                "mean": np.mean(degree_values),
                "median": np.median(degree_values),
                "std": np.std(degree_values),
                "min": min(degree_values) if degree_values else 0,
                "max": max(degree_values) if degree_values else 0
            },
            "entropy": self._calculate_degree_entropy(degree_counts)
        }

    def _calculate_degree_entropy(self, degree_counts: dict) -> float:

        total_nodes = sum(degree_counts.values())
        if total_nodes == 0:
            return 0

        entropy = 0
        for count in degree_counts.values():
            if count > 0:
                prob = count / total_nodes
                entropy -= prob * np.log2(prob)

        return entropy

    def simulate_random_failures(self, failure_ratio: float = 0.2, num_simulations: int = 100) -> dict:

        results = []

        for _ in range(num_simulations):

            num_failures = int(self.num_nodes * failure_ratio)
            failed_nodes = random.sample(self.get_all_nodes(), num_failures)

            result = self.simulate_node_failure(failed_nodes)
            results.append(result)

        connectivity_preserved = sum(1 for r in results if r["is_connected_after_failure"])
        avg_largest_component = np.mean([r["largest_component_size"] for r in results])

        return {
            "num_simulations": num_simulations,
            "failure_ratio": failure_ratio,
            "connectivity_preservation_rate": connectivity_preserved / num_simulations,
            "average_largest_component_size": avg_largest_component,
            "robustness_score": avg_largest_component / self.num_nodes
        }

    def regenerate_with_new_probability(self, new_probability: float):

        self.connection_probability = new_probability
        self.graph.clear()
        self.build_topology() 