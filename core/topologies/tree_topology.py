
import networkx as nx
import numpy as np
import math
from .base_topology import BaseTopology

class TreeTopology(BaseTopology):

    def __init__(self, num_nodes: int, branching_factor: int = 2, root_node_id: str = None, **kwargs):
        self.branching_factor = branching_factor        
        self.root_node_id = root_node_id or "agent_0"
        self.tree_levels = {}             
        super().__init__(num_nodes, "tree", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        if self.root_node_id not in nodes:
            self.root_node_id = nodes[0]

        self._build_tree_structure(nodes)

        self.node_positions = self._calculate_tree_positions()

        self.logger.info(f"树形拓扑构建完成: {self.num_nodes}个节点, {self.graph.number_of_edges()}条边, 根节点: {self.root_node_id}")
        return self.graph

    def _build_tree_structure(self, nodes: list):

        self.tree_levels[self.root_node_id] = 0

        queue = [self.root_node_id]
        remaining_nodes = [node for node in nodes if node != self.root_node_id]

        while queue and remaining_nodes:
            parent = queue.pop(0)
            parent_level = self.tree_levels[parent]

            children_count = min(self.branching_factor, len(remaining_nodes))

            for _ in range(children_count):
                if not remaining_nodes:
                    break

                child = remaining_nodes.pop(0)
                self.graph.add_edge(parent, child)
                self.tree_levels[child] = parent_level + 1
                queue.append(child)

    def _calculate_tree_positions(self) -> dict:

        positions = {}

        levels = {}
        for node, level in self.tree_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(node)

        max_level = max(levels.keys()) if levels else 0

        for level, nodes_in_level in levels.items():
            y = max_level - level          

            if len(nodes_in_level) == 1:
                positions[nodes_in_level[0]] = (0, y)
            else:

                width = len(nodes_in_level) - 1
                for i, node in enumerate(nodes_in_level):
                    x = (i - width / 2) * 2 / max(width, 1)
                    positions[node] = (x, y)

        return positions

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        tree_properties = {
            "root_node": self.root_node_id,
            "branching_factor": self.branching_factor,
            "tree_height": max(self.tree_levels.values()) if self.tree_levels else 0,
            "is_tree": nx.is_tree(self.graph),
            "leaf_nodes": self.get_leaf_nodes(),
            "internal_nodes": self.get_internal_nodes(),
            "redundancy": "none",             
            "fault_tolerance_level": "poor"          
        }

        properties.update(tree_properties)
        return properties

    def get_leaf_nodes(self) -> list:

        return [node for node in self.graph.nodes() if self.graph.degree(node) == 1 and node != self.root_node_id]

    def get_internal_nodes(self) -> list:

        leaf_nodes = set(self.get_leaf_nodes())
        return [node for node in self.graph.nodes() if node not in leaf_nodes]

    def get_children(self, node_id: str) -> list:

        if node_id not in self.tree_levels:
            return []

        node_level = self.tree_levels[node_id]
        children = []

        for neighbor in self.get_neighbors(node_id):
            if self.tree_levels.get(neighbor, -1) == node_level + 1:
                children.append(neighbor)

        return children

    def get_parent(self, node_id: str) -> str:

        if node_id == self.root_node_id or node_id not in self.tree_levels:
            return None

        node_level = self.tree_levels[node_id]

        for neighbor in self.get_neighbors(node_id):
            if self.tree_levels.get(neighbor, -1) == node_level - 1:
                return neighbor

        return None

    def calculate_byzantine_resilience(self) -> dict:

        max_faults = 0               

        return {
            "topology_type": "tree",
            "max_byzantine_faults": max_faults,
            "resilience_ratio": max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(log n)",
            "consensus_rounds": self.tree_levels.get(self.root_node_id, 0) + max(self.tree_levels.values()) if self.tree_levels else 1,
            "advantages": [
                "层次化结构清晰",
                "通信路径明确",
                "扩展性好",
                "管理简单"
            ],
            "disadvantages": [
                "单点故障风险高",
                "无冗余路径",
                "容错能力极差",
                "根节点负载重"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        root_is_malicious = self.root_node_id in malicious_nodes
        internal_malicious = len([node for node in malicious_nodes if node in self.get_internal_nodes()])

        consensus_simulation = {
            "topology": "tree",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "root_is_malicious": root_is_malicious,
            "internal_malicious_count": internal_malicious,
            "communication_rounds": max(self.tree_levels.values()) * 2 if self.tree_levels else 2,         
            "messages_per_round": self.num_nodes - 1,        
            "consensus_possible": len(malicious_nodes) == 0,                  
            "isolation_risk": "high" if internal_malicious > 0 else "low",
            "partition_risk": "high" if internal_malicious > 0 else "none"
        }

        return consensus_simulation

    def find_critical_paths(self) -> list:

        critical_paths = []
        leaf_nodes = self.get_leaf_nodes()

        for leaf in leaf_nodes:
            try:
                path = nx.shortest_path(self.graph, self.root_node_id, leaf)
                critical_paths.append(path)
            except nx.NetworkXNoPath:
                continue

        return critical_paths

    def simulate_subtree_failure(self, failed_node: str) -> dict:

        result = super().simulate_node_failure([failed_node])

        if failed_node == self.root_node_id:

            result["impact_level"] = "catastrophic"
            result["affected_subtree_size"] = self.num_nodes
        else:

            children = self.get_children(failed_node)
            affected_size = 1            

            def count_subtree_size(node):
                size = 1
                for child in self.get_children(node):
                    size += count_subtree_size(child)
                return size

            for child in children:
                affected_size += count_subtree_size(child)

            result["impact_level"] = "moderate" if affected_size < self.num_nodes * 0.3 else "severe"
            result["affected_subtree_size"] = affected_size
            result["isolated_nodes"] = affected_size - 1          

        return result 