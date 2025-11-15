
import random
import logging
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
import networkx as nx

logger = logging.getLogger(__name__)

class NodePositionType(Enum):

    RANDOM = "random"

    STAR_CENTER = "star_center"
    STAR_LEAF = "star_leaf"

    TREE_ROOT = "tree_root"
    TREE_INTERNAL = "tree_internal"
    TREE_LEAF = "tree_leaf"

    CHAIN_HEAD = "chain_head"
    CHAIN_TAIL = "chain_tail"
    CHAIN_MIDDLE = "chain_middle"

    COMPLETE_ANY = "complete_any"

    LAYERED_TOP = "layered_top"
    LAYERED_MIDDLE = "layered_middle"
    LAYERED_BOTTOM = "layered_bottom"

    HIGH_CENTRALITY = "high_centrality"
    LOW_CENTRALITY = "low_centrality"
    HIGH_DEGREE = "high_degree"
    LOW_DEGREE = "low_degree"

class MaliciousNodePositionController:

    def __init__(self):
        self.position_strategies = {
            "star": self._get_star_positions,
            "tree": self._get_tree_positions,
            "chain": self._get_chain_positions,
            "complete": self._get_complete_positions,
            "random": self._get_random_positions,
            "layered_graph": self._get_layered_positions,

            "dynamic": self._get_dynamic_positions
        }

    def select_malicious_nodes(self, 
                              topology_type: str,
                              topology_instance,
                              num_malicious: int,
                              position_strategy: NodePositionType = NodePositionType.RANDOM,
                              specific_positions: Optional[List[int]] = None,
                              seed: Optional[int] = None,
                              position_seed: Optional[int] = None) -> List[int]:

        effective_seed = position_seed if position_seed is not None else seed
        if effective_seed is not None:
            random.seed(effective_seed)
        else:

            import time
            random.seed(int(time.time() * 1000) % 2**32)

        if hasattr(topology_type, 'value'):
            topology_type = topology_type.value

        if isinstance(position_strategy, str):
            try:
                position_strategy = NodePositionType(position_strategy)
            except Exception:

                normalized = position_strategy.strip().lower()
                try:
                    position_strategy = NodePositionType(normalized)
                except Exception:
                    logger.warning(f"未知的位置策略: {position_strategy}，使用随机策略")
                    position_strategy = NodePositionType.RANDOM

        total_nodes = topology_instance.num_nodes

        if specific_positions is not None:
            valid_positions = [pos for pos in specific_positions if 0 <= pos < total_nodes]
            return valid_positions[:num_malicious]

        if topology_type in self.position_strategies:
            return self.position_strategies[topology_type](
                topology_instance, num_malicious, position_strategy
            )
        else:
            logger.warning(f"未支持的拓扑类型: {topology_type}, 使用随机选择")
            return self._random_selection(total_nodes, num_malicious)

    def _get_star_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        nodes = topology.get_all_nodes()
        center_node = topology.center_node_id
        leaf_nodes = topology.get_leaf_nodes()

        center_idx = nodes.index(center_node)
        leaf_indices = [nodes.index(node) for node in leaf_nodes]

        if strategy == NodePositionType.STAR_CENTER:

            positions = [center_idx]
            if num_malicious > 1:
                remaining = random.sample(leaf_indices, min(num_malicious - 1, len(leaf_indices)))
                positions.extend(remaining)
            return positions

        elif strategy == NodePositionType.STAR_LEAF:

            return random.sample(leaf_indices, min(num_malicious, len(leaf_indices)))

        else:          
            return self._random_selection(len(nodes), num_malicious)

    def _get_tree_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        nodes = topology.get_all_nodes()
        root_node = topology.root_node_id
        leaf_nodes = topology.get_leaf_nodes()
        internal_nodes = topology.get_internal_nodes()

        root_idx = nodes.index(root_node)
        leaf_indices = [nodes.index(node) for node in leaf_nodes]
        internal_indices = [nodes.index(node) for node in internal_nodes if node != root_node]

        if strategy == NodePositionType.TREE_ROOT:

            positions = [root_idx]
            if num_malicious > 1:

                candidates = internal_indices + leaf_indices
                if candidates:
                    remaining_needed = num_malicious - 1
                    remaining = random.sample(candidates, min(remaining_needed, len(candidates)))
                    positions.extend(remaining)
                    logger.info(f"树根节点策略: 根节点+{len(remaining)}个其他节点 (需要{num_malicious}个恶意节点)")
                else:
                    logger.warning(f"树根节点策略: 只有根节点可用，但需要{num_malicious}个恶意节点")
            return positions

        elif strategy == NodePositionType.TREE_LEAF:

            selected = random.sample(leaf_indices, min(num_malicious, len(leaf_indices)))
            if len(selected) < num_malicious:
                logger.warning(f"树叶子节点策略: 只有{len(selected)}个叶子节点，但需要{num_malicious}个恶意节点")
            return selected

        elif strategy == NodePositionType.TREE_INTERNAL:

            positions = random.sample(internal_indices, min(num_malicious, len(internal_indices)))
            if len(positions) < num_malicious:

                remaining_needed = num_malicious - len(positions)
                additional = random.sample(leaf_indices, min(remaining_needed, len(leaf_indices)))
                positions.extend(additional)
                logger.info(f"树内部节点策略: {len(positions)-len(additional)}个内部节点+{len(additional)}个叶子节点")

                if len(positions) < num_malicious:
                    logger.warning(f"树内部节点策略: 总共只有{len(positions)}个节点，但需要{num_malicious}个恶意节点")
            return positions

        else:          
            return self._random_selection(len(nodes), num_malicious)

    def _get_chain_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        total_nodes = topology.num_nodes

        if strategy == NodePositionType.CHAIN_HEAD:

            positions = [0]
            if num_malicious > 1:

                candidates = list(range(1, total_nodes))
                remaining = random.sample(candidates, min(num_malicious - 1, len(candidates)))
                positions.extend(remaining)
            return positions

        elif strategy == NodePositionType.CHAIN_TAIL:

            positions = [total_nodes - 1]
            if num_malicious > 1:
                candidates = list(range(total_nodes - 1))
                remaining = random.sample(candidates, min(num_malicious - 1, len(candidates)))
                positions.extend(remaining)
            return positions

        elif strategy == NodePositionType.CHAIN_MIDDLE:

            if total_nodes <= 2:
                return self._random_selection(total_nodes, num_malicious)

            middle_nodes = list(range(1, total_nodes - 1))
            selected = random.sample(middle_nodes, min(num_malicious, len(middle_nodes)))
            if len(selected) < num_malicious:

                remaining_needed = num_malicious - len(selected)
                end_nodes = [0, total_nodes - 1]
                additional = random.sample(end_nodes, min(remaining_needed, len(end_nodes)))
                selected.extend(additional)
                logger.info(f"链中间节点策略: {len(selected)-len(additional)}个中间节点+{len(additional)}个端点节点")
            return selected

        else:          
            return self._random_selection(total_nodes, num_malicious)

    def _get_complete_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        total_nodes = topology.num_nodes
        return self._random_selection(total_nodes, num_malicious)

    def _get_layered_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        nodes = topology.get_all_nodes()
        total_nodes = len(nodes)

        if strategy == NodePositionType.HIGH_DEGREE:

            node_degrees = []
            for node in nodes:
                neighbors = topology.get_neighbors(node)
                degree = len(neighbors)
                node_id = int(node.split('_')[1])          
                node_degrees.append((node_id, degree))

            node_degrees.sort(key=lambda x: (-x[1], x[0]))
            selected_indices = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Layered High-Degree策略: 选择度数最高的{num_malicious}个节点: {selected_indices}")
            return selected_indices

        elif strategy == NodePositionType.LOW_DEGREE:

            node_degrees = []
            for node in nodes:
                neighbors = topology.get_neighbors(node)
                degree = len(neighbors)
                node_id = int(node.split('_')[1])
                node_degrees.append((node_id, degree))

            node_degrees.sort(key=lambda x: (x[1], x[0]))
            selected_indices = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Layered Low-Degree策略: 选择度数最低的{num_malicious}个节点: {selected_indices}")
            return selected_indices

        elif strategy == NodePositionType.HIGH_CENTRALITY:

            try:
                import networkx as nx

                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))

                node_centrality.sort(key=lambda x: (-x[1], x[0]))
                selected_indices = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Layered High-Centrality策略: 选择中心性最高的{num_malicious}个节点: {selected_indices}")
                return selected_indices
            except Exception as e:
                logger.warning(f"计算中心性失败: {e}, 回退到度数策略")
                return self._get_layered_positions(topology, num_malicious, NodePositionType.HIGH_DEGREE)

        elif strategy == NodePositionType.LOW_CENTRALITY:

            try:
                import networkx as nx
                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))

                node_centrality.sort(key=lambda x: (x[1], x[0]))
                selected_indices = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Layered Low-Centrality策略: 选择中心性最低的{num_malicious}个节点: {selected_indices}")
                return selected_indices
            except Exception as e:
                logger.warning(f"计算中心性失败: {e}, 回退到度数策略")
                return self._get_layered_positions(topology, num_malicious, NodePositionType.LOW_DEGREE)

        elif strategy == NodePositionType.LAYERED_TOP:

            try:
                layer_info = topology.get_layer_info()
                top_layer_nodes = layer_info.get(0, [])            
                if not top_layer_nodes:
                    logger.warning("未找到顶层节点，使用随机策略")
                    return self._random_selection(total_nodes, num_malicious)

                top_indices = [int(node.split('_')[1]) for node in top_layer_nodes]
                selected = random.sample(top_indices, min(num_malicious, len(top_indices)))

                if len(selected) < num_malicious:
                    remaining_needed = num_malicious - len(selected)
                    all_indices = list(range(total_nodes))
                    remaining_candidates = [i for i in all_indices if i not in selected]
                    additional = random.sample(remaining_candidates, min(remaining_needed, len(remaining_candidates)))
                    selected.extend(additional)

                logger.info(f"Layered Top策略: 选择顶层{len(selected)}个节点: {selected}")
                return selected
            except Exception as e:
                logger.warning(f"获取层信息失败: {e}, 使用随机策略")
                return self._random_selection(total_nodes, num_malicious)

        elif strategy == NodePositionType.LAYERED_BOTTOM:

            try:
                layer_info = topology.get_layer_info()
                if not layer_info:
                    logger.warning("未找到层信息，使用随机策略")
                    return self._random_selection(total_nodes, num_malicious)

                max_layer = max(layer_info.keys())
                bottom_layer_nodes = layer_info.get(max_layer, [])

                if not bottom_layer_nodes:
                    logger.warning("未找到底层节点，使用随机策略")
                    return self._random_selection(total_nodes, num_malicious)

                bottom_indices = [int(node.split('_')[1]) for node in bottom_layer_nodes]
                selected = random.sample(bottom_indices, min(num_malicious, len(bottom_indices)))

                if len(selected) < num_malicious:
                    remaining_needed = num_malicious - len(selected)
                    all_indices = list(range(total_nodes))
                    remaining_candidates = [i for i in all_indices if i not in selected]
                    additional = random.sample(remaining_candidates, min(remaining_needed, len(remaining_candidates)))
                    selected.extend(additional)

                logger.info(f"Layered Bottom策略: 选择底层{len(selected)}个节点: {selected}")
                return selected
            except Exception as e:
                logger.warning(f"获取层信息失败: {e}, 使用随机策略")
                return self._random_selection(total_nodes, num_malicious)

        elif strategy == NodePositionType.LAYERED_MIDDLE:

            try:
                layer_info = topology.get_layer_info()
                if not layer_info or len(layer_info) < 3:
                    logger.warning("层数不足或未找到层信息，使用随机策略")
                    return self._random_selection(total_nodes, num_malicious)

                layers = sorted(layer_info.keys())
                middle_layers = layers[1:-1]              

                middle_nodes = []
                for layer in middle_layers:
                    middle_nodes.extend(layer_info[layer])

                if not middle_nodes:
                    logger.warning("未找到中间层节点，使用随机策略")
                    return self._random_selection(total_nodes, num_malicious)

                middle_indices = [int(node.split('_')[1]) for node in middle_nodes]
                selected = random.sample(middle_indices, min(num_malicious, len(middle_indices)))

                if len(selected) < num_malicious:
                    remaining_needed = num_malicious - len(selected)
                    all_indices = list(range(total_nodes))
                    remaining_candidates = [i for i in all_indices if i not in selected]
                    additional = random.sample(remaining_candidates, min(remaining_needed, len(remaining_candidates)))
                    selected.extend(additional)

                logger.info(f"Layered Middle策略: 选择中间层{len(selected)}个节点: {selected}")
                return selected
            except Exception as e:
                logger.warning(f"获取层信息失败: {e}, 使用随机策略")
                return self._random_selection(total_nodes, num_malicious)

        else:          
            return self._random_selection(total_nodes, num_malicious)

    def _get_random_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        nodes = topology.get_all_nodes()
        total_nodes = len(nodes)

        if strategy == NodePositionType.HIGH_DEGREE:

            node_degrees = []
            for node in nodes:
                degree = len(topology.get_neighbors(node))
                node_id = int(node.split('_')[1])
                node_degrees.append((node_id, degree))
            node_degrees.sort(key=lambda x: (-x[1], x[0]))
            selected = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Random High-Degree策略: 选择度数最高的{num_malicious}个节点: {selected}")
            return selected

        if strategy == NodePositionType.LOW_DEGREE:

            node_degrees = []
            for node in nodes:
                degree = len(topology.get_neighbors(node))
                node_id = int(node.split('_')[1])
                node_degrees.append((node_id, degree))
            node_degrees.sort(key=lambda x: (x[1], x[0]))
            selected = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Random Low-Degree策略: 选择度数最低的{num_malicious}个节点: {selected}")
            return selected

        if strategy == NodePositionType.HIGH_CENTRALITY:

            try:
                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))
                node_centrality.sort(key=lambda x: (-x[1], x[0]))
                selected = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Random High-Centrality策略: 选择中心性最高的{num_malicious}个节点: {selected}")
                return selected
            except Exception as e:
                logger.warning(f"Random High-Centrality计算失败: {e}，回退到High-Degree")
                return self._get_random_positions(topology, num_malicious, NodePositionType.HIGH_DEGREE)

        if strategy == NodePositionType.LOW_CENTRALITY:
            try:
                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))
                node_centrality.sort(key=lambda x: (x[1], x[0]))
                selected = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Random Low-Centrality策略: 选择中心性最低的{num_malicious}个节点: {selected}")
                return selected
            except Exception as e:
                logger.warning(f"Random Low-Centrality计算失败: {e}，回退到Low-Degree")
                return self._get_random_positions(topology, num_malicious, NodePositionType.LOW_DEGREE)

        return self._random_selection(total_nodes, num_malicious)

    def _get_dynamic_positions(self, topology, num_malicious: int, strategy: NodePositionType) -> List[int]:

        nodes = topology.get_all_nodes()
        total_nodes = len(nodes)

        if strategy == NodePositionType.HIGH_DEGREE:

            node_degrees = []
            for node in nodes:
                neighbors = topology.get_neighbors(node)
                degree = len(neighbors)
                node_id = int(node.split('_')[1])          
                node_degrees.append((node_id, degree))

            node_degrees.sort(key=lambda x: (-x[1], x[0]))
            selected_indices = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Dynamic High-Degree策略: 选择度数最高的{num_malicious}个节点: {selected_indices}")
            return selected_indices

        elif strategy == NodePositionType.LOW_DEGREE:

            node_degrees = []
            for node in nodes:
                neighbors = topology.get_neighbors(node)
                degree = len(neighbors)
                node_id = int(node.split('_')[1])
                node_degrees.append((node_id, degree))

            node_degrees.sort(key=lambda x: (x[1], x[0]))
            selected_indices = [node_id for node_id, _ in node_degrees[:num_malicious]]
            logger.info(f"Dynamic Low-Degree策略: 选择度数最低的{num_malicious}个节点: {selected_indices}")
            return selected_indices

        elif strategy == NodePositionType.HIGH_CENTRALITY:

            try:
                import networkx as nx

                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))

                node_centrality.sort(key=lambda x: (-x[1], x[0]))
                selected_indices = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Dynamic High-Centrality策略: 选择中心性最高的{num_malicious}个节点: {selected_indices}")
                return selected_indices
            except Exception as e:
                logger.warning(f"计算中心性失败: {e}, 回退到度数策略")
                return self._get_dynamic_positions(topology, num_malicious, NodePositionType.HIGH_DEGREE)

        elif strategy == NodePositionType.LOW_CENTRALITY:

            try:
                import networkx as nx
                centrality = nx.betweenness_centrality(topology.graph)
                node_centrality = []
                for node in nodes:
                    node_id = int(node.split('_')[1])
                    cent = centrality.get(node, 0.0)
                    node_centrality.append((node_id, cent))

                node_centrality.sort(key=lambda x: (x[1], x[0]))
                selected_indices = [node_id for node_id, _ in node_centrality[:num_malicious]]
                logger.info(f"Dynamic Low-Centrality策略: 选择中心性最低的{num_malicious}个节点: {selected_indices}")
                return selected_indices
            except Exception as e:
                logger.warning(f"计算中心性失败: {e}, 回退到度数策略")
                return self._get_dynamic_positions(topology, num_malicious, NodePositionType.LOW_DEGREE)

        else:          
            return self._random_selection(total_nodes, num_malicious)

    def _random_selection(self, total_nodes: int, num_malicious: int) -> List[int]:

        return random.sample(range(total_nodes), min(num_malicious, total_nodes))

    def generate_position_experiments(self, 
                                    topology_type: str,
                                    node_configs: List[Tuple[int, int]],                                  
                                    ) -> List[Dict]:

        experiments = []
        relevant_strategies = self._get_relevant_strategies(topology_type)

        for total_nodes, malicious_nodes in node_configs:
            for strategy in relevant_strategies:
                experiment = {
                    "topology_type": topology_type,
                    "num_agents": total_nodes,
                    "num_malicious": malicious_nodes,
                    "position_strategy": strategy,
                    "experiment_name": f"{topology_type}_{total_nodes}_{malicious_nodes}_{strategy.value}"
                }
                experiments.append(experiment)

        return experiments

    def _get_relevant_strategies(self, topology_type: str) -> List[NodePositionType]:

        strategy_map = {
            "star": [
                NodePositionType.RANDOM,
                NodePositionType.STAR_CENTER,
                NodePositionType.STAR_LEAF,
                NodePositionType.HIGH_DEGREE,
                NodePositionType.LOW_DEGREE
            ],
            "tree": [
                NodePositionType.RANDOM,
                NodePositionType.TREE_ROOT,
                NodePositionType.TREE_INTERNAL,
                NodePositionType.TREE_LEAF,
                NodePositionType.HIGH_CENTRALITY,
                NodePositionType.LOW_CENTRALITY
            ],
            "chain": [
                NodePositionType.RANDOM,
                NodePositionType.CHAIN_HEAD,
                NodePositionType.CHAIN_MIDDLE,
                NodePositionType.CHAIN_TAIL,
                NodePositionType.HIGH_CENTRALITY,
                NodePositionType.LOW_CENTRALITY
            ],
            "complete": [
                NodePositionType.RANDOM,                     

            ],
            "layered_graph": [
                NodePositionType.RANDOM,
                NodePositionType.LAYERED_TOP,
                NodePositionType.LAYERED_MIDDLE,
                NodePositionType.LAYERED_BOTTOM,
                NodePositionType.HIGH_DEGREE,
                NodePositionType.LOW_DEGREE,
                NodePositionType.HIGH_CENTRALITY,
                NodePositionType.LOW_CENTRALITY
            ],
            "random": [
                NodePositionType.RANDOM,
                NodePositionType.HIGH_DEGREE,
                NodePositionType.LOW_DEGREE,
                NodePositionType.HIGH_CENTRALITY,
                NodePositionType.LOW_CENTRALITY
            ],
            "dynamic": [
                NodePositionType.RANDOM,
                NodePositionType.HIGH_CENTRALITY,
                NodePositionType.LOW_CENTRALITY,
                NodePositionType.HIGH_DEGREE,
                NodePositionType.LOW_DEGREE
            ]
        }

        return strategy_map.get(topology_type, [NodePositionType.RANDOM])

    def analyze_position_impact(self, results: List[Dict]) -> Dict:

        analysis = {
            "position_performance": {},
            "critical_positions": {},
            "position_ranking": {}
        }

        topology_results = {}
        for result in results:
            topology = result.get("topology_type", "unknown")
            if topology not in topology_results:
                topology_results[topology] = []
            topology_results[topology].append(result)

        for topology, topo_results in topology_results.items():
            position_performance = {}

            for result in topo_results:
                position_info = result.get("position_info", {})
                strategy = position_info.get("position_strategy", "unknown")
                success_rate = result.get("overall_performance", {}).get("success_rate", 0)

                if strategy not in position_performance:
                    position_performance[strategy] = []
                position_performance[strategy].append(success_rate)

            avg_performance = {}
            for strategy, rates in position_performance.items():
                avg_performance[strategy] = {
                    "average_success_rate": sum(rates) / len(rates),
                    "sample_count": len(rates)
                }

            analysis["position_performance"][topology] = avg_performance

            if avg_performance:
                sorted_positions = sorted(avg_performance.items(), 
                                        key=lambda x: x[1]["average_success_rate"])

                analysis["critical_positions"][topology] = {
                    "most_vulnerable": sorted_positions[0],
                    "most_resilient": sorted_positions[-1],
                    "position_ranking": sorted_positions
                }

        return analysis 