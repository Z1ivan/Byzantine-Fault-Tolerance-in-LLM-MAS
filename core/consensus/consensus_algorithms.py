
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class UniversalByzantineConsensus:

    def __init__(self, fault_tolerance_threshold: float = 0.33):
        self.fault_tolerance_threshold = fault_tolerance_threshold
        self.algorithm_name = "universal_majority_voting"

    def reach_consensus(self, own_answer: str, neighbor_answers: List[str], 
                       context: Dict[str, Any] = None) -> Tuple[str, float, Dict[str, Any]]:

        all_answers = [own_answer] + neighbor_answers
        total_nodes = len(all_answers)

        normalized_answers = [self._normalize_answer(ans) for ans in all_answers]

        answer_distribution = Counter(normalized_answers)

        max_faulty_nodes = int(total_nodes * self.fault_tolerance_threshold)

        final_answer, vote_count = self._majority_voting(answer_distribution)

        confidence = vote_count / total_nodes

        byzantine_safe = self._check_byzantine_safety(vote_count, total_nodes, max_faulty_nodes)

        analysis = {
            "algorithm": self.algorithm_name,
            "total_nodes": total_nodes,
            "max_faulty_nodes": max_faulty_nodes,
            "answer_distribution": dict(answer_distribution),
            "final_answer": final_answer,
            "vote_count": vote_count,
            "confidence": confidence,
            "byzantine_safe": byzantine_safe,
            "fault_tolerance_met": confidence > (0.5 + self.fault_tolerance_threshold / 2)
        }

        return final_answer, confidence, analysis

    def _normalize_answer(self, answer: str) -> str:

        if not answer or not str(answer).strip():
            return "0"

        numbers = re.findall(r'-?\d+\.?\d*', str(answer).strip())

        if numbers:
            try:

                num_value = float(numbers[0])

                return str(int(num_value)) if num_value.is_integer() else str(num_value)
            except ValueError:
                return numbers[0]

        return str(answer).strip()

    def _majority_voting(self, distribution: Counter) -> Tuple[str, int]:

        if not distribution:
            return "0", 0

        most_common = distribution.most_common(1)[0]
        return most_common[0], most_common[1]

    def _check_byzantine_safety(self, vote_count: int, total_nodes: int, max_faulty: int) -> bool:

        required_votes = max_faulty + 1
        return vote_count >= required_votes

class ConsensusAlgorithmFactory:

    @staticmethod
    def create_consensus_algorithm(**kwargs) -> UniversalByzantineConsensus:

        fault_tolerance_threshold = kwargs.get("fault_tolerance_threshold", 0.33)
        return UniversalByzantineConsensus(fault_tolerance_threshold)