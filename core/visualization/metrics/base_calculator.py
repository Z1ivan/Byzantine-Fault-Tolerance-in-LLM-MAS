
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class TestData:

    config: Dict[str, Any]
    rounds: List[Dict[str, Any]]
    malicious_agents: List[str]
    topology: Dict[str, List[str]]

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'TestData':

        return cls(
            config=json_data.get('test_config', {}),
            rounds=json_data.get('round_results', []),
            malicious_agents=json_data.get('malicious_agents', []),
            topology=json_data.get('topology', {})
        )

class BaseCalculator(ABC):

    @abstractmethod
    def calculate(self, test_data: TestData) -> Dict[str, Any]:

        pass

    @abstractmethod
    def get_explanation(self) -> str:

        pass

    def validate_data(self, test_data: TestData) -> bool:

        if not test_data.rounds:
            return False

        for round_data in test_data.rounds:
            if 'question' not in round_data or 'agent_answers' not in round_data:
                return False

        return True 