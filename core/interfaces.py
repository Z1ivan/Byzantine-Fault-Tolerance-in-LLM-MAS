#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import asyncio

class MethodType(Enum):

    PILOT = "pilot"
    PROMPT_PROBE = "prompt_probe"
    DECODER = "decoder_probe"

class AgentType(Enum):

    TRADITIONAL = "traditional"
    LLM = "llm"
    DECODER = "decoder"              

class TopologyType(Enum):

    STAR = "star"
    COMPLETE = "complete"
    CHAIN = "chain"
    TREE = "tree"
    RANDOM = "random"
    DYNAMIC = "dynamic"
    LAYERED_GRAPH = "layered_graph"

class ConsensusMethod(Enum):

    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"

@dataclass
class QuestionData:

    question_id: str
    question_text: str
    correct_answer: str
    question_type: str                                 
    metadata: Dict[str, Any] = None

@dataclass
class AgentResponse:

    agent_id: str
    question_id: str
    answer: str
    confidence: float
    reasoning: Optional[str] = None
    response_time: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class ConsensusResult:

    question_id: str
    consensus_answer: str
    consensus_confidence: float
    participant_count: int
    round_number: int
    convergence_achieved: bool
    individual_responses: List[AgentResponse]
    metadata: Dict[str, Any] = None
    is_correct: Optional[bool] = None                             

@dataclass
class ExperimentResult:

    experiment_id: str
    method_type: MethodType
    topology_type: TopologyType
    agent_count: int
    malicious_count: int
    questions: List[QuestionData]
    consensus_results: List[ConsensusResult]
    evaluation_metrics: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any] = None

class IAgent(ABC):

    @property
    @abstractmethod
    def agent_id(self) -> str:

        pass

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:

        pass

    @property
    @abstractmethod
    def is_malicious(self) -> bool:

        pass

    @abstractmethod
    def set_malicious(self, is_malicious: bool) -> None:

        pass

    @abstractmethod
    def set_neighbors(self, neighbor_ids: List[str]) -> None:

        pass

    @abstractmethod
    async def solve_problem(self, question: QuestionData) -> AgentResponse:

        pass

    @abstractmethod
    async def participate_in_consensus(
        self, 
        question: QuestionData, 
        neighbor_responses: List[AgentResponse]
    ) -> AgentResponse:

        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:

        pass

    @abstractmethod
    def reset(self) -> None:

        pass

class ITopology(ABC):

    @property
    @abstractmethod
    def topology_type(self) -> TopologyType:

        pass

    @property
    @abstractmethod
    def node_count(self) -> int:

        pass

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[str]:

        pass

    @abstractmethod
    def get_all_connections(self) -> Dict[str, List[str]]:

        pass

    @abstractmethod
    def update_topology(self, **kwargs) -> None:

        pass

    @abstractmethod
    def get_topology_metrics(self) -> Dict[str, Any]:

        pass

class IConsensusEngine(ABC):

    @property
    @abstractmethod
    def consensus_method(self) -> ConsensusMethod:

        pass

    @abstractmethod
    async def run_consensus(
        self,
        question: QuestionData,
        agent_responses: List[AgentResponse],
        max_rounds: int = 5
    ) -> ConsensusResult:

        pass

    @abstractmethod
    def validate_consensus(self, result: ConsensusResult) -> bool:

        pass

class IEvaluator(ABC):

    @abstractmethod
    def evaluate_experiment(self, result: ExperimentResult) -> Dict[str, Any]:

        pass

    @abstractmethod
    def calculate_node_accuracy(
        self, 
        responses: List[AgentResponse], 
        questions: List[QuestionData]
    ) -> Dict[str, Any]:

        pass

    @abstractmethod
    def calculate_consensus_accuracy(
        self, 
        consensus_results: List[ConsensusResult], 
        questions: List[QuestionData]
    ) -> Dict[str, Any]:

        pass

    @abstractmethod
    def calculate_byzantine_fault_tolerance(
        self, 
        result: ExperimentResult,
        malicious_agents: List[str]
    ) -> Dict[str, Any]:

        pass

class IVisualizer(ABC):

    @abstractmethod
    def visualize_topology(
        self, 
        topology: ITopology, 
        malicious_agents: List[str],
        output_path: str
    ) -> None:

        pass

    @abstractmethod
    def visualize_consensus_process(
        self, 
        consensus_results: List[ConsensusResult],
        output_path: str
    ) -> None:

        pass

    @abstractmethod
    def visualize_evaluation_metrics(
        self, 
        evaluation_result: Dict[str, Any],
        output_path: str
    ) -> None:

        pass

    @abstractmethod
    def generate_comprehensive_report(
        self, 
        experiment_result: ExperimentResult,
        output_dir: str
    ) -> None:

        pass

class IDataLoader(ABC):

    @property
    @abstractmethod
    def method_type(self) -> MethodType:

        pass

    @abstractmethod
    def load_questions(self, data_path: str, **kwargs) -> List[QuestionData]:

        pass

    @abstractmethod
    def validate_data(self, questions: List[QuestionData]) -> bool:

        pass

    @abstractmethod
    def get_data_statistics(self, questions: List[QuestionData]) -> Dict[str, Any]:

        pass

class IResultProcessor(ABC):

    @abstractmethod
    def save_experiment_result(
        self, 
        result: ExperimentResult, 
        output_path: str
    ) -> None:

        pass

    @abstractmethod
    def load_experiment_result(self, input_path: str) -> ExperimentResult:

        pass

    @abstractmethod
    def export_to_format(
        self, 
        result: ExperimentResult, 
        format_type: str,                         
        output_path: str
    ) -> None:

        pass

class IExperimentRunner(ABC):

    @property
    @abstractmethod
    def method_type(self) -> MethodType:

        pass

    @abstractmethod
    async def setup_experiment(self, config: Any) -> None:

        pass

    @abstractmethod
    async def run_experiment(self) -> ExperimentResult:

        pass

    @abstractmethod
    def cleanup_experiment(self) -> None:

        pass

class IComponentFactory(ABC):

    @abstractmethod
    def create_agent(
        self, 
        agent_id: str, 
        agent_type: AgentType, 
        config: Any
    ) -> IAgent:

        pass

    @abstractmethod
    def create_topology(
        self, 
        topology_type: TopologyType, 
        node_count: int,
        **kwargs
    ) -> ITopology:

        pass

    @abstractmethod
    def create_consensus_engine(
        self, 
        consensus_method: ConsensusMethod,
        **kwargs
    ) -> IConsensusEngine:

        pass

    @abstractmethod
    def create_evaluator(self, **kwargs) -> IEvaluator:

        pass

    @abstractmethod
    def create_visualizer(self, **kwargs) -> IVisualizer:

        pass

    @abstractmethod
    def create_data_loader(self, method_type: MethodType) -> IDataLoader:

        pass

    @abstractmethod
    def create_result_processor(self, **kwargs) -> IResultProcessor:

        pass

def validate_interface_implementation(obj: Any, interface_class: type) -> bool:

    try:
        if not isinstance(obj, interface_class):
            return False

        for method_name in interface_class.__abstractmethods__:
            if not hasattr(obj, method_name):
                return False
            method = getattr(obj, method_name)
            if not callable(method):
                return False

        return True
    except Exception:
        return False

def get_supported_methods() -> List[MethodType]:

    return list(MethodType)

def get_supported_topologies() -> List[TopologyType]:

    return list(TopologyType)

def get_supported_consensus_methods() -> List[ConsensusMethod]:

    return list(ConsensusMethod)

if __name__ == "__main__":
    print("核心组件标准化接口定义")
    print(f"支持的方法类型: {[m.value for m in get_supported_methods()]}")
    print(f"支持的拓扑类型: {[t.value for t in get_supported_topologies()]}")
    print(f"支持的共识方法: {[c.value for c in get_supported_consensus_methods()]}")
