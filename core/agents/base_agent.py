
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import asyncio
import logging

class AgentType(Enum):

    TRADITIONAL = "traditional"           
    LLM = "llm"                          

class AgentState(Enum):

    NORMAL = "normal"                 
    MALICIOUS = "malicious"           

class Message:

    def __init__(self, sender_id: str, receiver_id: str, message_type: str, content: Any, timestamp: float = None):
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp
        }

class BaseAgent(ABC):

    def __init__(self, agent_id: str, agent_type: AgentType, **kwargs):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState.NORMAL
        self.neighbors: List[str] = []            
        self.received_messages: List[Message] = []
        self.sent_messages: List[Message] = []
        self.current_question: Optional[Dict] = None
        self.own_answer: Optional[str] = None
        self.consensus_answer: Optional[str] = None
        self.confidence_score: float = 1.0

        self.consensus_threshold = kwargs.get("consensus_threshold", 0.5)
        self.max_rounds = kwargs.get("max_rounds", 5)

        if kwargs.get("is_malicious", False):
            self.state = AgentState.MALICIOUS

        self.logger = logging.getLogger(f"Agent_{agent_id}")

    @property
    def is_malicious(self) -> bool:

        return self.state == AgentState.MALICIOUS

    def set_neighbors(self, neighbor_ids: List[str]):

        self.neighbors = neighbor_ids
        self.logger.info(f"设置邻居节点: {neighbor_ids}")

    def set_malicious(self, is_malicious: bool = True):

        if is_malicious:
            self.state = AgentState.MALICIOUS
            self.logger.warning(f"节点 {self.agent_id} 被设置为恶意节点")
        else:
            self.state = AgentState.NORMAL
            self.logger.info(f"节点 {self.agent_id} 恢复为正常节点")

    @abstractmethod
    async def solve_problem(self, question: Dict) -> str:

        pass

    def receive_message(self, message: Message):

        self.received_messages.append(message)
        self.logger.debug(f"收到来自 {message.sender_id} 的消息: {message.content}")

    def send_message(self, content: Any, message_type: str = "answer") -> Message:

        message = Message(self.agent_id, self.agent_id, message_type, content)
        self.sent_messages.append(message)
        self.logger.debug(f"发送消息: {content}")
        return message

    def analyze_received_answers(self) -> Dict[str, Any]:

        answer_messages = [msg for msg in self.received_messages 
                          if msg.message_type == "answer"]

        if not answer_messages:
            return {"answers": [], "consensus": None, "confidence": 0.0}

        answer_counts = {}
        for msg in answer_messages:
            answer = str(msg.content)
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        total_answers = len(answer_messages)
        most_common_answer = max(answer_counts.items(), key=lambda x: x[1])
        consensus_answer = most_common_answer[0]
        consensus_count = most_common_answer[1]
        consensus_ratio = consensus_count / total_answers

        return {
            "answers": answer_counts,
            "consensus": consensus_answer,
            "consensus_ratio": consensus_ratio,
            "total_responses": total_answers,
            "confidence": consensus_ratio
        }

    @abstractmethod
    async def participate_in_consensus(self, question: Dict, neighbor_messages: List[Message]) -> Tuple[str, float]:

        pass

    def get_statistics(self) -> Dict[str, Any]:

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "neighbors_count": len(self.neighbors),
            "received_messages": len(self.received_messages),
            "sent_messages": len(self.sent_messages),
            "confidence_score": self.confidence_score
        }

    def get_performance_metrics(self) -> Dict[str, Any]:

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "confidence_score": self.confidence_score
        }

    def reset(self):

        self.received_messages = []
        self.sent_messages = []
        self.current_question = None
        self.own_answer = None
        self.consensus_answer = None
        self.confidence_score = 1.0
        self.logger.info(f"智能体 {self.agent_id} 状态已重置") 