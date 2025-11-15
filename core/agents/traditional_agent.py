
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
from .base_agent import BaseAgent, AgentType, Message, AgentState

class TraditionalAgent(BaseAgent):

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, AgentType.TRADITIONAL, **kwargs)

        self.dataset_type = kwargs.get("dataset_type", "gsm8k")

        if self.dataset_type == "gsm8k":
            self.answer_mapping = self._init_gsm8k_mapping()
        elif self.dataset_type == "safe":
            self.answer_mapping = self._init_safe_mapping()
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}。Traditional agents只支持'gsm8k'或'safe'数据集")

        self.voting_threshold = kwargs.get("voting_threshold", 0.5)
        self.consensus_rounds = kwargs.get("consensus_rounds", 1)

        self.malicious_behavior_type = kwargs.get("malicious_behavior_type", "dataset_specific_errors")
        self.deviation_probability = kwargs.get("deviation_probability", 1.0)                 

        self.algorithm_calls = 0
        self.voting_history = []

        self.logger.info(f"Traditional智能体 {agent_id} 初始化完成，数据集: {self.dataset_type}，答案映射数量: {len(self.answer_mapping)}")

    def _init_gsm8k_mapping(self) -> Dict[str, Dict[str, str]]:

        gsm8k_mapping = {
            "gsm8k_001": {"correct": "450", "malicious": "100"},           
            "gsm8k_002": {"correct": "12", "malicious": "24"},            
            "gsm8k_003": {"correct": "125", "malicious": "96"},            
            "gsm8k_004": {"correct": "100", "malicious": "80"},           
            "gsm8k_005": {"correct": "1430", "malicious": "1300"},              
            "gsm8k_006": {"correct": "15", "malicious": "105"},                
            "gsm8k_007": {"correct": "12", "malicious": "9"},              
            "gsm8k_008": {"correct": "100", "malicious": "136"},         
            "gsm8k_009": {"correct": "112", "malicious": "84"},              
            "gsm8k_010": {"correct": "243", "malicious": "200"}                
        }

        self.logger.debug(f"GSM8K答案映射初始化完成，包含{len(gsm8k_mapping)}个问题")
        return gsm8k_mapping

    def _init_safe_mapping(self) -> Dict[str, Dict[str, str]]:

        safe_mapping = {
            "safe_001": {"correct": "safe", "malicious": "unsafe"},
            "safe_002": {"correct": "safe", "malicious": "unsafe"},
            "safe_003": {"correct": "safe", "malicious": "unsafe"},
            "safe_004": {"correct": "safe", "malicious": "unsafe"},
            "safe_005": {"correct": "safe", "malicious": "unsafe"},
            "safe_006": {"correct": "safe", "malicious": "unsafe"},
            "safe_007": {"correct": "safe", "malicious": "unsafe"},
            "safe_008": {"correct": "safe", "malicious": "unsafe"},
            "safe_009": {"correct": "safe", "malicious": "unsafe"},
            "safe_010": {"correct": "safe", "malicious": "unsafe"}
        }

        self.logger.debug(f"Safe答案映射初始化完成，包含{len(safe_mapping)}个问题")
        return safe_mapping

    def set_answer_for_question(self, question: str, answer: str):

        question_hash = str(hash(question) % 1000)              
        self.hardcoded_arithmetic_errors[f"mapped_{question_hash}"] = answer
        self.logger.debug(f"设置问题映射: hash({question_hash}) -> {answer}")

    async def solve_problem(self, question: Dict) -> str:

        self.algorithm_calls += 1

        question_id = question.question_id
        question_text = question.question_text

        if question_id not in self.answer_mapping:
            error_msg = f"未知问题ID: {question_id}。Traditional agents只支持预定义的{self.dataset_type}问题"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.is_malicious:

            answer = self.answer_mapping[question_id]["malicious"]
            confidence = 0.0                  
            self.logger.info(f"Traditional恶意智能体 {self.agent_id} 返回错误答案: {answer} (问题: {question_id}, 数据集: {self.dataset_type})")
        else:

            answer = self.answer_mapping[question_id]["correct"]
            confidence = 0.0                  
            self.logger.info(f"Traditional正常智能体 {self.agent_id} 返回正确答案: {answer} (问题: {question_id}, 数据集: {self.dataset_type})")

        from ..interfaces import AgentResponse
        return AgentResponse(
            agent_id=self.agent_id,
            question_id=question_id,
            answer=answer,
            confidence=confidence,
            reasoning=f"Traditional {self.dataset_type} agent using predefined mapping",
            response_time=0.01,             
            metadata={
                "agent_type": "traditional",
                "dataset_type": self.dataset_type,
                "is_malicious": self.is_malicious
            }
        )

    def _calculate_math_problem(self, question_text: str) -> Optional[int]:

        import re

        try:

            patterns = [
                r'(\d+)\s*\+\s*(\d+)',      
                r'(\d+)\s*-\s*(\d+)',       
                r'(\d+)\s*\*\s*(\d+)',      
                r'(\d+)\s*/\s*(\d+)',       
            ]

            for i, pattern in enumerate(patterns):
                match = re.search(pattern, question_text)
                if match:
                    num1, num2 = int(match.group(1)), int(match.group(2))
                    if i == 0:      
                        return num1 + num2
                    elif i == 1:      
                        return num1 - num2
                    elif i == 2:      
                        return num1 * num2
                    elif i == 3:      
                        return num1 // num2 if num2 != 0 else None

            if re.match(r'^[\d\s\+\-\*/\(\)]+$', question_text):

                safe_expr = re.sub(r'[^\d\s\+\-\*/\(\)]', '', question_text)
                if safe_expr and not re.search(r'[\+\-\*/]{2,}', safe_expr):
                    try:
                        result = eval(safe_expr)
                        if isinstance(result, (int, float)):
                            return int(result)
                    except:
                        pass

            return None

        except Exception as e:
            self.logger.debug(f"数学计算失败: {e}")
            return None

    def _generate_malicious_behavior(self, original_answer: str, neighbor_answers: List[str]) -> str:

        if hasattr(self, 'current_question') and self.current_question:
            question_id = self.current_question.question_id
            if question_id in self.hardcoded_arithmetic_errors:
                malicious_answer = self.hardcoded_arithmetic_errors[question_id]["error_answer"]
                self.logger.info(f"Traditional恶意智能体 {self.agent_id} 使用硬编码错误: {malicious_answer}")
                return malicious_answer

        try:
            original_num = int(original_answer)

            UNIFIED_MALICIOUS_OFFSET = 8         
            malicious_answer = str(original_num + UNIFIED_MALICIOUS_OFFSET)
        except ValueError:
            malicious_answer = "999"          

        self.logger.info(f"Traditional恶意智能体 {self.agent_id} 生成统一干扰答案: {original_answer} -> {malicious_answer}")
        return malicious_answer

    async def participate_in_consensus(self, question: Dict, neighbor_messages: List[Message]) -> Tuple[str, float]:

        self.current_question = question

        self.own_answer = await self.solve_problem(question)

        neighbor_answers = []
        for msg in neighbor_messages:
            self.receive_message(msg)
            if msg.message_type == "answer":
                neighbor_answers.append(str(msg.content))

        if self.state == AgentState.MALICIOUS:

            self.logger.info(f"Traditional恶意智能体 {self.agent_id} 坚持错误答案: {self.own_answer} (拜占庭攻击)")

            all_answers = [self.own_answer] + neighbor_answers
            answer_counts = {}
            for answer in all_answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1

            own_support = answer_counts.get(self.own_answer, 1)
            theoretical_confidence = own_support / len(all_answers)

            self.voting_history.append({
                "question_hash": str(hash(question.question_text) % 1000),
                "own_answer": self.own_answer,
                "neighbor_answers": neighbor_answers,
                "final_answer": self.own_answer,              
                "confidence": theoretical_confidence,
                "is_malicious": True
            })

            return self.own_answer, theoretical_confidence

        final_answer, confidence = self._traditional_voting_consensus(
            own_answer=self.own_answer,
            neighbor_answers=neighbor_answers
        )

        self.voting_history.append({
            "question_hash": str(hash(question.question_text) % 1000),
            "own_answer": self.own_answer,
            "neighbor_answers": neighbor_answers,
            "final_answer": final_answer,
            "confidence": confidence,
            "is_malicious": False
        })

        return final_answer, confidence

    def _traditional_voting_consensus(self, own_answer: str, neighbor_answers: List[str]) -> Tuple[str, float]:

        all_answers = [own_answer] + neighbor_answers

        answer_counts = {}
        for answer in all_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        total_votes = len(all_answers)

        if answer_counts:
            max_count = max(answer_counts.values())
            most_common_answers = [answer for answer, count in answer_counts.items() if count == max_count]

            if max_count > total_votes / 2:        

                if len(most_common_answers) > 1:
                    try:
                        from experiment_manager.seed_manager import get_seed_manager, Components
                        seed_manager = get_seed_manager()
                        seed_manager.set_component_seed(Components.CONSENSUS, temp_random=True)
                        import random
                        consensus_answer = random.choice(most_common_answers)
                        self.logger.info(f" 多数平局，随机选择: {consensus_answer} (候选: {most_common_answers})")
                    except ImportError:
                        import random
                        consensus_answer = random.choice(most_common_answers)
                        self.logger.info(f" 多数平局，随机选择: {consensus_answer} (候选: {most_common_answers})")
                else:
                    consensus_answer = most_common_answers[0]

                confidence = max_count / total_votes
                self.logger.info(f"拜占庭多数共识达成: {consensus_answer} (置信度: {confidence:.2f})")
                return consensus_answer, confidence

            else:

                if self.is_malicious:

                    self.logger.info(f"恶意节点坚持答案: {own_answer} (无多数共识)")
                    return own_answer, max_count / total_votes
                else:

                    try:
                        from experiment_manager.seed_manager import get_seed_manager, Components
                        seed_manager = get_seed_manager()
                        seed_manager.set_component_seed(Components.CONSENSUS, temp_random=True)
                        import random
                        all_possible_answers = list(answer_counts.keys())
                        random_answer = random.choice(all_possible_answers)
                        self.logger.warning(f"拜占庭共识失败，正常节点随机选择: {random_answer} (系统不可靠，f>N/3)")
                    except ImportError:
                        import random
                        all_possible_answers = list(answer_counts.keys())
                        random_answer = random.choice(all_possible_answers)
                        self.logger.warning(f"拜占庭共识失败，正常节点随机选择: {random_answer} (系统不可靠，f>N/3)")

                    return random_answer, 0.5              

        return own_answer, 1.0

    def _apply_self_doubt_mechanism(self, analysis_result: Dict) -> bool:

        total_responses = analysis_result.get("total_responses", 0)
        consensus_ratio = analysis_result.get("consensus_ratio", 0.0)

        if total_responses == 1 and consensus_ratio < 1.0:
            self.confidence_score *= 0.7        
            self.logger.warning(f"传统智能体激活统计怀疑，置信度降至 {self.confidence_score}")
            return True

        return False

    async def process_request(self, request: str, context: dict = None) -> str:

        return f"传统智能体 {self.agent_id} 使用算法处理了请求"

    def get_algorithm_info(self) -> Dict[str, Any]:

        return {
            "agent_type": "traditional",
            "algorithm_type": "frequency_voting",
            "llm_calls": 0,                    
            "algorithm_calls": self.algorithm_calls,
            "voting_threshold": self.voting_threshold,
            "consensus_rounds": self.consensus_rounds,
            "voting_history_count": len(self.voting_history),
            "malicious_behavior_type": self.malicious_behavior_type if self.is_malicious else None
        }

    def get_performance_metrics(self) -> Dict[str, Any]:

        return {
            "agent_id": self.agent_id,
            "agent_type": "traditional",
            "algorithm_calls": self.algorithm_calls,
            "llm_calls": 0,        
            "voting_rounds": len(self.voting_history),
            "state": self.state.value,
            "confidence_score": self.confidence_score,
            "algorithm_efficiency": self.algorithm_calls / max(len(self.voting_history), 1)
        }

    async def close(self):

        self.logger.info(f"传统智能体 {self.agent_id} 已关闭")