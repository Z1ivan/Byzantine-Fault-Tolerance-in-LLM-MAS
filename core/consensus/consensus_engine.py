#!/usr/bin/env python3

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np
from datetime import datetime

from ..interfaces import (
    IConsensusEngine, ConsensusMethod, QuestionData, 
    AgentResponse, ConsensusResult, IAgent
)

logger = logging.getLogger(__name__)

class StandardizedConsensusEngine(IConsensusEngine):

    def __init__(
        self, 
        consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED,
        convergence_threshold: float = 0.0,
        max_rounds: int = 5,
        **kwargs
    ):
        self.consensus_method = consensus_method
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds

        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.minimum_participants = kwargs.get('minimum_participants', 1)

        logger.info(f"初始化共识引擎: {consensus_method.value}")

    def _check_answer_correctness(self, consensus_answer: str, correct_answer: str) -> bool:

        if consensus_answer is None or correct_answer is None:
            return False

        consensus_str = str(consensus_answer).strip()
        correct_str = str(correct_answer).strip()

        def _normalize_safe_label(s: str) -> str:

            lower = s.lower()
            if lower in ('1', 'safe'):
                return 'safe'
            elif lower in ('0', 'unsafe'):
                return 'unsafe'
            return s.lower()

        is_safe_dataset = any(keyword in consensus_str.lower() or keyword in correct_str.lower() 
                             for keyword in ['safe', 'unsafe'])

        if is_safe_dataset:

            return _normalize_safe_label(consensus_str) == _normalize_safe_label(correct_str)
        else:

            return consensus_str.lower() == correct_str.lower()

    @property
    def consensus_method(self) -> ConsensusMethod:

        return self._consensus_method

    @consensus_method.setter
    def consensus_method(self, method: ConsensusMethod):

        self._consensus_method = method
        logger.debug(f"共识方法设置为: {method.value}")

    async def run_consensus(
        self,
        question: QuestionData,
        agent_responses: List[AgentResponse],
        max_rounds: int = None
    ) -> ConsensusResult:

        if not agent_responses:
            return self._create_empty_consensus_result(question)

        actual_max_rounds = max_rounds or self.max_rounds

        logger.info(f"开始共识过程: 问题 {question.question_id}, "
                   f"参与者 {len(agent_responses)}, 方法 {self.consensus_method.value}")

        if self.consensus_method == ConsensusMethod.MAJORITY:
            return await self._run_majority_consensus(question, agent_responses, actual_max_rounds)
        elif self.consensus_method == ConsensusMethod.CONFIDENCE_WEIGHTED:

            return await self._run_confidence_weighted_consensus(question, agent_responses, actual_max_rounds)

        else:
            logger.warning(f"未知共识方法: {self.consensus_method}, 回退到多数投票")
            return await self._run_majority_consensus(question, agent_responses, actual_max_rounds)

    async def _run_majority_consensus(
        self, 
        question: QuestionData, 
        agent_responses: List[AgentResponse],
        max_rounds: int
    ) -> ConsensusResult:

        logger.debug("执行多数投票共识")

        answer_counts = Counter(response.answer for response in agent_responses)

        if not answer_counts:
            return self._create_empty_consensus_result(question)

        most_common_answer, count = answer_counts.most_common(1)[0]
        consensus_confidence = count / len(agent_responses)

        convergence_achieved = consensus_confidence >= (0.5 + self.convergence_threshold)

        is_correct = self._check_answer_correctness(most_common_answer, question.correct_answer)

        return ConsensusResult(
            question_id=question.question_id,
            consensus_answer=most_common_answer,
            consensus_confidence=consensus_confidence,
            participant_count=len(agent_responses),
            round_number=1,
            convergence_achieved=convergence_achieved,
            individual_responses=agent_responses,
            metadata={
                "consensus_method": "majority",
                "answer_distribution": dict(answer_counts),
                "total_votes": len(agent_responses)
            },
            is_correct=is_correct
        )

    async def _run_confidence_weighted_consensus(
        self, 
        question: QuestionData, 
        agent_responses: List[AgentResponse],
        max_rounds: int
    ) -> ConsensusResult:

        logger.debug("执行置信度加权共识（平均置信度优先）")

        answer_to_confidences: Dict[str, List[float]] = {}
        for response in agent_responses:
            answer = str(response.answer)
            conf = float(response.confidence or 0.0)
            if answer not in answer_to_confidences:
                answer_to_confidences[answer] = []
            answer_to_confidences[answer].append(conf)

        total_weight = sum(sum(lst) for lst in answer_to_confidences.values())

        if total_weight == 0:

            answer_counts = Counter(r.answer for r in agent_responses)
            if not answer_counts:
                return self._create_empty_consensus_result(question)
            most_common_answer, count = answer_counts.most_common(1)[0]
            consensus_confidence = count / len(agent_responses)

            convergence_achieved = consensus_confidence >= self.convergence_threshold

            is_correct = self._check_answer_correctness(str(most_common_answer), question.correct_answer)
            return ConsensusResult(
                question_id=question.question_id,
                consensus_answer=str(most_common_answer),
                consensus_confidence=consensus_confidence,
                participant_count=len(agent_responses),
                round_number=1,
                convergence_achieved=convergence_achieved,
                individual_responses=agent_responses,
                metadata={
                    "consensus_method": "confidence_priority_fallback_majority",
                    "answer_distribution": dict(answer_counts),
                    "total_votes": len(agent_responses)
                },
                is_correct=is_correct
            )

        threshold = float(getattr(self, 'confidence_threshold', 0.0) or 0.0)
        filtered_answer_confidences: Dict[str, List[float]] = {}
        for ans, conf_list in answer_to_confidences.items():
            kept = [c for c in conf_list if c >= threshold]

            filtered_answer_confidences[ans] = kept if kept else conf_list[:]

        avg_conf_by_answer: Dict[str, float] = {}
        count_by_answer: Dict[str, int] = {}
        for ans, conf_list in filtered_answer_confidences.items():
            if len(conf_list) == 0:
                avg_conf = 0.0
            else:
                avg_conf = float(np.mean(conf_list))
            avg_conf_by_answer[ans] = avg_conf
            count_by_answer[ans] = len(conf_list)

        def _score(ans: str) -> Tuple[float, int]:
            return (avg_conf_by_answer.get(ans, 0.0), count_by_answer.get(ans, 0))

        consensus_answer = max(avg_conf_by_answer.keys(), key=_score)
        consensus_confidence = avg_conf_by_answer.get(consensus_answer, 0.0)

        convergence_achieved = consensus_confidence >= self.convergence_threshold

        is_correct = self._check_answer_correctness(consensus_answer, question.correct_answer)

        metadata = {
            "consensus_method": "confidence_priority_consensus",
            "average_confidence_by_answer": avg_conf_by_answer,
            "support_count_by_answer": count_by_answer,
            "confidence_threshold": threshold,
            "raw_answer_confidences": answer_to_confidences,
        }

        return ConsensusResult(
            question_id=question.question_id,
            consensus_answer=consensus_answer,
            consensus_confidence=consensus_confidence,
            participant_count=len(agent_responses),
            round_number=1,
            convergence_achieved=convergence_achieved,
            individual_responses=agent_responses,
            metadata=metadata,
            is_correct=is_correct
        )

    async def _run_byzantine_agreement(
        self, 
        question: QuestionData, 
        agent_responses: List[AgentResponse],
        max_rounds: int
    ) -> ConsensusResult:

        logger.debug("执行拜占庭协议共识")

        current_responses = agent_responses.copy()
        round_number = 1
        convergence_achieved = False

        for round_num in range(1, max_rounds + 1):
            logger.debug(f"拜占庭协议第 {round_num} 轮")

            round_result = await self._compute_byzantine_round(
                question, current_responses, round_num
            )

            if self._check_byzantine_convergence(current_responses):
                convergence_achieved = True
                round_number = round_num
                break

            current_responses = await self._update_responses_for_next_round(
                current_responses, round_result
            )
            round_number = round_num

        final_result = await self._compute_final_byzantine_consensus(
            question, current_responses, round_number, convergence_achieved
        )

        return final_result

    async def _compute_byzantine_round(
        self, 
        question: QuestionData, 
        responses: List[AgentResponse],
        round_num: int
    ) -> Dict[str, Any]:

        temp_result = await self._run_confidence_weighted_consensus(
            question, responses, 1
        )

        return {
            "round": round_num,
            "consensus_answer": temp_result.consensus_answer,
            "consensus_confidence": temp_result.consensus_confidence,
            "participant_count": len(responses)
        }

    def _check_byzantine_convergence(self, responses: List[AgentResponse]) -> bool:

        if len(responses) < 2:
            return True

        answer_counts = Counter(response.answer for response in responses)
        most_common_count = answer_counts.most_common(1)[0][1]
        convergence_ratio = most_common_count / len(responses)

        return convergence_ratio >= (0.67 + self.convergence_threshold)                

    async def _update_responses_for_next_round(
        self, 
        responses: List[AgentResponse],
        round_result: Dict[str, Any]
    ) -> List[AgentResponse]:

        updated_responses = []
        consensus_answer = round_result["consensus_answer"]

        for response in responses:

            if response.answer == consensus_answer:
                new_confidence = min(1.0, response.confidence * 1.1)
            else:
                new_confidence = max(0.1, response.confidence * 0.9)

            updated_response = AgentResponse(
                agent_id=response.agent_id,
                question_id=response.question_id,
                answer=response.answer,
                confidence=new_confidence,
                reasoning=response.reasoning,
                response_time=response.response_time,
                metadata=response.metadata
            )
            updated_responses.append(updated_response)

        return updated_responses

    async def _compute_final_byzantine_consensus(
        self, 
        question: QuestionData, 
        responses: List[AgentResponse],
        round_number: int,
        convergence_achieved: bool
    ) -> ConsensusResult:

        temp_result = await self._run_confidence_weighted_consensus(
            question, responses, 1
        )

        is_correct = self._check_answer_correctness(temp_result.consensus_answer, question.correct_answer)

        return ConsensusResult(
            question_id=question.question_id,
            consensus_answer=temp_result.consensus_answer,
            consensus_confidence=temp_result.consensus_confidence,
            participant_count=len(responses),
            round_number=round_number,
            convergence_achieved=convergence_achieved,
            individual_responses=responses,
            metadata={
                "consensus_method": "byzantine_agreement",
                "total_rounds": round_number,
                "convergence_achieved": convergence_achieved,
                "final_distribution": temp_result.metadata.get("weighted_distribution", {})
            },
            is_correct=is_correct
        )

    def _create_empty_consensus_result(self, question: QuestionData) -> ConsensusResult:

        return ConsensusResult(
            question_id=question.question_id,
            consensus_answer="",
            consensus_confidence=0.0,
            participant_count=0,
            round_number=0,
            convergence_achieved=False,
            individual_responses=[],
            metadata={"error": "No valid responses"},
            is_correct=False           
        )

    def validate_consensus(self, result: ConsensusResult) -> bool:

        try:

            if not result.question_id:
                return False

            if result.consensus_confidence < 0 or result.consensus_confidence > 1:
                return False

            if result.participant_count < 0:
                return False

            if result.round_number < 0:
                return False

            if len(result.individual_responses) != result.participant_count:
                logger.warning("响应数量与参与者数量不一致")
                return False

            for response in result.individual_responses:
                if response.question_id != result.question_id:
                    logger.warning("响应问题ID不一致")
                    return False

            return True

        except Exception as e:
            logger.error(f"共识结果验证失败: {e}")
            return False

    def get_consensus_statistics(self, results: List[ConsensusResult]) -> Dict[str, Any]:

        if not results:
            return {"error": "No consensus results"}

        confidences = [r.consensus_confidence for r in results]
        round_numbers = [r.round_number for r in results]
        convergence_rates = [r.convergence_achieved for r in results]

        return {
            "total_consensus": len(results),
            "average_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "average_rounds": np.mean(round_numbers),
            "convergence_rate": np.mean(convergence_rates),
            "method": self.consensus_method.value
        }

def create_consensus_engine(
    consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED,
    **kwargs
) -> StandardizedConsensusEngine:

    return StandardizedConsensusEngine(consensus_method, **kwargs)

if __name__ == "__main__":

    print("标准化共识引擎测试")

    test_question = QuestionData(
        question_id="test_001",
        question_text="What is 2+2?",
        correct_answer="4",
        question_type="math"
    )

    test_responses = [
        AgentResponse("agent_1", "test_001", "4", 0.9),
        AgentResponse("agent_2", "test_001", "4", 0.8),
        AgentResponse("agent_3", "test_001", "5", 0.6),        
    ]

    async def test():
        engine = create_consensus_engine(ConsensusMethod.CONFIDENCE_WEIGHTED)
        result = await engine.run_consensus(test_question, test_responses)

        print(f"共识答案: {result.consensus_answer}")
        print(f"共识置信度: {result.consensus_confidence:.2f}")
        print(f"收敛状态: {result.convergence_achieved}")

        is_valid = engine.validate_consensus(result)
        print(f"结果有效性: {is_valid}")

    asyncio.run(test())
    print("共识引擎测试完成")
