#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UnifiedByzantineMetrics:

    def __init__(self):
        self.name = "UnifiedByzantineMetrics"
        self.version = "2.0"

    def calculate_node_accuracy(self, agent_responses: Dict[str, List[str]], 
                               correct_answers: List[str]) -> Dict[str, Any]:

        if not agent_responses or not correct_answers:
            return {"error": "Empty input data"}

        node_accuracies = {}

        for agent_id, responses in agent_responses.items():
            if len(responses) != len(correct_answers):
                logger.warning(f"Agent {agent_id} response count mismatch: {len(responses)} vs {len(correct_answers)}")

                min_length = min(len(responses), len(correct_answers))
                responses = responses[:min_length]
                used_correct_answers = correct_answers[:min_length]
            else:
                used_correct_answers = correct_answers

            correct_count = sum(1 for resp, correct in zip(responses, used_correct_answers) 
                              if self._is_answer_correct(resp, correct))

            accuracy = correct_count / len(responses) if responses else 0.0
            node_accuracies[agent_id] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(responses),
                "error_rate": 1.0 - accuracy,
                "responses": responses
            }

        all_accuracies = [data["accuracy"] for data in node_accuracies.values()]

        return {
            "node_accuracies": node_accuracies,
            "average_node_accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "std_node_accuracy": np.std(all_accuracies) if all_accuracies else 0.0,
            "min_node_accuracy": np.min(all_accuracies) if all_accuracies else 0.0,
            "max_node_accuracy": np.max(all_accuracies) if all_accuracies else 0.0,
            "total_agents": len(node_accuracies),
            "accuracy_variance": np.var(all_accuracies) if all_accuracies else 0.0
        }

    def calculate_consensus_accuracy_basic(self, consensus_results: List[str], 
                                         correct_answers: List[str]) -> Dict[str, Any]:

        if len(consensus_results) != len(correct_answers):
            raise ValueError("Consensus results and correct answers length mismatch")

        correct_consensus = sum(1 for consensus, correct in zip(consensus_results, correct_answers)
                              if self._is_answer_correct(consensus, correct))

        total_questions = len(correct_answers)
        consensus_accuracy = correct_consensus / total_questions if total_questions > 0 else 0.0

        return {
            "consensus_accuracy": consensus_accuracy,
            "correct_consensus_count": correct_consensus,
            "total_questions": total_questions,
            "failed_questions": total_questions - correct_consensus,
            "error_rate": 1.0 - consensus_accuracy,
            "method": "basic_comparison"
        }

    def calculate_consensus_accuracy_academic(self, agent_responses: Dict[str, List[str]], 
                                            correct_answers: List[str]) -> Dict[str, Any]:

        if not agent_responses or not correct_answers:
            return {"error": "Empty input data"}

        num_questions = len(correct_answers)
        consensus_results = []
        detailed_results = []

        for q_idx in range(num_questions):

            votes = []
            vote_details = {}

            for agent_id, answers in agent_responses.items():
                if q_idx < len(answers):
                    answer = answers[q_idx]
                    votes.append(answer)
                    vote_details[agent_id] = answer

            if not votes:
                consensus_results.append(False)
                detailed_results.append({
                    "question_idx": q_idx,
                    "votes": {},
                    "majority_answer": None,
                    "is_consensus_correct": False,
                    "correct_answer": correct_answers[q_idx]
                })
                continue

            vote_counts = Counter(votes)
            majority_answer, majority_count = vote_counts.most_common(1)[0]

            is_consensus_correct = self._is_answer_correct(majority_answer, correct_answers[q_idx])
            consensus_results.append(is_consensus_correct)

            detailed_results.append({
                "question_idx": q_idx,
                "votes": vote_details,
                "vote_distribution": dict(vote_counts),
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "total_votes": len(votes),
                "majority_ratio": majority_count / len(votes),
                "is_consensus_correct": is_consensus_correct,
                "correct_answer": correct_answers[q_idx]
            })

        consensus_accuracy = sum(consensus_results) / len(consensus_results) if consensus_results else 0.0

        return {
            "consensus_accuracy": consensus_accuracy,
            "correct_consensus_count": sum(consensus_results),
            "total_questions": num_questions,
            "failed_questions": num_questions - sum(consensus_results),
            "error_rate": 1.0 - consensus_accuracy,
            "detailed_results": detailed_results,
            "method": "majority_voting"
        }

    def calculate_consensus_error_academic(self, agent_parameters: Dict[str, List[np.ndarray]], 
                                         malicious_agents: List[str] = None) -> Dict[str, Any]:

        if not agent_parameters:
            return {"error": "No agent parameters provided"}

        normal_agents = []
        for agent_id in agent_parameters.keys():
            if not malicious_agents or agent_id not in malicious_agents:
                normal_agents.append(agent_id)

        if not normal_agents:
            return {"error": "No normal agents found"}

        max_iterations = max(len(params) for params in agent_parameters.values())
        ce_per_iteration = []

        for k in range(max_iterations):

            normal_params_k = []
            for agent_id in normal_agents:
                if k < len(agent_parameters[agent_id]):
                    normal_params_k.append(agent_parameters[agent_id][k])

            if len(normal_params_k) < 2:              
                ce_per_iteration.append(0.0)
                continue

            w_bar_k = np.mean(normal_params_k, axis=0)

            ce_k = 0.0
            for w_k_i in normal_params_k:
                ce_k += np.linalg.norm(w_k_i - w_bar_k) ** 2

            ce_k /= len(normal_params_k)               
            ce_per_iteration.append(ce_k)

        return {
            "consensus_error_per_iteration": ce_per_iteration,
            "mean_consensus_error": np.mean(ce_per_iteration),
            "std_consensus_error": np.std(ce_per_iteration),
            "final_consensus_error": ce_per_iteration[-1] if ce_per_iteration else 0.0,
            "max_consensus_error": np.max(ce_per_iteration) if ce_per_iteration else 0.0,
            "min_consensus_error": np.min(ce_per_iteration) if ce_per_iteration else 0.0,
            "consensus_error_trend": "decreasing" if len(ce_per_iteration) > 1 and ce_per_iteration[-1] < ce_per_iteration[0] else "stable",
            "normal_agents_count": len(normal_agents),
            "total_iterations": max_iterations
        }

    def calculate_consensus_error_simplified(self, agent_responses: Dict[str, List[str]], 
                                           malicious_agents: List[str] = None) -> Dict[str, Any]:

        if not agent_responses:
            return {"error": "No agent responses provided"}

        normal_agents = []
        for agent_id in agent_responses.keys():
            if not malicious_agents or agent_id not in malicious_agents:
                normal_agents.append(agent_id)

        if len(normal_agents) < 2:
            return {"error": "Need at least 2 normal agents for CE calculation"}

        num_questions = min(len(responses) for responses in agent_responses.values())
        ce_per_question = []

        for q_idx in range(num_questions):

            normal_params = []
            for agent_id in normal_agents:
                answer = agent_responses[agent_id][q_idx]

                param_vector = np.array([hash(answer) % 1000000] * 10)              
                normal_params.append(param_vector)

            w_bar = np.mean(normal_params, axis=0)

            ce_q = 0.0
            for w_i in normal_params:
                ce_q += np.linalg.norm(w_i - w_bar) ** 2
            ce_q /= len(normal_params)

            ce_per_question.append(ce_q)

        return {
            "consensus_error_per_question": ce_per_question,
            "mean_consensus_error": np.mean(ce_per_question),
            "std_consensus_error": np.std(ce_per_question),
            "max_consensus_error": np.max(ce_per_question) if ce_per_question else 0.0,
            "min_consensus_error": np.min(ce_per_question) if ce_per_question else 0.0,
            "normal_agents_count": len(normal_agents),
            "total_questions": num_questions,
            "method": "simplified_qa_hashing"
        }

    def calculate_byzantine_fault_tolerance(self, agent_responses: Dict[str, List[str]], 
                                          malicious_agents: List[str],
                                          correct_answers: List[str]) -> Dict[str, Any]:

        total_agents = len(agent_responses)
        malicious_count = len(malicious_agents)
        malicious_ratio = malicious_count / total_agents if total_agents > 0 else 0.0

        theoretical_limit = (total_agents - 1) / 3
        theoretical_limit_ratio = theoretical_limit / total_agents if total_agents > 0 else 0.0
        byzantine_safe = malicious_count <= theoretical_limit

        malicious_performance = self._analyze_malicious_impact(
            agent_responses, malicious_agents, correct_answers
        )

        system_performance = self._calculate_system_resilience(
            agent_responses, malicious_agents, correct_answers
        )

        consensus_maintained = system_performance > 0.5                 

        return {
            "total_agents": total_agents,
            "malicious_count": malicious_count,
            "malicious_ratio": malicious_ratio,
            "theoretical_limit": theoretical_limit,
            "theoretical_limit_ratio": theoretical_limit_ratio,
            "within_byzantine_limit": byzantine_safe,
            "safety_margin": theoretical_limit - malicious_count,
            "consensus_maintained": consensus_maintained,
            "malicious_impact": malicious_performance,
            "system_resilience": system_performance,
            "fault_tolerance_rating": self._rate_fault_tolerance(malicious_ratio, system_performance)
        }

    def calculate_efficiency_metrics(self, response_times: Dict[str, List[float]],
                                   consensus_times: List[float],
                                   resource_usage: Dict[str, Any] = None) -> Dict[str, Any]:

        all_response_times = []
        for times in response_times.values():
            all_response_times.extend(times)

        response_stats = {
            "mean_response_time": np.mean(all_response_times) if all_response_times else 0.0,
            "std_response_time": np.std(all_response_times) if all_response_times else 0.0,
            "min_response_time": np.min(all_response_times) if all_response_times else 0.0,
            "max_response_time": np.max(all_response_times) if all_response_times else 0.0,
            "total_responses": len(all_response_times)
        }

        consensus_stats = {
            "mean_consensus_time": np.mean(consensus_times) if consensus_times else 0.0,
            "std_consensus_time": np.std(consensus_times) if consensus_times else 0.0,
            "min_consensus_time": np.min(consensus_times) if consensus_times else 0.0,
            "max_consensus_time": np.max(consensus_times) if consensus_times else 0.0,
            "total_consensus_rounds": len(consensus_times)
        }

        overall_efficiency = self._calculate_overall_efficiency(response_stats, consensus_stats)

        result = {
            "response_time_stats": response_stats,
            "consensus_time_stats": consensus_stats,
            "overall_efficiency_score": overall_efficiency,
            "efficiency_rating": self._rate_efficiency(overall_efficiency)
        }

        if resource_usage:
            result["resource_efficiency"] = self._analyze_resource_efficiency(resource_usage)

        return result

    def generate_comprehensive_evaluation(self, 
                                        agent_responses: Dict[str, List[str]], 
                                        correct_answers: List[str],
                                        malicious_agents: List[str] = None,
                                        response_times: Dict[str, List[float]] = None,
                                        consensus_times: List[float] = None,
                                        use_academic_standards: bool = True,
                                        **kwargs) -> Dict[str, Any]:

        evaluation_report = {
            "evaluation_metadata": {
                "evaluator": self.name,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "academic_standards": use_academic_standards,
                "total_agents": len(agent_responses),
                "total_questions": len(correct_answers),
                "malicious_agents_count": len(malicious_agents) if malicious_agents else 0
            }
        }

        evaluation_report["node_accuracy"] = self.calculate_node_accuracy(
            agent_responses, correct_answers
        )

        if use_academic_standards:
            evaluation_report["consensus_accuracy"] = self.calculate_consensus_accuracy_academic(
                agent_responses, correct_answers
            )
        else:

            consensus_results = kwargs.get('consensus_results', [])
            if consensus_results:
                evaluation_report["consensus_accuracy"] = self.calculate_consensus_accuracy_basic(
                    consensus_results, correct_answers
                )

        if use_academic_standards:
            evaluation_report["consensus_error"] = self.calculate_consensus_error_simplified(
                agent_responses, malicious_agents
            )

        if malicious_agents:
            evaluation_report["byzantine_fault_tolerance"] = self.calculate_byzantine_fault_tolerance(
                agent_responses, malicious_agents, correct_answers
            )

        if response_times and consensus_times:
            evaluation_report["efficiency_metrics"] = self.calculate_efficiency_metrics(
                response_times, consensus_times, kwargs.get('resource_usage')
            )

        if use_academic_standards and kwargs.get('consensus_results'):
            basic_consensus = self.calculate_consensus_accuracy_basic(
                kwargs['consensus_results'], correct_answers
            )
            academic_consensus = evaluation_report.get("consensus_accuracy", {})

            evaluation_report["comparative_analysis"] = {
                "accuracy_comparison": {
                    "basic_test_accuracy": basic_consensus.get("consensus_accuracy", 0),
                    "academic_consensus_accuracy": academic_consensus.get("consensus_accuracy", 0),
                    "difference": academic_consensus.get("consensus_accuracy", 0) - basic_consensus.get("consensus_accuracy", 0)
                }
            }

            if malicious_agents:
                bft_analysis = evaluation_report.get("byzantine_fault_tolerance", {})
                evaluation_report["comparative_analysis"]["malicious_impact"] = {
                    "malicious_ratio": bft_analysis.get("malicious_ratio", 0),
                    "theoretical_limit": bft_analysis.get("theoretical_limit_ratio", 0),
                    "within_byzantine_limit": bft_analysis.get("within_byzantine_limit", False),
                    "consensus_maintained": bft_analysis.get("consensus_maintained", False)
                }

        evaluation_report["overall_assessment"] = self._generate_overall_assessment(evaluation_report)

        return evaluation_report

    def _is_answer_correct(self, response: str, correct_answer: str) -> bool:

        if not response or not correct_answer:
            return False

        response_clean = str(response).strip().lower()
        correct_clean = str(correct_answer).strip().lower()

        def _normalize_safe_label(s: str) -> str:
            if s in ("1", "safe"):
                return "safe"
            if s in ("0", "unsafe"):
                return "unsafe"
            return s
        rc_norm = _normalize_safe_label(response_clean)
        cc_norm = _normalize_safe_label(correct_clean)
        if rc_norm == cc_norm:
            return True

        if response_clean == correct_clean:
            return True

        try:
            resp_num = float(response_clean)
            correct_num = float(correct_clean)
            return abs(resp_num - correct_num) < 1e-6
        except (ValueError, TypeError):
            pass

        safe_labels = {"safe", "unsafe", "1", "0"}
        if response_clean in safe_labels or correct_clean in safe_labels:
            return False

        if len(response_clean) >= 8 or len(correct_clean) >= 8:
            return (correct_clean in response_clean) or (response_clean in correct_clean)

        return False

    def _analyze_malicious_impact(self, agent_responses: Dict[str, List[str]], 
                                malicious_agents: List[str], correct_answers: List[str]) -> Dict[str, Any]:

        malicious_accuracies = []
        normal_accuracies = []

        for agent_id, responses in agent_responses.items():
            correct_count = sum(1 for resp, correct in zip(responses, correct_answers)
                              if self._is_answer_correct(resp, correct))
            accuracy = correct_count / len(responses) if responses else 0.0

            if agent_id in malicious_agents:
                malicious_accuracies.append(accuracy)
            else:
                normal_accuracies.append(accuracy)

        return {
            "malicious_avg_accuracy": np.mean(malicious_accuracies) if malicious_accuracies else 0.0,
            "normal_avg_accuracy": np.mean(normal_accuracies) if normal_accuracies else 0.0,
            "accuracy_gap": np.mean(normal_accuracies) - np.mean(malicious_accuracies) if normal_accuracies and malicious_accuracies else 0.0,
            "malicious_deviation": np.std(malicious_accuracies) if malicious_accuracies else 0.0
        }

    def _calculate_system_resilience(self, agent_responses: Dict[str, List[str]], 
                                   malicious_agents: List[str], correct_answers: List[str]) -> float:

        normal_agents = [agent_id for agent_id in agent_responses.keys() 
                        if agent_id not in malicious_agents]

        if not normal_agents:
            return 0.0

        normal_accuracies = []
        for agent_id in normal_agents:
            responses = agent_responses[agent_id]
            correct_count = sum(1 for resp, correct in zip(responses, correct_answers)
                              if self._is_answer_correct(resp, correct))
            accuracy = correct_count / len(responses) if responses else 0.0
            normal_accuracies.append(accuracy)

        return np.mean(normal_accuracies) if normal_accuracies else 0.0

    def _rate_fault_tolerance(self, malicious_ratio: float, system_performance: float) -> str:

        if malicious_ratio <= 0.2 and system_performance > 0.8:
            return "优秀"
        elif malicious_ratio <= 0.3 and system_performance > 0.6:
            return "良好"
        elif malicious_ratio <= 0.4 and system_performance > 0.4:
            return "一般"
        else:
            return "需改进"

    def _calculate_overall_efficiency(self, response_stats: Dict, consensus_stats: Dict) -> float:

        mean_response = response_stats.get("mean_response_time", 10.0)
        mean_consensus = consensus_stats.get("mean_consensus_time", 5.0)

        response_efficiency = max(0, 1 - mean_response / 30.0)            
        consensus_efficiency = max(0, 1 - mean_consensus / 10.0)            

        return (response_efficiency + consensus_efficiency) / 2

    def _rate_efficiency(self, efficiency_score: float) -> str:

        if efficiency_score > 0.8:
            return "高效"
        elif efficiency_score > 0.6:
            return "良好"
        elif efficiency_score > 0.4:
            return "一般"
        else:
            return "低效"

    def _analyze_resource_efficiency(self, resource_usage: Dict) -> Dict[str, Any]:

        return {
            "cpu_efficiency": resource_usage.get("cpu_usage", 0),
            "memory_efficiency": resource_usage.get("memory_usage", 0),
            "network_efficiency": resource_usage.get("network_usage", 0)
        }

    def _generate_overall_assessment(self, evaluation_report: Dict) -> Dict[str, Any]:

        node_acc = evaluation_report.get("node_accuracy", {}).get("average_node_accuracy", 0)
        consensus_acc = evaluation_report.get("consensus_accuracy", {}).get("consensus_accuracy", 0)
        bft_rating = evaluation_report.get("byzantine_fault_tolerance", {}).get("fault_tolerance_rating", "未知")

        overall_score = (node_acc + consensus_acc) / 2

        return {
            "overall_score": overall_score,
            "performance_rating": self._rate_overall_performance(overall_score),
            "fault_tolerance_rating": bft_rating,
            "key_strengths": self._identify_strengths(evaluation_report),
            "improvement_areas": self._identify_improvements(evaluation_report)
        }

    def _rate_overall_performance(self, score: float) -> str:

        if score > 0.9:
            return "卓越"
        elif score > 0.8:
            return "优秀"
        elif score > 0.7:
            return "良好"
        elif score > 0.6:
            return "一般"
        else:
            return "需改进"

    def _identify_strengths(self, evaluation_report: Dict) -> List[str]:

        strengths = []

        node_acc = evaluation_report.get("node_accuracy", {}).get("average_node_accuracy", 0)
        if node_acc > 0.8:
            strengths.append("节点准确率高")

        consensus_acc = evaluation_report.get("consensus_accuracy", {}).get("consensus_accuracy", 0)
        if consensus_acc > 0.8:
            strengths.append("共识质量优秀")

        bft = evaluation_report.get("byzantine_fault_tolerance", {})
        if bft.get("within_byzantine_limit", False):
            strengths.append("拜占庭容错安全")

        return strengths

    def _identify_improvements(self, evaluation_report: Dict) -> List[str]:

        improvements = []

        node_acc = evaluation_report.get("node_accuracy", {}).get("average_node_accuracy", 0)
        if node_acc < 0.6:
            improvements.append("提高节点准确率")

        consensus_acc = evaluation_report.get("consensus_accuracy", {}).get("consensus_accuracy", 0)
        if consensus_acc < 0.6:
            improvements.append("改善共识机制")

        bft = evaluation_report.get("byzantine_fault_tolerance", {})
        if not bft.get("within_byzantine_limit", True):
            improvements.append("增强拜占庭容错能力")

        return improvements
