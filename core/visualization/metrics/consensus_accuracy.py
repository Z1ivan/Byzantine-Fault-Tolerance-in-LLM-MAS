
import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseCalculator, TestData

class ConsensusAccuracyCalculator(BaseCalculator):

    def calculate(self, test_data: TestData) -> Dict[str, Any]:

        if not self.validate_data(test_data):
            return self._empty_result()

        correct_consensus = 0
        total_questions = 0
        question_details = []

        for round_data in test_data.rounds:
            try:
                correct_answer = round_data['question']['answer']
                consensus_results = round_data.get('consensus_results', {})
                question_id = round_data['question'].get('id', round_data['question'].get('question_id', 'unknown'))

                total_questions += 1             

                final_consensus = (
                    round_data.get('final_consensus')
                    or round_data.get('consensus_result')
                    or round_data.get('final_consensus_answer')
                    or round_data.get('consensus_answer')
                )

                is_safe_dataset = (correct_answer in ['safe', 'unsafe'] or
                                   ('_dataset_type' in round_data['question'] and round_data['question']['_dataset_type'] == 'safe'))

                def _norm_safe(val: str) -> str:
                    v = str(val).strip().lower()
                    if v in ('safe','unsafe'):
                        return v
                    return 'safe' if v == '1' else 'unsafe'

                if final_consensus is not None:
                    if is_safe_dataset:
                        is_correct = _norm_safe(final_consensus) == _norm_safe(correct_answer)
                    else:
                        is_correct = str(final_consensus).strip() == str(correct_answer).strip()
                    if is_correct:
                        correct_consensus += 1
                    question_details.append({
                        'question_id': question_id,
                        'correct_answer': correct_answer,
                        'consensus_answer': final_consensus,
                        'is_correct': is_correct,
                        'consensus_achieved': True
                    })
                    continue

                if consensus_results:

                    final_answers = list(consensus_results.values())
                    answer_counts = {}
                    for ans in final_answers:
                        ans_str = str(ans).strip()
                        answer_counts[ans_str] = answer_counts.get(ans_str, 0) + 1
                    if answer_counts:
                        max_count = max(answer_counts.values())
                        total_nodes = len(final_answers)
                        if max_count > total_nodes / 2:
                            consensus_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x])
                            if is_safe_dataset:
                                is_correct = _norm_safe(consensus_answer) == _norm_safe(correct_answer)
                            else:
                                is_correct = consensus_answer == str(correct_answer).strip()
                            if is_correct:
                                correct_consensus += 1
                            question_details.append({
                                'question_id': question_id,
                                'correct_answer': correct_answer,
                                'consensus_answer': consensus_answer,
                                'is_correct': is_correct,
                                'consensus_count': answer_counts[consensus_answer],
                                'total_nodes': len(final_answers),
                                'consensus_achieved': True
                            })
                        else:
                            question_details.append({
                                'question_id': question_id,
                                'correct_answer': correct_answer,
                                'consensus_answer': 'No Consensus',
                                'is_correct': False,
                                'consensus_count': max_count,
                                'total_nodes': len(final_answers),
                                'consensus_achieved': False,
                                'answer_distribution': dict(answer_counts)
                            })

            except (KeyError, TypeError):
                continue

        accuracy = correct_consensus / total_questions if total_questions > 0 else 0

        return {
            'consensus_accuracy': accuracy,
            'correct_consensus': correct_consensus,
            'total_consensus': total_questions,             
            'question_details': question_details,
            'explanation': self.get_explanation()
        }

    def get_explanation(self) -> str:

        return '共识准确率：系统达成正确多数共识的问题比例（需要>50%节点同意）'

    def _empty_result(self) -> Dict[str, Any]:

        return {
            'consensus_accuracy': 0,
            'correct_consensus': 0,
            'total_consensus': 0,
            'question_details': [],
            'note': '无有效共识数据',
            'explanation': self.get_explanation()
        } 