
import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseCalculator, TestData

class ConsensusErrorCalculator(BaseCalculator):

    def calculate(self, test_data: TestData) -> Dict[str, Any]:

        if not self.validate_data(test_data):
            return self._empty_result()

        consensus_errors = []
        question_details = []

        for round_data in test_data.rounds:
            try:
                correct_answer = round_data['question']['answer']
                consensus_results = round_data.get('consensus_results', {})
                question_id = round_data['question'].get('id', 'unknown')

                if consensus_results:

                    final_answers = list(consensus_results.values())

                    answer_counts = {}
                    for ans in final_answers:
                        ans_str = str(ans).strip()
                        answer_counts[ans_str] = answer_counts.get(ans_str, 0) + 1

                    if answer_counts:
                        consensus_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x])

                        try:
                            correct_val = float(correct_answer)
                            consensus_val = float(consensus_answer)
                            error = (consensus_val - correct_val) ** 2
                            calculation_type = 'numerical'
                        except (ValueError, TypeError):

                            error = 0 if consensus_answer == str(correct_answer).strip() else 1
                            calculation_type = 'categorical'

                        consensus_errors.append(error)
                        question_details.append({
                            'question_id': question_id,
                            'correct_answer': correct_answer,
                            'consensus_answer': consensus_answer,
                            'ce': error,
                            'calculation_type': calculation_type
                        })

            except (KeyError, TypeError):
                continue

        if consensus_errors:
            return {
                'mean_ce': np.mean(consensus_errors),
                'std_ce': np.std(consensus_errors),
                'min_ce': min(consensus_errors),
                'max_ce': max(consensus_errors),
                'valid_questions': len(consensus_errors),
                'question_details': question_details,
                'explanation': self.get_explanation()
            }
        else:
            return self._empty_result()

    def get_explanation(self) -> str:

        return 'CE计算：最终共识答案与正确答案的误差'

    def _empty_result(self) -> Dict[str, Any]:

        return {
            'mean_ce': 0,
            'std_ce': 0,
            'min_ce': 0,
            'max_ce': 0,
            'valid_questions': 0,
            'note': '无有效共识数据',
            'explanation': self.get_explanation()
        } 