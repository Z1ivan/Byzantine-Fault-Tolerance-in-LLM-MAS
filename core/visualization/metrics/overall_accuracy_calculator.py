
import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseCalculator, TestData

class OverallAccuracyCalculator(BaseCalculator):

    def calculate(self, test_data: TestData) -> Dict[str, Any]:

        if not self.validate_data(test_data):
            return self._empty_result()

        total_initial_correct = 0
        total_final_correct = 0
        total_answers = 0

        for round_data in test_data.rounds:
            try:
                correct_answer = round_data['question']['answer']
                agent_answers = round_data['agent_answers']
                consensus_results = round_data.get('consensus_results', {})

                is_safe_dataset = (correct_answer in ['safe', 'unsafe'] or 
                                 '_dataset_type' in round_data['question'] and 
                                 round_data['question']['_dataset_type'] == 'safe')

                for agent_id, answer in agent_answers.items():

                    if is_safe_dataset:
                        mapped_answer = 'safe' if str(answer) == '1' else 'unsafe'
                        if mapped_answer == correct_answer:
                            total_initial_correct += 1
                    else:

                        if str(answer).strip() == str(correct_answer).strip():
                            total_initial_correct += 1
                    total_answers += 1

                if consensus_results:
                    for agent_id, final_answer in consensus_results.items():

                        if is_safe_dataset:
                            mapped_final_answer = 'safe' if str(final_answer) == '1' else 'unsafe'
                            if mapped_final_answer == correct_answer:
                                total_final_correct += 1
                        else:

                            if str(final_answer).strip() == str(correct_answer).strip():
                                total_final_correct += 1

            except (KeyError, TypeError):
                continue

        initial_accuracy = total_initial_correct / total_answers if total_answers > 0 else 0
        final_accuracy = total_final_correct / total_answers if total_answers > 0 else 0
        improvement = final_accuracy - initial_accuracy

        return {
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'total_initial_correct': total_initial_correct,
            'total_final_correct': total_final_correct,
            'total_answers': total_answers,
            'explanation': self.get_explanation()
        }

    def get_explanation(self) -> str:

        return '总体准确率：所有节点在所有问题上的平均准确率'

    def _empty_result(self) -> Dict[str, Any]:

        return {
            'initial_accuracy': 0,
            'final_accuracy': 0,
            'improvement': 0,
            'total_initial_correct': 0,
            'total_final_correct': 0,
            'total_answers': 0,
            'note': '无有效数据',
            'explanation': self.get_explanation()
        }
