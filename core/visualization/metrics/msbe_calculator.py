
import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseCalculator, TestData

class MSBECalculator(BaseCalculator):

    def calculate(self, test_data: TestData) -> Dict[str, Any]:

        if not self.validate_data(test_data):
            return self._empty_result()

        bellman_errors = []
        question_details = []

        for round_data in test_data.rounds:
            try:
                correct_answer = round_data['question']['answer']
                agent_answers = round_data['agent_answers']
                question_id = round_data['question'].get('id', 'unknown')

                try:
                    correct_val = float(correct_answer)
                    errors = []
                    for agent, answer in agent_answers.items():
                        try:
                            answer_val = float(answer)
                            error = (answer_val - correct_val) ** 2
                            errors.append(error)
                        except (ValueError, TypeError):

                            error = 0 if str(answer).strip() == str(correct_answer).strip() else 1
                            errors.append(error)

                    mean_error = np.mean(errors) if errors else 0
                    bellman_errors.append(mean_error)
                    question_details.append({
                        'question_id': question_id,
                        'correct_answer': correct_answer,
                        'msbe': mean_error,
                        'calculation_type': 'numerical'
                    })

                except (ValueError, TypeError):

                    errors = []
                    for agent, answer in agent_answers.items():
                        error = 0 if str(answer).strip() == str(correct_answer).strip() else 1
                        errors.append(error)

                    mean_error = np.mean(errors) if errors else 0
                    bellman_errors.append(mean_error)
                    question_details.append({
                        'question_id': question_id,
                        'correct_answer': correct_answer,
                        'msbe': mean_error,
                        'calculation_type': 'categorical'
                    })

            except (KeyError, TypeError):
                continue

        if bellman_errors:
            return {
                'mean_msbe': np.mean(bellman_errors),
                'std_msbe': np.std(bellman_errors),
                'min_msbe': min(bellman_errors),
                'max_msbe': max(bellman_errors),
                'valid_questions': len(bellman_errors),
                'question_details': question_details,
                'explanation': self.get_explanation()
            }
        else:
            return self._empty_result()

    def get_explanation(self) -> str:

        return 'MSBE计算：数值问题使用平方误差，分类问题使用0/1误差'

    def _empty_result(self) -> Dict[str, Any]:

        return {
            'mean_msbe': 0,
            'std_msbe': 0,
            'min_msbe': 0,
            'max_msbe': 0,
            'valid_questions': 0,
            'note': '无有效数据',
            'explanation': self.get_explanation()
        } 