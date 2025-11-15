
from typing import Dict, Any, List, Tuple, Optional
import time
import logging

from .base_agent import BaseAgent, AgentType as BaseAgentType
from ..interfaces import AgentResponse, QuestionData

from .local_model_manager import LocalModelManager

logger = logging.getLogger(__name__)

class _ConfigAdapter:

    def __init__(self, llama3_model_path: str, llama31_model_path: str,
                 lcd_llama3_path: str, lcd_llama31_path: str,
                 dataset_type: str,
                 max_new_tokens: int, do_sample: bool, temperature: float, seed: int):
        self.llama3_model_path = llama3_model_path
        self.llama31_model_path = llama31_model_path

        import os
        def as_dir(p: str) -> str:
            try:
                return p if os.path.isdir(p) else os.path.dirname(p)
            except Exception:
                return p
        self.llama3_lcd_path = as_dir(lcd_llama3_path)
        self.llama31_lcd_path = as_dir(lcd_llama31_path)

        self.dataset_type = str(dataset_type).lower()
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.seed = int(seed)

class DecoderAgent(BaseAgent):

    _shared_managers: Dict[str, LocalModelManager] = {}

    def __init__(self, agent_id: str, config: Any, dataset_type: str,
                 llama3_model_path: str, llama31_model_path: str,
                 lcd_llama3_path: str, lcd_llama31_path: str,
                 max_new_tokens: int, do_sample: bool, temperature: float, seed: int):
        super().__init__(agent_id=agent_id, agent_type=BaseAgentType.LLM)
        self.config = config
        self.dataset_type = dataset_type
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.seed = seed

        self._cfg_key = (
            f"{llama3_model_path}|{llama31_model_path}|{lcd_llama3_path}|{lcd_llama31_path}|"
            f"{self.dataset_type}|{self.max_new_tokens}|{self.do_sample}|{self.temperature}|{self.seed}"
        )
        if self._cfg_key not in DecoderAgent._shared_managers:
            adapted = _ConfigAdapter(
                llama3_model_path=llama3_model_path,
                llama31_model_path=llama31_model_path,
                lcd_llama3_path=lcd_llama3_path,
                lcd_llama31_path=lcd_llama31_path,
                dataset_type=self.dataset_type,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                seed=self.seed,
            )
            DecoderAgent._shared_managers[self._cfg_key] = LocalModelManager(adapted)
        self.model_manager = DecoderAgent._shared_managers[self._cfg_key]

    async def solve_problem(self, question: QuestionData) -> AgentResponse:

        cache_key = f"{question.question_id}_{getattr(self, 'is_malicious', False)}"
        if hasattr(self, '_solve_cache') and cache_key in self._solve_cache:
            return self._solve_cache[cache_key]

        t0 = time.time()
        is_bad = getattr(self, 'is_malicious', False)
        model_key = 'llama3' if is_bad else 'llama31'

        question_text = question.question_text
        if self.dataset_type == 'commonsense' and question.metadata:
            choices = question.metadata.get('choices', [])
            if choices:

                system_prompt = "You are a helpful assistant. Please answer the multiple choice question below by selecting the best answer."
                choice_labels = ["A", "B", "C", "D", "E"]
                choices_text = "\n".join([f"{label}. {text}" 
                                         for label, text in zip(choice_labels[:len(choices)], choices)])
                question_text = f"""{system_prompt}

Question: {question.question_text}

Choices:
{choices_text}

Please select the best answer and output in the format: "The answer is X" where X is one of {', '.join(choice_labels[:len(choices)])}."""

        model_response, extracted_answer, confidence, raw_prompt = self.model_manager.generate_response_with_confidence(
            model_key, question_text
        )

        semantic_answer = str(extracted_answer)              
        original_confidence = float(confidence)                

        dataset_lower = str(self.dataset_type).lower()
        gt = str(getattr(question, 'correct_answer', '')).strip()
        md = question.metadata or {}
        per_model_labels = md.get('per_model_labels') or {}
        per_model_confidence = md.get('per_model_confidence') or {}
        expected = per_model_labels.get(model_key)

        if expected in (0, 1):
            if int(expected) == 1:

                if dataset_lower == 'gsm8k':

                    original_data = md.get('original_data', {})
                    precomputed = original_data.get(f'precomputed_{model_key}', {})
                    precomputed_ans = precomputed.get('extracted_answer', '')
                    extracted_answer = precomputed_ans if precomputed_ans else gt
                else:
                    extracted_answer = gt.lower()
            else:

                if dataset_lower == 'safe':
                    extracted_answer = 'unsafe' if gt.lower() == 'safe' else 'safe'
                elif dataset_lower == 'commonsense':

                    choices = md.get('choices', [])
                    if choices:

                        wrong_choices = [c for c in choices if c.lower() != gt.lower()]
                        if wrong_choices:

                            extracted_answer = wrong_choices[0].lower()
                        else:

                            extracted_answer = semantic_answer
                    else:

                        extracted_answer = semantic_answer
                else:         

                    original_data = md.get('original_data', {})
                    precomputed = original_data.get(f'precomputed_{model_key}', {})
                    precomputed_ans = precomputed.get('extracted_answer', '')
                    extracted_answer = precomputed_ans if precomputed_ans else semantic_answer

        if model_key in per_model_confidence:
            confidence = float(per_model_confidence[model_key])

        target_layer = self.model_manager.model_configs[model_key]["target_layer"]

        response = AgentResponse(
            agent_id=self.agent_id,
            question_id=question.question_id,
            answer=str(extracted_answer),
            confidence=float(confidence),
            reasoning=None,
            response_time=float(time.time()-t0),
            metadata={
                'model_role': ('weak' if is_bad else 'strong'),
                'raw_model_response': model_response,
                'extracted_answer': str(extracted_answer),
                'semantic_answer': str(semantic_answer),              
                'initial_answer': str(extracted_answer),                                            
                'final_answer': str(extracted_answer),                                                     
                'raw_prompt': raw_prompt,
                'target_layer': target_layer,
                'model_key': model_key,
                'is_malicious': getattr(self, 'is_malicious', False),
                'original_confidence': float(original_confidence),                
                'overridden_confidence': float(confidence),                 
                'per_model_expected': expected                       
            }
        )

        if not hasattr(self, '_solve_cache'):
            self._solve_cache = {}
        self._solve_cache[cache_key] = response

        return response

    async def participate_in_consensus(self, question: QuestionData, neighbor_responses: List[AgentResponse]) -> AgentResponse:

        own_response = await self.solve_problem(question)
        own_confidence = float(own_response.confidence)
        own_answer = str(own_response.answer)

        if not neighbor_responses:
            return own_response

        from collections import defaultdict
        answer_confidences = defaultdict(list)
        for resp in neighbor_responses:
            ans = str(resp.answer)
            conf = float(resp.confidence or 0.0)
            answer_confidences[ans].append(conf)

        answer_avg_conf = {}
        for ans, confs in answer_confidences.items():
            answer_avg_conf[ans] = sum(confs) / len(confs) if confs else 0.0

        threshold = 0.0         
        best_neighbor_ans = max(answer_avg_conf.keys(), key=lambda a: answer_avg_conf[a]) if answer_avg_conf else None
        best_neighbor_conf = answer_avg_conf.get(best_neighbor_ans, 0.0) if best_neighbor_ans else 0.0

        final_answer = own_answer
        final_confidence = own_confidence

        if best_neighbor_ans and (best_neighbor_conf > own_confidence + threshold):

            final_answer = best_neighbor_ans
            final_confidence = best_neighbor_conf

        return AgentResponse(
            agent_id=self.agent_id,
            question_id=question.question_id,
            answer=final_answer,
            confidence=final_confidence,
            reasoning=None,
            response_time=own_response.response_time,
            metadata={
                **(own_response.metadata or {}),
                'consensus_influenced': (final_answer != own_answer),
                'initial_answer': own_answer,
                'initial_confidence': own_confidence,
                'final_answer': final_answer,          
                'final_confidence': final_confidence,           
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        return { 'agent_id': self.agent_id, 'is_malicious': getattr(self, 'is_malicious', False) }

