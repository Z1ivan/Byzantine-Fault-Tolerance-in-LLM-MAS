
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from ..interfaces import IDataLoader, MethodType, QuestionData

logger = logging.getLogger(__name__)

class BaseDataLoader(IDataLoader):

    def __init__(self, method_type: MethodType):
        self._method_type = method_type
        logger.info(f"初始化数据加载器: {method_type.value}")

    @property
    def method_type(self) -> MethodType:

        return self._method_type

    def validate_data(self, questions: List[QuestionData]) -> bool:

        if not questions:
            logger.error("问题列表为空")
            return False

        for i, question in enumerate(questions):
            if not question.question_id:
                logger.error(f"问题 {i} 缺少ID")
                return False

            if not question.question_text:
                logger.error(f"问题 {question.question_id} 缺少文本")
                return False

            if not question.correct_answer:
                logger.warning(f"问题 {question.question_id} 缺少正确答案")

        logger.info(f"数据验证通过: {len(questions)} 个问题")
        return True

    def get_data_statistics(self, questions: List[QuestionData]) -> Dict[str, Any]:

        if not questions:
            return {"error": "No questions provided"}

        total_questions = len(questions)
        question_types = {}
        has_correct_answers = 0

        for question in questions:

            q_type = question.question_type or "unknown"
            question_types[q_type] = question_types.get(q_type, 0) + 1

            if question.correct_answer:
                has_correct_answers += 1

        return {
            "total_questions": total_questions,
            "question_types": question_types,
            "has_correct_answers": has_correct_answers,
            "correct_answer_ratio": has_correct_answers / total_questions,
            "method_type": self.method_type.value
        }

class GSM8KDataLoader(BaseDataLoader):

    def __init__(self):

        super().__init__(MethodType.PILOT)

    def load_questions(self, data_path: str, **kwargs) -> List[QuestionData]:

        logger.info(f"加载数据: {data_path}")
        dataset_type = kwargs.get('dataset_type', 'gsm8k')
        if dataset_type == 'safe':

            logger.info("dataset_type=safe: 复用 SafeDataLoader 解析先导SAFE数据")
            safe_loader = SafeDataLoader()
            return safe_loader.load_questions(data_path, **kwargs)

        if dataset_type == 'commonsense':

            logger.info("dataset_type=commonsense: 加载CommonsenseQA数据")
            return self._load_commonsense_questions(data_path, **kwargs)

        data_file = Path(data_path)
        if not data_file.exists():
            logger.error(f"数据文件不存在: {data_path}")
            return []

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            questions = []

            if isinstance(raw_data, list):

                for i, item in enumerate(raw_data):

                    item_id = item.get('id') or f"gsm8k_{i+1:03d}"
                    question = self._parse_gsm8k_item(item, item_id, dataset_type)
                    if question:
                        questions.append(question)
            elif isinstance(raw_data, dict):

                if 'results' in raw_data and isinstance(raw_data['results'], list):
                    for i, entry in enumerate(raw_data['results']):
                        try:
                            q_text = entry.get('question', '')

                            ans = entry.get('correct_answer') or entry.get('ground_truth') or ''

                            per_model_labels: Dict[str, int] = {}
                            for mk in ('llama3', 'llama31'):
                                precomputed_key = f'precomputed_{mk}'
                                if precomputed_key in entry and isinstance(entry[precomputed_key], dict):
                                    precomputed = entry[precomputed_key]

                                    if 'is_correct' in precomputed:
                                        try:
                                            per_model_labels[mk] = int(precomputed['is_correct'])
                                        except Exception:
                                            pass

                            per_model_confidence: Dict[str, float] = {}
                            for mk in ('llama3', 'llama31'):
                                conf_key = f'{mk}_confidence'
                                if conf_key in entry:
                                    try:
                                        per_model_confidence[mk] = float(entry[conf_key])
                                    except Exception:
                                        pass

                            metadata = {"source": dataset_type, "original_data": entry}
                            if per_model_labels:
                                metadata["per_model_labels"] = per_model_labels
                            if per_model_confidence:
                                metadata["per_model_confidence"] = per_model_confidence

                            item_id = entry.get('id') or f"gsm8k_{i+1:03d}"

                            q = QuestionData(
                                question_id=str(item_id),
                                question_text=q_text,
                                correct_answer=str(self._extract_numerical_answer(ans) if isinstance(ans, str) else ans),
                                question_type="math",
                                metadata=metadata
                            )
                            questions.append(q)
                        except Exception:
                            continue

                elif 'questions' in raw_data and isinstance(raw_data['questions'], list):
                    for i, item in enumerate(raw_data['questions']):
                        question = self._parse_gsm8k_item(item, f"gsm8k_{i+1:03d}", dataset_type)
                        if question:
                            questions.append(question)
                else:

                    for key, item in raw_data.items():
                        question = self._parse_gsm8k_item(item, key, dataset_type)
                        if question:
                            questions.append(question)

            logger.info(f"成功加载 {len(questions)} 个GSM8K问题")
            return questions

        except Exception as e:
            logger.error(f"加载GSM8K数据失败: {e}")
            return []

    def _parse_gsm8k_item(self, item: Dict[str, Any], identifier: Any, dataset_type: str = 'gsm8k') -> Optional[QuestionData]:

        try:

            question_text = (
                item.get('question') or 
                item.get('problem') or 
                item.get('query') or 
                str(item.get('text', ''))
            )

            correct_answer = (
                item.get('answer') or 
                item.get('solution') or 
                item.get('correct_answer') or
                str(item.get('target', ''))
            )

            if correct_answer and isinstance(correct_answer, str):
                correct_answer = self._extract_numerical_answer(correct_answer)

            question_id = str(identifier)

            per_model_labels: Dict[str, int] = {}
            per_model_confidence: Dict[str, float] = {}

            if 'malicious_answer' in item and 'answer' in item and not any(k.startswith('precomputed_') for k in item.keys()):

                per_model_labels['llama31'] = 1
                per_model_labels['llama3'] = 0
                try:

                    gt_ans = str(correct_answer)
                    mal_ans = str(item.get('malicious_answer', ''))

                    item['precomputed_llama31'] = {
                        'extracted_answer': gt_ans,
                        'is_correct': 1,
                        'question': question_text,
                    }
                    item['precomputed_llama3'] = {
                        'extracted_answer': mal_ans if mal_ans else gt_ans,
                        'is_correct': 0,
                        'question': question_text,
                    }
                except Exception:
                    pass
            else:

                for mk in ('llama3', 'llama31'):
                    precomputed_key = f'precomputed_{mk}'
                    if precomputed_key in item and isinstance(item[precomputed_key], dict):
                        precomputed = item[precomputed_key]
                        if 'is_correct' in precomputed:
                            try:
                                per_model_labels[mk] = int(precomputed['is_correct'])
                            except Exception:
                                pass

            for mk in ('llama3', 'llama31'):
                conf_key = f'{mk}_confidence'
                if conf_key in item:
                    try:
                        per_model_confidence[mk] = float(item[conf_key])
                    except Exception:
                        pass

            metadata = {
                "source": dataset_type,
                "original_data": item
            }
            if per_model_labels:
                metadata["per_model_labels"] = per_model_labels
            if per_model_confidence:
                metadata["per_model_confidence"] = per_model_confidence

            return QuestionData(
                question_id=question_id,
                question_text=question_text,
                correct_answer=str(correct_answer),
                question_type="math" if dataset_type == 'gsm8k' else 'safety',
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"解析GSM8K项目失败 {identifier}: {e}")
            return None

    def _extract_numerical_answer(self, answer_text: str) -> str:

        import re

        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]               

        return answer_text

    def _load_commonsense_questions(self, data_path: str, **kwargs) -> List[QuestionData]:

        data_file = Path(data_path)
        if not data_file.exists():
            logger.error(f"CommonsenseQA数据文件不存在: {data_path}")
            return []

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            questions = []

            if 'questions' in raw_data and isinstance(raw_data['questions'], list):
                for item in raw_data['questions']:
                    question = self._parse_commonsense_item(item)
                    if question:
                        questions.append(question)
            elif isinstance(raw_data, list):

                for item in raw_data:
                    question = self._parse_commonsense_item(item)
                    if question:
                        questions.append(question)

            logger.info(f"成功加载 {len(questions)} 个CommonsenseQA问题")
            return questions

        except Exception as e:
            logger.error(f"加载CommonsenseQA数据失败: {e}")
            return []

    def _parse_commonsense_item(self, item: Dict[str, Any]) -> Optional[QuestionData]:

        try:

            question_id = str(item.get('id') or item.get('question_id', ''))
            question_text = item.get('question', '')
            choices = item.get('choices', [])
            correct_answer = item.get('correct_answer', '')

            per_model_labels = {}
            for mk in ('llama3', 'llama31'):
                if isinstance(item.get(mk), dict):
                    m = item[mk]

                    if 'correct' in m:
                        try:
                            correct_val = m['correct']

                            if isinstance(correct_val, bool):
                                per_model_labels[mk] = 1 if correct_val else 0
                            else:
                                per_model_labels[mk] = int(correct_val)
                        except Exception:
                            pass

            metadata = {
                "source": "commonsense",
                "choices": choices,
                "original_data": item
            }
            if per_model_labels:
                metadata["per_model_labels"] = per_model_labels

            return QuestionData(
                question_id=question_id,
                question_text=question_text,
                correct_answer=correct_answer,
                question_type="commonsense_qa",
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"解析CommonsenseQA项目失败: {e}")
            return None

class StandardQuestionDataLoader(GSM8KDataLoader):

    pass

class PromptProbeDataLoader(BaseDataLoader):

    def __init__(self):
        super().__init__(MethodType.PROMPT_PROBE)

    def load_questions(self, data_path: str, **kwargs) -> List[QuestionData]:

        dataset_type = kwargs.get('dataset_type', 'gsm8k')
        logger.info(f"加载Prompt Probe数据: {data_path}, 类型: {dataset_type}")

        if dataset_type == 'gsm8k':
            return self._load_gsm8k_for_prompt_probe(data_path, **kwargs)
        elif dataset_type == 'safe':
            return self._load_safe_for_prompt_probe(data_path, **kwargs)
        elif dataset_type == 'commonsense':
            return self._load_commonsense_for_prompt_probe(data_path, **kwargs)
        else:
            logger.error(f"不支持的Prompt Probe数据集类型: {dataset_type}")
            return []

    def _load_gsm8k_for_prompt_probe(self, data_path: str, **kwargs) -> List[QuestionData]:

        gsm8k_loader = GSM8KDataLoader()
        questions = gsm8k_loader.load_questions(data_path, **kwargs)

        for question in questions:
            question.question_type = "prompt_probe_gsm8k"
            if question.metadata:
                question.metadata["prompt_probe_dataset"] = "gsm8k"

        return questions

    def _load_safe_for_prompt_probe(self, data_path: str, **kwargs) -> List[QuestionData]:

        safe_loader = SafeDataLoader()
        questions = safe_loader.load_questions(data_path, **kwargs)

        for question in questions:
            question.question_type = "prompt_probe_safe"
            if question.metadata:
                question.metadata["prompt_probe_dataset"] = "safe"

        return questions

    def _load_commonsense_for_prompt_probe(self, data_path: str, **kwargs) -> List[QuestionData]:

        gsm8k_loader = GSM8KDataLoader()

        kwargs_copy = kwargs.copy()
        kwargs_copy['dataset_type'] = 'commonsense'
        questions = gsm8k_loader.load_questions(data_path, **kwargs_copy)

        for question in questions:
            question.question_type = "prompt_probe_commonsense"
            if question.metadata:
                question.metadata["prompt_probe_dataset"] = "commonsense"

        return questions

class SafeDataLoader(BaseDataLoader):

    def __init__(self):

        super().__init__(MethodType.PILOT)

    def load_questions(self, data_path: str, **kwargs) -> List[QuestionData]:

        logger.info(f"加载Safe数据: {data_path}")

        data_file = Path(data_path)
        if not data_file.exists():
            logger.error(f"Safe数据文件不存在: {data_path}")
            return []

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            questions = []

            if isinstance(raw_data, list):
                for i, item in enumerate(raw_data):

                    question = self._parse_safe_item(item, f"safe_{i+1:03d}")
                    if question:
                        questions.append(question)
            elif isinstance(raw_data, dict):

                if 'questions' in raw_data and isinstance(raw_data['questions'], list):
                    for i, item in enumerate(raw_data['questions']):
                        question = self._parse_safe_item(item, f"safe_{i+1:03d}")
                        if question:
                            questions.append(question)
                else:
                    for idx, (key, item) in enumerate(raw_data.items()):

                        if not isinstance(item, dict):
                            continue
                        question = self._parse_safe_item(item, f"safe_{idx+1:03d}")
                        if question:
                            questions.append(question)

            logger.info(f"成功加载 {len(questions)} 个Safe问题")
            return questions

        except Exception as e:
            logger.error(f"加载Safe数据失败: {e}")
            return []

    def _parse_safe_item(self, item: Dict[str, Any], identifier: Any) -> Optional[QuestionData]:

        try:

            question_text = (
                item.get('prompt') or 
                item.get('question') or 
                item.get('text') or
                str(item.get('input', ''))
            )

            correct_answer = (
                item.get('correct_answer') or
                item.get('label') or 
                item.get('answer') or 
                item.get('target') or
                str(item.get('output', ''))
            )

            per_model_labels = {}
            per_model_confidence = {}
            for mk in ('llama3', 'llama31'):
                if isinstance(item.get(mk), dict):
                    m = item[mk]

                    if 'safety_label' in m:
                        try:
                            per_model_labels[mk] = int(m['safety_label'])
                        except Exception:
                            pass

                    if 'precomputed_confidence' in m:
                        try:
                            per_model_confidence[mk] = float(m['precomputed_confidence'])
                        except Exception:
                            pass

            metadata: Dict[str, Any] = {
                "source": "safe",
                "original_data": item
            }
            if per_model_labels:
                metadata["per_model_labels"] = per_model_labels
            if per_model_confidence:
                metadata["per_model_confidence"] = per_model_confidence

            question_id = str(identifier)
            return QuestionData(
                question_id=question_id,
                question_text=question_text,
                correct_answer=str(correct_answer),
                question_type="safety",
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"解析Safe项目失败 {identifier}: {e}")
            return None

class DataLoaderFactory:

    @staticmethod
    def create_data_loader(method_type: MethodType) -> IDataLoader:

        if method_type in (MethodType.PILOT, MethodType.DECODER):

            return StandardQuestionDataLoader()
        elif method_type == MethodType.PROMPT_PROBE:
            return PromptProbeDataLoader()

        else:
            raise ValueError(f"不支持的方法类型: {method_type}")

    @staticmethod
    def get_supported_methods() -> List[MethodType]:

        return [MethodType.PILOT, MethodType.PROMPT_PROBE, MethodType.DECODER]

def load_questions(
    method_type: MethodType, 
    data_path: str, 
    **kwargs
) -> List[QuestionData]:

    loader = DataLoaderFactory.create_data_loader(method_type)
    return loader.load_questions(data_path, **kwargs)

def validate_questions(questions: List[QuestionData]) -> bool:

    if not questions:
        return False

    method_type = MethodType.PILOT      

    loader = DataLoaderFactory.create_data_loader(method_type)
    return loader.validate_data(questions)

def get_data_statistics(questions: List[QuestionData]) -> Dict[str, Any]:

    if not questions:
        return {"error": "No questions provided"}

    method_type = MethodType.PILOT      
    loader = DataLoaderFactory.create_data_loader(method_type)
    return loader.get_data_statistics(questions)

if __name__ == "__main__":

    print("标准化数据加载器测试")

    factory = DataLoaderFactory()
    supported_methods = factory.get_supported_methods()

    print("支持的方法类型:")
    for method in supported_methods:
        print(f"  {method.value}")

    test_questions = [
        QuestionData(
            question_id="test_001",
            question_text="What is 2+2?",
            correct_answer="4",
            question_type="math",
            metadata={"source": "gsm8k"}
        )
    ]

    is_valid = validate_questions(test_questions)
    stats = get_data_statistics(test_questions)

    print(f"数据有效性: {is_valid}")
    print(f"数据统计: {stats}")

    print("数据加载器测试完成")
