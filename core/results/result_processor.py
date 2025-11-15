#!/usr/bin/env python3

import json
import copy
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from ..interfaces import IResultProcessor, ExperimentResult, MethodType

logger = logging.getLogger(__name__)

class StandardizedResultProcessor(IResultProcessor):

    def __init__(self, output_base_dir: str = None):
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path("results")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化结果处理器: {self.output_base_dir}")

    def save_experiment_result(self, result: ExperimentResult, output_path: str) -> None:

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:

            serializable_result = self._make_serializable(result)

            cfg = (serializable_result.get("metadata") or {}).get("config") or {}
            strip_legacy = bool(cfg.get("strip_legacy_original_data", True))
            if strip_legacy:
                try:
                    for q in serializable_result.get("questions", []) or []:
                        md = q.get("metadata") or {}
                        if "original_data" in md:

                            od = md.get("original_data") or {}

                            brief = {
                                "question": od.get("question"),
                                "correct_answer": od.get("correct_answer"),
                                "strong_model": cfg.get("strong_model", od.get("strong_model")),
                                "weak_model": cfg.get("weak_model", od.get("weak_model")),
                            }
                            md["provenance_brief"] = brief
                            md.pop("original_data", None)
                            q["metadata"] = md
                except Exception:
                    pass

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"实验结果已保存: {output_file}")

            summary_file = output_file.with_suffix('.summary.json')
            summary = self._create_result_summary(result)
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"实验摘要已保存: {summary_file}")

            try:
                self._maybe_export_provenance_and_rounds(result, serializable_result, output_file)
            except Exception as e:
                logger.warning(f"扩展结果导出失败: {e}")

        except Exception as e:
            logger.error(f"保存实验结果失败: {e}")
            raise

    def load_experiment_result(self, input_path: str) -> ExperimentResult:

        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"结果文件不存在: {input_path}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = self._reconstruct_experiment_result(data)
            logger.info(f"实验结果已加载: {input_file}")
            return result

        except Exception as e:
            logger.error(f"加载实验结果失败: {e}")
            raise

    def export_to_format(self, result: ExperimentResult, format_type: str, output_path: str) -> None:

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format_type.lower() == 'json':
                self.save_experiment_result(result, output_path)
            elif format_type.lower() == 'csv':
                self._export_to_csv(result, output_file)
            elif format_type.lower() == 'xlsx':
                self._export_to_excel(result, output_file)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

            logger.info(f"结果已导出为 {format_type}: {output_file}")

        except Exception as e:
            logger.error(f"导出结果失败: {e}")
            raise

    def _make_serializable(self, result: ExperimentResult) -> Dict[str, Any]:

        metrics = copy.deepcopy(result.evaluation_metrics) if result.evaluation_metrics else {}
        try:
            bft = metrics.get('byzantine_fault_tolerance', {})
            if isinstance(bft, dict) and 'consensus_maintained' in bft:
                val = bft.get('consensus_maintained')

                if isinstance(val, str):
                    lowered = val.strip().lower()
                    if lowered in ('true', '1', 'yes'):
                        val_bool = True
                    elif lowered in ('false', '0', 'no'):
                        val_bool = False
                    else:
                        val_bool = bool(val)
                else:

                    try:
                        val_bool = bool(val)
                    except Exception:
                        val_bool = True if val in (1, '1') else False
                bft['consensus_maintained'] = val_bool
                metrics['byzantine_fault_tolerance'] = bft
        except Exception:
            pass

        metadata = copy.deepcopy(result.metadata) if result.metadata else {}
        try:
            cfg = metadata.get('config')
            if isinstance(cfg, dict):

                for k in list(cfg.keys()):
                    kl = str(k).lower()
                    if kl == 'api_key' or kl.endswith('_api_key') or 'api_key' in kl:
                        cfg.pop(k, None)

                if 'validate' in cfg:
                    cfg.pop('validate', None)

                for k, v in list(cfg.items()):
                    if isinstance(v, str) and 'api_key=' in v:

                        cfg[k] = v.replace('api_key=', 'api_key=***')
                metadata['config'] = cfg
        except Exception:
            pass

        def _enum_val(x):
            try:
                return x.value
            except Exception:
                return str(x)

        return {
            "experiment_id": result.experiment_id,
            "method_type": _enum_val(result.method_type),
            "topology_type": _enum_val(result.topology_type),
            "agent_count": result.agent_count,
            "malicious_count": result.malicious_count,
            "execution_time": result.execution_time,
            "timestamp": datetime.now().isoformat(),
            "questions": [self._serialize_question(q) for q in result.questions],
            "consensus_results": [self._serialize_consensus_result(cr) for cr in result.consensus_results],
            "evaluation_metrics": metrics,
            "metadata": metadata or {}
        }

    def _maybe_export_provenance_and_rounds(self, result: ExperimentResult, serializable: Dict[str, Any], output_file: Path) -> None:
        cfg = (serializable.get("metadata") or {}).get("config") or {}
        save_provenance = bool(cfg.get("save_provenance", False))
        save_raw = bool(cfg.get("save_raw_llm_output", False))
        save_rounds_min = bool(cfg.get("save_rounds_min", False))
        raw_max = int(cfg.get("raw_max_chars", 1024) or 1024)

        if not (save_provenance or save_raw or save_rounds_min):
            return

        if save_provenance:
            provenance = self._build_provenance(serializable)
            with open(output_file.with_suffix('.provenance.json'), 'w', encoding='utf-8') as pf:
                json.dump(provenance, pf, indent=2, ensure_ascii=False)

        if save_rounds_min:
            rounds_min = self._build_rounds_min_view(serializable)
            with open(output_file.with_name(f"{output_file.stem}.run_min.json"), 'w', encoding='utf-8') as rf:
                json.dump(rounds_min, rf, indent=2, ensure_ascii=False)

        if save_raw:
            self._export_raw_ndjson(serializable, output_file.with_name(f"{output_file.stem}.run_raw.ndjson"), raw_max)

    def _build_provenance(self, serializable: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        for q in serializable.get("questions", []) or []:
            md = q.get("metadata") or {}
            od = md.get("provenance_brief") or (md.get("original_data") or {})
            items.append({
                "question_id": q.get("question_id"),
                "question": q.get("question_text"),
                "correct_answer": q.get("correct_answer"),
                "strong_model": od.get("strong_model"),
                "weak_model": od.get("weak_model")
            })
        return {"provenance": items}

    def _map_safe_label(self, v: Any) -> str:
        s = str(v).strip().lower()
        if s in ("1", "safe"): return "safe"
        if s in ("0", "unsafe"): return "unsafe"
        return str(v)

    def _build_rounds_min_view(self, serializable: Dict[str, Any]) -> Dict[str, Any]:
        cfg = (serializable.get("metadata") or {}).get("config") or {}
        dataset = str(cfg.get('dataset_type', '')).lower()
        malicious_agents = (serializable.get("metadata") or {}).get("malicious_agents") or []
        strong_model = cfg.get('strong_model')
        weak_model = cfg.get('weak_model')

        qmap = {q.get('question_id'): q for q in serializable.get('questions', [])}

        rounds = []
        for cr in serializable.get('consensus_results', []) or []:
            qid = cr.get('question_id')
            q = qmap.get(qid) or {}
            prompt = (q.get('question_text') or '')
            brief = prompt[:120] + ('...' if len(prompt) > 120 else '')
            correct_label = q.get('correct_answer')
            if dataset == 'safe':
                correct_label = self._map_safe_label(correct_label)

            agents = []
            for ir in cr.get('individual_responses', []) or []:
                aid = ir.get('agent_id')
                role = 'malicious' if aid in malicious_agents else 'normal'
                model_used = weak_model if role == 'malicious' else strong_model
                meta = ir.get('metadata') or {}
                init = meta.get('initial_answer', ir.get('answer'))
                fin = meta.get('final_answer', ir.get('answer'))
                if dataset == 'safe':
                    init = self._map_safe_label(init)
                    fin = self._map_safe_label(fin)
                agents.append({
                    'agent_id': aid,
                    'role': role,
                    'model_used': model_used,
                    'initial': {'label': init, 'changed': init != fin},
                    'final': {'label': fin, 'changed': init != fin}
                })

            dist = (cr.get('metadata') or {}).get('answer_distribution') or {}
            if dataset == 'safe':
                dist = {self._map_safe_label(k): v for k, v in dist.items()}
            cons_label = cr.get('consensus_answer')
            if dataset == 'safe':
                cons_label = self._map_safe_label(cons_label)
            rounds.append({
                'question_id': qid,
                'prompt_brief': brief,
                'correct_label': correct_label,
                'agents': agents,
                'consensus': {
                    'label': cons_label,
                    'distribution': dist,
                    'is_correct': (cons_label == correct_label)
                }
            })

        files = {}
        try:
            base = output_file = None                                          
        except Exception:
            pass
        return {
            'config': cfg,
            'malicious_agents': malicious_agents,
            'rounds': rounds,
            'summaries': {
                'consensus_accuracy': (serializable.get('evaluation_metrics') or {}).get('consensus_accuracy', {})
            }
        }

    def _export_raw_ndjson(self, serializable: Dict[str, Any], raw_path: Path, raw_max: int) -> None:
        try:
            import io
            buf = io.StringIO()
            for cr in serializable.get('consensus_results', []) or []:
                qid = cr.get('question_id')
                for ir in cr.get('individual_responses', []) or []:
                    aid = ir.get('agent_id')
                    meta = ir.get('metadata') or {}
                    raw_text = meta.get('raw_response') or ''
                    if raw_text:
                        trunc = raw_text[:raw_max]
                    else:
                        trunc = ''
                    rec = {
                        'question_id': qid,
                        'agent_id': aid,
                        'phase': 'final',
                        'model_used': None,
                        'raw_text_trunc': trunc
                    }
                    buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(raw_path, 'w', encoding='utf-8') as rf:
                rf.write(buf.getvalue())
        except Exception:

            pass

    def _serialize_question(self, question) -> Dict[str, Any]:

        meta = question.metadata or {}
        cfg = (getattr(self, 'current_config', None) or {})
        try:

            dataset_type = ''
            if isinstance(cfg, dict):
                dataset_type = str(cfg.get('dataset_type', '')).lower()
            if not dataset_type:
                dataset_type = str(((self.last_serializable or {}).get('metadata') or {}).get('config', {}) .get('dataset_type', '')).lower()  # type: ignore
        except Exception:
            dataset_type = ''

        if dataset_type == 'safe':

            m = dict(meta)
            mp = (m.get('provenance_brief') or {})
            if isinstance(mp, dict):
                mp['correct_answer'] = 'safe'
                m['provenance_brief'] = mp
            m['source'] = 'safe'

            qtype = 'prompt_probe_safe' if str(getattr(question, 'question_type', '')).startswith('prompt_probe') else 'safe'
            qid = question.question_id

            try:
                if isinstance(qid, str) and qid.lower().startswith('gsm8k_'):
                    qid = 'safe_' + qid.split('_', 1)[1]
            except Exception:
                pass
            return {
                "question_id": qid,
                "question_text": question.question_text,
                "correct_answer": 'safe' if str(question.correct_answer).strip().lower() in ('1', 'safe') else str(question.correct_answer),
                "question_type": qtype,
                "metadata": m
            }

        return {
            "question_id": question.question_id,
            "question_text": question.question_text,
            "correct_answer": question.correct_answer,
            "question_type": question.question_type,
            "metadata": meta
        }

    def _serialize_consensus_result(self, consensus_result) -> Dict[str, Any]:

        return {
            "question_id": consensus_result.question_id,
            "consensus_answer": consensus_result.consensus_answer,
            "consensus_confidence": consensus_result.consensus_confidence,
            "participant_count": consensus_result.participant_count,
            "round_number": consensus_result.round_number,
            "convergence_achieved": consensus_result.convergence_achieved,
            "individual_responses": [self._serialize_agent_response(ar) for ar in consensus_result.individual_responses],
            "metadata": consensus_result.metadata or {},
            "is_correct": consensus_result.is_correct                  
        }

    def _serialize_agent_response(self, response) -> Dict[str, Any]:

        return {
            "agent_id": response.agent_id,
            "question_id": response.question_id,
            "answer": response.answer,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "response_time": response.response_time,
            "metadata": response.metadata or {}
        }

    def _reconstruct_experiment_result(self, data: Dict[str, Any]) -> ExperimentResult:

        from ..interfaces import QuestionData, ConsensusResult, AgentResponse

        questions = []
        for q_data in data.get("questions", []):
            question = QuestionData(
                question_id=q_data["question_id"],
                question_text=q_data["question_text"],
                correct_answer=q_data["correct_answer"],
                question_type=q_data["question_type"],
                metadata=q_data.get("metadata")
            )
            questions.append(question)

        consensus_results = []
        for cr_data in data.get("consensus_results", []):

            individual_responses = []
            for ar_data in cr_data.get("individual_responses", []):
                response = AgentResponse(
                    agent_id=ar_data["agent_id"],
                    question_id=ar_data["question_id"],
                    answer=ar_data["answer"],
                    confidence=ar_data["confidence"],
                    reasoning=ar_data.get("reasoning"),
                    response_time=ar_data.get("response_time", 0.0),
                    metadata=ar_data.get("metadata")
                )
                individual_responses.append(response)

            consensus_result = ConsensusResult(
                question_id=cr_data["question_id"],
                consensus_answer=cr_data["consensus_answer"],
                consensus_confidence=cr_data["consensus_confidence"],
                participant_count=cr_data["participant_count"],
                round_number=cr_data["round_number"],
                convergence_achieved=cr_data["convergence_achieved"],
                individual_responses=individual_responses,
                metadata=cr_data.get("metadata"),
                is_correct=cr_data.get("is_correct")                       
            )
            consensus_results.append(consensus_result)

        try:
            from ..interfaces import TopologyType as _Topo
            _topo = data.get("topology_type")
            topo_enum = _Topo(_topo) if not hasattr(_topo, 'value') else _topo
        except Exception:
            topo_enum = data.get("topology_type")

        result = ExperimentResult(
            experiment_id=data["experiment_id"],
            method_type=MethodType(data["method_type"]),
            topology_type=topo_enum,
            agent_count=data["agent_count"],
            malicious_count=data["malicious_count"],
            questions=questions,
            consensus_results=consensus_results,
            evaluation_metrics=data.get("evaluation_metrics", {}),
            execution_time=data.get("execution_time", 0.0),
            metadata=data.get("metadata")
        )

        return result

    def _create_result_summary(self, result: ExperimentResult) -> Dict[str, Any]:

        total_questions = len(result.questions)
        total_consensus = len(result.consensus_results)

        def _norm_safe_label(s: str) -> str:
            t = str(s).strip().lower()
            if t in ("1", "safe"): return "safe"
            if t in ("0", "unsafe"): return "unsafe"
            return t

        correct_consensus = 0
        if result.questions and result.consensus_results:

            dataset_type = str(((result.metadata or {}).get('config') or {}).get('dataset_type', '')).lower()
            is_safe = (dataset_type == 'safe')
            question_answers = {q.question_id: ( _norm_safe_label(q.correct_answer) if is_safe else str(q.correct_answer).strip() ) for q in result.questions}
            for cr in result.consensus_results:
                if cr.question_id in question_answers:
                    consensus_ans = _norm_safe_label(cr.consensus_answer) if is_safe else str(cr.consensus_answer).strip()
                    if consensus_ans == question_answers[cr.question_id]:
                        correct_consensus += 1

        consensus_accuracy = correct_consensus / total_consensus if total_consensus > 0 else 0.0

        is_traditional = False
        if result.metadata and 'config' in result.metadata:
            config = result.metadata['config']
            is_traditional = config.get('agent_type') == 'traditional'

        avg_confidence = 0.0
        if not is_traditional and result.consensus_results:
            confidences = [cr.consensus_confidence for cr in result.consensus_results]
            avg_confidence = sum(confidences) / len(confidences)

        convergence_rate = 0.0
        if result.consensus_results:
            converged = sum(1 for cr in result.consensus_results if cr.convergence_achieved)
            convergence_rate = converged / len(result.consensus_results)

        summary = {
            "experiment_id": result.experiment_id,
            "method_type": result.method_type.value,
            "topology_type": result.topology_type.value if hasattr(result.topology_type, 'value') else str(result.topology_type),
            "agent_count": result.agent_count,
            "malicious_count": result.malicious_count,
            "malicious_ratio": result.malicious_count / result.agent_count if result.agent_count > 0 else 0.0,
            "execution_time": result.execution_time,
            "total_questions": total_questions,
            "total_consensus": total_consensus,
            "consensus_accuracy": consensus_accuracy,
            "convergence_rate": convergence_rate
        }

        if not is_traditional:
            summary["average_confidence"] = avg_confidence

        return summary

    def _export_to_csv(self, result: ExperimentResult, output_file: Path) -> None:

        rows = []
        question_answers = {q.question_id: q.correct_answer for q in result.questions}

        cfg = (result.metadata or {}).get('config') or {}
        is_safe_dataset = str(cfg.get('dataset_type', '')).lower() == 'safe'

        for cr in result.consensus_results:
            correct_answer = question_answers.get(cr.question_id, "")
            if is_safe_dataset:
                is_correct = self._map_safe_label(cr.consensus_answer) == self._map_safe_label(correct_answer)
            else:
                is_correct = str(cr.consensus_answer).strip() == str(correct_answer).strip()

            row = {
                "question_id": cr.question_id,
                "consensus_answer": cr.consensus_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "consensus_confidence": cr.consensus_confidence,
                "participant_count": cr.participant_count,
                "round_number": cr.round_number,
                "convergence_achieved": cr.convergence_achieved
            }
            rows.append(row)

        main_file = output_file.with_suffix('.csv')
        with open(main_file, 'w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        detail_rows = []
        for cr in result.consensus_results:
            for response in cr.individual_responses:
                detail_row = {
                    "question_id": response.question_id,
                    "agent_id": response.agent_id,
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "response_time": response.response_time,
                    "reasoning": response.reasoning or ""
                }
                detail_rows.append(detail_row)

        detail_file = output_file.with_name(f"{output_file.stem}_details.csv")
        with open(detail_file, 'w', newline='', encoding='utf-8') as f:
            if detail_rows:
                writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
                writer.writeheader()
                writer.writerows(detail_rows)

    def _export_to_excel(self, result: ExperimentResult, output_file: Path) -> None:

        try:
            with pd.ExcelWriter(output_file.with_suffix('.xlsx'), engine='openpyxl') as writer:

                summary = self._create_result_summary(result)
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                consensus_data = []
                question_answers = {q.question_id: q.correct_answer for q in result.questions}

                cfg = (result.metadata or {}).get('config') or {}
                is_safe_dataset = str(cfg.get('dataset_type', '')).lower() == 'safe'

                for cr in result.consensus_results:
                    correct_answer = question_answers.get(cr.question_id, "")
                    if is_safe_dataset:
                        _is_correct = self._map_safe_label(cr.consensus_answer) == self._map_safe_label(correct_answer)
                    else:
                        _is_correct = str(cr.consensus_answer).strip() == str(correct_answer).strip()
                    consensus_data.append({
                        "question_id": cr.question_id,
                        "consensus_answer": cr.consensus_answer,
                        "correct_answer": correct_answer,
                        "is_correct": _is_correct,
                        "consensus_confidence": cr.consensus_confidence,
                        "participant_count": cr.participant_count,
                        "round_number": cr.round_number,
                        "convergence_achieved": cr.convergence_achieved
                    })

                if consensus_data:
                    consensus_df = pd.DataFrame(consensus_data)
                    consensus_df.to_excel(writer, sheet_name='Consensus Results', index=False)

                response_data = []
                for cr in result.consensus_results:
                    for response in cr.individual_responses:
                        response_data.append({
                            "question_id": response.question_id,
                            "agent_id": response.agent_id,
                            "answer": response.answer,
                            "confidence": response.confidence,
                            "response_time": response.response_time,
                            "reasoning": response.reasoning or ""
                        })

                if response_data:
                    response_df = pd.DataFrame(response_data)
                    response_df.to_excel(writer, sheet_name='Agent Responses', index=False)

                if result.evaluation_metrics:
                    metrics_data = []
                    for key, value in result.evaluation_metrics.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                metrics_data.append({
                                    "category": key,
                                    "metric": sub_key,
                                    "value": sub_value
                                })
                        else:
                            metrics_data.append({
                                "category": "general",
                                "metric": key,
                                "value": value
                            })

                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)

        except ImportError:
            logger.warning("pandas或openpyxl未安装，无法导出Excel格式")

            self._export_to_csv(result, output_file)

def create_result_processor(output_base_dir: str = None) -> StandardizedResultProcessor:

    return StandardizedResultProcessor(output_base_dir)

